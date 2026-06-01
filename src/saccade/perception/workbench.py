import torch
from dataclasses import dataclass
from typing import Any, Optional

from saccade_tracking_ext import (
    Workbench as _WorkbenchExt,
    PerceptionPipeline,
    PerceptionPipelineConfig,
    quality_scale_scores as _cpp_quality_scale,
    narrow_bonus_scores as _cpp_narrow_bonus,
)
from saccade.perception.tracking import GPUByteTracker


def _apply_fp_hard_filter(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    ids: torch.Tensor,
    det_idx: torch.Tensor,
    *,
    min_score: float = 0.25,
    max_suspicious_area: int = 10000,
    max_suspicious_score: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove suspicious low-score large-area detections.

    Filters ALL five output tensors with the same keep mask so that
    box/cls/score/ids/det_idx stay perfectly aligned.
    """
    if boxes.numel() == 0:
        return boxes[:0], scores[:0], classes[:0], ids[:0], det_idx[:0]
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    area = bw * bh
    suspicious = (scores < max_suspicious_score) & (area > max_suspicious_area)
    very_low = scores < min_score
    keep = ~(suspicious | very_low)
    if not keep.any():
        return boxes[:0], scores[:0], classes[:0], ids[:0], det_idx[:0]
    return boxes[keep], scores[keep], classes[keep], ids[keep], det_idx[keep]


def _ptr_or_zero_extractor(extractor: Any) -> int:
    """Get C++ pointer from extractor (returns 0 if not available)."""
    try:
        return int(extractor.cpp_ptr) if hasattr(extractor, "cpp_ptr") else 0
    except Exception:
        return 0


def _ptr_or_zero_cropper(cropper: Any) -> int:
    """Get C++ pointer from cropper (returns 0 if not available)."""
    try:
        return int(cropper.cpp_ptr) if hasattr(cropper, "cpp_ptr") else 0
    except Exception:
        return 0


@dataclass
class WorkbenchResult:
    ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    det_idx: torch.Tensor


def _ptr_or_zero(tensor: Optional[torch.Tensor]) -> int:
    return tensor.contiguous().data_ptr() if tensor is not None else 0


def _ptrs_or_zero_priors(
    priors: Optional[torch.Tensor], prior_classes: Optional[torch.Tensor]
) -> tuple[int, int, int]:
    if priors is not None and prior_classes is not None:
        return (
            priors.contiguous().data_ptr(),
            prior_classes.contiguous().data_ptr(),
            int(priors.shape[0]),
        )
    return 0, 0, 0


class Workbench:
    """Quality-aware workbench: wraps C++ filter+NMS+tracker with Python pre/post-processing.

    When extractor/cropper are provided, ReID embeddings are computed for budgeted
    candidates and injected into the tracker's data-association cost matrix.
    When gmc_estimator is provided, GMC warps are computed and passed to the tracker.
    Quality scaling, detection cap, and FP filter are applied as Python pre/post-processing.
    """

    def __init__(
        self,
        proxy: Any,
        pipeline_cfg: PerceptionPipelineConfig,
        device: str = "cuda:0",
        max_dets: int = 2048,
        max_tracks: int = 256,
        # ── Optional quality/ReID components ──────────────────────────
        extractor: Any = None,  # TRTFeatureExtractor (for ReID)
        cropper: Any = None,  # ZeroCopyCropper (for ReID crop)
        gmc_estimator: Any = None,  # SparseOpticalFlowGMC or C++ GMC
        scene_adapt_policy: Any = None,  # SceneAdaptivePolicy (optional)
        narrow_bonus: float = 0.0,  # narrow_person_score_bonus (0.0 if scene_adapt)
        # ── Quality filter params ────────────────────────────────────
        quality_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        max_detections: int = 30,
        fp_hard_filter: bool = True,
        fp_min_score: float = 0.25,
        fp_max_suspicious_area: int = 10000,
        fp_max_suspicious_score: float = 0.45,
        # ── ReID params ──────────────────────────────────────────────
        reid_budget_raw: float = 0.0,
        reid_interval: int = 20,
        need_reid: bool = False,
        dynamic_reid: Any = None,
        gmc_uncertain: bool = False,
        # ── ReID frame tracking ──────────────────────────────────────
        last_reid_frame: int = -100,
        # ── Output buffer capacity ───────────────────────────────────
        # Must match C++ tracker max_objs_ = max_tracks * 8 to avoid
        # buffer overflow in update_into (which unconditionally copies
        # max_objs_ elements).  Default of 256 was a dangerous mismatch
        # when max_tracks=256 → tracker writes 2048 to a 256-buffer.
        output_capacity: int | None = None,
    ):
        self.device = device
        self.max_dets = max_dets
        self.max_tracks = max_tracks

        # ── Output buffer capacity (align with C++ max_objs = max_tracks * 8) ──
        if output_capacity is None:
            output_capacity = max_tracks * 8

        # ── Optional components ──────────────────────────────────────
        self.extractor = extractor
        self.cropper = cropper
        self.gmc_estimator = gmc_estimator
        self.scene_adapt_policy = scene_adapt_policy
        self.narrow_bonus = narrow_bonus
        self.reid_budget_raw = reid_budget_raw
        self.reid_interval = reid_interval
        self.need_reid = need_reid
        self.dynamic_reid = dynamic_reid
        self.gmc_uncertain = gmc_uncertain

        # ── ReID frame tracking ──────────────────────────────────────
        self.last_reid_frame = last_reid_frame

        # ── Quality filter params ────────────────────────────────────
        self.quality_weights = quality_weights
        self.max_detections = max_detections
        self.fp_hard_filter = fp_hard_filter
        self.fp_min_score = fp_min_score
        self.fp_max_suspicious_area = fp_max_suspicious_area
        self.fp_max_suspicious_score = fp_max_suspicious_score

        # ── C++ pipeline ─────────────────────────────────────────────
        # If extractor+cropper are provided, pass them to PerceptionPipeline
        # so that native_reid_available=True and extract_reid() works.
        extractor_ptr = _ptr_or_zero_extractor(extractor) if extractor else 0
        cropper_ptr = _ptr_or_zero_cropper(cropper) if cropper else 0
        self.pipeline = PerceptionPipeline(extractor_ptr, cropper_ptr, pipeline_cfg)
        self.has_reid_support = extractor is not None and cropper is not None

        # ── Per-workbench tracker ────────────────────────────────────
        self.tracker = GPUByteTracker(max_objects=max_tracks * 8)
        self.stream = torch.cuda.Stream(device=device)

        self.proxy = proxy
        if hasattr(self.proxy, "tracker"):
            self.proxy.tracker = self.tracker

        # Output staging buffers
        self.out_boxes = torch.empty(
            (output_capacity, 4), dtype=torch.float32, device=device
        )
        self.out_scores = torch.empty(
            (output_capacity,), dtype=torch.float32, device=device
        )
        self.out_ids = torch.empty((output_capacity,), dtype=torch.int32, device=device)
        self.out_classes = torch.empty(
            (output_capacity,), dtype=torch.int32, device=device
        )
        self.out_det_idx = torch.empty(
            (output_capacity,), dtype=torch.int32, device=device
        )
        self.out_count = torch.zeros((1,), dtype=torch.int32, device=device)

        self._wb = _WorkbenchExt(
            self.pipeline.cpp_ptr,
            self.tracker.tracker.cpp_ptr,  # Use the internal C++ tracker's pointer
            int(self.stream.cuda_stream),
            max_dets,
            max_tracks,
        )

    def process_frame(
        self,
        frame_chw: torch.Tensor,
        *,
        frame_w: int,
        frame_h: int,
        embeddings: Optional[torch.Tensor] = None,
        gmc: Optional[torch.Tensor] = None,
        priors: Optional[torch.Tensor] = None,
        prior_classes: Optional[torch.Tensor] = None,
        pre_postyolo_hook: Optional[Any] = None,
    ) -> WorkbenchResult:
        with torch.cuda.stream(self.stream):
            # 1. Batch submission & inference
            raw = self.proxy.detect_raw(frame_chw)

            priors_ptr, prior_classes_ptr, num_priors = _ptrs_or_zero_priors(
                priors, prior_classes
            )

            raw_boxes = raw[0, :, :4].contiguous()
            raw_scores = raw[0, :, 4].contiguous()
            raw_classes = raw[0, :, 5].to(torch.int32).contiguous()

            if pre_postyolo_hook is not None:
                raw_scores = pre_postyolo_hook(raw_boxes, raw_scores, raw_classes)

            # 2. GIL-free post-YOLO hot path (filter+NMS+update_into)
            n = self._wb.process_frame_postyolo(
                raw_boxes.data_ptr(),
                raw_scores.data_ptr(),
                raw_classes.data_ptr(),
                int(raw.shape[1]),
                frame_w,
                frame_h,
                False,  # is_tiled
                priors_ptr,
                prior_classes_ptr,
                num_priors,
                0.5,  # prior_iou_threshold
                _ptr_or_zero(embeddings),
                _ptr_or_zero(gmc),
                0.0,  # light_factor
                1.0,  # mid_thresh_scale
                self.out_boxes.data_ptr(),
                self.out_scores.data_ptr(),
                self.out_ids.data_ptr(),
                self.out_classes.data_ptr(),
                self.out_det_idx.data_ptr(),
                self.out_count.data_ptr(),
            )
            wb_done = torch.cuda.Event()
            wb_done.record(self.stream)

        torch.cuda.current_stream().wait_event(wb_done)
        # Output slice views
        return WorkbenchResult(
            ids=self.out_ids[:n],
            boxes=self.out_boxes[:n],
            scores=self.out_scores[:n],
            classes=self.out_classes[:n],
            det_idx=self.out_det_idx[:n],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Quality-aware detection processing (baseline-aligned)
    # ──────────────────────────────────────────────────────────────────────

    def _apply_narrow_bonus(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        frame_w: int,
        frame_h: int,
        person_class: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply narrow-person score bonus (scene-adapt P5-4)."""
        if self.narrow_bonus <= 0.0:
            return boxes, scores, classes
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        aspect = bh / bw.clamp(min=1e-6)
        is_person = classes == person_class
        is_narrow = (aspect < 2.1) & (bh / frame_h < 0.5)
        mask = is_person & is_narrow
        if mask.any():
            scores = scores.clone()
            scores[mask] += self.narrow_bonus
        return boxes, scores, classes

    def _compute_reid_embeddings(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        frame_chw: torch.Tensor,
        frame_w: int,
        frame_h: int,
        *,
        frame_id: int,
        last_reid_frame: int,
    ) -> tuple[torch.Tensor | None, int]:
        """Compute ReID embeddings for budgeted detections.

        Returns (embeddings_tensor, next_reid_frame).
        embeddings_tensor: (N, dim) with zero-vectors for non-budgeted detections.
        """
        if not self.has_reid_support or boxes.numel() == 0:
            return None, last_reid_frame

        # ── ReID interval gate ────────────────────────────────────
        time_since_last = frame_id - last_reid_frame
        if time_since_last < 2:
            return None, last_reid_frame

        # ── Need-reid gate ────────────────────────────────────────
        do_reid = False
        if self.need_reid:
            if self.dynamic_reid is not None:
                do_reid = self.dynamic_reid.should_reid(boxes.shape[0])
            else:
                do_reid = time_since_last >= max(2, round(self.reid_interval / 2))
        else:
            do_reid = time_since_last >= self.reid_interval

        if not do_reid:
            return None, last_reid_frame

        # ── ReID budget ───────────────────────────────────────────
        num_dets = boxes.shape[0]
        if self.reid_budget_raw >= 1.0:
            actual_budget = int(self.reid_budget_raw)
        elif self.reid_budget_raw > 0.0:
            actual_budget = max(1, int(self.reid_budget_raw * num_dets))
        else:
            actual_budget = num_dets  # Unlimited

        # Budget selection: top-k by score (aligned with helpers.py budget_reid_candidates)
        if actual_budget < num_dets:
            _, top_idx = torch.topk(scores, actual_budget)
        else:
            top_idx = torch.arange(num_dets, device=boxes.device)
        budget_indices = top_idx.sort().values

        # ── Extract embeddings ────────────────────────────────────
        embeddings = torch.zeros(
            (num_dets, self.extractor.feature_dim),
            device=boxes.device,
            dtype=torch.float32,
        )
        if budget_indices.numel() > 0:
            budgeted_boxes = boxes[budget_indices].contiguous()
            if self.has_reid_support:
                # Use native PerceptionPipeline.extract_reid (zero-copy)
                frame_hwc = frame_chw.permute(1, 2, 0).contiguous()
                budget_embeddings = torch.empty(
                    (budget_indices.numel(), self.extractor.feature_dim),
                    device=boxes.device,
                    dtype=torch.float32,
                )
                self.pipeline.extract_reid(
                    frame_hwc.data_ptr(),
                    frame_h,
                    frame_w,
                    budgeted_boxes.data_ptr(),
                    int(budgeted_boxes.shape[0]),
                    budget_embeddings.data_ptr(),
                    self.stream.cuda_stream,
                )
                embeddings[budget_indices] = budget_embeddings
            else:
                # Fallback: cropper + extractor
                crops = self.cropper.process(frame_chw.unsqueeze(0), budgeted_boxes)
                if crops.numel() > 0:
                    budget_embeddings = self.extractor.extract(crops)
                    embeddings[budget_indices] = budget_embeddings

        return embeddings, frame_id

    def _compute_gmc_warp(
        self,
        prev_gray: torch.Tensor | None,
        curr_gray: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool]:
        """Compute GMC warp between previous and current frame.

        Returns (gmc_warp_tensor | None, gmc_uncertain).
        gmc_warp: 6-float tensor (homography coefficients) on GPU.
        """
        if self.gmc_estimator is None or prev_gray is None:
            return None, False

        try:
            if hasattr(self.gmc_estimator, "estimate"):
                result = self.gmc_estimator.estimate(curr_gray)
                if result is None:
                    return None, False
                if isinstance(result, list):
                    gmc_warp = torch.tensor(
                        result[:6],
                        dtype=torch.float32,
                        device=curr_gray.device,
                    )
                else:
                    gmc_warp = result.to(curr_gray.device)[:6]
            else:
                return None, False

            # Check uncertainty
            uncertain = False
            if hasattr(self.gmc_estimator, "pcr_score"):
                pcr = self.gmc_estimator.pcr_score()
                uncertain = 0.0 < pcr < 0.7  # threshold from config

            return gmc_warp, uncertain
        except Exception:
            return None, False

    def process_detections_quality_aware(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        *,
        frame_w: int,
        frame_h: int,
        frame_chw: Optional[torch.Tensor] = None,
        frame_id: int = 0,
        last_reid_frame: int = -100,
        prev_gray: Optional[torch.Tensor] = None,
        priors: Optional[torch.Tensor] = None,
        prior_classes: Optional[torch.Tensor] = None,
        person_class: int = 0,
        is_tiled: bool = False,
    ) -> WorkbenchResult:
        """Quality-aware detection → tracking with baseline-aligned preprocessing.

        Pipeline (all GPU ops GIL-free on self.stream):
          1. Narrow bonus kernel (C++ GPU, in-place on scores)
          2. Quality scaling kernel (C++ GPU, in-place on scores)
          3. ReID embeddings for budgeted candidates (optional)
          4. GMC warp (optional, CPU-side)
          5. C++ filter+NMS+tracker (process_frame_postyolo)
          6. FP hard filter on output (Python, post-tracker)
        """
        if boxes.numel() == 0:
            return WorkbenchResult(
                ids=self.out_ids[:0],
                boxes=self.out_boxes[:0],
                scores=self.out_scores[:0],
                classes=self.out_classes[:0],
                det_idx=self.out_det_idx[:0],
            )

        n = boxes.shape[0]
        raw_boxes = boxes.to(torch.float32).contiguous()
        # Clone so quality ops can modify scores in-place without aliasing input
        raw_scores = scores.to(torch.float32).contiguous().clone()
        raw_classes = classes.to(torch.int32).contiguous()

        # GMC: CPU-side optical flow (sync point, compute before entering stream context)
        gmc_warp_tensor = None
        if prev_gray is not None and frame_chw is not None:
            gray = (
                0.299 * frame_chw[2] + 0.587 * frame_chw[1] + 0.114 * frame_chw[0]
            ).unsqueeze(0)
            gmc_warp_tensor, gmc_uncertain = self._compute_gmc_warp(prev_gray, gray)
            if gmc_uncertain:
                self.gmc_uncertain = True

        priors_ptr, prior_classes_ptr, num_priors = _ptrs_or_zero_priors(
            priors, prior_classes
        )
        caller_event = torch.cuda.Event()
        caller_event.record()

        with torch.cuda.stream(self.stream):
            self.stream.wait_event(caller_event)
            stream_ptr = self.stream.cuda_stream

            # Step 1: narrow bonus (C++ GPU kernel, in-place, GIL-free)
            if self.narrow_bonus > 0.0:
                _cpp_narrow_bonus(
                    raw_scores.data_ptr(),
                    raw_boxes.data_ptr(),
                    raw_classes.data_ptr(),
                    n,
                    self.narrow_bonus,
                    person_class,
                    2.1,
                    0.5,
                    frame_h,
                    stream_ptr,
                )

            # Step 2: quality scaling (C++ GPU kernel, in-place, GIL-free)
            _cpp_quality_scale(
                raw_scores.data_ptr(),
                raw_boxes.data_ptr(),
                n,
                frame_w,
                frame_h,
                *self.quality_weights,
                stream_ptr,
            )

            # Step 3: ReID embeddings (on self.stream via explicit stream_ptr)
            reid_embeddings = None
            if frame_chw is not None:
                reid_embeddings, next_reid = self._compute_reid_embeddings(
                    raw_boxes,
                    raw_scores,
                    frame_chw,
                    frame_w,
                    frame_h,
                    frame_id=frame_id,
                    last_reid_frame=self.last_reid_frame,
                )
                if next_reid > self.last_reid_frame:
                    self.last_reid_frame = next_reid

            # Step 4: C++ filter+NMS+tracker (GIL released inside)
            n_out = self._wb.process_frame_postyolo(
                raw_boxes.data_ptr(),
                raw_scores.data_ptr(),
                raw_classes.data_ptr(),
                n,
                frame_w,
                frame_h,
                is_tiled,
                priors_ptr,
                prior_classes_ptr,
                num_priors,
                0.5,
                _ptr_or_zero(reid_embeddings),
                _ptr_or_zero(gmc_warp_tensor),
                0.0,
                1.0,
                self.out_boxes.data_ptr(),
                self.out_scores.data_ptr(),
                self.out_ids.data_ptr(),
                self.out_classes.data_ptr(),
                self.out_det_idx.data_ptr(),
                self.out_count.data_ptr(),
            )
            wb_done = torch.cuda.Event()
            wb_done.record(self.stream)

        torch.cuda.current_stream().wait_event(wb_done)
        n_out = int(self.out_count[0].item())  # saccade-allow-cpu

        if n_out == 0:
            return WorkbenchResult(
                ids=self.out_ids[:0],
                boxes=self.out_boxes[:0],
                scores=self.out_scores[:0],
                classes=self.out_classes[:0],
                det_idx=self.out_det_idx[:0],
            )

        out_ids = self.out_ids[:n_out]
        out_boxes = self.out_boxes[:n_out]
        out_scores = self.out_scores[:n_out]
        out_classes = self.out_classes[:n_out]
        out_det_idx = self.out_det_idx[:n_out]

        # Step 5: FP hard filter (Python, fast — applied to post-tracker output)
        if self.fp_hard_filter:
            (
                out_boxes,
                out_scores,
                out_classes,
                out_ids,
                out_det_idx,
            ) = _apply_fp_hard_filter(
                out_boxes,
                out_scores,
                out_classes,
                out_ids,
                out_det_idx,
                min_score=self.fp_min_score,
                max_suspicious_area=self.fp_max_suspicious_area,
                max_suspicious_score=self.fp_max_suspicious_score,
            )

        return WorkbenchResult(
            ids=out_ids,
            boxes=out_boxes,
            scores=out_scores,
            classes=out_classes,
            det_idx=out_det_idx,
        )
