import torch
from dataclasses import dataclass
from typing import Any, Optional

from saccade_tracking_ext import (
    Workbench as _WorkbenchExt,
    PerceptionPipeline,
    PerceptionPipelineConfig,
)
from saccade.perception.tracking import GPUByteTracker


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
    def __init__(
        self,
        proxy: Any,
        pipeline_cfg: PerceptionPipelineConfig,
        device: str = "cuda:0",
        max_dets: int = 2048,
        max_tracks: int = 256,
    ):
        self.device = device
        self.max_dets = max_dets
        self.max_tracks = max_tracks

        # Per-workbench instances — no sharing of mutable state
        # Use Python wrapper so evaluator.py is happy
        self.tracker = GPUByteTracker(max_objects=max_tracks * 8)
        # reid_ptr=0, num_identities=0
        self.pipeline = PerceptionPipeline(0, 0, pipeline_cfg)
        self.stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]

        self.proxy = proxy
        if hasattr(self.proxy, "tracker"):
            self.proxy.tracker = self.tracker

        # Output staging buffers
        self.out_boxes = torch.empty(
            (max_tracks, 4), dtype=torch.float32, device=device
        )
        self.out_scores = torch.empty((max_tracks,), dtype=torch.float32, device=device)
        self.out_ids = torch.empty((max_tracks,), dtype=torch.int32, device=device)
        self.out_classes = torch.empty((max_tracks,), dtype=torch.int32, device=device)
        self.out_det_idx = torch.empty((max_tracks,), dtype=torch.int32, device=device)
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
            wb_done = torch.cuda.Event()  # type: ignore[no-untyped-call]
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

    def process_detections(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        *,
        frame_w: int,
        frame_h: int,
        embeddings: Optional[torch.Tensor] = None,
        gmc: Optional[torch.Tensor] = None,
        priors: Optional[torch.Tensor] = None,
        prior_classes: Optional[torch.Tensor] = None,
        is_tiled: bool = False,
        prior_iou_threshold: float = 0.5,
        light_factor: float = 0.0,
        mid_thresh_scale: float = 1.0,
    ) -> WorkbenchResult:
        """GIL-free hot path for pre-decoded boxes already in original image coordinates.

        Use this when the caller has already run detect_fn (letterbox+YOLO+rescale).
        """
        priors_ptr, prior_classes_ptr, num_priors = _ptrs_or_zero_priors(
            priors, prior_classes
        )
        # Materialize tensors on the caller's current stream, then record an event
        # so self.stream waits for those writes before reading the raw pointers.
        raw_boxes = boxes.to(torch.float32).contiguous()
        raw_scores = scores.to(torch.float32).contiguous()
        raw_classes = classes.to(torch.int32).contiguous()
        caller_event = torch.cuda.Event()  # type: ignore[no-untyped-call]
        caller_event.record()  # records on the calling thread's current stream

        with torch.cuda.stream(self.stream):
            self.stream.wait_event(caller_event)
            n = self._wb.process_frame_postyolo(
                raw_boxes.data_ptr(),
                raw_scores.data_ptr(),
                raw_classes.data_ptr(),
                int(raw_boxes.shape[0]),
                frame_w,
                frame_h,
                is_tiled,
                priors_ptr,
                prior_classes_ptr,
                num_priors,
                prior_iou_threshold,
                _ptr_or_zero(embeddings),
                _ptr_or_zero(gmc),
                light_factor,
                mid_thresh_scale,
                self.out_boxes.data_ptr(),
                self.out_scores.data_ptr(),
                self.out_ids.data_ptr(),
                self.out_classes.data_ptr(),
                self.out_det_idx.data_ptr(),
                self.out_count.data_ptr(),
            )
            # Record event so caller's stream sees the completed writes
            wb_done = torch.cuda.Event()  # type: ignore[no-untyped-call]
            wb_done.record(self.stream)

        torch.cuda.current_stream().wait_event(wb_done)
        return WorkbenchResult(
            ids=self.out_ids[:n],
            boxes=self.out_boxes[:n],
            scores=self.out_scores[:n],
            classes=self.out_classes[:n],
            det_idx=self.out_det_idx[:n],
        )
