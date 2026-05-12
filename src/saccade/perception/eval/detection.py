# NOTE: Import perception modules before torchvision in the calling script to avoid libjpeg conflict.
import torch
import numpy as np
from typing import Dict, Tuple, Any, List, Optional, cast

try:
    from saccade_tracking_ext import (
        filter_detections as cpp_filter_detections,
        filter_detections_cuda as cpp_filter_detections_cuda,
        merge_cross_tile_duplicates as cpp_merge_cross_tile_duplicates,
        merge_cross_tile_duplicates_cuda as cpp_merge_cross_tile_duplicates_cuda,
        nms_cuda as cpp_nms_cuda,
        letterbox_gpu as cpp_letterbox_gpu,
    )
except ImportError:
    cpp_filter_detections = None
    cpp_filter_detections_cuda = None
    cpp_merge_cross_tile_duplicates = None
    cpp_merge_cross_tile_duplicates_cuda = None
    cpp_nms_cuda = None
    cpp_letterbox_gpu = None

from torchvision.ops import batched_nms, nms

# 使用 Any 以避免 Mypy 對 Tensor | int 混合類型的屬性存取報錯
_filter_detections_cuda_workspace: Dict[Tuple[int], Dict[str, Any]] = {}
_nms_cuda_workspace: Dict[Tuple[int], Dict[str, Any]] = {}
_duplicate_merge_cuda_workspace: Dict[
    Tuple[int, torch.dtype, torch.dtype], Dict[str, Any]
] = {}
_SAHI_2X2_TILINGS = {"960p_2x2", "sahi_960p_2x2"}


def _is_2x2_tiling(tiling: str | None) -> bool:
    return tiling in _SAHI_2X2_TILINGS


def _is_tiled_tiling(tiling: str | None) -> bool:
    return bool(tiling) and (_is_2x2_tiling(tiling) or tiling == "960p_3x2")


def _prepare_canvas_960p(
    pool: Any,
    h_orig: int,
    w_orig: int,
) -> tuple[float, int, int, int, int]:
    r = 960.0 / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2
    if cpp_letterbox_gpu is not None:
        stream = torch.cuda.current_stream().cuda_stream
        cpp_letterbox_gpu(
            pool.frame_buffer.data_ptr(),
            w_orig,
            h_orig,
            pool.canvas_960p.data_ptr(),
            960,
            x_off,
            y_off,
            w_new,
            h_new,
            114.0 / 255.0,
            stream,
        )
    else:
        img_resized = torch.nn.functional.interpolate(
            pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
        ).squeeze(0)
        pool.canvas_960p.fill_(114.0 / 255.0)
        pool.canvas_960p[:, y_off : y_off + h_new, x_off : x_off + w_new].copy_(
            img_resized
        )
    return r, h_new, w_new, y_off, x_off


def _canvas_to_numpy_rgb(canvas: torch.Tensor) -> np.ndarray:
    return (
        canvas.clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


def _get_detector_static_batch_size(detector: Any) -> int:
    if getattr(detector, "is_dynamic", False):
        return 1
    shape = getattr(detector, "input_shape", None)
    if not shape and hasattr(detector, "engine"):
        shape = detector.engine.get_tensor_shape(detector.input_name)
    if not shape:
        return 1
    try:
        batch_dim = int(shape[0])
    except (TypeError, ValueError, IndexError):
        return 1
    return max(1, batch_dim)


def _build_sahi_trt_detection_model(
    detector: Any,
    detector_box_format: str,
):
    try:
        from sahi.models.base import DetectionModel
        from sahi.prediction import ObjectPrediction
    except ImportError as exc:
        raise RuntimeError(
            "tiling=sahi_960p_2x2 requires the optional 'sahi' package in the runtime environment"
        ) from exc

    class _TRTSahiDetectionModel(DetectionModel):
        def load_model(self):
            raise RuntimeError(
                "TRT-backed SAHI adapter requires an already-initialized detector"
            )

        def set_model(self, model: Any, **kwargs):
            self.model = model

        def perform_inference(self, image: np.ndarray):
            if self.model is None:
                raise ValueError("Model is not loaded, load it by calling .set_model()")
            image_tensor = (
                torch.from_numpy(np.array(image, copy=True))
                .to(device=self.model.device, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )
            batch_size = _get_detector_static_batch_size(self.model)
            if batch_size > 1:
                image_tensor = image_tensor.expand(batch_size, -1, -1, -1).contiguous()
            raw = self.model.detect_raw(image_tensor)
            self._original_predictions = [raw[0].clone()]
            self._original_shape = image.shape

        def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: list[list[int]] | None = [[0, 0]],
            full_shape_list: list[list[int]] | None = None,
        ):
            if shift_amount_list and shift_amount_list and isinstance(
                shift_amount_list[0], int
            ):
                shift_amount_list = [cast(list[int], shift_amount_list)]
            if full_shape_list and full_shape_list and isinstance(full_shape_list[0], int):
                full_shape_list = [cast(list[int], full_shape_list)]
            predictions = self._original_predictions or []
            object_prediction_list_per_image = []
            for image_ind, image_predictions in enumerate(predictions):
                shift_amount = (
                    [0, 0] if shift_amount_list is None else shift_amount_list[image_ind]
                )
                full_shape = (
                    list(self._original_shape[:2])
                    if full_shape_list is None
                    else full_shape_list[image_ind]
                )
                boxes = _decode_detector_boxes(
                    image_predictions[:, :4], detector_box_format
                )
                scores = image_predictions[:, 4]
                classes = image_predictions[:, 5].to(torch.int32)
                keep = scores >= self.confidence_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                classes = classes[keep]
                object_prediction_list = []
                for box, score, class_id in zip(boxes, scores, classes):
                    bbox = box.detach().cpu().tolist()
                    bbox[0] = max(0.0, bbox[0])
                    bbox[1] = max(0.0, bbox[1])
                    bbox[2] = min(float(full_shape[1]), bbox[2])
                    bbox[3] = min(float(full_shape[0]), bbox[3])
                    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                        continue
                    class_key = str(int(class_id.item()))
                    category_name = self.category_mapping.get(
                        class_key, f"class_{class_key}"
                    )
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=bbox,
                            category_id=int(class_id.item()),
                            category_name=category_name,
                            score=float(score.item()),
                            shift_amount=shift_amount,
                            full_shape=full_shape,
                        )
                    )
                object_prediction_list_per_image.append(object_prediction_list)
            self._object_prediction_list_per_image = object_prediction_list_per_image

    return _TRTSahiDetectionModel(
        model=detector,
        confidence_threshold=0.0,
        category_mapping={"0": "person"},
    )


def _decode_detector_boxes(
    boxes: torch.Tensor,
    detector_box_format: str,
) -> torch.Tensor:
    if detector_box_format == "xyxy":
        return boxes.clone()
    if detector_box_format == "cxcywh":
        cxcy = boxes[..., :2]
        wh = boxes[..., 2:4]
        return torch.cat([cxcy - wh * 0.5, cxcy + wh * 0.5], dim=-1)
    raise ValueError(f"Unsupported detector_box_format: {detector_box_format}")


def expand_boxes_with_ankle_keypoints(
    boxes: torch.Tensor,
    keypoints: Optional[torch.Tensor],
    frame_h: int,
    ankle_conf_thresh: float = 0.30,
    margin: float = 0.05,
    flat_aspect_thresh: float = 0.0,
) -> torch.Tensor:
    """Extend box bottom to ankle keypoint position if ankles are visible and below current bottom.

    COCO-17 indices 15 (left_ankle) and 16 (right_ankle) in absolute pixel coords.
    Keeps box top fixed; only extends y2 downward. No-ops when keypoints=None.

    flat_aspect_thresh: if > 0, only expand boxes with h/w < this value (flat/crouched boxes).
    Typical upright person h/w ≈ 2-3; set ~1.5 to skip well-proportioned boxes.
    """
    if keypoints is None or boxes.numel() == 0:
        return boxes

    # keypoints: [N, 17, 3] — (x, y, conf) in absolute pixels
    ankle_y = keypoints[:, 15:17, 1]  # [N, 2]
    ankle_conf = keypoints[:, 15:17, 2]  # [N, 2]

    # mask NaN (unmatched detections get NaN keypoints)
    valid = (ankle_conf > ankle_conf_thresh) & ~torch.isnan(ankle_y)
    has_ankle = valid.any(dim=1)  # [N]

    if not has_ankle.any():
        return boxes

    # Highest y (= lowest in image) among valid ankles per detection
    ankle_y_masked = ankle_y.clone()
    ankle_y_masked[~valid] = -1e6
    max_ankle_y = ankle_y_masked.amax(dim=1)  # [N]

    box_h = boxes[:, 3] - boxes[:, 1]
    box_w = boxes[:, 2] - boxes[:, 0]
    new_bottom = max_ankle_y + margin * box_h

    # Only expand where ankle falls below current box bottom
    should_expand = has_ankle & (new_bottom > boxes[:, 3])

    # Optionally restrict to flat (under-segmented) boxes: h/w < flat_aspect_thresh
    if flat_aspect_thresh > 0.0:
        safe_w = box_w.clamp(min=1.0)
        is_flat = (box_h / safe_w) < flat_aspect_thresh
        should_expand = should_expand & is_flat

    if not should_expand.any():
        return boxes

    boxes = boxes.clone()
    boxes[should_expand, 3] = torch.clamp(new_bottom[should_expand], max=float(frame_h))
    return boxes


def match_keypoints_to_boxes(
    target_boxes: torch.Tensor,
    source_boxes: torch.Tensor,
    source_keypoints: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Align keypoints from source detections to target boxes by nearest center."""
    if (
        source_keypoints is None
        or source_keypoints.numel() == 0
        or target_boxes.numel() == 0
        or source_boxes.numel() == 0
    ):
        return None
    src_cx = (source_boxes[:, 0] + source_boxes[:, 2]) * 0.5
    src_cy = (source_boxes[:, 1] + source_boxes[:, 3]) * 0.5
    dst_cx = (target_boxes[:, 0] + target_boxes[:, 2]) * 0.5
    dst_cy = (target_boxes[:, 1] + target_boxes[:, 3]) * 0.5
    dx = dst_cx.unsqueeze(1) - src_cx.unsqueeze(0)
    dy = dst_cy.unsqueeze(1) - src_cy.unsqueeze(0)
    nearest = (dx * dx + dy * dy).argmin(dim=1)
    return source_keypoints[nearest]


def _tile_seam_mask_for_boxes(
    boxes: torch.Tensor,
    *,
    tiling: str,
    frame_w: int,
    frame_h: int,
    seam_margin_canvas_px: float,
) -> torch.Tensor:
    if boxes.numel() == 0 or not _is_tiled_tiling(tiling):
        return torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.bool)

    r = 960.0 / max(frame_h, frame_w)
    h_new = int(frame_h * r)
    w_new = int(frame_w * r)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2

    seam_x_canvas = [320.0, 640.0]
    seam_y_canvas = [320.0, 640.0]
    if tiling == "960p_3x2":
        seam_x_canvas = [160.0, 320.0, 640.0, 800.0]

    seam_margin_orig = seam_margin_canvas_px / max(r, 1e-6)
    seam_x = [(sx - x_off) / r for sx in seam_x_canvas if x_off < sx < x_off + w_new]
    seam_y = [(sy - y_off) / r for sy in seam_y_canvas if y_off < sy < y_off + h_new]
    if not seam_x and not seam_y:
        return torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.bool)

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    near_any = torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.bool)
    for sx in seam_x:
        near_any |= (boxes[:, 0] <= sx) & (boxes[:, 2] >= sx)
        near_any |= (cx - sx).abs() <= seam_margin_orig
    for sy in seam_y:
        near_any |= (boxes[:, 1] <= sy) & (boxes[:, 3] >= sy)
        near_any |= (cy - sy).abs() <= seam_margin_orig
    return near_any


def _box_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-6)


def _box_iou_pairwise_diag(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """IoU between boxes_a[i] and boxes_b[i] for each i. Both (N,4). Returns (N,) on same device."""
    lt = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    rb = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def _box_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Full (N,M) IoU matrix between N and M boxes. Both on same device."""
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[
        :, None
    ]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[
        None, :
    ]
    return inter / (area_a + area_b - inter + 1e-6)


def filter_detections_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    *,
    score_threshold: float,
    track_person_only: bool,
    person_class: int,
    is_tiled: bool,
    frame_w: int,
    frame_h: int,
    person_geometry_prior: bool,
    geometry_suspect_support: bool,
    person_min_height_ratio: float,
    person_min_aspect: float,
    person_max_aspect: float,
    person_min_area_ratio: float,
    person_max_area_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if boxes.numel() == 0:
        return (
            torch.empty((0,), device=boxes.device, dtype=torch.long),
            torch.empty((0,), device=boxes.device, dtype=torch.bool),
            torch.empty((0,), device=boxes.device, dtype=torch.float32),
        )

    boxes = boxes.contiguous()
    scores = scores.contiguous()
    classes_i32 = classes.to(torch.int32).contiguous()

    if cpp_filter_detections_cuda is not None and boxes.is_cuda:
        workspace = _get_filter_detections_cuda_workspace(
            int(boxes.size(0)), boxes.device
        )
        cpp_filter_detections_cuda(
            boxes.data_ptr(),
            scores.data_ptr(),
            classes_i32.data_ptr(),
            int(boxes.size(0)),
            workspace["keep_indices"].data_ptr(),
            workspace["suspect_flags"].data_ptr(),
            workspace["quality_scores"].data_ptr(),
            workspace["out_count"].data_ptr(),
            score_threshold,
            track_person_only,
            person_class,
            is_tiled,
            frame_w,
            frame_h,
            person_geometry_prior,
            geometry_suspect_support,
            person_min_height_ratio,
            person_min_aspect,
            person_max_aspect,
            person_min_area_ratio,
            person_max_area_ratio,
            torch.cuda.current_stream().cuda_stream,
        )
        keep_count = int(workspace["out_count"].item())
        keep_i32 = workspace["keep_indices"][:keep_count]
        order = torch.argsort(keep_i32)
        keep = keep_i32[order].to(torch.long)
        suspect = workspace["suspect_flags"][:keep_count][order]
        quality = workspace["quality_scores"][:keep_count][order]
        return keep, suspect, quality

    if cpp_filter_detections is not None:
        device = boxes.device
        keep_np, suspect_np = cpp_filter_detections(
            boxes.detach().to("cpu").numpy(),
            scores.detach().to("cpu").numpy(),
            classes_i32.detach().to("cpu").numpy(),
            score_threshold,
            track_person_only,
            person_class,
            is_tiled,
            frame_w,
            frame_h,
            person_geometry_prior,
            geometry_suspect_support,
            person_min_height_ratio,
            person_min_aspect,
            person_max_aspect,
            person_min_area_ratio,
            person_max_area_ratio,
        )
        keep = torch.from_numpy(np.asarray(keep_np)).to(device=device, dtype=torch.long)
        suspect = torch.from_numpy(np.asarray(suspect_np)).to(
            device=device, dtype=torch.bool
        )
        return keep, suspect, None

    keep_mask = scores > score_threshold
    geometry_clean_mask = torch.ones_like(keep_mask, dtype=torch.bool)
    if track_person_only:
        keep_mask = keep_mask & (classes_i32 == person_class)
    if is_tiled:
        box_cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        box_cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        keep_mask = (
            keep_mask
            & (box_cx >= 0)
            & (box_cx < frame_w)
            & (box_cy >= 0)
            & (box_cy < frame_h)
        )
    if person_geometry_prior:
        box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6)
        box_w = box_wh[:, 0]
        box_h = box_wh[:, 1]
        box_aspect = box_h / box_w
        box_area_ratio = (box_w * box_h) / max(float(frame_w * frame_h), 1.0)
        if person_min_height_ratio > 0.0:
            geometry_clean_mask = geometry_clean_mask & (
                box_h >= frame_h * person_min_height_ratio
            )
        if person_min_aspect > 0.0:
            geometry_clean_mask = geometry_clean_mask & (
                box_aspect >= person_min_aspect
            )
        if person_max_aspect > 0.0:
            geometry_clean_mask = geometry_clean_mask & (
                box_aspect <= person_max_aspect
            )
        if person_min_area_ratio > 0.0:
            geometry_clean_mask = geometry_clean_mask & (
                box_area_ratio >= person_min_area_ratio
            )
        if person_max_area_ratio > 0.0:
            geometry_clean_mask = geometry_clean_mask & (
                box_area_ratio <= person_max_area_ratio
            )
        if not geometry_suspect_support:
            keep_mask = keep_mask & geometry_clean_mask

    suspect_full = (
        keep_mask & ~geometry_clean_mask
        if (person_geometry_prior and geometry_suspect_support)
        else torch.zeros_like(keep_mask, dtype=torch.bool)
    )
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return keep_indices, suspect_full[keep_indices], None


def _get_filter_detections_cuda_workspace(
    num_boxes: int,
    box_device: torch.device,
) -> Dict[str, Any]:
    device_index = (
        box_device.index
        if box_device.index is not None
        else torch.cuda.current_device()
    )
    key = (device_index,)
    workspace = _filter_detections_cuda_workspace.get(key)
    if workspace is not None and int(workspace["capacity"]) >= num_boxes:
        return workspace

    capacity = max(num_boxes, 1)
    workspace = {
        "capacity": capacity,
        "keep_indices": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "suspect_flags": torch.empty((capacity,), device=box_device, dtype=torch.bool),
        "quality_scores": torch.empty(
            (capacity,), device=box_device, dtype=torch.float32
        ),
        "out_count": torch.zeros((), device=box_device, dtype=torch.int32),
    }
    _filter_detections_cuda_workspace[key] = workspace
    return workspace


def _get_nms_cuda_workspace(
    num_boxes: int,
    box_device: torch.device,
) -> Dict[str, Any]:
    device_index = (
        box_device.index
        if box_device.index is not None
        else torch.cuda.current_device()
    )
    key = (device_index,)
    col_blocks = (num_boxes + 63) // 64
    mask_slots = max(num_boxes * col_blocks, 1)
    workspace = _nms_cuda_workspace.get(key)
    if (
        workspace is not None
        and int(workspace["capacity"]) >= num_boxes
        and int(workspace["mask_slots"]) >= mask_slots
        and int(workspace["col_blocks"]) >= col_blocks
    ):
        return workspace

    capacity = max(num_boxes, 1)
    workspace = {
        "capacity": capacity,
        "col_blocks": max(col_blocks, 1),
        "mask_slots": mask_slots,
        "keep_indices": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "suppression_masks": torch.empty(
            (mask_slots,), device=box_device, dtype=torch.int64
        ),
        "remv": torch.empty(
            (max(col_blocks, 1),), device=box_device, dtype=torch.int64
        ),
        "out_count": torch.zeros((), device=box_device, dtype=torch.int32),
    }
    _nms_cuda_workspace[key] = workspace
    return workspace


def nms_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
    *,
    class_aware: bool,
    priors: torch.Tensor | None = None,
    prior_classes: torch.Tensor | None = None,
    prior_iou_threshold: float = 0.50,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    boxes = boxes.contiguous()
    scores = scores.contiguous()
    classes_i32 = classes.to(torch.int32).contiguous()

    # O-NMS logic: priors can rescue boxes that vanilla NMS would suppress.
    if priors is not None and priors.numel() > 0:
        from torchvision.ops import nms, batched_nms, box_iou
        if class_aware:
            keep = batched_nms(boxes, scores, classes_i32, iou_threshold)
        else:
            keep = nms(boxes, scores, iou_threshold)

        keep_set = set(keep.tolist())
        all_indices = set(range(boxes.shape[0]))
        suppressed = list(all_indices - keep_set)

        if not suppressed:
            return keep

        suppressed_tensor = torch.tensor(suppressed, dtype=torch.long, device=boxes.device)
        suppressed_boxes = boxes[suppressed_tensor]

        if class_aware and prior_classes is not None and prior_classes.numel() > 0:
            prior_classes_i32 = prior_classes.to(torch.int32).contiguous()
            suppressed_classes = classes_i32[suppressed_tensor]
            rescued_parts: list[torch.Tensor] = []
            for cls in torch.unique(suppressed_classes).tolist():
                cls_int = int(cls)
                class_suppressed = suppressed_tensor[suppressed_classes == cls_int]
                class_priors = priors[prior_classes_i32 == cls_int]
                if class_priors.numel() == 0:
                    continue
                class_iou = box_iou(boxes[class_suppressed], class_priors)
                max_ious, _ = class_iou.max(dim=1)
                rescued = class_suppressed[max_ious > prior_iou_threshold]
                if rescued.numel() > 0:
                    rescued_parts.append(rescued)
            rescued_indices = (
                torch.cat(rescued_parts)
                if rescued_parts
                else torch.empty((0,), device=boxes.device, dtype=torch.long)
            )
        else:
            iou_matrix = box_iou(suppressed_boxes, priors)
            max_ious, _ = iou_matrix.max(dim=1)
            rescued_indices = suppressed_tensor[max_ious > prior_iou_threshold]

        if rescued_indices.numel() > 0:
            print(f"  [O-NMS] Rescued {rescued_indices.numel()} suppressed boxes that overlapped with active tracks!")
            final_keep = torch.cat([keep, rescued_indices])
            final_keep_scores = scores[final_keep]
            _, sort_idx = final_keep_scores.sort(descending=True)
            return final_keep[sort_idx]
        return keep

    if cpp_nms_cuda is not None and boxes.is_cuda:
        order = torch.argsort(scores, descending=True).contiguous()
        workspace = _get_nms_cuda_workspace(int(boxes.size(0)), boxes.device)
        cpp_nms_cuda(
            boxes.data_ptr(),
            scores.data_ptr(),
            classes_i32.data_ptr(),
            order.data_ptr(),
            int(boxes.size(0)),
            workspace["keep_indices"].data_ptr(),
            workspace["suppression_masks"].data_ptr(),
            workspace["remv"].data_ptr(),
            workspace["out_count"].data_ptr(),
            iou_threshold,
            class_aware,
            torch.cuda.current_stream().cuda_stream,
        )
        keep_count = int(workspace["out_count"].item())
        return cast(torch.Tensor, workspace["keep_indices"][:keep_count].to(torch.long))

    if class_aware:
        res_b: torch.Tensor = cast(
            torch.Tensor, batched_nms(boxes, scores, classes_i32, iou_threshold)
        )
        return res_b
    res_n: torch.Tensor = cast(torch.Tensor, nms(boxes, scores, iou_threshold))
    return res_n


def _get_duplicate_merge_cuda_workspace(
    num_boxes: int,
    box_device: torch.device,
    box_dtype: torch.dtype,
    score_dtype: torch.dtype,
) -> Dict[str, Any]:
    device_index = (
        box_device.index
        if box_device.index is not None
        else torch.cuda.current_device()
    )
    key = (device_index, box_dtype, score_dtype)
    workspace = _duplicate_merge_cuda_workspace.get(key)
    if workspace is not None and int(workspace["capacity"]) >= num_boxes:
        return workspace

    capacity = max(num_boxes, 1)
    workspace = {
        "capacity": capacity,
        "anchor_indices": torch.empty(
            (capacity,), device=box_device, dtype=torch.int32
        ),
        "box_sums": torch.empty((capacity, 4), device=box_device, dtype=torch.float32),
        "score_sums": torch.empty((capacity,), device=box_device, dtype=torch.float32),
        "score_bits_max": torch.empty(
            (capacity,), device=box_device, dtype=torch.int32
        ),
        "best_boxes": torch.empty(
            (capacity, 4), device=box_device, dtype=torch.float32
        ),
        "best_key_bits": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "cluster_counts": torch.empty(
            (capacity,), device=box_device, dtype=torch.int32
        ),
        "out_boxes": torch.empty((capacity, 4), device=box_device, dtype=box_dtype),
        "out_scores": torch.empty((capacity,), device=box_device, dtype=score_dtype),
        "out_classes": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "out_count": torch.zeros((), device=box_device, dtype=torch.int32),
    }
    _duplicate_merge_cuda_workspace[key] = workspace
    return workspace


def merge_cross_tile_duplicates(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    center_threshold: float = 0.18,
    area_ratio_threshold: float = 0.6,
    *,
    tiling: str | None = None,
    frame_w: int = 0,
    frame_h: int = 0,
    seam_margin_canvas_px: float = 24.0,
    seam_center_scale: float = 1.8,
    seam_area_ratio_threshold: float = 0.30,
    seam_min_overlap_ratio: float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (boxes, scores, classes, merge_counts) where merge_counts[i] is the
    number of original detections fused into output box i (1 = no merge happened)."""
    tiled_seam_coord_weight = 0.5
    tiled_best_blend = 0.5
    n = boxes.size(0)
    if boxes.numel() == 0 or n <= 1:
        return (
            boxes,
            scores,
            classes,
            torch.ones(n, device=boxes.device, dtype=torch.long),
        )

    order = torch.argsort(scores, descending=True)
    remaining = order.tolist()
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    merged_counts: List[int] = []
    seam_mask = _tile_seam_mask_for_boxes(
        boxes,
        tiling=tiling or "",
        frame_w=frame_w,
        frame_h=frame_h,
        seam_margin_canvas_px=seam_margin_canvas_px,
    )

    while remaining:
        anchor_idx = remaining[0]
        anchor_box = boxes[anchor_idx]
        anchor_class = classes[anchor_idx]

        candidate_indices = torch.tensor(
            remaining, device=boxes.device, dtype=torch.long
        )
        candidate_boxes = boxes[candidate_indices]
        candidate_classes = classes[candidate_indices]

        same_class = candidate_classes == anchor_class
        ious = _box_iou_single(anchor_box, candidate_boxes)

        anchor_center = (anchor_box[:2] + anchor_box[2:]) * 0.5
        candidate_centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) * 0.5
        center_dist = torch.linalg.norm(candidate_centers - anchor_center, dim=1)

        anchor_wh = (anchor_box[2:] - anchor_box[:2]).clamp(min=1e-6)
        candidate_wh = (candidate_boxes[:, 2:] - candidate_boxes[:, :2]).clamp(min=1e-6)
        min_wh = torch.minimum(candidate_wh, anchor_wh.unsqueeze(0))
        center_gate = torch.linalg.norm(min_wh, dim=1) * center_threshold

        anchor_area = float(anchor_wh[0] * anchor_wh[1])
        candidate_areas = candidate_wh[:, 0] * candidate_wh[:, 1]
        area_ratio = torch.minimum(
            candidate_areas / max(anchor_area, 1e-6),
            torch.tensor(anchor_area, device=boxes.device)
            / candidate_areas.clamp(min=1e-6),
        )
        inter_lt = torch.maximum(anchor_box[:2], candidate_boxes[:, :2])
        inter_rb = torch.minimum(anchor_box[2:], candidate_boxes[:, 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        overlap_ratio_x = inter_wh[:, 0] / min_wh[:, 0].clamp(min=1e-6)
        overlap_ratio_y = inter_wh[:, 1] / min_wh[:, 1].clamp(min=1e-6)

        anchor_is_seam = bool(seam_mask[anchor_idx].item())
        candidate_is_seam = seam_mask[candidate_indices]
        seam_pair = candidate_is_seam | anchor_is_seam

        base_duplicate_mask = same_class & (
            (ious >= iou_threshold)
            | ((center_dist <= center_gate) & (area_ratio >= area_ratio_threshold))
        )
        seam_duplicate_mask = (
            same_class
            & seam_pair
            & (
                (center_dist <= center_gate * seam_center_scale)
                & (area_ratio >= seam_area_ratio_threshold)
                & (overlap_ratio_x >= seam_min_overlap_ratio)
                & (overlap_ratio_y >= seam_min_overlap_ratio)
            )
        )
        duplicate_mask = base_duplicate_mask | seam_duplicate_mask

        cluster_indices = candidate_indices[duplicate_mask]
        cluster_boxes = boxes[cluster_indices]
        cluster_scores = scores[cluster_indices]
        cluster_seam = seam_mask[cluster_indices]
        if (~cluster_seam).any():
            preferred = torch.where(~cluster_seam)[0]
            best_local = preferred[torch.argmax(cluster_scores[preferred])]
        else:
            best_local = torch.argmax(cluster_scores)
        if tiling:
            coord_weights = cluster_scores * torch.where(
                cluster_seam,
                torch.full_like(cluster_scores, tiled_seam_coord_weight),
                torch.ones_like(cluster_scores),
            )
            fused_box = (cluster_boxes * coord_weights[:, None]).sum(
                dim=0
            ) / coord_weights.sum().clamp(min=1e-6)
            if cluster_indices.numel() > 1:
                fused_box = (
                    fused_box * (1.0 - tiled_best_blend)
                    + cluster_boxes[best_local] * tiled_best_blend
                )
            else:
                fused_box = cluster_boxes[best_local]
        else:
            fused_box = cluster_boxes[best_local]
        fused_score = cluster_scores[best_local]

        merged_boxes.append(fused_box)
        merged_scores.append(fused_score)
        merged_classes.append(anchor_class)
        merged_counts.append(int(cluster_indices.size(0)))

        cluster_set = set(int(idx) for idx in cluster_indices.tolist())
        remaining = [idx for idx in remaining if idx not in cluster_set]

    if not merged_boxes:
        return (
            torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype),
            torch.empty((0,), device=boxes.device, dtype=scores.dtype),
            torch.empty((0,), device=boxes.device, dtype=classes.dtype),
            torch.empty((0,), device=boxes.device, dtype=torch.long),
        )

    return (
        torch.stack(merged_boxes, dim=0),
        torch.stack(merged_scores, dim=0),
        torch.stack(merged_classes, dim=0),
        torch.tensor(merged_counts, device=boxes.device, dtype=torch.long),
    )


def merge_cross_tile_duplicates_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    center_threshold: float = 0.18,
    area_ratio_threshold: float = 0.6,
    *,
    tiling: str | None = None,
    frame_w: int = 0,
    frame_h: int = 0,
    seam_margin_canvas_px: float = 24.0,
    seam_center_scale: float = 1.8,
    seam_area_ratio_threshold: float = 0.30,
    seam_min_overlap_ratio: float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (boxes, scores, classes, merge_counts).  merge_counts[i] is the
    number of original detections fused into output box i (1 = unmerged)."""
    if boxes.numel() == 0 or boxes.size(0) <= 1:
        return merge_cross_tile_duplicates(
            boxes,
            scores,
            classes,
            iou_threshold,
            center_threshold,
            area_ratio_threshold,
            tiling=tiling,
            frame_w=frame_w,
            frame_h=frame_h,
            seam_margin_canvas_px=seam_margin_canvas_px,
            seam_center_scale=seam_center_scale,
            seam_area_ratio_threshold=seam_area_ratio_threshold,
            seam_min_overlap_ratio=seam_min_overlap_ratio,
        )

    boxes = boxes.contiguous()
    scores = scores.contiguous()
    classes_i32 = classes.to(torch.int32).contiguous()

    if cpp_merge_cross_tile_duplicates_cuda is not None and boxes.is_cuda:
        tiling_mode = 0
        if _is_2x2_tiling(tiling):
            tiling_mode = 1
        elif tiling == "960p_3x2":
            tiling_mode = 2
        workspace = _get_duplicate_merge_cuda_workspace(
            int(boxes.size(0)),
            boxes.device,
            boxes.dtype,
            scores.dtype,
        )
        cpp_merge_cross_tile_duplicates_cuda(
            boxes.data_ptr(),
            scores.data_ptr(),
            classes_i32.data_ptr(),
            int(boxes.size(0)),
            workspace["anchor_indices"].data_ptr(),
            workspace["box_sums"].data_ptr(),
            workspace["score_sums"].data_ptr(),
            workspace["score_bits_max"].data_ptr(),
            workspace["best_boxes"].data_ptr(),
            workspace["best_key_bits"].data_ptr(),
            workspace["cluster_counts"].data_ptr(),
            workspace["out_boxes"].data_ptr(),
            workspace["out_scores"].data_ptr(),
            workspace["out_classes"].data_ptr(),
            workspace["out_count"].data_ptr(),
            iou_threshold,
            center_threshold,
            area_ratio_threshold,
            tiling_mode,
            frame_w,
            frame_h,
            seam_margin_canvas_px,
            seam_center_scale,
            seam_area_ratio_threshold,
            seam_min_overlap_ratio,
            torch.cuda.current_stream().cuda_stream,
        )
        merged_count = int(workspace["out_count"].item())
        return (
            workspace["out_boxes"][:merged_count],
            workspace["out_scores"][:merged_count],
            workspace["out_classes"][:merged_count],
            workspace["cluster_counts"][:merged_count].to(torch.long),
        )

    if cpp_merge_cross_tile_duplicates is not None:
        device = boxes.device
        boxes_np = boxes.detach().to("cpu").numpy()
        scores_np = scores.detach().to("cpu").numpy()
        classes_np = classes_i32.detach().to("cpu").numpy()
        merged_boxes_np, merged_scores_np, merged_classes_np = (
            cpp_merge_cross_tile_duplicates(
                boxes_np,
                scores_np,
                classes_np,
                iou_threshold,
                center_threshold,
                area_ratio_threshold,
                1 if _is_2x2_tiling(tiling) else 2 if tiling == "960p_3x2" else 0,
                frame_w,
                frame_h,
                seam_margin_canvas_px,
                seam_center_scale,
                seam_area_ratio_threshold,
                seam_min_overlap_ratio,
            )
        )
        merged_boxes = torch.from_numpy(np.asarray(merged_boxes_np)).to(
            device=device, dtype=boxes.dtype
        )
        merged_scores = torch.from_numpy(np.asarray(merged_scores_np)).to(
            device=device, dtype=scores.dtype
        )
        merged_classes = torch.from_numpy(np.asarray(merged_classes_np)).to(
            device=device, dtype=classes_i32.dtype
        )
        # C++ CPU path does not expose per-cluster counts; treat all as unmerged.
        merged_count = int(merged_boxes.size(0))
        return (
            merged_boxes,
            merged_scores,
            merged_classes,
            torch.ones(merged_count, device=device, dtype=torch.long),
        )

    return merge_cross_tile_duplicates(
        boxes,
        scores,
        classes_i32,
        iou_threshold,
        center_threshold,
        area_ratio_threshold,
    )


def detect_single_patch_640(
    detector: Any,
    pool: Any,
    h_orig: int,
    w_orig: int,
    preprocess_modes: List[str],
    detector_box_format: str = "xyxy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "letterbox" in preprocess_modes:
        r = 640.0 / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * r), int(w_orig * r)
        y_off = (640 - h_new) // 2
        x_off = (640 - w_new) // 2
        if cpp_letterbox_gpu is not None:
            stream = torch.cuda.current_stream().cuda_stream
            cpp_letterbox_gpu(
                pool.frame_buffer.data_ptr(),
                w_orig,
                h_orig,
                pool.canvas_640p.data_ptr(),
                640,
                x_off,
                y_off,
                w_new,
                h_new,
                114.0 / 255.0,
                stream,
            )
        else:
            img_resized = torch.nn.functional.interpolate(
                pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
            ).squeeze(0)
            pool.canvas_640p.fill_(114.0 / 255.0)
            pool.canvas_640p[:, y_off : y_off + h_new, x_off : x_off + w_new].copy_(
                img_resized
            )

        raw_dets = detector.detect_raw(pool.canvas_640p.unsqueeze(0))
        boxes = _decode_detector_boxes(raw_dets[0, :, :4], detector_box_format)
        scores = raw_dets[0, :, 4]
        classes = raw_dets[0, :, 5]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_off) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_off) / r
        return boxes, scores, classes

    img_input = torch.nn.functional.interpolate(
        pool.frame_buffer.unsqueeze(0), size=(640, 640)
    )
    raw_dets = detector.detect_raw(img_input)
    boxes = _decode_detector_boxes(raw_dets[0, :, :4], detector_box_format)
    scores = raw_dets[0, :, 4]
    classes = raw_dets[0, :, 5]

    boxes[:, [0, 2]] /= 640.0 / w_orig
    boxes[:, [1, 3]] /= 640.0 / h_orig
    return boxes, scores, classes


def detect_single_patch_960(
    detector: Any,
    pool: Any,
    h_orig: int,
    w_orig: int,
    preprocess_modes: List[str],
    detector_box_format: str = "xyxy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if "letterbox" in preprocess_modes:
        r, _h_new, _w_new, y_off, x_off = _prepare_canvas_960p(
            pool, h_orig, w_orig
        )

        keypoints = None
        if hasattr(detector, "pose"):
            boxes, scores, classes, keypoints = detector.detect(
                pool.canvas_960p.unsqueeze(0),
                conf_threshold=0.0,
            )
        else:
            raw_dets = detector.detect_raw(pool.canvas_960p.unsqueeze(0))
            boxes = _decode_detector_boxes(raw_dets[0, :, :4], detector_box_format)
            scores = raw_dets[0, :, 4]
            classes = raw_dets[0, :, 5]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_off) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_off) / r
        if keypoints is not None:
            keypoints = keypoints.clone()
            keypoints[:, :, 0] = (keypoints[:, :, 0] - x_off) / r
            keypoints[:, :, 1] = (keypoints[:, :, 1] - y_off) / r
        return boxes, scores, classes, keypoints

    img_input = torch.nn.functional.interpolate(
        pool.frame_buffer.unsqueeze(0), size=(960, 960)
    )
    keypoints = None
    if hasattr(detector, "pose"):
        boxes, scores, classes, keypoints = detector.detect(
            img_input,
            conf_threshold=0.0,
        )
    else:
        raw_dets = detector.detect_raw(img_input)
        boxes = _decode_detector_boxes(raw_dets[0, :, :4], detector_box_format)
        scores = raw_dets[0, :, 4]
        classes = raw_dets[0, :, 5]

    boxes[:, [0, 2]] /= 960.0 / w_orig
    boxes[:, [1, 3]] /= 960.0 / h_orig
    if keypoints is not None:
        keypoints = keypoints.clone()
        keypoints[:, :, 0] /= 960.0 / w_orig
        keypoints[:, :, 1] /= 960.0 / h_orig
    return boxes, scores, classes, keypoints


def detect_native_960(
    detector: Any,
    pool: Any,
    h_orig: int,
    w_orig: int,
    preprocess_modes: List[str],
    detector_box_format: str = "xyxy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
    boxes, scores, classes, keypoints = detect_single_patch_960(
        detector, pool, h_orig, w_orig, preprocess_modes, detector_box_format
    )
    return boxes, scores, classes, False, keypoints


def _detect_tiles(detector: Any, tiles_batch: torch.Tensor) -> torch.Tensor:
    """Run detect_raw on a tile batch; fall back to sequential if engine is static batch-1.
    Each slice must be cloned because detect_raw returns a reference to a shared buffer."""
    raw = detector.detect_raw(tiles_batch)
    n_tiles = tiles_batch.shape[0]
    if raw.shape[0] == n_tiles:
        return raw
    pieces = [
        detector.detect_raw(tiles_batch[i : i + 1]).clone() for i in range(n_tiles)
    ]
    return torch.cat(pieces, dim=0)


def detect_960p_3x2_tiled(
    detector: Any,
    pool: Any,
    h_orig: int,
    w_orig: int,
    preprocess_modes: List[str],
    detector_box_format: str = "xyxy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
    """3×2 tiling on 960p canvas with overlap. Two batch-4 passes (6 tiles).
    x: 3 cols at stride=160 on 960p (75% x-overlap, same scale as 2×2).
    y: 2 rows at stride=320 on 960p (50% y-overlap, same as 2×2).
    """
    if w_orig <= 960 and h_orig <= 960:
        boxes, scores, classes = detect_single_patch_640(
            detector, pool, h_orig, w_orig, preprocess_modes, detector_box_format
        )
        return boxes, scores, classes, False, None

    r, _h_new, _w_new, y_off, x_off = _prepare_canvas_960p(pool, h_orig, w_orig)

    # Batch 6: 6 tiles in one pass
    pool.tiles_batch6[0].copy_(pool.canvas_960p[:, 0:640, 0:640])
    pool.tiles_batch6[1].copy_(pool.canvas_960p[:, 0:640, 160:800])
    pool.tiles_batch6[2].copy_(pool.canvas_960p[:, 0:640, 320:960])
    pool.tiles_batch6[3].copy_(pool.canvas_960p[:, 320:960, 0:640])
    pool.tiles_batch6[4].copy_(pool.canvas_960p[:, 320:960, 160:800])
    pool.tiles_batch6[5].copy_(pool.canvas_960p[:, 320:960, 320:960])

    raw = _detect_tiles(detector, pool.tiles_batch6)

    boxes = _decode_detector_boxes(raw[:, :, :4], detector_box_format)
    boxes[:, :, [0, 2]] = (boxes[:, :, [0, 2]] + pool.tile_3x2_dx - x_off) / r
    boxes[:, :, [1, 3]] = (boxes[:, :, [1, 3]] + pool.tile_3x2_dy - y_off) / r

    all_boxes = boxes.reshape(-1, 4)
    all_scores = raw[:, :, 4].reshape(-1)
    all_classes = raw[:, :, 5].reshape(-1)
    return all_boxes, all_scores, all_classes, True, None


def detect_adaptive_960_tiled(
    detector: Any,
    pool: Any,
    h_orig: int,
    w_orig: int,
    preprocess_modes: List[str],
    detector_box_format: str = "xyxy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
    if w_orig <= 960 and h_orig <= 960:
        boxes, scores, classes = detect_single_patch_640(
            detector, pool, h_orig, w_orig, preprocess_modes, detector_box_format
        )
        return boxes, scores, classes, False, None

    r, _h_new, _w_new, y_off, x_off = _prepare_canvas_960p(pool, h_orig, w_orig)

    pool.tiles_batch4[0].copy_(pool.canvas_960p[:, 0:640, 0:640])
    pool.tiles_batch4[1].copy_(pool.canvas_960p[:, 0:640, 320:960])
    pool.tiles_batch4[2].copy_(pool.canvas_960p[:, 320:960, 0:640])
    pool.tiles_batch4[3].copy_(pool.canvas_960p[:, 320:960, 320:960])

    raw_dets = _detect_tiles(detector, pool.tiles_batch4)
    # One clone for all tiles, two broadcast ops using pre-allocated offsets
    # instead of 4 individual clones + 8 per-tile scalar ops + 3 cat calls.
    all_boxes = _decode_detector_boxes(raw_dets[:, :, :4], detector_box_format)
    all_boxes[:, :, [0, 2]] = (all_boxes[:, :, [0, 2]] + pool.tile_dx - x_off) / r
    all_boxes[:, :, [1, 3]] = (all_boxes[:, :, [1, 3]] + pool.tile_dy - y_off) / r
    # Scores and classes are non-contiguous views; reshape() copies them automatically.
    return (
        all_boxes.reshape(-1, 4),
        raw_dets[:, :, 4].reshape(-1),
        raw_dets[:, :, 5].reshape(-1),
        True,
        None,
    )


def detect_sahi_960p_2x2(
    detector: Any,
    pool: Any,
    h_orig: int,
    w_orig: int,
    preprocess_modes: List[str],
    detector_box_format: str = "xyxy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
    if w_orig <= 960 and h_orig <= 960:
        boxes, scores, classes = detect_single_patch_640(
            detector, pool, h_orig, w_orig, preprocess_modes, detector_box_format
        )
        return boxes, scores, classes, False, None

    if "letterbox" not in preprocess_modes:
        raise RuntimeError(
            "tiling=sahi_960p_2x2 currently requires letterbox preprocessing"
        )

    from sahi.predict import get_sliced_prediction

    r, _h_new, _w_new, y_off, x_off = _prepare_canvas_960p(pool, h_orig, w_orig)
    canvas_rgb = _canvas_to_numpy_rgb(pool.canvas_960p)

    model_cache = getattr(detector, "_sahi_model_cache", None)
    if model_cache is None:
        model_cache = {}
        setattr(detector, "_sahi_model_cache", model_cache)
    sahi_model = model_cache.get(detector_box_format)
    if sahi_model is None:
        sahi_model = _build_sahi_trt_detection_model(detector, detector_box_format)
        model_cache[detector_box_format] = sahi_model

    prediction = get_sliced_prediction(
        image=canvas_rgb,
        detection_model=sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
        perform_standard_pred=False,
        auto_slice_resolution=False,
        postprocess_class_agnostic=False,
        verbose=0,
    )
    object_predictions = prediction.object_prediction_list
    device = pool.frame_buffer.device
    if not object_predictions:
        return (
            torch.empty((0, 4), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.float32),
            True,
            None,
        )

    boxes = [object_prediction.bbox.to_xyxy() for object_prediction in object_predictions]
    scores = [object_prediction.score.value for object_prediction in object_predictions]
    classes = [float(object_prediction.category.id) for object_prediction in object_predictions]
    boxes_t = torch.tensor(boxes, device=device, dtype=torch.float32)
    scores_t = torch.tensor(scores, device=device, dtype=torch.float32)
    classes_t = torch.tensor(classes, device=device, dtype=torch.float32)
    boxes_t[:, [0, 2]] = (boxes_t[:, [0, 2]] - x_off) / r
    boxes_t[:, [1, 3]] = (boxes_t[:, [1, 3]] - y_off) / r
    return boxes_t, scores_t, classes_t, True, None
