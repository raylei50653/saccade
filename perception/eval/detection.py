# NOTE: Import perception modules before torchvision in the calling script to avoid libjpeg conflict.
import torch
import numpy as np
from torchvision.ops import batched_nms, nms

try:
    from saccade_tracking_ext import (
        merge_cross_tile_duplicates as cpp_merge_cross_tile_duplicates,
        merge_cross_tile_duplicates_cuda as cpp_merge_cross_tile_duplicates_cuda,
    )
except ImportError:
    cpp_merge_cross_tile_duplicates = None
    cpp_merge_cross_tile_duplicates_cuda = None

_duplicate_merge_cuda_workspace: dict[tuple[int, torch.dtype, torch.dtype], dict[str, torch.Tensor | int]] = {}


def _box_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-6)


def _get_duplicate_merge_cuda_workspace(
    num_boxes: int,
    box_device: torch.device,
    box_dtype: torch.dtype,
    score_dtype: torch.dtype,
) -> dict[str, torch.Tensor | int]:
    device_index = box_device.index if box_device.index is not None else torch.cuda.current_device()
    key = (device_index, box_dtype, score_dtype)
    workspace = _duplicate_merge_cuda_workspace.get(key)
    if workspace is not None and int(workspace["capacity"]) >= num_boxes:
        return workspace

    capacity = max(num_boxes, 1)
    workspace = {
        "capacity": capacity,
        "anchor_indices": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "box_sums": torch.empty((capacity, 4), device=box_device, dtype=torch.float32),
        "score_sums": torch.empty((capacity,), device=box_device, dtype=torch.float32),
        "score_bits_max": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "cluster_counts": torch.empty((capacity,), device=box_device, dtype=torch.int32),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0 or boxes.size(0) <= 1:
        return boxes, scores, classes

    order = torch.argsort(scores, descending=True)
    remaining = order.tolist()
    merged_boxes = []
    merged_scores = []
    merged_classes = []

    while remaining:
        anchor_idx = remaining[0]
        anchor_box = boxes[anchor_idx]
        anchor_class = classes[anchor_idx]

        candidate_indices = torch.tensor(remaining, device=boxes.device, dtype=torch.long)
        candidate_boxes = boxes[candidate_indices]
        candidate_scores = scores[candidate_indices]
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
            torch.tensor(anchor_area, device=boxes.device) / candidate_areas.clamp(min=1e-6),
        )

        duplicate_mask = same_class & (
            (ious >= iou_threshold)
            | ((center_dist <= center_gate) & (area_ratio >= area_ratio_threshold))
        )

        cluster_indices = candidate_indices[duplicate_mask]
        cluster_boxes = boxes[cluster_indices]
        cluster_scores = scores[cluster_indices]
        weights = cluster_scores / cluster_scores.sum().clamp(min=1e-6)
        fused_box = (cluster_boxes * weights.unsqueeze(1)).sum(dim=0)
        fused_score = cluster_scores.max()

        merged_boxes.append(fused_box)
        merged_scores.append(fused_score)
        merged_classes.append(anchor_class)

        cluster_set = set(int(idx) for idx in cluster_indices.tolist())
        remaining = [idx for idx in remaining if idx not in cluster_set]

    return (
        torch.stack(merged_boxes, dim=0),
        torch.stack(merged_scores, dim=0),
        torch.stack(merged_classes, dim=0),
    )


def merge_cross_tile_duplicates_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    center_threshold: float = 0.18,
    area_ratio_threshold: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0 or boxes.size(0) <= 1:
        return merge_cross_tile_duplicates(
            boxes, scores, classes, iou_threshold, center_threshold, area_ratio_threshold
        )

    boxes = boxes.contiguous()
    scores = scores.contiguous()
    classes_i32 = classes.to(torch.int32).contiguous()

    if cpp_merge_cross_tile_duplicates_cuda is not None and boxes.is_cuda:
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
            workspace["cluster_counts"].data_ptr(),
            workspace["out_boxes"].data_ptr(),
            workspace["out_scores"].data_ptr(),
            workspace["out_classes"].data_ptr(),
            workspace["out_count"].data_ptr(),
            iou_threshold,
            center_threshold,
            area_ratio_threshold,
            torch.cuda.current_stream().cuda_stream,
        )
        merged_count = int(workspace["out_count"].item())
        return (
            workspace["out_boxes"][:merged_count],
            workspace["out_scores"][:merged_count],
            workspace["out_classes"][:merged_count],
        )

    if cpp_merge_cross_tile_duplicates is not None:
        device = boxes.device
        boxes_np = boxes.detach().to("cpu").numpy()
        scores_np = scores.detach().to("cpu").numpy()
        classes_np = classes_i32.detach().to("cpu").numpy()
        merged_boxes_np, merged_scores_np, merged_classes_np = cpp_merge_cross_tile_duplicates(
            boxes_np,
            scores_np,
            classes_np,
            iou_threshold,
            center_threshold,
            area_ratio_threshold,
        )
        merged_boxes = torch.from_numpy(np.asarray(merged_boxes_np)).to(device=device, dtype=boxes.dtype)
        merged_scores = torch.from_numpy(np.asarray(merged_scores_np)).to(device=device, dtype=scores.dtype)
        merged_classes = torch.from_numpy(np.asarray(merged_classes_np)).to(device=device, dtype=classes_i32.dtype)
        return merged_boxes, merged_scores, merged_classes

    return merge_cross_tile_duplicates(
        boxes, scores, classes_i32, iou_threshold, center_threshold, area_ratio_threshold
    )


def detect_single_patch_640(detector, pool, h_orig, w_orig, preprocess_modes):
    if "letterbox" in preprocess_modes:
        r = 640.0 / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * r), int(w_orig * r)
        img_resized = torch.nn.functional.interpolate(
            pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
        ).squeeze(0)
        pool.canvas_640p.fill_(114.0 / 255.0)
        y_off = (640 - h_new) // 2
        x_off = (640 - w_new) // 2
        pool.canvas_640p[:, y_off:y_off + h_new, x_off:x_off + w_new].copy_(img_resized)

        raw_dets = detector.detect_raw(pool.canvas_640p.unsqueeze(0))
        boxes = raw_dets[0, :, :4]
        scores = raw_dets[0, :, 4]
        classes = raw_dets[0, :, 5]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_off) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_off) / r
        return boxes, scores, classes

    img_input = torch.nn.functional.interpolate(pool.frame_buffer.unsqueeze(0), size=(640, 640))
    raw_dets = detector.detect_raw(img_input)
    boxes = raw_dets[0, :, :4]
    scores = raw_dets[0, :, 4]
    classes = raw_dets[0, :, 5]

    boxes[:, [0, 2]] /= (640.0 / w_orig)
    boxes[:, [1, 3]] /= (640.0 / h_orig)
    return boxes, scores, classes


def detect_960p_3x2_tiled(detector, pool, h_orig, w_orig, preprocess_modes):
    """3×2 tiling on 960p canvas with overlap. Two batch-4 passes (6 tiles).
    x: 3 cols at stride=160 on 960p (75% x-overlap, same scale as 2×2).
    y: 2 rows at stride=320 on 960p (50% y-overlap, same as 2×2).
    """
    if w_orig <= 960 and h_orig <= 960:
        boxes, scores, classes = detect_single_patch_640(detector, pool, h_orig, w_orig, preprocess_modes)
        return boxes, scores, classes, False

    r = 960.0 / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    img_resized = torch.nn.functional.interpolate(
        pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
    ).squeeze(0)
    pool.canvas_960p.fill_(114.0 / 255.0)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2
    pool.canvas_960p[:, y_off:y_off + h_new, x_off:x_off + w_new].copy_(img_resized)

    # Batch 6: 6 tiles in one pass
    pool.tiles_batch6[0].copy_(pool.canvas_960p[:, 0:640,   0:640])
    pool.tiles_batch6[1].copy_(pool.canvas_960p[:, 0:640, 160:800])
    pool.tiles_batch6[2].copy_(pool.canvas_960p[:, 0:640, 320:960])
    pool.tiles_batch6[3].copy_(pool.canvas_960p[:, 320:960,  0:640])
    pool.tiles_batch6[4].copy_(pool.canvas_960p[:, 320:960, 160:800])
    pool.tiles_batch6[5].copy_(pool.canvas_960p[:, 320:960, 320:960])

    raw = detector.detect_raw(pool.tiles_batch6)

    boxes = raw[:, :, :4].clone()
    boxes[:, :, [0, 2]] = (boxes[:, :, [0, 2]] + pool.tile_3x2_dx - x_off) / r
    boxes[:, :, [1, 3]] = (boxes[:, :, [1, 3]] + pool.tile_3x2_dy - y_off) / r

    all_boxes = boxes.reshape(-1, 4)
    all_scores = raw[:, :, 4].reshape(-1)
    all_classes = raw[:, :, 5].reshape(-1)
    return all_boxes, all_scores, all_classes, True


def detect_adaptive_960_tiled(detector, pool, h_orig, w_orig, preprocess_modes):
    if w_orig <= 960 and h_orig <= 960:
        boxes, scores, classes = detect_single_patch_640(
            detector, pool, h_orig, w_orig, preprocess_modes
        )
        return boxes, scores, classes, False

    r = 960.0 / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    img_resized = torch.nn.functional.interpolate(
        pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
    ).squeeze(0)

    pool.canvas_960p.fill_(114.0 / 255.0)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2
    pool.canvas_960p[:, y_off:y_off + h_new, x_off:x_off + w_new].copy_(img_resized)

    pool.tiles_batch4[0].copy_(pool.canvas_960p[:, 0:640, 0:640])
    pool.tiles_batch4[1].copy_(pool.canvas_960p[:, 0:640, 320:960])
    pool.tiles_batch4[2].copy_(pool.canvas_960p[:, 320:960, 0:640])
    pool.tiles_batch4[3].copy_(pool.canvas_960p[:, 320:960, 320:960])

    raw_dets = detector.detect_raw(pool.tiles_batch4)
    # One clone for all tiles, two broadcast ops using pre-allocated offsets
    # instead of 4 individual clones + 8 per-tile scalar ops + 3 cat calls.
    all_boxes = raw_dets[:, :, :4].clone()  # [4, 300, 4]
    all_boxes[:, :, [0, 2]] = (all_boxes[:, :, [0, 2]] + pool.tile_dx - x_off) / r
    all_boxes[:, :, [1, 3]] = (all_boxes[:, :, [1, 3]] + pool.tile_dy - y_off) / r
    # Scores and classes are non-contiguous views; reshape() copies them automatically.
    return all_boxes.reshape(-1, 4), raw_dets[:, :, 4].reshape(-1), raw_dets[:, :, 5].reshape(-1), True
