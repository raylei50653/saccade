"""Shared box geometry helpers for Python fallback paths."""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch

BoxLike = Sequence[float] | torch.Tensor
UnionMode = Literal["clamp", "add", "zero"]


def _coord(box: BoxLike, idx: int) -> float:
    return float(box[idx])


def box_iou(a: BoxLike, b: BoxLike, union_mode: UnionMode = "clamp") -> float:
    """Compute IoU for [x1, y1, x2, y2] boxes.

    union_mode preserves existing fallback semantics:
    - clamp: inter / max(union, 1e-6)
    - add:   inter / (union + 1e-6)
    - zero:  0.0 when union < 1e-6, otherwise inter / union
    """
    ax1, ay1, ax2, ay2 = (_coord(a, i) for i in range(4))
    bx1, by1, bx2, by2 = (_coord(b, i) for i in range(4))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union_mode == "add":
        return float(inter / (union + 1e-6))
    if union_mode == "zero" and union < 1e-6:
        return 0.0
    return float(inter / max(union, 1e-6))


def center_distance_norm(a: BoxLike, b: BoxLike, frame_w: int, frame_h: int) -> float:
    ax1, ay1, ax2, ay2 = (_coord(a, i) for i in range(4))
    bx1, by1, bx2, by2 = (_coord(b, i) for i in range(4))
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / max(
        float(frame_w), float(frame_h), 1.0
    )


def center_shift_ratio(a: BoxLike, b: BoxLike) -> float:
    ax1, ay1, ax2, ay2 = (_coord(a, i) for i in range(4))
    bx1, by1, bx2, by2 = (_coord(b, i) for i in range(4))
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    aw = max(ax2 - ax1, 1e-6)
    ah = max(ay2 - ay1, 1e-6)
    bw = max(bx2 - bx1, 1e-6)
    bh = max(by2 - by1, 1e-6)
    scale = max((math.sqrt(aw * ah) + math.sqrt(bw * bh)) * 0.5, 1e-6)
    return float(math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / scale)


def spatial_metrics(
    box: BoxLike,
    old_box: BoxLike,
    frame_w: int,
    frame_h: int,
    union_mode: UnionMode = "clamp",
) -> tuple[float, float]:
    return (
        center_distance_norm(box, old_box, frame_w, frame_h),
        box_iou(box, old_box, union_mode=union_mode),
    )


def torch_box_iou_single(
    box: torch.Tensor,
    boxes: torch.Tensor,
    union_mode: UnionMode = "add",
) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    if union_mode == "add":
        return inter / (union + 1e-6)
    if union_mode == "zero":
        return torch.where(
            union < 1e-6, torch.zeros_like(inter), inter / union.clamp(min=1e-6)
        )
    return inter / union.clamp(min=1e-6)


def torch_box_iou_pairwise_diag(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    union_mode: UnionMode = "add",
) -> torch.Tensor:
    lt = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    rb = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    if union_mode == "add":
        return inter / (union + 1e-6)
    if union_mode == "zero":
        return torch.where(
            union < 1e-6, torch.zeros_like(inter), inter / union.clamp(min=1e-6)
        )
    return inter / union.clamp(min=1e-6)


def torch_box_iou_matrix(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    union_mode: UnionMode = "clamp",
    area_min: float = 0.0,
) -> torch.Tensor:
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[
        :, None
    ]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[
        None, :
    ]
    if area_min > 0.0:
        area_a = area_a.clamp(min=area_min)
        area_b = area_b.clamp(min=area_min)
    union = area_a + area_b - inter
    if union_mode == "add":
        return inter / (union + 1e-6)
    if union_mode == "zero":
        return torch.where(
            union < 1e-6, torch.zeros_like(inter), inter / union.clamp(min=1e-6)
        )
    return inter / union.clamp(min=1e-6)
