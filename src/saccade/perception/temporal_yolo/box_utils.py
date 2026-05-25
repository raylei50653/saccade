"""
Shared box conversion utilities for temporal YOLO training scripts.

Import:
    from saccade.perception.temporal_yolo.box_utils import (
        xyxy_to_cxcywh_norm,
        make_yolo_batch,
    )
"""

from __future__ import annotations

import random

import torch


def xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert (N, 4) xyxy absolute to (N, 4) cxcywh normalized [0, 1]."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=1)


def make_yolo_batch(
    gt_boxes_list: list[torch.Tensor],
    img_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Assemble batch dict for Ultralytics v8DetectionLoss.

    Returns:
        {"batch_idx": (N,), "cls": (N,), "bboxes": (N, 4)}
    """
    batch_idxs: list[torch.Tensor] = []
    clss: list[torch.Tensor] = []
    bboxes: list[torch.Tensor] = []
    for b, boxes in enumerate(gt_boxes_list):
        if boxes.numel() == 0:
            continue
        n = boxes.shape[0]
        batch_idxs.append(torch.full((n,), float(b)))
        clss.append(torch.zeros(n))
        bboxes.append(xyxy_to_cxcywh_norm(boxes, img_size))
    if not batch_idxs:
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros(0, device=device),
            "bboxes": torch.zeros(0, 4, device=device),
        }
    return {
        "batch_idx": torch.cat(batch_idxs).to(device),
        "cls": torch.cat(clss).to(device),
        "bboxes": torch.cat(bboxes).to(device),
    }


def build_gate_inputs(
    prev_gt_boxes: list[torch.Tensor],
    gt_ratio: float,
    img_size: int,
    device: torch.device,
) -> list | None:
    """Build TrackerGateInput list for training with probability gt_ratio."""
    from saccade.perception.temporal_yolo.yolo_conditioned import (
        TrackerGateInput,
    )

    if random.random() >= gt_ratio:
        return None
    return [
        TrackerGateInput.from_boxes_scores(
            b.to(device), None, (img_size, img_size), assume_absolute=True
        ).to(device)
        for b in prev_gt_boxes
    ]
