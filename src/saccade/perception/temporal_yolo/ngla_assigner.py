"""Normalized-Gaussian label assignment utilities for YOLO losses.

This keeps Ultralytics' TaskAlignedAssigner flow intact and swaps only the
localization metric from CIoU to a normalized Bhattacharyya-distance similarity.
"""

from __future__ import annotations

import torch
from ultralytics.utils.tal import TaskAlignedAssigner


def gaussian_box_nbcd(
    gt_bboxes: torch.Tensor,
    pd_bboxes: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Return normalized Bhattacharyya-distance similarity for xyxy boxes.

    Boxes are modelled as axis-aligned 2D Gaussians whose center is the box
    center and whose diagonal covariance uses std=(w/2, h/2). Identical boxes
    produce 1.0; larger center/scale mismatch decays smoothly toward 0.
    """

    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bboxes.unbind(-1)
    pd_x1, pd_y1, pd_x2, pd_y2 = pd_bboxes.unbind(-1)

    gt_cx = (gt_x1 + gt_x2) * 0.5
    gt_cy = (gt_y1 + gt_y2) * 0.5
    pd_cx = (pd_x1 + pd_x2) * 0.5
    pd_cy = (pd_y1 + pd_y2) * 0.5

    gt_w = (gt_x2 - gt_x1).clamp_min(eps)
    gt_h = (gt_y2 - gt_y1).clamp_min(eps)
    pd_w = (pd_x2 - pd_x1).clamp_min(eps)
    pd_h = (pd_y2 - pd_y1).clamp_min(eps)

    gt_vx = (gt_w * 0.5).square().clamp_min(eps)
    gt_vy = (gt_h * 0.5).square().clamp_min(eps)
    pd_vx = (pd_w * 0.5).square().clamp_min(eps)
    pd_vy = (pd_h * 0.5).square().clamp_min(eps)

    dx = gt_cx - pd_cx
    dy = gt_cy - pd_cy

    mean_term = 0.25 * (dx.square() / (gt_vx + pd_vx) + dy.square() / (gt_vy + pd_vy))
    x_cov_term = 0.5 * torch.log(
        (gt_vx + pd_vx) / (2.0 * torch.sqrt(gt_vx * pd_vx) + eps) + eps
    )
    y_cov_term = 0.5 * torch.log(
        (gt_vy + pd_vy) / (2.0 * torch.sqrt(gt_vy * pd_vy) + eps) + eps
    )
    distance = (mean_term + x_cov_term + y_cov_term).clamp_min(0)
    return (1.0 / (1.0 + distance)).clamp_(0.0, 1.0)


class NGLAAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner variant using Gaussian NBCD as localization score."""

    def iou_calculation(
        self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor
    ) -> torch.Tensor:
        return gaussian_box_nbcd(gt_bboxes, pd_bboxes, eps=self.eps).squeeze(-1)


def install_ngla_assigner(criterion: object) -> None:
    """Replace ``criterion.assigner`` with an NGLA-compatible assigner in-place."""

    old = criterion.assigner  # type: ignore[attr-defined]
    criterion.assigner = NGLAAssigner(  # type: ignore[attr-defined]
        topk=old.topk,
        num_classes=old.num_classes,
        alpha=old.alpha,
        beta=old.beta,
        stride=old.stride,
        eps=old.eps,
        topk2=getattr(old, "topk2", None),
    )


def install_assigner(criterion: object, name: str) -> None:
    """Install a supported assigner by name."""

    if name == "tal":
        return
    if name == "ngla":
        install_ngla_assigner(criterion)
        return
    raise ValueError(f"unknown assigner: {name}")
