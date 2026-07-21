"""Unit tests for the cross-frame consistency-protection term (route 3).

Covers ``_temporal_consistency_loss`` from ``train_mamba_gt`` — the explicit
classification-score consistency penalty that protects the T3→T1 shaping during
full-grad training (mamba-t3t1-curriculum-20260613.md §5).

The training module imports the C++ tracking extension at top level, which is
unavailable on the no-GPU CI runner, so the import is guarded.
"""

# scope: detection
# function: behavior
# lifecycle: active

from __future__ import annotations

import pytest
import torch

train_mamba_gt = pytest.importorskip(
    "scripts.train.temporal_yolo.train_mamba_gt",
    reason="train_mamba_gt requires the saccade_tracking_ext C++ extension",
    exc_type=ImportError,
)

_consistency_loss = train_mamba_gt._temporal_consistency_loss


def _cls_maps(B: int, T: int, hw: int = 16, nc: int = 1) -> list[torch.Tensor]:
    """Batch-major (B*T, nc, H, W) P3 cls map plus two coarser dummy scales."""
    p3 = torch.zeros(B * T, nc, hw, hw, requires_grad=True)
    p4 = torch.zeros(B * T, nc, hw // 2, hw // 2)
    p5 = torch.zeros(B * T, nc, hw // 4, hw // 4)
    return [p3, p4, p5]


def test_no_matched_tracks_returns_zero() -> None:
    # Disjoint ids across frames -> nothing to penalise.
    s_cls_T = _cls_maps(B=1, T=2)
    gt_boxes = [[torch.tensor([[10.0, 10.0, 30.0, 30.0]])] * 2]
    gt_ids = [[torch.tensor([1]), torch.tensor([2])]]
    loss = _consistency_loss(s_cls_T, gt_boxes, gt_ids, T=2, B=1, img_size=128)
    assert loss.item() == 0.0


def test_empty_frame_returns_zero() -> None:
    s_cls_T = _cls_maps(B=1, T=2)
    gt_boxes = [[torch.zeros(0, 4), torch.tensor([[10.0, 10.0, 30.0, 30.0]])]]
    gt_ids = [[torch.zeros(0, dtype=torch.int64), torch.tensor([1])]]
    loss = _consistency_loss(s_cls_T, gt_boxes, gt_ids, T=2, B=1, img_size=128)
    assert loss.item() == 0.0


def test_identical_logits_zero_loss() -> None:
    # Uniform-zero map: a track sampled in both frames has identical logits.
    s_cls_T = _cls_maps(B=1, T=2)
    box = torch.tensor([[40.0, 40.0, 80.0, 80.0]])
    gt_boxes = [[box, box]]
    gt_ids = [[torch.tensor([7]), torch.tensor([7])]]
    loss = _consistency_loss(s_cls_T, gt_boxes, gt_ids, T=2, B=1, img_size=128)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_diverging_logits_penalised_and_differentiable() -> None:
    # Frame 1's map is high at the track centre, frame 0's is zero -> nonzero
    # penalty with gradient flowing back to the logit maps.
    s_cls_T = _cls_maps(B=1, T=2, hw=16)
    p3 = s_cls_T[0]
    img_size = 128
    box = torch.tensor([[56.0, 56.0, 72.0, 72.0]])  # centre (64, 64) -> P3 cell 8
    with torch.no_grad():
        p3[1, 0, 8, 8] = 5.0  # frame t=1 (row b*T+t = 1) hot at centre
    gt_boxes = [[box, box]]
    gt_ids = [[torch.tensor([3]), torch.tensor([3])]]
    loss = _consistency_loss(s_cls_T, gt_boxes, gt_ids, T=2, B=1, img_size=img_size)
    assert loss.item() > 0.0
    loss.backward()
    assert p3.grad is not None
    assert torch.isfinite(p3.grad).all()
    # Gradient must touch both frames (symmetric L2 penalty).
    assert p3.grad[0].abs().sum() > 0.0
    assert p3.grad[1].abs().sum() > 0.0


def test_motion_compensated_by_box_following() -> None:
    # Same track moves across frames; each frame's map is hot at that frame's
    # own centre. Sampling at each frame's GT centre -> consistent -> ~0 loss,
    # confirming motion compensation (no flow needed).
    s_cls_T = _cls_maps(B=1, T=2, hw=16)
    p3 = s_cls_T[0]
    img_size = 128
    box0 = torch.tensor([[24.0, 24.0, 40.0, 40.0]])  # centre (32,32) -> cell 4
    box1 = torch.tensor([[88.0, 88.0, 104.0, 104.0]])  # centre (96,96) -> cell 12
    with torch.no_grad():
        p3[0, 0, 4, 4] = 5.0
        p3[1, 0, 12, 12] = 5.0
    gt_boxes = [[box0, box1]]
    gt_ids = [[torch.tensor([9]), torch.tensor([9])]]
    loss = _consistency_loss(s_cls_T, gt_boxes, gt_ids, T=2, B=1, img_size=img_size)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)
