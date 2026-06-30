"""Unit tests for keyframe GT linear interpolation (_interpolate_gt_for_frame).

For keyframe-annotated benchmarks (e.g. PersonPath22, gt every ~5th frame) the
tracker/model can run at the real frame cadence; the in-between frames get a
linearly-interpolated GT box per track so temporal continuity is preserved,
while the loss is masked to real keyframes (is_keyframe). These tests exercise
the interpolation math directly without any dataset on disk.
"""

import torch

from saccade.perception.temporal_yolo.dataset import _interpolate_gt_for_frame


def _gt():
    # Two keyframes at frames 1 and 6 (5-frame keyframe spacing).
    # Track 1 present in both; track 2 only at frame 1 (dies); track 3 only at 6 (born).
    return {
        1: (
            torch.tensor([[0.0, 0, 10, 10], [100.0, 100, 110, 110]]),
            torch.tensor([1, 2]),
        ),
        6: (
            torch.tensor([[10.0, 20, 20, 30], [200.0, 200, 210, 210]]),
            torch.tensor([1, 3]),
        ),
    }


def test_linear_midpoint():
    gt = _gt()
    keyframes = [1, 6]
    # frame 1 + 0.6*(6-1) = frame... use frame 4: w = (4-1)/(6-1) = 0.6
    boxes, ids = _interpolate_gt_for_frame(4, keyframes, gt)
    # only track 1 is common to both keyframes
    assert ids.tolist() == [1]
    expected = torch.tensor([[0.0, 0, 10, 10]]) + 0.6 * (
        torch.tensor([[10.0, 20, 20, 30]]) - torch.tensor([[0.0, 0, 10, 10]])
    )
    assert torch.allclose(boxes, expected)


def test_excludes_birth_and_death():
    # track 2 (only frame 1) and track 3 (only frame 6) must never be interpolated
    gt = _gt()
    boxes, ids = _interpolate_gt_for_frame(3, [1, 6], gt)
    assert 2 not in ids.tolist()
    assert 3 not in ids.tolist()


def test_no_extrapolation_outside_span():
    gt = _gt()
    # before first / after last keyframe → empty (no extrapolation)
    for f in (0, 7, 99):
        boxes, ids = _interpolate_gt_for_frame(f, [1, 6], gt)
        assert boxes.shape == (0, 4)
        assert ids.numel() == 0


def test_uses_adjacent_brackets():
    # three keyframes; a frame between 6 and 11 must interpolate from (6, 11) not (1, 11)
    gt = _gt()
    gt[11] = (torch.tensor([[30.0, 40, 40, 50]]), torch.tensor([1]))
    boxes, ids = _interpolate_gt_for_frame(8, [1, 6, 11], gt)
    assert ids.tolist() == [1]
    w = (8 - 6) / (11 - 6)
    expected = torch.tensor([[10.0, 20, 20, 30]]) + w * (
        torch.tensor([[30.0, 40, 40, 50]]) - torch.tensor([[10.0, 20, 20, 30]])
    )
    assert torch.allclose(boxes, expected)


def test_empty_keyframes():
    boxes, ids = _interpolate_gt_for_frame(4, [], {})
    assert boxes.shape == (0, 4)
    assert ids.numel() == 0
