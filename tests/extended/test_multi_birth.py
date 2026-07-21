"""Tests for saccade.perception.eval.multi_birth (P5-1 multi-signal birth)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import pytest
import torch

from saccade.perception.eval.multi_birth import (
    MultiSignalBirthManager,
    _Candidate,
    _box_iou_nm,
)


# ── _box_iou_nm ──────────────────────────────────────────────────────────


def test_box_iou_nm_self() -> None:
    boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    iou = _box_iou_nm(boxes, boxes)
    assert iou[0, 0] == pytest.approx(1.0)


def test_box_iou_nm_disjoint() -> None:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    iou = _box_iou_nm(a, b)
    assert iou[0, 0] == pytest.approx(0.0)


def test_box_iou_nm_partial_overlap() -> None:
    """a=(0,0)-(10,10), b=(5,0)-(15,10). Overlap: ix=[5,10], iy=[0,10] => 50/150."""
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    iou = _box_iou_nm(a, b)
    # Intersection: 5*10=50, Union: 100+100-50=150 => IoU=1/3
    assert iou[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_box_iou_nm_batch() -> None:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    iou = _box_iou_nm(a, b)
    assert iou.shape == (2, 2)
    assert iou[1, 1] == pytest.approx(1.0)  # identical boxes


# ── _geometry_score ──────────────────────────────────────────────────────


def test_geometry_score_ideal_aspect() -> None:
    mgr = MultiSignalBirthManager()
    # aspect = 3.0 (ideal human silhouette range 2-4)
    box = torch.tensor([0.0, 0.0, 100.0, 300.0])  # w=100, h=300, aspect=3.0
    q = mgr._geometry_score(box)
    assert q == pytest.approx(1.0)


def test_geometry_score_low_aspect_reject() -> None:
    mgr = MultiSignalBirthManager()
    box = torch.tensor([0.0, 0.0, 100.0, 50.0])  # aspect = 0.5
    q = mgr._geometry_score(box)
    assert q == pytest.approx(0.0)


def test_geometry_score_extreme_aspect_penalty() -> None:
    mgr = MultiSignalBirthManager()
    box = torch.tensor([0.0, 0.0, 100.0, 700.0])  # aspect = 7.0
    q = mgr._geometry_score(box)
    assert q == pytest.approx(0.0)  # penalized to 0 at aspect=7


def test_geometry_score_hard_reject_min_aspect() -> None:
    mgr = MultiSignalBirthManager(min_aspect=1.5)
    box = torch.tensor([0.0, 0.0, 100.0, 100.0])  # aspect = 1.0
    q = mgr._geometry_score(box)
    assert q == pytest.approx(-1.0)


def test_geometry_score_hard_reject_max_area() -> None:
    mgr = MultiSignalBirthManager(max_area_px=5000)
    box = torch.tensor([0.0, 0.0, 200.0, 200.0])  # area = 40000
    q = mgr._geometry_score(box)
    assert q == pytest.approx(-1.0)


def test_geometry_score_aspect_1_0() -> None:
    mgr = MultiSignalBirthManager()
    box = torch.tensor([0.0, 0.0, 100.0, 200.0])  # aspect = 2.0 (transition)
    q = mgr._geometry_score(box)
    assert q == pytest.approx(1.0)


def test_geometry_score_aspect_between_1_and_2() -> None:
    mgr = MultiSignalBirthManager()
    box = torch.tensor([0.0, 0.0, 100.0, 150.0])  # aspect = 1.5
    q = mgr._geometry_score(box)
    assert q == pytest.approx(0.5)  # q = aspect - 1.0 = 0.5


# ── _compute_evidence ───────────────────────────────────────────────────


def test_compute_evidence_insufficient_frames() -> None:
    mgr = MultiSignalBirthManager(min_frames=3)
    cand = _Candidate(
        history=[(1, torch.tensor([0.0, 0.0, 50.0, 150.0]), 0.3)],
        last_frame=1,
    )
    ev = mgr._compute_evidence(cand)
    assert ev == pytest.approx(0.0)


def test_compute_evidence_min_frames_reached() -> None:
    mgr = MultiSignalBirthManager(min_frames=3, target_motion_px=10.0)
    box1 = torch.tensor([0.0, 0.0, 50.0, 150.0])  # w=50, h=150, aspect=3.0
    box2 = torch.tensor([10.0, 0.0, 60.0, 150.0])
    box3 = torch.tensor([20.0, 0.0, 70.0, 150.0])
    cand = _Candidate(
        history=[
            (1, box1, 0.3),
            (2, box2, 0.4),
            (3, box3, 0.5),
        ],
        last_frame=3,
    )
    ev = mgr._compute_evidence(cand)
    # streak = (3-1)/(3-1) = 1.0
    # score: best=0.5, range=0.35-0.12=0.23 => norm = min(1, (0.5-0.12)/0.23) = 1.0
    # geometry: aspect=3.0 => q=1.0
    # motion: centroid moved 10px/frame => norm = 10/10 = 1.0
    # E = 0.35*1.0 + 0.30*1.0 + 0.20*1.0 + 0.15*1.0 = 1.0
    assert ev == pytest.approx(1.0)


def test_compute_evidence_hard_reject_geometry() -> None:
    mgr = MultiSignalBirthManager(min_frames=3, min_aspect=1.5)
    box = torch.tensor([0.0, 0.0, 100.0, 80.0])  # aspect = 0.8 < 1.5
    cand = _Candidate(
        history=[
            (1, box, 0.3),
            (2, box, 0.4),
            (3, box, 0.5),
        ],
        last_frame=3,
    )
    ev = mgr._compute_evidence(cand)
    assert ev == pytest.approx(0.0)


# ── reset ────────────────────────────────────────────────────────────────


def test_reset_clears_candidates() -> None:
    mgr = MultiSignalBirthManager()
    mgr._candidates = [
        _Candidate(
            history=[(1, torch.tensor([0.0, 0.0, 50.0, 150.0]), 0.3)], last_frame=1
        )
    ]
    mgr.reset()
    assert len(mgr._candidates) == 0
    assert mgr._promoted_ids == set()
    assert mgr._replace_indices == set()


# ── update: promote ─────────────────────────────────────────────────────


def test_update_empty_input() -> None:
    mgr = MultiSignalBirthManager()
    promote, replace = mgr.update(1, torch.empty((0, 4)), torch.empty((0,)))
    assert promote.shape == (0,)
    assert replace.shape == (0,)
    assert len(mgr._candidates) == 0


def test_update_low_score_below_min() -> None:
    mgr = MultiSignalBirthManager(min_score=0.2)
    boxes = torch.tensor([[0.0, 0.0, 50.0, 150.0]])
    scores = torch.tensor([0.1])  # below min_score
    promote, replace = mgr.update(1, boxes, scores)
    # Creates candidate but doesn't promote (score < min_score)
    assert not promote[0]


def test_update_promote_after_min_frames() -> None:
    mgr = MultiSignalBirthManager(
        min_score=0.1,
        min_frames=3,
        evidence_threshold=0.5,
        replace_mode=False,
        target_motion_px=5.0,
    )
    mgr.reset()
    # Frame 1: below threshold but above min_score
    promote1, _ = mgr.update(
        1, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.2])
    )
    assert not promote1[0]
    # Frame 2
    promote2, _ = mgr.update(
        2, torch.tensor([[5.0, 0.0, 55.0, 150.0]]), torch.tensor([0.3])
    )
    assert not promote2[0]  # only 2 frames
    # Frame 3: evidence should be enough
    promote3, _ = mgr.update(
        3, torch.tensor([[10.0, 0.0, 60.0, 150.0]]), torch.tensor([0.4])
    )
    assert promote3[0]  # promoted!


def test_update_promote_removes_candidate() -> None:
    mgr = MultiSignalBirthManager(
        min_score=0.1,
        min_frames=2,
        evidence_threshold=0.3,
        target_motion_px=5.0,
    )
    mgr.reset()
    mgr.update(1, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.3]))
    mgr.update(2, torch.tensor([[5.0, 0.0, 55.0, 150.0]]), torch.tensor([0.4]))
    assert len(mgr._candidates) == 0  # removed after promotion


def test_update_new_candidate_added() -> None:
    mgr = MultiSignalBirthManager()
    mgr.reset()
    mgr.update(1, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.3]))
    assert len(mgr._candidates) == 1


def test_update_no_iou_match_starts_new_candidate() -> None:
    mgr = MultiSignalBirthManager(iou_match=0.3)
    mgr.reset()
    # Frame 1
    mgr.update(1, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.3]))
    # Frame 2: far away, won't match IoU
    mgr.update(2, torch.tensor([[200.0, 200.0, 250.0, 350.0]]), torch.tensor([0.4]))
    assert len(mgr._candidates) == 2  # two separate candidates


def test_update_replace_mode_high_evidence() -> None:
    mgr = MultiSignalBirthManager(
        min_score=0.1,
        min_frames=3,
        evidence_threshold=0.5,
        replace_evidence_threshold=0.85,
        replace_mode=True,
        target_motion_px=5.0,
    )
    mgr.reset()
    promote1, replace1 = mgr.update(
        1, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.3])
    )
    assert not promote1[0]
    assert not replace1[0]
    promote2, replace2 = mgr.update(
        2, torch.tensor([[5.0, 0.0, 55.0, 150.0]]), torch.tensor([0.4])
    )
    assert not promote2[0]
    assert not replace2[0]
    promote3, replace3 = mgr.update(
        3, torch.tensor([[10.0, 0.0, 60.0, 150.0]]), torch.tensor([0.6])
    )
    assert promote3[0]  # promoted (evidence >= 0.5)


def test_update_ttl_expiration() -> None:
    mgr = MultiSignalBirthManager(ttl_frames=2)
    mgr.reset()
    mgr.update(1, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.3]))
    assert len(mgr._candidates) == 1
    mgr.update(5, torch.tensor([[0.0, 0.0, 50.0, 150.0]]), torch.tensor([0.3]))
    # Frame 5 - candidate from frame 1 is expired (5 - 1 = 4 > 2)
    assert len(mgr._candidates) == 1  # only the new one


def test_update_multiple_detections_same_frame() -> None:
    mgr = MultiSignalBirthManager()
    mgr.reset()
    boxes = torch.tensor(
        [
            [0.0, 0.0, 50.0, 150.0],
            [100.0, 100.0, 150.0, 250.0],
        ]
    )
    scores = torch.tensor([0.3, 0.4])
    promote, _ = mgr.update(1, boxes, scores)
    # Both should be new candidates (IoU match = 0 for both)
    assert len(mgr._candidates) == 2


def test_update_device_preservation() -> None:
    """promote/replace masks should be on the same device as input."""
    mgr = MultiSignalBirthManager()
    mgr.reset()
    boxes = torch.zeros(3, 4, dtype=torch.float32)
    scores = torch.zeros(3)
    promote, replace = mgr.update(1, boxes, scores)
    assert promote.device == boxes.device
    assert replace.device == boxes.device
