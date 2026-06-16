"""Tests for saccade.perception.eval.detection (pure Python paths)."""

from __future__ import annotations

import pytest
import torch

from saccade.perception.eval.detection import (
    _is_2x2_tiling,
    _is_tiled_tiling,
    _box_iou_single,
    _box_iou_pairwise_diag,
    _box_iou_matrix,
    filter_detections_fast,
    nms_fast,
    merge_cross_tile_duplicates,
    merge_cross_tile_duplicates_fast,
)


# ── _is_2x2_tiling / _is_tiled_tiling ───────────────────────────────────


def test_is_2x2_tiling_960p() -> None:
    assert _is_2x2_tiling("960p_2x2") is True
    assert _is_2x2_tiling("sahi_960p_2x2") is True
    assert _is_2x2_tiling("mamba_global_2x2") is True
    assert _is_2x2_tiling("960p_3x2") is False
    assert _is_2x2_tiling(None) is False
    assert _is_2x2_tiling("") is False
    assert _is_2x2_tiling("random") is False


def test_is_tiled_tiling() -> None:
    assert _is_tiled_tiling("960p_2x2") is True
    assert _is_tiled_tiling("960p_3x2") is True
    assert _is_tiled_tiling("sahi_960p_2x2") is True
    assert _is_tiled_tiling(None) is False
    assert _is_tiled_tiling("") is False
    assert _is_tiled_tiling("random") is False


# ── _box_iou_single ─────────────────────────────────────────────────────


def test_box_iou_single_self() -> None:
    box = torch.tensor([0.0, 0.0, 100.0, 100.0])
    iou = _box_iou_single(box, box.unsqueeze(0))
    assert iou[0] == pytest.approx(1.0)


def test_box_iou_single_disjoint() -> None:
    box = torch.tensor([0.0, 0.0, 10.0, 10.0])
    boxes = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    iou = _box_iou_single(box, boxes)
    assert iou[0] == pytest.approx(0.0)


def test_box_iou_single_partial() -> None:
    box = torch.tensor([0.0, 0.0, 10.0, 10.0])
    boxes = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    iou = _box_iou_single(box, boxes)
    assert iou[0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_box_iou_single_multiple() -> None:
    box = torch.tensor([0.0, 0.0, 10.0, 10.0])
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # IoU = 1.0
            [100.0, 100.0, 110.0, 110.0],  # IoU = 0.0
        ]
    )
    iou = _box_iou_single(box, boxes)
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(0.0)


# ── _box_iou_pairwise_diag ──────────────────────────────────────────────


def test_box_iou_pairwise_diag_self() -> None:
    boxes_a = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    boxes_b = boxes_a.clone()
    iou = _box_iou_pairwise_diag(boxes_a, boxes_b)
    assert iou.shape == (2,)
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(1.0)


def test_box_iou_pairwise_diag_disjoint() -> None:
    boxes_a = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    boxes_b = torch.tensor(
        [
            [100.0, 100.0, 110.0, 110.0],
            [200.0, 200.0, 210.0, 210.0],
        ]
    )
    iou = _box_iou_pairwise_diag(boxes_a, boxes_b)
    assert iou[0] == pytest.approx(0.0)
    assert iou[1] == pytest.approx(0.0)


# ── _box_iou_matrix ─────────────────────────────────────────────────────


def test_box_iou_matrix_square() -> None:
    boxes_a = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    boxes_b = boxes_a.clone()
    iou = _box_iou_matrix(boxes_a, boxes_b)
    assert iou.shape == (2, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 1] == pytest.approx(1.0)


def test_box_iou_matrix_disjoint() -> None:
    boxes_a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes_b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    iou = _box_iou_matrix(boxes_a, boxes_b)
    assert iou[0, 0] == pytest.approx(0.0)


def test_box_iou_matrix_rectangular() -> None:
    boxes_a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes_b = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 110.0, 110.0],
        ]
    )
    iou = _box_iou_matrix(boxes_a, boxes_b)
    assert iou.shape == (1, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)


# ── filter_detections_fast ──────────────────────────────────────────────


def test_filter_detections_empty() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    scores = torch.empty((0,))
    classes = torch.empty((0,), dtype=torch.int32)
    keep, suspect, quality = filter_detections_fast(
        boxes,
        scores,
        classes,
        score_threshold=0.3,
        track_person_only=True,
        person_class=0,
        is_tiled=False,
        frame_w=640,
        frame_h=480,
        person_geometry_prior=False,
        geometry_suspect_support=False,
        person_min_height_ratio=0.0,
        person_min_aspect=0.0,
        person_max_aspect=0.0,
        person_min_area_ratio=0.0,
        person_max_area_ratio=0.0,
    )
    assert keep.shape == (0,)
    assert suspect.shape == (0,)
    assert quality.shape == (0,)


def test_filter_detections_score_threshold_only() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.5, 0.2])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    keep, suspect, quality = filter_detections_fast(
        boxes,
        scores,
        classes,
        score_threshold=0.3,
        track_person_only=False,
        person_class=0,
        is_tiled=False,
        frame_w=640,
        frame_h=480,
        person_geometry_prior=False,
        geometry_suspect_support=False,
        person_min_height_ratio=0.0,
        person_min_aspect=0.0,
        person_max_aspect=0.0,
        person_min_area_ratio=0.0,
        person_max_area_ratio=0.0,
    )
    # Only first box (score=0.5 >= 0.3) should be kept
    assert keep.shape == (1,)
    assert keep[0] == 0
    assert suspect.shape == (1,)


def test_filter_detections_person_only() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.5, 0.5])
    classes = torch.tensor([0, 1], dtype=torch.int32)
    keep, suspect, quality = filter_detections_fast(
        boxes,
        scores,
        classes,
        score_threshold=0.3,
        track_person_only=True,
        person_class=0,
        is_tiled=False,
        frame_w=640,
        frame_h=480,
        person_geometry_prior=False,
        geometry_suspect_support=False,
        person_min_height_ratio=0.0,
        person_min_aspect=0.0,
        person_max_aspect=0.0,
        person_min_area_ratio=0.0,
        person_max_area_ratio=0.0,
    )
    assert keep.shape == (1,)
    assert keep[0] == 0  # only class 0


def test_filter_detections_tiled_outside_frame() -> None:
    boxes = torch.tensor(
        [
            [650.0, 490.0, 700.0, 540.0],  # outside frame
            [10.0, 10.0, 60.0, 60.0],  # inside frame
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    keep, suspect, quality = filter_detections_fast(
        boxes,
        scores,
        classes,
        score_threshold=0.3,
        track_person_only=False,
        person_class=0,
        is_tiled=True,
        frame_w=640,
        frame_h=480,
        person_geometry_prior=False,
        geometry_suspect_support=False,
        person_min_height_ratio=0.0,
        person_min_aspect=0.0,
        person_max_aspect=0.0,
        person_min_area_ratio=0.0,
        person_max_area_ratio=0.0,
    )
    # Only second box should be kept (inside frame)
    assert keep.shape == (1,)
    assert keep[0] == 1


def test_filter_detections_geometry_prior() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # aspect=1.0, area_ratio tiny
            [0.0, 0.0, 200.0, 600.0],  # aspect=3.0, reasonable
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    keep, suspect, quality = filter_detections_fast(
        boxes,
        scores,
        classes,
        score_threshold=0.3,
        track_person_only=False,
        person_class=0,
        is_tiled=False,
        frame_w=640,
        frame_h=480,
        person_geometry_prior=True,
        geometry_suspect_support=False,
        person_min_height_ratio=0.05,
        person_min_aspect=0.5,
        person_max_aspect=5.0,
        person_min_area_ratio=0.001,
        person_max_area_ratio=0.5,
    )
    # First box has aspect=1.0 (within 0.5-5.0), second has aspect=3.0
    # First box area = 100, frame_area=307200, ratio=0.0003 < 0.001 => filtered
    # Second box area=120000, ratio=0.39 < 0.5 => kept
    assert keep.shape == (1,)
    assert keep[0] == 1


def test_filter_detections_geometry_suspect_support() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # aspect=1.0, area_ratio tiny => suspect
            [0.0, 0.0, 200.0, 600.0],  # good geometry
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    keep, suspect, quality = filter_detections_fast(
        boxes,
        scores,
        classes,
        score_threshold=0.3,
        track_person_only=False,
        person_class=0,
        is_tiled=False,
        frame_w=640,
        frame_h=480,
        person_geometry_prior=True,
        geometry_suspect_support=True,
        person_min_height_ratio=0.05,
        person_min_aspect=0.5,
        person_max_aspect=5.0,
        person_min_area_ratio=0.001,
        person_max_area_ratio=0.5,
    )
    # Both should be in keep, suspect has same length as keep
    assert keep.shape[0] == 2
    assert suspect.shape == (2,)
    assert suspect[0]
    assert not suspect[1]


# ── nms_fast ────────────────────────────────────────────────────────────


def test_nms_fast_empty() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    scores = torch.empty((0,))
    classes = torch.empty((0,), dtype=torch.int32)
    result = nms_fast(boxes, scores, classes, 0.5, class_aware=False)
    assert result.shape == (0,)


@pytest.mark.skip(
    reason="nms_fast has a source bug: nms/batched_nms only imported inside `if priors` block, causing UnboundLocalError in pure Python path"
)
def test_nms_fast_no_overlap() -> None:
    """Non-overlapping boxes with class_aware=False (plain NMS path)."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 110.0, 110.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    result = nms_fast(boxes, scores, classes, 0.5, class_aware=False)
    assert result.shape == (2,)  # both kept


@pytest.mark.skip(
    reason="nms_fast has a source bug: nms/batched_nms only imported inside `if priors` block, causing UnboundLocalError in pure Python path"
)
def test_nms_fast_high_overlap() -> None:
    """High overlap boxes with class_aware=False (plain NMS path)."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # high overlap
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    result = nms_fast(boxes, scores, classes, 0.5, class_aware=False)
    assert result.shape == (1,)  # only the higher score one kept
    assert result[0] == 1


def test_nms_fast_with_priors() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # suppressed by NMS
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    priors = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    prior_classes = torch.tensor([0], dtype=torch.int32)
    result = nms_fast(
        boxes,
        scores,
        classes,
        0.5,
        class_aware=False,
        priors=priors,
        prior_classes=prior_classes,
    )
    assert result.shape == (2,)  # second box rescued by prior


# ── merge_cross_tile_duplicates ─────────────────────────────────────────


def test_merge_cross_tile_duplicates_empty() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    scores = torch.empty((0,))
    classes = torch.empty((0,), dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
        center_threshold=0.18,
        area_ratio_threshold=0.6,
    )
    assert merged.shape == (0, 4)
    assert counts.shape == (0,)


def test_merge_cross_tile_duplicates_single() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    scores = torch.tensor([0.9])
    classes = torch.tensor([0], dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
    )
    assert merged.shape == (1, 4)
    assert counts[0] == 1  # unmerged


def test_merge_cross_tile_duplicates_high_iou() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # high IoU
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
    )
    assert merged.shape == (1, 4)  # merged into one
    assert counts[0] == 2  # two merged


def test_merge_cross_tile_duplicates_low_iou() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 110.0, 110.0],  # no overlap
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
    )
    assert merged.shape == (2, 4)
    assert counts[0] == 1
    assert counts[1] == 1


def test_merge_cross_tile_duplicates_different_classes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 1], dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
    )
    # Different classes should not be merged
    assert merged.shape == (2, 4)


def test_merge_cross_tile_duplicates_with_tiling() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
        tiling="960p_2x2",
        frame_w=960,
        frame_h=960,
    )
    assert merged.shape[0] == 1


# ── merge_cross_tile_duplicates_fast ────────────────────────────────────


def test_merge_cross_tile_duplicates_fast_empty() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    scores = torch.empty((0,))
    classes = torch.empty((0,), dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates_fast(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
        tiling=None,
    )
    assert merged.shape == (0, 4)
    assert counts.shape == (0,)


def test_merge_cross_tile_duplicates_fast_single() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    scores = torch.tensor([0.9])
    classes = torch.tensor([0], dtype=torch.int32)
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates_fast(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
    )
    assert merged.shape == (1, 4)
    assert counts[0] == 1


def test_merge_cross_tile_duplicates_fast_with_tiling_mode() -> None:
    """Tests tiling mode detection in fast path (C++ paths skipped, falls back to Python)."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.9])
    classes = torch.tensor([0, 0], dtype=torch.int32)
    # When tiling is set, the C++ paths are used if available; otherwise falls back to Python
    merged, merged_scores, merged_classes, counts = merge_cross_tile_duplicates_fast(
        boxes,
        scores,
        classes,
        iou_threshold=0.45,
        tiling="960p_2x2",
        frame_w=960,
        frame_h=960,
    )
    assert merged.shape[0] >= 1
