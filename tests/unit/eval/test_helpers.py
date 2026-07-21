"""Tests for saccade.perception.eval.helpers.

Tests pure helper functions without requiring GPU or native extensions.
"""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import torch

from saccade.perception.eval.helpers import (
    materialize_gpu_track_results,
    build_dynamic_reid_observations,
)


# ── materialize_gpu_track_results ───────────────────────────────────────


class MockResultBuffers:
    """Simulates GPU result buffers with __getitem__ and slicing."""

    def __init__(self, count: int, dim: int = 4):
        self._data = {
            "count": torch.tensor([count]),
            "boxes": torch.rand(count, dim),
            "scores": torch.rand(count),
            "ids": torch.arange(count, dtype=torch.int32),
            "classes": torch.randint(0, 5, (count,), dtype=torch.int32),
            "det_idx": torch.arange(count, dtype=torch.int32),
        }

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]


def test_materialize_empty_result() -> None:
    buffers = MockResultBuffers(0)
    result = materialize_gpu_track_results(buffers)
    assert result["count"] == 0
    assert result["boxes"].shape == (0, 4)
    assert result["scores"].shape == (0,)
    assert result["ids"].shape == (0,)


def test_materialize_single_detection() -> None:
    buffers = MockResultBuffers(1)
    result = materialize_gpu_track_results(buffers)
    assert result["count"] == 1
    assert result["boxes"].shape == (1, 4)
    assert result["scores"].shape == (1,)
    assert result["ids"].shape == (1,)
    assert result["classes"].shape == (1,)
    assert result["det_idx"].shape == (1,)


def test_materialize_multiple_detections() -> None:
    buffers = MockResultBuffers(5)
    result = materialize_gpu_track_results(buffers)
    assert result["count"] == 5
    assert result["boxes"].shape == (5, 4)
    assert result["scores"].shape == (5,)
    assert result["ids"].shape == (5,)
    assert result["det_idx"].shape == (5,)


def test_materialize_with_default_class() -> None:
    """When default_class_id is set, classes should all be that value."""
    buffers = MockResultBuffers(3)
    result = materialize_gpu_track_results(buffers, default_class_id=0)
    assert result["classes"] is not None
    assert torch.all(result["classes"] == 0)
    assert result["classes"].shape == (3,)


def test_materialize_no_det_idx() -> None:
    buffers = MockResultBuffers(2)
    result = materialize_gpu_track_results(buffers, include_det_idx=False)
    assert result["det_idx"] is None


def test_materialize_cpu_transfer() -> None:
    """Result should be on CPU (not GPU)."""
    buffers = MockResultBuffers(2)
    result = materialize_gpu_track_results(buffers)
    assert result["boxes"].device.type == "cpu"
    assert result["scores"].device.type == "cpu"
    assert result["ids"].device.type == "cpu"


# ── build_dynamic_reid_observations ─────────────────────────────────────


def test_build_reid_observations_empty() -> None:
    result = build_dynamic_reid_observations(
        track_ids=[],
        track_boxes=[],
        track_scores=[],
        track_classes=None,
        person_class=0,
    )
    assert result == {}


def test_build_reid_observations_no_classes() -> None:
    result = build_dynamic_reid_observations(
        track_ids=[1, 2],
        track_boxes=[(0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 200.0, 200.0)],
        track_scores=[0.8, 0.9],
        track_classes=None,
        person_class=0,
    )
    assert result == {}


def test_build_reid_observations_filter_by_class() -> None:
    """Only person_class (0) should be included."""
    result = build_dynamic_reid_observations(
        track_ids=[1, 2, 3],
        track_boxes=[
            (0.0, 0.0, 100.0, 100.0),
            (0.0, 0.0, 200.0, 200.0),
            (0.0, 0.0, 300.0, 300.0),
        ],
        track_scores=[0.8, 0.9, 0.7],
        track_classes=[0, 1, 0],  # 1 is not person_class
        person_class=0,
    )
    assert set(result.keys()) == {1, 3}  # Only IDs 1 and 3 (class=0)
    assert 2 not in result  # ID 2 has class=1, filtered out


def test_build_reid_observations_includes_scores() -> None:
    from saccade.perception.tracking.tracker_gpu import ReIDTrackObservation

    result = build_dynamic_reid_observations(
        track_ids=[42],
        track_boxes=[(10.0, 20.0, 50.0, 60.0)],
        track_scores=[0.85],
        track_classes=[0],
        person_class=0,
    )
    assert 42 in result
    obs = result[42]
    assert isinstance(obs, ReIDTrackObservation)
    assert obs.box == (10.0, 20.0, 50.0, 60.0)
    assert obs.det_score == 0.85


def test_build_reid_observations_multiple_persons() -> None:
    result = build_dynamic_reid_observations(
        track_ids=[1, 2, 3, 4],
        track_boxes=[
            (0.0, 0.0, 100.0, 100.0),
            (0.0, 0.0, 200.0, 200.0),
            (0.0, 0.0, 300.0, 300.0),
            (0.0, 0.0, 400.0, 400.0),
        ],
        track_scores=[0.8, 0.9, 0.7, 0.95],
        track_classes=[0, 0, 1, 0],
        person_class=0,
    )
    assert set(result.keys()) == {1, 2, 4}  # ID 3 has class=1
    assert len(result) == 3
