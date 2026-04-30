from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.relink import PythonSemanticRelinker


def test_motion_candidate_ids_filters_by_lost_age_window() -> None:
    relinker = PythonSemanticRelinker(
        ttl=10,
        min_lost_frames=2,
        mahalanobis_threshold=6.6,
    )
    relinker.features = {
        10: torch.ones(4),
        20: torch.ones(4),
        30: torch.ones(4),
    }
    relinker.last_seen = {
        10: 98,  # age 2 -> eligible
        20: 99,  # age 1 -> too recent
        30: 80,  # age 20 -> too old
    }

    assert relinker.motion_candidate_ids(100) == [10]


def test_motion_candidate_ids_without_frame_id_returns_all_feature_ids() -> None:
    relinker = PythonSemanticRelinker()
    relinker.features = {
        7: torch.ones(4),
        9: torch.ones(4),
    }

    assert relinker.motion_candidate_ids() == [7, 9]


def test_motion_candidate_ids_empty_when_mahalanobis_disabled() -> None:
    relinker = PythonSemanticRelinker(mahalanobis_threshold=0.0)
    relinker.features = {
        7: torch.ones(4),
        9: torch.ones(4),
    }
    relinker.last_seen = {7: 10, 9: 10}

    assert relinker.motion_candidate_ids(12) == []


def test_resolve_many_matches_sequential_resolve() -> None:
    batch_relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        ema_beta=0.5,
        spatial_gate=0.2,
        min_lost_frames=1,
        min_iou=0.1,
        mahalanobis_threshold=0.0,
    )
    sequential_relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        ema_beta=0.5,
        spatial_gate=0.2,
        min_lost_frames=1,
        min_iou=0.1,
        mahalanobis_threshold=0.0,
    )

    seed_embedding_a = torch.tensor([1.0, 0.0], dtype=torch.float32)
    seed_embedding_b = torch.tensor([0.0, 1.0], dtype=torch.float32)
    batch_relinker.features = {11: seed_embedding_a, 22: seed_embedding_b}
    sequential_relinker.features = {11: seed_embedding_a, 22: seed_embedding_b}
    batch_relinker.last_seen = {11: 3, 22: 3}
    sequential_relinker.last_seen = {11: 3, 22: 3}
    batch_relinker.last_boxes = {
        11: torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float32),
        22: torch.tensor([20.0, 20.0, 30.0, 30.0], dtype=torch.float32),
    }
    sequential_relinker.last_boxes = {
        11: torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float32),
        22: torch.tensor([20.0, 20.0, 30.0, 30.0], dtype=torch.float32),
    }

    candidates = [
        (
            31,
            torch.tensor([0.99, 0.01], dtype=torch.float32),
            torch.tensor([0.5, 0.5, 10.5, 10.5], dtype=torch.float32),
            0.91,
        ),
        (
            44,
            torch.tensor([0.01, 0.99], dtype=torch.float32),
            torch.tensor([20.5, 20.5, 30.5, 30.5], dtype=torch.float32),
            0.87,
        ),
    ]

    batch_resolved = batch_relinker.resolve_many(
        candidates,
        frame_id=4,
        w=100,
        h=100,
    )
    assigned: set[int] = set()
    sequential_resolved = [
        sequential_relinker.resolve(raw_id, embedding, box, score, 4, 100, 100, assigned)
        for raw_id, embedding, box, score in candidates
    ]

    assert batch_resolved == sequential_resolved == [11, 22]
    assert batch_relinker.alias == sequential_relinker.alias
