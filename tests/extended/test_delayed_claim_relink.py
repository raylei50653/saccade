from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.post_merge import apply_deferred_alias  # noqa: E402
from saccade.perception.eval.relink import PythonSemanticRelinker  # noqa: E402


def test_delayed_claim_waits_for_warmup_then_records_deferred_alias() -> None:
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=0.5,
        min_iou=0.0,
        mahalanobis_threshold=0.0,
        delayed_claim=True,
        claim_warmup_frames=2,
    )
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    box = torch.tensor([0.0, 0.0, 10.0, 20.0])
    relinker.features[1] = emb
    relinker.last_seen[1] = 1
    relinker.last_boxes[1] = box

    first = relinker.resolve(2, emb, box, 0.9, 5, 100, 100, set())
    second = relinker.resolve(2, emb, box, 0.9, 6, 100, 100, set())

    assert first == 2
    assert second == 1
    assert relinker.deferred_alias == {2: 1}
    assert relinker.stats["delayed_claim_accepted"] == 1


def test_apply_deferred_alias_remaps_mot_lines() -> None:
    lines = [
        "1,10,0.0,0.0,10.0,10.0,0.900,-1,-1,-1",
        "2,11,1.0,0.0,10.0,10.0,0.800,-1,-1,-1",
    ]

    remapped, stats = apply_deferred_alias(lines, {11: 10})

    assert stats == {
        "aliases": 1,
        "aliases_skipped_overlap": 0,
        "lines_remapped": 1,
        "ids_before": 2,
        "ids_after": 1,
    }
    assert remapped == [
        "1,10,0.00,0.00,10.00,10.00,0.9000,-1,-1,-1",
        "2,10,1.00,0.00,10.00,10.00,0.8000,-1,-1,-1",
    ]


def test_apply_deferred_alias_skips_overlapping_tracks() -> None:
    lines = [
        "1,10,0.0,0.0,10.0,10.0,0.900,-1,-1,-1",
        "1,11,50.0,0.0,10.0,10.0,0.800,-1,-1,-1",
    ]

    remapped, stats = apply_deferred_alias(lines, {11: 10})

    assert stats["aliases"] == 0
    assert stats["aliases_skipped_overlap"] == 1
    assert stats["lines_remapped"] == 0
    assert remapped == [
        "1,10,0.00,0.00,10.00,10.00,0.9000,-1,-1,-1",
        "1,11,50.00,0.00,10.00,10.00,0.8000,-1,-1,-1",
    ]
