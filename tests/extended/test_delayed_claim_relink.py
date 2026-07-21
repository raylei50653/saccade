"""Tests for delayed-claim relink via post-merge deferred alias (perception.eval.relink/post_merge)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import sys
import types
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


def test_cheb_gr_delayed_claim_uses_warmup_head_and_records_alias() -> None:
    relinker = PythonSemanticRelinker(
        sim_threshold=0.99,
        ttl=10,
        spatial_gate=0.5,
        min_iou=0.0,
        mahalanobis_threshold=0.0,
        delayed_claim=True,
        claim_warmup_frames=2,
        cheb_gr_claim=True,
        cheb_gr_max_cost=0.9,
        cheb_gr_margin=0.05,
        cheb_gr_min_head=2,
        cheb_gr_max_fwd=0,
        buffer_size=4,
    )
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    box = torch.tensor([0.0, 0.0, 10.0, 20.0])
    relinker.features[1] = emb
    relinker.buffers[1] = [emb, emb.clone()]
    relinker.last_seen[1] = 1
    relinker.last_boxes[1] = box

    first = relinker.resolve(2, emb, box, 0.9, 5, 100, 100, set())
    second = relinker.resolve(2, emb, box, 0.9, 6, 100, 100, set())

    assert first == 2
    assert second == 1
    assert relinker.deferred_alias == {2: 1}
    assert relinker.stats["cheb_gr_claim_attempts"] == 1
    assert relinker.stats["cheb_gr_claim_accepted"] == 1
    assert relinker.stats["delayed_claim_accepted"] == 1


def test_cheb_gr_delayed_claim_waits_for_min_head_before_finalizing() -> None:
    relinker = PythonSemanticRelinker(
        sim_threshold=0.99,
        ttl=10,
        spatial_gate=0.5,
        min_iou=0.0,
        mahalanobis_threshold=0.0,
        delayed_claim=True,
        claim_warmup_frames=1,
        cheb_gr_claim=True,
        cheb_gr_max_cost=0.9,
        cheb_gr_margin=0.0,
        cheb_gr_min_head=3,
        cheb_gr_max_fwd=0,
        buffer_size=4,
    )
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    box = torch.tensor([0.0, 0.0, 10.0, 20.0])
    relinker.features[1] = emb
    relinker.buffers[1] = [emb, emb.clone()]
    relinker.last_seen[1] = 1
    relinker.last_boxes[1] = box

    first = relinker.resolve(2, emb, box, 0.9, 5, 100, 100, set())
    second = relinker.resolve(2, emb, box, 0.9, 6, 100, 100, set())

    assert first == 2
    assert second == 2
    assert 2 in relinker._pending_claims
    assert len(relinker._pending_heads[2]) == 2
    assert relinker.stats["cheb_gr_claim_reject_min_head"] == 0
    assert relinker.deferred_alias == {}

    third = relinker.resolve(2, emb, box, 0.9, 7, 100, 100, set())

    assert third == 1
    assert relinker.deferred_alias == {2: 1}
    assert 2 not in relinker._pending_claims
    assert 2 not in relinker._pending_heads
    assert relinker.stats["cheb_gr_claim_attempts"] == 1
    assert relinker.stats["cheb_gr_claim_accepted"] == 1


def test_cheb_gr_delayed_claim_respects_spatial_gate() -> None:
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=0.1,
        min_iou=0.2,
        mahalanobis_threshold=0.0,
        delayed_claim=True,
        claim_warmup_frames=2,
        cheb_gr_claim=True,
        cheb_gr_max_cost=0.9,
        cheb_gr_margin=0.0,
        cheb_gr_min_head=2,
        cheb_gr_max_fwd=0,
        buffer_size=4,
    )
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features[1] = emb
    relinker.buffers[1] = [emb, emb.clone()]
    relinker.last_seen[1] = 1
    relinker.last_boxes[1] = torch.tensor([80.0, 80.0, 90.0, 100.0])
    box = torch.tensor([0.0, 0.0, 10.0, 20.0])

    first = relinker.resolve(2, emb, box, 0.9, 5, 100, 100, set())
    second = relinker.resolve(2, emb, box, 0.9, 6, 100, 100, set())

    assert first == 2
    assert second == 2
    assert relinker.deferred_alias == {}
    assert relinker.stats["cheb_gr_claim_reject_spatial"] == 1
    assert relinker.stats["delayed_claim_accepted"] == 0


def test_delayed_claim_rejects_wrong_dim_reference_without_crashing() -> None:
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=0.5,
        min_iou=0.0,
        mahalanobis_threshold=0.0,
        delayed_claim=True,
        claim_warmup_frames=1,
    )
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    box = torch.tensor([0.0, 0.0, 10.0, 20.0])
    relinker.features[1] = torch.zeros(1)
    relinker.last_seen[1] = 1
    relinker.last_boxes[1] = box

    resolved = relinker.resolve(2, emb, box, 0.9, 5, 100, 100, set())

    assert resolved == 2
    assert relinker.deferred_alias == {}
    assert relinker.stats["reject_similarity"] == 1


def test_no_embedding_without_motion_only_self_claims() -> None:
    relinker = PythonSemanticRelinker(
        sim_threshold=0.1,
        ttl=10,
        spatial_gate=1.0,
        min_iou=0.0,
        mahalanobis_threshold=0.0,
        motion_enable_motion_only=False,
    )
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    box = torch.tensor([0.0, 0.0, 10.0, 20.0])
    relinker.features[1] = emb
    relinker.last_seen[1] = 1
    relinker.last_boxes[1] = box

    resolved = relinker.resolve(2, None, box, 0.9, 5, 100, 100, set())

    assert resolved == 2
    assert relinker.alias[2] == 2
    assert relinker.stats["accepted"] == 0


def test_gpu_relink_gate_is_disabled_on_cpu_device(monkeypatch) -> None:
    fake_ext = types.SimpleNamespace(relink_gate_batch=lambda *args: None)
    monkeypatch.setitem(sys.modules, "saccade_tracking_ext", fake_ext)

    relinker = PythonSemanticRelinker(
        gpu_relink_gate=True,
        gpu_relink_gate_graph=True,
        device="cpu",
    )

    assert relinker.device.type == "cpu"
    assert relinker.gpu_relink_gate is False
    assert relinker.gpu_relink_gate_graph is False


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
