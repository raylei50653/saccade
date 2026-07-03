"""Unit tests for the causal online Cheb-GR ID handover (numeric core)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from saccade.perception.eval.cheb_gr_online import causal_handover_lines
from saccade.perception.reid.cheb_gr import (
    cheb_gr_kreciprocal,
    cheb_gr_kreciprocal_self_dense,
    cheb_gr_kreciprocal_self_dense_padded,
)


def _normed(
    rng: np.random.Generator, n: int, d: int, center: np.ndarray
) -> torch.Tensor:
    raw = center[None, :] + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return F.normalize(torch.from_numpy(raw), dim=1)


def _track(lines: list[str], tid: int, frames: range, x: float = 10.0) -> None:
    for fr in frames:
        lines.append(f"{fr},{tid},{x},10,20,40,0.9,-1,-1,-1")


def _ids(lines: list[str]) -> set[int]:
    return {int(ln.split(",")[1]) for ln in lines}


def test_dense_self_kreciprocal_matches_sparse_reference():
    rng = np.random.default_rng(42)
    feats = _normed(rng, 7, 16, rng.standard_normal(16).astype(np.float32))

    ref = cheb_gr_kreciprocal(
        feats,
        feats,
        cheb_lambda=2.0,
        k2=4,
        max_fwd=5,
        fuse_lambda=0.3,
    )
    dense = cheb_gr_kreciprocal_self_dense(
        feats,
        cheb_lambda=2.0,
        k2=4,
        max_fwd=5,
        fuse_lambda=0.3,
    )

    torch.testing.assert_close(dense, ref, rtol=1e-5, atol=1e-6)


def test_padded_self_kreciprocal_matches_dense_valid_block():
    rng = np.random.default_rng(43)
    feats = _normed(rng, 5, 16, rng.standard_normal(16).astype(np.float32))
    cap = 16
    padded = torch.zeros((cap, feats.shape[1]), dtype=feats.dtype)
    padded[: feats.shape[0]] = feats
    valid_mask = torch.zeros((cap * 2,), dtype=torch.bool)
    n = feats.shape[0]
    valid_mask[: n * 2] = True
    order = torch.arange(cap * 2, dtype=torch.long)
    order[:n] = torch.arange(n)
    order[n : n * 2] = torch.arange(cap, cap + n)
    order[n * 2 : n * 2 + cap - n] = torch.arange(n, cap)
    order[n * 2 + cap - n :] = torch.arange(cap + n, cap * 2)
    gallery_indices = torch.arange(cap, dtype=torch.long)
    gallery_indices[:n] = torch.arange(n, n * 2)

    ref = cheb_gr_kreciprocal_self_dense(
        feats,
        cheb_lambda=2.0,
        k2=min(6, feats.shape[0]),
        max_fwd=50,
        fuse_lambda=0.3,
    )
    padded_dist = cheb_gr_kreciprocal_self_dense_padded(
        padded,
        valid_mask,
        order,
        gallery_indices,
        cheb_lambda=2.0,
        k2=6,
        max_fwd=50,
        fuse_lambda=0.3,
    )

    torch.testing.assert_close(
        padded_dist[: feats.shape[0], : feats.shape[0]],
        ref,
        rtol=1e-5,
        atol=1e-6,
    )


def test_disabled_is_passthrough():
    lines = ["1,1,10,10,20,40,0.9,-1,-1,-1"]
    out, stats = causal_handover_lines(lines, {}, {}, enabled=False)
    assert out == lines
    assert stats["handovers"] == 0


def test_simple_handover_relabels_newborn():
    rng = np.random.default_rng(0)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    c1 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))  # dies frame 10
    _track(lines, 2, range(16, 31))  # born frame 16, same identity
    _track(lines, 3, range(1, 31), x=200.0)  # distractor, different identity

    head = {2: _normed(rng, 3, d, c0), 3: _normed(rng, 3, d, c1)}
    bank = {1: _normed(rng, 8, d, c0), 3: _normed(rng, 8, d, c1)}
    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.9, max_fwd=0
    )
    assert stats["handovers"] == 1
    assert _ids(out) == {1, 3}
    assert stats["ids_after"] == 2


def test_gap_gate_blocks_far_and_overlapping():
    rng = np.random.default_rng(1)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    head = {2: _normed(rng, 3, d, c0), 4: _normed(rng, 3, d, c0)}
    bank = {
        1: _normed(rng, 8, d, c0),
        3: _normed(rng, 8, d, c0),
        2: _normed(rng, 8, d, c0),
        4: _normed(rng, 8, d, c0),
    }

    # Gap beyond max_gap -> no handover.
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 2, range(81, 91))
    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.9, max_gap=60, max_fwd=0
    )
    assert stats["handovers"] == 0

    # Temporal overlap (gap < 1) -> no handover.
    lines = []
    _track(lines, 3, range(1, 20))
    _track(lines, 4, range(19, 30))
    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.9, max_fwd=0
    )
    assert stats["handovers"] == 0


def test_identity_revived_at_most_once():
    rng = np.random.default_rng(2)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))  # dead identity
    _track(lines, 2, range(16, 26))  # first claimant (earlier decision)
    _track(lines, 3, range(20, 30))  # second claimant, same appearance
    head = {2: _normed(rng, 3, d, c0), 3: _normed(rng, 3, d, c0)}
    bank = {tid: _normed(rng, 8, d, c0) for tid in (1, 2, 3)}

    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.9, max_fwd=0
    )
    assert stats["handovers"] == 1
    out_ids = _ids(out)
    assert 1 in out_ids and len(out_ids) == 2
    # No frame may contain the same id twice.
    seen: set[tuple[int, int]] = set()
    for ln in out:
        fr, tid = int(ln.split(",")[0]), int(ln.split(",")[1])
        assert (fr, tid) not in seen
        seen.add((fr, tid))


def test_chain_handover_follows_labels():
    rng = np.random.default_rng(3)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 2, range(15, 25))
    _track(lines, 3, range(30, 40))
    head = {2: _normed(rng, 3, d, c0), 3: _normed(rng, 3, d, c0)}
    bank = {tid: _normed(rng, 8, d, c0) for tid in (1, 2, 3)}

    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.9, max_fwd=0
    )
    # 2 takes over 1; when 2 dies, 3 takes over the chain -> all labeled 1.
    assert stats["handovers"] == 2
    assert _ids(out) == {1}


def test_causality_no_future_candidates():
    """A track that dies AFTER the newborn's birth is never a candidate."""
    rng = np.random.default_rng(4)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 20))  # dies frame 19 — after 2's birth (10)
    _track(lines, 2, range(10, 30), x=200.0)
    head = {2: _normed(rng, 3, d, c0)}
    bank = {1: _normed(rng, 8, d, c0)}
    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.99, max_fwd=0
    )
    assert stats["handovers"] == 0


def test_cost_gate_rejects_different_identity():
    rng = np.random.default_rng(5)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    c1 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 2, range(16, 26))
    head = {2: _normed(rng, 3, d, c1)}
    bank = {1: _normed(rng, 8, d, c0)}
    out, stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.3, max_fwd=0
    )
    assert stats["handovers"] == 0
    assert _ids(out) == {1, 2}


def test_newborn_without_head_samples_keeps_id():
    rng = np.random.default_rng(6)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 2, range(16, 26))
    out, stats = causal_handover_lines(
        lines, {}, {1: _normed(rng, 8, d, c0)}, enabled=True, max_cost=0.9
    )
    assert stats["handovers"] == 0
    assert _ids(out) == {1, 2}


def test_min_head_gate_blocks_single_clean_head_sample():
    rng = np.random.default_rng(7)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 2, range(16, 26))
    head = {2: _normed(rng, 1, d, c0)}
    bank = {1: _normed(rng, 8, d, c0)}

    out, stats = causal_handover_lines(
        lines,
        head,
        bank,
        enabled=True,
        max_cost=0.9,
        min_head_samples=2,
        max_fwd=0,
    )
    assert stats["handovers"] == 0
    assert stats["reject_min_head"] == 1
    assert _ids(out) == {1, 2}


def test_margin_gate_rejects_ambiguous_candidate():
    rng = np.random.default_rng(8)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 3, range(3, 13))
    _track(lines, 2, range(16, 26))
    head = {2: _normed(rng, 3, d, c0)}
    shared_bank = _normed(rng, 8, d, c0)
    bank = {
        1: shared_bank,
        3: shared_bank.clone(),
    }

    out, stats = causal_handover_lines(
        lines,
        head,
        bank,
        enabled=True,
        max_cost=0.9,
        margin=0.05,
        max_fwd=0,
    )
    assert stats["handovers"] == 0
    assert stats["reject_margin"] == 1
    assert _ids(out) == {1, 2, 3}


def test_decision_log_records_accepted_handover():
    rng = np.random.default_rng(9)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    _track(lines, 1, range(1, 11))
    _track(lines, 2, range(16, 26))
    head = {2: _normed(rng, 3, d, c0)}
    bank = {1: _normed(rng, 8, d, c0)}
    decision_log: list[dict[str, int | float | str | bool]] = []

    out, stats = causal_handover_lines(
        lines,
        head,
        bank,
        enabled=True,
        max_cost=0.9,
        max_fwd=0,
        decision_log=decision_log,
    )
    assert stats["handovers"] == 1
    assert stats["decisions_logged"] == 1
    assert _ids(out) == {1}
    assert decision_log[0]["newborn_id"] == 2
    assert decision_log[0]["candidate_id"] == 1
    assert decision_log[0]["accepted"] is True
    assert decision_log[0]["reason"] == "accepted"
