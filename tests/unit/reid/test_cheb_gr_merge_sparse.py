"""Sparse/pre-gated Cheb-GR merge path equals the dense reference path.

The sparse path (``distance_impl="sparse"``, default) must reproduce the dense
``tracklet_distance_matrix`` entries up to float summation order, and must make
identical merge decisions. Real-embedding equivalence on MOT17 / MOT20 /
DanceTrack is recorded by ``scripts/eval/experiments/merge_impl_equivalence.py``.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from saccade.perception.eval.cheb_gr_merge import (
    cheb_gr_merge_output_tracklets,
    tracklet_distance_matrix,
    tracklet_distance_pairs,
)


def _clustered(seed: int, n_ids: int, per_track: list[int], d: int = 24):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_ids, d)).astype(np.float32)
    feats, owner = [], []
    for t, n in enumerate(per_track):
        c = centers[t % n_ids]
        raw = c[None, :] + 0.35 * rng.standard_normal((n, d)).astype(np.float32)
        feats.append(F.normalize(torch.from_numpy(raw), dim=1))
        owner.extend([t] * n)
    return torch.cat(feats), torch.tensor(owner, dtype=torch.long)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_fwd": 50, "k2": 6, "fuse_lambda": 0.3},
        {"max_fwd": 4, "k2": 6, "fuse_lambda": 0.3},
        {"max_fwd": 0, "k2": 6, "fuse_lambda": 0.3},
        {"max_fwd": 50, "k2": 1, "fuse_lambda": 0.3},
        {"max_fwd": 50, "k2": 6, "fuse_lambda": 1.0},
        {"max_fwd": 7, "k2": 3, "fuse_lambda": 0.5, "cheb_lambda": 1.0},
    ],
)
# row_chunk=None is one block (the dense GEMM call). Smaller blocks may use a
# different BLAS kernel and differ by ulps; with the doubled node set that can
# flip a duplicate-distance tie at the cap (e.g. a 1-row CPU GEMV does here).
# Blocked-mode decision equality is proven on real data, not asserted here.
@pytest.mark.parametrize("row_chunk", [7, None])
def test_sparse_pairs_match_dense_matrix(kwargs: dict[str, Any], row_chunk):
    per_track = [5, 1, 8, 3, 12, 2, 6, 4]
    feats, owner = _clustered(11, 4, per_track)
    t = len(per_track)
    dense = tracklet_distance_matrix(feats, owner, t, **kwargs)
    pairs = [(a, b) for a in range(t) for b in range(a + 1, t)]
    sparse = tracklet_distance_pairs(
        feats, owner, t, pairs, row_chunk=row_chunk, **kwargs
    )
    assert set(sparse) == set(pairs)
    for (a, b), c in sparse.items():
        assert c == pytest.approx(float(dense[a, b]), abs=1e-6)


def test_sparse_pairs_only_scores_requested_pairs():
    feats, owner = _clustered(3, 2, [4, 4, 4])
    out = tracklet_distance_pairs(feats, owner, 3, [(0, 2)])
    assert list(out) == [(0, 2)]
    assert tracklet_distance_pairs(feats, owner, 3, []) == {}


def test_sparse_pairs_block_limit_does_not_change_values():
    feats, owner = _clustered(5, 3, [9, 7, 11, 5])
    pairs = [(0, 1), (0, 3), (1, 2), (2, 3)]
    wide = tracklet_distance_pairs(feats, owner, 4, pairs)
    tiny = tracklet_distance_pairs(feats, owner, 4, pairs, max_block_elems=16)
    for key in pairs:
        assert tiny[key] == pytest.approx(wide[key], abs=1e-6)


def _lines_and_embeddings(seed: int):
    """Fragmented tracks of 3 identities with gaps, overlaps and far gaps."""
    rng = np.random.default_rng(seed)
    d = 24
    centers = rng.standard_normal((3, d)).astype(np.float32)
    spans = [
        (1, 0, 10),
        (2, 0, 14),
        (3, 1, 8),
        (
            4,
            0,
            20,
        ),
        (5, 1, 18),
        (6, 2, 25),
        (7, 0, 40),
        (8, 2, 47),
        (9, 1, 30),
        (10, 2, 120),
        (11, 0, 200),
        (12, 1, 22),
    ]
    lines: list[str] = []
    emb: dict[int, torch.Tensor] = {}
    for tid, ident, start in spans:
        for fr in range(start, start + 6):
            lines.append(f"{fr},{tid},{10 + ident * 50},10,20,40,0.9,-1,-1,-1")
        raw = centers[ident][None, :] + 0.4 * rng.standard_normal((6, d)).astype(
            np.float32
        )
        emb[tid] = F.normalize(torch.from_numpy(raw), dim=1)
    return lines, emb


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("max_cost", [0.3, 0.45, 0.9])
def test_merge_decisions_identical_to_dense(seed: int, max_cost: float):
    lines, emb = _lines_and_embeddings(seed)
    kw = dict(enabled=True, max_cost=max_cost, max_gap=30, max_fwd=5, k2=3)
    log_d: list[dict[str, Any]] = []
    log_s: list[dict[str, Any]] = []
    out_d, st_d = cheb_gr_merge_output_tracklets(
        lines, emb, distance_impl="dense", decision_log=log_d, **kw
    )
    out_s, st_s = cheb_gr_merge_output_tracklets(
        lines, emb, distance_impl="sparse", decision_log=log_s, **kw
    )
    assert out_s == out_d
    assert st_s == st_d

    def accepted(log):
        return [
            (r["a_id"], r["b_id"])
            for r in log
            if r["kind"] == "pair" and r["verdict"] == "accepted"
        ]

    assert accepted(log_s) == accepted(log_d)
    pair_d = [r for r in log_d if r["kind"] == "pair"]
    pair_s = [r for r in log_s if r["kind"] == "pair"]
    assert [(r["a_id"], r["b_id"]) for r in pair_s] == [
        (r["a_id"], r["b_id"]) for r in pair_d
    ]
    for rd, rs in zip(pair_d, pair_s):
        if rs["cost"] is None:
            # Temporally impossible: never scored under the pre-gate.
            assert rs["verdict"] == "reject_temporal"
            assert rd["verdict"] in ("reject_temporal", "reject_cost")
        else:
            assert rs["verdict"] == rd["verdict"]
            assert rs["cost"] == pytest.approx(rd["cost"], abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only contract")
def test_cuda_sparse_pairs_match_dense_with_row_blocks():
    per_track = [50, 3, 44, 50, 1, 38, 50, 12]
    feats, owner = _clustered(8, 4, per_track, d=128)
    feats = feats.cuda()
    t = len(per_track)
    dense = tracklet_distance_matrix(feats, owner, t)
    pairs = [(a, b) for a in range(t) for b in range(a + 1, t)]
    sparse = tracklet_distance_pairs(feats, owner, t, pairs, row_chunk=17)
    for (a, b), c in sparse.items():
        assert c == pytest.approx(float(dense[a, b]), abs=1e-6)


def test_unknown_distance_impl_is_rejected():
    lines, emb = _lines_and_embeddings(0)
    with pytest.raises(ValueError, match="distance_impl"):
        cheb_gr_merge_output_tracklets(lines, emb, enabled=True, distance_impl="approx")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ordered_group_sum_preserves_cancellation_order(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from saccade.perception.eval.cheb_gr_merge import _ordered_group_sum

    keys = torch.tensor([2, 1, 2, 1, 2], device=device)
    values = torch.tensor([1e20, 4.0, -1e20, 5.0, 3.0], device=device)
    for _ in range(5):
        groups, sums = _ordered_group_sum(keys, values)
        assert groups.tolist() == [1, 2]
        assert sums.tolist() == [9.0, 3.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only contract")
def test_cuda_sparse_costs_repeat_bitwise_and_reject_autocast():
    feats, owner = _clustered(8, 4, [50, 3, 44, 50, 1, 38, 50, 12], d=128)
    feats = feats.cuda()
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)]
    expected = tracklet_distance_pairs(feats, owner, 8, pairs, row_chunk=17)
    for _ in range(4):
        assert tracklet_distance_pairs(feats, owner, 8, pairs, row_chunk=17) == expected
    with torch.autocast("cuda"):
        with pytest.raises(ValueError, match="autocast off"):
            tracklet_distance_pairs(feats, owner, 8, pairs)
    with pytest.raises(ValueError, match="FP32"):
        tracklet_distance_pairs(feats.half(), owner, 8, pairs)
