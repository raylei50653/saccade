"""Unit tests for Cheb-GR offline tracklet merge (numeric core)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import saccade.perception.eval.cheb_gr_merge as cheb_gr_merge
from saccade.perception.eval.cheb_gr_merge import (
    cheb_gr_merge_output_tracklets,
    extract_tracklet_embeddings,
    temporal_sample_indices,
    tracklet_distance_matrix,
)


def test_temporal_sample_returns_all_when_short():
    assert temporal_sample_indices(3, 20) == [0, 1, 2]
    assert temporal_sample_indices(0, 20) == []
    assert temporal_sample_indices(5, 0) == []


def test_temporal_sample_is_distributed_and_increasing():
    idx = temporal_sample_indices(100, 10)
    assert len(idx) == 10
    assert idx == sorted(idx)
    assert len(set(idx)) == 10
    # Coverage spans the whole lifespan (first bin near start, last near end).
    assert idx[0] < 10
    assert idx[-1] >= 90


def test_temporal_sample_prefers_high_score_within_bin():
    n = 20
    scores = np.zeros(n, dtype=np.float32)
    # Put a quality spike at index 3 (inside the first bin of size 2 -> bin0=[0,2)).
    # Use 2 bins so bin0 = [0,10), bin1 = [10,20); spikes at 4 and 17.
    scores[4] = 1.0
    scores[17] = 1.0
    idx = temporal_sample_indices(n, 2, scores=scores)
    assert idx == [4, 17]


def _normed(
    rng: np.random.Generator, n: int, d: int, center: np.ndarray
) -> torch.Tensor:
    raw = center[None, :] + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return F.normalize(torch.from_numpy(raw), dim=1)


def test_tracklet_distance_matrix_separates_identities():
    rng = np.random.default_rng(0)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    c1 = rng.standard_normal(d).astype(np.float32)
    # 3 tracklets: 0 and 2 share identity c0, tracklet 1 is c1.
    f0 = _normed(rng, 8, d, c0)
    f1 = _normed(rng, 8, d, c1)
    f2 = _normed(rng, 8, d, c0)
    feats = torch.cat([f0, f1, f2], dim=0)
    owner = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)

    dmat = tracklet_distance_matrix(feats, owner, 3, max_fwd=0)

    assert torch.isinf(dmat.diagonal()).all()
    # Same-identity pair (0,2) must be closer than cross-identity (0,1)/(1,2).
    assert dmat[0, 2] < dmat[0, 1]
    assert dmat[0, 2] < dmat[1, 2]
    assert torch.allclose(dmat, dmat.t(), equal_nan=True)


def test_merge_links_disjoint_same_identity_tracklets():
    rng = np.random.default_rng(1)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    c1 = rng.standard_normal(d).astype(np.float32)

    # tid 1: frames 1-5, tid 2: frames 20-25 (same identity, disjoint, gap=14)
    # tid 3: frames 1-25 (different identity, overlaps both -> must NOT merge)
    lines: list[str] = []
    for fr in range(1, 6):
        lines.append(f"{fr},1,10,10,20,40,0.9,-1,-1,-1")
    for fr in range(20, 26):
        lines.append(f"{fr},2,10,10,20,40,0.9,-1,-1,-1")
    for fr in range(1, 26):
        lines.append(f"{fr},3,200,200,20,40,0.9,-1,-1,-1")

    embeddings = {
        1: _normed(rng, 6, d, c0),
        2: _normed(rng, 6, d, c0),
        3: _normed(rng, 6, d, c1),
    }

    out, stats = cheb_gr_merge_output_tracklets(
        lines, embeddings, enabled=True, max_cost=0.9, max_gap=30, max_fwd=0
    )
    assert stats["merges"] == 1
    assert stats["ids_before"] == 3
    assert stats["ids_after"] == 2

    # tid 2 should have been relabeled to tid 1; tid 3 untouched.
    out_ids = {int(line.split(",")[1]) for line in out}
    assert out_ids == {1, 3}


def test_merge_disabled_is_passthrough():
    lines = ["1,1,10,10,20,40,0.9,-1,-1,-1"]
    out, stats = cheb_gr_merge_output_tracklets(lines, {}, enabled=False)
    assert out == lines
    assert stats["merges"] == 0


def test_merge_respects_temporal_overlap_gate():
    rng = np.random.default_rng(2)
    d = 32
    c0 = rng.standard_normal(d).astype(np.float32)
    # Two same-identity tracklets that fully overlap in time -> never merge.
    lines = []
    for fr in range(1, 11):
        lines.append(f"{fr},1,10,10,20,40,0.9,-1,-1,-1")
        lines.append(f"{fr},2,50,50,20,40,0.9,-1,-1,-1")
    embeddings = {1: _normed(rng, 6, d, c0), 2: _normed(rng, 6, d, c0)}
    out, stats = cheb_gr_merge_output_tracklets(
        lines, embeddings, enabled=True, max_cost=0.99, max_gap=30, max_fwd=0
    )
    assert stats["merges"] == 0


def test_extract_tracklet_embeddings_filters_mnv4_dirty_crops(tmp_path):
    from PIL import Image

    class DummyExtractor:
        model_type = "mobilenetv4_reid"
        device = "cpu"
        feature_dim = 4
        input_hw = (12, 8)

        def __init__(self) -> None:
            self.batches: list[torch.Tensor] = []

        def extract(self, t: torch.Tensor) -> torch.Tensor:
            self.batches.append(t.clone())
            return torch.ones((t.shape[0], self.feature_dim), dtype=torch.float32)

    seq_dir = tmp_path / "img1"
    seq_dir.mkdir()
    for frame in (1, 2):
        Image.new("RGB", (32, 32), color=(frame, 10, 20)).save(
            seq_dir / f"{frame:06d}.jpg"
        )

    # tid 1 in frame 1 is covered 50% by tid 2, whose lower foot makes it the
    # foreground occluder. The mnv4 visclean path should sample only tid 1's
    # clean frame 2 crop.
    lines = [
        "1,1,0,0,10,10,0.9,-1,-1,-1",
        "1,2,0,5,10,10,0.8,-1,-1,-1",
        "2,1,20,0,10,10,0.7,-1,-1,-1",
    ]
    ext = DummyExtractor()

    out = extract_tracklet_embeddings(
        lines,
        str(seq_dir),
        ext,
        n_samples=10,
        batch=8,
    )

    assert {tid: emb.shape[0] for tid, emb in out.items()} == {1: 1, 2: 1}
    assert len(ext.batches) == 1
    assert tuple(ext.batches[0].shape) == (2, 3, 12, 8)


def test_extract_tracklet_embeddings_prefers_native_for_mnv4(monkeypatch):
    class DummyExtractor:
        model_type = "mobilenetv4_reid"
        device = "cpu"
        feature_dim = 4
        input_hw = (12, 8)

    calls: list[tuple[int, tuple[int, int], str, int]] = []

    def fake_native(
        samples,
        by_frame,
        seq_dir,
        extractor,
        *,
        crop_hw,
        im_ext,
        batch,
    ):
        del by_frame, seq_dir, extractor
        calls.append((len(samples), crop_hw, im_ext, batch))
        return torch.arange(len(samples) * 4, dtype=torch.float32).reshape(-1, 4)

    monkeypatch.setattr(cheb_gr_merge, "_extract_native_crops_trt", fake_native)

    lines = [
        "1,1,0,0,10,10,0.9,-1,-1,-1",
        "2,1,2,0,10,10,0.8,-1,-1,-1",
        "1,2,20,0,10,10,0.7,-1,-1,-1",
    ]

    out = extract_tracklet_embeddings(
        lines,
        "/unused/img1",
        DummyExtractor(),
        n_samples=10,
        batch=8,
    )

    assert calls == [(3, (12, 8), ".jpg", 8)]
    assert {tid: emb.shape for tid, emb in out.items()} == {
        1: torch.Size([2, 4]),
        2: torch.Size([1, 4]),
    }
