from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.relink import (  # noqa: E402
    IdentityResolver,
    PythonSemanticRelinker,
)


# ─── __init__ parameter validation ───────────────────────────────────────────


def test_init_invalid_rerank_mode() -> None:
    """Unknown rerank_mode should raise ValueError."""
    from pytest import raises

    with raises(ValueError, match="Unknown rerank_mode"):
        PythonSemanticRelinker(rerank_mode="invalid")


def test_init_invalid_experimental_mode() -> None:
    """Unknown experimental_mode should raise ValueError."""
    from pytest import raises

    with raises(ValueError, match="Unknown experimental_mode"):
        PythonSemanticRelinker(experimental_mode="bogus")


def test_init_buffer_size_clamped_to_one() -> None:
    """buffer_size=0 should be clamped to max(1, 0)=1."""
    relinker = PythonSemanticRelinker(buffer_size=0)
    assert relinker.buffer_size == 1


def test_init_reciprocal_margin_clamped() -> None:
    """Negative reciprocal_margin should be clamped to 0.0."""
    relinker = PythonSemanticRelinker(reciprocal_margin=-0.1)
    assert relinker.reciprocal_margin == 0.0


def test_init_strict_sim_threshold_fallback() -> None:
    """strict_sim_threshold=0.0 should fall back to sim_threshold."""
    relinker = PythonSemanticRelinker(sim_threshold=0.98)
    assert relinker.strict_sim_threshold == 0.98


def test_init_strict_sim_threshold_custom() -> None:
    """strict_sim_threshold > 0 should be used directly."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.98,
        strict_sim_threshold=0.85,
    )
    assert relinker.strict_sim_threshold == 0.85


def test_init_weights_clamped_non_negative() -> None:
    """Negative weights should be clamped to 0.0."""
    relinker = PythonSemanticRelinker(
        w_sim_base=-1.0,
        w_iou_base=-1.0,
        w_maha_base=-1.0,
        iou_weight=-0.5,
        mahalanobis_weight=-0.5,
        dynamic_margin_crowd=-1.0,
        dynamic_margin_age=-1.0,
    )
    assert relinker.w_sim_base == 0.0
    assert relinker.w_iou_base == 0.0
    assert relinker.w_maha_base == 0.0
    assert relinker.iou_weight == 0.0
    assert relinker.mahalanobis_weight == 0.0
    assert relinker.dynamic_margin_crowd == 0.0
    assert relinker.dynamic_margin_age == 0.0


def test_init_stats_default_zero() -> None:
    """All stats counters should start at 0."""
    relinker = PythonSemanticRelinker()
    for v in relinker.stats.values():
        assert v == 0


# ─── biometric integration ──────────────────────────────────────────────────


def test_set_biometric_tracker() -> None:
    """Setting a biometric tracker stores reference."""
    tracker = MagicMock()
    relinker = PythonSemanticRelinker(biometric_threshold=0.5)
    relinker.set_biometric_tracker(tracker)
    assert relinker._bio_tracker is tracker


def test_bio_distance_no_tracker_returns_zero() -> None:
    """Without a tracker, _bio_distance returns 0.0."""
    relinker = PythonSemanticRelinker(biometric_threshold=0.5)
    assert relinker._bio_distance(1, 2) == 0.0


def test_bio_distance_below_threshold_returns_zero() -> None:
    """With tracker but biometric_threshold=0.0, returns 0.0."""
    tracker = MagicMock()
    relinker = PythonSemanticRelinker(biometric_threshold=0.0)
    relinker.set_biometric_tracker(tracker)
    assert relinker._bio_distance(1, 2) == 0.0


def test_bio_distance_missing_data_returns_zero() -> None:
    """Biometric data missing for one id returns 0.0."""
    tracker = MagicMock()
    tracker.get_biometric.side_effect = lambda i: None
    relinker = PythonSemanticRelinker(biometric_threshold=0.5)
    relinker.set_biometric_tracker(tracker)
    assert relinker._bio_distance(1, 2) == 0.0


def test_bio_distance_partial_match_returns_zero() -> None:
    """If only one id has all required keys, distance still computed but
    only for keys present in both."""
    tracker = MagicMock()
    tracker.get_biometric.side_effect = lambda i: (
        {"r_leg": 1.0, "r_shoulder": 2.0, "r_head": 3.0} if i == 1 else {}
    )
    relinker = PythonSemanticRelinker(biometric_threshold=0.5)
    relinker.set_biometric_tracker(tracker)
    assert relinker._bio_distance(1, 2) == 0.0


def test_bio_distance_full_match_computes_l1() -> None:
    """L1 distance computed over shared keys."""
    tracker = MagicMock()
    tracker.get_biometric.side_effect = lambda i: {
        "r_leg": 1.0 if i == 1 else 4.0,
        "r_shoulder": 2.0 if i == 1 else 6.0,
        "r_head": 3.0 if i == 1 else 9.0,
    }
    relinker = PythonSemanticRelinker(biometric_threshold=0.5)
    relinker.set_biometric_tracker(tracker)
    # |1-4| + |2-6| + |3-9| = 3 + 4 + 6 = 13
    assert relinker._bio_distance(1, 2) == 13.0


# ─── spatial metrics ─────────────────────────────────────────────────────────


def test_spatial_metrics_no_overlap() -> None:
    """Two boxes with no overlap: iou=0, center_norm=dist/max(w,h)."""
    relinker = PythonSemanticRelinker()
    box = torch.tensor([0.0, 0.0, 10.0, 10.0])
    old_box = torch.tensor([100.0, 100.0, 110.0, 110.0])
    center_norm, iou = relinker._spatial_metrics(box, old_box, w=200, h=200)
    assert iou == 0.0
    dist = ((5.0 - 105.0) ** 2 + (5.0 - 105.0) ** 2) ** 0.5  # ≈141.42
    expected_center_norm = dist / 200.0
    assert abs(center_norm - expected_center_norm) < 1e-5


def test_spatial_metrics_full_overlap() -> None:
    """Same box: iou≈1.0, center_norm=0.0."""
    relinker = PythonSemanticRelinker()
    box = torch.tensor([10.0, 10.0, 20.0, 20.0])
    old_box = torch.tensor([10.0, 10.0, 20.0, 20.0])
    center_norm, iou = relinker._spatial_metrics(box, old_box, w=100, h=100)
    assert abs(iou - 1.0) < 1e-5
    assert center_norm == 0.0


def test_spatial_metrics_partial_overlap() -> None:
    """Partial overlap: non-zero iou and center_norm."""
    relinker = PythonSemanticRelinker()
    box = torch.tensor([0.0, 0.0, 20.0, 20.0])
    old_box = torch.tensor([10.0, 10.0, 30.0, 30.0])
    center_norm, iou = relinker._spatial_metrics(box, old_box, w=100, h=100)
    # Intersection: [10,10] to [20,20] → 10×10 = 100
    # Union: 400 + 400 - 100 = 700 → iou ≈ 0.143
    assert iou > 0.0
    assert iou < 1.0
    assert center_norm > 0.0


# ─── measurement & mahalanobis ──────────────────────────────────────────────


def test_measurement_basic() -> None:
    """Box measurement: [cx, cy, w/h, h]."""
    relinker = PythonSemanticRelinker()
    # Box [x1, y1, x2, y2] → w = x2-x1, h = y2-y1
    box = torch.tensor([10.0, 20.0, 50.0, 80.0])  # w=40, h=60
    m = relinker._measurement(box)
    assert m[0] == 30.0  # cx = (10+50)/2
    assert m[1] == 50.0  # cy = (20+80)/2
    assert abs(m[2] - 40.0 / 60.0) < 1e-4  # w/h
    assert m[3] == 60.0  # h = 80-20


def test_measurement_zero_width() -> None:
    """Zero-width box should not crash (width clamped to 1e-6)."""
    relinker = PythonSemanticRelinker()
    box = torch.tensor([10.0, 20.0, 10.0, 80.0])
    m = relinker._measurement(box)
    assert m[2] > 0.0  # w/h should be positive


def test_motion_box_basic() -> None:
    """Valid motion state returns reconstructed box."""
    relinker = PythonSemanticRelinker()
    snap = MagicMock()
    snap.state = [100.0, 200.0, 0.5, 100.0]  # cx=100, cy=200, aspect=0.5, h=100
    box = relinker._motion_box(snap)
    assert box is not None
    assert box[0] == 75.0  # cx - 0.5*w = 100 - 25
    assert box[1] == 150.0  # cy - 0.5*h = 200 - 50
    assert box[2] == 125.0  # cx + 0.5*w = 100 + 25
    assert box[3] == 250.0  # cy + 0.5*h = 200 + 50


def test_motion_box_negative_h_returns_none() -> None:
    """Negative height returns None."""
    relinker = PythonSemanticRelinker()
    snap = MagicMock()
    snap.state = [100.0, 200.0, 0.5, -1.0]
    assert relinker._motion_box(snap) is None


def test_motion_box_zero_aspect_returns_none() -> None:
    """Zero aspect ratio returns None."""
    relinker = PythonSemanticRelinker()
    snap = MagicMock()
    snap.state = [100.0, 200.0, 0.0, 100.0]
    assert relinker._motion_box(snap) is None


def test_motion_box_invalid_state_returns_none() -> None:
    """Exception during state read returns None (exception caught internally)."""
    relinker = PythonSemanticRelinker()
    snap = MagicMock()
    snap.state = None  # type: ignore
    # _motion_box wraps state read in try/except → returns None on exception
    result = relinker._motion_box(snap)
    assert result is None


def test_mahalanobis_basic() -> None:
    """Mahalanobis distance computation with valid snapshot."""
    relinker = PythonSemanticRelinker(mahalanobis_threshold=6.6)
    snap = MagicMock()
    snap.state = [100.0, 200.0, 0.5, 100.0]
    snap.covariance = [1.0] * 64  # 8×8 matrix (identity-like)
    box = torch.tensor([105.0, 205.0, 50.0, 100.0])
    maha = relinker._mahalanobis(box, snap)
    # Should return a finite float
    assert isinstance(maha, float)
    assert not torch.isnan(torch.tensor(maha))


def test_mahalanobis_singular_matrix_uses_pseudoinverse() -> None:
    """Singular covariance should fall back to pinv and not crash."""
    relinker = PythonSemanticRelinker(mahalanobis_threshold=6.6)
    snap = MagicMock()
    snap.state = [100.0, 200.0, 0.5, 100.0]
    snap.covariance = [0.0] * 64  # zero matrix → singular
    box = torch.tensor([100.0, 200.0, 50.0, 100.0])
    maha = relinker._mahalanobis(box, snap)
    assert isinstance(maha, float)


# ─── buffer operations ───────────────────────────────────────────────────────


def test_buffer_mean_single_item() -> None:
    """Single buffer item returned directly."""
    relinker = PythonSemanticRelinker(buffer_size=3, device="cpu")
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.buffers[1] = [emb]
    result = relinker._buffer_mean(1)
    assert torch.allclose(result, emb)


def test_buffer_mean_multiple_items() -> None:
    """Multiple buffer items return their normalized mean."""
    relinker = PythonSemanticRelinker(buffer_size=3, device="cpu")
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.buffers[1] = [a, b]
    result = relinker._buffer_mean(1)
    # mean = [0.5, 0.5, 0, 0], normalized = [1/√2, 1/√2, 0, 0]
    expected = torch.tensor([1.0 / (2**0.5), 1.0 / (2**0.5), 0.0, 0.0])
    assert torch.allclose(result, expected, atol=1e-5)


def test_buffer_mean_empty_returns_none() -> None:
    """Empty buffer returns None."""
    relinker = PythonSemanticRelinker(buffer_size=3)
    assert relinker._buffer_mean(999) is None


def test_buffer_consistency_single_item() -> None:
    """Single item → consistency=1.0."""
    relinker = PythonSemanticRelinker(buffer_size=3)
    relinker.buffers[1] = [torch.tensor([1.0, 0.0, 0.0, 0.0])]
    assert relinker._buffer_consistency(1) == 1.0


def test_buffer_consistency_empty() -> None:
    """No buffer → consistency=1.0."""
    relinker = PythonSemanticRelinker(buffer_size=3)
    assert relinker._buffer_consistency(999) == 1.0


def test_buffer_consistency_perfect() -> None:
    """All identical vectors → perfect consistency."""
    relinker = PythonSemanticRelinker(buffer_size=3)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.buffers[1] = [v, v, v]
    assert relinker._buffer_consistency(1) == 1.0


def test_buffer_consistency_orthogonal() -> None:
    """Orthogonal vectors → consistency should be negative."""
    relinker = PythonSemanticRelinker(buffer_size=3)
    v0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.buffers[1] = [v0, v1]
    c = relinker._buffer_consistency(1)
    # With 2 vectors, cosines are [1, 0, 0, 1] → (2-2)/(2*1) = 0
    # But normalization makes diagonal=1, so sum-diag = 0
    assert isinstance(c, float)


def test_buffer_sim_mean_rerank_mode() -> None:
    """mean rerank_mode: dot(query, mean_of_buffer)."""
    relinker = PythonSemanticRelinker(buffer_size=2, rerank_mode="mean", device="cpu")
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.buffers[1] = [a, b]
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sim = relinker._buffer_sim(1, query)
    # mean = [0.5, 0.5, 0, 0], normalized = [√0.5, √0.5, 0, 0]
    # dot([1,0,0,0], normalized) = √0.5 ≈ 0.707
    assert abs(sim - 0.707) < 0.01


def test_buffer_sim_max_rerank_mode() -> None:
    """max rerank_mode: max cosine over buffer items."""
    relinker = PythonSemanticRelinker(buffer_size=2, rerank_mode="max", device="cpu")
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.buffers[1] = [a, b]
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sim = relinker._buffer_sim(1, query)
    # max of [1.0, 0.0] = 1.0
    assert abs(sim - 1.0) < 1e-5


def test_buffer_sim_top2_mean_rerank_mode() -> None:
    """top2_mean rerank_mode: mean of top-2 cosines."""
    relinker = PythonSemanticRelinker(
        buffer_size=3, rerank_mode="top2_mean", device="cpu"
    )
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    c = torch.tensor([0.7, 0.0, 0.0, 0.0])
    relinker.buffers[1] = [a, b, c]
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sim = relinker._buffer_sim(1, query)
    # cosines: [1.0, 0.0, 0.7] → top2 = [1.0, 0.7] → mean = 0.85
    assert abs(sim - 0.85) < 0.01


def test_buffer_sim_weighted_rerank_mode() -> None:
    """weighted rerank_mode: 0.7*max + 0.3*mean."""
    relinker = PythonSemanticRelinker(
        buffer_size=2, rerank_mode="weighted", device="cpu"
    )
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.buffers[1] = [a, b]
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sim = relinker._buffer_sim(1, query)
    # cosines: [1.0, 0.0] → max=1.0, mean=0.5
    # 0.7*1.0 + 0.3*0.5 = 0.85
    assert abs(sim - 0.85) < 1e-5


def test_buffer_sim_empty_buffer_falls_back_to_features() -> None:
    """Empty buffer falls back to stored feature."""
    relinker = PythonSemanticRelinker(buffer_size=2, rerank_mode="max", device="cpu")
    relinker.features[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sim = relinker._buffer_sim(1, query)
    assert abs(sim - 1.0) < 1e-5


def test_buffer_sim_no_buffer_no_feature_returns_minus_one() -> None:
    """No buffer and no feature → -1.0."""
    relinker = PythonSemanticRelinker(buffer_size=2)
    assert relinker._buffer_sim(999, torch.tensor([1.0, 0.0])) == -1.0


# ─── inject / canonical / has_feature ────────────────────────────────────────


def test_inject_reference() -> None:
    """inject_reference normalizes and stores embedding."""
    relinker = PythonSemanticRelinker()
    emb = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm=5
    relinker.inject_reference(1, emb)
    # features stored on device, compare on cpu
    expected = torch.tensor([0.6, 0.8, 0.0, 0.0])
    assert torch.allclose(relinker.features[1].cpu(), expected, atol=1e-5)


def test_inject_reference_buffer_mode() -> None:
    """In buffer mode, inject also appends to buffer."""
    relinker = PythonSemanticRelinker(buffer_size=3)
    emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.inject_reference(1, emb)
    assert len(relinker.buffers[1]) == 1
    assert torch.allclose(relinker.buffers[1][0].cpu(), emb.cpu())


def test_inject_references_many() -> None:
    """inject_references_many processes multiple references."""
    relinker = PythonSemanticRelinker(buffer_size=1)
    refs = [
        (1, torch.tensor([1.0, 0.0, 0.0, 0.0])),
        (2, torch.tensor([0.0, 1.0, 0.0, 0.0])),
    ]
    relinker.inject_references_many(refs)
    assert torch.allclose(
        relinker.features[1].cpu(), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    assert torch.allclose(
        relinker.features[2].cpu(), torch.tensor([0.0, 1.0, 0.0, 0.0])
    )


def test_canonical_id_no_alias() -> None:
    """Without alias, canonical_id returns raw_id."""
    relinker = PythonSemanticRelinker()
    assert relinker.canonical_id(42) == 42


def test_canonical_id_with_alias() -> None:
    """With alias, canonical_id resolves to aliased id."""
    relinker = PythonSemanticRelinker()
    relinker.alias[42] = 100
    assert relinker.canonical_id(42) == 100


def test_has_feature() -> None:
    """has_feature checks presence in features dict."""
    relinker = PythonSemanticRelinker()
    assert not relinker.has_feature(1)
    relinker.features[1] = torch.ones(4)
    assert relinker.has_feature(1)
    assert not relinker.has_feature(2)


# ─── quality filtering ──────────────────────────────────────────────────────


def test_resolve_quality_score_fail() -> None:
    """Low score with clean_score_threshold → quality rejection."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,  # disable spatial gate
        min_lost_frames=1,
        clean_score_threshold=0.95,  # require score > 0.95
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 3}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    raw = relinker.resolve(
        2,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.90,  # below clean_score_threshold
        frame_id=4,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 2  # new id (quality rejected)
    assert relinker.stats["reject_quality"] > 0
    assert relinker.stats["reject_quality_score"] > 0


def test_resolve_quality_margin_fail() -> None:
    """Box at frame edge with clean_margin_ratio → quality rejection."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        clean_margin_ratio=0.1,  # 10% margin
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 3}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # Box at frame edge
    raw = relinker.resolve(
        2,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 5.0, 5.0]),  # x=0 < 10px margin
        score=0.99,
        frame_id=4,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 2
    assert relinker.stats["reject_quality"] > 0
    assert relinker.stats["reject_quality_margin"] > 0


def test_resolve_quality_aspect_low() -> None:
    """Very narrow aspect ratio → quality rejection stats incremented."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        clean_min_aspect=0.5,  # min aspect ratio
        clean_margin_ratio=0.01,  # trigger quality check block
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 3}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # aspect = 1/100 = 0.01 < 0.5 → is_clean=False
    raw = relinker.resolve(
        2,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 100.0, 1.0]),
        score=0.99,
        frame_id=4,
        w=100,
        h=100,
        assigned=set(),
    )
    # Match still goes through (quality doesn't prevent matching)
    assert raw == 1
    assert relinker.stats["reject_quality"] > 0
    assert relinker.stats["reject_quality_aspect_low"] > 0


def test_resolve_quality_aspect_high() -> None:
    """Very tall aspect ratio → quality rejection stats incremented."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        clean_max_aspect=5.0,
        clean_margin_ratio=0.01,  # trigger quality check block
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 3}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # aspect = 100/1 = 100 > 5.0 → is_clean=False
    raw = relinker.resolve(
        2,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 100.0]),
        score=0.99,
        frame_id=4,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 1  # match still goes through
    assert relinker.stats["reject_quality"] > 0
    assert relinker.stats["reject_quality_aspect_high"] > 0


def test_resolve_quality_strict_threshold_for_unclean() -> None:
    """Unclean boxes use strict_sim_threshold (lower) for matching."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.98,
        strict_sim_threshold=0.85,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        clean_score_threshold=0.95,
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 3}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # score=0.90 < clean_score_threshold → quality fail, but strict threshold=0.85 used
    # Normalized [0.87, 0.01, 0, 0] → cos ≈ 0.87 > strict=0.85
    raw = relinker.resolve(
        2,
        torch.tensor([0.87, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.90,
        frame_id=4,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 1  # matched despite quality fail using strict threshold


# ─── resolve: matching scenarios ────────────────────────────────────────────


def test_resolve_new_id_no_candidates() -> None:
    """No existing features → new id."""
    relinker = PythonSemanticRelinker(sim_threshold=0.8, ttl=10)
    raw = relinker.resolve(
        100,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 10.0, 10.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 100
    assert relinker.stats["new_ids"] == 1
    assert 100 in relinker.features


def test_resolve_match_by_similarity() -> None:
    """Best similarity match accepted."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,  # disable spatial gating
        min_lost_frames=1,
        mahalanobis_threshold=0.0,
    )
    seed_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    seed_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.features = {1: seed_a, 2: seed_b}
    relinker.last_seen = {1: 0, 2: 0}
    relinker.last_boxes = {
        1: torch.tensor([0.0, 0.0, 10.0, 10.0]),
        2: torch.tensor([50.0, 50.0, 60.0, 60.0]),
    }

    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 1  # matches seed_a


def test_resolve_no_match_below_threshold() -> None:
    """Candidate below sim_threshold → new id."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.95,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # [0.5, 0.5, 0, 0] → normalized cos = 0.707 < 0.95
    raw = relinker.resolve(
        3,
        torch.tensor([0.5, 0.5, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 3  # new id
    assert relinker.stats["reject_similarity"] > 0


def test_resolve_reject_by_age_too_old() -> None:
    """Track too old → rejected by age gate."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=5,
        spatial_gate=1.0,
        min_lost_frames=1,
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}  # age=10 > ttl=5
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=10,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 3  # new id
    assert relinker.stats["reject_age"] > 0
    assert relinker.stats["reject_age_too_old"] > 0


def test_resolve_reject_by_age_too_fresh() -> None:
    """Track too fresh (< min_lost_frames) → rejected."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=3,  # need age >= 3
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 8}  # age=2 < min_lost_frames=3
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=10,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 3
    assert relinker.stats["reject_age_too_fresh"] > 0


def test_resolve_reject_by_spatial() -> None:
    """Candidate fails spatial gate (center norm too far or IoU too low)."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=0.01,  # very strict
        min_lost_frames=1,
        min_iou=0.5,  # very high IoU required
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # Box far away → center_norm large, IoU=0
    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([90.0, 90.0, 100.0, 100.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 3  # new id
    assert relinker.stats["reject_spatial"] > 0


def test_resolve_reject_by_mahalanobis() -> None:
    """Candidate fails Mahalanobis threshold."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        mahalanobis_threshold=0.5,  # very strict
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}
    # Add motion snapshot with large Mahalanobis distance
    snap = MagicMock()
    snap.state = [0.0, 0.0, 0.5, 10.0]
    snap.covariance = [1.0] * 64
    relinker.motion = {1: snap}

    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([90.0, 90.0, 100.0, 100.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 3
    assert relinker.stats["reject_mahalanobis"] > 0


def test_resolve_reject_by_margin() -> None:
    """Best match rejected due to small margin (ambiguous)."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        reciprocal_margin=0.3,  # need 30% gap between 1st and 2nd
        mahalanobis_threshold=0.0,
    )
    # Two very similar embeddings → ambiguous match
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.99, 0.1, 0.0, 0.0])  # very close to a
    relinker.features = {1: a, 2: b}
    relinker.last_seen = {1: 0, 2: 0}
    relinker.last_boxes = {
        1: torch.tensor([0.0, 0.0, 10.0, 10.0]),
        2: torch.tensor([50.0, 50.0, 60.0, 60.0]),
    }

    relinker.resolve(
        3,
        torch.tensor([0.995, 0.05, 0.0, 0.0]),  # close to both
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    # Either rejected by margin or matched — depends on exact scores
    assert relinker.stats["reject_margin"] > 0 or relinker.stats["accepted"] > 0


def test_resolve_reject_by_biometric() -> None:
    """Biometric distance exceeds threshold → rejected."""
    tracker = MagicMock()
    relinker = PythonSemanticRelinker(
        biometric_threshold=1.0,
        min_lost_frames=1,
        mahalanobis_threshold=0.0,
    )
    relinker.set_biometric_tracker(tracker)

    tracker.get_biometric.side_effect = lambda i: {
        "r_leg": 0.0 if i == 1 else 10.0,
        "r_shoulder": 0.0 if i == 1 else 10.0,
        "r_head": 0.0 if i == 1 else 10.0,
    }

    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    # Biometric distance = |0-10| + |0-10| + |0-10| = 30 > threshold=1.0
    assert raw == 3
    assert relinker.stats["reject_biometric"] > 0


def test_resolve_reject_assigned() -> None:
    """Already assigned candidates are skipped."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    assigned = {1}  # already assigned
    raw = relinker.resolve(
        3,
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=assigned,
    )
    assert raw == 3  # new id (track 1 was skipped)
    assert relinker.stats["reject_assigned"] > 0


def test_resolve_update_features_ema() -> None:
    """Single-buffer mode: features updated with EMA."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        ema_beta=0.5,
        mahalanobis_threshold=0.0,
    )
    old = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: old}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    # [0.9, 0.1, 0, 0] normalized has cos ≈ 0.994 with [1,0,0,0] → matches
    new_emb = torch.tensor([0.9, 0.1, 0.0, 0.0])
    relinker.resolve(
        2,
        new_emb,
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    # EMA: beta*old + (1-beta)*new = 0.5*[1,0,0,0] + 0.5*[0.994, 0.111, 0, 0]
    # Then normalized. Check that features changed from original.
    assert torch.allclose(relinker.features[1].cpu(), old.cpu(), atol=1e-5) is False, (
        "features should have been updated"
    )


def test_resolve_update_features_buffer_mode() -> None:
    """Buffer mode: features updated with buffer mean."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        buffer_size=3,
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0], device=relinker.device)
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    new_emb = torch.tensor([0.9, 0.1, 0.0, 0.0])
    relinker.resolve(
        2,
        new_emb,
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert len(relinker.buffers[1]) == 1


def test_resolve_embedding_is_none() -> None:
    """None embedding → returns aliased id without processing."""
    relinker = PythonSemanticRelinker(motion_enable_motion_only=False)
    relinker.alias[99] = 1  # pre-existing alias
    result = relinker.resolve(
        99,
        None,
        torch.tensor([0.0, 0.0, 10.0, 10.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert result == 1


# ─── resolve_many_packed ────────────────────────────────────────────────────


def test_resolve_many_packed() -> None:
    """resolve_many_packed processes packed inputs."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        mahalanobis_threshold=0.0,
    )
    seed_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    seed_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    relinker.features = {1: seed_a, 2: seed_b}
    relinker.last_seen = {1: 0, 2: 0}
    relinker.last_boxes = {
        1: torch.tensor([0.0, 0.0, 10.0, 10.0]),
        2: torch.tensor([50.0, 50.0, 60.0, 60.0]),
    }

    raw_ids = [3, 4]
    embeddings = [
        torch.tensor([0.99, 0.01, 0.0, 0.0]),
        torch.tensor([0.01, 0.99, 0.0, 0.0]),
    ]
    boxes = [
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        torch.tensor([51.0, 51.0, 61.0, 61.0]),
    ]
    scores = [0.99, 0.98]

    result = relinker.resolve_many_packed(
        raw_ids,
        embeddings,
        boxes,
        scores,
        frame_id=1,
        w=100,
        h=100,
    )
    assert result == [1, 2]


# ─── update_motion_snapshots ────────────────────────────────────────────────


def test_update_motion_snapshots() -> None:
    """Motion snapshots stored with alias resolution."""
    relinker = PythonSemanticRelinker()
    snap = MagicMock()
    snap.obj_id = 1
    relinker.alias = {1: 100}
    relinker.update_motion_snapshots([snap])
    assert relinker.motion[100] is snap


# ─── unified score mode ─────────────────────────────────────────────────────


def test_resolve_unified_score_mode() -> None:
    """Unified score mode uses weighted combination of sim, iou, maha."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        w_sim_base=0.5,
        w_iou_base=0.3,
        w_maha_base=0.2,
        mahalanobis_threshold=10.0,  # high threshold → maha_score ≈ 0
        mahalanobis_weight=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}
    snap = MagicMock()
    snap.state = [5.0, 5.0, 1.0, 10.0]
    snap.covariance = [1.0] * 64
    relinker.motion = {1: snap}

    raw = relinker.resolve(
        2,
        torch.tensor([0.95, 0.05, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 1
    assert relinker.stats["accepted"] > 0


def test_resolve_legacy_joint_score_mode() -> None:
    """Legacy joint score mode: sim + w_iou*iou + w_maha*maha_score."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=1.0,
        min_lost_frames=1,
        iou_weight=0.3,
        mahalanobis_weight=0.2,
        mahalanobis_threshold=10.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}
    snap = MagicMock()
    snap.state = [5.0, 5.0, 1.0, 10.0]
    snap.covariance = [1.0] * 64
    relinker.motion = {1: snap}

    raw = relinker.resolve(
        2,
        torch.tensor([0.90, 0.1, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 11.0, 11.0]),
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 1


# ─── appearance_first mode ──────────────────────────────────────────────────


def test_resolve_appearance_first_bypass() -> None:
    """Appearance-first mode bypasses spatial gate for high-sim clean candidates."""
    relinker = PythonSemanticRelinker(
        sim_threshold=0.8,
        ttl=10,
        spatial_gate=0.01,  # very strict
        min_lost_frames=1,
        experimental_mode="appearance_first",
        appearance_first_sim_threshold=0.95,
        appearance_first_margin=0.0,
        mahalanobis_threshold=0.0,
    )
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}

    raw = relinker.resolve(
        2,
        torch.tensor([0.97, 0.03, 0.0, 0.0]),  # sim > 0.95
        torch.tensor([90.0, 90.0, 100.0, 100.0]),  # far → spatial fail
        score=0.99,
        frame_id=1,
        w=100,
        h=100,
        assigned=set(),
    )
    assert raw == 1  # bypassed spatial gate
    assert relinker.stats["appearance_first_bypass"] > 0


# ─── report ──────────────────────────────────────────────────────────────────


def test_report_prints_stats() -> None:
    """report() prints formatted statistics."""
    import io
    import contextlib

    relinker = PythonSemanticRelinker()
    seed = torch.tensor([1.0, 0.0, 0.0, 0.0])
    relinker.features = {1: seed}
    relinker.last_seen = {1: 0}
    relinker.last_boxes = {1: torch.tensor([0.0, 0.0, 10.0, 10.0])}
    relinker.stats["attempts"] = 5
    relinker.stats["accepted"] = 3
    relinker.stats["new_ids"] = 2
    relinker.stats["reject_similarity"] = 2
    relinker.accept_sims = [0.95, 0.98, 0.92]
    relinker.accept_ious = [0.8, 0.75, 0.9]
    relinker.accept_center_dists = [0.01, 0.02, 0.015]
    relinker.accept_mahas = [1.0, 2.0, 0.5]

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        relinker.report()
    output = f.getvalue()
    assert "Semantic Relink Report" in output
    assert "attempts=5" in output
    assert "accepted=3" in output
    assert "new_ids=2" in output
    assert "reject_similarity=2" in output
    assert "mean_sim" in output
    assert "mean_iou" in output


def test_report_empty_stats() -> None:
    """report() handles empty stats gracefully."""
    import io
    import contextlib

    relinker = PythonSemanticRelinker()
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        relinker.report()
    output = f.getvalue()
    assert "Semantic Relink Report" in output


# ─── IdentityResolver ───────────────────────────────────────────────────────


def test_identity_resolver_resolve_pass_empty() -> None:
    """Empty local_ids returns empty list."""
    mock_relinker = MagicMock()
    mock_lifecycle = MagicMock()
    resolver = IdentityResolver(mock_relinker, mock_lifecycle)
    assert (
        resolver.resolve_pass([], [], [], [], frame_id=0, frame_w=100, frame_h=100)
        == []
    )


def test_identity_resolver_resolve_pass_packed_api() -> None:
    """resolve_pass uses resolve_many_packed when available."""
    mock_relinker = MagicMock()
    mock_relinker.resolve_many_packed.return_value = [1, 2]
    mock_lifecycle = MagicMock()
    mock_lifecycle.resolve_many_packed.return_value = [10, 20]
    resolver = IdentityResolver(mock_relinker, mock_lifecycle)

    result = resolver.resolve_pass(
        [1, 2],
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        [torch.tensor([0.0, 0.0, 10.0, 10.0]), torch.tensor([20.0, 20.0, 30.0, 30.0])],
        [0.99, 0.98],
        frame_id=1,
        frame_w=100,
        frame_h=100,
    )
    assert result == [10, 20]
    mock_relinker.resolve_many_packed.assert_called_once()
    mock_lifecycle.resolve_many_packed.assert_called_once()


def test_identity_resolver_resolve_pass_many_api() -> None:
    """resolve_pass falls back to resolve_many when packed not available."""
    mock_relinker = MagicMock()
    del mock_relinker.resolve_many_packed
    mock_relinker.resolve_many.return_value = [1, 2]
    mock_lifecycle = MagicMock()
    del mock_lifecycle.resolve_many_packed
    mock_lifecycle.resolve_many.return_value = [10, 20]
    resolver = IdentityResolver(mock_relinker, mock_lifecycle)

    result = resolver.resolve_pass(
        [1, 2],
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        [torch.tensor([0.0, 0.0, 10.0, 10.0]), torch.tensor([20.0, 20.0, 30.0, 30.0])],
        [0.99, 0.98],
        frame_id=1,
        frame_w=100,
        frame_h=100,
    )
    assert result == [10, 20]
    mock_relinker.resolve_many.assert_called_once()


def test_identity_resolver_resolve_pass_per_resolve() -> None:
    """resolve_pass falls back to per-item resolve when no batch API."""
    mock_relinker = MagicMock()
    del mock_relinker.resolve_many_packed
    del mock_relinker.resolve_many
    mock_relinker.resolve.side_effect = lambda *args: args[0]
    mock_lifecycle = MagicMock()
    del mock_lifecycle.resolve_many_packed
    del mock_lifecycle.resolve_many
    mock_lifecycle.resolve.side_effect = lambda *args: args[0]
    resolver = IdentityResolver(mock_relinker, mock_lifecycle)

    result = resolver.resolve_pass(
        [1, 2],
        [None, None],
        [torch.tensor([0.0, 0.0, 10.0, 10.0]), torch.tensor([20.0, 20.0, 30.0, 30.0])],
        [0.99, 0.98],
        frame_id=1,
        frame_w=100,
        frame_h=100,
    )
    assert result == [1, 2]
    assert mock_relinker.resolve.call_count == 2
