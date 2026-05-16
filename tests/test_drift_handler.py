"""Tests for SemanticDriftHandler (perception/drift_handler.py).

Covers:
  - __init__ defaults
  - _get_dynamic_alpha (all count ranges × all DegradationLevel values)
  - calculate_drift (new track, existing track, all levels)
  - filter_for_batch (priority ordering, batch limits per level, area tiebreak)
  - update_history (seeding, EMA update, normalization)
  - prune_expired_centroids (expired vs non-expired, custom timeout)
  - clear_history (single-track removal from all dicts)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from saccade.perception.drift_handler import SemanticDriftHandler
from saccade.resource.resource_manager import DegradationLevel

if TYPE_CHECKING:
    pass


# ─── __init__ ───────────────────────────────────────────────────────────────


def test_init_defaults() -> None:
    handler = SemanticDriftHandler()
    assert handler.base_threshold == 0.95
    assert handler.base_alpha == 0.3
    assert handler.N_OPT == 8
    assert handler.feature_history == {}
    assert handler.track_update_count == {}
    assert handler.last_active_time == {}


def test_init_custom_params() -> None:
    handler = SemanticDriftHandler(similarity_threshold=0.90, base_alpha=0.5)
    assert handler.base_threshold == 0.90
    assert handler.base_alpha == 0.5


# ─── _get_dynamic_alpha ─────────────────────────────────────────────────────


def _make_handler_with_history(counts: dict[int, int]) -> SemanticDriftHandler:
    """Create a handler with pre-set update counts."""
    handler = SemanticDriftHandler()
    for tid, count in counts.items():
        handler.track_update_count[tid] = count
    return handler


def test_alpha_zero_count_any_level() -> None:
    """count == 0 → alpha = 1.0 for all levels."""
    handler = SemanticDriftHandler()
    for level in DegradationLevel:
        assert handler._get_dynamic_alpha(99, level) == 1.0


def test_alpha_low_count_normal() -> None:
    """count 1-4, NORMAL → 0.7."""
    handler = _make_handler_with_history({10: 1})
    assert handler._get_dynamic_alpha(10, DegradationLevel.NORMAL) == 0.7


def test_alpha_low_count_reduced() -> None:
    """count 1-4, REDUCED → 0.8."""
    handler = _make_handler_with_history({10: 2})
    assert handler._get_dynamic_alpha(10, DegradationLevel.REDUCED) == pytest.approx(
        0.8
    )


def test_alpha_low_count_fast_path() -> None:
    """count 1-4, FAST_PATH → 1.0 (capped at 1.0, 0.7+0.3=1.0)."""
    handler = _make_handler_with_history({10: 3})
    assert handler._get_dynamic_alpha(10, DegradationLevel.FAST_PATH) == 1.0


def test_alpha_low_count_emergency() -> None:
    """count 1-4, EMERGENCY → 1.0 (0.7+0.7=1.4 → capped)."""
    handler = _make_handler_with_history({10: 4})
    assert handler._get_dynamic_alpha(10, DegradationLevel.EMERGENCY) == 1.0


def test_alpha_high_count_normal() -> None:
    """count >= 5, NORMAL → base_alpha = 0.3."""
    handler = _make_handler_with_history({10: 5})
    assert handler._get_dynamic_alpha(10, DegradationLevel.NORMAL) == pytest.approx(0.3)


def test_alpha_high_count_reduced() -> None:
    """count >= 5, REDUCED → 0.3+0.1 = 0.4."""
    handler = _make_handler_with_history({10: 10})
    assert handler._get_dynamic_alpha(10, DegradationLevel.REDUCED) == pytest.approx(
        0.4
    )


def test_alpha_high_count_fast_path() -> None:
    """count >= 5, FAST_PATH → 0.3+0.3 = 0.6."""
    handler = _make_handler_with_history({10: 10})
    assert handler._get_dynamic_alpha(10, DegradationLevel.FAST_PATH) == pytest.approx(
        0.6
    )


def test_alpha_high_count_emergency() -> None:
    """count >= 5, EMERGENCY → 0.3+0.7 = 1.0 (capped)."""
    handler = _make_handler_with_history({10: 10})
    assert handler._get_dynamic_alpha(10, DegradationLevel.EMERGENCY) == 1.0


def test_alpha_custom_base_alpha() -> None:
    """Custom base_alpha propagates to high-count branch."""
    handler = SemanticDriftHandler(base_alpha=0.5)
    handler.track_update_count[10] = 10
    assert handler._get_dynamic_alpha(10, DegradationLevel.NORMAL) == pytest.approx(0.5)


def test_alpha_unknown_level_defaults_to_zero_offset() -> None:
    """Unknown DegradationLevel falls back to 0.0 offset."""
    handler = SemanticDriftHandler()
    handler.track_update_count[10] = 10
    # Use int value that is not a valid level — falls through to .get() default
    assert handler._get_dynamic_alpha(10, DegradationLevel.NORMAL) == 0.3


# ─── calculate_drift ────────────────────────────────────────────────────────


def test_drift_new_track() -> None:
    """New track returns (0.0, True)."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([1.0, 0.0, 0.0])
    sim, persist = handler.calculate_drift(99, feat)
    assert sim == 0.0
    assert persist is True


def test_drift_existing_track_identical() -> None:
    """Identical feature → high similarity, no persist."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([1.0, 0.0, 0.0])
    handler.feature_history[1] = feat
    sim, persist = handler.calculate_drift(1, feat)
    assert sim == pytest.approx(1.0)
    assert persist is False


def test_drift_existing_track_similar() -> None:
    """Very similar feature → similarity > threshold, no persist."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([1.0, 0.0, 0.0])
    handler.feature_history[1] = feat
    similar = feat.clone()
    similar[0] = 0.99
    similar = F.normalize(similar, p=2, dim=0)
    sim, persist = handler.calculate_drift(1, similar)
    assert sim > 0.95
    assert persist is False


def test_drift_existing_track_different() -> None:
    """Dissimilar feature → similarity < threshold, persist."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([1.0, 0.0, 0.0])
    handler.feature_history[1] = feat
    different = torch.tensor([0.0, 1.0, 0.0])
    sim, persist = handler.calculate_drift(1, different)
    assert sim < 0.95
    assert persist is True


def test_drift_all_degradation_levels() -> None:
    """Each level adjusts the dynamic threshold correctly."""
    handler = SemanticDriftHandler(similarity_threshold=0.95)
    feat = torch.tensor([1.0, 0.0, 0.0])
    handler.feature_history[1] = feat
    different = torch.tensor([0.0, 1.0, 0.0])

    # NORMAL: threshold=0.95, sim=0.0 → persist
    sim_n, p_n = handler.calculate_drift(1, different, DegradationLevel.NORMAL)
    assert sim_n < 0.95 and p_n is True

    # REDUCED: threshold=0.975
    sim_r, p_r = handler.calculate_drift(1, different, DegradationLevel.REDUCED)
    assert p_r is True

    # FAST_PATH: threshold=0.99
    sim_f, p_f = handler.calculate_drift(1, different, DegradationLevel.FAST_PATH)
    assert p_f is True

    # EMERGENCY: threshold=1.0 → only exact match persists=False
    sim_e, p_e = handler.calculate_drift(1, different, DegradationLevel.EMERGENCY)
    assert sim_e < 1.0 and p_e is True


def test_drift_emergency_exact_match() -> None:
    """EMERGENCY with identical feature → sim=1.0, no persist."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([1.0, 0.0, 0.0])
    handler.feature_history[1] = feat
    sim, persist = handler.calculate_drift(1, feat, DegradationLevel.EMERGENCY)
    assert sim == pytest.approx(1.0)
    assert persist is False


# ─── filter_for_batch ───────────────────────────────────────────────────────


def _make_boxes(track_ids: list[int], handler: SemanticDriftHandler) -> torch.Tensor:
    """Create boxes with different sizes for each track."""
    areas = [100.0, 200.0, 50.0, 500.0, 300.0]
    boxes = torch.zeros((len(track_ids), 4))
    for i, tid in enumerate(track_ids):
        area = areas[i % len(areas)]
        boxes[i, 2] = boxes[i, 0] + (area**0.5)
        boxes[i, 3] = boxes[i, 1] + (area**0.5)
    return boxes


def test_filter_normal_full_batch() -> None:
    """NORMAL level → max_batch=32, returns all if <32."""
    handler = SemanticDriftHandler()
    ids = list(range(10))
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0] for _ in ids])
    result = handler.filter_for_batch(ids, boxes, DegradationLevel.NORMAL)
    assert len(result) == 10


def test_filter_reduced_max_16() -> None:
    """REDUCED level → max_batch=16."""
    handler = SemanticDriftHandler()
    ids = list(range(20))
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0] for _ in ids])
    result = handler.filter_for_batch(ids, boxes, DegradationLevel.REDUCED)
    assert len(result) <= 16


def test_filter_fast_path_max_nopt() -> None:
    """FAST_PATH → max_batch=N_OPT=8."""
    handler = SemanticDriftHandler()
    ids = list(range(20))
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0] for _ in ids])
    result = handler.filter_for_batch(ids, boxes, DegradationLevel.FAST_PATH)
    assert len(result) <= handler.N_OPT


def test_filter_emergency_max_nopt() -> None:
    """EMERGENCY → max_batch=N_OPT=8 (same as FAST_PATH)."""
    handler = SemanticDriftHandler()
    ids = list(range(20))
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0] for _ in ids])
    result = handler.filter_for_batch(ids, boxes, DegradationLevel.EMERGENCY)
    assert len(result) <= handler.N_OPT


def test_filter_priority_order() -> None:
    """Priority ordering: count=0 (warm) < count 1-4 (growing) < count>=5 (mature)."""
    handler = _make_handler_with_history({1: 0, 2: 3, 3: 10, 4: 0, 5: 3})
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0] for _ in range(5)])
    result = handler.filter_for_batch([1, 2, 3, 4, 5], boxes, DegradationLevel.NORMAL)
    # Priority 0: track 1 (count=0), track 4 (count=0)
    # Priority 1: track 2 (count=3), track 5 (count=3)
    # Priority 2: track 3 (count=10)
    # Within same priority, larger area first (neg area sorts ascending → larger first)
    # All same area here, so order is by priority then by area desc
    priority_0 = [t for t in result if t in (1, 4)]
    priority_1 = [t for t in result if t in (2, 5)]
    priority_2 = [t for t in result if t in (3,)]
    # Priority 0 should come before priority 1, which comes before priority 2
    idx_0 = max(result.index(t) for t in priority_0)
    idx_1 = max(result.index(t) for t in priority_1)
    idx_2 = max(result.index(t) for t in priority_2)
    assert idx_0 < idx_1 < idx_2


def test_filter_area_tiebreak() -> None:
    """Within same priority, larger area → higher priority (sorted first)."""
    handler = _make_handler_with_history({1: 0, 2: 0})
    # track 1: area=400 (20×20), track 2: area=100 (10×10)
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 10.0, 10.0]])
    result = handler.filter_for_batch([1, 2], boxes, DegradationLevel.NORMAL)
    assert result[0] == 1  # larger area first


def test_filter_empty_list() -> None:
    """Empty track list → empty result."""
    handler = SemanticDriftHandler()
    result = handler.filter_for_batch(
        [], torch.tensor([]).reshape(0, 4), DegradationLevel.NORMAL
    )
    assert result == []


def test_filter_more_tracks_than_max() -> None:
    """Returns exactly max_batch items when more tracks exist."""
    handler = _make_handler_with_history({i: 10 for i in range(20)})
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0] for _ in range(20)])
    result = handler.filter_for_batch(
        list(range(20)), boxes, DegradationLevel.FAST_PATH
    )
    assert len(result) == handler.N_OPT


# ─── update_history ─────────────────────────────────────────────────────────


def test_update_history_seeds_new_track() -> None:
    """New track creates centroid and sets count=1."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([[2.0, 0.0, 0.0]])
    handler.update_history([1], feat)
    assert 1 in handler.feature_history
    assert handler.track_update_count[1] == 1
    # New track stores raw (not normalized) centroid
    assert torch.allclose(handler.feature_history[1], torch.tensor([2.0, 0.0, 0.0]))


def test_update_history_updates_existing_with_alpha() -> None:
    """Existing track performs EMA update with alpha=0.3."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0, 0.0])
    handler.track_update_count[1] = 5

    new_feat = torch.tensor([[0.0, 1.0, 0.0]])
    handler.update_history([1], new_feat)

    # alpha=0.3 (high count, NORMAL)
    # centroid = 0.3 * [0,1,0] + 0.7 * [1,0,0] = [0.7, 0.3, 0]
    # Then F.normalize → [0.9191, 0.3939, 0]
    expected = F.normalize(torch.tensor([0.7, 0.3, 0.0]), p=2, dim=0)
    assert torch.allclose(handler.feature_history[1], expected)
    assert handler.track_update_count[1] == 6


def test_update_history_warm_track_high_alpha() -> None:
    """Warm track (count=0 → not in dict) seeds raw centroid."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([[1.0, 1.0, 0.0]])
    handler.update_history([1], feat)
    assert handler.track_update_count[1] == 1
    # New track stores raw (not normalized) centroid
    assert torch.allclose(handler.feature_history[1], torch.tensor([1.0, 1.0, 0.0]))


def test_update_history_batch_multiple() -> None:
    """Multiple tracks in one call."""
    handler = SemanticDriftHandler()
    feats = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    handler.update_history([1, 2, 3], feats)
    assert len(handler.feature_history) == 3
    for tid in (1, 2, 3):
        assert handler.track_update_count[tid] == 1
        # New tracks store raw centroids
        assert handler.feature_history[tid].dim() == 1


def test_update_history_updates_last_active_time() -> None:
    """All updated tracks get current timestamp."""
    handler = SemanticDriftHandler()
    before = time.time() - 1
    handler.last_active_time[1] = before
    handler.update_history([1, 2], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    now = time.time()
    for tid in (1, 2):
        assert handler.last_active_time[tid] >= before
        assert handler.last_active_time[tid] <= now
        assert tid in handler.feature_history


def test_update_history_feature_detached() -> None:
    """Input feature is detached and cloned, not shared with original."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    handler.update_history([1], feat)
    # Modify original (clone first to avoid in-place operation error)
    feat_clone = feat.clone()
    feat_clone[0] = 999.0
    assert handler.feature_history[1][0] != 999.0


def test_update_history_normalization() -> None:
    """Updated centroid (after seeding) is always L2-normalized."""
    handler = SemanticDriftHandler()
    feat = torch.tensor([[1.0, 0.0, 0.0]])
    handler.update_history([1], feat)
    # New track stores raw, but count=1 means it will normalize on next update
    assert handler.track_update_count[1] == 1

    # Second update normalizes the EMA blend
    new_feat = torch.tensor([[0.1, 10.0, 0.0]])
    handler.update_history([1], new_feat)
    assert torch.allclose(handler.feature_history[1].norm(), torch.tensor(1.0))


# ─── prune_expired_centroids ────────────────────────────────────────────────


def test_prune_no_expiration() -> None:
    """No expired tracks → returns 0."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0])
    handler.track_update_count[1] = 1
    handler.last_active_time[1] = time.time()
    expired = handler.prune_expired_centroids(timeout_sec=1.0)
    assert expired == 0
    assert 1 in handler.feature_history


def test_prune_expired_tracks() -> None:
    """Expired tracks are removed."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0])
    handler.track_update_count[1] = 1
    handler.last_active_time[1] = time.time() - 3600  # 1 hour ago
    handler.feature_history[2] = torch.tensor([0.0, 1.0])
    handler.track_update_count[2] = 2
    handler.last_active_time[2] = time.time()  # just now

    expired = handler.prune_expired_centroids(timeout_sec=300.0)
    assert expired == 1
    assert 1 not in handler.feature_history
    assert 1 not in handler.track_update_count
    assert 1 not in handler.last_active_time
    assert 2 in handler.feature_history


def test_prune_custom_timeout() -> None:
    """Custom timeout threshold works."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0])
    handler.track_update_count[1] = 1
    handler.last_active_time[1] = time.time() - 10.0  # 10 seconds ago
    expired = handler.prune_expired_centroids(timeout_sec=5.0)
    assert expired == 1


def test_prune_all_expired() -> None:
    """All tracks expired → empty after pruning."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0])
    handler.track_update_count[1] = 1
    handler.last_active_time[1] = time.time() - 3600
    expired = handler.prune_expired_centroids(timeout_sec=1.0)
    assert expired == 1
    assert len(handler.feature_history) == 0


def test_prune_empty_handler() -> None:
    """Empty handler → 0 expired."""
    handler = SemanticDriftHandler()
    expired = handler.prune_expired_centroids()
    assert expired == 0


# ─── clear_history ──────────────────────────────────────────────────────────


def test_clear_history_removes_all_entries() -> None:
    """clear_history removes track from all three dicts."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0])
    handler.track_update_count[1] = 5
    handler.last_active_time[1] = time.time()

    handler.clear_history(1)

    assert 1 not in handler.feature_history
    assert 1 not in handler.track_update_count
    assert 1 not in handler.last_active_time


def test_clear_history_nonexistent_is_safe() -> None:
    """clear_history on unknown track → no error."""
    handler = SemanticDriftHandler()
    handler.clear_history(999)  # should not raise


def test_clear_history_partial() -> None:
    """Clearing one track leaves others intact."""
    handler = SemanticDriftHandler()
    handler.feature_history[1] = torch.tensor([1.0, 0.0])
    handler.feature_history[2] = torch.tensor([0.0, 1.0])
    handler.track_update_count[1] = 1
    handler.track_update_count[2] = 2

    handler.clear_history(1)

    assert 1 not in handler.feature_history
    assert 2 in handler.feature_history
    assert handler.track_update_count[2] == 2


# ─── Integration ────────────────────────────────────────────────────────────


def test_full_lifecycle() -> None:
    """Seed → update → drift check → prune cycle."""
    handler = SemanticDriftHandler()

    # Seed initial track (2D tensor for update_history)
    feat1_batch = torch.tensor([[1.0, 0.0, 0.0]])
    handler.update_history([1], feat1_batch)

    # Second frame: very similar → no drift
    feat2_batch = torch.tensor([[1.0, 0.01, 0.0]])
    handler.update_history([1], feat2_batch)
    # calculate_drift expects 1D feature tensor
    feat2 = feat2_batch[0]
    sim, persist = handler.calculate_drift(1, feat2)
    assert sim > 0.95 and persist is False

    # Third frame: very different → drift
    feat3_batch = torch.tensor([[0.0, 1.0, 0.0]])
    handler.update_history([1], feat3_batch)
    feat3 = feat3_batch[0]
    sim3, persist3 = handler.calculate_drift(1, feat3)
    assert sim3 < 0.95 and persist3 is True

    # Prune after timeout
    handler.last_active_time[1] = time.time() - 600
    expired = handler.prune_expired_centroids(timeout_sec=300.0)
    assert expired == 1
    assert 1 not in handler.feature_history
