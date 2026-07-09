"""Unit tests for scripts/tools/check_headline_decision_contract.py.

No GPU. Uses pure ``check_presets`` / ``check_inject_map`` helpers plus live
repo presets and inject sources.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "tools" / "check_headline_decision_contract.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_headline_decision_contract", _SCRIPT
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _minimal_ok_pair(checker) -> tuple[dict, dict]:
    """Build a synthetic s/m pair that satisfies the contract tables."""
    shared = {
        **checker.SHARED_EXPECTED,
        "fuse_score_weight": 0.0,
        "person_geometry_prior": False,
        "detection_quality_scaling": False,
        "geometry_suspect_support": False,
        "id_stability_filter": False,
        "track_person_only": False,
        "per_seq_adapt": False,
    }
    s = {
        **shared,
        "kalman_r_scale": 2.8,
        "relink_bridge_px": 0.25,
        "relink_bridge_h_lo": 0.75,
        "relink_bridge_h_hi": 1.33,
        "relink_bridge_dir_bonus": 0.8,
    }
    m = {
        **shared,
        "kalman_r_scale": 3.5,
        "relink_bridge_px": 0.4,
        "relink_bridge_h_lo": 0.6,
        "relink_bridge_h_hi": 1.7,
        "relink_bridge_dir_bonus": 0.0,
    }
    return s, m


def _minimal_ok_inject_sources() -> tuple[str, str, str]:
    """Synthetic sources that satisfy C8 patterns."""
    pipeline = """
detector.tracker.set_params(
    match_thresh=cfg.core.match_thresh,
    r_scale=cfg.geometry.kalman_r_scale,
)
detector.tracker.set_occ_params(
    enabled=cfg.geometry.occ_state_enabled,
    iou_thresh=cfg.geometry.occ_iou_thresh,
    foot_gap=cfg.geometry.occ_foot_gap,
    ttl=cfg.geometry.occ_ttl,
    cost_weight=cfg.geometry.occ_cost_weight,
)
detector.tracker.set_relink_params(...)
detector.tracker.set_multiplicative_cost(enabled=True)
setter = getattr(detector.tracker, "set_stability_cost_w", None)
"""
    filters = """
def _append_private_continuation_candidates(
    ...,
    frame_new_track_thresh: float,
):
    birth_ceiling = float(frame_new_track_thresh) - score_eps
"""
    tracker = """
def set_params(
    self,
    match_thresh: float = 0.8,
    r_scale: float = 1.0,
) -> None:
    self.tracker.set_params(..., r_scale)
"""
    return pipeline, filters, tracker


def test_synthetic_ok(checker):
    s, m = _minimal_ok_pair(checker)
    failures, notes = checker.check_presets(s, m)
    assert failures == []
    assert any("C6 dual stability" in n for n in notes)


def test_occ_state_missing_fails(checker):
    s, m = _minimal_ok_pair(checker)
    del s["occ_state_enabled"]
    failures, _ = checker.check_presets(s, m)
    assert any("occ_state_enabled" in f for f in failures)


def test_occ_state_off_fails(checker):
    s, m = _minimal_ok_pair(checker)
    s["occ_state_enabled"] = False
    m["occ_state_enabled"] = False
    failures, _ = checker.check_presets(s, m)
    assert any("occ_state_enabled" in f and "C1" in f for f in failures)


def test_shared_key_mismatch_fails(checker):
    s, m = _minimal_ok_pair(checker)
    m["match_thresh"] = 0.99
    failures, _ = checker.check_presets(s, m)
    assert any("match_thresh" in f and "C2" in f for f in failures)


def test_m_delta_regression_fails(checker):
    s, m = _minimal_ok_pair(checker)
    m["kalman_r_scale"] = 2.8  # collapsed to s
    failures, _ = checker.check_presets(s, m)
    assert any("kalman_r_scale" in f and "C3" in f for f in failures)


def test_dir_bonus_m_must_be_explicit_zero(checker):
    s, m = _minimal_ok_pair(checker)
    m["relink_bridge_dir_bonus"] = 0.8
    failures, _ = checker.check_presets(s, m)
    assert any("relink_bridge_dir_bonus" in f for f in failures)


def test_no_go_fuse_score_fails(checker):
    s, m = _minimal_ok_pair(checker)
    s["fuse_score_weight"] = 0.5
    m["fuse_score_weight"] = 0.5
    failures, _ = checker.check_presets(s, m)
    assert any("fuse_score_weight" in f and "C7" in f for f in failures)


def test_private_continuation_off_fails(checker):
    s, m = _minimal_ok_pair(checker)
    s["private_continuation_enabled"] = False
    m["private_continuation_enabled"] = False
    failures, _ = checker.check_presets(s, m)
    assert any("private_continuation" in f for f in failures)


def test_real_headline_presets_pass(checker):
    """Regression: production YAMLs must satisfy the locked contract."""
    s = checker.load_preset(checker.PRESET_S)
    m = checker.load_preset(checker.PRESET_M)
    failures, notes = checker.check_presets(s, m)
    assert failures == [], failures
    assert notes  # dual-stability NOTE always emitted


def test_inject_map_synthetic_ok(checker):
    pipeline, filters, tracker = _minimal_ok_inject_sources()
    failures, notes = checker.check_inject_map(pipeline, filters, tracker)
    assert failures == [], failures
    assert any("C8 inject map" in n for n in notes)


def test_inject_map_missing_set_occ_fails(checker):
    pipeline, filters, tracker = _minimal_ok_inject_sources()
    pipeline = pipeline.replace("set_occ_params", "set_occ_REMOVED")
    failures, _ = checker.check_inject_map(pipeline, filters, tracker)
    assert any("set_occ_params" in f and "C8" in f for f in failures)


def test_inject_map_missing_r_scale_remap_fails(checker):
    pipeline, filters, tracker = _minimal_ok_inject_sources()
    pipeline = pipeline.replace("r_scale=cfg.geometry.kalman_r_scale", "r_scale=1.0")
    failures, _ = checker.check_inject_map(pipeline, filters, tracker)
    assert any("kalman_r_scale" in f and "C8" in f for f in failures)


def test_inject_map_private_must_stay_det_set(checker):
    pipeline, filters, tracker = _minimal_ok_inject_sources()
    filters = filters.replace(
        "def _append_private_continuation_candidates",
        "def _append_other",
    )
    failures, _ = checker.check_inject_map(pipeline, filters, tracker)
    assert any("private" in f.lower() and "C8" in f for f in failures)


def test_inject_map_rejects_tracker_private_setter(checker):
    pipeline, filters, tracker = _minimal_ok_inject_sources()
    tracker = tracker + "\ndef set_private_continuation(self, enabled: bool): ...\n"
    failures, _ = checker.check_inject_map(pipeline, filters, tracker)
    assert any("private" in f.lower() and "C8" in f for f in failures)


def test_real_inject_map_pass(checker):
    failures, notes = checker.check_inject_map_from_repo()
    assert failures == [], failures
    assert any("C8" in n for n in notes)


def test_main_cli_on_repo_presets(checker):
    rc = checker.main([])
    assert rc == 0
