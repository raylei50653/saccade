"""Unit tests for scripts/tools/check_headline_decision_contract.py.

No GPU. Uses the pure ``check_presets`` helper plus the real headline YAMLs.
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


def test_main_cli_on_repo_presets(checker):
    rc = checker.main([])
    assert rc == 0
