"""Focused contract tests for R1 RegionAsset converter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/tools/convert_safe_region_asset_r1.py"
PACK = REPO / "out/signal_study/m_b1_5_safe_region_asset_r1_20260710"
Q45 = REPO / "out/signal_study/m_b1_5_stage2_q45_20260710"
EVENTS = REPO / "out/signal_study/m_b1_5_stage2_q1q3_20260710/d_online_events.parquet"


def _load_mod():
    import sys

    name = "convert_safe_region_asset_r1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve module namespace.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_mod()


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_canonical_json_and_content_id_stable(mod) -> None:
    a = {"b": 1, "a": [2, 3]}
    b = {"a": [2, 3], "b": 1}
    assert mod.canonical_json(a) == mod.canonical_json(b)
    assert mod.content_id(a) == mod.content_id(b)
    assert len(mod.content_id(a)) == 64


def test_pairwise_leaf_swap_invariance(mod) -> None:
    left = mod.canonicalize_pairwise_leaf(
        "score_m_bridge",
        "high_tail",
        3,
        "P::score_m_bridge::high_tail::q3",
        "abs_log_h",
        "low_tail",
        5,
        "P::abs_log_h::low_tail::q5",
    )
    right = mod.canonicalize_pairwise_leaf(
        "abs_log_h",
        "low_tail",
        5,
        "P::abs_log_h::low_tail::q5",
        "score_m_bridge",
        "high_tail",
        3,
        "P::score_m_bridge::high_tail::q3",
    )
    assert left == right
    assert left["feature_0"] == "abs_log_h"
    assert left["feature_1"] == "score_m_bridge"
    assert left["thr_index_0"] == 5
    assert left["thr_index_1"] == 3


def test_pairwise_axis_pair_count(mod) -> None:
    pairs = mod.pairwise_axis_pairs()
    assert len(pairs) == 40
    # all leaves ordered
    for fa, da, fb, db in pairs:
        assert (
            (fa, da) < (fb, db)
            or ((fa, da) == (fb, db) and False)
            or (fa, da) <= (fb, db)
        )
        assert fa != fb  # distinct features only


def test_claim_level_derivation(mod) -> None:
    assert mod.derive_claim_level(1, "isolated_point") == "L0"
    assert mod.derive_claim_level(2, "row_strip") == "L1"
    assert mod.derive_claim_level(19, "2d_region") == "L1"


def test_thr_value_repr_roundtrip(mod) -> None:
    for v in (0.0, 0.5, 0.028840784914791584, 1.25):
        s = mod.thr_value_repr(v)
        assert float(s) == float(v) or abs(float(s) - float(v)) == 0.0


@pytest.mark.skipif(
    not Q45.is_dir() or not EVENTS.is_file(), reason="runtime atlases absent"
)
def test_preflight_ok(mod) -> None:
    rep = mod.preflight(mod.default_paths())
    assert rep["status"] == "OK", rep.get("blocking")
    assert rep["verified_seals"]


@pytest.mark.skipif(not PACK.is_dir(), reason="R1 pack not generated yet")
def test_pack_validation_and_firewalls(mod) -> None:
    rep = mod.validate_pack(PACK)
    assert rep["ok"], rep["errors"]
    assert rep["counts"]["regions"] == 26
    assert rep["counts"]["membership"] == 154
    assert rep["counts"]["g1_L0"] == 1
    assert rep["counts"]["g2_L0"] == 6
    assert rep["counts"]["g2_L1"] == 19
    assert rep["counts"]["nulls"] == 1
    # RB8: membership digest ≠ source event table sha
    assert rep["universe_membership_digest"] != rep["source_event_table_sha256"]

    man = json.loads((PACK / "region_asset_manifest.json").read_text(encoding="utf-8"))
    assert man["maturity_declared"] == "A0"
    assert man["composition_level"] == "observational"
    assert man["production_forbidden"] is True
    assert man["terminal_letter"] == "B"
    assert man["pack_claim_ceiling"] == "L1"
    assert man["review_status"] == "A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW"

    feas = json.loads((PACK / "feasibility_contract.json").read_text(encoding="utf-8"))
    assert "claim_level" not in feas

    uni = json.loads(
        (PACK / "candidate_universe_instances.json").read_text(encoding="utf-8")
    )[0]
    assert uni["membership_status"] == "SEALED"
    assert uni["universe_membership_digest"] == uni["universe_hash"]

    nulls = (PACK / "null_records.csv").read_text(encoding="utf-8")
    assert "G3_or" in nulls


@pytest.mark.skipif(not PACK.is_dir(), reason="R1 pack not generated yet")
def test_e1_contract_non_implications() -> None:
    text = (REPO / "docs/research/eval/safe_region_asset_contract.md").read_text(
        encoding="utf-8"
    )
    assert "**Status:** **ACCEPTED**" in text
    assert "generator-contract equality ⇏" in text
    assert "source_event_table_sha256 ⇏" in text
    assert "policy family ⇏" in text
    assert "thr_index without registry ⇏" in text
