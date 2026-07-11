"""Contract tests for the Boolean-atom partial-order audit packet (issue #106)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKET = (
    REPO / "docs/modules/semantic/research/evidence/boolean_atom_partial_order_20260711"
)
STEP0 = (
    REPO
    / "docs/modules/semantic/research/evidence/gt_support_morphology_step0_20260711"
)
PRC = REPO / "docs/modules/semantic/research/evidence/escape_tail_forensic_20260711"
PAIRS = REPO / "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"

ATOMS = [
    "score_m_bridge",
    "bridge_dist",
    "dist_h",
    "log_h_ratio",
    "resid_mean",
    "dir_cos",
    "speed_mismatch",
    "gap",
]
ROLES = {
    "global_orderable",
    "conditional_orderable",
    "context_only",
    "unresolved",
}
TERMINALS = {
    "GLOBAL_PARTIAL_ORDER_READY",
    "CONDITIONAL_STRUCTURE_ONLY",
    "ORDERABILITY_UNRESOLVED",
}
MOTION = {"speed_mismatch", "dir_cos", "resid_mean"}
# Nontrivial global subcube after demoting bridge_dist (PR #107 review).
GLOBAL = {"dist_h", "log_h_ratio"}
CONDITIONAL = MOTION | {"bridge_dist"}
CONTEXT_ONLY = {"score_m_bridge", "gap"}


def _load_runner() -> Any:
    path = PACKET / "run_partial_order_audit.py"
    spec = importlib.util.spec_from_file_location("run_partial_order_audit", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _j(name: str) -> dict[str, Any]:
    return json.loads((PACKET / name).read_text(encoding="utf-8"))


def test_packet_files_present() -> None:
    required = [
        "atom_roles.json",
        "atom_dependency_graph.json",
        "atom_metrics.csv",
        "pairwise_violation_profile.csv",
        "threshold_sensitivity.json",
        "allowed_global_order.json",
        "forbidden_order.json",
        "scale_guard.json",
        "aggregate.json",
        "manifest.json",
        "run_partial_order_audit.py",
    ]
    missing = [n for n in required if not (PACKET / n).is_file()]
    assert not missing, missing


def test_eight_atoms_exactly_one_role() -> None:
    roles = _j("atom_roles.json")["roles"]
    assert set(roles) == set(ATOMS)
    assert all(r in ROLES for r in roles.values())
    assert len(roles) == 8


def test_role_assignment_self_declared_research_judgment() -> None:
    roles_doc = _j("atom_roles.json")
    assert "research_judgment" in roles_doc["role_assignment_mode"]
    assert roles_doc["statistics_role"] == "descriptive_only"
    for name in ATOMS:
        card = roles_doc["cards"][name]
        assert card["role_assignment"] == "research_judgment"
        assert "global_admissibility" in card


def test_aggregate_terminal_and_global_set() -> None:
    agg = _j("aggregate.json")
    assert agg["terminal"] in TERMINALS
    assert agg["terminal"] == "GLOBAL_PARTIAL_ORDER_READY"
    assert agg["terminal_status"] == "accepted_with_limits"
    assert set(agg["global_atoms"]) == GLOBAL
    assert set(agg["conditional_atoms"]) == CONDITIONAL
    assert set(agg["context_only_atoms"]) == CONTEXT_ONLY
    assert agg["unresolved_atoms"] == []
    assert agg["prc_binding_respected"] is True
    assert agg["score_m_bridge_not_global"] is True
    assert agg["bridge_dist_not_global"] is True
    assert agg["authorizes_restricted_closure_prototype"] is True
    acc = agg["research_acceptance"]
    assert acc["status"] == "ACCEPTED_WITH_LIMITS"
    assert set(acc["accepted_global_atoms"]) == GLOBAL
    assert acc["accepted_roles"]["bridge_dist"] == "conditional_orderable"
    assert "bridge_dist" in acc["initial_global_atoms"]
    assert (
        "separate" in acc["restricted_closure"].lower()
        or "after_merge" in acc["restricted_closure"]
    )
    assert "dist_h" in acc["restricted_closure"]


def test_prc_motion_not_global() -> None:
    roles = _j("atom_roles.json")["roles"]
    for name in MOTION:
        assert roles[name] != "global_orderable"
        assert roles[name] == "conditional_orderable"


def test_bridge_dist_is_motion_extrapolation_not_global() -> None:
    """PR #107 review blocker: bridge_dist is not a parentless geometry leaf."""
    roles = _j("atom_roles.json")
    assert roles["roles"]["bridge_dist"] == "conditional_orderable"
    card = roles["cards"]["bridge_dist"]
    assert card["provenance"] == "motion_extrapolation_composite"
    assert card["global_admissibility"]["global_admissible"] is False
    assert any(
        "motion-extrapolation" in r
        for r in card["global_admissibility"]["block_reasons"]
    )

    dep = _j("atom_dependency_graph.json")
    node = dep["nodes"]["bridge_dist"]
    assert node["kind"] == "motion_extrapolation_composite"
    parents = set(node["parents"])
    assert "lost_exit_velocity" in parents
    assert "cand_entry_velocity" in parents
    assert "gap" in parents
    assert "h_ref" in parents
    assert node["motion_derived"] is True
    # Must not be recorded as a parentless builder_raw leaf.
    assert node["parents"]  # non-empty


def test_motion_derived_composites_cannot_silent_global_promote() -> None:
    """Executable guard: motion-derived composites fail global admissibility."""
    mod = _load_runner()
    dep = mod.dependency_graph()
    for name in ("bridge_dist", "score_m_bridge", *MOTION):
        check = mod.global_admissibility_check(name, dep)
        assert check["global_admissible"] is False, name
        assert check["block_reasons"], name
    # Pure structural / height leaves pass.
    for name in ("dist_h", "log_h_ratio"):
        check = mod.global_admissibility_check(name, dep)
        assert check["global_admissible"] is True, (name, check)


def test_score_m_bridge_scale_guard_blocks_without_unit_mismatch() -> None:
    sg = _j("scale_guard.json")
    assert sg["blocks_global_orderable"] is True
    assert sg["recompute_vs_sealed_max_abs"] == 0.0
    units = sg["unit_scale_compatibility"]
    assert units["compatible"] is True
    assert units["unit_incompatibility_is_block_reason"] is False
    # Block reasons must not claim px-vs-h unit mismatch.
    joined = " ".join(sg["block_reasons"]).lower()
    assert "px-like" not in joined
    assert "unit incompatibility" not in joined
    assert any("resid_mean" in r or "parent" in r for r in sg["block_reasons"])
    roles = _j("atom_roles.json")["roles"]
    assert roles["score_m_bridge"] == "context_only"


def test_allowed_forbidden_contract_complete() -> None:
    allowed = _j("allowed_global_order.json")
    forbidden = _j("forbidden_order.json")
    assert set(allowed["global_atoms"]) == GLOBAL
    assert "z_convention" in allowed
    forbidden_atoms = {d["atom"] for d in forbidden["forbidden_global_dimensions"]}
    assert forbidden_atoms == set(ATOMS) - set(allowed["global_atoms"])
    for name in CONDITIONAL | CONTEXT_ONLY:
        assert name in forbidden_atoms
    # Conditional proposals marked proposal-only (motion + bridge_dist).
    prop_atoms = {prop["atom"] for prop in forbidden["conditional_arc_proposals"]}
    assert prop_atoms == CONDITIONAL
    for prop in forbidden["conditional_arc_proposals"]:
        assert prop["status"] == "proposal-only"
    assert any("bridge_dist" in b for b in forbidden["hard_blocks"])


def test_dependency_graph_covers_composites() -> None:
    dep = _j("atom_dependency_graph.json")
    nodes = dep["nodes"]
    assert "score_m_bridge" in nodes
    parents = set(nodes["score_m_bridge"]["parents"])
    assert "resid_mean" in parents
    assert "dist_h" in parents
    assert "lost_exit_speed" in parents
    assert set(dep["frozen_atoms"]) == set(ATOMS)
    assert "bridge_dist" in dep["motion_derived_frozen_atoms"]


def test_dir_cos_parents_match_builder_formula() -> None:
    """dir_cos = cos(v_lost_exit, x_cand - x_lost); no cand entry velocity."""
    node = _j("atom_dependency_graph.json")["nodes"]["dir_cos"]
    parents = set(node["parents"])
    assert parents == {"lost_exit_velocity", "lost_foot_xy", "cand_foot_xy"}
    assert "cand_entry_velocity" not in parents
    assert "cosine" in node["transform"].lower() or "cos(" in node["transform"]


def test_residual_parents_are_formula_level() -> None:
    """fwd/bwd residuals are not parentless builder_raw leaves."""
    nodes = _j("atom_dependency_graph.json")["nodes"]
    fwd = set(nodes["fwd_resid"]["parents"])
    bwd = set(nodes["bwd_resid"]["parents"])
    assert fwd == {
        "lost_foot_xy",
        "cand_foot_xy",
        "lost_exit_velocity",
        "gap",
        "h_ref",
    }
    assert bwd == {
        "lost_foot_xy",
        "cand_foot_xy",
        "cand_entry_velocity",
        "gap",
        "h_ref",
    }
    assert nodes["fwd_resid"]["kind"] == "derived"
    assert nodes["bwd_resid"]["kind"] == "derived"
    # Residual role of resid_mean is unchanged (still conditional via PR-C).
    assert _j("atom_roles.json")["roles"]["resid_mean"] == "conditional_orderable"


def test_metrics_csv_rows() -> None:
    with (PACKET / "atom_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["atom"] for r in rows] == ATOMS
    assert all(r["role"] in ROLES for r in rows)
    assert all(int(r["n_tracks"]) == 209 for r in rows)
    by_atom = {r["atom"]: r for r in rows}
    assert by_atom["bridge_dist"]["role"] == "conditional_orderable"
    assert by_atom["dist_h"]["role"] == "global_orderable"


def test_manifest_seals_step0_source() -> None:
    man = _j("manifest.json")
    step0 = json.loads((STEP0 / "manifest.json").read_text(encoding="utf-8"))
    assert (
        man["depends_on"]["source_pairs_csv_sha256"] == step0["source_pairs_csv_sha256"]
    )
    assert man["atom_order"] == ATOMS
    assert man["aggregate_terminal"] == "GLOBAL_PARTIAL_ORDER_READY"
    assert set(man["files"]) >= {
        "atom_roles.json",
        "allowed_global_order.json",
        "forbidden_order.json",
        "aggregate.json",
    }


def test_manifest_file_digests_match_disk() -> None:
    man = _j("manifest.json")
    mod = _load_runner()
    for name, digest in man["files"].items():
        path = PACKET / name
        assert path.is_file(), name
        assert mod.sha256(path) == digest, name


def test_prc_aggregate_binding_present() -> None:
    prc = json.loads((PRC / "aggregate.json").read_text(encoding="utf-8"))
    assert prc["terminal"] == "ROLE_REVERSAL_SUPPORTED"
    assert prc["research_acceptance"]["status"] == "ACCEPTED_WITH_LIMITS"


@pytest.mark.skipif(not PAIRS.is_file(), reason="sealed pairs.csv not present")
def test_emit_verify_reproducible() -> None:
    mod = _load_runner()
    mod.verify(PAIRS)
