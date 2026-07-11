"""Contract tests for the Boolean-atom partial-order audit packet (issue #106)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
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
    # Exactly one role each (dict values already unique per key).
    assert len(roles) == 8


def test_aggregate_terminal_and_global_set() -> None:
    agg = _j("aggregate.json")
    assert agg["terminal"] in TERMINALS
    assert agg["terminal"] == "GLOBAL_PARTIAL_ORDER_READY"
    assert set(agg["global_atoms"]) == {"bridge_dist", "dist_h", "log_h_ratio"}
    assert set(agg["conditional_atoms"]) == MOTION
    assert set(agg["context_only_atoms"]) == {"score_m_bridge", "gap"}
    assert agg["unresolved_atoms"] == []
    assert agg["prc_binding_respected"] is True
    assert agg["score_m_bridge_not_global"] is True
    assert agg["authorizes_restricted_closure_prototype"] is True


def test_prc_motion_not_global() -> None:
    roles = _j("atom_roles.json")["roles"]
    for name in MOTION:
        assert roles[name] != "global_orderable"
        assert roles[name] == "conditional_orderable"


def test_score_m_bridge_scale_guard_blocks_global() -> None:
    sg = _j("scale_guard.json")
    assert sg["blocks_global_orderable"] is True
    assert sg["recompute_vs_sealed_max_abs"] == 0.0
    roles = _j("atom_roles.json")["roles"]
    assert roles["score_m_bridge"] == "context_only"


def test_allowed_forbidden_contract_complete() -> None:
    allowed = _j("allowed_global_order.json")
    forbidden = _j("forbidden_order.json")
    assert set(allowed["global_atoms"]) == {"bridge_dist", "dist_h", "log_h_ratio"}
    assert "z_convention" in allowed
    forbidden_atoms = {d["atom"] for d in forbidden["forbidden_global_dimensions"]}
    assert forbidden_atoms == set(ATOMS) - set(allowed["global_atoms"])
    for name in MOTION:
        assert name in forbidden_atoms
    assert "score_m_bridge" in forbidden_atoms
    assert "gap" in forbidden_atoms
    # Conditional proposals marked proposal-only.
    for prop in forbidden["conditional_arc_proposals"]:
        assert prop["status"] == "proposal-only"
        assert prop["atom"] in MOTION


def test_dependency_graph_covers_composites() -> None:
    dep = _j("atom_dependency_graph.json")
    nodes = dep["nodes"]
    assert "score_m_bridge" in nodes
    parents = set(nodes["score_m_bridge"]["parents"])
    assert "resid_mean" in parents
    assert "dist_h" in parents
    assert "lost_exit_speed" in parents
    assert set(dep["frozen_atoms"]) == set(ATOMS)


def test_metrics_csv_rows() -> None:
    with (PACKET / "atom_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["atom"] for r in rows] == ATOMS
    assert all(r["role"] in ROLES for r in rows)
    assert all(int(r["n_tracks"]) == 209 for r in rows)


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
