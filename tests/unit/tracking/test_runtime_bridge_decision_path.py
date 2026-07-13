"""P0 must fail closed before outcome access when capture provenance is foreign."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/tools/audit_runtime_bridge_decision_path.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p0_audit", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0_detects_foreign_m_capture_before_label_access() -> None:
    runner = _load_runner()
    result = runner.audit(ROOT)

    assert result["terminal"] == "P0_CAPTURE_SEMANTICS_INVALID"
    assert result["label_access"] == {
        "gt_or_fp_labels_accessed": False,
        "p5": "not_entered",
    }
    assert result["provenance"]["d0_alignment"]["all_fields_match"] is False
    assert result["provenance"]["r1_frozen_preset"].endswith("mamba_whole_graph_m.yaml")


def test_p0_keeps_candidate_and_commit_replay_below_l2() -> None:
    runner = _load_runner()
    result = runner.audit(ROOT)
    matrix = {row["stage"]: row for row in result["field_sufficiency"]}

    assert matrix["D_pair_cutoff"]["complete"] is False
    assert matrix["E_candidate_local_ranking"]["complete"] is False
    assert matrix["F_claim_competition"]["complete"] is False
    assert matrix["G_commit"]["complete"] is False
    assert (
        result["replay"]["counterfactual_ceiling_if_headline_alignment_existed"]
        == "L1_pair_cutoff_replay"
    )
