from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence/gap_conditioned_motion_phase_b_20260711/run_phase_b.py"
)
PACKET = RUNNER.parent


def _load():
    spec = importlib.util.spec_from_file_location("gap_motion_phase_b", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_packet_records_one_bounded_v5_verdict():
    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((PACKET / "phase_b_result.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "PHASE_B_EXECUTED"
    assert manifest["a1_a8_computed"] is True
    assert manifest["verdict"] == "V5"
    assert result["verdict"]["code"] == "V5"
    assert all(not member["boxes_pass"] for member in result["member_boxes"].values())
    assert len(result["a5_separability"]) > len(result["a2_role_reversal"])
    assert {
        "bridge_dist_auc",
        "speed_mismatch_auc",
        "dir_cos_auc",
        "resid_mean_auc",
    } <= set(result["a5_separability"][0])
    assert "q_motion_auc" in result["a3_short_gap_retention"][0]
    assert "bridge_dist_in_pooled_q90" in result["a4_escape_tail"][0]["pairs"][0]
    assert "a6_pooled_summary" in result
    assert "a6_threshold_link" in result["a7_loo_transfer"][0]
    assert {
        "training_nll",
        "log_det_growth_by_gap",
        "primary_calibration_classes",
    } <= set(result["a8_attribution"][0])


def test_role_firewall_rejects_held_out_contamination():
    runner = _load()
    # This must remain fixture-free: the canonical pair CSV is intentionally
    # not checked into CI, while the role firewall is validated before any
    # model score or research artifact is consumed.
    bad = runner.pd.DataFrame(
        [
            {
                "evaluation_role": "held_out",
                "seq": "SEQ-A",
                "held_out_sequence": "SEQ-A",
            },
            {
                "evaluation_role": "train",
                "seq": "SEQ-B",
                "held_out_sequence": "SEQ-B",
            },
        ]
    )

    with pytest.raises(ValueError, match="role firewall"):
        runner.compute(bad, {})


def test_tail_mutation_flips_the_frozen_role_reversal_criterion():
    runner = _load()
    frame = runner.pd.DataFrame(
        {
            "gt_match": [True, True, False, False],
            "seq": ["A", "A", "B", "B"],
            "nll_motion": [10.0, 9.0, 1.0, 0.0],
        }
    )
    assert runner.tail_cell(frame, "nll_motion")["role_reversal"] is True

    frame["nll_motion"] = [0.0, 1.0, 9.0, 10.0]
    assert runner.tail_cell(frame, "nll_motion")["role_reversal"] is False


def test_a3_a5_a6_and_a8_criteria_are_not_snapshot_only():
    runner = _load()
    assert runner.retention_pass(e_motion_auc=0.70, bridge_dist_auc=0.74, n_gt=15)
    assert not runner.retention_pass(e_motion_auc=0.68, bridge_dist_auc=0.74, n_gt=15)
    assert (
        runner.retention_pass(e_motion_auc=0.7, bridge_dist_auc=0.74, n_gt=14) is None
    )

    frame = runner.pd.DataFrame(
        {
            "gt_match": [True, True, False, False],
            "nll_motion": [0.0, 1.0, 4.0, 5.0],
            "q_motion": [0.1, 0.2, 0.8, 0.9],
            "bridge_dist": [0.0, 1.0, 4.0, 5.0],
            "lost_exit_speed": [1.0, 1.0, 4.0, 4.0],
            "cand_entry_speed": [1.0, 1.0, 1.0, 1.0],
            "dir_cos": [1.0, 1.0, 0.0, 0.0],
            "fwd_resid": [0.0, 1.0, 4.0, 5.0],
            "bwd_resid": [0.0, 1.0, 4.0, 5.0],
        }
    )
    a5 = runner.separability_row(
        frame, model_id="M1", support_layer="S_A", gap_cell="1-10"
    )
    assert {
        "e_motion_auc",
        "q_motion_auc",
        "bridge_dist_auc",
        "speed_mismatch_auc",
        "dir_cos_auc",
        "resid_mean_auc",
    } <= set(a5)
    assert a5["bridge_dist_auc"] == pytest.approx(1.0)
    assert a5["speed_mismatch_auc"] == pytest.approx(1.0)
    assert a5["dir_cos_auc"] == pytest.approx(1.0)
    assert a5["resid_mean_auc"] == pytest.approx(1.0)

    cells = [
        {
            "old": {"n_fp": 100, "fp_removed_count": 40},
            "new": {"n_fp": 100, "fp_removed_count": 40},
        },
        {
            "old": {"n_fp": 1, "fp_removed_count": 1},
            "new": {"n_fp": 1, "fp_removed_count": 0},
        },
    ]
    old = runner.pooled_fp_removed(cells, "old")
    new = runner.pooled_fp_removed(cells, "new")
    assert old["fp_removed"] == pytest.approx(41 / 101)
    assert new["fp_removed"] == pytest.approx(40 / 101)
    assert not runner.a6_no_thinner(
        safety=True,
        fold_pools=[{"old": old, "new": new}],
        global_old=old,
        global_new=new,
    )

    assert runner.m2_dominates(
        retention=True, held_out_nll_better=True, calibration_mismatch=False
    )
    assert not runner.m2_dominates(
        retention=True, held_out_nll_better=True, calibration_mismatch=True
    )


def test_verdict_partition_transitions_are_sealed_and_exhaustive():
    runner = _load()

    def members():
        return {
            model: {"boxes_pass": False, "dominates_m1": False}
            for model in runner.MODELS
        }

    assert runner.verdict_partition(members(), low_primary=True) == ("V4", None)

    v2 = members()
    v2["M2P-GLOBAL-OU-H270"] = {"boxes_pass": True, "dominates_m1": True}
    assert runner.verdict_partition(v2, low_primary=False) == ("V2", None)

    v1 = members()
    v1["M1P-GLOBAL-CV"]["boxes_pass"] = True
    assert runner.verdict_partition(v1, low_primary=False) == ("V1", None)

    v5 = members()
    assert runner.verdict_partition(v5, low_primary=False) == ("V5", None)

    anomaly = members()
    anomaly["M2P-GLOBAL-OU-H90"]["boxes_pass"] = True
    assert runner.verdict_partition(anomaly, low_primary=False) == (
        "V5",
        "a member passed all success boxes without a claimable verdict slot",
    )
