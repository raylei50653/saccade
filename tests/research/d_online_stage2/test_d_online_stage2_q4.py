"""Unit tests for M-B1.5 Stage 2 Q4 separability audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from saccade.perception.eval.d_online_stage2_q4 import (
    classify_q4_terminal,
    evaluate_feature,
    lock_q4_cohort,
    make_q4_primary_row,
    pure_negative_tail_audit,
    reconcile_q4,
    sibling_feature_matrix,
    stability_flags,
    per_sequence_feature_rows,
    loo_sequence_feature_rows,
    _pair_auc_and_cliff,
)

Q1Q3 = Path("out/signal_study/m_b1_5_stage2_q1q3_20260710")


def test_cohort_excludes_unresolved_and_non_selected() -> None:
    rows = [
        {
            **make_q4_primary_row(
                event_id="n1",
                sequence="MOT17-02-SDP",
                pair_label="negative",
                score_m_bridge=0.3,
            ),
        },
        {
            **make_q4_primary_row(
                event_id="g1",
                sequence="MOT17-02-SDP",
                pair_label="gt_consistent",
                score_m_bridge=0.1,
            ),
        },
        {
            "event_id": "u1",
            "join_key": "u1",
            "sequence": "MOT17-02-SDP",
            "frame": 1,
            "label_status": "unresolved",
            "pair_label": "unknown",
            "baseline_selected": 1,
            "score_m_bridge": 0.2,
            "abs_log_h": 0.1,
            "dist_h": 0.1,
            "abs_ratio_m1": 0.1,
            "resid_mean": 0.1,
            "competitor_count": 0,
        },
        {
            "event_id": "ns1",
            "join_key": "ns1",
            "sequence": "MOT17-02-SDP",
            "frame": 1,
            "label_status": "resolved",
            "pair_label": "negative",
            "baseline_selected": 0,
            "score_m_bridge": 0.5,
            "abs_log_h": 0.1,
            "dist_h": 0.1,
            "abs_ratio_m1": 0.1,
            "resid_mean": 0.1,
            "competitor_count": 1,
        },
    ]
    locked = lock_q4_cohort(rows)
    assert locked["summary"]["n_primary"] == 2
    assert locked["summary"]["n_primary_negative"] == 1
    assert locked["summary"]["n_primary_positive_protect"] == 1
    assert locked["summary"]["n_secondary_non_selected"] == 1
    assert locked["summary"]["n_excluded"] == 1
    # non-selected not in primary
    assert all(r["event_id"] != "ns1" for r in locked["primary"])
    assert all(r["event_id"] != "u1" for r in locked["primary"])


def test_auc_perfect_separation() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    x = np.array([3.0, 4.0, 5.0, 0.0, 1.0, 2.0])  # higher in neg
    auc, oriented, cliff, direction = _pair_auc_and_cliff(y, x)
    assert auc == pytest.approx(1.0)
    assert oriented == pytest.approx(1.0)
    assert cliff == pytest.approx(1.0)
    assert direction == "higher_in_negative"


def test_auc_reversed() -> None:
    y = np.array([1, 1, 0, 0])
    x = np.array([0.0, 1.0, 3.0, 4.0])  # lower in neg
    auc, oriented, cliff, direction = _pair_auc_and_cliff(y, x)
    assert auc == pytest.approx(0.0)
    assert oriented == pytest.approx(1.0)
    assert direction == "lower_in_negative"


def test_pure_tail_descriptive_not_rule() -> None:
    y = np.array([1, 1, 1, 0, 0])
    x = np.array([9.0, 8.0, 7.0, 1.0, 0.0])
    seqs = np.array(["A", "B", "A", "A", "B"], dtype=object)
    t = pure_negative_tail_audit(y, x, seqs)
    assert t["from_high"]["pure_negative_prefix_n"] == 3
    assert t["from_high"]["not_a_rule_or_candidate"] is True
    assert t["from_high"]["claim_status"] == "descriptive_pure_neg_tail"


def test_sibling_transforms_no_new_sources() -> None:
    rows = [
        make_q4_primary_row(
            event_id="a",
            sequence="S",
            pair_label="negative",
            score_m_bridge=0.2,
        )
    ]
    feats = sibling_feature_matrix(rows)
    assert "score_m_bridge" in feats
    assert "log1p__score_m_bridge" in feats
    assert "margin_to_online_bridge_gate" in feats
    assert "resid_over_dist_h" in feats
    # no arbitrary learned combo names
    assert "boolean_or" not in feats


def test_terminal_d_insufficient_mass() -> None:
    cohort = {
        "n_primary_negative": 2,
        "n_primary_positive_protect": 2,
    }
    t = classify_q4_terminal(
        cohort_summary=cohort,
        pooled_features=[],
        stability_by_feature={},
        slice_rows=[],
    )
    assert t["stage2_q4_separability"] == "insufficient_labeled_decision_mass"
    assert t["terminal_letter"] == "D"


def test_terminal_a_strong_stable() -> None:
    pooled = [
        {
            "feature": "score_m_bridge",
            "auc_oriented": 0.85,
            "effect_band": "strong",
            "pure_tail": {
                "from_high": {
                    "pure_negative_prefix_n": 5,
                    "pure_negative_prefix_n_sequences": 3,
                },
                "from_low": {
                    "pure_negative_prefix_n": 0,
                    "pure_negative_prefix_n_sequences": 0,
                },
            },
        }
    ]
    stab = {
        "score_m_bridge": {
            "stable_candidate": True,
            "loo_direction_flip": False,
        }
    }
    t = classify_q4_terminal(
        cohort_summary={
            "n_primary_negative": 20,
            "n_primary_positive_protect": 40,
        },
        pooled_features=pooled,
        stability_by_feature=stab,
        slice_rows=[],
    )
    assert t["terminal_letter"] == "A"
    assert "restricted_safe_region_modeling_authorized" in t["claims_allowed"]


def test_terminal_c_weak() -> None:
    pooled = [
        {
            "feature": sig,
            "auc_oriented": 0.55,
            "effect_band": "negligible",
            "pure_tail": {
                "from_high": {
                    "pure_negative_prefix_n": 0,
                    "pure_negative_prefix_n_sequences": 0,
                },
                "from_low": {
                    "pure_negative_prefix_n": 0,
                    "pure_negative_prefix_n_sequences": 0,
                },
            },
        }
        for sig in (
            "score_m_bridge",
            "abs_log_h",
            "dist_h",
            "abs_ratio_m1",
            "resid_mean",
        )
    ]
    stab = {p["feature"]: {"stable_candidate": False} for p in pooled}
    t = classify_q4_terminal(
        cohort_summary={
            "n_primary_negative": 23,
            "n_primary_positive_protect": 64,
        },
        pooled_features=pooled,
        stability_by_feature=stab,
        slice_rows=[],
    )
    assert t["terminal_letter"] == "C"
    assert t["stage2_q4_separability"] == "separability_weak_or_unstable"
    assert "safe_region_modeling_not_authorized" in t["claims_blocked"]
    assert "threshold_search_not_authorized" in t["claims_blocked"]


def test_terminal_b_conditional() -> None:
    pooled = [
        {
            "feature": "score_m_bridge",
            "auc_oriented": 0.55,
            "effect_band": "weak",
            "pure_tail": {
                "from_high": {
                    "pure_negative_prefix_n": 0,
                    "pure_negative_prefix_n_sequences": 0,
                },
                "from_low": {
                    "pure_negative_prefix_n": 0,
                    "pure_negative_prefix_n_sequences": 0,
                },
            },
        }
    ]
    for sig in ("abs_log_h", "dist_h", "abs_ratio_m1", "resid_mean"):
        pooled.append(
            {
                "feature": sig,
                "auc_oriented": 0.52,
                "effect_band": "negligible",
                "pure_tail": {
                    "from_high": {
                        "pure_negative_prefix_n": 0,
                        "pure_negative_prefix_n_sequences": 0,
                    },
                    "from_low": {
                        "pure_negative_prefix_n": 0,
                        "pure_negative_prefix_n_sequences": 0,
                    },
                },
            }
        )
    slice_rows = [
        {
            "slice": "bdist_gt_half_gate",
            "feature": "abs_log_h",
            "stable_candidate": True,
            "effect_band": "strong",
            "n_sequences_with_both_classes": 4,
            "coverage": 0.4,
            "auc_oriented": 0.8,
            "pure_tail_high": 4,
            "pure_tail_low": 0,
        }
    ]
    t = classify_q4_terminal(
        cohort_summary={
            "n_primary_negative": 20,
            "n_primary_positive_protect": 40,
        },
        pooled_features=pooled,
        stability_by_feature={
            p["feature"]: {"stable_candidate": False} for p in pooled
        },
        slice_rows=slice_rows,
    )
    assert t["terminal_letter"] == "B"


def test_recon_partition() -> None:
    rows = [
        make_q4_primary_row(
            event_id="n1",
            sequence="A",
            pair_label="negative",
            score_m_bridge=0.3,
        ),
        make_q4_primary_row(
            event_id="g1",
            sequence="A",
            pair_label="gt_consistent",
            score_m_bridge=0.1,
        ),
        {
            "event_id": "ns",
            "join_key": "ns",
            "sequence": "B",
            "frame": 1,
            "label_status": "resolved",
            "pair_label": "negative",
            "baseline_selected": 0,
            "score_m_bridge": 0.2,
            "abs_log_h": 0.1,
            "dist_h": 0.1,
            "abs_ratio_m1": 0.1,
            "resid_mean": 0.1,
            "competitor_count": 0,
        },
    ]
    locked = lock_q4_cohort(rows)
    recon = reconcile_q4(
        n_d_online=len(rows),
        cohort=locked["summary"],
        primary=locked["primary"],
    )
    assert recon["ok"] is True


def test_loo_and_per_seq_run() -> None:
    # two seqs with both classes
    y = np.array([1, 0, 1, 0, 1, 0])
    x = np.array([1.0, 0.0, 1.1, 0.1, 0.9, 0.2])
    seqs = np.array(["A", "A", "B", "B", "C", "C"], dtype=object)
    per = per_sequence_feature_rows("x", x, y, seqs)
    assert len(per) == 3
    loo = loo_sequence_feature_rows("x", x, y, seqs)
    assert len(loo) == 3
    pooled = evaluate_feature("x", x, y, seqs)
    stab = stability_flags(pooled, per, loo)
    assert "loo_direction_flip" in stab


@pytest.mark.skipif(
    not (Q1Q3 / "d_online_events.parquet").is_file(),
    reason="Q1–Q3 study missing",
)
def test_authoritative_q4_smoke(tmp_path: Path) -> None:
    from saccade.perception.eval.d_online_stage2_q4 import run_stage2_q4_audit

    out = tmp_path / "q4"
    summary = run_stage2_q4_audit(
        q1q3_study_dir=Q1Q3,
        out_dir=out,
        git_commit="test",
        study_id="test_q4",
    )
    assert summary["D_online_total"] == 244
    assert summary["n_primary_negative"] == 23
    assert summary["n_primary_positive_protect"] == 64
    assert summary["reconciliation_acceptance"] == "PASS"
    assert summary["stage2_q4_separability"] in {
        "single_signal_separability_supported",
        "conditional_separability_supported",
        "separability_weak_or_unstable",
        "insufficient_labeled_decision_mass",
    }
    assert summary["production_preset"] == "unchanged"
    assert "threshold_search_not_authorized" in summary["claims_blocked"]
    assert (out / "q4_cohort.csv").is_file()
    assert (out / "q4_signal_separability.csv").is_file()
    assert (out / "q4_loo.csv").is_file()
    assert (out / "manifest.json").is_file()
    # primary must not mix non-selected
    import csv

    with (out / "q4_cohort.csv").open() as f:
        for row in csv.DictReader(f):
            assert int(row["baseline_selected"]) == 1
            assert row["pair_label"] in ("negative", "gt_consistent")
