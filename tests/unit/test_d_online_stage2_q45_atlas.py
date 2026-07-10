"""Unit tests for M-B1.5 Stage 2 Q4.5 threshold-combination atlas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from saccade.perception.eval.d_online_stage2_q45_atlas import (
    TERMINAL_A,
    TERMINAL_B,
    TERMINAL_C,
    TERMINAL_D,
    atom_mask,
    build_pareto_frontier,
    build_single_atom_atlas,
    build_threshold_registry,
    check_frame_column_provenance,
    classify_q45_terminal,
    classify_region_stability,
    region_metrics,
    true_holdout_sequence_validation,
)
from saccade.perception.eval.d_online_stage2_q4 import make_q4_primary_row

Q1Q3 = Path("out/signal_study/m_b1_5_stage2_q1q3_20260710")
STAGE1 = Path("out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close")


def test_atom_mask_high_low() -> None:
    x = np.array([0.1, 0.5, 0.9, np.nan])
    assert atom_mask(x, "high_tail", 0.5).tolist() == [False, True, True, False]
    assert atom_mask(x, "low_tail", 0.5).tolist() == [True, True, False, False]


def test_region_metrics_productive_safe() -> None:
    y = np.array([1, 1, 0, 0, 0])
    seqs = np.array(["A", "B", "A", "B", "C"], dtype=object)
    mask = np.array([True, True, False, False, False])
    m = region_metrics(mask, y, seqs, base_neg_rate=0.4)
    assert m["n_neg_captured"] == 2
    assert m["gt_hurt"] == 0
    assert m["productive_safe_point"] is True
    assert m["not_a_safe_rule"] is True
    assert m["n_sequences_with_neg"] == 2
    assert m["loo_not_portability_evidence"] is True
    assert m["loo_is_deletion_consistency_only"] is True


def test_unresolved_contamination_blocks_productive_safe() -> None:
    y = np.array([1, 1, 0, 0])
    seqs = np.array(["A", "B", "A", "B"], dtype=object)
    mask = np.array([True, True, False, False])
    unknown = np.array([True, False])  # one unresolved selected in region
    m = region_metrics(mask, y, seqs, base_neg_rate=0.5, unknown_mask=unknown)
    assert m["n_unresolved_selected"] == 1
    assert m["safety_status"] == "unresolved_contaminated"
    assert m["productive_safe_point"] is False
    assert m["productive_safe_resolved_only"] is True
    assert m["pessimistic_gt_hurt"] >= 1


def test_true_holdout_can_fail_when_deletion_loo_passes() -> None:
    # Full cohort: high values are pure neg → productive-safe on full data
    # Holdout A: only GT above thr → true holdout GT hurt > 0
    x = np.array([0.1, 0.2, 0.9, 0.95, 0.15, 0.12])
    y = np.array([0, 0, 1, 1, 0, 0])  # high = neg on B; A has only lows as GT
    seqs = np.array(["A", "A", "B", "B", "C", "C"], dtype=object)
    # thr such that only 0.9, 0.95 fire — both on B (neg)
    mask = atom_mask(x, "high_tail", 0.85)
    m = region_metrics(mask, y, seqs, base_neg_rate=1 / 3)
    assert m["gt_hurt"] == 0
    assert m["leave_one_sequence_deleted_all_gt_hurt_zero"] is True
    # Inject a holdout-specific GT high: change y for evaluation on true holdout
    # Construct case: thr 0.5 captures GT on A when held out
    x2 = np.array([0.8, 0.1, 0.9, 0.2])
    y2 = np.array([0, 1, 1, 0])  # A has GT at 0.8
    s2 = np.array(["A", "B", "B", "C"], dtype=object)
    ho = true_holdout_sequence_validation(
        x_parts={"score_m_bridge": x2},
        y=y2,
        sequences=s2,
        feature="score_m_bridge",
        direction="high_tail",
        thr_value=0.5,
    )
    # holdout A: x=0.8 GT → gt_hurt 1
    a_row = next(r for r in ho["true_holdout_rows"] if r["hold_out_sequence"] == "A")
    assert a_row["holdout_gt_hurt"] == 1
    assert ho["true_holdout_all_gt_hurt_zero"] is False


def test_duplicate_masks_do_not_form_thick_region() -> None:
    # two thr indices, same mask signature → only one unique topology node
    sig = "abc123"
    atoms = [
        {
            "atom_id": "S::f::high::u0",
            "feature": "score_m_bridge",
            "direction": "high_tail",
            "thr_index": 0,
            "thr_value": 0.1,
            "productive_safe_point": 1,
            "is_secondary_feature": 0,
            "semantic_duplicate_mask": 0,
            "safety_status": "resolved_sample_zero_gt",
            "mask_sha256": sig,
            "support": 2,
            "n_neg_captured": 2,
            "gt_hurt": 0,
            "n_sequences_with_neg": 2,
            "max_neg_sequence_share": 0.5,
            "single_seq_neg_dominance": 0,
            "n_unresolved_selected": 0,
        },
        {
            "atom_id": "S::f::high::u1",
            "feature": "score_m_bridge",
            "direction": "high_tail",
            "thr_index": 1,
            "thr_value": 0.2,
            "productive_safe_point": 1,
            "is_secondary_feature": 0,
            "semantic_duplicate_mask": 0,
            "safety_status": "resolved_sample_zero_gt",
            "mask_sha256": sig,  # same mask
            "support": 2,
            "n_neg_captured": 2,
            "gt_hurt": 0,
            "n_sequences_with_neg": 2,
            "max_neg_sequence_share": 0.5,
            "single_seq_neg_dominance": 0,
            "n_unresolved_selected": 0,
        },
    ]
    stab = classify_region_stability(atoms, [], [])
    # only first unique mask kept; thr_index 0 is boundary (no bilateral)
    assert len(stab) == 1
    assert stab[0]["is_region_candidate"] == 0
    assert stab[0]["stability_class"] in (
        "isolated_safe_point",
        "edge_candidate",
        "thin_safe_edge",
    )


def test_boundary_never_region_candidate() -> None:
    # single cell: boundary
    atoms = [
        {
            "atom_id": "solo",
            "feature": "abs_log_h",
            "direction": "high_tail",
            "thr_index": 5,
            "thr_value": 1.0,
            "productive_safe_point": 1,
            "is_secondary_feature": 0,
            "semantic_duplicate_mask": 0,
            "safety_status": "resolved_sample_zero_gt",
            "mask_sha256": "solo",
            "support": 3,
            "n_neg_captured": 3,
            "gt_hurt": 0,
            "n_sequences_with_neg": 3,
            "max_neg_sequence_share": 0.4,
            "single_seq_neg_dominance": 0,
            "n_unresolved_selected": 0,
        }
    ]
    stab = classify_region_stability(atoms, [], [])
    assert all(int(r["is_region_candidate"]) == 0 for r in stab)


def test_region_metrics_gt_contamination() -> None:
    y = np.array([1, 0, 0])
    seqs = np.array(["A", "A", "B"], dtype=object)
    mask = np.array([True, True, False])
    m = region_metrics(mask, y, seqs, base_neg_rate=1 / 3)
    assert m["gt_hurt"] == 1
    assert m["productive_safe_point"] is False
    assert m["observed_safe_point"] is False


def test_frame_provenance_not_absolute_mot() -> None:
    events = [
        {
            "event_id": "MOT17-02-SDP:f4:c1:l2:i0",
            "join_key": "MOT17-02-SDP|4|10|20|1|2",
            "frame": 4,
            "sequence": "MOT17-02-SDP",
            "cand_track_id": 1,
            "lost_track_id": 2,
        }
    ]
    prov = check_frame_column_provenance(events, stage1_study_dir=None)
    assert prov["semantic"]["is_absolute_mot_frame"] is False
    assert prov["semantic"]["kind"] == "host_audit_propose_invocation_counter"
    assert prov["conclusions"]["affects_q45_threshold_atlas_mainline"] is False
    assert (
        prov["conclusions"][
            "may_claim_temporal_information_unavailable_from_frame_alone"
        ]
        is False
    )
    assert prov["event_id_consistent"] is True
    assert prov["join_key_consistent"] is True


def test_threshold_registry_complete_unique_single() -> None:
    rows = []
    for i, (lab, score) in enumerate(
        [
            ("negative", 0.1),
            ("negative", 0.3),
            ("gt_consistent", 0.2),
            ("gt_consistent", 0.4),
            ("gt_consistent", 0.5),
        ]
    ):
        r = make_q4_primary_row(
            event_id=f"e{i}",
            sequence="MOT17-02-SDP" if i < 3 else "MOT17-05-SDP",
            pair_label=lab,
            score_m_bridge=score,
            abs_log_h=score,
            dist_h=score,
            abs_ratio_m1=score,
            resid_mean=score,
        )
        r["q4_y"] = 1 if lab == "negative" else 0
        rows.append(r)
    # pad to avoid lock? registry doesn't lock n=87
    reg = build_threshold_registry(rows)
    assert reg["single_lattice_kind"] == "primary_unique_boundaries"
    # 5 signals * 2 dirs * 5 unique scores
    assert reg["n_single_atoms"] == 5 * 2 * 5
    atoms = build_single_atom_atlas(reg)
    assert len(atoms) == reg["n_single_atoms"]
    # full family retained — not only best
    supports = {r["support"] for r in atoms if r["feature"] == "score_m_bridge"}
    assert len(supports) >= 2


def test_pareto_keeps_tradeoffs() -> None:
    rows = [
        {
            "atom_id": "a",
            "gt_hurt": 0,
            "n_neg_captured": 1,
            "coverage": 0.1,
            "support": 1,
            "n_sequences_with_neg": 1,
            "max_sequence_share": 1.0,
            "loo_all_gt_hurt_zero": 1,
            "loo_max_gt_hurt": 0,
            "enrichment": 2.0,
            "precision": 1.0,
            "productive_safe_point": 1,
            "feature": "score_m_bridge",
            "direction": "high_tail",
            "semantic_duplicate_mask": 0,
        },
        {
            "atom_id": "b",
            "gt_hurt": 1,
            "n_neg_captured": 5,
            "coverage": 0.3,
            "support": 6,
            "n_sequences_with_neg": 3,
            "max_sequence_share": 0.4,
            "loo_all_gt_hurt_zero": 0,
            "loo_max_gt_hurt": 1,
            "enrichment": 1.5,
            "precision": 0.8,
            "productive_safe_point": 0,
            "feature": "abs_log_h",
            "direction": "high_tail",
            "semantic_duplicate_mask": 0,
        },
        {
            "atom_id": "c",
            "gt_hurt": 0,
            "n_neg_captured": 1,
            "coverage": 0.1,
            "support": 1,
            "n_sequences_with_neg": 1,
            "max_sequence_share": 1.0,
            "loo_all_gt_hurt_zero": 1,
            "loo_max_gt_hurt": 0,
            "enrichment": 2.0,
            "precision": 1.0,
            "productive_safe_point": 1,
            "feature": "dist_h",
            "direction": "high_tail",
            "semantic_duplicate_mask": 0,
        },
    ]
    front = build_pareto_frontier(rows, kind="single_atom")
    ids = {r["region_id"] for r in front}
    # a and b trade off; c same objectives as a → one of a/c kept
    assert "b" in ids
    assert ("a" in ids) or ("c" in ids)


def test_terminal_a_requires_true_holdout_portability() -> None:
    # without true holdout flag → not A
    t0 = classify_q45_terminal(
        stability_rows=[
            {
                "region_id": "R0",
                "stability_class": "loo_stable_region",
                "is_region_candidate": 1,
                "true_holdout_portability_ok": 0,
            }
        ],
        atom_rows=[],
        pairwise_and=[],
        pairwise_or=[],
    )
    assert t0["terminal_letter"] != "A"

    stability = [
        {
            "region_id": "R1",
            "stability_class": "loo_stable_region",
            "is_region_candidate": 1,
            "true_holdout_portability_ok": 1,
        }
    ]
    t = classify_q45_terminal(
        stability_rows=stability,
        atom_rows=[],
        pairwise_and=[],
        pairwise_or=[],
    )
    assert t["stage2_q45_terminal"] == TERMINAL_A
    assert t["terminal_letter"] == "A"


def test_terminal_b_isolated() -> None:
    stability = [
        {
            "region_id": "R1",
            "stability_class": "isolated_safe_point",
            "is_region_candidate": 0,
        }
    ]
    atoms = [
        {
            "productive_safe_point": 1,
            "is_secondary_feature": 0,
            "support": 1,
            "enrichment": 3.0,
            "n_neg_captured": 1,
            "gt_hurt": 0,
        }
    ]
    t = classify_q45_terminal(
        stability_rows=stability,
        atom_rows=atoms,
        pairwise_and=[],
        pairwise_or=[],
    )
    assert t["stage2_q45_terminal"] == TERMINAL_B
    assert t["terminal_letter"] == "B"


def test_terminal_c_enrichment_with_gt() -> None:
    atoms = [
        {
            "productive_safe_point": 0,
            "is_secondary_feature": 0,
            "support": 5,
            "enrichment": 2.0,
            "n_neg_captured": 3,
            "gt_hurt": 2,
        }
    ]
    t = classify_q45_terminal(
        stability_rows=[],
        atom_rows=atoms,
        pairwise_and=[],
        pairwise_or=[],
    )
    assert t["stage2_q45_terminal"] == TERMINAL_C
    assert t["terminal_letter"] == "C"


def test_terminal_d_null() -> None:
    atoms = [
        {
            "productive_safe_point": 0,
            "is_secondary_feature": 0,
            "support": 10,
            "enrichment": 1.0,
            "n_neg_captured": 2,
            "gt_hurt": 5,
        }
    ]
    t = classify_q45_terminal(
        stability_rows=[],
        atom_rows=atoms,
        pairwise_and=[],
        pairwise_or=[],
    )
    assert t["stage2_q45_terminal"] == TERMINAL_D
    assert t["terminal_letter"] == "D"


def test_real_cohort_frame_and_lock() -> None:
    if not Q1Q3.is_dir():
        pytest.skip("q1q3 artifacts missing")
    from saccade.perception.eval.d_online_stage2_q4 import load_d_online_events
    from saccade.perception.eval.d_online_stage2_q45_atlas import lock_q45_cohort

    events = load_d_online_events(Q1Q3)
    prov = check_frame_column_provenance(
        events, stage1_study_dir=STAGE1 if STAGE1.is_dir() else None
    )
    assert prov["all_equal_to_4"] is True
    assert prov["semantic"]["is_absolute_mot_frame"] is False
    locked = lock_q45_cohort(events)
    assert locked["summary"]["n_primary"] == 87
    assert locked["summary"]["n_primary_negative"] == 23
    assert locked["summary"]["n_primary_positive_protect"] == 64
