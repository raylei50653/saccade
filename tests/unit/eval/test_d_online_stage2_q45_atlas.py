"""Unit tests for M-B1.5 Stage 2 Q4.5 threshold-combination atlas."""

# scope: eval
# function: behavior
# lifecycle: active

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
    fixed_full_sample_region_partition_check,
    nested_loso_portability_audit,
    region_metrics,
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


def test_fixed_full_sample_partition_is_not_portability() -> None:
    """Fixed thr on subsets of full sample is diagnostic only, not nested LOSO."""
    x = np.array([0.8, 0.1, 0.9, 0.2])
    y = np.array([0, 1, 1, 0])
    s = np.array(["A", "B", "B", "C"], dtype=object)
    ho = fixed_full_sample_region_partition_check(
        x_parts={"score_m_bridge": x},
        y=y,
        sequences=s,
        feature="score_m_bridge",
        direction="high_tail",
        thr_value=0.5,
    )
    assert ho["not_portability_evidence"] is True
    assert ho["kind"] == "fixed_full_sample_region_partition_check"
    a_row = next(r for r in ho["rows"] if r["hold_out_sequence"] == "A")
    assert a_row["holdout_gt_hurt"] == 1


def test_nested_loso_can_fail_when_full_sample_safe() -> None:
    """Nested train-select → holdout-eval is non-tautological.

    Sequence A has a high GT; sequences B/C have high negatives. Full-sample
    high_tail thr that only fires on B/C is productive-safe on the full pool,
    but when B or C is held out the train lattice may select thr that hurt A,
    or a thr selected without A may still hurt A on holdout.
    """
    # Build primary-like rows with make_q4 helper fields + q4_y
    primary = []
    # A: GT high scores (would hurt high_tail if selected)
    for i, score in enumerate([0.95, 0.92]):
        r = make_q4_primary_row(
            event_id=f"A{i}",
            sequence="A",
            pair_label="gt_consistent",
            score_m_bridge=score,
            abs_log_h=0.1,
            dist_h=0.1,
            abs_ratio_m1=0.1,
            resid_mean=0.1,
        )
        r["q4_y"] = 0
        primary.append(r)
    # B, C: negatives at high scores; GT at low
    for seq in ("B", "C", "D"):
        for i, (lab, score, y) in enumerate(
            [
                ("negative", 0.91, 1),
                ("negative", 0.88, 1),
                ("gt_consistent", 0.05, 0),
                ("gt_consistent", 0.08, 0),
            ]
        ):
            r = make_q4_primary_row(
                event_id=f"{seq}{i}",
                sequence=seq,
                pair_label=lab,
                score_m_bridge=score,
                abs_log_h=0.1,
                dist_h=0.1,
                abs_ratio_m1=0.1,
                resid_mean=0.1,
            )
            r["q4_y"] = y
            primary.append(r)

    audit = nested_loso_portability_audit(
        primary,
        selected_unresolved=[],
        selected_ambiguous=[],
        signals=("score_m_bridge",),
    )
    assert audit["kind"] == "nested_loso_train_select_holdout_eval"
    assert audit["not_fixed_full_sample_partition"] is True
    assert audit["n_folds"] == 4
    # Holding out A: train may select high_tail thr on B/C/D negs; A holdout GT hurt
    a_folds = [
        r
        for r in audit["fold_detail_rows"]
        if r["hold_out_sequence"] == "A" and r["kind"] == "single"
    ]
    assert a_folds, "expected some clauses selected when A held out"
    assert any(int(r["holdout_gt_hurt"]) > 0 for r in a_folds)
    # Not every clause is automatically portable
    assert any(
        int(r["nested_loso_portability_ok"]) == 0 for r in audit["clause_summary_rows"]
    )


def _safe_atom(
    *,
    thr_index: int,
    thr_value: float,
    mask: str,
    feature: str = "score_m_bridge",
    direction: str = "high_tail",
    is_dup: int = 0,
    n_seq: int = 3,
) -> dict:
    return {
        "atom_id": f"S::{feature}::{direction}::u{thr_index}::{mask[:8]}",
        "feature": feature,
        "direction": direction,
        "thr_index": thr_index,
        "thr_value": thr_value,
        "productive_safe_point": 1,
        "is_secondary_feature": 0,
        "semantic_duplicate_mask": is_dup,
        "safety_status": "resolved_sample_zero_gt",
        "mask_sha256": mask,
        "support": 4,
        "n_neg_captured": 4,
        "gt_hurt": 0,
        "n_sequences_with_neg": n_seq,
        "max_neg_sequence_share": 0.4,
        "single_seq_neg_dominance": 0,
        "n_unresolved_selected": 0,
    }


def _safe_pair(
    *,
    thr_a: int,
    thr_b: int,
    mask: str,
    comb: str = "AND",
    n_seq: int = 3,
) -> dict:
    return {
        "region_id": f"P::{comb}::{thr_a}::{thr_b}::{mask[:8]}",
        "feature_a": "score_m_bridge",
        "direction_a": "high_tail",
        "feature_b": "abs_log_h",
        "direction_b": "high_tail",
        "combinator": comb,
        "thr_index_a": thr_a,
        "thr_index_b": thr_b,
        "thr_value_a": float(thr_a) * 0.1,
        "thr_value_b": float(thr_b) * 0.1,
        "productive_safe_point": 1,
        "semantic_duplicate_mask": 0,
        "safety_status": "resolved_sample_zero_gt",
        "mask_sha256": mask,
        "support": 4,
        "n_neg_captured": 4,
        "gt_hurt": 0,
        "n_sequences_with_neg": n_seq,
        "max_neg_sequence_share": 0.4,
        "single_seq_neg_dominance": 0,
        "n_unresolved_selected": 0,
    }


def test_quotient_keeps_plateau_coordinates_including_dups() -> None:
    """Same mask at thr 0,1,2 is one quotient node with interior at thr 1.

    semantic_duplicate_mask on later cells must not discard their coordinates.
    """
    sig = "plateau_abc"
    atoms = [
        _safe_atom(thr_index=ti, thr_value=tv, mask=sig, is_dup=is_dup)
        for ti, tv, is_dup in ((0, 0.1, 0), (1, 0.2, 1), (2, 0.3, 1))
    ]
    stab = classify_region_stability(atoms, [], [])
    assert len(stab) == 1
    assert int(stab[0]["n_coordinates"]) == 3
    assert int(stab[0]["has_interior_coordinate"]) == 1
    assert int(stab[0]["same_mask_plateau_has_interior"]) == 1
    assert int(stab[0]["n_interior_coordinates"]) == 1
    assert int(stab[0]["component_size_coordinates"]) == 3
    assert int(stab[0]["component_size_unique_masks"]) == 1
    assert stab[0]["stability_class"] in (
        "locally_stable_region",
        "loo_stable_region",
    )
    assert int(stab[0]["is_region_candidate"]) == 1


def test_multi_mask_1d_safe_interval_has_interior() -> None:
    """Three consecutive productive-safe cells with *different* masks.

    Safe-region thickness is on the coordinate union: center thr must be
    interior even though each mask occupies only one coordinate.
    """
    atoms = [
        _safe_atom(thr_index=5, thr_value=0.5, mask="mask_A"),
        _safe_atom(thr_index=6, thr_value=0.6, mask="mask_B"),
        _safe_atom(thr_index=7, thr_value=0.7, mask="mask_C"),
    ]
    stab = classify_region_stability(atoms, [], [])
    assert len(stab) == 3
    by_mask = {r["mask_sha256"]: r for r in stab}
    # Center is interior under coordinate-union topology.
    assert int(by_mask["mask_B"]["has_interior_coordinate"]) == 1
    assert int(by_mask["mask_B"]["n_interior_coordinates"]) == 1
    assert int(by_mask["mask_B"]["is_region_candidate"]) == 1
    assert by_mask["mask_B"]["stability_class"] in (
        "locally_stable_region",
        "loo_stable_region",
    )
    # Edges of the interval are not interior, but belong to same component.
    assert int(by_mask["mask_A"]["has_interior_coordinate"]) == 0
    assert int(by_mask["mask_C"]["has_interior_coordinate"]) == 0
    assert by_mask["mask_A"]["stability_class"] == "edge_candidate"
    assert int(by_mask["mask_A"]["component_size_coordinates"]) == 3
    assert int(by_mask["mask_A"]["component_size_unique_masks"]) == 3
    # Same-mask plateau metric stays false (single coord per mask).
    assert int(by_mask["mask_B"]["same_mask_plateau_has_interior"]) == 0


def test_multi_mask_2d_block_center_has_interior() -> None:
    """3×3 productive-safe block with nine distinct masks → center is interior."""
    pairs = []
    mid = None
    for ia in range(3):
        for ib in range(3):
            m = f"m{ia}{ib}"
            row = _safe_pair(thr_a=ia, thr_b=ib, mask=m)
            pairs.append(row)
            if ia == 1 and ib == 1:
                mid = m
    assert mid is not None
    stab = classify_region_stability([], pairs, [])
    assert len(stab) == 9
    by_mask = {r["mask_sha256"]: r for r in stab}
    center = by_mask[mid]
    assert int(center["has_interior_coordinate"]) == 1
    assert int(center["n_interior_coordinates"]) == 1
    assert int(center["is_region_candidate"]) == 1
    assert int(center["same_mask_plateau_has_interior"]) == 0
    assert int(center["component_size_coordinates"]) == 9
    assert int(center["component_size_unique_masks"]) == 9
    # Corners: no full 4-neighborhood → not interior.
    corner = by_mask["m00"]
    assert int(corner["has_interior_coordinate"]) == 0
    assert corner["stability_class"] == "edge_candidate"


def test_thin_two_point_plateau_not_interior() -> None:
    # two thr indices, same mask → width 2 but no thr with both neighbors
    sig = "abc123"
    atoms = [
        _safe_atom(thr_index=0, thr_value=0.1, mask=sig, is_dup=0, n_seq=2),
        _safe_atom(thr_index=1, thr_value=0.2, mask=sig, is_dup=1, n_seq=2),
    ]
    stab = classify_region_stability(atoms, [], [])
    assert len(stab) == 1
    assert int(stab[0]["n_coordinates"]) == 2
    assert int(stab[0]["has_interior_coordinate"]) == 0
    assert int(stab[0]["is_region_candidate"]) == 0
    assert stab[0]["stability_class"] in (
        "isolated_safe_point",
        "edge_candidate",
        "thin_safe_edge",
    )


def test_nested_loso_reports_exact_absolute_portability_name() -> None:
    """Portability counter is exact-absolute clause repeatability, not region."""
    primary = []
    for seq, scores, labels in (
        ("A", [0.1, 0.2], ["negative", "gt_consistent"]),
        ("B", [0.15, 0.25], ["negative", "gt_consistent"]),
        ("C", [0.12, 0.22], ["negative", "gt_consistent"]),
    ):
        for i, (sc, lab) in enumerate(zip(scores, labels)):
            r = make_q4_primary_row(
                event_id=f"{seq}{i}",
                sequence=seq,
                pair_label=lab,
                score_m_bridge=sc,
                abs_log_h=sc,
                dist_h=sc,
                abs_ratio_m1=sc,
                resid_mean=sc,
            )
            r["q4_y"] = 1 if lab == "negative" else 0
            primary.append(r)
    audit = nested_loso_portability_audit(
        primary, selected_unresolved=[], selected_ambiguous=[]
    )
    assert "n_exact_absolute_clauses_nested_loso_portable" in audit
    assert audit["clause_identity"] == "exact_absolute_threshold_float_round12"
    # Alias retained for wiring.
    assert (
        audit["n_clauses_nested_loso_portable"]
        == audit["n_exact_absolute_clauses_nested_loso_portable"]
    )


def test_boundary_never_region_candidate() -> None:
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
    reg = build_threshold_registry(rows)
    assert reg["single_lattice_kind"] == "primary_unique_boundaries"
    assert reg["n_single_atoms"] == 5 * 2 * 5
    atoms = build_single_atom_atlas(reg)
    assert len(atoms) == reg["n_single_atoms"]
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
    assert "b" in ids
    assert ("a" in ids) or ("c" in ids)


def test_terminal_a_requires_nested_loso_portability() -> None:
    t0 = classify_q45_terminal(
        stability_rows=[
            {
                "region_id": "R0",
                "stability_class": "loo_stable_region",
                "is_region_candidate": 1,
                "nested_loso_portability_ok": 0,
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
            "nested_loso_portability_ok": 1,
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
