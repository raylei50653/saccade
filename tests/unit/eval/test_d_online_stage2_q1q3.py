"""Unit tests for M-B1.5 Stage 2 Q1–Q3 D_online audit."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

from pathlib import Path

import pytest

from saccade.perception.eval.d_online_stage2 import (
    Stage2AuditError,
    apply_claim_firewall,
    build_per_sequence_rows,
    build_population_summary,
    build_safe_negative_mass_summary,
    classify_negative_status,
    join_labels_to_d_online,
    make_synthetic_d_online_row,
    reconcile_stage2,
)


def _row(**kw):
    return make_synthetic_d_online_row(**kw)


# ---------------------------------------------------------------------------
# 1. Join key uniqueness fail-closed
# ---------------------------------------------------------------------------


def test_duplicate_event_id_fail_closed(tmp_path: Path) -> None:
    events = [
        {
            "event_id": "dup",
            "join_key": "a",
            "sequence": "MOT17-02-SDP",
            "frame": 1,
            "cand_slot": 0,
            "lost_slot": 1,
            "cand_track_id": 1,
            "lost_track_id": 2,
            "score_m_bridge": 0.1,
            "abs_log_h": 0.0,
            "dist_h": 0.0,
            "abs_ratio_m1": 0.0,
            "resid_mean": 0.0,
            "baseline_rank": 0,
            "baseline_accepted_candidate": 1,
            "baseline_reconnect_decision": 1,
            "rejected_by_hook": 0,
            "runtime_candidate_id": "x",
        },
        {
            "event_id": "dup",
            "join_key": "b",
            "sequence": "MOT17-02-SDP",
            "frame": 2,
            "cand_slot": 0,
            "lost_slot": 1,
            "cand_track_id": 3,
            "lost_track_id": 4,
            "score_m_bridge": 0.1,
            "abs_log_h": 0.0,
            "dist_h": 0.0,
            "abs_ratio_m1": 0.0,
            "resid_mean": 0.0,
            "baseline_rank": 0,
            "baseline_accepted_candidate": 1,
            "baseline_reconnect_decision": 1,
            "rejected_by_hook": 0,
            "runtime_candidate_id": "y",
        },
    ]
    # Need dummy mot/gt for join path — but fail happens before file IO on dups
    with pytest.raises(Stage2AuditError, match="duplicate event_id"):
        join_labels_to_d_online(
            events,
            mot_dir=tmp_path,
            gt_root=tmp_path,
            global_id_map_path=tmp_path / "missing.txt",
            study_id="t",
            source_event_table="t",
        )


def test_duplicate_join_key_fail_closed(tmp_path: Path) -> None:
    events = [
        {
            "event_id": "e1",
            "join_key": "same",
            "sequence": "MOT17-02-SDP",
            "frame": 1,
            "cand_slot": 0,
            "lost_slot": 1,
            "cand_track_id": 1,
            "lost_track_id": 2,
            "score_m_bridge": 0.1,
            "abs_log_h": 0.0,
            "dist_h": 0.0,
            "abs_ratio_m1": 0.0,
            "resid_mean": 0.0,
            "baseline_rank": 0,
            "baseline_accepted_candidate": 1,
            "baseline_reconnect_decision": 1,
            "rejected_by_hook": 0,
            "runtime_candidate_id": "x",
        },
        {
            "event_id": "e2",
            "join_key": "same",
            "sequence": "MOT17-02-SDP",
            "frame": 1,
            "cand_slot": 0,
            "lost_slot": 1,
            "cand_track_id": 1,
            "lost_track_id": 2,
            "score_m_bridge": 0.1,
            "abs_log_h": 0.0,
            "dist_h": 0.0,
            "abs_ratio_m1": 0.0,
            "resid_mean": 0.0,
            "baseline_rank": 0,
            "baseline_accepted_candidate": 1,
            "baseline_reconnect_decision": 1,
            "rejected_by_hook": 0,
            "runtime_candidate_id": "y",
        },
    ]
    with pytest.raises(Stage2AuditError, match="duplicate join_key"):
        join_labels_to_d_online(
            events,
            mot_dir=tmp_path,
            gt_root=tmp_path,
            global_id_map_path=tmp_path / "missing.txt",
            study_id="t",
            source_event_table="t",
        )


# ---------------------------------------------------------------------------
# 2. Missing / ambiguous not defaulted to negative
# ---------------------------------------------------------------------------


def test_unresolved_not_defaulted_to_negative() -> None:
    r = _row(
        event_id="u1",
        pair_label="unknown",
        label_status="unresolved",
        decision_relevance="selected",
    )
    assert r["pair_label"] != "negative"
    assert classify_negative_status(r) == "not_negative"


def test_ambiguous_not_defaulted_to_negative() -> None:
    r = _row(
        event_id="a1",
        pair_label="ambiguous",
        label_status="ambiguous",
        decision_relevance="selected",
    )
    assert classify_negative_status(r) == "not_negative"


# ---------------------------------------------------------------------------
# 3. Taxonomy MECE
# ---------------------------------------------------------------------------


def test_taxonomy_mutually_exclusive_exhaustive() -> None:
    cases = [
        ("gt_consistent", "selected", "not_negative"),
        ("negative", "selected", "negative_safe_removable"),
        ("negative", "active_competitor", "negative_decision_neutral"),
        ("negative", "non_selected", "negative_decision_neutral"),
        ("negative", "unresolved", "negative_unresolved_effect"),
        ("unknown", "selected", "not_negative"),
        ("ambiguous", "selected", "not_negative"),
    ]
    for pl, rel, expected in cases:
        r = _row(
            event_id=f"{pl}_{rel}",
            pair_label=pl,
            label_status=(
                "resolved"
                if pl in ("gt_consistent", "negative")
                else ("ambiguous" if pl == "ambiguous" else "unresolved")
            ),
            decision_relevance=rel,
            baseline_accepted=1 if rel == "selected" else 0,
        )
        assert classify_negative_status(r) == expected


# ---------------------------------------------------------------------------
# 4–5. Aggregate == per-seq; recon group-by
# ---------------------------------------------------------------------------


def test_aggregate_equals_per_sequence_sum() -> None:
    rows = [
        _row(
            event_id="s1a",
            sequence="MOT17-02-SDP",
            pair_label="negative",
            decision_relevance="selected",
            baseline_accepted=1,
        ),
        _row(
            event_id="s1b",
            sequence="MOT17-02-SDP",
            pair_label="gt_consistent",
            decision_relevance="selected",
            baseline_accepted=1,
            cand_slot=1,
            lost_slot=2,
            cand_track_id=11,
            lost_track_id=21,
        ),
        _row(
            event_id="s2a",
            sequence="MOT17-04-SDP",
            pair_label="negative",
            decision_relevance="active_competitor",
            baseline_accepted=0,
            baseline_rank=1,
        ),
        _row(
            event_id="s2b",
            sequence="MOT17-04-SDP",
            pair_label="unknown",
            label_status="unresolved",
            decision_relevance="selected",
            baseline_accepted=1,
            cand_slot=2,
        ),
    ]
    # refresh negative_status after fields set
    for r in rows:
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = int(
            r["negative_status"] == "negative_safe_removable"
        )

    recon = reconcile_stage2(rows)
    assert recon["ok"] is True
    assert recon["acceptance"] == "PASS"
    per = build_per_sequence_rows(rows)
    assert sum(p["n_total"] for p in per) == len(rows)
    assert sum(p["n_negative"] for p in per) == 2
    assert sum(p["n_safe_removable_negative"] for p in per) == 1

    pop = build_population_summary(rows)
    assert pop["funnel"]["D_online_total"] == len(rows)
    assert pop["identity_checks"]["n_label_status_partition"] is True


# ---------------------------------------------------------------------------
# 6–7. Claim firewall
# ---------------------------------------------------------------------------


def test_claim_firewall_zero_decision_relevant_blocks_thr() -> None:
    rows = [
        _row(
            event_id="n1",
            pair_label="negative",
            decision_relevance="active_competitor",
            baseline_accepted=0,
            baseline_rank=1,
        ),
        _row(
            event_id="g1",
            pair_label="gt_consistent",
            decision_relevance="selected",
            baseline_accepted=1,
            cand_slot=1,
            cand_track_id=99,
            lost_track_id=98,
        ),
    ]
    for r in rows:
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = 0
    mass = build_safe_negative_mass_summary(rows)
    recon = reconcile_stage2(rows)
    join = {
        "n_total": 2,
        "n_joined": 2,
        "join_coverage": 1.0,
    }
    fw = apply_claim_firewall(
        join_summary=join, mass=mass, recon=recon, frozen_triggered=0
    )
    assert fw["stage2_q3_safe_negative_mass"] == ("INSUFFICIENT_DECISION_RELEVANT_MASS")
    assert "threshold_or_boolean_claim_not_authorized_in_q1q3" in fw["claims_blocked"]
    assert (
        "reject_policy_study_at_current_placement_unsupported" in fw["claims_blocked"]
    )


def test_claim_firewall_triggered_zero_blocks_effect() -> None:
    rows = [
        _row(
            event_id="s1",
            sequence="MOT17-02-SDP",
            pair_label="negative",
            decision_relevance="selected",
        ),
        _row(
            event_id="s2",
            sequence="MOT17-04-SDP",
            pair_label="negative",
            decision_relevance="selected",
            cand_slot=3,
            cand_track_id=30,
            lost_track_id=40,
        ),
    ]
    for r in rows:
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = 1
    mass = build_safe_negative_mass_summary(rows)
    recon = reconcile_stage2(rows)
    fw = apply_claim_firewall(
        join_summary={"n_total": 2, "n_joined": 2, "join_coverage": 1.0},
        mass=mass,
        recon=recon,
        frozen_triggered=0,
    )
    assert fw["policy_effect_supported"] is False
    assert "frozen_policy_effect_claim_inadmissible" in fw["claims_blocked"]


def test_claim_firewall_insufficient_join_inadmissible() -> None:
    mass = {
        "counts": {
            "N_negative": 0,
            "N_negative_decision_relevant": 0,
            "N_negative_safe_removable": 0,
        },
        "single_sequence_dominance": False,
        "n_sequences_with_safe_removable": 0,
    }
    recon = {"ok": True}
    fw = apply_claim_firewall(
        join_summary={"n_total": 10, "n_joined": 1, "join_coverage": 0.1},
        mass=mass,
        recon=recon,
        frozen_triggered=0,
        min_join_coverage=0.5,
    )
    assert fw["stage2_q1_label_join"] == "FAILED"
    assert fw["stage2_q3_safe_negative_mass"] == "INADMISSIBLE"


# ---------------------------------------------------------------------------
# 8. Fixture A / B / C terminals
# ---------------------------------------------------------------------------


def test_fixture_terminal_a_sufficient() -> None:
    rows = []
    for i, seq in enumerate(["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP"]):
        rows.append(
            _row(
                event_id=f"a{i}",
                sequence=seq,
                pair_label="negative",
                decision_relevance="selected",
                cand_slot=i,
                cand_track_id=100 + i,
                lost_track_id=200 + i,
            )
        )
        rows.append(
            _row(
                event_id=f"g{i}",
                sequence=seq,
                pair_label="gt_consistent",
                decision_relevance="selected",
                cand_slot=i + 10,
                cand_track_id=300 + i,
                lost_track_id=400 + i,
            )
        )
    for r in rows:
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = int(
            r["negative_status"] == "negative_safe_removable"
        )
    mass = build_safe_negative_mass_summary(rows)
    recon = reconcile_stage2(rows)
    fw = apply_claim_firewall(
        join_summary={
            "n_total": len(rows),
            "n_joined": len(rows),
            "join_coverage": 1.0,
        },
        mass=mass,
        recon=recon,
        frozen_triggered=0,
    )
    assert fw["stage2_q3_safe_negative_mass"] == "SUFFICIENT"
    assert mass["counts"]["N_negative_safe_removable"] == 3
    assert mass["single_sequence_dominance"] is False


def test_fixture_terminal_b_decision_neutral() -> None:
    rows = [
        _row(
            event_id="bn1",
            pair_label="negative",
            decision_relevance="active_competitor",
            baseline_accepted=0,
            baseline_rank=1,
        ),
        _row(
            event_id="bn2",
            sequence="MOT17-04-SDP",
            pair_label="negative",
            decision_relevance="non_selected",
            baseline_accepted=0,
            baseline_rank=2,
            cand_slot=5,
        ),
    ]
    for r in rows:
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = 0
    mass = build_safe_negative_mass_summary(rows)
    fw = apply_claim_firewall(
        join_summary={"n_total": 2, "n_joined": 2, "join_coverage": 1.0},
        mass=mass,
        recon=reconcile_stage2(rows),
        frozen_triggered=0,
    )
    assert fw["stage2_q3_safe_negative_mass"] == ("INSUFFICIENT_DECISION_RELEVANT_MASS")


def test_fixture_terminal_c_no_negatives() -> None:
    rows = [
        _row(
            event_id="c1",
            pair_label="gt_consistent",
            decision_relevance="selected",
        ),
        _row(
            event_id="c2",
            sequence="MOT17-04-SDP",
            pair_label="gt_consistent",
            decision_relevance="selected",
            cand_slot=2,
            cand_track_id=50,
            lost_track_id=51,
        ),
    ]
    for r in rows:
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = 0
    mass = build_safe_negative_mass_summary(rows)
    fw = apply_claim_firewall(
        join_summary={"n_total": 2, "n_joined": 2, "join_coverage": 1.0},
        mass=mass,
        recon=reconcile_stage2(rows),
        frozen_triggered=0,
    )
    assert fw["stage2_q3_safe_negative_mass"] == (
        "CURRENT_PLACEMENT_TOO_LATE_CANDIDATE"
    )
