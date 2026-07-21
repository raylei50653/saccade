"""WP2: occ-exit sequence conditioning classification (analysis only)."""

# scope: eval
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import json
from pathlib import Path

from saccade.perception.eval.occ_audit_seq_conditioning import (
    SeqEvidence,
    Thresholds,
    aggregate_occ_audit_rows,
    attach_metrics,
    build_applicability_table,
    classify_seq,
    decide_promotion,
    load_metrics_json,
    load_occ_audit_csv,
    render_applicability_md,
    seq_type,
)


def test_empty_input_insufficient_evidence():
    by_seq = aggregate_occ_audit_rows([])
    assert by_seq == {}
    # synthetic empty-ish seq
    ev = SeqEvidence(seq="MOT17-04-SDP", audited=0)
    assert classify_seq(ev) == "insufficient_evidence"
    # missing csv path
    assert load_occ_audit_csv(Path("/no/such/_occ_audit.csv")) == []


def test_low_audited_count_insufficient():
    ev = SeqEvidence(
        seq="MOT17-04-SDP",
        audited=3,
        cosine_flags=2,
        has_metrics=True,
        idf1_delta=1.0,
        ids_delta=-2,
    )
    assert classify_seq(ev) == "insufficient_evidence"


def test_positive_metric_and_useful_flags_enable_candidate():
    ev = SeqEvidence(
        seq="MOT17-05-SDP",
        seq_type=seq_type("MOT17-05-SDP"),
        audited=20,
        cosine_flags=6,
        chebgr_flags=5,
        flag_delta_same=15,
        flag_delta_same_flagged=4,
        flag_delta_cosine_only=2,
        flag_delta_chebgr_only=1,
        has_chebgr_columns=True,
        has_metrics=True,
        idf1_delta=0.5,
        ids_delta=-3,
    )
    assert classify_seq(ev) == "enable_candidate"
    assert ev.useful_flags >= 2


def test_mixed_or_low_flags_abstain():
    # metrics near zero, few useful flags
    ev = SeqEvidence(
        seq="MOT17-09-SDP",
        audited=12,
        cosine_flags=1,
        chebgr_flags=1,
        flag_delta_same=11,
        flag_delta_same_flagged=1,
        has_chebgr_columns=True,
        has_metrics=True,
        idf1_delta=0.05,
        ids_delta=1,
    )
    assert classify_seq(ev) == "abstain"


def test_negative_metric_harmful():
    ev = SeqEvidence(
        seq="MOT17-02-SDP",
        audited=30,
        cosine_flags=10,
        has_metrics=True,
        idf1_delta=-0.8,
        ids_delta=12,
    )
    assert classify_seq(ev) == "harmful"


def test_ids_material_regress_with_flat_idf1_harmful():
    ev = SeqEvidence(
        seq="MOT17-04-SDP",
        audited=25,
        cosine_flags=8,
        has_metrics=True,
        idf1_delta=0.0,
        ids_delta=8,
    )
    assert classify_seq(ev) == "harmful"


def test_parser_tolerates_missing_chebgr_columns():
    rows = [
        {
            "seq": "MOT17-04-SDP",
            "flagged": "True",
            "min_cos": "0.3",
            "tau": "0.45",
        },
        {
            "seq": "MOT17-04-SDP",
            "flagged": "False",
            "min_cos": "0.7",
            "tau": "0.45",
        },
        {
            "seq": "MOT17-05-SDP",
            "flagged": "False",
            "min_cos": "0.8",
        },
    ]
    by_seq = aggregate_occ_audit_rows(rows)
    assert by_seq["MOT17-04-SDP"].audited == 2
    assert by_seq["MOT17-04-SDP"].cosine_flags == 1
    assert by_seq["MOT17-04-SDP"].has_chebgr_columns is False
    assert by_seq["MOT17-04-SDP"].flag_delta_chebgr_only == 0
    assert by_seq["MOT17-05-SDP"].audited == 1
    # no metrics → insufficient
    assert classify_seq(by_seq["MOT17-04-SDP"]) == "insufficient_evidence"


def test_flag_delta_aggregation_probe_on():
    rows = [
        {
            "seq": "MOT17-10-SDP",
            "flagged": "True",
            "cosine_flag": "True",
            "chebgr_flag": "True",
            "flag_delta": "same",
        },
        {
            "seq": "MOT17-10-SDP",
            "flagged": "True",
            "cosine_flag": "True",
            "chebgr_flag": "False",
            "flag_delta": "cosine_only",
        },
        {
            "seq": "MOT17-10-SDP",
            "flagged": "False",
            "cosine_flag": "False",
            "chebgr_flag": "True",
            "flag_delta": "chebgr_only",
        },
        {
            "seq": "MOT17-10-SDP",
            "flagged": "False",
            "cosine_flag": "False",
            "chebgr_flag": "False",
            "flag_delta": "same",
        },
    ]
    ev = aggregate_occ_audit_rows(rows)["MOT17-10-SDP"]
    assert ev.has_chebgr_columns is True
    assert ev.flag_delta_same == 2
    assert ev.flag_delta_same_flagged == 1
    assert ev.flag_delta_cosine_only == 1
    assert ev.flag_delta_chebgr_only == 1
    assert ev.cosine_flags == 2
    assert ev.chebgr_flags == 2
    assert ev.useful_flags == 3  # 1 same_flagged + 1 cos_only + 1 chebgr_only


def test_chebgr_only_domination_without_metrics_harmful():
    ev = SeqEvidence(
        seq="MOT17-13-SDP",
        audited=20,
        cosine_flags=1,
        chebgr_flags=10,
        flag_delta_same=10,
        flag_delta_same_flagged=1,
        flag_delta_cosine_only=0,
        flag_delta_chebgr_only=9,
        has_chebgr_columns=True,
        has_metrics=False,
    )
    assert classify_seq(ev) == "harmful"


def test_metrics_json_control_treatment_pairs(tmp_path: Path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "per_sequence": {
                    "MOT17-05-SDP": {
                        "idf1_control": 78.0,
                        "idf1_treatment": 78.6,
                        "ids_control": 40,
                        "ids_treatment": 35,
                    }
                }
            }
        )
    )
    m = load_metrics_json(path)
    assert abs(float(m["MOT17-05-SDP"]["idf1_delta"]) - 0.6) < 1e-9
    assert m["MOT17-05-SDP"]["ids_delta"] == -5


def test_build_table_and_md_roundtrip(tmp_path: Path):
    rows = [
        {
            "seq": "MOT17-05-SDP",
            "cosine_flag": "True",
            "chebgr_flag": "True",
            "flag_delta": "same",
            "flagged": "True",
        }
    ] * 8
    by_seq = aggregate_occ_audit_rows(rows)
    attach_metrics(
        by_seq,
        {"MOT17-05-SDP": {"idf1_delta": 0.4, "ids_delta": -2}},
    )
    table = build_applicability_table(by_seq, Thresholds(min_audited=5))
    assert len(table) == 1
    assert table[0]["recommendation"] == "enable_candidate"
    md = render_applicability_md(table, provenance={"occ_audit_csv": "x.csv"})
    assert "enable_candidate" in md
    assert "No default-on sequence gate" in md
    assert "MOT17-05-SDP" in md

    csv_path = tmp_path / "_occ_audit.csv"
    csv_path.write_text(
        "seq,flagged,cosine_flag,chebgr_flag,flag_delta\n"
        + "\n".join("MOT17-11-SDP,False,False,False,same" for _ in range(6))
        + "\n"
    )
    loaded = load_occ_audit_csv(csv_path)
    assert len(loaded) == 6
    assert classify_seq(aggregate_occ_audit_rows(loaded)["MOT17-11-SDP"]) == (
        "insufficient_evidence"
    )


def test_seq_type_mapping():
    assert seq_type("MOT17-02-SDP") == "crowded_static"
    assert seq_type("MOT17-05") == "moving_low"
    assert seq_type("custom-seq") == "unknown"


def test_decide_promotion_promote_default_off_gate():
    table = [
        {"seq": "A", "recommendation": "enable_candidate"},
        {"seq": "B", "recommendation": "enable_candidate"},
        {"seq": "C", "recommendation": "abstain"},
        {"seq": "D", "recommendation": "harmful"},
    ]
    out = decide_promotion(table, overall_idf1_delta_pp=0.1, overall_ids_delta=-2)
    assert out["decision"] == "promote_default_off_gate"
    assert out["gate_implemented"] is False
    assert "A" in out["enable_seqs"]


def test_decide_promotion_no_go_multi_harm():
    table = [{"seq": f"s{i}", "recommendation": "harmful"} for i in range(3)] + [
        {"seq": "z", "recommendation": "abstain"}
    ]
    out = decide_promotion(table, overall_idf1_delta_pp=-0.5, overall_ids_delta=12)
    assert out["decision"] == "no_go"


def test_decide_promotion_split_feat_local_enable():
    table = [
        {"seq": "A", "recommendation": "enable_candidate"},
        {"seq": "B", "recommendation": "harmful"},
        {"seq": "C", "recommendation": "abstain"},
    ]
    out = decide_promotion(table, overall_idf1_delta_pp=-0.05, overall_ids_delta=2)
    assert out["decision"] == "split_feat_pr"


def test_decide_promotion_research_only_insufficient():
    table = [
        {"seq": "A", "recommendation": "insufficient_evidence"},
        {"seq": "B", "recommendation": "insufficient_evidence"},
    ]
    out = decide_promotion(table, overall_idf1_delta_pp=0.0, overall_ids_delta=0)
    assert out["decision"] == "research_only"
    assert out["gate_implemented"] is False
