import json
import subprocess
import sys
from pathlib import Path


def _summary(pred_dir: str, *, accept_rate: tuple[int, int]) -> dict:
    accept_ok, accept_total = accept_rate
    accept_bad = accept_total - accept_ok
    return {
        "schema": "cheb_gr_offline_handover_summary/v1",
        "provenance": {"pred_dir": pred_dir},
        "candidate_rules": [
            {
                "name": "accept_rule",
                "decision": "accept-candidate",
                "expression": "best_cost <= 0.25",
                "selected": accept_total,
                "correct": accept_ok,
                "wrong": accept_bad,
                "same_rate": accept_ok / accept_total,
                "use": "accept probe",
            },
            {
                "name": "veto_rule",
                "decision": "reject/veto",
                "expression": "center_dist_norm >= 2",
                "selected": 100,
                "correct": 3,
                "wrong": 97,
                "same_rate": 0.03,
                "use": "veto probe",
            },
        ],
        "features": {
            "best_cost": {
                "buckets": [
                    {
                        "bucket": "[-inf,0.25)",
                        "correct": accept_ok,
                        "wrong": accept_bad,
                        "total": accept_total,
                        "same_rate": accept_ok / accept_total,
                        "zone": "accept-candidate",
                    },
                    {
                        "bucket": "[0.5,inf)",
                        "correct": 2,
                        "wrong": 98,
                        "total": 100,
                        "same_rate": 0.02,
                        "zone": "danger",
                    },
                ]
            }
        },
        "pollution": {
            "feature_buckets": {
                "neighbor_iou": [
                    {
                        "bucket": "[0.5,0.7)",
                        "endpoint_polluted": 15,
                        "total": 40,
                        "pollution_rate": 0.375,
                    }
                ],
                "head_tail_neighbor_iou": [],
                "match_iou": [],
            }
        },
    }


def test_synthesize_handover_applicability_marks_stable_veto_and_drifting_accept(
    tmp_path: Path,
):
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    out_json = tmp_path / "applicability.json"
    out_md = tmp_path / "applicability.md"
    path_a.write_text(json.dumps(_summary("results/a", accept_rate=(9, 10))))
    path_b.write_text(json.dumps(_summary("results/b", accept_rate=(7, 10))))

    subprocess.run(
        [
            sys.executable,
            "scripts/eval/diagnostics/synthesize_handover_applicability.py",
            str(path_a),
            str(path_b),
            "--feature",
            "best_cost",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads(out_json.read_text())
    assert data["schema"] == "cheb_gr_offline_handover_applicability/v1"
    rules = {rule["name"]: rule for rule in data["rule_applicability"]}
    assert rules["veto_rule"]["classification"] == "stable-veto"
    assert rules["accept_rule"]["classification"] == "condition-sensitive-accept"

    ranges = {
        (item["feature"], item["bucket"]): item
        for item in data["feature_range_applicability"]
    }
    assert ranges[("best_cost", "[0.5,inf)")]["classification"] == "stable-danger"
    assert ranges[("best_cost", "[-inf,0.25)")]["classification"] == (
        "condition-sensitive-support-range"
    )

    pollution = {
        (item["feature"], item["bucket"]): item
        for item in data["pollution_applicability"]
    }
    assert pollution[("neighbor_iou", "[0.5,0.7)")]["classification"] == (
        "stable-high-pollution"
    )

    md = out_md.read_text()
    assert "## Rule Applicability" in md
    assert "`veto_rule` | stable-veto" in md
    assert "`accept_rule` | condition-sensitive-accept" in md
