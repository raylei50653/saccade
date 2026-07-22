"""Tests for the compare-handover-summaries diagnostics CLI (scripts/eval/diagnostics)."""

# scope: eval
# function: behavior
# lifecycle: active

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


def _summary(*, pred_dir: str, correct: int, wrong: int, polluted: int) -> dict:
    total = correct + wrong
    return {
        "schema": "cheb_gr_offline_handover_summary/v1",
        "provenance": {"pred_dir": pred_dir},
        "counts": {
            "rows": total,
            "accepted": 4,
            "known": total,
            "unknown": 0,
            "correct": correct,
            "wrong": wrong,
            "same_rate": correct / total,
            "accepted_known": 4,
            "accepted_correct": correct,
            "accepted_wrong": wrong,
            "accepted_precision": correct / total,
        },
        "candidate_rules": [
            {
                "name": "best_cost_accept_strict",
                "decision": "accept-candidate",
                "expression": "best_cost <= 0.25",
                "selected": total,
                "correct": correct,
                "wrong": wrong,
                "same_rate": correct / total,
                "loo_same_rate_min": 0.5,
                "loo_same_rate_max": 0.9,
            }
        ],
        "policy_simulation": [
            {
                "name": "best_cost_accept_strict",
                "action": "keep",
                "kept_correct": correct,
                "accepted_correct": correct,
                "kept_wrong": wrong,
                "accepted_wrong": wrong,
                "wrong_cut": 0,
                "correct_cut": 0,
            }
        ],
        "discovered_gates": {
            "all_known_single_feature": [
                {
                    "expression": "best_cost <= 0.25",
                    "selected": total,
                    "correct": correct,
                    "wrong": wrong,
                    "precision": correct / total,
                    "correct_recall": 1.0,
                    "wrong_keep": 1.0,
                }
            ],
            "accepted_known_single_feature": [],
            "accepted_known_two_feature": [
                {
                    "expression": "best_cost <= 0.25 && margin >= 0.12",
                    "selected": total,
                    "correct": correct,
                    "wrong": wrong,
                    "precision": correct / total,
                    "correct_recall": 1.0,
                    "wrong_keep": 1.0,
                }
            ],
        },
        "features": {
            "best_cost": {
                "buckets": [
                    {
                        "bucket": "[-inf,0.25)",
                        "correct": correct,
                        "wrong": wrong,
                        "total": total,
                        "same_rate": correct / total,
                        "zone": "accept-candidate",
                    }
                ]
            }
        },
        "pollution": {
            "eligible": total,
            "endpoint_polluted": polluted,
            "pollution_rate": polluted / total,
            "feature_buckets": {
                "neighbor_iou": [
                    {
                        "bucket": "[0.5,0.7)",
                        "total": total,
                        "endpoint_polluted": polluted,
                        "pollution_rate": polluted / total,
                    }
                ]
            },
            "matrices": [],
        },
    }


def test_compare_handover_summaries_reports_rule_and_pollution_deltas(
    tmp_path: Path,
):
    run_a = _summary(pred_dir="results/run_a", correct=3, wrong=1, polluted=1)
    run_b = deepcopy(run_a)
    run_b.update(_summary(pred_dir="results/run_b", correct=2, wrong=2, polluted=3))

    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(run_a))
    path_b.write_text(json.dumps(run_b))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/diagnostics/compare_handover_summaries.py",
            str(path_a),
            str(path_b),
            "--feature",
            "best_cost",
            "--min-selected",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "# Cheb-GR Offline Handover Summary Comparison" in out
    assert "| run_a | 4 | 4 | 4 | 3 | 1 | 0.750 | 0.750 |" in out
    assert "| run_b | 4 | 4 | 4 | 2 | 2 | 0.500 | 0.500 |" in out
    assert (
        "| `best_cost_accept_strict` | accept-candidate | `best_cost <= 0.25` | "
        "3/4 (0.75); LOO 0.50-0.90 | 2/4 (0.50); LOO 0.50-0.90 |" in out
    )
    assert "## Discovered Gates" in out
    assert (
        "| `best_cost <= 0.25 && margin >= 0.12` | "
        "3/4 (0.75); p=0.75 r=1.00 bad_keep=1.00 | "
        "2/4 (0.50); p=0.50 r=1.00 bad_keep=1.00 |" in out
    )
    assert "| run_a | 4 | 1 | 0.250 |" in out
    assert "| run_b | 4 | 3 | 0.750 |" in out
    assert "| `[0.5,0.7)` | 1/4 (0.25) | 3/4 (0.75) |" in out
    assert "| `[-inf,0.25)` | 3/4 (0.75); thin | 2/4 (0.50); thin |" in out
