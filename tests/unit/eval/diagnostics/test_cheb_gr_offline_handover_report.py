"""Tests for the offline Cheb-GR handover report CLI (scripts.eval.diagnostics.cheb_gr_offline_handover_report)."""

# scope: eval
# function: behavior
# lifecycle: active

import csv
import json
import subprocess
import sys
from pathlib import Path


def _mot_line(frame: int, tid: int, x: float, y: float) -> str:
    return f"{frame},{tid},{x},{y},10,20,1,1,1\n"


def _write_track(
    gt_lines: list[str],
    pred_lines: list[str],
    *,
    gt_id: int,
    pred_id: int,
    frames: range,
    x: float,
    y: float = 0.0,
) -> None:
    for frame in frames:
        line = _mot_line(frame, gt_id, x, y)
        gt_lines.append(line)
        pred_lines.append(_mot_line(frame, pred_id, x, y))


def _handover_row(
    *,
    seq: str,
    candidate_id: int,
    candidate_start: int,
    candidate_end: int,
    newborn_id: int,
    newborn_start: int,
    newborn_end: int,
    accepted: bool,
    best_cost: float,
    margin: float,
    match_iou: float,
    neighbor_iou: float,
    head_tail_neighbor_iou: float,
    center_dist_norm: float,
    candidate_count: int,
    key_best_sim: float = 0.8,
    key_margin: float = 0.1,
) -> dict[str, object]:
    return {
        "seq": seq,
        "newborn_id": newborn_id,
        "newborn_start": newborn_start,
        "newborn_end": newborn_end,
        "candidate_id": candidate_id,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "accepted": str(accepted).lower(),
        "best_cost": best_cost,
        "second_cost": best_cost + margin,
        "margin": margin,
        "required_margin": 0.05,
        "max_cost": 0.5,
        "key_best_sim": key_best_sim,
        "key_mean_topk_sim": key_best_sim - 0.02,
        "key_best_other_id": -1 if key_margin >= 999.0 else candidate_id + 1,
        "key_best_other_sim": -1.0
        if key_margin >= 999.0
        else key_best_sim - key_margin,
        "key_margin": key_margin,
        "key_support": 4,
        "key_other_support": 0 if key_margin >= 999.0 else 4,
        "match_iou": match_iou,
        "direct_iou": match_iou,
        "candidate_forward_iou": match_iou,
        "newborn_backward_iou": match_iou,
        "neighbor_iou": neighbor_iou,
        "head_tail_neighbor_iou": head_tail_neighbor_iou,
        "newborn_neighbor_iou": neighbor_iou,
        "candidate_neighbor_iou": neighbor_iou,
        "newborn_head_neighbor_iou": head_tail_neighbor_iou,
        "candidate_tail_neighbor_iou": head_tail_neighbor_iou,
        "center_dist_norm": center_dist_norm,
        "gap": newborn_start - candidate_end,
        "candidate_count": candidate_count,
        "head_n": 2,
        "bank_n": 2,
        "newborn_mean_score": 0.9,
        "newborn_start_score": 0.9,
        "candidate_mean_score": 0.9,
        "candidate_end_score": 0.9,
    }


def test_offline_handover_report_labels_edges_and_writes_registry(tmp_path: Path):
    seq = "MOT17-UNIT-FRCNN"
    gt_dir = tmp_path / "gt_root" / seq / "gt"
    baseline_dir = tmp_path / "baseline"
    gt_dir.mkdir(parents=True)
    baseline_dir.mkdir()

    gt_lines: list[str] = []
    pred_lines: list[str] = []

    # Correct local/full edge: track 10 and newborn 20 are both GT 1.
    _write_track(gt_lines, pred_lines, gt_id=1, pred_id=10, frames=range(1, 5), x=10)
    _write_track(gt_lines, pred_lines, gt_id=1, pred_id=20, frames=range(5, 9), x=12)

    # Wrong edge: candidate and newborn are different GT ids.
    _write_track(gt_lines, pred_lines, gt_id=2, pred_id=30, frames=range(1, 5), x=100)
    _write_track(gt_lines, pred_lines, gt_id=3, pred_id=40, frames=range(5, 9), x=200)

    # Polluted candidate: full majority starts as GT 4, but the local tail is GT 5.
    _write_track(gt_lines, pred_lines, gt_id=4, pred_id=50, frames=range(1, 4), x=300)
    _write_track(gt_lines, pred_lines, gt_id=5, pred_id=50, frames=range(4, 7), x=400)
    _write_track(gt_lines, pred_lines, gt_id=5, pred_id=60, frames=range(7, 11), x=402)

    # Rejected candidate must still be labeled; accepted-only logs would hide it.
    _write_track(gt_lines, pred_lines, gt_id=6, pred_id=70, frames=range(1, 4), x=500)
    _write_track(gt_lines, pred_lines, gt_id=7, pred_id=80, frames=range(4, 8), x=600)

    (gt_dir / "gt.txt").write_text("".join(gt_lines))
    (baseline_dir / f"{seq}.txt").write_text("".join(pred_lines))

    rows = [
        _handover_row(
            seq=seq,
            candidate_id=10,
            candidate_start=1,
            candidate_end=4,
            newborn_id=20,
            newborn_start=5,
            newborn_end=8,
            accepted=True,
            best_cost=0.2,
            margin=0.15,
            match_iou=0.6,
            neighbor_iou=0.05,
            head_tail_neighbor_iou=0.05,
            center_dist_norm=0.3,
            candidate_count=6,
            key_best_sim=0.93,
            key_margin=0.18,
        ),
        _handover_row(
            seq=seq,
            candidate_id=30,
            candidate_start=1,
            candidate_end=4,
            newborn_id=40,
            newborn_start=5,
            newborn_end=8,
            accepted=True,
            best_cost=0.6,
            margin=0.01,
            match_iou=0.05,
            neighbor_iou=0.1,
            head_tail_neighbor_iou=0.1,
            center_dist_norm=3.0,
            candidate_count=4,
            key_best_sim=0.40,
            key_margin=0.01,
        ),
        _handover_row(
            seq=seq,
            candidate_id=50,
            candidate_start=1,
            candidate_end=6,
            newborn_id=60,
            newborn_start=7,
            newborn_end=10,
            accepted=True,
            best_cost=0.22,
            margin=0.18,
            match_iou=0.75,
            neighbor_iou=0.8,
            head_tail_neighbor_iou=0.8,
            center_dist_norm=0.2,
            candidate_count=8,
            key_best_sim=0.88,
            key_margin=0.02,
        ),
        _handover_row(
            seq=seq,
            candidate_id=70,
            candidate_start=1,
            candidate_end=3,
            newborn_id=80,
            newborn_start=4,
            newborn_end=7,
            accepted=False,
            best_cost=0.55,
            margin=0.02,
            match_iou=0.05,
            neighbor_iou=0.2,
            head_tail_neighbor_iou=0.2,
            center_dist_norm=4.0,
            candidate_count=3,
            key_best_sim=0.45,
            key_margin=-0.04,
        ),
    ]
    handover_log = tmp_path / "_cheb_gr_offline_handover.csv"
    with handover_log.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out_csv = tmp_path / "labeled.csv"
    registry_md = tmp_path / "parameter_registry.md"
    summary_json = tmp_path / "parameter_summary.json"
    script = Path("scripts/eval/diagnostics/cheb_gr_offline_handover_report.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--handover-log",
            str(handover_log),
            "--baseline-dir",
            str(baseline_dir),
            "--gt-root",
            str(tmp_path / "gt_root"),
            "--edge-window",
            "2",
            "--out-csv",
            str(out_csv),
            "--registry-md",
            str(registry_md),
            "--summary-json",
            str(summary_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "rows=4 accepted=3 known=4 unknown=0" in result.stdout
    assert "accepted known: correct=2 wrong=1 precision=0.667" in result.stdout
    assert "candidate_full_tail_mismatch=1" in result.stdout

    labeled = list(csv.DictReader(out_csv.open()))
    by_candidate = {int(row["candidate_id"]): row for row in labeled}
    assert by_candidate[10]["label"] == "correct"
    assert by_candidate[30]["label"] == "wrong"
    assert by_candidate[70]["accepted"] == "False"
    assert by_candidate[70]["label"] == "wrong"

    polluted = by_candidate[50]
    assert polluted["label"] == "correct"
    assert polluted["same_gt_local"] == "True"
    assert polluted["same_gt_full"] == "False"
    assert polluted["candidate_full_tail_same"] == "False"

    registry = registry_md.read_text()
    assert "## Candidate Rule Map" in registry
    assert "## Candidate Rule Policy Simulation" in registry
    assert "## Endpoint Pollution Evidence" in registry
    assert "endpoint polluted rows: `1/4 = 0.250`" in registry
    assert "### Pollution Matrix: match_iou x neighbor_iou" in registry

    summary = json.loads(summary_json.read_text())
    assert summary["schema"] == "cheb_gr_offline_handover_summary/v1"
    assert summary["counts"] == {
        "accepted": 3,
        "accepted_correct": 2,
        "accepted_known": 3,
        "accepted_precision": 2 / 3,
        "accepted_wrong": 1,
        "correct": 2,
        "known": 4,
        "rows": 4,
        "same_rate": 0.5,
        "unknown": 0,
        "wrong": 2,
    }
    assert "best_cost" in summary["features"]
    assert "key_best_sim" in summary["features"]
    assert "key_margin" in summary["features"]
    assert summary["features"]["neighbor_iou"]["failure_mode"].startswith(
        "Not a standalone identity signal"
    )
    assert summary["features"]["key_margin"]["failure_mode"].startswith(
        "No hard negative"
    )
    discovered = summary["discovered_gates"]
    assert discovered["all_known_single_feature"]
    assert discovered["accepted_known_single_feature"]
    assert discovered["accepted_known_two_feature"]
    assert {
        "expression",
        "gates",
        "selected",
        "correct",
        "wrong",
        "precision",
        "correct_recall",
        "wrong_keep",
    } <= set(discovered["all_known_single_feature"][0])
    assert {rule["name"] for rule in summary["candidate_rules"]} >= {
        "best_cost_accept_strict",
        "center_dist_danger",
        "margin_x_candidate_count_accept",
    }
    center_policy = next(
        r for r in summary["policy_simulation"] if r["name"] == "center_dist_danger"
    )
    assert center_policy["action"] == "cut"
    assert center_policy["wrong_cut"] == 1
    assert summary["pollution"]["eligible"] == 4
    assert summary["pollution"]["endpoint_polluted"] == 1
    assert summary["pollution"]["pollution_rate"] == 0.25
    assert any(
        cell["neighbor_iou"] == "[0.7,inf)"
        and cell["match_iou"] == "[0.7,0.9)"
        and cell["endpoint_polluted"] == 1
        for cell in summary["pollution"]["matrices"][0]["cells"]
    )
