"""Tests for scripts/tools/summarize_relink_pairs.py B1 output contract."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
SCRIPT = project_root / "scripts" / "tools" / "summarize_relink_pairs.py"


def _write_pairs(path: Path) -> None:
    # Mix of easy far negatives and near hard-pool rows.
    rows = [
        # pos near
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 1,
            "cand_id": 10,
            "gt_match": 1,
            "gt_valid": 1,
            "bridge_dist": 0.2,
            "gap": 5,
            "lost_last_frame": 10,
            "cand_first_frame": 15,
        },
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 2,
            "cand_id": 11,
            "gt_match": 1,
            "gt_valid": 1,
            "bridge_dist": 0.4,
            "gap": 12,
            "lost_last_frame": 20,
            "cand_first_frame": 32,
        },
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 3,
            "cand_id": 12,
            "gt_match": 1,
            "gt_valid": 1,
            "bridge_dist": 0.8,
            "gap": 40,
            "lost_last_frame": 30,
            "cand_first_frame": 70,
        },
        # neg near (hard)
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 4,
            "cand_id": 13,
            "gt_match": 0,
            "gt_valid": 1,
            "bridge_dist": 0.9,
            "gap": 8,
            "lost_last_frame": 40,
            "cand_first_frame": 48,
        },
        # neg far (easy)
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 5,
            "cand_id": 14,
            "gt_match": 0,
            "gt_valid": 1,
            "bridge_dist": 3.0,
            "gap": 20,
            "lost_last_frame": 50,
            "cand_first_frame": 70,
        },
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 6,
            "cand_id": 15,
            "gt_match": 0,
            "gt_valid": 1,
            "bridge_dist": 4.5,
            "gap": 100,
            "lost_last_frame": 60,
            "cand_first_frame": 160,
        },
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 7,
            "cand_id": 16,
            "gt_match": 0,
            "gt_valid": 1,
            "bridge_dist": 5.0,
            "gap": 200,
            "lost_last_frame": 70,
            "cand_first_frame": 270,
        },
        # invalid row ignored
        {
            "seq": "MOT17-04-SDP",
            "lost_id": 8,
            "cand_id": 17,
            "gt_match": 1,
            "gt_valid": 0,
            "bridge_dist": 0.1,
            "gap": 3,
            "lost_last_frame": 1,
            "cand_first_frame": 4,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_summarize_writes_b1_artifacts(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.csv"
    study = tmp_path / "m_b1_test"
    _write_pairs(pairs)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pairs",
            str(pairs),
            "--study-dir",
            str(study),
            "--hard-dist",
            "1.0",
            "--commit",
            "testsha",
            "--preset",
            "mamba_whole_graph_m",
            "--min-n",
            "4",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (study / "context.json").is_file()
    assert (study / "metrics_auc.json").is_file()
    assert (study / "metrics_thr.csv").is_file()

    ctx = json.loads((study / "context.json").read_text(encoding="utf-8"))
    assert ctx["commit"] == "testsha"
    assert ctx["preset"] == "mamba_whole_graph_m"
    assert ctx["pool"]["full"]["n"] == 7
    assert ctx["pool"]["full"]["n_pos"] == 3
    assert ctx["pool"]["hard"]["n"] == 4  # three pos + one near neg
    assert ctx["pool"]["hard"]["n_pos"] == 3
    assert "1-10" in ctx["gap_bins"]
    assert ctx["score_dist"]["pos"]["median"] < ctx["score_dist"]["neg"]["median"]

    auc = json.loads((study / "metrics_auc.json").read_text(encoding="utf-8"))
    assert auc["score_field"] == "bridge_dist"
    assert auc["full"]["n_pos"] == 3
    assert auc["full"]["skipped_reason"] == ""
    assert auc["full"]["auc"] > 0.5

    thr_text = (study / "metrics_thr.csv").read_text(encoding="utf-8")
    assert "pool,threshold" in thr_text
    assert "full" in thr_text and "hard" in thr_text


def test_summarize_missing_columns_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("seq,lost_id\nx,1\n", encoding="utf-8")
    study = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pairs",
            str(bad),
            "--study-dir",
            str(study),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "missing columns" in (proc.stdout + proc.stderr)
