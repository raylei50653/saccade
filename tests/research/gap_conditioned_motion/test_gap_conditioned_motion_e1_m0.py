from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RUNNER = (
    REPO / "docs/modules/semantic/research/evidence/"
    "gap_conditioned_motion_e1_m0_20260711/run_e1_m0.py"
)
FIELDS = [
    "seq",
    "gt_match",
    "gt_valid",
    "gap",
    "bridge_dist",
    "lost_exit_speed",
    "cand_entry_speed",
    "dir_cos",
    "fwd_resid",
    "bwd_resid",
]


def _load_runner():
    spec = importlib.util.spec_from_file_location("gap_motion_e1", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pairs(path: Path) -> None:
    rows = []
    for index in range(20):
        is_gt = index >= 10
        mismatch = float(index)
        rows.append(
            {
                "seq": "MOT17-10-SDP",
                "gt_match": str(int(is_gt)),
                "gt_valid": "1",
                "gap": "5",
                "bridge_dist": mismatch,
                "lost_exit_speed": mismatch,
                "cand_entry_speed": "0",
                "dir_cos": 1.0 - mismatch,
                "fwd_resid": mismatch,
                "bwd_resid": mismatch,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_auc_is_tie_aware():
    runner = _load_runner()
    assert runner._auc([False, True, True], [0.0, 1.0, 1.0]) == 1.0
    assert runner._auc([False, True], [1.0, 1.0]) == 0.5


def test_m0_flags_predeclared_reversal_pattern(tmp_path):
    runner = _load_runner()
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs)

    result = runner.analyze(pairs)
    short_cells = [cell for cell in result["cells"] if cell["gap_bin"] == "1-10"]

    assert len(short_cells) == 4
    assert all(cell["auc_gt_low_mismatch"] < 0.5 for cell in short_cells)
    assert all(cell["role_reversal_descriptive"] for cell in short_cells)
