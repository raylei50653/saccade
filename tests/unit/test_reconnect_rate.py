"""Unit tests for B2 reconnect_rate summarization / export."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eval.diagnostics.reconnect_rate import (
    analyze_sequence,
    summarize_records,
)


def test_analyze_success_and_fail() -> None:
    # GT id 1 present frames 1,2 then gap then 5,6
    # success: pred 7 before and after; fail case separate gt
    gt = {
        1: {1: (0, 0, 10, 10)},
        2: {1: (0, 0, 10, 10)},
        5: {1: (0, 0, 10, 10)},
        6: {1: (0, 0, 10, 10)},
    }
    pred_ok = {
        1: {7: (0, 0, 10, 10)},
        2: {7: (0, 0, 10, 10)},
        5: {7: (0, 0, 10, 10)},
        6: {7: (0, 0, 10, 10)},
    }
    recs = analyze_sequence(gt, pred_ok, iou_thr=0.5, min_gap=1)
    assert len(recs) == 1
    assert recs[0]["success"] == 1
    assert recs[0]["gap"] == 2

    pred_fail = {
        1: {7: (0, 0, 10, 10)},
        2: {7: (0, 0, 10, 10)},
        5: {8: (0, 0, 10, 10)},
        6: {8: (0, 0, 10, 10)},
    }
    recs_f = analyze_sequence(gt, pred_fail, iou_thr=0.5, min_gap=1)
    assert len(recs_f) == 1
    assert recs_f[0]["success"] == 0


def test_summarize_and_json_roundtrip(tmp_path: Path) -> None:
    records = [
        {"gap": 5, "disp": 0.1, "success": 1, "seq": "s"},
        {"gap": 5, "disp": 0.2, "success": 0, "seq": "s"},
        {"gap": 40, "disp": 0.3, "success": 1, "seq": "s"},
    ]
    s = summarize_records(records, "bridge_on")
    assert s["n_opportunities"] == 3
    assert s["n_success"] == 2
    assert abs(s["rate"] - 2 / 3) < 1e-9
    assert any(b["gap"] == "1-10" for b in s["by_gap"])
    out = tmp_path / "metrics_reconnect.json"
    out.write_text(json.dumps({"pred": s}))
    loaded = json.loads(out.read_text())
    assert loaded["pred"]["label"] == "bridge_on"
