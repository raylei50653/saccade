"""The #465 PR-2 head parity runner implements its frozen declaration.

``scripts/eval/diagnostics/native_head_parity.py`` turns measurements into one
of the five terminals of
``docs/reference/native_runtime_head_parity_declaration.md``. The measurement
needs a GPU, engines and gitignored checkpoints; the decision logic and the
constants copied from the declaration do not, so they are pinned here: the
tolerance formula (floor, cap, two-sided), the κ_L1 thresholds, the terminal
order, the first-divergence scan, the unrounded metric formulas, and the
declaration blob the runner refuses to run without.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "eval" / "diagnostics" / "native_head_parity.py"


def _tool():
    spec = importlib.util.spec_from_file_location("native_head_parity", TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _tool()


def test_declaration_blob_is_the_frozen_one():
    blob = subprocess.run(
        ["git", "hash-object", P.DECLARATION],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # An §9 amendment changes the blob; the runner constant must move with it.
    assert blob == P.FROZEN_DECLARATION_BLOB


def test_constants_appear_in_the_declaration():
    text = (REPO / P.DECLARATION).read_text()
    assert P.HEAD_ONNX_SHA256 in text
    assert P.CKPT_SHA256 in text
    assert ",".join(P.SEQUENCES) in text
    assert f"共 {P.TOTAL_FRAMES} frames" in text
    assert P.HEAD_ENGINE in text
    assert "`score_maxabs ≤ 0.05` **且** `box_maxabs_px ≤ 4.0`" in text
    assert (
        "floor = 0.20 pt（IDF1／HOTA／MOTA）、5（IDs）；cap = 1.00 pt、30（IDs）"
        in text
    )
    assert "`A_C#1, A_T#1, A_N#1, A_C#2, A_T#2, A_N#2`" in text
    assert [f"{a}#{r}" for a, r in P.RUN_ORDER] == [
        "A_C#1",
        "A_T#1",
        "A_N#1",
        "A_C#2",
        "A_T#2",
        "A_N#2",
    ]
    assert P.ARMS == {
        "A_C": [],
        "A_T": ["--mamba-head-engine", P.HEAD_ENGINE],
        "A_N": ["--no-compile"],
    }


@pytest.mark.parametrize(
    ("delta_n", "metric", "expected"),
    [
        (0.0, "IDF1", 0.20),  # floor when the reference equals the oracle
        (-0.35, "HOTA", 0.35),  # |Δ_N|, sign ignored
        (2.5, "MOTA", 1.00),  # cap
        (0.0, "IDs", 5.0),
        (-12.0, "IDs", 12.0),
        (80.0, "IDs", 30.0),
    ],
)
def test_tolerance(delta_n, metric, expected):
    assert P.tolerance(delta_n, metric) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("score", "box", "verdict"),
    [
        (0.05, 4.0, "L1_PASS"),  # inclusive bounds
        (0.0500001, 0.0, "L1_GROSS_ERROR"),
        (0.0, 4.0001, "L1_GROSS_ERROR"),
        (math.nan, 0.0, "L1_GROSS_ERROR"),
        (0.0, math.inf, "L1_GROSS_ERROR"),
    ],
)
def test_l1_verdict(score, box, verdict):
    assert P.l1_verdict(score, box) == verdict


def _m(idf1=80.0, hota=65.0, mota=75.0, ids=400.0):
    return {"IDF1": idf1, "HOTA": hota, "MOTA": mota, "IDs": ids}


def test_l2_exact_wins_over_deltas():
    verdict, _ = P.l2_verdict(_m(), _m(idf1=90.0), _m(), exact=True)
    assert verdict == "L2_EXACT"


def test_l2_within_uses_reference_scale():
    verdict, detail = P.l2_verdict(_m(), _m(idf1=80.3), _m(idf1=79.6), exact=False)
    assert verdict == "L2_WITHIN"
    assert detail["IDF1"]["tolerance"] == pytest.approx(0.4)


def test_l2_is_two_sided():
    better, _ = P.l2_verdict(_m(), _m(idf1=80.25), _m(), exact=False)
    worse, _ = P.l2_verdict(_m(), _m(idf1=79.75), _m(), exact=False)
    assert better == worse == "L2_OUT"


def test_l2_any_metric_out_fails():
    verdict, detail = P.l2_verdict(_m(), _m(ids=406.0), _m(), exact=False)
    assert verdict == "L2_OUT"
    assert not detail["IDs"]["within"]
    assert detail["IDF1"]["within"]


def test_l2_cap_bounds_a_noisy_reference():
    verdict, _ = P.l2_verdict(_m(), _m(mota=73.9), _m(mota=72.0), exact=False)
    assert verdict == "L2_OUT"  # |Δ_T| 1.1 > cap 1.0 although |Δ_N| is 3.0


@pytest.mark.parametrize(
    ("valid", "l1", "l2", "terminal"),
    [
        (False, "L1_PASS", "L2_EXACT", "UNRESOLVED"),
        (True, None, "L2_EXACT", "UNRESOLVED"),
        (True, "L1_PASS", None, "UNRESOLVED"),
        (True, "L1_GROSS_ERROR", "L2_EXACT", "HEAD_PARITY_GROSS_ERROR"),
        (True, "L1_PASS", "L2_EXACT", "HEAD_PARITY_EXACT"),
        (True, "L1_PASS", "L2_WITHIN", "HEAD_PARITY_WITHIN_TOLERANCE"),
        (True, "L1_PASS", "L2_OUT", "HEAD_PARITY_OUT_OF_TOLERANCE"),
        (True, "bogus", "L2_EXACT", "UNRESOLVED"),
    ],
)
def test_terminal_order(valid, l1, l2, terminal):
    assert P.decide_terminal(valid, l1, l2) == terminal
    assert terminal in P.TERMINALS


def test_first_divergent_frame():
    a = b"1,1,0,0,1,1,1,-1,-1,-1\n2,1,0,0,1,1,1,-1,-1,-1\n3,1,0,0,1,1,1,-1,-1,-1\n"
    b = b"1,1,0,0,1,1,1,-1,-1,-1\n2,1,0,0,1,1,1,-1,-1,-1\n3,2,0,0,1,1,1,-1,-1,-1\n"
    assert P.first_divergent_frame(a, a) is None
    assert P.first_divergent_frame(a, b) == 3
    assert P.first_divergent_frame(a, a[: a.index(b"3,")]) == 3  # missing frame
    assert P.first_divergent_frame(a, a + b"\n") == -1  # rows equal, bytes not


def test_metrics_from_counts_match_the_harness_formulas():
    from saccade.perception.eval.metrics import _format_overall_metrics_from_counts

    counts = {
        "idtp": 70123,
        "idfp": 9876,
        "idfn": 14321,
        "num_false_positives": 3456,
        "num_misses": 17890,
        "num_switches": 412,
        "num_objects": 112297,
        "num_detections": 94407,
        "num_predictions": 97863,
    }
    got = P.metrics_from_counts(counts, {"HOTA": 0.65, "DetA": 0.6, "AssA": 0.7})
    shown = _format_overall_metrics_from_counts(counts)
    assert shown["IDF1"] == f"{got['IDF1']:.1f}%"
    assert shown["MOTA"] == f"{got['MOTA']:.1f}%"
    assert shown["IDs"] == got["IDs"]
    assert got["HOTA"] == pytest.approx(65.0)


def test_hist_quantile_is_an_upper_bound():
    edges = P.hist_edges()
    counts = [0] * (len(edges) + 2)
    counts[0] = 998  # exact zeros
    counts[500] = 2  # values in [edges[498], edges[499])
    assert P.hist_quantile(counts, 0.998) == 0.0
    assert P.hist_quantile(counts, 0.999) == edges[499]
    counts[-1] = 10  # overflow
    assert P.hist_quantile(counts, 1.0) == math.inf
    assert P.hist_quantile([0] * len(counts), 0.999) is None
