#!/usr/bin/env python3
"""Render the golden fixture for the native host's CPU detection filters.

Issue #465 Phase B PR-5 (U3a). The native post-detector host
(``shipping/src/post_detector_plan.cpp``) runs host-side twins of two Python
tensor filters the oracle applies between NMS and the tracker update
(``stages.py``, serial path):

* ``_apply_external_fp_filter`` in mode ``"rule"`` with the penalty branch off
  (the only mode ``plan_post_detector`` accepts) -- which rows survive;
* ``_fp_hard_reject_mask`` + ``masked_fill(reject, _FP_HARD_REJECT_SCORE)`` --
  which rows are rejected and the resulting scores.

This tool runs the Python functions themselves on CPU float32 tensors and
writes their results to ``tests/native/fixtures/shipping_detection_filters.json``,
which ``tests/native/test_shipping_detection_filters.cpp`` (CI job
``shipping-config-loader``) replays through the C++ twins bit for bit. Every
float is stored as its IEEE-754 binary32 bit pattern.

Two threshold sets are recorded: ``headline`` (read from the committed
``configs/shipping/mamba_whole_graph.resolved.json``; the C++ test also checks
that ``plan_post_detector`` derives exactly these float32 thresholds from that
file) and ``stress`` (values not representable in binary32). Inputs mix seeded
random rows with rows placed exactly on, and one ulp either side of, every
threshold and its float32 rounding, plus degenerate boxes (width/height <= 0
hit the ``clamp(min=1e-6)``), so the fixture also pins how torch compares a
float32 tensor with a Python float.

When CUDA is available the same inputs are also run on the GPU and must give
the same results (the oracle runs these ops on CUDA).

Usage:
  .venv/bin/python scripts/model/render_shipping_detection_filters_fixture.py
  .venv/bin/python scripts/model/render_shipping_detection_filters_fixture.py --check
"""

# status: stable

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from saccade.perception.eval.detection_filters import (  # noqa: E402
    _FP_HARD_REJECT_SCORE,
    _apply_external_fp_filter,
    _fp_hard_reject_mask,
)
from saccade.perception.eval.external_fp_model import RuleBaselineConfig  # noqa: E402

CONFIG = REPO / "configs" / "shipping" / "mamba_whole_graph.resolved.json"
OUTPUT = REPO / "tests" / "native" / "fixtures" / "shipping_detection_filters.json"
FORMAT = "saccade.shipping_detection_filters_fixture/v1"
SEED = 465
RANDOM_ROWS = 256

_SCALAR_LIST = re.compile(
    r"\[\s*((?:-?\d+|true|false)(?:,\s*(?:-?\d+|true|false))*)\s*\]"
)

RULE_KEYS = (
    "min_score",
    "low_score",
    "medium_score",
    "min_height",
    "medium_height",
    "min_aspect",
)
HARD_KEYS = ("min_score", "max_suspicious_area", "max_suspicious_score")


def _bits(x: np.ndarray) -> list[int]:
    return np.asarray(x, dtype="<f4").view("<u4").astype(np.int64).tolist()


def _f32_bits(value: float) -> int:
    return int(np.array([value], dtype="<f4").view("<u4")[0])


def _headline_thresholds(path: Path = CONFIG) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    hp = doc["host_params"]
    cfg = hp["cfg"]
    rule = hp["external_fp_rule_config"]
    return {
        "external_fp": {
            "max_score": cfg["external_fp_max_score"],
            **{k: rule[k] for k in RULE_KEYS},
        },
        "fp_hard": {
            "min_score": cfg["fp_hard_filter_min_score"],
            "max_suspicious_area": cfg["fp_hard_filter_max_suspicious_area"],
            "max_suspicious_score": cfg["fp_hard_filter_max_suspicious_score"],
            "reject_score": hp["fp_hard_reject_score"],
        },
    }


def _stress_thresholds() -> dict[str, Any]:
    return {
        "external_fp": {
            "max_score": 0.1 + 0.2,
            "min_score": 1.0 / 30.0,
            "low_score": 0.1 + 0.05,
            # Above max_score, so the subset boundary (score <= max_score) is
            # observable: a row at max_score is rejected only inside the subset.
            "medium_score": 0.3 + 0.0333,
            "min_height": 50.3,
            "medium_height": 90.7,
            "min_aspect": 1.0 / 0.7,
        },
        "fp_hard": {
            "min_score": 0.07 + 0.01,
            "max_suspicious_area": 12345,
            "max_suspicious_score": 0.3 + 0.0001,
            "reject_score": _FP_HARD_REJECT_SCORE,
        },
    }


def _around(value: float) -> list[np.float32]:
    """``value`` rounded to binary32 and its two neighbours."""
    f = np.float32(value)
    return [
        np.nextafter(f, np.float32(-np.inf)),
        f,
        np.nextafter(f, np.float32(np.inf)),
    ]


def _rows(th: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (n, 4) and scores (n,) as float32."""
    rng = np.random.default_rng(SEED)
    boxes: list[list[np.float32]] = []
    scores: list[np.float32] = []

    def add(
        score: float, w: float, h: float, x0: float = 100.0, y0: float = 100.0
    ) -> None:
        x0f, y0f = np.float32(x0), np.float32(y0)
        boxes.append(
            [x0f, y0f, np.float32(x0f + np.float32(w)), np.float32(y0f + np.float32(h))]
        )
        scores.append(np.float32(score))

    # Seeded random rows spanning both filters' ranges.
    for _ in range(RANDOM_ROWS):
        w = float(rng.uniform(1.0, 400.0))
        h = float(rng.uniform(1.0, 500.0))
        add(
            float(rng.uniform(0.0, 0.6)),
            w,
            h,
            float(rng.uniform(0, 1500)),
            float(rng.uniform(0, 800)),
        )

    ef, hd = th["external_fp"], th["fp_hard"]
    # External FP: score gates (with a box that passes the geometry tests).
    for key in ("max_score", "min_score", "low_score", "medium_score"):
        for s in _around(ef[key]):
            for h in (ef["min_height"] * 0.5, ef["medium_height"] * 2.0):
                add(float(s), 20.0, h)
    # External FP: height gates (exact heights via y0 = 0) inside the low-score subset.
    for key in ("min_height", "medium_height"):
        for hgt in _around(ef[key]):
            for s in (
                ef["min_score"],
                (ef["min_score"] + ef["low_score"]) / 2,
                ef["medium_score"],
            ):
                boxes.append(
                    [np.float32(0), np.float32(0), np.float32(10.0), np.float32(hgt)]
                )
                scores.append(np.float32(s))
    # External FP: aspect gate h/w around min_aspect at a short height.
    h0 = np.float32(ef["medium_height"] * 0.9)
    for a in _around(ef["min_aspect"]):
        for w in (
            h0 / a,
            np.nextafter(h0 / a, np.float32(0)),
            np.nextafter(h0 / a, np.float32(np.inf)),
        ):
            boxes.append([np.float32(0), np.float32(0), np.float32(w), h0])
            scores.append(np.float32((ef["low_score"] + ef["medium_score"]) / 2))
    # FP hard: score gates and the area gate (side = sqrt(area) on exact squares).
    for key in ("min_score", "max_suspicious_score"):
        for s in _around(hd[key]):
            for side in (50.0, 400.0):
                add(float(s), side, side)
    side = np.float32(np.sqrt(np.float32(hd["max_suspicious_area"])))
    for sd in (
        np.nextafter(side, np.float32(0)),
        side,
        np.nextafter(side, np.float32(np.inf)),
    ):
        boxes.append([np.float32(0), np.float32(0), sd, sd])
        scores.append(np.float32(hd["max_suspicious_score"]) - np.float32(0.01))
    # Degenerate boxes: zero / negative width and height (clamp(min=1e-6)).
    for x1, y1 in ((0.0, 0.0), (-5.0, 30.0), (30.0, -5.0), (0.0, 200.0), (200.0, 0.0)):
        for s in (ef["min_score"], ef["max_score"], hd["min_score"]):
            boxes.append([np.float32(0), np.float32(0), np.float32(x1), np.float32(y1)])
            scores.append(np.float32(s))
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _run(
    boxes: np.ndarray, scores: np.ndarray, th: dict[str, Any], device: str
) -> dict[str, Any]:
    b = torch.from_numpy(boxes).to(device)
    s = torch.from_numpy(scores).to(device)
    # Row index carried in the class column, to read survivors back.
    c = torch.arange(len(scores), dtype=torch.int32, device=device)
    ef, hd = th["external_fp"], th["fp_hard"]
    rule = RuleBaselineConfig(**{k: ef[k] for k in RULE_KEYS})
    _, ef_scores, ef_rows = _apply_external_fp_filter(
        b,
        s,
        c,
        image_width=1920,
        image_height=1080,
        mode="rule",
        rule_config=rule,
        logistic_model=None,
        logistic_threshold=0.5,
        max_score=ef["max_score"],
        penalty=1.0,
        min_score=0.0,
        softmax_min_scale=0.0,
    )
    if not torch.equal(ef_scores, s[ef_rows.long()]):
        raise AssertionError(
            "external FP rule filter changed a score with the penalty off"
        )
    reject = _fp_hard_reject_mask(b, s, **{k: hd[k] for k in HARD_KEYS})
    filled = s.masked_fill(reject, hd["reject_score"])
    return {
        "external_fp_kept_rows": ef_rows.cpu().tolist(),
        "fp_hard_reject": reject.cpu().tolist(),
        "fp_hard_scores_bits": _bits(filled.cpu().numpy()),
    }


def _case(name: str, th: dict[str, Any]) -> dict[str, Any]:
    boxes, scores = _rows(th)
    expected = _run(boxes, scores, th, "cpu")
    if torch.cuda.is_available() and _run(boxes, scores, th, "cuda") != expected:
        raise AssertionError(f"{name}: CPU and CUDA filter results differ")
    return {
        "name": name,
        "thresholds_bits": {
            group: {k: _f32_bits(v) for k, v in values.items()}
            for group, values in th.items()
        },
        "boxes_bits": _bits(boxes.reshape(-1)),
        "scores_bits": _bits(scores),
        "expected": expected,
    }


def render(config: Path = CONFIG) -> str:
    doc = {
        "format": FORMAT,
        "generator": "scripts/model/render_shipping_detection_filters_fixture.py",
        "note": "float values are IEEE-754 binary32 bit patterns; do not edit by hand",
        "cases": [
            _case("headline", _headline_thresholds(config)),
            _case("stress", _stress_thresholds()),
        ],
    }
    text = json.dumps(doc, indent=1, separators=(",", ": "))
    # One line per scalar list (bit patterns, row indices, masks).
    text = _SCALAR_LIST.sub(
        lambda m: "[" + ", ".join(m.group(1).split()).replace(",,", ",") + "]", text
    )
    return text + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed fixture is stale"
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print(
                f"{OUTPUT.relative_to(REPO)} is stale; re-run {Path(__file__).relative_to(REPO)}"
            )
            return 1
        print(f"{OUTPUT.relative_to(REPO)} is fresh")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
