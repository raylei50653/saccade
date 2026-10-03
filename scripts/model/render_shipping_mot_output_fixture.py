#!/usr/bin/env python3
"""Render the golden fixture for the native MOT output path.

Issue #465 Phase B PR-6 (U4; boundary §5 B3, B4). The native host turns
tracker rows into MOT lines and applies the sequence tail with twins of the
oracle's Python functions (``shipping/src/mot_output.cpp``):

* **B3** -- ``helpers.fast_emit_mot_lines`` (the headline's fast emit) with a
  ``tracking.GlobalTrackIdMapper`` for one sequence: ids in first-appearance
  order from 1, one ``frame,id,x,y,w,h,score,-1,-1,-1`` line per tracker row;
* **B4** -- ``post_merge.interpolate_tracklets`` (pandas parse, float64
  interpolation, ``:.2f``/``:.4f`` formatting, stable ``(frame, id)`` merge).

This tool runs those Python functions and writes their outputs to
``tests/native/fixtures/shipping_mot_output.json``, which
``tests/native/test_shipping_mot_output.cpp`` (CI job ``shipping-config-loader``)
replays through the C++ twins and compares byte for byte. Tracker floats are
stored as IEEE-754 binary32 bit patterns; MOT lines as text.

Emit inputs place float32 values on, and one ulp either side of, the decimal
rounding midpoints of both precisions, plus signed zero, tiny negatives, huge
values, NaN and infinities. Interpolation cases cover the boundaries the
boundary doc names (gap exactly ``max_gap`` and one more, track length exactly
``min_track_len`` and one less, single-frame tracks, ``min_h`` on and just
below the threshold), the early returns (empty input, ``max_gap <= 0``, no
confirmed track, nothing to fill -- the input comes back unsorted), duplicate
``(id, frame)`` rows, non-finite fields, and a seeded random stress set run
with the headline parameters and a second parameter set. The tool also checks
that pandas parses every float field of the interpolation inputs to the
correctly rounded double (what the C++ parser computes).

Usage:
  .venv/bin/python scripts/model/render_shipping_mot_output_fixture.py
  .venv/bin/python scripts/model/render_shipping_mot_output_fixture.py --check
"""

# status: stable

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from saccade.perception.eval.helpers import fast_emit_mot_lines  # noqa: E402
from saccade.perception.eval.post_merge import interpolate_tracklets  # noqa: E402
from saccade.perception.eval.tracking import GlobalTrackIdMapper  # noqa: E402

CONFIG = REPO / "configs" / "shipping" / "mamba_whole_graph.resolved.json"
OUTPUT = REPO / "tests" / "native" / "fixtures" / "shipping_mot_output.json"
FORMAT = "saccade.shipping_mot_output_fixture/v1"
SEED = 465
SEQUENCE = "FIXTURE-SEQ"

_SCALAR_LIST = re.compile(r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]")


def _bits(x: np.ndarray) -> list[int]:
    return np.asarray(x, dtype="<f4").reshape(-1).view("<u4").astype(np.int64).tolist()


def _f32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _neighbours(values: np.ndarray) -> np.ndarray:
    """Each float32 value and its two float32 neighbours."""
    v = _f32(values)
    return np.concatenate(
        [v, np.nextafter(v, np.float32(-np.inf)), np.nextafter(v, np.float32(np.inf))]
    )


def _headline_interpolation() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["host_params"]["cfg"]
    return {
        "max_gap": cfg["interpolate_max_gap"],
        "min_track_len": cfg["interpolate_min_track_len"],
        "min_h": float(cfg["interpolate_min_h"]),
    }


# ---------------------------------------------------------------------------
# B3: ids + lines
# ---------------------------------------------------------------------------


def _emit_frames(
    frames: list[tuple[int, np.ndarray, np.ndarray, list[int]]],
) -> dict[str, Any]:
    mapper = GlobalTrackIdMapper()
    lines: list[str] = []
    recorded = []
    for frame_id, boxes, scores, ids in frames:
        boxes = _f32(boxes).reshape(-1, 4)
        scores = _f32(scores)
        track_results = {
            "count": len(ids),
            "boxes": torch.from_numpy(boxes.copy()),
            "scores": torch.from_numpy(scores.copy()),
            "ids": torch.tensor(ids, dtype=torch.int32),
        }
        lines += fast_emit_mot_lines(
            track_results=track_results,
            global_id_mapper=mapper,
            seq=SEQUENCE,
            frame_id=frame_id,
            frame_w=1920,
            frame_h=1080,
        )
        recorded.append(
            {
                "frame": frame_id,
                "boxes_bits": _bits(boxes),
                "scores_bits": _bits(scores),
                "ids": list(ids),
            }
        )
    return {"frames": recorded, "expected_lines": lines}


def _emit_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    # Decimal midpoints of the 2- and 4-digit roundings, as float32, +-1 ulp.
    mid2 = _neighbours(np.arange(-3, 40) + 0.005 + 0.01 * rng.integers(0, 100, 43))
    mid2 = np.concatenate(
        [mid2, _neighbours(rng.uniform(-100, 4000, 30).round(2) + 0.005)]
    )
    mid4 = _neighbours(rng.integers(0, 10000, 60) / 10000 + 0.00005)
    rows = len(mid2)
    x1 = mid2
    y1 = rng.permutation(mid2)
    x2 = (x1 + _f32(rng.uniform(0, 300, rows))).astype(np.float32)
    y2 = (y1 + _f32(rng.uniform(0, 600, rows))).astype(np.float32)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = np.resize(mid4, rows)
    frames: list[tuple[int, np.ndarray, np.ndarray, list[int]]] = []
    pool = [int(v) for v in rng.permutation(np.arange(1, 400))[:60]]
    start = 0
    frame = 1
    while start < rows:
        n = int(rng.integers(0, 12))
        ids = [
            int(v) for v in rng.choice(pool, size=min(n, rows - start), replace=False)
        ]
        end = start + len(ids)
        frames.append((frame, boxes[start:end], scores[start:end], ids))
        start = end
        frame += int(rng.integers(1, 4))
    special_boxes = _f32(
        [
            [-0.0, 0.0, -0.0, 0.0],
            [-0.001, -0.004, -0.0049, 0.0049],
            [1.0e7, 3.0e38, 1.0e7 + 1.0, 3.4e38],
            [-3.4e38, 12.5, 3.4e38, 12.5],
            [np.nan, 1.0, 2.0, np.nan],
            [np.inf, -np.inf, np.inf, 5.0],
            [0.125, 0.375, 0.625, 0.875],
            [2.675, 1.005, 1.115, 8.345],
        ]
    )
    special_scores = _f32([0.0, -0.0, 1.0, 0.99995, 0.00005, np.nan, np.inf, 0.12345])
    special = [
        (7, special_boxes[:4], special_scores[:4], [9, 2147483647, 1, 0]),
        (8, special_boxes[4:], special_scores[4:], [2, -5, 9, 1000]),
        (9, special_boxes[:0], special_scores[:0], []),
    ]
    return [
        {"name": "rounding_midpoints", **_emit_frames(frames)},
        {"name": "special_values", **_emit_frames(special)},
    ]


# ---------------------------------------------------------------------------
# B4: interpolation
# ---------------------------------------------------------------------------


def _line(
    frame: int, tid: int, x: float, y: float, w: float, h: float, s: float
) -> str:
    return f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1"


def _track(
    tid: int, frames: list[int], h: float = 120.0, x0: float = 10.0
) -> list[str]:
    return [
        _line(
            f,
            tid,
            x0 + 3.37 * i,
            50.0 - 1.13 * i,
            40.0 + 0.71 * i,
            h + 0.29 * i,
            0.5 + 0.0123 * i,
        )
        for i, f in enumerate(frames)
    ]


def _by_frame(lines: list[str]) -> list[str]:
    """Emission order: frame by frame (stable), as the tracker emits."""
    return sorted(lines, key=lambda ln: int(ln.split(",", 1)[0]))


def _random_lines(rng: np.random.Generator, tracks: int) -> list[str]:
    lines: list[str] = []
    ids = rng.permutation(np.arange(1, tracks * 3))[:tracks]
    for tid in ids:
        n = int(rng.integers(1, 16))
        frame = int(rng.integers(1, 200))
        x, y = rng.uniform(-60, 1900), rng.uniform(-60, 1000)
        w, h = rng.uniform(5, 200), rng.uniform(10, 500)
        for _ in range(n):
            lines.append(_line(frame, int(tid), x, y, w, h, rng.uniform(0, 1)))
            frame += int(rng.choice([1, 1, 1, 2, 3, 4, 7, 11, 12, 35, 36, 37]))
            x += rng.normal(0, 7)
            y += rng.normal(0, 3)
            w = max(1.0, w + rng.normal(0, 2))
            h = max(1.0, h + rng.normal(0, 4))
    rng.shuffle(lines)
    return _by_frame(lines)


def _interp_case(name: str, lines: list[str], params: dict[str, Any]) -> dict[str, Any]:
    _check_pandas_parse(lines)
    with np.errstate(invalid="ignore"):  # inf - inf in the non_finite case
        out, stats = interpolate_tracklets(list(lines), **params)
    return {
        "name": name,
        "params": params,
        "lines": lines,
        "expected_lines": out,
        "expected_stats": stats,
    }


def _check_pandas_parse(lines: list[str]) -> None:
    """pandas' C parser must give the correctly rounded double (Python float)
    for every float field -- the C++ twin parses with std::from_chars."""
    if not lines:
        return
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        header=None,
        names=["frame", "tid", "x", "y", "w", "h", "score"],
        usecols=[0, 1, 2, 3, 4, 5, 6],
        dtype={
            "frame": int,
            "tid": int,
            "x": float,
            "y": float,
            "w": float,
            "h": float,
            "score": float,
        },
    )
    want = np.array(
        [[float(v) for v in ln.split(",")[2:7]] for ln in lines], dtype=np.float64
    )
    got = df[["x", "y", "w", "h", "score"]].to_numpy(dtype=np.float64)
    same = (got.view(np.uint64) == want.view(np.uint64)) | (
        np.isnan(got) & np.isnan(want)
    )
    if not same.all():
        raise AssertionError("pandas parse differs from the correctly rounded double")


def _interp_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    head = _headline_interpolation()
    g = head["max_gap"]
    m = head["min_track_len"]
    boundaries = _by_frame(
        _track(
            1, [1, 2, 3, 3 + g + 1, 3 + g + 1 + g + 2, 3 + 2 * g + 4]
        )  # gap = g, g + 1, 1
        + _track(2, list(range(1, m + 1)[:-1]) + [m + 4])  # length m, one gap of 3
        + _track(3, list(range(1, m))[:-1] + [m + 3])  # length m - 1: not confirmed
        + _track(4, [17])  # single frame
        + _track(5, [5, 7, 9, 11, 13, 15])  # gaps of 1
    )
    min_h_lines = _by_frame(
        _track(10, [1, 4, 8, 12, 20, 22], h=50.0, x0=3.0)
        + _track(11, [2, 6, 7, 9, 15], h=49.99, x0=8.0)
        + [
            _line(30, 12, 1.0, 1.0, 2.0, 50.0, 0.9),
            _line(33, 12, 1.0, 1.0, 2.0, 50.0, 0.9),
        ]
        + [
            _line(30, 13, 1.0, 1.0, 2.0, 50.0, 0.9),
            _line(33, 13, 1.0, 1.0, 2.0, 49.99, 0.9),
        ]
    )
    duplicate = _by_frame(
        _track(20, [1, 2, 6, 6, 9, 14])
        + [_line(6, 20, 99.99, -0.01, 1.0, 2.0, 0.1)]
        + _track(21, [3, 4, 5, 8, 9])
    )
    no_gaps = list(
        reversed(_track(30, [1, 2, 3, 4, 5]) + _track(31, [2, 3, 4, 5, 6, 7]))
    )
    nonfinite = [
        "1,40,nan,10.00,5.00,inf,0.5000,-1,-1,-1",
        "2,40,1.00,10.00,5.00,20.00,0.5000,-1,-1,-1",
        "5,40,4.00,-inf,5.00,20.00,nan,-1,-1,-1",
        "6,40,5.00,11.00,5.00,20.00,0.5000,-1,-1,-1",
        "9,40,8.00,12.00,inf,-inf,0.7500,-1,-1,-1",
        "13,40,-0.00,-0.01,0.00,0.01,0.0001,-1,-1,-1",
    ]
    stress = _random_lines(rng, 40)
    alt = {"max_gap": 10, "min_track_len": 3, "min_h": 60.0}
    return [
        _interp_case("boundaries", boundaries, head),
        _interp_case("min_h", min_h_lines, {**head, "min_h": 50.0}),
        _interp_case("duplicate_frames", duplicate, head),
        _interp_case("nothing_to_fill_keeps_order", no_gaps, head),
        _interp_case(
            "no_confirmed_track", _track(32, [1, 5]) + _track(33, [2, 9]), head
        ),
        _interp_case("empty", [], head),
        _interp_case(
            "max_gap_zero", _track(34, [1, 3, 5, 7, 9]), {**head, "max_gap": 0}
        ),
        _interp_case("non_finite", nonfinite, {**head, "min_track_len": 1}),
        _interp_case("random_headline", stress, head),
        _interp_case("random_alt_params", stress, alt),
    ]


def render() -> str:
    rng = np.random.default_rng(SEED)
    doc = {
        "format": FORMAT,
        "generator": "scripts/model/render_shipping_mot_output_fixture.py",
        "note": "tracker floats are IEEE-754 binary32 bit patterns; do not edit by hand",
        "headline_interpolation": _headline_interpolation(),
        "emit_cases": _emit_cases(rng),
        "interpolation_cases": _interp_cases(rng),
    }
    text = json.dumps(doc, indent=1, separators=(",", ": "))
    text = _SCALAR_LIST.sub(
        lambda mt: "[" + ", ".join(mt.group(1).split()).replace(",,", ",") + "]", text
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
