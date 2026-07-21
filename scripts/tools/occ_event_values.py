#!/usr/bin/env python3
"""Per-event ACTUAL-VALUE table for the same-height occlusion gate (one run, no thresholding).

Instead of storing a gate decision (1/0 at one threshold), this dumps the *continuous* geometry
each crossing event actually takes — peak track-track IoU and |foot_gap|/h — so any
(occ_iou_thresh, occ_foot_gap) setting is derivable by comparison, and the optimal operating
band is read off as an **interval intersection** over events (no per-threshold re-run).

Events come from the GT crossing population (deterministic), each labelled with the GT-front
(higher-visibility occluder) and whether the foot-y front call points the right way — the only
thing the gate needs to get right.  Emits a CSV you can query for any threshold.

Usage
-----
  .venv/bin/python scripts/tools/occ_event_values.py --csv results/occ_tune/event_values.csv
"""
# status: stable

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import depth_ordering_probe as P  # reuse load_gt / iou / find_cross_events


def collect(
    root: Path,
    det: str,
    iou_extract: float,
    max_occ: int,
    min_life: int,
    pre_win: int,
    vis_margin: float,
) -> list[dict]:
    rows = []
    for seq in P.SEQS:
        gt_path = root / f"{seq}-{det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        gt = P.load_gt(gt_path)
        for ev in P.find_cross_events(gt, iou_extract, 0.3, max_occ, min_life, pre_win):
            a, b = ev["a"], ev["b"]
            vmax, vmin = max(ev["vis_a"], ev["vis_b"]), min(ev["vis_a"], ev["vis_b"])
            ambiguous = (vmax - vmin) < vis_margin or vmin >= 0.5
            gt_front = a if ev["vis_a"] > ev["vis_b"] else b
            ious = [(P.iou(gt[a][f], gt[b][f]), f) for f in ev["occ"]]
            peak_iou, pf = max(ious, key=lambda t: t[0])
            xa, ya, wa, ha, _ = gt[a][pf]
            xb, yb, wb, hb, _ = gt[b][pf]
            foot_a, foot_b = ya + ha, yb + hb
            h_ref = 0.5 * (ha + hb)
            foot_gap_h = abs(foot_a - foot_b) / max(h_ref, 1e-6)
            foot_front = a if foot_a > foot_b else b
            rows.append(
                dict(
                    seq=seq,
                    frame=pf,
                    id_a=a,
                    id_b=b,
                    gt_front=gt_front,
                    peak_iou=round(peak_iou, 4),
                    foot_gap_h=round(foot_gap_h, 4),
                    vis_gap=round(vmax - vmin, 3),
                    occ_len=ev["occ_len"],
                    ambiguous=int(ambiguous),
                    foot_front_correct=int(foot_front == gt_front),
                )
            )
    return rows


def interval_report(
    rows: list[dict], iou_lo: float, default_iou: float, default_foot: float
) -> None:
    """Read the optimal band as an interval intersection over events."""
    usable = [r for r in rows if not r["ambiguous"]]
    arr_iou = np.array([r["peak_iou"] for r in usable])
    arr_gap = np.array([r["foot_gap_h"] for r in usable])
    ok = np.array([r["foot_front_correct"] for r in usable])

    print(f"\nusable events (non-ambiguous): {len(usable)} / {len(rows)}")
    print(
        "\n── interval read-off: how each candidate threshold is DERIVED from stored values ──"
    )
    print(
        "  occ_iou_thresh τ  →  events gated = #{peak_iou ≥ τ}  (a pure query on the stored column)"
    )
    for tau in (0.40, 0.45, 0.50):
        g = arr_iou >= tau
        print(
            f"    τ={tau:.2f}: gated {g.sum():3d}  front-correct {100 * ok[g].mean():.0f}%"
            if g.sum()
            else f"    τ={tau:.2f}: gated 0"
        )
    print("\n  occ_foot_gap φ  →  same-depth events = #{foot_gap ≤ φ}")
    for phi in (0.10, 0.15, 0.20):
        g = arr_gap <= phi
        print(
            f"    φ={phi:.2f}: {g.sum():3d} events  front-correct {100 * ok[g].mean():.0f}%"
        )

    # The "intersection": the band that keeps the correct events and drops the wrong ones.
    correct_gap = arr_gap[ok == 1]
    wrong_gap = arr_gap[ok == 0]
    print("\n── interval intersection (the safe operating band) ──")
    print(
        f"  foot_gap of CORRECT front-calls : p50 {np.median(correct_gap):.3f}  "
        f"p90 {np.quantile(correct_gap, 0.9):.3f}  max {correct_gap.max():.3f}"
    )
    if len(wrong_gap):
        print(
            f"  foot_gap of WRONG   front-calls : p50 {np.median(wrong_gap):.3f}  "
            f"min {wrong_gap.min():.3f}  (these are what a loose φ wrongly admits)"
        )
    g = (arr_iou >= default_iou) & (arr_gap <= default_foot)
    print(
        f"  current default (iou≥{default_iou} foot≤{default_foot}): "
        f"gated {g.sum()}  front-correct {100 * ok[g].mean():.0f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--iou-extract", type=float, default=0.30)
    ap.add_argument("--max-occ", type=int, default=4)
    ap.add_argument("--min-life", type=int, default=5)
    ap.add_argument("--pre-win", type=int, default=6)
    ap.add_argument("--vis-margin", type=float, default=0.10)
    ap.add_argument("--default-iou", type=float, default=0.45)
    ap.add_argument("--default-foot", type=float, default=0.15)
    ap.add_argument(
        "--csv", type=Path, default=Path("results/occ_tune/event_values.csv")
    )
    args = ap.parse_args()

    rows = collect(
        args.root,
        args.det,
        args.iou_extract,
        args.max_occ,
        args.min_life,
        args.pre_win,
        args.vis_margin,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} per-event rows (actual values) → {args.csv}")
    interval_report(rows, args.iou_extract, args.default_iou, args.default_foot)


if __name__ == "__main__":
    main()
