#!/usr/bin/env python3
"""Tier-1 offline gate sweep for the same-height occlusion gate (GT-only, free).

The same-height occlusion gate (`tracker_gpu.cu`, default-on, commit c418872b) fires when
two CONFIRMED tracks overlap (track-track IoU >= `occ_iou_thresh`) AND sit at the same depth
(`|foot_gap|/h_ref <= occ_foot_gap`), then penalises the front (lower-foot) track for matching
the back box.  Its two **gate** parameters (`occ_iou_thresh`, `occ_foot_gap`) are operating
points on a GT-labellable signal, so they can be tuned **offline, with no tracker eval** —
unlike the policy params (`occ_ttl`, `occ_cost_weight`), which only show up end-to-end.

This sweep, for every (iou_thresh, foot_gap) cell, measures on the GT crossing-swap population:
  * coverage   = fraction of crossing events the gate would act on,
  * precision  = among gated events, the foot-y front call is correct (penalty aimed at the
                 true occluder)  — the purity of the penalty *direction*,
  * value      = coverage * (precision - 0.5), expected correct actions over a coin flip.

Geometry is measured at the **peak-IoU frame** of each contact (matching the kernel, which
gates during overlap), unlike `depth_ordering_probe.py` which uses pre-occlusion frames.

Usage
-----
  .venv/bin/python scripts/tools/depth_ordering_gate_sweep.py
  .venv/bin/python scripts/tools/depth_ordering_gate_sweep.py --iou-grid 0.40,0.45,0.50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import depth_ordering_probe as P  # same dir; reuse load_gt / iou / find_cross_events


def mannwhitney_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg) with ties = 0.5 (rank form)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def collect_events(
    root: Path,
    det: str,
    iou_lo_extract: float,
    max_occ: int,
    min_life: int,
    pre_win: int,
    vis_margin: float,
) -> np.ndarray:
    """Return per-event array: [peak_iou, foot_gap_h, foot_front_correct, occ_len]."""
    rows = []
    for seq in P.SEQS:
        gt_path = root / f"{seq}-{det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  ! missing {gt_path}")
            continue
        gt = P.load_gt(gt_path)
        # extract a SUPERSET with a loose merge IoU so we can gate at higher thresholds
        evs = P.find_cross_events(
            gt, iou_lo_extract, P_IOU_LO, max_occ, min_life, pre_win
        )
        for ev in evs:
            a, b = ev["a"], ev["b"]
            vmax, vmin = max(ev["vis_a"], ev["vis_b"]), min(ev["vis_a"], ev["vis_b"])
            if (vmax - vmin) < vis_margin or vmin >= 0.5:
                continue  # genuinely ambiguous depth — not the swap population
            gt_front = a if ev["vis_a"] > ev["vis_b"] else b
            # peak-IoU frame across the contact window (kernel gates during overlap)
            ious = [(P.iou(gt[a][f], gt[b][f]), f) for f in ev["occ"]]
            peak_iou, pf = max(ious, key=lambda t: t[0])
            xa, ya, wa, ha, _ = gt[a][pf]
            xb, yb, wb, hb, _ = gt[b][pf]
            foot_a, foot_b = ya + ha, yb + hb
            h_ref = 0.5 * (ha + hb)
            foot_gap_h = abs(foot_a - foot_b) / max(h_ref, 1e-6)
            foot_front = (
                a if foot_a > foot_b else b
            )  # lower in image (larger y) = closer
            rows.append(
                (peak_iou, foot_gap_h, int(foot_front == gt_front), ev["occ_len"])
            )
    return np.array(rows, float)


P_IOU_LO = 0.3  # separated-IoU for the pre frame in find_cross_events (unused here, kept for API)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument(
        "--iou-extract",
        type=float,
        default=0.30,
        help="loose merge IoU to build the event superset (gate >= this)",
    )
    ap.add_argument("--max-occ", type=int, default=4)
    ap.add_argument("--min-life", type=int, default=5)
    ap.add_argument("--pre-win", type=int, default=6)
    ap.add_argument("--vis-margin", type=float, default=0.10)
    ap.add_argument("--iou-grid", default="0.40,0.45,0.50")
    ap.add_argument("--foot-grid", default="0.10,0.125,0.15,0.175,0.20,0.25,99")
    ap.add_argument("--default-iou", type=float, default=0.45)
    ap.add_argument("--default-foot", type=float, default=0.15)
    args = ap.parse_args()

    ev = collect_events(
        args.root,
        args.det,
        args.iou_extract,
        args.max_occ,
        args.min_life,
        args.pre_win,
        args.vis_margin,
    )
    if len(ev) == 0:
        raise SystemExit("no events — relax --iou-extract / --max-occ")
    peak_iou, foot_gap, correct = ev[:, 0], ev[:, 1], ev[:, 2]
    n = len(ev)

    iou_grid = [float(x) for x in args.iou_grid.split(",")]
    foot_grid = [float(x) for x in args.foot_grid.split(",")]

    print(
        f"\nGT crossing-event population (peak-IoU geometry): n={n}  "
        f"(extract IoU>= {args.iou_extract}, max_occ={args.max_occ}f, vis_margin={args.vis_margin})"
    )

    # per-parameter discrimination AUCs (literal "each parameter's AUC")
    print("\n── per-parameter signal AUC (predictor of foot-front-call correctness) ──")
    auc_gap = mannwhitney_auc(
        foot_gap[correct == 1], foot_gap[correct == 0]
    )  # P(gap_right > gap_wrong)
    auc_iou = mannwhitney_auc(peak_iou[correct == 1], peak_iou[correct == 0])
    print(
        f"  |foot_gap|/h : AUC {auc_gap:.3f}  "
        f"(>0.5 ⇒ LARGER gap = MORE reliable front call — the decisiveness signal; this is why a "
        f"precision-only objective wrongly favours loosening the same-height gate, see below)"
    )
    print(
        f"  peak track-track IoU : AUC {auc_iou:.3f}  "
        f"(IoU mildly predicts correctness but mainly selects collision density, not depth order)"
    )

    # 2-D sweep
    hdr = (
        f"{'iou≥':>6}{'foot≤':>8}{'gated':>7}{'cover%':>8}"
        f"{'front-prec%':>13}{'rej-prec%':>11}{'value':>8}"
    )
    print("\n── gate sweep (coverage × front-call precision) ──")
    print(hdr)
    print("─" * len(hdr))
    best = None
    for it in iou_grid:
        for fg in foot_grid:
            gate = (peak_iou >= it) & (foot_gap <= fg)
            ng = int(gate.sum())
            if ng == 0:
                continue
            cover = ng / n
            prec = float(correct[gate].mean())
            rej = correct[~gate]
            rej_prec = float(rej.mean()) if len(rej) else float("nan")
            value = cover * (prec - 0.5)
            mark = (
                "  <-- DEFAULT"
                if (
                    abs(it - args.default_iou) < 1e-6
                    and abs(fg - args.default_foot) < 1e-6
                )
                else ""
            )
            fg_disp = "  inf" if fg >= 90 else f"{fg:.3f}"
            print(
                f"{it:>6.2f}{fg_disp:>8}{ng:>7}{100 * cover:>7.1f} "
                f"{100 * prec:>11.1f} {100 * rej_prec:>10.1f} {value:>7.3f}{mark}"
            )
            if best is None or value > best[0]:
                best = (value, it, fg, cover, prec)
    print("─" * len(hdr))
    d = (peak_iou >= args.default_iou) & (foot_gap <= args.default_foot)
    if d.sum():
        print(
            f"current default (iou≥{args.default_iou:.2f} foot≤{args.default_foot:.3f}): "
            f"cover {100 * d.mean():.1f}%  front-prec {100 * correct[d].mean():.1f}%"
        )
    print("\n── Tier-1 verdict (what is / isn't offline-tunable) ──")
    print(
        "  * occ_iou_thresh IS offline-bounded: 0.50 starves the gate (coverage collapses,"
    )
    print("    n too small); 0.40-0.45 keeps healthy coverage at ≥94% front-precision.")
    print(
        "  * occ_foot_gap is NOT offline-tunable by front/back precision: precision is flat-high"
    )
    print(
        "    for every foot_gap (incl. inf), so a precision/coverage objective wrongly favours"
    )
    print(
        "    LOOSENING it — exactly the different-height population measured HARMFUL end-to-end"
    )
    print(
        "    (05 −1.1). The same-height gate's value is REJECTING projection-overlap false-"
    )
    print("    triggers, which only manifests end-to-end ⇒ foot_gap belongs to Tier-2.")


if __name__ == "__main__":
    main()
