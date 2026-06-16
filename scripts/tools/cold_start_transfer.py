#!/usr/bin/env python3
# mypy: ignore-errors
"""Cold-start transfer test: do the normalized occ-gate landmarks hold on MOT20?

The same-height occlusion gate's defaults were cold-started from MOT17 distributions:
  occ_iou_thresh 0.45  (≈ peak-IoU of real crossing swaps)
  occ_foot_gap   0.15h (≈ above the foot-gap p75 of same-height collisions)
Both are DIMENSIONLESS (IoU in [0,1]; gap in units of object height h). The cold-start recipe
claims dimensionless landmarks transfer across datasets. MOT20 (≈4× denser crowds, different
camera heights) is the stress test. This is GT-only — zero model, zero end-to-end.

For each dataset it reports the front/back signal accuracy (foot_y vs GT visibility) and the
peak-IoU / |foot_gap|/h distribution of the crossing-swap population, so the MOT17 landmarks can
be checked directly against MOT20.

Usage
-----
  .venv/bin/python scripts/tools/cold_start_transfer.py --dataset mot17
  .venv/bin/python scripts/tools/cold_start_transfer.py --dataset mot20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import depth_ordering_probe as P

DATASETS = {
    "mot17": dict(
        seqs=[f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")],
        gt=lambda s: Path(f"datasets/MOT17/train/{s}/gt/gt.txt"),
    ),
    "mot20": dict(
        seqs=["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"],
        gt=lambda s: Path(f"datasets/MOT20/MOT20/train/{s}/gt/gt.txt"),
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", choices=list(DATASETS), default="mot20")
    ap.add_argument("--iou-hi", type=float, default=0.4)
    ap.add_argument("--max-occ", type=int, default=4)
    ap.add_argument("--min-life", type=int, default=5)
    ap.add_argument("--pre-win", type=int, default=6)
    ap.add_argument("--vis-margin", type=float, default=0.10)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    foot_ok, area_ok, peak_iou, gap_h = [], [], [], []
    per_seq = {}
    for seq in spec["seqs"]:
        gt_path = spec["gt"](seq)
        if not gt_path.exists():
            print(f"  ! missing {gt_path}")
            continue
        gt = P.load_gt(gt_path)
        evs = P.find_cross_events(
            gt, args.iou_hi, 0.3, args.max_occ, args.min_life, args.pre_win
        )
        sf = []
        for ev in evs:
            a, b = ev["a"], ev["b"]
            vmax, vmin = max(ev["vis_a"], ev["vis_b"]), min(ev["vis_a"], ev["vis_b"])
            if (vmax - vmin) < args.vis_margin or vmin >= 0.5:
                continue
            gt_front = a if ev["vis_a"] > ev["vis_b"] else b
            ious = [(P.iou(gt[a][f], gt[b][f]), f) for f in ev["occ"]]
            pk, pf = max(ious, key=lambda t: t[0])
            xa, ya, wa, ha, _ = gt[a][pf]
            xb, yb, wb, hb, _ = gt[b][pf]
            foot_a, foot_b = ya + ha, yb + hb
            ar_a, ar_b = wa * ha, wb * hb
            h_ref = 0.5 * (ha + hb)
            foot_front = a if foot_a > foot_b else b
            area_front = a if ar_a > ar_b else b
            foot_ok.append(int(foot_front == gt_front))
            sf.append(int(foot_front == gt_front))
            area_ok.append(int(area_front == gt_front))
            peak_iou.append(pk)
            gap_h.append(abs(foot_a - foot_b) / max(h_ref, 1e-6))
        if sf:
            per_seq[seq] = (len(sf), 100 * np.mean(sf))

    n = len(foot_ok)
    if n == 0:
        raise SystemExit("no crossing events found")
    fo, ao, pi, gh = (np.array(x) for x in (foot_ok, area_ok, peak_iou, gap_h))
    print(f"\n=== {args.dataset.upper()} cold-start landmarks (GT-only, n={n}) ===")
    print(
        f"  front/back signal: foot_y {100 * fo.mean():.1f}%   area {100 * ao.mean():.1f}%   (random 50%)"
    )
    print(
        f"  peak_iou   p25/p50/p75 = {np.percentile(pi, 25):.2f} / {np.percentile(pi, 50):.2f} / {np.percentile(pi, 75):.2f}"
    )
    print(
        f"  |foot_gap|/h p25/p50/p75 = {np.percentile(gh, 25):.3f} / {np.percentile(gh, 50):.3f} / {np.percentile(gh, 75):.3f}"
    )
    print(
        f"  → frac with |gap|≤0.15h: {100 * (gh <= 0.15).mean():.0f}%   peak_iou≥0.45: {100 * (pi >= 0.45).mean():.0f}%"
    )
    print("  per-seq foot%:", {s: f"{v[1]:.0f}({v[0]})" for s, v in per_seq.items()})
    print(
        "\n  MOT17 reference: foot 89.8% | peak_iou p25/p50 0.45/0.45 (swaps) | gap p50≈0.05-0.15 | defaults iou0.45 foot0.15"
    )


if __name__ == "__main__":
    main()
