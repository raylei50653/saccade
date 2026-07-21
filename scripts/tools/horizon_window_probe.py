#!/usr/bin/env python3
"""Frame-to-frame horizon motion — the GMC signal test.

The aggregate probes showed the vertex-homothety horizon converges (4/7 seqs,
survives real detections). But horizon-as-GMC needs the horizon to *move* with
the camera (pitch/bounce -> horizon_y shift; roll -> tilt change). This probe
estimates the horizon per sliding window and asks, per sequence:

    Does horizon_y / tilt change across windows by MORE than the within-window
    estimation noise (bootstrap)?  signal_ratio = across_window_std / within_noise

  * signal_ratio >> 1  -> the horizon genuinely tracks camera motion = GMC signal.
  * signal_ratio ~ 1   -> variation is just estimation noise = static, NO GMC info
                          (only a static roll/depth prior survives).

Built-in control: 02/09 are static cameras (expect flat), 05/13 are moving
(expect motion IF horizon carries it). GT boxes (amodal) are used so "does it
move" is not confounded with detector noise.

Per-window min-people / scale-diversity gate (the requirement flagged earlier):
a window is only estimated if it has >= --min-track gated tracklets AND
>= --min-pairs valid (>=30% height-diff) pairs; else the window is skipped.

Usage
-----
  .venv/bin/python scripts/tools/horizon_window_probe.py --win 30 --step 15
"""
# status: experiment

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
from horizon_homothety_probe import (  # noqa: E402
    CAM,
    IMG_H,
    IMG_W,
    SEQS,
    fit_line_tls,
    horizon_center_y_angle,
    load_gt,
    lsq_intersection,
)


def window_boxes(gt, f0, f1, args, H):
    """Per-id median box over frames [f0,f1) after gates. Returns (N,4)."""
    boxes = []
    for gid, frames in gt.items():
        fs = [f for f in frames if f0 <= f < f1]
        if len(fs) < args.min_obs:
            continue
        rows = np.array([list(frames[f]) for f in fs])
        x, y, w, h, vis = rows.T
        h_med = float(np.median(h))
        if h_med <= 0 or float(np.std(h)) / h_med > args.h_var:
            continue
        if float(np.median(vis)) < args.min_vis:
            continue
        if h_med < args.min_h or h_med > args.max_h_frac * H:
            continue
        ar = float(np.median(w)) / h_med
        if not (args.ar_lo <= ar <= args.ar_hi):
            continue
        x1, y1 = float(np.median(x)), float(np.median(y))
        x2, y2 = float(np.median(x + w)), float(np.median(y + h))
        if y1 < args.edge or y2 > H - args.edge:
            continue
        boxes.append((x1, y1, x2, y2))
    return np.array(boxes) if boxes else np.empty((0, 4))


def horizon_of(boxes, H, W, args, rng):
    """Return (cy/H, tilt, lam, nV, boot_cy/H) or None if under-supported."""
    if len(boxes) < args.min_track:
        return None
    heights = boxes[:, 3] - boxes[:, 1]
    Vs = []
    for i, j in combinations(range(len(boxes)), 2):
        hi, hj = heights[i], heights[j]
        if max(hi, hj) / max(min(hi, hj), 1e-6) < args.ratio:
            continue
        vx, vy, _ = lsq_intersection(boxes[i], boxes[j])
        if not np.isfinite(vx) or abs(vx) > args.vfar * W or abs(vy) > args.vfar * H:
            continue
        Vs.append((vx, vy))
    if len(Vs) < args.min_pairs:
        return None
    Vs = np.array(Vs)
    m = Vs.mean(0)
    _, s, _ = np.linalg.svd(Vs - m)
    lam = s[0] ** 2 / max(s[1] ** 2, 1e-9)
    thr = args.inlier_frac * H
    mm, dd, _, _ = fit_line_tls(Vs, thr)
    cy, tilt = horizon_center_y_angle(mm, dd, W)
    # within-window bootstrap noise
    cy_s, tilt_s = [], []
    n = len(Vs)
    for _ in range(args.boot):
        idx = rng.integers(0, n, n)
        a, b, _, _ = fit_line_tls(Vs[idx], thr)
        c, t = horizon_center_y_angle(a, b, W)
        if np.isfinite(c):
            cy_s.append(c / H)
        tilt_s.append(t)
    return dict(
        cy=cy / H,
        tilt=tilt,
        lam=lam,
        nV=len(Vs),
        boot_cy=float(np.std(cy_s)) if cy_s else np.nan,
        boot_tilt=float(np.std(tilt_s)) if tilt_s else np.nan,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--win", type=int, default=30, help="window length (frames)")
    ap.add_argument("--step", type=int, default=15, help="window step (frames)")
    ap.add_argument(
        "--min-obs", type=int, default=5, help="min frames per id in window"
    )
    ap.add_argument(
        "--min-track", type=int, default=6, help="min gated tracklets per window"
    )
    ap.add_argument(
        "--min-pairs", type=int, default=15, help="min valid pairs per window"
    )
    ap.add_argument("--h-var", type=float, default=0.30)
    ap.add_argument("--min-vis", type=float, default=0.6)
    ap.add_argument("--min-h", type=float, default=20.0)
    ap.add_argument("--max-h-frac", type=float, default=0.9)
    ap.add_argument("--ar-lo", type=float, default=0.2)
    ap.add_argument("--ar-hi", type=float, default=0.8)
    ap.add_argument("--edge", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=1.30)
    ap.add_argument("--vfar", type=float, default=3.0)
    ap.add_argument("--inlier-frac", type=float, default=0.03)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"\nFrame-to-frame horizon motion (GT)  win={args.win} step={args.step}\n")
    hdr = (
        f"{'sequence':<16}{'camera':<16}{'wins':>5}  "
        f"{'Vy/H range':>13}{'across_std':>11}{'noise':>8}{'sig':>6}   "
        f"{'tilt range':>13}{'a_std':>7}{'noise':>7}{'sig':>6}"
    )
    print(hdr)
    print("─" * len(hdr))

    for seq in SEQS:
        gt_path = args.root / f"{seq}-{args.det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        gt = load_gt(gt_path)
        H, W = IMG_H[seq], IMG_W[seq]
        frames = [f for fr in gt.values() for f in fr]
        fmin, fmax = min(frames), max(frames)

        cys, tilts, bcy, btilt = [], [], [], []
        for f0 in range(fmin, fmax - args.win + 2, args.step):
            res = horizon_of(
                window_boxes(gt, f0, f0 + args.win, args, H), H, W, args, rng
            )
            if res is None:
                continue
            cys.append(res["cy"])
            tilts.append(res["tilt"])
            bcy.append(res["boot_cy"])
            btilt.append(res["boot_tilt"])
        if len(cys) < 3:
            print(f"{seq:<16}{CAM[seq]:<16}{len(cys):>5}  too few valid windows")
            continue
        cys, tilts = np.array(cys), np.array(tilts)
        cy_std = float(np.std(cys))
        cy_noise = float(np.median(bcy))
        t_std = float(np.std(tilts))
        t_noise = float(np.median(btilt))
        cy_sig = cy_std / max(cy_noise, 1e-6)
        t_sig = t_std / max(t_noise, 1e-6)
        print(
            f"{seq:<16}{CAM[seq]:<16}{len(cys):>5}  "
            f"[{cys.min():.2f},{cys.max():.2f}]{cy_std:>11.3f}{cy_noise:>8.3f}{cy_sig:>6.1f}   "
            f"[{tilts.min():>5.1f},{tilts.max():>5.1f}]{t_std:>7.2f}{t_noise:>7.2f}{t_sig:>6.1f}"
        )

    print("─" * len(hdr))
    print(
        "\nsig = across-window std / within-window bootstrap noise.\n"
        "  sig >> 1  -> horizon genuinely moves frame-to-frame = GMC signal exists.\n"
        "  sig ~ 1   -> variation is estimation noise = static, no frame-to-frame GMC info.\n"
        "Control: 02/09 static (expect sig~1), 05/13 moving (sig>>1 means usable GMC)."
    )


if __name__ == "__main__":
    main()
