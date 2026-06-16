#!/usr/bin/env python3
"""Horizon-convergence probe — does the pedestrian bbox scale field actually
converge to a stable horizon line? (Experiment 1 of horizon_depth_prior_design.md)

Why this first
--------------
The horizon/depth prior (and the horizon-as-GMC idea behind it) is only worth
building if the bbox scale field *robustly fits a single horizon line*. Per the
design's own rule #10: "收不到線就完全不用". So before wiring anything into the
tracker we answer one cheap question, per sequence:

    Given clean GT tracklets, does `h = a*foot_y + b` (and the 2-D scale field
    `h = a*cx + b*cy + c`) converge — high inlier ratio, low normalized residual,
    bootstrap-stable horizon_y / angle — and pass the self-validation gates of §6?

This is an UPPER BOUND: we use perfect GT tracklet medians (no detector jitter,
no ID switches). If a sequence does not converge here, it cannot converge on
noisy detections. A low enabled-ratio across sequences kills the whole direction
for ~zero cost.

GMC-signal note
---------------
For the horizon-as-GMC angle we also need the horizon to *move* frame-to-frame
(roll → angle change). This probe only measures static per-sequence convergence;
a follow-up windowed pass would measure angle drift. Convergence is the prereq.

Method (GT-only, tracker-independent, reproducible)
---------------------------------------------------
  * per id: median over its lifetime of h, foot_y=y2, cx, cy  (design §1.1)
  * quality gates (design §5.1): min lifetime, height stability, not edge-
    truncated, sane aspect ratio, min height
  * Model A: robust fit h = a*foot_y + b  -> horizon_y = -b/a
  * Model B: lstsq fit  h = a*cx + b*cy + c -> horizon line + tilt angle
  * inlier_ratio via normalized residual |h - pred|/h < TAU
  * bootstrap stability: resample tracklets K times, std(horizon_y)/H, std(angle)
  * enabled = passes §6 self-validation gates

Usage
-----
  .venv/bin/python scripts/tools/horizon_convergence_probe.py
  .venv/bin/python scripts/tools/horizon_convergence_probe.py --tau 0.20 --boot 200
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]
CAM = {
    "MOT17-02": "static elevated",
    "MOT17-04": "static high",
    "MOT17-05": "moving low",
    "MOT17-09": "static low",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}
# imHeight per sequence (from seqinfo.ini) for H-normalized reporting
IMG_H = {
    "MOT17-02": 1080,
    "MOT17-04": 1080,
    "MOT17-05": 480,
    "MOT17-09": 1080,
    "MOT17-10": 1080,
    "MOT17-11": 1080,
    "MOT17-13": 1080,
}
IMG_W = {
    "MOT17-02": 1920,
    "MOT17-04": 1920,
    "MOT17-05": 640,
    "MOT17-09": 1920,
    "MOT17-10": 1920,
    "MOT17-11": 1920,
    "MOT17-13": 1920,
}


def load_gt(
    path: Path,
) -> dict[int, dict[int, tuple[float, float, float, float, float]]]:
    """gt_id -> {frame: (x, y, w, h, vis)} for class==1, conf==1 rows."""
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 9:
            continue
        frame, gid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf, cls, vis = int(p[6]), int(p[7]), float(p[8])
        if conf != 1 or cls != 1:
            continue
        by_id[gid][frame] = (x, y, w, h, vis)
    return dict(by_id)


def tracklet_features(gt, seq, args):
    """Return arrays (foot_y, h, cx, cy) of per-id median geometry after gates."""
    H, _W = IMG_H[seq], IMG_W[seq]
    feats = []
    n_total = 0
    rej = defaultdict(int)
    for gid, frames in gt.items():
        n_total += 1
        if len(frames) < args.min_life:
            rej["short"] += 1
            continue
        rows = np.array([list(frames[f]) for f in sorted(frames)])  # (n,5)
        x, y, w, h, vis = rows.T
        # height stability (relative std) — exclude jittery / heavily occluded
        h_med = float(np.median(h))
        if h_med <= 0 or float(np.std(h)) / h_med > args.h_var:
            rej["unstable_h"] += 1
            continue
        # visibility — want mostly-visible tracklets for clean scale
        if float(np.median(vis)) < args.min_vis:
            rej["occluded"] += 1
            continue
        # min / max height in pixels (far-tiny noisy, near-huge truncated)
        if h_med < args.min_h or h_med > args.max_h_frac * H:
            rej["height"] += 1
            continue
        # aspect ratio sane
        ar = float(np.median(w)) / h_med
        if not (args.ar_lo <= ar <= args.ar_hi):
            rej["aspect"] += 1
            continue
        cx = float(np.median(x + w / 2))
        cy = float(np.median(y + h / 2))
        foot_y = float(np.median(y + h))
        top_y = float(np.median(y))
        # edge truncation gates
        if top_y < args.edge:
            rej["top_edge"] += 1
            continue
        if foot_y > H - args.edge:
            rej["bottom_edge"] += 1
            continue
        feats.append((foot_y, h_med, cx, cy))
    return np.array(feats) if feats else np.empty((0, 4)), n_total, dict(rej)


def fit_1d_robust(foot_y, h, tau, iters=3):
    """Robust fit h = a*foot_y + b with IRLS-style inlier refinement.

    Returns (a, b, inlier_mask). Inlier = normalized residual |h-pred|/h < tau.
    """
    mask = np.ones(len(h), dtype=bool)
    a = b = 0.0
    for _ in range(iters):
        if mask.sum() < 2:
            break
        A = np.vstack([foot_y[mask], np.ones(mask.sum())]).T
        sol, *_ = np.linalg.lstsq(A, h[mask], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        pred = a * foot_y + b
        norm_res = np.abs(h - pred) / np.maximum(h, 1e-6)
        mask = norm_res < tau
    return a, b, mask


def fit_plane(cx, cy, h):
    """lstsq fit h = a*cx + b*cy + c. Returns (a, b, c)."""
    A = np.vstack([cx, cy, np.ones(len(h))]).T
    sol, *_ = np.linalg.lstsq(A, h, rcond=None)
    return float(sol[0]), float(sol[1]), float(sol[2])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--min-life", type=int, default=15, help="min GT lifetime (frames)")
    ap.add_argument("--h-var", type=float, default=0.30, help="max rel std of height")
    ap.add_argument("--min-vis", type=float, default=0.6, help="min median visibility")
    ap.add_argument("--min-h", type=float, default=20.0, help="min median height (px)")
    ap.add_argument("--max-h-frac", type=float, default=0.9, help="max height / imH")
    ap.add_argument("--ar-lo", type=float, default=0.2, help="min w/h aspect")
    ap.add_argument("--ar-hi", type=float, default=0.8, help="max w/h aspect")
    ap.add_argument("--edge", type=float, default=5.0, help="edge-truncation margin px")
    ap.add_argument(
        "--tau", type=float, default=0.20, help="normalized residual inlier thr"
    )
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    # §6 self-validation gates
    ap.add_argument("--gate-inlier", type=float, default=0.50)
    ap.add_argument("--gate-resid", type=float, default=0.20)
    ap.add_argument(
        "--gate-booty", type=float, default=0.08, help="max std(horizon_y)/H"
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(
        f"\nHorizon convergence (GT upper bound)  det={args.det}  tau={args.tau}  boot={args.boot}"
    )
    print(
        f"gates: inlier>{args.gate_inlier} resid<{args.gate_resid} "
        f"std(hy)/H<{args.gate_booty}\n"
    )
    hdr = (
        f"{'sequence':<16}{'camera':<16}{'usable/all':>11}  "
        f"{'inlier%':>8}{'nresid':>8}{'slope_a':>9}  "
        f"{'hy/H':>7}{'boot_hy/H':>10}{'tilt°':>7}{'boot°':>7}  {'ENABLED':>8}"
    )
    print(hdr)
    print("─" * len(hdr))

    summary = []
    for seq in SEQS:
        gt_path = args.root / f"{seq}-{args.det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  ! missing {gt_path}")
            continue
        gt = load_gt(gt_path)
        feats, n_total, rej = tracklet_features(gt, seq, args)
        H = IMG_H[seq]
        if len(feats) < 5:
            print(
                f"{seq:<16}{CAM[seq]:<16}{len(feats):>4}/{n_total:<6}  too few tracklets {rej}"
            )
            summary.append((seq, False))
            continue

        foot_y, h, cx, cy = feats.T

        # Model A: h = a*foot_y + b
        a, b, mask = fit_1d_robust(foot_y, h, args.tau)
        inlier_ratio = float(mask.mean())
        pred = a * foot_y + b
        nresid = float(np.median(np.abs(h - pred)[mask] / np.maximum(h[mask], 1e-6)))
        horizon_y = -b / a if abs(a) > 1e-9 else float("nan")

        # Model B: tilt angle from 2-D scale field h = a2*cx + b2*cy + c2
        a2, b2, c2 = fit_plane(cx, cy, h)
        # horizon line a2*x + b2*y + c2 = 0 -> slope -a2/b2 -> tilt from horizontal
        tilt_deg = (
            float(np.degrees(np.arctan2(-a2, b2))) if abs(b2) > 1e-9 else float("nan")
        )

        # Bootstrap stability
        hy_samples, tilt_samples = [], []
        n = len(feats)
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            fa, fb, _ = fit_1d_robust(foot_y[idx], h[idx], args.tau)
            if abs(fa) > 1e-9:
                hy_samples.append(-fb / fa)
            pa, pb, _ = fit_plane(cx[idx], cy[idx], h[idx])
            if abs(pb) > 1e-9:
                tilt_samples.append(np.degrees(np.arctan2(-pa, pb)))
        boot_hy = float(np.std(hy_samples)) / H if hy_samples else float("nan")
        boot_tilt = float(np.std(tilt_samples)) if tilt_samples else float("nan")

        # §6 self-validation gates
        slope_ok = a > 0  # lower in image => larger box (§6.5)
        pos_ok = -0.5 * H < horizon_y < 0.8 * H  # §6.4
        enabled = (
            inlier_ratio >= args.gate_inlier
            and nresid <= args.gate_resid
            and boot_hy <= args.gate_booty
            and slope_ok
            and pos_ok
        )
        summary.append((seq, enabled))

        flag = "✓ YES" if enabled else "✗ no"
        why = ""
        if not enabled:
            reasons = []
            if inlier_ratio < args.gate_inlier:
                reasons.append("inlier")
            if nresid > args.gate_resid:
                reasons.append("resid")
            if boot_hy > args.gate_booty:
                reasons.append("boot")
            if not slope_ok:
                reasons.append("slope")
            if not pos_ok:
                reasons.append("pos")
            why = " [" + ",".join(reasons) + "]"
        print(
            f"{seq:<16}{CAM[seq]:<16}{len(feats):>4}/{n_total:<6}  "
            f"{100 * inlier_ratio:>7.1f} {nresid:>8.3f}{a:>9.3f}  "
            f"{horizon_y / H:>7.2f}{boot_hy:>10.3f}{tilt_deg:>7.2f}{boot_tilt:>7.2f}  {flag:>8}{why}"
        )

    print("─" * len(hdr))
    n_en = sum(1 for _, e in summary if e)
    print(
        f"\nENABLED sequences: {n_en}/{len(summary)}  "
        f"({', '.join(s for s, e in summary if e) or 'none'})\n"
    )
    print(
        "Reading: inlier%=support for single horizon, nresid=median |Δh|/h on inliers,\n"
        "slope_a>0 sanity (lower box=bigger), hy/H=horizon position, boot_*=bootstrap std.\n"
        "If ENABLED count is low even on clean GT, the scale field does not converge → NO-GO."
    )


if __name__ == "__main__":
    main()
