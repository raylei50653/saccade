#!/usr/bin/env python3
"""Control for the pre-loss turning signal: is mid-track turning just as high?

Slides an 8-consecutive-frame window across every track and classifies each
window as PRE-LOSS (window ends at the track's final frame, track lost mid-seq)
or INTERIOR ("normal cruising": window ends >=`--gap-from-loss` frames before
loss).  Reports GMC-compensated turning (net + max per-step) AND mean speed for
both, so we can tell a real pre-loss maneuver from velocity-shrink angle noise.

  .venv/bin/python scripts/tools/analyze_turn_baseline.py --window 8
"""
# status: stable

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_preloss_motion import (  # noqa: E402
    DEFAULT_SEQS,
    load_boxes,
    estimate_gmc_chain,
    world_point,
)


def _ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))


def window_stats(C, win, min_speed):
    """Robust turning gated by speed.

    Returns (rnet_deg, max_fast_deg, mean_speed_h, n_fast) where
      rnet      = angle between first-half and second-half MEAN velocity
                  (each half-velocity must clear min_speed), else nan
      max_fast  = max per-step turn over steps whose BOTH velocities clear
                  min_speed, else nan
    min_speed is in heights/frame.
    """
    pts, hs = [], []
    for f, x, y, w, h in win:
        pts.append(world_point(C, f, x + w / 2.0, y + h))  # foot
        hs.append(h)
    pts = np.array(pts)
    href = max(np.mean(hs), 1.0)
    vels = np.diff(pts, axis=0) / href  # heights/frame, per step
    speeds = np.linalg.norm(vels, axis=1)

    # robust net: mean velocity of first vs second half
    k = len(vels) // 2
    vf, vs = vels[:k].mean(axis=0), vels[-k:].mean(axis=0)
    rnet = (
        _ang(vf, vs)
        if (np.linalg.norm(vf) >= min_speed and np.linalg.norm(vs) >= min_speed)
        else float("nan")
    )

    # per-step turns over fast steps only
    fast_angs = []
    for i in range(1, len(vels)):
        if speeds[i - 1] >= min_speed and speeds[i] >= min_speed:
            a = _ang(vels[i - 1], vels[i])
            if a is not None:
                fast_angs.append(a)
    max_fast = max(fast_angs) if fast_angs else float("nan")
    return rnet, max_fast, float(np.mean(speeds)), len(fast_angs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--end-margin", type=int, default=2)
    ap.add_argument(
        "--gap-from-loss",
        type=int,
        default=12,
        help="interior window must end this many frames before loss",
    )
    ap.add_argument(
        "--min-speed",
        type=float,
        default=0.03,
        help="min velocity (heights/frame) to count a vector as moving",
    )
    ap.add_argument("--out", type=Path, default=Path("scripts/tools/out/turn_baseline"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    W = args.window

    pre = {"net": [], "max": [], "spd": []}
    intr = {"net": [], "max": [], "spd": []}
    n_pre_win = n_intr_win = 0
    for seq in args.seqs:
        mot = args.mot_dir / f"{seq}.txt"
        seqdir = args.img_root / seq
        if not mot.exists():
            continue
        ini = configparser.ConfigParser()
        ini.read(seqdir / "seqinfo.ini")
        n_frames = ini.getint("Sequence", "seqLength")
        print(f"{seq}: GMC over {n_frames} frames ...", flush=True)
        C = estimate_gmc_chain(seqdir / "img1", n_frames, args.downscale)
        tracks = load_boxes(mot)
        for tid, tr in tracks.items():
            by_f = {r[0]: r for r in tr}
            last_f = tr[-1][0]
            lost = last_f <= n_frames - args.end_margin
            # slide consecutive windows
            for end in range(tr[0][0] + W - 1, last_f + 1):
                want = list(range(end - W + 1, end + 1))
                if not all(f in by_f and f in C for f in want):
                    continue
                net, mx, spd, _ = window_stats(
                    C, [by_f[f] for f in want], args.min_speed
                )
                if lost and end == last_f:
                    bucket = pre
                elif end <= last_f - args.gap_from_loss:
                    bucket = intr
                else:
                    continue
                if bucket is pre:
                    n_pre_win += 1
                else:
                    n_intr_win += 1
                if not np.isnan(net):
                    bucket["net"].append(net)  # robust half-window net turn
                if not np.isnan(mx):
                    bucket["max"].append(mx)  # max fast per-step turn
                bucket["spd"].append(spd)

    def line(tag, d):
        for k in ("net", "max", "spd"):
            a = np.array(d[k])
            q = np.percentile(a, [25, 50, 75, 90]) if len(a) else [np.nan] * 4
            unit = "deg" if k != "spd" else "h/f"
            print(
                f"  {tag:<9} {k:<4} n={len(a):>5} "
                f"p25={q[0]:6.2f} p50={q[1]:6.2f} p75={q[2]:6.2f} p90={q[3]:6.2f} {unit}"
            )

    print(
        f"\n=== turning gated at speed>={args.min_speed} h/f "
        f"(GMC, {W} consec frames) ==="
    )
    print(f"  windows: pre-loss={n_pre_win}  interior={n_intr_win}")
    print(
        f"  robust-net survived speed gate: pre-loss {len(pre['net'])}/{n_pre_win}"
        f" ({len(pre['net']) / max(n_pre_win, 1):.0%})"
        f"  interior {len(intr['net'])}/{n_intr_win}"
        f" ({len(intr['net']) / max(n_intr_win, 1):.0%})"
    )
    line("pre-loss", pre)
    line("interior", intr)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for col, k, ttl, rng in [
        (
            0,
            "net",
            f"Robust half-window turn, gated (deg) [spd>={args.min_speed}]",
            (0, 180),
        ),
        (1, "max", "Max fast per-step turn (deg)", (0, 180)),
        (2, "spd", "Mean speed (heights/frame)", (0, 0.2)),
    ]:
        ax[col].hist(
            np.clip(pre[k], *rng),
            bins=36,
            range=rng,
            density=True,
            alpha=0.55,
            color="#C44E52",
            label=f"pre-loss (n={len(pre[k])})",
        )
        ax[col].hist(
            np.clip(intr[k], *rng),
            bins=36,
            range=rng,
            density=True,
            alpha=0.55,
            color="#4C72B0",
            label=f"interior (n={len(intr[k])})",
        )
        ax[col].axvline(np.median(pre[k]), color="#C44E52", ls="--")
        ax[col].axvline(np.median(intr[k]), color="#4C72B0", ls="--")
        ax[col].set(title=ttl, ylabel="density")
        ax[col].legend()
    fig.suptitle(f"Pre-loss vs interior motion, GMC-compensated ({W} consec frames)")
    fig.tight_layout()
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"\nWrote chart -> {png}")


if __name__ == "__main__":
    main()
