#!/usr/bin/env python3
"""Sweep per-step speed vs turning: at what move-speed/box-height ratio does
heading stop being noise?

Pools every consecutive 3-frame foot triplet (GMC-compensated) across all
tracks, records (step_speed, turn_angle), bins by speed (= displacement per
frame in units of box height), and reports median/IQR turning per bin for
PRE-LOSS vs INTERIOR steps.  The knee where median turn drops well below the
uniform-noise level (~90 deg) marks the speed regime where direction is usable.

  .venv/bin/python scripts/tools/sweep_speed_turn.py
"""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--end-margin", type=int, default=2)
    ap.add_argument(
        "--preloss-w",
        type=int,
        default=8,
        help="a step is PRE-LOSS if it lies in the last W frames of a lost track",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("scripts/tools/out/speed_turn_sweep")
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # records: speed (min of the two step speeds), angle, is_preloss
    spd_all, ang_all, pre_all, h_all = [], [], [], []
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
        for tid, tr in load_boxes(mot).items():
            by_f = {r[0]: r for r in tr}
            last_f = tr[-1][0]
            lost = last_f <= n_frames - args.end_margin
            # walk consecutive triplets f-1,f,f+1
            for f, x, y, w, h in tr:
                if not (
                    f - 1 in by_f and f + 1 in by_f and {f - 1, f, f + 1} <= C.keys()
                ):
                    continue
                p0 = world_point(
                    C, f - 1, *(lambda r: (r[1] + r[3] / 2, r[2] + r[4]))(by_f[f - 1])
                )
                p1 = world_point(C, f, x + w / 2, y + h)
                p2 = world_point(
                    C, f + 1, *(lambda r: (r[1] + r[3] / 2, r[2] + r[4]))(by_f[f + 1])
                )
                href = max((by_f[f - 1][4] + h + by_f[f + 1][4]) / 3.0, 1.0)
                v0, v1 = (p1 - p0) / href, (p2 - p1) / href
                n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
                if n0 < 1e-9 or n1 < 1e-9:
                    continue
                ang = np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1, 1)))
                spd_all.append(min(n0, n1))
                ang_all.append(ang)
                pre_all.append(lost and f + 1 > last_f - args.preloss_w)
                h_all.append(h)

    spd = np.array(spd_all)
    ang = np.array(ang_all)
    pre = np.array(pre_all, dtype=bool)
    print(f"\ntotal steps: {len(spd)}  pre-loss: {pre.sum()}  interior: {(~pre).sum()}")

    edges = np.array(
        [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.18, 0.30]
    )
    print(
        f"\n{'speed bin (h/f)':>16} {'frames/H':>9} {'n_int':>7} {'med_int':>8} "
        f"{'IQR_int':>14} {'n_pre':>6} {'med_pre':>8}"
    )
    centers, med_int, med_pre, lo_int, hi_int, n_int = [], [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (spd >= a) & (spd < b)
        mi, mp = m & ~pre, m & pre
        c = (a + b) / 2
        centers.append(c)
        ai, ap = ang[mi], ang[mp]
        n_int.append(len(ai))
        if len(ai):
            q = np.percentile(ai, [25, 50, 75])
            med_int.append(q[1])
            lo_int.append(q[0])
            hi_int.append(q[2])
        else:
            med_int.append(np.nan)
            lo_int.append(np.nan)
            hi_int.append(np.nan)
        med_pre.append(np.median(ap) if len(ap) else np.nan)
        fph = 1.0 / c if c > 0 else float("inf")
        print(
            f"  [{a:.3f},{b:.3f}) {fph:>9.1f} {len(ai):>7} "
            f"{med_int[-1]:>8.1f} [{lo_int[-1]:>5.1f},{hi_int[-1]:>5.1f}] "
            f"{len(ap):>6} {med_pre[-1]:>8.1f}"
        )

    centers = np.array(centers)
    med_int = np.array(med_int)
    n_int = np.array(n_int)

    # ── Fit "straight walk + Gaussian foot-jitter" model (projected normal) ──
    # p_t = true(constant-velocity, straight) + N(0, sigma^2 I); sigma in h/f units.
    # Single free parameter sigma -> predicted median per-step turn vs OBSERVED speed.
    rng = np.random.default_rng(0)

    def mc_curve(sigma, n=300000):
        s_true = np.exp(rng.uniform(np.log(1e-3), np.log(0.3), n))
        th = rng.uniform(0, 2 * np.pi, n)
        u = np.stack([np.cos(th), np.sin(th)], 1)
        p1 = s_true[:, None] * u
        e0 = rng.normal(0, sigma, (n, 2))
        e1 = rng.normal(0, sigma, (n, 2))
        e2 = rng.normal(0, sigma, (n, 2))
        v0 = p1 + e1 - e0  # shares the middle point -> negatively correlated
        v1 = 2 * p1 + e2 - (p1 + e1)
        m0 = np.linalg.norm(v0, axis=1)
        m1 = np.linalg.norm(v1, axis=1)
        obs = np.minimum(m0, m1)
        cosv = np.clip((v0 * v1).sum(1) / (m0 * m1), -1, 1)
        aa = np.degrees(np.arccos(cosv))
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (obs >= lo) & (obs < hi)
            out.append(np.median(aa[mm]) if mm.sum() > 50 else np.nan)
        return np.array(out)

    best = None
    for sg in np.linspace(0.002, 0.030, 29):
        pred = mc_curve(sg)
        mk = ~np.isnan(pred) & ~np.isnan(med_int) & (n_int > 100)
        err = np.sum(n_int[mk] * (pred[mk] - med_int[mk]) ** 2) / np.sum(n_int[mk])
        if best is None or err < best[0]:
            best = (err, sg, pred)
    rmse, sigma_fit, pred_curve = best
    med_h = float(np.median(h_all))
    print(
        "\n=== Fit: straight-walk + Gaussian foot-jitter (projected-normal angle) ==="
    )
    print(
        f"  best sigma = {sigma_fit:.4f} h/f  = {sigma_fit * med_h:.2f} px "
        f"(median box height {med_h:.0f} px)"
    )
    print(f"  weighted RMSE over speed bins: {np.sqrt(rmse):.2f} deg")
    print(
        "  => turning is governed by SNR = speed/sigma; no intrinsic-turn component needed."
    )

    np.savez(
        args.out.with_suffix(".npz"),
        speed=spd,
        angle=ang,
        preloss=pre,
        height=np.array(h_all),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(90, color="grey", ls=":", label="uniform-noise level (90°)")
    ax.fill_between(
        centers, lo_int, hi_int, color="#4C72B0", alpha=0.2, label="interior IQR"
    )
    ax.plot(centers, med_int, "-o", color="#4C72B0", label="interior median turn")
    ax.plot(centers, med_pre, "-s", color="#C44E52", label="pre-loss median turn")
    ax.plot(
        centers,
        pred_curve,
        "--",
        color="k",
        lw=2,
        label=f"straight-walk + jitter σ={sigma_fit:.3f} h/f ({sigma_fit * med_h:.1f}px)",
    )
    ax.set(
        xlabel="per-step speed = move/box-height (heights/frame)",
        ylabel="per-step turn angle (deg)",
        title="Turning vs speed (GMC-compensated): heading becomes coherent as speed rises",
    )
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"\nWrote chart -> {png}")


if __name__ == "__main__":
    main()
