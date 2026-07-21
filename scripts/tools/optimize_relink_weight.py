#!/usr/bin/env python3
"""Offline optimisation of the speed-weighted relink gate score.

Gate score = blend( velocity-residual , spatial dist_h ) with a per-candidate
weight w(speed) that grows with how fast both endpoints were moving:

    score(cand) = -( w(s)·vel + (1-w(s))·dist_h )     # lower distance = more likely true
    w(s) = shape( clip(s / sat, 0, 1) )

Grids over {velocity residual, speed metric, saturation, weight shape, blend
normalisation}, ranks by AP (average precision) and AUC on the full pool and on
the spatially-plausible hard region (dist_h ≤ 1), and reports a
leave-one-sequence-out (LOSO) CV estimate for the winner vs baselines to guard
against overfitting (only 256 positives / 7 sequences).

  .venv/bin/python scripts/tools/optimize_relink_weight.py
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


# ── metrics ──────────────────────────────────────────────────────────────────
def auc(score, y):
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    o = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    ss = score[o]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def ap(score, y):
    """Average precision (area under PR), higher score => positive."""
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    o = np.argsort(-score, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / pos
    # integrate precision over recall increments
    drec = np.diff(np.concatenate([[0], rec]))
    return float(np.sum(prec * drec))


# ── weight shapes ────────────────────────────────────────────────────────────
def w_shape(t, shape):
    t = np.clip(t, 0, 1)
    if shape == "linear":
        return t
    if shape == "smooth":
        return t * t * (3 - 2 * t)
    if shape == "sqrt":
        return np.sqrt(t)
    raise ValueError(shape)


def robust_z(x):
    med = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75, 25])) or 1.0
    return (x - med) / iqr


def build_score(d, vel_name, speed_name, sat, shape, norm):
    vel = d[vel_name]
    dist = d["dist_h"]
    s = d[speed_name]
    w = w_shape(s / sat, shape)
    if norm == "z":
        vel, dist = robust_z(vel), robust_z(dist)
    blended = w * vel + (1 - w) * dist
    return -blended  # higher = more likely true


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/relink_candidates.csv")
    )
    ap_.add_argument(
        "--hard-dist",
        type=float,
        default=1.0,
        help="hard region = dist_h <= this (spatially plausible)",
    )
    args = ap_.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r["gt_valid"] == "1"]
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    d = {
        k: col(k)
        for k in (
            "dist_h",
            "bridge_dist",
            "fwd_resid",
            "bwd_resid",
            "lost_exit_speed",
            "cand_entry_speed",
            "gap",
        )
    }
    d["sym_fb"] = 0.5 * (d["fwd_resid"] + d["bwd_resid"])
    d["min_fb"] = np.minimum(d["fwd_resid"], d["bwd_resid"])
    d["smin"] = np.minimum(d["lost_exit_speed"], d["cand_entry_speed"])
    d["slost"] = d["lost_exit_speed"]
    d["sgeo"] = np.sqrt(d["lost_exit_speed"] * d["cand_entry_speed"])
    y = np.array([int(r["gt_match"]) for r in rows])
    seq = np.array([r["seq"] for r in rows])
    hard = d["dist_h"] <= args.hard_dist
    print(
        f"pairs={len(y)} pos={int(y.sum())} | hard(dist_h<={args.hard_dist}): "
        f"n={int(hard.sum())} pos={int(y[hard].sum())}"
    )

    VEL = ["bridge_dist", "fwd_resid", "sym_fb", "min_fb"]
    SPD = ["smin", "slost", "sgeo"]
    SAT = [0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
    SHAPE = ["linear", "smooth", "sqrt"]
    NORM = ["raw", "z"]

    results = []
    for vel, spd, sat, shape, norm in itertools.product(VEL, SPD, SAT, SHAPE, NORM):
        sc = build_score(d, vel, spd, sat, shape, norm)
        results.append(
            dict(
                vel=vel,
                spd=spd,
                sat=sat,
                shape=shape,
                norm=norm,
                auc_full=auc(sc, y),
                ap_full=ap(sc, y),
                auc_hard=auc(sc[hard], y[hard]),
                ap_hard=ap(sc[hard], y[hard]),
            )
        )

    # baselines
    base = {}
    for name, sc in {"bridge_dist": -d["bridge_dist"], "dist_h": -d["dist_h"]}.items():
        base[name] = dict(
            auc_full=auc(sc, y),
            ap_full=ap(sc, y),
            auc_hard=auc(sc[hard], y[hard]),
            ap_hard=ap(sc[hard], y[hard]),
        )

    results.sort(key=lambda r: r["ap_hard"], reverse=True)
    print("\n=== top 12 configs by AP_hard ===")
    print(
        f"  {'vel':<11}{'spd':<6}{'sat':>5}{'shape':>8}{'norm':>5}"
        f"{'AUC_f':>7}{'AP_f':>7}{'AUC_h':>7}{'AP_h':>7}"
    )
    for r in results[:12]:
        print(
            f"  {r['vel']:<11}{r['spd']:<6}{r['sat']:>5}{r['shape']:>8}{r['norm']:>5}"
            f"{r['auc_full']:>7.3f}{r['ap_full']:>7.3f}{r['auc_hard']:>7.3f}{r['ap_hard']:>7.3f}"
        )
    print("  --- baselines ---")
    for name, b in base.items():
        print(
            f"  {name:<35}{b['auc_full']:>7.3f}{b['ap_full']:>7.3f}"
            f"{b['auc_hard']:>7.3f}{b['ap_hard']:>7.3f}"
        )

    # ── LOSO CV: pick params on 6 seqs (by ap_hard), evaluate on held-out seq ──
    seqs = sorted(set(seq))
    cfgs = [
        (v, s, sat, sh, n)
        for v in VEL
        for s in SPD
        for sat in SAT
        for sh in SHAPE
        for n in NORM
    ]
    cv_win, cv_bridge, cv_dist = [], [], []
    chosen = {}
    for held in seqs:
        tr = seq != held
        te = seq == held
        # choose config maximising ap_hard on training seqs
        best, bestv = None, -1
        for cfg in cfgs:
            sc = build_score(d, *cfg)
            h_tr = tr & hard
            v = ap(sc[h_tr], y[h_tr])
            if v > bestv:
                bestv, best = v, cfg
        chosen[held] = best
        sc = build_score(d, *best)
        h_te = te & hard
        if y[h_te].sum() == 0:
            continue
        cv_win.append(ap(sc[h_te], y[h_te]))
        cv_bridge.append(ap(-d["bridge_dist"][h_te], y[h_te]))
        cv_dist.append(ap(-d["dist_h"][h_te], y[h_te]))

    print("\n=== leave-one-sequence-out CV (AP in hard region, held-out) ===")
    print(
        f"  speed-weighted (per-fold tuned): {np.mean(cv_win):.3f} ± {np.std(cv_win):.3f}"
    )
    print(
        f"  bridge_dist baseline:            {np.mean(cv_bridge):.3f} ± {np.std(cv_bridge):.3f}"
    )
    print(
        f"  dist_h baseline:                 {np.mean(cv_dist):.3f} ± {np.std(cv_dist):.3f}"
    )
    print("  per-fold chosen cfg (vel,spd,sat,shape,norm):")
    for h in seqs:
        print(f"    hold {h}: {chosen[h]}")


if __name__ == "__main__":
    main()
