#!/usr/bin/env python3
"""Offline sweep of lost-track exit velocity estimation schemes for the bridge score.

Each variant uses a different velocity estimator for the lost track (exit) and the
candidate track (entry).  All other things (h_ref, dist_h, gap) are held constant.

Variants
--------
v4ep    : 4-frame endpoint diff     — current CSV default (fwd_resid/bwd_resid cols)
v4ols   : 4-frame OLS               — matches GPU bridge_vel4 kernel formula
v8ols   : 8-frame OLS               — fits current FOOT_RING_CAP=8, no GPU change
v16ols  : 16-frame OLS              — ring_cap 16
v32ols  : 32-frame OLS              — ring_cap 32
lb10    : long baseline T=10 W=4    — ring_cap 14
lb20    : long baseline T=20 W=5    — ring_cap 25
lb30    : long baseline T=30 W=10   — ring_cap 40 (user's proposed scheme)

Bridge score formula (same as GPU kernel):
    w      = clip(sqrt(s_lost / 0.12), 0, 1)
    bdist  = w * 0.5*(fwd_r + bwd_r) + (1-w) * dist_h
Higher score = lower bdist.

GPU feasibility column shows the minimum foot-ring buffer size required.

Usage
-----
  .venv/bin/python scripts/tools/sweep_velocity_variants.py
  .venv/bin/python scripts/tools/sweep_velocity_variants.py --csv scripts/tools/out/relink_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# ── metrics ───────────────────────────────────────────────────────────────────
def _auc(score: np.ndarray, y: np.ndarray) -> float:
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


def _ap(score: np.ndarray, y: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    o = np.argsort(-score, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / pos
    drec = np.diff(np.concatenate([[0], rec]))
    return float((prec * drec).sum())


def _bridge_score(fwd_r, bwd_r, dist_h, s_lost):
    w = np.clip(np.sqrt(np.clip(s_lost / 0.12, 0.0, None)), 0.0, 1.0)
    return -(w * 0.5 * (fwd_r + bwd_r) + (1.0 - w) * dist_h)


def _loso_cv(seqs, fwd_r, bwd_r, dist_h, s_lost, y) -> np.ndarray:
    aps = []
    for seq in np.unique(seqs):
        mask = seqs == seq
        if y[mask].sum() == 0:
            continue
        sc = _bridge_score(fwd_r[mask], bwd_r[mask], dist_h[mask], s_lost[mask])
        aps.append(_ap(sc, y[mask]))
    return np.array(aps)


# ── GPU ring-cap required per variant ─────────────────────────────────────────
GPU_CAP = {
    "v4ep": 4,
    "v4ols": 4,
    "v8ols": 8,
    "v16ols": 16,
    "v32ols": 32,
    "lb10": 14,  # T+W = 10+4
    "lb20": 25,  # T+W = 20+5
    "lb30": 40,  # T+W = 30+10
}

VARIANT_LABELS = {
    "v4ep": "v4_endpt  (current CSV, _velocity)        ",
    "v4ols": "v4_ols    (GPU kernel bridge_vel4)         ",
    "v8ols": "v8_ols    (ring_cap=8, no GPU change)      ",
    "v16ols": "v16_ols   (ring_cap=16)                    ",
    "v32ols": "v32_ols   (ring_cap=32)                    ",
    "lb10": "lb_T10_W4 (ring_cap=14)                    ",
    "lb20": "lb_T20_W5 (ring_cap=25)                    ",
    "lb30": "lb_T30_W10 (ring_cap=40, user proposal)    ",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/relink_candidates.csv")
    )
    parser.add_argument(
        "--hard-dist",
        type=float,
        default=1.0,
        help="bridge_dist threshold defining the hard pool (default 1.0 h)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"CSV not found: {args.csv}\nRun build_relink_candidates.py first."
        )

    rows = list(csv.DictReader(args.csv.open()))
    if not rows:
        raise SystemExit("CSV is empty.")

    # Check which new columns are present
    new_tags = ["v4ols", "v8ols", "v16ols", "v32ols", "lb10", "lb20", "lb30"]
    has_new = all(f"fwd_{t}" in rows[0] for t in new_tags)
    if not has_new:
        raise SystemExit(
            "CSV is missing velocity-variant columns.\n"
            "Re-run build_relink_candidates.py to regenerate."
        )

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    seqs = np.array([r["seq"] for r in rows])
    y = col("gt_match").astype(bool) & col("gt_valid").astype(bool)
    dist_h = col("dist_h")
    bd_orig = col("bridge_dist")  # used only to define hard mask

    # v4ep: current default — uses fwd_resid/bwd_resid/lost_exit_speed directly
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "v4ep": (col("fwd_resid"), col("bwd_resid"), col("lost_exit_speed")),
    }
    for tag in new_tags:
        variants[tag] = (col(f"fwd_{tag}"), col(f"bwd_{tag}"), col(f"sl_{tag}"))

    hard_mask = bd_orig <= args.hard_dist

    # ── per-variant stats ─────────────────────────────────────────────────────
    # For understanding noise, compute per-variant mean s_lost and mean |fwd-v4ep|
    fwd_ref = variants["v4ep"][0]
    bwd_ref = variants["v4ep"][1]

    print(f"\nPool : {len(y)} pairs | {int(y.sum())} pos ({100 * y.mean():.1f}%)")
    print(
        f"Hard (bridge_dist ≤ {args.hard_dist}h): "
        f"{int(hard_mask.sum())} pairs | {int(y[hard_mask].sum())} pos"
    )

    hdr = (
        f"{'variant':<47}  {'GPU cap':>7}  "
        f"{'AUC full':>9}  {'AUC hard':>9}  "
        f"{'AP full':>8}  {'AP hard':>8}  "
        f"{'LOSO±σ':>12}  "
        f"{'Δfwd':>7}  {'Δbwd':>7}  {'s̄_lost':>7}"
    )
    print(f"\n{hdr}")
    print("─" * len(hdr))

    for tag, (fwd_r, bwd_r, s_lost) in variants.items():
        sc_full = _bridge_score(fwd_r, bwd_r, dist_h, s_lost)
        sc_hard = sc_full[hard_mask]
        y_hard = y[hard_mask]

        auc_f = _auc(sc_full, y)
        auc_h = _auc(sc_hard, y_hard)
        ap_f = _ap(sc_full, y)
        ap_h = _ap(sc_hard, y_hard)
        loso = _loso_cv(seqs, fwd_r, bwd_r, dist_h, s_lost, y)
        loso_s = f"{loso.mean():.3f}±{loso.std():.3f}" if len(loso) else "n/a"

        dfwd = float(np.abs(fwd_r - fwd_ref).mean())
        dbwd = float(np.abs(bwd_r - bwd_ref).mean())
        s_mean = float(s_lost.mean())

        cap = GPU_CAP.get(tag, "?")
        label = VARIANT_LABELS.get(tag, f"{tag:<45}")

        print(
            f"{label}  {cap:>7}  "
            f"{auc_f:>9.4f}  {auc_h:>9.4f}  "
            f"{ap_f:>8.4f}  {ap_h:>8.4f}  "
            f"{loso_s:>12}  "
            f"{dfwd:>7.4f}  {dbwd:>7.4f}  {s_mean:>7.4f}"
        )

    # ── breakdown by gap bucket ───────────────────────────────────────────────
    print("\n── AP by gap bucket (pos / total / AP_v4ep / AP_best) ──")
    buckets = [(1, 5), (6, 15), (16, 30), (31, 60)]
    gaps = col("gap")

    for lo, hi in buckets:
        m = (gaps >= lo) & (gaps <= hi)
        if y[m].sum() == 0:
            continue
        # find best variant in this bucket by AP
        best_tag, best_ap = "v4ep", 0.0
        ap_v4ep = 0.0
        for tag, (fwd_r, bwd_r, s_lost) in variants.items():
            sc = _bridge_score(fwd_r[m], bwd_r[m], dist_h[m], s_lost[m])
            a = _ap(sc, y[m])
            if tag == "v4ep":
                ap_v4ep = a
            if a > best_ap:
                best_ap, best_tag = a, tag
        print(
            f"  gap [{lo:2d},{hi:2d}]  pos={int(y[m].sum()):3d}  total={int(m.sum()):4d}  "
            f"AP_v4ep={ap_v4ep:.4f}  AP_best={best_ap:.4f} ({best_tag})"
        )

    # ── breakdown by lost track lifespan ─────────────────────────────────────
    print(
        "\n── AP by lost-track lifespan (short tracks lack history for long-baseline) ──"
    )
    lifespans = col("lost_lifespan")
    lbuckets = [(1, 8), (9, 20), (21, 40), (41, 200)]

    for lo, hi in lbuckets:
        m = (lifespans >= lo) & (lifespans <= hi)
        if y[m].sum() == 0:
            continue
        best_tag, best_ap = "v4ep", 0.0
        ap_v4ep = 0.0
        for tag, (fwd_r, bwd_r, s_lost) in variants.items():
            sc = _bridge_score(fwd_r[m], bwd_r[m], dist_h[m], s_lost[m])
            a = _ap(sc, y[m])
            if tag == "v4ep":
                ap_v4ep = a
            if a > best_ap:
                best_ap, best_tag = a, tag
        print(
            f"  lifespan [{lo:3d},{hi:3d}]  pos={int(y[m].sum()):3d}  total={int(m.sum()):4d}  "
            f"AP_v4ep={ap_v4ep:.4f}  AP_best={best_ap:.4f} ({best_tag})"
        )

    # ── AUC / AP vs speed gate ────────────────────────────────────────────────
    # Gate: keep only pairs where v4ep s_lost >= threshold (tracks moving fast
    # enough for velocity to carry signal).  Shows whether long-baseline velocity
    # is better in the velocity-active regime.
    print("\n── AUC full vs minimum-speed gate (s_lost_v4ep ≥ thr) ──")
    s_lost_ref = variants["v4ep"][2]  # reference speed for gating
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
    tag_short = {
        "v4ep": "v4ep ",
        "v4ols": "v4ols",
        "v8ols": "v8ols",
        "v16ols": "v16ol",
        "v32ols": "v32ol",
        "lb10": "lb10 ",
        "lb20": "lb20 ",
        "lb30": "lb30 ",
    }
    col_hdr = "  ".join(f"{tag_short[t]:>5}" for t in variants)
    print(f"  {'thr':>5}  {'pos':>4}  {'tot':>5}  {col_hdr}")
    print(
        f"  {'─' * 5}  {'─' * 4}  {'─' * 5}  {'─' * (7 * len(variants) + 2 * (len(variants) - 1))}"
    )

    for thr in thresholds:
        m = s_lost_ref >= thr
        if y[m].sum() < 3:
            continue
        row_vals = []
        for tag, (fwd_r, bwd_r, s_lost) in variants.items():
            sc = _bridge_score(fwd_r[m], bwd_r[m], dist_h[m], s_lost[m])
            row_vals.append(f"{_auc(sc, y[m]):>5.3f}")
        print(
            f"  {thr:>5.2f}  {int(y[m].sum()):>4}  {int(m.sum()):>5}  {'  '.join(row_vals)}"
        )

    print("\n── AP full vs minimum-speed gate ──")
    print(f"  {'thr':>5}  {'pos':>4}  {'tot':>5}  {col_hdr}")
    print(
        f"  {'─' * 5}  {'─' * 4}  {'─' * 5}  {'─' * (7 * len(variants) + 2 * (len(variants) - 1))}"
    )

    for thr in thresholds:
        m = s_lost_ref >= thr
        if y[m].sum() < 3:
            continue
        row_vals = []
        for tag, (fwd_r, bwd_r, s_lost) in variants.items():
            sc = _bridge_score(fwd_r[m], bwd_r[m], dist_h[m], s_lost[m])
            row_vals.append(f"{_ap(sc, y[m]):>5.3f}")
        print(
            f"  {thr:>5.2f}  {int(y[m].sum()):>4}  {int(m.sum()):>5}  {'  '.join(row_vals)}"
        )

    # ── speed-bucket AUC matrix ───────────────────────────────────────────────
    print("\n── AUC per speed bucket (hard pool only, bridge_dist ≤ 1h) ──")
    sbuckets = [(0.0, 0.05), (0.05, 0.12), (0.12, 0.25), (0.25, 99.9)]
    print(f"  {'bucket':>14}  {'pos':>4}  {'tot':>5}  {col_hdr}")
    for slo, shi in sbuckets:
        m = (s_lost_ref >= slo) & (s_lost_ref < shi) & hard_mask
        if y[m].sum() < 2:
            print(f"  [{slo:.2f},{shi:.2f})     pos={int(y[m].sum())}  (skip)")
            continue
        row_vals = []
        for tag, (fwd_r, bwd_r, s_lost) in variants.items():
            sc = _bridge_score(fwd_r[m], bwd_r[m], dist_h[m], s_lost[m])
            row_vals.append(f"{_auc(sc, y[m]):>5.3f}")
        print(
            f"  [{slo:.2f},{shi:.2f})  {int(y[m].sum()):>4}  {int(m.sum()):>5}  {'  '.join(row_vals)}"
        )


if __name__ == "__main__":
    main()
