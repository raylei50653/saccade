#!/usr/bin/env python3
"""Offline sweep of h_ref normalization variants for the bridge score.

Tests whether using trimmed or extrapolated heights instead of the raw last/first
frame heights improves discrimination on the relink candidate pool.

Variants
--------
avg_raw     : avg(h_lost_raw, h_cand_raw)           — current default
avg_trim1   : avg(h_lost_trim1, h_cand_trim1)       — A: skip 1 terminal frame each
avg_win6    : avg(h_lost_win6, h_cand_win6)         — B: 6-frame window, skip 1
avg_extrap  : avg(h_lost_extrap, h_cand_extrap)     — C: linear extrap to gap midpoint
lost_only   : h_lost_raw                            — b605ac3e era (regressed)
trim1_raw   : avg(h_lost_trim1, h_cand_raw)         — trim lost only
extrap_raw  : avg(h_lost_extrap, h_cand_raw)        — extrap lost, raw cand

For each variant, re-normalise the raw pixel distances (recovered via dist_h * h_ref_orig)
and report AUC + AP on the full pool and the hard region (bridge_dist_orig ≤ 1).
Also runs LOSO-CV AP for the speed-weighted score under each variant.

Usage
-----
  .venv/bin/python scripts/tools/sweep_href_variants.py
  .venv/bin/python scripts/tools/sweep_href_variants.py --csv scripts/tools/out/relink_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# ── metrics ──────────────────────────────────────────────────────────────────
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


def _speed_weighted_score(dist_h, fwd_r, bwd_r, s_lost):
    sym_fb = 0.5 * (fwd_r + bwd_r)
    w = np.clip(np.sqrt(np.clip(s_lost / 0.12, 0, None)), 0, 1)
    return -(w * sym_fb + (1 - w) * dist_h)


def _loso_cv(seqs, dist_h, fwd_r, bwd_r, s_lost, y):
    aps = []
    for seq in np.unique(seqs):
        mask_tr = seqs != seq
        mask_te = ~mask_tr
        if y[mask_te].sum() == 0:
            continue
        sc = _speed_weighted_score(
            dist_h[mask_te], fwd_r[mask_te], bwd_r[mask_te], s_lost[mask_te]
        )
        aps.append(_ap(sc, y[mask_te]))
    return np.array(aps)


# ── main ─────────────────────────────────────────────────────────────────────
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
        help="bridge_dist threshold defining the hard region",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"CSV not found: {args.csv}\nRun build_relink_candidates.py first."
        )

    rows = list(csv.DictReader(args.csv.open()))
    required_new = {
        "h_lost_raw",
        "h_cand_raw",
        "h_lost_trim1",
        "h_cand_trim1",
        "h_lost_win6",
        "h_cand_win6",
        "h_lost_extrap",
        "h_cand_extrap",
    }
    if not required_new.issubset(rows[0].keys()):
        raise SystemExit(
            "CSV is missing h_ref variant columns.\n"
            "Re-run build_relink_candidates.py to regenerate."
        )

    def col(name):
        return np.array([float(r[name]) for r in rows])

    seqs = np.array([r["seq"] for r in rows])
    y = col("gt_match").astype(bool) & col("gt_valid").astype(bool)
    h_ref_orig = col("h_ref")  # avg(last, first) — used to recover raw px
    bd_orig = col("bridge_dist")  # normalised by h_ref_orig

    # raw pixel distances (re-normalisable)
    dist_px = col("dist_h") * h_ref_orig
    fwd_px = col("fwd_resid") * h_ref_orig
    bwd_px = col("bwd_resid") * h_ref_orig

    # speed is in h/f; convert to px/f then re-normalise per variant below
    s_lost_hf = col("lost_exit_speed")  # in heights/frame, divided by h_ref_orig
    s_lost_px = s_lost_hf * h_ref_orig  # px/frame (variant-independent)

    h_lost_raw = col("h_lost_raw")
    h_cand_raw = col("h_cand_raw")
    h_lost_trim1 = col("h_lost_trim1")
    h_cand_trim1 = col("h_cand_trim1")
    h_lost_win6 = col("h_lost_win6")
    h_cand_win6 = col("h_cand_win6")
    h_lost_extrap = col("h_lost_extrap")
    h_cand_extrap = col("h_cand_extrap")

    variants = {
        "avg_raw    (current default)": np.maximum(
            (h_lost_raw + h_cand_raw) * 0.5, 1.0
        ),
        "avg_trim1  (A: skip 1 terminal)": np.maximum(
            (h_lost_trim1 + h_cand_trim1) * 0.5, 1.0
        ),
        "avg_win6   (B: 6-frame window)": np.maximum(
            (h_lost_win6 + h_cand_win6) * 0.5, 1.0
        ),
        "avg_extrap (C: extrap midpoint)": np.maximum(
            (h_lost_extrap + h_cand_extrap) * 0.5, 1.0
        ),
        "lost_only  (b605ac3e, regressed)": np.maximum(h_lost_raw, 1.0),
        "trim1_raw  (trim lost, raw cand)": np.maximum(
            (h_lost_trim1 + h_cand_raw) * 0.5, 1.0
        ),
        "extrap_raw (extrap lost, raw cand)": np.maximum(
            (h_lost_extrap + h_cand_raw) * 0.5, 1.0
        ),
    }

    hard_mask = bd_orig <= args.hard_dist

    print(f"\nPool: {len(y)} pairs, {int(y.sum())} pos ({100 * y.mean():.1f}%)")
    print(
        f"Hard region (bridge_dist ≤ {args.hard_dist}): "
        f"{int(hard_mask.sum())} pairs, {int(y[hard_mask].sum())} pos"
    )

    header = f"{'variant':<42}  {'AUC full':>9}  {'AUC hard':>9}  {'AP full':>8}  {'AP hard':>8}  {'LOSO-CV AP':>11}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, h_ref_v in variants.items():
        dist_h_v = dist_px / h_ref_v
        fwd_r_v = fwd_px / h_ref_v
        bwd_r_v = bwd_px / h_ref_v
        s_lost_v = s_lost_px / h_ref_v

        sc_full = _speed_weighted_score(dist_h_v, fwd_r_v, bwd_r_v, s_lost_v)
        sc_hard = sc_full[hard_mask]
        y_hard = y[hard_mask]

        auc_f = _auc(sc_full, y)
        auc_h = _auc(sc_hard, y_hard)
        ap_f = _ap(sc_full, y)
        ap_h = _ap(sc_hard, y_hard)
        loso = _loso_cv(seqs, dist_h_v, fwd_r_v, bwd_r_v, s_lost_v, y)
        loso_s = f"{loso.mean():.3f}±{loso.std():.3f}" if len(loso) else "n/a"

        print(
            f"{name:<42}  {auc_f:>9.4f}  {auc_h:>9.4f}  {ap_f:>8.4f}  {ap_h:>8.4f}  {loso_s:>11}"
        )

    # also show the pure spatial baseline for reference
    sc_spatial = -dist_px / np.maximum((h_lost_raw + h_cand_raw) * 0.5, 1.0)
    print(
        f"\n{'dist_h only (spatial baseline)':<42}  "
        f"{_auc(sc_spatial, y):>9.4f}  "
        f"{_auc(sc_spatial[hard_mask], y[hard_mask]):>9.4f}  "
        f"{_ap(sc_spatial, y):>8.4f}  "
        f"{_ap(sc_spatial[hard_mask], y[hard_mask]):>8.4f}"
    )


if __name__ == "__main__":
    main()
