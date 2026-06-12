#!/usr/bin/env python3
"""Validate the reach-gate model  R_total(G) = s*G + R_search(G)  against the
plain bridge / spatial gates on the offline relink-candidate table.

Idea: a lost track at x0 moving at exit speed s can reach ~ s*G over a G-frame
gap; a candidate at foot-distance dist_h is admissible if
    dist_h <= s*G + R_search(G).
So the "reach residual"  (dist_h - s*G) / shape(G)  is a score (lower = more
plausible). We compare its ranking power (AUC + recall/FP) to bridge_dist and
plain dist_h, overall and split by slow/fast exit speed.

  .venv/bin/python scripts/tools/validate_reach_gate.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def auc(score, y):
    """Mann-Whitney AUC; higher score => more likely positive (y==1)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    i = 0
    s_sorted = score[order]
    while i < len(score):
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = r[y == 1]
    return (rp.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def recall_fp(score, y, n_pts=200):
    """Sweep threshold (admit if score>=thr); return (fp_counts, recalls)."""
    thr = np.quantile(score, np.linspace(0, 1, n_pts))
    npos = max(int(y.sum()), 1)
    fps, recs = [], []
    for t in thr:
        sel = score >= t
        fps.append(int((sel & (y == 0)).sum()))
        recs.append((sel & (y == 1)).sum() / npos)
    return np.array(fps), np.array(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/relink_candidates.csv")
    )
    ap.add_argument("--out", type=Path, default=Path("scripts/tools/out/reach_gate"))
    ap.add_argument(
        "--fast-thr",
        type=float,
        default=0.02,
        help="lost_exit_speed (h/f) split for slow/fast",
    )
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r["gt_valid"] == "1"]
    g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    y = np.array([int(r["gt_match"]) for r in rows])
    gap, dist_h, bridge = g("gap"), g("dist_h"), g("bridge_dist")
    s = g("lost_exit_speed")
    print(f"gt_valid pairs: {len(y)}  pos={int(y.sum())}  neg={int((y == 0).sum())}")

    drift = s * gap
    scores = {
        "bridge_dist": -bridge,  # baseline (online gate)
        "dist_h (spatial only)": -dist_h,  # naive
        "reach: dist_h - s*G": -(dist_h - drift),  # R_search const
        "reach: (dist_h - s*G)/sqrtG": -(dist_h - drift) / np.sqrt(gap),
        "reach: (dist_h - s*G)/G": -(dist_h - drift) / gap,
    }

    print("\n=== AUC (higher score => true relink) ===")
    print(f"  {'score':<32} {'AUC_all':>8} {'AUC_slow':>9} {'AUC_fast':>9}")
    slow, fast = s < args.fast_thr, s >= args.fast_thr
    print(
        f"  (n: slow={int(slow.sum())} pos={int(y[slow].sum())} | "
        f"fast={int(fast.sum())} pos={int(y[fast].sum())})"
    )
    for name, sc in scores.items():
        a_all = auc(sc, y)
        a_slow = auc(sc[slow], y[slow])
        a_fast = auc(sc[fast], y[fast])
        print(f"  {name:<32} {a_all:>8.4f} {a_slow:>9.4f} {a_fast:>9.4f}")

    # recall at matched FP budgets, reach vs bridge
    print("\n=== recall at fixed FP budget (overall) ===")
    print(f"  {'FP<=':>6} {'bridge':>8} {'reach-sqrtG':>12} {'dist_h':>8}")
    fb, rb = recall_fp(scores["bridge_dist"], y)
    fr, rr = recall_fp(scores["reach: (dist_h - s*G)/sqrtG"], y)
    fd, rd = recall_fp(scores["dist_h (spatial only)"], y)
    for budget in [100, 250, 500, 1000, 2000]:

        def rec_at(fps, recs):
            ok = fps <= budget
            return recs[ok].max() if ok.any() else 0.0

        print(
            f"  {budget:>6} {rec_at(fb, rb):>8.1%} {rec_at(fr, rr):>12.1%} "
            f"{rec_at(fd, rd):>8.1%}"
        )

    # ── chart: recall vs FP ──
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, sc in scores.items():
        fps, recs = recall_fp(sc, y)
        ax[0].plot(fps, recs, label=name)
    ax[0].set(
        xlabel="false positives (of 19k neg)",
        ylabel="recall (of 256 pos)",
        title="Reach gate vs bridge / spatial (overall)",
        xlim=(0, 3000),
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # fast subset only (where direction/drift should help most)
    for name, sc in scores.items():
        fps, recs = recall_fp(sc[fast], y[fast])
        ax[1].plot(fps, recs, label=name)
    ax[1].set(
        xlabel="false positives (fast subset)",
        ylabel="recall (fast pos)",
        title=f"Fast subset (exit speed >= {args.fast_thr} h/f)",
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"\nWrote chart -> {png}")


if __name__ == "__main__":
    main()
