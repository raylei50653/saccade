#!/usr/bin/env python3
"""Plot box height change analysis for relink pairs."""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOT_DIR = PROJECT_ROOT / "results" / "MOT17_mamba_whole_graph_nointerp"
CSV_PATH = PROJECT_ROOT / "scripts" / "tools" / "out" / "relink_candidates_mamba_nointerp.csv"
OUT_DIR = PROJECT_ROOT / "scratch" / "pipeline_math_validation" / "output"

SEQUENCES = [
    "MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP",
    "MOT17-09-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP",
]


def load_tracks(path: Path) -> dict[int, list]:
    tracks: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            tracks[tid].append((frm, x + w / 2.0, y + h / 2.0, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading...")

    all_tracks = {}
    for seq in SEQUENCES:
        p = MOT_DIR / f"{seq}.txt"
        if p.exists():
            all_tracks[seq] = load_tracks(p)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    data = []
    for row in rows:
        seq = row["seq"]
        lid, cid = int(row["lost_id"]), int(row["cand_id"])
        tracks = all_tracks.get(seq, {})
        if lid not in tracks or cid not in tracks:
            continue
        ta, tb = tracks[lid], tracks[cid]
        if len(ta) < 2 or len(tb) < 2:
            continue
        if ta[-1][0] >= tb[0][0]:
            continue
        if int(row["gt_valid"]) != 1:
            continue
        h_lost = ta[-1][3]
        h_cand = tb[0][3]
        gap = tb[0][0] - ta[-1][0]
        h_ratio = h_cand / max(h_lost, 1.0)
        log_rate = math.log(max(h_ratio, 0.01)) / max(gap, 1)
        h_diff = abs(h_cand - h_lost) / max((h_lost + h_cand) * 0.5, 1.0)
        data.append((gap, h_lost, h_cand, h_ratio, log_rate, h_diff,
                     int(row["gt_match"])))

    gaps = np.array([d[0] for d in data])
    h_ratios = np.array([d[3] for d in data])
    log_rates = np.array([d[4] for d in data])
    h_diffs = np.array([d[5] for d in data])
    y = np.array([d[6] for d in data])
    pos = y == 1
    neg = y == 0
    print(f"Total: pos={pos.sum()}, neg={neg.sum()}")

    # ── Figure 1: Height ratio distributions ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1a: Histogram of h_ratio (linear scale, clipped)
    ax = axes[0, 0]
    bins = np.linspace(0, 4, 80)
    ax.hist(np.clip(h_ratios[pos], 0, 4), bins=bins, alpha=0.7, color="#4CAF50",
            density=True, label=f"True (n={pos.sum()})")
    ax.hist(np.clip(h_ratios[neg], 0, 4), bins=bins, alpha=0.5, color="#F44336",
            density=True, label=f"False (n={neg.sum()})")
    # Gate zone
    ax.axvspan(0.7, 1.4, alpha=0.1, color="#4CAF50")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.text(0.72, 0.95, f"gate [0.7,1.4]\nkeeps {((h_ratios[pos]>=0.7)&(h_ratios[pos]<=1.4)).mean():.0%} true\n"
            f"rejects {1-((h_ratios[neg]>=0.7)&(h_ratios[neg]<=1.4)).mean():.0%} false",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.set_xlabel("Height ratio (candidate / lost)")
    ax.set_ylabel("Density")
    ax.set_title("Box Height Ratio Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)

    # 1b: Log-scale histogram
    ax = axes[0, 1]
    log_bins = np.logspace(-1, 1, 60)
    ax.hist(h_ratios[pos], bins=log_bins, alpha=0.7, color="#4CAF50",
            density=True, label=f"True")
    ax.hist(h_ratios[neg], bins=log_bins, alpha=0.5, color="#F44336",
            density=True, label=f"False")
    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.axvspan(0.7, 1.4, alpha=0.1, color="#4CAF50")
    ax.set_xlabel("Height ratio (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Box Height Ratio (log scale)")
    ax.legend(fontsize=8)

    # 1c: h_ratio vs gap (scatter, true only)
    ax = axes[1, 0]
    ax.scatter(gaps[neg][::20], h_ratios[neg][::20], s=1, alpha=0.03, color="gray",
               rasterized=True)
    ax.scatter(gaps[pos], h_ratios[pos], s=20, alpha=0.8, color="#4CAF50",
               edgecolors="white", linewidth=0.3, zorder=5, label=f"True (n={pos.sum()})")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.axhspan(0.7, 1.4, alpha=0.1, color="#4CAF50")
    ax.set_xlabel("Gap (frames)")
    ax.set_ylabel("Height ratio")
    ax.set_title("Height Ratio vs Gap")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 3.5)

    # 1d: log change rate vs gap
    ax = axes[1, 1]
    abs_log = np.abs(log_rates)
    gap_bins = [(1, 10), (11, 30), (31, 60), (61, 150), (151, 300)]
    gap_labels = ["1-10", "11-30", "31-60", "61-150", "151-300"]
    x = np.arange(len(gap_bins))
    width = 0.3

    mean_pos_list, mean_neg_list = [], []
    for lo, hi in gap_bins:
        gm = (gaps >= lo) & (gaps <= hi)
        mean_pos_list.append(np.mean(abs_log[gm & pos]) if (gm & pos).sum() > 0 else 0)
        mean_neg_list.append(np.mean(abs_log[gm & neg]) if (gm & neg).sum() > 0 else 0)
    ax.bar(x - width/2, mean_pos_list, width, color="#4CAF50", alpha=0.8,
           label="True relink", edgecolor="white")
    ax.bar(x + width/2, mean_neg_list, width, color="#F44336", alpha=0.6,
           label="False", edgecolor="white")
    for i, (mp, mn) in enumerate(zip(mean_pos_list, mean_neg_list)):
        ax.text(i, max(mp, mn) + 0.001, f"{mp:.3f}", ha="center", fontsize=7, color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(gap_labels)
    ax.set_xlabel("Gap (frames)")
    ax.set_ylabel("Mean |log(ratio)| / gap")
    ax.set_title("Height Change Rate by Gap")
    ax.legend(fontsize=8)
    ax.axhline(0.006, color="#FF9800", ls="--", lw=0.8, alpha=0.5,
               label="doc's ±6%/frame threshold")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out1 = OUT_DIR / "relink_height_ratio.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    # ── Figure 2: Gate effectiveness ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 2a: Recall vs precision trade-off for different ratio thresholds
    ax = axes[0]
    thresholds = np.linspace(0.01, 0.5, 50)  # symmetric half-width
    recalls, prec_gains, reject_rates = [], [], []
    for t in thresholds:
        lo, hi = 1.0 - t, 1.0 + t
        keep_pos = (h_ratios[pos] >= lo) & (h_ratios[pos] <= hi)
        keep_neg = (h_ratios[neg] >= lo) & (h_ratios[neg] <= hi)
        rec = keep_pos.mean()
        rej = 1 - keep_neg.mean()
        prec_gains.append(rej)
        recalls.append(rec)
    ax.plot(thresholds, recalls, "-", color="#4CAF50", lw=2, label="Recall (true kept)")
    ax.plot(thresholds, prec_gains, "-", color="#2196F3", lw=2, label="Precision gain (false rejected)")
    # Mark key points
    for t_val, label in [(0.10, "±10%"), (0.20, "±20%"), (0.30, "±30%"), (0.50, "±50%")]:
        idx = np.argmin(np.abs(thresholds - t_val))
        ax.axvline(t_val, color="gray", ls=":", lw=0.8, alpha=0.3)
        ax.text(t_val + 0.01, 0.1, label, fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("Symmetric ratio threshold (1 ± t)")
    ax.set_ylabel("Fraction")
    ax.set_title("Height Ratio Gate: Recall vs Precision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2b: Per-frame change rate histogram with gate line
    ax = axes[1]
    rates = np.abs(log_rates)
    bins = np.linspace(0, 0.03, 60)
    ax.hist(rates[pos], bins=bins, alpha=0.7, color="#4CAF50", density=True,
            label=f"True (median={np.median(rates[pos]):.4f})")
    ax.hist(rates[neg], bins=bins, alpha=0.5, color="#F44336", density=True,
            label=f"False (median={np.median(rates[neg]):.4f})")
    ax.axvline(0.006, color="#FF9800", ls="--", lw=1.5, alpha=0.8,
               label="gate: |log(r)|/gap ≤ 0.006 (±6%/frame)")
    ax.set_xlabel("|log(height ratio)| / gap")
    ax.set_ylabel("Density")
    ax.set_title("Per-Frame Height Change Rate")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 0.03)

    plt.tight_layout()
    out2 = OUT_DIR / "relink_height_gate.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig)

    # Summary stats
    print(f"\n=== Gate effectiveness ===")
    for lo, hi, label in [(0.5, 2.0, "[0.5, 2.0]"), (0.6, 1.67, "[0.6, 1.67]"),
                           (0.7, 1.4, "[0.7, 1.4]"), (0.8, 1.25, "[0.8, 1.25]"),
                           (0.85, 1.18, "[0.85, 1.18]")]:
        keep_pos = (h_ratios[pos] >= lo) & (h_ratios[pos] <= hi)
        keep_neg = (h_ratios[neg] >= lo) & (h_ratios[neg] <= hi)
        rec = keep_pos.mean()
        rej = 1 - keep_neg.mean()
        prec = pos.sum() / (pos.sum() + neg.sum())
        prec_after = keep_pos.sum() / max(keep_pos.sum() + keep_neg.sum(), 1)
        n_kept_pos = keep_pos.sum()
        n_kept_neg = keep_neg.sum()
        print(f"  gate {label:>12}: recall={rec:.1%}  reject_false={rej:.1%}  "
              f"prec {prec:.1%}→{prec_after:.1%}  ({n_kept_pos}/{n_kept_neg})")

    print("\nDone.")


if __name__ == "__main__":
    main()
