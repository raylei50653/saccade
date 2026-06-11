#!/usr/bin/env python3
"""Distribution plots for relink speed and bridge_dist analysis."""

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


def _foot(cx, cy, h):
    return cx, cy + 0.5 * h


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


def velocity_regression_4(seg):
    if len(seg) < 4:
        return 0.0, 0.0
    x0, y0 = _foot(*seg[-4][1:4])
    x1, y1 = _foot(*seg[-3][1:4])
    x2, y2 = _foot(*seg[-2][1:4])
    x3, y3 = _foot(*seg[-1][1:4])
    return (3.0 * x3 + x2 - x1 - 3.0 * x0) / 10.0, (3.0 * y3 + y2 - y1 - 3.0 * y0) / 10.0


def velocity_regression_4_reverse(seg):
    if len(seg) < 4:
        return 0.0, 0.0
    x0, y0 = _foot(*seg[3][1:4])
    x1, y1 = _foot(*seg[2][1:4])
    x2, y2 = _foot(*seg[1][1:4])
    x3, y3 = _foot(*seg[0][1:4])
    return (3.0 * x3 + x2 - x1 - 3.0 * x0) / 10.0, (3.0 * y3 + y2 - y1 - 3.0 * y0) / 10.0


def velocity_mean(seg, n=4):
    seg = seg[-n:]
    if len(seg) < 2:
        return 0.0, 0.0
    vx = vy = 0.0
    count = 0
    for (f0, cx0, cy0, h0), (f1, cx1, cy1, h1) in zip(seg[:-1], seg[1:]):
        dt = max(f1 - f0, 1)
        x0, y0 = _foot(cx0, cy0, h0)
        x1, y1 = _foot(cx1, cy1, h1)
        vx += (x1 - x0) / dt
        vy += (y1 - y0) / dt
        count += 1
    return (vx / count, vy / count) if count else (0.0, 0.0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")

    all_tracks = {}
    for seq in SEQUENCES:
        p = MOT_DIR / f"{seq}.txt"
        if p.exists():
            all_tracks[seq] = load_tracks(p)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    # Collect all features
    speeds_mean, speeds_regr = [], []
    lost_speeds, cand_speeds = [], []
    gaps, dists_h, bridges_mean, bridges_regr = [], [], [], []
    y_vals, is_hard = [], []

    for row in rows:
        seq = row["seq"]
        lid, cid = int(row["lost_id"]), int(row["cand_id"])
        tracks = all_tracks.get(seq, {})
        if lid not in tracks or cid not in tracks:
            continue
        traj_a, traj_b = tracks[lid], tracks[cid]
        if len(traj_a) < 2 or len(traj_b) < 2:
            continue
        if traj_a[-1][0] >= traj_b[0][0]:
            continue
        if int(row["gt_valid"]) != 1:
            continue

        h_ref = max((traj_a[-1][3] + traj_b[0][3]) * 0.5, 1.0)
        ax, ay = _foot(*traj_a[-1][1:4])
        bx, by = _foot(*traj_b[0][1:4])
        gap = traj_b[0][0] - traj_a[-1][0]

        # closed-form regression velocity
        vax_r, vay_r = velocity_regression_4(traj_a)
        vbx_r, vby_r = velocity_regression_4_reverse(traj_b)
        half = gap * 0.5
        mlx_r, mly_r = ax + vax_r * half, ay + vay_r * half
        mcx_r, mcy_r = bx - vbx_r * half, by - vby_r * half
        bd_r = math.hypot(mlx_r - mcx_r, mly_r - mcy_r) / h_ref

        # mean velocity
        vax_m, vay_m = velocity_mean(traj_a, 4)
        vbx_m, vby_m = velocity_mean(traj_b, 4)
        mlx_m, mly_m = ax + vax_m * half, ay + vay_m * half
        mcx_m, mcy_m = bx - vbx_m * half, by - vby_m * half
        bd_m = math.hypot(mlx_m - mcx_m, mly_m - mcy_m) / h_ref

        dh = math.hypot(bx - ax, by - ay) / h_ref
        ls = math.hypot(vax_m, vay_m) / h_ref
        cs = math.hypot(vbx_m, vby_m) / h_ref
        ls_r = math.hypot(vax_r, vay_r) / h_ref
        cs_r = math.hypot(vbx_r, vby_r) / h_ref

        ms_m = min(ls, cs)
        ms_r = min(ls_r, cs_r)

        speeds_mean.append(ms_m)
        speeds_regr.append(ms_r)
        lost_speeds.append(ls)
        cand_speeds.append(cs)
        gaps.append(gap)
        dists_h.append(dh)
        bridges_mean.append(bd_m)
        bridges_regr.append(bd_r)
        y_vals.append(int(row["gt_match"]))
        is_hard.append(bd_m <= 1.0)

    speeds_mean = np.array(speeds_mean)
    speeds_regr = np.array(speeds_regr)
    lost_speeds = np.array(lost_speeds)
    cand_speeds = np.array(cand_speeds)
    gaps = np.array(gaps)
    dists_h = np.array(dists_h)
    bridges_mean = np.array(bridges_mean)
    bridges_regr = np.array(bridges_regr)
    y_vals = np.array(y_vals)
    is_hard = np.array(is_hard)

    pos_mask = y_vals == 1
    neg_mask = y_vals == 0
    hard_mask = is_hard
    n_all = len(y_vals)
    n_pos = int(pos_mask.sum())
    print(f"GT-valid pairs: {n_all} ({n_pos} pos, {n_all - n_pos} neg)")

    # ── Figure 1: Speed distributions (4 subplots) ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    speed_bins = np.logspace(-3, 0, 60)

    # 1a: Lost exit speed histogram
    ax = axes[0, 0]
    ax.hist(lost_speeds[pos_mask], bins=speed_bins, alpha=0.7, color="#4CAF50",
            label=f"True relink (n={n_pos})", density=True)
    ax.hist(lost_speeds[neg_mask], bins=speed_bins, alpha=0.5, color="#2196F3",
            label=f"False (n={n_all-n_pos})", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Lost exit speed (h/f)")
    ax.set_ylabel("Density")
    ax.set_title("Lost Track Exit Speed Distribution")
    ax.legend(fontsize=8)
    ax.axvline(0.01, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.05, color="red", ls="--", lw=0.8, alpha=0.5, label="0.05")
    ax.text(0.03, 0.95, f"median={np.median(lost_speeds):.4f}", transform=ax.transAxes, fontsize=7, va="top")

    # 1b: Cand entry speed histogram
    ax = axes[0, 1]
    ax.hist(cand_speeds[pos_mask], bins=speed_bins, alpha=0.7, color="#4CAF50",
            label=f"True relink", density=True)
    ax.hist(cand_speeds[neg_mask], bins=speed_bins, alpha=0.5, color="#2196F3",
            label=f"False", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Candidate entry speed (h/f)")
    ax.set_ylabel("Density")
    ax.set_title("Candidate Entry Speed Distribution")
    ax.legend(fontsize=8)
    ax.axvline(0.01, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.05, color="red", ls="--", lw=0.8, alpha=0.5)

    # 1c: min(lost, cand) speed histogram (decisive for blend)
    ax = axes[1, 0]
    ax.hist(speeds_mean[pos_mask], bins=speed_bins, alpha=0.7, color="#4CAF50",
            label=f"True relink", density=True)
    ax.hist(speeds_mean[neg_mask], bins=speed_bins, alpha=0.5, color="#2196F3",
            label=f"False", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("min(lost_speed, cand_speed) (h/f)")
    ax.set_ylabel("Density")
    ax.set_title(r"$\min(v_{lost}, v_{cand})$ Distribution (mean-vel)")
    ax.legend(fontsize=8)
    pct_001 = (speeds_mean < 0.01).sum() / n_all * 100
    pct_005 = (speeds_mean < 0.05).sum() / n_all * 100
    pct_002 = (speeds_mean < 0.02).sum() / n_all * 100
    ax.axvline(0.01, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.02, color="orange", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.05, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.text(0.95, 0.95, f"<0.01: {pct_001:.0f}%\n<0.02: {pct_002:.0f}%\n<0.05: {pct_005:.0f}%",
            transform=ax.transAxes, fontsize=7, va="top", ha="right")

    # 1d: Gap histogram
    ax = axes[1, 1]
    gap_bins = np.linspace(0, 300, 40)
    ax.hist(gaps[pos_mask], bins=gap_bins, alpha=0.7, color="#4CAF50",
            label=f"True relink", density=True)
    ax.hist(gaps[neg_mask], bins=gap_bins, alpha=0.5, color="#2196F3",
            label=f"False", density=True)
    ax.set_xlabel("Gap (frames)")
    ax.set_ylabel("Density")
    ax.set_title("Gap Distribution")
    ax.legend(fontsize=8)
    ax.text(0.95, 0.95, f"median gap={np.median(gaps):.0f}", transform=ax.transAxes,
            fontsize=7, va="top", ha="right")

    plt.tight_layout()
    out_speed_dist = OUT_DIR / "relink_speed_distributions.png"
    fig.savefig(out_speed_dist, dpi=150, bbox_inches="tight")
    print(f"Saved {out_speed_dist}")
    plt.close(fig)

    # ── Figure 2: Scatter — bridge vs dist_h colored by speed ───────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (bd, title) in enumerate(zip(
        [bridges_mean, bridges_regr],
        ["bridge_dist (mean-vel)", "bridge_dist (closed-form regression)"]
    )):
        ax = axes[ax_idx]
        mask_slow = speeds_mean < 0.01
        mask_mid = (speeds_mean >= 0.01) & (speeds_mean < 0.05)
        mask_fast = speeds_mean >= 0.05

        # Plot all points as background
        ax.scatter(dists_h[neg_mask], bd[neg_mask], s=1, alpha=0.1, color="gray", label="_nolegend_")
        # True positives highlighted
        ax.scatter(dists_h[pos_mask & mask_slow], bd[pos_mask & mask_slow],
                   s=20, alpha=0.8, color="#2196F3", marker="o", edgecolors="white", linewidth=0.3,
                   label=f"pos (slow <0.01)")
        ax.scatter(dists_h[pos_mask & mask_mid], bd[pos_mask & mask_mid],
                   s=20, alpha=0.8, color="#FF9800", marker="s", edgecolors="white", linewidth=0.3,
                   label=f"pos (mid 0.01-0.05)")
        ax.scatter(dists_h[pos_mask & mask_fast], bd[pos_mask & mask_fast],
                   s=25, alpha=0.9, color="#4CAF50", marker="^", edgecolors="white", linewidth=0.3,
                   label=f"pos (fast ≥0.05)")

        # y=x line
        lims = (0, max(dists_h.max(), bd.max()) * 1.05)
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.3)
        ax.set_xlabel("dist_h (spatial only)")
        ax.set_ylabel("bridge_dist")
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=0.8, loc="upper left")
        ax.set_xlim(0, min(lims[1], 50))
        ax.set_ylim(0, min(lims[1], 50))

    plt.tight_layout()
    out_scatter = OUT_DIR / "relink_bridge_vs_disth_scatter.png"
    fig.savefig(out_scatter, dpi=150, bbox_inches="tight")
    print(f"Saved {out_scatter}")
    plt.close(fig)

    # ── Figure 3: Bar chart — fraction of pairs by speed bin ────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    speed_bins_edges = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 10]
    speed_labels = ["0–0.005", "0.005–0.01", "0.01–0.015", "0.015–0.02",
                    "0.02–0.03", "0.03–0.04", "0.04–0.05", "0.05–0.07", "0.07–0.10", "0.10+"]

    counts_all = []
    counts_pos = []
    for lo, hi in zip(speed_bins_edges[:-1], speed_bins_edges[1:]):
        m = (speeds_mean >= lo) & (speeds_mean < hi)
        counts_all.append(m.sum())
        counts_pos.append((m & pos_mask).sum())

    counts_all = np.array(counts_all)
    counts_pos = np.array(counts_pos)
    x = np.arange(len(speed_labels))
    width = 0.35

    ax = axes[0]
    bars = ax.bar(x, counts_all / counts_all.sum() * 100, color="#607D8B", edgecolor="white")
    for i, (c, pct) in enumerate(zip(counts_all, counts_all / counts_all.sum() * 100)):
        ax.text(i, pct + 0.3, f"{pct:.0f}%\n(n={c})", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(speed_labels, rotation=45, fontsize=7)
    ax.set_ylabel("% of all pairs")
    ax.set_title("Pair Distribution by Speed")

    ax = axes[1]
    ax.bar(x - width/2, counts_pos / counts_pos.sum() * 100, width, label="True relink (pos)",
           color="#4CAF50", edgecolor="white")
    ax.bar(x + width/2, (counts_all - counts_pos) / (counts_all.sum() - counts_pos.sum()) * 100,
           width, label="False (neg)", color="#F44336", edgecolor="white")
    for i, (cp, ca) in enumerate(zip(counts_pos, counts_all)):
        if cp > 0:
            ax.text(i, cp / counts_pos.sum() * 100 + 0.5, f"{cp}", ha="center", fontsize=6.5,
                    color="#2E7D32", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(speed_labels, rotation=45, fontsize=7)
    ax.set_ylabel("% of respective class")
    ax.set_title("Pos vs Neg Distribution by Speed")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_bar = OUT_DIR / "relink_speed_bar_chart.png"
    fig.savefig(out_bar, dpi=150, bbox_inches="tight")
    print(f"Saved {out_bar}")
    plt.close(fig)

    # ── Figure 4: AUC vs speed with sample size overlay ─────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))

    auc_dh_list, auc_bd_list, n_list, pos_list = [], [], [], []
    labels = []
    for lo, hi in [(0, 0.005), (0.005, 0.01), (0.01, 0.015), (0.015, 0.02),
                   (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.07),
                   (0.07, 0.10), (0.10, 10)]:
        m = (speeds_mean >= lo) & (speeds_mean < hi)
        n = int(m.sum())
        p_pos = int(y_vals[m].sum())
        if n < 2 or p_pos == 0:
            auc_dh_list.append(float("nan"))
            auc_bd_list.append(float("nan"))
        else:
            # Simple auc function
            def auc_lower(score, y):
                pos = int(y.sum())
                neg = len(y) - pos
                o = np.argsort(score)
                ss = score[o]
                r = np.empty(len(score))
                i = 0
                while i < len(score):
                    j = i
                    while j + 1 < len(score) and ss[j + 1] == ss[i]:
                        j += 1
                    r[o[i:j + 1]] = (i + j) / 2.0 + 1
                    i = j + 1
                return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
            auc_dh_list.append(auc_lower(dists_h[m], y_vals[m]))
            auc_bd_list.append(auc_lower(bridges_mean[m], y_vals[m]))
        n_list.append(n)
        pos_list.append(p_pos)
        labels.append(f"{lo:.3f}\n–{hi:.2f}" if hi < 10 else f"≥{lo:.2f}")

    x = np.arange(len(labels))
    width = 0.3
    valid = ~np.isnan(auc_dh_list)

    bars1 = ax1.bar(x[valid] - width/2, np.array(auc_dh_list)[valid], width,
                    label="dist_h (spatial)", color="#FF9800", edgecolor="white")
    bars2 = ax1.bar(x[valid] + width/2, np.array(auc_bd_list)[valid], width,
                    label="bridge_dist (mean-vel)", color="#2196F3", edgecolor="white")
    # Annotate with sample size
    for i in range(len(x)):
        if not valid[i]:
            continue
        ymax = max(auc_dh_list[i], auc_bd_list[i])
        ax1.text(i, ymax + 0.03, f"n={n_list[i]}\npos={pos_list[i]}",
                 ha="center", fontsize=6, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("AUC")
    ax1.set_title("AUC by Speed Bin with Sample Size")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, max(max(auc_dh_list), max(auc_bd_list)) * 1.25)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.3)

    # Add sample count on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(x[valid], np.array(n_list)[valid], width * 3, alpha=0.08, color="gray")
    ax2.set_ylabel("Sample count", alpha=0.4)
    ax2.set_ylim(0, max(n_list) * 1.2)

    plt.tight_layout()
    out_auc_speed = OUT_DIR / "relink_auc_by_speed_with_samples.png"
    fig.savefig(out_auc_speed, dpi=150, bbox_inches="tight")
    print(f"Saved {out_auc_speed}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
