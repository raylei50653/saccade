#!/usr/bin/env python3
"""Analyze linear interpolation accuracy: how well does linear interp recover true positions?

For each confirmed track in the MOT output, we create "synthetic gaps" by removing
intermediate frames and measuring how well linear interpolation from surrounding frames
recovers the true intermediate position. This directly measures the interpolation
error as a function of gap length.
"""

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
GT_ROOT = PROJECT_ROOT / "datasets" / "MOT17" / "train"
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


def load_gt_tracks(path: Path) -> dict[int, list]:
    tracks: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, gid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            if gid <= 0:
                continue
            tracks[gid].append((frm, x + w / 2.0, y + h / 2.0, h))
    return dict(tracks)


def map_track_to_gt(track_id, tracks, gt_tracks, min_overlap=5):
    """Map predicted track to GT id."""
    gt_by_frame = defaultdict(list)
    for gid, traj in gt_tracks.items():
        for f, cx, cy, h in traj:
            gt_by_frame[f].append((gid, cx, cy, h))

    traj = tracks[track_id]
    votes = defaultdict(int)
    for f, cx, cy, h in traj:
        best_gid, best_d = -1, 1e30
        for gid, gx, gy, gh in gt_by_frame.get(f, ()):
            d = math.hypot(cx - gx, cy - gy)
            if d < max(h, gh) * 1.0 and d < best_d:
                best_d, best_gid = d, gid
        if best_gid >= 0:
            votes[best_gid] += 1
    if votes:
        gid, n = max(votes.items(), key=lambda kv: kv[1])
        if n >= min_overlap:
            return gid
    return -1


def interp_error(traj_a, traj_b, gap):
    """Linear interpolation error for synthetic gap between two frames.

    Returns per-frame error normalized by box height (h/f).
    """
    f0, cx0, cy0, h0 = traj_a
    f1, cx1, cy1, h1 = traj_b
    assert f1 - f0 == gap + 1, f"Expected gap={gap}, got {f1-f0-1}"

    errors_foot = []
    errors_cx = []
    errors_h = []
    for t in range(1, gap + 1):
        alpha = t / (gap + 1)
        # Interpolated center
        icx = cx0 + alpha * (cx1 - cx0)
        icy = cy0 + alpha * (cy1 - cy0)
        ih = h0 + alpha * (h1 - h0)
        # Actual center at frame f0 + t
        actual = traj_a  # placeholder — we need the actual frame

    return errors_foot, errors_cx, errors_h


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading tracks...")

    all_tracks = {}
    all_gt = {}
    for seq in SEQUENCES:
        p = MOT_DIR / f"{seq}.txt"
        if p.exists():
            all_tracks[seq] = load_tracks(p)
        g = GT_ROOT / seq / "gt" / "gt.txt"
        if g.exists():
            all_gt[seq] = load_gt_tracks(g)

    # Collect interpolation experiments
    # For each confirmed track (≥10 frames), take pairs of frames i and i+g+1
    # Compute linear interpolation to frame i+1..i+g
    # Compare with actual frame i+1..i+g
    min_len = 10
    max_gap_test = 30

    gaps = []
    errors_foot = []  # foot position error / h_ref
    errors_cx = []    # center x error / h_ref
    errors_cy = []
    errors_h = []     # height error / h_ref
    speeds = []       # actual per-frame speed between the two endpoints
    track_lens = []
    gt_quality = []   # -1=no GT, 0=wrong GT, 1=correct GT (for QC)

    n_tracks = 0
    for seq in SEQUENCES:
        tracks = all_tracks.get(seq, {})
        gt = all_gt.get(seq, {})
        for tid, traj in tracks.items():
            if len(traj) < min_len:
                continue
            n_tracks += 1
            # Map to GT
            gt_id = map_track_to_gt(tid, {tid: traj}, gt)
            has_gt = gt_id > 0

            # For each consecutive pair separated by various gaps
            for i in range(len(traj)):
                for g in range(1, min(max_gap_test + 1, len(traj) - i - 1)):
                    j = i + g + 1
                    if j >= len(traj):
                        break
                    if traj[j][0] - traj[i][0] != g + 1:
                        continue  # need exactly consecutive frame indices

                    f0, cx0, cy0, h0 = traj[i]
                    fj, cxj, cyj, hj = traj[j]

                    # Collect true intermediate positions
                    true_positions = []
                    interp_errors_foot = []
                    interp_errors_cx = []
                    interp_errors_cy = []
                    interp_errors_h = []
                    for t in range(1, g + 1):
                        fk = f0 + t
                        # Find actual frame in between
                        actual = None
                        for entry in traj[i+1:j]:
                            if entry[0] == fk:
                                actual = entry
                                break
                        if actual is None:
                            break  # missing intermediate frame

                        alpha = t / (g + 1)
                        icx = cx0 + alpha * (cxj - cx0)
                        icy = cy0 + alpha * (cyj - cy0)
                        ih = h0 + alpha * (hj - h0)
                        _, acx, acy, ah = actual

                        h_ref = max((h0 + hj) * 0.5, 1.0)
                        # Foot position
                        ifx, ify = _foot(icx, icy, ih)
                        afx, afy = _foot(acx, acy, ah)
                        err_foot = math.hypot(ifx - afx, ify - afy) / h_ref

                        interp_errors_foot.append(err_foot)
                        interp_errors_cx.append(abs(icx - acx) / h_ref)
                        interp_errors_cy.append(abs(icy - acy) / h_ref)
                        interp_errors_h.append(abs(ih - ah) / h_ref)

                    if len(interp_errors_foot) == g:  # all intermediate frames present
                        # Average error across the gap
                        mean_foot = np.mean(interp_errors_foot)
                        mean_cx = np.mean(interp_errors_cx)
                        mean_cy = np.mean(interp_errors_cy)
                        mean_h = np.mean(interp_errors_h)

                        speed = math.hypot(cxj - cx0, cyj - cy0) / (g + 1) / h_ref

                        gaps.append(g)
                        errors_foot.append(mean_foot)
                        errors_cx.append(mean_cx)
                        errors_cy.append(mean_cy)
                        errors_h.append(mean_h)
                        speeds.append(speed)
                        track_lens.append(len(traj))
                        gt_quality.append(1 if has_gt else (-1 if gt_id == 0 else 0))

    gaps = np.array(gaps)
    errors_foot = np.array(errors_foot)
    errors_cx = np.array(errors_cx)
    errors_cy = np.array(errors_cy)
    errors_h = np.array(errors_h)
    speeds = np.array(speeds)
    track_lens = np.array(track_lens)
    gt_quality = np.array(gt_quality)

    n_samples = len(gaps)
    print(f"Tracks analyzed: {n_tracks}")
    print(f"Synthetic gaps tested: {n_samples}")

    # ── Figure 1: Interpolation error vs gap ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1a: Foot error vs gap (boxplot)
    ax = axes[0, 0]
    gap_groups = {}
    for g in range(1, min(21, max_gap_test + 1)):
        mask = gaps == g
        if mask.sum() > 0:
            gap_groups[g] = errors_foot[mask]
    positions = list(gap_groups.keys())
    data = [gap_groups[g] for g in positions]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, medianprops={"color": "red", "lw": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor("#64B5F6")
        patch.set_alpha(0.7)
    # Mean line
    means = [np.mean(gap_groups[g]) for g in positions]
    ax.plot(positions, means, "o-", color="#1565C0", lw=2, markersize=4, label="mean error")
    ax.set_xlabel("Gap (frames)")
    ax.set_ylabel("Foot error (h)")
    ax.set_title("Linear Interpolation Foot Error by Gap")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 1b: Error CDF for selected gaps
    ax = axes[0, 1]
    for g, color, ls in [(1, "#4CAF50", "-"), (5, "#FF9800", "-"), (10, "#2196F3", "-"),
                           (20, "#F44336", "-"), (30, "#9C27B0", "-")]:
        mask = gaps == g
        if mask.sum() < 10:
            continue
        sorted_err = np.sort(errors_foot[mask])
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf, color=color, ls=ls, lw=1.5,
                label=f"gap={g} (n={mask.sum()})")
    ax.set_xlabel("Foot error (h)")
    ax.set_ylabel("CDF")
    ax.set_title("Error CDF by Gap")
    ax.legend(fontsize=7)
    ax.set_xlim(0, np.percentile(errors_foot, 99))
    ax.grid(alpha=0.3)

    # 1c: Scatter: error vs speed, colored by gap
    ax = axes[1, 0]
    scatter = ax.scatter(speeds[::10], errors_foot[::10], c=gaps[::10], s=2, alpha=0.3,
                         cmap="viridis", vmin=1, vmax=20, rasterized=True)
    ax.set_xlabel("Speed (h/f)")
    ax.set_ylabel("Foot error (h)")
    ax.set_title("Interpolation Error vs Speed")
    cbar = plt.colorbar(scatter, ax=ax, label="Gap (frames)", shrink=0.8)
    ax.set_xlim(0, np.percentile(speeds, 99))
    ax.set_ylim(0, np.percentile(errors_foot, 99.5))
    ax.grid(alpha=0.3)

    # 1d: Error components (cx, cy, h) by gap
    ax = axes[1, 1]
    gap_range = range(1, min(21, max_gap_test + 1))
    mean_cx_err = [np.mean(errors_cx[gaps == g]) for g in gap_range]
    mean_cy_err = [np.mean(errors_cy[gaps == g]) for g in gap_range]
    mean_h_err = [np.mean(errors_h[gaps == g]) for g in gap_range]
    ax.plot(gap_range, mean_cx_err, "o-", color="#2196F3", lw=1.5, markersize=3, label="cx error")
    ax.plot(gap_range, mean_cy_err, "s-", color="#4CAF50", lw=1.5, markersize=3, label="cy error")
    ax.plot(gap_range, mean_h_err, "^-", color="#FF9800", lw=1.5, markersize=3, label="height error")
    ax.set_xlabel("Gap (frames)")
    ax.set_ylabel("Mean error (h)")
    ax.set_title("Error Decomposition by Component")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out1 = OUT_DIR / "interp_error_analysis.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    # ── Figure 2: Speed vs error, and error growth ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 2a: Mean error vs gap with error bars
    ax = axes[0]
    gap_range = range(1, min(31, max_gap_test + 1))
    means = [np.mean(errors_foot[gaps == g]) for g in gap_range]
    p25s = [np.percentile(errors_foot[gaps == g], 25) for g in gap_range]
    p75s = [np.percentile(errors_foot[gaps == g], 75) for g in gap_range]
    p90s = [np.percentile(errors_foot[gaps == g], 90) for g in gap_range]
    counts = [int((gaps == g).sum()) for g in gap_range]

    ax.fill_between(gap_range, p25s, p75s, alpha=0.2, color="#2196F3")
    ax.plot(gap_range, p90s, "--", color="#2196F3", lw=1, alpha=0.5, label="p90")
    ax.plot(gap_range, means, "o-", color="#1565C0", lw=2, markersize=4, label="mean")
    # Linear fit
    from numpy.polynomial import polynomial as P
    coef = P.polyfit(list(gap_range), means, 1)
    fit_y = P.polyval(list(gap_range), coef)
    ax.plot(gap_range, fit_y, "--", color="red", lw=1, alpha=0.5,
            label=f"fit: {coef[1]:.4f} h/frame")
    ax.set_xlabel("Gap (frames)")
    ax.set_ylabel("Foot error (h)")
    ax.set_title(f"Mean Interpolation Error (slope={coef[1]:.4f} h/f)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 2b: Error histogram at gap=10 vs gap=20
    ax = axes[1]
    for g, color, alpha in [(5, "#4CAF50", 0.6), (10, "#FF9800", 0.5), (20, "#F44336", 0.4)]:
        mask = gaps == g
        if mask.sum() < 10:
            continue
        ax.hist(np.clip(errors_foot[mask], 0, 1.0), bins=50, alpha=alpha, color=color,
                label=f"gap={g} (n={mask.sum()}, mean={np.mean(errors_foot[mask]):.3f})")
    ax.set_xlabel("Foot error (h)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution at Key Gaps")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1.0)

    # 2c: Error growth vs speed — by gap
    ax = axes[2]
    for g, color in [(1, "#E0E0E0"), (5, "#90CAF9"), (10, "#42A5F5"),
                      (15, "#1E88E5"), (20, "#1565C0")]:
        mask = gaps == g
        if mask.sum() < 100:
            continue
        # Bin by speed
        speed_bins = np.linspace(0, 0.5, 20)
        bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2
        bin_means = []
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            m = mask & (speeds >= lo) & (speeds < hi)
            bin_means.append(np.mean(errors_foot[m]) if m.sum() > 5 else float("nan"))
        valid = ~np.isnan(bin_means)
        if valid.sum() > 1:
            ax.plot(bin_centers[valid], np.array(bin_means)[valid], "o-", color=color,
                    lw=1.5, markersize=3, label=f"gap={g}")
    ax.set_xlabel("Speed (h/f)")
    ax.set_ylabel("Mean foot error (h)")
    ax.set_title("Error by Speed and Gap")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out2 = OUT_DIR / "interp_error_growth.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig)

    # ── Summary statistics ───────────────────────────────────────────────
    print(f"\n{'Gap':>5} {'n':>8} {'mean_err':>10} {'p50':>10} {'p90':>10} {'p95':>10} {'max_err':>10}")
    print("-" * 68)
    for g in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
        mask = gaps == g
        if mask.sum() < 5:
            continue
        e = errors_foot[mask]
        print(f"{g:>5} {mask.sum():>8} {np.mean(e):>10.4f} {np.median(e):>10.4f} "
              f"{np.percentile(e,90):>10.4f} {np.percentile(e,95):>10.4f} {np.max(e):>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
