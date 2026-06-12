#!/usr/bin/env python3
"""Plot directional signal analysis: where true relinks appear relative to lost-track velocity."""

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


def vel_regression_4(seg):
    if len(seg) < 4:
        return 0.0, 0.0
    x0, y0 = _foot(*seg[-4][1:4])
    x1, y1 = _foot(*seg[-3][1:4])
    x2, y2 = _foot(*seg[-2][1:4])
    x3, y3 = _foot(*seg[-1][1:4])
    return (3.0 * x3 + x2 - x1 - 3.0 * x0) / 10.0, (3.0 * y3 + y2 - y1 - 3.0 * y0) / 10.0


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

    # Collect angle + speed data
    speeds, cos_vals, angles = [], [], []
    y_vals = []
    for row in rows:
        seq = row["seq"]
        lid, cid = int(row["lost_id"]), int(row["cand_id"])
        tracks = all_tracks.get(seq, {})
        if lid not in tracks or cid not in tracks:
            continue
        traj_a, traj_b = tracks[lid], tracks[cid]
        if len(traj_a) < 4 or len(traj_b) < 2:
            continue
        if traj_a[-1][0] >= traj_b[0][0]:
            continue
        if int(row["gt_valid"]) != 1:
            continue

        vx, vy = vel_regression_4(traj_a)
        h_ref = max((traj_a[-1][3] + traj_b[0][3]) * 0.5, 1.0)
        speed_h = math.hypot(vx, vy) / h_ref

        ax, ay = _foot(*traj_a[-1][1:4])
        bx, by = _foot(*traj_b[0][1:4])
        dx, dy = bx - ax, by - ay
        nd = math.hypot(dx, dy)
        nv = math.hypot(vx, vy)
        if nv > 1e-6 and nd > 1e-6:
            cos_v = max(-1.0, min(1.0, (vx * dx + vy * dy) / (nv * nd)))
            angle_v = math.degrees(math.acos(cos_v))
        else:
            cos_v = 0.0
            angle_v = 90.0

        speeds.append(speed_h)
        cos_vals.append(cos_v)
        angles.append(angle_v)
        y_vals.append(int(row["gt_match"]))

    speeds = np.array(speeds)
    cos_vals = np.array(cos_vals)
    angles = np.array(angles)
    y_vals = np.array(y_vals)
    pos_mask = y_vals == 1
    neg_mask = y_vals == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    print(f"Total: pos={n_pos}, neg={n_neg}")

    # ── Figure 1: Polar histogram (rose plot) ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": "polar"})

    bins = 36  # 10° per bin
    theta_bins = np.linspace(0, np.pi, bins + 1)

    for ax_idx, (mask, label, color, title) in enumerate([
        (pos_mask, "True relink", "#4CAF50", f"True Relink (n={n_pos})"),
        (neg_mask, "False", "#F44336", f"False (n={n_neg})"),
    ]):
        ax = axes[ax_idx]
        counts, _ = np.histogram(angles[mask], bins=theta_bins)
        # Symmetrize for full 360°
        counts_full = np.concatenate([counts, counts[::-1]])
        theta_full = np.linspace(0, 2 * np.pi, 2 * bins + 1)
        width = 2 * np.pi / (2 * bins)
        bars = ax.bar(theta_full[:-1], counts_full, width=width, color=color, alpha=0.8,
                      edgecolor="white", linewidth=0.3)

        # Forward direction marker
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                          ["0°(fwd)", "45°", "90°", "135°", "180°(back)", "", "", ""], fontsize=7)
        ax.set_title(title, fontsize=10, pad=15)
        ax.set_ylim(0, max(counts_full) * 1.15)

    # Speed-weighted rose (only speed > 0.02)
    ax = axes[2]
    fast = speeds > 0.02
    fast_pos = pos_mask & fast
    fast_neg = neg_mask & fast
    counts_pos, _ = np.histogram(angles[fast_pos], bins=theta_bins, weights=speeds[fast_pos])
    counts_neg, _ = np.histogram(angles[fast_neg], bins=theta_bins, weights=speeds[fast_neg])
    counts_full_f = np.concatenate([counts_pos, counts_pos[::-1]])
    counts_full_n = np.concatenate([counts_neg, counts_neg[::-1]])
    theta_full = np.linspace(0, 2 * np.pi, 2 * bins + 1)
    width = 2 * np.pi / (2 * bins)
    ax.bar(theta_full[:-1], counts_full_f, width=width, color="#4CAF50", alpha=0.7,
           edgecolor="white", linewidth=0.3, label=f"True (speed>0.02)")
    ax.bar(theta_full[:-1], counts_full_n, width=width, color="#F44336", alpha=0.4,
           edgecolor="white", linewidth=0.3, label=f"False (speed>0.02)")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ["0°(fwd)", "", "90°", "", "180°(back)", "", "", ""], fontsize=7)
    ax.set_title(f"Speed-Weighted (v>0.02 h/f)", fontsize=10, pad=15)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    out_polar = OUT_DIR / "relink_direction_polar.png"
    fig.savefig(out_polar, dpi=150, bbox_inches="tight")
    print(f"Saved {out_polar}")
    plt.close(fig)

    # ── Figure 2: Direction signal by speed (bar + line) ────────────────
    speed_bins = [(0, 0.005), (0.005, 0.01), (0.01, 0.015), (0.015, 0.02),
                  (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.07),
                  (0.07, 0.10), (0.10, 10)]
    labels = ["0–0.005", "0.005–\n0.01", "0.01–\n0.015", "0.015–\n0.02",
              "0.02–\n0.03", "0.03–\n0.04", "0.04–\n0.05", "0.05–\n0.07",
              "0.07–\n0.10", "≥0.10"]

    cos_pos_list, cos_neg_list, cos_delta = [], [], []
    p30_pos_list, p30_neg_list = [], []
    n_pos_list, n_neg_list = [], []
    for lo, hi in speed_bins:
        m = (speeds >= lo) & (speeds < hi)
        n_p = int((m & pos_mask).sum())
        n_n = int((m & neg_mask).sum())
        n_pos_list.append(n_p)
        n_neg_list.append(n_n)
        cp = cos_vals[m & pos_mask].mean() if n_p > 0 else float("nan")
        cn = cos_vals[m & neg_mask].mean() if n_n > 0 else float("nan")
        cos_pos_list.append(cp)
        cos_neg_list.append(cn)
        cos_delta.append((cp - cn) if not (np.isnan(cp) or np.isnan(cn)) else float("nan"))
        p30_pos_list.append((angles[m & pos_mask] < 30).mean() if n_p > 0 else float("nan"))
        p30_neg_list.append((angles[m & neg_mask] < 30).mean() if n_n > 0 else float("nan"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(labels))
    width = 0.3

    valid = np.array([not (np.isnan(cp) or np.isnan(cn)) for cp, cn in zip(cos_pos_list, cos_neg_list)])

    # Top: cos(theta) bar chart
    bars1 = ax1.bar(x[valid] - width/2, np.array(cos_pos_list)[valid], width,
                    label="True relink", color="#4CAF50", edgecolor="white")
    bars2 = ax1.bar(x[valid] + width/2, np.array(cos_neg_list)[valid], width,
                    label="False", color="#F44336", edgecolor="white")
    # Delta line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x[valid], np.array(cos_delta)[valid], "o-", color="#2196F3", lw=2, markersize=8,
                  label="Δ (true − false)")
    ax1_twin.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.3)
    ax1_twin.set_ylabel("Δ cos(θ)", color="#2196F3")
    ax1_twin.tick_params(axis="y", labelcolor="#2196F3")
    ax1_twin.set_ylim(-0.2, 1.0)

    # Annotate sample sizes
    for i in range(len(x)):
        if not valid[i]:
            continue
        ax1.text(i, max(cos_pos_list[i], cos_neg_list[i]) + 0.05,
                 f"n={n_pos_list[i]}", ha="center", fontsize=6, color="#2E7D32")
        ax1.text(i, min(cos_pos_list[i], cos_neg_list[i]) - 0.08,
                 f"n={n_neg_list[i]}", ha="center", fontsize=6, color="#C62828")

    ax1.set_ylabel("Mean cos(θ)  (+1 = forward)")
    ax1.set_title("Directional Signal by Exit Speed")
    ax1.legend(loc="upper left", fontsize=8)
    ax1_twin.legend(loc="upper right", fontsize=8)
    ax1.axhline(0, color="gray", ls="--", lw=0.5)
    ax1.set_ylim(-0.65, 0.6)

    # Bottom: P(angle < 30°) bar chart
    bars3 = ax2.bar(x[valid] - width/2, np.array(p30_pos_list)[valid], width,
                    label="True relink", color="#4CAF50", edgecolor="white")
    bars4 = ax2.bar(x[valid] + width/2, np.array(p30_neg_list)[valid], width,
                    label="False", color="#F44336", edgecolor="white")
    ax2.axhline(0.167, color="gray", ls="--", lw=0.8, alpha=0.5, label="uniform (16.7%)")

    # Ratio annotation
    for i in range(len(x)):
        if not valid[i] or p30_neg_list[i] == 0:
            continue
        ratio = p30_pos_list[i] / p30_neg_list[i] if p30_neg_list[i] > 0 else float("inf")
        if not np.isnan(ratio) and ratio > 1.0:
            ax2.text(i, max(p30_pos_list[i], p30_neg_list[i]) + 0.03,
                     f"{ratio:.1f}×", ha="center", fontsize=7, color="#2E7D32", fontweight="bold")

    ax2.set_ylabel("P(angle < 30°)")
    ax2.set_xlabel("Lost-track exit speed (h/f)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)

    plt.tight_layout()
    out_bar = OUT_DIR / "relink_direction_by_speed.png"
    fig.savefig(out_bar, dpi=150, bbox_inches="tight")
    print(f"Saved {out_bar}")
    plt.close(fig)

    # ── Figure 3: Scatter — angle vs speed, colored by true/false ───────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Background: all false as density
    ax.scatter(speeds[neg_mask], angles[neg_mask], s=1, alpha=0.05, color="gray",
               label=f"False (n={n_neg})", rasterized=True)
    # True relinks highlighted
    ax.scatter(speeds[pos_mask], angles[pos_mask], s=25, alpha=0.9, color="#4CAF50",
               edgecolors="white", linewidth=0.5, label=f"True relink (n={n_pos})",
               zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Lost-track exit speed (h/f)")
    ax.set_ylabel("Angle from velocity direction (°)")
    ax.set_title("True Relink Position vs Lost-track Exit Speed")
    ax.legend(fontsize=9, markerscale=1.5)

    # Annotate zones
    ax.axvline(0.01, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.02, color="orange", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.05, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(30, color="#4CAF50", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(90, color="gray", ls=":", lw=0.5, alpha=0.3)

    # Zone labels
    ax.text(0.003, 175, "noise zone\n(v < 0.01)", fontsize=7, color="gray", ha="center")
    ax.text(0.014, 175, "dead zone\n(0.01–0.02)", fontsize=7, color="orange", ha="center")
    ax.text(0.035, 175, "signal emerges\n(0.02–0.05)", fontsize=7, color="darkorange", ha="center")
    ax.text(0.1, 175, "strong signal\n(v ≥ 0.05)", fontsize=7, color="red", ha="center")

    ax.set_ylim(0, 185)
    ax.set_xlim(0.0005, 1.5)

    plt.tight_layout()
    out_scatter = OUT_DIR / "relink_angle_vs_speed.png"
    fig.savefig(out_scatter, dpi=150, bbox_inches="tight")
    print(f"Saved {out_scatter}")
    plt.close(fig)

    # ── Figure 4: Forward-fraction cumulative by speed ──────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    thresholds = np.logspace(-3, 0, 100)
    frac_pos_30 = []
    frac_neg_30 = []
    frac_pos_45 = []
    n_above = []
    for thr in thresholds:
        m = speeds >= thr
        n_p = int((m & pos_mask).sum())
        n_above.append(n_p)
        frac_pos_30.append((angles[m & pos_mask] < 30).mean() if n_p > 0 else float("nan"))
        frac_neg_30.append((angles[m & neg_mask] < 30).mean() if n_p > 0 else float("nan"))
        frac_pos_45.append((angles[m & pos_mask] < 45).mean() if n_p > 0 else float("nan"))

    valid = np.array([not np.isnan(v) for v in frac_pos_30])
    thr_valid = thresholds[valid]
    f30_pos = np.array(frac_pos_30)[valid]
    f30_neg = np.array(frac_neg_30)[valid]
    f45_pos = np.array(frac_pos_45)[valid]
    n_above = np.array(n_above)[valid]

    ax1.plot(thr_valid, f45_pos, "-", color="#4CAF50", lw=2, label="P(angle < 45°) True")
    ax1.plot(thr_valid, f30_pos, "-", color="#2196F3", lw=2, label="P(angle < 30°) True")
    ax1.plot(thr_valid, f30_neg, "--", color="#F44336", lw=1.5, label="P(angle < 30°) False")
    ax1.set_xscale("log")
    ax1.set_xlabel("Speed threshold (h/f) — keep pairs with speed ≥ threshold")
    ax1.set_ylabel("Forward fraction")
    ax1.set_title("Forward Fraction vs Speed Cutoff")
    ax1.legend(fontsize=8)
    ax1.axvline(0.01, color="gray", ls=":", lw=0.8)
    ax1.axvline(0.02, color="orange", ls=":", lw=0.8)
    ax1.axvline(0.05, color="red", ls=":", lw=0.8)
    ax1.set_ylim(0, 0.7)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(thr_valid, n_above, alpha=0.15, color="#4CAF50")
    ax2.plot(thr_valid, n_above, "-", color="#4CAF50", lw=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Speed threshold (h/f)")
    ax2.set_ylabel("True relink count")
    ax2.set_title("Surviving True Relinks by Speed Cutoff")
    ax2.set_ylim(0, n_pos + 10)
    ax2.axvline(0.01, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax2.axvline(0.02, color="orange", ls=":", lw=0.8, alpha=0.5)
    ax2.axvline(0.05, color="red", ls=":", lw=0.8, alpha=0.5)
    ax2.axhline(n_pos * 0.7, color="gray", ls="--", lw=0.8, alpha=0.3,
                label=f"70% recall ({int(n_pos*0.7)})")
    ax2.axhline(n_pos * 0.3, color="gray", ls="--", lw=0.8, alpha=0.3,
                label=f"30% recall ({int(n_pos*0.3)})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Find key thresholds
    idx_01 = np.argmin(np.abs(thr_valid - 0.01))
    idx_02 = np.argmin(np.abs(thr_valid - 0.02))
    idx_05 = np.argmin(np.abs(thr_valid - 0.05))

    print(f"\nKey thresholds:")
    print(f"  v≥0.01: P<30°={f30_pos[idx_01]:.1%}, P<45°={f45_pos[idx_01]:.1%}, n={n_above[idx_01]} ({n_above[idx_01]/n_pos:.0%})")
    print(f"  v≥0.02: P<30°={f30_pos[idx_02]:.1%}, P<45°={f45_pos[idx_02]:.1%}, n={n_above[idx_02]} ({n_above[idx_02]/n_pos:.0%})")
    print(f"  v≥0.05: P<30°={f30_pos[idx_05]:.1%}, P<45°={f45_pos[idx_05]:.1%}, n={n_above[idx_05]} ({n_above[idx_05]/n_pos:.0%})")

    plt.tight_layout()
    out_cum = OUT_DIR / "relink_direction_cumulative.png"
    fig.savefig(out_cum, dpi=150, bbox_inches="tight")
    print(f"Saved {out_cum}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
