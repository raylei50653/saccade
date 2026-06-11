#!/usr/bin/env python3
"""Plot ROC and precision-recall curves for relink bridge_dist discriminability.

Compares full-pool vs hard-pool (bd≤1), matching the analysis in
docs/modules/semantic/research/offline_relink_candidate_analysis.md.
"""

import csv
import math
import sys
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


def compute_features(traj_a, traj_b, vel_method="regression"):
    la_f, la_cx, la_cy, la_h = traj_a[-1]
    fb_f, fb_cx, fb_cy, fb_h = traj_b[0]
    gap = fb_f - la_f
    h_ref = max((la_h + fb_h) * 0.5, 1.0)
    ax, ay = _foot(la_cx, la_cy, la_h)
    bx, by = _foot(fb_cx, fb_cy, fb_h)

    if vel_method == "regression":
        vax, vay = velocity_regression_4(traj_a)
        vbx, vby = velocity_regression_4_reverse(traj_b)
    else:
        vax, vay = velocity_mean(traj_a, n=4)
        vbx, vby = velocity_mean(traj_b, n=4)

    half = gap * 0.5
    mlx, mly = ax + vax * half, ay + vay * half
    mcx, mcy = bx - vbx * half, by - vby * half
    bridge_dist = math.hypot(mlx - mcx, mly - mcy) / h_ref

    fx, fy = ax + vax * gap, ay + vay * gap
    fwd_resid = math.hypot(fx - bx, fy - by) / h_ref
    rx, ry = bx - vbx * gap, by - vby * gap
    bwd_resid = math.hypot(rx - ax, ry - ay) / h_ref
    sym_fb = 0.5 * (fwd_resid + bwd_resid)

    nd = math.hypot(bx - ax, by - ay)
    dist_h = nd / h_ref

    lost_speed = math.hypot(vax, vay) / h_ref
    cand_speed = math.hypot(vbx, vby) / h_ref

    return {
        "bridge_dist": bridge_dist, "fwd_resid": fwd_resid, "bwd_resid": bwd_resid,
        "sym_fb": sym_fb, "dist_h": dist_h,
        "lost_speed": lost_speed, "cand_speed": cand_speed,
    }


def roc_curve(score, y, points=1000):
    """ROC: TPR vs FPR. Lower score = more positive."""
    order = np.argsort(score)
    ys = y[order]
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return np.array([]), np.array([]), float("nan")
    tpr_curve = np.zeros(points + 1)
    fpr_curve = np.zeros(points + 1)
    for i, thr_idx in enumerate(np.linspace(0, len(ys) - 1, points + 1).astype(int)):
        tp = ys[thr_idx:].sum()
        fp = (len(ys) - thr_idx) - tp
        tpr_curve[i] = tp / pos
        fpr_curve[i] = fp / neg
    try:
        auc_val = np.trapezoid(tpr_curve, fpr_curve)
    except AttributeError:
        auc_val = np.trapz(tpr_curve, fpr_curve)
    return fpr_curve, tpr_curve, auc_val


def pr_curve(score, y, points=1000):
    """Precision-Recall. Lower score = more positive."""
    order = np.argsort(score)
    ys = y[order]
    pos = int(y.sum())
    if pos == 0:
        return np.array([]), np.array([]), float("nan")
    prec_curve = np.zeros(points + 1)
    rec_curve = np.zeros(points + 1)
    for i, thr_idx in enumerate(np.linspace(0, len(ys) - 1, points + 1).astype(int)):
        tp = ys[thr_idx:].sum()
        fp = (len(ys) - thr_idx) - tp
        prec_curve[i] = tp / max(tp + fp, 1)
        rec_curve[i] = tp / pos
    try:
        ap_val = -np.trapezoid(prec_curve, rec_curve)
    except AttributeError:
        ap_val = -np.trapz(prec_curve, rec_curve)
    return rec_curve, prec_curve, ap_val


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

    # Build features
    closed_bd, mean_bd, dist_h_vals, sym_fb_vals = [], [], [], []
    lost_speeds, cand_speeds = [], []
    y_vals = []
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

        fc = compute_features(traj_a, traj_b, "regression")
        fm = compute_features(traj_a, traj_b, "mean")
        closed_bd.append(fc["bridge_dist"])
        mean_bd.append(fm["bridge_dist"])
        dist_h_vals.append(fc["dist_h"])
        sym_fb_vals.append(fc["sym_fb"])
        lost_speeds.append(fc["lost_speed"])
        cand_speeds.append(fc["cand_speed"])
        y_vals.append(int(row["gt_match"]))

    closed_bd = np.array(closed_bd)
    mean_bd = np.array(mean_bd)
    dist_h_vals = np.array(dist_h_vals)
    sym_fb_vals = np.array(sym_fb_vals)
    lost_speeds = np.array(lost_speeds)
    cand_speeds = np.array(cand_speeds)
    y_vals = np.array(y_vals)

    n_pos = int(y_vals.sum())
    n_all = len(y_vals)
    print(f"GT-valid pairs: {n_all} ({n_pos} pos, {n_all - n_pos} neg)")

    # Speed-weighted blend
    min_speed = np.minimum(lost_speeds, cand_speeds)
    w = np.clip(np.sqrt(min_speed / 0.12), 0.0, 1.0)
    blend_score = w * sym_fb_vals + (1.0 - w) * dist_h_vals

    # Hard pool mask
    hard1 = closed_bd <= 1.0
    n_hard1 = int(hard1.sum())
    print(f"Hard pool (bd≤1): {n_hard1} pairs")

    # ── Figure 1: ROC curves (full + hard) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full pool ROC
    ax = axes[0]
    methods = [
        ("bridge_dist (closed-form)", closed_bd, "#2196F3"),
        ("bridge_dist (mean-vel)", mean_bd, "#64B5F6"),
        ("dist_h (spatial only)", dist_h_vals, "#FF9800"),
        (r"blend $w(s) \cdot$sym_fb + $(1-w) \cdot$dist_h", blend_score, "#4CAF50"),
    ]
    for label, score, color in methods:
        fpr, tpr, a = roc_curve(score, y_vals)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — Full Pool ({n_all} pairs, {n_pos} pos)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    # Hard pool ROC (bd≤1)
    ax = axes[1]
    for label, score, color in methods:
        _, _, a_full = roc_curve(score, y_vals)
        fpr, tpr, a_hard = roc_curve(score[hard1], y_vals[hard1])
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{label} (full={a_full:.3f}, hard={a_hard:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — Hard Pool (bd≤1, {n_hard1} pairs)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    plt.tight_layout()
    out_roc = OUT_DIR / "relink_roc_full_vs_hard.png"
    fig.savefig(out_roc, dpi=150, bbox_inches="tight")
    print(f"Saved {out_roc}")
    plt.close(fig)

    # ── Figure 2: Precision-Recall curves ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    base_rate = n_pos / n_all

    ax = axes[0]
    for label, score, color in methods:
        rec, prec, ap = pr_curve(score, y_vals)
        ax.plot(rec, prec, color=color, lw=1.5, label=f"{label} (AP={ap:.3f})")
    ax.axhline(base_rate, color="gray", ls="--", lw=0.8, alpha=0.5,
               label=f"base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — Full Pool ({n_all} pairs)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 0.55)

    ax = axes[1]
    for label, score, color in methods:
        _, _, ap_full = pr_curve(score, y_vals)
        rec, prec, ap_hard = pr_curve(score[hard1], y_vals[hard1])
        ax.plot(rec, prec, color=color, lw=1.5,
                label=f"{label} (full={ap_full:.3f}, hard={ap_hard:.3f})")
    base_rate_hard = y_vals[hard1].sum() / n_hard1
    ax.axhline(base_rate_hard, color="gray", ls="--", lw=0.8, alpha=0.5,
               label=f"base rate ({base_rate_hard:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — Hard Pool (bd≤1, {n_hard1} pairs)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 0.55)

    plt.tight_layout()
    out_pr = OUT_DIR / "relink_pr_full_vs_hard.png"
    fig.savefig(out_pr, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pr}")
    plt.close(fig)

    # ── Figure 3: AUC by gap bin ────────────────────────────────────────
    gaps = np.array([int(r["gap"]) for r in rows if r["seq"] in all_tracks])
    gap_bins = [(1, 10), (11, 30), (31, 60), (61, 150), (151, 300)]
    gap_labels = [f"{a}-{b}" for a, b in gap_bins]

    fig, ax = plt.subplots(figsize=(8, 4))
    # Rebuild aligned features and gaps in one pass
    aligned_gaps = []
    pair_idx = 0
    for row in rows:
        seq = row["seq"]
        if seq not in all_tracks:
            continue
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
        aligned_gaps.append(int(row["gap"]))

    aligned_gaps = np.array(aligned_gaps)
    assert len(aligned_gaps) == len(closed_bd), f"gap {len(aligned_gaps)} != bd {len(closed_bd)}"

    auc_by_gap = {}
    for gmin, gmax in gap_bins:
        mask = (aligned_gaps >= gmin) & (aligned_gaps <= gmax)
        n_p = int(y_vals[mask].sum())
        if n_p == 0 or (len(y_vals[mask]) - n_p) == 0:
            a = float("nan")
        else:
            a = roc_curve(closed_bd[mask], y_vals[mask])[2]
        auc_by_gap[f"{gmin}-{gmax}"] = (a, n_p)

    xs = range(len(gap_bins))
    vals = [auc_by_gap[l][0] for l in gap_labels]
    counts = [auc_by_gap[l][1] for l in gap_labels]
    bars = ax.bar(xs, vals, color="#2196F3", edgecolor="white")
    for i, (v, c) in enumerate(zip(vals, counts)):
        ax.text(i, v + 0.01, f"{v:.3f}\n(n={c})", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(gap_labels)
    ax.set_ylabel("AUC")
    ax.set_title("bridge_dist AUC by Gap Bin (closed-form regression)")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="chance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_gap = OUT_DIR / "relink_auc_by_gap.png"
    fig.savefig(out_gap, dpi=150, bbox_inches="tight")
    print(f"Saved {out_gap}")
    plt.close(fig)

    # ── Figure 4: AUC by speed (velocity contribution analysis) ─────────
    speed_bins = [(0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 10)]
    speed_labels = ["<0.01", "0.01–0.02", "0.02–0.05", "≥0.05"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(speed_bins))
    width = 0.35
    bridge_aucs, dist_aucs, deltas = [], [], []
    for lo, hi in speed_bins:
        mask = (min_speed >= lo) & (min_speed < hi)
        n_s = int(mask.sum())
        if n_s < 2:
            bridge_aucs.append(0)
            dist_aucs.append(0)
            deltas.append(0)
            continue
        a_b = roc_curve(closed_bd[mask], y_vals[mask])[2]
        a_d = roc_curve(dist_h_vals[mask], y_vals[mask])[2]
        bridge_aucs.append(a_b)
        dist_aucs.append(a_d)
        deltas.append(a_b - a_d)

    bars1 = ax.bar(x_pos - width/2, bridge_aucs, width, label="bridge_dist", color="#2196F3")
    bars2 = ax.bar(x_pos + width/2, dist_aucs, width, label="dist_h (spatial)", color="#FF9800")
    for i, d in enumerate(deltas):
        y_max = max(bridge_aucs[i], dist_aucs[i])
        ax.text(i, y_max + 0.02, f"Δ={d:+.3f}", ha="center", fontsize=8, color="red" if d < 0 else "green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(speed_labels)
    ax.set_ylabel("AUC")
    ax.set_title("AUC by Speed bin — velocity contribution Δ = bridge − dist_h")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out_speed = OUT_DIR / "relink_auc_by_speed.png"
    fig.savefig(out_speed, dpi=150, bbox_inches="tight")
    print(f"Saved {out_speed}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
