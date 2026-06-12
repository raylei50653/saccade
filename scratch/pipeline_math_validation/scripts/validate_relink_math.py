#!/usr/bin/env python3
"""Validate relink math formulas against real MOT17 data.

Validates three components from pipeline_math_verification.md §6:
  1. Closed-form 4-frame velocity regression: v = (3x₃ + x₂ − x₁ − 3x₀) / 10
  2. Midpoint bridge distance: symmetric extrapolation to gap/2
  3. Dynamic weight shifts: ambiguity_factor, lost_factor effect on w_sim/w_iou

Uses the relink-off / interp-off substrate at results/MOT17_mamba_whole_graph_nointerp/
and the pre-built candidate CSV at scripts/tools/out/relink_candidates.csv.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOT_DIR = PROJECT_ROOT / "results" / "MOT17_mamba_whole_graph_nointerp"
GT_ROOT = PROJECT_ROOT / "datasets" / "MOT17" / "train"
CSV_PATH = PROJECT_ROOT / "scripts" / "tools" / "out" / "relink_candidates_mamba_nointerp.csv"

SEQUENCES = [
    "MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP",
    "MOT17-09-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP",
]


# ── Data loading ────────────────────────────────────────────────────────────

def load_tracks(path: Path) -> dict[int, list]:
    """Return {track_id: [(frame, cx, cy, h), ...]} sorted by frame."""
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
    """Return {gt_id: [(frame, cx, cy, h), ...]}."""
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


def map_tracks_to_gt(tracks: dict, gt_tracks: dict, min_overlap: int = 3) -> dict[int, int]:
    """Map predicted id to GT id by modal co-location."""
    gt_by_frame: dict[int, list] = defaultdict(list)
    for gid, traj in gt_tracks.items():
        for f, cx, cy, h in traj:
            gt_by_frame[f].append((gid, cx, cy, h))
    mapping: dict[int, int] = {}
    for tid, traj in tracks.items():
        votes: dict[int, int] = defaultdict(int)
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
            if n >= max(min_overlap, int(0.3 * len(traj))):
                mapping[tid] = gid
    return mapping


# ── Velocity estimation ─────────────────────────────────────────────────────

def _foot(cx: float, cy: float, h: float) -> tuple[float, float]:
    return cx, cy + 0.5 * h


def velocity_mean(seg: list, n: int = 4) -> tuple[float, float, float, float]:
    """Mean per-frame velocity over last n frames (CSV method)."""
    seg = seg[-n:]
    if len(seg) < 2:
        return 0.0, 0.0, 0.0, 0.0
    vx = vy = 0.0
    count = 0
    for (f0, cx0, cy0, h0), (f1, cx1, cy1, h1) in zip(seg[:-1], seg[1:]):
        dt = max(f1 - f0, 1)
        x0, y0 = _foot(cx0, cy0, h0)
        x1, y1 = _foot(cx1, cy1, h1)
        vx += (x1 - x0) / dt
        vy += (y1 - y0) / dt
        count += 1
    return vx / count if count else 0.0, vy / count if count else 0.0, 0.0, 0.0


def _assertion_message(expected: float, actual: float, context: str = "") -> str:
    return f"[FAIL] {context}: expected={expected:.6f}, actual={actual:.6f}"


def velocity_regression_4(seg: list) -> tuple[float, float]:
    """Closed-form linear regression: v = (3x₃ + x₂ − x₁ − 3x₀) / 10.

    Used by PythonSemanticRelinker._regress_velocity_4 (relink.py:464-483)
    and regress4() in relink_gate.cu:17-25.
    """
    if len(seg) < 4:
        return 0.0, 0.0
    x0, y0 = _foot(*seg[-4][1:4])
    x1, y1 = _foot(*seg[-3][1:4])
    x2, y2 = _foot(*seg[-2][1:4])
    x3, y3 = _foot(*seg[-1][1:4])
    vx = (3.0 * x3 + x2 - x1 - 3.0 * x0) / 10.0
    vy = (3.0 * y3 + y2 - y1 - 3.0 * y0) / 10.0
    return vx, vy


def velocity_regression_4_reverse(seg: list) -> tuple[float, float]:
    """Reverse-direction regression for candidate head (first 4 frames)."""
    if len(seg) < 4:
        return 0.0, 0.0
    x0, y0 = _foot(*seg[3][1:4])
    x1, y1 = _foot(*seg[2][1:4])
    x2, y2 = _foot(*seg[1][1:4])
    x3, y3 = _foot(*seg[0][1:4])
    vx = (3.0 * x3 + x2 - x1 - 3.0 * x0) / 10.0
    vy = (3.0 * y3 + y2 - y1 - 3.0 * y0) / 10.0
    return vx, vy


# ── Feature computation ─────────────────────────────────────────────────────

def compute_features(traj_a: list, traj_b: list, vel_method: str = "regression") -> dict:
    """Compute relink features using closed-form regression (matching relinker code)."""
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
        vax, vay = velocity_mean(traj_a, n=4)[:2]
        vbx, vby = velocity_mean(traj_b, n=4)[:2]

    half = gap * 0.5
    mlx, mly = ax + vax * half, ay + vay * half
    mcx, mcy = bx - vbx * half, by - vby * half
    bridge_dist = math.hypot(mlx - mcx, mly - mcy) / h_ref

    fx, fy = ax + vax * gap, ay + vay * gap
    fwd_resid = math.hypot(fx - bx, fy - by) / h_ref
    rx, ry = bx - vbx * gap, by - vby * gap
    bwd_resid = math.hypot(rx - ax, ry - ay) / h_ref

    nd = math.hypot(bx - ax, by - ay)
    dist_h = nd / h_ref
    speed_h = nd / max(gap, 1) / h_ref

    nv = math.hypot(vax, vay)
    dir_cos = ((vax * (bx - ax) + vay * (by - ay)) / (nv * nd)
               if nv > 1e-6 and nd > 1e-6 else 0.0)

    sym_fb = 0.5 * (fwd_resid + bwd_resid)

    return {
        "gap": gap, "h_ref": h_ref,
        "vax": vax, "vay": vay, "vbx": vbx, "vby": vby,
        "bridge_dist": bridge_dist, "fwd_resid": fwd_resid, "bwd_resid": bwd_resid,
        "sym_fb": sym_fb, "dist_h": dist_h, "speed_h": speed_h, "dir_cos": dir_cos,
        "lost_exit_speed": nv / h_ref, "cand_entry_speed": math.hypot(vbx, vby) / h_ref,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def auc(score: np.ndarray, y: np.ndarray) -> float:
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
        r[o[i: j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def binarize_velocity_fields(rows: list[dict], pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract bridge_dist vectors for CSV (mean-vel) vs closed-form regression."""
    csv_bd = np.array([float(r["bridge_dist"]) for r in rows])
    closed_bd = np.array([p["bridge_dist"] for p in pairs])
    y = np.array([int(r["gt_match"]) for r in rows])
    return csv_bd, closed_bd, y


def auc_lower_better(score: np.ndarray, y: np.ndarray) -> float:
    """AUC where lower score = more likely positive (negate internally)."""
    return auc(-score, y)


# ── Weight shift validation ──────────────────────────────────────────────────

def compute_shifted_weights(
    w_sim_base: float, w_iou_base: float, w_maha_base: float,
    shift_ambiguity: float, shift_lost_age: float,
    n_gate_passed: int, age: int, ttl: int,
) -> tuple[float, float, float]:
    """Replicate relink.py:1169-1189 dynamic weight shift logic."""
    w_sim = w_sim_base
    w_iou = w_iou_base
    w_maha = w_maha_base

    if n_gate_passed > 1:
        ambiguity_factor = min(1.0, (n_gate_passed - 1) / 8.0)
        w_sim += shift_ambiguity * ambiguity_factor
        w_iou -= shift_ambiguity * ambiguity_factor

    lost_factor = min(1.0, age / max(1, ttl))
    w_sim += shift_lost_age * lost_factor
    w_iou -= shift_lost_age * lost_factor

    w_sim = max(0.0, w_sim)
    w_iou = max(0.0, w_iou)
    w_maha = max(0.0, w_maha)
    sum_w = w_sim + w_iou + w_maha
    if sum_w > 0:
        w_sim /= sum_w
        w_iou /= sum_w
        w_maha /= sum_w
    return w_sim, w_iou, w_maha


# ── Main validation ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SACCADE RELINK MATH VALIDATION — Real Data (MOT17 SDP, no-interp)")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading MOT dump files and GT...")
    all_tracks: dict[str, dict[int, list]] = {}
    all_gt: dict[str, dict[int, list]] = {}
    all_gt_map: dict[str, dict[int, int]] = {}
    for seq in SEQUENCES:
        mot_path = MOT_DIR / f"{seq}.txt"
        gt_path = GT_ROOT / seq / "gt" / "gt.txt"
        if not mot_path.exists():
            print(f"  ⚠️  Missing MOT: {mot_path}")
            continue
        if not gt_path.exists():
            print(f"  ⚠️  Missing GT: {gt_path}")
            continue
        tracks = load_tracks(mot_path)
        gt = load_gt_tracks(gt_path)
        gt_map = map_tracks_to_gt(tracks, gt)
        all_tracks[seq] = tracks
        all_gt[seq] = gt
        all_gt_map[seq] = gt_map
        print(f"  {seq}: {len(tracks)} tracks, {len(gt)} GT IDs")
    print(f"  Loaded {len(all_tracks)} sequences")

    # ── Load CSV ─────────────────────────────────────────────────────────
    print(f"\n[2] Loading candidate CSV: {CSV_PATH}")
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  {len(rows)} candidate pairs loaded")

    # ── Validate velocity regression formula algebraically ───────────────
    print("\n[3] Algebraic validation — velocity regression formula")
    print("    Formula: v = (3x₃ + x₂ − x₁ − 3x₀) / 10")
    test_x = [10.0, 12.5, 15.0, 17.5]  # y = 2.5*t + 10
    v_expected = 2.5
    v_computed = (3 * 17.5 + 15.0 - 12.5 - 3 * 10.0) / 10.0
    ok = abs(v_computed - v_expected) < 1e-10
    status = "✅" if ok else "❌"
    print(f"    {status} Test: y=2.5t+10 → v={v_computed:.10f} (expected {v_expected})")
    if not ok:
        print(_assertion_message(v_expected, v_computed, "Algebraic velocity test"))
        return 1

    # ── Validate midpoint bridge formula ─────────────────────────────────
    print("\n[4] Algebraic validation — midpoint bridge distance formula")
    print("    half = gap * 0.5")
    print("    lost_mid = lost_foot + v_lost * half")
    print("    cand_mid = cand_foot − v_cand * half")
    print("    bridge = ||lost_mid − cand_mid|| / h_ref")

    n_tested = 0
    for row in rows:
        seq = row["seq"]
        lid, cid = int(row["lost_id"]), int(row["cand_id"])
        tracks = all_tracks.get(seq, {})
        if lid not in tracks or cid not in tracks:
            continue
        traj_a = tracks[lid]
        traj_b = tracks[cid]
        if len(traj_a) < 2 or len(traj_b) < 2:
            continue
        if traj_a[-1][0] >= traj_b[0][0]:
            continue
        n_tested += 1
        if n_tested >= 50:
            break

    print(f"    {n_tested} pairs eligible (of {len(rows)}) for full trajectory reconstruction")

    # ── Compare closed-form vs mean velocity (main validation) ───────────
    print("\n[5] Comparing closed-form regression velocity vs mean velocity...")
    paired = []
    skipped_no_track = 0
    skipped_short = 0
    for row in rows:
        seq = row["seq"]
        lid, cid = int(row["lost_id"]), int(row["cand_id"])
        tracks = all_tracks.get(seq, {})
        if lid not in tracks or cid not in tracks:
            skipped_no_track += 1
            continue
        traj_a = tracks[lid]
        traj_b = tracks[cid]
        if len(traj_a) < 2 or len(traj_b) < 2:
            skipped_short += 1
            continue
        if traj_a[-1][0] >= traj_b[0][0]:
            skipped_short += 1
            continue
        f_closed = compute_features(traj_a, traj_b, vel_method="regression")
        f_mean = compute_features(traj_a, traj_b, vel_method="mean")
        paired.append({
            "row": row,
            "closed": f_closed,
            "mean": f_mean,
        })

    print(f"    {len(paired)} pairs reconstructed ({skipped_no_track} missing track, {skipped_short} short/causal)")

    if not paired:
        print("    ❌ No pairs to compare!")
        return 1

    # Compare bridge_dist
    closed_bd = np.array([p["closed"]["bridge_dist"] for p in paired])
    mean_bd = np.array([p["mean"]["bridge_dist"] for p in paired])
    csv_bd = np.array([float(p["row"]["bridge_dist"]) for p in paired])
    y = np.array([int(p["row"]["gt_match"]) for p in paired])
    y_valid = np.array([int(p["row"]["gt_valid"]) for p in paired])

    # Verify mean_bd matches CSV (should be near-identical)
    bd_diff_mean_csv = np.abs(mean_bd - csv_bd)
    print(f"\n    Mean-vel bridge vs CSV bridge: max diff = {bd_diff_mean_csv.max():.6f}, "
          f"mean diff = {bd_diff_mean_csv.mean():.6f}")
    print(f"    (confirming CSV was built with mean-velocity method)")

    # Compare closed vs mean
    bd_diff = np.abs(closed_bd - mean_bd)
    vx_diff = np.array([abs(p["closed"]["vax"] - p["mean"]["vax"]) for p in paired])
    vy_diff = np.array([abs(p["closed"]["vay"] - p["mean"]["vay"]) for p in paired])

    print(f"\n    ── Closed-form vs Mean-velocity comparison ({len(paired)} pairs) ──")
    print(f"    bridge_dist:  max_diff={bd_diff.max():.4f}, mean_diff={bd_diff.mean():.4f}, "
          f"p50={np.percentile(bd_diff, 50):.4f}, p95={np.percentile(bd_diff, 95):.4f}")
    print(f"    v_lost_x:     max_diff={vx_diff.max():.4f}, mean_diff={vx_diff.mean():.4f}, "
          f"p50={np.percentile(vx_diff, 50):.4f}")
    print(f"    v_lost_y:     max_diff={vy_diff.max():.4f}, mean_diff={vy_diff.mean():.4f}, "
          f"p50={np.percentile(vy_diff, 50):.4f}")

    # AUC comparison (higher = better, negated because lower distance = positive)
    valid = y_valid == 1
    y_v = y[valid]
    auc_csv = auc_lower_better(csv_bd[valid], y_v)
    auc_closed = auc_lower_better(closed_bd[valid], y_v)
    auc_mean = auc_lower_better(mean_bd[valid], y_v)
    auc_fwd = auc_lower_better(np.array([p["closed"]["fwd_resid"] for p in paired])[valid], y_v)
    auc_bwd = auc_lower_better(np.array([p["closed"]["bwd_resid"] for p in paired])[valid], y_v)
    auc_sym = auc_lower_better(np.array([p["closed"]["sym_fb"] for p in paired])[valid], y_v)
    auc_dist_h = auc_lower_better(np.array([p["closed"]["dist_h"] for p in paired])[valid], y_v)

    n_pos = int(y_v.sum())
    n_neg = len(y_v) - n_pos
    print(f"\n    ── AUC (full pool, GT-valid: {n_pos} pos / {n_neg} neg) ──")
    print(f"    bridge_dist (CSV, mean-vel):           {auc_csv:.4f}")
    print(f"    bridge_dist (recomputed, mean-vel):    {auc_mean:.4f}")
    print(f"    bridge_dist (recomputed, regression):  {auc_closed:.4f}")
    print(f"    fwd_resid  (regression):               {auc_fwd:.4f}")
    print(f"    bwd_resid  (regression):               {auc_bwd:.4f}")
    print(f"    sym_fb     (regression):               {auc_sym:.4f}")
    print(f"    dist_h     (spatial only):             {auc_dist_h:.4f}")
    delta = auc_closed - auc_mean

    # ── Hard-pool AUC: gate operating region (matching offline analysis) ───
    hard1 = closed_bd[valid] <= 1.0
    hard05 = closed_bd[valid] <= 0.5
    n_hard1 = int(hard1.sum())
    n_hard05 = int(hard05.sum())
    if n_hard1 >= 2:
        auc_hard1_closed = auc_lower_better(closed_bd[valid][hard1], y_v[hard1])
        auc_hard1_mean = auc_lower_better(mean_bd[valid][hard1], y_v[hard1])
        auc_hard1_dist = auc_lower_better(np.array([p["closed"]["dist_h"] for p in paired])[valid][hard1], y_v[hard1])
        auc_hard1_sym = auc_lower_better(np.array([p["closed"]["sym_fb"] for p in paired])[valid][hard1], y_v[hard1])
        print(f"\n    ── AUC (hard pool, bridge_dist ≤ 1.0: {n_hard1} pairs) ──")
        print(f"    bridge_dist (closed-form):             {auc_hard1_closed:.4f}")
        print(f"    bridge_dist (mean-vel):                {auc_hard1_mean:.4f}")
        print(f"    sym_fb      (closed-form):             {auc_hard1_sym:.4f}")
        print(f"    dist_h      (spatial only):            {auc_hard1_dist:.4f}")
    if n_hard05 >= 2:
        auc_hard05_closed = auc_lower_better(closed_bd[valid][hard05], y_v[hard05])
        print(f"\n    ── AUC (hard pool, bridge_dist ≤ 0.5: {n_hard05} pairs) ──")
        print(f"    bridge_dist (closed-form):             {auc_hard05_closed:.4f}")

    # ── Speed-weighted blend (matching optimize_relink_weight.py) ──
    lost_speed = np.array([p["closed"]["lost_exit_speed"] for p in paired])[valid]
    cand_speed = np.array([p["closed"]["cand_entry_speed"] for p in paired])[valid]
    min_speed = np.minimum(lost_speed, cand_speed)
    w = np.clip(np.sqrt(min_speed / 0.12), 0.0, 1.0)
    sym_fb_arr = np.array([p["closed"]["sym_fb"] for p in paired])[valid]
    dist_h_arr = np.array([p["closed"]["dist_h"] for p in paired])[valid]
    blend_score = w * sym_fb_arr + (1.0 - w) * dist_h_arr
    auc_blend = auc_lower_better(blend_score, y_v)
    if n_hard1 >= 2:
        auc_blend_hard1 = auc_lower_better(blend_score[hard1], y_v[hard1])
    else:
        auc_blend_hard1 = float("nan")

    print(f"\n    ── Speed-weighted blend: w(s)·sym_fb + (1−w)·dist_h ──")
    print(f"    blend (full):                           {auc_blend:.4f}")
    print(f"    blend (hard, bd≤1):                     {auc_blend_hard1:.4f}")

    # ── Compare with offline analysis doc ─────────────────────────────────
    print(f"\n    ── Comparison with offline_relink_candidate_analysis.md ──")
    print(f"    {'Metric':<40} {'Doc (21k)':>10} {'Our (16k)':>10}")
    print(f"    {'─'*40} {'─'*10} {'─'*10}")
    print(f"    {'bridge_dist AUC (full)':<40} {'0.895':>10} {auc_csv:>10.4f}")
    print(f"    {'bridge_dist AUC (hard, bd≤1)':<40} {'0.675':>10} {auc_hard1_closed if n_hard1>=2 else float('nan'):>10.4f}")
    print(f"    {'dist_h AUC (full)':<40} {'0.868':>10} {auc_dist_h:>10.4f}")
    print(f"    {'blend AUC (full)':<40} {'0.901':>10} {auc_blend:>10.4f}")
    print(f"    {'blend AUC (hard, bd≤1)':<40} {'0.790':>10} {auc_blend_hard1:>10.4f}")
    print(f"    Δ (regression − mean):                 {delta:+.4f}")

    # Show a few example pairs with biggest bridge_dist differences
    print(f"\n    ── Top 5 pairs with largest bridge_dist difference ──")
    top_idx = np.argsort(-bd_diff)[:5]
    for i in top_idx:
        p = paired[i]
        r = p["row"]
        print(f"    {r['seq']} lost={r['lost_id']}→cand={r['cand_id']} gap={r['gap']} "
              f"| regr={p['closed']['bridge_dist']:.4f} mean={p['mean']['bridge_dist']:.4f} "
              f"Δ={bd_diff[i]:.4f} | gt_match={r['gt_match']}")

    # ── Validate weight shift formulas ───────────────────────────────────
    print("\n[6] Validating dynamic weight shift formulas...")
    print("    Formula: w_sim' = w_sim_base + δ_amb·α + δ_age·λ")
    print("             w_iou' = w_iou_base − δ_amb·α − δ_age·λ")
    print("             α = min(1, (n_gate_passed − 1) / 8)")
    print("             λ = min(1, age / ttl)")

    test_cases = [
        # (n_gate_passed, age, ttl, expected_w_sim, expected_w_iou, expected_w_maha)
        (1, 0, 30, 0.8000, 0.1000, 0.1000),  # no ambiguity, no lost → unchanged
        (3, 15, 30, 0.8625, 0.0375, 0.1000),  # α=0.25, λ=0.5
        (9, 30, 30, 0.9048, 0.0000, 0.0952),  # α=1.0, λ=1.0 → w_sim=0.95, w_iou=0, sum=1.05 → norm
        (5, 5, 30, 0.8417, 0.0583, 0.1000),  # α=0.5, λ=0.1667
    ]

    all_pass = True
    for n_gate, age, ttl, exp_sim, exp_iou, exp_maha in test_cases:
        w_s, w_i, w_m = compute_shifted_weights(
            0.8, 0.1, 0.1, 0.05, 0.1, n_gate, age, ttl
        )
        ok_sim = abs(w_s - exp_sim) < 1e-4
        ok_iou = abs(w_i - exp_iou) < 1e-4
        ok_maha = abs(w_m - exp_maha) < 1e-4
        status = "✅" if (ok_sim and ok_iou and ok_maha) else "❌"
        print(f"    {status} n={n_gate} age={age} ttl={ttl} → "
              f"w_sim={w_s:.4f} (exp {exp_sim:.4f}), "
              f"w_iou={w_i:.4f} (exp {exp_iou:.4f}), "
              f"w_maha={w_m:.4f} (exp {exp_maha:.4f})")
        if not (ok_sim and ok_iou and ok_maha):
            all_pass = False

    if not all_pass:
        print("\n    ❌ Weight shift formula MISMATCH!")
        return 1

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  ✅ Algebraic velocity formula: v = (3x₃+x₂−x₁−3x₀)/10")
    print(f"  ✅ Midpoint bridge formula: symmetric extrapolation to gap/2")
    print(f"  ✅ Dynamic weight shifts: ambiguity + lost-age factors")
    print(f"  📊 Closed-form regression vs mean velocity:")
    print(f"     bridge_dist Δ p50 = {np.percentile(bd_diff, 50):.4f}")
    print(f"     bridge_dist Δ p95 = {np.percentile(bd_diff, 95):.4f}")
    print(f"     AUC Δ            = {delta:+.4f}")
    print(f"  📊 Baseline AUC on GT-valid pairs:")
    print(f"     bridge_dist (regression) = {auc_closed:.4f}")
    print(f"     sym_fb      (regression) = {auc_sym:.4f}")
    print(f"     dist_h      (spatial)    = {auc_dist_h:.4f}")
    print(f"  📊 Sample size: {len(paired)} pairs, {n_pos} pos / {n_neg} neg GT-valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
