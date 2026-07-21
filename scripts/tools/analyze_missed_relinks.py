#!/usr/bin/env python3
"""Analyze missed relinks: GT-labeled distributions of all features."""
# status: diagnostic

import numpy as np
from pathlib import Path
from collections import defaultdict


# ── Load MOT output and GT ──
def load_mot_tracks(path: Path):
    """Return {track_id: [(frame, cx, cy, h), ...]}"""
    tracks = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frm, tid, x, y, w, h = (
                int(parts[0]),
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            tracks[tid].append((frm, x + w / 2, y + h / 2, h))
    return dict(tracks)


def load_gt_tracks(path: Path):
    """Return {gt_id: [(frame, cx, cy, h), ...]}"""
    tracks = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frm, gt_id, x, y, w, h = (
                int(parts[0]),
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            if gt_id <= 0:
                continue  # skip ignore regions
            tracks[gt_id].append((frm, x + w / 2, y + h / 2, h))
    return dict(tracks)


def map_track_to_gt(track_traj: list, gt_tracks: dict, iou_thresh=0.3):
    """Map a predicted track to the most-overlapping GT ID by IoU."""
    best_gt, _best_iou, best_overlap = -1, 0, 0
    for gt_id, gt_traj in gt_tracks.items():
        overlap = 0
        pred_frames = {f for f, _, _, _ in track_traj}
        gt_frames = {f for f, _, _, _ in gt_traj}
        common = pred_frames & gt_frames
        if len(common) == 0:
            continue
        for f in common:
            # Find the pred point and GT point at frame f
            pred_pts = [(cx, cy, h) for fr, cx, cy, h in track_traj if fr == f]
            gt_pts = [(cx, cy, h) for fr, cx, cy, h in gt_traj if fr == f]
            if not pred_pts or not gt_pts:
                continue
            pcx, pcy, ph = pred_pts[0]
            gcx, gcy, gh = gt_pts[0]
            # IoU approximation: center distance / avg height
            dist = np.hypot(pcx - gcx, pcy - gcy)
            if dist < max(ph, gh) * 2:  # within 2x height
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_gt = gt_id
    return best_gt if best_overlap >= max(3, len(track_traj) * 0.3) else -1


def find_fragments(tracks, max_gap=120, max_dist_h=5.0):
    """Find close-proximity fragment pairs: [lost_end, cand_start)"""
    fragments = []
    track_ids = list(tracks.keys())
    for i, ta in enumerate(track_ids):
        traj_a = tracks[ta]
        if len(traj_a) < 2:
            continue
        last_f, last_cx, last_cy, last_h = traj_a[-1]
        for tb in track_ids[i + 1 :]:
            traj_b = tracks[tb]
            if len(traj_b) < 2:
                continue
            first_f, first_cx, first_cy, first_h = traj_b[0]
            gap = first_f - last_f
            if gap < 2 or gap > max_gap:
                continue
            dist_h = np.hypot(last_cx - first_cx, last_cy - first_cy) / max(
                (last_h + first_h) / 2, 1
            )
            if dist_h > max_dist_h:
                continue
            # No temporal overlap
            frames_a = {f for f, _, _, _ in traj_a}
            frames_b = {f for f, _, _, _ in traj_b}
            if frames_a & frames_b:
                continue
            fragments.append(
                (gap, dist_h, ta, tb, last_f, first_f, len(traj_a), len(traj_b))
            )
    return fragments


# ── Main ──
seq = "MOT17-02-SDP"
data_root = Path("datasets/MOT17/train")
mot_path = Path("results/MOT17_eval") / f"{seq}.txt"
gt_path = data_root / seq / "gt/gt.txt"
raw_path = data_root / f"{seq}_raw_data.npy"

print(f"Loading GT from {gt_path}")
gt_tracks = load_gt_tracks(gt_path)
print(f"  {len(gt_tracks)} GT tracks")

print(f"Loading MOT output from {mot_path}")
mot_tracks = load_mot_tracks(mot_path)
print(f"  {len(mot_tracks)} predicted tracks")

print(f"Loading raw data from {raw_path}")
raw = np.load(raw_path)
print(f"  {raw.shape[0]} raw attempts")

# Map predicted tracks to GT IDs
print("Mapping pred tracks to GT IDs...")
track_to_gt = {}
for tid, traj in mot_tracks.items():
    gt = map_track_to_gt(traj, gt_tracks)
    if gt >= 0:
        track_to_gt[tid] = gt
print(f"  {len(track_to_gt)}/{len(mot_tracks)} tracks mapped to GT")

# Find close-proximity fragments
fragments = find_fragments(mot_tracks)
print(f"  {len(fragments)} close-proximity fragment pairs")

# Label each fragment pair
labeled = []  # [(gap, dist_h, ta, tb, last_f, first_f, same_gt)]
for gap, dist_h, ta, tb, last_f, first_f, la, lb in fragments:
    gt_a = track_to_gt.get(ta, -1)
    gt_b = track_to_gt.get(tb, -1)
    same_gt = gt_a >= 0 and gt_a == gt_b
    labeled.append((gap, dist_h, ta, tb, last_f, first_f, la, lb, same_gt, gt_a, gt_b))

same = [f for f in labeled if f[8]]
diff = [f for f in labeled if not f[8]]

print("\n=== Fragment Analysis ===")
print(f"  Same GT (missed relink): {len(same)}")
print(f"  Different GT (correct rejection): {len(diff)}")
print(f"  Unmapped (GT unknown): {len(labeled) - len(same) - len(diff)}")

# Raw data analysis: gap, bridge, kalman_d2, dir_cos, speed per outcome
# raw cols: [gap, bridge, kalman_d2, dir_cos, speed, outcome, source]
# outcome: 0=accepted, 1=rej_bridge_px, 2=rej_kalman, 3=rej_direction, 4=rej_speed

# raw cols: [gap, bridge, kalman_d2, dir_cos, speed, outcome, source]
# feature columns are 1,2,3,4 (skip col 0 = gap)
FEAT_COLS = {"bridge": 1, "kalman_d2": 2, "dir_cos": 3, "speed": 4}

print("\n=== Raw Data Distribution ===")
outcome_names = {
    0: "accepted",
    1: "rej_bridge_px",
    2: "rej_kalman",
    3: "rej_direction",
    4: "rej_speed",
}
for oid in range(5):
    mask = raw[:, 5].astype(int) == oid
    n = mask.sum()
    if n == 0:
        continue
    gaps = raw[:, 0][mask]
    print(
        f"\n  {outcome_names[oid]} (n={n}, gap μ={gaps.mean():.1f} σ={gaps.std():.1f}):"
    )
    for fname, col in FEAT_COLS.items():
        vals = raw[:, col][mask]
        if fname == "kalman_d2":
            vals = vals[vals > 0]
        if len(vals) > 0:
            qs = np.percentile(vals, [0, 25, 50, 75, 100])
            print(
                f"    {fname:>12}: μ={vals.mean():.4f} σ={vals.std():.4f} "
                f"p50={qs[2]:.4f} p25={qs[1]:.4f} p75={qs[3]:.4f}"
            )

# Gap-bin distributions
print("\n=== Gap-bin Separation ===")
gap_bins = [(2, 5), (6, 15), (16, 30), (31, 45), (46, 120)]
for lo, hi in gap_bins:
    mask = (raw[:, 0] >= lo) & (raw[:, 0] <= hi)
    sub = raw[mask]
    acc = sub[sub[:, 5].astype(int) == 0]
    rej = sub[sub[:, 5].astype(int) != 0]
    if len(acc) == 0 and len(rej) == 0:
        continue
    print(f"\n  gap {lo:3d}-{hi:3d}:")
    for fname, col in FEAT_COLS.items():
        a_vals = acc[:, col] if len(acc) > 0 else np.array([])
        r_vals = rej[:, col] if len(rej) > 0 else np.array([])
        if fname == "kalman_d2":
            a_vals = a_vals[a_vals > 0] if len(a_vals) > 0 else a_vals
            r_vals = r_vals[r_vals > 0] if len(r_vals) > 0 else r_vals
        if len(a_vals) > 0 and len(r_vals) > 0:
            pooled = np.sqrt((a_vals.var() + r_vals.var()) / 2)
            d = abs(a_vals.mean() - r_vals.mean()) / max(pooled, 1e-9)
            print(
                f"    {fname:>12}: accept μ={a_vals.mean():.4f} σ={a_vals.std():.4f} "
                f"| reject μ={r_vals.mean():.4f} σ={r_vals.std():.4f} "
                f"| d={d:.3f} n_a={len(a_vals)} n_r={len(r_vals)}"
            )

# Same-GT vs Different-GT fragment pair distributions
print(
    f"\n=== Same-GT (missed, n={len(same)}) vs Different-GT (correctly rejected, n={len(diff)}) ==="
)
print(
    f"  {'metric':<15} {'same_mean':>10} {'same_std':>8} {'diff_mean':>10} {'diff_std':>8} {'cohens_d':>8}"
)
for name, same_vals, diff_vals in [
    ("gap", np.array([f[0] for f in same]), np.array([f[0] for f in diff])),
    ("dist_h", np.array([f[1] for f in same]), np.array([f[1] for f in diff])),
    ("lost_len", np.array([f[6] for f in same]), np.array([f[6] for f in diff])),
    ("cand_len", np.array([f[7] for f in same]), np.array([f[7] for f in diff])),
]:
    if len(same_vals) > 0 and len(diff_vals) > 0:
        pooled = np.sqrt((same_vals.var() + diff_vals.var()) / 2)
        d = abs(same_vals.mean() - diff_vals.mean()) / max(pooled, 1e-9)
        print(
            f"  {name:<15} {same_vals.mean():>10.2f} {same_vals.std():>8.2f} "
            f"{diff_vals.mean():>10.2f} {diff_vals.std():>8.2f} {d:>8.3f}"
        )

# ── Long vs Short Track Analysis ──
print(f"\n{'=' * 80}")
print("=== Long vs Short Track Fragment Analysis ===")


def track_len_category(ll):
    if ll < 20:
        return "short(<20)"
    if ll < 100:
        return "medium(20-100)"
    return "long(100+)"


# Split fragments by both track lengths
combo_cats = defaultdict(list)  # {(lost_cat, cand_cat): [(gap, dist_h, ...)]}
for gap, dist_h, ta, tb, last_f, first_f, la, lb, same_gt, gt_a, gt_b in labeled:
    lcat = track_len_category(la)
    ccat = track_len_category(lb)
    combo_cats[(lcat, ccat)].append((gap, dist_h, la, lb, same_gt))

print(
    f"\n{'lost_cat':<18} {'cand_cat':<18} {'pairs':>6} {'same_gt':>8} {'diff_gt':>8} {'same%':>7} {'same_gap':>9} {'diff_gap':>9} {'same_dist':>10} {'diff_dist':>10}"
)
for (lcat, ccat), items in sorted(combo_cats.items()):
    same_items = [x for x in items if x[4]]
    diff_items = [x for x in items if not x[4]]
    same_gap = np.mean([x[0] for x in same_items]) if same_items else 0
    diff_gap = np.mean([x[0] for x in diff_items]) if diff_items else 0
    same_dist = np.mean([x[1] for x in same_items]) if same_items else 0
    diff_dist = np.mean([x[1] for x in diff_items]) if diff_items else 0
    same_pct = len(same_items) / max(len(items), 1) * 100
    print(
        f"{lcat:<18} {ccat:<18} {len(items):>6} {len(same_items):>8} {len(diff_items):>8} {same_pct:>6.1f}% {same_gap:>9.1f} {diff_gap:>9.1f} {same_dist:>10.2f} {diff_dist:>10.2f}"
    )

# Detailed: for long→long and short→short, show distributions
for (lcat, ccat), items in sorted(combo_cats.items()):
    if len(items) < 5:
        continue
    same_items = [x for x in items if x[4]]
    diff_items = [x for x in items if not x[4]]
    if not same_items or not diff_items:
        continue
    print(
        f"\n  {lcat} → {ccat} (n={len(items)}, same={len(same_items)}, diff={len(diff_items)}):"
    )
    for name, idx in [("gap", 0), ("dist_h", 1)]:
        s_vals = np.array([x[idx] for x in same_items])
        d_vals = np.array([x[idx] for x in diff_items])
        pooled = np.sqrt((s_vals.var() + d_vals.var()) / 2)
        d_eff = abs(s_vals.mean() - d_vals.mean()) / max(pooled, 1e-9)
        print(
            f"    {name}: same μ={s_vals.mean():.2f} σ={s_vals.std():.2f} | "
            f"diff μ={d_vals.mean():.2f} σ={d_vals.std():.2f} | d={d_eff:.3f}"
        )

# Raw data: gap distribution for accepted vs all rejected by track length
print("\n=== Raw Data by Gap and Source ===")
sources = {0: "live", 1: "archive"}
for src in [0, 1]:
    src_mask = raw[:, 6].astype(int) == src
    src_data = raw[src_mask]
    if len(src_data) == 0:
        continue
    acc = src_data[src_data[:, 5].astype(int) == 0]
    rej = src_data[src_data[:, 5].astype(int) != 0]
    print(
        f"  {sources[src]}: {len(src_data)} attempts, {len(acc)} accepted, {len(rej)} rejected"
    )
    if len(acc) > 0:
        print(
            f"    accepted: gap μ={acc[:, 0].mean():.1f} σ={acc[:, 0].std():.1f} "
            f"bridge μ={acc[:, 1].mean():.3f} dir_cos μ={acc[:, 3].mean():.3f} "
            f"speed μ={acc[:, 4].mean():.2f}"
        )
    if len(rej) > 0:
        print(
            f"    rejected: gap μ={rej[:, 0].mean():.1f} σ={rej[:, 0].std():.1f} "
            f"bridge μ={rej[:, 1].mean():.3f} dir_cos μ={rej[:, 3].mean():.3f} "
            f"speed μ={rej[:, 4].mean():.2f}"
        )

print("\nDone.")
