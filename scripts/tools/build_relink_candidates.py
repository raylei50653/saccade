#!/usr/bin/env python3
"""Build a relink-candidate dataset from a no-relink / no-interp MOT dump.

Consumes the per-frame ``results/MOT17_eval/{seq}.txt`` produced by a
relink-OFF, interpolate-OFF run (every track death/birth is raw) plus the
matching GT, and enumerates every *plausible* (lost -> candidate) pair under a
small ruleset, labels each pair with GT correctness, and tags which pairs
survive the "an ID cannot be linked twice" constraint.

The output is a flat per-candidate table (one row per pair) — the substrate for
measuring the discriminability of any relink strategy (ROC/AUC over the feature
columns, split by ``gt_match``) without being limited to whatever the live GPU
bridge kernel happened to log.

Ruleset (per lost track A, candidate track B):
  1. causality   : B.first_frame > A.last_frame        (only link IDs born after A is lost)
  2. no overlap  : frames(A) ∩ frames(B) == ∅           (co-existing => different people)
  3. gap gate    : 1 <= gap <= --max-gap                (wide by default)
  4. spatial gate: foot-dist / mean-height <= --spatial-gate (0 = disabled, default off)

Uniqueness ("an ID cannot be linked twice") is NOT a hard filter — it is a
greedy tag: pairs are visited in priority order (smallest foot-bridge distance
first, mirroring relink_bidir_propose_kernel) and once a lost id or a cand id is
consumed, later pairs touching it are tagged already_linked=True / accepted=False.

Usage:
  .venv/bin/python scripts/tools/build_relink_candidates.py \
      --mot-dir results/MOT17_eval --gt-root datasets/MOT17/train \
      --out scripts/tools/out/relink_candidates
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


# ── loaders ───────────────────────────────────────────────────────────────────
def load_tracks(path: Path, drop_nonpos_id: bool = False):
    """Return {id: [(frame, cx, cy, h), ...]} sorted by frame."""
    tracks: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            if drop_nonpos_id and tid <= 0:
                continue
            tracks[tid].append((frm, x + w / 2.0, y + h / 2.0, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


# ── GT mapping ────────────────────────────────────────────────────────────────
def map_tracks_to_gt(tracks: dict, gt_tracks: dict, min_overlap: int = 3):
    """Map each predicted id to the GT id it co-locates with most (center-dist)."""
    # index GT by frame: {frame: [(gid, cx, cy, h), ...]}
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
                d = np.hypot(cx - gx, cy - gy)
                if d < max(h, gh) * 1.0 and d < best_d:
                    best_d, best_gid = d, gid
            if best_gid >= 0:
                votes[best_gid] += 1
        if votes:
            gid, n = max(votes.items(), key=lambda kv: kv[1])
            if n >= max(min_overlap, int(0.3 * len(traj))):
                mapping[tid] = gid
    return mapping


# ── geometry helpers ──────────────────────────────────────────────────────────
def _foot(cx, cy, h):
    return cx, cy + 0.5 * h


def _velocity(seg):
    """Mean per-frame velocity of foot points over a trajectory segment."""
    if len(seg) < 2:
        return 0.0, 0.0
    vx = vy = 0.0
    n = 0
    for (f0, cx0, cy0, h0), (f1, cx1, cy1, h1) in zip(seg[:-1], seg[1:]):
        dt = max(f1 - f0, 1)
        x0, y0 = _foot(cx0, cy0, h0)
        x1, y1 = _foot(cx1, cy1, h1)
        vx += (x1 - x0) / dt
        vy += (y1 - y0) / dt
        n += 1
    return (vx / n, vy / n) if n else (0.0, 0.0)


def _vel_ols(traj, n: int, forward: bool = True) -> tuple[float, float]:
    """OLS velocity from the last/first `n` frames of a trajectory.

    Uses actual frame numbers (handles detection gaps correctly).
    Returns (vx, vy) in px/frame pointing in the direction of motion.
    Falls back to _velocity on < 2 usable frames.
    """
    seg = traj[-n:] if forward else traj[:n]
    if len(seg) < 2:
        return _velocity(traj[-4:] if forward else traj[:4])
    frames = np.array([r[0] for r in seg], dtype=np.float64)
    fps = np.array([_foot(r[1], r[2], r[3]) for r in seg], dtype=np.float64)
    t_mean = frames.mean()
    Stt = ((frames - t_mean) ** 2).sum()
    if Stt < 1e-9:
        return 0.0, 0.0
    vx = float(((frames - t_mean) * fps[:, 0]).sum() / Stt)
    vy = float(((frames - t_mean) * fps[:, 1]).sum() / Stt)
    return vx, vy


def _vel_long_baseline(
    traj, T: int, W: int, forward: bool = True
) -> tuple[float, float]:
    """Long-baseline windowed velocity: (mean(anchor_W) − mean(base_W)) / T.

    For the lost track (forward=True):
        anchor = last W frames  (recent end)
        base   = W frames starting T frames before the anchor
        v = (anchor_mean − base_mean) / T   [points forward]

    For the candidate track (forward=False):
        anchor = first W frames  (early end)
        base   = W frames starting T frames after the anchor
        v = (base_mean − anchor_mean) / T   [points forward = entry direction]

    Falls back to _vel_ols(n=4) when the track is too short.
    """
    if forward:
        anchor_seg = traj[-W:] if len(traj) >= W else traj
        base_seg = traj[-(T + W) : -T] if len(traj) >= T + W else []
    else:
        anchor_seg = traj[:W] if len(traj) >= W else traj
        base_seg = traj[T : T + W] if len(traj) >= T + W else []

    if len(base_seg) < 1:
        return _vel_ols(traj, n=4, forward=forward)

    def _mean_foot(seg):
        pts = np.array([_foot(r[1], r[2], r[3]) for r in seg], dtype=np.float64)
        return pts.mean(axis=0)

    a = _mean_foot(anchor_seg)
    b = _mean_foot(base_seg)
    if forward:
        return float((a[0] - b[0]) / T), float((a[1] - b[1]) / T)
    else:
        return float((b[0] - a[0]) / T), float((b[1] - a[1]) / T)


def _h_trim(traj, window: int = 4, skip: int = 1) -> float:
    """Average height over `window` frames, skipping `skip` terminal frames."""
    seg = traj[-(window + skip) : -skip] if len(traj) > skip else traj[:-1]
    if not seg:
        return float(traj[-1][3])
    return float(np.mean([t[3] for t in seg]))


def _h_extrap(
    traj, gap_half: float, window: int = 4, skip: int = 1, forward: bool = True
) -> float:
    """Linear-extrapolate height to the gap midpoint.

    For the lost track (forward=True) extrapolate `gap_half` frames ahead of the
    last observed frame.  For the candidate (forward=False) extrapolate `gap_half`
    frames *before* the first observed frame.  Uses last/first `window` frames
    (excluding `skip` terminal/initial frames) for the trend fit.
    """
    if forward:
        seg = traj[-(window + skip) : -skip] if len(traj) > skip else traj[:-1]
    else:
        seg = traj[skip : skip + window] if len(traj) > skip else traj[1:]
    if len(seg) < 2:
        return float(traj[-1 if forward else 0][3])
    frames = np.array([t[0] for t in seg], dtype=np.float64)
    heights = np.array([t[3] for t in seg], dtype=np.float64)
    f0 = frames[0]
    x = frames - f0
    sx = x.sum()
    sy = heights.sum()
    sxy = (x * heights).sum()
    sx2 = (x**2).sum()
    n = len(x)
    denom = n * sx2 - sx**2
    if abs(denom) < 1e-6:
        rate = 0.0
    else:
        rate = (n * sxy - sx * sy) / denom
    h_anchor = float(heights[-1]) if forward else float(heights[0])
    h_mid = h_anchor + rate * (gap_half if forward else -gap_half)
    return max(h_mid, 1.0)


def pair_features(traj_a, traj_b, anchor=4):
    """Compute relink features for lost A -> candidate B."""
    la_f, la_cx, la_cy, la_h = traj_a[-1]
    fb_f, fb_cx, fb_cy, fb_h = traj_b[0]
    gap = fb_f - la_f
    h_ref = max((la_h + fb_h) * 0.5, 1.0)

    # h_ref variant columns
    h_lost_raw = float(la_h)
    h_cand_raw = float(fb_h)
    h_lost_trim1 = _h_trim(traj_a, window=4, skip=1)
    h_cand_trim1 = _h_trim(traj_b[::-1], window=4, skip=1)  # skip first by reversing
    h_lost_win6 = _h_trim(traj_a, window=6, skip=1)
    h_cand_win6 = _h_trim(traj_b[::-1], window=6, skip=1)
    half = gap * 0.5
    h_lost_extrap = _h_extrap(traj_a, half, window=4, skip=1, forward=True)
    h_cand_extrap = _h_extrap(traj_b, half, window=4, skip=1, forward=False)

    ax, ay = _foot(la_cx, la_cy, la_h)  # lost exit foot
    bx, by = _foot(fb_cx, fb_cy, fb_h)  # cand entry foot

    vax, vay = _velocity(traj_a[-anchor:])  # lost exit velocity (current default)
    vbx, vby = _velocity(traj_b[:anchor])  # cand entry velocity (current default)

    # forward: extrapolate lost to cand's first frame
    fx, fy = ax + vax * gap, ay + vay * gap
    fwd_resid = np.hypot(fx - bx, fy - by) / h_ref
    # backward: extrapolate cand back to lost's last frame
    rx, ry = bx - vbx * gap, by - vby * gap
    bwd_resid = np.hypot(rx - ax, ry - ay) / h_ref
    mlx, mly = ax + vax * half, ay + vay * half
    mcx, mcy = bx - vbx * half, by - vby * half
    bridge_dist = np.hypot(mlx - mcx, mly - mcy) / h_ref
    # straight-line foot distance & implied speed
    foot_dist = np.hypot(ax - bx, ay - by)
    dist_h = foot_dist / h_ref
    speed_h = foot_dist / max(gap, 1) / h_ref  # heights per frame
    # direction cosine: lost exit velocity vs displacement
    dx, dy = bx - ax, by - ay
    nv = np.hypot(vax, vay)
    nd = np.hypot(dx, dy)
    dir_cos = (vax * dx + vay * dy) / (nv * nd) if nv > 1e-6 and nd > 1e-6 else 0.0
    # own exit/entry speeds (heights/frame) for the reach-gate model
    lost_exit_speed = nv / h_ref
    cand_entry_speed = np.hypot(vbx, vby) / h_ref

    # ── velocity variant columns ──────────────────────────────────────────────
    # Each variant stores fwd_r, bwd_r (normalised residuals, same h_ref) and
    # s_lost (exit speed in h/frame) so sweep_velocity_variants.py can reconstruct
    # the full bridge score without needing to re-read trajectories.
    # Naming: fwd_{tag}, bwd_{tag}, sl_{tag}
    def _vel_feats(vl, vc, tag):
        """Compute fwd/bwd residuals and s_lost for a given (lost, cand) velocity pair."""
        fx_ = ax + vl[0] * gap
        fy_ = ay + vl[1] * gap
        rx_ = bx - vc[0] * gap
        ry_ = by - vc[1] * gap
        return {
            f"fwd_{tag}": np.hypot(fx_ - bx, fy_ - by) / h_ref,
            f"bwd_{tag}": np.hypot(rx_ - ax, ry_ - ay) / h_ref,
            f"sl_{tag}": np.hypot(vl[0], vl[1]) / h_ref,
        }

    # v4_ols: 4-frame OLS (matches GPU bridge_vel4 formula)
    v4l = _vel_ols(traj_a, n=4, forward=True)
    v4c = _vel_ols(traj_b, n=4, forward=False)
    # v8_ols: 8-frame OLS (fits current FOOT_RING_CAP=8, no GPU change)
    v8l = _vel_ols(traj_a, n=8, forward=True)
    v8c = _vel_ols(traj_b, n=8, forward=False)
    # v16_ols: 16-frame OLS (ring_cap=16 needed)
    v16l = _vel_ols(traj_a, n=16, forward=True)
    v16c = _vel_ols(traj_b, n=16, forward=False)
    # v32_ols: 32-frame OLS (ring_cap=32 needed)
    v32l = _vel_ols(traj_a, n=32, forward=True)
    v32c = _vel_ols(traj_b, n=32, forward=False)
    # lb_T10_W4: long baseline T=10, W=4 (ring_cap=14 needed)
    lb10l = _vel_long_baseline(traj_a, T=10, W=4, forward=True)
    lb10c = _vel_long_baseline(traj_b, T=10, W=4, forward=False)
    # lb_T20_W5: long baseline T=20, W=5 (ring_cap=25 needed)
    lb20l = _vel_long_baseline(traj_a, T=20, W=5, forward=True)
    lb20c = _vel_long_baseline(traj_b, T=20, W=5, forward=False)
    # lb_T30_W10: user's proposal T=30, W=10 (ring_cap=40 needed)
    lb30l = _vel_long_baseline(traj_a, T=30, W=10, forward=True)
    lb30c = _vel_long_baseline(traj_b, T=30, W=10, forward=False)

    vel_variants = {}
    for tag, vl, vc in [
        ("v4ols", v4l, v4c),
        ("v8ols", v8l, v8c),
        ("v16ols", v16l, v16c),
        ("v32ols", v32l, v32c),
        ("lb10", lb10l, lb10c),
        ("lb20", lb20l, lb20c),
        ("lb30", lb30l, lb30c),
    ]:
        vel_variants.update(_vel_feats(vl, vc, tag))

    return {
        "gap": gap,
        "dist_h": dist_h,
        "fwd_resid": fwd_resid,
        "bwd_resid": bwd_resid,
        "bridge_dist": bridge_dist,
        "dir_cos": dir_cos,
        "speed_h": speed_h,
        "lost_exit_speed": lost_exit_speed,
        "cand_entry_speed": cand_entry_speed,
        "lost_last_frame": la_f,
        "cand_first_frame": fb_f,
        "lost_lifespan": traj_a[-1][0] - traj_a[0][0] + 1,
        "cand_lifespan": traj_b[-1][0] - traj_b[0][0] + 1,
        "lost_len": len(traj_a),
        "cand_len": len(traj_b),
        "lost_foot_x": ax,
        "lost_foot_y": ay,
        "cand_foot_x": bx,
        "cand_foot_y": by,
        "h_ref": h_ref,
        # h_ref variant columns (raw pixel distances recoverable via * h_ref)
        "h_lost_raw": h_lost_raw,
        "h_cand_raw": h_cand_raw,
        "h_lost_trim1": h_lost_trim1,
        "h_cand_trim1": h_cand_trim1,
        "h_lost_win6": h_lost_win6,
        "h_cand_win6": h_cand_win6,
        "h_lost_extrap": h_lost_extrap,
        "h_cand_extrap": h_cand_extrap,
        **vel_variants,
    }


# ── candidate enumeration ─────────────────────────────────────────────────────
def build_candidates(tracks, max_gap, spatial_gate, min_len):
    ids = [t for t, tr in tracks.items() if len(tr) >= min_len]
    last = {t: tracks[t][-1][0] for t in ids}
    first = {t: tracks[t][0][0] for t in ids}
    frames = {t: {r[0] for r in tracks[t]} for t in ids}

    rows = []
    for a in ids:
        for b in ids:
            if a == b:
                continue
            gap = first[b] - last[a]
            if gap < 1 or gap > max_gap:  # rules 1 + 3
                continue
            if frames[a] & frames[b]:  # rule 2
                continue
            feat = pair_features(tracks[a], tracks[b])
            if spatial_gate > 0 and feat["dist_h"] > spatial_gate:  # rule 4
                continue
            feat["lost_id"] = a
            feat["cand_id"] = b
            rows.append(feat)
    return rows


def tag_uniqueness(rows, priority="bridge_dist"):
    """Greedy one-to-one tag: smallest `priority` wins; consumed ids blocked."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][priority])
    used_lost: set[int] = set()
    used_cand: set[int] = set()
    for r in rows:
        r["accepted"] = False
        r["already_linked"] = False
    for i in order:
        r = rows[i]
        if r["lost_id"] in used_lost or r["cand_id"] in used_cand:
            r["already_linked"] = True
        else:
            r["accepted"] = True
            used_lost.add(r["lost_id"])
            used_cand.add(r["cand_id"])
    return rows


# ── main ──────────────────────────────────────────────────────────────────────
COLS = [
    "seq",
    "lost_id",
    "cand_id",
    "gt_lost",
    "gt_cand",
    "gt_match",
    "gt_valid",
    "accepted",
    "already_linked",
    "gap",
    "dist_h",
    "fwd_resid",
    "bwd_resid",
    "bridge_dist",
    "dir_cos",
    "speed_h",
    "lost_exit_speed",
    "cand_entry_speed",
    "lost_last_frame",
    "cand_first_frame",
    "lost_lifespan",
    "cand_lifespan",
    "lost_len",
    "cand_len",
    "lost_foot_x",
    "lost_foot_y",
    "cand_foot_x",
    "cand_foot_y",
    "h_ref",
    "h_lost_raw",
    "h_cand_raw",
    "h_lost_trim1",
    "h_cand_trim1",
    "h_lost_win6",
    "h_cand_win6",
    "h_lost_extrap",
    "h_cand_extrap",
    # velocity variant columns: fwd_{tag}, bwd_{tag}, sl_{tag}
    # tag legend: v4ep=current(endpoint-diff), v4ols=GPU-matched OLS, v8ols, v16ols, v32ols,
    #             lb10=long-baseline T=10 W=4, lb20=T=20 W=5, lb30=T=30 W=10
    "fwd_v4ols",
    "bwd_v4ols",
    "sl_v4ols",
    "fwd_v8ols",
    "bwd_v8ols",
    "sl_v8ols",
    "fwd_v16ols",
    "bwd_v16ols",
    "sl_v16ols",
    "fwd_v32ols",
    "bwd_v32ols",
    "sl_v32ols",
    "fwd_lb10",
    "bwd_lb10",
    "sl_lb10",
    "fwd_lb20",
    "bwd_lb20",
    "sl_lb20",
    "fwd_lb30",
    "bwd_lb30",
    "sl_lb30",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--gt-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/tools/out/relink_candidates"),
        help="output prefix; writes {prefix}.csv (combined) + per-seq stats",
    )
    ap.add_argument(
        "--max-gap", type=int, default=300, help="max gap in frames (wide gate)"
    )
    ap.add_argument(
        "--spatial-gate",
        type=float,
        default=0.0,
        help="foot-dist/height gate; 0 disables (wide gate, default)",
    )
    ap.add_argument(
        "--min-len", type=int, default=2, help="min track length to consider"
    )
    ap.add_argument(
        "--priority",
        default="bridge_dist",
        choices=["bridge_dist", "gap", "dist_h", "fwd_resid"],
        help="greedy uniqueness priority (smallest wins)",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")

    all_rows: list[dict] = []
    print(
        f"{'seq':<16} {'tracks':>7} {'cands':>7} {'pos':>6} {'neg':>6} {'acc':>5} {'acc+':>5}"
    )
    for seq in args.seqs:
        mot_path = args.mot_dir / f"{seq}.txt"
        gt_path = args.gt_root / seq / "gt" / "gt.txt"
        if not mot_path.exists() or not gt_path.exists():
            print(
                f"{seq:<16}  SKIP (missing {mot_path if not mot_path.exists() else gt_path})"
            )
            continue

        tracks = load_tracks(mot_path)
        gt_tracks = load_tracks(gt_path, drop_nonpos_id=True)
        t2g = map_tracks_to_gt(tracks, gt_tracks)

        rows = build_candidates(tracks, args.max_gap, args.spatial_gate, args.min_len)
        tag_uniqueness(rows, args.priority)

        pos = neg = acc = acc_pos = 0
        for r in rows:
            gl, gc = t2g.get(r["lost_id"], -1), t2g.get(r["cand_id"], -1)
            r["seq"] = seq
            r["gt_lost"], r["gt_cand"] = gl, gc
            r["gt_valid"] = int(gl >= 0 and gc >= 0)
            r["gt_match"] = int(r["gt_valid"] and gl == gc)
            if r["gt_valid"]:
                pos += r["gt_match"]
                neg += 1 - r["gt_match"]
            if r["accepted"]:
                acc += 1
                acc_pos += r["gt_match"]
        all_rows.extend(rows)
        print(
            f"{seq:<16} {len(tracks):>7} {len(rows):>7} {pos:>6} {neg:>6} {acc:>5} {acc_pos:>5}"
        )

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    tot_pos = sum(r["gt_match"] for r in all_rows if r["gt_valid"])
    tot_acc = sum(r["accepted"] for r in all_rows)
    tot_acc_pos = sum(r["accepted"] and r["gt_match"] for r in all_rows)
    print(f"\nWrote {len(all_rows)} candidate pairs -> {csv_path}")
    print(f"  GT-labelled positives (true relink chances): {tot_pos}")
    print(
        f"  greedy-accepted (unique tag): {tot_acc}  of which GT-correct: {tot_acc_pos}"
        f"  (precision {tot_acc_pos / max(tot_acc, 1):.1%})"
    )


if __name__ == "__main__":
    main()
