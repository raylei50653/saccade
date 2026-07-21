#!/usr/bin/env python3
"""Classify relink gaps: person-person overlap vs non-person.

For each (lost, candidate) pair in a relink candidate CSV, checks tracker output
and GT data to label the gap cause.  Uses TWO independent signals:

  1. TRACKER IoU: at loss frame, does any other tracker box overlap the lost track?
     At reappearance frame, does any other tracker box overlap the candidate?
     -> person_overlap_at_loss / person_overlap_at_reappear  (bool)

  2. GT visibility: is the target GT identity visible (vis>0) during the gap?
     -> detector_miss vs real_occlusion

Usage:
  .venv/bin/python scripts/tools/classify_gap_cause.py \
      --csv scripts/tools/out/relink_candidates_clean.csv \
      --mot-dir results/MOT17_clean_substrate \
      --gt-root datasets/MOT17/train
"""
# status: diagnostic

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT / "scripts/tools/out/relink_candidates_clean.csv"
DEFAULT_GT = PROJECT / "datasets/MOT17/train"


def load_mot_tracks(
    path: Path,
) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Load tracker output: {id: [(frame, cx, cy, w, h), ...]} sorted by frame."""
    tracks: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            tracks[tid].append((frm, x + w * 0.5, y + h * 0.5, w, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


def box_iou(b1, b2) -> float:
    """b1, b2 = (cx, cy, w, h)."""
    x1a, y1a = b1[0] - b1[2] * 0.5, b1[1] - b1[3] * 0.5
    x2a, y2a = b1[0] + b1[2] * 0.5, b1[1] + b1[3] * 0.5
    x1b, y1b = b2[0] - b2[2] * 0.5, b2[1] - b2[3] * 0.5
    x2b, y2b = b2[0] + b2[2] * 0.5, b2[1] + b2[3] * 0.5
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = b1[2] * b1[3]
    area_b = b2[2] * b2[3]
    return inter / (area_a + area_b - inter + 1e-6)


def max_person_iou(
    frame: int, target_tid: int, tracks: dict, target_box: tuple
) -> float:
    """Max IoU between target_box and any OTHER track's box at the given frame."""
    max_iou = 0.0
    for tid, traj in tracks.items():
        if tid == target_tid:
            continue
        for frm, cx, cy, w, h in traj:
            if frm == frame:
                iou = box_iou(target_box, (cx, cy, w, h))
                if iou > max_iou:
                    max_iou = iou
                break
    return max_iou


def get_track_box_at(tracks: dict, tid: int, frame: int) -> tuple | None:
    """Return (cx, cy, w, h) for track tid at frame, or None."""
    traj = tracks.get(tid, [])
    for frm, cx, cy, w, h in traj:
        if frm == frame:
            return (cx, cy, w, h)
    return None


def _track_frames(tracks: dict, tid: int) -> set[int]:
    """All frames where track tid has a box."""
    return {frm for frm, _, _, _, _ in tracks.get(tid, [])}


def sustained_overlap(
    tracks: dict,
    target_tid: int,
    target_traj: list,
    start_frame: int,
    end_frame: int,
    iou_thresh: float,
    min_consecutive: int,
) -> bool:
    """Check if target track has sustained IoU >= iou_thresh with any other track.

    Scans frames from start_frame to end_frame (inclusive, in order).
    Returns True if any other track maintains IoU >= iou_thresh
    for at least min_consecutive consecutive frames with the target.

    Uses the target_traj boxes (pre-loaded for efficiency).
    """
    other_tids = [tid for tid in tracks if tid != target_tid]

    # Build frame->box lookup for target
    target_at = {}
    for frm, cx, cy, w, h in target_traj:
        target_at[frm] = (cx, cy, w, h)

    # For each other track, count consecutive overlaps
    for oid in other_tids:
        other_traj = tracks.get(oid, [])
        if not other_traj:
            continue
        consecutive = 0
        for frm, ocx, ocy, ow, oh in other_traj:
            if frm < start_frame or frm > end_frame:
                continue
            if frm not in target_at:
                consecutive = 0
                continue
            t_box = target_at[frm]
            iou = box_iou(t_box, (ocx, ocy, ow, oh))
            if iou >= iou_thresh:
                consecutive += 1
                if consecutive >= min_consecutive:
                    return True
            else:
                # Don't reset — just check if the overlap is sustained.
                # A single-frame gap in overlap should NOT break the count
                # if the overlap resumes immediately. But to be conservative,
                # require ≥ min_consecutive in the window without long breaks.
                if consecutive > 0:
                    # Allow 1-frame gap before resetting
                    pass
    return False


def sustained_overlap_max(
    tracks: dict,
    target_tid: int,
    target_traj: list,
    start_frame: int,
    end_frame: int,
    iou_thresh: float,
    min_consecutive: int,
) -> float:
    """Like sustained_overlap but returns max consecutive overlap length.

    Returns max number of consecutive frames with IoU >= iou_thresh
    between target and any other track in the frame range.
    """
    other_tids = [tid for tid in tracks if tid != target_tid]
    target_at = {}
    for frm, cx, cy, w, h in target_traj:
        target_at[frm] = (cx, cy, w, h)

    best_consecutive = 0
    for oid in other_tids:
        other_traj = tracks.get(oid, [])
        if not other_traj:
            continue
        consecutive = 0
        for frm, ocx, ocy, ow, oh in other_traj:
            if frm < start_frame or frm > end_frame:
                continue
            if frm not in target_at:
                consecutive = 0
                continue
            t_box = target_at[frm]
            iou = box_iou(t_box, (ocx, ocy, ow, oh))
            if iou >= iou_thresh:
                consecutive += 1
                if consecutive > best_consecutive:
                    best_consecutive = consecutive
            else:
                consecutive = 0
    return float(best_consecutive)


def load_gt(
    gt_path: Path,
) -> dict[int, list[tuple[int, float, float, float, float, float]]]:
    """Load GT: {id: [(frame, cx, cy, w, h, vis), ...]}."""
    tracks: dict[int, list] = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            frm, gid = int(p[0]), int(p[1])
            if gid <= 0:
                continue
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            vis = float(p[8]) if len(p) > 8 else 1.0
            tracks[gid].append((frm, x + w * 0.5, y + h * 0.5, w, h, vis))
    return dict(tracks)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument(
        "--mot-dir", default=None, help="Tracker output dir (contains {seq}.txt)"
    )
    ap.add_argument("--gt-root", default=str(DEFAULT_GT))
    ap.add_argument(
        "--iou-thresh",
        type=float,
        default=0.3,
        help="IoU threshold for person-person overlap",
    )
    ap.add_argument(
        "--sustained-frames",
        type=int,
        default=5,
        help="Minimum consecutive overlapping frames for sustained-overlap labels",
    )
    ap.add_argument(
        "--out", default=None, help="output CSV (default: input with _caused suffix)"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Determine MOT dir
    if args.mot_dir:
        mot_dir = Path(args.mot_dir)
    else:
        # Try to infer from CSV path or results dir
        mot_dir = Path("results/MOT17_eval")

    gt_root = Path(args.gt_root)
    iou_thresh = args.iou_thresh
    sustained_frames = max(1, args.sustained_frames)

    # Classify ALL pairs (both same-GT and diff-GT)
    all_labels = []
    person_iou_loss = []
    person_iou_appear = []

    # Cache per sequence
    mot_cache: dict[str, dict] = {}
    gt_cache: dict[str, dict] = {}

    for idx, row in df.iterrows():
        seq = row["seq"]
        lost_id = int(row["lost_id"])
        cand_id = int(row["cand_id"])
        gap_start = int(row["lost_last_frame"])
        gap_end = int(row["cand_first_frame"])
        gt_lost = int(row["gt_lost"]) if row["gt_lost"] > 0 else -1
        gt_match = int(row["gt_match"])

        # Load MOT tracks if needed
        if seq not in mot_cache:
            mot_path = mot_dir / f"{seq}.txt"
            if mot_path.exists():
                mot_cache[seq] = load_mot_tracks(mot_path)
            else:
                mot_cache[seq] = {}

        # Load GT if needed (for same-GT pairs)
        if seq not in gt_cache and gt_match == 1:
            gt_path = gt_root / seq / "gt" / "gt.txt"
            if gt_path.exists():
                gt_cache[seq] = load_gt(gt_path)
            else:
                gt_cache[seq] = {}

        tracks = mot_cache[seq]

        # --- Sustained person-person overlap at LOSS time ---
        # Check the 15 frames before loss for ≥ 5 consecutive frames of IoU ≥ thresh
        lost_traj = tracks.get(lost_id, [])
        if lost_traj:
            max_consec_loss = sustained_overlap_max(
                tracks,
                lost_id,
                lost_traj,
                max(1, gap_start - 15),
                gap_start,
                iou_thresh,
                sustained_frames,
            )
        else:
            max_consec_loss = 0.0

        # --- Sustained person-person overlap at REAPPEARANCE time ---
        cand_traj = tracks.get(cand_id, [])
        if cand_traj:
            max_consec_appear = sustained_overlap_max(
                tracks,
                cand_id,
                cand_traj,
                gap_end,
                min(gap_end + 15, 99999),
                iou_thresh,
                sustained_frames,
            )
        else:
            max_consec_appear = 0.0

        person_iou_loss.append(max_consec_loss)
        person_iou_appear.append(max_consec_appear)

        overlap_loss = max_consec_loss >= sustained_frames
        overlap_appear = max_consec_appear >= sustained_frames

        # --- Gap cause label ---
        if gt_match == 0:
            label = "diff_gt"
        elif gt_match == -1 or gt_match != 1:
            label = "unmapped"
        else:
            # Same-GT: use GT visibility during gap
            gt_tracks = gt_cache.get(seq, {})
            target_gt = gt_lost  # same as gt_cand
            target_traj = gt_tracks.get(target_gt, [])
            target_by_frame = {}
            for frm, cx, cy, w, h, vis in target_traj:
                target_by_frame[frm] = vis

            gap_len = gap_end - gap_start
            max_samples = min(10, max(gap_len - 1, 1))
            sample_frames = np.unique(
                np.linspace(gap_start + 1, gap_end - 1, max_samples).astype(int)
            )
            target_visible = False
            for frm in sample_frames:
                if frm in target_by_frame and target_by_frame[frm] > 0:
                    target_visible = True
                    break

            if target_visible:
                label = "detector_miss"
            elif overlap_loss or overlap_appear:
                label = "person_occlusion"
            else:
                label = "static_obstacle"

        all_labels.append(label)

    df["gap_cause"] = all_labels
    df["person_iou_loss"] = person_iou_loss
    df["person_iou_appear"] = person_iou_appear

    # ── Stats ──────────────────────────────────────────────────────────────
    same_gt = df[df["gt_match"] == 1]
    print(f"Total pairs: {len(df)}")
    print(f"Same-GT (true links): {len(same_gt)}")

    print(f"\n{'=' * 60}")
    print(
        f"SUSTAINED overlap (≥{iou_thresh} IoU for ≥{sustained_frames} consecutive frames, "
        f"±15 frames around loss/reappearance):"
    )
    print(
        f"  max_consec_loss:   mean={same_gt['person_iou_loss'].mean():.1f}  "
        f"med={same_gt['person_iou_loss'].median():.1f}"
    )
    print(
        f"  max_consec_appear: mean={same_gt['person_iou_appear'].mean():.1f}  "
        f"med={same_gt['person_iou_appear'].median():.1f}"
    )

    n_overlap_loss = (same_gt["person_iou_loss"] >= sustained_frames).sum()
    n_overlap_appear = (same_gt["person_iou_appear"] >= sustained_frames).sum()
    n_overlap_both = (
        (same_gt["person_iou_loss"] >= sustained_frames)
        & (same_gt["person_iou_appear"] >= sustained_frames)
    ).sum()
    n_no_overlap = (
        (same_gt["person_iou_loss"] < sustained_frames)
        & (same_gt["person_iou_appear"] < sustained_frames)
    ).sum()
    print(
        f"  overlap at loss:     {n_overlap_loss:4d}  ({n_overlap_loss / len(same_gt) * 100:.1f}%)"
    )
    print(
        f"  overlap at reappear:  {n_overlap_appear:4d}  ({n_overlap_appear / len(same_gt) * 100:.1f}%)"
    )
    print(
        f"  overlap at BOTH:      {n_overlap_both:4d}  ({n_overlap_both / len(same_gt) * 100:.1f}%)"
    )
    print(
        f"  NO overlap at either: {n_no_overlap:4d}  ({n_no_overlap / len(same_gt) * 100:.1f}%)"
    )

    print(f"\n{'=' * 60}")
    print("GAP CAUSE distribution (same-GT only):")
    counts = same_gt["gap_cause"].value_counts()
    for cause, cnt in counts.items():
        print(f"  {cause:20s}: {cnt:4d}  ({cnt / len(same_gt) * 100:.1f}%)")

    # ── Bridge AUC: sustained-no-overlap subset vs full same-GT ──
    print(f"\n{'=' * 60}")
    print(
        f"BRIDGE P/R on NO-SUSTAINED-OVERLAP true links "
        f"(< {sustained_frames} consecutive frames at both ends):"
    )
    try:
        from sklearn.metrics import roc_auc_score

        valid = df[df["gt_valid"] == 1].copy()

        # Subset: only same-GT pairs AND no person overlap at either end
        no_ol = valid[
            (valid["gt_match"] == 1)
            & (valid["person_iou_loss"] < sustained_frames)
            & (valid["person_iou_appear"] < sustained_frames)
        ]
        n_no_ol_pos = len(no_ol)

        # Compare: all same-GT vs no-overlap same-GT
        print("\n  (A) All same-GT pairs:")
        all_same = valid[valid["gt_match"] == 1]
        y_all = valid["gt_match"].values.astype(int)
        bd_all = -valid["bridge_dist"].values
        auc_all = roc_auc_score(y_all, bd_all)
        print(f"      pos={len(all_same):4d}  AUC(full)={auc_all:.4f}")

        hard_all = valid[valid["bridge_dist"] <= 1]
        if len(hard_all) >= 2:
            auc_hard_all = roc_auc_score(
                hard_all["gt_match"].values.astype(int), -hard_all["bridge_dist"].values
            )
            print(f"      AUC(hard, bd<=1)={auc_hard_all:.4f}")

        print(f"\n  (B) No-overlap same-GT only (n_pos={n_no_ol_pos}):")
        # For this analysis: treat no-overlap same-GT as positives, rest as negatives
        # BUT we can't compute AUC within just positives — we need the full pool
        # Instead, compute bridge_dist percentiles on the no-overlap subset

        if n_no_ol_pos > 0:
            print(
                f"      bridge_dist: mean={no_ol['bridge_dist'].mean():.3f}  "
                f"med={no_ol['bridge_dist'].median():.3f}  "
                f"p25={no_ol['bridge_dist'].quantile(0.25):.3f}  "
                f"p75={no_ol['bridge_dist'].quantile(0.75):.3f}"
            )
            print(
                f"      gap: mean={no_ol['gap'].mean():.1f}  med={no_ol['gap'].median():.1f}"
            )

        # Isolate no-overlap pos within valid: compare bridge_dist rank ability
        # to pick out no-overlap true links from all other valid pairs.
        valid_no_ol_idx = set(no_ol.index)
        y_restricted = np.array(
            [1 if idx in valid_no_ol_idx else 0 for idx in valid.index]
        )
        if y_restricted.sum() >= 2 and (1 - y_restricted).sum() >= 2:
            auc_restricted = roc_auc_score(y_restricted, -valid["bridge_dist"].values)
            print(
                f"      AUC(isolate no-overlap pos in all valid): {auc_restricted:.4f}"
            )

        hard_restrict = valid[valid["bridge_dist"] <= 1]
        if len(hard_restrict) >= 2:
            valid_no_ol_idx = set(no_ol.index)
            y_hard = np.array(
                [1 if idx in valid_no_ol_idx else 0 for idx in hard_restrict.index]
            )
            if y_hard.sum() >= 2 and (1 - y_hard).sum() >= 2:
                auc_hard_restricted = roc_auc_score(
                    y_hard, -hard_restrict["bridge_dist"].values
                )
                print(
                    f"      AUC(hard, bd<=1, isolate no-overlap): {auc_hard_restricted:.4f}"
                )

    except ImportError:
        pass

    # Save
    out_path = args.out or args.csv.replace(".csv", "_caused.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
