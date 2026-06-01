"""
Analyze ID switch types in MOT17 tracking output.

Classification:
  FRAG_SHORT   : same GT person, gap 1-5 frames
  FRAG_MEDIUM  : same GT person, gap 6-30 frames
  FRAG_LONG    : same GT person, gap >30 frames
  SWAP         : two different GT persons exchange tracker IDs
  ENTER        : new person enters the scene (not a switch)
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_mot(path: str) -> dict[int, dict[int, tuple]]:
    """frame_id -> {track_id -> (x1,y1,x2,y2)}"""
    data: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame, tid = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        data[frame][tid] = (x, y, x + w, y + h)
    return data


def iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def match_frame(pred: dict[int, tuple], gt: dict[int, tuple], thr: float = 0.5):
    """Returns {pred_id: gt_id} for IoU >= thr (greedy best-match)."""
    scores = []
    for pid, pb in pred.items():
        for gid, gb in gt.items():
            s = iou(pb, gb)
            if s >= thr:
                scores.append((s, pid, gid))
    scores.sort(reverse=True)
    used_p, used_g = set(), set()
    match = {}
    for s, pid, gid in scores:
        if pid in used_p or gid in used_g:
            continue
        match[pid] = gid
        used_p.add(pid)
        used_g.add(gid)
    return match


def analyze_seq(pred_path: str, gt_path: str, seq_name: str):
    pred_data = load_mot(pred_path)
    gt_data = load_mot(gt_path)

    frames = sorted(set(pred_data) | set(gt_data))

    # Build: pred_id -> list of (frame, gt_id) matched pairs
    pred_gt_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for frame in frames:
        if frame not in pred_data or frame not in gt_data:
            continue
        match = match_frame(pred_data[frame], gt_data[frame])
        for pid, gid in match.items():
            pred_gt_history[pid].append((frame, gid))

    # For each GT id: find all tracker IDs that were ever matched to it
    gt_to_pred_frames: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pid, hist in pred_gt_history.items():
        for frame, gid in hist:
            gt_to_pred_frames[gid].append((frame, pid))

    # Classify ID switches per GT track
    stats = {
        "FRAG_SHORT": 0,  # gap 1-5
        "FRAG_MEDIUM": 0,  # gap 6-30
        "FRAG_LONG": 0,  # gap >30
        "SWAP": 0,
    }
    gap_lengths = []
    events = []

    for gid, fp_list in gt_to_pred_frames.items():
        fp_list.sort()
        # Segment by pred_id changes
        segments: list[tuple[int, int, int]] = []  # (start_frame, end_frame, pred_id)
        seg_start = fp_list[0][0]
        seg_pid = fp_list[0][1]
        seg_end = fp_list[0][0]
        for frame, pid in fp_list[1:]:
            if pid == seg_pid and frame == seg_end + 1:
                seg_end = frame
            elif pid == seg_pid:
                # gap within same pred_id: just extend
                seg_end = frame
            else:
                segments.append((seg_start, seg_end, seg_pid))
                seg_start, seg_pid, seg_end = frame, pid, frame
        segments.append((seg_start, seg_end, seg_pid))

        for i in range(1, len(segments)):
            prev_end, prev_pid = segments[i - 1][1], segments[i - 1][2]
            curr_start, curr_pid = segments[i][0], segments[i][2]
            gap = curr_start - prev_end  # frames between last-seen and re-seen

            # Check if prev_pid is still active when curr segment starts
            # (SWAP: prev_pid matched to DIFFERENT gt at curr_start)
            prev_pid_at_curr = None
            if curr_start in pred_data and prev_pid in pred_data[curr_start]:
                if curr_start in gt_data:
                    m = match_frame(pred_data[curr_start], gt_data[curr_start])
                    prev_pid_at_curr = m.get(prev_pid)

            if prev_pid_at_curr is not None and prev_pid_at_curr != gid:
                stats["SWAP"] += 1
                events.append(
                    {
                        "type": "SWAP",
                        "seq": seq_name,
                        "gt_id": gid,
                        "from_pid": prev_pid,
                        "to_pid": curr_pid,
                        "frame": curr_start,
                        "gap": gap,
                    }
                )
            else:
                gap_lengths.append(gap)
                if gap <= 5:
                    stats["FRAG_SHORT"] += 1
                    t = "FRAG_SHORT"
                elif gap <= 30:
                    stats["FRAG_MEDIUM"] += 1
                    t = "FRAG_MEDIUM"
                else:
                    stats["FRAG_LONG"] += 1
                    t = "FRAG_LONG"
                events.append(
                    {
                        "type": t,
                        "seq": seq_name,
                        "gt_id": gid,
                        "from_pid": prev_pid,
                        "to_pid": curr_pid,
                        "frame": curr_start,
                        "gap": gap,
                    }
                )

    return stats, gap_lengths, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ids_analysis")
    parser.add_argument("--data-root", default="datasets/MOT17")
    args = parser.parse_args()

    out_dir = Path(args.output)
    data_root = Path(args.data_root)

    total_stats = {k: 0 for k in ["FRAG_SHORT", "FRAG_MEDIUM", "FRAG_LONG", "SWAP"]}
    all_gaps = []
    all_events = []

    for pred_file in sorted(out_dir.glob("MOT17-*-SDP.txt")):
        seq = pred_file.stem
        gt_file = data_root / "train" / seq / "gt" / "gt.txt"
        if not gt_file.exists():
            print(f"  [skip] GT not found: {gt_file}")
            continue
        stats, gaps, events = analyze_seq(str(pred_file), str(gt_file), seq)
        all_gaps.extend(gaps)
        all_events.extend(events)
        total = sum(stats.values())
        print(
            f"\n{seq}:  FRAG_SHORT={stats['FRAG_SHORT']}  FRAG_MEDIUM={stats['FRAG_MEDIUM']}"
            f"  FRAG_LONG={stats['FRAG_LONG']}  SWAP={stats['SWAP']}  total={total}"
        )
        for k in total_stats:
            total_stats[k] += stats[k]

    grand_total = sum(total_stats.values())
    print("\n" + "=" * 60)
    print(f"TOTAL IDs events: {grand_total}")
    for k, v in total_stats.items():
        pct = 100 * v / grand_total if grand_total else 0
        print(f"  {k:14s}: {v:4d}  ({pct:5.1f}%)")

    if all_gaps:
        arr = np.array(all_gaps)
        print(f"\nFragmentation gap distribution (n={len(arr)}):")
        for lo, hi in [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 9999)]:
            cnt = int(((arr >= lo) & (arr <= hi)).sum())
            print(
                f"  {lo:4d}-{hi if hi < 9999 else '∞':>4} frames: {cnt:4d}  ({100 * cnt / len(arr):.1f}%)"
            )
        print(
            f"  median={np.median(arr):.0f}  mean={arr.mean():.1f}  max={arr.max():.0f}"
        )

    # Top sequences by swap events
    swap_events = [e for e in all_events if e["type"] == "SWAP"]
    if swap_events:
        print(f"\nSWAP events ({len(swap_events)}):")
        for e in swap_events[:20]:
            print(
                f"  seq={e['seq']}  gt={e['gt_id']}  frame={e['frame']}  gap={e['gap']}"
            )


if __name__ == "__main__":
    main()
