#!/usr/bin/env python3
"""Diagnose ID switches by gap type to determine P3-B (Dormant Bank + HNSW) value.

Uses motmetrics' own accumulator to get the authoritative switch events,
then annotates each switch with:
  - gt_gap:    frames the GT identity was absent before returning (0 = continuously present)
  - match_gap: frames since the tracker last had a match for this GT identity

Gap buckets for gt_gap:
  0:        GT continuously present — tracker oscillation / primary association failure
  1–30:     short occlusion
  31–90:    medium re-ID window (lost_tracks + bank may still hold)
  91+:      long re-ID — P3-B (Dormant Bank + HNSW) territory

Usage:
    uv run python scripts/tools/diagnose_id_switches.py [--results results/a2_candidate]
"""
# status: experiment

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# NumPy 2.0 compat patch for motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm  # noqa: E402

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# GT gap index: for each (seq, gt_id) what frames does it appear in?
# ---------------------------------------------------------------------------


def build_gt_frames(gt_path: Path) -> dict[int, list[int]]:
    """Return gt_id → sorted list of frames (class=1, conf=1 only)."""
    by_id: dict[int, list[int]] = defaultdict(list)
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        frame, gt_id = int(parts[0]), int(parts[1])
        conf = int(parts[6])
        class_id = int(parts[7]) if len(parts) > 7 else 1
        if conf != 1 or class_id != 1:
            continue
        by_id[gt_id].append(frame)
    return {k: sorted(v) for k, v in by_id.items()}


def build_gt_visibility(gt_path: Path) -> dict[int, dict[int, float]]:
    """Return gt_id → {frame: visibility} for class=1, conf=1 rows."""
    by_id: dict[int, dict[int, float]] = defaultdict(dict)
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        frame, gt_id = int(parts[0]), int(parts[1])
        conf = int(parts[6])
        class_id = int(parts[7]) if len(parts) > 7 else 1
        vis = float(parts[8]) if len(parts) > 8 else 1.0
        if conf != 1 or class_id != 1:
            continue
        by_id[gt_id][frame] = vis
    return dict(by_id)


# ---------------------------------------------------------------------------
# Analysis using motmetrics accumulator
# ---------------------------------------------------------------------------

VIS_THRESH = 0.3  # below this = "effectively occluded"


def analyze_sequence(gt_path: Path, hyp_path: Path, seq_name: str) -> dict:
    gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5)

    # Build GT frame presence and visibility
    gt_frames_by_id = build_gt_frames(gt_path)
    gt_vis_by_id = build_gt_visibility(gt_path)

    # Extract SWITCH events from the accumulator
    events = acc.events
    switch_events = events[events["Type"] == "SWITCH"]

    # For each SWITCH event we need:
    #   - OId: GT id that switched
    #   - HId: new hyp id
    #   - FrameId: frame of switch (this is the accumulator's internal index)
    # We also need the last MATCH frame for each GT id

    # Walk all events to build: gt_id → list of (frame, matched_hyp_id) for MATCH events
    match_history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # acc.events index is (FrameId, EventId) — FrameId is the actual frame number
    for (frame_id, _), row in events.iterrows():
        if row["Type"] == "MATCH":
            oid = row["OId"]  # GT id
            hid = row["HId"]  # hyp id
            if not (isinstance(oid, float) and np.isnan(oid)):
                match_history[int(oid)].append((int(frame_id), int(hid)))

    # For each SWITCH event, compute gaps
    switches = []
    for (frame_id, _), row in switch_events.iterrows():
        gt_id = int(row["OId"])
        new_hyp = row["HId"]

        # match_gap: frames since last MATCH for this GT id
        history = match_history.get(gt_id, [])
        # Find last match before this frame
        prev_matches = [(f, h) for f, h in history if f < frame_id]
        if prev_matches:
            last_match_frame, _ = prev_matches[-1]
            match_gap = frame_id - last_match_frame - 1
        else:
            last_match_frame, _ = None, None
            match_gap = frame_id  # never matched before — treat as large

        # gt_gap: frames GT id was absent before frame_id (conf=0 or off-screen)
        gt_frame_list = gt_frames_by_id.get(gt_id, [])
        prev_gt_frames = [f for f in gt_frame_list if f < frame_id]
        if prev_gt_frames:
            last_gt_frame = prev_gt_frames[-1]
            gt_gap = frame_id - last_gt_frame - 1
        else:
            gt_gap = 0

        # vis_gap: consecutive frames of low visibility (< VIS_THRESH) before frame_id
        gt_vis = gt_vis_by_id.get(gt_id, {})
        vis_gap = 0
        for f in sorted(gt_frame_list, reverse=True):
            if f >= frame_id:
                continue
            if gt_vis.get(f, 1.0) < VIS_THRESH:
                vis_gap += 1
            else:
                break

        switches.append(
            {
                "frame": frame_id,
                "gt_id": gt_id,
                "new_hyp": new_hyp,
                "match_gap": match_gap,
                "gt_gap": gt_gap,
                "vis_gap": vis_gap,  # frames of low visibility before switch
            }
        )

    # Also compute total motmetrics IDs for reference
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_switches", "num_misses", "num_false_positives", "idf1", "mota"],
        name=seq_name,
    )
    mm_ids = int(summary.loc[seq_name, "num_switches"])

    return {
        "switches": switches,
        "mm_ids": mm_ids,  # motmetrics authoritative count
        "n_gt_ids": len(gt_frames_by_id),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

GT_BUCKETS = [
    (0, 0, "0 f    (GT continuously present — tracker oscillation)"),
    (1, 30, "1–30 f (short occlusion)                               "),
    (31, 90, "31–90 f (medium re-ID window)                          "),
    (91, 999999, "91+ f  (P3-B territory)                                "),
]

MATCH_BUCKETS = [
    (0, 5, "0–5 f  (dropped briefly)  "),
    (6, 30, "6–30 f (short gap)         "),
    (31, 90, "31–90 f (medium gap)       "),
    (91, 999999, "91+ f  (long gap / P3-B)   "),
]

VIS_BUCKETS = [
    (0, 0, "0 f    (target was visible at last GT frame)         "),
    (1, 10, "1–10 f (brief occlusion before switch)               "),
    (11, 30, "11–30 f (moderate occlusion before switch)           "),
    (31, 999999, "31+ f  (heavy/long occlusion before switch)          "),
]


def bucket(gap: int, buckets) -> int:
    for i, (lo, hi, _) in enumerate(buckets):
        if lo <= gap <= hi:
            return i
    return len(buckets) - 1


def report_section(switches, label, buckets_def):
    n = len(switches)
    counts = defaultdict(int)
    for s in switches:
        counts[bucket(s[label], buckets_def)] += 1
    for i, (_, _, desc) in enumerate(buckets_def):
        print(
            f"    {desc}: {counts[i]:5d}  ({100 * counts[i] / n:.1f}%)"
            if n
            else f"    {desc}:     0"
        )


def report_seq(seq_name: str, result: dict) -> None:
    switches = result["switches"]
    n = len(switches)
    print(f"\n{'─' * 68}")
    print(
        f"  {seq_name}  ({result['n_gt_ids']} GT ids)  "
        f"[motmetrics IDs={result['mm_ids']}, our switches={n}]"
    )
    print(f"{'─' * 68}")
    if not switches:
        print("  No ID switches detected.")
        return

    print("  By low-visibility gap (frames target occluded, vis<0.3, before switch):")
    report_section(switches, "vis_gap", VIS_BUCKETS)
    print()
    print("  By match gap (how long tracker had no match):")
    report_section(switches, "match_gap", MATCH_BUCKETS)


def main():
    parser = argparse.ArgumentParser(description="Diagnose ID switch gap distribution")
    parser.add_argument(
        "--results",
        default="results/a2_candidate",
        help="Result directory with MOT17-XX-SDP.txt files",
    )
    parser.add_argument(
        "--data-root", default="datasets/MOT17/train", help="MOT17 train root"
    )
    parser.add_argument("--detector", default="SDP")
    args = parser.parse_args()

    results_dir = project_root / args.results
    data_root = project_root / args.data_root

    hyp_files = sorted(results_dir.glob(f"MOT17-*-{args.detector}.txt"))
    if not hyp_files:
        print(f"No result files found in {results_dir}")
        return

    all_switches: list[dict] = []
    total_mm_ids = 0
    total_gt_ids = 0

    for hyp_path in hyp_files:
        seq_name = hyp_path.stem
        gt_path = data_root / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"GT not found: {gt_path}")
            continue

        print(f"Processing {seq_name}...", end=" ", flush=True)
        result = analyze_sequence(gt_path, hyp_path, seq_name)
        print("done")
        report_seq(seq_name, result)
        all_switches.extend(result["switches"])
        total_mm_ids += result["mm_ids"]
        total_gt_ids += result["n_gt_ids"]

    # Overall summary
    n = len(all_switches)
    print(f"\n{'═' * 68}")
    print(f"  OVERALL  ({total_gt_ids} GT ids, 7 sequences)")
    print(f"  motmetrics total IDs = {total_mm_ids}  |  our switch events = {n}")
    print(f"{'═' * 68}")
    if not all_switches:
        return

    print("\n  By low-visibility gap (frames occluded before switch, vis<0.3):")
    report_section(all_switches, "vis_gap", VIS_BUCKETS)

    print("\n  By match gap (how long tracker had no match):")
    report_section(all_switches, "match_gap", MATCH_BUCKETS)

    # P3-B impact estimate (match_gap ≥ 91 is the binding constraint)
    p3b = sum(1 for s in all_switches if s["match_gap"] >= 91)
    primary = sum(1 for s in all_switches if s["match_gap"] <= 5)
    print("\n  ── Verdict ─────────────────────────────────────────")
    print(
        f"  Primary assoc oscillation (match_gap ≤ 5f): "
        f"{primary} / {n}  ({100 * primary / n:.1f}%)  ← main target"
    )
    print(
        f"  P3-B relevant (match_gap ≥ 91f):             "
        f"{p3b} / {n}  ({100 * p3b / n:.1f}%)  ← ceiling if P3-B perfect"
    )

    long_mg = [s["match_gap"] for s in all_switches if s["match_gap"] >= 91]
    if long_mg:
        arr = np.array(long_mg)
        print(
            f"  Long match-gap stats: median={np.median(arr):.0f}  "
            f"p75={np.percentile(arr, 75):.0f}  max={arr.max():.0f}"
        )


if __name__ == "__main__":
    main()
