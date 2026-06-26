#!/usr/bin/env python
"""Score a tracker result dir against sparse-keyframe GT (PersonPath22 eval mode).

When a sequence was converted with ``--keep-all-frames`` the tracker runs on
consecutive frames (correct IoU/motion), but GT exists only at keyframes. Raw
scoring would count every non-keyframe prediction as a FP. This filters each
result file to the frames present in its GT, then re-scores via the normal
CLEAR+HOTA path — yielding association numbers that are NOT depressed by the
keyframe gap.

Usage:
  score_keyframe_filtered.py --results-dir out/<run> \
      --data-root datasets/PersonPath22 --split mot_test_full [--sequences a,b,c]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from saccade.perception.eval.metrics import run_motmetrics_evaluation


def _gt_frames(gt_path: Path) -> set[int]:
    frames: set[int] = set()
    for line in gt_path.read_text().splitlines():
        if line.strip():
            frames.add(int(float(line.split(",")[0])))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sequences", default="")
    args = ap.parse_args()

    results = Path(args.results_dir)
    split_dir = Path(args.data_root) / args.split
    filtered = Path(str(results).rstrip("/") + "_kf")
    filtered.mkdir(parents=True, exist_ok=True)

    seq_filter = {s for s in args.sequences.split(",") if s} if args.sequences else None

    kept = 0
    for ts in sorted(results.glob("*.txt")):
        seq = ts.stem
        if seq.startswith("_"):  # _fps_summary etc.
            continue
        if seq_filter is not None and seq not in seq_filter:
            continue
        gt_path = split_dir / seq / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        kf = _gt_frames(gt_path)
        rows = [
            ln
            for ln in ts.read_text().splitlines()
            if ln.strip() and int(float(ln.split(",")[0])) in kf
        ]
        (filtered / ts.name).write_text("\n".join(rows) + ("\n" if rows else ""))
        kept += 1

    print(f"[filter] wrote {kept} keyframe-filtered result files -> {filtered}")
    metrics = run_motmetrics_evaluation(
        args.data_root, args.split, str(filtered), args.sequences
    )
    if metrics:
        print("\n=== KEYFRAME-FILTERED METRICS ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("[filter] no metrics (missing motmetrics or no matched seqs)")


if __name__ == "__main__":
    main()
