#!/usr/bin/env python3
"""Attribution: is MOT17-05's occluder-mechanism regression from inaccurate boxes
(框不准) or from too many simultaneous occlusions (occ 太多)?

Two per-sequence measurements on the substrate output + GT:

  localization : median IoU of matched hyp→GT boxes (how accurate the boxes the
                 mechanism reasons over actually are). Low = 框不准.
  cluster size : at front-flag candidate frames (track-track IoU ≥ thr + foot-decisive),
                 the mean number of OTHER tracks overlapping the flagged front (IoU ≥ 0.3).
                 High = many simultaneous occluders (occ 太多), where a one-partner
                 mutual-exclusion mis-resolves.

Read against the live ΔIDF1 to see which variable tracks the 05 regression.

Usage
-----
  .venv/bin/python scripts/eval/analyze_05_cause.py --substrate /tmp/occ2_off
"""
# status: experiment

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm  # noqa: E402

SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]
CAM = {
    "02": "static elevated",
    "04": "static high",
    "05": "moving low",
    "09": "static low",
    "10": "moving",
    "11": "moving",
    "13": "moving",
}
DELTA = {
    "02": -0.4,
    "04": +0.3,
    "05": -1.1,
    "09": +2.9,
    "10": +1.5,
    "11": 0.0,
    "13": +0.8,
}


def load_boxes(path: Path, gt: bool) -> dict[int, dict[int, tuple]]:
    by: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        f, tid = int(p[0]), int(p[1])
        if gt and (len(p) < 9 or int(p[6]) != 1 or int(p[7]) != 1):
            continue
        by[tid][f] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return dict(by)


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-6) if inter > 0 else 0.0


def localization_iou(gt_path: Path, hyp_path: Path) -> float:
    """Median IoU over motmetrics-matched (gt, hyp) pairs."""
    gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5)
    d = acc.events.loc[acc.events["Type"] == "MATCH", "D"].to_numpy(dtype=float)
    d = d[~np.isnan(d)]
    return float(1.0 - np.median(d)) if len(d) else float("nan")  # D = 1 - IoU


def cluster_size(hyp_path: Path, iou_thresh: float, foot_gap: float) -> float:
    tracks = load_boxes(hyp_path, gt=False)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for tid, fr in tracks.items():
        for f in fr:
            by_frame[f].append(tid)
    sizes = []
    for f, cur in by_frame.items():
        boxes = {t: tracks[t][f] for t in cur}
        for a in cur:
            # is a a confident front of its argmax partner?
            best, bestb = 0.0, None
            for b in cur:
                if a == b:
                    continue
                v = iou(boxes[a], boxes[b])
                if v > best:
                    best, bestb = v, b
            if bestb is None or best < iou_thresh:
                continue
            fa = boxes[a][1] + boxes[a][3]
            fb = boxes[bestb][1] + boxes[bestb][3]
            href = 0.5 * (boxes[a][3] + boxes[bestb][3])
            if (fa - fb) < foot_gap * max(href, 1e-3):
                continue
            # a is flagged front → count how many OTHER tracks overlap it (IoU>=0.3)
            n_overlap = sum(1 for b in cur if b != a and iou(boxes[a], boxes[b]) >= 0.3)
            sizes.append(n_overlap)
    return float(np.mean(sizes)) if sizes else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, default=Path("/tmp/occ2_off"))
    ap.add_argument("--iou-thresh", type=float, default=0.45)
    ap.add_argument("--foot-gap", type=float, default=0.15)
    args = ap.parse_args()
    gt_root = PROJECT_ROOT / "datasets" / "MOT17" / "train"

    print(f"\nsubstrate={args.substrate}\n")
    print(
        f"{'seq':<6}{'camera':<17}{'ΔIDF1':>7}{'matched IoU':>13}{'flag cluster':>14}"
    )
    print("─" * 57)
    for seq in SEQS:
        gt = gt_root / seq / "gt" / "gt.txt"
        hyp = args.substrate / f"{seq}.txt"
        if not gt.exists() or not hyp.exists():
            continue
        loc = localization_iou(gt, hyp)
        clu = cluster_size(hyp, args.iou_thresh, args.foot_gap)
        s = seq.replace("MOT17-", "").replace("-SDP", "")
        print(f"{s:<6}{CAM[s]:<17}{DELTA[s]:>+7.1f}{loc:>13.3f}{clu:>14.2f}")
    print("\nmatched IoU low → 框不准; flag cluster high → 同時 occ 太多")


if __name__ == "__main__":
    main()
