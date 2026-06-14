#!/usr/bin/env python3
"""Attribution: cue-conflict — does box SIZE confirm the foot-y "front" call?

The front-flag picks the occluder by foot-y only. If foot-y and box size (the other
depth cue) disagree, the flag may tag the wrong track as front and the penalty
mis-resolves. This measures, per sequence, on the LIVE flagged-front events
(track-track IoU ≥ thr + foot-decisive):

  size-agree % : fraction where the foot-y front is ALSO the larger box (ratio > 1)
  median ratio : median(front_area / partner_area)

Read against ΔIDF1 and the probe's GT area% (the "correct" rate at which the larger
box is the GT occluder). If 05's live size-agreement is low / ratio < 1, foot-y is
flagging the wrong track and the cue-conflict is the 05 cause.

Usage
-----
  .venv/bin/python scripts/eval/analyze_occ_size.py --substrate /tmp/occ2_off
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

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
GT_AREA = {
    "02": 75.0,
    "04": 62.5,
    "05": 83.9,
    "09": 100.0,
    "10": 85.7,
    "11": 100.0,
    "13": 75.0,
}


def load(path: Path) -> dict[int, dict[int, tuple]]:
    by: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        f, tid = int(p[0]), int(p[1])
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


def flag_size_stats(path: Path, iou_thresh: float, foot_gap: float):
    tracks = load(path)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for tid, fr in tracks.items():
        for f in fr:
            by_frame[f].append(tid)
    ratios = []
    for f, cur in by_frame.items():
        box = {t: tracks[t][f] for t in cur}
        for a in cur:
            best, bestb = 0.0, None
            for b in cur:
                if a == b:
                    continue
                v = iou(box[a], box[b])
                if v > best:
                    best, bestb = v, b
            if bestb is None or best < iou_thresh:
                continue
            fa = box[a][1] + box[a][3]
            fb = box[bestb][1] + box[bestb][3]
            href = 0.5 * (box[a][3] + box[bestb][3])
            if (fa - fb) < foot_gap * max(href, 1e-3):
                continue
            area_a = box[a][2] * box[a][3]
            area_b = box[bestb][2] * box[bestb][3]
            ratios.append(area_a / max(area_b, 1e-6))
    if not ratios:
        return 0, float("nan"), float("nan")
    r = np.array(ratios)
    return len(r), 100.0 * float((r > 1.0).mean()), float(np.median(r))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, default=Path("/tmp/occ2_off"))
    ap.add_argument("--iou-thresh", type=float, default=0.45)
    ap.add_argument("--foot-gap", type=float, default=0.15)
    args = ap.parse_args()

    print(
        f"\nsubstrate={args.substrate}  (live flagged-front box-size vs foot-y call)\n"
    )
    print(
        f"{'seq':<6}{'camera':<17}{'ΔIDF1':>7}{'n':>5}{'size-agree%':>13}{'med ratio':>11}{'GT area%':>10}"
    )
    print("─" * 69)
    for seq in SEQS:
        p = args.substrate / f"{seq}.txt"
        if not p.exists():
            continue
        n, agree, med = flag_size_stats(p, args.iou_thresh, args.foot_gap)
        s = seq.replace("MOT17-", "").replace("-SDP", "")
        print(
            f"{s:<6}{CAM[s]:<17}{DELTA[s]:>+7.1f}{n:>5}{agree:>12.0f}%{med:>11.2f}{GT_AREA[s]:>9.0f}%"
        )
    print(
        "\nsize-agree% low or med ratio<1 on 05 → foot-y flags the smaller (wrong) box = cue conflict"
    )


if __name__ == "__main__":
    main()
