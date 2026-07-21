#!/usr/bin/env python3
"""Attribution: why does the occluder-side mechanism regress MOT17-05?

Hypothesis: the "confident front/occluder" flag (track-track IoU ≥ occ_iou_thresh AND foot
decisively lower than the partner) fires not only on true 1-2 frame crossings but on
*persistent* overlaps — two people walking together in a busy moving-camera scene. There the
back-box penalty is applied continuously and makes the front track refuse legitimate boxes
(→ +FN / −AssA), with no occludee waiting to take the freed box.

This measures, per sequence, from the *emitted tracker output* (a proxy for the live flag on
visible tracks):
  - exposure   : flagged front-track-frames per 1000 frames (how broadly the penalty applies)
  - persistent : fraction of flagged events whose overlap lasts >= PERSIST frames (walking
                 together = a FALSE occlusion, not a crossing)

If 05 has high exposure dominated by persistent overlaps vs 09/10 (where the win lives), the
regression is false-firing on non-crossing overlaps → gate the flag on the partner having
actually just vanished.

Usage
-----
  .venv/bin/python scripts/eval/diagnostics/analyze_front_flag_exposure.py --substrate /tmp/occ2_off
"""
# status: stable

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

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
PERSIST = 5  # overlap lasting >= this many frames = persistent (walking together)


def load_tracks(path: Path) -> dict[int, dict[int, tuple]]:
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


def analyze(path: Path, iou_thresh: float, foot_gap: float):
    tracks = load_tracks(path)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for tid, fr in tracks.items():
        for f in fr:
            by_frame[f].append(tid)
    frames = sorted(by_frame)
    n_frames = len(frames)

    # per (front_tid, partner_tid) consecutive-frame overlap runs while front-flag holds
    flagged_frames = 0
    run_len: dict[tuple[int, int], int] = defaultdict(int)
    run_maxlen: dict[tuple[int, int], int] = defaultdict(int)
    flag_events: set[tuple[int, int, int]] = set()  # (front, partner, run_start)
    active_run: dict[tuple[int, int], int] = {}

    for f in frames:
        cur = by_frame[f]
        flagged_pairs_this_frame = set()
        for i in range(len(cur)):
            for j in range(len(cur)):
                if i == j:
                    continue
                a, b = cur[i], cur[j]
                ba, bb = tracks[a][f], tracks[b][f]
                if iou(ba, bb) < iou_thresh:
                    continue
                footy_a = ba[1] + ba[3]
                footy_b = bb[1] + bb[3]
                h_ref = 0.5 * (ba[3] + bb[3])
                if (footy_a - footy_b) >= foot_gap * max(h_ref, 1e-3):
                    flagged_pairs_this_frame.add((a, b))  # a is front of b
        flagged_frames += len(flagged_pairs_this_frame)
        # track run lengths per directed pair
        seen = set()
        for key in flagged_pairs_this_frame:
            seen.add(key)
            if key in active_run:
                run_len[key] = f - active_run[key] + 1
            else:
                active_run[key] = f
                run_len[key] = 1
                flag_events.add((key[0], key[1], f))
            run_maxlen[key] = max(run_maxlen[key], run_len[key])
        for key in list(active_run):
            if key not in seen:
                del active_run[key]

    persist = sum(1 for v in run_maxlen.values() if v >= PERSIST)
    total_events = len(flag_events)
    return dict(
        n_frames=n_frames,
        exposure=1000.0 * flagged_frames / max(n_frames, 1),
        events=total_events,
        persist_frac=(persist / total_events) if total_events else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, default=Path("/tmp/occ2_off"))
    ap.add_argument("--iou-thresh", type=float, default=0.45)
    ap.add_argument("--foot-gap", type=float, default=0.15)
    args = ap.parse_args()

    print(
        f"\nsubstrate={args.substrate}  iou≥{args.iou_thresh} foot≥{args.foot_gap}  persist≥{PERSIST}f\n"
    )
    print(
        f"{'seq':<6}{'camera':<17}{'exposure/1k':>12}{'events':>8}{'persistent%':>13}"
    )
    print("─" * 56)
    for seq in SEQS:
        p = args.substrate / f"{seq}.txt"
        if not p.exists():
            continue
        r = analyze(p, args.iou_thresh, args.foot_gap)
        s = seq.replace("MOT17-", "").replace("-SDP", "")
        print(
            f"{s:<6}{CAM[s]:<17}{r['exposure']:>12.1f}{r['events']:>8}{100 * r['persist_frac']:>12.0f}%"
        )


if __name__ == "__main__":
    main()
