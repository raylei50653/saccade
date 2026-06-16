#!/usr/bin/env python3
"""Depth-ordering probe — can pre-occlusion geometry tell which of two crossing
boxes is in FRONT?

Motivation
----------
`offline_relink_candidate_analysis.md` §8 / `mamba-score-distribution-20260613.md` §10
identified **109 occlusion crossing-swaps = 22 % of baseline IDs**: two confirmed
tracks mutually occlude (IoU >= 0.5) for 1-2 frames, then separate, and the ids swap.
Velocity-direction locking was measured harmful (motion direction is at the box-jitter
floor, §4/§5). This probe tests a *different* signal that is NOT pre-refuted: **depth
ordering** from the reliable geometry channels (box area + foot y-coordinate), which
the auction currently ignores.

The occlusion condition itself guarantees a depth gap exists: if only one box survives,
one person is substantially in front. The swap happens because the associator is
size/depth-blind, not because the signal is absent.

Method (GT-only, tracker-independent, fully reproducible)
--------------------------------------------------------
For every pair of GT identities that *cross* (separated -> IoU >= IOU_HI -> separated):
  * GROUND TRUTH front  = the id with higher GT visibility during the occlusion window
    (the occluder is closer to the camera and stays visible; the occludee is hidden).
  * PREDICTED front (signal under test), measured on the last clean pre-occlusion frames:
      - area    : front = larger box        (closer => bigger)
      - foot_y  : front = lower box bottom  (closer => lower in image, ground-plane cue)
      - combo   : foot_y, area as tiebreak
  * accuracy = (predicted front == GT front).  Random baseline = 50 %.

A high accuracy (esp. on short 1-2 frame occlusions) means a depth-locked association
policy would fix the swaps; per-sequence split shows whether it is global or only on
elevated/static cameras.

Usage
-----
  .venv/bin/python scripts/tools/depth_ordering_probe.py
  .venv/bin/python scripts/tools/depth_ordering_probe.py --iou-hi 0.5 --max-occ 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]
# rough camera geometry note for reading the per-seq table
CAM = {
    "MOT17-02": "static elevated",
    "MOT17-04": "static high",
    "MOT17-05": "moving low",
    "MOT17-09": "static low",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}


def load_gt(
    path: Path,
) -> dict[int, dict[int, tuple[float, float, float, float, float]]]:
    """Return gt_id -> {frame: (x, y, w, h, vis)} for class==1, conf==1 rows."""
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 9:
            continue
        frame, gid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf, cls, vis = int(p[6]), int(p[7]), float(p[8])
        if conf != 1 or cls != 1:
            continue
        by_id[gid][frame] = (x, y, w, h, vis)
    return dict(by_id)


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def find_cross_events(gt, iou_hi, iou_lo, max_occ, min_life, pre_win):
    """Yield crossing/occlusion events.

    Each event: dict(a, b, occ_frames, pre_frames, vis_a, vis_b).
    """
    # frame -> set of present ids, to limit pair scan
    present: dict[int, list[int]] = defaultdict(list)
    for gid, fr in gt.items():
        for f in fr:
            present[f].append(gid)

    events = []
    seen: set[tuple[int, int, int]] = set()  # (a,b,occ_start) dedup
    for f in sorted(present):
        cur = present[f]
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                a, b = cur[i], cur[j]
                ba, bb = gt[a][f], gt[b][f]
                if iou(ba, bb) < iou_hi:
                    continue
                # walk back to occlusion start
                s = f
                while (
                    (s - 1) in gt[a]
                    and (s - 1) in gt[b]
                    and iou(gt[a][s - 1], gt[b][s - 1]) >= iou_hi
                ):
                    s -= 1
                if (a, b, s) in seen:
                    continue
                # walk forward to occlusion end
                e = f
                while (
                    (e + 1) in gt[a]
                    and (e + 1) in gt[b]
                    and iou(gt[a][e + 1], gt[b][e + 1]) >= iou_hi
                ):
                    e += 1
                seen.add((a, b, s))
                occ_len = e - s + 1
                if occ_len > max_occ:
                    continue
                # require a clean pre-occlusion approach: both present & separated
                pre = [
                    fr for fr in range(s - pre_win, s) if fr in gt[a] and fr in gt[b]
                ]
                pre = [fr for fr in pre if iou(gt[a][fr], gt[b][fr]) < iou_lo]
                if not pre:
                    continue
                # both identities must be substantial (confirmable / swap-capable)
                if len(gt[a]) < min_life or len(gt[b]) < min_life:
                    continue
                occ = list(range(s, e + 1))
                vis_a = float(np.mean([gt[a][fr][4] for fr in occ]))
                vis_b = float(np.mean([gt[b][fr][4] for fr in occ]))
                events.append(
                    dict(
                        a=a,
                        b=b,
                        occ=occ,
                        pre=pre,
                        occ_len=occ_len,
                        vis_a=vis_a,
                        vis_b=vis_b,
                    )
                )
    return events


def median_geom(gt_id_frames, frames):
    """Median (area, foot_y, height) over frames."""
    areas, foots, hs = [], [], []
    for fr in frames:
        x, y, w, h, _ = gt_id_frames[fr]
        areas.append(w * h)
        foots.append(y + h)
        hs.append(h)
    return float(np.median(areas)), float(np.median(foots)), float(np.median(hs))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--iou-hi", type=float, default=0.5, help="occlusion (merge) IoU")
    ap.add_argument(
        "--iou-lo", type=float, default=0.3, help="separated IoU for pre frame"
    )
    ap.add_argument(
        "--max-occ", type=int, default=2, help="max occlusion length (frames)"
    )
    ap.add_argument(
        "--pre-win", type=int, default=4, help="pre-occlusion window to measure geom"
    )
    ap.add_argument(
        "--min-life", type=int, default=10, help="min GT lifetime (frames) per id"
    )
    ap.add_argument(
        "--vis-margin",
        type=float,
        default=0.10,
        help="min GT-visibility gap during occlusion to define a front (else ambiguous depth)",
    )
    args = ap.parse_args()

    rows = []  # per-event records
    skipped_ambig = 0
    for seq in SEQS:
        gt_path = args.root / f"{seq}-{args.det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  ! missing {gt_path}")
            continue
        gt = load_gt(gt_path)
        evs = find_cross_events(
            gt, args.iou_hi, args.iou_lo, args.max_occ, args.min_life, args.pre_win
        )
        for ev in evs:
            a, b = ev["a"], ev["b"]
            # GT front = higher visibility during occlusion (the occluder)
            vmax, vmin = max(ev["vis_a"], ev["vis_b"]), min(ev["vis_a"], ev["vis_b"])
            if (vmax - vmin) < args.vis_margin or vmin >= 0.5:
                skipped_ambig += 1
                continue
            gt_front = a if ev["vis_a"] > ev["vis_b"] else b
            # predicted front from pre-occlusion geometry
            ar_a, fy_a, h_a = median_geom(gt[a], ev["pre"])
            ar_b, fy_b, h_b = median_geom(gt[b], ev["pre"])
            pred_area = a if ar_a > ar_b else b
            pred_foot = a if fy_a > fy_b else b
            pred_combo = pred_foot if fy_a != fy_b else pred_area
            h_ref = 0.5 * (h_a + h_b)
            rows.append(
                dict(
                    seq=seq,
                    occ_len=ev["occ_len"],
                    gt_front=gt_front,
                    ok_area=int(pred_area == gt_front),
                    ok_foot=int(pred_foot == gt_front),
                    ok_combo=int(pred_combo == gt_front),
                    area_ratio=max(ar_a, ar_b) / max(min(ar_a, ar_b), 1e-6),
                    foot_gap_h=abs(fy_a - fy_b) / max(h_ref, 1e-6),
                    vis_gap=vmax - vmin,
                )
            )

    if not rows:
        raise SystemExit("No crossing events found — relax --iou-hi / --max-occ.")

    def acc(recs, key):
        return 100.0 * np.mean([r[key] for r in recs]) if recs else float("nan")

    print(
        f"\nCrossing events: {len(rows)} usable "
        f"(skipped {skipped_ambig} with ambiguous depth: vis-gap<{args.vis_margin} or both visible)"
    )
    print(
        f"Params: iou_hi={args.iou_hi} max_occ={args.max_occ}f pre_win={args.pre_win}f "
        f"min_life={args.min_life}f vis_margin={args.vis_margin}\n"
    )

    hdr = f"{'sequence':<16}{'camera':<18}{'n':>4}  {'area%':>7}{'foot%':>7}{'combo%':>7}  {'ar_ratio':>9}{'foot/h':>8}"
    print(hdr)
    print("─" * len(hdr))
    for seq in SEQS:
        recs = [r for r in rows if r["seq"] == seq]
        if not recs:
            continue
        print(
            f"{seq:<16}{CAM[seq]:<18}{len(recs):>4}  "
            f"{acc(recs, 'ok_area'):>6.1f} {acc(recs, 'ok_foot'):>6.1f} {acc(recs, 'ok_combo'):>6.1f}  "
            f"{np.median([r['area_ratio'] for r in recs]):>8.2f}x"
            f"{np.median([r['foot_gap_h'] for r in recs]):>8.2f}"
        )
    print("─" * len(hdr))
    print(
        f"{'OVERALL':<34}{len(rows):>4}  "
        f"{acc(rows, 'ok_area'):>6.1f} {acc(rows, 'ok_foot'):>6.1f} {acc(rows, 'ok_combo'):>6.1f}"
    )

    # breakdown by occlusion length (the 1-2 frame swaps are the §8 target)
    print("\n── by occlusion length ──")
    for ol in sorted({r["occ_len"] for r in rows}):
        recs = [r for r in rows if r["occ_len"] == ol]
        print(
            f"  occ={ol}f  n={len(recs):>4}  "
            f"area={acc(recs, 'ok_area'):.1f}%  foot={acc(recs, 'ok_foot'):.1f}%  "
            f"combo={acc(recs, 'ok_combo'):.1f}%"
        )

    # decisiveness: accuracy when the cue is strong vs weak
    print("\n── combo accuracy vs cue decisiveness (foot_gap / height) ──")
    fg = np.array([r["foot_gap_h"] for r in rows])
    for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 99)]:
        m = [r for r, g in zip(rows, fg) if lo <= g < hi]
        if m:
            print(
                f"  foot_gap [{lo:.2f},{hi:.2f})h  n={len(m):>4}  combo={acc(m, 'ok_combo'):.1f}%"
            )


if __name__ == "__main__":
    main()
