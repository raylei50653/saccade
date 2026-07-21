#!/usr/bin/env python3
"""Attribution: what does the tracker physically DO at each occlusion crossing-swap?

The Phase-1 `Occluded(by=A)` mechanism was NO-GO (one-sided occludee depth penalty was
bit-identically inert; the OCCLUDED state only cannibalised the bridge relink). Before any
redesign, this script attributes — offline, no rebuild — *where the fix must hook* by
classifying each crossing-swap SWITCH event on the substrate output:

  REBORN  : the new hyp id is freshly born at the swap frame
            → the occludee lost its identity and re-entered as a NEW track
            → fix lives in birth-suppression / relink (the bridge should catch this).
  ABSORB  : the new hyp id is an OLDER existing track (typically the occluder)
            → an existing track took over the GT's box
            → fix lives in auction mutual-exclusion on the OCCLUDER side
              (a one-sided penalty on the occludee's cost row cannot prevent this —
               this is exactly why the Phase-1 cost term was inert).

It also tags whether the new id is the spatial occluder (overlaps the GT box and is more
frontal) at the swap, and whether the bridge could plausibly have relinked (short gap).

Usage
-----
  .venv/bin/python scripts/eval/diagnostics/analyze_crossing_swaps.py \
      --substrate results/mamba_whole_graph_current_7seq_recheck
"""
# status: stable

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
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
VIS_THRESH = 0.3
REBORN_AGE = 2  # new id younger than this at swap = freshly born


def load_boxes(path: Path, cls_filter: bool) -> dict[int, dict[int, tuple]]:
    """id -> {frame: (x,y,w,h)}; cls_filter=True keeps GT class==1 conf==1."""
    by: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        frame, tid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        if cls_filter:
            if len(p) < 9 or int(p[6]) != 1 or int(p[7]) != 1:
                continue
        by[tid][frame] = (x, y, w, h)
    return dict(by)


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-6) if inter > 0 else 0.0


def analyze(gt_path: Path, hyp_path: Path, max_gap: int):
    gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5)
    events = acc.events

    match_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (f, _), row in events.iterrows():
        if row["Type"] == "MATCH" and not (
            isinstance(row["OId"], float) and np.isnan(row["OId"])
        ):
            match_hist[int(row["OId"])].append((int(f), int(row["HId"])))

    gt_box = load_boxes(gt_path, cls_filter=True)
    hyp_box = load_boxes(hyp_path, cls_filter=False)
    hyp_birth = {tid: min(fr) for tid, fr in hyp_box.items()}
    gt_vis: dict[int, dict[int, float]] = defaultdict(dict)
    for line in gt_path.read_text().splitlines():
        p = line.split(",")
        if len(p) >= 9 and int(p[6]) == 1 and int(p[7]) == 1:
            gt_vis[int(p[1])][int(p[0])] = float(p[8])

    recs = []
    for (f, _), row in events[events["Type"] == "SWITCH"].iterrows():
        g, h_new, f = int(row["OId"]), int(row["HId"]), int(f)
        prev = [(mf, mh) for mf, mh in match_hist.get(g, []) if mf < f]
        if not prev:
            continue
        last_f, h_old = prev[-1]
        match_gap = f - last_f - 1
        # crossing-swap profile: short gap + occluded before
        vis_before = gt_vis.get(g, {}).get(last_f, 1.0)
        if match_gap > max_gap or vis_before >= VIS_THRESH:
            continue

        age_new = f - hyp_birth.get(h_new, f)
        kind = "REBORN" if age_new <= REBORN_AGE else "ABSORB"

        # is h_new the spatial occluder of g at the swap? (overlaps g's GT box, more frontal)
        is_occluder = False
        gb = gt_box.get(g, {}).get(f)
        nb = hyp_box.get(h_new, {}).get(f)
        if gb and nb:
            footy_g = gb[1] + gb[3]
            footy_n = nb[1] + nb[3]
            is_occluder = iou(gb, nb) >= 0.3 and footy_n > footy_g

        # Viability: were there TWO distinct hyp boxes in the occludee's cluster at the
        # swap (→ auction reassignment can fix), or only ONE merged box (→ a detection
        # gap no association change can recover)? Count hyp tracks overlapping g's GT box.
        two_box = False
        if gb:
            near = sum(
                1
                for tid, fr in hyp_box.items()
                if (b := fr.get(f)) and iou(gb, b) >= 0.3
            )
            two_box = near >= 2

        recs.append(
            dict(
                seq=hyp_path.stem,
                kind=kind,
                gap=match_gap,
                is_occluder=is_occluder,
                age_new=age_new,
                two_box=two_box,
            )
        )
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--substrate",
        type=Path,
        default=Path("results/mamba_whole_graph_current_7seq_recheck"),
    )
    ap.add_argument("--max-gap", type=int, default=3)
    args = ap.parse_args()

    gt_root = PROJECT_ROOT / "datasets" / "MOT17" / "train"
    all_recs = []
    print(
        f"\n{'seq':<14}{'N':>4}{'REBORN':>8}{'ABSORB':>8}{'occluder?':>11}{'gap≤1':>7}"
    )
    print("─" * 52)
    for seq in SEQS:
        gt = gt_root / seq / "gt" / "gt.txt"
        hyp = args.substrate / f"{seq}.txt"
        if not gt.exists() or not hyp.exists():
            continue
        r = analyze(gt, hyp, args.max_gap)
        all_recs += r
        if not r:
            continue
        n = len(r)
        nb = sum(x["kind"] == "REBORN" for x in r)
        na = sum(x["kind"] == "ABSORB" for x in r)
        nocc = sum(x["is_occluder"] for x in r)
        ng1 = sum(x["gap"] <= 1 for x in r)
        print(
            f"{seq.replace('MOT17-', '').replace('-SDP', ''):<14}{n:>4}{nb:>8}{na:>8}{nocc:>11}{ng1:>7}"
        )

    n = len(all_recs)
    if not n:
        raise SystemExit("no crossing-swaps found")
    nb = sum(x["kind"] == "REBORN" for x in all_recs)
    na = sum(x["kind"] == "ABSORB" for x in all_recs)
    nocc = sum(x["is_occluder"] for x in all_recs)
    ng1 = sum(x["gap"] <= 1 for x in all_recs)
    print("─" * 52)
    print(f"{'OVERALL':<14}{n:>4}{nb:>8}{na:>8}{nocc:>11}{ng1:>7}")
    print(
        f"\nREBORN {100 * nb / n:.0f}% (occludee re-entered as a NEW id → birth/relink hook; "
        f"bridge should catch — why does it miss?)"
    )
    print(
        f"ABSORB {100 * na / n:.0f}% (existing track took the box; {100 * nocc / n:.0f}% is the "
        f"spatial occluder → needs OCCLUDER-side auction exclusion, not an occludee penalty)"
    )
    print(
        f"gap≤1: {100 * ng1 / n:.0f}% (1-frame separations — the bridge's hardest timing)"
    )
    # Viability split among ABSORB (the auction-fix target)
    absorb = [x for x in all_recs if x["kind"] == "ABSORB"]
    if absorb:
        tb = sum(x["two_box"] for x in absorb)
        print(
            f"\nABSORB viability: {tb}/{len(absorb)} ({100 * tb / len(absorb):.0f}%) have TWO hyp "
            f"boxes at the swap → auction reassignment can fix; the rest are a single merged box "
            f"(detection gap, association cannot recover)."
        )


if __name__ == "__main__":
    main()
