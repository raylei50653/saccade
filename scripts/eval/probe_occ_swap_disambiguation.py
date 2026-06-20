#!/usr/bin/env python3
"""At occluder-ABSORB crossing-swaps: does an OCCLUSION signal disambiguate the
occludee's box where foot-line geometry can't?

Context
-------
72% of crossing-swaps are ABSORB (an existing track grabs the box); 63% the absorber
IS the spatial occluder (analyze_crossing_swaps). The fix is OCCLUDER-side exclusion:
the occluder track must be penalized away from the OCCLUDEE's box. The current
occ_state does this geometrically (occ_front_ttl + under-foot penalty: penalize the
detection whose foot is ABOVE the occluder's foot = the further/occludee box). That
foot-line call is 92-100% accurate at the occlusion PEAK (occ_event_values) -- but
72% of swaps happen at gap<=1 SEPARATION, where the two feet converge and foot-line
geometry weakens. The user's idea: drive the exclusion by a per-detection OCCLUSION
signal (occludee = the more-occluded box) instead of foot-line.

This probe tests, at each occluder-ABSORB swap frame, on the TWO competing GT boxes
(occludee g's box vs the occluder's own box):
  foot rule : occludee = higher foot (further)        -> correct iff foot_B > foot_A
  occ  rule : occludee = more occluded (lower vis)     -> correct iff vis_B  < vis_A
GT visibility = oracle occlusion signal (ceiling a visibility head could give).

Decisive read: in the foot-AMBIGUOUS subset (small |foot_A-foot_B|), does the occ
rule still pick the occludee correctly? If yes -> the per-detection occlusion signal
fills the geometry blind spot (visibility head has a home). If no -> at separation
both boxes are already visible / occ is non-decisive too -> close the direction.

Usage
-----
  .venv/bin/python scripts/eval/probe_occ_swap_disambiguation.py \
      --substrate results/mamba_whole_graph_current_7seq_recheck
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm  # noqa: E402

from analyze_crossing_swaps import (  # noqa: E402
    REBORN_AGE,
    VIS_THRESH,
    iou,
    load_boxes,
)

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]


def collect(gt_path: Path, hyp_path: Path, max_gap: int) -> list[dict]:
    gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5)
    events = acc.events

    match_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)  # OId -> [(f,HId)]
    hyp_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)  # HId -> [(f,OId)]
    for (f, _), row in events.iterrows():
        if row["Type"] != "MATCH":
            continue
        if isinstance(row["OId"], float) and np.isnan(row["OId"]):
            continue
        match_hist[int(row["OId"])].append((int(f), int(row["HId"])))
        hyp_hist[int(row["HId"])].append((int(f), int(row["OId"])))

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
        last_f, _ = prev[-1]
        match_gap = f - last_f - 1
        vis_before = gt_vis.get(g, {}).get(last_f, 1.0)
        if match_gap > max_gap or vis_before >= VIS_THRESH:
            continue
        if (f - hyp_birth.get(h_new, f)) <= REBORN_AGE:
            continue  # REBORN, not ABSORB

        # occluder identity = the GT id h_new was tracking just before the swap
        occ_prev = [(mf, og) for mf, og in hyp_hist.get(h_new, []) if mf < f]
        if not occ_prev:
            continue
        occluder_gt = occ_prev[-1][1]
        if occluder_gt == g:
            continue

        box_B = gt_box.get(g, {}).get(f)  # occludee (true exclusion target)
        box_A = gt_box.get(occluder_gt, {}).get(f)  # occluder's own box
        if not box_B or not box_A:
            continue
        # require this really be the occluder-ABSORB geometry (boxes overlap)
        if iou(box_A, box_B) < 0.3:
            continue

        foot_A = box_A[1] + box_A[3]
        foot_B = box_B[1] + box_B[3]
        h_ref = 0.5 * (box_A[3] + box_B[3])
        vis_A = gt_vis.get(occluder_gt, {}).get(f, 1.0)
        vis_B = gt_vis.get(g, {}).get(f, 1.0)

        recs.append(
            dict(
                seq=hyp_path.stem,
                foot_gap=abs(foot_A - foot_B) / max(h_ref, 1e-6),
                vis_gap=abs(vis_A - vis_B),
                # occludee is behind => further => higher in image => SMALLER foot (y+h).
                # occluder foot > occludee foot (cf. analyze_crossing_swaps is_occluder).
                foot_correct=int(foot_B < foot_A),  # geom: occludee = smaller foot
                vis_correct=int(vis_B < vis_A),  # occ: occludee = lower visibility
                vis_A=vis_A,
                vis_B=vis_B,
            )
        )
    return recs


def report(recs: list[dict], foot_amb: float, vis_dec: float) -> None:
    n = len(recs)
    if n == 0:
        print("no occluder-ABSORB events collected")
        return
    fc = np.array([r["foot_correct"] for r in recs])
    vc = np.array([r["vis_correct"] for r in recs])
    fg = np.array([r["foot_gap"] for r in recs])
    vg = np.array([r["vis_gap"] for r in recs])

    print(f"\n=== occluder-ABSORB swap disambiguation (N={n}) ===")
    print(f"  foot-line rule correct : {100 * fc.mean():.0f}%")
    print(f"  occlusion  rule correct: {100 * vc.mean():.0f}%")

    amb = fg < foot_amb  # geometry-ambiguous: feet nearly aligned
    clr = ~amb
    print(f"\n  by foot-line decisiveness (ambiguous = foot_gap < {foot_amb}):")
    print(f"{'subset':18s}{'n':>5s}{'foot_ok':>9s}{'occ_ok':>8s}{'occ_decisive':>14s}")
    for name, m in [("foot-CLEAR", clr), ("foot-AMBIG", amb)]:
        k = int(m.sum())
        if k == 0:
            print(f"{name:18s}{k:>5d}{'-':>9s}{'-':>8s}{'-':>14s}")
            continue
        occ_dec = float((vg[m] >= vis_dec).mean())
        print(
            f"{name:18s}{k:>5d}{100 * fc[m].mean():>8.0f}%{100 * vc[m].mean():>7.0f}%"
            f"{100 * occ_dec:>13.0f}%"
        )

    # The money number: in the foot-ambiguous subset, occ rule accuracy among the
    # events where occ is decisive (a usable per-box occlusion gap exists).
    usable = amb & (vg >= vis_dec)
    ku = int(usable.sum())
    print(
        f"\n  foot-AMBIG & occ-decisive: {ku}/{n} events ({100 * ku / n:.0f}% of all)"
    )
    if ku:
        print(
            f"    -> occlusion rule picks occludee correctly in {100 * vc[usable].mean():.0f}%"
        )
    print(
        "\n  (oracle GT visibility = ceiling a visibility head ~0.79 AUC could approach;"
        " uses GT boxes, real tracker dets are noisier)"
    )


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
    ap.add_argument(
        "--foot-amb",
        type=float,
        default=0.10,
        help="foot_gap below this = geometry ambiguous (same-depth)",
    )
    ap.add_argument(
        "--vis-dec",
        type=float,
        default=0.20,
        help="vis_gap above this = occlusion signal is decisive",
    )
    args = ap.parse_args()

    gt_root = PROJECT_ROOT / "datasets" / "MOT17" / "train"
    all_recs: list[dict] = []
    for seq in SEQS:
        hyp = args.substrate / f"{seq}.txt"
        gt = gt_root / seq / "gt" / "gt.txt"
        if not hyp.exists() or not gt.exists():
            continue
        all_recs.extend(collect(gt, hyp, args.max_gap))
    report(all_recs, args.foot_amb, args.vis_dec)


if __name__ == "__main__":
    main()
