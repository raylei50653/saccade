#!/usr/bin/env python3
"""Does an explicit OCCLUSION signal separate true vs false relink bridges?

Motivation
----------
The relink/bridge decision (is this lost-track -> candidate pair the same person?)
currently relies on motion-residual / geometry, whose discriminative power is
~chance in the operating region (analyze_bidir_relink: all geom features AUC~0.55;
appearance + mamba-feature-as-identity both NO-GO). The relink is effectively
guessing -- it has NO explicit "was this gap an occlusion?" signal.

We just showed (probe_occ_activation_separability) that the head ACTIVATION decodes
occlusion at AUC 0.79 (non-geometric). Occlusion-state is NOT identity, so it dodges
the ReID wall. This probe asks the prerequisite question with the ORACLE occlusion
signal (GT visibility = the ceiling a visibility head could provide):

  Does GT visibility at the bridge endpoints separate TRUE bridges (gt_match=1, same
  person, occlusion-driven gap) from FALSE bridges (gt_match=0, different people)?

If even the oracle occlusion signal is ~chance here, a visibility head is useless for
relink. If it beats the geometric features, relink is where the visibility head pays.

Input: scripts/tools/out/relink_candidates.csv (cols incl. gt_match, gt_valid,
seq, gt_lost, gt_cand, lost_last_frame, cand_first_frame, gap, bridge_dist, ...).

Usage
-----
  .venv/bin/python scripts/tools/probe_relink_occlusion_signal.py
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "datasets" / "MOT17" / "train"


def load_gt_vis(seq: str) -> dict[tuple[int, int], float]:
    """{(frame, gt_id): visibility} for class-1 pedestrian GT."""
    out: dict[tuple[int, int], float] = {}
    p = GT_DIR / seq / "gt" / "gt.txt"
    for line in p.read_text().splitlines():
        c = line.strip().split(",")
        if len(c) < 9:
            continue
        if int(c[6]) != 1 or int(c[7]) != 1:
            continue
        out[(int(c[0]), int(c[1]))] = float(c[8])
    return out


def vis_window(vis: dict, gid: int, frame: int, lo: int, hi: int) -> float:
    """min visibility of gid over [frame+lo, frame+hi] (NaN if no GT there)."""
    vals = [vis[(f, gid)] for f in range(frame + lo, frame + hi + 1) if (f, gid) in vis]
    return float(np.min(vals)) if vals else float("nan")


def auc_safe(label: np.ndarray, feat: np.ndarray, higher_is_true: bool) -> tuple:
    m = ~np.isnan(feat)
    if m.sum() < 20 or len(np.unique(label[m])) < 2:
        return float("nan"), int(m.sum())
    s = feat[m] if higher_is_true else -feat[m]
    return float(roc_auc_score(label[m], s)), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="scripts/tools/out/relink_candidates.csv")
    ap.add_argument(
        "--hard-bridge-dist",
        type=float,
        default=8.0,
        help="operating region: candidates with bridge_dist below this",
    )
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / args.csv)))
    vis_cache: dict[str, dict] = {}

    recs = []
    for r in rows:
        if int(r["gt_valid"]) != 1:
            continue
        seq = r["seq"]
        if seq not in vis_cache:
            vis_cache[seq] = load_gt_vis(seq)
        vis = vis_cache[seq]
        gl, gc = int(r["gt_lost"]), int(r["gt_cand"])
        lf, cf = int(r["lost_last_frame"]), int(r["cand_first_frame"])
        label = int(r["gt_match"])  # 1 = true bridge (same person)
        # Oracle occlusion features at the bridge endpoints:
        v_lost_last = vis.get((lf, gl), float("nan"))  # vis when track vanished
        v_lost_enter = vis_window(vis, gl, lf, -4, 0)  # min vis entering the gap
        v_cand_first = vis.get((cf, gc), float("nan"))  # vis when candidate appeared
        v_cand_exit = vis_window(vis, gc, cf, 0, 4)  # min vis leaving the gap
        v_endpoints_min = np.nanmin([v_lost_enter, v_cand_exit])
        recs.append(
            {
                "label": label,
                "bridge_dist": float(r["bridge_dist"]),
                "fwd_resid": float(r["fwd_resid"]),
                "dist_h": float(r["dist_h"]),
                "gap": int(r["gap"]),
                "v_lost_last": v_lost_last,
                "v_lost_enter": v_lost_enter,
                "v_cand_first": v_cand_first,
                "v_cand_exit": v_cand_exit,
                "v_endpoints_min": v_endpoints_min,
            }
        )

    label = np.array([x["label"] for x in recs])
    bd = np.array([x["bridge_dist"] for x in recs])
    n_true = int(label.sum())
    print(
        f"valid candidates: {len(recs)}  true bridges: {n_true} "
        f"({100 * n_true / len(recs):.1f}%)  false: {len(recs) - n_true}"
    )

    # Operating region: candidates the relink would actually consider (small bridge_dist).
    hard = bd < args.hard_bridge_dist
    print(
        f"hard pool (bridge_dist<{args.hard_bridge_dist}): {int(hard.sum())} "
        f"candidates, true={int(label[hard].sum())} "
        f"({100 * label[hard].mean():.1f}% base rate)"
    )

    # AUC convention: occlusion features -> LOWER visibility should indicate a TRUE
    # (occlusion-driven) bridge, so higher_is_true=False. Geometry -> lower dist =
    # true, also higher_is_true=False.
    feats = [
        ("bridge_dist (geom)", "bridge_dist", False),
        ("fwd_resid  (geom)", "fwd_resid", False),
        ("dist_h     (geom)", "dist_h", False),
        ("v_lost_last  (occ)", "v_lost_last", False),
        ("v_lost_enter (occ)", "v_lost_enter", False),
        ("v_cand_exit  (occ)", "v_cand_exit", False),
        ("v_endpoints_min(occ)", "v_endpoints_min", False),
    ]
    for pool_name, mask in [
        ("ALL valid", np.ones(len(recs), bool)),
        ("HARD pool", hard),
    ]:
        print(f"\n=== separability of TRUE vs FALSE bridge | {pool_name} ===")
        print(
            f"{'feature':22s}{'AUC':>8s}{'n':>9s}{'true_mean':>11s}{'false_mean':>12s}"
        )
        lab = label[mask]
        for disp, key, hit in feats:
            col = np.array([x[key] for x in recs])[mask]
            a, n = auc_safe(lab, col, hit)
            mt = np.nanmean(col[lab == 1]) if (lab == 1).any() else float("nan")
            mf = np.nanmean(col[lab == 0]) if (lab == 0).any() else float("nan")
            print(f"{disp:22s}{a:>8.3f}{n:>9d}{mt:>11.3f}{mf:>12.3f}")


if __name__ == "__main__":
    main()
