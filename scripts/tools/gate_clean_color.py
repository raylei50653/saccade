#!/usr/bin/env python3
"""Test the 'use color ReID only on non-occluded relink candidates' gate.

Reuses GT visibility (oracle occlusion) joined to relink_candidates, same as
probe_relink_occlusion_signal.py. Splits candidates into CLEAN (both endpoints
high GT visibility) vs OCCLUDED, then reports:
  (1) population & true-bridge base rate of the CLEAN subset
  (2) gap distribution clean vs occluded  (claim: clean ~= short gap)
  (3) geometry separability inside clean vs occluded (does clean even need color?)
"""
# status: diagnostic

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "datasets" / "MOT17" / "train"
CSV = ROOT / "scripts/tools/out/relink_candidates.csv"


def load_gt_vis(seq: str) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    p = GT_DIR / seq / "gt" / "gt.txt"
    for line in p.read_text().splitlines():
        c = line.strip().split(",")
        if len(c) < 9 or int(c[6]) != 1 or int(c[7]) != 1:
            continue
        out[(int(c[0]), int(c[1]))] = float(c[8])
    return out


def vis_window(vis, gid, frame, lo, hi):
    vals = [vis[(f, gid)] for f in range(frame + lo, frame + hi + 1) if (f, gid) in vis]
    return float(np.min(vals)) if vals else float("nan")


rows = list(csv.DictReader(CSV.open()))
viscache: dict[str, dict] = {}
gt_match, gap, bdist, dhist, vmin = [], [], [], [], []
for r in rows:
    if r["gt_valid"] != "1":
        continue
    seq = r["seq"]
    vis = viscache.setdefault(seq, load_gt_vis(seq))
    gl, gc = int(r["gt_lost"]), int(r["gt_cand"])
    lf, cf = int(r["lost_last_frame"]), int(r["cand_first_frame"])
    # vis just before lost disappears, and just after cand appears
    v_lost = vis_window(vis, gl, lf, -3, 0)
    v_cand = vis_window(vis, gc, cf, 0, 3)
    vm = np.nanmin([v_lost, v_cand])
    gt_match.append(int(r["gt_match"]))
    gap.append(int(r["gap"]))
    bdist.append(float(r["bridge_dist"]))
    dhist.append(float(r["dist_h"]))
    vmin.append(vm)

gt_match = np.array(gt_match)
gap = np.array(gap)
bdist = np.array(bdist)
dhist = np.array(dhist)
vmin = np.array(vmin)

ok = ~np.isnan(vmin)
print(f"valid w/ vis: {ok.sum()}   true bridges: {gt_match[ok].sum()}")

for thr in (0.5, 0.7, 0.9):
    clean = ok & (vmin >= thr)
    occ = ok & (vmin < thr)
    nt_c = gt_match[clean].sum()
    nt_o = gt_match[occ].sum()
    print(f"\n===== CLEAN gate: min endpoint visibility >= {thr} =====")
    print(
        f"  CLEAN subset : {clean.sum():5d} cand  ({100 * clean.sum() / ok.sum():.1f}% of pool)"
        f"  true={nt_c}  base_rate={100 * nt_c / max(clean.sum(), 1):.2f}%"
    )
    print(
        f"  OCCL  subset : {occ.sum():5d} cand  true={nt_o}  base_rate={100 * nt_o / max(occ.sum(), 1):.2f}%"
    )
    print(
        f"  share of all TRUE bridges that are CLEAN: {100 * nt_c / max(gt_match[ok].sum(), 1):.1f}%"
    )
    print(
        f"  gap median  clean={np.median(gap[clean]):.0f}  occ={np.median(gap[occ]):.0f}"
        f"   (clean short-gap<=5: {100 * np.mean(gap[clean] <= 5):.0f}%  occ: {100 * np.mean(gap[occ] <= 5):.0f}%)"
    )
    for name, feat in (("bridge_dist", bdist), ("dist_h", dhist)):
        for tag, msk in (("clean", clean), ("occ", occ)):
            lab = gt_match[msk]
            if len(np.unique(lab)) < 2 or len(lab) < 20:
                continue
            auc = roc_auc_score(lab, -feat[msk])
            print(f"    geom {name:11s} | {tag:5s}  AUC={auc:.3f}  n={msk.sum()}")
