#!/usr/bin/env python3
"""Does nearest-box distance separate the TP from the FP that cst040 newly confirms?

cst040 (lower confirm_score_thresh) outputs tracks earlier than baseline. The
extra boxes split into recovered-TP (real establishing people we want) and
added-FP (the precision cost). The user's hypothesis: real establishing people
are spatially ISOLATED while false confirmations cluster next to existing boxes,
so a proximity-conditioned confirm (relax only when isolated) could keep the TP
and drop the FP.

We take the per-frame set difference (cst040 boxes not in baseline), label each
TP/FP by GT, and measure its nearest-neighbour distance normalised by box height
(same convention as the spawn kernel's birth_prox_norm_thresh). If FP-new sit at
small normalised distance and TP-new at large, a proximity gate has headroom.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torchvision  # noqa: E402


def _load(path: Path) -> dict[int, np.ndarray]:
    by: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        by[int(r[0])].append([r[2], r[3], r[2] + r[4], r[3] + r[5]])
    return {f: np.array(v) for f, v in by.items()}


def _load_gt(path: Path) -> dict[int, np.ndarray]:
    by: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        if int(r[6]) != 1 or int(r[7]) != 1 or (len(r) > 8 and r[8] < 0.1):
            continue
        by[int(r[0])].append([r[2], r[3], r[2] + r[4], r[3] + r[5]])
    return {f: np.array(v) for f, v in by.items()}


def _nn_norm_dist(box: np.ndarray, others: np.ndarray) -> float:
    """min center-distance to any other box, normalised by max(h_self,h_other)."""
    if others.shape[0] == 0:
        return float("inf")
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    h = max(box[3] - box[1], 1.0)
    ocx = (others[:, 0] + others[:, 2]) * 0.5
    ocy = (others[:, 1] + others[:, 3]) * 0.5
    oh = np.maximum(others[:, 3] - others[:, 1], 1.0)
    dist = np.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2)
    refh = np.maximum(h, oh)
    return float((dist / refh).min())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="results/conf_baseline")
    p.add_argument("--variant", default="results/conf_cst040")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    p.add_argument("--match-iou", type=float, default=0.5)
    args = p.parse_args()
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    tp_d, fp_d = [], []  # nearest-neighbour normalised distances
    for seq in seqs:
        base = _load(Path(args.baseline) / f"{seq}.txt")
        var = _load(Path(args.variant) / f"{seq}.txt")
        gt = _load_gt(Path(args.data_root) / args.split / seq / "gt" / "gt.txt")
        for frame, vb in var.items():
            if vb.shape[0] == 0:
                continue
            bb = base.get(frame, np.zeros((0, 4)))
            # new = variant box not matched (IoU) to any baseline box
            if bb.shape[0]:
                iou_vb = torchvision.ops.box_iou(
                    torch.tensor(vb, dtype=torch.float32),
                    torch.tensor(bb, dtype=torch.float32),
                ).numpy()
                is_new = iou_vb.max(axis=1) < args.match_iou
            else:
                is_new = np.ones(vb.shape[0], dtype=bool)
            gtb = gt.get(frame, np.zeros((0, 4)))
            for i in np.nonzero(is_new)[0]:
                box = vb[i]
                # TP if matches a GT
                is_tp = False
                if gtb.shape[0]:
                    iou_g = torchvision.ops.box_iou(
                        torch.tensor(box[None], dtype=torch.float32),
                        torch.tensor(gtb, dtype=torch.float32),
                    ).numpy()
                    is_tp = bool(iou_g.max() >= args.match_iou)
                others = np.delete(vb, i, axis=0)
                d = _nn_norm_dist(box, others)
                (tp_d if is_tp else fp_d).append(d)

    tp = np.array(tp_d)
    fp = np.array(fp_d)
    print(
        f"newly-confirmed boxes (cst040 \\ baseline): {len(tp) + len(fp)}  "
        f"TP={len(tp)} FP={len(fp)}  TP-rate={len(tp) / (len(tp) + len(fp)):.2f}"
    )
    print("\nnearest-neighbour normalised distance (dist / box-height):")
    for name, a in (("TP-new (want keep)", tp), ("FP-new (want drop)", fp)):
        if len(a) == 0:
            continue
        fin = a[np.isfinite(a)]
        print(
            f"  {name:20s} n={len(a):5d} isolated(inf)={np.isinf(a).mean():.2f} "
            f"median={np.median(fin):.2f} p25={np.percentile(fin, 25):.2f} "
            f"frac<0.5={np.mean(a < 0.5):.2f} frac<1.0={np.mean(a < 1.0):.2f}"
        )
    # Gate sim: keep new box only if nn_dist >= g  (relax-if-isolated). Sweep g.
    print("\nproximity gate sim — keep new box iff nn_dist>=g (TP kept / FP dropped):")
    print(
        f"  {'g':>5s} {'TP_kept':>8s} {'TP_kept%':>9s} {'FP_kept':>8s} {'FP_kept%':>9s}"
    )
    for g in (0.0, 0.5, 1.0, 1.5, 2.0):
        tpk = int((tp >= g).sum())
        fpk = int((fp >= g).sum())
        print(
            f"  {g:5.1f} {tpk:8d} {100 * tpk / len(tp):8.1f}% {fpk:8d} {100 * fpk / len(fp):8.1f}%"
        )


if __name__ == "__main__":
    main()
