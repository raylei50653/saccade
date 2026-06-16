#!/usr/bin/env python3
"""Do cst040's new confirmations sit on a PREVIOUS-frame confirmed track?

Refinement of analyze_confirm_proximity: instead of distance to same-frame
boxes (which failed to separate), measure each newly-confirmed box's relation
to the PREVIOUS frame's already-confirmed tracks (= baseline output at f-1).
Hypothesis: a false confirmation is a duplicate/fragment of an already-tracked
person, so it overlaps a prev-frame confirmed box (high IoU / small distance);
a genuine new person enters empty space (low IoU / large distance).

If TP-new and FP-new separate on max_iou_prev, a gate "relax confirm only when
the candidate does NOT overlap a previously-confirmed track" has headroom.
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


def _iou(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    return torchvision.ops.box_iou(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    ).numpy()


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

    tp_iou, fp_iou = [], []  # max IoU to previous-frame confirmed (baseline) boxes
    for seq in seqs:
        base = _load(Path(args.baseline) / f"{seq}.txt")
        var = _load(Path(args.variant) / f"{seq}.txt")
        gt = _load_gt(Path(args.data_root) / args.split / seq / "gt" / "gt.txt")
        for frame, vb in var.items():
            if vb.shape[0] == 0:
                continue
            cur_base = base.get(frame, np.zeros((0, 4)))
            is_new = (
                _iou(vb, cur_base).max(axis=1) < args.match_iou
                if cur_base.shape[0]
                else np.ones(vb.shape[0], dtype=bool)
            )
            prev_conf = base.get(frame - 1, np.zeros((0, 4)))  # confirmed @ f-1
            gtb = gt.get(frame, np.zeros((0, 4)))
            iou_prev_all = _iou(vb, prev_conf)
            for i in np.nonzero(is_new)[0]:
                is_tp = (
                    bool(_iou(vb[i][None], gtb).max() >= args.match_iou)
                    if gtb.shape[0]
                    else False
                )
                miou = float(iou_prev_all[i].max()) if prev_conf.shape[0] else 0.0
                (tp_iou if is_tp else fp_iou).append(miou)

    tp = np.array(tp_iou)
    fp = np.array(fp_iou)
    print(
        f"newly-confirmed boxes: {len(tp) + len(fp)}  TP={len(tp)} FP={len(fp)} "
        f"TP-rate={len(tp) / (len(tp) + len(fp)):.2f}"
    )
    print("\nmax IoU to a PREVIOUS-frame confirmed track:")
    for name, a in (("TP-new (want keep)", tp), ("FP-new (want drop)", fp)):
        print(
            f"  {name:20s} n={len(a):5d} median={np.median(a):.2f} "
            f"mean={a.mean():.2f} frac=0(empty space)={np.mean(a < 1e-6):.2f} "
            f"frac>=0.3={np.mean(a >= 0.3):.2f} frac>=0.5={np.mean(a >= 0.5):.2f}"
        )
    print("\ngate sim — relax confirm only if max_iou_prev < t (keep TP / drop FP):")
    print(f"  {'t':>5s} {'TP_kept%':>9s} {'FP_dropped%':>12s} {'kept TP-rate':>13s}")
    for t in (1.01, 0.7, 0.5, 0.3, 0.1):
        tpk = int((tp < t).sum())
        fpk = int((fp < t).sum())
        kept = tpk + fpk
        rate = tpk / kept if kept else float("nan")
        print(
            f"  {t:5.2f} {100 * tpk / len(tp):8.1f}% {100 * (len(fp) - fpk) / len(fp):11.1f}% "
            f"{rate:12.2f}"
        )


if __name__ == "__main__":
    main()
