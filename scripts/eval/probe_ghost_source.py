#!/usr/bin/env python3
"""Decompose low-score 'ghost' boxes by source.

A box is a 'ghost' (in probe_ghost_rate_by_score) if it has no pedestrian-GT
match at IoU>=0.5. But that bucket mixes very different things. Here we find,
for each ghost box, its best-overlapping GT *of any class* and classify:

  REAL_BADBOX : overlaps an eval pedestrian (class1 & consider) at IoU [0.1,0.5)
                -> a real person with a poor box. fw suppressing it = real FN.
  DISTRACTOR  : overlaps a person-like distractor (class 2/7/8/12, or class1
                consider=0) at IoU>=0.1 -> eval-ignored, fw-safe.
  OCCL_OBJ    : overlaps occluder/vehicle (class 3/4/5/6/9/10/11/13) -> fw-safe.
  BACKGROUND  : no GT overlap >=0.1 -> true hallucination, fw-good.

MOT17 GT cols: frame,id,x,y,w,h,consider,class,visibility.
"""
# status: experiment

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "datasets" / "MOT17" / "train"
SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]
DELTA = {
    "MOT17-02-SDP": -0.4,
    "MOT17-04-SDP": -0.1,
    "MOT17-05-SDP": -0.1,
    "MOT17-09-SDP": +2.0,
    "MOT17-10-SDP": +2.1,
    "MOT17-11-SDP": -1.8,
    "MOT17-13-SDP": -3.1,
}
SCORE_HI = 0.30  # analyze the low-score region fw acts on
IOU_MATCH = 0.5
IOU_TOUCH = 0.1
DISTRACTOR = {2, 7, 8, 12}
OCCL_OBJ = {3, 4, 5, 6, 9, 10, 11, 13}


def iou_matrix(a, b):
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    iy1 = np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    return inter / (
        (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter + 1e-9
    )


def classify_seq(out_dir, seq):
    det = np.loadtxt(Path(out_dir) / f"{seq}.txt", delimiter=",", usecols=range(7))
    gt = np.loadtxt(GT_DIR / seq / "gt" / "gt.txt", delimiter=",")
    df, dxywh, dscore = det[:, 0].astype(int), det[:, 2:6], det[:, 6]
    gf = gt[:, 0].astype(int)
    counts = {"REAL_BADBOX": 0, "DISTRACTOR": 0, "OCCL_OBJ": 0, "BACKGROUND": 0}
    total_low = 0
    for fr in np.unique(df):
        di = np.where((df == fr) & (dscore < SCORE_HI))[0]
        if len(di) == 0:
            continue
        total_low += len(di)
        gi = np.where(gf == fr)[0]
        ped = gt[gi][(gt[gi, 6] == 1) & (gt[gi, 7] == 1)]
        # matched to eval pedestrian -> not a ghost
        if len(ped):
            iou_ped = iou_matrix(dxywh[di], ped[:, 2:6])
            matched = iou_ped.max(axis=1) >= IOU_MATCH
        else:
            matched = np.zeros(len(di), dtype=bool)
        if len(gi) == 0:
            allg = np.empty((0, 9))
        else:
            allg = gt[gi]
        iou_all = iou_matrix(dxywh[di], allg[:, 2:6]) if len(allg) else None
        for k, d in enumerate(di):
            if matched[k]:
                continue  # real matched pedestrian, skip
            if iou_all is None:
                counts["BACKGROUND"] += 1
                continue
            order = np.argsort(-iou_all[k])
            assigned = None
            for j in order:
                io = iou_all[k, j]
                if io < IOU_TOUCH:
                    break
                cls = int(allg[j, 7])
                cons = int(allg[j, 6])
                if cls == 1 and cons == 1:
                    if io >= IOU_TOUCH:
                        assigned = "REAL_BADBOX"
                        break
                elif cls in DISTRACTOR or (cls == 1 and cons == 0):
                    assigned = "DISTRACTOR"
                    break
                elif cls in OCCL_OBJ:
                    assigned = "OCCL_OBJ"
                    break
            counts[assigned or "BACKGROUND"] += 1
    return counts, total_low


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/linear_crowd_fw00"
    print(
        f"Low-score boxes (score<{SCORE_HI}) ghost-source decomposition — {out_dir}\n"
    )
    hdr = (
        f"{'seq':<14}{'ΔIDF1':>7}{'lowN':>7} | {'REAL_BADBOX':>14}{'DISTRACTOR':>13}"
        f"{'OCCL_OBJ':>11}{'BACKGROUND':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for seq in SEQS:
        c, n = classify_seq(out_dir, seq)
        ghosts = sum(c.values())

        def pct(x):
            return f"{x:4d} ({100 * x / ghosts:4.0f}%)" if ghosts else f"{x:4d}   -"

        print(
            f"{seq:<14}{DELTA[seq]:>+7.1f}{n:>7} | "
            f"{pct(c['REAL_BADBOX']):>14}{pct(c['DISTRACTOR']):>13}"
            f"{pct(c['OCCL_OBJ']):>11}{pct(c['BACKGROUND']):>13}"
        )
    print("\nKey: REAL_BADBOX = real pedestrian, poor box -> fw kills real (FN risk)")
    print("     BACKGROUND/DISTRACTOR/OCCL = eval-safe -> fw suppression is free")


if __name__ == "__main__":
    main()
