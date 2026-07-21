#!/usr/bin/env python3
"""Redundancy probe for low-score REAL_BADBOX boxes.

For each low-score (<0.3) box that overlaps a real pedestrian at IoU[0.1,0.5)
(a 'REAL_BADBOX'), assign it to its best pedestrian GT and ask: is that same
GT *also* covered by a HIGH-score (>=0.5) box at IoU>=0.5 this frame?

  REDUNDANT : yes -> the low box is a duplicate/fragment; fw suppressing it is
              free (the high-score box keeps the target).
  UNIQUE    : no  -> the low box is the only decent evidence for that target;
              fw suppressing it = lose the target (FN).

Hypothesis: MOT17-10 (fw GO) -> mostly REDUNDANT; MOT17-13 (fw NOGO) -> UNIQUE.
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
LOW_HI = 0.30
HIGH_LO = 0.50
IOU_MATCH = 0.5
IOU_TOUCH = 0.1


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
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


def analyze(out_dir, seq):
    det = np.loadtxt(Path(out_dir) / f"{seq}.txt", delimiter=",", usecols=range(7))
    gt = np.loadtxt(GT_DIR / seq / "gt" / "gt.txt", delimiter=",")
    df, dxywh, dscore = det[:, 0].astype(int), det[:, 2:6], det[:, 6]
    gf = gt[:, 0].astype(int)
    redundant = unique = 0
    for fr in np.unique(df):
        gi = np.where(gf == fr)[0]
        ped = gt[gi][(gt[gi, 6] == 1) & (gt[gi, 7] == 1)]
        if len(ped) == 0:
            continue
        pg = ped[:, 2:6]
        di = np.where(df == fr)[0]
        low = di[dscore[di] < LOW_HI]
        high = di[dscore[di] >= HIGH_LO]
        # which ped GTs are covered by a high-score box
        iou_hg = iou_matrix(dxywh[high], pg)
        gt_covered = (
            (iou_hg >= IOU_MATCH).any(axis=0) if len(high) else np.zeros(len(pg), bool)
        )
        # low boxes that are REAL_BADBOX: best ped IoU in [touch, match)
        iou_lg = iou_matrix(dxywh[low], pg)
        for k in range(len(low)):
            if len(pg) == 0:
                continue
            j = int(np.argmax(iou_lg[k]))
            io = iou_lg[k, j]
            if IOU_TOUCH <= io < IOU_MATCH:  # REAL_BADBOX
                if gt_covered[j]:
                    redundant += 1
                else:
                    unique += 1
    return redundant, unique


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/linear_crowd_fw00"
    print(f"REAL_BADBOX low-score redundancy — {out_dir}")
    print(f"(low<{LOW_HI}, high>={HIGH_LO}, covered IoU>={IOU_MATCH})\n")
    hdr = f"{'seq':<14}{'ΔIDF1':>7} | {'REDUNDANT':>16}{'UNIQUE':>16}{'%unique':>9}"
    print(hdr)
    print("-" * len(hdr))
    for seq in SEQS:
        r, u = analyze(out_dir, seq)
        tot = r + u
        pu = 100 * u / tot if tot else float("nan")
        tag = "GO " if DELTA[seq] > 1 else ("NOGO" if DELTA[seq] < -1 else "  · ")
        print(
            f"{tag} {seq:<10}{DELTA[seq]:>+7.1f} | {f'{r}':>16}{f'{u}':>16}{pu:>8.0f}%"
        )
    print("\nHigh %unique -> fw kills irreplaceable evidence -> fw hurts.")


if __name__ == "__main__":
    main()
