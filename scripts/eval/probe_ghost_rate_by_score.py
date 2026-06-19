#!/usr/bin/env python3
"""Ghost-rate-by-score probe.

For each MOT17 sequence, bin the *baseline* tracker output boxes by detection
score (col 7) and measure ghost-rate = fraction of boxes with NO pedestrian-GT
match (IoU>=0.5). The fuse_score_weight penalty acts only on low-score boxes;
if low-score boxes are mostly ghosts -> penalty helps (FP down), if mostly real
-> penalty hurts (FN up). We test whether low-score ghost-rate separates the
fw-helps seqs {09,10} from the fw-hurts seqs {11,13}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "datasets" / "MOT17" / "train"
SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]
DELTA_IDF1 = {  # fw0.1 - fw0.0, from per-seq extract
    "MOT17-02-SDP": -0.4,
    "MOT17-04-SDP": -0.1,
    "MOT17-05-SDP": -0.1,
    "MOT17-09-SDP": +2.0,
    "MOT17-10-SDP": +2.1,
    "MOT17-11-SDP": -1.8,
    "MOT17-13-SDP": -3.1,
}
BINS = [0.075, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
IOU_THR = 0.5


def load_boxes(path, score_col=None):
    rows = np.loadtxt(path, delimiter=",", usecols=range(7))
    frame = rows[:, 0].astype(int)
    xywh = rows[:, 2:6]
    score = rows[:, 6]
    return frame, xywh, score


def load_gt(seq):
    rows = np.loadtxt(GT_DIR / seq / "gt" / "gt.txt", delimiter=",")
    # cols: frame,id,x,y,w,h,consider,class,visibility
    keep = (rows[:, 6] == 1) & (rows[:, 7] == 1)  # consider & pedestrian
    rows = rows[keep]
    return rows[:, 0].astype(int), rows[:, 2:6]


def iou_matrix(a, b):
    # a:[N,4] b:[M,4] xywh
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def seq_ghost(out_dir, seq):
    f, xywh, score = load_boxes(Path(out_dir) / f"{seq}.txt")
    gf, gxywh = load_gt(seq)
    is_ghost = np.ones(len(f), dtype=bool)
    for fr in np.unique(f):
        di = np.where(f == fr)[0]
        gi = np.where(gf == fr)[0]
        if len(gi) == 0:
            continue
        iou = iou_matrix(xywh[di], gxywh[gi])
        # a det is "real" if it has any GT with IoU>=thr
        is_ghost[di] = iou.max(axis=1) < IOU_THR if iou.size else True
    return score, is_ghost


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/linear_crowd_fw00"
    print(f"Baseline boxes: {out_dir}  (IoU>={IOU_THR}, GT=pedestrian&consider)\n")
    hdr = f"{'seq':<14}{'ΔIDF1':>7} | " + "".join(
        f"{f'[{BINS[i]:.2f},{BINS[i + 1]:.2f})':>14}" for i in range(len(BINS) - 1)
    )
    print(hdr)
    print("-" * len(hdr))
    low_ghost = {}
    for seq in SEQS:
        score, ghost = seq_ghost(out_dir, seq)
        line = f"{seq:<14}{DELTA_IDF1[seq]:>+7.1f} | "
        for i in range(len(BINS) - 1):
            m = (score >= BINS[i]) & (score < BINS[i + 1])
            n = int(m.sum())
            gr = float(ghost[m].mean()) * 100 if n else float("nan")
            line += f"{f'{gr:4.0f}% (n={n})':>14}"
            if i == 0:  # lowest bin
                low_ghost[seq] = (gr, n)
        print(line)
    print("\n=== Low-score bin [0.075,0.20) ghost-rate vs ΔIDF1 ===")
    for seq in sorted(SEQS, key=lambda s: -low_ghost[s][0]):
        gr, n = low_ghost[seq]
        tag = (
            "GO "
            if DELTA_IDF1[seq] > 1
            else ("NOGO" if DELTA_IDF1[seq] < -1 else "  · ")
        )
        print(
            f"  {tag} {seq:<14} ghost={gr:5.1f}%  n={n:<5}  ΔIDF1={DELTA_IDF1[seq]:+.1f}"
        )


if __name__ == "__main__":
    main()
