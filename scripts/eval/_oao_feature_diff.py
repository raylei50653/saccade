#!/usr/bin/env python3
"""OAO gain attribution: characterise boxes that differ between baseline and B.

For each sequence, match baseline-output boxes and B-output boxes frame-by-frame
(greedy IoU>=0.5). A baseline box with no B counterpart is "suppressed"; a B box
with no baseline counterpart is "added". Each box is GT-labelled (TP if IoU>=0.5
to a GT box, else FP) and tagged with its inter-track IoU (occ_coeff proxy = max
IoU to any OTHER box in the same frame of the SAME tracker).

Prints, per seq: counts + mean occ-coeff of suppressed/added boxes, split by
TP/FP, vs the kept-box baseline. Confirms whether OAO drops high-overlap FPs.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "datasets" / "MOT17" / "train"
SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]


def load(path):
    fr = defaultdict(list)
    with open(path) as f:
        for ln in f:
            p = ln.split(",")
            if len(p) < 6:
                continue
            fid = int(float(p[0]))
            x, y, w, h = (float(p[i]) for i in range(2, 6))
            sc = float(p[6]) if len(p) > 6 else 1.0
            fr[fid].append([x, y, x + w, y + h, sc])
    return {k: np.array(v, dtype=np.float64) for k, v in fr.items()}


def load_gt(path):
    fr = defaultdict(list)
    with open(path) as f:
        for ln in f:
            p = ln.split(",")
            if len(p) < 9:
                continue
            cls = int(float(p[7]))
            vis = float(p[8])
            mark = int(float(p[6]))
            # pedestrian class=1, marked, ignore low-vis distractors loosely
            if cls != 1 or mark == 0:
                continue
            fid = int(float(p[0]))
            x, y, w, h = (float(p[i]) for i in range(2, 6))
            fr[fid].append([x, y, x + w, y + h, vis])
    return {k: np.array(v, dtype=np.float64) for k, v in fr.items()}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + ab - inter + 1e-9)


def occ_coeff(boxes):
    """max IoU of each box to any OTHER box in the same frame."""
    if len(boxes) <= 1:
        return np.zeros(len(boxes))
    m = iou_mat(boxes, boxes)
    np.fill_diagonal(m, 0.0)
    return m.max(axis=1)


def greedy_match(a, b, thr=0.5):
    """return matched pairs (ia, ib); unmatched a; unmatched b."""
    if len(a) == 0:
        return [], [], list(range(len(b)))
    if len(b) == 0:
        return [], list(range(len(a)))[:], []
    m = iou_mat(a, b)
    pairs = []
    ua, ub = set(range(len(a))), set(range(len(b)))
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    for i, j in order:
        if m[i, j] < thr:
            break
        if i in ua and j in ub:
            pairs.append((i, j))
            ua.discard(i)
            ub.discard(j)
    return pairs, sorted(ua), sorted(ub)


def gt_label(box, gtf):
    """TP if box matches any GT (IoU>=0.5), else FP."""
    if gtf is None or len(gtf) == 0:
        return "FP"
    m = iou_mat(box[None, :4], gtf[:, :4])[0]
    return "TP" if m.max() >= 0.5 else "FP"


def main():
    base_dir, b_dir = sys.argv[1], sys.argv[2]
    print(f"baseline={base_dir}  B={b_dir}\n")
    hdr = (
        f"{'seq':<14}{'supp':>6}{'occ':>6}{'sFP':>5}{'sTP':>5}"
        f"{'sFPocc':>8}{'sTPocc':>8}{'add':>6}{'aFP':>5}{'aTP':>5}{'keptOcc':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(int)
    for seq in SEQS:
        base = load(f"{base_dir}/{seq}.txt")
        bb = load(f"{b_dir}/{seq}.txt")
        gt = load_gt(GT_DIR / seq / "gt" / "gt.txt")
        supp_occ, supp_fp, supp_tp = [], 0, 0
        sfp_occ, stp_occ = [], []
        add_n, add_fp, add_tp = 0, 0, 0
        kept_occ = []
        for fid in set(base) | set(bb):
            a = base.get(fid, np.zeros((0, 5)))
            b = bb.get(fid, np.zeros((0, 5)))
            gtf = gt.get(fid)
            oa = occ_coeff(a)
            _, ua, ub = greedy_match(a, b)
            # kept (matched in baseline) occ ref
            matched_a = set(range(len(a))) - set(ua)
            for i in matched_a:
                kept_occ.append(oa[i])
            for i in ua:  # suppressed (in baseline, gone in B)
                supp_occ.append(oa[i])
                lab = gt_label(a[i], gtf)
                if lab == "FP":
                    supp_fp += 1
                    sfp_occ.append(oa[i])
                else:
                    supp_tp += 1
                    stp_occ.append(oa[i])
            _ob = occ_coeff(b)
            for j in ub:  # added (in B, not baseline)
                add_n += 1
                if gt_label(b[j], gtf) == "FP":
                    add_fp += 1
                else:
                    add_tp += 1

        def _mo(v):
            return np.mean(v) if v else 0.0

        print(
            f"{seq:<14}{len(supp_occ):>6}{_mo(supp_occ):>6.2f}{supp_fp:>5}{supp_tp:>5}"
            f"{_mo(sfp_occ):>8.2f}{_mo(stp_occ):>8.2f}{add_n:>6}{add_fp:>5}{add_tp:>5}"
            f"{_mo(kept_occ):>8.2f}"
        )
        tot["supp_fp"] += supp_fp
        tot["supp_tp"] += supp_tp
        tot["add_fp"] += add_fp
        tot["add_tp"] += add_tp
    print("-" * len(hdr))
    print(
        f"TOTAL suppressed: FP={tot['supp_fp']} TP={tot['supp_tp']}  "
        f"added: FP={tot['add_fp']} TP={tot['add_tp']}"
    )
    net_fp = tot["add_fp"] - tot["supp_fp"]
    net_tp = tot["add_tp"] - tot["supp_tp"]
    print(f"NET (B - baseline): FP {net_fp:+d}  TP {net_tp:+d}  (TP loss ~= FN gain)")


if __name__ == "__main__":
    main()
