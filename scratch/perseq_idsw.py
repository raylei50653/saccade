import numpy as _np

if not hasattr(_np, "asfarray"):
    _np.asfarray = lambda a, dtype=_np.float64: _np.asarray(a, dtype=dtype)
import glob
import os

import motmetrics as mm
import numpy as np

GT = "datasets/MOT17/train"


def load_mot(path):
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        p = ln.strip().split(",")
        if len(p) < 6:
            continue
        fr = int(float(p[0]))
        tid = int(float(p[1]))
        x, y, w, h = map(float, p[2:6])
        d.setdefault(fr, []).append((tid, x, y, w, h))
    return d


def eval_seq(seq, res):
    gtf = f"{GT}/{seq}/gt/gt.txt"
    gt = {}
    for ln in open(gtf):
        p = ln.strip().split(",")
        fr = int(p[0])
        tid = int(p[1])
        x, y, w, h = map(float, p[2:6])
        mark = int(p[6])
        cls = int(p[7])
        if mark == 0 or cls != 1:
            continue
        gt.setdefault(fr, []).append((tid, x, y, w, h))
    rs = load_mot(res)
    acc = mm.MOTAccumulator(auto_id=True)
    for fr in sorted(set(gt) | set(rs)):
        g = gt.get(fr, [])
        r = rs.get(fr, [])
        gi = [t[0] for t in g]
        ri = [t[0] for t in r]
        gb = np.array([t[1:] for t in g]) if g else np.empty((0, 4))
        rb = np.array([t[1:] for t in r]) if r else np.empty((0, 4))
        dist = (
            mm.distances.iou_matrix(gb, rb, max_iou=0.5)
            if len(g) and len(r)
            else np.empty((len(g), len(r)))
        )
        acc.update(gi, ri, dist)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["num_switches", "idf1", "recall"], name=seq)
    return int(s["num_switches"][seq])


seqs = [
    os.path.basename(p)
    for p in sorted(glob.glob("scratch/ab_runs/frozen_v2_repro/MOT17-*-SDP.txt"))
]
seqs = [s[:-4] for s in seqs]
print(f"{'seq':14} {'+GMC+bridge':>12} {'full':>6} {'Δ':>5}")
tot_a = tot_b = 0
for seq in sorted(seqs):
    a = eval_seq(seq, f"scratch/ab_runs/gmc_bridge/{seq}.txt")
    b = eval_seq(seq, f"scratch/ab_runs/frozen_v2_repro/{seq}.txt")
    tot_a += a
    tot_b += b
    print(f"{seq:14} {a:12d} {b:6d} {b - a:+5d}")
print(f"{'TOTAL':14} {tot_a:12d} {tot_b:6d} {tot_b - tot_a:+5d}")
