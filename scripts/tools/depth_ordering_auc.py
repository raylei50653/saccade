"""Compute discrimination AUC for the depth-ordering (occlusion front/back) signal,
reusing the probe's GT crossing-event extractor. Two AUCs:
  (1) front/back discrimination AUC: how well does foot_y (and area) rank the GT-front
      above the GT-back box across all crossing events.  AUC = P(score_front > score_back)
      with ties = 0.5 (Mann-Whitney on paired samples).
  (2) decisiveness/calibration AUC: does a larger |foot_gap|/h predict the foot cue is
      correct (separating right vs wrong predictions)?
Contrast: appearance hard-pool AUC ~= 0.50 (registry #2/#32/#35).
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts/tools")
import depth_ordering_probe as P

ROOT = Path("datasets/MOT17/train")
DET = "SDP"
IOU_HI, IOU_LO, MAX_OCC, MIN_LIFE, PRE_WIN, VIS_MARGIN = 0.4, 0.3, 4, 5, 6, 0.10

ev_rows = []  # (signed_foot_score_toward_front, signed_area_score, foot_correct, foot_gap_h)
for seq in P.SEQS:
    gt_path = ROOT / f"{seq}-{DET}" / "gt" / "gt.txt"
    if not gt_path.exists():
        continue
    gt = P.load_gt(gt_path)
    evs = P.find_cross_events(gt, IOU_HI, IOU_LO, MAX_OCC, MIN_LIFE, PRE_WIN)
    for ev in evs:
        a, b = ev["a"], ev["b"]
        vmax, vmin = max(ev["vis_a"], ev["vis_b"]), min(ev["vis_a"], ev["vis_b"])
        if (vmax - vmin) < VIS_MARGIN or vmin >= 0.5:
            continue
        gt_front = a if ev["vis_a"] > ev["vis_b"] else b
        gt_back = b if gt_front == a else a
        ar_f, fy_f, h_f = P.median_geom(gt[gt_front], ev["pre"])
        ar_b, fy_b, h_b = P.median_geom(gt[gt_back], ev["pre"])
        h_ref = 0.5 * (h_f + h_b)
        foot_gap_h = abs(fy_f - fy_b) / max(h_ref, 1e-6)
        foot_correct = int(
            fy_f > fy_b
        )  # foot_y(front) should be lower in image = larger y
        area_correct = int(ar_f > ar_b)
        ev_rows.append((foot_correct, area_correct, foot_gap_h))

ev = np.array(ev_rows, float)
n = len(ev)
foot_ok = ev[:, 0]
area_ok = ev[:, 1]
gap = ev[:, 2]

# (1) front/back discrimination AUC = P(score_front > score_back), ties=0.5.
# For a paired binary decision this equals mean(correct) + 0.5*mean(tie). No ties in foot_y here.
auc_foot = foot_ok.mean()
auc_area = area_ok.mean()

# (2) decisiveness AUC: rank |foot_gap| as predictor of foot-cue correctness (Mann-Whitney U).
pos = gap[foot_ok == 1]
neg = gap[foot_ok == 0]


def mannwhitney_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    avg = sums / cnt
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


auc_gap = mannwhitney_auc(pos, neg)

print(f"n usable crossing events = {n}")
print(
    f"(1) front/back discrimination AUC  foot_y = {auc_foot:.3f}   area = {auc_area:.3f}"
)
print(
    f"(2) decisiveness AUC |foot_gap|/h -> correct = {auc_gap:.3f}  "
    f"(median gap correct={np.median(pos):.2f}h, wrong={np.median(neg):.2f}h, n_wrong={len(neg)})"
)
print("    appearance hard-pool reference AUC ~= 0.50 (registry #2/#32/#35)")
