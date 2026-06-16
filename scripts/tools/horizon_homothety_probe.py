#!/usr/bin/env python3
"""Horizon-convergence probe via VERTEX HOMOTHETY (design §3) — does the
construction "connect the 4 corresponding bbox corners of a pair, intersect to a
vanishing point, collect over pairs, fit a horizon line" actually converge?

Construction (the intended method, NOT scale-field regression)
--------------------------------------------------------------
For two boxes i, j viewed as a perspective rescaling of an upright object on the
ground plane, the four corner-correspondence lines are ground-parallel lines at
the box-top / box-bottom heights, so they all converge to ONE vanishing point
V_ij that lies on the horizon:

      TL_i ── TL_j
      TR_i ── TR_j   ⟶  concur at  V_ij  (on horizon)
      BL_i ── BL_j
      BR_i ── BR_j

Real boxes are not exact homothets, so the 4 lines do not meet perfectly. We take
the LEAST-SQUARES intersection (point minimising the sum of squared perpendicular
distances to the 4 lines) and keep the RMS residual as a per-pair concurrence /
homothety-quality signal (design §3.3 warns this is jitter-sensitive — that is
exactly what we are measuring).

Collect V_ij over all gated pairs, robust-fit a horizon line through them, and
report convergence:
  * per-pair concurrence RMS (do the 4 corners actually correspond?)
  * cross-pair collinearity: inlier ratio + total-least-squares residual
  * bootstrap stability of horizon (center-y / H) and tilt angle

Gates (design §4)
-----------------
  * scale ratio max(h_i,h_j)/min >= --ratio  (near-equal -> lines ~parallel ->
    V flies to infinity; §3.3). Default 1.15 (min), 1.30 is steadier.
  * drop V that lands absurdly far off-frame (near-parallel residue).

GT tracklet medians are used (most-favourable upper bound), same gates as
horizon_convergence_probe.py. If the homothety construction does not converge on
clean GT, it cannot on detections.

Usage
-----
  .venv/bin/python scripts/tools/horizon_homothety_probe.py
  .venv/bin/python scripts/tools/horizon_homothety_probe.py --ratio 1.30 --boot 200
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

SEQS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]
CAM = {
    "MOT17-02": "static elevated",
    "MOT17-04": "static high",
    "MOT17-05": "moving low",
    "MOT17-09": "static low",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}
IMG_H = {s: 480 if s == "MOT17-05" else 1080 for s in SEQS}
IMG_W = {s: 640 if s == "MOT17-05" else 1920 for s in SEQS}


def load_gt(
    path: Path,
) -> dict[int, dict[int, tuple[float, float, float, float, float]]]:
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 9:
            continue
        frame, gid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf, cls, vis = int(p[6]), int(p[7]), float(p[8])
        if conf != 1 or cls != 1:
            continue
        by_id[gid][frame] = (x, y, w, h, vis)
    return dict(by_id)


def median_boxes(gt, seq, args):
    """Return (N,4) median bbox [x1,y1,x2,y2] per gated tracklet, and counts."""
    H = IMG_H[seq]
    boxes = []
    n_total = 0
    rej = defaultdict(int)
    for gid, frames in gt.items():
        n_total += 1
        if len(frames) < args.min_life:
            rej["short"] += 1
            continue
        rows = np.array([list(frames[f]) for f in sorted(frames)])
        x, y, w, h, vis = rows.T
        h_med = float(np.median(h))
        if h_med <= 0 or float(np.std(h)) / h_med > args.h_var:
            rej["unstable_h"] += 1
            continue
        if float(np.median(vis)) < args.min_vis:
            rej["occluded"] += 1
            continue
        if h_med < args.min_h or h_med > args.max_h_frac * H:
            rej["height"] += 1
            continue
        ar = float(np.median(w)) / h_med
        if not (args.ar_lo <= ar <= args.ar_hi):
            rej["aspect"] += 1
            continue
        x1 = float(np.median(x))
        y1 = float(np.median(y))
        x2 = float(np.median(x + w))
        y2 = float(np.median(y + h))
        if y1 < args.edge:
            rej["top_edge"] += 1
            continue
        if y2 > H - args.edge:
            rej["bottom_edge"] += 1
            continue
        boxes.append((x1, y1, x2, y2))
    return np.array(boxes) if boxes else np.empty((0, 4)), n_total, dict(rej)


def corners(box):
    """4 corners TL,TR,BL,BR of [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])


def lsq_intersection(box_i, box_j):
    """Least-squares intersection of the 4 corner-correspondence lines.

    Returns (Vx, Vy, rms_residual_px). Lines normalised so a^2+b^2=1, so the
    residual is the RMS perpendicular distance of V to the 4 lines (px).
    """
    ci, cj = corners(box_i), corners(box_j)
    # line through (p1,p2) = cross([x1,y1,1],[x2,y2,1]) -> (a,b,c)
    sumaa = sumab = sumbb = sumac = sumbc = 0.0
    abc = []
    for k in range(4):
        x1, y1 = ci[k]
        x2, y2 = cj[k]
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        n = np.hypot(a, b)
        if n < 1e-9:
            return np.nan, np.nan, np.inf
        a, b, c = a / n, b / n, c / n
        abc.append((a, b, c))
        sumaa += a * a
        sumab += a * b
        sumbb += b * b
        sumac += a * c
        sumbc += b * c
    M = np.array([[sumaa, sumab], [sumab, sumbb]])
    if abs(np.linalg.det(M)) < 1e-9:  # 4 lines ~parallel
        return np.nan, np.nan, np.inf
    vx, vy = np.linalg.solve(M, [-sumac, -sumbc])
    res = np.sqrt(np.mean([(a * vx + b * vy + c) ** 2 for a, b, c in abc]))
    return float(vx), float(vy), float(res)


def fit_line_tls(pts, thr, iters=3):
    """Total-least-squares (PCA) line fit with inlier rejection.

    Returns (centroid, direction_unit, normal_unit, inlier_mask).
    inlier = perpendicular distance < thr.
    """
    mask = np.ones(len(pts), dtype=bool)
    m = pts.mean(0)
    d = np.array([1.0, 0.0])
    nrm = np.array([0.0, 1.0])
    for _ in range(iters):
        if mask.sum() < 2:
            break
        P = pts[mask]
        m = P.mean(0)
        u, s, vt = np.linalg.svd(P - m)
        d = vt[0]
        nrm = vt[1]
        dist = np.abs((pts - m) @ nrm)
        mask = dist < thr
    return m, d, nrm, mask


def horizon_center_y_angle(m, d, W):
    """y of the fitted line at x=W/2, and tilt angle (deg from horizontal)."""
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    # normalize angle to [-90,90]
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    if abs(d[0]) < 1e-9:
        return float("nan"), angle
    slope = d[1] / d[0]
    y_at = m[1] + slope * (W / 2 - m[0])
    return float(y_at), angle


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--min-life", type=int, default=15)
    ap.add_argument("--h-var", type=float, default=0.30)
    ap.add_argument("--min-vis", type=float, default=0.6)
    ap.add_argument("--min-h", type=float, default=20.0)
    ap.add_argument("--max-h-frac", type=float, default=0.9)
    ap.add_argument("--ar-lo", type=float, default=0.2)
    ap.add_argument("--ar-hi", type=float, default=0.8)
    ap.add_argument("--edge", type=float, default=5.0)
    ap.add_argument(
        "--ratio",
        type=float,
        default=1.30,
        help="min pair height ratio (§4.1). Below ~1.30 the similitude center "
        "denominator (h_i-h_j)->0 and V drifts; std(Vy)/H knee measured at 30%%.",
    )
    ap.add_argument(
        "--vfar", type=float, default=3.0, help="drop |V| beyond this * img extent"
    )
    ap.add_argument(
        "--inlier-frac", type=float, default=0.03, help="horizon inlier thr = frac*H px"
    )
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(
        f"\nHorizon via vertex homothety (GT upper bound)  det={args.det}  "
        f"ratio>={args.ratio}  boot={args.boot}"
    )
    hdr = (
        f"{'sequence':<16}{'camera':<16}{'boxes':>6}{'pairs':>7}  "
        f"{'concRMS':>8}{'inlier%':>8}{'tlsRMS/H':>9}  "
        f"{'cy/H':>7}{'boot_cy/H':>10}{'tilt°':>7}{'boot°':>7}"
    )
    print(hdr)
    print("─" * len(hdr))

    for seq in SEQS:
        gt_path = args.root / f"{seq}-{args.det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  ! missing {gt_path}")
            continue
        gt = load_gt(gt_path)
        boxes, n_total, rej = median_boxes(gt, seq, args)
        H, W = IMG_H[seq], IMG_W[seq]
        if len(boxes) < 5:
            print(f"{seq:<16}{CAM[seq]:<16}{len(boxes):>6}  too few tracklets {rej}")
            continue

        heights = boxes[:, 3] - boxes[:, 1]
        far_x = args.vfar * W
        far_y = args.vfar * H

        Vs, concs = [], []
        for i, j in combinations(range(len(boxes)), 2):
            hi, hj = heights[i], heights[j]
            if max(hi, hj) / max(min(hi, hj), 1e-6) < args.ratio:
                continue
            vx, vy, res = lsq_intersection(boxes[i], boxes[j])
            if not np.isfinite(vx):
                continue
            if abs(vx) > far_x + W or abs(vy) > far_y + H:  # near-parallel residue
                continue
            Vs.append((vx, vy))
            concs.append(res)
        if len(Vs) < 5:
            print(
                f"{seq:<16}{CAM[seq]:<16}{len(boxes):>6}{len(Vs):>7}  too few V points"
            )
            continue
        Vs = np.array(Vs)
        conc_rms = float(np.median(concs))

        # robust horizon line through the vanishing points
        thr = args.inlier_frac * H
        m, d, nrm, mask = fit_line_tls(Vs, thr)
        inlier_ratio = float(mask.mean())
        tls_rms = float(np.sqrt(np.mean((np.abs((Vs[mask] - m) @ nrm)) ** 2))) / H
        cy, tilt = horizon_center_y_angle(m, d, W)

        # bootstrap stability of the horizon line
        cy_s, tilt_s = [], []
        n = len(Vs)
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            mm, dd, _, _ = fit_line_tls(Vs[idx], thr)
            cyy, tt = horizon_center_y_angle(mm, dd, W)
            if np.isfinite(cyy):
                cy_s.append(cyy)
            tilt_s.append(tt)
        boot_cy = float(np.std(cy_s)) / H if cy_s else float("nan")
        boot_tilt = float(np.std(tilt_s)) if tilt_s else float("nan")

        print(
            f"{seq:<16}{CAM[seq]:<16}{len(boxes):>6}{len(Vs):>7}  "
            f"{conc_rms:>8.1f}{100 * inlier_ratio:>8.1f}{tls_rms:>9.3f}  "
            f"{cy / H:>7.2f}{boot_cy:>10.3f}{tilt:>7.2f}{boot_tilt:>7.2f}"
        )

    print("─" * len(hdr))
    print(
        "\nReading:\n"
        "  concRMS   = median per-pair RMS dist of V to its 4 corner lines (px).\n"
        "              Small => the 4 vertices really do correspond (homothety holds).\n"
        "  inlier%   = fraction of V points lying on ONE horizon line (collinearity).\n"
        "  tlsRMS/H  = perpendicular spread of inlier V about the horizon line.\n"
        "  boot_cy/H, boot° = bootstrap std of horizon center-y / tilt angle.\n"
        "If concRMS is large or boot_* explode even on clean GT, the homothety\n"
        "construction does not converge to a stable horizon → NO-GO."
    )


if __name__ == "__main__":
    main()
