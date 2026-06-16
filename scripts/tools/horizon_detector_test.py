#!/usr/bin/env python3
"""Horizon homothety on REAL detections vs GT — does the vertex-homothety horizon
survive detector truncation / jitter? (the decisive test the GT upper bound cannot answer)

GT boxes are amodal (full body even under occlusion), so the GT probe is an upper
bound. Real tracker output boxes are truncated/jittered by the detector. This
compares, per sequence, the convergence (λ1/λ2 of the vanishing-point cloud) and
horizon position Vy/H of:
  * GT      : datasets/MOT17/train/{seq}-{det}/gt/gt.txt        (amodal)
  * DET     : tracker MOT-format output {pred}/{seq}-{det}.txt  (real boxes)

Same gates both sides (tracklet median, height stability, aspect, edge, pair
ratio ≥ 1.30 from the §4.1 drift knee). If λ1/λ2 collapses or Vy diverges from GT
on DET, the construction does not survive real detections → the prereq for any
downstream (relink / normalization / GMC) fails on deployment data.

Usage
-----
  .venv/bin/python scripts/tools/horizon_detector_test.py \
      --pred scripts/eval/output/ablation_mot17/baseline
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
from horizon_homothety_probe import (  # noqa: E402
    CAM,
    IMG_H,
    IMG_W,
    SEQS,
    fit_line_tls,
    horizon_center_y_angle,
    load_gt,
    lsq_intersection,
    median_boxes,
)


def load_pred(path: Path):
    """MOT-format tracker output -> id -> {frame:(x,y,w,h,vis=1.0)} (vis dummy)."""
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        frame, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        by_id[tid][frame] = (x, y, w, h, 1.0)
    return dict(by_id)


def convergence(boxes, H, W, ratio, vfar, inlier_frac, rng, boot):
    heights = boxes[:, 3] - boxes[:, 1]
    Vs = []
    for i, j in combinations(range(len(boxes)), 2):
        hi, hj = heights[i], heights[j]
        if max(hi, hj) / max(min(hi, hj), 1e-6) < ratio:
            continue
        vx, vy, _ = lsq_intersection(boxes[i], boxes[j])
        if not np.isfinite(vx) or abs(vx) > vfar * W or abs(vy) > vfar * H:
            continue
        Vs.append((vx, vy))
    if len(Vs) < 5:
        return None
    Vs = np.array(Vs)
    m = Vs.mean(0)
    _, s, _ = np.linalg.svd(Vs - m)
    lam = s[0] ** 2 / max(s[1] ** 2, 1e-9)
    thr = inlier_frac * H
    mm, dd, nrm, mask = fit_line_tls(Vs, thr)
    cy, tilt = horizon_center_y_angle(mm, dd, W)
    cy_s = []
    n = len(Vs)
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        a, b, _, _ = fit_line_tls(Vs[idx], thr)
        c, _ = horizon_center_y_angle(a, b, W)
        if np.isfinite(c):
            cy_s.append(c)
    boot_cy = float(np.std(cy_s)) / H if cy_s else float("nan")
    return dict(
        nV=len(Vs), lam=lam, cy=cy / H, inlier=mask.mean(), boot_cy=boot_cy, tilt=tilt
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--pred", type=Path, required=True, help="dir with {seq}-{det}.txt")
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--min-life", type=int, default=15)
    ap.add_argument("--h-var", type=float, default=0.30)
    ap.add_argument("--min-vis", type=float, default=0.6)
    ap.add_argument("--min-h", type=float, default=20.0)
    ap.add_argument("--max-h-frac", type=float, default=0.9)
    ap.add_argument("--ar-lo", type=float, default=0.2)
    ap.add_argument("--ar-hi", type=float, default=0.8)
    ap.add_argument("--edge", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=1.30)
    ap.add_argument("--vfar", type=float, default=3.0)
    ap.add_argument("--inlier-frac", type=float, default=0.03)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(
        f"\nHorizon homothety: GT (amodal) vs DET (real boxes)  ratio>={args.ratio}\n"
    )
    hdr = (
        f"{'sequence':<16}{'camera':<16}"
        f"{'│ GT: box  nV   λ1/λ2  Vy/H':<30}"
        f"{'│ DET: box  nV   λ1/λ2  Vy/H  boot':<34}{'│ ΔVy/H':>8}"
    )
    print(hdr)
    print("─" * len(hdr))

    for seq in SEQS:
        gt_path = args.root / f"{seq}-{args.det}" / "gt" / "gt.txt"
        pred_path = args.pred / f"{seq}-{args.det}.txt"
        if not gt_path.exists() or not pred_path.exists():
            print(f"  ! missing {seq}")
            continue
        H, W = IMG_H[seq], IMG_W[seq]

        gt_boxes, _, _ = median_boxes(load_gt(gt_path), seq, args)
        pred_boxes, _, _ = median_boxes(load_pred(pred_path), seq, args)

        g = convergence(
            gt_boxes, H, W, args.ratio, args.vfar, args.inlier_frac, rng, args.boot
        )
        d = convergence(
            pred_boxes, H, W, args.ratio, args.vfar, args.inlier_frac, rng, args.boot
        )
        if g is None or d is None:
            print(
                f"{seq:<16}{CAM[seq]:<16} insufficient (GT={g is not None} DET={d is not None})"
            )
            continue
        dvy = abs(d["cy"] - g["cy"])
        verdict = (
            "✓" if (d["lam"] > 100 and dvy < 0.06) else ("~" if d["lam"] > 40 else "✗")
        )
        print(
            f"{seq:<16}{CAM[seq]:<16}"
            f"│{len(gt_boxes):>5}{g['nV']:>5}{g['lam']:>8.0f}{g['cy']:>7.2f}  "
            f"│{len(pred_boxes):>5}{d['nV']:>5}{d['lam']:>8.0f}{d['cy']:>7.2f}{d['boot_cy']:>7.3f}  "
            f"│{dvy:>6.2f} {verdict}"
        )

    print("─" * len(hdr))
    print(
        "\nVerdict: ✓ = DET converges (λ1/λ2>100) AND Vy within 0.06H of GT.\n"
        "         ~ = weak line (λ1/λ2 40-100).  ✗ = collapsed on real boxes.\n"
        "If DET ✗ where GT ✓, the horizon does not survive detector truncation."
    )


if __name__ == "__main__":
    main()
