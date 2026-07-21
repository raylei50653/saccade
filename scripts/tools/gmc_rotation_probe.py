#!/usr/bin/env python3
"""GMC rotation distribution — does the default affine GMC leak spurious rotation?

The default pipeline GMC (CppGMC, gpu mode) and the eager SparseOpticalFlowGMC
both estimate a 4-DOF similarity via cv2.estimateAffinePartial2D and apply the
rotation WITHOUT any plausibility bound (gmc.cpp:456-472 has no rot/scale gate).
The horizon probe showed the true camera roll on MOT17 is ≪ a few degrees
(tilt across-window std ~1.3°, frame-to-frame ~0). So any per-frame GMC rotation
beyond ~1-3° is almost certainly crowd motion masquerading as camera rotation
(design §9.1).

This runs the same estimateAffinePartial2D GMC on raw frames and reports the
per-frame |rotation| distribution per sequence. If a meaningful fraction exceeds
the horizon bound, a horizon-justified rotation clamp has a real target.

  rot   = atan2(warp[1,0], warp[0,0])   (frame-to-frame rotation, deg)
  scale = hypot(warp[0,0], warp[1,0])

Usage
-----
  .venv/bin/python scripts/tools/gmc_rotation_probe.py
"""
# status: experiment

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from saccade.perception.eval.gmc import SparseOpticalFlowGMC  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument(
        "--downscale", type=int, default=8, help="match default gmc_downscale"
    )
    ap.add_argument(
        "--bound", type=float, default=3.0, help="horizon-justified rot bound (deg)"
    )
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"\nGMC per-frame |rotation| (cv2.estimateAffinePartial2D, ds={args.downscale})  device={dev}"
    )
    print(f"horizon-justified bound = {args.bound}°\n")
    hdr = (
        f"{'sequence':<16}{'camera':<16}{'frames':>7}  "
        f"{'med°':>6}{'p90°':>6}{'p99°':>6}{'max°':>7}  "
        f"{'>1°':>6}{'>3°':>6}{'>5°':>6}  {'scale p99':>9}"
    )
    print(hdr)
    print("─" * len(hdr))

    agg = {}
    for seq in SEQS:
        img_dir = args.root / f"{seq}-{args.det}" / "img1"
        imgs = sorted(img_dir.glob("*.jpg"))
        if not imgs:
            print(f"  ! no frames {img_dir}")
            continue
        gmc = SparseOpticalFlowGMC(downscale=args.downscale)
        rots, scales = [], []
        for p in imgs:
            bgr = cv2.imread(str(p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float() / 255.0
            warp = gmc.estimate(t)
            if warp is None:
                continue
            a, b = float(warp[0, 0]), float(warp[1, 0])
            rots.append(abs(math.degrees(math.atan2(b, a))))
            scales.append(math.hypot(a, b))
        if len(rots) < 10:
            print(f"{seq:<16}{CAM[seq]:<16}{len(rots):>7}  too few")
            continue
        r = np.array(rots)
        s = np.array(scales)
        agg[seq] = r
        print(
            f"{seq:<16}{CAM[seq]:<16}{len(r):>7}  "
            f"{np.median(r):>6.2f}{np.percentile(r, 90):>6.2f}{np.percentile(r, 99):>6.2f}{r.max():>7.2f}  "
            f"{100 * (r > 1).mean():>5.1f}%{100 * (r > 3).mean():>5.1f}%{100 * (r > 5).mean():>5.1f}%  "
            f"{np.percentile(s, 99):>9.3f}"
        )

    print("─" * len(hdr))
    print(
        f"\n>{args.bound:.0f}° column = fraction of frames where GMC rotation exceeds the\n"
        "horizon-justified bound = candidate SPURIOUS (crowd-as-rotation) frames.\n"
        "If this is a meaningful fraction (esp. on moving/crowded seqs), a consistent\n"
        "horizon-bounded rotation clamp on the default GMC has a real target."
    )


if __name__ == "__main__":
    main()
