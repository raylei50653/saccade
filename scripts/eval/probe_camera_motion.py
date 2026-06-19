#!/usr/bin/env python3
"""Capstone: per-sequence camera-motion magnitude.

Estimate global inter-frame translation on the raw MOT17 frames (ORB +
partial-affine RANSAC on downscaled grayscale) and report the median/p90
displacement in *original* pixels. Tests the attribution that fw helps on
slow-camera seqs (10) and hurts on fast-camera seqs (13): a suppressed
low-score box is recoverable next frame iff the camera barely moved.
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "datasets" / "MOT17" / "train"
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
DS = 0.5  # downscale for speed
STEP = 3  # sample every Nth frame pair
MAX_PAIRS = 120


def seq_motion(seq):
    files = sorted((IMG_DIR / seq / "img1").glob("*.jpg"))
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev = None
    disps = []
    idxs = list(range(0, len(files), STEP))[:MAX_PAIRS]
    for i in idxs:
        img = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=DS, fy=DS)
        kp, des = orb.detectAndCompute(img, None)
        if prev is not None and des is not None and prev[1] is not None:
            m = bf.match(prev[1], des)
            if len(m) >= 12:
                src = np.float32([prev[0][x.queryIdx].pt for x in m])
                dst = np.float32([kp[x.trainIdx].pt for x in m])
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
                if M is not None:
                    # translation in original px (account for downscale + STEP)
                    tx, ty = M[0, 2] / DS, M[1, 2] / DS
                    disps.append(np.hypot(tx, ty) / STEP)  # per-frame
        prev = (kp, des)
    return np.array(disps)


def main():
    print(
        "Per-sequence camera motion (global inter-frame translation, orig px/frame)\n"
    )
    hdr = f"{'seq':<14}{'ΔIDF1':>7} | {'median':>9}{'p90':>9}{'max':>9}  pairs"
    print(hdr)
    print("-" * (len(hdr) + 3))
    rows = []
    for seq in SEQS:
        d = seq_motion(seq)
        med, p90, mx = (
            (np.median(d), np.percentile(d, 90), d.max()) if len(d) else (0, 0, 0)
        )
        rows.append((seq, med))
        tag = "GO " if DELTA[seq] > 1 else ("NOGO" if DELTA[seq] < -1 else "  · ")
        print(
            f"{tag} {seq:<10}{DELTA[seq]:>+7.1f} | {med:>9.2f}{p90:>9.2f}{mx:>9.2f}  {len(d)}"
        )
    print("\nRanked by camera motion (median px/frame):")
    for seq, med in sorted(rows, key=lambda r: -r[1]):
        tag = "GO " if DELTA[seq] > 1 else ("NOGO" if DELTA[seq] < -1 else "  · ")
        print(f"  {tag} {seq:<14} {med:6.2f} px/frame   ΔIDF1={DELTA[seq]:+.1f}")


if __name__ == "__main__":
    main()
