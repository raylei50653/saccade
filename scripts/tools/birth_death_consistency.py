#!/usr/bin/env python3
"""Birth-vs-death appearance consistency for long-lived tracks.

The sharpest within-track proxy for a *successful* relink pair: take tracks
that lived a long time, extract features at their birth (cand-side sampling)
and just before their death (lost-side sampling) using the exact endpoint
sampling the relink experiments use, and ask whether the feature still matches
across that long baseline — same-id (birth, death) pairs vs cross-id pairs
(death of A vs birth of B, mirroring the relink direction).

If same-id consistency collapses at long lifespans even for tracks that were
never fragmented, then "imaging conditions are not conserved across time" is
confirmed independently of occlusion/fragment quality.

Usage:
  .venv/bin/python scripts/tools/birth_death_consistency.py
"""
# status: experiment

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from color_relink_features import (  # noqa: E402
    _auc,
    endpoint_features,
    load_boxes,
    pair_sims,
    pick_sample_frames,
)

DEFAULT_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]
LIFE_BUCKETS = [(30, 100), (100, 300), (300, 10**9)]
REPORT_COLS = [
    "color_split_nomask",
    "color_pool_center",
    "color_pool_bgsub",
    "color_shiftalign_bgsub",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cross-per-seq", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    same_rows = []  # (seq, lifespan, sims dict)
    cross_rows = []  # (seq, sims dict)
    for seq in args.seqs:
        by_id, by_frame = load_boxes(args.mot_dir / f"{seq}.txt")
        plan: dict[int, list] = defaultdict(list)
        lifespan: dict[int, int] = {}
        for tid, traj in by_id.items():
            if len(traj) < args.min_len:
                continue
            lifespan[tid] = traj[-1][0] - traj[0][0] + 1
            for side in ("lost", "cand"):
                for frm, box, occ, vis in pick_sample_frames(
                    traj, by_frame, side, args.window, args.topk, args.skip
                ):
                    plan[frm].append(((tid, side), box, occ))
        img_dir = args.img_root / seq / "img1"
        acc: dict[tuple, list] = defaultdict(list)
        for frm in sorted(plan):
            img = cv2.imread(str(img_dir / f"{frm:06d}.jpg"))
            if img is None:
                continue
            for key, box, occ in plan[frm]:
                f = endpoint_features(img, box, occ)
                if f is not None:
                    acc[key].append(f)
        feats = {
            k: {n: np.mean([f[n] for f in lst], axis=0) for n in lst[0]}
            for k, lst in acc.items()
        }
        done = [t for t in lifespan if (t, "lost") in feats and (t, "cand") in feats]
        for t in done:
            same_rows.append(
                (seq, lifespan[t], pair_sims(feats[(t, "lost")], feats[(t, "cand")]))
            )
        for _ in range(args.cross_per_seq):
            a, b = rng.choice(done, 2, replace=False)
            cross_rows.append((seq, pair_sims(feats[(a, "lost")], feats[(b, "cand")])))
        print(
            f"  {seq:<16} {len(done)} tracks "
            f"(lifespan med {int(np.median([lifespan[t] for t in done]))})"
        )

    for col in REPORT_COLS:
        print(f"\n══ {col} ══  (same-id birth↔death vs cross-id, endpoint sampling)")
        print(f"{'lifespan':<12} {'n_same':>7} {'same μ':>8} {'cross μ':>8} {'AUC':>6}")
        cs = np.array([d[col] for _, d in cross_rows])
        for lo, hi in LIFE_BUCKETS:
            ss = np.array([d[col] for _, life, d in same_rows if lo <= life < hi])
            if len(ss) < 10:
                continue
            y = np.concatenate([np.ones(len(ss)), np.zeros(len(cs))])
            a = _auc(np.concatenate([ss, cs]), y)
            hi_txt = f"{hi}" if hi < 10**9 else "inf"
            print(
                f"[{lo:>3},{hi_txt:>4}) {len(ss):>8} {ss.mean():>8.2f} "
                f"{cs.mean():>8.2f} {a:>6.3f}"
            )

    col = "color_pool_bgsub"
    print(f"\n── {col}: per-seq AUC (lifespan >= 100) ──")
    for seq in args.seqs:
        ss = np.array([d[col] for s, life, d in same_rows if s == seq and life >= 100])
        cs = np.array([d[col] for s, d in cross_rows if s == seq])
        if len(ss) < 5 or len(cs) < 10:
            continue
        y = np.concatenate([np.ones(len(ss)), np.zeros(len(cs))])
        print(f"  {seq}: n={len(ss):>3}  AUC {_auc(np.concatenate([ss, cs]), y):.3f}")


if __name__ == "__main__":
    main()
