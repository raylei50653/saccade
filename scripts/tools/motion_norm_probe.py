#!/usr/bin/env python3
"""§8.4 motion normalization — does scale-normalizing displacement improve the
true-vs-false association/relink decision? And does the horizon depth_proxy beat
plain bbox-height normalization?

The relink/association gate decides, after a gap of g frames, whether a track's
last box and a candidate box are the same person. The discriminating feature is
the center displacement. Raw pixel displacement is scale-biased: a near (large)
person genuinely moves more pixels than a far (small) one, so a single raw
threshold over-rejects near tracks and under-rejects far ones. §8.4 proposes
normalizing displacement by scale.

This is a GT-only AUC probe (no tracker). For each gap g and each GT id A present
at t and t+g:
  * TRUE  : (A@t -> A@t+g)                       label 1
  * FALSE : (A@t -> B@t+g) for other ids B that are spatially plausible confusors
            (B@t+g within REACH*h of A@t)        label 0
Three features ranked by separability (AUC, higher=better):
  raw   = ||disp||
  /h    = ||disp|| / h(A@t)                       (bbox-height norm, no horizon)
  /dp   = ||disp|| / depth_proxy                  (horizon norm; dp=foot_y-horizon_y)

If /h does not beat raw, §8.4 is dead. If /dp does not beat /h, the horizon adds
nothing over cheap bbox-height normalization.

Usage
-----
  .venv/bin/python scripts/tools/motion_norm_probe.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
from horizon_homothety_probe import IMG_H, SEQS, load_gt  # noqa: E402

# measured GT horizon (Vy/H) from horizon_homothety_probe; None = non-converging seq
HORIZON = {
    "MOT17-02": 0.42,
    "MOT17-04": None,
    "MOT17-05": 0.48,
    "MOT17-09": 0.51,
    "MOT17-10": None,
    "MOT17-11": None,
    "MOT17-13": 0.44,
}
CAM = {
    "MOT17-02": "static elevated",
    "MOT17-04": "static high",
    "MOT17-05": "moving low",
    "MOT17-09": "static low",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC that the feature SEPARATES classes; flip so >0.5 means 'smaller=true'."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # true matches should have SMALLER displacement -> rank ascending, true first
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[: len(pos)].sum()
    a = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return 1.0 - a  # smaller-feature = positive => want 1-a


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument("--gaps", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument(
        "--reach", type=float, default=8.0, help="confusor radius = reach*h"
    )
    args = ap.parse_args()

    print(
        f"\n§8.4 motion-normalization separability (GT, AUC; higher=better)  gaps={args.gaps}\n"
    )
    hdr = f"{'sequence':<16}{'camera':<16}{'pairs':>8}{'raw':>8}{'/h':>8}{'/dp':>8}   {'winner':>8}"
    print(hdr)
    print("─" * len(hdr))

    overall = defaultdict(list)
    for seq in SEQS:
        gt_path = args.root / f"{seq}-{args.det}" / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        gt = load_gt(gt_path)
        H = IMG_H[seq]
        hy = HORIZON[seq]
        # frame -> [(id, cx, cy, h, foot_y)]
        present: dict[int, list] = defaultdict(list)
        for gid, fr in gt.items():
            for f, (x, y, w, h, _) in fr.items():
                present[f].append((gid, x + w / 2, y + h / 2, h, y + h))

        raw_p, raw_n, h_p, h_n, dp_p, dp_n = [], [], [], [], [], []
        for gid, fr in gt.items():
            frames = sorted(fr)
            fset = set(frames)
            for t in frames:
                x, y, w, h, _ = fr[t]
                cx, cy, foot = x + w / 2, y + h / 2, y + h
                dp = max(foot - hy * H, 1.0) if hy is not None else None
                for g in args.gaps:
                    tg = t + g
                    if tg not in fset:
                        continue
                    xg, yg, wg, hg, _ = fr[tg]
                    cxg, cyg = xg + wg / 2, yg + hg / 2
                    d = float(np.hypot(cxg - cx, cyg - cy))
                    raw_p.append(d)
                    h_p.append(d / h)
                    if dp is not None:
                        dp_p.append(d / dp)
                    # false confusors: other ids at tg within reach*h of A@t
                    for bid, bcx, bcy, bh, bfoot in present.get(tg, []):
                        if bid == gid:
                            continue
                        df = float(np.hypot(bcx - cx, bcy - cy))
                        if df > args.reach * h:
                            continue
                        raw_n.append(df)
                        h_n.append(df / h)
                        if dp is not None:
                            dp_n.append(df / dp)
        if len(raw_p) < 20 or len(raw_n) < 20:
            print(f"{seq:<16}{CAM[seq]:<16}{len(raw_p):>8}  too few")
            continue
        a_raw = auc(np.array(raw_p), np.array(raw_n))
        a_h = auc(np.array(h_p), np.array(h_n))
        a_dp = auc(np.array(dp_p), np.array(dp_n)) if hy is not None else float("nan")
        overall["raw"].append(a_raw)
        overall["h"].append(a_h)
        if hy is not None:
            overall["dp"].append(a_dp)
        winner = max(
            [("raw", a_raw), ("/h", a_h)] + ([("/dp", a_dp)] if hy is not None else []),
            key=lambda kv: kv[1],
        )[0]
        dp_s = f"{a_dp:>8.3f}" if hy is not None else f"{'—':>8}"
        print(
            f"{seq:<16}{CAM[seq]:<16}{len(raw_p):>8}{a_raw:>8.3f}{a_h:>8.3f}{dp_s}   {winner:>8}"
        )

    print("─" * len(hdr))
    print(
        f"\nMEAN   raw={np.mean(overall['raw']):.3f}  /h={np.mean(overall['h']):.3f}  "
        f"/dp={np.mean(overall['dp']):.3f} (converging seqs only)\n"
    )
    print(
        "AUC = P(true match ranked closer than a plausible confusor).\n"
        "  /h > raw  -> bbox-height normalization helps association (§8.4 alive).\n"
        "  /dp > /h  -> horizon depth_proxy beats cheap bbox-height norm.\n"
        "  /dp ≈ /h  -> horizon adds nothing over bbox-height; use /h, drop horizon."
    )


if __name__ == "__main__":
    main()
