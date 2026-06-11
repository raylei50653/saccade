#!/usr/bin/env python3
"""Intra-track appearance consistency: how stable is a feature on the SAME id?

Upstream diagnostic for the relink appearance gate (color_relink_features.py /
osnet_relink_features.py, both NO-GO at hard-pool AUC ~0.5): before asking a
feature to match a lost track to a candidate across a gap, check it is even
consistent *within* one continuous track.

Per track (len >= --min-len) we sample up to --samples evenly spaced frames,
build the same color histograms (and optionally OSNet embeddings), then:

  same-id pairs : all sample pairs within a track, binned by frame offset dt
  cross-id pairs: random pairs from two different tracks in the same seq,
                  binned by the same |dt| (controls scene/lighting drift)

Report per dt bucket: mean similarity same vs cross, and AUC(same vs cross) —
overall and restricted to clean samples (both visibilities >= --vis-thr).
If the same-id curve decays into the cross-id floor by dt ~ relink gaps, the
feature cannot support relink gating no matter how endpoints are sampled.

Usage:
  .venv/bin/python scripts/tools/intra_track_consistency.py --osnet
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))
sys.path.insert(0, str(project_root / "scripts" / "tools"))

from color_relink_features import (  # noqa: E402
    SIM_COLS,
    _auc,
    endpoint_features,
    load_boxes,
    occluder_mask,
    pair_sims,
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
DT_BUCKETS = [(1, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 300), (300, 10**9)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--samples", type=int, default=24, help="max samples per track")
    ap.add_argument("--cross-per-seq", type=int, default=6000)
    ap.add_argument("--vis-thr", type=float, default=0.7)
    ap.add_argument("--osnet", action="store_true", help="also run OSNet embeddings")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    extractor = None
    if args.osnet:
        import torch

        from saccade.perception.feature_extractor import TRTFeatureExtractor
        from osnet_relink_features import crop_chw

        extractor = TRTFeatureExtractor(model_type="osnet")
        hw = extractor.input_hw

    # (seq, tid) -> list of (frame, vis, hists, emb_idx)
    samples: dict[tuple, list] = defaultdict(list)
    all_crops: list = []  # for osnet
    for seq in args.seqs:
        by_id, by_frame = load_boxes(args.mot_dir / f"{seq}.txt")
        plan: dict[int, list] = defaultdict(list)
        for tid, traj in by_id.items():
            if len(traj) < args.min_len:
                continue
            idx = np.unique(np.linspace(0, len(traj) - 1, args.samples).astype(int))
            for i in idx:
                frm, x, y, w, h = traj[i]
                others = [
                    (ox, oy, ow, oh)
                    for otid, ox, oy, ow, oh in by_frame.get(frm, ())
                    if otid != tid
                ]
                occ, vis = occluder_mask((x, y, w, h), others)
                plan[frm].append((tid, (x, y, w, h), occ, vis))
        img_dir = args.img_root / seq / "img1"
        n = 0
        for frm in sorted(plan):
            img = cv2.imread(str(img_dir / f"{frm:06d}.jpg"))
            if img is None:
                continue
            for tid, box, occ, vis in plan[frm]:
                f = endpoint_features(img, box, occ)
                if f is None:
                    continue
                emb_idx = -1
                if extractor is not None:
                    c = crop_chw(img, box, hw)
                    if c is not None:
                        emb_idx = len(all_crops)
                        all_crops.append(c)
                samples[(seq, tid)].append((frm, vis, f, emb_idx))
                n += 1
        print(
            f"  {seq:<16} {n} samples / {sum(1 for k in samples if k[0] == seq)} tracks"
        )

    embs = None
    if extractor is not None and all_crops:
        import torch

        embs = np.empty((len(all_crops), extractor.feature_dim), dtype=np.float32)
        for i in range(0, len(all_crops), args.batch):
            b = (
                torch.from_numpy(np.stack(all_crops[i : i + args.batch]))
                .cuda()
                .float()
                .div_(255.0)
                .contiguous()
            )
            embs[i : i + len(b)] = extractor.extract(b).cpu().numpy()

    feat_names = SIM_COLS + (["osnet_cos"] if embs is not None else [])

    def pair_sim_all(sa, sb):
        _, _, fa, ia = sa
        _, _, fb, ib = sb
        d = pair_sims(fa, fb)
        if embs is not None and ia >= 0 and ib >= 0:
            d["osnet_cos"] = float(embs[ia] @ embs[ib])
        return d

    # same-id pairs
    rng = np.random.default_rng(0)
    same = {
        n: defaultdict(list) for n in feat_names
    }  # name -> bucket -> [(sim, minvis)]
    cross = {n: defaultdict(list) for n in feat_names}
    for (seq, tid), lst in samples.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                dt = abs(lst[j][0] - lst[i][0])
                mv = min(lst[i][1], lst[j][1])
                for name, s in pair_sim_all(lst[i], lst[j]).items():
                    same[name][dt].append((s, mv))
    # cross-id pairs, per seq, dt recorded
    by_seq: dict[str, list] = defaultdict(list)
    for (seq, tid), lst in samples.items():
        by_seq[seq].append(lst)
    for seq, tracks in by_seq.items():
        if len(tracks) < 2:
            continue
        for _ in range(args.cross_per_seq):
            a, b = rng.choice(len(tracks), 2, replace=False)
            sa = tracks[a][rng.integers(len(tracks[a]))]
            sb = tracks[b][rng.integers(len(tracks[b]))]
            dt = abs(sb[0] - sa[0])
            mv = min(sa[1], sb[1])
            for name, s in pair_sim_all(sa, sb).items():
                cross[name][dt].append((s, mv))

    for name in feat_names:
        print(f"\n══ {name} ══")
        print(
            f"{'dt bucket':<12} {'n_same':>7} {'same μ':>7} {'n_cross':>8} {'cross μ':>8} "
            f"{'AUC':>6} {'AUC vis≥' + str(args.vis_thr):>10}"
        )
        for lo, hi in DT_BUCKETS:
            s_all = [
                (s, v)
                for dt, lst in same[name].items()
                if lo <= dt < hi
                for s, v in lst
            ]
            c_all = [
                (s, v)
                for dt, lst in cross[name].items()
                if lo <= dt < hi
                for s, v in lst
            ]
            if len(s_all) < 10 or len(c_all) < 10:
                continue
            ss = np.array([s for s, _ in s_all])
            cs = np.array([s for s, _ in c_all])
            y = np.concatenate([np.ones(len(ss)), np.zeros(len(cs))])
            auc = _auc(np.concatenate([ss, cs]), y)
            sv = np.array([s for s, v in s_all if v >= args.vis_thr])
            cv_ = np.array([s for s, v in c_all if v >= args.vis_thr])
            if len(sv) >= 10 and len(cv_) >= 10:
                yv = np.concatenate([np.ones(len(sv)), np.zeros(len(cv_))])
                auc_v = _auc(np.concatenate([sv, cv_]), yv)
                vtxt = f"{auc_v:>10.3f}"
            else:
                vtxt = f"{'—':>10}"
            hi_txt = f"{hi}" if hi < 10**9 else "inf"
            print(
                f"[{lo:>3},{hi_txt:>4}) {len(ss):>8} {ss.mean():>7.3f} {len(cs):>8} "
                f"{cs.mean():>8.3f} {auc:>6.3f} {vtxt}"
            )


if __name__ == "__main__":
    main()
