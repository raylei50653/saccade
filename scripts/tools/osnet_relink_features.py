#!/usr/bin/env python3
"""Deep-ReID upper bound for relink-candidate appearance matching.

Companion to color_relink_features.py: same candidate pairs, same
visibility-based endpoint sampling, but OSNet TRT embeddings + cosine
similarity instead of color histograms.  Answers whether the hard pool
(bridge_dist <= 1.0 h) carries *any* appearance signal — if OSNet also sits at
AUC ~0.5 there, appearance as a whole is dead for relink gating on MOT17, not
just cheap color features.

Usage:
  .venv/bin/python scripts/tools/osnet_relink_features.py \
      --csv scripts/tools/out/relink_candidates.csv \
      --skip 3 --window 12 --topk 3
"""

from __future__ import annotations

import argparse
import csv
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

import torch  # noqa: E402

from color_relink_features import _ap, _auc, load_boxes, pick_sample_frames  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402


def crop_chw(img, box, hw):
    """uint8 RGB (3, H, W) crop resized to engine input, or None if degenerate."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x0, y0 = int(max(x, 0)), int(max(y, 0))
    x1, y1 = int(min(x + w, W)), int(min(y + h, H))
    if x1 - x0 < 8 or y1 - y0 < 16:
        return None
    c = cv2.resize(img[y0:y1, x0:x1], (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(c[..., ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/relink_candidates.csv")
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument(
        "--out", type=Path, default=Path("scripts/tools/out/osnet_relink_features.csv")
    )
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hard-dist", type=float, default=1.0)
    ap.add_argument("--engine", default="osnet", help="TRTFeatureExtractor model_type")
    args = ap.parse_args()

    extractor = TRTFeatureExtractor(model_type=args.engine)
    hw = extractor.input_hw
    device = torch.device("cuda")
    print(f"{args.engine} engine: input {hw}, dim {extractor.feature_dim}")

    rows = list(csv.DictReader(open(args.csv)))
    need: dict[str, set] = defaultdict(set)
    for r in rows:
        need[r["seq"]].add((int(r["lost_id"]), "lost"))
        need[r["seq"]].add((int(r["cand_id"]), "cand"))

    embs: dict[tuple, np.ndarray] = {}
    vis_max: dict[tuple, float] = {}
    for seq, endpoints in sorted(need.items()):
        by_id, by_frame = load_boxes(args.mot_dir / f"{seq}.txt")
        plan: dict[int, list] = defaultdict(list)
        for tid, side in endpoints:
            traj = by_id.get(tid)
            if not traj:
                continue
            for frm, box, occ, vis in pick_sample_frames(
                traj, by_frame, side, args.window, args.topk, args.skip
            ):
                plan[frm].append(((seq, tid, side), box, vis))
        img_dir = args.img_root / seq / "img1"
        keys, crops = [], []
        for frm in sorted(plan):
            img = cv2.imread(str(img_dir / f"{frm:06d}.jpg"))
            if img is None:
                continue
            for key, box, vis in plan[frm]:
                c = crop_chw(img, box, hw)
                if c is not None:
                    keys.append(key)
                    crops.append(c)
                    vis_max[key] = max(vis_max.get(key, 0.0), vis)
        # batch-extract embeddings
        feats = np.empty((len(crops), extractor.feature_dim), dtype=np.float32)
        for i in range(0, len(crops), args.batch):
            batch = (
                torch.from_numpy(np.stack(crops[i : i + args.batch]))
                .to(device)
                .float()
                .div_(255.0)
                .contiguous()
            )
            feats[i : i + len(batch)] = extractor.extract(batch).cpu().numpy()
        acc: dict[tuple, list] = defaultdict(list)
        for k, f in zip(keys, feats):
            acc[k].append(f)
        n0 = len(embs)
        for k, fs in acc.items():
            m = np.mean(fs, axis=0)
            embs[k] = m / max(np.linalg.norm(m), 1e-9)
        print(
            f"  {seq:<16} endpoints {len(embs) - n0}/{len(endpoints)} ({len(crops)} crops)"
        )

    n_valid = 0
    for r in rows:
        seq = r["seq"]
        ea = embs.get((seq, int(r["lost_id"]), "lost"))
        eb = embs.get((seq, int(r["cand_id"]), "cand"))
        if ea is None or eb is None:
            r["osnet_cos"] = ""
            continue
        r["osnet_cos"] = f"{float(ea @ eb):.6f}"
        n_valid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({n_valid} with embeddings) -> {args.out}")

    ok = [r for r in rows if r["gt_valid"] == "1" and r["osnet_cos"] != ""]
    y = np.array([int(r["gt_match"]) for r in ok])
    bd = np.array([float(r["bridge_dist"]) for r in ok])
    sc = np.array([float(r["osnet_cos"]) for r in ok])
    gap = np.array([int(r["gap"]) for r in ok])
    hard = bd <= args.hard_dist
    print(f"\nPool : {len(y)} pairs | {int(y.sum())} pos ({100 * y.mean():.1f}%)")
    print(f"Hard : {int(hard.sum())} pairs | {int(y[hard].sum())} pos")
    print(
        f"\n{'variant':<22} {'AUC full':>9} {'AUC hard':>9} {'AP full':>8} {'AP hard':>8}"
    )
    print(
        f"{'-bridge_dist (geom)':<22} {_auc(-bd, y):>9.4f} {_auc(-bd[hard], y[hard]):>9.4f} "
        f"{_ap(-bd, y):>8.4f} {_ap(-bd[hard], y[hard]):>8.4f}"
    )
    print(
        f"{'osnet_cos':<22} {_auc(sc, y):>9.4f} {_auc(sc[hard], y[hard]):>9.4f} "
        f"{_ap(sc, y):>8.4f} {_ap(sc[hard], y[hard]):>8.4f}"
    )

    print("\n── osnet_cos: hard-pool AUC by gap bucket ──")
    for glo, ghi in [(1, 10), (10, 30), (30, 80), (80, 301)]:
        m = hard & (gap >= glo) & (gap < ghi)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(
                f"  gap [{glo:>3},{ghi:>3}): {int(m.sum()):>5} pairs, "
                f"{int(y[m].sum()):>4} pos, AUC {_auc(sc[m], y[m]):.4f}"
            )

    print("\n── osnet_cos: hard-pool AUC per seq ──")
    seqs = np.array([r["seq"] for r in ok])
    for s in np.unique(seqs):
        m = hard & (seqs == s)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(
                f"  {s}: n={int(m.sum()):>4} pos={int(y[m].sum()):>3} "
                f"osnet {_auc(sc[m], y[m]):.3f}  geom {_auc(-bd[m], y[m]):.3f}"
            )


if __name__ == "__main__":
    main()
