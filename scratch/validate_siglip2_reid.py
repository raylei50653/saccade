#!/usr/bin/env python3
"""
Validate marketaju_siglip2_reid embedding discriminability for SWAP resolution.

Crops GT-annotated identities from real MOT17 frames, embeds them with the TRT
siglip2_reid model, and measures intra-ID vs inter-ID cosine similarity. The
decisive metric for SWAP is inter-ID similarity between DIFFERENT people that
co-occur in the SAME frame (spatial neighbours that the tracker can confuse).

If same-frame inter-ID sim overlaps intra-ID sim, appearance cannot break
crossings no matter how the tracker plumbing is tuned -> NO-GO.

Usage:
    uv run scratch/validate_siglip2_reid.py --sequences MOT17-09-SDP,MOT17-04-SDP
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402


def load_gt(seq_dir: Path, min_vis: float) -> dict[int, list[tuple[int, list[float]]]]:
    """Return {frame_id: [(track_id, xyxy), ...]} for visible pedestrians (class 1)."""
    gt_file = seq_dir / "gt" / "gt.txt"
    by_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for line in gt_file.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 9:
            continue
        fid, tid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf, cls, vis = int(p[6]), int(p[7]), float(p[8])
        if conf == 0 or cls != 1 or vis < min_vis:
            continue
        by_frame[fid].append((tid, [x, y, x + w, y + h]))
    return by_frame


def crop_batch(img: np.ndarray, boxes: list[list[float]], size: int) -> torch.Tensor:
    """Crop xyxy boxes from a BGR image -> [N,3,size,size] RGB float32 in [0,1]."""
    H, W = img.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(W, int(x2)), min(H, int(y2))
        if xi2 <= xi1 or yi2 <= yi1:
            crops.append(np.zeros((size, size, 3), np.uint8))
            continue
        c = cv2.resize(img[yi1:yi2, xi1:xi2], (size, size))
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    arr = np.stack(crops).astype(np.float32) / 255.0  # [N,H,W,3]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().cuda()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="MOT17-09-SDP")
    ap.add_argument("--data-root", default="datasets/MOT17/train")
    ap.add_argument("--min-vis", type=float, default=0.5)
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--max-per-track", type=int, default=8)
    args = ap.parse_args()

    ext = TRTFeatureExtractor(model_type="siglip2_reid", max_batch=64)
    size = ext.input_hw[0]

    # Per-identity embedding samples, and per-frame (tid, embedding) for same-frame inter-ID.
    track_embs: dict[str, list[torch.Tensor]] = defaultdict(list)
    same_frame_pairs: list[float] = []

    for seq in args.sequences.split(","):
        seq_dir = Path(args.data_root) / seq
        img_dir = seq_dir / "img1"
        by_frame = load_gt(seq_dir, args.min_vis)
        frames = sorted(by_frame.keys())[:: args.frame_stride]
        per_track_count: dict[str, int] = defaultdict(int)
        for fid in frames:
            entries = by_frame[fid]
            if len(entries) < 2:
                continue
            img = cv2.imread(str(img_dir / f"{fid:06d}.jpg"))
            if img is None:
                continue
            tids = [e[0] for e in entries]
            boxes = [e[1] for e in entries]
            embs = ext.extract(crop_batch(img, boxes, size))  # [N,D] L2-normed
            # same-frame inter-ID similarities (different people, same frame = SWAP risk)
            sim = (embs @ embs.T).cpu().numpy()
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    if tids[i] != tids[j]:
                        same_frame_pairs.append(float(sim[i, j]))
            # collect per-track samples for intra-ID
            for k, tid in enumerate(tids):
                key = f"{seq}:{tid}"
                if per_track_count[key] < args.max_per_track:
                    track_embs[key].append(embs[k].cpu())
                    per_track_count[key] += 1

    # intra-ID: sim between samples of the same track
    intra: list[float] = []
    for key, lst in track_embs.items():
        if len(lst) < 2:
            continue
        m = F.normalize(torch.stack(lst), dim=-1)
        s = (m @ m.T).numpy()
        iu = np.triu_indices(len(lst), k=1)
        intra.extend(s[iu].tolist())

    # global inter-ID: sim between mean embeddings of different tracks
    keys = [k for k, v in track_embs.items() if len(v) >= 1]
    means = F.normalize(
        torch.stack([torch.stack(track_embs[k]).mean(0) for k in keys]), dim=-1
    )
    gsim = (means @ means.T).numpy()
    gi = np.triu_indices(len(keys), k=1)
    inter_global = gsim[gi].tolist()

    def stats(name: str, xs: list[float]) -> None:
        a = np.array(xs)
        if a.size == 0:
            print(f"  {name:24s}: (no samples)")
            return
        print(
            f"  {name:24s}: n={a.size:6d}  mean={a.mean():.3f}  "
            f"median={np.median(a):.3f}  p10={np.percentile(a, 10):.3f}  "
            f"p90={np.percentile(a, 90):.3f}"
        )

    print(f"\n=== siglip2_reid discriminability ({args.sequences}) ===")
    print(f"  identities={len(keys)}  feature_dim={ext.feature_dim}")
    stats("intra-ID (same person)", intra)
    stats("inter-ID global (mean)", inter_global)
    stats("inter-ID same-frame", same_frame_pairs)
    if intra and same_frame_pairs:
        margin = float(np.median(intra) - np.median(same_frame_pairs))
        print(
            f"\n  separation margin (intra median - same-frame inter median): {margin:.3f}"
        )
        print("  -> >0.15 = usable for SWAP; <0.05 = NO-GO (cannot break crossings)")


if __name__ == "__main__":
    main()
