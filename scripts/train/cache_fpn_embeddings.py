#!/usr/bin/env python3
"""
Phase 1: Pre-extract FPN embeddings for all training sequences.

Runs GatedYOLODetector once per frame (frozen), saves 896-dim ROI embeddings
per GT box to disk. Training then reads from cache instead of re-running YOLO.

Output: cache_dir/<seq_name>.pt  →  {frame_id: {"track_ids": [int], "embs": Tensor(N,896)}}

Usage:
    uv run scripts/train/cache_fpn_embeddings.py \\
        --ckpt runs/gated_det_v1/best.ckpt \\
        --cache-dir runs/reid_cache \\
        [--dancetrack-train datasets/DanceTrack/train] \\
        [--mot20-train datasets/MOT20/MOT20/train]
"""
# status: diagnostic

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.io as tv_io

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.yolo_gated_detector import (
    GatedDetConfig,
    GatedYOLODetector,
)


def load_gt(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    by_frame: dict[int, list] = {}
    with open(gt_path) as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 6:
                continue
            conf = int(cols[6]) if len(cols) > 6 else 1
            cls = int(cols[7]) if len(cols) > 7 else 1
            if conf != 1 or cls != 1:
                continue
            fid = int(cols[0])
            tid = int(cols[1])
            x, y, w, h = float(cols[2]), float(cols[3]), float(cols[4]), float(cols[5])
            by_frame.setdefault(fid, []).append((tid, x, y, x + w, y + h))
    return by_frame


class _SeqDataset(torch.utils.data.Dataset):
    """Per-frame dataset for a single sequence: loads and resizes images."""

    def __init__(self, img_dir: Path, frame_ids: list[int], img_size: int):
        self.img_dir = img_dir
        self.frame_ids = frame_ids
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int):
        fid = self.frame_ids[idx]
        candidates = (
            sorted(self.img_dir.glob(f"{fid:08d}.*"))
            or sorted(self.img_dir.glob(f"{fid:06d}.*"))
            or sorted(self.img_dir.glob(f"{fid:05d}.*"))
        )
        if not candidates:
            return fid, torch.zeros(3, self.img_size, self.img_size)
        img = tv_io.read_image(str(candidates[0])).float() / 255.0
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return fid, img


def cache_sequence(
    model: GatedYOLODetector,
    seq_dir: Path,
    out_path: Path,
    img_size: int,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
) -> int:
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    if not gt_path.exists() or not img_dir.exists():
        return 0

    gt = load_gt(gt_path)
    if not gt:
        return 0

    raw_paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if not raw_paths:
        return 0
    sample = tv_io.read_image(str(raw_paths[0]))
    orig_h, orig_w = sample.shape[1], sample.shape[2]
    sx, sy = img_size / orig_w, img_size / orig_h

    frame_ids = sorted(gt.keys())
    loader = torch.utils.data.DataLoader(
        _SeqDataset(img_dir, frame_ids, img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    result: dict[int, dict] = {}
    model.cache_feats = True

    with torch.no_grad():
        for fids_batch, imgs_batch in loader:
            # True batched YOLO inference: (B, 3, H, W) → feat_cache (B, C, H, W)
            imgs_batch = imgs_batch.to(device, non_blocking=True)
            B = imgs_batch.shape[0]

            model._feat_cache.clear()
            model(imgs_batch)  # feat_cache[scale] is now (B, C, H, W)

            if not model._feat_cache:
                continue

            # Build flat boxes list with batch indices for roi_align
            all_boxes: list[list[float]] = []  # [b_idx, x1, y1, x2, y2]
            all_tids: list[list[int]] = []
            frame_map: list[int] = []  # which fid each box group belongs to
            box_offsets: list[int] = [0]  # cumulative count

            for b_idx in range(B):
                fid = int(fids_batch[b_idx])
                tracks = gt.get(fid)
                if not tracks:
                    box_offsets.append(box_offsets[-1])
                    frame_map.append(fid)
                    all_tids.append([])
                    continue
                for t in tracks:
                    all_boxes.append(
                        [b_idx, t[1] * sx, t[2] * sy, t[3] * sx, t[4] * sy]
                    )
                all_tids.append([t[0] for t in tracks])
                frame_map.append(fid)
                box_offsets.append(box_offsets[-1] + len(tracks))

            if not all_boxes:
                continue

            batch_boxes_t = torch.tensor(
                all_boxes, device=device, dtype=torch.float32
            )  # (N,5)

            # ROI-align over batched feat maps
            parts: list[torch.Tensor] = []
            for scale, sf in (("p3", 1 / 8), ("p4", 1 / 16), ("p5", 1 / 32)):
                feat = model._feat_cache.get(scale)
                if feat is None:
                    continue
                from torchvision.ops import roi_align as _roi_align

                pooled = _roi_align(
                    feat.float(),
                    batch_boxes_t,
                    output_size=1,
                    spatial_scale=sf,
                    aligned=True,
                )
                parts.append(pooled.flatten(1))  # (N, C)

            if not parts:
                continue

            import torch.nn.functional as _F

            all_embs = _F.normalize(torch.cat(parts, dim=1), dim=1)  # (N, 896)

            # Split back per frame
            for b_idx in range(B):
                fid = frame_map[b_idx]
                lo, hi = box_offsets[b_idx], box_offsets[b_idx + 1]
                if lo == hi:
                    continue
                result[fid] = {
                    "track_ids": all_tids[b_idx],
                    "embs": all_embs[lo:hi].cpu(),
                }

    model.cache_feats = False
    torch.save(result, out_path)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--cache-dir", default="runs/reid_cache")
    parser.add_argument("--dancetrack-train", default="datasets/DanceTrack/train")
    parser.add_argument("--mot20-train", default="datasets/MOT20/MOT20/train")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frames per DataLoader batch (prefetched by workers)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader CPU workers for image loading",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-cache even if file exists"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading GatedYOLODetector from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: GatedDetConfig = ckpt["cfg"]
    model = GatedYOLODetector(ckpt["args"]["yolo_weights"], cfg=cfg, device=str(device))
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Collect all sequence directories
    roots = []
    if args.dancetrack_train and Path(args.dancetrack_train).exists():
        roots.append(Path(args.dancetrack_train))
    if args.mot20_train and Path(args.mot20_train).exists():
        roots.append(Path(args.mot20_train))

    seq_dirs = []
    for root in roots:
        seq_dirs.extend(sorted(d for d in root.iterdir() if d.is_dir()))

    print(f"Caching {len(seq_dirs)} sequences → {cache_dir}")
    t0 = time.perf_counter()
    total_frames = 0

    for seq_dir in seq_dirs:
        out_path = cache_dir / f"{seq_dir.name}.pt"
        if out_path.exists() and not args.force:
            print(f"  [skip] {seq_dir.name} (already cached)")
            continue
        n = cache_sequence(
            model,
            seq_dir,
            out_path,
            args.img_size,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        total_frames += n
        elapsed = time.perf_counter() - t0
        fps = total_frames / max(elapsed, 1e-6)
        print(f"  {seq_dir.name}: {n} frames  ({fps:.0f} frames/s total)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {total_frames} frames in {elapsed:.0f}s  →  {cache_dir}")


if __name__ == "__main__":
    main()
