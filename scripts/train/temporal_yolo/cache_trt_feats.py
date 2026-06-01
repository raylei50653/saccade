"""
Pre-extract TRT backbone features (P3/P4/P5) for all MOT17 training frames.

Output: cache_dir/<seq_name>.pt  →  [dict] {frame_idx: (p3_fp16, p4_fp16, p5_fp16)}

FP16 storage: ~2.8 MB/frame.  5316 frames ≈ 15 GB.

Usage:
    uv run scripts/train/temporal_yolo/cache_trt_feats.py \
        --data-root datasets/MOT17 \
        --cache-dir runs/trt_feat_cache \
        [--seqs MOT17-02-SDP,MOT17-04-SDP] \
        [--batch-size 8]
"""

import argparse
import sys
import time
import torch
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "build"))
import saccade_tracking_ext  # noqa
from saccade.perception.temporal_yolo.mamba_gated_detector import TRTYoloBackbone  # noqa
from saccade.perception.temporal_yolo.data_pipeline import resize_stretch_batch_gpu  # noqa
import torchvision.io as tv_io


def load_frames(
    seq_dir: Path, img_size: int, device: torch.device, batch_size: int = 8
):
    """Load and resize all frames in a sequence, return list of tensors."""
    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg"))
    tensors = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_imgs = torch.stack(
            [tv_io.read_image(str(p)).float() / 255.0 for p in batch_paths]
        ).to(device, non_blocking=True)
        batch_resized = resize_stretch_batch_gpu(batch_imgs, img_size, device)
        for j in range(len(batch_paths)):
            tensors.append(batch_resized[j : j + 1])
    return tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument(
        "--trt-engine", default="models/yolo/yolo26s_backbone_640_batch32.engine"
    )
    p.add_argument("--cache-dir", default="runs/trt_feat_cache")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8, help="Frames per TRT batch")
    p.add_argument("--seqs", default="", help="Comma-separated; empty = all SDP")
    args = p.parse_args()

    dev = torch.device("cuda")
    engine_path = _root / args.trt_engine
    if not engine_path.exists():
        raise FileNotFoundError(str(engine_path))
    trunk = TRTYoloBackbone(str(engine_path))

    data_root = _root / args.data_root / "train"
    if args.seqs:
        seq_names = [s.strip() for s in args.seqs.split(",") if s.strip()]
        seq_dirs = [data_root / s for s in seq_names if (data_root / s).is_dir()]
    else:
        seq_dirs = sorted(
            d for d in data_root.iterdir() if d.is_dir() and d.name.endswith("-SDP")
        )

    cache_dir = _root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    t0 = time.perf_counter()

    for seq_dir in seq_dirs:
        name = seq_dir.name
        out_path = cache_dir / f"{name}.pt"
        if out_path.exists():
            print(f"[skip] {name}")
            continue

        print(f"[{name}] Loading frames...")
        frames = load_frames(seq_dir, args.img_size, dev, args.batch_size)
        n_frames = len(frames)
        print(f"  {n_frames} frames, extracting features (batch={args.batch_size})...")

        cache = {}
        for i in range(0, n_frames, args.batch_size):
            batch = torch.cat(frames[i : i + args.batch_size], dim=0)
            p3, p4, p5 = trunk.infer(batch)
            for j in range(batch.shape[0]):
                cache[i + j] = (
                    p3[j : j + 1].half().cpu(),
                    p4[j : j + 1].half().cpu(),
                    p5[j : j + 1].half().cpu(),
                )

            if (i + 1) % 50 == 0 or i + args.batch_size >= n_frames:
                eta = (
                    (n_frames - i)
                    * (time.perf_counter() - t0)
                    / max(i + args.batch_size, 1)
                )
                print(
                    f"\r  {min(i + args.batch_size, n_frames)}/{n_frames}  ETA {eta:.0f}s",
                    end="",
                    flush=True,
                )

        torch.save(cache, out_path)
        total_frames += n_frames
        dt = time.perf_counter() - t0
        print(
            f"\r  ✓ {name}: {n_frames} frames saved ({os.path.getsize(out_path) / 1e9:.1f} GB)  {dt / 60:.1f}min total"
        )

    print(f"\nDone: {total_frames} frames → {cache_dir}")


import os  # noqa

if __name__ == "__main__":
    main()
