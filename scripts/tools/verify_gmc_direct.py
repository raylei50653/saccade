#!/usr/bin/env python3
"""Verify GMC estimate_into_direct produces identical warp as estimate_into.

Compares C++ GMC outputs between the original path (internal gmc_stream_)
and the new estimate_into_direct (caller stream, no cross-stream sync).

Usage:
    uv run scripts/tools/verify_gmc_direct.py [--frames N] [--downscale D] [--seq NAME]
"""
# status: diagnostic

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))


def load_frames(seq_path: Path, n_frames: int = 100):
    """Load first n_frames frames from a MOT17 sequence using GPU decode."""
    from saccade.perception.eval.evaluator import TorchvisionGpuStreamer

    img_dir = seq_path / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    frames = []
    streamer = TorchvisionGpuStreamer(img_dir)
    for _, frame_gpu in zip(range(n_frames), streamer):
        frames.append(frame_gpu.permute(2, 0, 1).float() / 255.0)
    return frames


def run_comparison(frames: list[torch.Tensor], downscale: int = 4):
    from saccade_tracking_ext import GMC as CppGMC

    gmc_ref = CppGMC(downscale=downscale)
    gmc_direct = CppGMC(downscale=downscale)

    stream = torch.cuda.current_stream().cuda_stream

    max_err_dx = 0.0
    max_err_dy = 0.0
    total_dx = 0.0
    total_dy = 0.0
    n_valid = 0
    n_mismatch = 0

    print(
        f"{'Frame':>6}  {'ref_dx':>10}  {'ref_dy':>10}  "
        f"{'dir_dx':>10}  {'dir_dy':>10}  "
        f"{'err_dx':>10}  {'err_dy':>10}  {'match':>6}"
    )
    print("-" * 80)

    d_warp_ref = torch.zeros(6, dtype=torch.float32, device="cuda")
    d_warp_dir = torch.zeros(6, dtype=torch.float32, device="cuda")

    for i, frame in enumerate(frames):
        h, w = frame.shape[1], frame.shape[2]

        gmc_ref.estimate_into(
            frame.data_ptr(),
            w,
            h,
            stream,
            d_warp_ref.data_ptr(),
            True,
            True,
        )
        torch.cuda.synchronize()

        gmc_direct.estimate_into_direct(
            frame.data_ptr(),
            w,
            h,
            stream,
            d_warp_dir.data_ptr(),
        )
        torch.cuda.synchronize()

        warp_ref = d_warp_ref.cpu().tolist()
        warp_dir = d_warp_dir.cpu().tolist()

        dx_ref = warp_ref[2]
        dy_ref = warp_ref[5]
        dx_dir = warp_dir[2]
        dy_dir = warp_dir[5]

        err_dx = abs(dx_ref - dx_dir)
        err_dy = abs(dy_ref - dy_dir)

        if i > 0:
            max_err_dx = max(max_err_dx, err_dx)
            max_err_dy = max(max_err_dy, err_dy)
            total_dx += err_dx
            total_dy += err_dy
            n_valid += 1

        match = "✓" if (err_dx < 0.01 and err_dy < 0.01) else "✗"
        if err_dx >= 0.01 or err_dy >= 0.01:
            n_mismatch += 1

        print(
            f"{i:>6}  {dx_ref:>10.4f}  {dy_ref:>10.4f}  "
            f"{dx_dir:>10.4f}  {dy_dir:>10.4f}  "
            f"{err_dx:>10.4f}  {err_dy:>10.4f}  {match:>6}"
        )

    print("-" * 80)
    print(f"\nSummary ({n_valid} frames, skip first):")
    print(f"  Max error   dx={max_err_dx:.6f}  dy={max_err_dy:.6f}")
    if n_valid > 0:
        print(f"  Mean error  dx={total_dx / n_valid:.6f}  dy={total_dy / n_valid:.6f}")
    print(
        f"  Mismatches  {n_mismatch}/{n_valid} "
        f"({100 * n_mismatch / max(n_valid, 1):.1f}%)"
    )

    if n_mismatch == 0:
        print("\n✅ PASS: estimate_into_direct matches estimate_into exactly.")
        return 0
    elif max_err_dx < 0.5 and max_err_dy < 0.5:
        print("\n⚠️  WARN: sub-pixel differences within tolerance.")
        return 0
    else:
        print("\n❌ FAIL: significant differences detected.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Verify GMC estimate_into_direct")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--seq", default="MOT17-02-SDP")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    seq_path = Path(args.data_root) / args.split / args.seq
    if not seq_path.exists():
        print(f"❌ Sequence not found: {seq_path}")
        # Try to find any SDP sequence
        split_dir = Path(args.data_root) / args.split
        if split_dir.exists():
            seqs = sorted(
                d.name
                for d in split_dir.iterdir()
                if d.is_dir() and d.name.endswith("-SDP")
            )
            if seqs:
                args.seq = seqs[0]
                seq_path = split_dir / args.seq
                print(f"   Using: {seq_path}")

    if not seq_path.exists():
        print("❌ No sequences found.")
        return 1

    print(f"Loading up to {args.frames} frames from {seq_path} ...")
    frames = load_frames(seq_path, args.frames)
    print(f"Loaded {len(frames)} frames.\n")

    return run_comparison(frames, args.downscale)


if __name__ == "__main__":
    sys.exit(main())
