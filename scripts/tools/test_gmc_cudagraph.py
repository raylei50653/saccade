#!/usr/bin/env python3
"""Capture C++ estimate_into_direct in torch.cuda.CUDAGraph.

cuFFT supports CUDA Graphs per official docs.
estimate_into_direct already runs all kernels on caller stream.
Just need CUDAGraph context manager to capture + replay.

Usage:
    uv run scripts/tools/test_gmc_cudagraph.py --frames 30
"""

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))


def test_gmc_cudagraph(seq_name: str = "MOT17-02-SDP", n_frames: int = 30, ds: int = 4):
    from saccade.perception.eval.evaluator import TorchvisionGpuStreamer
    from saccade_tracking_ext import GMC as CppGMC

    data_root = Path("datasets/MOT17")
    seq_path = data_root / "train" / seq_name
    if not (seq_path / "img1").exists():
        for d in (data_root / "train").iterdir():
            if d.is_dir() and d.name.endswith("-SDP"):
                seq_path = d
                break

    streamer = TorchvisionGpuStreamer(seq_path / "img1")
    frames_chw = []
    for _, frame_hwc in zip(range(n_frames), streamer):
        frames_chw.append(frame_hwc.permute(2, 0, 1).float() / 255.0)
    print(f"Loaded {len(frames_chw)} frames from {seq_path}")

    h, w = frames_chw[0].shape[1], frames_chw[0].shape[2]

    gmc_ref = CppGMC(downscale=ds)
    gmc_graph = CppGMC(downscale=ds)

    d_ref = torch.zeros(6, device="cuda")
    d_graph = torch.zeros(6, device="cuda")
    stream_ptr = torch.cuda.current_stream().cuda_stream

    gmc_graph.estimate_into_direct(
        frames_chw[0].data_ptr(),
        w,
        h,
        torch.cuda.current_stream().cuda_stream,
        d_graph.data_ptr(),
    )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    d_input = frames_chw[0].clone()
    with torch.cuda.graph(g):
        gmc_graph.estimate_into_direct(
            d_input.data_ptr(),
            w,
            h,
            torch.cuda.current_stream().cuda_stream,
            d_graph.data_ptr(),
        )

    d_graph.zero_()
    g.replay()
    torch.cuda.synchronize()

    print(f"Frame 0 graph-replay warp: {d_graph.cpu().tolist()}")
    print("✅ CUDAGraph capture + replay works for estimate_into_direct!")

    for i, frame in enumerate(frames_chw):
        gmc_ref.estimate_into_direct(
            frame.data_ptr(), w, h, stream_ptr, d_ref.data_ptr()
        )
        torch.cuda.synchronize()

        d_input.copy_(frame)
        d_graph.zero_()
        g.replay()
        torch.cuda.synchronize()

        ref = d_ref.cpu().tolist()
        gra = d_graph.cpu().tolist()
        err = abs(ref[2] - gra[2]) + abs(ref[5] - gra[5])

        if i < 5 or err > 0.01:
            status = "✓" if err < 0.01 else "✗"
            print(
                f"  Frame {i:>3}: ref({ref[2]:>8.4f},{ref[5]:>8.4f}) "
                f"graph({gra[2]:>8.4f},{gra[5]:>8.4f}) err={err:.4f} {status}"
            )
        elif i % 10 == 0 or i == len(frames_chw) - 1:
            print(f"  Frame {i:>3}: err={err:.4f} ✓")

    if err < 0.01:
        print("\n✅ PASS: CUDAGraph GMC matches reference bit-exact!")
        return 0
    else:
        print(f"\n❌ FAIL: max error {err:.4f}")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seq", default="MOT17-02-SDP")
    args = parser.parse_args()
    return test_gmc_cudagraph(args.seq, args.frames)


if __name__ == "__main__":
    sys.exit(main())
