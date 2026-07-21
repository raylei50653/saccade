#!/usr/bin/env python3
"""Decompose _cross_scan_mamba: flip/stack prep vs MambaBlock(4B) vs unflip/mean.

Answers: how much would 'reading in flipped order' (eliminating torch.flip copies)
actually save, vs the irreducible 4x-batch SSM compute that produces the 4
directional results being averaged.
"""
# status: diagnostic

from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
import torch  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

WARMUP, ITERS, IMG = 50, 300, 640


def bench(fn) -> float:
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def main() -> None:
    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=IMG,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine="models/yolo/yolo26s_backbone_640_best.engine",
    )
    det.eval()
    head = det.mamba_head

    # Reconstruct x_small per scale (post input_proj + downsample), the input to
    # _cross_scan_mamba, for the 3 FPN scales at 640.
    frame = torch.rand(1, 3, IMG, IMG, device="cuda")
    with torch.no_grad():
        feats = list(det._trt_backbone.infer(frame))
        x_smalls = []
        for i, x in enumerate(feats):
            xp = head.input_proj[i](x)
            xs = head.downsample[i](xp)
            x_smalls.append(xs.clone())
    print(f"x_small shapes: {[tuple(t.shape) for t in x_smalls]}  (B,C,Hs,Ws)")

    def flip_prep(x):
        scans = torch.stack(
            [x, torch.flip(x, [2, 3]), torch.flip(x, [3]), torch.flip(x, [2])], 0
        )
        B, C = x.shape[0], x.shape[1]
        H, W = x.shape[2], x.shape[3]
        return scans.reshape(4 * B, C, H, W).flatten(2).transpose(1, 2)

    def block_4b(x, blocks):
        seq = flip_prep(x)
        for b in blocks:
            seq = b(seq) + seq
        return seq

    def block_1b(x, blocks):
        seq = x.flatten(2).transpose(1, 2)  # (B, L, C) — single direction
        for b in blocks:
            seq = b(seq) + seq
        return seq

    def full_cross(x, blocks, H, W):
        from saccade.perception.temporal_yolo.mamba_head import _cross_scan_mamba

        return _cross_scan_mamba(x, blocks, H, W)

    print(f"\n=== per-scale (ms/frame, {ITERS} iters) ===")
    print(
        f"{'scale':<8}{'full_cross':>11}{'flip_prep':>11}{'block_4B':>10}{'block_1B':>10}{'4B/1B':>7}"
    )
    tot_full = tot_prep = tot_4b = tot_1b = 0.0
    for i, xs in enumerate(x_smalls):
        blocks = head.mamba_blocks[i]
        H, W = xs.shape[2], xs.shape[3]
        t_full = bench(
            lambda xs=xs, blocks=blocks, H=H, W=W: full_cross(xs, blocks, H, W)
        )
        t_prep = bench(lambda xs=xs: flip_prep(xs))
        t_4b = bench(lambda xs=xs, blocks=blocks: block_4b(xs, blocks))
        t_1b = bench(lambda xs=xs, blocks=blocks: block_1b(xs, blocks))
        tot_full += t_full
        tot_prep += t_prep
        tot_4b += t_4b
        tot_1b += t_1b
        print(
            f"P{i + 3:<7}{t_full:>11.3f}{t_prep:>11.3f}{t_4b:>10.3f}{t_1b:>10.3f}{t_4b / t_1b:>6.2f}x"
        )
    print("-" * 57)
    print(
        f"{'Σ':<8}{tot_full:>11.3f}{tot_prep:>11.3f}{tot_4b:>10.3f}{tot_1b:>10.3f}{tot_4b / tot_1b:>6.2f}x"
    )
    print(f"\nflip/stack/mean glue (full - block_4B) ≈ {tot_full - tot_4b:.3f} ms")
    print(f"irreducible 4-dir SSM compute (block_4B)  ≈ {tot_4b:.3f} ms")
    print(
        f"if single-direction (block_1B)            ≈ {tot_1b:.3f} ms  (loses 2D context)"
    )


if __name__ == "__main__":
    main()
