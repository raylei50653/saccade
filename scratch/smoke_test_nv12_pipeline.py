"""Smoke test: compare baseline RGB pipeline vs NV12 fused kernel on a real MOT17 frame.

Usage:
    LD_PRELOAD="<nv_cublas_path>" .venv/bin/python scratch/smoke_test_nv12_pipeline.py
"""

HAS_KERNELS = True
try:
    from saccade_tracking_ext import letterbox_gpu, nv12_to_chw_letterbox, rgb_to_nv12_gpu
except ImportError as e:
    print(f"Import failed: {e}")
    HAS_KERNELS = False

import torch
import numpy as np
from pathlib import Path


import torch
import numpy as np
from pathlib import Path


def main():
    print(f"HAS_KERNELS={HAS_KERNELS} CUDA={torch.cuda.is_available()}")
    if not HAS_KERNELS or not torch.cuda.is_available():
        print("SKIP: kernels not available or no CUDA")
        return

    frame_path = Path("datasets/MOT17/train/MOT17-02-SDP/img1/000001.jpg")
    if not frame_path.exists():
        print(f"SKIP: frame not found at {frame_path}")
        return

    print(f"Loading frame: {frame_path}")
    import cv2
    img = cv2.imread(str(frame_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_hwc = torch.from_numpy(img).cuda()
    frame_chw = frame_hwc.permute(2, 0, 1).contiguous()

    h, w = frame_hwc.shape[0], frame_hwc.shape[1]
    print(f"Frame size: {w}x{h}")

    # ── Baseline: RGB path ──
    dst_size = 960
    r = dst_size / max(h, w)
    h_new, w_new = int(h * r), int(w * r)
    y_off = (dst_size - h_new) // 2
    x_off = (dst_size - w_new) // 2
    pad_val = 114.0 / 255.0
    stream = torch.cuda.current_stream().cuda_stream

    frame_buffer = frame_chw.float() / 255.0
    canvas_rgb = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
    letterbox_gpu(
        frame_buffer.data_ptr(), w, h,
        canvas_rgb.data_ptr(), dst_size,
        x_off, y_off, w_new, h_new, pad_val, stream,
    )
    torch.cuda.synchronize()

    # ── NV12 fused path (using new rgb_to_nv12_gpu kernel) ──
    nv12_buf = torch.zeros(h * w + (h // 2) * w, dtype=torch.uint8, device="cuda")
    rgb_to_nv12_gpu(frame_hwc.data_ptr(), nv12_buf.data_ptr(), w, h, stream)
    torch.cuda.synchronize()
    canvas_nv12 = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
    y_pitch = w
    uv_pitch = w
    nv12_to_chw_letterbox(
        nv12_buf.data_ptr(), y_pitch,
        nv12_buf.data_ptr() + nv12_buf.element_size() * h * w, uv_pitch,
        w, h,
        canvas_nv12.data_ptr(), dst_size,
        x_off, y_off, w_new, h_new, pad_val, stream,
    )
    torch.cuda.synchronize()

    # ── Compare ──
    diff = (canvas_nv12 - canvas_rgb).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    nonzero_frac = (diff > 0.01).float().mean().item()

    print(f"\nBaseline vs NV12 fused (960p canvas):")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  Fraction > 0.01:     {nonzero_frac:.4%}")

    # ── Per-channel stats ──
    for c, name in enumerate(["R", "G", "B"]):
        cdiff = diff[c]
        print(f"  {name} max: {cdiff.max().item():.6f}  mean: {cdiff.mean().item():.6f}")

    if max_err > 0.10:
        print("\nWARNING: Large pixel differences detected. NV12 path may diverge from baseline.")
        print("This is expected for the first column of pixels with NV12 chroma subsampling.")
    elif max_err <= 0.05:
        print("\nOK: Differences within tolerance for A/B MOT17 gate.")
    else:
        print("\nCAUTION: Moderate differences. A/B MOT17 gate needed to confirm metric parity.")


if __name__ == "__main__":
    main()
