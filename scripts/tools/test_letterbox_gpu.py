"""Unit test: cpp_letterbox_gpu output matches PyTorch 3-op reference."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "build"))
sys.path.insert(0, str(project_root / "src"))

import torch  # noqa: E402

try:
    from saccade_tracking_ext import letterbox_gpu as cpp_letterbox_gpu
except ImportError:
    print("Cannot import saccade_tracking_ext.")
    sys.exit(1)


def pytorch_letterbox(src: torch.Tensor, dst_size: int, pad_val: float = 114.0 / 255.0):
    _, h_orig, w_orig = src.shape
    r = dst_size / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    y_off = (dst_size - h_new) // 2
    x_off = (dst_size - w_new) // 2
    # Use mode='nearest' to match torch default (no mode arg) behavior
    img_resized = torch.nn.functional.interpolate(
        src.unsqueeze(0), size=(h_new, w_new), mode="nearest"
    ).squeeze(0)
    canvas = torch.full((3, dst_size, dst_size), pad_val, device=src.device)
    canvas[:, y_off : y_off + h_new, x_off : x_off + w_new].copy_(img_resized)
    return canvas, x_off, y_off, w_new, h_new, r


def cuda_letterbox(src: torch.Tensor, dst_size: int, pad_val: float = 114.0 / 255.0):
    _, h_orig, w_orig = src.shape
    r = dst_size / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    y_off = (dst_size - h_new) // 2
    x_off = (dst_size - w_new) // 2
    canvas = torch.empty((3, dst_size, dst_size), device=src.device, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    cpp_letterbox_gpu(
        src.data_ptr(), w_orig, h_orig,
        canvas.data_ptr(), dst_size,
        x_off, y_off, w_new, h_new,
        pad_val, stream,
    )
    torch.cuda.synchronize()
    return canvas, x_off, y_off, w_new, h_new, r


def test_case(label, h_orig, w_orig, dst_size, max_err=1e-6):
    src = torch.rand(3, h_orig, w_orig, device="cuda")
    ref, x_off, y_off, w_new, h_new, r = pytorch_letterbox(src, dst_size)
    got, *_ = cuda_letterbox(src, dst_size)
    torch.cuda.synchronize()

    err = (ref - got).abs()
    max_abs = float(err.max())
    mean_abs = float(err.mean())

    # Padding region: should be exactly equal
    pad_mask = torch.ones(3, dst_size, dst_size, dtype=torch.bool, device="cuda")
    pad_mask[:, y_off:y_off+h_new, x_off:x_off+w_new] = False
    pad_err = float(err[pad_mask].max()) if pad_mask.any() else 0.0

    print(f"  [{label}] h={h_orig} w={w_orig} → {dst_size}×{dst_size}  "
          f"max={max_abs:.2e}  mean={mean_abs:.2e}  pad_err={pad_err:.2e}", end="")
    assert pad_err < 1e-6, f"Pad region mismatch: {pad_err}"
    assert max_abs < max_err, f"Content region max error too large: {max_abs}"
    print("  PASS")


if __name__ == "__main__":
    cases = [
        ("1920×1080 → 960", 1080, 1920, 960),
        ("1080×1920 → 960", 1920, 1080, 960),
        ("1920×1080 → 640", 1080, 1920, 640),
        ("720×1280 → 960",  1280,  720, 960),
        ("960×960 → 960",    960,  960, 960),   # square: no padding
        ("480×640 → 640",    640,  480, 640),
    ]
    failed = []
    for label, h, w, dst in cases:
        try:
            test_case(label, h, w, dst)
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(label)
    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(cases)} cases passed.")
