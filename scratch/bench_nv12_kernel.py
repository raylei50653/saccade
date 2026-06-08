import torch
import time
from typing import Optional

try:
    from saccade_tracking_ext import letterbox_gpu, nv12_to_chw_letterbox

    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

WARMUP = 10
REPEATS = 100


def _rgb_to_nv12(rgb: torch.Tensor):
    _, h, w = rgb.shape
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    yf = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0 / 255.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0 / 255.0
    yf = yf.clamp(0.0, 1.0)
    cb = cb.clamp(0.0, 1.0)
    cr = cr.clamp(0.0, 1.0)
    y_pitch = w
    uv_pitch = w
    nv12 = torch.zeros(h * y_pitch + (h // 2) * uv_pitch, dtype=torch.uint8, device=rgb.device)
    y_vals = (yf * 255.0).round().to(torch.uint8).contiguous()
    for row in range(h):
        nv12[row * y_pitch : row * y_pitch + w] = y_vals[row]
    cb_vals = (cb * 255.0).round().to(torch.uint8)
    cr_vals = (cr * 255.0).round().to(torch.uint8)
    uv_base = h * y_pitch
    for row in range(h // 2):
        row_cb = cb_vals[row * 2]
        row_cr = cr_vals[row * 2]
        start = uv_base + row * uv_pitch
        for col in range(w // 2):
            nv12[start + col * 2] = row_cb[col * 2].to(torch.uint8)
            nv12[start + col * 2 + 1] = row_cr[col * 2].to(torch.uint8)
    return nv12, y_pitch, uv_pitch


def _cuda_event_timed(fn, stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    fn()
    end.record(stream)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def bench_current_rgb_path(
    frame_rgb: torch.Tensor, src_h: int, src_w: int,
    dst_size: int, x_off: int, y_off: int, w_new: int, h_new: int
) -> float:
    stream_obj = torch.cuda.Stream()
    stream = stream_obj.cuda_stream
    pad_val = 114.0 / 255.0

    def step():
        frame_chw = frame_rgb.permute(2, 0, 1).float() / 255.0
        dst = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
        letterbox_gpu(
            frame_chw.data_ptr(), src_w, src_h,
            dst.data_ptr(), dst_size,
            x_off, y_off, w_new, h_new,
            pad_val, stream,
        )

    with torch.cuda.stream(stream_obj):
        for _ in range(WARMUP):
            step()
        times = []
        for _ in range(REPEATS):
            times.append(_cuda_event_timed(step, stream_obj))
    return sum(times) / len(times)


def bench_nv12_fused_path(
    nv12_buf: torch.Tensor, y_pitch: int, uv_pitch: int,
    src_h: int, src_w: int,
    dst_size: int, x_off: int, y_off: int, w_new: int, h_new: int
) -> float:
    stream_obj = torch.cuda.Stream()
    stream = stream_obj.cuda_stream
    pad_val = 114.0 / 255.0

    def step():
        dst = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
        nv12_to_chw_letterbox(
            nv12_buf.data_ptr(), y_pitch,
            nv12_buf.data_ptr() + nv12_buf.element_size() * src_h * src_w, uv_pitch,
            src_w, src_h,
            dst.data_ptr(), dst_size,
            x_off, y_off, w_new, h_new,
            pad_val, stream,
        )

    with torch.cuda.stream(stream_obj):
        for _ in range(WARMUP):
            step()
        times = []
        for _ in range(REPEATS):
            times.append(_cuda_event_timed(step, stream_obj))
    return sum(times) / len(times)


def main():
    if not HAS_KERNELS or not torch.cuda.is_available():
        print("Kernels not available or no CUDA — skipping benchmark")
        return

    configs = [
        ("1080p (1920x1080)", 1080, 1920, 960),
        ("720p (1280x720)", 720, 1280, 960),
        ("4K (3840x2160)", 2160, 3840, 960),
    ]

    print(f"{'Config':<24} {'Current RGB (ms)':>16} {'Fused NV12 (ms)':>16} {'Speedup':>10} {'Bandwidth Saved':>18}")
    print("-" * 90)

    for name, src_h, src_w, dst_size in configs:
        torch.manual_seed(42)
        frame_rgb = torch.randint(0, 256, (src_h, src_w, 3), dtype=torch.uint8, device="cuda")
        rgb_src = frame_rgb.float() / 255.0
        nv12_buf, y_pitch, uv_pitch = _rgb_to_nv12(rgb_src.permute(2, 0, 1))

        r = dst_size / max(src_h, src_w)
        h_new, w_new = int(src_h * r), int(src_w * r)
        y_off = (dst_size - h_new) // 2
        x_off = (dst_size - w_new) // 2

        cur_time = bench_current_rgb_path(frame_rgb, src_h, src_w, dst_size, x_off, y_off, w_new, h_new)
        fused_time = bench_nv12_fused_path(nv12_buf, y_pitch, uv_pitch, src_h, src_w, dst_size, x_off, y_off, w_new, h_new)

        speedup = cur_time / fused_time if fused_time > 0 else 0.0

        rgb_bytes = src_h * src_w * 3
        nv12_bytes = src_h * src_w * 3 // 2
        saved_frac = (rgb_bytes - nv12_bytes) / rgb_bytes * 100

        print(f"{name:<24} {cur_time:>14.3f} ms {fused_time:>14.3f} ms {speedup:>8.2f}x {saved_frac:>15.1f}%")


if __name__ == "__main__":
    main()
