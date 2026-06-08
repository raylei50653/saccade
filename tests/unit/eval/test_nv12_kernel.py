import pytest
import torch

from typing import Tuple

pytestmark = pytest.mark.gpu

try:
    from saccade_tracking_ext import nv12_to_chw_letterbox

    HAS_NV12_KERNEL = True
except ImportError:
    HAS_NV12_KERNEL = False


def _rgb_to_nv12(rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
    """Build a packed NV12 buffer from an RGB tensor [3, H, W] float [0,1].

    Returns (nv12_buf, y_pitch, uv_pitch, H, W) where nv12_buf is uint8.
    """
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
    nv12 = torch.zeros(
        h * y_pitch + (h // 2) * uv_pitch, dtype=torch.uint8, device=rgb.device
    )

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
            cb_v = int(row_cb[col * 2].item())
            cr_v = int(row_cr[col * 2].item())
            nv12[start + col * 2] = cb_v
            nv12[start + col * 2 + 1] = cr_v

    return nv12, y_pitch, uv_pitch, h, w


def _nv12_to_rgb_reference(
    nv12_buf: torch.Tensor, y_pitch: int, uv_pitch: int, h: int, w: int
) -> torch.Tensor:
    """Pure-PyTorch full-range BT.601 NV12→RGB, returns [3, H, W] float [0,1]."""
    y_plane = nv12_buf[: h * w].view(h, w).float() / 255.0
    uv_start = h * w
    cb_plane = torch.zeros(h, w, dtype=torch.float32, device=nv12_buf.device)
    cr_plane = torch.zeros(h, w, dtype=torch.float32, device=nv12_buf.device)
    for row in range(h):
        uv_row = row // 2
        for col in range(w):
            uv_col = col // 2
            idx = uv_start + uv_row * uv_pitch + uv_col * 2
            cb_plane[row, col] = nv12_buf[idx].float() / 255.0
            cr_plane[row, col] = nv12_buf[idx + 1].float() / 255.0

    yf = (y_plane * 255.0).float()
    cb = (cb_plane * 255.0).float() - 128.0
    cr = (cr_plane * 255.0).float() - 128.0

    r = yf + 1.402 * cr
    g = yf - 0.344136 * cb - 0.714136 * cr
    b = yf + 1.772 * cb

    rgb = torch.stack([r, g, b], dim=0).clamp(0.0, 255.0) / 255.0
    return rgb


def _nv12_to_rgb_reference_vectorized(
    nv12_buf: torch.Tensor, y_pitch: int, uv_pitch: int, h: int, w: int
) -> torch.Tensor:
    """Vectorized pure-PyTorch full-range BT.601 NV12→RGB, returns [3, H, W] float [0,1]."""
    y_plane = nv12_buf[: h * w].view(h, w).float()
    uv_start = h * w

    uv = nv12_buf[uv_start : uv_start + (h // 2) * uv_pitch].view(h // 2, uv_pitch)
    cb_raw = uv[:, ::2][:, : w // 2].float()
    cr_raw = uv[:, 1::2][:, : w // 2].float()

    cb_up = (
        torch.nn.functional.interpolate(
            cb_raw.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
        )
        .squeeze(0)
        .squeeze(0)
    )
    cr_up = (
        torch.nn.functional.interpolate(
            cr_raw.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
        )
        .squeeze(0)
        .squeeze(0)
    )

    yf = y_plane
    cb = cb_up - 128.0
    cr = cr_up - 128.0

    r = yf + 1.402 * cr
    g = yf - 0.344136 * cb - 0.714136 * cr
    b = yf + 1.772 * cb

    rgb = torch.stack([r, g, b], dim=0).clamp(0.0, 255.0) / 255.0
    return rgb


def _build_test_rgb(h: int, w: int) -> torch.Tensor:
    """Build a simple RGB test pattern [3, H, W] float [0,1]."""
    torch.manual_seed(42)
    return torch.rand(3, h, w, device="cuda")


class TestNv12Kernel:
    def test_identity_geometry_matches_reference(self) -> None:
        if not HAS_NV12_KERNEL or not torch.cuda.is_available():
            pytest.skip("nv12_to_chw_letterbox kernel not available or no CUDA")

        h, w = 256, 256
        rgb_src = _build_test_rgb(h, w)
        nv12_buf, y_pitch, uv_pitch, _, _ = _rgb_to_nv12(rgb_src)

        ref_rgb = _nv12_to_rgb_reference_vectorized(nv12_buf, y_pitch, uv_pitch, h, w)

        dst = torch.zeros(3, h, w, device="cuda", dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        nv12_to_chw_letterbox(
            nv12_buf.data_ptr(),
            y_pitch,
            nv12_buf.data_ptr() + nv12_buf.element_size() * h * w,
            uv_pitch,
            w,
            h,
            dst.data_ptr(),
            h,
            0,
            0,
            w,
            h,
            114.0 / 255.0,
            stream,
        )
        torch.cuda.synchronize()

        max_err = (dst - ref_rgb).abs().max().item()
        assert max_err <= 0.02, f"Max error {max_err} exceeds tolerance"

    def test_identity_geometry_matches_pixel_ref(self) -> None:
        if not HAS_NV12_KERNEL or not torch.cuda.is_available():
            pytest.skip("nv12_to_chw_letterbox kernel not available or no CUDA")

        h, w = 256, 256
        rgb_src = _build_test_rgb(h, w)
        nv12_buf, y_pitch, uv_pitch, _, _ = _rgb_to_nv12(rgb_src)

        ref_rgb = _nv12_to_rgb_reference(nv12_buf, y_pitch, uv_pitch, h, w)

        dst = torch.zeros(3, h, w, device="cuda", dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        nv12_to_chw_letterbox(
            nv12_buf.data_ptr(),
            y_pitch,
            nv12_buf.data_ptr() + nv12_buf.element_size() * h * w,
            uv_pitch,
            w,
            h,
            dst.data_ptr(),
            h,
            0,
            0,
            w,
            h,
            114.0 / 255.0,
            stream,
        )
        torch.cuda.synchronize()

        max_err = (dst - ref_rgb).abs().max().item()
        assert max_err <= 0.02, f"Pixel reference max error {max_err} exceeds tolerance"

    def test_letterbox_geometry_pads_borders(self) -> None:
        if not HAS_NV12_KERNEL or not torch.cuda.is_available():
            pytest.skip("nv12_to_chw_letterbox kernel not available or no CUDA")

        src_h, src_w = 128, 256
        rgb_src = _build_test_rgb(src_h, src_w)
        nv12_buf, y_pitch, uv_pitch, _, _ = _rgb_to_nv12(rgb_src)

        dst_size = 320
        r = dst_size / max(src_h, src_w)
        h_new, w_new = int(src_h * r), int(src_w * r)
        y_off = (dst_size - h_new) // 2
        x_off = (dst_size - w_new) // 2
        pad_val = 114.0 / 255.0

        dst = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        nv12_to_chw_letterbox(
            nv12_buf.data_ptr(),
            y_pitch,
            nv12_buf.data_ptr() + nv12_buf.element_size() * src_h * src_w,
            uv_pitch,
            src_w,
            src_h,
            dst.data_ptr(),
            dst_size,
            x_off,
            y_off,
            w_new,
            h_new,
            pad_val,
            stream,
        )
        torch.cuda.synchronize()

        channel_pixels_outside = []
        for c in range(3):
            border_mask = torch.ones(
                dst_size, dst_size, dtype=torch.bool, device="cuda"
            )
            if h_new > 0 and w_new > 0:
                border_mask[y_off : y_off + h_new, x_off : x_off + w_new] = False
            outside_vals = dst[c][border_mask]
            if outside_vals.numel() > 0:
                channel_pixels_outside.append(outside_vals)

        if channel_pixels_outside:
            all_outside = torch.cat(channel_pixels_outside)
            max_dev = (all_outside - pad_val).abs().max().item()
            assert max_dev < 1e-6, f"Border pad values deviate from {pad_val}"

        overall_min = dst.min().item()
        overall_max = dst.max().item()
        assert overall_min >= 0.0, f"Output has negative values: min={overall_min}"
        assert overall_max <= 1.0, f"Output has values >1: max={overall_max}"

    def test_y_plane_equals_bt601_luma(self) -> None:
        """Verify Y plane == BT.601 luma (0.299R + 0.587G + 0.114B)."""
        h, w = 256, 256
        rgb_src = _build_test_rgb(h, w)
        nv12_buf, y_pitch, _, _, _ = _rgb_to_nv12(rgb_src)

        r = rgb_src[0]
        g = rgb_src[1]
        b = rgb_src[2]
        expected_luma = (0.299 * r + 0.587 * g + 0.114 * b) * 255.0

        y_plane = nv12_buf[: h * w].view(h, w).float()
        max_err = (y_plane - expected_luma).abs().max().item()
        assert max_err <= 1.0, f"Y plane round-trip error {max_err} > 1 (quantization)"

        y_plane_f32 = y_plane / 255.0
        luma_f32 = expected_luma / 255.0
        max_err_f32 = (y_plane_f32 - luma_f32).abs().max().item()
        assert max_err_f32 <= 0.01, f"Y plane float error {max_err_f32}"

    def test_strided_uv_pitch(self) -> None:
        """NV12 with UV pitch > required (simulating nvjpeg row padding)."""
        if not HAS_NV12_KERNEL or not torch.cuda.is_available():
            pytest.skip("nv12_to_chw_letterbox kernel not available or no CUDA")

        h, w = 128, 192
        # Use a padded UV pitch (nvjpeg-style alignment)
        uv_pitch_padded = ((w + 63) // 64) * 64
        y_pitch = w

        # Build NV12 with padded UV strides
        y_plane = torch.randint(0, 256, (h, w), dtype=torch.uint8, device="cuda")
        uv_plane = torch.randint(
            0, 256, (h // 2, uv_pitch_padded), dtype=torch.uint8, device="cuda"
        )

        nv12_buf = torch.zeros(
            h * y_pitch + (h // 2) * uv_pitch_padded, dtype=torch.uint8, device="cuda"
        )
        for row in range(h):
            nv12_buf[row * y_pitch : row * y_pitch + w] = y_plane[row]
        uv_base = h * y_pitch
        for row in range(h // 2):
            nv12_buf[
                uv_base + row * uv_pitch_padded : uv_base
                + row * uv_pitch_padded
                + uv_pitch_padded
            ] = uv_plane[row]

        dst_size = 128
        dst = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        nv12_to_chw_letterbox(
            nv12_buf.data_ptr(),
            y_pitch,
            nv12_buf.data_ptr() + nv12_buf.element_size() * h * y_pitch,
            uv_pitch_padded,
            w,
            h,
            dst.data_ptr(),
            dst_size,
            0,
            0,
            w,
            h,
            114.0 / 255.0,
            stream,
        )
        torch.cuda.synchronize()

        ref_rgb = _nv12_to_rgb_reference_vectorized(
            nv12_buf, y_pitch, uv_pitch_padded, h, w
        )[:, :dst_size, :dst_size]

        max_err = (dst[:, :dst_size, :dst_size] - ref_rgb).abs().max().item()
        assert max_err <= 0.02, (
            f"Strided UV pitch max error {max_err} exceeds tolerance"
        )


def _kernel_letterbox_pytorch_ref(
    nv12_buf: torch.Tensor,
    y_pitch: int,
    uv_pitch: int,
    src_h: int,
    src_w: int,
    dst_size: int,
    x_off: int,
    y_off: int,
    w_new: int,
    h_new: int,
    pad_val: float,
) -> torch.Tensor:
    """PyTorch reference matching nv12_to_chw_letterbox geometry.

    First decodes NV12→RGB at full src resolution, then applies letterbox
    with the same nearest-neighbor mapping used by the kernel.
    """
    rgb_src = _nv12_to_rgb_reference_vectorized(
        nv12_buf, y_pitch, uv_pitch, src_h, src_w
    )

    dst = torch.full(
        (3, dst_size, dst_size), pad_val, device=rgb_src.device, dtype=torch.float32
    )
    if h_new <= 0 or w_new <= 0:
        return dst

    for c in range(3):
        for dy in range(dst_size):
            ly = dy - y_off
            if ly < 0 or ly >= h_new:
                continue
            sy = min(int(ly * (src_h / h_new)), src_h - 1)
            for dx in range(dst_size):
                lx = dx - x_off
                if lx < 0 or lx >= w_new:
                    continue
                sx = min(int(lx * (src_w / w_new)), src_w - 1)
                dst[c, dy, dx] = rgb_src[c, sy, sx]

    return dst


class TestNv12KernelLetterbox:
    def test_letterbox_960p_matches_pytorch_ref(self) -> None:
        if not HAS_NV12_KERNEL or not torch.cuda.is_available():
            pytest.skip("nv12_to_chw_letterbox kernel not available or no CUDA")

        src_h, src_w = 720, 1280
        rgb_src = _build_test_rgb(src_h, src_w)
        nv12_buf, y_pitch, uv_pitch, _, _ = _rgb_to_nv12(rgb_src)

        dst_size = 960
        r = dst_size / max(src_h, src_w)
        h_new, w_new = int(src_h * r), int(src_w * r)
        y_off = (dst_size - h_new) // 2
        x_off = (dst_size - w_new) // 2
        pad_val = 114.0 / 255.0

        dst = torch.zeros(3, dst_size, dst_size, device="cuda", dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        nv12_to_chw_letterbox(
            nv12_buf.data_ptr(),
            y_pitch,
            nv12_buf.data_ptr() + nv12_buf.element_size() * src_h * src_w,
            uv_pitch,
            src_w,
            src_h,
            dst.data_ptr(),
            dst_size,
            x_off,
            y_off,
            w_new,
            h_new,
            pad_val,
            stream,
        )
        torch.cuda.synchronize()

        ref = _kernel_letterbox_pytorch_ref(
            nv12_buf,
            y_pitch,
            uv_pitch,
            src_h,
            src_w,
            dst_size,
            x_off,
            y_off,
            w_new,
            h_new,
            pad_val,
        )

        max_err = (dst - ref).abs().max().item()
        assert max_err <= 0.02, f"960p letterbox max error {max_err} exceeds tolerance"
