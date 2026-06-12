from typing import Any, Union

import torch


def rgb_hwc_to_nv12_gpu(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB uint8 [H, W, 3] to packed NV12 uint8 on GPU.

    Uses the fused CUDA kernel when available, otherwise pure PyTorch.
    Forces contiguous layout before calling kernel (torchvision decode_jpeg
    returns CHW; .permute(1,2,0) creates a non-contiguous view).
    """
    rgb = rgb.contiguous()
    h, w = rgb.shape[0], rgb.shape[1]
    nv12 = torch.zeros(h * w + (h // 2) * w, dtype=torch.uint8, device=rgb.device)
    try:
        from saccade_tracking_ext import rgb_to_nv12_gpu

        rgb_to_nv12_gpu(
            rgb.data_ptr(),
            nv12.data_ptr(),
            w,
            h,
            torch.cuda.current_stream().cuda_stream,
        )
        return nv12
    except ImportError:
        pass

    r = rgb[:, :, 0].float()
    g = rgb[:, :, 1].float()
    b = rgb[:, :, 2].float()
    yf = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    yf = yf.clamp(0.0, 255.0).round().to(torch.uint8)
    cb = cb.clamp(0.0, 255.0).round().to(torch.uint8)
    cr = cr.clamp(0.0, 255.0).round().to(torch.uint8)
    nv12[: h * w] = yf.reshape(-1)
    uv_base = h * w
    cb_subsampled = cb[::2, ::2].contiguous()
    cr_subsampled = cr[::2, ::2].contiguous()
    for row in range(h // 2):
        offset = uv_base + row * w
        nv12[offset : offset + w : 2] = cb_subsampled[row]
        nv12[offset + 1 : offset + w : 2] = cr_subsampled[row]
    return nv12


def rgb_chw_to_nv12_gpu(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB float [3, H, W] in [0,1] to packed NV12 uint8 on GPU."""
    rgb_u8_hwc = (
        rgb.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
    )
    return rgb_hwc_to_nv12_gpu(rgb_u8_hwc)


class AdaptiveFramePool:
    def __init__(
        self, h: int, w: int, device: Union[str, torch.device] = "cuda"
    ) -> None:
        print(f"🕯️ Allocating VRAM Buffers for adaptive 960 tiled eval ({w}x{h})...")
        self.frame_buffer = torch.zeros((3, h, w), device=device, dtype=torch.float32)
        self.canvas_640p = torch.zeros(
            (3, 640, 640), device=device, dtype=torch.float32
        )
        self.canvas_960p = torch.zeros(
            (3, 960, 960), device=device, dtype=torch.float32
        )
        self.tiles_batch4 = torch.zeros(
            (4, 3, 640, 640), device=device, dtype=torch.float32
        )
        self.tiles_batch6 = torch.zeros(
            (6, 3, 640, 640), device=device, dtype=torch.float32
        )
        self.canvas_960p_flip = torch.zeros(
            (3, 960, 960), device=device, dtype=torch.float32
        )

        self.frame_buffer_nv12 = torch.zeros(
            (h * w * 3 // 2,), device=device, dtype=torch.uint8
        )
        self.use_nv12 = False
        self._rgb_current = True

        # Pre-allocated tile x/y offsets — avoids per-frame GPU tensor creation.
        self.tile_dx = torch.tensor(
            [0.0, 320.0, 0.0, 320.0], device=device, dtype=torch.float32
        ).view(4, 1, 1)
        self.tile_dy = torch.tensor(
            [0.0, 0.0, 320.0, 320.0], device=device, dtype=torch.float32
        ).view(4, 1, 1)

        # 3×2 tiling on 960p canvas (same scale as 2×2, adds a middle column).
        # x: 3 cols at stride=160 on 960p → [0:640],[160:800],[320:960] (75% x-overlap)
        # y: 2 rows at stride=320 on 960p → [0:640],[320:960]  (50% y-overlap, same as 2×2)
        self.tile_3x2_dx = torch.tensor(
            [0.0, 160.0, 320.0, 0.0, 160.0, 320.0], device=device, dtype=torch.float32
        ).view(6, 1, 1)
        self.tile_3x2_dy = torch.tensor(
            [0.0, 0.0, 0.0, 320.0, 320.0, 320.0], device=device, dtype=torch.float32
        ).view(6, 1, 1)

    def _import_nv12_ops(self) -> tuple[Any | None, Any | None]:
        try:
            from saccade_tracking_ext import (
                nv12_to_chw_letterbox,
                nv12_to_chw_resize,
            )

            return nv12_to_chw_letterbox, nv12_to_chw_resize
        except ImportError:
            return None, None

    def get_frame_luma(self) -> torch.Tensor:
        """Return a luma tensor [1, H, W] f32 [0,1] for GMC/optical flow.

        Always computed from frame_buffer (RGB float) for precision —
        NV12 Y-plane is uint8-quantized which costs ~0.5pp IDF1 in GMC.
        """
        if self.use_nv12 and not self._rgb_current:
            h = self.frame_h
            w = self.frame_w
            return self.frame_buffer_nv12[: h * w].view(1, h, w).float() / 255.0
        return (
            0.299 * self.frame_buffer[0:1]
            + 0.587 * self.frame_buffer[1:2]
            + 0.114 * self.frame_buffer[2:3]
        )

    def mark_rgb_current(self) -> None:
        self._rgb_current = True

    def mark_nv12_current(self) -> None:
        self._rgb_current = False

    @property
    def frame_h(self) -> int:
        return self.frame_buffer.shape[1]

    @property
    def frame_w(self) -> int:
        return self.frame_buffer.shape[2]

    def as_rgb_chw(self) -> torch.Tensor:
        if not self.use_nv12:
            return self.frame_buffer
        if self._rgb_current:
            return self.frame_buffer
        import torch.nn.functional as F

        h = self.frame_h
        w = self.frame_w
        y_plane = self.frame_buffer_nv12[: h * w].view(1, 1, h, w).float()
        uv_offset = h * w
        uv_raw = self.frame_buffer_nv12[uv_offset : uv_offset + (h // 2) * w]
        uv = uv_raw.view(h // 2, w).float()
        cb = uv[:, ::2][:, : w // 2]
        cr = uv[:, 1::2][:, : w // 2]
        cb_up = (
            F.interpolate(cb[None, None, :, :], size=(h, w), mode="nearest")
            .squeeze(0)
            .squeeze(0)
        )
        cr_up = (
            F.interpolate(cr[None, None, :, :], size=(h, w), mode="nearest")
            .squeeze(0)
            .squeeze(0)
        )
        yf = y_plane.squeeze(0).squeeze(0)
        cb_f = cb_up - 128.0
        cr_f = cr_up - 128.0
        r = yf + 1.402 * cr_f
        g = yf - 0.344136 * cb_f - 0.714136 * cr_f
        b = yf + 1.772 * cb_f
        rgb = torch.stack([r, g, b], dim=0).clamp(0.0, 255.0) / 255.0
        return rgb

    def prepare_canvas_640_stretch(self, h_orig: int, w_orig: int) -> torch.Tensor:
        if self.use_nv12:
            _, nv12_resize = self._import_nv12_ops()
            if nv12_resize is not None:
                stream = torch.cuda.current_stream().cuda_stream
                nv12_resize(
                    self.frame_buffer_nv12.data_ptr(),
                    w_orig,
                    self.frame_buffer_nv12.data_ptr()
                    + self.frame_buffer_nv12.element_size() * h_orig * w_orig,
                    w_orig,
                    w_orig,
                    h_orig,
                    self.canvas_640p.data_ptr(),
                    640,
                    640,
                    stream,
                )
                return self.canvas_640p

        img_input = torch.nn.functional.interpolate(
            self.as_rgb_chw().unsqueeze(0),
            size=(640, 640),
            mode="bilinear",
            align_corners=False,
        )
        self.canvas_640p.copy_(img_input[0])
        return self.canvas_640p

    def prepare_canvas_640_letterbox(
        self, h_orig: int, w_orig: int
    ) -> tuple[torch.Tensor, float, int, int, int, int]:
        r = 640.0 / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * r), int(w_orig * r)
        y_off = (640 - h_new) // 2
        x_off = (640 - w_new) // 2
        nv12_letterbox, _ = self._import_nv12_ops()
        if self.use_nv12 and nv12_letterbox is not None:
            stream = torch.cuda.current_stream().cuda_stream
            nv12_letterbox(
                self.frame_buffer_nv12.data_ptr(),
                w_orig,
                self.frame_buffer_nv12.data_ptr()
                + self.frame_buffer_nv12.element_size() * h_orig * w_orig,
                w_orig,
                w_orig,
                h_orig,
                self.canvas_640p.data_ptr(),
                640,
                x_off,
                y_off,
                w_new,
                h_new,
                114.0 / 255.0,
                stream,
            )
        else:
            img_resized = torch.nn.functional.interpolate(
                self.as_rgb_chw().unsqueeze(0),
                size=(h_new, w_new),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            self.canvas_640p.fill_(114.0 / 255.0)
            self.canvas_640p[:, y_off : y_off + h_new, x_off : x_off + w_new].copy_(
                img_resized
            )
        return self.canvas_640p, r, h_new, w_new, y_off, x_off

    def prepare_canvas_960_letterbox(
        self, h_orig: int, w_orig: int
    ) -> tuple[torch.Tensor, float, int, int, int, int]:
        r = 960.0 / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * r), int(w_orig * r)
        y_off = (960 - h_new) // 2
        x_off = (960 - w_new) // 2
        nv12_letterbox, _ = self._import_nv12_ops()
        if self.use_nv12 and nv12_letterbox is not None:
            stream = torch.cuda.current_stream().cuda_stream
            nv12_letterbox(
                self.frame_buffer_nv12.data_ptr(),
                w_orig,
                self.frame_buffer_nv12.data_ptr()
                + self.frame_buffer_nv12.element_size() * h_orig * w_orig,
                w_orig,
                w_orig,
                h_orig,
                self.canvas_960p.data_ptr(),
                960,
                x_off,
                y_off,
                w_new,
                h_new,
                114.0 / 255.0,
                stream,
            )
        else:
            img_resized = torch.nn.functional.interpolate(
                self.as_rgb_chw().unsqueeze(0),
                size=(h_new, w_new),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            self.canvas_960p.fill_(114.0 / 255.0)
            self.canvas_960p[:, y_off : y_off + h_new, x_off : x_off + w_new].copy_(
                img_resized
            )
        return self.canvas_960p, r, h_new, w_new, y_off, x_off
