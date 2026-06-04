from __future__ import annotations

from typing import Optional, cast

import cv2
import numpy as np
import torch


class SparseOpticalFlowGMC:
    """Estimate a prev->curr 2x3 affine warp from sparse LK optical flow."""

    def __init__(
        self,
        downscale: int = 8,
        max_corners: int = 100,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
    ) -> None:
        self.downscale = max(1, int(downscale))
        self.max_corners = max(10, int(max_corners))
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

    def _prepare_gray(
        self, frame_tensor: torch.Tensor
    ) -> tuple[np.ndarray, float, float]:
        with torch.no_grad():
            t = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
            h = max(1, int(t.shape[-2] / self.downscale))
            w = max(1, int(t.shape[-1] / self.downscale))
            small = torch.nn.functional.interpolate(
                t, size=(h, w), mode="area"
            ).squeeze(0)
            gray_t = 0.299 * small[0] + 0.587 * small[1] + 0.114 * small[2]
            gray = (gray_t * 255).byte().cpu().numpy()
        sx = float(t.shape[-1]) / float(w)
        sy = float(t.shape[-2]) / float(h)
        return gray, sx, sy

    def estimate(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        curr_gray, sx, sy = self._prepare_gray(frame_tensor)
        h_gmc = None

        if self.prev_gray is not None:
            if self.prev_points is None or len(self.prev_points) < 20:
                self.prev_points = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                )
            if self.prev_points is not None:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, self.prev_points, cast(np.ndarray, None)
                )
                if status is not None and curr_pts is not None:
                    good_prev = self.prev_points[status.flatten() == 1]
                    good_curr = curr_pts[status.flatten() == 1]
                    if len(good_prev) > 10:
                        warp, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                        if warp is not None:
                            warp[0, 2] *= sx
                            warp[1, 2] *= sy
                            h_gmc = torch.from_numpy(warp.astype(np.float32)).to(
                                frame_tensor.device
                            )
                    self.prev_points = good_curr.reshape(-1, 1, 2)

        self.prev_gray = curr_gray
        return h_gmc

    def apply(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self.estimate(frame_tensor)


class GlobalMotionCompensator(SparseOpticalFlowGMC):
    """Backward-compatible alias for the eval GMC estimator."""

    def __init__(
        self, method: str = "lk", device: str = "cuda", downscale: int = 8
    ) -> None:
        super().__init__(downscale=downscale)
        self.method = method
        self.device = device


class PyGraphedGMC:
    """Drop-in pure-Python GMC phase correlation that is fully compatible with CUDA Graph capture.

    Uses torch.fft and vectorized neighborhood peak finding to achieve subpixel accuracy
    mathematically identical to C++ cuFFT GMC.
    """

    def __init__(self, downscale: int = 4) -> None:
        self.downscale = downscale
        self._captured = False
        self.h_orig = 0
        self.w_orig = 0
        self.prev_gray = None
        self.warp_out = None
        self._graphed = None
        self._last_pcr = 0.0
        self._fg_boxes = None

    def ensure_buffers(self, h_orig: int, w_orig: int, device: torch.device) -> None:
        if self._captured and self.h_orig == h_orig and self.w_orig == w_orig:
            return

        import torch.nn.functional as F

        self.h_orig = h_orig
        self.w_orig = w_orig
        d = device
        self.h_ds = h_orig // self.downscale
        self.w_ds = w_orig // self.downscale
        self._ds_tensor = torch.tensor(
            [float(self.downscale)], dtype=torch.float32, device=d
        )
        self.prev_gray = torch.zeros(
            self.h_ds, self.w_ds, dtype=torch.float32, device=d
        )

        hh = torch.hann_window(self.h_ds, device=d, dtype=torch.float32)
        hw = torch.hann_window(self.w_ds, device=d, dtype=torch.float32)
        self.hann = (hh.unsqueeze(1) * hw.unsqueeze(0)).contiguous()

        self.warp_out = torch.zeros(6, dtype=torch.float32, device=d)
        self._warp_id = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=d
        )
        self.update_indices = torch.tensor([2, 5], dtype=torch.int64, device=d)
        self.offsets_dy = torch.tensor(
            [-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.int64, device=d
        )
        self.offsets_dx = torch.tensor(
            [-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.int64, device=d
        )
        self._frame_buf = torch.zeros(3, h_orig, w_orig, dtype=torch.float32, device=d)
        self._gray_w = torch.tensor([0.299, 0.587, 0.114], device=d).view(3, 1, 1)

        hds, wds = self.h_ds, self.w_ds
        ds_t = self._ds_tensor
        hann = self.hann
        pg = self.prev_gray
        warp_id = self._warp_id

        with torch.no_grad():
            # warmup 1
            fb = self._frame_buf
            gray_ds = F.interpolate(
                fb.unsqueeze(0), size=(hds, wds), mode="nearest"
            ).squeeze(0)
            gray_ds = (gray_ds * self._gray_w).sum(dim=0) * hann
            fft_p = torch.fft.rfft2(pg)
            fft_c = torch.fft.rfft2(gray_ds)
            cross = fft_p.conj() * fft_c
            cross = cross / (cross.abs() + 1e-6)
            torch.fft.irfft2(cross, s=(hds, wds))  # warmup (result discarded)
            torch.cuda.synchronize()
            pg.zero_()

            # warmup 2
            gray_ds2 = F.interpolate(
                fb.unsqueeze(0), size=(hds, wds), mode="nearest"
            ).squeeze(0)
            gray_ds2 = (gray_ds2 * self._gray_w).sum(dim=0) * hann
            fft_p2 = torch.fft.rfft2(pg)
            fft_c2 = torch.fft.rfft2(gray_ds2)
            cross2 = fft_p2.conj() * fft_c2
            cross2 = cross2 / (cross2.abs() + 1e-6)
            torch.fft.irfft2(cross2, s=(hds, wds))  # warmup (result discarded)
            torch.cuda.synchronize()
            pg.zero_()

            def _graph_fn(frame_chw, pg_in, warp_out):
                gray_ds = F.interpolate(
                    frame_chw.unsqueeze(0), size=(hds, wds), mode="nearest"
                ).squeeze(0)
                gray_ds = (gray_ds * self._gray_w).sum(dim=0) * hann

                fft_p = torch.fft.rfft2(pg_in)
                fft_c = torch.fft.rfft2(gray_ds)
                cross = fft_p.conj() * fft_c
                cross = cross / (cross.abs() + 1e-6)
                corr_irfft = torch.fft.irfft2(cross, s=(hds, wds))

                max_v, max_i = corr_irfft.view(-1).max(dim=0)
                py = max_i // wds
                px = max_i % wds

                n_pixels = hds * wds
                sum_sq = (corr_irfft * corr_irfft).sum()
                rms = torch.sqrt(sum_sq / n_pixels + 1e-12)
                ratio = max_v / (rms + 1e-12)

                ny = (py + self.offsets_dy + hds) % hds
                nx = (px + self.offsets_dx + wds) % wds
                n_indices = ny * wds + nx

                v_all = torch.gather(corr_irfft.view(-1), 0, n_indices)
                v_all = torch.clamp(v_all, min=0.0)

                px_all = px.float() + self.offsets_dx.float()
                py_all = py.float() + self.offsets_dy.float()

                sum_v = v_all.sum()
                sum_x = (v_all * px_all).sum()
                sum_y = (v_all * py_all).sum()

                peak_x = sum_x / (sum_v + 1e-6)
                peak_y = sum_y / (sum_v + 1e-6)

                valid = ratio >= 5.0

                px_f = peak_x - torch.where(peak_x > wds / 2.0, float(wds), 0.0)
                py_f = peak_y - torch.where(peak_y > hds / 2.0, float(hds), 0.0)

                warp_out.copy_(warp_id)
                tx = torch.where(valid, px_f * ds_t, 0.0)
                ty = torch.where(valid, py_f * ds_t, 0.0)

                updates = torch.cat([tx.view(1), ty.view(1)])
                warp_out.scatter_(0, self.update_indices, updates)

                pg_in.copy_(gray_ds)

                return warp_out, pg_in, ratio

            fb_clone = self._frame_buf.clone()
            pg_clone = self.prev_gray.clone()
            warp_clone = self.warp_out.clone()
            self._graphed = torch.cuda.make_graphed_callables(
                _graph_fn, (fb_clone, pg_clone, warp_clone)
            )
            self.prev_gray.zero_()

        self._captured = True
        print(f"🕯️ [PyGMC] Captured graph for img=({hds}×{wds} ds={self.downscale})")

    def estimate_into_direct(
        self, frame_chw: torch.Tensor, d_out_warp: torch.Tensor
    ) -> None:
        h, w = frame_chw.shape[-2], frame_chw.shape[-1]
        self.ensure_buffers(h, w, frame_chw.device)
        self._frame_buf.copy_(frame_chw)
        if self._fg_boxes is not None and len(self._fg_boxes) > 0:
            for box in self._fg_boxes:
                x1, y1, x2, y2 = box
                ix1 = max(0, min(int(x1.item()), w - 1))
                iy1 = max(0, min(int(y1.item()), h - 1))
                ix2 = max(0, min(int(x2.item()), w - 1))
                iy2 = max(0, min(int(y2.item()), h - 1))
                if ix2 >= ix1 and iy2 >= iy1:
                    self._frame_buf[:, iy1 : iy2 + 1, ix1 : ix2 + 1] = 0.0
            self._fg_boxes = None

        warp_out, pg_out, ratio_out = self._graphed(
            self._frame_buf, self.prev_gray, self.warp_out
        )
        self.prev_gray.copy_(pg_out)
        d_out_warp.copy_(warp_out)
        self._last_pcr = ratio_out.item()

    def estimate(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        h, w = frame_tensor.shape[-2], frame_tensor.shape[-1]
        self.ensure_buffers(h, w, frame_tensor.device)
        self._frame_buf.copy_(frame_tensor)
        warp_out, pg_out, ratio_out = self._graphed(
            self._frame_buf, self.prev_gray, self.warp_out
        )
        self.prev_gray.copy_(pg_out)
        self._last_pcr = ratio_out.item()
        return warp_out.clone()

    def apply(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self.estimate(frame_tensor)

    def set_fg_mask_boxes_tensor(self, boxes: torch.Tensor) -> None:
        self._fg_boxes = boxes

    def set_fg_mask_boxes(self, boxes_flat: list[float]) -> None:
        if boxes_flat:
            device = self._frame_buf.device if self._frame_buf is not None else "cuda"
            self._fg_boxes = torch.tensor(
                boxes_flat, dtype=torch.float32, device=device
            ).view(-1, 4)

    def set_fg_mask_boxes_gpu(self, d_boxes_ptr: int, n_boxes: int) -> None:
        # Pointers not supported in python; evaluator intercepts and calls set_fg_mask_boxes_tensor
        pass

    def pcr_score(self) -> float:
        return self._last_pcr

    def reset(self) -> None:
        if self.prev_gray is not None:
            self.prev_gray.zero_()
