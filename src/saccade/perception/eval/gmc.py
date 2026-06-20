from __future__ import annotations

from typing import Any, Callable, Optional, cast

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

    def __init__(self, downscale: int = 4, pcr_thresh: float = 5.0) -> None:
        self.downscale = downscale
        self.pcr_thresh = float(pcr_thresh)
        self._captured = False
        self.h_orig = 0
        self.w_orig = 0
        self.prev_gray: Optional[torch.Tensor] = None
        self.warp_out: Optional[torch.Tensor] = None
        self._graphed: Optional[Callable[..., Any]] = None
        self._last_pcr = 0.0
        self._fg_boxes: Optional[torch.Tensor] = None

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
        self._pcr_thresh_tensor = torch.tensor(
            self.pcr_thresh, dtype=torch.float32, device=d
        )
        self.prev_gray = torch.zeros(
            self.h_ds, self.w_ds, dtype=torch.float32, device=d
        )

        hh = torch.hann_window(self.h_ds, periodic=False, device=d, dtype=torch.float32)
        hw = torch.hann_window(self.w_ds, periodic=False, device=d, dtype=torch.float32)
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
        pcr_thresh = self._pcr_thresh_tensor
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

            def _graph_fn(
                frame_chw: torch.Tensor, pg_in: torch.Tensor, warp_out: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

                px_f = peak_x - torch.where(peak_x > wds / 2.0, float(wds), 0.0)
                py_f = peak_y - torch.where(peak_y > hds / 2.0, float(hds), 0.0)

                # 25% displacement plausibility cap (matches C++ gmc_kernel.cu)
                displ_ok = (px_f.abs() <= wds * 0.25) & (py_f.abs() <= hds * 0.25)

                # Soft confidence scaling (matches C++ gmc_kernel.cu)
                confidence = torch.where(
                    ratio < pcr_thresh, torch.clamp(ratio / pcr_thresh, 0.0, 1.0), 1.0
                )

                warp_out.copy_(warp_id)
                tx = torch.where(displ_ok, px_f * ds_t * confidence, 0.0)
                ty = torch.where(displ_ok, py_f * ds_t * confidence, 0.0)

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

        assert (
            self._graphed is not None
            and self.prev_gray is not None
            and self.warp_out is not None
        )
        warp_out, pg_out, ratio_out = self._graphed(
            self._frame_buf, self.prev_gray, self.warp_out
        )
        self.prev_gray.copy_(pg_out)
        d_out_warp.copy_(warp_out)
        self._last_pcr = ratio_out.item()

    def estimate(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        h, w = frame_tensor.shape[-2], frame_tensor.shape[-1]
        self.ensure_buffers(h, w, frame_tensor.device)
        assert (
            self._graphed is not None
            and self.prev_gray is not None
            and self.warp_out is not None
        )
        self._frame_buf.copy_(frame_tensor)
        warp_out, pg_out, ratio_out = self._graphed(
            self._frame_buf, self.prev_gray, self.warp_out
        )
        self.prev_gray.copy_(pg_out)
        self._last_pcr = ratio_out.item()
        return cast(torch.Tensor, warp_out.clone())

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

    def set_fg_mask_boxes_gpu(
        self, d_boxes_ptr: int, n_boxes: int, stream_ptr: int = 0
    ) -> None:
        # Pointers not supported in python; evaluator intercepts and calls set_fg_mask_boxes_tensor
        pass

    def pcr_score(self) -> float:
        return self._last_pcr

    def reset(self) -> None:
        if self.prev_gray is not None:
            self.prev_gray.zero_()


class TilePhaseCorrAffineGMC:
    """Tile-based phase-correlation GMC that recovers a 4-DOF similarity (s, θ, tx, ty).

    Splits the downscaled gray frame into an overlapping grid of tiles, runs a
    batched phase correlation per tile to get a per-tile translation + PCR
    confidence, then robustly fits a similarity transform (PCR-weighted LS with
    one Huber reweight) from the tile-center correspondences. This captures the
    camera rotation/zoom that the global translation-only PCR cannot.

    Fallback policy (per design): if the similarity fit is not confident or is
    physically implausible, fall back to the **global PCR translation** (the
    proven current GMC behaviour) — never to identity. Identity is used only
    when even the global translation fails its own PCR/displacement gate, which
    is exactly what the current GMC already does. Tile mode is therefore a
    strict superset of the global translation GMC: match-or-improve by construction.
    """

    def __init__(
        self,
        downscale: int = 8,
        tile: int = 64,
        overlap: float = 0.5,
        pcr_thresh: float = 5.0,
        min_tiles: int = 4,
        max_disp_frac: float = 0.25,
        max_scale_dev: float = 0.10,
        max_rot_deg: float = 15.0,
        fg_tile_max: float = 0.35,
    ) -> None:
        self.downscale = max(1, int(downscale))
        self.tile = max(16, int(tile))
        self.overlap = float(min(max(overlap, 0.0), 0.9))
        self.pcr_thresh = float(pcr_thresh)
        self.min_tiles = max(3, int(min_tiles))
        self.max_disp_frac = float(max_disp_frac)
        self.max_scale_dev = float(max_scale_dev)
        self.max_rot_rad = float(max_rot_deg) * 3.14159265358979 / 180.0
        # Crowd awareness: tiles whose person-box coverage exceeds this fraction
        # measure people-flow, not camera motion, so they are excluded from the
        # fit. Only active when detection boxes are fed via set_fg_mask_boxes_*.
        self.fg_tile_max = float(fg_tile_max)
        self.prev_gray: Optional[torch.Tensor] = None
        self._last_pcr = 0.0
        self._origins: Optional[torch.Tensor] = None  # [N, 2] (oy, ox) tile origins
        self._centers: Optional[torch.Tensor] = None  # [N, 2] (cx, cy) tile centers
        self._hann2d: Optional[torch.Tensor] = None
        self._grid_hw: tuple[int, int] = (0, 0)
        self._fg_boxes: Optional[torch.Tensor] = None  # [M, 4] xyxy, original coords

    def _prepare_gray(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
            h = max(1, int(t.shape[-2] // self.downscale))
            w = max(1, int(t.shape[-1] // self.downscale))
            small = torch.nn.functional.interpolate(
                t, size=(h, w), mode="area"
            ).squeeze(0)
            gray = 0.299 * small[0] + 0.587 * small[1] + 0.114 * small[2]
        return gray.contiguous()

    def _ensure_grid(self, h: int, w: int, device: torch.device) -> None:
        if self._grid_hw == (h, w) and self._origins is not None:
            return
        t = min(self.tile, h, w)
        stride = max(1, int(t * (1.0 - self.overlap)))
        ys = list(range(0, max(1, h - t + 1), stride))
        xs = list(range(0, max(1, w - t + 1), stride))
        if not ys or ys[-1] != h - t:
            ys.append(max(0, h - t))
        if not xs or xs[-1] != w - t:
            xs.append(max(0, w - t))
        ys = sorted(set(ys))
        xs = sorted(set(xs))
        origins = [(oy, ox) for oy in ys for ox in xs]
        self._tile_sz = t
        self._origins = torch.tensor(origins, dtype=torch.int64, device=device)
        self._centers = torch.tensor(
            [[ox + t / 2.0, oy + t / 2.0] for (oy, ox) in origins],
            dtype=torch.float32,
            device=device,
        )
        hh = torch.hann_window(t, periodic=False, device=device, dtype=torch.float32)
        self._hann2d = (hh.unsqueeze(1) * hh.unsqueeze(0)).contiguous()
        self._grid_hw = (h, w)

    def set_fg_mask_boxes_tensor(self, boxes: torch.Tensor) -> None:
        """Feed this frame's detection boxes (xyxy, original coords) so crowd
        tiles can be excluded from the camera-motion fit. Consumed once."""
        self._fg_boxes = boxes

    def _fg_fraction(
        self, h: int, w: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Per-tile foreground (person-box) coverage fraction in [0, 1]."""
        if self._fg_boxes is None or len(self._fg_boxes) == 0:
            return None
        import math

        mask = torch.zeros(h, w, dtype=torch.float32, device=device)
        b = self._fg_boxes.to(device).float() / float(self.downscale)
        for box in b:
            x1 = int(max(0, math.floor(box[0].item())))
            y1 = int(max(0, math.floor(box[1].item())))
            x2 = int(min(w, math.ceil(box[2].item())))
            y2 = int(min(h, math.ceil(box[3].item())))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
        t = self._tile_sz
        assert self._origins is not None
        ar = torch.arange(t, device=device)
        rows = self._origins[:, 0].view(-1, 1) + ar.view(1, -1)
        cols = self._origins[:, 1].view(-1, 1) + ar.view(1, -1)
        tiles = mask[rows.unsqueeze(2), cols.unsqueeze(1)]  # [N, t, t]
        return tiles.mean(dim=(1, 2))

    def _phase_corr_batch(
        self, prev: torch.Tensor, curr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched per-tile phase correlation.

        Returns (disp [N,2] as (dx,dy) in downscaled px, pcr [N]).
        """
        t = self._tile_sz
        assert self._origins is not None and self._hann2d is not None
        oy = self._origins[:, 0]
        ox = self._origins[:, 1]
        # Gather tiles via unfold-free explicit slicing using advanced indexing.
        ar = torch.arange(t, device=prev.device)
        rows = oy.view(-1, 1) + ar.view(1, -1)  # [N, t]
        cols = ox.view(-1, 1) + ar.view(1, -1)  # [N, t]
        # [N, t, t]
        p = prev[rows.unsqueeze(2), cols.unsqueeze(1)]
        c = curr[rows.unsqueeze(2), cols.unsqueeze(1)]
        # DC removal + Hann window (range-agnostic).
        p = (p - p.mean(dim=(1, 2), keepdim=True)) * self._hann2d
        c = (c - c.mean(dim=(1, 2), keepdim=True)) * self._hann2d

        fp = torch.fft.rfft2(p)
        fc = torch.fft.rfft2(c)
        cross = fp.conj() * fc
        cross = cross / (cross.abs() + 1e-6)
        corr = torch.fft.irfft2(cross, s=(t, t))  # [N, t, t]

        n = corr.shape[0]
        flat = corr.view(n, -1)
        max_v, max_i = flat.max(dim=1)
        py = (max_i // t).float()
        px = (max_i % t).float()

        rms = torch.sqrt((flat * flat).mean(dim=1) + 1e-12)
        pcr = max_v / (rms + 1e-12)

        # 3x3 centroid sub-pixel refine (wrap-around).
        dyk = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=prev.device)
        dxk = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=prev.device)
        ny = (py.long().view(-1, 1) + dyk.view(1, -1) + t) % t  # [N,9]
        nx = (px.long().view(-1, 1) + dxk.view(1, -1) + t) % t
        nidx = ny * t + nx
        vv = torch.clamp(torch.gather(flat, 1, nidx), min=0.0)  # [N,9]
        sx = (vv * (px.view(-1, 1) + dxk.view(1, -1).float())).sum(dim=1)
        sy = (vv * (py.view(-1, 1) + dyk.view(1, -1).float())).sum(dim=1)
        sv = vv.sum(dim=1) + 1e-6
        peak_x = sx / sv
        peak_y = sy / sv
        # Wrap to [-t/2, t/2].
        dx = peak_x - torch.where(peak_x > t / 2.0, float(t), 0.0)
        dy = peak_y - torch.where(peak_y > t / 2.0, float(t), 0.0)
        disp = torch.stack([dx, dy], dim=1)
        return disp, pcr

    @staticmethod
    def _fit_similarity(
        pts: torch.Tensor, dst: torch.Tensor, w: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Weighted LS similarity fit. Returns [a, b, tx, ty] or None.

        Model: x' = a*x - b*y + tx ; y' = b*x + a*y + ty.
        """
        x = pts[:, 0]
        y = pts[:, 1]
        xp = dst[:, 0]
        yp = dst[:, 1]
        n = x.shape[0]
        # Two rows per correspondence.
        A = torch.zeros(2 * n, 4, dtype=torch.float32, device=pts.device)
        A[0::2, 0] = x
        A[0::2, 1] = -y
        A[0::2, 2] = 1.0
        A[1::2, 0] = y
        A[1::2, 1] = x
        A[1::2, 3] = 1.0
        bvec = torch.empty(2 * n, dtype=torch.float32, device=pts.device)
        bvec[0::2] = xp
        bvec[1::2] = yp
        wv = torch.repeat_interleave(w, 2)
        Aw = A * wv.unsqueeze(1)
        bw = bvec * wv
        ata = A.transpose(0, 1) @ Aw
        atb = A.transpose(0, 1) @ bw
        try:
            theta = torch.linalg.solve(ata, atb)
        except Exception:
            return None
        if not torch.isfinite(theta).all():
            return None
        return theta  # type: ignore[no-any-return]

    def estimate(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        curr = self._prepare_gray(frame_tensor)
        h, w = curr.shape[-2], curr.shape[-1]
        device = curr.device
        ds = float(self.downscale)

        out: Optional[torch.Tensor] = None
        if self.prev_gray is not None and self.prev_gray.shape == curr.shape:
            self._ensure_grid(h, w, device)
            with torch.no_grad():
                disp, pcr = self._phase_corr_batch(self.prev_gray, curr)

                # ── Global translation (whole-frame median of confident tiles) ──
                # This is the proven current-GMC behaviour and the fallback target.
                disp_bound = self.tile * self.max_disp_frac
                ok = (pcr >= self.pcr_thresh) & (disp.abs().amax(dim=1) <= disp_bound)
                global_warp: Optional[torch.Tensor] = None
                if int(ok.sum()) >= 1:
                    gd = disp[ok]
                    gdx = gd[:, 0].median().item() * ds
                    gdy = gd[:, 1].median().item() * ds
                    self._last_pcr = float(pcr[ok].median().item())
                    if abs(gdx) <= w * ds * 0.25 and abs(gdy) <= h * ds * 0.25:
                        global_warp = torch.tensor(
                            [[1.0, 0.0, gdx], [0.0, 1.0, gdy]],
                            dtype=torch.float32,
                            device=device,
                        )
                else:
                    self._last_pcr = 0.0

                # ── Crowd awareness ──────────────────────────────────────────
                # Tiles dominated by person boxes measure people-flow, which can
                # masquerade as camera rotation/scale. Exclude them from the
                # affine fit only (global translation keeps all confident tiles
                # so crowded scenes never lose the robust translation baseline).
                ok_aff = ok
                fg_frac = self._fg_fraction(h, w, device)
                if fg_frac is not None:
                    ok_aff = ok & (fg_frac <= self.fg_tile_max)
                self._fg_boxes = None  # consume once per frame

                # ── Similarity affine from confident, non-crowd tiles ──
                affine_warp = self._try_affine(disp, pcr, ok_aff, device, ds)

                # Affine if confident+sane, else global translation, else identity.
                out = affine_warp if affine_warp is not None else global_warp
        else:
            self._fg_boxes = None

        self.prev_gray = curr
        return out

    def _try_affine(
        self,
        disp: torch.Tensor,
        pcr: torch.Tensor,
        ok: torch.Tensor,
        device: torch.device,
        ds: float,
    ) -> Optional[torch.Tensor]:
        assert self._centers is not None
        if int(ok.sum()) < self.min_tiles:
            return None
        pts = self._centers[ok]  # prev tile centers (downscaled coords)
        dst = pts + disp[ok]  # mapped curr positions
        wts = (pcr[ok] - self.pcr_thresh).clamp(min=1e-3)  # PCR-above-gate weight

        theta = self._fit_similarity(pts, dst, wts)
        if theta is None:
            return None
        # One Huber reweight to reject foreground / outlier tiles.
        a, b, tx, ty = theta[0], theta[1], theta[2], theta[3]
        pred_x = a * pts[:, 0] - b * pts[:, 1] + tx
        pred_y = b * pts[:, 0] + a * pts[:, 1] + ty
        res = torch.sqrt((pred_x - dst[:, 0]) ** 2 + (pred_y - dst[:, 1]) ** 2 + 1e-9)
        med = res.median()
        mad = (res - med).abs().median() + 1e-6
        k = 1.345 * 1.4826 * mad
        huber = torch.where(res <= k, torch.ones_like(res), k / res)
        theta = self._fit_similarity(pts, dst, wts * huber)
        if theta is None:
            return None
        a_v, b_v, tx_v, ty_v = (
            theta[0].item(),
            theta[1].item(),
            theta[2].item(),
            theta[3].item(),
        )
        import math

        s = math.hypot(a_v, b_v)
        rot = math.atan2(b_v, a_v)
        # Upper plausibility gate → else fall back to global translation.
        # NOTE: a selective lower gate (apply affine only on "confident" frames,
        # translation otherwise) was tried and BACKFIRED — intermittent affine
        # creates a temporally inconsistent warp that hurts association more than
        # applying one model consistently (MOT17-13 IDF1 dropped below baseline,
        # MOT17-10 AssA −2.4). Keep it consistent: accept any plausible affine.
        if abs(s - 1.0) > self.max_scale_dev or abs(rot) > self.max_rot_rad:
            return None
        return torch.tensor(
            [[a_v, -b_v, tx_v * ds], [b_v, a_v, ty_v * ds]],
            dtype=torch.float32,
            device=device,
        )

    def apply(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self.estimate(frame_tensor)

    def pcr_score(self) -> float:
        return self._last_pcr

    def reset(self) -> None:
        self.prev_gray = None
        self._last_pcr = 0.0
