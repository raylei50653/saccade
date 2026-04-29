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
