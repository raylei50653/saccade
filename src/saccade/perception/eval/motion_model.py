"""Per-track motion model for improved gap closure and motion-based relinking.

Design
------
Each track maintains an EMA-averaged velocity (Δcx, Δcy) and an optional
acceleration signal.  At each observed frame we:

  1. Update velocity/acceleration from the current box centre
  2. Project the *next* box centre forward by 1 frame

This gives us a cheap but effective motion predictor that works even when
the C++ Kalman filter is unavailable (e.g. in pure-Python eval or when the
C++ tracker does not expose snapshots).

Key functions
-------------
* `update()`         — consume a new box observation
* `predict(offset)`  — project the centre forward by *offset* frames
* `motion_iou()`     — IoU between a detection box and the projected box
* `velocity_consistent()` — check if a new detection's motion direction is
                            compatible with the track's velocity history
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from saccade.perception.box_ops import box_iou


@dataclass(frozen=False)
class MotionModel:
    """Per-track motion model with EMA velocity + optional acceleration."""

    # --- EMA parameters ---
    vel_alpha: float = 0.3  # velocity EMA smoothing factor
    acc_alpha: float = 0.15  # acceleration EMA smoothing factor
    min_observations: int = 2  # minimum observations before trusting motion

    # --- Motion state (updated in-place) ---
    cx: float = 0.0  # last observed centre x
    cy: float = 0.0  # last observed centre y
    bw: float = 1.0  # last observed box width
    bh: float = 1.0  # last observed box height
    vel_x: float = 0.0  # EMA-averaged Δcx per frame
    vel_y: float = 0.0  # EMA-averaged Δcy per frame
    acc_x: float = 0.0  # EMA-averaged acceleration in x
    acc_y: float = 0.0  # EMA-averaged acceleration in y
    obs_count: int = 0  # total observations
    last_frame: int = -999  # frame of last observation (for gap detection)

    def update(self, cx: float, cy: float, bw: float, bh: float, frame: int) -> None:
        """Consume a new box observation and update velocity/acceleration."""
        if self.obs_count > 0 and frame > self.last_frame:
            # Compute instantaneous velocity
            dt = max(1, frame - self.last_frame)
            inst_vx = (cx - self.cx) / dt
            inst_vy = (cy - self.cy) / dt

            # Update acceleration
            new_acc_x = (inst_vx - self.vel_x) / dt
            new_acc_y = (inst_vy - self.vel_y) / dt
            if self.obs_count >= 2:
                self.acc_x = (
                    1 - self.acc_alpha
                ) * self.acc_x + self.acc_alpha * new_acc_x
                self.acc_y = (
                    1 - self.acc_alpha
                ) * self.acc_y + self.acc_alpha * new_acc_y
            else:
                self.acc_x = new_acc_x
                self.acc_y = new_acc_y

            # Update velocity
            self.vel_x = (1 - self.vel_alpha) * self.vel_x + self.vel_alpha * inst_vx
            self.vel_y = (1 - self.vel_alpha) * self.vel_y + self.vel_alpha * inst_vy
        else:
            # First observation or gap too large — reset acceleration
            if self.obs_count == 0:
                self.acc_x = 0.0
                self.acc_y = 0.0

        self.cx = cx
        self.cy = cy
        self.bw = bw
        self.bh = bh
        self.obs_count += 1
        self.last_frame = frame

    def predict(self, offset: int = 1) -> tuple[float, float, float, float]:
        """Project the next box (cx, cy, bw, bh) forward by *offset* frames.

        Returns (pred_cx, pred_cy, pred_bw, pred_bh).
        If the model has not seen enough observations, returns (None, None, bw, bh).
        """
        if self.obs_count < self.min_observations:
            return (self.cx, self.cy, self.bw, self.bh)

        if offset <= 0:
            return (self.cx, self.cy, self.bw, self.bh)

        # Predicted centre = last_centre + velocity * offset + 0.5 * acceleration * offset^2
        pred_cx = self.cx + self.vel_x * offset + 0.5 * self.acc_x * offset * offset
        pred_cy = self.cy + self.vel_y * offset + 0.5 * self.acc_y * offset * offset

        # Predicted size stays constant (simple model)
        return (pred_cx, pred_cy, self.bw, self.bh)

    def motion_box(self, offset: int = 1) -> Optional[torch.Tensor]:
        """Return a predicted box tensor [x1, y1, x2, y2] or None."""
        pcx, pcy, pbw, pbh = self.predict(offset)
        if self.obs_count < self.min_observations:
            return None
        return torch.tensor(
            [pcx - 0.5 * pbw, pcy - 0.5 * pbh, pcx + 0.5 * pbw, pcy + 0.5 * pbh],
            dtype=torch.float32,
        )

    @staticmethod
    def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        return box_iou(box_a, box_b, union_mode="zero")

    def motion_iou(self, box: torch.Tensor, offset: int = 1) -> float:
        """IoU between a detection box and the predicted box."""
        pred_box = self.motion_box(offset)
        if pred_box is None:
            return 0.0
        return self.compute_iou(box, pred_box)

    def velocity_consistent(
        self, box: torch.Tensor, frame_id: int, tol_factor: float = 2.0
    ) -> bool:
        """Check if a new detection's displacement is consistent with track velocity.

        A detection is consistent if its displacement from the last observed centre
        lies within `tol_factor *` standard deviation of the velocity distribution.

        Returns True if consistent or if model is not ready.
        """
        if self.obs_count < self.min_observations:
            return True  # trust motion once we have enough history

        dt = max(1, frame_id - self.last_frame)
        if dt <= 0:
            return True

        # Observed displacement
        cx_obs = (float(box[0]) + float(box[2])) * 0.5
        cy_obs = (float(box[1]) + float(box[3])) * 0.5
        dx_obs = (cx_obs - self.cx) / dt
        dy_obs = (cy_obs - self.cy) / dt

        # Expected velocity (with heuristic std approximation).
        # We approximate velocity uncertainty as |acceleration| × sqrt(dt) / sqrt(obs_count).
        # This is a heuristic: |acc| is the magnitude, not a true standard deviation,
        # and the forced minimum of 1.0 provides a floor when acceleration is near zero.
        # Units: velocity (px/frame). More observations reduce uncertainty as 1/sqrt(n).
        std_x = max(
            1.0, abs(self.acc_x) * math.sqrt(dt) / max(1, math.sqrt(self.obs_count))
        )
        std_y = max(
            1.0, abs(self.acc_y) * math.sqrt(dt) / max(1, math.sqrt(self.obs_count))
        )

        # Z-score check
        z_x = abs(dx_obs - self.vel_x) / std_x
        z_y = abs(dy_obs - self.vel_y) / std_y
        return (z_x < tol_factor) and (z_y < tol_factor)

    def reset(self) -> None:
        """Reset the model for a new track."""
        self.cx = 0.0
        self.cy = 0.0
        self.bw = 1.0
        self.bh = 1.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.obs_count = 0
        self.last_frame = -999

    def copy(self) -> "MotionModel":
        """Create a shallow copy."""
        obj = MotionModel()
        obj.cx = self.cx
        obj.cy = self.cy
        obj.bw = self.bw
        obj.bh = self.bh
        obj.vel_x = self.vel_x
        obj.vel_y = self.vel_y
        obj.acc_x = self.acc_x
        obj.acc_y = self.acc_y
        obj.obs_count = self.obs_count
        obj.last_frame = self.last_frame
        return obj


class MotionModelRegistry:
    """Registry that maintains a MotionModel per track ID.

    Usage
    -----
    registry = MotionModelRegistry()
    registry.update(track_id, cx, cy, bw, bh, frame_id)
    pred_box = registry.predict(track_id, offset=2)
    registry.prune(active_ids)
    """

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._models: dict[int, MotionModel] = {}
        self._params = kwargs

    def get_or_create(self, track_id: int) -> MotionModel:
        if track_id not in self._models:
            self._models[track_id] = MotionModel(**self._params)
        return self._models[track_id]

    def update(
        self, track_id: int, cx: float, cy: float, bw: float, bh: float, frame: int
    ) -> None:
        model = self.get_or_create(track_id)
        model.update(cx, cy, bw, bh, frame)

    def predict(self, track_id: int, offset: int = 1) -> Optional[torch.Tensor]:
        model = self._models.get(track_id)
        if model is None:
            return None
        return model.motion_box(offset)

    def motion_iou(self, track_id: int, box: torch.Tensor, offset: int = 1) -> float:
        model = self._models.get(track_id)
        if model is None:
            return 0.0
        return model.motion_iou(box, offset)

    def velocity_consistent(
        self, track_id: int, box: torch.Tensor, frame_id: int, tol_factor: float = 2.0
    ) -> bool:
        model = self._models.get(track_id)
        if model is None:
            return True
        return model.velocity_consistent(box, frame_id, tol_factor)

    def prune(self, active_ids: set[int]) -> None:
        """Remove models for tracks that are no longer active."""
        for tid in [t for t in self._models if t not in active_ids]:
            del self._models[tid]

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._models

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[int]:
        return iter(self._models)
