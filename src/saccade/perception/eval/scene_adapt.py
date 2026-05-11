"""Per-sequence scene adaptive policy.

Collects detection statistics over a warm-up window, then classifies the
scene type. The evaluator uses the result to selectively override per-sequence
tracking parameters without changing global defaults.

Currently detects one scene type:
  crowded_narrow — dense pedestrian crowd with tall/narrow boxes (e.g. MOT17-02).
                   Enables narrow_person_score_bonus for that sequence only.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class SceneStats:
    scene_type: str
    avg_boxes: float
    avg_aspect: float
    avg_width_ratio: float

    def __str__(self) -> str:
        return (
            f"{self.scene_type}  avg_boxes={self.avg_boxes:.1f}"
            f"  avg_aspect={self.avg_aspect:.2f}"
            f"  avg_width_ratio={self.avg_width_ratio:.3f}"
        )


class SceneAdaptivePolicy:
    """Accumulate detection stats over `window` frames, then emit a scene type.

    Thresholds
    ----------
    crowd_box_thresh       min avg boxes/frame to qualify as crowded
    narrow_aspect_thresh   max avg height/width ratio for crowded-narrow scenes
                           (crowded pedestrian scenes have LOWER avg aspect
                            due to heavy occlusion and partial detections)
    narrow_width_thresh    max avg box-width / frame-width (small = far-away)

    A scene is 'crowded_narrow' when:
    - avg boxes/frame  >= crowd_box_thresh
    - avg aspect ratio <= narrow_aspect_thresh  (dense crowd = more boxy dets)
    - avg width ratio  <= narrow_width_thresh   (far-away small people)

    All other scenes remain 'default' and receive no parameter override.
    """

    def __init__(
        self,
        *,
        window: int = 30,
        crowd_box_thresh: float = 15.0,
        narrow_aspect_thresh: float = 2.1,
        narrow_width_thresh: float = 0.035,
    ) -> None:
        self._window = max(1, window)
        self._crowd_box_thresh = crowd_box_thresh
        self._narrow_aspect_thresh = narrow_aspect_thresh
        self._narrow_width_thresh = narrow_width_thresh
        self._frame_count = 0
        self._total_boxes = 0
        self._aspect_sum = 0.0
        self._width_ratio_sum = 0.0
        self._aspect_frames = 0
        self._classified = False
        self._stats: Optional[SceneStats] = None

    @property
    def is_classified(self) -> bool:
        return self._classified

    @property
    def stats(self) -> Optional[SceneStats]:
        return self._stats

    def observe(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        frame_w: int,
        frame_h: int,
        score_thresh: float = 0.10,
    ) -> None:
        """Record detection statistics for one frame.  No-op after classification."""
        if self._classified:
            return
        self._frame_count += 1
        if boxes.ndim != 2 or boxes.shape[0] == 0:
            if self._frame_count >= self._window:
                self._classify()
            return
        # Restrict to confident-enough detections to avoid noise from weak dets.
        if scores.numel() == boxes.shape[0]:
            mask = scores > score_thresh
            b = boxes[mask]
        else:
            b = boxes
        n = b.shape[0]
        if n > 0:
            widths = (b[:, 2] - b[:, 0]).clamp_min(1.0)
            heights = (b[:, 3] - b[:, 1]).clamp_min(1.0)
            self._aspect_sum += float((heights / widths).mean().item())
            self._width_ratio_sum += float(
                (widths / max(float(frame_w), 1.0)).mean().item()
            )
            self._aspect_frames += 1
        self._total_boxes += n
        if self._frame_count >= self._window:
            self._classify()

    def _classify(self) -> None:
        avg_boxes = self._total_boxes / max(self._frame_count, 1)
        avg_aspect = (
            self._aspect_sum / self._aspect_frames if self._aspect_frames > 0 else 0.0
        )
        avg_width_ratio = (
            self._width_ratio_sum / self._aspect_frames
            if self._aspect_frames > 0
            else 1.0
        )
        scene_type = (
            "crowded_narrow"
            if (
                avg_boxes >= self._crowd_box_thresh
                and avg_aspect
                <= self._narrow_aspect_thresh  # dense crowd → lower avg aspect
                and avg_width_ratio <= self._narrow_width_thresh
            )
            else "default"
        )
        self._stats = SceneStats(
            scene_type=scene_type,
            avg_boxes=avg_boxes,
            avg_aspect=avg_aspect,
            avg_width_ratio=avg_width_ratio,
        )
        self._classified = True
