"""Multi-signal birth policy (P5-1) for sub-threshold detection recovery.

Implements a joint evidence model over score × streak × motion × geometry.

Unlike P5-3 (consecutive birth gate) which applies sequential binary gates
(score AND streak AND IoU-match AND motion), this uses a weighted sum so that
strong evidence in one dimension can compensate for weaker evidence in another.

Primary use case: recover narrow-person sub-threshold true positives that
appear across multiple frames with consistent motion.

Usage:
    manager = MultiSignalBirthManager(new_track_thresh=0.35, ...)
    manager.reset()  # call once per sequence
    # per frame:
    promote_mask = manager.update(frame_id, sub_boxes, sub_scores)
    # fused_scores[promote_mask] = new_track_thresh + 0.01  # done by caller
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class _Candidate:
    history: list[tuple[int, torch.Tensor, float]]  # (frame_id, box_cpu, score)
    last_frame: int


def _box_iou_nm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between every (a_i, b_j) pair.  a: (N,4)  b: (M,4) → (N,M)."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = torch.maximum(ax1.unsqueeze(1), bx1.unsqueeze(0))
    iy1 = torch.maximum(ay1.unsqueeze(1), by1.unsqueeze(0))
    ix2 = torch.minimum(ax2.unsqueeze(1), bx2.unsqueeze(0))
    iy2 = torch.minimum(ay2.unsqueeze(1), by2.unsqueeze(0))
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    aa = ((ax2 - ax1) * (ay2 - ay1)).unsqueeze(1)
    ba = ((bx2 - bx1) * (by2 - by1)).unsqueeze(0)
    return inter / (aa + ba - inter).clamp(min=1e-6)


class MultiSignalBirthManager:
    """Joint multi-signal birth evidence model.

    Evidence formula (each term in [0, 1]):
        E = w_score * score_norm
          + w_motion * motion_norm
          + w_quality * geometry_quality
          + w_streak * streak_norm

    When E >= evidence_threshold and n_frames >= min_frames, the candidate
    detection is eligible to have its score promoted to new_track_thresh + ε.

    The caller is responsible for applying the score boost in fused_scores.
    """

    def __init__(
        self,
        new_track_thresh: float = 0.35,
        min_score: float = 0.12,
        min_frames: int = 3,
        target_motion_px: float = 12.0,
        evidence_threshold: float = 0.60,
        iou_match: float = 0.30,
        ttl_frames: int = 5,
        w_score: float = 0.35,
        w_motion: float = 0.30,
        w_quality: float = 0.20,
        w_streak: float = 0.15,
        min_aspect: float = 0.0,
        max_area_px: int = 0,
    ) -> None:
        self.new_track_thresh = new_track_thresh
        self.min_score = min_score
        self.min_frames = min_frames
        self.target_motion_px = max(target_motion_px, 1.0)
        self.evidence_threshold = evidence_threshold
        self.iou_match = iou_match
        self.ttl_frames = ttl_frames
        self.w_score = w_score
        self.w_motion = w_motion
        self.w_quality = w_quality
        self.w_streak = w_streak
        self.min_aspect = min_aspect
        self.max_area_px = max_area_px
        self._candidates: list[_Candidate] = []
        self._promoted_ids: set[int] = (
            set()
        )  # indices into _candidates already promoted

    def reset(self) -> None:
        self._candidates = []
        self._promoted_ids = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _geometry_score(self, box: torch.Tensor) -> float:
        """Return geometry quality in [0,1], or -1 to hard-reject the box."""
        x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        aspect = h / w  # > 1 means taller than wide (person-like)
        area = w * h

        if self.min_aspect > 0.0 and aspect < self.min_aspect:
            return -1.0
        if self.max_area_px > 0 and area > self.max_area_px:
            return -1.0

        # Quality peaks at aspect 2-4 (human silhouette range)
        if aspect < 1.0:
            q = 0.0
        elif aspect < 2.0:
            q = aspect - 1.0  # 0 → 1 as aspect goes 1 → 2
        elif aspect <= 4.0:
            q = 1.0  # ideal
        else:
            q = max(0.0, 1.0 - (aspect - 4.0) / 3.0)  # 1 → 0 as aspect goes 4 → 7
        return q

    def _compute_evidence(self, cand: _Candidate) -> float:
        """Joint evidence score in [0, 1].  Returns 0 if min_frames not reached."""
        n = len(cand.history)
        if n < self.min_frames:
            return 0.0

        # Streak: normalized over [0, min_frames-1]
        streak = min(1.0, (n - 1) / max(self.min_frames - 1, 1))

        # Score: best observed score relative to [min_score, new_track_thresh)
        best_score = max(h[2] for h in cand.history)
        score_range = max(self.new_track_thresh - self.min_score, 0.01)
        score_norm = min(1.0, max(0.0, (best_score - self.min_score) / score_range))

        # Geometry of latest observation
        latest_box = cand.history[-1][1]
        gq = self._geometry_score(latest_box)
        if gq < 0.0:
            return 0.0  # hard reject

        # Motion: mean per-frame centroid displacement across history
        if n >= 2:
            total = 0.0
            for i in range(1, n):
                b0, b1 = cand.history[i - 1][1], cand.history[i][1]
                cx0 = float((b0[0] + b0[2]) / 2)
                cy0 = float((b0[1] + b0[3]) / 2)
                cx1 = float((b1[0] + b1[2]) / 2)
                cy1 = float((b1[1] + b1[3]) / 2)
                total += ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
            motion_norm = min(1.0, (total / (n - 1)) / self.target_motion_px)
        else:
            motion_norm = 0.0

        return (
            self.w_score * score_norm
            + self.w_motion * motion_norm
            + self.w_quality * gq
            + self.w_streak * streak
        )

    def _expire(self, frame_id: int) -> None:
        self._candidates = [
            c for c in self._candidates if frame_id - c.last_frame <= self.ttl_frames
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_id: int,
        sub_boxes: torch.Tensor,  # (K, 4) detections below new_track_thresh
        sub_scores: torch.Tensor,  # (K,)
    ) -> torch.Tensor:
        """Update candidates and return bool promote-mask over sub_boxes.

        A True entry means the corresponding detection has accumulated enough
        joint evidence to be promoted to new_track_thresh + ε for track birth.
        Promoted candidates are removed from the buffer immediately so they
        cannot be boosted again next frame.

        sub_boxes / sub_scores must already be filtered to
        [min_score, new_track_thresh) by the caller.
        """
        n = sub_boxes.shape[0]
        promote_mask = torch.zeros(n, dtype=torch.bool, device=sub_boxes.device)

        if n == 0:
            self._expire(frame_id)
            return promote_mask

        # IoU-match incoming boxes to existing candidates
        if self._candidates:
            cand_last_boxes = torch.stack(
                [c.history[-1][1].to(sub_boxes.device) for c in self._candidates]
            )
            iou = _box_iou_nm(sub_boxes, cand_last_boxes)  # (K, C)
            max_iou, best_cand_idx = iou.max(dim=1)
        else:
            max_iou = torch.zeros(n, device=sub_boxes.device)
            best_cand_idx = torch.full(
                (n,), -1, dtype=torch.long, device=sub_boxes.device
            )

        matched_cand_set: set[int] = set()
        to_remove: list[int] = []

        for i in range(n):
            score = float(sub_scores[i].item())
            if score < self.min_score:
                continue

            ci = int(best_cand_idx[i].item())
            matched = max_iou[i].item() >= self.iou_match and ci not in matched_cand_set

            if matched:
                matched_cand_set.add(ci)
                cand = self._candidates[ci]
                cand.history.append((frame_id, sub_boxes[i].cpu().clone(), score))
                cand.last_frame = frame_id

                ev = self._compute_evidence(cand)
                if ev >= self.evidence_threshold:
                    promote_mask[i] = True
                    to_remove.append(ci)  # remove so it doesn't get promoted again
            else:
                # Start tracking as new candidate
                self._candidates.append(
                    _Candidate(
                        history=[(frame_id, sub_boxes[i].cpu().clone(), score)],
                        last_frame=frame_id,
                    )
                )

        # Remove promoted candidates (by index descending to preserve indices)
        for ci in sorted(set(to_remove), reverse=True):
            self._candidates.pop(ci)

        self._expire(frame_id)
        return promote_mask
