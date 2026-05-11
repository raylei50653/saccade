"""Biometric ratio computation and per-track accumulation.

Output format from detect_batch(): keypoints [N, 17, 3] — (x, y, visibility).

COCO-17 index map used here:
  5/6  = L/R shoulder   11/12 = L/R hip   15/16 = L/R ankle
  0    = nose

All ratios are dimensionless (length / length), view-angle-invariant when
both segments share the same orientation (vertical for R_leg, horizontal-
normalised for R_shoulder once divided by torso height).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch


# ──────────────────────────────────────────────────────────────────────────────
# COCO-17 keypoint indices
# ──────────────────────────────────────────────────────────────────────────────
_SHOULDER = [5, 6]
_HIP = [11, 12]
_ANKLE = [15, 16]
_NOSE = [0]

_VIS_THR = 0.4  # minimum per-keypoint visibility to trust the point
_MIN_LEN = 1e-3  # minimum segment length to avoid division by zero


def _mid(
    kpts: torch.Tensor, idx: list[int], vis_thr: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return midpoint [N, 2] and boolean valid mask [N] for a keypoint group."""
    pts = kpts[:, idx, :2]  # [N, len(idx), 2]
    vis = kpts[:, idx, 2]  # [N, len(idx)]
    valid = (vis > vis_thr).all(dim=1)  # [N]
    mid = pts.mean(dim=1)  # [N, 2]
    return mid, valid


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame ratio computation  (all → [N], nan where invisible)
# ──────────────────────────────────────────────────────────────────────────────


def leg_body_ratio(kpts: torch.Tensor, vis_thr: float = _VIS_THR) -> torch.Tensor:
    """R_leg = d(hip, ankle) / d(shoulder, hip).  Both segments are vertical → scale-invariant."""
    sh, sh_ok = _mid(kpts, _SHOULDER, vis_thr)
    hi, hi_ok = _mid(kpts, _HIP, vis_thr)
    an, an_ok = _mid(kpts, _ANKLE, vis_thr)
    ok = sh_ok & hi_ok & an_ok
    leg = (hi - an).norm(dim=1)
    torso = (sh - hi).norm(dim=1)
    nan = torch.full_like(leg, float("nan"))
    return torch.where(ok & (torso > _MIN_LEN), leg / torso, nan)


def shoulder_width_ratio(kpts: torch.Tensor, vis_thr: float = _VIS_THR) -> torch.Tensor:
    """R_shoulder = d(L_shoulder, R_shoulder) / d(shoulder_mid, hip_mid).

    Uses torso height as denominator (not full body height) to avoid ankle occlusion.
    Note: sensitive to body rotation; use only when front-facing frames dominate.
    """
    sh, sh_ok = _mid(kpts, _SHOULDER, vis_thr)
    hi, hi_ok = _mid(kpts, _HIP, vis_thr)
    pts_sh = kpts[:, _SHOULDER, :2]  # [N, 2, 2]
    vis_sh = kpts[:, _SHOULDER, 2]  # [N, 2]
    both_sh = (vis_sh > vis_thr).all(dim=1)
    ok = sh_ok & hi_ok & both_sh
    shoulder_w = (pts_sh[:, 0] - pts_sh[:, 1]).norm(dim=1)
    torso = (sh - hi).norm(dim=1)
    nan = torch.full_like(shoulder_w, float("nan"))
    return torch.where(ok & (torso > _MIN_LEN), shoulder_w / torso, nan)


def head_body_ratio(kpts: torch.Tensor, vis_thr: float = _VIS_THR) -> torch.Tensor:
    """R_head = d(shoulder_mid, hip_mid) / d(nose, shoulder_mid).

    Inverted from doc formula so > 1 always; neck approximated as shoulder midpoint.
    """
    sh, sh_ok = _mid(kpts, _SHOULDER, vis_thr)
    hi, hi_ok = _mid(kpts, _HIP, vis_thr)
    no, no_ok = _mid(kpts, _NOSE, vis_thr)
    ok = sh_ok & hi_ok & no_ok
    torso = (sh - hi).norm(dim=1)
    head_seg = (sh - no).norm(dim=1)
    nan = torch.full_like(torso, float("nan"))
    return torch.where(ok & (head_seg > _MIN_LEN), torso / head_seg, nan)


# ──────────────────────────────────────────────────────────────────────────────
# Per-track temporal accumulator
# ──────────────────────────────────────────────────────────────────────────────


class BiometricAccumulator:
    """Collect per-track ratio samples and emit a stable median fingerprint.

    Usage::

        acc = BiometricAccumulator(window=15, vis_thr=0.4)

        # Called every frame from on_track_result callback:
        acc.update(track_ids, tracked_kpts)  # tracked_kpts: [M, 17, 3]

        # Query at any time:
        ratios = acc.get(track_id)  # dict or None if not enough frames
    """

    def __init__(self, window: int = 15, vis_thr: float = _VIS_THR) -> None:
        self.window = window
        self.vis_thr = vis_thr
        # track_id → list of (r_leg, r_shoulder, r_head) floats (nan = missing)
        self._buf: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    def update(
        self,
        track_ids: torch.Tensor,  # [M] int
        keypoints: Optional[torch.Tensor],  # [M, 17, 3] or None
    ) -> None:
        if keypoints is None or keypoints.numel() == 0:
            return
        r_leg = leg_body_ratio(keypoints, self.vis_thr)
        r_sh = shoulder_width_ratio(keypoints, self.vis_thr)
        r_hd = head_body_ratio(keypoints, self.vis_thr)
        for i, tid in enumerate(track_ids.tolist()):
            tid = int(tid)
            buf = self._buf[tid]
            buf.append((float(r_leg[i]), float(r_sh[i]), float(r_hd[i])))
            if len(buf) > self.window:
                buf.pop(0)

    def get(self, track_id: int) -> Optional[dict[str, float]]:
        """Return median biometric ratios, or None if insufficient frames accumulated."""
        buf = self._buf.get(track_id)
        min_frames = max(2, self.window // 2)
        if not buf or len(buf) < min_frames:
            return None
        t = torch.tensor(buf, dtype=torch.float32)  # [K, 3]
        out: dict[str, float] = {}
        for col, name in enumerate(("r_leg", "r_shoulder", "r_head")):
            vals = t[:, col]
            valid = vals[~torch.isnan(vals)]
            if valid.numel() >= 2:  # at least 2 non-nan observations
                out[name] = float(valid.median())
        return out if out else None

    def evict(self, track_id: int) -> None:
        self._buf.pop(track_id, None)

    def get_biometric(self, track_id: int) -> Optional[dict[str, float]]:
        return self.get(track_id)

    def evict_stale(self, active_ids: set[int]) -> None:
        for tid in list(self._buf.keys()):
            if tid not in active_ids:
                self._buf.pop(tid)
