"""
TemporalFeatureFusion — Option E-v2: Quality-Gated Temporal Feature Fusion.

Core formula:
    P_t_fused = P_t + α_tier × Q_spatial × warp(P_{t-1}.detach(), GMC)

Where:
    P_t      = current FPN feature (post-gate from Option E)
    α_tier   = per-track-state fusion strength (occluded/recent/stable/tentative)
    Q_spatial= per-pixel quality heatmap (tracker Gaussian + detector score)
    P_{t-1}  = previous frame raw FPN feature, detached (no BPTT)
    warp()   = perspective warp via F.grid_sample using GMC affine matrix

Phases:
    0 → α=0, identical to gated_det_v1 baseline
    1 → α fixed, no warp, Q = tracker heatmap only
    2 → GMC warp enabled (NO-GO)
    3 → α_tier per-track-state
    4 → Lock-in detection + age decay
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# GMC warp utilities
# ---------------------------------------------------------------------------
def _build_affine_grid(
    feat: Tensor,
    gmc_matrix: Tensor,  # 2×3 affine [A|t] in image pixel coordinates
    img_h: int,
    img_w: int,
) -> Tensor:
    B, C, H, W = feat.shape
    device = feat.device
    dtype = feat.dtype

    stride_y = img_h / H
    stride_x = img_w / W

    inverse = _invert_affine(gmc_matrix)

    gy_px = (torch.arange(H, device=device, dtype=dtype) + 0.5) * stride_y
    gx_px = (torch.arange(W, device=device, dtype=dtype) + 0.5) * stride_x

    gx_grid, gy_grid = torch.meshgrid(gx_px, gy_px, indexing="xy")

    coords = torch.stack([gx_grid, gy_grid, torch.ones_like(gx_grid)], dim=0)
    coords_flat = coords.reshape(3, -1)

    warped_px = inverse @ coords_flat
    warped_px = warped_px.reshape(2, H, W)

    cell_x = warped_px[0] / stride_x
    cell_y = warped_px[1] / stride_y
    grid_x = 2.0 * cell_x / (W - 1) - 1.0
    grid_y = 2.0 * cell_y / (H - 1) - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    return grid


def _invert_affine(affine: Tensor) -> Tensor:
    A = affine[:, :2]
    t = affine[:, 2:3]

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-8:
        return torch.eye(2, 3, device=affine.device, dtype=affine.dtype)

    inv_A = torch.stack(
        [
            torch.stack([A[1, 1], -A[0, 1]]) / det,
            torch.stack([-A[1, 0], A[0, 0]]) / det,
        ]
    )
    inv_t = -inv_A @ t

    return torch.cat([inv_A, inv_t], dim=-1)


# ---------------------------------------------------------------------------
# α_tier configuration (Phase 3)
# ---------------------------------------------------------------------------
@dataclass
class AlphaTierConfig:
    occluded: float = 0.20  # det_idx == -1, pure Kalman prediction
    confirmed_recent: float = 0.15  # age 1-10, recently confirmed
    confirmed_stable: float = 0.05  # age >10, lock-in prevention
    tentative: float = 0.05  # unconfirmed tracks

    age_decay_window: int = 100  # Q' = Q * max(0, 1 - age / window)
    recent_age_threshold: int = 10  # age <= this is "recent"


def _compute_alpha_tier(
    confirmed_ages: Tensor | None,
    confirmed_occluded: Tensor | None,
    tentative_count: int = 0,
    tier_cfg: AlphaTierConfig | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    if tier_cfg is None:
        tier_cfg = AlphaTierConfig()

    confirmed_alphas = None
    tentative_alpha = None

    if confirmed_ages is not None and confirmed_occluded is not None:
        N = confirmed_ages.shape[0]
        if N > 0:
            ages = confirmed_ages.float()
            occluded = confirmed_occluded.float()

            recent = (ages >= 1).float() * (
                ages <= tier_cfg.recent_age_threshold
            ).float()
            stable = (ages > tier_cfg.recent_age_threshold).float()

            alphas = (
                occluded * tier_cfg.occluded
                + recent * (1.0 - occluded) * tier_cfg.confirmed_recent
                + stable * (1.0 - occluded) * tier_cfg.confirmed_stable
            )

            age_penalty = (1.0 - ages / tier_cfg.age_decay_window).clamp(min=0.0)
            alphas = alphas * age_penalty

            confirmed_alphas = alphas

    if tentative_count > 0:
        tentative_alpha = torch.tensor(tier_cfg.tentative)

    return confirmed_alphas, tentative_alpha


# ---------------------------------------------------------------------------
# TemporalFeatureFusion
# ---------------------------------------------------------------------------
class TemporalFeatureFusion(nn.Module):
    def __init__(
        self,
        scales: tuple[str, ...],
        img_size: int = 640,
        tier_cfg: AlphaTierConfig | None = None,
    ):
        super().__init__()
        self.scales = list(scales)
        self.img_size = img_size
        self.tier_cfg = tier_cfg or AlphaTierConfig()

        self.fusion_alphas = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(1)) for s in scales}
        )

        self.prev_feats: dict[str, Tensor] = {}
        self.prev_q_spatial: dict[str, Tensor] = {}

        self.fixed_alpha: float | None = None
        self._gmc_matrix: Tensor | None = None

    def set_gmc(self, matrix: Tensor | None) -> None:
        self._gmc_matrix = matrix

    def set_fixed_alpha(self, alpha: float | None) -> None:
        self.fixed_alpha = alpha

    def update_prev(self, feats: dict[str, Tensor]) -> None:
        self.prev_feats = {s: f.detach() for s, f in feats.items()}

    def _get_alpha(self, scale: str) -> Tensor:
        if self.fixed_alpha is not None:
            return torch.tensor(
                self.fixed_alpha, device=self.fusion_alphas[scale].device
            )
        return self.fusion_alphas[scale]

    def fuse(
        self,
        scale: str,
        feat: Tensor,
        q_spatial: Tensor,
    ) -> Tensor:
        if scale not in self.prev_feats or self.prev_feats[scale] is None:
            return feat

        prev = self.prev_feats[scale].to(device=feat.device, dtype=feat.dtype)

        if prev.shape[2:] != feat.shape[2:]:
            prev = F.interpolate(
                prev, size=feat.shape[2:], mode="bilinear", align_corners=False
            )

        if self._gmc_matrix is not None:
            gmc = self._gmc_matrix.to(device=feat.device, dtype=feat.dtype)
            grid = _build_affine_grid(prev, gmc, self.img_size, self.img_size)
            prev = F.grid_sample(
                prev, grid, mode="bilinear", padding_mode="border", align_corners=False
            )

        alpha = self._get_alpha(scale)
        if alpha == 0.0:
            return feat

        fusion = alpha * q_spatial * prev
        return feat + fusion

    def reset(self) -> None:
        self.prev_feats.clear()
        self.prev_q_spatial.clear()
        self._gmc_matrix = None

    def alpha_summary(self) -> str:
        parts = []
        for s in self.scales:
            a = self._get_alpha(s)
            parts.append(f"{s}={float(a):.4f}")
        return "  ".join(parts)
