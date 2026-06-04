"""
Convolutional ReID head operating on raw FPN feature maps (P3/P4/P5).

Does NOT share parameters with MambaDetectionHead.emb_head.
Designed for JDE-style per-pixel embedding extraction.

Architecture (per FPN scale):
    Conv2d(in_c, mid_c, 3×3) → BN → ReLU → Conv2d(mid_c, emb_dim, 1×1)

Usage:
    head = ReIDConvHead(in_channels=[128, 256, 512], emb_dim=128)
    emb_maps = head(fpn_feats)          # list of (N, emb_dim, H, W)
    pooled = head.pool_center(emb_maps)  # (N, emb_dim * 3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDConvHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        emb_dim: int = 128,
        mid_factor: int = 2,
    ):
        super().__init__()
        self.nl = len(in_channels)
        self.emb_dim = emb_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c // mid_factor, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c // mid_factor),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c // mid_factor, emb_dim, 1, bias=False),
                )
                for c in in_channels
            ]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [h(f) for h, f in zip(self.heads, feats)]

    @staticmethod
    def pool_center(emb_preds: list[torch.Tensor]) -> torch.Tensor:
        """Extract center pixel from each scale, concat, L2-normalize."""
        parts = []
        for emb in emb_preds:
            h, w = emb.shape[2], emb.shape[3]
            center = emb[:, :, h // 2, w // 2]
            parts.append(center)
        return F.normalize(torch.cat(parts, dim=1), dim=1)
