"""
Zero-training FPN-based ReID feature extractor.

Extracts per-detection embeddings by center-pooling raw YOLO FPN features
(P3/P4/P5). No extra head, no projector, no training required.

The features come from the YOLO backbone's internal FPN layers, which encode
appearance information learned during detection training.

Usage:
    extractor = FPNReIDExtractor(teacher_model, yolo_model, img_size=640)
    embeddings = extractor.extract(frame_hwc, boxes_xyxy)
    # embeddings: (N, total_dim) L2-normalized float32 on CUDA
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DimReduceHead(nn.Module):
    """Per-scale 1×1 Conv → center-pool → concat → optional projector → L2."""

    def __init__(self, in_channels: list[int], out_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.nl = len(in_channels)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_dim, 1, bias=False) for c in in_channels]
        )
        nn.init.normal_(self.convs[-1].weight, std=0.001)

        mid_dim = out_dim * self.nl
        if mid_dim != out_dim:
            self.proj = nn.Sequential(
                nn.Linear(mid_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        parts = []
        for conv, f in zip(self.convs, feats):
            x = conv(f)
            h, w = x.shape[2], x.shape[3]
            center = x[:, :, h // 2, w // 2]
            parts.append(center)
        pooled = torch.cat(parts, dim=1)
        return F.normalize(self.proj(pooled), dim=1)


class FPNReIDExtractor:
    """Zero-training ReID from YOLO FPN center-pixel features."""

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        yolo_model: torch.nn.Module,
        img_size: int = 640,
        device: torch.device | None = None,
    ):
        self.teacher = teacher_model
        self.yolo_model = yolo_model
        self.img_size = img_size
        self.device = device or next(teacher_model.parameters()).device

        # Probe dimensions
        dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
        feats = self._get_fpn_feats(dummy)
        self.fpn_dims = [f.shape[1] for f in feats]
        self.total_dim = sum(self.fpn_dims)

    def _get_fpn_feats(self, frame_bchw: torch.Tensor) -> list[torch.Tensor]:
        """Extract P3/P4/P5 FPN features from the YOLO model."""
        from saccade.perception.temporal_yolo.yolo_gated_detector import (
            _GATE_LAYER_IDX,
        )

        layers = self.yolo_model.model
        save: set[int] = set(self.yolo_model.save)
        y: list[torch.Tensor | None] = []
        x = frame_bchw
        for i in range(23):
            m = layers[i]
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if i in save else None)
        fpn_indices = [_GATE_LAYER_IDX[s] for s in ("p3", "p4", "p5")]
        return [y[i] for i in fpn_indices]

    def extract(
        self,
        frame_bchw: torch.Tensor,
        boxes_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-box ReID embeddings from FPN center pixels.

        Args:
            frame_bchw: (1, 3, H, W) preprocessed frame on CUDA
            boxes_xyxy: (N, 4) detection boxes in pixel coordinates [x1, y1, x2, y2]

        Returns:
            (N, total_dim) L2-normalized embeddings on CUDA
        """
        h_img, w_img = frame_bchw.shape[2], frame_bchw.shape[3]
        h_ratio = h_img / self.img_size
        w_ratio = w_img / self.img_size

        with torch.no_grad():
            feats = self._get_fpn_feats(frame_bchw)

        n = boxes_xyxy.shape[0]
        embeddings = torch.empty(
            n, self.total_dim, device=boxes_xyxy.device, dtype=torch.float32
        )

        offset = 0
        for f in feats:
            f_h, f_w = f.shape[2], f.shape[3]
            dim = f.shape[1]

            cx = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 / w_ratio).float()
            cy = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 / h_ratio).float()

            cx_norm = cx / self.img_size
            cy_norm = cy / self.img_size

            cx_idx = (cx_norm * f_w).long().clamp(0, f_w - 1)
            cy_idx = (cy_norm * f_h).long().clamp(0, f_h - 1)

            torch.arange(n, device=boxes_xyxy.device)
            center_feat = f[0][:, cy_idx, cx_idx].T  # (N, dim)
            embeddings[:, offset : offset + dim] = center_feat
            offset += dim

        return F.normalize(embeddings, dim=1)

    @property
    def feature_dim(self) -> int:
        return self.total_dim
