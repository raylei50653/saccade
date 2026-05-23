"""
MambaGatedDetector — GatedYOLODetector backbone + MambaDetectionHead.

Architecture:
    YOLO26s backbone (layers 0-22)
        ↓  patched forward at layer 16 (P3), 19 (P4), 22 (P5)
    TrackSpatialGate  (gate = 1 + alpha * heatmap)
        ↓  gated FPN features captured via hooks
    MambaDetectionHead  (replaces YOLO Detect head, layer 23)
        ↓  per-scale cls_preds / reg_preds
    Postprocess decoder  (dist2bbox + top-k)
        ↓  (B, max_det, 6) xyxy boxes
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from torch import Tensor
from typing import Any

from .yolo_gated_detector import (
    GatedDetConfig,
    build_gated_yolo_detector,
    _GATE_LAYER_IDX,
)
from .mamba_head import MambaDetectionHead
from .yolo_conditioned import TrackerGateInput


def _postprocess_mamba(
    cls_preds: list[Tensor],
    reg_preds: list[Tensor],
    strides: Tensor,
    conf_thr: float,
    max_det: int,
) -> Tensor:
    from ultralytics.utils.tal import make_anchors, dist2bbox

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
    B, _, N = cls_all.shape

    anchors, anchor_strides = make_anchors(cls_preds, strides, 0.5)  # type: ignore[no-untyped-call]
    anchors = anchors.to(device=cls_all.device, dtype=cls_all.dtype)
    anchor_strides = anchor_strides.to(device=cls_all.device, dtype=cls_all.dtype)

    bboxes = dist2bbox(reg_all, anchors.T.unsqueeze(0), xywh=True, dim=1)  # type: ignore[no-untyped-call]
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)  # (1, N)
    bboxes = bboxes * strides_t

    xywh = bboxes.permute(0, 2, 1)
    x1y1 = xywh[..., :2] - xywh[..., 2:4] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:4] / 2
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)

    scores = cls_all.sigmoid()
    scores_max, class_ids = scores.max(dim=1)

    results = boxes_xyxy.new_zeros(B, max_det, 6)
    for b in range(B):
        mask = scores_max[b] >= conf_thr
        s = scores_max[b][mask]
        c = class_ids[b][mask].float()
        bx = boxes_xyxy[b][mask]

        n = min(s.shape[0], max_det)
        if n > 0:
            if s.shape[0] > max_det:
                _, topk = s.topk(max_det)
                s = s[topk]
                c = c[topk]
                bx = bx[topk]
            results[b, :n, :4] = bx[:n]
            results[b, :n, 4] = s[:n]
            results[b, :n, 5] = c[:n]

    return results


class MambaGatedDetector(nn.Module):
    """Gated YOLO backbone with Mamba SSM detection head.

    Wraps a pre-trained GatedYOLODetector for backbone + spatial gate,
    replaces YOLO Detect head with MambaDetectionHead.
    """

    def __init__(
        self,
        yolo_pt_path: str,
        teacher_ckpt: str,
        mamba_ckpt: str,
        cfg: GatedDetConfig | None = None,
        device: str | torch.device = "cuda",
        conf_thr: float = 0.25,
        max_det: int = 300,
    ):
        super().__init__()
        if cfg is None:
            cfg = GatedDetConfig()

        self.conf_thr = conf_thr
        self.max_det = max_det
        self.device = device
        self.img_size = cfg.img_size

        self.teacher = build_gated_yolo_detector(
            yolo_pt_path,
            cfg=cfg,
            device=device,
            weights_path=teacher_ckpt,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        mamba_state = torch.load(mamba_ckpt, map_location="cpu", weights_only=False)
        mamba_args = mamba_state["mamba_args"]

        self.mamba_head = MambaDetectionHead(
            in_channels=(128, 256, 512),
            d_model=mamba_args["d_model"],
            d_state=mamba_args["d_state"],
            num_blocks=mamba_args["num_blocks"],
            num_classes=mamba_args["num_classes"],
            reg_max=1,
            spatial_reduction=mamba_args["spatial_reduction"],
        ).to(device)
        sd = {
            k.replace("._orig_mod.", "."): v for k, v in mamba_state["student"].items()
        }
        self.mamba_head.load_state_dict(sd, strict=True)
        self.mamba_head.eval()
        for p in self.mamba_head.parameters():
            p.requires_grad_(False)

        self.stride = torch.tensor([8.0, 16.0, 32.0], device=device)

        self._fpn_feats: dict[str, Tensor] = {}
        self._hooks: list[Any] = []
        for scale in ("p3", "p4", "p5"):
            idx = _GATE_LAYER_IDX[scale]

            def _capture(
                _m: nn.Module,
                _i: tuple[Any, ...],
                _o: Tensor,
                s: str = scale,
            ) -> None:
                self._fpn_feats[s] = _o

            self._hooks.append(
                self.teacher.yolo_model.model[idx].register_forward_hook(_capture)
            )

    def forward(
        self,
        frame: Tensor,
        gate_input: TrackerGateInput | list[TrackerGateInput] | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        self._fpn_feats.clear()

        _ = self.teacher(frame, gate_input=gate_input)

        feats = [self._fpn_feats[s] for s in ("p3", "p4", "p5")]
        cls_preds, reg_preds = self.mamba_head(feats)

        detections = _postprocess_mamba(
            cls_preds, reg_preds, self.stride, self.conf_thr, self.max_det
        )

        return detections, {}

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def build_mamba_gated_detector(
    yolo_pt_path: str,
    teacher_ckpt: str,
    mamba_ckpt: str,
    img_size: int = 640,
    device: str | torch.device = "cuda",
    conf_thr: float = 0.25,
    max_det: int = 300,
) -> MambaGatedDetector:
    teacher_raw = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    train_args = teacher_raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))

    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
        gate_min_score=train_args.get("gate_min_score", 0.5),
        freeze_backbone=True,
        img_size=img_size,
    )

    model = MambaGatedDetector(
        yolo_pt_path=str(Path(yolo_pt_path).resolve()),
        teacher_ckpt=str(Path(teacher_ckpt).resolve()),
        mamba_ckpt=str(Path(mamba_ckpt).resolve()),
        cfg=cfg,
        device=device,
        conf_thr=conf_thr,
        max_det=max_det,
    )
    return model.to(device)
