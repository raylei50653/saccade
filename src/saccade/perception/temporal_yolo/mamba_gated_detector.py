"""
MambaGatedDetector — GatedYOLODetector backbone + MambaDetectionHead.

Architecture:
    YOLO26s backbone (layers 0-22) [PyTorch or TRT]
        ↓
    TrackSpatialGate  (gate = 1 + alpha * heatmap) [applied in PyTorch]
        ↓  gated FPN features
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
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)
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


# ---------------------------------------------------------------------------
# TRT YOLO backbone wrapper
# ---------------------------------------------------------------------------
class TRTYoloBackbone(nn.Module):
    """Runs YOLO backbone (layers 0-22) via TRT, returns P3/P4/P5 features."""

    def __init__(self, engine_path: str):
        super().__init__()
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [
            self.engine.get_tensor_name(i) for i in range(1, self.engine.num_io_tensors)
        ]
        self._stream = torch.cuda.Stream()

    def infer(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, C, H, W = images.shape
        images = images.contiguous()

        self.context.set_input_shape(self.input_name, (B, C, H, W))
        self.context.set_tensor_address(self.input_name, images.data_ptr())

        outputs: list[Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            shape = tuple(B if d == -1 else d for d in shape)  # type: ignore[assignment]
            buf = torch.empty(shape, dtype=torch.float32, device=images.device)
            self.context.set_tensor_address(name, buf.data_ptr())
            outputs.append(buf)

        self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return outputs[0], outputs[1], outputs[2]


# ---------------------------------------------------------------------------
# MambaGatedDetector
# ---------------------------------------------------------------------------
class MambaGatedDetector(nn.Module):
    """Gated YOLO backbone with Mamba SSM detection head.

    Supports both PyTorch and TRT backbone. When use_trt=True, the YOLO
    backbone runs via TensorRT (FP16) and the gate is applied in PyTorch
    after feature extraction.
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
        trt_backbone_engine: str = "",
    ):
        super().__init__()
        if cfg is None:
            cfg = GatedDetConfig()

        self.conf_thr = conf_thr
        self.max_det = max_det
        self.device = device
        self.img_size = cfg.img_size
        self._trt_backbone: TRTYoloBackbone | None = None

        self.teacher = build_gated_yolo_detector(
            yolo_pt_path,
            cfg=cfg,
            device=device,
            weights_path=teacher_ckpt,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Extract gate module + alpha for explicit gate application
        self.gate_module = self.teacher.gate

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

        # TRT backbone: loads pre-built engine, replaces PyTorch layer loop
        if trt_backbone_engine:
            self._trt_backbone = TRTYoloBackbone(trt_backbone_engine)

    def _forward_pytorch_backbone(self, frame: Tensor) -> list[Tensor]:
        teacher = self.teacher
        layers = teacher.yolo_model.model
        save: set[int] = set(teacher.yolo_model.save)

        y: list[Tensor | None] = []
        x: Any = frame
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
        return [y[i] for i in fpn_indices]  # type: ignore[return-value]

    def _apply_gate(
        self,
        feats: list[Tensor],
        gate_input: TrackerGateInput | list[TrackerGateInput] | None,
    ) -> list[Tensor]:
        if gate_input is None:
            return feats

        renderer = self.gate_module.renderer
        scales = ("p3", "p4", "p5")
        gated: list[Tensor] = []
        for i, (scale, feat) in enumerate(zip(scales, feats)):
            alpha = self.gate_module.alphas[scale]
            hw = (feat.shape[2], feat.shape[3])

            if isinstance(gate_input, list):
                maps = renderer.batch_forward(gate_input, hw)
                gate_map = 1.0 + alpha * maps.unsqueeze(1)
            else:
                heatmap = renderer(gate_input, hw)
                gate_map = (1.0 + alpha * heatmap).unsqueeze(0)

            gated.append(feat * gate_map)
        return gated

    def forward(
        self,
        frame: Tensor,
        gate_input: TrackerGateInput | list[TrackerGateInput] | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if self._trt_backbone is not None:
            p3, p4, p5 = self._trt_backbone.infer(frame)
            feats_raw = [p3, p4, p5]
            feats = self._apply_gate(feats_raw, gate_input)
        else:
            # PyTorch backbone already has gates injected → feats are gated
            teacher = self.teacher
            gls = teacher._gate_layers
            for gl in gls.values():
                gl._gate_input = gate_input

            feats = self._forward_pytorch_backbone(frame)

            for gl in gls.values():
                gl._gate_input = None

        cls_preds, reg_preds = self.mamba_head(feats)

        detections = _postprocess_mamba(
            cls_preds, reg_preds, self.stride, self.conf_thr, self.max_det
        )

        return detections, {}

    def remove_hooks(self) -> None:
        pass


def build_mamba_gated_detector(
    yolo_pt_path: str,
    teacher_ckpt: str,
    mamba_ckpt: str,
    img_size: int = 640,
    device: str | torch.device = "cuda",
    conf_thr: float = 0.25,
    max_det: int = 300,
    trt_backbone_engine: str = "",
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
        trt_backbone_engine=str(Path(trt_backbone_engine).resolve())
        if trt_backbone_engine
        else "",
    )
    return model.to(device)
