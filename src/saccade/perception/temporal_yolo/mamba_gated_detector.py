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
import torch.nn.functional as F
from pathlib import Path
from torch import Tensor
from typing import Any

from .yolo_gated_detector import (
    GatedDetConfig,
    build_gated_yolo_detector,
    _GATE_LAYER_IDX,
)
from .mamba_head import MambaDetectionHead, EmbeddingProjector
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
        emb_dim: int = 0,
        jde_proj_ckpt: str = "",
    ):
        super().__init__()
        if cfg is None:
            cfg = GatedDetConfig()

        self.conf_thr = conf_thr
        self.max_det = max_det
        self._device = device
        self.img_size = cfg.img_size
        self._trt_backbone: TRTYoloBackbone | None = None
        self.emb_dim = emb_dim
        self._emb_projector: EmbeddingProjector | None = None

        self.teacher = build_gated_yolo_detector(
            yolo_pt_path,
            cfg=cfg,
            device=device,
            weights_path=teacher_ckpt,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

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
            emb_dim=emb_dim,
        ).to(device)
        sd = {
            k.replace("._orig_mod.", "."): v for k, v in mamba_state["student"].items()
        }
        missing, unexpected = self.mamba_head.load_state_dict(sd, strict=False)
        if missing:
            print(f"[MambaHead] New params (init randomly): {missing}")
        if unexpected:
            print(f"[MambaHead] Unused params: {unexpected}")
        self.mamba_head.eval()
        for p in self.mamba_head.parameters():
            p.requires_grad_(False)

        self.stride = torch.tensor([8.0, 16.0, 32.0], device=device)

        from saccade.perception.tracking import GPUByteTracker  # noqa: E402

        self.tracker = GPUByteTracker(max_objects=2048)

        if trt_backbone_engine:
            self._trt_backbone = TRTYoloBackbone(trt_backbone_engine)

        if jde_proj_ckpt and emb_dim > 0:
            self._load_jde_projector(jde_proj_ckpt, device)

    def _load_jde_projector(self, ckpt_path: str, device: torch.device) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        proj_state = ckpt.get("projector", ckpt)
        self._emb_projector = EmbeddingProjector(
            emb_dim=self.emb_dim,
            hidden=256,
            out_dim=proj_state.get("out_dim", proj_state.get("emb_out_dim", 128)),
        ).to(device)
        self._emb_projector.load_state_dict(proj_state)
        self._emb_projector.eval()
        print(f"[JDE] Loaded embedding projector from {ckpt_path}")

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

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
            teacher = self.teacher
            gls = teacher._gate_layers
            # Enable feat caching so we can reuse raw FPN for ReID
            teacher.cache_feats = True
            for gl in gls.values():
                gl._gate_input = gate_input

            feats = self._forward_pytorch_backbone(frame)

            for gl in gls.values():
                gl._gate_input = None

        want_emb = self.mamba_head.emb_head is not None
        head_out = self.mamba_head(feats, return_embeddings=want_emb)
        if want_emb:
            cls_preds, reg_preds, emb_preds = head_out
        else:
            cls_preds, reg_preds = head_out

        detections = _postprocess_mamba(
            cls_preds, reg_preds, self.stride, self.conf_thr, self.max_det
        )

        extra: dict[str, Any] = {}
        if want_emb:
            extra["emb_preds"] = emb_preds

        return detections, extra

    def extract_det_embeddings(
        self, emb_preds: list[Tensor], boxes_xyxy: Tensor
    ) -> Tensor:
        pooled = self.mamba_head.pool_embeddings(emb_preds, boxes_xyxy)
        if self._emb_projector is not None:
            return self._emb_projector(pooled)
        return pooled

    def extract_fpn_embeddings(
        self,
        frame_bchw: Tensor | None,
        boxes_xyxy: Tensor,
    ) -> Tensor:
        """Zero-training ReID: center-pool raw FPN at each bbox.

        If called after forward(), reuses cached raw FPN features (single
        YOLO pass). Otherwise does a standalone backbone forward.

        Args:
            frame_bchw: (1, 3, H, W) frame, or None to reuse cached features.
            boxes_xyxy: (N, 4) detection boxes in pixel coords [x1, y1, x2, y2]

        Returns:
            (N, total_fpn_dim) L2-normalized embeddings on CUDA
        """
        from saccade.perception.temporal_yolo.yolo_gated_detector import (
            _GATE_LAYER_IDX,
        )

        # Try to reuse FPN features cached by forward()
        cache = {}
        if hasattr(self.teacher, "_gate_layers"):
            cache = {}
            for gl in self.teacher._gate_layers.values():
                cache.update(gl._feat_cache)
        if set(cache.keys()) == {"p3", "p4", "p5"}:
            feats = [cache[s] for s in ("p3", "p4", "p5")]
            frame_for_resolution = frame_bchw
        elif frame_bchw is not None:
            # Standalone forward
            layers = self.teacher.yolo_model.model
            save: set[int] = set(self.teacher.yolo_model.save)
            y: list[Tensor | None] = []
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
            feats = [y[i] for i in fpn_indices]
            frame_for_resolution = frame_bchw
        else:
            raise RuntimeError(
                "No cached FPN features and no frame provided. "
                "Call forward() before extract_fpn_embeddings(None, boxes)."
            )

        h_img = (
            frame_for_resolution.shape[2]
            if frame_for_resolution is not None
            else self.img_size
        )
        w_img = (
            frame_for_resolution.shape[3]
            if frame_for_resolution is not None
            else self.img_size
        )
        h_ratio = h_img / self.img_size
        w_ratio = w_img / self.img_size

        boxes_xyxy.shape[0]
        parts: list[Tensor] = []
        for f in feats:
            f_h, f_w = f.shape[2], f.shape[3]
            cx = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 / w_ratio).float()
            cy = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 / h_ratio).float()
            cx_norm = cx / self.img_size
            cy_norm = cy / self.img_size
            cx_idx = (cx_norm * f_w).long().clamp(0, f_w - 1)
            cy_idx = (cy_norm * f_h).long().clamp(0, f_h - 1)
            center_feat = f[0][:, cy_idx, cx_idx].mT
            parts.append(center_feat)
        return F.normalize(torch.cat(parts, dim=1), dim=1)

    def detect_raw(self, input_tensor: Tensor) -> Tensor:
        detections, _ = self.forward(input_tensor, gate_input=None)
        return detections

    def reset_tracker(self) -> None:
        from saccade.perception.tracking import GPUByteTracker  # noqa: E402

        self.tracker = GPUByteTracker(max_objects=2048)

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
    emb_dim: int = 0,
    jde_proj_ckpt: str = "",
) -> MambaGatedDetector:
    if teacher_ckpt and Path(teacher_ckpt).exists():
        teacher_raw = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        train_args = teacher_raw.get("args", {})
        scales = tuple(
            s.strip() for s in train_args.get("scales", "p3,p4,p5").split(",")
        )
        cfg = GatedDetConfig(
            scales=scales,
            gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
            gate_min_score=train_args.get("gate_min_score", 0.5),
            freeze_backbone=True,
            img_size=img_size,
        )
        teacher_path = teacher_ckpt
    else:
        cfg = GatedDetConfig(
            scales=("p3", "p4", "p5"),
            freeze_backbone=True,
            img_size=img_size,
        )
        teacher_path = ""

    model = MambaGatedDetector(
        yolo_pt_path=str(Path(yolo_pt_path).resolve()),
        teacher_ckpt=teacher_path,
        mamba_ckpt=str(Path(mamba_ckpt).resolve()),
        cfg=cfg,
        device=device,
        conf_thr=conf_thr,
        max_det=max_det,
        trt_backbone_engine=str(Path(trt_backbone_engine).resolve())
        if trt_backbone_engine
        else "",
        emb_dim=emb_dim,
        jde_proj_ckpt=str(Path(jde_proj_ckpt).resolve()) if jde_proj_ckpt else "",
    )
    return model.to(device)
