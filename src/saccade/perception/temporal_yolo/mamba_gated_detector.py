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

from collections import deque
import hashlib

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
from .mamba_head import (
    DEFAULT_MAMBA_IN_CHANNELS,
    EmbeddingProjector,
    MambaDetectionHead,
    resolve_mamba_in_channels,
)
from .yolo_conditioned import TrackerGateInput


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fuse_small_p3_scores(
    scores_max: Tensor,
    boxes_xyxy: Tensor,
    cls_preds: list[Tensor],
    threshold: float,
    *,
    box_scale_x: Tensor | None = None,
    box_scale_y: Tensor | None = None,
) -> Tensor:
    """Use P3 boxes with the strongest aligned P3/P4/P5 score for small boxes."""
    level_counts = [pred.shape[-2] * pred.shape[-1] for pred in cls_preds]
    level_scores = scores_max.split(level_counts, dim=1)
    level_boxes = boxes_xyxy.split(level_counts, dim=1)
    p3_h, p3_w = cls_preds[0].shape[-2:]

    aligned_scores = [level_scores[0].reshape(-1, 1, p3_h, p3_w)]
    for score, pred in zip(level_scores[1:], cls_preds[1:]):
        score_map = score.reshape(-1, 1, pred.shape[-2], pred.shape[-1])
        aligned_scores.append(
            F.interpolate(score_map, size=(p3_h, p3_w), mode="nearest")
        )
    fused_p3 = torch.stack(aligned_scores, dim=0).amax(dim=0).flatten(1)

    routed_scores: list[Tensor] = []
    for level, (score, boxes) in enumerate(zip(level_scores, level_boxes)):
        box_wh = boxes[..., 2:4] - boxes[..., :2]
        width = box_wh[..., 0]
        height = box_wh[..., 1]
        if box_scale_x is not None:
            width = width * box_scale_x
        if box_scale_y is not None:
            height = height * box_scale_y
        small = torch.minimum(width, height) < threshold
        if level == 0:
            routed_scores.append(torch.where(small, fused_p3, score))
        else:
            routed_scores.append(score.masked_fill(small, -1.0))
    return torch.cat(routed_scores, dim=1)


def _dfl_decode(reg_all: Tensor) -> Tensor:
    """DFL decode: (B, reg_max*4, N) distribution logits -> (B, 4, N) distances.

    reg_max is inferred from the channel count (reg_max*4). Channel layout is
    4 sides x reg_max bins (side-outer, bin-inner), matching ultralytics
    v8DetectionLoss.bbox_decode. No-op when reg_max == 1 (direct regression).
    """
    b, c, n = reg_all.shape
    reg_max = c // 4
    if reg_max <= 1:
        return reg_all
    proj = torch.arange(reg_max, dtype=reg_all.dtype, device=reg_all.device).view(
        1, 1, reg_max, 1
    )
    return (reg_all.view(b, 4, reg_max, n).softmax(2) * proj).sum(2)


def _postprocess_mamba(
    cls_preds: list[Tensor],
    reg_preds: list[Tensor],
    strides: Tensor,
    conf_thr: float,
    max_det: int,
    small_p3_max_threshold: float = 0.0,
) -> Tensor:
    from ultralytics.utils.tal import make_anchors, dist2bbox

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
    B, _, N = cls_all.shape

    anchors, anchor_strides = make_anchors(cls_preds, strides, 0.5)  # type: ignore[no-untyped-call]
    anchors = anchors.to(device=cls_all.device, dtype=cls_all.dtype)
    anchor_strides = anchor_strides.to(device=cls_all.device, dtype=cls_all.dtype)

    bboxes = dist2bbox(_dfl_decode(reg_all), anchors.T.unsqueeze(0), xywh=True, dim=1)  # type: ignore[no-untyped-call]
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)
    bboxes = bboxes * strides_t

    xywh = bboxes.permute(0, 2, 1)
    x1y1 = xywh[..., :2] - xywh[..., 2:4] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:4] / 2
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)

    scores = cls_all.sigmoid()
    scores_max, class_ids = scores.max(dim=1)
    if small_p3_max_threshold > 0:
        scores_max = _fuse_small_p3_scores(
            scores_max, boxes_xyxy, cls_preds, small_p3_max_threshold
        )

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
# Precomputed anchors for CUDA-graph-safe postprocess (torch.full / arange
# are not capturable).
# ---------------------------------------------------------------------------


def _precompute_anchor_grid(
    stride_tensor: Tensor,
    feat_shapes: list[tuple[int, int, int, int]],
) -> tuple[Tensor, Tensor]:
    anchor_points = []
    stride_points = []
    for i in range(len(stride_tensor)):
        _, _, h, w = feat_shapes[i]
        sx = torch.arange(w, dtype=torch.float32, device=stride_tensor.device) + 0.5
        sy = torch.arange(h, dtype=torch.float32, device=stride_tensor.device) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_points.append(
            torch.full(
                (h * w, 1),
                stride_tensor[i].item(),
                dtype=torch.float32,
                device=stride_tensor.device,
            )
        )
    return torch.cat(anchor_points), torch.cat(stride_points)


def _postprocess_mamba_fixed(
    cls_preds: list[Tensor],
    reg_preds: list[Tensor],
    strides: Tensor,
    conf_thr: float,
    max_det: int,
    *,
    anchors: Tensor | None = None,
    anchor_strides: Tensor | None = None,
    small_p3_max_threshold: float = 0.0,
    box_scale_x: Tensor | None = None,
    box_scale_y: Tensor | None = None,
    _compile: bool | None = None,
) -> Tensor:
    """Decode Mamba head outputs to (B, max_det, 6) detections.

    Set ``_compile=False`` to force eager execution (e.g. for correctness
    checks). By default, uses ``torch.compile`` when available (PyTorch ≥ 2.0)
    to fuse the pointwise decode chain.
    """
    if _compile is None:
        _compile = _POSTPROCESS_COMPILE_ENABLED
    if _compile:
        compiled = _get_compiled_postprocess()
        return compiled(
            cls_preds,
            reg_preds,
            strides,
            conf_thr,
            max_det,
            anchors=anchors,
            anchor_strides=anchor_strides,
            small_p3_max_threshold=small_p3_max_threshold,
            box_scale_x=box_scale_x,
            box_scale_y=box_scale_y,
        )
    return _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr,
        max_det,
        anchors=anchors,
        anchor_strides=anchor_strides,
        small_p3_max_threshold=small_p3_max_threshold,
        box_scale_x=box_scale_x,
        box_scale_y=box_scale_y,
    )


def _postprocess_mamba_fixed_eager(
    cls_preds: list[Tensor],
    reg_preds: list[Tensor],
    strides: Tensor,
    conf_thr: float,
    max_det: int,
    *,
    anchors: Tensor | None = None,
    anchor_strides: Tensor | None = None,
    small_p3_max_threshold: float = 0.0,
    box_scale_x: Tensor | None = None,
    box_scale_y: Tensor | None = None,
) -> Tensor:
    from ultralytics.utils.tal import dist2bbox

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)

    if anchors is None or anchor_strides is None:
        anchors, anchor_strides = _precompute_anchor_grid(
            strides, [tuple(c.shape) for c in cls_preds]
        )

    bboxes = dist2bbox(_dfl_decode(reg_all), anchors.T.unsqueeze(0), xywh=True, dim=1)
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)
    bboxes = bboxes * strides_t

    xywh = bboxes.permute(0, 2, 1)
    x1y1 = xywh[..., :2] - xywh[..., 2:4] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:4] / 2
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)

    scores = cls_all.sigmoid()
    scores_max, class_ids = scores.max(dim=1)
    if small_p3_max_threshold > 0:
        scores_max = _fuse_small_p3_scores(
            scores_max,
            boxes_xyxy,
            cls_preds,
            small_p3_max_threshold,
            box_scale_x=box_scale_x,
            box_scale_y=box_scale_y,
        )

    results = boxes_xyxy.new_zeros(cls_all.shape[0], max_det, 6)
    for b in range(cls_all.shape[0]):
        topk_scores, topk_idx = scores_max[b].topk(max_det)
        results[b, :, :4] = boxes_xyxy[b][topk_idx]
        results[b, :, 4] = topk_scores
        results[b, :, 5] = class_ids[b][topk_idx].float()
    return results


# Compiled variant (created lazily on first call).
_postprocess_mamba_fixed_compiled: Any = None
_POSTPROCESS_COMPILE_ENABLED: bool = True


def _get_compiled_postprocess() -> Any:
    global _postprocess_mamba_fixed_compiled
    if _postprocess_mamba_fixed_compiled is None and _POSTPROCESS_COMPILE_ENABLED:
        try:
            _postprocess_mamba_fixed_compiled = torch.compile(
                _postprocess_mamba_fixed_eager,
                mode="default",
                fullgraph=False,
            )
        except Exception:
            _postprocess_mamba_fixed_compiled = _postprocess_mamba_fixed_eager
    return _postprocess_mamba_fixed_compiled


def set_postprocess_compile(enabled: bool) -> None:
    """Enable or disable ``torch.compile`` on the detection postprocess."""
    global _POSTPROCESS_COMPILE_ENABLED, _postprocess_mamba_fixed_compiled
    _POSTPROCESS_COMPILE_ENABLED = enabled
    if not enabled:
        _postprocess_mamba_fixed_compiled = None


# ---------------------------------------------------------------------------
# TRT YOLO backbone wrapper
# ---------------------------------------------------------------------------
class TRTYoloBackbone(nn.Module):
    """Runs YOLO backbone (layers 0-22) via TRT, returns P3/P4/P5 features."""

    def __init__(self, engine_path: str):
        super().__init__()
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [
            self.engine.get_tensor_name(i) for i in range(1, self.engine.num_io_tensors)
        ]
        self.output_channels = tuple(
            int(self.engine.get_tensor_shape(name)[1]) for name in self.output_names
        )
        self._stream = torch.cuda.Stream()
        self._output_bufs: list[Tensor] | None = None
        self._last_batch = -1

    def infer(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, C, H, W = images.shape
        images = images.contiguous()

        self.context.set_input_shape(self.input_name, (B, C, H, W))
        self.context.set_tensor_address(self.input_name, images.data_ptr())

        if B != self._last_batch:
            self._output_bufs = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                shape = tuple(B if d == -1 else d for d in shape)
                buf = torch.empty(shape, dtype=torch.float32, device=images.device)
                self.context.set_tensor_address(name, buf.data_ptr())
                self._output_bufs.append(buf)
            self._last_batch = B
        else:
            for name, buf in zip(self.output_names, self._output_bufs):
                self.context.set_tensor_address(name, buf.data_ptr())

        current_stream = torch.cuda.current_stream()
        event_input = torch.cuda.Event()
        event_input.record(current_stream)
        self._stream.wait_event(event_input)

        self.context.execute_async_v3(self._stream.cuda_stream)

        event_output = torch.cuda.Event()
        event_output.record(self._stream)
        current_stream.wait_event(event_output)

        return self._output_bufs[0], self._output_bufs[1], self._output_bufs[2]

    def infer_graph(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """TRT inference on the *calling* stream — graph-capturable.

        Uses no dedicated stream or cross-stream events.  Callers must ensure
        the calling stream is the CUDA-graph capture stream.
        ``set_tensor_address`` calls are host-side API (not captured); only
        ``execute_async_v3`` lands in the graph.
        """
        B, C, H, W = images.shape
        images = images.contiguous()

        self.context.set_input_shape(self.input_name, (B, C, H, W))
        self.context.set_tensor_address(self.input_name, images.data_ptr())

        if B != self._last_batch:
            self._output_bufs = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                shape = tuple(B if d == -1 else d for d in shape)
                buf = torch.empty(shape, dtype=torch.float32, device=images.device)
                self.context.set_tensor_address(name, buf.data_ptr())
                self._output_bufs.append(buf)
            self._last_batch = B
        else:
            for name, buf in zip(self.output_names, self._output_bufs):
                self.context.set_tensor_address(name, buf.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self._output_bufs[0], self._output_bufs[1], self._output_bufs[2]


# ---------------------------------------------------------------------------
# Per-stream temporal state
# ---------------------------------------------------------------------------
class StreamState:
    """Per-stream temporal state: FPN feature ring buffers + GMC warp buffer.

    A single-stream ``MambaGatedDetector`` owns exactly one of these; the
    multi-stream server owns one per stream so that each stream's temporal
    window and GMC trajectory stay independent.
    """

    def __init__(
        self,
        temporal_T: int,
        temporal_buffer: list[deque[Tensor]] | None,
        gmc_mat_buffer: deque[Tensor],
    ) -> None:
        self.temporal_T = temporal_T
        self.temporal_buffer = temporal_buffer
        self.gmc_mat_buffer = gmc_mat_buffer

    @classmethod
    def create(cls, temporal_T: int) -> StreamState:
        buf = [deque[Tensor]() for _ in range(3)] if temporal_T > 0 else None
        return cls(temporal_T, buf, deque[Tensor]())

    def push_gmc(self, warp: Tensor) -> None:
        self.gmc_mat_buffer.append(warp.detach())
        if len(self.gmc_mat_buffer) > self.temporal_T:
            self.gmc_mat_buffer.popleft()

    def reset(self) -> None:
        if self.temporal_buffer is not None:
            for buf in self.temporal_buffer:
                buf.clear()
        self.gmc_mat_buffer.clear()


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
        temporal_T: int = 0,
        use_cuda_graph: bool = False,
        use_whole_graph: bool = False,
        small_p3_max_threshold: float = 0.0,
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
        self._trt_feat_cache: dict[str, Tensor] = {}

        self._temporal_T: int = temporal_T
        self._stream_state = StreamState.create(temporal_T)

        self.use_whole_graph = use_whole_graph
        self.small_p3_max_threshold = float(small_p3_max_threshold)
        self._whole_graph_nms_pad = 0  # set to 2048 to pad whole-graph output for NMS
        self._whole_graph_anchors: Tensor | None = None
        self._whole_graph_anchor_strides: Tensor | None = None
        self._whole_graphed_callables: dict = {}
        self._whole_graph_warm = False
        self._whole_graph_sx: Tensor = torch.ones(1, device=device)
        self._whole_graph_sy: Tensor = torch.ones(1, device=device)
        self._whole_graph_x_idx: Tensor = torch.tensor(
            [0, 2], device=device, dtype=torch.long
        )
        self._whole_graph_y_idx: Tensor = torch.tensor(
            [1, 3], device=device, dtype=torch.long
        )
        self._whole_graph_img_shape: tuple[int, int] = (0, 0)

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
        self.in_channels = resolve_mamba_in_channels(mamba_state)
        expected_yolo_sha = mamba_args.get("base_yolo_sha256")
        if expected_yolo_sha and _sha256_file(yolo_pt_path) != expected_yolo_sha:
            raise ValueError(
                "Base YOLO weights do not match the Mamba checkpoint lineage: "
                f"checkpoint={mamba_args.get('base_yolo_path')!r}, "
                f"selected={yolo_pt_path!r}"
            )
        expected_teacher_sha = mamba_args.get("teacher_checkpoint_sha256")
        if (
            expected_teacher_sha
            and teacher_ckpt
            and _sha256_file(teacher_ckpt) != expected_teacher_sha
        ):
            raise ValueError(
                "Teacher checkpoint does not match the Mamba checkpoint lineage: "
                f"selected={teacher_ckpt!r}"
            )
        dense_detail_fusion = mamba_args.get("use_detail_fusion", False)
        self.use_strip_detail = mamba_args.get("use_strip_detail", False)
        self.strip_detail_type = mamba_args.get("strip_detail_type", "mamba")
        self.strip_route_budget = int(mamba_args.get("strip_route_budget", 640))
        self.strip_small_threshold = float(mamba_args.get("strip_small_threshold", 8.0))
        self.use_detail_fusion = dense_detail_fusion or self.use_strip_detail
        self.detail_source = mamba_args.get("detail_source", "high")
        self.detail_size = (
            int(mamba_args.get("detail_height", 768)),
            int(mamba_args.get("detail_width", 1280)),
        )
        if self.use_detail_fusion and temporal_T > 0:
            raise ValueError(
                "detail fusion inference currently supports spatial v14 only "
                "(temporal_T must be 0)"
            )
        self._strip_detail_positions: Tensor | None = None
        self._strip_detail_position_mask: Tensor | None = None

        self.mamba_head = MambaDetectionHead(
            in_channels=self.in_channels,
            d_model=mamba_args["d_model"],
            d_state=mamba_args["d_state"],
            num_blocks=mamba_args["num_blocks"],
            num_classes=mamba_args["num_classes"],
            reg_max=mamba_args.get("reg_max", 1),
            spatial_reduction=mamba_args["spatial_reduction"],
            emb_dim=emb_dim,
            use_pixel_shuffle=mamba_args.get("use_pixel_shuffle", False),
            use_cross_scan=mamba_args.get("use_cross_scan", False),
            use_hybrid_head=mamba_args.get("use_hybrid_head", False),
            use_temporal_mamba=mamba_args.get("use_temporal_mamba", False),
            per_channel_a=mamba_args.get("per_channel_a", False),
            legacy_n1_scan=mamba_args.get("legacy_n1_scan", False),
            use_cuda_graph=use_cuda_graph,
            reduction_variant=mamba_args.get("reduction_variant", "conv"),
            use_detail_fusion=dense_detail_fusion,
            detail_channels=mamba_args.get("detail_channels", 32),
            detail_patch_size=mamba_args.get("detail_patch_size", 3),
            detail_feature_channels=mamba_args.get("detail_feature_channels", 0),
            use_strip_detail=self.use_strip_detail,
            strip_detail_type=self.strip_detail_type,
            strip_stem_channels=mamba_args.get("strip_stem_channels", 32),
            strip_length=mamba_args.get("strip_length", 32),
            strip_width=mamba_args.get("strip_width", 3),
            strip_route_chunk_size=mamba_args.get("strip_route_chunk_size", 16),
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
            if self._trt_backbone.output_channels != self.in_channels:
                raise ValueError(
                    "TRT backbone FPN channels do not match the Mamba checkpoint: "
                    f"engine={self._trt_backbone.output_channels}, "
                    f"checkpoint={self.in_channels}"
                )

        if self.use_whole_graph and self._trt_backbone is not None:
            if (
                self.detail_source in {"native-p3", "strip-oracle"}
                or self.use_strip_detail
            ):
                raise RuntimeError(
                    f"{self.detail_source} is Python-eager only; "
                    "whole-graph TRT is unsupported"
                )
            _s = self.img_size
            FEAT_SHAPES = [
                (1, self.in_channels[0], _s // 8, _s // 8),
                (1, self.in_channels[1], _s // 16, _s // 16),
                (1, self.in_channels[2], _s // 32, _s // 32),
            ]
            self._whole_graph_anchors, self._whole_graph_anchor_strides = (
                _precompute_anchor_grid(self.stride, FEAT_SHAPES)
            )

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

    def set_whole_graph_img_dims(self, h_orig: int, w_orig: int) -> None:
        if (h_orig, w_orig) == self._whole_graph_img_shape:
            return
        self._whole_graph_img_shape = (h_orig, w_orig)
        self._whole_graph_sx.fill_(w_orig / self.img_size)
        self._whole_graph_sy.fill_(h_orig / self.img_size)
        self._whole_graphed_callables.clear()
        self._whole_graph_warm = False

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def cpp_ptr(self) -> int:
        if self.use_detail_fusion:
            raise RuntimeError(
                "C++ Mamba detector does not yet support the dual-resolution detail input"
            )
        if self.in_channels != DEFAULT_MAMBA_IN_CHANNELS:
            raise RuntimeError(
                "C++ Mamba detector currently assumes yolo26s FPN channels "
                f"{DEFAULT_MAMBA_IN_CHANNELS}; checkpoint uses {self.in_channels}. "
                "Use the Python whole-graph path for yolo26m/yolo26l."
            )
        if not hasattr(self, "_cpp_detector") or self._cpp_detector is None:
            from saccade_perception_ext import (
                MambaGatedDetector as CppMambaGatedDetector,
            )
            from pathlib import Path

            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent

            # Resolve TRT backbone path
            trt_path = project_root / "models/yolo/yolo26s_backbone_640_best.engine"
            if not trt_path.exists():
                trt_path = Path("models/yolo/yolo26s_backbone_640_best.engine")

            # Resolve Mamba head TorchScript path
            mamba_head_path = project_root / "models/yolo/mamba_head_best.pt"
            if not mamba_head_path.exists():
                mamba_head_path = Path("models/yolo/mamba_head_best.pt")

            print("[Python MambaGatedDetector] Initializing C++ counterpart with:")
            print(f"  Backbone: {trt_path}")
            print(f"  Mamba Head JIT: {mamba_head_path}")

            self._cpp_detector = CppMambaGatedDetector(
                str(trt_path.resolve()),
                str(mamba_head_path.resolve()),
                self.img_size,
                self.conf_thr,
            )
        return int(self._cpp_detector.cpp_ptr)

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

    def _prepare_detail_view(self, frame: Tensor) -> tuple[Tensor, Tensor]:
        """Resize with preserved aspect ratio and pad right/bottom to the bucket."""
        target_h, target_w = self.detail_size
        src_h, src_w = frame.shape[-2:]
        scale = min(target_h / src_h, target_w / src_w)
        valid_h = max(1, min(target_h, int(round(src_h * scale))))
        valid_w = max(1, min(target_w, int(round(src_w * scale))))
        detail = F.interpolate(
            frame,
            size=(valid_h, valid_w),
            mode="bilinear",
            align_corners=False,
        )
        detail = F.pad(
            detail, (0, target_w - valid_w, 0, target_h - valid_h), value=114.0 / 255.0
        )
        valid_hw = torch.tensor(
            [valid_h, valid_w],
            device=frame.device,
            dtype=torch.int64,
        ).expand(frame.shape[0], -1)
        return detail, valid_hw

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

    @torch.no_grad()
    def forward(
        self,
        frame: Tensor,
        gate_input: TrackerGateInput | list[TrackerGateInput] | None = None,
        detail_frame: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if self.use_whole_graph and self._trt_backbone is not None:
            if detail_frame is not None:
                raise ValueError(
                    "whole-graph detail fusion uses the original frame argument; "
                    "do not pass detail_frame separately"
                )
            return self._forward_whole_graph(frame)
        detail_images = None
        detail_valid_hw = None
        detail_features = None
        if self.use_detail_fusion:
            if self.detail_source == "global":
                detail_images = frame
                detail_valid_hw = torch.tensor(
                    frame.shape[-2:],
                    device=frame.device,
                    dtype=torch.int64,
                ).expand(frame.shape[0], -1)
            elif detail_frame is None:
                raise ValueError(
                    "detail fusion requires the original-resolution detail_frame"
                )
            else:
                detail_images, detail_valid_hw = self._prepare_detail_view(detail_frame)
                if self.detail_source == "native-p3":
                    detail_features = self._forward_pytorch_backbone(detail_images)[0]
                    detail_images = None
        if self.use_strip_detail and self._strip_detail_positions is None:
            batch_size = frame.shape[0]
            device = frame.device
            budget = self.strip_route_budget
            cols = max(1, int(budget**0.5))
            dummy_pos = torch.zeros(
                (batch_size, budget, 2), device=device, dtype=torch.long
            )
            for j in range(budget):
                dummy_pos[:, j, 0] = 40 + (j // cols)
                dummy_pos[:, j, 1] = 40 + (j % cols)
            self._strip_detail_positions = dummy_pos
            self._strip_detail_position_mask = torch.ones(
                (batch_size, budget), device=device, dtype=torch.bool
            )

        if self._trt_backbone is not None:
            p3, p4, p5 = self._trt_backbone.infer(frame)
            self._trt_feat_cache = {"p3": p3, "p4": p4, "p5": p5}
            feats_raw = [p3, p4, p5]
            feats = self._apply_gate(feats_raw, gate_input)
        else:
            teacher = self.teacher
            gls = teacher._gate_layers
            teacher.cache_feats = True
            for gl in gls.values():
                gl._gate_input = gate_input

            feats = self._forward_pytorch_backbone(frame)

            for gl in gls.values():
                gl._gate_input = None

        res = self._detect_from_feats(
            feats,
            self._stream_state,
            detail_images=detail_images,
            detail_valid_hw=detail_valid_hw,
            detail_features=detail_features,
            detail_feature_stride=8,
            detail_positions=self._strip_detail_positions,
            detail_position_mask=self._strip_detail_position_mask,
        )

        if self.use_strip_detail:
            self._strip_detail_positions = None
            self._strip_detail_position_mask = None

        return res

    @torch.no_grad()
    def _forward_whole_graph(self, frame: Tensor) -> Tensor:
        key = (
            tuple(frame.shape)
            + self._whole_graph_img_shape
            + (self._whole_graph_nms_pad,)
        )
        if key not in self._whole_graphed_callables:
            if not self._whole_graph_warm:
                self._whole_graph_warmup(frame)
            self._whole_graph_capture(frame)

        graphed = self._whole_graphed_callables[key]
        detections = graphed(frame)
        return detections, {}

    @torch.no_grad()
    def _forward_whole_graph_preprocessed(self, frame: Tensor) -> Tensor:
        key = (
            "preprocessed",
            tuple(frame.shape),
            self._whole_graph_nms_pad,
        )
        if key not in self._whole_graphed_callables:
            if not self._whole_graph_warm:
                self._whole_graph_warmup_preprocessed(frame)
            self._whole_graph_capture_preprocessed(frame)

        graphed = self._whole_graphed_callables[key]
        detections = graphed(frame)
        return detections, {}

    def _whole_graph_fn(self, frame: Tensor) -> Tensor:
        frame_640 = F.interpolate(
            frame,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        backbone = self._trt_backbone
        p3, p4, p5 = backbone.infer_graph(frame_640)
        detail_images = None
        detail_valid_hw = None
        if self.use_detail_fusion:
            if self.detail_source == "global":
                detail_images = frame_640
                detail_valid_hw = torch.full(
                    (frame.shape[0], 2),
                    self.img_size,
                    device=frame.device,
                    dtype=torch.int64,
                )
            else:
                detail_images, detail_valid_hw = self._prepare_detail_view(frame)
        cls_preds, reg_preds = self.mamba_head._forward_eager(
            [p3, p4, p5],
            return_embeddings=False,
            detail_images=detail_images,
            detail_valid_hw=detail_valid_hw,
        )
        detections = _postprocess_mamba_fixed(
            cls_preds,
            reg_preds,
            self.stride,
            self.conf_thr,
            max(self.max_det, self._whole_graph_nms_pad),
            anchors=self._whole_graph_anchors,
            anchor_strides=self._whole_graph_anchor_strides,
            small_p3_max_threshold=self.small_p3_max_threshold,
            box_scale_x=self._whole_graph_sx,
            box_scale_y=self._whole_graph_sy,
        )
        detections[:, :, self._whole_graph_x_idx] *= self._whole_graph_sx
        detections[:, :, self._whole_graph_y_idx] *= self._whole_graph_sy
        return detections

    def _whole_graph_fn_preprocessed(self, frame: Tensor) -> Tensor:
        if self.use_detail_fusion:
            raise RuntimeError(
                "preprocessed whole-graph input has no original-resolution detail frame"
            )
        backbone = self._trt_backbone
        p3, p4, p5 = backbone.infer_graph(frame)
        cls_preds, reg_preds = self.mamba_head._forward_eager(
            [p3, p4, p5],
            return_embeddings=False,
        )
        return _postprocess_mamba_fixed(
            cls_preds,
            reg_preds,
            self.stride,
            self.conf_thr,
            max(self.max_det, self._whole_graph_nms_pad),
            anchors=self._whole_graph_anchors,
            anchor_strides=self._whole_graph_anchor_strides,
            small_p3_max_threshold=self.small_p3_max_threshold,
        )

    def _whole_graph_warmup(self, frame: Tensor) -> None:
        with torch.no_grad():
            warm = frame.clone()
            _ = self._whole_graph_fn(warm)
            torch.cuda.synchronize()
        self._whole_graph_warm = True

    def _whole_graph_warmup_preprocessed(self, frame: Tensor) -> None:
        with torch.no_grad():
            warm = frame.clone()
            _ = self._whole_graph_fn_preprocessed(warm)
            torch.cuda.synchronize()
        self._whole_graph_warm = True

    def _whole_graph_capture(self, frame: Tensor) -> None:
        if len(self._whole_graphed_callables) >= 10:
            self._whole_graphed_callables.clear()
        key = (
            tuple(frame.shape)
            + self._whole_graph_img_shape
            + (self._whole_graph_nms_pad,)
        )
        print(
            f"🕯️ [WholeDetectGraph] Capturing graphed callable for "
            f"shape {tuple(frame.shape)} img={self._whole_graph_img_shape}"
        )
        sample = frame.clone()
        graphed = torch.cuda.make_graphed_callables(self._whole_graph_fn, (sample,))
        self._whole_graphed_callables[key] = graphed

    def _whole_graph_capture_preprocessed(self, frame: Tensor) -> None:
        if len(self._whole_graphed_callables) >= 10:
            self._whole_graphed_callables.clear()
        key = (
            "preprocessed",
            tuple(frame.shape),
            self._whole_graph_nms_pad,
        )
        print(
            f"🕯️ [WholeDetectGraph] Capturing graphed callable for "
            f"preprocessed shape {tuple(frame.shape)}"
        )
        sample = frame.clone()
        graphed = torch.cuda.make_graphed_callables(
            self._whole_graph_fn_preprocessed, (sample,)
        )
        self._whole_graphed_callables[key] = graphed

    @torch.no_grad()
    def _detect_from_feats(
        self,
        feats: list[Tensor],
        state: StreamState,
        detail_images: Tensor | None = None,
        detail_valid_hw: Tensor | None = None,
        detail_features: Tensor | None = None,
        detail_feature_stride: int = 8,
        detail_positions: Tensor | None = None,
        detail_position_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Temporal-window assembly + Mamba head + post-process for one frame.

        Shared by the single-stream ``forward()`` and the multi-stream server.
        ``feats`` are the (already gated) raw FPN features for the *current*
        frame; ``state`` holds this stream's temporal feature ring buffer and
        GMC warp buffer, so concurrent streams stay independent.
        """
        emb_preds: list[Tensor] | None = None
        if state.temporal_buffer is not None:
            for si in range(3):
                state.temporal_buffer[si].append(feats[si].detach())
                if len(state.temporal_buffer[si]) > state.temporal_T:
                    state.temporal_buffer[si].popleft()
            T_buf = len(state.temporal_buffer[0])
            feats = [
                torch.cat(list(state.temporal_buffer[si]), dim=0) for si in range(3)
            ]
            flows = None
            if len(state.gmc_mat_buffer) >= 2:
                flows = []
                for si in range(3):
                    _, _, Hf, Wf = feats[si].shape
                    B = feats[si].shape[0] // T_buf
                    flow_scales = []
                    for t in range(T_buf):
                        f = _gmc_matrices_to_flow(
                            state.gmc_mat_buffer, t, T_buf - 1, Hf, Wf, feats[si].device
                        )
                        flow_scales.append(f.unsqueeze(0).expand(B, -1, -1, -1))
                    flows.append(torch.cat(flow_scales, dim=0))
            want_emb = self.mamba_head.emb_head is not None
            head_out = self.mamba_head(
                feats, return_embeddings=want_emb, T=T_buf, flows=flows
            )
            B = feats[0].shape[0] // T_buf
            if want_emb:
                cls_preds, reg_preds, emb_preds = head_out
            else:
                cls_preds, reg_preds = head_out
            cls_preds = [c[(T_buf - 1) * B :] for c in cls_preds]
            reg_preds = [r[(T_buf - 1) * B :] for r in reg_preds]
        else:
            want_emb = self.mamba_head.emb_head is not None
            head_out = self.mamba_head(
                feats,
                return_embeddings=want_emb,
                detail_images=detail_images,
                detail_valid_hw=detail_valid_hw,
                detail_features=detail_features,
                detail_feature_stride=detail_feature_stride,
                detail_positions=detail_positions,
                detail_position_mask=detail_position_mask,
            )
            if want_emb:
                cls_preds, reg_preds, emb_preds = head_out
            else:
                cls_preds, reg_preds = head_out

        detections = _postprocess_mamba(
            cls_preds,
            reg_preds,
            self.stride,
            self.conf_thr,
            self.max_det,
            self.small_p3_max_threshold,
        )

        extra: dict[str, Any] = {}
        if want_emb and emb_preds is not None:
            extra["emb_preds"] = emb_preds

        return detections, extra

    @torch.no_grad()
    def _detect_batch(
        self,
        feats_list: list[list[Tensor]],
        states: list[StreamState],
    ) -> list[tuple[Tensor, dict[str, Any]]]:
        """Batched detection across N streams sharing the same temporal depth.

        Streams are stacked **batch-major** (flat row = ``b * T + t``) so the
        layout matches the head's internal ``x.reshape(B, T_frames, ...)``
        (mamba_head.py). Each stream keeps its own temporal buffer + GMC flows,
        so per-stream results equal the single-stream ``_detect_from_feats``
        path — this only amortizes the head launch across streams.

        NOTE on layout: the single-stream path builds flows time-major and uses
        the slice ``c[(T-1)*B:]``; both coincide with batch-major only at B=1.
        This method uses batch-major throughout, which is the correct general
        form for B>1.

        ``feats_list[b]`` is stream b's ``[p3, p4, p5]`` for the *current* frame
        (each leading dim 1). All states must already share the same buffer
        depth (the server groups by depth before calling).
        """
        n = len(feats_list)
        assert n > 0 and n == len(states)
        if self.use_detail_fusion:
            raise RuntimeError(
                "batched multi-stream detail fusion requires per-stream original frames"
            )
        if self.use_strip_detail:
            raise RuntimeError(
                "batched multi-stream detection does not support strip detail fusion"
            )
        want_emb = self.mamba_head.emb_head is not None
        temporal = states[0].temporal_buffer is not None

        if temporal:
            for feats, st in zip(feats_list, states):
                assert st.temporal_buffer is not None
                for si in range(3):
                    st.temporal_buffer[si].append(feats[si].detach())
                    if len(st.temporal_buffer[si]) > st.temporal_T:
                        st.temporal_buffer[si].popleft()
            T_buf = len(states[0].temporal_buffer[0])  # type: ignore[index]
            for st in states:
                assert st.temporal_buffer is not None
                assert len(st.temporal_buffer[0]) == T_buf, (
                    "mixed temporal depth in batch"
                )

            # Batch-major feats: per stream cat over T, then cat across streams.
            feats_combined: list[Tensor] = []
            for si in range(3):
                per_stream = [
                    torch.cat(list(st.temporal_buffer[si]), dim=0)  # type: ignore[index]
                    for st in states
                ]  # each (T_buf, C, H, W)
                feats_combined.append(
                    torch.cat(per_stream, dim=0)
                )  # (n*T_buf, C, H, W)

            flows: list[Tensor] | None = None
            if all(len(st.gmc_mat_buffer) >= 2 for st in states):
                flows = []
                for si in range(3):
                    _, _, Hf, Wf = feats_combined[si].shape
                    dev = feats_combined[si].device
                    per_stream_flow = []
                    for st in states:
                        flow_t = [
                            _gmc_matrices_to_flow(
                                st.gmc_mat_buffer, t, T_buf - 1, Hf, Wf, dev
                            ).unsqueeze(0)
                            for t in range(T_buf)
                        ]
                        per_stream_flow.append(
                            torch.cat(flow_t, dim=0)
                        )  # (T_buf,2,Hf,Wf)
                    flows.append(torch.cat(per_stream_flow, dim=0))  # (n*T_buf,2,Hf,Wf)

            head_out = self.mamba_head(
                feats_combined, return_embeddings=want_emb, T=T_buf, flows=flows
            )
            if want_emb:
                cls_preds, reg_preds, emb_preds = head_out
            else:
                cls_preds, reg_preds = head_out
                emb_preds = None
            # last frame per stream: (n*T_buf, ...) -> (n, T_buf, ...) -> [:, -1]
            cls_last = [c.reshape(n, T_buf, *c.shape[1:])[:, -1] for c in cls_preds]
            reg_last = [r.reshape(n, T_buf, *r.shape[1:])[:, -1] for r in reg_preds]
            emb_last = (
                [e.reshape(n, T_buf, *e.shape[1:])[:, -1] for e in emb_preds]
                if emb_preds is not None
                else None
            )
        else:
            feats_combined = [
                torch.cat([f[si] for f in feats_list], dim=0) for si in range(3)
            ]
            head_out = self.mamba_head(feats_combined, return_embeddings=want_emb)
            if want_emb:
                cls_last, reg_last, emb_last = head_out
            else:
                cls_last, reg_last = head_out
                emb_last = None

        results: list[tuple[Tensor, dict[str, Any]]] = []
        for b in range(n):
            cls_b = [c[b : b + 1] for c in cls_last]
            reg_b = [r[b : b + 1] for r in reg_last]
            dets = _postprocess_mamba(
                cls_b,
                reg_b,
                self.stride,
                self.conf_thr,
                self.max_det,
                self.small_p3_max_threshold,
            )
            extra: dict[str, Any] = {}
            if emb_last is not None:
                extra["emb_preds"] = [e[b : b + 1] for e in emb_last]
            results.append((dets, extra))
        return results

    def set_use_cuda_graph(self, enabled: bool) -> None:
        """Enable or disable CUDA Graph capture and replay for the Mamba head."""
        self.mamba_head.set_use_cuda_graph(enabled)

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

        if set(self._trt_feat_cache.keys()) == {"p3", "p4", "p5"}:
            feats = [self._trt_feat_cache[s] for s in ("p3", "p4", "p5")]
            frame_for_resolution = frame_bchw

        elif hasattr(self.teacher, "_gate_layers"):
            cache = {}
            for gl in self.teacher._gate_layers.values():
                cache.update(gl._feat_cache)
            if set(cache.keys()) == {"p3", "p4", "p5"}:
                feats = [cache[s] for s in ("p3", "p4", "p5")]
                frame_for_resolution = frame_bchw
            elif frame_bchw is not None:
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
        elif frame_bchw is not None:
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
        self._last_detections_ptr = detections.data_ptr()
        return detections

    def set_strip_detail_positions(
        self,
        positions: Tensor,
        position_mask: Tensor | None = None,
    ) -> None:
        """Set fixed-budget P3 routes consumed by the next oracle inference."""
        if not self.use_strip_detail:
            raise RuntimeError("loaded checkpoint does not use strip detail")
        self._strip_detail_positions = positions.to(
            device=self._device, dtype=torch.long
        )
        self._strip_detail_position_mask = (
            None
            if position_mask is None
            else position_mask.to(device=self._device, dtype=torch.bool)
        )

    def detect_raw_with_detail(
        self, input_tensor: Tensor, detail_tensor: Tensor
    ) -> Tensor:
        detections, _ = self.forward(
            input_tensor,
            gate_input=None,
            detail_frame=detail_tensor,
        )
        self._last_detections_ptr = detections.data_ptr()
        return detections

    def detect_raw_preprocessed(self, input_tensor: Tensor) -> Tensor:
        if not self.use_whole_graph or self._trt_backbone is None:
            raise RuntimeError(
                "detect_raw_preprocessed requires whole-graph TRT backbone mode."
            )
        detections, _ = self._forward_whole_graph_preprocessed(input_tensor)
        self._last_detections_ptr = detections.data_ptr()
        return detections

    def set_gmc_warp(
        self, warp: Tensor | None, orig_h: int = 0, orig_w: int = 0
    ) -> None:
        if warp is not None:
            self._stream_state.push_gmc(warp)

    def reset_tracker(self) -> None:
        from saccade.perception.tracking import GPUByteTracker  # noqa: E402

        self.tracker = GPUByteTracker(max_objects=2048)
        self._stream_state.reset()


def _gmc_matrices_to_flow(
    mat_buffer: deque[Tensor],
    t_idx: int,
    target_idx: int,
    H: int,
    W: int,
    device: torch.device,
) -> Tensor:
    """Convert deque of GMC affine matrices to (2, H, W) flow for frame t→target."""
    if t_idx >= target_idx:
        return torch.zeros(2, H, W, device=device)
    buf_list = list(mat_buffer)
    cumulative = torch.eye(2, 3, device=device, dtype=torch.float32)
    for i in range(t_idx + 1, target_idx + 1):
        if i >= len(buf_list):
            break
        mat = buf_list[i].to(device).reshape(2, 3)
        m3 = torch.eye(3, device=device, dtype=mat.dtype)
        m3[:2, :] = mat
        cumulative = mat @ m3
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([gx, gy, torch.ones_like(gx)], dim=0).reshape(3, -1)
    warped = cumulative @ coords
    flow = warped - coords[:2]
    return flow.reshape(2, H, W)

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
    temporal_T_override: int | None = None,
    use_cuda_graph: bool = False,
    use_whole_graph: bool = False,
    small_p3_max_threshold: float = 0.0,
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

    mamba_path = str(Path(mamba_ckpt).resolve())
    mamba_state = torch.load(mamba_path, map_location="cpu", weights_only=False)
    mamba_args = mamba_state.get("mamba_args", {})
    temporal_T = 3 if mamba_args.get("use_temporal_mamba", False) else 0
    if temporal_T_override is not None:
        temporal_T = temporal_T_override
    if mamba_args.get("use_temporal_mamba", False):
        if temporal_T > 0 and use_whole_graph:
            print(
                f"🧊 [MambaBuilder] temporal checkpoint, temporal_T={temporal_T}, "
                "but whole-graph forward is single-frame — temporal blocks "
                "BYPASSED (effective T=1)."
            )
        elif temporal_T > 0:
            print(
                f"⚠️ [MambaBuilder] temporal checkpoint: STREAMING temporal path "
                f"ACTIVE (T={temporal_T}, sliding feature window; train/eval "
                "mismatch risk). For pure T=1 eval pass --no-temporal / "
                "temporal_T_override=0."
            )
        else:
            print(
                "🧊 [MambaBuilder] temporal checkpoint, temporal_T=0 — temporal "
                "blocks BYPASSED (pure T=1)."
            )

    model = MambaGatedDetector(
        yolo_pt_path=str(Path(yolo_pt_path).resolve()),
        teacher_ckpt=teacher_path,
        mamba_ckpt=mamba_path,
        cfg=cfg,
        device=device,
        conf_thr=conf_thr,
        max_det=max_det,
        trt_backbone_engine=str(Path(trt_backbone_engine).resolve())
        if trt_backbone_engine
        else "",
        emb_dim=emb_dim,
        jde_proj_ckpt=str(Path(jde_proj_ckpt).resolve()) if jde_proj_ckpt else "",
        temporal_T=temporal_T,
        use_cuda_graph=use_cuda_graph,
        use_whole_graph=use_whole_graph,
        small_p3_max_threshold=small_p3_max_threshold,
    )
    return model.to(device)
