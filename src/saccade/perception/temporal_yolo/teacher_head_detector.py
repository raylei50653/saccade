# mypy: ignore-errors
"""Drop-in detector that runs the *native YOLO detect head* of a gated teacher
through the standard eval tracker pipeline.

Purpose
-------
Matched-baseline control for the "does the Mamba detection head earn its place?"
question. The deployed system (``mamba_whole_graph``) runs the Mamba head on a
gated YOLO backbone; the ONLY way to prove the Mamba head beats — or even matches
— the original architecture is to swap the head for the stock YOLO Detect head
and keep everything else (backbone lineage, tracker, all post-processing)
identical. This wrapper is that swap: same gated backbone (the teacher), original
detect head, fed into the exact same tracker preset.

It exposes only ``detect_raw(input) -> [B, N, 6]`` (xyxy, conf, cls in img_size px
space, pre-NMS top-``max_det`` candidates) plus the handful of attributes/methods
``eval/detection.py`` and ``eval/evaluator.py`` probe on a detector, so it slots
in wherever a ``MambaGatedDetector`` would without touching the tracker.

The teacher deploys gate-free (``gate_input=None`` → identity boost), matching the
Mamba deploy path (null teacher gate, gt-ratio-0 lineage).
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from saccade.perception.temporal_yolo.yolo_gated_detector import (
    GatedDetConfig,
    build_gated_yolo_detector,
)


def _get_topk_index_graph_safe(
    head: Any, scores: Tensor, max_det: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Ultralytics Detect.get_topk_index without the CPU batch arange.

    The default class-aware path indexes a CUDA tensor with ``torch.arange`` on
    CPU, which breaks CUDA graph capture. Keep the exact class-aware ranking
    semantics and cache the batch index on the scores device instead.
    """
    batch_size, anchors, nc = scores.shape
    k = max_det if head.export else min(max_det, anchors)
    if head.agnostic_nms:
        scores, labels = scores.max(dim=-1, keepdim=True)
        scores, indices = scores.topk(k, dim=1)
        labels = labels.gather(1, indices)
        return scores, labels, indices

    ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
    scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(k)

    batch_index = getattr(head, "_saccade_graph_batch_index", None)
    if (
        not isinstance(batch_index, Tensor)
        or batch_index.device != scores.device
        or batch_index.numel() < batch_size
    ):
        batch_index = torch.arange(batch_size, device=scores.device)
        head._saccade_graph_batch_index = batch_index
    idx = ori_index[batch_index[:batch_size, None], index // nc]
    return scores[..., None], (index % nc)[..., None].float(), idx


def _install_graph_safe_class_aware_topk(
    head: Any, device: torch.device | str | None = None, max_batch: int = 1024
) -> None:
    if device is not None:
        batch_index = torch.arange(max_batch, device=device)
        buffers = getattr(head, "_buffers", {})
        if (
            hasattr(head, "register_buffer")
            and "_saccade_graph_batch_index" not in buffers
        ):
            head.register_buffer(
                "_saccade_graph_batch_index", batch_index, persistent=False
            )
        else:
            head._saccade_graph_batch_index = batch_index
    head.get_topk_index = MethodType(_get_topk_index_graph_safe, head)


class TeacherHeadDetector:
    """Native YOLO detect head of a gated teacher, as an eval detector.

    Interface contract consumed by the eval pipeline:
      * ``detect_raw(input_tensor) -> Tensor[B, N, 6]`` — (x1,y1,x2,y2,conf,cls),
        img_size px space, pre-NMS (the tracker does its own NMS / private
        continuation).
      * attributes ``use_whole_graph``/``_trt_backbone``/``use_detail_fusion``/
        ``is_dynamic``/``input_shape``/``img_size``/``device``.
      * ``reset_tracker()`` — no-op (the pipeline owns the real tracker).

    Deliberately does NOT define ``set_gmc_warp``/``set_whole_graph_img_dims`` so
    the pipeline skips detector-level GMC/temporal state (irrelevant to a
    single-frame YOLO head; tracker-level GMC still runs).
    """

    # Static (batch-1 eager) — routes through the plain 640 path.
    use_whole_graph = False
    _trt_backbone = None
    use_detail_fusion = False
    is_dynamic = False
    input_shape = None

    def __init__(
        self,
        teacher_ckpt: str,
        yolo_pt_path: str = "models/yolo/yolo26s.pt",
        img_size: int = 640,
        device: str | torch.device = "cuda",
        max_det: int = 300,
        trt_backbone_engine: str = "",
        whole_graph: bool = False,
    ) -> None:
        self.img_size = int(img_size)
        self.max_det = int(max_det)
        self.device = torch.device(device)

        ckpt_path = Path(teacher_ckpt)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_args = raw.get("args", {})
        scales = tuple(
            s.strip() for s in str(train_args.get("scales", "p3,p4,p5")).split(",")
        )
        cfg = GatedDetConfig(
            scales=scales,
            gate_sigma_scale=float(train_args.get("gate_sigma_scale", 0.5)),
            gate_min_score=float(train_args.get("gate_min_score", 0.5)),
            freeze_backbone=True,
            img_size=self.img_size,
        )
        # The teacher may have been trained from a different base than the default
        # yolo26s.pt; honor the checkpoint's recorded base weights when present so
        # the head/backbone lineage matches.
        base = train_args.get("yolo_weights", yolo_pt_path)
        self.teacher = build_gated_yolo_detector(
            str(base),
            cfg=cfg,
            device=device,
            weights_path=str(ckpt_path),
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Optional TRT backbone: run layers 0-22 (P3/P4/P5) on TensorRT, then the
        # native Detect head (layer 23) in PyTorch. At deploy the gate is identity
        # (gate_input=None), so the TRT features == the gated features the head
        # would see — the swap is numerically faithful (FP16 tolerance) and isolates
        # the "deployed backbone" speed for the native head, mirroring the Mamba
        # head's fpn_backbone_engine path.
        self._trt_backbone = None
        self._detect_head = self.teacher.yolo_model.model[-1]
        if trt_backbone_engine:
            from saccade.perception.temporal_yolo.mamba_gated_detector import (
                TRTYoloBackbone,
            )

            self._trt_backbone = TRTYoloBackbone(
                str(Path(trt_backbone_engine).resolve())
            )
            print(
                f"🚀 [TeacherHead] TRT backbone {trt_backbone_engine} "
                f"channels={self._trt_backbone.output_channels}"
            )

        # Full whole-graph: CUDA-graph-capture backbone (infer_graph) + native
        # Detect head + box-scale into one callable, mirroring MambaGatedDetector's
        # use_whole_graph path. Requires a TRT backbone (graph-capturable infer).
        # The eval passes the FULL-RES frame to detect_raw; the graph interpolates
        # to img_size and scales boxes back via sx/sy (set_whole_graph_img_dims).
        self.use_whole_graph = bool(whole_graph and self._trt_backbone is not None)
        self._wg_sx = torch.ones(1, device=self.device)
        self._wg_sy = torch.ones(1, device=self.device)
        self._wg_img_shape: tuple[int, int] = (0, 0)
        self._wg_graphed: dict[Any, Any] = {}
        self._wg_warm = False
        if whole_graph and self._trt_backbone is None:
            raise ValueError(
                "--teacher-head-whole-graph requires --teacher-head-backbone-engine "
                "(the whole graph captures the TRT backbone's infer_graph)."
            )
        if self.use_whole_graph:
            # The end2end Detect postprocess's default class-aware path does
            # `torch.arange(batch_size)` on CPU to index a GPU tensor, which CUDA
            # graph capture rejects. Patch only this instance so whole-graph keeps
            # the same class-aware top-k semantics as the eager teacher-head path.
            _install_graph_safe_class_aware_topk(self._detect_head, self.device)
            print("🕸️ [TeacherHead] whole-graph runtime ENABLED (backbone+head graph)")

        # The eval pipeline drives the DETECTOR's own association tracker and
        # configures it from the preset (set_params/set_relink_params/…), exactly
        # as it does for the Mamba detector. Own one so the tracker is identical.
        self.tracker: Any = None
        self.reset_tracker()

        epoch = raw.get("epoch")
        print(
            f"🧩 [TeacherHead] native YOLO detect head from {ckpt_path} "
            f"(epoch={epoch}, base={base}, scales={scales}, img={self.img_size})"
        )

    # ------------------------------------------------------------------
    # Whole-graph (CUDA-graphed backbone+head), enabled when use_whole_graph.
    # ------------------------------------------------------------------
    def set_whole_graph_img_dims(self, h_orig: int, w_orig: int) -> None:
        """Set the original-frame size so the graph can scale 640-space boxes back.

        Called by the evaluator per sequence. Changing dims invalidates the
        captured graph (the scale constants sx/sy are baked into it)."""
        if (h_orig, w_orig) == self._wg_img_shape:
            return
        self._wg_img_shape = (h_orig, w_orig)
        self._wg_sx.fill_(w_orig / self.img_size)
        self._wg_sy.fill_(h_orig / self.img_size)
        self._wg_graphed.clear()
        self._wg_warm = False

    def _wg_fn(self, frame: Tensor) -> Tensor:
        # frame: full-res [B,3,H,W]. Stretch to img_size (matches Mamba whole-graph;
        # NOT letterbox), TRT backbone, native Detect head, scale boxes to orig.
        frame_r = F.interpolate(
            frame,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        p3, p4, p5 = self._trt_backbone.infer_graph(frame_r)
        out = self._detect_head([p3, p4, p5])
        dets = out[0] if isinstance(out, (tuple, list)) else out
        dets = dets.clone()  # fixed [B,300,6]; own the buffer inside the graph
        dets[:, :, 0] *= self._wg_sx
        dets[:, :, 2] *= self._wg_sx
        dets[:, :, 1] *= self._wg_sy
        dets[:, :, 3] *= self._wg_sy
        return dets

    def _run_whole_graph(self, frame: Tensor) -> Tensor:
        key = tuple(frame.shape) + self._wg_img_shape
        if key not in self._wg_graphed:
            if not self._wg_warm:
                with torch.no_grad():
                    _ = self._wg_fn(frame.clone())
                    torch.cuda.synchronize()
                self._wg_warm = True
            print(
                f"🕯️ [TeacherHead WholeGraph] capturing shape {tuple(frame.shape)} "
                f"img={self._wg_img_shape}"
            )
            self._wg_graphed[key] = torch.cuda.make_graphed_callables(
                self._wg_fn, (frame.clone(),)
            )
        return self._wg_graphed[key](frame)

    @torch.no_grad()  # no_grad (not inference_mode): CUDA-graph capture needs it
    def detect_raw(self, input_tensor: Tensor) -> Tensor:
        if self.use_whole_graph:
            # Full-res frame in; graph interpolates + scales boxes to orig coords.
            out = self._run_whole_graph(input_tensor)
        elif self._trt_backbone is not None:
            # TRT backbone (layers 0-22) -> native Detect head (layer 23). Gate is
            # identity at deploy, so no gate application is needed on the features.
            p3, p4, p5 = self._trt_backbone.infer(input_tensor)
            out = self._detect_head([p3, p4, p5])
        else:
            # Gate-free deploy (identity boost), matching the Mamba deploy path.
            out = self.teacher(input_tensor, gate_input=None)
        # out[0]: (B, max_det, 6) = (x1, y1, x2, y2, conf, cls), img_size px space.
        dets = out[0] if isinstance(out, (tuple, list)) else out
        # The teacher is an 80-class COCO head; the Mamba head it is being
        # compared against is person-only, and the preset sets
        # track_person_only=false (it assumes the head already emits person only).
        # Keep only person (COCO class 0) so both arms see the same object set.
        # Return [1, N, 6] (batch-1 eager path handles variable N).
        person = dets[..., 5].to(torch.int32) == 0
        b = dets.shape[0]
        kept = [dets[i][person[i]] for i in range(b)]
        n = max((k.shape[0] for k in kept), default=0)
        if n == 0:
            return dets[:, :0, :]
        # Pad to a common N so the [B, N, 6] contract holds for B>1 (B=1 no-op).
        padded = dets.new_zeros((b, n, 6))
        for i, k in enumerate(kept):
            padded[i, : k.shape[0]] = k
        return padded

    def reset_tracker(self) -> None:
        """(Re)create the association tracker, mirroring MambaGatedDetector."""
        from saccade.perception.tracking import GPUByteTracker

        self.tracker = GPUByteTracker(max_objects=2048)

    # Some helpers probe .eval()/.to(); keep them harmless.
    def eval(self) -> "TeacherHeadDetector":
        self.teacher.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "TeacherHeadDetector":
        self.teacher.to(*args, **kwargs)
        return self


def build_teacher_head_detector(
    teacher_ckpt: str,
    yolo_pt_path: str = "models/yolo/yolo26s.pt",
    img_size: int = 640,
    device: str | torch.device = "cuda",
    max_det: int = 300,
    trt_backbone_engine: str = "",
    whole_graph: bool = False,
) -> TeacherHeadDetector:
    return TeacherHeadDetector(
        teacher_ckpt=teacher_ckpt,
        yolo_pt_path=yolo_pt_path,
        img_size=img_size,
        device=device,
        max_det=max_det,
        trt_backbone_engine=trt_backbone_engine,
        whole_graph=whole_graph,
    )
