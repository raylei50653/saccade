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
from typing import Any

import torch
from torch import Tensor

from saccade.perception.temporal_yolo.yolo_gated_detector import (
    GatedDetConfig,
    build_gated_yolo_detector,
)


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

    @torch.inference_mode()
    def detect_raw(self, input_tensor: Tensor) -> Tensor:
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
) -> TeacherHeadDetector:
    return TeacherHeadDetector(
        teacher_ckpt=teacher_ckpt,
        yolo_pt_path=yolo_pt_path,
        img_size=img_size,
        device=device,
        max_det=max_det,
    )
