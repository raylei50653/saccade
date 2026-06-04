#!/usr/bin/env python3
"""
Mamba head training latency profiler.

Measures per-stage wall-clock time for both Phase 1 (distillation) and
Phase 2 (GT fine-tuning), with optional torch.compile on the teacher.

Stages:
  data    – DataLoader next() → raw batch
  device  – .to(device) + /255.0 + CUDA sync
  teacher – Teacher forward (YOLO26s backbone + FPN)
  detect  – Teacher detect head (cv2/cv3) on captured FPN features  [Phase 1 only]
  mamba   – MambaDetectionHead forward
  loss    – Loss computation
  bwd     – .backward() + clip_grad_norm
  step    – optimizer.step()

Usage:
  uv run scripts/benchmarks/mamba_train_prof.py \
      --data-root datasets/MOT17 \
      --yolo-weights models/yolo/yolo26s.pt \
      --teacher-ckpt runs/gated_det_v1/best.ckpt \
      --phase 1 --batch-size 8

  uv run scripts/benchmarks/mamba_train_prof.py \
      --data-root datasets/MOT17 \
      --yolo-weights models/yolo/yolo26s.pt \
      --teacher-ckpt runs/gated_det_v1/best.ckpt \
      --mamba-ckpt runs/mamba_distill_v1/best.ckpt \
      --phase 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa: E402
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
    _GATE_LAYER_IDX,
)
from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead  # noqa: E402

SEP = "─" * 72


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Phase 1 helpers (from train_mamba_head.py)
# ---------------------------------------------------------------------------
def _profile_phase1(
    teacher: Any,
    student: MambaDetectionHead,
    loader: Any,
    device: torch.device,
    n_warmup: int,
    n_iters: int,
    compile_teacher: bool,
) -> dict[str, float]:
    detect_head = teacher.yolo_model.model[-1]

    fpn_feats: dict[str, torch.Tensor] = {}
    _hooks = []
    for scale in ("p3", "p4", "p5"):
        idx = _GATE_LAYER_IDX[scale]

        def _capture(_m: nn.Module, _i: Any, _o: torch.Tensor, s: str = scale) -> None:
            fpn_feats[s] = _o

        _hooks.append(teacher.yolo_model.model[idx].register_forward_hook(_capture))

    if compile_teacher:
        teacher.yolo_model = torch.compile(teacher.yolo_model, mode="default")
        _sync()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    times: dict[str, float] = {
        "data": 0.0,
        "device": 0.0,
        "teacher": 0.0,
        "detect": 0.0,
        "mamba": 0.0,
        "loss": 0.0,
        "bwd": 0.0,
        "step": 0.0,
    }

    data_iter = iter(loader)
    print(f"  Warmup {n_warmup} + measure {n_iters} iters ...")

    for i in range(n_warmup + n_iters):
        # ── data ──
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t1 = time.perf_counter()

        # ── device ──
        frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
        _sync()
        t2 = time.perf_counter()
        B, T, _, H, W = frames.shape

        optimizer.zero_grad(set_to_none=True)

        teacher_time = 0.0
        detect_time = 0.0
        mamba_time = 0.0
        loss_time = 0.0

        batch_loss = frames.new_zeros(())

        for ti in range(T):
            frame = frames[:, ti]
            fpn_feats.clear()

            # ── teacher ──
            ta = time.perf_counter()
            with torch.no_grad():
                _ = teacher(frame, gate_input=None)
            _sync()
            tb = time.perf_counter()

            feats = [fpn_feats[s] for s in ("p3", "p4", "p5")]

            # ── detect head ──
            tc = time.perf_counter()
            t_cls = [detect_head.cv3[si](feats[si]) for si in range(len(feats))]
            t_reg = [detect_head.cv2[si](feats[si]) for si in range(len(feats))]
            _sync()
            td = time.perf_counter()

            # ── mamba ──
            te = time.perf_counter()
            s_cls, s_reg = student(feats)
            _sync()
            tf = time.perf_counter()

            # ── loss ──
            tg = time.perf_counter()
            for si in range(len(feats)):
                batch_loss = batch_loss + (
                    nn.functional.mse_loss(s_cls[si], t_cls[si])
                    + nn.functional.mse_loss(s_reg[si], t_reg[si])
                )
            _sync()
            th = time.perf_counter()

            teacher_time += tb - ta
            detect_time += td - tc
            mamba_time += tf - te
            loss_time += th - tg

        batch_loss = batch_loss / T

        # ── backward ──
        t3 = time.perf_counter()
        batch_loss.backward()
        _sync()
        t4 = time.perf_counter()

        # ── step ──
        t5 = time.perf_counter()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        _sync()
        t6 = time.perf_counter()

        if i >= n_warmup:
            times["data"] += t1 - t0
            times["device"] += t2 - t1
            times["teacher"] += teacher_time
            times["detect"] += detect_time
            times["mamba"] += mamba_time
            times["loss"] += loss_time
            times["bwd"] += t4 - t3
            times["step"] += t6 - t5

    for h in _hooks:
        h.remove()

    return {k: v / n_iters * 1000 for k, v in times.items()}


# ---------------------------------------------------------------------------
# Phase 2 helpers (from train_mamba_gt.py)
# ---------------------------------------------------------------------------
def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=1)


def _make_yolo_batch(
    gt_boxes_list: list[torch.Tensor],
    img_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_idxs, clss, bboxes = [], [], []
    for b, boxes in enumerate(gt_boxes_list):
        if boxes.numel() == 0:
            continue
        n = boxes.shape[0]
        batch_idxs.append(torch.full((n,), float(b)))
        clss.append(torch.zeros(n))
        bboxes.append(_xyxy_to_cxcywh_norm(boxes, img_size))
    if not batch_idxs:
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros(0, device=device),
            "bboxes": torch.zeros(0, 4, device=device),
        }
    return {
        "batch_idx": torch.cat(batch_idxs).to(device),
        "cls": torch.cat(clss).to(device),
        "bboxes": torch.cat(bboxes).to(device),
    }


def _build_preds_dict(
    cls_preds: list[torch.Tensor],
    reg_preds: list[torch.Tensor],
    feats: list[torch.Tensor],
) -> dict[str, Any]:
    cls_cat = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_cat = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
    return {"boxes": reg_cat, "scores": cls_cat, "feats": feats}


def _profile_phase2(
    teacher: Any,
    mamba: MambaDetectionHead,
    loader: Any,
    device: torch.device,
    img_size: int,
    gt_ratio: float,
    n_warmup: int,
    n_iters: int,
    compile_teacher: bool,
) -> dict[str, float]:
    from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402

    fpn_feats: dict[str, torch.Tensor] = {}
    _hooks = []
    for scale in ("p3", "p4", "p5"):
        idx = _GATE_LAYER_IDX[scale]

        def _capture(_m: nn.Module, _i: Any, _o: torch.Tensor, s: str = scale) -> None:
            fpn_feats[s] = _o

        _hooks.append(teacher.yolo_model.model[idx].register_forward_hook(_capture))

    if compile_teacher:
        teacher.yolo_model = torch.compile(teacher.yolo_model, mode="default")
        _sync()

    base_args = (
        dict(teacher.yolo_model.args)
        if isinstance(teacher.yolo_model.args, dict)
        else {}
    )
    base_args.setdefault("box", 7.5)
    base_args.setdefault("cls", 0.5)
    base_args.setdefault("dfl", 1.5)
    teacher.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(teacher.yolo_model)

    optimizer = torch.optim.AdamW(mamba.parameters(), lr=1e-3)

    times: dict[str, float] = {
        "data": 0.0,
        "device": 0.0,
        "teacher": 0.0,
        "mamba": 0.0,
        "loss": 0.0,
        "bwd": 0.0,
        "step": 0.0,
    }
    # Phase 2 has no separate detect-head stage

    data_iter = iter(loader)
    print(f"  Warmup {n_warmup} + measure {n_iters} iters ...")

    import random

    for i in range(n_warmup + n_iters):
        # ── data ──
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t1 = time.perf_counter()

        # ── device ──
        frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
        gt_boxes_batch = batch["gt_boxes"]
        _sync()
        t2 = time.perf_counter()
        B, T = frames.shape[:2]

        optimizer.zero_grad(set_to_none=True)

        teacher_time = 0.0
        mamba_time = 0.0
        loss_time = 0.0

        batch_loss = frames.new_zeros(())

        for ti in range(T):
            frame_t = frames[:, ti]
            gt_t = [gt_boxes_batch[b][ti] for b in range(B)]

            gate_inputs = None
            if ti > 0 and random.random() < gt_ratio:
                prev_gt = [gt_boxes_batch[b][ti - 1] for b in range(B)]
                gate_inputs = [
                    TrackerGateInput.from_boxes_scores(
                        boxes.to(device),
                        None,
                        (img_size, img_size),
                        assume_absolute=True,
                    ).to(device)
                    for boxes in prev_gt
                ]

            fpn_feats.clear()

            # ── teacher ──
            ta = time.perf_counter()
            with torch.no_grad():
                _ = teacher(frame_t, gate_input=gate_inputs)
            _sync()
            tb = time.perf_counter()

            feats = [fpn_feats[s] for s in ("p3", "p4", "p5")]

            # ── mamba ──
            tc = time.perf_counter()
            s_cls, s_reg = mamba(feats)
            preds = _build_preds_dict(s_cls, s_reg, feats)
            _sync()
            td = time.perf_counter()

            # ── loss ──
            te = time.perf_counter()
            yolo_batch = _make_yolo_batch(gt_t, img_size, device)
            step_loss_vec, _ = criterion(preds, yolo_batch)
            batch_loss = batch_loss + step_loss_vec.sum()
            _sync()
            tf = time.perf_counter()

            teacher_time += tb - ta
            mamba_time += td - tc
            loss_time += tf - te

        batch_loss = batch_loss / T

        # ── backward ──
        t3 = time.perf_counter()
        batch_loss.backward()
        _sync()
        t4 = time.perf_counter()

        # ── step ──
        t5 = time.perf_counter()
        nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=1.0)
        optimizer.step()
        _sync()
        t6 = time.perf_counter()

        if i >= n_warmup:
            times["data"] += t1 - t0
            times["device"] += t2 - t1
            times["teacher"] += teacher_time
            times["mamba"] += mamba_time
            times["loss"] += loss_time
            times["bwd"] += t4 - t3
            times["step"] += t6 - t5

    for h in _hooks:
        h.remove()

    return {k: v / n_iters * 1000 for k, v in times.items()}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _print_table(
    label: str,
    avg: dict[str, float],
    phase: int,
    compile_teacher: bool,
    clip_len: int,
    batch_size: int,
) -> None:
    compile_tag = " + compile" if compile_teacher else ""
    print(f"\n  Phase {phase}{compile_tag}  B={batch_size} T={clip_len}")
    print(f"  {SEP}")

    # Compute per-frame teacher cost
    n_frames = batch_size * clip_len
    teacher_ms = avg.get("teacher", 0)
    teacher_per_frame = teacher_ms / n_frames

    rows = [
        (k, v)
        for k, v in avg.items()
        if k not in ("teacher", "detect", "mamba", "loss")
    ]
    rows.extend(
        [
            ("teacher", teacher_ms),
            ("  └ per-frame", teacher_per_frame),
        ]
    )
    if phase == 1:
        rows.append(("detect", avg.get("detect", 0)))
    rows.append(("mamba", avg.get("mamba", 0)))
    rows.append(("loss", avg.get("loss", 0)))

    total = sum(v for k, v in rows if not k.startswith("  ") and k != label)
    label_w = max(len(r[0]) for r in rows) + 2

    for name, ms in rows:
        if name.startswith("  "):
            print(f"    {name:<{label_w - 2}} {ms:7.2f} ms")
        else:
            pct = ms / total * 100 if total > 0 else 0
            bar_w = 24
            filled = round(pct / 100 * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            print(f"    {name:<{label_w}} {ms:7.2f} ms ({pct:5.1f}%)  {bar}")

    print(f"    {'total':<{label_w}} {total:7.2f} ms")
    it_s = 1000 / total if total > 0 else 0
    print(f"    {'throughput':<{label_w}} {it_s:7.2f} it/s  ({n_frames} fwd/iter)")
    print(f"  {SEP}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba head training latency profiler")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_distill_v1/best.ckpt")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--spatial-reduction", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--seqs", default="MOT17-04-SDP")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="torch.compile teacher")
    parser.add_argument("--gt-ratio", type=float, default=0.5)
    args = parser.parse_args()

    if args.clip_len is None:
        args.clip_len = 1 if args.phase == 1 else 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Teacher ──
    print(f"\n  Loading teacher from {args.teacher_ckpt} ...")
    teacher_ckpt = project_root / args.teacher_ckpt
    teacher_raw = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    train_args = teacher_raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))

    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
        gate_min_score=train_args.get("gate_min_score", 0.5),
        freeze_backbone=True,
        img_size=args.img_size,
    )
    teacher = build_gated_yolo_detector(
        str(project_root / args.yolo_weights),
        cfg=cfg,
        device=device,
        weights_path=str(teacher_ckpt),
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    nc = teacher.yolo_model.model[-1].nc
    print(f"  Teacher ready  nc={nc}  scales={scales}")

    # ── Student (Mamba head) ──
    mamba = MambaDetectionHead(
        in_channels=(128, 256, 512),
        d_model=args.d_model,
        d_state=args.d_state,
        num_blocks=args.num_blocks,
        num_classes=nc,
        reg_max=1,
        spatial_reduction=args.spatial_reduction,
    ).to(device)
    mamba.train()

    if args.phase == 2:
        mamba_ckpt = project_root / args.mamba_ckpt
        if mamba_ckpt.exists():
            sd = torch.load(mamba_ckpt, map_location="cpu", weights_only=False)
            student_sd = sd.get("student", sd)
            student_sd = {
                k.replace("._orig_mod.", "."): v for k, v in student_sd.items()
            }
            mamba.load_state_dict(student_sd, strict=False)
        else:
            print(f"  [Warn] Mamba ckpt not found: {mamba_ckpt} — using random init")

    n_params = sum(p.numel() for p in mamba.parameters())
    print(
        f"  Mamba head: {n_params:,} params  "
        f"d_model={args.d_model} d_state={args.d_state} "
        f"blocks={args.num_blocks} sr={args.spatial_reduction}"
    )

    # ── DataLoader ──
    loader = build_mot17_dataloader(
        data_root=project_root / args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        stride=args.clip_len * 2,
        shuffle=False,
        seqs=args.seqs.split(",") if args.seqs else None,
    )
    print(f"  DataLoader: {len(loader)} batches")

    # ── Profile ──
    if args.phase == 1:
        avg = _profile_phase1(
            teacher,
            mamba,
            loader,
            device,
            args.warmup,
            args.iters,
            args.compile,
        )
    else:
        avg = _profile_phase2(
            teacher,
            mamba,
            loader,
            device,
            args.img_size,
            args.gt_ratio,
            args.warmup,
            args.iters,
            args.compile,
        )

    _print_table(
        "Mamba Training", avg, args.phase, args.compile, args.clip_len, args.batch_size
    )


if __name__ == "__main__":
    main()
