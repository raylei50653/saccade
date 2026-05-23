"""
Option F: MambaDetectionHead GT fine-tuning (Phase 2).

Fine-tunes the distilled MambaDetectionHead on MOT17 ground truth boxes
using v8DetectionLoss (DFL + CIoU + BCE) instead of MSE distillation loss.

Architecture:
    Frozen GatedYOLODetector (backbone + TrackSpatialGate)
        ↓  hooks capture gated FPN features (P3/P4/P5)
    MambaDetectionHead (fine-tuned)
        ↓  per-scale cls_preds / reg_preds
    v8DetectionLoss (standard YOLO detection loss)

Usage:
    uv run train/temporal_yolo/train_mamba_gt.py \
        --data-root datasets/MOT17 \
        --yolo-weights models/yolo/yolo26s.pt \
        --teacher-ckpt runs/gated_det_v1/best.ckpt \
        --mamba-ckpt runs/mamba_distill_v1/best.ckpt \
        --epochs 30 --batch-size 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import random
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
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    state: dict[str, Any], run_dir: Path, epoch: int, is_best: bool = False
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / "latest.ckpt")
    torch.save(state, run_dir / f"epoch_{epoch:04d}.ckpt")
    if is_best:
        torch.save(state, run_dir / "best.ckpt")
    tag = " [BEST]" if is_best else ""
    print(f"  Saved epoch_{epoch:04d}.ckpt{tag}")


def _strip_compiled_keys(sd: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# GT helpers
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


def _build_gate_inputs(
    prev_gt_boxes: list[torch.Tensor],
    gt_ratio: float,
    img_size: int,
    device: torch.device,
) -> list[TrackerGateInput] | None:
    if random.random() >= gt_ratio:
        return None
    return [
        TrackerGateInput.from_boxes_scores(
            boxes.to(device), None, (img_size, img_size), assume_absolute=True
        ).to(device)
        for boxes in prev_gt_boxes
    ]


def _build_preds_dict(
    cls_preds: list[torch.Tensor],
    reg_preds: list[torch.Tensor],
    feats: list[torch.Tensor],
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    cls_cat = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_cat = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
    return {"boxes": reg_cat, "scores": cls_cat, "feats": feats}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Option F: Mamba head GT fine-tuning")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_distill_v1/best.ckpt")
    parser.add_argument("--run-dir", default="runs/mamba_gt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-gate", type=float, default=0.0)
    parser.add_argument("--cls-weight", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--gt-ratio", type=float, default=0.5)
    parser.add_argument("--spatial-reduction", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--freeze-teacher", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = project_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Teacher: frozen GatedYOLODetector
    # ------------------------------------------------------------------
    print("Loading teacher...")
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

    nc = teacher.yolo_model.model[-1].nc

    # Freeze or unfreeze teacher
    if args.freeze_teacher:
        for p in teacher.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Student: MambaDetectionHead
    # ------------------------------------------------------------------
    print("Loading Mamba head...")
    mamba_ckpt = project_root / args.mamba_ckpt
    mamba_state = torch.load(mamba_ckpt, map_location="cpu", weights_only=False)
    mamba_args = mamba_state.get("mamba_args", {})
    mamba_args_default = {
        "d_model": args.d_model,
        "d_state": args.d_state,
        "num_blocks": args.num_blocks,
        "spatial_reduction": args.spatial_reduction,
        "num_classes": nc,
    }
    for k, v in mamba_args_default.items():
        mamba_args.setdefault(k, v)

    mamba = MambaDetectionHead(
        in_channels=(128, 256, 512),
        d_model=mamba_args["d_model"],
        d_state=mamba_args["d_state"],
        num_blocks=mamba_args["num_blocks"],
        num_classes=mamba_args["num_classes"],
        reg_max=1,
        spatial_reduction=mamba_args["spatial_reduction"],
    ).to(device)
    sd = _strip_compiled_keys(mamba_state["student"])
    mamba.load_state_dict(sd, strict=True)
    mamba.train()

    n_params = sum(p.numel() for p in mamba.parameters())
    print(f"  Mamba params: {n_params:,}")

    # FPN feature capture hooks
    fpn_feats: dict[str, torch.Tensor] = {}
    _hooks: list = []
    for scale in ("p3", "p4", "p5"):
        idx = _GATE_LAYER_IDX[scale]

        def _capture(
            _m: nn.Module,
            _i: tuple,
            _o: torch.Tensor,
            s: str = scale,
        ) -> None:
            fpn_feats[s] = _o

        _hooks.append(teacher.yolo_model.model[idx].register_forward_hook(_capture))

    # ------------------------------------------------------------------
    # Loss: v8DetectionLoss
    # ------------------------------------------------------------------
    base_args = (
        dict(teacher.yolo_model.args)
        if isinstance(teacher.yolo_model.args, dict)
        else {}
    )
    base_args.setdefault("box", 7.5)
    base_args.setdefault("cls", args.cls_weight)
    base_args.setdefault("dfl", 1.5)
    teacher.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(teacher.yolo_model)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(mamba.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(
            project_root / args.resume, map_location="cpu", weights_only=False
        )
        sd_resume = _strip_compiled_keys(ckpt["student"])
        mamba.load_state_dict(sd_resume, strict=True)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("[Warn] Optimizer state not loaded — full restart at new LR")
            for pg, lr_val in zip(optimizer.param_groups, [args.lr]):
                pg["lr"] = lr_val
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[Resume] epoch={start_epoch}  best_loss={best_loss:.4f}")

    remaining = max(args.epochs - start_epoch + 1, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    seqs = args.seqs.split(",") if args.seqs else None
    loader = build_mot17_dataloader(
        data_root=project_root / args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        stride=args.clip_len * 2,
        shuffle=True,
        seqs=seqs,
    )
    print(
        f"[MambaGT] {len(loader)} batches/epoch  gt_ratio={args.gt_ratio}  lr={args.lr}"
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs + 1):
        mamba.train()
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.perf_counter()

        for batch in loader:
            frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
            gt_boxes_batch = batch["gt_boxes"]
            B, T = frames.shape[:2]

            optimizer.zero_grad()
            batch_loss = frames.new_zeros(())

            for t in range(T):
                frame_t = frames[:, t]
                gt_t = [gt_boxes_batch[b][t] for b in range(B)]

                gate_inputs = None
                if t > 0:
                    prev_gt = [gt_boxes_batch[b][t - 1] for b in range(B)]
                    gate_inputs = _build_gate_inputs(
                        prev_gt, args.gt_ratio, args.img_size, device
                    )

                fpn_feats.clear()

                # Teacher forward (float32 — Mamba CUDA kernel requires f32)
                _ = teacher(frame_t, gate_input=gate_inputs)

                feats = [fpn_feats[s] for s in ("p3", "p4", "p5")]

                s_cls, s_reg = mamba(feats)
                preds = _build_preds_dict(s_cls, s_reg, feats)
                yolo_batch = _make_yolo_batch(gt_t, args.img_size, device)
                step_loss_vec, _ = criterion(preds, yolo_batch)
                batch_loss = batch_loss + step_loss_vec.sum() / T

            batch_loss.backward()
            nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=1.0)
            optimizer.step()

            step_loss = batch_loss.detach().item()
            if torch.isfinite(batch_loss):
                epoch_loss += step_loss
                n_steps += 1

            if n_steps % 50 == 0:
                print(f"  epoch {epoch:3d}  step {n_steps:4d}  loss={step_loss:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)
        dt = time.perf_counter() - t0
        lr = scheduler.get_last_lr()[0]
        is_best = avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
            f"lr={lr:.2e}  {dt / 60:.1f}min" + (" [BEST]" if is_best else "")
        )

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "student": mamba.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "args": vars(args),
                    "mamba_args": {
                        "d_model": mamba_args["d_model"],
                        "d_state": mamba_args["d_state"],
                        "num_blocks": mamba_args["num_blocks"],
                        "spatial_reduction": mamba_args["spatial_reduction"],
                        "num_classes": nc,
                    },
                },
                run_dir,
                epoch,
                is_best,
            )

    for h in _hooks:
        h.remove()
    print(f"Done. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
