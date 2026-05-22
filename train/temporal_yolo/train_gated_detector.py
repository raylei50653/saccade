"""
Option D (revised): GatedYOLODetector training.

Architecture: YOLO26s + TrackSpatialGate at P3/P4/P5.
Loss:         v8DetectionLoss (standard YOLO detection loss with DFL + IoU).
Gate input:   Previous frame GT boxes with probability gt_ratio; else empty.

Key fix vs TemporalYOLOConditioned:
  gt_ratio is CONSTANT throughout training (not annealed to 0).
  This ensures gate always receives GT oracle signal → alpha learns a useful value.
  The YOLO Detect head (shallow CNN) cannot bypass the gate unlike a Transformer decoder.

Usage:
    uv run train/temporal_yolo/train_gated_detector.py \\
        --data-root datasets/MOT17 \\
        --yolo-weights models/yolo/yolo26s.pt \\
        --epochs 30 --gt-ratio 0.5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa: E402
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    GatedYOLODetector,
    build_gated_yolo_detector,
)


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


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    print(f"[Resume] {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = _strip_compiled_keys(state["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)})")
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception:
        print("  [Warn] Optimizer state not loaded")
    return state.get("epoch", 0) + 1  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------
def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """(N, 4) xyxy absolute → (N, 4) cxcywh normalized [0, 1]."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=1)


def _make_yolo_batch(
    gt_boxes_list: list[torch.Tensor],  # B items, each (N, 4) xyxy abs
    img_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build Ultralytics v8DetectionLoss batch dict from GT boxes."""
    batch_idxs, clss, bboxes = [], [], []
    for b, boxes in enumerate(gt_boxes_list):
        if boxes.numel() == 0:
            continue
        n = boxes.shape[0]
        batch_idxs.append(torch.full((n,), float(b)))
        clss.append(torch.zeros(n))  # class 0 = person
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
    prev_gt_boxes: list[
        torch.Tensor
    ],  # B items, each (N, 4) xyxy abs in img_size space
    gt_ratio: float,
    img_size: int,
    device: torch.device,
) -> list[TrackerGateInput] | None:
    """
    With probability gt_ratio, use GT boxes from previous frame as gate input.
    Returns list[TrackerGateInput] (per-sample) or None (skip gate).
    """
    if random.random() >= gt_ratio:
        return None
    return [
        TrackerGateInput.from_boxes_scores(
            boxes.to(device), None, (img_size, img_size), assume_absolute=True
        ).to(device)
        for boxes in prev_gt_boxes
    ]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(  # type: ignore[no-untyped-def]
    model: GatedYOLODetector,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    scaler: Any,  # GradScaler
    device: torch.device,
    img_size: int,
    gt_ratio: float,
    clip_grad: float = 10.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        frames: torch.Tensor = (
            batch["frames"].to(device, dtype=torch.float32) / 255.0
        )  # (B, T, 3, H, W)
        gt_boxes_batch: list[list[torch.Tensor]] = batch["gt_boxes"]
        B, T = frames.shape[:2]

        optimizer.zero_grad()
        batch_loss = frames.new_zeros(())

        for t in range(T):
            frame_t = frames[:, t]  # (B, 3, H, W)
            gt_t = [gt_boxes_batch[b][t] for b in range(B)]  # list[B] of (N,4) xyxy

            # Gate input from t-1 GT boxes (oracle with probability gt_ratio)
            gate_inputs: list[TrackerGateInput] | None = None
            if t > 0:
                prev_gt = [gt_boxes_batch[b][t - 1] for b in range(B)]
                gate_inputs = _build_gate_inputs(prev_gt, gt_ratio, img_size, device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
                out = model(frame_t, gate_input=gate_inputs)
                # Use one2many: computed from non-detached features → gradient flows to gate alphas.
                # one2one uses x_detach internally and cuts the gradient chain.
                preds = out["one2many"]
                yolo_batch = _make_yolo_batch(gt_t, img_size, device)
                # v8DetectionLoss returns (loss_vec[3], loss_detach[3]); sum for backward
                step_loss_vec, _ = criterion(preds, yolo_batch)
                batch_loss = batch_loss + step_loss_vec.sum() / T

        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            clip_grad,
        )
        scaler.step(optimizer)
        scaler.update()

        step_loss = batch_loss.detach().item()
        if torch.isfinite(batch_loss):
            total_loss += step_loss
            n_steps += 1

    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GatedYOLODetector training (Option D revised)"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--run-dir", default="runs/gated_det")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--lr-gate",
        type=float,
        default=1e-3,
        help="Gate alpha LR (higher than YOLO since alpha starts at 0)",
    )
    parser.add_argument(
        "--lr-yolo", type=float, default=1e-5, help="YOLO backbone+detect LR (0=freeze)"
    )
    parser.add_argument(
        "--gt-ratio",
        type=float,
        default=0.5,
        help="Fraction of steps with GT oracle gate (constant, not annealed)",
    )
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scales", default="p3,p4,p5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    seqs = args.seqs.split(",") if args.seqs else None
    scales = tuple(s.strip() for s in args.scales.split(","))

    # ── Model ──
    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=0.5,
        gate_min_score=0.5,
        freeze_backbone=(args.lr_yolo == 0.0),
    )
    yolo_weights = project_root / args.yolo_weights
    model = build_gated_yolo_detector(str(yolo_weights), cfg, device)

    # ── Loss (v8DetectionLoss uses model's anchor/stride info) ──
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace

    # Ultralytics Trainer sets model.args to a namespace with box/cls/dfl hyps;
    # when loading a pre-trained .pt, model.args is a plain dict — patch it here.
    base_args = (
        dict(model.yolo_model.args) if isinstance(model.yolo_model.args, dict) else {}
    )
    base_args.setdefault("box", 7.5)
    base_args.setdefault("cls", 0.5)
    base_args.setdefault("dfl", 1.5)
    model.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(model.yolo_model)

    # ── Optimizer ──
    param_groups = model.parameter_groups(lr_gate=args.lr_gate, lr_yolo=args.lr_yolo)
    n_gate = sum(
        p.numel() for g in param_groups if g["name"] == "gate" for p in g["params"]
    )
    print(f"[GatedDet] gate params: {n_gate}  (alphas × {len(scales)} scales)")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_gate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]

    # ── Resume ──
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer)

    # ── DataLoader ──
    loader = build_mot17_dataloader(
        data_root=args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seqs=seqs,
        preload_to_ram=True,
    )
    print(f"[GatedDet] {len(loader)} batches/epoch  gt_ratio={args.gt_ratio}")
    print(f"[GatedDet] Initial alphas — {model.alpha_summary()}")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.img_size,
            args.gt_ratio,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
            f"alphas=[{model.alpha_summary()}]  "
            f"{elapsed / 60:.1f}min" + (" [BEST]" if is_best else "")
        )

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "args": vars(args),
                    "cfg": cfg,
                },
                run_dir,
                epoch,
                is_best=is_best,
            )

    print(f"\n[Done] best_loss={best_loss:.4f}  ckpt → {run_dir}/best.ckpt")


if __name__ == "__main__":
    main()
