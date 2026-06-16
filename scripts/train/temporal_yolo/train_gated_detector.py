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
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa: E402
from saccade.perception.temporal_yolo.training_utils import (  # noqa: E402
    build_warmup_cosine_scheduler,
    capture_rng_state,
    resolve_training_sequences,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    sha256_file,
    strip_compiled_keys,
)
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    GatedYOLODetector,
    build_gated_yolo_detector,
)


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
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0
    acc_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        frames: torch.Tensor = (
            batch["frames"].to(device, dtype=torch.float32) / 255.0
        )  # (B, T, 3, H, W)
        gt_boxes_batch: list[list[torch.Tensor]] = batch["gt_boxes"]
        B, T = frames.shape[:2]

        batch_loss = frames.new_zeros(())

        for t in range(T):
            frame_t = frames[:, t]  # (B, 3, H, W)
            gt_t = [gt_boxes_batch[b][t] for b in range(B)]  # list[B] of (N,4) xyxy

            # Gate input from t-1 GT boxes (oracle with probability gt_ratio)
            gate_inputs: list[TrackerGateInput] | None = None
            if t > 0:
                prev_gt = [gt_boxes_batch[b][t - 1] for b in range(B)]
                gate_inputs = _build_gate_inputs(prev_gt, gt_ratio, img_size, device)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                out = model(frame_t, gate_input=gate_inputs)
                # Use one2many: computed from non-detached features → gradient flows to gate alphas.
                # one2one uses x_detach internally and cuts the gradient chain.
                preds = out["one2many"]
                yolo_batch = _make_yolo_batch(gt_t, img_size, device)
                # v8DetectionLoss returns (loss_vec[3], loss_detach[3]); sum for backward
                step_loss_vec, _ = criterion(preds, yolo_batch)
                batch_loss = batch_loss + step_loss_vec.sum() / T

        if not torch.isfinite(batch_loss):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_idx + 1}/{len(loader)}"
            )

        if not batch_loss.requires_grad:
            # Frozen-YOLO teacher: the only trainable params are the gate alphas,
            # which enter the graph solely through gate_input. When the gt_ratio
            # draw skips the gate for every frame of the clip, the loss has no
            # grad path. Log it and move on (no-op step). With a trainable YOLO
            # (lr-yolo>0) this never triggers — the weights always carry grad.
            total_loss += batch_loss.detach().item()
            n_steps += 1
            continue

        scaler.scale(batch_loss / accum_steps).backward()
        acc_batches += 1
        if acc_batches == accum_steps:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                clip_grad,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            acc_batches = 0

        total_loss += batch_loss.detach().item()
        n_steps += 1

    if acc_batches > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            clip_grad,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def recalibrate_bn(
    model: GatedYOLODetector,
    loader: Any,
    device: torch.device,
    img_size: int,
    n_batches: int,
) -> None:
    """AdaBN: recompute YOLO BN running stats on MOT, then freeze them.

    Weights and BN affine stay at the COCO prior; only running_mean/var are
    replaced with MOT-domain statistics (momentum=None => cumulative average
    over the seen batches). Isolates BN domain-shift recalibration from weight
    learning, and avoids the train-mode batch-stat instability that NaNs a
    frozen COCO head on small batches.
    """
    bns = [
        m
        for m in model.yolo_model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    saved_momentum = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
        m.train()
    seen = 0
    for batch in loader:
        frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
        for t in range(frames.shape[1]):
            model(frames[:, t], gate_input=None)
        seen += 1
        if seen >= n_batches:
            break
    for m, mom in zip(bns, saved_momentum, strict=True):
        m.eval()
        m.momentum = mom
    print(f"[GatedDet] AdaBN recalibrated {len(bns)} BN layers over {seen} batches")


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "diff_status": "dirty" if run("status", "--porcelain") else "clean",
    }


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
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=0,
        help="Clip start stride. 0 uses clip-len.",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--lr-gate",
        type=float,
        default=1e-3,
        help="Gate alpha LR (higher than YOLO since alpha starts at 0)",
    )
    parser.add_argument(
        "--lr-yolo",
        type=float,
        default=0.0,
        help="YOLO backbone+detect LR. 0 freezes weights and BatchNorm stats.",
    )
    parser.add_argument(
        "--adapt-bn",
        action="store_true",
        help="With --lr-yolo 0: AdaBN — recompute BN running stats on MOT "
        "(forward-only), then freeze them, while weights + BN affine stay frozen. "
        "Isolates BN domain-shift recalibration from weight learning.",
    )
    parser.add_argument(
        "--adapt-bn-batches",
        type=int,
        default=500,
        help="Number of batches for the AdaBN recalibration pass.",
    )
    parser.add_argument(
        "--gt-ratio",
        type=float,
        default=0.5,
        help="Fraction of steps with GT oracle gate (constant, not annealed)",
    )
    parser.add_argument("--seqs", default="")
    parser.add_argument("--holdout-seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--best-by",
        choices=("none", "train-loss"),
        default="none",
        help="train-loss is diagnostic only; deployment selection is external.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scales", default="p3,p4,p5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--clip-grad", type=float, default=10.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="Load model/epoch from a legacy checkpoint and create a new optimizer "
        "and LR schedule.",
    )
    parser.add_argument("--protocol-revision", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model, split, resume state, and provenance without training.",
    )
    args = parser.parse_args()

    if args.clip_stride < 0:
        parser.error("--clip-stride must be >= 0")
    if args.accum_steps < 1:
        parser.error("--accum-steps must be >= 1")
    if not 0.0 <= args.gt_ratio <= 1.0:
        parser.error("--gt-ratio must be between 0 and 1")
    if args.resume_reset_optimizer and not args.resume:
        parser.error("--resume-reset-optimizer requires --resume")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = project_root / args.run_dir
    data_root = project_root / args.data_root
    seqs, holdout_seqs = resolve_training_sequences(
        data_root, args.seqs, args.holdout_seqs
    )
    clip_stride = args.clip_stride or args.clip_len
    scales = tuple(s.strip() for s in args.scales.split(","))

    # ── Model ──
    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=0.5,
        gate_min_score=0.5,
        freeze_backbone=(args.lr_yolo == 0.0),
        img_size=args.img_size,
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
    yolo_frozen = args.lr_yolo == 0.0
    bn_frozen = yolo_frozen and not args.adapt_bn
    if not yolo_frozen:
        yolo_state = f"TRAINABLE (lr={args.lr_yolo:.2e}, BatchNorm train mode)"
    elif args.adapt_bn:
        yolo_state = (
            "FROZEN weights + affine, BatchNorm stats AdaBN-recalibrated to MOT"
        )
    else:
        yolo_state = "FROZEN (weights + BatchNorm stats)"
    print("[GatedDet] YOLO " + yolo_state)

    # ── Resume ──
    start_epoch = 1
    best_loss = float("inf")
    resume_ckpt: dict[str, Any] | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(
            strip_compiled_keys(resume_ckpt["model"]),
            strict=True,
        )
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        if not args.resume_reset_optimizer:
            saved_args = resume_ckpt.get("args", {})
            exact_fields = (
                "epochs",
                "batch_size",
                "clip_len",
                "clip_stride",
                "lr_gate",
                "lr_yolo",
                "gt_ratio",
                "seed",
                "warmup_epochs",
                "accum_steps",
            )
            changed = {
                field: (saved_args[field], getattr(args, field))
                for field in exact_fields
                if field in saved_args and saved_args[field] != getattr(args, field)
            }
            if changed:
                raise ValueError(
                    f"Exact resume argument mismatch: {changed}. Use "
                    "--resume-reset-optimizer to start a new schedule."
                )
            saved_provenance = resume_ckpt.get("provenance", {})
            expected_train = seqs or []
            saved_train = saved_provenance.get("training_sequences")
            saved_selection = saved_provenance.get("selection_sequences")
            if saved_train is not None and saved_train != expected_train:
                raise ValueError(
                    "Exact resume training sequence mismatch. Use the original "
                    "--seqs/--holdout-seqs."
                )
            if saved_selection is not None and saved_selection != holdout_seqs:
                raise ValueError(
                    "Exact resume selection sequence mismatch. Use the original "
                    "--holdout-seqs."
                )
            best_loss = resume_ckpt.get("best_loss", float("inf"))

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    schedule_epochs = (
        max(args.epochs - start_epoch + 1, 1)
        if args.resume_reset_optimizer
        else args.epochs
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=schedule_epochs,
        warmup_epochs=min(args.warmup_epochs, schedule_epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if resume_ckpt is not None:
        if args.resume_reset_optimizer:
            print(
                f"[Resume] model epoch={start_epoch - 1}; reset optimizer/scheduler "
                f"for {schedule_epochs} remaining epoch(s)"
            )
        else:
            missing = [
                key
                for key in ("optimizer", "scheduler", "scaler", "rng_state")
                if key not in resume_ckpt
            ]
            if missing:
                raise ValueError(
                    f"Exact resume missing {missing}. Use --resume-reset-optimizer "
                    "for legacy checkpoints."
                )
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            scaler.load_state_dict(resume_ckpt["scaler"])
            print(
                f"[Resume] exact epoch={start_epoch} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # ── DataLoader ──
    loader = build_mot17_dataloader(
        data_root=data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=clip_stride,
        seqs=seqs,
        preload_to_ram=not args.dry_run,
        seed=args.seed,
    )
    data_generator = loader.generator
    if resume_ckpt is not None and not args.resume_reset_optimizer:
        restore_rng_state(resume_ckpt["rng_state"], data_generator)

    print(
        f"[GatedDet] {len(loader)} batches/epoch  gt_ratio={args.gt_ratio}  "
        f"seed={args.seed}  clip_stride={clip_stride}"
    )
    if holdout_seqs:
        print(
            f"[Split] holdout={','.join(holdout_seqs)} "
            "(external detector/tracking selection)"
        )
    print(f"[GatedDet] Initial alphas — {model.alpha_summary()}")

    git_state = _git_state()
    provenance = {
        "protocol_revision": args.protocol_revision,
        "command": shlex.join(sys.argv),
        "git_commit": git_state["commit"],
        "git_diff_status": git_state["diff_status"],
        "training_sequences": seqs or list(loader.dataset.sequences),  # type: ignore[attr-defined]
        "selection_sequences": holdout_seqs,
        "base_yolo_path": args.yolo_weights,
        "base_yolo_sha256": sha256_file(yolo_weights),
        "yolo_weights_frozen": yolo_frozen,
        "yolo_bn_frozen": bn_frozen,
        "parent_checkpoint_path": args.resume,
        "parent_checkpoint_sha256": (sha256_file(resume_path) if args.resume else ""),
    }
    if args.dry_run:
        print(f"[DryRun] provenance={provenance}")
        return

    # ── AdaBN recalibration (BN-only domain adaptation) ──
    # Recompute BN running stats on MOT with frozen weights, then freeze them.
    # On resume the recalibrated stats already live in the checkpoint.
    if args.adapt_bn and start_epoch == 1:
        recalibrate_bn(model, loader, device, args.img_size, args.adapt_bn_batches)

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        epoch_lrs = [group["lr"] for group in optimizer.param_groups]
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.img_size,
            args.gt_ratio,
            clip_grad=args.clip_grad,
            accum_steps=args.accum_steps,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0
        is_best = args.best_by == "train-loss" and loss < best_loss
        best_loss = min(best_loss, loss)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
            f"lr={epoch_lrs[0]:.2e}  "
            f"alphas=[{model.alpha_summary()}]  "
            f"{elapsed / 60:.1f}min" + (" [BEST]" if is_best else "")
        )

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": capture_rng_state(data_generator),
                    "best_loss": best_loss,
                    "epoch_loss": loss,
                    "epoch_lrs": epoch_lrs,
                    "selection": {
                        "best_by": args.best_by,
                        "holdout_seqs": holdout_seqs,
                        "status": (
                            "diagnostic_train_loss"
                            if args.best_by == "train-loss"
                            else "candidate_requires_external_selection"
                        ),
                    },
                    "provenance": provenance,
                    "args": vars(args),
                    "cfg": cfg,
                },
                run_dir,
                epoch,
                is_best=is_best,
            )

    print(f"\n[Done] min_train_loss={best_loss:.4f}  latest={run_dir / 'latest.ckpt'}")


if __name__ == "__main__":
    main()
