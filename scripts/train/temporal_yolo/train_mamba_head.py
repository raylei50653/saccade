"""
Option F: MambaDetectionHead distillation training.

Distills YOLO Detect head knowledge into a Mamba SSM detection head.
Teacher: frozen YOLO backbone + pre-trained Detect head (gated_det_v1).
Student: MambaDetectionHead, trained via per-scale MSE loss on cls + reg outputs.

Phase 1 (this script): distillation with MSE, no GT labels needed.
Phase 2 (future): fine-tune on MOT17 ground truth with v8DetectionLoss.

Usage:
    uv run train/temporal_yolo/train_mamba_head.py \
        --data-root datasets/MOT17 \
        --yolo-weights models/yolo/yolo26s.pt \
        --teacher-ckpt runs/gated_det_v1/best.ckpt \
        --epochs 20 --batch-size 8 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

# Import tracker first to avoid libtiff symbol conflict with torchvision
import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
    _GATE_LAYER_IDX,
)
from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint helpers (shared pattern with other training scripts)
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Option F: Mamba head distillation")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--run-dir", default="runs/mamba_distill")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--clip-len", type=int, default=1)
    parser.add_argument("--spatial-reduction", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile teacher (mode=default)"
    )
    parser.add_argument(
        "--accum-steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = project_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Teacher: frozen GatedYOLODetector
    # ------------------------------------------------------------------
    print("Loading teacher...")
    cfg = GatedDetConfig(
        scales=("p3", "p4", "p5"),
        freeze_backbone=True,
        img_size=args.img_size,
    )
    teacher = build_gated_yolo_detector(
        str(project_root / args.yolo_weights),
        cfg=cfg,
        device=device,
        weights_path=str(project_root / args.teacher_ckpt),
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    nc = teacher.yolo_model.model[-1].nc  # number of classes (80)

    # ------------------------------------------------------------------
    # Student: MambaDetectionHead
    # ------------------------------------------------------------------
    print("Building Mamba head...")
    student = MambaDetectionHead(
        in_channels=(128, 256, 512),
        d_model=args.d_model,
        d_state=args.d_state,
        num_blocks=args.num_blocks,
        num_classes=nc,
        reg_max=1,
        spatial_reduction=args.spatial_reduction,
    ).to(device)
    student.train()

    n_params = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} total  {n_trainable:,} trainable")

    # Hooks to capture FPN features from teacher forward pass
    fpn_feats: dict[str, torch.Tensor] = {}
    _hooks = []
    for scale in ("p3", "p4", "p5"):
        idx = _GATE_LAYER_IDX[scale]

        def _capture(_m: nn.Module, _i: Any, _o: torch.Tensor, s: str = scale) -> None:
            fpn_feats[s] = _o

        _hooks.append(teacher.yolo_model.model[idx].register_forward_hook(_capture))

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(
            project_root / args.resume, map_location="cpu", weights_only=False
        )
        sd = _strip_compiled_keys(ckpt["student"])
        student.load_state_dict(sd, strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[Resume] epoch={start_epoch}  best_loss={best_loss:.4f}")

    if args.compile:
        print("[Compile] torch.compile teacher (mode=default) — ~30s cold-start")
        teacher.yolo_model = torch.compile(teacher.yolo_model, mode="default")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loader = build_mot17_dataloader(
        data_root=project_root / args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        stride=args.clip_len * 2,
        shuffle=True,
    )

    # Reference to teacher Detect head's per-scale branches
    detect_head = teacher.yolo_model.model[-1]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining {args.epochs} epochs...")
    accum = args.accum_steps
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
            B, T, _, H, W = frames.shape

            batch_loss = frames.new_zeros(())

            for t in range(T):
                frame = frames[:, t]

                fpn_feats.clear()
                with torch.no_grad():
                    _ = teacher(frame, gate_input=None)

                feats = [fpn_feats[s] for s in ("p3", "p4", "p5")]

                t_cls = [detect_head.cv3[si](feats[si]) for si in range(len(feats))]
                t_reg = [detect_head.cv2[si](feats[si]) for si in range(len(feats))]

                s_cls, s_reg = student(feats)

                for si in range(len(feats)):
                    batch_loss = batch_loss + (
                        nn.functional.mse_loss(s_cls[si], t_cls[si])
                        + nn.functional.mse_loss(s_reg[si], t_reg[si])
                    )

            batch_loss = batch_loss / (T * accum)
            batch_loss.backward()

            if (i + 1) % accum == 0:
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += batch_loss.item() * accum

            if (i + 1) % 50 == 0:
                print(
                    f"  epoch {epoch:3d}  batch {i + 1:4d}  loss={batch_loss.item() * accum:.4f}"
                )

        if len(loader) % accum != 0:
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        dt = time.time() - t0
        print(
            f"  epoch {epoch:3d}  loss={avg_loss:.4f}  time={dt:.0f}s  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        save_checkpoint(
            {
                "epoch": epoch,
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "mamba_args": {
                    "d_model": args.d_model,
                    "d_state": args.d_state,
                    "num_blocks": args.num_blocks,
                    "spatial_reduction": args.spatial_reduction,
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
