"""
Option C：YOLO26s + Cross-Attention Decoder 聯合訓練

複用：
  saccade.perception.temporal_yolo.yolo_joint   — TemporalYOLOJoint, JointConfig
  saccade.perception.temporal_yolo.dataset       — build_mot17_dataloader
  saccade.perception.temporal_yolo.loss          — TemporalTrackingLoss

使用方式：
    # 預設設定（P5-only, lr=1e-4, backbone_lr=1e-5）
    uv run train/temporal_yolo/train_joint.py --data-root /path/to/MOT17

    # 多尺度 P3+P4+P5
    uv run train/temporal_yolo/train_joint.py --data-root /path/to/MOT17 --scales p3,p4,p5

    # 從 checkpoint 續訓
    uv run train/temporal_yolo/train_joint.py --data-root /path/to/MOT17 --resume runs/joint/latest.ckpt

    # 快速驗證（單序列）
    uv run train/temporal_yolo/train_joint.py --data-root /path/to/MOT17 --seqs MOT17-02-SDP
"""
# status: archive-candidate

from __future__ import annotations
import argparse
import time
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from saccade.perception.temporal_yolo.yolo_joint import (
    JointConfig,
    TemporalYOLOJoint,
    build_temporal_yolo_joint,
)
from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader
from saccade.perception.temporal_yolo.dataset_joint import build_joint_dataloader
from saccade.perception.temporal_yolo.loss import TemporalTrackingLoss


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


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    print(f"[Resume] {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Use strict=False to allow loading from p5-only model to p3,p4,p5 model
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected if architecture changed)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # Only load optimizer if architecture matches, otherwise it might fail due to param size mismatch
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception as e:
        print(f"  Warning: Could not load optimizer state (architecture change?): {e}")

    epoch = state["epoch"] + 1
    print(
        f"  Resumed from epoch {state['epoch']}  best_loss={state.get('best_loss', float('inf')):.4f}"
    )
    return epoch  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(  # type: ignore[no-untyped-def]
    model: TemporalYOLOJoint,
    loader,
    criterion: TemporalTrackingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
) -> float:
    model.train()
    # BatchNorm 在 YOLO backbone 訓練時保持 train mode；
    # 若 freeze_backbone=True 則 eval mode 由 YOLOFeaturePyramid 內部控制。

    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        frames = (
            batch["frames"].to(device, non_blocking=True).float() / 255.0
        )  # (B, T, 3, H, W)
        gt_boxes_batch = batch["gt_boxes"]  # list[list[Tensor]]

        B, T, _, H, W = frames.shape
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
            batch_loss = frames.new_zeros(())
            model.reset_sequence()
            prev_queries = None

            pred_boxes_b_t: list[list[torch.Tensor]] = [[] for _ in range(B)]
            pred_scores_b_t: list[list[torch.Tensor]] = [[] for _ in range(B)]

            # 1. Batched Forward Pass
            # We don't detach prev_queries to allow BPTT across the clip
            for t in range(T):
                out = model(frames[:, t], prev_queries)
                prev_queries = out["updated_queries"]

                for b in range(B):
                    pred_boxes_b_t[b].append(out["boxes"][b])
                    pred_scores_b_t[b].append(out["scores"][b])

            # 2. Batched Loss Computation
            losses = criterion(pred_boxes_b_t, pred_scores_b_t, gt_boxes_batch, (H, W))
            batch_loss = losses["loss_total"]

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += batch_loss.item()
        if (i + 1) % log_interval == 0 or i == n_batches - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_batches - i - 1) / max(rate, 1e-6)

            l1 = losses["loss_l1"].item()
            giou = losses["loss_giou"].item()
            bce = losses["loss_bce"].item()
            n_m = int(losses["n_matched"].item())

            print(
                f"  [{epoch:3d}] {i + 1:4d}/{n_batches}"
                f"  loss={batch_loss.item():.3f} (L1:{l1:.2f} G:{giou:.2f} B:{bce:.2f})"
                f"  n_m={n_m:<3d}"
                f"  {rate:.1f}it/s  ETA {eta / 60:.1f}m"
            )
            # print timers if t_stats is defined
            if "t_stats" in dir():
                print(
                    "    TIMING: "
                    + " | ".join(
                        f"{k}: {v / i * 1000:.1f}ms"
                        for k, v in t_stats.items()  # type: ignore[name-defined]  # noqa: F821
                    )
                )

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option C: Joint YOLO + Decoder training"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--run-dir", default="runs/joint")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder LR")
    parser.add_argument(
        "--lr-backbone", type=float, default=1e-5, help="Backbone LR (joint)"
    )
    parser.add_argument("--scales", default="p5", help="FPN scales: p5 or p3,p4,p5")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--checkpointing", action="store_true", help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--no-self-attn", action="store_true", help="Disable query self-attention"
    )
    parser.add_argument(
        "--datasets", default="mot17", help="Datasets to use: mot17,mot20,dancetrack"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    seqs = args.seqs.split(",") if args.seqs else None
    scales = tuple(s.strip() for s in args.scales.split(","))

    cfg = JointConfig(
        embed_dim=256,
        num_heads=8,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        ffn_dim=1024,
        score_threshold=0.3,
        scales=scales,
        freeze_backbone=False,
        use_checkpointing=args.checkpointing,
        use_self_attention=not args.no_self_attn,
    )
    model = build_temporal_yolo_joint(args.yolo_weights, cfg, device)
    param_groups = model.parameter_groups(
        lr_backbone=args.lr_backbone, lr_decoder=args.lr
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = TemporalTrackingLoss()

    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer)

    # Multi-dataset support
    dataset_names = [s.strip().lower() for s in args.datasets.split(",")]
    if len(dataset_names) == 1 and dataset_names[0] == "mot17":
        loader = build_mot17_dataloader(
            data_root=args.data_root,
            clip_len=args.clip_len,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seqs=seqs,
        )
    else:
        dataset_configs = []
        for name in dataset_names:
            if name == "mot17":
                dataset_configs.append(
                    {"name": "MOT17", "root": Path(args.data_root), "type": "mot"}
                )
            elif name == "mot20":
                dataset_configs.append(
                    {
                        "name": "MOT20",
                        "root": Path(args.data_root).parent / "MOT20" / "MOT20",
                        "type": "mot",
                    }
                )
            elif name == "dancetrack":
                dataset_configs.append(
                    {
                        "name": "DanceTrack",
                        "root": Path(args.data_root).parent / "DanceTrack",
                        "type": "dancetrack",
                    }
                )

        loader = build_joint_dataloader(
            dataset_configs=dataset_configs,
            clip_len=args.clip_len,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stride=5,
            shuffle=True,
            preload_to_ram=True,
        )

    print(f"\n{'=' * 60}")
    print("  Option D — Joint Training (Multi-dataset + Self-Attn)")
    print(f"  Epochs: {start_epoch}→{args.epochs}  Device: {device}  BF16: ON")
    print(f"  Scales: {scales}  LR backbone={args.lr_backbone} decoder={args.lr}")
    print(f"  Datasets: {dataset_names}  Self-Attn: {not args.no_self_attn}")
    print(f"  Run dir: {run_dir}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──")
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch)
        scheduler.step()

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "best_loss": best_loss,
                    "args": vars(args),
                },
                run_dir,
                epoch,
                is_best,
            )
        print(f"  avg={avg_loss:.4f}  best={best_loss:.4f}")

    print(f"\nDone. Best loss: {best_loss:.4f}  →  {run_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
