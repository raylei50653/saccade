"""
Option D：Track-Conditioned YOLO Neck 訓練

架構：Track Queries 的歷史位置以 Gaussian heatmap 注入 YOLO FPN 輸出，
讓 decoder 在已知目標存在的區域取到更強的特徵。

複用：
  saccade.perception.temporal_yolo.yolo_joint       — YOLOFeaturePyramid, FPNSequenceProjection
  saccade.perception.temporal_yolo.yolo_conditioned — TemporalYOLOConditioned（待實作）
  saccade.perception.temporal_yolo.dataset           — build_mot17_dataloader
  saccade.perception.temporal_yolo.loss              — TemporalTrackingLoss

訓練策略（兩階段）：
  Phase 1  凍結 backbone + decoder，只訓 TrackSpatialGate.alpha（5~10 epochs）
           heatmap 用 GT box 渲染（oracle，確保 gate 學到有意義的值）
  Phase 2  全部解凍，差分 LR，逐步用 predicted box 取代 GT box

使用方式：
    # Phase 1：從 Option C checkpoint 熱啟動，只訓 gate
    uv run train/temporal_yolo/train_conditioned.py \\
        --data-root /path/to/MOT17 \\
        --resume runs/joint/best.ckpt \\
        --phase 1 --epochs 10 --run-dir runs/conditioned

    # Phase 2：全部聯合訓練
    uv run train/temporal_yolo/train_conditioned.py \\
        --data-root /path/to/MOT17 \\
        --resume runs/conditioned/phase1_best.ckpt \\
        --phase 2 --epochs 50 --run-dir runs/conditioned_p2

    # 快速驗證
    uv run train/temporal_yolo/train_conditioned.py \\
        --data-root /path/to/MOT17 --seqs MOT17-02-SDP --phase 1 --epochs 2
"""

from __future__ import annotations
import argparse
import random
import time
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader
from saccade.perception.temporal_yolo.loss import TemporalTrackingLoss
from saccade.perception.temporal_yolo.yolo_conditioned import (
    ConditionedConfig,
    TemporalYOLOConditioned,
    TrackerGateInput,
    build_temporal_yolo_conditioned,
)

_CONDITIONED_AVAILABLE = True


# ---------------------------------------------------------------------------
# Checkpoint helpers（與 train_joint.py 相同介面）
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
    """Remove _orig_mod. prefix inserted by torch.compile so checkpoints are compile-agnostic."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strict: bool = True,
) -> int:
    print(f"[Resume] {path}  strict={strict}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = _strip_compiled_keys(state["model"])
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(
            f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not strict or (not missing and not unexpected):
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception:
            print(
                "  [Warn] Optimizer state not loaded (param group mismatch, expected for phase change)"
            )
    epoch = state.get("epoch", 0) + 1
    print(
        f"  Resumed from epoch {state.get('epoch', 0)}  best_loss={state.get('best_loss', float('inf')):.4f}"
    )
    return epoch  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# GT-box curriculum
# ---------------------------------------------------------------------------
def _select_heatmap_boxes(
    gt_boxes: torch.Tensor,
    prev_pred_boxes: torch.Tensor | None,
    gt_ratio: float,
) -> torch.Tensor:
    """
    訓練時決定 heatmap 用哪種 box：
      gt_ratio=1.0 → 全 GT oracle（Phase 1）
      gt_ratio=0.5 → 50% GT + 50% predicted（Phase 2 早期）
      gt_ratio=0.0 → 全 predicted（Phase 2 後期，最接近推論行為）
    """
    if prev_pred_boxes is None or random.random() < gt_ratio:
        return gt_boxes
    return prev_pred_boxes


# ---------------------------------------------------------------------------
# Phase 1 proxy loss (feature sharpening — bypasses decoder entirely)
# ---------------------------------------------------------------------------
def _proxy_step(
    model: "TemporalYOLOConditioned",
    frames: torch.Tensor,  # (B, T, 3, H, W)
    gt_boxes_batch: list[Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Feature sharpening proxy for Phase 1: maximise cosine alignment between
    gated FPN feature norms and GT heatmaps, without running the decoder.

    Gradient path: loss → feat_norm → gate_maps → alpha  (short chain, fast backward).
    """
    B, T, _, H, W = frames.shape
    loss = frames.new_zeros(())

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
        for t in range(T):
            # Gate receives t-1 boxes (simulates tracker state arriving before frame t).
            # t=0 has no prior frame → empty boxes; gate produces near-zero maps, no gradient.
            prev_boxes = [
                gt_boxes_batch[b][t - 1] if t > 0 else torch.zeros(0, 4)
                for b in range(B)
            ]
            gate_inputs_t = [
                TrackerGateInput.from_boxes_scores(
                    prev_boxes[b], None, (H, W), assume_absolute=True
                ).to(device)
                for b in range(B)
            ]
            # Target heatmap uses current-frame GT (where objects actually are at t).
            target_inputs_t = [
                TrackerGateInput.from_boxes_scores(
                    gt_boxes_batch[b][t], None, (H, W), assume_absolute=True
                ).to(device)
                for b in range(B)
            ]
            gated, gate_maps, feat_hws = model.forward_gate_only(
                frames[:, t], gate_inputs_t
            )
            for s in model.cfg.scales:
                feat_norm = gated[s].float().norm(dim=1)  # (B, H_s, W_s)
                scale = feat_norm.detach().mean().clamp(min=1e-6)
                feat_norm_n = feat_norm / scale

                heatmap_s = model.gate.renderer.batch_forward(
                    target_inputs_t, feat_hws[s]
                )  # (B, H_s, W_s)

                loss = loss - (feat_norm_n * heatmap_s).mean()

    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _gpu_forward(
    model: "TemporalYOLOConditioned",
    frames: torch.Tensor,  # (B, T, 3, H, W) float32 on device
    gt_boxes_batch: list[Any],
    gt_box_ratio: float,
    device: torch.device,
) -> tuple[list[Any], list[Any]]:
    """Run all T forward passes under autocast. Returns pred_boxes_b_t, pred_scores_b_t."""
    B, T, _, H, W = frames.shape
    model.reset_sequence()
    prev_queries: torch.Tensor | None = None
    prev_boxes: torch.Tensor | None = None
    pred_boxes_b_t: list[list[torch.Tensor]] = [[] for _ in range(B)]
    pred_scores_b_t: list[list[torch.Tensor]] = [[] for _ in range(B)]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
        for t in range(T):
            gate_inputs_t: list[TrackerGateInput] = []
            for b in range(B):
                b_prev = prev_boxes[b] if prev_boxes is not None else None
                heatmap_boxes = _select_heatmap_boxes(
                    gt_boxes_batch[b][t], b_prev, gt_box_ratio
                )
                gate_inputs_t.append(
                    TrackerGateInput.from_boxes_scores(
                        heatmap_boxes, None, (H, W), assume_absolute=True
                    ).to(device)
                )
            out = model(
                frames[:, t], prev_queries=prev_queries, gate_inputs=gate_inputs_t
            )
            prev_queries = out["updated_queries"].detach()
            prev_boxes = out["boxes"].detach()
            for b in range(B):
                pred_boxes_b_t[b].append(out["boxes"][b])
                pred_scores_b_t[b].append(out["scores"][b])

    return pred_boxes_b_t, pred_scores_b_t


def train_one_epoch(  # type: ignore[no-untyped-def]
    model: "TemporalYOLOConditioned",
    loader,
    criterion: TemporalTrackingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gt_box_ratio: float = 1.0,
    log_interval: int = 20,
    proxy_loss: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True).float() / 255.0
        gt_boxes_batch = batch["gt_boxes"]
        B, T, _, H, W = frames.shape

        optimizer.zero_grad(set_to_none=True)

        if proxy_loss:
            batch_loss = _proxy_step(model, frames, gt_boxes_batch, device)
        else:
            pred_boxes_b_t, pred_scores_b_t = _gpu_forward(
                model, frames, gt_boxes_batch, gt_box_ratio, device
            )
            losses = criterion.forward_fast(
                pred_boxes_b_t, pred_scores_b_t, gt_boxes_batch, (H, W)
            )
            batch_loss = losses["loss_total"]

        batch_loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += batch_loss.item()
        if (i + 1) % log_interval == 0 or i == n_batches - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_batches - i - 1) / max(rate, 1e-6)
            print(
                f"  [{epoch:3d}] {i + 1:4d}/{n_batches}"
                f"  loss={batch_loss.item():.4f}"
                f"  avg={total_loss / (i + 1):.4f}"
                f"  {'proxy' if proxy_loss else f'gt={gt_box_ratio:.2f}'}"
                f"  {rate:.1f}it/s  ETA {eta / 60:.1f}m"
            )

    return total_loss / max(n_batches, 1)


def _gt_ratio_schedule(epoch: int, phase: int, total_epochs: int) -> float:
    """
    Phase 1：固定 1.0（純 GT oracle）
    Phase 2：線性從 0.8 降到 0.0（逐漸切到 predicted box）
    """
    if phase == 1:
        return 1.0
    # Phase 2：線性退火
    return max(0.0, 0.8 * (1.0 - epoch / total_epochs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option D: Track-Conditioned YOLO training"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--run-dir", default="runs/conditioned")
    parser.add_argument("--resume", default="", help="Option C or Phase 1 checkpoint")
    parser.add_argument(
        "--resume-strict",
        action="store_true",
        default=False,
        help="Strict state_dict load (False=allow missing gate keys)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder LR")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-gate", type=float, default=5e-5)
    parser.add_argument("--scales", default="p5")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument(
        "--gate-sigma-scale",
        type=float,
        default=0.5,
        help="Gaussian sigma = box_size * sigma_scale",
    )
    parser.add_argument(
        "--gate-min-score",
        type=float,
        default=0.5,
        help="Minimum score to render track into heatmap",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help=(
            "1 = freeze backbone+decoder, train gate only (GT oracle boxes). "
            "2 = full joint training with predicted-box curriculum."
        ),
    )
    # Phase 2 只訓 gate 的 warm-up epochs
    parser.add_argument(
        "--gate-warmup",
        type=int,
        default=0,
        help="Phase 2 開頭先只訓 gate N epochs 再解凍全部（0=直接全解凍）",
    )
    parser.add_argument(
        "--proxy-loss",
        action="store_true",
        help="Phase 1 only: feature sharpening loss, skip decoder (faster)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile backbone (reduce-overhead, ~6x faster, 30s cold-start)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    seqs = args.seqs.split(",") if args.seqs else None
    scales = tuple(s.strip() for s in args.scales.split(","))

    # ── 建立模型 ──
    cfg = ConditionedConfig(
        embed_dim=256,
        num_heads=8,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        ffn_dim=1024,
        score_threshold=0.3,
        scales=scales,
        gate_sigma_scale=args.gate_sigma_scale,
        gate_min_score=args.gate_min_score,
        freeze_backbone=(args.phase == 1),
        freeze_decoder=(args.phase == 1),
    )
    model = build_temporal_yolo_conditioned(args.yolo_weights, cfg, device)

    # ── Optimizer（差分 LR）──
    param_groups = model.parameter_groups(
        lr_backbone=args.lr_backbone,
        lr_gate=args.lr_gate,
        lr_decoder=args.lr,
    )
    if args.phase == 1:
        # Phase 1 只更新 gate 參數
        param_groups = [g for g in param_groups if g["name"] == "gate"]
        print(
            f"[Phase 1] Only training gate  ({sum(p.numel() for g in param_groups for p in g['params']):,} params)"
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_gate * 0.01
    )
    criterion = TemporalTrackingLoss()

    # ── Resume (must happen before torch.compile so state_dict keys match) ──
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        resumed_epoch = load_checkpoint(
            Path(args.resume), model, optimizer, strict=args.resume_strict
        )
        # Cross-phase resume (e.g. Option C → Phase 1): epoch counter resets.
        # Same-phase resume: keep the epoch to continue where we left off.
        resume_state = torch.load(
            Path(args.resume), map_location="cpu", weights_only=False
        )
        resumed_phase = resume_state.get("phase")
        if resumed_phase is not None and resumed_phase != args.phase:
            print(
                f"  [Resume] Phase change ({resumed_phase}→{args.phase}): resetting epoch to 1"
            )
            start_epoch = 1
        elif resumed_epoch <= args.epochs:
            start_epoch = resumed_epoch
        else:
            print(
                f"  [Resume] start_epoch ({resumed_epoch}) > epochs ({args.epochs}): resetting to 1"
            )
            start_epoch = 1

    # ── torch.compile (after resume so checkpoint keys match) ──
    if args.compile:
        print(
            "[Compile] torch.compile backbone (mode=default) — first epoch has cold-start ~30s"
        )
        model.pyramid = torch.compile(model.pyramid, mode="default")  # type: ignore[assignment]

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

    print(f"\n{'=' * 60}")
    print(f"  Option D — Track-Conditioned YOLO (Phase {args.phase})")
    print(f"  Epochs: {start_epoch}→{args.epochs}  Device: {device}  BF16: ON")
    print(
        f"  Scales: {scales}  σ={args.gate_sigma_scale}  min_score={args.gate_min_score}"
    )
    if args.phase == 1:
        print("  Freeze: backbone+decoder  Train: gate only (GT oracle)")
    else:
        print(
            f"  Full joint  LR: backbone={args.lr_backbone} gate={args.lr_gate} decoder={args.lr}"
        )
    print(f"  Run dir: {run_dir}")
    print(f"{'=' * 60}\n")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        gt_ratio = _gt_ratio_schedule(epoch - start_epoch, args.phase, args.epochs)
        print(f"\n── Epoch {epoch}/{args.epochs}  gt_ratio={gt_ratio:.2f} ──")

        avg_loss = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
            epoch,
            gt_box_ratio=gt_ratio,
            proxy_loss=args.proxy_loss,
        )
        scheduler.step()

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": _strip_compiled_keys(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "phase": args.phase,
                    "best_loss": best_loss,
                    "args": vars(args),
                },
                run_dir,
                epoch,
                is_best,
            )
        alpha_str = "  ".join(
            f"α_{s}={p.item():.5f}" for s, p in model.gate.alphas.items()
        )
        print(f"  {alpha_str}  avg={avg_loss:.4f}  best={best_loss:.4f}")

    print(f"\nDone. Best loss: {best_loss:.4f}  →  {run_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
