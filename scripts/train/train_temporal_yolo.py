"""
TemporalYOLOHybrid 訓練腳本

支援：
  - FP16 混合精度訓練（torch.amp）
  - Checkpoint 儲存與續訓（--resume）
  - Option C：Frozen YOLO Backbone + 可訓練 Cross-Attention Head
  - MOT17 時序 DataLoader

使用方式：
    # 開始訓練
    uv run scripts/train/train_temporal_yolo.py --data-root /path/to/MOT17

    # 從 checkpoint 續訓
    uv run scripts/train/train_temporal_yolo.py \\
        --data-root /path/to/MOT17 \\
        --resume runs/temporal_yolo/latest.ckpt

    # 指定部分序列做快速驗證
    uv run scripts/train/train_temporal_yolo.py \\
        --data-root /path/to/MOT17 \\
        --seqs MOT17-02-SDP,MOT17-04-SDP
"""
# status: experiment

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 確保 saccade 套件可找到
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from saccade.perception.temporal_yolo.model import (
    TemporalYOLOConfig,
    TemporalYOLOHybrid,
)
from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader
from saccade.perception.temporal_yolo.loss import TemporalTrackingLoss
from saccade.perception.temporal_yolo.yolo_joint import (
    JointConfig,
    TemporalYOLOJoint,
    build_temporal_yolo_joint,
)


# ---------------------------------------------------------------------------
# YOLO Feature Extractor (Option C: 從真實 yolo26s 取 Neck 特徵)
# ---------------------------------------------------------------------------
class YOLONeckExtractor(nn.Module):
    """
    掛接到 Ultralytics yolo26s，從 model.22（Neck 最後一層）
    以 forward hook 提取 (B, 512, H/32, W/32) 的特徵圖。

    在 Option C 訓練中：
      - YOLO backbone 參數 Frozen（不計梯度，節省 VRAM）
      - 只有 Cross-Attention Head 的參數參與訓練

    若要 Phase 2 聯合訓練，把 freeze=False 即可解凍。
    """

    def __init__(self, yolo_pt_path: str, freeze: bool = True):
        super().__init__()
        from ultralytics import YOLO as UltralyticsYOLO

        yolo = UltralyticsYOLO(yolo_pt_path)
        self.yolo_model = yolo.model
        self.feat_channels: int = 0
        self._feat: torch.Tensor | None = None
        self._hook = None

        # 掛 Hook 到 Neck 最後一層（model.22）
        target = dict(self.yolo_model.named_modules()).get("model.22")
        if target is None:
            raise RuntimeError("Cannot find model.22 in yolo26s. Check YOLO version.")

        def _hook_fn(mod, inp, out):
            self._feat = out
            if self.feat_channels == 0:
                self.feat_channels = out.shape[1]

        self._hook = target.register_forward_hook(_hook_fn)

        # BUG FIX: Ultralytics loads all params with requires_grad=False by default.
        # Must explicitly enable/disable; not doing so made freeze=False a silent no-op.
        for p in self.yolo_model.parameters():
            p.requires_grad_(not freeze)
        if freeze:
            self.yolo_model.eval()
            print(f"[YOLONeckExtractor] YOLO Frozen (Neck → {self.feat_channels}ch)")
        else:
            n = sum(p.numel() for p in self.yolo_model.parameters())
            print(
                f"[YOLONeckExtractor] YOLO Trainable ({n:,} params, Neck → {self.feat_channels}ch)"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32
        Returns:
            feat: (B, C, H/32, W/32) Neck feature map
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            _ = self.yolo_model(x)
        feat = self._feat
        if feat is None:
            raise RuntimeError("Hook did not capture feature. Check hook registration.")
        return feat

    def __del__(self):
        if self._hook is not None:
            self._hook.remove()


# ---------------------------------------------------------------------------
# 完整訓練模型（YOLO Neck + Temporal Cross-Attention Head）
# ---------------------------------------------------------------------------
class TemporalYOLOTrainer(nn.Module):
    """
    將 YOLO Neck Extractor 與 TemporalYOLOHybrid 的 Cross-Attention Decoder 串接。
    取代 TemporalYOLOHybrid 裡的 Stub backbone，接上真實 YOLO 特徵。
    """

    def __init__(
        self,
        yolo_pt_path: str,
        cfg: TemporalYOLOConfig,
        freeze_yolo: bool = True,
    ):
        super().__init__()
        self.neck = YOLONeckExtractor(yolo_pt_path, freeze=freeze_yolo)

        # 取得真實 channel 數（需先跑一次 forward 來觸發 hook）
        dummy = torch.zeros(1, 3, 640, 640)
        _ = self.neck(dummy)
        neck_ch = self.neck.feat_channels
        print(f"[Trainer] YOLO Neck channels: {neck_ch}")

        # 更新 config 讓 feat_proj 能對應真實 channel 數
        cfg = TemporalYOLOConfig(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_queries=cfg.num_queries,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            num_decoder_layers=cfg.num_decoder_layers,
            score_threshold=cfg.score_threshold,
            yolo_feat_channels=neck_ch,  # ← 使用真實 channel 數
        )
        self.hybrid = TemporalYOLOHybrid(cfg)

    def forward(
        self,
        frame: torch.Tensor,
        prev_queries: torch.Tensor | None = None,
    ) -> dict:
        feat = self.neck(frame)
        # 繞過 hybrid.backbone，直接用真實 YOLO 特徵餵給 decoder
        if prev_queries is None:
            B = frame.shape[0]
            prev_queries = self.hybrid._make_init_queries(B, frame.device)
        if self.hybrid._TemporalYOLOHybrid__lifecycle is None:
            self.hybrid._init_lifecycle()
        lc = self.hybrid._TemporalYOLOHybrid__lifecycle
        boxes, scores, updated_queries = self.hybrid.decoder(prev_queries, feat)
        updated_queries, query_ids = lc.update(updated_queries, scores)
        return {
            "boxes": boxes,
            "scores": scores,
            "updated_queries": updated_queries,
            "query_ids": query_ids,
        }

    def reset_sequence(self):
        self.hybrid.reset_sequence()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    state: dict,
    run_dir: Path,
    epoch: int,
    is_best: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.ckpt"
    epoch_f = run_dir / f"epoch_{epoch:04d}.ckpt"
    torch.save(state, latest)
    torch.save(state, epoch_f)
    if is_best:
        torch.save(state, run_dir / "best.ckpt")
    print(f"  💾 Checkpoint saved: {epoch_f.name}" + (" [BEST]" if is_best else ""))


def load_checkpoint(path: Path, model: nn.Module, optimizer) -> int:
    print(f"[Resume] Loading checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = state["epoch"] + 1
    print(
        f"  Resumed from epoch {state['epoch']}  "
        f"(best_loss={state.get('best_loss', float('inf')):.4f})"
    )
    return start_epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TemporalYOLOTrainer,
    loader,
    criterion: TemporalTrackingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
) -> float:
    model.train()
    model.neck.yolo_model.eval()  # YOLO Frozen 時保持 eval mode

    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device)  # (B, T, 3, H, W)
        gt_boxes_batch = batch["gt_boxes"]  # list[list[Tensor]]

        B, T, C, H, W = frames.shape
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            batch_loss = frames.new_zeros(())

            for b in range(B):
                model.reset_sequence()
                prev_queries = None
                pred_boxes_seq, pred_scores_seq, gt_seq = [], [], []

                for t in range(T):
                    frame_t = frames[b, t].unsqueeze(0)  # (1, 3, H, W)
                    out = model(frame_t, prev_queries)
                    prev_queries = out["updated_queries"].detach()

                    pred_boxes_seq.append(out["boxes"][0])  # (N_q, 4)
                    pred_scores_seq.append(out["scores"][0])  # (N_q,)
                    gt_seq.append(gt_boxes_batch[b][t])

                losses = criterion(pred_boxes_seq, pred_scores_seq, gt_seq, (H, W))
                batch_loss = batch_loss + losses["loss_total"] / B

        # BF16 不需要 GradScaler
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        loss_val = batch_loss.item()
        total_loss += loss_val

        if (batch_idx + 1) % log_interval == 0 or batch_idx == n_batches - 1:
            elapsed = time.perf_counter() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (n_batches - batch_idx - 1) / max(rate, 1e-6)
            print(
                f"  Epoch {epoch:3d} [{batch_idx + 1:4d}/{n_batches}]"
                f"  loss={loss_val:.4f}"
                f"  avg={total_loss / (batch_idx + 1):.4f}"
                f"  {rate:.1f} it/s"
                f"  ETA {eta / 60:.1f}m"
            )

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train TemporalYOLO on MOT17  [--mode frozen | joint]"
    )
    parser.add_argument("--data-root", required=True, help="Path to MOT17 dataset root")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--run-dir", default="runs/temporal_yolo")
    parser.add_argument(
        "--resume", default="", help="Path to checkpoint to resume from"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Decoder LR (also used as backbone LR when --mode frozen)",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-5,
        help="Backbone LR for joint training (--mode joint). Typically lr/10.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seqs", default="", help="Comma-separated sequence names")
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N epochs"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["frozen", "joint"],
        default="frozen",
        help=(
            "frozen: Option B — freeze YOLO backbone, train decoder only. "
            "joint:  Option C — end-to-end joint training with differential LR."
        ),
    )
    # Multi-scale FPN (joint mode only; ignored in frozen mode)
    parser.add_argument(
        "--scales",
        default="p5",
        help="Comma-separated FPN scales for joint mode: p3,p4,p5 or p5 (default).",
    )

    # Legacy compat aliases
    parser.add_argument(
        "--freeze-yolo",
        action="store_true",
        dest="_freeze_compat",
        default=None,
        help="Deprecated: use --mode frozen instead.",
    )
    parser.add_argument("--no-freeze-yolo", dest="_freeze_compat", action="store_false")

    args = parser.parse_args()

    # Handle legacy --freeze-yolo / --no-freeze-yolo flags
    if args._freeze_compat is not None:
        args.mode = "frozen" if args._freeze_compat else "joint"
        print(f"[Compat] --freeze-yolo translated to --mode {args.mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seqs = args.seqs.split(",") if args.seqs else None
    scales = tuple(s.strip() for s in args.scales.split(","))

    # ── Model ──
    if args.mode == "joint":
        cfg = JointConfig(
            embed_dim=256,
            num_heads=8,
            num_queries=100,
            num_decoder_layers=3,
            ffn_dim=1024,
            score_threshold=0.3,
            scales=scales,
            freeze_backbone=False,
        )
        model: nn.Module = build_temporal_yolo_joint(
            yolo_pt_path=args.yolo_weights,
            cfg=cfg,
            device=device,
        )
        assert isinstance(model, TemporalYOLOJoint)
        param_groups = model.parameter_groups(
            lr_backbone=args.lr_backbone,
            lr_decoder=args.lr,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        base_cfg = TemporalYOLOConfig(
            embed_dim=256,
            num_heads=8,
            num_queries=100,
            num_decoder_layers=3,
            ffn_dim=1024,
            score_threshold=0.3,
        )
        model = TemporalYOLOTrainer(
            yolo_pt_path=args.yolo_weights,
            cfg=base_cfg,
            freeze_yolo=True,
        ).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"[Train] Trainable params: {n_params:,}")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = TemporalTrackingLoss()

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
    )

    print(f"\n{'=' * 60}")
    print(f"  Mode:   {args.mode.upper()}")
    print(f"  Epochs: {start_epoch} → {args.epochs}")
    print(f"  Device: {device} | BF16 AMP: ON")
    if args.mode == "joint":
        print(f"  LR backbone={args.lr_backbone}  decoder={args.lr}  scales={scales}")
    else:
        print(f"  LR: {args.lr}")
    print(f"  Save dir: {run_dir}")
    print(f"{'=' * 60}\n")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n──── Epoch {epoch}/{args.epochs} ────")
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch)
        scheduler.step()

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg if args.mode == "joint" else base_cfg,
                    "mode": args.mode,
                    "best_loss": best_loss,
                    "args": vars(args),
                },
                run_dir=run_dir,
                epoch=epoch,
                is_best=is_best,
            )

        print(f"  Epoch {epoch} done | avg_loss={avg_loss:.4f} | best={best_loss:.4f}")

    print(f"\n🎉 Training complete! Best loss: {best_loss:.4f}")
    print(f"   Best checkpoint: {run_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
