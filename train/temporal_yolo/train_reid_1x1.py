#!/usr/bin/env python3
"""Train lightweight 1×1 Conv dimension-reduction head on Market-1501.

Uses raw FPN features (cached from single YOLO forward) + per-scale
1×1 Conv → center-pool → classifier. Much smaller than v8's ReIDConvHead.

Modes:
  --scales p5         P5 only (512→128, ~65K params)
  --scales p3p4p5     Full FPN (896→384→128, ~164K params)

Usage:
    # v9a: P5 only
    uv run train/temporal_yolo/train_reid_1x1.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --run-dir runs/jde_market_v9a --scales p5 --epochs 30

    # v9b: Full FPN
    uv run train/temporal_yolo/train_reid_1x1.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --run-dir runs/jde_market_v9b --scales p3p4p5 --epochs 30
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    _GATE_LAYER_IDX,
)
from saccade.perception.temporal_yolo.training_utils import (  # noqa: E402
    save_checkpoint,
)
from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    DataPreloader,
    resize_stretch_batch_gpu,
)

IMG_SIZE = 640


# ── 1×1 Conv dimension reduction head ──


class DimReduceHead(nn.Module):
    """Per-scale 1×1 Conv → center-pool → concat → optional projector → L2."""

    def __init__(self, in_channels: list[int], out_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.nl = len(in_channels)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_dim, 1, bias=False) for c in in_channels]
        )
        nn.init.normal_(self.convs[-1].weight, std=0.001)  # smaller init for P5

        mid_dim = out_dim * self.nl
        if mid_dim != out_dim:
            self.proj = nn.Sequential(
                nn.Linear(mid_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        parts = []
        for conv, f in zip(self.convs, feats):
            x = conv(f)
            h, w = x.shape[2], x.shape[3]
            center = x[:, :, h // 2, w // 2]
            parts.append(center)
        pooled = torch.cat(parts, dim=1)
        return F.normalize(self.proj(pooled), dim=1)


# ── Dataset helpers ──


def _parse_pid(filename: str) -> int:
    return int(Path(filename).stem.split("_")[0])


def _build_market_dataset(
    market_root: Path,
) -> tuple[list[tuple[Path, int]], dict[int, int]]:
    train_dir = market_root / "bounding_box_train"
    items: list[tuple[Path, int]] = []
    for p in sorted(train_dir.glob("*.jpg")):
        pid = _parse_pid(p.name)
        items.append((p, pid))
    unique_pids = sorted(set(pid for _, pid in items))
    pid_to_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
    return items, pid_to_idx


class MarketBatchSampler:
    def __init__(
        self, items: list[tuple[Path, int]], batch_size: int, samples_per_id: int = 4
    ):
        self.items = items
        self.batch_size = batch_size
        self.samples_per_id = samples_per_id
        self._by_id: dict[int, list[Path]] = {}
        for path, pid in items:
            self._by_id.setdefault(pid, []).append(path)
        self._ids = sorted(self._by_id.keys())

    def sample_batch(self) -> list[Path]:
        n_ids = max(1, self.batch_size // self.samples_per_id)
        chosen_ids = random.sample(self._ids, min(n_ids, len(self._ids)))
        batch: list[Path] = []
        for pid in chosen_ids:
            pool = self._by_id[pid]
            n = min(self.samples_per_id, len(pool))
            batch.extend(random.sample(pool, n))
        while len(batch) < self.batch_size:
            pid = random.choice(self._ids)
            batch.append(random.choice(self._by_id[pid]))
        random.shuffle(batch)
        return batch[: self.batch_size]


def _get_teacher_feats(teacher, yolo_model, frame):
    layers = yolo_model.model
    save = set(yolo_model.save)
    y: list[torch.Tensor | None] = []
    x = frame
    for i in range(23):
        m = layers[i]
        if m.f != -1:
            if isinstance(m.f, int):
                x = y[m.f]
            else:
                x = [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if i in save else None)
    fpn_indices = [_GATE_LAYER_IDX[s] for s in ("p3", "p4", "p5")]
    return [y[i] for i in fpn_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--scales", choices=["p5", "p3p4p5"], default="p5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-id", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--run-dir", default="runs/jde_market_v9a")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    num_workers = args.workers or min(os.cpu_count() or 4, 16)

    teacher_ckpt = project_root / args.teacher_ckpt
    mamba_ckpt = project_root / args.mamba_ckpt
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root
    run_dir = project_root / args.run_dir

    print(f"Device: {device}  Scales: {args.scales}")

    # ── Data ──
    items, pid_to_idx = _build_market_dataset(market_root)
    num_ids = len(pid_to_idx)
    sampler = MarketBatchSampler(items, args.batch_size, args.samples_per_id)
    print(f"  Images: {len(items)}  IDs: {num_ids}")

    all_paths = [p for p, _pid in items]
    preloader = DataPreloader(all_paths, num_workers=num_workers)
    preloader.load()

    # ── Build detector (for FPN access) ──
    detector = build_mamba_gated_detector(
        yolo_pt_path=str((project_root / args.yolo_weights).resolve()),
        teacher_ckpt=str(teacher_ckpt.resolve()) if teacher_ckpt.exists() else "",
        mamba_ckpt=str(mamba_ckpt.resolve()),
        img_size=IMG_SIZE,
        device=device,
        emb_dim=128,
    )
    detector.eval()
    teacher = detector.teacher
    yolo_model = teacher.yolo_model

    # ── FPN feature dimensions ──
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    fpn_feats = _get_teacher_feats(teacher, yolo_model, dummy)
    fpn_channels = [f.shape[1] for f in fpn_feats]
    print(f"  FPN: P3={fpn_channels[0]} P4={fpn_channels[1]} P5={fpn_channels[2]}")

    if args.scales == "p5":
        in_channels = [fpn_channels[2]]  # P5 only
    else:
        in_channels = fpn_channels  # P3, P4, P5

    # ── Build head and classifier ──
    head = DimReduceHead(in_channels, out_dim=128).to(device)
    classifier = nn.Linear(128, num_ids).to(device)
    nn.init.normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)

    trainable = sum(p.numel() for p in head.parameters()) + sum(
        p.numel() for p in classifier.parameters()
    )
    print(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"  Total trainable: {trainable:,}")

    # ── Resume ──
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        head.load_state_dict(ckpt["head"])
        classifier.load_state_dict(ckpt["classifier"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if start_epoch >= args.epochs:
            print(f"  Already done {start_epoch}/{args.epochs}")
            sys.exit(0)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.lr},
        ]
    )
    steps_per_epoch = max(1, len(items) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print(f"\nTraining {args.epochs} epochs, ~{steps_per_epoch} steps/epoch")
    print(f"  LR: {args.lr}  Label smooth: {args.label_smoothing}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        head.train()
        classifier.train()

        for step in range(steps_per_epoch):
            batch_paths = sampler.sample_batch()
            imgs_uint8 = torch.stack([preloader[p] for p in batch_paths]).to(device)
            frame_640 = resize_stretch_batch_gpu(imgs_uint8, IMG_SIZE, device)

            with torch.no_grad():
                fpn = _get_teacher_feats(teacher, yolo_model, frame_640.float())
            if args.scales == "p5":
                fpn = [fpn[2]]

            embeddings = head(fpn)
            logits = classifier(embeddings)
            raw_pids = torch.tensor(
                [_parse_pid(p.name) for p in batch_paths], device=device
            )
            labels = torch.tensor([pid_to_idx[int(p)] for p in raw_pids], device=device)
            loss = ce_loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += len(labels)
            total_loss += float(loss)

            if step % 50 == 0:
                acc = total_correct / max(total_samples, 1) * 100
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch + 1:3d} step {step:4d}  loss={float(loss):.4f}  acc={acc:.1f}%  lr={lr:.2e}"
                )

        avg_loss = total_loss / max(steps_per_epoch, 1)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        elapsed = time.perf_counter() - t0
        acc = total_correct / max(total_samples, 1) * 100
        print(
            f"  epoch {epoch + 1:3d} done  loss={avg_loss:.4f}  acc={acc:.1f}%  time={elapsed:.1f}s  {'[BEST]' if is_best else ''}"
        )

        save_checkpoint(
            {
                "head": head.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "num_ids": num_ids,
                "pid_to_idx": pid_to_idx,
                "scales": args.scales,
                "in_channels": in_channels,
            },
            run_dir,
            epoch + 1,
            is_best=is_best,
        )

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
