"""
JDE-style embedding training on Market-1501 — Pipeline Convention
================================================================

Trains the embedding projection head (EmbeddingProjector) on top of
MambaDetectionHead's per-pixel emb_head. Supports two loss modes:

  --loss crossentropy   ID classification (default, recommended)
  --loss supcon         Supervised contrastive (v5 and earlier)

Only the EmbeddingProjector is trained by default; the YOLO backbone,
MambaDetectionHead (cls/reg/emb), and TrackSpatialGate remain frozen.

Dataset: Market-1501 person crops → stretch-resize → global pool → train.


Pipeline Convention
-------------------

All training scripts in this directory follow a 3-phase pipeline:

  Phase 1 — PREPARE: preprocess data that will be reused across epochs
    1a. Preload raw images to RAM (DataPreloader)
    1b. If encoder weights are frozen: precompute outputs, persist to disk (FeatureCache)

  Phase 2 — TRAIN: training loop only touches trainable parameters
    - Data source is either FeatureCache (fast) or live encoder forward pass (slow)
    - Optimizer + scheduler + loss + logging

  Phase 3 — SAVE: checkpointing via shared training_utils.save_checkpoint()


Usage
-----

    # CrossEntropy (v6 — recommended):
    uv run train/temporal_yolo/train_jde_market.py \
        --yolo-weights models/yolo/yolo26s.pt \
        --teacher-ckpt runs/gated_det_v1/best.ckpt \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --market-root datasets/Market-1501-v15.09.15 \
        --run-dir runs/jde_market_v6 \
        --loss crossentropy --label-smoothing 0.1 \
        --train-emb-head --epochs 30 --batch-size 64

    # SupCon (v5 compat):
    uv run train/temporal_yolo/train_jde_market.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --market-root datasets/Market-1501-v15.09.15 \
        --run-dir runs/jde_market_v5 \
        --loss supcon --train-emb-head --epochs 30 --batch-size 64
"""
# status: archive-candidate

from __future__ import annotations

import argparse
import random
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

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.mamba_head import (  # noqa: E402
    EmbeddingProjector,
)
from saccade.perception.temporal_yolo.reid_head import supcon_loss  # noqa: E402
from saccade.perception.temporal_yolo.reid_conv_head import ReIDConvHead  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    _GATE_LAYER_IDX,
)
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.temporal_yolo.training_utils import (  # noqa: E402
    save_checkpoint,
)
from scripts.provenance.run_manifest import open_run  # noqa: E402
from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    DataPreloader,
    FeatureCache,
    resize_stretch_batch_gpu,
)

EMB_DIM_PER_SCALE = 128
EMB_OUT_DIM = 128
IMG_SIZE = 640


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


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
    """P×K batch sampler for person re-identification."""

    def __init__(
        self,
        items: list[tuple[Path, int]],
        batch_size: int,
        samples_per_id: int = 4,
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


# ---------------------------------------------------------------------------
# Teacher forward pass (frozen encoder)
# ---------------------------------------------------------------------------


def _get_teacher_feats(
    teacher: nn.Module, yolo_model: nn.Module, frame: torch.Tensor
) -> list[torch.Tensor]:
    layers = yolo_model.model
    save: set[int] = set(yolo_model.save)
    y: list[torch.Tensor | None] = []
    x: Any = frame
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
    return [y[i] for i in fpn_indices]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-id", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="SupCon temperature (ignored for crossentropy)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for CrossEntropy",
    )
    parser.add_argument(
        "--loss", choices=["crossentropy", "supcon"], default="crossentropy"
    )
    parser.add_argument(
        "--pool",
        choices=["center", "global"],
        default="center",
        help="Pooling: center pixel vs global avg (default: center)",
    )
    parser.add_argument(
        "--reid-head",
        action="store_true",
        help="Use standalone ReIDConvHead (vs Mamba emb_head)",
    )
    parser.add_argument("--run-dir", default="runs/jde_market_v1")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--train-emb-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the mamba emb_head (default: True)",
    )
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = args.img_size

    teacher_ckpt = project_root / args.teacher_ckpt
    mamba_ckpt = project_root / args.mamba_ckpt
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root
    run_dir = project_root / args.run_dir

    # ADR 021 AP-2: claim the run directory before the embedding cache or the
    # first checkpoint is written.
    open_run(
        run_dir,
        produced_by="train",
        dataset=str(args.market_root),
    )

    if args.precompute and args.train_emb_head:
        print("[ERROR] --precompute is incompatible with --train_emb_head")
        print("  Precomputed embeddings would be stale when emb_head is being trained.")
        sys.exit(1)

    import os

    num_workers = args.workers or min(os.cpu_count() or 4, 16)

    print(f"Device: {device}")
    print(f"Market-1501 root: {market_root}")
    print(f"Loss mode: {args.loss}  Pool: {args.pool}")

    items, pid_to_idx = _build_market_dataset(market_root)
    num_ids = len(pid_to_idx)
    sampler = MarketBatchSampler(items, args.batch_size, args.samples_per_id)
    id_counts = [len(v) for v in sampler._by_id.values()]
    print(
        f"  Images: {len(items)}  IDs: {num_ids}  "
        f"Imgs/ID: min={min(id_counts)} max={max(id_counts)} avg={sum(id_counts) / len(id_counts):.1f}"
    )

    # 1a. Preload images to RAM (eliminate JPEG decode from training loop)
    all_paths = [p for p, _pid in items]
    preloader = DataPreloader(all_paths, num_workers=num_workers)
    preloader.load()

    # ------------------------------------------------------------------
    # Build model (needed for both FeatureCache precompute and training)
    # ------------------------------------------------------------------

    detector = build_mamba_gated_detector(
        yolo_pt_path=str((project_root / args.yolo_weights).resolve()),
        teacher_ckpt=str(teacher_ckpt.resolve()) if teacher_ckpt.exists() else "",
        mamba_ckpt=str(mamba_ckpt.resolve()),
        img_size=img_size,
        device=device,
        emb_dim=EMB_DIM_PER_SCALE,
    )
    detector.eval()

    mamba_head = detector.mamba_head
    teacher = detector.teacher
    yolo_model = teacher.yolo_model

    # ------------------------------------------------------------------
    # Decide backbone pipeline: emb_head (Mamba) or standalone ReIDConvHead
    # ------------------------------------------------------------------
    reid_conv_head: ReIDConvHead | None = None
    if args.reid_head:
        # Detect FPN channel dims from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=device)
            fpn_feats = _get_teacher_feats(teacher, yolo_model, dummy)
            fpn_channels = [f.shape[1] for f in fpn_feats]
        print(
            f"  FPN channels: P3={fpn_channels[0]} P4={fpn_channels[1]} P5={fpn_channels[2]}"
        )
        reid_conv_head = ReIDConvHead(fpn_channels, emb_dim=EMB_DIM_PER_SCALE).to(
            device
        )
        print(
            f"  ReIDConvHead params: {sum(p.numel() for p in reid_conv_head.parameters()):,}"
        )
    elif mamba_head.emb_head is None:
        print("[ERROR] MambaDetectionHead has no emb_head (emb_dim=0 in checkpoint).")
        print("  Re-train the Mamba head with emb_dim > 0 or use --reid-head.")
        sys.exit(1)

    pool_fn = (
        mamba_head.pool_embeddings_center
        if args.pool == "center"
        else mamba_head.pool_embeddings_global
    )
    if reid_conv_head is not None:
        pool_fn = ReIDConvHead.pool_center

    projector = EmbeddingProjector(
        emb_dim=mamba_head.emb_dim if reid_conv_head is None else EMB_DIM_PER_SCALE,
        hidden=256,
        out_dim=EMB_OUT_DIM,
    ).to(device)

    # ID classifier for cross-entropy mode
    id_classifier: nn.Linear | None = None
    if args.loss == "crossentropy":
        id_classifier = nn.Linear(EMB_OUT_DIM, num_ids).to(device)
        nn.init.normal_(id_classifier.weight, std=0.01)
        nn.init.zeros_(id_classifier.bias)

    # ------------------------------------------------------------------
    # Phase 1b — Precompute frozen encoder outputs (if --precompute)
    # ------------------------------------------------------------------

    pooled_cache: FeatureCache | None = None
    if args.precompute:
        pooled_cache = FeatureCache(run_dir / "precomputed_embeddings.pt")
        if not pooled_cache.exists():
            pooled_cache.compute(
                preloader=preloader,
                paths=all_paths,
                device=device,
                encoder_fn=_make_encoder_fn(
                    teacher,
                    yolo_model,
                    mamba_head,
                    pool_fn,
                    img_size,
                    reid_conv_head,
                ),
                batch_size=args.batch_size * 2,
            )
        pooled_cache.load()

    # ------------------------------------------------------------------
    # Phase 2 — TRAIN
    # ------------------------------------------------------------------

    print("\n=== Phase 2: Train ===")

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        projector.load_state_dict(ckpt["projector"])
        if reid_conv_head is not None and ckpt.get("reid_conv_head") is not None:
            reid_conv_head.load_state_dict(ckpt["reid_conv_head"])
        elif (
            ckpt.get("emb_head") is not None
            and args.train_emb_head
            and reid_conv_head is None
        ):
            mamba_head.emb_head.load_state_dict(ckpt["emb_head"])
        if ckpt.get("id_classifier") is not None and id_classifier is not None:
            id_classifier.load_state_dict(ckpt["id_classifier"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if start_epoch >= args.epochs:
            print(
                f"  Already completed {start_epoch}/{args.epochs} epochs, nothing to do."
            )
            sys.exit(0)
        print(f"  Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    for p in detector.parameters():
        p.requires_grad_(False)
    if args.train_emb_head and reid_conv_head is None:
        for p in mamba_head.emb_head.parameters():
            p.requires_grad_(True)
    if reid_conv_head is not None:
        for p in reid_conv_head.parameters():
            p.requires_grad_(True)
    for p in projector.parameters():
        p.requires_grad_(True)
    if id_classifier is not None:
        for p in id_classifier.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    if args.train_emb_head and reid_conv_head is None:
        trainable += sum(
            p.numel() for p in mamba_head.emb_head.parameters() if p.requires_grad
        )
    if reid_conv_head is not None:
        trainable += sum(
            p.numel() for p in reid_conv_head.parameters() if p.requires_grad
        )
    if id_classifier is not None:
        trainable += sum(
            p.numel() for p in id_classifier.parameters() if p.requires_grad
        )
    print(f"  Trainable params: {trainable:,}")

    param_groups: list[dict] = [
        {"params": projector.parameters(), "lr": args.lr},
    ]
    if args.train_emb_head and reid_conv_head is None:
        param_groups.append(
            {"params": mamba_head.emb_head.parameters(), "lr": args.lr * 0.1}
        )
    if reid_conv_head is not None:
        param_groups.append({"params": reid_conv_head.parameters(), "lr": args.lr})
    if id_classifier is not None:
        param_groups.append({"params": id_classifier.parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = max(1, len(items) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Cross-entropy loss factory
    if args.loss == "crossentropy":
        ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        ce_loss_fn = None

    print(f"\nTraining {args.epochs} epochs, ~{steps_per_epoch} steps/epoch")
    print(f"  Batch size: {args.batch_size}  Samples/ID: {args.samples_per_id}")
    print(f"  LR: {args.lr}  Loss: {args.loss}  Pool: {args.pool}")
    print(f"  Mode: {'ReIDConvHead' if reid_conv_head is not None else 'emb_head'}")
    if args.loss == "crossentropy":
        print(f"  Label smoothing: {args.label_smoothing}")
        print(f"  ID classifier: Linear({EMB_OUT_DIM}, {num_ids})")
    else:
        print(f"  Temperature: {args.temperature}")
    print(f"  Feature cache:  {pooled_cache is not None}")
    print(f"  Image cache:    {len(preloader)} images in RAM")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        projector.train()
        if args.train_emb_head and reid_conv_head is None:
            mamba_head.emb_head.train()
        if reid_conv_head is not None:
            reid_conv_head.train()
        if id_classifier is not None:
            id_classifier.train()

        for step in range(steps_per_epoch):
            batch_paths = sampler.sample_batch()

            if pooled_cache is not None:
                pooled = pooled_cache.get_batch(batch_paths, device)
            else:
                imgs_uint8 = torch.stack([preloader[p] for p in batch_paths])
                images = resize_stretch_batch_gpu(imgs_uint8, img_size, device)
                with torch.no_grad():
                    feats = _get_teacher_feats(teacher, yolo_model, images)
                if reid_conv_head is not None:
                    emb_preds = reid_conv_head(feats)
                    pooled = ReIDConvHead.pool_center(emb_preds)
                else:
                    _, _, emb_preds = mamba_head(feats, return_embeddings=True)
                    pooled = pool_fn(emb_preds)

            embeddings = projector(pooled)

            if args.loss == "crossentropy":
                raw_pids = torch.tensor(
                    [_parse_pid(p.name) for p in batch_paths], device=device
                )
                labels = torch.tensor(
                    [pid_to_idx[int(p)] for p in raw_pids], device=device
                )
                logits = id_classifier(embeddings)
                loss = ce_loss_fn(logits, labels)
                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += len(labels)
            else:
                labels = torch.tensor(
                    [_parse_pid(p.name) for p in batch_paths], device=device
                )
                loss = supcon_loss(embeddings, labels, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                extra = ""
                if args.loss == "crossentropy" and total_samples > 0:
                    acc = total_correct / total_samples * 100
                    extra = f"  acc={acc:.1f}%"
                print(
                    f"  epoch {epoch + 1:3d} step {step:4d}/{steps_per_epoch}  "
                    f"loss={float(loss):.4f}  lr={lr:.2e}{extra}"
                )

        avg_loss = total_loss / max(steps_per_epoch, 1)
        elapsed = time.perf_counter() - t0
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        train_acc = ""
        if args.loss == "crossentropy" and total_samples > 0:
            train_acc = f"  acc={total_correct / total_samples * 100:.1f}%"
        print(
            f"  epoch {epoch + 1:3d} done  avg_loss={avg_loss:.4f}{train_acc}  "
            f"time={elapsed:.1f}s  {'[BEST]' if is_best else ''}"
        )

        # ------------------------------------------------------------------
        # Phase 3 — SAVE
        # ------------------------------------------------------------------
        ckpt: dict[str, Any] = {
            "projector": projector.state_dict(),
            "emb_head": mamba_head.emb_head.state_dict()
            if (args.train_emb_head and reid_conv_head is None)
            else None,
            "reid_conv_head": reid_conv_head.state_dict()
            if reid_conv_head is not None
            else None,
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "emb_dim_per_scale": EMB_DIM_PER_SCALE,
            "emb_out_dim": EMB_OUT_DIM,
            "img_size": IMG_SIZE,
            "loss_mode": args.loss,
            "pool_mode": args.pool,
            "reid_head_mode": True if reid_conv_head is not None else False,
        }
        if args.loss == "crossentropy":
            ckpt["num_ids"] = num_ids
            ckpt["pid_to_idx"] = pid_to_idx
            ckpt["id_classifier"] = id_classifier.state_dict()

        save_checkpoint(ckpt, run_dir, epoch + 1, is_best=is_best)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# Encoder closure (Phase 1b) — stretch-resize + global pool
# ---------------------------------------------------------------------------


def _make_encoder_fn(
    teacher: nn.Module,
    yolo_model: nn.Module,
    mamba_head: nn.Module,
    pool_fn,
    img_size: int,
    reid_conv_head: nn.Module | None = None,
):
    """Return encoder_fn(imgs_uint8_cpu, device) → (N, emb_dim*3) on device."""

    def encode(imgs_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
        images = resize_stretch_batch_gpu(imgs_uint8, img_size, device)
        feats = _get_teacher_feats(teacher, yolo_model, images)
        if reid_conv_head is not None:
            emb_preds = reid_conv_head(feats)
            return ReIDConvHead.pool_center(emb_preds)
        _, _, emb_preds = mamba_head(feats, return_embeddings=True)
        return pool_fn(emb_preds)

    return encode


if __name__ == "__main__":
    main()
