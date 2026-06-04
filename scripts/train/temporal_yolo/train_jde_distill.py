"""
OSNet → JDE embedding head knowledge distillation.

Method B (Direct KD):
  Teacher : OSNet x1.0 ONNX (512-dim, frozen)
            → trainable osnet_proj  Linear(512 → 128)
  Student : Mamba emb_head  (trainable, lr × 0.1)
            + EmbeddingProjector   (trainable)
  Loss    : cosine KD  +  α × CE (collapse prevention)

The CE term on the student side prevents the trivial solution where
both osnet_proj and student collapse to the same constant vector.

Usage
-----
    uv run scripts/train/temporal_yolo/train_jde_distill.py \\
        --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt \\
        --market-root datasets/Market-1501-v15.09.15 \\
        --run-dir runs/jde_distill_v1 \\
        --epochs 50 --batch-size 64

After training, evaluate with:
    uv run scripts/eval/jde_market1501.py \\
        --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt \\
        --proj-ckpt runs/jde_distill_v1/best.ckpt \\
        --market-root datasets/Market-1501-v15.09.15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.temporal_yolo.mamba_head import EmbeddingProjector  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    _GATE_LAYER_IDX,
)
from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    DataPreloader,
    resize_stretch_batch_gpu,
)
from saccade.perception.temporal_yolo.training_utils import save_checkpoint  # noqa: E402

IMG_SIZE = 640
EMB_DIM_PER_SCALE = 128
EMB_OUT_DIM = 128
OSNET_H, OSNET_W = 256, 128  # OSNet input size
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Dataset helpers (reused from train_jde_market.py)
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
    def __init__(
        self,
        items: list[tuple[Path, int]],
        batch_size: int,
        samples_per_id: int,
    ) -> None:
        self._by_id: dict[int, list[Path]] = {}
        for path, pid in items:
            self._by_id.setdefault(pid, []).append(path)
        self._ids = list(self._by_id.keys())
        self.batch_size = batch_size
        self.samples_per_id = samples_per_id

    def sample_batch(self) -> list[Path]:
        import random

        ids = random.sample(self._ids, self.batch_size // self.samples_per_id)
        batch: list[Path] = []
        for pid in ids:
            pool = self._by_id[pid]
            chosen = random.choices(pool, k=self.samples_per_id)
            batch.extend(chosen)
        return batch


# ---------------------------------------------------------------------------
# OSNet ONNX teacher
# ---------------------------------------------------------------------------


class OSNetTeacher:
    """Wraps OSNet ONNX for batched inference. Returns L2-normalised 512-dim."""

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self._sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name

    def __call__(self, imgs_uint8_batch: torch.Tensor) -> torch.Tensor:
        """
        imgs_uint8_batch : (N, 3, H, W) uint8 on any device
        Returns          : (N, 512) float32 L2-normalised on CPU
        """
        imgs_f = imgs_uint8_batch.float() / 255.0
        imgs_osnet = F.interpolate(
            imgs_f, size=(OSNET_H, OSNET_W), mode="bilinear", align_corners=False
        )
        # ImageNet mean/std normalisation (same as production feature_extractor.py)
        mean = _IMAGENET_MEAN.to(imgs_osnet.device)
        std = _IMAGENET_STD.to(imgs_osnet.device)
        imgs_osnet = (imgs_osnet - mean) / std
        imgs_np = imgs_osnet.cpu().numpy().astype(np.float32)
        out = self._sess.run(None, {self._input_name: imgs_np})[0]  # (N, 512)
        t = torch.from_numpy(out)
        return F.normalize(t, dim=1)  # (N, 512)


# ---------------------------------------------------------------------------
# FPN feature extraction helper
# ---------------------------------------------------------------------------


def _get_fpn_feats(
    teacher: nn.Module,
    yolo_model: nn.Module,
    frame: torch.Tensor,
) -> list[torch.Tensor]:
    layers = yolo_model.model
    save: set[int] = set(yolo_model.save)
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
    return [y[_GATE_LAYER_IDX[s]] for s in ("p3", "p4", "p5")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OSNet → JDE distillation")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_vgt_mamba_v14/best.ckpt")
    parser.add_argument(
        "--osnet-onnx", default="models/embedding/osnet_x1_0_256x128.onnx"
    )
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-id", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=1.0,
        help="Weight for cosine KD loss term",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.5,
        help="Weight for CE collapse-prevention loss term",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for CE loss",
    )
    parser.add_argument("--run-dir", default="runs/jde_distill_v1")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = project_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — PREPARE
    # ------------------------------------------------------------------
    print("Loading Market-1501...")
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root
    items, pid_to_idx = _build_market_dataset(market_root)
    num_ids = len(pid_to_idx)
    sampler = MarketBatchSampler(items, args.batch_size, args.samples_per_id)
    print(f"  Images: {len(items)}  IDs: {num_ids}")

    all_paths = [p for p, _ in items]
    import os

    num_workers = args.workers or min(os.cpu_count() or 4, 16)
    preloader = DataPreloader(all_paths, num_workers=num_workers)
    preloader.load()

    print("Loading OSNet teacher (ONNX)...")
    osnet_path = project_root / args.osnet_onnx
    osnet_teacher = OSNetTeacher(str(osnet_path))
    print(f"  OSNet: {osnet_path.name}  output=512-dim")

    # Phase 1b — Precompute OSNet embeddings (fixed teacher, run once)
    osnet_cache_path = run_dir / "osnet_embs.pt"
    if osnet_cache_path.exists():
        print(f"[Cache] Loading OSNet embeddings from {osnet_cache_path}")
        osnet_cache: dict[str, torch.Tensor] = torch.load(
            osnet_cache_path, map_location="cpu", weights_only=True
        )
    else:
        print(f"[Cache] Precomputing OSNet embeddings for {len(all_paths)} images...")
        osnet_cache = {}
        batch_size_pre = 128
        t_pre = time.perf_counter()
        for i in range(0, len(all_paths), batch_size_pre):
            batch_paths_pre = all_paths[i : i + batch_size_pre]
            imgs_pre = torch.stack([preloader[p] for p in batch_paths_pre])
            embs = osnet_teacher(imgs_pre)  # (N, 512) L2-norm, on CPU
            for p, e in zip(batch_paths_pre, embs):
                osnet_cache[str(p)] = e
            if (i // batch_size_pre) % 20 == 0:
                print(
                    f"  {i + len(batch_paths_pre)}/{len(all_paths)}",
                    end="\r",
                    flush=True,
                )
        torch.save(osnet_cache, osnet_cache_path)
        print(
            f"\n  Done in {time.perf_counter() - t_pre:.1f}s  saved → {osnet_cache_path}"
        )
    del osnet_teacher  # no longer needed after cache is built

    print("Building Mamba student...")
    detector = build_mamba_gated_detector(
        yolo_pt_path=str((project_root / args.yolo_weights).resolve()),
        teacher_ckpt=str((project_root / args.teacher_ckpt).resolve()),
        mamba_ckpt=str((project_root / args.mamba_ckpt).resolve()),
        img_size=args.img_size,
        device=device,
        emb_dim=EMB_DIM_PER_SCALE,
    )
    detector.eval()
    mamba_head = detector.mamba_head
    teacher_model = detector.teacher
    yolo_model = teacher_model.yolo_model

    if mamba_head.emb_head is None:
        print("[ERROR] No emb_head. Ensure emb_dim=128 is set in mamba_args.")
        sys.exit(1)

    # OSNet projector: 512 → 128 (trainable, learns best compression for this student)
    osnet_proj = nn.Linear(512, EMB_OUT_DIM, bias=False).to(device)
    nn.init.normal_(osnet_proj.weight, std=0.01)

    # Student projector: 128*3 → 256 → 128
    projector = EmbeddingProjector(
        emb_dim=EMB_DIM_PER_SCALE, hidden=256, out_dim=EMB_OUT_DIM
    ).to(device)

    # ID classifier for CE loss (collapse prevention)
    id_classifier = nn.Linear(EMB_OUT_DIM, num_ids).to(device)
    nn.init.normal_(id_classifier.weight, std=0.01)
    nn.init.zeros_(id_classifier.bias)

    # ------------------------------------------------------------------
    # Phase 2 — TRAIN
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(
            project_root / args.resume, map_location="cpu", weights_only=False
        )
        projector.load_state_dict(ckpt["projector"])
        osnet_proj.load_state_dict(ckpt["osnet_proj"])
        if ckpt.get("emb_head") is not None:
            mamba_head.emb_head.load_state_dict(ckpt["emb_head"])
        if ckpt.get("id_classifier") is not None:
            id_classifier.load_state_dict(ckpt["id_classifier"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[Resume] epoch={start_epoch}  best_loss={best_loss:.4f}")

    # Freeze backbone, train emb_head (slow lr) + projector + osnet_proj + classifier
    for p in detector.parameters():
        p.requires_grad_(False)
    for p in mamba_head.emb_head.parameters():
        p.requires_grad_(True)

    param_groups = [
        {"params": projector.parameters(), "lr": args.lr},
        {"params": osnet_proj.parameters(), "lr": args.lr},
        {"params": id_classifier.parameters(), "lr": args.lr},
        {"params": mamba_head.emb_head.parameters(), "lr": args.lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    steps_per_epoch = max(1, len(items) // args.batch_size)
    total_steps = steps_per_epoch * (args.epochs - start_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    trainable = sum(p.numel() for p in projector.parameters())
    trainable += sum(p.numel() for p in osnet_proj.parameters())
    trainable += sum(p.numel() for p in id_classifier.parameters())
    trainable += sum(p.numel() for p in mamba_head.emb_head.parameters())

    print(f"\nTraining {args.epochs} epochs  ~{steps_per_epoch} steps/epoch")
    print(f"  Trainable params: {trainable:,}")
    print(f"  kd_weight={args.kd_weight}  ce_weight={args.ce_weight}")
    print(f"  lr={args.lr}  label_smoothing={args.label_smoothing}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        total_correct = 0
        total_samples = 0

        projector.train()
        osnet_proj.train()
        id_classifier.train()
        mamba_head.emb_head.train()

        for step in range(steps_per_epoch):
            batch_paths = sampler.sample_batch()
            imgs_uint8 = torch.stack([preloader[p] for p in batch_paths])

            # Teacher: look up precomputed OSNet embeddings (no ONNX inference in loop)
            osnet_embs = torch.stack([osnet_cache[str(p)] for p in batch_paths]).to(
                device
            )  # (N, 512)

            # Compress teacher to 128-dim (osnet_proj is trainable)
            teacher_128 = F.normalize(osnet_proj(osnet_embs), dim=1)  # (N, 128)

            # Student: Mamba backbone (frozen) → emb_head (trainable) → projector
            images = resize_stretch_batch_gpu(imgs_uint8, args.img_size, device)
            with torch.no_grad():
                fpn_feats = _get_fpn_feats(teacher_model, yolo_model, images)
            _, _, emb_preds = mamba_head(fpn_feats, return_embeddings=True)
            pooled = mamba_head.pool_embeddings_global(emb_preds)  # (N, 128*3)
            student_128 = projector(pooled)  # (N, 128), L2-normalised

            # KD loss: student → match compressed teacher
            loss_kd = (
                1.0 - F.cosine_similarity(student_128, teacher_128.detach())
            ).mean()

            # CE loss: student ID classification (prevents collapse)
            raw_pids = torch.tensor(
                [_parse_pid(p.name) for p in batch_paths], device=device
            )
            labels = torch.tensor([pid_to_idx[int(p)] for p in raw_pids], device=device)
            logits = id_classifier(student_128)
            loss_ce = ce_loss_fn(logits, labels)

            loss = args.kd_weight * loss_kd + args.ce_weight * loss_ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)
            total_kd_loss += float(loss_kd)
            total_ce_loss += float(loss_ce)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                acc = total_correct / max(total_samples, 1) * 100
                print(
                    f"\r  epoch {epoch + 1:3d}/{args.epochs}  "
                    f"step {step:4d}/{steps_per_epoch}  "
                    f"loss={float(loss):.4f}  kd={float(loss_kd):.4f}  "
                    f"ce={float(loss_ce):.4f}  acc={acc:.1f}%  lr={lr:.2e}",
                    end="",
                    flush=True,
                )

        avg_loss = total_loss / steps_per_epoch
        avg_kd = total_kd_loss / steps_per_epoch
        avg_ce = total_ce_loss / steps_per_epoch
        acc = total_correct / max(total_samples, 1) * 100
        elapsed = time.perf_counter() - t0
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        print(
            f"\r  epoch {epoch + 1:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  kd={avg_kd:.4f}  ce={avg_ce:.4f}  "
            f"acc={acc:.1f}%  time={elapsed:.0f}s  {'[BEST]' if is_best else ''}",
        )

        # ------------------------------------------------------------------
        # Phase 3 — SAVE
        # ------------------------------------------------------------------
        ckpt: dict[str, Any] = {
            "projector": projector.state_dict(),
            "osnet_proj": osnet_proj.state_dict(),
            "emb_head": mamba_head.emb_head.state_dict(),
            "id_classifier": id_classifier.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "emb_dim_per_scale": EMB_DIM_PER_SCALE,
            "emb_out_dim": EMB_OUT_DIM,
            "img_size": args.img_size,
            "loss_mode": "distill_kd+ce",
            "pool_mode": "global",
            "num_ids": num_ids,
            "pid_to_idx": pid_to_idx,
        }
        save_checkpoint(ckpt, run_dir, epoch + 1, is_best=is_best)


if __name__ == "__main__":
    main()
