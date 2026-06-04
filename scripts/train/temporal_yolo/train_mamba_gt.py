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

project_root = Path(__file__).resolve().parent.parent.parent.parent
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
# Feature-cache RAM/VRAM preload
# ---------------------------------------------------------------------------
# In cache mode the backbone never runs, so each step's only heavy work was
# B*T synchronous torch.load() calls from disk — that starves the GPU (util
# ~30%). Preloading the (only the used) cached features into RAM/VRAM once
# turns per-step loading into a dict lookup + H2D copy.
_feature_ram_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _preload_feature_cache(
    cache_dir: Path, seqs: list[str], device: torch.device
) -> None:
    global _feature_ram_cache
    if _feature_ram_cache:
        return

    seq_dirs = [cache_dir / s for s in seqs if (cache_dir / s).is_dir()]
    total = sum(1 for sd in seq_dirs for _ in sd.glob("*.pt"))

    # Hold features in VRAM only if there is comfortable headroom; otherwise CPU
    # RAM (still removes the disk read + deserialize from the hot path).
    target = torch.device("cpu")
    if device.type == "cuda":
        try:
            free_b, _ = torch.cuda.mem_get_info(device)
            # ~3 MB/frame; require ~2x headroom for activations + model.
            if free_b > total * 3 * 1024**2 * 2:
                target = device
        except Exception:
            pass

    print(
        f"[FeatureCache] Preloading {total} files from {len(seq_dirs)} seq(s) to {target}...",
        flush=True,
    )
    t0 = time.time()
    done = 0
    try:
        for sd in seq_dirs:
            for pt_file in sorted(sd.glob("*.pt")):
                feat = torch.load(pt_file, map_location=target, weights_only=True)
                _feature_ram_cache[(sd.name, int(pt_file.stem))] = (
                    feat["p3"],
                    feat["p4"],
                    feat["p5"],
                )
                done += 1
        print(
            f"[FeatureCache] {done}/{total} on {target} in {time.time() - t0:.0f}s",
            flush=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and target.type == "cuda":
            print("[FeatureCache] VRAM OOM — falling back to CPU RAM...", flush=True)
            _feature_ram_cache.clear()
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            for sd in seq_dirs:
                for pt_file in sorted(sd.glob("*.pt")):
                    feat = torch.load(pt_file, map_location="cpu", weights_only=True)
                    _feature_ram_cache[(sd.name, int(pt_file.stem))] = (
                        feat["p3"],
                        feat["p4"],
                        feat["p5"],
                    )
        else:
            raise


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
    parser.add_argument(
        "--per-channel-a",
        action="store_true",
        help="Per-channel SSM A (d_inner, N) instead of shared (1, N). "
        "Warm-starts from a shared-A checkpoint (broadcast → init == base).",
    )
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--freeze-teacher", type=bool, default=True)
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile teacher (mode=default)"
    )
    parser.add_argument(
        "--accum-steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="linear LR warmup epochs"
    )
    parser.add_argument(
        "--clip-grad", type=float, default=1.0, help="gradient clipping max_norm"
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Use precomputed teacher FPN features (skip backbone forward pass)",
    )
    parser.add_argument(
        "--no-preload-cache",
        action="store_true",
        help="Disable preloading the feature cache into RAM/VRAM (load per-step "
        "from disk instead). Preload is on by default in cache mode and removes "
        "the per-step torch.load that otherwise starves the GPU.",
    )
    parser.add_argument(
        "--freeze-temporal",
        action="store_true",
        help="Freeze temporal_blocks and flow_gate_conv; only train spatial path.",
    )
    parser.add_argument(
        "--add-temporal",
        action="store_true",
        help="Load spatial weights from --mamba-ckpt (strict=False), force use_temporal_mamba=True. "
        "Use when the base checkpoint has no temporal blocks (e.g. Cross-Scan).",
    )
    parser.add_argument(
        "--freeze-spatial",
        action="store_true",
        help="Freeze all spatial path (input_proj, downsample, mamba_blocks, upsample, heads). "
        "Only temporal_blocks and flow_gate_conv are trained. Use with --add-temporal.",
    )
    parser.add_argument(
        "--use-temporal-attention",
        action="store_true",
        help="Use TemporalAttentionBlock instead of SSM for temporal path. "
        "No hidden state → no train/eval mismatch. Use with --add-temporal.",
    )
    parser.add_argument(
        "--use-pixel-shuffle",
        action="store_true",
        help="Override checkpoint's use_pixel_shuffle — replace bilinear "
        "upsampling with Conv2d + PixelShuffle learned upsampler "
        "(+10–15% less active GPU time; requires retraining).",
    )
    parser.add_argument(
        "--t1-weight",
        type=float,
        default=0.0,
        help="Weight for T=1 auxiliary loss (pure spatial, temporal blocks bypassed). "
        "Prevents spatial path from drifting toward temporal-dependent features. "
        "1.0 = equal weight to the full T-clip loss. Requires --add-temporal.",
    )
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
        "use_pixel_shuffle": False,
        "per_channel_a": False,
    }
    for k, v in mamba_args_default.items():
        mamba_args.setdefault(k, v)
    if args.add_temporal:
        if getattr(args, "use_temporal_attention", False):
            mamba_args["use_temporal_attention"] = True
        else:
            mamba_args["use_temporal_mamba"] = True
    if args.per_channel_a:
        # Force per-channel A; the shared-A base ckpt warm-starts via broadcast
        # inside MambaDetectionHead.load_state_dict (init == base, then fine-tune).
        mamba_args["per_channel_a"] = True
    if args.use_pixel_shuffle:
        mamba_args["use_pixel_shuffle"] = True

    mamba = MambaDetectionHead(
        in_channels=(128, 256, 512),
        d_model=mamba_args["d_model"],
        d_state=mamba_args["d_state"],
        num_blocks=mamba_args["num_blocks"],
        num_classes=mamba_args["num_classes"],
        reg_max=1,
        spatial_reduction=mamba_args["spatial_reduction"],
        use_pixel_shuffle=mamba_args["use_pixel_shuffle"],
        use_cross_scan=mamba_args.get("use_cross_scan", False),
        use_hybrid_head=mamba_args.get("use_hybrid_head", False),
        use_temporal_mamba=mamba_args.get("use_temporal_mamba", False),
        use_temporal_attention=mamba_args.get("use_temporal_attention", False),
        per_channel_a=mamba_args.get("per_channel_a", False),
    ).to(device)
    sd = _strip_compiled_keys(mamba_state["student"])
    # per_channel_a from a shared-A base warm-starts via A_log broadcast; relax
    # strict so any incidental layout drift is tolerated alongside it.
    base_per_channel = mamba_state.get("mamba_args", {}).get("per_channel_a", False)
    strict = not args.add_temporal and not (args.per_channel_a and not base_per_channel)
    missing, unexpected = mamba.load_state_dict(sd, strict=strict)
    mamba.to(device)  # ensure new temporal keys are on correct device
    if missing:
        temporal_keys = [k for k in missing if "temporal" in k or "flow_gate" in k]
        other_missing = [k for k in missing if k not in temporal_keys]
        print(f"  New temporal keys (random init): {len(temporal_keys)}")
        if other_missing:
            print(f"  WARNING: other missing keys: {other_missing}")
    mamba.train()

    if args.freeze_temporal:
        frozen = []
        for mod in [mamba.temporal_blocks, mamba.flow_gate_conv]:
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)
                frozen.append(mod.__class__.__name__)
        print(f"  Frozen: {frozen} (temporal FP gate preserved from checkpoint)")

    if args.freeze_spatial:
        # Freeze feature extraction only — NOT cls_head/reg_head so they can
        # adapt to temporal-modified features (x_up + temporal_delta).
        spatial_mods = [mamba.input_proj, mamba.downsample, mamba.mamba_blocks]
        if hasattr(mamba, "upsample") and mamba.upsample is not None:
            spatial_mods.append(mamba.upsample)
        if mamba.hybrid_conv is not None:
            spatial_mods.append(mamba.hybrid_conv)
        for mod in spatial_mods:
            for p in mod.parameters():
                p.requires_grad_(False)
        print(
            "  Frozen: feature extraction (input_proj/downsample/mamba_blocks/upsample)"
        )

    n_params = sum(p.numel() for p in mamba.parameters())
    n_trainable = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
    print(f"  Mamba params: {n_params:,}  trainable: {n_trainable:,}")

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

    if args.compile:
        print("[Compile] torch.compile teacher (mode=default) — ~30s cold-start")
        teacher.yolo_model = torch.compile(teacher.yolo_model, mode="default")

    remaining = max(args.epochs - start_epoch + 1, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=args.lr * 0.01
    )
    warmup_scheduler: Any = None
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: (
                (e + 1) / args.warmup_epochs if e < args.warmup_epochs else 1.0
            ),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    seqs = args.seqs.split(",") if args.seqs else None
    # In cache mode the backbone never runs, so raw frame pixels are unused —
    # skip RAM preload to allow large mixes (e.g. MOT17+MOT20) without OOM.
    preload = args.cache_dir == ""
    loader = build_mot17_dataloader(
        data_root=project_root / args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        stride=args.clip_len * 2,
        shuffle=True,
        seqs=seqs,
        preload_to_ram=preload,
    )
    print(
        f"[MambaGT] {len(loader)} batches/epoch  gt_ratio={args.gt_ratio}  lr={args.lr}"
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        print(f"[Cache] Loading precomputed teacher features from {cache_dir}")
        if not args.no_preload_cache:
            _preload_feature_cache(cache_dir, list(loader.dataset.sequences), device)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_batches = len(loader)
    print(
        f"\nTraining {args.epochs} epochs  ({n_batches} batches/epoch  "
        f"B={args.batch_size} T={args.clip_len} accum={args.accum_steps})...\n"
    )
    accum = args.accum_steps
    for epoch in range(start_epoch, args.epochs + 1):
        mamba.train()
        epoch_total_loss = 0.0
        n_steps = 0
        acc_batches = 0
        t0 = time.perf_counter()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
            gt_boxes_batch = batch["gt_boxes"]
            B, T = frames.shape[:2]

            # Collect features for all T frames, then forward as one temporal clip.
            # This ensures temporal blocks see actual frame-to-frame context during
            # training (instead of per-frame independent passes that never activate
            # the temporal SSM). All-frames loss prevents temporal hallucination.
            all_feats: list[list[torch.Tensor]] = [[], [], []]
            all_gt: list[list[torch.Tensor]] = []

            for t in range(T):
                frame_t = frames[:, t]
                gt_t = [gt_boxes_batch[b][t] for b in range(B)]
                all_gt.append(gt_t)

                gate_inputs = None
                if t > 0:
                    prev_gt = [gt_boxes_batch[b][t - 1] for b in range(B)]
                    gate_inputs = _build_gate_inputs(
                        prev_gt, args.gt_ratio, args.img_size, device
                    )

                if cache_dir is not None:
                    p3s, p4s, p5s = [], [], []
                    for b in range(B):
                        seq: str = batch["seq"][b]  # type: ignore[index]
                        fid: int = batch["frame_ids"][b][t]  # type: ignore[index]
                        cached = _feature_ram_cache.get((seq, fid))
                        if cached is None:
                            feat = torch.load(
                                cache_dir / seq / f"{fid:06d}.pt",
                                map_location="cpu",
                                weights_only=True,
                            )
                            cached = (feat["p3"], feat["p4"], feat["p5"])
                        p3s.append(cached[0].to(device, dtype=torch.float32))
                        p4s.append(cached[1].to(device, dtype=torch.float32))
                        p5s.append(cached[2].to(device, dtype=torch.float32))
                    feats_t = [
                        torch.stack(p3s),
                        torch.stack(p4s),
                        torch.stack(p5s),
                    ]
                else:
                    fpn_feats.clear()
                    _ = teacher(frame_t, gate_input=gate_inputs)
                    feats_t = [fpn_feats[s] for s in ("p3", "p4", "p5")]

                for si in range(3):
                    all_feats[si].append(feats_t[si])

            # Stack T frames in batch-major order: (B*T, C, H, W) per scale.
            # MambaDetectionHead.forward() uses reshape(B, T, ...) which requires
            # [b0t0, b0t1, ..., b0tT, b1t0, ...] ordering so each SSM sequence
            # corresponds to one sample across T frames, not B samples at one frame.
            feats_T = [
                torch.stack(all_feats[si], dim=1).reshape(
                    B * T, *all_feats[si][0].shape[1:]
                )
                for si in range(3)
            ]

            # Temporal forward — T_frames activates temporal blocks
            s_cls_T, s_reg_T = mamba(feats_T, T=T)

            # All-frames loss: every frame supervised with its own GT.
            # Batch-major layout: [b0t0, b0t1, ..., b0tT, b1t0, ...]
            # Frame t for all B items: indices t, T+t, 2T+t, ... → c[t::T]
            batch_loss = frames.new_zeros(())
            for t in range(T):
                s_cls_t = [c[t::T] for c in s_cls_T]
                s_reg_t = [r[t::T] for r in s_reg_T]
                feats_t = [all_feats[si][t] for si in range(3)]
                preds = _build_preds_dict(s_cls_t, s_reg_t, feats_t)
                yolo_batch = _make_yolo_batch(all_gt[t], args.img_size, device)
                step_loss_vec, _ = criterion(preds, yolo_batch)
                batch_loss = batch_loss + step_loss_vec.sum()
            batch_loss = batch_loss / (T * accum)

            batch_loss.backward()
            acc_batches += 1

            if acc_batches == accum:
                nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                acc_batches = 0

            step_loss = batch_loss.detach().item() * accum
            if torch.isfinite(batch_loss):
                epoch_total_loss += step_loss
                n_steps += 1

            eta_s = (
                (n_batches - batch_idx - 1)
                * (time.perf_counter() - t0)
                / max(batch_idx + 1, 1)
            )
            pct = (batch_idx + 1) / n_batches * 100
            print(
                f"\r  epoch {epoch:3d}/{args.epochs}  "
                f"[{pct:5.1f}%  {batch_idx + 1:4d}/{n_batches}]  "
                f"loss={step_loss:7.2f}  ETA {eta_s:5.0f}s",
                end="",
                flush=True,
            )

        if acc_batches > 0:
            nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        if (
            warmup_scheduler is not None
            and epoch <= start_epoch + args.warmup_epochs - 1
        ):
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_total_loss / max(n_steps, 1)
        dt = time.perf_counter() - t0
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        is_best = avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)
        best_marker = " [BEST]" if is_best else ""
        print(
            f"\r  epoch {epoch:3d}/{args.epochs}  "
            f"loss={avg_loss:7.2f}  lr={current_lr:.2e}  "
            f"VRAM={vram_gb:.1f}GB  {dt / 60:.1f}min{best_marker}"
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
                        "use_pixel_shuffle": mamba_args.get("use_pixel_shuffle", False),
                        "use_cross_scan": mamba_args.get("use_cross_scan", False),
                        "use_hybrid_head": mamba_args.get("use_hybrid_head", False),
                        "use_temporal_mamba": mamba_args.get(
                            "use_temporal_mamba", False
                        ),
                        "use_temporal_attention": mamba_args.get(
                            "use_temporal_attention", False
                        ),
                        "per_channel_a": mamba_args.get("per_channel_a", False),
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
