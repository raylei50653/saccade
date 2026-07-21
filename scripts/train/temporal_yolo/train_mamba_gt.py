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
# status: stable

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.dataset import (  # noqa: E402
    build_mot17_dataloader,
    gpu_decode_clip_batch,
)
from saccade.perception.temporal_yolo.training_utils import (  # noqa: E402
    build_warmup_cosine_scheduler,
    resolve_training_sequences,
    seed_everything,
    sha256_file,
)
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
    _GATE_LAYER_IDX,
)
from saccade.perception.temporal_yolo.mamba_head import (  # noqa: E402
    MambaDetectionHead,
    build_strip_oracle_positions,
    resolve_mamba_in_channels,
)
from saccade.perception.temporal_yolo.ngla_assigner import install_assigner  # noqa: E402
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
_feature_ram_cache: dict[
    tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


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


def _temporal_consistency_loss(
    s_cls_T: list[torch.Tensor],
    gt_boxes_batch: list[list[torch.Tensor]],
    gt_ids_batch: list[list[torch.Tensor]],
    T: int,
    B: int,
    img_size: int,
) -> torch.Tensor:
    """Cross-frame classification-score consistency at matched-track GT centers.

    Route 3 of the T3→T1 study (mamba-t3t1-curriculum-20260613.md §5): the proven
    AssA gain is box/score cross-frame stability, but the per-frame v8 detection
    loss has no term protecting it — so full-grad training erases the shaping
    (§4.1). This penalises the L2 distance between the predicted P3 classification
    logit of the *same track id* in adjacent frames, sampled at each frame's own
    GT box centre. The ROI follows the box, so motion is compensated by
    construction (no GMC/flow needed). Classification-only by design: box geometry
    legitimately moves frame-to-frame, but a track's objectness should be stable.

    s_cls_T layout is batch-major (B*T, nc, H, W); frame t of item b is row b*T+t.
    """
    cls_p3 = s_cls_T[0]  # finest scale (stride 8); spans the full image
    losses: list[torch.Tensor] = []
    for b in range(B):
        for t in range(1, T):
            ids_prev = gt_ids_batch[b][t - 1]
            ids_cur = gt_ids_batch[b][t]
            if ids_prev.numel() == 0 or ids_cur.numel() == 0:
                continue
            match = ids_prev[:, None] == ids_cur[None, :]  # (Np, Nc)
            pi, ci = torch.where(match)
            if pi.numel() == 0:
                continue
            boxes_prev = gt_boxes_batch[b][t - 1].to(cls_p3.device)[pi]
            boxes_cur = gt_boxes_batch[b][t].to(cls_p3.device)[ci]
            cen_prev = (boxes_prev[:, :2] + boxes_prev[:, 2:]) * 0.5
            cen_cur = (boxes_cur[:, :2] + boxes_cur[:, 2:]) * 0.5

            def _grid(cen: torch.Tensor) -> torch.Tensor:
                # px centre -> grid_sample coords in [-1, 1] over the image extent
                g = cen / img_size * 2.0 - 1.0
                return g.view(1, -1, 1, 2)

            map_prev = cls_p3[b * T + (t - 1)].unsqueeze(0)  # (1, nc, H, W)
            map_cur = cls_p3[b * T + t].unsqueeze(0)
            logit_prev = F.grid_sample(map_prev, _grid(cen_prev), align_corners=False)
            logit_cur = F.grid_sample(map_cur, _grid(cen_cur), align_corners=False)
            losses.append(F.mse_loss(logit_cur, logit_prev))

    if not losses:
        return cls_p3.new_zeros(())
    return torch.stack(losses).mean()


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
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke tests; 0 runs the full epoch.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-gate", type=float, default=0.0)
    parser.add_argument("--cls-weight", type=float, default=0.5)
    parser.add_argument(
        "--assigner",
        choices=("tal", "ngla"),
        default="tal",
        help="GT fine-tuning label assigner. tal is Ultralytics TaskAlignedAssigner; "
        "ngla swaps CIoU localization scoring for Gaussian NBCD.",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=0,
        help="Clip start stride. 0 uses clip-len, covering nearly every frame once.",
    )
    parser.add_argument("--gt-ratio", type=float, default=0.5)
    parser.add_argument(
        "--interpolate-gt",
        action="store_true",
        help="Keyframe-cadence training: when img1 holds every frame but gt is on "
        "keyframes only (e.g. PersonPath22 full-cadence layout), linearly "
        "interpolate gt between keyframes for temporal continuity / gt-injection "
        "and mask the supervised loss to real keyframes. No-op on densely "
        "annotated data (every frame is a keyframe).",
    )
    parser.add_argument(
        "--no-preload-images",
        action="store_true",
        help="Live mode only: skip RAM-preloading frames; decode per-step from "
        "disk instead. Needed when the frame set is too large for RAM (e.g. "
        "full-cadence PersonPath22 ~70 GB). Pair with --num-workers for parallel "
        "decode.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (parallel JPEG decode when not "
        "RAM-preloaded). 0 = main-thread.",
    )
    parser.add_argument(
        "--gpu-decode",
        action="store_true",
        help="Live mode only: DataLoader workers return raw JPEG bytes (no decode) "
        "and the train loop batch-decodes them on the GPU (nvJPEG). Avoids CPU "
        "JPEG-decode bottleneck on 1080p full-cadence PersonPath22. Implies "
        "--no-preload-images. Incompatible with --cache-dir and detail sources.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-reduction", type=int, default=4)
    parser.add_argument(
        "--reduction-variant",
        choices=("auto", "conv", "blur-conv", "space-to-depth", "wavelet"),
        default="auto",
        help="Reduction before Mamba. auto keeps checkpoint metadata or conv "
        "for new runs; blur-conv, space-to-depth, and wavelet are recall probes.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument(
        "--reg-max",
        type=int,
        default=1,
        help="Box regression bins. 1 = direct regression (legacy). 16 = DFL "
        "distributional regression (sub-pixel localization; recovers small/mid "
        "recall lost to reg-miss). Changing from the warm-start ckpt reinitializes "
        "reg_head and requires the criterion to switch to DFL.",
    )
    parser.add_argument(
        "--head-depth",
        type=int,
        default=1,
        help="3x3 conv blocks in the cls/reg detection heads. 1 = legacy shallow "
        "head (no BN). 2 = YOLO Detect-aligned (2 blocks + BatchNorm) to test "
        "whether head capacity/depth is the residual small-object recall gap. "
        "Changing from the warm-start ckpt reinitializes cls_head/reg_head.",
    )
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument(
        "--detail-source",
        choices=("none", "global", "high", "native-p3", "strip-oracle"),
        default="none",
        help="none=v14, global=640 control, high=shallow high-res, "
        "native-p3=frozen YOLO high-res P3 oracle, "
        "strip-oracle=fixed-budget GT-routed Detail Mamba",
    )
    parser.add_argument("--detail-height", type=int, default=768)
    parser.add_argument("--detail-width", type=int, default=1280)
    parser.add_argument("--detail-channels", type=int, default=32)
    parser.add_argument("--detail-patch-size", type=int, default=3)
    parser.add_argument("--strip-route-budget", type=int, default=640)
    parser.add_argument("--strip-small-threshold", type=float, default=8.0)
    parser.add_argument("--strip-length", type=int, default=32)
    parser.add_argument("--strip-width", type=int, default=3)
    parser.add_argument("--strip-stem-channels", type=int, default=32)
    parser.add_argument("--strip-route-chunk-size", type=int, default=16)
    parser.add_argument(
        "--per-channel-a",
        action="store_true",
        help="Per-channel SSM A (d_inner, N) instead of shared (1, N). "
        "Warm-starts from a shared-A checkpoint (broadcast → init == base).",
    )
    parser.add_argument("--seqs", default="")
    parser.add_argument(
        "--holdout-seqs",
        default="",
        help="Comma-separated sequences excluded from training for external "
        "detector-recall/HOTA checkpoint selection.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable temporal-consistent clip augmentation (hflip + scale + "
        "translation + brightness/contrast). Only effective in live stages that "
        "load images (e.g. GT1, no --cache-dir); silently inert under --cache-dir "
        "because cached frozen-backbone features are keyed to fixed frames.",
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--best-by",
        choices=("none", "train-loss"),
        default="none",
        help="none saves candidates without claiming a best model. train-loss is "
        "diagnostic only and writes best.ckpt from training loss.",
    )
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
        "--resume-reset-optimizer",
        action="store_true",
        help="Load model/epoch from --resume but start a fresh optimizer and LR "
        "schedule. Required when changing LR or extending an old checkpoint.",
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
        "(+10–15%% less active GPU time; requires retraining).",
    )
    parser.add_argument(
        "--scan-stop-grad",
        action="store_true",
        help="Detach the selective-scan output during training (v14 historical "
        "regime: SSM internals frozen at init; only gate/readout/heads learn).",
    )
    parser.add_argument(
        "--no-scan-stop-grad",
        action="store_true",
        help="Force full scan gradients even when the warm-start checkpoint's "
        "mamba_args carry scan_stop_grad=True (un-freeze SSM internals).",
    )
    parser.add_argument(
        "--legacy-n1-scan",
        action="store_true",
        help="Reproduce the historical pre-v14 N=1 CUDA launch and flattened "
        "rank-16 B indexing bug while retaining d_state=16 tensors and rank-1 C.",
    )
    parser.add_argument(
        "--no-legacy-n1-scan",
        action="store_true",
        help="Disable legacy N=1 scan semantics inherited from a warm-start checkpoint.",
    )
    parser.add_argument(
        "--t1-weight",
        type=float,
        default=0.0,
        help="Weight for T=1 auxiliary loss (pure spatial, temporal blocks bypassed). "
        "Prevents spatial path from drifting toward temporal-dependent features. "
        "1.0 = equal weight to the full T-clip loss. Requires --add-temporal.",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Weight for the cross-frame classification-score consistency term "
        "(route 3 of mamba-t3t1-curriculum-20260613.md). Penalises L2 between the "
        "predicted P3 cls logit of the same track id in adjacent frames, sampled at "
        "each frame's GT box centre. Explicitly protects the T3→T1 shaping during "
        "full-grad training. 0 = off. Requires --add-temporal (T>1).",
    )
    args = parser.parse_args()

    if args.clip_stride < 0:
        parser.error("--clip-stride must be >= 0")
    if args.max_batches_per_epoch < 0:
        parser.error("--max-batches-per-epoch must be >= 0")
    if not 0.0 <= args.gt_ratio <= 1.0:
        parser.error("--gt-ratio must be between 0 and 1")
    if args.cache_dir and args.gt_ratio > 0.0:
        parser.error(
            "--cache-dir contains precomputed ungated FPN features, so --gt-ratio "
            "must be 0. Use eager teacher features when training with gate feedback."
        )
    if args.resume_reset_optimizer and not args.resume:
        parser.error("--resume-reset-optimizer requires --resume")
    if args.consistency_weight > 0.0 and not args.add_temporal:
        parser.error("--consistency-weight requires --add-temporal (needs T>1 clips)")
    if args.t1_weight > 0.0 and not args.add_temporal:
        parser.error("--t1-weight requires --add-temporal (needs T>1 clips)")
    if args.gpu_decode:
        if args.cache_dir:
            parser.error(
                "--gpu-decode is live-mode only (incompatible with --cache-dir)"
            )
        if args.detail_source != "none":
            parser.error("--gpu-decode does not support detail sources")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = project_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Teacher: frozen GatedYOLODetector
    # ------------------------------------------------------------------
    print("Loading teacher...")
    teacher_ckpt = project_root / args.teacher_ckpt if args.teacher_ckpt else None
    teacher_raw = (
        torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        if teacher_ckpt is not None
        else {}
    )
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
        weights_path=str(teacher_ckpt) if teacher_ckpt is not None else "",
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
    in_channels = resolve_mamba_in_channels(mamba_state)
    mamba_args["in_channels"] = list(in_channels)
    mamba_args.setdefault("base_yolo_path", args.yolo_weights)
    expected_yolo_sha = mamba_args.get("base_yolo_sha256")
    actual_yolo_sha = sha256_file(project_root / args.yolo_weights)
    if expected_yolo_sha and expected_yolo_sha != actual_yolo_sha:
        raise ValueError(
            "Mamba checkpoint was trained with different YOLO weights: "
            f"checkpoint={mamba_args.get('base_yolo_path')!r}, "
            f"selected={args.yolo_weights!r}"
        )
    mamba_args["base_yolo_sha256"] = actual_yolo_sha
    actual_teacher_sha = sha256_file(teacher_ckpt) if teacher_ckpt is not None else ""
    expected_teacher_sha = mamba_args.get("teacher_checkpoint_sha256")
    if expected_teacher_sha and expected_teacher_sha != actual_teacher_sha:
        raise ValueError(
            "Mamba checkpoint was trained with a different teacher checkpoint: "
            f"selected={args.teacher_ckpt!r}"
        )
    mamba_args["teacher_checkpoint_sha256"] = actual_teacher_sha
    mamba_args_default = {
        "d_model": args.d_model,
        "d_state": args.d_state,
        "num_blocks": args.num_blocks,
        "spatial_reduction": args.spatial_reduction,
        "reduction_variant": (
            args.reduction_variant if args.reduction_variant != "auto" else "conv"
        ),
        "num_classes": nc,
        "use_pixel_shuffle": False,
        "per_channel_a": False,
    }
    for k, v in mamba_args_default.items():
        mamba_args.setdefault(k, v)
    # reg_max: force to the CLI value (the warm-start base may be reg_max=1). When
    # it differs from the base ckpt, reg_head is reinitialized (filtered from the
    # warm-start state dict below) and the criterion switches to DFL.
    base_reg_max = mamba_state.get("mamba_args", {}).get("reg_max", 1)
    mamba_args["reg_max"] = args.reg_max
    # spatial_reduction: force to the CLI value (setdefault above keeps the base
    # ckpt's value otherwise). When it differs from the base, the downsample conv
    # is reg-shaped to the factor and is reinitialized (filtered below). Default 4
    # matches the base, so normal runs are unaffected.
    base_spatial_reduction = mamba_state.get("mamba_args", {}).get(
        "spatial_reduction", 4
    )
    mamba_args["spatial_reduction"] = args.spatial_reduction
    # head_depth: force to CLI value; on change, cls_head/reg_head reinitialize.
    base_head_depth = mamba_state.get("mamba_args", {}).get("head_depth", 1)
    mamba_args["head_depth"] = args.head_depth
    if args.reduction_variant != "auto":
        mamba_args["reduction_variant"] = args.reduction_variant
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
    if args.scan_stop_grad and args.no_scan_stop_grad:
        parser.error("--scan-stop-grad and --no-scan-stop-grad are exclusive")
    if args.legacy_n1_scan and args.no_legacy_n1_scan:
        parser.error("--legacy-n1-scan and --no-legacy-n1-scan are exclusive")
    if args.scan_stop_grad:
        mamba_args["scan_stop_grad"] = True
    if args.no_scan_stop_grad:
        mamba_args["scan_stop_grad"] = False
    if args.legacy_n1_scan:
        mamba_args["legacy_n1_scan"] = True
        mamba_args["legacy_n1_source"] = "77fcc262^"
    if args.no_legacy_n1_scan:
        mamba_args["legacy_n1_scan"] = False
        mamba_args.pop("legacy_n1_source", None)
    use_strip_detail = args.detail_source == "strip-oracle"
    use_detail_fusion = args.detail_source not in {"none", "strip-oracle"}
    mamba_args["use_detail_fusion"] = use_detail_fusion
    mamba_args["use_strip_detail"] = use_strip_detail
    mamba_args["detail_source"] = args.detail_source
    if use_detail_fusion or use_strip_detail:
        mamba_args["detail_channels"] = args.detail_channels
        mamba_args["detail_patch_size"] = args.detail_patch_size
        mamba_args["detail_feature_channels"] = (
            in_channels[0] if args.detail_source == "native-p3" else 0
        )
        mamba_args["detail_height"] = (
            args.img_size if args.detail_source == "global" else args.detail_height
        )
        mamba_args["detail_width"] = (
            args.img_size if args.detail_source == "global" else args.detail_width
        )
    if use_strip_detail:
        mamba_args["strip_route_budget"] = args.strip_route_budget
        mamba_args["strip_small_threshold"] = args.strip_small_threshold
        mamba_args["strip_length"] = args.strip_length
        mamba_args["strip_width"] = args.strip_width
        mamba_args["strip_stem_channels"] = args.strip_stem_channels
        mamba_args["strip_route_chunk_size"] = args.strip_route_chunk_size

    mamba = MambaDetectionHead(
        in_channels=in_channels,
        d_model=mamba_args["d_model"],
        d_state=mamba_args["d_state"],
        num_blocks=mamba_args["num_blocks"],
        num_classes=mamba_args["num_classes"],
        reg_max=mamba_args["reg_max"],
        head_depth=mamba_args["head_depth"],
        spatial_reduction=mamba_args["spatial_reduction"],
        use_pixel_shuffle=mamba_args["use_pixel_shuffle"],
        use_cross_scan=mamba_args.get("use_cross_scan", False),
        use_hybrid_head=mamba_args.get("use_hybrid_head", False),
        use_temporal_mamba=mamba_args.get("use_temporal_mamba", False),
        use_temporal_attention=mamba_args.get("use_temporal_attention", False),
        per_channel_a=mamba_args.get("per_channel_a", False),
        scan_stop_grad=mamba_args.get("scan_stop_grad", False),
        legacy_n1_scan=mamba_args.get("legacy_n1_scan", False),
        reduction_variant=mamba_args.get("reduction_variant", "conv"),
        use_detail_fusion=mamba_args.get("use_detail_fusion", False),
        detail_channels=mamba_args.get("detail_channels", 32),
        detail_patch_size=mamba_args.get("detail_patch_size", 3),
        detail_feature_channels=mamba_args.get("detail_feature_channels", 0),
        use_strip_detail=mamba_args.get("use_strip_detail", False),
        strip_stem_channels=mamba_args.get("strip_stem_channels", 32),
        strip_length=mamba_args.get("strip_length", 32),
        strip_width=mamba_args.get("strip_width", 3),
        strip_route_chunk_size=mamba_args.get("strip_route_chunk_size", 16),
    ).to(device)
    sd = _strip_compiled_keys(mamba_state["student"])
    # reg_max change (e.g. 1 -> 16 DFL): reg_head output channels differ
    # (reg_max*4), so its warm-start weights are shape-incompatible. Drop them so
    # reg_head reinitializes; everything else still warm-starts.
    reg_max_changed = args.reg_max != base_reg_max
    if reg_max_changed:
        sd = {k: v for k, v in sd.items() if not k.startswith("reg_head.")}
        print(
            f"  reg_max {base_reg_max} -> {args.reg_max}: reg_head reinitialized "
            f"(DFL={'on' if args.reg_max > 1 else 'off'})"
        )
    # spatial_reduction change: the downsample (and pixel-shuffle upsample, if any)
    # convs are shaped to the reduction factor, so their warm-start weights are
    # shape-incompatible. Drop them so they reinitialize; the rest warm-starts.
    sr_changed = args.spatial_reduction != base_spatial_reduction
    if sr_changed:
        sd = {
            k: v
            for k, v in sd.items()
            if not (k.startswith("downsample.") or k.startswith("upsample."))
        }
        print(
            f"  spatial_reduction {base_spatial_reduction} -> "
            f"{args.spatial_reduction}: downsample/upsample reinitialized"
        )
    # head_depth change: cls_head/reg_head layer structure differs → reinitialize
    # both. (reg_head is also filtered on a reg_max change above; harmless overlap.)
    head_depth_changed = args.head_depth != base_head_depth
    if head_depth_changed:
        sd = {
            k: v
            for k, v in sd.items()
            if not (k.startswith("cls_head.") or k.startswith("reg_head."))
        }
        print(
            f"  head_depth {base_head_depth} -> {args.head_depth}: "
            "cls_head/reg_head reinitialized"
        )
    # per_channel_a from a shared-A base warm-starts via A_log broadcast; relax
    # strict so any incidental layout drift is tolerated alongside it.
    base_per_channel = mamba_state.get("mamba_args", {}).get("per_channel_a", False)
    strict = (
        not args.add_temporal
        and not (args.per_channel_a and not base_per_channel)
        and not reg_max_changed
        and not sr_changed
        and not head_depth_changed
        and mamba_args.get("reduction_variant", "conv")
        == mamba_state.get("mamba_args", {}).get("reduction_variant", "conv")
        and not use_detail_fusion
        and not use_strip_detail
    )
    missing, unexpected = mamba.load_state_dict(sd, strict=strict)
    mamba.to(device)  # ensure new temporal keys are on correct device
    if args.detail_source == "native-p3" and mamba.detail_fusion is not None:
        for parameter in mamba.detail_fusion.encoder.parameters():
            parameter.requires_grad_(False)
        print(
            "  Detail initialization: external P3 adapter/gates=random, "
            "cls/reg projection=zero-init, shallow encoder=frozen/unused"
        )
    if (
        use_detail_fusion
        and args.detail_source != "native-p3"
        and not any(k.startswith("detail_fusion.") for k in sd)
    ):
        copied = mamba.initialize_detail_from_yolo_stem(teacher.yolo_model.model[0])
        init_name = "YOLO Conv+BN" if copied else "random"
        print(f"  Detail encoder initialization: {init_name}")
    if missing:
        temporal_keys = [k for k in missing if "temporal" in k or "flow_gate" in k]
        detail_keys = [k for k in missing if k.startswith("detail_fusion.")]
        strip_keys = [k for k in missing if k.startswith("strip_detail_fusion.")]
        reduction_keys = [
            k
            for k in missing
            if k.startswith("downsample.")
            and mamba_args.get("reduction_variant", "conv") != "conv"
        ]
        expected_keys = set(temporal_keys + detail_keys + strip_keys + reduction_keys)
        unexpected_missing = [k for k in missing if k not in expected_keys]
        if temporal_keys:
            print(f"  New temporal keys (random init): {len(temporal_keys)}")
        if detail_keys:
            print(f"  New detail keys (expected warm-start): {len(detail_keys)}")
        if strip_keys:
            print(f"  New strip detail keys (random/zero init): {len(strip_keys)}")
        if reduction_keys:
            print(f"  New reduction keys (random init): {len(reduction_keys)}")
        if unexpected_missing:
            print(f"  WARNING: unexpected missing keys: {unexpected_missing}")
    if unexpected:
        stale_reduction = [
            k
            for k in unexpected
            if k.startswith("downsample.")
            and mamba_args.get("reduction_variant", "conv") != "conv"
        ]
        unexpected_other = [k for k in unexpected if k not in set(stale_reduction)]
        if stale_reduction:
            print(f"  Replaced reduction checkpoint keys: {len(stale_reduction)}")
        if unexpected_other:
            print(f"  WARNING: unexpected checkpoint keys: {unexpected_other}")
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
    print(f"  FPN channels: {in_channels}")

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
    # The criterion inherits reg_max from the teacher's Detect head (reg_max=1).
    # When the student head uses DFL (reg_max>1), switch the loss to distributional
    # box regression so it decodes the reg_max*4 channels via softmax-expectation
    # (bbox_decode's use_dfl path) instead of treating them as direct distances.
    if args.reg_max != criterion.reg_max:
        from ultralytics.utils.loss import BboxLoss

        criterion.reg_max = args.reg_max
        criterion.use_dfl = args.reg_max > 1
        criterion.no = nc + args.reg_max * 4
        criterion.proj = torch.arange(args.reg_max, dtype=torch.float, device=device)
        criterion.bbox_loss = BboxLoss(args.reg_max).to(device)
        print(
            f"  Loss: reg_max={args.reg_max} use_dfl={criterion.use_dfl} "
            "(overrode teacher's reg_max)"
        )
    install_assigner(criterion, args.assigner)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    start_epoch = 1
    best_loss = float("inf")
    resume_ckpt: dict[str, Any] | None = None

    if args.resume:
        resume_ckpt = torch.load(
            project_root / args.resume, map_location="cpu", weights_only=False
        )
        sd_resume = _strip_compiled_keys(resume_ckpt["student"])
        mamba.load_state_dict(sd_resume, strict=True)
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        best_loss = resume_ckpt.get("best_loss", float("inf"))
        print(f"[Resume] epoch={start_epoch}  best_loss={best_loss:.4f}")

    optimizer = torch.optim.AdamW(mamba.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule_epochs = (
        max(args.epochs - start_epoch + 1, 1)
        if args.resume_reset_optimizer
        else args.epochs
    )
    schedule_warmup = min(args.warmup_epochs, schedule_epochs)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=schedule_epochs,
        warmup_epochs=schedule_warmup,
    )

    if resume_ckpt is not None:
        if args.resume_reset_optimizer:
            print(
                f"[Resume] Reset optimizer/scheduler at lr={args.lr:.2e} for "
                f"{schedule_epochs} remaining epoch(s)"
            )
        else:
            missing_state = [
                key for key in ("optimizer", "scheduler") if key not in resume_ckpt
            ]
            if missing_state:
                raise ValueError(
                    "Exact resume requires optimizer and scheduler state; missing "
                    f"{missing_state}. Use --resume-reset-optimizer for legacy "
                    "checkpoints or a new LR schedule."
                )
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            loaded_lr = optimizer.param_groups[0]["lr"]
            print(f"[Resume] Restored optimizer/scheduler at lr={loaded_lr:.2e}")

    if args.compile:
        print("[Compile] torch.compile teacher (mode=default) — ~30s cold-start")
        teacher.yolo_model = torch.compile(teacher.yolo_model, mode="default")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root = project_root / args.data_root
    seqs, holdout_seqs = resolve_training_sequences(
        data_root, args.seqs, args.holdout_seqs
    )
    clip_stride = args.clip_stride or args.clip_len
    # Live (no --cache-dir): the backbone runs on raw frames, so images are
    # needed. RAM-preloading the whole clip set is the fast path, but a
    # full-cadence PersonPath22 set (~60k frames @640) is ~70 GB and OOMs a
    # 54 GB box — --no-preload-images falls back to per-step JPEG decode
    # (pair with --num-workers for parallel decode). In cache mode the backbone
    # never runs, so raw frame pixels are unused and preload is skipped anyway.
    live_mode = args.cache_dir == ""
    # GPU-decode: workers return raw bytes, so in-worker RAM-preload is skipped and
    # frames are decoded on the GPU in the loop (see gpu_decode_clip_batch).
    preload = live_mode and not args.no_preload_images and not args.gpu_decode
    detail_size = (
        (args.detail_height, args.detail_width)
        if args.detail_source in {"high", "native-p3", "strip-oracle"}
        else None
    )
    need_images = live_mode or detail_size is not None
    # Augmentation transforms the input frame, so it is only sound where the
    # backbone runs live on that frame (e.g. GT1). Under --cache-dir the
    # frozen-backbone features are keyed to fixed frames, so augmenting would
    # misalign inputs and supervision — gate it off and warn.
    augment = args.augment and live_mode
    if args.augment and not augment:
        print(
            "[MambaGT] WARNING: --augment ignored under --cache-dir — cached "
            "frozen-backbone features are keyed to fixed frames; augmentation "
            "only applies in live stages (no --cache-dir, e.g. GT1)."
        )
    loader = build_mot17_dataloader(
        data_root=data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        stride=clip_stride,
        shuffle=True,
        seqs=seqs,
        num_workers=args.num_workers,
        preload_to_ram=preload,
        load_images=need_images,
        detail_size=detail_size,
        seed=args.seed,
        augment=augment,
        interpolate_gt=args.interpolate_gt,
        return_jpeg_bytes=args.gpu_decode,
    )
    if args.gpu_decode:
        print(
            "[MambaGT] GPU-decode ACTIVE: workers return JPEG bytes, nvJPEG decode in-loop"
        )
    if augment:
        print("[MambaGT] augmentation ACTIVE (live stage, load_images=True)")
    if args.interpolate_gt:
        print(
            "[MambaGT] keyframe interpolation ACTIVE: full-cadence frames, "
            "linear-interp GT between keyframes, loss masked to real keyframes"
        )
    print(
        f"[MambaGT] {len(loader)} batches/epoch  gt_ratio={args.gt_ratio}  "
        f"lr={args.lr}  detail={args.detail_source}  seed={args.seed}  "
        f"clip_stride={clip_stride}"
    )
    if holdout_seqs:
        print(
            f"[Split] holdout={','.join(holdout_seqs)} "
            "(external detector recall/HOTA selection)"
        )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        if not cache_dir.is_absolute():
            cache_dir = project_root / cache_dir
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"Missing teacher-cache manifest: {manifest_path}. "
                "Rebuild the cache for this YOLO/teacher lineage."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_channels = tuple(manifest.get("fpn_channels", in_channels))
        if cached_channels != in_channels:
            raise ValueError(
                "Teacher-cache FPN channels do not match the Mamba checkpoint: "
                f"cache={cached_channels}, checkpoint={in_channels}"
            )
        if manifest.get("base_yolo_sha256") != actual_yolo_sha:
            raise ValueError(
                "Teacher cache was built from different YOLO weights: "
                f"cache={manifest.get('base_yolo_path')!r}, "
                f"selected={args.yolo_weights!r}"
            )
        if manifest.get("teacher_checkpoint_sha256") != actual_teacher_sha:
            raise ValueError(
                "Teacher cache checkpoint does not match --teacher-ckpt: "
                f"cache={manifest.get('teacher_checkpoint_path')!r}, "
                f"selected={args.teacher_ckpt!r}"
            )
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
        epoch_lr = optimizer.param_groups[0]["lr"]
        epoch_total_loss = 0.0
        epoch_cons_sum = 0.0
        epoch_cons_n = 0
        epoch_t1_sum = 0.0
        epoch_t1_n = 0
        n_steps = 0
        acc_batches = 0
        t0 = time.perf_counter()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            if (
                args.max_batches_per_epoch > 0
                and batch_idx >= args.max_batches_per_epoch
            ):
                break
            if args.gpu_decode:
                frames = (
                    gpu_decode_clip_batch(batch["jpeg_bytes"], args.img_size, device)
                    / 255.0
                )
            else:
                frames = batch["frames"].to(device, dtype=torch.float32) / 255.0
            gt_boxes_batch = batch["gt_boxes"]
            gt_ids_batch = batch["gt_ids"]
            B, T = frames.shape[:2]
            # Keyframe loss mask (interpolate_gt): the model forwards every frame at
            # the real cadence so the temporal SSM sees true displacement, but the
            # supervised loss is applied only on frames carrying real GT — boxes on
            # in-between frames are linearly interpolated (approximate) and would
            # otherwise teach the head against noisy targets. Default: every frame
            # is a keyframe → no masking (bit-exact).
            is_keyframe_batch = batch.get("is_keyframe", [[1] * T for _ in range(B)])
            detail_frames = None
            detail_valid_hw = None
            if args.detail_source == "global":
                detail_frames = frames
                detail_valid_hw = torch.full(
                    (B, T, 2),
                    args.img_size,
                    device=device,
                    dtype=torch.int64,
                )
            elif args.detail_source == "high":
                detail_frames = (
                    batch["detail_frames"].to(device, dtype=torch.float32) / 255.0
                )
                detail_valid_hw = batch["detail_valid_hw"].to(device)
            elif args.detail_source == "native-p3":
                detail_frames = (
                    batch["detail_frames"].to(device, dtype=torch.float32) / 255.0
                )
                detail_valid_hw = batch["detail_valid_hw"].to(device)
            elif args.detail_source == "strip-oracle":
                # Keep uint8 on GPU; the strip gather reads <10% of pixels and
                # normalizes only the gathered strips (see _gather_strip_pixels).
                # Avoids materializing a 180 MiB float32 copy of the full clip.
                detail_frames = batch["detail_frames"].to(device)
                detail_valid_hw = batch["detail_valid_hw"].to(device)

            # Collect features for all T frames, then forward as one temporal clip.
            # This ensures temporal blocks see actual frame-to-frame context during
            # training (instead of per-frame independent passes that never activate
            # the temporal SSM). All-frames loss prevents temporal hallucination.
            all_feats: list[list[torch.Tensor]] = [[], [], []]
            all_detail_feats: list[torch.Tensor] = []
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

                if args.detail_source == "native-p3":
                    fpn_feats.clear()
                    _ = teacher(detail_frames[:, t], gate_input=None)
                    all_detail_feats.append(fpn_feats["p3"])

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

            detail_T = None
            detail_features_T = None
            detail_valid_hw_T = None
            detail_positions_T = None
            detail_position_mask_T = None
            if detail_frames is not None and detail_valid_hw is not None:
                if args.detail_source == "native-p3":
                    detail_features_T = torch.stack(all_detail_feats, dim=1).reshape(
                        B * T, *all_detail_feats[0].shape[1:]
                    )
                else:
                    detail_T = detail_frames.reshape(B * T, *detail_frames.shape[2:])
                detail_valid_hw_T = detail_valid_hw.reshape(B * T, 2)
            if args.detail_source == "strip-oracle":
                flattened_gt = [
                    gt_boxes_batch[b][t] for b in range(B) for t in range(T)
                ]
                detail_positions_T, detail_position_mask_T = (
                    build_strip_oracle_positions(
                        flattened_gt,
                        p3_hw=tuple(feats_T[0].shape[-2:]),
                        img_size=args.img_size,
                        budget=args.strip_route_budget,
                        small_threshold=args.strip_small_threshold,
                        device=device,
                    )
                )

            # Temporal forward — T_frames activates temporal blocks
            s_cls_T, s_reg_T = mamba(
                feats_T,
                T=T,
                detail_images=detail_T,
                detail_valid_hw=detail_valid_hw_T,
                detail_features=detail_features_T,
                detail_feature_stride=8,
                detail_positions=detail_positions_T,
                detail_position_mask=detail_position_mask_T,
            )

            # All-frames loss: every frame supervised with its own GT.
            # Batch-major layout: [b0t0, b0t1, ..., b0tT, b1t0, ...]
            # Frame t for all B items: indices t, T+t, 2T+t, ... → c[t::T]
            batch_loss = frames.new_zeros(())
            n_kf_frames = 0
            for t in range(T):
                # Per-sample keyframe mask: supervise only batch items whose frame t
                # carries real GT (slice preds/feats/GT to that subset so interpolated
                # frames contribute neither loss nor background-FP pressure).
                keep = [b for b in range(B) if is_keyframe_batch[b][t]]
                if not keep:
                    continue
                s_cls_t = [c[t::T][keep] for c in s_cls_T]
                s_reg_t = [r[t::T][keep] for r in s_reg_T]
                feats_t = [all_feats[si][t][keep] for si in range(3)]
                gt_t = [all_gt[t][b] for b in keep]
                preds = _build_preds_dict(s_cls_t, s_reg_t, feats_t)
                yolo_batch = _make_yolo_batch(gt_t, args.img_size, device)
                step_loss_vec, _ = criterion(preds, yolo_batch)
                batch_loss = batch_loss + step_loss_vec.sum()
                n_kf_frames += 1
            batch_loss = batch_loss / (max(n_kf_frames, 1) * accum)

            if args.t1_weight > 0.0 and T > 1:
                # Joint loss (route 2): also supervise the pure-spatial T=1 forward
                # (temporal blocks bypassed) on the same frames, so the spatial path
                # stays deploy-good while the T-clip loss keeps cross-frame pressure —
                # simultaneously every step, not sequentially (vs T3→T1 staging). The
                # B*T frames are treated as independent single-frame samples; GT is
                # flattened batch-major to match feats_T's [b0t0,b0t1,...,b1t0,...].
                s_cls_1, s_reg_1 = mamba(
                    feats_T,
                    T=1,
                    detail_images=detail_T,
                    detail_valid_hw=detail_valid_hw_T,
                    detail_features=detail_features_T,
                    detail_feature_stride=8,
                    detail_positions=detail_positions_T,
                    detail_position_mask=detail_position_mask_T,
                )
                # Mask to keyframe samples in the flattened B*T layout
                # (k = b*T + t matches feats_T's batch-major reshape).
                keep_flat = [
                    b * T + t
                    for b in range(B)
                    for t in range(T)
                    if is_keyframe_batch[b][t]
                ]
                if keep_flat:
                    flat_gt = [
                        all_gt[t][b]
                        for b in range(B)
                        for t in range(T)
                        if is_keyframe_batch[b][t]
                    ]
                    s_cls_1k = [c[keep_flat] for c in s_cls_1]
                    s_reg_1k = [r[keep_flat] for r in s_reg_1]
                    feats_1k = [f[keep_flat] for f in feats_T]
                    preds_1 = _build_preds_dict(s_cls_1k, s_reg_1k, feats_1k)
                    yolo_batch_1 = _make_yolo_batch(flat_gt, args.img_size, device)
                    t1_loss_vec, _ = criterion(preds_1, yolo_batch_1)
                    t1_loss = t1_loss_vec.sum() / len(keep_flat)
                    batch_loss = batch_loss + (args.t1_weight * t1_loss) / accum
                    epoch_t1_sum += t1_loss.detach().item()
                    epoch_t1_n += 1

            if args.consistency_weight > 0.0 and T > 1:
                cons_loss = _temporal_consistency_loss(
                    s_cls_T, gt_boxes_batch, gt_ids_batch, T, B, args.img_size
                )
                batch_loss = batch_loss + (args.consistency_weight * cons_loss) / accum
                epoch_cons_sum += cons_loss.detach().item()
                epoch_cons_n += 1

            if not batch_loss.requires_grad:
                # No supervised signal this batch: every frame in the B*T clip is
                # interpolated (no keyframe anywhere) and no aux loss (t1/cons)
                # contributed a grad term, so batch_loss is a grad-less constant.
                # Happens under --interpolate-gt when clip_len spans a keyframe gap
                # (PP22 ~5-frame cadence → a 4-frame clip can miss all keyframes).
                # Nothing to backprop — skip this batch.
                continue

            if not torch.isfinite(batch_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch} batch={batch_idx + 1}"
                )
            batch_loss.backward()
            acc_batches += 1

            if acc_batches == accum:
                nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                acc_batches = 0

            step_loss = batch_loss.detach().item() * accum
            epoch_total_loss += step_loss
            n_steps += 1

            effective_batches = (
                min(n_batches, args.max_batches_per_epoch)
                if args.max_batches_per_epoch > 0
                else n_batches
            )
            eta_s = (
                (effective_batches - batch_idx - 1)
                * (time.perf_counter() - t0)
                / max(batch_idx + 1, 1)
            )
            pct = (batch_idx + 1) / effective_batches * 100
            print(
                f"\r  epoch {epoch:3d}/{args.epochs}  "
                f"[{pct:5.1f}%  {batch_idx + 1:4d}/{effective_batches}]  "
                f"loss={step_loss:7.2f}  ETA {eta_s:5.0f}s",
                end="",
                flush=True,
            )

        if acc_batches > 0:
            nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = epoch_total_loss / max(n_steps, 1)
        dt = time.perf_counter() - t0
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        is_best = args.best_by == "train-loss" and avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)
        best_marker = " [BEST]" if is_best else ""
        cons_marker = (
            f"  cons={epoch_cons_sum / epoch_cons_n:.3f}(×{args.consistency_weight:g})"
            if epoch_cons_n > 0
            else ""
        )
        t1_marker = (
            f"  t1={epoch_t1_sum / epoch_t1_n:.3f}(×{args.t1_weight:g})"
            if epoch_t1_n > 0
            else ""
        )
        print(
            f"\r  epoch {epoch:3d}/{args.epochs}  "
            f"loss={avg_loss:7.2f}{t1_marker}{cons_marker}  lr={epoch_lr:.2e}  "
            f"VRAM={vram_gb:.1f}GB  {dt / 60:.1f}min{best_marker}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "student": mamba.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "epoch_loss": avg_loss,
                    "epoch_lr": epoch_lr,
                    "selection": {
                        "best_by": args.best_by,
                        "holdout_seqs": holdout_seqs,
                        "status": (
                            "diagnostic_train_loss"
                            if args.best_by == "train-loss"
                            else "candidate_requires_external_selection"
                        ),
                    },
                    "args": vars(args),
                    "mamba_args": {
                        "in_channels": list(in_channels),
                        "base_yolo_path": mamba_args.get(
                            "base_yolo_path", args.yolo_weights
                        ),
                        "base_yolo_sha256": mamba_args["base_yolo_sha256"],
                        "teacher_checkpoint_sha256": mamba_args.get(
                            "teacher_checkpoint_sha256", ""
                        ),
                        "d_model": mamba_args["d_model"],
                        "d_state": mamba_args["d_state"],
                        "num_blocks": mamba_args["num_blocks"],
                        "reg_max": mamba_args["reg_max"],
                        "head_depth": mamba_args["head_depth"],
                        "spatial_reduction": mamba_args["spatial_reduction"],
                        "reduction_variant": mamba_args.get(
                            "reduction_variant", "conv"
                        ),
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
                        "scan_stop_grad": mamba_args.get("scan_stop_grad", False),
                        "legacy_n1_scan": mamba_args.get("legacy_n1_scan", False),
                        "legacy_n1_source": mamba_args.get("legacy_n1_source", ""),
                        "per_channel_a": mamba_args.get("per_channel_a", False),
                        "use_detail_fusion": mamba_args.get("use_detail_fusion", False),
                        "use_strip_detail": mamba_args.get("use_strip_detail", False),
                        "detail_source": mamba_args.get("detail_source", "none"),
                        "detail_channels": mamba_args.get("detail_channels", 32),
                        "detail_patch_size": mamba_args.get("detail_patch_size", 3),
                        "detail_feature_channels": mamba_args.get(
                            "detail_feature_channels", 0
                        ),
                        "detail_height": mamba_args.get(
                            "detail_height", args.detail_height
                        ),
                        "detail_width": mamba_args.get(
                            "detail_width", args.detail_width
                        ),
                        "strip_route_budget": mamba_args.get(
                            "strip_route_budget", args.strip_route_budget
                        ),
                        "strip_small_threshold": mamba_args.get(
                            "strip_small_threshold", args.strip_small_threshold
                        ),
                        "strip_length": mamba_args.get(
                            "strip_length", args.strip_length
                        ),
                        "strip_width": mamba_args.get("strip_width", args.strip_width),
                        "strip_detail_type": mamba_args.get(
                            "strip_detail_type", "mamba"
                        ),
                        "strip_stem_channels": mamba_args.get(
                            "strip_stem_channels", args.strip_stem_channels
                        ),
                        "strip_route_chunk_size": mamba_args.get(
                            "strip_route_chunk_size", args.strip_route_chunk_size
                        ),
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
