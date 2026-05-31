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
import torch.multiprocessing
from torch.nn import functional as F

# Use file_system shared memory strategy to avoid hitting Linux's open-file-descriptor
# limit (EMFILE/ulimit) when spawning DataLoader workers on systems with low fd limits.
torch.multiprocessing.set_sharing_strategy("file_system")

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
# Feature precomputation
# ---------------------------------------------------------------------------
def _precompute_teacher_features(
    loader: Any,
    out_dir: Path,
    device: torch.device,
    teacher: nn.Module,
    fpn_feats: dict[str, torch.Tensor],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset: Any = loader.dataset
    img_cache: dict = dataset._img_cache
    frame_lists: dict = dataset._frame_lists
    sequences: list[str] = dataset.sequences

    total = sum(len(paths) for paths in frame_lists.values())
    done = 0
    skipped = 0
    t0 = time.time()

    for seq in sequences:
        (out_dir / seq).mkdir(parents=True, exist_ok=True)
        seq_imgs = img_cache[seq]
        seq_paths = frame_lists[seq]

        for i, fpath in enumerate(seq_paths):
            fid = int(fpath.stem)
            cache_path = out_dir / seq / f"{fid:06d}.pt"
            if cache_path.exists():
                skipped += 1
                continue

            img = seq_imgs[i].float().to(device) / 255.0
            img = img.unsqueeze(0)

            with torch.no_grad():
                fpn_feats.clear()
                _ = teacher(img, gate_input=None)

            torch.save(
                {
                    "p3": fpn_feats["p3"].squeeze(0).cpu().half(),
                    "p4": fpn_feats["p4"].squeeze(0).cpu().half(),
                    "p5": fpn_feats["p5"].squeeze(0).cpu().half(),
                },
                cache_path,
            )

            done += 1
            if done % 100 == 0 or done == 1:
                elapsed = time.time() - t0
                fps = done / elapsed if elapsed > 0 else 0
                print(
                    f"  [Precompute] {done}/{total}  ({fps:.1f} fps)  "
                    f"ETA: {(total - done) / fps:.0f}s"
                    if fps > 0
                    else ""
                )

    elapsed = time.time() - t0
    print(
        f"  [Precompute] Done: {done} new + {skipped} skipped in {elapsed:.0f}s "
        f"-> {out_dir}"
    )


# ---------------------------------------------------------------------------
# Feature loading helper (shared by temporal and per-frame training)
# ---------------------------------------------------------------------------
_feature_ram_cache: dict[
    str, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
] = {}
_gmc_cache: dict[str, torch.Tensor] = {}  # seq -> (N, 2, 3) affine matrices
_flow_grid_cache: dict[tuple[int, int], torch.Tensor] = {}  # (H,W) -> (3, H*W) coords
_affine_cum_cache: dict[
    tuple[str, int, int], torch.Tensor
] = {}  # (seq,fid_from,fid_to) -> (2,3)


def _get_flow_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    key = (H, W)
    if key not in _flow_grid_cache:
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(gx)
        _flow_grid_cache[key] = torch.stack([gx, gy, ones], dim=0).reshape(3, -1)
    return _flow_grid_cache[key].to(device)


def _preload_gmc_matrices(gmc_path: Path, device: torch.device) -> None:
    global _gmc_cache, _affine_cum_cache
    if _gmc_cache:
        return
    _gmc_cache = torch.load(gmc_path, map_location=device, weights_only=False)
    _affine_cum_cache = {}
    n_matrices = sum(m.shape[0] for m in _gmc_cache.values())
    print(
        f"[GMC] Loaded {n_matrices} affine matrices on {device} for {len(_gmc_cache)} sequences"
    )


def _get_cum_affine(
    seq: str, fid_from: int, fid_to: int, device: torch.device
) -> torch.Tensor | None:
    gmc_mats = _gmc_cache.get(seq)
    if gmc_mats is None or fid_from >= fid_to:
        return None
    n_mats = gmc_mats.shape[0]
    cache_key = (seq, fid_from, fid_to)
    if cache_key in _affine_cum_cache:
        return _affine_cum_cache[cache_key]
    cumulative = torch.eye(2, 3, device=device)
    for fid in range(fid_from, fid_to):
        idx = fid - 1
        if idx < 0 or idx >= n_mats:
            continue
        m = gmc_mats[idx]
        m3 = torch.eye(3, device=device)
        m3[:2, :] = m
        cumulative = m @ m3
    _affine_cum_cache[cache_key] = cumulative[:2]
    return cumulative[:2]


def _build_gmc_flow_batch(
    seq_names: list[str],
    fids_from: list[int],
    fids_to: list[int],
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    B = len(seq_names)
    affines = []
    for b in range(B):
        aff = _get_cum_affine(seq_names[b], fids_from[b], fids_to[b], device)
        if aff is None:
            aff = torch.eye(2, 3, device=device)
        affines.append(aff)
    affines_t = torch.stack(affines)  # (B, 2, 3) already on device
    coords = _get_flow_grid(H, W, device)  # (3, H*W)
    coords_expanded = coords.unsqueeze(0).expand(B, -1, -1)
    warped = torch.bmm(affines_t, coords_expanded)  # (B, 2, H*W)
    flow = warped - coords[:2].unsqueeze(0)
    return flow.reshape(B, 2, H, W)


def _warp_feat_with_flow(
    feat: torch.Tensor,  # (B, C, H, W)
    flow: torch.Tensor | None,  # (B, H, W, 2) or (H, W, 2)
) -> torch.Tensor:
    if flow is None:
        return feat
    B, C, H, W = feat.shape
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    flow = flow.to(feat.device, feat.dtype)
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype),
        torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    grid = base_grid + torch.stack(
        [flow[..., 0] / (W / 2), flow[..., 1] / (H / 2)], dim=-1
    )
    return F.grid_sample(
        feat, grid, align_corners=False, mode="bilinear", padding_mode="border"
    )


def _preload_feature_cache(cache_dir: Path, device: torch.device) -> None:
    global _feature_ram_cache
    if _feature_ram_cache:
        return
    _feature_ram_cache = {}
    seq_dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir())
    total = sum(1 for sd in seq_dirs for _ in sd.glob("*.pt"))
    done = 0
    t0 = time.time()

    # Automatically detect GPU VRAM to determine default preloading location.
    total_vram = 0
    if device.type == "cuda":
        try:
            total_vram = torch.cuda.get_device_properties(device).total_memory
        except Exception:
            pass

    # We need at least 20 GB of VRAM to hold 14.8 GB of features and train safely.
    # If total VRAM is less than 20 GB (e.g. 12 GB), cache in CPU RAM directly to prevent OOM.
    if total_vram > 0 and total_vram < 20 * 1024**3:
        print(
            f"[FeatureCache] GPU VRAM ({total_vram / 1024**3:.1f} GB) is < 20 GB. Defaulting to CPU RAM to prevent OOM/swapping."
        )
        target_device = torch.device("cpu")
    else:
        target_device = device

    print(f"[FeatureCache] Loading {total} feature files to {target_device}...")
    try:
        for sd in seq_dirs:
            seq_name = sd.name
            seq_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for pt_file in sorted(sd.glob("*.pt")):
                fid = int(pt_file.stem)
                feat = torch.load(
                    pt_file, map_location=target_device, weights_only=True
                )
                seq_cache[fid] = (feat["p3"], feat["p4"], feat["p5"])
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{total}", flush=True)
            _feature_ram_cache[seq_name] = seq_cache
        print(
            f"  {done}/{total} — loaded to {target_device} in {time.time() - t0:.0f}s",
            flush=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and target_device.type == "cuda":
            print(
                "[FeatureCache] GPU OOM! Releasing GPU VRAM and falling back to CPU RAM..."
            )
            _feature_ram_cache.clear()
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            done = 0
            t0 = time.time()
            for sd in seq_dirs:
                seq_name = sd.name
                seq_cache = {}
                for pt_file in sorted(sd.glob("*.pt")):
                    fid = int(pt_file.stem)
                    feat = torch.load(pt_file, map_location="cpu", weights_only=True)
                    seq_cache[fid] = (feat["p3"], feat["p4"], feat["p5"])
                    done += 1
                    if done % 500 == 0:
                        print(f"  {done}/{total}", flush=True)
                _feature_ram_cache[seq_name] = seq_cache
            print(
                f"  {done}/{total} — loaded to CPU RAM in {time.time() - t0:.0f}s",
                flush=True,
            )
        else:
            raise e


def _load_or_forward_feats(
    batch: dict[str, Any],
    t: int,
    cache_dir: Path | None,
    device: torch.device,
    teacher: nn.Module,
    fpn_feats: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    B = batch["frames"].shape[0]
    if cache_dir is not None and _feature_ram_cache:
        p3s, p4s, p5s = [], [], []
        for b in range(B):
            seq: str = batch["seq"][b]  # type: ignore[index]
            fid: int = batch["frame_ids"][b][t]  # type: ignore[index]
            p3, p4, p5 = _feature_ram_cache[seq][fid]
            p3s.append(p3)
            p4s.append(p4)
            p5s.append(p5)
        return [
            torch.stack(p3s).to(device, dtype=torch.float32),
            torch.stack(p4s).to(device, dtype=torch.float32),
            torch.stack(p5s).to(device, dtype=torch.float32),
        ]
    elif cache_dir is not None:
        p3s, p4s, p5s = [], [], []
        for b in range(B):
            seq = batch["seq"][b]  # type: ignore[index]
            fid = batch["frame_ids"][b][t]  # type: ignore[index]
            feat = torch.load(
                cache_dir / seq / f"{fid:06d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            p3s.append(feat["p3"])
            p4s.append(feat["p4"])
            p5s.append(feat["p5"])
        return [
            torch.stack(p3s).to(device, dtype=torch.float32),
            torch.stack(p4s).to(device, dtype=torch.float32),
            torch.stack(p5s).to(device, dtype=torch.float32),
        ]
    else:
        frame = batch["frames"][:, t].to(device, dtype=torch.float32) / 255.0
        fpn_feats.clear()
        with torch.no_grad():
            _ = teacher(frame, gate_input=None)
        return [fpn_feats[s] for s in ("p3", "p4", "p5")]


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
    parser.add_argument(
        "--use-pixel-shuffle",
        action="store_true",
        help="enable PixelShuffle upsampler in Mamba head",
    )
    parser.add_argument(
        "--use-cross-scan",
        action="store_true",
        help="enable 4-directional cross-scan in Mamba head",
    )
    parser.add_argument(
        "--use-hybrid-head",
        action="store_true",
        help="hybrid head: conv on P3, Mamba on P4/P5",
    )
    parser.add_argument(
        "--use-temporal-attention",
        action="store_true",
        help="T=4 self-attention over frames (replaces temporal SSM, no flow gate)",
    )
    parser.add_argument(
        "--use-temporal-mamba",
        action="store_true",
        help="spatio-temporal SSM across T frames",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--precompute-dir",
        default="",
        help="Precompute teacher FPN features for all frames and exit",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Use precomputed teacher FPN features (skip backbone forward pass)",
    )

    args = parser.parse_args()

    if args.use_temporal_mamba and args.clip_len < 2:
        args.clip_len = 3
        print(f"[Temporal] Auto-setting --clip-len={args.clip_len}")

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
        use_pixel_shuffle=args.use_pixel_shuffle,
        use_cross_scan=args.use_cross_scan,
        use_hybrid_head=args.use_hybrid_head,
        use_temporal_mamba=args.use_temporal_mamba,
        use_temporal_attention=getattr(args, "use_temporal_attention", False),
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
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            if scheduler.T_max != args.epochs:
                print(
                    f"[Resume] Overriding scheduler.T_max from {scheduler.T_max} to {args.epochs}"
                )
                scheduler.T_max = args.epochs
        else:
            scheduler.last_epoch = start_epoch - 1
        print(
            f"[Resume] epoch={start_epoch}  best_loss={best_loss:.4f}  initial_lr={optimizer.param_groups[0]['lr']:.2e}"
        )

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
        stride=1 if args.use_temporal_mamba else args.clip_len * 2,
        shuffle=not args.use_temporal_mamba,
    )

    # ------------------------------------------------------------------
    # Precompute mode: extract and cache teacher FPN features, then exit
    # ------------------------------------------------------------------
    if args.precompute_dir:
        _precompute_teacher_features(
            loader,
            project_root / args.precompute_dir,
            device,
            teacher,
            fpn_feats,
        )
        for h in _hooks:
            h.remove()
        return

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        print(f"[Cache] Loading precomputed teacher features from {cache_dir}")
        _preload_feature_cache(cache_dir, device)
        gmc_path = cache_dir / "gmc_matrices_cpp.pt"
        if not gmc_path.exists():
            gmc_path = cache_dir / "gmc_matrices.pt"
        if gmc_path.exists() and args.use_temporal_mamba:
            _preload_gmc_matrices(gmc_path, device)

    # Reference to teacher Detect head's per-scale branches
    detect_head = teacher.yolo_model.model[-1]

    print(f"\nTraining {args.epochs} epochs...")
    accum = args.accum_steps
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            B = batch["frames"].shape[0]
            T = batch["frames"].shape[1]

            batch_loss: torch.Tensor = torch.zeros((), device=device)

            if args.use_temporal_mamba:
                all_feats: list[list[torch.Tensor]] = [[], [], []]
                all_t_cls: list[list[torch.Tensor]] = [[], [], []]
                all_t_reg: list[list[torch.Tensor]] = [[], [], []]

                for t in range(T):
                    feats = _load_or_forward_feats(
                        batch, t, cache_dir, device, teacher, fpn_feats
                    )
                    for si in range(3):
                        all_feats[si].append(feats[si])
                        all_t_cls[si].append(detect_head.cv3[si](feats[si]))
                        all_t_reg[si].append(detect_head.cv2[si](feats[si]))

                if _gmc_cache:
                    last_t = T - 1
                    all_flows: list[list[torch.Tensor]] = [[], [], []]
                    for t in range(T):
                        for si in range(3):
                            seq_names = [batch["seq"][b] for b in range(B)]
                            fids_from = [
                                int(batch["frame_ids"][b][t]) for b in range(B)
                            ]
                            fids_to = [
                                int(batch["frame_ids"][b][last_t]) for b in range(B)
                            ]
                            h_sc, w_sc = (
                                all_feats[si][t].shape[2],
                                all_feats[si][t].shape[3],
                            )
                            flow_batch = _build_gmc_flow_batch(
                                seq_names, fids_from, fids_to, h_sc, w_sc, device
                            )
                            all_flows[si].append(flow_batch)

                student_input = [torch.cat(all_feats[si], dim=0) for si in range(3)]
                flow_input = (
                    [torch.cat(all_flows[si], dim=0) for si in range(3)]
                    if _gmc_cache
                    else None
                )
                s_cls, s_reg = student(student_input, T=T, flows=flow_input)

                for t in range(T):
                    start_idx, end_idx = t * B, (t + 1) * B
                    for si in range(3):
                        batch_loss = batch_loss + (
                            nn.functional.mse_loss(
                                s_cls[si][start_idx:end_idx], all_t_cls[si][t]
                            )
                            + nn.functional.mse_loss(
                                s_reg[si][start_idx:end_idx], all_t_reg[si][t]
                            )
                        )
            else:
                for t in range(T):
                    feats = _load_or_forward_feats(
                        batch, t, cache_dir, device, teacher, fpn_feats
                    )

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
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "mamba_args": {
                    "d_model": args.d_model,
                    "d_state": args.d_state,
                    "num_blocks": args.num_blocks,
                    "spatial_reduction": args.spatial_reduction,
                    "num_classes": nc,
                    "use_pixel_shuffle": args.use_pixel_shuffle,
                    "use_cross_scan": args.use_cross_scan,
                    "use_hybrid_head": args.use_hybrid_head,
                    "use_temporal_mamba": args.use_temporal_mamba,
                    "use_temporal_attention": getattr(
                        args, "use_temporal_attention", False
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
