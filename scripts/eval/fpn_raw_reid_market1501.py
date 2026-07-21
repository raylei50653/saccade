#!/usr/bin/env python3
"""Market-1501 ReID evaluation — raw YOLO FPN features (zero-training baseline).

Extracts center-pixel features from YOLO6s FPN layers (P3/P4/P5),
concatenates, L2-normalizes. No extra head, no training, no projector.

Usage:
    uv run scripts/eval/fpn_raw_reid_market1501.py \
        --market-root datasets/Market-1501-v15.09.15 \
        --batch-size 32
"""
# status: experiment

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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
from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    DataPreloader,
    resize_stretch_batch_gpu,
)

IMG_SIZE = 640


def _parse_pid(filename: str) -> int:
    return int(Path(filename).stem.split("_")[0])


def _parse_cam(filename: str) -> int:
    parts = Path(filename).stem.split("_")
    for p in parts[1:]:
        if p.startswith("c"):
            cam_str = p[1:]
            s_idx = cam_str.find("s")
            if s_idx > 0:
                cam_str = cam_str[:s_idx]
            return int(cam_str)
    return -1


def _get_teacher_feats(
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
    fpn_indices = [_GATE_LAYER_IDX[s] for s in ("p3", "p4", "p5")]
    return [y[i] for i in fpn_indices]


def compute_embeddings(
    preloader: DataPreloader,
    paths: list[Path],
    img_size: int,
    batch_size: int,
    device: torch.device,
    teacher: nn.Module,
    yolo_model: nn.Module,
) -> np.ndarray:
    """Center-pixel extract from raw FPN (P3+P4+P5) → concat → L2-norm."""
    n = len(paths)

    # Probe feature dimensions
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    feats = _get_teacher_feats(teacher, yolo_model, dummy)
    fpn_dims = [f.shape[1] for f in feats]
    total_dim = sum(fpn_dims)
    print(
        f"  FPN dims: P3={fpn_dims[0]} P4={fpn_dims[1]} P5={fpn_dims[2]} total={total_dim}"
    )

    all_embs = np.empty((n, total_dim), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs_uint8 = torch.stack([preloader[p] for p in batch_paths])
        images = resize_stretch_batch_gpu(imgs_uint8, img_size, device)

        with torch.no_grad():
            feats = _get_teacher_feats(teacher, yolo_model, images)

        parts = []
        for f in feats:
            h, w = f.shape[2], f.shape[3]
            center = f[:, :, h // 2, w // 2]
            parts.append(center)
        pooled = torch.cat(parts, dim=1)
        embeddings = F.normalize(pooled, dim=1)

        all_embs[i : i + batch_size] = embeddings.cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"  {min(i + batch_size, n):5d}/{n}", flush=True)

    return all_embs


def cheb_gr_score_matrix(
    q_embs: np.ndarray,
    g_embs: np.ndarray,
    *,
    cheb_lambda: float,
    gconv_layers: int,
    residual_gamma: float,
    k_max: int,
    fuse_lambda: float,
    device: torch.device,
) -> np.ndarray:
    """Cheb-GR re-ranked similarity scores [Nq, Ng] (higher = better)."""
    from saccade.perception.reid.cheb_gr import cheb_gr_rerank_distance

    q = torch.from_numpy(q_embs).to(device)
    g = torch.from_numpy(g_embs).to(device)
    dist = cheb_gr_rerank_distance(
        q,
        g,
        cheb_lambda=cheb_lambda,
        gconv_layers=gconv_layers,
        residual_gamma=residual_gamma,
        k_max=k_max,
        fuse_lambda=fuse_lambda,
    )
    return (-dist).cpu().numpy()


def compute_mAP_cmc(
    q_embs: np.ndarray,
    g_embs: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 50,
    score_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    n_q, _ = len(q_embs), len(g_embs)
    # `sim` is any higher-is-better score; default cosine, override for re-ranking.
    sim = score_matrix if score_matrix is not None else q_embs @ g_embs.T
    indices = np.argsort(-sim, axis=1)

    aps: list[float] = []
    cmc = np.zeros(max_rank, dtype=np.int32)

    for q_idx in range(n_q):
        pid = q_pids[q_idx]
        cam = q_cams[q_idx]

        ranked = indices[q_idx]
        good = (g_pids[ranked] == pid) & (g_cams[ranked] != cam)
        junk = (
            (g_pids[ranked] == -1)
            | (g_pids[ranked] == 0)
            | ((g_pids[ranked] == pid) & (g_cams[ranked] == cam))
        )

        valid = ~junk
        good_filtered = good[valid]
        good_cumsum = np.cumsum(good_filtered)
        for k in range(max_rank):
            if k < len(good_cumsum) and good_cumsum[k] > 0:
                cmc[k] += 1

        selected = valid
        n_good = good[selected].sum()
        if n_good == 0:
            continue
        scores = sim[q_idx, ranked[selected]]
        labels = good[selected].astype(np.float32)
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        tp = np.cumsum(labels_sorted)
        fp = np.cumsum(1 - labels_sorted)
        precision = tp / (tp + fp + 1e-12)
        ap = 0.0
        for t in np.where(labels_sorted == 1)[0]:
            ap += precision[t]
        ap /= n_good
        aps.append(ap)

    mAP = np.mean(aps) * 100 if aps else 0.0
    cmc_rates = {f"rank-{k + 1}": cmc[k] / n_q * 100 for k in [0, 4, 9, 19]}
    return {"mAP": mAP, **cmc_rates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-gallery-preload", action="store_true")
    # ── Cheb-GR re-ranking (CVPR 2025) ──────────────────────────────────────
    parser.add_argument(
        "--rerank",
        choices=["none", "cheb_gr"],
        default="none",
        help="Re-ranking method applied before mAP/CMC (default: none).",
    )
    parser.add_argument("--cheb-lambda", type=float, default=1.0)
    parser.add_argument("--gconv-layers", type=int, default=2)
    parser.add_argument("--residual-gamma", type=float, default=0.0)
    parser.add_argument("--k-max", type=int, default=0)
    parser.add_argument("--fuse-lambda", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    num_workers = args.workers or min(os.cpu_count() or 4, 16)

    teacher_ckpt = project_root / args.teacher_ckpt
    mamba_ckpt = project_root / args.mamba_ckpt
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root

    print(f"Device: {device}")

    # Load data
    query_dir = market_root / "query"
    gallery_dir = market_root / "bounding_box_test"
    query_paths = sorted(query_dir.glob("*.jpg"))
    gallery_paths = sorted(gallery_dir.glob("*.jpg"))

    q_pids = np.array([_parse_pid(p.name) for p in query_paths], dtype=np.int32)
    g_pids = np.array([_parse_pid(p.name) for p in gallery_paths], dtype=np.int32)
    q_cams = np.array([_parse_cam(p.name) for p in query_paths], dtype=np.int32)
    g_cams = np.array([_parse_cam(p.name) for p in gallery_paths], dtype=np.int32)

    valid_q = q_pids > 0
    n_distractors = (g_pids <= 0).sum()
    print(
        f"Queries: {valid_q.sum()}/{len(query_paths)}  "
        f"Gallery: {len(gallery_paths)} (distractors: {n_distractors})"
    )
    print(
        f"IDs in query: {len(np.unique(q_pids[valid_q]))}  "
        f"IDs in gallery: {len(np.unique(g_pids[g_pids > 0]))}"
    )

    # Preload
    print("\n[Phase 1] Preloading images...")
    t0 = time.perf_counter()
    q_preloader = DataPreloader(query_paths, num_workers=num_workers)
    q_preloader.load()
    if args.no_gallery_preload:
        g_preloader = DataPreloader(gallery_paths, num_workers=num_workers)
        g_preloader.load()
    else:
        all_paths = query_paths + gallery_paths
        preloader = DataPreloader(all_paths, num_workers=num_workers)
        preloader.load()
        q_preloader = preloader
        g_preloader = preloader
    print(f"  Total preload: {time.perf_counter() - t0:.1f}s")

    # Build model (only need backbone + FPN, no Mamba)
    print("\n[Phase 2] Building model (FPN backbone only)...")
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

    # Compute embeddings
    print("\n[Phase 3] Computing query embeddings (raw FPN, no head)...")
    t1 = time.perf_counter()
    q_embs = compute_embeddings(
        q_preloader,
        query_paths,
        IMG_SIZE,
        args.batch_size,
        device,
        teacher,
        yolo_model,
    )
    print(
        f"  {q_embs.shape[0]} queries, {q_embs.shape[1]}-dim in {time.perf_counter() - t1:.1f}s"
    )

    print("\n[Phase 4] Computing gallery embeddings (raw FPN, no head)...")
    t1 = time.perf_counter()
    g_embs = compute_embeddings(
        g_preloader,
        gallery_paths,
        IMG_SIZE,
        args.batch_size,
        device,
        teacher,
        yolo_model,
    )
    print(
        f"  {g_embs.shape[0]} gallery, {g_embs.shape[1]}-dim in {time.perf_counter() - t1:.1f}s"
    )

    # Evaluate (always report the raw-cosine baseline)
    print("\n[Phase 5] Computing mAP and CMC...")
    t1 = time.perf_counter()
    metrics = compute_mAP_cmc(q_embs, g_embs, q_pids, g_pids, q_cams, g_cams)
    print(f"  Evaluation done in {time.perf_counter() - t1:.1f}s\n")

    print("=" * 50)
    print("Market-1501 ReID Results — Raw YOLO FPN (cosine baseline)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:6.2f}%")
    print("=" * 50)

    if args.rerank == "cheb_gr":
        print("\n[Phase 6] Cheb-GR re-ranking...")
        t1 = time.perf_counter()
        scores = cheb_gr_score_matrix(
            q_embs,
            g_embs,
            cheb_lambda=args.cheb_lambda,
            gconv_layers=args.gconv_layers,
            residual_gamma=args.residual_gamma,
            k_max=args.k_max,
            fuse_lambda=args.fuse_lambda,
            device=device,
        )
        rr_metrics = compute_mAP_cmc(
            q_embs, g_embs, q_pids, g_pids, q_cams, g_cams, score_matrix=scores
        )
        print(f"  Re-rank + eval done in {time.perf_counter() - t1:.1f}s\n")
        print("=" * 50)
        print(
            f"Market-1501 — Cheb-GR (lambda={args.cheb_lambda} L={args.gconv_layers} "
            f"gamma={args.residual_gamma} k_max={args.k_max} fuse={args.fuse_lambda})"
        )
        print("=" * 50)
        for k, v in rr_metrics.items():
            delta = v - metrics.get(k, 0.0)
            print(f"  {k:>10s}: {v:6.2f}%  ({delta:+.2f})")
        print("=" * 50)


if __name__ == "__main__":
    main()
