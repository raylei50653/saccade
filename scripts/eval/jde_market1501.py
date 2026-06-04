#!/usr/bin/env python3
"""Market-1501 ReID evaluation for JDE projector.

Computes mAP and CMC (rank-1/5/10/20) for the trained EmbeddingProjector.

Usage:
    uv run scripts/eval/jde_market1501.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --proj-ckpt runs/jde_market_v2/best.ckpt \
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

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.mamba_head import EmbeddingProjector  # noqa: E402
from saccade.perception.temporal_yolo.reid_conv_head import ReIDConvHead  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.temporal_yolo.yolo_gated_detector import _GATE_LAYER_IDX  # noqa: E402
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


def compute_embeddings(
    preloader: DataPreloader,
    paths: list[Path],
    img_size: int,
    batch_size: int,
    device: torch.device,
    teacher: nn.Module,
    yolo_model: nn.Module,
    mamba_head: nn.Module,
    projector: EmbeddingProjector,
    pool_mode: str = "global",
    reid_conv_head: nn.Module | None = None,
) -> np.ndarray:
    """Compute L2-normalized embeddings for all images."""
    n = len(paths)
    emb_dim = projector.out_dim
    all_embs = np.empty((n, emb_dim), dtype=np.float32)

    if reid_conv_head is not None:
        pool_fn = ReIDConvHead.pool_center
    elif pool_mode == "center":
        pool_fn = mamba_head.pool_embeddings_center
    else:
        pool_fn = mamba_head.pool_embeddings_global

    for i in range(0, n, batch_size):
        batch_paths = paths[i : i + batch_size]
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

        all_embs[i : i + batch_size] = embeddings.cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"  {min(i + batch_size, n):5d}/{n}", flush=True)

    return all_embs


def compute_mAP_cmc(
    q_embs: np.ndarray,
    g_embs: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 50,
) -> dict[str, float]:
    """Compute mAP and CMC for Market-1501 protocol.

    For each query, gallery images with the same PID and same camera
    are treated as junk (ignored in ranking). Distractors (pid==-1
    or pid==0) are also treated as junk.
    """
    n_q, _ = len(q_embs), len(g_embs)
    sim = q_embs @ g_embs.T  # (n_q, n_g) cosine similarity

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

        # CMC
        valid = ~junk
        good_filtered = good[valid]
        good_cumsum = np.cumsum(good_filtered)
        for k in range(max_rank):
            if k < len(good_cumsum) and good_cumsum[k] > 0:
                cmc[k] += 1

        # AP — exclude junk; use ALL gallery items
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
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--proj-ckpt", default="runs/jde_market_v2/best.ckpt")
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-gallery-preload", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    num_workers = args.workers or min(os.cpu_count() or 4, 16)

    mamba_ckpt = project_root / args.mamba_ckpt
    proj_ckpt = project_root / args.proj_ckpt
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root

    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Preload images
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preloading images...")
    t0 = time.perf_counter()

    # Always preload queries (small)
    q_preloader = DataPreloader(query_paths, num_workers=num_workers)
    q_preloader.load()

    if args.no_gallery_preload:
        g_preloader = DataPreloader(gallery_paths, num_workers=num_workers)
        g_preloader.load()
        print(f"  Total preload: {time.perf_counter() - t0:.1f}s")
    else:
        all_paths = query_paths + gallery_paths
        preloader = DataPreloader(all_paths, num_workers=num_workers)
        preloader.load()
        q_preloader = preloader
        g_preloader = preloader
        print(f"  Total preload: {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print("\n[Phase 2] Building model...")
    detector = build_mamba_gated_detector(
        yolo_pt_path=str((project_root / "models/yolo/yolo26s.pt").resolve()),
        teacher_ckpt="",
        mamba_ckpt=str(mamba_ckpt.resolve()),
        img_size=IMG_SIZE,
        device=device,
        emb_dim=128,
    )
    detector.eval()

    mamba_head = detector.mamba_head
    teacher = detector.teacher
    yolo_model = teacher.yolo_model

    proj_state = torch.load(proj_ckpt, map_location="cpu", weights_only=False)
    projector = EmbeddingProjector(
        emb_dim=mamba_head.emb_dim,
        hidden=256,
        out_dim=proj_state.get("emb_out_dim", 128),
    ).to(device)
    projector.load_state_dict(proj_state["projector"])
    projector.eval()

    if proj_state.get("emb_head") is not None:
        mamba_head.emb_head.load_state_dict(proj_state["emb_head"])
        mamba_head.emb_head.eval()
        print("  Loaded emb_head weights from JDE checkpoint")

    pool_mode = proj_state.get("pool_mode", "global")
    print(f"  Pool mode: {pool_mode}")

    reid_conv_head: nn.Module | None = None
    if proj_state.get("reid_conv_head") is not None:
        # Detect FPN channels from a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
            fpn_feats = _get_teacher_feats(teacher, yolo_model, dummy)
            fpn_channels = [f.shape[1] for f in fpn_feats]
        reid_conv_head = ReIDConvHead(
            fpn_channels,
            emb_dim=proj_state.get("emb_dim_per_scale", 128),
        ).to(device)
        reid_conv_head.load_state_dict(proj_state["reid_conv_head"])
        reid_conv_head.eval()
        print("  Loaded ReIDConvHead weights from JDE checkpoint")

    # ------------------------------------------------------------------
    # Compute embeddings
    # ------------------------------------------------------------------
    print("\n[Phase 3] Computing query embeddings...")
    t1 = time.perf_counter()
    q_embs = compute_embeddings(
        q_preloader,
        query_paths,
        IMG_SIZE,
        args.batch_size,
        device,
        teacher,
        yolo_model,
        mamba_head,
        projector,
        pool_mode,
        reid_conv_head,
    )
    print(f"  {q_embs.shape[0]} queries in {time.perf_counter() - t1:.1f}s")

    print("\n[Phase 4] Computing gallery embeddings...")
    t1 = time.perf_counter()
    g_embs = compute_embeddings(
        g_preloader,
        gallery_paths,
        IMG_SIZE,
        args.batch_size,
        device,
        teacher,
        yolo_model,
        mamba_head,
        projector,
        pool_mode,
        reid_conv_head,
    )
    print(f"  {g_embs.shape[0]} gallery in {time.perf_counter() - t1:.1f}s")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n[Phase 5] Computing mAP and CMC...")
    t1 = time.perf_counter()
    metrics = compute_mAP_cmc(
        q_embs,
        g_embs,
        q_pids,
        g_pids,
        q_cams,
        g_cams,
    )
    print(f"  Evaluation done in {time.perf_counter() - t1:.1f}s\n")

    print("=" * 50)
    print("Market-1501 ReID Results (JDE Projector)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:6.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
