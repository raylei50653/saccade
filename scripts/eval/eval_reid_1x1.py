#!/usr/bin/env python3
"""Market-1501 ReID eval for 1×1 Conv dimension-reduction heads.

Loads DimReduceHead from a v9 checkpoint and evaluates rank-1/mAP.

Usage:
    uv run scripts/eval/eval_reid_1x1.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --head-ckpt runs/jde_market_v9a/best.ckpt \
        --market-root datasets/Market-1501-v15.09.15
"""

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


class DimReduceHead(nn.Module):
    def __init__(self, in_channels: list[int], out_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.nl = len(in_channels)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_dim, 1, bias=False) for c in in_channels]
        )
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


def _get_fpn_feats(teacher, yolo_model, frame):
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


def compute_mAP_cmc(q_embs, g_embs, q_pids, g_pids, q_cams, g_cams, max_rank=50):
    n_q = len(q_embs)
    sim = q_embs @ g_embs.T
    indices = np.argsort(-sim, axis=1)
    aps, cmc = [], np.zeros(max_rank, dtype=np.int32)

    for q_idx in range(n_q):
        pid, cam = q_pids[q_idx], q_cams[q_idx]
        ranked = indices[q_idx]
        good = (g_pids[ranked] == pid) & (g_cams[ranked] != cam)
        junk = (
            (g_pids[ranked] == -1)
            | (g_pids[ranked] == 0)
            | ((g_pids[ranked] == pid) & (g_cams[ranked] == cam))
        )

        ng = good.sum()
        if ng == 0:
            continue
        valid = ~junk
        good_cumsum = np.cumsum(good & valid)
        hit_ranks = np.where(good_cumsum > 0, good_cumsum, 0)
        for k in range(max_rank):
            if hit_ranks[k] > 0:
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
        ap = sum(precision[t] for t in np.where(labels_sorted == 1)[0]) / n_good
        aps.append(ap)

    mAP = np.mean(aps) * 100 if aps else 0.0
    cmc_rates = {f"rank-{k + 1}": cmc[k] / n_q * 100 for k in [0, 4, 9, 19]}
    return {"mAP": mAP, **cmc_rates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--head-ckpt", required=True)
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    num_workers = args.workers or min(os.cpu_count() or 4, 16)

    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root
    head_ckpt_path = Path(args.head_ckpt)
    if not head_ckpt_path.is_absolute():
        head_ckpt_path = project_root / head_ckpt_path

    # Load checkpoint
    ckpt = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
    scales = ckpt.get("scales", "p5")
    in_channels = ckpt.get("in_channels", [512])
    print(f"Device: {device}  Scales: {scales}  In channels: {in_channels}")

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

    # Preload
    print("\nPreloading images...")
    t0 = time.perf_counter()
    all_paths = query_paths + gallery_paths
    preloader = DataPreloader(all_paths, num_workers=num_workers)
    preloader.load()
    print(f"  {len(all_paths)} images in {time.perf_counter() - t0:.1f}s")

    # Build detector (for FPN access)
    print("\nBuilding detector...")
    detector = build_mamba_gated_detector(
        yolo_pt_path=str((project_root / args.yolo_weights).resolve()),
        teacher_ckpt=str((project_root / args.teacher_ckpt).resolve()),
        mamba_ckpt=str((project_root / args.mamba_ckpt).resolve()),
        img_size=IMG_SIZE,
        device=device,
        emb_dim=128,
    )
    detector.eval()
    teacher = detector.teacher
    yolo_model = teacher.yolo_model

    # Build head
    head = DimReduceHead(in_channels, out_dim=128).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()
    print(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")

    # Compute embeddings
    all_paths_list = query_paths + gallery_paths
    n_total = len(all_paths_list)
    emb_dim = 128
    all_embs = np.empty((n_total, emb_dim), dtype=np.float32)

    print(f"\nComputing embeddings ({n_total} images)...")
    for i in range(0, n_total, args.batch_size):
        batch_paths = all_paths_list[i : i + args.batch_size]
        imgs_uint8 = torch.stack([preloader[p] for p in batch_paths]).to(device)
        frame_640 = resize_stretch_batch_gpu(imgs_uint8, IMG_SIZE, device)

        with torch.no_grad():
            fpn = _get_fpn_feats(teacher, yolo_model, frame_640.float())
        if scales == "p5":
            fpn = [fpn[2]]

        with torch.no_grad():
            embeddings = head(fpn)
        all_embs[i : i + args.batch_size] = embeddings.cpu().numpy()

        if (i // args.batch_size) % 30 == 0:
            print(f"  {min(i + args.batch_size, n_total):5d}/{n_total}", flush=True)

    q_embs = all_embs[: len(query_paths)]
    g_embs = all_embs[len(query_paths) :]

    # Evaluate
    print("\nEvaluating...")
    t1 = time.perf_counter()
    metrics = compute_mAP_cmc(q_embs, g_embs, q_pids, g_pids, q_cams, g_cams)
    print(f"  Done in {time.perf_counter() - t1:.1f}s\n")

    print("=" * 50)
    print(f"Market-1501 — DimReduceHead ({scales})")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:6.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
