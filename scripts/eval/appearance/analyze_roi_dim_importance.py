#!/usr/bin/env python3
"""
分析 YOLO ROI embedding 896維中哪些維度對行人最有區分力。

方法：Fisher Score per dimension
  F(d) = inter-ID variance of dim d / intra-ID variance of dim d
  高 F = 不同人差距大、同一人穩定 → 對 ReID 有用

Usage:
    uv run scripts/eval/appearance/analyze_roi_dim_importance.py \
        --ckpt runs/gated_det_v1/best.ckpt \
        --sequences MOT17-09-SDP,MOT17-02-SDP,MOT17-05-SDP
"""
# status: stable

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
)
from saccade.perception.temporal_yolo.roi_embedder import extract_roi_embeddings  # noqa: E402


# ---------------------------------------------------------------------------
# GT loader (same as validate_roi_embeddings.py)
# ---------------------------------------------------------------------------
def load_gt(seq_dir: Path, img_size: int) -> list[dict]:
    gt_file = seq_dir / "gt" / "gt.txt"
    ini = seq_dir / "seqinfo.ini"
    orig_w, orig_h = 1920, 1080
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth"):
                orig_w = int(line.split("=")[1])
            elif line.startswith("imHeight"):
                orig_h = int(line.split("=")[1])

    sx, sy = img_size / orig_w, img_size / orig_h
    entries = []
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        fid, tid = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        vis = float(parts[8]) if len(parts) > 8 else 1.0
        cls = int(parts[7]) if len(parts) > 7 else 1
        if cls != 1 or vis < 0.3 or w < 10 or h < 10:
            continue
        entries.append(
            {
                "frame_id": fid,
                "track_id": tid,
                "bbox": [x * sx, y * sy, (x + w) * sx, (y + h) * sy],
            }
        )
    return entries


def load_frame(seq_dir: Path, frame_id: int, img_size: int) -> torch.Tensor:
    import torchvision.io as tv_io

    path = seq_dir / "img1" / f"{frame_id:06d}.jpg"
    img = tv_io.read_image(str(path)).float() / 255.0
    return F.interpolate(
        img.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False
    )


# ---------------------------------------------------------------------------
# Collect embeddings per track across sequences
# ---------------------------------------------------------------------------
def collect_track_embeddings(
    model,
    seqs: list[Path],
    img_size: int,
    device: torch.device,
    max_frames: int = 300,
) -> dict[str, torch.Tensor]:
    """Returns {global_track_id: tensor(K, D)} for all tracks across sequences."""
    all_tracks: dict[str, list[torch.Tensor]] = defaultdict(list)

    model.eval()
    model.cache_feats = True

    for seq_dir in seqs:
        print(f"  {seq_dir.name}...", end=" ", flush=True)
        gt = load_gt(seq_dir, img_size)
        by_frame: dict[int, list] = defaultdict(list)
        for e in gt:
            by_frame[e["frame_id"]].append(e)

        frames_done = 0
        with torch.no_grad():
            for fid in sorted(by_frame.keys())[:max_frames]:
                anns = by_frame[fid]
                frame = load_frame(seq_dir, fid, img_size).to(device)
                model._feat_cache.clear()
                _ = model(frame)
                if not model._feat_cache:
                    continue

                boxes = torch.tensor(
                    [a["bbox"] for a in anns], dtype=torch.float32, device=device
                )
                embs = extract_roi_embeddings(model._feat_cache, boxes)  # (N, 896)

                for i, ann in enumerate(anns):
                    gid = f"{seq_dir.name}/{ann['track_id']}"
                    all_tracks[gid].append(embs[i].cpu())

                frames_done += 1

        print(
            f"{frames_done} frames, {sum(1 for g in all_tracks if g.startswith(seq_dir.name))} tracks"
        )

    model.cache_feats = False

    # Filter tracks with at least 5 observations
    result = {
        gid: torch.stack(embs) for gid, embs in all_tracks.items() if len(embs) >= 5
    }
    print(f"Tracks with ≥5 obs: {len(result)} / {len(all_tracks)}")
    return result


# ---------------------------------------------------------------------------
# Fisher Score computation
# ---------------------------------------------------------------------------
def compute_fisher_scores(track_embs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Per-dimension Fisher Score = inter-ID variance / intra-ID variance.

    Returns tensor (D,) of scores.
    """
    all_means = []
    intra_vars = []

    for embs in track_embs.values():  # embs: (K, D)
        mean = embs.mean(0)  # (D,)
        var = embs.var(0, unbiased=False)  # (D,)
        all_means.append(mean)
        intra_vars.append(var)

    # Global mean across all tracks
    all_means_t = torch.stack(all_means)  # (T, D)
    global_mean = all_means_t.mean(0)  # (D,)

    # Inter-ID variance: variance of per-track means
    inter_var = ((all_means_t - global_mean) ** 2).mean(0)  # (D,)

    # Intra-ID variance: average within-track variance
    intra_var = torch.stack(intra_vars).mean(0)  # (D,)

    fisher = inter_var / (intra_var + 1e-8)
    return fisher


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument(
        "--sequences", default="MOT17-09-SDP,MOT17-02-SDP,MOT17-05-SDP,MOT17-11-SDP"
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--topk", type=int, default=128, help="Number of top dimensions to report"
    )
    parser.add_argument(
        "--save", default="", help="Save top-K dim indices to .npy file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))
    cfg = GatedDetConfig(scales=scales)
    yolo_weights = project_root / train_args.get(
        "yolo_weights", "models/yolo/yolo26s.pt"
    )
    model = build_gated_yolo_detector(
        str(yolo_weights), cfg=cfg, device=device, weights_path=str(ckpt_path)
    )
    model.eval()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root

    seqs = [
        data_root / "train" / s.strip() for s in args.sequences.split(",") if s.strip()
    ]
    seqs = [s for s in seqs if s.exists()]
    print(f"Collecting embeddings from {len(seqs)} sequences...")
    track_embs = collect_track_embeddings(
        model, seqs, args.img_size, device, args.max_frames
    )

    if not track_embs:
        print("No tracks collected.")
        return

    print("\nComputing Fisher scores...")
    scores = compute_fisher_scores(track_embs)  # (896,)

    # Sort by score descending
    sorted_idx = scores.argsort(descending=True)
    topk_idx = sorted_idx[: args.topk]
    botk_idx = sorted_idx[-args.topk :]

    print(f"\n=== Fisher Score Summary (D={scores.shape[0]}) ===")
    print(f"  Max:    {scores.max():.4f}  (dim {scores.argmax().item()})")
    print(f"  Min:    {scores.min():.4f}  (dim {scores.argmin().item()})")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {scores.median():.4f}")

    # Per-scale breakdown (P3: 0-127, P4: 128-383, P5: 384-895)
    p3_score = scores[:128].mean()
    p4_score = scores[128:384].mean()
    p5_score = scores[384:].mean()
    print("\n  Per-scale mean Fisher score:")
    print(f"    P3 (dims   0-127, 128-dim): {p3_score:.4f}")
    print(f"    P4 (dims 128-383, 256-dim): {p4_score:.4f}")
    print(f"    P5 (dims 384-895, 512-dim): {p5_score:.4f}")

    # How many top dims come from each scale?
    p3_count = (topk_idx < 128).sum().item()
    p4_count = ((topk_idx >= 128) & (topk_idx < 384)).sum().item()
    p5_count = (topk_idx >= 384).sum().item()
    print(f"\n  Top-{args.topk} dim distribution:")
    print(f"    P3: {p3_count}/{args.topk}  ({p3_count / args.topk * 100:.1f}%)")
    print(f"    P4: {p4_count}/{args.topk}  ({p4_count / args.topk * 100:.1f}%)")
    print(f"    P5: {p5_count}/{args.topk}  ({p5_count / args.topk * 100:.1f}%)")

    print(f"\n  Top-10 Fisher dims: {topk_idx[:10].tolist()}")
    print(f"  Bot-10 Fisher dims: {botk_idx[:10].tolist()}")

    # Cumulative variance explained by top-K
    total_score = scores.sum().item()
    topk_score = scores[topk_idx].sum().item()
    print(
        f"\n  Top-{args.topk} dims capture {topk_score / total_score * 100:.1f}% of total Fisher score"
    )

    if args.save:
        save_path = Path(args.save)
        np.save(save_path, topk_idx.numpy())
        print(f"\nSaved top-{args.topk} dim indices → {save_path}")

    # Histogram of scores
    print("\n  Score distribution:")
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for thr in thresholds:
        cnt = (scores > thr).sum().item()
        print(
            f"    F > {thr:5.1f}: {cnt:4d} dims  ({cnt / scores.shape[0] * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
