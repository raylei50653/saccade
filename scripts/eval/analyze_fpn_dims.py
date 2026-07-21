#!/usr/bin/env python3
"""Analyze FPN feature dimension importance for ReID discrimination.

Extracts 896-dim FPN embeddings from Market-1501 person crops,
then compares same-ID vs different-ID feature distributions
to identify which dimensions carry ReID signal.

Usage:
    uv run scripts/eval/analyze_fpn_dims.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --market-root datasets/Market-1501-v15.09.15 \
        --num-ids 50 --samples-per-id 6
"""
# status: diagnostic

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    DataPreloader,
    resize_stretch_batch_gpu,
)

IMG_SIZE = 640


def _parse_pid(filename: str) -> int:
    return int(Path(filename).stem.split("_")[0])


def build_dataset(market_root: Path, num_ids: int, samples_per_id: int):
    train_dir = market_root / "bounding_box_train"
    by_id: dict[int, list[Path]] = {}
    for p in sorted(train_dir.glob("*.jpg")):
        pid = _parse_pid(p.name)
        by_id.setdefault(pid, []).append(p)

    selected_ids = sorted(by_id.keys())[:num_ids]
    selected_ids = [pid for pid in selected_ids if len(by_id[pid]) >= samples_per_id]

    items: list[tuple[Path, int]] = []
    for pid in selected_ids:
        paths = sorted(by_id[pid])[:samples_per_id]
        for p in paths:
            items.append((p, pid))
    return items, selected_ids


def extract_all(
    detector, preloader, items, batch_size, device
) -> tuple[np.ndarray, np.ndarray]:
    all_paths = [p for p, _pid in items]
    all_pids = np.array([pid for _p, pid in items], dtype=np.int32)
    n = len(all_paths)

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_det, _ = detector.forward(dummy.float(), gate_input=None)
    dummy_emb = detector.extract_fpn_embeddings(None, torch.zeros(1, 4).to(device))

    dim = dummy_emb.shape[1]
    all_embs = np.empty((n, dim), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch_paths = all_paths[i : i + batch_size]
        imgs_uint8 = torch.stack([preloader[p] for p in batch_paths]).to(device)
        frame_640 = resize_stretch_batch_gpu(imgs_uint8, IMG_SIZE, device)

        # Run full forward (caches FPN), then extract center-pooled embeddings
        detector.forward(frame_640.float(), gate_input=None)
        center_box = torch.tensor(
            [
                [
                    IMG_SIZE // 2 - 1,
                    IMG_SIZE // 2 - 1,
                    IMG_SIZE // 2 + 1,
                    IMG_SIZE // 2 + 1,
                ]
            ]
            * len(batch_paths),
            device=device,
            dtype=torch.float32,
        )
        embeddings = detector.extract_fpn_embeddings(None, center_box)
        all_embs[i : i + batch_size] = embeddings.cpu().numpy()

        if (i // batch_size) % 20 == 0:
            print(f"  {min(i + batch_size, n):5d}/{n}", flush=True)

    return all_embs, all_pids


def analyze(embs: np.ndarray, pids: np.ndarray, num_ids: int, samples_per_id: int):
    """Per-dimension same-ID vs different-ID cosine similarity analysis."""
    dim = embs.shape[1]

    # Build per-ID mean embeddings
    id_means: dict[int, np.ndarray] = {}
    id_samples: dict[int, list[np.ndarray]] = {}
    for i, pid in enumerate(pids):
        id_samples.setdefault(int(pid), []).append(embs[i])
    for pid, s_list in id_samples.items():
        id_means[pid] = np.mean(s_list, axis=0)

    # For each dimension: compute same-ID cosine_std and different-ID cosine_mean
    # We analyze: per dimension, the correlation between all same-ID pairs vs all diff-ID pairs
    same_cos = []
    diff_cos = []

    pids_list = sorted(id_samples.keys())

    for pid in pids_list:
        samples = id_samples[pid]
        # Same-ID: all pairs within this ID
        for a in range(len(samples)):
            for b in range(a + 1, len(samples)):
                # Per-dimension cosine similarity
                cos_per_dim = (
                    samples[a] * samples[b]
                )  # element-wise, since L2-normalized
                same_cos.append(cos_per_dim)

    for i, pid_a in enumerate(pids_list):
        for pid_b in pids_list[i + 1 :]:
            # Different-ID: mean vs mean
            cos_per_dim = id_means[pid_a] * id_means[pid_b]
            diff_cos.append(cos_per_dim)

    same_cos = np.array(same_cos)  # (num_same_pairs, dim)
    diff_cos = np.array(diff_cos)  # (num_diff_pairs, dim)

    same_mean = same_cos.mean(axis=0)  # per-dim mean cosine for same-ID
    diff_mean = diff_cos.mean(axis=0)  # per-dim mean cosine for diff-ID
    same_std = same_cos.std(axis=0)
    diff_std = diff_cos.std(axis=0)

    # Discriminability score per dimension: separation / combined std
    # Higher = better separation between same-ID and diff-ID distributions
    separation = same_mean - diff_mean
    combined_std = np.sqrt(same_std**2 + diff_std**2)
    discriminability = separation / (combined_std + 1e-8)

    return {
        "same_mean": same_mean,
        "diff_mean": diff_mean,
        "separation": separation,
        "discriminability": discriminability,
        "dim": dim,
    }


def plot(results: dict, out_path: Path) -> None:
    dim = results["dim"]
    separation = results["separation"]
    discriminability = results["discriminability"]
    same_mean = results["same_mean"]
    diff_mean = results["diff_mean"]

    # Sort by discriminability
    order = np.argsort(-discriminability)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Per-dim same vs diff cosine similarity
    ax = axes[0, 0]
    ax.plot(same_mean, alpha=0.7, label="Same-ID cos mean", linewidth=0.5)
    ax.plot(diff_mean, alpha=0.7, label="Diff-ID cos mean", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Per-dimension Same-ID vs Different-ID cosine")
    ax.legend()

    # 2. Separation sorted
    ax = axes[0, 1]
    ax.plot(separation[order], linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Dimension (sorted by discriminability)")
    ax.set_ylabel("Same − Diff cosine")
    ax.set_title("Per-dimension separation (sorted)")

    # 3. Discriminability histogram
    ax = axes[1, 0]
    ax.hist(discriminability, bins=50, edgecolor="none", alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--")
    ax.set_xlabel("Discriminability score")
    ax.set_ylabel("Count")
    ax.set_title(f"Discriminability distribution ({dim} dims)")

    # 4. Cumulative separation
    ax = axes[1, 1]
    top_k = np.arange(1, dim + 1)
    cum_sep = np.cumsum(separation[order])
    ax.plot(top_k, cum_sep, linewidth=1)
    ax.axhline(
        y=cum_sep[-1] * 0.9,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="90% cumulative",
    )
    k90 = np.searchsorted(cum_sep, cum_sep[-1] * 0.9)
    ax.axvline(x=k90, color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Top-K dimensions")
    ax.set_ylabel("Cumulative separation")
    ax.set_title(f"Cumulative separation (90% in top {k90}/{dim} dims)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    # Text summary
    top_50 = np.argsort(-discriminability)[:50]
    print(f"\nTop-50 discriminative dimensions: {sorted(top_50)}")
    print("\nDim ranges (896 = P3:0-127  P4:128-383  P5:384-895):")
    for name, start, end in [("P3", 0, 128), ("P4", 128, 384), ("P5", 384, 896)]:
        in_top = sum(1 for i in top_50 if start <= i < end)
        mean_sep = separation[start:end].mean()
        print(
            f"  {name} ({end - start}d): {in_top}/50 in top-50, mean_sep={mean_sep:.4f}"
        )

    id_dim = np.argmax(discriminability)
    print(f"\nSingle best dim: {id_dim} (separation={separation[id_dim]:.4f})")
    print(
        "  If P3: yes, P4: yes, P5: yes"
        if id_dim < 128
        else (
            f"  P4 dim {id_dim - 128}" if id_dim < 384 else f"  P5 dim {id_dim - 384}"
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--num-ids", type=int, default=50)
    parser.add_argument("--samples-per-id", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="runs/fpn_dim_analysis.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root

    print(f"Device: {device}")
    print(
        f"IDs: {args.num_ids} × {args.samples_per_id} = {args.num_ids * args.samples_per_id} images"
    )

    items, selected_ids = build_dataset(market_root, args.num_ids, args.samples_per_id)
    print(f"  Selected: {len(selected_ids)} IDs, {len(items)} images")

    # Preload
    all_paths = [p for p, _pid in items]
    preloader = DataPreloader(all_paths, num_workers=8)
    preloader.load()

    # Build detector
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

    # Extract
    print("\nExtracting FPN embeddings...")
    embs, pids = extract_all(detector, preloader, items, args.batch_size, device)
    print(f"  Embeddings: {embs.shape}")

    # Analyze
    print("\nAnalyzing dimension importance...")
    results = analyze(embs, pids, args.num_ids, args.samples_per_id)

    # Plot
    out_path = project_root / args.output
    plot(results, out_path)


if __name__ == "__main__":
    main()
