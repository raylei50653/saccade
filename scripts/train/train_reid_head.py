#!/usr/bin/env python3
"""
Train ReID projection head from pre-extracted FPN embedding cache.

Reads cache produced by cache_fpn_embeddings.py:
  cache_dir/<seq>.pt  →  {frame_id: {"track_ids": [int], "embs": Tensor(N, 896)}}

No YOLO inference at train time — pure MLP forward + SupCon loss.
GPU stays near 100% utilised.

Usage:
    # Step 1: cache embeddings (once)
    uv run scripts/train/cache_fpn_embeddings.py \\
        --ckpt runs/gated_det_v1/best.ckpt \\
        --cache-dir runs/reid_cache

    # Step 2: train head
    uv run scripts/train/train_reid_head.py \\
        --cache-dir runs/reid_cache \\
        --run-dir runs/reid_head_v1 \\
        [--epochs 30] [--lr 3e-4]
"""
# status: experiment

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.reid_head import ReIDHead, supcon_loss, EMB_DIM_IN


# ---------------------------------------------------------------------------
# Cache-based pair dataset
# ---------------------------------------------------------------------------


class CachedEmbDataset(Dataset):
    """
    Reads pre-extracted FPN embeddings from cache/.pt files.
    Each item: two frames from same sequence with random gap [gap_min, gap_max].
    Returns (embs_a, ids_a, embs_b, ids_b) — all CPU tensors.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        gap_min: int = 10,
        gap_max: int = 60,
        min_shared: int = 2,
        pairs_per_seq: int = 300,
        seed: int = 42,
    ):
        cache_dir = Path(cache_dir)
        self._gap_min = gap_min
        self._gap_max = gap_max
        self._min_shared = min_shared
        self._seed = seed

        print(f"Loading embedding cache from {cache_dir} ...")
        self._seqs: dict[str, list[tuple[int, dict]]] = {}
        for pt in sorted(cache_dir.glob("*.pt")):
            data: dict = torch.load(pt, map_location="cpu", weights_only=False)
            if not data:
                continue
            frames = sorted(data.items())  # [(frame_id, {track_ids, embs})]
            if len(frames) < gap_min + 1:
                continue
            self._seqs[pt.stem] = frames

        print(f"  {len(self._seqs)} sequences loaded")
        self._pairs = self._build_pairs(pairs_per_seq)

    def _build_pairs(self, pairs_per_seq: int) -> list[tuple[str, int, int]]:
        rng = random.Random(self._seed)
        pairs: list[tuple[str, int, int]] = []

        for seq, frames in self._seqs.items():
            n = len(frames)
            seq_pairs: list[tuple[str, int, int]] = []

            for i in range(n):
                lo = i + self._gap_min
                hi = min(i + self._gap_max, n - 1)
                if lo > hi:
                    continue
                j = rng.randint(lo, hi)
                ids_a = set(frames[i][1]["track_ids"])
                ids_b = set(frames[j][1]["track_ids"])
                if len(ids_a & ids_b) >= self._min_shared:
                    seq_pairs.append((seq, i, j))

            if len(seq_pairs) > pairs_per_seq:
                seq_pairs = rng.sample(seq_pairs, pairs_per_seq)
            pairs.extend(seq_pairs)

        rng.shuffle(pairs)
        return pairs

    def reshuffle(self, seed: int | None = None) -> None:
        self._seed = seed if seed is not None else self._seed + 1
        self._pairs = self._build_pairs(300)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        seq, i, j = self._pairs[idx]
        frames = self._seqs[seq]
        _, data_a = frames[i]
        _, data_b = frames[j]
        embs_a = data_a["embs"].float()  # (N_a, 896)
        ids_a = torch.tensor(data_a["track_ids"], dtype=torch.long)
        embs_b = data_b["embs"].float()  # (N_b, 896)
        ids_b = torch.tensor(data_b["track_ids"], dtype=torch.long)
        return embs_a, ids_a, embs_b, ids_b


def _collate(batch):
    """Variable-length track counts — keep as lists."""
    embs_a, ids_a, embs_b, ids_b = zip(*batch)
    return list(embs_a), list(ids_a), list(embs_b), list(ids_b)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = CachedEmbDataset(
        args.cache_dir,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        min_shared=2,
        pairs_per_seq=300,
    )
    print(f"  {len(dataset)} training pairs")

    # num_workers=0: all .pt data is preloaded into RAM in __init__; workers
    # would fork-copy the full _seqs dict (3+ GB) and add IPC overhead.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )

    head = ReIDHead(in_dim=EMB_DIM_IN, hidden=args.hidden_dim, out_dim=args.out_dim).to(
        device
    )
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.epochs + 1):
        dataset.reshuffle(epoch)
        epoch_loss = 0.0
        n_valid = 0
        t0 = time.perf_counter()
        optimizer.zero_grad()

        pending = 0  # accumulated steps since last optimizer update
        for step, (embs_a_list, ids_a_list, embs_b_list, ids_b_list) in enumerate(
            loader
        ):
            # Concatenate all embeddings + labels across the batch
            embs_raw = torch.cat(embs_a_list + embs_b_list, dim=0).to(
                device, non_blocking=True
            )
            labels = torch.cat(ids_a_list + ids_b_list, dim=0).to(
                device, non_blocking=True
            )

            # Need ≥1 track ID that appears ≥2 times (positive pair exists)
            _, counts = labels.unique(return_counts=True)
            if (counts >= 2).sum() < 1:
                continue

            with torch.amp.autocast("cuda"):
                projected = head(embs_raw)
                loss = supcon_loss(projected, labels, temperature=args.temperature)

            if loss.item() == 0.0 or torch.isnan(loss):
                continue

            scaler.scale(loss / args.accum_steps).backward()
            epoch_loss += loss.item()
            n_valid += 1
            pending += 1

            if pending % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pending = 0

        # Flush remaining accumulated gradient (if any)
        if pending > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()

        avg_loss = epoch_loss / max(n_valid, 1)
        elapsed = time.perf_counter() - t0
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  n={n_valid}  {elapsed:.1f}s"
        )

        if avg_loss < best_loss and n_valid > 0:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "head": head.state_dict(),
                    "best_loss": best_loss,
                    "cfg": {
                        "in_dim": EMB_DIM_IN,
                        "hidden": args.hidden_dim,
                        "out_dim": args.out_dim,
                    },
                },
                run_dir / "best.ckpt",
            )
            print(f"    ✓ saved best (loss={best_loss:.4f})")

    print(f"\nDone. Best loss: {best_loss:.4f}  →  {run_dir}/best.ckpt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="runs/reid_cache",
        help="Directory with *.pt cache files from cache_fpn_embeddings.py",
    )
    parser.add_argument("--run-dir", default="runs/reid_head_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Frame-pair batch size. Each pair contributes ~100-200 tracks; "
        "SupCon memory scales O(n^2) in total tracks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0: cache is preloaded in RAM.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gap-min", type=int, default=10)
    parser.add_argument("--gap-max", type=int, default=60)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
