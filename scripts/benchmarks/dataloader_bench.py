"""
DataLoader throughput benchmark: preload_to_ram vs num_workers.

Usage:
    uv run scripts/benchmarks/dataloader_bench.py --data-root datasets/MOT17
"""
# status: diagnostic

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader


def bench(label: str, **kwargs) -> float:
    loader = build_mot17_dataloader(**kwargs)
    n_warmup, n_iters = 2, 10
    it = iter(loader)
    for _ in range(n_warmup):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    t0 = time.perf_counter()
    for i in range(n_iters):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
    elapsed = time.perf_counter() - t0
    its = n_iters / elapsed
    print(f"  {label:40s}  {its:5.2f} it/s  ({elapsed / n_iters * 1000:.1f} ms/iter)")
    return its


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--seqs", default="MOT17-02-SDP")
    args = parser.parse_args()

    common = dict(
        data_root=args.data_root,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        seqs=args.seqs.split(","),
        shuffle=False,
    )

    print(f"\nDataLoader benchmark  B={args.batch_size} T={args.clip_len}\n")

    a = bench(
        "num_workers=0  preload=True ", num_workers=0, preload_to_ram=True, **common
    )
    b = bench(
        "num_workers=2  preload=False", num_workers=2, preload_to_ram=False, **common
    )
    c = bench(
        "num_workers=4  preload=False", num_workers=4, preload_to_ram=False, **common
    )

    print(f"\n  workers=2 speedup vs preload: {b / a:+.2f}×")
    print(f"  workers=4 speedup vs preload: {c / a:+.2f}×")


if __name__ == "__main__":
    main()
