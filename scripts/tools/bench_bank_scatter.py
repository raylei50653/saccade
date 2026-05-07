"""
Benchmark: P0/P1 bank scatter hot-path gain.

Measures wall time of update_reference_features + set_clean_embedding_flags
(new path) vs a Python-side simulation of the old O(N×MAX_OBJ) scan path,
at multiple active-track counts T.

Usage:
    uv run python scripts/tools/bench_bank_scatter.py
    uv run python scripts/tools/bench_bank_scatter.py --iters 500 --embed-dim 768
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "build"))
from saccade_tracking_ext import GPUByteTracker  # noqa: E402

WARMUP = 50
DEFAULT_ITERS = 200
MAX_OBJ = 2048


def spawn_tracks(tracker: GPUByteTracker, n: int, d: int) -> list[int]:
    """Confirm n tracks by running 3 frames with confirm_streak=1."""
    boxes = torch.zeros(n, 4, device="cuda")
    for i in range(n):
        x, y = 100.0 + i * (480.0 / max(n, 1)), 200.0
        boxes[i] = torch.tensor([x, y, x + 50.0, y + 100.0])
    scores  = torch.full((n,), 0.82, device="cuda")
    classes = torch.zeros(n, dtype=torch.int32, device="cuda")
    stream  = torch.cuda.current_stream().cuda_stream
    results = []
    for _ in range(3):
        results = tracker.update(
            boxes.data_ptr(), scores.data_ptr(), classes.data_ptr(), n, stream
        )
        boxes += 1.0
    return [r.obj_id for r in results]


def bench_new_path(
    tracker: GPUByteTracker,
    ids: list[int],
    d: int,
    iters: int,
) -> float:
    """Time the new C++ path: ensure_slot_map (lazy) + D2D scatter."""
    n = len(ids)
    feats   = torch.stack([F.normalize(torch.randn(d), dim=0).cuda() for _ in ids])
    ids_gpu = torch.tensor(ids, dtype=torch.int32, device="cuda")
    flags   = torch.ones(n, dtype=torch.uint8, device="cuda")
    stream  = torch.cuda.current_stream().cuda_stream

    # warm up
    for _ in range(WARMUP):
        tracker.update_reference_features(ids_gpu.data_ptr(), feats.data_ptr(), n, stream)
        tracker.set_clean_embedding_flags(ids_gpu.data_ptr(), flags.data_ptr(), n, stream)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        tracker.update_reference_features(ids_gpu.data_ptr(), feats.data_ptr(), n, stream)
        tracker.set_clean_embedding_flags(ids_gpu.data_ptr(), flags.data_ptr(), n, stream)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def bench_old_path_sim(
    n_tracks: int,
    max_obj: int,
    d: int,
    iters: int,
) -> float:
    """
    Simulate the old O(N×MAX_OBJ) C++ path cost:
      - 2× D2H of max_obj bools + max_obj ints  (blocking)
      - 1× D2H of n ints                          (blocking)
      - O(N × max_obj) CPU scan
      - N× D2D async copies (embed_dim floats each)
    Uses actual cudaMemcpy timing on dummy buffers.
    """
    d_active   = torch.zeros(max_obj, dtype=torch.bool,  device="cuda")
    d_tids     = torch.zeros(max_obj, dtype=torch.int32, device="cuda")
    d_src_tids = torch.zeros(n_tracks, dtype=torch.int32, device="cuda")
    d_src_feat = torch.zeros(n_tracks, d, device="cuda")
    d_dst_feat = torch.zeros(max_obj,  d, device="cuda")

    h_active = torch.zeros(max_obj, dtype=torch.bool)
    h_tids   = torch.zeros(max_obj, dtype=torch.int32)
    h_src    = torch.zeros(n_tracks, dtype=torch.int32)

    # warm up
    for _ in range(WARMUP):
        # 3 blocking D2H (active array, track_id array, src_tids)
        h_active.copy_(d_active)
        h_tids.copy_(d_tids)
        h_src.copy_(d_src_tids)
        torch.cuda.synchronize()
        # O(N × max_obj) scan in Python
        for i in range(n_tracks):
            tid = int(h_src[i].item())
            for slot in range(max_obj):
                if h_active[slot] and h_tids[slot] == tid:
                    break
        # N scatter copies (D2D async)
        for i in range(n_tracks):
            d_dst_feat[i].copy_(d_src_feat[i], non_blocking=True)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        h_active.copy_(d_active)
        h_tids.copy_(d_tids)
        h_src.copy_(d_src_tids)
        torch.cuda.synchronize()
        for i in range(n_tracks):
            tid = int(h_src[i].item())
            for slot in range(max_obj):
                if h_active[slot] and h_tids[slot] == tid:
                    break
        for i in range(n_tracks):
            d_dst_feat[i].copy_(d_src_feat[i], non_blocking=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters",     type=int, default=DEFAULT_ITERS)
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--tracks",    type=int, nargs="+", default=[5, 20, 50, 100, 200])
    args = ap.parse_args()

    D = args.embed_dim
    print(f"Benchmark: embed_dim={D}, MAX_OBJ={MAX_OBJ}, iters={args.iters}")
    print(f"{'T':>5}  {'old (ms)':>10}  {'new (ms)':>10}  {'speedup':>8}  {'saved/frame':>12}")
    print("-" * 55)

    for T in args.tracks:
        # New path
        tracker = GPUByteTracker(MAX_OBJ, D)
        tracker.set_params(0.5, 0.6, 0.8, 30, 0.4, 1)
        ids = spawn_tracks(tracker, T, D)
        if len(ids) < T:
            print(f"  T={T}: only {len(ids)} tracks confirmed, skipping")
            continue
        ids = ids[:T]

        t_new = bench_new_path(tracker, ids, D, args.iters)

        # Old path simulation
        t_old = bench_old_path_sim(T, MAX_OBJ, D, args.iters)

        speedup = t_old / t_new
        saved_us = (t_old - t_new) * 1e3  # µs
        print(f"{T:>5}  {t_old:>10.3f}  {t_new:>10.3f}  {speedup:>7.1f}×  {saved_us:>+11.0f} µs")

    print()
    # FPS impact at T=50 (typical MOT17)
    T_ref = 50
    if T_ref in args.tracks:
        pass  # already printed above
    else:
        tracker = GPUByteTracker(MAX_OBJ, D)
        tracker.set_params(0.5, 0.6, 0.8, 30, 0.4, 1)
        ids = spawn_tracks(tracker, T_ref, D)[:T_ref]
        t_new = bench_new_path(tracker, ids, D, args.iters)
        t_old = bench_old_path_sim(T_ref, MAX_OBJ, D, args.iters)
        print(f"T={T_ref} reference: old={t_old:.3f}ms new={t_new:.3f}ms")

    # Estimate FPS headroom freed at 30fps baseline
    print()
    print("At 30 fps baseline (33.3ms/frame budget):")
    budget_ms = 1000.0 / 30.0
    for T in [50, 100, 200]:
        tracker = GPUByteTracker(MAX_OBJ, D)
        tracker.set_params(0.5, 0.6, 0.8, 30, 0.4, 1)
        ids = spawn_tracks(tracker, T, D)[:T]
        t_new = bench_new_path(tracker, ids, D, args.iters)
        t_old = bench_old_path_sim(T, MAX_OBJ, D, args.iters)
        saved_pct = (t_old - t_new) / budget_ms * 100
        print(f"  T={T:>3}: saved {t_old-t_new:.3f}ms/frame  ({saved_pct:+.1f}% of 33ms budget)")


if __name__ == "__main__":
    main()
