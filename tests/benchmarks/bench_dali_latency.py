"""
DALI / NVDEC 延遲基準測試

量化 DALIMediaClient 各子階段的延遲：
  - Pipeline 初始化時間
  - grab_tensor()：NVDEC decode + resize + normalize（GPU 端）
  - grab()：GPU + D2H copy + numpy 轉換

執行:
  uv run python tests/benchmarks/bench_dali_latency.py
  uv run python tests/benchmarks/bench_dali_latency.py --video datasets/MOT17_videos/MOT17-04-SDP.mp4 --frames 500
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

# Suppress noisy DALI info logs (file_list_include_preceding_frame, legacy impl)
logging.getLogger("nvidia.dali").setLevel(logging.WARNING)
os.environ.setdefault("DALI_LOG_LEVEL", "3")  # 0=DEBUG … 3=ERROR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saccade.media.dali_pipeline import DALIMediaClient  # noqa: E402

DEFAULT_VIDEO = "datasets/MOT17_videos/MOT17-05-SDP.mp4"


# ── stats ─────────────────────────────────────────────────────────────────────


class Stats:
    def __init__(self, name: str):
        self.name = name
        self.samples: list[float] = []

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def report(self) -> dict[str, float]:
        a = np.array(self.samples)
        return {
            "mean": float(np.mean(a)),
            "p50":  float(np.percentile(a, 50)),
            "p95":  float(np.percentile(a, 95)),
            "p99":  float(np.percentile(a, 99)),
            "min":  float(np.min(a)),
            "max":  float(np.max(a)),
            "std":  float(np.std(a)),
        }


def print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    hdr = f"{'Metric':<28} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Min':>8} {'Max':>8} {'Std':>8}"
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for name, r in rows:
        print(
            f"{name:<28} "
            f"{r['mean']:>7.3f}  "
            f"{r['p50']:>7.3f}  "
            f"{r['p95']:>7.3f}  "
            f"{r['p99']:>7.3f}  "
            f"{r['min']:>7.3f}  "
            f"{r['max']:>7.3f}  "
            f"{r['std']:>7.3f}"
        )
    print(sep)


# ── benchmark sections ────────────────────────────────────────────────────────


def bench_init(video: str, device_id: int) -> float:
    t0 = time.perf_counter()
    client = DALIMediaClient(video_path=video, batch_size=1, device_id=device_id)
    ok = client.connect()
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    if not ok:
        print(f"[ERROR] Failed to connect to {video}")
        sys.exit(1)
    client.release()
    return elapsed_ms


def bench_grab_tensor(
    client: DALIMediaClient,
    n_frames: int,
    n_warmup: int,
) -> tuple[Stats, float]:
    """Per-frame latency via CUDA events (sync per call — conservative upper bound)."""
    stats = Stats("grab_tensor (ms)")

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    grabbed = 0
    total = n_warmup + n_frames

    for i in range(total):
        torch.cuda.synchronize()
        start_evt.record()
        ok, _ = client.grab_tensor()
        end_evt.record()
        torch.cuda.synchronize()

        if not ok:
            print(f"[WARN] grab_tensor returned False at frame {i}")
            break

        if i >= n_warmup:
            stats.add(start_evt.elapsed_time(end_evt))
            grabbed += 1

    print(f"  Per-frame mode: measured {grabbed} frames (warmup={n_warmup})")
    return stats, grabbed / (sum(stats.samples) / 1_000)  # FPS from sum


def bench_grab_tensor_bulk(
    client: DALIMediaClient,
    n_frames: int,
    n_warmup: int,
) -> float:
    """True throughput: no per-frame sync so DALI prefetch pipeline stays hot."""
    for _ in range(n_warmup):
        client.grab_tensor()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    grabbed = 0
    for _ in range(n_frames):
        ok, _ = client.grab_tensor()
        if not ok:
            break
        grabbed += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = grabbed / elapsed
    print(f"  Bulk mode: {grabbed} frames in {elapsed:.2f}s → {fps:.1f} FPS "
          f"({elapsed / grabbed * 1000:.2f} ms/frame)")
    return fps


def bench_grab_cpu(
    client: DALIMediaClient,
    n_frames: int,
    n_warmup: int,
) -> Stats:
    stats = Stats("grab/D2H (ms)")

    for i in range(n_warmup + n_frames):
        t0 = time.perf_counter()
        ok, arr = client.grab()
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        if not ok:
            print(f"[WARN] grab returned False at frame {i}")
            break
        if i >= n_warmup:
            stats.add(elapsed_ms)

    return stats


# ── GPU memory helper ─────────────────────────────────────────────────────────


def gpu_mem_mb(device_id: int) -> float:
    return torch.cuda.memory_allocated(device_id) / 1_048_576


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="DALI/NVDEC latency benchmark")
    parser.add_argument("--video",    default=DEFAULT_VIDEO)
    parser.add_argument("--frames",   type=int, default=300, help="frames to measure")
    parser.add_argument("--warmup",   type=int, default=30,  help="warmup frames")
    parser.add_argument("--device",   type=int, default=0,   help="CUDA device id")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(args.device)
    print("\n=== DALI / NVDEC Latency Benchmark ===")
    print(f"  Video   : {args.video}")
    print(f"  GPU     : {device_name}")
    print(f"  Frames  : {args.frames}  (warmup={args.warmup})")
    print()

    # ── 1. Init time ──────────────────────────────────────────────────────────
    print("[1/4] Measuring pipeline init time (3 runs)...")
    init_times = [bench_init(args.video, args.device) for _ in range(3)]
    print(f"  Init: mean={np.mean(init_times):.1f} ms  "
          f"min={np.min(init_times):.1f} ms  max={np.max(init_times):.1f} ms\n")

    # ── 2a. grab_tensor — per-frame latency ──────────────────────────────────
    print("[2/4] Measuring grab_tensor() per-frame latency (sync per call)...")
    mem_before = gpu_mem_mb(args.device)

    client_gpu = DALIMediaClient(video_path=args.video, batch_size=1, device_id=args.device)
    client_gpu.connect()
    gt_stats, _ = bench_grab_tensor(client_gpu, args.frames, args.warmup)
    mem_after = gpu_mem_mb(args.device)
    client_gpu.release()
    print(f"  GPU mem delta: {mem_after - mem_before:+.1f} MB\n")

    # ── 2b. grab_tensor — bulk throughput ────────────────────────────────────
    print("[3/4] Measuring grab_tensor() bulk throughput (no per-frame sync)...")
    client_bulk = DALIMediaClient(video_path=args.video, batch_size=1, device_id=args.device)
    client_bulk.connect()
    bulk_fps = bench_grab_tensor_bulk(client_bulk, args.frames, args.warmup)
    client_bulk.release()
    print()

    # ── 3. grab (CPU) ─────────────────────────────────────────────────────────
    print("[4/4] Measuring grab() latency (GPU decode + D2H copy + numpy)...")
    client_cpu = DALIMediaClient(video_path=args.video, batch_size=1, device_id=args.device)
    client_cpu.connect()
    cpu_stats = bench_grab_cpu(client_cpu, args.frames, args.warmup)
    client_cpu.release()
    print()

    # ── report ────────────────────────────────────────────────────────────────
    gt_r  = gt_stats.report()
    cpu_r = cpu_stats.report()
    d2h   = cpu_r["mean"] - gt_r["mean"]

    print("=== Results (all times in ms) ===")
    print_table([
        ("grab_tensor [GPU, per-frame]", gt_r),
        ("grab [GPU+D2H, per-frame]",    cpu_r),
    ])

    print(f"\n  Bulk throughput (pipeline hot)  : {bulk_fps:.1f} FPS ({1000/bulk_fps:.2f} ms/frame)")
    print(f"  Per-frame mean (sync overhead)  : {gt_r['mean']:.2f} ms/frame")
    print(f"  D2H copy overhead               : {d2h:+.3f} ms / frame")
    print(f"  P99 jitter above mean           : {gt_r['p99'] - gt_r['mean']:.3f} ms\n")


if __name__ == "__main__":
    main()
