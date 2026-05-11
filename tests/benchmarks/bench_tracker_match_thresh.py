"""Tracker match_thresh FPS anomaly micro-benchmark.

Isolates the GPUByteTracker from the full eval pipeline to determine whether
the FPS drop at match_thresh >= 0.73 (with new_track_thresh >= 0.35) originates
in the C++ CUDA tracker or in the Python evaluator.

Usage:
    uv run python tests/benchmarks/bench_tracker_match_thresh.py
    uv run python tests/benchmarks/bench_tracker_match_thresh.py --thresholds 0.70,0.72,0.73,0.74,0.75
    uv run python tests/benchmarks/bench_tracker_match_thresh.py --new-track-thresh 0.28  # presets use 0.28
    uv run python tests/benchmarks/bench_tracker_match_thresh.py --frames 500 --warmup 100

Results:
    If timing jumps at 0.73 → root in C++ tracker.
    If timing is flat → root in Python evaluator, compare with full eval.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if BUILD_DIR.exists():
    sys.path.insert(0, str(BUILD_DIR))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tracker match_thresh FPS anomaly benchmark"
    )
    p.add_argument(
        "--thresholds",
        default="0.70,0.71,0.72,0.73,0.74,0.75",
        help="Comma-separated match_thresh values to test",
    )
    p.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.35,
        help="new_track_thresh value (anomaly requires >= 0.35)",
    )
    p.add_argument(
        "--frames", type=int, default=300, help="Measurement frames per threshold"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Warm-up frames per threshold (pre-populate tracker state)",
    )
    p.add_argument(
        "--n-dets",
        type=int,
        default=80,
        help="Synthetic detections per frame (typical MOT17 SDP)",
    )
    p.add_argument(
        "--max-objs", type=int, default=2048, help="GPUByteTracker max_objects capacity"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _make_tracker(
    max_objs: int, match_thresh: float, new_track_thresh: float, device: str
):
    try:
        from saccade_tracking_ext import GPUByteTracker
    except ImportError as e:
        print(f"Cannot import GPUByteTracker: {e}")
        sys.exit(1)

    tracker = GPUByteTracker(max_objects=max_objs)
    tracker.set_params(
        track_thresh=0.05,
        high_thresh=0.45,
        match_thresh=match_thresh,
        track_buffer=30,
        mid_thresh=0.10,
        confirm_streak=1,
        confirm_score_thresh=0.0,
        adaptive_confirmation=False,
        new_track_thresh=new_track_thresh,
        nsa_kalman=False,
        r_scale=0.75,
        vel_dir_weight=0.0,
    )
    tracker.set_frame_size(1920, 1080)
    return tracker


def _gen_dets(
    n: int, seed_offset: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed_offset)
    # boxes: [N, 4] in xyxy format within a 1920x1080 frame
    cx = torch.rand(n, generator=rng, device=device) * 1920
    cy = torch.rand(n, generator=rng, device=device) * 1080
    w = (torch.rand(n, generator=rng, device=device) * 60 + 20).clamp(10, 200)
    h = (torch.rand(n, generator=rng, device=device) * 150 + 50).clamp(30, 400)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    boxes[:, 0].clamp_(0, 1919)
    boxes[:, 2].clamp_(0, 1920)
    boxes[:, 1].clamp_(0, 1079)
    boxes[:, 3].clamp_(0, 1080)
    # scores: mix of high-conf and mid-conf, matching MOT17 distribution
    scores = torch.rand(n, generator=rng, device=device) * 0.8 + 0.05
    classes = torch.zeros(n, dtype=torch.int32, device=device)
    return boxes.contiguous(), scores.contiguous(), classes


def _run_tracker_frames(
    tracker,
    n_frames: int,
    n_dets: int,
    seed_base: int,
    device: str,
    stream_ptr: int,
) -> float:
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for f in range(n_frames):
        boxes, scores, classes = _gen_dets(n_dets, seed_base + f, device)
        tracker.update(
            boxes.data_ptr(),
            scores.data_ptr(),
            classes.data_ptr(),
            n_dets,
            stream_ptr,
        )
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def main() -> None:
    args = _parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]
    device = args.device

    if not torch.cuda.is_available():
        print("CUDA not available — cannot run benchmark")
        sys.exit(1)

    stream = torch.cuda.Stream(device=device)
    stream_ptr = stream.cuda_stream

    print("Tracker match_thresh FPS anomaly benchmark")
    print(f"  new_track_thresh = {args.new_track_thresh}")
    print(f"  frames per threshold = {args.frames}  warmup = {args.warmup}")
    print(f"  detections per frame = {args.n_dets}  max_objs = {args.max_objs}")
    print(f"  device = {device}")
    print()
    print(f"{'match_thresh':>12} | {'ms/frame':>10} | {'FPS':>8} | {'Δms vs 0.72':>12}")
    print("-" * 52)

    results: dict[float, float] = {}
    ref_ms: float | None = None

    for thresh in thresholds:
        tracker = _make_tracker(args.max_objs, thresh, args.new_track_thresh, device)

        # Warm-up: pre-populate tracker with tracked objects so steady-state is reached
        _run_tracker_frames(tracker, args.warmup, args.n_dets, 0, device, stream_ptr)

        # Measure
        elapsed = _run_tracker_frames(
            tracker, args.frames, args.n_dets, 1000, device, stream_ptr
        )
        ms_per_frame = elapsed / args.frames * 1000
        fps = 1000.0 / ms_per_frame
        results[thresh] = ms_per_frame

        if abs(thresh - 0.72) < 1e-6:
            ref_ms = ms_per_frame

        delta_str = ""
        if ref_ms is not None:
            delta = ms_per_frame - ref_ms
            delta_str = f"{delta:+.3f}ms"
        elif thresh < 0.72:
            delta_str = "(before ref)"

        print(
            f"{thresh:>12.2f} | {ms_per_frame:>10.3f} | {fps:>8.1f} | {delta_str:>12}"
        )

    print()
    # Summary: detect discrete jump
    jump_detected = False
    prev_ms = None
    for thresh in sorted(results):
        ms = results[thresh]
        if prev_ms is not None:
            delta = ms - prev_ms
            if abs(delta) > 1.0:  # >1ms jump between adjacent thresholds
                jump_detected = True
                print(
                    f"  ** Discrete jump detected: {thresh - 0.01:.2f} → {thresh:.2f}: Δ{delta:+.2f}ms"
                )
        prev_ms = ms

    if jump_detected:
        print(
            "\nConclusion: FPS anomaly IS present in isolated tracker. Root cause is in C++ CUDA kernels."
        )
        print("Next: rebuild with new NVTX markers and run:")
        print(
            "  nsys profile --trace=cuda,nvtx uv run python tests/benchmarks/bench_tracker_match_thresh.py"
        )
    else:
        print(
            "\nConclusion: FPS anomaly NOT present in isolated tracker. Root cause is in Python evaluator."
        )
        print(
            "Next: add CUDA event timing in evaluator to isolate which Python-side stage is slow."
        )


if __name__ == "__main__":
    main()
