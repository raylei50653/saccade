#!/usr/bin/env python3
"""Isolate inference-loop concurrency: does the GIL-free C++ detector parallelize
across threads, and do per-stream CUDA streams unlock it?

No DALI file IO, no TrackEval — just N threads looping forward_ptr (backbone+head,
GIL released) on preloaded GPU frames. Compares:
  * 1 thread vs N threads  (scaling)
  * shared default stream vs per-thread CUDA stream  (sync/stream serialization)
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

import torch

_old = sys.getdlopenflags()
sys.setdlopenflags(_old | ctypes.RTLD_GLOBAL)
import saccade_tracking_ext  # noqa: F401, E402

sys.setdlopenflags(_old)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Importing mamba_head registers the saccade::selective_scan_fwd custom op that
# the TorchScript head dispatches to.
import saccade.perception.temporal_yolo.mamba_head  # noqa: E402, F401

from saccade_perception_ext import MambaGatedDetector  # noqa: E402

BACKBONE = str(ROOT / "models/yolo/yolo26s_backbone_640_best.engine")
HEAD = str(ROOT / "models/yolo/mamba_head_best.pt")
N = 4
ITERS = 200
OUT_ROWS = 4096  # forward_ptr writes compact NMS dets; generous buffer


def make_detector() -> MambaGatedDetector:
    return MambaGatedDetector(
        trt_backbone_path=BACKBONE,
        mamba_head_script_path=HEAD,
        img_size=640,
        conf_thr=0.05,
    )


def worker(det, frame, out, iters, stream, barrier):
    barrier.wait()
    if stream is not None:
        with torch.cuda.stream(stream):
            for _ in range(iters):
                det.forward_ptr(frame.data_ptr(), out.data_ptr())
            stream.synchronize()
    else:
        for _ in range(iters):
            det.forward_ptr(frame.data_ptr(), out.data_ptr())
        torch.cuda.synchronize()


def run(det, n_threads, iters, per_stream):
    frames = [
        torch.rand(1, 3, 640, 640, device="cuda").contiguous() for _ in range(n_threads)
    ]
    outs = [torch.empty(OUT_ROWS, 6, device="cuda") for _ in range(n_threads)]
    streams = (
        [torch.cuda.Stream() for _ in range(n_threads)]
        if per_stream
        else [None] * n_threads
    )
    # warmup each thread's TRT context + head load (thread-local), off the clock
    threads = [
        threading.Thread(
            target=worker,
            args=(det, frames[i], outs[i], 3, streams[i], threading.Barrier(1)),
        )
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()

    barrier = threading.Barrier(n_threads)
    threads = [
        threading.Thread(
            target=worker, args=(det, frames[i], outs[i], iters, streams[i], barrier)
        )
        for i in range(n_threads)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    total = n_threads * iters
    return dt, total / dt


def main():
    det = make_detector()
    print(f"frames={N}  iters/thread={ITERS}\n")

    dt1, fps1 = run(det, 1, ITERS, per_stream=False)
    print(
        f"1 thread,  default stream : {dt1:.2f}s  {fps1:7.1f} det/s  (per-frame {1000 / fps1:.2f}ms)"
    )

    dtN, fpsN = run(det, N, ITERS, per_stream=False)
    print(
        f"{N} threads, default stream : {dtN:.2f}s  {fpsN:7.1f} det/s  (scaling {fpsN / fps1:.2f}x)"
    )

    dtP, fpsP = run(det, N, ITERS, per_stream=True)
    print(
        f"{N} threads, per-thread strm: {dtP:.2f}s  {fpsP:7.1f} det/s  (scaling {fpsP / fps1:.2f}x)"
    )

    print(f"\nper-stream vs default (N={N}): {fpsP / fpsN:.2f}x")


if __name__ == "__main__":
    main()
