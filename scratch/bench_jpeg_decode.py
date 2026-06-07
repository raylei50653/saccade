"""Benchmark the two JPEG decode (incl. YCbCr->RGB) paths used in eval.

DALIStreamerStream (CPU/libjpeg) vs TorchvisionGpuStreamer (GPU/nvJPEG).
"""

import time
from pathlib import Path

import torch

from saccade.perception.eval.streaming import (
    DALIStreamerStream,
    TorchvisionGpuStreamer,
)

SEQ = Path("datasets/MOT17/train/MOT17-04-SDP/img1")
N = 300  # frames to time after warmup


def bench(name, streamer_factory, gpu_sync):
    it = iter(streamer_factory())
    # warmup
    for _ in range(20):
        next(it)
    if gpu_sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(N):
        frame = next(it)
        n += 1
    if gpu_sync:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(
        f"{name:24s}  {dt * 1e3 / n:7.3f} ms/frame   "
        f"{n / dt:7.1f} fps   (shape={tuple(frame.shape)}, dev={frame.device})"
    )


if __name__ == "__main__":
    bench("DALI (CPU/libjpeg)", lambda: DALIStreamerStream(SEQ), gpu_sync=False)
    bench(
        "torchvision (GPU/nvJPEG)",
        lambda: TorchvisionGpuStreamer(SEQ),
        gpu_sync=True,
    )
