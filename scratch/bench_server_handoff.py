#!/usr/bin/env python3
"""Measure the MultiStreamMambaServer inference throughput (no DALI / no
TrackEval) for three head paths, and verify event-handoff parity:

  * server head, default stream     (baseline: head runs on server thread)
  * event-handoff, per-stream stream (head runs on each worker's own stream)

Each "stream" pushes K preloaded GPU frames through proxy.detect_raw in its own
thread; we time the aggregate.
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

from saccade.perception.multistream_mamba_server import (  # noqa: E402
    MambaStreamProxy,
    MultiStreamMambaServer,
)

BACKBONE_B4 = str(ROOT / "models/yolo/yolo26s_backbone_640_batch4.engine")
CKPT = str(ROOT / "runs/mamba_gt_vgt_mamba_v14/best.ckpt")
N = 4
ITERS = 200


def build(event_handoff: bool) -> MultiStreamMambaServer:
    return MultiStreamMambaServer(
        backbone_engine=BACKBONE_B4,
        mamba_ckpt=CKPT,
        img_size=640,
        conf_thr=0.001,
        max_det=300,
        max_batch=N,
        event_handoff=event_handoff,
    )


def run(server, frames, iters):
    proxies = [MambaStreamProxy(server, i) for i in range(N)]

    def worker(proxy, frame, out_holder):
        if proxy.stream is not None:
            torch.cuda.set_stream(proxy.stream)
        last = None
        for _ in range(iters):
            last = proxy.detect_raw(frame)
        if proxy.stream is not None:
            proxy.stream.synchronize()
        else:
            torch.cuda.synchronize()
        out_holder.append(last)

    outs: list = [[] for _ in range(N)]
    threads = [
        threading.Thread(target=worker, args=(proxies[i], frames[i], outs[i]))
        for i in range(N)
    ]
    # warmup (head load / context) off the clock
    for p, f in zip(proxies, frames):
        p.detect_raw(f)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    return dt, (N * iters) / dt, [o[0] for o in outs]


def main():
    frames = [torch.rand(1, 3, 640, 640, device="cuda").contiguous() for _ in range(N)]

    print(f"streams={N}  iters/stream={ITERS}\n")

    srv = build(event_handoff=False)
    dt0, fps0, dets0 = run(srv, frames, ITERS)
    print(f"server head, default stream : {dt0:.2f}s  {fps0:7.1f} det/s")

    srv_h = build(event_handoff=True)
    dt1, fps1, dets1 = run(srv_h, frames, ITERS)
    print(
        f"event-handoff, per-stream   : {dt1:.2f}s  {fps1:7.1f} det/s  ({fps1 / fps0:.2f}x)"
    )

    # Parity: same frame -> same detections (head is identical, only the stream
    # / thread differs). dets are [1, max_det, 6] padded.
    print("\n=== parity (handoff vs server-head, per stream) ===")
    for i in range(N):
        a, b = dets0[i], dets1[i]
        d = (a - b).abs().max().item()
        print(f"  stream {i}: max|Δ|={d:.3e}  shape={tuple(b.shape)}")


if __name__ == "__main__":
    main()
