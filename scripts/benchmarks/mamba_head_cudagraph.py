#!/usr/bin/env python3
"""Test the launch-bound hypothesis for the mamba_head: capture head.forward in a
CUDA graph (fixed shapes, single frame) and compare eager vs graph replay.

If the head is launch-bound, graph replay (one launch for the whole subgraph)
should be substantially faster than eager. If compute-bound, ~no change.
Pure measurement; identical math.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
import torch  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

WARMUP, ITERS, IMG = 50, 300, 640


def bench(fn) -> float:
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def main() -> None:
    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=IMG,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine="models/yolo/yolo26s_backbone_640_best.engine",
    )
    det.eval()
    head = det.mamba_head

    # Real FPN feature shapes for 640 input: P3 80x80/128, P4 40x40/256, P5 20x20/512.
    frame = torch.rand(1, 3, IMG, IMG, device="cuda")
    with torch.no_grad():
        p3, p4, p5 = det._trt_backbone.infer(frame)
    feats = [p3.clone(), p4.clone(), p5.clone()]
    print(f"feat shapes: {[tuple(f.shape) for f in feats]}")

    eager_ms = bench(lambda: head(feats))
    print(f"\nhead eager           : {eager_ms:7.3f} ms")

    # --- CUDA graph capture ---
    static_in = [f.clone() for f in feats]
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                head(static_in)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = head(static_in)

    def replay():
        for d, sv in zip(static_in, feats):
            d.copy_(sv)
        g.replay()
        return static_out

    graph_ms = bench(replay)
    print(f"head cudagraph replay: {graph_ms:7.3f} ms")
    print(
        f"\nspeedup: {eager_ms / graph_ms:.2f}x   saved {eager_ms - graph_ms:.3f} ms/frame"
    )
    print(
        "→ launch-bound"
        if graph_ms < eager_ms * 0.8
        else "→ compute-bound (graph gives little)"
    )


if __name__ == "__main__":
    main()
