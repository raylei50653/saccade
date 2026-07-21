#!/usr/bin/env python3
"""Attribute the eval `detect` stage (mamba_optimal/SDP ~7.45ms) into sub-components.

Mirrors the eval call path exactly: detect_native_640 -> detector.detect_raw ->
MambaGatedDetector.forward(frame, gate_input=None). Wraps the 4 heavy sub-calls
(TRT backbone / gate / mamba_head SSM / postprocess decoder) in sync-free CUDA
events and reports per-component mean + the residual (inline temporal-buffer/flow
Python glue). Pure measurement; no tracking logic touched.
"""
# status: diagnostic

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

# Import TRT detector first to avoid libjpeg conflict (same as mot17.py).
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
import torch  # noqa: E402
from saccade.perception.temporal_yolo import mamba_gated_detector as mgd  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

WARMUP = 30
ITERS = 200
IMG = 640


class _Timer:
    """Accumulates CUDA-event elapsed ms across iters for one named component."""

    def __init__(self) -> None:
        self.pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.total_ms = 0.0
        self.n = 0
        self.active = False

    def wrap(self, fn):
        def inner(*a, **kw):
            if not self.active:
                return fn(*a, **kw)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(torch.cuda.current_stream())
            out = fn(*a, **kw)
            e.record(torch.cuda.current_stream())
            self.pairs.append((s, e))
            return out

        return inner

    def harvest(self) -> None:
        for s, e in self.pairs:
            self.total_ms += s.elapsed_time(e)
            self.n += 1
        self.pairs.clear()

    @property
    def mean(self) -> float:
        return self.total_ms / self.n if self.n else 0.0


def main() -> None:
    torch.backends.cudnn.benchmark = True
    print("Building mamba_optimal detector (TRT backbone, temporal v14)...")
    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=IMG,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine="models/yolo/yolo26s_backbone_640_best.engine",
        temporal_T_override=None,
    )
    det.eval()
    print(
        f"  temporal_T={det._temporal_T}  trt_backbone={det._trt_backbone is not None}"
    )

    t_backbone, t_gate, t_head, t_post = _Timer(), _Timer(), _Timer(), _Timer()
    det._trt_backbone.infer = t_backbone.wrap(det._trt_backbone.infer)  # type: ignore[union-attr]
    det._apply_gate = t_gate.wrap(det._apply_gate)  # type: ignore[method-assign]
    det.mamba_head.forward = t_head.wrap(det.mamba_head.forward)  # type: ignore[method-assign]
    mgd._postprocess_mamba = t_post.wrap(mgd._postprocess_mamba)

    frame = torch.rand(1, 3, IMG, IMG, device="cuda")
    # Populate GMC buffer so the temporal flow-gate path runs (realistic worst case).
    for _ in range(det._temporal_T):
        det.set_gmc_warp(torch.eye(2, 3, device="cuda"))

    timers = [t_backbone, t_gate, t_head, t_post]

    forward_wall: list[float] = []
    with torch.no_grad():
        for it in range(WARMUP + ITERS):
            active = it >= WARMUP
            for t in timers:
                t.active = active
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            det.detect_raw(frame)
            torch.cuda.synchronize()
            if active:
                forward_wall.append((time.perf_counter() - t0) * 1000.0)

    for t in timers:
        t.harvest()

    wall_mean = sum(forward_wall) / len(forward_wall)
    comp = {
        "trt_backbone": t_backbone.mean,
        "gate(identity)": t_gate.mean,
        "mamba_head(SSM)": t_head.mean,
        "postprocess_decode": t_post.mean,
    }
    comp_sum = sum(comp.values())
    residual = wall_mean - comp_sum  # temporal-buffer/flow Python glue + launch

    print(f"\n=== detect breakdown  (img={IMG}, iters={ITERS}, dummy frame) ===")
    print(f"{'forward wall (sync)':<24}{wall_mean:7.3f} ms   (100.0%)")
    print("-" * 52)
    for name, ms in comp.items():
        print(f"{name:<24}{ms:7.3f} ms   ({100 * ms / wall_mean:5.1f}%)")
    print(
        f"{'residual (temporal/glue)':<24}{residual:7.3f} ms   ({100 * residual / wall_mean:5.1f}%)"
    )
    print("-" * 52)
    print(f"{'Σ components':<24}{comp_sum:7.3f} ms")


if __name__ == "__main__":
    main()
