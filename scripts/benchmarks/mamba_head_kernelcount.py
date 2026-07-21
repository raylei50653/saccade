#!/usr/bin/env python3
"""Count CUDA kernel launches in one mamba_head.forward, to locate the
launch-bound cost. The 4-direction cross-scan SSM is already 1 launch/scale
(well-batched); this shows how many OTHER small ops surround it.
"""
# status: diagnostic

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)


def main() -> None:
    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=640,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine="models/yolo/yolo26s_backbone_640_best.engine",
    )
    det.eval()
    head = det.mamba_head
    frame = torch.rand(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        feats = [f.clone() for f in det._trt_backbone.infer(frame)]
        for _ in range(10):
            head(feats)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(20):
                head(feats)
            torch.cuda.synchronize()

    evts = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
    total_launches = sum(e.count for e in evts) // 20  # per forward
    total_us = sum(e.self_device_time_total for e in evts) / 20.0
    print("\n=== mamba_head.forward: CUDA kernel profile (per forward) ===")
    print(f"kernel launches / forward : {total_launches}")
    print(f"GPU self time  / forward  : {total_us:.1f} us")
    print(f"avg per kernel            : {total_us / max(total_launches, 1):.2f} us\n")
    print(f"{'kernel':<48}{'count/fwd':>10}{'us/fwd':>10}")
    print("-" * 68)
    for e in sorted(evts, key=lambda e: -e.self_device_time_total)[:18]:
        name = e.key[:46]
        print(f"{name:<48}{e.count / 20:>10.1f}{e.self_device_time_total / 20:>10.1f}")


if __name__ == "__main__":
    main()
