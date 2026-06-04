#!/usr/bin/env python3
"""Sub-attribute the mamba_head (~3ms of detect) into its module groups.

Uses forward pre/post hooks (sync-free CUDA events) on the head's submodule
ModuleLists + patches the module-level _cross_scan_mamba to isolate the SSM.
Mirrors eval: single-frame, gate_input=None, v14 cross-scan + pixel-shuffle.
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
from saccade.perception.temporal_yolo import mamba_head as mh_mod  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

WARMUP, ITERS, IMG = 30, 200, 640


class _Acc:
    def __init__(self) -> None:
        self.pairs: list[tuple] = []
        self.total = 0.0
        self.n = 0

    active = True

    def harvest(self) -> None:
        for s, e in self.pairs:
            self.total += s.elapsed_time(e)
            self.n += 1
        self.pairs.clear()

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


_ENABLED = {"v": False}


def hook_module(mod, acc: _Acc) -> None:
    def pre(_m, _inp):
        if not _ENABLED["v"]:
            return
        s = torch.cuda.Event(enable_timing=True)
        s.record(torch.cuda.current_stream())
        acc.pairs.append([s, None])

    def post(_m, _inp, _out):
        if not _ENABLED["v"]:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream())
        acc.pairs[-1][1] = e

    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)


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

    accs = {
        "input_proj": _Acc(),
        "downsample": _Acc(),
        "upsample(pxshuf)": _Acc(),
        "cls_head": _Acc(),
        "reg_head": _Acc(),
    }
    # Each is a ModuleList over 3 scales; hook every scale module → summed per iter.
    for name, key in [
        ("input_proj", "input_proj"),
        ("downsample", "downsample"),
        ("upsample", "upsample(pxshuf)"),
        ("cls_head", "cls_head"),
        ("reg_head", "reg_head"),
    ]:
        ml = getattr(head, name, None)
        if ml is not None:
            for sub in ml:
                hook_module(sub, accs[key])

    # Isolate the cross-scan SSM (module-level fn called inside head.forward).
    ssm = _Acc()
    orig_cross = mh_mod._cross_scan_mamba

    def timed_cross(*a, **kw):
        if not _ENABLED["v"]:
            return orig_cross(*a, **kw)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(torch.cuda.current_stream())
        out = orig_cross(*a, **kw)
        e.record(torch.cuda.current_stream())
        ssm.pairs.append((s, e))
        return out

    mh_mod._cross_scan_mamba = timed_cross

    # Whole-head wall (sync) for reference.
    frame = torch.rand(1, 3, IMG, IMG, device="cuda")
    head_wall: list[float] = []
    orig_head_fwd = head.forward

    def timed_head(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_head_fwd(*a, **kw)
        torch.cuda.synchronize()
        if _ENABLED["v"]:
            head_wall.append((time.perf_counter() - t0) * 1000.0)
        return out

    head.forward = timed_head  # type: ignore[method-assign]

    with torch.no_grad():
        for it in range(WARMUP + ITERS):
            _ENABLED["v"] = it >= WARMUP
            det.detect_raw(frame)

    for a in accs.values():
        a.harvest()
    ssm.harvest()

    wall = sum(head_wall) / len(head_wall)
    rows = {
        "input_proj (1x1)": accs["input_proj"].mean,
        "downsample (s4 conv)": accs["downsample"].mean,
        "cross_scan SSM": ssm.mean,
        "upsample (pixshuffle)": accs["upsample(pxshuf)"].mean,
        "cls_head (3x3+1x1)": accs["cls_head"].mean,
        "reg_head (3x3+1x1)": accs["reg_head"].mean,
    }
    ssum = sum(rows.values())
    print(f"\n=== mamba_head breakdown (img={IMG}, iters={ITERS}) ===")
    print(f"{'head wall (sync)':<24}{wall:7.3f} ms  (100.0%)")
    print("-" * 50)
    for k, v in rows.items():
        print(f"{k:<24}{v:7.3f} ms  ({100 * v / wall:5.1f}%)")
    print(
        f"{'residual (reshape/cat)':<24}{wall - ssum:7.3f} ms  ({100 * (wall - ssum) / wall:5.1f}%)"
    )
    print("-" * 50)
    print(f"{'Σ groups':<24}{ssum:7.3f} ms")


if __name__ == "__main__":
    main()
