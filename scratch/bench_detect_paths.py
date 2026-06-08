"""Benchmark Mamba whole-graph detect paths:

  A. detect_raw(frame_1080p)            -- baseline: F.interpolate(1080->640) INSIDE the graph
  B. detect_raw_preprocessed(canvas640) -- NV12 path: resize OUTSIDE the graph, graph takes 640 直接

Goal: explain why the NV12 fused preprocess kernel is faster in isolation, yet the
full eval is slower. We isolate (1) graph replay cost, (2) the preprocessing step
that feeds each path, and (3) the per-frame Python box-rescale tail in detection.py.

Run with the cublas LD_PRELOAD shim (same as scripts/eval/mot17.py NV12 path):

  NV=.venv/lib/python3.12/site-packages/nvidia/cu13/lib
  LD_PRELOAD="$PWD/$NV/libcublasLt.so.13:$PWD/$NV/libcublas.so.13" \
      .venv/bin/python scratch/bench_detect_paths.py
"""

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
_build = Path(os.environ.get("SACCADE_BUILD_PATH", _root / "build"))
if _build.exists():
    sys.path.insert(0, str(_build))

# MUST import saccade_tracking_ext BEFORE torch/torchvision to avoid libjpeg/libtiff
# symbol conflict (same pattern as scripts/eval/mot17.py)
from saccade_tracking_ext import (  # noqa: E402
    rgb_to_nv12_gpu as _nv12_convert_kernel,
    nv12_to_chw_resize as _nv12_resize_kernel,
    box_rescale_inplace as _box_rescale_kernel,
)

import torch  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.eval.pool import AdaptiveFramePool, rgb_hwc_to_nv12_gpu  # noqa: E402

WARMUP = 20
REPEATS = 200
H, W = 1080, 1920

MAMBA_CKPT = "runs/mamba_gt_vgt_mamba_v14/best.ckpt"
TEACHER_CKPT = "runs/gated_det_v1/best.ckpt"
BACKBONE = "models/yolo/yolo26s_backbone_640_best.engine"


def cuda_timed(fn) -> float:
    """Mean ms over REPEATS, GPU-side via CUDA events (excludes nothing python-side
    because we record around the whole fn on the default stream)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(REPEATS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEATS


def main() -> None:
    torch.manual_seed(0)
    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt=TEACHER_CKPT,
        mamba_ckpt=MAMBA_CKPT,
        img_size=640,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine=BACKBONE,
        temporal_T_override=0,
        use_cuda_graph=True,
        use_whole_graph=True,
    )
    det.eval()

    # Frame buffers
    frame_hwc_u8 = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, device="cuda")
    frame_chw = frame_hwc_u8.permute(2, 0, 1).float() / 255.0
    frame_chw = frame_chw.contiguous()

    pool = AdaptiveFramePool(H, W)
    pool.use_nv12 = True
    pool.frame_buffer_nv12.copy_(rgb_hwc_to_nv12_gpu(frame_hwc_u8))
    pool.mark_nv12_current()

    rgb_pool = AdaptiveFramePool(H, W)
    rgb_pool.frame_buffer.copy_(frame_chw)

    # Trigger graph capture once for each path
    print("Capturing graphs...")
    _ = det.detect_raw(frame_chw.unsqueeze(0))
    canvas0 = pool.prepare_canvas_640_stretch(H, W)
    _ = det.detect_raw_preprocessed(canvas0.unsqueeze(0))
    torch.cuda.synchronize()

    # ---- 1. Graph replay only (the actual detect compute) ----
    def replay_baseline():
        det.detect_raw(frame_chw.unsqueeze(0))

    def replay_preprocessed():
        canvas = pool.canvas_640p  # reuse, isolate replay
        det.detect_raw_preprocessed(canvas.unsqueeze(0))

    t_base_graph = cuda_timed(replay_baseline)
    t_pre_graph = cuda_timed(replay_preprocessed)

    # ---- 2. Preprocessing step that feeds each path ----
    # baseline feeds frame_buffer directly (interpolate is INSIDE the graph, so
    # the "preprocess" for baseline is just the ingest copy permute/float).
    def ingest_baseline():
        rgb_pool.frame_buffer.copy_(frame_hwc_u8.permute(2, 0, 1).float() / 255.0)

    # nv12 path: ingest = rgb->nv12, then detect-time resize nv12->640 canvas
    def ingest_nv12():
        pool.frame_buffer_nv12.copy_(rgb_hwc_to_nv12_gpu(frame_hwc_u8))

    def resize_nv12_canvas():
        pool.prepare_canvas_640_stretch(H, W)

    t_ingest_base = cuda_timed(ingest_baseline)
    t_ingest_nv12 = cuda_timed(ingest_nv12)
    t_resize_nv12 = cuda_timed(resize_nv12_canvas)

    # ---- 3. Full detect_single_patch_640-equivalent (incl. box rescale tail) ----
    def full_baseline():
        raw = det.detect_raw(rgb_pool.frame_buffer.unsqueeze(0))
        _ = raw[0, :, :4], raw[0, :, 4], raw[0, :, 5]

    def full_preprocessed():
        canvas = pool.prepare_canvas_640_stretch(H, W)
        raw = det.detect_raw_preprocessed(canvas.unsqueeze(0))
        _box_rescale_kernel(
            raw.data_ptr(),
            raw.shape[0] * raw.shape[1],
            float(W / 640.0),
            float(H / 640.0),
            torch.cuda.current_stream().cuda_stream,
        )
        _ = raw[0, :, :4], raw[0, :, 4], raw[0, :, 5]

    t_full_base = cuda_timed(full_baseline)
    t_full_pre = cuda_timed(full_preprocessed)

    print("\n=== Mamba whole-graph detect path benchmark (1080p->640) ===")
    print(f"{'stage':<42}{'ms':>10}")
    print("-" * 52)
    print(f"{'[A] baseline graph replay (1080p in)':<42}{t_base_graph:>10.4f}")
    print(f"{'[B] preproc graph replay (640 in)':<42}{t_pre_graph:>10.4f}")
    print("-" * 52)
    print(f"{'baseline ingest (hwc->chw f32)':<42}{t_ingest_base:>10.4f}")
    print(f"{'nv12 ingest (rgb->nv12)':<42}{t_ingest_nv12:>10.4f}")
    print(f"{'nv12 detect-time resize (nv12->640)':<42}{t_resize_nv12:>10.4f}")
    print("-" * 52)
    print(f"{'[A] FULL baseline (graph+tail)':<42}{t_full_base:>10.4f}")
    print(f"{'[B] FULL preproc (resize+graph+box tail)':<42}{t_full_pre:>10.4f}")
    print("-" * 52)

    base_total = t_ingest_base + t_full_base
    pre_total = t_ingest_nv12 + t_full_pre
    print(f"{'baseline per-frame (ingest+full)':<42}{base_total:>10.4f}")
    print(f"{'nv12     per-frame (ingest+full)':<42}{pre_total:>10.4f}")
    print(f"{'delta (nv12 - baseline)':<42}{pre_total - base_total:>+10.4f}")


if __name__ == "__main__":
    main()
