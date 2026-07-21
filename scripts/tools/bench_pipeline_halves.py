"""Measure the GPU-time split between the two per-frame pipeline halves.

Spike for the odd/even double-buffer idea: in non-temporal mode the *detection
half* (TRT backbone + Mamba head + decode, bundled in the whole-graph replay) is
frame-independent and could run ahead on a second stream/buffer, while the
*tracker half* (GMC + tracker.update + materialize D2H) is a strict per-frame
state machine and cannot. The realistic win of double-buffering is therefore
bounded by how much of each frame's GPU time the tracker half occupies:

    overlap_ceiling = track_half / (detect_half + track_half)

This script times both halves per frame on real MOT17 frames with sync-free
CUDA events (one elapsed read per frame) and prints mean/median/p95 + the
ceiling. It does NOT touch the eval loop or change any production behaviour.

GMC is intentionally omitted (gmc=None) — it is additional *sequential* work, so
the real tracker-half share (and thus the ceiling) is at least what we report.

Run:
    .venv/bin/python scripts/tools/bench_pipeline_halves.py \
        --seq MOT17-02-SDP --frames 200 --warmup 20
"""
# status: experiment

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch


def _load_frames(seq_dir: Path, n: int) -> tuple[list[torch.Tensor], int, int]:
    """Load up to n JPEG frames as (1,3,H,W) float CUDA tensors in [0,1]."""
    from torchvision.io import read_image

    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg"))[:n]
    if not paths:
        raise FileNotFoundError(f"no frames under {img_dir}")
    frames: list[torch.Tensor] = []
    for p in paths:
        img = read_image(str(p)).cuda().float() / 255.0  # (3,H,W)
        frames.append(img.unsqueeze(0).contiguous())
    _, _, h, w = frames[0].shape
    return frames, h, w


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MOT17-02-SDP")
    ap.add_argument("--data-root", default="datasets/MOT17/train")
    ap.add_argument("--mamba-ckpt", default="runs/mamba_gt_vgt_mamba_v14/best.ckpt")
    ap.add_argument(
        "--backbone", default="models/yolo/yolo26s_backbone_640_best.engine"
    )
    ap.add_argument("--yolo-pt", default="models/yolo/yolo26s.pt")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
        set_postprocess_compile,
    )
    from saccade.perception.eval.helpers import (
        materialize_gpu_track_results_async,
    )

    seq_dir = Path(args.data_root) / args.seq
    frames, h_orig, w_orig = _load_frames(seq_dir, args.frames + args.warmup)
    print(f"loaded {len(frames)} frames @ {w_orig}x{h_orig} from {seq_dir}")

    detector = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_pt,
        teacher_ckpt="",
        mamba_ckpt=args.mamba_ckpt,
        img_size=640,
        device="cuda",
        conf_thr=args.conf,
        max_det=300,
        trt_backbone_engine=args.backbone,
        temporal_T_override=0,  # non-temporal: the case double-buffering targets
        use_cuda_graph=True,
        use_whole_graph=True,
    )
    set_postprocess_compile(True)
    detector.mamba_head.set_head_compile(True)
    detector.set_whole_graph_img_dims(h_orig, w_orig)
    detector.tracker.set_frame_size(w_orig, h_orig)

    result_buffers = detector.tracker.allocate_result_buffers(device="cuda")
    max_obj = detector.tracker.max_objects
    pinned = {
        "boxes": torch.empty((max_obj, 4), dtype=torch.float32, pin_memory=True),
        "scores": torch.empty((max_obj,), dtype=torch.float32, pin_memory=True),
        "ids": torch.empty((max_obj,), dtype=torch.int32, pin_memory=True),
        "classes": torch.empty((max_obj,), dtype=torch.int32, pin_memory=True),
        "det_idx": torch.empty((max_obj,), dtype=torch.int32, pin_memory=True),
        "count": torch.empty((), dtype=torch.int32, pin_memory=True),
    }

    def detect_half(frame: torch.Tensor) -> torch.Tensor:
        return detector.detect_raw(frame)  # (1, max_det, 6), original coords

    def track_half(dets: torch.Tensor) -> torch.cuda.Event:
        boxes = dets[0, :, :4]
        scores = dets[0, :, 4]
        classes = dets[0, :, 5].to(torch.int32)
        keep = scores > args.conf
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        detector.tracker.update_into(boxes, scores, classes, result_buffers)
        ev, _ = materialize_gpu_track_results_async(
            result_buffers, pinned, default_class_id=None, include_det_idx=False
        )
        return ev

    # Warmup: graph capture + tracker spin-up.
    for i in range(args.warmup):
        track_half(detect_half(frames[i]))
    torch.cuda.synchronize()

    work = frames[args.warmup :]
    n = len(work)

    # --- (1) Per-half GPU-span breakdown (per-frame sync to read elapsed) -----
    detect_ms: list[float] = []
    track_ms: list[float] = []
    n_dets: list[int] = []
    ev_a = torch.cuda.Event(enable_timing=True)
    ev_b = torch.cuda.Event(enable_timing=True)
    ev_c = torch.cuda.Event(enable_timing=True)
    for f in work:
        ev_a.record(torch.cuda.current_stream())
        dets = detect_half(f)
        ev_b.record(torch.cuda.current_stream())
        _ = track_half(dets)
        ev_c.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        detect_ms.append(ev_a.elapsed_time(ev_b))
        track_ms.append(ev_b.elapsed_time(ev_c))
        n_dets.append(int(pinned["count"].item()))
    d_mean, t_mean = statistics.mean(detect_ms), statistics.mean(track_ms)
    total = d_mean + t_mean

    print(f"\n=== {args.seq}: {n} frames, {statistics.mean(n_dets):.0f} obj/frame ===")
    print(f"{'half':<12}{'mean':>8}{'median':>9}{'p95':>8}  (ms, GPU span)")
    print(
        f"{'detect':<12}{d_mean:>8.3f}{statistics.median(detect_ms):>9.3f}"
        f"{_percentile(detect_ms, 95):>8.3f}"
    )
    print(
        f"{'track+mat':<12}{t_mean:>8.3f}{statistics.median(track_ms):>9.3f}"
        f"{_percentile(track_ms, 95):>8.3f}"
    )
    print(f"{'sum':<12}{total:>8.3f}   tracker share {t_mean / total * 100:.1f}%")

    # --- (2) Serial wall-clock (single sync at end) --------------------------
    def run_serial() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for f in work:
            track_half(detect_half(f))
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    # --- (3) Double-buffered wall-clock: detect(N) ‖ track(N-1) --------------
    # Two streams: detection runs on stream_det, tracker stays on the main
    # stream (sequential state machine). Each frame issues detect(N) *before*
    # track(N-1), so detect(N) overlaps track(N-1) on the GPU. detect output is
    # cloned (the whole-graph callable reuses one static output buffer) so
    # detect(N) does not clobber detect(N-1) while the tracker still reads it.
    stream_det = torch.cuda.Stream()
    main_stream = torch.cuda.current_stream()

    def detect_on_det_stream(f: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event]:
        with torch.cuda.stream(stream_det):
            dets = detector.detect_raw(f).clone()
            dets.record_stream(main_stream)  # consumed cross-stream by tracker
            ev = torch.cuda.Event()
            ev.record(stream_det)
        return dets, ev

    def run_pipeline() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dets_prev, ev_prev = detect_on_det_stream(work[0])
        for f in work[1:]:
            dets_cur, ev_cur = detect_on_det_stream(f)  # overlaps track below
            main_stream.wait_event(ev_prev)
            track_half(dets_prev)
            dets_prev, ev_prev = dets_cur, ev_cur
        main_stream.wait_event(ev_prev)
        track_half(dets_prev)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    # Correctness: pipeline must reproduce the serial per-frame track counts.
    def collect_counts(runner_kind: str) -> list[int]:
        counts: list[int] = []
        detector.reset_tracker()
        torch.cuda.synchronize()
        if runner_kind == "serial":
            for f in work:
                track_half(detect_half(f))
                torch.cuda.synchronize()
                counts.append(int(pinned["count"].item()))
        else:
            dp, ep = detect_on_det_stream(work[0])
            for f in work[1:]:
                dc, ec = detect_on_det_stream(f)
                main_stream.wait_event(ep)
                track_half(dp)
                torch.cuda.synchronize()
                counts.append(int(pinned["count"].item()))
                dp, ep = dc, ec
            main_stream.wait_event(ep)
            track_half(dp)
            torch.cuda.synchronize()
            counts.append(int(pinned["count"].item()))
        return counts

    c_ser, c_pipe = collect_counts("serial"), collect_counts("pipeline")
    mism = sum(a != b for a, b in zip(c_ser, c_pipe))
    print(f"\ncorrectness: {mism}/{len(c_ser)} frames with mismatched track count")
    detector.reset_tracker()

    serial = min(run_serial() for _ in range(3))
    pipe = min(run_pipeline() for _ in range(3))
    print(
        f"\nwall ms/frame   serial {serial:.3f}   double-buffer {pipe:.3f}   "
        f"speedup {serial / pipe:.3f}× ({(serial - pipe) / serial * 100:+.1f}%)"
    )
    print(f"theoretical floor max(detect,track) = {max(d_mean, t_mean):.3f} ms")


if __name__ == "__main__":
    main()
