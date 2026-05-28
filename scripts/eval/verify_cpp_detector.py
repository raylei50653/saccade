#!/usr/bin/env python3
"""Validation and benchmark script for C++ LibTorch + TensorRT MambaGatedDetector.

Usage:
    uv run scripts/eval/verify_cpp_detector.py \
        --trt-engine models/yolo/yolo26s_backbone_640_best.engine \
        --mamba-head models/yolo/mamba_head_best.pt \
        --seq MOT17-04-SDP --max-frames 100
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch

# Force cuBLAS and PyTorch CUDA context initialization
_ = torch.zeros(1, device="cuda") @ torch.zeros(1, device="cuda")

import saccade_tracking_ext  # noqa: F401
from saccade_perception_ext import MambaGatedDetector
from saccade.perception.tracking.tracker_gpu import GPUByteTracker
from saccade.perception.temporal_yolo.data_pipeline import resize_stretch_batch_gpu
from saccade.perception.eval.metrics import run_motmetrics_evaluation
from scripts.eval.eval_fpn_reid import load_sequence_frames, TopKAppearanceBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trt-engine", default="models/yolo/yolo26s_backbone_640_best.engine"
    )
    parser.add_argument("--mamba-head", default="models/yolo/mamba_head_best.pt")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--seq", default="MOT17-04-SDP")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output", default="output/fpn_reid_cpp")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cost-cos-w", type=float, default=0.10)
    parser.add_argument("--cost-iou-w", type=float, default=0.75)
    parser.add_argument("--cost-score-w", type=float, default=0.15)
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    parser.add_argument("--reid-weight", type=float, default=0.10)
    parser.add_argument("--cos-threshold", type=float, default=0.55)
    parser.add_argument("--iou-low", type=float, default=0.30)
    parser.add_argument("--iou-high", type=float, default=0.60)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--bank-k", type=int, default=5)
    parser.add_argument("--bank-alpha", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequence
    print(f"\nLoading {args.seq}...")
    frames = load_sequence_frames(args.seq, args.data_root, args.max_frames)
    h_orig, w_orig = frames[0].shape[:2]
    print(f"  Frames: {len(frames)}  Resolution: {w_orig}×{h_orig}")

    # Force PyTorch CUDA/cuBLAS context initialization
    _ = torch.zeros(1, device="cuda")

    # Build C++ MambaGatedDetector
    print("\nBuilding C++ LibTorch + TensorRT MambaGatedDetector...")
    detector = MambaGatedDetector(
        trt_backbone_path=str((project_root / args.trt_engine).resolve()),
        mamba_head_script_path=str((project_root / args.mamba_head).resolve()),
        img_size=args.img_size,
        conf_thr=args.conf_thresh,
    )

    fpn_dim = detector.fpn_dim
    print(f"  C++ Detector FPN Dim: {fpn_dim}")

    # Tracker
    tracker = GPUByteTracker(max_objects=2048, embedding_dim=fpn_dim)
    tracker.set_params(
        track_thresh=0.05,
        high_thresh=0.45,
        match_thresh=0.66,
        track_buffer=30,
        mid_thresh=0.10,
        new_track_thresh=0.28,
    )
    tracker.set_reid_params(
        cos_threshold=args.cos_threshold,
        iou_low=args.iou_low,
        iou_high=args.iou_high,
        weight=args.reid_weight,
        cost_cos_w=args.cost_cos_w,
        cost_iou_w=args.cost_iou_w,
        cost_score_w=args.cost_score_w,
    )

    result_bufs = tracker.allocate_result_buffers()

    bank = TopKAppearanceBank(
        max_samples=args.bank_k,
        min_samples=2,
        consistency_threshold=0.85,
        ema_alpha=args.bank_alpha,
    )

    print(f"\nRunning C++ accelerated tracking on {len(frames)} frames...")
    t0 = time.perf_counter()

    mot_lines: list[str] = []

    # Pre-allocate output buffers on GPU to achieve maximum speed
    max_dets = 1000
    d_out_dets = torch.empty((max_dets, 6), dtype=torch.float32, device=device)
    d_out_embs = torch.empty((max_dets, fpn_dim), dtype=torch.float32, device=device)

    # Run inference entirely inside C++ context
    with torch.inference_mode():
        for frame_idx, frame_np in enumerate(frames):
            frame_uint8 = (
                torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            frame_640 = resize_stretch_batch_gpu(frame_uint8, args.img_size, device)

            # ── Run C++ Detector Forward ──
            n_dets = detector.forward_ptr(frame_640.data_ptr(), d_out_dets.data_ptr())
            detections_raw = d_out_dets[:n_dets]

            boxes = detections_raw[:, :4]
            scores = detections_raw[:, 4]
            class_ids = detections_raw[:, 5].long()

            # Rescale boxes from 640 space to original resolution
            scale_x = float(w_orig) / args.img_size
            scale_y = float(h_orig) / args.img_size
            boxes_track = boxes.clone()
            boxes_track[:, 0] *= scale_x
            boxes_track[:, 1] *= scale_y
            boxes_track[:, 2] *= scale_x
            boxes_track[:, 3] *= scale_y

            # ── Run C++ FPN Embeddings Extraction ──
            embeddings = None
            if n_dets > 0 and not args.no_reid:
                detector.extract_fpn_embeddings_ptr(
                    boxes.data_ptr(), n_dets, d_out_embs.data_ptr()
                )
                embeddings = d_out_embs[:n_dets]

                # ── Push bank to tracker ──
                if len(bank) > 0 and not args.no_reid:
                    reps = bank.representatives(device)
                    _ids_list = sorted(reps.keys())
                    _ids = torch.tensor(_ids_list, device=device, dtype=torch.int32)
                    _feats = torch.stack([reps[k] for k in _ids_list])
                    tracker.update_reference_features(_ids, _feats)
                    _clean = bank.clean_ids()
                    if _clean:
                        _cids = torch.tensor(
                            sorted(_clean), device=device, dtype=torch.int32
                        )
                        tracker.set_clean_embedding_flags(
                            _cids,
                            torch.ones(len(_clean), device=device, dtype=torch.bool),
                        )

            # ── Track ──
            tracker.update_into(
                boxes_track,
                scores,
                class_ids,
                result_bufs,
                embeddings=embeddings,
            )

            # ── Update bank ──
            count = result_bufs["count"].item()
            for i in range(count):
                x1 = result_bufs["boxes"][i, 0].item()
                y1 = result_bufs["boxes"][i, 1].item()
                x2 = result_bufs["boxes"][i, 2].item()
                y2 = result_bufs["boxes"][i, 3].item()
                tid = result_bufs["ids"][i].item()
                score = result_bufs["scores"][i].item()
                w, h = x2 - x1, y2 - y1
                mot_lines.append(
                    f"{frame_idx + 1},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
                )

            active_tids: set[int] = set()
            if count > 0 and embeddings is not None and not args.no_reid:
                track_ids = result_bufs["ids"][:count]
                det_indices = result_bufs["det_idx"][:count]
                matched = (det_indices >= 0) & (det_indices < n_dets)
                active_tids.update(int(t) for t in track_ids.tolist())
                if matched.any():
                    for tid, didx in zip(
                        track_ids[matched].tolist(), det_indices[matched].tolist()
                    ):
                        bank.update(int(tid), embeddings[int(didx)])

            if bank:
                bank.prune(active_tids)

            if frame_idx % 20 == 0 or frame_idx == len(frames) - 1:
                print(
                    f"  frame {frame_idx:4d}: {n_dets} dets, "
                    f"{count} tracks, bank={len(bank)}",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed
    print(
        f"\nDone. C++ Pipeline processed {len(frames)} frames in {elapsed:.1f}s ({fps:.1f} FPS)"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_output = output_dir / f"{args.seq}.txt"
    seq_output.write_text("\n".join(mot_lines))
    print(f"Saved {len(mot_lines)} MOT lines to {seq_output}")

    try:
        metrics = run_motmetrics_evaluation(
            data_root=str(Path(args.data_root) / "MOT17"),
            split=args.split,
            output=str(output_dir),
            sequences=args.seq,
            detector=None,
        )
        if metrics:
            print("\n=== METRICS ===")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")


if __name__ == "__main__":
    main()
