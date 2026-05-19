#!/usr/bin/env python3
"""
Eval script for GatedYOLODetector + GPUByteTracker.

Supports optional YOLO ROI embedding ReID (--roi-reid):
  - P3/P4/P5 ROI-avg-pool → 896-dim L2-normalised embedding per detection
  - Quality-weighted EMA bank denoising across frames
  - Passed to GPUByteTracker as appearance features for lost-track re-association

Usage:
    uv run scripts/eval/eval_gated_bytetrack.py \
        --ckpt runs/gated_det_v1/best.ckpt \
        [--no-gate] [--roi-reid] [--reid-cos-thr 0.5] [--reid-weight 0.5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import tracker first: torchvision.ops (loaded by roi_embedder) poisons saccade_tracking_ext
# if torchvision is imported first due to a libtiff symbol conflict on this system.
from saccade.perception.tracking.tracker_gpu import GPUByteTracker  # noqa: E402
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
)
from saccade.perception.temporal_yolo.roi_embedder import (  # noqa: E402
    ROIEmbeddingBank,
    extract_roi_embeddings,
    EMB_DIM,
)

_SDP_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_frames(seq_dir: Path, img_size: int) -> list[tuple[int, torch.Tensor]]:
    import torchvision.io as tv_io

    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    result = []
    for p in paths:
        fid = int(p.stem)
        img = tv_io.read_image(str(p)).float() / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        result.append((fid, img))
    return result


def _orig_hw(seq_dir: Path) -> tuple[int, int]:
    ini = seq_dir / "seqinfo.ini"
    w, h = 1920, 1080
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth"):
                w = int(line.split("=")[1])
            elif line.startswith("imHeight"):
                h = int(line.split("=")[1])
    return h, w


def _to_mot_lines(frame_id: int, track_results) -> list[str]:
    lines = []
    for tr in track_results:
        w, h = tr.x2 - tr.x1, tr.y2 - tr.y1
        lines.append(
            f"{frame_id},{tr.obj_id},{tr.x1:.2f},{tr.y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1"
        )
    return lines


# ---------------------------------------------------------------------------
# Per-sequence eval
# ---------------------------------------------------------------------------
def eval_sequence(
    model,
    tracker: GPUByteTracker,
    seq_dir: Path,
    output_dir: Path,
    img_size: int,
    conf_threshold: float,
    device: torch.device,
    disable_gate: bool = False,
    use_roi_reid: bool = False,
    dim_selector: torch.Tensor
    | None = None,  # (K,) top-K dim indices from Fisher analysis
) -> int:
    seq_name = seq_dir.name
    frames = _load_frames(seq_dir, img_size)
    orig_h, orig_w = _orig_hw(seq_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker.tracker.reset() if hasattr(tracker.tracker, "reset") else None
    tracker.set_frame_size(orig_w, orig_h)

    sx, sy = orig_w / img_size, orig_h / img_size
    prev_boxes_640: torch.Tensor | None = None
    prev_scores: torch.Tensor | None = None

    # ROI ReID bank: reset per sequence
    emb_bank = ROIEmbeddingBank(img_size=img_size) if use_roi_reid else None

    mot_lines: list[str] = []
    t0 = time.perf_counter()

    for frame_id, frame in frames:
        frame = frame.to(device)

        # Gate input from previous frame
        gate_input = None
        if (
            not disable_gate
            and prev_boxes_640 is not None
            and prev_boxes_640.numel() > 0
        ):
            gate_input = TrackerGateInput.from_boxes_scores(
                prev_boxes_640,
                prev_scores,
                (img_size, img_size),
                assume_absolute=True,
            ).to(device)

        # Forward — also caches FPN features if roi_reid enabled
        if use_roi_reid:
            model.cache_feats = True
            model._feat_cache.clear()

        with torch.no_grad():
            out = model(frame, gate_input=gate_input)

        if use_roi_reid:
            model.cache_feats = False

        raw = out[0][0]  # (300, 6)
        keep = raw[:, 4] > conf_threshold
        dets = raw[keep]

        embeddings = None
        if dets.numel() > 0:
            prev_boxes_640 = dets[:, :4].clone()
            prev_scores = dets[:, 4].clone()

            boxes_orig = dets[:, :4].clone()
            boxes_orig[:, [0, 2]] *= sx
            boxes_orig[:, [1, 3]] *= sy
            scores = dets[:, 4]
            classes = dets[:, 5].to(torch.int32)

            # Extract ROI embeddings (640px space)
            if use_roi_reid and model._feat_cache:
                embeddings = extract_roi_embeddings(
                    model._feat_cache, dets[:, :4]
                )  # (N, 896) — in 640px space
                if dim_selector is not None:
                    # Select top-K Fisher dims and re-normalize to reduce noise
                    embeddings = torch.nn.functional.normalize(
                        embeddings[:, dim_selector], dim=1
                    )
        else:
            prev_boxes_640 = None
            prev_scores = None
            boxes_orig = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            classes = torch.zeros((0,), device=device, dtype=torch.int32)

        # tracker.update must run outside inference_mode (inference tensors cause silent failures)
        track_results = tracker.update(
            boxes_orig, scores, classes, embeddings=embeddings
        )

        # Update ROI bank for confirmed tracks, then inject smoothed embeddings as reference
        if (
            use_roi_reid
            and emb_bank is not None
            and embeddings is not None
            and dets.numel() > 0
        ):
            matched_ids = [
                tr.obj_id
                for tr in track_results
                if tr.det_idx >= 0 and tr.det_idx < len(embeddings)
            ]
            matched_idx = [
                tr.det_idx
                for tr in track_results
                if tr.det_idx >= 0 and tr.det_idx < len(embeddings)
            ]
            if matched_ids:
                m_idx = torch.tensor(matched_idx, dtype=torch.long)
                emb_bank.update(
                    matched_ids,
                    embeddings[m_idx],
                    scores[m_idx],
                    dets[m_idx, :4],
                )

        # Inject bank's EMA-smoothed embeddings into tracker's reference buffer
        # so lost-track re-association uses denoised history, not raw per-frame embeddings
        if use_roi_reid and emb_bank is not None and tracker.is_cuda:
            tracker.set_reference_features_from_bank(emb_bank._bank)

        mot_lines.extend(_to_mot_lines(frame_id, track_results))

    elapsed = time.perf_counter() - t0
    fps = len(frames) / max(elapsed, 1e-6)
    (output_dir / f"{seq_name}.txt").write_text("\n".join(mot_lines))
    reid_str = f"  bank={len(emb_bank)}" if emb_bank else ""
    print(
        f"  {seq_name}: {len(frames)} frames, {fps:.1f} FPS, "
        f"{len(mot_lines)} tracks{reid_str}"
    )
    return len(frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--output", default="/tmp/gated_det_bytetrack")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument(
        "--roi-reid",
        action="store_true",
        help="Enable YOLO ROI embedding for lost-track ReID",
    )
    # ByteTrack params
    parser.add_argument("--track-thresh", type=float, default=0.05)
    parser.add_argument("--high-thresh", type=float, default=0.45)
    parser.add_argument("--match-thresh", type=float, default=0.66)
    parser.add_argument("--new-track-thresh", type=float, default=0.28)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--fuse-score-weight", type=float, default=0.4)
    # ReID params
    parser.add_argument(
        "--reid-cos-thr",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for lost-track association",
    )
    parser.add_argument(
        "--reid-weight",
        type=float,
        default=0.5,
        help="Appearance cost weight in association (0=IoU only)",
    )
    parser.add_argument(
        "--dim-select",
        default="",
        help="Path to .npy file with top-K Fisher dim indices (from analyze_roi_dim_importance.py)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))

    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
        gate_min_score=train_args.get("gate_min_score", 0.5),
        freeze_backbone=False,
    )
    yolo_weights = project_root / train_args.get(
        "yolo_weights", "models/yolo/yolo26s.pt"
    )
    model = build_gated_yolo_detector(
        str(yolo_weights), cfg=cfg, device=device, weights_path=str(ckpt_path)
    )
    model.eval()
    print(
        f"Loaded: {ckpt_path}  epoch={raw.get('epoch')}  loss={raw.get('best_loss', 0):.4f}"
    )
    print(
        f"Gate {'DISABLED' if args.no_gate else 'ENABLED'}  "
        f"ROI-ReID {'ENABLED' if args.roi_reid else 'DISABLED'}"
    )
    # Load dim selector from Fisher analysis if provided
    dim_selector: torch.Tensor | None = None
    if args.dim_select and Path(args.dim_select).exists():
        import numpy as np

        dim_selector = torch.from_numpy(np.load(args.dim_select)).long().to(device)
        print(
            f"  dim_selector: top-{len(dim_selector)} Fisher dims from {args.dim_select}"
        )

    if args.roi_reid:
        active_dim = len(dim_selector) if dim_selector is not None else EMB_DIM
        print(
            f"  embedding_dim={active_dim}  cos_thr={args.reid_cos_thr}  "
            f"reid_weight={args.reid_weight}"
        )

    emb_dim = (
        (len(dim_selector) if dim_selector is not None else EMB_DIM)
        if args.roi_reid
        else 768
    )
    tracker = GPUByteTracker(max_objects=2048, embedding_dim=emb_dim)
    tracker.set_params(
        track_thresh=args.track_thresh,
        high_thresh=args.high_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        new_track_thresh=args.new_track_thresh,
        fuse_score_weight=args.fuse_score_weight,
    )
    if args.roi_reid:
        tracker.set_reid_params(
            cos_threshold=args.reid_cos_thr,
            iou_low=0.1,
            iou_high=0.5,
            weight=args.reid_weight,
        )

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()] or _SDP_SEQS
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    output_dir = Path(args.output)

    print(f"\nEvaluating {len(seqs)} sequences → {output_dir}")
    for seq in seqs:
        seq_dir = data_root / args.split / seq
        if not seq_dir.exists():
            print(f"  [SKIP] {seq_dir}")
            continue
        eval_sequence(
            model,
            tracker,
            seq_dir,
            output_dir,
            args.img_size,
            args.conf_threshold,
            device,
            disable_gate=args.no_gate,
            use_roi_reid=args.roi_reid,
            dim_selector=dim_selector,
        )

    try:
        from saccade.perception.eval.metrics import run_motmetrics_evaluation

        metrics = run_motmetrics_evaluation(
            data_root=str(data_root),
            split=args.split,
            output=str(output_dir),
            sequences=",".join(seqs),
            detector="SDP",
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")


if __name__ == "__main__":
    main()
