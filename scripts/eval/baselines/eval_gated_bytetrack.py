#!/usr/bin/env python3
"""
Eval script for GatedYOLODetector + GPUByteTracker.

Supports optional YOLO ROI embedding ReID (--roi-reid):
  - P3/P4/P5 ROI-avg-pool → 896-dim L2-normalised embedding per detection
  - Quality-weighted EMA bank denoising across frames
  - Passed to GPUByteTracker as appearance features for lost-track re-association

Usage:
    uv run scripts/eval/baselines/eval_gated_bytetrack.py \
        --ckpt runs/gated_det_v1/best.ckpt \
        [--no-gate] [--roi-reid] [--reid-cos-thr 0.5] [--reid-weight 0.5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
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
from saccade.perception.temporal_yolo.temporal_fusion import AlphaTierConfig  # noqa: E402
from saccade.perception.temporal_yolo.roi_embedder import (  # noqa: E402
    ROIEmbeddingBank,
    FPNCropEmbedder,
    extract_roi_embeddings,
    EMB_DIM,
)
from saccade.perception.temporal_yolo.reid_head import load_reid_head  # noqa: E402

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
    dim_selector: torch.Tensor | None = None,
    reid_head=None,
    fpn_embedder=None,
    use_gmc_warp: bool = False,
    use_detector_heatmap: bool = False,
    profile: bool = False,
) -> int:
    seq_name = seq_dir.name
    frames = _load_frames(seq_dir, img_size)
    orig_h, orig_w = _orig_hw(seq_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker.tracker.reset() if hasattr(tracker.tracker, "reset") else None
    tracker.set_frame_size(orig_w, orig_h)
    if hasattr(model, "reset_fusion"):
        model.reset_fusion()

    sx, sy = orig_w / img_size, orig_h / img_size
    prev_gate_input: object | None = None
    prev_boxes_640: torch.Tensor | None = None
    prev_scores: torch.Tensor | None = None
    prev_det_boxes: torch.Tensor | None = None
    prev_det_scores: torch.Tensor | None = None

    # GMC for temporal fusion warp (Phase 2)
    gmc: SparseOpticalFlowGMC | None = None
    gmc_matrix: torch.Tensor | None = None
    if use_gmc_warp:
        from saccade.perception.eval.gmc import SparseOpticalFlowGMC

        gmc = SparseOpticalFlowGMC()

    # ROI ReID bank: reset per sequence
    emb_bank = ROIEmbeddingBank(img_size=img_size) if use_roi_reid else None

    mot_lines: list[str] = []
    t0 = time.perf_counter()

    # Profiling accumulators
    t_gate_input = 0.0
    t_model_fwd = 0.0
    t_postprocess = 0.0
    t_tracker = 0.0
    n_prof = 0

    for frame_id, frame in frames:
        frame = frame.to(device)

        # Update GMC for temporal fusion warp (before model forward)
        if gmc is not None:
            gmc_matrix = gmc.estimate(frame.squeeze(0))

        if hasattr(model, "set_gmc") and use_gmc_warp:
            model.set_gmc(gmc_matrix)

        # Gate input from previous frame's tracker state (Phase 3)
        t1 = time.perf_counter() if profile else 0
        gate_input = None
        if not disable_gate and prev_gate_input is not None:
            gate_input = prev_gate_input
        elif (
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
        if profile:
            t_gate_input += time.perf_counter() - t1

        if use_roi_reid:
            model.cache_feats = True
            model._feat_cache.clear()

        if hasattr(model, "set_prev_detections") and use_detector_heatmap:
            model.set_prev_detections(prev_det_boxes, prev_det_scores)

        t2 = time.perf_counter() if profile else 0
        with torch.no_grad():
            out = model(frame, gate_input=gate_input)
        if profile:
            torch.cuda.synchronize()
            t_model_fwd += time.perf_counter() - t2

        if use_roi_reid:
            model.cache_feats = False

        t3 = time.perf_counter() if profile else 0
        raw = out[0][0]  # (300, 6)
        keep = raw[:, 4] > conf_threshold
        dets = raw[keep]

        embeddings = None
        if dets.numel() > 0:
            prev_boxes_640 = dets[:, :4].clone()
            prev_scores = dets[:, 4].clone()
            if use_detector_heatmap:
                prev_det_boxes = dets[:, :4].clone()
                prev_det_scores = dets[:, 4].clone()

            boxes_orig = dets[:, :4].clone()
            boxes_orig[:, [0, 2]] *= sx
            boxes_orig[:, [1, 3]] *= sy
            scores = dets[:, 4]
            classes = dets[:, 5].to(torch.int32)

            if use_roi_reid and model._feat_cache:
                if reid_head is not None:
                    embeddings = extract_roi_embeddings(
                        model._feat_cache, dets[:, :4], reid_head=reid_head
                    )
                elif fpn_embedder is not None:
                    embeddings = fpn_embedder.extract(model._feat_cache, dets[:, :4])
                else:
                    embeddings = extract_roi_embeddings(model._feat_cache, dets[:, :4])
                    if dim_selector is not None:
                        embeddings = torch.nn.functional.normalize(
                            embeddings[:, dim_selector], dim=1
                        )
        else:
            prev_boxes_640 = None
            prev_scores = None
            prev_det_boxes = None
            prev_det_scores = None
            boxes_orig = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            classes = torch.zeros((0,), device=device, dtype=torch.int32)

        t4 = time.perf_counter() if profile else 0
        track_results = tracker.update(
            boxes_orig, scores, classes, embeddings=embeddings
        )

        if not disable_gate:
            try:
                snapshots = tracker.get_state_snapshots()
                candidates = tracker.get_tentative_candidates()
                prev_gate_input = TrackerGateInput.from_tracker_results(
                    track_results,
                    snapshots,
                    candidates,
                    (img_size, img_size),
                ).to(device)
            except Exception:
                prev_gate_input = None

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

        if profile:
            torch.cuda.synchronize()
            t_tracker += time.perf_counter() - t4
            t_postprocess += t4 - t3
            n_prof += 1

    elapsed = time.perf_counter() - t0
    fps = len(frames) / max(elapsed, 1e-6)
    (output_dir / f"{seq_name}.txt").write_text("\n".join(mot_lines))
    reid_str = f"  bank={len(emb_bank)}" if emb_bank else ""
    print(
        f"  {seq_name}: {len(frames)} frames, {fps:.1f} FPS, "
        f"{len(mot_lines)} tracks{reid_str}"
    )
    if profile and n_prof > 0:
        total = t_gate_input + t_model_fwd + t_postprocess + t_tracker
        n = n_prof
        print(
            f"    gate_input: {t_gate_input * 1000 / n:5.1f} ms  "
            f"model_fwd: {t_model_fwd * 1000 / n:5.1f} ms  "
            f"post: {t_postprocess * 1000 / n:5.1f} ms  "
            f"tracker: {t_tracker * 1000 / n:5.1f} ms  "
            f"total: {total * 1000 / n:.1f} ms"
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
    parser.add_argument(
        "--score-on-gt-frames",
        action="store_true",
        help="Keyframe-aware scoring: run full cadence but score only on frames "
        "carrying GT (PP22 mot_test_kf). Avoids counting non-keyframe predictions "
        "as FP.",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument(
        "--temporal-fusion",
        action="store_true",
        help="Enable Option E-v2 temporal feature fusion",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.0,
        help="Fixed fusion alpha for temporal fusion (0=sanity check, 0.1=Phase 1)",
    )
    parser.add_argument(
        "--fusion-warp",
        action="store_true",
        help="Enable GMC warp in temporal feature fusion (Phase 2)",
    )
    parser.add_argument(
        "--detector-heatmap",
        action="store_true",
        help="Add prev-frame detector score heatmap to Q_spatial (Phase 3)",
    )
    parser.add_argument(
        "--alpha-tier-occluded",
        type=float,
        default=0.20,
        help="α_tier for occluded tracks (det_idx == -1)",
    )
    parser.add_argument(
        "--alpha-tier-recent",
        type=float,
        default=0.15,
        help="α_tier for confirmed tracks age 1-10",
    )
    parser.add_argument(
        "--alpha-tier-stable",
        type=float,
        default=0.05,
        help="α_tier for confirmed tracks age >10",
    )
    parser.add_argument(
        "--alpha-tier-tentative",
        type=float,
        default=0.05,
        help="α_tier for tentative/unconfirmed tracks",
    )
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
    parser.add_argument(
        "--reid-head",
        default="",
        help="Path to trained ReIDHead checkpoint (runs/reid_head_v1/best.ckpt). "
        "Projects 896-dim raw FPN → 128-dim discriminative embedding.",
    )
    parser.add_argument(
        "--fpn-scale",
        default="p4",
        choices=["p3", "p4", "p5"],
        help="FPN scale for FPNCropEmbedder (used when --roi-reid without --reid-head). "
        "p3=128-dim, p4=256-dim, p5=512-dim.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-stage timing breakdown",
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
        enable_temporal_fusion=args.temporal_fusion,
        fusion_alpha=args.fusion_alpha,
        fusion_fixed_alpha=True,
        enable_detector_heatmap=args.detector_heatmap,
        alpha_tier=AlphaTierConfig(
            occluded=args.alpha_tier_occluded,
            confirmed_recent=args.alpha_tier_recent,
            confirmed_stable=args.alpha_tier_stable,
            tentative=args.alpha_tier_tentative,
        )
        if args.temporal_fusion
        else None,
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
    fusion_str = ""
    if args.temporal_fusion:
        warp_str = " +warp" if args.fusion_warp else ""
        deth_str = " +det" if args.detector_heatmap else ""
        tier_str = (
            f" tier=[{args.alpha_tier_occluded:.2f}/{args.alpha_tier_recent:.2f}"
            f"/{args.alpha_tier_stable:.2f}/{args.alpha_tier_tentative:.2f}]"
        )
        fusion_str = (
            f"  temporal_fusion α={args.fusion_alpha}{warp_str}{deth_str}{tier_str}"
        )
    print(
        f"Gate {'DISABLED' if args.no_gate else 'ENABLED'}  "
        f"ROI-ReID {'ENABLED' if args.roi_reid else 'DISABLED'}{fusion_str}"
    )
    # Load dim selector from Fisher analysis if provided
    dim_selector: torch.Tensor | None = None
    if args.dim_select and Path(args.dim_select).exists():
        import numpy as np

        dim_selector = torch.from_numpy(np.load(args.dim_select)).long().to(device)
        print(
            f"  dim_selector: top-{len(dim_selector)} Fisher dims from {args.dim_select}"
        )

    # Load trained ReID projection head if provided
    reid_head = None
    if args.reid_head and Path(args.reid_head).exists():
        reid_head = load_reid_head(args.reid_head, device=device)
        print(f"  ReIDHead loaded: {args.reid_head}  out_dim={reid_head.out_dim}")

    # Build FPN crop embedder (used when --roi-reid and no --reid-head)
    fpn_embedder: FPNCropEmbedder | None = None
    if args.roi_reid and reid_head is None and not args.dim_select:
        fpn_embedder = FPNCropEmbedder(scale=args.fpn_scale)
        print(
            f"  FPNCropEmbedder: scale={args.fpn_scale}  emb_dim={fpn_embedder.emb_dim}"
        )

    if args.roi_reid:
        if reid_head is not None:
            active_dim = reid_head.out_dim
        elif fpn_embedder is not None:
            active_dim = fpn_embedder.emb_dim
        elif dim_selector is not None:
            active_dim = len(dim_selector)
        else:
            active_dim = EMB_DIM
        print(
            f"  embedding_dim={active_dim}  cos_thr={args.reid_cos_thr}  reid_weight={args.reid_weight}"
        )

    if args.roi_reid:
        if reid_head is not None:
            emb_dim = reid_head.out_dim
        elif fpn_embedder is not None:
            emb_dim = fpn_embedder.emb_dim
        elif dim_selector is not None:
            emb_dim = len(dim_selector)
        else:
            emb_dim = EMB_DIM
    else:
        emb_dim = 768
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
            reid_head=reid_head,
            fpn_embedder=fpn_embedder,
            use_gmc_warp=args.fusion_warp,
            use_detector_heatmap=args.detector_heatmap,
            profile=args.profile,
        )

    try:
        from saccade.perception.eval.metrics import run_motmetrics_evaluation

        metrics = run_motmetrics_evaluation(
            data_root=str(data_root),
            split=args.split,
            output=str(output_dir),
            sequences=",".join(seqs),
            detector=None if args.sequences else "SDP",
            score_on_gt_frames=args.score_on_gt_frames,
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")


if __name__ == "__main__":
    main()
