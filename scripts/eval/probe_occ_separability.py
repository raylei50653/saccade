#!/usr/bin/env python3
"""Feasibility probe: does the Mamba head's SCORE separate occluded from visible GT?

Question
--------
The occlusion gate lives entirely in the tracker (track-vs-track foot-gap geometry);
the detection head has no visibility/occlusion output channel. Before spending any
training budget on adding a per-box visibility head, test whether the head's EXISTING
output (the detection score) already carries a signal that separates "occluded" from
"visible" GT boxes. If it does not, a score-derived visibility head is a non-starter.

Two channels are measured, because occlusion can manifest two different ways:

  1. RECALL-by-visibility (the FN channel): is an occluded GT box detected at all?
     A prior finding claims occlusion shows up as *missed detections*, not as
     low-score detections. Recall dropping with visibility confirms that — but a
     missed box has no box to attach a per-box signal to, so this channel is NOT
     usable by a per-box visibility head. It is reported for context only.

  2. SCORE-AUC among MATCHED GT (the decision channel): among GT the head actually
     detected, can the score rank occluded below visible? This is the only thing a
     per-box visibility head could exploit. AUC ~0.5 => no separable signal => NO-GO
     for a score-based visibility head. AUC >> 0.5 => residual signal worth a head.

Detector-only (no tracker), greedy GT->detection match at IoU>=match_iou. Reuses the
loaders / detector build from analyze_score_distribution.py.

Usage
-----
  .venv/bin/python scripts/eval/probe_occ_separability.py \
      --output results/occ_separability/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: E402, F401
import torchvision  # noqa: E402

from saccade.perception.eval.detection import detect_single_patch_640  # noqa: E402
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
    set_postprocess_compile,
)

# Reuse the exact loaders + matcher the score-distribution study uses.
from analyze_score_distribution import (  # noqa: E402
    _load_frame,
    _load_ground_truth,
    _match_scores,
)

# Visibility bins (MOT17 gt visibility column = fraction of box not occluded).
VIS_BINS = (
    ("vis_lt30", 0.0, 0.30),
    ("vis_30to50", 0.30, 0.50),
    ("vis_50to70", 0.50, 0.70),
    ("vis_70to90", 0.70, 0.90),
    ("vis_ge90", 0.90, 1.01),
)


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score(pos) > score(neg)) via Mann-Whitney U, ties counted as 0.5.

    Here 'pos' = visible GT, 'neg' = occluded GT, so AUC>0.5 means visible boxes
    score higher than occluded ones (the head would 'see' occlusion in the score).
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0  # 1-based average rank per unique value
    ranks = avg[inv]
    sum_pos = ranks[:n_pos].sum()
    u_pos = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mamba-ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,"
        "MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    p.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    p.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    p.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf-floor", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--occ-thresh",
        type=float,
        default=0.5,
        help="visibility < this = occluded (the binary split for AUC)",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    set_postprocess_compile(False)
    detector = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=640,
        device=args.device,
        conf_thr=args.conf_floor,
        max_det=args.max_det,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    detector.eval()
    detector.mamba_head.set_head_compile(False)

    # Per-GT records: (visibility, matched_score [NaN if missed], height)
    vis_all: list[float] = []
    score_all: list[float] = []  # NaN == missed
    height_all: list[float] = []

    for sequence in (s.strip() for s in args.sequences.split(",") if s.strip()):
        seq_root = Path(args.data_root) / args.split / sequence
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        gts = _load_ground_truth(seq_root / "gt" / "gt.txt", max_frames=args.max_frames)
        first = _load_frame(frame_paths[0], args.device)
        orig_h, orig_w = first.shape[-2:]
        pool = AdaptiveFramePool(orig_h, orig_w, device=args.device)
        detector.reset_tracker()
        with torch.inference_mode():
            for idx, fp in enumerate(frame_paths, start=1):
                frame_id = int(fp.stem)
                frame = first if idx == 1 else _load_frame(fp, args.device)
                pool.frame_buffer.copy_(frame)
                frame_gt = gts.get(frame_id, [])
                if not frame_gt:
                    continue
                boxes, scores, classes = detect_single_patch_640(
                    detector,
                    pool,
                    orig_h,
                    orig_w,
                    preprocess_modes=[],
                    detector_box_format="xyxy",
                )
                person = (classes.to(torch.int32) == 0) & (scores >= args.conf_floor)
                boxes = boxes[person]
                scores = scores[person]
                if boxes.numel() > 0:
                    keep = torchvision.ops.nms(boxes, scores, args.nms_iou)
                    boxes = boxes[keep]
                    scores = scores[keep]
                gt_boxes = torch.tensor(
                    [it["bbox"] for it in frame_gt], dtype=torch.float32
                ).reshape(-1, 4)
                matched = _match_scores(
                    gt_boxes, boxes, scores, iou_threshold=args.match_iou
                )
                for i, it in enumerate(frame_gt):
                    vis_all.append(float(it["visibility"]))
                    score_all.append(float(matched[i]))
                    height_all.append(float(it["height"]))
                if idx % 200 == 0:
                    print(f"{sequence} [{idx}/{len(frame_paths)}] gt={len(vis_all)}")

    vis = np.array(vis_all)
    score = np.array(score_all)
    detected = ~np.isnan(score)

    report: dict = {"n_gt": int(len(vis)), "occ_thresh": args.occ_thresh}

    # --- Channel 1: recall by visibility (the FN channel) ---
    print(f"\n=== RECALL by visibility (n_gt={len(vis)}) ===")
    print(f"{'vis bin':12s}{'n_gt':>10s}{'recall':>10s}{'mean_score(det)':>18s}")
    recall_rows = {}
    for name, lo, hi in VIS_BINS:
        m = (vis >= lo) & (vis < hi)
        n = int(m.sum())
        rec = float(detected[m].mean()) if n else float("nan")
        ms = float(np.nanmean(score[m])) if n and detected[m].any() else float("nan")
        recall_rows[name] = {"n": n, "recall": rec, "mean_score_detected": ms}
        print(f"{name:12s}{n:>10d}{rec:>10.3f}{ms:>18.3f}")
    report["recall_by_visibility"] = recall_rows

    # --- Channel 2: score-AUC among MATCHED GT (the decision channel) ---
    occ = vis < args.occ_thresh
    pos = score[detected & ~occ]  # visible & detected
    neg = score[detected & occ]  # occluded & detected
    auc_score = rank_auc(pos, neg)
    print(
        "\n=== SCORE separability among DETECTED GT "
        f"(occluded vis<{args.occ_thresh}) ==="
    )
    print(
        f"  visible&det   n={len(pos):6d}  mean_score={np.mean(pos) if len(pos) else float('nan'):.3f}"
    )
    print(
        f"  occluded&det  n={len(neg):6d}  mean_score={np.mean(neg) if len(neg) else float('nan'):.3f}"
    )
    print(
        f"  AUC(score: visible>occluded) = {auc_score:.3f}   "
        f"(0.50=no signal, >0.65=usable)"
    )
    report["score_auc_detected"] = {
        "auc": auc_score,
        "n_visible_det": int(len(pos)),
        "n_occluded_det": int(len(neg)),
        "mean_score_visible": float(np.mean(pos)) if len(pos) else float("nan"),
        "mean_score_occluded": float(np.mean(neg)) if len(neg) else float("nan"),
    }

    # Robustness: a harder split (vis<0.3 occluded vs vis>=0.7 visible).
    occ_h = vis < 0.30
    vis_h = vis >= 0.70
    auc_hard = rank_auc(score[detected & vis_h], score[detected & occ_h])
    report["score_auc_detected_hard_split"] = {
        "auc": auc_hard,
        "n_visible_det": int((detected & vis_h).sum()),
        "n_occluded_det": int((detected & occ_h).sum()),
    }
    print(f"  AUC hard split (vis<0.30 vs vis>=0.70) = {auc_hard:.3f}")

    # --- Context: detect-vs-miss as an AUC (occlusion DOES live in the FN channel,
    #     but a missed box carries no attachable per-box signal). ---
    auc_miss = rank_auc(detected[~occ].astype(float), detected[occ].astype(float))
    report["detect_vs_miss_auc"] = {
        "auc": auc_miss,
        "note": "recall framed as AUC; not usable by a per-box head (missed box has no box)",
    }
    print(
        f"\n=== CONTEXT: detect/miss AUC (visible>occluded) = {auc_miss:.3f} "
        "(this is the FN channel, NOT per-box-usable) ==="
    )

    # --- Verdict ---
    usable = (not np.isnan(auc_score)) and auc_score >= 0.65
    report["verdict"] = (
        "GO: score carries usable occlusion signal"
        if usable
        else "NO-GO: score does not separate occluded from visible per-box"
    )
    print(f"\nVERDICT: {report['verdict']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
