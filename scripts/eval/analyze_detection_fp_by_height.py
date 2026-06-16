#!/usr/bin/env python3
"""Detection-centric TP/FP split by box height x score band.

Oracle-ceiling estimate for a HEIGHT-CONDITIONED birth threshold. For each
detection we record (score, height, is_TP) where TP = matched a GT at IoU>=0.5.
We then tabulate, inside the birth-relevant score band, how TP/FP splits by
height -- the precondition for "ramp new_track_thresh up with box height":
large low-score detections should be mostly FP, small ones mostly TP.

This is detector-only (no tracker). It approximates births by detections; it
does NOT model association, so treat the numbers as an upper-bound sanity check
on whether a height-conditioned birth gate can help BEFORE touching C++.
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

# Re-use the same loaders as analyze_score_distribution.
from analyze_score_distribution import _load_frame, _load_ground_truth  # noqa: E402

HEIGHT_BINS = (
    ("h_lt64", 0.0, 64.0),
    ("h_64to128", 64.0, 128.0),
    ("h_128to256", 128.0, 256.0),
    ("h_ge256", 256.0, float("inf")),
)
SCORE_BANDS = (
    ("s_lt15", 0.0, 0.15),
    ("s_15to20", 0.15, 0.20),
    ("s_20to28", 0.20, 0.28),
    ("s_28to35", 0.28, 0.35),
    ("s_35to50", 0.35, 0.50),
    ("s_ge50", 0.50, 1.01),
)


def _bin(value, bins):
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return bins[-1][0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mamba-ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
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
    p.add_argument("--max-frames", type=int, default=0)
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
        max_det=300,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    detector.eval()
    detector.mamba_head.set_head_compile(False)

    # Per-detection records: (score, height, is_tp)
    recs: list[tuple[float, float, int]] = []
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
                if boxes.numel() == 0:
                    continue
                frame_gt = gts.get(frame_id, [])
                gt_boxes = torch.tensor(
                    [it["bbox"] for it in frame_gt], dtype=torch.float32
                ).reshape(-1, 4)
                # Match each DETECTION to a GT (greedy by score).
                b_cpu = boxes.cpu()
                s_cpu = scores.cpu()
                heights = (b_cpu[:, 3] - b_cpu[:, 1]).clamp(min=0)
                is_tp = torch.zeros(b_cpu.shape[0], dtype=torch.int32)
                if gt_boxes.numel() > 0:
                    ious = torchvision.ops.box_iou(b_cpu, gt_boxes)
                    used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
                    for di in s_cpu.argsort(descending=True).tolist():
                        avail = ious[di].masked_fill(used, -1.0)
                        biou, gi = avail.max(dim=0)
                        if float(biou) >= args.match_iou:
                            used[gi] = True
                            is_tp[di] = 1
                for di in range(b_cpu.shape[0]):
                    recs.append((float(s_cpu[di]), float(heights[di]), int(is_tp[di])))
                if idx % 200 == 0:
                    print(f"{sequence} [{idx}/{len(frame_paths)}] dets={len(recs)}")

    arr = np.array(recs, dtype=np.float64)  # cols: score, height, is_tp
    score, height, tp = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)

    # Cross-tab: TP / FP counts and precision by (height bin x score band).
    print(
        "\n=== detection precision by height x score band (det count / precision) ==="
    )
    hdr = f"{'height':12s}" + "".join(f"{b[0]:>11s}" for b in SCORE_BANDS)
    print(hdr)
    table = {}
    for hname, hlo, hhi in HEIGHT_BINS:
        hm = (height >= hlo) & (height < hhi)
        row = f"{hname:12s}"
        table[hname] = {}
        for sname, slo, shi in SCORE_BANDS:
            sm = hm & (score >= slo) & (score < shi)
            n = int(sm.sum())
            prec = float(tp[sm].mean()) if n else float("nan")
            table[hname][sname] = {"n": n, "precision": prec}
            cell = f"{n}/{prec:.2f}" if n else "-"
            row += f"{cell:>11s}"
        print(row)

    # Oracle policy estimate: ramp threshold with height. For each height bin
    # pick a per-bin threshold; report TP kept vs FP removed relative to a flat
    # threshold. We sweep flat-vs-ramped on the band [0.15, 0.50).
    print("\n=== birth-band [0.15,0.50) TP/FP by height (what a ramp would gate) ===")
    band = (score >= 0.15) & (score < 0.50)
    for hname, hlo, hhi in HEIGHT_BINS:
        hm = band & (height >= hlo) & (height < hhi)
        n = int(hm.sum())
        ntp = int(tp[hm].sum())
        print(
            f"  {hname:12s} dets={n:6d} TP={ntp:6d} FP={n - ntp:6d} "
            f"precision={ntp / n if n else float('nan'):.3f}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "n_dets": len(recs),
                "table": table,
                "height_bins": [b[0] for b in HEIGHT_BINS],
                "score_bands": [b[0] for b in SCORE_BANDS],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
