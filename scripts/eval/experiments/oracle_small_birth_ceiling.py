#!/usr/bin/env python3
"""Recall-side oracle for LOWERING new_track_thresh on small boxes.

For every GT we decide: was it TRACKED in the baseline output (IoU>=0.5 to an
output box)? If NOT (a false negative), what is the best detector score that
GT received in the raw conf-floor stream? A missed GT is recoverable by a
LOWER birth threshold only if (a) the detector actually saw it at some score
below the current 0.28 birth gate, i.e. best_det_score in [thr_low, 0.28).

We tabulate FN by height x best-det-score band. The small-height cells in the
[0.15,0.28) column upper-bound the recall a small-box threshold drop could buy
(over-counts: some FN are association/confirm failures a threshold won't fix,
and the detection may belong to an already-tracked neighbour).
"""
# status: experiment

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "eval"))
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
from analyze_score_distribution import _load_frame, _load_ground_truth  # noqa: E402

HEIGHT_BINS = (
    ("h_lt32", 0, 32),
    ("h_32to64", 32, 64),
    ("h_64to128", 64, 128),
    ("h_128to256", 128, 256),
    ("h_ge256", 256, 10**9),
)
# best-detector-score band for the MISSED gt
DET_BANDS = (
    ("none", -1.0, 0.0),
    ("d_lt15", 0.0, 0.15),
    ("d_15to28", 0.15, 0.28),
    ("d_28to50", 0.28, 0.50),
    ("d_ge50", 0.50, 1.01),
)


def _load_out(path: Path) -> dict[int, list[list[float]]]:
    by: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        by[int(r[0])].append([r[2], r[3], r[2] + r[4], r[3] + r[5]])
    return by


def _bin(v, bins):
    for n, lo, hi in bins:
        if lo <= v < hi:
            return n
    return bins[-1][0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mamba-ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt")
    p.add_argument("--output-dir", default="results/strat_baseline")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    p.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    p.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    p.add_argument(
        "--trt-backbone-engine", default="models/yolo/yolo26s_backbone_640_best.engine"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf-floor", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--match-iou", type=float, default=0.5)
    args = p.parse_args()

    set_postprocess_compile(False)
    det = build_mamba_gated_detector(
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
    det.eval()
    det.mamba_head.set_head_compile(False)

    # counts[height][det_band] = number of FN GT; plus total GT per height.
    fn = defaultdict(lambda: defaultdict(int))
    gt_total = defaultdict(int)
    tracked = defaultdict(int)
    for seq in (s.strip() for s in args.sequences.split(",") if s.strip()):
        seq_root = Path(args.data_root) / args.split / seq
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        gts = _load_ground_truth(seq_root / "gt" / "gt.txt", max_frames=0)
        out = _load_out(Path(args.output_dir) / f"{seq}.txt")
        first = _load_frame(frame_paths[0], args.device)
        oh, ow = first.shape[-2:]
        pool = AdaptiveFramePool(oh, ow, device=args.device)
        det.reset_tracker()
        with torch.inference_mode():
            for i, fp in enumerate(frame_paths, start=1):
                fid = int(fp.stem)
                frame = first if i == 1 else _load_frame(fp, args.device)
                pool.frame_buffer.copy_(frame)
                frame_gt = gts.get(fid, [])
                if not frame_gt:
                    continue
                boxes, scores, classes = detect_single_patch_640(
                    det, pool, oh, ow, preprocess_modes=[], detector_box_format="xyxy"
                )
                person = (classes.to(torch.int32) == 0) & (scores >= args.conf_floor)
                boxes, scores = boxes[person], scores[person]
                if boxes.numel():
                    keep = torchvision.ops.nms(boxes, scores, args.nms_iou)
                    boxes, scores = boxes[keep], scores[keep]
                gtb = torch.tensor(
                    [it["bbox"] for it in frame_gt], dtype=torch.float32
                ).reshape(-1, 4)
                # best detector score per GT (any IoU>=match)
                best_det = torch.full((gtb.shape[0],), -1.0)
                if boxes.numel():
                    di = torchvision.ops.box_iou(boxes.cpu(), gtb)  # [det, gt]
                    hit = di >= args.match_iou
                    sc = scores.cpu().unsqueeze(1).expand_as(di)
                    masked = torch.where(hit, sc, torch.full_like(sc, -1.0))
                    best_det = masked.max(dim=0).values
                # tracked? match GT to output boxes
                outb = torch.tensor(out.get(fid, []), dtype=torch.float32).reshape(
                    -1, 4
                )
                is_tracked = torch.zeros(gtb.shape[0], dtype=torch.bool)
                if outb.numel():
                    oi = torchvision.ops.box_iou(gtb, outb)
                    is_tracked = oi.max(dim=1).values >= args.match_iou
                for g in range(gtb.shape[0]):
                    h = float(frame_gt[g]["height"])
                    hb = _bin(h, HEIGHT_BINS)
                    gt_total[hb] += 1
                    if bool(is_tracked[g]):
                        tracked[hb] += 1
                    else:
                        fn[hb][_bin(float(best_det[g]), DET_BANDS)] += 1
                if i % 200 == 0:
                    print(f"{seq} [{i}/{len(frame_paths)}]")

    print("\n=== FN GT by height x best-detector-score band ===")
    print(
        f"{'height':12s} {'GT':>7s} {'trk%':>5s} {'FN':>6s} | "
        + " ".join(f"{b[0]:>9s}" for b in DET_BANDS)
    )
    for hn, _, _ in HEIGHT_BINS:
        g = gt_total[hn]
        if not g:
            continue
        fnrow = fn[hn]
        total_fn = sum(fnrow.values())
        cells = " ".join(f"{fnrow.get(b[0], 0):9d}" for b in DET_BANDS)
        print(f"{hn:12s} {g:7d} {100 * tracked[hn] / g:4.1f} {total_fn:6d} | {cells}")

    print(
        "\n=== recall headroom (FN that had a detection in [0.15,0.28) — upper bound) ==="
    )
    small = ("h_lt32", "h_32to64", "h_64to128")
    rec_small = sum(fn[h].get("d_15to28", 0) for h in small)
    rec_all = sum(fn[h].get("d_15to28", 0) for h, _, _ in HEIGHT_BINS)
    tot_gt = sum(gt_total.values())
    print(
        f"  small (h<128): {rec_small} GT-frames  ({100 * rec_small / tot_gt:.3f}% of all GT)"
    )
    print(
        f"  all heights:   {rec_all} GT-frames  ({100 * rec_all / tot_gt:.3f}% of all GT)"
    )
    # also the [0.28,0.50) FN = birth-eligible but still missed (assoc/confirm, not threshold)
    assoc_fn = sum(
        fn[h].get("d_28to50", 0) + fn[h].get("d_ge50", 0) for h, _, _ in HEIGHT_BINS
    )
    nodet_fn = sum(fn[h].get("none", 0) for h, _, _ in HEIGHT_BINS)
    print(f"  FN already birth-eligible (det>=0.28, assoc/confirm loss): {assoc_fn}")
    print(f"  FN never detected (det<floor, threshold cannot help):      {nodet_fn}")


if __name__ == "__main__":
    main()
