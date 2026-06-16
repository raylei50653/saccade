#!/usr/bin/env python3
"""Why are detected GT still missed? Reason distribution for association-failure FN.

A GT-frame is an "assoc-failure FN" if the detector saw it (best matching
detection score >= birth_thresh) yet it was NOT tracked in the baseline output.
We place each such FN on its GT *identity* timeline (GT col 1) to attribute the
miss to a cause:

  confirm_latency : within the first `confirm_streak` present frames AND the id
                    has no tracked frame yet -> birth/confirmation delay.
  never_tracked   : the id is tracked in zero frames -> tracker never established
                    it (perennially hard: small/occluded/edge).
  mid_break       : the id was tracked < gap_short frames ago -> short
                    association break (occlusion / fast motion / GMC drift).
  lost_reappear   : the id was tracked but the last tracked frame is >= gap_short
                    frames ago -> lost identity not relinked.

Also reports the FN distribution over visibility / max-overlap / height / gap so
the headroom can be read directly.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
from analyze_score_distribution import _load_frame  # noqa: E402


def _load_gt_with_ids(path: Path) -> dict[int, list[dict]]:
    by: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        if int(r[6]) != 1 or int(r[7]) != 1 or (len(r) > 8 and r[8] < 0.1):
            continue
        x, y, w, h = r[2], r[3], r[4], r[5]
        by[int(r[0])].append(
            {
                "id": int(r[1]),
                "bbox": [x, y, x + w, y + h],
                "h": h,
                "vis": float(r[8]) if len(r) > 8 else 1.0,
            }
        )
    return by


def _load_out(path: Path) -> dict[int, list[list[float]]]:
    by: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        by[int(r[0])].append([r[2], r[3], r[2] + r[4], r[3] + r[5]])
    return by


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
    p.add_argument("--birth-thresh", type=float, default=0.28)
    p.add_argument("--confirm-streak", type=int, default=3)
    p.add_argument("--gap-short", type=int, default=5)
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

    # records: per GT-frame -> (seq, gtid, frame, tracked, det_score, vis, overlap, h)
    recs: list[dict] = []
    for seq in (s.strip() for s in args.sequences.split(",") if s.strip()):
        seq_root = Path(args.data_root) / args.split / seq
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        gts = _load_gt_with_ids(seq_root / "gt" / "gt.txt")
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
                fg = gts.get(fid, [])
                if not fg:
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
                    [g["bbox"] for g in fg], dtype=torch.float32
                ).reshape(-1, 4)
                best_det = torch.full((len(fg),), -1.0)
                if boxes.numel():
                    di = torchvision.ops.box_iou(boxes.cpu(), gtb)
                    hit = di >= args.match_iou
                    sc = scores.cpu().unsqueeze(1).expand_as(di)
                    best_det = (
                        torch.where(hit, sc, torch.full_like(sc, -1.0))
                        .max(dim=0)
                        .values
                    )
                outb = torch.tensor(out.get(fid, []), dtype=torch.float32).reshape(
                    -1, 4
                )
                tracked = torch.zeros(len(fg), dtype=torch.bool)
                if outb.numel():
                    tracked = (
                        torchvision.ops.box_iou(gtb, outb).max(dim=1).values
                        >= args.match_iou
                    )
                # GT-GT overlap (occlusion proxy)
                ov = torch.zeros(len(fg))
                if len(fg) > 1:
                    gg = torchvision.ops.box_iou(gtb, gtb)
                    gg.fill_diagonal_(0)
                    ov = gg.max(dim=1).values
                for g in range(len(fg)):
                    recs.append(
                        dict(
                            seq=seq,
                            gtid=fg[g]["id"],
                            frame=fid,
                            tracked=bool(tracked[g]),
                            det=float(best_det[g]),
                            vis=fg[g]["vis"],
                            ov=float(ov[g]),
                            h=fg[g]["h"],
                        )
                    )
                if i % 200 == 0:
                    print(f"{seq} [{i}/{len(frame_paths)}]")

    # Build per-identity timelines.
    by_id: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        by_id[(r["seq"], r["gtid"])].append(r)
    for k in by_id:
        by_id[k].sort(key=lambda r: r["frame"])

    cat = defaultdict(int)
    gap_hist = defaultdict(int)
    cov = defaultdict(lambda: defaultdict(int))  # category -> covariate bucket -> count
    total_fn_det = 0
    for k, tl in by_id.items():
        tracked_frames = [r["frame"] for r in tl if r["tracked"]]
        tl[0]["frame"]
        ever_tracked = len(tracked_frames) > 0
        for j, r in enumerate(tl):
            if r["tracked"]:
                continue
            if r["det"] < args.birth_thresh:
                cat["nodet(<thresh)"] += 1
                continue
            total_fn_det += 1
            prev_tracked = [f for f in tracked_frames if f < r["frame"]]
            present_idx = j  # index in identity timeline (#present frames so far)
            if not ever_tracked:
                c = "never_tracked"
            elif not prev_tracked:
                if present_idx < args.confirm_streak:
                    c = "confirm_latency"
                else:
                    c = "pre_first_track"  # present a while before first ever tracked
            else:
                gap = r["frame"] - prev_tracked[-1]
                gap_hist[min(gap, 60)] += 1
                c = "mid_break" if gap <= args.gap_short else "lost_reappear"
            cat[c] += 1
            vb = (
                "vis<.3"
                if r["vis"] < 0.3
                else "vis.3-.7"
                if r["vis"] < 0.7
                else "vis>=.7"
            )
            ob = "ov<.2" if r["ov"] < 0.2 else "ov.2-.4" if r["ov"] < 0.4 else "ov>=.4"
            hb = "h<64" if r["h"] < 64 else "h64-128" if r["h"] < 128 else "h>=128"
            cov[c]["vis:" + vb] += 1
            cov[c]["ov:" + ob] += 1
            cov[c]["h:" + hb] += 1

    print(f"\n=== assoc-failure FN reason distribution (det>={args.birth_thresh}) ===")
    print(f"total detected-but-untracked GT-frames: {total_fn_det}")
    order = [
        "confirm_latency",
        "never_tracked",
        "pre_first_track",
        "mid_break",
        "lost_reappear",
    ]
    for c in order:
        n = cat.get(c, 0)
        if not n:
            continue
        pct = 100 * n / total_fn_det
        print(f"  {c:16s} {n:6d}  {pct:5.1f}%")
    print(f"  (nodet<thresh, excluded: {cat['nodet(<thresh)']})")

    print(
        "\n=== gap-length distribution for re-association FN (mid_break + lost_reappear) ==="
    )
    tot_gap = sum(gap_hist.values())
    for lo, hi, lbl in [
        (1, 1, "gap=1"),
        (2, 2, "gap=2"),
        (3, 5, "gap3-5"),
        (6, 10, "gap6-10"),
        (11, 30, "gap11-30"),
        (31, 60, "gap31-60+"),
    ]:
        n = sum(v for g, v in gap_hist.items() if lo <= g <= hi)
        print(f"  {lbl:10s} {n:6d}  {100 * n / tot_gap if tot_gap else 0:5.1f}%")

    print("\n=== covariates per reason (share within category) ===")
    for c in order:
        if not cat.get(c):
            continue
        tot = cat[c]
        vis = " ".join(
            f"{k.split(':')[1]}={100 * cov[c][k] // tot}%"
            for k in sorted(cov[c])
            if k.startswith("vis")
        )
        ov = " ".join(
            f"{k.split(':')[1]}={100 * cov[c][k] // tot}%"
            for k in sorted(cov[c])
            if k.startswith("ov")
        )
        hh = " ".join(
            f"{k.split(':')[1]}={100 * cov[c][k] // tot}%"
            for k in sorted(cov[c])
            if k.startswith("h:")
        )
        print(f"  {c:16s} | {vis} | {ov} | {hh}")


if __name__ == "__main__":
    main()
