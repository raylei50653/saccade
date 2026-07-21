#!/usr/bin/env python3
"""Does the geometric pairwise-position signal (gap_h/dx_norm, from the #46
follow-up in probe_occ_pairwise_confound.py) help disambiguate WHICH track a
private_continuation-rescued NMS candidate should be assigned to, when a
crossing/occlusion makes plain IoU-to-track cost ambiguous between two
nearby tracks?

Background
----------
private_continuation (mamba_whole_graph.yaml: private_candidate_nms_iou=0.70,
private_prior_iou_threshold=0.30) re-admits boxes a standard NMS (iou=0.5)
would suppress as duplicates, gated on matching a live track's predicted
position. Once admitted, the candidate goes through the SAME auction cost
(1-IoU vs each track's Kalman-predicted box) as any normal detection -- there
is no dedicated disambiguation logic for the case where the candidate sits
between two nearby tracks (A/B) with near-tied IoU cost. This is a genuinely
untested application, distinct from registry #46's 4 applications (relink /
OAO-source / swap-consistency / low-IoU-gate) which all concerned deciding
identity for a track already known to be occluded, not choosing between two
competing assignment targets for a recovered candidate box.

Method (offline oracle-track proxy, no C++/production changes)
----------------------------------------------------------------
For each frame with a GT crossing pair (A, B present in both this frame and
the previous frame, current boxes IoU > 0.1):
  1. Run the raw (pre-NMS) detector on the frame.
  2. Standard NMS (iou=0.5) -> "main" survivors. Wide NMS (iou=0.70) -> the
     private-continuation pool. rescued = wide_pool - main_survivors.
  3. For each rescued box, use the PREVIOUS frame's GT boxes for A and B as
     the track-predicted-position proxy (oracle track state -- best case).
  4. Ground truth label = whichever of A/B's CURRENT frame GT box the
     rescued box best overlaps (must be >= 0.3 IoU, else drop as not a real
     recovered detection of either).
  5. Restrict to the AMBIGUOUS subset: |cost_iou_A - cost_iou_B| < margin
     (plain-IoU cost is a near-tie -- exactly where a tie-break would help).
  6. Compare accuracy of two deterministic decision rules against GT:
       baseline   = argmin(cost_iou)
       augmented  = argmin(cost_iou + w*|gap_h| + w*|dx_norm|)   for a small
                    weight sweep, where gap_h/dx_norm are computed between
                    the rescued box and EACH hypothesis (A's or B's prior
                    box), matching the tracker_gpu.cu definition exactly.

Usage
-----
  .venv/bin/python scripts/eval/probe_private_continuation_assignment.py \
      --output results/occ_separability/private_continuation_assignment.json
"""
# status: experiment

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
import torchvision  # noqa: E402, F401  # side-effect: order before torchvision.ops
from torchvision.ops import nms  # noqa: E402

from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    resize_stretch_batch_gpu,
)
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

IMG_SIZE = 640


def load_gt(gt_path: Path) -> dict[int, dict[int, tuple]]:
    """{frame: {gid: (x1, y1, x2, y2)}} in native pixel coords (640-scaled later)."""
    out: dict[int, dict[int, tuple]] = {}
    for line in gt_path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 9:
            continue
        fid, gid = int(p[0]), int(p[1])
        mark, cls_id, vis = int(p[6]), int(p[7]), float(p[8])
        if mark != 1 or cls_id != 1 or vis < 0.1:
            continue
        x, y, w, h = (float(v) for v in p[2:6])
        out.setdefault(fid, {})[gid] = (x, y, x + w, y + h)
    return out


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def gap_dx(box, ref) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = ref
    h_b, h_r = y2 - y1, ry2 - ry1
    w_b, w_r = x2 - x1, rx2 - rx1
    h_ref = max(0.5 * (h_b + h_r), 1e-3)
    w_ref = max(0.5 * (w_b + w_r), 1e-3)
    footy_b, footy_r = y2, ry2
    cx_b, cx_r = 0.5 * (x1 + x2), 0.5 * (rx1 + rx2)
    gap_h = (footy_b - footy_r) / h_ref
    dx_norm = (cx_b - cx_r) / w_ref
    return gap_h, dx_norm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mamba-ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt")
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,"
        "MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    ap.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    ap.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    ap.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-floor", type=float, default=0.10)  # private_min_score
    ap.add_argument("--main-nms-iou", type=float, default=0.50)  # nms_iou_threshold
    ap.add_argument(
        "--private-nms-iou", type=float, default=0.70
    )  # private_candidate_nms_iou
    ap.add_argument(
        "--prior-iou-threshold", type=float, default=0.30
    )  # private_prior_iou_threshold
    ap.add_argument(
        "--crossing-iou", type=float, default=0.1
    )  # GT pair overlap to call "crossing"
    ap.add_argument(
        "--gt-match-iou", type=float, default=0.30
    )  # rescued box -> which GT it is
    ap.add_argument(
        "--tie-margin", type=float, default=0.15
    )  # ambiguity band on cost_iou diff
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import cv2

    detector = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=IMG_SIZE,
        device=args.device,
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    detector.eval()
    detector.mamba_head.set_head_compile(False)

    weight_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    records = []  # per-ambiguous-case dict for offline dump
    n_crossing_frames = 0
    n_rescued_total = 0
    n_ambiguous = 0

    for seq in (s.strip() for s in args.sequences.split(",") if s.strip()):
        seq_root = Path(args.data_root) / args.split / seq
        gts = load_gt(seq_root / "gt" / "gt.txt")
        img_dir = seq_root / "img1"
        frame_ids = sorted(gts.keys())
        if args.max_frames > 0:
            frame_ids = frame_ids[: args.max_frames]

        for n, fid in enumerate(frame_ids, start=1):
            cur = gts.get(fid, {})
            prev = gts.get(fid - 1, {})
            if len(cur) < 2:
                continue
            # find crossing pairs present in both frames
            ids = list(cur.keys())
            pairs = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if a not in prev or b not in prev:
                        continue
                    if iou_xyxy(cur[a], cur[b]) > args.crossing_iou:
                        pairs.append((a, b))
            if not pairs:
                continue
            n_crossing_frames += 1

            img_path = img_dir / f"{fid:06d}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sx, sy = w / IMG_SIZE, h / IMG_SIZE

            def to640(box):
                x1, y1, x2, y2 = box
                return (x1 / sx, y1 / sy, x2 / sx, y2 / sy)

            fb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(args.device)
            f640 = resize_stretch_batch_gpu(fb, IMG_SIZE, args.device)
            with torch.inference_mode():
                dets, _ = detector.forward(f640.float(), gate_input=None)
            if isinstance(dets, list):
                dets = dets[0]
            if dets.dim() == 3:
                dets = dets.squeeze(0)
            keep = dets[:, 4] > args.score_floor
            dets = dets[keep]
            if dets.numel() == 0:
                continue
            boxes = dets[:, :4]
            scores = dets[:, 4]

            main_idx = set(nms(boxes, scores, args.main_nms_iou).tolist())
            wide_idx = set(nms(boxes, scores, args.private_nms_iou).tolist())
            rescued_idx = sorted(wide_idx - main_idx)
            if not rescued_idx:
                continue
            n_rescued_total += len(rescued_idx)
            boxes_np = boxes.cpu().numpy()

            for a, b in pairs:
                cur_a640, cur_b640 = to640(cur[a]), to640(cur[b])
                prev_a640, prev_b640 = to640(prev[a]), to640(prev[b])
                for ridx in rescued_idx:
                    rbox = tuple(boxes_np[ridx].tolist())
                    # prior-eligibility gate (matches append_private_continuation_kernel)
                    elig_a = iou_xyxy(rbox, prev_a640) >= args.prior_iou_threshold
                    elig_b = iou_xyxy(rbox, prev_b640) >= args.prior_iou_threshold
                    if not (elig_a or elig_b):
                        continue
                    # ground-truth label: which current-frame GT does this box match?
                    iou_cur_a = iou_xyxy(rbox, cur_a640)
                    iou_cur_b = iou_xyxy(rbox, cur_b640)
                    if max(iou_cur_a, iou_cur_b) < args.gt_match_iou:
                        continue
                    label = "A" if iou_cur_a > iou_cur_b else "B"

                    cost_a = 1.0 - iou_xyxy(rbox, prev_a640)
                    cost_b = 1.0 - iou_xyxy(rbox, prev_b640)
                    if abs(cost_a - cost_b) >= args.tie_margin:
                        continue  # not ambiguous, plain IoU already decides
                    n_ambiguous += 1

                    gap_a, dx_a = gap_dx(rbox, prev_a640)
                    gap_b, dx_b = gap_dx(rbox, prev_b640)
                    records.append(
                        {
                            "seq": seq,
                            "frame": fid,
                            "label": label,
                            "cost_a": cost_a,
                            "cost_b": cost_b,
                            "gap_a": gap_a,
                            "gap_b": gap_b,
                            "dx_a": dx_a,
                            "dx_b": dx_b,
                        }
                    )
            if n % 200 == 0:
                print(f"{seq} [{n}/{len(frame_ids)}] ambiguous_so_far={n_ambiguous}")

    print(
        f"\ncrossing_frames={n_crossing_frames}  rescued_total={n_rescued_total}  "
        f"ambiguous_cases={n_ambiguous}"
    )

    report: dict = {
        "n_crossing_frames": n_crossing_frames,
        "n_rescued_total": n_rescued_total,
        "n_ambiguous": n_ambiguous,
        "params": vars(args) | {"device": args.device},
    }

    if n_ambiguous == 0:
        report["verdict"] = "no ambiguous cases found -- cannot evaluate"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print("no ambiguous cases -- wrote empty report")
        return

    labels = np.array([1 if r["label"] == "A" else 0 for r in records])  # 1=A, 0=B
    cost_a = np.array([r["cost_a"] for r in records])
    cost_b = np.array([r["cost_b"] for r in records])
    gap_a = np.array([r["gap_a"] for r in records])
    gap_b = np.array([r["gap_b"] for r in records])
    dx_a = np.array([r["dx_a"] for r in records])
    dx_b = np.array([r["dx_b"] for r in records])

    baseline_pred = (cost_a < cost_b).astype(int)  # 1 -> A predicted
    baseline_acc = float((baseline_pred == labels).mean())
    print(
        f"\nbaseline (IoU-only) accuracy on ambiguous subset: {baseline_acc:.3f}  (n={n_ambiguous})"
    )

    report["baseline_accuracy"] = baseline_acc
    sweep = {}
    for w in weight_grid:
        aug_a = cost_a + w * np.abs(gap_a) + w * np.abs(dx_a)
        aug_b = cost_b + w * np.abs(gap_b) + w * np.abs(dx_b)
        pred = (aug_a < aug_b).astype(int)
        acc = float((pred == labels).mean())
        sweep[str(w)] = acc
        print(f"  w={w:>4}: accuracy={acc:.3f}  delta={acc - baseline_acc:+.3f}")
    report["weight_sweep_accuracy"] = sweep
    best_w = max(sweep, key=lambda k: sweep[k])
    best_acc = sweep[best_w]
    report["best_weight"] = best_w
    report["best_accuracy"] = best_acc
    report["best_delta_vs_baseline"] = best_acc - baseline_acc

    if best_acc - baseline_acc >= 0.05:
        verdict = (
            f"GO signal: gap_h/dx_norm improves ambiguous-assignment accuracy "
            f"{baseline_acc:.3f} -> {best_acc:.3f} (w={best_w}, +{best_acc - baseline_acc:.3f})"
        )
    elif baseline_acc - best_acc >= 0.05:
        verdict = (
            f"NO-GO (harmful): best weighted variant {best_acc:.3f} < baseline "
            f"{baseline_acc:.3f}"
        )
    else:
        verdict = (
            f"NO-GO (neutral): best {best_acc:.3f} vs baseline {baseline_acc:.3f}, "
            f"within noise for n={n_ambiguous}"
        )
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    npz_path = str(Path(args.output).with_suffix(".npz"))
    np.savez_compressed(
        npz_path,
        labels=labels,
        cost_a=cost_a,
        cost_b=cost_b,
        gap_a=gap_a,
        gap_b=gap_b,
        dx_a=dx_a,
        dx_b=dx_b,
    )
    print(f"wrote {args.output} and {npz_path}")


if __name__ == "__main__":
    main()
