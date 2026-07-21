#!/usr/bin/env python3
"""Can an OCCLUSION signal SAFELY relax the low-IoU association gate?

Idea (user): a track's matching gate rejects low-IoU detections. An OCCLUDED person's
detection is partial/shifted -> low IoU with the track's box -> gate-rejected ->
mid_break FN / fragmentation. If the low IoU is EXPLAINED by occlusion, accept it;
if not, reject. This needs occlusion to DISCRIMINATE true vs false matches *within the
low-IoU band* (else relaxing the gate just admits false matches -- the same trap that
killed the relink-occlusion idea).

This probe builds frame-to-frame association candidates with an oracle track (GT box at
t-1) and real detections at t, then asks: among candidate pairs in the LOW-IoU band,
does the detection's occlusion (oracle GT visibility) separate TRUE matches (same GT id)
from FALSE (different person nearby)?

  AUC(occluded -> true) in low-IoU band ~0.5  => occlusion non-discriminative => NO-GO
  AUC >> 0.5                                  => occlusion safely gates relaxation => GO

Detector-only (resize-stretch 640, single pass), reuses build_mamba_gated_detector.

Usage
-----
  .venv/bin/python scripts/eval/probe_lowiou_occ_gate.py \
      --output results/occ_separability/lowiou_gate.json
"""
# status: stable

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
from sklearn.metrics import roc_auc_score  # noqa: E402

from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    resize_stretch_batch_gpu,
)
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

IMG = 640


def load_gt(p: Path) -> dict[int, list[tuple]]:
    """{frame: [(gid, x1,y1,x2,y2, vis), ...]} in original px (class-1 ped)."""
    out: dict[int, list[tuple]] = {}
    for line in p.read_text().splitlines():
        c = line.strip().split(",")
        if len(c) < 9 or int(c[6]) != 1 or int(c[7]) != 1 or float(c[8]) < 0.1:
            continue
        f, gid = int(c[0]), int(c[1])
        x, y, w, h = (float(v) for v in c[2:6])
        out.setdefault(f, []).append((gid, x, y, x + w, y + h, float(c[8])))
    return out


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
        "--trt-backbone-engine", default="models/yolo/yolo26s_backbone_640_best.engine"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--gt-match-iou", type=float, default=0.5)
    ap.add_argument(
        "--cand-iou",
        type=float,
        default=0.05,
        help="min IoU(track_{t-1}, det) to enter the candidate gate",
    )
    ap.add_argument(
        "--low-iou",
        type=float,
        default=0.5,
        help="IoU below this = the low-IoU band the gate would reject",
    )
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import cv2

    det = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=IMG,
        device=args.device,
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    det.eval()
    det.mamba_head.set_head_compile(False)

    # candidate records: iou(track,det), is_true, det_vis (occlusion of the det)
    iou_c: list[float] = []
    is_true: list[int] = []
    det_vis: list[float] = []

    for seq in (s.strip() for s in args.sequences.split(",") if s.strip()):
        root = Path(args.data_root) / args.split / seq
        gts = load_gt(root / "gt" / "gt.txt")
        frames = sorted((root / "img1").glob("*.jpg"))
        if args.max_frames > 0:
            frames = frames[: args.max_frames]
        prev_gt: dict[int, tuple] = {}  # gid -> (x1,y1,x2,y2) at t-1
        for n, fp in enumerate(frames, 1):
            fid = int(fp.stem)
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            sx, sy = w / IMG, h / IMG
            fb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(args.device)
            f640 = resize_stretch_batch_gpu(fb, IMG, args.device)
            with torch.inference_mode():
                d, _ = det.forward(f640.float(), gate_input=None)
            if isinstance(d, list):
                d = d[0]
            if d.dim() == 3:
                d = d.squeeze(0)
            d = d[d[:, 4] > args.conf_thresh]
            if d.numel():
                keep = torchvision.ops.nms(d[:, :4], d[:, 4], args.nms_iou)
                d = d[keep]
            # det boxes -> original px coords
            det_boxes = d[:, :4].clone()
            det_boxes[:, [0, 2]] *= sx
            det_boxes[:, [1, 3]] *= sy
            det_boxes = det_boxes.cpu()

            items = gts.get(fid, [])
            # match detections to GT_t -> det_gt id + det visibility
            det_gt = [-1] * len(det_boxes)
            det_v = [1.0] * len(det_boxes)
            if items and len(det_boxes):
                gtb = torch.tensor([[it[1], it[2], it[3], it[4]] for it in items])
                ious = torchvision.ops.box_iou(det_boxes, gtb)
                for j in range(len(det_boxes)):
                    bi, gi = ious[j].max(0)
                    if float(bi) >= args.gt_match_iou:
                        det_gt[j] = items[int(gi)][0]
                        det_v[j] = items[int(gi)][5]

            # candidate pairs: oracle track = GT box at t-1, vs each detection
            if prev_gt and len(det_boxes):
                trk_ids = list(prev_gt.keys())
                trk_box = torch.tensor([prev_gt[i] for i in trk_ids])
                tio = torchvision.ops.box_iou(trk_box, det_boxes)
                for ti, gid in enumerate(trk_ids):
                    for j in range(len(det_boxes)):
                        iouv = float(tio[ti, j])
                        if iouv < args.cand_iou:
                            continue
                        iou_c.append(iouv)
                        is_true.append(int(det_gt[j] == gid))
                        det_vis.append(det_v[j])
            # update prev_gt = GT boxes at t (true track positions)
            prev_gt = {it[0]: (it[1], it[2], it[3], it[4]) for it in items}
            if n % 300 == 0:
                print(f"{seq} [{n}/{len(frames)}] cand_pairs={len(iou_c)}")

    iou = np.array(iou_c)
    y = np.array(is_true)
    occ = 1.0 - np.array(det_vis)  # higher = more occluded
    rep: dict = {"n_pairs": int(len(y)), "n_true": int(y.sum())}

    print(f"\ncandidate pairs: {len(y)}  true: {int(y.sum())} ({100 * y.mean():.1f}%)")

    bands = [
        ("ALL", iou >= args.cand_iou),
        (
            "low-IoU [{:.2f},{:.2f})".format(args.cand_iou, args.low_iou),
            (iou >= args.cand_iou) & (iou < args.low_iou),
        ),
        (
            "very-low [{:.2f},0.30)".format(args.cand_iou),
            (iou >= args.cand_iou) & (iou < 0.30),
        ),
    ]
    print(f"\n{'band':24s}{'n':>8s}{'base%':>8s}{'occAUC':>9s}{'iouAUC':>9s}")
    rep["bands"] = {}
    for name, m in bands:
        k = int(m.sum())
        if k < 30 or len(np.unique(y[m])) < 2:
            print(f"{name:24s}{k:>8d}{'-':>8s}{'-':>9s}{'-':>9s}")
            continue
        base = float(y[m].mean())
        occ_auc = float(roc_auc_score(y[m], occ[m]))  # does occlusion->true?
        iou_auc = float(roc_auc_score(y[m], iou[m]))  # geom baseline in-band
        rep["bands"][name] = {
            "n": k,
            "base_rate": base,
            "occ_auc": occ_auc,
            "iou_auc": iou_auc,
        }
        print(f"{name:24s}{k:>8d}{100 * base:>7.1f}%{occ_auc:>9.3f}{iou_auc:>9.3f}")

    low = (iou >= args.cand_iou) & (iou < args.low_iou)
    if low.sum() >= 30 and len(np.unique(y[low])) > 1:
        a = rep["bands"].get(
            "low-IoU [{:.2f},{:.2f})".format(args.cand_iou, args.low_iou), {}
        )
        occ_auc = a.get("occ_auc", float("nan"))
        rep["verdict"] = (
            f"GO: occlusion separates true/false in low-IoU band (AUC {occ_auc:.3f})"
            if occ_auc >= 0.65
            else f"NO-GO: occlusion non-discriminative in low-IoU band (AUC {occ_auc:.3f}) "
            f"-> relaxing the gate by occlusion admits false matches too"
        )
        print(f"\nVERDICT: {rep['verdict']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
