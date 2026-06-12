"""Detector-only mAP for the Mamba head (no tracker), eager backbone.

Compares pure detection quality across resolutions/checkpoints, isolating the
detector from the tracker. Reuses detection_map.py infra + detect_single_patch_960.

Usage:
  LD_PRELOAD=... .venv/bin/python scratch/det_map_mamba.py \
      --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt --img-size 640
"""

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts" / "eval"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from detection_map import (  # noqa: E402
    _load_sequence_ground_truth,
    _sequence_frame_ids,
    _load_frame_rgb_tensor,
)
from saccade.perception.eval.detection import detect_single_patch_960  # noqa: E402
from saccade.perception.eval.metrics import compute_detection_mean_ap  # noqa: E402
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)


def _load_ignore_gt(gt_path: Path, max_frames: int) -> dict:
    """Non-target GT (class!=1 or mark!=1): distractor/static/reflection/vehicle.
    Detections matching these are ignored (neither TP nor FP) per MOTChallenge."""
    rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    out: dict = {}
    for row in rows:
        fid = int(row[0])
        if max_frames > 0 and fid > max_frames:
            continue
        if int(row[6]) == 1 and int(row[7]) == 1:
            continue  # this is a target pedestrian, not ignore
        x, y, w, h = [float(v) for v in row[2:6]]
        out.setdefault(f"{fid:06d}", []).append([x, y, x + w, y + h])
    return out


def _iou_max(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros(len(pred_boxes))
    pa = pred_boxes[:, None, :]
    ga = gt_boxes[None, :, :]
    ix1 = np.maximum(pa[..., 0], ga[..., 0])
    iy1 = np.maximum(pa[..., 1], ga[..., 1])
    ix2 = np.minimum(pa[..., 2], ga[..., 2])
    iy2 = np.minimum(pa[..., 3], ga[..., 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_p = (pa[..., 2] - pa[..., 0]) * (pa[..., 3] - pa[..., 1])
    area_g = (ga[..., 2] - ga[..., 0]) * (ga[..., 3] - ga[..., 1])
    iou = inter / (area_p + area_g - inter + 1e-9)
    return iou.max(axis=1)


def _predict_mamba_with_nms(
    detector,
    pool,
    frame: torch.Tensor,
    conf_threshold: float = 0.001,
    nms_iou_threshold: float = 0.45,
) -> list[dict]:
    h_orig, w_orig = int(frame.shape[1]), int(frame.shape[2])
    pool.frame_buffer.copy_(frame)

    boxes, scores, classes, _ = detect_single_patch_960(
        detector, pool, h_orig, w_orig,
        preprocess_modes=[], detector_box_format="xyxy",
    )

    keep = (classes.to(torch.int32) == 0) & (scores >= conf_threshold)
    if not keep.any():
        return []

    boxes = boxes[keep]
    scores = scores[keep]

    nms_idx = nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_idx]
    scores = scores[nms_idx]

    preds: list[dict] = []
    for box, score in zip(boxes.detach().cpu().numpy(), scores.detach().cpu().numpy()):
        preds.append({
            "bbox": [float(v) for v in box.tolist()],
            "score": float(score),
            "class_id": 0,
        })
    return preds


def _filter_ignored(preds: list, target_gt: list, ignore_boxes: list, thr: float = 0.5) -> list:
    """Drop predictions that fall in an ignore zone and don't match a target."""
    if not preds or not ignore_boxes:
        return preds
    pb = np.array([p["bbox"] for p in preds], dtype=np.float64)
    tb = np.array([g["bbox"] for g in target_gt], dtype=np.float64) if target_gt else np.zeros((0, 4))
    ib = np.array(ignore_boxes, dtype=np.float64)
    iou_t = _iou_max(pb, tb)
    iou_i = _iou_max(pb, ib)
    return [p for k, p in enumerate(preds) if iou_t[k] >= thr or iou_i[k] < thr]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mamba-ckpt", required=True)
    p.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,"
        "MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    p.add_argument("--conf-threshold", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45).")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-ignore", action="store_true",
                   help="Disable MOT17 ignore-region handling (count all as FP).")
    args = p.parse_args()

    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=args.img_size,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine="",   # eager backbone (no broken TRT engine)
        use_cuda_graph=False,
        use_whole_graph=False,
    )

    all_gt: dict = {}
    all_pred: dict = {}
    pool = None
    pool_shape = None

    with torch.inference_mode():
        for seq in [s.strip() for s in args.sequences.split(",") if s.strip()]:
            seq_root = Path(args.data_root) / args.split / seq
            img_dir = seq_root / "img1"
            gt_path = seq_root / "gt" / "gt.txt"
            frame_ids = _sequence_frame_ids(img_dir, args.max_frames)
            seq_gt = _load_sequence_ground_truth(
                gt_path, max_frames=args.max_frames, gt_mark=1, gt_class_id=1
            )
            ignore_gt = (
                {} if args.no_ignore
                else _load_ignore_gt(gt_path, args.max_frames)
            )
            seq_pred: dict = {}
            for fid in frame_ids:
                frame = _load_frame_rgb_tensor(img_dir / f"{fid}.jpg", "cuda")
                h, w = int(frame.shape[1]), int(frame.shape[2])
                if pool is None or pool_shape != (h, w):
                    pool = AdaptiveFramePool(h, w, device="cuda")
                    pool_shape = (h, w)
                preds = _predict_mamba_with_nms(
                    det, pool, frame,
                    conf_threshold=args.conf_threshold,
                    nms_iou_threshold=args.nms_iou,
                )
                seq_pred[fid] = _filter_ignored(
                    preds, seq_gt.get(fid, []), ignore_gt.get(fid, [])
                )
            all_gt.update({f"{seq}/{k}": v for k, v in seq_gt.items()})
            all_pred.update({f"{seq}/{k}": v for k, v in seq_pred.items()})
            m50 = compute_detection_mean_ap(
                {f"{seq}/{k}": v for k, v in seq_gt.items()},
                {f"{seq}/{k}": v for k, v in seq_pred.items()},
                iou_thresholds=(0.5,), class_ids=(0,),
            )
            print(f"  {seq}: AP50={m50['mAP']:.4f}")

    m50 = compute_detection_mean_ap(
        all_gt, all_pred, iou_thresholds=(0.5,), class_ids=(0,)
    )
    mcoco = compute_detection_mean_ap(
        all_gt, all_pred,
        iou_thresholds=tuple(np.arange(0.5, 1.0, 0.05).tolist()), class_ids=(0,),
    )
    print(f"\n=== {args.mamba_ckpt} @ {args.img_size} (no tracker) ===")
    print(f"  AP@0.5      = {m50['mAP']:.4f}")
    print(f"  AP@0.5:0.95 = {mcoco['mAP']:.4f}")


if __name__ == "__main__":
    main()
