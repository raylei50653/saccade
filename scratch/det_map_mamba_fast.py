"""Fast detector-only mAP for the Mamba head with TRT backbone + NMS.

Batches image decoding and skips AdaptiveFramePool/detect_single_patch_960
overhead for maximum throughput.

Usage:
  uv run scratch/det_map_mamba_fast.py \
      --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt --img-size 640
"""

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts" / "eval"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from tqdm import tqdm

from detection_map import _load_sequence_ground_truth, _sequence_frame_ids
from saccade.perception.eval.metrics import compute_detection_mean_ap
from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)


def _load_ignore_gt(gt_path: Path, max_frames: int) -> dict:
    rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    out: dict = {}
    for row in rows:
        fid = int(row[0])
        if max_frames > 0 and fid > max_frames:
            continue
        if int(row[6]) == 1 and int(row[7]) == 1:
            continue
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


def _filter_ignored(preds, target_gt, ignore_boxes, thr=0.5):
    if not preds or not ignore_boxes:
        return preds
    pb = np.array([p["bbox"] for p in preds], dtype=np.float64)
    tb = np.array([g["bbox"] for g in target_gt], dtype=np.float64) if target_gt else np.zeros((0, 4))
    ib = np.array(ignore_boxes, dtype=np.float64)
    iou_t = _iou_max(pb, tb)
    iou_i = _iou_max(pb, ib)
    return [p for k, p in enumerate(preds) if iou_t[k] >= thr or iou_i[k] < thr]


def _load_frame_batch(img_dir: Path, frame_ids: list[str], device: str) -> torch.Tensor:
    """Load a batch of frames as [N, 3, H, W] normalized RGB tensor on GPU."""
    frames = []
    for fid in frame_ids:
        bgr = cv2.imread(str(img_dir / f"{fid}.jpg"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(t)
    return torch.stack(frames)


def _predict_batch(
    detector,
    frames: torch.Tensor,
    h_orig: int,
    w_orig: int,
    conf_thr: float,
    nms_iou: float,
) -> list[list[dict]]:
    """Detect on a batch of frames, returning per-frame prediction dicts.

    Calls detect_raw once with the full batch for maximum TRT throughput.
    """
    B = frames.shape[0]
    sz = detector.img_size
    scale_x = w_orig / sz
    scale_y = h_orig / sz

    frames_640 = F.interpolate(frames, size=(sz, sz), mode="bilinear", align_corners=False)
    raw = detector.detect_raw(frames_640)

    all_preds: list[list[dict]] = []

    for b in range(B):
        boxes = raw[b, :, :4].clone()
        scores = raw[b, :, 4].clone()
        classes = raw[b, :, 5].clone()

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        keep = (classes.to(torch.int32) == 0) & (scores >= conf_thr)
        if not keep.any():
            all_preds.append([])
            continue

        boxes = boxes[keep]
        scores = scores[keep]

        nms_idx = nms(boxes, scores, nms_iou)
        boxes_np = boxes[nms_idx].cpu().numpy()
        scores_np = scores[nms_idx].cpu().numpy()

        preds = []
        for box, score in zip(boxes_np, scores_np):
            preds.append({"bbox": [float(v) for v in box], "score": float(score), "class_id": 0})
        all_preds.append(preds)

    return all_preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mamba-ckpt", required=True)
    p.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    p.add_argument("--trt-backbone", default="models/yolo/yolo26s_backbone_640_best.engine")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument("--sequences", default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,"
                   "MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP")
    p.add_argument("--conf-threshold", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Frames to load in one batch (higher = less CPU/GPU sync, more VRAM).")
    p.add_argument("--max-det", type=int, default=300,
                   help="Max detections per frame before NMS (default: 300).")
    p.add_argument("--no-ignore", action="store_true")
    args = p.parse_args()

    print(f"Building detector (TRT backbone: {args.trt_backbone})...")
    det = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=args.img_size,
        device="cuda",
        conf_thr=0.001,
        max_det=args.max_det,
        trt_backbone_engine=args.trt_backbone,
        use_cuda_graph=True,
        use_whole_graph=False,
    )

    seq_list = [s.strip() for s in args.sequences.split(",") if s.strip()]

    all_gt: dict = {}
    all_pred: dict = {}

    with torch.inference_mode():
        for seq in seq_list:
            seq_root = Path(args.data_root) / args.split / seq
            img_dir = seq_root / "img1"
            gt_path = seq_root / "gt" / "gt.txt"

            frame_ids = _sequence_frame_ids(img_dir, args.max_frames)
            if not frame_ids:
                continue

            seq_gt = _load_sequence_ground_truth(
                gt_path, max_frames=args.max_frames, gt_mark=1, gt_class_id=1
            )
            ignore_gt = {} if args.no_ignore else _load_ignore_gt(gt_path, args.max_frames)

            # Determine frame dimensions from first image
            first = cv2.imread(str(img_dir / f"{frame_ids[0]}.jpg"))
            h_orig, w_orig = first.shape[:2]

            seq_pred: dict = {}
            bs = max(1, args.batch_size)

            for i in tqdm(range(0, len(frame_ids), bs), desc=f"  {seq}", leave=False):
                batch_ids = frame_ids[i:i + bs]
                frames = _load_frame_batch(img_dir, batch_ids, "cuda")
                batch_preds = _predict_batch(det, frames, h_orig, w_orig,
                                             args.conf_threshold, args.nms_iou)
                for fid, preds in zip(batch_ids, batch_preds):
                    seq_pred[fid] = _filter_ignored(preds, seq_gt.get(fid, []), ignore_gt.get(fid, []))

            all_gt.update({f"{seq}/{k}": v for k, v in seq_gt.items()})
            all_pred.update({f"{seq}/{k}": v for k, v in seq_pred.items()})

            m50 = compute_detection_mean_ap(
                {f"{seq}/{k}": v for k, v in seq_gt.items()},
                {f"{seq}/{k}": v for k, v in seq_pred.items()},
                iou_thresholds=(0.5,), class_ids=(0,),
            )
            print(f"  {seq}: AP50={m50['mAP']:.4f}")

    m50 = compute_detection_mean_ap(all_gt, all_pred, iou_thresholds=(0.5,), class_ids=(0,))
    mcoco = compute_detection_mean_ap(
        all_gt, all_pred,
        iou_thresholds=tuple(np.arange(0.5, 1.0, 0.05).tolist()), class_ids=(0,),
    )
    print(f"\n=== {args.mamba_ckpt} @ {args.img_size} (TRT backbone + NMS) ===")
    print(f"  AP@0.5      = {m50['mAP']:.4f}")
    print(f"  AP@0.5:0.95 = {mcoco['mAP']:.4f}")


if __name__ == "__main__":
    main()
