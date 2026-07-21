#!/usr/bin/env python3
"""Analyze FPN embedding discriminability: intra-ID vs inter-ID cosine similarity.

Matches detector output to MOT17 ground-truth box identities across frames,
extracts raw FPN embeddings (centre-pool or trained head), and computes
same-person vs different-person feature separation.

Usage:
    # Centre-pool only (default):
    uv run scripts/eval/analyze_fpn_embeddings.py --seq MOT17-04-SDP --max-frames 200

    # With trained DimReduceHead:
    uv run scripts/eval/analyze_fpn_embeddings.py --seq MOT17-04-SDP --head-ckpt runs/jde_market_v9b/best.ckpt

    # All 7 SDP sequences:
    uv run scripts/eval/analyze_fpn_embeddings.py --seq all --max-frames 100
"""
# status: diagnostic

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "build"))

import saccade_tracking_ext  # noqa: E402, F401

from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    resize_stretch_batch_gpu,
)
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

IMG_SIZE = 640


def load_gt(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Parse MOT17 gt.txt → {frame_id: [(gt_id, x, y, w, h), ...]}."""
    gt: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        fid, gid = int(parts[0]), int(parts[1])
        cls_id = int(parts[7]) if len(parts) > 7 else 1
        visibility = float(parts[6]) if len(parts) > 6 else 1.0
        if cls_id != 1 or visibility < 0.3:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        gt.setdefault(fid, []).append((gid, x, y, w, h))
    return gt


def match_detections_to_gt(
    dets_xyxy: torch.Tensor,  # (N, 4) in 640 space
    gt_items: list[tuple[int, float, float, float, float]],
    sx: float,
    sy: float,
) -> list[tuple[int, torch.Tensor]]:
    """IoU-match detections to GT boxes. Returns [(gt_id, detection_box), ...]."""
    if dets_xyxy.numel() == 0 or not gt_items:
        return []

    gt_boxes_640 = torch.tensor(
        [
            [(gx) / sx, (gy) / sy, (gx + gw) / sx, (gy + gh) / sy]
            for _, gx, gy, gw, gh in gt_items
        ],
        device=dets_xyxy.device,
        dtype=torch.float32,
    )

    dx1, dy1 = dets_xyxy[:, 0], dets_xyxy[:, 1]
    dx2, dy2 = dets_xyxy[:, 2], dets_xyxy[:, 3]
    area_d = (dx2 - dx1) * (dy2 - dy1)

    pairs: list[tuple[int, torch.Tensor]] = []
    for j, (gid, _, _, _, _) in enumerate(gt_items):
        gx1, gy1, gx2, gy2 = gt_boxes_640[j]
        ix1 = torch.max(dx1, gx1)
        iy1 = torch.max(dy1, gy1)
        ix2 = torch.min(dx2, gx2)
        iy2 = torch.min(dy2, gy2)
        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        area_g = max((gx2 - gx1) * (gy2 - gy1), 1.0)
        iou = inter / (area_d + area_g - inter + 1e-7)
        best = iou.argmax().item()
        if iou[best] > 0.3:
            pairs.append((gid, dets_xyxy[best]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--seq", default="MOT17-04-SDP")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    parser.add_argument("--head-ckpt", default="", help="Trained DimReduceHead .ckpt")
    parser.add_argument(
        "--method",
        default="centre",
        choices=["centre", "roi"],
        help="Feature extraction method",
    )
    parser.add_argument(
        "--min-samples-per-id",
        type=int,
        default=2,
        help="Min samples per identity to include in stats",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    detector = build_mamba_gated_detector(
        str((_project_root / args.yolo_weights).resolve()),
        str((_project_root / args.teacher_ckpt).resolve()),
        str((_project_root / args.mamba_ckpt).resolve()),
        img_size=IMG_SIZE,
        device=device,
        emb_dim=128,
    )
    detector.eval()

    # Warm-up (same as eval_fpn_reid.py)
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    detector.forward(dummy.float(), gate_input=None)

    teacher = detector.teacher

    # Load trained head if provided
    reid_head = None
    conv_weights = None
    proj_weight = running_mean = running_var = None
    if args.head_ckpt:
        from scripts.eval.eval_fpn_reid import DimReduceHead  # noqa: E402
        from saccade.perception.tracking.fpn_reid_cuda import fpn_reid_extract_cuda  # noqa: E402

        head_path = Path(args.head_ckpt)
        if not head_path.is_absolute():
            head_path = _project_root / head_path
        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        in_channels = ckpt.get("in_channels", [128, 256, 512])
        reid_head = DimReduceHead(in_channels, out_dim=128).to(device)
        reid_head.load_state_dict(ckpt["head"])
        reid_head.eval()
        conv_weights = [reid_head.convs[i].weight.data for i in range(len(in_channels))]
        if hasattr(reid_head.proj, "__getitem__") and hasattr(
            reid_head.proj[0], "weight"
        ):
            proj_weight = reid_head.proj[0].weight.data
            if len(reid_head.proj) > 1:
                bn = reid_head.proj[1]
                running_mean = bn.running_mean.data
                running_var = bn.running_var.data
        dim = 128
        print(f"Loaded DimReduceHead: {dim}-dim")
    else:
        dim = 896
        print(f"Raw FPN: {dim}-dim (centre-pool)")

    # Collect: gt_id → [embedding, ...]
    id_embs: dict[int, list[np.ndarray]] = {}

    data_root = _project_root / args.data_root / "MOT17/train"
    if args.seq == "all":
        seqs = [
            "MOT17-02-SDP",
            "MOT17-04-SDP",
            "MOT17-05-SDP",
            "MOT17-09-SDP",
            "MOT17-10-SDP",
            "MOT17-11-SDP",
            "MOT17-13-SDP",
        ]
    else:
        seqs = [args.seq]

    for seq in seqs:
        seq_dir = data_root / seq
        gt = load_gt(seq_dir / "gt/gt.txt")
        img_dir = seq_dir / "img1"
        seq_id_embs: dict[int, list[np.ndarray]] = {}

        frame_ids = sorted(gt.keys())[: args.max_frames]
        print(f"\n{seq}: {len(frame_ids)} frames")

        for fid in frame_ids:
            img_path = img_dir / f"{fid:06d}.jpg"
            if not img_path.exists():
                continue

            import cv2

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sx, sy = w / IMG_SIZE, h / IMG_SIZE

            fb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            f640 = resize_stretch_batch_gpu(fb, IMG_SIZE, device)

            dets, _ = detector.forward(f640.float(), gate_input=None)
            if isinstance(dets, list):
                dets = dets[0]
            if dets.dim() == 3:
                dets = dets.squeeze(0)
            valid = dets[:, 4] > args.conf_thresh
            dets = dets[valid]
            if dets.numel() == 0:
                continue

            det_boxes = dets[:, :4]
            matches = match_detections_to_gt(det_boxes, gt.get(fid, []), sx, sy)

            if not matches:
                continue

            matched_boxes = torch.stack([b for _, b in matches])
            matched_gids = [gid for gid, _ in matches]

            if args.head_ckpt and reid_head is not None:
                fpn = [
                    teacher._gate_layers[s]._feat_cache[s] for s in ("p3", "p4", "p5")
                ]
                with torch.no_grad():
                    embs = fpn_reid_extract_cuda(
                        fpn,
                        conv_weights,
                        matched_boxes,
                        IMG_SIZE,
                        proj_weight=proj_weight,
                        running_mean=running_mean,
                        running_var=running_var,
                    )
            elif args.method == "roi":
                from saccade.perception.temporal_yolo.roi_embedder import (  # noqa: E402
                    extract_roi_embeddings,
                )

                feat_cache = {
                    s: teacher._gate_layers[s]._feat_cache[s]
                    for s in ("p3", "p4", "p5")
                }
                embs = extract_roi_embeddings(feat_cache, matched_boxes)
            else:
                embs = detector.extract_fpn_embeddings(None, matched_boxes)

            for gid, emb in zip(matched_gids, embs.cpu().numpy()):
                seq_id_embs.setdefault(gid, []).append(emb)

        # Merge seq-level into global
        for gid, e_list in seq_id_embs.items():
            id_embs.setdefault(gid, []).extend(e_list)

    # Filter to identities with >= min_samples
    valid_ids = [
        gid for gid, e_list in id_embs.items() if len(e_list) >= args.min_samples_per_id
    ]
    if len(valid_ids) < 2:
        print(
            f"\nOnly {len(valid_ids)} identities with >= {args.min_samples_per_id} samples. "
            "Increase --max-frames or lower --min-samples-per-id."
        )
        return

    # Compute intra-ID and inter-ID cosine similarities
    intra_sims, inter_sims = [], []
    for i, gid in enumerate(valid_ids):
        embs = np.array(id_embs[gid], dtype=np.float32)
        n = len(embs)
        for a in range(n):
            for b in range(a + 1, n):
                intra_sims.append(float(np.dot(embs[a], embs[b])))
        for j, other_gid in enumerate(valid_ids):
            if i >= j:
                continue
            other = np.array(id_embs[other_gid], dtype=np.float32)
            for a in range(len(embs)):
                for b in range(len(other)):
                    inter_sims.append(float(np.dot(embs[a], other[b])))

    intra = np.array(intra_sims, dtype=np.float64)
    inter = np.array(inter_sims, dtype=np.float64)

    # Print report
    print(f"\n{'=' * 60}")
    print("Embedding Discriminability Report")
    print(f"{'=' * 60}")
    print(f"Sequences: {', '.join(seqs)}")
    print(
        f"Identities: {len(valid_ids)}  "
        f"Total embeddings: {sum(len(id_embs[g]) for g in valid_ids)}"
    )
    print(f"Embedding dim: {dim}")
    print(f"Method: {args.method if not args.head_ckpt else 'trained head'}")
    print(f"\n{'─' * 60}")
    print(f"{'Metric':20s} {'Intra-ID':>12s} {'Inter-ID':>12s} {'Δ':>12s}")
    print(f"{'─' * 60}")
    print(f"{'Count':20s} {len(intra):>12,d} {len(inter):>12,d}")
    print(
        f"{'Mean':20s} {intra.mean():>12.4f} {inter.mean():>12.4f} {intra.mean() - inter.mean():>12.4f}"
    )
    print(f"{'Std':20s} {intra.std():>12.4f} {inter.std():>12.4f}")
    print(f"{'Min':20s} {intra.min():>12.4f} {inter.min():>12.4f}")
    print(
        f"{'P05':20s} {np.percentile(intra, 5):>12.4f} {np.percentile(inter, 5):>12.4f}"
    )
    print(
        f"{'P25':20s} {np.percentile(intra, 25):>12.4f} {np.percentile(inter, 25):>12.4f}"
    )
    print(
        f"{'P50':20s} {np.percentile(intra, 50):>12.4f} {np.percentile(inter, 50):>12.4f}"
    )
    print(
        f"{'P75':20s} {np.percentile(intra, 75):>12.4f} {np.percentile(inter, 75):>12.4f}"
    )
    print(
        f"{'P95':20s} {np.percentile(intra, 95):>12.4f} {np.percentile(inter, 95):>12.4f}"
    )
    print(f"{'Max':20s} {intra.max():>12.4f} {inter.max():>12.4f}")
    print(f"{'─' * 60}")
    print(f"{'Separation (mean Δ)':20s} {intra.mean() - inter.mean():>12.4f}")
    print(
        f"{'Cohen d':20s} {(intra.mean() - inter.mean()) / max((intra.std() ** 2 + inter.std() ** 2) / 2, 1e-10) ** 0.5:>12.4f}"
    )
    overlap = (intra < np.percentile(inter, 50)).mean()
    print(f"{'Intra < Inter median':20s} {overlap:>11.1%}")
    overlap95 = (intra < np.percentile(inter, 95)).mean()
    print(f"{'Intra < Inter p95':20s} {overlap95:>11.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
