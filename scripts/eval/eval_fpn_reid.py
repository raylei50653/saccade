#!/usr/bin/env python3
"""MOT evaluation with FPN-based ReID (raw or 1×1 Conv head).

Supports two modes:
  --head-ckpt runs/.../best.ckpt    Trained DimReduceHead (128-dim, fast)
  (no --head-ckpt)                  Raw 896-dim FPN (zero-training)

Single YOLO forward pass per frame: detection + FPN feature caching.
Appearance bank maintains per-track embeddings with EMA.

Usage:
    # Trained head (128-dim, ~55 FPS expected):
    uv run scripts/eval/eval_fpn_reid.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --head-ckpt runs/jde_market_v9b/best.ckpt \
        --seq MOT17-04-SDP --max-frames 200

    # Raw FPN (896-dim, zero-training):
    uv run scripts/eval/eval_fpn_reid.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --seq MOT17-04-SDP --max-frames 200

    # Baseline without ReID:
    uv run scripts/eval/eval_fpn_reid.py \
        --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
        --seq MOT17-04-SDP --max-frames 200 --no-reid
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: F401, E402

from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)
from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    GPUByteTracker,
)
from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    resize_stretch_batch_gpu,
)
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    _GATE_LAYER_IDX,
)
from saccade.perception.tracking.fpn_reid_cuda import (  # noqa: E402
    fpn_reid_extract_cuda,
)
from saccade.perception.eval.metrics import run_motmetrics_evaluation  # noqa: E402

IMG_SIZE = 640


# ── 1×1 Conv head (mirrors train_reid_1x1.py) ──


class DimReduceHead(nn.Module):
    def __init__(self, in_channels: list[int], out_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.nl = len(in_channels)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_dim, 1, bias=False) for c in in_channels]
        )
        mid_dim = out_dim * self.nl
        if mid_dim != out_dim:
            self.proj = nn.Sequential(
                nn.Linear(mid_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        parts = []
        for conv, f in zip(self.convs, feats):
            x = conv(f)
            h, w = x.shape[2], x.shape[3]
            center = x[:, :, h // 2, w // 2]
            parts.append(center)
        pooled = torch.cat(parts, dim=1)
        return F.normalize(self.proj(pooled), dim=1)

    def forward_boxes(
        self,
        feats: list[torch.Tensor],
        boxes_xyxy: torch.Tensor,
        img_size: int = 640,
    ) -> torch.Tensor:
        """Per-box embeddings: center-pool each FPN level at bbox centers.

        Args:
            feats: list of (1, C, H, W) raw FPN feature maps
            boxes_xyxy: (N, 4) in 640×640 coords
            img_size: 640

        Returns:
            (N, out_dim * nl) pooled concat, L2-normalized
        """
        boxes_xyxy.shape[0]
        parts = []
        for conv, f in zip(self.convs, feats):
            x = conv(f)  # (1, out_dim, H, W)
            f_h, f_w = x.shape[2], x.shape[3]
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5
            cx_norm = cx / img_size
            cy_norm = cy / img_size
            cx_idx = (cx_norm * f_w).long().clamp(0, f_w - 1)
            cy_idx = (cy_norm * f_h).long().clamp(0, f_h - 1)
            feat_per_box = x[0][:, cy_idx, cx_idx].mT  # (N, out_dim)
            parts.append(feat_per_box)
        pooled = torch.cat(parts, dim=1)
        return F.normalize(self.proj(pooled), dim=1)


def load_sequence_frames(seq_name: str, data_root: str, max_frames: int):
    import cv2

    seq_dir = Path(data_root) / "MOT17" / "train" / seq_name
    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg"))
    if max_frames > 0:
        paths = paths[:max_frames]

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return frames


def _get_teacher_feats(teacher, yolo_model, frame):
    layers = yolo_model.model
    save = set(yolo_model.save)
    y: list[torch.Tensor | None] = []
    x = frame
    for i in range(23):
        m = layers[i]
        if m.f != -1:
            if isinstance(m.f, int):
                x = y[m.f]
            else:
                x = [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if i in save else None)
    fpn_indices = [_GATE_LAYER_IDX[s] for s in ("p3", "p4", "p5")]
    return [y[i] for i in fpn_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_960_v2/best.ckpt")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--seq", default="MOT17-04-SDP")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--output",
        default="output/fpn_reid",
        help="Output directory for MOT txt and metrics",
    )
    parser.add_argument(
        "--split", default="train", help="Data split subdirectory (e.g. train, test)"
    )
    parser.add_argument(
        "--cost-cos-w", type=float, default=0.10, help="Appearance cost: cos_sim weight"
    )
    parser.add_argument(
        "--cost-iou-w", type=float, default=0.70, help="Appearance cost: IoU weight"
    )
    parser.add_argument(
        "--cost-score-w",
        type=float,
        default=0.20,
        help="Appearance cost: detection score weight",
    )
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    parser.add_argument("--reid-weight", type=float, default=0.80)
    parser.add_argument("--cos-threshold", type=float, default=0.90)
    parser.add_argument("--iou-low", type=float, default=0.30)
    parser.add_argument("--iou-high", type=float, default=0.60)
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable FPN ReID for baseline comparison",
    )
    parser.add_argument(
        "--head-ckpt",
        default="",
        help="Trained DimReduceHead checkpoint (128-dim, recommended)",
    )
    parser.add_argument(
        "--bank-k", type=int, default=5, help="Appearance bank max samples per track"
    )
    parser.add_argument(
        "--bank-alpha",
        type=float,
        default=0.7,
        help="EMA decay for per-track embedding average",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequence
    print(f"\nLoading {args.seq}...")
    frames = load_sequence_frames(args.seq, args.data_root, args.max_frames)
    h_orig, w_orig = frames[0].shape[:2]
    print(f"  Frames: {len(frames)}  Resolution: {w_orig}×{h_orig}")

    # Build MambaGatedDetector
    print("\nBuilding MambaGatedDetector...")
    detector = build_mamba_gated_detector(
        yolo_pt_path=str((project_root / args.yolo_weights).resolve()),
        teacher_ckpt=str((project_root / args.teacher_ckpt).resolve()),
        mamba_ckpt=str((project_root / args.mamba_ckpt).resolve()),
        img_size=args.img_size,
        device=device,
        emb_dim=128,
    )
    detector.eval()

    teacher = detector.teacher

    # Probe FPN dimension
    dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
    detector.forward(dummy.float(), gate_input=None)

    # ── Load trained head if provided ──
    reid_head: DimReduceHead | None = None
    head_scales: str = "p3p4p5"
    if args.head_ckpt:
        head_path = Path(args.head_ckpt)
        if not head_path.is_absolute():
            head_path = project_root / head_path
        head_ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        in_channels = head_ckpt.get("in_channels", [128, 256, 512])
        head_scales = head_ckpt.get("scales", "p3p4p5")
        reid_head = DimReduceHead(in_channels, out_dim=128).to(device)
        reid_head.load_state_dict(head_ckpt["head"])
        reid_head.eval()
        fpn_dim = 128
        print(
            f"  Loaded DimReduceHead ({head_scales}): {sum(p.numel() for p in reid_head.parameters()):,} params, {fpn_dim}-dim"
        )

        # Extract weights for CUDA kernel
        conv_weights = [reid_head.convs[i].weight.data for i in range(len(in_channels))]
        has_proj = hasattr(reid_head.proj, "__getitem__") and hasattr(
            reid_head.proj[0], "weight"
        )
        proj_weight = reid_head.proj[0].weight.data if has_proj else None
        if has_proj and len(reid_head.proj) > 1:
            bn = reid_head.proj[1]
            running_mean = bn.running_mean.data
            running_var = bn.running_var.data
        else:
            running_mean = running_var = None
    else:
        fpn_dim = detector.extract_fpn_embeddings(
            None, torch.zeros(1, 4).to(device)
        ).shape[1]
        print(f"  Raw FPN embedding dim: {fpn_dim}")

    # Tracker — use correct embedding dimension
    tracker = GPUByteTracker(max_objects=2048, embedding_dim=fpn_dim)
    tracker.set_params(
        track_thresh=0.05,
        high_thresh=0.45,
        match_thresh=0.66,
        track_buffer=30,
        mid_thresh=0.10,
        new_track_thresh=0.28,
    )
    tracker.set_reid_params(
        cos_threshold=args.cos_threshold,
        iou_low=args.iou_low,
        iou_high=args.iou_high,
        weight=args.reid_weight,
        cost_cos_w=args.cost_cos_w,
        cost_iou_w=args.cost_iou_w,
        cost_score_w=args.cost_score_w,
    )

    result_bufs = tracker.allocate_result_buffers()

    # Per-track embedding bank: track_id → latest embedding (no EMA)
    bank_embeds: dict[int, torch.Tensor] = {}
    bank_counts: dict[int, int] = {}

    SYNC_INTERVAL = 1

    print(f"\nRunning tracking on {len(frames)} frames...")
    print(
        f"  Config: conf={args.conf_thresh} reid_w={args.reid_weight} "
        f"cos={args.cos_threshold} iou=({args.iou_low},{args.iou_high})"
    )
    if not args.no_reid:
        print(f"  FPN dim: {fpn_dim}  Sync every {SYNC_INTERVAL}f")

    t0 = time.perf_counter()

    mot_lines: list[str] = []

    for frame_idx, frame_np in enumerate(frames):
        frame_uint8 = (
            torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        frame_640 = resize_stretch_batch_gpu(frame_uint8, args.img_size, device)

        # ── Detect (caches raw FPN internally) ──
        detections_raw, _extra = detector.forward(frame_640.float(), gate_input=None)

        if isinstance(detections_raw, list):
            detections_raw = detections_raw[0]
        if detections_raw.dim() == 3:
            detections_raw = detections_raw.squeeze(0)
        if detections_raw.numel() > 0:
            valid = (detections_raw[:, 4] > args.conf_thresh) & (
                (detections_raw[:, 2] - detections_raw[:, 0]) > 0
            )
            detections_raw = detections_raw[valid]
        if detections_raw.numel() > 0:
            from torchvision.ops import nms

            keep = nms(detections_raw[:, :4], detections_raw[:, 4], 0.5)
            detections_raw = detections_raw[keep]
        if detections_raw.numel() == 0:
            detections_raw = torch.zeros(0, 6, device=device)

        boxes = detections_raw[:, :4]
        scores = detections_raw[:, 4]
        class_ids = (
            detections_raw[:, 5].long()
            if detections_raw.shape[1] >= 6
            else torch.zeros(len(detections_raw), dtype=torch.long, device=device)
        )
        n_dets = boxes.shape[0]

        # Rescale boxes from 640 space to original resolution
        scale_x = float(w_orig) / args.img_size
        scale_y = float(h_orig) / args.img_size
        boxes_track = boxes.clone()
        boxes_track[:, 0] *= scale_x
        boxes_track[:, 1] *= scale_y
        boxes_track[:, 2] *= scale_x
        boxes_track[:, 3] *= scale_y

        # ── FPN embeddings ──
        embeddings = None
        if n_dets > 0 and not args.no_reid:
            if reid_head is not None:
                fpn = [
                    teacher._gate_layers[s]._feat_cache[s] for s in ("p3", "p4", "p5")
                ]
                if head_scales == "p5":
                    fpn = [fpn[2]]
                with torch.no_grad():
                    embeddings = fpn_reid_extract_cuda(
                        fpn,
                        conv_weights,
                        boxes,
                        args.img_size,
                        proj_weight=proj_weight,
                        running_mean=running_mean,
                        running_var=running_var,
                    )
            else:
                embeddings = detector.extract_fpn_embeddings(None, boxes)

            # ── Push bank to tracker (every frame) ──
        if bank_embeds and not args.no_reid and frame_idx % SYNC_INTERVAL == 0:
            _ids_list = sorted(bank_embeds.keys())
            _ids = torch.tensor(_ids_list, device=device, dtype=torch.int32)
            _feats = torch.stack([bank_embeds[k] for k in _ids_list])
            tracker.update_reference_features(_ids, _feats)
            _clean_ids = [k for k in _ids_list if bank_counts.get(k, 0) >= 2]
            if _clean_ids:
                _cids = torch.tensor(_clean_ids, device=device, dtype=torch.int32)
                tracker.set_clean_embedding_flags(
                    _cids,
                    torch.ones(len(_clean_ids), device=device, dtype=torch.bool),
                )

        # ── Track ──
        tracker.update_into(
            boxes_track,
            scores,
            class_ids,
            result_bufs,
            embeddings=embeddings,
        )

        # ── Update bank: store latest embedding per track ──
        count = result_bufs["count"].item()
        for i in range(count):
            x1 = result_bufs["boxes"][i, 0].item()
            y1 = result_bufs["boxes"][i, 1].item()
            x2 = result_bufs["boxes"][i, 2].item()
            y2 = result_bufs["boxes"][i, 3].item()
            tid = result_bufs["ids"][i].item()
            score = result_bufs["scores"][i].item()
            w, h = x2 - x1, y2 - y1
            mot_lines.append(
                f"{frame_idx + 1},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
            )
        active_tids: set[int] = set()
        if count > 0 and embeddings is not None and not args.no_reid:
            track_ids = result_bufs["ids"][:count]
            det_indices = result_bufs["det_idx"][:count]
            matched = (det_indices >= 0) & (det_indices < n_dets)
            active_tids.update(int(t) for t in track_ids.tolist())
            if matched.any():
                for tid, didx in zip(
                    track_ids[matched].tolist(), det_indices[matched].tolist()
                ):
                    new_emb = embeddings[int(didx)].detach()
                    if tid in bank_embeds:
                        bank_embeds[int(tid)] = F.normalize(
                            0.7 * bank_embeds[int(tid)] + 0.3 * new_emb, dim=0
                        )
                        bank_counts[int(tid)] += 1
                    else:
                        bank_embeds[int(tid)] = new_emb
                        bank_counts[int(tid)] = 1

        # Purge
        if bank_embeds:
            for tid in list(bank_embeds.keys()):
                if tid not in active_tids:
                    del bank_embeds[tid]
                    bank_counts.pop(tid, None)

        if frame_idx % 20 == 0 or frame_idx == len(frames) - 1:
            print(
                f"  frame {frame_idx:4d}: {n_dets} dets, "
                f"{count} tracks, bank={len(bank_embeds)}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed
    print(f"\nDone. {len(frames)} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_output = output_dir / f"{args.seq}.txt"
    seq_output.write_text("\n".join(mot_lines))
    print(f"Saved {len(mot_lines)} MOT lines to {seq_output}")

    try:
        metrics = run_motmetrics_evaluation(
            data_root=str(Path(args.data_root) / "MOT17"),
            split=args.split,
            output=str(output_dir),
            sequences=args.seq,
            detector=None,
        )
        if metrics:
            print("\n=== METRICS ===")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")


if __name__ == "__main__":
    main()
