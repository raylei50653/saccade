#!/usr/bin/env python3
"""
Eval script for TemporalYOLOConditioned (Option D, Phase 2).

Inference loop:
  - frame t gate_input = predicted boxes from frame t-1 (self-conditioned)
  - prev_queries passed across frames within each sequence
  - NMS applied per-frame before writing MOT txt

Usage:
    uv run scripts/eval/baselines/eval_conditioned.py \
        --ckpt runs/conditioned_p2/best.ckpt \
        --data-root datasets/MOT17 \
        --output /tmp/conditioned_p2 \
        [--sequences MOT17-02-SDP,MOT17-04-SDP,...] \
        [--score-thr 0.3] [--nms-iou 0.5] [--img-size 640]
"""
# status: stable

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.yolo_conditioned import (  # noqa: E402
    ConditionedConfig,
    TrackerGateInput,
    build_temporal_yolo_conditioned,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SDP_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


def _load_frames(seq_dir: Path, img_size: int) -> list[tuple[int, torch.Tensor]]:
    """Load and resize all frames of a sequence. Returns list of (frame_id, tensor)."""
    import torchvision.io as tv_io

    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    result = []
    for p in paths:
        fid = int(p.stem)
        img = tv_io.read_image(str(p)).float() / 255.0  # (3, H, W)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )  # (1, 3, S, S)
        result.append((fid, img))
    return result


def _apply_nms(
    boxes_norm: torch.Tensor,  # (N, 4) cxcywh normalised
    scores: torch.Tensor,  # (N,)
    score_thr: float,
    nms_iou: float,
    img_h: int,
    img_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter by score + NMS. Returns (boxes_xyxy_abs, kept_scores)."""
    keep = scores >= score_thr
    boxes_f = boxes_norm[keep]
    scores_f = scores[keep]
    if boxes_f.numel() == 0:
        return boxes_f.new_zeros((0, 4)), scores_f.new_zeros((0,))

    # cxcywh norm → xyxy abs
    cx, cy, bw, bh = boxes_f[:, 0], boxes_f[:, 1], boxes_f[:, 2], boxes_f[:, 3]
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    nms_idx = torchvision.ops.nms(xyxy, scores_f, nms_iou)
    return xyxy[nms_idx], scores_f[nms_idx]


def _to_mot_lines(
    frame_id: int,
    xyxy: torch.Tensor,  # (N, 4) absolute pixel coords
    track_ids: list[int],
) -> list[str]:
    lines = []
    for (x1, y1, x2, y2), tid in zip(xyxy.tolist(), track_ids):
        w, h = x2 - x1, y2 - y1
        lines.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1")
    return lines


# ---------------------------------------------------------------------------
# Simple track-ID manager (assigns IDs to detections via greedy IoU matching)
# ---------------------------------------------------------------------------
class SimpleTracker:
    """Assigns persistent IDs by greedy IoU matching across frames."""

    def __init__(self, iou_thr: float = 0.4, max_age: int = 1):
        self.next_id = 1
        self.tracks: list[dict] = []  # {id, xyxy, age}
        self.iou_thr = iou_thr
        self.max_age = max_age

    def update(self, xyxy: torch.Tensor) -> list[int]:
        if xyxy.numel() == 0:
            for t in self.tracks:
                t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return []

        n_det = xyxy.shape[0]
        assigned = [-1] * n_det

        if self.tracks:
            track_boxes = torch.stack([t["xyxy"] for t in self.tracks]).to(xyxy.device)
            iou = torchvision.ops.box_iou(xyxy, track_boxes)  # (D, T)
            matched_det: set[int] = set()
            matched_trk: set[int] = set()
            vals, cols = iou.max(dim=1)
            order = vals.argsort(descending=True)
            for di in order.tolist():
                ti = int(cols[di])
                if (
                    float(vals[di]) >= self.iou_thr
                    and di not in matched_det
                    and ti not in matched_trk
                ):
                    assigned[di] = self.tracks[ti]["id"]
                    self.tracks[ti]["xyxy"] = xyxy[di]
                    self.tracks[ti]["age"] = 0
                    matched_det.add(di)
                    matched_trk.add(ti)
            for ti, t in enumerate(self.tracks):
                if ti not in matched_trk:
                    t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        ids: list[int] = []
        for di in range(n_det):
            if assigned[di] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append({"id": tid, "xyxy": xyxy[di], "age": 0})
                ids.append(tid)
            else:
                ids.append(assigned[di])
        return ids

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1


# ---------------------------------------------------------------------------
# Per-sequence eval
# ---------------------------------------------------------------------------
def eval_sequence(
    model: "TemporalYOLOConditioned",  # type: ignore[name-defined]  # noqa: F821
    seq_dir: Path,
    output_dir: Path,
    img_size: int,
    score_thr: float,
    nms_iou: float,
    device: torch.device,
    disable_gate: bool = False,
) -> int:
    seq_name = seq_dir.name
    frames = _load_frames(seq_dir, img_size)
    orig_h, orig_w = _orig_hw(seq_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{seq_name}.txt"

    tracker = SimpleTracker()
    model.reset_sequence()
    prev_queries: torch.Tensor | None = None
    prev_boxes_xyxy: torch.Tensor | None = None  # absolute for gate
    prev_scores: torch.Tensor | None = None

    mot_lines: list[str] = []
    t0 = time.perf_counter()
    sx, sy = orig_w / img_size, orig_h / img_size

    with torch.inference_mode():
        for frame_id, frame in frames:
            frame = frame.to(device)

            # Build gate_input from previous frame's predictions (img_size space)
            gate_input = None
            if (
                not disable_gate
                and prev_boxes_xyxy is not None
                and prev_boxes_xyxy.numel() > 0
            ):
                gate_input = TrackerGateInput.from_boxes_scores(
                    prev_boxes_xyxy.to(device),
                    prev_scores.to(device) if prev_scores is not None else None,
                    (img_size, img_size),
                    assume_absolute=True,
                ).to(device)

            outputs = model(frame, prev_queries=prev_queries, gate_input=gate_input)
            prev_queries = outputs["updated_queries"].detach()

            boxes_norm = outputs["boxes"][0].float()  # (N_q, 4) cxcywh norm
            scores = outputs["scores"][0].float()  # (N_q,)

            # NMS in img_size space; keep img_size coords for gate_input next frame
            xyxy_640, kept_scores = _apply_nms(
                boxes_norm, scores, score_thr, nms_iou, img_size, img_size
            )
            prev_boxes_xyxy = xyxy_640  # img_size space for gate
            prev_scores = kept_scores

            # Scale to original frame space for MOT output
            xyxy = xyxy_640.clone()
            if xyxy.numel() > 0:
                xyxy[:, [0, 2]] *= sx
                xyxy[:, [1, 3]] *= sy

            track_ids = tracker.update(xyxy)
            mot_lines.extend(_to_mot_lines(frame_id, xyxy, track_ids))

    elapsed = time.perf_counter() - t0
    fps = len(frames) / max(elapsed, 1e-6)
    result_file.write_text("\n".join(mot_lines))
    print(
        f"  {seq_name}: {len(frames)} frames, {fps:.1f} FPS, {len(mot_lines)} dets written"
    )
    return len(frames)


def _orig_hw(seq_dir: Path) -> tuple[int, int]:
    """Read original frame size from seqinfo.ini."""
    ini = seq_dir / "seqinfo.ini"
    w, h = 1920, 1080
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth"):
                w = int(line.split("=")[1])
            elif line.startswith("imHeight"):
                h = int(line.split("=")[1])
    return h, w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval TemporalYOLOConditioned on MOT17"
    )
    parser.add_argument("--ckpt", default="runs/conditioned_p2/best.ckpt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--sequences",
        default="",
        help="Comma-separated sequence names; empty = all SDP train seqs",
    )
    parser.add_argument("--output", default="/tmp/conditioned_p2_eval")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable spatial gate (gate_input=None every frame) for ablation",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    # Load checkpoint metadata to reconstruct config
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))

    cfg = ConditionedConfig(
        embed_dim=256,
        num_heads=8,
        num_queries=train_args.get("num_queries", 100),
        num_decoder_layers=train_args.get("num_decoder_layers", 3),
        ffn_dim=1024,
        score_threshold=args.score_thr,
        scales=scales,
        gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
        gate_min_score=train_args.get("gate_min_score", 0.5),
        freeze_backbone=False,
        freeze_decoder=False,
    )

    yolo_weights = project_root / train_args.get(
        "yolo_weights", "models/yolo/yolo26s.pt"
    )
    model = build_temporal_yolo_conditioned(
        str(yolo_weights), cfg=cfg, device=device, weights_path=str(ckpt_path)
    )
    model.eval()
    print(
        f"Loaded checkpoint: {ckpt_path}  (epoch {raw.get('epoch')}, loss {raw.get('best_loss', 0):.4f})"
    )

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()] or _SDP_SEQS
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    output_dir = Path(args.output)

    print(f"\nEvaluating {len(seqs)} sequences → {output_dir}")
    for seq in seqs:
        seq_dir = data_root / args.split / seq
        if not seq_dir.exists():
            print(f"  [SKIP] {seq_dir} not found")
            continue
        eval_sequence(
            model,
            seq_dir,
            output_dir,
            args.img_size,
            args.score_thr,
            args.nms_iou,
            device,
            disable_gate=args.no_gate,
        )

    # Compute motmetrics
    try:
        sys.path.insert(0, str(project_root / "src"))
        from saccade.perception.eval.metrics import run_motmetrics_evaluation

        metrics = run_motmetrics_evaluation(
            data_root=str(data_root),
            split=args.split,
            output=str(output_dir),
            sequences=",".join(seqs),
            detector="SDP",
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print("\n[Warn] motmetrics returned None — check output dir and GT paths")
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")


if __name__ == "__main__":
    main()
