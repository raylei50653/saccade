#!/usr/bin/env python3
"""
Eval script for GatedYOLODetector (Method B).

Inference loop:
  - frame t gate_input = predicted boxes from frame t-1 (self-conditioned)
  - NMS applied per-frame before writing MOT txt

Usage:
    uv run scripts/eval/eval_gated_detector.py \
        --ckpt runs/gated_det_v1/best.ckpt \
        --data-root datasets/MOT17 \
        --output /tmp/gated_det_v1_eval \
        [--sequences MOT17-02-SDP,...] \
        [--score-thr 0.3] [--nms-iou 0.5] [--no-gate]
"""
# status: archive-candidate

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
)

_SDP_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_frames(seq_dir: Path, img_size: int) -> list[tuple[int, torch.Tensor]]:
    import torchvision.io as tv_io

    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    result = []
    for p in paths:
        fid = int(p.stem)
        img = tv_io.read_image(str(p)).float() / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        result.append((fid, img))
    return result


def _orig_hw(seq_dir: Path) -> tuple[int, int]:
    ini = seq_dir / "seqinfo.ini"
    w, h = 1920, 1080
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth"):
                w = int(line.split("=")[1])
            elif line.startswith("imHeight"):
                h = int(line.split("=")[1])
    return h, w


def _nms_filter(
    boxes_xyxy: torch.Tensor,  # (300, 4)
    scores: torch.Tensor,  # (300,)
    score_thr: float,
    nms_iou: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = scores >= score_thr
    boxes_f, scores_f = boxes_xyxy[keep], scores[keep]
    if boxes_f.numel() == 0:
        return boxes_f.new_zeros((0, 4)), scores_f.new_zeros((0,))
    idx = torchvision.ops.nms(boxes_f, scores_f, nms_iou)
    return boxes_f[idx], scores_f[idx]


def _to_mot_lines(frame_id: int, xyxy: torch.Tensor, track_ids: list[int]) -> list[str]:
    lines = []
    for (x1, y1, x2, y2), tid in zip(xyxy.tolist(), track_ids):
        w, h = x2 - x1, y2 - y1
        lines.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1")
    return lines


# ---------------------------------------------------------------------------
# Simple IoU tracker (same as eval_conditioned.py)
# ---------------------------------------------------------------------------
class SimpleTracker:
    def __init__(self, iou_thr: float = 0.4, max_age: int = 1):
        self.next_id = 1
        self.tracks: list[dict] = []
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
            iou = torchvision.ops.box_iou(xyxy, track_boxes)
            matched_det: set[int] = set()
            matched_trk: set[int] = set()
            vals, cols = iou.max(dim=1)
            for di in vals.argsort(descending=True).tolist():
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
    model,
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

    tracker = SimpleTracker()
    prev_boxes_xyxy: torch.Tensor | None = None
    prev_scores: torch.Tensor | None = None
    sx, sy = orig_w / img_size, orig_h / img_size

    mot_lines: list[str] = []
    t0 = time.perf_counter()

    with torch.inference_mode():
        for frame_id, frame in frames:
            frame = frame.to(device)

            gate_input = None
            if (
                not disable_gate
                and prev_boxes_xyxy is not None
                and prev_boxes_xyxy.numel() > 0
            ):
                gate_input = TrackerGateInput.from_boxes_scores(
                    prev_boxes_xyxy,
                    prev_scores,
                    (img_size, img_size),
                    assume_absolute=True,
                ).to(device)

            # out[0]: (B, 300, 6) = (x1, y1, x2, y2, conf, cls) in img_size px space
            out = model(frame, gate_input=gate_input)
            raw = out[0][0]  # (300, 6)

            boxes_xyxy_640, kept_scores = _nms_filter(
                raw[:, :4], raw[:, 4], score_thr, nms_iou
            )
            prev_boxes_xyxy = boxes_xyxy_640
            prev_scores = kept_scores

            # Scale to original frame space for MOT output
            xyxy = boxes_xyxy_640.clone()
            if xyxy.numel() > 0:
                xyxy[:, [0, 2]] *= sx
                xyxy[:, [1, 3]] *= sy

            track_ids = tracker.update(xyxy)
            mot_lines.extend(_to_mot_lines(frame_id, xyxy, track_ids))

    elapsed = time.perf_counter() - t0
    fps = len(frames) / max(elapsed, 1e-6)
    (output_dir / f"{seq_name}.txt").write_text("\n".join(mot_lines))
    print(f"  {seq_name}: {len(frames)} frames, {fps:.1f} FPS, {len(mot_lines)} dets")
    return len(frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--output", default="/tmp/gated_det_v1_eval")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))

    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
        gate_min_score=train_args.get("gate_min_score", 0.5),
        freeze_backbone=False,
    )
    yolo_weights = project_root / train_args.get(
        "yolo_weights", "models/yolo/yolo26s.pt"
    )
    model = build_gated_yolo_detector(
        str(yolo_weights), cfg=cfg, device=device, weights_path=str(ckpt_path)
    )
    model.eval()
    print(
        f"Loaded: {ckpt_path}  epoch={raw.get('epoch')}  loss={raw.get('best_loss', 0):.4f}"
    )
    print(
        f"Gate {'DISABLED' if args.no_gate else 'ENABLED'}  score_thr={args.score_thr}"
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
            print(f"  [SKIP] {seq_dir}")
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

    try:
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
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")


if __name__ == "__main__":
    main()
