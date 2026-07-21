#!/usr/bin/env python3
"""
Eval: YOLOX-X detector + our GPUByteTracker.

Loads YOLOX-X weights (99M, COCO pretrained or MOT17 fine-tuned),
runs person detection on MOT17-SDP, feeds into GPUByteTracker.
Lets us isolate detector quality vs tracker logic vs FastTracker.

Usage:
    uv run scripts/eval/eval_yolox_bytetrack.py \
        --weights models/yolo/yolox_x.pth \
        [--conf-threshold 0.01] [--nms-threshold 0.65]
"""
# status: experiment

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# MUST import tracker before torchvision to avoid libtiff symbol conflict on this system.
from saccade.perception.tracking.tracker_gpu import GPUByteTracker  # noqa: E402

# YOLOX via sys.path (no install needed) — imported AFTER tracker
sys.path.insert(0, "/tmp/YOLOX")
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

_SDP_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

# YOLOX-X architecture constants
_DEPTH, _WIDTH = 1.33, 1.25
_NUM_CLASSES = 80
_INPUT_SIZE = 800  # YOLOX standard test size; can override with --img-size


def build_yolox_x(num_classes: int = _NUM_CLASSES) -> YOLOX:
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(_DEPTH, _WIDTH, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, _WIDTH, in_channels=in_channels, act="silu")
    return YOLOX(backbone, head)


def load_yolox_x(weights_path: str, device: torch.device) -> YOLOX:
    model = build_yolox_x()
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def detect_frame(
    model: YOLOX,
    img_bgr: torch.Tensor,  # (3, H, W) float32 [0,1] RGB
    img_size: int,
    conf_thr: float,
    nms_thr: float,
    device: torch.device,
) -> torch.Tensor:
    """Returns (N, 6) tensor: x1,y1,x2,y2,obj_conf*cls_conf,cls_id  in orig coords."""
    orig_h, orig_w = img_bgr.shape[1], img_bgr.shape[2]

    # Letterbox resize to square
    r = min(img_size / orig_h, img_size / orig_w)
    new_h, new_w = int(orig_h * r), int(orig_w * r)
    resized = F.interpolate(
        img_bgr.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    # Pad to square, YOLOX expects [0,255] BGR - we use RGB, should be OK for eval
    padded = torch.zeros(1, 3, img_size, img_size, device=device)
    padded[0, :, :new_h, :new_w] = resized[0].to(device) * 255.0

    outputs = model(padded)  # raw YOLOX output
    dets = postprocess(outputs, _NUM_CLASSES, conf_thr, nms_thr, class_agnostic=True)

    if dets[0] is None:
        return torch.zeros((0, 6), device=device)

    d = dets[0]  # (N, 7): x1y1x2y2, obj_conf, cls_conf, cls_id
    # Scale back to original image coords
    d[:, :4] /= r
    # Merge scores
    scores = d[:, 4] * d[:, 5]
    cls_ids = d[:, 6]

    return torch.stack([d[:, 0], d[:, 1], d[:, 2], d[:, 3], scores, cls_ids], dim=1)


def _load_frames(seq_dir: Path, img_size: int):
    import torchvision.io as tv_io

    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    result = []
    for p in paths:
        fid = int(p.stem)
        img = tv_io.read_image(str(p)).float() / 255.0  # (3, H, W)
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


def eval_sequence(
    model: YOLOX,
    tracker: GPUByteTracker,
    seq_dir: Path,
    output_dir: Path,
    img_size: int,
    conf_thr: float,
    nms_thr: float,
    person_only: bool,
    device: torch.device,
) -> int:
    seq_name = seq_dir.name
    frames = _load_frames(seq_dir, img_size)
    orig_h, orig_w = _orig_hw(seq_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker.tracker.reset() if hasattr(tracker.tracker, "reset") else None
    tracker.set_frame_size(orig_w, orig_h)

    mot_lines: list[str] = []
    t0 = time.perf_counter()

    for frame_id, frame in frames:
        dets_all = detect_frame(model, frame, img_size, conf_thr, nms_thr, device)

        # Filter to person class (class 0 in COCO)
        if person_only and dets_all.numel() > 0:
            dets_all = dets_all[dets_all[:, 5] == 0]

        if dets_all.numel() > 0:
            boxes = dets_all[:, :4]
            scores = dets_all[:, 4]
            classes = dets_all[:, 5].to(torch.int32)
        else:
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            classes = torch.zeros((0,), device=device, dtype=torch.int32)

        track_results = tracker.update(boxes, scores, classes)

        for tr in track_results:
            w = tr.x2 - tr.x1
            h = tr.y2 - tr.y1
            mot_lines.append(
                f"{frame_id},{tr.obj_id},{tr.x1:.2f},{tr.y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1"
            )

    elapsed = time.perf_counter() - t0
    fps = len(frames) / max(elapsed, 1e-6)
    (output_dir / f"{seq_name}.txt").write_text("\n".join(mot_lines))
    print(f"  {seq_name}: {len(frames)} frames, {fps:.1f} FPS, {len(mot_lines)} tracks")
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolo/yolox_x.pth")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--output", default="/tmp/yolox_bytetrack")
    parser.add_argument("--img-size", type=int, default=800)
    parser.add_argument("--conf-threshold", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    parser.add_argument("--person-only", action="store_true", default=True)
    # ByteTrack params (same as our best baseline)
    parser.add_argument("--track-thresh", type=float, default=0.05)
    parser.add_argument("--high-thresh", type=float, default=0.45)
    parser.add_argument("--match-thresh", type=float, default=0.66)
    parser.add_argument("--new-track-thresh", type=float, default=0.28)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--fuse-score-weight", type=float, default=0.4)
    parser.add_argument("--confirm-streak", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading YOLOX-X from {args.weights} ...")
    model = load_yolox_x(args.weights, device)
    print(f"  params: 99.2M  input_size: {args.img_size}")

    tracker = GPUByteTracker(max_objects=2048, embedding_dim=768)
    tracker.set_params(
        track_thresh=args.track_thresh,
        high_thresh=args.high_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        new_track_thresh=args.new_track_thresh,
        fuse_score_weight=args.fuse_score_weight,
        confirm_streak=args.confirm_streak,
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
            tracker,
            seq_dir,
            output_dir,
            args.img_size,
            args.conf_threshold,
            args.nms_threshold,
            args.person_only,
            device,
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
