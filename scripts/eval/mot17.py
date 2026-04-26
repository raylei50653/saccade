#!/usr/bin/env python3
# mypy: ignore-errors
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from perception.detector_trt import TRTYoloDetector  # noqa: F401
from perception.eval.runner import run_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="models/yolo/yolo26s_batch4.engine")
    parser.add_argument("--output", default="results/MOT17_eval")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--track-thresh", type=float, default=0.1)
    parser.add_argument("--high-thresh", type=float, default=0.5)
    parser.add_argument("--mid-thresh", type=float, default=0.40)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--confirm-streak", type=int, default=3)
    parser.add_argument("--confirm-score-thresh", type=float, default=0.50)
    parser.add_argument("--adaptive-confirmation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometry-mid-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometry-ref-height-ratio", type=float, default=0.12)
    parser.add_argument("--geometry-min-scale", type=float, default=0.875)
    parser.add_argument("--geometry-max-scale", type=float, default=1.20)
    parser.add_argument("--geometry-ema-beta", type=float, default=0.80)
    parser.add_argument("--geometry-loosen-step", type=float, default=0.08)
    parser.add_argument("--geometry-tighten-step", type=float, default=0.03)
    parser.add_argument("--geometry-min-samples", type=int, default=5)
    parser.add_argument("--profile-lazy-reid-candidates", action="store_true")
    parser.add_argument("--profile-lazy-reid-embeddings", action="store_true")
    parser.add_argument("--lazy-reid-min-hit-streak", type=int, default=2)
    parser.add_argument("--lazy-reid-self-threshold", type=float, default=0.85)
    parser.add_argument("--reid-mode", choices=("off", "tracker", "semantic", "hybrid"), default="off")
    parser.add_argument("--reid-model", choices=("siglip2", "dinov2", "transreid"), default="siglip2")
    parser.add_argument("--reid-engine-path", default="")
    parser.add_argument("--reid-interval", type=int, default=4)
    parser.add_argument("--reid-cos-threshold", type=float, default=0.90)
    parser.add_argument("--reid-iou-low", type=float, default=0.30)
    parser.add_argument("--reid-iou-high", type=float, default=0.60)
    parser.add_argument("--reid-weight", type=float, default=0.40)
    parser.add_argument("--reid-crop-mode", choices=("tight", "square", "square_mean"), default="tight")
    parser.add_argument("--reid-crop-padding", type=float, default=0.0)
    parser.add_argument("--reid-crop-layout", choices=("full", "parts"), default="full")
    parser.add_argument("--semantic-threshold", type=float, default=0.95)
    parser.add_argument("--semantic-ttl", type=int, default=45)
    parser.add_argument("--semantic-ema", type=float, default=0.83)
    parser.add_argument("--semantic-spatial-gate", type=float, default=0.20)
    parser.add_argument("--semantic-min-lost-frames", type=int, default=2)
    parser.add_argument("--semantic-min-iou", type=float, default=0.20)
    parser.add_argument("--semantic-mahalanobis-threshold", type=float, default=0.0)
    parser.add_argument("--semantic-debug", action="store_true")
    parser.add_argument("--person-class", type=int, default=0)
    parser.add_argument("--track-person-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess", default="letterbox")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gamma-luma-threshold", type=float, default=0.35)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--cross-tile-merge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-frames", type=int, default=50)
    args = parser.parse_args()
    run_eval(**vars(args))
