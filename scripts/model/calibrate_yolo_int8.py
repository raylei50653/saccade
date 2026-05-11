#!/usr/bin/env python3
"""
INT8 static quantization for YOLO ONNX via onnxruntime.
Produces a QDQ-annotated ONNX that TRT 10.x (explicit quantization) can build.

Usage:
    uv run python scripts/model/calibrate_yolo_int8.py \
        --onnx models/yolo/yolo26m.onnx \
        --output models/yolo/yolo26m_int8_qdq.onnx \
        --img-size 960 \
        --num-images 512
"""
import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image


class YoloCalibrationReader(CalibrationDataReader):
    def __init__(self, img_paths: list[Path], img_size: int, input_name: str):
        self.img_paths = img_paths
        self.img_size = img_size
        self.input_name = input_name
        self._iter = iter(self.img_paths)

    def get_next(self):
        try:
            p = next(self._iter)
        except StopIteration:
            return None
        img = Image.open(p).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC
        arr = arr.transpose(2, 0, 1)[None]             # 1CHW
        return {self.input_name: arr}


def collect_images(data_root: Path, num_images: int, seed: int = 42) -> list[Path]:
    imgs = sorted(data_root.rglob("*.jpg")) + sorted(data_root.rglob("*.png"))
    rng = random.Random(seed)
    if len(imgs) > num_images:
        imgs = rng.sample(imgs, num_images)
    return imgs


def get_input_name(onnx_path: str) -> str:
    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="datasets/MOT17/train")
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--num-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Collecting {args.num_images} calibration images from {args.data_root}...")
    imgs = collect_images(Path(args.data_root), args.num_images, args.seed)
    print(f"  Found {len(imgs)} images")

    input_name = get_input_name(args.onnx)
    print(f"  Model input: {input_name!r}  size: {args.img_size}x{args.img_size}")

    reader = YoloCalibrationReader(imgs, args.img_size, input_name)

    print("Running INT8 static calibration (QDQ format)...")
    quantize_static(
        model_input=args.onnx,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    print(f"QDQ ONNX saved → {args.output}")


if __name__ == "__main__":
    main()
