#!/usr/bin/env python
"""Export fine-tuned MobileNetV4 ReID checkpoint to ONNX for TRT compilation.

The exported graph expects already-normalized ImageNet pixel values, matching
TRTFeatureExtractor's input contract. It outputs pre-L2 BNNeck embeddings as
`image_embeds`; TRTFeatureExtractor applies the final L2 normalization.

Usage:
    uv run python scripts/model/export_mobilenetv4_reid.py
    uv run python scripts/model/build_reid.py \
        --onnx models/embedding/mobilenetv4_reid_visclean_224.onnx \
        --engine models/embedding/mobilenetv4_reid_visclean_224.engine \
        --input-hw 224 224
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MobileNetV4ReIDWrapper(nn.Module):
    """Fine-tuned backbone + BNNeck, with normalization handled upstream."""

    def __init__(self, backbone: nn.Module, bnneck: nn.BatchNorm1d) -> None:
        super().__init__()
        self.backbone = backbone
        self.bnneck = bnneck

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = cast(torch.Tensor, self.backbone(pixel_values))
        return self.bnneck(feat)


def _load_model(checkpoint_path: Path) -> tuple[nn.Module, tuple[int, int]]:
    import timm

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = str(ckpt["arch"])
    input_hw = tuple(int(v) for v in ckpt["input_hw"])
    mean = tuple(float(v) for v in ckpt["mean"])
    std = tuple(float(v) for v in ckpt["std"])
    if mean != _IMAGENET_MEAN or std != _IMAGENET_STD:
        raise ValueError(
            f"{checkpoint_path} uses mean/std {mean}/{std}, but TRTFeatureExtractor "
            f"normalizes mobilenetv4_reid with ImageNet {_IMAGENET_MEAN}/{_IMAGENET_STD}"
        )

    backbone = timm.create_model(arch, pretrained=False, num_classes=0)
    backbone.load_state_dict(ckpt["backbone"], strict=True)

    dim = int(ckpt["bnneck"]["weight"].shape[0])
    bnneck = nn.BatchNorm1d(dim)
    bnneck.load_state_dict(ckpt["bnneck"], strict=True)

    model = MobileNetV4ReIDWrapper(backbone, bnneck)
    model.eval().cpu()
    return model, cast(tuple[int, int], input_hw)


def export(
    checkpoint: str = "runs/reid_mnv4_ft_visclean/best.ckpt",
    output: str = "models/embedding/mobilenetv4_reid_visclean_224.onnx",
    force: bool = False,
) -> str:
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"ONNX already exists: {output_path}")
        return str(output_path)

    model, input_hw = _load_model(checkpoint_path)
    img_h, img_w = input_hw
    dummy = torch.randn(1, 3, img_h, img_w, dtype=torch.float32)

    with torch.no_grad():
        probe = model(dummy)
    print(
        f"Exporting MobileNetV4 ReID: checkpoint={checkpoint_path} "
        f"input={img_h}x{img_w} dim={int(probe.shape[-1])}"
    )

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        verbose=False,
    )
    print(f"Exported: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/reid_mnv4_ft_visclean/best.ckpt")
    parser.add_argument(
        "--output", default="models/embedding/mobilenetv4_reid_visclean_224.onnx"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    onnx_path = export(args.checkpoint, args.output, args.force)
    engine_path = str(Path(onnx_path).with_suffix(".engine"))
    print(
        "\nNext step:\n"
        "uv run python scripts/model/build_reid.py "
        f"--onnx {onnx_path} --engine {engine_path} --input-hw 224 224"
    )
