"""Export TransReID-compatible ViT-B/16 (256×128) to ONNX for TRT compilation.

Architecture: timm vit_base_patch16_224, positional embeddings interpolated to
256×128 input (16×8 grid = 128 patches). ImageNet pretrained by default;
pass --checkpoint to load official TransReID fine-tuned weights.

Official TransReID weights (Market-1501 / MSMT17 / DukeMTMC) can be obtained
from: https://github.com/damo-cv/TransReID (Releases section).
Load with: python scripts/model/export_transreid.py --checkpoint path/to/model.pth

Usage:
    # ImageNet backbone only (still better than SigLIP for person ReID):
    uv run python scripts/model/export_transreid.py

    # With TransReID fine-tuned checkpoint:
    uv run python scripts/model/export_transreid.py --checkpoint path/to/transreid.pth

Then compile with trtexec (printed at end of script).
"""

import argparse
import os
import torch
import torch.nn as nn
import timm
from typing import cast
from pathlib import Path


class TransReIDWrapper(nn.Module):
    """ViT-B/16 backbone stripped to global CLS feature."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.backbone(pixel_values))  # [B, 768]


def export(
    checkpoint_path: str | None = None,
    output_dir: str = "models/embedding",
    base_model: str = "vit_base_patch16_224.orig_in21k_ft_in1k",
    img_h: int = 256,
    img_w: int = 128,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag = "transreid" if checkpoint_path else "vitb16_imagenet"
    onnx_path = os.path.join(output_dir, f"{tag}_{img_h}x{img_w}.onnx")

    if os.path.exists(onnx_path):
        print(f"ONNX already exists: {onnx_path}")
        return onnx_path

    # num_classes=0 removes classification head → raw CLS token [B, 768]
    # img_size=(H, W) causes timm to interpolate positional embeddings from
    # the standard 14×14 / 16×16 grid to the target H/16 × W/16 grid.
    pretrained = checkpoint_path is None
    print(f"Building {base_model} [{img_h}×{img_w}], pretrained={pretrained}...")
    backbone = timm.create_model(
        base_model,
        img_size=(img_h, img_w),
        pretrained=pretrained,
        num_classes=0,
    )

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # TransReID official checkpoint may use 'model' or 'state_dict' key.
        sd = state.get("model", state.get("state_dict", state))
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"Loaded checkpoint: {checkpoint_path}")

    model = TransReIDWrapper(backbone)
    model.eval().cpu()

    dummy = torch.randn(1, 3, img_h, img_w, dtype=torch.float32)

    print("Exporting to ONNX (dynamic batch)...")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch_size"}},
        verbose=False,
    )
    print(f"Exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to TransReID pretrained .pth checkpoint (optional)",
    )
    args = parser.parse_args()

    path = export(checkpoint_path=args.checkpoint)
    engine_path = path.replace(".onnx", ".engine")
    print(
        f"\nNext step — compile with trtexec:\n"
        f"trtexec --onnx={path} --saveEngine={engine_path} "
        f"--fp16 "
        f"--minShapes=pixel_values:1x3x256x128 "
        f"--optShapes=pixel_values:8x3x256x128 "
        f"--maxShapes=pixel_values:32x3x256x128"
    )
