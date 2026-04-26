"""Export facebook/dinov2-base (ViT-B/14, 768-dim) to ONNX for TRT compilation.

Usage:
    uv run python scripts/model/export_dinov2.py

Then compile with trtexec (printed at end of script).
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path


class DINOv2Wrapper(nn.Module):
    """Strips the DINOv2 model to CLS token output only."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # CLS token → [B, 768]


def export(
    model_name: str = "facebook/dinov2-base",
    output_dir: str = "models/embedding",
    img_size: int = 224,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

    if os.path.exists(onnx_path):
        print(f"ONNX already exists: {onnx_path}")
        return onnx_path

    print(f"Loading {model_name}...")
    base = AutoModel.from_pretrained(model_name)
    model = DINOv2Wrapper(base)
    model.eval().cpu()

    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

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
    path = export()
    engine_path = path.replace(".onnx", ".engine")
    s = f"224x{224}"
    print(
        f"\nNext step — compile with trtexec:\n"
        f"trtexec --onnx={path} --saveEngine={engine_path} "
        f"--fp16 "
        f"--minShapes=pixel_values:1x3x224x224 "
        f"--optShapes=pixel_values:8x3x224x224 "
        f"--maxShapes=pixel_values:32x3x224x224"
    )
