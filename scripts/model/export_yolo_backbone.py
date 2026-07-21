"""
Export YOLO backbone (layers 0-22) to ONNX with 3 FPN feature outputs.

Outputs: P3 (layer 16), P4 (layer 19), P5 (layer 22) — same features fed to Detect head.
This excludes the Detect head (layer 23), leaving gating + detection to PyTorch.

Usage:
    uv run scripts/model/export_yolo_backbone.py \
        --weights models/yolo/yolo26s.pt \
        --output models/yolo/yolo26s_backbone_640.onnx
"""
# status: stable

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class YOLOBackboneExport(nn.Module):
    """Runs YOLO layers 0-22, returns P3/P4/P5 FPN features."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = model.model
        self.save_set = set(model.save)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_list: list[torch.Tensor | None] = []
        x: torch.Tensor = images

        for i in range(23):
            m = self.layers[i]
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y_list[m.f]
                else:
                    x = [x if j == -1 else y_list[j] for j in m.f]
            x = m(x)
            y_list.append(x if i in self.save_set else None)

        return y_list[16], y_list[19], y_list[22]  # type: ignore[index]


def export(
    weights: Path,
    output: Path,
    batch: int,
    imgsz: int,
    opset: int,
) -> None:
    yolo = YOLO(str(weights))
    model = yolo.model.eval()
    wrapper = YOLOBackboneExport(model).eval()

    dummy = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        p3, p4, p5 = wrapper(dummy)
    print(f"P3: {tuple(p3.shape)}  P4: {tuple(p4.shape)}  P5: {tuple(p5.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["p3_feat", "p4_feat", "p5_feat"],
        dynamic_axes={
            "images": {0: "batch"},
            "p3_feat": {0: "batch"},
            "p4_feat": {0: "batch"},
            "p5_feat": {0: "batch"},
        },
        dynamo=False,
        verbose=False,
    )
    print(f"Saved ONNX: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--output", default="models/yolo/yolo26s_backbone_640.onnx")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=20)
    args = parser.parse_args()

    export(
        weights=Path(args.weights),
        output=Path(args.output),
        batch=args.batch,
        imgsz=args.imgsz,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
