"""
Export YOLO backbone ONNX from a GatedYOLODetector checkpoint.

Usage:
    uv run scripts/model/export_yolo_backbone_ckpt.py \
        --teacher-ckpt runs/gated_det_v1/best.ckpt \
        --output models/yolo/yolo26s_backbone_640_v1.onnx
"""

import argparse
import sys
from pathlib import Path

import saccade_tracking_ext  # noqa: F401  # must be before any torchvision import

import torch
from torch import nn
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


class YOLOBackboneExport(nn.Module):
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


def export_from_checkpoint(
    teacher_ckpt: Path, output: Path, imgsz: int, opset: int
) -> None:
    from saccade.perception.temporal_yolo.yolo_gated_detector import (
        build_gated_yolo_detector,
        GatedDetConfig,
    )

    device = "cuda"

    ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))

    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=train_args.get("gate_sigma_scale", 0.5),
        gate_min_score=train_args.get("gate_min_score", 0.5),
        freeze_backbone=True,
        img_size=imgsz,
    )

    yolo_weights = project_root / train_args.get(
        "yolo_weights", "models/yolo/yolo26s.pt"
    )
    teacher = build_gated_yolo_detector(
        str(yolo_weights),
        cfg=cfg,
        device=device,
        weights_path=str(teacher_ckpt),
    )
    teacher.eval()

    # Use the teacher's YOLO model (but layers are patched with gates)
    # We need ORIGINAL unpatched layers for TRT export.
    # Load a fresh YOLO and copy the teacher's weights into it.
    fresh_yolo = YOLO(str(yolo_weights)).model.to(device).eval()

    # Copy all parameters from teacher's YOLO to fresh YOLO
    teacher_sd = teacher.yolo_model.state_dict()
    fresh_yolo.load_state_dict(teacher_sd, strict=True)

    wrapper = YOLOBackboneExport(fresh_yolo).eval()

    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        p3, p4, p5 = wrapper(dummy)
    print(f"P3: {tuple(p3.shape)}  P4: {tuple(p4.shape)}  P5: {tuple(p5.shape)}")

    torch.onnx.export(
        wrapper.cpu(),
        dummy.cpu(),
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
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--output", default="models/yolo/yolo26s_backbone_640_v1.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=20)
    args = parser.parse_args()
    export_from_checkpoint(
        teacher_ckpt=Path(args.teacher_ckpt),
        output=Path(args.output),
        imgsz=args.imgsz,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
