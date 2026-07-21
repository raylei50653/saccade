"""Export YOLO detect model with person class-filtered top-k outputs."""

# status: stable
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class YOLOPersonTopKExport(nn.Module):
    """Export wrapper that keeps only selected classes before top-k postprocess."""

    def __init__(
        self,
        model: nn.Module,
        class_ids: list[int],
        max_det: int,
        disable_end2end: bool,
        output_box_format: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.layers = model.model
        self.save = set(model.save)
        self.head = self.layers[-1]
        self.head.export = True
        self.head.dynamic = False
        self.disable_end2end = bool(disable_end2end)
        self.output_box_format = output_box_format
        if disable_end2end:
            self.head.end2end = False
        self.max_det = int(max_det)
        self.num_classes = int(getattr(self.head, "nc", 0))
        if self.num_classes <= 0:
            raise ValueError("Detect head is missing a valid class count")
        if class_ids:
            keep_ids = torch.tensor(class_ids, dtype=torch.long)
        else:
            keep_ids = torch.arange(self.num_classes, dtype=torch.long)
        self.register_buffer("class_ids", keep_ids, persistent=False)

    def _run_backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        saved: list[torch.Tensor | None] = []
        for layer in self.layers[:-1]:
            if layer.f != -1:
                x_in = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if j == -1 else saved[j] for j in layer.f]
                )
            else:
                x_in = x
            x = layer(x_in)
            saved.append(x if layer.i in self.save else None)

        head_from = self.head.f
        return [saved[j] for j in head_from]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self._run_backbone(images)
        preds = self.head.forward_head(
            feats,
            **(self.head.one2one if self.head.end2end else self.head.one2many),
        )
        boxes = self.head._get_decode_boxes(preds).permute(0, 2, 1)
        scores = preds["scores"].sigmoid().permute(0, 2, 1)
        scores = scores.index_select(dim=-1, index=self.class_ids)
        top_scores, top_classes, top_idx = self.head.get_topk_index(
            scores, self.max_det
        )
        top_boxes = boxes.gather(dim=1, index=top_idx.repeat(1, 1, 4))
        if self.disable_end2end and self.output_box_format == "xyxy":
            cxcy = top_boxes[:, :, :2]
            wh = top_boxes[:, :, 2:4]
            top_boxes = torch.cat([cxcy - wh * 0.5, cxcy + wh * 0.5], dim=-1)
        return torch.cat([top_boxes, top_scores, top_classes], dim=-1)


def export_yolo_person(
    weights: Path,
    output: Path,
    batch: int,
    imgsz: int,
    opset: int,
    class_ids: list[int],
    max_det: int,
    disable_end2end: bool,
    output_box_format: str,
) -> None:
    yolo = YOLO(str(weights))
    model = yolo.model.eval()
    wrapper = YOLOPersonTopKExport(
        model,
        class_ids=class_ids,
        max_det=max_det,
        disable_end2end=disable_end2end,
        output_box_format=output_box_format,
    ).eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        detections = wrapper(dummy)
    print("Export output:", tuple(detections.shape))

    torch.onnx.export(
        wrapper,
        dummy,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch"},
            "output0": {0: "batch"},
        },
        dynamo=False,
        verbose=False,
    )
    print(f"Saved ONNX: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO detect model with class-filtered top-k output."
    )
    parser.add_argument("--weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--output", default="models/yolo/yolo26s_person_topk1000.onnx")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument(
        "--disable-end2end",
        action="store_true",
        help="Force the detect head to export through its non-end2end one-to-many path.",
    )
    parser.add_argument(
        "--output-box-format",
        choices=["xyxy", "cxcywh"],
        default="xyxy",
        help="Box layout written to the exported ONNX output tensor.",
    )
    parser.add_argument(
        "--class-ids",
        default="",
        help=(
            "Comma-separated class ids to keep before top-k. "
            "Empty keeps all classes; use 0 for person-only."
        ),
    )
    args = parser.parse_args()
    class_ids = [int(item) for item in args.class_ids.split(",") if item.strip()]

    export_yolo_person(
        weights=Path(args.weights),
        output=Path(args.output),
        batch=args.batch,
        imgsz=args.imgsz,
        opset=args.opset,
        class_ids=class_ids,
        max_det=args.max_det,
        disable_end2end=args.disable_end2end,
        output_box_format=args.output_box_format,
    )


if __name__ == "__main__":
    main()
