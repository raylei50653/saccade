import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class YOLOPoseTopKExport(nn.Module):
    """Export wrapper for YOLO Pose models.

    Output shape: [batch, max_det, 57]
      [:, :4]  xyxy boxes
      [:, 4]   confidence
      [:, 5]   class (always 0 – person-only model)
      [:, 6:]  keypoints, flat [17*3] = (x, y, visibility) per joint
    """

    def __init__(self, model: nn.Module, max_det: int = 300) -> None:
        super().__init__()
        self.layers = model.model
        self.save = set(model.save)
        self.head = self.layers[-1]
        if not hasattr(self.head, "kpt_shape"):
            raise ValueError("Not a Pose model: head is missing kpt_shape")
        self.nkpt = self.head.kpt_shape[0] * self.head.kpt_shape[1]
        self.head.export = True
        self.head.dynamic = False
        self.head.max_det = int(max_det)  # used by end2end postprocess
        self.max_det = int(max_det)
        self.end2end = getattr(self.head, "end2end", False)

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
        return [saved[j] for j in self.head.f]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self._run_backbone(images)

        if self.end2end:
            # Pose26 (end2end=True): head.forward() runs its own topk postprocess
            # and returns [batch, max_det, 57] with xyxy boxes directly.
            return self.head(feats)

        # YOLO11 non-end2end: head returns [batch, 4+nc+nkpt, N_anchors] in xywh.
        raw = self.head(feats)
        # Normalize to [batch, N_anchors, C] (channels-last)
        if raw.dim() == 3 and raw.shape[1] < raw.shape[2]:
            raw = raw.permute(0, 2, 1).contiguous()
        # Top-k by confidence (column 4)
        k = min(self.max_det, raw.shape[1])
        _, top_idx = raw[:, :, 4].topk(k, dim=1)
        top = raw.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, raw.shape[2]))

        # YOLO11 Detect._inference() returns xywh (cx,cy,w,h) when end2end=False.
        cxcy = top[:, :, :2]
        wh = top[:, :, 2:4]
        boxes_xyxy = torch.cat([cxcy - wh * 0.5, cxcy + wh * 0.5], dim=-1)

        # Insert explicit cls=0 column between conf and keypoints for format
        # compatibility with the detection pipeline: [box4, conf, cls, kpts51]
        cls = torch.zeros(*top.shape[:2], 1, device=raw.device, dtype=raw.dtype)
        return torch.cat([boxes_xyxy, top[:, :, 4:5], cls, top[:, :, 5:]], dim=-1)


def export_yolo_pose(
    weights: Path,
    output: Path,
    batch: int,
    imgsz: int,
    opset: int,
    max_det: int,
) -> None:
    yolo = YOLO(str(weights))
    model = yolo.model.eval()
    wrapper = YOLOPoseTopKExport(model, max_det=max_det).eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Export output shape: {tuple(out.shape)}")  # expect [batch, max_det, 57]
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
        dynamo=False,
        verbose=False,
    )
    print(f"Saved ONNX: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO Pose model with person top-k output."
    )
    parser.add_argument("--weights", default="models/yolo/yolo11n-pose.pt")
    parser.add_argument("--output", default="models/yolo/yolo11n_pose_topk300.onnx")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    export_yolo_pose(
        weights=Path(args.weights),
        output=Path(args.output),
        batch=args.batch,
        imgsz=args.imgsz,
        opset=args.opset,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
