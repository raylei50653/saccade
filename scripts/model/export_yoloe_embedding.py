import argparse
import sys
from pathlib import Path
from typing import List

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import LRPCHead

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class YOLOEEmbeddingExport(nn.Module):
    """Export wrapper that exposes YOLOE top-k detection embeddings.

    Ultralytics' default YOLOE segmentation export returns detections and mask
    prototypes. This wrapper follows the same head path, then gathers the
    per-anchor classification feature with the exact same top-k detection index.
    """

    def __init__(self, model: nn.Module, class_ids: list[int] | None = None) -> None:
        super().__init__()
        self.model = model
        self.layers = model.model
        self.save = set(model.save)
        self.head = self.layers[-1]
        self.head.export = True
        self.head.dynamic = False
        if class_ids:
            self.register_buffer(
                "class_ids", torch.tensor(class_ids, dtype=torch.long), persistent=False
            )
            self.num_export_classes = len(class_ids)
        else:
            self.class_ids = None
            self.num_export_classes = self.head.nc

    def _run_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        saved: List[torch.Tensor | None] = []
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

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self._run_backbone(images)
        head = self.head
        bs = feats[0].shape[0]

        boxes, scores, cls_embeddings = [], [], []
        cv2 = head.one2one_cv2 if head.end2end else head.cv2
        cv3 = head.one2one_cv3 if head.end2end else head.cv3
        cv5 = head.one2one_cv5 if head.end2end else head.cv5

        for i in range(head.nl):
            cls_feat = cv3[i](feats[i])
            loc_feat = cv2[i](feats[i])
            assert isinstance(head.lrpc[i], LRPCHead)

            # conf=0 preserves all anchors, which keeps the exported tensor shape
            # fixed and lets us align embeddings with the same postprocess top-k.
            box, score, _ = head.lrpc[i](cls_feat, loc_feat, 0)
            if self.class_ids is not None:
                score = score.index_select(1, self.class_ids)
            boxes.append(box.view(bs, head.reg_max * 4, -1))
            scores.append(score)
            cls_embeddings.append(cls_feat.flatten(2))

        mask_coeff = torch.cat(
            [cv5[i](feats[i]).view(bs, head.nm, -1) for i in range(head.nl)], 2
        )
        anchor_embeddings = torch.cat(cls_embeddings, 2).transpose(1, 2)

        preds = {
            "boxes": torch.cat(boxes, 2),
            "scores": torch.cat(scores, 2),
            "feats": feats,
            "mask_coefficient": mask_coeff,
        }
        dbox = head._get_decode_boxes(preds)
        decoded = torch.cat([dbox, preds["scores"].sigmoid(), mask_coeff], dim=1).permute(
            0, 2, 1
        )

        raw_boxes, raw_scores, raw_mask_coeff = decoded.split(
            [4, self.num_export_classes, head.nm], dim=-1
        )
        top_scores, top_classes, top_idx = head.get_topk_index(
            raw_scores, head.max_det
        )

        top_boxes = raw_boxes.gather(dim=1, index=top_idx.repeat(1, 1, 4))
        top_masks = raw_mask_coeff.gather(dim=1, index=top_idx.repeat(1, 1, head.nm))
        top_embeddings = anchor_embeddings.gather(
            dim=1, index=top_idx.repeat(1, 1, anchor_embeddings.shape[-1])
        )
        detections = torch.cat([top_boxes, top_scores, top_classes, top_masks], dim=-1)
        proto = head.proto([feat.detach() for feat in feats], return_semseg=False)

        return detections, proto, top_embeddings


def export_yoloe_embedding(
    weights: Path,
    output: Path,
    batch: int,
    imgsz: int,
    opset: int,
    class_ids: list[int] | None,
) -> None:
    yolo = YOLO(str(weights))
    model = yolo.model.eval()
    wrapper = YOLOEEmbeddingExport(model, class_ids=class_ids).eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        detections, proto, embeddings = wrapper(dummy)
    print(
        "Export outputs:",
        tuple(detections.shape),
        tuple(proto.shape),
        tuple(embeddings.shape),
    )

    torch.onnx.export(
        wrapper,
        dummy,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["detections", "proto", "embeddings"],
        dynamic_axes={
            "images": {0: "batch"},
            "detections": {0: "batch"},
            "proto": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        dynamo=False,
        verbose=False,
    )
    print(f"Saved ONNX: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOE with top-k embeddings.")
    parser.add_argument("--weights", default="models/yolo/yoloe-26m-seg-pf.pt")
    parser.add_argument("--output", default="models/yolo/yoloe-26m-seg-pf_embed.onnx")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--class-ids",
        default="",
        help="Comma-separated prompt-free vocab class ids to export. Output class ids are remapped to 0..N-1.",
    )
    args = parser.parse_args()
    class_ids = [int(item) for item in args.class_ids.split(",") if item.strip()]

    export_yoloe_embedding(
        weights=Path(args.weights),
        output=Path(args.output),
        batch=args.batch,
        imgsz=args.imgsz,
        opset=args.opset,
        class_ids=class_ids or None,
    )


if __name__ == "__main__":
    main()
