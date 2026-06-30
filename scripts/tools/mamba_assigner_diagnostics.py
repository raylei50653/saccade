"""Measure real TaskAlignedAssigner capacity for the v14 Mamba detector."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa: E402
from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead  # noqa: E402
from saccade.perception.temporal_yolo.ngla_assigner import install_assigner  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    _GATE_LAYER_IDX,
    build_gated_yolo_detector,
)
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402


def _strip_compiled_keys(sd: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def _make_yolo_batch(
    gt_boxes: list[torch.Tensor], img_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    batch_idx: list[torch.Tensor] = []
    boxes_norm: list[torch.Tensor] = []
    for index, boxes in enumerate(gt_boxes):
        if boxes.numel() == 0:
            continue
        xyxy = boxes.to(device)
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5 / img_size
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5 / img_size
        w = (xyxy[:, 2] - xyxy[:, 0]) / img_size
        h = (xyxy[:, 3] - xyxy[:, 1]) / img_size
        boxes_norm.append(torch.stack((cx, cy, w, h), dim=1))
        batch_idx.append(
            torch.full((xyxy.shape[0],), index, device=device, dtype=torch.float32)
        )
    if not batch_idx:
        return {
            "batch_idx": torch.empty(0, device=device),
            "cls": torch.empty(0, device=device),
            "bboxes": torch.empty(0, 4, device=device),
        }
    count = sum(item.shape[0] for item in batch_idx)
    return {
        "batch_idx": torch.cat(batch_idx),
        "cls": torch.zeros(count, device=device),
        "bboxes": torch.cat(boxes_norm),
    }


def _size_bin(min_side: float) -> str:
    if min_side < 4:
        return "lt4"
    if min_side < 8:
        return "4to8"
    if min_side < 16:
        return "8to16"
    return "ge16"


def _increment(stats: dict[str, float], key: str, value: float = 1.0) -> None:
    stats[key] += value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_gt_vgt_mamba_v14/best.ckpt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seqs", default="")
    parser.add_argument(
        "--assigner",
        choices=("tal", "ngla"),
        default="tal",
        help="Label assignment variant used for the diagnostic probe.",
    )
    parser.add_argument(
        "--output",
        default="report_data/mamba_assigner_diagnostics.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_state = torch.load(
        project_root / args.teacher_ckpt, map_location="cpu", weights_only=False
    )
    teacher_args = teacher_state.get("args", {})
    cfg = GatedDetConfig(
        scales=("p3", "p4", "p5"),
        gate_sigma_scale=teacher_args.get("gate_sigma_scale", 0.5),
        gate_min_score=teacher_args.get("gate_min_score", 0.5),
        freeze_backbone=True,
        img_size=args.img_size,
    )
    teacher = build_gated_yolo_detector(
        str(project_root / args.yolo_weights),
        cfg=cfg,
        device=device,
        weights_path=str(project_root / args.teacher_ckpt),
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    checkpoint = torch.load(
        project_root / args.mamba_ckpt, map_location="cpu", weights_only=False
    )
    model_args = checkpoint["mamba_args"]
    head = MambaDetectionHead(
        in_channels=(128, 256, 512),
        d_model=model_args["d_model"],
        d_state=model_args["d_state"],
        num_blocks=model_args["num_blocks"],
        num_classes=model_args["num_classes"],
        reg_max=model_args.get("reg_max", 1),
        spatial_reduction=model_args["spatial_reduction"],
        use_pixel_shuffle=model_args.get("use_pixel_shuffle", False),
        use_cross_scan=model_args.get("use_cross_scan", False),
        use_hybrid_head=model_args.get("use_hybrid_head", False),
        per_channel_a=model_args.get("per_channel_a", False),
        reduction_variant=model_args.get("reduction_variant", "conv"),
    ).to(device)
    head.load_state_dict(_strip_compiled_keys(checkpoint["student"]), strict=False)
    head.eval()

    base_args = (
        dict(teacher.yolo_model.args)
        if isinstance(teacher.yolo_model.args, dict)
        else {}
    )
    base_args.setdefault("box", 7.5)
    base_args.setdefault("cls", 0.5)
    base_args.setdefault("dfl", 1.5)
    teacher.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(teacher.yolo_model)
    install_assigner(criterion, args.assigner)

    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for scale in ("p3", "p4", "p5"):
        index = _GATE_LAYER_IDX[scale]

        def _capture(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
            name: str = scale,
        ) -> None:
            captured[name] = output

        hooks.append(teacher.yolo_model.model[index].register_forward_hook(_capture))

    sequences = args.seqs.split(",") if args.seqs else None
    loader = build_mot17_dataloader(
        data_root=project_root / args.data_root,
        clip_len=1,
        img_size=args.img_size,
        batch_size=args.batch_size,
        stride=1,
        shuffle=False,
        seqs=sequences,
        preload_to_ram=False,
    )

    stats: dict[str, float] = defaultdict(float)
    level_names = ("p3", "p4", "p5")
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            frames = batch["frames"][:, 0].to(device, dtype=torch.float32) / 255.0
            gt_boxes = [items[0] for items in batch["gt_boxes"]]

            captured.clear()
            teacher(frames, gate_input=None)
            feats = [captured[name] for name in level_names]
            cls_preds, reg_preds = head(feats)
            pred_scores = torch.cat(
                [item.flatten(2) for item in cls_preds], dim=2
            ).permute(0, 2, 1)
            pred_dist = torch.cat(
                [item.flatten(2) for item in reg_preds], dim=2
            ).permute(0, 2, 1)
            anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)

            yolo_batch = _make_yolo_batch(gt_boxes, args.img_size, device)
            targets = torch.cat(
                (
                    yolo_batch["batch_idx"].view(-1, 1),
                    yolo_batch["cls"].view(-1, 1),
                    yolo_batch["bboxes"],
                ),
                dim=1,
            )
            image_size = (
                torch.tensor(feats[0].shape[2:], device=device, dtype=pred_scores.dtype)
                * criterion.stride[0]
            )
            targets = criterion.preprocess(
                targets,
                frames.shape[0],
                scale_tensor=image_size[[1, 0, 1, 0]],
            )
            gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0)
            if not mask_gt.any():
                continue

            pred_bboxes = criterion.bbox_decode(anchor_points, pred_dist)
            criterion.assigner.bs = frames.shape[0]
            criterion.assigner.n_max_boxes = gt_bboxes.shape[1]
            mask_pre, align_metric, overlaps = criterion.assigner.get_pos_mask(
                pred_scores.sigmoid(),
                pred_bboxes * stride_tensor,
                gt_labels,
                gt_bboxes,
                anchor_points * stride_tensor,
                mask_gt,
            )
            conflict_anchors = mask_pre.sum(dim=1) > 1
            _, _, mask_post = criterion.assigner.select_highest_overlaps(
                mask_pre.clone(),
                overlaps,
                criterion.assigner.n_max_boxes,
                align_metric,
            )

            pre_counts = mask_pre.sum(dim=-1)
            post_counts = mask_post.sum(dim=-1)
            level_sizes = [item.shape[-2] * item.shape[-1] for item in cls_preds]
            level_offsets = [0]
            for size in level_sizes:
                level_offsets.append(level_offsets[-1] + size)

            for batch_pos, boxes in enumerate(gt_boxes):
                seq = batch["seq"][batch_pos]
                scale_h, scale_w = loader.dataset._scale_hw[seq]  # type: ignore[attr-defined]
                for gt_index, box in enumerate(boxes):
                    resized_min = float(min(box[2] - box[0], box[3] - box[1]).item())
                    original_min = float(
                        min(
                            (box[2] - box[0]).item() / scale_w,
                            (box[3] - box[1]).item() / scale_h,
                        )
                    )
                    resized_bin = _size_bin(resized_min)
                    original_bin = _size_bin(original_min)
                    pre = int(pre_counts[batch_pos, gt_index].item())
                    post = int(post_counts[batch_pos, gt_index].item())

                    _increment(stats, "gt_total")
                    _increment(stats, f"resized_{resized_bin}_gt")
                    _increment(stats, f"original_{original_bin}_gt")
                    _increment(stats, "positive_pre_total", pre)
                    _increment(stats, "positive_post_total", post)
                    _increment(stats, f"resized_{resized_bin}_positive_pre_total", pre)
                    _increment(
                        stats, f"resized_{resized_bin}_positive_post_total", post
                    )
                    if pre == 0:
                        _increment(stats, "gt_zero_pre")
                        _increment(stats, f"resized_{resized_bin}_zero_pre")
                    if post == 0:
                        _increment(stats, "gt_zero_post")
                        _increment(stats, f"resized_{resized_bin}_zero_post")
                    if pre > 0 and post == 0:
                        _increment(stats, "gt_lost_all_to_conflict")
                        _increment(stats, f"resized_{resized_bin}_lost_all_to_conflict")
                    for level, start, end in zip(
                        level_names, level_offsets[:-1], level_offsets[1:]
                    ):
                        count = int(mask_post[batch_pos, gt_index, start:end].sum())
                        _increment(stats, f"{level}_positive_total", count)
                        _increment(
                            stats,
                            f"resized_{resized_bin}_{level}_positive_total",
                            count,
                        )

            _increment(stats, "frames", frames.shape[0])
            _increment(stats, "anchors", conflict_anchors.numel())
            _increment(stats, "conflict_anchors", int(conflict_anchors.sum()))

    for hook in hooks:
        hook.remove()

    gt_total = max(stats["gt_total"], 1.0)
    anchor_total = max(stats["anchors"], 1.0)
    report = dict(sorted(stats.items()))
    report.update(
        {
            "assigner": args.assigner,
            "zero_pre_rate": stats["gt_zero_pre"] / gt_total,
            "zero_post_rate": stats["gt_zero_post"] / gt_total,
            "lost_all_to_conflict_rate": stats["gt_lost_all_to_conflict"] / gt_total,
            "conflict_anchor_rate": stats["conflict_anchors"] / anchor_total,
            "mean_positive_pre": stats["positive_pre_total"] / gt_total,
            "mean_positive_post": stats["positive_post_total"] / gt_total,
        }
    )
    for size_bin in ("lt4", "4to8", "8to16", "ge16"):
        bin_gt = stats[f"resized_{size_bin}_gt"]
        if bin_gt <= 0:
            continue
        report.update(
            {
                f"resized_{size_bin}_zero_pre_rate": stats[
                    f"resized_{size_bin}_zero_pre"
                ]
                / bin_gt,
                f"resized_{size_bin}_zero_post_rate": stats[
                    f"resized_{size_bin}_zero_post"
                ]
                / bin_gt,
                f"resized_{size_bin}_lost_all_to_conflict_rate": stats[
                    f"resized_{size_bin}_lost_all_to_conflict"
                ]
                / bin_gt,
                f"resized_{size_bin}_mean_positive_pre": stats[
                    f"resized_{size_bin}_positive_pre_total"
                ]
                / bin_gt,
                f"resized_{size_bin}_mean_positive_post": stats[
                    f"resized_{size_bin}_positive_post_total"
                ]
                / bin_gt,
                f"resized_{size_bin}_p3_positive_share": stats[
                    f"resized_{size_bin}_p3_positive_total"
                ]
                / max(stats[f"resized_{size_bin}_positive_post_total"], 1.0),
                f"resized_{size_bin}_p4_positive_share": stats[
                    f"resized_{size_bin}_p4_positive_total"
                ]
                / max(stats[f"resized_{size_bin}_positive_post_total"], 1.0),
                f"resized_{size_bin}_p5_positive_share": stats[
                    f"resized_{size_bin}_p5_positive_total"
                ]
                / max(stats[f"resized_{size_bin}_positive_post_total"], 1.0),
            }
        )
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
