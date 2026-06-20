from typing import Any

import torch
from saccade.perception.box_ops import box_iou
from .types import MotRecord


def parse_debug_frame_ranges(raw: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            parts = item.split("-", 1)
            start = int(parts[0].strip())
            end = int(parts[1].strip())
        else:
            start = int(item)
            end = start
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def debug_frame_selected(
    seq: str,
    frame_id: int,
    target_seq: str,
    frame_ranges: list[tuple[int, int]],
) -> bool:
    if not target_seq or seq != target_seq:
        return False
    if not frame_ranges:
        return True
    return any(start <= frame_id <= end for start, end in frame_ranges)


def append_stage_dump_rows(
    rows: list[dict[str, Any]],
    *,
    seq: str,
    frame_id: int,
    stage: str,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
) -> None:
    if boxes.numel() == 0:
        return
    boxes_cpu = boxes.detach().to(torch.float32).cpu()
    scores_cpu = scores.detach().to(torch.float32).cpu()
    classes_cpu = classes.detach().to(torch.int32).cpu()
    for idx in range(int(boxes_cpu.shape[0])):
        box = boxes_cpu[idx]
        rows.append(
            {
                "seq": seq,
                "frame": frame_id,
                "stage": stage,
                "det_idx": idx,
                "x1": float(box[0].item()),
                "y1": float(box[1].item()),
                "x2": float(box[2].item()),
                "y2": float(box[3].item()),
                "w": float((box[2] - box[0]).item()),
                "h": float((box[3] - box[1]).item()),
                "score": float(scores_cpu[idx].item()),
                "cls": int(classes_cpu[idx].item()),
            }
        )


def mot_box(record: MotRecord) -> tuple[float, float, float, float]:
    return (record.x, record.y, record.x + record.w, record.y + record.h)


def box_iou_tuple(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    return box_iou(a, b, union_mode="clamp")


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def shift_box(
    box: tuple[float, float, float, float],
    velocity: tuple[float, float],
    dt: float,
) -> tuple[float, float, float, float]:
    dx = velocity[0] * dt
    dy = velocity[1] * dt
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def mot_result_line(
    frame_id: int,
    global_tid: int,
    box: tuple[float, float, float, float],
    score: float,
    frame_w: int,
    frame_h: int,
) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{frame_id},{global_tid},{x1:.2f},{y1:.2f},"
        f"{x2 - x1:.2f},{y2 - y1:.2f},"
        f"{score:.4f},-1,-1,-1"
    )


def safe_cpp_ptr(obj: object) -> int:
    try:
        return int(getattr(obj, "cpp_ptr"))
    except Exception:
        return 0


def apply_narrow_person_score_bonus(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    *,
    frame_w: int,
    frame_h: int,
    person_class: int,
    bonus: float,
    max_width_ratio: float,
    min_height_ratio: float,
    min_aspect: float,
    max_aspect: float,
) -> torch.Tensor:
    if (
        bonus <= 0.0
        or boxes.numel() == 0
        or scores.numel() == 0
        or classes.numel() == 0
    ):
        return scores
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
    width_ratio = widths / max(float(frame_w), 1.0)
    height_ratio = heights / max(float(frame_h), 1.0)
    aspect = heights / widths
    mask = (
        (classes == person_class)
        & (width_ratio <= max_width_ratio)
        & (height_ratio >= min_height_ratio)
        & (aspect >= min_aspect)
        & (aspect <= max_aspect)
    )
    if not bool(mask.any()):
        return scores
    boosted = scores.clone()
    boosted[mask] = torch.clamp(boosted[mask] + bonus, max=1.0)
    return boosted


def tile_seam_mask(
    boxes: torch.Tensor,
    *,
    tiling: str,
    h_orig: int,
    w_orig: int,
    seam_margin_canvas_px: float = 24.0,
) -> torch.Tensor:
    """Return a bool mask for boxes near tiled seam boundaries."""
    if boxes.numel() == 0 or tiling not in {
        "960p_2x2",
        "960p_3x2",
        "sahi_960p_2x2",
        "mamba_global_2x2",
    }:
        return torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.bool)

    r = 960.0 / max(h_orig, w_orig)
    h_new = int(h_orig * r)
    w_new = int(w_orig * r)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2

    seam_x_canvas = [320.0, 640.0]
    seam_y_canvas = [320.0, 640.0]
    if tiling == "960p_3x2":
        seam_x_canvas = [160.0, 320.0, 640.0, 800.0]

    seam_margin_orig = seam_margin_canvas_px / max(r, 1e-6)
    seam_x = [(sx - x_off) / r for sx in seam_x_canvas if x_off < sx < x_off + w_new]
    seam_y = [(sy - y_off) / r for sy in seam_y_canvas if y_off < sy < y_off + h_new]
    if not seam_x and not seam_y:
        return torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.bool)

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    near_any = torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.bool)

    for sx in seam_x:
        near_any |= (boxes[:, 0] <= sx) & (boxes[:, 2] >= sx)
        near_any |= (cx - sx).abs() <= seam_margin_orig
    for sy in seam_y:
        near_any |= (boxes[:, 1] <= sy) & (boxes[:, 3] >= sy)
        near_any |= (cy - sy).abs() <= seam_margin_orig
    return near_any


def count_tile_seam_boxes(
    boxes: torch.Tensor,
    *,
    tiling: str,
    h_orig: int,
    w_orig: int,
    seam_margin_canvas_px: float = 24.0,
) -> int:
    return int(
        tile_seam_mask(
            boxes,
            tiling=tiling,
            h_orig=h_orig,
            w_orig=w_orig,
            seam_margin_canvas_px=seam_margin_canvas_px,
        )
        .sum()
        .item()
    )
