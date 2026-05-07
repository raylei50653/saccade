import math
import torch


def compute_detection_quality_batch(
    boxes: torch.Tensor,
    frame_w: int,
    frame_h: int,
    w_aspect: float = 0.50,
    w_center: float = 0.30,
    w_area: float = 0.20,
) -> torch.Tensor:
    """Vectorised composite quality score for all detections in a frame.

    Returns a tensor of shape (N,) with values in [0, 1].
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.float32)

    # Aspect ratio: Gaussian peak at 2.5
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    aspect = bh / bw
    aspect_q = torch.exp(-0.5 * ((aspect - 2.5) / 1.2) ** 2)

    # Center bias (truncation proxy)
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    cx_norm = cx / max(frame_w, 1)
    cy_norm = cy / max(frame_h, 1)
    center_q = (
        torch.stack([cx_norm, 1.0 - cx_norm, cy_norm, 1.0 - cy_norm], dim=1)
        .min(dim=1)
        .values
        * 4.0
    )
    center_q = center_q.clamp(0.0, 1.0)

    # Area ratio: Gaussian peak at 0.01
    area_ratio = (bw * bh) / max(float(frame_w * frame_h), 1.0)
    area_q = torch.exp(-0.5 * ((area_ratio - 0.01) / 0.01) ** 2)

    # Combined quality (geometrical only)
    return w_aspect * aspect_q + w_center * center_q + w_area * area_q


def compute_bank_quality_score(
    det_score: float,
    iou: float,
    aspect_ratio: float,
    box: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    *,
    w_det: float = 0.45,
    w_iou: float = 0.20,
    w_aspect: float = 0.15,
    w_center: float = 0.10,
    w_area: float = 0.10,
) -> float:
    """Composite bank sample quality score combining detection, motion, geometry signals."""
    ideal_aspect = 2.5
    sigma_aspect = 1.2
    if aspect_ratio > 0.0:
        aspect_q = math.exp(-0.5 * ((aspect_ratio - ideal_aspect) / sigma_aspect) ** 2)
    else:
        aspect_q = 0.5  # unknown → neutral

    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    cx_norm = cx / max(frame_w, 1)
    cy_norm = cy / max(frame_h, 1)
    center_q = min(cx_norm, 1.0 - cx_norm, cy_norm, 1.0 - cy_norm) * 4.0
    center_q = max(0.0, min(1.0, center_q))

    box_w = x2 - x1
    box_h = y2 - y1
    area_ratio = (box_w * box_h) / max(float(frame_w * frame_h), 1.0)
    ideal_area = 0.01
    sigma_area = 0.01
    area_q = math.exp(-0.5 * ((area_ratio - ideal_area) / sigma_area) ** 2)

    return (
        w_det * det_score
        + w_iou * iou
        + w_aspect * aspect_q
        + w_center * center_q
        + w_area * area_q
    )
