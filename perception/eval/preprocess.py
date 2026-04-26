import torch
from dataclasses import dataclass


def parse_preprocess(value):
    modes = [mode.strip().lower() for mode in value.split(",") if mode.strip()]
    if not modes or "none" in modes:
        return []
    allowed = {"gamma", "contrast", "letterbox"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unsupported preprocess mode(s): {', '.join(unknown)}")
    return modes


@dataclass
class GeometryScaleState:
    prev_scale: float = 1.0
    median_ratio_ema: float = 0.0
    initialized: bool = False


def geometry_mid_thresh_scale(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    frame_height: int,
    *,
    enabled: bool,
    person_class: int,
    track_person_only: bool,
    ref_height_ratio: float,
    min_scale: float,
    max_scale: float,
    ema_beta: float,
    loosen_step: float,
    tighten_step: float,
    min_samples: int,
    state: GeometryScaleState,
) -> float:
    if not enabled or boxes.numel() == 0 or frame_height <= 0:
        return state.prev_scale if state.initialized else 1.0
    geom_boxes = boxes
    if not track_person_only:
        person_mask = classes == person_class
        if not bool(person_mask.any()):
            return state.prev_scale if state.initialized else 1.0
        geom_boxes = boxes[person_mask]
    if geom_boxes.shape[0] < min_samples:
        return state.prev_scale if state.initialized else 1.0
    heights = (geom_boxes[:, 3] - geom_boxes[:, 1]).clamp_min(1.0)
    median_ratio = float(torch.median(heights).item()) / float(frame_height)
    if median_ratio <= 1e-6:
        return state.prev_scale if state.initialized else 1.0

    if not state.initialized:
        state.median_ratio_ema = median_ratio
        raw_scale = max(min_scale, min(max_scale, ref_height_ratio / state.median_ratio_ema))
        state.prev_scale = raw_scale
        state.initialized = True
        return raw_scale

    state.median_ratio_ema = ema_beta * state.median_ratio_ema + (1.0 - ema_beta) * median_ratio
    raw_scale = max(min_scale, min(max_scale, ref_height_ratio / state.median_ratio_ema))
    diff = raw_scale - state.prev_scale
    if diff < 0.0:
        step = max(diff, -loosen_step) if loosen_step > 0.0 else diff
    else:
        step = min(diff, tighten_step) if tighten_step > 0.0 else diff
    state.prev_scale = max(min_scale, min(max_scale, state.prev_scale + step))
    return state.prev_scale


def apply_frame_preprocess(frame, modes, gamma, gamma_luma_threshold, contrast):
    if not modes:
        return
    for mode in modes:
        if mode == "gamma":
            if gamma_luma_threshold <= 0.0 or float(frame.mean()) < gamma_luma_threshold:
                frame.copy_(frame.clamp(0.0, 1.0).pow(gamma))
        elif mode == "contrast":
            mean = frame.mean()
            frame.copy_(((frame - mean) * contrast + mean).clamp(0.0, 1.0))
