from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class DetectionConfig:
    # Preprocessing
    preprocess: str = "letterbox,gamma,contrast"
    gamma: float = 0.8
    gamma_luma_threshold: float = 0.35
    contrast: float = 1.2
    # Tiling
    tiling: str = "native_960"
    cross_tile_merge: bool = True
    cross_tile_score_penalty: float = 1.0
    tile_diagnostics: bool = False
    tile_seam_score_penalty: float = 1.0
    tile_seam_margin_canvas_px: float = 24.0
    cross_tile_seam_center_scale: float = 1.8
    cross_tile_seam_area_ratio_threshold: float = 0.30
    cross_tile_seam_min_overlap_ratio: float = 0.45
    # Basic detector settings
    person_class: int = 0
    track_person_only: bool = True
    detector_box_format: str = "xyxy"
    nms_iou_threshold: float | None = None
    # Crowd low-score mode
    crowd_low_score_mode: bool = False
    crowd_low_score_trigger: int = 25
    crowd_conf_threshold: float = 0.02
    # FP hard filter
    fp_hard_filter_enabled: bool = True
    fp_hard_filter_min_score: float = 0.10
    fp_hard_filter_max_suspicious_area: int = 40000
    fp_hard_filter_max_suspicious_score: float = 0.40
    # Narrow person score bonus
    narrow_person_score_bonus: float = 0.0
    narrow_person_max_width_ratio: float = 0.018
    narrow_person_min_height_ratio: float = 0.045
    narrow_person_min_aspect: float = 2.0
    narrow_person_max_aspect: float = 4.8
    # Per-frame detection cap
    per_frame_detection_cap: int = 0
    detection_cap_rank_method: str = "fp_filter_quality"
    adaptive_detection_cap: bool = False
    adaptive_cap_base: int = 40
    adaptive_cap_max: int = 60
    adaptive_cap_min: int = 15
    # Temporal consistency
    temporal_consistency_min_frames: int = 3
    # Scene-adaptive policy (P5-4)
    scene_adapt_enabled: bool = False
    scene_adapt_window: int = 30
    scene_adapt_crowd_thresh: float = 15.0
    scene_adapt_narrow_aspect_thresh: float = 2.1
    scene_adapt_narrow_width_thresh: float = 0.035

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectionConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_detection_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group(_tier("Detection and preprocessing", "Tier 1/2"))
    grp.description = (
        "Front-end controls for detector filtering and image preparation. "
        "Load with: --module-detection configs/modules/detection.yaml"
    )
    grp.add_argument(
        "--preprocess",
        default="letterbox,gamma,contrast",
        help="Comma-separated preprocessing pipeline.",
    )
    grp.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help=_help(
            "Gamma correction factor.",
            range_hint="0.6-1.4",
            edge="too low brightens shadows but amplifies sensor noise",
        ),
    )
    grp.add_argument(
        "--gamma-luma-threshold",
        type=float,
        default=0.35,
        help=_help("Only apply gamma below this luma threshold.", range_hint="0-1"),
    )
    grp.add_argument(
        "--contrast",
        type=float,
        default=1.2,
        help=_help(
            "Linear contrast multiplier.",
            range_hint="0.8-1.6",
            edge="too high clips highlights and destabilizes embeddings",
        ),
    )
    grp.add_argument(
        "--tiling",
        choices=["960p_2x2", "960p_3x2", "native_960", "sahi_960p_2x2"],
        default="native_960",
        help="Inference tiling preset.",
    )
    grp.add_argument(
        "--cross-tile-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge duplicate detections across tile boundaries.",
    )
    grp.add_argument(
        "--cross-tile-score-penalty",
        type=float,
        default=1.0,
        help=_help("Multiply score of cross-tile-merged boxes.", range_hint="0-1"),
    )
    grp.add_argument(
        "--tile-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit per-sequence diagnostics for tiled detection.",
    )
    grp.add_argument(
        "--tile-seam-score-penalty",
        type=float,
        default=1.0,
        help=_help(
            "Multiply scores of seam-near tiled detections.",
            range_hint="0-1",
            edge="lower suppresses tile-boundary artifacts",
        ),
    )
    grp.add_argument(
        "--tile-seam-margin-canvas-px",
        type=float,
        default=24.0,
        help=_help(
            "Half-width of the seam-near band on the 960 canvas.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--cross-tile-seam-center-scale",
        type=float,
        default=1.8,
        help=_help(
            "Scale factor widening center-distance gating for seam-near merge.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--cross-tile-seam-area-ratio-threshold",
        type=float,
        default=0.30,
        help=_help(
            "Minimum area-ratio for seam-near duplicate merge candidates.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--cross-tile-seam-min-overlap-ratio",
        type=float,
        default=0.45,
        help=_help(
            "Minimum overlap ratio on both axes for seam-near merge.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--person-class",
        type=int,
        default=0,
        help="Person class id in detector output.",
    )
    grp.add_argument(
        "--track-person-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only person-class detections for tracking.",
    )
    grp.add_argument(
        "--detector-box-format",
        choices=["xyxy", "cxcywh"],
        default="xyxy",
        help="Raw detector box layout before evaluator decoding.",
    )
    grp.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=None,
        help=_help(
            "Override detector NMS IoU.",
            range_hint="0-1 or unset",
            edge="lower removes duplicates earlier; higher preserves crowded detections",
        ),
    )
    grp.add_argument(
        "--crowd-low-score-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable lower detector/tracker score floors only on crowded frames.",
    )
    grp.add_argument(
        "--crowd-low-score-trigger",
        type=int,
        default=25,
        help=_help(
            "Activate crowd mode when post-merge detections reach this count.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--crowd-conf-threshold",
        type=float,
        default=0.02,
        help=_help("Detector confidence floor during crowd mode.", range_hint="0-1"),
    )
    grp.add_argument(
        "--fp-hard-filter-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove extremely suspicious low-score large-area detections before tracking.",
    )
    grp.add_argument(
        "--fp-hard-filter-min-score",
        type=float,
        default=0.10,
        help=_help(
            "Detections below this score are FP filter candidates.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--fp-hard-filter-max-suspicious-area",
        type=int,
        default=40000,
        help=_help(
            "Bbox pixel area above which a low-score detection is suspicious.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--fp-hard-filter-max-suspicious-score",
        type=float,
        default=0.40,
        help=_help(
            "Score ceiling defining the suspicious range for hard filtering.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--narrow-person-score-bonus",
        type=float,
        default=0.0,
        help=_help(
            "Additive score bonus for narrow person-like raw detections.",
            range_hint=">=0",
            edge="too high promotes background slivers into tracks",
        ),
    )
    grp.add_argument(
        "--narrow-person-max-width-ratio",
        type=float,
        default=0.018,
        help=_help(
            "Maximum box width / frame width for narrow-person bonus.", range_hint=">0"
        ),
    )
    grp.add_argument(
        "--narrow-person-min-height-ratio",
        type=float,
        default=0.045,
        help=_help(
            "Minimum box height / frame height for narrow-person bonus.",
            range_hint=">0",
        ),
    )
    grp.add_argument(
        "--narrow-person-min-aspect",
        type=float,
        default=2.0,
        help=_help(
            "Minimum h/w aspect ratio for narrow-person bonus.", range_hint=">0"
        ),
    )
    grp.add_argument(
        "--narrow-person-max-aspect",
        type=float,
        default=4.8,
        help=_help(
            "Maximum h/w aspect ratio for narrow-person bonus.", range_hint=">0"
        ),
    )
    grp.add_argument(
        "--per-frame-detection-cap",
        type=int,
        default=0,
        help=_help("Maximum detections kept per frame (0=off).", range_hint=">=0"),
    )
    grp.add_argument(
        "--detection-cap-rank-method",
        default="fp_filter_quality",
        choices=["score", "quality", "fp_filter", "fp_filter_quality"],
        help="Ranking method for per-frame detection cap.",
    )
    grp.add_argument(
        "--adaptive-detection-cap",
        action="store_true",
        help="Enable adaptive per-frame detection cap based on scene density.",
    )
    grp.add_argument(
        "--adaptive-cap-base",
        type=int,
        default=40,
        help=_help("Base cap for adaptive detection cap.", range_hint=">=0"),
    )
    grp.add_argument(
        "--adaptive-cap-max",
        type=int,
        default=60,
        help=_help("Maximum cap for adaptive detection cap.", range_hint=">=0"),
    )
    grp.add_argument(
        "--adaptive-cap-min",
        type=int,
        default=15,
        help=_help("Minimum cap for adaptive detection cap.", range_hint=">=0"),
    )
    grp.add_argument(
        "--temporal-consistency-min-frames",
        type=int,
        default=3,
        help=_help(
            "Minimum consecutive frames a detection must appear to be kept.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--scene-adapt-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-sequence scene-adaptive policy (P5-4). "
        "Classifies scene from first N frames and selectively enables narrow-person bonus.",
    )
    grp.add_argument(
        "--scene-adapt-window",
        type=int,
        default=30,
        help=_help(
            "Frames to observe before scene classification.", range_hint="10-60"
        ),
    )
    grp.add_argument(
        "--scene-adapt-crowd-thresh",
        type=float,
        default=15.0,
        help=_help("Min avg boxes/frame to qualify as crowded.", range_hint="8-30"),
    )
    grp.add_argument(
        "--scene-adapt-narrow-aspect-thresh",
        type=float,
        default=2.1,
        help=_help(
            "Max avg h/w aspect for crowded-narrow classification "
            "(dense crowd scenes have LOWER avg aspect due to occlusion).",
            range_hint="1.8-2.5",
        ),
    )
    grp.add_argument(
        "--scene-adapt-narrow-width-thresh",
        type=float,
        default=0.035,
        help=_help(
            "Max avg box-width/frame-width for small/far-away people.",
            range_hint="0.02-0.05",
        ),
    )
