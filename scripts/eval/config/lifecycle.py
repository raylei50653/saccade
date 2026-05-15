from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class LifecycleConfig:
    # Birth gates (Experimental, related to track lifecycle)
    birth_quality_gate: bool = False
    birth_min_quality: float = 0.0
    birth_quality_score_bias: float = 0.15
    stage2_quality_gate: bool = False
    stage2_quality_min: float = 0.40
    birth_consecutive_gate: bool = False
    birth_consecutive_frames: int = 2
    birth_consecutive_iou: float = 0.40
    birth_consecutive_boost: float = 0.05
    birth_consecutive_min_score: float = 0.20
    birth_consecutive_min_motion: float = 0.0
    # Pre-output lifecycle merge
    lifecycle_merge: bool = False
    lifecycle_ttl: int = 45
    lifecycle_min_gap: int = 2
    lifecycle_spatial_gate: float = 0.08
    lifecycle_min_iou: float = 0.0
    lifecycle_sim_threshold: float = 0.90
    lifecycle_require_embedding: bool = False
    lifecycle_ema: float = 0.83
    # Post-output lifecycle merge
    post_lifecycle_merge: bool = False
    post_lifecycle_ttl: int = 60
    post_lifecycle_min_gap: int = 1
    post_lifecycle_velocity_samples: int = 5
    post_lifecycle_spatial_weight: float = 0.35
    post_lifecycle_motion_weight: float = 0.45
    post_lifecycle_time_weight: float = 0.10
    post_lifecycle_direction_weight: float = 0.25
    post_lifecycle_max_cost: float = 1.25
    post_lifecycle_appearance_gate: bool = False
    post_lifecycle_appearance_threshold: float = 0.90
    post_lifecycle_appearance_min_samples: int = 1
    post_lifecycle_appearance_max_samples: int = 5
    post_lifecycle_appearance_min_score: float = 0.0
    post_lifecycle_appearance_min_consistency: float = 0.0
    post_lifecycle_appearance_weight: float = 0.0
    post_lifecycle_gap_uncertainty_weight: float = 0.0
    post_lifecycle_consistency_weight: float = 0.0
    post_lifecycle_missing_appearance_cost: float = 0.5
    # Duplicate suppression: remove near-duplicate detections within the same frame
    # (detector artifact where multiple overlapping boxes are produced for the same person)
    duplicate_suppression_enabled: bool = False
    duplicate_suppression_iou_threshold: float = 0.85
    duplicate_suppression_min_score_ratio: float = 1.05
    # Multi-signal birth policy (P5-1): joint evidence over score × streak × motion × geometry
    multi_birth_enabled: bool = False
    multi_birth_min_score: float = 0.12
    multi_birth_min_frames: int = 3
    multi_birth_target_motion: float = 12.0
    multi_birth_evidence_threshold: float = 0.60
    multi_birth_iou_match: float = 0.30
    multi_birth_ttl_frames: int = 5
    multi_birth_w_score: float = 0.35
    multi_birth_w_motion: float = 0.30
    multi_birth_w_quality: float = 0.20
    multi_birth_w_streak: float = 0.15
    multi_birth_min_aspect: float = 0.0
    multi_birth_max_area_px: int = 0
    # Tracklet cleanup
    min_tracklet_len: int = 1
    min_tracklet_score: float = 0.0
    # Tracklet interpolation
    interpolate_tracklets: bool = True
    interpolate_max_gap: int = 20
    interpolate_min_track_len: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LifecycleConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group(
        _tier("Lifecycle merge and tracklet cleanup", "Experimental")
    )
    grp.description = (
        "Late-stage identity stitching, birth gates, and output pruning. "
        "Load with: --module-lifecycle configs/modules/lifecycle.yaml"
    )
    grp.add_argument(
        "--birth-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow detection quality to add a birth-score bias before a new tentative track spawns.",
    )
    grp.add_argument(
        "--birth-min-quality",
        type=float,
        default=0.0,
        help=_help(
            "Quality pivot above which unmatched detections receive a birth-score bonus.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--birth-quality-score-bias",
        type=float,
        default=0.15,
        help=_help(
            "Scale factor for converting quality into additive birth score.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--stage2-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove Stage 2 detections in [track_thresh, mid_thresh) below stage2-quality-min.",
    )
    grp.add_argument(
        "--stage2-quality-min",
        type=float,
        default=0.40,
        help=_help(
            "Geometry quality floor for Stage 2 quality gate.",
            range_hint="0-1",
            edge="too high removes valid Stage 2 matches for partially occluded people",
        ),
    )
    grp.add_argument(
        "--birth-consecutive-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boost sub-threshold detections only when they appear in N consecutive frames (rolling IoU window).",
    )
    grp.add_argument(
        "--birth-consecutive-frames",
        type=int,
        default=2,
        help=_help(
            "Number of consecutive frames a detection must appear to qualify for birth boost.",
            range_hint=">=2",
            edge="higher values reduce FP births but delay recovery",
        ),
    )
    grp.add_argument(
        "--birth-consecutive-iou",
        type=float,
        default=0.40,
        help=_help(
            "Minimum IoU to match a detection across consecutive frames.",
            range_hint="0-1",
            edge="too low admits noisy matches; too high misses slow-moving pedestrians",
        ),
    )
    grp.add_argument(
        "--birth-consecutive-boost",
        type=float,
        default=0.05,
        help=_help(
            "Score added to qualifying consecutive-gate detections.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--birth-consecutive-min-score",
        type=float,
        default=0.20,
        help=_help(
            "Minimum score for a sub-threshold detection to be eligible for consecutive-gate boost.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--birth-consecutive-min-motion",
        type=float,
        default=0.0,
        help=_help(
            "Minimum center displacement (pixels) from oldest window frame. 0 disables.",
            range_hint=">=0, suggested 3-8 px",
            edge="too high excludes slow-moving distant pedestrians",
        ),
    )
    grp.add_argument(
        "--lifecycle-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable pre-output lifecycle merge.",
    )
    grp.add_argument(
        "--lifecycle-ttl",
        type=int,
        default=45,
        help=_help("Frames to keep dead tracks mergeable.", range_hint=">=1"),
    )
    grp.add_argument(
        "--lifecycle-min-gap",
        type=int,
        default=2,
        help=_help(
            "Minimum dead/alive gap before lifecycle merge is considered.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--lifecycle-spatial-gate",
        type=float,
        default=0.08,
        help=_help("Spatial gate for lifecycle merge candidates.", range_hint=">=0"),
    )
    grp.add_argument(
        "--lifecycle-min-iou",
        type=float,
        default=0.0,
        help=_help("Minimum IoU for lifecycle merge.", range_hint="0-1"),
    )
    grp.add_argument(
        "--lifecycle-sim-threshold",
        type=float,
        default=0.90,
        help=_help(
            "Appearance similarity threshold for lifecycle merge.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--lifecycle-require-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require appearance embedding to allow lifecycle merge.",
    )
    grp.add_argument(
        "--lifecycle-ema",
        type=float,
        default=0.83,
        help=_help("EMA decay for lifecycle merge appearance state.", range_hint="0-1"),
    )
    grp.add_argument(
        "--post-lifecycle-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable post-output lifecycle merge.",
    )
    grp.add_argument(
        "--post-lifecycle-ttl",
        type=int,
        default=60,
        help=_help("Frames to keep tracklets for post merge.", range_hint=">=1"),
    )
    grp.add_argument(
        "--post-lifecycle-min-gap",
        type=int,
        default=1,
        help=_help("Minimum gap before post merge.", range_hint=">=0"),
    )
    grp.add_argument(
        "--post-lifecycle-velocity-samples",
        type=int,
        default=5,
        help=_help("Samples used to estimate post-merge motion.", range_hint=">=1"),
    )
    grp.add_argument(
        "--post-lifecycle-spatial-weight",
        type=float,
        default=0.35,
        help=_help("Spatial term weight in post-merge cost.", range_hint=">=0"),
    )
    grp.add_argument(
        "--post-lifecycle-motion-weight",
        type=float,
        default=0.45,
        help=_help("Motion term weight in post-merge cost.", range_hint=">=0"),
    )
    grp.add_argument(
        "--post-lifecycle-time-weight",
        type=float,
        default=0.10,
        help=_help("Time-gap term weight in post-merge cost.", range_hint=">=0"),
    )
    grp.add_argument(
        "--post-lifecycle-direction-weight",
        type=float,
        default=0.25,
        help=_help(
            "Direction-consistency weight in post-merge cost.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--post-lifecycle-max-cost",
        type=float,
        default=1.25,
        help=_help(
            "Maximum total post-merge cost to accept a stitch.",
            range_hint=">0",
            edge="lower is safer but leaves fragments",
        ),
    )
    grp.add_argument(
        "--post-lifecycle-appearance-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require appearance gate during post merge.",
    )
    grp.add_argument(
        "--post-lifecycle-appearance-threshold",
        type=float,
        default=0.90,
        help=_help("Appearance threshold for post merge.", range_hint="0-1"),
    )
    grp.add_argument(
        "--post-lifecycle-appearance-min-samples",
        type=int,
        default=1,
        help=_help(
            "Minimum appearance samples before post-merge gating activates.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--post-lifecycle-appearance-max-samples",
        type=int,
        default=5,
        help=_help(
            "Maximum appearance samples used in post-merge gating.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--post-lifecycle-appearance-min-score",
        type=float,
        default=0.0,
        help=_help(
            "Minimum sample score for post-merge appearance gating.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--post-lifecycle-appearance-min-consistency",
        type=float,
        default=0.0,
        help=_help(
            "Minimum consistency for post-merge appearance gating.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--post-lifecycle-appearance-weight",
        type=float,
        default=0.0,
        help=_help(
            "Add appearance cost (1-sim) as a soft term in post-merge cost. 0 disables.",
            range_hint="0-2",
        ),
    )
    grp.add_argument(
        "--post-lifecycle-gap-uncertainty-weight",
        type=float,
        default=0.0,
        help=_help(
            "Scale appearance weight by (1 + k * gap/ttl). Longer gaps need stronger match.",
            range_hint="0-2",
        ),
    )
    grp.add_argument(
        "--post-lifecycle-consistency-weight",
        type=float,
        default=0.0,
        help=_help(
            "Penalise low-consistency tracklets in post-merge cost.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--post-lifecycle-missing-appearance-cost",
        type=float,
        default=0.5,
        help=_help(
            "Appearance cost assigned when embedding is unavailable.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--duplicate-suppression-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress near-duplicate detections within the same frame (detector artifact removal).",
    )
    grp.add_argument(
        "--duplicate-suppression-iou-threshold",
        type=float,
        default=0.85,
        help=_help(
            "IoU threshold for duplicate detection. Boxes with IoU above this are candidates for suppression.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--duplicate-suppression-min-score-ratio",
        type=float,
        default=1.05,
        help=_help(
            "Minimum score ratio between high-score and low-score detection for duplicate suppression.",
            range_hint=">1",
        ),
    )
    grp.add_argument(
        "--multi-birth-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable multi-signal birth policy (P5-1): joint evidence over score, streak, motion, geometry.",
    )
    grp.add_argument(
        "--multi-birth-min-score",
        type=float,
        default=0.12,
        help=_help(
            "Minimum detection score to enter the multi-birth candidate buffer.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--multi-birth-min-frames",
        type=int,
        default=3,
        help=_help(
            "Minimum frames a candidate must appear before evidence is evaluated.",
            range_hint=">=2",
        ),
    )
    grp.add_argument(
        "--multi-birth-target-motion",
        type=float,
        default=12.0,
        help=_help(
            "Centroid motion (px/frame) where motion evidence reaches 1.0.",
            range_hint=">0",
        ),
    )
    grp.add_argument(
        "--multi-birth-evidence-threshold",
        type=float,
        default=0.60,
        help=_help(
            "Joint evidence threshold to promote a candidate to birth.",
            range_hint="0-1",
            edge="lower promotes more candidates but increases FP risk",
        ),
    )
    grp.add_argument(
        "--multi-birth-iou-match",
        type=float,
        default=0.30,
        help=_help(
            "Minimum IoU to associate a detection with an existing candidate.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--multi-birth-ttl-frames",
        type=int,
        default=5,
        help=_help(
            "Frames before an unmatched candidate is expired.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--multi-birth-w-score",
        type=float,
        default=0.35,
        help="Evidence weight for detection score signal.",
    )
    grp.add_argument(
        "--multi-birth-w-motion",
        type=float,
        default=0.30,
        help="Evidence weight for centroid motion signal.",
    )
    grp.add_argument(
        "--multi-birth-w-quality",
        type=float,
        default=0.20,
        help="Evidence weight for geometry quality signal.",
    )
    grp.add_argument(
        "--multi-birth-w-streak",
        type=float,
        default=0.15,
        help="Evidence weight for temporal streak signal.",
    )
    grp.add_argument(
        "--multi-birth-min-aspect",
        type=float,
        default=0.0,
        help=_help(
            "Hard-reject candidates with h/w aspect below this. 0=disabled.",
            range_hint=">=0",
            edge="set 1.5-2.0 to restrict to narrow-person geometry",
        ),
    )
    grp.add_argument(
        "--multi-birth-max-area-px",
        type=int,
        default=0,
        help=_help(
            "Hard-reject candidates with pixel area above this. 0=disabled.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--min-tracklet-len",
        type=int,
        default=1,
        help=_help("Drop tracklets shorter than this length.", range_hint=">=1"),
    )
    grp.add_argument(
        "--multi-birth-replace-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable replace mode: suppress competing detection when evidence is very high.",
    )
    grp.add_argument(
        "--multi-birth-replace-evidence-threshold",
        type=float,
        default=0.85,
        help=_help(
            "Evidence threshold for replace mode (higher than evidence_threshold).",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--min-tracklet-score",
        type=float,
        default=0.0,
        help=_help("Drop tracklets below this mean detection score.", range_hint="0-1"),
    )
    grp.add_argument(
        "--interpolate-tracklets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill short gaps in confirmed tracklets with linear interpolation.",
    )
    grp.add_argument(
        "--interpolate-max-gap",
        type=int,
        default=20,
        help=_help(
            "Max gap (frames) to interpolate across.",
            range_hint="1-30",
            edge="larger recoveries risk hallucinating boxes across scene cuts",
        ),
    )
    grp.add_argument(
        "--interpolate-min-track-len",
        type=int,
        default=5,
        help=_help(
            "Minimum track observations required before gaps are interpolated.",
            range_hint=">=2",
            edge="too low interpolates noisy short tracks",
        ),
    )
