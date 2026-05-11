from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class GeometryConfig:
    # Crowd tracking thresholds (applied when crowd_low_score_mode is active)
    crowd_track_thresh: float = 0.02
    crowd_mid_thresh: float = 0.05
    crowd_new_track_thresh: float = 0.25
    # Adaptive geometry scaling
    geometry_mid_scale: bool = False
    geometry_ref_height_ratio: float = 0.12
    geometry_min_scale: float = 0.875
    geometry_max_scale: float = 1.20
    geometry_ema_beta: float = 0.80
    geometry_loosen_step: float = 0.08
    geometry_tighten_step: float = 0.03
    geometry_min_samples: int = 5
    # ID stability filter
    id_stability_filter: bool = True
    id_stability_min_hits: int = 2
    id_stability_min_iou: float = 0.05
    id_stability_max_center_shift: float = 2.0
    id_stability_max_gap: int = 1
    id_stability_score_ema: float = 0.70
    id_stability_min_score_ema: float = 0.15
    # Person geometry prior (hard filters)
    person_geometry_prior: bool = True
    person_min_height_ratio: float = 0.018
    person_min_aspect: float = 1.0
    person_max_aspect: float = 5.5
    person_min_area_ratio: float = 0.00006
    person_max_area_ratio: float = 0.0
    # Detection quality scaling
    detection_quality_scaling: bool = True
    detection_quality_w_aspect: float = 0.50
    detection_quality_w_center: float = 0.30
    detection_quality_w_area: float = 0.20
    # Geometry suspect
    geometry_suspect_support: bool = True
    geometry_suspect_score: float | None = None
    # Kalman
    nsa_kalman: bool = False
    kalman_r_scale: float = 0.75
    vel_dir_weight: float = 0.0
    fuse_score_weight: float = 0.0
    stage2_match_thresh: float = 0.5
    birth_low_score_thresh: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeometryConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_geometry_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group(_tier("Geometry priors and ID stability", "Tier 2"))
    grp.description = (
        "Shape, scale, and motion guards that reject implausible person boxes. "
        "Load with: --module-geometry configs/modules/geometry.yaml"
    )
    grp.add_argument("--crowd-track-thresh", type=float, default=0.02,
                     help=_help("Low-confidence association floor during crowd mode.", range_hint="0-1"))
    grp.add_argument("--crowd-mid-thresh", type=float, default=0.05,
                     help=_help("Mid-tier confidence bucket during crowd mode.", range_hint="0-1"))
    grp.add_argument("--crowd-new-track-thresh", type=float, default=0.25,
                     help=_help("New-track score threshold during crowd mode.", range_hint="0-1"))
    grp.add_argument("--geometry-mid-scale", action=argparse.BooleanOptionalAction, default=False,
                     help="Enable adaptive middle-scale geometry gating.")
    grp.add_argument("--geometry-ref-height-ratio", type=float, default=0.12,
                     help=_help("Reference person height ratio for geometry normalization.", range_hint="0-1"))
    grp.add_argument("--geometry-min-scale", type=float, default=0.875,
                     help=_help("Allowed lower scale multiplier around geometry reference.", range_hint=">0, usually 0.7-1.0"))
    grp.add_argument("--geometry-max-scale", type=float, default=1.20,
                     help=_help("Allowed upper scale multiplier around geometry reference.", range_hint=">1, usually 1.0-1.5"))
    grp.add_argument("--geometry-ema-beta", type=float, default=0.80,
                     help=_help("EMA smoothing for geometry reference updates.", range_hint="0-1",
                                edge="near 1 is stable but slow to adapt"))
    grp.add_argument("--geometry-loosen-step", type=float, default=0.08,
                     help=_help("Per-update rate for relaxing geometry gates.", range_hint="0-1"))
    grp.add_argument("--geometry-tighten-step", type=float, default=0.03,
                     help=_help("Per-update rate for tightening geometry gates.", range_hint="0-1"))
    grp.add_argument("--geometry-min-samples", type=int, default=5,
                     help=_help("Samples needed before geometry adaptation is trusted.", range_hint=">=1"))
    grp.add_argument("--id-stability-filter", action=argparse.BooleanOptionalAction, default=True,
                     help="Filter unstable identity handoffs with short-term spatial consistency checks.")
    grp.add_argument("--id-stability-min-hits", type=int, default=2,
                     help=_help("Minimum hits before stability checks are reliable.", range_hint=">=1"))
    grp.add_argument("--id-stability-min-iou", type=float, default=0.05,
                     help=_help("Minimum IoU to treat a continuation as stable.", range_hint="0-1"))
    grp.add_argument("--id-stability-max-center-shift", type=float, default=2.0,
                     help=_help("Maximum normalized center shift for stable continuation.", range_hint=">0",
                                edge="too low breaks fast movers, too high lets jumps through"))
    grp.add_argument("--id-stability-max-gap", type=int, default=1,
                     help=_help("Maximum frame gap still considered short-term stable.", range_hint=">=0"))
    grp.add_argument("--id-stability-score-ema", type=float, default=0.70,
                     help=_help("EMA decay for stability confidence.", range_hint="0-1"))
    grp.add_argument("--id-stability-min-score-ema", type=float, default=0.15,
                     help=_help("Minimum EMA score to keep a stable ID hypothesis.", range_hint="0-1"))
    grp.add_argument("--person-geometry-prior", action=argparse.BooleanOptionalAction, default=True,
                     help="Enable hard geometric filtering for person boxes.")
    grp.add_argument("--person-min-height-ratio", type=float, default=0.018,
                     help=_help("Minimum bbox height as image-height ratio.", range_hint="0-1",
                                edge="higher drops far-away pedestrians"))
    grp.add_argument("--person-min-aspect", type=float, default=1.0,
                     help=_help("Minimum h/w aspect ratio for person boxes.", range_hint=">0"))
    grp.add_argument("--person-max-aspect", type=float, default=5.5,
                     help=_help("Maximum h/w aspect ratio for person boxes.", range_hint="> person-min-aspect"))
    grp.add_argument("--person-min-area-ratio", type=float, default=0.00006,
                     help=_help("Minimum bbox area as image-area ratio.", range_hint="0-1"))
    grp.add_argument("--person-max-area-ratio", type=float, default=0.0,
                     help=_help("Maximum bbox area as image-area ratio; 0 disables.", range_hint="0-1",
                                edge="use only when very large FPs dominate"))
    grp.add_argument("--detection-quality-scaling", action=argparse.BooleanOptionalAction, default=True,
                     help="Scale detection scores by a continuous quality factor (aspect+center+area).")
    grp.add_argument("--detection-quality-w-aspect", type=float, default=0.50,
                     help="Weight for aspect ratio in detection quality factor.")
    grp.add_argument("--detection-quality-w-center", type=float, default=0.30,
                     help="Weight for center bias (truncation) in detection quality factor.")
    grp.add_argument("--detection-quality-w-area", type=float, default=0.20,
                     help="Weight for area ratio in detection quality factor.")
    grp.add_argument("--geometry-suspect-support", action=argparse.BooleanOptionalAction, default=True,
                     help="Allow suspicious geometry only with extra tracker support.")
    grp.add_argument("--geometry-suspect-score", type=float, default=None,
                     help=_help("Override score gate for suspect geometry cases.", range_hint="0-1 or unset"))
    grp.add_argument("--nsa-kalman", action=argparse.BooleanOptionalAction, default=False,
                     help="Enable Noise Scale Adaptive Kalman: scale R by (1-score)^2 per detection.")
    grp.add_argument("--kalman-r-scale", type=float, default=0.75,
                     help=_help("Global Kalman measurement noise scale.", range_hint="0.1-2.0",
                                edge="too low causes jitter when detections are noisy"))
    grp.add_argument("--vel-dir-weight", type=float, default=0.0,
                     help=_help("OC-SORT velocity direction penalty weight. Penalises matches "
                                "where the detection direction opposes the track's Kalman velocity.",
                                range_hint="0.0-0.5", edge="too high rejects valid re-entries after direction change"))
    grp.add_argument("--fuse-score-weight", type=float, default=0.0,
                     help=_help("fuse_score weight (botsort-style): scales IoU by (1 - weight * det_score) "
                                "so low-confidence detections have higher association cost.",
                                range_hint="0.0-1.0", edge="1.0 = full botsort; reduces FP but may drop low-score recall"))
    grp.add_argument("--stage2-match-thresh", type=float, default=0.5,
                     help=_help("match_thresh for Stage 2 (low-conf dets → confirmed tracks). "
                                "Higher = stricter; reduces FP at cost of recall for low-score dets.",
                                range_hint="0.4-0.8", edge="too high blocks recovery of occluded tracks via low-conf dets"))
    grp.add_argument("--birth-low-score-thresh", type=float, default=0.0,
                     help=_help("Tracks born below this score require confirm_streak+1 hits before confirming. "
                                "0.0 = off (backward compat).",
                                range_hint="0.0-0.4", edge="too high delays valid low-score person detections"))
