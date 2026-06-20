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
    kalman_adapt_mode: int = (
        0  # 0=off, 1=score(legacy), 2=innovation(A), 3=lifestage(B), 4=aspect(C)
    )
    kalman_r_scale: float = 0.75
    vel_dir_weight: float = 0.0
    fuse_score_weight: float = 0.0
    stage2_match_thresh: float = 0.5
    birth_low_score_thresh: float = 0.0
    birth_prox_norm_thresh: float = 0.0  # NO-GO (2026-05-18): FP reduced but FN surged — proximity cannot distinguish ghost from real crowd

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
    grp.add_argument(
        "--crowd-track-thresh",
        type=float,
        default=0.02,
        help=_help(
            "Low-confidence association floor during crowd mode.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--crowd-mid-thresh",
        type=float,
        default=0.05,
        help=_help("Mid-tier confidence bucket during crowd mode.", range_hint="0-1"),
    )
    grp.add_argument(
        "--crowd-new-track-thresh",
        type=float,
        default=0.25,
        help=_help("New-track score threshold during crowd mode.", range_hint="0-1"),
    )
    grp.add_argument(
        "--geometry-mid-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive middle-scale geometry gating.",
    )
    grp.add_argument(
        "--geometry-ref-height-ratio",
        type=float,
        default=0.12,
        help=_help(
            "Reference person height ratio for geometry normalization.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--geometry-min-scale",
        type=float,
        default=0.875,
        help=_help(
            "Allowed lower scale multiplier around geometry reference.",
            range_hint=">0, usually 0.7-1.0",
        ),
    )
    grp.add_argument(
        "--geometry-max-scale",
        type=float,
        default=1.20,
        help=_help(
            "Allowed upper scale multiplier around geometry reference.",
            range_hint=">1, usually 1.0-1.5",
        ),
    )
    grp.add_argument(
        "--geometry-ema-beta",
        type=float,
        default=0.80,
        help=_help(
            "EMA smoothing for geometry reference updates.",
            range_hint="0-1",
            edge="near 1 is stable but slow to adapt",
        ),
    )
    grp.add_argument(
        "--geometry-loosen-step",
        type=float,
        default=0.08,
        help=_help("Per-update rate for relaxing geometry gates.", range_hint="0-1"),
    )
    grp.add_argument(
        "--geometry-tighten-step",
        type=float,
        default=0.03,
        help=_help("Per-update rate for tightening geometry gates.", range_hint="0-1"),
    )
    grp.add_argument(
        "--geometry-min-samples",
        type=int,
        default=5,
        help=_help(
            "Samples needed before geometry adaptation is trusted.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--id-stability-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter unstable identity handoffs with short-term spatial consistency checks.",
    )
    grp.add_argument(
        "--id-stability-min-hits",
        type=int,
        default=2,
        help=_help(
            "Minimum hits before stability checks are reliable.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--id-stability-min-iou",
        type=float,
        default=0.05,
        help=_help("Minimum IoU to treat a continuation as stable.", range_hint="0-1"),
    )
    grp.add_argument(
        "--id-stability-max-center-shift",
        type=float,
        default=2.0,
        help=_help(
            "Maximum normalized center shift for stable continuation.",
            range_hint=">0",
            edge="too low breaks fast movers, too high lets jumps through",
        ),
    )
    grp.add_argument(
        "--id-stability-max-gap",
        type=int,
        default=1,
        help=_help(
            "Maximum frame gap still considered short-term stable.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--id-stability-score-ema",
        type=float,
        default=0.70,
        help=_help("EMA decay for stability confidence.", range_hint="0-1"),
    )
    grp.add_argument(
        "--id-stability-min-score-ema",
        type=float,
        default=0.15,
        help=_help(
            "Minimum EMA score to keep a stable ID hypothesis.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--person-geometry-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hard geometric filtering for person boxes.",
    )
    grp.add_argument(
        "--person-min-height-ratio",
        type=float,
        default=0.018,
        help=_help(
            "Minimum bbox height as image-height ratio.",
            range_hint="0-1",
            edge="higher drops far-away pedestrians",
        ),
    )
    grp.add_argument(
        "--person-min-aspect",
        type=float,
        default=1.0,
        help=_help("Minimum h/w aspect ratio for person boxes.", range_hint=">0"),
    )
    grp.add_argument(
        "--person-max-aspect",
        type=float,
        default=5.5,
        help=_help(
            "Maximum h/w aspect ratio for person boxes.",
            range_hint="> person-min-aspect",
        ),
    )
    grp.add_argument(
        "--person-min-area-ratio",
        type=float,
        default=0.00006,
        help=_help("Minimum bbox area as image-area ratio.", range_hint="0-1"),
    )
    grp.add_argument(
        "--person-max-area-ratio",
        type=float,
        default=0.0,
        help=_help(
            "Maximum bbox area as image-area ratio; 0 disables.",
            range_hint="0-1",
            edge="use only when very large FPs dominate",
        ),
    )
    grp.add_argument(
        "--detection-quality-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale detection scores by a continuous quality factor (aspect+center+area).",
    )
    grp.add_argument(
        "--detection-quality-w-aspect",
        type=float,
        default=0.50,
        help="Weight for aspect ratio in detection quality factor.",
    )
    grp.add_argument(
        "--detection-quality-w-center",
        type=float,
        default=0.30,
        help="Weight for center bias (truncation) in detection quality factor.",
    )
    grp.add_argument(
        "--detection-quality-w-area",
        type=float,
        default=0.20,
        help="Weight for area ratio in detection quality factor.",
    )
    grp.add_argument(
        "--geometry-suspect-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow suspicious geometry only with extra tracker support.",
    )
    grp.add_argument(
        "--geometry-suspect-score",
        type=float,
        default=None,
        help=_help(
            "Override score gate for suspect geometry cases.", range_hint="0-1 or unset"
        ),
    )
    grp.add_argument(
        "--kalman-adapt-mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help=_help(
            "Kalman R adaptation mode: 0=off, 1=score(legacy NSA), "
            "2=innovation/d^2 outlier, 3=lifestage, 4=aspect jitter"
        ),
    )
    grp.add_argument(
        "--nsa-kalman",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated; use --kalman-adapt-mode 1 instead. (legacy NSA: score-based R scaling)",
    )
    grp.add_argument(
        "--kalman-r-scale",
        type=float,
        default=0.75,
        help=_help(
            "Global Kalman measurement noise scale.",
            range_hint="0.1-2.0",
            edge="too low causes jitter when detections are noisy",
        ),
    )
    grp.add_argument(
        "--vel-dir-weight",
        type=float,
        default=0.0,
        help=_help(
            "OC-SORT velocity direction penalty weight. Penalises matches "
            "where the detection direction opposes the track's Kalman velocity.",
            range_hint="0.0-0.5",
            edge="too high rejects valid re-entries after direction change",
        ),
    )
    grp.add_argument(
        "--fuse-score-weight",
        type=float,
        default=0.0,
        help=_help(
            "fuse_score weight (botsort-style): scales IoU by (1 - weight * det_score) "
            "so low-confidence detections have higher association cost.",
            range_hint="0.0-1.0",
            edge="1.0 = full botsort; reduces FP but may drop low-score recall",
        ),
    )
    grp.add_argument(
        "--oao-tau",
        type=float,
        default=0.0,
        dest="oao_tau",
        help=_help(
            "OA-SORT OAO penalty: cost += tau * inter-track-IoU for occluded tracks. "
            "Reduces cost confusion when tracks overlap during occlusion.",
            range_hint="0.0-0.4",
            edge="too high increases cost for dense scenes, hurts recall",
        ),
    )
    grp.add_argument(
        "--oao-contest-thresh",
        type=float,
        default=-1.0,
        dest="oao_contest_thresh",
        help=_help(
            "Contention gate for OAO. <0 (default) = plain OAO (penalise every "
            "detection of an overlapped track). >=0 = only penalise detections "
            "also claimed by the track's max-overlap partner (partner-pred IoU "
            ">= thresh), sparing uncontested side-by-side real pedestrians.",
            range_hint="-1 (off) or 0.3-0.5",
            edge="too low penalises almost everything (≈plain OAO); too high never fires",
        ),
    )
    grp.add_argument(
        "--oao-score-w",
        type=float,
        default=-1.0,
        dest="oao_score_w",
        help=_help(
            "Soft score weighting for OAO. <=0 (default) = full penalty. >0 = "
            "scale penalty by (1 - score_w * det_score), so confident detections "
            "get a reduced (not cut) penalty. 0.5 halves the penalty for a "
            "score-1.0 box; 1.0 zeroes it.",
            range_hint="-1 (off) or 0.3-0.7",
            edge="too high over-spares low-FP-density seqs; too low ≈ plain OAO",
        ),
    )
    grp.add_argument(
        "--oao-occ-mode",
        type=int,
        default=0,
        choices=[0, 1],
        dest="oao_occ_mode",
        help=_help(
            "OAO occlusion signal. 0 (default) = max single inter-track IoU. "
            "1 = union coverage (fraction of the track covered by the union of "
            "other boxes, 8x8 grid), amplifying dense crowds vs sparse overlaps.",
            range_hint="0 (max) or 1 (union)",
            edge="union raises penalty magnitude — pair with a lower oao-tau",
        ),
    )
    grp.add_argument(
        "--oao-crowd-radius",
        type=float,
        default=0.0,
        dest="oao_crowd_radius",
        help=_help(
            "OAO crowd multiplier radius (units of box height). <=0 (default) = off. "
            ">0 = scale penalty by (1 - 1/N), N = tracks within radius*h of t (incl. "
            "self): sparse overlaps (real side-by-side) damped, dense crowds full.",
            range_hint="0 (off) or ~0.5-1.0",
            edge="too small N→1 kills penalty everywhere; too large = no damping",
        ),
    )
    grp.add_argument(
        "--oao-height-gate",
        type=float,
        default=0.0,
        dest="oao_height_gate",
        help=_help(
            "OAO same-height gate. <=0 (default) = off. >0 = only partners with "
            "|h_t - h_j| <= gate*max(h) contribute to the OAO signal (same-depth "
            "occlusions only), sparing near/far projection overlaps.",
            range_hint="0 (off) or ~0.2-0.4",
            edge="too tight drops real same-depth occlusions; too loose ≈ no gate",
        ),
    )
    grp.add_argument(
        "--oao-foot-gate",
        type=float,
        default=0.0,
        dest="oao_foot_gate",
        help=_help(
            "OAO same-foot gate. <=0 (default) = off. >0 = only partners with "
            "|footy_t - footy_j| <= gate*h_ref contribute (truer same-depth proxy "
            "than height; matches the proven occ_front foot-gap signal).",
            range_hint="0 (off) or ~0.15-0.3",
            edge="too tight drops real same-depth occlusions; too loose ≈ no gate",
        ),
    )
    grp.add_argument(
        "--oao-ramp-frames",
        type=float,
        default=0.0,
        dest="oao_ramp_frames",
        help=_help(
            "OAO duration ramp. <=0 (default) = off. >0 = scale penalty by "
            "min(1, overlap_frames/ramp): transient crossings (sparse scenes) get "
            "a reduced penalty, persistent overlaps (crowds) reach the full penalty.",
            range_hint="0 (off) or ~15-30",
            edge="too small = no damping; too large over-spares medium crossings",
        ),
    )
    # Depth-gated occlusion-state machine (Occluded(by=A)); default off → bit-identical.
    grp.add_argument(
        "--occ-state-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="occ_state_enabled",
        help=_help(
            "Same-height occlusion gate: when two confirmed tracks overlap with similar "
            "foot depths, penalise the front (lower-foot) track for matching the back box "
            "to resolve crossing-swaps (no appearance/velocity cue).",
        ),
    )
    grp.add_argument(
        "--occ-iou-thresh",
        type=float,
        default=0.45,
        dest="occ_iou_thresh",
        help=_help(
            "Min track-track IoU for the same-height gate to activate.",
            range_hint="0.4-0.5",
        ),
    )
    grp.add_argument(
        "--occ-foot-gap",
        type=float,
        default=0.15,
        dest="occ_foot_gap",
        help=_help(
            "Max normalized foot-y gap for tracks to be at the *same depth* "
            "(occlusion crossing) — smaller = only true depth-plane collisions.",
            range_hint="0.10-0.20",
        ),
    )
    grp.add_argument(
        "--occ-ttl",
        type=int,
        default=4,
        dest="occ_ttl",
        help=_help(
            "Frames the front-track flag persists after the triggering overlap is gone.",
            range_hint="2-6",
        ),
    )
    grp.add_argument(
        "--occ-cost-weight",
        type=float,
        default=0.50,
        dest="occ_cost_weight",
        help=_help(
            "Penalty weight keeping the front occluder on its own box at re-acquisition.",
            range_hint="0.3-0.7",
        ),
    )
    grp.add_argument(
        "--multiplicative-cost",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="multiplicative_cost",
        help=_help(
            "Use log-linear cost form: cost = 1 - IoU * exp(-Σ penalty). "
            "No clamp between terms, supports reward signals via negative beta. "
            "Default off (additive, bit-identical backward compat).",
        ),
    )
    grp.add_argument(
        "--sinkhorn-lambda",
        type=float,
        default=30.0,
        dest="sinkhorn_lambda",
        help=_help(
            "Sinkhorn exponential temperature. Lower (10-15) gives softer "
            "discrimination; default 30. Only meaningful with --multiplicative-cost.",
            range_hint="5-30",
        ),
    )
    grp.add_argument(
        "--stability-cost-w",
        type=float,
        default=0.0,
        dest="stability_cost_w",
        help=_help(
            "Stability reward weight for multiplicative cost form. "
            "λ-normalized: effective bid boost = exp(stab_w). "
            "0.20 is the current best value.",
            range_hint="0.0-0.5",
        ),
    )
    grp.add_argument(
        "--stage2-match-thresh",
        type=float,
        default=0.5,
        help=_help(
            "match_thresh for Stage 2 (low-conf dets → confirmed tracks). "
            "Higher = stricter; reduces FP at cost of recall for low-score dets.",
            range_hint="0.4-0.8",
            edge="too high blocks recovery of occluded tracks via low-conf dets",
        ),
    )
    grp.add_argument(
        "--birth-low-score-thresh",
        type=float,
        default=0.0,
        help=_help(
            "Tracks born below this score require confirm_streak+1 hits before confirming. "
            "0.0 = off (backward compat).",
            range_hint="0.0-0.4",
            edge="too high delays valid low-score person detections",
        ),
    )
    grp.add_argument(
        "--birth-prox-norm-thresh",
        type=float,
        default=0.0,
        help=_help(
            "NO-GO (2026-05-18): Suppress new track birth near confirmed tracks. "
            "FP reduced but FN surged — proximity cannot distinguish ghost from real crowd. "
            "0.0 = off.",
            range_hint="0.0-1.0",
            edge="too high suppresses valid new persons in crowded scenes",
        ),
    )
