"""Track lifecycle (birth/death/confirm) config fields."""

# status: stable
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from saccade.perception.eval.assoc_basis import PERSON_HEIGHT_M, SCENE_FPS

from ._helpers import _help, _tier


@dataclasses.dataclass
class LifecycleConfig:
    # Birth gates (Experimental, related to track lifecycle)
    birth_quality_gate: bool = False
    birth_min_quality: float = 0.0
    birth_quality_score_bias: float = 0.15
    stage2_quality_gate: bool = False  # NO-GO (2026-05-11): overlapped with detection_quality_scaling, zero effect
    stage2_quality_min: float = 0.40
    birth_consecutive_gate: bool = False  # NO-GO (2026-05-18): statistically neutral, FP reduction cancelled by FN increase
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
    post_lifecycle_merge: bool = (
        False  # NO-GO (2026-04-27): confirmed harmful on baseline
    )
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
    # Cheb-GR offline tracklet merge (path 2; appearance stitching for AssA). Default off.
    cheb_gr_merge_enabled: bool = False
    cheb_gr_merge_max_cost: float = 0.55
    cheb_gr_merge_max_gap: int = 60
    cheb_gr_merge_min_overlap: int = 1
    cheb_gr_merge_n_samples: int = 50
    cheb_gr_pool_frac: float = 0.3
    cheb_gr_lambda: float = 2.0
    cheb_gr_k2: int = 6
    cheb_gr_max_fwd: int = 50
    cheb_gr_fuse_lambda: float = 0.3
    cheb_gr_engine: str = ""
    cheb_gr_model: str = "siglip2_reid"
    cheb_gr_online: bool = False
    cheb_gr_online_decide_n: int = 5
    cheb_gr_online_max_cost: float = 0.45
    cheb_gr_online_min_head: int = 1
    cheb_gr_online_margin: float = 0.0
    cheb_gr_online_key_sim_min: float = 0.0
    cheb_gr_online_key_sim_cost_floor: float = 0.0
    cheb_gr_online_key_margin_min: float = 0.0
    cheb_gr_online_center_dist_veto: float = 0.0
    cheb_gr_online_pollution_veto: float = 0.0
    cheb_gr_online_neighbor_iou_max: float = 0.0
    cheb_gr_online_bank_mode: str = "spread"
    cheb_gr_online_bank_n: int = 0
    cheb_gr_online_log: bool = False
    occ_audit: bool = False
    occ_audit_tau: float = 0.45
    occ_audit_ref_n: int = 5
    occ_audit_min_ref: int = 2
    occ_audit_crops: int = 3
    occ_audit_window: int = 30
    occ_audit_min_occ: int = 2
    occ_audit_log: bool = False
    # Cheb-GR graph decision probe for occ-exit (#55 WP1b). Default-off /
    # log-only: never changes cosine flag_frame / cuts / stats.flags.
    occ_audit_chebgr_probe: bool = False
    occ_audit_chebgr_max_cost: float = 0.45
    occ_audit_chebgr_margin: float = 0.0
    occ_audit_chebgr_pool_frac: float = 0.3
    occ_audit_chebgr_lambda: float = 2.0
    occ_audit_chebgr_k2: int = 6
    occ_audit_chebgr_max_fwd: int = 50
    occ_audit_chebgr_fuse_lambda: float = 0.3
    # Lost-track memory (ByteTrack track_buffer): frames a lost track survives in
    # the tracker (and stays available for association + bridge relink) before
    # removal. Per-seq fps-scaled when per_seq_adapt is on.
    track_buffer: int = 30
    # Birth-time lost-bank ReID relink (online, GPU). Revive a lost identity at
    # spawn time instead of minting a new id. Precision-first: ID consistency is
    # protected by a high sim threshold + spatial gate (a wrong revive = merging
    # two people = worse than a fragment). Default off.
    relink_enabled: bool = False
    relink_bank_cap: int = 256
    relink_sim_thresh: float = 0.6
    relink_lambda: float = 2.5
    relink_spatial_gate: float = 4.0
    relink_max_age: int = 300
    # GPU tracker-core bidirectional foot-bridge relink (Kalman-free, no ReID).
    # A young track that just stabilized adopts a still-live lost track's id by
    # regressing both ends' last/first 4 foot points and bridging at the midpoint.
    # Independent of relink_enabled (the bank-ReID path). Default off (bit-identical).
    relink_bridge_enabled: bool = False
    relink_bridge_px: float = 0.25
    relink_bridge_at: int = 4
    relink_bridge_min_lost: int = 2
    relink_bridge_ttl: int = 120
    relink_bridge_max_speed: float = 0.0
    relink_bridge_person_height: float = PERSON_HEIGHT_M
    relink_bridge_fps: float = SCENE_FPS
    relink_bridge_margin: float = 0.0
    relink_bridge_spatial_gate: float = 0.0
    relink_bridge_anchor: str = "adaptive"
    relink_bridge_anchor_rate: float = 0.03
    relink_bridge_h_lo: float = 0.0
    relink_bridge_h_hi: float = 0.0
    # Directional consistency bonus: relax bridge_px when lost and candidate
    # velocities point in the same direction (cos_sim^2 * gap_scale). 0 = off.
    # Suggested 0.3-1.0; higher values are increasingly permissive.
    relink_bridge_dir_bonus: float = 0.0
    # Gap-occupancy gates (long-gap bridges only): occ_cover = fraction of the
    # interpolated lost→cand gap path covered by other tracks' boxes. True
    # relinks have HIGHER coverage (occlusion explains the disappearance).
    relink_bridge_occ_gate_cover: float = 0.0
    relink_bridge_occ_gap_min: int = 30
    relink_bridge_occ_expand_px: float = 0.0
    relink_bridge_occ_expand_cover: float = 0.9
    # Appearance cosine veto on bridge candidates (ranking use of the ReID
    # embedding; needs reid_mode=tracker). <= -1 disables (bit-exact).
    relink_bridge_app_veto: float = -1.0
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
    multi_birth_replace_mode: bool = False
    multi_birth_replace_evidence_threshold: float = 0.85
    # Tracklet cleanup
    min_tracklet_len: int = 1
    min_tracklet_score: float = 0.0
    # Tracklet interpolation
    interpolate_tracklets: bool = True
    interpolate_max_gap: int = 20
    interpolate_min_track_len: int = 5
    interpolate_min_h: float = 0.0

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
        "--track-buffer",
        type=int,
        default=30,
        help=_help(
            "Lost-track memory (ByteTrack track_buffer): frames a lost track "
            "survives for association and bridge relink before removal. "
            "Per-seq fps-scaled when per-seq adaptation is on.",
            range_hint=">=1, default 30",
            edge="longer memory lets stale lost tracks steal associations",
        ),
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
        help="NO-GO (2026-05-11): Remove Stage 2 detections below quality-min. Zero effect — overlaps with detection_quality_scaling.",
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
        help="NO-GO (2026-05-18): Boost sub-threshold detections when appearing in consecutive frames. Statistically neutral.",
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
        help=_help(
            "Enable pre-output lifecycle merge. Offline/experimental ID merge — "
            "not headline production policy while reid_mode=off.",
            policy="LATENT",
        ),
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
        help=_help(
            "Post-output lifecycle merge. Confirmed harmful on baseline "
            "(2026-04-27) — do not enable as headline policy.",
            policy="NO-GO",
        ),
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
    grp.add_argument(
        "--interpolate-min-h",
        type=float,
        default=0.0,
        help=_help(
            "Minimum box height (px) for both sides of a gap to be interpolated.",
            range_hint="0-500",
            edge="100 filters ~77%% of wrong interpolations",
        ),
    )
    grp.add_argument(
        "--cheb-gr-merge-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cheb-GR offline tracklet merge (path 2): stitch temporally-disjoint "
        "tracklets by appearance to recover AssA. Re-crops img1 with "
        "--cheb-gr-model; uses native C++/CUDA crop + TRT extractor when "
        "available; mobilenetv4_reid uses the visclean occlusion crop gate.",
    )
    grp.add_argument(
        "--cheb-gr-merge-max-cost",
        type=float,
        default=0.55,
        help=_help(
            "Max Cheb-GR distance accepted for a tracklet merge.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--cheb-gr-merge-max-gap",
        type=int,
        default=60,
        help=_help(
            "Max frame gap between an earlier tracklet's end and a later one's start.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--cheb-gr-merge-min-overlap",
        type=int,
        default=1,
        help=_help(
            "Tracklets overlapping by more than this many frames never merge.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--cheb-gr-merge-n-samples",
        type=int,
        default=50,
        help=_help(
            "Temporally-distributed appearance samples kept per tracklet (20-100).",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--cheb-gr-pool-frac",
        type=float,
        default=0.3,
        help=_help(
            "Tracklet distance = mean of smallest this-fraction of cross-sample distances.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--cheb-gr-lambda",
        type=float,
        default=2.0,
        help=_help(
            "Chebyshev threshold lambda for the sample-level graph.", range_hint=">0"
        ),
    )
    grp.add_argument(
        "--cheb-gr-k2",
        type=int,
        default=6,
        help=_help(
            "k2 local query expansion for k-reciprocal re-ranking.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--cheb-gr-max-fwd",
        type=int,
        default=50,
        help=_help(
            "Cap on forward neighbours per sample (0 = pure adaptive).",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--cheb-gr-fuse-lambda",
        type=float,
        default=0.3,
        help=_help(
            "Weight on the Jaccard term vs original distance.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--cheb-gr-engine",
        default="",
        help="Optional ReID engine path for tracklet crops (default: siglip2_reid).",
    )
    grp.add_argument(
        "--cheb-gr-model",
        default="siglip2_reid",
        help="TRTFeatureExtractor model_type for Cheb-GR tracklet embeddings "
        "(e.g. mobilenetv4_reid).",
    )
    grp.add_argument(
        "--cheb-gr-offline-handover",
        "--cheb-gr-online",
        action=argparse.BooleanOptionalAction,
        dest="cheb_gr_online",
        default=False,
        help="Offline/output-layer Cheb-GR ID handover: after tracker output "
        "is complete, extract newborn heads and dead-track banks, then relabel "
        "eligible tracklets without feeding decisions back into the tracker. "
        "--cheb-gr-online is a deprecated alias.",
    )
    grp.add_argument(
        "--cheb-gr-offline-decide-n",
        "--cheb-gr-online-decide-n",
        dest="cheb_gr_online_decide_n",
        type=int,
        default=5,
        help=_help(
            "Frames after birth used as the newborn head window before an "
            "offline handover decision.",
            range_hint=">=1",
            edge="probe 2026-07-03: 5 already matches full offline evidence",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-max-cost",
        "--cheb-gr-online-max-cost",
        dest="cheb_gr_online_max_cost",
        type=float,
        default=0.45,
        help=_help(
            "Max Cheb-GR distance accepted for an offline handover (own knob: "
            "the offline merge config default 0.55 is too loose here).",
            range_hint="0-1",
            edge="0.45 = offline parity 80.3; 0.40 = all per-seq drawdowns <=0.2",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-min-head",
        "--cheb-gr-online-min-head",
        dest="cheb_gr_online_min_head",
        type=int,
        default=1,
        help=_help(
            "Minimum clean newborn head samples required before accepting an "
            "offline handover.",
            range_hint=">=1",
            edge="raise to 2+ to avoid decisions from a single clean crop",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-margin",
        "--cheb-gr-online-margin",
        dest="cheb_gr_online_margin",
        type=float,
        default=0.0,
        help=_help(
            "Minimum separation between best and second-best Cheb-GR candidate "
            "costs. 0 preserves the original top-1 behavior.",
            range_hint=">=0",
            edge="use 0.03-0.05 to reject ambiguous handovers",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-key-sim-min",
        "--cheb-gr-online-key-sim-min",
        dest="cheb_gr_online_key_sim_min",
        type=float,
        default=0.0,
        help=_help(
            "Reject an offline handover whose direct key-bank similarity "
            "support is below this cosine threshold (0 = off).",
            range_hint="0-1",
            edge="diagnostic 2026-07-04: key_best_sim is strong confirm evidence; "
            "combine with geometry before enabling as policy",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-key-margin-min",
        "--cheb-gr-online-key-margin-min",
        dest="cheb_gr_online_key_margin_min",
        type=float,
        default=0.0,
        help=_help(
            "Reject an offline handover whose direct key-bank hard-negative "
            "margin is below this threshold (0 = off). No-hard-negative events "
            "use sentinel margin 999 and pass.",
            range_hint=">=0",
            edge="diagnostic 2026-07-04: key_margin <=0.03 was a strong "
            "ambiguity veto on m-substrate; validate cross-condition before defaulting",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-key-sim-cost-floor",
        "--cheb-gr-online-key-sim-cost-floor",
        dest="cheb_gr_online_key_sim_cost_floor",
        type=float,
        default=0.0,
        help=_help(
            "Only apply key-sim veto when Cheb-GR best_cost is at least this "
            "value (0 = apply key-sim veto to all accepted-cost candidates).",
            range_hint=">=0",
            edge="use 0.25-0.30 to avoid vetoing very strong Cheb-GR matches",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-center-dist-veto",
        dest="cheb_gr_online_center_dist_veto",
        type=float,
        default=0.0,
        help=_help(
            "Reject an offline handover whose forward-projected candidate "
            "center is >= this many avg box heights from the newborn center "
            "(0 = off).",
            range_hint=">=0",
            edge="applicability map 2026-07-04: >=2 is a stable veto (m/s)",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-pollution-veto",
        dest="cheb_gr_online_pollution_veto",
        type=float,
        default=0.0,
        help=_help(
            "Reject an offline handover whose newborn-head / candidate-tail "
            "crops overlap another track at or above this IoU (0 = off).",
            range_hint="0-1",
            edge="applicability map 2026-07-04: >=0.5 = stable-high-pollution",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-neighbor-iou-max",
        dest="cheb_gr_online_neighbor_iou_max",
        type=float,
        default=0.0,
        help=_help(
            "Drop head/bank crop samples whose box overlaps another track at "
            "or above this IoU before embedding extraction (0 = off). Head is "
            "filtered strictly; a bank falls back to unfiltered clean crops.",
            range_hint="0-1",
            edge="crowd-pollution proxy; complements the front-occlusion gate",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-bank-mode",
        dest="cheb_gr_online_bank_mode",
        choices=("spread", "recent"),
        default="spread",
        help=_help(
            "Dead-track bank construction for the offline handover: 'spread' "
            "= temporally-distributed samples over the whole life; 'recent' = "
            "most recent clean samples before death (visclean-gated FIFO).",
            edge="probe 2026-07-04: recent-20 +0.07/+0.05 vs dense on m/s, "
            "highest accept-set fidelity; pair with --cheb-gr-offline-bank-n 20",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-bank-n",
        dest="cheb_gr_online_bank_n",
        type=int,
        default=0,
        help=_help(
            "Bank sample budget for the offline handover (0 = use "
            "--cheb-gr-merge-n-samples).",
            range_hint=">=0",
            edge="recent mode: 20 is the validated safe point; 10 and below degrade",
        ),
    )
    grp.add_argument(
        "--cheb-gr-offline-log",
        "--cheb-gr-online-log",
        action=argparse.BooleanOptionalAction,
        dest="cheb_gr_online_log",
        default=False,
        help="Write _cheb_gr_offline_handover.csv with accept/reject decision rows.",
    )
    grp.add_argument(
        "--occ-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Causal occ-exit identity audit (ABSORB-side twin of "
        "--cheb-gr-offline-handover): after a geometric occlusion episode ends, compare "
        "the track's first clean crops against its pre-occlusion reference and "
        "split off a fresh id from the first crop that contradicts it. Shares "
        "the --cheb-gr-engine/--cheb-gr-model extractor.",
    )
    grp.add_argument(
        "--occ-audit-tau",
        type=float,
        default=0.45,
        help=_help(
            "Cosine below this flags an identity transfer (clean-stream quantile).",
            range_hint="0-1",
            edge="probe 2026-07-03: 0.386 = 0.1%% false rate, 0.498 = 0.5%%",
        ),
    )
    grp.add_argument(
        "--occ-audit-ref-n",
        type=int,
        default=5,
        help=_help(
            "Clean pre-occlusion frames kept as the appearance reference.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--occ-audit-min-ref",
        type=int,
        default=2,
        help=_help(
            "Minimum clean reference samples; episodes with fewer abstain.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--occ-audit-crops",
        type=int,
        default=3,
        help=_help(
            "Post-exit clean crops checked per episode (min-of-N decision).",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--occ-audit-window",
        type=int,
        default=30,
        help=_help(
            "Frames after occlusion exit in which audit crops may be collected.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--occ-audit-min-occ",
        type=int,
        default=2,
        help=_help(
            "Minimum consecutive dirty frames for an episode to be audited "
            "(single-frame blips are skipped).",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--occ-audit-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write _occ_audit.csv with one row per audited episode.",
    )
    grp.add_argument(
        "--occ-audit-chebgr-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "RESEARCH/DEBUG: log-only Cheb-GR graph decision probe on the "
            "bank-reference occ-exit path. Appends chebgr_* columns to "
            "_occ_audit.csv; does not change cosine flag_frame / cuts / "
            "stats.flags. Requires --occ-audit-bank-reference for the bank path."
        ),
    )
    grp.add_argument(
        "--occ-audit-chebgr-max-cost",
        type=float,
        default=0.45,
        help=_help(
            "Cheb-GR probe: flag when audit↔own-ref graph cost exceeds this "
            "(high self_cost = post-exit no longer matches pre-episode ref).",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--occ-audit-chebgr-margin",
        type=float,
        default=0.0,
        help=_help(
            "Cheb-GR probe: if >0 and occluder ref exists, require "
            "self_cost - occluder_cost >= this before flagging.",
            range_hint=">=0; 0 disables margin gate",
        ),
    )
    grp.add_argument(
        "--occ-audit-chebgr-pool-frac",
        type=float,
        default=0.3,
        help=_help(
            "Cheb-GR probe: mean of the smallest pool_frac cross-block distances.",
            range_hint="(0,1]",
        ),
    )
    grp.add_argument(
        "--occ-audit-chebgr-lambda",
        type=float,
        default=2.0,
        help=_help("Cheb-GR probe: Chebyshev neighbourhood lambda.", range_hint=">0"),
    )
    grp.add_argument(
        "--occ-audit-chebgr-k2",
        type=int,
        default=6,
        help=_help("Cheb-GR probe: k2 local query expansion.", range_hint=">=1"),
    )
    grp.add_argument(
        "--occ-audit-chebgr-max-fwd",
        type=int,
        default=50,
        help=_help("Cheb-GR probe: max forward neighbourhood size.", range_hint=">=1"),
    )
    grp.add_argument(
        "--occ-audit-chebgr-fuse-lambda",
        type=float,
        default=0.3,
        help=_help(
            "Cheb-GR probe: Jaccard fusion weight on the distance.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--relink-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_help(
            "Birth-time lost-bank ReID relink: revive a lost identity at spawn "
            "instead of minting a new id (online, GPU). Needs reid embeddings; "
            "headline path uses geometry bridge only (reid_mode=off).",
            policy="LATENT",
        ),
    )
    grp.add_argument(
        "--relink-bank-cap",
        type=int,
        default=256,
        help=_help(
            "Capacity (LRU ring) of the lost-identity embedding bank.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--relink-sim-thresh",
        type=float,
        default=0.6,
        help=_help(
            "Cosine floor to revive a lost id. High = precision-first (avoid merging two people).",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--relink-lambda",
        type=float,
        default=2.5,
        help=_help(
            "Chebyshev threshold lambda: revive only if distance <= mu - lambda*sigma "
            "(statistically exclusive match). Higher = stricter/fewer revives.",
            range_hint="2.0-3.0",
        ),
    )
    grp.add_argument(
        "--relink-spatial-gate",
        type=float,
        default=4.0,
        help=_help(
            "Base radius gamma (track-height units); full gate is gamma*h + 0.8*|v|*lost_age.",
            range_hint=">0",
        ),
    )
    grp.add_argument(
        "--relink-max-age",
        type=int,
        default=300,
        help=_help(
            "Hard cap (frames) a lost identity stays revivable; bounds false cross-time revives.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--relink-bridge-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="GPU tracker-core bidirectional foot-bridge relink (Kalman-free, no "
        "ReID): a young track that just stabilized adopts a still-live lost id by "
        "regressing both ends' 4 foot points and bridging at the midpoint. "
        "Independent of --relink-enabled.",
    )
    grp.add_argument(
        "--relink-bridge-px",
        type=float,
        default=0.25,
        help=_help(
            "Box-height-normalized meeting distance to accept a speed-weighted "
            "foot-bridge. Smaller = stricter. MOT17-SDP optimum is a 0.25-0.30 "
            "plateau (IDF1 74.8/HOTA 68.0/AssA 66.2 at 0.25); <=0.2 over-tightens, "
            ">=0.4 over-bridges. See offline_relink_candidate_analysis.md §6d.",
            range_hint=">0, suggested 0.25-0.3",
        ),
    )
    grp.add_argument(
        "--relink-bridge-at",
        type=int,
        default=4,
        help=_help(
            "hit_streak at which a young track first attempts a bridge (fires once).",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--relink-bridge-min-lost",
        type=int,
        default=2,
        help=_help(
            "Minimum coasting age of a lost track to be a bridge target.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--relink-bridge-ttl",
        type=int,
        default=120,
        help=_help(
            "Maximum coasting age of a lost track to remain a bridge target.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--relink-bridge-max-speed",
        type=float,
        default=0.0,
        help=_help(
            "Physical speed gate (m/s) on the bridge endpoints. 0 disables.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-person-height",
        type=float,
        default=PERSON_HEIGHT_M,
        help=_help(
            "Assumed person height (m) for px-per-m in the speed gate.",
            range_hint=">0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-fps",
        type=float,
        default=SCENE_FPS,
        help=_help(
            "Sequence FPS used by the speed gate to convert age to seconds.",
            range_hint=">0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-margin",
        type=float,
        default=0.0,
        help=_help(
            "Reciprocal margin: reject if (2nd-best - best) bridge distance is below "
            "this (ambiguous). 0 disables.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-spatial-gate",
        type=float,
        default=0.0,
        help=_help(
            "Optional center-distance/h_ref gate before the bridge test. 0 disables.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-anchor",
        choices=["center", "foot", "adaptive"],
        default="adaptive",
        help="Foot-bridge anchor point: 'center' (box centre, legacy), 'foot' "
        "(bottom edge / ground-contact), or 'adaptive' (default; residual-weighted "
        "blend of top/bottom edges so an occlusion-clipped edge is down-weighted; "
        "degrades to centre when neither edge deforms).",
    )
    grp.add_argument(
        "--relink-bridge-anchor-rate",
        type=float,
        default=0.03,
        help=_help(
            "Adaptive-anchor deformation gate: only re-anchor on edges when a "
            "window's mean |Δh|/h̄ exceeds this; stable boxes keep the centre "
            "(reduces FP from perturbing clean detections). 0 = always-on. "
            "MOT17-SDP sweet spot 0.03 (dominates the un-gated anchor on FP+FN).",
            range_hint=">=0, suggested 0.02-0.10",
        ),
    )
    grp.add_argument(
        "--relink-bridge-h-lo",
        type=float,
        default=0.0,
        help=_help(
            "Bridge scale gate lower bound: reject when lost/cand EMA-height "
            "ratio falls below this. Offline: [0.75,1.33] kills 53%% of wrong "
            "relinks at zero short-gap TP loss.",
            range_hint="0-1, suggested 0.75; needs --relink-bridge-h-hi",
        ),
    )
    grp.add_argument(
        "--relink-bridge-h-hi",
        type=float,
        default=0.0,
        help=_help(
            "Bridge scale gate upper bound: reject when lost/cand EMA-height "
            "ratio exceeds this. 0 disables the gate.",
            range_hint="0 or >=1, suggested 1.33",
            edge="lost TPs are all gap>=37 long-gap bridges near the band edge",
        ),
    )
    grp.add_argument(
        "--relink-bridge-dir-bonus",
        type=float,
        default=0.0,
        help=_help(
            "Directional consistency bonus for bridge px threshold. When lost "
            "and candidate velocities point in the same direction, the effective "
            "bridge_px is relaxed by bonus * cos_sim^2 * gap_scale. 0 = off.",
            range_hint=">=0, suggested 0.3-1.0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-occ-gate-cover",
        type=float,
        default=0.0,
        help=_help(
            "Gap-occupancy veto: reject long-gap bridges whose interpolated gap "
            "path has occ_cover below this (an unexplained disappearance is "
            "likely a different person). Offline @0.5: -53 wrong / -1 correct. "
            "0 disables.",
            range_hint="0-1, suggested 0.5",
        ),
    )
    grp.add_argument(
        "--relink-bridge-occ-gap-min",
        type=int,
        default=30,
        help=_help(
            "Occupancy gates apply only to gaps >= this many frames; short-gap "
            "occ_cover is noise (crowd AUC 0.44, would kill true positives).",
            range_hint=">=1, suggested 30",
        ),
    )
    grp.add_argument(
        "--relink-bridge-occ-expand-px",
        type=float,
        default=0.0,
        help=_help(
            "Tiered expansion: long-gap bridges with occ_cover >= "
            "--relink-bridge-occ-expand-cover are accepted up to this looser "
            "bridge score instead of --relink-bridge-px. Offline tiered "
            "(0.3/1.0 @ occ>=0.9): +15 correct at held precision. 0 disables.",
            range_hint="> bridge_px, suggested 0.5-1.0",
        ),
    )
    grp.add_argument(
        "--relink-bridge-occ-expand-cover",
        type=float,
        default=0.9,
        help=_help(
            "Minimum occ_cover to unlock the expanded bridge threshold.",
            range_hint="0-1, suggested 0.9",
        ),
    )
    grp.add_argument(
        "--relink-bridge-app-veto",
        type=float,
        default=-1.0,
        help=_help(
            "Appearance cosine veto for bridge candidates: a geometry-plausible "
            "(lost, cand) pair is dropped when both tracks have reference "
            "embeddings and cosine < this floor (ranking use: the correct "
            "runner-up then wins the bdist ranking). Needs per-detection "
            "embeddings (reid_mode=tracker). <= -1 disables (bit-exact).",
            range_hint="mnv4 scale: 0.20-0.25 (in-zone false-kill 35-43%% at 1-2%% true cost)",
        ),
    )
