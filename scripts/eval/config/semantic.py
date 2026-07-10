from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class SemanticConfig:
    # Core semantic relink
    semantic_threshold: float = 0.91
    semantic_ttl: int = 45
    semantic_ema: float = 0.83
    semantic_spatial_gate: float = 0.20
    semantic_min_lost_frames: int = 2
    semantic_min_iou: float = 0.20
    semantic_mahalanobis_threshold: float = 0.0
    # Kalman probabilistic relink gate
    semantic_kalman_gate: bool = False
    semantic_kalman_chi2: float = 9.4877
    semantic_kalman_penalty_weight: float = 0.0
    semantic_kalman_dir_min_cos: float = -1.0
    semantic_kalman_dir_min_speed: float = 1.0
    semantic_kalman_person_height_m: float = 0.0
    semantic_kalman_accel_long: float = 2.0
    semantic_kalman_accel_lat: float = 1.0
    semantic_kalman_fps: float = 0.0
    semantic_kalman_max_speed_mps: float = 0.0
    semantic_delayed_claim: bool = False
    semantic_claim_warmup_frames: int = 3
    semantic_cheb_gr_claim: bool = False
    semantic_cheb_gr_max_cost: float = 0.45
    semantic_cheb_gr_margin: float = 0.05
    semantic_cheb_gr_min_head: int = 2
    semantic_cheb_gr_pool_frac: float = 0.3
    semantic_cheb_gr_min_sim: float = 0.0
    semantic_gpu_relink_gate: bool | None = None
    semantic_gpu_relink_gate_graph: bool = True
    semantic_gpu_relink_gate_init_query_cap: int = 64
    semantic_gpu_relink_gate_init_candidate_cap: int = 128
    semantic_cheb_gr_graph_init_cap: int = 32
    semantic_bidirectional: bool = False
    semantic_bridge_px: float = 350.0
    semantic_debug: bool = False
    # Reference buffer
    semantic_buffer_size: int = 10
    semantic_min_consistency: float = 0.0
    semantic_rerank_mode: str = "mean"
    semantic_reciprocal_margin: float = 0.0
    # Unified score weights (A1)
    semantic_iou_weight: float = 0.0
    semantic_mahalanobis_weight: float = 0.0
    semantic_dynamic_margin_crowd: float = 0.0
    semantic_dynamic_margin_age: float = 0.0
    semantic_biometric_threshold: float = 0.0
    semantic_w_sim_base: float = 0.80
    semantic_w_iou_base: float = 0.34
    semantic_w_maha_base: float = 0.31
    semantic_shift_ambiguity: float = 0.34
    semantic_shift_lost_age: float = 0.18
    # Python relinker controls
    force_python_relinker: bool = False
    semantic_experimental_mode: str = "standard"
    semantic_appearance_first_sim_threshold: float = 0.95
    semantic_appearance_first_margin: float = 0.03
    # Experimental density gating controls
    semantic_exp_density_gating: bool = False
    semantic_exp_density_k: float = 2.0
    semantic_exp_density_eta: float = 0.15
    # Appearance bank
    semantic_bank_inject: bool = False
    appearance_bank: bool = False
    appearance_bank_size: int = 5
    appearance_bank_min_score: float = 0.45
    appearance_bank_min_iou: float = 0.35
    appearance_bank_consistency_threshold: float = 0.75
    appearance_bank_high_quality_min_score: float = 0.75
    appearance_bank_min_aspect: float = 1.2
    appearance_bank_max_aspect: float = 4.5
    appearance_occlusion_gate: bool = False
    appearance_occlusion_cov: float = 0.4
    # Bank quality scoring (A6)
    bank_quality_v2: bool = True
    bank_weighted_mean: bool = False
    bank_quality_w_det: float = 0.45
    bank_quality_w_iou: float = 0.20
    bank_quality_w_aspect: float = 0.15
    bank_quality_w_center: float = 0.10
    bank_quality_w_area: float = 0.10
    # False-accept filter (Phase 3)
    semantic_clean_score_threshold: float = 0.65
    semantic_clean_min_aspect: float = 1.2
    semantic_clean_max_aspect: float = 4.5
    semantic_clean_margin_ratio: float = 0.0
    semantic_strict_sim_threshold: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_semantic_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group(
        _tier("Semantic relink and appearance memory", "Tier 1/2")
    )
    grp.description = (
        "Higher-level relinking and appearance-bank logic for recovering IDs after gaps. "
        "Load with: --module-semantic configs/modules/semantic.yaml"
    )
    grp.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.91,
        help=_help("Similarity gate for semantic relinking.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-ttl",
        type=int,
        default=45,
        help=_help(
            "Frames to keep lost tracks in semantic memory.",
            range_hint=">=1",
            edge="larger helps long occlusion but raises stale-ID risk",
        ),
    )
    grp.add_argument(
        "--semantic-ema",
        type=float,
        default=0.83,
        help=_help("EMA decay for semantic appearance updates.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-spatial-gate",
        type=float,
        default=0.20,
        help=_help(
            "Spatial gate for semantic relink candidates.",
            range_hint=">=0",
            edge="too small blocks cross-lane recovery",
        ),
    )
    grp.add_argument(
        "--semantic-min-lost-frames",
        type=int,
        default=2,
        help=_help(
            "Track must be lost this many frames before semantic relink is considered.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-min-iou",
        type=float,
        default=0.20,
        help=_help("Minimum IoU for semantic relink acceptance.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-mahalanobis-threshold",
        type=float,
        default=0.0,
        help=_help(
            "Optional motion gate for semantic relink; 0 disables.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--semantic-kalman-gate",
        action="store_true",
        help=(
            "Replace the static spatial gate with a Kalman chi-square gate: lost "
            "tracks are velocity-extrapolated and their covariance inflated with "
            "time, gating relink candidates by Mahalanobis distance to that cloud."
        ),
    )
    grp.add_argument(
        "--semantic-kalman-chi2",
        type=float,
        default=9.4877,
        help=_help(
            "Chi-square gate threshold for the Kalman relink gate "
            "(4 DoF: 9.4877=95%%, 13.28=99%%).",
            range_hint=">0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-penalty-weight",
        type=float,
        default=0.0,
        help=_help(
            "Weight of the spatial probability penalty 1-exp(-D^2/2) added to the "
            "joint relink score (0 = pure hard gate).",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-dir-min-cos",
        type=float,
        default=-1.0,
        help=_help(
            "Velocity-direction gate: reject relink candidates whose displacement "
            "from the lost track has cos(angle, velocity) below this (e.g. 0.0 "
            "rejects anything >90 deg behind). <=-1 disables. Needs --semantic-kalman-gate.",
            range_hint="-1..1",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-dir-min-speed",
        type=float,
        default=1.0,
        help=_help(
            "Minimum track speed (px/frame) before the direction gate applies; "
            "near-stationary tracks have unreliable direction.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-person-height-m",
        type=float,
        default=0.0,
        help=_help(
            "Assumed real person height (m) for physically-grounded covariance "
            "diffusion: px/m scale = box_height/this. >0 enables the physical model "
            "(replaces tracker Q). Needs --semantic-kalman-gate.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-accel-long",
        type=float,
        default=2.0,
        help=_help(
            "Max longitudinal (speed-change) human acceleration (m/s^2) for the "
            "physical diffusion; larger → cloud stretches more along motion.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-accel-lat",
        type=float,
        default=1.0,
        help=_help(
            "Max lateral (turning) human acceleration (m/s^2); smaller than "
            "accel-long encodes velocity inertia (tight sideways).",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-fps",
        type=float,
        default=0.0,
        help=_help(
            "Frame rate to convert m/s^2 acceleration to px/frame^2 for the physical "
            "diffusion. 0 = auto-resolve per sequence (seqinfo.ini frameRate → .mp4 "
            "probe → 30). Set >0 to override.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-kalman-max-speed-mps",
        type=float,
        default=0.0,
        help=_help(
            "Physical reachability cap (m/s): reject a relink whose implied average "
            "speed (dist/time via person-height px/m scale) exceeds this (human "
            "sprint ~8). Snapshot-independent, always on. 0 disables. Needs "
            "--semantic-kalman-person-height-m and --semantic-kalman-fps.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-delayed-claim",
        action="store_true",
        help=(
            "Delay first-time semantic relink claims for new raw track IDs until "
            "they have survived the warmup window, then retroactively remap output "
            "lines before interpolation. Default off."
        ),
    )
    grp.add_argument(
        "--semantic-claim-warmup-frames",
        type=int,
        default=3,
        help=_help(
            "Frames a new raw ID must survive before semantic delayed-claim "
            "arbitration runs.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-claim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Cheb-GR multi-sample scoring for delayed-claim relink. "
            "Requires ReID embeddings and forces the Python delayed-claim relinker."
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-max-cost",
        type=float,
        default=0.45,
        help=_help(
            "Max Cheb-GR distance accepted by semantic delayed-claim relink.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-margin",
        type=float,
        default=0.05,
        help=_help(
            "Minimum separation between best and second-best Cheb-GR claim costs.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-min-head",
        type=int,
        default=2,
        help=_help(
            "Minimum clean warmup embeddings required before a Cheb-GR claim.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-pool-frac",
        type=float,
        default=0.3,
        help=_help(
            "Fraction of lowest head-bank distances averaged for Cheb-GR claim cost.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-min-sim",
        type=float,
        default=0.0,
        help=_help(
            "Minimum raw cosine similarity required before Cheb-GR may accept a "
            "delayed-claim relink candidate.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--semantic-gpu-relink-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the CUDA relink gate kernel for semantic relink candidate gates. "
            "Default auto-enables for --semantic-cheb-gr-claim."
        ),
    )
    grp.add_argument(
        "--semantic-gpu-relink-gate-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture the CUDA relink gate kernel in a fixed-capacity CUDA graph "
            "when semantic GPU relink gate is enabled."
        ),
    )
    grp.add_argument(
        "--semantic-gpu-relink-gate-init-query-cap",
        type=int,
        default=64,
        help=_help(
            "Initial fixed CUDA graph query capacity for semantic relink gates.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-gpu-relink-gate-init-candidate-cap",
        type=int,
        default=128,
        help=_help(
            "Initial fixed CUDA graph candidate capacity for semantic relink gates.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-cheb-gr-graph-init-cap",
        type=int,
        default=32,
        help=_help(
            "Initial fixed CUDA graph sample capacity for Cheb-GR delayed claims.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-bidirectional",
        action="store_true",
        help=(
            "Bidirectional Kalman gate: in addition to the forward cloud (lost track "
            "→ candidate), also check the backward cloud (candidate → lost track's "
            "last position). Both chi-square tests must pass. "
            "Needs --semantic-kalman-gate."
        ),
    )
    grp.add_argument(
        "--semantic-bridge-px",
        type=float,
        default=350.0,
        help=_help(
            "Bidirectional midpoint bridge gate (px): forward and backward Kalman "
            "clouds are each propagated to gap/2 through the full predict model, "
            "and the Euclidean distance between the two foot-point centres is "
            "compared against this threshold.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-debug",
        action="store_true",
        help="Print extra semantic relink diagnostics.",
    )
    grp.add_argument(
        "--semantic-buffer-size",
        type=int,
        default=10,
        help=_help("Appearance vectors kept for semantic reference.", range_hint=">=1"),
    )
    grp.add_argument(
        "--semantic-min-consistency",
        type=float,
        default=0.0,
        help=_help(
            "Minimum internal consistency for semantic references.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--semantic-rerank-mode",
        choices=("mean", "max", "top2_mean", "weighted"),
        default="mean",
        help="How to aggregate multiple reference embeddings.",
    )
    grp.add_argument(
        "--semantic-reciprocal-margin",
        type=float,
        default=0.0,
        help=_help(
            "Reject relink if top-1 and top-2 similarities are too close; 0 disables.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-iou-weight",
        type=float,
        default=0.0,
        help=_help(
            "Blend IoU into joint ranking score (0 = pure cosine).", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--semantic-mahalanobis-weight",
        type=float,
        default=0.0,
        help=_help(
            "Blend normalised motion evidence into joint ranking score (0 = off).",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-dynamic-margin-crowd",
        type=float,
        default=0.0,
        help=_help("Add this × crowd_factor to reciprocal margin.", range_hint=">=0"),
    )
    grp.add_argument(
        "--semantic-dynamic-margin-age",
        type=float,
        default=0.0,
        help=_help(
            "Add this × (lost_frames/ttl) to reciprocal margin.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--semantic-biometric-threshold",
        type=float,
        default=0.0,
        help=_help(
            "YOLO Pose biometric veto: reject relink when body-ratio distance exceeds this; 0 disables.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-w-sim-base",
        type=float,
        default=0.80,
        help=_help("A1: Base weight for appearance similarity.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-w-iou-base",
        type=float,
        default=0.34,
        help=_help("A1: Base weight for spatial IoU.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-w-maha-base",
        type=float,
        default=0.31,
        help=_help("A1: Base weight for Mahalanobis score.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-shift-ambiguity",
        type=float,
        default=0.34,
        help=_help(
            "A1: Shift weight from IoU to Sim under crowd ambiguity.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--semantic-shift-lost-age",
        type=float,
        default=0.18,
        help=_help(
            "A1: Shift weight from IoU to Sim as lost age increases.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--force-python-relinker",
        action="store_true",
        default=False,
        help="Force Python relinker path (confound control).",
    )
    grp.add_argument(
        "--semantic-experimental-mode",
        choices=("standard", "appearance_first"),
        default="standard",
        help="Experimental Python relinker mode.",
    )
    grp.add_argument(
        "--semantic-appearance-first-sim-threshold",
        type=float,
        default=0.95,
        help=_help("High-sim threshold for appearance_first bypass.", range_hint="0-1"),
    )
    grp.add_argument(
        "--semantic-appearance-first-margin",
        type=float,
        default=0.03,
        help=_help(
            "Minimum winner-vs-runner margin for appearance_first bypass.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-bank-inject",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject TrackAppearanceBank representative when a track dies.",
    )
    grp.add_argument(
        "--appearance-bank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Top-K appearance bank for primary association (Phase 1).",
    )
    grp.add_argument(
        "--appearance-bank-size",
        type=int,
        default=5,
        help=_help(
            "Top-K samples kept per track in the appearance bank.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--appearance-bank-min-score",
        type=float,
        default=0.45,
        help=_help("Minimum detection score to admit a bank sample.", range_hint="0-1"),
    )
    grp.add_argument(
        "--appearance-bank-min-iou",
        type=float,
        default=0.35,
        help=_help(
            "Minimum track/detection IoU to admit a bank sample.", range_hint="0-1"
        ),
    )
    grp.add_argument(
        "--appearance-bank-consistency-threshold",
        type=float,
        default=0.75,
        help=_help(
            "Minimum pairwise cosine mean to trust the bank for relink.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--appearance-bank-high-quality-min-score",
        type=float,
        default=0.75,
        help=_help(
            "Minimum score for a bank sample to qualify as high-quality (Phase 3).",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--appearance-bank-min-aspect",
        type=float,
        default=1.2,
        help=_help(
            "Minimum h/w aspect for a high-quality bank sample.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--appearance-bank-max-aspect",
        type=float,
        default=4.5,
        help=_help(
            "Maximum h/w aspect for a high-quality bank sample.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--appearance-occlusion-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip appearance extraction and bank updates for boxes covered by a "
            "lower-foot foreground box."
        ),
    )
    grp.add_argument(
        "--appearance-occlusion-cov",
        type=float,
        default=0.4,
        help=_help(
            "Coverage threshold for --appearance-occlusion-gate; matches "
            "MobileNetV4 visclean training --occ-cov.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--bank-quality-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use composite quality score (det+iou+aspect+center+area) for bank sample ranking.",
    )
    grp.add_argument(
        "--bank-weighted-mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_help(
            "Weight bank samples by quality_score when computing the representative embedding. "
            "Low-quality samples contribute less to the mean.",
            range_hint="on/off",
            edge="may slightly tighten consistency estimate; test with appearance-bank on",
        ),
    )
    grp.add_argument(
        "--bank-quality-w-det",
        type=float,
        default=0.45,
        help="Weight for detection score in composite quality score.",
    )
    grp.add_argument(
        "--bank-quality-w-iou",
        type=float,
        default=0.20,
        help="Weight for IoU in composite quality score.",
    )
    grp.add_argument(
        "--bank-quality-w-aspect",
        type=float,
        default=0.15,
        help="Weight for aspect ratio in composite quality score.",
    )
    grp.add_argument(
        "--bank-quality-w-center",
        type=float,
        default=0.10,
        help="Weight for center bias in composite quality score.",
    )
    grp.add_argument(
        "--bank-quality-w-area",
        type=float,
        default=0.10,
        help="Weight for area ratio in composite quality score.",
    )
    grp.add_argument(
        "--semantic-clean-score-threshold",
        type=float,
        default=0.65,
        help=_help(
            "Detection score below which observation is treated as low-quality "
            "(triggers strict_sim gate, Phase 3).",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--semantic-clean-min-aspect",
        type=float,
        default=1.2,
        help=_help(
            "Minimum h/w aspect for current observation to be treated as clean.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-clean-max-aspect",
        type=float,
        default=4.5,
        help=_help(
            "Maximum h/w aspect for current observation to be treated as clean.",
            range_hint=">=0",
        ),
    )
    grp.add_argument(
        "--semantic-clean-margin-ratio",
        type=float,
        default=0.0,
        help=_help(
            "Frame-edge margin: boxes touching within this fraction are low-quality; 0=disabled.",
            range_hint="0-0.15",
        ),
    )
    grp.add_argument(
        "--semantic-strict-sim-threshold",
        type=float,
        default=0.0,
        help=_help(
            "Similarity threshold for low-quality observations (0=use sim_threshold).",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--semantic-exp-density-gating",
        action="store_true",
        help="Enable experimental crowd-density adaptive Mahalanobis gating.",
    )
    grp.add_argument(
        "--semantic-exp-density-k",
        type=float,
        default=2.0,
        help="Crowd-density window multiplier of pedestrian height.",
    )
    grp.add_argument(
        "--semantic-exp-density-eta",
        type=float,
        default=0.15,
        help="Crowd-density exponential decay multiplier for Mahalanobis threshold.",
    )
    # ── M-B1 research-only default-off portable OR-tail hook (Stage 1) ──
    # Not a production preset knob. Unset/empty = hook disabled (zero behavior change).
    research = parser.add_argument_group(
        _tier("Research portable OR-tail hook (M-B1 Stage 1)", "Experimental")
    )
    research.description = (
        "Default-off frozen portable OR-tail policy injection for online e2e A/B. "
        "See docs/modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md "
        "and m_b1_to_m_b1_5_two_stage_plan_20260710.md. Never enable from production presets."
    )
    research.add_argument(
        "--research-portable-or-tail-policy",
        default=None,
        help=(
            "Path to freeze portable_policy.json. Default: off (also env "
            "SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY). When set, load thr and reject "
            "bridge candidates if ANY frozen tail atom fires. Fail-closed on schema mismatch."
        ),
    )
    research.add_argument(
        "--research-portable-or-tail-audit",
        action="store_true",
        default=False,
        help=(
            "When hook is on, enable full candidate-event audit export "
            "(separate from policy runtime overhead)."
        ),
    )
    research.add_argument(
        "--research-portable-or-tail-audit-dir",
        default=None,
        help="Directory for hook audit artifacts (events/json). Required when audit is on.",
    )
