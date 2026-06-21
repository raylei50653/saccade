"""Single source of truth for the *basis* parameters shared across the
association / relink / auction subsystems.

Two problems this module solves (see the 2026-06-20 config-unification work):

1. **消重 — physical constants.** The same real-world scene constants
   (person height, scene fps, the height-normalisation basis) used to be
   re-declared as separate knobs per subsystem (``relink_bridge_person_height``
   vs ``semantic_kalman_person_height_m``; ``relink_bridge_fps`` vs
   ``semantic_kalman_fps``) with literal defaults duplicated across *both*
   config layers (``scripts/eval/config/*`` and
   ``saccade.perception.eval.config``). Changing one and forgetting the other
   silently diverged. The constants below are now the single place those
   defaults live; both config layers import them.

2. **分群 — association-primitive namespace.** During experiments it was hard
   to discuss a config's *combination* of basic primitives, because the
   height / IoU / velocity / cost-weight knobs are scattered across 4-5
   dataclasses owned by different subsystems. :data:`ASSOC_BASIS` is a logical
   grouping that maps each primitive *family* to its underlying flat config
   keys, so a whole bundle can be resolved and printed at once
   (``scripts/eval/print_assoc_basis.py``).

This module is intentionally dependency-free (pure literals + stdlib) so the
CLI config layer can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Physical / geometric constants — single source of truth.
# A subsystem that wants the *off* sentinel keeps its own 0.0 default; these
# are only the genuine scene constants, not enable flags.
# --------------------------------------------------------------------------
PERSON_HEIGHT_M: float = 1.65
"""Assumed standing person height in metres (speed-gate metric scale)."""

SCENE_FPS: float = 30.0
"""Default capture frame-rate (frames -> seconds for metric speed gates)."""

REF_HEIGHT_RATIO: float = 0.12
"""Reference person-height fraction of frame height — the dimensionless basis
that geometry / occlusion / relink use to normalise box height and foot gaps."""


@dataclass(frozen=True)
class PrimitiveFamily:
    """A logical group of association primitives that share a meaning."""

    name: str
    description: str
    keys: tuple[str, ...]


# --------------------------------------------------------------------------
# Association-primitive namespace.
#
# Each family lists the *flat config keys* (as seen in EvalConfig / the flat
# dict produced by PipelineConfig.to_flat_dict) that belong to it. Keys may
# appear in more than one family on purpose (e.g. an IoU-weighted cost term is
# both an `iou_gate` consumer and a `cost_weight`).
# --------------------------------------------------------------------------
ASSOC_BASIS: tuple[PrimitiveFamily, ...] = (
    PrimitiveFamily(
        name="physical_const",
        description="True scene constants (single source: assoc_basis.py).",
        keys=(
            "relink_bridge_person_height",
            "semantic_kalman_person_height_m",
            "relink_bridge_fps",
            "semantic_kalman_fps",
            "geometry_ref_height_ratio",
        ),
    ),
    PrimitiveFamily(
        name="height_norm",
        description="Height / foot-depth normalisation and height-ratio gates.",
        keys=(
            "geometry_ref_height_ratio",
            "person_min_height_ratio",
            "narrow_person_min_height_ratio",
            "occ_foot_gap",
            "relink_bridge_h_lo",
            "relink_bridge_h_hi",
        ),
    ),
    PrimitiveFamily(
        name="iou_gate",
        description="IoU thresholds gating association / merge / birth.",
        keys=(
            "match_thresh",
            "id_stability_min_iou",
            "motion_only_iou_threshold",
            "semantic_min_iou",
            "occ_iou_thresh",
            "multi_birth_iou_match",
            "lifecycle_min_iou",
            "duplicate_suppression_iou_threshold",
            "appearance_bank_min_iou",
        ),
    ),
    PrimitiveFamily(
        name="velocity_gate",
        description="Velocity / acceleration / speed gates and EMA factors.",
        keys=(
            "relink_bridge_max_speed",
            "semantic_kalman_max_speed_mps",
            "semantic_kalman_accel_long",
            "semantic_kalman_accel_lat",
            "semantic_kalman_dir_min_speed",
            "consistency_tol",
            "vel_alpha",
            "acc_alpha",
            "motion_only_lost_frames",
        ),
    ),
    PrimitiveFamily(
        name="cost_weight",
        description="Weights on association / merge cost terms.",
        keys=(
            "occ_cost_weight",
            "fuse_score_weight",
            "vel_dir_weight",
            "relink_bridge_dir_bonus",
            "w_motion_iou",
            "semantic_w_sim_base",
            "semantic_w_iou_base",
            "semantic_w_maha_base",
            "post_lifecycle_spatial_weight",
            "post_lifecycle_motion_weight",
            "post_lifecycle_time_weight",
            "post_lifecycle_direction_weight",
        ),
    ),
)

_MISSING = object()


def resolve(flat: dict[str, object]) -> dict[str, dict[str, object]]:
    """Group a flat config dict by association-primitive family.

    Returns ``{family_name: {key: value}}``. Keys absent from *flat* are
    reported with the string ``"<unset>"`` so a config that predates a knob is
    still legible rather than silently dropped.
    """
    out: dict[str, dict[str, object]] = {}
    for fam in ASSOC_BASIS:
        group: dict[str, object] = {}
        for key in fam.keys:
            val = flat.get(key, _MISSING)
            group[key] = "<unset>" if val is _MISSING else val
        out[fam.name] = group
    return out


def format_basis(flat: dict[str, object], *, title: str | None = None) -> str:
    """Render the grouped association basis of *flat* as an aligned table."""
    grouped = resolve(flat)
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    fam_by_name = {f.name: f for f in ASSOC_BASIS}
    for fam_name, group in grouped.items():
        fam = fam_by_name[fam_name]
        lines.append("")
        lines.append(f"[{fam_name}] {fam.description}")
        width = max((len(k) for k in group), default=0)
        for key, val in group.items():
            lines.append(f"  {key:<{width}}  {val}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Signal x consumer matrix.
#
# The *other* axis of the same problem: which raw signal each association
# consumer (the auction cost, the bridge relink, the python semantic relink)
# actually uses, and how it stacks. Verified against code 2026-06-20:
#   auction  -> src/tracking/tracker_gpu.cu, include/tracking/kalman_gpu.cuh
#   bridge   -> src/tracking/relink_gate.cu, include/tracking/relink_gate.hpp
#   semantic -> src/saccade/perception/eval/relink.py
# A ``None`` cell means the consumer does not use that signal.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalRow:
    signal: str
    auction: str | None
    bridge: str | None
    semantic: str | None


SIGNAL_CONSUMERS: tuple[SignalRow, ...] = (
    SignalRow(
        "IoU",
        "primary cost term (iou_cost_val, iou_gate)",
        None,
        "joint score w_iou term",
    ),
    SignalRow(
        "box height h",
        "Mahalanobis pos sigma=h/20; stability reward 1/(1+|dh|/h); foot calc",
        None,
        None,
    ),
    SignalRow("box width w", "indirect via IoU", None, None),
    SignalRow(
        "aspect",
        "aspect_q gate (peak 2.5); Sinkhorn G_aspect penalty",
        None,
        "quality filter",
    ),
    SignalRow(
        "center (cx,cy)",
        "Mahalanobis innovation innov[:2]",
        "foot-history velocity regression",
        "center_norm spatial gate",
    ),
    SignalRow(
        "score",
        "S0-S2 staged score gates; fuse with IoU",
        None,
        "quality filter",
    ),
    SignalRow(
        "velocity (vx,vy)",
        "cos_dir direction penalty (vel_dir_weight, default off)",
        "4/8-pt OLS regress (vxl/vyl,vxc/vyc); s_lost; bdist_dir",
        "Kalman extrapolation; dir_behind gate; motion consistency",
    ),
    SignalRow(
        "appearance cos_sim",
        "ReID mode: w_cos*sim + w_iou*iou",
        None,
        "primary ranking signal w_sim*sim",
    ),
    SignalRow(
        "foot history (cx,cy,h)",
        None,
        "4-pt velocity regress; bridge-distance extrapolation",
        "midpoint bridge distance",
    ),
    SignalRow(
        "lost age",
        None,
        "TTL / min_lost gate",
        "dynamic weight switch (lost_factor)",
    ),
    SignalRow(
        "hit_streak",
        None,
        "trigger timing (hit_streak == bridge_at)",
        None,
    ),
    SignalRow("track_avg", "score penalty drop/avg", None, None),
    SignalRow("OAO occ_coeff", "occ_coeff penalty", None, None),
    SignalRow(
        "Mahalanobis d^2",
        "gate + cost term",
        None,
        "Kalman joint-score penalty",
    ),
)


# --------------------------------------------------------------------------
# Unmanaged constants — magic numbers that set the *scale* of the signals
# above but live hardcoded in C++/Python, outside any config. These cannot be
# swept or re-probed per dataset; they are the next gap in unified management.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UnmanagedConstant:
    name: str
    value: str
    location: str
    affects: str


UNMANAGED_CONSTANTS: tuple[UnmanagedConstant, ...] = (
    UnmanagedConstant(
        "aspect_q peak / width",
        "2.5 / 1.2",
        "src/tracking/tracker_gpu.cu:186",
        "aspect signal (auction quality + Sinkhorn)",
    ),
    UnmanagedConstant(
        "std_weight_position",
        "1/20",
        "include/tracking/kalman_gpu.cuh:155",
        "h -> center Mahalanobis sigma (pos_std = h/20)",
    ),
    UnmanagedConstant(
        "std_weight_velocity",
        "1/160",
        "include/tracking/kalman_gpu.cuh:156",
        "h -> velocity Kalman sigma (vel_std = h/160)",
    ),
    UnmanagedConstant(
        "occ crowd saturation",
        "/0.25",
        "src/tracking/tracker_gpu.cu:500",
        "occ_coeff -> crowd penalty scale",
    ),
    UnmanagedConstant(
        "ema_h alpha",
        "0.05",
        "src/tracking/tracker_gpu.cu:1706",
        "box-height EMA smoothing",
    ),
)


def format_signals() -> str:
    """Render the signal x consumer matrix and the unmanaged-constant list."""
    cols = ("auction", "bridge", "semantic")
    sig_w = max(len(r.signal) for r in SIGNAL_CONSUMERS)
    lines = ["signal x consumer matrix", "=" * 24, ""]
    header = f"  {'signal':<{sig_w}}  " + "  ".join(f"{c:<48}" for c in cols)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in SIGNAL_CONSUMERS:
        cells = [
            (r.auction or "—"),
            (r.bridge or "—"),
            (r.semantic or "—"),
        ]
        lines.append(f"  {r.signal:<{sig_w}}  " + "  ".join(f"{c:<48}" for c in cells))

    lines += ["", "", "unmanaged constants (hardcoded, not in any config)", "=" * 48]
    nm_w = max(len(c.name) for c in UNMANAGED_CONSTANTS)
    val_w = max(len(c.value) for c in UNMANAGED_CONSTANTS)
    for c in UNMANAGED_CONSTANTS:
        lines.append(
            f"  {c.name:<{nm_w}}  {c.value:<{val_w}}  {c.location}\n"
            f"  {'':<{nm_w}}  {'':<{val_w}}  -> {c.affects}"
        )

    lines += ["", "", format_env_overrides()]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Environment-variable escape hatches.
#
# These read straight from os.environ inside the C++/Python hot path, bypassing
# the config system entirely — so they are invisible to MLflow logging and to
# print_assoc_basis's config view. Several default to *on* (stability_w=0.1,
# enable_dda=true), so two runs with the same config can still differ. Listed
# here so a run's true tuning state can be inspected.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvOverride:
    env: str
    default: str
    location: str
    affects: str


ENV_OVERRIDES: tuple[EnvOverride, ...] = (
    EnvOverride(
        "SACCADE_FRESHNESS_W",
        "0.0 (off)",
        "src/tracking/tracker_gpu.cu:2650",
        "auction bid += w/(1+age) — recency boost",
    ),
    EnvOverride(
        "SACCADE_STABILITY_W",
        "0.1 (ON)",
        "src/tracking/tracker_gpu.cu:2666",
        "auction bid += w/(1+dh_rel) — height-stability boost",
    ),
    EnvOverride(
        "SACCADE_ENABLE_DDA",
        "true (ON)",
        "src/tracking/tracker_gpu.cu:2404",
        "enable DDA association mode",
    ),
    EnvOverride(
        "SACCADE_DDA_MAX_COST",
        "0.12",
        "src/tracking/tracker_gpu.cu:2405",
        "DDA association max-cost gate",
    ),
    EnvOverride(
        "SACCADE_GMC_PCR_THRESH",
        "(see gmc_kernel.cu)",
        "src/tracking/gmc_kernel.cu:169",
        "GMC phase-correlation uncertainty accept threshold",
    ),
    EnvOverride(
        "SACCADE_GPU_RELINK_GATE",
        "build=1 (C++) / use=0 (py)",
        "src/saccade/perception/eval/evaluator.py:2093",
        "GPU relink-gate: evaluator builds the table (default 1, C++ relinker); "
        "PythonSemanticRelinker's use-flag defaults 0 as the bit-exact reference",
    ),
)


def resolved_env_overrides() -> dict[str, str]:
    """Map each escape-hatch env var to its effective value for this process.

    Returns the literal ``os.environ`` value if set, else ``"<default>"`` so a
    run's MLflow record captures whether a hatch was flipped (the hatches
    otherwise bypass config logging entirely).
    """
    import os

    return {e.env: os.environ.get(e.env, "<default>") for e in ENV_OVERRIDES}


def format_env_overrides() -> str:
    """List the env-var escape hatches with their default and *current* value."""
    import os

    lines = ["env-var escape hatches (bypass config; not in MLflow)", "=" * 52]
    env_w = max(len(e.env) for e in ENV_OVERRIDES)
    def_w = max(len(e.default) for e in ENV_OVERRIDES)
    for e in ENV_OVERRIDES:
        cur = os.environ.get(e.env)
        cur_s = f"current={cur}" if cur is not None else "current=<unset>"
        lines.append(
            f"  {e.env:<{env_w}}  default={e.default:<{def_w}}  {cur_s}\n"
            f"  {'':<{env_w}}  {'':<{def_w + 8}}  -> {e.affects}  [{e.location}]"
        )
    return "\n".join(lines)
