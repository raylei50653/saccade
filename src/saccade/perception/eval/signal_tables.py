"""Signal-table schema for transparent multi-method / multi-dimension analysis.

Normative doc: ``docs/research/eval/signal_table_schema.md``.

Design:
  * One universe per table (U_gt / U_det / U_cand / U_relink_pair / U_err).
  * ``U_cand`` = frame association only; ``U_relink_pair`` = offline lost→cand.
  * Labels ``y`` are frozen by ``StudyMeta``; do not re-greedy with a new IoU
    inside ad-hoc scripts without a new study_id.
  * Methods are gates or pipeline cuts; dimensions are covariates for strata.
  * Pipeline nodes fix **layer + order + parents** so studies can be stacked.
  * Relink B1 studies must set ``hard_pool_rule`` and report full+hard AUC.
  * Prefer parquet for frame-level tables; small e2e counts may stay CSV.

This module is the typed contract + light IO helpers. It does not run MOT eval.

Pipeline path reference (headline m):
  ``docs/research/pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md``
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UniverseId(str, Enum):
    """Sample universe — one parquet table per value.

    Do **not** mix frame-level association candidates (``U_cand``) with offline
    relink (lost→cand) pairs (``U_relink_pair``). Their labels and AUC pools differ.
    """

    U_GT = "U_gt"
    U_DET = "U_det"
    # Frame-level (track, det) association candidates only — not relink pairs.
    U_CAND = "U_cand"
    # Offline (lost track → birth candidate) pairs; builder: build_relink_candidates.py
    U_RELINK_PAIR = "U_relink_pair"
    U_ERR = "U_err"
    METHOD_METRICS = "method_metrics"  # small summary, not frame-level


# Alias for docs / call sites that prefer an explicit assoc name.
U_ASSOC_CAND = UniverseId.U_CAND


class MethodId(str, Enum):
    """Processing cut / gate identity (not a covariate)."""

    RAW = "raw"
    POST_FILTER = "post_filter"
    POST_NMS = "post_nms"
    POST_MERGE = "post_merge"
    BARE_TRACK = "bare_track"
    BARE_GMC = "bare_gmc"
    BARE_BRIDGE = "bare_bridge"
    FULL_PRESET = "full_preset"
    FULL_NO_INTERP = "full_no_interp"
    CUSTOM = "custom"


class DataSourceId(str, Enum):
    """Where a column or table originated."""

    MOT17_GT = "mot17_gt"
    STAGE_DUMP_CSV = "stage_dump_csv"
    SCORE_DIST_TOOL = "score_dist_tool"
    FP_BY_HEIGHT_TOOL = "fp_by_height_tool"
    MOT_RESULT = "mot_result"
    MOTMETRICS_EVAL = "motmetrics_eval"
    TRACKEVAL_HOTA = "trackeval_hota"
    NEUTRAL_NOGO_TOOL = "neutral_nogo_tool"
    NEAR_MISS_STAGE = "near_miss_stage"
    PIPELINE_CONTRIBUTION = "pipeline_contribution"
    MANUAL_LABEL = "manual_label"
    DERIVED = "derived"


class ErrEventType(str, Enum):
    FN_MISS = "fn_miss"
    ID_SWITCH = "id_switch"
    FP_BIRTH = "fp_birth"
    FP_INTERP = "fp_interp"
    FRAGMENT = "fragment"
    OTHER = "other"


class PipelineLayer(str, Enum):
    """Coarse stack for analysis bookkeeping (not CUDA graph L1/L2/L3).

    Order is causal for the *logical* MOT path. Double-buffer may overlap
    detect(N+1) with track(N) in wall time without changing this order for
    frame N's data dependencies.
    """

    L0_INGEST = "L0_ingest"  # fetch / decode / resize
    L1_DETECT = "L1_detect"  # backbone+head raw boxes
    L2_POST = "L2_post"  # filter / NMS / private cont. / merge
    L3_MOTION = "L3_motion"  # GMC (parallel branch into track)
    L4_ASSOC = "L4_assoc"  # Kalman + auction + birth/confirm
    L5_IDENTITY = "L5_identity"  # bridge relink / (ReID off on headline)
    L6_EMIT = "L6_emit"  # materialize / write frame rows
    L7_POSTSEQ = "L7_postseq"  # interpolate_tracklets
    L8_METRICS = "L8_metrics"  # motmetrics + TrackEval


class CutDesign(str, Enum):
    """How multiple MethodId runs relate in one study."""

    SINGLE = "single"  # one method only
    CUMULATIVE = "cumulative"  # each row enables one more node along a path
    SINGLE_ON_BASE = "single_on_base"  # each method = base + one module
    ORTHOGONAL = "orthogonal"  # independent ablations (not causal Δ)
    SWEEP = "sweep"  # fixed method + 1..N continuous/grid axes
    CUSTOM = "custom"


class SweepMode(str, Enum):
    """How a sweep is realized physically."""

    # Re-threshold / re-gate on a frozen dump (no tracker re-run).
    # Typical: score floor on U_det, cost gate on U_cand.
    OFFLINE = "offline"
    # Each grid point is a full (or partial) pipeline re-run.
    ONLINE = "online"
    # Mix: some axes offline, some online (document in notes).
    MIXED = "mixed"


class SweepGridKind(str, Enum):
    """How axis values are generated (stored expanded in ``values``)."""

    MANUAL = "manual"  # only ``values``
    LINSPACE = "linspace"  # lo, hi, num
    ARANGE = "arange"  # lo, hi, step (hi exclusive or inclusive via include_hi)
    LOGSPACE = "logspace"  # lo, hi, num (log10)


# Stage names emitted by append_stage_dump_rows / near-miss tools.
STAGE_DUMP_METHODS: tuple[str, ...] = (
    MethodId.RAW.value,
    MethodId.POST_FILTER.value,
    MethodId.POST_NMS.value,
    MethodId.POST_MERGE.value,
)

# evaluator.py top-level profile stage names (mot17_default_config §4).
PROFILE_STAGE_ORDER: tuple[str, ...] = (
    "fetch",
    "ingest_preprocess",
    "detect",
    "postprocess",
    "reid_bank_sync",
    "reid_budget",
    "reid_crop",
    "reid_extract",
    "lazy_reid",
    "gmc",
    "track",
    "materialize",
    "bg_relink_wait",
    "relink_write",
    "frame_total",
)

# ---------------------------------------------------------------------------
# Pipeline graph (nodes + edges)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineNode:
    """One causal unit in the eval pipeline for signal studies.

    ``parents`` are prerequisite node_ids (data must exist before this node).
    ``profile_stages`` map to evaluator timing names when applicable.
    ``method_ids`` are study cuts that *end* at this node (inclusive).
    """

    node_id: str
    layer: PipelineLayer
    order: int  # global topological-ish order for sorting reports
    parents: tuple[str, ...]
    label: str
    profile_stages: tuple[str, ...] = ()
    method_ids: tuple[str, ...] = ()
    primary_universe: str = ""  # UniverseId value or ""
    claimed_signal: str = ""
    headline_m_on: bool = True  # mamba_whole_graph_m default
    optional: bool = False
    notes: str = ""


def _n(
    node_id: str,
    layer: PipelineLayer,
    order: int,
    parents: tuple[str, ...],
    label: str,
    **kwargs: Any,
) -> PipelineNode:
    return PipelineNode(
        node_id=node_id,
        layer=layer,
        order=order,
        parents=parents,
        label=label,
        **kwargs,
    )


# Canonical logical graph for MOT17 headline whole-graph path (s/m).
# GMC is a sibling of detect/post that joins at track (not a child of NMS).
PIPELINE_NODES: tuple[PipelineNode, ...] = (
    _n(
        "fetch",
        PipelineLayer.L0_INGEST,
        10,
        (),
        "Fetch / decode frame",
        profile_stages=("fetch",),
        claimed_signal="image available",
    ),
    _n(
        "ingest_preprocess",
        PipelineLayer.L0_INGEST,
        20,
        ("fetch",),
        "Ingest / resize to model input",
        profile_stages=("ingest_preprocess",),
        claimed_signal="letterbox/stretch domain",
    ),
    _n(
        "detect",
        PipelineLayer.L1_DETECT,
        30,
        ("ingest_preprocess",),
        "Detector forward (backbone + head)",
        profile_stages=("detect",),
        method_ids=(MethodId.RAW.value,),
        primary_universe=UniverseId.U_DET.value,
        claimed_signal="box+score before filter/NMS",
        notes="stage_dump stage=raw; conf floor may already apply in export",
    ),
    _n(
        "post_filter",
        PipelineLayer.L2_POST,
        40,
        ("detect",),
        "Score / class filter",
        profile_stages=("postprocess",),
        method_ids=(MethodId.POST_FILTER.value,),
        primary_universe=UniverseId.U_DET.value,
        claimed_signal="low-score rejection",
    ),
    _n(
        "post_nms",
        PipelineLayer.L2_POST,
        50,
        ("post_filter",),
        "NMS (+ optional private continuation expand)",
        profile_stages=("postprocess",),
        method_ids=(MethodId.POST_NMS.value,),
        primary_universe=UniverseId.U_DET.value,
        claimed_signal="duplicate suppression vs TP kill",
    ),
    _n(
        "post_merge",
        PipelineLayer.L2_POST,
        55,
        ("post_nms",),
        "Optional post-merge on dets",
        profile_stages=("postprocess",),
        method_ids=(MethodId.POST_MERGE.value,),
        primary_universe=UniverseId.U_DET.value,
        claimed_signal="merge overlapping dets",
        headline_m_on=False,
        optional=True,
        notes="Often inactive on headline; dump stage may still appear empty",
    ),
    _n(
        "reid",
        PipelineLayer.L5_IDENTITY,
        60,
        ("post_nms",),
        "ReID bank/crop/extract (headline OFF)",
        profile_stages=(
            "reid_bank_sync",
            "reid_budget",
            "reid_crop",
            "reid_extract",
            "lazy_reid",
        ),
        claimed_signal="appearance identity",
        headline_m_on=False,
        optional=True,
        notes="reid_mode=off on mamba_whole_graph*; skip in default study",
    ),
    _n(
        "gmc",
        PipelineLayer.L3_MOTION,
        70,
        ("fetch",),  # gray path from frame; not child of NMS
        "Global motion compensation warp",
        profile_stages=("gmc",),
        claimed_signal="camera-motion residual for predict",
        notes="Joins at track; measure Δ vs bare_track without GMC",
    ),
    _n(
        "track",
        PipelineLayer.L4_ASSOC,
        80,
        ("post_nms", "gmc"),
        "GPUByteTracker: predict + associate + birth/confirm",
        profile_stages=("track",),
        method_ids=(MethodId.BARE_TRACK.value, MethodId.BARE_GMC.value),
        primary_universe=UniverseId.U_CAND.value,
        claimed_signal="frame association / ID continuity",
        notes=(
            "bare_track = track without GMC warp; bare_gmc = track+gmc. "
            "Both emit before bridge if bridge disabled."
        ),
    ),
    _n(
        "bridge_relink",
        PipelineLayer.L5_IDENTITY,
        90,
        ("track",),
        "Geometry bridge relink",
        profile_stages=("bg_relink_wait", "relink_write"),
        method_ids=(MethodId.BARE_BRIDGE.value,),
        primary_universe=UniverseId.U_ERR.value,
        claimed_signal="long-gap reconnect without appearance",
        notes="m uses looser bridge gates than s",
    ),
    _n(
        "materialize",
        PipelineLayer.L6_EMIT,
        100,
        ("track", "bridge_relink"),
        "Materialize host track rows / emit",
        profile_stages=("materialize",),
        claimed_signal="output boxes this frame",
        notes="Parents: track always; bridge when enabled",
    ),
    _n(
        "interpolate",
        PipelineLayer.L7_POSTSEQ,
        110,
        ("materialize",),
        "Post-sequence tracklet interpolation",
        method_ids=(
            MethodId.FULL_PRESET.value,
            MethodId.FULL_NO_INTERP.value,
        ),
        primary_universe=UniverseId.U_ERR.value,
        claimed_signal="gap fill continuity (eval cosmetic + FN/IDs trade)",
        notes="full_no_interp ends before this node; full_preset includes it",
    ),
    _n(
        "metrics",
        PipelineLayer.L8_METRICS,
        120,
        ("interpolate",),
        "motmetrics + TrackEval HOTA",
        method_ids=(),
        primary_universe=UniverseId.METHOD_METRICS.value,
        claimed_signal="aggregate counts → IDF1/MOTA/HOTA",
        notes="Not a tracker module; always last for a given emit set",
    ),
)

PIPELINE_BY_ID: dict[str, PipelineNode] = {n.node_id: n for n in PIPELINE_NODES}

# Default cumulative analysis spine (headline m, ReID off, post_merge off).
DEFAULT_CUMULATIVE_SPINE: tuple[str, ...] = (
    "detect",
    "post_filter",
    "post_nms",
    "track",  # with gmc parent required for fair production-like bare_gmc
    "bridge_relink",
    "interpolate",
    "metrics",
)

# MethodId → terminal pipeline node (cut ends here, inclusive).
METHOD_TERMINAL_NODE: dict[str, str] = {
    MethodId.RAW.value: "detect",
    MethodId.POST_FILTER.value: "post_filter",
    MethodId.POST_NMS.value: "post_nms",
    MethodId.POST_MERGE.value: "post_merge",
    MethodId.BARE_TRACK.value: "track",  # GMC parent off by study config
    MethodId.BARE_GMC.value: "track",  # GMC on
    MethodId.BARE_BRIDGE.value: "bridge_relink",
    MethodId.FULL_NO_INTERP.value: "materialize",
    MethodId.FULL_PRESET.value: "interpolate",
}


def get_pipeline_node(node_id: str) -> PipelineNode:
    try:
        return PIPELINE_BY_ID[node_id]
    except KeyError as exc:
        raise KeyError(f"unknown pipeline node_id={node_id!r}") from exc


def pipeline_nodes_sorted() -> list[PipelineNode]:
    return sorted(PIPELINE_NODES, key=lambda n: n.order)


def pipeline_parents(node_id: str) -> tuple[str, ...]:
    return get_pipeline_node(node_id).parents


def pipeline_ancestors(node_id: str) -> list[str]:
    """All ancestors of node_id (parents first, DFS), excluding self."""
    seen: list[str] = []
    stack = list(pipeline_parents(node_id))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.append(cur)
        stack.extend(pipeline_parents(cur))
    return seen


def pipeline_layer_of_method(method_id: str | MethodId) -> PipelineLayer | None:
    mid = method_id.value if isinstance(method_id, MethodId) else method_id
    node_id = METHOD_TERMINAL_NODE.get(mid)
    if node_id is None:
        return None
    return get_pipeline_node(node_id).layer


def method_terminal_node(method_id: str | MethodId) -> str | None:
    mid = method_id.value if isinstance(method_id, MethodId) else method_id
    return METHOD_TERMINAL_NODE.get(mid)


def methods_for_node(node_id: str) -> tuple[str, ...]:
    return get_pipeline_node(node_id).method_ids


def validate_method_order(method_ids: Sequence[str]) -> None:
    """Ensure methods appear in non-decreasing pipeline order (for cumulative).

    Uses terminal node order. ``custom`` is skipped. Raises SchemaError-like
    ValueError on regression.
    """
    orders: list[tuple[str, int]] = []
    for mid in method_ids:
        if mid == MethodId.CUSTOM.value:
            continue
        node_id = METHOD_TERMINAL_NODE.get(mid)
        if node_id is None:
            raise ValueError(f"method_id {mid!r} has no terminal pipeline node")
        orders.append((mid, get_pipeline_node(node_id).order))
    for i in range(1, len(orders)):
        if orders[i][1] < orders[i - 1][1]:
            raise ValueError(
                "method_ids not in pipeline order: "
                f"{orders[i - 1][0]} (order {orders[i - 1][1]}) then "
                f"{orders[i][0]} (order {orders[i][1]})"
            )


def cumulative_enabled_nodes(terminal_node_id: str) -> list[str]:
    """Nodes enabled when cutting the pipeline at terminal (ancestors + self).

    Note: sibling branches (e.g. gmc vs post_nms) are both included only if
    they are ancestors of the terminal or the terminal itself. For bare_track
    without GMC, callers should remove ``gmc`` explicitly via study notes /
    method semantics (BARE_TRACK vs BARE_GMC).
    """
    nodes = set(pipeline_ancestors(terminal_node_id))
    nodes.add(terminal_node_id)
    return [n.node_id for n in pipeline_nodes_sorted() if n.node_id in nodes]


def pipeline_ascii_art() -> str:
    """Compact text graph for docs / CLI help."""
    lines = [
        "Pipeline (logical order; GMC joins at track):",
        "",
        "  fetch → ingest_preprocess → detect → post_filter → post_nms → [post_merge?]",
        "    │                                              ↓",
        "    └────────────────────→ gmc ────────→ track → bridge_relink → materialize",
        "                                                      ↓",
        "                                              interpolate → metrics",
        "",
        "  Layers: L0 ingest · L1 detect · L2 post · L3 motion · L4 assoc",
        "          L5 identity · L6 emit · L7 postseq · L8 metrics",
        "  Headline m: ReID off, post_merge usually off, bridge+interp on.",
    ]
    return "\n".join(lines)


def pipeline_summary_rows() -> list[dict[str, Any]]:
    """Rows suitable for markdown / DataFrame export."""
    rows: list[dict[str, Any]] = []
    for n in pipeline_nodes_sorted():
        rows.append(
            {
                "order": n.order,
                "node_id": n.node_id,
                "layer": n.layer.value,
                "parents": ",".join(n.parents),
                "method_ids": ",".join(n.method_ids),
                "universe": n.primary_universe,
                "headline_m_on": n.headline_m_on,
                "claimed_signal": n.claimed_signal,
                "profile_stages": ",".join(n.profile_stages),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Column contracts (parquet / CSV)
# ---------------------------------------------------------------------------

# Required columns for validation (optional research covariates omitted).
GT_REQUIRED: tuple[str, ...] = (
    "seq",
    "frame",
    "gt_id",
    "x",
    "y",
    "w",
    "h",
    "matched",
)
DET_REQUIRED: tuple[str, ...] = (
    "seq",
    "frame",
    "stage",
    "det_idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "is_tp",
)
# Raw stage dump before labels (maps from append_stage_dump_rows).
DET_STAGE_DUMP_COLUMNS: tuple[str, ...] = (
    "seq",
    "frame",
    "stage",
    "det_idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "w",
    "h",
    "score",
    "cls",
)
CAND_REQUIRED: tuple[str, ...] = (
    "seq",
    "frame",
    "track_id",
    "det_idx",
    "is_correct",
)
# Minimal columns aligned with scripts/tools/build_relink_candidates.py COLS.
# Full builder CSV has more motion/height variants; those are optional extras.
RELINK_PAIR_REQUIRED: tuple[str, ...] = (
    "seq",
    "lost_id",
    "cand_id",
    "gt_match",
    "gt_valid",
    "bridge_dist",
    "gap",
    "lost_last_frame",
    "cand_first_frame",
)
# Optional but expected when importing builder CSV wholesale.
RELINK_PAIR_BUILDER_COLS: tuple[str, ...] = (
    "seq",
    "lost_id",
    "cand_id",
    "gt_lost",
    "gt_cand",
    "gt_match",
    "gt_valid",
    "accepted",
    "already_linked",
    "gap",
    "dist_h",
    "fwd_resid",
    "bwd_resid",
    "bridge_dist",
    "dir_cos",
    "speed_h",
    "lost_exit_speed",
    "cand_entry_speed",
    "lost_last_frame",
    "cand_first_frame",
    "lost_lifespan",
    "cand_lifespan",
    "lost_len",
    "cand_len",
    "lost_foot_x",
    "lost_foot_y",
    "cand_foot_x",
    "cand_foot_y",
    "h_ref",
)
ERR_REQUIRED: tuple[str, ...] = (
    "seq",
    "frame",
    "event_id",
    "event_type",
    "method",
)
METHOD_METRICS_REQUIRED: tuple[str, ...] = ("method",)
# One row per (method × grid point). params_* columns are free-form floats.
RUN_METRICS_REQUIRED: tuple[str, ...] = ("run_id", "method")

# Default hard-pool for offline relink AUC (see offline_relink_candidate_analysis.md).
DEFAULT_RELINK_HARD_POOL_RULE = "bridge_dist<=1.0"
# Canonical tools for dual-line studies (paths relative to repo root).
STUDY_SCRIPT_MAP: dict[str, str] = {
    "A_recall": "scripts/eval/analyze_score_distribution.py",
    "B1_ids_signal": "scripts/tools/build_relink_candidates.py",
    "B1_summarize": "scripts/tools/summarize_relink_pairs.py",
    "B2_ids_state": "scripts/eval/diagnostics/reconnect_rate.py",
}

UNIVERSE_REQUIRED: dict[UniverseId, tuple[str, ...]] = {
    UniverseId.U_GT: GT_REQUIRED,
    UniverseId.U_DET: DET_REQUIRED,
    UniverseId.U_CAND: CAND_REQUIRED,
    UniverseId.U_RELINK_PAIR: RELINK_PAIR_REQUIRED,
    UniverseId.U_ERR: ERR_REQUIRED,
    UniverseId.METHOD_METRICS: METHOD_METRICS_REQUIRED,
}

DEFAULT_Y_DEFINITIONS: dict[str, str] = {
    UniverseId.U_GT.value: (
        "matched if some det IoU>=iou_match under score-greedy claim"
    ),
    UniverseId.U_DET.value: ("is_tp if claimed a GT with IoU>=iou_match; else FP"),
    UniverseId.U_CAND.value: (
        "is_correct if det is the GT-matched target of this track at frame "
        "(frame-level association only; not relink pairs)"
    ),
    UniverseId.U_RELINK_PAIR.value: (
        "gt_match==1 if lost and cand map to the same GT id under builder rules "
        "(offline relink pair; substrate: relink-off + interp-off)"
    ),
    UniverseId.U_ERR.value: (
        "event_type in {fn_miss, id_switch, fp_birth, fp_interp, fragment}"
    ),
}

DEFAULT_SCORE_FIELDS: dict[str, str] = {
    UniverseId.U_GT.value: "match_score",
    UniverseId.U_DET.value: "score",
    UniverseId.U_CAND.value: "iou",  # or -cost; override in StudyMeta
    UniverseId.U_RELINK_PAIR.value: "bridge_dist",  # lower is better ranker
}

TABLE_FILENAMES: dict[UniverseId, str] = {
    UniverseId.U_GT: "u_gt.parquet",
    UniverseId.U_DET: "u_det.parquet",
    UniverseId.U_CAND: "u_cand.parquet",
    UniverseId.U_RELINK_PAIR: "u_relink_pair.parquet",
    UniverseId.U_ERR: "u_err.parquet",
    UniverseId.METHOD_METRICS: "metrics_by_method.csv",
}

# Sweep / multi-run aggregate table (not a UniverseId; lives next to meta).
RUN_METRICS_FILENAME = "metrics_by_run.csv"
SWEEP_CURVE_FILENAME = "sweep_curve_summary.csv"

META_FILENAME = "meta.json"

# B1 offline-relink study directory outputs (see signal_table_schema § B1 output).
# Narrative notes must link these files; do not embed master thr/AUC tables in markdown.
CONTEXT_FILENAME = "context.json"
METRICS_AUC_FILENAME = "metrics_auc.json"
METRICS_THR_FILENAME = "metrics_thr.csv"
PAIRS_CSV_FILENAME = "pairs.csv"
B1_OUTPUT_FILES: tuple[str, ...] = (
    CONTEXT_FILENAME,
    METRICS_AUC_FILENAME,
    METRICS_THR_FILENAME,
)
# Default bridge_dist thresholds for metrics_thr.csv (offline_relink note §3).
DEFAULT_RELINK_THR_GRID: tuple[float, ...] = (0.15, 0.30, 0.50, 1.00)
# Gap bins for context.json (aligned with offline_relink gap reporting).
DEFAULT_RELINK_GAP_BINS: tuple[tuple[str, int, int], ...] = (
    ("1-10", 1, 10),
    ("11-30", 11, 30),
    ("31-60", 31, 60),
    ("61-150", 61, 150),
    ("151-300", 151, 300),
)

# ---------------------------------------------------------------------------
# Sweep axis + run metrics
# ---------------------------------------------------------------------------


@dataclass
class SweepAxis:
    """One continuous or grid dimension to scan.

    Prefer storing the **expanded** ``values`` actually evaluated. ``lo/hi/num/step``
    document intent and can regenerate values via ``expanded_values()``.
    """

    name: str
    # Pipeline node the knob primarily acts on (for layer bookkeeping).
    node_id: str
    kind: str = SweepGridKind.MANUAL.value
    values: list[float] = field(default_factory=list)
    lo: float | None = None
    hi: float | None = None
    num: int | None = None
    step: float | None = None
    include_hi: bool = True
    unit: str = ""
    # Offline: apply gate on frozen table column without re-running MOT.
    offline: bool = False
    offline_score_col: str = ""  # e.g. "score" on U_det, "iou"/"cost" on U_cand
    offline_universe: str = ""  # UniverseId value when offline
    notes: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SweepAxis:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SweepAxis.name is required")
        if self.node_id and self.node_id not in PIPELINE_BY_ID:
            raise ValueError(
                f"SweepAxis {self.name!r}: unknown node_id={self.node_id!r}"
            )
        if self.kind not in {k.value for k in SweepGridKind}:
            raise ValueError(f"SweepAxis {self.name!r}: bad kind={self.kind!r}")
        if self.offline and not self.offline_universe:
            raise ValueError(
                f"SweepAxis {self.name!r}: offline=True requires offline_universe"
            )

    def expanded_values(self) -> list[float]:
        """Return concrete grid points (sorted unique)."""
        if self.values:
            pts = [float(v) for v in self.values]
        elif self.kind == SweepGridKind.LINSPACE.value:
            if self.lo is None or self.hi is None or not self.num:
                raise ValueError(f"{self.name}: linspace needs lo, hi, num")
            if self.num < 2:
                raise ValueError(f"{self.name}: linspace num must be >= 2")
            import numpy as np

            pts = [float(x) for x in np.linspace(self.lo, self.hi, int(self.num))]
        elif self.kind == SweepGridKind.LOGSPACE.value:
            if self.lo is None or self.hi is None or not self.num:
                raise ValueError(f"{self.name}: logspace needs lo, hi, num")
            import numpy as np

            pts = [
                float(x)
                for x in np.logspace(float(self.lo), float(self.hi), int(self.num))
            ]
        elif self.kind == SweepGridKind.ARANGE.value:
            if self.lo is None or self.hi is None or self.step is None:
                raise ValueError(f"{self.name}: arange needs lo, hi, step")
            import numpy as np

            stop = float(self.hi) + (float(self.step) * 0.5 if self.include_hi else 0.0)
            pts = [float(x) for x in np.arange(float(self.lo), stop, float(self.step))]
            if self.include_hi and pts and pts[-1] < float(self.hi) - 1e-12:
                pts.append(float(self.hi))
        else:
            pts = []
        # unique preserve order
        seen: set[float] = set()
        out: list[float] = []
        for p in pts:
            key = round(p, 10)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out


def expand_sweep_grid(axes: Sequence[SweepAxis]) -> list[dict[str, float]]:
    """Cartesian product of axis values → list of param dicts."""
    if not axes:
        return [{}]
    grids = [(a.name, a.expanded_values()) for a in axes]
    combos: list[dict[str, float]] = [{}]
    for name, vals in grids:
        nxt: list[dict[str, float]] = []
        for base in combos:
            for v in vals:
                row = dict(base)
                row[name] = v
                nxt.append(row)
        combos = nxt
    return combos


def make_run_id(method: str, params: Mapping[str, float]) -> str:
    """Stable run_id for metrics_by_run rows."""
    if not params:
        return method
    parts = [method] + [f"{k}={params[k]:.6g}" for k in sorted(params)]
    return "__".join(parts)


def offline_threshold_curve(
    scores: Sequence[float],
    is_positive: Sequence[bool],
    thresholds: Sequence[float],
    *,
    accept_if_ge: bool = True,
) -> list[dict[str, Any]]:
    """Scan a score/cost threshold offline; return P/R/TP/FP/FN per point.

    Does not re-run MOT — only set algebra on a frozen labeled universe.
    """
    if len(scores) != len(is_positive):
        raise ValueError("scores and is_positive length mismatch")
    n_pos = sum(1 for y in is_positive if y)
    n_neg = len(is_positive) - n_pos
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, is_positive, strict=True):
            accept = (s >= t) if accept_if_ge else (s <= t)
            if accept and y:
                tp += 1
            elif accept and not y:
                fp += 1
            elif (not accept) and y:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_pos if n_pos else 0.0
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "precision": prec,
                "recall": rec,
                "f1": ((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0),
            }
        )
    return rows


def summarize_sweep_metric(
    run_rows: Sequence[Mapping[str, Any]],
    *,
    axis_name: str,
    metric: str,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Aggregate a 1-D sweep curve: best point, endpoints, monotone-ish stats.

    ``run_rows`` each need ``params``-style key ``axis_name`` (or ``param_{axis}``)
    and ``metric`` column.
    """
    pts: list[tuple[float, float]] = []
    for r in run_rows:
        if axis_name in r:
            x = float(r[axis_name])  # type: ignore[arg-type]
        elif f"param_{axis_name}" in r:
            x = float(r[f"param_{axis_name}"])  # type: ignore[arg-type]
        else:
            continue
        if r.get(metric) is None:
            continue
        pts.append((x, float(r[metric])))  # type: ignore[arg-type]
    if not pts:
        raise ValueError(f"no points for axis={axis_name!r} metric={metric!r}")
    pts.sort(key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if higher_is_better:
        best_i = max(range(len(ys)), key=lambda i: ys[i])
    else:
        best_i = min(range(len(ys)), key=lambda i: ys[i])
    # simple consecutive Δ for noise / flatness
    deltas = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    return {
        "axis": axis_name,
        "metric": metric,
        "n_points": len(pts),
        "x_min": xs[0],
        "x_max": xs[-1],
        "y_at_x_min": ys[0],
        "y_at_x_max": ys[-1],
        "y_best": ys[best_i],
        "x_best": xs[best_i],
        "y_mean": sum(ys) / len(ys),
        "y_span": max(ys) - min(ys),
        "delta_mean": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "delta_max_abs": max((abs(d) for d in deltas), default=0.0),
        "higher_is_better": higher_is_better,
    }


def parse_simple_hard_pool_rule(rule: str) -> tuple[str, str, float]:
    """Parse rules like ``bridge_dist<=1.0`` or ``dist_h<0.6``.

    Returns ``(column, op, threshold)`` with op in {``<=``, ``<``, ``>=``, ``>``}.
    """
    text = rule.strip().replace(" ", "")
    for op in ("<=", ">=", "<", ">"):
        if op in text:
            col, thr_s = text.split(op, 1)
            if not col:
                raise ValueError(f"bad hard_pool_rule (empty column): {rule!r}")
            return col, op, float(thr_s)
    raise ValueError(f"hard_pool_rule must look like 'bridge_dist<=1.0' (got {rule!r})")


def apply_hard_pool_mask(
    column_values: Sequence[float],
    rule: str,
) -> list[bool]:
    """Boolean mask for rows inside the hard pool defined by ``rule``."""
    _col, op, thr = parse_simple_hard_pool_rule(rule)
    out: list[bool] = []
    for v in column_values:
        x = float(v)
        if op == "<=":
            out.append(x <= thr)
        elif op == "<":
            out.append(x < thr)
        elif op == ">=":
            out.append(x >= thr)
        else:
            out.append(x > thr)
    return out


def auc_full_and_hard_pool(
    scores: Sequence[float],
    is_positive: Sequence[bool],
    hard_mask: Sequence[bool],
    *,
    lower_is_better: bool = True,
    min_n: int = 20,
) -> dict[str, Any]:
    """Compute full-pool and hard-pool AUC with base rates.

    For distance-like scores (``bridge_dist``), set ``lower_is_better=True`` so
    ranking uses ``-score``. Citing full-pool AUC alone is non-compliant for
    ``U_relink_pair`` studies — always report both + n_pos/n_neg.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError("auc_full_and_hard_pool requires scikit-learn") from exc

    if not (len(scores) == len(is_positive) == len(hard_mask)):
        raise ValueError("scores, is_positive, hard_mask length mismatch")

    def _one(s: list[float], y: list[bool], *, label: str) -> dict[str, Any]:
        n = len(y)
        n_pos = sum(1 for v in y if v)
        n_neg = n - n_pos
        rec: dict[str, Any] = {
            "pool": label,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "base_rate": (n_pos / n) if n else 0.0,
            "auc": float("nan"),
            "skipped_reason": "",
        }
        if n < min_n:
            rec["skipped_reason"] = f"n<{min_n}"
            return rec
        if n_pos == 0 or n_neg == 0:
            rec["skipped_reason"] = "single_class"
            return rec
        rank = [-x for x in s] if lower_is_better else list(s)
        rec["auc"] = float(roc_auc_score(y, rank))
        return rec

    s_all = [float(x) for x in scores]
    y_all = [bool(v) for v in is_positive]
    full = _one(s_all, y_all, label="full")
    s_h = [s for s, m in zip(s_all, hard_mask, strict=True) if m]
    y_h = [y for y, m in zip(y_all, hard_mask, strict=True) if m]
    hard = _one(s_h, y_h, label="hard")
    return {
        "full": full,
        "hard": hard,
        "lower_is_better": lower_is_better,
        "citation_ok": (
            full.get("skipped_reason") == ""
            and hard.get("skipped_reason") == ""
            and full["n_pos"] > 0
        ),
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StudyMeta:
    """Contract for one signal-study directory."""

    study_id: str
    created_utc: str
    commit: str
    preset: str
    detector: str
    double_buffer: bool
    iou_match: float = 0.5
    universes: list[str] = field(default_factory=list)
    y_definitions: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_Y_DEFINITIONS)
    )
    score_fields: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SCORE_FIELDS)
    )
    method_ids: list[str] = field(default_factory=list)
    host: str = ""
    notes: str = ""
    # Optional: map MethodId -> DataSourceId for provenance
    method_sources: dict[str, str] = field(default_factory=dict)
    # Pipeline bookkeeping (how methods stack)
    cut_design: str = CutDesign.SINGLE.value
    pipeline_profile: str = "headline_m_whole_graph"  # path map name
    # Terminal node_id per method_id (defaults from METHOD_TERMINAL_NODE)
    method_terminal_nodes: dict[str, str] = field(default_factory=dict)
    # Ordered spine used for cumulative studies
    cumulative_spine: list[str] = field(default_factory=list)
    # Continuous / grid sweeps (see SweepAxis). Empty = no sweep.
    sweep_mode: str = ""
    sweep_axes: list[dict[str, Any]] = field(default_factory=list)
    # Method held fixed while axes vary (required when cut_design=sweep)
    sweep_base_method: str = ""
    # Offline relink (U_relink_pair) reporting contract — see offline_relink_candidate_analysis.md
    hard_pool_rule: str = (
        ""  # e.g. "bridge_dist<=1.0"; required if U_relink_pair in universes
    )
    report_base_rate: bool = True
    # Which dual-line recipe this study follows: A | B1 | B2 | "" (unset)
    study_line: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StudyMeta:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)  # type: ignore[arg-type]

    def parsed_sweep_axes(self) -> list[SweepAxis]:
        return [SweepAxis.from_mapping(a) for a in self.sweep_axes]

    def score_field_for(self, universe: UniverseId | str) -> str:
        key = universe.value if isinstance(universe, UniverseId) else universe
        if key not in self.score_fields:
            raise KeyError(f"No score_field for universe {key!r} in StudyMeta")
        return self.score_fields[key]

    def y_definition_for(self, universe: UniverseId | str) -> str:
        key = universe.value if isinstance(universe, UniverseId) else universe
        if key not in self.y_definitions:
            raise KeyError(f"No y_definition for universe {key!r} in StudyMeta")
        return self.y_definitions[key]

    def resolved_terminal(self, method_id: str) -> str | None:
        if method_id in self.method_terminal_nodes:
            return self.method_terminal_nodes[method_id]
        return METHOD_TERMINAL_NODE.get(method_id)

    def uses_relink_pair(self) -> bool:
        return UniverseId.U_RELINK_PAIR.value in self.universes

    def validate_pipeline_methods(self) -> None:
        """Raise ValueError if method_ids disagree with cut_design / order."""
        self._validate_universes_and_relink()
        if self.cut_design == CutDesign.SWEEP.value:
            self._validate_sweep()
        if not self.method_ids:
            return
        if self.cut_design == CutDesign.CUMULATIVE.value:
            validate_method_order(self.method_ids)
        for mid in self.method_ids:
            if mid == MethodId.CUSTOM.value:
                if mid not in self.method_terminal_nodes and not self.notes:
                    raise ValueError(
                        "custom method requires method_terminal_nodes entry or notes"
                    )
                continue
            term = self.resolved_terminal(mid)
            if term is None:
                raise ValueError(f"method_id {mid!r} has no terminal pipeline node")
            get_pipeline_node(term)  # raises if unknown

    def _validate_universes_and_relink(self) -> None:
        """Enforce universe naming + B1 relink reporting requirements."""
        if self.study_line not in ("", "A", "B1", "B2"):
            raise ValueError(f"study_line must be A|B1|B2|'' (got {self.study_line!r})")
        known = {u.value for u in UniverseId}
        for u in self.universes:
            if u not in known:
                raise ValueError(f"unknown universe {u!r}; known={sorted(known)}")
        if not self.uses_relink_pair():
            return
        if not self.hard_pool_rule.strip():
            raise ValueError(
                "universes includes U_relink_pair: hard_pool_rule is required "
                f"(default recommendation: {DEFAULT_RELINK_HARD_POOL_RULE!r}). "
                "Full-pool-only AUC is non-compliant for citation."
            )
        y_key = UniverseId.U_RELINK_PAIR.value
        y_def = self.y_definitions.get(y_key) or DEFAULT_Y_DEFINITIONS.get(y_key, "")
        if "gt_match" not in y_def:
            raise ValueError(
                "U_relink_pair y_definitions must mention gt_match "
                "(true relink under builder GT mapping)"
            )
        score = self.score_fields.get(y_key) or DEFAULT_SCORE_FIELDS.get(y_key, "")
        if not score:
            raise ValueError(
                "U_relink_pair requires score_fields['U_relink_pair'] "
                "(default bridge_dist)"
            )
        if self.study_line == "B1" and not self.report_base_rate:
            raise ValueError(
                "study_line=B1 requires report_base_rate=True "
                "(n_pos/n_neg must be reported with AUC)"
            )

    def _validate_sweep(self) -> None:
        if not self.sweep_axes:
            raise ValueError("cut_design=sweep requires non-empty sweep_axes")
        base = self.sweep_base_method or (self.method_ids[0] if self.method_ids else "")
        if not base:
            raise ValueError(
                "cut_design=sweep requires sweep_base_method or method_ids[0]"
            )
        if base != MethodId.CUSTOM.value and self.resolved_terminal(base) is None:
            raise ValueError(f"sweep_base_method {base!r} has no terminal node")
        if self.sweep_mode and self.sweep_mode not in {m.value for m in SweepMode}:
            raise ValueError(f"unknown sweep_mode={self.sweep_mode!r}")
        axes = self.parsed_sweep_axes()
        names = [a.name for a in axes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate sweep axis names: {names}")
        for axis in axes:
            axis.validate()
            pts = axis.expanded_values()
            if len(pts) < 2:
                raise ValueError(
                    f"sweep axis {axis.name!r} needs >=2 points for a range "
                    f"(got {len(pts)})"
                )


@dataclass(frozen=True)
class GtRow:
    seq: str
    frame: int
    gt_id: int
    x: float
    y: float
    w: float
    h: float
    matched: bool
    vis: float = float("nan")
    cls: int = 1
    height: float | None = None
    neighbors: float = float("nan")
    max_overlap: float = float("nan")
    frame_gt: int = -1
    match_score: float = float("nan")
    match_iou: float = float("nan")
    match_det_key: str = ""
    source: str = DataSourceId.MOT17_GT.value

    def __post_init__(self) -> None:
        if self.height is None:
            object.__setattr__(self, "height", float(self.h))


@dataclass(frozen=True)
class DetRow:
    seq: str
    frame: int
    stage: str
    det_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    is_tp: bool
    w: float = float("nan")
    h: float = float("nan")
    cls: int = 0
    gt_id: int = -1
    match_iou: float = float("nan")
    height: float = float("nan")
    vis: float = float("nan")
    det_key: str = ""
    source: str = DataSourceId.STAGE_DUMP_CSV.value

    def resolved_det_key(self) -> str:
        if self.det_key:
            return self.det_key
        return make_det_key(self.seq, self.frame, self.stage, self.det_idx)


@dataclass(frozen=True)
class CandRow:
    """Frame-level (track, det) association candidate — not a relink pair."""

    seq: str
    frame: int
    track_id: int
    det_idx: int
    is_correct: bool
    iou: float = float("nan")
    maha2: float = float("nan")
    cost: float = float("nan")
    affinity: float = float("nan")
    penalty: float = float("nan")
    score_det: float = float("nan")
    height_trk: float = float("nan")
    height_det: float = float("nan")
    speed: float = float("nan")
    occ_iou: float = float("nan")
    accepted: bool = False
    method: str = MethodId.CUSTOM.value
    cand_key: str = ""
    source: str = DataSourceId.DERIVED.value

    def resolved_cand_key(self) -> str:
        if self.cand_key:
            return self.cand_key
        return make_cand_key(self.seq, self.frame, self.track_id, self.det_idx)


@dataclass(frozen=True)
class RelinkPairRow:
    """Offline (lost → cand) pair from build_relink_candidates.py.

    ``gt_match`` may be int 0/1 in CSV; treat as bool for y.
    ``bridge_dist`` is lower-is-better; for sklearn roc_auc_score use ``-bridge_dist``.
    """

    seq: str
    lost_id: int
    cand_id: int
    gt_match: bool
    gt_valid: bool
    bridge_dist: float
    gap: int
    lost_last_frame: int
    cand_first_frame: int
    gt_lost: int = -1
    gt_cand: int = -1
    dist_h: float = float("nan")
    fwd_resid: float = float("nan")
    bwd_resid: float = float("nan")
    dir_cos: float = float("nan")
    speed_h: float = float("nan")
    accepted: bool = False
    already_linked: bool = False
    source: str = DataSourceId.DERIVED.value

    @staticmethod
    def gt_match_as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return bool(int(value))


@dataclass(frozen=True)
class ErrRow:
    seq: str
    frame: int
    event_id: str
    event_type: str
    method: str
    gt_id: int = -1
    track_id: int = -1
    height: float = float("nan")
    notes: str = ""
    source: str = DataSourceId.DERIVED.value


@dataclass(frozen=True)
class MethodMetricsRow:
    """E2E metrics for a single method cut (no param axes)."""

    method: str
    idf1: float | None = None
    mota: float | None = None
    hota: float | None = None
    deta: float | None = None
    assa: float | None = None
    ids: int | None = None
    fp: int | None = None
    fn: int | None = None
    rcll: float | None = None
    prcn: float | None = None
    n_seq: int | None = None
    output_dir: str = ""
    source: str = DataSourceId.MOTMETRICS_EVAL.value


@dataclass(frozen=True)
class RunMetricsRow:
    """One pipeline evaluation at a concrete param point (sweep-friendly).

    Store continuous axes as ``params``; CSV export flattens to ``param_<name>``.
    ``method`` is the base MethodId / terminal cut; ``run_id`` is unique.
    """

    run_id: str
    method: str
    params: dict[str, float] = field(default_factory=dict)
    idf1: float | None = None
    mota: float | None = None
    hota: float | None = None
    deta: float | None = None
    assa: float | None = None
    ids: int | None = None
    fp: int | None = None
    fn: int | None = None
    rcll: float | None = None
    prcn: float | None = None
    # Offline gate curves may only fill these:
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    tp: int | None = None
    n_seq: int | None = None
    output_dir: str = ""
    source: str = DataSourceId.MOTMETRICS_EVAL.value

    def to_flat_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "method": self.method,
            "idf1": self.idf1,
            "mota": self.mota,
            "hota": self.hota,
            "deta": self.deta,
            "assa": self.assa,
            "ids": self.ids,
            "fp": self.fp,
            "fn": self.fn,
            "rcll": self.rcll,
            "prcn": self.prcn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "n_seq": self.n_seq,
            "output_dir": self.output_dir,
            "source": self.source,
        }
        for k, v in self.params.items():
            d[f"param_{k}"] = v
            d[k] = v  # convenience for summarize_sweep_metric
        return d


# ---------------------------------------------------------------------------
# Keys & set algebra helpers (pure; work on sequences of bools/ids)
# ---------------------------------------------------------------------------


def make_det_key(seq: str, frame: int, stage: str, det_idx: int) -> str:
    return f"{seq}:{int(frame)}:{stage}:{int(det_idx)}"


def make_cand_key(seq: str, frame: int, track_id: int, det_idx: int) -> str:
    return f"{seq}:{int(frame)}:t{int(track_id)}:d{int(det_idx)}"


def make_event_id(
    seq: str,
    frame: int,
    event_type: str,
    *,
    gt_id: int = -1,
    track_id: int = -1,
) -> str:
    return f"{seq}:{int(frame)}:{event_type}:g{int(gt_id)}:t{int(track_id)}"


@dataclass(frozen=True)
class SetCompareResult:
    """Binary set comparison for two method masks on a shared index."""

    n: int
    both: int
    only_a: int
    only_b: int
    neither: int
    jaccard: float
    # When positive labels are available:
    tp_both: int | None = None
    tp_only_a: int | None = None
    tp_only_b: int | None = None
    fp_both: int | None = None
    fp_only_a: int | None = None
    fp_only_b: int | None = None


def compare_accept_masks(
    accept_a: Sequence[bool],
    accept_b: Sequence[bool],
    is_positive: Sequence[bool] | None = None,
) -> SetCompareResult:
    """Compare two accept masks (same length / shared universe index)."""
    if len(accept_a) != len(accept_b):
        raise ValueError("accept_a and accept_b must have the same length")
    n = len(accept_a)
    both = only_a = only_b = neither = 0
    for a, b in zip(accept_a, accept_b, strict=True):
        if a and b:
            both += 1
        elif a and not b:
            only_a += 1
        elif b and not a:
            only_b += 1
        else:
            neither += 1
    union = both + only_a + only_b
    jaccard = (both / union) if union else 0.0

    tp_both = tp_only_a = tp_only_b = None
    fp_both = fp_only_a = fp_only_b = None
    if is_positive is not None:
        if len(is_positive) != n:
            raise ValueError("is_positive must match accept mask length")
        tp_both = tp_only_a = tp_only_b = 0
        fp_both = fp_only_a = fp_only_b = 0
        for a, b, y in zip(accept_a, accept_b, is_positive, strict=True):
            if y:
                if a and b:
                    tp_both += 1
                elif a and not b:
                    tp_only_a += 1
                elif b and not a:
                    tp_only_b += 1
            else:
                if a and b:
                    fp_both += 1
                elif a and not b:
                    fp_only_a += 1
                elif b and not a:
                    fp_only_b += 1

    return SetCompareResult(
        n=n,
        both=both,
        only_a=only_a,
        only_b=only_b,
        neither=neither,
        jaccard=jaccard,
        tp_both=tp_both,
        tp_only_a=tp_only_a,
        tp_only_b=tp_only_b,
        fp_both=fp_both,
        fp_only_a=fp_only_a,
        fp_only_b=fp_only_b,
    )


def error_set_diff(
    event_ids_a: Iterable[str],
    event_ids_b: Iterable[str],
) -> tuple[set[str], set[str], set[str]]:
    """Return (only_a, only_b, both) event_id sets."""
    a = set(event_ids_a)
    b = set(event_ids_b)
    return a - b, b - a, a & b


# ---------------------------------------------------------------------------
# Validation & IO
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a table or meta violates the signal-table contract."""


def validate_columns(
    universe: UniverseId,
    columns: Iterable[str],
    *,
    extra_ok: bool = True,
) -> None:
    have = set(columns)
    need = set(UNIVERSE_REQUIRED[universe])
    missing = sorted(need - have)
    if missing:
        raise SchemaError(f"{universe.value} missing required columns: {missing}")
    if not extra_ok:
        extra = sorted(have - need)
        if extra:
            raise SchemaError(f"{universe.value} unexpected columns: {extra}")


def save_study_meta(meta: StudyMeta, study_dir: str | Path) -> Path:
    root = Path(study_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / META_FILENAME
    path.write_text(
        json.dumps(meta.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_study_meta(study_dir: str | Path) -> StudyMeta:
    path = Path(study_dir) / META_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SchemaError("meta.json must be a JSON object")
    meta = StudyMeta.from_mapping(data)
    if not meta.study_id or not meta.commit or not meta.preset:
        raise SchemaError("meta.json requires study_id, commit, preset")
    if meta.iou_match <= 0.0 or meta.iou_match > 1.0:
        raise SchemaError(f"invalid iou_match={meta.iou_match}")
    try:
        meta.validate_pipeline_methods()
    except ValueError as exc:
        raise SchemaError(str(exc)) from exc
    return meta


def study_table_path(study_dir: str | Path, universe: UniverseId) -> Path:
    return Path(study_dir) / TABLE_FILENAMES[universe]


def dataframe_to_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize row mappings to plain dicts (for parquet writers)."""
    return [dict(r) for r in rows]


def try_import_pandas() -> Any:
    import pandas as pd

    return pd


def read_universe_table(
    study_dir: str | Path,
    universe: UniverseId,
    *,
    validate: bool = True,
) -> Any:
    """Load a universe table as a pandas DataFrame."""
    pd = try_import_pandas()
    path = study_table_path(study_dir, universe)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if validate:
        validate_columns(universe, df.columns)
    return df


def write_universe_table(
    study_dir: str | Path,
    universe: UniverseId,
    df: Any,
    *,
    validate: bool = True,
) -> Path:
    """Write a pandas DataFrame to the standard study path."""
    if validate:
        validate_columns(universe, df.columns)
    root = Path(study_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = study_table_path(root, universe)
    if universe is UniverseId.METHOD_METRICS or path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def stratified_auc_table(
    df: Any,
    *,
    y_col: str,
    score_col: str,
    bin_col: str,
    min_n: int = 50,
) -> Any:
    """Compute per-bin AUC; skip bins with n < min_n or single class.

    Returns a DataFrame: bin, n, n_frac, auc, med_pos, med_neg, skipped_reason.
    """
    pd = try_import_pandas()
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError("stratified_auc_table requires scikit-learn") from exc

    total = len(df)
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(bin_col, observed=True):
        n = len(g)
        rec: dict[str, Any] = {
            "bin": key,
            "n": n,
            "n_frac": (n / total) if total else 0.0,
            "auc": float("nan"),
            "med_pos": float("nan"),
            "med_neg": float("nan"),
            "skipped_reason": "",
        }
        if n < min_n:
            rec["skipped_reason"] = f"n<{min_n}"
            rows.append(rec)
            continue
        y = g[y_col].astype(bool)
        if y.nunique() < 2:
            rec["skipped_reason"] = "single_class"
            rows.append(rec)
            continue
        s = g[score_col].astype(float)
        rec["auc"] = float(roc_auc_score(y, s))
        rec["med_pos"] = float(s[y].median())
        rec["med_neg"] = float(s[~y].median())
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Provenance map (documentation helper)
# ---------------------------------------------------------------------------

DATA_SOURCE_PRODUCES: dict[DataSourceId, tuple[str, ...]] = {
    DataSourceId.MOT17_GT: ("U_gt geometry", "vis"),
    DataSourceId.STAGE_DUMP_CSV: ("U_det pre-label", *DET_STAGE_DUMP_COLUMNS),
    DataSourceId.SCORE_DIST_TOOL: ("U_gt match_score", "height/vis covariates"),
    DataSourceId.FP_BY_HEIGHT_TOOL: ("height-binned FP summary",),
    DataSourceId.MOT_RESULT: ("track emit boxes", "U_err inputs"),
    DataSourceId.MOTMETRICS_EVAL: ("MethodMetricsRow CLEAR/IDF1",),
    DataSourceId.TRACKEVAL_HOTA: ("HOTA DetA AssA",),
    DataSourceId.NEUTRAL_NOGO_TOOL: ("legacy U_cand-style scores",),
    DataSourceId.NEAR_MISS_STAGE: ("GT↔stage join",),
    DataSourceId.PIPELINE_CONTRIBUTION: ("cumulative MethodMetricsRow",),
    DataSourceId.MANUAL_LABEL: ("any y column",),
    DataSourceId.DERIVED: ("keys", "IoU", "set diffs"),
}
