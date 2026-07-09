"""Signal-table schema for transparent multi-method / multi-dimension analysis.

Normative doc: ``docs/research/eval/signal_table_schema.md``.

Design:
  * One universe per table (U_gt / U_det / U_cand / U_err).
  * Labels ``y`` are frozen by ``StudyMeta``; do not re-greedy with a new IoU
    inside ad-hoc scripts without a new study_id.
  * Methods are gates or pipeline cuts; dimensions are covariates for strata.
  * Pipeline nodes fix **layer + order + parents** so studies can be stacked.
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
    """Sample universe — one parquet table per value."""

    U_GT = "U_gt"
    U_DET = "U_det"
    U_CAND = "U_cand"
    U_ERR = "U_err"
    METHOD_METRICS = "method_metrics"  # small summary, not frame-level


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
    CUSTOM = "custom"


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
ERR_REQUIRED: tuple[str, ...] = (
    "seq",
    "frame",
    "event_id",
    "event_type",
    "method",
)
METHOD_METRICS_REQUIRED: tuple[str, ...] = ("method",)

UNIVERSE_REQUIRED: dict[UniverseId, tuple[str, ...]] = {
    UniverseId.U_GT: GT_REQUIRED,
    UniverseId.U_DET: DET_REQUIRED,
    UniverseId.U_CAND: CAND_REQUIRED,
    UniverseId.U_ERR: ERR_REQUIRED,
    UniverseId.METHOD_METRICS: METHOD_METRICS_REQUIRED,
}

DEFAULT_Y_DEFINITIONS: dict[str, str] = {
    UniverseId.U_GT.value: (
        "matched if some det IoU>=iou_match under score-greedy claim"
    ),
    UniverseId.U_DET.value: ("is_tp if claimed a GT with IoU>=iou_match; else FP"),
    UniverseId.U_CAND.value: (
        "is_correct if det is the GT-matched target of this track at frame"
    ),
    UniverseId.U_ERR.value: (
        "event_type in {fn_miss, id_switch, fp_birth, fp_interp, fragment}"
    ),
}

DEFAULT_SCORE_FIELDS: dict[str, str] = {
    UniverseId.U_GT.value: "match_score",
    UniverseId.U_DET.value: "score",
    UniverseId.U_CAND.value: "iou",  # or -cost; override in StudyMeta
}

TABLE_FILENAMES: dict[UniverseId, str] = {
    UniverseId.U_GT: "u_gt.parquet",
    UniverseId.U_DET: "u_det.parquet",
    UniverseId.U_CAND: "u_cand.parquet",
    UniverseId.U_ERR: "u_err.parquet",
    UniverseId.METHOD_METRICS: "metrics_by_method.csv",
}

META_FILENAME = "meta.json"

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

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StudyMeta:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)  # type: ignore[arg-type]

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

    def validate_pipeline_methods(self) -> None:
        """Raise ValueError if method_ids disagree with cut_design / order."""
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
