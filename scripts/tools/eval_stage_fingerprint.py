"""Opt-in per-stage fingerprints for issue #363 (hash-first, dump-on-divergence).

This is observability, not a causal-mechanism claim.  A first divergent
stage is the first producer-facing execution boundary where two
fixed-config evals produced different canonicalized values.  Locating
that boundary does not identify a buffer, stream, or race.

Lives under ``scripts/tools/``, not ``src/``: the comparator must not
move the published implementation identity axis.  Collection is injected
into a child eval process; default production eval does not call this
module.

GPU stages are snapshotted with a D2D clone on the current stream, then
copied D2H on a dedicated copy stream.  GPU clones are not retained until
end of eval.  Per-frame CUDA fences are not inserted.
"""
# status: stable

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_ID = "eval_stage_fingerprint_v1"

# Producer-facing order.  Earlier stages bound later ones.  post-decode
# input is intentionally absent: hashing full frames would add a D2H of
# the decoded image every frame, which this instrumentation refuses.
STAGES: tuple[str, ...] = (
    "detector_output",
    "post_nms",
    "tracker_input",
    "tracker_output",
    "mot",
    "mot_file",
)
STAGE_INDEX = {name: index for index, name in enumerate(STAGES)}
DETECTION_STAGES = frozenset(
    ("detector_output", "post_nms", "tracker_input", "tracker_output")
)
MOT_STAGES = frozenset(("mot", "mot_file"))
FINGERPRINT_DIRNAME = "stage_fingerprint"
MANIFEST_NAME = "manifest.json"
LOG_NAME = "fingerprints.jsonl"
DIVERGENCE_NAME = "first_divergence.json"

KIND_FIRST_OBSERVABLE = "first_observable_divergence"
KIND_INCOMPLETE = "incomplete"
KIND_INSUFFICIENT = "instrumentation_insufficient"

# Frozen before the live Block S localization.  Do not retune after seeing
# results.  ``sufficient`` means the producing-path boundary is concrete
# enough for condition 2; it does not close the issue and is not a mechanism.
VERDICT_INSUFFICIENT = "insufficient"
VERDICT_CANDIDATE_SUFFICIENT = "candidate_sufficient"
VERDICT_SUFFICIENT = "sufficient"
VERDICT_NOT_APPLICABLE = "not_applicable"

_LAST_IDENTICAL = {
    "detector_output": None,
    "post_nms": "detector_output",
    "tracker_input": "post_nms",
    "tracker_output": "tracker_input",
    "mot": "tracker_output",
    "mot_file": "mot",
}


@dataclass(frozen=True)
class Condition2Rule:
    """What a first-divergent stage is allowed to claim for #363 condition 2."""

    first_divergent_stage: str
    last_identical_stage: str | None
    producing_path_verdict: str
    allowed_claim: str


# Authority for condition 2.  ``candidate_sufficient`` / ``sufficient`` are
# producing-path boundaries, not issue-close and not causal mechanism.
CONDITION2_RULES: dict[str, Condition2Rule] = {
    "detector_output": Condition2Rule(
        first_divergent_stage="detector_output",
        last_identical_stage=None,
        producing_path_verdict=VERDICT_INSUFFICIENT,
        allowed_claim=(
            "divergence 已在 detect_fn output 出現；"
            "decode / preprocess / detector internals 未切開"
        ),
    ),
    "post_nms": Condition2Rule(
        first_divergent_stage="post_nms",
        last_identical_stage="detector_output",
        producing_path_verdict=VERDICT_CANDIDATE_SUFFICIENT,
        allowed_claim="divergence 被界定在 detector output → evaluator NMS",
    ),
    "tracker_input": Condition2Rule(
        first_divergent_stage="tracker_input",
        last_identical_stage="post_nms",
        producing_path_verdict=VERDICT_INSUFFICIENT,
        allowed_claim=(
            "divergence 在 post-NMS → _run_track 前產生；ReID / GMC 路徑仍多義"
        ),
    ),
    "tracker_output": Condition2Rule(
        first_divergent_stage="tracker_output",
        last_identical_stage="tracker_input",
        producing_path_verdict=VERDICT_INSUFFICIENT,
        allowed_claim=(
            "divergence 在 tracker execution 內產生；"
            "Kalman / association / buffer 未切開"
        ),
    ),
    "mot": Condition2Rule(
        first_divergent_stage="mot",
        last_identical_stage="tracker_output",
        producing_path_verdict=VERDICT_SUFFICIENT,
        allowed_claim=("divergence 在 global-ID mapping / serialization 路徑產生"),
    ),
    "mot_file": Condition2Rule(
        first_divergent_stage="mot_file",
        last_identical_stage="mot",
        producing_path_verdict=VERDICT_SUFFICIENT,
        allowed_claim="divergence 在 sequence-level postprocess 路徑產生",
    ),
}
if tuple(CONDITION2_RULES) != STAGES:
    raise RuntimeError("CONDITION2_RULES must cover STAGES in pipeline order")

INSTRUMENTATION_INSUFFICIENT_CLAIM = (
    "MOT diverged but every instrumented stage hash matched; "
    "instrumentation insufficient"
)

# Frozen before the next instrumented Block S localization session.
# A termination condition, not a rate sample size.  The 2026-09-07
# unbudgeted hunt does not count toward this budget.
# This is the Block S contract only, not a global cap.
LOCALIZATION_BUDGET_RUNS = 16
SESSION_PAIR_FOUND = "pair_found"
SESSION_IN_PROGRESS = "in_progress"
SESSION_BUDGET_EXHAUSTED_IDENTICAL = "budget_exhausted_identical"
LOCALIZATION_CONFIG_BLOCK_S = "block_s"
LOCALIZATION_CONFIG_GPU_DECODE = "gpu_decode"

BUDGET_EXHAUSTED_CLAIM = (
    "divergence was not observed under the instrumented Block S executions "
    "within the preregistered localization budget"
)

# Frozen before the instrumented GPU-decode localization session.
# Distinct from LOCALIZATION_BUDGET_RUNS: that 16 is the Block S session
# contract, not a global constant.  Historical arm G evidence was n=8
# and already produced a pair; 8 is this session's cap, not a rate sample.
GPU_DECODE_LOCALIZATION_BUDGET_RUNS = 8
GPU_DECODE_BUDGET_EXHAUSTED_CLAIM = (
    "divergence was not observed under the instrumented GPU-decode "
    "executions within the preregistered localization budget"
)

LOCALIZATION_BUDGETS: dict[str, int] = {
    LOCALIZATION_CONFIG_BLOCK_S: LOCALIZATION_BUDGET_RUNS,
    LOCALIZATION_CONFIG_GPU_DECODE: GPU_DECODE_LOCALIZATION_BUDGET_RUNS,
}
BUDGET_EXHAUSTED_CLAIMS: dict[str, str] = {
    LOCALIZATION_CONFIG_BLOCK_S: BUDGET_EXHAUSTED_CLAIM,
    LOCALIZATION_CONFIG_GPU_DECODE: GPU_DECODE_BUDGET_EXHAUSTED_CLAIM,
}

# Frozen issue-level reading when CONDITION2_RULES returns sufficient.
# The localization run records the table result; it does not close #363.
CONDITION2_SUFFICIENT_SESSION_CLAIM = (
    "Condition 1 remains unresolved. Condition 2 has produced a frozen-table "
    "`sufficient` producing-path result and awaits issue-level closure review; "
    "the localization run itself does not close #363."
)


@dataclass(frozen=True)
class LocalizationSessionReading:
    """Session-level localization outcome.  Distinct from CONDITION2_RULES."""

    kind: str
    n_runs: int
    budget_runs: int
    divergent_pair: bool
    apply_condition2_rules: bool
    allowed_claim: str
    producing_path_unresolved: bool
    config: str = LOCALIZATION_CONFIG_BLOCK_S
    issue_close: bool = False
    mechanism_claim: bool = False
    condition_1_advanced: bool = False
    condition_2_advanced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def localization_budget(config: str) -> int:
    if config not in LOCALIZATION_BUDGETS:
        raise ValueError(f"unknown localization config {config!r}")
    return LOCALIZATION_BUDGETS[config]


def localization_exhausted_claim(config: str) -> str:
    if config not in BUDGET_EXHAUSTED_CLAIMS:
        raise ValueError(f"unknown localization config {config!r}")
    return BUDGET_EXHAUSTED_CLAIMS[config]


def read_localization_session(
    *,
    n_runs: int,
    divergent_pair: bool,
    config: str = LOCALIZATION_CONFIG_BLOCK_S,
    producing_path_verdict: str | None = None,
) -> LocalizationSessionReading:
    """Frozen localization-session reading for one preregistered config.

    A pair inside that config's budget applies ``CONDITION2_RULES``.
    ``sufficient`` is condition-2 evidence from that table; it does not
    set ``issue_close``.  Exhausting the budget with identical runs does
    not mean divergence disappeared, and does not run the table.  Block S
    budget 16 is not a global cap.
    """

    budget_runs = localization_budget(config)
    exhausted_claim = localization_exhausted_claim(config)
    if n_runs < 2:
        raise ValueError("localization session needs at least two runs")
    if n_runs > budget_runs:
        raise ValueError(
            f"localization session n_runs={n_runs} exceeds preregistered "
            f"{config} budget {budget_runs}"
        )
    if divergent_pair:
        sufficient = producing_path_verdict == VERDICT_SUFFICIENT
        return LocalizationSessionReading(
            kind=SESSION_PAIR_FOUND,
            n_runs=n_runs,
            budget_runs=budget_runs,
            divergent_pair=True,
            apply_condition2_rules=True,
            allowed_claim=CONDITION2_SUFFICIENT_SESSION_CLAIM if sufficient else "",
            producing_path_unresolved=not sufficient,
            config=config,
            condition_2_advanced=sufficient,
        )
    if n_runs < budget_runs:
        return LocalizationSessionReading(
            kind=SESSION_IN_PROGRESS,
            n_runs=n_runs,
            budget_runs=budget_runs,
            divergent_pair=False,
            apply_condition2_rules=False,
            allowed_claim="",
            producing_path_unresolved=True,
            config=config,
        )
    return LocalizationSessionReading(
        kind=SESSION_BUDGET_EXHAUSTED_IDENTICAL,
        n_runs=n_runs,
        budget_runs=budget_runs,
        divergent_pair=False,
        apply_condition2_rules=False,
        allowed_claim=exhausted_claim,
        producing_path_unresolved=True,
        config=config,
    )


# Frozen before the instrumented/uninstrumented observer-effect measurement.
# This is not a rate sample, not a localization session, and does not move
# CONDITION2_RULES or LOCALIZATION_BUDGET_RUNS.
OE_IDENTIFIED = "observer_effect_identified"
OE_BOUNDED = "observer_effect_bounded"
OE_NONE = "no_material_perturbation"

# One CUDA caching-allocator block.  Reserved growth below this is slack.
ALLOCATOR_RESERVED_MATERIAL_BYTES = 2 * 1024 * 1024

OBSERVER_EFFECT_IDENTIFIED_CLAIM = (
    "enabling per-stage fingerprinting materially perturbs producing-path "
    "execution conditions at the inspected runtime boundaries"
)
OBSERVER_EFFECT_BOUNDED_CLAIM = (
    "fingerprint extra work exists but is bounded: no extra device-wide sync "
    "during the frame loop and no snapshot-attributable caching-allocator "
    "reserved growth; remaining ops are detection-sized D2D clones without "
    "fence, async D2H on a copy stream, and host hashing of already-copied "
    "MOT data"
)
OBSERVER_EFFECT_NONE_CLAIM = (
    "no material perturbation found at the inspected runtime boundaries"
)

# Static inventory of fingerprint-inserted ops.  Frozen before measurement.
# ``before_mot_write`` means the extra work runs while later frames, or later
# stages of the same frame, can still change MOT bytes.
OBSERVER_EFFECT_SITES: dict[str, dict[str, Any]] = {
    "detector_output_clone": {
        "stage": "detector_output",
        "op": "d2d_clone_no_fence",
        "stream": "current",
        "before_mot_write": True,
        "device_wide_sync": False,
        "note": "after _run_detect; Block S already full-device-syncs at detect return",
    },
    "post_nms_clone": {
        "stage": "post_nms",
        "op": "d2d_clone_no_fence",
        "stream": "current",
        "before_mot_write": True,
        "device_wide_sync": False,
        "note": "NMS→track span has no full-device barrier",
    },
    "tracker_input_clone": {
        "stage": "tracker_input",
        "op": "d2d_clone_no_fence",
        "stream": "current",
        "before_mot_write": True,
        "device_wide_sync": False,
        "note": "immediately before _run_track",
    },
    "gpu_snapshot_retention": {
        "stage": None,
        "op": "in_flight_gpu_clones_until_async_d2h",
        "stream": "copy",
        "before_mot_write": True,
        "device_wide_sync": False,
        "note": "GPU clones live only until copy-stream D2H completes; hashed from host",
    },
    "tracker_output_host_hash": {
        "stage": "tracker_output",
        "op": "host_hash",
        "stream": None,
        "before_mot_write": True,
        "device_wide_sync": False,
        "note": "after emit already copied track_results to host; delays next launch",
    },
    "mot_host_hash": {
        "stage": "mot",
        "op": "host_hash",
        "stream": None,
        "before_mot_write": True,
        "device_wide_sync": False,
        "note": "hashes emit lines already produced for this frame",
    },
    "mot_file_host_hash": {
        "stage": "mot_file",
        "op": "host_hash",
        "stream": None,
        "before_mot_write": False,
        "device_wide_sync": False,
        "note": "sequence_result_callback runs after the MOT file is written",
    },
    "finalize_synchronize": {
        "stage": None,
        "op": "copy_stream_event_wait_then_host_hash",
        "stream": "copy",
        "before_mot_write": False,
        "device_wide_sync": False,
        "note": "collector.finalize after mot17.py returns; waits copy-stream events, not a device-wide join",
    },
}


@dataclass(frozen=True)
class ObserverEffectReading:
    """Instrumentation-trust reading.  Distinct from CONDITION2_RULES."""

    kind: str
    extra_device_sync_during_frame_loop: bool
    allocator_reserved_grew_from_snapshots: bool
    producing_path_gpu_clone: bool
    post_eval_synchronize: bool
    allowed_claim: str
    issue_close: bool = False
    mechanism_claim: bool = False
    condition_1_advanced: bool = False
    condition_2_advanced: bool = False
    localization_budget_reopened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_observer_effect(
    *,
    extra_device_sync_during_frame_loop: bool,
    allocator_reserved_grew_from_snapshots: bool,
    producing_path_gpu_clone: bool,
    post_eval_synchronize: bool,
) -> ObserverEffectReading:
    """Frozen observer-effect reading for the #363 fingerprint instrument.

    Identified means the instrument can perturb producing-path execution
    conditions.  Bounded means extra work exists but did not add an in-loop
    device join or snapshot-attributable reserved growth.  Neither reading
    advances condition 1/2 or closes the issue.
    """

    identified = bool(
        extra_device_sync_during_frame_loop or allocator_reserved_grew_from_snapshots
    )
    if identified:
        kind = OE_IDENTIFIED
        claim = OBSERVER_EFFECT_IDENTIFIED_CLAIM
    elif producing_path_gpu_clone or post_eval_synchronize:
        kind = OE_BOUNDED
        claim = OBSERVER_EFFECT_BOUNDED_CLAIM
    else:
        kind = OE_NONE
        claim = OBSERVER_EFFECT_NONE_CLAIM
    return ObserverEffectReading(
        kind=kind,
        extra_device_sync_during_frame_loop=bool(extra_device_sync_during_frame_loop),
        allocator_reserved_grew_from_snapshots=bool(
            allocator_reserved_grew_from_snapshots
        ),
        producing_path_gpu_clone=bool(producing_path_gpu_clone),
        post_eval_synchronize=bool(post_eval_synchronize),
        allowed_claim=claim,
    )


def measure_allocator_reserved_growth(
    *,
    uninstrumented_reserved_bytes: int,
    instrumented_reserved_bytes: int,
    n_reserved_increases_on_clone: int,
    material_bytes: int = ALLOCATOR_RESERVED_MATERIAL_BYTES,
) -> bool:
    """True when snapshot retention expanded the caching-allocator pool."""

    if int(n_reserved_increases_on_clone) > 0:
        return True
    delta = int(instrumented_reserved_bytes) - int(uninstrumented_reserved_bytes)
    return delta >= int(material_bytes)


@dataclass(frozen=True)
class Condition2Reading:
    last_identical_stage: str | None
    first_divergent_stage: str | None
    producing_path_verdict: str
    allowed_claim: str
    issue_close: bool = False
    mechanism_claim: bool = False


def read_condition_2(*, kind: str, stage: str | None) -> Condition2Reading:
    """Frozen #363 condition-2 reading.  Unknown stages fail closed."""

    if kind == KIND_INCOMPLETE:
        return Condition2Reading(
            last_identical_stage=None,
            first_divergent_stage=None,
            producing_path_verdict=VERDICT_NOT_APPLICABLE,
            allowed_claim=(
                "fingerprints missing or incomplete; no condition-2 reading"
            ),
        )
    if kind == KIND_INSUFFICIENT:
        return Condition2Reading(
            last_identical_stage="mot_file",
            first_divergent_stage=None,
            producing_path_verdict=VERDICT_INSUFFICIENT,
            allowed_claim=INSTRUMENTATION_INSUFFICIENT_CLAIM,
        )
    if kind != KIND_FIRST_OBSERVABLE:
        raise ValueError(f"unknown divergence kind {kind!r}")
    if stage not in CONDITION2_RULES:
        raise ValueError(f"no condition-2 rule for stage {stage!r}")
    if _LAST_IDENTICAL[stage] != CONDITION2_RULES[stage].last_identical_stage:
        raise ValueError(f"last-identical predecessor drifted for {stage}")
    rule = CONDITION2_RULES[stage]
    return Condition2Reading(
        last_identical_stage=rule.last_identical_stage,
        first_divergent_stage=rule.first_divergent_stage,
        producing_path_verdict=rule.producing_path_verdict,
        allowed_claim=rule.allowed_claim,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int32)
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and isinstance(value, torch_mod.Tensor):
        return np.ascontiguousarray(value.detach().cpu().contiguous().numpy())
    return np.ascontiguousarray(np.asarray(value))


def _is_cuda_tensor(value: Any) -> bool:
    torch_mod = sys.modules.get("torch")
    return (
        torch_mod is not None
        and isinstance(value, torch_mod.Tensor)
        and value.device.type == "cuda"
    )


def _snapshot_tensor(value: Any) -> Any:
    """Host copy or D2D clone.  Does not fence the current CUDA stream."""

    if value is None:
        return None
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and isinstance(value, torch_mod.Tensor):
        if value.device.type == "cuda":
            return value.detach().clone()
        return np.ascontiguousarray(value.detach().contiguous().cpu().numpy())
    return np.ascontiguousarray(np.asarray(value))


def _tensor_nbytes(value: Any) -> int:
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    nelement = getattr(value, "nelement", None)
    element_size = getattr(value, "element_size", None)
    if callable(nelement) and callable(element_size):
        return int(nelement()) * int(element_size())
    return 0


def _cuda_allocator_snapshot() -> dict[str, int]:
    torch_mod = sys.modules.get("torch")
    if (
        torch_mod is None
        or not getattr(torch_mod.cuda, "is_available", lambda: False)()
    ):
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    try:
        stats = torch_mod.cuda.memory_stats()
    except Exception:
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0) or 0),
        "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0) or 0),
    }


def _array_bytes(name: str, array: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(array)
    return (
        name.encode("utf-8")
        + b"\0"
        + arr.dtype.str.encode("ascii")
        + b"\0"
        + str(tuple(int(dim) for dim in arr.shape)).encode("ascii")
        + b"\0"
        + arr.tobytes()
    )


def ordered_bit_hash(
    *,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    ids: np.ndarray | None = None,
) -> str:
    payload = bytearray()
    payload.extend(_array_bytes("boxes", boxes))
    payload.extend(_array_bytes("scores", scores))
    payload.extend(_array_bytes("classes", classes))
    if ids is not None:
        payload.extend(_array_bytes("ids", ids))
    return _sha256(bytes(payload))


def _serialized_int(value: float, decimals: int, scale: int) -> int:
    text = f"{float(value):.{decimals}f}"
    return int(Decimal(text) * scale)


def canonical_rows(
    *,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    ids: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Integer rows at MOT serialization scale (centipixel / 1e-4 score).

    ``boxes`` are xyxy.  Rounding follows the emit ``.2f`` / ``.4f`` text,
    so a canonical match with MOT geometry is meaningful.  Bit hashes are
    the identity used for first-divergence.
    """

    boxes_arr = np.ascontiguousarray(boxes, dtype=np.float64)
    scores_arr = np.ascontiguousarray(scores, dtype=np.float64).reshape(-1)
    classes_arr = np.ascontiguousarray(classes).reshape(-1)
    n = int(scores_arr.shape[0])
    if boxes_arr.size == 0 or n == 0:
        return ()
    if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
        raise ValueError(f"boxes must be (N, 4) xyxy, got {boxes_arr.shape}")
    if boxes_arr.shape[0] != n:
        raise ValueError("boxes and scores length mismatch")
    if classes_arr.shape[0] != n:
        raise ValueError("classes and scores length mismatch")
    id_arr: np.ndarray | None = None
    if ids is not None:
        id_arr = np.ascontiguousarray(ids).reshape(-1)
        if id_arr.shape[0] != n:
            raise ValueError("ids and scores length mismatch")
    rows: list[tuple[int, ...]] = []
    for index in range(n):
        x1, y1, x2, y2 = (float(v) for v in boxes_arr[index])
        row = (
            _serialized_int(x1, 2, 100),
            _serialized_int(y1, 2, 100),
            _serialized_int(x2 - x1, 2, 100),
            _serialized_int(y2 - y1, 2, 100),
            _serialized_int(float(scores_arr[index]), 4, 10_000),
            int(classes_arr[index]),
            int(id_arr[index]) if id_arr is not None else -1,
        )
        rows.append(row)
    return tuple(rows)


def multiset_canonical_hash(rows: Sequence[Sequence[int]], *, id_free: bool) -> str:
    payload = bytearray()
    payload.extend(len(rows).to_bytes(8, "little"))
    keyed: list[tuple[int, ...]]
    if id_free:
        keyed = [tuple(int(v) for v in row[:6]) for row in rows]
    else:
        keyed = [tuple(int(v) for v in row) for row in rows]
    for row in sorted(keyed):
        payload.extend(len(row).to_bytes(2, "little"))
        for value in row:
            payload.extend(int(value).to_bytes(8, "little", signed=True))
    return _sha256(bytes(payload))


def fingerprint_detections(
    *,
    boxes: Any,
    scores: Any,
    classes: Any,
    ids: Any = None,
) -> dict[str, Any]:
    boxes_np = _as_numpy(boxes)
    scores_np = _as_numpy(scores).reshape(-1)
    classes_np = _as_numpy(classes).reshape(-1)
    ids_np = None if ids is None else _as_numpy(ids).reshape(-1)
    if boxes_np.size == 0:
        boxes_np = np.empty((0, 4), dtype=np.float32)
        scores_np = np.empty((0,), dtype=np.float32)
        classes_np = np.empty((0,), dtype=np.int32)
        if ids is not None:
            ids_np = np.empty((0,), dtype=np.int32)
    rows = canonical_rows(
        boxes=boxes_np,
        scores=scores_np,
        classes=classes_np,
        ids=ids_np,
    )
    return {
        "count": int(scores_np.shape[0]),
        "ordered_bit_hash": ordered_bit_hash(
            boxes=boxes_np,
            scores=scores_np,
            classes=classes_np,
            ids=ids_np,
        ),
        "multiset_canonical_hash": multiset_canonical_hash(rows, id_free=False),
        "id_free_canonical_hash": multiset_canonical_hash(rows, id_free=True),
        "rows": [list(row) for row in rows],
    }


def fingerprint_mot_lines(lines: Iterable[str]) -> dict[str, Any]:
    nonempty = [line for line in lines if str(line).strip()]
    body = "\n".join(nonempty)
    if nonempty:
        body += "\n"
    rows: list[list[int]] = []
    for line in nonempty:
        columns = [part.strip() for part in str(line).split(",")]
        if len(columns) < 7:
            raise ValueError(f"MOT line has fewer than 7 columns: {line!r}")
        frame = int(columns[0])
        track_id = int(columns[1])
        x = int(Decimal(columns[2]) * 100)
        y = int(Decimal(columns[3]) * 100)
        w = int(Decimal(columns[4]) * 100)
        h = int(Decimal(columns[5]) * 100)
        score = int(Decimal(columns[6]) * 10_000)
        rows.append([x, y, w, h, score, -1, track_id, frame])
    tuple_rows = tuple(tuple(row) for row in rows)
    return {
        "count": len(nonempty),
        "ordered_bit_hash": _sha256(body.encode("utf-8")),
        "multiset_canonical_hash": multiset_canonical_hash(tuple_rows, id_free=False),
        "id_free_canonical_hash": multiset_canonical_hash(
            tuple(row[:6] for row in tuple_rows), id_free=True
        ),
        "rows": rows,
    }


def record_key(sequence: str, frame: int, stage: str) -> tuple[str, int, str]:
    return (sequence, int(frame), stage)


def sort_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            str(item["sequence"]),
            int(item["frame"]),
            STAGE_INDEX.get(str(item["stage"]), len(STAGES)),
            str(item["stage"]),
        ),
    )


@dataclass
class PendingGpuSnapshot:
    sequence: str
    frame: int
    stage: str
    boxes: Any
    scores: Any
    classes: Any
    ids: Any = None
    host_boxes: Any = None
    host_scores: Any = None
    host_classes: Any = None
    host_ids: Any = None
    d2h_event: Any = None


# Cap in-flight GPU clones so snapshot retention cannot grow the
# caching-allocator pool across the sequence.  Detection-sized D2H is
# short; this is a backstop, not a per-frame fence.
_MAX_IN_FLIGHT_GPU_SNAPSHOTS = 24


class StageFingerprintCollector:
    """Hash-first collector.  GPU tensors are cloned D2D, then async D2H."""

    def __init__(self, output_dir: Path, *, include_payloads: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.include_payloads = include_payloads
        self.records: list[dict[str, Any]] = []
        self._gpu: list[PendingGpuSnapshot] = []
        self._host: list[PendingGpuSnapshot] = []
        self._seen: set[tuple[str, int, str]] = set()
        self.observer_samples: list[dict[str, Any]] = []
        self._copy_stream: Any = None
        self.limitations: list[str] = [
            "no per-frame CUDA fence; GPU snapshots are D2D clones on the current stream",
            "GPU clones are D2H'd on a dedicated copy stream and not retained until end of eval",
            "post-decode/input is not instrumented: hashing frames would add a full-image D2H",
            "detector_output is detect_fn output, not a split of raw head vs detector postprocess",
            "tracker_output is the host track_results at MOT emit, not GPU tracker buffers",
            "mot is per-frame emit lines; mot_file is sequence-level lines after merge/interpolate",
        ]

    def _store(
        self, sequence: str, frame: int, stage: str, payload: dict[str, Any]
    ) -> None:
        if stage not in STAGE_INDEX:
            raise ValueError(f"unknown stage {stage!r}")
        key = record_key(sequence, frame, stage)
        if key in self._seen:
            return
        self._seen.add(key)
        row = {
            "sequence": sequence,
            "frame": int(frame),
            "stage": stage,
            "count": int(payload["count"]),
            "ordered_bit_hash": payload["ordered_bit_hash"],
            "multiset_canonical_hash": payload["multiset_canonical_hash"],
            "id_free_canonical_hash": payload["id_free_canonical_hash"],
        }
        if self.include_payloads:
            row["rows"] = payload["rows"]
        self.records.append(row)

    def _record_clone_sample(
        self,
        stage: str,
        clone_nbytes: int,
        *,
        before: Mapping[str, int],
        after: Mapping[str, int],
    ) -> None:
        reserved = int(after["reserved_bytes"])
        self.observer_samples.append(
            {
                "stage": stage,
                "clone_nbytes": int(clone_nbytes),
                "allocated_bytes": int(after["allocated_bytes"]),
                "reserved_bytes": reserved,
                "reserved_increase": reserved > int(before["reserved_bytes"]),
            }
        )

    def observer_effect_measured(self) -> dict[str, Any]:
        samples = list(self.observer_samples)
        reserved_values = [int(item["reserved_bytes"]) for item in samples]
        allocated_values = [int(item["allocated_bytes"]) for item in samples]
        return {
            "n_gpu_snapshots_pending": len(self._gpu),
            "n_clone_samples": len(samples),
            "clone_nbytes_total": int(
                sum(int(item["clone_nbytes"]) for item in samples)
            ),
            "n_reserved_increases_on_clone": int(
                sum(1 for item in samples if item.get("reserved_increase"))
            ),
            "reserved_bytes_min": min(reserved_values) if reserved_values else 0,
            "reserved_bytes_max": max(reserved_values) if reserved_values else 0,
            "allocated_bytes_min": min(allocated_values) if allocated_values else 0,
            "allocated_bytes_max": max(allocated_values) if allocated_values else 0,
        }

    def _ensure_copy_stream(self) -> Any:
        if self._copy_stream is None:
            import torch

            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    def _schedule_d2h(self, tensor: Any, copy_stream: Any, producer_event: Any) -> Any:
        if tensor is None:
            return None
        import torch

        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return tensor
        host = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu"
        )
        copy_stream.wait_event(producer_event)
        with torch.cuda.stream(copy_stream):
            host.copy_(tensor, non_blocking=True)
        return host

    def _snapshot_ready(self, snapshot: PendingGpuSnapshot) -> bool:
        event = snapshot.d2h_event
        if event is None:
            return True
        query = getattr(event, "query", None)
        if query is None:
            return True
        return bool(query())

    def _release_gpu(self, snapshot: PendingGpuSnapshot) -> None:
        snapshot.boxes = None
        snapshot.scores = None
        snapshot.classes = None
        snapshot.ids = None
        snapshot.d2h_event = None

    def _drain_inflight(self, *, flush: bool = False) -> None:
        if not self._gpu:
            return
        pending: list[PendingGpuSnapshot] = []
        for snapshot in self._gpu:
            if flush and snapshot.d2h_event is not None:
                snapshot.d2h_event.synchronize()
            if flush or self._snapshot_ready(snapshot):
                self._release_gpu(snapshot)
                self._host.append(snapshot)
            else:
                pending.append(snapshot)
        if not flush and len(pending) > _MAX_IN_FLIGHT_GPU_SNAPSHOTS:
            overflow = pending[: len(pending) - _MAX_IN_FLIGHT_GPU_SNAPSHOTS]
            keep = pending[len(pending) - _MAX_IN_FLIGHT_GPU_SNAPSHOTS :]
            for snapshot in overflow:
                if snapshot.d2h_event is not None:
                    snapshot.d2h_event.synchronize()
                self._release_gpu(snapshot)
                self._host.append(snapshot)
            pending = keep
        self._gpu = pending

    def observe_detection(
        self,
        sequence: str,
        frame: int,
        stage: str,
        boxes: Any,
        scores: Any,
        classes: Any,
        ids: Any = None,
    ) -> None:
        if stage not in DETECTION_STAGES:
            raise ValueError(f"{stage} is not a detection stage")
        if any(_is_cuda_tensor(item) for item in (boxes, scores, classes, ids)):
            import torch

            copy_stream = self._ensure_copy_stream()
            before = _cuda_allocator_snapshot()
            cloned_boxes = _snapshot_tensor(boxes)
            cloned_scores = _snapshot_tensor(scores)
            cloned_classes = _snapshot_tensor(classes)
            cloned_ids = None if ids is None else _snapshot_tensor(ids)
            producer = torch.cuda.Event()
            producer.record(torch.cuda.current_stream())
            host_boxes = self._schedule_d2h(cloned_boxes, copy_stream, producer)
            host_scores = self._schedule_d2h(cloned_scores, copy_stream, producer)
            host_classes = self._schedule_d2h(cloned_classes, copy_stream, producer)
            host_ids = self._schedule_d2h(cloned_ids, copy_stream, producer)
            done = torch.cuda.Event()
            done.record(copy_stream)
            after = _cuda_allocator_snapshot()
            self._record_clone_sample(
                stage,
                _tensor_nbytes(cloned_boxes)
                + _tensor_nbytes(cloned_scores)
                + _tensor_nbytes(cloned_classes)
                + _tensor_nbytes(cloned_ids),
                before=before,
                after=after,
            )
            self._gpu.append(
                PendingGpuSnapshot(
                    sequence=sequence,
                    frame=int(frame),
                    stage=stage,
                    boxes=cloned_boxes,
                    scores=cloned_scores,
                    classes=cloned_classes,
                    ids=cloned_ids,
                    host_boxes=host_boxes,
                    host_scores=host_scores,
                    host_classes=host_classes,
                    host_ids=host_ids,
                    d2h_event=done,
                )
            )
            self._drain_inflight()
            return
        self._store(
            sequence,
            frame,
            stage,
            fingerprint_detections(
                boxes=boxes, scores=scores, classes=classes, ids=ids
            ),
        )

    def observe_mot_lines(
        self, sequence: str, frame: int, stage: str, lines: Iterable[str]
    ) -> None:
        if stage not in MOT_STAGES:
            raise ValueError(f"{stage} is not a MOT stage")
        self._store(sequence, frame, stage, fingerprint_mot_lines(lines))

    def observe_stage_probe(
        self,
        sequence: str,
        frame: int,
        stage: str,
        boxes: Any,
        scores: Any,
        classes: Any,
    ) -> None:
        if stage not in DETECTION_STAGES:
            return
        self.observe_detection(sequence, frame, stage, boxes, scores, classes)

    def observe_emit(
        self,
        sequence: str,
        frame: int,
        track_results: Mapping[str, Any],
        lines: Sequence[str],
    ) -> None:
        count_raw = track_results.get("count", 0)
        count = int(count_raw.item() if hasattr(count_raw, "item") else count_raw)
        boxes = track_results.get("boxes")
        scores = track_results.get("scores")
        classes = track_results.get("classes")
        ids = track_results.get("ids")
        if count > 0 and boxes is not None and scores is not None:
            boxes = boxes[:count]
            scores = scores[:count]
            ids = None if ids is None else ids[:count]
            if classes is None:
                classes = np.zeros((count,), dtype=np.int32)
            else:
                classes = classes[:count]
            self.observe_detection(
                sequence,
                frame,
                "tracker_output",
                boxes,
                scores,
                classes,
                ids=ids,
            )
        else:
            self.observe_detection(
                sequence,
                frame,
                "tracker_output",
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                ids=np.empty((0,), dtype=np.int32),
            )
        self.observe_mot_lines(sequence, frame, "mot", lines)

    def observe_mot_file(self, sequence: str, lines: Sequence[str]) -> None:
        by_frame: dict[int, list[str]] = {}
        for line in lines:
            text = str(line).strip()
            if not text:
                continue
            frame = int(text.split(",", 1)[0])
            by_frame.setdefault(frame, []).append(text)
        for frame, frame_lines in by_frame.items():
            self.observe_mot_lines(sequence, frame, "mot_file", frame_lines)

    def finalize(self) -> dict[str, Any]:
        measured = self.observer_effect_measured()
        self._drain_inflight(flush=True)
        for snapshot in self._host:
            self._store(
                snapshot.sequence,
                snapshot.frame,
                snapshot.stage,
                fingerprint_detections(
                    boxes=snapshot.host_boxes
                    if snapshot.host_boxes is not None
                    else snapshot.boxes,
                    scores=snapshot.host_scores
                    if snapshot.host_scores is not None
                    else snapshot.scores,
                    classes=snapshot.host_classes
                    if snapshot.host_classes is not None
                    else snapshot.classes,
                    ids=snapshot.host_ids
                    if snapshot.host_ids is not None
                    else snapshot.ids,
                ),
            )
        self._gpu.clear()
        self._host.clear()
        self.records = [dict(item) for item in sort_records(self.records)]
        manifest = write_fingerprint_log(
            self.output_dir,
            self.records,
            include_payloads=self.include_payloads,
            limitations=self.limitations,
        )
        measured_path = self.output_dir / "observer_effect_measured.json"
        measured_path.write_text(
            json.dumps(measured, indent=2) + "\n", encoding="utf-8"
        )
        return manifest


def fingerprint_dir(run_dir: Path) -> Path:
    return Path(run_dir) / FINGERPRINT_DIRNAME


def write_fingerprint_log(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    include_payloads: bool,
    limitations: Sequence[str],
    complete: bool | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ordered = sort_records(records)
    stages_seen = {str(item["stage"]) for item in ordered}
    if complete is None:
        if any(str(item["stage"]) in MOT_STAGES for item in ordered):
            complete = all(stage in stages_seen for stage in STAGES)
        else:
            complete = "detector_output" in stages_seen
    manifest = {
        "schema": SCHEMA_ID,
        "stages": list(STAGES),
        "include_payloads": include_payloads,
        "n_records": len(ordered),
        "complete": complete,
        "hash": "ordered_bit_hash",
        "identity": "first_observable_divergence",
        "not": "causal_mechanism",
        "limitations": list(limitations),
        "observer_effect": {
            "per_frame_cuda_fence": False,
            "gpu_stage_snapshot": "d2d_clone_no_fence_async_d2h_copy_stream",
            "hash_time": "host_after_copy_stream_events",
            "emit_path": "cpu_hash_of_host_track_results_already_copied_for_mot",
        },
    }
    log_path = output / LOG_NAME
    with log_path.open("w", encoding="utf-8") as handle:
        for item in ordered:
            row = dict(item)
            if not include_payloads:
                row.pop("rows", None)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_fingerprint_log(
    run_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    directory = fingerprint_dir(run_dir)
    manifest_path = directory / MANIFEST_NAME
    log_path = directory / LOG_NAME
    if not manifest_path.is_file() or not log_path.is_file():
        return None, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return manifest, records


def index_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for item in records:
        indexed[
            record_key(str(item["sequence"]), int(item["frame"]), str(item["stage"]))
        ] = item
    return indexed


@dataclass(frozen=True)
class FirstStageDivergence:
    kind: str
    sequence: str
    frame: int | None
    stage: str | None
    reference_run: int | None
    other_run: int | None
    reference_hash: str | None
    other_hash: str | None
    ordered_only: bool
    reference_payload: list[Any] | None
    other_payload: list[Any] | None
    last_identical_stage: str | None = None
    first_divergent_stage: str | None = None
    producing_path_verdict: str = VERDICT_NOT_APPLICABLE
    allowed_claim: str = ""
    mechanism_claim: bool = False
    issue_close: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_first_divergence(
    *,
    kind: str,
    sequence: str = "",
    frame: int | None = None,
    stage: str | None = None,
    reference_run: int | None = None,
    other_run: int | None = None,
    reference_hash: str | None = None,
    other_hash: str | None = None,
    ordered_only: bool = False,
    reference_payload: list[Any] | None = None,
    other_payload: list[Any] | None = None,
) -> FirstStageDivergence:
    reading = read_condition_2(kind=kind, stage=stage)
    return FirstStageDivergence(
        kind=kind,
        sequence=sequence,
        frame=frame,
        stage=stage,
        reference_run=reference_run,
        other_run=other_run,
        reference_hash=reference_hash,
        other_hash=other_hash,
        ordered_only=ordered_only,
        reference_payload=reference_payload,
        other_payload=other_payload,
        last_identical_stage=reading.last_identical_stage,
        first_divergent_stage=reading.first_divergent_stage,
        producing_path_verdict=reading.producing_path_verdict,
        allowed_claim=reading.allowed_claim,
        mechanism_claim=reading.mechanism_claim,
        issue_close=reading.issue_close,
    )


@dataclass(frozen=True)
class StageRepeatReport:
    n_runs: int
    ok: bool
    complete: bool
    instrumentation_sufficient: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    first_divergence: FirstStageDivergence | None = None
    n_records: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _mode_run(hashes: Sequence[str | None]) -> int | None:
    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for index, item in enumerate(hashes):
        if not item:
            continue
        counts[item] = counts.get(item, 0) + 1
        first_index.setdefault(item, index)
    if not counts:
        return None
    mode = max(counts, key=lambda key: (counts[key], -first_index[key]))
    return first_index[mode]


def _payload_of(record: Mapping[str, Any] | None) -> list[Any] | None:
    if record is None:
        return None
    rows = record.get("rows")
    if rows is None:
        return None
    return list(rows)


def compare_stage_fingerprints(
    run_dirs: Sequence[Path],
    *,
    mot_diverged: bool | None = None,
) -> StageRepeatReport:
    """Fail-closed identity of per-stage fingerprints across run directories.

    Missing or incomplete logs fail.  The first ``(sequence, frame, stage)``
    whose ``ordered_bit_hash`` differs is the first observable divergence,
    not a causal mechanism.
    """

    resolved = [Path(path) for path in run_dirs]
    if len(resolved) < 2:
        return StageRepeatReport(
            n_runs=len(resolved),
            ok=False,
            complete=False,
            instrumentation_sufficient=True,
            reasons=("need at least two run directories",),
        )

    manifests: list[dict[str, Any] | None] = []
    records_per_run: list[list[dict[str, Any]]] = []
    reasons: list[str] = []
    complete = True
    for index, path in enumerate(resolved):
        manifest, records = load_fingerprint_log(path)
        manifests.append(manifest)
        records_per_run.append(records)
        if manifest is None:
            complete = False
            reasons.append(f"run {index}: missing {FINGERPRINT_DIRNAME}/")
            continue
        if manifest.get("schema") != SCHEMA_ID:
            complete = False
            reasons.append(
                f"run {index}: schema {manifest.get('schema')!r} != {SCHEMA_ID}"
            )
        if not records:
            complete = False
            reasons.append(f"run {index}: empty fingerprint log")
        if not manifest.get("complete", False):
            complete = False
            reasons.append(f"run {index}: fingerprint log marked incomplete")

    if not complete:
        return StageRepeatReport(
            n_runs=len(resolved),
            ok=False,
            complete=False,
            instrumentation_sufficient=True,
            reasons=tuple(reasons),
            n_records=tuple(len(item) for item in records_per_run),
            first_divergence=make_first_divergence(kind=KIND_INCOMPLETE),
        )

    indexes = [index_records(records) for records in records_per_run]
    keys = sorted(
        {key for index in indexes for key in index},
        key=lambda item: (
            item[0],
            item[1],
            STAGE_INDEX.get(item[2], len(STAGES)),
            item[2],
        ),
    )
    if not keys:
        return StageRepeatReport(
            n_runs=len(resolved),
            ok=False,
            complete=False,
            instrumentation_sufficient=True,
            reasons=("no fingerprint records",),
            n_records=tuple(len(item) for item in records_per_run),
        )

    first: FirstStageDivergence | None = None
    for sequence, frame, stage in keys:
        hashes = [
            None if key not in index else str(index[key]["ordered_bit_hash"])
            for index, key in ((item, (sequence, frame, stage)) for item in indexes)
        ]
        if any(item is None for item in hashes) or len({item for item in hashes}) > 1:
            reference_run = _mode_run(hashes)
            if reference_run is None:
                other_run = next(
                    index for index, item in enumerate(hashes) if item is None
                )
                ref_record = None
                other_record = None
                ref_hash = None
                other_hash = None
            else:
                ref_hash = hashes[reference_run]
                other_run = next(
                    index for index, item in enumerate(hashes) if item != ref_hash
                )
                ref_record = indexes[reference_run].get((sequence, frame, stage))
                other_record = indexes[other_run].get((sequence, frame, stage))
                other_hash = hashes[other_run]
            ordered_only = False
            if ref_record is not None and other_record is not None:
                ordered_only = (
                    ref_record.get("multiset_canonical_hash")
                    == other_record.get("multiset_canonical_hash")
                    and ref_hash != other_hash
                )
            first = make_first_divergence(
                kind=KIND_FIRST_OBSERVABLE,
                sequence=sequence,
                frame=frame,
                stage=stage,
                reference_run=reference_run,
                other_run=other_run,
                reference_hash=ref_hash,
                other_hash=other_hash,
                ordered_only=ordered_only,
                reference_payload=_payload_of(ref_record),
                other_payload=_payload_of(other_record),
            )
            reasons.append(
                f"{sequence} frame {frame} stage {stage}: first observable divergence"
            )
            break

    if first is None and mot_diverged:
        first = make_first_divergence(kind=KIND_INSUFFICIENT)
        reasons.append(
            "MOT files diverged but every instrumented stage hash matched; "
            "instrumentation is not sufficient to narrow the bound further"
        )
        return StageRepeatReport(
            n_runs=len(resolved),
            ok=False,
            complete=True,
            instrumentation_sufficient=False,
            reasons=tuple(reasons),
            first_divergence=first,
            n_records=tuple(len(item) for item in records_per_run),
        )

    ok = first is None
    return StageRepeatReport(
        n_runs=len(resolved),
        ok=ok,
        complete=True,
        instrumentation_sufficient=True,
        reasons=tuple(reasons),
        first_divergence=first,
        n_records=tuple(len(item) for item in records_per_run),
    )


def write_first_divergence(path: Path, report: StageRepeatReport) -> None:
    """Persist only the first mismatched payload, never a full tensor dump."""

    if report.first_divergence is None:
        return
    if report.first_divergence.kind != KIND_FIRST_OBSERVABLE:
        path.write_text(
            json.dumps(report.first_divergence.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return
    path.write_text(
        json.dumps(report.first_divergence.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def format_stage_report(report: StageRepeatReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"stage-fingerprint: runs={report.n_runs} {status} "
        f"complete={report.complete} sufficient={report.instrumentation_sufficient}"
    ]
    first = report.first_divergence
    if first is not None:
        lines.append(
            f"  first_observable: kind={first.kind} sequence={first.sequence or '-'} "
            f"frame={first.frame} "
            f"last_identical_stage={first.last_identical_stage or '-'} "
            f"first_divergent_stage={first.first_divergent_stage or '-'} "
            f"runs={first.reference_run}->{first.other_run} "
            f"ordered_only={first.ordered_only}"
        )
        lines.append(
            f"  producing_path_verdict={first.producing_path_verdict} "
            f"mechanism_claim={first.mechanism_claim} issue_close={first.issue_close}"
        )
        if first.allowed_claim:
            lines.append(f"  allowed_claim: {first.allowed_claim}")
        if first.reference_hash is not None:
            lines.append(f"    ref  {first.reference_hash}")
        if first.other_hash is not None:
            lines.append(f"    other {first.other_hash}")
    for reason in report.reasons:
        lines.append(f"  reason: {reason}")
    return "\n".join(lines)


def _track_count(track_results: Mapping[str, Any] | None) -> int:
    if not track_results:
        return 0
    raw = track_results.get("count", 0)
    if hasattr(raw, "item"):
        return int(raw.item())
    return int(raw or 0)


def install_eval_hooks(collector: StageFingerprintCollector) -> Any:
    """Inject collection into the already-imported eval stack.  Returns undo()."""

    import saccade.perception.eval.evaluator as evaluator_module
    import saccade.perception.eval.helpers as helpers_module
    import saccade.perception.eval.runner as runner
    from saccade.perception.eval import stages as stages_module

    original_run_eval = runner.run_eval
    original_helpers_emit = helpers_module.fast_emit_mot_lines
    original_evaluator_emit = evaluator_module._fast_emit_mot_lines
    original_stages_emit = stages_module._fast_emit_mot_lines
    original_run_emit = stages_module._run_emit
    original_evaluator_run_emit = evaluator_module._run_emit
    original_read_deferred = evaluator_module._read_deferred_result
    original_flush = evaluator_module._flush_deferred_emit
    deferred_stash: dict[str, Any] = {}

    def _observing_emit(original: Any) -> Any:
        def observing_emit(**kwargs: Any) -> list[str]:
            lines = original(**kwargs)
            if lines or _track_count(kwargs.get("track_results")) > 0:
                collector.observe_emit(
                    str(kwargs["seq"]),
                    int(kwargs["frame_id"]),
                    kwargs["track_results"],
                    lines,
                )
            return lines

        return observing_emit

    def wrapped_run_emit(state: Any, **kwargs: Any) -> Any:
        prev, lines = original_run_emit(state, **kwargs)
        track_results = kwargs.get("track_results")
        if lines or _track_count(track_results) > 0:
            collector.observe_emit(
                str(state.seq),
                int(kwargs["frame_id"]),
                track_results or {},
                lines,
            )
        return prev, lines

    def wrapped_read_deferred(*args: Any, **kwargs: Any) -> Any:
        track_results = original_read_deferred(*args, **kwargs)
        deferred_stash["track_results"] = track_results
        return track_results

    def wrapped_flush(*args: Any, **kwargs: Any) -> Any:
        lines, track_ids = original_flush(*args, **kwargs)
        track_results = deferred_stash.pop("track_results", None)
        if track_results is not None and (lines or _track_count(track_results) > 0):
            collector.observe_emit(
                str(kwargs["seq"]),
                int(kwargs["frame_id"]),
                track_results,
                lines,
            )
        return lines, track_ids

    def wrapped_run_eval(**kwargs: Any) -> Any:
        if int(kwargs.get("cpp_threads", 0) or 0) > 0:
            raise RuntimeError(
                "stage fingerprints require the Python evaluator; cpp_threads is set"
            )
        prior_probe = kwargs.get("stage_probe_callback")
        prior_sequence = kwargs.get("sequence_result_callback")

        def stage_callback(
            sequence: str,
            frame_id: int,
            stage: str,
            boxes: Any,
            scores: Any,
            classes: Any,
        ) -> None:
            if prior_probe is not None:
                prior_probe(sequence, frame_id, stage, boxes, scores, classes)
            collector.observe_stage_probe(
                sequence, frame_id, stage, boxes, scores, classes
            )

        def sequence_callback(sequence: str, lines: Sequence[str]) -> None:
            collector.observe_mot_file(sequence, lines)
            if prior_sequence is not None:
                prior_sequence(sequence, lines)

        kwargs["stage_probe_callback"] = stage_callback
        kwargs["sequence_result_callback"] = sequence_callback
        return original_run_eval(**kwargs)

    observing_helpers = _observing_emit(original_helpers_emit)
    runner.run_eval = wrapped_run_eval
    helpers_module.fast_emit_mot_lines = observing_helpers
    evaluator_module._fast_emit_mot_lines = _observing_emit(original_evaluator_emit)
    stages_module._fast_emit_mot_lines = _observing_emit(original_stages_emit)
    stages_module._run_emit = wrapped_run_emit
    evaluator_module._run_emit = wrapped_run_emit
    evaluator_module._read_deferred_result = wrapped_read_deferred
    evaluator_module._flush_deferred_emit = wrapped_flush

    def undo() -> None:
        runner.run_eval = original_run_eval
        helpers_module.fast_emit_mot_lines = original_helpers_emit
        evaluator_module._fast_emit_mot_lines = original_evaluator_emit
        stages_module._fast_emit_mot_lines = original_stages_emit
        stages_module._run_emit = original_run_emit
        evaluator_module._run_emit = original_evaluator_run_emit
        evaluator_module._read_deferred_result = original_read_deferred
        evaluator_module._flush_deferred_emit = original_flush

    return undo
