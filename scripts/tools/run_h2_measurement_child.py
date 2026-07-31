#!/usr/bin/env python3
"""H2 Phase-A measurement child and recorder.

The parent is the only supported caller.  It writes one canonical invocation,
launches this file with exactly ``--invocation <absolute path>``, and owns the
four-run ordering and the single deadline.  The child owns one fresh evaluator
process and the run-local artifacts:

* the A7.6 policy inventory;
* the complete capture packet and frozen packet-verifier report on capture-on;
* the MOT bytes used by the inventory identity; and
* the invocation state transition.

This file is ``plumbing_only``.  It imports the A7.6 member sets, run ids,
capture schema, trace capacities, and packet verifier from their existing
authorities.  It does not define a terminal, phase completion rule, or new
comparison vocabulary.
"""
# status: stable

from __future__ import annotations

import argparse
import configparser
import hashlib
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_run_spec as run_spec  # noqa: E402
import run_h0_phase_a as h0_controller  # noqa: E402
from export_headline_bridge_decision_trace import (  # noqa: E402
    OVERFLOW_KEYS,
    STREAMS,
    UNIVERSE_OVERFLOW_KEYS,
    UNIVERSE_STREAMS,
    canonical_semantic_packet,
)
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

CHILD_SCHEMA = "h2_measurement_child_v1"
INVOCATION_SCHEMA = "h2_measurement_child_invocation_v1"
ENVIRONMENT_DELTA_SCHEMA = "h2_child_environment_import_delta_v1"
ENVIRONMENT_DELTA_NAME = "environment_import_delta.json"

# One run's durable lifecycle record, and the only vocabulary for its states.
# This file performs both transitions out of `running`, so the names live with
# the transitions; the controller and any reader of an archive must import them
# rather than restate them (§ C3.9). "The run directory exists" is not one of
# these states and never answers whether a run finished.
INVOCATION_NAME = "invocation.json"
RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"

# Launch hygiene remains process policy, not evaluator configuration.  The four
# SACCADE configuration keys formerly mixed into H0's STATIC_ENV are now derived
# exclusively from the RunSpec by `h2_run_spec.environment_projection`.
HYGIENE_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}
DYNAMIC_ENV_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LD_LIBRARY_PATH",
        "PATH",
        "SACCADE_BUILD_PATH",
        "TMPDIR",
        "XDG_CACHE_HOME",
    }
)
EXPECTED_ENV_KEYS = frozenset(
    {*HYGIENE_ENV, *run_spec.CONFIG_ENV_KEYS, *DYNAMIC_ENV_KEYS}
)
REPOSITORY_OWNED_ENV_KEYS = run_spec.REPOSITORY_OWNED_ENV_KEYS

(
    ACTIVE_PAIRS_MEMBER,
    FINAL_ROWS_MEMBER,
    MOT_MEMBER,
    RELINK_MEMBER,
) = behavior.BEHAVIOR_MEMBERS
PROPOSAL_MEMBER, WINNER_MEMBER = behavior.A76_PROJECTION_MEMBERS
OVERFLOW_MEMBER = behavior.A76_OVERFLOW_MEMBER
POLICY_MEMBERS = frozenset(
    {
        *behavior.A76_EQUALITY_MEMBERS,
        *behavior.A76_PROJECTION_MEMBERS,
        behavior.A76_OVERFLOW_MEMBER,
        "schema",
    }
)
BASE_POLICY_MEMBERS = frozenset({*behavior.A76_EQUALITY_MEMBERS, "schema"})
OVERFLOW_FIELDS = tuple(OVERFLOW_KEYS[name] for name in STREAMS) + tuple(
    UNIVERSE_OVERFLOW_KEYS[name] for name in UNIVERSE_STREAMS
)


class ChildError(RuntimeError):
    """The child invocation or one produced artifact is invalid."""


@dataclass(frozen=True)
class RunProducts:
    mot_bytes: bytes
    policy_base_inventory: Mapping[str, Any]
    policy_inventory: Mapping[str, Any] | None
    packet: Mapping[str, Any] | None
    packet_verification: Mapping[str, Any] | None


@dataclass(frozen=True)
class CaptureProcessing:
    proposal: Mapping[str, Any] | None
    winner: Mapping[str, Any] | None
    overflow: list[int] | None
    verification: Mapping[str, Any]

    @property
    def valid(self) -> bool:
        return self.verification.get("state") == "pass"


def normalize_active_pairs(
    raw: Sequence[Sequence[int]], *, frame_id: int
) -> list[list[int]]:
    """Apply §5.1.1's recorder normalization and reject duplicate slots."""
    pairs = [[int(pair[0]), int(pair[1])] for pair in raw]
    pairs.sort(key=lambda pair: pair[1])
    slots = [pair[1] for pair in pairs]
    if len(set(slots)) != len(slots):
        raise ChildError(
            f"duplicate slot in active tid/slot pairs at frame {frame_id}: {pairs}"
        )
    return pairs


def _canonical_document(path: Path, *, schema: str) -> dict[str, Any]:
    try:
        payload = evidence.load_document(path.parent, path.name, schema=schema)
    except evidence.EvidenceError as exc:
        raise ChildError(str(exc)) from exc
    return payload


def load_invocation(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise ChildError("invocation path is not absolute")
    if path.is_symlink() or not path.is_file():
        raise ChildError("invocation is not a physical regular file")
    invocation = _canonical_document(path, schema=INVOCATION_SCHEMA)
    required = {
        "build_dir",
        "capture_phase",
        "capture_run_uuid",
        "environment_digest",
        "instrumentation_head",
        "run_spec",
        "run_dir",
        "run_id",
        "schema",
        "sequence",
        "state",
    }
    if set(invocation) != required:
        raise ChildError(
            "invocation has missing or unknown members: "
            f"{sorted(set(invocation) ^ required)}"
        )
    if invocation["run_id"] not in evidence.RUN_IDS:
        raise ChildError(f"unknown run id: {invocation['run_id']!r}")
    phase = evidence.PHASE_BY_CAPTURE_PHASE.get(invocation["capture_phase"])
    if phase != "a":
        raise ChildError("this child implements the scoped Phase-A controller only")
    if invocation["sequence"] not in evidence.expected_sequences(phase):
        raise ChildError("invocation sequence is outside the Phase-A fixture")
    run_dir = Path(str(invocation["run_dir"]))
    build_dir = Path(str(invocation["build_dir"]))
    if (
        not run_dir.is_absolute()
        or run_dir.resolve(strict=True) != path.parent.resolve(strict=True)
        or not build_dir.is_absolute()
        or not build_dir.is_dir()
    ):
        raise ChildError("invocation path binding is absent or inconsistent")
    if invocation["state"] != RUN_RUNNING:
        raise ChildError("invocation is not in the running state")
    try:
        run_spec.validate_run_spec(invocation["run_spec"], verify_projection=True)
    except (KeyError, TypeError, run_spec.RunSpecError) as exc:
        raise ChildError(f"invocation RunSpec is invalid: {exc}") from exc
    run_uuid = invocation["capture_run_uuid"]
    if not isinstance(run_uuid, str) or not run_uuid:
        raise ChildError("invocation has no capture_run_uuid")
    return invocation


def _environment_digest(environment: Mapping[str, str]) -> str:
    return hashlib.sha256(evidence.canonical_json_bytes(dict(environment))).hexdigest()


def validate_environment(
    environment: Mapping[str, str], invocation: Mapping[str, Any]
) -> None:
    if set(environment) != EXPECTED_ENV_KEYS:
        raise ChildError("child environment keys differ from the H2 launch contract")
    for key, value in HYGIENE_ENV.items():
        if environment.get(key) != value:
            raise ChildError(f"child environment value mismatch for {key}")
    try:
        projected = run_spec.environment_projection(invocation["run_spec"])
    except run_spec.RunSpecError as exc:
        raise ChildError(f"child RunSpec environment projection failed: {exc}") from exc
    for key, value in projected.items():
        if environment.get(key) != value:
            raise ChildError(f"child RunSpec environment mismatch for {key}")
    if environment.get("SACCADE_BUILD_PATH") != invocation["build_dir"]:
        raise ChildError("selected build directory differs from the invocation")
    if environment.get("SACCADE_GPU_DECODE") != "1":
        raise ChildError("the measurement child must keep GPU decode enabled")
    if _environment_digest(environment) != invocation["environment_digest"]:
        raise ChildError("child environment digest differs from the invocation")
    run_dir = Path(str(invocation["run_dir"]))
    for key, leaf in (
        ("HOME", "home"),
        ("TMPDIR", "tmp"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        expected = run_dir / "_env" / leaf
        if environment.get(key) != expected.as_posix() or not expected.is_dir():
            raise ChildError(f"derived environment directory mismatch for {key}")


def record_import_delta(
    run_dir: Path, before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, Any]:
    """Record the eval stack import's environment side effect. Never a gate.

    Declaration Review Correction 4: the authorization environment is the
    immutable launch snapshot `execute_child` validated before any third-party
    import.  What an import does to the live environment afterwards is derived
    state.  It is recorded here so the side effect is not invisible, and it is
    recorded as key names only — this document is diagnostic, so it has no
    reason to carry a stable fingerprint of any environment value.
    """
    before_keys, after_keys = set(before), set(after)
    document = {
        "schema": ENVIRONMENT_DELTA_SCHEMA,
        "authority": "diagnostic_only",
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": sorted(
            key for key in before_keys & after_keys if before[key] != after[key]
        ),
    }
    evidence.write_document(run_dir, ENVIRONMENT_DELTA_NAME, document)
    return document


def validate_repository_owned_mutation(
    baseline: Mapping[str, str],
    applied: Mapping[str, str],
    document: Mapping[str, Any],
) -> None:
    """Gate `configure_runtime_env` against its own explicitly named baseline.

    Separate subject matter from the ingress authorization predicate, and it
    never re-derives it: the baseline is taken *after* the eval stack import
    and immediately before the repository's own runtime configuration runs, so
    the import's side effect lies on neither side of this comparison.  Taking
    the launch snapshot as this baseline would charge the import's delta to the
    repository and reproduce the 2026-07-28 failure under a new name.
    """
    mutated = {
        key
        for key in set(baseline) | set(applied)
        if baseline.get(key) != applied.get(key)
    }
    outside = sorted(mutated - run_spec.REPOSITORY_OWNED_ENV_KEYS)
    if outside:
        raise ChildError(
            "repository-owned runtime configuration mutated keys outside its "
            f"declared set: {outside}"
        )
    try:
        projected = run_spec.environment_projection(document)
    except run_spec.RunSpecError as exc:
        raise ChildError(
            f"repository-owned runtime configuration has invalid RunSpec: {exc}"
        ) from exc
    mismatches = sorted(
        key for key, value in projected.items() if applied.get(key) != value
    )
    if mismatches:
        raise ChildError(
            "repository-owned runtime configuration violated RunSpec values: "
            f"{mismatches}"
        )
    if "SACCADE_STREAM_MODE" in applied:
        raise ChildError(
            "repository-owned runtime configuration left SACCADE_STREAM_MODE set"
        )


def _projection_records(
    capture: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = canonical_semantic_packet(capture)
    streams = packet["streams"]
    candidates = [
        row for row in streams["candidate_records"] if int(row["proposal_emitted"]) == 1
    ]
    claims = streams["claim_records"]
    commits = streams["commit_records"]
    proposal_payload = {"candidates": candidates, "claims": claims}
    winner_payload = {
        "commits": commits,
        "winning_claims": [row for row in claims if int(row["claim_won"]) == 1],
    }
    return (
        {
            "count": len(candidates),
            "digest": evidence.digest(proposal_payload),
            "records": proposal_payload,
        },
        {
            "count": len(commits),
            "digest": evidence.digest(winner_payload),
            "records": winner_payload,
        },
    )


def persist_and_process_capture(
    run_dir: Path, capture: Mapping[str, Any]
) -> CaptureProcessing:
    """Persist the raw capture before one total packet-verification operation."""
    evidence.write_document(run_dir, evidence.PACKET_NAME, capture)
    try:
        proposal, winner = _projection_records(capture)
        overflow = [int(capture[field]) for field in OVERFLOW_FIELDS]
        report = verify_capture(capture)
    except behavior.PACKET_INVALID_EXCEPTIONS:
        verification: Mapping[str, Any] = {
            "failure": "packet_invalid",
            "state": "fail",
        }
        evidence.write_document(
            run_dir, evidence.PACKET_VERIFICATION_NAME, verification
        )
        return CaptureProcessing(None, None, None, verification)
    verification = {"report": report, "state": "pass"}
    evidence.write_document(run_dir, evidence.PACKET_VERIFICATION_NAME, verification)
    return CaptureProcessing(proposal, winner, overflow, verification)


def record_raw_emission(
    rows: list[dict[str, Any]],
    *,
    lines: Sequence[str],
    track_results: Mapping[str, Any],
    frame: int,
    person_class: int,
) -> None:
    """Record one emission's binary32 evidence, or refuse the emission.

    This is the *local* observation surface: at the moment `_fast_emit_mot_lines`
    returns, the boxes and scores still in `track_results` are the ones that
    produced exactly these lines, so the two must agree here — same cardinality,
    same frame, same track id per row. That agreement is what makes the recorded
    bits evidence rather than a parallel guess.

    It says nothing about the sequence-level MOT output. The evaluator applies
    deferred alias remapping, the low-quality tracklet filter and interpolation
    after the last emission, so the rows below and the rows the sequence callback
    finally delivers are two observations taken at two times. A7.6 records them as
    two members — `final_track_rows` and `mot_output` — each compared capture-off
    to capture-on on its own, and no contract asks for a row projection between
    them.
    """
    count = int(track_results["count"])
    if len(lines) != count:
        raise ChildError("raw emission cardinality or sequence mismatch")
    boxes = track_results["boxes"][:count].numpy()
    scores = track_results["scores"][:count].numpy()
    classes_value = track_results.get("classes")
    classes = (
        [person_class] * count
        if classes_value is None
        else [int(value) for value in classes_value[:count].numpy()]
    )
    row_base = sum(1 for row in rows if row["frame"] == frame)
    for offset, (line, box, score, class_id) in enumerate(
        zip(lines, boxes, scores, classes, strict=True)
    ):
        fields = line.split(",")
        if len(fields) != 10 or int(fields[0]) != frame:
            raise ChildError("raw emission and MOT row disagree")
        x1, y1, x2, y2 = (float(value) for value in box)
        rows.append(
            {
                "binary32_bits": [
                    _binary32_bits(value)
                    for value in (x1, y1, x2 - x1, y2 - y1, float(score))
                ],
                "class": class_id,
                "frame": frame,
                "row_index": row_base + offset,
                "track_id": int(fields[1]),
            }
        )


def canonical_callback_bytes(lines: Sequence[str]) -> bytes:
    """The sequence callback's own evidence: its bytes, and its rows' shape.

    Every row must be a canonical ten-field MOT row whose frame and track id
    parse. What is deliberately *not* checked is any correspondence with the raw
    emission rows: their cardinality, their ids and their order may all differ,
    because a legal sequence-level transformation may add rows, remove rows or
    rename ids after the last emission was recorded.
    """
    for line in lines:
        fields = line.split(",")
        if len(fields) != 10:
            raise ChildError("non-canonical MOT result row")
        try:
            int(fields[0])
            int(fields[1])
        except ValueError as exc:
            raise ChildError("non-canonical MOT result row") from exc
    return ("\n".join(lines) + "\n").encode("utf-8")


def _binary32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


def repository_runner(invocation: Mapping[str, Any]) -> RunProducts:
    """Run the fixed A5 policy once and return the complete run-local products.

    The ingress authorization decision belongs to `execute_child`, which made it
    once against the immutable launch snapshot before this file imported
    anything third-party.  Nothing here re-derives it: the two live observations
    below are compared only against each other, and their contracts are named
    separately.
    """
    run_dir = Path(str(invocation["run_dir"]))
    document = invocation["run_spec"]
    pre_import = dict(os.environ)
    (
        _yaml,
        build_parser,
        configure_runtime_env,
        _fingerprint,
        evaluator_module,
        stages_module,
        EvalPipeline,
        build_mamba_gated_detector,
        set_postprocess_compile,
    ) = behavior._import_eval_stack()
    record_import_delta(run_dir, pre_import, dict(os.environ))

    run_id = str(invocation["run_id"])
    sequence = str(invocation["sequence"])
    build_dir = Path(str(invocation["build_dir"]))
    if behavior.resolve_build_dir() != build_dir:
        raise ChildError("runtime build selection differs from the invocation")
    extension = behavior._assert_extension_consumed(build_dir)

    try:
        args = run_spec.parse_runtime_namespace(
            document, run_dir, parser=build_parser()
        )
    except run_spec.RunSpecError as exc:
        raise ChildError(f"cannot project runtime parser namespace: {exc}") from exc
    configuration_baseline = dict(os.environ)
    configure_runtime_env(args, os.environ)
    validate_repository_owned_mutation(
        configuration_baseline, dict(os.environ), document
    )
    try:
        run_spec.assert_runtime_matches(document, args, os.environ, run_dir)
    except run_spec.RunSpecError as exc:
        raise ChildError(f"pre-execution RunSpec mismatch: {exc}") from exc

    tiling = getattr(args, "tiling", "native_640")
    detector = build_mamba_gated_detector(
        yolo_pt_path=args.mamba_yolo_weights,
        teacher_ckpt=args.mamba_teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=behavior._image_size(tiling),
        device="cuda",
        conf_thr=0.001,
        max_det=getattr(args, "max_det", 300),
        trt_backbone_engine=getattr(args, "fpn_backbone_engine", ""),
        trt_head_engine=getattr(args, "mamba_head_engine", ""),
        temporal_T_override=0 if getattr(args, "no_temporal", False) else None,
        use_cuda_graph=bool(getattr(args, "use_cuda_graph", False)),
        use_whole_graph=bool(getattr(args, "use_whole_graph", False)),
        small_p3_max_threshold=getattr(args, "mamba_small_p3_max_threshold", 0.0),
    )
    set_postprocess_compile(True)
    detector.mamba_head.set_head_compile(True)
    detector.mamba_head.set_block_compile(True)

    active_pairs: list[dict[str, Any]] = []
    # Named for when it is taken, not for what the sequence finally emits: these
    # rows are the per-emission observation, and the MOT output the callback
    # delivers is a later one. The archive member keeps its frozen name.
    raw_emission_rows: list[dict[str, Any]] = []
    pipelines: list[Any] = []
    emitted: list[tuple[str, tuple[str, ...]]] = []
    original_init = EvalPipeline.__init__
    original_frame = evaluator_module._run_frame
    original_evaluator_emit = evaluator_module._fast_emit_mot_lines
    original_stages_emit = stages_module._fast_emit_mot_lines

    def controlled_init(self: Any, *positional: Any, **keywords: Any) -> None:
        original_init(self, *positional, **keywords)
        if self.seq != sequence or pipelines:
            raise ChildError("evaluator constructed an unexpected or second sequence")
        pipelines.append(self)
        tracker = self.detector.tracker
        enabled = run_id != evidence.CAPTURE_OFF_RUN
        pair, candidate, claim, commit = h0_controller.TRACE_CAPACITIES
        tracker.set_research_h0_bridge_trace(
            enabled,
            pair_capacity=pair,
            candidate_capacity=candidate,
            claim_capacity=claim,
            commit_capacity=commit,
        )
        if enabled:
            tracker.clear_research_h0_bridge_trace()

    def controlled_frame(
        state: Any, *, frame_id: int, prepared_detection: Any = None
    ) -> bool:
        outcome = original_frame(
            state, frame_id=frame_id, prepared_detection=prepared_detection
        )
        raw = state.detector.tracker.tracker.get_active_tid_slot_pairs()
        active_pairs.append(
            {
                "frame": int(frame_id),
                "pairs": normalize_active_pairs(raw, frame_id=frame_id),
            }
        )
        return outcome

    def capturing_emit(
        original: Callable[..., list[str]], **keywords: Any
    ) -> list[str]:
        lines = original(**keywords)
        if keywords["seq"] != sequence:
            raise ChildError("raw emission cardinality or sequence mismatch")
        record_raw_emission(
            raw_emission_rows,
            lines=lines,
            track_results=keywords["track_results"],
            frame=int(keywords["frame_id"]),
            person_class=int(getattr(args, "person_class", 0)),
        )
        return lines

    def callback(name: str, lines: tuple[str, ...]) -> None:
        if emitted or name != sequence or not isinstance(lines, tuple):
            raise ChildError("sequence callback cardinality or type mismatch")
        emitted.append((name, lines))

    EvalPipeline.__init__ = controlled_init
    evaluator_module.EvalPipeline = EvalPipeline
    evaluator_module._run_frame = controlled_frame
    evaluator_module._fast_emit_mot_lines = lambda **kw: capturing_emit(
        original_evaluator_emit, **kw
    )
    stages_module._fast_emit_mot_lines = lambda **kw: capturing_emit(
        original_stages_emit, **kw
    )

    excluded = {
        "module_detection",
        "module_geometry",
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
        "mamba_ckpt",
        "mamba_teacher_ckpt",
        "mamba_yolo_weights",
        "teacher_head_ckpt",
        "visualize",
        "visualize_scale",
        "visualize_fps",
        "no_visualize_score",
        "visualize_trail_len",
    }
    eval_kwargs = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    eval_kwargs.update(
        {
            "detector": detector,
            "engine": "mamba",
            "sequence_result_callback": callback,
            "tiling": tiling,
        }
    )
    try:
        result = evaluator_module.run_eval(**eval_kwargs)
    finally:
        EvalPipeline.__init__ = original_init
        evaluator_module.EvalPipeline = EvalPipeline
        evaluator_module._run_frame = original_frame
        evaluator_module._fast_emit_mot_lines = original_evaluator_emit
        stages_module._fast_emit_mot_lines = original_stages_emit
        try:
            run_spec.assert_runtime_matches(document, args, os.environ, run_dir)
        except run_spec.RunSpecError as exc:
            raise ChildError(f"post-execution RunSpec mismatch: {exc}") from exc
    if result != {} or len(emitted) != 1 or len(pipelines) != 1:
        raise ChildError("evaluator did not return at the sole no-metrics boundary")

    sequence_info = configparser.ConfigParser(interpolation=None)
    sequence_info.optionxform = str
    try:
        with (
            REPO_ROOT / "datasets" / "MOT17" / "train" / sequence / "seqinfo.ini"
        ).open("r", encoding="utf-8") as handle:
            sequence_info.read_file(handle)
        frame_count = int(sequence_info["Sequence"]["seqLength"])
    except (OSError, KeyError, ValueError, configparser.Error) as exc:
        raise ChildError("canonical sequence frame count is unavailable") from exc
    if [row["frame"] for row in active_pairs] != list(range(1, frame_count + 1)):
        raise ChildError("child did not process the complete sequence exactly once")

    # Two evidence sets, each complete on its own terms. The equality that used to
    # stand here — the raw emission keys against the callback's rows — was never a
    # contract requirement: A7.6 names `final_track_rows` and `mot_output` as two
    # members and asks each to be identical between the capture-off and capture-on
    # runs, not to be projections of one another. It was also unsatisfiable under
    # this RunSpec, because deferred alias remapping, the low-quality tracklet
    # filter and interpolation all run after the last emission and may rename,
    # remove or insert rows. Requiring it made a truthful execution unrecordable.
    lines = emitted[0][1]
    mot_bytes = canonical_callback_bytes(lines)

    tracker = pipelines[0].detector.tracker
    relink = [int(value) for value in tracker.get_relink_debug()]
    if len(relink) != 13:
        raise ChildError("native relink debug vector length drift")

    base_inventory = {
        ACTIVE_PAIRS_MEMBER: active_pairs,
        FINAL_ROWS_MEMBER: raw_emission_rows,
        MOT_MEMBER: {
            "length": len(mot_bytes),
            "sha256": hashlib.sha256(mot_bytes).hexdigest(),
        },
        RELINK_MEMBER: relink,
        "schema": evidence.BASE_POLICY_INVENTORY_SCHEMA,
    }
    _persist_base_products(run_dir, sequence, mot_bytes, base_inventory)

    packet: Mapping[str, Any] | None = None
    packet_verification: Mapping[str, Any] | None = None
    proposal: Mapping[str, Any] | None = None
    winner: Mapping[str, Any] | None = None
    overflow = list(behavior.A76_OVERFLOW_ZERO_VECTOR)
    if run_id != evidence.CAPTURE_OFF_RUN:
        capture = tracker.drain_research_h0_bridge_trace(
            seq=sequence,
            capture_phase=invocation["capture_phase"],
            require_candidate_exposure=True,
            require_commit_exposure=False,
            capture_run_uuid=invocation["capture_run_uuid"],
        )
        packet = capture
        processed = persist_and_process_capture(run_dir, capture)
        packet_verification = processed.verification
        if processed.valid:
            proposal = processed.proposal
            winner = processed.winner
            assert processed.overflow is not None
            overflow = processed.overflow
        else:
            # A structurally invalid raw packet is decisive terminal-3
            # evidence. Do not fabricate packet-derived policy projections.
            behavior._assert_build_components_consumed(build_dir, extension)
            return RunProducts(
                mot_bytes,
                base_inventory,
                None,
                packet,
                packet_verification,
            )

    inventory = {
        ACTIVE_PAIRS_MEMBER: active_pairs,
        FINAL_ROWS_MEMBER: raw_emission_rows,
        MOT_MEMBER: {
            "length": len(mot_bytes),
            "sha256": hashlib.sha256(mot_bytes).hexdigest(),
        },
        OVERFLOW_MEMBER: overflow,
        PROPOSAL_MEMBER: proposal,
        RELINK_MEMBER: relink,
        "schema": evidence.POLICY_INVENTORY_SCHEMA,
        WINNER_MEMBER: winner,
    }
    behavior._assert_build_components_consumed(build_dir, extension)
    return RunProducts(
        mot_bytes,
        base_inventory,
        inventory,
        packet,
        packet_verification,
    )


def validate_products(run_id: str, products: RunProducts) -> None:
    base = products.policy_base_inventory
    if (
        set(base) != BASE_POLICY_MEMBERS
        or base.get("schema") != evidence.BASE_POLICY_INVENTORY_SCHEMA
    ):
        raise ChildError("base policy inventory schema mismatch")
    if base[MOT_MEMBER] != {
        "length": len(products.mot_bytes),
        "sha256": hashlib.sha256(products.mot_bytes).hexdigest(),
    }:
        raise ChildError("MOT bytes differ from the base policy inventory")
    inventory = products.policy_inventory
    if inventory is None:
        if (
            run_id == evidence.CAPTURE_OFF_RUN
            or products.packet is None
            or products.packet_verification
            != {"failure": "packet_invalid", "state": "fail"}
        ):
            raise ChildError("only a persisted invalid capture may omit its inventory")
        return
    if set(inventory) != POLICY_MEMBERS:
        raise ChildError("policy inventory has missing or unknown members")
    if inventory.get("schema") != evidence.POLICY_INVENTORY_SCHEMA:
        raise ChildError("policy inventory schema mismatch")
    if any(
        inventory[member] != base[member] for member in behavior.A76_EQUALITY_MEMBERS
    ):
        raise ChildError("full policy inventory differs from its base members")
    if inventory[MOT_MEMBER] != {
        "length": len(products.mot_bytes),
        "sha256": hashlib.sha256(products.mot_bytes).hexdigest(),
    }:
        raise ChildError("MOT bytes differ from the policy inventory")
    if run_id == evidence.CAPTURE_OFF_RUN:
        if products.packet is not None or products.packet_verification is not None:
            raise ChildError("capture-off produced packet artifacts")
        if (
            inventory[PROPOSAL_MEMBER] is not None
            or inventory[WINNER_MEMBER] is not None
        ):
            raise ChildError("capture-off fabricated trace-only projections")
    elif products.packet is None or products.packet_verification is None:
        raise ChildError("capture-on omitted packet artifacts")


def _replace_invocation(path: Path, payload: Mapping[str, Any]) -> None:
    evidence.write_document(path.parent, path.name, payload)


def _write_mot(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short MOT artifact write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        evidence._fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _persist_base_products(
    run_dir: Path,
    sequence: str,
    mot_bytes: bytes,
    inventory: Mapping[str, Any],
) -> None:
    """Commit raw MOT first, then atomically commit all four base members.

    A durable MOT final path is the independent commit point for `mot_output`.
    The base-inventory replace plus directory fsync is the commit point for the
    complete four-member base record. Replay deliberately accepts the window
    between those two points.
    """
    mot_path = run_dir / f"{sequence}.txt"
    if mot_path.is_file():
        if mot_path.read_bytes() != mot_bytes:
            raise ChildError("persisted MOT output differs from run products")
    else:
        _write_mot(mot_path, mot_bytes)
    inventory_path = run_dir / evidence.BASE_POLICY_INVENTORY_NAME
    if inventory_path.is_file():
        if (
            evidence.load_document(run_dir, evidence.BASE_POLICY_INVENTORY_NAME)
            != inventory
        ):
            raise ChildError(
                "persisted base policy inventory differs from run products"
            )
    else:
        evidence.write_document(
            run_dir,
            evidence.BASE_POLICY_INVENTORY_NAME,
            inventory,
        )


def execute_child(
    invocation_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    runner: Callable[[Mapping[str, Any]], RunProducts] = repository_runner,
) -> int:
    invocation = load_invocation(invocation_path)
    # Declaration Review Correction 4: the authorization environment is this
    # immutable snapshot, captured before any third-party import and consumed
    # exactly once.  `environment` overrides where the snapshot is taken from,
    # never how it is judged, and nothing downstream compares against it.
    launch_environment = dict(os.environ if environment is None else environment)
    validate_environment(launch_environment, invocation)
    run_dir = invocation_path.parent
    try:
        products = runner(invocation)
        validate_products(str(invocation["run_id"]), products)
        _persist_base_products(
            run_dir,
            str(invocation["sequence"]),
            products.mot_bytes,
            products.policy_base_inventory,
        )
        if products.policy_inventory is not None:
            evidence.write_document(
                run_dir,
                evidence.POLICY_INVENTORY_NAME,
                products.policy_inventory,
            )
        if products.packet is not None:
            packet_path = run_dir / evidence.PACKET_NAME
            if packet_path.is_file():
                if (
                    evidence.load_document(run_dir, evidence.PACKET_NAME)
                    != products.packet
                ):
                    raise ChildError("persisted raw capture differs from run products")
            else:
                evidence.write_document(run_dir, evidence.PACKET_NAME, products.packet)
        if products.packet_verification is not None:
            verification_path = run_dir / evidence.PACKET_VERIFICATION_NAME
            if verification_path.is_file():
                if (
                    evidence.load_document(run_dir, evidence.PACKET_VERIFICATION_NAME)
                    != products.packet_verification
                ):
                    raise ChildError(
                        "persisted packet verification differs from run products"
                    )
            else:
                evidence.write_document(
                    run_dir,
                    evidence.PACKET_VERIFICATION_NAME,
                    products.packet_verification,
                )
        completed = {**invocation, "state": RUN_COMPLETED}
        _replace_invocation(invocation_path, completed)
        return 0
    except BaseException:
        try:
            _replace_invocation(invocation_path, {**invocation, "state": RUN_FAILED})
        except BaseException:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--invocation", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return execute_child(args.invocation)
    except BaseException as exc:
        print(f"H2 measurement child rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
