#!/usr/bin/env python3
"""RC1 fixed Phase-A runtime child (parent-only entry point)."""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


RUN_IDS = (
    "00_capture_off",
    "01_capture_on_1",
    "02_capture_on_2",
    "03_capture_on_3",
)
CHILD_SCHEMA = "h0_phase_a_child_v1"
EXPECTED_ENV_KEYS = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "SACCADE_BUILD_PATH",
        "SACCADE_DETECT_BARRIER",
        "SACCADE_DOUBLE_BUFFER",
        "SACCADE_GPU_DECODE",
        "SACCADE_MAIN_NMS_GRAPHED",
        "TMPDIR",
        "TZ",
        "XDG_CACHE_HOME",
    }
)
STATIC_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "SACCADE_DETECT_BARRIER": "event",
    "SACCADE_DOUBLE_BUFFER": "1",
    "SACCADE_GPU_DECODE": "1",
    "SACCADE_MAIN_NMS_GRAPHED": "1",
    "TZ": "UTC",
}
FORBIDDEN_LABEL_PARTS = frozenset({"gt", "det"})


class ChildContractError(RuntimeError):
    pass


class RuntimeProvenanceError(PermissionError):
    pass


def _parse_argv(argv: Sequence[str]) -> str:
    """Accept exactly ``--run-id <enumerated>``; argparse abbreviation is absent."""
    if len(argv) != 2 or argv[0] != "--run-id" or argv[1] not in RUN_IDS:
        raise ChildContractError("child accepts exactly --run-id <enumerated-id>")
    return argv[1]


def _initial_environment_gate(environment: Mapping[str, str]) -> None:
    if set(environment) != EXPECTED_ENV_KEYS:
        missing = sorted(EXPECTED_ENV_KEYS - set(environment))
        extra = sorted(set(environment) - EXPECTED_ENV_KEYS)
        raise ChildContractError(
            f"RC1.2 environment key mismatch: missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, str) for value in environment.values()):
        raise ChildContractError("RC1.2 environment contains a non-string value")
    for key, expected in STATIC_ENV.items():
        if environment.get(key) != expected:
            raise ChildContractError(f"RC1.2 environment value mismatch for {key}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _environment_digest(environment: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(environment))).hexdigest()


def _physical_root() -> Path:
    script = Path(__file__)
    resolved = script.resolve(strict=True)
    if resolved != script.absolute():
        raise ChildContractError("child script path is symlink-substituted")
    root = resolved.parents[2]
    if root.resolve(strict=True) != root:
        raise ChildContractError("repository root is not physical")
    return root


def invocation_path(root: Path, head: str, run_id: str) -> Path:
    return (
        root
        / "docs/modules/semantic/research/evidence"
        / f"h0_phase_a_{head}.incomplete"
        / "runs"
        / run_id
        / "invocation.json"
    )


def _load_parent_module() -> Any:
    # The child reaches this only after the standard-library-only environment gate.
    tools = Path(__file__).resolve().parent
    if tools.as_posix() not in sys.path:
        sys.path.insert(0, tools.as_posix())
    import run_h0_phase_a

    return run_h0_phase_a


def _validate_full_environment(
    environment: Mapping[str, str],
    root: Path,
    run_dir: Path,
    invocation: Mapping[str, Any],
) -> None:
    parent = _load_parent_module()
    if dict(environment) != invocation.get("environment"):
        raise ChildContractError("process environment differs from child invocation")
    if invocation.get("environment_digest") != _environment_digest(environment):
        raise ChildContractError("child environment digest mismatch")
    expected_run_tmp = run_dir / "_env"
    path_expectations = {
        "HOME": expected_run_tmp / "home",
        "TMPDIR": expected_run_tmp / "tmp",
        "XDG_CACHE_HOME": expected_run_tmp / "xdg-cache",
        "SACCADE_BUILD_PATH": root / "build/h0_phase_a",
    }
    for key, expected in path_expectations.items():
        if environment.get(key) != expected.as_posix():
            raise ChildContractError(f"RC1.2 derived path mismatch for {key}")
        if not expected.is_dir() or expected.resolve(strict=True) != expected:
            raise ChildContractError(
                f"RC1.2 derived directory is absent/non-physical for {key}"
            )
    expected_path = f"{root.as_posix()}/.venv/bin:/usr/bin:/bin"
    if environment.get("PATH") != expected_path:
        raise ChildContractError("RC1.2 PATH mismatch")
    library_members = environment.get("LD_LIBRARY_PATH", "").split(":")
    if (
        len(library_members) != 4
        or not all(library_members)
        or len(set(library_members)) != 4
    ):
        raise ChildContractError(
            "RC1.2 LD_LIBRARY_PATH cardinality/order domain mismatch"
        )
    if library_members[0] != (root / "build/h0_phase_a").as_posix():
        raise ChildContractError("RC1.2 build library directory is not first")
    for member in library_members:
        parent.require_canonical_absolute(member, directory=True)


def classify_access(
    path_value: object, *, root: Path, run_dir: Path, allowed: frozenset[Path]
) -> str:
    """Classify a filesystem access without normalizing a non-canonical request."""
    if isinstance(path_value, int):
        return "file_descriptor"
    if not isinstance(path_value, (str, bytes, os.PathLike)):
        return "unrecognized"
    try:
        path = Path(os.fsdecode(os.fspath(path_value)))
    except (TypeError, UnicodeError):
        return "unrecognized"
    candidate = path if path.is_absolute() else root / path
    lexical_parts = PurePosixPath(candidate.as_posix()).parts
    if ".." in lexical_parts or "." in lexical_parts:
        return "non_canonical"
    if (
        any(part in FORBIDDEN_LABEL_PARTS for part in lexical_parts)
        or "motmetrics" in candidate.as_posix().lower()
    ):
        return "forbidden_label"
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return "unrecognized"
    if resolved == run_dir or run_dir in resolved.parents:
        return "writable_output"
    if resolved in allowed or any(resolved in item.parents for item in allowed):
        return "bound_input"
    return "unexpected"


def install_access_guard(
    *, root: Path, run_dir: Path, allowed: frozenset[Path]
) -> None:
    """Fail closed on Python-audited opens/listings outside bound input/output paths."""

    def audit(event: str, args: tuple[object, ...]) -> None:
        path_value: object | None = None
        if event == "open" and args:
            path_value = args[0]
        elif (
            event in {"os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir"}
            and args
        ):
            path_value = args[0]
        if path_value is None:
            return
        classification = classify_access(
            path_value, root=root, run_dir=run_dir, allowed=allowed
        )
        if classification in {
            "forbidden_label",
            "non_canonical",
            "unexpected",
            "unrecognized",
        }:
            raise RuntimeProvenanceError(
                f"H0 child rejected {classification} filesystem access"
            )

    sys.addaudithook(audit)


def _assert_os_confinement(root: Path, invocation: Mapping[str, Any]) -> None:
    """Prove that the pre-exec OS boundary denies its unbound canary."""
    if invocation.get("confinement_backend") != "landlock_seccomp_ptrace_v1":
        raise ChildContractError("runtime confinement backend mismatch")
    plan_digest = invocation.get("confinement_plan_digest")
    if (
        not isinstance(plan_digest, str)
        or len(plan_digest) != 64
        or any(char not in "0123456789abcdef" for char in plan_digest)
    ):
        raise ChildContractError("runtime confinement plan digest is absent")
    probe = root / invocation["incomplete_root"] / "_runtime_confinement_denial_probe"
    try:
        probe.read_bytes()
    except PermissionError:
        return
    except OSError as exc:
        raise ChildContractError(
            f"runtime confinement denial probe was not mechanically denied: {exc}"
        ) from exc
    raise ChildContractError("runtime confinement denial probe was readable")


@dataclass(frozen=True)
class RunProducts:
    mot_lines: tuple[str, ...]
    policy_inventory: Mapping[str, Any]
    packet: Mapping[str, Any] | None
    packet_verification: Mapping[str, Any] | None


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short child artifact write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value: object) -> None:
    _write_exclusive(path, _canonical_json_bytes(value) + b"\n")


def _validate_products(run_id: str, products: RunProducts) -> None:
    inventory = products.policy_inventory
    required = {
        "schema",
        "mot_output",
        "final_track_rows",
        "active_tid_slot_pairs",
        "relink_debug_raw",
        "proposal_projection",
        "winner_commit_projection",
        "overflow_vector",
    }
    if (
        set(inventory) != required
        or inventory.get("schema") != "h0_phase_a_policy_inventory_v1"
    ):
        raise ChildContractError("policy inventory has missing or unknown members")
    if (
        not isinstance(inventory["relink_debug_raw"], list)
        or len(inventory["relink_debug_raw"]) != 13
    ):
        raise ChildContractError(
            "relink_debug_raw is not the complete 13-integer vector"
        )
    if (
        not isinstance(inventory["overflow_vector"], list)
        or len(inventory["overflow_vector"]) != 9
    ):
        raise ChildContractError(
            "overflow_vector is not the four-semantic/five-native vector"
        )
    if run_id == "00_capture_off":
        if products.packet is not None or products.packet_verification is not None:
            raise ChildContractError("capture-off produced packet data")
        if (
            inventory["proposal_projection"] is not None
            or inventory["winner_commit_projection"] is not None
        ):
            raise ChildContractError("capture-off fabricated trace-only projections")
    elif products.packet is None:
        raise ChildContractError("capture-on omitted its packet")


def _repository_runner(
    run_id: str, evaluator_vector: Sequence[str], capture_run_uuid: str
) -> RunProducts:
    """Resolve the sole preset and invoke ``evaluator.run_eval`` exactly once."""
    root = _physical_root()
    eval_dir = root / "scripts/eval"
    if eval_dir.as_posix() not in sys.path:
        sys.path.insert(0, eval_dir.as_posix())
    if (root / "src").as_posix() not in sys.path:
        sys.path.insert(0, (root / "src").as_posix())

    import yaml
    from mot17_args import build_parser, configure_runtime_env
    from resolved_bridge_policy_config import fingerprint
    from saccade.perception.eval import evaluator as evaluator_module
    from saccade.perception.eval import stages as stages_module
    from saccade.perception.eval.pipeline import EvalPipeline
    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
        set_postprocess_compile,
    )
    from export_headline_bridge_decision_trace import canonical_semantic_packet

    parser = build_parser()
    preset_path = root / "configs/presets/mamba_whole_graph_m.yaml"
    defaults = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(defaults, dict):
        raise ChildContractError("sealed preset is not a mapping")
    parser.set_defaults(**defaults)
    args = parser.parse_args(list(evaluator_vector))
    if (
        fingerprint("mamba_whole_graph_m")
        != "c7a6dbb35168cba75249b7f2c67d8455b6f634732493e455a4bb920aab6d7782"
    ):
        raise ChildContractError("resolved m policy fingerprint mismatch")
    configure_runtime_env(args, os.environ)
    if set(os.environ) != EXPECTED_ENV_KEYS:
        raise ChildContractError(
            "runtime configuration mutated the sanitized environment"
        )

    tiling = getattr(args, "tiling", "native_640")
    image_size = (
        1280
        if "1280" in tiling
        else 1024
        if "1024" in tiling
        else 960
        if "960" in tiling
        else 640
    )
    detector = build_mamba_gated_detector(
        yolo_pt_path=args.mamba_yolo_weights,
        teacher_ckpt=args.mamba_teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=image_size,
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
    raw_final_rows: list[dict[str, Any]] = []
    pipeline_slots: list[Any] = []
    original_init = EvalPipeline.__init__

    def controlled_init(self: Any, *positional: Any, **keywords: Any) -> None:
        original_init(self, *positional, **keywords)
        if self.seq != "MOT17-04-SDP" or pipeline_slots:
            raise ChildContractError(
                "evaluator constructed a missing/duplicate/second sequence"
            )
        pipeline_slots.append(self)
        tracker = self.detector.tracker
        enabled = run_id != "00_capture_off"
        tracker.set_research_h0_bridge_trace(
            enabled,
            pair_capacity=65536,
            candidate_capacity=16384,
            claim_capacity=16384,
            commit_capacity=16384,
        )
        if enabled:
            tracker.clear_research_h0_bridge_trace()

    EvalPipeline.__init__ = controlled_init
    evaluator_module.EvalPipeline = EvalPipeline
    original_run_frame = evaluator_module._run_frame

    def controlled_run_frame(
        state: Any, *, frame_id: int, prepared_detection: Any = None
    ) -> bool:
        outcome = original_run_frame(
            state, frame_id=frame_id, prepared_detection=prepared_detection
        )
        native = state.detector.tracker.tracker
        pairs = [
            [int(tid), int(slot)] for tid, slot in native.get_active_tid_slot_pairs()
        ]
        if pairs != sorted(pairs, key=lambda pair: pair[1]):
            raise ChildContractError("active tid/slot pairs are not sorted by slot")
        active_pairs.append({"frame": frame_id, "pairs": pairs})
        return outcome

    evaluator_module._run_frame = controlled_run_frame
    original_evaluator_emit = evaluator_module._fast_emit_mot_lines
    original_stages_emit = stages_module._fast_emit_mot_lines

    def capturing_emit(
        original: Callable[..., list[str]], **keywords: Any
    ) -> list[str]:
        lines = original(**keywords)
        track_results = keywords["track_results"]
        count = int(track_results["count"])
        if len(lines) != count or keywords["seq"] != "MOT17-04-SDP":
            raise ChildContractError("raw emission cardinality/sequence mismatch")
        boxes = track_results["boxes"][:count].numpy()
        scores = track_results["scores"][:count].numpy()
        classes_value = track_results.get("classes")
        if classes_value is None:
            classes = [int(getattr(args, "person_class", 0))] * count
        else:
            classes = [int(value) for value in classes_value[:count].numpy()]
        frame = int(keywords["frame_id"])
        row_base = sum(1 for row in raw_final_rows if row["frame"] == frame)
        for offset, (line, box, score, class_id) in enumerate(
            zip(lines, boxes, scores, classes, strict=True)
        ):
            fields = line.split(",")
            if len(fields) != 10 or int(fields[0]) != frame:
                raise ChildContractError("raw emission/MOT row mismatch")
            x1, y1, x2, y2 = (float(value) for value in box)
            values = (x1, y1, x2 - x1, y2 - y1, float(score))
            raw_final_rows.append(
                {
                    "binary32_bits": [
                        struct.unpack("!I", struct.pack("!f", value))[0]
                        for value in values
                    ],
                    "class": class_id,
                    "frame": frame,
                    "row_index": row_base + offset,
                    "track_id": int(fields[1]),
                }
            )
        return lines

    def evaluator_emit(**keywords: Any) -> list[str]:
        return capturing_emit(original_evaluator_emit, **keywords)

    def stages_emit(**keywords: Any) -> list[str]:
        return capturing_emit(original_stages_emit, **keywords)

    evaluator_module._fast_emit_mot_lines = evaluator_emit
    stages_module._fast_emit_mot_lines = stages_emit
    callbacks: list[tuple[str, tuple[str, ...]]] = []

    def sequence_callback(sequence: str, lines: tuple[str, ...]) -> None:
        if callbacks or sequence != "MOT17-04-SDP" or not isinstance(lines, tuple):
            raise ChildContractError(
                "sequence_result_callback cardinality/type mismatch"
            )
        callbacks.append((sequence, lines))

    module_keys = {
        "module_detection",
        "module_geometry",
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    }
    mamba_keys = {
        "mamba_ckpt",
        "mamba_teacher_ckpt",
        "mamba_yolo_weights",
        "teacher_head_ckpt",
    }
    visual_keys = {
        "visualize",
        "visualize_scale",
        "visualize_fps",
        "no_visualize_score",
        "visualize_trail_len",
    }
    eval_kwargs = {
        key: value
        for key, value in vars(args).items()
        if key not in module_keys | mamba_keys | visual_keys
    }
    eval_kwargs["detector"] = detector
    eval_kwargs["tiling"] = tiling
    eval_kwargs["engine"] = "mamba"
    eval_kwargs["sequence_result_callback"] = sequence_callback
    try:
        result = evaluator_module.run_eval(**eval_kwargs)
    finally:
        EvalPipeline.__init__ = original_init
        evaluator_module.EvalPipeline = EvalPipeline
        evaluator_module._run_frame = original_run_frame
        evaluator_module._fast_emit_mot_lines = original_evaluator_emit
        stages_module._fast_emit_mot_lines = original_stages_emit
    if result != {} or len(callbacks) != 1 or len(pipeline_slots) != 1:
        raise ChildContractError(
            "evaluator did not return at the sole no-metrics boundary"
        )
    sequence_info = configparser.ConfigParser(interpolation=None)
    sequence_info.optionxform = str
    sequence_path = root / "datasets/MOT17/train/MOT17-04-SDP/seqinfo.ini"
    try:
        with sequence_path.open("r", encoding="utf-8") as handle:
            sequence_info.read_file(handle)
        frame_count = int(sequence_info["Sequence"]["seqLength"])
    except (OSError, KeyError, ValueError, configparser.Error) as exc:
        raise ChildContractError(
            "canonical sequence frame count is unavailable"
        ) from exc
    if [row["frame"] for row in active_pairs] != list(range(1, frame_count + 1)):
        raise ChildContractError(
            "child did not process the complete sequence exactly once"
        )

    lines = callbacks[0][1]
    mot_bytes = "\n".join(lines).encode("utf-8")
    callback_rows: list[tuple[int, int, int]] = []
    row_positions: dict[int, int] = {}
    for line in lines:
        fields = line.split(",")
        if len(fields) != 10:
            raise ChildContractError("non-canonical MOT result row")
        frame = int(fields[0])
        position = row_positions.get(frame, 0)
        row_positions[frame] = position + 1
        callback_rows.append((frame, position, int(fields[1])))
    raw_row_keys = [
        (int(row["frame"]), int(row["row_index"]), int(row["track_id"]))
        for row in raw_final_rows
    ]
    if raw_row_keys != callback_rows:
        raise ChildContractError(
            "raw binary32 emission rows do not equal callback order"
        )
    tracker = pipeline_slots[0].detector.tracker
    debug = [int(value) for value in tracker.get_relink_debug()]
    if len(debug) != 13:
        raise ChildContractError("native relink debug vector length drift")

    packet: Mapping[str, Any] | None = None
    packet_verification: Mapping[str, Any] | None = None
    proposal: Mapping[str, Any] | None = None
    winner_commit: Mapping[str, Any] | None = None
    overflow = [0] * 9
    if run_id != "00_capture_off":
        capture = tracker.drain_research_h0_bridge_trace(
            seq="MOT17-04-SDP",
            capture_phase="phase_a",
            require_candidate_exposure=True,
            require_commit_exposure=False,
            capture_run_uuid=capture_run_uuid,
        )
        packet = canonical_semantic_packet(capture)
        streams = packet["streams"]
        candidates = [
            row
            for row in streams["candidate_records"]
            if int(row["proposal_emitted"]) == 1
        ]
        claims = streams["claim_records"]
        commits = streams["commit_records"]
        proposal_payload = {"candidates": candidates, "claims": claims}
        winner_payload = {
            "commits": commits,
            "winning_claims": [row for row in claims if int(row["claim_won"]) == 1],
        }
        proposal = {
            "count": len(candidates),
            "digest": hashlib.sha256(
                _canonical_json_bytes(proposal_payload)
            ).hexdigest(),
            "records": proposal_payload,
        }
        winner_commit = {
            "count": len(commits),
            "digest": hashlib.sha256(_canonical_json_bytes(winner_payload)).hexdigest(),
            "records": winner_payload,
        }
        overflow = [
            int(capture[key])
            for key in (
                "overflow_pair_records",
                "overflow_candidate_records",
                "overflow_claim_records",
                "overflow_commit_records",
                "overflow_native_candidate_keys",
                "overflow_native_pair_keys",
                "overflow_native_proposal_keys",
                "overflow_native_claim_winner_keys",
                "overflow_native_commit_keys",
            )
        ]
        # ``packet.json`` retains the complete verifier input.  The canonical
        # exporter above is invoked exactly once here; replay is deferred until
        # run 3 can decide the cross-run A7.6 comparison without speculation.
        packet = capture
    inventory = {
        "active_tid_slot_pairs": active_pairs,
        "final_track_rows": raw_final_rows,
        "mot_output": {
            "length": len(mot_bytes),
            "sha256": hashlib.sha256(mot_bytes).hexdigest(),
        },
        "overflow_vector": overflow,
        "proposal_projection": proposal,
        "relink_debug_raw": debug,
        "schema": "h0_phase_a_policy_inventory_v1",
        "winner_commit_projection": winner_commit,
    }
    return RunProducts(lines, inventory, packet, packet_verification)


def _finalize_packet_verifications_if_required(incomplete: Path) -> None:
    """After run 3, create V iff the four complete D inventories compare equal."""
    parent = _load_parent_module()
    equal, _comparison = parent._compare_policy_inventories(incomplete)
    if not equal:
        return
    from verify_headline_bridge_decision_trace import verify_capture

    for run_id in RUN_IDS[1:]:
        run_dir = incomplete / "runs" / run_id
        capture = parent.read_canonical_json(run_dir / "packet.json")
        try:
            report = verify_capture(capture)
            verification: dict[str, Any] = {"report": report, "state": "pass"}
        except (KeyError, TypeError, ValueError):
            verification = {"failure": "packet_invalid", "state": "fail"}
        _write_json(run_dir / "packet_verification.json", verification)


def _replace_invocation(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(".invocation.json.tmp")
    _write_exclusive(temporary, _canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def execute_child(
    run_id: str,
    *,
    environment: Mapping[str, str] | None = None,
    runner: Callable[[str, Sequence[str], str], RunProducts] = _repository_runner,
) -> int:
    env = dict(os.environ if environment is None else environment)
    _initial_environment_gate(env)
    root = _physical_root()
    parent = _load_parent_module()
    _freeze_path, controller = parent._discover_controller_input(root)
    head = controller["instrumentation_head"]
    path = invocation_path(root, head, run_id)
    invocation = parent.read_canonical_json(path)
    parent.validate_schema_document(invocation, "child_invocation")
    if (
        invocation.get("run_id") != run_id
        or invocation.get("instrumentation_head") != head
    ):
        raise ChildContractError("child invocation run/head mismatch")
    expected_vector = list(parent.child_argv(root, run_id))
    if invocation.get("vector") != expected_vector:
        raise ChildContractError("canonical child vector mismatch")
    run_dir = path.parent
    expected_evaluator = list(parent.evaluator_argv(run_dir))
    if invocation.get("evaluator_argv") != expected_evaluator:
        raise ChildContractError("canonical synthetic evaluator vector mismatch")
    _validate_full_environment(env, root, run_dir, invocation)
    _assert_os_confinement(root, invocation)
    confinement_probe_passed = True
    probed = dict(invocation)
    probed["confinement_probe_passed"] = True
    _replace_invocation(path, probed)
    invocation = probed
    if controller["instrumentation_head"] != head:
        raise ChildContractError("child and v3 instrumentation heads differ")
    if controller["bound_inputs"]["digest"] != invocation["bound_inputs_digest"]:
        raise ChildContractError("child and v3 bound-input digests differ")
    allowed_paths = {
        path.resolve(strict=True) for path in parent.bound_file_paths(controller)
    }
    allowed_paths.add(Path(os.devnull).resolve(strict=True))
    build_root = root / "build/h0_phase_a"
    allowed_paths.update(
        path.resolve(strict=True) for path in build_root.rglob("*") if path.is_file()
    )
    allowed = frozenset(allowed_paths)
    install_access_guard(root=root, run_dir=run_dir, allowed=allowed)

    try:
        products = runner(run_id, expected_evaluator, invocation["capture_run_uuid"])
        _validate_products(run_id, products)
        mot_bytes = "\n".join(products.mot_lines).encode("utf-8")
        mot_slot = products.policy_inventory["mot_output"]
        if mot_slot != {
            "length": len(mot_bytes),
            "sha256": hashlib.sha256(mot_bytes).hexdigest(),
        }:
            raise ChildContractError(
                "MOT callback bytes disagree with policy inventory"
            )
        _write_exclusive(run_dir / "MOT17-04-SDP.txt", mot_bytes)
        _write_json(run_dir / "policy_inventory.json", products.policy_inventory)
        if products.packet is not None:
            _write_json(run_dir / "packet.json", products.packet)
        if products.packet_verification is not None:
            _write_json(
                run_dir / "packet_verification.json", products.packet_verification
            )
        if run_id == "03_capture_on_3":
            incomplete = root / invocation["incomplete_root"]
            _finalize_packet_verifications_if_required(incomplete)
        completed = dict(invocation)
        completed["confinement_probe_passed"] = confinement_probe_passed
        completed["result"] = "run_completed"
        completed["state"] = "completed"
        _replace_invocation(path, completed)
        return 0
    except BaseException as exc:
        failed = dict(invocation)
        failed["confinement_probe_passed"] = locals().get(
            "confinement_probe_passed", False
        )
        failed["result"] = (
            "provenance_invalid"
            if isinstance(exc, RuntimeProvenanceError)
            else "runner_nonzero"
        )
        failed["state"] = "failed"
        try:
            _replace_invocation(path, failed)
        except BaseException:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    values = tuple(sys.argv[1:] if argv is None else argv)
    try:
        run_id = _parse_argv(values)
        _initial_environment_gate(os.environ)
        return execute_child(run_id)
    except BaseException as exc:
        print(f"H0 Phase-A child rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
