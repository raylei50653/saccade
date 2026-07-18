#!/usr/bin/env python3
"""Independent A7/RC1 aggregate verifier for Phase-A execution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = TOOLS_DIR / "h0_phase_a_execution_schema_v1.json"
RUN_IDS = ("00_capture_off", "01_capture_on_1", "02_capture_on_2", "03_capture_on_3")
RESULTS = (
    "provenance_invalid",
    "build_failed",
    "extension_load_failed",
    "runner_nonzero",
    "runner_timeout",
    "serialization_failed",
    "artifact_missing_or_unreadable",
    "unclassified_execution_failure",
    "capture_perturbs_policy",
    "packet_invalid",
    "phase_a_pass",
)
BUILD_VECTORS = (
    (
        "uv",
        "run",
        "--frozen",
        "cmake",
        "--fresh",
        "-S",
        ".",
        "-B",
        "build/h0_phase_a",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DENABLE_NATIVE_TESTS=OFF",
        "-DSACCADE_ENABLE_NVTX=ON",
        "-DPython3_EXECUTABLE=.venv/bin/python",
    ),
    (
        "uv",
        "run",
        "--frozen",
        "cmake",
        "--build",
        "build/h0_phase_a",
        "--target",
        "saccade_tracking_ext",
        "saccade_scan_plugin",
        "--parallel",
        "1",
    ),
)
BUILD_ENVIRONMENT_V1_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "TMPDIR",
    "TZ",
    "XDG_CACHE_HOME",
)
BUILD_ENVIRONMENT_KEYS = (
    "CUDACXX",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "TMPDIR",
    "TZ",
    "XDG_CACHE_HOME",
)
EVALUATOR_PREFIX = (
    "--preset",
    "mamba_whole_graph_m",
    "--detector",
    "SDP",
    "--data-root",
    "datasets/MOT17",
    "--split",
    "train",
    "--sequences",
    "MOT17-04-SDP",
    "--max-frames",
    "0",
    "--warmup-frames",
    "0",
    "--latency-only",
    "--gpu-decode",
    "--double-buffer",
    "--detect-barrier",
    "event",
    "--main-nms-graphed",
    "--processes",
    "0",
    "--output",
)
ENV_KEYS = (
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
)
INOTIFY_NAMES = (
    "IN_CLOSE_WRITE",
    "IN_MODIFY",
    "IN_ATTRIB",
    "IN_DELETE_SELF",
    "IN_MOVE_SELF",
    "IN_CREATE",
    "IN_DELETE",
    "IN_MOVED_FROM",
    "IN_MOVED_TO",
)
CHECKPOINTS = (
    "T0",
    "T1",
    "T2a_0",
    "T2b_0",
    "T2a_1",
    "T2b_1",
    "T2a_2",
    "T2b_2",
    "T2a_3",
    "T2b_3",
    "T3",
    "T4",
)
MODEL_INPUTS = (
    "models/yolo/mamba_head_26m.engine",
    "models/yolo/yolo26m.pt",
    "models/yolo/yolo26m_backbone_640_best.engine",
    "runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt",
    "runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt",
)
REQUIRED_REPOSITORY_INPUTS = (
    "configs/presets/mamba_whole_graph_m.yaml",
    "docs/modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md",
    "docs/modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.policy.yaml",
    "scripts/tools/export_headline_bridge_decision_trace.py",
    "scripts/tools/h0_bridge_decision_trace_schema_v2.json",
    "scripts/tools/h0_phase_a_execution_schema_v1.json",
    "scripts/tools/h0_runtime_confinement.py",
    "scripts/tools/resolved_bridge_policy_config.py",
    "scripts/tools/run_h0_phase_a.py",
    "scripts/tools/run_h0_phase_a_child.py",
    "scripts/tools/verify_h0_phase_a.py",
    "scripts/tools/verify_headline_bridge_decision_trace.py",
    "uv.lock",
)
C_PATHS = (
    "manifest.json",
    "build_identity.json",
    "runtime_identity.json",
    "gpu_identity.json",
    "input_binding.json",
    "comparison.json",
    "result.json",
    "checksums.sha256",
    "logs/00_cmake_configure.stdout.log",
    "logs/00_cmake_configure.stderr.log",
    "logs/01_cmake_build.stdout.log",
    "logs/01_cmake_build.stderr.log",
    "runs/00_capture_off/invocation.json",
    "runs/00_capture_off/stdout.log",
    "runs/00_capture_off/stderr.log",
    "runs/01_capture_on_1/invocation.json",
    "runs/01_capture_on_1/stdout.log",
    "runs/01_capture_on_1/stderr.log",
    "runs/02_capture_on_2/invocation.json",
    "runs/02_capture_on_2/stdout.log",
    "runs/02_capture_on_2/stderr.log",
    "runs/03_capture_on_3/invocation.json",
    "runs/03_capture_on_3/stdout.log",
    "runs/03_capture_on_3/stderr.log",
    "verification/aggregate.json",
)
D_PATHS = (
    "runs/00_capture_off/policy_inventory.json",
    "runs/00_capture_off/MOT17-04-SDP.txt",
    "runs/01_capture_on_1/policy_inventory.json",
    "runs/01_capture_on_1/MOT17-04-SDP.txt",
    "runs/01_capture_on_1/packet.json",
    "runs/02_capture_on_2/policy_inventory.json",
    "runs/02_capture_on_2/MOT17-04-SDP.txt",
    "runs/02_capture_on_2/packet.json",
    "runs/03_capture_on_3/policy_inventory.json",
    "runs/03_capture_on_3/MOT17-04-SDP.txt",
    "runs/03_capture_on_3/packet.json",
)
V_PATHS = (
    "runs/01_capture_on_1/packet_verification.json",
    "runs/02_capture_on_2/packet_verification.json",
    "runs/03_capture_on_3/packet_verification.json",
)
BLOCKED_RUNTIME_INGRESS_SYSCALLS = (
    "pipe",
    "shmget",
    "shmat",
    "shmctl",
    "socket",
    "connect",
    "accept",
    "sendto",
    "recvfrom",
    "sendmsg",
    "recvmsg",
    "shutdown",
    "bind",
    "listen",
    "getsockname",
    "getpeername",
    "socketpair",
    "setsockopt",
    "getsockopt",
    "semget",
    "semop",
    "semctl",
    "shmdt",
    "msgget",
    "msgsnd",
    "msgrcv",
    "msgctl",
    "ptrace",
    "syslog",
    "mq_open",
    "mq_unlink",
    "mq_timedsend",
    "mq_timedreceive",
    "mq_notify",
    "mq_getsetattr",
    "add_key",
    "request_key",
    "keyctl",
    "inotify_init",
    "inotify_add_watch",
    "inotify_rm_watch",
    "splice",
    "tee",
    "vmsplice",
    "accept4",
    "pipe2",
    "inotify_init1",
    "perf_event_open",
    "recvmmsg",
    "fanotify_init",
    "fanotify_mark",
    "name_to_handle_at",
    "open_by_handle_at",
    "sendmmsg",
    "process_vm_readv",
    "process_vm_writev",
    "kcmp",
    "memfd_create",
    "bpf",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "pidfd_open",
    "pidfd_getfd",
    "process_madvise",
)


class VerificationError(RuntimeError):
    pass


_SCHEMA_CACHE: Any | None = None


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON member {key}")
        result[key] = value
    return result


def load_json(path: Path, *, canonical_file: bool) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                VerificationError(f"non-finite number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"malformed JSON {path}: {exc}") from exc
    if canonical_file and raw != canonical_json(value) + b"\n":
        raise VerificationError(f"non-canonical execution JSON: {path}")
    return value


def _schema_validate(value: object) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise VerificationError("jsonschema dependency unavailable") from exc
    schema = _schema_document()
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(value),
            key=lambda error: list(error.absolute_path),
        )
    except jsonschema.SchemaError as exc:
        raise VerificationError(f"invalid execution schema: {exc.message}") from exc
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise VerificationError(f"schema rejection at {location}: {error.message}")


def _schema_document() -> Any:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = load_json(SCHEMA_PATH, canonical_file=False)
    return _SCHEMA_CACHE


def expected_matrix() -> dict[str, dict[str, list[str]]]:
    all_paths = C_PATHS + D_PATHS + V_PATHS
    matrix = {
        result: {"forbidden": list(D_PATHS + V_PATHS), "required": list(C_PATHS)}
        for result in RESULTS[:8]
    }
    matrix["capture_perturbs_policy"] = {
        "forbidden": list(V_PATHS),
        "required": list(C_PATHS + D_PATHS),
    }
    matrix["packet_invalid"] = {"forbidden": [], "required": list(all_paths)}
    matrix["phase_a_pass"] = {"forbidden": [], "required": list(all_paths)}
    return matrix


def select_result(predicates: Mapping[str, bool]) -> str:
    if not predicates["provenance_ok"]:
        return "provenance_invalid"
    if not predicates["build_ok"]:
        return "build_failed"
    if not predicates["extension_ok"]:
        return "extension_load_failed"
    if not predicates["runners_ok"]:
        return "runner_nonzero"
    if predicates["timed_out"]:
        return "runner_timeout"
    if not predicates["serialization_ok"]:
        return "serialization_failed"
    if not predicates["artifacts_ok"]:
        return "artifact_missing_or_unreadable"
    if not predicates["classified_execution"]:
        return "unclassified_execution_failure"
    if not predicates["policy_equal"]:
        return "capture_perturbs_policy"
    if not predicates["packets_valid"]:
        return "packet_invalid"
    return "phase_a_pass"


def _expected_child_vector(root: str, run_id: str) -> list[str]:
    return [
        f"{root}/.venv/bin/python",
        "-I",
        "-B",
        f"{root}/scripts/tools/run_h0_phase_a_child.py",
        "--run-id",
        run_id,
    ]


def _expected_environment(controller: Mapping[str, Any], run_id: str) -> dict[str, str]:
    root = controller["repository_root"]
    run_dir = f"{root}/{controller['incomplete_root']}/runs/{run_id}"
    temp = f"{run_dir}/_env"
    libraries = controller["library_dirs"]
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": controller["gpu"]["uuid"],
        "HOME": f"{temp}/home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ":".join(
            (
                f"{root}/build/h0_phase_a",
                libraries["tensorrt_library_dir"],
                libraries["pytorch_library_dir"],
                libraries["cuda_library_dir"],
            )
        ),
        "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SACCADE_BUILD_PATH": f"{root}/build/h0_phase_a",
        "SACCADE_DETECT_BARRIER": "event",
        "SACCADE_DOUBLE_BUFFER": "1",
        "SACCADE_GPU_DECODE": "1",
        "SACCADE_MAIN_NMS_GRAPHED": "1",
        "TMPDIR": f"{temp}/tmp",
        "TZ": "UTC",
        "XDG_CACHE_HOME": f"{temp}/xdg-cache",
    }


def _expected_build_environment(controller: Mapping[str, Any]) -> dict[str, str]:
    root = controller["repository_root"]
    environment_root = f"{root}/{controller['incomplete_root']}/_build_env"
    environment = {
        "HOME": f"{environment_root}/home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": f"{environment_root}/tmp",
        "TZ": "UTC",
        "XDG_CACHE_HOME": f"{environment_root}/xdg-cache",
    }
    if _build_environment_algorithm(controller) == "h0_build_environment_v2":
        return {"CUDACXX": _bound_nvcc_path(controller), **environment}
    return environment


def _build_environment_algorithm(controller: Mapping[str, Any]) -> str:
    """Admit only the exact historical v1 or repaired v2 build environment."""
    try:
        constants = controller["execution_constants"]
        algorithm = constants["build_environment_algorithm"]
        keys = constants["build_environment_keys"]
    except (KeyError, TypeError) as exc:
        raise VerificationError("build environment declaration is malformed") from exc
    expected = {
        "h0_build_environment_v1": list(BUILD_ENVIRONMENT_V1_KEYS),
        "h0_build_environment_v2": list(BUILD_ENVIRONMENT_KEYS),
    }
    if algorithm not in expected or keys != expected[algorithm]:
        raise VerificationError("build environment declaration mismatch")
    return algorithm


def _bound_nvcc_record(controller: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the frozen compiler from the controller's bound inventory.

    Archive verification is host-independent: the selected compiler identity
    is proven by the frozen tool_runtime record alone, never by re-reading the
    execution host's absolute paths.
    """
    try:
        selected = str(controller["tool_paths"]["nvcc"])
        records = controller["bound_inputs"]["tool_runtime"]
    except (KeyError, TypeError) as exc:
        raise VerificationError("controller has no selected nvcc binding") from exc
    matches = [record for record in records if record.get("logical_path") == selected]
    if len(matches) != 1:
        raise VerificationError("selected nvcc is absent or ambiguous in tool_runtime")
    record = matches[0]
    try:
        realpath = str(record["realpath"])
        length = int(record["length"])
        sha256_value = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("selected nvcc bound identity is malformed") from exc
    if (
        not realpath.startswith("/")
        or realpath != selected
        or length < 0
        or len(sha256_value) != 64
        or any(char not in "0123456789abcdef" for char in sha256_value)
    ):
        raise VerificationError("selected nvcc differs from its bound identity")
    return {"length": length, "realpath": realpath, "sha256": sha256_value}


def _bound_nvcc_path(controller: Mapping[str, Any]) -> str:
    return str(_bound_nvcc_record(controller)["realpath"])


def _expected_extension_load_environment(
    controller: Mapping[str, Any],
) -> dict[str, str]:
    root = controller["repository_root"]
    libraries = controller["library_dirs"]
    return {
        **_expected_build_environment(controller),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": controller["gpu"]["uuid"],
        "LD_LIBRARY_PATH": ":".join(
            (
                f"{root}/build/h0_phase_a",
                libraries["tensorrt_library_dir"],
                libraries["pytorch_library_dir"],
                libraries["cuda_library_dir"],
            )
        ),
        "SACCADE_BUILD_PATH": f"{root}/build/h0_phase_a",
    }


def _expected_extension_load_vector(
    controller: Mapping[str, Any], identity: Mapping[str, Any]
) -> list[str]:
    root = controller["repository_root"]
    incomplete = f"{root}/{controller['incomplete_root']}"
    extension = f"{root}/{identity['artifacts'][0]['path']}"
    plugin = f"{root}/{identity['artifacts'][1]['path']}"
    probe = f"{incomplete}/_runtime_confinement_denial_probe"
    script = (
        "import ctypes,pathlib;"
        f"p=pathlib.Path({probe!r});"
        "denied=False;"
        "\ntry:p.read_bytes()"
        "\nexcept PermissionError:denied=True"
        "\nassert denied;"
        "\nimport saccade_tracking_ext;"
        f"e=pathlib.Path({extension!r}).resolve(strict=True);"
        "a=pathlib.Path(saccade_tracking_ext.__file__).resolve(strict=True);"
        "assert a==e;"
        f"ctypes.CDLL({plugin!r},mode=ctypes.RTLD_LOCAL)"
    )
    return [f"{root}/.venv/bin/python", "-I", "-B", "-c", script]


def _verify_constants(controller: Mapping[str, Any]) -> None:
    root = controller["repository_root"]
    constants = controller["execution_constants"]
    if constants["operator_vector"] != [
        "uv",
        "run",
        "--frozen",
        "python",
        "scripts/tools/run_h0_phase_a.py",
    ]:
        raise VerificationError("operator vector mismatch")
    if constants["build_vectors"] != [list(value) for value in BUILD_VECTORS]:
        raise VerificationError("build vector mismatch")
    algorithm = _build_environment_algorithm(controller)
    expected_tools = {
        "git",
        "ldd",
        "readelf",
        "uv",
    }
    if algorithm == "h0_build_environment_v2":
        expected_tools.add("nvcc")
    if set(controller["tool_paths"]) != expected_tools:
        raise VerificationError("build tool-path declaration mismatch")
    if constants["ordered_run_plan"] != list(RUN_IDS):
        raise VerificationError("ordered run plan mismatch")
    if constants["child_vectors"] != [
        _expected_child_vector(root, run_id) for run_id in RUN_IDS
    ]:
        raise VerificationError("child vector mismatch")
    if constants["evaluator_argv_prefix"] != list(EVALUATOR_PREFIX):
        raise VerificationError("synthetic evaluator vector mismatch")
    if constants["environment_keys"] != list(ENV_KEYS):
        raise VerificationError("environment table key order mismatch")
    if constants["inotify_mask"] != list(INOTIFY_NAMES):
        raise VerificationError("inotify mask mismatch")
    if constants["model_inputs"] != list(MODEL_INPUTS):
        raise VerificationError("resolved model/engine input set mismatch")
    if constants["required_repository_inputs"] != list(REQUIRED_REPOSITORY_INPUTS):
        raise VerificationError("required controller/runtime authority set mismatch")
    if constants["checkpoints"] != list(CHECKPOINTS):
        raise VerificationError("bound-input checkpoint order mismatch")
    if constants["deadline_seconds"] != 3600 or constants["trace_capacities"] != [
        65536,
        16384,
        16384,
        16384,
    ]:
        raise VerificationError("deadline or trace capacity mismatch")
    if constants["result_enum"] != list(RESULTS):
        raise VerificationError("result enum/order mismatch")
    if (
        constants["c_paths"] != list(C_PATHS)
        or constants["d_paths"] != list(D_PATHS)
        or constants["v_paths"] != list(V_PATHS)
    ):
        raise VerificationError("C/D/V path set mismatch")
    if constants["result_matrix"] != expected_matrix():
        raise VerificationError("execution-constant C/D/V matrix mismatch")


def _verify_bound_inputs(controller: Mapping[str, Any]) -> None:
    inventory = controller["bound_inputs"]
    if (
        tuple(record["logical_path"] for record in inventory["models_engines"])
        != MODEL_INPUTS
    ):
        raise VerificationError(
            "bound model/engine inventory differs from resolved evaluator"
        )
    sequence = inventory["sequence"]
    paths = [record["path"] for record in sequence["files"]]
    if paths != sorted(paths, key=lambda path: path.encode("utf-8")) or len(
        paths
    ) != len(set(paths)):
        raise VerificationError("sequence inventory path ordering/uniqueness mismatch")
    for path in paths:
        parts = Path(path).parts
        if path.startswith("/") or ".." in parts or "." in parts or "\\" in path:
            raise VerificationError("non-canonical sequence path")
        if parts and parts[0] in {"gt", "det"}:
            raise VerificationError("sequence inventory includes excluded subtree")
    expected_sequence = digest(
        {"algorithm": "h0_sequence_inputs_v1", "files": sequence["files"]}
    )
    if (
        sequence["digest"] != expected_sequence
        or controller["sequence_input_digest"] != expected_sequence
    ):
        raise VerificationError("canonical sequence-input digest mismatch")
    for category in ("repository", "models_engines", "tool_runtime"):
        path_key = "path" if category == "repository" else "logical_path"
        encoded = [record[path_key].encode("utf-8") for record in inventory[category]]
        if encoded != sorted(encoded) or len(encoded) != len(set(encoded)):
            raise VerificationError(f"{category} inventory order/uniqueness mismatch")
        if category != "repository":
            realpaths = [record["realpath"] for record in inventory[category]]
            if len(realpaths) != len(set(realpaths)):
                raise VerificationError(
                    f"{category} inventory duplicates a physical path"
                )
    repository_paths = {record["path"] for record in inventory["repository"]}
    if not set(REQUIRED_REPOSITORY_INPUTS).issubset(repository_paths):
        raise VerificationError("repository inventory omits an A7/RC1 authority")
    payload = {
        "models_engines": inventory["models_engines"],
        "repository": inventory["repository"],
        "schema": "h0_bound_inputs_v1",
        "sequence": sequence,
        "tool_runtime": inventory["tool_runtime"],
    }
    if inventory["digest"] != digest(payload):
        raise VerificationError("h0_bound_inputs_v1 digest mismatch")


def _verify_children(evidence: Mapping[str, Any]) -> None:
    controller = evidence["controller_input"]
    children = evidence["child_invocations"]
    if [child["run_id"] for child in children] != list(RUN_IDS):
        raise VerificationError("child cardinality/order mismatch")
    for child, run_id in zip(children, RUN_IDS, strict=True):
        if child["instrumentation_head"] != controller["instrumentation_head"]:
            raise VerificationError("parent/child head mismatch")
        if child["bound_inputs_digest"] != controller["bound_inputs"]["digest"]:
            raise VerificationError("parent/child bound-input digest mismatch")
        expected_vector = _expected_child_vector(controller["repository_root"], run_id)
        if child["vector"] != expected_vector:
            raise VerificationError("parent/child command vector mismatch")
        run_dir = f"{controller['repository_root']}/{controller['incomplete_root']}/runs/{run_id}"
        if child["evaluator_argv"] != list(EVALUATOR_PREFIX) + [f"{run_dir}/_runtime"]:
            raise VerificationError("child evaluator vector/output mismatch")
        environment = _expected_environment(controller, run_id)
        if child["environment"] != environment or child["environment_digest"] != digest(
            environment
        ):
            raise VerificationError("child environment table/digest mismatch")
        if child["state"] == "completed" and (
            child["confinement_probe_passed"] is not True
            or child["confinement_plan_digest"] is None
            or child["runtime_inputs_digest"] is None
        ):
            raise VerificationError("completed child lacks runtime confinement proof")
        if child["state"] == "not_run" and any(
            child[key] is not None
            for key in (
                "confinement_plan_digest",
                "confinement_probe_passed",
                "runtime_inputs_digest",
            )
        ):
            raise VerificationError(
                "not-run child fabricates runtime confinement proof"
            )


def _verify_input_binding(evidence: Mapping[str, Any]) -> None:
    binding = evidence["input_binding"]
    controller_digest = evidence["controller_input"]["bound_inputs"]["digest"]
    new_format = "failure" in binding
    if binding["t0_digest"] != controller_digest:
        raise VerificationError("T0 inventory digest differs from controller binding")
    if [checkpoint["name"] for checkpoint in binding["checkpoints"]] != list(
        CHECKPOINTS
    ):
        raise VerificationError("input-binding checkpoint order mismatch")
    if binding["inotify_mask"] != list(INOTIFY_NAMES):
        raise VerificationError("mutation observation mask mismatch")
    checkpoints = binding["checkpoints"]
    states = [checkpoint["state"] for checkpoint in checkpoints]
    completed_count = 0
    failed_count = 0
    terminal_seen = False
    for checkpoint in checkpoints:
        state = checkpoint["state"]
        if state == "completed":
            if terminal_seen:
                raise VerificationError(
                    "completed checkpoint follows a failed/not-reached checkpoint"
                )
            completed_count += 1
            if (
                checkpoint["digest"] != controller_digest
                or checkpoint["inventory_equal"] is not True
                or checkpoint["events_before"]
                or checkpoint["events_after"]
            ):
                raise VerificationError(
                    "bound input changed: completed checkpoint is not an exact clean T0 match"
                )
        elif state == "failed":
            if terminal_seen:
                raise VerificationError(
                    "input binding has multiple or out-of-order failed checkpoints"
                )
            terminal_seen = True
            failed_count += 1
            if (
                not new_format
                and not checkpoint["events_before"]
                and not checkpoint["events_after"]
            ):
                raise VerificationError(
                    "failed checkpoint has no mechanical mutation evidence"
                )
        else:
            terminal_seen = True
    if failed_count > 1:
        raise VerificationError("input binding has multiple failed checkpoints")

    # Packets emitted before the corrective failure envelope omit `failure`.
    # Preserve their schema and verifier behaviour verbatim.  New packets bind
    # their terminal failure and checkpoint operation state machine together.
    if new_format:
        failure = binding["failure"]
        if failure is None:
            if evidence["result"] == "provenance_invalid":
                raise VerificationError(
                    "new-format provenance_invalid lacks its failure record"
                )
            if failed_count:
                raise VerificationError(
                    "failed checkpoint lacks a checkpoint failure stage"
                )
        else:
            if (
                not isinstance(failure, Mapping)
                or set(failure) != {"reason", "stage"}
                or not all(
                    isinstance(failure[key], str) and failure[key]
                    for key in ("reason", "stage")
                )
            ):
                raise VerificationError("input-binding failure record is malformed")
            if evidence["result"] == "phase_a_pass":
                raise VerificationError("phase_a_pass carries a failure record")
            stage = failure["stage"]
            allowed_stages = {
                "preflight",
                "build",
                "build_binding",
                "extension_load",
                "runs",
                "comparison",
            } | {f"checkpoint_{name}" for name in CHECKPOINTS}
            if stage not in allowed_stages:
                raise VerificationError("failure stage is not a controller stage")
            failed_rows = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint["state"] == "failed"
            ]
            if stage.startswith("checkpoint_"):
                if evidence["result"] != "provenance_invalid":
                    raise VerificationError(
                        "checkpoint failure did not select provenance_invalid"
                    )
                named = stage.removeprefix("checkpoint_")
                if len(failed_rows) != 1 or failed_rows[0]["name"] != named:
                    raise VerificationError(
                        "checkpoint failure does not match its exact failed checkpoint row"
                    )
                row = failed_rows[0]
                required_failure_fields = {
                    "inventory_comparison_executed",
                    "observed_digest",
                }
                if not required_failure_fields.issubset(row):
                    raise VerificationError(
                        "new-format failed checkpoint lacks its operation record"
                    )
                cause = row.get("cause")
                legal_causes = {
                    "events_before",
                    "recompute_failed",
                    "events_after",
                    "inventory_mismatch",
                }
                if cause not in legal_causes:
                    raise VerificationError(
                        "new-format failed checkpoint lacks a recognized cause"
                    )
                compared = row["inventory_comparison_executed"]
                observed = row["observed_digest"]
                before = row["events_before"]
                after = row["events_after"]
                mutations = binding["mutation_events"]
                if cause == "events_before":
                    valid = (
                        bool(before)
                        and not after
                        and not compared
                        and row["inventory_equal"] is None
                        and observed is None
                        and mutations == before
                    )
                elif cause == "recompute_failed":
                    valid = (
                        not before
                        and not after
                        and not compared
                        and row["inventory_equal"] is None
                        and observed is None
                        and not mutations
                    )
                elif cause == "events_after":
                    valid = (
                        not before
                        and bool(after)
                        and not compared
                        and row["inventory_equal"] is None
                        and (observed is None or isinstance(observed, str))
                        and mutations == after
                    )
                else:
                    valid = (
                        not before
                        and not after
                        and compared
                        and row["inventory_equal"] is False
                        and isinstance(observed, str)
                        and observed != controller_digest
                        and not mutations
                    )
                if not valid:
                    raise VerificationError(
                        "checkpoint failure cause disagrees with its operation record"
                    )
            else:
                if failed_rows:
                    raise VerificationError(
                        "non-checkpoint failure stage fabricates a failed checkpoint"
                    )
                if any(
                    checkpoint["state"] != "not_reached"
                    for checkpoint in checkpoints[completed_count:]
                ):
                    raise VerificationError(
                        "non-checkpoint failure did not leave later checkpoints not_reached"
                    )
    mutations = binding["mutation_events"]
    if mutations:
        if (
            evidence["result"] != "provenance_invalid"
            or binding["final_equal"] is not False
        ):
            raise VerificationError(
                "observed bound-input mutation did not select provenance_invalid"
            )
        classifications = {event["classification"] for event in mutations}
        expected_monitor = (
            "queue_overflow"
            if "queue_overflow" in classifications
            else "ignored_watch"
            if "ignored_watch" in classifications
            else "watch_failed"
            if "watch_failed" in classifications
            else "drift"
        )
        if binding["monitor_state"] != expected_monitor:
            raise VerificationError(
                "mutation classification and monitor state disagree"
            )
    elif failed_count:
        if (
            evidence["result"] != "provenance_invalid"
            or binding["monitor_state"] != "closed_clean"
            or binding["final_equal"] is not False
        ):
            raise VerificationError(
                "failed inventory comparison has an inconsistent clean-monitor state"
            )
    elif completed_count == 0:
        if (
            evidence["result"] != "provenance_invalid"
            or states != ["not_reached"] * len(CHECKPOINTS)
            or binding["monitor_state"] != "not_started"
            or binding["final_equal"] is not None
        ):
            raise VerificationError(
                "unstarted monitor state is not a preflight provenance rejection"
            )
    else:
        if binding["monitor_state"] != "closed_clean":
            raise VerificationError(
                "clean checkpoint prefix has an inconsistent monitor state"
            )
        expected_final = True if completed_count == len(CHECKPOINTS) else None
        if binding["final_equal"] is not expected_final:
            raise VerificationError(
                "input-binding final equality state is not mechanically derived"
            )


def _verify_result(evidence: Mapping[str, Any]) -> None:
    result = evidence["result"]
    predicates = evidence["decision_predicates"]
    if result != select_result(predicates):
        raise VerificationError(
            "result enum disagrees with first-applicable decision predicates"
        )
    matrix = expected_matrix()
    if evidence["result_matrix"] != matrix:
        raise VerificationError("evidence C/D/V matrix mismatch")
    required = matrix[result]["required"]
    if evidence["artifact_inventory"] != required:
        raise VerificationError(
            "artifact inventory does not exactly equal the result row"
        )
    policy_state = evidence["policy_comparison"]
    packet_states = evidence["packet_verification_states"]
    child_states = [child["state"] for child in evidence["child_invocations"]]
    if result in {"build_failed", "extension_load_failed"}:
        if child_states != ["not_run"] * 4:
            raise VerificationError(
                f"{result} has a child that should not have launched"
            )
    if result == "provenance_invalid":
        first_non_completed = next(
            (index for index, state in enumerate(child_states) if state != "completed"),
            len(child_states),
        )
        tail = child_states[first_non_completed:]
        if tail:
            if tail[0] == "failed":
                failed = evidence["child_invocations"][first_non_completed]
                if failed["result"] != "provenance_invalid" or tail[1:] != [
                    "not_run"
                ] * (len(tail) - 1):
                    raise VerificationError(
                        "provenance failure child state order/result mismatch"
                    )
            elif tail != ["not_run"] * len(tail):
                raise VerificationError("provenance failure child state order mismatch")
    if result in {"runner_nonzero", "runner_timeout"}:
        first_non_completed = next(
            (index for index, state in enumerate(child_states) if state != "completed"),
            len(child_states),
        )
        tail = child_states[first_non_completed:]
        controller_timeout = result == "runner_timeout" and tail == ["not_run"] * len(
            tail
        )
        if not controller_timeout:
            if not tail or tail[1:] != ["not_run"] * (len(tail) - 1):
                raise VerificationError(
                    "runner failure child state order is not fail-closed"
                )
            failed_child = evidence["child_invocations"][first_non_completed]
            expected_child_result = (
                "runner_timeout" if result == "runner_timeout" else "runner_nonzero"
            )
            if (
                failed_child["state"] != "failed"
                or failed_child["result"] != expected_child_result
            ):
                raise VerificationError(
                    "runner failure child result does not match controller result"
                )
    if result == "capture_perturbs_policy":
        if policy_state != "unequal" or packet_states != ["not_produced"] * 3:
            raise VerificationError("capture_perturbs_policy evidence-state mismatch")
    elif result == "packet_invalid":
        if (
            policy_state != "equal"
            or "fail" not in packet_states
            or "not_produced" in packet_states
        ):
            raise VerificationError("packet_invalid evidence-state mismatch")
    elif result == "phase_a_pass":
        if policy_state != "equal" or packet_states != ["pass"] * 3:
            raise VerificationError("phase_a_pass evidence-state mismatch")
        if any(
            child["state"] != "completed" or child["result"] != "run_completed"
            for child in evidence["child_invocations"]
        ):
            raise VerificationError("phase_a_pass has a non-completed child")
    else:
        if policy_state != "not_produced" or packet_states != ["not_produced"] * 3:
            raise VerificationError("C-only result fabricated D/V decision evidence")


def verify_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild every controller decision; never consume a self-reported valid bit."""
    _schema_validate(evidence)
    if evidence.get("document_type") != "execution_evidence":
        raise VerificationError("not an execution_evidence document")
    _verify_constants(evidence["controller_input"])
    _verify_bound_inputs(evidence["controller_input"])
    _verify_children(evidence)
    _verify_input_binding(evidence)
    _verify_result(evidence)
    return {
        "document_type": "aggregate_verification",
        "result": evidence["result"],
        "schema": "h0_phase_a_verifier_v1",
        "valid": True,
    }


def _verify_policy_inventory(run_id: str, inventory: Mapping[str, Any]) -> None:
    members = {
        "active_tid_slot_pairs",
        "final_track_rows",
        "mot_output",
        "overflow_vector",
        "proposal_projection",
        "relink_debug_raw",
        "schema",
        "winner_commit_projection",
    }
    if (
        set(inventory) != members
        or inventory.get("schema") != "h0_phase_a_policy_inventory_v1"
    ):
        raise VerificationError(f"policy inventory schema mismatch: {run_id}")

    def exact_int(value: object) -> bool:
        return type(value) is int

    mot = inventory["mot_output"]
    if (
        not isinstance(mot, dict)
        or set(mot) != {"length", "sha256"}
        or not exact_int(mot["length"])
        or mot["length"] < 0
        or not isinstance(mot["sha256"], str)
        or len(mot["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in mot["sha256"])
    ):
        raise VerificationError(f"policy MOT identity shape mismatch: {run_id}")
    final_rows = inventory["final_track_rows"]
    active_rows = inventory["active_tid_slot_pairs"]
    if not isinstance(final_rows, list) or not isinstance(active_rows, list):
        raise VerificationError(f"policy inventory row collection mismatch: {run_id}")
    for row in final_rows:
        if not isinstance(row, dict) or set(row) != {
            "binary32_bits",
            "class",
            "frame",
            "row_index",
            "track_id",
        }:
            raise VerificationError(f"final-track row shape mismatch: {run_id}")
        bits = row["binary32_bits"]
        if (
            not isinstance(bits, list)
            or len(bits) != 5
            or any(
                not exact_int(value) or value < 0 or value > 0xFFFFFFFF
                for value in bits
            )
            or any(
                not exact_int(row[key])
                for key in ("class", "frame", "row_index", "track_id")
            )
            or row["frame"] < 1
            or row["row_index"] < 0
        ):
            raise VerificationError(f"final-track row value mismatch: {run_id}")
    positions: dict[int, list[int]] = {}
    for row in final_rows:
        positions.setdefault(row["frame"], []).append(row["row_index"])
    if any(values != list(range(len(values))) for values in positions.values()):
        raise VerificationError(
            f"final-track row positions are not emitted order: {run_id}"
        )
    for row in active_rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"frame", "pairs"}
            or not exact_int(row["frame"])
            or row["frame"] < 1
        ):
            raise VerificationError(f"active tid/slot row shape mismatch: {run_id}")
        pairs = row["pairs"]
        if (
            not isinstance(pairs, list)
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not exact_int(value) for value in pair)
                for pair in pairs
            )
            or pairs != sorted(pairs, key=lambda pair: pair[1])
            or len({pair[1] for pair in pairs}) != len(pairs)
        ):
            raise VerificationError(f"active tid/slot pairs mismatch: {run_id}")
    for member, length in (("relink_debug_raw", 13), ("overflow_vector", 9)):
        vector = inventory[member]
        if (
            not isinstance(vector, list)
            or len(vector) != length
            or any(not exact_int(value) for value in vector)
        ):
            raise VerificationError(
                f"policy inventory vector mismatch: {run_id}:{member}"
            )

    def projection(value: object, *, winner: bool) -> None:
        if not isinstance(value, dict) or set(value) != {"count", "digest", "records"}:
            raise VerificationError(f"trace projection shape mismatch: {run_id}")
        record_keys = (
            {"commits", "winning_claims"} if winner else {"candidates", "claims"}
        )
        records = value["records"]
        if (
            not isinstance(records, dict)
            or set(records) != record_keys
            or any(not isinstance(records[key], list) for key in record_keys)
        ):
            raise VerificationError(f"trace projection records mismatch: {run_id}")
        primary = records["commits" if winner else "candidates"]
        if (
            not exact_int(value["count"])
            or value["count"] != len(primary)
            or value["digest"] != digest(records)
        ):
            raise VerificationError(f"trace projection count/digest mismatch: {run_id}")

    if run_id == RUN_IDS[0]:
        if (
            inventory["proposal_projection"] is not None
            or inventory["winner_commit_projection"] is not None
        ):
            raise VerificationError("capture-off fabricated trace-only projections")
    else:
        projection(inventory["proposal_projection"], winner=False)
        projection(inventory["winner_commit_projection"], winner=True)


def _verify_complete_build_identity(
    identity: Mapping[str, Any], controller: Mapping[str, Any]
) -> None:
    repository_root = Path(controller["repository_root"])
    base_members = {
        "artifacts",
        "build_environment",
        "build_environment_digest",
        "build_vectors",
        "cmake",
        "cmake_cache_sha256",
        "compilers",
        "cuda_toolkit_root",
        "python",
        "python_ext_suffix",
        "state",
        "uv_lock_sha256",
    }
    if (
        set(identity) not in (base_members, base_members | {"extension_load"})
        or identity.get("state") != "complete"
    ):
        raise VerificationError("build identity has missing or unknown members")
    if identity["build_vectors"] != [list(vector) for vector in BUILD_VECTORS]:
        raise VerificationError("build identity command vectors differ from A7.4")
    expected_environment = _expected_build_environment(controller)
    if identity["build_environment"] != expected_environment or identity[
        "build_environment_digest"
    ] != digest(expected_environment):
        raise VerificationError("build identity environment table/digest mismatch")
    suffix = identity["python_ext_suffix"]
    if not isinstance(suffix, str) or not suffix or "/" in suffix or "\\" in suffix:
        raise VerificationError("build identity EXT_SUFFIX is non-canonical")
    expected_artifacts = [
        f"build/h0_phase_a/saccade_tracking_ext{suffix}",
        "build/h0_phase_a/libsaccade_scan_plugin.so",
    ]
    artifacts = identity["artifacts"]
    if (
        not isinstance(artifacts, list)
        or any(not isinstance(item, dict) for item in artifacts)
        or [item.get("path") for item in artifacts] != expected_artifacts
    ):
        raise VerificationError("build identity artifact cardinality/path mismatch")

    def sha(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and not any(char not in "0123456789abcdef" for char in value)
        )

    for artifact in artifacts:
        if set(artifact) != {
            "dynamic_dependencies",
            "elf_gnu_build_id",
            "length",
            "path",
            "sha256",
        }:
            raise VerificationError(
                "build artifact identity has missing or unknown members"
            )
        if (
            type(artifact["length"]) is not int
            or artifact["length"] < 0
            or not sha(artifact["sha256"])
        ):
            raise VerificationError("build artifact length/hash is malformed")
        build_id = artifact["elf_gnu_build_id"]
        if (
            not isinstance(build_id, str)
            or not build_id
            or any(char not in "0123456789abcdef" for char in build_id)
        ):
            raise VerificationError("ELF GNU build ID is malformed")
        dependencies = artifact["dynamic_dependencies"]
        if not isinstance(dependencies, list):
            raise VerificationError("dynamic dependency inventory is not an array")
        dependency_paths: list[str] = []
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {
                "length",
                "path",
                "realpath",
                "sha256",
            }:
                raise VerificationError(
                    "dynamic dependency has missing or unknown members"
                )
            if (
                type(dependency["length"]) is not int
                or dependency["length"] < 0
                or not sha(dependency["sha256"])
            ):
                raise VerificationError("dynamic dependency length/hash is malformed")
            if not str(dependency["path"]).startswith("/") or not str(
                dependency["realpath"]
            ).startswith("/"):
                raise VerificationError("dynamic dependency path is not absolute")
            dependency_paths.append(dependency["path"])
        if dependency_paths != sorted(
            dependency_paths, key=lambda value: value.encode("utf-8")
        ) or len(dependency_paths) != len(set(dependency_paths)):
            raise VerificationError("dynamic dependency order/uniqueness mismatch")
    cmake = identity["cmake"]
    if not isinstance(cmake, dict) or set(cmake) != {
        "generator",
        "length",
        "path",
        "sha256",
        "version",
    }:
        raise VerificationError("CMake identity has missing or unknown members")
    compilers = identity["compilers"]
    if not isinstance(compilers, dict) or set(compilers) != {"cxx", "cuda"}:
        raise VerificationError("compiler identity set mismatch")
    if _build_environment_algorithm(controller) == "h0_build_environment_v2":
        # Record-to-record binding: tool_paths.nvcc <-> unique frozen
        # tool_runtime record <-> build_environment.CUDACXX (checked via the
        # environment table above) <-> build_identity.compilers.cuda.
        frozen_nvcc = _bound_nvcc_record(controller)
        cuda = compilers["cuda"]
        if not isinstance(cuda, dict) or (
            cuda.get("path"),
            cuda.get("length"),
            cuda.get("sha256"),
        ) != (
            frozen_nvcc["realpath"],
            frozen_nvcc["length"],
            frozen_nvcc["sha256"],
        ):
            raise VerificationError(
                "CUDA compiler identity differs from the frozen nvcc record"
            )
    python_identity = identity["python"]
    if not isinstance(python_identity, dict) or set(python_identity) != {
        "abi",
        "length",
        "path",
        "sha256",
        "version",
    }:
        raise VerificationError("Python identity has missing or unknown members")
    for record in [cmake, python_identity, *compilers.values()]:
        if not isinstance(record, dict):
            raise VerificationError("tool/Python identity is not an object")
        if (
            record is not cmake
            and record is not python_identity
            and set(record) != {"length", "path", "sha256", "version"}
        ):
            raise VerificationError(
                "tool/Python identity has missing or unknown members"
            )
        if (
            type(record["length"]) is not int
            or record["length"] < 0
            or not sha(record["sha256"])
        ):
            raise VerificationError("tool/Python identity length/hash is malformed")
        if not isinstance(record["path"], str) or not record["path"].startswith("/"):
            raise VerificationError("tool/Python identity path is not absolute")
    if not sha(identity["cmake_cache_sha256"]) or not sha(identity["uv_lock_sha256"]):
        raise VerificationError("build input hash is malformed")
    if not isinstance(identity["cuda_toolkit_root"], str) or not identity[
        "cuda_toolkit_root"
    ].startswith("/"):
        raise VerificationError("CUDA toolkit root is not absolute")
    # Archive verification is host-independent: build artifacts and the CMake
    # cache are execution-host products that are not part of the packet, so
    # their identities are admitted as recorded; uv.lock is bound at I, so its
    # recorded hash must equal the frozen repository record.
    uv_lock_records = [
        record
        for record in controller["bound_inputs"]["repository"]
        if record["path"] == "uv.lock"
    ]
    if (
        len(uv_lock_records) != 1
        or identity["uv_lock_sha256"] != uv_lock_records[0]["sha256"]
    ):
        raise VerificationError("uv.lock identity mismatch")
    extension_load = identity.get("extension_load")
    if extension_load is None:
        return
    extension_members = {
        "confinement_plan_digest",
        "confinement_probe_passed",
        "environment",
        "environment_digest",
        "result",
        "returncode",
        "runtime_inputs",
        "runtime_inputs_digest",
        "state",
        "vector",
    }
    if (
        not isinstance(extension_load, dict)
        or set(extension_load) != extension_members
        or extension_load["state"] not in {"complete", "failed", "rejected"}
        or type(extension_load["returncode"]) is not int
    ):
        raise VerificationError("extension-load confinement record is malformed")
    expected_state = {
        "complete": ("extension_loaded", 0),
        "failed": ("extension_load_failed", None),
        "rejected": ("provenance_invalid", None),
    }[extension_load["state"]]
    if extension_load["result"] != expected_state[0] or (
        expected_state[1] is not None
        and extension_load["returncode"] != expected_state[1]
    ):
        raise VerificationError("extension-load state/result mismatch")
    if extension_load["state"] == "failed" and extension_load["returncode"] == 0:
        raise VerificationError("failed extension-load has a zero return code")
    if (
        extension_load["state"] in {"complete", "failed"}
        and extension_load["confinement_probe_passed"] is not True
    ):
        raise VerificationError("extension-load confinement probe did not pass")
    expected_environment = _expected_extension_load_environment(controller)
    if extension_load["environment"] != expected_environment or extension_load[
        "environment_digest"
    ] != digest(expected_environment):
        raise VerificationError("extension-load environment table/digest mismatch")
    if extension_load["vector"] != _expected_extension_load_vector(
        controller, identity
    ):
        raise VerificationError("extension-load vector mismatch")
    _verify_runtime_inputs(
        extension_load["runtime_inputs"],
        extension_load,
        controller,
        identity,
        expected_output_directories=(
            repository_root / controller["incomplete_root"] / "_extension_load",
        ),
    )
    observed = {
        record["realpath"]
        for record in extension_load["runtime_inputs"]["regular_files"]
    }
    expected_loaded = {
        (repository_root / artifact["path"]).as_posix() for artifact in artifacts
    }
    if not expected_loaded.issubset(observed):
        raise VerificationError(
            "extension/plugin load is absent from runtime attestation"
        )


def _verify_runtime_inputs(
    value: Mapping[str, Any],
    invocation: Mapping[str, Any],
    controller: Mapping[str, Any],
    build_identity: Mapping[str, Any],
    *,
    expected_output_directories: Sequence[Path] | None = None,
) -> None:
    members = {
        "backend",
        "confinement_plan",
        "confinement_plan_digest",
        "denial_probe_observed",
        "ingress_policy",
        "installed_before_exec",
        "landlock_abi",
        "process_tree_terminal",
        "regular_files",
        "resources",
        "schema",
        "state",
        "trace_scope",
        "violations",
    }
    if set(value) != members or value.get("schema") != "h0_runtime_inputs_v1":
        raise VerificationError("runtime input attestation shape mismatch")
    expected_scope = ["execve", "execveat", "mmap", "open", "openat", "openat2"]
    if (
        value["backend"] != "landlock_seccomp_ptrace_v1"
        or value["ingress_policy"] != "deny_external_bytes_v1"
        or value["trace_scope"] != expected_scope
        or value["installed_before_exec"] is not True
        or value["process_tree_terminal"] is not True
        or not isinstance(value["landlock_abi"], int)
        or value["landlock_abi"] < 3
    ):
        raise VerificationError("runtime OS boundary declaration mismatch")
    plan = value["confinement_plan"]
    if not isinstance(plan, dict) or value["confinement_plan_digest"] != digest(plan):
        raise VerificationError("runtime confinement plan digest mismatch")
    if invocation["confinement_plan_digest"] != value["confinement_plan_digest"]:
        raise VerificationError("parent/child runtime confinement plan mismatch")
    if invocation["runtime_inputs_digest"] != digest(value):
        raise VerificationError("child/runtime input inventory digest mismatch")
    plan_members = {
        "backend",
        "blocked_ingress_syscalls",
        "denial_probe",
        "files",
        "ingress_policy",
        "kernel_resources",
        "lookup_directories",
        "output_directories",
        "resource_rules",
        "schema",
        "trace_scope",
    }
    if (
        set(plan) != plan_members
        or plan["schema"] != "h0_runtime_confinement_plan_v1"
        or plan["backend"] != value["backend"]
        or plan["blocked_ingress_syscalls"] != list(BLOCKED_RUNTIME_INGRESS_SYSCALLS)
        or plan["ingress_policy"] != value["ingress_policy"]
        or plan["kernel_resources"] != ["exec_auxv", "getrandom"]
        or plan["trace_scope"] != expected_scope
    ):
        raise VerificationError("runtime confinement plan shape mismatch")

    repository_root = Path(controller["repository_root"])
    incomplete = repository_root / controller["incomplete_root"]
    expected_outputs = [
        path.as_posix()
        for path in (
            expected_output_directories
            if expected_output_directories is not None
            else tuple(incomplete / "runs" / run_id for run_id in RUN_IDS)
        )
    ]
    if (
        plan["denial_probe"]
        != (incomplete / "_runtime_confinement_denial_probe").as_posix()
        or plan["output_directories"] != expected_outputs
    ):
        raise VerificationError("runtime output/probe confinement plan mismatch")

    expected: dict[str, dict[str, Any]] = {}

    def admit(
        realpath: str,
        *,
        binding: str,
        logical_paths: Sequence[str],
        length: int,
        sha256: str,
    ) -> None:
        current = expected.setdefault(
            realpath,
            {
                "bindings": set(),
                "length": length,
                "logical_paths": set(),
                "sha256": sha256,
            },
        )
        if (current["length"], current["sha256"]) != (length, sha256):
            raise VerificationError("conflicting frozen runtime file identity")
        current["bindings"].add(binding)
        current["logical_paths"].update(logical_paths)
        current["logical_paths"].add(realpath)

    inventory = controller["bound_inputs"]
    for record in inventory["repository"]:
        if record["kind"] == "regular":
            path = (repository_root / record["path"]).as_posix()
            admit(
                path,
                binding="repository",
                logical_paths=(path,),
                length=record["length"],
                sha256=record["sha256"],
            )
    sequence_root = repository_root / inventory["sequence"]["root"]
    for record in inventory["sequence"]["files"]:
        path = (sequence_root / record["path"]).as_posix()
        admit(
            path,
            binding="sequence",
            logical_paths=(path,),
            length=record["length"],
            sha256=record["sha256"],
        )
    for binding in ("models_engines", "tool_runtime"):
        for record in inventory[binding]:
            logical = Path(record["logical_path"])
            if not logical.is_absolute():
                logical = repository_root / logical
            admit(
                record["realpath"],
                binding=binding,
                logical_paths=(logical.as_posix(), record["realpath"]),
                length=record["length"],
                sha256=record["sha256"],
            )
    for record in build_identity["artifacts"]:
        path = (repository_root / record["path"]).as_posix()
        admit(
            path,
            binding="build_artifact",
            logical_paths=(path,),
            length=record["length"],
            sha256=record["sha256"],
        )
        for dependency in record["dynamic_dependencies"]:
            admit(
                dependency["realpath"],
                binding="build_runtime_closure",
                logical_paths=(dependency["path"], dependency["realpath"]),
                length=dependency["length"],
                sha256=dependency["sha256"],
            )
    python_identity = build_identity["python"]
    admit(
        python_identity["path"],
        binding="tool_runtime",
        logical_paths=(
            (repository_root / ".venv/bin/python").as_posix(),
            python_identity["path"],
        ),
        length=python_identity["length"],
        sha256=python_identity["sha256"],
    )
    plan_files = plan["files"]
    plan_file_members = {
        "bindings",
        "executable",
        "length",
        "logical_paths",
        "realpath",
        "sha256",
    }
    if not isinstance(plan_files, list) or any(
        not isinstance(record, dict) or set(record) != plan_file_members
        for record in plan_files
    ):
        raise VerificationError("runtime confinement file record shape mismatch")
    if [record["realpath"] for record in plan_files] != sorted(
        expected, key=lambda path: path.encode("utf-8")
    ):
        raise VerificationError("runtime confinement file universe mismatch")
    for record in plan_files:
        frozen = expected[record["realpath"]]
        if (
            record["bindings"] != sorted(frozen["bindings"])
            or record["length"] != frozen["length"]
            or record["logical_paths"]
            != sorted(frozen["logical_paths"], key=lambda path: path.encode("utf-8"))
            or record["sha256"] != frozen["sha256"]
            or not isinstance(record["executable"], bool)
        ):
            raise VerificationError("runtime confinement identity differs from binding")
    expected_lookup = {
        str(Path(path).parent) for item in plan_files for path in item["logical_paths"]
    } | {
        parent.as_posix()
        for item in plan_files
        if "tool_runtime" in item["bindings"]
        for path in item["logical_paths"]
        for parent in Path(path).parents
        if parent.as_posix() != "/"
    }
    python_library_lookup = (
        Path(python_identity["path"]).parent.parent / "lib"
    ).as_posix()
    observed_lookup = set(plan["lookup_directories"])
    if observed_lookup not in (
        expected_lookup,
        expected_lookup | {python_library_lookup},
    ) or plan["lookup_directories"] != sorted(
        observed_lookup, key=lambda path: path.encode("utf-8")
    ):
        raise VerificationError("runtime lookup-directory plan mismatch")
    resource_rules = plan["resource_rules"]
    if not isinstance(resource_rules, list) or any(
        not isinstance(resource, dict) or set(resource) != {"kind", "path"}
        for resource in resource_rules
    ):
        raise VerificationError("runtime resource-rule shape mismatch")
    if [resource["path"] for resource in resource_rules] != sorted(
        {resource["path"] for resource in resource_rules},
        key=lambda path: path.encode("utf-8"),
    ):
        raise VerificationError("runtime resource-rule order/uniqueness mismatch")
    for resource in resource_rules:
        if not (
            resource == {"kind": "procfs", "path": "/proc"}
            or resource == {"kind": "sysfs", "path": "/sys"}
            or (
                resource["kind"] == "device"
                and (
                    resource["path"] in {"/dev/null", "/dev/zero", "/dev/urandom"}
                    or resource["path"].startswith("/dev/nvidia")
                    or resource["path"].startswith("/dev/dri/renderD")
                )
            )
        ):
            raise VerificationError("unclassified runtime resource rule")

    if value["state"] == "complete":
        if (
            value["violations"]
            or value["denial_probe_observed"] is not True
            or invocation["confinement_probe_passed"] is not True
        ):
            raise VerificationError("complete runtime attestation has a violation")
    elif value["state"] == "rejected":
        if not value["violations"] or invocation["result"] != "provenance_invalid":
            raise VerificationError("rejected runtime access did not fail provenance")
    else:
        raise VerificationError("unknown runtime attestation state")
    observed = value["regular_files"]
    observed_members = {
        "bindings",
        "length",
        "logical_paths",
        "operations",
        "realpath",
        "roles",
        "sha256",
    }
    if not isinstance(observed, list) or any(
        not isinstance(record, dict) or set(record) != observed_members
        for record in observed
    ):
        raise VerificationError("actual runtime file record shape mismatch")
    if [record["realpath"] for record in observed] != sorted(
        {record["realpath"] for record in observed},
        key=lambda path: path.encode("utf-8"),
    ):
        raise VerificationError(
            "actual runtime file inventory order/uniqueness mismatch"
        )
    plan_by_path = {record["realpath"]: record for record in plan_files}
    for record in observed:
        if record["realpath"] not in plan_by_path:
            raise VerificationError("actual runtime file is unbound or malformed")
        admitted = plan_by_path[record["realpath"]]
        if (
            record["bindings"] != admitted["bindings"]
            or record["length"] != admitted["length"]
            or record["logical_paths"] != admitted["logical_paths"]
            or record["sha256"] != admitted["sha256"]
            or not record["operations"]
            or any(
                operation
                not in {
                    "execve",
                    "execveat",
                    "mmap",
                    "mmap_exec",
                    "mmap_read",
                    "open",
                    "openat",
                    "openat2",
                    "startup_mapping",
                }
                for operation in record["operations"]
            )
        ):
            raise VerificationError("actual runtime file identity/operation mismatch")
        roles = set(record["bindings"])
        suffixes = Path(record["realpath"]).suffixes
        if any(suffix in {".py", ".pyc"} for suffix in suffixes):
            roles.add("python_module")
        if ".so" in suffixes or ".so." in Path(record["realpath"]).name:
            roles.add("shared_library")
        if any(
            operation in {"execve", "execveat"} for operation in record["operations"]
        ):
            roles.add("interpreter_or_executable")
        if record["roles"] != sorted(roles):
            raise VerificationError("actual runtime file role classification mismatch")
    resources = value["resources"]
    allowed_operations = {
        "execve",
        "execveat",
        "getrandom",
        "mmap",
        "mmap_exec",
        "mmap_read",
        "open",
        "openat",
        "openat2",
        "startup_mapping",
    }
    if not isinstance(resources, list) or any(
        not isinstance(resource, dict)
        or set(resource) != {"kind", "operations", "path"}
        or not isinstance(resource["path"], str)
        or not resource["operations"]
        or resource["operations"] != sorted(set(resource["operations"]))
        or any(
            operation not in allowed_operations for operation in resource["operations"]
        )
        for resource in resources
    ):
        raise VerificationError("runtime non-file resource inventory malformed")
    if [(resource["path"], resource["kind"]) for resource in resources] != sorted(
        {(resource["path"], resource["kind"]) for resource in resources},
        key=lambda item: (item[0].encode("utf-8"), item[1]),
    ):
        raise VerificationError("runtime resource inventory order/uniqueness mismatch")
    output_roots = tuple(Path(path) for path in plan["output_directories"])
    lookup_roots = tuple(Path(path) for path in plan["lookup_directories"])
    resource_roots = {
        (resource["kind"], resource["path"]) for resource in resource_rules
    }
    for resource in resources:
        kind = resource["kind"]
        path = resource["path"]
        candidate = Path(path)
        canonical = candidate.is_absolute() and not any(
            part in {".", ".."} for part in candidate.parts
        )
        if kind == "run_output":
            admitted = canonical and any(
                candidate == root or root in candidate.parents for root in output_roots
            )
        elif kind in {"procfs", "sysfs"}:
            root = Path("/proc" if kind == "procfs" else "/sys")
            admitted = (
                (kind, root.as_posix()) in resource_roots
                and canonical
                and (candidate == root or root in candidate.parents)
            )
        elif kind == "device":
            admitted = (kind, path) in resource_roots
        elif kind == "bound_directory":
            admitted = canonical and any(
                candidate == root or root in candidate.parents for root in lookup_roots
            )
        elif kind == "kernel_random":
            admitted = (
                path == "syscall:getrandom" and "getrandom" in plan["kernel_resources"]
            )
        elif kind == "kernel_auxv":
            admitted = (
                path == "kernel:exec_auxv" and "exec_auxv" in plan["kernel_resources"]
            )
        else:
            admitted = False
        if not admitted:
            raise VerificationError("runtime resource is absent from confinement plan")
    violations = value["violations"]
    if not isinstance(violations, list) or any(
        not isinstance(violation, dict)
        or set(violation) != {"operation", "path", "reason"}
        or not all(isinstance(member, str) for member in violation.values())
        for violation in violations
    ):
        raise VerificationError("runtime violation inventory malformed")
    if value["state"] == "complete" and not any(
        "interpreter_or_executable" in record["roles"] for record in observed
    ):
        raise VerificationError("interpreter startup is absent from runtime inventory")
    if value["state"] == "complete" and not any(
        "startup_mapping" in record["operations"] for record in observed
    ):
        raise VerificationError("kernel-created startup mappings are absent")


def verify_evidence_root(root: Path) -> dict[str, Any]:
    """Verify a complete staged or published evidence filesystem."""
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root.absolute()
    ):
        raise VerificationError("evidence root is absent, symlinked, or non-physical")
    actual_files: list[str] = []
    actual_directories: set[str] = set()
    pending_directories = [root]
    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as exc:
            raise VerificationError(
                f"evidence directory is unreadable: {directory.relative_to(root)}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise VerificationError(
                    f"evidence entry is unclassifiable: {relative}"
                ) from exc
            if stat.S_ISREG(info.st_mode):
                actual_files.append(relative)
            elif stat.S_ISDIR(info.st_mode):
                actual_directories.add(relative)
                pending_directories.append(path)
            else:
                raise VerificationError(f"forbidden evidence entry type: {relative}")
    actual_file_set = set(actual_files)
    required_control_files = {"checksums.sha256", "manifest.json"}
    if not required_control_files.issubset(actual_file_set):
        raise VerificationError(
            "manifest.json or checksums.sha256 is absent or non-regular"
        )
    manifest = load_json(root / "manifest.json", canonical_file=True)
    report = verify_evidence(manifest)
    expected_files = manifest["artifact_inventory"]
    if sorted(actual_files, key=lambda value: value.encode("utf-8")) != sorted(
        expected_files, key=lambda value: value.encode("utf-8")
    ):
        raise VerificationError("evidence regular-file universe differs from C/D/V row")
    expected_directories: set[str] = set()
    for relative in expected_files:
        current = Path(relative).parent
        while current.as_posix() != ".":
            expected_directories.add(current.as_posix())
            current = current.parent
    if actual_directories != expected_directories:
        raise VerificationError(
            "evidence directory universe has missing or unknown entries"
        )
    checksum_path = root / "checksums.sha256"
    try:
        checksum_bytes = checksum_path.read_bytes()
        checksum_text = checksum_bytes.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError(f"checksums.sha256 unreadable: {exc}") from exc
    expected_checksum_paths = sorted(
        (path for path in expected_files if path != "checksums.sha256"),
        key=lambda value: value.encode("utf-8"),
    )
    lines = checksum_text.splitlines(keepends=True)
    if len(lines) != len(expected_checksum_paths) or any(
        not line.endswith("\n") for line in lines
    ):
        raise VerificationError("checksums.sha256 cardinality/line ending mismatch")
    for index, line in enumerate(lines):
        raw = line[:-1]
        if len(raw) < 67 or raw[64:66] != "  ":
            raise VerificationError("malformed checksum line")
        hash_value, relative = raw[:64], raw[66:]
        if any(char not in "0123456789abcdef" for char in hash_value):
            raise VerificationError("non-lowercase SHA-256 in checksum line")
        pure = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or "\x00" in relative
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != relative
        ):
            raise VerificationError("non-canonical relative checksum path")
        if relative != expected_checksum_paths[index]:
            raise VerificationError("checksum path order/inventory mismatch")
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != hash_value:
            raise VerificationError(f"checksum mismatch: {relative}")
    for run_id, embedded in zip(RUN_IDS, manifest["child_invocations"], strict=True):
        actual = load_json(
            root / "runs" / run_id / "invocation.json", canonical_file=True
        )
        if actual != embedded:
            raise VerificationError(f"manifest/child invocation mismatch: {run_id}")
    if (
        load_json(root / "input_binding.json", canonical_file=True)
        != manifest["input_binding"]
    ):
        raise VerificationError("manifest/input_binding.json mismatch")
    result = load_json(root / "result.json", canonical_file=True)
    if result != {"result": manifest["result"], "schema": "h0_phase_a_execution_v1"}:
        raise VerificationError("manifest/result.json mismatch")
    aggregate = load_json(root / "verification/aggregate.json", canonical_file=True)
    if aggregate != report:
        raise VerificationError(
            "stored aggregate differs from independent reconstruction"
        )
    build_identity = load_json(root / "build_identity.json", canonical_file=True)
    runtime_identity = load_json(root / "runtime_identity.json", canonical_file=True)
    gpu_identity = load_json(root / "gpu_identity.json", canonical_file=True)
    comparison_identity = load_json(root / "comparison.json", canonical_file=True)
    if not all(
        isinstance(value, dict)
        for value in (
            build_identity,
            runtime_identity,
            gpu_identity,
            comparison_identity,
        )
    ):
        raise VerificationError("identity/comparison artifact is not an object")
    blocking_status = {"blocking_result": manifest["result"], "state": "not_produced"}
    if build_identity.get("state") == "complete":
        _verify_complete_build_identity(build_identity, manifest["controller_input"])
        extension_load = build_identity.get("extension_load")
        result = manifest["result"]
        if result == "extension_load_failed":
            if (
                not isinstance(extension_load, dict)
                or extension_load.get("state") != "failed"
            ):
                raise VerificationError(
                    "extension_load_failed lacks its confined failure record"
                )
        elif result in {
            "runner_nonzero",
            "capture_perturbs_policy",
            "packet_invalid",
            "phase_a_pass",
        }:
            if (
                not isinstance(extension_load, dict)
                or extension_load.get("state") != "complete"
            ):
                raise VerificationError(
                    "post-extension result lacks a complete confined load record"
                )
        elif (
            isinstance(extension_load, dict)
            and extension_load.get("state") == "rejected"
            and result != "provenance_invalid"
        ):
            raise VerificationError(
                "rejected extension-load record has a non-provenance result"
            )
        records = runtime_identity.get("child_runtime_inputs")
        if (
            set(runtime_identity)
            != {
                "bound_inputs_digest",
                "child_runtime_inputs",
                "library_dirs",
                "resolved_policy_fingerprint",
                "state",
                "tool_runtime",
            }
            or runtime_identity["state"] != "complete"
            or runtime_identity["bound_inputs_digest"]
            != manifest["controller_input"]["bound_inputs"]["digest"]
            or runtime_identity["library_dirs"]
            != manifest["controller_input"]["library_dirs"]
            or runtime_identity["resolved_policy_fingerprint"]
            != "c7a6dbb35168cba75249b7f2c67d8455b6f634732493e455a4bb920aab6d7782"
            or runtime_identity["tool_runtime"]
            != manifest["controller_input"]["bound_inputs"]["tool_runtime"]
            or not isinstance(records, list)
            or any(
                not isinstance(record, dict)
                or set(record) != {"run_id", "runtime_inputs"}
                for record in records
            )
            or [record["run_id"] for record in records] != list(RUN_IDS)
        ):
            raise VerificationError("runtime identity differs from controller binding")
        for invocation, record in zip(
            manifest["child_invocations"], records, strict=True
        ):
            runtime_value = record["runtime_inputs"]
            if not isinstance(runtime_value, dict):
                raise VerificationError("runtime input attestation is not an object")
            if runtime_value.get("state") in {"complete", "rejected"}:
                _verify_runtime_inputs(
                    runtime_value,
                    invocation,
                    manifest["controller_input"],
                    build_identity,
                )
            elif (
                runtime_value
                != {
                    "blocking_result": manifest["result"],
                    "schema": "h0_runtime_inputs_v1",
                    "state": "not_produced",
                }
                or invocation["runtime_inputs_digest"] is not None
            ):
                raise VerificationError("runtime input not-produced status mismatch")
        if gpu_identity != {
            **manifest["controller_input"]["gpu"],
            "state": "complete",
        }:
            raise VerificationError(
                "runtime/GPU identity differs from controller binding"
            )
    else:
        if (
            build_identity != blocking_status
            or runtime_identity != blocking_status
            or gpu_identity != blocking_status
        ):
            raise VerificationError(
                "not-produced identity status has missing or unknown members"
            )
        if manifest["result"] not in {
            "provenance_invalid",
            "build_failed",
            "runner_timeout",
            "serialization_failed",
            "artifact_missing_or_unreadable",
            "unclassified_execution_failure",
        }:
            raise VerificationError("post-build result lacks complete build identity")
    if manifest["result"] in {
        "capture_perturbs_policy",
        "packet_invalid",
        "phase_a_pass",
    }:
        inventories = {
            run_id: load_json(
                root / "runs" / run_id / "policy_inventory.json", canonical_file=True
            )
            for run_id in RUN_IDS
        }
        for run_id, inventory in inventories.items():
            _verify_policy_inventory(run_id, inventory)
            mot = (root / "runs" / run_id / "MOT17-04-SDP.txt").read_bytes()
            if inventory["mot_output"] != {
                "length": len(mot),
                "sha256": hashlib.sha256(mot).hexdigest(),
            }:
                raise VerificationError(
                    f"policy inventory MOT identity mismatch: {run_id}"
                )
        equality_members = (
            "mot_output",
            "final_track_rows",
            "active_tid_slot_pairs",
            "relink_debug_raw",
        )
        relations: list[dict[str, Any]] = []
        first_unequal: str | None = None
        for run_id in RUN_IDS[1:]:
            for member in equality_members:
                equal = inventories[RUN_IDS[0]][member] == inventories[run_id][member]
                relations.append(
                    {
                        "equal": equal,
                        "left": RUN_IDS[0],
                        "member": member,
                        "right": run_id,
                    }
                )
                if not equal and first_unequal is None:
                    first_unequal = f"{RUN_IDS[0]}:{run_id}:{member}"
        for member in ("proposal_projection", "winner_commit_projection"):
            reference = inventories[RUN_IDS[1]][member]
            for run_id in RUN_IDS[2:]:
                equal = reference == inventories[run_id][member]
                relations.append(
                    {
                        "equal": equal,
                        "left": RUN_IDS[1],
                        "member": member,
                        "right": run_id,
                    }
                )
                if not equal and first_unequal is None:
                    first_unequal = f"{RUN_IDS[1]}:{run_id}:{member}"
        for run_id in RUN_IDS[1:]:
            zero = inventories[run_id]["overflow_vector"] == [0] * 9
            relations.append(
                {
                    "equal": zero,
                    "left": run_id,
                    "member": "overflow_vector",
                    "right": "zero_vector",
                }
            )
            if not zero and first_unequal is None:
                first_unequal = f"{run_id}:overflow_vector"
        reconstructed_comparison = {
            "first_unequal": first_unequal,
            "relations": relations,
            "state": "equal" if first_unequal is None else "unequal",
        }
        if comparison_identity != reconstructed_comparison:
            raise VerificationError(
                "comparison.json differs from independent A7.6 reconstruction"
            )
        if (first_unequal is None) != (manifest["policy_comparison"] == "equal"):
            raise VerificationError(
                "manifest policy comparison differs from reconstructed inventories"
            )
        if first_unequal is not None:
            if manifest["result"] != "capture_perturbs_policy":
                raise VerificationError("unequal policy inventory has wrong result")
        else:
            _verify_packet_files(root, manifest)
    elif comparison_identity != blocking_status:
        raise VerificationError(
            "not-produced comparison status has missing or unknown members"
        )
    return report


def _verify_packet_files(root: Path, manifest: Mapping[str, Any]) -> None:
    if manifest["result"] not in {"packet_invalid", "phase_a_pass"}:
        raise VerificationError(
            "equal complete D requires packet-invalid or pass result"
        )
    if TOOLS_DIR.as_posix() not in sys.path:
        sys.path.insert(0, TOOLS_DIR.as_posix())
    from export_headline_bridge_decision_trace import canonical_semantic_packet
    from verify_headline_bridge_decision_trace import verify_capture

    reconstructed_states: list[str] = []
    semantic_digests: list[str] = []
    for run_id in RUN_IDS[1:]:
        capture = load_json(root / "runs" / run_id / "packet.json", canonical_file=True)
        stored = load_json(
            root / "runs" / run_id / "packet_verification.json", canonical_file=True
        )
        try:
            packet_report = verify_capture(capture)
            packet = canonical_semantic_packet(capture)
        except (KeyError, TypeError, ValueError):
            reconstructed_states.append("fail")
            if stored != {"failure": "packet_invalid", "state": "fail"}:
                raise VerificationError(
                    f"packet verifier failure record mismatch: {run_id}"
                )
        else:
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
            expected_proposal = {
                "count": len(candidates),
                "digest": digest(proposal_payload),
                "records": proposal_payload,
            }
            expected_winner = {
                "count": len(commits),
                "digest": digest(winner_payload),
                "records": winner_payload,
            }
            inventory = load_json(
                root / "runs" / run_id / "policy_inventory.json", canonical_file=True
            )
            if (
                inventory["proposal_projection"] != expected_proposal
                or inventory["winner_commit_projection"] != expected_winner
            ):
                raise VerificationError(f"packet/policy projection mismatch: {run_id}")
            expected_overflow = [
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
            if inventory["overflow_vector"] != expected_overflow:
                raise VerificationError(f"packet/policy overflow mismatch: {run_id}")
            reconstructed_states.append("pass")
            semantic_digests.append(packet_report["semantic_digest_sha256"])
            if stored != {"report": packet_report, "state": "pass"}:
                raise VerificationError(
                    f"packet verifier pass record mismatch: {run_id}"
                )
    if len(semantic_digests) == 3 and len(set(semantic_digests)) != 1:
        reconstructed_states[1] = "fail"
    if reconstructed_states != manifest["packet_verification_states"]:
        raise VerificationError(
            "manifest packet states differ from replay reconstruction"
        )
    packets_valid = reconstructed_states == ["pass"] * 3
    if packets_valid != (manifest["result"] == "phase_a_pass"):
        raise VerificationError(
            "packet replay/canonical-digest state disagrees with result"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.evidence.is_dir():
            report = verify_evidence_root(args.evidence)
        else:
            value = load_json(args.evidence, canonical_file=True)
            report = verify_evidence(value)
    except (VerificationError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"H0 Phase-A verification rejected: {exc}", file=sys.stderr)
        return 1
    print((canonical_json(report) + b"\n").decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
