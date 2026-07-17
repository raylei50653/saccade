#!/usr/bin/env python3
"""H0 Phase-A parent controller (``h0_phase_a_controller_v1``).

Authority: Amendment 7 (A7.2–A7.9) + RC1 (A7.RC1.1–A7.RC1.4).

This module is pre-seal engineering substrate. The operator entry point accepts
no positional arguments and no options other than ``-h``/``--help``. Without a
complete ``h0_preseal_freeze_v3`` and owner seal it refuses execution, emits no
H0 terminal, and writes no Phase-A evidence root.

Pure builders and validators are exported for mechanical tests and for the
independent verifier. Runtime and implementer discretion is forbidden: every
command vector, environment key, run order, digest algorithm, result enum row,
and C/D/V matrix cell is frozen in A7/RC1 and mirrored here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parents[1]
SCHEMA_PATH = TOOLS_DIR / "h0_phase_a_execution_schema_v1.json"
CHILD_PATH = TOOLS_DIR / "run_h0_phase_a_child.py"

CONTROLLER_SCHEMA_VERSION = "h0_phase_a_controller_v1"
CHILD_SCHEMA_VERSION = "h0_phase_a_child_v1"
EXECUTION_SCHEMA_VERSION = "h0_phase_a_execution_v1"
VERIFIER_SCHEMA_VERSION = "h0_phase_a_verifier_v1"
BOUND_INPUTS_SCHEMA_VERSION = "h0_bound_inputs_v1"
CONTROLLER_INPUT_SCHEMA_VERSION = "h0_phase_a_controller_input_v1"
CHILD_INPUT_SCHEMA_VERSION = "h0_phase_a_child_input_v1"
CHILD_RESULT_SCHEMA_VERSION = "h0_phase_a_child_result_v1"
CONTROLLER_PLAN_SCHEMA_VERSION = "h0_phase_a_controller_plan_v1"
MUTATION_OBS_SCHEMA_VERSION = "h0_phase_a_mutation_observation_v1"
SEQUENCE_DIGEST_ALGORITHM = "h0_sequence_input_digest_v1"

RUN_IDS: tuple[str, ...] = (
    "00_capture_off",
    "01_capture_on_1",
    "02_capture_on_2",
    "03_capture_on_3",
)

RESULT_ENUM: tuple[str, ...] = (
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

# Linux inotify bits (RC1.3).
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800

INOTIFY_MASK = (
    IN_CLOSE_WRITE
    | IN_MODIFY
    | IN_ATTRIB
    | IN_DELETE_SELF
    | IN_MOVE_SELF
    | IN_CREATE
    | IN_DELETE
    | IN_MOVED_FROM
    | IN_MOVED_TO
)  # 4046
assert INOTIFY_MASK == 4046

INOTIFY_MASK_NAMES: tuple[str, ...] = (
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

BOUND_INPUT_CHECKPOINTS: tuple[str, ...] = (
    "T0",
    "T1",
    "T2a",
    "T2b",
    "T3",
    "T4",
)

DEADLINE_SECONDS = 3600
SEQUENCE_ROOT_REL = "datasets/MOT17/train/MOT17-04-SDP"
EXCLUDED_SEQUENCE_SUBTREES: tuple[str, ...] = ("gt", "det")
BUILD_DIR_REL = "build/h0_phase_a"
NOT_RUN_LOG_BYTES = b"NOT_RUN\n"

PARENT_COMMAND_VECTOR: tuple[str, ...] = (
    "uv",
    "run",
    "--frozen",
    "python",
    "scripts/tools/run_h0_phase_a.py",
)

BUILD_CONFIGURE_VECTOR: tuple[str, ...] = (
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
)

BUILD_BUILD_VECTOR: tuple[str, ...] = (
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
)

EVALUATOR_ARGV_PREFIX: tuple[str, ...] = (
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
)

ENV_KEY_ORDER: tuple[str, ...] = (
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

PATH_SET_C: tuple[str, ...] = (
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

PATH_SET_D: tuple[str, ...] = (
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

PATH_SET_V: tuple[str, ...] = (
    "runs/01_capture_on_1/packet_verification.json",
    "runs/02_capture_on_2/packet_verification.json",
    "runs/03_capture_on_3/packet_verification.json",
)

# result -> (required_sets, forbidden_sets)
CDV_RESULT_MATRIX: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "provenance_invalid": (("C",), ("D", "V")),
    "build_failed": (("C",), ("D", "V")),
    "extension_load_failed": (("C",), ("D", "V")),
    "runner_nonzero": (("C",), ("D", "V")),
    "runner_timeout": (("C",), ("D", "V")),
    "serialization_failed": (("C",), ("D", "V")),
    "artifact_missing_or_unreadable": (("C",), ("D", "V")),
    "unclassified_execution_failure": (("C",), ("D", "V")),
    "capture_perturbs_policy": (("C", "D"), ("V",)),
    "packet_invalid": (("C", "D", "V"), ()),
    "phase_a_pass": (("C", "D", "V"), ()),
}

PATH_SETS: dict[str, tuple[str, ...]] = {
    "C": PATH_SET_C,
    "D": PATH_SET_D,
    "V": PATH_SET_V,
}

EXPOSURE_GATES: dict[str, Any] = {
    "capture_phase": "phase_a",
    "require_candidate_exposure": True,
    "require_commit_exposure": False,
}

TRACE_CAPACITIES: dict[str, int] = {
    "pair_capacity": 65536,
    "candidate_capacity": 16384,
    "claim_capacity": 16384,
    "commit_capacity": 16384,
}


class ContractError(ValueError):
    """Fail-closed contract violation (no warning path)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def load_schema() -> dict[str, Any]:
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    return json.loads(raw)


def canonical_json_bytes(obj: Any) -> bytes:
    """A7.8 canonical UTF-8 JSON: lexicographic keys, compact separators, trailing LF."""
    return (
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_mapping(obj: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(obj)))


def require_physical_abs_path(path: str, *, field: str) -> str:
    """Reject empty, relative, non-canonical, and symlink-substituted paths."""
    if not isinstance(path, str) or not path:
        raise ContractError("symlink_or_non_canonical_path", f"{field} empty")
    if "\0" in path:
        raise ContractError("symlink_or_non_canonical_path", f"{field} contains NUL")
    if not path.startswith("/"):
        raise ContractError(
            "symlink_or_non_canonical_path", f"{field} not absolute: {path!r}"
        )
    if path != os.path.normpath(path):
        raise ContractError(
            "symlink_or_non_canonical_path",
            f"{field} not normpath-canonical: {path!r}",
        )
    if any(part == ".." for part in path.split("/")):
        raise ContractError(
            "symlink_or_non_canonical_path", f"{field} has ..: {path!r}"
        )
    p = Path(path)
    if p.exists():
        if p.is_symlink():
            raise ContractError(
                "symlink_or_non_canonical_path",
                f"{field} is a symlink: {path!r}",
            )
        try:
            real = os.path.realpath(path, strict=True)
        except OSError as exc:
            raise ContractError(
                "symlink_or_non_canonical_path",
                f"{field} realpath failed: {exc}",
            ) from exc
        if real != path:
            raise ContractError(
                "symlink_or_non_canonical_path",
                f"{field} realpath drift {path!r} -> {real!r}",
            )
    return path


def parent_command_vector() -> list[str]:
    return list(PARENT_COMMAND_VECTOR)


def build_configure_vector() -> list[str]:
    return list(BUILD_CONFIGURE_VECTOR)


def build_build_vector() -> list[str]:
    return list(BUILD_BUILD_VECTOR)


def ordered_run_plan() -> list[str]:
    return list(RUN_IDS)


def evidence_root_rel(instrumentation_head: str) -> str:
    if len(instrumentation_head) != 40 or any(
        c not in "0123456789abcdef" for c in instrumentation_head
    ):
        raise ContractError(
            "illegal_enum", "instrumentation_head must be 40 lowercase hex"
        )
    return f"docs/modules/semantic/research/evidence/h0_phase_a_{instrumentation_head}"


def evidence_incomplete_rel(instrumentation_head: str) -> str:
    return evidence_root_rel(instrumentation_head) + ".incomplete"


def run_dir_rel(instrumentation_head: str, run_id: str) -> str:
    if run_id not in RUN_IDS:
        raise ContractError("illegal_enum", f"unknown run_id {run_id!r}")
    return f"{evidence_incomplete_rel(instrumentation_head)}/runs/{run_id}"


def child_command_vector(repository_root: str, run_id: str) -> list[str]:
    """RC1.1 exact child argv. No PATH lookup; absolute executable and script."""
    root = require_physical_abs_path(repository_root, field="repository_root")
    if run_id not in RUN_IDS:
        raise ContractError("illegal_enum", f"unknown run_id {run_id!r}")
    return [
        f"{root}/.venv/bin/python",
        "-I",
        "-B",
        f"{root}/scripts/tools/run_h0_phase_a_child.py",
        "--run-id",
        run_id,
    ]


def evaluator_argv(run_dir: str) -> list[str]:
    """RC1.1 synthetic evaluator argv. ``run_dir`` is repository-relative or absolute RUN path."""
    if not isinstance(run_dir, str) or not run_dir:
        raise ContractError("argument_mismatch", "run_dir empty")
    if "\0" in run_dir or any(part == ".." for part in Path(run_dir).parts):
        raise ContractError(
            "symlink_or_non_canonical_path", f"run_dir invalid: {run_dir!r}"
        )
    return list(EVALUATOR_ARGV_PREFIX) + ["--output", f"{run_dir}/_runtime"]


def child_environment(
    *,
    repository_root: str,
    run_id: str,
    instrumentation_head: str,
    cuda_device_uuid: str,
    tensorrt_lib_dir: str,
    pytorch_lib_dir: str,
    cuda_lib64_dir: str,
) -> dict[str, str]:
    """RC1.2 exact sanitized child environment from an empty mapping."""
    root = require_physical_abs_path(repository_root, field="repository_root")
    trt = require_physical_abs_path(tensorrt_lib_dir, field="tensorrt_lib_dir")
    pth = require_physical_abs_path(pytorch_lib_dir, field="pytorch_lib_dir")
    cuda = require_physical_abs_path(cuda_lib64_dir, field="cuda_lib64_dir")
    if not isinstance(cuda_device_uuid, str) or not cuda_device_uuid:
        raise ContractError("environment_mismatch", "cuda_device_uuid empty")
    if run_id not in RUN_IDS:
        raise ContractError("illegal_enum", f"unknown run_id {run_id!r}")

    run_rel = run_dir_rel(instrumentation_head, run_id)
    run_abs = f"{root}/{run_rel}"
    run_tmp = f"{run_abs}/_env"
    build_abs = f"{root}/{BUILD_DIR_REL}"

    # Colon-join four physical dirs; no empty member; fixed order (RC1.2).
    ld = ":".join((build_abs, trt, pth, cuda))
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": cuda_device_uuid,
        "HOME": f"{run_tmp}/home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ld,
        "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SACCADE_BUILD_PATH": build_abs,
        "SACCADE_DETECT_BARRIER": "event",
        "SACCADE_DOUBLE_BUFFER": "1",
        "SACCADE_GPU_DECODE": "1",
        "SACCADE_MAIN_NMS_GRAPHED": "1",
        "TMPDIR": f"{run_tmp}/tmp",
        "TZ": "UTC",
        "XDG_CACHE_HOME": f"{run_tmp}/xdg-cache",
    }
    # Guarantee exact key set and order for digests.
    if tuple(sorted(env)) != tuple(sorted(ENV_KEY_ORDER)):
        raise ContractError("environment_mismatch", "environment key set drift")
    return {k: env[k] for k in ENV_KEY_ORDER}


def environment_digest(env: Mapping[str, str]) -> str:
    if set(env) != set(ENV_KEY_ORDER):
        raise ContractError("environment_mismatch", f"env keys {sorted(env)} != frozen")
    ordered = {k: env[k] for k in ENV_KEY_ORDER}
    return digest_mapping(ordered)


def _is_excluded_sequence_path(rel: str) -> bool:
    parts = rel.split("/")
    return bool(parts) and parts[0] in EXCLUDED_SEQUENCE_SUBTREES


def sequence_file_record_bytes(
    relative_path: str, byte_length: int, sha256_hex: str
) -> bytes:
    """Canonical per-file record for the sequence-input digest (A7.3).

    Format (UTF-8):
        <relative_path>\\0<decimal_byte_length>\\0<lowercase_sha256>\\n
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise ContractError("sequence_input_digest_mismatch", "empty relative_path")
    if relative_path.startswith("/") or "\\" in relative_path or "\0" in relative_path:
        raise ContractError(
            "symlink_or_non_canonical_path",
            f"non-POSIX relative path: {relative_path!r}",
        )
    if any(part in ("", ".", "..") for part in relative_path.split("/")):
        raise ContractError(
            "symlink_or_non_canonical_path",
            f"non-canonical relative path: {relative_path!r}",
        )
    if byte_length < 0:
        raise ContractError("sequence_input_digest_mismatch", "negative byte_length")
    if len(sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in sha256_hex):
        raise ContractError(
            "sequence_input_digest_mismatch", "sha256 must be 64 lowercase hex"
        )
    return f"{relative_path}\0{byte_length}\0{sha256_hex}\n".encode("utf-8")


def compute_sequence_input_digest(
    sequence_root: Path,
    *,
    sequence_root_rel: str = SEQUENCE_ROOT_REL,
) -> dict[str, Any]:
    """A7.3 canonical sequence-input digest.

    Inventory every regular file under ``sequence_root`` whose path relative to
    the sequence root is not under ``gt/`` or ``det/``. Sort by relative-path
    bytes. Aggregate SHA-256 of the concatenation of per-file records.
    """
    root = sequence_root
    if not root.is_dir():
        raise ContractError(
            "sequence_input_digest_mismatch",
            f"sequence root missing: {root}",
        )
    try:
        root_real = Path(os.path.realpath(str(root), strict=True))
    except OSError as exc:
        raise ContractError(
            "symlink_or_non_canonical_path",
            f"sequence root realpath failed: {exc}",
        ) from exc

    records: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded subtrees at the sequence-root level only when they are
        # direct children named gt/det; also prune any path whose relative
        # prefix is excluded.
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        # Do not follow or enter excluded first-level subtrees.
        if rel_dir:
            top = rel_dir.split(os.sep)[0]
            if top in EXCLUDED_SEQUENCE_SUBTREES:
                dirnames[:] = []
                continue
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not (
                (not rel_dir and d in EXCLUDED_SEQUENCE_SUBTREES)
                or Path(dirpath, d).is_symlink()
            )
        )
        for name in sorted(filenames):
            fp = Path(dirpath) / name
            if fp.is_symlink():
                raise ContractError(
                    "symlink_or_non_canonical_path",
                    f"symlink in sequence inventory: {fp}",
                )
            if not fp.is_file():
                continue
            rel = fp.relative_to(root).as_posix()
            if _is_excluded_sequence_path(rel):
                continue
            # Ensure file remains under the physical sequence root.
            try:
                real_fp = Path(os.path.realpath(str(fp), strict=True))
            except OSError as exc:
                raise ContractError(
                    "symlink_or_non_canonical_path",
                    f"sequence file realpath failed: {exc}",
                ) from exc
            if root_real not in real_fp.parents and real_fp != root_real:
                # must be under root_real
                try:
                    real_fp.relative_to(root_real)
                except ValueError as exc:
                    raise ContractError(
                        "symlink_or_non_canonical_path",
                        f"path traversal: {fp}",
                    ) from exc
            data = fp.read_bytes()
            records.append(
                {
                    "relative_path": rel,
                    "byte_length": len(data),
                    "sha256": sha256_bytes(data),
                }
            )

    records.sort(key=lambda r: r["relative_path"].encode("utf-8"))
    blob = b"".join(
        sequence_file_record_bytes(r["relative_path"], r["byte_length"], r["sha256"])
        for r in records
    )
    return {
        "algorithm": SEQUENCE_DIGEST_ALGORITHM,
        "sequence_root": sequence_root_rel,
        "excluded_subtrees": list(EXCLUDED_SEQUENCE_SUBTREES),
        "file_records": records,
        "aggregate_sha256": sha256_bytes(blob),
    }


def bound_inputs_inventory_digest(inventory: Mapping[str, Any]) -> str:
    """Digest over the four exhaustive members of ``h0_bound_inputs_v1``."""
    payload = {
        "schema_version": BOUND_INPUTS_SCHEMA_VERSION,
        "instrumentation_head": inventory["instrumentation_head"],
        "repository": inventory["repository"],
        "models_engines": inventory["models_engines"],
        "sequence": inventory["sequence"],
        "tool_runtime_inputs": inventory["tool_runtime_inputs"],
    }
    return digest_mapping(payload)


def build_bound_inputs_v1(
    *,
    instrumentation_head: str,
    repository: Sequence[Mapping[str, Any]],
    models_engines: Sequence[Mapping[str, Any]],
    sequence: Mapping[str, Any],
    tool_runtime_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """RC1.3 ``h0_bound_inputs_v1`` assembly. Caller supplies frozen inventories."""
    inv: dict[str, Any] = {
        "schema_version": BOUND_INPUTS_SCHEMA_VERSION,
        "instrumentation_head": instrumentation_head,
        "repository": list(repository),
        "models_engines": list(models_engines),
        "sequence": dict(sequence),
        "tool_runtime_inputs": list(tool_runtime_inputs),
    }
    inv["inventory_digest"] = bound_inputs_inventory_digest(inv)
    return inv


def classify_inotify_event(
    *,
    mask: int,
    watch_path: str,
    event_path: str,
    bound_paths: Sequence[str],
    output_prefixes: Sequence[str],
) -> str:
    """RC1.3 mutation classification. Reads/output paths are ignored → ``none``."""
    if mask & ~(INOTIFY_MASK) and not (mask & INOTIFY_MASK):
        return "unclassified_monitor_failure"
    # Ignore pure read-like events outside the frozen mask.
    if not (mask & INOTIFY_MASK):
        return "none"

    # Output paths ignored.
    for pref in output_prefixes:
        if event_path == pref or event_path.startswith(pref.rstrip("/") + "/"):
            return "none"

    bound = set(bound_paths)
    if event_path in bound or watch_path in bound:
        return "bound_path_mutation"

    # Ancestor move/delete of a bound path.
    if mask & (IN_DELETE_SELF | IN_MOVE_SELF | IN_MOVED_FROM | IN_MOVED_TO | IN_DELETE):
        for bp in bound:
            if bp == watch_path or bp.startswith(watch_path.rstrip("/") + "/"):
                return "ancestor_move_or_delete"
            if event_path and (
                bp == event_path or bp.startswith(event_path.rstrip("/") + "/")
            ):
                return "ancestor_move_or_delete"

    # Event on a path that is neither bound nor output → fail closed if it
    # touches a bound ancestor directory that we watch.
    for bp in bound:
        parent = str(Path(bp).parent)
        if event_path == parent or watch_path == parent:
            return "bound_path_mutation"

    return "none"


def empty_mutation_observation() -> dict[str, Any]:
    return {
        "schema_version": MUTATION_OBS_SCHEMA_VERSION,
        "inotify_mask": INOTIFY_MASK,
        "inotify_mask_names": list(INOTIFY_MASK_NAMES),
        "monitor_status": "stopped",
        "events": [],
        "final_classification": "none",
    }


def cdv_paths_for_result(result: str) -> tuple[list[str], list[str]]:
    if result not in CDV_RESULT_MATRIX:
        raise ContractError("illegal_enum", f"unknown result {result!r}")
    required_sets, forbidden_sets = CDV_RESULT_MATRIX[result]
    required: list[str] = []
    for name in required_sets:
        required.extend(PATH_SETS[name])
    forbidden: list[str] = []
    for name in forbidden_sets:
        forbidden.extend(PATH_SETS[name])
    # Deterministic order: sorted path bytes within each set already frozen;
    # overall required is set-order C then D then V as listed.
    return required, forbidden


def cdv_matrix_row(result: str) -> dict[str, Any]:
    if result not in CDV_RESULT_MATRIX:
        raise ContractError("illegal_enum", f"unknown result {result!r}")
    required_sets, forbidden_sets = CDV_RESULT_MATRIX[result]
    required_paths, forbidden_paths = cdv_paths_for_result(result)
    return {
        "result": result,
        "required_sets": list(required_sets),
        "forbidden_sets": list(forbidden_sets),
        "required_paths": required_paths,
        "forbidden_paths": forbidden_paths,
    }


def full_cdv_matrix() -> list[dict[str, Any]]:
    return [cdv_matrix_row(r) for r in RESULT_ENUM]


def build_controller_plan(
    *,
    repository_root: str,
    instrumentation_head: str,
    cuda_device_uuid: str,
    tensorrt_lib_dir: str,
    pytorch_lib_dir: str,
    cuda_lib64_dir: str,
) -> dict[str, Any]:
    """Deterministic ordered run plan + all frozen command/env vectors."""
    root = require_physical_abs_path(repository_root, field="repository_root")
    child_vectors = {run_id: child_command_vector(root, run_id) for run_id in RUN_IDS}
    eval_vectors = {
        run_id: evaluator_argv(run_dir_rel(instrumentation_head, run_id))
        for run_id in RUN_IDS
    }
    env_tables = {
        run_id: child_environment(
            repository_root=root,
            run_id=run_id,
            instrumentation_head=instrumentation_head,
            cuda_device_uuid=cuda_device_uuid,
            tensorrt_lib_dir=tensorrt_lib_dir,
            pytorch_lib_dir=pytorch_lib_dir,
            cuda_lib64_dir=cuda_lib64_dir,
        )
        for run_id in RUN_IDS
    }
    return {
        "schema_version": CONTROLLER_PLAN_SCHEMA_VERSION,
        "parent_command_vector": parent_command_vector(),
        "build_configure_vector": build_configure_vector(),
        "build_build_vector": build_build_vector(),
        "ordered_run_plan": ordered_run_plan(),
        "child_command_vectors": child_vectors,
        "evaluator_argv_by_run": eval_vectors,
        "environment_tables_by_run": env_tables,
        "deadline_seconds": DEADLINE_SECONDS,
        "exposure_gates": dict(EXPOSURE_GATES),
        "result_enum": list(RESULT_ENUM),
        "cdv_matrix": full_cdv_matrix(),
        "inotify_mask": INOTIFY_MASK,
        "inotify_mask_names": list(INOTIFY_MASK_NAMES),
        "bound_input_checkpoints": list(BOUND_INPUT_CHECKPOINTS),
    }


def build_child_input(
    *,
    repository_root: str,
    instrumentation_head: str,
    run_id: str,
    capture_run_uuid: str,
    cuda_device_uuid: str,
    tensorrt_lib_dir: str,
    pytorch_lib_dir: str,
    cuda_lib64_dir: str,
    bound_inputs_digest: str,
    sequence_input_digest: str,
) -> dict[str, Any]:
    root = require_physical_abs_path(repository_root, field="repository_root")
    env = child_environment(
        repository_root=root,
        run_id=run_id,
        instrumentation_head=instrumentation_head,
        cuda_device_uuid=cuda_device_uuid,
        tensorrt_lib_dir=tensorrt_lib_dir,
        pytorch_lib_dir=pytorch_lib_dir,
        cuda_lib64_dir=cuda_lib64_dir,
    )
    return {
        "schema_version": CHILD_INPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "repository_root": root,
        "instrumentation_head": instrumentation_head,
        "capture_run_uuid": capture_run_uuid,
        "environment": env,
        "command_vector": child_command_vector(root, run_id),
        "evaluator_argv": evaluator_argv(run_dir_rel(instrumentation_head, run_id)),
        "bound_inputs_digest": bound_inputs_digest,
        "sequence_input_digest": sequence_input_digest,
    }


def build_child_result(
    *,
    run_id: str,
    exit_class: str,
    command_vector: Sequence[str],
    evaluator_argv_vec: Sequence[str],
    environment_digest_hex: str,
    bound_inputs_digest: str,
    sequence_input_digest: str,
    capture_run_uuid: str,
    stdout_sha256: str,
    stderr_sha256: str,
    failure_reason_code: str = "none",
) -> dict[str, Any]:
    return {
        "schema_version": CHILD_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "exit_class": exit_class,
        "command_vector": list(command_vector),
        "evaluator_argv": list(evaluator_argv_vec),
        "environment_digest": environment_digest_hex,
        "bound_inputs_digest": bound_inputs_digest,
        "sequence_input_digest": sequence_input_digest,
        "capture_run_uuid": capture_run_uuid,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "failure_reason_code": failure_reason_code,
    }


def child_popen_spec(
    *,
    repository_root: str,
    run_id: str,
    instrumentation_head: str,
    cuda_device_uuid: str,
    tensorrt_lib_dir: str,
    pytorch_lib_dir: str,
    cuda_lib64_dir: str,
    stdout_path: str,
    stderr_path: str,
) -> dict[str, Any]:
    """Exact ``subprocess.Popen`` kwargs (RC1.1). No shell, fixed stdio files."""
    root = require_physical_abs_path(repository_root, field="repository_root")
    env = child_environment(
        repository_root=root,
        run_id=run_id,
        instrumentation_head=instrumentation_head,
        cuda_device_uuid=cuda_device_uuid,
        tensorrt_lib_dir=tensorrt_lib_dir,
        pytorch_lib_dir=pytorch_lib_dir,
        cuda_lib64_dir=cuda_lib64_dir,
    )
    return {
        "args": child_command_vector(root, run_id),
        "cwd": root,
        "env": env,
        "shell": False,
        "close_fds": True,
        "start_new_session": True,
        "stdin": "DEVNULL",
        "stdout": stdout_path,
        "stderr": stderr_path,
    }


def select_controller_result(predicates: Mapping[str, bool]) -> str:
    """A7.7 top-to-bottom first-true selection. Unknown predicate keys fail closed."""
    allowed = set(RESULT_ENUM) | {"_force_unclassified"}
    extra = set(predicates) - allowed
    if extra:
        raise ContractError(
            "unrecognized_state", f"unknown predicates: {sorted(extra)}"
        )
    for name in RESULT_ENUM:
        if predicates.get(name, False):
            return name
    # If nothing matched, mandatory catch-all.
    return "unclassified_execution_failure"


def validate_controller_input(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the sole legal bound controller-input contract (strict)."""
    required = {
        "schema_version",
        "instrumentation_head",
        "repository_root",
        "tensorrt_lib_dir",
        "pytorch_lib_dir",
        "cuda_lib64_dir",
        "cuda_device_uuid",
        "cuda_pci_bus_id",
        "bound_inputs",
        "capture_run_uuids",
    }
    keys = set(obj)
    missing = required - keys
    if missing:
        raise ContractError("missing_required_field", f"missing {sorted(missing)}")
    extra = keys - required
    if extra:
        raise ContractError("unknown_field", f"extra fields {sorted(extra)}")
    if obj["schema_version"] != CONTROLLER_INPUT_SCHEMA_VERSION:
        raise ContractError("illegal_enum", "controller input schema_version mismatch")
    head = obj["instrumentation_head"]
    if (
        not isinstance(head, str)
        or len(head) != 40
        or any(c not in "0123456789abcdef" for c in head)
    ):
        raise ContractError("illegal_enum", "bad instrumentation_head")
    require_physical_abs_path(obj["repository_root"], field="repository_root")
    for field in (
        "tensorrt_lib_dir",
        "pytorch_lib_dir",
        "cuda_lib64_dir",
    ):
        require_physical_abs_path(obj[field], field=field)
    for field in ("cuda_device_uuid", "cuda_pci_bus_id"):
        if not isinstance(obj[field], str) or not obj[field]:
            raise ContractError("missing_required_field", f"{field} empty")
    uuids = obj["capture_run_uuids"]
    if not isinstance(uuids, Mapping):
        raise ContractError("schema_nonconformance", "capture_run_uuids not object")
    if set(uuids) != set(RUN_IDS):
        raise ContractError(
            "schema_nonconformance",
            f"capture_run_uuids keys {sorted(uuids)} != run plan",
        )
    for run_id in RUN_IDS:
        if not isinstance(uuids[run_id], str) or not uuids[run_id]:
            raise ContractError("missing_required_field", f"uuid for {run_id}")
    bi = obj["bound_inputs"]
    if not isinstance(bi, Mapping):
        raise ContractError("schema_nonconformance", "bound_inputs not object")
    validate_bound_inputs(bi, expected_head=head)
    return dict(obj)


def validate_bound_inputs(
    obj: Mapping[str, Any],
    *,
    expected_head: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "instrumentation_head",
        "repository",
        "models_engines",
        "sequence",
        "tool_runtime_inputs",
        "inventory_digest",
    }
    keys = set(obj)
    if keys - required:
        raise ContractError(
            "unknown_field", f"bound_inputs extra {sorted(keys - required)}"
        )
    if required - keys:
        raise ContractError(
            "missing_required_field",
            f"bound_inputs missing {sorted(required - keys)}",
        )
    if obj["schema_version"] != BOUND_INPUTS_SCHEMA_VERSION:
        raise ContractError("illegal_enum", "bound_inputs schema_version")
    if expected_head is not None and obj["instrumentation_head"] != expected_head:
        raise ContractError("bound_input_digest_mismatch", "instrumentation_head drift")
    recomputed = bound_inputs_inventory_digest(obj)
    if recomputed != obj["inventory_digest"]:
        raise ContractError(
            "bound_input_digest_mismatch",
            f"inventory_digest {obj['inventory_digest']} != recomputed {recomputed}",
        )
    seq = obj["sequence"]
    if not isinstance(seq, Mapping):
        raise ContractError("schema_nonconformance", "sequence not object")
    validate_sequence_digest_object(seq)
    return dict(obj)


def validate_sequence_digest_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "algorithm",
        "sequence_root",
        "excluded_subtrees",
        "file_records",
        "aggregate_sha256",
    }
    keys = set(obj)
    if keys - required:
        raise ContractError(
            "unknown_field", f"sequence extra {sorted(keys - required)}"
        )
    if required - keys:
        raise ContractError(
            "missing_required_field",
            f"sequence missing {sorted(required - keys)}",
        )
    if obj["algorithm"] != SEQUENCE_DIGEST_ALGORITHM:
        raise ContractError("illegal_enum", "sequence algorithm")
    if obj["sequence_root"] != SEQUENCE_ROOT_REL:
        raise ContractError("sequence_input_digest_mismatch", "sequence_root")
    if list(obj["excluded_subtrees"]) != list(EXCLUDED_SEQUENCE_SUBTREES):
        raise ContractError("sequence_input_digest_mismatch", "excluded_subtrees")
    records = obj["file_records"]
    if not isinstance(records, list):
        raise ContractError("schema_nonconformance", "file_records not list")
    # Recompute aggregate from records; order must already be path-sorted.
    sorted_recs = sorted(records, key=lambda r: r["relative_path"].encode("utf-8"))
    if [r["relative_path"] for r in records] != [
        r["relative_path"] for r in sorted_recs
    ]:
        raise ContractError("sequence_input_digest_mismatch", "file_records not sorted")
    blob = b"".join(
        sequence_file_record_bytes(
            r["relative_path"], int(r["byte_length"]), r["sha256"]
        )
        for r in records
    )
    agg = sha256_bytes(blob)
    if agg != obj["aggregate_sha256"]:
        raise ContractError(
            "sequence_input_digest_mismatch",
            f"aggregate {obj['aggregate_sha256']} != {agg}",
        )
    return dict(obj)


def validate_child_input(obj: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_id",
        "repository_root",
        "instrumentation_head",
        "capture_run_uuid",
        "environment",
        "command_vector",
        "evaluator_argv",
        "bound_inputs_digest",
        "sequence_input_digest",
    }
    keys = set(obj)
    if keys - required:
        raise ContractError(
            "unknown_field", f"child_input extra {sorted(keys - required)}"
        )
    if required - keys:
        raise ContractError(
            "missing_required_field",
            f"child_input missing {sorted(required - keys)}",
        )
    if obj["schema_version"] != CHILD_INPUT_SCHEMA_VERSION:
        raise ContractError("illegal_enum", "child_input schema_version")
    if obj["run_id"] not in RUN_IDS:
        raise ContractError("illegal_enum", f"run_id {obj['run_id']!r}")
    root = require_physical_abs_path(obj["repository_root"], field="repository_root")
    expected_cmd = child_command_vector(root, obj["run_id"])
    if list(obj["command_vector"]) != expected_cmd:
        raise ContractError(
            "command_vector_mismatch",
            f"child command vector mismatch for {obj['run_id']}",
        )
    expected_eval = evaluator_argv(
        run_dir_rel(obj["instrumentation_head"], obj["run_id"])
    )
    if list(obj["evaluator_argv"]) != expected_eval:
        raise ContractError("argument_mismatch", "evaluator_argv mismatch")
    env = obj["environment"]
    if not isinstance(env, Mapping) or set(env) != set(ENV_KEY_ORDER):
        raise ContractError("environment_mismatch", "environment key set")
    # Re-derive expected environment needs library dirs from LD_LIBRARY_PATH.
    # Child input must already carry the exact frozen table; verify digests.
    for key in ENV_KEY_ORDER:
        if key not in env or not isinstance(env[key], str):
            raise ContractError("environment_mismatch", f"missing/non-str {key}")
    if env["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
        raise ContractError("environment_mismatch", "CUDA_DEVICE_ORDER")
    if env["LANG"] != "C.UTF-8" or env["LC_ALL"] != "C.UTF-8":
        raise ContractError("environment_mismatch", "locale")
    if env["PYTHONHASHSEED"] != "0" or env["PYTHONNOUSERSITE"] != "1":
        raise ContractError("environment_mismatch", "python isolation")
    if env["TZ"] != "UTC":
        raise ContractError("environment_mismatch", "TZ")
    if env["PATH"] != f"{root}/.venv/bin:/usr/bin:/bin":
        raise ContractError("environment_mismatch", "PATH")
    if env["SACCADE_BUILD_PATH"] != f"{root}/{BUILD_DIR_REL}":
        raise ContractError("environment_mismatch", "SACCADE_BUILD_PATH")
    for lit_key, lit_val in (
        ("SACCADE_DETECT_BARRIER", "event"),
        ("SACCADE_DOUBLE_BUFFER", "1"),
        ("SACCADE_GPU_DECODE", "1"),
        ("SACCADE_MAIN_NMS_GRAPHED", "1"),
    ):
        if env[lit_key] != lit_val:
            raise ContractError("environment_mismatch", lit_key)
    return dict(obj)


def write_child_context(path: Path, child_input: Mapping[str, Any]) -> None:
    """Controller-written fixed context file read by the child (UUID + digests)."""
    validate_child_input(child_input)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(child_input)))


def child_context_path(
    repository_root: str, instrumentation_head: str, run_id: str
) -> Path:
    rel = run_dir_rel(instrumentation_head, run_id)
    return Path(repository_root) / rel / "child_input.json"


def assemble_synthetic_evidence(
    *,
    controller_input: Mapping[str, Any],
    result: str,
    mutation_observation: Mapping[str, Any] | None = None,
    child_exit_class: str = "completed",
    controller_self_report_valid: bool = True,
    checkpoint_equal: bool = True,
) -> dict[str, Any]:
    """Build a complete synthetic evidence package for verifier/unit tests.

    Does not run build, capture, or Phase A. Uses only frozen vectors and the
    bound contract.
    """
    cin = validate_controller_input(controller_input)
    if result not in RESULT_ENUM:
        raise ContractError("illegal_enum", f"result {result!r}")
    root = cin["repository_root"]
    head = cin["instrumentation_head"]
    plan = build_controller_plan(
        repository_root=root,
        instrumentation_head=head,
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    bi = json.loads(json.dumps(cin["bound_inputs"]))
    # Distinct object from bound_inputs.sequence so independent tamper tests apply.
    seq = json.loads(json.dumps(bi["sequence"]))
    env_digests = {
        run_id: environment_digest(plan["environment_tables_by_run"][run_id])
        for run_id in RUN_IDS
    }
    empty = sha256_bytes(b"")
    child_results: dict[str, Any] = {}
    for run_id in RUN_IDS:
        child_results[run_id] = build_child_result(
            run_id=run_id,
            exit_class=child_exit_class,
            command_vector=plan["child_command_vectors"][run_id],
            evaluator_argv_vec=plan["evaluator_argv_by_run"][run_id],
            environment_digest_hex=env_digests[run_id],
            bound_inputs_digest=bi["inventory_digest"],
            sequence_input_digest=seq["aggregate_sha256"],
            capture_run_uuid=cin["capture_run_uuids"][run_id],
            stdout_sha256=empty,
            stderr_sha256=empty,
            failure_reason_code="none"
            if child_exit_class == "completed"
            else "unexpected_exit",
        )

    required_paths, forbidden_paths = cdv_paths_for_result(result)
    published = list(required_paths)
    artifact_states = {p: "produced" for p in required_paths}
    for p in forbidden_paths:
        artifact_states[p] = "not_produced"

    t0_digest = bi["inventory_digest"]
    now = time.monotonic_ns()
    checkpoints = []
    for i, cp in enumerate(BOUND_INPUT_CHECKPOINTS):
        checkpoints.append(
            {
                "checkpoint_id": cp,
                "inventory_digest": t0_digest
                if checkpoint_equal
                else sha256_bytes(b"drift"),
                "equal_to_t0": checkpoint_equal,
                "monotonic_ns": now + i,
            }
        )

    mut = (
        dict(mutation_observation)
        if mutation_observation is not None
        else empty_mutation_observation()
    )

    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "controller_schema_version": CONTROLLER_SCHEMA_VERSION,
        "child_schema_version": CHILD_SCHEMA_VERSION,
        "verifier_schema_version": VERIFIER_SCHEMA_VERSION,
        "instrumentation_head": head,
        "result": result,
        "controller_self_report_valid": controller_self_report_valid,
        "parent_command_vector": plan["parent_command_vector"],
        "build_configure_vector": plan["build_configure_vector"],
        "build_build_vector": plan["build_build_vector"],
        "ordered_run_plan": plan["ordered_run_plan"],
        "child_command_vectors": plan["child_command_vectors"],
        "evaluator_argv_by_run": plan["evaluator_argv_by_run"],
        "environment_tables_by_run": plan["environment_tables_by_run"],
        "environment_digests_by_run": env_digests,
        "bound_inputs": bi,
        "sequence_input_digest": seq,
        "mutation_observation": mut,
        "checkpoint_records": checkpoints,
        "child_results": child_results,
        "cdv_matrix_row": cdv_matrix_row(result),
        "published_paths": published,
        "artifact_states": artifact_states,
    }


def run_hermetic_contract_session(
    controller_input: Mapping[str, Any],
    *,
    result: str = "phase_a_pass",
    spawn_hook: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hermetic session: validate contract, build plan, optional spawn hook.

    Never performs real cmake/build/capture/Phase A. ``spawn_hook`` if provided
    receives the exact Popen spec per run and must return a child_result dict.
    """
    cin = validate_controller_input(controller_input)
    plan = build_controller_plan(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    if spawn_hook is not None:
        child_results: dict[str, Any] = {}
        for run_id in plan["ordered_run_plan"]:
            spec = child_popen_spec(
                repository_root=cin["repository_root"],
                run_id=run_id,
                instrumentation_head=cin["instrumentation_head"],
                cuda_device_uuid=cin["cuda_device_uuid"],
                tensorrt_lib_dir=cin["tensorrt_lib_dir"],
                pytorch_lib_dir=cin["pytorch_lib_dir"],
                cuda_lib64_dir=cin["cuda_lib64_dir"],
                stdout_path="/dev/null",
                stderr_path="/dev/null",
            )
            child_results[run_id] = spawn_hook(
                run_id=run_id, popen_spec=spec, plan=plan, controller_input=cin
            )
        evidence = assemble_synthetic_evidence(
            controller_input=cin,
            result=result,
        )
        evidence["child_results"] = child_results
        return evidence
    return assemble_synthetic_evidence(controller_input=cin, result=result)


def operator_main(argv: Sequence[str] | None = None) -> int:
    """A7.2/A7.3 sole operator entry: no args except -h/--help; no execution authority yet."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        sys.stdout.write(
            "H0 Phase-A controller (h0_phase_a_controller_v1).\n"
            "Sole invocation after v3 freeze + owner seal:\n"
            "  uv run --frozen python scripts/tools/run_h0_phase_a.py\n"
            "No other arguments are accepted. Pre-seal: no execution authority.\n"
        )
        return 0
    if args:
        sys.stderr.write(
            "ContractError: argument_mismatch: controller accepts no arguments "
            "other than -h/--help\n"
        )
        return 2

    # Pre-seal engineering: refuse to launch Phase A without v3 freeze + seal.
    # This path must not write evidence, emit H0 terminals, or run build/capture.
    sys.stderr.write(
        "H0 Phase-A controller: no execution authority. "
        "h0_preseal_freeze_v3 + owner SEALED are required before invocation. "
        "H0 remains on Route 0′ pre-seal engineering. "
        "No terminal emitted; no evidence root written.\n"
    )
    return 2


def main() -> None:
    raise SystemExit(operator_main())


if __name__ == "__main__":
    main()
