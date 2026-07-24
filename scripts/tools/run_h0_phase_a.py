#!/usr/bin/env python3
"""A7/RC1 fail-closed Phase-A parent controller.

This file is an implementation substrate, not execution authority.  The
operator entry point accepts no arguments and will not execute unless exactly
one complete, canonical v3 freeze supplies a conforming ``controller_input``
contract for the current, clean, sealed head.
"""
# status: stable

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


CONTROLLER_SCHEMA = "h0_phase_a_controller_v1"
EXECUTION_SCHEMA = "h0_phase_a_execution_v1"
CHILD_SCHEMA = "h0_phase_a_child_v1"
BOUND_INPUTS_SCHEMA = "h0_bound_inputs_v1"
BUILD_TOOL_BINDING_SCHEMA = "h0_build_tool_binding_v1"
BUILD_TOOL_BINDING_RESOLVER = "h0_build_tool_binding_resolver_v1"
# Amendment 10: owner-authority overlay is typed separately from runtime
# repository binding.  The controller-input member key remains
# ``authority_landing`` for member-parity continuity with historical freezes.
OWNER_AUTHORITY_OVERLAY_SCHEMA = "h0_owner_authority_overlay_v1"
# Historical RC2 landing schema retained for archive/cross-version admission.
HISTORICAL_AUTHORITY_LANDING_SCHEMA = "h0_authority_landing_v1"
DECLARATION_PATH = (
    "docs/modules/semantic/research/"
    "headline_bridge_full_decision_capture_declaration_20260713.md"
)
# Declaration is owner-overlay authority only; never a runtime repository input.
RUNTIME_EXCLUDED_REPOSITORY_PATHS = frozenset({DECLARATION_PATH})
# Canonical controller-input member declaration for the authoritative current
# v3 pre-seal artifact.  The freeze assembler builds exactly this set, the
# execution schema enumerates it as its property universe, and the independent
# pre-seal verifier holds a byte-identical transcription; equality across all of
# them (plus an explicit literal) is pinned by
# tests/contract/test_h0_controller_input_member_parity.py.
CONTROLLER_INPUT_MEMBERS = frozenset(
    {
        "authority_landing",
        "bound_inputs",
        "build_tool_binding",
        "document_type",
        "evidence_root",
        "execution_constants",
        "gpu",
        "incomplete_root",
        "instrumentation_head",
        "library_dirs",
        "repository_root",
        "schema",
        "sequence_input_digest",
        "tool_paths",
    }
)
SCHEMA_PATH = Path(__file__).with_name("h0_phase_a_execution_schema_v1.json")
ROOT = Path(__file__).resolve().parents[2]

OPERATOR_ARGV = ("uv", "run", "--frozen", "python", "scripts/tools/run_h0_phase_a.py")
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
BUILD_ENVIRONMENT_SCHEMA = "h0_build_environment_v2"
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
BUILD_TOOL_ROLES = (
    ("cxx", "c++"),
    ("cmake", "cmake"),
)
RUN_IDS = (
    "00_capture_off",
    "01_capture_on_1",
    "02_capture_on_2",
    "03_capture_on_3",
)
CAPTURE_ON_RUN_IDS = RUN_IDS[1:]
EVALUATOR_ARGV_PREFIX = (
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
TRACE_CAPACITIES = (65536, 16384, 16384, 16384)
DEADLINE_SECONDS = 3600
SEQUENCE_REL = "datasets/MOT17/train/MOT17-04-SDP"
SEQUENCE_NAME = "MOT17-04-SDP"
POLICY_FINGERPRINT = "c7a6dbb35168cba75249b7f2c67d8455b6f634732493e455a4bb920aab6d7782"
MODEL_LOGICAL_PATHS = (
    "models/yolo/mamba_head_26m.engine",
    "models/yolo/yolo26m.pt",
    "models/yolo/yolo26m_backbone_640_best.engine",
    "runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt",
    "runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt",
)
# Amendment 10: declaration.md is owner-overlay authority, not a runtime input.
REQUIRED_REPOSITORY_INPUTS = (
    "configs/presets/mamba_whole_graph_m.yaml",
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
# Pre-Amendment-10 packets still carry the declaration as a required repository
# input; archive verification admits that exact historical tuple only.
HISTORICAL_REQUIRED_REPOSITORY_INPUTS = (
    "configs/presets/mamba_whole_graph_m.yaml",
    DECLARATION_PATH,
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

RESULT_ENUM = (
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
ALL_ARTIFACT_PATHS = C_PATHS + D_PATHS + V_PATHS

RESULT_MATRIX: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    result: (C_PATHS, D_PATHS + V_PATHS) for result in RESULT_ENUM[:8]
}
RESULT_MATRIX.update(
    {
        "capture_perturbs_policy": (C_PATHS + D_PATHS, V_PATHS),
        "packet_invalid": (ALL_ARTIFACT_PATHS, ()),
        "phase_a_pass": (ALL_ARTIFACT_PATHS, ()),
    }
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
CHECKPOINT_FAILURE_CAUSES = frozenset(
    {
        "events_before",
        "recompute_failed",
        "events_after",
        "inventory_mismatch",
    }
)
INOTIFY_MASK_NAMES = (
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
IN_CLOSE_WRITE = 0x00000008
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
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
)

# These records publish declaration choices only; confinement backend and ingress
# mechanism remain implementation detail, bound exclusively by v3 file hashes.
CHILD_ENVIRONMENT_SCHEMA = "h0_child_environment_v1"
CHILD_ENVIRONMENT_TEMPLATE = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "<GPU_UUID>",
    "HOME": "<RUN_TMP>/home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": "<ROOT>/build/h0_phase_a:<TENSORRT_LIBRARY_DIR>:<PYTORCH_LIBRARY_DIR>:<CUDA_LIBRARY_DIR>",
    "PATH": "<ROOT>/.venv/bin:/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "SACCADE_BUILD_PATH": "<ROOT>/build/h0_phase_a",
    "SACCADE_DETECT_BARRIER": "event",
    "SACCADE_DOUBLE_BUFFER": "1",
    "SACCADE_GPU_DECODE": "1",
    "SACCADE_MAIN_NMS_GRAPHED": "1",
    "TMPDIR": "<RUN_TMP>/tmp",
    "TZ": "UTC",
    "XDG_CACHE_HOME": "<RUN_TMP>/xdg-cache",
}
TRACE_LIFECYCLE = {
    "schema": "h0_trace_lifecycle_v1",
    "capture_off": ["set_research_h0_bridge_trace(false,65536,16384,16384,16384)"],
    "capture_on": [
        "set_research_h0_bridge_trace(true,65536,16384,16384,16384)",
        "clear_research_h0_bridge_trace()",
        "drain_research_h0_bridge_trace(seq=MOT17-04-SDP,capture_phase=phase_a,require_candidate_exposure=true,require_commit_exposure=false,capture_run_uuid=<CONTROLLER_UUID>)",
    ],
}
BOUND_INPUT_ALGORITHMS = {
    "bound_inputs": "h0_bound_inputs_v1",
    "repository": "git_ls_tree_r_full_tree_z",
    "sequence": "h0_sequence_inputs_v1",
    "actual_loaded_attestation": "h0_runtime_inputs_v1",
}
CANONICALIZATION = {
    "json": "utf8_lexicographic_keys_compact_finite_trailing_lf_v1",
    "checksums": "lowercase_sha256_two_spaces_posix_path_sorted_utf8_bytes_v1",
}
PUBLICATION_ROLLBACK = {
    "publication": "atomic_rename_incomplete_to_final_then_fsync_parent_v1",
    "rollback": "remove_partial_D_V_before_checksums_or_leave_incomplete_unpublished_v1",
}

CHILD_ENV_KEYS = (
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
FORBIDDEN_SELECTOR_KEYS = frozenset(
    {
        "PYTHONPATH",
        "LD_PRELOAD",
        "SACCADE_STREAM_MODE",
        "SACCADE_NV12_BUFFER",
        "H0_PRESET",
        "H0_SEQUENCE",
        "H0_DETECTOR",
        "H0_DATA_ROOT",
        "H0_OUTPUT_ROOT",
        "H0_REPEAT_COUNT",
        "H0_DEADLINE",
        "H0_BUILD_DIR",
        "H0_PHASE_B",
    }
)
# Implementation mechanism for RC1.2/RC1.3 enforcement, bound only through the
# v3 file hashes.  These names are not RC1 declaration constants and must never
# be published through execution_constants or pinned by the execution schema.
RUNTIME_CONFINEMENT_BACKEND = "landlock_seccomp_ptrace_v1"
RUNTIME_INGRESS_POLICY = "deny_external_bytes_v1"
RUNTIME_TRACE_SCOPE = ("execve", "execveat", "mmap", "open", "openat", "openat2")


class ContractError(RuntimeError):
    """A fail-closed controller-contract violation."""


class DriftError(ContractError):
    """A bound-input mutation or inventory mismatch."""


class CheckpointDriftError(DriftError):
    """Drift proven while one named bound-input checkpoint was executing.

    ``checkpoint_record`` is the operation's own terminal observation.  It
    prevents the controller from inferring a failed row later from monitor
    history that may belong to a surrounding stage instead.
    """

    def __init__(
        self,
        checkpoint: str,
        message: str,
        *,
        checkpoint_record: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.checkpoint = checkpoint
        self.checkpoint_record = (
            dict(checkpoint_record) if checkpoint_record is not None else None
        )


# The controller's authoritative build subtree.  Its pre-existence at launch is
# the first ordered-terminal (`provenance_invalid`) hazard: both prior
# owner-authorized re-entries (#209 and #224/#227) consumed their exactly-once
# authorization only to fail preflight here, never reaching the capture
# checkpoints.  The predicate below is the single source of that check so the
# non-authoritative launch-hygiene gate can reuse the controller's own verdict
# *before* an authorization is spent — see scripts/tools/h0_launch_hygiene_gate.py.
AUTHORITATIVE_BUILD_SUBTREE = "build/h0_phase_a"


def assert_no_preexisting_build_tree(root: Path) -> None:
    """Fail closed when the authoritative build subtree already exists.

    This is the sole source of the ``build/h0_phase_a exists at controller
    launch`` preflight terminal.  ``preflight_controller_input`` and the
    non-authoritative launch-hygiene gate both call it, so the gate's verdict is
    the controller's own verdict rather than a re-implementation that could
    silently drift from it.
    """
    build_dir = root / AUTHORITATIVE_BUILD_SUBTREE
    if build_dir.exists():
        raise ContractError(
            f"{AUTHORITATIVE_BUILD_SUBTREE} exists at controller launch"
        )


def canonical_json_bytes(value: object) -> bytes:
    """A7.8 canonical JSON bytes, excluding the required file trailing LF."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_file_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def read_canonical_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_object_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"unreadable or malformed JSON {path}: {exc}") from exc
    try:
        expected = canonical_json_file_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-canonical JSON value in {path}: {exc}") from exc
    if raw != expected:
        raise ContractError(f"JSON is not canonical UTF-8 with one trailing LF: {path}")
    return value


def read_strict_json(path: Path) -> Any:
    """Read JSON without canonicalizing it, while still rejecting duplicates."""
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"unreadable or malformed JSON {path}: {exc}") from exc


def validate_schema_document(value: object, document_type: str) -> None:
    """Validate through the frozen strict schema; no normalization occurs."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - execution environment admission
        raise ContractError("jsonschema dependency unavailable") from exc
    schema = read_strict_json(SCHEMA_PATH)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(part) for part in exc.absolute_path)
        raise ContractError(
            f"schema rejection at {where or '<root>'}: {exc.message}"
        ) from exc
    if not isinstance(value, dict) or value.get("document_type") != document_type:
        raise ContractError(f"expected {document_type} document")


def require_canonical_relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ContractError(f"non-canonical relative POSIX path: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"non-canonical relative POSIX path: {path!r}")
    if pure.as_posix() != path:
        raise ContractError(f"non-canonical relative POSIX path: {path!r}")
    path.encode("utf-8", errors="strict")
    return path


def require_canonical_absolute(path: str, *, directory: bool | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.as_posix() != path:
        raise ContractError(f"non-canonical absolute POSIX path: {path!r}")
    if candidate.resolve(strict=True) != candidate:
        raise ContractError(f"symlink-substituted or non-physical path: {path}")
    if directory is True and not candidate.is_dir():
        raise ContractError(f"required directory is absent: {path}")
    if directory is False and not candidate.is_file():
        raise ContractError(f"required regular file is absent: {path}")
    return candidate


def require_lexical_absolute(path: str) -> Path:
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or "\\" in path
        or "\x00" in path
    ):
        raise ContractError(f"non-canonical absolute POSIX path: {path!r}")
    candidate = Path(path)
    if candidate.as_posix() != path or any(
        part in {".", ".."} for part in PurePosixPath(path).parts
    ):
        raise ContractError(f"non-canonical absolute POSIX path: {path!r}")
    return candidate


def _regular_file_record(path: Path, logical_path: str) -> dict[str, Any]:
    require_canonical_relative(logical_path)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ContractError(f"bound input is not a non-symlink regular file: {path}")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise DriftError(f"bound input changed while hashing: {path}")
    return {"length": len(data), "path": logical_path, "sha256": sha256_bytes(data)}


def sequence_input_inventory(sequence_root: Path) -> dict[str, Any]:
    """A7.3 canonical sequence inventory, with complete gt/ and det/ exclusion."""
    sequence_root = require_canonical_absolute(sequence_root.as_posix(), directory=True)
    records: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(sequence_root, followlinks=False):
        current = Path(dirpath)
        relative_dir = current.relative_to(sequence_root)
        if current.is_symlink():
            raise ContractError(f"sequence contains a symlink directory: {current}")
        if relative_dir == Path("."):
            dirnames[:] = sorted(name for name in dirnames if name not in {"gt", "det"})
        else:
            dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = current / filename
            relative = path.relative_to(sequence_root).as_posix()
            records.append(_regular_file_record(path, relative))
    records.sort(key=lambda item: item["path"].encode("utf-8"))
    digest_payload = {"algorithm": "h0_sequence_inputs_v1", "files": records}
    return {
        "algorithm": "h0_sequence_inputs_v1",
        "digest": sha256_bytes(canonical_json_bytes(digest_payload)),
        "files": records,
        "root": SEQUENCE_REL,
    }


def bound_inventory_digest(inventory: Mapping[str, Any]) -> str:
    """RC1.3 digest over the four exhaustive, canonically ordered categories."""
    payload = {
        "models_engines": inventory["models_engines"],
        "repository": inventory["repository"],
        "schema": BOUND_INPUTS_SCHEMA,
        "sequence": inventory["sequence"],
        "tool_runtime": inventory["tool_runtime"],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def validate_bound_inventory(inventory: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "digest",
        "repository",
        "models_engines",
        "sequence",
        "tool_runtime",
    }
    if (
        set(inventory) != expected_keys
        or inventory.get("schema") != BOUND_INPUTS_SCHEMA
    ):
        raise ContractError("h0_bound_inputs_v1 has missing or unknown members")
    for category in ("repository", "models_engines", "tool_runtime"):
        records = inventory[category]
        if not isinstance(records, list):
            raise ContractError(f"bound inventory {category} is not an array")
        path_key = "path" if category == "repository" else "logical_path"
        sort_keys = [record[path_key].encode("utf-8") for record in records]
        if sort_keys != sorted(sort_keys) or len(sort_keys) != len(set(sort_keys)):
            raise ContractError(
                f"bound inventory {category} is not unique canonical order"
            )
        if category != "repository":
            for record in records:
                logical_path = record["logical_path"]
                if str(logical_path).startswith("/"):
                    require_lexical_absolute(logical_path)
                else:
                    require_canonical_relative(logical_path)
                require_lexical_absolute(record["realpath"])
                for link in record["symlink_chain"]:
                    require_lexical_absolute(link)
            realpaths = [record["realpath"] for record in records]
            if len(realpaths) != len(set(realpaths)):
                raise ContractError(
                    f"bound inventory {category} has duplicate physical paths"
                )
    sequence = inventory["sequence"]
    if (
        not isinstance(sequence, dict)
        or sequence.get("algorithm") != "h0_sequence_inputs_v1"
    ):
        raise ContractError("sequence inventory algorithm mismatch")
    seq_files = sequence.get("files")
    if not isinstance(seq_files, list):
        raise ContractError("sequence files are not an array")
    seq_paths = [record.get("path") for record in seq_files if isinstance(record, dict)]
    if len(seq_paths) != len(seq_files) or seq_paths != sorted(
        seq_paths, key=lambda p: p.encode("utf-8")
    ):
        raise ContractError("sequence inventory is not sorted by UTF-8 path bytes")
    expected_seq = sha256_bytes(
        canonical_json_bytes({"algorithm": "h0_sequence_inputs_v1", "files": seq_files})
    )
    if sequence.get("digest") != expected_seq:
        raise ContractError("canonical sequence-input digest mismatch")
    if inventory.get("digest") != bound_inventory_digest(inventory):
        raise ContractError("h0_bound_inputs_v1 aggregate digest mismatch")
    repository_paths = {record["path"] for record in inventory["repository"]}
    if not set(REQUIRED_REPOSITORY_INPUTS).issubset(repository_paths):
        raise ContractError(
            "repository inventory omits an A7/RC1 controller/runtime authority"
        )
    # Amendment 10 fail-closed: declaration must never re-enter runtime inventory.
    leaked = repository_paths & RUNTIME_EXCLUDED_REPOSITORY_PATHS
    if leaked:
        raise ContractError(
            "authority-overlay path leaked into runtime repository inventory: "
            + ", ".join(sorted(leaked))
        )
    if (
        tuple(record["logical_path"] for record in inventory["models_engines"])
        != MODEL_LOGICAL_PATHS
    ):
        raise ContractError(
            "resolved model/engine path set differs from the sole evaluator vector"
        )


def _symlink_chain(path: Path) -> list[str]:
    chain: list[str] = []
    absolute = require_lexical_absolute(Path(os.path.abspath(path)).as_posix())
    current = Path("/")
    pending = list(absolute.parts[1:])
    seen: set[Path] = set()
    while pending:
        current /= pending.pop(0)
        if current.is_symlink():
            if current in seen:
                raise ContractError(f"symlink loop in bound input: {path}")
            seen.add(current)
            chain.append(current.as_posix())
            target = Path(os.readlink(current))
            replacement = target if target.is_absolute() else current.parent / target
            replacement = require_lexical_absolute(
                Path(os.path.abspath(replacement)).as_posix()
            )
            current = Path("/")
            pending = list(replacement.parts[1:]) + pending
    return chain


def external_input_record(root: Path, logical_path: str) -> dict[str, Any]:
    logical = Path(logical_path)
    path = (
        require_lexical_absolute(logical_path)
        if logical.is_absolute()
        else root / require_canonical_relative(logical_path)
    )
    chain = _symlink_chain(path)
    real = path.resolve(strict=True)
    info = real.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ContractError(f"external bound input is not regular: {logical_path}")
    data = real.read_bytes()
    return {
        "length": len(data),
        "logical_path": logical_path,
        "realpath": real.as_posix(),
        "sha256": sha256_bytes(data),
        "symlink_chain": chain,
    }


def repository_inventory(
    root: Path,
    head: str,
    git_path: Path,
    *,
    started: float,
    monitor: "BoundInputMonitor | None" = None,
    landing_overlay_paths: Iterable[str] = (),
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    try:
        process = _run_auxiliary_subprocess(
            ["git", "ls-tree", "-r", "--full-tree", "-z", head],
            executable=git_path,
            cwd=root,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            started=started,
            monitor=monitor,
            stage="git repository inventory",
            clock=clock,
        )
    except subprocess.CalledProcessError as exc:
        raise DriftError("git repository inventory exited nonzero") from exc
    overlays = set(landing_overlay_paths)
    records: list[dict[str, Any]] = []
    for raw in process.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode_raw, type_raw, oid_raw = metadata.split(b" ", 2)
            path_value = raw_path.decode("utf-8", errors="strict")
            mode = mode_raw.decode("ascii")
            git_type = type_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ContractError("git ls-tree emitted a non-canonical record") from exc
        require_canonical_relative(path_value)
        # Amendment 10: declaration is owner-overlay authority only.  Exclude it
        # from the runtime-bound repository inventory so S's one-line SEALED
        # append cannot collide with F-frozen runtime byte equality.
        if path_value in RUNTIME_EXCLUDED_REPOSITORY_PATHS:
            continue
        if git_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise ContractError(
                f"unsupported repository input {mode} {git_type} {path_value}"
            )
        working = root / path_value
        if mode == "120000":
            if not working.is_symlink():
                raise DriftError(f"repository symlink mismatch: {path_value}")
            data = os.fsencode(os.readlink(working))
            kind = "symlink"
        else:
            info = working.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or working.is_symlink():
                raise DriftError(f"repository regular-file mismatch: {path_value}")
            executable = bool(info.st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise DriftError(f"repository executable-mode mismatch: {path_value}")
            # RC2 admits freeze-artifact post-head overlay paths only after
            # preflight has independently proven I -> F -> S.  Runtime-bound
            # repository inventory remains an I inventory; owner-overlay paths
            # (declaration) are excluded above and monitored against S bytes.
            data = (
                _run_auxiliary_subprocess(
                    ["git", "show", f"{head}:{path_value}"],
                    executable=git_path,
                    cwd=root,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
                    started=started,
                    monitor=monitor,
                    stage="RC2 instrumentation-overlay inventory",
                    clock=clock,
                ).stdout
                if path_value in overlays
                else working.read_bytes()
            )
            kind = "regular"
        framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        computed_oid = (
            hashlib.sha1(framed, usedforsecurity=False).hexdigest()
            if len(oid) == 40
            else hashlib.sha256(framed).hexdigest()
        )
        if computed_oid != oid:
            raise DriftError(f"working-tree bytes differ from {head}: {path_value}")
        records.append(
            {
                "git_object": oid,
                "git_type": git_type,
                "kind": kind,
                "length": len(data),
                "mode": mode,
                "path": path_value,
                "sha256": sha256_bytes(data),
            }
        )
    records.sort(key=lambda record: record["path"].encode("utf-8"))
    return records


def recompute_bound_inventory(
    contract: Mapping[str, Any],
    *,
    started: float,
    monitor: "BoundInputMonitor | None" = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    root = require_canonical_absolute(contract["repository_root"], directory=True)
    git_path = require_canonical_absolute(
        contract["tool_paths"]["git"], directory=False
    )
    frozen = contract["bound_inputs"]
    models = sorted(
        (
            external_input_record(root, record["logical_path"])
            for record in frozen["models_engines"]
        ),
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    tools = sorted(
        (
            external_input_record(root, record["logical_path"])
            for record in frozen["tool_runtime"]
        ),
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    inventory: dict[str, Any] = {
        "digest": "",
        "models_engines": models,
        "repository": repository_inventory(
            root,
            contract["instrumentation_head"],
            git_path,
            started=started,
            monitor=monitor,
            landing_overlay_paths=contract["authority_landing"][
                "post_head_allowed_paths"
            ],
            clock=clock,
        ),
        "schema": BOUND_INPUTS_SCHEMA,
        "sequence": sequence_input_inventory(root / SEQUENCE_REL),
        "tool_runtime": tools,
    }
    inventory["digest"] = bound_inventory_digest(inventory)
    validate_bound_inventory(inventory)
    return inventory


def owner_authority_overlay(
    *,
    artifact_path: str,
    declaration_at_f: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the Amendment-10 owner authority overlay (freeze-time)."""
    require_canonical_relative(artifact_path)
    length = declaration_at_f.get("length")
    sha256 = declaration_at_f.get("sha256")
    if (
        not isinstance(length, int)
        or length < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise ContractError("declaration_at_f identity is malformed")
    return {
        "schema": OWNER_AUTHORITY_OVERLAY_SCHEMA,
        "artifact_path": artifact_path,
        "declaration_path": DECLARATION_PATH,
        "declaration_at_f": {"length": length, "sha256": sha256},
        "post_head_allowed_paths": [artifact_path, DECLARATION_PATH],
    }


def declaration_byte_identity(root: Path) -> dict[str, Any]:
    path = root / DECLARATION_PATH
    if path.is_symlink() or not path.is_file():
        raise ContractError("declaration is not a physical regular file")
    data = path.read_bytes()
    return {"length": len(data), "sha256": sha256_bytes(data)}


def verify_owner_authority_overlay_s_bytes(
    contract: Mapping[str, Any],
    *,
    started: float,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Continuous T0–T4: declaration worktree must equal sealed S bytes.

    Inotify already watches the declaration path via the authority overlay.  This
    check deliberately does not share the bound-input monitor with the git
    helper so a concurrent inventory mutation is attributed to the checkpoint's
    inventory comparison, not to a nested helper stage.
    """
    landing = contract.get("authority_landing")
    if not isinstance(landing, Mapping):
        raise ContractError("controller input has no authority-landing descriptor")
    schema = landing.get("schema")
    if schema == HISTORICAL_AUTHORITY_LANDING_SCHEMA:
        # Historical dual-bound packets are never re-executed under this path.
        return
    if schema != OWNER_AUTHORITY_OVERLAY_SCHEMA:
        raise ContractError("authority overlay schema is not the Amendment-10 type")
    if landing.get("declaration_path") != DECLARATION_PATH:
        raise ContractError("authority overlay declaration path drift")
    declaration_at_f = landing.get("declaration_at_f")
    if not isinstance(declaration_at_f, Mapping):
        raise ContractError("authority overlay lacks declaration_at_f")
    root = require_canonical_absolute(contract["repository_root"], directory=True)
    git_path = require_canonical_absolute(
        contract["tool_paths"]["git"], directory=False
    )
    # Execution checkout is S.  Continuous drift uses S bytes as the baseline,
    # not the F-time declaration_at_f record (which deliberately lacks the
    # SEALED line).
    sealed = _run_auxiliary_subprocess(
        ["git", "show", f"HEAD:{DECLARATION_PATH}"],
        executable=git_path,
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        started=started,
        stage="owner-overlay S-byte declaration",
        clock=clock,
    ).stdout
    working = (root / DECLARATION_PATH).read_bytes()
    if working != sealed:
        raise DriftError("declaration drifted from sealed S bytes")
    # F-time identity must remain a strict prefix of S (append-only owner event).
    f_length = int(declaration_at_f["length"])
    f_sha = str(declaration_at_f["sha256"])
    if len(sealed) < f_length or sha256_bytes(sealed[:f_length]) != f_sha:
        raise DriftError("sealed declaration is not an append-only extension of F")
    # Also prove runtime inventory remains free of the declaration.
    repository_paths = {
        record["path"] for record in contract["bound_inputs"]["repository"]
    }
    if DECLARATION_PATH in repository_paths:
        raise DriftError(
            "declaration re-entered runtime repository inventory during execution"
        )


def bound_file_paths(contract: Mapping[str, Any]) -> tuple[Path, ...]:
    root = Path(contract["repository_root"])
    inventory = contract["bound_inputs"]
    paths: list[Path] = []
    for record in inventory["repository"]:
        paths.append(root / record["path"])
    for path in contract["authority_landing"]["post_head_allowed_paths"]:
        paths.append(root / require_canonical_relative(path))
    for record in inventory["sequence"]["files"]:
        paths.append(root / SEQUENCE_REL / record["path"])
    for category in ("models_engines", "tool_runtime"):
        for record in inventory[category]:
            logical = Path(record["logical_path"])
            paths.append(logical if logical.is_absolute() else root / logical)
            paths.append(Path(record["realpath"]))
            paths.extend(Path(item) for item in record["symlink_chain"])
    return tuple(paths)


def verify_bound_checkpoint(
    contract: Mapping[str, Any],
    monitor: "BoundInputMonitor",
    name: str,
    *,
    started: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if name not in CHECKPOINTS:
        raise ContractError(f"unknown checkpoint: {name}")
    before = monitor.drain()
    if before:
        return _checkpoint_inventory_verdict(
            name,
            contract["bound_inputs"],
            None,
            events_before=before,
        )
    try:
        current = recompute_bound_inventory(
            contract, started=started, monitor=monitor, clock=clock
        )
        verify_owner_authority_overlay_s_bytes(contract, started=started, clock=clock)
    except DriftError as exc:
        # A recomputation failure did not yield an inventory that can be
        # compared.  Preserve only events observed during that operation.
        after = monitor.drain()
        raise CheckpointDriftError(
            name,
            str(exc) or exc.__class__.__name__,
            checkpoint_record=_failed_checkpoint(
                name,
                cause="events_after" if after else "recompute_failed",
                events_after=after,
                inventory_comparison_executed=False,
                inventory_equal=None,
                observed_digest=None,
            ),
        ) from exc
    return _checkpoint_inventory_verdict(
        name, contract["bound_inputs"], current, events_after=monitor.drain()
    )


def _event_records(events: Sequence[InotifyEvent]) -> list[dict[str, Any]]:
    return [
        {
            "classification": event.classification,
            "mask": event.mask,
            "path": event.path,
        }
        for event in events
    ]


def _failed_checkpoint(
    name: str,
    *,
    cause: str,
    events_before: Sequence[InotifyEvent] = (),
    events_after: Sequence[InotifyEvent] = (),
    inventory_comparison_executed: bool,
    inventory_equal: bool | None,
    observed_digest: str | None,
) -> dict[str, Any]:
    if cause not in CHECKPOINT_FAILURE_CAUSES:
        raise ContractError(f"unknown checkpoint failure cause: {cause}")
    return {
        "cause": cause,
        "digest": None,
        "events_after": _event_records(events_after),
        "events_before": _event_records(events_before),
        "inventory_comparison_executed": inventory_comparison_executed,
        "inventory_equal": inventory_equal,
        "monotonic_ns": time.monotonic_ns(),
        "name": name,
        "observed_digest": observed_digest,
        "state": "failed",
    }


def _not_reached_checkpoint(name: str) -> dict[str, Any]:
    return {
        "digest": None,
        "events_after": [],
        "events_before": [],
        "inventory_comparison_executed": False,
        "inventory_equal": None,
        "monotonic_ns": None,
        "name": name,
        "observed_digest": None,
        "state": "not_reached",
    }


def _checkpoint_inventory_verdict(
    name: str,
    frozen: Mapping[str, Any],
    current: Mapping[str, Any] | None,
    *,
    events_before: Sequence[InotifyEvent] = (),
    events_after: Sequence[InotifyEvent] = (),
) -> dict[str, Any]:
    """Produce the one truthful row for an executed checkpoint operation.

    The qualification harness calls this with synthetic inventories, so the
    same producer owns both authoritative and non-authoritative checkpoint
    row semantics without granting the latter an H0 terminal.
    """
    if events_before:
        raise CheckpointDriftError(
            name,
            f"mutation events before {name}: {events_before!r}",
            checkpoint_record=_failed_checkpoint(
                name,
                cause="events_before",
                events_before=events_before,
                inventory_comparison_executed=False,
                inventory_equal=None,
                observed_digest=None,
            ),
        )
    if current is None:
        raise ContractError("checkpoint verdict lacks a recomputed inventory")
    if events_after:
        raise CheckpointDriftError(
            name,
            f"mutation events after {name}: {events_after!r}",
            checkpoint_record=_failed_checkpoint(
                name,
                cause="events_after",
                events_after=events_after,
                inventory_comparison_executed=False,
                inventory_equal=None,
                observed_digest=current["digest"],
            ),
        )
    if current != frozen:
        raise CheckpointDriftError(
            name,
            f"bound inventory mismatch at {name}",
            checkpoint_record=_failed_checkpoint(
                name,
                cause="inventory_mismatch",
                inventory_comparison_executed=True,
                inventory_equal=False,
                observed_digest=current["digest"],
            ),
        )
    return {
        "digest": current["digest"],
        "events_after": [],
        "events_before": [],
        "inventory_comparison_executed": True,
        "inventory_equal": True,
        "monotonic_ns": time.monotonic_ns(),
        "name": name,
        "observed_digest": current["digest"],
        "state": "completed",
    }


def _failure_record(stage: str, exc: BaseException) -> dict[str, str]:
    if isinstance(exc, CheckpointDriftError):
        stage = f"checkpoint_{exc.checkpoint}"
    return {"reason": str(exc) or exc.__class__.__name__, "stage": stage}


def child_argv(root: Path, run_id: str) -> tuple[str, ...]:
    if run_id not in RUN_IDS:
        raise ContractError(f"unknown run id: {run_id}")
    root = require_canonical_absolute(root.as_posix(), directory=True)
    return (
        (root / ".venv/bin/python").as_posix(),
        "-I",
        "-B",
        (root / "scripts/tools/run_h0_phase_a_child.py").as_posix(),
        "--run-id",
        run_id,
    )


def evaluator_argv(run_dir: Path) -> tuple[str, ...]:
    run_dir = require_canonical_absolute(run_dir.as_posix(), directory=True)
    return EVALUATOR_ARGV_PREFIX + ((run_dir / "_runtime").as_posix(),)


def child_environment(
    root: Path,
    run_dir: Path,
    *,
    gpu_uuid: str,
    tensorrt_library_dir: str,
    pytorch_library_dir: str,
    cuda_library_dir: str,
    validate_paths: bool = True,
) -> dict[str, str]:
    root = require_canonical_absolute(root.as_posix(), directory=True)
    run_dir = require_canonical_absolute(run_dir.as_posix(), directory=True)
    if validate_paths:
        library_dirs = tuple(
            require_canonical_absolute(value, directory=True).as_posix()
            for value in (tensorrt_library_dir, pytorch_library_dir, cuda_library_dir)
        )
    else:
        library_dirs = (tensorrt_library_dir, pytorch_library_dir, cuda_library_dir)
        if any(
            not Path(value).is_absolute() or Path(value).as_posix() != value
            for value in library_dirs
        ):
            raise ContractError("RC1.2 library directory is not an absolute POSIX path")
    if len(set(library_dirs)) != 3:
        raise ContractError(
            "RC1.2 library directories must be three distinct physical paths"
        )
    run_tmp = run_dir / "_env"
    values = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "HOME": (run_tmp / "home").as_posix(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ":".join(
            ((root / "build/h0_phase_a").as_posix(),) + library_dirs
        ),
        "PATH": f"{root.as_posix()}/.venv/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SACCADE_BUILD_PATH": (root / "build/h0_phase_a").as_posix(),
        "SACCADE_DETECT_BARRIER": "event",
        "SACCADE_DOUBLE_BUFFER": "1",
        "SACCADE_GPU_DECODE": "1",
        "SACCADE_MAIN_NMS_GRAPHED": "1",
        "TMPDIR": (run_tmp / "tmp").as_posix(),
        "TZ": "UTC",
        "XDG_CACHE_HOME": (run_tmp / "xdg-cache").as_posix(),
    }
    if tuple(sorted(values)) != tuple(sorted(CHILD_ENV_KEYS)):
        raise AssertionError("internal RC1.2 environment key drift")
    return values


def environment_digest(environment: Mapping[str, str]) -> str:
    if set(environment) != set(CHILD_ENV_KEYS) or not all(
        isinstance(v, str) for v in environment.values()
    ):
        raise ContractError(
            "child environment has missing, unknown, or non-string members"
        )
    return sha256_bytes(canonical_json_bytes(dict(environment)))


def build_environment_digest(environment: Mapping[str, str]) -> str:
    if set(environment) != set(BUILD_ENVIRONMENT_KEYS) or not all(
        isinstance(value, str) for value in environment.values()
    ):
        raise ContractError(
            "build environment has missing, unknown, or non-string members"
        )
    return sha256_bytes(canonical_json_bytes(dict(environment)))


def normalize_pci_bus_id(value: str) -> str:
    try:
        domain_bus, device_function = value.lower().split(":", 1)
        bus, device_function = device_function.split(":", 1)
        device, function = device_function.split(".", 1)
        normalized = f"{int(domain_bus, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(function, 16)}"
    except (ValueError, TypeError) as exc:
        raise ContractError(f"non-canonical PCI bus ID: {value!r}") from exc
    if int(function, 16) > 7:
        raise ContractError(f"PCI function out of range: {value!r}")
    return normalized


def select_gpu_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ContractError("no physical NVIDIA GPU is available")
    normalized: list[dict[str, Any]] = []
    for record in records:
        exact = dict(record)
        exact["normalized_pci_bus_id"] = normalize_pci_bus_id(
            str(record["normalized_pci_bus_id"])
        )
        normalized.append(exact)
    normalized.sort(key=lambda record: record["normalized_pci_bus_id"])
    if len({record["normalized_pci_bus_id"] for record in normalized}) != len(
        normalized
    ):
        raise ContractError("duplicate normalized NVIDIA PCI bus ID")
    return normalized[0]


def nvml_gpu_inventory() -> list[dict[str, Any]]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except BaseException as exc:
        raise ContractError(f"NVML initialization failed: {exc}") from exc
    records: list[dict[str, Any]] = []
    try:

        def text_value(value: object) -> str:
            return (
                value.decode("utf-8", errors="strict")
                if isinstance(value, bytes)
                else str(value)
            )

        driver = text_value(pynvml.nvmlSystemGetDriverVersion())
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            records.append(
                {
                    "compute_capability": f"{int(major)}.{int(minor)}",
                    "driver": driver,
                    "name": text_value(pynvml.nvmlDeviceGetName(handle)),
                    "normalized_pci_bus_id": normalize_pci_bus_id(
                        text_value(pci.busId)
                    ),
                    "total_memory": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    "uuid": text_value(pynvml.nvmlDeviceGetUUID(handle)),
                    "vbios": text_value(pynvml.nvmlDeviceGetVbiosVersion(handle)),
                }
            )
    finally:
        pynvml.nvmlShutdown()
    return records


def execution_constants(root: Path) -> dict[str, Any]:
    """All A7/RC1 choices that v3 must mechanically reproduce."""
    return {
        "actual_loaded_input_attestation": "h0_runtime_inputs_v1",
        "bound_input_algorithms": BOUND_INPUT_ALGORITHMS,
        "build_environment_algorithm": BUILD_ENVIRONMENT_SCHEMA,
        "build_environment_keys": list(BUILD_ENVIRONMENT_KEYS),
        "build_tool_binding_algorithm": BUILD_TOOL_BINDING_RESOLVER,
        "build_vectors": [list(vector) for vector in BUILD_VECTORS],
        "c_paths": list(C_PATHS),
        "canonicalization": CANONICALIZATION,
        "checkpoints": list(CHECKPOINTS),
        "child_vectors": [list(child_argv(root, run_id)) for run_id in RUN_IDS],
        "child_environment_algorithm": CHILD_ENVIRONMENT_SCHEMA,
        "child_environment_template": CHILD_ENVIRONMENT_TEMPLATE,
        "d_paths": list(D_PATHS),
        "deadline_seconds": DEADLINE_SECONDS,
        "environment_keys": list(CHILD_ENV_KEYS),
        "evaluator_argv_prefix": list(EVALUATOR_ARGV_PREFIX),
        "inotify_mask": list(INOTIFY_MASK_NAMES),
        "model_inputs": list(MODEL_LOGICAL_PATHS),
        "operator_vector": list(OPERATOR_ARGV),
        "ordered_run_plan": list(RUN_IDS),
        "publication_rollback": PUBLICATION_ROLLBACK,
        "required_repository_inputs": list(REQUIRED_REPOSITORY_INPUTS),
        "result_enum": list(RESULT_ENUM),
        "result_matrix": {
            result: {"forbidden": list(forbidden), "required": list(required)}
            for result, (required, forbidden) in RESULT_MATRIX.items()
        },
        "trace_capacities": list(TRACE_CAPACITIES),
        "trace_lifecycle": TRACE_LIFECYCLE,
        "v_paths": list(V_PATHS),
    }


@dataclass(frozen=True)
class InotifyEvent:
    path: str
    mask: int
    classification: str


class BoundInputMonitor:
    """Linux inotify monitor for every bound file and ancestor directory."""

    _event_header = struct.Struct("iIII")

    def __init__(
        self, bound_paths: Iterable[Path], ignored_roots: Iterable[Path] = ()
    ) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._fd = int(self._libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC))
        if self._fd < 0:
            error = ctypes.get_errno()
            raise DriftError(f"inotify_init1 failed: errno={error}")
        self._watch_paths: dict[int, Path] = {}
        self.history: list[InotifyEvent] = []
        # Keep both the lexical path and its physical target.  Resolving only
        # the target would miss a repository symlink that is replaced and then
        # restored between checkpoints.  Output roots intentionally do not
        # exist when watches are installed, so their lexical physical-parent
        # derivation must not require the leaf to exist.
        self._bound: set[Path] = set()
        for path in bound_paths:
            lexical = Path(os.path.abspath(path))
            physical = path.resolve(strict=True)
            self._bound.update((lexical, physical))
        self._ignored_roots = tuple(
            Path(os.path.abspath(path)) for path in ignored_roots
        )
        watch_paths: set[Path] = set()
        for bound in self._bound:
            watch_paths.add(bound)
            watch_paths.update(bound.parents)
        for path in sorted(
            watch_paths, key=lambda item: (len(item.parts), item.as_posix())
        ):
            encoded = os.fsencode(path)
            wd = int(self._libc.inotify_add_watch(self._fd, encoded, INOTIFY_MASK))
            if wd < 0:
                error = ctypes.get_errno()
                self.close()
                raise DriftError(f"inotify_add_watch failed for {path}: errno={error}")
            self._watch_paths[wd] = path

    def close(self) -> None:
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "BoundInputMonitor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ignored(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self._ignored_roots)

    def drain(self) -> list[InotifyEvent]:
        events: list[InotifyEvent] = []
        while True:
            try:
                data = os.read(self._fd, 1024 * 1024)
            except BlockingIOError:
                break
            if not data:
                break
            offset = 0
            while offset < len(data):
                wd, mask, _cookie, name_len = self._event_header.unpack_from(
                    data, offset
                )
                offset += self._event_header.size
                raw_name = data[offset : offset + name_len].rstrip(b"\0")
                offset += name_len
                base = self._watch_paths.get(wd)
                if mask & IN_Q_OVERFLOW:
                    events.append(InotifyEvent("", mask, "queue_overflow"))
                    continue
                if base is None or mask & IN_IGNORED:
                    events.append(InotifyEvent("", mask, "ignored_watch"))
                    continue
                path = base / os.fsdecode(raw_name) if raw_name else base
                if self._ignored(path):
                    continue
                filtered = path in self._bound
                ancestor_destructive = any(
                    bound == path or path in bound.parents for bound in self._bound
                ) and bool(
                    mask
                    & (
                        IN_DELETE_SELF
                        | IN_MOVE_SELF
                        | IN_DELETE
                        | IN_MOVED_FROM
                        | IN_MOVED_TO
                    )
                )
                if filtered or ancestor_destructive:
                    events.append(InotifyEvent(path.as_posix(), mask, "bound_mutation"))
        self.history.extend(events)
        return events

    def assert_clean(self) -> None:
        events = self.drain()
        if events:
            raise DriftError(f"bound-input mutation observation: {events!r}")


def _bounded_remaining(
    started: float, now: Callable[[], float] = time.monotonic
) -> float:
    remaining = DEADLINE_SECONDS - (now() - started)
    if remaining <= 0:
        raise TimeoutError("single Phase-A monotonic deadline exhausted")
    return remaining


def _deadline_checked_call(
    started: float,
    clock: Callable[[], float],
    action: Callable[..., Any],
    *args: object,
) -> Any:
    """Run a finalization operation without admitting an over-deadline result."""
    _bounded_remaining(started, now=clock)
    value = action(*args)
    _bounded_remaining(started, now=clock)
    return value


def _write_canonical_fsync(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        payload = canonical_json_file_bytes(value)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short canonical JSON write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_canonical_fsync(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.replacement")
    _write_canonical_fsync(temporary, value)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_execution_constants(contract: Mapping[str, Any], root: Path) -> None:
    if contract.get("execution_constants") != execution_constants(root):
        raise ContractError("execution choices differ from A7/RC1 constants")
    validate_bound_inventory(contract["bound_inputs"])
    if (
        contract.get("sequence_input_digest")
        != contract["bound_inputs"]["sequence"]["digest"]
    ):
        raise ContractError("controller and bound-input sequence digests disagree")


def result_artifact_sets(result: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        return RESULT_MATRIX[result]
    except KeyError as exc:
        raise ContractError(f"unrecognized controller result: {result!r}") from exc


def classify_result(
    *,
    provenance_ok: bool,
    build_ok: bool,
    extension_ok: bool,
    runners_ok: bool,
    timed_out: bool,
    serialization_ok: bool,
    artifacts_ok: bool,
    classified_execution: bool,
    policy_equal: bool,
    packets_valid: bool,
) -> str:
    """A7.7 first-applicable-row result selection."""
    if not provenance_ok:
        return "provenance_invalid"
    if not build_ok:
        return "build_failed"
    if not extension_ok:
        return "extension_load_failed"
    if not runners_ok:
        return "runner_nonzero"
    if timed_out:
        return "runner_timeout"
    if not serialization_ok:
        return "serialization_failed"
    if not artifacts_ok:
        return "artifact_missing_or_unreadable"
    if not classified_execution:
        return "unclassified_execution_failure"
    if not policy_equal:
        return "capture_perturbs_policy"
    if not packets_valid:
        return "packet_invalid"
    return "phase_a_pass"


def _parse_no_options(argv: Sequence[str]) -> bool:
    if tuple(argv) in (("-h",), ("--help",)):
        return False
    if argv:
        raise ContractError("controller accepts no positional arguments or options")
    return True


def _classify_landing_candidates(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Independently classify every v3 landing candidate (discovery-only).

    This is the exact per-candidate classification the controller relies on
    during input discovery, factored out without the exactly-one-current-landing
    requirement so the non-authoritative qualification dry-run can exercise the
    real path over a mixed-version evidence tree.  Historical artifacts (which
    predate ``build_tool_binding``) and the current authoritative artifact must
    all classify without error; only a genuinely malformed candidate raises.
    """
    evidence = root / "docs/modules/semantic/research/evidence"
    if evidence.is_symlink() or not evidence.is_dir():
        raise ContractError("evidence root is not a physical directory")
    candidates = sorted(evidence.glob("**/h0_preseal_freeze_v3.json"))
    import verify_h0_preseal_freeze

    return [
        (
            candidate,
            verify_h0_preseal_freeze.verify_current_landing_candidate(candidate, root),
        )
        for candidate in candidates
    ]


def _discover_controller_input(root: Path) -> tuple[Path, dict[str, Any]]:
    """Select the sole v3 artifact whose independently verified landing is HEAD."""
    try:
        classified = _classify_landing_candidates(root)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        raise ContractError(
            f"v3 landing candidate rejected by independent verifier: {exc}"
        ) from exc
    current = [
        candidate
        for candidate, report in classified
        if report.get("matches_current_checkout") is True
    ]
    if len(current) != 1:
        raise ContractError(
            "expected exactly one current-HEAD h0_preseal_freeze_v3 landing, "
            f"found {len(current)} among {len(classified)} candidates"
        )
    freeze_path = current[0]
    if (
        freeze_path.is_symlink()
        or not freeze_path.is_file()
        or freeze_path.resolve(strict=True) != freeze_path.absolute()
    ):
        raise ContractError("h0_preseal_freeze_v3.json is not a physical regular file")
    freeze = read_canonical_json(freeze_path)
    if (
        not isinstance(freeze, dict)
        or freeze.get("freeze_schema_version") != "h0_preseal_freeze_v3"
        or freeze.get("complete") is not True
    ):
        raise ContractError("candidate freeze is not h0_preseal_freeze_v3")
    contract = freeze.get("phase_a_controller_input")
    validate_schema_document(contract, "controller_input")
    assert isinstance(contract, dict)
    # The execution schema deliberately leaves ``build_tool_binding`` optional so
    # historical evidence still validates; the authoritative current artifact
    # must carry the full canonical member set, matching the independent
    # pre-seal verifier's exact-member check on the selected candidate.
    if set(contract) != CONTROLLER_INPUT_MEMBERS:
        raise ContractError(
            "selected controller input has a non-canonical member set: "
            f"{sorted(set(contract) ^ CONTROLLER_INPUT_MEMBERS)}"
        )
    if freeze.get("instrumentation_head") != contract.get("instrumentation_head"):
        raise ContractError("v3 freeze/controller instrumentation heads differ")
    landing = contract.get("authority_landing")
    if (
        not isinstance(landing, dict)
        or landing.get("artifact_path") != freeze_path.relative_to(root).as_posix()
    ):
        raise ContractError("v3 freeze is not at its RC2 deterministic artifact path")
    return freeze_path, contract


def _verify_authority_landing(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, str]:
    """Delegate RC2's independent I -> F -> S verification before any build read."""
    landing = contract.get("authority_landing")
    if not isinstance(landing, Mapping) or not isinstance(
        landing.get("artifact_path"), str
    ):
        raise ContractError("controller input has no authority-landing descriptor")
    artifact_path = root / str(landing["artifact_path"])
    try:
        import verify_h0_preseal_freeze

        report = verify_h0_preseal_freeze.verify_artifact_path(
            artifact_path, root, require_complete=True, verify_landing=True
        )
        relation = report["landing"]
        if not isinstance(
            relation, dict
        ):  # defensive boundary for the independent verifier
            raise ContractError("independent v3 verifier returned no landing relation")
        return {key: str(value) for key, value in relation.items()}
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError(f"authority landing verification failed: {exc}") from exc


def preflight_controller_input(
    contract: Mapping[str, Any],
    root: Path,
    *,
    started: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    if started is None:
        started = clock()
    physical_root = root.resolve(strict=True)
    if physical_root != root or Path.cwd().resolve(strict=True) != physical_root:
        raise ContractError("physical cwd does not equal repository root")
    if contract.get("repository_root") != root.as_posix():
        raise ContractError("controller input repository_root mismatch")
    for key in os.environ:
        if (
            key in FORBIDDEN_SELECTOR_KEYS
            or key.startswith("MLFLOW_")
            or key.startswith("H0_")
        ):
            raise ContractError(f"forbidden execution-selector environment key: {key}")
    validate_execution_constants(contract, root)
    head = contract["instrumentation_head"]
    landing = _verify_authority_landing(root, contract)
    if landing.get("instrumentation_head") != head:
        raise ContractError(
            "authority landing and controller instrumentation heads differ"
        )
    # Amendment 10: after topology is proven, pin continuous monitoring to S.
    verify_owner_authority_overlay_s_bytes(contract, started=started, clock=clock)
    if select_gpu_record(nvml_gpu_inventory()) != contract["gpu"]:
        raise ContractError(
            "lexicographically selected NVML GPU identity differs from v3"
        )
    git_path = require_canonical_absolute(
        contract["tool_paths"]["git"], directory=False
    )
    try:
        top = _run_auxiliary_subprocess(
            ["git", "rev-parse", "--show-toplevel"],
            executable=git_path,
            cwd=root,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            started=started,
            stage="git repository-root preflight",
            clock=clock,
            text=True,
        ).stdout.strip()
        actual_head = _run_auxiliary_subprocess(
            ["git", "rev-parse", "HEAD"],
            executable=git_path,
            cwd=root,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            started=started,
            stage="git HEAD preflight",
            clock=clock,
            text=True,
        ).stdout.strip()
        status_bytes = _run_auxiliary_subprocess(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            executable=git_path,
            cwd=root,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            started=started,
            stage="git cleanliness preflight",
            clock=clock,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ContractError("git provenance preflight exited nonzero") from exc
    if (
        top != root.as_posix()
        or actual_head != landing.get("execution_checkout")
        or status_bytes != b""
    ):
        raise ContractError("checkout root/head/cleanliness provenance mismatch")
    assert_no_preexisting_build_tree(root)
    evidence_root = root / contract["evidence_root"]
    incomplete_root = root / contract["incomplete_root"]
    expected_root = f"docs/modules/semantic/research/evidence/h0_phase_a_{head}"
    if (
        contract["evidence_root"] != expected_root
        or contract["incomplete_root"] != expected_root + ".incomplete"
    ):
        raise ContractError("evidence-root derivation mismatch")
    if evidence_root.exists() or incomplete_root.exists():
        raise ContractError("stale final or incomplete evidence root")
    validate_build_tool_binding(
        contract,
        root=root,
        started=started,
        clock=clock,
    )


def _create_run_directories(incomplete: Path, run_id: str) -> Path:
    run_dir = incomplete / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("home", "tmp", "xdg-cache"):
        (run_dir / "_env" / name).mkdir(parents=True, exist_ok=False)
    return run_dir


def build_environment(contract: Mapping[str, Any]) -> dict[str, str]:
    """Rebuild the sole admitted environment for both frozen build vectors."""
    root = Path(contract["repository_root"])
    environment_root = root / contract["incomplete_root"] / "_build_env"
    return {
        "CUDACXX": _bound_nvcc_path(contract).as_posix(),
        "HOME": (environment_root / "home").as_posix(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": build_tool_environment_path(root),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": (environment_root / "tmp").as_posix(),
        "TZ": "UTC",
        "XDG_CACHE_HOME": (environment_root / "xdg-cache").as_posix(),
    }


def _bound_nvcc_path(contract: Mapping[str, Any]) -> Path:
    """Derive the frozen CUDA compiler from its bound physical identity."""
    try:
        declared = str(contract["tool_paths"]["nvcc"])
    except (KeyError, TypeError) as exc:
        raise ContractError("controller input has no selected nvcc path") from exc
    selected = require_canonical_absolute(declared, directory=False)
    records = [
        record
        for record in contract["bound_inputs"]["tool_runtime"]
        if record.get("logical_path") == declared
    ]
    if len(records) != 1:
        raise ContractError("selected nvcc is absent or ambiguous in tool_runtime")
    record = records[0]
    try:
        frozen = require_canonical_absolute(str(record["realpath"]), directory=False)
        data = frozen.read_bytes()
        expected = (int(record["length"]), str(record["sha256"]))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ContractError("selected nvcc bound identity is malformed") from exc
    if frozen != selected or (len(data), sha256_bytes(data)) != expected:
        raise DriftError("selected nvcc differs from its bound tool_runtime identity")
    return frozen


def _create_build_environment(contract: Mapping[str, Any]) -> dict[str, str]:
    environment = build_environment(contract)
    if tuple(environment) != BUILD_ENVIRONMENT_KEYS:
        raise ContractError("build environment key order drift")
    environment_root = Path(environment["HOME"]).parent
    environment_root.mkdir(parents=False, exist_ok=False)
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME"):
        path = Path(environment[key])
        path.mkdir(exist_ok=False)
        if path.is_symlink() or path.resolve(strict=True) != path.absolute():
            raise ContractError(f"non-physical build environment directory: {key}")
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    terminate_tree = getattr(process, "terminate_tree", None)
    if callable(terminate_tree):
        terminate_tree()
        return
    # Every controller-owned subprocess starts a fresh session.  Kill the
    # process group even when its leader has already exited: a helper can fork,
    # close the captured pipes, and otherwise leave a descendant behind after
    # the leader becomes waitable.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _wait_with_monitor(
    process: subprocess.Popen[bytes],
    *,
    started: float,
    monitor: BoundInputMonitor | None,
    stage: str,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    if monitor is None:
        return process.wait(timeout=_bounded_remaining(started, now=clock))
    while True:
        events = monitor.drain()
        if events:
            _terminate_process_group(process)
            raise DriftError(
                f"bound-input mutation while {stage} was active: {events!r}"
            )
        try:
            remaining = _bounded_remaining(started, now=clock)
        except TimeoutError:
            _terminate_process_group(process)
            raise
        try:
            return process.wait(timeout=min(remaining, 0.1))
        except subprocess.TimeoutExpired:
            continue


def _run_auxiliary_subprocess(
    vector: Sequence[str],
    *,
    executable: Path,
    cwd: Path,
    env: Mapping[str, str],
    started: float,
    monitor: BoundInputMonitor | None = None,
    stage: str,
    clock: Callable[[], float] = time.monotonic,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run one bounded helper with captured output and no surviving process tree."""
    _bounded_remaining(started, now=clock)
    process = subprocess.Popen(
        list(vector),
        executable=executable,
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        close_fds=True,
        start_new_session=True,
        text=text,
    )
    while True:
        if monitor is not None:
            events = monitor.drain()
            if events:
                _terminate_process_group(process)
                raise DriftError(
                    f"bound-input mutation while {stage} was active: {events!r}"
                )
        try:
            remaining = _bounded_remaining(started, now=clock)
        except TimeoutError:
            _terminate_process_group(process)
            raise
        try:
            stdout, stderr = process.communicate(timeout=min(remaining, 0.1))
        except subprocess.TimeoutExpired:
            continue
        if monitor is not None:
            events = monitor.drain()
            if events:
                _terminate_process_group(process)
                raise DriftError(
                    f"bound-input mutation while {stage} completed: {events!r}"
                )
        try:
            _bounded_remaining(started, now=clock)
        except TimeoutError:
            _terminate_process_group(process)
            raise
        completed = subprocess.CompletedProcess(
            list(vector), process.returncode, stdout, stderr
        )
        if process.returncode:
            _terminate_process_group(process)
            raise subprocess.CalledProcessError(
                process.returncode,
                list(vector),
                output=stdout,
                stderr=stderr,
            )
        return completed


def _runtime_attestation_details(
    process: Any, runtime_plan: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str, bool]:
    attestation = process.runtime_attestation()
    runtime_digest = sha256_bytes(canonical_json_bytes(attestation))
    valid = bool(
        attestation.get("state") == "complete"
        and attestation.get("confinement_plan_digest") == runtime_plan["digest"]
        and attestation.get("backend") == RUNTIME_CONFINEMENT_BACKEND
        and attestation.get("ingress_policy") == RUNTIME_INGRESS_POLICY
        and attestation.get("trace_scope") == list(RUNTIME_TRACE_SCOPE)
        and attestation.get("installed_before_exec") is True
        and attestation.get("process_tree_terminal") is True
        and attestation.get("denial_probe_observed") is True
        and not attestation.get("violations")
    )
    return attestation, runtime_digest, valid


def _collect_runtime_attestation(
    process: Any,
    runtime_plan: Mapping[str, Any],
    run_id: str,
    attestations: dict[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str, bool]:
    attestation, runtime_digest, valid = _runtime_attestation_details(
        process, runtime_plan
    )
    if run_id in attestations:
        raise ContractError("duplicate runtime input attestation")
    attestations[run_id] = attestation
    return attestation, runtime_digest, valid


def launch_child(
    contract: Mapping[str, Any],
    run_id: str,
    *,
    started: float,
    build_identity: Mapping[str, Any] | None = None,
    monitor: BoundInputMonitor | None = None,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    attestations: dict[str, Mapping[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Create one fixed invocation and execute one RC1.1 child process."""
    if attestations is None:
        attestations = {}
    root = Path(contract["repository_root"])
    incomplete = root / contract["incomplete_root"]
    run_dir = incomplete / "runs" / run_id
    invocation_path = run_dir / "invocation.json"
    if run_dir.exists():
        prior = read_canonical_json(invocation_path)
        validate_schema_document(prior, "child_invocation")
        if prior.get("state") != "not_run" or prior.get("run_id") != run_id:
            raise ContractError(
                f"pre-existing child slot is not canonical not_run: {run_id}"
            )
    else:
        run_dir = _create_run_directories(incomplete, run_id)
    env = child_environment(
        root, run_dir, gpu_uuid=contract["gpu"]["uuid"], **contract["library_dirs"]
    )
    vector = child_argv(root, run_id)
    expected_vector = contract["execution_constants"]["child_vectors"][
        RUN_IDS.index(run_id)
    ]
    if list(vector) != expected_vector:
        raise ContractError("child command vector mismatch")
    child_contract = {
        "bound_inputs_digest": contract["bound_inputs"]["digest"],
        "capture_run_uuid": str(uuid.uuid4()),
        "confinement_plan_digest": None,
        "confinement_probe_passed": None,
        "document_type": "child_invocation",
        "environment": env,
        "environment_digest": environment_digest(env),
        "evaluator_argv": list(evaluator_argv(run_dir)),
        "incomplete_root": contract["incomplete_root"],
        "instrumentation_head": contract["instrumentation_head"],
        "result": None,
        "run_id": run_id,
        "runtime_inputs_digest": None,
        "schema": CHILD_SCHEMA,
        "state": "running_interrupted",
        "vector": list(vector),
    }
    validate_schema_document(child_contract, "child_invocation")
    if invocation_path.exists():
        _replace_canonical_fsync(invocation_path, child_contract)
    else:
        _write_canonical_fsync(invocation_path, child_contract)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    for log_path in (stdout_path, stderr_path):
        if log_path.exists() or log_path.is_symlink():
            if log_path.is_symlink() or log_path.read_bytes() != b"NOT_RUN\n":
                raise ContractError(f"child log placeholder drift: {log_path}")
            log_path.unlink()
    confinement_error_type: Any = ()
    with (
        open(os.devnull, "rb", buffering=0) as stdin,
        open(stdout_path, "xb", buffering=0) as stdout,
        open(stderr_path, "xb", buffering=0) as stderr,
    ):
        runtime_plan: Mapping[str, Any] | None = None
        if popen_factory is subprocess.Popen:
            if build_identity is None:
                raise ContractError(
                    "runtime confinement requires complete build identity"
                )
            tools = SCHEMA_PATH.parent
            if tools.as_posix() not in sys.path:
                sys.path.insert(0, tools.as_posix())
            import h0_runtime_confinement

            confinement_error_type = h0_runtime_confinement.ConfinementError
            denial_probe = incomplete / "_runtime_confinement_denial_probe"
            if not denial_probe.exists():
                _write_bytes_fsync(denial_probe, b"MUST_BE_DENIED\n")
            if denial_probe.is_symlink() or not denial_probe.is_file():
                raise ContractError("runtime confinement denial probe is not regular")
            try:
                runtime_plan = h0_runtime_confinement.build_plan(
                    root=root,
                    incomplete=incomplete,
                    inventory=contract["bound_inputs"],
                    build_identity=build_identity,
                    denial_probe=denial_probe,
                    run_ids=RUN_IDS,
                )
                child_contract["confinement_plan_digest"] = runtime_plan["digest"]
                _replace_canonical_fsync(invocation_path, child_contract)
                process = h0_runtime_confinement.spawn_confined(
                    vector,
                    cwd=root,
                    env=env,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    plan=runtime_plan,
                )
            except (h0_runtime_confinement.ConfinementError, OSError) as exc:
                interrupted = dict(child_contract)
                interrupted["result"] = "provenance_invalid"
                interrupted["state"] = "failed"
                _replace_canonical_fsync(invocation_path, interrupted)
                raise DriftError(f"runtime confinement setup failed: {exc}") from exc
        else:
            process = popen_factory(
                vector,
                executable=vector[0],
                cwd=root,
                env=env,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
        try:
            returncode = _wait_with_monitor(
                process,
                started=started,
                monitor=monitor,
                stage=f"child {run_id}",
                clock=clock,
            )
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            _terminate_process_group(process)
            interrupted = dict(child_contract)
            if runtime_plan is not None:
                _attestation, runtime_digest, valid_runtime = (
                    _collect_runtime_attestation(
                        process, runtime_plan, run_id, attestations
                    )
                )
                current = read_canonical_json(invocation_path)
                interrupted["confinement_probe_passed"] = current.get(
                    "confinement_probe_passed"
                )
                interrupted["runtime_inputs_digest"] = runtime_digest
                if (
                    not valid_runtime
                    or interrupted["confinement_probe_passed"] is not True
                ):
                    interrupted["result"] = "provenance_invalid"
                    interrupted["state"] = "failed"
                    _replace_canonical_fsync(invocation_path, interrupted)
                    raise DriftError(
                        "runtime confinement failed while child timed out"
                    ) from exc
            interrupted["result"] = "runner_timeout"
            interrupted["state"] = "failed"
            _replace_canonical_fsync(invocation_path, interrupted)
            raise TimeoutError("child exceeded the single Phase-A deadline") from exc
        except DriftError:
            _terminate_process_group(process)
            interrupted = dict(child_contract)
            if runtime_plan is not None:
                _attestation, runtime_digest, _valid_runtime = (
                    _collect_runtime_attestation(
                        process, runtime_plan, run_id, attestations
                    )
                )
                current = read_canonical_json(invocation_path)
                interrupted["confinement_probe_passed"] = current.get(
                    "confinement_probe_passed"
                )
                interrupted["runtime_inputs_digest"] = runtime_digest
            interrupted["result"] = "provenance_invalid"
            interrupted["state"] = "failed"
            _replace_canonical_fsync(invocation_path, interrupted)
            raise
        except confinement_error_type as exc:
            try:
                _terminate_process_group(process)
            except confinement_error_type:
                pass
            interrupted = dict(child_contract)
            interrupted["result"] = "provenance_invalid"
            interrupted["state"] = "failed"
            _replace_canonical_fsync(invocation_path, interrupted)
            raise DriftError(f"runtime confinement supervision failed: {exc}") from exc
    runtime_attestation: Mapping[str, Any] | None = None
    if runtime_plan is not None:
        runtime_attestation, runtime_digest, valid_runtime = (
            _collect_runtime_attestation(process, runtime_plan, run_id, attestations)
        )
        if not valid_runtime:
            rejected = dict(child_contract)
            rejected["confinement_probe_passed"] = False
            rejected["result"] = "provenance_invalid"
            rejected["runtime_inputs_digest"] = runtime_digest
            rejected["state"] = "failed"
            _replace_canonical_fsync(invocation_path, rejected)
            return 2, rejected
    result = read_canonical_json(invocation_path)
    if runtime_attestation is not None:
        if result.get("confinement_probe_passed") is not True:
            result["result"] = "provenance_invalid"
            result["state"] = "failed"
            returncode = 2
        result["runtime_inputs_digest"] = sha256_bytes(
            canonical_json_bytes(runtime_attestation)
        )
        _replace_canonical_fsync(invocation_path, result)
    validate_schema_document(result, "child_invocation")
    if (
        result.get("state") not in {"failed", "completed"}
        or result.get("result") is None
    ):
        raise ContractError("child left malformed or interrupted structured output")
    if result.get("vector") != list(vector) or result.get(
        "environment_digest"
    ) != environment_digest(env):
        raise ContractError("child output command/environment drift")
    return returncode, result


def _run_build_vector(
    contract: Mapping[str, Any],
    vector: tuple[str, ...],
    stdout_path: Path,
    stderr_path: Path,
    *,
    started: float,
    monitor: BoundInputMonitor | None = None,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    uv_path = require_canonical_absolute(contract["tool_paths"]["uv"], directory=False)
    environment = build_environment(contract)
    if tuple(environment) != BUILD_ENVIRONMENT_KEYS:
        raise ContractError("build environment key order drift")
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME"):
        require_canonical_absolute(environment[key], directory=True)
    with (
        open(os.devnull, "rb", buffering=0) as stdin,
        open(stdout_path, "xb", buffering=0) as stdout,
        open(stderr_path, "xb", buffering=0) as stderr,
    ):
        process = popen_factory(
            vector,
            executable=uv_path,
            cwd=contract["repository_root"],
            env=environment,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
        try:
            return _wait_with_monitor(
                process,
                started=started,
                monitor=monitor,
                stage="build",
                clock=clock,
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise TimeoutError("build exceeded the single Phase-A deadline") from exc


def _tool_version(
    path: Path,
    *,
    root: Path,
    started: float,
    monitor: BoundInputMonitor | None,
    clock: Callable[[], float],
) -> str:
    result = _run_auxiliary_subprocess(
        [path.as_posix(), "--version"],
        executable=path,
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        started=started,
        monitor=monitor,
        stage=f"tool version identity {path}",
        clock=clock,
        text=True,
    )
    value = (result.stdout + result.stderr).strip()
    if not value:
        raise ContractError(f"tool emitted no version identity: {path}")
    return value


def _elf_build_id(
    path: Path,
    readelf: Path,
    *,
    root: Path,
    started: float,
    monitor: BoundInputMonitor | None,
    clock: Callable[[], float],
) -> str:
    result = _run_auxiliary_subprocess(
        ["readelf", "-n", path.as_posix()],
        executable=readelf,
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        started=started,
        monitor=monitor,
        stage=f"ELF build-id identity {path}",
        clock=clock,
        text=True,
    )
    matches = [
        line.split("Build ID:", 1)[1].strip()
        for line in result.stdout.splitlines()
        if "Build ID:" in line
    ]
    if (
        len(matches) != 1
        or not matches[0]
        or any(char not in "0123456789abcdef" for char in matches[0].lower())
    ):
        raise ContractError(f"ELF GNU build ID is absent or ambiguous: {path}")
    return matches[0].lower()


def discover_python_interpreter_runtime_paths(python: Path) -> list[str]:
    """Discover logical paths the frozen interpreter needs under confinement.

    The Phase-A extension-load vector runs a real CPython process.  The frozen
    ``tool_runtime`` inventory must therefore include the interpreter's
    base_prefix stdlib, venv bootstrap files, and the small set of host files
    CPython opens during ``-I -B -c`` startup.  Paths are reported in the form
    the interpreter itself uses (including symlink path forms under uv-managed
    base prefixes).

    The ``python`` argument may be a venv symlink (common under uv/CI); the
    process is launched through that path.  Freeze-time physical-file admission
    remains a separate require_canonical_absolute / non-symlink check.
    """
    python = Path(python)
    if not python.is_file():
        raise ContractError(f"python interpreter is absent: {python}")
    script = (
        "import pathlib, site, sys\n"
        "paths = set()\n"
        # Only walk the interpreter base_prefix stdlib tree.  Never rglob the
        # venv prefix (sys.prefix): that would expand the entire site-packages
        # universe into tool_runtime.
        "base = pathlib.Path(sys.base_prefix)\n"
        "if base.is_dir():\n"
        "    for path in base.rglob('*'):\n"
        "        if path.is_file() and not path.is_symlink():\n"
        "            paths.add(path.as_posix())\n"
        "cfg = pathlib.Path(sys.executable).resolve().parent.parent / 'pyvenv.cfg'\n"
        "if cfg.is_file() and not cfg.is_symlink():\n"
        "    paths.add(cfg.as_posix())\n"
        "for entry in site.getsitepackages():\n"
        "    base = pathlib.Path(entry)\n"
        "    if not base.is_dir():\n"
        "        continue\n"
        "    for path in base.glob('*.pth'):\n"
        "        if path.is_file() and not path.is_symlink():\n"
        "            paths.add(path.as_posix())\n"
        "    for name in ('_virtualenv.py',):\n"
        "        candidate = base / name\n"
        "        if candidate.is_file() and not candidate.is_symlink():\n"
        "            paths.add(candidate.as_posix())\n"
        "        pyc = base / '__pycache__' / (name.replace('.py', '.cpython-312.pyc'))\n"
        "        if pyc.is_file() and not pyc.is_symlink():\n"
        "            paths.add(pyc.as_posix())\n"
        "    hack = base / '_distutils_hack'\n"
        "    if hack.is_dir():\n"
        "        for path in hack.rglob('*'):\n"
        "            if path.is_file() and not path.is_symlink():\n"
        "                paths.add(path.as_posix())\n"
        "for extra in (\n"
        "    '/etc/ld.so.cache',\n"
        "    '/usr/share/locale/locale.alias',\n"
        "    '/usr/lib/locale/locale-archive',\n"
        "    '/usr/share/zoneinfo/UTC',\n"
        "    '/etc/passwd', '/etc/group', '/etc/nsswitch.conf',\n"
        "    '/etc/host.conf', '/etc/hosts', '/etc/resolv.conf',\n"
        "    '/etc/gnutls/config',\n"
        "):\n"
        "    path = pathlib.Path(extra)\n"
        "    if path.is_file() and not path.is_symlink():\n"
        "        paths.add(path.as_posix())\n"
        "gconv = pathlib.Path('/usr/lib/gconv')\n"
        "if gconv.is_dir():\n"
        "    for path in gconv.iterdir():\n"
        "        if path.is_file() and not path.is_symlink():\n"
        "            paths.add(path.as_posix())\n"
        "c_utf8 = pathlib.Path('/usr/lib/locale/C.utf8')\n"
        "if c_utf8.is_dir():\n"
        "    for path in c_utf8.rglob('*'):\n"
        "        if path.is_file() and not path.is_symlink():\n"
        "            paths.add(path.as_posix())\n"
        "print('\\n'.join(sorted(paths)))\n"
    )
    result = subprocess.run(
        [python.as_posix(), "-I", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in result.stdout.splitlines() if line.startswith("/")]
    if not paths:
        raise ContractError("python interpreter runtime discovery returned no paths")
    return paths


def _dynamic_dependencies(
    path: Path,
    ldd: Path,
    *,
    root: Path,
    started: float,
    monitor: BoundInputMonitor | None,
    clock: Callable[[], float],
) -> list[dict[str, Any]]:
    result = _run_auxiliary_subprocess(
        ["ldd", path.as_posix()],
        executable=ldd,
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        started=started,
        monitor=monitor,
        stage=f"dynamic dependency identity {path}",
        clock=clock,
        text=True,
    )
    records: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or "linux-vdso" in stripped:
            continue
        if "not found" in stripped:
            raise ContractError(f"unresolved dynamic dependency for {path}: {stripped}")
        candidate = (
            stripped.split("=>", 1)[1].strip().split(" ", 1)[0]
            if "=>" in stripped
            else stripped.split(" ", 1)[0]
        )
        if not candidate.startswith("/"):
            continue
        dependency = Path(candidate).resolve(strict=True)
        data = dependency.read_bytes()
        records.append(
            {
                "length": len(data),
                "path": candidate,
                "realpath": dependency.as_posix(),
                "sha256": sha256_bytes(data),
            }
        )
    # Recursively close the ldd graph so transitive NEEDED members are admitted.
    pending = [Path(record["realpath"]) for record in records]
    seen = {record["realpath"] for record in records}
    while pending:
        source = pending.pop(0)
        nested = _run_auxiliary_subprocess(
            ["ldd", source.as_posix()],
            executable=ldd,
            cwd=root,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            started=started,
            monitor=monitor,
            stage=f"dynamic dependency identity {source}",
            clock=clock,
            text=True,
        )
        for line in nested.stdout.splitlines():
            stripped = line.strip()
            if not stripped or "linux-vdso" in stripped or "not found" in stripped:
                continue
            candidate = (
                stripped.split("=>", 1)[1].strip().split(" ", 1)[0]
                if "=>" in stripped
                else stripped.split(" ", 1)[0]
            )
            if not candidate.startswith("/"):
                continue
            dependency = Path(candidate).resolve(strict=True)
            realpath = dependency.as_posix()
            if realpath in seen:
                continue
            data = dependency.read_bytes()
            records.append(
                {
                    "length": len(data),
                    "path": candidate,
                    "realpath": realpath,
                    "sha256": sha256_bytes(data),
                }
            )
            seen.add(realpath)
            pending.append(dependency)
    # One-hop sibling libraries that are commonly dlopened by members of the
    # static ldd graph (e.g. libtbb → libtbbmalloc) but never appear in NEEDED.
    sibling_names = {
        "libtbb.so.12": ("libtbbmalloc.so.2", "libtbbmalloc_proxy.so.2"),
        "libtbb.so.12.19": ("libtbbmalloc.so.2", "libtbbmalloc_proxy.so.2"),
    }
    seen_paths = {record["realpath"] for record in records}
    for record in list(records):
        name = Path(record["path"]).name
        for sibling in sibling_names.get(name, ()):
            candidate = Path(record["path"]).with_name(sibling)
            if (
                not candidate.exists()
                and Path(record["realpath"]).parent.joinpath(sibling).exists()
            ):
                candidate = Path(record["realpath"]).parent / sibling
            if not candidate.exists():
                continue
            try:
                real = candidate.resolve(strict=True)
                data = real.read_bytes()
            except OSError:
                continue
            if real.as_posix() in seen_paths:
                continue
            records.append(
                {
                    "length": len(data),
                    "path": candidate.as_posix(),
                    "realpath": real.as_posix(),
                    "sha256": sha256_bytes(data),
                }
            )
            seen_paths.add(real.as_posix())
    records.sort(key=lambda record: record["path"].encode("utf-8"))
    if len({record["path"] for record in records}) != len(records):
        raise ContractError(f"duplicate dynamic dependency identity: {path}")
    return records


def _runtime_maps_dependencies(
    *,
    python: Path,
    extension: Path,
    plugin: Path,
    root: Path,
    library_path: str,
    started: float,
    monitor: BoundInputMonitor | None,
    clock: Callable[[], float],
) -> list[dict[str, Any]]:
    """Capture shared objects actually mapped while loading extension+plugin.

    Complements ``ldd``: some libraries (e.g. tbbmalloc) are dlopened at load
    time and never appear in the static NEEDED graph.
    """
    script = (
        "import ctypes, pathlib, sys\n"
        # Match the controller extension-load vector: import the extension as a
        # Python module (runs module-init dlopen paths) then CDLL the plugin.
        f"sys.path.insert(0, {extension.parent.as_posix()!r})\n"
        "import saccade_tracking_ext\n"
        "assert pathlib.Path(saccade_tracking_ext.__file__).resolve() == "
        f"pathlib.Path({extension.as_posix()!r}).resolve()\n"
        f"ctypes.CDLL({plugin.as_posix()!r}, mode=ctypes.RTLD_LOCAL)\n"
        "print(pathlib.Path('/proc/self/maps').read_text())\n"
    )
    result = _run_auxiliary_subprocess(
        [python.as_posix(), "-I", "-B", "-c", script],
        executable=python,
        cwd=root,
        env={
            "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "LD_LIBRARY_PATH": library_path,
            "SACCADE_BUILD_PATH": (root / "build/h0_phase_a").as_posix(),
            "PYTHONNOUSERSITE": "1",
        },
        started=started,
        monitor=monitor,
        stage="runtime maps dependency identity",
        clock=clock,
        text=True,
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        candidate = fields[5].split(" (deleted)", 1)[0]
        path = Path(candidate)
        # Maps closure is for shared libraries only; interpreter/stdlib belong
        # in tool_runtime, not build-artifact dynamic_dependencies.
        name = path.name
        if not (
            name.endswith(".so")
            or ".so." in name
            or name.endswith(".so.1")
            or ".so." in path.as_posix()
        ):
            continue
        if not path.is_file() and not path.is_symlink():
            continue
        try:
            real = path.resolve(strict=True)
            if not real.is_file():
                continue
            data = real.read_bytes()
        except OSError:
            continue
        realpath = real.as_posix()
        if realpath in seen:
            continue
        if realpath in {
            extension.resolve(strict=True).as_posix(),
            plugin.resolve(strict=True).as_posix(),
        }:
            continue
        seen.add(realpath)
        records.append(
            {
                "length": len(data),
                "path": candidate,
                "realpath": realpath,
                "sha256": sha256_bytes(data),
            }
        )
    records.sort(key=lambda record: record["path"].encode("utf-8"))
    return records


def build_tool_environment_path(root: Path) -> str:
    """Return the exact PATH visible to both authoritative build vectors.

    This is intentionally not the controller process's ambient PATH.  CMake's
    compiler discovery is therefore tied to the same namespace that `uv run
    --frozen cmake` receives at execution time.
    """
    physical_root = require_canonical_absolute(root.as_posix(), directory=True)
    return f"{physical_root}/.venv/bin:/usr/bin:/bin"


def _physical_executable_in_path(command: str, search_path: str) -> Path:
    found = shutil.which(command, path=search_path)
    if not found:
        raise ContractError(
            f"required build tool is absent from authoritative PATH: {command}"
        )
    candidate = Path(found).resolve(strict=True)
    details = candidate.stat(follow_symlinks=False)
    if candidate.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise ContractError(
            f"authoritative build tool is not a physical regular file: {command}"
        )
    return candidate


def _binding_digest(value: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "build_environment_path": value["build_environment_path"],
                "loader_closure": value["loader_closure"],
                "resolver": value["resolver"],
                "schema": value["schema"],
                "tools": value["tools"],
            }
        )
    )


def resolve_build_tool_binding(
    root: Path,
    *,
    ldd_path: Path,
    started: float | None = None,
    monitor: BoundInputMonitor | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Resolve the exact CMake/C++ build-tool and loader closure at freeze time.

    The same resolver is also called by controller preflight and controlled-host
    qualification.  It resolves only the two commands that the admitted build
    vectors delegate to (`cmake` and CMake's default `c++` compiler driver),
    then recursively closes their actual `ldd` graph.  Every member is a
    physical, hashed external input; an incomplete, duplicate, or substituted
    closure is a provenance failure rather than a best-effort observation.
    """
    physical_root = require_canonical_absolute(root.as_posix(), directory=True)
    ldd = require_canonical_absolute(ldd_path.as_posix(), directory=False)
    if started is None:
        started = clock()
    search_path = build_tool_environment_path(physical_root)
    tools: list[dict[str, Any]] = []
    tool_paths: list[Path] = []
    for role, command in BUILD_TOOL_ROLES:
        path = _physical_executable_in_path(command, search_path)
        tools.append(
            {
                "command": command,
                "record": external_input_record(physical_root, path.as_posix()),
                "role": role,
            }
        )
        tool_paths.append(path)
    if [tool["role"] for tool in tools] != [
        role for role, _command in BUILD_TOOL_ROLES
    ]:
        raise ContractError("build-tool role order drift")
    tool_realpaths = [tool["record"]["realpath"] for tool in tools]
    if len(tool_realpaths) != len(set(tool_realpaths)):
        raise ContractError("build-tool binding has duplicate physical tools")

    closure_by_realpath: dict[str, dict[str, Any]] = {}
    pending = list(tool_paths)
    seen = {path.as_posix() for path in tool_paths}
    while pending:
        source = pending.pop(0)
        for dependency in _dynamic_dependencies(
            source,
            ldd,
            root=physical_root,
            started=started,
            monitor=monitor,
            clock=clock,
        ):
            dependency_path = require_canonical_absolute(
                str(dependency["realpath"]), directory=False
            )
            realpath = dependency_path.as_posix()
            if realpath in tool_realpaths:
                raise ContractError(
                    "build-tool loader closure duplicates a primary build tool"
                )
            record = external_input_record(physical_root, realpath)
            if (
                record["length"],
                record["sha256"],
            ) != (dependency["length"], dependency["sha256"]):
                raise DriftError(
                    f"build-tool loader changed while resolving closure: {realpath}"
                )
            prior = closure_by_realpath.setdefault(realpath, record)
            if prior != record:
                raise DriftError(f"inconsistent build-tool loader identity: {realpath}")
            if realpath not in seen:
                seen.add(realpath)
                pending.append(dependency_path)
    closure = sorted(
        closure_by_realpath.values(),
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    if not closure:
        raise ContractError("build-tool loader/shared-library closure is empty")
    binding: dict[str, Any] = {
        "build_environment_path": search_path,
        "digest": "",
        "loader_closure": closure,
        "resolver": BUILD_TOOL_BINDING_RESOLVER,
        "schema": BUILD_TOOL_BINDING_SCHEMA,
        "tools": tools,
    }
    binding["digest"] = _binding_digest(binding)
    return binding


def _validate_build_tool_binding_shape(binding: Mapping[str, Any]) -> None:
    required = {
        "build_environment_path",
        "digest",
        "loader_closure",
        "resolver",
        "schema",
        "tools",
    }
    if set(binding) != required:
        raise ContractError("build-tool binding has missing or unknown members")
    if (
        binding.get("schema") != BUILD_TOOL_BINDING_SCHEMA
        or binding.get("resolver") != BUILD_TOOL_BINDING_RESOLVER
        or binding.get("digest") != _binding_digest(binding)
    ):
        raise ContractError("build-tool binding identity or digest drift")
    if not isinstance(binding["build_environment_path"], str):
        raise ContractError("build-tool binding PATH is malformed")
    tools = binding["tools"]
    if not isinstance(tools, list) or [
        item.get("role") if isinstance(item, dict) else None for item in tools
    ] != [role for role, _command in BUILD_TOOL_ROLES]:
        raise ContractError("build-tool binding tool order differs from resolver")
    expected_commands = [command for _role, command in BUILD_TOOL_ROLES]
    if [
        item.get("command") if isinstance(item, dict) else None for item in tools
    ] != expected_commands:
        raise ContractError("build-tool binding command set differs from resolver")
    records: list[Mapping[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict) or set(item) != {"command", "record", "role"}:
            raise ContractError("build-tool primary record shape is malformed")
        record = item["record"]
        if not isinstance(record, Mapping):
            raise ContractError("build-tool primary record is malformed")
        records.append(record)
    closure = binding["loader_closure"]
    if not isinstance(closure, list) or not closure:
        raise ContractError("build-tool loader/shared-library closure is malformed")
    if any(not isinstance(record, Mapping) for record in closure):
        raise ContractError("build-tool loader record is malformed")
    records.extend(closure)
    for record in records:
        if set(record) != {
            "length",
            "logical_path",
            "realpath",
            "sha256",
            "symlink_chain",
        }:
            raise ContractError("build-tool external record shape is malformed")
        require_lexical_absolute(str(record["logical_path"]))
        require_canonical_absolute(str(record["realpath"]), directory=False)
        if record["logical_path"] != record["realpath"] or record["symlink_chain"]:
            raise ContractError("build-tool record is not a physical resolved identity")
    realpaths = [str(record["realpath"]) for record in records]
    if len(realpaths) != len(set(realpaths)):
        raise ContractError("build-tool binding contains duplicate physical identities")
    closure_paths = [str(record["logical_path"]) for record in closure]
    if closure_paths != sorted(closure_paths, key=lambda path: path.encode("utf-8")):
        raise ContractError("build-tool loader closure is not canonically ordered")


def validate_build_tool_binding(
    contract: Mapping[str, Any],
    *,
    root: Path,
    started: float,
    monitor: BoundInputMonitor | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Prove the current build resolver is exactly the freeze-time binding."""
    binding = contract.get("build_tool_binding")
    if not isinstance(binding, Mapping):
        raise ContractError("controller input has no build-tool binding")
    _validate_build_tool_binding_shape(binding)
    if binding["build_environment_path"] != build_tool_environment_path(root):
        raise DriftError("build-tool PATH differs from the authoritative build PATH")
    ldd = require_canonical_absolute(
        str(contract["tool_paths"]["ldd"]), directory=False
    )
    current = resolve_build_tool_binding(
        root, ldd_path=ldd, started=started, monitor=monitor, clock=clock
    )
    if dict(binding) != current:
        raise DriftError("resolved build-tool binding differs from freeze-time binding")
    frozen = {
        record["realpath"]: (record["length"], record["sha256"])
        for record in contract["bound_inputs"]["tool_runtime"]
    }
    for item in [*binding["tools"], *binding["loader_closure"]]:
        record = item["record"] if "record" in item else item
        if frozen.get(record["realpath"]) != (record["length"], record["sha256"]):
            raise DriftError(
                "build-tool binding member is absent from h0_bound_inputs_v1: "
                + str(record["realpath"])
            )
    expected_paths = {
        item["role"]: item["record"]["realpath"] for item in binding["tools"]
    }
    if {
        "cxx": contract["tool_paths"].get("cxx"),
        "cmake": contract["tool_paths"].get("cmake"),
    } != expected_paths:
        raise ContractError("tool_paths and build-tool binding differ")


def validate_resolved_build_tool_identity(
    binding: Mapping[str, Any], *, cmake: Mapping[str, Any], cxx: Mapping[str, Any]
) -> None:
    """Require CMake's recorded tools to be the pre-resolved primary records."""
    _validate_build_tool_binding_shape(binding)
    expected = {item["role"]: item["record"] for item in binding["tools"]}
    for role, observed in (("cmake", cmake), ("cxx", cxx)):
        frozen = expected.get(role)
        if not isinstance(frozen, Mapping) or (
            observed.get("path"),
            observed.get("length"),
            observed.get("sha256"),
        ) != (
            frozen.get("realpath"),
            frozen.get("length"),
            frozen.get("sha256"),
        ):
            raise DriftError(
                f"CMake selected {role} outside the frozen build-tool binding"
            )


def _cmake_cache_entries(cache: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if (
            not line
            or line.startswith(("#", "//"))
            or "=" not in line
            or ":" not in line.split("=", 1)[0]
        ):
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in entries:
            raise ContractError(f"duplicate CMake cache identity: {key}")
        entries[key] = value
    return entries


def _build_identity(
    contract: Mapping[str, Any],
    root: Path,
    *,
    started: float,
    monitor: BoundInputMonitor | None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    python = root / ".venv/bin/python"
    query = _run_auxiliary_subprocess(
        [
            python.as_posix(),
            "-I",
            "-B",
            "-c",
            "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))",
        ],
        executable=python,
        cwd=root,
        env={
            "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
        started=started,
        monitor=monitor,
        stage="Python EXT_SUFFIX identity",
        clock=clock,
        text=True,
    )
    suffix = query.stdout.strip()
    if not suffix or "/" in suffix or "\\" in suffix:
        raise ContractError("Python EXT_SUFFIX is absent or non-canonical")
    artifact_paths = (
        root / f"build/h0_phase_a/saccade_tracking_ext{suffix}",
        root / "build/h0_phase_a/libsaccade_scan_plugin.so",
    )
    readelf = require_canonical_absolute(
        contract["tool_paths"]["readelf"], directory=False
    )
    ldd = require_canonical_absolute(contract["tool_paths"]["ldd"], directory=False)
    libraries = contract["library_dirs"]
    library_path = ":".join(
        (
            (root / "build/h0_phase_a").as_posix(),
            libraries["tensorrt_library_dir"],
            libraries["pytorch_library_dir"],
            libraries["cuda_library_dir"],
        )
    )
    python_path = root / ".venv/bin/python"
    maps_records = _runtime_maps_dependencies(
        python=python_path,
        extension=artifact_paths[0],
        plugin=artifact_paths[1],
        root=root,
        library_path=library_path,
        started=started,
        monitor=monitor,
        clock=clock,
    )
    artifacts = []
    for artifact in artifact_paths:
        record = _regular_file_record(artifact, artifact.relative_to(root).as_posix())
        ldd_records = _dynamic_dependencies(
            artifact,
            ldd,
            root=root,
            started=started,
            monitor=monitor,
            clock=clock,
        )
        # Merge ldd closure with maps-observed runtime loads (dlopen members).
        by_realpath = {item["realpath"]: item for item in ldd_records}
        for item in maps_records:
            by_realpath.setdefault(item["realpath"], item)
        # Never list the two top-level artifacts as their own dependencies.
        by_realpath.pop(artifact.resolve(strict=True).as_posix(), None)
        record["dynamic_dependencies"] = sorted(
            by_realpath.values(),
            key=lambda item: item["path"].encode("utf-8"),
        )
        record["elf_gnu_build_id"] = _elf_build_id(
            artifact,
            readelf,
            root=root,
            started=started,
            monitor=monitor,
            clock=clock,
        )
        artifacts.append(record)
    cache = root / "build/h0_phase_a/CMakeCache.txt"
    if not cache.is_file() or cache.is_symlink():
        raise ContractError("CMakeCache.txt is absent or not regular")
    cache_entries = _cmake_cache_entries(cache)
    required_cache = (
        "CMAKE_COMMAND",
        "CMAKE_GENERATOR",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_COMPILER",
    )
    if any(not cache_entries.get(key) for key in required_cache):
        raise ContractError("CMake cache lacks generator/compiler identity")
    compiler_records: dict[str, dict[str, Any]] = {}
    for language, key in (
        ("cxx", "CMAKE_CXX_COMPILER"),
        ("cuda", "CMAKE_CUDA_COMPILER"),
    ):
        executable = Path(cache_entries[key]).resolve(strict=True)
        data = executable.read_bytes()
        compiler_records[language] = {
            "length": len(data),
            "path": executable.as_posix(),
            "sha256": sha256_bytes(data),
            "version": _tool_version(
                executable,
                root=root,
                started=started,
                monitor=monitor,
                clock=clock,
            ),
        }
    binding = contract.get("build_tool_binding")
    if not isinstance(binding, Mapping):
        raise ContractError("build identity has no frozen build-tool binding")
    python_data = python.resolve(strict=True).read_bytes()
    python_query = _run_auxiliary_subprocess(
        [
            python.as_posix(),
            "-I",
            "-B",
            "-c",
            "import json,sys,sysconfig;print(json.dumps({'abi':sysconfig.get_config_var('SOABI'),'version':sys.version},sort_keys=True,separators=(',',':')))",
        ],
        executable=python,
        cwd=root,
        env={
            "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
        started=started,
        monitor=monitor,
        stage="Python ABI identity",
        clock=clock,
        text=True,
    )
    try:
        python_identity = json.loads(python_query.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("Python ABI identity is malformed") from exc
    uv_lock = root / "uv.lock"
    cmake_path = Path(cache_entries["CMAKE_COMMAND"]).resolve(strict=True)
    cmake_data = cmake_path.read_bytes()
    cmake_identity = {
        "length": len(cmake_data),
        "path": cmake_path.as_posix(),
        "sha256": sha256_bytes(cmake_data),
    }
    validate_resolved_build_tool_identity(
        binding, cmake=cmake_identity, cxx=compiler_records["cxx"]
    )
    return {
        "artifacts": artifacts,
        "build_environment": build_environment(contract),
        "build_environment_digest": build_environment_digest(
            build_environment(contract)
        ),
        "build_vectors": [list(vector) for vector in BUILD_VECTORS],
        "cmake_cache_sha256": sha256_bytes(cache.read_bytes()),
        "cmake": {
            "generator": cache_entries["CMAKE_GENERATOR"],
            **cmake_identity,
            "version": _tool_version(
                cmake_path,
                root=root,
                started=started,
                monitor=monitor,
                clock=clock,
            ),
        },
        "build_tool_binding": dict(binding),
        "compilers": compiler_records,
        "cuda_toolkit_root": Path(cache_entries["CMAKE_CUDA_COMPILER"])
        .resolve(strict=True)
        .parent.parent.as_posix(),
        "python": {
            **python_identity,
            "length": len(python_data),
            "path": python.resolve(strict=True).as_posix(),
            "sha256": sha256_bytes(python_data),
        },
        "python_ext_suffix": suffix,
        "state": "complete",
        "uv_lock_sha256": sha256_bytes(uv_lock.read_bytes()),
    }


def _verify_extension_load(
    contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    started: float,
    monitor: BoundInputMonitor,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    root = Path(contract["repository_root"])
    incomplete = root / contract["incomplete_root"]
    python = root / ".venv/bin/python"
    extension = root / identity["artifacts"][0]["path"]
    plugin = root / identity["artifacts"][1]["path"]
    libraries = contract["library_dirs"]
    environment = {
        **build_environment(contract),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": contract["gpu"]["uuid"],
        "LD_LIBRARY_PATH": ":".join(
            (
                (root / "build/h0_phase_a").as_posix(),
                libraries["tensorrt_library_dir"],
                libraries["pytorch_library_dir"],
                libraries["cuda_library_dir"],
            )
        ),
        "SACCADE_BUILD_PATH": (root / "build/h0_phase_a").as_posix(),
    }
    workspace = incomplete / "_extension_load"
    workspace.mkdir(exist_ok=False)
    denial_probe = incomplete / "_runtime_confinement_denial_probe"
    if not denial_probe.exists():
        _write_bytes_fsync(denial_probe, b"MUST_BE_DENIED\n")
    if denial_probe.is_symlink() or not denial_probe.is_file():
        raise ContractError("runtime confinement denial probe is not regular")
    script = (
        "import ctypes,pathlib;"
        f"p=pathlib.Path({denial_probe.as_posix()!r});"
        "denied=False;"
        "\ntry:p.read_bytes()"
        "\nexcept PermissionError:denied=True"
        "\nassert denied;"
        "\nimport saccade_tracking_ext;"
        f"e=pathlib.Path({extension.as_posix()!r}).resolve(strict=True);"
        "a=pathlib.Path(saccade_tracking_ext.__file__).resolve(strict=True);"
        "assert a==e;"
        f"ctypes.CDLL({plugin.as_posix()!r},mode=ctypes.RTLD_LOCAL)"
    )
    vector = [python.as_posix(), "-I", "-B", "-c", script]
    tools = SCHEMA_PATH.parent
    if tools.as_posix() not in sys.path:
        sys.path.insert(0, tools.as_posix())
    import h0_runtime_confinement

    try:
        runtime_plan = h0_runtime_confinement.build_plan(
            root=root,
            incomplete=incomplete,
            inventory=contract["bound_inputs"],
            build_identity=identity,
            denial_probe=denial_probe,
            output_directories=(workspace,),
        )
    except h0_runtime_confinement.ConfinementError as exc:
        raise DriftError(f"extension-load confinement plan failed: {exc}") from exc
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    _bounded_remaining(started, now=clock)
    with (
        open(os.devnull, "rb", buffering=0) as stdin,
        open(stdout_path, "xb", buffering=0) as stdout,
        open(stderr_path, "xb", buffering=0) as stderr,
    ):
        try:
            process = h0_runtime_confinement.spawn_confined(
                vector,
                cwd=root,
                env=environment,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                plan=runtime_plan,
            )
        except h0_runtime_confinement.ConfinementError as exc:
            raise DriftError(f"extension-load confinement setup failed: {exc}") from exc
        try:
            returncode = _wait_with_monitor(
                process,
                started=started,
                monitor=monitor,
                stage="confined extension/plugin load",
                clock=clock,
            )
        except (subprocess.TimeoutExpired, TimeoutError):
            _terminate_process_group(process)
            raise
    attestation, runtime_digest, valid_runtime = _runtime_attestation_details(
        process, runtime_plan
    )
    state = "complete"
    result = "extension_loaded"
    if not valid_runtime:
        state = "rejected"
        result = "provenance_invalid"
    elif returncode != 0:
        state = "failed"
        result = "extension_load_failed"

    def _artifact_identity(index: int, path: Path) -> dict[str, Any]:
        record = identity["artifacts"][index]
        if "length" in record and "sha256" in record:
            return {
                "path": path,
                "length": int(record["length"]),
                "sha256": str(record["sha256"]),
            }
        data = path.read_bytes()
        return {
            "path": path,
            "length": len(data),
            "sha256": sha256_bytes(data),
        }

    try:
        membership = h0_runtime_confinement.assert_extension_plugin_membership(
            attestation,
            extension=extension,
            plugin=plugin,
            extension_identity=_artifact_identity(0, extension),
            plugin_identity=_artifact_identity(1, plugin),
        )
    except h0_runtime_confinement.ConfinementError as exc:
        raise DriftError(
            "extension/plugin load is absent from runtime attestation"
            if "absent from runtime attestation" in str(exc)
            else f"extension/plugin runtime attestation membership failed: {exc}"
        ) from exc
    return {
        "confinement_plan_digest": runtime_plan["digest"],
        "confinement_probe_passed": attestation.get("denial_probe_observed") is True,
        "environment": environment,
        "environment_digest": sha256_bytes(canonical_json_bytes(environment)),
        "extension_artifact_observed": membership["extension_artifact_observed"],
        "extension_identity_equal": membership["extension_identity_equal"],
        "plugin_artifact_observed": membership["plugin_artifact_observed"],
        "plugin_identity_equal": membership["plugin_identity_equal"],
        "result": result,
        "returncode": returncode,
        "runtime_inputs": attestation,
        "runtime_inputs_digest": runtime_digest,
        "state": state,
        "vector": vector,
    }


def _validate_build_tool_runtime_binding(
    contract: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    """Tie the observed build identity back to the freeze-time tool binding.

    Build-artifact closure remains a distinct build-derived runtime class.  In
    contrast, the CMake executable, C++ driver, and both tools' loader/shared
    library closure are all pre-frozen in ``tool_runtime`` and copied verbatim
    into the packet's build identity.
    """
    binding = contract.get("build_tool_binding")
    if not isinstance(binding, Mapping) or identity.get("build_tool_binding") != dict(
        binding
    ):
        raise DriftError("build identity differs from the frozen build-tool binding")
    _validate_build_tool_binding_shape(binding)
    frozen = {
        record["realpath"]: (record["length"], record["sha256"])
        for record in contract["bound_inputs"]["tool_runtime"]
    }
    required: list[tuple[str, int, str]] = []
    for compiler in identity["compilers"].values():
        required.append((compiler["path"], compiler["length"], compiler["sha256"]))
    for name in ("cmake", "python"):
        record = identity[name]
        required.append((record["path"], record["length"], record["sha256"]))
    for name in ("git", "ldd", "nvcc", "readelf", "uv"):
        path = Path(contract["tool_paths"][name]).resolve(strict=True)
        data = path.read_bytes()
        required.append((path.as_posix(), len(data), sha256_bytes(data)))
    missing = [
        realpath
        for realpath, length, sha256 in required
        if frozen.get(realpath) != (length, sha256)
    ]
    if missing:
        raise DriftError(
            f"runtime-loaded tool/library absent from h0_bound_inputs_v1: {missing}"
        )


def _validate_policy_inventory(run_id: str, inventory: Mapping[str, Any]) -> None:
    required_members = {
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
        set(inventory) != required_members
        or inventory.get("schema") != "h0_phase_a_policy_inventory_v1"
    ):
        raise ContractError(f"policy inventory schema mismatch: {run_id}")

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
        raise ContractError(f"policy MOT identity shape mismatch: {run_id}")
    if not isinstance(inventory["final_track_rows"], list) or not isinstance(
        inventory["active_tid_slot_pairs"], list
    ):
        raise ContractError(f"policy inventory row collection mismatch: {run_id}")
    for row in inventory["final_track_rows"]:
        if not isinstance(row, dict) or set(row) != {
            "binary32_bits",
            "class",
            "frame",
            "row_index",
            "track_id",
        }:
            raise ContractError(f"final-track row shape mismatch: {run_id}")
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
            raise ContractError(f"final-track row value mismatch: {run_id}")
    positions: dict[int, list[int]] = {}
    for row in inventory["final_track_rows"]:
        positions.setdefault(row["frame"], []).append(row["row_index"])
    if any(values != list(range(len(values))) for values in positions.values()):
        raise ContractError(
            f"final-track row positions are not emitted order: {run_id}"
        )
    for row in inventory["active_tid_slot_pairs"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"frame", "pairs"}
            or not exact_int(row["frame"])
            or row["frame"] < 1
        ):
            raise ContractError(f"active tid/slot row shape mismatch: {run_id}")
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
            raise ContractError(f"active tid/slot pairs mismatch: {run_id}")
    for member, length in (("relink_debug_raw", 13), ("overflow_vector", 9)):
        vector = inventory[member]
        if (
            not isinstance(vector, list)
            or len(vector) != length
            or any(not exact_int(value) for value in vector)
        ):
            raise ContractError(f"policy inventory vector mismatch: {run_id}:{member}")

    def validate_projection(value: object, *, winner: bool) -> None:
        if not isinstance(value, dict) or set(value) != {"count", "digest", "records"}:
            raise ContractError(f"trace projection shape mismatch: {run_id}")
        records_key = (
            {"commits", "winning_claims"} if winner else {"candidates", "claims"}
        )
        records = value["records"]
        if (
            not isinstance(records, dict)
            or set(records) != records_key
            or any(not isinstance(records[key], list) for key in records_key)
        ):
            raise ContractError(f"trace projection records mismatch: {run_id}")
        primary = records["commits" if winner else "candidates"]
        if (
            not exact_int(value["count"])
            or value["count"] != len(primary)
            or value["digest"] != sha256_bytes(canonical_json_bytes(records))
        ):
            raise ContractError(f"trace projection count/digest mismatch: {run_id}")

    if run_id == RUN_IDS[0]:
        if (
            inventory["proposal_projection"] is not None
            or inventory["winner_commit_projection"] is not None
        ):
            raise ContractError("capture-off fabricated trace-only projections")
    else:
        validate_projection(inventory["proposal_projection"], winner=False)
        validate_projection(inventory["winner_commit_projection"], winner=True)


def _compare_policy_inventories(incomplete: Path) -> tuple[bool, dict[str, Any]]:
    inventories = {
        run_id: read_canonical_json(
            incomplete / "runs" / run_id / "policy_inventory.json"
        )
        for run_id in RUN_IDS
    }
    for run_id, inventory in inventories.items():
        _validate_policy_inventory(run_id, inventory)
        mot_bytes = (incomplete / "runs" / run_id / "MOT17-04-SDP.txt").read_bytes()
        if inventory["mot_output"] != {
            "length": len(mot_bytes),
            "sha256": sha256_bytes(mot_bytes),
        }:
            raise ContractError(f"MOT bytes differ from policy inventory: {run_id}")
    off = inventories[RUN_IDS[0]]
    equality_members = (
        "mot_output",
        "final_track_rows",
        "active_tid_slot_pairs",
        "relink_debug_raw",
    )
    relations: list[dict[str, Any]] = []
    first_unequal: str | None = None
    for run_id in CAPTURE_ON_RUN_IDS:
        for member in equality_members:
            equal = off[member] == inventories[run_id][member]
            relations.append(
                {"equal": equal, "left": RUN_IDS[0], "member": member, "right": run_id}
            )
            if not equal and first_unequal is None:
                first_unequal = f"{RUN_IDS[0]}:{run_id}:{member}"
    for member in ("proposal_projection", "winner_commit_projection"):
        reference = inventories[CAPTURE_ON_RUN_IDS[0]][member]
        for run_id in CAPTURE_ON_RUN_IDS[1:]:
            equal = reference == inventories[run_id][member]
            relations.append(
                {
                    "equal": equal,
                    "left": CAPTURE_ON_RUN_IDS[0],
                    "member": member,
                    "right": run_id,
                }
            )
            if not equal and first_unequal is None:
                first_unequal = f"{CAPTURE_ON_RUN_IDS[0]}:{run_id}:{member}"
    for run_id in CAPTURE_ON_RUN_IDS:
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
    return first_unequal is None, {
        "first_unequal": first_unequal,
        "relations": relations,
        "state": "equal" if first_unequal is None else "unequal",
    }


def _not_produced(blocking_result: str) -> dict[str, str]:
    return {"blocking_result": blocking_result, "state": "not_produced"}


def _packet_verification_states(incomplete: Path) -> list[str]:
    states: list[str] = []
    digests: list[str | None] = []
    for relative in V_PATHS:
        value = read_canonical_json(incomplete / relative)
        state = "pass" if value.get("state") == "pass" else "fail"
        states.append(state)
        report = value.get("report")
        digests.append(
            report.get("semantic_digest_sha256") if isinstance(report, dict) else None
        )
    if states == ["pass"] * 3 and len(set(digests)) != 1:
        reference = digests[0]
        states = ["pass" if value == reference else "fail" for value in digests]
        if states == ["pass"] * 3:
            states[1] = "fail"
    return states


def _independent_packet_states(incomplete: Path) -> list[str]:
    tools = SCHEMA_PATH.parent
    if tools.as_posix() not in sys.path:
        sys.path.insert(0, tools.as_posix())
    from verify_headline_bridge_decision_trace import verify_capture

    states: list[str] = []
    digests: list[str] = []
    for run_id, relative in zip(CAPTURE_ON_RUN_IDS, V_PATHS, strict=True):
        capture = read_canonical_json(incomplete / "runs" / run_id / "packet.json")
        stored = read_canonical_json(incomplete / relative)
        try:
            report = verify_capture(capture)
        except (KeyError, TypeError, ValueError):
            if stored != {"failure": "packet_invalid", "state": "fail"}:
                raise ContractError(f"child packet failure record mismatch: {run_id}")
            states.append("fail")
        else:
            if stored != {"report": report, "state": "pass"}:
                raise ContractError(f"child packet pass record mismatch: {run_id}")
            states.append("pass")
            digests.append(report["semantic_digest_sha256"])
    if states == ["pass"] * 3 and len(set(digests)) != 1:
        states[1] = "fail"
    return states


def _ensure_not_run_slots(contract: Mapping[str, Any], incomplete: Path) -> None:
    root = Path(contract["repository_root"])
    for run_id in RUN_IDS:
        run_dir = incomplete / "runs" / run_id
        invocation_path = run_dir / "invocation.json"
        if not run_dir.exists():
            run_dir = _create_run_directories(incomplete, run_id)
        for log_name in ("stdout.log", "stderr.log"):
            log = run_dir / log_name
            if not log.exists():
                _write_bytes_fsync(log, b"NOT_RUN\n")
        if not invocation_path.exists():
            environment = child_environment(
                root,
                run_dir,
                gpu_uuid=contract["gpu"]["uuid"],
                validate_paths=False,
                **contract["library_dirs"],
            )
            value = {
                "bound_inputs_digest": contract["bound_inputs"]["digest"],
                "capture_run_uuid": str(uuid.uuid4()),
                "confinement_plan_digest": None,
                "confinement_probe_passed": None,
                "document_type": "child_invocation",
                "environment": environment,
                "environment_digest": environment_digest(environment),
                "evaluator_argv": list(evaluator_argv(run_dir)),
                "incomplete_root": contract["incomplete_root"],
                "instrumentation_head": contract["instrumentation_head"],
                "result": None,
                "run_id": run_id,
                "runtime_inputs_digest": None,
                "schema": CHILD_SCHEMA,
                "state": "not_run",
                "vector": list(child_argv(root, run_id)),
            }
            _write_canonical_fsync(invocation_path, value)


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_forbidden(incomplete: Path, result: str) -> None:
    _required, forbidden = result_artifact_sets(result)
    for relative in forbidden:
        path = incomplete / relative
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"forbidden non-regular artifact: {relative}")
            path.unlink()


def _artifact_inventory(incomplete: Path) -> list[str]:
    files: list[str] = []
    for path in incomplete.rglob("*"):
        relative = path.relative_to(incomplete).as_posix()
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ContractError(f"non-regular evidence entry: {relative}")
        if path.is_file():
            files.append(relative)
    return sorted(files, key=lambda value: value.encode("utf-8"))


def _remove_transient_run_trees(incomplete: Path) -> None:
    resolved_root = incomplete.resolve(strict=True)
    for transient in ("_build_env", "_extension_load"):
        transient_root = incomplete / transient
        if transient_root.exists():
            resolved = transient_root.resolve(strict=True)
            if resolved_root not in resolved.parents or transient_root.is_symlink():
                raise ContractError(f"unsafe transient tree: {transient_root}")
            shutil.rmtree(transient_root)
    denial_probe = incomplete / "_runtime_confinement_denial_probe"
    if denial_probe.exists() or denial_probe.is_symlink():
        if denial_probe.is_symlink() or not denial_probe.is_file():
            raise ContractError("unsafe runtime confinement denial probe")
        denial_probe.unlink()
    for run_id in RUN_IDS:
        for name in ("_env", "_runtime"):
            target = incomplete / "runs" / run_id / name
            if not target.exists():
                continue
            resolved = target.resolve(strict=True)
            if resolved_root not in resolved.parents or target.is_symlink():
                raise ContractError(f"unsafe transient run tree: {target}")
            shutil.rmtree(target)


def _validate_directory_universe(incomplete: Path, required: Sequence[str]) -> None:
    expected: set[str] = set()
    for relative in required:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected.add(parent.as_posix())
            parent = parent.parent
    actual = {
        path.relative_to(incomplete).as_posix()
        for path in incomplete.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual != expected:
        raise ContractError(
            f"evidence directory universe mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_evidence_root(
    incomplete: Path,
    final: Path,
    *,
    started: float,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Atomically publish only if rename and parent fsync finish in time."""
    _bounded_remaining(started, now=clock)
    renamed = False
    try:
        os.replace(incomplete, final)
        renamed = True
        _bounded_remaining(started, now=clock)
        _fsync_directory(final.parent)
        _bounded_remaining(started, now=clock)
    except BaseException:
        if renamed:
            if incomplete.exists() or incomplete.is_symlink():
                raise ContractError(
                    "publication rollback refused an occupied incomplete root"
                )
            if not final.exists() or final.is_symlink():
                raise ContractError(
                    "publication rollback could not identify the renamed root"
                )
            os.replace(final, incomplete)
            _fsync_directory(incomplete.parent)
        raise


def _finalize_bundle_once(
    contract: Mapping[str, Any],
    *,
    started: float,
    result: str,
    predicates: Mapping[str, bool],
    checkpoints: list[dict[str, Any]],
    comparison: Mapping[str, Any] | None,
    build_identity: Mapping[str, Any] | None,
    mutation_events: Sequence[Mapping[str, Any]],
    failure: Mapping[str, str] | None = None,
    runtime_attestations: Mapping[str, Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    enforce_deadline: bool = True,
    publish: bool = True,
) -> Path:
    if publish and not enforce_deadline:
        raise ContractError("publication requires active deadline admission")
    if runtime_attestations is None:
        runtime_attestations = {}

    def admit() -> None:
        if enforce_deadline:
            _bounded_remaining(started, now=clock)

    def finish(action: Callable[..., Any], *args: object) -> Any:
        if enforce_deadline:
            return _deadline_checked_call(started, clock, action, *args)
        return action(*args)

    root = Path(contract["repository_root"])
    incomplete = root / contract["incomplete_root"]
    final = root / contract["evidence_root"]
    admit()
    incomplete.mkdir(parents=True, exist_ok=True)
    (incomplete / "logs").mkdir(exist_ok=True)
    (incomplete / "verification").mkdir(exist_ok=True)
    admit()
    for log_name in (
        "00_cmake_configure.stdout.log",
        "00_cmake_configure.stderr.log",
        "01_cmake_build.stdout.log",
        "01_cmake_build.stderr.log",
    ):
        path = incomplete / "logs" / log_name
        if not path.exists():
            finish(_write_bytes_fsync, path, b"NOT_RUN\n")
    _ensure_not_run_slots(contract, incomplete)
    admit()
    _remove_forbidden(incomplete, result)
    _remove_transient_run_trees(incomplete)
    admit()
    child_runtime_inputs = []
    if build_identity:
        for run_id in RUN_IDS:
            invocation = read_canonical_json(
                incomplete / "runs" / run_id / "invocation.json"
            )
            attestation = runtime_attestations.get(run_id)
            if attestation is not None:
                if invocation["runtime_inputs_digest"] != sha256_bytes(
                    canonical_json_bytes(attestation)
                ):
                    raise ContractError(f"runtime attestation digest drift: {run_id}")
                record: Mapping[str, Any] = attestation
            else:
                if invocation["runtime_inputs_digest"] is not None:
                    raise ContractError(
                        f"recorded runtime digest lacks its attestation: {run_id}"
                    )
                record = {
                    "blocking_result": result,
                    "schema": "h0_runtime_inputs_v1",
                    "state": "not_produced",
                }
            child_runtime_inputs.append({"run_id": run_id, "runtime_inputs": record})
    identities: dict[str, Mapping[str, Any]] = {
        "build_identity.json": build_identity or _not_produced(result),
        "runtime_identity.json": {
            "bound_inputs_digest": contract["bound_inputs"]["digest"],
            "child_runtime_inputs": child_runtime_inputs,
            "library_dirs": contract["library_dirs"],
            "resolved_policy_fingerprint": POLICY_FINGERPRINT,
            "state": "complete",
            "tool_runtime": contract["bound_inputs"]["tool_runtime"],
        }
        if build_identity
        else _not_produced(result),
        "gpu_identity.json": {**contract["gpu"], "state": "complete"}
        if build_identity
        else _not_produced(result),
        "comparison.json": comparison or _not_produced(result),
    }
    for name, value in identities.items():
        path = incomplete / name
        if not path.exists():
            finish(_write_canonical_fsync, path, value)
    completed_all = all(item["state"] == "completed" for item in checkpoints)
    failed_checkpoint = any(item["state"] == "failed" for item in checkpoints)
    if mutation_events:
        classifications = {str(item["classification"]) for item in mutation_events}
        if "queue_overflow" in classifications:
            monitor_state = "queue_overflow"
        elif "ignored_watch" in classifications:
            monitor_state = "ignored_watch"
        elif "watch_failed" in classifications:
            monitor_state = "watch_failed"
        else:
            monitor_state = "drift"
        final_equal: bool | None = False
    elif failed_checkpoint:
        # A completed comparison may prove an unequal inventory even when
        # inotify observed nothing.  Its failed checkpoint is the evidence;
        # the monitor itself still closed cleanly.
        monitor_state = "closed_clean"
        final_equal = False
    elif all(item["state"] == "not_reached" for item in checkpoints):
        monitor_state = "not_started"
        final_equal = None
    else:
        monitor_state = "closed_clean"
        final_equal = True if completed_all else None
    input_binding = {
        "algorithm": BOUND_INPUTS_SCHEMA,
        "checkpoints": checkpoints,
        "failure": dict(failure) if failure is not None else None,
        "final_equal": final_equal,
        "inotify_mask": list(INOTIFY_MASK_NAMES),
        "monitor_state": monitor_state,
        "mutation_events": list(mutation_events),
        "t0_digest": contract["bound_inputs"]["digest"],
    }
    finish(
        _write_canonical_fsync,
        incomplete / "input_binding.json",
        input_binding,
    )
    children = [
        read_canonical_json(incomplete / "runs" / run_id / "invocation.json")
        for run_id in RUN_IDS
    ]
    packet_states = ["not_produced"] * 3
    policy_state = "not_produced"
    if result == "capture_perturbs_policy":
        policy_state = "unequal"
    elif result in {"packet_invalid", "phase_a_pass"}:
        policy_state = "equal"
        packet_states = _packet_verification_states(incomplete)
    manifest = {
        "artifact_inventory": list(result_artifact_sets(result)[0]),
        "child_invocations": children,
        "controller_input": dict(contract),
        "decision_predicates": dict(predicates),
        "document_type": "execution_evidence",
        "input_binding": input_binding,
        "packet_verification_states": packet_states,
        "policy_comparison": policy_state,
        "result": result,
        "result_matrix": execution_constants(root)["result_matrix"],
        "schema": EXECUTION_SCHEMA,
    }
    finish(
        _write_canonical_fsync,
        incomplete / "manifest.json",
        manifest,
    )
    finish(
        _write_canonical_fsync,
        incomplete / "result.json",
        {"result": result, "schema": EXECUTION_SCHEMA},
    )
    tools = SCHEMA_PATH.parent
    if tools.as_posix() not in sys.path:
        sys.path.insert(0, tools.as_posix())
    import verify_h0_phase_a

    aggregate = verify_h0_phase_a.verify_evidence(manifest)
    admit()
    finish(
        _write_canonical_fsync,
        incomplete / "verification/aggregate.json",
        aggregate,
    )
    actual_without_checksums = _artifact_inventory(incomplete)
    expected_without_checksums = [
        path for path in result_artifact_sets(result)[0] if path != "checksums.sha256"
    ]
    if sorted(
        actual_without_checksums, key=lambda value: value.encode("utf-8")
    ) != sorted(expected_without_checksums, key=lambda value: value.encode("utf-8")):
        raise ContractError("final artifact inventory does not equal the RC1.4 row")
    admit()
    checksum_lines = []
    for relative in sorted(
        actual_without_checksums, key=lambda value: value.encode("utf-8")
    ):
        checksum_lines.append(
            f"{sha256_bytes((incomplete / relative).read_bytes())}  {relative}\n"
        )
        admit()
    finish(
        _write_bytes_fsync,
        incomplete / "checksums.sha256",
        "".join(checksum_lines).encode("ascii"),
    )
    if _artifact_inventory(incomplete) != sorted(
        result_artifact_sets(result)[0], key=lambda value: value.encode("utf-8")
    ):
        raise ContractError("checksummed artifact inventory mismatch")
    admit()
    _validate_directory_universe(incomplete, result_artifact_sets(result)[0])
    admit()
    reconstructed_aggregate = finish(
        verify_h0_phase_a.verify_evidence_root,
        incomplete,
    )
    if reconstructed_aggregate != aggregate:
        raise ContractError("staged-root reconstruction differs from stored aggregate")
    admit()
    if not publish:
        return incomplete
    _publish_evidence_root(
        incomplete,
        final,
        started=started,
        clock=clock,
    )
    return final


def _reset_incomplete_finalization(incomplete: Path) -> None:
    """Remove only derived publication members before timeout reclassification."""
    for relative in (
        "build_identity.json",
        "runtime_identity.json",
        "gpu_identity.json",
        "comparison.json",
        "input_binding.json",
        "manifest.json",
        "result.json",
        "verification/aggregate.json",
        "checksums.sha256",
    ):
        path = incomplete / relative
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"unsafe derived publication member: {relative}")
        path.unlink()


def _finalize_bundle(
    contract: Mapping[str, Any],
    *,
    started: float,
    result: str,
    predicates: Mapping[str, bool],
    checkpoints: list[dict[str, Any]],
    comparison: Mapping[str, Any] | None,
    build_identity: Mapping[str, Any] | None,
    mutation_events: Sequence[Mapping[str, Any]],
    failure: Mapping[str, str] | None = None,
    runtime_attestations: Mapping[str, Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Publish in-deadline, or leave a verified timeout envelope staged."""
    try:
        _finalize_bundle_once(
            contract,
            started=started,
            result=result,
            predicates=predicates,
            checkpoints=checkpoints,
            comparison=comparison,
            build_identity=build_identity,
            mutation_events=mutation_events,
            failure=failure,
            runtime_attestations=runtime_attestations,
            clock=clock,
        )
        return result
    except TimeoutError:
        timed_out_predicates = dict(predicates)
        timed_out_predicates["timed_out"] = True
        timeout_result = classify_result(**timed_out_predicates)
        root = Path(contract["repository_root"])
        incomplete = root / contract["incomplete_root"]
        final = root / contract["evidence_root"]
        if final.exists() or final.is_symlink():
            raise ContractError("deadline recovery found a published evidence root")
        incomplete.mkdir(parents=True, exist_ok=True)
        _reset_incomplete_finalization(incomplete)
        _finalize_bundle_once(
            contract,
            started=started,
            result=timeout_result,
            predicates=timed_out_predicates,
            checkpoints=checkpoints,
            comparison=None,
            build_identity=build_identity,
            mutation_events=mutation_events,
            failure=failure,
            runtime_attestations=runtime_attestations,
            clock=clock,
            enforce_deadline=False,
            publish=False,
        )
        return timeout_result


def execute_controller(
    contract: Mapping[str, Any],
    *,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    started: float | None = None,
) -> str:
    """Execute A7 and publish RC1.4 only while the deadline remains valid."""
    root = Path(contract["repository_root"])
    if started is None:
        started = clock()
    incomplete = root / contract["incomplete_root"]
    checkpoints: list[dict[str, Any]] = []
    predicates = {
        "artifacts_ok": True,
        "build_ok": True,
        "classified_execution": True,
        "extension_ok": True,
        "packets_valid": True,
        "policy_equal": True,
        "provenance_ok": True,
        "runners_ok": True,
        "serialization_ok": True,
        "timed_out": False,
    }
    build_identity: Mapping[str, Any] | None = None
    comparison: Mapping[str, Any] | None = None
    result = "unclassified_execution_failure"
    stage = "preflight"
    monitor: BoundInputMonitor | None = None
    mutation_events: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None
    runtime_attestations: dict[str, Mapping[str, Any]] = {}
    try:
        preflight_controller_input(contract, root, started=started, clock=clock)
        tools = SCHEMA_PATH.parent
        if tools.as_posix() not in sys.path:
            sys.path.insert(0, tools.as_posix())
        import verify_h0_phase_a

        verify_h0_phase_a._schema_document()
        monitor = BoundInputMonitor(
            bound_file_paths(contract),
            ignored_roots=(root / "build/h0_phase_a", incomplete),
        )
        checkpoints.append(
            verify_bound_checkpoint(
                contract, monitor, "T0", started=started, clock=clock
            )
        )
        stage = "build"
        incomplete.mkdir(parents=True, exist_ok=False)
        (incomplete / "logs").mkdir()
        _ensure_not_run_slots(contract, incomplete)
        _create_build_environment(contract)
        configure_rc = _run_build_vector(
            contract,
            BUILD_VECTORS[0],
            incomplete / "logs/00_cmake_configure.stdout.log",
            incomplete / "logs/00_cmake_configure.stderr.log",
            started=started,
            monitor=monitor,
            popen_factory=popen_factory,
            clock=clock,
        )
        if configure_rc != 0:
            predicates["build_ok"] = False
            result = "build_failed"
            raise ContractError("CMake configure exited nonzero")
        build_rc = _run_build_vector(
            contract,
            BUILD_VECTORS[1],
            incomplete / "logs/01_cmake_build.stdout.log",
            incomplete / "logs/01_cmake_build.stderr.log",
            started=started,
            monitor=monitor,
            popen_factory=popen_factory,
            clock=clock,
        )
        if build_rc != 0:
            predicates["build_ok"] = False
            result = "build_failed"
            raise ContractError("CMake build exited nonzero")
        try:
            build_identity = _build_identity(
                contract,
                root,
                started=started,
                monitor=monitor,
                clock=clock,
            )
        except (ContractError, OSError, subprocess.SubprocessError) as exc:
            predicates["build_ok"] = False
            result = "build_failed"
            raise ContractError(f"build identity failed: {exc}") from exc
        stage = "build_binding"
        _validate_build_tool_runtime_binding(contract, build_identity)
        checkpoints.append(
            verify_bound_checkpoint(
                contract, monitor, "T1", started=started, clock=clock
            )
        )
        stage = "extension_load"
        try:
            extension_load = _verify_extension_load(
                contract,
                build_identity,
                started=started,
                monitor=monitor,
                clock=clock,
            )
            build_identity = {**build_identity, "extension_load": extension_load}
            if extension_load["state"] == "rejected":
                raise DriftError(
                    "extension/plugin load consumed an unbound runtime input"
                )
            if extension_load["state"] != "complete":
                predicates["extension_ok"] = False
                result = "extension_load_failed"
                raise ContractError("confined extension/plugin load exited nonzero")
        except DriftError:
            raise
        except (subprocess.TimeoutExpired, TimeoutError):
            raise
        except (OSError, subprocess.SubprocessError, ContractError) as exc:
            predicates["extension_ok"] = False
            result = "extension_load_failed"
            raise ContractError(f"extension/plugin load failed: {exc}") from exc
        stage = "runs"
        for index, run_id in enumerate(RUN_IDS):
            checkpoints.append(
                verify_bound_checkpoint(
                    contract,
                    monitor,
                    f"T2a_{index}",
                    started=started,
                    clock=clock,
                )
            )
            child_error: BaseException | None = None
            returncode = -1
            child_result: Mapping[str, Any] = {}
            try:
                returncode, child_result = launch_child(
                    contract,
                    run_id,
                    started=started,
                    build_identity=build_identity,
                    monitor=monitor,
                    popen_factory=popen_factory,
                    clock=clock,
                    attestations=runtime_attestations,
                )
            except BaseException as exc:
                child_error = exc
            # T2b is mandatory after the active process has been reaped.  It
            # performs (and, if necessary, records) its own terminal verdict
            # before a child error is propagated.
            checkpoints.append(
                verify_bound_checkpoint(
                    contract,
                    monitor,
                    f"T2b_{index}",
                    started=started,
                    clock=clock,
                )
            )
            if child_error is not None:
                raise child_error
            if child_result.get("result") == "provenance_invalid":
                predicates["provenance_ok"] = False
                result = "provenance_invalid"
                raise ContractError(f"child {run_id} rejected an unbound runtime input")
            if returncode != 0 or child_result["state"] != "completed":
                predicates["runners_ok"] = False
                result = "runner_nonzero"
                raise ContractError(f"child {run_id} exited nonzero")
        checkpoints.append(
            verify_bound_checkpoint(
                contract, monitor, "T3", started=started, clock=clock
            )
        )
        stage = "comparison"
        equal, comparison = _compare_policy_inventories(incomplete)
        predicates["policy_equal"] = equal
        if not equal:
            result = "capture_perturbs_policy"
        else:
            packet_pass = _independent_packet_states(incomplete) == ["pass"] * 3
            predicates["packets_valid"] = packet_pass
            result = "phase_a_pass" if packet_pass else "packet_invalid"
        checkpoints.append(
            verify_bound_checkpoint(
                contract, monitor, "T4", started=started, clock=clock
            )
        )
        monitor.assert_clean()
        monitor.close()
        monitor = None
    except TimeoutError as exc:
        predicates["timed_out"] = True
        result = "runner_timeout"
        failure = _failure_record(stage, exc)
    except DriftError as exc:
        predicates["provenance_ok"] = False
        result = "provenance_invalid"
        failure = _failure_record(stage, exc)
        # Only a checkpoint operation may emit its failed checkpoint row.  A
        # surrounding-stage DriftError leaves later checkpoints not_reached,
        # regardless of monitor history.
        if isinstance(exc, CheckpointDriftError):
            if exc.checkpoint_record is None:
                raise ContractError("checkpoint drift lacks an operation record")
            checkpoints.append(exc.checkpoint_record)
        if monitor is not None:
            mutation_events.extend(
                {
                    "classification": event.classification,
                    "mask": event.mask,
                    "path": event.path,
                }
                for event in monitor.history
            )
    except ContractError as exc:
        failure = _failure_record(stage, exc)
        if stage == "preflight":
            predicates["provenance_ok"] = False
            result = "provenance_invalid"
        elif stage == "runs" and result == "unclassified_execution_failure":
            predicates["runners_ok"] = False
            result = "runner_nonzero"
        elif stage == "comparison" and result == "unclassified_execution_failure":
            predicates["artifacts_ok"] = False
            result = "artifact_missing_or_unreadable"
        elif result == "unclassified_execution_failure":
            predicates["classified_execution"] = False
    except BaseException as exc:
        predicates["classified_execution"] = False
        result = "unclassified_execution_failure"
        failure = _failure_record(stage, exc)
    finally:
        if monitor is not None:
            monitor.close()
    while len(checkpoints) < len(CHECKPOINTS):
        checkpoints.append(_not_reached_checkpoint(CHECKPOINTS[len(checkpoints)]))
    # The deadline includes all work preceding result/checksum publication.
    # If it expired while classifying or closing T4, the fixed result priority
    # is recomputed before any final bundle member is serialized.
    try:
        _bounded_remaining(started, now=clock)
    except TimeoutError:
        predicates["timed_out"] = True
    selected = classify_result(**predicates)
    if selected != result:
        result = selected
    result = _finalize_bundle(
        contract,
        started=started,
        result=result,
        predicates=predicates,
        checkpoints=checkpoints,
        comparison=comparison,
        build_identity=build_identity,
        mutation_events=mutation_events,
        failure=failure,
        runtime_attestations=runtime_attestations,
        clock=clock,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic()
    values = tuple(sys.argv[1:] if argv is None else argv)
    try:
        should_execute = _parse_no_options(values)
        if not should_execute:
            argparse.ArgumentParser(description=__doc__).print_help()
            return 0
        _freeze_path, contract = _discover_controller_input(ROOT)
        result = execute_controller(contract, started=started)
        return 0 if result == "phase_a_pass" else 1
    except (ContractError, DriftError, OSError, subprocess.SubprocessError) as exc:
        print(f"H0 Phase-A controller rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
