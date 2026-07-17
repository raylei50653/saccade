#!/usr/bin/env python3
"""Independently verify a canonical ``h0_preseal_freeze_v3`` artifact.

This verifier intentionally does not import the assembler or the Phase-A
controller.  Its declaration literals, Git/tree reconstruction, byte hashing,
and I -> F -> S landing checks are a separate authority path.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = (
    "docs/modules/semantic/research/"
    "headline_bridge_full_decision_capture_declaration_20260713.md"
)
POLICY_BASE_HEAD = "7581c9720569e17593d1844ad494253ce664fed8"
POLICY_BASE_TREE = "2706ee3af0ddd6cd304f83289b575b2ae9b72fc6"
POLICY_PRESET = "configs/presets/mamba_whole_graph_m.yaml"
POLICY_PRESET_SHA256 = (
    "496c4ec22b497c70bc8409227513939b4cd86834bf2210475d0ad655be6937af"
)
POLICY_RESOLVED_SHA256 = (
    "c7a6dbb35168cba75249b7f2c67d8455b6f634732493e455a4bb920aab6d7782"
)
FREEZE_SCHEMA = "h0_preseal_freeze_v3"
LANDING_SCHEMA = "h0_authority_landing_v1"
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are transcribed from A7/RC1, rather than imported from the controller.
RUN_IDS = ("00_capture_off", "01_capture_on_1", "02_capture_on_2", "03_capture_on_3")
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
BUILD_ENV_KEYS = (
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
TRACE_CAPACITIES = (65536, 16384, 16384, 16384)
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
INOTIFY = (
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
MODEL_INPUTS = (
    "models/yolo/mamba_head_26m.engine",
    "models/yolo/yolo26m.pt",
    "models/yolo/yolo26m_backbone_640_best.engine",
    "runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt",
    "runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt",
)
REPOSITORY_INPUTS = (
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

IMPLEMENTATIONS = (
    ("scripts/tools/run_h0_phase_a.py", "h0_phase_a_controller_v1"),
    ("scripts/tools/run_h0_phase_a_child.py", "h0_phase_a_child_v1"),
    ("scripts/tools/h0_phase_a_execution_schema_v1.json", "h0_phase_a_execution_v1"),
    ("scripts/tools/h0_runtime_confinement.py", "h0_runtime_confinement_plan_v1"),
    ("scripts/tools/verify_h0_phase_a.py", "h0_phase_a_verifier_v1"),
    (
        "scripts/tools/export_headline_bridge_decision_trace.py",
        "h0_bridge_decision_trace_v2",
    ),
    (
        "scripts/tools/verify_headline_bridge_decision_trace.py",
        "h0_bridge_decision_trace_v2",
    ),
    ("scripts/tools/build_h0_preseal_freeze.py", "h0_preseal_freeze_v3"),
    (
        "scripts/tools/check_h0_bridge_decision_trace_contract.py",
        "h0_bridge_decision_trace_contract_v1",
    ),
    (
        "scripts/tools/h0_bridge_decision_trace_schema_v2.json",
        "h0_bridge_decision_trace_v2",
    ),
    ("scripts/tools/verify_h0_preseal_freeze.py", "h0_preseal_freeze_v3_verifier_v1"),
)
GOVERNANCE_ALLOWLIST = (
    DECLARATION_PATH,
    "docs/modules/semantic/research/headline_bridge_full_decision_capture_results_20260713.md",
    "docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_declaration_20260713.md",
    "docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_results_20260713.md",
    "docs/modules/semantic/research/closed/runtime_bridge_decision_path_identifiability_results_20260713.md",
    "docs/modules/semantic/research/evidence/p0_runtime_bridge_decision_path_20260713/manifest.json",
    "docs/modules/semantic/research/evidence/p0_runtime_bridge_decision_path_20260713/field_sufficiency.json",
    "docs/modules/semantic/research/evidence/p0_runtime_bridge_decision_path_20260713/decision_funnel.csv",
    "docs/modules/semantic/research/evidence/p0_runtime_bridge_decision_path_20260713/metrics.json",
    "docs/modules/semantic/README.md",
    "docs/modules/semantic/TODO.md",
    "docs/TODO.md",
)
RENAME_SOURCE = "docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_results_20260713.md"
RENAME_DESTINATION = "docs/modules/semantic/research/closed/runtime_bridge_decision_path_identifiability_results_20260713.md"
ADMITTED_RUNTIME_PATHS = {
    "include/tracking/tracker_gpu.hpp",
    "src/tracking/tracker_gpu.cu",
    "src/tracking/tracker_gpu_python.cpp",
    "src/saccade/perception/tracking/tracker_gpu.py",
    "src/saccade/perception/eval/stages.py",
}


class VerificationError(RuntimeError):
    """A malformed or unauthoritative v3 artifact/check-out."""


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"value cannot be canonical JSON: {exc}") from exc


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise VerificationError(f"duplicate JSON member {key!r}")
        value[key] = member
    return value


def load_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON value {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"malformed v3 JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise VerificationError("v3 artifact is non-canonical or is not an object")
    return value


def _git(root: Path, *args: str, text: bool = False) -> bytes | str:
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", *args],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise VerificationError(f"git {' '.join(args)} failed") from exc
    return out.decode("utf-8").strip() if text else out


def _tree_entry(root: Path, rev: str, path: str) -> tuple[str, str, str]:
    output = _git(root, "ls-tree", "-z", rev, "--", path)
    assert isinstance(output, bytes)
    rows = [row for row in output.split(b"\0") if row]
    if len(rows) != 1:
        raise VerificationError(f"missing or ambiguous Git entry {rev}:{path}")
    try:
        metadata, found = rows[0].split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
        if found.decode("utf-8", errors="strict") != path:
            raise ValueError
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerificationError(f"malformed Git entry for {path}") from exc
    return mode, kind, oid


def _blob(root: Path, rev: str, path: str) -> bytes:
    result = _git(root, "show", f"{rev}:{path}")
    assert isinstance(result, bytes)
    return result


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob_slot(root: Path, rev: str, path: str) -> dict[str, Any]:
    try:
        data = _blob(root, rev, path)
    except VerificationError:
        return {"state": "absent", "sha256": None}
    return {"state": "present", "sha256": _sha(data)}


def _classify(path: str) -> str:
    if path.startswith(("src/", "include/", "configs/", "cmake/")) or path in {
        "pyproject.toml",
        "uv.lock",
        "CMakeLists.txt",
        "setup.py",
        "setup.cfg",
        "Makefile",
    }:
        return "runtime_build_consumable"
    if path.startswith(("docs/", ".github/", "tests/", "scripts/")):
        return "non_runtime_recorded"
    return "runtime_build_consumable"


def _mutation_cases(cuda: str) -> list[tuple[str, str, str]]:
    claim = "        const int claim_record_index = h0_append_record(\n            h0.claim_records, h0.claim_capacity, h0.claim_cursor, h0.claim_overflow, h0_claim);\n"
    proposal = "        h0_append_record(h0.native_proposal_keys, h0.native_proposal_capacity,\n                         h0.native_proposal_cursor, h0.native_proposal_overflow, proposal);\n"
    field = "        proposal.frame = h0_frame;\n"
    if any(cuda.count(anchor) != 1 for anchor in (claim, proposal, field)):
        raise VerificationError("v2 mutation anchor is not unique")
    return [
        (
            "A3.2-delete-claim-append",
            "claim_record",
            cuda.replace(claim, "        const int claim_record_index = -1;\n"),
        ),
        (
            "A3.2-record-cursor-for-native-cursor",
            "native_universe_v2",
            cuda.replace(
                proposal,
                proposal.replace("h0.native_proposal_cursor", "h0.claim_cursor"),
            ),
        ),
        (
            "A3.2-native-key-field-after-append",
            "native_universe_v2",
            cuda.replace(
                field + "        proposal.proposing_cand_slot = cand;\n",
                "        proposal.proposing_cand_slot = cand;\n",
            ).replace(proposal, proposal + field),
        ),
        (
            "A4-claim-append-only-in-comment",
            "claim_record",
            cuda.replace(
                claim,
                "        // const int claim_record_index = h0_append_record(\n        //     h0.claim_records, h0.claim_capacity, h0.claim_cursor, h0.claim_overflow, h0_claim);\n        const int claim_record_index = -1;\n",
            ),
        ),
        (
            "A4-native-key-assign-commented-before-live-after",
            "native_universe_v2",
            cuda.replace(field, "        // proposal.frame = h0_frame;\n").replace(
                proposal, proposal + field
            ),
        ),
    ]


def _physical_executable(command: str) -> Path:
    """Reproduce the assembler's sole host executable selection algorithm."""
    found = shutil.which(command)
    if not found:
        raise VerificationError(f"required host executable is absent: {command}")
    candidate = Path(found).resolve(strict=True)
    details = candidate.stat(follow_symlinks=False)
    if candidate.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise VerificationError(
            f"host executable is not a physical regular file: {command}"
        )
    return candidate


def _host_file_record(path: Path) -> dict[str, Any]:
    """Rebuild an RC1 external-input record for an already physical path."""
    resolved = path.resolve(strict=True)
    details = resolved.lstat()
    if not stat.S_ISREG(details.st_mode) or resolved.is_symlink():
        raise VerificationError(f"host runtime input is not a regular file: {path}")
    data = resolved.read_bytes()
    return {
        "length": len(data),
        "logical_path": path.as_posix(),
        "realpath": resolved.as_posix(),
        "sha256": _sha(data),
        "symlink_chain": [],
    }


def _normalize_pci_bus_id(value: str) -> str:
    try:
        domain_bus, device_function = value.lower().split(":", 1)
        bus, device_function = device_function.split(":", 1)
        device, function = device_function.split(".", 1)
        normalized = (
            f"{int(domain_bus, 16):04x}:{int(bus, 16):02x}:"
            f"{int(device, 16):02x}.{int(function, 16)}"
        )
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"non-canonical PCI bus ID: {value!r}") from exc
    if int(function, 16) > 7:
        raise VerificationError(f"PCI function out of range: {value!r}")
    return normalized


def _independently_selected_gpu() -> dict[str, Any]:
    """Rebuild the A7 lexicographic physical-NVML selection without controller code."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except BaseException as exc:
        raise VerificationError(f"NVML initialization failed: {exc}") from exc
    records: list[dict[str, Any]] = []
    try:

        def text(value: object) -> str:
            return (
                value.decode("utf-8", errors="strict")
                if isinstance(value, bytes)
                else str(value)
            )

        driver = text(pynvml.nvmlSystemGetDriverVersion())
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            records.append(
                {
                    "compute_capability": f"{int(major)}.{int(minor)}",
                    "driver": driver,
                    "name": text(pynvml.nvmlDeviceGetName(handle)),
                    "normalized_pci_bus_id": _normalize_pci_bus_id(text(pci.busId)),
                    "total_memory": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    "uuid": text(pynvml.nvmlDeviceGetUUID(handle)),
                    "vbios": text(pynvml.nvmlDeviceGetVbiosVersion(handle)),
                }
            )
    finally:
        pynvml.nvmlShutdown()
    if not records:
        raise VerificationError("no physical NVIDIA GPU is available")
    records.sort(key=lambda record: record["normalized_pci_bus_id"])
    if len({record["normalized_pci_bus_id"] for record in records}) != len(records):
        raise VerificationError("duplicate normalized NVIDIA PCI bus ID")
    return records[0]


def _independent_host_execution_inputs(root: Path) -> dict[str, Any]:
    """Reconstruct every host-selected execution input from A7's algorithms.

    The artifact is deliberately not consulted for path selection.  This closes
    the self-consistent-but-substituted ``tool_runtime`` model rejected in P1.
    """
    tool_paths = {
        name: _physical_executable(name).as_posix()
        for name in ("git", "ldd", "nvcc", "readelf", "uv")
    }
    python = root / ".venv/bin/python"
    if not python.is_file() or python.is_symlink():
        raise VerificationError("frozen .venv/bin/python is absent or symlinked")
    pyvenv_config = root / ".venv/pyvenv.cfg"
    if not pyvenv_config.is_file() or pyvenv_config.is_symlink():
        raise VerificationError("frozen .venv/pyvenv.cfg is absent or symlinked")
    try:
        query = subprocess.run(
            [
                python.as_posix(),
                "-I",
                "-c",
                "import pathlib,torch; print((pathlib.Path(torch.__file__).resolve().parent/'lib').as_posix()); import tensorrt_libs; print(pathlib.Path(tensorrt_libs.__file__).resolve().parent.as_posix())",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationError(
            "frozen Python could not derive runtime library directories"
        ) from exc
    if len(query) != 2:
        raise VerificationError(
            "frozen Python did not derive exactly two runtime library directories"
        )
    cuda = Path(tool_paths["nvcc"]).parent.parent / "lib64"
    library_dirs = {
        "tensorrt_library_dir": Path(query[1]).resolve(strict=True).as_posix(),
        "pytorch_library_dir": Path(query[0]).resolve(strict=True).as_posix(),
        "cuda_library_dir": cuda.resolve(strict=True).as_posix(),
    }
    if any(
        not Path(value).is_dir() or Path(value).is_symlink()
        for value in library_dirs.values()
    ):
        raise VerificationError(
            "a independently derived runtime library directory is non-physical or absent"
        )
    candidates = [Path(path) for path in tool_paths.values()] + [
        python,
        pyvenv_config,
    ]
    for directory in library_dirs.values():
        candidates.extend(
            sorted(
                path
                for path in Path(directory).rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        )
    tool_runtime = sorted(
        (_host_file_record(path) for path in candidates),
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    if len({record["realpath"] for record in tool_runtime}) != len(tool_runtime):
        raise VerificationError("independent tool/runtime inventory has duplicates")
    return {
        "tool_paths": tool_paths,
        "library_dirs": library_dirs,
        "gpu": _independently_selected_gpu(),
        "tool_runtime": tool_runtime,
    }


def freeze_path(head: str) -> str:
    if not HEAD_RE.fullmatch(head):
        raise VerificationError("instrumentation head is not 40 lowercase hex")
    return f"docs/modules/semantic/research/evidence/h0_preseal_freeze_{head}/h0_preseal_freeze_v3.json"


def _matrix() -> dict[str, dict[str, list[str]]]:
    all_paths = C_PATHS + D_PATHS + V_PATHS
    value = {
        result: {"required": list(C_PATHS), "forbidden": list(D_PATHS + V_PATHS)}
        for result in RESULTS[:8]
    }
    value["capture_perturbs_policy"] = {
        "required": list(C_PATHS + D_PATHS),
        "forbidden": list(V_PATHS),
    }
    for result in ("packet_invalid", "phase_a_pass"):
        value[result] = {"required": list(all_paths), "forbidden": []}
    return value


def expected_execution_constants(root: Path) -> dict[str, Any]:
    physical = root.resolve(strict=True).as_posix()
    child_vectors = [
        [
            f"{physical}/.venv/bin/python",
            "-I",
            "-B",
            f"{physical}/scripts/tools/run_h0_phase_a_child.py",
            "--run-id",
            run_id,
        ]
        for run_id in RUN_IDS
    ]
    return {
        "actual_loaded_input_attestation": "h0_runtime_inputs_v1",
        "bound_input_algorithms": {
            "bound_inputs": "h0_bound_inputs_v1",
            "repository": "git_ls_tree_r_full_tree_z",
            "sequence": "h0_sequence_inputs_v1",
            "actual_loaded_attestation": "h0_runtime_inputs_v1",
        },
        "build_environment_algorithm": "h0_build_environment_v2",
        "build_environment_keys": list(BUILD_ENV_KEYS),
        "build_vectors": [list(v) for v in BUILD_VECTORS],
        "c_paths": list(C_PATHS),
        "canonicalization": {
            "json": "utf8_lexicographic_keys_compact_finite_trailing_lf_v1",
            "checksums": "lowercase_sha256_two_spaces_posix_path_sorted_utf8_bytes_v1",
        },
        "checkpoints": list(CHECKPOINTS),
        "child_vectors": child_vectors,
        "child_environment_algorithm": "h0_child_environment_v1",
        "child_environment_template": {
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
        },
        "d_paths": list(D_PATHS),
        "deadline_seconds": 3600,
        "environment_keys": list(ENV_KEYS),
        "evaluator_argv_prefix": list(EVALUATOR_PREFIX),
        "inotify_mask": list(INOTIFY),
        "model_inputs": list(MODEL_INPUTS),
        "operator_vector": [
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/tools/run_h0_phase_a.py",
        ],
        "ordered_run_plan": list(RUN_IDS),
        "publication_rollback": {
            "publication": "atomic_rename_incomplete_to_final_then_fsync_parent_v1",
            "rollback": "remove_partial_D_V_before_checksums_or_leave_incomplete_unpublished_v1",
        },
        "required_repository_inputs": list(REPOSITORY_INPUTS),
        "result_enum": list(RESULTS),
        "result_matrix": _matrix(),
        "trace_capacities": list(TRACE_CAPACITIES),
        "trace_lifecycle": {
            "schema": "h0_trace_lifecycle_v1",
            "capture_off": [
                "set_research_h0_bridge_trace(false,65536,16384,16384,16384)"
            ],
            "capture_on": [
                "set_research_h0_bridge_trace(true,65536,16384,16384,16384)",
                "clear_research_h0_bridge_trace()",
                "drain_research_h0_bridge_trace(seq=MOT17-04-SDP,capture_phase=phase_a,require_candidate_exposure=true,require_commit_exposure=false,capture_run_uuid=<CONTROLLER_UUID>)",
            ],
        },
        "v_paths": list(V_PATHS),
    }


def _require_exact_members(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise VerificationError(
            f"{name} has missing or unknown members: {sorted(set(value) ^ expected)}"
        )


def _verify_landing_shape(value: Mapping[str, Any], head: str) -> None:
    expected_path = freeze_path(head)
    _require_exact_members(
        value,
        {"schema", "artifact_path", "declaration_path", "post_head_allowed_paths"},
        "authority_landing",
    )
    if (
        value.get("schema") != LANDING_SCHEMA
        or value.get("artifact_path") != expected_path
        or value.get("declaration_path") != DECLARATION_PATH
        or value.get("post_head_allowed_paths") != [expected_path, DECLARATION_PATH]
    ):
        raise VerificationError("authority_landing_v1 literals differ from RC2")


def _verify_implementation_bindings(
    value: object, root: Path, head: str, *, check_worktree: bool
) -> None:
    if not isinstance(value, list) or len(value) != len(IMPLEMENTATIONS):
        raise VerificationError("implementation bindings have wrong cardinality")
    expected_paths = [path for path, _ in IMPLEMENTATIONS]
    if [
        entry.get("path") if isinstance(entry, dict) else None for entry in value
    ] != expected_paths:
        raise VerificationError(
            "implementation binding paths are not the one ordered v3 set"
        )
    for entry, (path, identity) in zip(value, IMPLEMENTATIONS):
        if not isinstance(entry, dict):
            raise VerificationError("implementation binding is not an object")
        _require_exact_members(
            entry,
            {"path", "identity", "mode", "git_type", "git_object", "length", "sha256"},
            f"implementation binding {path}",
        )
        if entry["identity"] != identity or entry["path"] != path:
            raise VerificationError(f"implementation identity drift: {path}")
        mode, kind, oid = _tree_entry(root, head, path)
        blob = _blob(root, head, path)
        if mode not in {"100644", "100755"} or kind != "blob":
            raise VerificationError(
                f"implementation path is not a regular Git blob: {path}"
            )
        expected = {
            "path": path,
            "identity": identity,
            "mode": mode,
            "git_type": kind,
            "git_object": oid,
            "length": len(blob),
            "sha256": _sha(blob),
        }
        if entry != expected:
            raise VerificationError(
                f"implementation Git object or byte hash mismatch: {path}"
            )
        if check_worktree:
            candidate = root / path
            try:
                mode_bits = candidate.lstat().st_mode
                data = candidate.read_bytes()
            except OSError as exc:
                raise VerificationError(
                    f"implementation file unreadable: {path}"
                ) from exc
            if candidate.is_symlink() or not stat.S_ISREG(mode_bits) or data != blob:
                raise VerificationError(
                    f"implementation worktree byte/mode drift: {path}"
                )


def _verify_host_execution_inputs(
    controller: Mapping[str, Any], inventory: Mapping[str, Any], root: Path
) -> None:
    """Reject self-consistent host substitutions before inventory replay."""
    independent_host = _independent_host_execution_inputs(root)
    if controller["tool_paths"] != independent_host["tool_paths"]:
        raise VerificationError("tool paths differ from independent which() selection")
    if controller["library_dirs"] != independent_host["library_dirs"]:
        raise VerificationError(
            "library directories differ from independent Python/nvcc derivation"
        )
    if controller["gpu"] != independent_host["gpu"]:
        raise VerificationError("GPU identity differs from independent NVML selection")
    if inventory["tool_runtime"] != independent_host["tool_runtime"]:
        raise VerificationError(
            "tool/runtime inventory differs from independent host expansion"
        )


def _verify_controller_input(value: object, root: Path, head: str) -> None:
    if not isinstance(value, dict):
        raise VerificationError("phase_a_controller_input is not an object")
    required = {
        "authority_landing",
        "bound_inputs",
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
    _require_exact_members(value, required, "phase_a_controller_input")
    if (
        value["schema"] != "h0_phase_a_controller_v1"
        or value["document_type"] != "controller_input"
        or value["instrumentation_head"] != head
    ):
        raise VerificationError("controller-input identity differs from A7")
    if value["execution_constants"] != expected_execution_constants(root):
        raise VerificationError("A7/RC1 execution constants drift")
    _verify_landing_shape(value["authority_landing"], head)
    expected_root = f"docs/modules/semantic/research/evidence/h0_phase_a_{head}"
    if (
        value["evidence_root"] != expected_root
        or value["incomplete_root"] != expected_root + ".incomplete"
    ):
        raise VerificationError("evidence-root derivation drift")
    if value["repository_root"] != root.resolve(strict=True).as_posix():
        raise VerificationError("controller input repository root drift")
    if not isinstance(value["gpu"], dict) or set(value["gpu"]) != {
        "compute_capability",
        "driver",
        "name",
        "normalized_pci_bus_id",
        "total_memory",
        "uuid",
        "vbios",
    }:
        raise VerificationError("GPU identity has missing or unknown members")
    if not isinstance(value["library_dirs"], dict) or set(value["library_dirs"]) != {
        "cuda_library_dir",
        "pytorch_library_dir",
        "tensorrt_library_dir",
    }:
        raise VerificationError(
            "library-directory identity has missing or unknown members"
        )
    libraries = list(value["library_dirs"].values())
    if len(libraries) != len(set(libraries)):
        raise VerificationError("library-directory identity is ambiguous")
    for directory in libraries:
        candidate = Path(str(directory))
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_dir()
        ):
            raise VerificationError("library-directory identity is non-physical")
    if not isinstance(value["tool_paths"], dict) or set(value["tool_paths"]) != {
        "git",
        "ldd",
        "nvcc",
        "readelf",
        "uv",
    }:
        raise VerificationError("tool-path identity has missing or unknown members")
    for path in value["tool_paths"].values():
        candidate = Path(str(path))
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise VerificationError("tool-path identity is non-physical")
    inventory = value["bound_inputs"]
    if (
        not isinstance(inventory, dict)
        or set(inventory)
        != {
            "schema",
            "digest",
            "repository",
            "models_engines",
            "sequence",
            "tool_runtime",
        }
        or inventory.get("schema") != "h0_bound_inputs_v1"
    ):
        raise VerificationError("h0_bound_inputs_v1 shape drift")
    _verify_host_execution_inputs(value, inventory, root)
    if value["sequence_input_digest"] != inventory.get("sequence", {}).get("digest"):
        raise VerificationError("controller sequence digest mismatch")
    repository = inventory["repository"]
    if not isinstance(repository, list) or [
        record.get("path") for record in repository if isinstance(record, dict)
    ] != sorted(
        [record.get("path") for record in repository if isinstance(record, dict)],
        key=lambda item: str(item).encode("utf-8"),
    ):
        raise VerificationError("bound repository inventory ordering drift")
    if not set(REPOSITORY_INPUTS).issubset(
        {record.get("path") for record in repository if isinstance(record, dict)}
    ):
        raise VerificationError("bound repository inventory omits A7 inputs")
    raw_tree = _git(root, "ls-tree", "-r", "--full-tree", "-z", head)
    assert isinstance(raw_tree, bytes)
    expected_repository: list[dict[str, Any]] = []
    overlays = set(value["authority_landing"]["post_head_allowed_paths"])
    for row in (part for part in raw_tree.split(b"\0") if part):
        try:
            meta, raw_path = row.split(b"\t", 1)
            mode, kind, oid = meta.decode("ascii").split(" ")
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise VerificationError("non-canonical repository tree record") from exc
        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise VerificationError(f"unsupported bound repository entry {path}")
        blob = _blob(root, head, path)
        expected_repository.append(
            {
                "git_object": oid,
                "git_type": "blob",
                "kind": "symlink" if mode == "120000" else "regular",
                "length": len(blob),
                "mode": mode,
                "path": path,
                "sha256": _sha(blob),
            }
        )
        candidate = root / path
        try:
            details = candidate.lstat()
            if mode == "120000":
                if not candidate.is_symlink():
                    raise VerificationError(
                        f"repository symlink worktree drift: {path}"
                    )
            else:
                current = candidate.read_bytes()
                if candidate.is_symlink() or not stat.S_ISREG(details.st_mode):
                    raise VerificationError(
                        f"repository worktree entry is not regular: {path}"
                    )
                if path not in overlays and current != blob:
                    raise VerificationError(f"repository worktree byte drift: {path}")
        except OSError as exc:
            raise VerificationError(
                f"repository worktree entry missing: {path}"
            ) from exc
    if repository != expected_repository:
        raise VerificationError("bound repository Git-object/byte inventory drift")
    for category in ("models_engines", "tool_runtime"):
        records = inventory[category]
        if not isinstance(records, list):
            raise VerificationError(f"bound {category} is not an array")
        keys = [
            record.get("logical_path") for record in records if isinstance(record, dict)
        ]
        if len(keys) != len(records) or keys != sorted(
            keys, key=lambda item: str(item).encode("utf-8")
        ):
            raise VerificationError(f"bound {category} ordering drift")
        realpaths: set[str] = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "length",
                "logical_path",
                "realpath",
                "sha256",
                "symlink_chain",
            }:
                raise VerificationError(f"bound {category} record shape drift")
            logical = str(record["logical_path"])
            candidate = Path(logical) if logical.startswith("/") else root / logical
            try:
                resolved = candidate.resolve(strict=True)
                details = resolved.lstat()
                data = resolved.read_bytes()
            except OSError as exc:
                raise VerificationError(
                    f"bound {category} path is unavailable: {logical}"
                ) from exc
            if (
                not stat.S_ISREG(details.st_mode)
                or record["realpath"] != resolved.as_posix()
                or record["length"] != len(data)
                or record["sha256"] != _sha(data)
                or resolved.as_posix() in realpaths
            ):
                raise VerificationError(
                    f"bound {category} file identity drift: {logical}"
                )
            realpaths.add(resolved.as_posix())
        if category == "models_engines" and tuple(keys) != MODEL_INPUTS:
            raise VerificationError("bound model/engine paths differ from A7")
    sequence = inventory["sequence"]
    if (
        not isinstance(sequence, dict)
        or set(sequence) != {"algorithm", "digest", "files", "root"}
        or sequence.get("algorithm") != "h0_sequence_inputs_v1"
        or sequence.get("root") != "datasets/MOT17/train/MOT17-04-SDP"
    ):
        raise VerificationError("sequence inventory shape/algorithm drift")
    sequence_root = root / str(sequence["root"])
    rebuilt_files: list[dict[str, Any]] = []
    try:
        for directory, dirs, filenames in os.walk(sequence_root, followlinks=False):
            current = Path(directory)
            relative_dir = current.relative_to(sequence_root)
            if current.is_symlink():
                raise VerificationError("sequence contains a symlink directory")
            if relative_dir == Path("."):
                dirs[:] = sorted(name for name in dirs if name not in {"gt", "det"})
            else:
                dirs.sort()
            for filename in sorted(filenames):
                path = current / filename
                details = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(details.st_mode):
                    raise VerificationError("sequence contains a non-regular input")
                data = path.read_bytes()
                rebuilt_files.append(
                    {
                        "path": path.relative_to(sequence_root).as_posix(),
                        "length": len(data),
                        "sha256": _sha(data),
                    }
                )
    except OSError as exc:
        raise VerificationError("sequence inventory cannot be rebuilt") from exc
    rebuilt_files.sort(key=lambda record: record["path"].encode("utf-8"))
    expected_sequence_digest = _sha(
        canonical_json({"algorithm": "h0_sequence_inputs_v1", "files": rebuilt_files})
    )
    if (
        sequence.get("files") != rebuilt_files
        or sequence.get("digest") != expected_sequence_digest
    ):
        raise VerificationError("sequence-input inventory/digest drift")
    digest_payload = {
        "models_engines": inventory["models_engines"],
        "repository": repository,
        "schema": "h0_bound_inputs_v1",
        "sequence": sequence,
        "tool_runtime": inventory["tool_runtime"],
    }
    if inventory.get("digest") != _sha(canonical_json(digest_payload)):
        raise VerificationError("h0_bound_inputs_v1 aggregate digest drift")


def _verify_v2_retained(value: Mapping[str, Any], root: Path, head: str) -> None:
    if (
        value["capture_schema_version"] != "h0_bridge_decision_trace_v2"
        or value["policy_base_head"] != POLICY_BASE_HEAD
        or value["policy_base_tree"] != POLICY_BASE_TREE
    ):
        raise VerificationError("retained v2 policy-base identity drift")
    if (
        _git(root, "rev-parse", f"{POLICY_BASE_HEAD}^{{tree}}", text=True)
        != POLICY_BASE_TREE
    ):
        raise VerificationError("policy base tree cannot be reconstructed")
    if value["instrumentation_tree"] != _git(
        root, "rev-parse", f"{head}^{{tree}}", text=True
    ):
        raise VerificationError("instrumentation tree mismatch")
    tree_list = _git(root, "ls-tree", "-r", "--full-tree", head)
    assert isinstance(tree_list, bytes)
    if value["tree_list_sha256"] != _sha(tree_list):
        raise VerificationError("repository provenance tree-list drift")
    diff = _git(
        root,
        "diff",
        "--no-color",
        "--binary",
        "--full-index",
        "--no-renames",
        f"{POLICY_BASE_HEAD}..{head}",
    )
    assert isinstance(diff, bytes)
    if value["full_repo_diff_sha256"] != _sha(diff):
        raise VerificationError("complete v2 repository diff drift")
    projection_args = [
        "diff",
        "--no-color",
        "--binary",
        "--full-index",
        "--no-renames",
        f"{POLICY_BASE_HEAD}..{head}",
        "--",
        ".",
    ] + [f":(exclude){path}" for path in GOVERNANCE_ALLOWLIST]
    projection_diff = _git(root, *projection_args)
    assert isinstance(projection_diff, bytes)
    names = _git(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        f"{POLICY_BASE_HEAD}..{head}",
        "--",
        ".",
        *[f":(exclude){path}" for path in GOVERNANCE_ALLOWLIST],
    )
    assert isinstance(names, bytes)
    projection_paths = names.decode("utf-8").splitlines()
    projection = value["runtime_policy_code_projection_v1"]
    if not isinstance(projection, dict):
        raise VerificationError("retained v2 projection is not an object")
    _require_exact_members(
        projection,
        {
            "classifier",
            "projection_diff_sha256",
            "paths",
            "excluded_governance_allowlist",
            "governance_rename_v1",
            "projection_admitted",
        },
        "retained v2 projection",
    )
    classes = {path: _classify(path) for path in projection_paths}
    expected_paths = [
        {
            "path": path,
            "class": classes[path],
            "before": _blob_slot(root, POLICY_BASE_HEAD, path),
            "after": _blob_slot(root, head, path),
        }
        for path in sorted(projection_paths)
    ]
    expected_excluded = [
        {
            "path": path,
            "before": _blob_slot(root, POLICY_BASE_HEAD, path),
            "after": _blob_slot(root, head, path),
        }
        for path in GOVERNANCE_ALLOWLIST
    ]
    expected_rename = {
        "schema": "governance_rename_v1",
        "source_path": RENAME_SOURCE,
        "destination_path": RENAME_DESTINATION,
        "source_blob_before": _blob_slot(root, POLICY_BASE_HEAD, RENAME_SOURCE),
        "source_blob_after": _blob_slot(root, head, RENAME_SOURCE),
        "destination_blob_before": _blob_slot(
            root, POLICY_BASE_HEAD, RENAME_DESTINATION
        ),
        "destination_blob_after": _blob_slot(root, head, RENAME_DESTINATION),
    }
    admitted = not [
        path
        for path, kind in classes.items()
        if kind == "runtime_build_consumable" and path not in ADMITTED_RUNTIME_PATHS
    ]
    if projection != {
        "classifier": "h0_projection_path_class_v1",
        "projection_diff_sha256": _sha(projection_diff),
        "paths": expected_paths,
        "excluded_governance_allowlist": expected_excluded,
        "governance_rename_v1": expected_rename,
        "projection_admitted": admitted,
    }:
        raise VerificationError("retained v2 projection/admission drift")
    policy = value["policy_target"]
    if not isinstance(policy, dict) or policy != {
        "preset": POLICY_PRESET,
        "preset_sha256": POLICY_PRESET_SHA256,
        "resolved_schema": "resolved_bridge_policy_config_v1",
        "resolved_fingerprint": POLICY_RESOLVED_SHA256,
    }:
        raise VerificationError("retained v2 policy identity drift")
    if value["assembler_sha256"] != next(
        entry["sha256"]
        for entry in value["implementation_bindings"]
        if entry["path"] == "scripts/tools/build_h0_preseal_freeze.py"
    ):
        raise VerificationError("assembler self-identity drift")
    try:
        from check_h0_bridge_decision_trace_contract import (
            CHECKER_PATH,
            CUDA_PATH,
            EXPORT_PATH,
            HEADER_PATH,
            PYTHON_BINDING_PATH,
            SCHEMA_PATH,
            TRACKER_WRAPPER_PATH,
            VERIFIER_PATH,
            coverage_report,
        )
    except ImportError as exc:  # pragma: no cover - execution admission
        raise VerificationError("trace-contract checker is unavailable") from exc
    checker_paths = (
        HEADER_PATH,
        CUDA_PATH,
        PYTHON_BINDING_PATH,
        TRACKER_WRAPPER_PATH,
        EXPORT_PATH,
        VERIFIER_PATH,
        CHECKER_PATH,
        SCHEMA_PATH,
    )
    local_root = Path(__file__).resolve().parents[2]
    checker_rel = {
        path: path.resolve().relative_to(local_root).as_posix()
        for path in checker_paths
    }
    overrides = {
        path: _blob(root, head, checker_rel[path]).decode("utf-8")
        for path in checker_paths
    }
    coverage, failures = coverage_report(source_overrides=overrides)
    if (
        value["h0_coverage_v2"] != coverage
        or coverage.get("all_components_true") is not True
        or failures
    ):
        raise VerificationError("retained v2 coverage object drift")
    mutation_rows: list[dict[str, Any]] = []
    all_pass = True
    for name, component, mutated in _mutation_cases(overrides[CUDA_PATH]):
        report, _ = coverage_report(source_overrides={**overrides, CUDA_PATH: mutated})
        flipped = report["coverage_components"][component] is False
        all_pass = all_pass and flipped
        mutation_rows.append(
            {
                "case": name,
                "expected_false_component": component,
                "component_false": flipped,
            }
        )
    if (
        value["mutation_admission"] != {"cases": mutation_rows, "all_pass": all_pass}
        or not all_pass
    ):
        raise VerificationError("retained v2 mutation admission drift")
    expected_seal_paths = sorted(
        {*checker_rel.values(), POLICY_PRESET, *(path for path, _ in IMPLEMENTATIONS)}
    )
    if value["input_binding"] != {
        "mode": "head_blobs_with_post_assembly_reverify",
        "seal_relevant_paths": expected_seal_paths,
    }:
        raise VerificationError("retained v2 input-binding drift")
    produced_by = value["produced_by"]
    if (
        not isinstance(produced_by, dict)
        or set(produced_by) != {"command_line", "python", "platform", "git"}
        or not isinstance(produced_by["command_line"], list)
        or not all(isinstance(part, str) for part in produced_by["command_line"])
        or not all(
            isinstance(produced_by[key], str) for key in ("python", "platform", "git")
        )
    ):
        raise VerificationError("retained v2 producer provenance shape drift")


def verify_artifact(
    value: Mapping[str, Any],
    root: Path,
    *,
    check_worktree: bool = False,
    require_complete: bool = True,
) -> dict[str, Any]:
    required = {
        "freeze_schema_version",
        "capture_schema_version",
        "instrumentation_head",
        "instrumentation_tree",
        "tree_list_sha256",
        "policy_base_head",
        "policy_base_tree",
        "full_repo_diff_sha256",
        "runtime_policy_code_projection_v1",
        "h0_coverage_v2",
        "mutation_admission",
        "policy_target",
        "assembler_sha256",
        "input_binding",
        "produced_by",
        "authority_landing",
        "implementation_bindings",
        "phase_a_controller_input",
        "complete",
        "problems",
    }
    _require_exact_members(value, required, "h0_preseal_freeze_v3")
    if (
        value["freeze_schema_version"] != FREEZE_SCHEMA
        or not isinstance(value["instrumentation_head"], str)
        or not HEAD_RE.fullmatch(value["instrumentation_head"])
    ):
        raise VerificationError("freeze schema or instrumentation head mismatch")
    head = value["instrumentation_head"]
    _verify_landing_shape(value["authority_landing"], head)
    _verify_implementation_bindings(
        value["implementation_bindings"], root, head, check_worktree=check_worktree
    )
    _verify_controller_input(value["phase_a_controller_input"], root, head)
    _verify_v2_retained(value, root, head)
    if (
        type(value["complete"]) is not bool
        or not isinstance(value["problems"], list)
        or not all(isinstance(item, str) for item in value["problems"])
    ):
        raise VerificationError("complete/problems type drift")
    if value["complete"] and value["problems"]:
        raise VerificationError("complete=true has unresolved admissions")
    if require_complete and value["complete"] is not True:
        raise VerificationError("v3 freeze is incomplete")
    return {
        "schema": "h0_preseal_freeze_v3_verifier_v1",
        "instrumentation_head": head,
        "complete": value["complete"],
        "valid": True,
    }


def _single_parent(root: Path, commit: str) -> str:
    parents = _git(root, "show", "-s", "--format=%P", commit, text=True)
    assert isinstance(parents, str)
    values = parents.split()
    if len(values) != 1 or not HEAD_RE.fullmatch(values[0]):
        raise VerificationError(f"{commit} is not an ordinary one-parent commit")
    return values[0]


def _changed_paths(root: Path, before: str, after: str) -> list[str]:
    raw = _git(root, "diff", "--name-only", "--no-renames", before, after)
    assert isinstance(raw, bytes)
    return raw.decode("utf-8").splitlines()


def verify_authority_landing(
    root: Path, artifact: Mapping[str, Any], *, checkout: str | None = None
) -> dict[str, str]:
    """Verify RC2's exact I -> F -> S (= execution checkout) relation."""
    head = artifact.get("instrumentation_head")
    if not isinstance(head, str) or not HEAD_RE.fullmatch(head):
        raise VerificationError("artifact has no valid instrumentation head")
    execution = checkout or _git(root, "rev-parse", "HEAD", text=True)
    assert isinstance(execution, str)
    if not HEAD_RE.fullmatch(execution):
        raise VerificationError("execution checkout is not a commit")
    freeze_commit = _single_parent(root, execution)
    instrumentation = _single_parent(root, freeze_commit)
    if instrumentation != head:
        raise VerificationError("execution checkout does not derive I -> F -> S")
    path = freeze_path(head)
    if _changed_paths(root, instrumentation, freeze_commit) != [path]:
        raise VerificationError("freeze commit has an extra or missing post-head delta")
    if _changed_paths(root, freeze_commit, execution) != [DECLARATION_PATH]:
        raise VerificationError("seal commit has an extra or missing post-head delta")
    mode, kind, _oid = _tree_entry(root, freeze_commit, path)
    if mode not in {"100644", "100755"} or kind != "blob":
        raise VerificationError("freeze artifact is not a tracked regular blob")
    frozen = _blob(root, freeze_commit, path)
    sealed = _blob(root, execution, path)
    if frozen != sealed or frozen != canonical_json(dict(artifact)) + b"\n":
        raise VerificationError("v3 bytes differ between freeze/seal/current checkout")
    before = _blob(root, freeze_commit, DECLARATION_PATH)
    after = _blob(root, execution, DECLARATION_PATH)
    event = re.compile(
        rf"^\| (?P<date>[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}) \| `{head}` \| `{freeze_commit}` \| `SEALED` \|$"
    )
    if not after.startswith(before):
        raise VerificationError("seal declaration was not append-only")
    try:
        suffix_text = after[len(before) :].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("seal declaration suffix is not UTF-8") from exc
    lines = after.decode("utf-8", errors="strict").splitlines()
    # The grammar allows the review date but nothing else; checking the suffix
    # directly keeps the owner event to one newly appended line.
    match = event.fullmatch(suffix_text[:-1]) if suffix_text.endswith("\n") else None
    if suffix_text.count("\n") != 1 or not suffix_text.endswith("\n") or match is None:
        raise VerificationError(
            "seal row is missing, duplicated, wrong, or not the exact append"
        )
    try:
        datetime.date.fromisoformat(match.group("date"))
    except ValueError as exc:
        raise VerificationError(
            "seal row date is not a valid ISO calendar date"
        ) from exc
    if len([entry for entry in lines if event.fullmatch(entry)]) != 1:
        raise VerificationError("seal row is missing or duplicated")
    return {
        "instrumentation_head": head,
        "freeze_commit": freeze_commit,
        "seal_commit": execution,
        "execution_checkout": execution,
    }


def verify_artifact_path(
    path: Path,
    root: Path,
    *,
    require_complete: bool = True,
    verify_landing: bool = True,
) -> dict[str, Any]:
    try:
        details = path.lstat()
    except OSError as exc:
        raise VerificationError(f"v3 artifact is unreadable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise VerificationError("v3 artifact is not a physical regular file")
    value = load_canonical_json(path)
    expected = root / freeze_path(str(value.get("instrumentation_head", "")))
    if path.absolute() != expected.absolute():
        raise VerificationError("v3 artifact is not at its deterministic RC2 path")
    report = verify_artifact(
        value, root, check_worktree=True, require_complete=require_complete
    )
    if verify_landing:
        report["landing"] = verify_authority_landing(root, value)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify_artifact_path(args.artifact, ROOT),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
