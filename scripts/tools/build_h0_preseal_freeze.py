#!/usr/bin/env python3
"""Assemble the sole H0 pre-seal artifact (``h0_preseal_freeze_v3``).

Declaration authority: H0 declaration §2 / A2.2 / Amendment 6. This tool is a
non-production offline input under the A2.1 allowance: it reads git history,
H0 sources, and the trace schema; it is never a build input or runtime-policy
path. It runs no capture and can emit no H0 execution terminal — an incomplete
result is the pre-seal engineering status that prohibits seal.

RC2's landing topology deliberately separates the reviewed implementation head
from the later artifact and owner-event commits.  This tool runs only at clean
``instrumentation_head`` I and writes the one deterministic F-path; it never
creates either commit, writes an owner event, or executes Phase A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from check_h0_bridge_decision_trace_contract import (  # noqa: E402
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

FREEZE_SCHEMA_VERSION = "h0_preseal_freeze_v3"
CAPTURE_SCHEMA_VERSION = "h0_bridge_decision_trace_v2"
CLASSIFIER_VERSION = "h0_projection_path_class_v1"
LANDING_SCHEMA_VERSION = "h0_authority_landing_v1"
DECLARATION_PATH = (
    "docs/modules/semantic/research/"
    "headline_bridge_full_decision_capture_declaration_20260713.md"
)

# §2 frozen policy base (pre-H0 decision path).
POLICY_BASE_HEAD = "7581c9720569e17593d1844ad494253ce664fed8"
POLICY_BASE_TREE = "2706ee3af0ddd6cd304f83289b575b2ae9b72fc6"

# §2 sole policy target (Amendment 5: m, never s).
POLICY_TARGET_PRESET = "configs/presets/mamba_whole_graph_m.yaml"
POLICY_TARGET_PRESET_SHA256 = (
    "496c4ec22b497c70bc8409227513939b4cd86834bf2210475d0ad655be6937af"
)
POLICY_TARGET_RESOLVED_SHA256 = (
    "c7a6dbb35168cba75249b7f2c67d8455b6f634732493e455a4bb920aab6d7782"
)

# §2 h0_governance_docs_allowlist_v1 — the only paths excluded from the
# runtime projection. Fixed; Amendment 6 does not extend it.
GOVERNANCE_ALLOWLIST = (
    "docs/modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md",
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

# §2 governance_rename_v1 — the one allowed P0 results move.
GOVERNANCE_RENAME_SOURCE = "docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_results_20260713.md"
GOVERNANCE_RENAME_DESTINATION = "docs/modules/semantic/research/closed/runtime_bridge_decision_path_identifiability_results_20260713.md"

# Amendment 6 §A6.2 — frozen admitted runtime surface.
ADMITTED_RUNTIME_PATHS = frozenset(
    {
        "include/tracking/tracker_gpu.hpp",
        "src/tracking/tracker_gpu.cu",
        "src/tracking/tracker_gpu_python.cpp",
        "src/saccade/perception/tracking/tracker_gpu.py",
        "src/saccade/perception/eval/stages.py",
    }
)

# Amendment 6 Correction 1 — content-pinned admissions for the frozen CUDA
# substrate (#214/#216).  A path here is admitted only while its after-blob
# SHA-256 at the instrumentation head equals the pinned value; any further
# change to these files requires a fresh append-only admission extension, not
# an ordinary review.
ADMITTED_RUNTIME_BLOBS = {
    "CMakeLists.txt": (
        "3d5a576d632109255c2f0537fbd9b302b66d69a61ee90759301ae4da3960890e"
    ),
    "pyproject.toml": (
        "85ec43e498adfdf0d433b6a8a621055a5b5232999ec8366e2616125d2cc8627b"
    ),
    "uv.lock": ("c45f37916351fd246eb54bb8bac7fd28df7816180b551c50574c189d85687923"),
    "src/perception/preprocessor.cpp": (
        "11aa959b94efc49e4bb4544beb1462757c229c2f5175980540636f61c1a98313"
    ),
}

RUNTIME_PREFIXES = ("src/", "include/", "configs/", "cmake/")
ROOT_BUILD_MANIFESTS = frozenset(
    {"pyproject.toml", "uv.lock", "CMakeLists.txt", "setup.py", "setup.cfg", "Makefile"}
)
NON_RUNTIME_PREFIXES = ("docs/", ".github/", "tests/", "scripts/")

ASSEMBLER_PATH = Path(__file__).resolve()
PRESET_PATH = ROOT / POLICY_TARGET_PRESET

# A7/RC1 plus RC2.  This ordered table is intentionally the only producer-side
# implementation universe.  The independent v3 verifier transcribes it rather
# than importing this module, so coordinated implementation drift still fails.
IMPLEMENTATION_IDENTITIES = (
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
# Archive codecs and qualification harnesses are intentionally absent from this
# list. They validate historical evidence or exercise pre-seal substrate paths
# without changing the runtime authority sealed at I.
IMPLEMENTATION_PATHS = tuple(
    ROOT / path for path, _identity in IMPLEMENTATION_IDENTITIES
)

# Every input whose bytes determine the seal decision. Coverage/mutations are
# computed from these head blobs directly (never the working tree), and after
# assembly the working tree must still be byte-identical to them — otherwise
# an editor/formatter racing the run could bind `complete=true` to content
# that is not the named head's (the TOCTOU gap from PR #171 review).
CHECKER_INPUT_PATHS = (
    HEADER_PATH,
    CUDA_PATH,
    PYTHON_BINDING_PATH,
    TRACKER_WRAPPER_PATH,
    EXPORT_PATH,
    VERIFIER_PATH,
    CHECKER_PATH,
    SCHEMA_PATH,
)
SEAL_RELEVANT_PATHS = tuple(
    dict.fromkeys(CHECKER_INPUT_PATHS + (PRESET_PATH,) + IMPLEMENTATION_PATHS)
)


def classify_path(path: str) -> str:
    """Amendment 6 ``h0_projection_path_class_v1`` — fail-closed.

    Anything not explicitly non-runtime is runtime/build-consumable.  The sole
    root-file exception is ``DEVELOPMENT.md`` (Amendment 6 Correction 1):
    documentation only, never a build or runtime-import input.
    """
    if path == "DEVELOPMENT.md":
        return "non_runtime_recorded"
    if path.startswith(RUNTIME_PREFIXES) or path in ROOT_BUILD_MANIFESTS:
        return "runtime_build_consumable"
    if path.startswith(NON_RUNTIME_PREFIXES):
        return "non_runtime_recorded"
    return "runtime_build_consumable"


def admission_verdict(
    classified: dict[str, str], after_sha256: dict[str, str | None]
) -> tuple[bool, list[str]]:
    """Projection admitted iff every runtime path is admitted.

    A runtime path is admitted either as a frozen §A6.2 member or through an
    Amendment 6 Correction 1 content pin: its after-blob SHA-256 at the
    instrumentation head must equal the pinned value exactly.
    """
    violations = sorted(
        path
        for path, cls in classified.items()
        if cls == "runtime_build_consumable"
        and path not in ADMITTED_RUNTIME_PATHS
        and (
            path not in ADMITTED_RUNTIME_BLOBS
            or ADMITTED_RUNTIME_BLOBS[path] != after_sha256.get(path)
        )
    )
    return (not violations, violations)


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def _git(*args: str) -> str:
    return _git_bytes(*args).decode("utf-8").strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob_slot(rev: str, path: str) -> dict[str, Any]:
    """Tagged content slot: {state: present, sha256} or {state: absent, sha256: null}."""
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{rev}:{path}"], cwd=ROOT, capture_output=True
    )
    if probe.returncode != 0:
        return {"state": "absent", "sha256": None}
    return {
        "state": "present",
        "sha256": _sha256_bytes(_git_bytes("show", f"{rev}:{path}")),
    }


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _head_blob(head: str, path: Path) -> bytes:
    """Read a seal-relevant input from the head tree, never the working tree."""
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "show", f"{head}:{_rel(path)}"],
        cwd=ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"pre-seal engineering status: seal-relevant input {_rel(path)} "
            f"is not in the tree of {head}; commit it before assembling"
        )
    return result.stdout


def freeze_artifact_path(head: str) -> str:
    """A7.RC2.2's sole producer destination, with no operator choice."""
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise SystemExit(f"invalid instrumentation head for v3 path: {head!r}")
    return (
        "docs/modules/semantic/research/evidence/"
        f"h0_preseal_freeze_{head}/h0_preseal_freeze_v3.json"
    )


def _tree_entry(head: str, path: str) -> tuple[str, str, str]:
    raw = _git_bytes("ls-tree", "-z", head, "--", path)
    entries = [entry for entry in raw.split(b"\0") if entry]
    if len(entries) != 1:
        raise SystemExit(f"missing or ambiguous implementation path at {head}: {path}")
    try:
        metadata, found_path = entries[0].split(b"\t", 1)
        mode, git_type, object_id = metadata.decode("ascii").split(" ")
        if found_path.decode("utf-8", errors="strict") != path:
            raise ValueError
    except (UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(
            f"malformed Git tree entry for implementation path {path}"
        ) from exc
    return mode, git_type, object_id


def implementation_bindings(head: str) -> list[dict[str, Any]]:
    """Bind every A7/RC1 implementation blob from I, never the worktree."""
    records: list[dict[str, Any]] = []
    for path, identity in IMPLEMENTATION_IDENTITIES:
        mode, git_type, object_id = _tree_entry(head, path)
        blob = _git_bytes("show", f"{head}:{path}")
        if mode not in {"100644", "100755"} or git_type != "blob":
            raise SystemExit(
                f"implementation path is not a regular Git blob: {path} {mode} {git_type}"
            )
        records.append(
            {
                "path": path,
                "identity": identity,
                "mode": mode,
                "git_type": git_type,
                "git_object": object_id,
                "length": len(blob),
                "sha256": _sha256_bytes(blob),
            }
        )
    return records


def _physical_executable(command: str) -> Path:
    found = shutil.which(command)
    if not found:
        raise RuntimeError(f"required host executable is absent: {command}")
    candidate = Path(found).resolve(strict=True)
    details = candidate.stat(follow_symlinks=False)
    if candidate.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise RuntimeError(f"host executable is not a physical regular file: {command}")
    return candidate


def _derive_controller_input(head: str) -> dict[str, Any]:
    """Construct the host-specific RC1 input inventory without a CLI override.

    This is intentionally pre-seal only.  Any unavailable GPU, model, tool, or
    runtime library makes the prospective artifact incomplete instead of asking
    an operator for a substitute path.
    """
    import run_h0_phase_a as controller  # local implementation, bound above

    root = ROOT.resolve(strict=True)
    if root != ROOT or root.is_symlink():
        raise RuntimeError("repository root is not a physical canonical directory")
    git = _physical_executable("git")
    tool_paths = {
        name: _physical_executable(name).as_posix()
        for name in ("git", "ldd", "readelf", "uv")
    }
    python = root / ".venv/bin/python"
    if not python.is_file() or python.is_symlink():
        raise RuntimeError("frozen .venv/bin/python is absent or symlinked")
    pyvenv_config = root / ".venv/pyvenv.cfg"
    if not pyvenv_config.is_file() or pyvenv_config.is_symlink():
        raise RuntimeError("frozen .venv/pyvenv.cfg is absent or symlinked")
    # The locations are read from the frozen interpreter itself; no environment
    # variable or user argument can select an alternative runtime.
    query = subprocess.run(
        [
            python.as_posix(),
            "-I",
            "-c",
            "import pathlib,sysconfig,torch; print((pathlib.Path(torch.__file__).resolve().parent/'lib').as_posix()); import tensorrt_libs; print(pathlib.Path(tensorrt_libs.__file__).resolve().parent.as_posix()); print(pathlib.Path(sysconfig.get_path('purelib')).resolve().as_posix())",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(query) != 3:
        raise RuntimeError(
            "frozen Python did not derive the three runtime library directories"
        )
    # The CUDA build toolchain and runtime tree are uv-locked venv content
    # (issue #214); the rolling system toolkit must never be the authority.
    cuda_root = Path(query[2]).resolve(strict=True) / "nvidia/cu13"
    nvcc = cuda_root / "bin/nvcc"
    if nvcc.is_symlink() or not nvcc.is_file():
        raise RuntimeError("frozen venv CUDA compiler is absent or non-physical")
    tool_paths["nvcc"] = nvcc.as_posix()
    libraries = {
        "tensorrt_library_dir": Path(query[1]).resolve(strict=True).as_posix(),
        "pytorch_library_dir": Path(query[0]).resolve(strict=True).as_posix(),
        "cuda_library_dir": (cuda_root / "lib").resolve(strict=True).as_posix(),
    }
    if any(
        not Path(value).is_dir() or Path(value).is_symlink()
        for value in libraries.values()
    ):
        raise RuntimeError(
            "a derived runtime library directory is non-physical or absent"
        )
    gpu = controller.select_gpu_record(controller.nvml_gpu_inventory())
    repository = controller.repository_inventory(
        root, head, git, started=time.monotonic()
    )
    models = sorted(
        (
            controller.external_input_record(root, path)
            for path in controller.MODEL_LOGICAL_PATHS
        ),
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    # Tool/runtime inventory starts from every executable and library directory
    # named by RC1.  The controller's actual-loaded attestation remains the
    # final admission for dependencies discovered while the frozen process runs.
    tool_candidates = [Path(path) for path in tool_paths.values()] + [
        python,
        pyvenv_config,
    ]
    for directory in libraries.values():
        tool_candidates.extend(
            sorted(
                path
                for path in Path(directory).rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        )
    tools = sorted(
        (
            controller.external_input_record(root, path.as_posix())
            for path in tool_candidates
        ),
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    if len({record["realpath"] for record in tools}) != len(tools):
        raise RuntimeError("tool/runtime inventory has duplicate physical files")
    bound_inputs: dict[str, Any] = {
        "schema": controller.BOUND_INPUTS_SCHEMA,
        "digest": "",
        "repository": repository,
        "models_engines": models,
        "sequence": controller.sequence_input_inventory(root / controller.SEQUENCE_REL),
        "tool_runtime": tools,
    }
    bound_inputs["digest"] = controller.bound_inventory_digest(bound_inputs)
    controller.validate_bound_inventory(bound_inputs)
    evidence_root = f"docs/modules/semantic/research/evidence/h0_phase_a_{head}"
    landing = {
        "schema": LANDING_SCHEMA_VERSION,
        "artifact_path": freeze_artifact_path(head),
        "declaration_path": DECLARATION_PATH,
        "post_head_allowed_paths": [freeze_artifact_path(head), DECLARATION_PATH],
    }
    return {
        "authority_landing": landing,
        "bound_inputs": bound_inputs,
        "document_type": "controller_input",
        "evidence_root": evidence_root,
        "execution_constants": controller.execution_constants(root),
        "gpu": gpu,
        "incomplete_root": evidence_root + ".incomplete",
        "instrumentation_head": head,
        "library_dirs": libraries,
        "repository_root": root.as_posix(),
        "schema": controller.CONTROLLER_SCHEMA,
        "sequence_input_digest": bound_inputs["sequence"]["digest"],
        "tool_paths": tool_paths,
    }


DIFF_FLAGS = ("--no-color", "--binary", "--full-index", "--no-renames")


def _mutation_cases(cuda: str) -> list[tuple[str, str, str]]:
    """The five A3.2/A4 mutation admission cases, anchored to unique source."""
    live_claim_append = (
        "        const int claim_record_index = h0_append_record(\n"
        "            h0.claim_records, h0.claim_capacity, h0.claim_cursor, "
        "h0.claim_overflow, h0_claim);\n"
    )
    native_proposal_append = (
        "        h0_append_record(h0.native_proposal_keys, h0.native_proposal_capacity,\n"
        "                         h0.native_proposal_cursor, h0.native_proposal_overflow, proposal);\n"
    )
    native_key_field = "        proposal.frame = h0_frame;\n"
    for anchor, name in (
        (live_claim_append, "claim append"),
        (native_proposal_append, "native proposal append"),
        (native_key_field, "native key field"),
    ):
        if cuda.count(anchor) != 1:
            raise SystemExit(f"mutation anchor not unique in tracker_gpu.cu: {name}")
    return [
        (
            "A3.2-delete-claim-append",
            "claim_record",
            cuda.replace(
                live_claim_append, "        const int claim_record_index = -1;\n"
            ),
        ),
        (
            "A3.2-record-cursor-for-native-cursor",
            "native_universe_v2",
            cuda.replace(
                native_proposal_append,
                native_proposal_append.replace(
                    "h0.native_proposal_cursor", "h0.claim_cursor"
                ),
            ),
        ),
        (
            "A3.2-native-key-field-after-append",
            "native_universe_v2",
            cuda.replace(
                native_key_field + "        proposal.proposing_cand_slot = cand;\n",
                "        proposal.proposing_cand_slot = cand;\n",
            ).replace(
                native_proposal_append, native_proposal_append + native_key_field
            ),
        ),
        (
            "A4-claim-append-only-in-comment",
            "claim_record",
            cuda.replace(
                live_claim_append,
                "        // const int claim_record_index = h0_append_record(\n"
                "        //     h0.claim_records, h0.claim_capacity, h0.claim_cursor, "
                "h0.claim_overflow, h0_claim);\n"
                "        const int claim_record_index = -1;\n",
            ),
        ),
        (
            "A4-native-key-assign-commented-before-live-after",
            "native_universe_v2",
            cuda.replace(
                native_key_field, "        // proposal.frame = h0_frame;\n"
            ).replace(
                native_proposal_append, native_proposal_append + native_key_field
            ),
        ),
    ]


def mutation_admission(
    overrides: dict[Path, str],
) -> tuple[list[dict[str, Any]], bool]:
    cuda = overrides[CUDA_PATH]
    results: list[dict[str, Any]] = []
    all_pass = True
    for name, component, mutated in _mutation_cases(cuda):
        if mutated == cuda:
            raise SystemExit(f"mutation case is a no-op: {name}")
        report, _ = coverage_report(source_overrides={**overrides, CUDA_PATH: mutated})
        flipped = report["coverage_components"][component] is False
        all_pass = all_pass and flipped
        results.append(
            {
                "case": name,
                "expected_false_component": component,
                "component_false": flipped,
            }
        )
    return results, all_pass


def collect_static_evidence(
    command_line: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Head-static sealability evidence: no research input, GPU, model, engine,
    sequence-inventory, or runtime-library enumeration is performed here.

    Everything that requires the execution substrate lives in
    ``_derive_controller_input`` and is reached only through
    ``build_artifact`` at actual seal assembly (Amendment 6 Correction 1).
    """
    problems: list[str] = []

    if _git("status", "--porcelain"):
        problems.append(
            "working tree is not clean; the artifact must be produced at an exact head"
        )

    # The head is resolved once and pinned: every git read below names it
    # explicitly, so a concurrent commit cannot split the evidence across two
    # trees. Seal-relevant inputs are read from this head's blobs, never from
    # the working tree.
    head = _git("rev-parse", "HEAD")
    tree_id = _git("rev-parse", f"{head}^{{tree}}")
    tree_list = _git_bytes("ls-tree", "-r", "--full-tree", head)
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", POLICY_BASE_HEAD, head], cwd=ROOT
        ).returncode
        != 0
    ):
        problems.append("instrumentation head does not descend from policy_base_head")
    if _git("rev-parse", f"{POLICY_BASE_HEAD}^{{tree}}") != POLICY_BASE_TREE:
        problems.append("policy_base_tree mismatch against §2")

    seal_inputs = {path: _head_blob(head, path) for path in SEAL_RELEVANT_PATHS}
    bindings = implementation_bindings(head)

    diff_range = f"{POLICY_BASE_HEAD}..{head}"
    full_diff = _git_bytes("diff", *DIFF_FLAGS, diff_range)
    projection_pathspec = ["--", "."] + [f":(exclude){p}" for p in GOVERNANCE_ALLOWLIST]
    projection_diff = _git_bytes("diff", *DIFF_FLAGS, diff_range, *projection_pathspec)
    projection_paths = _git(
        "diff", "--name-only", "--no-renames", diff_range, *projection_pathspec
    ).splitlines()

    classified = {path: classify_path(path) for path in projection_paths}
    path_records = [
        {
            "path": path,
            "class": classified[path],
            "before": _blob_slot(POLICY_BASE_HEAD, path),
            "after": _blob_slot(head, path),
        }
        for path in sorted(projection_paths)
    ]
    admitted, violations = admission_verdict(
        classified,
        {record["path"]: record["after"]["sha256"] for record in path_records},
    )
    if not admitted:
        problems.append(
            "projection not admitted; runtime paths outside h0_admitted_runtime_paths_v1: "
            + ", ".join(violations)
        )

    excluded_records = [
        {
            "path": path,
            "before": _blob_slot(POLICY_BASE_HEAD, path),
            "after": _blob_slot(head, path),
        }
        for path in GOVERNANCE_ALLOWLIST
    ]

    rename_record = {
        "schema": "governance_rename_v1",
        "source_path": GOVERNANCE_RENAME_SOURCE,
        "destination_path": GOVERNANCE_RENAME_DESTINATION,
        "source_blob_before": _blob_slot(POLICY_BASE_HEAD, GOVERNANCE_RENAME_SOURCE),
        "source_blob_after": _blob_slot(head, GOVERNANCE_RENAME_SOURCE),
        "destination_blob_before": _blob_slot(
            POLICY_BASE_HEAD, GOVERNANCE_RENAME_DESTINATION
        ),
        "destination_blob_after": _blob_slot(head, GOVERNANCE_RENAME_DESTINATION),
    }

    checker_overrides = {
        path: seal_inputs[path].decode("utf-8") for path in CHECKER_INPUT_PATHS
    }
    coverage, coverage_failures = coverage_report(source_overrides=checker_overrides)
    if not coverage["all_components_true"]:
        problems.append(
            "h0_coverage_v2 is not all-true: " + "; ".join(coverage_failures)
        )

    mutation_results, mutations_pass = mutation_admission(checker_overrides)
    if not mutations_pass:
        problems.append("A3.2/A4 mutation admission did not flip every named component")

    preset_bytes = seal_inputs[PRESET_PATH]
    preset_sha = _sha256_bytes(preset_bytes)
    if preset_sha != POLICY_TARGET_PRESET_SHA256:
        problems.append(f"policy target preset hash drifted: {preset_sha}")
    resolved = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "resolved_bridge_policy_config.py"),
            "--preset",
            "mamba_whole_graph_m",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    resolved_sha = _sha256_bytes(resolved.encode("utf-8"))
    if resolved_sha != POLICY_TARGET_RESOLVED_SHA256:
        problems.append(
            f"resolved_bridge_policy_config_v1 fingerprint drifted: {resolved_sha}"
        )

    # TOCTOU guard (PR #171 review): nothing may have moved between the pinned
    # head resolution above and the `complete` verdict below. The evidence was
    # computed from head blobs; the working tree must still agree with them,
    # HEAD must not have advanced, and the tree must still be clean — the
    # resolver subprocess above reads the working tree, so byte-agreement is
    # what ties its output back to the named head.
    head_after = _git("rev-parse", "HEAD")
    if head_after != head:
        problems.append(
            f"HEAD moved during assembly: artifact names {head} but HEAD is now {head_after}"
        )
    if _git("status", "--porcelain"):
        problems.append("working tree or index is not clean after assembly")
    drifted = sorted(
        _rel(path)
        for path, blob in seal_inputs.items()
        if not path.is_file() or path.read_bytes() != blob
    )
    if drifted:
        problems.append(
            "working-tree bytes differ from the head blobs for seal-relevant "
            "inputs: " + ", ".join(drifted)
        )

    evidence = {
        "head": head,
        "tree_id": tree_id,
        "tree_list": tree_list,
        "seal_inputs": seal_inputs,
        "bindings": bindings,
        "full_diff": full_diff,
        "projection_diff": projection_diff,
        "path_records": path_records,
        "excluded_records": excluded_records,
        "rename_record": rename_record,
        "admitted": admitted,
        "coverage": coverage,
        "mutation_results": mutation_results,
        "mutations_pass": mutations_pass,
        "preset_sha": preset_sha,
        "resolved_sha": resolved_sha,
    }
    return evidence, problems


def check_preseal_sealability(command_line: list[str]) -> dict[str, Any]:
    """The static sealability gate for non-authoritative qualification.

    Reads only git history, head blobs, and the working tree of the checked-out
    head; never derives the A7/RC1 controller input, so no research sequence,
    model, engine, GPU, or runtime-library inventory is touched and no file is
    written.
    """
    evidence, problems = collect_static_evidence(command_line)
    return {
        "schema": "h0_preseal_static_sealability_v1",
        "instrumentation_head": evidence["head"],
        "instrumentation_tree": evidence["tree_id"],
        "projection_admitted": evidence["admitted"],
        "sealable": not problems,
        "problems": problems,
    }


def build_artifact(
    command_line: list[str], *, controller_input: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    evidence, problems = collect_static_evidence(command_line)
    head = evidence["head"]
    tree_id = evidence["tree_id"]
    tree_list = evidence["tree_list"]
    seal_inputs = evidence["seal_inputs"]
    bindings = evidence["bindings"]
    full_diff = evidence["full_diff"]
    projection_diff = evidence["projection_diff"]
    path_records = evidence["path_records"]
    excluded_records = evidence["excluded_records"]
    rename_record = evidence["rename_record"]
    admitted = evidence["admitted"]
    coverage = evidence["coverage"]
    mutation_results = evidence["mutation_results"]
    mutations_pass = evidence["mutations_pass"]
    preset_sha = evidence["preset_sha"]
    resolved_sha = evidence["resolved_sha"]

    if controller_input is None:
        try:
            controller_input = _derive_controller_input(head)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            problems.append(f"A7/RC1 controller-input inventory is unavailable: {exc}")
            controller_input = {
                "authority_landing": {
                    "schema": LANDING_SCHEMA_VERSION,
                    "artifact_path": freeze_artifact_path(head),
                    "declaration_path": DECLARATION_PATH,
                    "post_head_allowed_paths": [
                        freeze_artifact_path(head),
                        DECLARATION_PATH,
                    ],
                },
                "unavailable": True,
            }
    if controller_input.get("instrumentation_head") != head:
        problems.append(
            "controller input does not bind the resolved instrumentation head"
        )

    artifact = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "instrumentation_head": head,
        "instrumentation_tree": tree_id,
        "tree_list_sha256": _sha256_bytes(tree_list),
        "policy_base_head": POLICY_BASE_HEAD,
        "policy_base_tree": POLICY_BASE_TREE,
        "full_repo_diff_sha256": _sha256_bytes(full_diff),
        "runtime_policy_code_projection_v1": {
            "classifier": CLASSIFIER_VERSION,
            "projection_diff_sha256": _sha256_bytes(projection_diff),
            "paths": path_records,
            "excluded_governance_allowlist": excluded_records,
            "governance_rename_v1": rename_record,
            "projection_admitted": admitted,
        },
        "h0_coverage_v2": coverage,
        "mutation_admission": {
            "cases": mutation_results,
            "all_pass": mutations_pass,
        },
        "policy_target": {
            "preset": POLICY_TARGET_PRESET,
            "preset_sha256": preset_sha,
            "resolved_schema": "resolved_bridge_policy_config_v1",
            "resolved_fingerprint": resolved_sha,
        },
        "assembler_sha256": _sha256_bytes(seal_inputs[ASSEMBLER_PATH]),
        "input_binding": {
            "mode": "head_blobs_with_post_assembly_reverify",
            "seal_relevant_paths": sorted(_rel(path) for path in SEAL_RELEVANT_PATHS),
        },
        "produced_by": {
            "command_line": command_line,
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git("--version"),
        },
        "authority_landing": {
            "schema": LANDING_SCHEMA_VERSION,
            "artifact_path": freeze_artifact_path(head),
            "declaration_path": DECLARATION_PATH,
            "post_head_allowed_paths": [freeze_artifact_path(head), DECLARATION_PATH],
        },
        "implementation_bindings": bindings,
        "phase_a_controller_input": controller_input,
        "complete": not problems,
        "problems": problems,
    }
    return artifact, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    artifact, problems = build_artifact(sys.argv)
    payload = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    output = ROOT / freeze_artifact_path(artifact["instrumentation_head"])
    if output.exists() or output.is_symlink():
        raise SystemExit(
            f"refusing to replace deterministic v3 artifact path: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short v3 artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"instrumentation_head {artifact['instrumentation_head']}")
    print(
        f"projection_admitted  {artifact['runtime_policy_code_projection_v1']['projection_admitted']}"
    )
    print(f"coverage all true    {artifact['h0_coverage_v2']['all_components_true']}")
    print(f"mutations all pass   {artifact['mutation_admission']['all_pass']}")
    print(f"complete             {artifact['complete']}")
    if problems:
        for problem in problems:
            print(f"pre-seal engineering status: {problem}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
