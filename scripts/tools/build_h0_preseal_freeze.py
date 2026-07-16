#!/usr/bin/env python3
"""Assemble the H0 pre-seal freeze artifact (``h0_preseal_freeze_v2``).

Declaration authority: H0 declaration §2 / A2.2 / Amendment 6. This tool is a
non-production offline input under the A2.1 allowance: it reads git history,
H0 sources, and the trace schema; it is never a build input or runtime-policy
path. It runs no capture and can emit no H0 execution terminal — an incomplete
result is the pre-seal engineering status that prohibits seal.

The emitted artifact names the exact ``instrumentation_head`` it was produced
at; that head is the only head an owner may seal (A2.2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
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

FREEZE_SCHEMA_VERSION = "h0_preseal_freeze_v2"
CAPTURE_SCHEMA_VERSION = "h0_bridge_decision_trace_v2"
CLASSIFIER_VERSION = "h0_projection_path_class_v1"

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

RUNTIME_PREFIXES = ("src/", "include/", "configs/", "cmake/")
ROOT_BUILD_MANIFESTS = frozenset(
    {"pyproject.toml", "uv.lock", "CMakeLists.txt", "setup.py", "setup.cfg", "Makefile"}
)
NON_RUNTIME_PREFIXES = ("docs/", ".github/", "tests/", "scripts/")

ASSEMBLER_PATH = Path(__file__).resolve()
PRESET_PATH = ROOT / POLICY_TARGET_PRESET

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
SEAL_RELEVANT_PATHS = CHECKER_INPUT_PATHS + (PRESET_PATH, ASSEMBLER_PATH)


def classify_path(path: str) -> str:
    """Amendment 6 ``h0_projection_path_class_v1`` — fail-closed.

    Anything not explicitly non-runtime is runtime/build-consumable.
    """
    if path.startswith(RUNTIME_PREFIXES) or path in ROOT_BUILD_MANIFESTS:
        return "runtime_build_consumable"
    if path.startswith(NON_RUNTIME_PREFIXES):
        return "non_runtime_recorded"
    return "runtime_build_consumable"


def admission_verdict(classified: dict[str, str]) -> tuple[bool, list[str]]:
    """Projection admitted iff every runtime path is in the admitted set."""
    violations = sorted(
        path
        for path, cls in classified.items()
        if cls == "runtime_build_consumable" and path not in ADMITTED_RUNTIME_PATHS
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


def build_artifact(command_line: list[str]) -> tuple[dict[str, Any], list[str]]:
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

    diff_range = f"{POLICY_BASE_HEAD}..{head}"
    full_diff = _git_bytes("diff", *DIFF_FLAGS, diff_range)
    projection_pathspec = ["--", "."] + [f":(exclude){p}" for p in GOVERNANCE_ALLOWLIST]
    projection_diff = _git_bytes("diff", *DIFF_FLAGS, diff_range, *projection_pathspec)
    projection_paths = _git(
        "diff", "--name-only", "--no-renames", diff_range, *projection_pathspec
    ).splitlines()

    classified = {path: classify_path(path) for path in projection_paths}
    admitted, violations = admission_verdict(classified)
    if not admitted:
        problems.append(
            "projection not admitted; runtime paths outside h0_admitted_runtime_paths_v1: "
            + ", ".join(violations)
        )

    path_records = [
        {
            "path": path,
            "class": classified[path],
            "before": _blob_slot(POLICY_BASE_HEAD, path),
            "after": _blob_slot(head, path),
        }
        for path in sorted(projection_paths)
    ]

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
        "complete": not problems,
        "problems": problems,
    }
    return artifact, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write canonical freeze JSON here")
    args = parser.parse_args()
    artifact, problems = build_artifact(sys.argv)
    payload = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
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
