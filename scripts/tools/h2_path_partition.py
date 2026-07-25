#!/usr/bin/env python3
"""H2 firewall: classify paths and decide Layer-P retry admissibility.

The partition distinguishes code, the authority that defines identity semantics,
the two fixtures, runtime assets, retryable plumbing, and genuine non-execution
material. A finite behavior probe is only a change detector; it is never the
fallback for a missing path classification and never proves global equivalence.

    decision_relevant       executable source/config that can change decisions
    identity_semantics      probe/equivalence/certificate authority
    identity_fixture_input  the selected deterministic probe fixture
    measurement_input       data and metadata consumed by Layer M
    runtime_asset           weights, engines, plugins, third-party components
    plumbing_only           retryable build/serialization infrastructure
    non_execution           prose, tests and outputs not consumed by a run
    unclassified            fail-closed until deliberately classified

Usage:
  uv run python scripts/tools/h2_path_partition.py --classify src/tracking/tracker_gpu.cu
  uv run python scripts/tools/h2_path_partition.py --check-retry <path> [<path> ...]
  uv run python scripts/tools/h2_path_partition.py --check-retry --from-git <base>
"""
# status: stable

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Literal, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]

PathClass = Literal[
    "decision_relevant",
    "identity_semantics",
    "identity_fixture_input",
    "measurement_input",
    "runtime_asset",
    "plumbing_only",
    "non_execution",
    "unclassified",
]

# The five H0 A6.2 paths remain a minimum, but not the complete implementation
# coordinate. A branch outside MOT17-09's execution path must still move that
# coordinate even when the probe cannot observe it.
_ADMITTED_RUNTIME_PATHS = (
    "include/tracking/tracker_gpu.hpp",
    "src/tracking/tracker_gpu.cu",
    "src/tracking/tracker_gpu_python.cpp",
    "src/saccade/perception/tracking/tracker_gpu.py",
    "src/saccade/perception/eval/stages.py",
)
_OBSERVATION_SITES = (
    "src/saccade/perception/eval/pipeline.py",
    "src/saccade/perception/eval/evaluator.py",
)
_POLICY_SURFACE = (
    "configs/presets/mamba_whole_graph_m.yaml",
    "scripts/tools/resolved_bridge_policy_config.py",
    "scripts/eval/mot17.py",
    "scripts/eval/mot17_args.py",
)
DECISION_RELEVANT_PATHS: frozenset[str] = frozenset(
    _ADMITTED_RUNTIME_PATHS + _OBSERVATION_SITES + _POLICY_SURFACE
)
DECISION_RELEVANT_PREFIXES: tuple[str, ...] = (
    "include/",
    "scripts/eval/config/",
    "src/",
)

# The ruler has an identity of its own. Exact rules win over broad plumbing and
# docs prefixes below.
IDENTITY_SEMANTICS_PATHS: frozenset[str] = frozenset(
    (
        ".github/workflows/runtime_identity.yml",
        "docs/modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.policy.yaml",
        "scripts/pre_push.sh",
        "scripts/tools/build_runtime_identity.py",
        "scripts/tools/check_runtime_identity_staleness.py",
        "scripts/tools/h2_behavioral_identity.py",
        "scripts/tools/h2_path_partition.py",
        "scripts/tools/h2_runtime_inputs.py",
        "scripts/tools/h2_terminal_partition.py",
        "scripts/tools/run_h2_layer_p.py",
    )
)

IDENTITY_FIXTURE_PREFIXES: tuple[str, ...] = ("datasets/MOT17/train/MOT17-09-SDP/",)
MEASUREMENT_INPUT_PREFIXES: tuple[str, ...] = ("datasets/",)
RUNTIME_ASSET_PREFIXES: tuple[str, ...] = (
    "build/",
    "models/",
    "runs/",
    "third_party/",
)

PLUMBING_ONLY_PATHS: frozenset[str] = frozenset(
    (
        "CMakeLists.txt",
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "docker-compose.yml",
    )
)
PLUMBING_ONLY_PREFIXES: tuple[str, ...] = (
    "scripts/tools/h0_",
    "scripts/tools/h2_",
    "scripts/tools/run_h0_",
    "scripts/tools/run_h2_",
    "scripts/tools/verify_h0_",
    "scripts/tools/verify_h2_",
    "scripts/tools/qualify_h0_",
    "scripts/tools/check_h0_",
    "scripts/tools/check_h2_",
    "scripts/tools/build_h0_",
    "scripts/tools/build_runtime_identity",
    "scripts/tools/export_headline_bridge_decision_trace",
    "scripts/tools/verify_headline_bridge_decision_trace",
    ".github/",
    "infra/",
    "cmake/",
)

NON_EXECUTION_PREFIXES: tuple[str, ...] = (
    "docs/",
    "tests/",
    "reports/",
    "report_data/",
    "results/",
    "out/",
    "output/",
    "logs/",
    "scratch/",
    "storage/",
    "graphify-out/",
    "Testing/",
)
NON_EXECUTION_PATHS: frozenset[str] = frozenset(
    (
        "README.md",
        "DEVELOPMENT.md",
        "REPO_LAYOUT.md",
        "LICENSE",
        "uv.lock.license",
        ".gitignore",
    )
)


class RetryVerdict(NamedTuple):
    admissible: bool
    decision_relevant: tuple[str, ...]
    identity_semantics: tuple[str, ...]
    identity_fixture_input: tuple[str, ...]
    measurement_input: tuple[str, ...]
    runtime_asset: tuple[str, ...]
    unclassified: tuple[str, ...]
    plumbing_only: tuple[str, ...]
    non_execution: tuple[str, ...]

    def reason(self) -> str:
        if self.admissible:
            return "retry admissible: only plumbing and non-execution paths changed"
        parts: list[str] = []
        if self.decision_relevant:
            parts.append(
                "decision-relevant paths may not change in a Layer-P retry: "
                + ", ".join(self.decision_relevant)
            )
        if self.identity_semantics:
            parts.append(
                "identity-semantics paths may not change in a Layer-P retry — "
                "the probe/equivalence guard cannot edit itself: "
                + ", ".join(self.identity_semantics)
            )
        for label, paths in (
            ("identity-fixture inputs", self.identity_fixture_input),
            ("measurement inputs", self.measurement_input),
            ("runtime assets", self.runtime_asset),
        ):
            if paths:
                parts.append(
                    f"{label} may not change in a Layer-P retry: " + ", ".join(paths)
                )
        if self.unclassified:
            parts.append(
                "unclassified paths (fail-closed; classify them in "
                "h2_path_partition.py first): " + ", ".join(self.unclassified)
            )
        return "; ".join(parts)


def _normalize(path: str | Path) -> str:
    text = path.as_posix() if isinstance(path, Path) else str(path).replace("\\", "/")
    text = text.removeprefix("./")
    if text.startswith("/"):
        try:
            text = Path(text).resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return text
    return text


def classify(path: str | Path) -> PathClass:
    """Classify one repo-relative path. Deterministic and total."""
    rel = _normalize(path)
    if rel in DECISION_RELEVANT_PATHS or rel.startswith(DECISION_RELEVANT_PREFIXES):
        return "decision_relevant"
    if rel in IDENTITY_SEMANTICS_PATHS:
        return "identity_semantics"
    if rel.startswith(IDENTITY_FIXTURE_PREFIXES):
        return "identity_fixture_input"
    if rel.startswith(MEASUREMENT_INPUT_PREFIXES):
        return "measurement_input"
    if rel.startswith(RUNTIME_ASSET_PREFIXES):
        return "runtime_asset"
    if rel in PLUMBING_ONLY_PATHS or rel.startswith(PLUMBING_ONLY_PREFIXES):
        return "plumbing_only"
    # Prose under a protected prefix stays protected because this rule comes last.
    if rel.endswith((".md", ".rst", ".txt")):
        return "non_execution"
    if rel in NON_EXECUTION_PATHS or rel.startswith(NON_EXECUTION_PREFIXES):
        return "non_execution"
    return "unclassified"


def check_retry(paths: Iterable[str | Path]) -> RetryVerdict:
    """Allow a retry only when every change is plumbing or non-execution."""
    buckets: dict[PathClass, list[str]] = {
        "decision_relevant": [],
        "identity_semantics": [],
        "identity_fixture_input": [],
        "measurement_input": [],
        "runtime_asset": [],
        "plumbing_only": [],
        "non_execution": [],
        "unclassified": [],
    }
    for path in paths:
        rel = _normalize(path)
        buckets[classify(rel)].append(rel)
    blocking = (
        buckets["decision_relevant"]
        or buckets["identity_semantics"]
        or buckets["identity_fixture_input"]
        or buckets["measurement_input"]
        or buckets["runtime_asset"]
        or buckets["unclassified"]
    )
    return RetryVerdict(
        admissible=not blocking,
        decision_relevant=tuple(sorted(buckets["decision_relevant"])),
        identity_semantics=tuple(sorted(buckets["identity_semantics"])),
        identity_fixture_input=tuple(sorted(buckets["identity_fixture_input"])),
        measurement_input=tuple(sorted(buckets["measurement_input"])),
        runtime_asset=tuple(sorted(buckets["runtime_asset"])),
        unclassified=tuple(sorted(buckets["unclassified"])),
        plumbing_only=tuple(sorted(buckets["plumbing_only"])),
        non_execution=tuple(sorted(buckets["non_execution"])),
    )


def changed_paths_since(base: str) -> tuple[str, ...]:
    """Repo-relative tracked/untracked paths changed since *base*."""
    completed = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in completed.stdout.splitlines() if line]
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return tuple(
        sorted({*tracked, *(line for line in untracked.stdout.splitlines() if line)})
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify", metavar="PATH", help="print one path's class")
    parser.add_argument(
        "--check-retry", action="store_true", help="check Layer-P retry admissibility"
    )
    parser.add_argument(
        "--from-git", metavar="BASE", help="take changed paths from git diff vs BASE"
    )
    parser.add_argument("paths", nargs="*", help="paths for --check-retry")
    args = parser.parse_args(argv)

    if args.classify:
        print(classify(args.classify))
        return 0
    if not args.check_retry:
        parser.error("one of --classify or --check-retry is required")

    paths = tuple(args.paths)
    if args.from_git:
        paths += changed_paths_since(args.from_git)
    if not paths:
        parser.error("--check-retry needs paths or --from-git")

    verdict = check_retry(paths)
    print(verdict.reason())
    for name in (
        "plumbing_only",
        "non_execution",
        "decision_relevant",
        "identity_semantics",
        "identity_fixture_input",
        "measurement_input",
        "runtime_asset",
        "unclassified",
    ):
        for item in getattr(verdict, name):
            print(f"  {name:24} {item}")
    return 0 if verdict.admissible else 1


if __name__ == "__main__":
    sys.exit(main())
