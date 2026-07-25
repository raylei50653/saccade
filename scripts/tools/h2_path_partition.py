#!/usr/bin/env python3
"""H2 firewall: classify repository paths and decide Layer-P retry admissibility.

Layer P (build / load / confinement / attestation / evidence serialization) is
retryable pre-seal engineering, which is only sound if the retries provably
cannot touch what the measurement measures. This module is the single source of
that partition.

    decision_relevant    a change here can change bridge decisions
    invariant_authority  the guard itself: digest producer, classifier, publisher
    plumbing_only        build recipe, controller/verifier scaffolding, CI, infra
    non_execution        docs, tests, datasets, results — never executed by a run
    unclassified         fail-closed: not admissible in a retry until classified

`invariant_authority` exists because the guard must not be editable by the thing
it guards. The behavior digest is what makes a retryable Layer P sound; if a
retry could edit the digest producer or this classifier, the firewall would hold
only as long as nobody weakened it — a circular oracle with extra steps.

The `decision_relevant` core is **not invented here**: it is
`ADMITTED_RUNTIME_PATHS` (the five A6.2 trace paths of the H0 declaration), plus
the evaluator/pipeline emission sites the A7.6 inventory reads through and the
resolved-parameter surface of the sealed preset.

Why this classifier can be wrong without being dangerous — unlike H0's
`h0_projection_path_class_v1`, which blocked seals whenever an ordinary `main`
landing touched a `runtime_build_consumable` path (Amendment 6 Correction 1):
here a misclassification does not gate a seal, it gets **caught by the behavior
digest**. A `plumbing_only` edit that moves the behavior axis is by definition
misclassified, and `h2_behavioral_identity` refuses it. Classification decides
what a retry may attempt; the digest decides what is true.

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
    "invariant_authority",
    "plumbing_only",
    "non_execution",
    "unclassified",
]

# --------------------------------------------------------------------------- #
# decision_relevant — exact paths only. A prefix rule here would silently adopt
# future files into the protected set, and "protected" must be a deliberate act.
# --------------------------------------------------------------------------- #

# The five A6.2 admitted runtime paths, transcribed from
# verify_h0_preseal_freeze.ADMITTED_RUNTIME_PATHS (H0 declaration Amendment 6).
_ADMITTED_RUNTIME_PATHS = (
    "include/tracking/tracker_gpu.hpp",
    "src/tracking/tracker_gpu.cu",
    "src/tracking/tracker_gpu_python.cpp",
    "src/saccade/perception/tracking/tracker_gpu.py",
    "src/saccade/perception/eval/stages.py",
)

# Emission and injection sites the A7.6 inventory observes through: `pipeline.py`
# is the production `set_*` site (routing matrix C8) and constructs the tracker;
# `evaluator.py` owns `_run_frame` and `_fast_emit_mot_lines`, which produce
# `final_track_rows` and `active_tid_slot_pairs`.
_OBSERVATION_SITES = (
    "src/saccade/perception/eval/pipeline.py",
    "src/saccade/perception/eval/evaluator.py",
)

# The sealed policy target (declaration § 3.1 / Amendment 5) and the resolver
# that turns it into the `decision_surface` axis.
_POLICY_SURFACE = (
    "configs/presets/mamba_whole_graph_m.yaml",
    "scripts/tools/resolved_bridge_policy_config.py",
    "scripts/eval/mot17_args.py",
)

DECISION_RELEVANT_PATHS: frozenset[str] = frozenset(
    _ADMITTED_RUNTIME_PATHS + _OBSERVATION_SITES + _POLICY_SURFACE
)

# The whole resolved-parameter surface: any default here lands in the
# `decision_surface` digest.
DECISION_RELEVANT_PREFIXES: tuple[str, ...] = ("scripts/eval/config/",)

# --------------------------------------------------------------------------- #
# invariant_authority — the guard. Rejected in a retry for the same reason as
# decision_relevant, with a different explanation: not "you changed what we
# measure" but "you changed what decides whether it changed".
# --------------------------------------------------------------------------- #
INVARIANT_AUTHORITY_PATHS: frozenset[str] = frozenset(
    (
        "scripts/tools/h2_path_partition.py",
        "scripts/tools/h2_behavioral_identity.py",
        "scripts/tools/build_runtime_identity.py",
        "scripts/tools/check_runtime_identity_staleness.py",
    )
)

# --------------------------------------------------------------------------- #
# plumbing_only — retryable. Build recipe, H0/H2 controller and verifier
# scaffolding, CI, infrastructure.
# --------------------------------------------------------------------------- #
PLUMBING_ONLY_PATHS: frozenset[str] = frozenset(
    (
        "CMakeLists.txt",
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "docker-compose.yml",
        "scripts/pre_push.sh",
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

# --------------------------------------------------------------------------- #
# non_execution — present in the repository, never executed by a measurement.
# --------------------------------------------------------------------------- #
NON_EXECUTION_PREFIXES: tuple[str, ...] = (
    "docs/",
    "tests/",
    "datasets/",
    "reports/",
    "report_data/",
    "results/",
    "runs/",
    "out/",
    "output/",
    "logs/",
    "scratch/",
    "storage/",
    "third_party/",
    "graphify-out/",
    "models/",
    "build/",
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
    invariant_authority: tuple[str, ...]
    unclassified: tuple[str, ...]
    plumbing_only: tuple[str, ...]
    non_execution: tuple[str, ...]

    def reason(self) -> str:
        if self.admissible:
            return "retry admissible: only plumbing and non-execution paths changed"
        parts = []
        if self.decision_relevant:
            parts.append(
                "decision-relevant paths may not change in a Layer-P retry: "
                + ", ".join(self.decision_relevant)
            )
        if self.invariant_authority:
            parts.append(
                "invariant-authority paths may not change in a Layer-P retry — the "
                "guard cannot be edited by the retry it guards: "
                + ", ".join(self.invariant_authority)
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
        # Absolute paths are accepted only inside the repository.
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
    # Exact invariant-authority paths win over the `scripts/tools/h2_` prefix rule.
    if rel in INVARIANT_AUTHORITY_PATHS:
        return "invariant_authority"
    if rel in PLUMBING_ONLY_PATHS or rel.startswith(PLUMBING_ONLY_PREFIXES):
        return "plumbing_only"
    # Prose does not execute, wherever it lives. Checked after the decision and
    # invariant tables so a `.md` inside a protected prefix (e.g.
    # `scripts/eval/config/README.md`) stays protected rather than becoming freely
    # editable in a retry.
    if rel.endswith((".md", ".rst", ".txt")):
        return "non_execution"
    if rel in NON_EXECUTION_PATHS or rel.startswith(NON_EXECUTION_PREFIXES):
        return "non_execution"
    return "unclassified"


def check_retry(paths: Iterable[str | Path]) -> RetryVerdict:
    """A Layer-P retry may change plumbing and docs — nothing else.

    `decision_relevant` is rejected because the measurement's invariant is
    exactly that those files did not move. `unclassified` is rejected because a
    file nobody has classified cannot be asserted harmless: that assertion is
    what H0's enumerative admission kept getting wrong.
    """
    buckets: dict[PathClass, list[str]] = {
        "decision_relevant": [],
        "invariant_authority": [],
        "plumbing_only": [],
        "non_execution": [],
        "unclassified": [],
    }
    for path in paths:
        rel = _normalize(path)
        buckets[classify(rel)].append(rel)
    return RetryVerdict(
        admissible=not (
            buckets["decision_relevant"]
            or buckets["invariant_authority"]
            or buckets["unclassified"]
        ),
        decision_relevant=tuple(sorted(buckets["decision_relevant"])),
        invariant_authority=tuple(sorted(buckets["invariant_authority"])),
        unclassified=tuple(sorted(buckets["unclassified"])),
        plumbing_only=tuple(sorted(buckets["plumbing_only"])),
        non_execution=tuple(sorted(buckets["non_execution"])),
    )


def changed_paths_since(base: str) -> tuple[str, ...]:
    """Repo-relative paths changed since *base*, including the working tree."""
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
    for name, bucket in (
        ("plumbing_only", verdict.plumbing_only),
        ("non_execution", verdict.non_execution),
        ("decision_relevant", verdict.decision_relevant),
        ("invariant_authority", verdict.invariant_authority),
        ("unclassified", verdict.unclassified),
    ):
        for item in bucket:
            print(f"  {name:18} {item}")
    return 0 if verdict.admissible else 1


if __name__ == "__main__":
    sys.exit(main())
