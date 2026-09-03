#!/usr/bin/env python3
"""Fail-closed coverage checker for ADR 021 AP-2 artifact producers.

AP-2's contract is that an eval/train entry point which produces an artifact
directory claims it — manifest first — via ``run_manifest.open_run``.  PR #330
delivered that for a **hand-picked** list of entry points.  A hand-picked list
is exactly the thing that drifts: a new producer added later is covered by
nobody's decision, and its absence looks identical to a deliberate exclusion.

This checker removes the hand-picking by requiring every file in a declared
domain to carry an **explicit** classification in
``artifact_producer_registry.json``.  The registry is the authority; this module
only enforces consistency between it, the repository, and the path partition.
Nothing here infers a classification from a filename, a directory, an import, or
the current shape of the code — an unlisted file is a hard failure, never an
implicit exclusion.

The one classification that could otherwise become an escape hatch is
``run_producer_blocked``: "this produces artifacts but cannot be wired".  It is
admissible only when the path really does sit in a protected partition, which is
checked against ``h2_path_partition`` rather than taken on the registry's word.
A producer cannot be excused by asserting it is blocked.
"""

# status: stable

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_path_partition as partition  # noqa: E402

REGISTRY_REL = "scripts/provenance/artifact_producer_registry.json"
REGISTRY_SCHEMA = "artifact_producer_registry_v1"

# Partition classes in which an AP-2 hook cannot be added without a controlled
# re-attestation.  Editing such a file moves the published runtime coordinate,
# so the block is a property of the repository, not of anyone's judgement.
PROTECTED_PARTITION_CLASSES = frozenset({"decision_relevant", "identity_semantics"})

CLASSIFICATIONS = frozenset(
    {
        # Produces an artifact directory and claims it via open_run.
        "run_producer_wired",
        # Produces an artifact directory but lives in a protected path.
        "run_producer_blocked",
        # Produces artifact directories that ADR 021 §4.3 places outside W-A.
        "run_producer_out_of_scope",
        # Produces no artifact-root directory at all.
        "not_a_run_producer",
    }
)

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "run_producer_wired": ("classification", "reason"),
    "run_producer_blocked": (
        "classification",
        "reason",
        "blocked_by",
        "unblock_requires",
    ),
    "run_producer_out_of_scope": ("classification", "reason", "excluded_by"),
    "not_a_run_producer": ("classification", "reason"),
}

TOP_LEVEL_FIELDS = frozenset(
    {"schema", "authority", "asset_roots", "domain", "entries"}
)
DOMAIN_FIELDS = frozenset({"prefixes", "suffix"})


class CoverageError(RuntimeError):
    pass


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CoverageError(f"no producer registry at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CoverageError(f"{path}: invalid JSON: {exc}") from exc
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise CoverageError(f"{path}: not an {REGISTRY_SCHEMA} payload")
    unknown = set(payload) - TOP_LEVEL_FIELDS
    if unknown:
        raise CoverageError(f"{path}: unknown top-level field(s): {sorted(unknown)}")
    domain = payload.get("domain")
    if not isinstance(domain, dict) or set(domain) - DOMAIN_FIELDS:
        raise CoverageError(f"{path}: malformed domain")
    if not isinstance(payload.get("entries"), dict):
        raise CoverageError(f"{path}: missing entries")
    return payload


def domain_files(payload: dict[str, Any], repo_root: Path) -> set[str]:
    """Every tracked file the registry must account for.

    Tracked, not globbed: an untracked scratch file is not part of the
    repository's declared surface, and letting the filesystem widen the domain
    would make the check depend on whatever happens to be lying around.
    """
    prefixes = tuple(payload["domain"]["prefixes"])
    suffix = payload["domain"]["suffix"]
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *prefixes],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return {p for p in listed if p.endswith(suffix)}


def calls_open_run(source: str) -> bool:
    """True when the module contains an actual ``open_run(...)`` call.

    Parsed, not grepped: a docstring or a comment naming ``open_run`` must not
    satisfy the wiring requirement, or deleting the call while leaving the
    import behind would pass.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CoverageError(f"unparseable source: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open_run":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "open_run":
                return True
    return False


def check(repo_root: Path, payload: dict[str, Any] | None = None) -> list[str]:
    """Report every way the registry and the repository disagree.

    ``payload`` overrides the on-disk registry so a caller can evaluate a
    hypothetical one. Tests use it to mutate rules without writing to the
    working tree, which would otherwise leave the repository dirty on failure.
    """
    if payload is None:
        payload = load_registry(repo_root / REGISTRY_REL)
    entries: dict[str, Any] = payload["entries"]
    tracked = domain_files(payload, repo_root)
    failures: list[str] = []

    # An entry point nobody classified is the failure this checker exists for.
    for path in sorted(tracked - set(entries)):
        failures.append(
            f"{path}: in the AP-2 domain but absent from {REGISTRY_REL}. "
            "Classify it explicitly; there is no implicit exclusion."
        )
    # A registry that outlives its files stops describing the repository.
    for path in sorted(set(entries) - tracked):
        failures.append(
            f"{path}: listed in {REGISTRY_REL} but not a tracked file in the domain"
        )

    for path in sorted(set(entries) & tracked):
        entry = entries[path]
        if not isinstance(entry, dict):
            failures.append(f"{path}: entry is not an object")
            continue
        classification = entry.get("classification")
        if classification not in CLASSIFICATIONS:
            failures.append(
                f"{path}: unknown classification {classification!r} "
                f"(expected one of {sorted(CLASSIFICATIONS)})"
            )
            continue
        required = REQUIRED_FIELDS[classification]
        for field in required:
            if not entry.get(field):
                failures.append(
                    f"{path}: {classification} requires a non-empty {field!r}"
                )
        unknown = set(entry) - set(required)
        if unknown:
            failures.append(f"{path}: unknown field(s) {sorted(unknown)}")

        source = (repo_root / path).read_text(encoding="utf-8", errors="replace")
        wired = calls_open_run(source)
        partition_class = partition.classify(path)

        if classification == "run_producer_wired" and not wired:
            failures.append(
                f"{path}: classified run_producer_wired but calls no open_run(). "
                "Either wire it manifest-first or reclassify it."
            )
        if classification == "not_a_run_producer" and wired:
            failures.append(
                f"{path}: classified not_a_run_producer but calls open_run(). "
                "A file that claims an artifact directory is a producer."
            )
        if classification == "run_producer_blocked":
            if partition_class not in PROTECTED_PARTITION_CLASSES:
                failures.append(
                    f"{path}: classified run_producer_blocked but its partition class is "
                    f"{partition_class!r}, which is not protected. Nothing prevents wiring it."
                )
            if wired:
                failures.append(
                    f"{path}: classified run_producer_blocked but already calls open_run()"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()
    try:
        failures = check(Path(args.repo_root))
    except CoverageError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    if failures:
        for line in failures:
            print(f"FAIL: {line}", file=sys.stderr)
        return 1
    print("AP-2 producer coverage: every domain entry point is explicitly classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
