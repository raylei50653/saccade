#!/usr/bin/env python3
"""Flag version lag between the online track and the research claims that cite it.

This closes the `online → research` direction, which had no protection at all:
the claim-state registry has always carried `substrate` / `target_substrate` and
the rule that substrate does not inherit, but **nothing checked whether the
substrate still existed**. A preset default or a kernel constant could move and
every state proven on it would quietly stop meaning what it says.

Two jobs, both cheap and GPU-free:

1. **Publication freshness.** Recompute the three static axes and compare them to
   `docs/reference/runtime_identity.generated.json`.
     * `decision_surface` moved and was not republished  → **hard failure**. The
       policy surface changed; leaving the published identity behind would leave
       research citing a substrate that no longer exists.
     * `implementation` / `environment` moved            → **warning**, plus a
       note that the `behavior` axis needs re-attestation on a GPU host. These
       are behavior-preserving *if and only if* the behavior digest is unchanged,
       and only the identity run can decide that.
     * `behavior` is never recomputed here (it needs a GPU); it is compared only
       when a fresh identity-mode result is supplied with `--behavior-from`.

2. **Binding staleness.** For every non-null `captured_under` in
   `runtime_identity_bindings_v1.json`, classify it `current`,
   `behavior_preserving`, or `stale`.

A stale flag is version lag, not a retraction — a closure established on an
identity stays true of that identity (fidelity protocol § 2.8).

Usage:
  uv run python scripts/tools/check_runtime_identity_staleness.py
  uv run python scripts/tools/check_runtime_identity_staleness.py --behavior-from out/h2_behavior/g1_a.json
  uv run python scripts/tools/check_runtime_identity_staleness.py --strict   # warnings fail too
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402

PUBLISHED_REL = "docs/reference/runtime_identity.generated.json"
BINDINGS_REL = "docs/research/contracts/runtime_identity_bindings_v1.json"

# Declaration § 8.1: which axes make a binding stale rather than behavior-preserving.
DECISION_AFFECTING_AXES = ("behavior", "decision_surface")
BEHAVIOR_PRESERVING_AXES = ("environment", "implementation")

REGENERATE_HINT = (
    "regenerate with: uv run python scripts/tools/build_runtime_identity.py "
    "--behavior-from <identity-run.json> --emit " + PUBLISHED_REL
)


class StalenessError(RuntimeError):
    pass


def load_published(path: Path) -> dict[str, Any]:
    if not path.is_file():
        # Not `relative_to(REPO_ROOT)`: the path may legitimately be outside the
        # repository, and a fail-closed branch that raises the wrong exception type
        # is not fail-closed.
        shown = (
            path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT)
            else path.as_posix()
        )
        raise StalenessError(
            f"no published runtime identity at {shown} — " + REGENERATE_HINT
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != identity.IDENTITY_SCHEMA:
        raise StalenessError(f"{path}: not an {identity.IDENTITY_SCHEMA} payload")
    if not isinstance(payload.get("identity"), dict):
        raise StalenessError(f"{path}: missing identity axes")
    return payload


def load_bindings(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StalenessError(f"no bindings file at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "runtime_identity_bindings_v1":
        raise StalenessError(f"{path}: not a runtime_identity_bindings_v1 payload")
    if not isinstance(payload.get("bindings"), list):
        raise StalenessError(f"{path}: bindings is not a list")
    return payload


def classify_binding(
    captured_under: dict[str, Any] | None, published: dict[str, Any]
) -> str:
    """`unattested` / `current` / `behavior_preserving` / `stale`."""
    if captured_under is None:
        return "unattested"
    if not isinstance(captured_under, dict):
        raise StalenessError(f"captured_under is not a mapping: {captured_under!r}")
    missing = [
        axis
        for axis in (*DECISION_AFFECTING_AXES, *BEHAVIOR_PRESERVING_AXES)
        if axis not in captured_under
    ]
    if missing:
        # A partial binding cannot be judged; fail closed rather than guess.
        raise StalenessError(f"captured_under is missing axes {missing}")
    for axis in DECISION_AFFECTING_AXES:
        if captured_under[axis] != published.get(axis):
            return "stale"
    for axis in BEHAVIOR_PRESERVING_AXES:
        if captured_under[axis] != published.get(axis):
            return "behavior_preserving"
    return "current"


def compare_publication(
    published: dict[str, Any], *, behavior: str | None
) -> tuple[list[str], list[str]]:
    """Returns (hard_failures, warnings)."""
    recomputed = {
        "decision_surface": identity.decision_surface_axis()["digest"],
        "environment": identity.environment_axis()["digest"],
        "implementation": identity.implementation_axis()["digest"],
    }
    axes = published["identity"]
    failures: list[str] = []
    warnings: list[str] = []

    if axes.get("decision_surface") != recomputed["decision_surface"]:
        failures.append(
            "decision_surface moved and was not republished:\n"
            f"    published  {axes.get('decision_surface')}\n"
            f"    recomputed {recomputed['decision_surface']}\n"
            "  The policy surface changed. Every state captured under the published "
            "identity is stale until the identity is republished and the affected "
            "bindings re-attested.\n  " + REGENERATE_HINT
        )

    for axis in BEHAVIOR_PRESERVING_AXES:
        if axes.get(axis) != recomputed[axis]:
            warnings.append(
                f"{axis} moved: published {axes.get(axis)} vs recomputed "
                f"{recomputed[axis]}. Behavior-preserving only if the behavior axis "
                "is unchanged — re-attest on a GPU host with "
                "h2_behavioral_identity.py --identity-mode."
            )

    if behavior is not None and axes.get("behavior") != behavior:
        failures.append(
            "behavior moved:\n"
            f"    published {axes.get('behavior')}\n"
            f"    measured  {behavior}\n"
            "  Policy-visible behavior changed; this is decision-affecting."
        )
    elif behavior is None and axes.get("behavior") is None:
        warnings.append(
            "published identity has no behavior axis — it is incomplete and must "
            "not be cited as a substrate coordinate"
        )
    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behavior-from",
        type=Path,
        default=None,
        help="an h2_behavioral_identity result to compare the behavior axis against",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    args = parser.parse_args(argv)

    try:
        published = load_published(REPO_ROOT / PUBLISHED_REL)
        bindings = load_bindings(REPO_ROOT / BINDINGS_REL)
        behavior = None
        if args.behavior_from:
            behavior = identity.load_behavior(args.behavior_from)["digest"]
        failures, warnings = compare_publication(published, behavior=behavior)

        lag: dict[str, list[str]] = {}
        for binding in bindings["bindings"]:
            verdict = classify_binding(
                binding.get("captured_under"), published["identity"]
            )
            lag.setdefault(verdict, []).append(str(binding.get("object")))
    except (StalenessError, identity.IdentityError, json.JSONDecodeError) as exc:
        print(f"runtime-identity staleness check failed: {exc}", file=sys.stderr)
        return 1

    for verdict in ("stale", "behavior_preserving", "current", "unattested"):
        for name in sorted(lag.get(verdict, [])):
            print(f"  {verdict:20} {name}")
    if lag.get("stale"):
        failures.append(
            "stale bindings (consumers inadmissible until re-attested): "
            + ", ".join(sorted(lag["stale"]))
        )

    for text in warnings:
        print(f"warning: {text}")
    for text in failures:
        print(f"FAIL: {text}", file=sys.stderr)

    if failures:
        return 1
    if warnings:
        # Never report "current" while reporting drift: the summary line is what
        # a reader remembers, and a reassuring summary over a warning is how a
        # version-lag flag gets ignored.
        if args.strict:
            print("strict mode: warnings are failures", file=sys.stderr)
            return 1
        print(
            f"runtime identity: {len(warnings)} behavior-preserving drift warning(s); "
            "behavior axis needs re-attestation on a GPU host"
        )
        return 0
    print("runtime identity: published axes current, no stale bindings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
