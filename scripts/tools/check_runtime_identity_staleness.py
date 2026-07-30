#!/usr/bin/env python3
"""Check runtime-coordinate lag without treating probe equality as equivalence.

Static coordinate drift is a hard publication failure. For bound research,
decision-surface, identity-semantics, or observed-probe drift is ``stale``;
implementation, environment, or runtime-input drift with the same probe is
``re_attestation_required``. There is no behavior-preserving shortcut in this
schema because no equivalence verifier exists.
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402

PUBLISHED_REL = "docs/reference/runtime_identity.generated.json"
BINDINGS_REL = "docs/research/contracts/runtime_identity_bindings_v1.json"
BINDINGS_SCHEMA = "runtime_coordinate_bindings_v1"

STALE_COORDINATE_AXES = ("decision_surface", "identity_semantics")
RE_ATTESTATION_AXES = ("environment", "implementation", "runtime_inputs")
ALL_COORDINATE_AXES = (*STALE_COORDINATE_AXES, *RE_ATTESTATION_AXES)

REGENERATE_HINT = (
    "regenerate with: uv run python scripts/tools/build_runtime_identity.py "
    "--probe-from <identity-probe.json> --runtime-inputs-from "
    "<runtime-inputs.json> --emit " + PUBLISHED_REL
)


class StalenessError(RuntimeError):
    pass


def load_published(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StalenessError(
            f"no published runtime coordinate at {path} — {REGENERATE_HINT}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StalenessError(f"{path}: invalid JSON: {exc}") from exc
    if payload.get("schema") != identity.IDENTITY_SCHEMA:
        raise StalenessError(f"{path}: not an {identity.IDENTITY_SCHEMA} payload")
    coordinate = payload.get("coordinate")
    if not isinstance(coordinate, Mapping):
        raise StalenessError(f"{path}: missing coordinate")
    missing = [axis for axis in ALL_COORDINATE_AXES if axis not in coordinate]
    if missing:
        raise StalenessError(f"{path}: coordinate is missing axes {missing}")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping) or "digest" not in probe:
        raise StalenessError(f"{path}: missing identity probe")
    equivalence = payload.get("equivalence")
    if not isinstance(equivalence, Mapping) or equivalence.get("state") != "unproven":
        raise StalenessError(
            f"{path}: equivalence must remain unproven until a verifier is versioned"
        )
    return payload


def load_bindings(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StalenessError(f"no bindings file at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StalenessError(f"{path}: invalid JSON: {exc}") from exc
    if payload.get("schema") != BINDINGS_SCHEMA:
        raise StalenessError(f"{path}: not a {BINDINGS_SCHEMA} payload")
    if not isinstance(payload.get("bindings"), list):
        raise StalenessError(f"{path}: bindings is not a list")
    return payload


def classify_binding(
    captured_under: dict[str, Any] | None, published: Mapping[str, Any]
) -> str:
    """Return unattested/current/re_attestation_required/stale."""
    if captured_under is None:
        return "unattested"
    if not isinstance(captured_under, dict):
        raise StalenessError(f"captured_under is not a mapping: {captured_under!r}")
    captured_coordinate = captured_under.get("coordinate")
    captured_probe = captured_under.get("probe")
    published_coordinate = published.get("coordinate")
    published_probe = published.get("probe")
    if not isinstance(captured_coordinate, Mapping):
        raise StalenessError("captured_under.coordinate is not a mapping")
    if not isinstance(published_coordinate, Mapping):
        raise StalenessError("published.coordinate is not a mapping")
    if not isinstance(captured_probe, str) or not isinstance(published_probe, str):
        raise StalenessError("captured/published probe digest is missing")
    missing = [axis for axis in ALL_COORDINATE_AXES if axis not in captured_coordinate]
    if missing:
        raise StalenessError(f"captured_under is missing coordinate axes {missing}")
    if captured_probe != published_probe:
        return "stale"
    for axis in STALE_COORDINATE_AXES:
        if captured_coordinate[axis] != published_coordinate.get(axis):
            return "stale"
    for axis in RE_ATTESTATION_AXES:
        if captured_coordinate[axis] != published_coordinate.get(axis):
            return "re_attestation_required"
    return "current"


def _published_binding(publication: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coordinate": dict(publication["coordinate"]),
        "probe": publication["probe"]["digest"],
    }


def compare_publication(
    published: Mapping[str, Any],
    *,
    probe: str | None,
    runtime_input_manifest: Mapping[str, Any] | None = None,
    verify_environment: bool = False,
) -> tuple[list[str], list[str]]:
    """Return hard failures and explicitly unresolved checks.

    Source-derived axes are portable and can be checked on every host. The
    environment coordinate contains observed Torch/CUDA/TensorRT/device state,
    so it is checked only on the controlled attestation host. A generic CPU CI
    runner must report that check as unresolved instead of comparing itself to
    a GPU publication and manufacturing drift.
    """
    recomputed = {
        "decision_surface": identity.decision_surface_axis()["digest"],
        "implementation": identity.implementation_axis()["digest"],
        "identity_semantics": identity.identity_semantics_axis()["digest"],
    }
    coordinate = published["coordinate"]
    failures: list[str] = []
    warnings: list[str] = []
    for axis, measured in recomputed.items():
        if coordinate.get(axis) != measured:
            failures.append(
                f"{axis} moved and was not republished: published "
                f"{coordinate.get(axis)}, recomputed {measured}. {REGENERATE_HINT}"
            )

    if verify_environment:
        measured_environment = identity.environment_axis()["digest"]
        if coordinate.get("environment") != measured_environment:
            failures.append(
                "environment moved and was not republished: published "
                f"{coordinate.get('environment')}, recomputed "
                f"{measured_environment}. {REGENERATE_HINT}"
            )
    else:
        warnings.append(
            "host-specific environment was not recomputed; the manual controlled-host "
            "diagnostic may observe it, while successor executions bind their own "
            "runtime environment and artifacts"
        )

    if runtime_input_manifest is None:
        warnings.append(
            "runtime-input content was not recomputed; fixture/model/engine currentness "
            "is unresolved for this legacy publication until --runtime-inputs-from is "
            "supplied; successor executions bind the inputs they consume"
        )
    else:
        current_inputs = runtime_inputs.publication_axis(runtime_input_manifest)[
            "digest"
        ]
        if coordinate.get("runtime_inputs") != current_inputs:
            failures.append(
                "runtime-input content moved: published "
                f"{coordinate.get('runtime_inputs')}, recomputed {current_inputs}"
            )

    published_probe = published["probe"]["digest"]
    if probe is None:
        warnings.append(
            "identity probe was not recomputed; equality and equivalence are both unclaimed"
        )
    elif published_probe != probe:
        failures.append(
            f"identity probe moved: published {published_probe}, measured {probe}"
        )
    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-from", type=Path, default=None)
    parser.add_argument("--runtime-inputs-from", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="fail unresolved checks")
    args = parser.parse_args(argv)

    try:
        published = load_published(REPO_ROOT / PUBLISHED_REL)
        bindings = load_bindings(REPO_ROOT / BINDINGS_REL)
        probe_digest = None
        if args.probe_from:
            probe_digest = identity.load_identity_behavior_probe(args.probe_from)[
                "digest"
            ]
        manifest = None
        if args.runtime_inputs_from:
            manifest = runtime_inputs.load_manifest(
                args.runtime_inputs_from, verify_files=True
            )
        failures, warnings = compare_publication(
            published,
            probe=probe_digest,
            runtime_input_manifest=manifest,
            verify_environment=args.strict,
        )

        lag: dict[str, list[str]] = {}
        binding_target = _published_binding(published)
        for binding in bindings["bindings"]:
            verdict = classify_binding(binding.get("captured_under"), binding_target)
            lag.setdefault(verdict, []).append(str(binding.get("object")))
    except (
        StalenessError,
        identity.IdentityError,
        runtime_inputs.RuntimeInputError,
        OSError,
    ) as exc:
        print(f"runtime-coordinate staleness check failed: {exc}", file=sys.stderr)
        return 1

    for verdict in (
        "stale",
        "re_attestation_required",
        "current",
        "unattested",
    ):
        for name in sorted(lag.get(verdict, [])):
            print(f"  {verdict:26} {name}")
    for verdict in ("stale", "re_attestation_required"):
        if lag.get(verdict):
            failures.append(
                f"{verdict} bindings are inadmissible: "
                + ", ".join(sorted(lag[verdict]))
            )
    for message in warnings:
        print(f"warning: {message}")
    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    if failures:
        return 1
    if args.strict and warnings:
        print("strict mode: unresolved checks are failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
