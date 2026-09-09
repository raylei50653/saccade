#!/usr/bin/env python3
"""Check runtime-coordinate lag without treating probe equality as equivalence.

For bound research, decision-surface, identity-semantics, or observed-probe
drift is ``stale``; implementation, environment, or runtime-input drift with the
same probe is ``re_attestation_required``. There is no behavior-preserving
shortcut in this schema because no equivalence verifier exists.

ADR 022 separates two questions that used to share one exit code.

``--mode development`` (the default) asks whether anything is **consuming** the
publication as true of HEAD. When no binding classifies ``current``, portable
lag means only that the publication describes an older HEAD, and that is
reported as a warning: ordinary development does not owe a republication for
evidence nobody is claiming. Every other failure stays fail-closed, including
lag the moment a binding does classify ``current`` -- that combination is a
false current-attestation, not lag.

``--mode attested`` adds the claim that the publication describes HEAD now.
Portable lag is a failure there. Run it when you intend to assert that: before
promoting a publication, or when a consumer binds evidence to it.

Portable lag is what any host can recompute from git objects: the three source
axes and the environment *recipe* (``CMakeLists.txt``/``pyproject.toml``/
``uv.lock``). The observed Torch/CUDA/TensorRT closure is host state and is
compared only under ``--strict`` on the controlled host. Those two used to be
one atom behind ``--strict``, which is how a ``pyproject.toml`` change stayed
invisible to every ordinary check.
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

# The source-derived coordinate axes, recomputable from git on any host.
PORTABLE_COORDINATE_AXES = (
    "decision_surface",
    "implementation",
    "identity_semantics",
)
# Not a coordinate axis: the portable half of the environment axis, reported
# under its own name so it is never confused with the observed toolchain.
ENVIRONMENT_RECIPE = "environment.recipe"

DEVELOPMENT_MODE = "development"
ATTESTED_MODE = "attested"
MODES = (DEVELOPMENT_MODE, ATTESTED_MODE)

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
    # ADR 022 §4: an incomplete publication must never stand in for a complete
    # one. The builder refuses to write one over the canonical path; this is the
    # second, independent place, so a file that arrives some other way is still
    # refused at read time by every consumer.
    if payload.get("publication_complete") is not True:
        raise StalenessError(
            f"{path}: publication_complete is "
            f"{payload.get('publication_complete')!r}; an incomplete publication "
            f"cannot stand in for a complete one — {REGENERATE_HINT}"
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


def _environment_recipe_lag(
    published: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Compare the environment axis's git blobs; None when there is nothing to compare.

    Returns ``(published_blobs, recomputed_blobs)`` when they differ. A
    publication that carries no `axes.environment.recipe` detail predates this
    check; that is reported as unresolved by the caller rather than invented as
    drift, which is the same treatment runtime inputs and the probe already get.
    """
    axes = published.get("axes")
    if not isinstance(axes, Mapping):
        return None
    environment = axes.get("environment")
    if not isinstance(environment, Mapping):
        return None
    recorded = environment.get("recipe")
    if not isinstance(recorded, list) or not recorded:
        return None
    was = {
        item.get("path"): item.get("blob")
        for item in recorded
        if isinstance(item, Mapping)
    }
    now = {item["path"]: item["blob"] for item in identity.environment_recipe()}
    return None if was == now else (was, now)


def _publication_records_its_recipe(published: Mapping[str, Any]) -> bool:
    axes = published.get("axes")
    if not isinstance(axes, Mapping):
        return False
    environment = axes.get("environment")
    if not isinstance(environment, Mapping):
        return False
    recorded = environment.get("recipe")
    return isinstance(recorded, list) and bool(recorded)


def static_axis_lag(
    published: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Portable lag only: name -> (published, recomputed), empty when current.

    "Portable" means recomputable from git objects on any host, so every entry
    here is checkable during ordinary development. Callers decide what it means:
    `compare_publication` treats it as failure, `--mode development` treats it as
    a warning until something claims the publication as current. Splitting that
    decision out of the recomputation is what lets both readings share one
    definition of what moved.

    The observed Torch/CUDA/TensorRT closure is deliberately absent: it is host
    state, not a git object, and only the controlled host can compare it.
    """
    coordinate = published["coordinate"]
    recomputed = {
        "decision_surface": identity.decision_surface_axis()["digest"],
        "implementation": identity.implementation_axis()["digest"],
        "identity_semantics": identity.identity_semantics_axis()["digest"],
    }
    lag: dict[str, tuple[Any, Any]] = {
        axis: (coordinate.get(axis), measured)
        for axis, measured in recomputed.items()
        if coordinate.get(axis) != measured
    }
    recipe = _environment_recipe_lag(published)
    if recipe is not None:
        lag[ENVIRONMENT_RECIPE] = recipe
    return lag


def _lag_message(name: str, was: Any, measured: Any) -> str:
    """The wording is load-bearing: two consumers match on these strings."""
    if name == ENVIRONMENT_RECIPE:
        moved = sorted(path for path, blob in measured.items() if was.get(path) != blob)
        return (
            "environment recipe moved and was not republished: "
            + ", ".join(f"{path} {was.get(path)} -> {measured[path]}" for path in moved)
            + f". {REGENERATE_HINT}"
        )
    return (
        f"{name} moved and was not republished: published "
        f"{was}, recomputed {measured}. {REGENERATE_HINT}"
    )


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

    This is the attested reading: portable lag is a failure. `--mode
    development` composes the same pieces with lag reclassified, so this
    function's signature, return shape and failure strings stay exactly as its
    two external consumers (`research_lock.publication_precondition` and
    `run_h2_layer_p.preflight`, which stores `warnings` in a certificate) read
    them today.
    """
    coordinate = published["coordinate"]
    failures: list[str] = []
    warnings: list[str] = []
    lag = static_axis_lag(published)
    for axis in PORTABLE_COORDINATE_AXES:
        if axis in lag:
            failures.append(_lag_message(axis, *lag[axis]))

    # New coverage, deliberately outside `verify_environment`: these are git
    # blobs, so the check is portable even though the axis they belong to is
    # not. Bundling them with the observed toolchain is what hid a
    # `pyproject.toml`/`uv.lock` change from every non-controlled-host check.
    if ENVIRONMENT_RECIPE in lag:
        failures.append(_lag_message(ENVIRONMENT_RECIPE, *lag[ENVIRONMENT_RECIPE]))
    elif not _publication_records_its_recipe(published):
        warnings.append(
            "publication records no environment recipe; its portable "
            "CMakeLists.txt/pyproject.toml/uv.lock identity is unresolved"
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
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=DEVELOPMENT_MODE,
        help=(
            "development (default): portable lag is a warning while no binding is "
            "current. attested: portable lag is a failure, i.e. the publication "
            "claims to describe HEAD."
        ),
    )
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

        # Bindings first: whether portable lag is a failure depends on whether
        # anything claims the publication as current, so the walk cannot run
        # after the decision it informs.
        verdicts: dict[str, list[str]] = {}
        binding_target = _published_binding(published)
        for binding in bindings["bindings"]:
            verdict = classify_binding(binding.get("captured_under"), binding_target)
            verdicts.setdefault(verdict, []).append(str(binding.get("object")))
        consumed_as_current = bool(verdicts.get("current"))

        failures, warnings = compare_publication(
            published,
            probe=probe_digest,
            runtime_input_manifest=manifest,
            verify_environment=args.strict,
        )
        if args.mode == DEVELOPMENT_MODE and not consumed_as_current:
            # Reclassify by name, never by matching the message text: the lag
            # entries come from the same helper `compare_publication` used.
            lag = static_axis_lag(published)
            demoted = {_lag_message(name, *value) for name, value in lag.items()}
            failures = [message for message in failures if message not in demoted]
            warnings.extend(
                f"publication lag (nothing consumes it as current): "
                f"{_lag_message(name, *lag[name])}"
                for name in sorted(lag)
            )
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
        for name in sorted(verdicts.get(verdict, [])):
            print(f"  {verdict:26} {name}")
    for verdict in ("stale", "re_attestation_required"):
        if verdicts.get(verdict):
            failures.append(
                f"{verdict} bindings are inadmissible: "
                + ", ".join(sorted(verdicts[verdict]))
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
