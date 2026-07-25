"""Version lag must be detected, and typed correctly.

This is the `online → research` guard that never existed: the registry has always
carried `substrate` / `target_substrate` and the rule that substrate does not
inherit, but nothing checked whether the substrate still existed. A preset default
could move and every state proven on it would quietly stop meaning what it says.

The property that makes a dual track real is the **asymmetry** between axes:

  * `implementation` / `environment` may move freely while `behavior` holds — that
    is the online track shipping without invalidating sealed research;
  * `decision_surface` or `behavior` moving is decision-affecting — consumers go
    inadmissible until re-attested.

Getting that asymmetry backwards would either freeze the online track or silently
bless stale claims, so each direction is pinned here. A missing binding is also
never read as agreement: absence fails closed.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import check_runtime_identity_staleness as staleness  # noqa: E402

_PUBLISHED = {
    "behavior": "b" * 64,
    "decision_surface": "d" * 64,
    "environment": "e" * 64,
    "implementation": "i" * 64,
}


def _captured(**overrides: str) -> dict[str, str]:
    return {**_PUBLISHED, **overrides}


def test_an_identical_binding_is_current() -> None:
    assert staleness.classify_binding(_captured(), _PUBLISHED) == "current"


@pytest.mark.parametrize("axis", ["implementation", "environment"])
def test_implementation_and_environment_drift_is_behavior_preserving(axis: str) -> None:
    """The online track must be able to move without invalidating research."""
    verdict = staleness.classify_binding(_captured(**{axis: "9" * 64}), _PUBLISHED)
    assert verdict == "behavior_preserving"


@pytest.mark.parametrize("axis", ["decision_surface", "behavior"])
def test_decision_surface_and_behavior_drift_is_stale(axis: str) -> None:
    verdict = staleness.classify_binding(_captured(**{axis: "9" * 64}), _PUBLISHED)
    assert verdict == "stale"


def test_decision_affecting_drift_wins_over_behavior_preserving_drift() -> None:
    """A release that moves everything is stale, not behavior-preserving."""
    verdict = staleness.classify_binding(
        _captured(implementation="9" * 64, behavior="9" * 64), _PUBLISHED
    )
    assert verdict == "stale"


def test_the_axis_classes_are_disjoint_and_complete() -> None:
    decision = set(staleness.DECISION_AFFECTING_AXES)
    preserving = set(staleness.BEHAVIOR_PRESERVING_AXES)
    assert not decision & preserving
    assert decision | preserving == set(_PUBLISHED)


def test_a_null_binding_is_unattested_not_current() -> None:
    """Absence of a claim is not agreement with the current identity."""
    assert staleness.classify_binding(None, _PUBLISHED) == "unattested"


def test_a_partial_binding_fails_closed() -> None:
    partial = {"behavior": "b" * 64, "decision_surface": "d" * 64}
    with pytest.raises(staleness.StalenessError, match="missing axes"):
        staleness.classify_binding(partial, _PUBLISHED)


def test_a_malformed_binding_fails_closed() -> None:
    with pytest.raises(staleness.StalenessError):
        staleness.classify_binding("b" * 64, _PUBLISHED)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Intake                                                                       #
# --------------------------------------------------------------------------- #
def test_a_missing_published_identity_fails_with_a_regeneration_hint(
    tmp_path: Path,
) -> None:
    with pytest.raises(staleness.StalenessError, match="build_runtime_identity"):
        staleness.load_published(tmp_path / "absent.json")


def test_a_foreign_published_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"schema": "other_v1", "identity": {}}))
    with pytest.raises(staleness.StalenessError):
        staleness.load_published(path)


def test_a_foreign_bindings_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps({"schema": "other_v1", "bindings": []}))
    with pytest.raises(staleness.StalenessError):
        staleness.load_bindings(path)


# --------------------------------------------------------------------------- #
# The checked-in artifacts                                                     #
# --------------------------------------------------------------------------- #
def test_the_published_identity_is_complete_and_current() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert published["complete"] is True, (
        "the published identity has no behavior axis; it must not be cited as a "
        "substrate coordinate"
    )
    failures, _warnings = staleness.compare_publication(published, behavior=None)
    assert not failures, failures


def test_every_checked_in_binding_classifies() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    assert bindings["bindings"], "no bindings: the guard would be checking nothing"
    for binding in bindings["bindings"]:
        verdict = staleness.classify_binding(
            binding.get("captured_under"), published["identity"]
        )
        assert verdict in {"current", "behavior_preserving", "stale", "unattested"}


def test_no_binding_claims_a_capture_that_does_not_exist() -> None:
    """H0 produced no faithful capture, so nothing may be bound retroactively.

    A binding here would assert that some accepted evidence was captured under a
    published identity. Five spent chains produced none, and inventing one would
    be exactly the laundering the fidelity protocol § 2.8 forbids.
    """
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    for binding in bindings["bindings"]:
        if binding["object"] == "quantity.bridge_capture_provenance":
            assert binding["captured_under"] is None


def test_the_registry_remains_the_state_owner() -> None:
    """C5.1: this sidecar owns digests, never state."""
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    for binding in bindings["bindings"]:
        assert "state" not in binding
        assert binding["state_owner"].endswith("claim_state_registry.md")


def test_the_published_identity_records_physical_data_as_witness_only() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert "no decision authority" in published["witness"]["note"]
    assert set(published["identity"]) == set(_PUBLISHED)
    # Physical artifact identity must not appear among the axes.
    assert "extension_sha256" not in published["identity"]


def test_the_identity_schema_matches_the_builder() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert published["schema"] == identity.IDENTITY_SCHEMA
