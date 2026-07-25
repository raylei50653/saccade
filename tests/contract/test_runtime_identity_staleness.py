"""Coordinate drift never becomes equivalence merely because one probe is equal."""

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

_COORDINATE = {
    "decision_surface": "d" * 64,
    "environment": "e" * 64,
    "implementation": "i" * 64,
    "identity_semantics": "s" * 64,
    "runtime_inputs": "r" * 64,
}
_PUBLISHED = {"coordinate": _COORDINATE, "probe": "p" * 64}


def _captured(
    *, probe: str = "p" * 64, **coordinate_overrides: str
) -> dict[str, object]:
    return {
        "coordinate": {**_COORDINATE, **coordinate_overrides},
        "probe": probe,
    }


def test_an_identical_binding_is_current() -> None:
    assert staleness.classify_binding(_captured(), _PUBLISHED) == "current"


@pytest.mark.parametrize("axis", staleness.RE_ATTESTATION_AXES)
def test_equal_probe_does_not_make_coordinate_drift_preserving(axis: str) -> None:
    verdict = staleness.classify_binding(_captured(**{axis: "9" * 64}), _PUBLISHED)
    assert verdict == "re_attestation_required"


@pytest.mark.parametrize("axis", staleness.STALE_COORDINATE_AXES)
def test_decision_authority_drift_is_stale(axis: str) -> None:
    verdict = staleness.classify_binding(_captured(**{axis: "9" * 64}), _PUBLISHED)
    assert verdict == "stale"


def test_probe_drift_is_stale() -> None:
    assert staleness.classify_binding(_captured(probe="9" * 64), _PUBLISHED) == "stale"


def test_stale_drift_wins_over_unresolved_coordinate_drift() -> None:
    verdict = staleness.classify_binding(
        _captured(implementation="9" * 64, decision_surface="9" * 64),
        _PUBLISHED,
    )
    assert verdict == "stale"


def test_there_is_no_behavior_preserving_axis_class() -> None:
    assert not hasattr(staleness, "BEHAVIOR_PRESERVING_AXES")
    assert set(staleness.ALL_COORDINATE_AXES) == set(_COORDINATE)


def test_a_null_binding_is_unattested_not_current() -> None:
    assert staleness.classify_binding(None, _PUBLISHED) == "unattested"


def test_a_partial_binding_fails_closed() -> None:
    partial = {
        "coordinate": {"decision_surface": "d" * 64},
        "probe": "p" * 64,
    }
    with pytest.raises(staleness.StalenessError, match="missing coordinate axes"):
        staleness.classify_binding(partial, _PUBLISHED)


def test_a_malformed_binding_fails_closed() -> None:
    with pytest.raises(staleness.StalenessError):
        staleness.classify_binding("bad", _PUBLISHED)  # type: ignore[arg-type]


def test_a_missing_publication_fails_with_a_regeneration_hint(
    tmp_path: Path,
) -> None:
    with pytest.raises(staleness.StalenessError, match="build_runtime_identity"):
        staleness.load_published(tmp_path / "absent.json")


def test_a_foreign_publication_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"schema": "other_v1", "coordinate": {}}))
    with pytest.raises(staleness.StalenessError):
        staleness.load_published(path)


def test_a_foreign_bindings_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps({"schema": "other_v1", "bindings": []}))
    with pytest.raises(staleness.StalenessError):
        staleness.load_bindings(path)


def test_the_published_coordinate_is_complete_and_static_axes_are_current() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert published["publication_complete"] is True
    assert published["equivalence"]["state"] == "unproven"
    failures, warnings = staleness.compare_publication(
        published, probe=None, runtime_input_manifest=None
    )
    assert not failures, failures
    assert warnings


def test_host_environment_is_checked_only_on_a_controlled_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    monkeypatch.setattr(identity, "environment_axis", lambda: {"digest": "9" * 64})

    failures, warnings = staleness.compare_publication(
        published, probe=None, runtime_input_manifest=None
    )
    assert not failures
    assert any(
        "host-specific environment was not recomputed" in item for item in warnings
    )

    failures, _warnings = staleness.compare_publication(
        published,
        probe=None,
        runtime_input_manifest=None,
        verify_environment=True,
    )
    assert any("environment moved" in item for item in failures)


def test_every_checked_in_binding_classifies() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    target = {
        "coordinate": published["coordinate"],
        "probe": published["probe"]["digest"],
    }
    assert bindings["bindings"]
    for binding in bindings["bindings"]:
        verdict = staleness.classify_binding(binding.get("captured_under"), target)
        assert verdict in {
            "current",
            "re_attestation_required",
            "stale",
            "unattested",
        }


def test_no_binding_claims_a_capture_that_does_not_exist() -> None:
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    for binding in bindings["bindings"]:
        if binding["object"] == "quantity.bridge_capture_provenance":
            assert binding["captured_under"] is None


def test_the_registry_remains_the_state_owner() -> None:
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    for binding in bindings["bindings"]:
        assert "state" not in binding
        assert binding["state_owner"].endswith("claim_state_registry.md")


def test_the_publication_separates_coordinate_probe_and_equivalence() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert set(published["coordinate"]) == set(identity.ALL_COORDINATE_AXES)
    assert published["probe"]["kind"] == "identity_probe"
    assert published["probe"]["sufficiency"] == "fixture_change_detector_only"
    assert published["equivalence"] == {
        "proof": None,
        "state": "unproven",
        "note": published["equivalence"]["note"],
    }


def test_the_identity_schema_matches_the_builder() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert published["schema"] == identity.IDENTITY_SCHEMA
