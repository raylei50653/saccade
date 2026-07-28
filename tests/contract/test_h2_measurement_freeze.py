"""The repository-owned H2 Phase-A freeze producer is deterministic and strict."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_measurement_freeze as producer  # noqa: E402
import build_runtime_identity as identity  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402


def _inputs() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
    coordinate = {
        axis: character * 64
        for axis, character in zip(
            identity.ALL_COORDINATE_AXES,
            "12345",
            strict=True,
        )
    }
    probe_digest = "a" * 64
    manifest = {
        "schema": runtime_inputs.SCHEMA,
        "coordinate_digest": coordinate["runtime_inputs"],
        "full_digest": "b" * 64,
        "build_artifacts": {"digest": "c" * 64},
    }
    reference = {
        "schema": behavior.RESULT_SCHEMA,
        "digest": probe_digest,
        "identical": True,
        "mode": "identity",
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    published = {
        "schema": identity.IDENTITY_SCHEMA,
        "coordinate": coordinate,
        "probe": {"digest": probe_digest},
        "equivalence": {"state": "unproven"},
        "publication_complete": True,
    }
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "source_head": head,
        "selected_base": head,
        "changed_path_verdict": {"admissible": True, "base": head},
        "published_coordinate": coordinate,
        "behavior_probe": probe_digest,
        "published_probe": probe_digest,
        "equivalence": "unproven",
        "runtime_input_coordinate_digest": manifest["coordinate_digest"],
        "runtime_input_full_digest": manifest["full_digest"],
        "build_artifact_digest": manifest["build_artifacts"]["digest"],
        "probe_result_file_digest": "d" * 64,
        "runtime_input_manifest_file_digest": "e" * 64,
        "published_identity_file_digest": "f" * 64,
    }
    return {
        "certificate": certificate,
        "certificate_digest": evidence.digest(certificate),
        "reference_probe": reference,
        "reference_probe_file_digest": certificate["probe_result_file_digest"],
        "runtime_manifest": manifest,
        "runtime_manifest_file_digest": certificate[
            "runtime_input_manifest_file_digest"
        ],
        "published_identity": published,
        "published_identity_file_digest": certificate["published_identity_file_digest"],
        "executed_surfaces": {
            path: evidence.digest({"path": path})
            for path in evidence.PHASE_A_EXECUTED_SURFACE_PATHS
        },
        "capture_abi_digest": "9" * 64,
    }


def test_freeze_producer_is_canonical_deterministic_and_complete() -> None:
    first = producer.build_freeze(**_inputs())
    second = producer.build_freeze(**_inputs())
    assert first == second
    assert set(first) == evidence.PHASE_A_FREEZE_MEMBERS
    assert evidence.canonical_json_bytes(first) == evidence.canonical_json_bytes(second)


@pytest.mark.parametrize(
    ("target", "value"),
    (
        ("selected_base", "main"),
        ("selected_base", "a" * 39),
        ("selected_base", "A" * 40),
        ("runtime_coordinate", "9" * 64),
        ("reference_probe", "9" * 64),
        ("published_probe", "9" * 64),
        ("executed_surface_missing", None),
        ("executed_surface_extra", "9" * 64),
    ),
)
def test_freeze_producer_rejects_malformed_or_mismatched_primary_bindings(
    target: str, value: str | None
) -> None:
    inputs = _inputs()
    if target == "selected_base":
        inputs["certificate"]["selected_base"] = value
    elif target == "runtime_coordinate":
        inputs["certificate"]["runtime_input_coordinate_digest"] = value
    elif target == "reference_probe":
        inputs["reference_probe"]["digest"] = value
    elif target == "published_probe":
        inputs["published_identity"]["probe"]["digest"] = value
    elif target == "executed_surface_missing":
        inputs["executed_surfaces"].pop(evidence.PHASE_A_EXECUTED_SURFACE_PATHS[0])
    elif target == "executed_surface_extra":
        inputs["executed_surfaces"]["private.py"] = value
    else:  # pragma: no cover
        raise AssertionError(target)
    with pytest.raises(producer.FreezeError):
        producer.build_freeze(**inputs)


def test_freeze_producer_rejects_changed_path_verdict_for_another_base() -> None:
    inputs = _inputs()
    inputs["certificate"]["changed_path_verdict"]["base"] = "9" * 40
    inputs["certificate_digest"] = evidence.digest(inputs["certificate"])
    with pytest.raises(producer.FreezeError, match="verdict base"):
        producer.build_freeze(**inputs)


def test_freeze_producer_rejects_explicitly_blocked_changed_path_verdict() -> None:
    inputs = _inputs()
    inputs["certificate"]["changed_path_verdict"]["admissible"] = False
    inputs["certificate_digest"] = evidence.digest(inputs["certificate"])
    with pytest.raises(producer.FreezeError, match="admissibility"):
        producer.build_freeze(**inputs)
