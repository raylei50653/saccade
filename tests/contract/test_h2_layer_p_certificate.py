"""Layer-P cannot skip admissibility and its certificate binds the full coordinate."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_behavioral_identity as behavior  # noqa: E402
import run_h2_layer_p as layer_p  # noqa: E402


def test_base_is_required_by_the_cli() -> None:
    with pytest.raises(SystemExit):
        layer_p.main([])


def test_certificate_binds_every_review_required_coordinate(tmp_path: Path) -> None:
    controller = layer_p.LayerP.__new__(layer_p.LayerP)
    controller.base = "selected-I"
    controller.build_dir_rel = "build/h2"
    controller.retry_verdict = {
        "admissible": True,
        "base": "selected-I",
        "changed_count": 0,
        "decision_relevant": [],
        "identity_semantics": [],
        "identity_fixture_input": [],
        "measurement_input": [],
        "runtime_asset": [],
        "unclassified": [],
        "plumbing_only": [],
        "non_execution": [],
    }
    probe_path = tmp_path / "probe.json"
    manifest_path = tmp_path / "runtime-inputs.json"
    probe_path.write_bytes(b"probe\n")
    manifest_path.write_bytes(b"manifest\n")
    published = {
        "coordinate": {
            "decision_surface": "d" * 64,
            "environment": "e" * 64,
            "implementation": "i" * 64,
            "identity_semantics": "s" * 64,
            "runtime_inputs": "r" * 64,
        },
        "probe": {"digest": "p" * 64},
    }
    probe = {
        "build_witness": {"digest": "b" * 64},
        "digest": "p" * 64,
        "mode": "identity",
        "schema": behavior.RESULT_SCHEMA,
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    manifest = {
        "build_artifacts": {"digest": "a" * 64},
        "coordinate_digest": "r" * 64,
        "full_digest": "f" * 64,
    }
    certificate = controller.build_certificate(
        published=published,
        probe=probe,
        manifest=manifest,
        probe_path=probe_path,
        manifest_path=manifest_path,
        extension_witness={"extension_sha256": "x" * 64},
    )
    required = {
        "source_head",
        "source_tree",
        "selected_base",
        "changed_path_verdict",
        "decision_relevant_digest",
        "identity_semantics_digest",
        "plumbing_set_digest",
        "published_identity_file_digest",
        "probe_result_file_digest",
        "runtime_input_manifest_file_digest",
        "runtime_input_coordinate_digest",
        "runtime_input_full_digest",
        "build_artifact_digest",
        "behavior_probe",
        "probe_schema",
        "fixture",
        "mode",
        "equivalence",
    }
    assert required <= set(certificate)
    assert certificate["selected_base"] == "selected-I"
    assert certificate["equivalence"] == "unproven"
    assert certificate["changed_path_verdict"]["admissible"] is True
