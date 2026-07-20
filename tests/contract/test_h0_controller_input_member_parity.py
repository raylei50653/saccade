"""Contract regression for the H0 controller-input member declaration.

Issue #227.  The authorized Stage-E authoritative invocation from the sealed
identity ``S = 8970841d...`` was rejected pre-terminal because the pre-seal
verifier's landing-discovery header enumerated a controller-input member set
that omitted ``build_tool_binding`` while the freeze assembler and the
full-artifact verifier required it.  These tests pin one canonical member
declaration across every runtime transcription, the execution schema, and an
explicit literal, and lock the full/discovery parity behaviour so the two paths
can never silently diverge again.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import run_h0_phase_a as controller  # noqa: E402
import verify_h0_preseal_freeze as freeze_verifier  # noqa: E402

# The single explicit literal, kept independent of every runtime constant so a
# member dropped from *all* transcriptions at once still fails this test.
LITERAL_CONTROLLER_INPUT_MEMBERS = frozenset(
    {
        "authority_landing",
        "bound_inputs",
        "build_tool_binding",
        "document_type",
        "evidence_root",
        "execution_constants",
        "gpu",
        "incomplete_root",
        "instrumentation_head",
        "library_dirs",
        "repository_root",
        "schema",
        "sequence_input_digest",
        "tool_paths",
    }
)
# ``build_tool_binding`` is the sole member introduced after the pre-#224
# substrate: historical artifacts omit it, the current authoritative artifact
# carries it.
LITERAL_OPTIONAL = frozenset({"build_tool_binding"})

SEALED_ARTIFACT = (
    ROOT
    / "docs/modules/semantic/research/evidence"
    / "h0_preseal_freeze_6bc5192c228b752bce42173a058a24374180093c"
    / "h0_preseal_freeze_v3.json"
)
SCHEMA_PATH = TOOLS / "h0_phase_a_execution_schema_v1.json"


def _canonical_controller_input() -> dict[str, object]:
    value = json.loads(SEALED_ARTIFACT.read_text(encoding="utf-8"))
    return value["phase_a_controller_input"]


def test_literal_matches_every_runtime_transcription() -> None:
    assert LITERAL_CONTROLLER_INPUT_MEMBERS == controller.CONTROLLER_INPUT_MEMBERS
    assert LITERAL_CONTROLLER_INPUT_MEMBERS == freeze_verifier.CONTROLLER_INPUT_MEMBERS
    assert LITERAL_OPTIONAL == freeze_verifier.CONTROLLER_INPUT_CROSS_VERSION_OPTIONAL
    assert (
        freeze_verifier.CONTROLLER_INPUT_REQUIRED_BASE
        == LITERAL_CONTROLLER_INPUT_MEMBERS - LITERAL_OPTIONAL
    )
    assert len(LITERAL_CONTROLLER_INPUT_MEMBERS) == 14


def test_execution_schema_pins_the_same_property_universe() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    controller_input = schema["$defs"]["controller_input"]
    assert set(controller_input["properties"]) == LITERAL_CONTROLLER_INPUT_MEMBERS
    assert controller_input["additionalProperties"] is False
    # The shared execution schema stays permissive on ``build_tool_binding`` so
    # historical evidence packets (which predate it) keep validating; the
    # authoritative current artifact's presence is enforced by the strict paths.
    assert (
        set(controller_input["required"])
        == LITERAL_CONTROLLER_INPUT_MEMBERS - LITERAL_OPTIONAL
    )


def test_sealed_artifact_carries_the_canonical_member_set() -> None:
    # Pins the freeze assembler's produced member set through a committed fixture.
    assert set(_canonical_controller_input()) == LITERAL_CONTROLLER_INPUT_MEMBERS


# The two member policies, exercised in isolation from the deep artifact
# verification so parity is asserted independently of git/worktree state:
#   * full/authoritative path -> exact-member gate over CONTROLLER_INPUT_MEMBERS
#     (the first check inside ``_verify_controller_input``);
#   * discovery header path -> ``_require_controller_member_envelope``.
def _full_member_gate(value: object) -> None:
    freeze_verifier._require_exact_members(
        value, freeze_verifier.CONTROLLER_INPUT_MEMBERS, "phase_a_controller_input"
    )


def test_both_paths_accept_the_canonical_artifact() -> None:
    canonical = _canonical_controller_input()
    _full_member_gate(canonical)
    freeze_verifier._require_controller_member_envelope(canonical)


def test_both_paths_reject_a_missing_base_member() -> None:
    broken = copy.deepcopy(_canonical_controller_input())
    del broken["gpu"]
    with pytest.raises(freeze_verifier.VerificationError):
        _full_member_gate(broken)
    with pytest.raises(freeze_verifier.VerificationError):
        freeze_verifier._require_controller_member_envelope(broken)


def test_both_paths_reject_an_unknown_member() -> None:
    broken = copy.deepcopy(_canonical_controller_input())
    broken["surprise"] = True
    with pytest.raises(freeze_verifier.VerificationError):
        _full_member_gate(broken)
    with pytest.raises(freeze_verifier.VerificationError):
        freeze_verifier._require_controller_member_envelope(broken)


def test_build_tool_binding_omission_is_full_strict_but_discovery_tolerant() -> None:
    # The deliberate, documented cross-version asymmetry: the discovery header
    # must accept a historical (build_tool_binding-absent) artifact so
    # enumeration over the mixed-version evidence tree never aborts, while the
    # full/authoritative path — which only ever runs on the selected current
    # candidate — rejects the omission.
    historical = copy.deepcopy(_canonical_controller_input())
    del historical["build_tool_binding"]
    with pytest.raises(freeze_verifier.VerificationError):
        _full_member_gate(historical)
    # Must NOT raise: this is exactly the pre-#224 shape discovery has to accept.
    freeze_verifier._require_controller_member_envelope(historical)


def test_discovery_header_accepts_every_committed_artifact() -> None:
    # End-to-end regression for the Stage-E escape: the discovery member gate
    # must accept every committed v3 controller-input — the current
    # build_tool_binding-bearing artifact and the four historical artifacts that
    # predate it — so enumeration never aborts the way it did at exact S.
    evidence = ROOT / "docs/modules/semantic/research/evidence"
    artifacts = sorted(evidence.glob("**/h0_preseal_freeze_v3.json"))
    assert artifacts, "expected committed v3 landing candidates"
    saw_binding = saw_without = False
    for artifact in artifacts:
        controller_input = json.loads(artifact.read_text(encoding="utf-8"))[
            "phase_a_controller_input"
        ]
        freeze_verifier._require_controller_member_envelope(controller_input)
        if "build_tool_binding" in controller_input:
            saw_binding = True
            _full_member_gate(controller_input)
        else:
            saw_without = True
    # The corpus must actually contain both shapes for this to be a real guard.
    assert saw_binding and saw_without
