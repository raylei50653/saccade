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


# ── Wired-path exercises ──────────────────────────────────────────────────────
# These drive the *actual* verifier functions, not the bare member helpers:
#   * discovery header  -> freeze_verifier._verify_landing_candidate_header
#     (which internally calls the discovery member envelope);
#   * full/authoritative -> freeze_verifier._verify_controller_input
#     (whose first check is the exact-14 member gate);
#   * controller entry   -> run_h0_phase_a._discover_controller_input.
# The full/authoritative host-fidelity tail (_verify_host_execution_inputs: nvml
# GPU, ldd build-tool binding, the whole repository/model/sequence inventory, and
# a physical .venv at ROOT) is only satisfiable in a real host-bound sealed
# checkout, so a *positive* full-verify selection cannot be produced portably or
# pre-seal; it is exercised by the controlled-host qualification step. The member
# contract these tests target is fully reachable because it fires before that
# tail.


def _write_landing_candidate(
    tmp_root: Path, controller_input: dict[str, object]
) -> tuple[Path, str]:
    """Materialise a committed-shaped v3 artifact at its deterministic path.

    Re-homes the sealed artifact under ``tmp_root`` (only ``repository_root`` is
    host-specific) so ``_verify_landing_candidate_header`` — a git/host-free,
    file-and-shape gate — runs end-to-end against a controlled member set.
    """
    artifact = json.loads(SEALED_ARTIFACT.read_text(encoding="utf-8"))
    head = artifact["instrumentation_head"]
    controller_input = dict(controller_input)
    controller_input["repository_root"] = tmp_root.resolve().as_posix()
    artifact["phase_a_controller_input"] = controller_input
    path = tmp_root / freeze_verifier.freeze_path(head)
    path.parent.mkdir(parents=True)
    path.write_bytes(freeze_verifier.canonical_json(artifact) + b"\n")
    return path, head


def test_actual_landing_header_accepts_canonical_and_historical(tmp_path: Path) -> None:
    # The real discovery header accepts both the current 14-member shape and the
    # pre-#224 13-member (build_tool_binding-absent) shape.
    canonical = _canonical_controller_input()
    path, _head = _write_landing_candidate(tmp_path / "current", canonical)
    freeze_verifier._verify_landing_candidate_header(path, tmp_path / "current")

    historical = copy.deepcopy(canonical)
    del historical["build_tool_binding"]
    path, _head = _write_landing_candidate(tmp_path / "historical", historical)
    freeze_verifier._verify_landing_candidate_header(path, tmp_path / "historical")


def test_actual_landing_header_rejects_unknown_and_missing_base(tmp_path: Path) -> None:
    unknown = copy.deepcopy(_canonical_controller_input())
    unknown["surprise"] = True
    path, _head = _write_landing_candidate(tmp_path / "unknown", unknown)
    with pytest.raises(freeze_verifier.VerificationError, match="missing or unknown"):
        freeze_verifier._verify_landing_candidate_header(path, tmp_path / "unknown")

    missing = copy.deepcopy(_canonical_controller_input())
    del missing["gpu"]
    path, _head = _write_landing_candidate(tmp_path / "missing", missing)
    with pytest.raises(freeze_verifier.VerificationError, match="missing or unknown"):
        freeze_verifier._verify_landing_candidate_header(path, tmp_path / "missing")


def test_actual_full_verifier_member_gate(tmp_path: Path) -> None:
    # ``_verify_controller_input``'s exact-14 member gate is its first check, so
    # it fires (and rejects) before any host-fidelity derivation. A missing base
    # member, an unknown member, and a build_tool_binding omission are each
    # rejected by the actual full/authoritative verifier.
    head = _canonical_controller_input()["instrumentation_head"]
    for mutate in (
        lambda ci: ci.pop("gpu"),
        lambda ci: ci.__setitem__("surprise", True),
        lambda ci: ci.pop("build_tool_binding"),
    ):
        broken = copy.deepcopy(_canonical_controller_input())
        mutate(broken)
        with pytest.raises(
            freeze_verifier.VerificationError, match="missing or unknown"
        ):
            freeze_verifier._verify_controller_input(broken, ROOT, head)


def test_build_tool_binding_omission_is_full_strict_but_discovery_tolerant(
    tmp_path: Path,
) -> None:
    # The deliberate, documented cross-version asymmetry, through the *actual*
    # verifier entry points: the discovery header accepts a historical
    # (build_tool_binding-absent) artifact so enumeration never aborts, while the
    # full/authoritative gate rejects the omission.
    head = _canonical_controller_input()["instrumentation_head"]
    historical = copy.deepcopy(_canonical_controller_input())
    del historical["build_tool_binding"]
    path, _head = _write_landing_candidate(tmp_path / "historical", historical)
    freeze_verifier._verify_landing_candidate_header(path, tmp_path / "historical")
    with pytest.raises(freeze_verifier.VerificationError, match="missing or unknown"):
        freeze_verifier._verify_controller_input(historical, ROOT, head)


def test_actual_discover_controller_input_does_not_reject_binding_member() -> None:
    # End-to-end regression for the Stage-E escape through the controller's real
    # entry. Over the repo's mixed-version evidence tree (the current
    # build_tool_binding-bearing artifact plus the four historical artifacts),
    # _discover_controller_input either selects exactly one current landing and
    # returns the canonical 14-member contract, or fail-closes with the canonical
    # zero/one landing-count error — but it must NEVER again reject a candidate at
    # the independent verifier over the build_tool_binding member.
    try:
        _path, contract = controller._discover_controller_input(ROOT)
    except controller.ContractError as exc:
        message = str(exc)
        assert "rejected by independent verifier" not in message, message
        assert "current-HEAD h0_preseal_freeze_v3 landing" in message, message
    else:
        assert set(contract) == LITERAL_CONTROLLER_INPUT_MEMBERS
        assert "build_tool_binding" in contract


def test_discovery_header_accepts_every_committed_artifact() -> None:
    # The discovery member gate must accept every committed v3 controller-input —
    # the current build_tool_binding-bearing artifact and the four historical
    # artifacts that predate it — so enumeration never aborts the way it did at S.
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
        else:
            saw_without = True
    # The corpus must actually contain both shapes for this to be a real guard.
    assert saw_binding and saw_without
