"""Contract tests for the GCTM runtime-native candidate-universe freeze."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
SCHEMA = TOOLS / "gctm_runtime_universe_schema_v1.json"
VALIDATOR = TOOLS / "validate_gctm_runtime_universe.py"
PACKET = (
    ROOT
    / "docs/modules/semantic/research/evidence"
    / "gctm_runtime_native_candidate_universe_20260724"
)
DECLARATION = PACKET / "universe_declaration.json"
FROZEN = PACKET / "frozen_input_identities.json"
REG_REQ = PACKET / "h0_native_universe_completeness_registration_requirements_v1.json"
FIXTURES = (
    ROOT / "tests/contract/fixtures" / "gctm_runtime_universe_fixture_catalog_v1.json"
)

sys.path.insert(0, TOOLS.as_posix())
import validate_gctm_runtime_universe as universe  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_mutation(record: dict[str, Any], mutation: dict[str, Any]) -> None:
    path = list(mutation["path"])
    target: Any = record
    for part in path[:-1]:
        target = target[part]
    leaf = path[-1]
    if mutation["op"] == "replace":
        target[leaf] = deepcopy(mutation["value"])
    elif mutation["op"] == "remove":
        del target[leaf]
    else:  # pragma: no cover - fixture catalog contract
        raise AssertionError(f"unsupported fixture mutation {mutation['op']!r}")


def _fixture_record(fixture: dict[str, Any]) -> dict[str, Any]:
    record = _load(DECLARATION)
    for mutation in fixture["mutations"]:
        _apply_mutation(record, mutation)
    return record


def test_schema_and_canonical_declaration_validate() -> None:
    schema = _load(SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"] == universe.SCHEMA_ID
    assert schema["additionalProperties"] is False
    record = _load(DECLARATION)
    jsonschema.Draft202012Validator(schema).validate(record)
    result = universe.validate_declaration(
        record, frozen_path=FROZEN, reg_path=REG_REQ, packet_dir=PACKET
    )
    assert result["valid"] is True
    assert result["selected_terminal"] == universe.TERMINAL_SEALABLE
    assert result["fixed_validator_output"] == universe.FIXED_NON_AUTHORITY
    assert result["packet_bindings"]["valid"] is True
    assert result["packet_bindings"]["packet_id"] == universe.PACKET_ID


def test_cli_validates_canonical_declaration() -> None:
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(DECLARATION),
            "--frozen-inputs",
            str(FROZEN),
            "--registration-requirements",
            str(REG_REQ),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["selected_terminal"] == universe.TERMINAL_SEALABLE
    assert report["result"]["packet_bindings"]["valid"] is True
    assert report["authority_verified"] is False
    assert report["runtime_guarantee_established"] is False
    assert report["runtime_compatibility_established"] is False
    assert report["h0_reentry_authorized"] is False
    assert report["b1_activation_eligible"] is False


@pytest.mark.parametrize(
    "fixture",
    _load(FIXTURES)["fixtures"],
    ids=lambda item: item["fixture_id"],
)
def test_fixture_catalog(fixture: dict[str, Any]) -> None:
    # Fixture mutations exercise declaration structure only; packet integrity
    # is validated separately against the committed on-disk packet.
    record = _fixture_record(fixture)
    if fixture["expected_ok"]:
        result = universe.validate_declaration(
            record, frozen_path=FROZEN, reg_path=REG_REQ, packet_dir=None
        )
        assert result["selected_terminal"] == fixture["expected_terminal"]
        assert result["fixed_validator_output"] == universe.FIXED_NON_AUTHORITY
        assert "packet_bindings" not in result
    else:
        with pytest.raises(universe.RuntimeUniverseValidationError):
            universe.validate_declaration(
                record, frozen_path=FROZEN, reg_path=REG_REQ, packet_dir=None
            )


def test_frozen_input_hashes_match_disk() -> None:
    frozen = _load(FROZEN)
    assert frozen["mutable_branch_tip_used"] is False
    for item in frozen["inputs"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert _sha256(path) == item["sha256"], item["path"]


def test_registration_requirements_are_requirements_only() -> None:
    req = _load(REG_REQ)
    assert req["modifies_registration_v2"] is False
    assert req["claims_registration_v3_exists"] is False
    assert req["guarantee_class"] == "universe_completeness"
    bindings = {row["binding"] for row in req["required_bindings"]}
    assert "universe_identity" in bindings
    assert "inclusion_stage_identity" in bindings
    assert "invalidation_set" in bindings


def test_packet_sidecars_exist() -> None:
    for name in (
        "universe_declaration.json",
        "frozen_input_identities.json",
        "event_candidate_identity.json",
        "inclusion_stage_decision.json",
        "composition_completeness_contract.json",
        "h0_native_universe_completeness_registration_requirements_v1.json",
        "terminal_report.json",
        "manifest.json",
    ):
        assert (PACKET / name).is_file(), name


def test_packet_bindings_verify_exact_committed_packet() -> None:
    packet = universe.validate_packet_bindings(PACKET)
    assert packet["valid"] is True
    assert packet["declaration_id"] == universe.DECLARATION_ID
    assert packet["selected_terminal"] == universe.TERMINAL_SEALABLE
    assert packet["fixed_validator_output"] == universe.FIXED_NON_AUTHORITY
    for name in universe.REQUIRED_PACKET_ARTIFACTS:
        assert name in packet["artifacts_checked"]
    for key in universe.REQUIRED_TOOLING_KEYS:
        assert key in packet["tooling_checked"]
    for binding in universe.TERMINAL_BINDING_MAP:
        assert binding in packet["terminal_bindings_checked"]


def test_packet_bindings_fail_on_stale_manifest_artifact_hash(
    tmp_path: Path,
) -> None:
    import shutil

    clone = tmp_path / "packet"
    shutil.copytree(PACKET, clone)
    manifest_path = clone / "manifest.json"
    manifest = _load(manifest_path)
    # Tamper: keep disk content, break declared hash.
    name = "universe_declaration.json"
    manifest["artifacts"][name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(universe.RuntimeUniverseValidationError) as exc:
        universe.validate_packet_bindings(clone)
    assert exc.value.error_class == "packet_artifact_hash_mismatch"


def test_packet_bindings_fail_on_sidecar_content_tamper(tmp_path: Path) -> None:
    import shutil

    clone = tmp_path / "packet"
    shutil.copytree(PACKET, clone)
    target = clone / "event_candidate_identity.json"
    data = _load(target)
    data["tampered"] = True
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(universe.RuntimeUniverseValidationError) as exc:
        universe.validate_packet_bindings(clone)
    assert exc.value.error_class == "packet_artifact_hash_mismatch"


def test_packet_bindings_fail_on_terminal_manifest_binding_drift(
    tmp_path: Path,
) -> None:
    import shutil

    clone = tmp_path / "packet"
    shutil.copytree(PACKET, clone)
    terminal_path = clone / "terminal_report.json"
    terminal = _load(terminal_path)
    # Keep manifest correct vs disk; break terminal_report binding only.
    terminal["artifact_bindings"]["universe_declaration_sha256"] = "1" * 64
    terminal_path.write_text(json.dumps(terminal, indent=2) + "\n", encoding="utf-8")
    # terminal_report itself is hashed in manifest, so this may fail either as
    # terminal hash mismatch or binding mismatch depending on check order.
    with pytest.raises(universe.RuntimeUniverseValidationError) as exc:
        universe.validate_packet_bindings(clone)
    assert exc.value.error_class in {
        "packet_artifact_hash_mismatch",
        "terminal_manifest_binding_mismatch",
        "terminal_binding_disk_mismatch",
    }


def test_packet_bindings_fail_on_identity_cross_check(tmp_path: Path) -> None:
    import shutil

    clone = tmp_path / "packet"
    shutil.copytree(PACKET, clone)
    manifest_path = clone / "manifest.json"
    manifest = _load(manifest_path)
    manifest["runtime_consumer_identity"] = "not_the_runtime_universe"
    # Recompute only is not done: this is intentional drift. But changing
    # manifest content also invalidates nothing about itself (manifest is not
    # self-hashed). Artifact hashes remain correct.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(universe.RuntimeUniverseValidationError) as exc:
        universe.validate_packet_bindings(clone)
    assert exc.value.error_class == "identity_cross_check"


def test_packet_bindings_fail_on_stale_tooling_hash(tmp_path: Path) -> None:
    import shutil

    clone = tmp_path / "packet"
    shutil.copytree(PACKET, clone)
    manifest_path = clone / "manifest.json"
    manifest = _load(manifest_path)
    manifest["tooling"]["schema"]["sha256"] = "2" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(universe.RuntimeUniverseValidationError) as exc:
        universe.validate_packet_bindings(clone)
    assert exc.value.error_class == "packet_tooling_hash_mismatch"


def test_d1_synthetic_universe_not_replaced() -> None:
    record = _load(DECLARATION)
    assert record["ownership"]["is_gctm_d1_candidate_universe"] is False
    assert "closed" in record["ownership"]["d1_status"].lower()
    frozen = _load(FROZEN)
    closed = {item["object"] for item in frozen["read_only_closed_objects"]}
    assert "GCTM_D1_INTERFACE_READY" in closed
    assert "synthetic_event_candidate_set_v1" in closed


def test_inclusion_stage_is_unique_and_score_independent() -> None:
    record = _load(DECLARATION)
    incl = record["inclusion_stage"]
    assert incl["exactly_one_stage_selected"] is True
    assert incl["selected_stage"] == "pre_score_eligible_set"
    assert incl["score_independent"] is True
    for field in (
        "claim_won",
        "commit_executed",
        "best_lost_slot",
        "true_match_label",
    ):
        assert field in incl["forbidden_inclusion_fields"]


def test_positive_and_negative_fixture_coverage() -> None:
    fixtures = _load(FIXTURES)["fixtures"]
    kinds = {item["kind"] for item in fixtures}
    assert "positive" in kinds
    assert "negative_invalid" in kinds
    assert "negative_unsealable" in kinds
    assert any(
        item["expected_terminal"] == universe.TERMINAL_SEALABLE for item in fixtures
    )
    assert any(
        item["expected_terminal"] == universe.TERMINAL_UNSEALABLE for item in fixtures
    )
    assert any(
        item["expected_terminal"] == universe.TERMINAL_INVALID for item in fixtures
    )
