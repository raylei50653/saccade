"""Successor grant/receipt binding and exactly-once consumption contract."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_path_partition as path_partition  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_successor_authorization as authority  # noqa: E402


def _request(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    ledger = tmp_path / "ledger"
    spec = run_spec_module.build_run_spec()
    request = authority.build_request(
        execution_id="successor-measurement-1", run_spec=spec, ledger=ledger
    )
    return request, spec, ledger


def _grant(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "authority": request["authority"],
        "authorization_id": "a" * 64,
        "execution_domain": request["execution_domain"],
        "execution_id": request["execution_id"],
        "execution_semantics_projection_digest": request[
            "execution_semantics_projection_digest"
        ],
        "issued_by": request["requested_issuer"],
        "phase": request["phase"],
        "resolved_run_spec_digest": request["resolved_run_spec_digest"],
        "schema": authority.GRANT_SCHEMA,
    }


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_request_and_grant_bind_the_exact_successor_projection(tmp_path: Path) -> None:
    request, spec, _ = _request(tmp_path)
    grant = _grant(request)
    authority.validate(request, authority.REQUEST_SCHEMA)
    authority.validate(grant, authority.GRANT_SCHEMA)
    assert request["resolved_run_spec_digest"] == spec["resolved_run_spec_digest"]
    assert (
        request["execution_semantics_projection_digest"]
        == spec["execution_semantics_projection_digest"]
    )
    assert authority.grant_mismatch_reasons(request, grant) == ()


def test_grant_mismatch_is_refused_before_consumption(tmp_path: Path) -> None:
    request, _, _ = _request(tmp_path)
    grant = _grant(request)
    grant["execution_semantics_projection_digest"] = "b" * 64
    path = tmp_path / "grant.json"
    _write(path, grant)
    with pytest.raises(authority.AuthorizationError, match="projection"):
        authority.load_grant(path, request)


def test_receipt_is_the_single_durable_consumption_event(tmp_path: Path) -> None:
    request, _, ledger = _request(tmp_path)
    grant = _grant(request)
    path = tmp_path / "grant.json"
    _write(path, grant)
    loaded = authority.load_grant(path, request)
    receipt = authority.consume(
        request=request, grant_path=path, grant=loaded, ledger=ledger
    )
    receipt_path = ledger / f"{grant['authorization_id']}.json"
    assert receipt_path.is_file()
    assert receipt["authorization_digest"] == authority.canonical_digest(grant)
    with pytest.raises(authority.AuthorizationError, match="already consumed"):
        authority.consume(request=request, grant_path=path, grant=loaded, ledger=ledger)


def test_grant_change_after_admission_is_refused(tmp_path: Path) -> None:
    request, _, ledger = _request(tmp_path)
    grant = _grant(request)
    path = tmp_path / "grant.json"
    _write(path, grant)
    loaded = authority.load_grant(path, request)
    changed = dict(grant)
    changed["authorization_id"] = "c" * 64
    _write(path, changed)
    with pytest.raises(authority.AuthorizationError, match="changed|differs"):
        authority.consume(request=request, grant_path=path, grant=loaded, ledger=ledger)
    assert not ledger.exists()


@pytest.mark.parametrize(
    "path",
    [
        "docs/research/contracts/h2_execution_result_v2.json",
        "docs/research/contracts/h2_execution_verification_v2.json",
        "docs/research/contracts/h2_successor_authorization_request_v1.json",
        "docs/research/contracts/h2_successor_exactly_once_authorization_v1.json",
        "docs/research/contracts/h2_successor_authorization_consumed_v1.json",
        "docs/research/contracts/h2_successor_measurement_envelope_verification_v1.json",
        "docs/research/contracts/h2_successor_measurement_rehearsal_witness_v1.json",
        "scripts/tools/h2_successor_authorization.py",
    ],
)
def test_successor_authority_contracts_are_identity_semantics(path: str) -> None:
    assert path_partition.classify(path) == "identity_semantics"
