#!/usr/bin/env python3
"""Successor exactly-once authorization request, grant and receipt contract.

The successor retired legacy head/freeze/F/S gates.  Its grant therefore binds
the complete resolved RunSpec, the execution-semantics projection, one execution
identifier and one execution domain.  The durable ledger receipt is the single
consumption event; its canonical digest is what v2 ``result.json`` records.

This module validates and writes authority evidence but never issues a grant.
Only externally supplied canonical grant bytes naming the repository's owner
issuer can pass ``load_grant``.
"""
# status: stable

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_measurement_evidence as evidence  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402

CONTRACT_DIR = REPO_ROOT / "docs" / "research" / "contracts"
CONTROLLED_DOMAIN_PATH = CONTRACT_DIR / "h2_controlled_host_execution_domain_v1.json"

REQUEST_NAME = "authorization_request.json"
GRANT_NAME = "authorization_grant.json"
DOMAIN_NAME = "authorization_execution_domain.json"
RECEIPT_NAME = "authorization_consumed.json"
ENVELOPE_VERIFICATION_NAME = "envelope_verification.json"
CHECKSUMS_NAME = evidence.CHECKSUMS_NAME
ARCHIVE_DIR = "archive"
RUNS_DIR = "runs"

REQUEST_SCHEMA = "h2_successor_authorization_request_v1"
GRANT_SCHEMA = "h2_successor_exactly_once_authorization_v1"
RECEIPT_SCHEMA = "h2_successor_authorization_consumed_v1"
ENVELOPE_VERIFICATION_SCHEMA = "h2_successor_measurement_envelope_verification_v1"
REHEARSAL_WITNESS_SCHEMA = "h2_successor_measurement_rehearsal_witness_v1"
RESULT_SCHEMA_V2 = "h2_execution_result_v2"
VERIFICATION_SCHEMA_V2 = "h2_execution_verification_v2"

SCHEMA_PATHS: dict[str, Path] = {
    REQUEST_SCHEMA: CONTRACT_DIR / "h2_successor_authorization_request_v1.json",
    GRANT_SCHEMA: CONTRACT_DIR / "h2_successor_exactly_once_authorization_v1.json",
    RECEIPT_SCHEMA: CONTRACT_DIR / "h2_successor_authorization_consumed_v1.json",
    ENVELOPE_VERIFICATION_SCHEMA: CONTRACT_DIR
    / "h2_successor_measurement_envelope_verification_v1.json",
    REHEARSAL_WITNESS_SCHEMA: CONTRACT_DIR
    / "h2_successor_measurement_rehearsal_witness_v1.json",
}


class AuthorizationError(RuntimeError):
    """The successor authority chain is absent, malformed or mismatched."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_digest(document: Mapping[str, Any]) -> str:
    return evidence.digest(document)


def _load_schema(schema: str) -> dict[str, Any]:
    try:
        document = json.loads(SCHEMA_PATHS[schema].read_text(encoding="utf-8"))
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationError(
            f"authorization schema is unavailable: {schema}"
        ) from exc
    if not isinstance(document, dict):
        raise AuthorizationError(f"authorization schema is not an object: {schema}")
    return document


def validate(document: Mapping[str, Any], schema: str) -> None:
    import jsonschema

    try:
        jsonschema.validate(instance=document, schema=_load_schema(schema))
    except jsonschema.ValidationError as exc:
        raise AuthorizationError(f"document violates {schema}: {exc.message}") from exc
    except jsonschema.SchemaError as exc:  # pragma: no cover - frozen contract
        raise AuthorizationError(f"invalid authorization schema: {schema}") from exc


def load_canonical(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuthorizationError(f"authorization input is not a physical file: {path}")
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationError(
            f"authorization input is unreadable JSON: {path}"
        ) from exc
    if not isinstance(document, dict):
        raise AuthorizationError(f"authorization input is not an object: {path}")
    if raw != evidence.canonical_json_bytes(document) + b"\n":
        raise AuthorizationError(f"authorization input is not canonical JSON: {path}")
    validate(document, schema)
    return document


def controlled_domain() -> dict[str, Any]:
    try:
        document = evidence.load_document(
            CONTROLLED_DOMAIN_PATH.parent,
            CONTROLLED_DOMAIN_PATH.name,
            schema=evidence.AUTHORIZATION_DOMAIN_SCHEMA,
        )
    except evidence.EvidenceError as exc:
        raise AuthorizationError(
            "controlled execution-domain anchor is invalid"
        ) from exc
    return document


def controlled_ledger() -> Path:
    value = controlled_domain().get("ledger_root")
    if not isinstance(value, str):
        raise AuthorizationError("controlled execution domain has no ledger root")
    path = Path(value)
    if not path.is_absolute():
        raise AuthorizationError("controlled authorization ledger is not absolute")
    return path


def live_domain(ledger: Path) -> dict[str, Any]:
    try:
        return evidence.authorization_execution_domain(ledger)
    except evidence.EvidenceError as exc:
        raise AuthorizationError(str(exc)) from exc


def build_request(
    *,
    execution_id: str,
    run_spec: Mapping[str, Any],
    ledger: Path | None = None,
) -> dict[str, Any]:
    selected_ledger = controlled_ledger() if ledger is None else ledger
    document = {
        "authority": partition.MEASUREMENT_AUTHORITY,
        "execution_domain": canonical_digest(live_domain(selected_ledger)),
        "execution_id": execution_id,
        "execution_semantics_projection_digest": run_spec.get(
            "execution_semantics_projection_digest"
        ),
        "phase": "a",
        "requested_issuer": evidence.AUTHORIZATION_ISSUER,
        "resolved_run_spec_digest": run_spec.get("resolved_run_spec_digest"),
        "schema": REQUEST_SCHEMA,
    }
    validate(document, REQUEST_SCHEMA)
    return document


def grant_mismatch_reasons(
    request: Mapping[str, Any], grant: Mapping[str, Any]
) -> tuple[str, ...]:
    fields = (
        "authority",
        "execution_domain",
        "execution_id",
        "execution_semantics_projection_digest",
        "phase",
        "resolved_run_spec_digest",
    )
    reasons = [
        f"grant {name} differs from the authorization request"
        for name in fields
        if grant.get(name) != request.get(name)
    ]
    if grant.get("issued_by") != request.get("requested_issuer"):
        reasons.append("grant issuer differs from the requested owner")
    return tuple(reasons)


def load_grant(path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    validate(request, REQUEST_SCHEMA)
    grant = load_canonical(path, GRANT_SCHEMA)
    reasons = grant_mismatch_reasons(request, grant)
    if reasons:
        raise AuthorizationError("; ".join(reasons))
    return grant


def receipt_for(
    request: Mapping[str, Any],
    grant: Mapping[str, Any],
    *,
    consumed_utc: str | None = None,
) -> dict[str, Any]:
    reasons = grant_mismatch_reasons(request, grant)
    if reasons:
        raise AuthorizationError("; ".join(reasons))
    document = {
        "authority": request["authority"],
        "authorization_digest": canonical_digest(grant),
        "authorization_id": grant["authorization_id"],
        "consumed_utc": consumed_utc or _utc(),
        "execution_domain": request["execution_domain"],
        "execution_id": request["execution_id"],
        "execution_semantics_projection_digest": request[
            "execution_semantics_projection_digest"
        ],
        "phase": request["phase"],
        "resolved_run_spec_digest": request["resolved_run_spec_digest"],
        "schema": RECEIPT_SCHEMA,
        "state": "consumed",
    }
    validate(document, RECEIPT_SCHEMA)
    return document


def consume(
    *,
    request: Mapping[str, Any],
    grant_path: Path,
    grant: Mapping[str, Any],
    ledger: Path,
) -> dict[str, Any]:
    """Durably consume one grant; exclusive receipt creation is linearization."""
    current = load_grant(grant_path, request)
    if current != grant:
        raise AuthorizationError("authorization grant changed before consumption")
    observed_domain = live_domain(ledger)
    if canonical_digest(observed_domain) != request.get("execution_domain"):
        raise AuthorizationError(
            "authorization execution domain changed before consumption"
        )
    if ledger.is_symlink():
        raise AuthorizationError("authorization ledger root is a symlink")
    ledger.mkdir(parents=True, exist_ok=True)
    receipt = receipt_for(request, grant)
    try:
        evidence.write_document_exclusive(
            ledger, f"{grant['authorization_id']}.json", receipt
        )
    except FileExistsError as exc:
        raise AuthorizationError(
            "exactly-once authorization was already consumed"
        ) from exc
    return receipt


def write_request(path: Path, request: Mapping[str, Any]) -> Path:
    validate(request, REQUEST_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    return evidence.write_document_exclusive(path.parent, path.name, request)
