#!/usr/bin/env python3
"""Independently verify and close one successor measurement authority envelope."""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_execution_driver as driver  # noqa: E402
import h2_import_witness as import_witness  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_successor_authorization as authority  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_execution as inner_verifier  # noqa: E402

VERIFICATION_PROCESS = "independent_command_separate_process"
CHECKS = (
    "artifact_schemas",
    "authorization_binding",
    "checksum_closure",
    "execution_domain",
    "inner_archive",
    "packet_layout",
    "run_evidence",
)
TOP_RECORDS = (
    authority.REQUEST_NAME,
    authority.GRANT_NAME,
    authority.DOMAIN_NAME,
    authority.RECEIPT_NAME,
)


class EnvelopeVerificationError(RuntimeError):
    """The packet cannot support an envelope-verification record."""


def _physical_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise EnvelopeVerificationError(f"{label} is not a physical directory: {path}")


def _packet_layout_reasons(root: Path) -> list[str]:
    reasons: list[str] = []
    try:
        _physical_directory(root, "measurement packet")
    except EnvelopeVerificationError as exc:
        return [str(exc)]
    permitted = {
        *TOP_RECORDS,
        authority.ARCHIVE_DIR,
        authority.RUNS_DIR,
        authority.ENVELOPE_VERIFICATION_NAME,
        authority.CHECKSUMS_NAME,
    }
    present = {item.name for item in root.iterdir()}
    required = set(TOP_RECORDS) | {authority.ARCHIVE_DIR, authority.RUNS_DIR}
    missing = sorted(required - present)
    unexpected = sorted(present - permitted)
    if missing:
        reasons.append(f"measurement packet is missing required members: {missing}")
    if unexpected:
        reasons.append(f"measurement packet holds unexpected members: {unexpected}")
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode()):
        if path.is_symlink():
            reasons.append(
                f"measurement packet contains a symlink: {path.relative_to(root)}"
            )
    return reasons


def _load_json(root: Path, name: str) -> tuple[dict[str, Any], bytes]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise EnvelopeVerificationError(f"packet record is not a physical file: {name}")
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvelopeVerificationError(
            f"packet record is unreadable JSON: {name}"
        ) from exc
    if not isinstance(document, dict):
        raise EnvelopeVerificationError(f"packet record is not an object: {name}")
    return document, raw


def _load_records(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    documents: dict[str, dict[str, Any]] = {}
    raws: dict[str, bytes] = {}
    for name in TOP_RECORDS:
        documents[name], raws[name] = _load_json(root, name)
    return documents, raws


def _artifact_schema_reasons(
    documents: Mapping[str, Mapping[str, Any]], raws: Mapping[str, bytes]
) -> list[str]:
    reasons: list[str] = []
    schemas = {
        authority.REQUEST_NAME: authority.REQUEST_SCHEMA,
        authority.GRANT_NAME: authority.GRANT_SCHEMA,
        authority.RECEIPT_NAME: authority.RECEIPT_SCHEMA,
    }
    for name, schema in schemas.items():
        try:
            authority.validate(documents[name], schema)
        except authority.AuthorizationError as exc:
            reasons.append(str(exc))
        expected = evidence.canonical_json_bytes(documents[name]) + b"\n"
        if raws[name] != expected:
            reasons.append(f"{name} is not canonical JSON")
    domain = documents[authority.DOMAIN_NAME]
    if set(domain) != evidence.AUTHORIZATION_DOMAIN_MEMBERS:
        reasons.append("authorization execution-domain member set is invalid")
    if domain.get("schema") != evidence.AUTHORIZATION_DOMAIN_SCHEMA:
        reasons.append("authorization execution-domain schema is invalid")
    if raws[authority.DOMAIN_NAME] != evidence.canonical_json_bytes(domain) + b"\n":
        reasons.append(f"{authority.DOMAIN_NAME} is not canonical JSON")
    return reasons


def _authorization_reasons(
    documents: Mapping[str, Mapping[str, Any]], inner: Mapping[str, Any]
) -> list[str]:
    request = documents[authority.REQUEST_NAME]
    grant = documents[authority.GRANT_NAME]
    receipt = documents[authority.RECEIPT_NAME]
    reasons = list(authority.grant_mismatch_reasons(request, grant))
    try:
        expected = authority.receipt_for(
            request, grant, consumed_utc=str(receipt.get("consumed_utc"))
        )
    except authority.AuthorizationError as exc:
        reasons.append(str(exc))
    else:
        if dict(receipt) != expected:
            reasons.append(
                "authorization receipt is not the receipt the grant produces"
            )
    result = inner.get("result.json")
    if not isinstance(result, Mapping):
        reasons.append("inner result.json is not an object")
        return reasons
    if result.get("schema") != authority.RESULT_SCHEMA_V2:
        reasons.append("measurement envelope does not contain a v2 result")
    if result.get("authority") != partition.MEASUREMENT_AUTHORITY:
        reasons.append("inner result does not record measurement authority")
    if result.get("authorization_binding_digest") != authority.canonical_digest(
        receipt
    ):
        reasons.append("inner result does not bind the durable consumption receipt")
    for name in (
        "execution_id",
        "resolved_run_spec_digest",
        "execution_semantics_projection_digest",
    ):
        if result.get(name) != receipt.get(name):
            reasons.append(f"inner result {name} differs from the receipt")
    return reasons


def _domain_reasons(documents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    domain = documents[authority.DOMAIN_NAME]
    expected = authority.canonical_digest(domain)
    reasons = []
    for name in (authority.REQUEST_NAME, authority.GRANT_NAME, authority.RECEIPT_NAME):
        if documents[name].get("execution_domain") != expected:
            reasons.append(f"{name} does not bind the archived execution domain")
    return reasons


def _inner_reasons(root: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    archive = root / authority.ARCHIVE_DIR
    try:
        documents, _ = inner_verifier.load_archive(archive)
        verification = inner_verifier.verify_archive(archive)
        inner_verifier.validate_verification(verification)
    except (
        inner_verifier.ExecutionVerificationError,
        evidence.EvidenceError,
        OSError,
    ) as exc:
        return [f"inner archive verification cannot be formed: {exc}"], {}, {}
    reasons: list[str] = []
    if verification.get("schema") != authority.VERIFICATION_SCHEMA_V2:
        reasons.append("inner archive does not carry a v2 verification record")
    if verification.get("valid") is not True:
        reasons.append(
            f"inner archive is not independently valid: {verification.get('reasons')}"
        )
    return reasons, documents, verification


def _import_witness_reasons(
    directory: Path,
    *,
    run_id: str,
    run_spec: Mapping[str, Any],
) -> list[str]:
    import jsonschema

    reasons: list[str] = []
    try:
        witness = evidence.load_document(
            directory,
            import_witness.WITNESS_NAME,
            schema=import_witness.WITNESS_SCHEMA,
        )
        schema = inner_verifier._load_contract(
            inner_verifier._SCHEMA_BY_BASENAME["h2_import_witness_v1.json"]
        )
        jsonschema.validate(instance=witness, schema=schema)
        import_witness.validate_witness(witness)
        reasons.extend(
            import_witness.containment_failures(
                witness["observations"], document=run_spec
            )
        )
    except (
        KeyError,
        TypeError,
        evidence.EvidenceError,
        import_witness.WitnessError,
        jsonschema.ValidationError,
        OSError,
    ) as exc:
        return [f"{run_id} import witness is unusable: {exc}"]

    projection = run_spec.get("execution_semantics_projection")
    closure = (
        projection.get("execution_code_closure")
        if isinstance(projection, Mapping)
        else None
    )
    if not isinstance(closure, Mapping):
        return [f"{run_id} import witness has no usable RunSpec declaration"]
    expected_declared = {
        "execution_code_closure_digest": closure.get("digest"),
        "execution_semantics_projection_digest": projection.get("digest"),
        "roots": closure.get("roots"),
    }
    if witness.get("declared") != expected_declared:
        reasons.append(f"{run_id} import witness binds a different declaration")
    return reasons


def _run_reasons(
    root: Path,
    result: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> list[str]:
    ordered = result.get("ordered_runs")
    if not isinstance(ordered, list):
        return ["inner result ordered_runs is not a list"]
    reasons: list[str] = []
    by_id = {run.get("run_id"): run for run in ordered if isinstance(run, Mapping)}
    for run_id in driver.producer.run_ids():
        record = by_id.get(run_id)
        if not isinstance(record, Mapping):
            reasons.append(f"run evidence has no result record for {run_id}")
            continue
        if record.get("state") != "completed":
            continue
        try:
            observed = driver.run_artifact_digest(root, run_id)
        except (driver.DriverError, OSError) as exc:
            reasons.append(f"{run_id} run evidence is unusable: {exc}")
            continue
        if record.get("artifact_digest") != observed:
            reasons.append(f"{run_id} artifact digest differs from its run evidence")
        directory = evidence.run_dir(root, driver.producer.sequence(), run_id)
        reasons.extend(
            _import_witness_reasons(
                directory,
                run_id=run_id,
                run_spec=run_spec,
            )
        )

    predicates = result.get("predicate_results")
    if not isinstance(predicates, Mapping):
        reasons.append("inner result has no usable predicate record for run replay")
        return reasons
    try:
        replay = driver.layer_m.replay_surviving_evidence(root)
    except (driver.layer_m.ControllerError, OSError, KeyError, TypeError) as exc:
        reasons.append(f"run evidence replay cannot be formed: {exc}")
        return reasons

    def state(name: str) -> Any:
        record = predicates.get(name)
        return record.get("state") if isinstance(record, Mapping) else None

    if replay.evidence_present:
        for name, observed in (
            ("capture_off_on_equal", replay.capture_equal),
            ("packets_valid", replay.packets_valid),
        ):
            expected = "pass" if observed else "fail"
            if state(name) != expected:
                reasons.append(
                    f"{name} replays to {expected!r}, and result.json records "
                    f"{state(name)!r}"
                )
    elif state("capture_off_on_equal") == "pass" or state("packets_valid") == "pass":
        reasons.append("run-derived predicates pass without replayable run evidence")

    all_completed = all(
        isinstance(record, Mapping) and record.get("state") == "completed"
        for record in by_id.values()
    ) and set(by_id) == set(driver.producer.run_ids())
    if state("execution_complete") == "pass" and (
        not replay.complete or replay.errors or not all_completed
    ):
        reasons.append(
            "execution_complete records pass, but archived run evidence does not "
            f"replay complete: errors={list(replay.errors)!r}"
        )
    return reasons


def _checksum_reasons(root: Path) -> list[str]:
    verdict = root / authority.ENVELOPE_VERIFICATION_NAME
    inventory = root / authority.CHECKSUMS_NAME
    if verdict.is_file() != inventory.is_file():
        return ["measurement envelope is half closed"]
    if not inventory.is_file():
        return []
    try:
        evidence.verify_checksum_inventory(root)
    except evidence.EvidenceError as exc:
        return [f"packet checksum closure is invalid: {exc}"]
    return []


def _packet_identity(request: Mapping[str, Any]) -> tuple[str, str, str]:
    fields = (
        "execution_id",
        "resolved_run_spec_digest",
        "execution_semantics_projection_digest",
    )
    values: list[str] = []
    for name in fields:
        value = request.get(name)
        if not isinstance(value, str):
            raise EnvelopeVerificationError(
                f"authorization request cannot identify the packet: {name}"
            )
        values.append(value)
    return values[0], values[1], values[2]


def _record(
    documents: Mapping[str, Mapping[str, Any]],
    raws: Mapping[str, bytes],
    archive_root: Path,
    reasons_by_check: Mapping[str, list[str]],
) -> dict[str, Any]:
    request = documents[authority.REQUEST_NAME]
    execution_id, run_spec_digest, projection_digest = _packet_identity(request)
    checks = {name: not reasons_by_check[name] for name in CHECKS}
    archive_inventory = Path(authority.ARCHIVE_DIR) / authority.CHECKSUMS_NAME
    return {
        "artifact_digests": {
            authority.REQUEST_NAME: hashlib.sha256(
                raws[authority.REQUEST_NAME]
            ).hexdigest(),
            authority.GRANT_NAME: hashlib.sha256(
                raws[authority.GRANT_NAME]
            ).hexdigest(),
            authority.DOMAIN_NAME: hashlib.sha256(
                raws[authority.DOMAIN_NAME]
            ).hexdigest(),
            authority.RECEIPT_NAME: hashlib.sha256(
                raws[authority.RECEIPT_NAME]
            ).hexdigest(),
            archive_inventory.as_posix(): evidence.sha256_file(
                archive_root / authority.CHECKSUMS_NAME
            ),
        },
        "checks": checks,
        "execution_id": execution_id,
        "execution_semantics_projection_digest": projection_digest,
        "producer_invoked": False,
        "reasons": [reason for name in CHECKS for reason in reasons_by_check[name]],
        "resolved_run_spec_digest": run_spec_digest,
        "schema": authority.ENVELOPE_VERIFICATION_SCHEMA,
        "valid": all(checks.values()),
        "verification_host_inputs_used": False,
        "verification_process": VERIFICATION_PROCESS,
    }


def validate_verification(document: Mapping[str, Any]) -> None:
    authority.validate(document, authority.ENVELOPE_VERIFICATION_SCHEMA)


def verify_packet(root: Path) -> dict[str, Any]:
    layout = _packet_layout_reasons(root)
    if layout and any("missing required" in reason for reason in layout):
        raise EnvelopeVerificationError("; ".join(layout))
    documents, raws = _load_records(root)
    inner_reasons, inner_documents, _inner_verification = _inner_reasons(root)
    if not inner_documents:
        raise EnvelopeVerificationError("; ".join(inner_reasons))
    result = inner_documents.get("result.json")
    if not isinstance(result, Mapping):
        raise EnvelopeVerificationError("inner result cannot identify the envelope")
    archive_root = root / authority.ARCHIVE_DIR
    run_spec = inner_documents.get("run_spec.json")
    if not isinstance(run_spec, Mapping):
        raise EnvelopeVerificationError("inner RunSpec cannot identify run authority")
    reasons_by_check = {
        "artifact_schemas": _artifact_schema_reasons(documents, raws),
        "authorization_binding": _authorization_reasons(documents, inner_documents),
        "checksum_closure": _checksum_reasons(root),
        "execution_domain": _domain_reasons(documents),
        "inner_archive": inner_reasons,
        "packet_layout": layout,
        "run_evidence": _run_reasons(root, result, run_spec),
    }
    expected = _record(documents, raws, archive_root, reasons_by_check)
    stored_path = root / authority.ENVELOPE_VERIFICATION_NAME
    if stored_path.is_file():
        stored, _ = _load_json(root, authority.ENVELOPE_VERIFICATION_NAME)
        try:
            validate_verification(stored)
        except authority.AuthorizationError as exc:
            reasons_by_check["checksum_closure"].append(
                f"stored envelope verdict is invalid: {exc}"
            )
        if stored != expected:
            reasons_by_check["checksum_closure"].append(
                "stored envelope verdict differs from the independently recomputed verdict"
            )
    return _record(documents, raws, archive_root, reasons_by_check)


def commit_verification(
    root: Path, final: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    document = verify_packet(root)
    validate_verification(document)
    path = evidence.write_document_exclusive(
        root, authority.ENVELOPE_VERIFICATION_NAME, document
    )
    evidence.write_checksum_inventory(root)
    closed = verify_packet(root)
    if closed != document or closed.get("valid") is not True:
        raise EnvelopeVerificationError(
            f"closed envelope does not reproduce its verdict: {closed.get('reasons')}"
        )
    if final is not None:
        if final.exists() or final.is_symlink():
            raise EnvelopeVerificationError(
                f"final packet path already exists: {final}"
            )
        os.replace(root, final)
        evidence._fsync_directory(final.parent)
        path = final / authority.ENVELOPE_VERIFICATION_NAME
    return path, document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--final", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.final is not None and not args.commit:
            raise EnvelopeVerificationError("--final requires --commit")
        if args.commit:
            _, document = commit_verification(args.packet, args.final)
        else:
            document = verify_packet(args.packet)
            validate_verification(document)
    except (
        EnvelopeVerificationError,
        authority.AuthorizationError,
        evidence.EvidenceError,
        OSError,
    ) as exc:
        print(f"envelope verification could not be formed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
