#!/usr/bin/env python3
"""Walk the v2 successor measurement path with a disposable-ledger grant."""
# status: diagnostic

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_h2_measure_archives as corpus  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_successor_authorization as authority  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import run_h2_successor_measurement as controller  # noqa: E402
import verify_h2_execution as inner_verifier  # noqa: E402
import verify_h2_measurement_envelope as envelope_verifier  # noqa: E402
import verify_h2_measurement as legacy_verifier  # noqa: E402

WITNESS_SCHEMA = authority.REHEARSAL_WITNESS_SCHEMA


class RehearsalError(RuntimeError):
    """The disposable rehearsal failed to exercise its promised boundary."""


def synthetic_grant(request: dict[str, Any]) -> dict[str, Any]:
    grant = {
        "authority": request["authority"],
        "authorization_id": secrets.token_hex(32),
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
    authority.validate(grant, authority.GRANT_SCHEMA)
    return grant


def run(
    *,
    execution_id: str,
    packet: Path,
    ledger: Path,
    selected_base: str,
    build_dir: Path | None,
) -> dict[str, Any]:
    request, _ = controller.request_for(execution_id=execution_id, ledger=ledger)
    grant = synthetic_grant(request)
    grant_path = packet.with_name(f".{packet.name}.rehearsal-grant.json")
    if grant_path.exists() or grant_path.is_symlink():
        raise RehearsalError(f"rehearsal grant path already exists: {grant_path}")
    grant_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_document_exclusive(grant_path.parent, grant_path.name, grant)

    incomplete, result = controller.execute(
        execution_id=execution_id,
        grant_path=grant_path,
        packet=packet,
        selected_base=selected_base,
        build_dir=build_dir,
        ledger=ledger,
        require_controlled_domain=False,
    )
    _, inner = inner_verifier.commit_verification(incomplete / authority.ARCHIVE_DIR)
    _, envelope = envelope_verifier.commit_verification(incomplete, packet)
    reasons: list[str] = []
    try:
        corpus.check_corpus([packet])
    except (
        corpus.CorpusError,
        evidence.EvidenceError,
        partition.PartitionError,
        legacy_verifier.VerificationError,
        OSError,
    ) as exc:
        reasons = [str(exc)]
    if not reasons:
        raise RehearsalError("disposable-ledger rehearsal entered the canonical corpus")
    if (
        len(reasons) != 1
        or "controlled authorization execution domain" not in reasons[0]
    ):
        raise RehearsalError(f"rehearsal refusal was not domain-only: {reasons}")

    witness = {
        "corpus_admission": {"admitted": False, "reasons": reasons},
        "envelope_valid": envelope.get("valid") is True,
        "execution_id": execution_id,
        "grant_digest": authority.canonical_digest(grant),
        "inner_valid": inner.get("valid") is True,
        "packet": packet.resolve().as_posix(),
        "receipt_digest": result.get("authorization_binding_digest"),
        "result": result.get("result"),
        "schema": WITNESS_SCHEMA,
        "terminal": result.get("terminal"),
    }
    if not witness["inner_valid"] or not witness["envelope_valid"]:
        raise RehearsalError("rehearsal archive or envelope is invalid")
    if witness["result"] != "measurement_pass" or witness["terminal"] is not None:
        raise RehearsalError(
            "rehearsal did not reach the successful measurement disposition"
        )
    authority.validate(witness, WITNESS_SCHEMA)
    witness_path = packet.with_name(f"{packet.name}.rehearsal_witness.json")
    evidence.write_document_exclusive(witness_path.parent, witness_path.name, witness)
    return witness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--disposable-ledger", type=Path, required=True)
    parser.add_argument("--selected-base", required=True)
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        witness = run(
            execution_id=args.execution_id,
            packet=args.packet,
            ledger=args.disposable_ledger,
            selected_base=args.selected_base,
            build_dir=args.build_dir,
        )
    except (
        RehearsalError,
        controller.MeasurementControllerError,
        authority.AuthorizationError,
        inner_verifier.ExecutionVerificationError,
        envelope_verifier.EnvelopeVerificationError,
        evidence.EvidenceError,
        OSError,
    ) as exc:
        print(f"successor measurement rehearsal failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(witness, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
