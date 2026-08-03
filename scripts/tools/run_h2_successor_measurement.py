#!/usr/bin/env python3
"""Consume one successor grant and produce one v2 measurement archive.

The production CLI has no ledger override.  It recomputes the exact request,
validates an externally owner-issued grant, starts the bound-input monitor,
durably consumes the grant in the controlled ledger, then and only then enters
the existing successor producer.  It writes no independent verdict and closes
no checksum inventory.
"""
# status: stable

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_measurement_evidence as evidence  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_successor_authorization as authority  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import run_h2_execution as producer  # noqa: E402


class MeasurementControllerError(RuntimeError):
    """A pre-consumption gate or the one consumed execution failed."""


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_exact_clean_head(selected_base: str) -> None:
    head = _git("rev-parse", "HEAD")
    if selected_base != head:
        raise MeasurementControllerError(
            f"selected base {selected_base!r} is not current HEAD {head}"
        )
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise MeasurementControllerError(
            "formal successor measurement requires a clean committed checkout"
        )


def _assert_no_symlink_ancestor(path: Path, label: str) -> None:
    current = path.absolute()
    while True:
        if current.exists() and current.is_symlink():
            raise MeasurementControllerError(f"{label} crosses a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent


def _assert_external(path: Path, label: str, forbidden: tuple[Path, ...]) -> Path:
    _assert_no_symlink_ancestor(path, label)
    resolved = path.resolve(strict=False)
    for root in forbidden:
        blocked = root.resolve(strict=False)
        if resolved == blocked or resolved.is_relative_to(blocked):
            raise MeasurementControllerError(
                f"{label} is inside forbidden root {blocked}"
            )
    return resolved


def request_for(
    *, execution_id: str, ledger: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = run_spec_module.build_run_spec()
    request = authority.build_request(
        execution_id=execution_id, run_spec=spec, ledger=ledger
    )
    return request, spec


def emit_request(
    *, execution_id: str, destination: Path, selected_base: str
) -> dict[str, Any]:
    assert_exact_clean_head(selected_base)
    destination = _assert_external(
        destination,
        "authorization request",
        (REPO_ROOT, authority.controlled_ledger()),
    )
    request, _ = request_for(execution_id=execution_id)
    authority.write_request(destination, request)
    return request


def _write_packet_record(root: Path, name: str, document: Mapping[str, Any]) -> None:
    evidence.write_document_exclusive(root, name, document)


def execute(
    *,
    execution_id: str,
    grant_path: Path,
    packet: Path,
    selected_base: str,
    build_dir: Path | None,
    ledger: Path,
    require_controlled_domain: bool,
) -> tuple[Path, dict[str, Any]]:
    """Run the one authority-consuming producer path; used by CLI and rehearsal."""
    assert_exact_clean_head(selected_base)
    if require_controlled_domain and ledger.resolve(
        strict=False
    ) != authority.controlled_ledger().resolve(strict=False):
        raise MeasurementControllerError(
            "production execution did not select the controlled ledger"
        )

    forbidden = (REPO_ROOT, ledger)
    final = _assert_external(packet, "measurement packet", forbidden)
    incomplete = Path(f"{final}.incomplete")
    _assert_external(incomplete, "incomplete measurement packet", forbidden)
    grant_path = _assert_external(grant_path, "owner grant", (REPO_ROOT, ledger))
    if (
        final.exists()
        or final.is_symlink()
        or incomplete.exists()
        or incomplete.is_symlink()
    ):
        raise MeasurementControllerError("measurement packet target already exists")

    selected_build = (build_dir or REPO_ROOT / "build" / "h2_layer_p").resolve(
        strict=False
    )
    if not selected_build.is_dir():
        raise MeasurementControllerError(
            "formal measurement requires an already-built Layer-P directory"
        )

    request, spec = request_for(execution_id=execution_id, ledger=ledger)
    grant = authority.load_grant(grant_path, request)
    domain = authority.live_domain(ledger)

    incomplete.mkdir(parents=True)
    _write_packet_record(incomplete, authority.REQUEST_NAME, request)
    _write_packet_record(incomplete, authority.GRANT_NAME, grant)
    _write_packet_record(incomplete, authority.DOMAIN_NAME, domain)
    run_root = incomplete / authority.RUNS_DIR
    run_root.mkdir()

    execution = producer.configured_execution(
        execution_id=execution_id,
        authority=partition.MEASUREMENT_AUTHORITY,
        authorization_binding_digest=None,
        run_root=run_root,
        selected_base=selected_base,
        build_dir=selected_build,
        skip_build=True,
        run_spec=spec,
        result_schema=authority.RESULT_SCHEMA_V2,
    )
    try:
        receipt = authority.consume(
            request=request,
            grant_path=grant_path,
            grant=grant,
            ledger=ledger,
        )
    except BaseException:
        abandon = getattr(execution.stages, "abandon", None)
        if callable(abandon):
            abandon()
        raise

    try:
        _write_packet_record(incomplete, authority.RECEIPT_NAME, receipt)
        execution.authorization_binding_digest = authority.canonical_digest(receipt)
        result = execution.produce(incomplete / authority.ARCHIVE_DIR)
    except BaseException:
        abandon = getattr(execution.stages, "abandon", None)
        if callable(abandon):
            abandon()
        raise
    return incomplete, result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--selected-base", required=True)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--emit-request", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.emit_request is not None:
            if args.authorization is not None or args.packet is not None:
                raise MeasurementControllerError(
                    "--emit-request cannot be combined with execution arguments"
                )
            document = emit_request(
                execution_id=args.execution_id,
                destination=args.emit_request,
                selected_base=args.selected_base,
            )
            print(json.dumps(document, indent=2, sort_keys=True))
            return 0
        if args.authorization is None or args.packet is None:
            raise MeasurementControllerError(
                "--authorization and --packet are required to execute"
            )
        incomplete, result = execute(
            execution_id=args.execution_id,
            grant_path=args.authorization,
            packet=args.packet,
            selected_base=args.selected_base,
            build_dir=args.build_dir,
            ledger=authority.controlled_ledger(),
            require_controlled_domain=True,
        )
    except (
        MeasurementControllerError,
        authority.AuthorizationError,
        evidence.EvidenceError,
        producer.ProducerError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"successor measurement refused or stopped: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "authority": result["authority"],
                "incomplete_packet": incomplete.as_posix(),
                "result": result["result"],
                "terminal": result["terminal"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
