#!/usr/bin/env python3
"""Walk the whole H2 Phase-A launch path without spending the owner's grant.

Source review, unit tests over synthetic environments and a green launch probe
have each failed, twice, to predict an execution-time structural self-negation:
`0a5dffe9` died at terminal 1 with 0/4 runs started, `7646f421` at terminal 4
with 1/4. Both authorizations are permanently spent and neither produced a
single captured decision. The gate those two attempts earned is a real run —
controller, child process, eval-stack import, environment validation, capture
initialisation, every ordered run — that consumes no *owner* authorization and
writes no evidence into the repository.

`run_h2_measurement.py` has one path: `--authorization` is required and the
ledger is the default one. It may not grow a rehearsal mode — branching before
admission would not exercise admission, and branching after it while skipping
consumption would change the production authorization invariant. So walking
admission at all requires *an* authorization, and "spends nothing" means the
owner's grant is untouched, not that no authorization artifact exists.

This harness therefore issues its own grant against its own disposable ledger
and runs the unmodified controller. It modifies no production file; the only
seams it uses, `evidence_parent` and `authorization_ledger`, already exist.

**The grant is owner-shaped but is not an owner issuance.** Admission requires
`issued_by == AUTHORIZATION_ISSUER`, so the record this harness writes names the
research owner in bytes. It is a credential the existing contract forces into
this shape, nothing more. What actually separates it from a real grant is its
execution domain: the domain binds the ledger root, so a rehearsal grant is
arithmetically unusable against the owner ledger, and the archive it produces is
refused by `check_h2_measure_archives`, which admits only attempts consumed
under the controlled host's ledger.

Its output can never stand in for `Acceptance` items 4-5, `F`, `S`, or any
capture. It proves the path runs; it proves nothing about behavioural identity.

Threat model for the path guards below. They refuse lexical aliases, symlink
aliases, ancestor aliases and accidental reuse *as they exist at launch*, and
they detect a destination substituted afterwards. They do not resist a hostile
concurrent process running as the same user: the controller writes through
pathnames, not directory handles, so a rename or mount between checks cannot be
prevented here — only caught. Making it impossible needs `openat`-relative I/O
throughout production, or a mount namespace, and both are outside this item.

Usage:
  .venv/bin/python scripts/tools/rehearse_h2_measurement.py \
      --freeze F.json --layer-p-certificate C.json --reference-probe P.json \
      --runtime-inputs R.json --published-identity I.json \
      --disposable-ledger /abs/path/ledger --evidence-parent /abs/path/evidence
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_measurement_evidence as evidence  # noqa: E402
import run_h2_measurement as controller  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402

WITNESS_SCHEMA = "h2_phase_a_rehearsal_witness_v1"
WITNESS_NAME = "rehearsal_witness.json"
WITNESS_AUTHORITY = "non_evidence_rehearsal"

# A rehearsal that reached a terminal is a *result*: the harness worked and the
# launch path did not. An invariant violation is the harness itself failing to
# hold what it promised, and the two must not be read as the same outcome.
REHEARSAL_TERMINAL = "rehearsal_terminal"
HARNESS_INVARIANT_VIOLATED = "harness_invariant_violated"


class RehearsalRefused(RuntimeError):
    """Raised before a safe witness destination exists. Nothing is recorded."""


class HarnessInvariantViolated(RuntimeError):
    """Raised after the witness exists. Recorded as a failed rehearsal."""


class DirectoryIdentity(NamedTuple):
    """What a directory pathname resolved to when the harness created it."""

    path: Path
    device: int
    inode: int

    def revalidate(self, stage: str) -> None:
        """Re-resolve the pathname the controller writes through.

        Holding a descriptor would only prove *that descriptor* still points at
        the same inode. The controller resolves pathnames, so the pathname is
        what has to be re-checked, at every point where the harness would
        otherwise be assuming continuity.
        """
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise HarnessInvariantViolated(
                f"{stage}: {self.path} is no longer readable ({exc})"
            ) from exc
        # One verdict for every way the pathname could have been repointed — a
        # symlink dropped in its place, a different directory renamed onto it.
        if not stat.S_ISDIR(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != (self.device, self.inode):
            raise HarnessInvariantViolated(
                f"{stage}: {self.path} now resolves to a different directory"
            )


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nearest_existing(path: Path) -> Path:
    """The deepest ancestor that exists, for a destination not yet created."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor)


def _real(path: Path) -> Path:
    """Resolve every symlink that exists today, without requiring the leaf."""
    existing = _nearest_existing(path)
    resolved = Path(os.path.realpath(existing))
    return resolved.joinpath(*path.parts[len(existing.parts) :])


def _contains(ancestor: Path, descendant: Path) -> bool:
    """Component containment, never string prefix: /repo does not hold /repo-x."""
    return descendant == ancestor or descendant.parts[: len(ancestor.parts)] == (
        ancestor.parts
    )


def _protected_locations() -> dict[str, Path]:
    """Everything a rehearsal output may neither be nor live inside."""
    return {
        "the repository": _real(REPO_ROOT),
        "the owner authorization ledger": _real(
            controller.default_authorization_ledger()
        ),
    }


def resolve_output_destination(label: str, raw: Path) -> Path:
    """Judge one requested output path, before anything is created.

    Refusal here is a refusal to start: the destination is what a witness would
    be written to, so an unsafe destination has nowhere to be recorded.
    """
    if not raw.is_absolute():
        raise RehearsalRefused(f"{label} is not an absolute path: {raw}")
    # Containment is judged before existence: a path that is an alias of a
    # protected location is also, usually, a path that already exists, and
    # "it already exists" would be the less informative of the two answers.
    resolved = _real(raw)
    for name, protected in _protected_locations().items():
        if _contains(protected, resolved) or _contains(resolved, protected):
            raise RehearsalRefused(
                f"{label} resolves to {resolved}, which shares a location with "
                f"{name} ({protected})"
            )
    if raw.exists() or raw.is_symlink():
        raise RehearsalRefused(
            f"{label} already exists: {raw}. A rehearsal writes into a destination "
            "it created, so reuse is refused rather than emptied"
        )
    return resolved


def create_output_destination(resolved: Path) -> DirectoryIdentity:
    """Create the destination exclusively and record what it resolved to."""
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(resolved, mode=0o700)
    except OSError as exc:
        raise RehearsalRefused(f"cannot create {resolved}: {exc}") from exc
    info = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or Path(os.path.realpath(resolved)) != resolved:
        raise RehearsalRefused(f"{resolved} is not the directory that was created")
    return DirectoryIdentity(resolved, info.st_dev, info.st_ino)


def synthetic_grant(
    bundle: controller.LaunchBundle, *, ledger: Path, destination: Path
) -> tuple[Path, str]:
    """Write one grant this harness issues against its own ledger.

    Every contract value is read from its authority at call time. A copy of any
    of them here would be a second answer to what a grant is, drifting silently
    from the one the controller and the verifier enforce (§ C3.9).
    """
    authorization_id = secrets.token_hex(32)
    invocation_id = secrets.token_hex(32)
    grant = {
        "schema": evidence.AUTHORIZATION_GRANT_SCHEMA,
        "authorization_id": authorization_id,
        "capture_phase": controller.CAPTURE_PHASE,
        "controller_digest": bundle.freeze["executed_surfaces"][
            "scripts/tools/run_h2_measurement.py"
        ],
        "execution_domain": evidence.digest(
            evidence.authorization_execution_domain(ledger)
        ),
        "freeze_digest": evidence.freeze_digest(bundle.freeze),
        "instrumentation_head": bundle.head,
        "invocation_id": invocation_id,
        "issued_by": evidence.AUTHORIZATION_ISSUER,
    }
    if set(grant) != evidence.AUTHORIZATION_GRANT_MEMBERS:
        raise HarnessInvariantViolated("synthetic grant member drift")
    path = evidence.write_document(
        destination, evidence.AUTHORIZATION_GRANT_NAME, grant
    )
    return path, invocation_id


def ordered_run_summary(root: Path) -> list[dict[str, Any]]:
    """Project the run inventory out of the archive, never out of a counter.

    A count the harness keeps is a second copy of a fact the evidence already
    carries, and it is the copy that would be believed when they disagree.
    """
    record = evidence.load_document(
        root, evidence.CONTROLLER_NAME, schema=evidence.CONTROLLER_SCHEMA
    )
    sequence = record["sequence"]
    return [
        {
            "run_id": run_id,
            "ordinal": ordinal,
            "recorded": run_id in record.get("ordered_runs", []),
            "present": evidence.run_dir(root, sequence, run_id).is_dir(),
        }
        for ordinal, run_id in enumerate(evidence.RUN_IDS)
    ]


def rehearsal_invariant_failures(
    root: Path,
    *,
    selection: Any,
    report: Mapping[str, Any],
    grant_record: Mapping[str, Any],
    ledger: DirectoryIdentity,
) -> tuple[str, ...]:
    """Everything that must hold for a rehearsal to have demonstrated anything.

    A terminal of `None` alone is not success: the controller can finish while
    the archive fails verification, while the receipt is missing or written
    somewhere else, or while runs the plan names never started.
    """
    failures: list[str] = []
    if selection.terminal is not None:
        failures.append(f"controller reached terminal {selection.terminal}")
    if report.get("valid") is not True:
        failures.append("the independent verifier refused the rehearsal archive")

    receipts = sorted(path.name for path in ledger.path.iterdir())
    expected = f"{grant_record['authorization_id']}.json"
    if receipts != [expected]:
        failures.append(
            f"disposable ledger holds {receipts}, not exactly the one receipt "
            f"{expected}"
        )
    else:
        receipt = evidence.load_document(
            ledger.path, expected, schema=evidence.AUTHORIZATION_SCHEMA
        )
        for member in ("authorization_id", "invocation_id"):
            if receipt.get(member) != grant_record[member]:
                failures.append(f"receipt {member} is not the synthetic grant's")
        if receipt.get("authorization_digest") != evidence.digest(grant_record):
            failures.append("receipt does not bind the synthetic grant's digest")

    for run in ordered_run_summary(root):
        if not (run["recorded"] and run["present"]):
            failures.append(f"ordered run {run['run_id']} did not complete")

    hygiene = controller.checkout_hygiene_reasons()
    if hygiene:
        failures.append(f"the rehearsal left the checkout dirty: {'; '.join(hygiene)}")
    return tuple(failures)


def _write_witness(destination: Path, document: Mapping[str, Any]) -> None:
    """Replace the witness atomically so it is never half-written."""
    staging = destination.parent / (destination.name + ".replacing")
    evidence.write_document(staging.parent, staging.name, document)
    os.replace(staging, destination)


def rehearse(
    *,
    bundle: controller.LaunchBundle,
    ledger: DirectoryIdentity,
    evidence_parent: DirectoryIdentity,
    witness_path: Path,
    started_utc: str,
) -> tuple[dict[str, Any], int]:
    """Run the production controller once against the disposable ledger."""
    pre_hygiene = tuple(controller.checkout_hygiene_reasons())
    witness: dict[str, Any] = {
        "schema": WITNESS_SCHEMA,
        "authority": WITNESS_AUTHORITY,
        "status": "started",
        "started_utc": started_utc,
        "finished_utc": None,
        "failure_class": None,
        "failures": [],
        "source_head": bundle.head,
        "freeze_digest": evidence.freeze_digest(bundle.freeze),
        "harness_digest": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "disposable_ledger": {
            "path": ledger.path.as_posix(),
            "device": ledger.device,
            "inode": ledger.inode,
        },
        "evidence_parent": evidence_parent.path.as_posix(),
        "evidence_root": None,
        "evidence_root_digest": None,
        "authorization_id": None,
        "invocation_id": None,
        "grant_digest": None,
        "receipt": None,
        "receipt_digest": None,
        "verifier_report_digest": None,
        "ordered_runs": [],
        "controller_terminal": None,
        "controller_result": None,
        "verifier_report": None,
        "checkout_hygiene_before": list(pre_hygiene),
        "checkout_hygiene_after": None,
    }
    # Exclusive: a rehearsal that crashed must never be indistinguishable from
    # one that was never marked, and must never quietly overwrite an earlier one.
    evidence.write_document_exclusive(witness_path.parent, witness_path.name, witness)

    try:
        # The harness owns every conjunct of its own success. The controller
        # refuses a dirty checkout too, but a contract that leans on someone
        # else re-deciding it is a contract with a hole where that seam moves.
        if pre_hygiene:
            raise HarnessInvariantViolated(
                f"the checkout was dirty before rehearsal: {'; '.join(pre_hygiene)}"
            )
        ledger.revalidate("before admission")
        evidence_parent.revalidate("before admission")
        grant_path, invocation_id = synthetic_grant(
            bundle, ledger=ledger.path, destination=evidence_parent.path
        )
        grant = controller.load_authorization(
            grant_path,
            bundle,
            invocation_id=invocation_id,
            authorization_ledger=ledger.path,
        )
        witness["authorization_id"] = grant.record["authorization_id"]
        witness["invocation_id"] = grant.record["invocation_id"]
        witness["grant_digest"] = grant.digest

        root, selection = controller.execute_controller(
            bundle,
            authorization=grant,
            evidence_parent=evidence_parent.path,
            authorization_ledger=ledger.path,
        )
        ledger.revalidate("after execution")
        evidence_parent.revalidate("after execution")

        report = verifier.VERIFIERS[verifier.classify(root)](root)
        receipt_name = f"{grant.record['authorization_id']}.json"
        witness["evidence_root"] = root.as_posix()
        # The identity `F_B` itself binds, not a second one invented here: a
        # witness carrying only pathnames could say that *some* archive was
        # verified, never that the archive later registered is that one.
        witness["evidence_root_digest"] = evidence.checksum_inventory_digest(root)
        witness["controller_terminal"] = selection.terminal
        witness["controller_result"] = selection.result
        witness["verifier_report"] = dict(report)
        witness["verifier_report_digest"] = evidence.digest(dict(report))
        witness["ordered_runs"] = ordered_run_summary(root)
        receipt_path = ledger.path / receipt_name
        witness["receipt"] = receipt_path.as_posix()
        # Conditional so a missing receipt reaches the conjunction below and is
        # reported as the named failure it is, rather than dying here unread.
        if receipt_path.is_file():
            witness["receipt_digest"] = evidence.digest(
                evidence.load_document(
                    ledger.path, receipt_name, schema=evidence.AUTHORIZATION_SCHEMA
                )
            )

        failures = rehearsal_invariant_failures(
            root,
            selection=selection,
            report=report,
            grant_record=grant.record,
            ledger=ledger,
        )
        witness["checkout_hygiene_after"] = list(controller.checkout_hygiene_reasons())
        witness["failures"] = list(failures)
        if failures:
            witness["status"] = "failed"
            witness["failure_class"] = (
                REHEARSAL_TERMINAL
                if selection.terminal is not None
                else HARNESS_INVARIANT_VIOLATED
            )
            exit_code = 1
        else:
            witness["status"] = "completed"
            exit_code = 0
    except BaseException as exc:  # recorded, then re-raised by the caller's report
        witness["status"] = "failed"
        witness["failure_class"] = HARNESS_INVARIANT_VIOLATED
        witness["failures"] = [f"{type(exc).__name__}: {exc}"]
        witness["finished_utc"] = _utc()
        _write_witness(witness_path, witness)
        raise

    ledger.revalidate("before the witness is finalised")
    evidence_parent.revalidate("before the witness is finalised")
    witness["finished_utc"] = _utc()
    _write_witness(witness_path, witness)
    return witness, exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--layer-p-certificate", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--runtime-inputs", type=Path, required=True)
    parser.add_argument("--published-identity", type=Path, required=True)
    parser.add_argument(
        "--disposable-ledger",
        type=Path,
        required=True,
        help="absolute path this harness creates; never the owner ledger",
    )
    parser.add_argument(
        "--evidence-parent",
        type=Path,
        required=True,
        help="absolute path this harness creates, outside the repository",
    )
    # No --authorization and no --invocation-id: a rehearsal issues its own
    # grant, so there is no argument through which the owner's could be spent.
    args = parser.parse_args(list(argv) if argv is not None else None)

    created: list[Path] = []
    try:
        ledger_target = resolve_output_destination(
            "--disposable-ledger", args.disposable_ledger
        )
        evidence_target = resolve_output_destination(
            "--evidence-parent", args.evidence_parent
        )
        if _contains(ledger_target, evidence_target) or _contains(
            evidence_target, ledger_target
        ):
            raise RehearsalRefused(
                f"--disposable-ledger ({ledger_target}) and --evidence-parent "
                f"({evidence_target}) resolve to one location"
            )
        bundle = controller.load_bundle(
            freeze_path=args.freeze,
            certificate_path=args.layer_p_certificate,
            reference_probe_path=args.reference_probe,
            runtime_manifest_path=args.runtime_inputs,
            published_identity_path=args.published_identity,
        )
        ledger = create_output_destination(ledger_target)
        created.append(ledger.path)
        evidence_parent = create_output_destination(evidence_target)
        created.append(evidence_parent.path)
    except (
        RehearsalRefused,
        controller.ControllerError,
        evidence.EvidenceError,
        OSError,
    ) as exc:
        for path in reversed(created):
            shutil.rmtree(path, ignore_errors=True)
        print(f"H2 Phase-A rehearsal refused: {exc}", file=sys.stderr)
        return 2

    witness_path = evidence_parent.path / WITNESS_NAME
    try:
        witness, exit_code = rehearse(
            bundle=bundle,
            ledger=ledger,
            evidence_parent=evidence_parent,
            witness_path=witness_path,
            started_utc=_utc(),
        )
    except Exception as exc:
        print(f"H2 Phase-A rehearsal failed: {exc}", file=sys.stderr)
        print(f"witness: {witness_path}", file=sys.stderr)
        return 1

    print(f"witness: {witness_path}")
    print(f"evidence: {witness['evidence_root']}")
    print(f"status: {witness['status']} ({witness['failure_class'] or 'no failure'})")
    for failure in witness["failures"]:
        print(f"  - {failure}", file=sys.stderr)
    print(
        "This rehearsal spends no owner authorization, seals nothing, and its "
        "archive is refused by the canonical corpus."
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
