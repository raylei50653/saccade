#!/usr/bin/env python3
"""Verify one closed H2 successor execution archive from its bytes alone.

Review Correction 5 splits the successor path in two: the producer emits
`run_spec.json`, `runtime_binding.json` and `result.json` and must not write
`verification.json`, and a separate command in a separate process reads only the
emitted artifacts and their checksum closure and writes the verdict. This module
is that separate command.

**It composes rules, it does not hold any.** Every semantic name here is
imported: the artifact shapes from the four frozen schemas, the RunSpec's
internal consistency from the resolver, the ordered verdict and the
cross-artifact algebra from `h2_terminal_partition`. That is why the file is
`plumbing_only` and moves no axis (§ C3.9) — a verifier that restated the
verdict algebra would be a second answer to the same question, and the archive
it accepted could drift from the one the ruler describes.

The composition is exactly:

    validate(run_spec.json, runtime_binding.json, result.json against the frozen
             schemas)
  ∧ recompute the selection from `predicate_results`
  ∧ the recorded result and terminal equal the recomputed ones
  ∧ `binding_agreement_reasons(...) == ()`
  ∧ the digest and execution-identity bindings agree

`binding_agreement_reasons` receives the **recomputed** `selection.terminal`.
Handing it the terminal the archive recorded would make the check circular: the
recorded verdict is one of the things being checked.

What "archive-only" means here, precisely. The verdict depends on two byte
sources: the archive, and this repository's frozen contract schemas — which are
this command's own versioned definition of what it is checking, not an
observation of the verifying host. It never reads the host's machine identity,
UID, environment, build outputs or working tree, never re-derives missing
evidence, and never invokes the producer. The one place the ruler *does* read
the local tree is `h2_run_spec.execution_semantics_projection()`, which hashes
the declared content set from the checkout; this module therefore calls
`validate_run_spec(..., verify_projection=False)`. That flag is the whole line
between an execution-integrity check and the environment reproducibility
Correction 5 retired, and `tests/contract/test_h2_execution_verifier.py` proves
the line holds by making that function raise and verifying a valid archive
anyway.

Two failure classes, drawn by two rules together. A defect *inside* a formable
archive — a schema violation, a malformed member, a digest that does not match, a
verdict that disagrees with the observation — is a recorded `valid: false` with
reasons, because that is a verdict about the archive. An archive is unformable
when either rule refuses it: `h2_execution_verification_v1` requires an execution
id and three artifact digests, so a missing or unreadable artifact leaves the
record's own required fields unfillable; and admission requires a physical flat
root, so a symlink, a subdirectory or a non-regular file is refused before any
schema is consulted. Either way this command writes nothing and exits non-zero.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
from h2_runtime_inputs import canonical_json_bytes, sha256_file  # noqa: E402

CONTRACT_DIR = REPO_ROOT / "docs" / "research" / "contracts"
VERIFICATION_NAME = "verification.json"
CHECKSUMS_NAME = evidence.CHECKSUMS_NAME
VERIFICATION_SCHEMA = "h2_execution_verification_v1"
VERIFICATION_PROCESS = "independent_command_separate_process"

# Which archive file carries which frozen contract. The schema paths come from
# the ruler's own tuple rather than a second list here; only the pairing with an
# archive filename is new information, and `test_h2_execution_verifier.py` pins
# it against the verification schema's required digest members so a renamed
# artifact cannot quietly stop being verified.
_SCHEMA_BY_BASENAME = {
    Path(relative).name: relative
    for relative in path_partition.EXECUTION_ARTIFACT_SCHEMA_PATHS
}
PRODUCER_ARTIFACTS: dict[str, str] = {
    "run_spec.json": _SCHEMA_BY_BASENAME["h2_phase_a_run_spec_v1.json"],
    "runtime_binding.json": _SCHEMA_BY_BASENAME["h2_runtime_binding_v1.json"],
    "result.json": _SCHEMA_BY_BASENAME["h2_execution_result_v1.json"],
}
VERIFICATION_CONTRACT = _SCHEMA_BY_BASENAME["h2_execution_verification_v1.json"]

# `h2_execution_result_v1` is the Phase-A contract: its run plan is the frozen
# four-run Phase-A sequence, and `select_successor_result` refuses any other
# phase. Nothing here selects a phase; it reads the one the contract fixes.
PHASE = "a"

_TRACKING_EXTENSION_ROLE = "tracking_extension"


class ExecutionVerificationError(RuntimeError):
    """The archive cannot support a verification record at all."""


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON token: {token}")


def _load_contract(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExecutionVerificationError(
            f"frozen contract is unreadable: {relative}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExecutionVerificationError(
            f"frozen contract is not an object: {relative}"
        )
    return payload


def _archive_files(root: Path) -> dict[str, Path]:
    """Every regular file directly in the archive root, refusing symlinks.

    An archive that can point outside itself is not a closed record, and a
    nested directory would let bytes exist that the closure never names.
    """
    if root.is_symlink() or not root.is_dir():
        raise ExecutionVerificationError(
            f"execution archive root is not a physical directory: {root}"
        )
    found: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name.encode()):
        if path.is_symlink():
            raise ExecutionVerificationError(
                f"execution archive contains a symlink: {path.name}"
            )
        if path.is_dir():
            raise ExecutionVerificationError(
                f"execution archive contains a subdirectory: {path.name}"
            )
        if not path.is_file():
            raise ExecutionVerificationError(
                f"execution archive contains a non-regular file: {path.name}"
            )
        found[path.name] = path
    return found


def _load_artifact(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExecutionVerificationError(
            f"execution artifact is unreadable JSON: {name}"
        ) from exc
    if not isinstance(document, dict):
        raise ExecutionVerificationError(f"execution artifact is not an object: {name}")
    return document, raw


def load_archive(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Read the three producer artifacts, or refuse to form a verdict."""
    files = _archive_files(root)
    missing = sorted(set(PRODUCER_ARTIFACTS) - set(files))
    if missing:
        raise ExecutionVerificationError(
            f"execution archive is missing producer artifacts: {missing}"
        )
    documents: dict[str, dict[str, Any]] = {}
    raws: dict[str, bytes] = {}
    for name in sorted(PRODUCER_ARTIFACTS):
        documents[name], raws[name] = _load_artifact(files[name], name)
    return documents, raws


def _identity(documents: Mapping[str, Mapping[str, Any]]) -> tuple[str, str, str]:
    """The three scalars `h2_execution_verification_v1` requires to exist."""
    result = documents["result.json"]
    run_spec = documents["run_spec.json"]
    execution_id = result.get("execution_id")
    spec_digest = run_spec.get("resolved_run_spec_digest")
    projection_digest = run_spec.get("execution_semantics_projection_digest")
    for label, value in (
        ("execution id", execution_id),
        ("resolved RunSpec digest", spec_digest),
        ("execution-semantics projection digest", projection_digest),
    ):
        if not isinstance(value, str) or not value:
            raise ExecutionVerificationError(
                f"execution archive records no {label}, so no verification record "
                "can be formed"
            )
    assert isinstance(execution_id, str)
    assert isinstance(spec_digest, str)
    assert isinstance(projection_digest, str)
    return execution_id, spec_digest, projection_digest


# -- the six checks -------------------------------------------------------- #


def _check_artifact_schemas(
    documents: Mapping[str, Mapping[str, Any]], raws: Mapping[str, bytes]
) -> list[str]:
    """Each artifact validates against its frozen schema, in its own byte domain."""
    import jsonschema

    reasons: list[str] = []
    for name, relative in sorted(PRODUCER_ARTIFACTS.items()):
        schema = _load_contract(relative)
        try:
            jsonschema.validate(instance=documents[name], schema=schema)
        except jsonschema.ValidationError as exc:
            reasons.append(f"{name} violates {Path(relative).name}: {exc.message}")
        except jsonschema.SchemaError as exc:  # pragma: no cover - frozen contract
            raise ExecutionVerificationError(
                f"frozen contract is not a valid schema: {relative}"
            ) from exc

    # Correction 7's two byte domains. The RunSpec is the artifact that declares
    # both, so it is the artifact whose serialization is checkable: object bytes
    # carry no trailing LF, the file adds exactly one.
    run_spec = documents["run_spec.json"]
    if run_spec.get("artifact_serialization") == run_spec_module.ARTIFACT_SERIALIZATION:
        try:
            expected = canonical_json_bytes(run_spec) + b"\n"
        except (TypeError, ValueError):
            reasons.append("run_spec.json is not finite canonical JSON")
        else:
            if raws["run_spec.json"] != expected:
                reasons.append(
                    "run_spec.json bytes are not the declared artifact serialization "
                    "(canonical object bytes followed by exactly one LF)"
                )
    return reasons


CHECKSUM_CLOSURE = "checksum_closure"

# The record members a stored verdict must still agree on when it is re-read.
# `valid` and `reasons` are excluded on purpose, and so is the closure check
# itself: all three are functions of every check, including this one, so
# comparing them here would make the check its own subject.
_CLOSURE_INDEPENDENT_MEMBERS = (
    "artifact_digests",
    "execution_id",
    "execution_semantics_projection_digest",
    "producer_invoked",
    "resolved_run_spec_digest",
    "schema",
    "verification_host_inputs_used",
    "verification_process",
)


def _closure_independent(record: Mapping[str, Any]) -> dict[str, Any]:
    checks = record.get("checks")
    return {
        "checks": {
            name: value
            for name, value in (checks if isinstance(checks, Mapping) else {}).items()
            if name != CHECKSUM_CLOSURE
        },
        **{member: record.get(member) for member in _CLOSURE_INDEPENDENT_MEMBERS},
    }


def _check_checksum_closure(
    root: Path, raws: Mapping[str, bytes], *, core: Mapping[str, Any]
) -> list[str]:
    """The archive holds these bytes and nothing else, and says the same thing twice.

    Three states are closed, and only three. Before this command runs, the
    archive is the three producer artifacts and neither closing record exists.
    After it runs, both exist and the inventory is total both ways over the same
    bytes. Anything between them is a half-committed archive — a verdict with no
    closure, or a closure with no verdict — and a crash between the two writes
    must not leave a record that verifies while `O_EXCL` refuses to redo it.

    When a verdict is already stored it is not merely present, it is compared:
    re-deriving *a* verdict from the producer artifacts says nothing about
    whether the verdict this archive carries is that one. Rewriting
    `verification.json` and regenerating the inventory would otherwise pass.
    """
    reasons: list[str] = []
    files = _archive_files(root)
    permitted = {*PRODUCER_ARTIFACTS, VERIFICATION_NAME, CHECKSUMS_NAME}
    unexpected = sorted(set(files) - permitted)
    if unexpected:
        reasons.append(
            f"execution archive holds files outside the closure: {unexpected}"
        )
    for name in sorted(PRODUCER_ARTIFACTS):
        if sha256_file(files[name]) != hashlib.sha256(raws[name]).hexdigest():
            # Unreachable while the file is stable; a moving archive is not one.
            reasons.append(f"{name} changed while it was being verified")

    has_verdict = VERIFICATION_NAME in files
    has_inventory = CHECKSUMS_NAME in files
    if has_verdict != has_inventory:
        present, absent = (
            (VERIFICATION_NAME, CHECKSUMS_NAME)
            if has_verdict
            else (CHECKSUMS_NAME, VERIFICATION_NAME)
        )
        reasons.append(
            f"the archive is half closed: it holds {present} without {absent}, so "
            "the verdict and its closure do not describe the same archive"
        )

    if has_verdict:
        try:
            stored, _ = _load_artifact(files[VERIFICATION_NAME], VERIFICATION_NAME)
        except ExecutionVerificationError as exc:
            reasons.append(f"the stored verdict is unreadable: {exc}")
        else:
            if _closure_independent(stored) != _closure_independent(core):
                reasons.append(
                    "the stored verdict is not the verdict these artifacts produce"
                )

    if has_inventory:
        try:
            inventory = evidence.read_checksum_inventory(root)
        except evidence.EvidenceError as exc:
            reasons.append(f"checksum inventory is unusable: {exc}")
        else:
            present_digests = {
                name: sha256_file(path)
                for name, path in files.items()
                if name != CHECKSUMS_NAME
            }
            for missing in sorted(set(present_digests) - set(inventory)):
                reasons.append(f"checksum inventory does not name {missing}")
            for absent_name in sorted(set(inventory) - set(present_digests)):
                reasons.append(
                    f"checksum inventory names an absent file: {absent_name}"
                )
            for name in sorted(set(present_digests) & set(inventory)):
                if present_digests[name] != inventory[name]:
                    reasons.append(f"{name} differs from the checksum inventory")
    return reasons


def _check_run_spec_binding(documents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The RunSpec is internally whole, and the other two name that RunSpec."""
    reasons: list[str] = []
    run_spec = documents["run_spec.json"]
    try:
        # `verify_projection=False` is the archive-only boundary: the projection
        # members are checked against the recorded RunSpec, never re-hashed from
        # the verifying host's checkout.
        run_spec_module.validate_run_spec(run_spec, verify_projection=False)
    except run_spec_module.RunSpecError as exc:
        reasons.append(f"run_spec.json is not a valid RunSpec: {exc}")
    except (TypeError, ValueError) as exc:  # a shape the schema check already named
        reasons.append(f"run_spec.json cannot be validated: {exc}")

    spec_digest = run_spec.get("resolved_run_spec_digest")
    for name in ("runtime_binding.json", "result.json"):
        recorded = documents[name].get("resolved_run_spec_digest")
        if recorded != spec_digest:
            reasons.append(
                f"{name} names RunSpec {recorded!r}, and the archive carries "
                f"{spec_digest!r}"
            )
    return reasons


def _projection_members(run_spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    projection = run_spec.get("execution_semantics_projection")
    members = projection.get("members") if isinstance(projection, Mapping) else None
    if not isinstance(members, list):
        return {}
    return {
        str(member["path"]): member
        for member in members
        if isinstance(member, Mapping) and isinstance(member.get("path"), str)
    }


def _check_projection_binding(documents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The bytes the execution used are the bytes the projection declares.

    Digest equality across the three artifacts says they agree on a number. This
    also asks the stronger question Correction 5 states: every executed surface
    and the capture ABI the binding recorded must be the same bytes, at the same
    path, that the declared content set names. A binding that ran different
    source bytes under the same projection digest is exactly what the
    execution-integrity requirement forbids.
    """
    reasons: list[str] = []
    run_spec = documents["run_spec.json"]
    expected = run_spec.get("execution_semantics_projection_digest")
    for name in ("runtime_binding.json", "result.json"):
        recorded = documents[name].get("execution_semantics_projection_digest")
        if recorded != expected:
            reasons.append(
                f"{name} names projection {recorded!r}, and the RunSpec declares "
                f"{expected!r}"
            )

    declared = _projection_members(run_spec)
    binding = documents["runtime_binding.json"]
    observed: list[Mapping[str, Any]] = []
    surfaces = binding.get("executed_surfaces")
    if isinstance(surfaces, list):
        observed.extend(member for member in surfaces if isinstance(member, Mapping))
    capture_abi = binding.get("capture_abi")
    if isinstance(capture_abi, Mapping):
        observed.append(capture_abi)
    for member in observed:
        path = member.get("path")
        if not isinstance(path, str):
            continue
        declared_member = declared.get(path)
        if declared_member is None:
            reasons.append(
                f"runtime_binding.json executed {path}, which the declared "
                "execution-semantics content set does not name"
            )
            continue
        if member.get("sha256") != declared_member.get("sha256") or member.get(
            "length"
        ) != declared_member.get("length"):
            reasons.append(
                f"runtime_binding.json executed different bytes of {path} than the "
                "declared content set"
            )
    return reasons


def _check_execution_binding(documents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """One execution: one id, and the bytes that loaded are the bytes that were built."""
    reasons: list[str] = []
    binding = documents["runtime_binding.json"]
    result = documents["result.json"]
    if binding.get("execution_id") != result.get("execution_id"):
        reasons.append(
            f"runtime_binding.json records execution {binding.get('execution_id')!r} "
            f"and result.json records {result.get('execution_id')!r}"
        )

    artifacts = binding.get("build_artifacts")
    built: dict[str, Any] = {}
    if isinstance(artifacts, list):
        built = {
            str(item.get("role")): item.get("sha256")
            for item in artifacts
            if isinstance(item, Mapping)
        }
    extension = built.get(_TRACKING_EXTENSION_ROLE)

    loaded = binding.get("extension_load")
    if isinstance(loaded, Mapping):
        if extension is None:
            reasons.append(
                "runtime_binding.json loaded an extension without recording the "
                "build artifact it came from"
            )
        elif loaded.get("sha256") != extension:
            reasons.append(
                "the extension bytes that loaded are not the extension bytes this "
                "execution built"
            )

    probe = binding.get("identity_probe")
    if isinstance(probe, Mapping):
        if extension is None:
            reasons.append(
                "runtime_binding.json recorded an identity probe without the build "
                "artifact it observed"
            )
        elif probe.get("build_artifact_digest") != extension:
            reasons.append(
                "the identity probe observed a build artifact this execution did "
                "not build"
            )
    return reasons


def _check_result_binding(documents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Recompute the verdict from the observation, then ask the ruler if it fits.

    The recorded `result` is passed to the selector only where the contract lets
    an archive name something the predicates cannot: terminal 4's cause. Every
    other recorded verdict is an answer being checked, not an input.

    The container shapes are checked here, at the plumbing boundary, before the
    ruler is asked anything. The ruler is total over *observations* — it names
    every unknown predicate state and every unknown run state — but a JSON
    document that reached this far is only guaranteed to be an object, and an
    archive whose `predicate_results` is a list or whose `ordered_runs` is a
    string is a schema violation, which is a verdict this command must record
    rather than a shape it may hand onward. Widening the ruler's tolerance
    instead would put a fail-closed rule in the file that holds none.
    """
    reasons: list[str] = []
    binding = documents["runtime_binding.json"]
    result = documents["result.json"]

    predicates = result.get("predicate_results")
    if not isinstance(predicates, Mapping):
        return ["result.json records predicate_results that are not an object"]

    recorded = result.get("result")
    named = (
        recorded
        if partition.RESULT_TO_TERMINAL.get(recorded)
        == partition.EXECUTION_INVALID_TERMINAL
        else None
    )
    try:
        selection = partition.select_successor_result(
            predicates,
            authority=result.get("authority"),
            phase=PHASE,
            execution_result=named,
        )
    except partition.PartitionError as exc:
        return [f"the recorded observation selects no result: {exc}"]

    if recorded != selection.result:
        reasons.append(
            f"result.json records result {recorded!r}, and its own predicates select "
            f"{selection.result!r}"
        )
    if result.get("terminal") != selection.terminal:
        reasons.append(
            f"result.json records terminal {result.get('terminal')!r}, and its own "
            f"predicates select {selection.terminal!r}"
        )

    monitor = binding.get("input_monitor")
    runs = result.get("ordered_runs")
    if not isinstance(monitor, Mapping):
        reasons.append(
            "runtime_binding.json records an input_monitor that is not an object"
        )
        return reasons
    # `list`, not `Sequence`: a string is a Sequence whose members are characters,
    # so the looser test admits exactly the archive this guard exists to refuse.
    if not isinstance(runs, list) or any(not isinstance(run, Mapping) for run in runs):
        reasons.append(
            "result.json records ordered_runs that are not a list of objects"
        )
        return reasons
    try:
        reasons.extend(
            partition.binding_agreement_reasons(
                selection.result,
                authority=result.get("authority"),
                selected_terminal=selection.terminal,
                failed_stage=binding.get("failed_stage"),
                input_monitor=monitor,
                ordered_runs=runs,
                identity_probe_present=isinstance(
                    binding.get("identity_probe"), Mapping
                ),
            )
        )
    except partition.PartitionError as exc:
        reasons.append(f"the archive cannot be checked against the ruler: {exc}")
    return reasons


CHECKS = (
    "artifact_schemas",
    "checksum_closure",
    "execution_binding",
    "projection_binding",
    "result_binding",
    "run_spec_binding",
)


def verify_archive(root: Path) -> dict[str, Any]:
    """Build the verification record for one archive. Writes nothing.

    The five artifact checks run first because the closure check compares an
    already-stored verdict against this one, and it can only do that once this
    one exists. Its own field is excluded from that comparison, so the ordering
    is a dependency, not a cycle.
    """
    documents, raws = load_archive(root)
    execution_id, spec_digest, projection_digest = _identity(documents)

    reasons_by_check: dict[str, list[str]] = {
        "artifact_schemas": _check_artifact_schemas(documents, raws),
        "execution_binding": _check_execution_binding(documents),
        "projection_binding": _check_projection_binding(documents),
        "result_binding": _check_result_binding(documents),
        "run_spec_binding": _check_run_spec_binding(documents),
    }
    core: dict[str, Any] = {
        "artifact_digests": {
            name: hashlib.sha256(raws[name]).hexdigest()
            for name in sorted(PRODUCER_ARTIFACTS)
        },
        "checks": {name: not reasons for name, reasons in reasons_by_check.items()},
        "execution_id": execution_id,
        "execution_semantics_projection_digest": projection_digest,
        "producer_invoked": False,
        "resolved_run_spec_digest": spec_digest,
        "schema": VERIFICATION_SCHEMA,
        "verification_host_inputs_used": False,
        "verification_process": VERIFICATION_PROCESS,
    }
    reasons_by_check[CHECKSUM_CLOSURE] = _check_checksum_closure(root, raws, core=core)

    checks = {name: not reasons_by_check[name] for name in CHECKS}
    reasons = [reason for name in CHECKS for reason in reasons_by_check[name]]
    return {
        **core,
        "checks": checks,
        "reasons": reasons,
        "valid": all(checks.values()),
        # `verification.json` never carries the checksum-file digest: the
        # inventory covers this record, so a back-reference would be a cycle.
    }


def validate_verification(document: Mapping[str, Any]) -> None:
    """Fail closed if this command would emit a record its own contract refuses."""
    import jsonschema

    schema = _load_contract(VERIFICATION_CONTRACT)
    try:
        jsonschema.validate(instance=document, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ExecutionVerificationError(
            f"verification record violates {VERIFICATION_SCHEMA}: {exc.message}"
        ) from exc


def commit_verification(root: Path) -> tuple[Path, dict[str, Any]]:
    """Write the verdict, then close the archive over all four records.

    `verification.json` is created exclusively — a second verdict for one
    execution is a new answer to a spent question — and only then does the final
    inventory cover the four JSON files.
    """
    document = verify_archive(root)
    validate_verification(document)
    path = evidence.write_document_exclusive(root, VERIFICATION_NAME, document)
    evidence.write_checksum_inventory(root)
    return path, document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path, help="closed H2 execution archive root")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report the verdict without writing verification.json",
    )
    args = parser.parse_args(argv)
    try:
        if args.check_only:
            document = verify_archive(args.archive)
            validate_verification(document)
        else:
            _, document = commit_verification(args.archive)
    except (ExecutionVerificationError, evidence.EvidenceError, OSError) as exc:
        print(f"verification could not be formed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
