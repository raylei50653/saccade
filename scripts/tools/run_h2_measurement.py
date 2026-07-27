#!/usr/bin/env python3
"""H2 S4 Phase-A Layer-M controller.

This is the producer for the evidence contract landed before it.  It consumes an
externally prepared freeze plus the exact Layer-P certificate and files that
certificate binds; it does not construct or seal ``I``/``F``/``S``.  A real
invocation is therefore meaningful only after the separate owner seal and
exactly-once authorization exist.

The implemented scope is the charter's S4 items 0–4:

* an H2 child vector and recorder;
* the fixed four-run block on the Phase-A fixture;
* an independently recorded A7.6 comparison;
* the three frozen packet-verifier invocations and cross-repeat digest check; and
* slot-order normalization in the child, without editing the frozen H0 child.

No Phase-B launch is implemented here.  The evidence contract and verifier are
phase-aware already, but Correction 3 explicitly allows the Phase-B controller
to land after a passing Phase A.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import check_runtime_identity_staleness as staleness  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import run_h0_phase_a as h0_controller  # noqa: E402
import run_h0_phase_a_child as h0_child  # noqa: E402
import run_h2_measurement_child as child  # noqa: E402
import verify_h0_phase_a as h0_verifier  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

PHASE = "a"
SEQUENCE = evidence.expected_sequences(PHASE)[0]
CAPTURE_PHASE = evidence.CAPTURE_PHASE[PHASE]
DEADLINE_SECONDS = h0_controller.DEADLINE_SECONDS

(
    ACTIVE_PAIRS_MEMBER,
    FINAL_ROWS_MEMBER,
    MOT_MEMBER,
    RELINK_MEMBER,
) = behavior.BEHAVIOR_MEMBERS
PROPOSAL_MEMBER, WINNER_MEMBER = behavior.A76_PROJECTION_MEMBERS
OVERFLOW_MEMBER = behavior.A76_OVERFLOW_MEMBER


class ControllerError(RuntimeError):
    """The launch bundle or produced evidence is invalid."""


class Monitor(Protocol):
    history: list[Any]

    def drain(self) -> list[Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LaunchBundle:
    freeze: dict[str, Any]
    certificate: dict[str, Any]
    reference_probe: dict[str, Any]
    runtime_manifest: dict[str, Any]
    published_identity: dict[str, Any]
    freeze_path: Path
    certificate_path: Path
    reference_probe_path: Path
    runtime_manifest_path: Path
    published_identity_path: Path

    @property
    def head(self) -> str:
        return str(self.freeze["instrumentation_head"])

    @property
    def build_dir(self) -> Path:
        value = Path(str(self.certificate["build_dir"]))
        return value if value.is_absolute() else REPO_ROOT / value


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_canonical(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ControllerError(f"launch input is not a physical regular file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError(
            f"launch input is unreadable JSON: {path} ({exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise ControllerError(f"launch input is not an object: {path}")
    if raw != evidence.canonical_json_bytes(payload) + b"\n":
        raise ControllerError(f"launch input is not canonical JSON: {path}")
    if schema is not None and payload.get("schema") != schema:
        raise ControllerError(
            f"{path}: schema {payload.get('schema')!r}, expected {schema!r}"
        )
    return payload


def load_bundle(
    *,
    freeze_path: Path,
    certificate_path: Path,
    reference_probe_path: Path,
    runtime_manifest_path: Path,
    published_identity_path: Path,
    verify_runtime_files: bool = True,
) -> LaunchBundle:
    freeze = _load_canonical(freeze_path, schema=evidence.FREEZE_SCHEMA)
    certificate = _load_canonical(certificate_path, schema=CERTIFICATE_SCHEMA)
    reference_probe = _load_canonical(
        reference_probe_path, schema=behavior.RESULT_SCHEMA
    )
    published = _load_canonical(
        published_identity_path, schema=identity.IDENTITY_SCHEMA
    )
    try:
        identity.load_identity_behavior_probe(
            reference_probe_path, verify_witness_files=verify_runtime_files
        )
        staleness.load_published(published_identity_path)
        manifest = runtime_inputs.load_manifest(
            runtime_manifest_path, verify_files=verify_runtime_files
        )
    except (
        identity.IdentityError,
        runtime_inputs.RuntimeInputError,
        staleness.StalenessError,
        OSError,
    ) as exc:
        raise ControllerError(f"launch bundle rejected: {exc}") from exc

    head = freeze.get("instrumentation_head")
    if not _hex(head, 40):
        raise ControllerError("freeze has no 40-hex instrumentation_head")
    coordinate = freeze.get("coordinate")
    if not isinstance(coordinate, Mapping) or any(
        not _hex(coordinate.get(axis), 64) for axis in identity.ALL_COORDINATE_AXES
    ):
        raise ControllerError("freeze coordinate is absent or malformed")
    if not _hex(freeze.get("probe"), 64):
        raise ControllerError("freeze has no bounded probe digest")
    binding = freeze.get("layer_p_certificate")
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema") != CERTIFICATE_SCHEMA
        or not _hex(binding.get("digest"), 64)
    ):
        raise ControllerError("freeze has no Layer-P certificate binding")
    return LaunchBundle(
        freeze=dict(freeze),
        certificate=dict(certificate),
        reference_probe=dict(reference_probe),
        runtime_manifest=dict(manifest),
        published_identity=dict(published),
        freeze_path=freeze_path.resolve(strict=True),
        certificate_path=certificate_path.resolve(strict=True),
        reference_probe_path=reference_probe_path.resolve(strict=True),
        runtime_manifest_path=runtime_manifest_path.resolve(strict=True),
        published_identity_path=published_identity_path.resolve(strict=True),
    )


def certificate_match_reasons(
    bundle: LaunchBundle,
    *,
    current_head: str,
    current_tree: str,
) -> tuple[str, ...]:
    """Return every mismatch; an empty tuple is the terminal-1 predicate pass."""
    freeze = bundle.freeze
    certificate = bundle.certificate
    published = bundle.published_identity
    manifest = bundle.runtime_manifest
    reference = bundle.reference_probe
    binding = freeze["layer_p_certificate"]
    reasons: list[str] = []

    checks = (
        (
            evidence.digest(certificate) == binding["digest"],
            "certificate digest differs from F",
        ),
        (
            certificate.get("source_head") == bundle.head == current_head,
            "certificate/freeze/current head differ",
        ),
        (
            certificate.get("source_tree") == current_tree,
            "certificate tree differs from the execution checkout",
        ),
        (
            isinstance(certificate.get("selected_base"), str)
            and bool(certificate["selected_base"]),
            "certificate has no selected base",
        ),
        (
            isinstance(certificate.get("changed_path_verdict"), Mapping)
            and certificate["changed_path_verdict"].get("admissible") is True,
            "certificate changed-path verdict is not clean",
        ),
        (
            certificate.get("equivalence") == "unproven",
            "certificate claims an undeclared equivalence upgrade",
        ),
        (
            certificate.get("decision_relevant_digest")
            == freeze["coordinate"]["implementation"],
            "certificate implementation digest differs from F",
        ),
        (
            certificate.get("identity_semantics_digest")
            == freeze["coordinate"]["identity_semantics"],
            "certificate identity-semantics digest differs from F",
        ),
        (
            certificate.get("plumbing_set_digest")
            == identity.plumbing_axis()["digest"],
            "certificate plumbing-set digest differs from the execution checkout",
        ),
        (
            certificate.get("published_coordinate") == freeze["coordinate"],
            "certificate coordinate differs from F",
        ),
        (
            certificate.get("behavior_probe") == freeze["probe"]
            and certificate.get("published_probe") == freeze["probe"],
            "certificate probe differs from F",
        ),
        (
            certificate.get("runtime_input_coordinate_digest")
            == manifest["coordinate_digest"],
            "certificate runtime-input coordinate differs from the manifest",
        ),
        (
            certificate.get("runtime_input_full_digest") == manifest["full_digest"],
            "certificate runtime-input full digest differs from the manifest",
        ),
        (
            certificate.get("build_artifact_digest")
            == manifest["build_artifacts"]["digest"],
            "certificate build artifacts differ from the manifest",
        ),
        (
            evidence.sha256_file(bundle.runtime_manifest_path)
            == certificate.get("runtime_input_manifest_file_digest"),
            "runtime-input file digest differs from the certificate",
        ),
        (
            evidence.sha256_file(bundle.reference_probe_path)
            == certificate.get("probe_result_file_digest"),
            "reference-probe file digest differs from the certificate",
        ),
        (
            evidence.sha256_file(bundle.published_identity_path)
            == certificate.get("published_identity_file_digest"),
            "published-identity file digest differs from the certificate",
        ),
        (
            reference.get("digest") == freeze["probe"]
            and reference.get("identical") is True
            and reference.get("mode") == "identity"
            and reference.get("sequence") == behavior.IDENTITY_SEQUENCE,
            "reference probe does not support F",
        ),
        (
            certificate.get("probe_schema") == behavior.RESULT_SCHEMA
            and certificate.get("mode") == "identity"
            and certificate.get("fixture") == behavior.IDENTITY_SEQUENCE,
            "certificate probe declaration differs from the identity fixture",
        ),
        (
            isinstance(reference.get("build_witness"), Mapping)
            and reference["build_witness"].get("digest")
            == manifest["build_artifacts"]["digest"]
            and certificate.get("build_witness") == reference["build_witness"],
            "certificate/reference build witness differs from the manifest",
        ),
        (
            published.get("coordinate") == freeze["coordinate"]
            and isinstance(published.get("probe"), Mapping)
            and published["probe"].get("digest") == freeze["probe"]
            and isinstance(published.get("equivalence"), Mapping)
            and published["equivalence"].get("state") == "unproven"
            and published.get("publication_complete") is True,
            "published identity does not support F",
        ),
        (
            bundle.build_dir.resolve(strict=True)
            == Path(str(manifest["build_artifacts"]["build_dir"])).resolve(strict=True),
            "selected build directory differs from the runtime manifest",
        ),
    )
    reasons.extend(detail for passed, detail in checks if not passed)
    return tuple(reasons)


def revalidate_bundle(bundle: LaunchBundle) -> tuple[str, ...]:
    """Re-read every launch record under the active mutation monitor."""
    reasons: list[str] = []
    records = (
        (
            bundle.freeze_path,
            evidence.FREEZE_SCHEMA,
            bundle.freeze,
            "freeze",
        ),
        (
            bundle.certificate_path,
            CERTIFICATE_SCHEMA,
            bundle.certificate,
            "Layer-P certificate",
        ),
        (
            bundle.reference_probe_path,
            behavior.RESULT_SCHEMA,
            bundle.reference_probe,
            "reference probe",
        ),
        (
            bundle.runtime_manifest_path,
            runtime_inputs.SCHEMA,
            bundle.runtime_manifest,
            "runtime-input manifest",
        ),
        (
            bundle.published_identity_path,
            identity.IDENTITY_SCHEMA,
            bundle.published_identity,
            "published identity",
        ),
    )
    for path, schema, expected, label in records:
        try:
            current = _load_canonical(path, schema=schema)
        except ControllerError as exc:
            reasons.append(f"{label} revalidation failed: {exc}")
            continue
        if current != expected:
            reasons.append(f"{label} changed after initial intake")
    try:
        identity.load_identity_behavior_probe(
            bundle.reference_probe_path, verify_witness_files=True
        )
    except (identity.IdentityError, OSError) as exc:
        reasons.append(f"reference probe revalidation failed: {exc}")
    try:
        runtime_inputs.load_manifest(bundle.runtime_manifest_path, verify_files=True)
    except (runtime_inputs.RuntimeInputError, OSError) as exc:
        reasons.append(f"runtime-input revalidation failed: {exc}")
    try:
        staleness.load_published(bundle.published_identity_path)
    except (staleness.StalenessError, OSError) as exc:
        reasons.append(f"published-identity revalidation failed: {exc}")
    return tuple(reasons)


def child_argv(invocation_path: Path) -> tuple[str, ...]:
    if not invocation_path.is_absolute():
        raise ControllerError("child invocation path is not absolute")
    return (
        (REPO_ROOT / ".venv/bin/python").as_posix(),
        "-I",
        "-B",
        (REPO_ROOT / "scripts/tools/run_h2_measurement_child.py").as_posix(),
        "--invocation",
        invocation_path.as_posix(),
    )


def child_environment(
    run_dir: Path,
    *,
    build_dir: Path,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = dict(os.environ if inherited is None else inherited)
    environment = dict(h0_child.STATIC_ENV)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": source.get("CUDA_VISIBLE_DEVICES", "0"),
            "HOME": (run_dir / "_env" / "home").as_posix(),
            "LD_LIBRARY_PATH": ":".join(
                value
                for value in (
                    build_dir.as_posix(),
                    source.get("LD_LIBRARY_PATH", ""),
                )
                if value
            ),
            "PATH": f"{REPO_ROOT.as_posix()}/.venv/bin:/usr/bin:/bin",
            "SACCADE_BUILD_PATH": build_dir.as_posix(),
            "TMPDIR": (run_dir / "_env" / "tmp").as_posix(),
            "XDG_CACHE_HOME": (run_dir / "_env" / "xdg-cache").as_posix(),
        }
    )
    if set(environment) != h0_child.EXPECTED_ENV_KEYS:
        raise ControllerError("internal child environment key drift")
    return environment


def _environment_digest(environment: Mapping[str, str]) -> str:
    return hashlib.sha256(evidence.canonical_json_bytes(dict(environment))).hexdigest()


def _remaining(started: float, clock: Callable[[], float]) -> float:
    remaining = DEADLINE_SECONDS - (clock() - started)
    if remaining <= 0:
        raise TimeoutError("H2 Phase-A monotonic deadline exhausted")
    return remaining


def default_launch_probe(
    root: Path,
    *,
    build_dir: Path,
    monitor: Monitor,
    started: float,
    clock: Callable[[], float],
) -> dict[str, Any]:
    output = root / evidence.LAUNCH_PROBE_NAME
    runtime = root / "_launch_probe_runtime"
    environment = {**os.environ, "SACCADE_BUILD_PATH": build_dir.as_posix()}
    vector = [
        (REPO_ROOT / ".venv/bin/python").as_posix(),
        (_TOOLS / "h2_behavioral_identity.py").as_posix(),
        "--identity-mode",
        "--emit",
        output.as_posix(),
        "--out-dir",
        runtime.as_posix(),
    ]
    with (
        (root / "launch_probe.stdout.log").open("xb", buffering=0) as stdout,
        (root / "launch_probe.stderr.log").open("xb", buffering=0) as stderr,
    ):
        process = subprocess.Popen(
            vector,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            start_new_session=True,
        )
        returncode = h0_controller._wait_with_monitor(
            process,
            started=started,
            monitor=monitor,
            stage="H2 launch probe",
            clock=clock,
        )
    if returncode != 0:
        raise ControllerError(f"launch probe exited {returncode}")
    try:
        return identity.load_identity_behavior_probe(output)
    except identity.IdentityError as exc:
        raise ControllerError(f"launch probe rejected: {exc}") from exc


def default_child_launcher(
    invocation_path: Path,
    environment: Mapping[str, str],
    *,
    monitor: Monitor,
    started: float,
    clock: Callable[[], float],
) -> int:
    run_dir = invocation_path.parent
    with (
        (run_dir / "stdout.log").open("xb", buffering=0) as stdout,
        (run_dir / "stderr.log").open("xb", buffering=0) as stderr,
    ):
        process = subprocess.Popen(
            child_argv(invocation_path),
            cwd=REPO_ROOT,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            start_new_session=True,
        )
        return h0_controller._wait_with_monitor(
            process,
            started=started,
            monitor=monitor,
            stage=f"H2 child {invocation_path.parent.name}",
            clock=clock,
        )


def _read_run_document(root: Path, run_id: str, name: str) -> dict[str, Any]:
    directory = evidence.run_dir(root, SEQUENCE, run_id)
    try:
        return evidence.load_document(directory, name)
    except evidence.EvidenceError as exc:
        raise ControllerError(f"{SEQUENCE}/{run_id}: {exc}") from exc


def compare_policy_inventories(root: Path) -> tuple[bool, dict[str, Any]]:
    """Controller-side A7.6 implementation; the verifier recomputes separately."""
    inventories = {
        run_id: _read_run_document(root, run_id, evidence.POLICY_INVENTORY_NAME)
        for run_id in evidence.RUN_IDS
    }
    for run_id, inventory in inventories.items():
        try:
            h0_verifier._verify_policy_inventory(run_id, inventory)
        except h0_verifier.VerificationError as exc:
            raise ControllerError(f"{SEQUENCE}/{run_id}: {exc}") from exc
        mot_path = evidence.run_dir(root, SEQUENCE, run_id) / f"{SEQUENCE}.txt"
        try:
            mot = mot_path.read_bytes()
        except OSError as exc:
            raise ControllerError(
                f"{SEQUENCE}/{run_id}: MOT output unreadable"
            ) from exc
        if inventory[MOT_MEMBER] != {
            "length": len(mot),
            "sha256": hashlib.sha256(mot).hexdigest(),
        }:
            raise ControllerError(f"{SEQUENCE}/{run_id}: MOT identity mismatch")

    relations: list[dict[str, Any]] = []
    first_unequal: str | None = None

    def record(left: str, member: str, right: str, equal: bool) -> None:
        nonlocal first_unequal
        relations.append(
            {"equal": equal, "left": left, "member": member, "right": right}
        )
        if not equal and first_unequal is None:
            first_unequal = f"{left}:{right}:{member}"

    off = inventories[evidence.CAPTURE_OFF_RUN]
    for run_id in evidence.CAPTURE_ON_RUNS:
        for member in behavior.A76_EQUALITY_MEMBERS:
            record(
                evidence.CAPTURE_OFF_RUN,
                member,
                run_id,
                off[member] == inventories[run_id][member],
            )
    reference = inventories[evidence.CAPTURE_ON_RUNS[0]]
    for member in behavior.A76_PROJECTION_MEMBERS:
        for run_id in evidence.CAPTURE_ON_RUNS[1:]:
            record(
                evidence.CAPTURE_ON_RUNS[0],
                member,
                run_id,
                reference[member] == inventories[run_id][member],
            )
    for run_id in evidence.CAPTURE_ON_RUNS:
        record(
            run_id,
            behavior.A76_OVERFLOW_MEMBER,
            "zero_vector",
            inventories[run_id][behavior.A76_OVERFLOW_MEMBER]
            == list(behavior.A76_OVERFLOW_ZERO_VECTOR),
        )
    comparison = {
        "first_unequal": first_unequal,
        "relations": relations,
        "state": "equal" if first_unequal is None else "unequal",
    }
    return first_unequal is None, comparison


def verify_packets(root: Path) -> bool:
    reports: list[Mapping[str, Any]] = []
    valid = True
    for run_id in evidence.CAPTURE_ON_RUNS:
        packet = _read_run_document(root, run_id, evidence.PACKET_NAME)
        stored = _read_run_document(root, run_id, evidence.PACKET_VERIFICATION_NAME)
        try:
            report = verify_capture(packet)
        except (KeyError, TypeError, ValueError):
            valid = False
            if stored != {"failure": "packet_invalid", "state": "fail"}:
                raise ControllerError(
                    f"{SEQUENCE}/{run_id}: packet failure record mismatch"
                )
            continue
        if stored != {"report": report, "state": "pass"}:
            raise ControllerError(f"{SEQUENCE}/{run_id}: packet pass record mismatch")
        reports.append(report)
    semantic_digests = [report["semantic_digest_sha256"] for report in reports]
    return (
        valid
        and len(reports) == len(evidence.CAPTURE_ON_RUNS)
        and len(set(semantic_digests)) == 1
    )


def _mutation_payload(events: Iterable[Any]) -> dict[str, Any]:
    rows = [
        {
            "classification": str(getattr(event, "classification", "")),
            "mask": int(getattr(event, "mask", 0)),
            "path": str(getattr(event, "path", "")),
        }
        for event in events
    ]
    return {"schema": evidence.MUTATION_SCHEMA, "events": rows, "mutated": bool(rows)}


def _terminal_record(selection: partition.Selection) -> dict[str, Any]:
    return {
        "schema": evidence.TERMINAL_SCHEMA,
        "order": selection.order,
        "phase": selection.phase,
        "result": selection.result,
        "terminal": selection.terminal,
    }


def _manifest(root: Path, *, head: str, result: str) -> dict[str, Any]:
    files = sorted(
        path.relative_to(root).as_posix() for path in evidence.evidence_files(root)
    )
    freeze = evidence.load_document(
        root, evidence.FREEZE_NAME, schema=evidence.FREEZE_SCHEMA
    )
    return {
        "schema": evidence.MANIFEST_SCHEMA,
        "artifact_inventory": sorted({*files, evidence.MANIFEST_NAME}),
        "capture_phase": CAPTURE_PHASE,
        "freeze_digest": evidence.freeze_digest(freeze),
        "instrumentation_head": head,
        "result": result,
    }


def _archive_bundle(root: Path, bundle: LaunchBundle) -> None:
    records = (
        (evidence.FREEZE_NAME, bundle.freeze),
        (evidence.CERTIFICATE_NAME, bundle.certificate),
        (evidence.REFERENCE_PROBE_NAME, bundle.reference_probe),
        (evidence.RUNTIME_INPUTS_NAME, bundle.runtime_manifest),
        (evidence.PUBLISHED_IDENTITY_NAME, bundle.published_identity),
    )
    for name, payload in records:
        evidence.write_document(root, name, payload)


def _monitor_paths(bundle: LaunchBundle) -> tuple[Path, ...]:
    paths = set(runtime_inputs.bound_paths(bundle.runtime_manifest))
    paths.update(
        {
            bundle.freeze_path,
            bundle.certificate_path,
            bundle.reference_probe_path,
            bundle.runtime_manifest_path,
            bundle.published_identity_path,
        }
    )
    for path_class in ("decision_relevant", "identity_semantics", "plumbing_only"):
        paths.update(
            REPO_ROOT / relative
            for relative in identity.tracked_files_for_class(path_class)
            if (REPO_ROOT / relative).is_file()
        )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _prepare_run(
    root: Path,
    *,
    bundle: LaunchBundle,
    run_id: str,
    policy_fingerprint: str,
    inherited_environment: Mapping[str, str] | None,
) -> tuple[Path, dict[str, str]]:
    run_dir = evidence.run_dir(root, SEQUENCE, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    for leaf in ("home", "tmp", "xdg-cache"):
        (run_dir / "_env" / leaf).mkdir(parents=True, exist_ok=False)
    environment = child_environment(
        run_dir,
        build_dir=bundle.build_dir.resolve(strict=True),
        inherited=inherited_environment,
    )
    invocation = {
        "schema": child.INVOCATION_SCHEMA,
        "build_dir": bundle.build_dir.resolve(strict=True).as_posix(),
        "capture_phase": CAPTURE_PHASE,
        "capture_run_uuid": str(uuid.uuid4()),
        "environment_digest": _environment_digest(environment),
        "instrumentation_head": bundle.head,
        "policy_fingerprint": policy_fingerprint,
        "run_dir": run_dir.resolve(strict=True).as_posix(),
        "run_id": run_id,
        "sequence": SEQUENCE,
        "state": "running",
    }
    invocation_path = evidence.write_document(run_dir, "invocation.json", invocation)
    return invocation_path, environment


def _finalize(
    incomplete: Path,
    final: Path,
    *,
    bundle: LaunchBundle,
    predicates: Mapping[str, bool],
    mutation_events: Sequence[Any],
    execution_result: str | None,
    controller_record: Mapping[str, Any],
) -> tuple[Path, partition.Selection]:
    observation = evidence.build_observation(
        predicates, execution_result=execution_result
    )
    selection = partition.select_terminal(
        evidence.observation_predicates(observation), phase=PHASE
    )
    evidence.write_document(
        incomplete, evidence.MUTATION_NAME, _mutation_payload(mutation_events)
    )
    evidence.write_document(incomplete, evidence.OBSERVATION_NAME, observation)
    evidence.write_document(
        incomplete, evidence.TERMINAL_NAME, _terminal_record(selection)
    )
    evidence.write_document(
        incomplete,
        evidence.CONTROLLER_NAME,
        {
            **controller_record,
            "finished_utc": _utc(),
            "result": selection.result,
            "terminal": selection.terminal,
        },
    )
    evidence.write_document(
        incomplete,
        evidence.MANIFEST_NAME,
        _manifest(incomplete, head=bundle.head, result=selection.result),
    )
    evidence.write_checksum_inventory(incomplete)
    if final.exists() or final.is_symlink():
        raise ControllerError(f"final evidence root already exists: {final}")
    incomplete.rename(final)
    return final, selection


ProbeRunner = Callable[..., dict[str, Any]]
ChildLauncher = Callable[..., int]
MonitorFactory = Callable[..., Monitor]
BundleRevalidator = Callable[[LaunchBundle], tuple[str, ...]]


def execute_controller(
    bundle: LaunchBundle,
    *,
    evidence_parent: Path | None = None,
    bound_paths: Sequence[Path] | None = None,
    require_clean_checkout: bool = True,
    launch_probe: ProbeRunner = default_launch_probe,
    launch_child: ChildLauncher = default_child_launcher,
    monitor_factory: MonitorFactory = h0_controller.BoundInputMonitor,
    bundle_revalidator: BundleRevalidator = revalidate_bundle,
    inherited_environment: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Path, partition.Selection]:
    """Execute one externally authorized Phase-A invocation exactly once."""
    started = clock()
    if require_clean_checkout and _git(
        "status", "--porcelain", "--untracked-files=normal"
    ):
        raise ControllerError("Layer-M requires the exact clean head bound by F")
    parent = evidence_parent or (REPO_ROOT / evidence.EVIDENCE_REL)
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / evidence.phase_a_root_name(bundle.head)
    incomplete = final.with_name(final.name + ".incomplete")
    if (
        final.exists()
        or incomplete.exists()
        or final.is_symlink()
        or incomplete.is_symlink()
    ):
        raise ControllerError("stale final or incomplete H2 evidence root")

    predicates = {key: True for key, _ in partition.ORDERED_PREDICATES}
    predicates["bound_input_mutated"] = False
    predicates["execution_complete"] = False
    controller_record: dict[str, Any] = {
        "schema": evidence.CONTROLLER_SCHEMA,
        "capture_phase": CAPTURE_PHASE,
        "instrumentation_head": bundle.head,
        "ordered_runs": list(evidence.RUN_IDS),
        "sequence": SEQUENCE,
        "started_utc": _utc(),
        "state": "running",
    }
    execution_result: str | None = None
    mutation_events: list[Any] = []
    monitor: Monitor | None = None
    monitor_failure: BaseException | None = None
    try:
        # The output root does not exist yet. Install the monitor first, then
        # re-read every launch input and the checkout under that monitor. This
        # closes the initial intake -> watch-install TOCTOU window.
        try:
            monitor = monitor_factory(
                tuple(bound_paths)
                if bound_paths is not None
                else _monitor_paths(bundle),
                ignored_roots=(incomplete,),
            )
        except (h0_controller.DriftError, OSError) as exc:
            monitor_failure = exc

        incomplete.mkdir()
        _archive_bundle(incomplete, bundle)
        if monitor_failure is not None:
            raise ControllerError(
                f"bound-input monitor could not start: {monitor_failure}"
            )
        assert monitor is not None

        current_head = _git("rev-parse", "HEAD")
        current_tree = _git("rev-parse", "HEAD^{tree}")
        reasons = list(bundle_revalidator(bundle))
        reasons.extend(
            certificate_match_reasons(
                bundle, current_head=current_head, current_tree=current_tree
            )
        )
        if require_clean_checkout and _git(
            "status", "--porcelain", "--untracked-files=normal"
        ):
            reasons.append("execution checkout changed before monitored revalidation")
        predicates["layer_p_certificate_matches_freeze"] = not reasons
        controller_record["certificate_mismatch_reasons"] = list(reasons)
        if not reasons:
            probe = launch_probe(
                incomplete,
                build_dir=bundle.build_dir.resolve(strict=True),
                monitor=monitor,
                started=started,
                clock=clock,
            )
            if not (incomplete / evidence.LAUNCH_PROBE_NAME).is_file():
                evidence.write_document(incomplete, evidence.LAUNCH_PROBE_NAME, probe)
            launch_witness = probe.get("build_witness")
            predicates["behavior_probe_equals_freeze"] = (
                probe.get("digest") == bundle.freeze["probe"]
                and isinstance(launch_witness, Mapping)
                and launch_witness.get("digest")
                == bundle.runtime_manifest["build_artifacts"]["digest"]
            )
            controller_record["launch_probe_build_witness_matches"] = (
                isinstance(launch_witness, Mapping)
                and launch_witness.get("digest")
                == bundle.runtime_manifest["build_artifacts"]["digest"]
            )

        if not reasons and predicates["behavior_probe_equals_freeze"]:
            policy_fingerprint = identity.decision_surface_axis()[
                "resolved_bridge_policy_config_v1"
            ]
            for run_id in evidence.RUN_IDS:
                _remaining(started, clock)
                invocation_path, environment = _prepare_run(
                    incomplete,
                    bundle=bundle,
                    run_id=run_id,
                    policy_fingerprint=policy_fingerprint,
                    inherited_environment=inherited_environment,
                )
                returncode = launch_child(
                    invocation_path,
                    environment,
                    monitor=monitor,
                    started=started,
                    clock=clock,
                )
                invocation = evidence.load_document(
                    invocation_path.parent,
                    invocation_path.name,
                    schema=child.INVOCATION_SCHEMA,
                )
                if returncode != 0 or invocation.get("state") != "completed":
                    raise ControllerError(f"child {run_id} exited nonzero")

            equal, comparison = compare_policy_inventories(incomplete)
            evidence.write_document(
                incomplete / evidence.RUNS_DIR / SEQUENCE,
                evidence.COMPARISON_NAME,
                comparison,
            )
            predicates["capture_off_on_equal"] = equal
            predicates["packets_valid"] = verify_packets(incomplete)
            predicates["execution_complete"] = True

            post_reasons = list(bundle_revalidator(bundle))
            post_head = _git("rev-parse", "HEAD")
            post_tree = _git("rev-parse", "HEAD^{tree}")
            if post_head != current_head or post_tree != current_tree:
                post_reasons.append("execution checkout head/tree moved during the run")
            if require_clean_checkout and _git(
                "status", "--porcelain", "--untracked-files=normal"
            ):
                post_reasons.append("execution checkout became dirty during the run")
            if post_reasons:
                predicates["layer_p_certificate_matches_freeze"] = False
                controller_record["certificate_mismatch_reasons"].extend(post_reasons)
    except h0_controller.DriftError as exc:
        if monitor is not None:
            mutation_events.extend(monitor.history)
        if not mutation_events:
            mutation_events.append(
                type(
                    "Mutation",
                    (),
                    {
                        "classification": "bound_mutation",
                        "mask": 0,
                        "path": str(exc),
                    },
                )()
            )
        predicates["bound_input_mutated"] = True
        controller_record["failure"] = {
            "reason": str(exc),
            "stage": "bound_input_monitor",
        }
    except TimeoutError as exc:
        execution_result = "runner_timeout"
        controller_record["failure"] = {"reason": str(exc), "stage": "execution"}
    except (evidence.EvidenceError, OSError, ControllerError) as exc:
        execution_result = (
            "unclassified_execution_failure"
            if monitor_failure is not None
            else "runner_nonzero"
        )
        controller_record["failure"] = {
            "reason": str(exc),
            "stage": (
                "bound_input_monitor" if monitor_failure is not None else "execution"
            ),
        }
    except BaseException as exc:
        execution_result = "unclassified_execution_failure"
        controller_record["failure"] = {
            "reason": str(exc) or type(exc).__name__,
            "stage": "execution",
        }
    finally:
        if monitor is not None:
            try:
                mutation_events.extend(monitor.drain())
            except BaseException as exc:
                predicates["execution_complete"] = False
                execution_result = "unclassified_execution_failure"
                controller_record["failure"] = {
                    "reason": str(exc) or type(exc).__name__,
                    "stage": "bound_input_monitor_final_drain",
                }
            predicates["bound_input_mutated"] = bool(mutation_events)
            try:
                monitor.close()
            except OSError as exc:
                predicates["execution_complete"] = False
                execution_result = "unclassified_execution_failure"
                controller_record["failure"] = {
                    "reason": str(exc),
                    "stage": "bound_input_monitor_close",
                }

    return _finalize(
        incomplete,
        final,
        bundle=bundle,
        predicates=predicates,
        mutation_events=mutation_events,
        execution_result=execution_result,
        controller_record={**controller_record, "state": "terminal"},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--layer-p-certificate", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--runtime-inputs", type=Path, required=True)
    parser.add_argument("--published-identity", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        bundle = load_bundle(
            freeze_path=args.freeze,
            certificate_path=args.layer_p_certificate,
            reference_probe_path=args.reference_probe,
            runtime_manifest_path=args.runtime_inputs,
            published_identity_path=args.published_identity,
        )
        root, selection = execute_controller(bundle)
        report = verifier.verify_evidence_root(root)
    except (
        ControllerError,
        evidence.EvidenceError,
        verifier.VerificationError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"H2 Phase-A controller rejected: {exc}", file=sys.stderr)
        return 2
    print(selection.describe())
    print(f"evidence: {root}")
    print(f"verifier: valid={report['valid']} result={report['result']}")
    return 0 if selection.terminal is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
