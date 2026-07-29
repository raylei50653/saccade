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
from export_headline_bridge_decision_trace import (  # noqa: E402
    OVERFLOW_KEYS,
    STREAMS,
    UNIVERSE_OVERFLOW_KEYS,
    UNIVERSE_STREAMS,
    canonical_semantic_packet,
)
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
OVERFLOW_FIELDS = tuple(OVERFLOW_KEYS[name] for name in STREAMS) + tuple(
    UNIVERSE_OVERFLOW_KEYS[name] for name in UNIVERSE_STREAMS
)


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


@dataclass(frozen=True)
class AuthorizationGrant:
    record: dict[str, Any]
    path: Path

    @property
    def digest(self) -> str:
        return evidence.digest(self.record)


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


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return completed.stdout


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
    runtime_binding = freeze.get("runtime_inputs")
    published_binding = freeze.get("published_identity")
    reference_binding = freeze.get("reference_probe")
    capture_abi = freeze.get("capture_abi")
    run_plan = freeze.get("run_plan")
    executed = freeze.get("executed_surfaces")
    selected_base = freeze.get("selected_base")
    changed_path_verdict = certificate.get("changed_path_verdict")
    if (
        set(freeze) != evidence.PHASE_A_FREEZE_MEMBERS
        or freeze.get("capture_phase") != CAPTURE_PHASE
        or not _hex(selected_base, 40)
        or certificate.get("selected_base") != selected_base
        or not isinstance(changed_path_verdict, Mapping)
        or changed_path_verdict.get("base") != selected_base
        or certificate.get("source_head") != head
        or certificate.get("equivalence") != "unproven"
        or freeze.get("equivalence") != "unproven"
        or binding.get("digest") != evidence.digest(certificate)
        or not isinstance(reference_binding, Mapping)
        or reference_binding
        != {
            "schema": behavior.RESULT_SCHEMA,
            "file_digest": evidence.sha256_file(reference_probe_path),
        }
        or not isinstance(runtime_binding, Mapping)
        or runtime_binding
        != {
            "schema": runtime_inputs.SCHEMA,
            "file_digest": evidence.sha256_file(runtime_manifest_path),
            "coordinate_digest": manifest.get("coordinate_digest"),
            "full_digest": manifest.get("full_digest"),
            "build_artifact_digest": manifest.get("build_artifacts", {}).get("digest"),
        }
        or not isinstance(published_binding, Mapping)
        or published_binding
        != {
            "schema": identity.IDENTITY_SCHEMA,
            "file_digest": evidence.sha256_file(published_identity_path),
        }
        or capture_abi
        != {
            "path": evidence.PHASE_A_CAPTURE_ABI_PATH,
            "sha256": hashlib.sha256(
                _git_bytes("show", f"{head}:{evidence.PHASE_A_CAPTURE_ABI_PATH}")
            ).hexdigest(),
        }
        or not isinstance(executed, Mapping)
        or set(executed) != set(evidence.PHASE_A_EXECUTED_SURFACE_PATHS)
        or any(
            executed.get(path)
            != hashlib.sha256(_git_bytes("show", f"{head}:{path}")).hexdigest()
            for path in evidence.PHASE_A_EXECUTED_SURFACE_PATHS
        )
        or run_plan
        != {
            "sequence": SEQUENCE,
            "run_ids": list(evidence.RUN_IDS),
        }
    ):
        raise ControllerError("freeze does not reconstruct from its primary bindings")
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


def _authorization_mismatch_reasons(
    grant: Mapping[str, Any],
    bundle: LaunchBundle,
    *,
    execution_domain_digest: str,
) -> tuple[str, ...]:
    controller_digest = bundle.freeze["executed_surfaces"][
        "scripts/tools/run_h2_measurement.py"
    ]
    checks = (
        (set(grant) == evidence.AUTHORIZATION_GRANT_MEMBERS, "member set"),
        (_hex(grant.get("authorization_id"), 64), "authorization id"),
        (_hex(grant.get("invocation_id"), 64), "invocation id"),
        (grant.get("capture_phase") == CAPTURE_PHASE, "capture phase"),
        (grant.get("instrumentation_head") == bundle.head, "head"),
        (
            grant.get("freeze_digest") == evidence.freeze_digest(bundle.freeze),
            "freeze",
        ),
        (grant.get("controller_digest") == controller_digest, "controller"),
        (
            grant.get("execution_domain") == execution_domain_digest,
            "execution domain",
        ),
        (grant.get("issued_by") == evidence.AUTHORIZATION_ISSUER, "issuer"),
    )
    return tuple(label for passed, label in checks if not passed)


def load_authorization(
    path: Path,
    bundle: LaunchBundle,
    *,
    invocation_id: str,
    authorization_ledger: Path | None = None,
) -> AuthorizationGrant:
    grant = _load_canonical(path, schema=evidence.AUTHORIZATION_GRANT_SCHEMA)
    ledger = (
        authorization_ledger
        if authorization_ledger is not None
        else default_authorization_ledger()
    ).resolve(strict=False)
    execution_domain_digest = evidence.digest(
        evidence.authorization_execution_domain(ledger)
    )
    if (
        not _hex(invocation_id, 64)
        or grant.get("invocation_id") != invocation_id
        or _authorization_mismatch_reasons(
            grant,
            bundle,
            execution_domain_digest=execution_domain_digest,
        )
    ):
        raise ControllerError(
            "exactly-once authorization is absent or bound to another "
            "head/freeze/controller/invocation"
        )
    return AuthorizationGrant(record=grant, path=path.resolve(strict=True))


def certificate_match_reasons(
    bundle: LaunchBundle,
    *,
    current_head: str,
    current_tree: str,
    checkout_witness: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every mismatch; an empty tuple is the terminal-1 predicate pass."""
    freeze = bundle.freeze
    certificate = bundle.certificate
    published = bundle.published_identity
    manifest = bundle.runtime_manifest
    reference = bundle.reference_probe
    binding = freeze["layer_p_certificate"]
    axes = checkout_witness["axes"]
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
            certificate.get("source_tree")
            == checkout_witness.get("source_tree")
            == current_tree,
            "certificate tree differs from the execution checkout",
        ),
        (
            _hex(certificate.get("selected_base"), 40)
            and certificate.get("selected_base") == freeze.get("selected_base"),
            "certificate selected base is not exact or differs from F",
        ),
        (
            isinstance(certificate.get("changed_path_verdict"), Mapping)
            and certificate["changed_path_verdict"].get("admissible") is True
            and certificate["changed_path_verdict"].get("base")
            == certificate.get("selected_base")
            == freeze.get("selected_base"),
            "certificate changed-path verdict is not clean",
        ),
        (
            certificate.get("equivalence") == "unproven",
            "certificate claims an undeclared equivalence upgrade",
        ),
        (
            certificate.get("decision_relevant_digest")
            == freeze["coordinate"]["implementation"]
            == axes["decision_relevant"]["digest"],
            "certificate implementation digest differs from F/checkout",
        ),
        (
            certificate.get("identity_semantics_digest")
            == freeze["coordinate"]["identity_semantics"]
            == axes["identity_semantics"]["digest"],
            "certificate identity-semantics digest differs from F/checkout",
        ),
        (
            certificate.get("plumbing_set_digest") == axes["plumbing_only"]["digest"],
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
            == Path(str(manifest["build_artifacts"]["build_dir"])).resolve(strict=True)
            == Path(str(checkout_witness["build_dir"])).resolve(strict=True),
            "selected build directory differs from the runtime manifest",
        ),
    )
    reasons.extend(detail for passed, detail in checks if not passed)
    return tuple(reasons)


def build_checkout_witness(
    bundle: LaunchBundle, *, current_head: str, current_tree: str
) -> dict[str, Any]:
    """Capture the live values that an archive verifier rebuilds from Git."""
    return {
        "schema": evidence.CHECKOUT_WITNESS_SCHEMA,
        "axes": {
            "decision_relevant": identity.implementation_axis(),
            "identity_semantics": identity.identity_semantics_axis(),
            "plumbing_only": identity.plumbing_axis(),
        },
        "build_dir": bundle.build_dir.resolve(strict=True).as_posix(),
        "repository_root": REPO_ROOT.resolve(strict=True).as_posix(),
        "source_head": current_head,
        "source_tree": current_tree,
    }


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
    on_started: Callable[[], None],
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
        try:
            on_started()
        except BaseException:
            h0_controller._terminate_process_group(process)
            process.wait()
            raise
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


@dataclass(frozen=True)
class SurvivingReplay:
    capture_equal: bool
    packets_valid: bool
    complete: bool
    comparison: dict[str, Any]
    errors: tuple[str, ...]
    evidence_present: bool


def _verify_base_policy_inventory(run_id: str, inventory: Mapping[str, Any]) -> None:
    expected = {*behavior.A76_EQUALITY_MEMBERS, "schema"}
    if (
        set(inventory) != expected
        or inventory.get("schema") != evidence.BASE_POLICY_INVENTORY_SCHEMA
    ):
        raise ControllerError(f"{SEQUENCE}/{run_id}: base inventory schema mismatch")
    synthetic = {
        **inventory,
        OVERFLOW_MEMBER: list(behavior.A76_OVERFLOW_ZERO_VECTOR),
        PROPOSAL_MEMBER: None,
        "schema": evidence.POLICY_INVENTORY_SCHEMA,
        WINNER_MEMBER: None,
    }
    try:
        h0_verifier._verify_policy_inventory(evidence.CAPTURE_OFF_RUN, synthetic)
    except h0_verifier.VerificationError as exc:
        raise ControllerError(f"{SEQUENCE}/{run_id}: {exc}") from exc


def replay_surviving_evidence(root: Path) -> SurvivingReplay:
    """Replay every durable base/full inventory and packet, even after failure."""
    bases: dict[str, dict[str, Any]] = {}
    base_records: set[str] = set()
    inventories: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for run_id in evidence.RUN_IDS:
        directory = evidence.run_dir(root, SEQUENCE, run_id)
        mot_path = directory / f"{SEQUENCE}.txt"
        if mot_path.is_file():
            try:
                mot = mot_path.read_bytes()
                bases[run_id] = {
                    MOT_MEMBER: {
                        "length": len(mot),
                        "sha256": hashlib.sha256(mot).hexdigest(),
                    }
                }
            except OSError as exc:
                errors.append(f"{SEQUENCE}/{run_id}: durable MOT is unreadable: {exc}")
        base_path = directory / evidence.BASE_POLICY_INVENTORY_NAME
        if base_path.is_file():
            try:
                base = _read_run_document(
                    root, run_id, evidence.BASE_POLICY_INVENTORY_NAME
                )
                _verify_base_policy_inventory(run_id, base)
                base_records.add(run_id)
                durable_mot = bases.get(run_id, {}).get(MOT_MEMBER)
                if durable_mot is None:
                    errors.append(
                        f"{SEQUENCE}/{run_id}: base inventory has no durable MOT"
                    )
                elif base[MOT_MEMBER] != durable_mot:
                    errors.append(f"{SEQUENCE}/{run_id}: base MOT identity mismatch")
                bases.setdefault(run_id, {}).update(
                    {
                        member: base[member]
                        for member in behavior.A76_EQUALITY_MEMBERS
                        if member != MOT_MEMBER
                    }
                )
            except ControllerError as exc:
                errors.append(str(exc))
        if (directory / evidence.POLICY_INVENTORY_NAME).is_file():
            try:
                inventories[run_id] = _read_run_document(
                    root, run_id, evidence.POLICY_INVENTORY_NAME
                )
            except ControllerError as exc:
                errors.append(str(exc))
    invalid_inventories: list[str] = []
    for run_id, inventory in inventories.items():
        try:
            h0_verifier._verify_policy_inventory(run_id, inventory)
        except h0_verifier.VerificationError as exc:
            errors.append(f"{SEQUENCE}/{run_id}: {exc}")
            invalid_inventories.append(run_id)
            continue
        if run_id not in base_records:
            errors.append(f"{SEQUENCE}/{run_id}: full inventory has no durable base")
        elif any(
            member not in bases[run_id] or inventory[member] != bases[run_id][member]
            for member in behavior.A76_EQUALITY_MEMBERS
        ):
            errors.append(f"{SEQUENCE}/{run_id}: full/base inventory mismatch")
    for run_id in invalid_inventories:
        inventories.pop(run_id)

    relations: list[dict[str, Any]] = []
    first_unequal: str | None = None
    base_unequal = False
    packet_relation_failure = False

    def record(
        left: str,
        member: str,
        right: str,
        equal: bool,
        *,
        base_relation: bool,
    ) -> None:
        nonlocal base_unequal, first_unequal, packet_relation_failure
        relations.append(
            {"equal": equal, "left": left, "member": member, "right": right}
        )
        if not equal and first_unequal is None:
            first_unequal = f"{left}:{right}:{member}"
        if not equal:
            if base_relation:
                base_unequal = True
            else:
                packet_relation_failure = True

    off_id = evidence.CAPTURE_OFF_RUN
    if off_id in bases:
        for run_id in evidence.CAPTURE_ON_RUNS:
            if run_id not in bases:
                continue
            for member in behavior.A76_EQUALITY_MEMBERS:
                if member not in bases[off_id] or member not in bases[run_id]:
                    continue
                record(
                    off_id,
                    member,
                    run_id,
                    bases[off_id][member] == bases[run_id][member],
                    base_relation=True,
                )
    on_present = [run for run in evidence.CAPTURE_ON_RUNS if run in inventories]
    if on_present:
        reference_id = on_present[0]
        reference = inventories[reference_id]
        for member in behavior.A76_PROJECTION_MEMBERS:
            for run_id in on_present[1:]:
                record(
                    reference_id,
                    member,
                    run_id,
                    reference[member] == inventories[run_id][member],
                    base_relation=False,
                )
    for run_id in on_present:
        record(
            run_id,
            behavior.A76_OVERFLOW_MEMBER,
            "zero_vector",
            inventories[run_id][behavior.A76_OVERFLOW_MEMBER]
            == list(behavior.A76_OVERFLOW_ZERO_VECTOR),
            base_relation=False,
        )
    comparison = {
        "first_unequal": first_unequal,
        "relations": relations,
        "state": "equal" if first_unequal is None else "unequal",
    }
    packet_states: dict[str, str] = {}
    semantic_digests: list[str] = []
    for run_id in evidence.CAPTURE_ON_RUNS:
        directory = evidence.run_dir(root, SEQUENCE, run_id)
        if not (directory / evidence.PACKET_NAME).is_file():
            packet_states[run_id] = "unavailable"
            continue
        try:
            packet = _read_run_document(root, run_id, evidence.PACKET_NAME)
        except ControllerError as exc:
            errors.append(str(exc))
            packet_states[run_id] = "unavailable"
            continue
        stored: Mapping[str, Any] | None = None
        if (directory / evidence.PACKET_VERIFICATION_NAME).is_file():
            try:
                stored = _read_run_document(
                    root, run_id, evidence.PACKET_VERIFICATION_NAME
                )
            except ControllerError as exc:
                errors.append(str(exc))
        try:
            report = verify_capture(packet)
            semantic_packet = canonical_semantic_packet(packet)
        except behavior.PACKET_INVALID_EXCEPTIONS:
            packet_states[run_id] = "fail"
            if stored is not None and stored != {
                "failure": "packet_invalid",
                "state": "fail",
            }:
                errors.append(f"{SEQUENCE}/{run_id}: packet failure record mismatch")
            continue
        except Exception as exc:
            errors.append(
                f"{SEQUENCE}/{run_id}: packet replay implementation failure: "
                f"{str(exc) or type(exc).__name__}"
            )
            packet_states[run_id] = "unavailable"
            continue
        packet_states[run_id] = "pass"
        semantic_digests.append(str(report["semantic_digest_sha256"]))
        inventory = inventories.get(run_id)
        if inventory is not None:
            streams = semantic_packet["streams"]
            candidates = [
                row
                for row in streams["candidate_records"]
                if int(row["proposal_emitted"]) == 1
            ]
            claims = streams["claim_records"]
            commits = streams["commit_records"]
            proposal_payload = {"candidates": candidates, "claims": claims}
            winner_payload = {
                "commits": commits,
                "winning_claims": [row for row in claims if int(row["claim_won"]) == 1],
            }
            expected_proposal = {
                "count": len(candidates),
                "digest": evidence.digest(proposal_payload),
                "records": proposal_payload,
            }
            expected_winner = {
                "count": len(commits),
                "digest": evidence.digest(winner_payload),
                "records": winner_payload,
            }
            expected_overflow = [int(packet[field]) for field in OVERFLOW_FIELDS]
            if (
                inventory[PROPOSAL_MEMBER] != expected_proposal
                or inventory[WINNER_MEMBER] != expected_winner
                or inventory[OVERFLOW_MEMBER] != expected_overflow
            ):
                packet_states[run_id] = "fail"
        if stored is not None and stored != {"report": report, "state": "pass"}:
            errors.append(f"{SEQUENCE}/{run_id}: packet pass record mismatch")
    if len(set(semantic_digests)) > 1:
        for run_id in evidence.CAPTURE_ON_RUNS:
            if packet_states.get(run_id) == "pass":
                packet_states[run_id] = "fail"
    if packet_relation_failure:
        for run_id in evidence.CAPTURE_ON_RUNS:
            if packet_states.get(run_id) == "pass":
                packet_states[run_id] = "fail"
    required_full_inventories = {
        evidence.CAPTURE_OFF_RUN,
        *(
            run_id
            for run_id in evidence.CAPTURE_ON_RUNS
            if packet_states.get(run_id) == "pass"
        ),
    }
    complete = (
        len(base_records) == len(evidence.RUN_IDS)
        and required_full_inventories.issubset(inventories)
        and len(packet_states) == len(evidence.CAPTURE_ON_RUNS)
        and all(state != "unavailable" for state in packet_states.values())
        and not errors
    )
    return SurvivingReplay(
        capture_equal=not base_unequal,
        packets_valid=not packet_relation_failure
        and not any(state == "fail" for state in packet_states.values()),
        complete=complete,
        comparison=comparison,
        errors=tuple(errors),
        evidence_present=bool(
            bases
            or inventories
            or any(state != "unavailable" for state in packet_states.values())
        ),
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
    os.replace(incomplete, final)
    evidence._fsync_directory(final.parent)
    return final, selection


ProbeRunner = Callable[..., dict[str, Any]]
ChildLauncher = Callable[..., int]
MonitorFactory = Callable[..., Monitor]
BundleRevalidator = Callable[[LaunchBundle], tuple[str, ...]]
CheckoutWitnessBuilder = Callable[..., dict[str, Any]]
CheckoutStateReader = Callable[[], tuple[str, str, bool]]


def read_checkout_state() -> tuple[str, str, bool]:
    return (
        _git("rev-parse", "HEAD"),
        _git("rev-parse", "HEAD^{tree}"),
        not bool(_git("status", "--porcelain", "--untracked-files=normal")),
    )


def checkout_hygiene_reasons(*, excluded_roots: Sequence[Path] = ()) -> tuple[str, ...]:
    """Return dirty paths except exact controller-owned output roots.

    ``--untracked-files=all`` is intentional: Git's default directory
    collapsing would make a path-bounded exclusion indistinguishable from a
    repository-wide untracked-directory bypass.
    """
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    entries = completed.stdout.split(b"\0")
    dirty: list[tuple[str, str]] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise ControllerError("git status returned malformed porcelain")
        status = entry[:2].decode("ascii", errors="strict")
        path = entry[3:].decode("utf-8", errors="surrogateescape")
        dirty.append((status, path))
        if ("R" in status or "C" in status) and index < len(entries):
            prior = entries[index]
            index += 1
            if prior:
                dirty.append(
                    (
                        status,
                        prior.decode("utf-8", errors="surrogateescape"),
                    )
                )

    repository = REPO_ROOT.resolve(strict=True)
    exclusions: list[Path] = []
    for root in excluded_roots:
        resolved = root.resolve(strict=False)
        if resolved.is_relative_to(repository):
            exclusions.append(resolved)

    reasons = []
    for status, relative in dirty:
        candidate = (REPO_ROOT / relative).resolve(strict=False)
        if any(
            candidate == excluded or candidate.is_relative_to(excluded)
            for excluded in exclusions
        ):
            continue
        reasons.append(f"{status}:{relative}")
    return tuple(sorted(reasons))


def _append_lifecycle(root: Path, ordinal: int, event: str, **fields: Any) -> None:
    payload = {
        "schema": "h2_controller_lifecycle_event_v1",
        "ordinal": ordinal,
        "event": event,
        **fields,
    }
    path = root / evidence.LIFECYCLE_NAME
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(evidence.canonical_json_bytes(payload) + b"\n")
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short lifecycle-event write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    evidence._fsync_directory(root)


def _consume_authorization(
    grant: AuthorizationGrant,
    *,
    bundle: LaunchBundle,
    ledger: Path,
) -> dict[str, Any]:
    if not isinstance(grant, AuthorizationGrant):
        raise ControllerError("exactly-once authorization is absent")
    current = _load_canonical(grant.path, schema=evidence.AUTHORIZATION_GRANT_SCHEMA)
    execution_domain = evidence.authorization_execution_domain(ledger)
    execution_domain_digest = evidence.digest(execution_domain)
    if current != grant.record or _authorization_mismatch_reasons(
        current,
        bundle,
        execution_domain_digest=execution_domain_digest,
    ):
        raise ControllerError("exactly-once authorization changed before consumption")
    receipt = {
        "schema": evidence.AUTHORIZATION_SCHEMA,
        "authorization_digest": grant.digest,
        "authorization_id": grant.record["authorization_id"],
        "capture_phase": CAPTURE_PHASE,
        "consumed_utc": _utc(),
        "controller_digest": grant.record["controller_digest"],
        "execution_domain": execution_domain_digest,
        "freeze_digest": evidence.freeze_digest(bundle.freeze),
        "instrumentation_head": bundle.head,
        "invocation_id": grant.record["invocation_id"],
        "state": "consumed",
    }
    if set(receipt) != evidence.AUTHORIZATION_CONSUMED_MEMBERS:
        raise ControllerError("internal authorization receipt member drift")
    try:
        evidence.write_document_exclusive(
            ledger,
            f"{grant.record['authorization_id']}.json",
            receipt,
        )
    except FileExistsError as exc:
        raise ControllerError(
            "exactly-once authorization was already consumed"
        ) from exc
    return receipt


def default_authorization_ledger() -> Path:
    """Return the controlled-host/operator ledger shared by every clone."""
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        state_home = Path(configured)
        if not state_home.is_absolute():
            raise ControllerError("XDG_STATE_HOME must be absolute")
    else:
        state_home = Path.home() / ".local" / "state"
    return (state_home / "saccade" / "h2_authorization_consumptions").resolve(
        strict=False
    )


def execute_controller(
    bundle: LaunchBundle,
    *,
    authorization: AuthorizationGrant,
    evidence_parent: Path | None = None,
    authorization_ledger: Path | None = None,
    bound_paths: Sequence[Path] | None = None,
    require_clean_checkout: bool = True,
    launch_probe: ProbeRunner = default_launch_probe,
    launch_child: ChildLauncher = default_child_launcher,
    monitor_factory: MonitorFactory = h0_controller.BoundInputMonitor,
    bundle_revalidator: BundleRevalidator = revalidate_bundle,
    checkout_witness_builder: CheckoutWitnessBuilder = build_checkout_witness,
    checkout_state_reader: CheckoutStateReader = read_checkout_state,
    inherited_environment: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Path, partition.Selection]:
    """Consume and execute one machine-bound Phase-A authorization exactly once."""
    started = clock()
    if not isinstance(authorization, AuthorizationGrant):
        raise ControllerError("exactly-once authorization is absent")
    parent = evidence_parent or (REPO_ROOT / evidence.EVIDENCE_REL)
    final = parent / evidence.phase_a_root_name(bundle.head)
    incomplete = final.with_name(final.name + ".incomplete")
    ledger = (
        authorization_ledger
        if authorization_ledger is not None
        else default_authorization_ledger()
    ).resolve(strict=False)
    execution_domain = evidence.authorization_execution_domain(ledger)
    owned_outputs = (incomplete, final)
    initial_hygiene_reasons = (
        checkout_hygiene_reasons(excluded_roots=owned_outputs)
        if require_clean_checkout
        else ()
    )
    if initial_hygiene_reasons:
        raise ControllerError(
            "Layer-M requires the exact clean head bound by F: "
            + "; ".join(initial_hygiene_reasons)
        )
    parent.mkdir(parents=True, exist_ok=True)
    if (
        final.exists()
        or incomplete.exists()
        or final.is_symlink()
        or incomplete.is_symlink()
    ):
        raise ControllerError("stale final or incomplete H2 evidence root")
    authorization_receipt = _consume_authorization(
        authorization,
        bundle=bundle,
        ledger=ledger,
    )

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
    baseline_head: str | None = None
    baseline_tree: str | None = None
    completed_runs: set[str] = set()
    stop_boundary: dict[str, Any] = {
        "schema": evidence.STOP_BOUNDARY_SCHEMA,
        "checkout_clean": None,
        "checkout_hygiene_reasons": [],
        "completed_utc": None,
        "final_drain_completed": False,
        "linearization": None,
        "monitor_closed": False,
        "monitor_started": False,
        "revalidation_completed_while_monitored": False,
        "revalidation_reasons": [],
        "source_head": None,
        "source_tree": None,
    }
    runtime_revalidation_reasons: list[str] = []
    checkout_reasons: list[str] = []
    certificate_reasons: list[str] = []
    lifecycle_ordinal = 0

    def record_event(event: str, **fields: Any) -> None:
        nonlocal lifecycle_ordinal
        lifecycle_ordinal += 1
        _append_lifecycle(incomplete, lifecycle_ordinal, event, **fields)

    try:
        incomplete.mkdir()
        evidence.write_document(
            incomplete,
            evidence.AUTHORIZATION_GRANT_NAME,
            authorization.record,
        )
        evidence.write_document(
            incomplete,
            evidence.AUTHORIZATION_DOMAIN_NAME,
            execution_domain,
        )
        evidence.write_document(
            incomplete,
            evidence.AUTHORIZATION_NAME,
            authorization_receipt,
        )
        record_event(
            "authorization_consumed",
            authorization_id=authorization_receipt["authorization_id"],
            invocation_id=authorization_receipt["invocation_id"],
        )
        _archive_bundle(incomplete, bundle)
        record_event("archive_created")

        # Install the monitor after the controller-owned archive exists, then
        # re-read every launch input.  The revalidation under the active monitor
        # closes the initial intake -> watch-install TOCTOU window.
        try:
            monitor = monitor_factory(
                tuple(bound_paths)
                if bound_paths is not None
                else _monitor_paths(bundle),
                ignored_roots=(incomplete,),
            )
            stop_boundary["monitor_started"] = True
            record_event("monitor_active")
        except (h0_controller.DriftError, OSError) as exc:
            monitor_failure = exc

        current_head = _git("rev-parse", "HEAD")
        current_tree = _git("rev-parse", "HEAD^{tree}")
        baseline_head = current_head
        baseline_tree = current_tree
        checkout_witness = checkout_witness_builder(
            bundle, current_head=current_head, current_tree=current_tree
        )
        evidence.write_document(
            incomplete, evidence.CHECKOUT_WITNESS_NAME, checkout_witness
        )
        if monitor_failure is not None:
            raise ControllerError(
                f"bound-input monitor could not start: {monitor_failure}"
            )
        assert monitor is not None

        runtime_revalidation_reasons = list(bundle_revalidator(bundle))
        certificate_reasons = list(
            certificate_match_reasons(
                bundle,
                current_head=current_head,
                current_tree=current_tree,
                checkout_witness=checkout_witness,
            )
        )
        checkout_reasons = (
            list(checkout_hygiene_reasons(excluded_roots=owned_outputs))
            if require_clean_checkout
            else []
        )
        record_event("launch_revalidation")
        predicates["layer_p_certificate_matches_freeze"] = not certificate_reasons
        controller_record["certificate_mismatch_reasons"] = list(certificate_reasons)
        controller_record["checkout_hygiene_reasons"] = list(checkout_reasons)
        if runtime_revalidation_reasons:
            mutation_events.extend(
                type(
                    "Mutation",
                    (),
                    {
                        "classification": "post_monitor_revalidation",
                        "mask": 0,
                        "path": reason,
                    },
                )()
                for reason in runtime_revalidation_reasons
            )
            predicates["bound_input_mutated"] = True
        if checkout_reasons:
            execution_result = "unclassified_execution_failure"
            controller_record["failure"] = {
                "reason": "; ".join(checkout_reasons),
                "stage": "execution_checkout_hygiene",
            }
        if (
            not certificate_reasons
            and not runtime_revalidation_reasons
            and not checkout_reasons
        ):
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

        if (
            not certificate_reasons
            and not runtime_revalidation_reasons
            and not checkout_reasons
            and predicates["behavior_probe_equals_freeze"]
        ):
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
                launch_recorded = False

                def on_started() -> None:
                    nonlocal launch_recorded
                    if launch_recorded:
                        raise ControllerError(
                            f"child {run_id} reported process start more than once"
                        )
                    record_event("child_launch", run_id=run_id)
                    launch_recorded = True

                returncode = launch_child(
                    invocation_path,
                    environment,
                    monitor=monitor,
                    started=started,
                    clock=clock,
                    on_started=on_started,
                )
                if not launch_recorded:
                    raise ControllerError(
                        f"child {run_id} returned without a process-start witness"
                    )
                invocation = evidence.load_document(
                    invocation_path.parent,
                    invocation_path.name,
                    schema=child.INVOCATION_SCHEMA,
                )
                if returncode != 0 or invocation.get("state") != "completed":
                    raise ControllerError(f"child {run_id} exited nonzero")
                completed_runs.add(run_id)
                record_event("child_completed", run_id=run_id)
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
        try:
            replay = replay_surviving_evidence(incomplete)
            if replay.evidence_present:
                evidence.write_document(
                    incomplete / evidence.RUNS_DIR / SEQUENCE,
                    evidence.COMPARISON_NAME,
                    replay.comparison,
                )
                predicates["capture_off_on_equal"] = replay.capture_equal
                predicates["packets_valid"] = replay.packets_valid
            predicates["execution_complete"] = (
                replay.complete
                and completed_runs == set(evidence.RUN_IDS)
                and execution_result is None
            )
            if replay.errors:
                predicates["execution_complete"] = False
                execution_result = execution_result or "unclassified_execution_failure"
                controller_record["surviving_replay_errors"] = list(replay.errors)
        except BaseException as exc:
            predicates["execution_complete"] = False
            execution_result = "unclassified_execution_failure"
            controller_record["surviving_replay_errors"] = [
                str(exc) or type(exc).__name__
            ]

        if monitor is not None:
            try:
                final_runtime_reasons = list(bundle_revalidator(bundle))
                final_head, final_tree, injected_checkout_clean = (
                    checkout_state_reader()
                )
                final_checkout_reasons = (
                    list(checkout_hygiene_reasons(excluded_roots=owned_outputs))
                    if require_clean_checkout
                    else ([] if injected_checkout_clean else ["checkout is dirty"])
                )
                if final_head != baseline_head or final_tree != baseline_tree:
                    final_checkout_reasons.append(
                        "execution checkout head/tree moved before stop linearization"
                    )
                runtime_revalidation_reasons.extend(final_runtime_reasons)
                checkout_reasons.extend(final_checkout_reasons)
                record_event("monitored_final_revalidation")
                stop_boundary.update(
                    {
                        "checkout_clean": not final_checkout_reasons,
                        "checkout_hygiene_reasons": final_checkout_reasons,
                        "revalidation_completed_while_monitored": True,
                        "revalidation_reasons": final_runtime_reasons,
                        "source_head": final_head,
                        "source_tree": final_tree,
                    }
                )
            except BaseException as exc:
                final_runtime_reasons = [
                    "monitored bound-input revalidation failed: "
                    f"{str(exc) or type(exc).__name__}"
                ]
                runtime_revalidation_reasons.extend(final_runtime_reasons)
                stop_boundary.update(
                    {
                        "checkout_clean": False,
                        "checkout_hygiene_reasons": [
                            "checkout hygiene could not be established"
                        ],
                        "revalidation_reasons": final_runtime_reasons,
                        "source_head": baseline_head or "",
                        "source_tree": baseline_tree or "",
                    }
                )
            if final_runtime_reasons:
                mutation_events.extend(
                    type(
                        "Mutation",
                        (),
                        {
                            "classification": "monitored_revalidation",
                            "mask": 0,
                            "path": reason,
                        },
                    )()
                    for reason in final_runtime_reasons
                )
            if stop_boundary["checkout_hygiene_reasons"]:
                predicates["execution_complete"] = False
                execution_result = execution_result or "unclassified_execution_failure"
            try:
                mutation_events.extend(monitor.drain())
                stop_boundary["final_drain_completed"] = True
                record_event("final_monitor_drain")
                if (
                    stop_boundary["revalidation_completed_while_monitored"]
                    and not mutation_events
                    and stop_boundary["checkout_clean"] is True
                ):
                    stop_boundary["linearization"] = "clean_final_drain"
            except BaseException as exc:
                predicates["execution_complete"] = False
                execution_result = "unclassified_execution_failure"
                controller_record["failure"] = {
                    "reason": str(exc) or type(exc).__name__,
                    "stage": "bound_input_monitor_final_drain",
                }
            try:
                monitor.close()
                stop_boundary["monitor_closed"] = True
            except BaseException as exc:
                predicates["execution_complete"] = False
                execution_result = "unclassified_execution_failure"
                controller_record["failure"] = {
                    "reason": str(exc) or type(exc).__name__,
                    "stage": "bound_input_monitor_close",
                }
            stop_boundary["completed_utc"] = _utc()
            predicates["bound_input_mutated"] = bool(mutation_events)

    evidence.write_document(incomplete, evidence.STOP_BOUNDARY_NAME, stop_boundary)
    record_event("stop_boundary_recorded")
    controller_record["checkout_hygiene_reasons"] = list(checkout_reasons)
    controller_record["predicate_ownership"] = {
        "execution_checkout_hygiene": {
            "passed": not checkout_reasons,
            "reasons": list(checkout_reasons),
        },
        "layer_p_certificate_matches_freeze": {
            "passed": not certificate_reasons,
            "reasons": list(certificate_reasons),
        },
        "monitored_runtime_inputs": {
            "mutated": bool(mutation_events),
            "revalidation_reasons": list(runtime_revalidation_reasons),
        },
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
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument(
        "--invocation-id",
        required=True,
        help="full 64-hex invocation identity bound by the owner authorization",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        bundle = load_bundle(
            freeze_path=args.freeze,
            certificate_path=args.layer_p_certificate,
            reference_probe_path=args.reference_probe,
            runtime_manifest_path=args.runtime_inputs,
            published_identity_path=args.published_identity,
        )
        authorization = load_authorization(
            args.authorization,
            bundle,
            invocation_id=args.invocation_id,
            authorization_ledger=default_authorization_ledger(),
        )
        root, selection = execute_controller(bundle, authorization=authorization)
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
