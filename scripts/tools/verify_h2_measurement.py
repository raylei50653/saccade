#!/usr/bin/env python3
"""Independent verifier for an H2 Layer-M evidence root.

Independent means it recomputes rather than reads, and it recomputes **two**
things, not one:

  * **the terminal** — re-selected from the archived observation through
    `h2_terminal_partition`, with the A7.6 comparison rebuilt from the archived
    policy inventories and every surviving capture-on packet re-verified through
    the packet verifier;
  * **the right to have spent `S_B`** — the § C3.6 admission gate is recomputed
    from the bound Phase-A evidence root, the two freeze records, the archived
    Layer-P certificate and the prior-attempt chain, and must be bit-identical to
    what the controller recorded. A verifier that recomputed the terminal while
    trusting `admission.json` would let a Phase-B archive assert its own
    eligibility, which is the one claim the gate exists to deny.

**Nothing in this file is a comparison or terminal authority.** § 6 forbids H2
from introducing comparison vocabulary of its own and § C3.9 pins why a
`plumbing_only` file must hold none: it can be edited without moving an axis, so
a rule stated here could change while `identity_semantics` stood still. The A7.6
member sets come from `h2_behavioral_identity`, the verify classes, surface-ban
terminals and repair vocabulary from `h2_terminal_partition`, the axis names from
`build_runtime_identity`, and the packet predicates from H0's own
`verify_capture` / `canonical_semantic_packet` / `_verify_policy_inventory`.

**Surviving evidence accumulates monotonically.** A missing artifact may reduce
what can be checked; it may never erase what was already found. Replay is
therefore per run — `pass` / `fail` / `unavailable` — and every comparison that
*can* be made from the inventories present is made, whether or not the sequence
is complete and whether or not the controller's own `comparison.json` survived.
Otherwise killing a process after the first invalid packet would launder a
terminal-3 ban into a re-attemptable terminal 4 (§ C3.5.1).

Usage:
  uv run python scripts/tools/verify_h2_measurement.py <evidence root>
  uv run python scripts/tools/verify_h2_measurement.py --class envelope <root>
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h0_phase_a as h0_verifier  # noqa: E402  (imported, never modified)
from build_runtime_identity import ALL_COORDINATE_AXES, IDENTITY_SCHEMA  # noqa: E402
from export_headline_bridge_decision_trace import (  # noqa: E402
    OVERFLOW_KEYS,
    STREAMS,
    UNIVERSE_OVERFLOW_KEYS,
    UNIVERSE_STREAMS,
    canonical_semantic_packet,
)
from h2_behavioral_identity import (  # noqa: E402
    A76_EQUALITY_MEMBERS,
    A76_OVERFLOW_MEMBER,
    A76_OVERFLOW_ZERO_VECTOR,
    A76_PROJECTION_MEMBERS,
)
from h2_runtime_inputs import SCHEMA as RUNTIME_INPUT_SCHEMA, digest  # noqa: E402
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

VERIFIER_SCHEMA = "h2_measurement_verifier_v1"

# Derived, never restated. The non-terminal progression is *defined* as the one
# result the partition maps to no terminal; spelling it out here would put a
# ruler fact in a `plumbing_only` file (§ C3.9).
NON_TERMINAL_RESULT = next(
    result
    for result, terminal in partition.RESULT_TO_TERMINAL.items()
    if terminal is None
)
EXECUTION_INVALID_TERMINAL = partition.EXECUTION_INVALID_TERMINAL
FULL_COMMIT_TERMINAL = partition.TERMINALS[4].name

# The capture ABI's own overflow fields, in the capture ABI's own order, taken
# from the exporter rather than transcribed (§ 6).
OVERFLOW_FIELDS: tuple[str, ...] = tuple(
    OVERFLOW_KEYS[stream] for stream in STREAMS
) + tuple(UNIVERSE_OVERFLOW_KEYS[stream] for stream in UNIVERSE_STREAMS)

# Per-run packet outcomes. `unavailable` is not a third kind of failure: it is
# the absence of an artifact, and it never cancels a `fail` found elsewhere.
PASS, FAIL, UNAVAILABLE = "pass", "fail", "unavailable"


class VerificationError(RuntimeError):
    """The archive does not support what it records. Always fail-closed."""


class CorpusError(VerificationError):
    """A defect visible only across roots: chain completeness, order, class.

    A subclass because it *is* a verification failure — § C3.6(e) makes the chain
    part of one root's admissibility — while staying separately catchable for the
    corpus checker, which reports it as a corpus verdict rather than as a defect
    of the root it was raised on.
    """


class RunReplay(NamedTuple):
    run_id: str
    inventory_present: bool
    packet_state: str


class SequenceReplay(NamedTuple):
    sequence: str
    runs: tuple[RunReplay, ...]
    complete: bool
    inequality_found: bool
    packet_relation_failure: bool

    @property
    def packet_states(self) -> tuple[str, ...]:
        return tuple(
            run.packet_state
            for run in self.runs
            if run.run_id in evidence.CAPTURE_ON_RUNS
        )


class Replay(NamedTuple):
    sequences: tuple[SequenceReplay, ...]
    complete: bool

    @property
    def perturbation_observed(self) -> bool:
        return any(item.inequality_found for item in self.sequences)

    @property
    def invalid_packet_observed(self) -> bool:
        return any(
            FAIL in item.packet_states or item.packet_relation_failure
            for item in self.sequences
        )

    @property
    def all_equal(self) -> bool:
        return self.complete and not self.perturbation_observed

    @property
    def all_packets_pass(self) -> bool:
        return self.complete and all(
            not item.packet_relation_failure
            and item.packet_states == (PASS,) * len(evidence.CAPTURE_ON_RUNS)
            for item in self.sequences
        )


# -- structure ------------------------------------------------------------- #


def _root_name(root: Path) -> evidence.RootName:
    if root.is_symlink() or not root.is_dir():
        raise VerificationError(f"evidence root is not a physical directory: {root}")
    try:
        return evidence.parse_root_name(root.name)
    except evidence.EvidenceError as exc:
        raise VerificationError(str(exc)) from exc


def _load(root: Path, name: str, *, schema: str | None = None) -> dict[str, Any]:
    try:
        return evidence.load_document(root, name, schema=schema)
    except evidence.EvidenceError as exc:
        raise VerificationError(str(exc)) from exc


def _inventory(root: Path) -> dict[str, str]:
    try:
        return evidence.verify_checksum_inventory(root)
    except (evidence.EvidenceError, OSError) as exc:
        raise VerificationError(f"checksum inventory rejected: {exc}") from exc


def _freeze(root: Path, name: evidence.RootName) -> dict[str, Any]:
    freeze = _load(root, evidence.FREEZE_NAME, schema=evidence.FREEZE_SCHEMA)
    recomputed = evidence.freeze_digest(freeze)
    if name.freeze_digest is not None and recomputed != name.freeze_digest:
        # § C3.1: the root name *is* the freeze identity, recomputed rather than
        # trusted, so two attempts cannot share a root even at an equal head.
        raise VerificationError(
            "evidence root name does not match the recorded freeze record: "
            f"name {name.freeze_digest}, recomputed {recomputed}"
        )
    return freeze


def _authorization(
    root: Path,
    phase: str,
    *,
    freeze: Mapping[str, Any],
    name: evidence.RootName,
) -> dict[str, Any]:
    if phase == "b":
        # § C3.5.1 step 5: this record's durable write *is* the consumption of
        # S_B, so a Phase-B root without it never spent an authorization.
        return _load(
            root, evidence.AUTHORIZATION_NAME, schema=evidence.AUTHORIZATION_SCHEMA
        )
    receipt = _load(
        root, evidence.AUTHORIZATION_NAME, schema=evidence.AUTHORIZATION_SCHEMA
    )
    surfaces = freeze.get("executed_surfaces")
    controller_digest = (
        surfaces.get("scripts/tools/run_h2_measurement.py")
        if isinstance(surfaces, Mapping)
        else None
    )
    if (
        set(receipt) != evidence.AUTHORIZATION_CONSUMED_MEMBERS
        or not _hex(receipt.get("authorization_digest"), 64)
        or not _hex(receipt.get("authorization_id"), 64)
        or not _hex(receipt.get("invocation_id"), 64)
        or receipt.get("capture_phase") != evidence.CAPTURE_PHASE[phase]
        or receipt.get("instrumentation_head") != name.i40
        or receipt.get("freeze_digest") != evidence.freeze_digest(freeze)
        or receipt.get("controller_digest") != controller_digest
        or receipt.get("state") != "consumed"
        or not isinstance(receipt.get("consumed_utc"), str)
        or not receipt["consumed_utc"]
    ):
        raise VerificationError(
            "Phase-A authorization consumption record is absent, malformed, "
            "or bound to another head/freeze/controller/invocation"
        )
    return receipt


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.SubprocessError as exc:
        raise VerificationError(
            f"archived source commit is unavailable: {exc}"
        ) from exc
    return completed.stdout.strip()


def _git_bytes(*args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
    except subprocess.SubprocessError as exc:
        raise VerificationError(
            f"archived source commit is unavailable: {exc}"
        ) from exc
    return completed.stdout


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _commit_content_axes(head: str) -> dict[str, dict[str, Any]]:
    """Rebuild all content axes from the archived commit, not controller code."""
    raw = _git("ls-tree", "-r", "-z", "--full-tree", head)
    classified: dict[str, list[dict[str, str]]] = {
        "decision_relevant": [],
        "identity_semantics": [],
        "plumbing_only": [],
    }
    for entry in raw.split("\0"):
        if not entry:
            continue
        try:
            metadata, path = entry.split("\t", 1)
            _mode, object_type, blob = metadata.split(" ", 2)
        except ValueError as exc:
            raise VerificationError("git tree listing is malformed") from exc
        if object_type != "blob" or path.endswith((".md", ".rst", ".txt")):
            continue
        path_class = path_partition.classify(path)
        if path_class in classified:
            classified[path_class].append({"blob": blob, "path": path})
    return {
        path_class: {
            "digest": evidence.digest(members),
            "file_count": len(members),
            "files": members,
        }
        for path_class, members in classified.items()
    }


def _checkout_witness(root: Path, head: str) -> dict[str, Any]:
    witness = _load(
        root,
        evidence.CHECKOUT_WITNESS_NAME,
        schema=evidence.CHECKOUT_WITNESS_SCHEMA,
    )
    expected_tree = _git("rev-parse", f"{head}^{{tree}}")
    expected_axes = _commit_content_axes(head)
    if (
        set(witness)
        != {
            "axes",
            "build_dir",
            "repository_root",
            "schema",
            "source_head",
            "source_tree",
        }
        or witness.get("source_head") != head
        or witness.get("source_tree") != expected_tree
        or witness.get("axes") != expected_axes
        or not isinstance(witness.get("repository_root"), str)
        or not Path(witness["repository_root"]).is_absolute()
        or not isinstance(witness.get("build_dir"), str)
        or not Path(witness["build_dir"]).is_absolute()
    ):
        raise VerificationError(
            "checkout identity witness differs from the independently rebuilt "
            "source tree/content axes"
        )
    return witness


def _bound_path(value: Any, repository_root: str) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(repository_root) / path
    return os.path.normpath(path.as_posix())


def _phase_a_freeze_bindings(
    root: Path,
    name: evidence.RootName,
    freeze: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    published: Mapping[str, Any],
) -> None:
    """Reconstruct the formal Phase-A freeze from archived primary artifacts."""
    head = name.i40
    runtime_binding = freeze.get("runtime_inputs")
    build_artifacts = runtime_manifest.get("build_artifacts")
    executed = freeze.get("executed_surfaces")
    expected_surfaces = {
        path: hashlib.sha256(_git_bytes("show", f"{head}:{path}")).hexdigest()
        for path in evidence.PHASE_A_EXECUTED_SURFACE_PATHS
    }
    expected_capture_abi = {
        "path": evidence.PHASE_A_CAPTURE_ABI_PATH,
        "sha256": hashlib.sha256(
            _git_bytes("show", f"{head}:{evidence.PHASE_A_CAPTURE_ABI_PATH}")
        ).hexdigest(),
    }
    if (
        set(freeze) != evidence.PHASE_A_FREEZE_MEMBERS
        or freeze.get("capture_phase") != evidence.CAPTURE_PHASE["a"]
        or freeze.get("instrumentation_head") != head
        or not _hex(freeze.get("selected_base"), 40)
        or freeze.get("equivalence") != "unproven"
        or not isinstance(freeze.get("layer_p_certificate"), Mapping)
        or freeze["layer_p_certificate"].get("schema") != CERTIFICATE_SCHEMA
        or not _hex(freeze["layer_p_certificate"].get("digest"), 64)
        or freeze.get("reference_probe")
        != {
            "schema": behavior.RESULT_SCHEMA,
            "file_digest": evidence.sha256_file(root / evidence.REFERENCE_PROBE_NAME),
        }
        or not isinstance(runtime_binding, Mapping)
        or not isinstance(build_artifacts, Mapping)
        or runtime_binding
        != {
            "schema": RUNTIME_INPUT_SCHEMA,
            "file_digest": evidence.sha256_file(root / evidence.RUNTIME_INPUTS_NAME),
            "coordinate_digest": runtime_manifest.get("coordinate_digest"),
            "full_digest": runtime_manifest.get("full_digest"),
            "build_artifact_digest": build_artifacts.get("digest"),
        }
        or runtime_manifest.get("schema") != RUNTIME_INPUT_SCHEMA
        or freeze.get("published_identity")
        != {
            "schema": IDENTITY_SCHEMA,
            "file_digest": evidence.sha256_file(
                root / evidence.PUBLISHED_IDENTITY_NAME
            ),
        }
        or freeze.get("capture_abi") != expected_capture_abi
        or not isinstance(executed, Mapping)
        or dict(executed) != expected_surfaces
        or freeze.get("run_plan")
        != {
            "sequence": evidence.expected_sequences("a")[0],
            "run_ids": list(evidence.RUN_IDS),
        }
        or reference.get("digest") != freeze.get("probe")
        or published.get("coordinate") != freeze.get("coordinate")
        or not isinstance(published.get("probe"), Mapping)
        or published["probe"].get("digest") != freeze.get("probe")
    ):
        raise VerificationError(
            "Phase-A freeze differs from independent primary-artifact reconstruction"
        )


def _verify_lifecycle(root: Path, observation: Mapping[str, Any]) -> None:
    path = root / evidence.LIFECYCLE_NAME
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VerificationError("controller lifecycle event log is absent") from exc
    if not raw.endswith(b"\n"):
        raise VerificationError("controller lifecycle event log is not durable JSONL")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerificationError(f"lifecycle event {number} is unreadable") from exc
        if (
            not isinstance(row, dict)
            or evidence.canonical_json_bytes(row) != line
            or row.get("schema") != "h2_controller_lifecycle_event_v1"
            or row.get("ordinal") != number
            or not isinstance(row.get("event"), str)
        ):
            raise VerificationError(f"lifecycle event {number} is malformed")
        rows.append(row)
    names = [str(row["event"]) for row in rows]
    if names[:2] != ["authorization_consumed", "archive_created"]:
        raise VerificationError(
            "authorization consumption does not precede archive creation"
        )
    if "child_launch" in names and (
        "monitor_active" not in names
        or "launch_revalidation" not in names
        or names.index("child_launch") < names.index("launch_revalidation")
    ):
        raise VerificationError("a child launch precedes monitored launch revalidation")
    launches = [row.get("run_id") for row in rows if row.get("event") == "child_launch"]
    completions = [
        row.get("run_id") for row in rows if row.get("event") == "child_completed"
    ]
    if launches != list(evidence.RUN_IDS[: len(launches)]):
        raise VerificationError(
            "child launch lifecycle order differs from the run plan"
        )
    if completions != list(evidence.RUN_IDS[: len(completions)]):
        raise VerificationError(
            "child completion lifecycle order differs from the run plan"
        )
    if observation.get("execution_complete") is True:
        expected = [
            "authorization_consumed",
            "archive_created",
            "monitor_active",
            "launch_revalidation",
        ]
        for _run_id in evidence.RUN_IDS:
            expected.extend(("child_launch", "child_completed"))
        expected.extend(
            (
                "monitored_final_revalidation",
                "final_monitor_drain",
                "stop_boundary_recorded",
            )
        )
        if names != expected:
            raise VerificationError(
                "completed execution lifecycle differs from launch-to-stop order"
            )


def _phase_a_launch_records(
    root: Path,
    name: evidence.RootName,
    freeze: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently cross-check the controller's archived terminal-1 inputs."""
    certificate = _load(root, evidence.CERTIFICATE_NAME, schema=CERTIFICATE_SCHEMA)
    reference = _load(
        root, evidence.REFERENCE_PROBE_NAME, schema=behavior.RESULT_SCHEMA
    )
    runtime_manifest = _load(root, evidence.RUNTIME_INPUTS_NAME)
    published = _load(root, evidence.PUBLISHED_IDENTITY_NAME, schema=IDENTITY_SCHEMA)
    mutation = _load(root, evidence.MUTATION_NAME, schema=evidence.MUTATION_SCHEMA)
    stop = _load(
        root, evidence.STOP_BOUNDARY_NAME, schema=evidence.STOP_BOUNDARY_SCHEMA
    )
    controller = _load(
        root, evidence.CONTROLLER_NAME, schema=evidence.CONTROLLER_SCHEMA
    )
    checkout = _checkout_witness(root, name.i40)
    _phase_a_freeze_bindings(
        root,
        name,
        freeze,
        reference=reference,
        runtime_manifest=runtime_manifest,
        published=published,
    )

    binding = freeze.get("layer_p_certificate")
    coordinate = freeze.get("coordinate")
    probe_digest = freeze.get("probe")
    build_artifacts = runtime_manifest.get("build_artifacts")
    reference_witness = reference.get("build_witness")
    checks = (
        (
            isinstance(binding, Mapping)
            and binding.get("schema") == CERTIFICATE_SCHEMA
            and binding.get("digest") == evidence.digest(certificate),
            "certificate digest differs from F",
        ),
        (
            freeze.get("instrumentation_head") == name.i40
            and certificate.get("source_head") == name.i40,
            "certificate/freeze/current head differ",
        ),
        (
            certificate.get("source_tree") == checkout["source_tree"],
            "certificate tree differs from the execution checkout",
        ),
        (
            _hex(certificate.get("selected_base"), 40)
            and certificate.get("selected_base") == freeze.get("selected_base"),
            "certificate selected base is not exact or differs from F",
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
            isinstance(coordinate, Mapping)
            and certificate.get("decision_relevant_digest")
            == coordinate.get("implementation")
            == checkout["axes"]["decision_relevant"]["digest"],
            "certificate implementation digest differs from F/checkout",
        ),
        (
            isinstance(coordinate, Mapping)
            and certificate.get("identity_semantics_digest")
            == coordinate.get("identity_semantics")
            == checkout["axes"]["identity_semantics"]["digest"],
            "certificate identity-semantics digest differs from F/checkout",
        ),
        (
            certificate.get("plumbing_set_digest")
            == checkout["axes"]["plumbing_only"]["digest"],
            "certificate plumbing-set digest differs from the execution checkout",
        ),
        (
            certificate.get("published_coordinate") == coordinate,
            "certificate coordinate differs from F",
        ),
        (
            certificate.get("behavior_probe") == probe_digest
            and certificate.get("published_probe") == probe_digest,
            "certificate probe differs from F",
        ),
        (
            certificate.get("runtime_input_coordinate_digest")
            == runtime_manifest.get("coordinate_digest"),
            "certificate runtime-input coordinate differs from the manifest",
        ),
        (
            certificate.get("runtime_input_full_digest")
            == runtime_manifest.get("full_digest"),
            "certificate runtime-input full digest differs from the manifest",
        ),
        (
            isinstance(build_artifacts, Mapping)
            and certificate.get("build_artifact_digest")
            == build_artifacts.get("digest"),
            "certificate build artifacts differ from the manifest",
        ),
        (
            certificate.get("runtime_input_manifest_file_digest")
            == evidence.sha256_file(root / evidence.RUNTIME_INPUTS_NAME),
            "runtime-input file digest differs from the certificate",
        ),
        (
            certificate.get("probe_result_file_digest")
            == evidence.sha256_file(root / evidence.REFERENCE_PROBE_NAME),
            "reference-probe file digest differs from the certificate",
        ),
        (
            certificate.get("published_identity_file_digest")
            == evidence.sha256_file(root / evidence.PUBLISHED_IDENTITY_NAME),
            "published-identity file digest differs from the certificate",
        ),
        (
            reference.get("digest") == probe_digest
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
            isinstance(reference_witness, Mapping)
            and isinstance(build_artifacts, Mapping)
            and reference_witness.get("digest") == build_artifacts.get("digest")
            and certificate.get("build_witness") == reference_witness,
            "certificate/reference build witness differs from the manifest",
        ),
        (
            published.get("coordinate") == coordinate
            and isinstance(published.get("probe"), Mapping)
            and published["probe"].get("digest") == probe_digest
            and isinstance(published.get("equivalence"), Mapping)
            and published["equivalence"].get("state") == "unproven"
            and published.get("publication_complete") is True,
            "published identity does not support F",
        ),
        (
            isinstance(build_artifacts, Mapping)
            and _bound_path(certificate.get("build_dir"), checkout["repository_root"])
            == _bound_path(
                build_artifacts.get("build_dir"), checkout["repository_root"]
            )
            == _bound_path(checkout["build_dir"], checkout["repository_root"]),
            "selected build directory differs from the runtime manifest",
        ),
    )
    independent_mismatch_reasons = [reason for passed, reason in checks if not passed]
    certificate_match = not independent_mismatch_reasons
    recorded_certificate_match = observation.get("layer_p_certificate_matches_freeze")
    mismatch_reasons = controller.get("certificate_mismatch_reasons")
    if recorded_certificate_match is not certificate_match:
        raise VerificationError(
            "recorded Layer-P certificate match disagrees with the archived "
            "freeze/certificate/content bindings and independent Git-tree "
            "recomputation"
        )
    if (
        not isinstance(mismatch_reasons, list)
        or mismatch_reasons != independent_mismatch_reasons
    ):
        raise VerificationError(
            "controller certificate reasons disagree with the independently "
            "recomputed predicate"
        )

    launch_path = root / evidence.LAUNCH_PROBE_NAME
    launch_matches: bool | None = None
    if launch_path.is_file():
        launch = _load(root, evidence.LAUNCH_PROBE_NAME, schema=behavior.RESULT_SCHEMA)
        launch_witness = launch.get("build_witness")
        launch_matches = (
            launch.get("digest") == probe_digest
            and isinstance(launch_witness, Mapping)
            and isinstance(build_artifacts, Mapping)
            and launch_witness.get("digest") == build_artifacts.get("digest")
        )
        if observation.get("behavior_probe_equals_freeze") is not launch_matches:
            raise VerificationError(
                "recorded launch-probe predicate differs from the archived probe"
            )
    elif recorded_certificate_match is True and observation.get(
        "execution_result"
    ) not in {
        result
        for result, terminal in partition.RESULT_TO_TERMINAL.items()
        if terminal == EXECUTION_INVALID_TERMINAL
    }:
        raise VerificationError(
            "a certificate-admitted Phase-A archive has no launch-time probe"
        )

    events = mutation.get("events")
    if (
        not isinstance(events, list)
        or not isinstance(mutation.get("mutated"), bool)
        or mutation["mutated"] is not bool(events)
        or observation.get("bound_input_mutated") is not mutation["mutated"]
    ):
        raise VerificationError(
            "recorded bound-input predicate differs from the mutation record"
        )
    stop_reasons = stop.get("revalidation_reasons")
    checkout_reasons = stop.get("checkout_hygiene_reasons")
    stop_event_reasons = [
        event.get("path")
        for event in events
        if isinstance(event, Mapping)
        and event.get("classification") == "monitored_revalidation"
    ]
    all_runtime_revalidation_reasons = [
        event.get("path")
        for event in events
        if isinstance(event, Mapping)
        and event.get("classification")
        in {"post_monitor_revalidation", "monitored_revalidation"}
    ]
    expected_stop_members = {
        "checkout_clean",
        "checkout_hygiene_reasons",
        "completed_utc",
        "final_drain_completed",
        "linearization",
        "monitor_closed",
        "monitor_started",
        "revalidation_completed_while_monitored",
        "revalidation_reasons",
        "schema",
        "source_head",
        "source_tree",
    }
    clean_linearization = (
        stop.get("monitor_started") is True
        and stop.get("revalidation_completed_while_monitored") is True
        and stop.get("final_drain_completed") is True
        and stop.get("checkout_clean") is True
        and not checkout_reasons
        and not stop_reasons
        and not events
    )
    if (
        set(stop) != expected_stop_members
        or not isinstance(stop.get("monitor_started"), bool)
        or not isinstance(stop.get("monitor_closed"), bool)
        or not isinstance(stop.get("revalidation_completed_while_monitored"), bool)
        or not isinstance(stop.get("final_drain_completed"), bool)
        or not isinstance(checkout_reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in checkout_reasons)
        or not isinstance(stop_reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in stop_reasons)
        or stop.get("linearization")
        != ("clean_final_drain" if clean_linearization else None)
        or (bool(stop_reasons) and mutation.get("mutated") is not True)
        or (bool(stop_reasons) and stop_event_reasons != stop_reasons)
        or (
            stop["monitor_started"]
            and (
                not isinstance(stop.get("completed_utc"), str)
                or not stop["completed_utc"]
                or not isinstance(stop.get("checkout_clean"), bool)
                or not isinstance(stop.get("source_head"), str)
                or len(stop["source_head"]) != 40
                or any(char not in "0123456789abcdef" for char in stop["source_head"])
                or not isinstance(stop.get("source_tree"), str)
                or len(stop["source_tree"]) != 40
                or any(char not in "0123456789abcdef" for char in stop["source_tree"])
            )
        )
        or (
            not stop["monitor_started"]
            and (
                stop["monitor_closed"] is not False
                or stop["revalidation_completed_while_monitored"] is not False
                or stop["final_drain_completed"] is not False
                or bool(stop_reasons)
                or bool(checkout_reasons)
                or any(
                    stop.get(field) is not None
                    for field in (
                        "checkout_clean",
                        "completed_utc",
                        "linearization",
                        "source_head",
                        "source_tree",
                    )
                )
            )
        )
    ):
        raise VerificationError("measurement stop boundary is malformed")
    if (
        stop["monitor_started"]
        and stop["revalidation_completed_while_monitored"] is not True
        and not stop_reasons
    ):
        raise VerificationError(
            "started monitor has no completed revalidation or failure reason"
        )
    if (
        stop.get("checkout_clean") is True
        and stop["revalidation_completed_while_monitored"] is not True
    ):
        raise VerificationError(
            "clean checkout state was recorded without monitored revalidation"
        )
    if (
        stop["revalidation_completed_while_monitored"] is True
        and stop.get("checkout_clean") is False
        and (not checkout_reasons)
    ):
        raise VerificationError(
            "dirty checkout after monitored revalidation has no checkout-hygiene reason"
        )
    if stop.get("linearization") == "clean_final_drain" and (
        stop.get("source_head") != checkout["source_head"]
        or stop.get("source_tree") != checkout["source_tree"]
    ):
        raise VerificationError(
            "clean final-drain source identity differs from the checkout witness"
        )
    ownership = controller.get("predicate_ownership")
    if not isinstance(ownership, Mapping) or set(ownership) != {
        "execution_checkout_hygiene",
        "layer_p_certificate_matches_freeze",
        "monitored_runtime_inputs",
    }:
        raise VerificationError("controller predicate ownership record is malformed")
    checkout_owner = ownership["execution_checkout_hygiene"]
    certificate_owner = ownership["layer_p_certificate_matches_freeze"]
    runtime_owner = ownership["monitored_runtime_inputs"]
    controller_checkout_reasons = controller.get("checkout_hygiene_reasons")
    if (
        not isinstance(checkout_owner, Mapping)
        or not isinstance(checkout_owner.get("reasons"), list)
        or checkout_owner.get("passed") is not (not checkout_owner["reasons"])
        or not isinstance(controller_checkout_reasons, list)
        or checkout_owner["reasons"] != controller_checkout_reasons
        or any(reason not in checkout_owner["reasons"] for reason in checkout_reasons)
        or not isinstance(certificate_owner, Mapping)
        or certificate_owner.get("passed") is not certificate_match
        or certificate_owner.get("reasons") != mismatch_reasons
        or not isinstance(runtime_owner, Mapping)
        or runtime_owner.get("mutated") is not mutation["mutated"]
        or not isinstance(runtime_owner.get("revalidation_reasons"), list)
        or runtime_owner["revalidation_reasons"] != all_runtime_revalidation_reasons
    ):
        raise VerificationError(
            "controller predicate ownership disagrees with independently "
            "recomputed certificate, checkout, or runtime-input state"
        )
    _verify_lifecycle(root, observation)
    terminal_four_results = {
        result
        for result, terminal in partition.RESULT_TO_TERMINAL.items()
        if terminal == EXECUTION_INVALID_TERMINAL
    }
    if (
        observation.get("execution_result") not in terminal_four_results
        and not stop_reasons
        and not events
        and (
            stop["monitor_started"] is not True
            or stop["monitor_closed"] is not True
            or stop["revalidation_completed_while_monitored"] is not True
            or stop["final_drain_completed"] is not True
            or stop["checkout_clean"] is not True
            or stop["linearization"] != "clean_final_drain"
            or stop.get("source_head") != checkout["source_head"]
            or stop.get("source_tree") != checkout["source_tree"]
        )
    ):
        raise VerificationError(
            "clean non-execution-invalid archive has no active-monitor "
            "revalidation/final-drain boundary"
        )
    if (
        controller.get("instrumentation_head") != name.i40
        or controller.get("capture_phase") != evidence.CAPTURE_PHASE[name.phase]
        or controller.get("ordered_runs") != list(evidence.RUN_IDS)
        or controller.get("sequence") not in evidence.expected_sequences(name.phase)
    ):
        raise VerificationError("controller record disagrees with the Phase-A plan")
    return controller


def classify(root: Path) -> str:
    """Exactly one § C3.5.1 class per root; an unclassifiable root is a defect.

    Lives here rather than in the corpus checker because the verifier needs it
    too: a prior attempt must be verified *in its class* before the admission
    gate that binds it can be recomputed.
    """
    name = _root_name(root)
    admission = root / evidence.ADMISSION_NAME
    authorization = root / evidence.AUTHORIZATION_NAME
    terminal = root / evidence.TERMINAL_NAME
    if admission.is_file():
        if name.phase != "b":
            raise VerificationError(
                f"{root.name}: a phase-A root carries an admission verdict; the "
                "§ C3.6 gate is phase-B only"
            )
        record = _load(root, evidence.ADMISSION_NAME, schema=evidence.ADMISSION_SCHEMA)
        try:
            verdict = partition.evaluate_admission(record, phase="b")
        except partition.PartitionError as exc:
            raise VerificationError(f"{root.name}: admission record rejected: {exc}")
        if not verdict.admitted:
            if authorization.is_file():
                raise VerificationError(
                    f"{root.name}: S_B was consumed after a refused admission gate "
                    "(§ C3.5.1 steps 4-5)"
                )
            return partition.INADMISSIBLE_CLASS
    elif name.phase == "b":
        raise VerificationError(
            f"{root.name}: a phase-B root records no § C3.6 admission verdict"
        )
    if name.phase == "b" and not authorization.is_file():
        raise VerificationError(
            f"{root.name}: a phase-B root passed admission but records no "
            "authorization_consumed write (§ C3.5.1 step 5)"
        )
    if not terminal.is_file():
        return "unterminated"
    if (root / evidence.MANIFEST_NAME).is_file() and (
        root / evidence.OBSERVATION_NAME
    ).is_file():
        return "complete"
    return "envelope"


# -- A7.6 comparison and packet replay, monotone under missing artifacts ---- #


def _inventories(root: Path, sequence: str) -> dict[str, dict[str, Any]]:
    """Load every policy inventory that survived — never all-or-nothing."""
    present: dict[str, dict[str, Any]] = {}
    for run_id in evidence.RUN_IDS:
        directory = evidence.run_dir(root, sequence, run_id)
        if not (directory / evidence.POLICY_INVENTORY_NAME).is_file():
            continue
        inventory = _load(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            schema=evidence.POLICY_INVENTORY_SCHEMA,
        )
        try:
            # Consumed verbatim: A7.6's shapes are H0's, and a re-typed copy here
            # would be exactly the vocabulary § 6 forbids.
            h0_verifier._verify_policy_inventory(run_id, inventory)
        except h0_verifier.VerificationError as exc:
            raise VerificationError(f"{sequence}/{run_id}: {exc}") from exc
        present[run_id] = inventory
    return present


def _base_inventories(
    root: Path, sequence: str
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load every committed base member, including MOT-only survivors."""
    present: dict[str, dict[str, Any]] = {}
    base_records: set[str] = set()
    expected = {*A76_EQUALITY_MEMBERS, "schema"}
    for run_id in evidence.RUN_IDS:
        directory = evidence.run_dir(root, sequence, run_id)
        mot_path = directory / f"{sequence}.txt"
        if mot_path.is_file():
            try:
                mot = mot_path.read_bytes()
            except OSError as exc:
                raise VerificationError(
                    f"{sequence}/{run_id}: durable MOT output is unreadable"
                ) from exc
            present[run_id] = {
                A76_EQUALITY_MEMBERS[0]: {
                    "length": len(mot),
                    "sha256": evidence.sha256_file(mot_path),
                }
            }
        if not (directory / evidence.BASE_POLICY_INVENTORY_NAME).is_file():
            continue
        inventory = _load(
            directory,
            evidence.BASE_POLICY_INVENTORY_NAME,
            schema=evidence.BASE_POLICY_INVENTORY_SCHEMA,
        )
        if set(inventory) != expected:
            raise VerificationError(
                f"base policy inventory schema mismatch: {sequence}/{run_id}"
            )
        synthetic = {
            **inventory,
            A76_OVERFLOW_MEMBER: list(A76_OVERFLOW_ZERO_VECTOR),
            A76_PROJECTION_MEMBERS[0]: None,
            "schema": evidence.POLICY_INVENTORY_SCHEMA,
            A76_PROJECTION_MEMBERS[1]: None,
        }
        try:
            h0_verifier._verify_policy_inventory(evidence.CAPTURE_OFF_RUN, synthetic)
        except h0_verifier.VerificationError as exc:
            raise VerificationError(f"{sequence}/{run_id}: {exc}") from exc
        base_records.add(run_id)
        durable_mot = present.get(run_id, {}).get(A76_EQUALITY_MEMBERS[0])
        if durable_mot is None:
            raise VerificationError(
                f"{sequence}/{run_id}: base inventory has no durable MOT output"
            )
        if inventory[A76_EQUALITY_MEMBERS[0]] != durable_mot:
            raise VerificationError(f"{sequence}/{run_id}: base MOT identity mismatch")
        present.setdefault(run_id, {}).update(
            {
                member: inventory[member]
                for member in A76_EQUALITY_MEMBERS
                if member != A76_EQUALITY_MEMBERS[0]
            }
        )
    return present, base_records


def _relations(
    bases: Mapping[str, Mapping[str, Any]],
    inventories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Every comparison the surviving inventories allow (H0's own grouping)."""
    off = evidence.CAPTURE_OFF_RUN
    relations: list[dict[str, Any]] = []
    first_unequal: str | None = None

    def record(left: str, member: str, right: str, equal: bool) -> None:
        nonlocal first_unequal
        relations.append(
            {"equal": equal, "left": left, "member": member, "right": right}
        )
        if not equal and first_unequal is None:
            first_unequal = f"{left}:{right}:{member}"

    if off in bases:
        for run_id in evidence.CAPTURE_ON_RUNS:
            if run_id not in bases:
                continue
            for member in A76_EQUALITY_MEMBERS:
                if member not in bases[off] or member not in bases[run_id]:
                    continue
                record(
                    off,
                    member,
                    run_id,
                    bases[off][member] == bases[run_id][member],
                )
    on_present = [run for run in evidence.CAPTURE_ON_RUNS if run in inventories]
    if on_present:
        reference = on_present[0]
        for member in A76_PROJECTION_MEMBERS:
            for run_id in on_present[1:]:
                record(
                    reference,
                    member,
                    run_id,
                    inventories[reference][member] == inventories[run_id][member],
                )
    for run_id in on_present:
        record(
            run_id,
            A76_OVERFLOW_MEMBER,
            "zero_vector",
            inventories[run_id][A76_OVERFLOW_MEMBER] == list(A76_OVERFLOW_ZERO_VECTOR),
        )
    return {
        "first_unequal": first_unequal,
        "relations": relations,
        "state": "equal" if first_unequal is None else "unequal",
    }


def _verify_packet(
    root: Path,
    sequence: str,
    run_id: str,
    inventory: Mapping[str, Any] | None,
) -> tuple[str, str | None]:
    """Re-verify one capture-on packet, independent of what else survived."""
    directory = evidence.run_dir(root, sequence, run_id)
    if not (directory / evidence.PACKET_NAME).is_file():
        return UNAVAILABLE, None
    capture = _load(directory, evidence.PACKET_NAME)
    stored: dict[str, Any] | None = None
    if (directory / evidence.PACKET_VERIFICATION_NAME).is_file():
        stored = _load(directory, evidence.PACKET_VERIFICATION_NAME)
    try:
        packet_report = verify_capture(capture)
        packet = canonical_semantic_packet(capture)
    except behavior.PACKET_INVALID_EXCEPTIONS:
        if stored is not None and stored != {
            "failure": "packet_invalid",
            "state": FAIL,
        }:
            raise VerificationError(
                f"packet verifier failure record mismatch: {sequence}/{run_id}"
            )
        return FAIL, None
    if inventory is not None:
        try:
            _cross_check_projections(packet, capture, inventory, sequence, run_id)
        except VerificationError:
            return FAIL, packet_report["semantic_digest_sha256"]
    if stored is not None and stored != {"report": packet_report, "state": PASS}:
        raise VerificationError(
            f"packet verifier pass record mismatch: {sequence}/{run_id}"
        )
    return PASS, packet_report["semantic_digest_sha256"]


def _cross_check_projections(
    packet: Mapping[str, Any],
    capture: Mapping[str, Any],
    inventory: Mapping[str, Any],
    sequence: str,
    run_id: str,
) -> None:
    streams = packet["streams"]
    candidates = [
        row for row in streams["candidate_records"] if int(row["proposal_emitted"]) == 1
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
        "digest": digest(proposal_payload),
        "records": proposal_payload,
    }
    expected_winner = {
        "count": len(commits),
        "digest": digest(winner_payload),
        "records": winner_payload,
    }
    if (
        inventory[A76_PROJECTION_MEMBERS[0]] != expected_proposal
        or inventory[A76_PROJECTION_MEMBERS[1]] != expected_winner
    ):
        raise VerificationError(
            f"packet/policy projection mismatch: {sequence}/{run_id}"
        )
    expected_overflow = [int(capture[key]) for key in OVERFLOW_FIELDS]
    if inventory[A76_OVERFLOW_MEMBER] != expected_overflow:
        raise VerificationError(f"packet/policy overflow mismatch: {sequence}/{run_id}")


def _sequence_replay(root: Path, sequence: str) -> SequenceReplay:
    bases, base_records = _base_inventories(root, sequence)
    inventories = _inventories(root, sequence)
    for run_id, inventory in inventories.items():
        if run_id not in base_records:
            raise VerificationError(
                f"{sequence}/{run_id}: full inventory has no durable base"
            )
        if any(
            member not in bases[run_id] or inventory[member] != bases[run_id][member]
            for member in A76_EQUALITY_MEMBERS
        ):
            raise VerificationError(
                f"{sequence}/{run_id}: full/base inventory mismatch"
            )
    reconstructed = _relations(bases, inventories)
    unequal_members = {
        relation["member"]
        for relation in reconstructed["relations"]
        if relation["equal"] is False
    }
    base_inequality = bool(unequal_members.intersection(A76_EQUALITY_MEMBERS))
    packet_relation_failure = bool(
        unequal_members.intersection((*A76_PROJECTION_MEMBERS, A76_OVERFLOW_MEMBER))
    )
    recorded_path = root / evidence.RUNS_DIR / sequence / evidence.COMPARISON_NAME
    if recorded_path.is_file():
        recorded = _load(root / evidence.RUNS_DIR / sequence, evidence.COMPARISON_NAME)
        if len(bases) == len(evidence.RUN_IDS) and len(inventories) == len(
            evidence.RUN_IDS
        ):
            if recorded != reconstructed:
                raise VerificationError(
                    f"{sequence}: comparison.json differs from the independent A7.6 "
                    "reconstruction"
                )
        elif reconstructed["state"] == "unequal" and recorded.get("state") == "equal":
            # Partial evidence cannot confirm a recorded equality, but it can
            # contradict one, and a contradiction is decisive.
            raise VerificationError(
                f"{sequence}: comparison.json records equality while the surviving "
                "inventories show an inequality"
            )

    states: list[tuple[str, str | None]] = []
    for run_id in evidence.CAPTURE_ON_RUNS:
        states.append(_verify_packet(root, sequence, run_id, inventories.get(run_id)))
    digests = [value for _, value in states if value is not None]
    if len(digests) > 1 and len(set(digests)) != 1:
        # Cross-repeat canonical digest equality, H0's own rule. Repeats that
        # verify individually but disagree canonically are not repeats of one
        # decision process — and two disagreeing survivors already establish
        # that, so the check does not wait for the third.
        first = None
        for index, (state, value) in enumerate(states):
            if value is None:
                continue
            if first is None:
                first = value
                continue
            if value != first:
                states[index] = (FAIL, value)
    if packet_relation_failure:
        states = [
            (FAIL, value) if state == PASS else (state, value)
            for state, value in states
        ]

    runs = tuple(
        RunReplay(
            run_id,
            run_id in base_records,
            UNAVAILABLE
            if run_id == evidence.CAPTURE_OFF_RUN
            else states[evidence.CAPTURE_ON_RUNS.index(run_id)][0],
        )
        for run_id in evidence.RUN_IDS
    )
    # Completeness and validity are different claims.  A present packet that
    # independently verifies `fail` is still a complete artifact and is exactly
    # how terminal 3 is evidenced.  Only `unavailable` means the completed
    # execution omitted a required packet; `Replay.all_packets_pass` owns the
    # separate validity verdict.
    required_full_inventories = {
        evidence.CAPTURE_OFF_RUN,
        *(
            run_id
            for run_id, (state, _) in zip(evidence.CAPTURE_ON_RUNS, states, strict=True)
            if state == PASS
        ),
    }
    complete = (
        len(base_records) == len(evidence.RUN_IDS)
        and required_full_inventories.issubset(inventories)
        and all(state != UNAVAILABLE for state, _ in states)
    )
    return SequenceReplay(
        sequence,
        runs,
        complete,
        base_inequality,
        packet_relation_failure,
    )


def _replay(root: Path, phase: str, *, strict: bool) -> Replay:
    expected = evidence.expected_sequences(phase)
    present = {path.name for path in evidence.sequence_dirs(root)}
    unexpected = sorted(present - set(expected))
    if unexpected:
        raise VerificationError(
            f"evidence root carries sequences the phase does not run: {unexpected}"
        )
    sequences = tuple(_sequence_replay(root, sequence) for sequence in expected)
    complete = all(item.complete for item in sequences)
    if strict and not complete:
        incomplete = [item.sequence for item in sequences if not item.complete]
        raise VerificationError(
            "a completed execution is missing policy inventories or capture-on "
            f"packets: {incomplete}"
        )
    return Replay(sequences, complete)


def _kill_switch(replay: Replay, observation: Mapping[str, Any]) -> None:
    """§ C3.5.1: surviving evidence may not sit under a predicate denying it.

    Without this, terminating a run at the first sign of perturbation would
    convert a forbidden terminal 2 or 3 into a re-attemptable terminal 4 — the
    same laundering § 8.1 forbids in the refit direction.
    """
    if replay.perturbation_observed and observation.get("capture_off_on_equal") is True:
        raise VerificationError(
            "surviving evidence shows a capture-off/on inequality while the "
            "recorded observation claims equality (§ C3.5.1)"
        )
    if replay.invalid_packet_observed and observation.get("packets_valid") is True:
        raise VerificationError(
            "surviving evidence shows an invalid packet while the recorded "
            "observation claims the packets are valid (§ C3.5.1)"
        )


# -- the § C3.6 admission gate, recomputed --------------------------------- #


def _required_mapping(freeze: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = freeze.get(field)
    if not isinstance(value, Mapping):
        raise VerificationError(f"F binds no {field} (§ C3.2)")
    return value


def _coordinate_and_probe(freeze: Mapping[str, Any], *, where: str) -> tuple[Any, Any]:
    coordinate = _required_mapping(freeze, "coordinate")
    missing = [axis for axis in ALL_COORDINATE_AXES if axis not in coordinate]
    if missing:
        raise VerificationError(f"{where} coordinate is missing axes: {missing}")
    probe = freeze.get("probe")
    if not isinstance(probe, str):
        raise VerificationError(f"{where} binds no bounded probe")
    return {axis: coordinate[axis] for axis in ALL_COORDINATE_AXES}, probe


def recompute_admission(
    root: Path, freeze: Mapping[str, Any], *, visiting: frozenset[str]
) -> dict[str, bool]:
    """Rebuild the § C3.6 verdict from artifacts, not from `admission.json`.

    Every condition is decided here from something outside the controller's own
    say-so: the bound Phase-A root and its verification, the two freeze records,
    the archived Layer-P certificate, and the prior-attempt chain.
    """
    conditions = {key: False for key, _ in partition.ADMISSION_CONDITIONS}

    # (a) and (b) — the bound Phase-A result exists, verifies, and passed.
    section = _required_mapping(freeze, "phase_a_evidence")
    bound_root_name = section.get("evidence_root")
    if not isinstance(bound_root_name, str):
        raise VerificationError("F_B binds no phase_a_evidence.evidence_root")
    phase_a_root = root.parent / bound_root_name
    phase_a_freeze: Mapping[str, Any] | None = None
    try:
        bound_name = _root_name(phase_a_root)
        if bound_name.phase != "a":
            raise VerificationError(
                f"phase_a_evidence names a phase-{bound_name.phase} root"
            )
        report = _verify_in_class(phase_a_root, "complete", visiting=visiting)
        if evidence.sha256_file(phase_a_root / evidence.MANIFEST_NAME) != section.get(
            "manifest_digest"
        ) or evidence.sha256_file(
            phase_a_root / evidence.CHECKSUMS_NAME
        ) != section.get("checksum_inventory_digest"):
            raise VerificationError(
                "the bound Phase-A manifest or checksum inventory differs from F_B"
            )
        phase_a_freeze = _load(
            phase_a_root, evidence.FREEZE_NAME, schema=evidence.FREEZE_SCHEMA
        )
    except (VerificationError, OSError):
        # A refused gate is a Layer-P coordinate, not an error: recompute it as
        # false and let the comparison against the record decide the outcome.
        return conditions
    conditions["phase_a_evidence_root_verifies"] = True
    conditions["phase_a_observation_selects_no_terminal"] = (
        report.get("result") == NON_TERMINAL_RESULT and report.get("terminal") is None
    )

    # (c) — § C3.1(b): the five axes and the probe, equal across both phases.
    try:
        mine = _coordinate_and_probe(freeze, where="F_B")
        theirs = _coordinate_and_probe(phase_a_freeze, where="F_A")
    except VerificationError:
        return conditions
    conditions["axes_and_probe_equal_freeze"] = mine == theirs

    # (d) — the certificate F_B binds is the one archived with the attempt.
    certificate = freeze.get("layer_p_certificate")
    if (
        isinstance(certificate, Mapping)
        and (root / evidence.CERTIFICATE_NAME).is_file()
    ):
        archived = _load(root, evidence.CERTIFICATE_NAME, schema=CERTIFICATE_SCHEMA)
        conditions["layer_p_certificate_matches_freeze"] = (
            certificate.get("digest") == digest(archived)
            and certificate.get("schema") == CERTIFICATE_SCHEMA
        )

    # (e) — the chain is complete, ordered, consumed-only, and verified. Note
    # what this is not: walking the list F_B supplied. A list cannot establish
    # its own completeness, so the corpus is scanned for the consumed attempts of
    # this Phase-A result and the list must equal what the scan finds.
    try:
        verify_prior_chain(root, freeze, visiting=visiting)
    except (VerificationError, CorpusError, OSError):
        return conditions
    conditions["prior_attempts_complete_and_verified"] = True
    return conditions


# -- § C3.6(e): the prior-attempt chain, discovered rather than accepted ---- #


def phase_a_group(freeze: Mapping[str, Any]) -> str:
    """The Phase-A result a Phase-B attempt binds — the chain's grouping key."""
    section = _required_mapping(freeze, "phase_a_evidence")
    group = section.get("evidence_root")
    if not isinstance(group, str):
        raise VerificationError("F_B binds no phase_a_evidence.evidence_root")
    return group


def _chain_position(freeze: Mapping[str, Any]) -> int:
    priors = freeze.get("prior_attempts")
    if not isinstance(priors, list) or any(
        not isinstance(item, str) for item in priors
    ):
        raise VerificationError("prior_attempts is not a list of evidence-root names")
    return len(priors)


def consumed_attempts(parent: Path, group: str) -> list[tuple[int, str]]:
    """Every consumed attempt for one Phase-A result, discovered from the corpus.

    Discovery is the point: § C3.6(e) asks whether `prior_attempts` is *complete*,
    and a list can only be checked for completeness against something the list did
    not produce. Reading the successors' own freeze records is what makes an
    omitted predecessor visible.

    `inadmissible` roots are excluded here, not merely rejected later: § C3.5.1
    step 4 says a refused gate is not a consumed attempt at all, so it neither
    belongs in a chain nor creates a hole by its absence.
    """
    members: list[tuple[int, str]] = []
    for candidate in sorted(
        parent.glob(f"{evidence.PHASE_B_ROOT_PREFIX}*"),
        key=lambda item: item.name.encode("utf-8"),
    ):
        if not candidate.is_dir():
            continue
        freeze = _load(candidate, evidence.FREEZE_NAME, schema=evidence.FREEZE_SCHEMA)
        if phase_a_group(freeze) != group:
            continue
        if classify(candidate) == partition.INADMISSIBLE_CLASS:
            continue
        members.append((_chain_position(freeze), candidate.name))
    return members


def verify_prior_chain(
    root: Path, freeze: Mapping[str, Any], *, visiting: frozenset[str]
) -> None:
    """§ C3.6(e) in full: complete, ordered, consumed-only, each verified in class.

    Raises rather than returning a verdict so the same message reaches an
    operator whether the caller is the per-root verifier or the corpus checker —
    there is one rule here, and it had better not be enforced twice with two
    different notions of what "complete" means.
    """
    group = phase_a_group(freeze)
    position = _chain_position(freeze)
    listed = list(freeze["prior_attempts"])
    members = consumed_attempts(root.parent, group)

    positions = [count for count, _ in members]
    duplicated = sorted({count for count in positions if positions.count(count) > 1})
    if duplicated:
        raise CorpusError(
            f"{root.name}: consumed attempts for {group} do not form one ordered "
            f"chain — position {duplicated} is claimed twice, so an existing "
            "consumed attempt is missing from a successor's prior_attempts "
            "(§ C3.6(e))"
        )
    expected = [
        name
        for count, name in sorted(members)
        if count < position and name != root.name
    ]
    if listed != expected:
        raise CorpusError(
            f"{root.name}: prior_attempts is not the complete ordered list of "
            f"preceding consumed attempts for the Phase-A result {group} "
            f"(expected {expected}, bound {listed})"
        )
    for name in listed:
        prior_root = root.parent / name
        if not prior_root.exists():
            raise CorpusError(f"{root.name}: prior attempt {name} does not exist")
        prior_class = classify(prior_root)
        if prior_class not in partition.VERIFY_CLASSES:
            # Reachable only if a root changed class between discovery and here.
            raise CorpusError(
                f"{root.name}: {name} is {prior_class} and is not a consumed "
                "attempt (§ C3.5.1 step 4)"
            )
        _verify_in_class(prior_root, prior_class, visiting=visiting | {root.name})


def _admission(
    root: Path, freeze: Mapping[str, Any], phase: str, *, visiting: frozenset[str]
) -> partition.Admission | None:
    if phase != "b":
        if (root / evidence.ADMISSION_NAME).exists():
            raise VerificationError(
                "a phase-A root carries an admission verdict; the § C3.6 gate is "
                "phase-B only"
            )
        return None
    recorded = _load(root, evidence.ADMISSION_NAME, schema=evidence.ADMISSION_SCHEMA)
    recomputed = recompute_admission(root, freeze, visiting=visiting)
    disagreed = sorted(
        key for key, value in recomputed.items() if recorded.get(key) != value
    )
    if disagreed:
        raise VerificationError(
            "recorded admission differs from the independent recomputation on "
            f"{disagreed}: the gate that decides whether S_B could be spent is "
            "not the controller's to assert (§ C3.6)"
        )
    try:
        verdict = partition.evaluate_admission(recomputed, phase="b")
    except partition.PartitionError as exc:
        raise VerificationError(f"admission record rejected: {exc}") from exc
    if not verdict.admitted:
        raise VerificationError(
            "admission was refused "
            f"({', '.join(verdict.reasons)}): this root is inadmissible, not a "
            "consumed attempt (§ C3.5.1 step 4)"
        )
    return verdict


# -- the verify classes ---------------------------------------------------- #


def _manifest(
    root: Path, name: evidence.RootName, present: Mapping[str, str]
) -> dict[str, Any]:
    manifest = _load(root, evidence.MANIFEST_NAME, schema=evidence.MANIFEST_SCHEMA)
    capture_phase = manifest.get("capture_phase")
    if capture_phase != evidence.CAPTURE_PHASE[name.phase]:
        raise VerificationError(
            f"manifest capture_phase {capture_phase!r} disagrees with the root name"
        )
    if manifest.get("instrumentation_head") != name.i40:
        raise VerificationError("manifest head disagrees with the root name")
    inventory = manifest.get("artifact_inventory")
    if inventory != sorted(present):
        raise VerificationError(
            "manifest artifact inventory differs from the checksum inventory"
        )
    return manifest


def _recompute_terminal(
    root: Path,
    *,
    phase: str,
    admission: partition.Admission | None,
    phase_b_complete: bool,
) -> tuple[dict[str, Any], partition.Selection]:
    observation = _load(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    try:
        rebuilt = evidence.build_observation(
            {
                key: observation[key]
                for key, _ in partition.ORDERED_PREDICATES
                if key in observation
            },
            execution_result=observation.get("execution_result"),
        )
    except (evidence.EvidenceError, KeyError) as exc:
        raise VerificationError(f"observation rejected: {exc}") from exc
    if rebuilt != observation:
        raise VerificationError(
            "observation carries fields outside the emitter's contract"
        )
    try:
        selection = partition.select_terminal(
            evidence.observation_predicates(observation),
            phase=phase,
            phase_b_complete=phase_b_complete,
            admission=admission,
        )
    except partition.PartitionError as exc:
        raise VerificationError(f"terminal selection rejected: {exc}") from exc
    recorded = _load(root, evidence.TERMINAL_NAME, schema=evidence.TERMINAL_SCHEMA)
    for field, value in (
        ("result", selection.result),
        ("terminal", selection.terminal),
        ("order", selection.order),
        ("phase", phase),
    ):
        if recorded.get(field) != value:
            raise VerificationError(
                f"recorded terminal {field}={recorded.get(field)!r} differs from the "
                f"independent selection {value!r}"
            )
    return observation, selection


def _completion_met(root: Path, phase: str, replay: Replay) -> bool:
    counts = evidence.completion(phase)
    required_sequences, required_packets, required_off_runs = (
        counts[key] for key in partition.COMPLETION_KEYS
    )
    capture_on = sum(
        1
        for item in replay.sequences
        for state in item.packet_states
        if state != UNAVAILABLE
    )
    capture_off = sum(
        1
        for item in replay.sequences
        for run in item.runs
        if run.run_id == evidence.CAPTURE_OFF_RUN and run.inventory_present
    )
    sequences = sum(1 for item in replay.sequences if item.complete)
    return (
        replay.complete
        and sequences == required_sequences
        and capture_on == required_packets
        and capture_off == required_off_runs
    )


def verify_evidence_root(root: Path) -> dict[str, Any]:
    """Verify a `complete` archive in full: structure, replay, gate, terminal."""
    return _verify_in_class(root, "complete", visiting=frozenset())


def _verify_complete(root: Path, *, visiting: frozenset[str]) -> dict[str, Any]:
    name = _root_name(root)
    present = _inventory(root)
    manifest = _manifest(root, name, present)
    freeze = _freeze(root, name)
    if manifest.get("freeze_digest") != evidence.freeze_digest(freeze):
        raise VerificationError("manifest freeze digest differs from the freeze record")
    admission = _admission(root, freeze, name.phase, visiting=visiting)
    _authorization(root, name.phase, freeze=freeze, name=name)

    # The observation is read once before the replay, because whether the replay
    # must be exhaustive is itself a recorded claim: only a completed execution
    # is required to have produced every artifact (§ C3.5.1).
    observation = _load(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    if not isinstance(observation.get("execution_complete"), bool):
        raise VerificationError("observation has no boolean execution_complete")
    controller_record = (
        _phase_a_launch_records(root, name, freeze, observation)
        if name.phase == "a"
        else None
    )
    strict = observation["execution_complete"]
    replay = _replay(root, name.phase, strict=strict)
    _kill_switch(replay, observation)
    if strict:
        for predicate, recomputed in (
            ("capture_off_on_equal", replay.all_equal),
            ("packets_valid", replay.all_packets_pass),
        ):
            if observation[predicate] != recomputed:
                raise VerificationError(
                    f"recorded {predicate}={observation[predicate]} differs from the "
                    f"independent replay ({recomputed})"
                )

    phase_b_complete = name.phase == "b" and _completion_met(root, name.phase, replay)
    observation, selection = _recompute_terminal(
        root,
        phase=name.phase,
        admission=admission,
        phase_b_complete=phase_b_complete,
    )
    if manifest.get("result") != selection.result:
        raise VerificationError("manifest result differs from the recomputed selection")
    if controller_record is not None and (
        controller_record.get("result") != selection.result
        or controller_record.get("terminal") != selection.terminal
        or controller_record.get("state") != "terminal"
    ):
        raise VerificationError(
            "controller record result/terminal differs from independent selection"
        )

    # A pass is the only outcome that claims completeness, so it is the only one
    # required to show it (§ 7 terminal 5 / the Phase-A progression).
    passing = selection.terminal == FULL_COMMIT_TERMINAL or (
        name.phase == "a" and selection.result == NON_TERMINAL_RESULT
    )
    if passing and not _completion_met(root, name.phase, replay):
        raise VerificationError(
            f"a passing {name.phase}-phase result does not meet "
            f"{evidence.completion(name.phase)}"
        )
    return {
        "schema": VERIFIER_SCHEMA,
        "verify_class": "complete",
        "capture_phase": manifest["capture_phase"],
        "evidence_root": root.name,
        "file_count": len(present),
        "freeze_digest": evidence.freeze_digest(freeze),
        "result": selection.result,
        "sequences": [item.sequence for item in replay.sequences],
        "terminal": selection.terminal,
        "valid": True,
    }


def _verify_spent(
    root: Path, *, require_terminal: bool, visiting: frozenset[str]
) -> dict[str, Any]:
    """`envelope` and `unterminated`: both spent `S_B`, both verify what survived."""
    name = _root_name(root)
    present = _inventory(root)
    freeze = _freeze(root, name)
    _admission(root, freeze, name.phase, visiting=visiting)
    _authorization(root, name.phase, freeze=freeze, name=name)
    observation: dict[str, Any] = {}
    if (root / evidence.OBSERVATION_NAME).is_file():
        observation = _load(
            root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
        )
    terminal_present = (root / evidence.TERMINAL_NAME).is_file()
    if require_terminal and not terminal_present:
        raise VerificationError(
            "an envelope records a caught failure and must carry its classification"
        )
    if not require_terminal and terminal_present:
        raise VerificationError(
            "an unterminated attempt records no terminal; this root carries one"
        )
    recorded: dict[str, Any] = {}
    if terminal_present:
        recorded = _load(root, evidence.TERMINAL_NAME, schema=evidence.TERMINAL_SCHEMA)
        if recorded.get("terminal") != EXECUTION_INVALID_TERMINAL:
            raise VerificationError(
                "an envelope is a caught execution failure; a root recording "
                f"{recorded.get('terminal')!r} must verify as complete"
            )
    replay = _replay(root, name.phase, strict=False)
    _kill_switch(replay, observation)
    return {
        "schema": VERIFIER_SCHEMA,
        "verify_class": "envelope" if require_terminal else "unterminated",
        "capture_phase": evidence.CAPTURE_PHASE[name.phase],
        "evidence_root": root.name,
        "file_count": len(present),
        "freeze_digest": evidence.freeze_digest(freeze),
        # § C3.5.1: an unterminated attempt selects no terminal — no observation
        # exists — and is treated as terminal 4 for re-attempt admissibility only.
        "result": recorded.get("result"),
        "terminal": recorded.get("terminal"),
        "perturbation_observed": replay.perturbation_observed,
        "invalid_packet_observed": replay.invalid_packet_observed,
        "valid": True,
    }


def _verify_inadmissible(root: Path, *, visiting: frozenset[str]) -> dict[str, Any]:
    """§ C3.5.1 step 4: refused before `S_B`, so it asserts nothing — and is still
    an artifact, so its identity and integrity are verified like any other."""
    del visiting
    name = _root_name(root)
    if name.phase != "b":
        raise VerificationError(
            "only a phase-B root can be inadmissible; the § C3.6 gate is phase-B only"
        )
    present = _inventory(root)
    freeze = _freeze(root, name)
    record = _load(root, evidence.ADMISSION_NAME, schema=evidence.ADMISSION_SCHEMA)
    try:
        verdict = partition.evaluate_admission(record, phase="b")
    except partition.PartitionError as exc:
        raise VerificationError(f"admission record rejected: {exc}") from exc
    if verdict.admitted:
        raise VerificationError(
            "this root records a passed admission gate and is not inadmissible"
        )
    for name_ in (evidence.AUTHORIZATION_NAME, evidence.TERMINAL_NAME):
        if (root / name_).exists():
            raise VerificationError(
                f"an inadmissible root spent no authorization and selected no "
                f"terminal, but carries {name_}"
            )
    return {
        "schema": VERIFIER_SCHEMA,
        "verify_class": partition.INADMISSIBLE_CLASS,
        "capture_phase": evidence.CAPTURE_PHASE[name.phase],
        "evidence_root": root.name,
        "file_count": len(present),
        "freeze_digest": evidence.freeze_digest(freeze),
        "admission_refused": list(verdict.reasons),
        "result": None,
        "terminal": None,
        "valid": True,
    }


def verify_envelope(root: Path) -> dict[str, Any]:
    """Verify the completeness of the envelope, not of the measurement."""
    return _verify_in_class(root, "envelope", visiting=frozenset())


def verify_unterminated(root: Path) -> dict[str, Any]:
    """Verify a root whose authorization was spent and whose process never exited."""
    return _verify_in_class(root, "unterminated", visiting=frozenset())


def verify_inadmissible(root: Path) -> dict[str, Any]:
    """Verify a root the § C3.6 gate refused before `S_B` was consumed."""
    return _verify_in_class(root, partition.INADMISSIBLE_CLASS, visiting=frozenset())


def _verify_in_class(
    root: Path, verify_class: str, *, visiting: frozenset[str]
) -> dict[str, Any]:
    if root.name in visiting:
        raise VerificationError(
            f"prior_attempts is cyclic through {root.name}: a chain cannot bind "
            "itself as its own predecessor"
        )
    visiting = visiting | {root.name}
    if verify_class == "complete":
        return _verify_complete(root, visiting=visiting)
    if verify_class == "envelope":
        return _verify_spent(root, require_terminal=True, visiting=visiting)
    if verify_class == "unterminated":
        return _verify_spent(root, require_terminal=False, visiting=visiting)
    if verify_class == partition.INADMISSIBLE_CLASS:
        return _verify_inadmissible(root, visiting=visiting)
    raise VerificationError(f"unknown verify class: {verify_class!r}")


def surviving_findings(root: Path) -> dict[str, bool]:
    """What the surviving artifacts already show, for the § C3.5.1 kill-switch."""
    name = _root_name(root)
    replay = _replay(root, name.phase, strict=False)
    return {
        "perturbation_observed": replay.perturbation_observed,
        "invalid_packet_observed": replay.invalid_packet_observed,
    }


VERIFIERS = {
    "complete": verify_evidence_root,
    "envelope": verify_envelope,
    "unterminated": verify_unterminated,
    partition.INADMISSIBLE_CLASS: verify_inadmissible,
}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--class",
        dest="verify_class",
        choices=sorted(VERIFIERS),
        default=None,
        help="the class to verify this root in (default: classify it)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        verify_class = args.verify_class or classify(args.evidence)
        report = VERIFIERS[verify_class](args.evidence)
    except (VerificationError, h0_verifier.VerificationError, OSError) as exc:
        print(f"H2 measurement evidence rejected: {exc}", file=sys.stderr)
        return 1
    print(evidence.canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
