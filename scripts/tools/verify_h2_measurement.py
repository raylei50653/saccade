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
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_measurement_evidence as evidence  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h0_phase_a as h0_verifier  # noqa: E402  (imported, never modified)
from build_runtime_identity import ALL_COORDINATE_AXES  # noqa: E402
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
from h2_runtime_inputs import digest  # noqa: E402
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


class RunReplay(NamedTuple):
    run_id: str
    inventory_present: bool
    packet_state: str


class SequenceReplay(NamedTuple):
    sequence: str
    runs: tuple[RunReplay, ...]
    complete: bool
    inequality_found: bool

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
        return any(FAIL in item.packet_states for item in self.sequences)

    @property
    def all_equal(self) -> bool:
        return self.complete and not self.perturbation_observed

    @property
    def all_packets_pass(self) -> bool:
        return self.complete and all(
            item.packet_states == (PASS,) * len(evidence.CAPTURE_ON_RUNS)
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


def _authorization(root: Path, phase: str) -> dict[str, Any] | None:
    path = root / evidence.AUTHORIZATION_NAME
    if phase == "b":
        # § C3.5.1 step 5: this record's durable write *is* the consumption of
        # S_B, so a Phase-B root without it never spent an authorization.
        return _load(
            root, evidence.AUTHORIZATION_NAME, schema=evidence.AUTHORIZATION_SCHEMA
        )
    if not path.exists():
        # Phase A consumes S_A at controller process launch (§ 5.2). The record
        # is a witness of that launch, not the consumption event, so its absence
        # is not a defect of the archive.
        return None
    return _load(
        root, evidence.AUTHORIZATION_NAME, schema=evidence.AUTHORIZATION_SCHEMA
    )


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


def _relations(inventories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
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

    if off in inventories:
        for run_id in evidence.CAPTURE_ON_RUNS:
            if run_id not in inventories:
                continue
            for member in A76_EQUALITY_MEMBERS:
                record(
                    off,
                    member,
                    run_id,
                    inventories[off][member] == inventories[run_id][member],
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
    except (KeyError, TypeError, ValueError):
        if stored is not None and stored != {
            "failure": "packet_invalid",
            "state": FAIL,
        }:
            raise VerificationError(
                f"packet verifier failure record mismatch: {sequence}/{run_id}"
            )
        return FAIL, None
    if inventory is not None:
        _cross_check_projections(packet, capture, inventory, sequence, run_id)
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
    inventories = _inventories(root, sequence)
    reconstructed = _relations(inventories)
    complete_inventories = len(inventories) == len(evidence.RUN_IDS)

    recorded_path = root / evidence.RUNS_DIR / sequence / evidence.COMPARISON_NAME
    if recorded_path.is_file():
        recorded = _load(root / evidence.RUNS_DIR / sequence, evidence.COMPARISON_NAME)
        if complete_inventories:
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

    runs = tuple(
        RunReplay(
            run_id,
            run_id in inventories,
            UNAVAILABLE
            if run_id == evidence.CAPTURE_OFF_RUN
            else states[evidence.CAPTURE_ON_RUNS.index(run_id)][0],
        )
        for run_id in evidence.RUN_IDS
    )
    complete = complete_inventories and all(state == PASS for state, _ in states)
    return SequenceReplay(sequence, runs, complete, reconstructed["state"] == "unequal")


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

    # (e) — every named prior attempt exists and verifies in its own class.
    priors = freeze.get("prior_attempts")
    if isinstance(priors, list) and all(isinstance(item, str) for item in priors):
        try:
            for name in priors:
                prior_root = root.parent / name
                _verify_in_class(
                    prior_root, classify(prior_root), visiting=visiting | {root.name}
                )
        except (VerificationError, OSError):
            return conditions
        conditions["prior_attempts_complete_and_verified"] = True
    return conditions


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
    _authorization(root, name.phase)

    # The observation is read once before the replay, because whether the replay
    # must be exhaustive is itself a recorded claim: only a completed execution
    # is required to have produced every artifact (§ C3.5.1).
    observation = _load(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    if not isinstance(observation.get("execution_complete"), bool):
        raise VerificationError("observation has no boolean execution_complete")
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
    _authorization(root, name.phase)
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
