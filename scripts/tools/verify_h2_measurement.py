#!/usr/bin/env python3
"""Independent verifier for an H2 Layer-M evidence root.

Independent means it recomputes rather than reads: the terminal is re-selected
from the archived observation through `h2_terminal_partition`, the A7.6
capture-off/on comparison is rebuilt from the archived policy inventories, and
every capture-on packet is re-verified through the packet verifier itself. A
controller that recorded a terminal its own artifacts do not support is rejected
here, which is the only reason this file exists separately from the controller.

**Nothing in this file is a comparison or terminal authority.** § 6 forbids H2
from introducing comparison vocabulary of its own, so the A7.6 members, the
policy-inventory shape and the packet predicates are consumed from H0's frozen
implementation by import — `verify_h0_phase_a._verify_policy_inventory`,
`verify_headline_bridge_decision_trace.verify_capture` and
`export_headline_bridge_decision_trace.canonical_semantic_packet` — and the
terminal comes from `h2_terminal_partition.select_terminal`. This module is
`plumbing_only` (§ C3.9) and holds no ruler.

Three verify classes exist because a terminal-4 attempt is defined by the
absence of artifacts (§ C3.5.1); `check_h2_measure_archives.py` is what
classifies a root, and this module provides the three verifications:

```text
verify_evidence_root   complete       the full measurement, verified in full
verify_envelope        envelope       a caught failure, verified as an envelope
verify_unterminated    unterminated   S_B spent, no terminal recorded
```

All three run the § C3.5.1 kill-switch check: surviving evidence that already
shows a capture-off/on inequality or an invalid packet may never sit under a
recorded predicate claiming otherwise.

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
from export_headline_bridge_decision_trace import (  # noqa: E402
    canonical_semantic_packet,
)
from h2_runtime_inputs import digest  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

VERIFIER_SCHEMA = "h2_measurement_verifier_v1"

VERIFY_CLASSES = ("complete", "envelope", "unterminated")

# A7.6's equality members, in H0's own grouping (`verify_h0_phase_a.py:2294`):
# four members compared capture-off against every capture-on run, two trace-only
# projections compared across capture-on runs, and the overflow vector required
# to be the zero vector. The membership is A7.6's; H2 adds nothing to it.
EQUALITY_MEMBERS = (
    "mot_output",
    "final_track_rows",
    "active_tid_slot_pairs",
    "relink_debug_raw",
)
PROJECTION_MEMBERS = ("proposal_projection", "winner_commit_projection")
OVERFLOW_ZERO_VECTOR = [0] * 9


class VerificationError(RuntimeError):
    """The archive does not support what it records. Always fail-closed."""


class SequenceReplay(NamedTuple):
    sequence: str
    comparison_equal: bool | None  # None: inventories absent, nothing replayed
    packet_states: tuple[str, ...]  # per capture-on run; () when absent


class Replay(NamedTuple):
    sequences: tuple[SequenceReplay, ...]
    complete: bool  # every expected sequence fully present and replayed

    @property
    def perturbation_observed(self) -> bool:
        return any(item.comparison_equal is False for item in self.sequences)

    @property
    def invalid_packet_observed(self) -> bool:
        return any("fail" in item.packet_states for item in self.sequences)

    @property
    def all_equal(self) -> bool:
        return self.complete and all(
            item.comparison_equal is True for item in self.sequences
        )

    @property
    def all_packets_pass(self) -> bool:
        return self.complete and all(
            item.packet_states == ("pass",) * len(evidence.CAPTURE_ON_RUNS)
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


def _admission(root: Path, phase: str) -> partition.Admission | None:
    if phase != "b":
        # § C3.6 is Phase-B only; Phase A's launch-time checks stay terminal-1
        # conditions and asking for a Phase-A admission verdict is a defect.
        if (root / evidence.ADMISSION_NAME).exists():
            raise VerificationError(
                "a phase-A root carries an admission verdict; the § C3.6 gate is "
                "phase-B only"
            )
        return None
    record = _load(root, evidence.ADMISSION_NAME, schema=evidence.ADMISSION_SCHEMA)
    try:
        verdict = partition.evaluate_admission(record, phase="b")
    except partition.PartitionError as exc:
        raise VerificationError(f"admission record rejected: {exc}") from exc
    if not verdict.admitted:
        raise VerificationError(
            "admission was refused "
            f"({', '.join(verdict.reasons)}): this root is inadmissible, not a "
            "consumed attempt (§ C3.5.1 step 4)"
        )
    return verdict


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


# -- A7.6 comparison and packet replay ------------------------------------- #


def _policy_inventories(root: Path, sequence: str) -> dict[str, dict[str, Any]] | None:
    inventories: dict[str, dict[str, Any]] = {}
    for run_id in evidence.RUN_IDS:
        path = evidence.run_dir(root, sequence, run_id) / evidence.POLICY_INVENTORY_NAME
        if not path.is_file():
            return None
        inventory = _load(
            evidence.run_dir(root, sequence, run_id),
            evidence.POLICY_INVENTORY_NAME,
            schema=evidence.POLICY_INVENTORY_SCHEMA,
        )
        try:
            # Consumed verbatim: A7.6's member set and shapes are H0's, and a
            # re-typed copy here would be exactly the vocabulary § 6 forbids.
            h0_verifier._verify_policy_inventory(run_id, inventory)
        except h0_verifier.VerificationError as exc:
            raise VerificationError(f"{sequence}/{run_id}: {exc}") from exc
        inventories[run_id] = inventory
    return inventories


def _reconstruct_comparison(
    inventories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """H0's A7.6 reconstruction (`verify_h0_phase_a.py:2294-2345`), per sequence."""
    off = evidence.CAPTURE_OFF_RUN
    relations: list[dict[str, Any]] = []
    first_unequal: str | None = None
    for run_id in evidence.CAPTURE_ON_RUNS:
        for member in EQUALITY_MEMBERS:
            equal = inventories[off][member] == inventories[run_id][member]
            relations.append(
                {"equal": equal, "left": off, "member": member, "right": run_id}
            )
            if not equal and first_unequal is None:
                first_unequal = f"{off}:{run_id}:{member}"
    reference_run = evidence.CAPTURE_ON_RUNS[0]
    for member in PROJECTION_MEMBERS:
        reference = inventories[reference_run][member]
        for run_id in evidence.CAPTURE_ON_RUNS[1:]:
            equal = reference == inventories[run_id][member]
            relations.append(
                {
                    "equal": equal,
                    "left": reference_run,
                    "member": member,
                    "right": run_id,
                }
            )
            if not equal and first_unequal is None:
                first_unequal = f"{reference_run}:{run_id}:{member}"
    for run_id in evidence.CAPTURE_ON_RUNS:
        zero = inventories[run_id]["overflow_vector"] == OVERFLOW_ZERO_VECTOR
        relations.append(
            {
                "equal": zero,
                "left": run_id,
                "member": "overflow_vector",
                "right": "zero_vector",
            }
        )
        if not zero and first_unequal is None:
            first_unequal = f"{run_id}:overflow_vector"
    return {
        "first_unequal": first_unequal,
        "relations": relations,
        "state": "equal" if first_unequal is None else "unequal",
    }


def _replay_packets(
    root: Path, sequence: str, inventories: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    """Re-verify the three capture-on packets through the packet verifier."""
    states: list[str] = []
    semantic_digests: list[str] = []
    for run_id in evidence.CAPTURE_ON_RUNS:
        directory = evidence.run_dir(root, sequence, run_id)
        if not (directory / evidence.PACKET_NAME).is_file():
            return ()
        capture = _load(directory, evidence.PACKET_NAME)
        stored = _load(directory, evidence.PACKET_VERIFICATION_NAME)
        try:
            packet_report = verify_capture(capture)
            packet = canonical_semantic_packet(capture)
        except (KeyError, TypeError, ValueError):
            states.append("fail")
            if stored != {"failure": "packet_invalid", "state": "fail"}:
                raise VerificationError(
                    f"packet verifier failure record mismatch: {sequence}/{run_id}"
                )
            continue
        streams = packet["streams"]
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
            "digest": digest(proposal_payload),
            "records": proposal_payload,
        }
        expected_winner = {
            "count": len(commits),
            "digest": digest(winner_payload),
            "records": winner_payload,
        }
        inventory = inventories[run_id]
        if (
            inventory["proposal_projection"] != expected_proposal
            or inventory["winner_commit_projection"] != expected_winner
        ):
            raise VerificationError(
                f"packet/policy projection mismatch: {sequence}/{run_id}"
            )
        expected_overflow = [
            int(capture[key])
            for key in (
                "overflow_pair_records",
                "overflow_candidate_records",
                "overflow_claim_records",
                "overflow_commit_records",
                "overflow_native_candidate_keys",
                "overflow_native_pair_keys",
                "overflow_native_proposal_keys",
                "overflow_native_claim_winner_keys",
                "overflow_native_commit_keys",
            )
        ]
        if inventory["overflow_vector"] != expected_overflow:
            raise VerificationError(
                f"packet/policy overflow mismatch: {sequence}/{run_id}"
            )
        states.append("pass")
        semantic_digests.append(packet_report["semantic_digest_sha256"])
        if stored != {"report": packet_report, "state": "pass"}:
            raise VerificationError(
                f"packet verifier pass record mismatch: {sequence}/{run_id}"
            )
    if len(semantic_digests) == len(evidence.CAPTURE_ON_RUNS) and (
        len(set(semantic_digests)) != 1
    ):
        # Cross-repeat canonical digest equality, H0's own rule: three repeats
        # that verify individually but disagree canonically are not three
        # observations of one decision process.
        states[1] = "fail"
    return tuple(states)


def _replay(root: Path, phase: str, *, strict: bool) -> Replay:
    """Replay whatever is present; `strict` demands every expected sequence."""
    expected = evidence.expected_sequences(phase)
    present = {path.name for path in evidence.sequence_dirs(root)}
    unexpected = sorted(present - set(expected))
    if unexpected:
        raise VerificationError(
            f"evidence root carries sequences the phase does not run: {unexpected}"
        )
    results: list[SequenceReplay] = []
    complete = True
    for sequence in expected:
        inventories = _policy_inventories(root, sequence)
        if inventories is None:
            complete = False
            if strict:
                raise VerificationError(
                    f"{sequence}: a completed execution is missing policy inventories"
                )
            results.append(SequenceReplay(sequence, None, ()))
            continue
        reconstructed = _reconstruct_comparison(inventories)
        recorded = _load(root / evidence.RUNS_DIR / sequence, evidence.COMPARISON_NAME)
        if recorded != reconstructed:
            raise VerificationError(
                f"{sequence}: comparison.json differs from the independent A7.6 "
                "reconstruction"
            )
        states = _replay_packets(root, sequence, inventories)
        if strict and len(states) != len(evidence.CAPTURE_ON_RUNS):
            raise VerificationError(
                f"{sequence}: a completed execution is missing capture-on packets"
            )
        if len(states) != len(evidence.CAPTURE_ON_RUNS):
            complete = False
        results.append(
            SequenceReplay(sequence, reconstructed["state"] == "equal", states)
        )
    return Replay(tuple(results), complete)


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


# -- the three verify classes ---------------------------------------------- #


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
    capture_on = sum(len(item.packet_states) for item in replay.sequences)
    capture_off = sum(
        1
        for item in replay.sequences
        if (
            root / evidence.RUNS_DIR / item.sequence / evidence.CAPTURE_OFF_RUN
        ).is_dir()
    )
    sequences = sum(1 for item in replay.sequences if item.comparison_equal is not None)
    return (
        replay.complete
        and sequences == counts["required_sequences"]
        and capture_on == counts["required_capture_on_packets"]
        and capture_off == counts["required_capture_off_runs"]
    )


def verify_evidence_root(root: Path) -> dict[str, Any]:
    """Verify a `complete` archive in full: structure, replay, and terminal."""
    name = _root_name(root)
    present = _inventory(root)
    manifest = _manifest(root, name, present)
    freeze = _freeze(root, name)
    if manifest.get("freeze_digest") != evidence.freeze_digest(freeze):
        raise VerificationError("manifest freeze digest differs from the freeze record")
    admission = _admission(root, name.phase)
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
    passing = selection.terminal == partition.TERMINALS[4].name or (
        name.phase == "a" and selection.result == "measurement_pass"
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


def _envelope_common(root: Path, *, require_terminal: bool) -> dict[str, Any]:
    name = _root_name(root)
    present = _inventory(root)
    freeze = _freeze(root, name)
    _admission(root, name.phase)
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
        if recorded.get("terminal") != "H2_MEASUREMENT_EXECUTION_INVALID":
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


def verify_envelope(root: Path) -> dict[str, Any]:
    """Verify the completeness of the envelope, not of the measurement."""
    return _envelope_common(root, require_terminal=True)


def verify_unterminated(root: Path) -> dict[str, Any]:
    """Verify a root whose authorization was spent and whose process never exited."""
    return _envelope_common(root, require_terminal=False)


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
}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--class",
        dest="verify_class",
        choices=VERIFY_CLASSES,
        default="complete",
        help="the § C3.5.1 class to verify this root in (default: complete)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = VERIFIERS[args.verify_class](args.evidence)
    except (VerificationError, h0_verifier.VerificationError, OSError) as exc:
        print(f"H2 measurement evidence rejected: {exc}", file=sys.stderr)
        return 1
    print(evidence.canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
