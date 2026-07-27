#!/usr/bin/env python3
"""The H2 Layer-M terminal partition: ordered, exhaustive, mechanically decidable.

This is the epistemic core of the redesign and the single load-bearing owner
decision (declaration § 7 / § 9.1). It is a module rather than prose because
§ 20.8's governing test is that *two independent implementers record the
bit-identical terminal*, and a partition that lives only in a table cannot be
executed by either of them.

Two changes from H0's A2.4 partition, and it matters which is which:

  * **`provenance_invalid` disappears as a predicate.** Not "is downgraded" —
    enumerative closure membership is no longer computed at all, so no observation
    can select it. What remains is the mutation detector, which is genuinely
    epistemic: if a bound input changed while the measurement ran, the measurement
    describes nothing.
  * **Pre-seal plumbing failures are not terminals.** Execution failures *after*
    the sealed launch still are (`H2_MEASUREMENT_EXECUTION_INVALID`, the
    mandatory fail-closed catch-all § 20.8 item 3 requires). The difference is
    where the failure happens, not whether it is excused: H0's only validation
    channel was post-seal, so plumbing defects necessarily cost authorizations.

Witness fields (physical hashes, loaded closures, GPU identity) may never select
a terminal; `select_terminal` refuses to look at them, and a test pins that.

**The partition is phase-aware, and the phase is never inferred** (§ C3.7). The
two chains check different things at different times: Phase B decides everything
decidable *before* launch in the § C3.6 admission gate, where refusal costs no
authorization, so its terminal 1 carries exactly one meaning — a bound input was
written while the measurement ran. Phase A's § 7 terminal 1 is untouched and
still admits a launch-time probe or certificate mismatch. Defaulting the phase
would silently pick one chain's ruler for the other's evidence, so `phase` is
required and inconsistent combinations raise.

Usage:
  uv run python scripts/tools/h2_terminal_partition.py --explain
  uv run python scripts/tools/h2_terminal_partition.py --explain --phase b
  uv run python scripts/tools/h2_terminal_partition.py --phase a --select result.json
  uv run python scripts/tools/h2_terminal_partition.py --phase b --admit gate.json
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple

PARTITION_SCHEMA = "h2_terminal_partition_v1"

PHASES: tuple[str, ...] = ("a", "b")

# The completion counts' own field names, so a consumer can index them without
# restating them (§ C3.9: a `plumbing_only` consumer may hold no ruler fact).
COMPLETION_KEYS: tuple[str, ...] = (
    "required_sequences",
    "required_capture_on_packets",
    "required_capture_off_runs",
)

# What "complete" means per phase, in the § 3.3 four-run block: one capture-off
# run and three capture-on packets per sequence. Phase A runs the measurement
# fixture alone; Phase B runs the § C3.2 item 10 seven-sequence plan.
PHASE_COMPLETION: dict[str, dict[str, int]] = {
    "a": {
        "required_sequences": 1,
        "required_capture_on_packets": 3,
        "required_capture_off_runs": 1,
    },
    "b": {
        "required_sequences": 7,
        "required_capture_on_packets": 21,
        "required_capture_off_runs": 7,
    },
}


class Terminal(NamedTuple):
    order: int
    name: str
    # phase -> that phase's exact condition. Not a single string: the two chains
    # genuinely differ, and a merged sentence would be true of neither.
    condition: dict[str, str]
    transition: str
    phase_a_reachable: bool

    def condition_for(self, phase: str) -> str:
        return self.condition[_checked_phase(phase)]


def _checked_phase(phase: str) -> str:
    if phase not in PHASES:
        raise PartitionError(
            f"unknown phase: {phase!r}; expected one of {list(PHASES)}"
        )
    return phase


def _completion_condition(phase: str) -> str:
    counts = PHASE_COMPLETION[phase]
    body = (
        "every preceding condition false, all "
        f"{counts['required_sequences']} sequence(s) complete, all "
        f"{counts['required_capture_on_packets']} capture-on packets and all "
        f"{counts['required_capture_off_runs']} capture-off run(s) recorded, and "
        "all verifications pass"
    )
    if phase == "a":
        return (
            f"{body} — but Phase A cannot select this terminal: a Phase-A pass is "
            "the non-terminal progression, and terminal 5 requires the frozen "
            "unlabelled seven-sequence Phase-B artifact"
        )
    return f"{body}, over the frozen unlabelled seven-sequence Phase B"


# Ordered; first applicable is authoritative (declaration § 7).
TERMINALS: tuple[Terminal, ...] = (
    Terminal(
        1,
        "H2_INPUT_MUTATED_DURING_MEASUREMENT",
        {
            "a": "a bound input was written during the invocation, or the behavior "
            "probe at launch differs from the reference bound in F, or the Layer-P "
            "certificate does not match F",
            "b": "a bound input — the Phase-A evidence root included — was written "
            "during the invocation. The launch-time probe and certificate checks are "
            "not here: § C3.6 decides them in the admission gate, before S_B is "
            "consumed, where a mismatch spends no authorization",
        },
        "closes the H2 measurement unit; object state unchanged; candidate set stays "
        "empty; a fresh I→F→S and a separate authorization would be required",
        True,
    ),
    Terminal(
        2,
        "H2_CAPTURE_PERTURBS_POLICY",
        {
            "a": "surviving evidence establishes that any A7.6 capture-off/on "
            "equality differs, even if a later packet or child step failed",
            "b": "surviving evidence establishes that any of the seven sequences' "
            "A7.6 capture-off/on equality differs (§ C3.4: disjunction of failure "
            "over the seven, so the terminal never depends on execution order)",
        },
        "closes the observational-capture route itself: decision-neutral shadow "
        "capture is not achievable at this ABI, so grounding must proceed by "
        "native-side reproduction or not at all",
        True,
    ),
    Terminal(
        3,
        "H2_PACKET_INVALID",
        {
            "a": "non-perturbation held but any packet, exposure, overflow, "
            "native-universe, conservation, cross-repeat canonical digest, or replay "
            "predicate fails",
            "b": "non-perturbation held but any of the seven sequences' packet, "
            "exposure, overflow, native-universe, conservation, cross-repeat "
            "canonical digest, or replay predicate fails",
        },
        "closes this measurement; routes to a separate capture-ABI-delta charter",
        True,
    ),
    Terminal(
        4,
        "H2_MEASUREMENT_EXECUTION_INVALID",
        {
            phase: "after the sealed launch: nonzero build, extension/plugin load "
            "failure, runner nonzero, deadline exhausted, serialization failure, "
            "missing or unreadable required artifact, or any unclassified execution "
            "failure"
            for phase in PHASES
        },
        "closes this measurement with no partial-capture reinterpretation; a fresh "
        "chain would be required",
        True,
    ),
    Terminal(
        5,
        "H2_FULL_COMMIT_CAPTURE_FAITHFUL",
        {phase: _completion_condition(phase) for phase in PHASES},
        "adds a decision capability: a runtime-fidelity edge becomes available for "
        "owner acceptance, which is the precondition — not the activation — of "
        "H0_ROUTE5_B1 / GCTM_B1 / O1",
        False,
    ),
)

# Every controller result must map to exactly one terminal, or to the explicit
# non-terminal progression. An unmapped result is a declaration defect, not a
# runtime surprise (§ 20.8 item 3).
RESULT_TO_TERMINAL: dict[str, str | None] = {
    # -> terminal 1
    "input_mutated": "H2_INPUT_MUTATED_DURING_MEASUREMENT",
    "behavior_probe_moved": "H2_INPUT_MUTATED_DURING_MEASUREMENT",
    "certificate_mismatch": "H2_INPUT_MUTATED_DURING_MEASUREMENT",
    # -> terminal 2
    "capture_perturbs_policy": "H2_CAPTURE_PERTURBS_POLICY",
    # -> terminal 3
    "packet_invalid": "H2_PACKET_INVALID",
    # -> terminal 4
    "build_failed": "H2_MEASUREMENT_EXECUTION_INVALID",
    "extension_load_failed": "H2_MEASUREMENT_EXECUTION_INVALID",
    "runner_nonzero": "H2_MEASUREMENT_EXECUTION_INVALID",
    "runner_timeout": "H2_MEASUREMENT_EXECUTION_INVALID",
    "serialization_failed": "H2_MEASUREMENT_EXECUTION_INVALID",
    "artifact_missing_or_unreadable": "H2_MEASUREMENT_EXECUTION_INVALID",
    "unclassified_execution_failure": "H2_MEASUREMENT_EXECUTION_INVALID",
    # Phase A pass is not a terminal. Terminal 5 needs the Phase-B artifact.
    "measurement_pass": None,
}

# Observation keys `select_terminal` is allowed to read, in evaluation order. A key
# outside this tuple cannot influence a terminal — which is how "witness fields
# carry no decision authority" is enforced rather than merely asserted.
ORDERED_PREDICATES: tuple[tuple[str, str], ...] = (
    ("bound_input_mutated", "input_mutated"),
    ("behavior_probe_equals_freeze", "behavior_probe_moved"),
    ("layer_p_certificate_matches_freeze", "certificate_mismatch"),
    ("capture_off_on_equal", "capture_perturbs_policy"),
    ("packets_valid", "packet_invalid"),
    ("execution_complete", "unclassified_execution_failure"),
)

# Predicates whose *false* value selects the failure, vs whose *true* value does.
_TRUE_IS_FAILURE = frozenset({"bound_input_mutated"})

# The § C3.6 admission gate, in the clause's own order (a–e). These are NOT
# predicates of the partition: they are evaluated *before* the § C3.5.1 step-5
# write that consumes S_B, and their failure selects no terminal at all. Adding
# them to ORDERED_PREDICATES would be the exact error the clause exists to
# prevent — an inadmissible launch recorded as an epistemic result.
ADMISSION_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("phase_a_evidence_root_verifies", "phase_a_evidence_root_unverified"),
    ("phase_a_observation_selects_no_terminal", "phase_a_did_not_pass"),
    ("axes_and_probe_equal_freeze", "freeze_mismatch"),
    ("layer_p_certificate_matches_freeze", "certificate_mismatch"),
    ("prior_attempts_complete_and_verified", "prior_attempts_unverified"),
)

# An admission failure is Layer-P class (§ 5.1): a coordinate to retry against,
# not a result about the world.
ADMISSION_FAILURE_CLASS = "layer_p"

# Terminal 4 by name, for consumers that must recognise the fail-closed
# catch-all without spelling it out.
EXECUTION_INVALID_TERMINAL = "H2_MEASUREMENT_EXECUTION_INVALID"

# § C3.5.1's verify classes, plus the step-4 outcome that is not one of them.
# An `inadmissible` root spent no authorization and is never a consumed attempt;
# the other three all spent `S_B` and differ only in how much of the measurement
# survived to be verified.
VERIFY_CLASSES: tuple[str, ...] = ("complete", "envelope", "unterminated")
INADMISSIBLE_CLASS = "inadmissible"

# § C3.5: the terminals that are properties of the sealed `F_B` measurement
# surface rather than of the attempt. A re-attempt against the same surface is
# forbidden; terminals 1 and 4 stay attempt-local and re-attemptable.
SURFACE_BAN_TERMINALS: frozenset[str] = frozenset(
    {TERMINALS[1].name, TERMINALS[2].name}
)

# H0 § 6, verbatim: "Only repairs that leave all those semantics unchanged —
# compilation, capacity sizing, serialization, or implementation bugs — may
# proceed under the same seal." § C3.5's first guard consumes that vocabulary
# unchanged, so a surface change outside it is not a repair and re-admits
# nothing. Declared here rather than in the archive checker: what may reopen a
# banned measurement is a ruler fact, and a `plumbing_only` file could extend it
# without moving an axis (§ C3.9).
REPAIR_VOCABULARY: frozenset[str] = frozenset(
    {"compilation", "capacity_sizing", "serialization", "implementation_bug"}
)

# Predicates § C3.6 moves out of Phase B's terminal 1 and into admission. Phase B
# still emits all six (§ C3.4), and these two are then already decided: admission
# passed only if both held. A post-launch move of either is a write to a bound
# input, which `bound_input_mutated` reports. So a Phase-B observation that
# reports one false while claiming admission passed is incoherent, and it fails
# closed here rather than selecting terminal 1 under a condition Phase B's
# terminal 1 no longer names.
PHASE_B_ADMISSION_DECIDED: frozenset[str] = frozenset(
    {"behavior_probe_equals_freeze", "layer_p_certificate_matches_freeze"}
)


# Results `RESULT_TO_TERMINAL` maps to terminal 1 but Phase B cannot select,
# because their predicates were decided in admission. `RESULT_TO_TERMINAL` is the
# phase-independent union; this is the Phase-B restriction of it, and both are
# published so an implementer consuming only `as_payload()` reaches the same
# verdict as one calling `select_terminal` (§ 20.8's two-implementer test).
PHASE_B_FORBIDDEN_RESULTS: tuple[str, ...] = tuple(
    result for key, result in ORDERED_PREDICATES if key in PHASE_B_ADMISSION_DECIDED
)


class PartitionError(RuntimeError):
    pass


class Admission(NamedTuple):
    admitted: bool
    reasons: tuple[str, ...]

    # Structural, not a computed field: no admission outcome selects a terminal.
    terminal: None = None

    def describe(self) -> str:
        if self.admitted:
            return "admission passed → S_B may be consumed and the measurement launched"
        return (
            f"admission refused ({', '.join(self.reasons)}) → no terminal, "
            f"no authorization spent, {ADMISSION_FAILURE_CLASS} class"
        )


def evaluate_admission(record: Mapping[str, Any], *, phase: str) -> Admission:
    """Evaluate the § C3.6 gate. Total, fail-closed, and pre-terminal.

    Phase-B only: § C3.6 narrows the Phase-B chain and explicitly does not align
    Phase A, whose § 7 launch-time checks remain terminal 1 conditions. Asking
    for a Phase-A admission verdict is a caller defect, not a pass.
    """
    if _checked_phase(phase) != "b":
        raise PartitionError(
            "the admission gate is defined for phase 'b' only; Phase A's "
            "launch-time checks are terminal-1 conditions (§ 7)"
        )
    missing = [key for key, _ in ADMISSION_CONDITIONS if key not in record]
    if missing:
        raise PartitionError(f"admission record is missing conditions: {missing}")
    reasons: list[str] = []
    for key, reason in ADMISSION_CONDITIONS:
        value = record[key]
        if not isinstance(value, bool):
            raise PartitionError(f"admission condition {key} is not a bool: {value!r}")
        if not value:
            reasons.append(reason)
    return Admission(not reasons, tuple(reasons))


class Selection(NamedTuple):
    result: str
    terminal: str | None
    order: int | None
    transition: str | None
    phase_a_emittable: bool
    phase: str = "a"

    def describe(self) -> str:
        if self.terminal is None:
            return (
                f"phase={self.phase} result={self.result} → no H2 terminal "
                "(non-terminal progression); terminal 5 requires the Phase-B artifact"
            )
        return (
            f"phase={self.phase} result={self.result} → {self.terminal} "
            f"(ordered #{self.order})"
        )


def terminal_by_name(name: str) -> Terminal:
    for terminal in TERMINALS:
        if terminal.name == name:
            return terminal
    raise PartitionError(f"unknown terminal: {name}")


def select_terminal(
    observation: Mapping[str, Any],
    *,
    phase: str,
    phase_b_complete: bool = False,
    admission: Admission | None = None,
) -> Selection:
    """Map one observation to exactly one terminal. Total and order-sensitive.

    Every predicate in `ORDERED_PREDICATES` must be present: a missing predicate
    is a defect in the caller, and guessing a default is how a fail-closed check
    becomes fail-open. An explicit `execution_result` may name a specific
    execution failure so terminal 4's cause is recorded rather than flattened.

    `phase` is required and `phase_b_complete=True` is admissible only under
    `phase="b"`; the inconsistent combination raises rather than defaulting.
    Under `phase="b"` a passed `Admission` is required — this is where "an
    admission failure yields no terminal" (§ C3.6) is executable rather than
    prose, since a refused gate must never reach a selection at all.

    **Phase B is total: it has no non-terminal progression.** The
    `measurement_pass`/no-terminal outcome exists for Phase A alone, whose pass
    is the progression into Phase B. By the time a Phase-B observation is
    selected on, admission has passed and the § C3.5.1 step-5 write has consumed
    `S_B`, so returning "no terminal" would leave an authorization permanently
    spent with nothing recorded — the exact state § C3.5.1 exists to make
    unformable. A clean Phase-B observation must therefore carry
    `phase_b_complete=True`, and a caller that omits it gets an error rather
    than a hole.
    """
    _checked_phase(phase)
    if phase_b_complete and phase != "b":
        raise PartitionError(
            "phase_b_complete is admissible only under phase='b': a Phase-A chain "
            "cannot hold the seven-sequence Phase-B artifact (§ 7, § C3.7)"
        )
    if phase == "b":
        if admission is None:
            raise PartitionError(
                "phase='b' requires the § C3.6 admission verdict: no terminal may be "
                "selected before the gate that precedes S_B consumption"
            )
        if not admission.admitted:
            raise PartitionError(
                "admission was refused "
                f"({', '.join(admission.reasons)}): no terminal is selected, no "
                f"authorization is spent, {ADMISSION_FAILURE_CLASS} class (§ C3.6)"
            )
    elif admission is not None:
        raise PartitionError(
            "phase='a' takes no admission verdict: the gate is Phase-B only (§ C3.6)"
        )

    missing = [key for key, _ in ORDERED_PREDICATES if key not in observation]
    if missing:
        raise PartitionError(f"observation is missing predicates: {missing}")

    if phase == "b":
        decided = sorted(
            key for key in PHASE_B_ADMISSION_DECIDED if observation[key] is False
        )
        if decided:
            raise PartitionError(
                f"phase='b' observation contradicts a passed admission gate: {decided} "
                "were decided before S_B was consumed, and a later move of either is a "
                "bound-input mutation (§ C3.6)"
            )

    for key, result in ORDERED_PREDICATES:
        value = observation[key]
        if not isinstance(value, bool):
            raise PartitionError(f"predicate {key} is not a bool: {value!r}")
        failed = value if key in _TRUE_IS_FAILURE else not value
        if not failed:
            continue
        if result == "unclassified_execution_failure":
            # Let the caller name the execution cause; every legal name still maps
            # to terminal 4, so a mislabelled cause cannot change the terminal.
            named = observation.get("execution_result", result)
            if RESULT_TO_TERMINAL.get(named) != "H2_MEASUREMENT_EXECUTION_INVALID":
                raise PartitionError(
                    f"execution_result {named!r} does not map to terminal 4"
                )
            result = named
        return _selection(result, phase=phase)

    if phase == "b":
        if not phase_b_complete:
            raise PartitionError(
                "a clean Phase-B observation requires phase_b_complete=True: Phase B "
                "has no non-terminal progression, and S_B is already consumed by the "
                "time an observation is selected on (§ C3.5.1). Returning no terminal "
                "here would spend an authorization and record nothing"
            )
        return _selection(
            "measurement_pass", phase=phase, terminal_override=TERMINALS[4].name
        )
    return _selection("measurement_pass", phase=phase)


def _selection(
    result: str, *, phase: str, terminal_override: str | None = None
) -> Selection:
    if result not in RESULT_TO_TERMINAL:
        raise PartitionError(f"unmapped controller result: {result}")
    name = terminal_override or RESULT_TO_TERMINAL[result]
    if name is None:
        return Selection(result, None, None, None, True, phase)
    terminal = terminal_by_name(name)
    return Selection(
        result,
        terminal.name,
        terminal.order,
        terminal.transition,
        terminal.phase_a_reachable,
        phase,
    )


def as_payload() -> dict[str, Any]:
    return {
        "schema": PARTITION_SCHEMA,
        "note": (
            "provenance_invalid is absent as a predicate: enumerative closure "
            "membership is not computed, so no observation can select it. Witness "
            "fields carry no decision authority."
        ),
        "admission_conditions": [list(item) for item in ADMISSION_CONDITIONS],
        "admission_failure_class": ADMISSION_FAILURE_CLASS,
        "inadmissible_class": INADMISSIBLE_CLASS,
        "ordered_predicates": [list(item) for item in ORDERED_PREDICATES],
        "repair_vocabulary": sorted(REPAIR_VOCABULARY),
        "surface_ban_terminals": sorted(SURFACE_BAN_TERMINALS),
        "verify_classes": list(VERIFY_CLASSES),
        "phase_completion": PHASE_COMPLETION,
        # `result_to_terminal` above is the phase-independent union. Phase B
        # narrows it in two ways, both published here: an implementer that
        # consumes only this payload must reach the same verdict as one calling
        # `select_terminal`, or § 20.8's two-implementer test is satisfied by
        # neither of them.
        "phase_narrowing": {
            "b": {
                "admission_decided_predicates": sorted(PHASE_B_ADMISSION_DECIDED),
                "forbidden_results": list(PHASE_B_FORBIDDEN_RESULTS),
                "clean_observation_requires_phase_b_complete": True,
                "has_non_terminal_progression": False,
            },
            "a": {
                "admission_decided_predicates": [],
                "forbidden_results": [],
                "clean_observation_requires_phase_b_complete": False,
                "has_non_terminal_progression": True,
            },
        },
        "phases": list(PHASES),
        "result_to_terminal": RESULT_TO_TERMINAL,
        "terminals": [terminal._asdict() for terminal in TERMINALS],
    }


def _explain(phases: tuple[str, ...]) -> None:
    for phase in phases:
        counts = PHASE_COMPLETION[phase]
        print(f"phase {phase}: {counts}")
        for terminal in TERMINALS:
            reach = "both phases" if terminal.phase_a_reachable else "phase B only"
            print(f"  {terminal.order}. {terminal.name}  [{reach}]")
            print(f"       when: {terminal.condition_for(phase)}")
            print(f"       then: {terminal.transition}")
        if phase == "b":
            print("  admission (pre-terminal, before S_B is consumed):")
            for key, reason in ADMISSION_CONDITIONS:
                print(f"       {key} — false selects no terminal, reason {reason}")
            print(
                "  narrowing: results "
                f"{list(PHASE_B_FORBIDDEN_RESULTS)} are unselectable (decided in "
                "admission); Phase B is total — a clean observation requires "
                "phase_b_complete=True and there is no non-terminal progression"
            )
        else:
            print(
                "  progression: a clean Phase-A observation selects no terminal "
                "(measurement_pass) — that pass is the entry to Phase B"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain", action="store_true", help="print the partition")
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default=None,
        help="the chain being decided; required for --select and --admit",
    )
    parser.add_argument(
        "--select",
        type=Path,
        default=None,
        help="select a terminal from a JSON observation",
    )
    parser.add_argument(
        "--admit",
        type=Path,
        default=None,
        help="evaluate the § C3.6 admission gate from a JSON record (phase b)",
    )
    parser.add_argument(
        "--phase-b-complete",
        action="store_true",
        help="the frozen seven-sequence Phase-B artifact exists (phase b only)",
    )
    args = parser.parse_args(argv)

    if args.explain and args.select is None and args.admit is None:
        _explain((args.phase,) if args.phase else PHASES)
        return 0

    if args.select is None and args.admit is None:
        parser.error("one of --explain, --select or --admit is required")
    if args.phase is None:
        parser.error("--phase is required with --select or --admit")

    admission: Admission | None = None
    try:
        if args.admit is not None:
            record = json.loads(args.admit.read_text(encoding="utf-8"))
            admission = evaluate_admission(record, phase=args.phase)
            print(admission.describe())
            if not admission.admitted:
                # Refused: Layer-P class, no terminal, no authorization spent.
                return 1
        if args.select is None:
            return 0
        observation = json.loads(args.select.read_text(encoding="utf-8"))
        selection = select_terminal(
            observation,
            phase=args.phase,
            phase_b_complete=args.phase_b_complete,
            admission=admission,
        )
    except (PartitionError, json.JSONDecodeError, OSError) as exc:
        print(f"terminal selection failed: {exc}", file=sys.stderr)
        return 1
    print(selection.describe())
    if selection.transition:
        print(f"mainline transition: {selection.transition}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
