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
from typing import Any, Mapping, NamedTuple, Sequence

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
    "runtime_binding_mismatch": "H2_INPUT_MUTATED_DURING_MEASUREMENT",
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

# ---------------------------------------------------------------------------
# The successor vocabulary (Review Correction 9)
#
# `h2_execution_result_v1` names the same partition with different words: three
# predicates are renamed, one of them with inverted polarity, and a predicate is
# no longer a bool. Both vocabularies must resolve to *this* partition or the two
# archive generations answer differently about the same world — and the schema
# alone cannot compute a selection, so the mapping lives here and is published.
# ---------------------------------------------------------------------------

AUTHORITIES: tuple[str, ...] = ("non_qualifying_diagnostic", "exactly_once_measurement")

# The only result a diagnostic may record. A diagnostic records every failed
# predicate and selects no terminal: it is not a measurement that happened to
# pass, and no green diagnostic qualifies or authorizes one (§ Review
# Correction 5).
DIAGNOSTIC_RESULT = "diagnostic_complete"

PREDICATE_STATES: tuple[str, ...] = ("pass", "fail", "error", "not_run")
PASS_STATE = "pass"
FAIL_STATE = "fail"
# Neither a pass nor a decided failure: the measurement did not decide this
# predicate. Reading either as a pass is how a fail-closed check becomes
# fail-open, so they are non-passes that name no result of their own.
UNDECIDED_STATES: tuple[str, ...] = ("error", "not_run")

# The successor spelling of `ORDERED_PREDICATES`, in the same order — the order
# is the partition, so it may not drift between the two vocabularies.
SUCCESSOR_PREDICATES: tuple[tuple[str, str], ...] = (
    ("bound_input_unchanged", "input_mutated"),
    ("runtime_projection_matches_resolved_run_spec", "runtime_binding_mismatch"),
    ("capture_off_on_equal", "capture_perturbs_policy"),
    ("packets_valid", "packet_invalid"),
    ("execution_complete", "unclassified_execution_failure"),
)

# Successor predicate key -> the legacy key it renames. Historical archives keep
# the legacy spelling, so the two must remain mutually resolvable.
SUCCESSOR_TO_LEGACY_PREDICATE: dict[str, str] = {
    "bound_input_unchanged": "bound_input_mutated",
    "capture_off_on_equal": "capture_off_on_equal",
    "packets_valid": "packets_valid",
    "execution_complete": "execution_complete",
}

# One successor predicate is not a rename of anything. The legacy
# `layer_p_certificate_matches_freeze` compared a whole binding against a
# published certificate — the gate Correction 5 retired. Its successor compares
# a narrower quantity against a different authority: the RunSpec-owned launch
# projection, recomputed from the resolved RunSpec the archive itself carries.
# Mapping the two names would claim they measure the same thing, which is how a
# retired gate returns through a rename.
SUCCESSOR_WITHOUT_LEGACY_PREDICATE: frozenset[str] = frozenset(
    {"runtime_projection_matches_resolved_run_spec"}
)

# The rename that also flips polarity: the legacy predicate is true when the
# world is *broken*, the successor predicate is true when it is intact. A
# consumer that maps names without the polarity gets terminal 1 exactly backwards.
INVERTED_POLARITY_PREDICATES = frozenset({"bound_input_unchanged"})

# Correction 5 retires the Layer-P certificate, so the result it named is
# superseded rather than deleted: the historical archives that recorded
# `certificate_mismatch` keep their meaning, and both tokens select terminal 1.
LEGACY_RESULT_SUPERSEDED_BY: dict[str, str] = {
    "certificate_mismatch": "runtime_binding_mismatch",
}

# Correction 10 retires a verdict chain rather than renaming it. `behavior_probe_moved`
# keeps its terminal for the historical archives that recorded it, and is refused
# to a successor archive: nothing in a successor execution may select it, because
# the comparison that once produced it has no normative right-hand side left. It
# has no successor spelling and no superseding token — the finding is gone, not
# moved, and a probe is now a diagnostic observation.
RETIRED_SUCCESSOR_RESULTS: frozenset[str] = frozenset({"behavior_probe_moved"})

# Layer P's retained stages in order (§ Review Correction 5). The first two run
# before any bytes are bound, so a failure there forms no archive at all — which
# is why only the last four can appear as a binding's `failed_stage`.
BINDING_STAGES: tuple[str, ...] = (
    "retry_admissibility",
    "preflight",
    "build",
    "build_binding",
    "extension_load",
    "identity_run",
)
BINDABLE_FAILURE_STAGES: tuple[str, ...] = BINDING_STAGES[2:]

# Which named cause requires which stage to have failed. Without this the two
# re-admitted tokens are interchangeable labels rather than stage evidence: a
# `build_failed` whose binding shows a completed build says nothing. The
# biconditional runs both ways — `build_binding` and `identity_run` have no
# dedicated token and are carried by the catch-all with the stage named.
RESULT_REQUIRES_FAILED_STAGE: dict[str, str] = {
    "build_failed": "build",
    "extension_load_failed": "extension_load",
}
CATCH_ALL_FAILURE_STAGES: tuple[str, ...] = ("build_binding", "identity_run")

# The monitor is the only witness of terminal 1, and terminal 1 outranks every
# other result, so a recorded change and `input_mutated` imply each other.
RESULT_REQUIRES_INPUT_MUTATION = "input_mutated"

# Which findings are *reachable* at which point in the execution — the axis a
# terminal-level classification loses. A terminal is not a time: two results
# sharing terminal 1 can differ in whether the evidence they name could exist yet.
#
#   * stage-independent — decidable from the binding members
#     `h2_runtime_binding_v1` requires at every stage (`runtime_inputs`,
#     `executed_surfaces`, `capture_abi`, `source_audit`, `input_monitor`), so they
#     survive a build failure. `input_mutated` belongs here because the monitor
#     starts before the binding by contract;
#   * probe-derived — **empty, and empty is the claim.** The behaviour probe was
#     the sole member, and Correction 10 retires the verdict chain it stood in:
#     Correction 5 had already retired the published probe and the Layer-P
#     certificate as gates, leaving `behavior_probe_equals_spec` a name with no
#     normative right-hand side. A probe is worth recording and is recorded, as a
#     diagnostic observation that selects nothing;
#   * run-derived — needs a measurement run to have started; no run starts until
#     every retained stage completed, and Correction 5's measurement mode is
#     fail-fast, so a stage failure means these findings cannot exist yet.
STAGE_INDEPENDENT_RESULTS: tuple[str, ...] = (
    "input_mutated",
    "runtime_binding_mismatch",
)
PROBE_DERIVED_RESULTS: tuple[str, ...] = ()
RUN_DERIVED_RESULTS: tuple[str, ...] = ("capture_perturbs_policy", "packet_invalid")
RUN_STATES: tuple[str, ...] = ("completed", "failed", "not_run")
RUN_NOT_STARTED_STATE = "not_run"

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
            # Shared with the successor vocabulary: one rule, one implementation.
            result = _named_execution_result(
                observation.get("execution_result"), result
            )
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


def select_successor_result(
    predicate_results: Mapping[str, Any],
    *,
    authority: str,
    phase: str,
    execution_result: str | None = None,
) -> Selection:
    """Select from a `h2_execution_result_v1` observation. Total and fail-closed.

    The successor artifact widened a predicate from a bool to four states, so the
    two undecided states need a rule the legacy partition never had, and getting
    it wrong reintroduces two defects this unit already paid for:

    * **A decided failure outranks an undecided predicate, wherever it sits.**
      Not "first applicable over the raw states": an `error` on an early
      predicate must not wash a later capture-perturbation or invalid-packet
      *finding* into terminal 4, or killing a process on sight would launder a
      banned terminal into a re-attemptable one. Among decided failures the
      partition's order decides, exactly as before.
    * **An undecided predicate cannot coexist with a complete execution.**
      If nothing failed but something was not decided, the execution did not
      complete, and `execution_complete` must say so; a record claiming both is
      internally contradictory and is refused rather than mapped.

    A diagnostic selects `diagnostic_complete` and no terminal whatever its
    predicates say — that is the authority boundary, not a shortcut.
    """
    if authority not in AUTHORITIES:
        raise PartitionError(
            f"unknown authority: {authority!r}; expected one of {list(AUTHORITIES)}"
        )
    if _checked_phase(phase) != "a":
        raise PartitionError(
            "the successor artifact contract defines the Phase-A four-run plan only; "
            "a Phase-B run plan is not part of h2_execution_result_v1 (§ C3.2 item 10)"
        )

    states: dict[str, str] = {}
    for key, _ in SUCCESSOR_PREDICATES:
        record = predicate_results.get(key)
        if not isinstance(record, Mapping) or "state" not in record:
            raise PartitionError(f"predicate {key} is missing its state record")
        state = record["state"]
        if state not in PREDICATE_STATES:
            raise PartitionError(
                f"predicate {key} has an unknown state: {state!r}; expected one of "
                f"{list(PREDICATE_STATES)}"
            )
        states[key] = state
    unknown = sorted(set(predicate_results) - set(states))
    if unknown:
        raise PartitionError(
            f"observation carries predicates outside the partition: {unknown}"
        )

    if authority == "non_qualifying_diagnostic":
        if execution_result is not None:
            raise PartitionError(
                "a diagnostic selects no terminal, so it names no execution result"
            )
        return Selection(DIAGNOSTIC_RESULT, None, None, None, True, phase)

    for key, result in SUCCESSOR_PREDICATES:
        if states[key] != FAIL_STATE:
            continue
        if result == "unclassified_execution_failure":
            return _selection(
                _named_execution_result(execution_result, result), phase=phase
            )
        if execution_result is not None:
            raise PartitionError(
                f"predicate {key} failed, which selects {result!r}: an execution "
                "result may name terminal 4's cause only when terminal 4 is selected"
            )
        return _selection(result, phase=phase)

    undecided = sorted(key for key, state in states.items() if state != PASS_STATE)
    if undecided:
        if states["execution_complete"] == PASS_STATE:
            raise PartitionError(
                f"predicates {undecided} are undecided while execution_complete "
                "passed: a complete execution decides every predicate, so this "
                "record contradicts itself"
            )
        return _selection(
            _named_execution_result(execution_result, "unclassified_execution_failure"),
            phase=phase,
        )

    if execution_result is not None:
        raise PartitionError(
            "every predicate passed, so no execution result may be named"
        )
    return _selection("measurement_pass", phase=phase)


def binding_agreement_reasons(
    result: str,
    *,
    authority: str,
    selected_terminal: str | None,
    failed_stage: str | None,
    input_monitor: Mapping[str, Any],
    ordered_runs: Sequence[Mapping[str, Any]],
    identity_probe_present: bool,
) -> tuple[str, ...]:
    """Cross-check `result.json` against `runtime_binding.json`.

    No JSON Schema sees two files at once, so Review Correction 9's cross-artifact
    rules live here and the archive-only verifier imports them. What they are
    **not** is unconditional: stage evidence is subordinate to the authority
    boundary and to the § 7 order, and a rule that ignores either makes truthful
    records unformable.

    * **A diagnostic records stage evidence and demands nothing.** It resolves to
      `diagnostic_complete` whatever it observed, so requiring a failed stage or a
      recorded mutation to change its result would contradict the very rule
      `select_successor_result` implements.
    * **The stage → token direction holds only when terminal 4 is the ordered
      winner.** A build that failed *and* a bound input that moved is a real
      observation: terminal 1 outranks terminal 4, so the result is
      `input_mutated` while `failed_stage` stays `build` as subordinate evidence.
      Demanding `build_failed` there would let stage evidence overturn a
      higher-order finding — and, combined with the mutation rule, would leave no
      admissible result at all.
    * **The token → stage direction holds unconditionally.** `build_failed` that
      does not name a failed build is a label, which is the defect the two
      re-admitted tokens exist to avoid.
    * **The mutation rule stays biconditional for a measurement**, because
      terminal 1 is the highest order: a recorded change cannot lose to anything.
    * **A measurement that selected no terminal cannot carry a failed stage.**
      `measurement_pass` requires `execution_complete` to pass, so a non-null
      `failed_stage` would have the two files describing different executions.
      Subordinate evidence is admissible under a terminal, never under the
      non-terminal progression.
    * **A finding must be reachable at the point the execution stopped**, which a
      terminal-level rule cannot express because a terminal is not a time. A stage
      failure means no measurement run ever started, so `capture_perturbs_policy`
      and `packet_invalid` name evidence that cannot exist yet, and
      `behavior_probe_moved` names a probe the binding is forbidden to carry.
      Only the stage-independent findings — a monitored mutation, or a binding that
      disagrees with the spec on members required at every stage — survive a stage
      failure. This is the axis the previous revision lost by generalising the
      mutation case into "terminals 1 to 3 may carry any stage failure".

    `selected_terminal` is the terminal the ruler selects from the *predicates*
    (`select_successor_result(...).terminal`), never the terminal the archive
    recorded — deriving it from `result` would make this check circular, since
    `result` is what it is checking.
    """
    if authority not in AUTHORITIES:
        raise PartitionError(
            f"unknown authority: {authority!r}; expected one of {list(AUTHORITIES)}"
        )
    if result not in RESULT_TO_TERMINAL and result != DIAGNOSTIC_RESULT:
        raise PartitionError(f"unmapped controller result: {result}")
    if selected_terminal is not None and selected_terminal not in {
        terminal.name for terminal in TERMINALS
    }:
        raise PartitionError(f"unknown selected terminal: {selected_terminal!r}")
    if failed_stage is not None and failed_stage not in BINDABLE_FAILURE_STAGES:
        raise PartitionError(
            f"unknown failed stage: {failed_stage!r}; expected null or one of "
            f"{list(BINDABLE_FAILURE_STAGES)} — the earlier stages bind nothing"
        )
    for key in ("changed_count", "final_drain_clean"):
        if key not in input_monitor:
            raise PartitionError(f"input monitor record is missing {key}")
    run_states: list[str] = []
    for index, run in enumerate(ordered_runs):
        state = run.get("state")
        if state not in RUN_STATES:
            raise PartitionError(
                f"ordered run {index} has an unknown state: {state!r}; expected one "
                f"of {list(RUN_STATES)}"
            )
        run_states.append(state)
    started = [state for state in run_states if state != RUN_NOT_STARTED_STATE]

    if authority == "non_qualifying_diagnostic":
        if selected_terminal is not None:
            raise PartitionError(
                "a diagnostic selects no terminal, so it has no selected terminal to "
                "cross-check (§ Review Correction 5)"
            )
        # Every failed predicate and every stage failure is recorded; none of it
        # changes `diagnostic_complete`, and none of it may be demanded here.
        return ()

    reasons: list[str] = []
    required = RESULT_REQUIRES_FAILED_STAGE.get(result)
    if required is not None and failed_stage != required:
        reasons.append(
            f"{result} requires failed_stage {required!r}, and the binding "
            f"records {failed_stage!r}"
        )

    if failed_stage is not None and selected_terminal is None:
        # The one verdict a stage failure cannot sit under. `measurement_pass`
        # requires `execution_complete` to pass, and a failed stage says the
        # execution did not reach the end of the retained six — so the two files
        # would be describing different executions. Subordinate evidence is
        # admissible under a *terminal*, never under the non-terminal progression.
        reasons.append(
            "a non-terminal measurement pass requires failed_stage null, and the "
            f"binding records {failed_stage!r}: a passed measurement decided every "
            "predicate, so no retained stage can have failed"
        )

    if failed_stage is not None:
        # No measurement run starts until every retained stage completed, and
        # measurement mode is fail-fast, so a stage failure means none did.
        if started:
            reasons.append(
                f"failed_stage {failed_stage!r} stopped the execution before the "
                f"measurement runs, so all four must be {RUN_NOT_STARTED_STATE!r}, "
                f"and {len(started)} of them are not"
            )
        if result in RUN_DERIVED_RESULTS:
            reasons.append(
                f"{result} names run-derived evidence, which cannot exist under "
                f"failed_stage {failed_stage!r}: no measurement run started"
            )
        if result in PROBE_DERIVED_RESULTS:
            # Subsumes the presence check below and says *why* the probe is absent;
            # emitting both would report one condition twice.
            reasons.append(
                f"{result} names the identity probe, which the binding cannot carry "
                f"under failed_stage {failed_stage!r}"
            )
    else:
        if result in RUN_DERIVED_RESULTS and not started:
            reasons.append(
                f"{result} names run-derived evidence while no measurement run started"
            )
        if result in PROBE_DERIVED_RESULTS and not identity_probe_present:
            reasons.append(
                f"{result} requires a computed identity probe, and the binding "
                "records none"
            )

    if failed_stage is not None and selected_terminal == EXECUTION_INVALID_TERMINAL:
        expected = [
            name
            for name, stage in RESULT_REQUIRES_FAILED_STAGE.items()
            if stage == failed_stage
        ]
        if expected and result != expected[0]:
            reasons.append(
                f"the ordered verdict is {EXECUTION_INVALID_TERMINAL}, so failed_stage "
                f"{failed_stage!r} requires result {expected[0]!r}, not {result!r}"
            )
        elif (
            failed_stage in CATCH_ALL_FAILURE_STAGES
            and result != "unclassified_execution_failure"
        ):
            reasons.append(
                f"the ordered verdict is {EXECUTION_INVALID_TERMINAL} and failed_stage "
                f"{failed_stage!r} has no dedicated result token, so it requires "
                f"'unclassified_execution_failure', not {result!r}"
            )

    mutated = bool(input_monitor["changed_count"]) or not bool(
        input_monitor["final_drain_clean"]
    )
    if result == RESULT_REQUIRES_INPUT_MUTATION and not mutated:
        reasons.append(
            f"{result} requires the monitor to record a change or an unclean final "
            "drain, and it records neither"
        )
    if mutated and result != RESULT_REQUIRES_INPUT_MUTATION:
        reasons.append(
            "a recorded input change outranks every other finding, so it requires "
            f"result {RESULT_REQUIRES_INPUT_MUTATION!r}, and the result is {result!r}"
        )
    return tuple(reasons)


def _named_execution_result(named: str | None, default: str) -> str:
    """Let the caller name terminal 4's cause; every legal name still maps to it."""
    if named is None:
        return default
    if RESULT_TO_TERMINAL.get(named) != EXECUTION_INVALID_TERMINAL:
        raise PartitionError(f"execution_result {named!r} does not map to terminal 4")
    return named


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
        # The successor vocabulary, published for the same reason as the phase
        # narrowing above: an implementer reading only this payload must reach the
        # verdict `select_successor_result` reaches, including the two rules the
        # four-state predicate needs (§ 20.8).
        "successor_vocabulary": {
            "authorities": list(AUTHORITIES),
            "diagnostic_result": DIAGNOSTIC_RESULT,
            "diagnostic_selects_no_terminal": True,
            "ordered_predicates": [list(item) for item in SUCCESSOR_PREDICATES],
            "predicate_states": list(PREDICATE_STATES),
            "undecided_states": list(UNDECIDED_STATES),
            "decided_failure_outranks_undecided": True,
            "undecided_requires_incomplete_execution": True,
            "predicate_renames": dict(SUCCESSOR_TO_LEGACY_PREDICATE),
            # Not every successor predicate renames a legacy one, and saying so is
            # what stops a reader from inferring the missing mapping.
            "predicates_without_a_legacy_name": sorted(
                SUCCESSOR_WITHOUT_LEGACY_PREDICATE
            ),
            "inverted_polarity_predicates": sorted(INVERTED_POLARITY_PREDICATES),
            "legacy_result_superseded_by": dict(LEGACY_RESULT_SUPERSEDED_BY),
            # Retired, not superseded: no successor observation selects these.
            "retired_successor_results": sorted(RETIRED_SUCCESSOR_RESULTS),
            # The cross-artifact rules. A named finding also requires its own
            # predicate to be `fail`: an undecided predicate names nothing, or two
            # results would share one terminal for one observation.
            "named_finding_requires_decided_fail": True,
            "binding_stages": list(BINDING_STAGES),
            "bindable_failure_stages": list(BINDABLE_FAILURE_STAGES),
            # Unconditional: a named cause must name its stage.
            "result_requires_failed_stage": dict(RESULT_REQUIRES_FAILED_STAGE),
            "catch_all_failure_stages": list(CATCH_ALL_FAILURE_STAGES),
            # Conditional: the reverse direction would otherwise let subordinate
            # stage evidence overturn a higher-order finding, and together with the
            # mutation rule would leave a build failure under a moved input with no
            # admissible result at all.
            "failed_stage_requires_result_only_when_terminal": (
                EXECUTION_INVALID_TERMINAL
            ),
            # Reachability, not terminals. A terminal is not a time: two results
            # sharing terminal 1 differ in whether their evidence can exist yet, so
            # the admissible companions of a stage failure are named per result.
            # The earlier terminal-level form said "terminals 1–3 may carry any
            # stage failure", which admitted a capture-perturbation finding from an
            # execution that stopped at `build`.
            "stage_independent_results": list(STAGE_INDEPENDENT_RESULTS),
            "probe_derived_results": list(PROBE_DERIVED_RESULTS),
            "run_derived_results": list(RUN_DERIVED_RESULTS),
            "results_admissible_with_a_failed_stage": sorted(
                {
                    *STAGE_INDEPENDENT_RESULTS,
                    *(
                        result
                        for result, terminal in RESULT_TO_TERMINAL.items()
                        if terminal == EXECUTION_INVALID_TERMINAL
                    ),
                }
            ),
            "failed_stage_requires_unstarted_runs": True,
            "failed_stage_forbidden_under_non_terminal_progression": True,
            # Biconditional under measurement authority only: terminal 1 is the
            # highest order, so a recorded change cannot lose to anything.
            "result_requires_input_mutation": RESULT_REQUIRES_INPUT_MUTATION,
            "diagnostic_records_evidence_without_demanding_a_result": True,
            "cross_artifact_checker": "binding_agreement_reasons",
        },
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
