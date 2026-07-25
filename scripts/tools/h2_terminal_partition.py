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

Usage:
  uv run python scripts/tools/h2_terminal_partition.py --explain
  uv run python scripts/tools/h2_terminal_partition.py --select result.json
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple

PARTITION_SCHEMA = "h2_terminal_partition_v1"


class Terminal(NamedTuple):
    order: int
    name: str
    condition: str
    transition: str
    phase_a_reachable: bool


# Ordered; first applicable is authoritative (declaration § 7).
TERMINALS: tuple[Terminal, ...] = (
    Terminal(
        1,
        "H2_INPUT_MUTATED_DURING_MEASUREMENT",
        "a bound input was written during the invocation, or the behavior axis at "
        "launch differs from the reference bound in F, or the Layer-P certificate "
        "does not match F",
        "closes the H2 measurement unit; object state unchanged; candidate set stays "
        "empty; a fresh I→F→S and a separate authorization would be required",
        True,
    ),
    Terminal(
        2,
        "H2_CAPTURE_PERTURBS_POLICY",
        "execution completed and any A7.6 capture-off/on equality differs",
        "closes the observational-capture route itself: decision-neutral shadow "
        "capture is not achievable at this ABI, so grounding must proceed by "
        "native-side reproduction or not at all",
        True,
    ),
    Terminal(
        3,
        "H2_PACKET_INVALID",
        "non-perturbation held but any packet, exposure, overflow, native-universe, "
        "conservation, cross-repeat canonical digest, or replay predicate fails",
        "closes this measurement; routes to a separate capture-ABI-delta charter",
        True,
    ),
    Terminal(
        4,
        "H2_MEASUREMENT_EXECUTION_INVALID",
        "after the sealed launch: nonzero build, extension/plugin load failure, "
        "runner nonzero, deadline exhausted, serialization failure, missing or "
        "unreadable required artifact, or any unclassified execution failure",
        "closes this measurement with no partial-capture reinterpretation; a fresh "
        "chain would be required",
        True,
    ),
    Terminal(
        5,
        "H2_FULL_COMMIT_CAPTURE_FAITHFUL",
        "every preceding condition false, all three capture-on packets and all "
        "verifications pass, and the frozen unlabelled seven-sequence Phase B is "
        "complete",
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
    "behavior_axis_moved": "H2_INPUT_MUTATED_DURING_MEASUREMENT",
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
    ("behavior_axis_equals_freeze", "behavior_axis_moved"),
    ("layer_p_certificate_matches_freeze", "certificate_mismatch"),
    ("capture_off_on_equal", "capture_perturbs_policy"),
    ("packets_valid", "packet_invalid"),
    ("execution_complete", "unclassified_execution_failure"),
)

# Predicates whose *false* value selects the failure, vs whose *true* value does.
_TRUE_IS_FAILURE = frozenset({"bound_input_mutated"})


class PartitionError(RuntimeError):
    pass


class Selection(NamedTuple):
    result: str
    terminal: str | None
    order: int | None
    transition: str | None
    phase_a_emittable: bool

    def describe(self) -> str:
        if self.terminal is None:
            return (
                f"result={self.result} → no H2 terminal (non-terminal progression); "
                "terminal 5 requires the Phase-B artifact"
            )
        return f"result={self.result} → {self.terminal} (ordered #{self.order})"


def terminal_by_name(name: str) -> Terminal:
    for terminal in TERMINALS:
        if terminal.name == name:
            return terminal
    raise PartitionError(f"unknown terminal: {name}")


def select_terminal(
    observation: Mapping[str, Any], *, phase_b_complete: bool = False
) -> Selection:
    """Map one observation to exactly one terminal. Total and order-sensitive.

    Every predicate in `ORDERED_PREDICATES` must be present: a missing predicate
    is a defect in the caller, and guessing a default is how a fail-closed check
    becomes fail-open. An explicit `execution_result` may name a specific
    execution failure so terminal 4's cause is recorded rather than flattened.
    """
    missing = [key for key, _ in ORDERED_PREDICATES if key not in observation]
    if missing:
        raise PartitionError(f"observation is missing predicates: {missing}")

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
        return _selection(result)

    if not phase_b_complete:
        return _selection("measurement_pass")
    return _selection("measurement_pass", terminal_override=TERMINALS[4].name)


def _selection(result: str, *, terminal_override: str | None = None) -> Selection:
    if result not in RESULT_TO_TERMINAL:
        raise PartitionError(f"unmapped controller result: {result}")
    name = terminal_override or RESULT_TO_TERMINAL[result]
    if name is None:
        return Selection(result, None, None, None, True)
    terminal = terminal_by_name(name)
    return Selection(
        result,
        terminal.name,
        terminal.order,
        terminal.transition,
        terminal.phase_a_reachable,
    )


def as_payload() -> dict[str, Any]:
    return {
        "schema": PARTITION_SCHEMA,
        "note": (
            "provenance_invalid is absent as a predicate: enumerative closure "
            "membership is not computed, so no observation can select it. Witness "
            "fields carry no decision authority."
        ),
        "ordered_predicates": [list(item) for item in ORDERED_PREDICATES],
        "result_to_terminal": RESULT_TO_TERMINAL,
        "terminals": [terminal._asdict() for terminal in TERMINALS],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain", action="store_true", help="print the partition")
    parser.add_argument(
        "--select",
        type=Path,
        default=None,
        help="select a terminal from a JSON observation",
    )
    parser.add_argument(
        "--phase-b-complete",
        action="store_true",
        help="the frozen seven-sequence Phase-B artifact exists",
    )
    args = parser.parse_args(argv)

    if args.explain:
        for terminal in TERMINALS:
            reach = "A" if terminal.phase_a_reachable else "B only"
            print(f"{terminal.order}. {terminal.name}  [{reach}]")
            print(f"     when: {terminal.condition}")
            print(f"     then: {terminal.transition}")
        return 0

    if args.select is None:
        parser.error("one of --explain or --select is required")

    try:
        observation = json.loads(args.select.read_text(encoding="utf-8"))
        selection = select_terminal(observation, phase_b_complete=args.phase_b_complete)
    except (PartitionError, json.JSONDecodeError) as exc:
        print(f"terminal selection failed: {exc}", file=sys.stderr)
        return 1
    print(selection.describe())
    if selection.transition:
        print(f"mainline transition: {selection.transition}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
