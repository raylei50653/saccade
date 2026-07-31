"""The terminal partition must be ordered, exhaustive, and blind to witness data.

§ 20.8's governing test is that two independent implementers, given the sealed
declaration and its frozen inputs, record the **bit-identical** terminal. Three
things can quietly break that:

  * **order** — the partition is first-applicable, so a run that both mutated an
    input and produced an invalid packet has exactly one right answer;
  * **exhaustiveness** — every controller result maps to a terminal or to the
    explicit non-terminal progression. An unmapped result is what § 20.8 item 3
    calls a declaration defect, and H0's own A7.7 keeps a mandatory catch-all for
    exactly this reason;
  * **blindness to witness** — physical hashes and loaded closures may be
    recorded but must never select a terminal. That inversion is the redesign; a
    test that only asserted it in prose would let it rot.

Also pinned: `provenance_invalid` cannot be reached at all. Its absence is the
substantive change, so it gets an explicit test rather than an implicit one.

Phase-awareness (§ C3.7) is pinned here too, because a partition that silently
defaulted the phase would apply one chain's ruler to the other's evidence. Three
things it must keep separate: the per-phase completion counts, the § C3.6
admission gate as a *pre-terminal* object whose failure selects nothing, and
Phase B's terminal 1 meaning bound-input mutation alone.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import itertools

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_terminal_partition as tp  # noqa: E402


def _clean() -> dict[str, object]:
    """An observation with every predicate satisfied."""
    return {
        "bound_input_mutated": False,
        "behavior_probe_equals_freeze": True,
        "layer_p_certificate_matches_freeze": True,
        "capture_off_on_equal": True,
        "packets_valid": True,
        "execution_complete": True,
    }


def _admitted() -> tp.Admission:
    """A passed § C3.6 gate: what a Phase-B selection is only reachable through."""
    return tp.evaluate_admission(
        {key: True for key, _ in tp.ADMISSION_CONDITIONS}, phase="b"
    )


# --------------------------------------------------------------------------- #
# Shape                                                                        #
# --------------------------------------------------------------------------- #
def test_the_partition_is_five_ordered_terminals() -> None:
    assert [t.order for t in tp.TERMINALS] == [1, 2, 3, 4, 5]
    assert len({t.name for t in tp.TERMINALS}) == 5


def test_every_terminal_names_a_mainline_transition() -> None:
    """§ 20.7: a study whose failure mode is 'describe more and continue' is not a
    mainline study. Every terminal, including the negative ones, must transition."""
    for terminal in tp.TERMINALS:
        assert terminal.transition.strip()
        assert "describe more" not in terminal.transition


def test_only_the_phase_b_terminal_is_unreachable_in_phase_a() -> None:
    unreachable = [t.name for t in tp.TERMINALS if not t.phase_a_reachable]
    assert unreachable == ["H2_FULL_COMMIT_CAPTURE_FAITHFUL"]


def test_provenance_invalid_is_not_reachable() -> None:
    """The substantive change: enumerative closure membership is not computed."""
    assert all("PROVENANCE" not in t.name for t in tp.TERMINALS)
    assert "provenance_invalid" not in tp.RESULT_TO_TERMINAL


def test_every_result_maps_to_a_known_terminal_or_to_none() -> None:
    names = {t.name for t in tp.TERMINALS}
    for result, terminal in tp.RESULT_TO_TERMINAL.items():
        assert terminal is None or terminal in names, result


def test_an_execution_catch_all_exists() -> None:
    """§ 20.8 item 3 / A7.7: the catch-all is mandatory, never unmapped."""
    assert (
        tp.RESULT_TO_TERMINAL["unclassified_execution_failure"]
        == "H2_MEASUREMENT_EXECUTION_INVALID"
    )


# --------------------------------------------------------------------------- #
# Selection                                                                    #
# --------------------------------------------------------------------------- #
def test_a_clean_phase_a_run_emits_no_terminal() -> None:
    selection = tp.select_terminal(_clean(), phase="a")
    assert selection.terminal is None
    assert selection.result == "measurement_pass"
    assert "Phase-B" in selection.describe()


def test_phase_b_completion_reaches_terminal_five() -> None:
    selection = tp.select_terminal(
        _clean(), phase="b", phase_b_complete=True, admission=_admitted()
    )
    assert selection.terminal == "H2_FULL_COMMIT_CAPTURE_FAITHFUL"
    assert "precondition" in selection.transition


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("bound_input_mutated", True, "H2_INPUT_MUTATED_DURING_MEASUREMENT"),
        (
            "behavior_probe_equals_freeze",
            False,
            "H2_INPUT_MUTATED_DURING_MEASUREMENT",
        ),
        (
            "layer_p_certificate_matches_freeze",
            False,
            "H2_INPUT_MUTATED_DURING_MEASUREMENT",
        ),
        ("capture_off_on_equal", False, "H2_CAPTURE_PERTURBS_POLICY"),
        ("packets_valid", False, "H2_PACKET_INVALID"),
        ("execution_complete", False, "H2_MEASUREMENT_EXECUTION_INVALID"),
    ],
)
def test_each_predicate_selects_its_terminal(
    key: str, value: bool, expected: str
) -> None:
    observation = {**_clean(), key: value}
    assert tp.select_terminal(observation, phase="a").terminal == expected


def test_order_decides_when_several_predicates_fail() -> None:
    """A mutated input outranks an invalid packet: if the inputs moved, the packet
    describes nothing, so reporting `PACKET_INVALID` would name the wrong cause."""
    observation = {**_clean(), "bound_input_mutated": True, "packets_valid": False}
    selection = tp.select_terminal(observation, phase="a")
    assert selection.terminal == "H2_INPUT_MUTATED_DURING_MEASUREMENT"
    assert selection.order == 1


def test_perturbation_outranks_packet_invalidity() -> None:
    observation = {**_clean(), "capture_off_on_equal": False, "packets_valid": False}
    assert tp.select_terminal(observation, phase="a").order == 2


def test_a_named_execution_cause_is_recorded_but_cannot_move_the_terminal() -> None:
    for cause in (
        "build_failed",
        "extension_load_failed",
        "runner_nonzero",
        "runner_timeout",
        "serialization_failed",
        "artifact_missing_or_unreadable",
    ):
        observation = {
            **_clean(),
            "execution_complete": False,
            "execution_result": cause,
        }
        selection = tp.select_terminal(observation, phase="a")
        assert selection.result == cause
        assert selection.terminal == "H2_MEASUREMENT_EXECUTION_INVALID"


def test_an_execution_cause_that_does_not_map_to_terminal_four_is_refused() -> None:
    observation = {
        **_clean(),
        "execution_complete": False,
        "execution_result": "capture_perturbs_policy",
    }
    with pytest.raises(tp.PartitionError, match="terminal 4"):
        tp.select_terminal(observation, phase="a")


# --------------------------------------------------------------------------- #
# Fail-closed                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", [key for key, _ in tp.ORDERED_PREDICATES])
def test_a_missing_predicate_fails_closed(key: str) -> None:
    observation = _clean()
    del observation[key]
    with pytest.raises(tp.PartitionError, match="missing predicates"):
        tp.select_terminal(observation, phase="a")


def test_a_non_boolean_predicate_fails_closed() -> None:
    """A truthy string would silently satisfy a safety predicate."""
    observation = {**_clean(), "packets_valid": "yes"}
    with pytest.raises(tp.PartitionError, match="not a bool"):
        tp.select_terminal(observation, phase="a")


def test_witness_fields_cannot_select_a_terminal() -> None:
    """Declaration § 4.1, enforced rather than asserted."""
    witness_laden = {
        **_clean(),
        "extension_sha256": "0" * 64,
        "tool_runtime_count": 4518,
        "observed_regular_files": ["/usr/lib/libtbbmalloc.so"],
        "gpu_uuid": "GPU-deadbeef",
        "build_artifact_absent_from_attestation": True,
    }
    selection = tp.select_terminal(witness_laden, phase="a")
    assert selection.terminal is None, (
        "a witness field selected a terminal — the whole inversion of the redesign "
        "is that physical observations are recorded, never decisive"
    )
    allowed = {key for key, _ in tp.ORDERED_PREDICATES} | {"execution_result"}
    assert set(tp.as_payload()["ordered_predicates"][0]) <= {
        "bound_input_mutated",
        "input_mutated",
    }
    assert allowed.isdisjoint({"extension_sha256", "tool_runtime_count", "gpu_uuid"})


def test_the_payload_is_serializable_and_self_describing() -> None:
    payload = tp.as_payload()
    assert payload["schema"] == tp.PARTITION_SCHEMA
    assert "provenance_invalid is absent" in payload["note"]
    assert len(payload["terminals"]) == 5


def test_the_cli_explains_without_arguments_beyond_the_flag() -> None:
    assert tp.main(["--explain"]) == 0


# --------------------------------------------------------------------------- #
# Phase-awareness (§ C3.7)                                                     #
# --------------------------------------------------------------------------- #
def test_the_phase_is_required_and_never_inferred() -> None:
    """A default would apply one chain's ruler to the other's evidence."""
    with pytest.raises(TypeError):
        tp.select_terminal(_clean())  # type: ignore[call-arg]


def test_an_unknown_phase_fails_closed() -> None:
    with pytest.raises(tp.PartitionError, match="unknown phase"):
        tp.select_terminal(_clean(), phase="c")


def test_each_phase_declares_its_own_completion_counts() -> None:
    """§ C3.7 item 2: the seven-sequence Phase B is 21 capture-on packets and
    7 capture-off runs; a Phase-A block is one sequence's four-run block."""
    assert tp.PHASE_COMPLETION["a"] == {
        "required_sequences": 1,
        "required_capture_on_packets": 3,
        "required_capture_off_runs": 1,
    }
    assert tp.PHASE_COMPLETION["b"] == {
        "required_sequences": 7,
        "required_capture_on_packets": 21,
        "required_capture_off_runs": 7,
    }


def test_terminal_five_states_a_different_condition_per_phase() -> None:
    terminal = tp.terminal_by_name("H2_FULL_COMMIT_CAPTURE_FAITHFUL")
    assert set(terminal.condition) == set(tp.PHASES)
    assert terminal.condition_for("a") != terminal.condition_for("b")
    assert "21 capture-on packets" in terminal.condition_for("b")
    assert "7 capture-off run" in terminal.condition_for("b")
    assert "3 capture-on packets" in terminal.condition_for("a")


def test_phase_b_terminal_one_is_bound_input_mutation_only() -> None:
    """§ C3.6: the probe and certificate checks moved into admission, so Phase B's
    terminal 1 names exactly one thing."""
    terminal = tp.terminal_by_name("H2_INPUT_MUTATED_DURING_MEASUREMENT")
    phase_b = terminal.condition_for("b")
    assert "bound input" in phase_b
    assert "admission gate" in phase_b
    # Phase A's § 7 condition is untouched: this correction narrows nothing there.
    assert "behavior probe at launch differs" in terminal.condition_for("a")


def test_phase_b_complete_is_refused_under_phase_a() -> None:
    with pytest.raises(tp.PartitionError, match="admissible only under phase='b'"):
        tp.select_terminal(_clean(), phase="a", phase_b_complete=True)


def test_a_phase_a_chain_takes_no_admission_verdict() -> None:
    with pytest.raises(tp.PartitionError, match="takes no admission verdict"):
        tp.select_terminal(_clean(), phase="a", admission=_admitted())


def test_a_phase_b_observation_may_not_contradict_the_passed_gate() -> None:
    """Admission already decided these two; a later move is a bound-input
    mutation, which has its own predicate. Reporting one false here is incoherent."""
    for key in sorted(tp.PHASE_B_ADMISSION_DECIDED):
        observation = {**_clean(), key: False}
        with pytest.raises(tp.PartitionError, match="contradicts a passed admission"):
            tp.select_terminal(observation, phase="b", admission=_admitted())
        # The same observation is a perfectly ordinary Phase-A terminal 1.
        assert (
            tp.select_terminal(observation, phase="a").terminal
            == "H2_INPUT_MUTATED_DURING_MEASUREMENT"
        )


# --------------------------------------------------------------------------- #
# The admission gate is pre-terminal (§ C3.6)                                  #
# --------------------------------------------------------------------------- #
def test_admission_conditions_are_not_partition_predicates() -> None:
    """They are evaluated before S_B is consumed; making one an ORDERED_PREDICATE
    would record an inadmissible launch as an epistemic result."""
    predicates = {key for key, _ in tp.ORDERED_PREDICATES}
    admission = {key for key, _ in tp.ADMISSION_CONDITIONS}
    assert len(admission) == 5
    # The certificate check appears in both vocabularies, in different roles:
    # a Phase-A terminal condition, and a Phase-B pre-launch gate.
    assert admission - predicates == {
        "phase_a_evidence_root_verifies",
        "phase_a_observation_selects_no_terminal",
        "axes_and_probe_equal_freeze",
        "prior_attempts_complete_and_verified",
    }


def test_a_passed_admission_selects_no_terminal_of_its_own() -> None:
    admission = _admitted()
    assert admission.admitted
    assert admission.terminal is None
    assert admission.reasons == ()


@pytest.mark.parametrize("failing", [key for key, _ in tp.ADMISSION_CONDITIONS])
def test_an_admission_failure_yields_no_terminal(failing: str) -> None:
    record = {key: True for key, _ in tp.ADMISSION_CONDITIONS}
    record[failing] = False
    admission = tp.evaluate_admission(record, phase="b")

    assert not admission.admitted
    assert admission.terminal is None
    assert tp.ADMISSION_FAILURE_CLASS in admission.describe()
    assert "no authorization spent" in admission.describe()

    # And it cannot be laundered into one by calling the partition anyway.
    with pytest.raises(tp.PartitionError, match="admission was refused"):
        tp.select_terminal(_clean(), phase="b", admission=admission)


def test_a_phase_b_selection_without_an_admission_verdict_fails_closed() -> None:
    with pytest.raises(tp.PartitionError, match="requires the § C3.6 admission"):
        tp.select_terminal(_clean(), phase="b")


def test_admission_reports_every_failed_condition_not_just_the_first() -> None:
    record = {key: False for key, _ in tp.ADMISSION_CONDITIONS}
    admission = tp.evaluate_admission(record, phase="b")
    assert admission.reasons == tuple(reason for _, reason in tp.ADMISSION_CONDITIONS)


def test_admission_is_phase_b_only() -> None:
    """§ C3.6 narrows the Phase-B chain and explicitly does not align Phase A."""
    record = {key: True for key, _ in tp.ADMISSION_CONDITIONS}
    with pytest.raises(tp.PartitionError, match="phase 'b' only"):
        tp.evaluate_admission(record, phase="a")


@pytest.mark.parametrize("key", [key for key, _ in tp.ADMISSION_CONDITIONS])
def test_a_missing_admission_condition_fails_closed(key: str) -> None:
    record = {name: True for name, _ in tp.ADMISSION_CONDITIONS}
    del record[key]
    with pytest.raises(tp.PartitionError, match="missing conditions"):
        tp.evaluate_admission(record, phase="b")


def test_a_non_boolean_admission_condition_fails_closed() -> None:
    record = {key: True for key, _ in tp.ADMISSION_CONDITIONS}
    record["axes_and_probe_equal_freeze"] = "yes"
    with pytest.raises(tp.PartitionError, match="not a bool"):
        tp.evaluate_admission(record, phase="b")


# --------------------------------------------------------------------------- #
# Phase B is total: no clean run may end without a terminal                    #
# --------------------------------------------------------------------------- #
def test_a_clean_phase_b_observation_cannot_end_without_a_terminal() -> None:
    """By selection time admission has passed and § C3.5.1 step 5 has consumed
    `S_B`. A no-terminal return would spend an authorization and record nothing —
    the state § C3.5.1 exists to make unformable."""
    with pytest.raises(tp.PartitionError, match="no non-terminal progression"):
        tp.select_terminal(_clean(), phase="b", admission=_admitted())


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("bound_input_mutated", True, "H2_INPUT_MUTATED_DURING_MEASUREMENT"),
        ("capture_off_on_equal", False, "H2_CAPTURE_PERTURBS_POLICY"),
        ("packets_valid", False, "H2_PACKET_INVALID"),
        ("execution_complete", False, "H2_MEASUREMENT_EXECUTION_INVALID"),
    ],
)
def test_phase_b_failure_terminals_do_not_need_a_complete_artifact(
    key: str, value: bool, expected: str
) -> None:
    """Terminals 1-4 are exactly the cases where the seven sequences did *not*
    complete. Requiring completeness to reach them would make the failures of an
    incomplete Phase B unselectable."""
    observation = {**_clean(), key: value}
    selection = tp.select_terminal(
        observation, phase="b", phase_b_complete=False, admission=_admitted()
    )
    assert selection.terminal == expected
    assert selection.phase == "b"


def test_a_clean_phase_a_run_is_still_a_non_terminal_progression() -> None:
    """The narrowing above is Phase-B only: Phase A's pass is the entry to Phase B
    and must stay non-terminal."""
    selection = tp.select_terminal(_clean(), phase="a")
    assert selection.terminal is None
    assert selection.result == "measurement_pass"


def test_the_payload_publishes_the_phase_b_narrowing() -> None:
    """§ 20.8's test is two *independent* implementers. One may consume only this
    payload, so a narrowing that lives solely in `select_terminal` would let the
    two record different terminals for the same observation."""
    narrowing = tp.as_payload()["phase_narrowing"]

    assert set(narrowing) == set(tp.PHASES)
    assert narrowing["b"]["forbidden_results"] == list(tp.PHASE_B_FORBIDDEN_RESULTS)
    assert narrowing["b"]["has_non_terminal_progression"] is False
    assert narrowing["b"]["clean_observation_requires_phase_b_complete"] is True
    assert narrowing["a"]["forbidden_results"] == []
    assert narrowing["a"]["has_non_terminal_progression"] is True


def test_the_forbidden_results_are_exactly_the_admission_decided_ones() -> None:
    """Derived, not transcribed: the payload cannot drift from the check."""
    assert set(tp.PHASE_B_FORBIDDEN_RESULTS) == {
        result
        for key, result in tp.ORDERED_PREDICATES
        if key in tp.PHASE_B_ADMISSION_DECIDED
    }
    # And each one still maps to terminal 1 in the phase-independent union.
    for result in tp.PHASE_B_FORBIDDEN_RESULTS:
        assert tp.RESULT_TO_TERMINAL[result] == "H2_INPUT_MUTATED_DURING_MEASUREMENT"


def test_the_payload_and_the_function_agree_on_every_phase_b_result() -> None:
    """Cross-check the two consumption routes rather than asserting each alone."""
    payload = tp.as_payload()
    forbidden = set(payload["phase_narrowing"]["b"]["forbidden_results"])
    for key, result in tp.ORDERED_PREDICATES:
        observation = {**_clean(), key: key in tp._TRUE_IS_FAILURE}
        if result in forbidden:
            with pytest.raises(tp.PartitionError):
                tp.select_terminal(observation, phase="b", admission=_admitted())
        else:
            selection = tp.select_terminal(
                observation, phase="b", admission=_admitted()
            )
            assert selection.terminal == payload["result_to_terminal"][selection.result]


def test_the_payload_carries_the_phase_and_admission_vocabulary() -> None:
    payload = tp.as_payload()
    assert payload["phases"] == ["a", "b"]
    assert payload["phase_completion"] == tp.PHASE_COMPLETION
    assert payload["admission_failure_class"] == tp.ADMISSION_FAILURE_CLASS
    assert len(payload["admission_conditions"]) == 5


def test_the_cli_explains_one_phase_and_requires_it_to_select(tmp_path: Path) -> None:
    assert tp.main(["--explain", "--phase", "b"]) == 0
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps(_clean()), encoding="utf-8")
    assert tp.main(["--phase", "a", "--select", observation.as_posix()]) == 0
    with pytest.raises(SystemExit):
        tp.main(["--select", observation.as_posix()])


def test_the_cli_refuses_a_failed_admission_before_any_selection(
    tmp_path: Path,
) -> None:
    record = {key: True for key, _ in tp.ADMISSION_CONDITIONS}
    record["prior_attempts_complete_and_verified"] = False
    gate = tmp_path / "admission.json"
    gate.write_text(json.dumps(record), encoding="utf-8")
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps(_clean()), encoding="utf-8")
    assert (
        tp.main(
            [
                "--phase",
                "b",
                "--admit",
                gate.as_posix(),
                "--select",
                observation.as_posix(),
                "--phase-b-complete",
            ]
        )
        == 1
    )


# --------------------------------------------------------------------------- #
# The successor vocabulary (Review Correction 9)                               #
#                                                                              #
# `h2_execution_result_v1` renames three predicates — one with inverted         #
# polarity — and widens a predicate from a bool to four states. Two vocabularies #
# for one partition is exactly the § C5 shape that drifts on the next edit, so   #
# the mapping is pinned here in both directions, and so are the two rules the    #
# four-state predicate needs.                                                   #
# --------------------------------------------------------------------------- #
def _successor(**states: str) -> dict[str, dict[str, object]]:
    """A successor observation with every predicate passing unless overridden."""
    record = {
        key: {"state": "pass", "reasons": []} for key, _ in tp.SUCCESSOR_PREDICATES
    }
    for key, state in states.items():
        record[key] = {"state": state, "reasons": [] if state == "pass" else ["why"]}
    return record


def test_the_successor_partition_is_the_legacy_one_minus_what_was_retired() -> None:
    """Correction 10 shortened the partition; it did not reorder or renumber it.

    The successor is the legacy order with the retired probe entry removed and
    the certificate slot replaced by a narrower quantity that keeps the
    superseded result token. Stating it as a derivation rather than as a second
    literal list is the point: a future edit to either vocabulary that is not
    also one of these two operations fails here.
    """
    expected = [
        tp.LEGACY_RESULT_SUPERSEDED_BY.get(result, result)
        for _, result in tp.ORDERED_PREDICATES
        if result not in tp.RETIRED_SUCCESSOR_RESULTS
    ]
    assert [result for _, result in tp.SUCCESSOR_PREDICATES] == expected
    assert [
        tp.SUCCESSOR_TO_LEGACY_PREDICATE[key]
        for key, _ in tp.SUCCESSOR_PREDICATES
        if key not in tp.SUCCESSOR_WITHOUT_LEGACY_PREDICATE
    ] == [
        key
        for key, result in tp.ORDERED_PREDICATES
        if result not in tp.RETIRED_SUCCESSOR_RESULTS
        and result not in tp.LEGACY_RESULT_SUPERSEDED_BY
    ]
    # Every renamed result still selects the terminal its legacy name selected.
    for legacy, successor in tp.LEGACY_RESULT_SUPERSEDED_BY.items():
        assert tp.RESULT_TO_TERMINAL[successor] == tp.RESULT_TO_TERMINAL[legacy]


def test_a_retired_result_keeps_its_terminal_and_leaves_the_successor_vocabulary() -> (
    None
):
    """Retired, not deleted: the historical archives that recorded it still parse.

    `PROBE_DERIVED_RESULTS == ()` is the executable half of the same claim — no
    successor observation reaches the finding, because the class it belonged to
    is now empty.
    """
    assert tp.PROBE_DERIVED_RESULTS == ()
    for retired in tp.RETIRED_SUCCESSOR_RESULTS:
        assert retired in tp.RESULT_TO_TERMINAL
        assert retired not in {result for _, result in tp.SUCCESSOR_PREDICATES}
        assert retired not in tp.LEGACY_RESULT_SUPERSEDED_BY.values()


def test_no_successor_observation_can_select_a_retired_result() -> None:
    """Exhaustive over the predicate vector, not over the answers we expected."""
    states = ("pass", "fail", "error", "not_run")
    keys = [key for key, _ in tp.SUCCESSOR_PREDICATES]
    for combination in itertools.product(states, repeat=len(keys)):
        observation = {
            key: {"reasons": [], "state": state}
            for key, state in zip(keys, combination)
        }
        try:
            selection = tp.select_successor_result(
                observation, authority="exactly_once_measurement", phase="a"
            )
        except tp.PartitionError:
            continue
        assert selection.result not in tp.RETIRED_SUCCESSOR_RESULTS


def test_the_inverted_predicate_is_the_one_whose_polarity_flipped() -> None:
    """Mapping the names without the polarity gets terminal 1 exactly backwards."""
    inverted = {
        key
        for key, _ in tp.SUCCESSOR_PREDICATES
        if tp.SUCCESSOR_TO_LEGACY_PREDICATE.get(key) in tp._TRUE_IS_FAILURE
    }
    assert inverted == set(tp.INVERTED_POLARITY_PREDICATES)
    healthy = tp.select_successor_result(
        _successor(), authority="exactly_once_measurement", phase="a"
    )
    assert healthy.result == "measurement_pass"
    mutated = tp.select_successor_result(
        _successor(bound_input_unchanged="fail"),
        authority="exactly_once_measurement",
        phase="a",
    )
    assert mutated.order == 1


@pytest.mark.parametrize(
    ("failing", "expected_order"),
    [
        ("bound_input_unchanged", 1),
        ("runtime_projection_matches_resolved_run_spec", 1),
        ("capture_off_on_equal", 2),
        ("packets_valid", 3),
        ("execution_complete", 4),
    ],
)
def test_each_successor_predicate_selects_its_own_terminal(
    failing: str, expected_order: int
) -> None:
    selection = tp.select_successor_result(
        _successor(**{failing: "fail"}),
        authority="exactly_once_measurement",
        phase="a",
    )
    assert selection.order == expected_order
    assert selection.terminal == tp.RESULT_TO_TERMINAL[selection.result]


def test_a_decided_failure_outranks_an_undecided_predicate() -> None:
    """Otherwise killing a process on sight launders a banned terminal into 4."""
    selection = tp.select_successor_result(
        _successor(
            capture_off_on_equal="fail",
            execution_complete="fail",
        ),
        authority="exactly_once_measurement",
        phase="a",
    )
    assert selection.result == "capture_perturbs_policy"
    assert selection.order == 2


def test_undecided_predicates_alone_select_the_execution_terminal() -> None:
    selection = tp.select_successor_result(
        _successor(
            capture_off_on_equal="not_run",
            packets_valid="not_run",
            execution_complete="not_run",
        ),
        authority="exactly_once_measurement",
        phase="a",
    )
    assert selection.result == "unclassified_execution_failure"
    assert selection.order == 4


def test_an_undecided_predicate_cannot_coexist_with_a_complete_execution() -> None:
    with pytest.raises(tp.PartitionError, match="contradicts itself"):
        tp.select_successor_result(
            _successor(capture_off_on_equal="not_run"),
            authority="exactly_once_measurement",
            phase="a",
        )


def test_a_diagnostic_selects_no_terminal_whatever_its_predicates_say() -> None:
    """Correction 5: a green diagnostic qualifies nothing, and a red one is not a terminal."""
    for states in ({}, {"capture_off_on_equal": "fail"}, {"packets_valid": "error"}):
        selection = tp.select_successor_result(
            _successor(**states), authority="non_qualifying_diagnostic", phase="a"
        )
        assert selection.result == tp.DIAGNOSTIC_RESULT
        assert selection.terminal is None
        assert selection.order is None


def test_a_diagnostic_result_is_unavailable_to_a_measurement() -> None:
    assert tp.DIAGNOSTIC_RESULT not in tp.RESULT_TO_TERMINAL
    for authority in tp.AUTHORITIES:
        selection = tp.select_successor_result(
            _successor(), authority=authority, phase="a"
        )
        assert (selection.result == tp.DIAGNOSTIC_RESULT) == (
            authority == "non_qualifying_diagnostic"
        )


def test_a_named_execution_cause_needs_terminal_four_to_be_selected() -> None:
    named = tp.select_successor_result(
        _successor(execution_complete="fail"),
        authority="exactly_once_measurement",
        phase="a",
        execution_result="runner_timeout",
    )
    assert named.result == "runner_timeout"
    assert named.order == 4
    with pytest.raises(tp.PartitionError, match="terminal 4"):
        tp.select_successor_result(
            _successor(execution_complete="fail"),
            authority="exactly_once_measurement",
            phase="a",
            execution_result="packet_invalid",
        )
    with pytest.raises(tp.PartitionError, match="only when terminal 4"):
        tp.select_successor_result(
            _successor(packets_valid="fail"),
            authority="exactly_once_measurement",
            phase="a",
            execution_result="runner_nonzero",
        )
    with pytest.raises(tp.PartitionError, match="no execution result"):
        tp.select_successor_result(
            _successor(),
            authority="exactly_once_measurement",
            phase="a",
            execution_result="runner_nonzero",
        )
    with pytest.raises(tp.PartitionError, match="names no execution result"):
        tp.select_successor_result(
            _successor(),
            authority="non_qualifying_diagnostic",
            phase="a",
            execution_result="runner_nonzero",
        )


@pytest.mark.parametrize("key", [key for key, _ in tp.SUCCESSOR_PREDICATES])
def test_a_missing_or_unknown_successor_state_fails_closed(key: str) -> None:
    missing = _successor()
    del missing[key]
    with pytest.raises(tp.PartitionError, match="missing its state record"):
        tp.select_successor_result(
            missing, authority="exactly_once_measurement", phase="a"
        )
    with pytest.raises(tp.PartitionError, match="unknown state"):
        tp.select_successor_result(
            _successor(**{key: "skipped"}),
            authority="exactly_once_measurement",
            phase="a",
        )


def test_the_legacy_spelling_is_not_silently_accepted_as_a_successor_record() -> None:
    """A mixed record means one of the two vocabularies was misread."""
    mixed = {**_successor(), "bound_input_mutated": {"state": "pass", "reasons": []}}
    with pytest.raises(tp.PartitionError, match="outside the partition"):
        tp.select_successor_result(
            mixed, authority="exactly_once_measurement", phase="a"
        )


def test_the_successor_contract_covers_phase_a_only() -> None:
    with pytest.raises(tp.PartitionError, match="Phase-A four-run plan"):
        tp.select_successor_result(
            _successor(), authority="exactly_once_measurement", phase="b"
        )
    with pytest.raises(tp.PartitionError, match="unknown authority"):
        tp.select_successor_result(_successor(), authority="rehearsal", phase="a")


def test_the_payload_publishes_the_successor_rules_it_executes() -> None:
    """§ 20.8: the payload reader and the function caller must agree."""
    published = tp.as_payload()["successor_vocabulary"]
    assert published["authorities"] == list(tp.AUTHORITIES)
    assert published["ordered_predicates"] == [
        list(item) for item in tp.SUCCESSOR_PREDICATES
    ]
    assert published["predicate_states"] == list(tp.PREDICATE_STATES)
    assert published["undecided_states"] == list(tp.UNDECIDED_STATES)
    assert published["predicate_renames"] == tp.SUCCESSOR_TO_LEGACY_PREDICATE
    assert published["inverted_polarity_predicates"] == sorted(
        tp.INVERTED_POLARITY_PREDICATES
    )
    assert published["legacy_result_superseded_by"] == tp.LEGACY_RESULT_SUPERSEDED_BY
    assert published["diagnostic_result"] == tp.DIAGNOSTIC_RESULT
    # The two rules a payload-only implementer would otherwise have to invent.
    assert published["decided_failure_outranks_undecided"] is True
    assert published["undecided_requires_incomplete_execution"] is True
    assert published["diagnostic_selects_no_terminal"] is True
