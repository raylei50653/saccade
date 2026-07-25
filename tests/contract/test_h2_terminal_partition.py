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
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

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
    selection = tp.select_terminal(_clean())
    assert selection.terminal is None
    assert selection.result == "measurement_pass"
    assert "Phase-B" in selection.describe()


def test_phase_b_completion_reaches_terminal_five() -> None:
    selection = tp.select_terminal(_clean(), phase_b_complete=True)
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
    assert tp.select_terminal(observation).terminal == expected


def test_order_decides_when_several_predicates_fail() -> None:
    """A mutated input outranks an invalid packet: if the inputs moved, the packet
    describes nothing, so reporting `PACKET_INVALID` would name the wrong cause."""
    observation = {**_clean(), "bound_input_mutated": True, "packets_valid": False}
    selection = tp.select_terminal(observation)
    assert selection.terminal == "H2_INPUT_MUTATED_DURING_MEASUREMENT"
    assert selection.order == 1


def test_perturbation_outranks_packet_invalidity() -> None:
    observation = {**_clean(), "capture_off_on_equal": False, "packets_valid": False}
    assert tp.select_terminal(observation).order == 2


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
        selection = tp.select_terminal(observation)
        assert selection.result == cause
        assert selection.terminal == "H2_MEASUREMENT_EXECUTION_INVALID"


def test_an_execution_cause_that_does_not_map_to_terminal_four_is_refused() -> None:
    observation = {
        **_clean(),
        "execution_complete": False,
        "execution_result": "capture_perturbs_policy",
    }
    with pytest.raises(tp.PartitionError, match="terminal 4"):
        tp.select_terminal(observation)


# --------------------------------------------------------------------------- #
# Fail-closed                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", [key for key, _ in tp.ORDERED_PREDICATES])
def test_a_missing_predicate_fails_closed(key: str) -> None:
    observation = _clean()
    del observation[key]
    with pytest.raises(tp.PartitionError, match="missing predicates"):
        tp.select_terminal(observation)


def test_a_non_boolean_predicate_fails_closed() -> None:
    """A truthy string would silently satisfy a safety predicate."""
    observation = {**_clean(), "packets_valid": "yes"}
    with pytest.raises(tp.PartitionError, match="not a bool"):
        tp.select_terminal(observation)


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
    selection = tp.select_terminal(witness_laden)
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
