"""The H2 Layer-M evidence contract: what an archive must support to be believed.

The controller is not written yet, and that is exactly why these tests exist
first: the evidence root is the whole interface between an execution that spends
an exactly-once authorization and the ruler that reads it. Four properties are
pinned here, each answering a specific way H0's structure let a bad archive look
fine:

  * **the observation cannot express what the partition cannot decide** — the
    emitter carries exactly `ORDERED_PREDICATES`, so a controller cannot record a
    predicate the ruler never reads, nor omit one it does;
  * **the recorded terminal is recomputed, never trusted** — the verifier
    re-selects from the archived observation and rejects disagreement, and it
    rebuilds the A7.6 comparison and re-verifies every capture-on packet rather
    than reading the controller's verdict for them;
  * **§ C3.5.1's kill-switch holds** — surviving evidence that already shows a
    capture-off/on inequality or an invalid packet may never sit under a recorded
    predicate claiming otherwise, or terminating early would convert a forbidden
    terminal 2/3 into a re-attemptable terminal 4;
  * **§ C3.9's trap stays shut** — the three new files must classify as
    `plumbing_only` *and* hold no ruler of their own, so no phase or admission
    logic can move inside the frozen window without `identity_semantics` moving.

The packets here are H0's own `_packet` builder, imported rather than re-typed:
§ 6 says H2 introduces no comparison vocabulary of its own, and a test that built
its own idea of a valid packet would be asserting against a private copy of the
capture ABI.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_h2_measure_archives as corpus  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402
from export_headline_bridge_decision_trace import (  # noqa: E402
    canonical_semantic_packet,
)
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

HEAD_A = "a" * 40
HEAD_B = "b" * 40
SEQUENCE_A = evidence.PHASE_SEQUENCES["a"][0]


def _h0_packet_builder():
    """H0's own valid capture, loaded from the test that owns it."""
    path = _REPO / "tests/unit/tracking/test_headline_bridge_decision_trace.py"
    spec = importlib.util.spec_from_file_location("_h0_trace_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._packet


_packet = _h0_packet_builder()


# -- evidence-root construction -------------------------------------------- #


def _projections(capture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = canonical_semantic_packet(capture)
    streams = packet["streams"]
    candidates = [
        row for row in streams["candidate_records"] if int(row["proposal_emitted"]) == 1
    ]
    claims = streams["claim_records"]
    commits = streams["commit_records"]
    proposal = {"candidates": candidates, "claims": claims}
    winner = {
        "commits": commits,
        "winning_claims": [row for row in claims if int(row["claim_won"]) == 1],
    }
    return (
        {
            "count": len(candidates),
            "digest": evidence.digest(proposal),
            "records": proposal,
        },
        {"count": len(commits), "digest": evidence.digest(winner), "records": winner},
    )


def _policy_inventory(
    *, capture: dict[str, Any] | None, mot_length: int = 11
) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "schema": evidence.POLICY_INVENTORY_SCHEMA,
        "active_tid_slot_pairs": [{"frame": 1, "pairs": [[9, 3], [7, 4]]}],
        "final_track_rows": [
            {
                "binary32_bits": [1, 2, 3, 4, 5],
                "class": 1,
                "frame": 1,
                "row_index": 0,
                "track_id": 9,
            }
        ],
        "mot_output": {
            "length": mot_length,
            "sha256": hashlib.sha256(b"x" * mot_length).hexdigest(),
        },
        "overflow_vector": [0] * 9,
        "relink_debug_raw": list(range(13)),
        "proposal_projection": None,
        "winner_commit_projection": None,
    }
    if capture is not None:
        proposal, winner = _projections(capture)
        inventory["proposal_projection"] = proposal
        inventory["winner_commit_projection"] = winner
    return inventory


def _write_sequence(
    root: Path, sequence: str, *, perturbed: bool = False, packets: bool = True
) -> None:
    captures = {
        run_id: _packet(run_uuid=f"{sequence}-{run_id}")
        for run_id in evidence.CAPTURE_ON_RUNS
    }
    evidence.write_document(
        evidence.run_dir(root, sequence, evidence.CAPTURE_OFF_RUN),
        evidence.POLICY_INVENTORY_NAME,
        _policy_inventory(capture=None),
    )
    for index, run_id in enumerate(evidence.CAPTURE_ON_RUNS):
        directory = evidence.run_dir(root, sequence, run_id)
        # A perturbation is a policy-visible difference between capture-off and
        # capture-on, which is what A7.6 compares.
        length = 12 if (perturbed and index == 0) else 11
        evidence.write_document(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            _policy_inventory(capture=captures[run_id], mot_length=length),
        )
        if not packets:
            continue
        evidence.write_document(directory, evidence.PACKET_NAME, captures[run_id])
        evidence.write_document(
            directory,
            evidence.PACKET_VERIFICATION_NAME,
            {"report": verify_capture(captures[run_id]), "state": "pass"},
        )
    inventories = {
        run_id: json.loads(
            (
                evidence.run_dir(root, sequence, run_id)
                / evidence.POLICY_INVENTORY_NAME
            ).read_text(encoding="utf-8")
        )
        for run_id in evidence.RUN_IDS
    }
    evidence.write_document(
        root / evidence.RUNS_DIR / sequence,
        evidence.COMPARISON_NAME,
        verifier._reconstruct_comparison(inventories),
    )


def _freeze_record(**fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": evidence.FREEZE_SCHEMA,
        "measurement_surface_digest": "0" * 64,
        "prior_attempts": [],
    }
    record.update(fields)
    return record


def _finalize(root: Path, *, phase: str, head: str, result: str) -> Path:
    files = sorted(
        path.relative_to(root).as_posix() for path in evidence.evidence_files(root)
    )
    freeze = json.loads((root / evidence.FREEZE_NAME).read_text(encoding="utf-8"))
    evidence.write_document(
        root,
        evidence.MANIFEST_NAME,
        {
            "schema": evidence.MANIFEST_SCHEMA,
            "artifact_inventory": sorted({*files, evidence.MANIFEST_NAME}),
            "capture_phase": evidence.CAPTURE_PHASE[phase],
            "freeze_digest": evidence.freeze_digest(freeze),
            "instrumentation_head": head,
            "result": result,
        },
    )
    evidence.write_checksum_inventory(root)
    return root


def _refinalize(root: Path) -> Path:
    """Re-seal a root after a test edits it, so staleness is not the finding."""
    manifest = json.loads((root / evidence.MANIFEST_NAME).read_text(encoding="utf-8"))
    return _finalize(
        root,
        phase=evidence.PHASE_BY_CAPTURE_PHASE[manifest["capture_phase"]],
        head=manifest["instrumentation_head"],
        result=manifest["result"],
    )


def _terminal_record(selection: partition.Selection) -> dict[str, Any]:
    return {
        "schema": evidence.TERMINAL_SCHEMA,
        "order": selection.order,
        "phase": selection.phase,
        "result": selection.result,
        "terminal": selection.terminal,
    }


def _clean_predicates(**overrides: bool) -> dict[str, bool]:
    values = {key: True for key, _ in partition.ORDERED_PREDICATES}
    values["bound_input_mutated"] = False
    values.update(overrides)
    return values


def phase_a_root(
    tmp_path: Path,
    *,
    perturbed: bool = False,
    head: str = HEAD_A,
    predicates: dict[str, bool] | None = None,
    runs: bool = True,
) -> Path:
    root = tmp_path / evidence.phase_a_root_name(head)
    root.mkdir(parents=True)
    freeze = _freeze_record()
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    if runs:
        _write_sequence(root, SEQUENCE_A, perturbed=perturbed)
    values = predicates or _clean_predicates(capture_off_on_equal=not perturbed)
    observation = evidence.build_observation(values)
    evidence.write_document(root, evidence.OBSERVATION_NAME, observation)
    selection = partition.select_terminal(
        evidence.observation_predicates(observation), phase="a"
    )
    evidence.write_document(root, evidence.TERMINAL_NAME, _terminal_record(selection))
    return _finalize(root, phase="a", head=head, result=selection.result)


def phase_b_terminal_4_root(
    tmp_path: Path,
    *,
    head: str = HEAD_B,
    prior_attempts: tuple[str, ...] = (),
    surface: str = "0" * 64,
    defect_repair: dict[str, Any] | None = None,
    phase_a_root_name: str = "h2_measure_" + HEAD_A,
    admitted: bool = True,
    consume: bool = True,
    terminal: bool = True,
) -> Path:
    """A Phase-B attempt that died after launch: cheap, and the common case."""
    fields: dict[str, Any] = {
        "measurement_surface_digest": surface,
        "prior_attempts": list(prior_attempts),
        "phase_a_evidence": {"evidence_root": phase_a_root_name},
    }
    if defect_repair is not None:
        fields["defect_repair"] = defect_repair
    freeze = _freeze_record(**fields)
    root = tmp_path / evidence.phase_b_root_name(head, evidence.freeze_digest(freeze))
    root.mkdir(parents=True)
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    evidence.write_document(
        root,
        evidence.ADMISSION_NAME,
        {
            "schema": evidence.ADMISSION_SCHEMA,
            **{key: admitted for key, _ in partition.ADMISSION_CONDITIONS},
        },
    )
    if consume:
        evidence.write_document(
            root,
            evidence.AUTHORIZATION_NAME,
            {"schema": evidence.AUTHORIZATION_SCHEMA, "authorization": "S_B"},
        )
    if not terminal:
        # § C3.5.1: an unterminated attempt records no terminal because no
        # observation exists — the process never reached an exit path.
        evidence.write_checksum_inventory(root)
        return root
    observation = evidence.build_observation(
        _clean_predicates(execution_complete=False), execution_result="runner_nonzero"
    )
    evidence.write_document(root, evidence.OBSERVATION_NAME, observation)
    selection = partition.select_terminal(
        evidence.observation_predicates(observation),
        phase="b",
        admission=partition.evaluate_admission(
            {key: True for key, _ in partition.ADMISSION_CONDITIONS}, phase="b"
        ),
    )
    evidence.write_document(root, evidence.TERMINAL_NAME, _terminal_record(selection))
    return _finalize(root, phase="b", head=head, result=selection.result)


# -- the emitter ----------------------------------------------------------- #


def test_observation_carries_exactly_the_predicates_the_partition_reads() -> None:
    observation = evidence.build_observation(_clean_predicates())
    assert set(observation) == {"schema"} | {
        key for key, _ in partition.ORDERED_PREDICATES
    }

    with pytest.raises(evidence.EvidenceError, match="missing predicates"):
        evidence.build_observation({"bound_input_mutated": False})
    with pytest.raises(evidence.EvidenceError, match="does not define"):
        evidence.build_observation({**_clean_predicates(), "gpu_serial_equal": True})
    with pytest.raises(evidence.EvidenceError, match="is not a bool"):
        evidence.build_observation({**_clean_predicates(), "packets_valid": "yes"})


def test_execution_result_may_only_name_a_terminal_4_cause() -> None:
    named = evidence.build_observation(
        _clean_predicates(execution_complete=False), execution_result="runner_timeout"
    )
    assert named["execution_result"] == "runner_timeout"
    with pytest.raises(evidence.EvidenceError, match="terminal 4"):
        evidence.build_observation(
            _clean_predicates(execution_complete=False),
            execution_result="capture_perturbs_policy",
        )


# -- a complete Phase-A archive -------------------------------------------- #


def test_clean_phase_a_archive_verifies_and_selects_no_terminal(
    tmp_path: Path,
) -> None:
    report = verifier.verify_evidence_root(phase_a_root(tmp_path))
    assert report["valid"] is True
    assert report["result"] == "measurement_pass"
    assert report["terminal"] is None
    assert report["capture_phase"] == "phase_a"


def test_perturbed_archive_verifies_as_terminal_2(tmp_path: Path) -> None:
    report = verifier.verify_evidence_root(phase_a_root(tmp_path, perturbed=True))
    assert report["terminal"] == "H2_CAPTURE_PERTURBS_POLICY"


def test_recorded_terminal_must_match_the_independent_selection(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    recorded = json.loads((root / evidence.TERMINAL_NAME).read_text(encoding="utf-8"))
    recorded["terminal"] = "H2_FULL_COMMIT_CAPTURE_FAITHFUL"
    evidence.write_document(root, evidence.TERMINAL_NAME, recorded)
    _refinalize(root)
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent"
    ):
        verifier.verify_evidence_root(root)


def test_recorded_predicate_must_match_the_replay(tmp_path: Path) -> None:
    """A clean archive may not be reported as a perturbation, or the reverse."""
    root = phase_a_root(
        tmp_path, predicates=_clean_predicates(capture_off_on_equal=False)
    )
    with pytest.raises(verifier.VerificationError, match="independent replay"):
        verifier.verify_evidence_root(root)


def test_a_pass_must_meet_the_phase_completion_counts(tmp_path: Path) -> None:
    root = phase_a_root(tmp_path)
    directory = evidence.run_dir(root, SEQUENCE_A, evidence.CAPTURE_ON_RUNS[2])
    for name in (evidence.PACKET_NAME, evidence.PACKET_VERIFICATION_NAME):
        (directory / name).unlink()
    _finalize(root, phase="a", head=HEAD_A, result="measurement_pass")
    with pytest.raises(verifier.VerificationError, match="missing capture-on packets"):
        verifier.verify_evidence_root(root)


def test_checksum_inventory_is_total_in_both_directions(tmp_path: Path) -> None:
    root = phase_a_root(tmp_path)
    (root / "stray.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="absent from the inventory"):
        verifier.verify_evidence_root(root)
    (root / "stray.json").unlink()

    freeze = root / evidence.FREEZE_NAME
    freeze.write_bytes(freeze.read_bytes().replace(b'"0000', b'"1000'))
    with pytest.raises(verifier.VerificationError, match="differ from the inventory"):
        verifier.verify_evidence_root(root)


def test_records_must_be_canonical(tmp_path: Path) -> None:
    root = phase_a_root(tmp_path)
    (root / evidence.TERMINAL_NAME).write_text(
        json.dumps({"schema": evidence.TERMINAL_SCHEMA}, indent=2), encoding="utf-8"
    )
    evidence.write_checksum_inventory(root)  # inventory current; the record is not
    with pytest.raises(verifier.VerificationError, match="canonical form"):
        verifier.verify_evidence_root(root)


# -- § C3.1 root identity and § C3.6 admission ------------------------------ #


def test_phase_b_root_name_is_recomputed_from_the_freeze_record(
    tmp_path: Path,
) -> None:
    root = phase_b_terminal_4_root(tmp_path)
    assert verifier.verify_evidence_root(root)["valid"] is True

    moved = root.parent / evidence.phase_b_root_name(HEAD_B, "c" * 64)
    root.rename(moved)
    with pytest.raises(verifier.VerificationError, match="does not match the recorded"):
        verifier.verify_evidence_root(moved)


def test_truncated_freeze_digest_is_not_a_root_name() -> None:
    with pytest.raises(evidence.EvidenceError, match="complete 64"):
        evidence.phase_b_root_name(HEAD_B, "c" * 16)


def test_a_refused_admission_gate_is_inadmissible_not_a_consumed_attempt(
    tmp_path: Path,
) -> None:
    root = phase_b_terminal_4_root(tmp_path, admitted=False, consume=False)
    assert corpus.classify(root) == corpus.INADMISSIBLE
    with pytest.raises(verifier.VerificationError, match="inadmissible"):
        verifier.verify_evidence_root(root)


def test_consuming_s_b_after_a_refused_gate_is_rejected(tmp_path: Path) -> None:
    root = phase_b_terminal_4_root(tmp_path, admitted=False, consume=True)
    with pytest.raises(corpus.CorpusError, match="refused admission gate"):
        corpus.classify(root)


def test_phase_a_root_may_not_carry_an_admission_verdict(tmp_path: Path) -> None:
    root = phase_a_root(tmp_path)
    evidence.write_document(
        root,
        evidence.ADMISSION_NAME,
        {
            "schema": evidence.ADMISSION_SCHEMA,
            **{key: True for key, _ in partition.ADMISSION_CONDITIONS},
        },
    )
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="phase-B only"):
        verifier.verify_evidence_root(root)


# -- § C3.5.1 classes and the kill-switch ---------------------------------- #


def test_the_three_verify_classes_are_distinguished(tmp_path: Path) -> None:
    complete = phase_b_terminal_4_root(tmp_path / "c")
    assert corpus.classify(complete) == "complete"

    unterminated = phase_b_terminal_4_root(tmp_path / "u", terminal=False)
    assert corpus.classify(unterminated) == "unterminated"
    report = verifier.verify_unterminated(unterminated)
    assert report["terminal"] is None and report["valid"] is True

    # An envelope is a caught failure: the terminal classification survives, the
    # measurement's own artifacts need not.
    envelope = phase_b_terminal_4_root(tmp_path / "e")
    (envelope / evidence.MANIFEST_NAME).unlink()
    (envelope / evidence.OBSERVATION_NAME).unlink()
    evidence.write_checksum_inventory(envelope)
    assert corpus.classify(envelope) == "envelope"
    assert verifier.verify_envelope(envelope)["valid"] is True


def test_an_unterminated_attempt_may_not_record_a_terminal(tmp_path: Path) -> None:
    root = phase_b_terminal_4_root(tmp_path)
    with pytest.raises(verifier.VerificationError, match="carries one"):
        verifier.verify_unterminated(root)


def test_kill_switch_rejects_a_survivor_that_contradicts_the_observation(
    tmp_path: Path,
) -> None:
    """Dying early may not launder a perturbation into a terminal-4 re-attempt."""
    root = phase_b_terminal_4_root(tmp_path)
    _write_sequence(root, SEQUENCE_A, perturbed=True)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="claims equality"):
        verifier.verify_evidence_root(root)
    assert verifier.surviving_findings(root)["perturbation_observed"] is True


def test_a_root_may_not_carry_a_sequence_its_phase_does_not_run(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    _write_sequence(root, "MOT17-02-SDP")
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="the phase does not run"):
        verifier.verify_evidence_root(root)


# -- § C3.5 re-attempt and prior_attempts ---------------------------------- #


def test_empty_corpus_passes(tmp_path: Path) -> None:
    assert corpus.archive_roots(tmp_path) == []
    assert corpus.check_corpus([]) == []


def test_prior_attempts_must_be_the_complete_ordered_chain(tmp_path: Path) -> None:
    first = phase_b_terminal_4_root(tmp_path, head="1" * 40)
    second = phase_b_terminal_4_root(
        tmp_path, head="2" * 40, prior_attempts=(first.name,)
    )
    assert len(corpus.check_corpus([first, second])) == 2

    orphan = phase_b_terminal_4_root(tmp_path, head="3" * 40)
    with pytest.raises(corpus.CorpusError, match="prior_attempts is incomplete"):
        corpus.check_corpus([first, second, orphan])

    missing = phase_b_terminal_4_root(
        tmp_path,
        head="4" * 40,
        prior_attempts=(evidence.phase_b_root_name("9" * 40, "9" * 64),),
    )
    with pytest.raises(corpus.CorpusError, match="does not exist"):
        corpus.check_corpus([missing])


def test_re_attempt_against_the_same_surface_after_terminal_2_is_banned(
    tmp_path: Path,
) -> None:
    prior = phase_a_root(tmp_path / "prior_survivor", perturbed=True)
    # A Phase-B attempt whose survivors already show the perturbation carries the
    # ban even though it never recorded a terminal (§ C3.5.1).
    banned = phase_b_terminal_4_root(tmp_path, head="1" * 40, terminal=False)
    _write_sequence(banned, SEQUENCE_A, perturbed=True)
    evidence.write_checksum_inventory(banned)
    assert verifier.verify_unterminated(banned)["perturbation_observed"] is True
    del prior

    same_surface = phase_b_terminal_4_root(
        tmp_path, head="2" * 40, prior_attempts=(banned.name,)
    )
    with pytest.raises(corpus.CorpusError, match="same measurement surface"):
        corpus.check_corpus([banned, same_surface])

    moved_surface = phase_b_terminal_4_root(
        tmp_path, head="3" * 40, prior_attempts=(banned.name,), surface="a" * 64
    )
    with pytest.raises(corpus.CorpusError, match="never sufficient"):
        corpus.check_corpus([banned, moved_surface])

    unnamed_class = phase_b_terminal_4_root(
        tmp_path,
        head="4" * 40,
        prior_attempts=(banned.name,),
        surface="b" * 64,
        defect_repair={"prior_attempt": banned.name, "defect_class": "recalibration"},
    )
    with pytest.raises(corpus.CorpusError, match="repair vocabulary"):
        corpus.check_corpus([banned, unnamed_class])

    repaired = phase_b_terminal_4_root(
        tmp_path,
        head="5" * 40,
        prior_attempts=(banned.name,),
        surface="c" * 64,
        defect_repair={"prior_attempt": banned.name, "defect_class": "serialization"},
    )
    assert len(corpus.check_corpus([banned, repaired])) == 2


def test_terminal_1_and_4_re_attempts_stay_expressible(tmp_path: Path) -> None:
    """The attempt-local terminals must not be swept up by the § C3.5 ban."""
    prior = phase_b_terminal_4_root(tmp_path, head="1" * 40)
    successor = phase_b_terminal_4_root(
        tmp_path, head="2" * 40, prior_attempts=(prior.name,)
    )
    assert len(corpus.check_corpus([prior, successor])) == 2


# -- § C3.9's trap --------------------------------------------------------- #


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/tools/h2_measurement_evidence.py",
        "scripts/tools/verify_h2_measurement.py",
        "scripts/tools/check_h2_measure_archives.py",
    ),
)
def test_the_new_layer_m_files_are_plumbing_only(relative: str) -> None:
    assert path_partition.classify(relative) == "plumbing_only"


def test_the_evidence_module_holds_no_ruler_of_its_own() -> None:
    """Every phase and terminal fact must come from the partition, by import.

    C3.9 pins the hazard: a new `h2_*` file reads as plumbing, so ruler logic
    placed in one would move the ruler inside the frozen window with nothing to
    catch it.
    """
    for phase, counts in partition.PHASE_COMPLETION.items():
        assert evidence.completion(phase) == counts
    source = (_TOOLS / "h2_measurement_evidence.py").read_text(encoding="utf-8")
    for literal in ('required_capture_on_packets":', "H2_CAPTURE_PERTURBS_POLICY"):
        assert literal not in source
    emitted = evidence.build_observation(_clean_predicates())
    assert [key for key in emitted if key != "schema"] == [
        key for key, _ in partition.ORDERED_PREDICATES
    ]
