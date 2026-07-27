"""The H2 Layer-M evidence contract: what an archive must support to be believed.

This contract was written before the controller and remains the interface
between an execution that spends an exactly-once authorization and the ruler
that reads it. Six properties are pinned here, each answering a specific way an
archive could look fine while supporting nothing:

  * **the observation cannot express what the partition cannot decide** — the
    emitter carries exactly `ORDERED_PREDICATES`, so a controller cannot record a
    predicate the ruler never reads, nor omit one it does;
  * **the recorded terminal is recomputed, never trusted** — the verifier
    re-selects from the archived observation, rebuilds the A7.6 comparison and
    re-verifies every capture-on packet;
  * **Phase-A terminal-1 inputs are cross-checked** — archived certificate,
    content bindings, launch probe and mutation observations must agree with the
    controller predicates instead of passing merely because they have checksums;
  * **the right to have spent `S_B` is recomputed too** — § C3.6's five
    conditions are rebuilt from the bound Phase-A root, both freeze records, the
    archived Layer-P certificate and the prior-attempt chain, and must match the
    record bit for bit. An archive that could attest its own admission would make
    the gate a formality;
  * **surviving evidence accumulates monotonically** — § C3.5.1's kill-switch
    only works if a later missing artifact cannot erase an earlier discovered
    inequality or invalid packet. Otherwise killing the process at the first sign
    of perturbation launders a forbidden terminal 2/3 into a re-attemptable 4;
  * **§ C3.9's trap stays shut** — the Layer-M files must classify as
    `plumbing_only` *and* hold no ruler of their own, so no semantic rule can
    move inside the frozen window without `identity_semantics` moving.

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
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402
from export_headline_bridge_decision_trace import (  # noqa: E402
    canonical_semantic_packet,
)
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

HEAD_A = "a" * 40
HEAD_B = "b" * 40
SEQUENCE_A = evidence.PHASE_SEQUENCES["a"][0]
ABSENT_ROOT = evidence.phase_a_root_name("9" * 40)

# One coordinate, shared by both phases: § C3.1(b) admits a Phase-A result only
# if all five axes and the probe are byte-equal across the two freeze records.
COORDINATE = dict(
    zip(
        verifier.ALL_COORDINATE_AXES,
        (f"{char}" * 64 for char in "12345"),
        strict=True,
    )
)
PROBE = "2d" * 32


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
    root: Path,
    sequence: str,
    *,
    perturbed: bool = False,
    comparison: bool = True,
    invalid_packets: tuple[str, ...] = (),
    omit_runs: tuple[str, ...] = (),
) -> None:
    captures = {
        run_id: _packet(run_uuid=f"{sequence}-{run_id}")
        for run_id in evidence.CAPTURE_ON_RUNS
    }
    if evidence.CAPTURE_OFF_RUN not in omit_runs:
        evidence.write_document(
            evidence.run_dir(root, sequence, evidence.CAPTURE_OFF_RUN),
            evidence.POLICY_INVENTORY_NAME,
            _policy_inventory(capture=None),
        )
    for index, run_id in enumerate(evidence.CAPTURE_ON_RUNS):
        if run_id in omit_runs:
            continue
        directory = evidence.run_dir(root, sequence, run_id)
        # A perturbation is a policy-visible difference between capture-off and
        # capture-on, which is what A7.6 compares.
        length = 12 if (perturbed and index == 0) else 11
        evidence.write_document(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            _policy_inventory(capture=captures[run_id], mot_length=length),
        )
        if run_id in invalid_packets:
            # An invalid packet is one the packet verifier rejects; the policy
            # inventory stays, because losing it is a different failure.
            (directory / evidence.POLICY_INVENTORY_NAME).unlink()
            evidence.write_document(
                directory, evidence.PACKET_NAME, {"capture_schema_version": "wrong"}
            )
            continue
        evidence.write_document(directory, evidence.PACKET_NAME, captures[run_id])
        evidence.write_document(
            directory,
            evidence.PACKET_VERIFICATION_NAME,
            {"report": verify_capture(captures[run_id]), "state": "pass"},
        )
    if not comparison:
        return
    inventories = {}
    for run_id in evidence.RUN_IDS:
        path = evidence.run_dir(root, sequence, run_id) / evidence.POLICY_INVENTORY_NAME
        if path.is_file():
            inventories[run_id] = json.loads(path.read_text(encoding="utf-8"))
    evidence.write_document(
        root / evidence.RUNS_DIR / sequence,
        evidence.COMPARISON_NAME,
        verifier._relations(inventories),
    )


def _certificate() -> dict[str, Any]:
    return {
        "schema": CERTIFICATE_SCHEMA,
        "behavior_probe": PROBE,
        "equivalence": "unproven",
        "published_coordinate": COORDINATE,
    }


def _freeze_record(**fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": evidence.FREEZE_SCHEMA,
        "coordinate": dict(COORDINATE),
        "measurement_surface_digest": "0" * 64,
        "prior_attempts": [],
        "probe": PROBE,
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


def _write_phase_a_launch_records(root: Path, *, head: str) -> dict[str, Any]:
    build_digest = "b" * 64
    runtime_manifest = {
        "schema": "h2_runtime_input_manifest_v1",
        "build_artifacts": {"digest": build_digest},
        "coordinate_digest": COORDINATE["runtime_inputs"],
        "full_digest": "f" * 64,
    }
    reference = {
        "schema": behavior.RESULT_SCHEMA,
        "build_witness": {"digest": build_digest},
        "digest": PROBE,
        "digests": [PROBE],
        "identical": True,
        "mode": "identity",
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    published = {
        "schema": verifier.IDENTITY_SCHEMA,
        "coordinate": COORDINATE,
        "equivalence": {"state": "unproven"},
        "probe": {"digest": PROBE},
        "publication_complete": True,
    }
    evidence.write_document(root, evidence.RUNTIME_INPUTS_NAME, runtime_manifest)
    evidence.write_document(root, evidence.REFERENCE_PROBE_NAME, reference)
    evidence.write_document(root, evidence.PUBLISHED_IDENTITY_NAME, published)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "behavior_probe": PROBE,
        "build_artifact_digest": build_digest,
        "build_witness": reference["build_witness"],
        "equivalence": "unproven",
        "probe_result_file_digest": evidence.sha256_file(
            root / evidence.REFERENCE_PROBE_NAME
        ),
        "published_coordinate": COORDINATE,
        "published_identity_file_digest": evidence.sha256_file(
            root / evidence.PUBLISHED_IDENTITY_NAME
        ),
        "published_probe": PROBE,
        "runtime_input_coordinate_digest": runtime_manifest["coordinate_digest"],
        "runtime_input_full_digest": runtime_manifest["full_digest"],
        "runtime_input_manifest_file_digest": evidence.sha256_file(
            root / evidence.RUNTIME_INPUTS_NAME
        ),
        "source_head": head,
    }
    evidence.write_document(root, evidence.CERTIFICATE_NAME, certificate)
    evidence.write_document(root, evidence.LAUNCH_PROBE_NAME, reference)
    evidence.write_document(
        root,
        evidence.MUTATION_NAME,
        {"schema": evidence.MUTATION_SCHEMA, "events": [], "mutated": False},
    )
    return certificate


def phase_a_root(
    parent: Path,
    *,
    perturbed: bool = False,
    head: str = HEAD_A,
    predicates: dict[str, bool] | None = None,
    runs: bool = True,
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / evidence.phase_a_root_name(head)
    root.mkdir()
    certificate = _write_phase_a_launch_records(root, head=head)
    evidence.write_document(
        root,
        evidence.FREEZE_NAME,
        _freeze_record(
            instrumentation_head=head,
            layer_p_certificate={
                "schema": CERTIFICATE_SCHEMA,
                "digest": evidence.digest(certificate),
            },
        ),
    )
    if runs:
        _write_sequence(root, SEQUENCE_A, perturbed=perturbed)
    values = predicates or _clean_predicates(capture_off_on_equal=not perturbed)
    observation = evidence.build_observation(values)
    evidence.write_document(root, evidence.OBSERVATION_NAME, observation)
    selection = partition.select_terminal(
        evidence.observation_predicates(observation), phase="a"
    )
    evidence.write_document(root, evidence.TERMINAL_NAME, _terminal_record(selection))
    evidence.write_document(
        root,
        evidence.CONTROLLER_NAME,
        {
            "schema": evidence.CONTROLLER_SCHEMA,
            "capture_phase": evidence.CAPTURE_PHASE["a"],
            "certificate_mismatch_reasons": [],
            "instrumentation_head": head,
            "ordered_runs": list(evidence.RUN_IDS),
            "result": selection.result,
            "sequence": SEQUENCE_A,
            "state": "terminal",
            "terminal": selection.terminal,
        },
    )
    return _finalize(root, phase="a", head=head, result=selection.result)


def _phase_a_binding(phase_a: Path | str) -> dict[str, Any]:
    if isinstance(phase_a, str):
        # A name that resolves to nothing: the recomputation must not be able to
        # confirm any of it.
        return {"evidence_root": phase_a}
    return {
        "evidence_root": phase_a.name,
        "manifest_digest": evidence.sha256_file(phase_a / evidence.MANIFEST_NAME),
        "checksum_inventory_digest": evidence.sha256_file(
            phase_a / evidence.CHECKSUMS_NAME
        ),
    }


def phase_b_root(
    parent: Path,
    *,
    phase_a: Path | str,
    head: str = HEAD_B,
    prior_attempts: tuple[str, ...] = (),
    surface: str = "0" * 64,
    defect_repair: dict[str, Any] | None = None,
    admission: dict[str, bool] | None = None,
    consume: bool = True,
    terminal: bool = True,
    certificate: dict[str, Any] | None = None,
    sequences: bool = False,
) -> Path:
    """A Phase-B attempt that died after launch: cheap, and the common case."""
    certificate_document = certificate or _certificate()
    fields: dict[str, Any] = {
        "layer_p_certificate": {
            "schema": CERTIFICATE_SCHEMA,
            "digest": evidence.digest(certificate_document),
        },
        "measurement_surface_digest": surface,
        "phase_a_evidence": _phase_a_binding(phase_a),
        "prior_attempts": list(prior_attempts),
    }
    if defect_repair is not None:
        fields["defect_repair"] = defect_repair
    freeze = _freeze_record(**fields)
    root = parent / evidence.phase_b_root_name(head, evidence.freeze_digest(freeze))
    root.mkdir(parents=True)
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    evidence.write_document(root, evidence.CERTIFICATE_NAME, certificate_document)
    evidence.write_document(
        root,
        evidence.ADMISSION_NAME,
        {
            "schema": evidence.ADMISSION_SCHEMA,
            **(admission or {key: True for key, _ in partition.ADMISSION_CONDITIONS}),
        },
    )
    if consume:
        evidence.write_document(
            root,
            evidence.AUTHORIZATION_NAME,
            {"schema": evidence.AUTHORIZATION_SCHEMA, "authorization": "S_B"},
        )
    if sequences:
        _write_sequence(root, SEQUENCE_A)
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


def refused_admission() -> dict[str, bool]:
    return {key: False for key, _ in partition.ADMISSION_CONDITIONS}


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


def test_launch_probe_predicate_is_recomputed_from_the_archive(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    launch = json.loads((root / evidence.LAUNCH_PROBE_NAME).read_text(encoding="utf-8"))
    launch["digest"] = "9" * 64
    evidence.write_document(root, evidence.LAUNCH_PROBE_NAME, launch)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="launch-probe predicate"):
        verifier.verify_evidence_root(root)


def test_certificate_predicate_is_recomputed_from_archived_bindings(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    certificate = json.loads(
        (root / evidence.CERTIFICATE_NAME).read_text(encoding="utf-8")
    )
    certificate["published_probe"] = "9" * 64
    evidence.write_document(root, evidence.CERTIFICATE_NAME, certificate)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="certificate match"):
        verifier.verify_evidence_root(root)


def test_mutation_predicate_must_match_the_monitor_record(tmp_path: Path) -> None:
    root = phase_a_root(tmp_path)
    evidence.write_document(
        root,
        evidence.MUTATION_NAME,
        {
            "schema": evidence.MUTATION_SCHEMA,
            "events": [{"classification": "bound_mutation", "mask": 2, "path": "x"}],
            "mutated": True,
        },
    )
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="mutation record"):
        verifier.verify_evidence_root(root)


def test_a_pass_must_meet_the_phase_completion_counts(tmp_path: Path) -> None:
    root = phase_a_root(tmp_path)
    directory = evidence.run_dir(root, SEQUENCE_A, evidence.CAPTURE_ON_RUNS[2])
    for name in (evidence.PACKET_NAME, evidence.PACKET_VERIFICATION_NAME):
        (directory / name).unlink()
    _finalize(root, phase="a", head=HEAD_A, result="measurement_pass")
    with pytest.raises(verifier.VerificationError, match="missing policy inventories"):
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


# -- § C3.1 root identity --------------------------------------------------- #


def test_phase_b_root_name_is_recomputed_from_the_freeze_record(
    tmp_path: Path,
) -> None:
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path))
    assert verifier.verify_evidence_root(root)["valid"] is True

    moved = root.parent / evidence.phase_b_root_name(HEAD_B, "c" * 64)
    root.rename(moved)
    with pytest.raises(verifier.VerificationError, match="does not match the recorded"):
        verifier.verify_evidence_root(moved)


def test_truncated_freeze_digest_is_not_a_root_name() -> None:
    with pytest.raises(evidence.EvidenceError, match="complete 64"):
        evidence.phase_b_root_name(HEAD_B, "c" * 16)


# -- § C3.6: the gate is recomputed, not read ------------------------------- #


def test_admission_is_recomputed_from_artifacts(tmp_path: Path) -> None:
    """A Phase-B archive may not attest its own eligibility to have spent S_B."""
    phase_a_root(tmp_path)
    root = phase_b_root(tmp_path, phase_a=ABSENT_ROOT)
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(root)


def test_admission_recomputation_reads_the_phase_a_result(tmp_path: Path) -> None:
    """A Phase-A root that selected a terminal is not a passed Phase A."""
    failed = phase_a_root(tmp_path, perturbed=True)
    root = phase_b_root(tmp_path, phase_a=failed)
    with pytest.raises(
        verifier.VerificationError, match="phase_a_did_not_pass|differs"
    ):
        verifier.verify_evidence_root(root)


def test_admission_recomputation_binds_the_phase_a_digests(tmp_path: Path) -> None:
    bound = phase_a_root(tmp_path)
    root = phase_b_root(tmp_path, phase_a=bound)
    freeze = json.loads((root / evidence.FREEZE_NAME).read_text(encoding="utf-8"))
    freeze["phase_a_evidence"]["manifest_digest"] = "f" * 64
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    moved = root.parent / evidence.phase_b_root_name(
        HEAD_B, evidence.freeze_digest(freeze)
    )
    root.rename(moved)
    _refinalize(moved)
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(moved)


def test_admission_recomputation_requires_the_coordinate_to_be_equal(
    tmp_path: Path,
) -> None:
    """§ C3.1(b): a moved axis makes the Phase-A evidence inadmissible."""
    bound = phase_a_root(tmp_path)
    phase_a_freeze = json.loads(
        (bound / evidence.FREEZE_NAME).read_text(encoding="utf-8")
    )
    phase_a_freeze["coordinate"]["runtime_inputs"] = "e" * 64
    evidence.write_document(bound, evidence.FREEZE_NAME, phase_a_freeze)
    _refinalize(bound)
    root = phase_b_root(tmp_path, phase_a=bound)
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(root)


def test_admission_recomputation_checks_the_archived_certificate(
    tmp_path: Path,
) -> None:
    bound = phase_a_root(tmp_path)
    root = phase_b_root(tmp_path, phase_a=bound)
    tampered = {**_certificate(), "equivalence": "claimed"}
    evidence.write_document(root, evidence.CERTIFICATE_NAME, tampered)
    _refinalize(root)
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(root)


def test_admission_recomputation_verifies_every_prior_attempt(tmp_path: Path) -> None:
    bound = phase_a_root(tmp_path)
    root = phase_b_root(
        tmp_path,
        phase_a=bound,
        head="2" * 40,
        prior_attempts=(evidence.phase_b_root_name("9" * 40, "9" * 64),),
    )
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(root)


# -- § C3.5.1 classes, and the monotonicity of surviving evidence ----------- #


def test_the_verify_classes_are_distinguished(tmp_path: Path) -> None:
    # One corpus per attempt: § C3.6(e) requires the consumed attempts of a
    # Phase-A result to form a single ordered chain, so three unrelated attempts
    # sharing one Phase-A root would be an ambiguous chain rather than a fixture.
    complete = phase_b_root(tmp_path / "c", phase_a=phase_a_root(tmp_path / "c"))
    assert corpus.classify(complete) == "complete"
    assert verifier.verify_evidence_root(complete)["valid"] is True

    unterminated = phase_b_root(
        tmp_path / "u", phase_a=phase_a_root(tmp_path / "u"), terminal=False
    )
    assert corpus.classify(unterminated) == "unterminated"
    report = verifier.verify_unterminated(unterminated)
    assert report["terminal"] is None and report["valid"] is True

    envelope = phase_b_root(tmp_path / "e", phase_a=phase_a_root(tmp_path / "e"))
    (envelope / evidence.MANIFEST_NAME).unlink()
    (envelope / evidence.OBSERVATION_NAME).unlink()
    evidence.write_checksum_inventory(envelope)
    assert corpus.classify(envelope) == "envelope"
    assert verifier.verify_envelope(envelope)["valid"] is True


def test_an_unterminated_attempt_may_not_record_a_terminal(tmp_path: Path) -> None:
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path))
    with pytest.raises(verifier.VerificationError, match="carries one"):
        verifier.verify_unterminated(root)


def test_kill_switch_rejects_a_survivor_that_contradicts_the_observation(
    tmp_path: Path,
) -> None:
    """Dying early may not launder a perturbation into a terminal-4 re-attempt."""
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path))
    _write_sequence(root, SEQUENCE_A, perturbed=True)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="claims equality"):
        verifier.verify_evidence_root(root)
    assert verifier.surviving_findings(root)["perturbation_observed"] is True


def test_a_missing_later_packet_cannot_erase_an_earlier_invalid_one(
    tmp_path: Path,
) -> None:
    """§ C3.5.1's ban must survive the artifacts that were never written."""
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path), terminal=False)
    _write_sequence(
        root,
        SEQUENCE_A,
        comparison=False,
        invalid_packets=(evidence.CAPTURE_ON_RUNS[0],),
        omit_runs=evidence.CAPTURE_ON_RUNS[1:],
    )
    evidence.write_checksum_inventory(root)
    findings = verifier.surviving_findings(root)
    assert findings["invalid_packet_observed"] is True
    assert verifier.verify_unterminated(root)["invalid_packet_observed"] is True


def test_a_missing_inventory_cannot_erase_a_surviving_inequality(
    tmp_path: Path,
) -> None:
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path), terminal=False)
    _write_sequence(
        root,
        SEQUENCE_A,
        perturbed=True,
        comparison=False,
        omit_runs=evidence.CAPTURE_ON_RUNS[1:],
    )
    evidence.write_checksum_inventory(root)
    assert verifier.surviving_findings(root)["perturbation_observed"] is True


def test_comparison_json_may_not_claim_equality_against_survivors(
    tmp_path: Path,
) -> None:
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path), terminal=False)
    _write_sequence(
        root, SEQUENCE_A, perturbed=True, comparison=False, omit_runs=("03_capture_on",)
    )
    evidence.write_document(
        root / evidence.RUNS_DIR / SEQUENCE_A,
        evidence.COMPARISON_NAME,
        {"first_unequal": None, "relations": [], "state": "equal"},
    )
    evidence.write_checksum_inventory(root)
    with pytest.raises(verifier.VerificationError, match="records equality"):
        verifier.verify_unterminated(root)


def test_a_root_may_not_carry_a_sequence_its_phase_does_not_run(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    _write_sequence(root, "MOT17-02-SDP")
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="the phase does not run"):
        verifier.verify_evidence_root(root)


# -- § C3.5.1 step 4: inadmissible is a verified class, not a skipped one ---- #


def test_an_inadmissible_root_verifies_in_its_own_class(tmp_path: Path) -> None:
    root = phase_b_root(
        tmp_path,
        phase_a=ABSENT_ROOT,
        admission=refused_admission(),
        consume=False,
        terminal=False,
    )
    assert corpus.classify(root) == partition.INADMISSIBLE_CLASS
    report = verifier.verify_inadmissible(root)
    assert report["valid"] is True and report["terminal"] is None
    with pytest.raises(verifier.VerificationError, match="inadmissible"):
        verifier.verify_evidence_root(root)


def test_an_inadmissible_root_is_still_checked_for_identity(tmp_path: Path) -> None:
    root = phase_b_root(
        tmp_path,
        phase_a=ABSENT_ROOT,
        admission=refused_admission(),
        consume=False,
        terminal=False,
    )
    moved = root.parent / evidence.phase_b_root_name(HEAD_B, "c" * 64)
    root.rename(moved)
    with pytest.raises(verifier.VerificationError, match="does not match the recorded"):
        verifier.verify_inadmissible(moved)


def test_consuming_s_b_after_a_refused_gate_is_rejected(tmp_path: Path) -> None:
    root = phase_b_root(
        tmp_path,
        phase_a=ABSENT_ROOT,
        admission=refused_admission(),
        consume=True,
        terminal=False,
    )
    with pytest.raises(verifier.VerificationError, match="refused admission gate"):
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
        corpus.classify(root)
    with pytest.raises(verifier.VerificationError, match="phase-B only"):
        verifier.verify_evidence_root(root)


def test_a_phase_b_root_must_record_a_gate_and_a_consumption(tmp_path: Path) -> None:
    root = phase_b_root(tmp_path, phase_a=phase_a_root(tmp_path))
    (root / evidence.AUTHORIZATION_NAME).unlink()
    evidence.write_checksum_inventory(root)
    with pytest.raises(verifier.VerificationError, match="authorization_consumed"):
        corpus.classify(root)


# -- § C3.5 re-attempt and prior_attempts ---------------------------------- #


def test_empty_corpus_passes(tmp_path: Path) -> None:
    assert corpus.archive_roots(tmp_path) == []
    assert corpus.check_corpus([]) == []


def test_prior_attempts_must_be_the_complete_ordered_chain(tmp_path: Path) -> None:
    bound = phase_a_root(tmp_path)
    first = phase_b_root(tmp_path, phase_a=bound, head="1" * 40)
    second = phase_b_root(
        tmp_path, phase_a=bound, head="2" * 40, prior_attempts=(first.name,)
    )
    assert len(corpus.check_corpus([first, second])) == 2
    assert verifier.verify_evidence_root(second)["valid"] is True


def test_an_omitted_predecessor_is_caught_by_the_verifier_alone(
    tmp_path: Path,
) -> None:
    """§ C3.6(e) asks whether the chain is *complete*, which a list cannot answer.

    The successor binds no predecessors while a consumed attempt for the same
    Phase-A result already exists. Walking only what `F_B` supplied would confirm
    an empty chain; the corpus scan is what makes the omission visible, and it
    must be visible to the per-root verifier, not only to the corpus checker —
    the omission is why this attempt had no right to consume `S_B`.
    """
    bound = phase_a_root(tmp_path)
    existing = phase_b_root(tmp_path, phase_a=bound, head="1" * 40)
    omitting = phase_b_root(tmp_path, phase_a=bound, head="2" * 40)
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(omitting)
    with pytest.raises(corpus.CorpusError, match="missing from a successor"):
        verifier.verify_prior_chain(
            omitting,
            json.loads((omitting / evidence.FREEZE_NAME).read_text(encoding="utf-8")),
            visiting=frozenset(),
        )
    del existing


def test_an_inadmissible_root_may_not_appear_in_prior_attempts(
    tmp_path: Path,
) -> None:
    """§ C3.5.1 step 4: a refused gate spent nothing and is not a predecessor."""
    bound = phase_a_root(tmp_path)
    refused = phase_b_root(
        tmp_path,
        phase_a=bound,
        head="1" * 40,
        admission=refused_admission(),
        consume=False,
        terminal=False,
    )
    assert corpus.classify(refused) == partition.INADMISSIBLE_CLASS
    successor = phase_b_root(
        tmp_path, phase_a=bound, head="2" * 40, prior_attempts=(refused.name,)
    )
    with pytest.raises(
        verifier.VerificationError, match="differs from the independent recomputation"
    ):
        verifier.verify_evidence_root(successor)
    with pytest.raises(corpus.CorpusError, match="complete ordered list"):
        verifier.verify_prior_chain(
            successor,
            json.loads((successor / evidence.FREEZE_NAME).read_text(encoding="utf-8")),
            visiting=frozenset(),
        )


def _banned_predecessor(parent: Path) -> tuple[Path, Path]:
    """A consumed attempt whose survivors already show a perturbation (§ C3.5.1).

    One corpus per case, because a successor is only admissible as *the* next
    link in the chain: two candidate successors in one directory would be
    rejected for the chain defect before the § C3.5 ban was ever reached.
    """
    bound = phase_a_root(parent)
    banned = phase_b_root(parent, phase_a=bound, head="1" * 40, terminal=False)
    _write_sequence(banned, SEQUENCE_A, perturbed=True)
    evidence.write_checksum_inventory(banned)
    assert verifier.verify_unterminated(banned)["perturbation_observed"] is True
    return bound, banned


def test_re_attempt_against_the_same_surface_is_banned(tmp_path: Path) -> None:
    bound, banned = _banned_predecessor(tmp_path / "same")
    successor = phase_b_root(
        tmp_path / "same", phase_a=bound, head="2" * 40, prior_attempts=(banned.name,)
    )
    with pytest.raises(corpus.CorpusError, match="same measurement surface"):
        corpus.check_corpus([banned, successor])


def test_a_moved_surface_alone_does_not_readmit_a_banned_measurement(
    tmp_path: Path,
) -> None:
    bound, banned = _banned_predecessor(tmp_path / "moved")
    successor = phase_b_root(
        tmp_path / "moved",
        phase_a=bound,
        head="2" * 40,
        prior_attempts=(banned.name,),
        surface="a" * 64,
    )
    with pytest.raises(corpus.CorpusError, match="never sufficient"):
        corpus.check_corpus([banned, successor])


def test_a_repair_outside_h0_section_6_vocabulary_is_not_a_repair(
    tmp_path: Path,
) -> None:
    bound, banned = _banned_predecessor(tmp_path / "vocab")
    successor = phase_b_root(
        tmp_path / "vocab",
        phase_a=bound,
        head="2" * 40,
        prior_attempts=(banned.name,),
        surface="b" * 64,
        defect_repair={"prior_attempt": banned.name, "defect_class": "recalibration"},
    )
    with pytest.raises(corpus.CorpusError, match="repair vocabulary"):
        corpus.check_corpus([banned, successor])


def test_a_named_defect_repair_on_a_moved_surface_is_admissible(
    tmp_path: Path,
) -> None:
    bound, banned = _banned_predecessor(tmp_path / "repair")
    successor = phase_b_root(
        tmp_path / "repair",
        phase_a=bound,
        head="2" * 40,
        prior_attempts=(banned.name,),
        surface="c" * 64,
        defect_repair={"prior_attempt": banned.name, "defect_class": "serialization"},
    )
    assert len(corpus.check_corpus([banned, successor])) == 2


def test_terminal_4_re_attempts_stay_expressible(tmp_path: Path) -> None:
    """The attempt-local terminals must not be swept up by the § C3.5 ban."""
    bound = phase_a_root(tmp_path)
    prior = phase_b_root(tmp_path, phase_a=bound, head="1" * 40)
    successor = phase_b_root(
        tmp_path, phase_a=bound, head="2" * 40, prior_attempts=(prior.name,)
    )
    assert len(corpus.check_corpus([prior, successor])) == 2
    assert verifier.verify_evidence_root(successor)["valid"] is True


# -- § C3.9's trap --------------------------------------------------------- #

LAYER_M_FILES = (
    "scripts/tools/h2_measurement_evidence.py",
    "scripts/tools/run_h2_measurement.py",
    "scripts/tools/run_h2_measurement_child.py",
    "scripts/tools/verify_h2_measurement.py",
    "scripts/tools/check_h2_measure_archives.py",
)


@pytest.mark.parametrize("relative", LAYER_M_FILES)
def test_the_new_layer_m_files_are_plumbing_only(relative: str) -> None:
    assert path_partition.classify(relative) == "plumbing_only"


@pytest.mark.parametrize("relative", LAYER_M_FILES)
def test_no_layer_m_file_restates_a_ruler_fact(relative: str) -> None:
    """Every semantic name must be imported, because these files move no axis.

    A `plumbing_only` file can be edited without `identity_semantics` moving, so
    a rule restated in one could change with nothing to catch it (§ C3.9). The
    scan covers all three files rather than the module that happens to be
    easiest to check: the verifier and the corpus checker are where the A7.6
    relation, the surface ban and the repair vocabulary would naturally be typed
    out.
    """
    source = (_REPO / relative).read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    forbidden = (
        # terminal and result names
        '"H2_CAPTURE_PERTURBS_POLICY"',
        '"H2_PACKET_INVALID"',
        '"H2_MEASUREMENT_EXECUTION_INVALID"',
        '"H2_FULL_COMMIT_CAPTURE_FAITHFUL"',
        '"measurement_pass"',
        # H0 § 6's repair vocabulary
        '"capacity_sizing"',
        '"implementation_bug"',
        # A7.6 members and shapes
        '"mot_output"',
        '"final_track_rows"',
        '"active_tid_slot_pairs"',
        '"relink_debug_raw"',
        '"proposal_projection"',
        '"winner_commit_projection"',
        '"overflow_vector"',
        '"h0_phase_a_policy_inventory_v1"',
        '"h2_layer_p_certificate_v2"',
        # the capture ABI's own overflow fields
        '"overflow_pair_records"',
        # phase completion counts
        '"required_capture_on_packets"',
    )
    restated = [name for name in forbidden if name in body]
    assert restated == [], f"{relative} restates ruler facts: {restated}"


def test_the_ruler_owns_the_semantic_constants() -> None:
    """The imports must resolve to the ruler's objects, not to equal copies."""
    assert verifier.A76_EQUALITY_MEMBERS is behavior.A76_EQUALITY_MEMBERS
    assert verifier.A76_PROJECTION_MEMBERS is behavior.A76_PROJECTION_MEMBERS
    assert corpus.SURFACE_TERMINALS is partition.SURFACE_BAN_TERMINALS
    assert corpus.REPAIR_VOCABULARY is partition.REPAIR_VOCABULARY
    assert evidence.POLICY_INVENTORY_SCHEMA is behavior.A76_POLICY_INVENTORY_SCHEMA
    # The A7.6 equality members are the probe's members in A7.6's own order, and
    # neither set may drift from the other.
    assert set(behavior.A76_EQUALITY_MEMBERS) == set(behavior.BEHAVIOR_MEMBERS)
    # Both consumption paths of the partition must publish the same narrowing.
    payload = partition.as_payload()
    assert set(payload["surface_ban_terminals"]) == set(partition.SURFACE_BAN_TERMINALS)
    assert set(payload["repair_vocabulary"]) == set(partition.REPAIR_VOCABULARY)
    assert payload["verify_classes"] == list(partition.VERIFY_CLASSES)


def test_the_evidence_module_holds_no_phase_completion_of_its_own() -> None:
    for phase, counts in partition.PHASE_COMPLETION.items():
        assert evidence.completion(phase) == counts
    emitted = evidence.build_observation(_clean_predicates())
    assert [key for key in emitted if key != "schema"] == [
        key for key, _ in partition.ORDERED_PREDICATES
    ]
