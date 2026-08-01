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
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_h2_measure_archives as corpus  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_run_spec as run_spec  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402
from export_headline_bridge_decision_trace import (  # noqa: E402
    canonical_semantic_packet,
)
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402

HEAD_A = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
).strip()
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
_COMMIT_AXES = verifier._commit_content_axes(HEAD_A)
COORDINATE["implementation"] = _COMMIT_AXES["decision_relevant"]["digest"]
COORDINATE["identity_semantics"] = _COMMIT_AXES["identity_semantics"]["digest"]
PROBE = "2d" * 32

_SUCCESSOR_SCHEMA_PATHS = {
    "authoring_profile": (
        _REPO / "docs/research/contracts/h2_phase_a_authoring_profile_v1.schema.json"
    ),
    "run_spec": _REPO / "docs/research/contracts/h2_phase_a_run_spec_v1.json",
    "runtime_binding": _REPO / "docs/research/contracts/h2_runtime_binding_v1.json",
    "result": _REPO / "docs/research/contracts/h2_execution_result_v1.json",
    "verification": _REPO / "docs/research/contracts/h2_execution_verification_v1.json",
}


def _h0_packet_builder():
    """H0's own valid capture, loaded from the test that owns it."""
    path = _REPO / "tests/unit/tracking/test_headline_bridge_decision_trace.py"
    spec = importlib.util.spec_from_file_location("_h0_trace_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._packet


_packet = _h0_packet_builder()


# -- successor artifact contract ------------------------------------------ #


@pytest.mark.parametrize("name", sorted(_SUCCESSOR_SCHEMA_PATHS))
def test_successor_artifact_schemas_are_valid_closed_draft_2020_12(name: str) -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS[name].read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"]


def test_run_spec_projection_declares_the_complete_content_set() -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["run_spec"].read_text(encoding="utf-8"))
    members = schema["$defs"]["execution_semantics_projection"]["properties"]["members"]
    declared = {
        clause["contains"]["properties"]["path"]["const"] for clause in members["allOf"]
    }
    expected = {
        *evidence.PHASE_A_EXECUTED_SURFACE_PATHS,
        evidence.PHASE_A_CAPTURE_ABI_PATH,
        run_spec.AUTHORING_PROFILE_REL,
        run_spec.AUTHORING_PROFILE_SCHEMA_REL,
        run_spec.AUTHORING_DECISION_REL,
        run_spec.IMPORT_WITNESS_SCHEMA_REL,
        "scripts/eval/mot17_args.py",
        "docs/research/contracts/h2_phase_a_run_spec_v1.json",
        "scripts/tools/h2_run_spec.py",
    }
    assert declared == expected
    assert set(run_spec.EXECUTION_SEMANTICS_PATHS) == expected
    assert members["minItems"] == members["maxItems"] == len(expected)


def _built_closure() -> dict[str, Any]:
    return run_spec.execution_code_closure()


def test_the_execution_code_closure_binds_the_code_that_computes_a_result() -> None:
    """The named content set is tooling; this is what produced the rows.

    `post_merge.py` is the assertion that matters. It is not one of the modules a
    reviewer would list from memory, and `interpolate_tracklets` inside it
    decided the whole W4b finding — so a closure that does not contain it is a
    closure enumerated the way that already failed once.
    """
    closure = _built_closure()
    paths = {str(member["path"]) for member in closure["members"]}
    assert "src/saccade/perception/eval/post_merge.py" in paths
    assert "src/saccade/perception/eval/evaluator.py" in paths
    assert "src/saccade/perception/eval/pipeline.py" in paths
    assert "src/tracking/tracker_gpu.cu" in paths
    assert all(
        path.startswith(run_spec.DECLARED_EXECUTION_CODE_ROOTS) for path in paths
    )


def test_the_closure_carries_no_extension_filter() -> None:
    """A rule with exceptions is a rule an editor may work in."""
    tracked = {
        path
        for path in run_spec._paths_under_execution_code_roots()
        if path.endswith((".md", ".txt"))
    }
    assert tracked, "the tree no longer exercises this case"
    paths = {str(member["path"]) for member in _built_closure()["members"]}
    assert tracked <= paths


def test_the_schema_and_the_resolver_name_one_closure_identity() -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["run_spec"].read_text(encoding="utf-8"))
    closure = schema["$defs"]["execution_code_closure"]["properties"]
    assert closure["schema"]["const"] == run_spec.CODE_CLOSURE_SCHEMA
    assert closure["selector"]["const"] == run_spec.CODE_CLOSURE_SELECTOR
    assert closure["algorithm"]["const"] == run_spec.CONTENT_MEMBER_ALGORITHM
    projection = schema["$defs"]["execution_semantics_projection"]
    assert "execution_code_closure" in projection["required"]
    assert projection["properties"]["schema"]["const"] == run_spec.PROJECTION_SCHEMA
    assert (
        projection["properties"]["algorithm"]["const"] == run_spec.PROJECTION_ALGORITHM
    )


def test_the_v1_projection_identifiers_are_not_permitted_aliases() -> None:
    """Correction 7's rule: a changed digest domain may not keep the old name."""
    assert run_spec.PROJECTION_SCHEMA.endswith("_v2")
    assert run_spec.PROJECTION_ALGORITHM != run_spec.CONTENT_MEMBER_ALGORITHM
    document = run_spec.build_run_spec()
    projection = dict(document["execution_semantics_projection"])
    projection["algorithm"] = run_spec.CONTENT_MEMBER_ALGORITHM
    document["execution_semantics_projection"] = projection
    with pytest.raises(run_spec.RunSpecError, match="schema or algorithm mismatch"):
        run_spec.validate_run_spec(document, verify_projection=False)


def test_the_closure_moves_the_projection_digest() -> None:
    """Neither half of the projection can move without moving the digest."""
    document = run_spec.build_run_spec()
    projection = document["execution_semantics_projection"]
    closure = dict(projection["execution_code_closure"])
    members = [dict(member) for member in closure["members"]]
    members[0]["sha256"] = "0" * 64
    closure["members"] = members
    closure["digest"] = runtime_inputs.digest(members)
    moved = dict(projection)
    moved["execution_code_closure"] = closure
    assert (
        runtime_inputs.digest(
            {
                "execution_code_closure": closure["digest"],
                "members": projection["members"],
            }
        )
        != projection["digest"]
    )
    document["execution_semantics_projection"] = moved
    with pytest.raises(run_spec.RunSpecError, match="projection digest mismatch"):
        run_spec.validate_run_spec(document, verify_projection=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda closure: closure.update(digest="0" * 64), "closure digest mismatch"),
        (
            lambda closure: closure.update(roots=["docs/"]),
            "closure identity mismatch",
        ),
        (
            lambda closure: closure.update(selector="filesystem_walk_v1"),
            "closure identity mismatch",
        ),
        (lambda closure: closure.update(members=[]), "closure is empty"),
    ),
)
def test_a_closure_is_checked_from_its_own_bytes(mutate: Any, message: str) -> None:
    """Every one of these is decidable without reading the checkout."""
    document = run_spec.build_run_spec()
    projection = dict(document["execution_semantics_projection"])
    closure = dict(projection["execution_code_closure"])
    mutate(closure)
    projection["execution_code_closure"] = closure
    document["execution_semantics_projection"] = projection
    with pytest.raises(run_spec.RunSpecError, match=message):
        run_spec.validate_run_spec(document, verify_projection=False)


def test_a_closure_member_outside_a_declared_root_is_refused() -> None:
    document = run_spec.build_run_spec()
    projection = dict(document["execution_semantics_projection"])
    closure = dict(projection["execution_code_closure"])
    members = [dict(member) for member in closure["members"]]
    members.append({"length": 1, "path": "docs/smuggled.py", "sha256": "1" * 64})
    members.sort(key=lambda member: str(member["path"]))
    closure["members"] = members
    closure["digest"] = runtime_inputs.digest(members)
    projection["execution_code_closure"] = closure
    document["execution_semantics_projection"] = projection
    with pytest.raises(run_spec.RunSpecError, match="outside a root"):
        run_spec.validate_run_spec(document, verify_projection=False)


def test_a_zero_length_closure_member_is_admitted() -> None:
    """`src/saccade/__init__.py` is empty, imported, and not an error."""
    closure = _built_closure()
    empty = [member for member in closure["members"] if int(member["length"]) == 0]
    assert empty, "the tree no longer exercises this case"
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["run_spec"].read_text(encoding="utf-8"))
    assert schema["$defs"]["closure_member"]["properties"]["length"]["minimum"] == 0
    assert schema["$defs"]["content_member"]["properties"]["length"]["minimum"] == 1


def test_run_spec_schema_names_distinct_object_and_artifact_byte_domains() -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["run_spec"].read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert "canonicalization" not in properties
    assert properties["object_canonicalization"]["const"] == (
        "utf8_lexicographic_keys_compact_finite_no_trailing_lf_v1"
    )
    assert properties["artifact_serialization"]["const"] == (
        "utf8_lexicographic_keys_compact_finite_single_trailing_lf_v1"
    )


def test_frozen_authoring_profile_is_complete_and_owner_bound() -> None:
    profile_path = _REPO / run_spec.AUTHORING_PROFILE_REL
    profile_schema = json.loads(
        (_REPO / run_spec.AUTHORING_PROFILE_SCHEMA_REL).read_text(encoding="utf-8")
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    decision = json.loads(
        (_REPO / run_spec.AUTHORING_DECISION_REL).read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(profile_schema).validate(profile)
    assert profile["key_count"] == len(profile["resolved_namespace"]) == 454
    assert profile["resolved_namespace_digest"] == evidence.digest(
        profile["resolved_namespace"]
    )
    assert profile["resolved_namespace"]["preset"] is None
    assert (
        decision["profile_sha256"]
        == hashlib.sha256(profile_path.read_bytes()).hexdigest()
    )
    assert decision["explicit_adjudications"] == {
        key: profile["resolved_namespace"][key]
        for key in ("detector", "max_frames", "preset", "warmup_frames")
    }


def test_authoring_profile_tamper_fails_before_run_spec_issuance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = run_spec._load_pretty_document

    def tampered_load(
        relative: str, *, require_canonical: bool = True
    ) -> tuple[dict[str, Any], bytes]:
        payload, raw = real_load(relative, require_canonical=require_canonical)
        if relative != run_spec.AUTHORING_PROFILE_REL:
            return payload, raw
        payload = json.loads(json.dumps(payload))
        payload["resolved_namespace"]["acc_alpha"] = 0.16
        payload["resolved_namespace_digest"] = evidence.digest(
            payload["resolved_namespace"]
        )
        raw = (
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        return payload, raw

    monkeypatch.setattr(run_spec, "_load_pretty_document", tampered_load)
    with pytest.raises(run_spec.RunSpecError, match="does not bind"):
        run_spec.build_run_spec()


def test_verification_schema_forbids_producer_and_host_derived_completion() -> None:
    schema = json.loads(
        _SUCCESSOR_SCHEMA_PATHS["verification"].read_text(encoding="utf-8")
    )
    properties = schema["properties"]
    assert properties["verification_process"]["const"] == (
        "independent_command_separate_process"
    )
    assert properties["producer_invoked"]["const"] is False
    assert properties["verification_host_inputs_used"]["const"] is False


def test_diagnostic_result_cannot_carry_measurement_authority() -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["result"].read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    instance = {
        "schema": "h2_execution_result_v1",
        "execution_id": "diagnostic-1",
        "authority": "non_qualifying_diagnostic",
        "authorization_binding_digest": "1" * 64,
        "resolved_run_spec_digest": "2" * 64,
        "execution_semantics_projection_digest": "3" * 64,
        "run_plan": {
            "sequence": "MOT17-04-SDP",
            "run_ids": [
                "00_capture_off",
                "01_capture_on",
                "02_capture_on",
                "03_capture_on",
            ],
        },
        "predicate_results": {
            name: {"state": "pass", "reasons": []}
            for name in (
                "bound_input_unchanged",
                "capture_off_on_equal",
                "execution_complete",
                "packets_valid",
                "runtime_projection_matches_resolved_run_spec",
            )
        },
        "ordered_runs": [
            {
                "run_id": run_id,
                "state": "completed",
                "artifact_digest": "4" * 64,
            }
            for run_id in evidence.RUN_IDS
        ],
        "result": "measurement_pass",
        "terminal": None,
    }
    messages = [error.message for error in validator.iter_errors(instance)]
    assert any("is not of type 'null'" in message for message in messages)
    assert any("'diagnostic_complete' was expected" in message for message in messages)


# -- cross-verdict constraints (Review Correction 9) ----------------------- #
#
# The schema and the ruler are two consumption paths over one partition, so the
# expectations below are *derived* from the schema and compared to the ruler.
# Retyping either side would let them drift and still pass.


def _result_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["result"].read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def _successor_states(**states: str) -> dict[str, dict[str, Any]]:
    record = {
        key: {"state": "pass", "reasons": []}
        for key, _ in partition.SUCCESSOR_PREDICATES
    }
    for key, state in states.items():
        record[key] = {"state": state, "reasons": [] if state == "pass" else ["why"]}
    return record


def _result_instance(
    selection: partition.Selection, states: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    measurement = selection.result != partition.DIAGNOSTIC_RESULT
    return {
        "schema": "h2_execution_result_v1",
        "execution_id": "successor-1",
        "authority": (
            "exactly_once_measurement" if measurement else "non_qualifying_diagnostic"
        ),
        "authorization_binding_digest": "1" * 64 if measurement else None,
        "resolved_run_spec_digest": "2" * 64,
        "execution_semantics_projection_digest": "3" * 64,
        "run_plan": {"sequence": "MOT17-04-SDP", "run_ids": list(evidence.RUN_IDS)},
        "predicate_results": states,
        "ordered_runs": [
            {"run_id": run_id, "state": "completed", "artifact_digest": "4" * 64}
            for run_id in evidence.RUN_IDS
        ],
        "result": selection.result,
        "terminal": selection.terminal,
    }


def test_the_result_schema_and_the_ruler_share_one_vocabulary() -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["result"].read_text(encoding="utf-8"))
    assert set(schema["properties"]["predicate_results"]["required"]) == {
        key for key, _ in partition.SUCCESSOR_PREDICATES
    }
    assert set(
        schema["$defs"]["predicate_result"]["properties"]["state"]["enum"]
    ) == set(partition.PREDICATE_STATES)
    # The successor enum is the ruler's mapping with the retired token dropped and
    # the diagnostic token added — not an independently maintained list.
    assert set(schema["properties"]["result"]["enum"]) == (
        set(partition.RESULT_TO_TERMINAL)
        - set(partition.LEGACY_RESULT_SUPERSEDED_BY)
        - set(partition.RETIRED_SUCCESSOR_RESULTS)
        | {partition.DIAGNOSTIC_RESULT}
    )
    assert set(schema["properties"]["terminal"]["oneOf"][0]["enum"]) == {
        name for name in partition.RESULT_TO_TERMINAL.values() if name
    }


def test_the_schema_pins_every_result_to_the_terminal_the_ruler_selects() -> None:
    schema = json.loads(_SUCCESSOR_SCHEMA_PATHS["result"].read_text(encoding="utf-8"))
    declared: dict[str, str] = {}
    for clause in schema["allOf"]:
        condition = clause.get("if", {}).get("properties", {}).get("result")
        terminal = clause.get("then", {}).get("properties", {}).get("terminal", {})
        if not condition or "const" not in terminal:
            continue
        for result in condition.get("enum", [condition.get("const")]):
            declared[result] = terminal["const"]
    assert declared == {
        result: name
        for result, name in partition.RESULT_TO_TERMINAL.items()
        if name
        and result not in partition.LEGACY_RESULT_SUPERSEDED_BY
        and result not in partition.RETIRED_SUCCESSOR_RESULTS
    }


@pytest.mark.parametrize(
    "states",
    [
        {},
        {"bound_input_unchanged": "fail"},
        {"runtime_projection_matches_resolved_run_spec": "fail"},
        {"capture_off_on_equal": "fail"},
        {"packets_valid": "fail"},
        {"execution_complete": "fail"},
        {"capture_off_on_equal": "fail", "execution_complete": "fail"},
        {"capture_off_on_equal": "not_run", "execution_complete": "not_run"},
    ],
)
def test_the_rulers_verdict_is_the_only_one_the_schema_accepts(
    states: dict[str, str],
) -> None:
    """§ 20.8, mechanised across the two paths rather than asserted on each."""
    validator = _result_validator()
    record = _successor_states(**states)
    for authority in partition.AUTHORITIES:
        selection = partition.select_successor_result(
            record, authority=authority, phase="a"
        )
        instance = _result_instance(selection, record)
        assert not list(validator.iter_errors(instance)), (
            f"the ruler's own verdict for {states} under {authority} is rejected by "
            "the artifact schema — the two consumption paths disagree"
        )
        if selection.terminal is None:
            continue
        for other in {
            name for name in partition.RESULT_TO_TERMINAL.values() if name
        } - {selection.terminal}:
            assert list(validator.iter_errors({**instance, "terminal": other})), (
                "the schema accepted a terminal the ruler did not select"
            )
        # Substituting the *result* while keeping the terminal is the case a
        # terminal-only sweep misses: two results share terminal 1, and an
        # undecided predicate must not let the wrong one stand.
        for other_result, other_terminal in partition.RESULT_TO_TERMINAL.items():
            if other_result in (selection.result, "measurement_pass"):
                continue
            if other_result in partition.LEGACY_RESULT_SUPERSEDED_BY:
                continue
            if other_result in partition.RETIRED_SUCCESSOR_RESULTS:
                continue
            if other_terminal != selection.terminal:
                continue
            if other_terminal == partition.EXECUTION_INVALID_TERMINAL:
                # Terminal 4's named cause is not decidable from result.json: its
                # tokens differ in which retained stage failed, which lives in
                # runtime_binding.json. That separation is asserted by
                # test_terminal_four_causes_are_separated_by_stage_evidence.
                continue
            assert list(validator.iter_errors({**instance, "result": other_result})), (
                f"the schema accepted result {other_result!r} for an observation the "
                f"ruler resolves to {selection.result!r} — same terminal, wrong cause"
            )


def test_the_schema_refuses_a_spent_authorization_with_no_terminal() -> None:
    """The successor shape of § C3.5.1's unformable state."""
    validator = _result_validator()
    record = _successor_states(execution_complete="fail")
    selection = partition.select_successor_result(
        record, authority="exactly_once_measurement", phase="a"
    )
    instance = _result_instance(selection, record)
    assert not list(validator.iter_errors(instance))
    assert list(validator.iter_errors({**instance, "terminal": None}))


def test_the_schema_refuses_a_clean_observation_carrying_a_terminal() -> None:
    validator = _result_validator()
    record = _successor_states()
    passing = _result_instance(
        partition.select_successor_result(
            record, authority="exactly_once_measurement", phase="a"
        ),
        record,
    )
    assert list(
        validator.iter_errors(
            {
                **passing,
                "result": "input_mutated",
                "terminal": "H2_INPUT_MUTATED_DURING_MEASUREMENT",
            }
        )
    )


def test_the_schema_refuses_washing_an_earlier_finding_into_terminal_four() -> None:
    """Surviving evidence accumulates: terminal 2 may not be relabelled as 4."""
    validator = _result_validator()
    record = _successor_states(capture_off_on_equal="fail", execution_complete="fail")
    selection = partition.select_successor_result(
        record, authority="exactly_once_measurement", phase="a"
    )
    assert selection.order == 2
    assert list(
        validator.iter_errors(
            {
                **_result_instance(selection, record),
                "result": "runner_nonzero",
                "terminal": partition.EXECUTION_INVALID_TERMINAL,
            }
        )
    )


def test_the_schema_refuses_an_undecided_predicate_under_a_complete_execution() -> None:
    validator = _result_validator()
    record = _successor_states(capture_off_on_equal="not_run")
    instance = _result_instance(
        partition.Selection(
            "runner_nonzero", partition.EXECUTION_INVALID_TERMINAL, 4, None, True, "a"
        ),
        record,
    )
    assert list(validator.iter_errors(instance))
    with pytest.raises(partition.PartitionError, match="contradicts itself"):
        partition.select_successor_result(
            record, authority="exactly_once_measurement", phase="a"
        )


def test_the_verification_schema_binds_valid_to_its_own_checks() -> None:
    schema = json.loads(
        _SUCCESSOR_SCHEMA_PATHS["verification"].read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    checks = list(schema["properties"]["checks"]["required"])
    assert set(schema["$defs"]["all_checks_true"]["required"]) == set(checks)

    def instance(**overrides: Any) -> dict[str, Any]:
        base = {
            "schema": "h2_execution_verification_v1",
            "execution_id": "successor-1",
            "resolved_run_spec_digest": "1" * 64,
            "execution_semantics_projection_digest": "2" * 64,
            "artifact_digests": {
                "run_spec.json": "3" * 64,
                "runtime_binding.json": "4" * 64,
                "result.json": "5" * 64,
            },
            "checks": {name: True for name in checks},
            "verification_process": "independent_command_separate_process",
            "producer_invoked": False,
            "verification_host_inputs_used": False,
            "valid": True,
            "reasons": [],
        }
        base.update(overrides)
        return base

    assert not list(validator.iter_errors(instance()))
    assert list(validator.iter_errors(instance(valid=False, reasons=["late"])))
    assert list(validator.iter_errors(instance(reasons=["unexplained"])))
    for name in checks:
        failed = {**{key: True for key in checks}, name: False}
        assert list(validator.iter_errors(instance(checks=failed))), (
            f"check {name} could be false while the verdict stayed valid"
        )
        assert not list(
            validator.iter_errors(
                instance(checks=failed, valid=False, reasons=[f"{name} failed"])
            )
        )
        assert list(validator.iter_errors(instance(checks=failed, valid=False)))


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


def _base_policy_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        **{member: inventory[member] for member in behavior.A76_EQUALITY_MEMBERS},
        "schema": evidence.BASE_POLICY_INVENTORY_SCHEMA,
    }


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
        directory = evidence.run_dir(root, sequence, evidence.CAPTURE_OFF_RUN)
        inventory = _policy_inventory(capture=None)
        evidence.write_document(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            inventory,
        )
        evidence.write_document(
            directory,
            evidence.BASE_POLICY_INVENTORY_NAME,
            _base_policy_inventory(inventory),
        )
        (directory / f"{sequence}.txt").write_bytes(b"x" * 11)
    for index, run_id in enumerate(evidence.CAPTURE_ON_RUNS):
        if run_id in omit_runs:
            continue
        directory = evidence.run_dir(root, sequence, run_id)
        # A perturbation is a policy-visible difference between capture-off and
        # capture-on, which is what A7.6 compares.
        length = 12 if (perturbed and index == 0) else 11
        inventory = _policy_inventory(capture=captures[run_id], mot_length=length)
        evidence.write_document(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            inventory,
        )
        evidence.write_document(
            directory,
            evidence.BASE_POLICY_INVENTORY_NAME,
            _base_policy_inventory(inventory),
        )
        (directory / f"{sequence}.txt").write_bytes(b"x" * length)
        if run_id in invalid_packets:
            # An invalid packet keeps the durable base inventory; the full
            # packet-derived inventory is deliberately absent.
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
    bases = {}
    inventories = {}
    for run_id in evidence.RUN_IDS:
        directory = evidence.run_dir(root, sequence, run_id)
        base_path = directory / evidence.BASE_POLICY_INVENTORY_NAME
        inventory_path = directory / evidence.POLICY_INVENTORY_NAME
        if base_path.is_file():
            bases[run_id] = json.loads(base_path.read_text(encoding="utf-8"))
        if inventory_path.is_file():
            inventories[run_id] = json.loads(inventory_path.read_text(encoding="utf-8"))
    evidence.write_document(
        root / evidence.RUNS_DIR / sequence,
        evidence.COMPARISON_NAME,
        verifier._relations(bases, inventories),
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
    build_dir = "/archive/h2/build"
    runtime_manifest = {
        "schema": "h2_runtime_input_manifest_v1",
        "build_artifacts": {"build_dir": build_dir, "digest": build_digest},
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
        "build_dir": build_dir,
        "build_witness": reference["build_witness"],
        "changed_path_verdict": {"admissible": True, "base": head},
        "decision_relevant_digest": COORDINATE["implementation"],
        "equivalence": "unproven",
        "fixture": behavior.IDENTITY_SEQUENCE,
        "identity_semantics_digest": COORDINATE["identity_semantics"],
        "mode": "identity",
        "plumbing_set_digest": _COMMIT_AXES["plumbing_only"]["digest"],
        "probe_schema": behavior.RESULT_SCHEMA,
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
        "selected_base": head,
        "source_head": head,
        "source_tree": subprocess.check_output(
            ["git", "rev-parse", f"{head}^{{tree}}"], cwd=_REPO, text=True
        ).strip(),
    }
    evidence.write_document(root, evidence.CERTIFICATE_NAME, certificate)
    evidence.write_document(root, evidence.LAUNCH_PROBE_NAME, reference)
    evidence.write_document(
        root,
        evidence.CHECKOUT_WITNESS_NAME,
        {
            "schema": evidence.CHECKOUT_WITNESS_SCHEMA,
            "axes": _COMMIT_AXES,
            "build_dir": build_dir,
            "repository_root": _REPO.as_posix(),
            "source_head": head,
            "source_tree": certificate["source_tree"],
        },
    )
    evidence.write_document(
        root,
        evidence.MUTATION_NAME,
        {"schema": evidence.MUTATION_SCHEMA, "events": [], "mutated": False},
    )
    evidence.write_document(
        root,
        evidence.STOP_BOUNDARY_NAME,
        {
            "schema": evidence.STOP_BOUNDARY_SCHEMA,
            "checkout_clean": True,
            "checkout_hygiene_reasons": [],
            "completed_utc": "2026-07-27T00:00:00Z",
            "final_drain_completed": True,
            "linearization": "clean_final_drain",
            "monitor_closed": True,
            "monitor_started": True,
            "revalidation_completed_while_monitored": True,
            "revalidation_reasons": [],
            "source_head": head,
            "source_tree": certificate["source_tree"],
        },
    )
    return certificate


def _phase_a_freeze(
    root: Path, certificate: dict[str, Any], head: str
) -> dict[str, Any]:
    runtime_manifest = evidence.load_document(root, evidence.RUNTIME_INPUTS_NAME)
    return {
        "schema": evidence.FREEZE_SCHEMA,
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "instrumentation_head": head,
        "selected_base": head,
        "coordinate": dict(COORDINATE),
        "probe": PROBE,
        "equivalence": "unproven",
        "layer_p_certificate": {
            "schema": CERTIFICATE_SCHEMA,
            "digest": evidence.digest(certificate),
        },
        "reference_probe": {
            "schema": behavior.RESULT_SCHEMA,
            "file_digest": evidence.sha256_file(root / evidence.REFERENCE_PROBE_NAME),
        },
        "runtime_inputs": {
            "schema": runtime_manifest["schema"],
            "file_digest": evidence.sha256_file(root / evidence.RUNTIME_INPUTS_NAME),
            "coordinate_digest": runtime_manifest["coordinate_digest"],
            "full_digest": runtime_manifest["full_digest"],
            "build_artifact_digest": runtime_manifest["build_artifacts"]["digest"],
        },
        "published_identity": {
            "schema": verifier.IDENTITY_SCHEMA,
            "file_digest": evidence.sha256_file(
                root / evidence.PUBLISHED_IDENTITY_NAME
            ),
        },
        "capture_abi": {
            "path": evidence.PHASE_A_CAPTURE_ABI_PATH,
            "sha256": hashlib.sha256(
                subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{head}:{evidence.PHASE_A_CAPTURE_ABI_PATH}",
                    ],
                    cwd=_REPO,
                )
            ).hexdigest(),
        },
        "executed_surfaces": {
            path: hashlib.sha256(
                subprocess.check_output(["git", "show", f"{head}:{path}"], cwd=_REPO)
            ).hexdigest()
            for path in evidence.PHASE_A_EXECUTED_SURFACE_PATHS
        },
        "run_plan": {
            "sequence": SEQUENCE_A,
            "run_ids": list(evidence.RUN_IDS),
        },
    }


def _write_phase_a_authorization(root: Path, freeze: dict[str, Any]) -> None:
    authorization_id = evidence.digest({"root": root.name})
    invocation_id = evidence.digest({"invocation": authorization_id})
    execution_domain = evidence.authorization_execution_domain(
        (root.parent / ".authorization-ledger").resolve()
    )
    execution_domain_digest = evidence.digest(execution_domain)
    grant = {
        "schema": evidence.AUTHORIZATION_GRANT_SCHEMA,
        "authorization_id": authorization_id,
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "controller_digest": freeze["executed_surfaces"][
            "scripts/tools/run_h2_measurement.py"
        ],
        "execution_domain": execution_domain_digest,
        "freeze_digest": evidence.freeze_digest(freeze),
        "instrumentation_head": freeze["instrumentation_head"],
        "invocation_id": invocation_id,
        "issued_by": "research_owner",
    }
    evidence.write_document(
        root,
        evidence.AUTHORIZATION_DOMAIN_NAME,
        execution_domain,
    )
    evidence.write_document(root, evidence.AUTHORIZATION_GRANT_NAME, grant)
    evidence.write_document(
        root,
        evidence.AUTHORIZATION_NAME,
        {
            "schema": evidence.AUTHORIZATION_SCHEMA,
            "authorization_digest": evidence.digest(grant),
            "authorization_id": authorization_id,
            "capture_phase": evidence.CAPTURE_PHASE["a"],
            "consumed_utc": "2026-07-28T00:00:00Z",
            "controller_digest": freeze["executed_surfaces"][
                "scripts/tools/run_h2_measurement.py"
            ],
            "execution_domain": execution_domain_digest,
            "freeze_digest": evidence.freeze_digest(freeze),
            "instrumentation_head": freeze["instrumentation_head"],
            "invocation_id": invocation_id,
            "state": "consumed",
        },
    )


def _rebind_phase_a_authorization(root: Path, freeze: dict[str, Any]) -> None:
    grant = evidence.load_document(root, evidence.AUTHORIZATION_GRANT_NAME)
    grant["freeze_digest"] = evidence.freeze_digest(freeze)
    grant["controller_digest"] = freeze["executed_surfaces"][
        "scripts/tools/run_h2_measurement.py"
    ]
    evidence.write_document(root, evidence.AUTHORIZATION_GRANT_NAME, grant)
    receipt = evidence.load_document(root, evidence.AUTHORIZATION_NAME)
    receipt["authorization_digest"] = evidence.digest(grant)
    receipt["freeze_digest"] = evidence.freeze_digest(freeze)
    receipt["controller_digest"] = freeze["executed_surfaces"][
        "scripts/tools/run_h2_measurement.py"
    ]
    evidence.write_document(root, evidence.AUTHORIZATION_NAME, receipt)


def _write_phase_a_lifecycle(root: Path) -> None:
    receipt = evidence.load_document(root, evidence.AUTHORIZATION_NAME)
    names = [
        "authorization_consumed",
        "archive_created",
        "monitor_active",
        "launch_revalidation",
    ]
    for _run_id in evidence.RUN_IDS:
        names.extend(("child_launch", "child_completed"))
    names.extend(
        (
            "monitored_final_revalidation",
            "final_monitor_drain",
            "stop_boundary_recorded",
        )
    )
    rows = []
    run_index = 0
    for ordinal, name in enumerate(names, start=1):
        row: dict[str, Any] = {
            "schema": "h2_controller_lifecycle_event_v1",
            "event": name,
            "ordinal": ordinal,
        }
        if name == "authorization_consumed":
            row["authorization_id"] = receipt["authorization_id"]
            row["invocation_id"] = receipt["invocation_id"]
        if name in {"child_launch", "child_completed"}:
            row["run_id"] = evidence.RUN_IDS[run_index // 2]
            run_index += 1
        rows.append(evidence.canonical_json_bytes(row))
    (root / evidence.LIFECYCLE_NAME).write_bytes(b"\n".join(rows) + b"\n")


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
    freeze = _phase_a_freeze(root, certificate, head)
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    _write_phase_a_authorization(root, freeze)
    _write_phase_a_lifecycle(root)
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
            "checkout_hygiene_reasons": [],
            "instrumentation_head": head,
            "ordered_runs": list(evidence.RUN_IDS),
            "result": selection.result,
            "sequence": SEQUENCE_A,
            "state": "terminal",
            "terminal": selection.terminal,
            "predicate_ownership": {
                "execution_checkout_hygiene": {"passed": True, "reasons": []},
                "layer_p_certificate_matches_freeze": {
                    "passed": True,
                    "reasons": [],
                },
                "monitored_runtime_inputs": {
                    "mutated": False,
                    "revalidation_reasons": [],
                },
            },
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


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_member",
        "extra_member",
        "selected_base_symbolic",
        "selected_base_uppercase",
        "runtime_input_stale",
        "executed_surface_stale",
        "capture_abi_stale",
        "run_plan_mismatch",
        "reference_probe_stale",
        "published_identity_stale",
        "coordinate_mismatch",
    ),
)
def test_phase_a_freeze_reconstruction_rejects_every_binding_drift(
    tmp_path: Path, mutation: str
) -> None:
    root = phase_a_root(tmp_path)
    freeze = evidence.load_document(root, evidence.FREEZE_NAME)
    if mutation == "missing_member":
        del freeze["run_plan"]
    elif mutation == "extra_member":
        freeze["private_binding"] = "not allowed"
    elif mutation == "selected_base_symbolic":
        freeze["selected_base"] = "main"
    elif mutation == "selected_base_uppercase":
        freeze["selected_base"] = HEAD_A.upper()
    elif mutation == "runtime_input_stale":
        freeze["runtime_inputs"]["file_digest"] = "9" * 64
    elif mutation == "executed_surface_stale":
        first = evidence.PHASE_A_EXECUTED_SURFACE_PATHS[0]
        freeze["executed_surfaces"][first] = "9" * 64
    elif mutation == "capture_abi_stale":
        freeze["capture_abi"]["sha256"] = "9" * 64
    elif mutation == "run_plan_mismatch":
        freeze["run_plan"]["run_ids"] = list(reversed(evidence.RUN_IDS))
    elif mutation == "reference_probe_stale":
        freeze["reference_probe"]["file_digest"] = "9" * 64
    elif mutation == "published_identity_stale":
        freeze["published_identity"]["file_digest"] = "9" * 64
    elif mutation == "coordinate_mismatch":
        freeze["coordinate"]["environment"] = "9" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    _rebind_phase_a_authorization(root, freeze)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="Phase-A freeze"):
        verifier.verify_evidence_root(root)


@pytest.mark.parametrize(
    "mutation",
    (
        "grant_absent",
        "domain_absent",
        "domain_binding",
        "grant_digest",
        "grant_execution_domain",
        "absent",
        "authorization_digest",
        "receipt_execution_domain",
        "authorization_id",
        "invocation_id",
        "head",
        "freeze",
        "controller",
        "state",
        "extra",
    ),
)
def test_phase_a_authorization_consumption_is_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    root = phase_a_root(tmp_path)
    path = root / evidence.AUTHORIZATION_NAME
    grant_path = root / evidence.AUTHORIZATION_GRANT_NAME
    if mutation == "grant_absent":
        grant_path.unlink()
    elif mutation == "domain_absent":
        (root / evidence.AUTHORIZATION_DOMAIN_NAME).unlink()
    elif mutation == "domain_binding":
        domain = evidence.load_document(root, evidence.AUTHORIZATION_DOMAIN_NAME)
        domain["ledger_root"] = (tmp_path / "another-ledger").resolve().as_posix()
        evidence.write_document(root, evidence.AUTHORIZATION_DOMAIN_NAME, domain)
    elif mutation == "grant_digest":
        grant = evidence.load_document(root, evidence.AUTHORIZATION_GRANT_NAME)
        grant["issued_by"] = "not_the_owner"
        evidence.write_document(root, evidence.AUTHORIZATION_GRANT_NAME, grant)
    elif mutation == "grant_execution_domain":
        grant = evidence.load_document(root, evidence.AUTHORIZATION_GRANT_NAME)
        grant["execution_domain"] = "9" * 64
        evidence.write_document(root, evidence.AUTHORIZATION_GRANT_NAME, grant)
    elif mutation == "absent":
        path.unlink()
    else:
        receipt = evidence.load_document(root, evidence.AUTHORIZATION_NAME)
        if mutation == "authorization_digest":
            receipt["authorization_digest"] = "not-a-digest"
        elif mutation == "receipt_execution_domain":
            receipt["execution_domain"] = "9" * 64
        elif mutation == "authorization_id":
            receipt["authorization_id"] = "not-an-id"
        elif mutation == "invocation_id":
            receipt["invocation_id"] = "short"
        elif mutation == "head":
            receipt["instrumentation_head"] = "9" * 40
        elif mutation == "freeze":
            receipt["freeze_digest"] = "9" * 64
        elif mutation == "controller":
            receipt["controller_digest"] = "9" * 64
        elif mutation == "state":
            receipt["state"] = "issued"
        elif mutation == "extra":
            receipt["grant"] = True
        else:  # pragma: no cover
            raise AssertionError(mutation)
        evidence.write_document(root, evidence.AUTHORIZATION_NAME, receipt)
    _refinalize(root)
    with pytest.raises(
        verifier.VerificationError,
        match=(
            "authorization grant/consumption|authorization_consumed|"
            "authorization_grant.json|authorization_execution_domain.json"
        ),
    ):
        verifier.verify_evidence_root(root)


@pytest.mark.parametrize(
    "mutation",
    (
        "completion_without_launch",
        "duplicate_launch",
        "duplicate_completion",
        "wrong_completion_id",
    ),
)
def test_partial_lifecycle_rejects_unpaired_or_duplicate_child_events(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = phase_a_root(tmp_path)
    path = root / evidence.LIFECYCLE_NAME
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    first_launch = next(
        index for index, row in enumerate(rows) if row["event"] == "child_launch"
    )
    first_completion = next(
        index for index, row in enumerate(rows) if row["event"] == "child_completed"
    )
    if mutation == "completion_without_launch":
        del rows[first_launch]
    elif mutation == "duplicate_launch":
        rows.insert(first_launch + 1, dict(rows[first_launch]))
    elif mutation == "duplicate_completion":
        rows.insert(first_completion + 1, dict(rows[first_completion]))
    elif mutation == "wrong_completion_id":
        rows[first_completion]["run_id"] = evidence.RUN_IDS[1]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    for ordinal, row in enumerate(rows, start=1):
        row["ordinal"] = ordinal
    path.write_bytes(
        b"".join(evidence.canonical_json_bytes(row) + b"\n" for row in rows)
    )
    _refinalize(root)
    with pytest.raises(
        verifier.VerificationError,
        match="completion has no unmatched launch|launch lifecycle is duplicated",
    ):
        verifier.verify_evidence_root(root)


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


@pytest.mark.parametrize(
    "condition",
    (
        "binding_digest",
        "source_head",
        "source_tree",
        "selected_base",
        "changed_path_verdict",
        "changed_path_verdict_base",
        "equivalence",
        "implementation_digest",
        "identity_semantics_digest",
        "plumbing_digest",
        "published_coordinate",
        "behavior_probe",
        "published_probe",
        "runtime_coordinate",
        "runtime_full",
        "build_artifact_relation",
        "runtime_manifest_file_digest",
        "probe_result_file_digest",
        "published_identity_file_digest",
        "reference_support",
        "probe_declaration",
        "reference_build_witness",
        "published_identity_support",
        "build_directory",
    ),
)
def test_every_certificate_condition_is_independently_recomputed(
    tmp_path: Path, condition: str
) -> None:
    root = phase_a_root(tmp_path)
    certificate = evidence.load_document(root, evidence.CERTIFICATE_NAME)
    freeze = evidence.load_document(root, evidence.FREEZE_NAME)
    runtime = evidence.load_document(root, evidence.RUNTIME_INPUTS_NAME)
    reference = evidence.load_document(root, evidence.REFERENCE_PROBE_NAME)
    published = evidence.load_document(root, evidence.PUBLISHED_IDENTITY_NAME)

    if condition == "binding_digest":
        freeze["layer_p_certificate"]["digest"] = "9" * 64
    elif condition == "source_head":
        certificate["source_head"] = "9" * 40
    elif condition == "source_tree":
        certificate["source_tree"] = "9" * 40
    elif condition == "selected_base":
        certificate["selected_base"] = ""
    elif condition == "changed_path_verdict":
        certificate["changed_path_verdict"]["admissible"] = False
    elif condition == "changed_path_verdict_base":
        certificate["changed_path_verdict"]["base"] = "9" * 40
    elif condition == "equivalence":
        certificate["equivalence"] = "claimed"
    elif condition == "implementation_digest":
        certificate["decision_relevant_digest"] = "9" * 64
    elif condition == "identity_semantics_digest":
        certificate["identity_semantics_digest"] = "9" * 64
    elif condition == "plumbing_digest":
        certificate["plumbing_set_digest"] = "9" * 64
    elif condition == "published_coordinate":
        certificate["published_coordinate"] = {
            **certificate["published_coordinate"],
            "environment": "9" * 64,
        }
    elif condition == "behavior_probe":
        certificate["behavior_probe"] = "9" * 64
    elif condition == "published_probe":
        certificate["published_probe"] = "9" * 64
    elif condition == "runtime_coordinate":
        certificate["runtime_input_coordinate_digest"] = "9" * 64
    elif condition == "runtime_full":
        certificate["runtime_input_full_digest"] = "9" * 64
    elif condition == "build_artifact_relation":
        runtime["build_artifacts"]["digest"] = "9" * 64
        evidence.write_document(root, evidence.RUNTIME_INPUTS_NAME, runtime)
        certificate["runtime_input_manifest_file_digest"] = evidence.sha256_file(
            root / evidence.RUNTIME_INPUTS_NAME
        )
    elif condition == "runtime_manifest_file_digest":
        certificate["runtime_input_manifest_file_digest"] = "9" * 64
    elif condition == "probe_result_file_digest":
        certificate["probe_result_file_digest"] = "9" * 64
    elif condition == "published_identity_file_digest":
        certificate["published_identity_file_digest"] = "9" * 64
    elif condition == "reference_support":
        reference["identical"] = False
        evidence.write_document(root, evidence.REFERENCE_PROBE_NAME, reference)
        certificate["probe_result_file_digest"] = evidence.sha256_file(
            root / evidence.REFERENCE_PROBE_NAME
        )
    elif condition == "probe_declaration":
        certificate["fixture"] = "MOT17-10-FRCNN"
    elif condition == "reference_build_witness":
        reference["build_witness"] = {"digest": "9" * 64}
        evidence.write_document(root, evidence.REFERENCE_PROBE_NAME, reference)
        certificate["probe_result_file_digest"] = evidence.sha256_file(
            root / evidence.REFERENCE_PROBE_NAME
        )
        certificate["build_witness"] = reference["build_witness"]
    elif condition == "published_identity_support":
        published["publication_complete"] = False
        evidence.write_document(root, evidence.PUBLISHED_IDENTITY_NAME, published)
        certificate["published_identity_file_digest"] = evidence.sha256_file(
            root / evidence.PUBLISHED_IDENTITY_NAME
        )
    elif condition == "build_directory":
        runtime["build_artifacts"]["build_dir"] = "/archive/h2/other-build"
        evidence.write_document(root, evidence.RUNTIME_INPUTS_NAME, runtime)
        certificate["runtime_input_manifest_file_digest"] = evidence.sha256_file(
            root / evidence.RUNTIME_INPUTS_NAME
        )
    else:  # pragma: no cover - parameter table is exhaustive
        raise AssertionError(condition)

    evidence.write_document(root, evidence.CERTIFICATE_NAME, certificate)
    if condition != "binding_digest":
        freeze["layer_p_certificate"]["digest"] = evidence.digest(certificate)
    evidence.write_document(root, evidence.FREEZE_NAME, freeze)
    _rebind_phase_a_authorization(root, freeze)
    _refinalize(root)
    with pytest.raises(
        verifier.VerificationError,
        match="certificate match|Phase-A freeze",
    ):
        verifier.verify_evidence_root(root)


def test_recorded_certificate_false_must_equal_independent_recomputation(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    observation = evidence.load_document(root, evidence.OBSERVATION_NAME)
    observation["layer_p_certificate_matches_freeze"] = False
    evidence.write_document(root, evidence.OBSERVATION_NAME, observation)
    selection = partition.select_terminal(
        evidence.observation_predicates(observation), phase="a"
    )
    evidence.write_document(root, evidence.TERMINAL_NAME, _terminal_record(selection))
    controller_record = evidence.load_document(root, evidence.CONTROLLER_NAME)
    controller_record["certificate_mismatch_reasons"] = ["fabricated mismatch"]
    controller_record["result"] = selection.result
    controller_record["terminal"] = selection.terminal
    evidence.write_document(root, evidence.CONTROLLER_NAME, controller_record)
    _finalize(root, phase="a", head=HEAD_A, result=selection.result)
    with pytest.raises(verifier.VerificationError, match="certificate match"):
        verifier.verify_evidence_root(root)


def test_checkout_witness_axes_are_rebuilt_from_the_bound_git_tree(
    tmp_path: Path,
) -> None:
    root = phase_a_root(tmp_path)
    witness = evidence.load_document(root, evidence.CHECKOUT_WITNESS_NAME)
    witness["axes"]["plumbing_only"]["digest"] = "9" * 64
    evidence.write_document(root, evidence.CHECKOUT_WITNESS_NAME, witness)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="checkout identity witness"):
        verifier.verify_evidence_root(root)


@pytest.mark.parametrize("tamper", ("checkout_dirty", "extra_member"))
def test_stop_boundary_is_strict_and_requires_a_clean_checkout(
    tmp_path: Path, tamper: str
) -> None:
    root = phase_a_root(tmp_path)
    stop = evidence.load_document(root, evidence.STOP_BOUNDARY_NAME)
    if tamper == "checkout_dirty":
        stop["checkout_clean"] = False
    else:
        stop["unrecognized"] = True
    evidence.write_document(root, evidence.STOP_BOUNDARY_NAME, stop)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="stop boundary"):
        verifier.verify_evidence_root(root)


@pytest.mark.parametrize(
    "tamper",
    (
        "checkout_dirty",
        "source_head",
        "source_tree",
        "linearization",
        "revalidation_incomplete",
    ),
)
def test_terminal_4_stop_boundary_tamper_is_rejected(
    tmp_path: Path, tamper: str
) -> None:
    root = phase_a_root(tmp_path)
    observation = evidence.build_observation(
        _clean_predicates(execution_complete=False),
        execution_result="runner_nonzero",
    )
    evidence.write_document(root, evidence.OBSERVATION_NAME, observation)
    selection = partition.select_terminal(
        evidence.observation_predicates(observation), phase="a"
    )
    assert selection.terminal == partition.EXECUTION_INVALID_TERMINAL
    evidence.write_document(root, evidence.TERMINAL_NAME, _terminal_record(selection))
    controller_record = evidence.load_document(root, evidence.CONTROLLER_NAME)
    controller_record["result"] = selection.result
    controller_record["terminal"] = selection.terminal
    evidence.write_document(root, evidence.CONTROLLER_NAME, controller_record)
    _finalize(root, phase="a", head=HEAD_A, result=selection.result)
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal

    stop = evidence.load_document(root, evidence.STOP_BOUNDARY_NAME)
    if tamper == "checkout_dirty":
        stop["checkout_clean"] = False
        stop["linearization"] = None
    elif tamper == "source_head":
        stop["source_head"] = "9" * 40
    elif tamper == "source_tree":
        stop["source_tree"] = "9" * 40
    elif tamper == "linearization":
        stop["linearization"] = None
    else:
        stop["revalidation_completed_while_monitored"] = False
        stop["linearization"] = None
    evidence.write_document(root, evidence.STOP_BOUNDARY_NAME, stop)
    _refinalize(root)
    with pytest.raises(verifier.VerificationError, match="stop|revalidation|source"):
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
    freeze.write_bytes(freeze.read_bytes().replace(b'"probe":"2d', b'"probe":"3d'))
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
    "scripts/tools/h2_measurement_freeze.py",
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


# -- stage-aware runtime binding (Review Correction 9, revision 1) ---------- #
#
# The first pass re-admitted `build_failed` and `extension_load_failed` as result
# tokens while `h2_runtime_binding_v1` still required two complete build
# artifacts, a successful load, a computed probe and a zero-change monitor
# unconditionally. Both tokens — and terminal 1 — were therefore unformable: a
# contract cannot narrow anything by making its own truthful negatives
# unrecordable.


def _binding_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(
        _SUCCESSOR_SCHEMA_PATHS["runtime_binding"].read_text(encoding="utf-8")
    )
    return jsonschema.Draft202012Validator(schema)


_DROP = object()


def _binding(**overrides: Any) -> dict[str, Any]:
    """A complete successful binding, with its shape read off the schema."""
    schema = json.loads(
        _SUCCESSOR_SCHEMA_PATHS["runtime_binding"].read_text(encoding="utf-8")
    )
    surfaces = [
        clause["contains"]["properties"]["path"]["const"]
        for clause in schema["properties"]["executed_surfaces"]["allOf"]
    ]
    roles = schema["$defs"]["build_artifact"]["properties"]["role"]["enum"]
    document: dict[str, Any] = {
        "schema": "h2_runtime_binding_v1",
        "execution_id": "successor-1",
        "resolved_run_spec_digest": "1" * 64,
        "execution_semantics_projection_digest": "2" * 64,
        "failed_stage": None,
        "build_artifacts": [
            {
                "role": role,
                "path": f"build/h2_layer_p/{role}.so",
                "sha256": "a" * 64,
                "length": 16,
            }
            for role in roles
        ],
        "extension_load": {
            "loaded_path": "/build/h2_layer_p/tracking.so",
            "length": 16,
            "sha256": "b" * 64,
        },
        "diagnostics": {
            "behavior_probe": {
                "schema": "h2_behavior_probe_result_v1",
                "role": "recorded_diagnostic_observation_selects_nothing",
                "state": "computed",
                "digest": "c" * 64,
            }
        },
        "runtime_projection": {
            "observations": [
                {
                    "environment": {
                        "SACCADE_DETECT_BARRIER": "event",
                        "SACCADE_DOUBLE_BUFFER": "1",
                        "SACCADE_GPU_DECODE": "1",
                        "SACCADE_MAIN_NMS_GRAPHED": "1",
                    },
                    "run_id": "00_capture_off",
                }
            ],
            "resolved_run_spec_digest": "9" * 64,
        },
        "input_monitor": {
            "started_before_binding": True,
            "changed_count": 0,
            "final_drain_clean": True,
        },
        "runtime_inputs": {
            # A `build` failure hashed no build artifacts, so the manifest it can
            # record is the coordinate form. The full form asserts artifacts.
            "manifest_schema": (
                "h2_runtime_input_coordinate_v1"
                if overrides.get("failed_stage") == "build"
                else "h2_runtime_input_manifest_v1"
            ),
            "manifest_digest": "e" * 64,
            "members": [
                {
                    "path": "data/MOT17/train/MOT17-04-SDP/img1/000001.jpg",
                    "role": "measurement_sequence",
                    "sha256": "f" * 64,
                    "length": 1,
                }
            ],
        },
        "executed_surfaces": [
            {"path": path, "sha256": "0" * 64, "length": 1} for path in surfaces
        ],
        "capture_abi": {
            "path": "scripts/tools/h0_bridge_decision_trace_schema_v2.json",
            "sha256": "1" * 64,
            "length": 1,
        },
        "source_audit": {"head": "a" * 40, "tree": "b" * 40},
    }
    document.update(overrides)
    return {key: value for key, value in document.items() if value is not _DROP}


def test_a_successful_binding_is_exactly_as_strict_as_before() -> None:
    validator = _binding_validator()
    assert not list(validator.iter_errors(_binding()))
    partial = _binding()
    partial["build_artifacts"] = partial["build_artifacts"][:1]
    assert list(validator.iter_errors(partial))
    assert list(validator.iter_errors(_binding(extension_load=_DROP)))
    # `diagnostics` is optional at every stage: Correction 10 forbids a probe's
    # absence from deciding anything, and unformability is deciding something.
    assert not list(validator.iter_errors(_binding(diagnostics=_DROP)))
    assert list(validator.iter_errors(_binding(runtime_projection=_DROP)))
    assert list(validator.iter_errors(_binding(failed_stage=_DROP)))
    assert list(validator.iter_errors(_binding(failed_stage="preflight")))


@pytest.mark.parametrize(
    ("failed_stage", "keep_artifacts", "keep_load"),
    [
        ("build", False, False),
        ("build_binding", True, False),
        ("extension_load", True, False),
        ("identity_run", True, True),
    ],
)
def test_every_stage_failure_can_form_a_binding(
    failed_stage: str, keep_artifacts: bool, keep_load: bool
) -> None:
    validator = _binding_validator()
    document = _binding(
        failed_stage=failed_stage,
        diagnostics=_DROP,
        # No run reached a launch boundary, so there is nothing to record and the
        # schema forbids recording it anyway.
        runtime_projection=_DROP,
        **({} if keep_load else {"extension_load": _DROP}),
    )
    if not keep_artifacts:
        document["build_artifacts"] = []
    assert not list(validator.iter_errors(document)), (
        f"a genuine {failed_stage} failure cannot be recorded, so its result token "
        "can never appear in a valid archive"
    )


@pytest.mark.parametrize(
    ("failed_stage", "fabricated"),
    [
        ("build", "extension_load"),
        ("build", "runtime_projection"),
        ("build_binding", "extension_load"),
        ("extension_load", "extension_load"),
        ("extension_load", "runtime_projection"),
        ("identity_run", "runtime_projection"),
    ],
)
def test_an_unreached_stage_cannot_be_fabricated(
    failed_stage: str, fabricated: str
) -> None:
    """Absence records an unreached stage; a success shape would be a claim."""
    validator = _binding_validator()
    document = _binding(failed_stage=failed_stage)
    if failed_stage == "build":
        document["build_artifacts"] = []
    for key in ("extension_load", "runtime_projection"):
        if key != fabricated:
            document.pop(key, None)
    assert list(validator.iter_errors(document))


def test_a_stage_failure_still_binds_what_it_did_reach() -> None:
    validator = _binding_validator()
    document = _binding(
        failed_stage="extension_load", extension_load=_DROP, diagnostics=_DROP
    )
    document["build_artifacts"] = document["build_artifacts"][:1]
    assert list(validator.iter_errors(document)), (
        "a load failure implies the build completed, so its artifacts stay complete"
    )


def test_a_detected_mutation_is_recordable() -> None:
    """Otherwise terminal 1 has no archive, which is the defect not the guard."""
    validator = _binding_validator()
    assert not list(
        validator.iter_errors(
            _binding(
                input_monitor={
                    "started_before_binding": True,
                    "changed_count": 3,
                    "final_drain_clean": False,
                }
            )
        )
    )
    assert list(
        validator.iter_errors(
            _binding(
                input_monitor={
                    "started_before_binding": False,
                    "changed_count": 0,
                    "final_drain_clean": True,
                }
            )
        )
    )


def test_the_cross_artifact_rules_are_published_for_the_verifier() -> None:
    """No JSON Schema sees two files, so these are ruler facts W3 must import."""
    published = partition.as_payload()["successor_vocabulary"]
    assert published["binding_stages"] == list(partition.BINDING_STAGES)
    assert published["bindable_failure_stages"] == list(
        partition.BINDABLE_FAILURE_STAGES
    )
    assert published["result_requires_failed_stage"] == {
        "build_failed": "build",
        "extension_load_failed": "extension_load",
    }
    assert published["result_requires_input_mutation"] == "input_mutated"
    # The reverse direction is gated on the ordered verdict, and the payload must
    # say so or a payload-only implementer rebuilds the deadlock.
    assert (
        published["failed_stage_requires_result_only_when_terminal"]
        == partition.EXECUTION_INVALID_TERMINAL
    )
    # Reachability, not terminals: a terminal-level list said "terminals 1-3 may
    # carry any stage failure", which admitted a capture finding from an execution
    # that stopped at `build`.
    assert published["stage_independent_results"] == list(
        partition.STAGE_INDEPENDENT_RESULTS
    )
    assert published["probe_derived_results"] == list(partition.PROBE_DERIVED_RESULTS)
    assert published["run_derived_results"] == list(partition.RUN_DERIVED_RESULTS)
    assert set(published["results_admissible_with_a_failed_stage"]).isdisjoint(
        {*partition.RUN_DERIVED_RESULTS, *partition.PROBE_DERIVED_RESULTS}
    )
    assert published["failed_stage_requires_unstarted_runs"] is True
    assert published["failed_stage_forbidden_under_non_terminal_progression"] is True
    assert published["diagnostic_records_evidence_without_demanding_a_result"] is True
    schema = json.loads(
        _SUCCESSOR_SCHEMA_PATHS["runtime_binding"].read_text(encoding="utf-8")
    )
    assert set(schema["properties"]["failed_stage"]["oneOf"][0]["enum"]) == set(
        partition.BINDABLE_FAILURE_STAGES
    )


_MONITOR_CLEAN = {
    "started_before_binding": True,
    "changed_count": 0,
    "final_drain_clean": True,
}
_MONITOR_MUTATED = {
    "started_before_binding": True,
    "changed_count": 1,
    "final_drain_clean": False,
}
_T1 = "H2_INPUT_MUTATED_DURING_MEASUREMENT"
_T2 = "H2_CAPTURE_PERTURBS_POLICY"
_T3 = "H2_PACKET_INVALID"
_T4 = "H2_MEASUREMENT_EXECUTION_INVALID"
_STAGES = list(partition.BINDABLE_FAILURE_STAGES)


def _runs(*, started: bool) -> list[dict[str, Any]]:
    state = "completed" if started else "not_run"
    return [
        {
            "run_id": run_id,
            "state": state,
            "artifact_digest": "4" * 64 if started else None,
        }
        for run_id in evidence.RUN_IDS
    ]


def _agreement(
    result: str,
    *,
    terminal: str | None,
    stage: str | None,
    authority: str = "exactly_once_measurement",
    mutated: bool = False,
    runs_started: bool | None = None,
    probe: bool | None = None,
) -> tuple[str, ...]:
    """Ask the checker about one archive.

    The two defaults encode what the binding schema already forces, so a test has
    to *opt in* to an incoherent archive rather than stumble into one: a stage
    failure means no run started and no probe exists.
    """
    if runs_started is None:
        runs_started = stage is None
    if probe is None:
        probe = stage is None
    return partition.binding_agreement_reasons(
        result,
        authority=authority,
        selected_terminal=terminal,
        failed_stage=stage,
        input_monitor=_MONITOR_MUTATED if mutated else _MONITOR_CLEAN,
        ordered_runs=_runs(started=runs_started),
        identity_probe_present=probe,
    )


@pytest.mark.parametrize(
    ("result", "terminal", "stage", "mutated", "expected"),
    [
        # the token -> stage direction is unconditional
        ("build_failed", _T4, "build", False, ()),
        ("build_failed", _T4, None, False, ("requires failed_stage 'build'",)),
        (
            # violates both directions at once, and says both
            "build_failed",
            _T4,
            "extension_load",
            False,
            (
                "requires failed_stage 'build'",
                "requires result 'extension_load_failed'",
            ),
        ),
        ("extension_load_failed", _T4, "extension_load", False, ()),
        # the stage -> token direction only when terminal 4 is the ordered winner
        ("runner_nonzero", _T4, None, False, ()),
        ("runner_nonzero", _T4, "build", False, ("requires result 'build_failed'",)),
        ("unclassified_execution_failure", _T4, "build_binding", False, ()),
        ("unclassified_execution_failure", _T4, "identity_run", False, ()),
        ("runner_timeout", _T4, "identity_run", False, ("no dedicated result token",)),
        # terminal 1 wins, and a stage-independent finding may sit beside a stage
        ("input_mutated", _T1, "build", True, ()),
        ("input_mutated", _T1, "extension_load", True, ()),
        ("input_mutated", _T1, None, True, ()),
        # Correction 10 narrowed this to what a launch boundary received, which no
        # stage failure ever reaches — so it left the stage-independent class.
        (
            "runtime_binding_mismatch",
            _T1,
            "build",
            False,
            ("run-derived evidence",),
        ),
        ("runtime_binding_mismatch", _T1, None, False, ()),
        ("build_failed", _T1, "build", True, ("outranks every other finding",)),
        # a finding whose evidence the execution never reached
        ("capture_perturbs_policy", _T2, "build", False, ("run-derived evidence",)),
        ("packet_invalid", _T3, "identity_run", False, ("run-derived evidence",)),
        # the mutation rule stays biconditional for a measurement
        ("input_mutated", _T1, None, False, ("requires the monitor to record",)),
        ("packet_invalid", _T3, None, True, ("outranks every other finding",)),
        ("measurement_pass", None, None, False, ()),
        ("measurement_pass", None, "build", False, ("requires failed_stage null",)),
    ],
)
def test_the_cross_artifact_rules_respect_authority_and_precedence(
    result: str,
    terminal: str | None,
    stage: str | None,
    mutated: bool,
    expected: tuple[str, ...],
) -> None:
    reasons = _agreement(result, terminal=terminal, stage=stage, mutated=mutated)
    assert len(reasons) == len(expected), reasons
    for reason, fragment in zip(reasons, expected):
        assert fragment in reason


def _admissible_results(
    *, terminal: str | None, stage: str | None, mutated: bool = False
) -> list[str]:
    """Every measurement result that agrees with this archive, under this verdict."""
    return [
        result
        for result, mapped in partition.RESULT_TO_TERMINAL.items()
        if result not in partition.LEGACY_RESULT_SUPERSEDED_BY
        and mapped == terminal
        and not _agreement(result, terminal=terminal, stage=stage, mutated=mutated)
    ]


@pytest.mark.parametrize("stage", [None, *_STAGES])
@pytest.mark.parametrize("mutated", [False, True])
def test_a_moved_input_always_leaves_one_admissible_result(
    stage: str | None, mutated: bool
) -> None:
    """No cell may be a deadlock: a formable observation keeps a sayable result."""
    terminal = _T1 if mutated else _T4
    admissible = _admissible_results(terminal=terminal, stage=stage, mutated=mutated)
    assert admissible, (
        f"no result agrees with failed_stage={stage!r} and mutated={mutated} — the "
        "cross-artifact rules deadlock on a formable observation"
    )
    if mutated:
        assert admissible == ["input_mutated"]


@pytest.mark.parametrize("stage", _STAGES)
def test_a_non_terminal_pass_cannot_carry_a_stage_failure(stage: str) -> None:
    """`measurement_pass` requires `execution_complete` to pass; a stage failure denies it."""
    reasons = _agreement("measurement_pass", terminal=None, stage=stage)
    assert reasons
    assert "failed_stage null" in reasons[0]
    assert _admissible_results(terminal=None, stage=stage) == []


@pytest.mark.parametrize("stage", _STAGES)
def test_a_stage_failure_admits_only_the_findings_it_could_have_produced(
    stage: str,
) -> None:
    """Reachability, not terminals — the axis a terminal-level rule cannot express.

    An execution that stopped at a retained stage started no measurement run and
    computed no probe, so a capture-perturbation or invalid-packet finding names
    evidence that cannot exist yet even though its terminal outranks terminal 4.
    Enumerating by result rather than by terminal is what makes that visible: the
    previous form asserted every Phase-A-reachable *terminal* admits every stage.
    """
    admissible = {
        result
        for result, terminal in partition.RESULT_TO_TERMINAL.items()
        if result not in partition.LEGACY_RESULT_SUPERSEDED_BY
        and not _agreement(
            result,
            terminal=terminal,
            stage=stage,
            mutated=result == "input_mutated",
        )
    }
    expected = {
        *partition.STAGE_INDEPENDENT_RESULTS,
        *(
            result
            for result, terminal in partition.RESULT_TO_TERMINAL.items()
            if terminal == partition.EXECUTION_INVALID_TERMINAL
            and not _agreement(result, terminal=terminal, stage=stage)
        ),
    }
    assert admissible == expected
    assert admissible.isdisjoint(partition.RUN_DERIVED_RESULTS)
    assert admissible.isdisjoint(partition.PROBE_DERIVED_RESULTS)
    assert "measurement_pass" not in admissible


@pytest.mark.parametrize("result", list(partition.RUN_DERIVED_RESULTS))
def test_a_run_derived_finding_needs_a_run_that_started(result: str) -> None:
    """Independent of the stage: an unstarted run block produced no evidence."""
    terminal = partition.RESULT_TO_TERMINAL[result]
    assert _agreement(result, terminal=terminal, stage=None, runs_started=True) == ()
    assert _agreement(result, terminal=terminal, stage=None, runs_started=False)


@pytest.mark.parametrize("stage", _STAGES)
def test_a_stage_failure_requires_every_run_block_unstarted(stage: str) -> None:
    """Correction 5's order plus fail-fast: no run starts before the stages finish."""
    reasons = _agreement(
        "build_failed" if stage == "build" else "unclassified_execution_failure",
        terminal=_T4,
        stage=stage,
        runs_started=True,
    )
    assert any(
        "stopped the execution before the measurement runs" in r for r in reasons
    )


@pytest.mark.parametrize("stage", [None, *_STAGES])
@pytest.mark.parametrize("mutated", [False, True])
def test_a_diagnostic_records_stage_evidence_without_demanding_a_result(
    stage: str | None, mutated: bool
) -> None:
    """A diagnostic stays `diagnostic_complete` however red it is (Correction 5)."""
    assert (
        _agreement(
            partition.DIAGNOSTIC_RESULT,
            terminal=None,
            stage=stage,
            authority="non_qualifying_diagnostic",
            mutated=mutated,
        )
        == ()
    )


def test_the_cross_artifact_checker_fails_closed_on_its_own_inputs() -> None:
    for kwargs in (
        {"authority": "rehearsal", "selected_terminal": None},
        {"authority": "exactly_once_measurement", "selected_terminal": "H2_UNKNOWN"},
        {"authority": "non_qualifying_diagnostic", "selected_terminal": _T1},
    ):
        with pytest.raises(partition.PartitionError):
            partition.binding_agreement_reasons(
                "measurement_pass",
                failed_stage=None,
                input_monitor=_MONITOR_CLEAN,
                ordered_runs=_runs(started=True),
                identity_probe_present=True,
                **kwargs,
            )
    with pytest.raises(partition.PartitionError, match="unknown failed stage"):
        partition.binding_agreement_reasons(
            "measurement_pass",
            authority="exactly_once_measurement",
            selected_terminal=None,
            failed_stage="preflight",
            input_monitor=_MONITOR_CLEAN,
            ordered_runs=_runs(started=True),
            identity_probe_present=True,
        )
    with pytest.raises(partition.PartitionError, match="missing changed_count"):
        partition.binding_agreement_reasons(
            "measurement_pass",
            authority="exactly_once_measurement",
            selected_terminal=None,
            failed_stage=None,
            input_monitor={"started_before_binding": True, "final_drain_clean": True},
            ordered_runs=_runs(started=True),
            identity_probe_present=True,
        )
    with pytest.raises(partition.PartitionError, match="unknown state"):
        partition.binding_agreement_reasons(
            "measurement_pass",
            authority="exactly_once_measurement",
            selected_terminal=None,
            failed_stage=None,
            input_monitor=_MONITOR_CLEAN,
            ordered_runs=[{"run_id": "00_capture_off", "state": "skipped"}],
            identity_probe_present=True,
        )


def test_an_unclean_final_drain_is_a_recorded_mutation() -> None:
    monitor = {
        "started_before_binding": True,
        "changed_count": 0,
        "final_drain_clean": False,
    }
    assert (
        partition.binding_agreement_reasons(
            "input_mutated",
            authority="exactly_once_measurement",
            selected_terminal=_T1,
            failed_stage=None,
            input_monitor=monitor,
            ordered_runs=_runs(started=True),
            identity_probe_present=True,
        )
        == ()
    )
    assert partition.binding_agreement_reasons(
        "measurement_pass",
        authority="exactly_once_measurement",
        selected_terminal=None,
        failed_stage=None,
        input_monitor=monitor,
        ordered_runs=_runs(started=True),
        identity_probe_present=True,
    )


def test_terminal_four_causes_are_separated_by_stage_evidence() -> None:
    """Where the named cause is pinned, and where it deliberately is not.

    Within `result.json` the terminal-4 tokens are interchangeable — they share a
    terminal and the same failing predicate — so a validator cannot tell
    `build_failed` from `runner_nonzero`. What separates them is the stage the
    binding says failed, which is why the biconditional is a verifier obligation
    and not a schema constraint: no JSON Schema sees two files.
    """
    validator = _result_validator()
    record = _successor_states(execution_complete="fail")
    selection = partition.select_successor_result(
        record, authority="exactly_once_measurement", phase="a"
    )
    instance = _result_instance(selection, record)
    for token, stage in partition.RESULT_REQUIRES_FAILED_STAGE.items():
        swapped = {**instance, "result": token}
        assert not list(validator.iter_errors(swapped)), (
            "the result schema is not the place this is decided"
        )
        assert _agreement(token, terminal=_T4, stage=None), (
            f"{token} was accepted against a binding that completed every stage"
        )
        assert _agreement(token, terminal=_T4, stage=stage) == ()
        assert _agreement("runner_nonzero", terminal=_T4, stage=stage), (
            "a stage failure was accepted under a result that does not name it"
        )
