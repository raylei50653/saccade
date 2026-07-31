"""The archive-only verifier: what a closed H2 execution must survive to be believed.

Review Correction 5 gives the successor path two commands and one rule about the
seam between them. The producer emits three artifacts and may not write the
verdict; a separate command reads only those bytes and their closure and writes
`verification.json`. This suite pins the seam rather than the implementation:

  * **the verdict is recomputed, never read** — the recorded result and terminal
    must equal what `select_successor_result` selects from the archive's own
    predicates, and `binding_agreement_reasons` is handed that recomputed
    terminal. Passing it the archive's terminal would let a record certify its
    own verdict;
  * **the verifier holds no rule of its own** — it is `plumbing_only`, so a rule
    typed out here could change without `identity_semantics` moving (§ C3.9).
    Every semantic name is imported;
  * **the verdict does not depend on the verifying host** — the one function in
    the ruler that hashes the local checkout is made to raise, and a faithful
    archive still verifies. This is Correction 5's retirement of reproducibility
    stated as a test rather than as a comment;
  * **a defect inside a formable archive is a verdict, not a crash** — schema
    violations, malformed members, disagreeing digests and refused verdicts are
    recorded as `valid: false`. Two rules together decide what is unformable
    instead: the verification record's own required fields must be fillable, and
    the root must be physically flat;
  * **the closure is a three-state machine** — neither closing record, or both.
    A half-closed archive must not verify, because `O_EXCL` will not let it be
    completed; and a stored verdict is compared, not merely counted, since
    re-deriving *a* verdict says nothing about the one the archive carries.

Every fixture is synthesised from the frozen schemas and the frozen authoring
profile, never from producer output: § 5.3's circular-oracle rule says the
contract may not be defined by whatever an implementation happens to emit, and
no producer exists yet to emit anything.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_execution as verifier  # noqa: E402

_CONTRACTS = _REPO / "docs" / "research" / "contracts"
_RESULT_SCHEMA = json.loads(
    (_CONTRACTS / "h2_execution_result_v1.json").read_text(encoding="utf-8")
)
_BINDING_SCHEMA = json.loads(
    (_CONTRACTS / "h2_runtime_binding_v1.json").read_text(encoding="utf-8")
)
_VERIFICATION_SCHEMA = json.loads(
    (_CONTRACTS / "h2_execution_verification_v1.json").read_text(encoding="utf-8")
)

# Read from the contracts rather than retyped: the run plan and the executed
# surface set are the schemas' own constants.
RUN_IDS: tuple[str, ...] = tuple(
    _RESULT_SCHEMA["properties"]["run_plan"]["properties"]["run_ids"]["const"]
)
SEQUENCE = _RESULT_SCHEMA["properties"]["run_plan"]["properties"]["sequence"]["const"]
EXECUTED_SURFACE_PATHS: tuple[str, ...] = tuple(
    block["contains"]["properties"]["path"]["const"]
    for block in _BINDING_SCHEMA["properties"]["executed_surfaces"]["allOf"]
)
CAPTURE_ABI_PATH = _BINDING_SCHEMA["properties"]["capture_abi"]["allOf"][1][
    "properties"
]["path"]["const"]
PREDICATES: tuple[str, ...] = tuple(
    _RESULT_SCHEMA["properties"]["predicate_results"]["required"]
)
# The predicate Correction 10 narrowed. Named once here and checked against the
# frozen schema's own predicate set, so a rename cannot leave these tests
# asserting about a predicate that no longer exists.
PROJECTION_PREDICATE = "runtime_projection_matches_resolved_run_spec"
assert PROJECTION_PREDICATE in PREDICATES

EXECUTION_ID = "h2exec-20260731T000000Z"


def _fake(seed: str) -> str:
    """A stable stand-in digest. Never a real file hash: nothing here reads bytes."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


_EXTENSION_DIGEST = _fake("tracking_extension")
_PLUGIN_DIGEST = _fake("tensorrt_scan_plugin")


def _projection_members() -> list[dict[str, Any]]:
    return sorted(
        (
            {"length": 1000 + index, "path": relative, "sha256": _fake(relative)}
            for index, relative in enumerate(run_spec_module.EXECUTION_SEMANTICS_PATHS)
        ),
        key=lambda member: str(member["path"]),
    )


def _by_path() -> dict[str, dict[str, Any]]:
    return {str(member["path"]): member for member in _projection_members()}


def _run_spec() -> dict[str, Any]:
    """A RunSpec the resolver accepts, built from the frozen profile's namespace."""
    profile = json.loads(
        (_CONTRACTS / "h2_phase_a_authoring_profile_v1.json").read_text(
            encoding="utf-8"
        )
    )
    resolved = dict(profile["resolved_namespace"])
    members = _projection_members()
    declared = _by_path()
    projection = {
        "algorithm": "sha256_canonical_json_content_members_v1",
        "digest": runtime_inputs.digest(members),
        "members": members,
        "schema": run_spec_module.PROJECTION_SCHEMA,
    }
    document: dict[str, Any] = {
        "artifact_serialization": run_spec_module.ARTIFACT_SERIALIZATION,
        "authoring_profile": {
            "authoring_lineage": {
                "derived_from": "configs/presets/mamba_whole_graph_m.yaml",
                "resolution": "one_time_parser_resolution_plus_owner_adjudication",
                "runtime_preset_loader": False,
                "source_sha256": _fake("preset"),
            },
            "owner_decision": run_spec_module.AUTHORING_DECISION_REL,
            "owner_decision_sha256": declared[run_spec_module.AUTHORING_DECISION_REL][
                "sha256"
            ],
            "profile": run_spec_module.AUTHORING_PROFILE_REL,
            "profile_schema": run_spec_module.AUTHORING_PROFILE_SCHEMA_REL,
            "profile_schema_sha256": declared[
                run_spec_module.AUTHORING_PROFILE_SCHEMA_REL
            ]["sha256"],
            "profile_sha256": declared[run_spec_module.AUTHORING_PROFILE_REL]["sha256"],
            "schema": run_spec_module.AUTHORING_BINDING_SCHEMA,
        },
        "execution_semantics_projection": projection,
        "execution_semantics_projection_digest": projection["digest"],
        "namespace_schema": run_spec_module.NAMESPACE_SCHEMA,
        "object_canonicalization": run_spec_module.OBJECT_CANONICALIZATION,
        "phase": "phase_a",
        "resolved_namespace": resolved,
        "resolved_namespace_digest": runtime_inputs.digest(resolved),
        "resolved_namespace_keys": sorted(resolved),
        "schema": run_spec_module.RUN_SPEC_SCHEMA,
    }
    document["resolved_run_spec_digest"] = runtime_inputs.digest(
        {
            key: value
            for key, value in document.items()
            if key != "resolved_run_spec_digest"
        }
    )
    return document


_RUN_SPEC = _run_spec()
_SPEC_DIGEST = _RUN_SPEC["resolved_run_spec_digest"]
_PROJECTION_DIGEST = _RUN_SPEC["execution_semantics_projection_digest"]

# What the launch boundary received, synthesised from the frozen profile's own
# namespace by this file's reading of the four RunSpec-owned keys. Deriving it
# by calling either implementation would make the fixture agree with whichever
# one it called, which is the circularity § 5.3 forbids.
_LAUNCH_ENVIRONMENT: dict[str, str] = {
    "SACCADE_DETECT_BARRIER": (
        "event"
        if _RUN_SPEC["resolved_namespace"]["double_buffer"]
        else (_RUN_SPEC["resolved_namespace"].get("detect_barrier") or "full")
    ),
    "SACCADE_DOUBLE_BUFFER": (
        "1" if _RUN_SPEC["resolved_namespace"]["double_buffer"] else "0"
    ),
    "SACCADE_GPU_DECODE": (
        "0" if _RUN_SPEC["resolved_namespace"]["no_gpu_decode"] else "1"
    ),
    "SACCADE_MAIN_NMS_GRAPHED": (
        "1" if _RUN_SPEC["resolved_namespace"]["main_nms_graphed"] else "0"
    ),
}


def _launch_projection(
    *, environment: dict[str, Any] | None = None, run_ids: tuple[str, ...] = RUN_IDS
) -> dict[str, Any]:
    received = _LAUNCH_ENVIRONMENT if environment is None else environment
    return {
        "observations": [
            {"environment": dict(received), "run_id": run_id} for run_id in run_ids
        ],
        "resolved_run_spec_digest": _SPEC_DIGEST,
    }


_LAUNCH_PROJECTION = _launch_projection()


def _binding(**overrides: Any) -> dict[str, Any]:
    declared = _by_path()
    document: dict[str, Any] = {
        "build_artifacts": [
            {
                "length": 4096,
                "path": "build/h2_layer_p/libsaccade_tracking.so",
                "role": "tracking_extension",
                "sha256": _EXTENSION_DIGEST,
            },
            {
                "length": 8192,
                "path": "build/h2_layer_p/libscan_plugin.so",
                "role": "tensorrt_scan_plugin",
                "sha256": _PLUGIN_DIGEST,
            },
        ],
        "capture_abi": declared[CAPTURE_ABI_PATH],
        "executed_surfaces": [declared[path] for path in EXECUTED_SURFACE_PATHS],
        "execution_id": EXECUTION_ID,
        "execution_semantics_projection_digest": _PROJECTION_DIGEST,
        "extension_load": {
            "length": 4096,
            "loaded_path": "/opt/saccade/build/h2_layer_p/libsaccade_tracking.so",
            "sha256": _EXTENSION_DIGEST,
        },
        "failed_stage": None,
        "diagnostics": {
            "behavior_probe": {
                "digest": _fake("probe"),
                "role": "recorded_diagnostic_observation_selects_nothing",
                "schema": "h2_behavior_probe_result_v1",
                "state": "computed",
            }
        },
        "runtime_projection": _LAUNCH_PROJECTION,
        "input_monitor": {
            "changed_count": 0,
            "final_drain_clean": True,
            "started_before_binding": True,
        },
        "resolved_run_spec_digest": _SPEC_DIGEST,
        "runtime_inputs": {
            "manifest_digest": _fake("manifest"),
            # An execution that stopped at `build` hashed no build artifacts, so
            # it records the coordinate form; every other binding records the
            # full manifest. The schema pins which one, per failed stage.
            "manifest_schema": (
                "h2_runtime_input_coordinate_v1"
                if overrides.get("failed_stage") == "build"
                else "h2_runtime_input_manifest_v1"
            ),
            "members": [
                {
                    "length": 512,
                    "path": "datasets/MOT17/train/MOT17-04-SDP/seqinfo.ini",
                    "role": "measurement_fixture_input",
                    "sha256": _fake("seqinfo"),
                }
            ],
        },
        "schema": "h2_runtime_binding_v1",
        "source_audit": {"head": "a" * 40, "tree": "b" * 40},
    }
    document.update(overrides)
    return {key: value for key, value in document.items() if value is not _ABSENT}


class _Absent:
    """Marker for a member a stage failure forbids the binding to carry."""


_ABSENT = _Absent()


def _predicates(**states: str) -> dict[str, dict[str, Any]]:
    return {
        name: {"reasons": [], "state": states.get(name, "pass")} for name in PREDICATES
    }


def _runs(state: str = "completed") -> list[dict[str, Any]]:
    return [
        {
            "artifact_digest": _fake(run_id) if state == "completed" else None,
            "run_id": run_id,
            "state": state,
        }
        for run_id in RUN_IDS
    ]


def _result(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "authority": "exactly_once_measurement",
        "authorization_binding_digest": _fake("authorization"),
        "execution_id": EXECUTION_ID,
        "execution_semantics_projection_digest": _PROJECTION_DIGEST,
        "ordered_runs": _runs(),
        "predicate_results": _predicates(),
        "resolved_run_spec_digest": _SPEC_DIGEST,
        "result": "measurement_pass",
        "run_plan": {"run_ids": list(RUN_IDS), "sequence": SEQUENCE},
        "schema": "h2_execution_result_v1",
        "terminal": None,
    }
    document.update(overrides)
    return document


def _archive(
    tmp_path: Path,
    *,
    run_spec: dict[str, Any] | None = None,
    binding: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    for name, document in (
        ("run_spec.json", _RUN_SPEC if run_spec is None else run_spec),
        ("runtime_binding.json", _binding() if binding is None else binding),
        ("result.json", _result() if result is None else result),
    ):
        (root / name).write_bytes(runtime_inputs.canonical_json_bytes(document) + b"\n")
    return root


# -- the fixtures are the contract, so they must satisfy it ----------------- #


def test_the_synthetic_measurement_archive_satisfies_the_frozen_schemas() -> None:
    """A fixture the schemas would refuse proves nothing about the verifier."""
    jsonschema.validate(
        instance=_RUN_SPEC,
        schema=verifier._load_contract(verifier.PRODUCER_ARTIFACTS["run_spec.json"]),
    )
    jsonschema.validate(instance=_binding(), schema=_BINDING_SCHEMA)
    jsonschema.validate(instance=_result(), schema=_RESULT_SCHEMA)


def test_the_verified_artifact_names_come_from_the_verification_contract() -> None:
    """A renamed artifact must break loudly, not silently stop being verified."""
    required = set(_VERIFICATION_SCHEMA["properties"]["artifact_digests"]["required"])
    assert set(verifier.PRODUCER_ARTIFACTS) == required
    assert set(verifier.CHECKS) == set(
        _VERIFICATION_SCHEMA["properties"]["checks"]["required"]
    )


# -- the faithful archive --------------------------------------------------- #


def test_a_faithful_measurement_archive_verifies(tmp_path: Path) -> None:
    record = verifier.verify_archive(_archive(tmp_path))
    assert record["reasons"] == []
    assert record["valid"] is True
    assert all(record["checks"].values())
    assert record["producer_invoked"] is False
    assert record["verification_host_inputs_used"] is False
    verifier.validate_verification(record)


def test_a_diagnostic_archive_verifies_and_selects_no_terminal(tmp_path: Path) -> None:
    """A diagnostic resolves to `diagnostic_complete` however red it is."""
    root = _archive(
        tmp_path,
        binding=_binding(failed_stage=None),
        result=_result(
            authority="non_qualifying_diagnostic",
            authorization_binding_digest=None,
            predicate_results=_predicates(packets_valid="fail"),
            result=partition.DIAGNOSTIC_RESULT,
            terminal=None,
        ),
    )
    record = verifier.verify_archive(root)
    assert record["valid"] is True, record["reasons"]


def test_the_verdict_does_not_depend_on_the_verifying_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Correction 5 retired reproducibility; this is where that retirement lives.

    `execution_semantics_projection()` is the one ruler function that hashes the
    local checkout. If the verdict needed it, a foreign host would have to
    reconstruct the execution's content set to verify an archive — which is the
    dependence the correction removed. Making it raise is a mechanical proof that
    the archive-only path never calls it.
    """

    def _forbidden() -> dict[str, Any]:
        raise AssertionError("the verifier read the verifying host's checkout")

    monkeypatch.setattr(run_spec_module, "execution_semantics_projection", _forbidden)
    record = verifier.verify_archive(_archive(tmp_path))
    assert record["valid"] is True, record["reasons"]

    # And the flag is a real boundary, not a decoration: with it on, the very
    # same document reaches the checkout-reading function this test forbids.
    with pytest.raises(AssertionError, match="verifying host's checkout"):
        run_spec_module.validate_run_spec(_RUN_SPEC, verify_projection=True)


# -- a verdict the archive cannot certify for itself ------------------------ #


def test_a_recorded_verdict_its_own_predicates_refuse_is_invalid(
    tmp_path: Path,
) -> None:
    """The archive names terminal 4's cause; it does not name the terminal."""
    root = _archive(
        tmp_path,
        result=_result(
            predicate_results=_predicates(execution_complete="fail"),
            result="measurement_pass",
            terminal=None,
        ),
    )
    record = verifier.verify_archive(root)
    assert record["valid"] is False
    assert record["checks"]["result_binding"] is False
    assert any("select" in reason for reason in record["reasons"])


def test_a_stage_failure_that_names_a_started_run_is_invalid(tmp_path: Path) -> None:
    """Schema-legal, ruler-refused: reachability is the cross-artifact axis.

    Nothing inside `result.json` forbids four completed runs while
    `runtime_binding.json` says the build never finished. Only the two files read
    together say it, which is why this is the verifier's job and not the schema's.
    """
    root = _archive(
        tmp_path,
        binding=_binding(
            failed_stage="build",
            runtime_projection=_ABSENT,
            build_artifacts=[
                {
                    "length": 4096,
                    "path": "build/h2_layer_p/libsaccade_tracking.so",
                    "role": "tracking_extension",
                    "sha256": _EXTENSION_DIGEST,
                }
            ],
            extension_load=_ABSENT,
            identity_probe=_ABSENT,
        ),
        result=_result(
            predicate_results=_predicates(execution_complete="fail"),
            result="build_failed",
            terminal=partition.EXECUTION_INVALID_TERMINAL,
        ),
    )
    record = verifier.verify_archive(root)
    assert record["checks"]["artifact_schemas"] is True, record["reasons"]
    assert record["checks"]["result_binding"] is False
    assert any("before the measurement runs" in reason for reason in record["reasons"])


def test_a_stage_failure_must_name_the_result_its_stage_requires(
    tmp_path: Path,
) -> None:
    root = _archive(
        tmp_path,
        binding=_binding(
            failed_stage="build",
            runtime_projection=_ABSENT,
            build_artifacts=[
                {
                    "length": 4096,
                    "path": "build/h2_layer_p/libsaccade_tracking.so",
                    "role": "tracking_extension",
                    "sha256": _EXTENSION_DIGEST,
                }
            ],
            extension_load=_ABSENT,
            identity_probe=_ABSENT,
        ),
        result=_result(
            ordered_runs=_runs("not_run"),
            predicate_results=_predicates(execution_complete="fail"),
            result="runner_nonzero",
            terminal=partition.EXECUTION_INVALID_TERMINAL,
        ),
    )
    record = verifier.verify_archive(root)
    assert record["checks"]["result_binding"] is False
    assert any("'build_failed'" in reason for reason in record["reasons"])


# -- the bytes that ran are the bytes that were recorded -------------------- #


def test_the_extension_that_loaded_must_be_the_extension_that_was_built(
    tmp_path: Path,
) -> None:
    loaded = {
        "length": 4096,
        "loaded_path": "/opt/saccade/build/h2_layer_p/libsaccade_tracking.so",
        "sha256": _fake("some other extension"),
    }
    record = verifier.verify_archive(
        _archive(tmp_path, binding=_binding(extension_load=loaded))
    )
    assert record["checks"]["execution_binding"] is False
    assert any("bytes this execution built" in reason for reason in record["reasons"])


def test_an_executed_surface_outside_the_declared_content_set_is_invalid(
    tmp_path: Path,
) -> None:
    """Equal projection digests are not the whole claim; the members must agree."""
    surfaces = [_by_path()[path] for path in EXECUTED_SURFACE_PATHS]
    surfaces[0] = {**surfaces[0], "sha256": _fake("a different revision")}
    record = verifier.verify_archive(
        _archive(tmp_path, binding=_binding(executed_surfaces=surfaces))
    )
    assert record["checks"]["projection_binding"] is False
    assert any("different bytes of" in reason for reason in record["reasons"])


def test_an_artifact_naming_another_run_spec_is_invalid(tmp_path: Path) -> None:
    record = verifier.verify_archive(
        _archive(tmp_path, result=_result(resolved_run_spec_digest=_fake("elsewhere")))
    )
    assert record["checks"]["run_spec_binding"] is False


def test_two_artifacts_naming_two_executions_are_invalid(tmp_path: Path) -> None:
    record = verifier.verify_archive(
        _archive(tmp_path, result=_result(execution_id="h2exec-someone-else"))
    )
    assert record["checks"]["execution_binding"] is False


def test_an_uncanonical_run_spec_serialization_is_invalid(tmp_path: Path) -> None:
    """Correction 7's two byte domains: the file adds exactly one LF, no more."""
    root = _archive(tmp_path)
    (root / "run_spec.json").write_bytes(
        runtime_inputs.canonical_json_bytes(_RUN_SPEC) + b"\n\n"
    )
    record = verifier.verify_archive(root)
    assert record["checks"]["artifact_schemas"] is False
    assert any("artifact serialization" in reason for reason in record["reasons"])


# -- a malformed member is a verdict, not a traceback ----------------------- #


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"predicate_results": []}, "predicate_results that are not an object"),
        ({"ordered_runs": "x"}, "ordered_runs that are not a list of objects"),
        (
            {"ordered_runs": [None, None, None, None]},
            "ordered_runs that are not a list of objects",
        ),
    ],
)
def test_a_malformed_container_is_recorded_not_raised(
    tmp_path: Path, overrides: dict[str, Any], fragment: str
) -> None:
    """Formable identity plus readable JSON plus an invalid shape is still a verdict.

    The ruler is total over observations, but only over observations: it names
    every unknown predicate and run *state*, and a list where an object belongs
    is not a state. Guarding the container shape here keeps the fail-closed rule
    at the plumbing boundary instead of widening the ruler's tolerance — and
    `ordered_runs` needs `list`, because a string satisfies `Sequence` and would
    walk into the algebra one character at a time.
    """
    record = verifier.verify_archive(_archive(tmp_path, result=_result(**overrides)))
    assert record["valid"] is False
    assert record["checks"]["result_binding"] is False
    assert record["checks"]["artifact_schemas"] is False
    assert any(fragment in reason for reason in record["reasons"])
    verifier.validate_verification(record)


@pytest.mark.parametrize(
    ("projection", "fragment"),
    [
        ([], "runtime_projection that is not an object"),
        (
            {"observations": [None], "resolved_run_spec_digest": _SPEC_DIGEST},
            "observation that is not an object",
        ),
    ],
)
def test_a_malformed_launch_projection_is_recorded_not_raised(
    tmp_path: Path, projection: Any, fragment: str
) -> None:
    """The same boundary as `ordered_runs`, for the newest container to cross it.

    The ruler calls `.get()` on the projection and on every observation, so a list
    where the object belongs — or a null inside the observation list — reached it
    as an `AttributeError` rather than as a verdict. The guard belongs before the
    ruler call, not after it.
    """
    record = verifier.verify_archive(
        _archive(tmp_path, binding=_binding(runtime_projection=projection))
    )
    assert record["valid"] is False
    assert record["checks"]["launch_projection"] is False
    assert record["checks"]["artifact_schemas"] is False
    assert any(fragment in reason for reason in record["reasons"])
    verifier.validate_verification(record)


@pytest.mark.parametrize("state", ["error", "not_run"])
def test_an_undecided_projection_predicate_is_invalid_when_it_recomputes(
    tmp_path: Path, state: str
) -> None:
    """Recomputing the predicate means recording the state, not just refusing one.

    Every launch observation matches the resolved RunSpec, so this verifier has
    decided the predicate: `pass`. An archive that keeps it undecided while a later
    predicate fails would ride the selector's "a decided failure outranks an
    undecided predicate" to terminal 4 with a verdict the verifier itself has
    contradicted, which is the second implementation declining to answer.
    """
    record = verifier.verify_archive(
        _archive(
            tmp_path,
            result=_result(
                predicate_results=_predicates(
                    **{
                        PROJECTION_PREDICATE: state,
                        "execution_complete": "fail",
                    }
                ),
                result="unclassified_execution_failure",
                terminal=partition.EXECUTION_INVALID_TERMINAL,
                ordered_runs=_runs("failed"),
            ),
        )
    )
    assert record["valid"] is False
    assert record["checks"]["launch_projection"] is False
    assert any("recomputes to 'pass'" in reason for reason in record["reasons"])
    verifier.validate_verification(record)


def test_a_non_string_result_token_is_recorded_not_raised(tmp_path: Path) -> None:
    """The recorded token is a lookup key, so it must be hashable before it is one.

    `result: []` reached `RESULT_TO_TERMINAL.get(...)` and raised `TypeError:
    unhashable type` — the same defect class as a list where an object belongs,
    one call site further along.
    """
    record = verifier.verify_archive(_archive(tmp_path, result=_result(result=[])))
    assert record["valid"] is False
    assert record["checks"]["result_binding"] is False
    assert record["checks"]["artifact_schemas"] is False
    verifier.validate_verification(record)


def test_a_non_object_runtime_binding_is_a_verdict_not_an_unformable_archive(
    tmp_path: Path,
) -> None:
    """Formability is about the record's own fields, not about every artifact.

    The execution id comes from `result.json`, both digests from `run_spec.json`,
    and this artifact's digest from its bytes — so every required member of the
    verification record can be filled. A `runtime_binding.json` that is not an
    object is therefore a schema violation inside a formable archive, and the
    four checks that need to read it say so individually.
    """
    root = _archive(tmp_path)
    (root / "runtime_binding.json").write_bytes(b"[]\n")

    record = verifier.verify_archive(root)
    verifier.validate_verification(record)
    assert record["valid"] is False
    assert record["checks"] == {
        "artifact_schemas": False,
        "checksum_closure": True,
        "execution_binding": False,
        "launch_projection": False,
        "projection_binding": False,
        "result_binding": False,
        "run_spec_binding": False,
    }
    assert (
        record["artifact_digests"]["runtime_binding.json"]
        == hashlib.sha256(b"[]\n").hexdigest()
    )


@pytest.mark.parametrize("name", ["result.json", "run_spec.json"])
def test_a_non_object_identity_artifact_forms_no_record(
    tmp_path: Path, name: str
) -> None:
    """These two are different: without them the record's own fields are unfillable."""
    root = _archive(tmp_path)
    (root / name).write_bytes(b"[]\n")
    with pytest.raises(verifier.ExecutionVerificationError):
        verifier.verify_archive(root)


# -- the closure ------------------------------------------------------------ #


def test_an_archive_holding_unexplained_bytes_is_invalid(tmp_path: Path) -> None:
    root = _archive(tmp_path)
    (root / "notes.txt").write_text("anything", encoding="utf-8")
    record = verifier.verify_archive(root)
    assert record["checks"]["checksum_closure"] is False


def test_the_verifier_writes_the_verdict_then_closes_the_inventory(
    tmp_path: Path,
) -> None:
    """The final inventory covers all four records; the record never covers it."""
    root = _archive(tmp_path)
    path, record = verifier.commit_verification(root)
    assert path.name == verifier.VERIFICATION_NAME
    assert record["valid"] is True

    inventory = evidence.read_checksum_inventory(root)
    assert set(inventory) == {*verifier.PRODUCER_ARTIFACTS, verifier.VERIFICATION_NAME}
    assert verifier.CHECKSUMS_NAME not in inventory
    assert json.loads(path.read_text(encoding="utf-8")) == record

    # Re-verifying the closed archive agrees, and cannot overwrite the verdict.
    assert verifier.verify_archive(root)["valid"] is True
    with pytest.raises(FileExistsError):
        verifier.commit_verification(root)


def test_a_tampered_inventory_is_invalid(tmp_path: Path) -> None:
    root = _archive(tmp_path)
    verifier.commit_verification(root)
    (root / verifier.CHECKSUMS_NAME).write_text(
        f"{_fake('not the result')}  result.json\n", encoding="utf-8"
    )
    record = verifier.verify_archive(root)
    assert record["checks"]["checksum_closure"] is False


@pytest.mark.parametrize(
    "interrupted", [verifier.VERIFICATION_NAME, verifier.CHECKSUMS_NAME]
)
def test_a_half_closed_archive_is_invalid(tmp_path: Path, interrupted: str) -> None:
    """The state between the two writes must not verify.

    `commit_verification` writes the verdict and then the inventory, so a crash
    between them leaves an archive with one and not the other. Without this
    parity the half-closed archive verified as valid *and* could not be
    completed, because `O_EXCL` refuses to write the verdict twice: a state that
    is simultaneously believable and unrepairable.
    """
    root = _archive(tmp_path)
    verifier.commit_verification(root)
    (root / interrupted).unlink()
    record = verifier.verify_archive(root)
    assert record["valid"] is False
    assert record["checks"]["checksum_closure"] is False
    assert any("half closed" in reason for reason in record["reasons"])


@pytest.mark.parametrize(
    "forge",
    [
        pytest.param(
            lambda record: {**record, "execution_id": "h2exec-someone-else"},
            id="identity",
        ),
        pytest.param(
            lambda record: {**record, "valid": False, "reasons": ["forged"]},
            id="valid-and-reasons",
        ),
        pytest.param(
            lambda record: {
                **record,
                "checks": {**record["checks"], "checksum_closure": False},
                "valid": False,
                "reasons": ["forged"],
            },
            id="the-closure-check-itself",
        ),
        pytest.param(
            lambda record: {**record, "smuggled": "anything"}, id="extra-member"
        ),
    ],
)
def test_a_rewritten_verdict_is_invalid_even_with_a_matching_inventory(
    tmp_path: Path, forge: Any
) -> None:
    """Re-deriving *a* verdict is not proof that the archive carries that one.

    The comparison covers the whole record, because every member left out of it
    is a member an editor may rewrite while the archive still verifies. An
    earlier form compared only a closure-independent core, which left exactly
    these four doors open: `valid`, `reasons`, the closure check's own field, and
    any additional property.
    """
    root = _archive(tmp_path)
    _, record = verifier.commit_verification(root)
    (root / verifier.VERIFICATION_NAME).write_bytes(
        runtime_inputs.canonical_json_bytes(forge(record)) + b"\n"
    )
    evidence.write_checksum_inventory(root)

    rechecked = verifier.verify_archive(root)
    assert rechecked["valid"] is False
    assert rechecked["checks"]["checksum_closure"] is False
    assert any("stored verdict" in reason for reason in rechecked["reasons"])


def test_re_verifying_a_closed_archive_reproduces_the_stored_record(
    tmp_path: Path,
) -> None:
    """The comparison must be stable, or a closed archive would rot on re-reading.

    What a stored verdict is compared against is built from the artifact checks
    and the *physical* closure alone, so it never depends on the stored verdict.
    The record computed before the two closing writes and the one recomputed
    after them are therefore the same document — which is what makes comparing
    the complete record, rather than a subset of it, a dependency and not a
    cycle.
    """
    root = _archive(tmp_path)
    before = verifier.verify_archive(root)
    assert before["checks"]["checksum_closure"] is True

    _, committed = verifier.commit_verification(root)
    stored = json.loads((root / verifier.VERIFICATION_NAME).read_text(encoding="utf-8"))
    assert stored == before == committed
    assert verifier.verify_archive(root) == stored


# -- what cannot be a verdict at all ---------------------------------------- #


@pytest.mark.parametrize("missing", sorted(verifier.PRODUCER_ARTIFACTS))
def test_a_missing_artifact_forms_no_verification_record(
    tmp_path: Path, missing: str
) -> None:
    """Fail closed with nothing written: the record's own fields cannot be filled."""
    root = _archive(tmp_path)
    (root / missing).unlink()
    with pytest.raises(verifier.ExecutionVerificationError):
        verifier.verify_archive(root)
    with pytest.raises(verifier.ExecutionVerificationError):
        verifier.commit_verification(root)
    assert not (root / verifier.VERIFICATION_NAME).exists()


def test_an_unreadable_artifact_forms_no_verification_record(tmp_path: Path) -> None:
    root = _archive(tmp_path)
    (root / "result.json").write_bytes(b"{not json")
    with pytest.raises(verifier.ExecutionVerificationError):
        verifier.verify_archive(root)


def test_an_archive_that_can_point_outside_itself_is_refused(tmp_path: Path) -> None:
    root = _archive(tmp_path)
    (root / "elsewhere.json").symlink_to(tmp_path)
    with pytest.raises(verifier.ExecutionVerificationError):
        verifier.verify_archive(root)


# -- § C3.9's trap ---------------------------------------------------------- #


def test_the_verifier_is_plumbing_only() -> None:
    assert path_partition.classify("scripts/tools/verify_h2_execution.py") == (
        "plumbing_only"
    )


def test_the_verifier_restates_no_ruler_fact() -> None:
    """It moves no axis, so any rule typed out here could change unnoticed."""
    source = (_TOOLS / "verify_h2_execution.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    forbidden = (
        '"H2_INPUT_MUTATED_DURING_MEASUREMENT"',
        '"H2_CAPTURE_PERTURBS_POLICY"',
        '"H2_PACKET_INVALID"',
        '"H2_MEASUREMENT_EXECUTION_INVALID"',
        '"measurement_pass"',
        '"input_mutated"',
        '"build_failed"',
        '"extension_load_failed"',
        '"diagnostic_complete"',
        '"exactly_once_measurement"',
        '"non_qualifying_diagnostic"',
        '"bound_input_unchanged"',
        '"execution_complete"',
        '"00_capture_off"',
    )
    restated = [name for name in forbidden if name in body]
    assert restated == [], f"the verifier restates ruler facts: {restated}"


def test_the_verifier_reads_no_host_state() -> None:
    """`verification_host_inputs_used: false` is a claim, so it is scanned for.

    Over the syntax tree rather than the text: the module has to be free to
    *explain* which host input it refuses and where that line sits, and a
    substring scan would then be measuring the prose instead of the code.
    """
    tree = ast.parse((_TOOLS / "verify_h2_execution.py").read_text(encoding="utf-8"))
    referenced: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {
        "cwd",
        "environ",
        "getcwd",
        "getenv",
        "gethostname",
        "getuid",
        "platform",
        "socket",
        "subprocess",
        # the ruler's one function that hashes the verifying host's checkout
        "execution_semantics_projection",
    }
    assert not forbidden & referenced, f"the verifier consults host state: {
        sorted(forbidden & referenced)
    }"
    producers = sorted(name for name in imported if name.startswith("run_h2_"))
    assert producers == [], f"the verifier imports a producer: {producers}"
