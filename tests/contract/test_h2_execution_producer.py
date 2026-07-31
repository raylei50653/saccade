"""The successor producer: what it must emit, and what it must never claim.

Review Correction 5 gives the producer three artifacts and one prohibition, and
both halves are pinned here:

  * **it emits three files and closes nothing** — no `verification.json`, no
    `checksums.sha256`. A producer that wrote either would be closing an archive
    over its own claim about itself;
  * **it decides no verdict** — `result` and `terminal` come back from
    `select_successor_result` and are transcribed. The only thing it may name
    that the predicates cannot derive is which cause put the execution in
    terminal 4;
  * **a stage failure is an observation, not an error** — the execution stops,
    every ordered run stays `not_run`, and the archive says so;
  * **the archive it produces verifies** — the end-to-end property that W3
    landing first is what makes checkable at all. The verifier was written
    against the frozen contracts, not against this module's output (§ 5.3), so
    an agreement between them is evidence rather than a tautology.

The RunSpec here comes from the real resolver rather than a synthetic fixture:
the producer runs on the host whose checkout *is* the execution's declared
content set, which is exactly the asymmetry that lets the verifier refuse to
look at a checkout at all.
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

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_path_partition as path_partition  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import run_h2_execution as producer  # noqa: E402
import verify_h2_execution as verifier  # noqa: E402

EXECUTION_ID = "h2exec-20260731T120000Z"


def _fake(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


_EXTENSION = _fake("tracking_extension")
_PLUGIN = _fake("tensorrt_scan_plugin")


@pytest.fixture(scope="module")
def run_spec() -> dict[str, Any]:
    """The real resolver's RunSpec, issued from the frozen authoring profile."""
    return run_spec_module.build_run_spec()


def _monitor(*, changed: int = 0, drained: bool = True) -> dict[str, Any]:
    return {
        "changed_count": changed,
        "final_drain_clean": drained,
        "started_before_binding": True,
    }


def _runtime_inputs() -> dict[str, Any]:
    return {
        "manifest_digest": _fake("manifest"),
        "manifest_schema": "h2_runtime_input_manifest_v1",
        "members": [
            {
                "length": 512,
                "path": "datasets/MOT17/train/MOT17-04-SDP/seqinfo.ini",
                "role": "measurement_fixture_input",
                "sha256": _fake("seqinfo"),
            }
        ],
    }


def _complete_stages(**overrides: Any) -> producer.StageEvidence:
    defaults: dict[str, Any] = {
        "build_artifacts": [
            {
                "length": 4096,
                "path": "build/h2_layer_p/saccade_tracking_ext.so",
                "role": "tracking_extension",
                "sha256": _EXTENSION,
            },
            {
                "length": 8192,
                "path": "build/h2_layer_p/libsaccade_scan_plugin.so",
                "role": "tensorrt_scan_plugin",
                "sha256": _PLUGIN,
            },
        ],
        "extension_load": {
            "length": 4096,
            "loaded_path": "/opt/saccade/build/h2_layer_p/saccade_tracking_ext.so",
            "sha256": _EXTENSION,
        },
        "identity_probe": producer.identity_probe_record(
            {"digest": _fake("probe")}, build_artifact_digest=_EXTENSION
        ),
        "input_monitor": _monitor(),
        "runtime_inputs": _runtime_inputs(),
        "source_audit": {"head": "a" * 40, "tree": "b" * 40},
    }
    defaults.update(overrides)
    return producer.StageEvidence(**defaults)


def _stopped_stages(stage: str, **overrides: Any) -> producer.StageEvidence:
    """A binding from an execution that stopped: partial artifacts, no load, no probe."""
    defaults: dict[str, Any] = {
        "failed_stage": stage,
        "input_monitor": _monitor(),
        "runtime_inputs": _runtime_inputs(),
        "source_audit": {"head": "a" * 40, "tree": "b" * 40},
    }
    if stage != "build":
        defaults["build_artifacts"] = _complete_stages().build_artifacts
    defaults.update(overrides)
    return producer.StageEvidence(**defaults)


class _FixedStages:
    def __init__(self, evidence: producer.StageEvidence) -> None:
        self._evidence = evidence

    def run(self) -> producer.StageEvidence:
        return self._evidence


class _FixedRuns:
    """Four completed runs and a decided observation, or whatever a test asks for."""

    def __init__(self, *, states: dict[str, str] | None = None) -> None:
        self._states = states or {}

    def run(
        self, stages: producer.StageEvidence
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        ordered = [
            {"artifact_digest": _fake(run_id), "run_id": run_id, "state": "completed"}
            for run_id in producer.run_ids()
        ]
        predicates = {
            name: {"reasons": [], "state": self._states.get(name, "pass")}
            for name in producer.predicate_names()
        }
        return ordered, predicates, None


def _execution(
    spec: dict[str, Any],
    *,
    stages: producer.StageEvidence | None = None,
    runs: Any = None,
    authority: str = "exactly_once_measurement",
) -> producer.Execution:
    """A diagnostic carries no authorization digest; the schema requires null."""
    measurement = authority == "exactly_once_measurement"
    return producer.Execution(
        execution_id=EXECUTION_ID,
        authority=authority,
        stages=_FixedStages(stages if stages is not None else _complete_stages()),
        runs=runs if runs is not None else _FixedRuns(),
        authorization_binding_digest=_fake("authorization") if measurement else None,
        run_spec=spec,
    )


# -- the end-to-end property W3-before-W4 buys ------------------------------ #


def test_a_produced_measurement_archive_verifies(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """The producer and the independent verifier agree, without having met.

    Both were written against the frozen contracts — the verifier first, on
    synthetic fixtures — so this is the first evidence that the contracts
    describe an archive something can actually emit.
    """
    root = tmp_path / "archive"
    result = _execution(run_spec).produce(root)
    assert result["result"] == "measurement_pass"
    assert result["terminal"] is None

    record = verifier.verify_archive(root)
    assert record["valid"] is True, record["reasons"]
    assert record["execution_id"] == EXECUTION_ID


def test_a_produced_stage_failure_archive_verifies(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """A truthful negative is archivable, which is the whole point of `failed_stage`."""
    root = tmp_path / "archive"
    result = _execution(run_spec, stages=_stopped_stages("build")).produce(root)
    assert result["result"] == "build_failed"
    assert result["terminal"] == partition.EXECUTION_INVALID_TERMINAL
    assert [run["state"] for run in result["ordered_runs"]] == ["not_run"] * 4

    record = verifier.verify_archive(root)
    assert record["valid"] is True, record["reasons"]


@pytest.mark.parametrize("stage", ["build_binding", "extension_load", "identity_run"])
def test_every_stage_failure_produces_a_verifiable_archive(
    tmp_path: Path, run_spec: dict[str, Any], stage: str
) -> None:
    root = tmp_path / "archive"
    stages = _stopped_stages(stage)
    if stage == "identity_run":
        stages = _stopped_stages(
            stage, extension_load=_complete_stages().extension_load
        )
    result = _execution(run_spec, stages=stages).produce(root)
    assert result["terminal"] == partition.EXECUTION_INVALID_TERMINAL
    record = verifier.verify_archive(root)
    assert record["valid"] is True, record["reasons"]


def test_a_produced_diagnostic_archive_verifies(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """A diagnostic resolves to `diagnostic_complete` and names no terminal."""
    root = tmp_path / "archive"
    result = _execution(
        run_spec,
        authority="non_qualifying_diagnostic",
        runs=_FixedRuns(states={"packets_valid": "fail"}),
    ).produce(root)
    assert result["result"] == partition.DIAGNOSTIC_RESULT
    assert result["terminal"] is None
    assert verifier.verify_archive(root)["valid"] is True


def test_a_moved_input_outranks_the_stage_that_failed(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """Terminal 1 wins, and the stage stays as subordinate evidence.

    The monitor is stage-independent — it starts before the binding by contract —
    so a stage failure does not hide what it recorded. Reporting only the stage
    would emit `build_failed` beside a recorded mutation, which the ruler refuses
    and the verifier rejects; this case is why the producer decides the monitor's
    predicate too, and then does not name a terminal-4 cause it cannot claim.
    """
    root = tmp_path / "archive"
    result = _execution(
        run_spec,
        stages=_stopped_stages(
            "build", input_monitor=_monitor(changed=1, drained=False)
        ),
    ).produce(root)

    assert result["result"] == "input_mutated"
    assert result["terminal"] == "H2_INPUT_MUTATED_DURING_MEASUREMENT"
    assert result["predicate_results"]["execution_complete"]["state"] == "fail"
    assert [run["state"] for run in result["ordered_runs"]] == ["not_run"] * 4

    record = verifier.verify_archive(root)
    assert record["valid"] is True, record["reasons"]
    binding = json.loads((root / "runtime_binding.json").read_text(encoding="utf-8"))
    assert binding["failed_stage"] == "build"


# -- what the producer may not do ------------------------------------------- #


def test_the_producer_writes_three_files_and_closes_nothing(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root = tmp_path / "archive"
    _execution(run_spec).produce(root)
    assert sorted(item.name for item in root.iterdir()) == sorted(
        verifier.PRODUCER_ARTIFACTS
    )
    assert not (root / verifier.VERIFICATION_NAME).exists()
    assert not (root / verifier.CHECKSUMS_NAME).exists()


def test_the_producer_refuses_to_overwrite_another_execution(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root = tmp_path / "archive"
    _execution(run_spec).produce(root)
    with pytest.raises(producer.ProducerError, match="not empty"):
        _execution(run_spec).produce(root)


def test_the_producer_transcribes_the_rulers_selection(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """Not a copy of the ruler's answer — the ruler's answer."""
    runs = _FixedRuns(states={"capture_off_on_equal": "fail"})
    ordered, predicates, _ = runs.run(_complete_stages())
    expected = partition.select_successor_result(
        predicates, authority="exactly_once_measurement", phase="a"
    )
    result = _execution(run_spec, runs=runs).produce(tmp_path / "archive")
    assert (result["result"], result["terminal"]) == (
        expected.result,
        expected.terminal,
    )
    assert result["result"] == "capture_perturbs_policy"


def test_an_incoherent_binding_is_refused_before_anything_is_written(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """A build that failed cannot also have loaded an extension."""
    root = tmp_path / "archive"
    stages = _stopped_stages("build", extension_load=_complete_stages().extension_load)
    with pytest.raises(producer.ProducerError, match="own contract"):
        _execution(run_spec, stages=stages).produce(root)
    assert not root.exists() or sorted(root.iterdir()) == []


def test_the_producer_names_a_terminal_4_cause_only_under_a_measurement(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """The named cause is a measurement's to give; a diagnostic names nothing."""
    root = tmp_path / "archive"
    result = _execution(
        run_spec,
        authority="non_qualifying_diagnostic",
        stages=_stopped_stages("extension_load"),
    ).produce(root)
    assert result["result"] == partition.DIAGNOSTIC_RESULT
    assert verifier.verify_archive(root)["valid"] is True


# -- the shapes it projects from the retained modules ----------------------- #


def test_the_declared_content_split_comes_from_the_run_spec(
    run_spec: dict[str, Any],
) -> None:
    surfaces, abi = producer.declared_content(
        run_spec["execution_semantics_projection"]
    )
    declared = {
        str(member["path"]): member
        for member in run_spec["execution_semantics_projection"]["members"]
    }
    assert [member["path"] for member in surfaces] == sorted(
        producer.executed_surface_paths()
    )
    assert abi["path"] == producer.capture_abi_path()
    for member in [*surfaces, abi]:
        assert member["sha256"] == declared[member["path"]]["sha256"]
        assert member["length"] == declared[member["path"]]["length"]


def test_a_content_set_missing_an_executed_surface_is_refused() -> None:
    with pytest.raises(producer.ProducerError, match="executed surface"):
        producer.declared_content({"members": []})


def test_the_build_artifact_roles_come_from_the_runtime_input_manifest() -> None:
    section = {
        "files": [
            {
                "length": 4096,
                "resolved_path": (_REPO / "build/x/saccade_tracking_ext.so").as_posix(),
                "role": "tracking_extension",
                "sha256": _EXTENSION,
            },
            {
                "length": 8192,
                "resolved_path": (
                    _REPO / "build/x/libsaccade_scan_plugin.so"
                ).as_posix(),
                "role": "tensorrt_scan_plugin",
                "sha256": _PLUGIN,
            },
        ]
    }
    artifacts = producer.build_artifacts_from_manifest(section)
    assert [item["role"] for item in artifacts] == [
        "tensorrt_scan_plugin",
        "tracking_extension",
    ]
    assert artifacts[1]["path"] == "build/x/saccade_tracking_ext.so"
    with pytest.raises(producer.ProducerError, match="both roles"):
        producer.build_artifacts_from_manifest({"files": section["files"][:1]})


def test_the_extension_load_record_is_the_load_probes_own_witness() -> None:
    witness = {
        "extension_length": 4096,
        "extension_path": "/opt/saccade/build/h2_layer_p/saccade_tracking_ext.so",
        "extension_sha256": _EXTENSION,
    }
    assert producer.extension_load_from_witness(witness) == {
        "length": 4096,
        "loaded_path": witness["extension_path"],
        "sha256": _EXTENSION,
    }


def test_the_identity_probe_is_recorded_as_an_observation() -> None:
    record = producer.identity_probe_record(
        {"digest": _fake("probe")}, build_artifact_digest=_EXTENSION
    )
    assert record["role"] == "recorded_observation_not_equivalence_or_gate"
    assert record["state"] == "computed"
    assert record["build_artifact_digest"] == _EXTENSION


# -- § C3.9's trap ---------------------------------------------------------- #


def test_the_producer_is_plumbing_only() -> None:
    assert path_partition.classify("scripts/tools/run_h2_execution.py") == (
        "plumbing_only"
    )


def test_the_producer_restates_no_ruler_fact() -> None:
    """A producer that typed out a verdict would be a second answer to one question."""
    source = (_TOOLS / "run_h2_execution.py").read_text(encoding="utf-8")
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
        '"00_capture_off"',
        '"MOT17-04-SDP"',
    )
    restated = [name for name in forbidden if name in body]
    assert restated == [], f"the producer restates ruler facts: {restated}"


def test_the_producer_never_writes_the_verdict_or_the_closure() -> None:
    """The prohibition is structural, so it is checked over the syntax tree."""
    tree = ast.parse((_TOOLS / "run_h2_execution.py").read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert verifier.VERIFICATION_NAME not in literals
    assert verifier.CHECKSUMS_NAME not in literals
    referenced = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "write_checksum_inventory" not in referenced
    assert "commit_verification" not in referenced
    assert "verify_archive" not in referenced


def test_the_command_line_refuses_to_execute_without_a_bound_driver(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """W4 landed the producer, not an execution: binding a build is a separate step."""
    code = producer.main(
        [
            "--execution-id",
            EXECUTION_ID,
            "--authority",
            "non_qualifying_diagnostic",
            "--archive",
            str(tmp_path / "archive"),
        ]
    )
    assert code == 2
    assert "no execution driver is bound" in capsys.readouterr().err


def test_the_command_line_can_issue_a_run_spec_without_executing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = producer.main(
        [
            "--execution-id",
            EXECUTION_ID,
            "--authority",
            "non_qualifying_diagnostic",
            "--archive",
            str(tmp_path / "archive"),
            "--emit-run-spec-only",
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == run_spec_module.RUN_SPEC_SCHEMA
    assert not (tmp_path / "archive").exists()
