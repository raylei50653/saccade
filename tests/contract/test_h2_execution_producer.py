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
import functools
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_h2_measure_archives as corpus  # noqa: E402
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


def _runtime_inputs(*, binds_build: bool = True) -> dict[str, Any]:
    """The coordinate form is what an execution that never built has to record."""
    return {
        "manifest_digest": _fake("manifest"),
        "manifest_schema": (
            "h2_runtime_input_manifest_v1"
            if binds_build
            else "h2_runtime_input_coordinate_v1"
        ),
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
        "diagnostics": {
            "behavior_probe": {
                "digest": _fake("probe"),
                "role": "recorded_diagnostic_observation_selects_nothing",
                "schema": "h2_behavior_probe_result_v1",
                "state": "computed",
            }
        },
        "runtime_inputs": _runtime_inputs(),
        "source_audit": {"head": "a" * 40, "tree": "b" * 40},
    }
    defaults.update(overrides)
    return producer.StageEvidence(**defaults)


def _stopped_stages(stage: str, **overrides: Any) -> producer.StageEvidence:
    """A binding from an execution that stopped: partial artifacts, no load, no probe."""
    defaults: dict[str, Any] = {
        "failed_stage": stage,
        "runtime_inputs": _runtime_inputs(binds_build=stage != "build"),
        "source_audit": {"head": "a" * 40, "tree": "b" * 40},
    }
    if stage != "build":
        defaults["build_artifacts"] = _complete_stages().build_artifacts
    defaults.update(overrides)
    return producer.StageEvidence(**defaults)


class _FixedStages:
    """A stage surface whose monitor record is taken when the producer closes it."""

    def __init__(
        self,
        evidence: producer.StageEvidence,
        monitor: dict[str, Any] | None = None,
    ) -> None:
        self._evidence = evidence
        self._monitor = monitor if monitor is not None else _monitor()
        self.closed_after: list[str] = []

    def run(self) -> producer.StageEvidence:
        return self._evidence

    def close(self) -> dict[str, Any]:
        self.closed_after.append("close")
        return self._monitor


@functools.lru_cache(maxsize=1)
def _canonical_environment() -> tuple[tuple[str, str], ...]:
    spec = run_spec_module.build_run_spec()
    return tuple(sorted(producer.canonical_launch_environment(spec).items()))


def _launch_projection(
    *, environment: dict[str, Any] | None = None, run_ids: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """What the launch boundary received — matching the RunSpec unless a test says otherwise."""
    received = dict(_canonical_environment()) if environment is None else environment
    return {
        "observations": [
            {"environment": dict(received), "run_id": run_id}
            for run_id in (run_ids if run_ids is not None else producer.run_ids())
        ],
        "resolved_run_spec_digest": run_spec_module.build_run_spec()[
            "resolved_run_spec_digest"
        ],
    }


class _FixedRuns:
    """Four completed runs and a decided observation, or whatever a test asks for."""

    def __init__(
        self,
        *,
        states: dict[str, str] | None = None,
        projection: dict[str, Any] | None = None,
    ) -> None:
        self._states = states or {}
        self._projection = projection

    def run(self, stages: producer.StageEvidence) -> producer.RunEvidence:
        ordered = [
            {"artifact_digest": _fake(run_id), "run_id": run_id, "state": "completed"}
            for run_id in producer.run_ids()
        ]
        predicates = {
            name: {"reasons": [], "state": self._states.get(name, "pass")}
            for name in producer.predicate_names()
            if name != producer.PROJECTION_PREDICATE
        }
        return producer.RunEvidence(
            ordered_runs=ordered,
            predicate_results=predicates,
            launch_projection=self._projection or _launch_projection(),
        )


def _execution(
    spec: dict[str, Any],
    *,
    stages: producer.StageEvidence | None = None,
    runs: Any = None,
    authority: str = "exactly_once_measurement",
    monitor: dict[str, Any] | None = None,
) -> producer.Execution:
    """A diagnostic carries no authorization digest; the schema requires null."""
    measurement = authority == "exactly_once_measurement"
    return producer.Execution(
        execution_id=EXECUTION_ID,
        authority=authority,
        stages=_FixedStages(
            stages if stages is not None else _complete_stages(), monitor
        ),
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
        stages=_stopped_stages("build"),
        monitor=_monitor(changed=1, drained=False),
    ).produce(root)

    assert result["result"] == "input_mutated"
    assert result["terminal"] == "H2_INPUT_MUTATED_DURING_MEASUREMENT"
    assert result["predicate_results"]["execution_complete"]["state"] == "fail"
    assert [run["state"] for run in result["ordered_runs"]] == ["not_run"] * 4

    record = verifier.verify_archive(root)
    assert record["valid"] is True, record["reasons"]
    binding = json.loads((root / "runtime_binding.json").read_text(encoding="utf-8"))
    assert binding["failed_stage"] == "build"


def test_the_monitor_is_closed_after_the_runs_not_after_the_stages(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """The binding's `input_monitor` covers the whole execution, runs included.

    The ruler reads a recorded change as outranking every other finding, so the
    window it describes decides which findings can exist. Closing the monitor
    when the stages ended would leave the four measurement runs — the part most
    able to move a bound input — outside the window the archive attests.
    """
    order: list[str] = []

    class _OrderedRuns(_FixedRuns):
        def run(
            self, stages: producer.StageEvidence
        ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
            order.append("runs")
            return super().run(stages)

    class _OrderedStages(_FixedStages):
        def close(self) -> dict[str, Any]:
            order.append("close")
            return super().close()

    execution = producer.Execution(
        execution_id=EXECUTION_ID,
        authority="exactly_once_measurement",
        stages=_OrderedStages(_complete_stages()),
        runs=_OrderedRuns(),
        authorization_binding_digest=_fake("authorization"),
        run_spec=run_spec,
    )
    execution.produce(tmp_path / "archive")
    assert order == ["runs", "close"]


def test_a_stage_failure_still_closes_the_monitor(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """No run started, but the window still has to be closed and reported.

    A stage failure is the case where the monitor's record is the *only*
    execution-wide evidence there is: the runs decided nothing, so a change it
    recorded is the finding that outranks the stage.
    """
    stages = _FixedStages(_stopped_stages("build"), _monitor(changed=2))
    execution = producer.Execution(
        execution_id=EXECUTION_ID,
        authority="exactly_once_measurement",
        stages=stages,
        runs=_FixedRuns(),
        authorization_binding_digest=_fake("authorization"),
        run_spec=run_spec,
    )
    result = execution.produce(tmp_path / "archive")
    assert stages.closed_after == ["close"]
    assert result["result"] == "input_mutated"


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
    evidence = runs.run(_complete_stages())
    expected = partition.select_successor_result(
        {
            **evidence.predicate_results,
            producer.PROJECTION_PREDICATE: {"reasons": [], "state": "pass"},
        },
        authority="exactly_once_measurement",
        phase="a",
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


@pytest.mark.parametrize(
    "case",
    [
        {},
        {"stages": _stopped_stages("build")},
        {"stages": _stopped_stages("build"), "monitor": _monitor(changed=1)},
        {
            "stages": _stopped_stages(
                "identity_run", extension_load=_complete_stages().extension_load
            )
        },
        {"runs": _FixedRuns(states={"packets_valid": "fail"})},
        {
            "authority": "non_qualifying_diagnostic",
            "runs": _FixedRuns(states={"packets_valid": "fail"}),
        },
    ],
    ids=[
        "measurement_pass",
        "stage_failure",
        "moved_input_outranks_the_stage",
        "late_stage_failure",
        "run_decided_failure",
        "diagnostic",
    ],
)
def test_naming_the_cause_moves_nothing_but_the_cause(
    tmp_path: Path, run_spec: dict[str, Any], case: dict[str, Any]
) -> None:
    """The second selection may refine terminal 4's cause and nothing else.

    The producer asks the ruler twice: once unnamed, to learn whether a cause may
    be named at all, then again carrying it. Replaying both selections over the
    observation the archive recorded pins the property that separates a refined
    verdict from a different one — the terminal never moves, and the result token
    moves only from `unclassified_execution_failure` to a cause that maps to the
    same terminal 4. Anything else would mean the producer's own sequencing, not
    the ruler, decided where the execution landed.
    """
    result = _execution(run_spec, **case).produce(tmp_path / "archive")
    observation = result["predicate_results"]
    authority = result["authority"]

    unnamed = partition.select_successor_result(
        observation, authority=authority, phase="a"
    )
    assert (result["result"], result["terminal"]) == (
        partition.select_successor_result(
            observation,
            authority=authority,
            phase="a",
            execution_result=(
                result["result"]
                if unnamed.terminal == partition.EXECUTION_INVALID_TERMINAL
                else None
            ),
        ).result,
        result["terminal"],
    )
    assert result["terminal"] == unnamed.terminal
    if unnamed.terminal != partition.EXECUTION_INVALID_TERMINAL:
        assert result["result"] == unnamed.result
    else:
        assert unnamed.result == "unclassified_execution_failure"
        assert (
            partition.RESULT_TO_TERMINAL[result["result"]]
            == partition.EXECUTION_INVALID_TERMINAL
        )


def test_an_observation_that_drifts_under_the_reader_is_read_once(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    """The observation is frozen when the runs hand it over, not when it is read.

    Two selections over a live structure would answer about two executions, and
    the archive would record the second — a verdict nothing ever authorised. A
    `Runs` implementation that returns a mapping still under its own control is
    the ordinary way that happens, so the drift is made explicit here: this
    observation reports `pass` to its first reader and `fail` to every one after.
    The archive records the first, because there is only one reader.
    """

    class _Drifting(Mapping):
        def __init__(self, base: Mapping[str, Any]) -> None:
            self._base = dict(base)
            self.reads = 0

        def __getitem__(self, key: str) -> Any:
            record = dict(self._base[key])
            if key == "packets_valid":
                self.reads += 1
                record["state"] = "pass" if self.reads == 1 else "fail"
            return record

        def __iter__(self) -> Any:
            return iter(self._base)

        def __len__(self) -> int:
            return len(self._base)

    class _DriftingRuns(_FixedRuns):
        def run(self, stages: producer.StageEvidence) -> producer.RunEvidence:
            evidence = super().run(stages)
            return producer.RunEvidence(
                ordered_runs=evidence.ordered_runs,
                predicate_results=_Drifting(evidence.predicate_results),
                launch_projection=evidence.launch_projection,
            )

    root = tmp_path / "archive"
    result = _execution(run_spec, runs=_DriftingRuns()).produce(root)
    assert result["predicate_results"]["packets_valid"]["state"] == "pass"
    assert result["result"] == "measurement_pass"
    assert result["terminal"] is None
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


def test_the_command_line_refuses_to_execute_without_the_launch_arguments(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """W4b bound the driver, so the refusal moved: it is now about what is missing."""
    code = producer.main(
        [
            "--execution-id",
            EXECUTION_ID,
            "--authority",
            producer.DIAGNOSTIC_AUTHORITY,
            "--archive",
            str(tmp_path / "archive"),
        ]
    )
    assert code == 2
    assert "--run-root and --selected-base are required" in capsys.readouterr().err
    assert not (tmp_path / "archive").exists()


def test_the_diagnostic_authority_is_the_rulers_own_token() -> None:
    """Named once in the producer, and never allowed to drift from the ruler's list."""
    assert producer.DIAGNOSTIC_AUTHORITY is partition.DIAGNOSTIC_AUTHORITY


# -- canonical-corpus admission (W5c) -------------------------------------- #


def test_successor_discovery_uses_a_family_specific_anchor(tmp_path: Path) -> None:
    h0 = tmp_path / "h0_phase_a_existing"
    h0.mkdir()
    (h0 / "result.json").write_text("{}\n", encoding="utf-8")
    (h0 / "verification.json").write_text("{}\n", encoding="utf-8")
    successor = tmp_path / "arbitrary-successor-name"
    successor.mkdir()
    (successor / corpus.SUCCESSOR_DISCOVERY_NAME).write_text("{}\n", encoding="utf-8")

    assert corpus.archive_roots(tmp_path) == [successor]
    assert corpus._is_successor_archive(h0) is False
    assert corpus._is_successor_archive(successor) is True


def test_a_closed_successor_measurement_is_admitted(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root = tmp_path / "a-name-is-navigation-not-identity"
    _execution(run_spec).produce(root)
    _, verification = verifier.commit_verification(root)
    assert verification["valid"] is True

    attempts = corpus.check_corpus([root])
    assert len(attempts) == 1
    attempt = attempts[0]
    assert isinstance(attempt, corpus.SuccessorAttempt)
    assert attempt.result["authority"] == partition.MEASUREMENT_AUTHORITY
    assert attempt.verification == verification


def test_a_green_diagnostic_is_valid_but_corpus_refused(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root = tmp_path / "diagnostic"
    _execution(run_spec, authority=partition.DIAGNOSTIC_AUTHORITY).produce(root)
    _, verification = verifier.commit_verification(root)
    assert verification["valid"] is True
    assert corpus.archive_roots(tmp_path) == [root]

    with pytest.raises(corpus.CorpusError, match="diagnostic is never canonical"):
        corpus.check_corpus([root])


def test_an_unclosed_successor_measurement_is_not_canonical(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root = tmp_path / "producer-only"
    _execution(run_spec).produce(root)
    assert verifier.verify_archive(root)["valid"] is True

    with pytest.raises(corpus.CorpusError, match="is not closed"):
        corpus.check_corpus([root])


def test_an_independently_invalid_successor_archive_is_not_canonical(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root = tmp_path / "invalid"
    _execution(run_spec).produce(root)
    binding = json.loads((root / "runtime_binding.json").read_text(encoding="utf-8"))
    binding["execution_id"] = "someone-else"
    (root / "runtime_binding.json").write_text(
        json.dumps(binding, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _, verification = verifier.commit_verification(root)
    assert verification["valid"] is False

    with pytest.raises(corpus.CorpusError, match="not independently valid"):
        corpus.check_corpus([root])


def test_the_command_line_refuses_to_bind_a_measurement(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The authority whose content is a spent grant may not be claimed without one.

    This entry point has no way to receive an exactly-once authorization, so it
    refuses the authority outright rather than emitting an archive that claims a
    grant was consumed and leaves the field it would have been recorded in empty.
    """
    other = [
        name for name in partition.AUTHORITIES if name != producer.DIAGNOSTIC_AUTHORITY
    ]
    code = producer.main(
        [
            "--execution-id",
            EXECUTION_ID,
            "--authority",
            other[0],
            "--archive",
            str(tmp_path / "archive"),
            "--run-root",
            str(tmp_path / "runs"),
            "--selected-base",
            "a" * 40,
        ]
    )
    assert code == 2
    assert "exactly-once authorization" in capsys.readouterr().err
    assert not (tmp_path / "archive").exists()


def test_the_run_tree_may_not_live_inside_the_verified_archive(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The archive root is flat by contract, and a run tree inside it is a subdirectory.

    The verifier refuses a root holding a subdirectory before any schema is read,
    so a run tree written there would make every archive unformable — and its
    bytes would be ones the closure never names.
    """
    code = producer.main(
        [
            "--execution-id",
            EXECUTION_ID,
            "--authority",
            producer.DIAGNOSTIC_AUTHORITY,
            "--archive",
            str(tmp_path / "archive"),
            "--run-root",
            str(tmp_path / "archive" / "runs"),
            "--selected-base",
            "a" * 40,
        ]
    )
    assert code == 2
    assert "--run-root must be outside --archive" in capsys.readouterr().err
    assert not (tmp_path / "archive").exists()


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
