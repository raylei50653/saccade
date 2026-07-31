"""The real driver: faithful transcription, and a monitor that cannot be claimed.

W4 bound the producer's `Stages` / `Runs` protocols to fakes, which proves the
producer's control flow and nothing about wiring. This file pins the properties
the real adapters must have before an execution is ever run:

  * **no verdict** — not merely no import of the ruler, but no consultation of
    it: the selector is monkeypatched to raise and both adapters still complete;
  * **no synthesised run** — a run plan that comes back short, doubled or
    outside the declared set raises, so `not_run` stays the producer's to emit;
  * **the monitor is a capability** — `started_before_binding` is obtainable
    only from a session that observed its own start before the binding mark, and
    only from the session the binding announced itself to;
  * **transcription is not repair** — sentinel values handed back by the retained
    modules reach the producer unchanged, and a missing one raises.

None of this needs a build, a GPU or an execution: the retained modules are
represented by fakes whose returns are the thing under test.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_execution_driver as driver  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import run_h2_execution as producer  # noqa: E402
import run_h2_layer_p as layer_p_module  # noqa: E402

DRIVER_PATH = "scripts/tools/h2_execution_driver.py"


class _FakeMonitor:
    """Hands back queued event batches, then empties, and records its close."""

    def __init__(self, batches: list[list[str]] | None = None) -> None:
        self._batches = list(batches or [])
        self.closed = False

    def drain(self) -> list[str]:
        return self._batches.pop(0) if self._batches else []

    def close(self) -> None:
        self.closed = True


def _session(
    order: driver.ExecutionOrder | None = None,
    *,
    batches: list[list[str]] | None = None,
    session_id: str = "s1",
) -> driver.MonitorSession:
    return driver.start_monitor(
        [],
        order=order or driver.ExecutionOrder(),
        monitor_factory=lambda _paths: _FakeMonitor(batches),
        session_id=session_id,
    )


# -- the monitor capability -------------------------------------------------- #


def test_a_monitor_that_started_first_can_report_that_it_did() -> None:
    session = _session()
    session.bind()
    record = session.finalize()
    assert record == {
        "changed_count": 0,
        "final_drain_clean": True,
        "started_before_binding": True,
    }
    assert session.monitor.closed is True


def test_a_monitor_started_after_the_binding_refuses_to_report() -> None:
    """The swapped order, which is the whole point of recording an order."""
    order = driver.ExecutionOrder()
    order.mark("inputs_bound:s1")
    session = _session(order)
    with pytest.raises(driver.DriverError, match="cannot witness what preceded it"):
        session.finalize()


def test_a_session_the_binding_never_announced_itself_to_refuses() -> None:
    """Start monitor A, bind against B: A has nothing to project from."""
    order = driver.ExecutionOrder()
    started = _session(order, session_id="a")
    other = _session(order, session_id="b")
    other.bind()
    with pytest.raises(driver.DriverError, match="never announced its binding"):
        started.finalize()


def test_a_binding_that_fails_still_leaves_the_same_session_to_report_from() -> None:
    """The case W3's mutation-over-stage precedence rests on in a real driver.

    A stage failed, so no run decided anything; the monitor's record is then the
    only execution-wide evidence there is, and it must survive the failure.
    """
    session = _session(batches=[["datasets/MOT17/train/x"]])
    session.bind()
    record = session.finalize()
    assert record["changed_count"] == 1
    assert record["final_drain_clean"] is False
    assert record["started_before_binding"] is True


def test_a_change_seen_at_a_checkpoint_is_accumulated_not_replaced() -> None:
    """Draining is not sampling: an event seen mid-execution stays counted."""
    session = _session(batches=[["moved"], []])
    session.bind()
    session.drain()
    record = session.finalize()
    assert record["changed_count"] == 1
    assert record["final_drain_clean"] is True


def test_an_unclean_final_drain_is_reported_as_unclean() -> None:
    session = _session(batches=[[], ["moved-late"]])
    session.bind()
    session.drain()
    record = session.finalize()
    assert (record["changed_count"], record["final_drain_clean"]) == (1, False)


def test_one_execution_binds_its_inputs_once() -> None:
    session = _session()
    session.bind()
    with pytest.raises(driver.DriverError, match="happened once already"):
        session.bind()


def test_a_closed_session_reports_nothing_further() -> None:
    session = _session()
    session.bind()
    session.finalize()
    for call in (session.drain, session.bind):
        with pytest.raises(driver.DriverError, match="already closed"):
            call()


# -- the run plan: fail closed, never padded -------------------------------- #


def _run_ids() -> tuple[str, ...]:
    import h2_measurement_evidence as evidence

    return tuple(evidence.RUN_IDS)


def test_a_run_outside_the_declared_plan_raises(tmp_path: Path) -> None:
    with pytest.raises(driver.DriverError, match="outside the plan"):
        driver.ordered_run_records(tmp_path, completed=[*_run_ids(), "99_extra"])


def test_a_run_reported_twice_raises(tmp_path: Path) -> None:
    ids = _run_ids()
    with pytest.raises(driver.DriverError, match="more than once"):
        driver.ordered_run_records(tmp_path, completed=[*ids, ids[0]])


def test_a_short_run_plan_is_recorded_as_failed_never_as_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`not_run` is a claim about an execution that stopped before the runs.

    These runs started. A driver that reported the missing ones as `not_run`
    would be moving the execution into the producer's stage-failure shape from
    the outside, which is exactly the synthesis this adapter must not do.
    """
    monkeypatch.setattr(driver, "run_artifact_digest", lambda root, run_id: "a" * 64)
    ids = _run_ids()
    records = driver.ordered_run_records(tmp_path, completed=ids[:2])
    assert [record["run_id"] for record in records] == list(ids)
    assert [record["state"] for record in records] == [
        "completed",
        "completed",
        "failed",
        "failed",
    ]
    assert "not_run" not in {record["state"] for record in records}
    assert [record["artifact_digest"] for record in records][2:] == [None, None]


def test_the_recorded_order_is_the_declared_order_not_the_completion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "run_artifact_digest", lambda root, run_id: "b" * 64)
    ids = _run_ids()
    records = driver.ordered_run_records(tmp_path, completed=list(reversed(ids)))
    assert [record["run_id"] for record in records] == list(ids)


# -- it holds no verdict ----------------------------------------------------- #


def _driver_tree() -> ast.Module:
    return ast.parse((_REPO / DRIVER_PATH).read_text(encoding="utf-8"))


def test_the_driver_never_imports_the_ruler() -> None:
    imported: set[str] = set()
    for node in ast.walk(_driver_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "h2_terminal_partition" not in imported


def test_the_driver_names_no_terminal_and_no_result_token() -> None:
    """Every successor token, from the ruler itself, must be absent from the code.

    Docstrings are exempt: this module has to be able to explain which decisions
    are not its own, and naming them in prose is how it does that.
    """
    tokens = {
        *partition.RESULT_TO_TERMINAL,
        *(terminal.name for terminal in partition.TERMINALS),
        partition.DIAGNOSTIC_RESULT,
    }
    found = {
        node.value
        for node in ast.walk(_driver_tree())
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in tokens
    }
    docstrings = {
        node.body[0].value.value
        for node in ast.walk(_driver_tree())
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    assert found - docstrings == set()


def test_the_driver_calls_no_selector_verifier_or_closure() -> None:
    forbidden = {
        "select_successor_result",
        "select_terminal",
        "verify_archive",
        "commit_verification",
        "write_checksum_inventory",
        "execute_controller",
    }
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(_driver_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert called & forbidden == set()


def test_neither_adapter_consults_the_ruler_even_when_it_could(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An oracle that would have been asked is stronger evidence than a missing import.

    Both adapters run to completion with the selector rigged to raise on sight,
    which no amount of indirection through a helper could survive.
    """

    def _refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("the driver asked the ruler for a verdict")

    monkeypatch.setattr(partition, "select_successor_result", _refuse)
    monkeypatch.setattr(partition, "select_terminal", _refuse)
    monkeypatch.setattr(driver, "run_artifact_digest", lambda root, run_id: "c" * 64)

    session = _session()
    session.bind()
    assert driver.ordered_run_records(tmp_path, completed=_run_ids())
    assert session.finalize()["started_before_binding"] is True


# -- transcription is not repair -------------------------------------------- #


class _FakeLayerP:
    """The six stage methods, each returning a sentinel or blocking on demand."""

    def __init__(self, *, blocks_at: str | None = None) -> None:
        self.blocks_at = blocks_at
        self.build_dir = Path("/nonexistent/build/h2_layer_p")
        self.calls: list[str] = []

    def _stage(self, name: str) -> None:
        self.calls.append(name)
        if self.blocks_at == name:
            raise layer_p_module.Blocked(name, "fake stage refusal")

    def retry_admissibility(self) -> None:
        self._stage("retry_admissibility")

    def preflight(self) -> dict[str, Any]:
        self._stage("preflight")
        return {"coordinate": {}, "probe": {"digest": "d" * 64}}

    def build(self) -> None:
        self._stage("build")

    def build_binding(self) -> None:
        self._stage("build_binding")

    def extension_load(self) -> dict[str, Any]:
        self._stage("extension_load")
        return {
            "extension_length": 4096,
            "extension_path": "/opt/saccade/build/h2_layer_p/saccade_tracking_ext.so",
            "extension_sha256": "e" * 64,
        }

    def identity_run(self, published: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        self._stage("identity_run")
        raise AssertionError("this fake never reaches a real identity run")


def test_a_stage_that_only_ever_blocks_is_refused_not_mistranslated() -> None:
    """`retry_admissibility` and `preflight` are refusals to start.

    The successor `failed_stage` vocabulary does not name them, and Layer P calls
    them `blocked` — retryable, no terminal, no budget consumed. Recording one as
    an execution failure would convert a retryable refusal into a spent attempt.
    """
    for stage in driver.UNSTARTED_STAGES:
        session = _session(session_id=stage)
        stages = driver.LayerPStages(
            layer_p=_FakeLayerP(blocks_at=stage), session=session
        )
        with pytest.raises(driver.DriverError, match="never started"):
            stages.run()


def test_the_stages_run_in_layer_ps_own_order() -> None:
    layer_p = _FakeLayerP(blocks_at="extension_load")
    session = _session()
    stages = driver.LayerPStages(layer_p=layer_p, session=session)
    with pytest.raises(Exception):
        stages.run()
    assert layer_p.calls == [
        "retry_admissibility",
        "preflight",
        "build",
        "build_binding",
        "extension_load",
    ]


def test_the_binding_is_announced_before_any_stage_runs() -> None:
    """The monitor's window has to open before the first stage, not before the last."""
    layer_p = _FakeLayerP(blocks_at="retry_admissibility")
    session = _session()
    stages = driver.LayerPStages(layer_p=layer_p, session=session)
    with pytest.raises(driver.DriverError):
        stages.run()
    assert session.bound_at is not None
    assert session.started_at < session.bound_at


def test_a_build_that_failed_records_the_coordinate_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No build artifacts were produced, so none may be claimed."""
    seen: dict[str, Any] = {}

    def _manifest(*, build_dir: Any, **_: Any) -> dict[str, Any]:
        seen["build_dir"] = build_dir
        return _coordinate_manifest()

    monkeypatch.setattr(driver.runtime_inputs, "build_manifest", _manifest)
    session = _session()
    stages = driver.LayerPStages(
        layer_p=_FakeLayerP(blocks_at="build"), session=session
    )
    record = stages.run()
    assert seen["build_dir"] is None
    assert record.failed_stage == "build"
    assert record.build_artifacts is None
    assert (
        record.runtime_inputs["manifest_schema"]
        == driver.runtime_inputs.COORDINATE_SCHEMA
    )


def test_a_later_stage_failure_still_hashes_the_build_it_ran_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def _manifest(*, build_dir: Any, **_: Any) -> dict[str, Any]:
        seen["build_dir"] = build_dir
        return _coordinate_manifest()

    monkeypatch.setattr(driver.runtime_inputs, "build_manifest", _manifest)
    layer_p = _FakeLayerP(blocks_at="build_binding")
    stages = driver.LayerPStages(layer_p=layer_p, session=_session())
    stages.run()
    assert seen["build_dir"] == layer_p.build_dir


def test_unreadable_runtime_inputs_raise_rather_than_resolve_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**_: Any) -> dict[str, Any]:
        raise driver.runtime_inputs.RuntimeInputError("datasets are absent")

    monkeypatch.setattr(driver.runtime_inputs, "build_manifest", _raise)
    stages = driver.LayerPStages(
        layer_p=_FakeLayerP(blocks_at="build"), session=_session()
    )
    with pytest.raises(driver.DriverError, match="unreadable"):
        stages.run()


def _coordinate_manifest() -> dict[str, Any]:
    """A manifest in the coordinate form, with one member per declared section."""
    sections = {
        name: {
            "digest": "f" * 64,
            "file_count": 1,
            "files": [
                {
                    "configured_path": f"/repo/{name}.bin",
                    "coordinate": f"{name}.bin",
                    "length": 8,
                    "resolved_path": f"/repo/{name}.bin",
                    "role": name,
                    "sha256": "0" * 64,
                    "symlink_chain": [],
                }
            ],
        }
        for name in driver.runtime_inputs.COORDINATE_SECTIONS
    }
    return {
        **sections,
        "coordinate_digest": "1" * 64,
        "data_root": "/repo/datasets",
        "policy_preset": driver.runtime_inputs.POLICY_PRESET_REL,
        "schema": driver.runtime_inputs.COORDINATE_SCHEMA,
    }


# -- classification ---------------------------------------------------------- #


def test_the_driver_is_plumbing_only() -> None:
    """It composes the retained modules and holds no rule, so it moves no ruler."""
    assert path_partition.classify(DRIVER_PATH) == "plumbing_only"


def test_the_producers_protocols_are_what_this_driver_implements() -> None:
    """A structural check, so a protocol change cannot silently orphan the driver."""
    for name in ("run", "close"):
        assert hasattr(driver.LayerPStages, name), name
    assert hasattr(driver.MeasurementRuns, "run")
    assert set(producer.StageEvidence.__dataclass_fields__) == {
        "source_audit",
        "runtime_inputs",
        "failed_stage",
        "build_artifacts",
        "extension_load",
        "identity_probe",
    }
