#!/usr/bin/env python3
"""Bind the retained H2 modules to the successor producer's two protocols.

`run_h2_execution` owns the sequencing and asks the ruler for the verdict; it
takes its evidence from `Stages` and `Runs`. This module implements those two
against the real Layer-P stages and the real Layer-M runner, and it is written
to make four things impossible rather than merely absent.

**It holds no verdict.** Nothing here imports the ruler, names a terminal or a
result token, or calls the selector, the verifier or the closure. A stage that
failed is reported as the stage that failed; what that means is decided in
`h2_terminal_partition` and transcribed by the producer. A test monkeypatches
the selector to raise and runs both adapters to completion, because an absent
import is weaker evidence than an oracle that would have been consulted.

**It does not guess a run's state.** Every ordered run is recorded from a
witness: `completed` and `failed` both require a process-start witness and differ
only in whether the runner reported completion, and `not_run` means no such
witness exists because the ordered loop never reached that run. Nothing is
inferred from the shape the schema wants — a run plan that comes back doubled,
reordered or outside the declared set raises instead, because padding it into a
legal shape is the one repair that would look exactly like evidence.

**The monitor is a capability, not a claim.** `started_before_binding` is a
`const: true` in the frozen schema — a value a driver could simply write. Here
it is obtainable only from a `MonitorSession` that observed its own start ahead
of the execution's binding mark in one shared `ExecutionOrder`, and only from
the session that the binding announced itself to. A session that started late,
was never bound, or is not the bound one raises instead of reporting.

**Transcription is not repair.** The evidence returned to the producer is
projected from what the retained modules actually returned. Nothing is defaulted
in, no failure is turned into a success, no exception type is mapped to a
result, and the ordered runs keep the runner's order.
"""
# status: stable

from __future__ import annotations

import hashlib
import itertools
import sys
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_measurement_evidence as evidence  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
import run_h2_execution as producer  # noqa: E402
import run_h2_layer_p as layer_p_module  # noqa: E402
import run_h2_measurement as layer_m  # noqa: E402

# The two Layer-P stages the successor binding has no vocabulary for. They are
# refusals to start, not execution outcomes: `failed_stage` does not name them,
# and Layer-P calls them `blocked` — retryable, no terminal, no budget consumed.
UNSTARTED_STAGES: tuple[str, ...] = ("retry_admissibility", "preflight")


class DriverError(RuntimeError):
    """The driver refuses to hand the producer evidence it cannot vouch for."""


# -- the observed order of this execution ----------------------------------- #


@dataclass
class ExecutionOrder:
    """Monotonic marks for the events whose *order* the contract depends on.

    Only ordering is recorded, never wall-clock time: the question the binding
    poses is which came first, and a timestamp would invite the answer to be
    reconstructed after the fact from something that merely correlates.
    """

    _next: Callable[[], int] = field(
        default_factory=lambda: itertools.count(1).__next__
    )
    marks: dict[str, int] = field(default_factory=dict)

    def mark(self, event: str) -> int:
        if event in self.marks:
            raise DriverError(f"{event} happened once already at {self.marks[event]}")
        ordinal = self._next()
        self.marks[event] = ordinal
        return ordinal

    def when(self, event: str) -> int | None:
        return self.marks.get(event)


# -- the monitor capability -------------------------------------------------- #


@dataclass
class MonitorSession:
    """A started bound-input monitor, and the only source of its evidence.

    Construct through `start_monitor`. The session is what the binding must
    announce itself to, so a driver cannot start one monitor and report from
    another: the binding mark is namespaced by `session_id`, and a session that
    was never bound has nothing to project `started_before_binding` from.
    """

    session_id: str
    started_at: int
    monitor: Any
    order: ExecutionOrder
    changed_count: int = 0
    last_drain_clean: bool = True
    bound_at: int | None = None
    closed: bool = False

    @property
    def _binding_event(self) -> str:
        return f"inputs_bound:{self.session_id}"

    def bind(self) -> int:
        """Announce that the execution is now binding its inputs to this session."""
        if self.closed:
            raise DriverError("this monitor session is already closed")
        self.bound_at = self.order.mark(self._binding_event)
        return self.bound_at

    def drain(self) -> int:
        """A checkpoint drain. Events are accumulated, never discarded."""
        if self.closed:
            raise DriverError("this monitor session is already closed")
        events = list(self.monitor.drain())
        self.changed_count += len(events)
        self.last_drain_clean = not events
        return len(events)

    def observed(self) -> int:
        """Every event the monitor has seen, including drains this session did not make.

        The child wait loop drains the monitor itself while a run is alive, and
        that is where a mutation *during the measurement* is seen — so counting
        only this session's own drains would report zero for exactly the case the
        monitor exists to catch. `BoundInputMonitor.drain` appends to a durable
        `history`, which is therefore the session's record rather than a second
        one: owning the monitor's lifetime is not the same as owning what it saw.
        """
        history = getattr(self.monitor, "history", None)
        if history is None:
            return self.changed_count
        return max(self.changed_count, len(history))

    def finalize(self) -> dict[str, Any]:
        """Drain once more, close, and project the binding's `input_monitor`.

        The binding mark is read from the shared order rather than from this
        object, so a binding that happened before this session started is found
        and refused — rather than looking like a session that was never bound.
        """
        bound_at = self.order.when(self._binding_event)
        if bound_at is None:
            raise DriverError(
                "the execution never announced its binding to this monitor session, "
                "so nothing here can say the monitor started first"
            )
        if not self.started_at < bound_at:
            raise DriverError(
                f"the monitor started at {self.started_at}, after the binding at "
                f"{bound_at}: a monitor cannot witness what preceded it"
            )
        self.drain()
        record = {
            "changed_count": self.observed(),
            "final_drain_clean": self.last_drain_clean,
            "started_before_binding": True,
        }
        self.monitor.close()
        self.closed = True
        return record

    def abandon(self) -> None:
        """Release the monitor without producing a record. Safe to call twice."""
        if not self.closed:
            self.monitor.close()
            self.closed = True


def start_monitor(
    bound_paths: Iterable[Path],
    *,
    order: ExecutionOrder,
    monitor_factory: Callable[..., Any] | None = None,
    session_id: str | None = None,
) -> MonitorSession:
    """Start watching the bound inputs, and record that this happened first."""
    factory = monitor_factory or _default_monitor_factory()
    identifier = session_id or uuid.uuid4().hex
    monitor = factory(sorted(set(bound_paths)))
    return MonitorSession(
        session_id=identifier,
        started_at=order.mark(f"monitor_started:{identifier}"),
        monitor=monitor,
        order=order,
    )


def _default_monitor_factory() -> Callable[..., Any]:
    import run_h0_phase_a as h0_controller

    return h0_controller.BoundInputMonitor


# -- Stages: the six retained Layer-P stages -------------------------------- #


@dataclass
class LayerPStages:
    """Run Layer P's retained stages and project what they returned.

    The stage order is Layer P's own. What this adds is the successor's frame:
    the two stages that only ever block are refused rather than mistranslated
    into `failed_stage`, and the runtime inputs recorded for a build that failed
    are the coordinate form, because there were no build artifacts to hash.
    """

    layer_p: Any
    session: MonitorSession

    def run(self) -> producer.StageEvidence:
        if self.session.closed:
            raise DriverError("this execution's monitor session is already closed")
        self.session.bind()

        published: dict[str, Any] | None = None
        witness: dict[str, Any] | None = None
        probe: dict[str, Any] | None = None
        manifest: Mapping[str, Any] | None = None
        failed_stage: str | None = None
        try:
            self.layer_p.retry_admissibility()
            published = self.layer_p.preflight()
            self.layer_p.build()
            self.layer_p.build_binding()
            witness = self.layer_p.extension_load()
            probe, manifest, _, _ = self.layer_p.identity_run(published)
        except layer_p_module.Blocked as blocked:
            if blocked.coordinate in UNSTARTED_STAGES:
                raise DriverError(
                    f"Layer P blocked at {blocked.coordinate}: the execution never "
                    "started, and the successor binding has no stage to name for it"
                ) from blocked
            failed_stage = blocked.coordinate

        if manifest is None:
            manifest = self._manifest(failed_stage)
        return producer.StageEvidence(
            source_audit=self._source_audit(),
            runtime_inputs=producer.runtime_input_binding(manifest),
            failed_stage=failed_stage,
            build_artifacts=(
                producer.build_artifacts_from_manifest(manifest["build_artifacts"])
                if "build_artifacts" in manifest
                else None
            ),
            extension_load=(
                producer.extension_load_from_witness(witness)
                if witness is not None
                else None
            ),
            identity_probe=(
                producer.identity_probe_record(
                    probe,
                    build_artifact_digest=manifest["build_artifacts"]["files"][0][
                        "sha256"
                    ],
                )
                if probe is not None
                else None
            ),
        )

    def close(self) -> Mapping[str, Any]:
        """Close the one session this execution started, after the runs."""
        return self.session.finalize()

    def abandon(self) -> None:
        """Release the session without a record, for the exits that write nothing."""
        self.session.abandon()

    def _manifest(self, failed_stage: str | None) -> Mapping[str, Any]:
        """The inputs this execution bound, in the form its outcome allows.

        A build that failed produced no artifacts, so the manifest it can record
        is the build-independent coordinate. Every later stage ran against a
        build, and must hash it.
        """
        try:
            return runtime_inputs.build_manifest(
                build_dir=None if failed_stage == "build" else self.layer_p.build_dir
            )
        except (runtime_inputs.RuntimeInputError, OSError) as exc:
            raise DriverError(
                f"the bound runtime inputs are unreadable: {exc}"
            ) from exc

    def _source_audit(self) -> dict[str, str]:
        return {
            "head": layer_p_module._git("rev-parse", "HEAD"),
            "tree": layer_p_module._git("rev-parse", "HEAD^{tree}"),
        }


# -- Runs: the four ordered measurement runs -------------------------------- #


@dataclass
class MeasurementRuns:
    """Launch Layer M's four ordered runs and transcribe what came back.

    The launches are `run_h2_measurement.launch_ordered_runs` — the same call
    the legacy controller makes, so the two paths cannot drift in what they run
    or in what order. What this does not do is call `execute_controller`, which
    would consume a legacy authorization and select a legacy terminal.
    """

    root: Path
    bundle: Any
    document: Mapping[str, Any]
    session: MonitorSession
    started: float
    clock: Callable[[], float]
    inherited_environment: Mapping[str, str] | None = None
    launch_child: Any = None

    def run(
        self, stages: producer.StageEvidence
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Launch the runs, and return evidence whether or not they all finished.

        The failures that end a measurement from *inside* the run phase — a
        mutation the wait loop saw, a child that exited nonzero, the deadline —
        are outcomes, not errors. They arrive as exceptions because that is how
        the runner stops, and letting them propagate would mean the one thing an
        execution most needs to record is the one thing it cannot: the producer
        would never close the monitor, never build a binding, never write an
        archive. The partial sets are read from accumulators this caller owns,
        because the runner's return value is not delivered when it raises.

        Orchestration defects still propagate. A run reported twice, a run
        outside the plan, unreadable evidence: those say the bookkeeping cannot
        be trusted, and there is nothing truthful to archive about them.
        """
        completed: set[str] = set()
        launched: set[str] = set()
        outcome: str | None = None
        try:
            layer_m.launch_ordered_runs(
                self.root,
                bundle=self.bundle,
                document=self.document,
                inherited_environment=self.inherited_environment,
                monitor=self.session.monitor,
                started=self.started,
                clock=self.clock,
                completed=completed,
                started_runs=launched,
                **({"launch_child": self.launch_child} if self.launch_child else {}),
            )
        except layer_m.ReachedRunFailure as failure:
            outcome = f"{failure.run_id}: {failure.detail}"
        except TimeoutError as expired:
            outcome = f"the measurement deadline expired: {expired}"
        except _mutation_errors() as moved:
            # The wait loop drained the monitor and stopped the child. The event
            # is in the monitor's durable history, so the session still reports
            # it, and the ruler — not this module — decides what it outranks.
            outcome = f"a bound input moved while a run was live: {moved}"

        replay = layer_m.replay_surviving_evidence(self.root)
        ordered = ordered_run_records(self.root, completed=completed, launched=launched)
        observed = _predicates(
            replay, completed=completed, session=self.session, outcome=outcome
        )
        return ordered, observed, None


def _mutation_errors() -> tuple[type[BaseException], ...]:
    import run_h0_phase_a as h0_controller

    return (h0_controller.DriftError,)


def ordered_run_records(
    root: Path,
    *,
    completed: Iterable[str],
    launched: Iterable[str],
) -> list[dict[str, Any]]:
    """One record per declared run, in the declared order, from its own evidence.

    Three states, each decided by a witness rather than inferred:

    * `completed` — a process-start witness, and the runner reported it finished;
    * `failed` — a process-start witness, and it did not finish;
    * `not_run` — no process-start witness: the ordered loop never reached it.

    `launched` is required, and that is the point. Without it this function
    cannot tell a run that started and failed from one that never started, and
    picking either would be a fabrication: `failed` claims a launch that did not
    happen, `not_run` claims the execution stopped earlier than it did. A default
    would let any other caller mint `failed` claims with no witness behind them,
    which is the same defect the real adapter was just fixed for.

    Fail-closed on the run plan itself: a run outside the plan, or one reported
    twice, is a statement about the bookkeeping and not about the measurement, so
    it raises rather than being trimmed into a legal shape.
    """
    finished = list(completed)
    started = set(launched)
    unknown = sorted((set(finished) | started) - set(evidence.RUN_IDS))
    if unknown:
        raise DriverError(f"the runner reported runs outside the plan: {unknown}")
    if len(finished) != len(set(finished)):
        raise DriverError("the runner reported the same run more than once")
    done = set(finished)
    if not done <= started:
        raise DriverError(
            f"the runner reported {sorted(done - started)} as completed without a "
            "process-start witness"
        )
    unstarted = set(evidence.RUN_IDS) - started
    return [
        {
            "artifact_digest": run_artifact_digest(root, run_id)
            if run_id in done
            else None,
            "run_id": run_id,
            "state": (
                "completed"
                if run_id in done
                else "not_run"
                if run_id in unstarted
                else "failed"
            ),
        }
        for run_id in evidence.RUN_IDS
    ]


def run_artifact_digest(root: Path, run_id: str) -> str:
    """Digest one run's own policy inventory — the record of what it produced."""
    directory = evidence.run_dir(root, layer_m.SEQUENCE, run_id)
    try:
        inventory = evidence.load_document(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            schema=evidence.POLICY_INVENTORY_SCHEMA,
        )
    except (evidence.EvidenceError, OSError) as exc:
        raise DriverError(f"{run_id}: completed run has no usable inventory: {exc}")
    return hashlib.sha256(evidence.canonical_json_bytes(inventory)).hexdigest()


def _predicates(
    replay: Any,
    *,
    completed: Iterable[str],
    session: MonitorSession,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Project what this execution observed into the successor's predicate records.

    Fail-closed on coverage rather than on shape: the predicate *set* belongs to
    the frozen result schema, so a predicate this driver cannot decide raises
    here instead of being defaulted to `pass`.

    The mutation predicate reads `session.observed()`, not the session's own
    drain count. A mutation during a live run is seen by the child wait loop, and
    a driver that counted only its own drains would report the execution clean in
    exactly the case the monitor exists to catch.
    """
    every_run = set(completed) == set(evidence.RUN_IDS)
    decided = bool(getattr(replay, "evidence_present", False))
    mutated = session.observed() > 0 or not session.last_drain_clean
    observed = {
        "bound_input_unchanged": _state(not mutated),
        "capture_off_on_equal": _state(replay.capture_equal) if decided else _error(),
        "packets_valid": _state(replay.packets_valid) if decided else _error(),
        "execution_complete": _state(
            bool(replay.complete)
            and every_run
            and not replay.errors
            and outcome is None
        ),
    }
    if outcome is not None:
        observed["execution_complete"]["reasons"] = [outcome]
    undecidable = sorted(set(producer.predicate_names()) - set(observed))
    if undecidable:
        raise DriverError(
            f"this execution observed nothing that decides {undecidable}: the "
            "resolved RunSpec carries no reference to compare against, because "
            "Review Correction 5 retired the published probe and the Layer-P "
            "certificate as gates. Reporting a state here would invent evidence"
        )
    return observed


def _state(passed: bool) -> dict[str, Any]:
    return {"reasons": [], "state": "pass" if passed else "fail"}


def _error() -> dict[str, Any]:
    return {
        "reasons": ["the surviving evidence does not decide this predicate"],
        "state": "error",
    }


def bound_paths(*, build_dir: Path) -> tuple[Path, ...]:
    """Every path this execution binds, for the monitor to watch from the start."""
    import h2_behavioral_identity as identity

    discovered = runtime_inputs.discover_bound_paths(build_dir=build_dir)
    watched: set[Path] = set(runtime_inputs.watch_paths(discovered))
    watched.update(
        REPO_ROOT / path
        for path_class in ("decision_relevant", "identity_semantics")
        for path in identity.tracked_files_for_class(path_class)
        if (REPO_ROOT / path).is_file()
    )
    return tuple(sorted(watched))
