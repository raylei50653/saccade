"""An open research instance holds the online surface shut, and says so before the push.

* the committed lock is well-formed and its history is append-only;
* `RESEARCH_OPEN` fails closed when a frozen axis moves — recomputed from
  source, or re-published under a new coordinate;
* `sealed` is refused under drift and `voided` is not — enforcement stops at
  `RESEARCH_CLOSED`, so sealing is the last moment the freeze can be checked;
* opening does not inherit consumer-binding admissibility, so a closed study
  going stale never gates the next one;
* `release` governs instance lifetime, not editability;
* the guard cannot be edited by the freeze it guards — the lock file and its
  tool are outside every axis an instance can freeze.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_path_partition as partition  # noqa: E402
import research_lock as lock  # noqa: E402

_FROZEN_SURFACE = "d" * 64
_FROZEN_SEMANTICS = "s" * 64
_FROZEN_PROBE = "p" * 64

_PUBLISHED = {
    "coordinate": {
        "decision_surface": _FROZEN_SURFACE,
        "identity_semantics": _FROZEN_SEMANTICS,
        "implementation": "i" * 64,
        "environment": "e" * 64,
        "runtime_inputs": "r" * 64,
    },
    "probe": {"digest": _FROZEN_PROBE},
}
_MEASURED = {
    "decision_surface": _FROZEN_SURFACE,
    "identity_semantics": _FROZEN_SEMANTICS,
}


def _instance(**overrides: Any) -> dict[str, Any]:
    instance = {
        "instance_id": "probe-instance",
        "declaration": "docs/modules/semantic/research/example_declaration.md",
        "evidence_root": "docs/modules/semantic/research/evidence/example",
        "opened_at": "2026-07-26T00:00:00+00:00",
        "frozen_axes": ["decision_surface", "identity_semantics"],
        "frozen": {
            "coordinate": {
                "decision_surface": _FROZEN_SURFACE,
                "identity_semantics": _FROZEN_SEMANTICS,
            },
            "probe": _FROZEN_PROBE,
        },
        "disposition": None,
        "registry_pointer": None,
    }
    instance.update(overrides)
    return instance


def _lock(state: str, instance: dict[str, Any] | None) -> dict[str, Any]:
    history = []
    if state != lock.ONLINE_OPEN:
        history.append(
            {
                "from": lock.ONLINE_OPEN,
                "to": state,
                "at": "2026-07-26T00:00:00+00:00",
                "instance_id": instance["instance_id"] if instance else None,
                "note": "fixture",
            }
        )
    return {
        "schema": lock.LOCK_SCHEMA,
        "state": state,
        "instance": instance,
        "history": history,
    }


# ── the committed lock ───────────────────────────────────────────────────────


def test_the_committed_lock_is_valid() -> None:
    lock.load_lock()


def test_a_missing_lock_is_not_an_open_online_surface(tmp_path: Path) -> None:
    with pytest.raises(lock.ResearchLockError, match="deleted guard"):
        lock.load_lock(tmp_path / "absent.json")


def test_the_committed_lock_enforces_its_own_freeze() -> None:
    """The real check: whatever state the repository is in, it must be consistent."""
    committed = lock.load_lock()
    if committed["state"] != lock.RESEARCH_OPEN:
        pytest.skip(f"no research instance is open ({committed['state']})")
    published = lock.staleness.load_published(_REPO / lock.staleness.PUBLISHED_REL)
    assert lock.verify(committed, published) == []


# ── freeze enforcement ───────────────────────────────────────────────────────


def test_an_unmoved_coordinate_passes() -> None:
    assert (
        lock.verify(
            _lock(lock.RESEARCH_OPEN, _instance()), _PUBLISHED, measured=_MEASURED
        )
        == []
    )


@pytest.mark.parametrize("axis", ["decision_surface", "identity_semantics"])
def test_a_moved_frozen_axis_fails_closed(axis: str) -> None:
    measured = {**_MEASURED, axis: "0" * 64}
    failures = lock.verify(
        _lock(lock.RESEARCH_OPEN, _instance()), _PUBLISHED, measured=measured
    )
    assert len(failures) == 1
    assert axis in failures[0]
    assert "close the instance as voided" in failures[0]


def test_the_drift_hint_offers_only_the_exit_that_is_still_legal() -> None:
    """Drift has already made `sealed` illegal, and `release` never gated editing."""
    assert "voided" in lock.DRIFT_EXIT_HINT
    assert "sealed" not in lock.DRIFT_EXIT_HINT
    assert "before editing online" not in lock.DRIFT_EXIT_HINT


def test_a_republish_is_an_online_move_too() -> None:
    published = {
        **_PUBLISHED,
        "coordinate": {**_PUBLISHED["coordinate"], "decision_surface": "0" * 64},
    }
    failures = lock.verify(
        _lock(lock.RESEARCH_OPEN, _instance()), published, measured=_MEASURED
    )
    assert any("re-published" in message for message in failures)


def test_a_moved_probe_fails_closed() -> None:
    published = {**_PUBLISHED, "probe": {"digest": "0" * 64}}
    failures = lock.verify(
        _lock(lock.RESEARCH_OPEN, _instance()), published, measured=_MEASURED
    )
    assert any("probe moved" in message for message in failures)


def test_an_axis_outside_the_frozen_set_may_move() -> None:
    """`implementation` drift is re-attestation, not invalidation — unless frozen."""
    published = {
        **_PUBLISHED,
        "coordinate": {**_PUBLISHED["coordinate"], "implementation": "0" * 64},
    }
    assert (
        lock.verify(
            _lock(lock.RESEARCH_OPEN, _instance()), published, measured=_MEASURED
        )
        == []
    )


def test_an_instance_may_freeze_implementation_as_well() -> None:
    instance = _instance(
        frozen_axes=["decision_surface", "identity_semantics", "implementation"],
        frozen={
            "coordinate": {
                "decision_surface": _FROZEN_SURFACE,
                "identity_semantics": _FROZEN_SEMANTICS,
                "implementation": "i" * 64,
            },
            "probe": _FROZEN_PROBE,
        },
    )
    published = {
        **_PUBLISHED,
        "coordinate": {**_PUBLISHED["coordinate"], "implementation": "0" * 64},
    }
    failures = lock.verify(
        _lock(lock.RESEARCH_OPEN, instance),
        published,
        measured={**_MEASURED, "implementation": "i" * 64},
    )
    assert any("implementation" in message for message in failures)


@pytest.mark.parametrize("state", [lock.ONLINE_OPEN, lock.RESEARCH_CLOSED])
def test_only_an_open_instance_holds_the_surface_shut(state: str) -> None:
    instance = None
    if state == lock.RESEARCH_CLOSED:
        instance = _instance(disposition="sealed")
    moved = {**_PUBLISHED, "probe": {"digest": "0" * 64}}
    assert lock.verify(_lock(state, instance), moved, measured={}) == []


def test_a_closed_instance_keeps_its_version_binding() -> None:
    closed = _lock(lock.RESEARCH_CLOSED, _instance(disposition="voided"))
    lock.validate_lock(closed)
    assert closed["instance"]["frozen"]["probe"] == _FROZEN_PROBE


# ── transitions ──────────────────────────────────────────────────────────────


def test_the_full_cycle_returns_to_online_open() -> None:
    current = _lock(lock.ONLINE_OPEN, None)
    current = lock.transition(
        current, "open", instance=_instance(), note="open", at="t1"
    )
    assert current["state"] == lock.RESEARCH_OPEN
    current = lock.transition(
        current,
        "close",
        instance=_instance(disposition="sealed"),
        note="close",
        at="t2",
    )
    assert current["state"] == lock.RESEARCH_CLOSED
    current = lock.transition(
        current, "release", instance=None, note="release", at="t3"
    )
    assert current["state"] == lock.ONLINE_OPEN
    assert [record["to"] for record in current["history"]] == [
        lock.RESEARCH_OPEN,
        lock.RESEARCH_CLOSED,
        lock.ONLINE_OPEN,
    ]


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (lock.ONLINE_OPEN, "close"),
        (lock.ONLINE_OPEN, "release"),
        (lock.RESEARCH_OPEN, "open"),
        (lock.RESEARCH_OPEN, "release"),
        (lock.RESEARCH_CLOSED, "open"),
        (lock.RESEARCH_CLOSED, "close"),
    ],
)
def test_transitions_outside_the_graph_are_refused(state: str, action: str) -> None:
    instance = None if state == lock.ONLINE_OPEN else _instance()
    if state == lock.RESEARCH_CLOSED:
        instance = _instance(disposition="sealed")
    with pytest.raises(lock.ResearchLockError, match="illegal transition"):
        lock.transition(_lock(state, instance), action, instance=None, note="n", at="t")


def test_history_is_append_only() -> None:
    before = _lock(lock.ONLINE_OPEN, None)
    before["history"] = [
        {
            "from": lock.RESEARCH_CLOSED,
            "to": lock.ONLINE_OPEN,
            "at": "t0",
            "instance_id": "earlier",
            "note": "earlier cycle",
        }
    ]
    after = lock.transition(before, "open", instance=_instance(), note="n", at="t1")
    assert after["history"][0] == before["history"][0]
    assert len(after["history"]) == 2


def test_a_released_lock_keeps_no_ghost_freeze() -> None:
    ghost = _lock(lock.ONLINE_OPEN, None)
    ghost["instance"] = _instance()
    with pytest.raises(lock.ResearchLockError, match="ghost freeze"):
        lock.validate_lock(ghost)


def test_state_must_match_the_last_transition() -> None:
    forged = _lock(lock.RESEARCH_OPEN, _instance())
    forged["state"] = lock.ONLINE_OPEN
    forged["instance"] = None
    with pytest.raises(lock.ResearchLockError, match="does not match the last"):
        lock.validate_lock(forged)


def test_a_non_recomputable_axis_cannot_be_frozen() -> None:
    """`environment` and `runtime_inputs` are excluded by construction."""
    for axis in ("environment", "runtime_inputs"):
        assert axis not in lock.LOCKABLE_AXES
    with pytest.raises(lock.ResearchLockError, match="non-recomputable"):
        lock.validate_lock(
            _lock(lock.RESEARCH_OPEN, _instance(frozen_axes=["environment"]))
        )


def test_a_closed_instance_names_a_disposition() -> None:
    with pytest.raises(lock.ResearchLockError, match="disposition"):
        lock.validate_lock(_lock(lock.RESEARCH_CLOSED, _instance()))


# ── sealing is the last moment the freeze can be checked ─────────────────────


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Drive the CLI against a throwaway lock file and a fixed publication."""
    path = tmp_path / "research_lock_v1.json"
    monkeypatch.setattr(lock, "lock_path", lambda: path)

    def run(argv: list[str], *, published: dict[str, Any] | None = None) -> int:
        monkeypatch.setattr(lock, "_published", lambda: published or _PUBLISHED)
        return lock.main(argv)

    def seed(state: str, instance: dict[str, Any] | None) -> None:
        lock.write_lock(_lock(state, instance), path)

    def read() -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    run.seed = seed  # type: ignore[attr-defined]
    run.read = read  # type: ignore[attr-defined]
    return run


def _source_backed_instance(**overrides: Any) -> dict[str, Any]:
    """An instance whose frozen coordinate matches what source actually hashes to."""
    measured = lock.recompute(lock.DEFAULT_FROZEN_AXES)
    return _instance(
        frozen={"coordinate": measured, "probe": _FROZEN_PROBE}, **overrides
    )


def test_source_drift_cannot_be_sealed(cli) -> None:
    """The frozen digests are fixtures, so live source cannot match them."""
    cli.seed(lock.RESEARCH_OPEN, _instance())
    assert cli(["close", "--disposition", "sealed", "--note", "n"]) == 1
    assert cli.read()["state"] == lock.RESEARCH_OPEN


def test_published_drift_cannot_be_sealed(cli) -> None:
    """Source is clean; the publication moved under the open instance."""
    instance = _source_backed_instance()
    cli.seed(lock.RESEARCH_OPEN, instance)
    moved = {
        "coordinate": {**_PUBLISHED["coordinate"], **instance["frozen"]["coordinate"]},
        "probe": {"digest": "0" * 64},
    }
    assert (
        cli(["close", "--disposition", "sealed", "--note", "n"], published=moved) == 1
    )
    assert cli.read()["state"] == lock.RESEARCH_OPEN


def test_drift_may_always_be_voided(cli) -> None:
    """The exit that claims nothing is the one a drifted instance keeps."""
    cli.seed(lock.RESEARCH_OPEN, _instance())
    assert cli(["close", "--disposition", "voided", "--note", "abandoned"]) == 0
    closed = cli.read()
    assert closed["state"] == lock.RESEARCH_CLOSED
    assert closed["instance"]["disposition"] == "voided"


def test_an_undrifted_instance_seals(cli) -> None:
    instance = _source_backed_instance()
    published = {
        "coordinate": {**_PUBLISHED["coordinate"], **instance["frozen"]["coordinate"]},
        "probe": {"digest": _FROZEN_PROBE},
    }
    cli.seed(lock.RESEARCH_OPEN, instance)
    assert (
        cli(["close", "--disposition", "sealed", "--note", "n"], published=published)
        == 0
    )
    assert cli.read()["instance"]["disposition"] == "sealed"


# ── opening does not inherit historical binding admissibility ────────────────


def test_open_does_not_consult_consumer_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale closed study is the expected end of its life, not the next one's gate.

    The precondition is proven not to read the bindings file at all: if it did,
    this would raise instead of returning no failures.
    """

    def explode(*_: Any, **__: Any) -> None:
        raise AssertionError("publication_precondition read the consumer bindings")

    monkeypatch.setattr(lock.staleness, "load_bindings", explode)
    published = lock.staleness.load_published(_REPO / lock.staleness.PUBLISHED_REL)
    assert lock.publication_precondition(published) == []


def test_open_refuses_a_publication_that_drifted_from_source() -> None:
    published = {
        "coordinate": {**_PUBLISHED["coordinate"]},
        "probe": {"digest": _FROZEN_PROBE},
    }
    failures = lock.publication_precondition(published)
    assert failures
    assert any("not republished" in message for message in failures)


def test_a_stale_binding_does_not_block_a_new_instance(cli) -> None:
    """End to end: the CLI opens while a historical binding would fail the consumer CLI."""

    def explode(*_: Any, **__: Any) -> None:
        raise AssertionError("open consulted the consumer bindings")

    published = lock.staleness.load_published(_REPO / lock.staleness.PUBLISHED_REL)
    cli.seed(lock.ONLINE_OPEN, None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lock.staleness, "load_bindings", explode)
        code = cli(
            [
                "open",
                "--instance-id",
                "next-study",
                "--declaration",
                "docs/example.md",
                "--evidence-root",
                "docs/evidence/example",
            ],
            published=published,
        )
    assert code == 0
    assert cli.read()["state"] == lock.RESEARCH_OPEN


# ── release governs instance lifetime, not editability ───────────────────────


def test_release_is_required_before_the_next_instance(
    cli, capsys: pytest.CaptureFixture[str]
) -> None:
    """`close` lifts the freeze; `release` is what lets a new study start.

    The output is asserted too: executable prose is the part a user actually
    follows, so it is where "release controls editability" would drift back in.
    """
    cli.seed(lock.RESEARCH_CLOSED, _instance(disposition="sealed"))
    assert (
        cli(
            [
                "open",
                "--instance-id",
                "second",
                "--declaration",
                "docs/example.md",
                "--evidence-root",
                "docs/evidence/example",
            ]
        )
        == 1
    )
    assert cli(["release", "--note", "acknowledged"]) == 0
    assert cli.read()["state"] == lock.ONLINE_OPEN

    output = capsys.readouterr().out
    assert "a new research instance may now open" in output
    assert "editable" not in output


# ── the guard cannot be edited by the freeze it guards ───────────────────────


@pytest.mark.parametrize(
    "path", [lock.LOCK_REL, "scripts/tools/research_lock.py", __file__]
)
def test_the_lock_is_outside_every_axis_it_can_freeze(path: str) -> None:
    """A lock inside its own frozen coordinate is H0 re-entry #3 again."""
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.relative_to(_REPO)
    assert partition.classify(relative) not in (
        "decision_relevant",
        "identity_semantics",
    )


def test_the_lock_file_is_not_a_document_the_master_map_owns() -> None:
    """JSON under contracts/ is state, not prose; the map indexes .md/.yaml."""
    assert lock.LOCK_REL.endswith(".json")
    assert partition.classify(lock.LOCK_REL) == "non_execution"


def test_the_default_freeze_is_the_accepted_stale_axis_set() -> None:
    """The default is not a new taxonomy: it is the accepted invalidating set."""
    bindings = json.loads(
        (_REPO / "docs/research/contracts/runtime_identity_bindings_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rule = bindings["consumption_rule"]["stale"]
    for axis in lock.DEFAULT_FROZEN_AXES:
        assert axis in rule
