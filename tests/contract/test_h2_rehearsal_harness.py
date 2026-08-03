"""The rehearsal harness must be unable to touch the owner's authorization.

Everything here defends one claim: a rehearsal walks the real admission and
consumption path while the owner's ledger and the canonical corpus are
mechanically out of reach. "Mechanically" is the whole point — the harness is
run by hand, at the end of a long chain, by someone who has already been told
twice that reading the code was not enough.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_measurement_evidence as evidence  # noqa: E402
import h2_path_partition as path_partition  # noqa: E402
import h2_rehearse_measurement as harness  # noqa: E402
import run_h2_measurement as controller  # noqa: E402
import run_h2_measurement_child as child  # noqa: E402

from tests.contract.test_h2_measurement_controller import _bundle  # noqa: E402

_HARNESS_REL = "scripts/tools/h2_rehearse_measurement.py"
_HARNESS_SOURCE = (_REPO / _HARNESS_REL).read_text(encoding="utf-8")


def _bundle_argv(tmp_path: Path) -> list[str]:
    """Bundle paths that never have to load: refusal precedes bundle loading."""
    return [
        "--freeze",
        (tmp_path / "freeze.json").as_posix(),
        "--layer-p-certificate",
        (tmp_path / "certificate.json").as_posix(),
        "--reference-probe",
        (tmp_path / "probe.json").as_posix(),
        "--runtime-inputs",
        (tmp_path / "runtime.json").as_posix(),
        "--published-identity",
        (tmp_path / "identity.json").as_posix(),
    ]


def _run(tmp_path: Path, ledger: Path, evidence_parent: Path) -> int:
    return harness.main(
        [
            *_bundle_argv(tmp_path),
            "--disposable-ledger",
            ledger.as_posix(),
            "--evidence-parent",
            evidence_parent.as_posix(),
        ]
    )


@pytest.fixture
def owner_ledger(tmp_path: Path, monkeypatch) -> Path:
    """A stand-in for the owner ledger. The real one is never involved."""
    ledger = tmp_path / "owner" / "state" / "h2_authorization_consumptions"
    ledger.mkdir(parents=True)
    monkeypatch.setattr(controller, "default_authorization_ledger", lambda: ledger)
    return ledger


def test_the_owner_ledger_is_a_protected_location(owner_ledger: Path) -> None:
    """The guard's inputs are resolved, so an alias of either is still it."""
    protected = harness._protected_locations()
    assert set(protected) == {"the repository", "the owner authorization ledger"}
    assert protected["the owner authorization ledger"] == owner_ledger.resolve()
    assert protected["the repository"] == _REPO.resolve()


@pytest.mark.parametrize(
    "case,expected",
    [
        ("relative", "not an absolute path"),
        ("inside_repo", "shares a location with the repository"),
        ("is_owner_ledger", "shares a location with the owner authorization ledger"),
        ("inside_owner_ledger", "shares a location with the owner authorization"),
        ("holds_owner_ledger", "shares a location with the owner authorization"),
        ("already_exists", "already exists"),
        ("dotdot_into_repo", "shares a location with the repository"),
        ("symlink_to_owner_ledger", "shares a location with the owner authorization"),
        ("parent_symlink_to_owner", "shares a location with the owner authorization"),
    ],
)
def test_unsafe_destinations_are_refused_and_nothing_is_created(
    tmp_path: Path,
    owner_ledger: Path,
    capsys,
    case: str,
    expected: str,
) -> None:
    """Every alias of a protected location, judged before any side effect.

    A lexical rule passes several of these: `..` normalises away, a symlink is
    a different spelling of the same directory, and `startswith` cannot tell
    `/repo` from `/repo-other`. The destination is resolved instead.
    """
    safe_parent = tmp_path / "outputs"
    safe_parent.mkdir()
    evidence_parent = safe_parent / "evidence"

    if case == "relative":
        ledger = Path("relative/ledger")
    elif case == "inside_repo":
        ledger = _REPO / "build" / "h2-rehearsal-ledger"
    elif case == "is_owner_ledger":
        ledger = owner_ledger
    elif case == "inside_owner_ledger":
        ledger = owner_ledger / "nested"
    elif case == "holds_owner_ledger":
        ledger = owner_ledger.parent.parent
    elif case == "already_exists":
        ledger = safe_parent / "used"
        ledger.mkdir()
    elif case == "dotdot_into_repo":
        ledger = _REPO / "build" / "elsewhere" / ".." / "h2-rehearsal-ledger"
    elif case == "symlink_to_owner_ledger":
        ledger = safe_parent / "ledger-link"
        ledger.symlink_to(owner_ledger, target_is_directory=True)
    elif case == "parent_symlink_to_owner":
        link = safe_parent / "state-link"
        link.symlink_to(owner_ledger.parent, target_is_directory=True)
        ledger = link / owner_ledger.name / "nested"
    else:  # pragma: no cover - the parametrisation is exhaustive
        raise AssertionError(case)

    assert _run(tmp_path, ledger, evidence_parent) == 2
    assert expected in capsys.readouterr().err
    assert not evidence_parent.exists()
    assert not (evidence_parent / harness.WITNESS_NAME).exists()


def test_two_destinations_that_resolve_to_one_place_are_refused(
    tmp_path: Path, owner_ledger: Path, capsys
) -> None:
    """Lexically disjoint, physically the same: only resolution can tell."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    link = outputs / "alias"
    link.symlink_to(outputs, target_is_directory=True)

    assert _run(tmp_path, outputs / "run" / "both", link / "run" / "both") == 2
    assert "resolve to one location" in capsys.readouterr().err

    assert _run(tmp_path, outputs / "run", link / "run" / "inside") == 2
    assert "resolve to one location" in capsys.readouterr().err


def test_a_refused_start_leaves_no_partial_output(
    tmp_path: Path, owner_ledger: Path
) -> None:
    """The first destination is rolled back when the second is refused."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    assert _run(tmp_path, outputs / "ledger", owner_ledger / "evidence") == 2
    assert not (outputs / "ledger").exists()


def test_a_substituted_destination_is_detected_not_prevented(tmp_path: Path) -> None:
    """The documented threat model, stated as behaviour.

    Holding a descriptor would not stop the controller, which resolves
    pathnames. What the harness promises is detection, and this is that promise.
    """
    target = tmp_path / "ledger"
    identity = harness.create_output_destination(target)
    identity.revalidate("unchanged")

    target.rename(tmp_path / "moved")
    (tmp_path / "substitute").mkdir()
    os.symlink(tmp_path / "substitute", target)
    with pytest.raises(harness.HarnessInvariantViolated) as excinfo:
        identity.revalidate("after substitution")
    assert "different directory" in str(excinfo.value)


def test_destinations_are_created_exclusively_and_privately(tmp_path: Path) -> None:
    identity = harness.create_output_destination(tmp_path / "ledger")
    assert identity.path.is_dir()
    assert oct(identity.path.stat().st_mode)[-3:] == "700"
    with pytest.raises(harness.RehearsalRefused):
        harness.create_output_destination(tmp_path / "ledger")


# -- the synthetic grant ---------------------------------------------------- #


def test_the_synthetic_grant_is_inert_against_the_owner_ledger(
    tmp_path: Path,
) -> None:
    """The isolation is arithmetic, not procedural.

    The execution domain binds the ledger root, so a grant issued against a
    disposable ledger cannot be admitted against the owner's — whatever anyone
    later does with the file.
    """
    bundle = _bundle(tmp_path)
    disposable = tmp_path / "disposable-ledger"
    disposable.mkdir()
    owner = tmp_path / "owner-ledger"
    owner.mkdir()
    grant_path, invocation_id = harness.synthetic_grant(
        bundle, ledger=disposable, destination=tmp_path
    )

    admitted = controller.load_authorization(
        grant_path, bundle, invocation_id=invocation_id, authorization_ledger=disposable
    )
    assert admitted.record["invocation_id"] == invocation_id

    with pytest.raises(controller.ControllerError):
        controller.load_authorization(
            grant_path, bundle, invocation_id=invocation_id, authorization_ledger=owner
        )


def test_the_synthetic_grant_reads_its_contract_from_the_authorities(
    tmp_path: Path, monkeypatch
) -> None:
    """Moving an authority must move the grant, with no edit to the harness."""
    bundle = _bundle(tmp_path)
    ledger = tmp_path / "ledger"
    ledger.mkdir()

    harness.synthetic_grant(bundle, ledger=ledger, destination=tmp_path)
    baseline = evidence.load_document(tmp_path, evidence.AUTHORIZATION_GRANT_NAME)
    assert set(baseline) == set(evidence.AUTHORIZATION_GRANT_MEMBERS)
    assert baseline["issued_by"] == evidence.AUTHORIZATION_ISSUER
    assert baseline["schema"] == evidence.AUTHORIZATION_GRANT_SCHEMA

    monkeypatch.setattr(evidence, "AUTHORIZATION_ISSUER", "successor_research_owner")
    monkeypatch.setattr(
        evidence, "AUTHORIZATION_GRANT_SCHEMA", "h2_exactly_once_authorization_v3"
    )
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    harness.synthetic_grant(bundle, ledger=ledger, destination=moved_dir)
    moved = evidence.load_document(moved_dir, evidence.AUTHORIZATION_GRANT_NAME)
    assert moved["issued_by"] == "successor_research_owner"
    assert moved["schema"] == "h2_exactly_once_authorization_v3"


def test_the_harness_restates_no_authorization_contract_value() -> None:
    """A literal here would be a second answer that drifts in silence."""
    for literal in (
        '"research_owner"',
        '"h2_exactly_once_authorization',
        '"phase_a"',
        '"authorization_grant.json"',
    ):
        assert literal not in _HARNESS_SOURCE, literal


def test_the_harness_walks_production_admission_and_execution() -> None:
    """A structural guardrail, not the semantic proof.

    The semantics are held by the tests above and by the real run; this only
    stops the harness from quietly growing a private path around admission.
    """
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(_HARNESS_SOURCE))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"load_authorization", "execute_controller"} <= calls
    assert "_consume_authorization" not in calls
    assert not any(
        "consume" in node.name
        for node in ast.walk(ast.parse(_HARNESS_SOURCE))
        if isinstance(node, ast.FunctionDef)
    )


def test_the_harness_takes_no_authorization_argument() -> None:
    """There is no argument through which the owner's grant could be spent."""
    with pytest.raises(SystemExit):
        harness.main(["--authorization", "/tmp/grant.json"])


def test_the_harness_is_classified_plumbing_only() -> None:
    """`unclassified` is fail-closed, not a resting state.

    `h2_path_partition` rejects unclassified paths from Layer-P retry
    admissibility on purpose: a path nobody has classified might be anything.
    This file landed under a name that matched none of the `scripts/tools/`
    plumbing prefixes, so a later Layer-P run whose base predated an edit to it
    would have been blocked by a file that only builds diagnostics. The name
    carries the classification, which makes the name itself a contract.
    """
    assert path_partition.classify(_HARNESS_REL) == "plumbing_only"


# -- the success conjunction ------------------------------------------------ #


class _Selection(NamedTuple):
    terminal: str | None
    result: str


def _invocation(
    directory: Path,
    run_id: str,
    state: str,
    *,
    schema: str = child.INVOCATION_SCHEMA,
) -> None:
    """The child's durable lifecycle record, in the shape the child writes it."""
    evidence.write_document(
        directory,
        child.INVOCATION_NAME,
        {
            "schema": schema,
            "capture_phase": evidence.CAPTURE_PHASE["a"],
            "run_id": run_id,
            "sequence": controller.SEQUENCE,
            "state": state,
        },
    )


def _archive(
    root: Path,
    *,
    runs: tuple[str, ...],
    states: dict[str, str | None] | None = None,
) -> Path:
    """The parts of an evidence root the harness projects its verdict from.

    `states` overrides one run's lifecycle: a state string, or `None` to
    materialize the directory with no lifecycle record at all.
    """
    root.mkdir(parents=True)
    sequence = controller.SEQUENCE
    evidence.write_document(
        root,
        evidence.CONTROLLER_NAME,
        {
            "schema": evidence.CONTROLLER_SCHEMA,
            "capture_phase": evidence.CAPTURE_PHASE["a"],
            "instrumentation_head": "a" * 40,
            "ordered_runs": list(runs),
            "sequence": sequence,
            "started_utc": "2026-07-29T00:00:00Z",
            "state": "terminal",
        },
    )
    overrides = states or {}
    for run_id in runs:
        directory = evidence.run_dir(root, sequence, run_id)
        directory.mkdir(parents=True)
        state = overrides.get(run_id, child.RUN_COMPLETED)
        if state is not None:
            _invocation(directory, run_id, state)
    evidence.write_checksum_inventory(root)
    return root


def _consumed(ledger: Path, grant: dict[str, Any]) -> None:
    evidence.write_document(
        ledger,
        f"{grant['authorization_id']}.json",
        {
            "schema": evidence.AUTHORIZATION_SCHEMA,
            "authorization_digest": evidence.digest(grant),
            "authorization_id": grant["authorization_id"],
            "capture_phase": grant["capture_phase"],
            "consumed_utc": "2026-07-29T00:00:00Z",
            "controller_digest": grant["controller_digest"],
            "execution_domain": grant["execution_domain"],
            "freeze_digest": grant["freeze_digest"],
            "instrumentation_head": grant["instrumentation_head"],
            "invocation_id": grant["invocation_id"],
            "state": "consumed",
        },
    )


def _grant_record(ledger: Path) -> dict[str, Any]:
    return {
        "schema": evidence.AUTHORIZATION_GRANT_SCHEMA,
        "authorization_id": "1" * 64,
        "capture_phase": controller.CAPTURE_PHASE,
        "controller_digest": "2" * 64,
        "execution_domain": evidence.digest(
            evidence.authorization_execution_domain(ledger)
        ),
        "freeze_digest": "3" * 64,
        "instrumentation_head": "4" * 40,
        "invocation_id": "5" * 64,
        "issued_by": evidence.AUTHORIZATION_ISSUER,
    }


@pytest.mark.parametrize(
    "defect,expected",
    [
        (None, None),
        ("terminal", "reached terminal"),
        ("invalid", "refused the rehearsal archive"),
        ("no_receipt", "not exactly the one receipt"),
        ("extra_receipt", "not exactly the one receipt"),
        ("foreign_receipt", "receipt authorization_id is not the synthetic grant's"),
        ("short_runs", "did not complete"),
        ("corpus_admitted", "canonical corpus did not refuse"),
    ],
)
def test_success_requires_every_conjunct_not_only_the_terminal(
    tmp_path: Path, monkeypatch, defect: str | None, expected: str | None
) -> None:
    """A `None` terminal is one conjunct of success, never the whole of it."""
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger = harness.DirectoryIdentity(
        ledger_dir, ledger_dir.stat().st_dev, ledger_dir.stat().st_ino
    )
    grant = _grant_record(ledger_dir)
    runs = tuple(evidence.RUN_IDS)
    if defect == "short_runs":
        runs = runs[:-1]
    root = _archive(tmp_path / "root", runs=runs)

    if defect != "no_receipt":
        _consumed(ledger_dir, grant)
    if defect == "foreign_receipt":
        # The expected filename, a different grant inside it: the name alone is
        # not evidence that this receipt is the one the harness issued.
        name = f"{grant['authorization_id']}.json"
        evidence.write_document(
            ledger_dir,
            name,
            {**evidence.load_document(ledger_dir, name), "authorization_id": "9" * 64},
        )
    if defect == "extra_receipt":
        _consumed(ledger_dir, {**grant, "authorization_id": "8" * 64})

    monkeypatch.setattr(controller, "checkout_hygiene_reasons", lambda **_: ())
    failures = harness.rehearsal_invariant_failures(
        root,
        selection=_Selection("H2_MEASUREMENT_EXECUTION_INVALID", "runner_nonzero")
        if defect == "terminal"
        else _Selection(None, "clean"),
        report={"valid": defect != "invalid"},
        grant_record=grant,
        ledger=ledger,
        corpus_admission=(
            {"admitted": True, "reasons": []}
            if defect == "corpus_admitted"
            else {"admitted": False, "reasons": ["controlled host mismatch"]}
        ),
    )
    if expected is None:
        assert failures == ()
    else:
        assert any(expected in failure for failure in failures), failures


def test_a_dirty_checkout_after_the_run_is_a_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """A rehearsal that contaminated the working tree has not demonstrated a run."""
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger = harness.DirectoryIdentity(
        ledger_dir, ledger_dir.stat().st_dev, ledger_dir.stat().st_ino
    )
    grant = _grant_record(ledger_dir)
    _consumed(ledger_dir, grant)
    root = _archive(tmp_path / "root", runs=tuple(evidence.RUN_IDS))

    monkeypatch.setattr(
        controller, "checkout_hygiene_reasons", lambda **_: ("untracked: runs/x",)
    )
    failures = harness.rehearsal_invariant_failures(
        root,
        selection=_Selection(None, "clean"),
        report={"valid": True},
        grant_record=grant,
        ledger=ledger,
        corpus_admission={
            "admitted": False,
            "reasons": ["controlled host mismatch"],
        },
    )
    assert any("left the checkout dirty" in failure for failure in failures)


def test_the_run_summary_is_projected_from_the_archive(tmp_path: Path) -> None:
    """Not a counter the harness kept: the evidence is the only source."""
    root = _archive(tmp_path / "root", runs=tuple(evidence.RUN_IDS)[:2])
    summary = harness.ordered_run_summary(root)
    assert [item["run_id"] for item in summary] == list(evidence.RUN_IDS)
    assert [item["materialized"] for item in summary] == [True, True, False, False]
    assert [item["recorded"] for item in summary] == [True, True, False, False]
    assert [item["completed"] for item in summary] == [True, True, False, False]


@pytest.mark.parametrize(
    ("state", "lifecycle_present", "completed"),
    [
        (child.RUN_COMPLETED, True, True),
        (child.RUN_FAILED, True, False),
        (child.RUN_RUNNING, True, False),
        ("something_new", True, False),
        (None, False, False),
    ],
    ids=["completed", "failed", "running", "unknown-state", "no-lifecycle-record"],
)
def test_a_materialized_run_is_not_a_completed_run(
    tmp_path: Path, state: str | None, lifecycle_present: bool, completed: bool
) -> None:
    """The 2026-07-29 rehearsal read a failed run as complete.

    Its directory existed, because the controller creates it before the child
    runs and the child leaves it behind when it dies. Completion is the child's
    own durable lifecycle record — the one the controller already reads — and
    anything else, including an unreadable or unknown state, is not completion.
    """
    run_id = evidence.RUN_IDS[0]
    root = _archive(tmp_path / "root", runs=(run_id,), states={run_id: state})
    first = harness.ordered_run_summary(root)[0]
    assert first["materialized"] is True
    assert first["lifecycle_present"] is lifecycle_present
    assert first["lifecycle_state"] == (state if lifecycle_present else None)
    assert first["completed"] is completed


def test_a_wrong_lifecycle_schema_is_not_present_or_completed(tmp_path: Path) -> None:
    """Even canonical JSON cannot claim completion under the wrong contract."""
    run_id = evidence.RUN_IDS[0]
    root = _archive(tmp_path / "root", runs=(run_id,), states={run_id: None})
    directory = evidence.run_dir(root, controller.SEQUENCE, run_id)
    _invocation(
        directory,
        run_id,
        child.RUN_COMPLETED,
        schema="wrong_invocation_schema",
    )

    # The ordinary loader accepts the bytes, proving this is canonical JSON;
    # only the required invocation schema is wrong.
    record = evidence.load_document(directory, child.INVOCATION_NAME)
    assert record["schema"] == "wrong_invocation_schema"
    assert record["state"] == child.RUN_COMPLETED

    first = harness.ordered_run_summary(root)[0]
    assert first["materialized"] is True
    assert first["lifecycle_present"] is False
    assert first["lifecycle_state"] is None
    assert first["completed"] is False


def test_a_failed_run_fails_the_rehearsal_even_with_no_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    """B's whole reason to exist, and it must not lean on the terminal conjunct.

    If the controller ever finishes without selecting a terminal while a child
    died, the run predicate is the only thing left standing between that archive
    and a rehearsal reported as passed.
    """
    run_id = evidence.RUN_IDS[0]
    root = _archive(
        tmp_path / "root",
        runs=tuple(evidence.RUN_IDS),
        states={run_id: child.RUN_FAILED},
    )
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger = harness.DirectoryIdentity(
        ledger_dir, ledger_dir.stat().st_dev, ledger_dir.stat().st_ino
    )
    grant = _grant_record(ledger_dir)
    _consumed(ledger_dir, grant)
    monkeypatch.setattr(controller, "checkout_hygiene_reasons", lambda **_: ())

    failures = harness.rehearsal_invariant_failures(
        root,
        selection=_Selection(None, "measurement_pass"),
        report={"valid": True},
        grant_record=grant,
        ledger=ledger,
        corpus_admission={
            "admitted": False,
            "reasons": ["controlled host mismatch"],
        },
    )
    assert [f for f in failures if run_id in f], failures
    assert all("reached terminal" not in failure for failure in failures)


def test_a_terminal_still_dominates_the_rehearsal_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    """A tightened run predicate must not displace the terminal it sits beside."""
    root = _archive(
        tmp_path / "root",
        runs=tuple(evidence.RUN_IDS),
        states={evidence.RUN_IDS[0]: child.RUN_FAILED},
    )
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger = harness.DirectoryIdentity(
        ledger_dir, ledger_dir.stat().st_dev, ledger_dir.stat().st_ino
    )
    grant = _grant_record(ledger_dir)
    _consumed(ledger_dir, grant)
    monkeypatch.setattr(controller, "checkout_hygiene_reasons", lambda **_: ())

    failures = harness.rehearsal_invariant_failures(
        root,
        selection=_Selection("H2_MEASUREMENT_EXECUTION_INVALID", "runner_nonzero"),
        report={"valid": True},
        grant_record=grant,
        ledger=ledger,
        corpus_admission={
            "admitted": False,
            "reasons": ["controlled host mismatch"],
        },
    )
    assert any("H2_MEASUREMENT_EXECUTION_INVALID" in f for f in failures)
    assert any(evidence.RUN_IDS[0] in f for f in failures)


def test_the_lifecycle_vocabulary_is_the_child_s_own(
    tmp_path: Path, monkeypatch
) -> None:
    """Bound by behaviour, not by string-matching.

    The child performs both transitions out of `running`, so it owns the names.
    Move the authority and the harness's verdict must move with it; a reader
    holding its own spelling would keep answering with the old one (§ C3.9).

    A literal scan is kept only for the two names the harness has no legitimate
    use for — `"completed"` and `"failed"` are also its own witness vocabulary,
    so their presence in the source proves nothing either way.
    """
    for literal in ('"running"', '"invocation.json"'):
        assert literal not in _HARNESS_SOURCE, literal

    run_id = evidence.RUN_IDS[0]
    monkeypatch.setattr(child, "RUN_COMPLETED", "finished_under_a_new_name")
    root = _archive(tmp_path / "root", runs=(run_id,), states={run_id: "completed"})
    assert harness.ordered_run_summary(root)[0]["completed"] is False

    moved = _archive(
        tmp_path / "moved",
        runs=(run_id,),
        states={run_id: "finished_under_a_new_name"},
    )
    assert harness.ordered_run_summary(moved)[0]["completed"] is True


# -- the witness ------------------------------------------------------------ #


def test_a_failure_after_the_witness_exists_is_recorded_as_failed(
    tmp_path: Path, monkeypatch
) -> None:
    """`started` must never be the last word of a run that actually failed."""
    bundle = _bundle(tmp_path)
    ledger = harness.create_output_destination(tmp_path / "ledger")
    parent = harness.create_output_destination(tmp_path / "evidence")
    witness_path = parent.path / harness.WITNESS_NAME

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise controller.ControllerError("the launch path died")

    monkeypatch.setattr(controller, "checkout_hygiene_reasons", lambda **_: ())
    monkeypatch.setattr(controller, "execute_controller", explode)
    with pytest.raises(controller.ControllerError):
        harness.rehearse(
            bundle=bundle,
            ledger=ledger,
            evidence_parent=parent,
            witness_path=witness_path,
            started_utc="2026-07-29T00:00:00Z",
        )

    witness = evidence.load_document(
        witness_path.parent, witness_path.name, schema=harness.WITNESS_SCHEMA
    )
    assert witness["status"] == "failed"
    assert witness["failure_class"] == harness.HARNESS_INVARIANT_VIOLATED
    assert witness["authority"] == harness.WITNESS_AUTHORITY
    assert any("the launch path died" in item for item in witness["failures"])
    assert witness["finished_utc"]


def test_a_rehearsal_consumes_its_own_grant_and_completes(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole lifecycle, with the production consumption primitive real.

    Only the controller's *execution* is stood in for — the machine it drives
    needs a GPU, a build and a dataset. Admission and consumption are the real
    ones, which is what this harness exists to exercise.
    """
    bundle = _bundle(tmp_path)
    ledger = harness.create_output_destination(tmp_path / "ledger")
    parent = harness.create_output_destination(tmp_path / "evidence")
    witness_path = parent.path / harness.WITNESS_NAME
    root = tmp_path / "evidence" / "archive"

    def execute(bundle_arg, *, authorization, evidence_parent, authorization_ledger):
        assert authorization_ledger == ledger.path
        assert evidence_parent == parent.path
        controller._consume_authorization(
            authorization, bundle=bundle_arg, ledger=authorization_ledger
        )
        _archive(root, runs=tuple(evidence.RUN_IDS))
        return root, _Selection(None, "clean")

    report = {"schema": "x", "valid": True, "verify_class": "complete"}
    monkeypatch.setattr(controller, "execute_controller", execute)
    monkeypatch.setattr(controller, "checkout_hygiene_reasons", lambda **_: ())
    monkeypatch.setattr(harness.verifier, "classify", lambda _root: "complete")
    monkeypatch.setattr(harness.verifier, "VERIFIERS", {"complete": lambda _r: report})
    corpus_refusal = {
        "admitted": False,
        "reasons": ["the disposable ledger is not the controlled host"],
    }
    monkeypatch.setattr(
        harness, "corpus_admission_witness", lambda _root: corpus_refusal
    )

    witness, exit_code = harness.rehearse(
        bundle=bundle,
        ledger=ledger,
        evidence_parent=parent,
        witness_path=witness_path,
        started_utc="2026-07-29T00:00:00Z",
    )

    assert exit_code == 0
    assert witness["schema"] == "h2_phase_a_rehearsal_witness_v3"
    assert witness["status"] == "completed"
    assert witness["failure_class"] is None
    assert witness["failures"] == []
    assert witness["verifier_report"] == report
    assert witness["corpus_admission"] == corpus_refusal
    assert witness["source_head"] == bundle.head
    assert witness["disposable_ledger"]["path"] == ledger.path.as_posix()
    assert Path(witness["receipt"]).parent == ledger.path

    # The receipt is in the disposable ledger and the owner's is untouched: the
    # exactly-once write happened somewhere no owner authority can be reached.
    receipts = sorted(path.name for path in ledger.path.iterdir())
    assert receipts == [f"{witness['authorization_id']}.json"]

    stored = evidence.load_document(
        witness_path.parent, witness_path.name, schema=harness.WITNESS_SCHEMA
    )
    assert stored == witness

    # Recomputed from the artifacts themselves: a witness carrying only
    # pathnames could not tell a later reader whether the archive that reached
    # registration is the archive that was verified here.
    assert witness["evidence_root_digest"] == evidence.checksum_inventory_digest(root)
    assert witness["verifier_report_digest"] == evidence.digest(report)
    assert witness["receipt_digest"] == evidence.digest(
        evidence.load_document(ledger.path, Path(witness["receipt"]).name)
    )


def test_a_dirty_checkout_before_the_run_stops_the_rehearsal(
    tmp_path: Path, monkeypatch
) -> None:
    """The harness owns this conjunct; it does not lean on the controller.

    The controller refuses a dirty checkout as well, but the seam it is reached
    through is injectable, and a success contract that depends on someone else
    still deciding it has a hole exactly where that seam moves.
    """
    bundle = _bundle(tmp_path)
    ledger = harness.create_output_destination(tmp_path / "ledger")
    parent = harness.create_output_destination(tmp_path / "evidence")
    witness_path = parent.path / harness.WITNESS_NAME

    readings = iter([("untracked: runs/stray.log",), (), ()])
    monkeypatch.setattr(
        controller, "checkout_hygiene_reasons", lambda **_: next(readings, ())
    )
    monkeypatch.setattr(
        controller,
        "execute_controller",
        lambda *a, **k: pytest.fail("the controller ran on a dirty checkout"),
    )

    with pytest.raises(harness.HarnessInvariantViolated) as excinfo:
        harness.rehearse(
            bundle=bundle,
            ledger=ledger,
            evidence_parent=parent,
            witness_path=witness_path,
            started_utc="2026-07-29T00:00:00Z",
        )
    assert "dirty before rehearsal" in str(excinfo.value)

    witness = evidence.load_document(
        witness_path.parent, witness_path.name, schema=harness.WITNESS_SCHEMA
    )
    assert witness["status"] == "failed"
    assert witness["failure_class"] == harness.HARNESS_INVARIANT_VIOLATED
    assert witness["checkout_hygiene_before"] == ["untracked: runs/stray.log"]
    assert not list(ledger.path.iterdir()), "no grant was consumed"


def test_the_owner_ledger_is_untouched_by_a_whole_rehearsal(
    tmp_path: Path, monkeypatch, owner_ledger: Path
) -> None:
    """Inventory and content before and after, plus a guard on the write path.

    The inventory comparison is the proof: a syscall guard can always be walked
    around by a library the harness does not know about. `atime` is excluded
    because reading a file to digest it changes it.
    """

    def _state(root: Path) -> dict[str, tuple[int, int, bytes]]:
        return {
            path.relative_to(root).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.read_bytes(),
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    (owner_ledger / "prior.json").write_text('{"schema":"x"}', encoding="utf-8")
    before = _state(owner_ledger)

    real_write = evidence.write_document_exclusive

    def guarded(root: Path, name: str, payload: Any) -> Path:
        assert not Path(root).resolve().is_relative_to(owner_ledger.resolve()), (
            f"the rehearsal wrote into the owner ledger: {root}/{name}"
        )
        return real_write(root, name, payload)

    monkeypatch.setattr(evidence, "write_document_exclusive", guarded)

    bundle = _bundle(tmp_path)
    ledger = harness.create_output_destination(tmp_path / "ledger")
    parent = harness.create_output_destination(tmp_path / "evidence")
    root = tmp_path / "evidence" / "archive"

    def execute(bundle_arg, *, authorization, evidence_parent, authorization_ledger):
        controller._consume_authorization(
            authorization, bundle=bundle_arg, ledger=authorization_ledger
        )
        _archive(root, runs=tuple(evidence.RUN_IDS))
        return root, _Selection(None, "clean")

    monkeypatch.setattr(controller, "execute_controller", execute)
    monkeypatch.setattr(controller, "checkout_hygiene_reasons", lambda **_: ())
    monkeypatch.setattr(harness.verifier, "classify", lambda _root: "complete")
    monkeypatch.setattr(
        harness.verifier, "VERIFIERS", {"complete": lambda _r: {"valid": True}}
    )
    monkeypatch.setattr(
        harness,
        "corpus_admission_witness",
        lambda _root: {
            "admitted": False,
            "reasons": ["the disposable ledger is not the controlled host"],
        },
    )

    _, exit_code = harness.rehearse(
        bundle=bundle,
        ledger=ledger,
        evidence_parent=parent,
        witness_path=parent.path / harness.WITNESS_NAME,
        started_utc="2026-07-29T00:00:00Z",
    )
    assert exit_code == 0
    assert _state(owner_ledger) == before


def test_witness_v3_records_the_corpus_owners_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    root.mkdir()

    def refused(_roots: list[Path]) -> None:
        raise harness.corpus.CorpusError("controlled host domain mismatch")

    monkeypatch.setattr(harness.corpus, "check_corpus", refused)
    assert harness.corpus_admission_witness(root) == {
        "admitted": False,
        "reasons": ["controlled host domain mismatch"],
    }


def test_witness_v3_does_not_turn_corpus_admission_into_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setattr(harness.corpus, "check_corpus", lambda _roots: [])
    assert harness.corpus_admission_witness(root) == {
        "admitted": True,
        "reasons": [],
    }
