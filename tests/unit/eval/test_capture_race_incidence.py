"""Harness self-verification for the #340 capture-race incidence campaign.

Two jobs.  First, the §13 step-4 equivalence check: every constant the harness
decides on is read back out of the frozen preregistration, so the harness
cannot drift from the document it claims to execute.  Second, the fail-closed
paths, which are the only ones that matter before run 1 — a harness that works
on the happy path and silently mis-files a broken run is worse than no harness,
because its output looks like data.

The live GPU workload is never launched here; ``execute_run`` takes an injected
runner so the classification and abort paths can be exercised exactly.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.capture_race_incidence import (  # noqa: E402
    BASE_ARGS,
    CAPTURE_FAILURE_SIGNATURES,
    COORDINATE_IMPLEMENTATION,
    ExecutionInvalid,
    N_PAIRS,
    PATH_EXTRA_ARGS,
    PATH_ORDER,
    PREREG_REL,
    SEQUENCES,
    Campaign,
    MAX_CONSECUTIVE_SETUP_INVALID,
    build_argv,
    capture_failure_hits,
    child_environment,
    classify,
    execute_run,
    main,
    preflight,
    run_campaign,
    schedule,
    setup_failure_signature,
    TARGET_SOURCE_SHA,
)

PREREG = (ROOT / PREREG_REL).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# §13 step 4: the harness must match the document it executes
# --------------------------------------------------------------------------


def test_preregistration_is_present_and_sealed() -> None:
    assert (ROOT / PREREG_REL).is_file()
    assert "## A1." in PREREG, "amendment A1 must be part of the frozen text"


def test_frozen_coordinate_matches_the_preregistration() -> None:
    assert TARGET_SOURCE_SHA in PREREG
    assert COORDINATE_IMPLEMENTATION in PREREG


def test_every_capture_signature_appears_in_the_preregistration() -> None:
    for code, signature in CAPTURE_FAILURE_SIGNATURES:
        assert signature in PREREG, f"{signature!r} is not in §3"
        assert code in PREREG


def test_workload_configuration_matches_the_preregistration() -> None:
    for token in BASE_ARGS:
        assert token in PREREG
    assert "--no-gpu-decode" in PATH_EXTRA_ARGS["B"]
    assert PATH_EXTRA_ARGS["A"] == (), "path A is the unmodified production path"
    # §2 writes the cohort compressed as MOT17-<a>/<b>/…-SDP; expand and compare.
    match = re.search(r"`MOT17-([0-9/]+)-SDP`", PREREG)
    assert match, "§2 no longer states the sequence cohort in the expected form"
    declared = tuple(f"MOT17-{n}-SDP" for n in match.group(1).split("/"))
    assert declared == SEQUENCES


def test_n_and_interleaving_are_the_frozen_ones() -> None:
    assert N_PAIRS == 100
    assert "100" in PREREG
    assert PATH_ORDER == ("A", "B"), "§2 freezes the interleaving as A first"
    # A2.2: the first element is the effective slot, not an attempt ordinal.
    assert list(schedule(3)) == [
        (0, "A"),
        (0, "B"),
        (1, "A"),
        (1, "B"),
        (2, "A"),
        (2, "B"),
    ]


def test_record_schema_fields_are_all_declared() -> None:
    """Every field the harness writes is named in §10 or in A1's additions."""
    record = _record_via_runner(exit_code=0, log="", complete=True)
    for field in record:
        assert field in PREREG, f"{field!r} is written but not declared"


def test_harness_never_imports_saccade() -> None:
    """The control worktree must not be able to leak into a measurement.

    A harness that imported saccade would resolve it through the venv's
    absolute editable path — the exact hazard preflight check 3 exists to
    catch — so it would be violating the property it is asked to verify.
    """
    source = (ROOT / "scripts" / "tools" / "capture_race_incidence.py").read_text(
        encoding="utf-8"
    )
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import saccade")
        assert not stripped.startswith("from saccade")


# --------------------------------------------------------------------------
# §3 predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, code",
    [
        (
            "RuntimeError: CUDA error: operation not permitted when stream is capturing",
            "900",
        ),
        ("cudaErrorStreamCaptureInvalidated", "901"),
        ("operation failed due to a previous error during capture", "901"),
        (
            "RuntimeError: CUDA error: operation would make the legacy stream "
            "depend on a capturing blocking stream",
            "906",
        ),
        ("  File c10/cuda/CUDAStream.cpp: currentStreamCaptureStatusMayInitCtx", "906"),
    ],
)
def test_capture_predicate_matches_driver_and_wrapper_forms(
    line: str, code: str
) -> None:
    hits = capture_failure_hits(f"noise\n{line}\nmore noise\n")
    assert hits, f"{line!r} should match §3"
    assert hits[0]["cuda_code"] == code
    assert hits[0]["line"] == 2


def test_capture_predicate_is_silent_on_a_healthy_log() -> None:
    log = "\n".join(
        [
            "loading checkpoint runs/x/best.ckpt",
            "TensorRT engine ready",
            "\N{CLAPPER BOARD} MOT17-02-SDP [100/600]",
            "done",
        ]
    )
    assert capture_failure_hits(log) == []


def test_exit_code_is_recorded_but_not_part_of_the_predicate() -> None:
    """§3: exit code is not a reliable detection surface for this failure."""
    clean = _record_via_runner(exit_code=1, log="unrelated tantrum\n", complete=True)
    assert clean["exit_code"] == 1
    assert clean["verdict"] == "ok"

    hit = _record_via_runner(
        exit_code=0,
        log="cudaErrorStreamCaptureImplicit\n",
        complete=True,
    )
    assert hit["exit_code"] == 0
    assert hit["verdict"] == "failure"


# --------------------------------------------------------------------------
# A1: the denominator boundary
# --------------------------------------------------------------------------


def test_setup_signature_needs_both_a_context_and_a_failure_term() -> None:
    assert setup_failure_signature("loading checkpoint from disk", "/out") is None
    assert setup_failure_signature("TensorRT engine ready", "/out") is None
    assert setup_failure_signature("FileNotFoundError: best.ckpt", "/out") == "weights"


def test_unrecognised_crash_with_no_progress_ends_the_campaign(tmp_path: Path) -> None:
    """The case A1 exists for.

    A crash inside the first 99 frames leaves no progress marker and no result
    file.  The naive reading files that as "never started" and quietly re-runs
    it, which is precisely the post-hoc removal §4 forbids.  It must abort.
    """
    verdict, observations = classify(
        "Segmentation fault (core dumped)\n", tmp_path, output_dir=str(tmp_path)
    )
    assert observations["progress_markers"] == 0
    assert observations["sequences_completed"] == []
    assert observations["setup_failure_signature"] is None
    assert observations["sequence_execution_started"] is True, "started is the default"
    assert verdict == "execution_invalid"


def test_recognised_setup_failure_is_the_only_rerunnable_invalid(
    tmp_path: Path,
) -> None:
    verdict, observations = classify(
        "FileNotFoundError: no such file: data/MOT17-02-SDP/seqinfo.ini\n",
        tmp_path,
        output_dir=str(tmp_path),
    )
    assert observations["sequence_execution_started"] is False
    assert observations["setup_failure_signature"] == "dataset"
    assert verdict == "invalid"


def test_a_started_run_that_stops_short_ends_the_campaign(tmp_path: Path) -> None:
    _complete(tmp_path, SEQUENCES[:3])
    log = "\N{CLAPPER BOARD} MOT17-05-SDP [400/837]\ntorch.OutOfMemoryError\n"
    verdict, observations = classify(log, tmp_path, output_dir=str(tmp_path))
    assert observations["sequence_execution_started"] is True
    assert observations["sequences_completed"] == list(SEQUENCES[:3])
    assert verdict == "execution_invalid"


def test_a_setup_signature_cannot_rescue_a_run_that_had_started(tmp_path: Path) -> None:
    """Both A1 conditions are required, so a late OOM cannot be re-labelled."""
    _complete(tmp_path, SEQUENCES[:2])
    log = "\N{CLAPPER BOARD} MOT17-02-SDP [600/600]\nFileNotFoundError: best.ckpt\n"
    verdict, observations = classify(log, tmp_path, output_dir=str(tmp_path))
    assert observations["setup_failure_signature"] == "weights"
    assert observations["sequence_execution_started"] is True
    assert verdict == "execution_invalid"


def test_capture_failure_outranks_an_incomplete_run(tmp_path: Path) -> None:
    """A capture error that also kills the run is a §3 failure, not invalid."""
    verdict, _ = classify(
        "cudaErrorStreamCaptureImplicit\n", tmp_path, output_dir=str(tmp_path)
    )
    assert verdict == "failure"


def test_all_seven_sequences_are_required_for_ok(tmp_path: Path) -> None:
    _complete(tmp_path, SEQUENCES)
    verdict, observations = classify(
        "\N{CLAPPER BOARD} x [100/600]\n", tmp_path, output_dir=str(tmp_path)
    )
    assert observations["sequences_completed"] == list(SEQUENCES)
    assert verdict == "ok"


# --------------------------------------------------------------------------
# §2 preflight — including the negative test for check 3
# --------------------------------------------------------------------------


def _fake_interpreter(tmp_path: Path, prints: str, *, rc: int = 0) -> Path:
    script = tmp_path / "fake-python"
    script.write_text(
        f'#!/bin/sh\necho "{prints}"\nexit {rc}\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _git_target(tmp_path: Path, *, dirty: bool = False) -> Path:
    import subprocess

    target = tmp_path / "target"
    (target / "src" / "saccade").mkdir(parents=True)
    (target / "src" / "saccade" / "__init__.py").write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", str(target)], check=True, env=env)
    subprocess.run(["git", "-C", str(target), "add", "-A"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(target), "commit", "-qm", "seed"], check=True, env=env
    )
    if dirty:
        (target / "src" / "saccade" / "__init__.py").write_text("x", encoding="utf-8")
    return target


def test_preflight_rejects_a_wrong_import_root_and_still_records_it(
    tmp_path: Path,
) -> None:
    """The negative test for check 3.

    Deliberately start with an interpreter whose ``saccade`` resolves outside
    the target.  The run must fail closed *before* the workload is launched,
    and all three observed identity fields must still land — a fail-closed stop
    with no record is not evidence that anything was checked.
    """
    target = _git_target(tmp_path)
    wrong_root = "/home/someone/other-checkout/src"
    interpreter = _fake_interpreter(tmp_path, wrong_root)

    launched: list[list[str]] = []

    def runner(argv, cwd, env):  # pragma: no cover - must never be reached
        launched.append(list(argv))
        return 0, ""

    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        artifact_dir=tmp_path / "artifacts",
        runner=runner,
    )

    assert launched == [], "the workload must not start once a check has failed"
    assert record["verdict"] == "execution_invalid"
    assert record["saccade_import_root"] == wrong_root
    assert record["target_worktree_clean"] is True
    assert record["target_head_observed"] is not None
    assert len(record["target_head_observed"]) == 40
    assert "editable install points outside" in record["invalid_reason"]
    assert record["exit_code"] is None


def test_preflight_passes_when_the_import_root_is_the_target(tmp_path: Path) -> None:
    target = _git_target(tmp_path)
    interpreter = _fake_interpreter(tmp_path, str((target / "src").resolve()))
    checks = preflight(target, interpreter)
    assert checks.import_root == str((target / "src").resolve())
    # HEAD is a throwaway repo, so check 1 is the only expected complaint.
    assert all("check 1" in failure for failure in checks.failures)


def test_preflight_rejects_a_dirty_target(tmp_path: Path) -> None:
    target = _git_target(tmp_path, dirty=True)
    interpreter = _fake_interpreter(tmp_path, str((target / "src").resolve()))
    checks = preflight(target, interpreter)
    assert checks.clean is False
    assert any("check 2" in failure for failure in checks.failures)


def test_preflight_records_a_broken_import_probe(tmp_path: Path) -> None:
    target = _git_target(tmp_path)
    interpreter = _fake_interpreter(tmp_path, "", rc=1)
    checks = preflight(target, interpreter)
    assert checks.import_root is None
    assert any("check 3" in failure for failure in checks.failures)


# --------------------------------------------------------------------------
# §8 observers, argv, and the campaign counters
# --------------------------------------------------------------------------


def test_observer_environment_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/opt/nsight/libcupti.so")
    with pytest.raises(ExecutionInvalid, match="§8 forbids observers"):
        child_environment()


def test_child_environment_is_unbuffered() -> None:
    assert child_environment({})["PYTHONUNBUFFERED"] == "1"


def test_argv_is_built_from_the_target_not_the_control_worktree(tmp_path: Path) -> None:
    argv = build_argv("B", tmp_path / "target", tmp_path / "py", tmp_path / "out")
    assert argv[1] == str(tmp_path / "target" / "scripts" / "eval" / "mot17.py")
    assert "--no-gpu-decode" in argv
    assert argv[-2:] == ["--output", str(tmp_path / "out")]


def test_first_capture_failure_terminates_that_path() -> None:
    campaign = Campaign.new()
    campaign.absorb(
        {"path": "A", "verdict": "failure", "capture_error_hits": [{"x": 1}]}
    )
    assert campaign.terminal == "FAILURE_OBSERVED_A"
    assert campaign.path_done("A", 100) is True


def test_execution_invalid_stops_everything() -> None:
    campaign = Campaign.new()
    campaign.absorb(
        {"path": "B", "verdict": "execution_invalid", "invalid_reason": "x"}
    )
    assert campaign.terminal == "EXECUTION_INVALID"
    assert campaign.finalise(100) == "EXECUTION_INVALID"


def test_invalid_runs_do_not_count_toward_the_denominator() -> None:
    campaign = Campaign.new()
    campaign.absorb({"path": "A", "verdict": "invalid", "invalid_reason": "dataset"})
    assert campaign.valid["A"] == 0
    assert campaign.invalid == 1
    assert campaign.terminal is None


def test_a_short_campaign_is_never_closed_bounded() -> None:
    campaign = Campaign.new()
    for path in PATH_ORDER:
        campaign.valid[path] = 99
    assert campaign.finalise(100) == "UNRESOLVED_INVALID_STUDY"


def test_pairs_may_not_be_lowered_for_the_real_campaign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--target", str(tmp_path), "--pairs", "3"])
    assert rc == 1
    assert "§2 freezes N" in capsys.readouterr().err


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _complete(run_dir: Path, sequences) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        (run_dir / f"{seq}.txt").write_text(
            "1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8"
        )


def _record_via_runner(*, exit_code: int, log: str, complete: bool) -> dict:
    """Drive execute_run with a stub target so classification can be checked."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target = _git_target(tmp_path)
        head = os.popen(f"git -C {target} rev-parse HEAD").read().strip()
        interpreter = _fake_interpreter(tmp_path, str((target / "src").resolve()))
        artifacts = tmp_path / "artifacts"

        def runner(argv, cwd, env):
            run_dir = Path(argv[argv.index("--output") + 1])
            if complete:
                _complete(run_dir, SEQUENCES)
            return exit_code, log

        import scripts.tools.capture_race_incidence as harness

        original = harness.TARGET_SOURCE_SHA
        harness.TARGET_SOURCE_SHA = head
        try:
            return execute_run(
                path="A",
                seq_index=0,
                slot_index=0,
                campaign_id="test",
                target=target,
                interpreter=interpreter,
                artifact_dir=artifacts,
                runner=runner,
            )
        finally:
            harness.TARGET_SOURCE_SHA = original


# --------------------------------------------------------------------------
# A2.2/A2.3: a setup-invalid must be replaced, not silently consume its slot
# --------------------------------------------------------------------------


def _scripted_executor(script):
    """Return an executor yielding the given verdicts, and its attempt log."""
    log: list[dict] = []
    verdicts = list(script)

    def executor(
        *, path, seq_index, slot_index, campaign_id, target, interpreter, artifact_dir
    ):
        verdict = verdicts.pop(0) if verdicts else "ok"
        record = {
            "schema": "capture_race_incidence_run_v1",
            "campaign_id": campaign_id,
            "seq_index": seq_index,
            "slot_index": slot_index,
            "path": path,
            "verdict": verdict,
            "invalid_reason": None,
            "capture_error_hits": [],
        }
        log.append(record)
        return record

    return executor, log


def test_a_setup_invalid_is_replaced_and_the_path_still_reaches_n(
    tmp_path: Path,
) -> None:
    """The P0 regression.

    §2 promises 100 *valid* runs per path and §4 promises a setup-invalid is
    replaced.  Reading the interleaving as 100 attempt-pairs breaks both: one
    legal setup-invalid on A caps A at 99 and the campaign lands on
    UNRESOLVED_INVALID_STUDY having never made the replacement it promised.
    """
    pairs = 4
    # Slot 0 path A fails setup once, then succeeds; everything else is fine.
    executor, log = _scripted_executor(["invalid", "ok"])
    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        artifact_dir=tmp_path,
        pairs=pairs,
        executor=executor,
    )

    assert campaign.invalid == 1
    assert campaign.valid == {"A": pairs, "B": pairs}, "the invalid run was replaced"
    assert terminal == "CLOSED_BOUNDED"
    # The extra attempt is an extra attempt, not an extra slot.
    assert len(log) == 2 * pairs + 1
    assert [r["seq_index"] for r in log] == list(range(2 * pairs + 1))
    # A2.2: valid runs still alternate strictly A,B,A,B,…
    kept = [r["path"] for r in log if r["verdict"] != "invalid"]
    assert kept == ["A", "B"] * pairs


def test_the_invalid_attempt_does_not_advance_the_slot(tmp_path: Path) -> None:
    executor, log = _scripted_executor(["invalid", "invalid", "ok"])
    campaign = Campaign.new()
    run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        artifact_dir=tmp_path,
        pairs=1,
        executor=executor,
    )
    assert [(r["slot_index"], r["path"]) for r in log[:3]] == [(0, "A")] * 3


def test_endless_setup_failure_is_a_validity_failure_not_a_spin(
    tmp_path: Path,
) -> None:
    """A2.3: retrying forever is not a terminal."""
    executor, log = _scripted_executor(["invalid"] * 50)
    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        artifact_dir=tmp_path,
        pairs=100,
        executor=executor,
    )
    assert terminal == "UNRESOLVED_INVALID_STUDY"
    assert len(log) == MAX_CONSECUTIVE_SETUP_INVALID
    assert "consecutive setup-invalid" in campaign.detail


def test_a_capture_failure_still_stops_immediately(tmp_path: Path) -> None:
    executor, log = _scripted_executor(["ok", "ok", "failure"])
    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        artifact_dir=tmp_path,
        pairs=100,
        executor=executor,
    )
    assert terminal == "FAILURE_OBSERVED_A"
    assert len(log) == 3, "no further attempts after the first §3 failure"


# --------------------------------------------------------------------------
# A2.4: a campaign-ending condition must leave a stamped terminal behind
# --------------------------------------------------------------------------


def test_a_harness_failure_stamps_the_manifest_and_logs_the_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The P1 regression.

    A manifest left at ``terminal: null`` is indistinguishable from a campaign
    still running, so a fail-closed abort that only prints to stderr loses the
    one durable record that says what happened.
    """
    import scripts.tools.capture_race_incidence as harness

    target = _git_target(tmp_path)
    monkeypatch.setattr(harness, "TARGET_SOURCE_SHA", _head(target))
    monkeypatch.setattr(harness, "N_PAIRS", 2)
    interpreter = _fake_interpreter(tmp_path, str((target / "src").resolve()))
    artifacts = tmp_path / "artifacts"

    def exploding_spawn(argv, cwd, env):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(harness, "_spawn", exploding_spawn)

    rc = harness.main(
        [
            "--target",
            str(target),
            "--interpreter",
            str(interpreter),
            "--artifact-dir",
            str(artifacts),
            "--pairs",
            "2",
        ]
    )

    assert rc == 1
    assert "EXECUTION_INVALID" in capsys.readouterr().err

    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["terminal"] == "EXECUTION_INVALID"
    assert "could not spawn the workload" in manifest["detail"]

    rows = [
        json.loads(line)
        for line in (artifacts / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1, "the interrupted attempt still gets a row"
    assert rows[0]["verdict"] == "execution_invalid"
    assert rows[0]["slot_index"] == 0
    assert rows[0]["path"] == "A"


def test_run_directory_creation_failure_is_execution_invalid(tmp_path: Path) -> None:
    """No filesystem failure may escape as a bare OSError (A2.4)."""
    blocker = tmp_path / "artifacts"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ExecutionInvalid, match="could not create run directories"):
        execute_run(
            path="A",
            seq_index=0,
            slot_index=0,
            campaign_id="test",
            target=tmp_path,
            interpreter=tmp_path / "py",
            artifact_dir=blocker,
            runner=lambda *a: (0, ""),
        )


def _head(target: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
