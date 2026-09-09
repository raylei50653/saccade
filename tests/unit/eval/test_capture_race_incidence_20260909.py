"""Harness self-verification for the 2026-09-09 #340 incidence campaign.

Document ↔ harness transcription plus fail-closed / attribution-ordering
paths.  The live GPU workload is never launched; ``execute_run`` takes an
injected runner.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
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

from scripts.tools.capture_race_incidence_20260909 import (  # noqa: E402
    ANALYZER_REL,
    ANALYZER_SOURCE_SHA256,
    BASE_ARGS,
    CAPTURE_FAILURE_SIGNATURES,
    COORDINATE_DECISION_SURFACE,
    COORDINATE_ENVIRONMENT,
    COORDINATE_IDENTITY_SEMANTICS,
    COORDINATE_IMPLEMENTATION,
    COORDINATE_RUNTIME_INPUTS,
    Campaign,
    ExecutionInvalid,
    MAX_CONSECUTIVE_SETUP_INVALID,
    N_PAIRS,
    OBSERVER_CONTROL_COMMIT,
    OBSERVER_CPP_REL,
    OBSERVER_CPP_SHA256,
    OBSERVER_SO_SHA256,
    PATH_EXTRA_ARGS,
    PATH_ORDER,
    PREREG_REL,
    PREREG_SEAL_COMMIT,
    PROBE_DIGEST,
    SCHEMA,
    SEAL_REL,
    SEQUENCES,
    TARGET_SOURCE_SHA,
    TERMINALS,
    WRAPPER_REL,
    ZERO_OF_N_BOUND,
    attribute_failure,
    blank_record,
    build_argv,
    capture_failure_hits,
    child_environment,
    classify,
    execute_run,
    main,
    preflight,
    run_campaign,
    run_paths,
    schedule,
    setup_failure_signature,
)

PREREG = (ROOT / PREREG_REL).read_text(encoding="utf-8")
SEAL = (ROOT / SEAL_REL).read_text(encoding="utf-8")
HARNESS_SRC = (
    ROOT / "scripts" / "tools" / "capture_race_incidence_20260909.py"
).read_text(encoding="utf-8")
HISTORICAL_SRC = (ROOT / "scripts" / "tools" / "capture_race_incidence.py").read_text(
    encoding="utf-8"
)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _complete(run_dir: Path, sequences) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        (run_dir / f"{seq}.txt").write_text(
            "1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8"
        )


def _fake_interpreter(tmp_path: Path, prints: str, *, rc: int = 0) -> Path:
    script = tmp_path / "fake-python"
    script.write_text(f'#!/bin/sh\necho "{prints}"\nexit {rc}\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _git_target(tmp_path: Path, *, dirty: bool = False) -> Path:
    import subprocess

    target = tmp_path / "target"
    if (target / ".git").is_dir():
        if dirty:
            (target / "src" / "saccade" / "__init__.py").write_text(
                "x", encoding="utf-8"
            )
        return target
    (target / "src" / "saccade").mkdir(parents=True)
    (target / "src" / "saccade" / "__init__.py").write_text("", encoding="utf-8")
    (target / "scripts" / "eval").mkdir(parents=True)
    (target / "scripts" / "eval" / "mot17.py").write_text("# target workload\n")
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


def _head(target: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _control_and_observer(tmp_path: Path) -> tuple[Path, Path]:
    control = tmp_path / "control"
    attr = control / "scripts" / "tools" / "capture_attribution"
    attr.mkdir(parents=True, exist_ok=True)
    (attr / "analyze.py").write_bytes((ROOT / ANALYZER_REL).read_bytes())
    (attr / "observer.cpp").write_bytes((ROOT / OBSERVER_CPP_REL).read_bytes())
    (attr / "run.py").write_text("# wrapper double\n", encoding="utf-8")
    observer = tmp_path / "observer.so"
    observer.write_bytes(b"fake-observer-so")
    return control, observer


def _pin(monkeypatch, target: Path, observer: Path) -> None:
    import scripts.tools.capture_race_incidence_20260909 as harness

    monkeypatch.setattr(harness, "TARGET_SOURCE_SHA", _head(target))
    monkeypatch.setattr(harness, "OBSERVER_SO_SHA256", _sha(observer))


def _write_wrapper_log(trace_dir: Path, log: str) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "stdout.log").write_text(log, encoding="utf-8")


def _runner(log: str, *, complete: bool, exit_code: int = 0, extra=None):
    def runner(argv, cwd, env):
        # argv: py wrapper --observer so --output trace -- mot17 ... --output run
        outputs = [i for i, item in enumerate(argv) if item == "--output"]
        trace_dir = Path(argv[outputs[0] + 1])
        run_dir = Path(argv[outputs[-1] + 1])
        _write_wrapper_log(trace_dir, log)
        if extra is not None:
            extra(trace_dir, run_dir, argv)
        if complete:
            _complete(run_dir, SEQUENCES)
        return exit_code, "wrapper-noise-must-not-decide-incidence\n"

    return runner


# --------------------------------------------------------------------------
# Document ↔ harness transcription
# --------------------------------------------------------------------------


def test_preregistration_and_seal_are_the_frozen_files() -> None:
    assert (ROOT / PREREG_REL).is_file()
    assert (ROOT / SEAL_REL).is_file()
    assert PREREG_REL in PREREG or "preregistration_20260909" in PREREG
    assert "capture_race_incidence_preregistration_20260909.md" in SEAL


def test_frozen_production_identities_match_prereg_and_seal() -> None:
    for value in (
        TARGET_SOURCE_SHA,
        COORDINATE_DECISION_SURFACE,
        COORDINATE_ENVIRONMENT,
        COORDINATE_IDENTITY_SEMANTICS,
        COORDINATE_IMPLEMENTATION,
        COORDINATE_RUNTIME_INPUTS,
        PROBE_DIGEST,
    ):
        assert value in PREREG
        assert value in SEAL


def test_frozen_observer_identities_are_not_the_production_implementation() -> None:
    assert OBSERVER_CONTROL_COMMIT in PREREG
    assert OBSERVER_CONTROL_COMMIT in SEAL
    assert ANALYZER_SOURCE_SHA256 in PREREG
    assert OBSERVER_CPP_SHA256 in PREREG
    assert OBSERVER_SO_SHA256 in PREREG
    assert OBSERVER_CONTROL_COMMIT != TARGET_SOURCE_SHA
    assert COORDINATE_IMPLEMENTATION != OBSERVER_CONTROL_COMMIT
    assert PREREG_SEAL_COMMIT in SEAL or PREREG_SEAL_COMMIT in HARNESS_SRC


def test_live_control_analyzer_and_observer_source_still_match_the_freeze() -> None:
    assert _sha(ROOT / ANALYZER_REL) == ANALYZER_SOURCE_SHA256
    assert _sha(ROOT / OBSERVER_CPP_REL) == OBSERVER_CPP_SHA256


def test_every_capture_signature_appears_in_the_preregistration() -> None:
    for code, signature in CAPTURE_FAILURE_SIGNATURES:
        assert signature in PREREG, f"{signature!r} is not in §3"
        assert code in PREREG


def test_workload_configuration_matches_the_preregistration() -> None:
    for token in BASE_ARGS:
        assert token in PREREG
    assert "--no-gpu-decode" in PATH_EXTRA_ARGS["B"]
    assert PATH_EXTRA_ARGS["A"] == ()
    match = re.search(r"`MOT17-([0-9/]+)-SDP`", PREREG)
    assert match
    declared = tuple(f"MOT17-{n}-SDP" for n in match.group(1).split("/"))
    assert declared == SEQUENCES


def test_n_interleaving_and_terminals_are_the_frozen_ones() -> None:
    assert N_PAIRS == 100
    assert "100" in PREREG
    assert ZERO_OF_N_BOUND in PREREG
    assert PATH_ORDER == ("A", "B")
    assert list(schedule(2)) == [(0, "A"), (0, "B"), (1, "A"), (1, "B")]
    for terminal in TERMINALS:
        assert terminal in PREREG


def test_record_schema_fields_are_all_declared() -> None:
    record = blank_record(
        campaign_id="x",
        seq_index=0,
        slot_index=0,
        path="A",
        run_dir=Path("/r"),
        trace_dir=Path("/t"),
    )
    assert record["schema"] == SCHEMA
    for field in record:
        assert field in PREREG, f"{field!r} is written but not declared in §10"


def test_harness_never_imports_saccade() -> None:
    for line in HARNESS_SRC.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import saccade")
        assert not stripped.startswith("from saccade")


def test_does_not_inherit_the_20260908_observer_prohibition() -> None:
    assert "FORBIDDEN_OBSERVER_ENV" in HISTORICAL_SRC
    assert "FORBIDDEN_OBSERVER_ENV" not in HARNESS_SRC
    assert "§8 forbids observers" not in HARNESS_SRC
    env = child_environment({"LD_PRELOAD": "/opt/nsight/libcupti.so", "CUPTI_X": "1"})
    assert env["LD_PRELOAD"] == "/opt/nsight/libcupti.so"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_historical_executor_is_untouched() -> None:
    assert "b649de68e36ad530ed883f579478ab656a238158" in HISTORICAL_SRC
    assert "FORBIDDEN_OBSERVER_ENV" in HISTORICAL_SRC
    assert PREREG_REL not in HISTORICAL_SRC


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
        ("cudaErrorStreamCaptureUnsupported", "900"),
        ("cudaErrorStreamCaptureInvalidated", "901"),
        ("operation failed due to a previous error during capture", "901"),
        (
            "RuntimeError: CUDA error: operation would make the legacy stream "
            "depend on a capturing blocking stream",
            "906",
        ),
        ("cudaErrorStreamCaptureImplicit", "906"),
        ("  File c10/cuda/CUDAStream.cpp: currentStreamCaptureStatusMayInitCtx", "906"),
    ],
)
def test_capture_predicate_matches_driver_and_wrapper_forms(
    line: str, code: str
) -> None:
    hits = capture_failure_hits(f"noise\n{line}\nmore noise\n")
    assert hits
    assert hits[0]["cuda_code"] == code
    assert hits[0]["line"] == 2


def test_capture_predicate_does_not_invent_strings() -> None:
    assert capture_failure_hits("cudaErrorUnknown\n") == []
    assert capture_failure_hits("CUPTI reported rc=901\n") == []
    assert "cudaErrorUnknown" not in {sig for _, sig in CAPTURE_FAILURE_SIGNATURES}


def test_exit_code_is_recorded_but_not_part_of_the_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record_via_runner(
        tmp_path, monkeypatch, exit_code=1, log="unrelated tantrum\n", complete=True
    )
    assert record["exit_code"] == 1
    assert record["verdict"] == "ok"

    hit = _record_via_runner(
        tmp_path,
        monkeypatch,
        exit_code=0,
        log="cudaErrorStreamCaptureImplicit\n",
        complete=True,
    )
    assert hit["exit_code"] == 0
    assert hit["verdict"] == "failure"


# --------------------------------------------------------------------------
# Table 4-1 / #375
# --------------------------------------------------------------------------


def test_engine_banner_plus_capture_failure_is_not_setup_invalid(
    tmp_path: Path,
) -> None:
    log = "\n".join(
        [
            "[MambaDetector] TRT MambaHead enabled: "
            "/home/ray/developer/ai/saccade/models/yolo/mamba_head_26m.engine",
            "RuntimeError: CUDA Error: operation failed due to a previous error "
            "during capture at src/tracking/tracker_gpu.cu:3553",
            "torch.AcceleratorError: CUDA error: operation failed due to a "
            "previous error during capture",
            "Search for `cudaErrorStreamCaptureInvalidated' in "
            "https://docs.nvidia.com/cuda/cuda-runtime-api/",
        ]
    )
    verdict, observations = classify(log, tmp_path, output_dir=str(tmp_path))
    assert observations["setup_failure_signature"] is None
    assert observations["sequence_execution_started"] is True
    assert verdict == "failure"


def test_engine_banner_plus_mid_run_oom_is_not_rerunnable_invalid(
    tmp_path: Path,
) -> None:
    log = (
        "[MambaDetector] TRT MambaHead enabled: "
        "/home/ray/developer/ai/saccade/models/yolo/mamba_head_26m.engine\n"
        "torch.OutOfMemoryError: CUDA out of memory\n"
    )
    verdict, observations = classify(log, tmp_path, output_dir=str(tmp_path))
    assert observations["setup_failure_signature"] is None
    assert observations["sequence_execution_started"] is True
    assert verdict == "execution_invalid"


def test_true_setup_local_tensorrt_failure_is_invalid(tmp_path: Path) -> None:
    verdict, observations = classify(
        "RuntimeError: TensorRT engine build failed\n",
        tmp_path,
        output_dir=str(tmp_path),
    )
    assert observations["setup_failure_signature"] == "tensorrt"
    assert observations["sequence_execution_started"] is False
    assert verdict == "invalid"


def test_ambiguous_setup_log_is_not_rerunnable(tmp_path: Path) -> None:
    verdict, observations = classify(
        "Segmentation fault (core dumped)\n", tmp_path, output_dir=str(tmp_path)
    )
    assert observations["setup_failure_signature"] is None
    assert observations["sequence_execution_started"] is True
    assert verdict == "execution_invalid"


def test_post_start_non_primary_abort_is_execution_invalid(tmp_path: Path) -> None:
    _complete(tmp_path, SEQUENCES[:3])
    log = "\N{CLAPPER BOARD} MOT17-05-SDP [400/837]\ntorch.OutOfMemoryError\n"
    verdict, observations = classify(log, tmp_path, output_dir=str(tmp_path))
    assert observations["sequence_execution_started"] is True
    assert verdict == "execution_invalid"


def test_setup_signature_needs_both_terms_on_the_same_line() -> None:
    assert setup_failure_signature("TensorRT engine ready", "/out") is None
    assert (
        setup_failure_signature("RuntimeError: TensorRT engine build failed\n", "/out")
        == "tensorrt"
    )


# --------------------------------------------------------------------------
# Overlay argv and incidence/CUPTI isolation
# --------------------------------------------------------------------------


def test_argv_uses_control_wrapper_and_target_workload(tmp_path: Path) -> None:
    argv = build_argv(
        "B",
        target=tmp_path / "target",
        interpreter=tmp_path / "py",
        control=tmp_path / "control",
        observer=tmp_path / "observer.so",
        run_dir=tmp_path / "out",
        trace_dir=tmp_path / "trace",
    )
    assert argv[1] == str((tmp_path / "control") / WRAPPER_REL)
    assert "--observer" in argv
    assert str(tmp_path / "observer.so") in argv
    assert str(tmp_path / "target" / "scripts" / "eval" / "mot17.py") in argv
    assert "--no-gpu-decode" in argv
    assert argv[-2:] == ["--output", str(tmp_path / "out")]


def test_cupti_901_without_workload_log_is_not_incidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def extra(trace_dir, run_dir, argv):
        (trace_dir / "cuda.jsonl").write_text(
            json.dumps({"api": "cuStreamEndCapture", "rc": 901}) + "\n"
        )

    record = _record_via_runner(
        tmp_path,
        monkeypatch,
        exit_code=0,
        log="healthy run, no CUDA capture error\n",
        complete=True,
        extra=extra,
    )
    assert record["verdict"] == "ok"
    assert record["capture_error_hits"] == []


def test_workload_901_without_observer_data_is_incidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record_via_runner(
        tmp_path,
        monkeypatch,
        exit_code=0,
        log="cudaErrorStreamCaptureInvalidated\n",
        complete=False,
    )
    assert record["verdict"] == "failure"
    assert record["attribution_status"] == "preserved"


# --------------------------------------------------------------------------
# Attribution ordering
# --------------------------------------------------------------------------


def test_attribution_runs_after_incidence_is_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    artifacts = tmp_path / "artifacts"

    def executor(**kwargs):
        order.append("execute")
        run_dir, _log, trace_dir, sealed = run_paths(artifacts, 0, "A")
        record = blank_record(
            campaign_id="test",
            seq_index=0,
            slot_index=0,
            path="A",
            run_dir=run_dir,
            trace_dir=trace_dir,
        )
        record["verdict"] = "failure"
        record["capture_error_hits"] = [
            {
                "cuda_code": "906",
                "signature": "cudaErrorStreamCaptureImplicit",
                "line": 1,
            }
        ]
        record["attribution_status"] = "preserved"
        order.append(f"sealed_exists_before_attr:{sealed.is_file()}")
        return record

    def attributor(control_root, trace_dir):
        order.append("attribute")
        sealed = artifacts / "sealed" / "0000-A.json"
        assert sealed.is_file(), "incidence must be on disk before attribution"
        body = json.loads(sealed.read_text(encoding="utf-8"))
        assert body["verdict"] == "failure"
        order.append("saw_failure_on_disk")
        raise RuntimeError("analyzer crashed")

    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=artifacts,
        pairs=1,
        executor=executor,
        attributor=attributor,
    )
    assert terminal == "FAILURE_OBSERVED_A"
    assert order[0] == "execute"
    assert "attribute" in order
    assert order.index("execute") < order.index("attribute")
    rows = [
        json.loads(line)
        for line in (artifacts / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["verdict"] == "failure"
    assert rows[0]["attribution_status"] == "analysis_incomplete"


def test_analyzer_exception_cannot_rewrite_a_906_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "verdict": "failure",
        "capture_error_hits": [
            {
                "cuda_code": "906",
                "signature": "cudaErrorStreamCaptureImplicit",
                "line": 1,
            }
        ],
        "trace_dir": str(tmp_path),
        "attribution_status": "preserved",
        "attribution_report": None,
    }

    def boom(control, trace_dir):
        record["verdict"] = "ok"
        record["capture_error_hits"] = []
        raise RuntimeError("parse failed")

    attribute_failure(record, control=tmp_path, attributor=boom)
    assert record["verdict"] == "failure"
    assert record["capture_error_hits"][0]["cuda_code"] == "906"
    assert record["attribution_status"] == "analysis_incomplete"


def test_successful_attribution_cannot_clear_incidence() -> None:
    record = {
        "verdict": "failure",
        "capture_error_hits": [
            {
                "cuda_code": "901",
                "signature": "cudaErrorStreamCaptureInvalidated",
                "line": 2,
            }
        ],
        "trace_dir": "/t",
        "attribution_status": "preserved",
        "attribution_report": None,
    }

    def fake(control, trace_dir):
        return {
            "trace_structure_ok": True,
            "ownership_evidence_ok": True,
            "problems": [],
            "evidence_gaps": ["symbol_unknown"],
            "captures": [],
            "capture_errors": [],
            "stream_lifetimes": [],
            "event_edges": [],
        }

    attribute_failure(record, control=Path("/control"), attributor=fake)
    assert record["verdict"] == "failure"
    assert record["attribution_status"] == "analyzed"
    assert record["attribution_report"]["evidence_gaps"] == ["symbol_unknown"]


# --------------------------------------------------------------------------
# Preflight poison
# --------------------------------------------------------------------------


def test_wrong_target_sha_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    import scripts.tools.capture_race_incidence_20260909 as harness

    monkeypatch.setattr(harness, "TARGET_SOURCE_SHA", "0" * 40)
    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner("", complete=True),
    )
    assert record["verdict"] == "execution_invalid"
    assert "check 1" in record["invalid_reason"]
    assert record["target_head_observed"] is not None


def test_dirty_target_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _git_target(tmp_path, dirty=True)
    control, observer = _control_and_observer(tmp_path)
    interpreter = _fake_interpreter(tmp_path, str((target / "src").resolve()))
    _pin(monkeypatch, target, observer)
    checks = preflight(target, interpreter, observer=observer, control=control)
    assert checks.clean is False
    assert any("check 2" in failure for failure in checks.failures)


def test_editable_install_pointing_at_control_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, _interpreter = _prepared(tmp_path, monkeypatch)
    wrong = "/home/someone/control-checkout/src"
    interpreter = _fake_interpreter(tmp_path, wrong)
    launched: list[list[str]] = []

    def runner(argv, cwd, env):
        launched.append(list(argv))
        return 0, ""

    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=tmp_path / "artifacts",
        runner=runner,
    )
    assert launched == []
    assert record["verdict"] == "execution_invalid"
    assert record["saccade_import_root"] == wrong
    assert "editable install points outside" in record["invalid_reason"]


def test_observer_hash_mismatch_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    import scripts.tools.capture_race_incidence_20260909 as harness

    monkeypatch.setattr(harness, "OBSERVER_SO_SHA256", "ab" * 32)
    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner("", complete=True),
    )
    assert record["verdict"] == "execution_invalid"
    assert "observer.so SHA256" in record["invalid_reason"]
    assert record["observer_sha256_observed"] == _sha(observer)


def test_observer_source_hash_mismatch_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    import scripts.tools.capture_race_incidence_20260909 as harness

    monkeypatch.setattr(harness, "OBSERVER_CPP_SHA256", "ef" * 32)
    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner("", complete=True),
    )
    assert record["verdict"] == "execution_invalid"
    assert "observer.cpp SHA256" in record["invalid_reason"]


def test_analyzer_hash_mismatch_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    import scripts.tools.capture_race_incidence_20260909 as harness

    monkeypatch.setattr(harness, "ANALYZER_SOURCE_SHA256", "cd" * 32)
    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner("", complete=True),
    )
    assert record["verdict"] == "execution_invalid"
    assert "analyzer SHA256" in record["invalid_reason"]


def test_missing_observer_overlay_is_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    missing = tmp_path / "no-such-observer.so"
    record = execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=missing,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner("", complete=True),
    )
    assert record["verdict"] == "execution_invalid"
    assert "observer overlay is missing" in record["invalid_reason"]
    assert "table 4-1" in record["invalid_reason"]


def test_missing_wrapper_log_is_execution_invalid_not_setup_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)

    def runner(argv, cwd, env):
        return 0, "child stdout without wrapper files\n"

    with pytest.raises(ExecutionInvalid, match="did not retain workload"):
        execute_run(
            path="A",
            seq_index=0,
            slot_index=0,
            campaign_id="test",
            target=target,
            interpreter=interpreter,
            control=control,
            observer=observer,
            artifact_dir=tmp_path / "artifacts",
            runner=runner,
        )


# --------------------------------------------------------------------------
# Campaign counters / stop rule
# --------------------------------------------------------------------------


def test_first_capture_failure_stops_both_paths(tmp_path: Path) -> None:
    executor, log = _scripted_executor(["ok", "failure"])
    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        control=tmp_path,
        observer=tmp_path / "o.so",
        artifact_dir=tmp_path,
        pairs=100,
        executor=executor,
        attributor=lambda *a, **k: {"problems": [], "evidence_gaps": []},
    )
    assert terminal == "FAILURE_OBSERVED_B"
    assert len(log) == 2


def test_invalid_runs_are_replaced_and_do_not_count(tmp_path: Path) -> None:
    executor, log = _scripted_executor(["invalid", "ok"])
    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        control=tmp_path,
        observer=tmp_path / "o.so",
        artifact_dir=tmp_path,
        pairs=1,
        executor=executor,
    )
    assert campaign.invalid == 1
    assert campaign.valid == {"A": 1, "B": 1}
    assert terminal == "CLOSED_BOUNDED"
    kept = [r["path"] for r in log if r["verdict"] != "invalid"]
    assert kept == ["A", "B"]


def test_endless_setup_failure_is_validity_failure(tmp_path: Path) -> None:
    executor, log = _scripted_executor(["invalid"] * 50)
    campaign = Campaign.new()
    terminal = run_campaign(
        campaign=campaign,
        campaign_id="test",
        target=tmp_path,
        interpreter=tmp_path / "py",
        control=tmp_path,
        observer=tmp_path / "o.so",
        artifact_dir=tmp_path,
        pairs=100,
        executor=executor,
    )
    assert terminal == "UNRESOLVED_INVALID_STUDY"
    assert len(log) == MAX_CONSECUTIVE_SETUP_INVALID


def test_pairs_may_not_be_lowered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "--target",
            str(tmp_path),
            "--observer",
            str(tmp_path / "o.so"),
            "--pairs",
            "3",
        ]
    )
    assert rc == 1
    assert "freezes N" in capsys.readouterr().err


def test_preflight_only_writes_no_campaign_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    artifacts = tmp_path / "artifacts"
    rc = main(
        [
            "--target",
            str(target),
            "--interpreter",
            str(interpreter),
            "--control",
            str(control),
            "--observer",
            str(observer),
            "--artifact-dir",
            str(artifacts),
            "--preflight-only",
        ]
    )
    assert rc == 0
    assert not (artifacts / "manifest.json").exists()


def test_unwritable_manifest_is_execution_invalid(tmp_path: Path) -> None:
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
            control=tmp_path,
            observer=tmp_path / "o.so",
            artifact_dir=blocker,
            runner=lambda *a: (0, ""),
        )


def test_post_start_non_primary_run_invalidates_the_campaign(tmp_path: Path) -> None:
    campaign = Campaign.new()
    campaign.absorb(
        {
            "path": "A",
            "verdict": "execution_invalid",
            "invalid_reason": "mid-run OOM",
        }
    )
    assert campaign.terminal == "EXECUTION_INVALID"
    assert campaign.finalise(100) == "EXECUTION_INVALID"


def test_a_short_campaign_is_never_closed_bounded() -> None:
    campaign = Campaign.new()
    for path in PATH_ORDER:
        campaign.valid[path] = 99
    assert campaign.finalise(100) == "UNRESOLVED_INVALID_STUDY"


def test_harness_failure_stamps_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.tools.capture_race_incidence_20260909 as harness

    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(harness, "N_PAIRS", 2)
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
            "--control",
            str(control),
            "--observer",
            str(observer),
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
    assert manifest["target_source_sha"] == harness.TARGET_SOURCE_SHA
    assert manifest["observer_control_commit"] == OBSERVER_CONTROL_COMMIT
    assert manifest["preregistration"] == PREREG_REL


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _prepared(tmp_path: Path, monkeypatch):
    target = _git_target(tmp_path)
    control, observer = _control_and_observer(tmp_path)
    interpreter = _fake_interpreter(tmp_path, str((target / "src").resolve()))
    _pin(monkeypatch, target, observer)
    return target, control, observer, interpreter


def _record_via_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_code: int,
    log: str,
    complete: bool,
    extra=None,
) -> dict:
    target, control, observer, interpreter = _prepared(tmp_path, monkeypatch)
    return execute_run(
        path="A",
        seq_index=0,
        slot_index=0,
        campaign_id="test",
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner(log, complete=complete, exit_code=exit_code, extra=extra),
    )


def _scripted_executor(script):
    log: list[dict] = []
    verdicts = list(script)

    def executor(
        *,
        path,
        seq_index,
        slot_index,
        campaign_id,
        target,
        interpreter,
        control,
        observer,
        artifact_dir,
    ):
        verdict = verdicts.pop(0) if verdicts else "ok"
        record = blank_record(
            campaign_id=campaign_id,
            seq_index=seq_index,
            slot_index=slot_index,
            path=path,
            run_dir=Path("/r"),
            trace_dir=Path("/t"),
        )
        record["verdict"] = verdict
        record["invalid_reason"] = None
        record["capture_error_hits"] = (
            [{"cuda_code": "901", "signature": "x", "line": 1}]
            if verdict == "failure"
            else []
        )
        if verdict == "failure":
            record["attribution_status"] = "preserved"
        log.append(record)
        return record

    return executor, log
