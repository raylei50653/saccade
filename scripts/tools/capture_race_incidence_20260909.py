#!/usr/bin/env python3
"""Run the 2026-09-09 #340 capture-race incidence campaign under its frozen preregistration.

The preregistration is
``docs/research/pipeline/capture_race_incidence_preregistration_20260909.md``.
This module executes that document and nothing else.  Every threshold, string,
count and terminal is transcribed from it, and the transcription is asserted
against the document by
``tests/unit/eval/test_capture_race_incidence_20260909.py``.  It makes no
statistical decision of its own.

This is a new campaign executor.  It does not replace
``scripts/tools/capture_race_incidence.py``, which remains the 20260908
historical executor.

**This module must never import saccade.**  Standard library only.  The
analyzer is invoked as a subprocess after incidence is sealed, so a control-side
import cannot leak into the measured target.

Fail-closed everywhere.  Observer/analyzer output never decides the incidence
bit.
"""
# status: experiment

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

PREREG_REL = (
    "docs/research/pipeline/closed/capture_race_incidence_preregistration_20260909.md"
)
SEAL_REL = "docs/research/pipeline/closed/capture_race_incidence_preregistration_20260909_seal.md"
SCHEMA = "capture_race_incidence_run_v1"
CAMPAIGN_SCHEMA = "capture_race_incidence_campaign_v1"

# --- §2.1 production target / coordinate -----------------------------------
TARGET_SOURCE_SHA = "4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f"
COORDINATE_DECISION_SURFACE = (
    "9b7faeb0f76a43483a924ac4028361bb155373027e9fe44e4cffbd5f1a2b0369"
)
COORDINATE_ENVIRONMENT = (
    "df4c89b6aae2c555aaac108bb8fbaf835aa21f1057c03e25ea7e26d009d5b091"
)
COORDINATE_IDENTITY_SEMANTICS = (
    "8fc9bd85dd651791ec552dd2d6ab0244e849d213dcee7a2b77dfb9246a7969ca"
)
COORDINATE_IMPLEMENTATION = (
    "2f69ac56e8cbfeb41300d479b2910ad6d750391b7099bb1af649956fc209d05d"
)
COORDINATE_RUNTIME_INPUTS = (
    "0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5"
)
PROBE_DIGEST = "2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5"

# --- §2.2 observer/control freeze (not a production target) ----------------
OBSERVER_CONTROL_COMMIT = "276d8d744d7050ad272f9b69cf3c60b0c31333a5"
PREREG_SEAL_COMMIT = "72adf71e6066b5f54abcbb9f263a1f8ae427b980"
ANALYZER_SOURCE_SHA256 = (
    "6d72a6107fedc739370489b13f6699f08e06a9e2ea4655e6836e127b9fe1fb42"
)
OBSERVER_CPP_SHA256 = "4a20104185394a565014f83fd2f6993e33aa4c95a6775ebe15e91017140c592d"
OBSERVER_SO_SHA256 = "e8b83ec4866c82f4e5a6ddbb64e5b207ff65dfc031db0101078f72083da30f29"
WRAPPER_REL = Path("scripts") / "tools" / "capture_attribution" / "run.py"
ANALYZER_REL = Path("scripts") / "tools" / "capture_attribution" / "analyze.py"
OBSERVER_CPP_REL = Path("scripts") / "tools" / "capture_attribution" / "observer.cpp"

# --- §2.3 / §2.4 workload --------------------------------------------------
BASE_ARGS: tuple[str, ...] = (
    "--preset",
    "mamba_whole_graph_m",
    "--detector",
    "SDP",
    "--double-buffer",
)
PATH_EXTRA_ARGS: dict[str, tuple[str, ...]] = {"A": (), "B": ("--no-gpu-decode",)}
PATH_ORDER: tuple[str, ...] = ("A", "B")
N_PAIRS = 100
MAX_CONSECUTIVE_SETUP_INVALID = 5
SEQUENCES: tuple[str, ...] = (
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
)
ZERO_OF_N_BOUND = "1 - 0.05^(1/100) = 0.029513 ≈ 2.95%"
TERMINALS: tuple[str, ...] = (
    "CLOSED_BOUNDED",
    "FAILURE_OBSERVED_A",
    "FAILURE_OBSERVED_B",
    "UNRESOLVED_INVALID_STUDY",
    "EXECUTION_INVALID",
)

# --- §3 primary failure predicate ------------------------------------------
# Substring matches, case-sensitive.  Driver messages are substrings of their
# torch wrapper forms, so the core message covers both rows of the §3 table.
# CUPTI rows and analyzer reports are not inputs.
CAPTURE_FAILURE_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("900", "cudaErrorStreamCaptureUnsupported"),
    ("900", "operation not permitted when stream is capturing"),
    ("901", "cudaErrorStreamCaptureInvalidated"),
    ("901", "operation failed due to a previous error during capture"),
    ("906", "cudaErrorStreamCaptureImplicit"),
    ("906", "legacy stream depend on a capturing blocking stream"),
    ("906", "currentStreamCaptureStatusMayInitCtx"),
)

# --- §4 / table 4-1 --------------------------------------------------------
PROGRESS_MARKER = "\N{CLAPPER BOARD} "

# Same-line, case-insensitive.  Whole-log matching is fail-open (#375).
SETUP_FAILURE_SIGNATURES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "dataset",
        ("seqinfo.ini", "data_root", "MOT17-"),
        ("no such file", "filenotfounderror", "not found", "does not exist"),
    ),
    (
        "weights",
        ("checkpoint", "state_dict", ".ckpt", ".pth"),
        ("no such file", "filenotfounderror", "not found"),
    ),
    (
        "tensorrt",
        ("tensorrt", "trtexec", "engine build", ".engine"),
        ("failed", "error", "exception"),
    ),
    (
        "cuda_device",
        (),
        (
            "no cuda-capable device is detected",
            "cuda driver version is insufficient",
            "cuda unknown error",
            "found no nvidia driver",
        ),
    ),
    ("output_dir", (), ("permission denied", "read-only file system")),
)

IMPORT_ROOT_PROBE = (
    "import os, saccade; "
    "print(os.path.dirname(os.path.dirname(os.path.abspath(saccade.__file__))))"
)

ATTRIBUTION_STATUSES: tuple[str, ...] = (
    "not_applicable",
    "preserved",
    "analyzed",
    "analysis_incomplete",
)


class ExecutionInvalid(RuntimeError):
    """Campaign-ending condition (§7 EXECUTION_INVALID).  Never per-run."""

    def __init__(self, message: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class Preflight:
    """Observed values for the §2 pre-run checks.

    Recorded whether or not the checks pass: a fail-closed stop with no record
    is not evidence that anything was checked.
    """

    head: str | None
    clean: bool | None
    import_root: str | None
    observer_sha256: str | None
    observer_source_sha256: str | None
    analyzer_sha256: str | None
    overlay_path: str | None
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(target: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ("git", "-C", str(target), *args),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ExecutionInvalid(f"could not run git in {target}: {exc}") from exc
    if proc.returncode != 0:
        raise ExecutionInvalid(
            f"git {' '.join(args)} failed in {target}: {proc.stderr.strip()}"
        )
    return proc.stdout


def preflight(
    target: Path,
    interpreter: Path,
    *,
    observer: Path,
    control: Path,
) -> Preflight:
    """Run the frozen identity checks.  Never raises for a check failure."""
    failures: list[str] = []
    head: str | None = None
    clean: bool | None = None
    import_root: str | None = None
    observer_sha256: str | None = None
    observer_source_sha256: str | None = None
    analyzer_sha256: str | None = None
    overlay_path: str | None = None

    try:
        head = _git(target, "rev-parse", "HEAD").strip()
    except ExecutionInvalid as exc:
        failures.append(str(exc))
    else:
        if head != TARGET_SOURCE_SHA:
            failures.append(
                f"check 1: target HEAD is {head}, expected {TARGET_SOURCE_SHA}"
            )

    try:
        porcelain = _git(target, "status", "--porcelain")
    except ExecutionInvalid as exc:
        failures.append(str(exc))
    else:
        clean = porcelain.strip() == ""
        if not clean:
            failures.append(
                "check 2: target worktree is not clean: "
                + " | ".join(porcelain.strip().splitlines()[:5])
            )

    proc = None
    try:
        proc = subprocess.run(
            (str(interpreter), "-c", IMPORT_ROOT_PROBE),
            capture_output=True,
            text=True,
            check=False,
            cwd=str(target),
        )
    except OSError as exc:
        failures.append(f"check 3: could not start the target interpreter: {exc}")

    if proc is None:
        pass
    elif proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "no stderr"
        failures.append(
            f"check 3: could not resolve saccade in the target interpreter: {detail}"
        )
    else:
        import_root = proc.stdout.strip()
        expected = target.resolve() / "src"
        if Path(import_root).resolve() != expected:
            failures.append(
                f"check 3: saccade imports from {import_root}, expected {expected} "
                "— the venv's editable install points outside the target worktree, "
                "so the pin measures the wrong source"
            )

    overlay = observer.resolve()
    overlay_path = str(overlay)
    if not overlay.is_file():
        failures.append(
            f"check 4: observer overlay is missing: {overlay} "
            "(not a table 4-1 setup-invalid)"
        )
    else:
        try:
            observer_sha256 = _sha256_file(overlay)
        except OSError as exc:
            failures.append(f"check 4: could not hash observer overlay: {exc}")
        else:
            if observer_sha256 != OBSERVER_SO_SHA256:
                failures.append(
                    f"check 4: observer.so SHA256 is {observer_sha256}, "
                    f"expected {OBSERVER_SO_SHA256}"
                )

    observer_cpp = (control / OBSERVER_CPP_REL).resolve()
    if not observer_cpp.is_file():
        failures.append(f"check 4: observer.cpp is missing: {observer_cpp}")
    else:
        try:
            observer_source_sha256 = _sha256_file(observer_cpp)
        except OSError as exc:
            failures.append(f"check 4: could not hash observer.cpp: {exc}")
        else:
            if observer_source_sha256 != OBSERVER_CPP_SHA256:
                failures.append(
                    f"check 4: observer.cpp SHA256 is {observer_source_sha256}, "
                    f"expected {OBSERVER_CPP_SHA256}"
                )

    analyzer = (control / ANALYZER_REL).resolve()
    if not analyzer.is_file():
        failures.append(f"check 4: analyzer is missing: {analyzer}")
    else:
        try:
            analyzer_sha256 = _sha256_file(analyzer)
        except OSError as exc:
            failures.append(f"check 4: could not hash analyzer: {exc}")
        else:
            if analyzer_sha256 != ANALYZER_SOURCE_SHA256:
                failures.append(
                    f"check 4: analyzer SHA256 is {analyzer_sha256}, "
                    f"expected {ANALYZER_SOURCE_SHA256}"
                )

    wrapper = (control / WRAPPER_REL).resolve()
    if not wrapper.is_file():
        failures.append(f"check 4: spawn wrapper is missing: {wrapper}")

    return Preflight(
        head=head,
        clean=clean,
        import_root=import_root,
        observer_sha256=observer_sha256,
        observer_source_sha256=observer_source_sha256,
        analyzer_sha256=analyzer_sha256,
        overlay_path=overlay_path,
        failures=tuple(failures),
    )


def run_paths(
    artifact_dir: Path, seq_index: int, path: str
) -> tuple[Path, Path, Path, Path]:
    """run_dir, log_path, trace_dir, sealed_path."""
    stem = f"{seq_index:04d}-{path}"
    return (
        artifact_dir / "runs" / stem,
        artifact_dir / "logs" / f"{stem}.log",
        artifact_dir / "traces" / stem,
        artifact_dir / "sealed" / f"{stem}.json",
    )


def blank_record(
    *,
    campaign_id: str,
    seq_index: int,
    slot_index: int,
    path: str,
    run_dir: Path,
    trace_dir: Path,
) -> dict[str, Any]:
    """The one ``capture_race_incidence_run_v1`` row shape (§10)."""
    return {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "seq_index": seq_index,
        "slot_index": slot_index,
        "path": path,
        "run_dir": str(run_dir),
        "trace_dir": str(trace_dir),
        "started_utc": _utc(),
        "finished_utc": None,
        "argv": None,
        "exit_code": None,
        "sequence_execution_started": False,
        "sequences_completed": [],
        "capture_error_hits": [],
        "progress_marker_seen": False,
        "progress_markers": 0,
        "setup_failure_signature": None,
        "verdict": "execution_invalid",
        "invalid_reason": None,
        "log_sha256": None,
        "target_source_sha": TARGET_SOURCE_SHA,
        "target_head_observed": None,
        "target_worktree_clean": None,
        "saccade_import_root": None,
        "coordinate_decision_surface": COORDINATE_DECISION_SURFACE,
        "coordinate_environment": COORDINATE_ENVIRONMENT,
        "coordinate_identity_semantics": COORDINATE_IDENTITY_SEMANTICS,
        "coordinate_implementation": COORDINATE_IMPLEMENTATION,
        "coordinate_runtime_inputs": COORDINATE_RUNTIME_INPUTS,
        "probe_digest": PROBE_DIGEST,
        "observer_control_commit": OBSERVER_CONTROL_COMMIT,
        "observer_sha256_observed": None,
        "analyzer_source_sha256": ANALYZER_SOURCE_SHA256,
        "attribution_status": "not_applicable",
        "attribution_report": None,
    }


def _abort(record: dict[str, Any], reason: str) -> dict[str, Any]:
    record["verdict"] = "execution_invalid"
    record["invalid_reason"] = reason
    record["finished_utc"] = _utc()
    return record


def capture_failure_hits(log: str) -> list[dict[str, Any]]:
    """§3 predicate.  Exit code, CUPTI, and analyzer output are not inputs."""
    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(log.splitlines(), start=1):
        for code, signature in CAPTURE_FAILURE_SIGNATURES:
            if signature in line:
                hits.append({"cuda_code": code, "signature": signature, "line": lineno})
    return hits


def setup_failure_signature(log: str, output_dir: str) -> str | None:
    """Table 4-1.  Context and failure terms must co-occur on the same line."""
    output_dir_l = output_dir.lower()
    for category, contexts, failures in SETUP_FAILURE_SIGNATURES:
        local_contexts = (output_dir_l,) if category == "output_dir" else contexts
        for line in log.splitlines():
            haystack = line.lower()
            if not any(term in haystack for term in failures):
                continue
            if not local_contexts or any(term in haystack for term in local_contexts):
                return category
    return None


def sequences_completed(run_dir: Path) -> list[str]:
    return [seq for seq in SEQUENCES if (run_dir / f"{seq}.txt").is_file()]


def classify(log: str, run_dir: Path, *, output_dir: str) -> tuple[str, dict[str, Any]]:
    """Return (verdict, observations) from the wrapper-retained workload log.

    Ordering follows the preregistration: §3 first, then the §4 boundary.
    Analyzer / CUPTI files are not consulted.
    """
    hits = capture_failure_hits(log)
    completed = sequences_completed(run_dir)
    markers = sum(1 for line in log.splitlines() if line.startswith(PROGRESS_MARKER))
    signature = setup_failure_signature(log, output_dir)

    started = True
    if markers == 0 and not completed and signature is not None:
        started = False

    observations: dict[str, Any] = {
        "sequence_execution_started": started,
        "sequences_completed": completed,
        "capture_error_hits": hits,
        "progress_marker_seen": markers > 0,
        "progress_markers": markers,
        "setup_failure_signature": signature,
    }

    if hits:
        return "failure", observations
    if not started:
        return "invalid", observations
    if len(completed) != len(SEQUENCES):
        return "execution_invalid", observations
    return "ok", observations


def build_argv(
    path: str,
    *,
    target: Path,
    interpreter: Path,
    control: Path,
    observer: Path,
    run_dir: Path,
    trace_dir: Path,
) -> list[str]:
    if path not in PATH_EXTRA_ARGS:
        raise ExecutionInvalid(f"unknown path {path!r}")
    wrapper = control / WRAPPER_REL
    mot17 = target / "scripts" / "eval" / "mot17.py"
    argv = [
        str(interpreter),
        str(wrapper),
        "--observer",
        str(observer),
        "--output",
        str(trace_dir),
        "--",
        str(mot17),
        *BASE_ARGS,
        *PATH_EXTRA_ARGS[path],
        "--output",
        str(run_dir),
    ]
    if "--observer" not in argv or str(observer) not in argv:
        raise ExecutionInvalid("observer overlay missing from planned argv")
    if str(mot17) not in argv:
        raise ExecutionInvalid("workload argv does not point at the target mot17.py")
    if str(wrapper) not in argv:
        raise ExecutionInvalid("spawn wrapper missing from planned argv")
    return argv


def child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Workload environment.  Observer overlay is the frozen instrument.

    20260908 §8 observer prohibition is not inherited.  PYTHONUNBUFFERED is
    buffering, not a numerator expansion.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def wrapper_retained_log(trace_dir: Path) -> str | None:
    """Workload stdout/stderr retained by the spawn wrapper, not cuda.jsonl."""
    stdout = trace_dir / "stdout.log"
    stderr = trace_dir / "stderr.log"
    if not stdout.is_file() and not stderr.is_file():
        return None
    parts: list[str] = []
    if stdout.is_file():
        parts.append(stdout.read_text(encoding="utf-8", errors="replace"))
    if stderr.is_file():
        parts.append(stderr.read_text(encoding="utf-8", errors="replace"))
    return "".join(parts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise ExecutionInvalid(f"could not write {path}: {exc}") from exc


def _write_record(runs_jsonl: Path, record: dict[str, Any]) -> None:
    try:
        with runs_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ExecutionInvalid(f"could not append to {runs_jsonl}: {exc}") from exc


def seal_incidence(path: Path, record: dict[str, Any]) -> None:
    """Durable incidence bit.  Must happen before attribution runs."""
    _write_json(path, record)


def execute_run(
    *,
    path: str,
    seq_index: int,
    slot_index: int,
    campaign_id: str,
    target: Path,
    interpreter: Path,
    control: Path,
    observer: Path,
    artifact_dir: Path,
    runner: Any = None,
) -> dict[str, Any]:
    """Run one trial and return its §10 record with incidence already sealed.

    Does not invoke the analyzer.  Attribution is a later step on the same
    record and cannot change ``verdict``.
    """
    run_dir, log_path, trace_dir, _sealed = run_paths(artifact_dir, seq_index, path)
    record = blank_record(
        campaign_id=campaign_id,
        seq_index=seq_index,
        slot_index=slot_index,
        path=path,
        run_dir=run_dir,
        trace_dir=trace_dir,
    )

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        trace_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        reason = f"could not create run directories: {exc}"
        raise ExecutionInvalid(reason, _abort(record, reason)) from exc

    checks = preflight(target, interpreter, observer=observer, control=control)
    argv = build_argv(
        path,
        target=target,
        interpreter=interpreter,
        control=control,
        observer=observer,
        run_dir=run_dir,
        trace_dir=trace_dir,
    )
    record["argv"] = argv
    record["target_head_observed"] = checks.head
    record["target_worktree_clean"] = checks.clean
    record["saccade_import_root"] = checks.import_root
    record["observer_sha256_observed"] = checks.observer_sha256

    if not checks.ok:
        record["invalid_reason"] = "; ".join(checks.failures)
        record["finished_utc"] = _utc()
        return record

    env = child_environment()
    run = runner if runner is not None else _spawn
    try:
        exit_code, _ignored = run(argv, target, env)
    except OSError as exc:
        reason = f"could not spawn the workload: {exc}"
        raise ExecutionInvalid(reason, _abort(record, reason)) from exc

    log = wrapper_retained_log(trace_dir)
    if log is None:
        reason = (
            "spawn wrapper did not retain workload stdout/stderr "
            f"under {trace_dir} (not a table 4-1 setup-invalid)"
        )
        raise ExecutionInvalid(reason, _abort(record, reason))

    try:
        log_path.write_text(log, encoding="utf-8")
    except OSError as exc:
        reason = f"could not write {log_path}: {exc}"
        raise ExecutionInvalid(reason, _abort(record, reason)) from exc

    verdict, observations = classify(log, run_dir, output_dir=str(run_dir))
    record.update(observations)
    record["exit_code"] = exit_code
    record["verdict"] = verdict
    record["finished_utc"] = _utc()
    record["log_sha256"] = hashlib.sha256(log.encode("utf-8")).hexdigest()
    if verdict == "failure":
        record["attribution_status"] = "preserved"
        record["attribution_report"] = None
    elif verdict == "execution_invalid":
        record["invalid_reason"] = (
            "sequence execution started and did not complete all "
            f"{len(SEQUENCES)} sequences for a non-§3 reason "
            f"(completed {len(observations['sequences_completed'])})"
        )
    elif verdict == "invalid":
        record["invalid_reason"] = (
            "setup-phase failure before execution: "
            f"{observations['setup_failure_signature']}"
        )
    return record


def _spawn(argv: Sequence[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """Launch the overlay wrapper.  Lets OSError through on purpose."""
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def compact_attribution_report(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the §8 fields.  Never used as an incidence input."""
    captures = []
    for item in raw.get("captures") or []:
        lifetime = item.get("stream_lifetime") or {}
        owner = lifetime.get("owner") or {}
        frame = owner.get("frame") or {}
        module = frame.get("module") or {}
        captures.append(
            {
                "label": item.get("label"),
                "site_id": item.get("site_id"),
                "tid": item.get("tid"),
                "stream": item.get("stream"),
                "context": item.get("context"),
                "flags": item.get("flags"),
                "mode": item.get("mode"),
                "capture_id": item.get("capture_id") or item.get("id"),
                "status": item.get("status"),
                "domain": item.get("domain"),
                "lifetime_id": lifetime.get("lifetime_id"),
                "generation": lifetime.get("generation"),
                "owner_status": owner.get("status"),
                "owner_resolution": owner.get("owner_resolution"),
                "owner_symbol": frame.get("symbol"),
                "owner_module": module.get("path"),
                "runtime_api_identity": owner.get("runtime_api_identity"),
                "nested_driver_api": owner.get("nested_driver_api"),
                "correlation": owner.get("correlation"),
            }
        )
    lifetimes = []
    nested = []
    for item in raw.get("stream_lifetimes") or []:
        owner = item.get("owner") or {}
        frame = owner.get("frame") or {}
        module = frame.get("module") or {}
        compact = {
            "lifetime_id": item.get("lifetime_id"),
            "generation": item.get("generation"),
            "flags": item.get("flags"),
            "logical_creation_api": item.get("logical_creation_api"),
            "owner_status": owner.get("status"),
            "owner_resolution": owner.get("owner_resolution"),
            "owner_symbol": frame.get("symbol"),
            "owner_module": module.get("path"),
            "runtime_api_identity": owner.get("runtime_api_identity"),
            "nested_driver_api": owner.get("nested_driver_api"),
            "correlation": owner.get("correlation"),
            "observed_creation_apis": item.get("observed_creation_apis"),
        }
        lifetimes.append(compact)
        if owner.get("owner_resolution") == "nested_driver_parentage":
            nested.append(
                {
                    "lifetime_id": item.get("lifetime_id"),
                    "runtime_api_identity": owner.get("runtime_api_identity"),
                    "nested_driver_api": owner.get("nested_driver_api"),
                    "correlation": owner.get("correlation"),
                    "caller_module": module.get("path"),
                    "caller_symbol": frame.get("symbol"),
                }
            )
    return {
        "trace_structure_ok": raw.get("trace_structure_ok"),
        "ownership_evidence_ok": raw.get("ownership_evidence_ok"),
        "problems": list(raw.get("problems") or []),
        "evidence_gaps": list(raw.get("evidence_gaps") or []),
        "captures": captures,
        "capture_errors": raw.get("capture_errors") or [],
        "stream_lifetimes": lifetimes,
        "event_join_history": raw.get("event_edges") or [],
        "nested_driver_parentage": nested,
        "raw_runtime_driver_stacks": [
            item.get("observed_creation_apis")
            for item in raw.get("stream_lifetimes") or []
        ],
    }


def _run_analyzer(control: Path, trace_dir: Path) -> dict[str, Any]:
    analyzer = control / ANALYZER_REL
    proc = subprocess.run(
        (sys.executable, str(analyzer), str(trace_dir)),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"analyzer produced no JSON (exit {proc.returncode}): "
            f"{proc.stderr[-500:] or proc.stdout[-500:] or exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("analyzer JSON is not an object")
    return parsed


def attribute_failure(
    record: dict[str, Any],
    *,
    control: Path,
    attributor: Any = None,
) -> None:
    """Second-layer analysis of the already-sealed failure-time trace.

    Restores the incidence bit afterwards so an attributor cannot rewrite it.
    """
    sealed_verdict = record["verdict"]
    sealed_hits = list(record["capture_error_hits"])
    if sealed_verdict != "failure":
        return
    try:
        raw = (attributor or _run_analyzer)(control, Path(record["trace_dir"]))
        record["attribution_status"] = "analyzed"
        record["attribution_report"] = compact_attribution_report(raw)
    except Exception as exc:  # noqa: BLE001 — diagnostic failure is secondary
        record["attribution_status"] = "analysis_incomplete"
        record["attribution_report"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    record["verdict"] = sealed_verdict
    record["capture_error_hits"] = sealed_hits


def schedule(pairs: int = N_PAIRS) -> Iterator[tuple[int, str]]:
    """§2.4 effective slots, A first."""
    for slot in range(pairs):
        for path in PATH_ORDER:
            yield slot, path


@dataclass
class Campaign:
    """Counters and the §5/§7 terminal decision.  No statistics live here."""

    valid: dict[str, int]
    failures: dict[str, int]
    invalid: int = 0
    terminal: str | None = None
    detail: str | None = None

    @classmethod
    def new(cls) -> "Campaign":
        return cls(
            valid={p: 0 for p in PATH_ORDER}, failures={p: 0 for p in PATH_ORDER}
        )

    def absorb(self, record: dict[str, Any]) -> None:
        path = record["path"]
        verdict = record["verdict"]
        if verdict == "execution_invalid":
            self.terminal = "EXECUTION_INVALID"
            self.detail = record.get("invalid_reason")
            return
        if verdict == "invalid":
            self.invalid += 1
            return
        self.valid[path] += 1
        if verdict == "failure":
            self.failures[path] += 1
            self.terminal = f"FAILURE_OBSERVED_{path}"
            self.detail = json.dumps(record["capture_error_hits"], ensure_ascii=False)

    def path_done(self, path: str, pairs: int) -> bool:
        return self.valid[path] >= pairs or self.failures[path] > 0

    def finalise(self, pairs: int) -> str:
        if self.terminal:
            return self.terminal
        if all(self.valid[p] >= pairs and self.failures[p] == 0 for p in PATH_ORDER):
            return "CLOSED_BOUNDED"
        return "UNRESOLVED_INVALID_STUDY"


def write_manifest(
    artifact_dir: Path, campaign_id: str, payload: dict[str, Any]
) -> None:
    body = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": campaign_id,
        "preregistration": PREREG_REL,
        "seal": SEAL_REL,
        "target_source_sha": TARGET_SOURCE_SHA,
        "observer_control_commit": OBSERVER_CONTROL_COMMIT,
        "prereg_seal_commit": PREREG_SEAL_COMMIT,
        "coordinate_decision_surface": COORDINATE_DECISION_SURFACE,
        "coordinate_environment": COORDINATE_ENVIRONMENT,
        "coordinate_identity_semantics": COORDINATE_IDENTITY_SEMANTICS,
        "coordinate_implementation": COORDINATE_IMPLEMENTATION,
        "coordinate_runtime_inputs": COORDINATE_RUNTIME_INPUTS,
        "probe_digest": PROBE_DIGEST,
        "analyzer_source_sha256": ANALYZER_SOURCE_SHA256,
        "observer_cpp_sha256": OBSERVER_CPP_SHA256,
        "observer_so_sha256": OBSERVER_SO_SHA256,
        **payload,
    }
    _write_json(artifact_dir / "manifest.json", body)


def _best_effort(action: Any) -> None:
    try:
        action()
    except (OSError, ExecutionInvalid):
        pass


def run_campaign(
    *,
    campaign: "Campaign",
    campaign_id: str,
    target: Path,
    interpreter: Path,
    control: Path,
    observer: Path,
    artifact_dir: Path,
    pairs: int,
    executor: Any = None,
    attributor: Any = None,
    report: Any = None,
) -> str:
    """Fill effective slots.  Attribution runs only after incidence is sealed."""
    runs_jsonl = artifact_dir / "runs.jsonl"
    execute = executor if executor is not None else execute_run
    attempt = 0

    for slot, path in schedule(pairs):
        if campaign.terminal:
            break
        if campaign.path_done(path, pairs):
            continue
        consecutive_invalid = 0
        while True:
            seq_index = attempt
            attempt += 1
            run_dir, _log_path, trace_dir, sealed_path = run_paths(
                artifact_dir, seq_index, path
            )
            try:
                record = execute(
                    path=path,
                    seq_index=seq_index,
                    slot_index=slot,
                    campaign_id=campaign_id,
                    target=target,
                    interpreter=interpreter,
                    control=control,
                    observer=observer,
                    artifact_dir=artifact_dir,
                )
            except ExecutionInvalid as exc:
                reason = str(exc)
                row = exc.record
                if row is None:
                    row = _abort(
                        blank_record(
                            campaign_id=campaign_id,
                            seq_index=seq_index,
                            slot_index=slot,
                            path=path,
                            run_dir=run_dir,
                            trace_dir=trace_dir,
                        ),
                        reason,
                    )
                _best_effort(lambda: seal_incidence(sealed_path, row))
                _best_effort(lambda: _write_record(runs_jsonl, row))
                raise
            seal_incidence(sealed_path, record)
            campaign.absorb(record)
            if report is not None:
                report(slot, seq_index, path, record, campaign)
            if record["verdict"] == "failure":
                attribute_failure(record, control=control, attributor=attributor)
                sidecar = Path(record["trace_dir"]) / "attribution.json"
                _best_effort(
                    lambda: _write_json(sidecar, record["attribution_report"] or {})
                )
                seal_incidence(sealed_path, record)
            _write_record(runs_jsonl, record)
            if campaign.terminal:
                break
            if record["verdict"] != "invalid":
                break
            consecutive_invalid += 1
            if consecutive_invalid >= MAX_CONSECUTIVE_SETUP_INVALID:
                campaign.terminal = "UNRESOLVED_INVALID_STUDY"
                campaign.detail = (
                    f"{consecutive_invalid} consecutive setup-invalid attempts on "
                    f"slot {slot} path {path}; §5 validity failure"
                )
                break

    return campaign.finalise(pairs)


def _print_progress(
    slot: int, seq_index: int, path: str, record: dict[str, Any], campaign: "Campaign"
) -> None:
    print(
        f"  slot {slot:04d} attempt {seq_index:04d} {path}  "
        f"{record['verdict']:16} valid A={campaign.valid['A']} "
        f"B={campaign.valid['B']} invalid={campaign.invalid}"
    )


def _control_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, default=None)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the frozen identity checks and stop; writes no campaign manifest",
    )
    args = parser.parse_args(argv)

    target = args.target.resolve()
    interpreter = (args.interpreter or (target / ".venv" / "bin" / "python")).resolve()
    control = (args.control or _control_root()).resolve()
    observer = args.observer.resolve()

    if args.preflight_only:
        checks = preflight(target, interpreter, observer=observer, control=control)
        print(f"  target_head_observed     {checks.head}")
        print(f"  target_worktree_clean    {checks.clean}")
        print(f"  saccade_import_root      {checks.import_root}")
        print(f"  observer_sha256_observed {checks.observer_sha256}")
        print(f"  observer_cpp_sha256      {checks.observer_source_sha256}")
        print(f"  analyzer_sha256          {checks.analyzer_sha256}")
        print(f"  overlay_path             {checks.overlay_path}")
        for failure in checks.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 0 if checks.ok else 1

    if args.pairs != N_PAIRS:
        print(
            f"refusing to run: §2.4 freezes N at {N_PAIRS} pairs; --pairs is for "
            "harness self-verification only and may not be used for the campaign",
            file=sys.stderr,
        )
        return 1

    campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = (
        args.artifact_dir
        or Path.home()
        / ".local"
        / "state"
        / "saccade"
        / "perf"
        / f"capture-race-incidence-{campaign_id}"
    ).resolve()

    campaign = Campaign.new()
    manifest_exists = False
    try:
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExecutionInvalid(f"could not create {artifact_dir}: {exc}") from exc
        write_manifest(
            artifact_dir,
            campaign_id,
            {
                "started_utc": _utc(),
                "pairs": args.pairs,
                "terminal": None,
                "observer": str(observer),
                "control": str(control),
            },
        )
        manifest_exists = True
        terminal = run_campaign(
            campaign=campaign,
            campaign_id=campaign_id,
            target=target,
            interpreter=interpreter,
            control=control,
            observer=observer,
            artifact_dir=artifact_dir,
            pairs=args.pairs,
            report=_print_progress,
        )
        write_manifest(
            artifact_dir,
            campaign_id,
            {
                "finished_utc": _utc(),
                "pairs": args.pairs,
                "terminal": terminal,
                "detail": campaign.detail,
                "valid": campaign.valid,
                "failures": campaign.failures,
                "invalid": campaign.invalid,
                "observer": str(observer),
                "control": str(control),
            },
        )
    except ExecutionInvalid as exc:
        detail = str(exc)
        print(f"EXECUTION_INVALID: {detail}", file=sys.stderr)
        if manifest_exists:
            _best_effort(
                lambda: write_manifest(
                    artifact_dir,
                    campaign_id,
                    {
                        "finished_utc": _utc(),
                        "pairs": args.pairs,
                        "terminal": "EXECUTION_INVALID",
                        "detail": detail,
                        "valid": campaign.valid,
                        "failures": campaign.failures,
                        "invalid": campaign.invalid,
                        "observer": str(observer),
                        "control": str(control),
                    },
                )
            )
        return 1

    print(f"terminal: {terminal}")
    print(f"artifacts: {artifact_dir}")
    return 0 if terminal == "CLOSED_BOUNDED" else 1


if __name__ == "__main__":
    sys.exit(main())
