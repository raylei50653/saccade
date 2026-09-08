#!/usr/bin/env python3
"""Run the #340 capture-race incidence campaign under a frozen preregistration.

The preregistration is
``docs/research/pipeline/closed/capture_race_incidence_preregistration_20260908.md``
(sealed, amendment A1).  This module executes it and nothing else: every
threshold, string, count and terminal below is transcribed from that document,
and the transcription is asserted against the document itself by
``tests/unit/eval/test_capture_race_incidence.py``.  It makes no statistical
decision of its own — where the preregistration is silent the harness aborts
rather than guessing.

Two worktrees (§2):

    target      clean, detached at TARGET_SOURCE_SHA; the only execution
                source.  ``mot17.py`` is spawned from here with this tree's
                interpreter and cwd.
    control     this file, the campaign log, the preregistration.  Free to
                advance.

**This module must never import saccade.** That is what makes it safe to run
from the control worktree while the target holds the source under measurement.
The venv's editable install resolves ``saccade`` to a fixed absolute path, so a
control-side import would silently pull in control-side source.  Preflight
check 3 exists for exactly that hazard, and a harness that imported saccade
would be checking a property it had already violated.  Standard library only.

Fail-closed everywhere: a failed preflight, an unwritable log, an unparseable
verdict, or a mid-run non-capture failure ends the campaign as
``EXECUTION_INVALID``.  Nothing here can quietly drop a run.
"""
# status: stable

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
    "docs/research/pipeline/closed/capture_race_incidence_preregistration_20260908.md"
)
SCHEMA = "capture_race_incidence_run_v1"

# --- §2 frozen coordinate --------------------------------------------------
TARGET_SOURCE_SHA = "b649de68e36ad530ed883f579478ab656a238158"
COORDINATE_IMPLEMENTATION = (
    "26c49eb3629ed4b703641277b9bcd177a373b3b248570cd2cb5056b346addfa7"
)

# --- §2 frozen workload ----------------------------------------------------
BASE_ARGS: tuple[str, ...] = (
    "--preset",
    "mamba_whole_graph_m",
    "--detector",
    "SDP",
    "--double-buffer",
)
PATH_EXTRA_ARGS: dict[str, tuple[str, ...]] = {"A": (), "B": ("--no-gpu-decode",)}
PATH_ORDER: tuple[str, ...] = ("A", "B")  # §2: interleaving frozen as A first
N_PAIRS = 100
# A2.3: consecutive setup-invalid re-attempts allowed on one effective slot.
# Frozen, because "try once more" is otherwise a knob that can be turned after
# seeing results, and an unbounded retry is not a terminal.
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

# --- §3 primary failure predicate ------------------------------------------
# Substring matches, case-sensitive.  The driver messages are substrings of
# their torch wrapper forms, so matching the core message covers both rows of
# the §3 table without enumerating every wrapper separately.
CAPTURE_FAILURE_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("900", "cudaErrorStreamCaptureUnsupported"),
    ("900", "operation not permitted when stream is capturing"),
    ("901", "cudaErrorStreamCaptureInvalidated"),
    ("901", "operation failed due to a previous error during capture"),
    ("906", "cudaErrorStreamCaptureImplicit"),
    ("906", "legacy stream depend on a capturing blocking stream"),
    ("906", "currentStreamCaptureStatusMayInitCtx"),
)

# --- A1 execution-boundary observables -------------------------------------
PROGRESS_MARKER = "\N{CLAPPER BOARD} "  # evaluator.py, every 100th frame

# Each row is (category, context terms, failure terms).  A row matches only if
# some context term AND some failure term co-occur on the **same line** —
# case-insensitive.  An empty context tuple means the failure terms are
# self-contextualising.  Whole-log matching is fail-open: every production run
# prints a `.engine` banner, so any later `error` would classify as `tensorrt`
# and widen the only re-runnable invalid class (#375).
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

# --- §8 observer prohibition -----------------------------------------------
# Primary rate measurement runs with no injected profiler.  Set in the child's
# environment, these would perturb the very race being counted.
FORBIDDEN_OBSERVER_ENV: tuple[str, ...] = (
    "CUDA_INJECTION64_PATH",
    "LD_PRELOAD",
    "CUDA_PROFILE",
    "COMPUTE_PROFILE",
    "NSYS_PROFILING_SESSION_ID",
    "CUPTI_ACTIVITY_KIND",
)

# Probe run in the target interpreter for preflight check 3.  Prints the
# directory that ``import saccade`` actually resolves to, which is the only
# thing that decides which source is measured.
IMPORT_ROOT_PROBE = (
    "import os, saccade; "
    "print(os.path.dirname(os.path.dirname(os.path.abspath(saccade.__file__))))"
)


class ExecutionInvalid(RuntimeError):
    """Campaign-ending condition (§7 EXECUTION_INVALID).  Never per-run.

    Carries the §10 row of the attempt it interrupted when one exists, so the
    abort keeps whatever was already observed instead of discarding it and
    writing nulls over verified fields.
    """

    def __init__(self, message: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class Preflight:
    """Observed values for the three §2 pre-run checks.

    Recorded whether or not the checks pass: A1 requires the observations to
    land even on a fail-closed abort, because a fail-closed stop with no record
    is not evidence that anything was checked.
    """

    head: str | None
    clean: bool | None
    import_root: str | None
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(target: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ("git", "-C", str(target), *args),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        # git missing, target path gone: still harness failure, not a crash.
        raise ExecutionInvalid(f"could not run git in {target}: {exc}") from exc
    if proc.returncode != 0:
        raise ExecutionInvalid(
            f"git {' '.join(args)} failed in {target}: {proc.stderr.strip()}"
        )
    return proc.stdout


def preflight(target: Path, interpreter: Path) -> Preflight:
    """Run the three §2 checks.  Never raises for a check failure — reports it.

    Checks 1 and 2 pin the checkout; check 3 pins what is actually imported.
    Only check 3 catches the editable-install hazard described in §2: a shared
    venv resolves ``saccade`` through an absolute ``.pth`` path, so the first
    two can both pass while the measured source is another worktree entirely.
    """
    failures: list[str] = []
    head: str | None = None
    clean: bool | None = None
    import_root: str | None = None

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
        # A missing or non-executable interpreter is a check-3 failure, not a
        # stacktrace.  Letting OSError out here would skip the record entirely
        # and leave the manifest at terminal null — the state A1 and A2.4 exist
        # to prevent.
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

    return Preflight(
        head=head, clean=clean, import_root=import_root, failures=tuple(failures)
    )


def run_paths(artifact_dir: Path, seq_index: int, path: str) -> tuple[Path, Path]:
    """Where one attempt's output and log live.  One definition, two callers."""
    stem = f"{seq_index:04d}-{path}"
    return artifact_dir / "runs" / stem, artifact_dir / "logs" / f"{stem}.log"


def blank_record(
    *,
    campaign_id: str,
    seq_index: int,
    slot_index: int,
    path: str,
    run_dir: Path,
) -> dict[str, Any]:
    """The one and only ``capture_race_incidence_run_v1`` row shape.

    Both the normal path and the abort path build from here.  A second,
    narrower shape for interrupted attempts would be a partial row wearing the
    frozen schema name, and that is worse than no row at all: a consumer that
    trusts ``schema`` would read the missing keys as observations that came
    back empty rather than as observations never made.  Unknowns are explicit
    nulls and empty collections instead.
    """
    return {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "seq_index": seq_index,
        "slot_index": slot_index,
        "path": path,
        "run_dir": str(run_dir),
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
        "coordinate_implementation": COORDINATE_IMPLEMENTATION,
    }


def _abort(record: dict[str, Any], reason: str) -> dict[str, Any]:
    """Stamp a row as the attempt a campaign-ending condition interrupted."""
    record["verdict"] = "execution_invalid"
    record["invalid_reason"] = reason
    record["finished_utc"] = _utc()
    return record


def capture_failure_hits(log: str) -> list[dict[str, Any]]:
    """§3 predicate.  Exit code is deliberately not an input."""
    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(log.splitlines(), start=1):
        for code, signature in CAPTURE_FAILURE_SIGNATURES:
            if signature in line:
                hits.append({"cuda_code": code, "signature": signature, "line": lineno})
    return hits


def setup_failure_signature(log: str, output_dir: str) -> str | None:
    """A1 table.  Returns the matching category, or None.

    Context and failure terms must co-occur on the same line.  Matching them
    independently anywhere in the log lets a normal TRT/Mamba ``.engine``
    banner plus any later ``error`` / ``failed`` / ``exception`` satisfy the
    ``tensorrt`` row — the campaign's capture failure was recorded that way
    even though the log contained no TensorRT failure (#375).
    """
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
    """Return (verdict, observations) for one completed subprocess.

    Ordering matters and follows the preregistration, not convenience:

    §3 first — a run that hit a capture error is a ``failure`` whatever else it
    did, including one that also failed to finish.  Then A1's boundary, whose
    default is ``started``.  A run that started and did not finish all seven
    sequences for a non-§3 reason is ``execution_invalid``: §4 gives it no
    re-run and no place in the denominator, so it ends the campaign.
    """
    hits = capture_failure_hits(log)
    completed = sequences_completed(run_dir)
    markers = sum(1 for line in log.splitlines() if line.startswith(PROGRESS_MARKER))
    signature = setup_failure_signature(log, output_dir)

    # A1: started is the default; false needs both conditions to hold.
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


def build_argv(path: str, target: Path, interpreter: Path, run_dir: Path) -> list[str]:
    if path not in PATH_EXTRA_ARGS:
        raise ExecutionInvalid(f"unknown path {path!r}")
    return [
        str(interpreter),
        str(target / "scripts" / "eval" / "mot17.py"),
        *BASE_ARGS,
        *PATH_EXTRA_ARGS[path],
        "--output",
        str(run_dir),
    ]


def child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the workload: observer-free (§8), unbuffered (A1)."""
    env = dict(os.environ if base is None else base)
    present = [name for name in FORBIDDEN_OBSERVER_ENV if env.get(name)]
    if present:
        raise ExecutionInvalid(
            "§8 forbids observers during primary rate measurement, but these are "
            "set in the environment: " + ", ".join(present)
        )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _write_record(runs_jsonl: Path, record: dict[str, Any]) -> None:
    """Append one record.  An unwritable log ends the campaign (§7)."""
    try:
        with runs_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ExecutionInvalid(f"could not append to {runs_jsonl}: {exc}") from exc


def execute_run(
    *,
    path: str,
    seq_index: int,
    slot_index: int,
    campaign_id: str,
    target: Path,
    interpreter: Path,
    artifact_dir: Path,
    runner: Any = None,
) -> dict[str, Any]:
    """Run one trial and return its §10 record.  Always writes the record."""
    run_dir, log_path = run_paths(artifact_dir, seq_index, path)
    # Built before anything can fail, so every abort below has a full row to
    # attach rather than a reconstruction.
    record = blank_record(
        campaign_id=campaign_id,
        seq_index=seq_index,
        slot_index=slot_index,
        path=path,
        run_dir=run_dir,
    )

    # A2.4: no filesystem failure may escape as a bare OSError.  Every one of
    # them is harness failure, which §7 names EXECUTION_INVALID.
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        reason = f"could not create run directories: {exc}"
        raise ExecutionInvalid(reason, _abort(record, reason)) from exc

    checks = preflight(target, interpreter)
    argv = build_argv(path, target, interpreter, run_dir)
    record["argv"] = argv
    record["target_head_observed"] = checks.head
    record["target_worktree_clean"] = checks.clean
    record["saccade_import_root"] = checks.import_root

    if not checks.ok:
        # A1: preflight failure is execution_invalid regardless of the started
        # flag, and the three observed fields are already in the record.
        record["invalid_reason"] = "; ".join(checks.failures)
        record["finished_utc"] = _utc()
        return record

    try:
        env = child_environment()
    except ExecutionInvalid as exc:
        raise ExecutionInvalid(str(exc), _abort(record, str(exc))) from exc

    run = runner if runner is not None else _spawn
    # A2.4 at the call site, not only inside _spawn: an injected runner must be
    # held to the same rule, or the normalisation is only as good as the
    # default path it happens to sit on.
    try:
        exit_code, log = run(argv, target, env)
    except OSError as exc:
        reason = f"could not spawn the workload: {exc}"
        raise ExecutionInvalid(reason, _abort(record, reason)) from exc

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
    if verdict == "execution_invalid":
        record["invalid_reason"] = (
            "sequence execution started and did not complete all "
            f"{len(SEQUENCES)} sequences for a non-§3 reason "
            f"(completed {len(observations['sequences_completed'])})"
        )
    elif verdict == "invalid":
        record["invalid_reason"] = (
            f"setup-phase failure before execution: {observations['setup_failure_signature']}"
        )
    return record


def _spawn(argv: Sequence[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """Launch the workload.  Lets OSError through on purpose.

    Normalising it here as well would shadow the call site in ``execute_run``,
    which is the only place that holds the row to attach — the abort would then
    lose the identity observations preflight had just made.  One owner for that
    conversion, and it is not this function.
    """
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def schedule(pairs: int = N_PAIRS) -> Iterator[tuple[int, str]]:
    """§2 interleaving as A2.2 **effective slots**, A first.  Frozen order.

    A slot is not an attempt.  It is filled by the first attempt at that path
    whose verdict is not ``invalid``; a setup-invalid re-attempts the same path
    in place without advancing the slot.  Reading these as attempts is what
    made §2's "100 pairs" contradict §4's promised replacement: one legal
    setup-invalid on A would cap A at 99 valid runs forever.
    """
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
    manifest = artifact_dir / "manifest.json"
    body = {
        "schema": "capture_race_incidence_campaign_v1",
        "campaign_id": campaign_id,
        "preregistration": PREREG_REL,
        "target_source_sha": TARGET_SOURCE_SHA,
        "coordinate_implementation": COORDINATE_IMPLEMENTATION,
        **payload,
    }
    try:
        manifest.write_text(
            json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ExecutionInvalid(f"could not write {manifest}: {exc}") from exc


def _best_effort(action: Any) -> None:
    """Run a recording action that must not mask the failure being reported."""
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
    artifact_dir: Path,
    pairs: int,
    executor: Any = None,
    report: Any = None,
) -> str:
    """Fill the effective slots of §2 in order and return the terminal.

    The retry loop is A2.2: a setup-invalid does not consume its slot, so §4's
    promised replacement actually happens and the valid runs stay strictly
    A,B,A,B,…  ``seq_index`` keeps counting attempts, including the invalid
    ones, so the log says what was executed rather than what was kept.
    """
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
            try:
                record = execute(
                    path=path,
                    seq_index=seq_index,
                    slot_index=slot,
                    campaign_id=campaign_id,
                    target=target,
                    interpreter=interpreter,
                    artifact_dir=artifact_dir,
                )
            except ExecutionInvalid as exc:
                # Bind now: `except ... as exc` unbinds at block exit, and the
                # recording callable below is evaluated lazily.
                reason = str(exc)
                row = exc.record
                if row is None:
                    # Raised before the attempt had a row of its own.
                    row = _abort(
                        blank_record(
                            campaign_id=campaign_id,
                            seq_index=seq_index,
                            slot_index=slot,
                            path=path,
                            run_dir=run_paths(artifact_dir, seq_index, path)[0],
                        ),
                        reason,
                    )
                _best_effort(lambda: _write_record(runs_jsonl, row))
                raise
            _write_record(runs_jsonl, record)
            campaign.absorb(record)
            if report is not None:
                report(slot, seq_index, path, record, campaign)
            if campaign.terminal:
                break
            if record["verdict"] != "invalid":
                break
            consecutive_invalid += 1
            if consecutive_invalid >= MAX_CONSECUTIVE_SETUP_INVALID:
                # A2.3: five consecutive setup failures is a broken
                # environment, not a transient one, and retrying forever is
                # not a terminal.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the three §2 checks against the target and stop",
    )
    args = parser.parse_args(argv)

    target = args.target.resolve()
    interpreter = (args.interpreter or (target / ".venv" / "bin" / "python")).resolve()

    if args.preflight_only:
        checks = preflight(target, interpreter)
        print(f"  target_head_observed  {checks.head}")
        print(f"  target_worktree_clean {checks.clean}")
        print(f"  saccade_import_root   {checks.import_root}")
        for failure in checks.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 0 if checks.ok else 1

    if args.pairs != N_PAIRS:
        print(
            f"refusing to run: §2 freezes N at {N_PAIRS} pairs; --pairs is for "
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
            {"started_utc": _utc(), "pairs": args.pairs, "terminal": None},
        )
        manifest_exists = True
        terminal = run_campaign(
            campaign=campaign,
            campaign_id=campaign_id,
            target=target,
            interpreter=interpreter,
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
            },
        )
    except ExecutionInvalid as exc:
        detail = str(exc)
        print(f"EXECUTION_INVALID: {detail}", file=sys.stderr)
        # A2.4: a manifest left at terminal null cannot be told apart from a
        # campaign still running, so stamp it if it exists at all.  Only a
        # genuinely unwritable manifest falls back to exit code plus stderr.
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
                    },
                )
            )
        return 1

    print(f"terminal: {terminal}")
    print(f"artifacts: {artifact_dir}")
    return 0 if terminal == "CLOSED_BOUNDED" else 1


if __name__ == "__main__":
    sys.exit(main())
