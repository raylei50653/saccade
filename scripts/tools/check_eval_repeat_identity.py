#!/usr/bin/env python3
"""Fail-closed detector of silent MOT run-to-run divergence (issue #363).

Eval can exit 0 and still write different MOT files.  This tool fails when
that happens.  It does not claim a CUDA-level mechanism.

Modes:

    compare DIR [DIR ...]
        Compare MOT17-*.txt across existing run directories.  Distinct hashes,
        empty files, or missing sequences fail.

    run [--n N] [mot17.py flags...]
        Launch N independent ``scripts/eval/mot17.py`` processes, then compare.
        Defaults match the #363 block-S configuration (``--preset baseline
        --detector SDP --no-gpu-decode --sequences MOT17-02-SDP``).  Forwarded
        flags override those defaults.

Pass/fail is raw MOT identity, including track IDs, **and** every eval
process exit.  A non-zero ``mot17.py`` status fails the harness even when
the MOT files are complete and identical.  A pass on N runs is not a
determinism proof; a single distinct hash or a single eval failure is a
failure.

``--stage-fingerprint`` is opt-in per-stage observability for condition 2.
It does not change default eval.  Missing or incomplete fingerprints fail
closed.  A first divergent stage is the first observable producer-facing
boundary, not a causal mechanism.

Not wired to pre-push: the current ``baseline`` path is known to diverge, so
a default CI gate would fail on main.  After a fix, ``run`` is the regression
gate.  ``compare`` on stored #363 evidence is the positive control.

The decimal-hash chain (``check_decimal_chain_routine.py``) is a different
question: same-process order contamination, ID-free, on
``mamba_whole_graph_m --double-buffer``.
"""
# status: stable

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SCRIPT_DIR))

from eval_repeat_identity import (  # noqa: E402
    EMPTY,
    MISSING,
    RepeatReport,
    compare_run_dirs,
    format_report,
)
from eval_stage_fingerprint import (  # noqa: E402
    KIND_FIRST_OBSERVABLE,
    KIND_INSUFFICIENT,
    LOCALIZATION_BUDGETS,
    LOCALIZATION_CONFIG_BLOCK_S,
    LOCALIZATION_CONFIG_GPU_DECODE,
    LOCALIZATION_BUDGET_RUNS,
    compare_stage_fingerprints,
    format_stage_report,
    localization_budget,
    read_localization_session,
    write_first_divergence,
)

DEFAULT_N = 8
DEFAULT_SLEEP = 1.0
DEFAULT_KV: dict[str, str] = {
    "--preset": "baseline",
    "--detector": "SDP",
    "--sequences": "MOT17-02-SDP",
    "--mlflow-uri": "",
}
DEFAULT_SWITCHES: tuple[str, ...] = ("--no-gpu-decode",)
MANAGED_FLAGS = ("--output",)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flag_present(args: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return flag in args or any(item.startswith(prefix) for item in args)


def merge_eval_flags(
    forwarded: Sequence[str],
    *,
    inject_no_gpu_decode: bool = True,
) -> list[str]:
    """Fill in #363 defaults unless the caller already set them.

    ``inject_no_gpu_decode`` is the Block S switch.  GPU-decode localization
    must pass False so the historical arm-G configuration is not rewritten.
    """

    merged = list(forwarded)
    for flag, value in DEFAULT_KV.items():
        if not _flag_present(merged, flag):
            merged.extend([flag, value])
    if inject_no_gpu_decode:
        for flag in DEFAULT_SWITCHES:
            if not _flag_present(merged, flag):
                merged.append(flag)
    return merged


def _write_summary(path: Path, report: RepeatReport) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def _mot_outputs_complete(report: RepeatReport) -> bool:
    if report.n_runs < 2 or not report.reports:
        return False
    return not any(
        len(item.hashes) != report.n_runs
        or any(value in {MISSING, EMPTY} for value in item.hashes)
        for item in report.reports
    )


def _has_complete_mot_divergence(report: RepeatReport) -> bool:
    """True only for a non-empty, coverage-complete divergent MOT comparison."""

    return _mot_outputs_complete(report) and any(
        item.n_distinct > 1 for item in report.reports
    )


def compare_and_emit(
    run_dirs: Sequence[Path],
    *,
    summary: Path | None,
    stage_fingerprint: bool = False,
) -> int:
    report = compare_run_dirs(run_dirs)
    print(format_report(report))
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        _write_summary(summary, report)
    stage_rc = 0
    if stage_fingerprint:
        stage_report = compare_stage_fingerprints(
            run_dirs,
            mot_diverged=_has_complete_mot_divergence(report),
        )
        print(format_stage_report(stage_report))
        if summary is not None:
            stage_path = summary.parent / "stage_fingerprint.json"
            stage_path.write_text(
                json.dumps(stage_report.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            if stage_report.first_divergence is not None:
                write_first_divergence(
                    summary.parent / "first_divergence.json", stage_report
                )
        stage_rc = 0 if stage_report.ok else 1
    return 1 if (not report.ok) or stage_rc != 0 else 0


def run_one_eval(
    *,
    python: str,
    eval_script: Path,
    out_dir: Path,
    eval_flags: Sequence[str],
    stage_fingerprint: bool = False,
) -> subprocess.CompletedProcess[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage_fingerprint:
        wrapper = _SCRIPT_DIR / "run_eval_stage_fingerprint.py"
        fingerprint_dir = out_dir / "stage_fingerprint"
        cmd = [
            python,
            str(wrapper),
            "--fingerprint-dir",
            str(fingerprint_dir),
            *eval_flags,
            "--output",
            str(out_dir),
        ]
    else:
        cmd = [python, str(eval_script), *eval_flags, "--output", str(out_dir)]
    log_path = out_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            cmd,
            cwd=_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def cmd_compare(
    dirs: Sequence[Path],
    summary: Path | None,
    *,
    stage_fingerprint: bool = False,
) -> int:
    return compare_and_emit(dirs, summary=summary, stage_fingerprint=stage_fingerprint)


def cmd_run(
    *,
    n: int,
    sleep: float,
    artifact_dir: Path,
    forwarded: Sequence[str],
    stage_fingerprint: bool = False,
    localization_config: str = LOCALIZATION_CONFIG_BLOCK_S,
) -> int:
    for flag in MANAGED_FLAGS:
        if _flag_present(forwarded, flag):
            print(f"{flag} is managed by this tool", file=sys.stderr)
            return 2
    if localization_config not in LOCALIZATION_BUDGETS:
        print(
            f"unknown localization config {localization_config!r}",
            file=sys.stderr,
        )
        return 2
    budget = localization_budget(localization_config)
    gpu_decode = localization_config == LOCALIZATION_CONFIG_GPU_DECODE
    if gpu_decode and _flag_present(forwarded, "--no-gpu-decode"):
        print(
            "gpu_decode localization cannot be combined with --no-gpu-decode",
            file=sys.stderr,
        )
        return 2
    if stage_fingerprint and n > budget:
        print(
            f"n={n} exceeds the preregistered {localization_config} "
            f"localization budget {budget}; this is a session cap, not a "
            "rate sample",
            file=sys.stderr,
        )
        return 2
    eval_flags = merge_eval_flags(forwarded, inject_no_gpu_decode=not gpu_decode)
    root = artifact_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    eval_script = _ROOT / "scripts" / "eval" / "mot17.py"
    run_dirs: list[Path] = []
    eval_returncodes: list[int] = []
    print(
        f"eval-repeat identity: n={n} sleep={sleep}s "
        f"stage_fingerprint={stage_fingerprint} "
        f"localization_config={localization_config} "
        f"budget={budget} flags={' '.join(eval_flags)}"
    )
    for index in range(n):
        out_dir = root / f"r{index + 1}"
        print(f"  run {index + 1}/{n} → {out_dir}")
        proc = run_one_eval(
            python=sys.executable,
            eval_script=eval_script,
            out_dir=out_dir,
            eval_flags=eval_flags,
            stage_fingerprint=stage_fingerprint,
        )
        run_dirs.append(out_dir)
        eval_returncodes.append(proc.returncode)
        if proc.returncode != 0:
            print(
                f"  run {index + 1} eval exit {proc.returncode} "
                f"(still compared; fail-closed)",
                file=sys.stderr,
            )
        mot_pair = False
        if stage_fingerprint and len(run_dirs) >= 2:
            mot_pair = not any(eval_returncodes) and _has_complete_mot_divergence(
                compare_run_dirs(run_dirs)
            )
            if mot_pair:
                print(
                    f"  MOT pair found at run {index + 1}/{budget}; "
                    "stopping within the preregistered localization budget"
                )
                break
        if index + 1 < n and sleep > 0 and not mot_pair:
            time.sleep(sleep)
    had_eval_failure = any(code != 0 for code in eval_returncodes)
    compare_rc = compare_and_emit(
        run_dirs,
        summary=root / "summary.json",
        stage_fingerprint=stage_fingerprint,
    )
    exits_path = root / "eval_exits.json"
    exits_path.write_text(
        json.dumps(
            {
                "returncodes": eval_returncodes,
                "had_eval_failure": had_eval_failure,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if had_eval_failure:
        print(
            "eval_failures: at least one mot17.py process returned non-zero "
            f"{eval_returncodes}",
            file=sys.stderr,
        )
    if stage_fingerprint:
        final_mot_report = compare_run_dirs(run_dirs)
        mot_pair_valid = not had_eval_failure and _has_complete_mot_divergence(
            final_mot_report
        )
        stage_payload = json.loads(
            (root / "stage_fingerprint.json").read_text(encoding="utf-8")
        )
        first = stage_payload.get("first_divergence") or {}
        fingerprint_reading_valid = first.get("kind") in {
            KIND_FIRST_OBSERVABLE,
            KIND_INSUFFICIENT,
        }
        session_valid = (
            not had_eval_failure
            and _mot_outputs_complete(final_mot_report)
            and bool(stage_payload.get("complete", False))
            and (not mot_pair_valid or fingerprint_reading_valid)
        )
        session = read_localization_session(
            n_runs=len(run_dirs),
            divergent_pair=mot_pair_valid,
            config=localization_config,
            producing_path_verdict=first.get("producing_path_verdict"),
            session_valid=session_valid,
        )
        session_payload = session.to_dict()
        session_payload["mot_pair_valid"] = mot_pair_valid
        session_payload["session_valid"] = session_valid
        session_payload["eval_flags"] = eval_flags
        session_payload["eval_returncodes"] = eval_returncodes
        (root / "localization_session.json").write_text(
            json.dumps(session_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"localization-session: kind={session.kind} "
            f"config={session.config} "
            f"n={session.n_runs}/{session.budget_runs} "
            f"apply_condition2_rules={session.apply_condition2_rules} "
            f"condition_2_advanced={session.condition_2_advanced}"
        )
        if session.allowed_claim:
            print(f"  allowed_claim: {session.allowed_claim}")
    return 1 if had_eval_failure or compare_rc != 0 else 0


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="compare existing MOT run directories")
    compare.add_argument("dirs", nargs="+", type=Path)
    compare.add_argument("--summary", type=Path, default=None)
    compare.add_argument(
        "--stage-fingerprint",
        action="store_true",
        help="require and compare opt-in per-stage fingerprints (fail-closed)",
    )

    run = sub.add_parser("run", help="launch N independent evals, then compare")
    run.add_argument(
        "-n",
        "--n",
        type=int,
        default=DEFAULT_N,
        help=(
            "independent evals; with --stage-fingerprint, n cannot exceed "
            "the preregistered budget of --localization-config"
        ),
    )
    run.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    run.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="default: out/determinism/eval_repeat_<timestamp>/",
    )
    run.add_argument(
        "--stage-fingerprint",
        action="store_true",
        help="opt-in per-stage fingerprints in each child eval (default off)",
    )
    run.add_argument(
        "--localization-config",
        choices=tuple(LOCALIZATION_BUDGETS),
        default=LOCALIZATION_CONFIG_BLOCK_S,
        help=(
            "preregistered localization session contract; block_s budget "
            f"{LOCALIZATION_BUDGET_RUNS}, gpu_decode budget "
            f"{LOCALIZATION_BUDGETS[LOCALIZATION_CONFIG_GPU_DECODE]}"
        ),
    )

    if not argv_list:
        parser.print_help()
        return 2
    if argv_list[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv_list[0] == "compare":
        args = parser.parse_args(argv_list)
        return cmd_compare(
            args.dirs, args.summary, stage_fingerprint=args.stage_fingerprint
        )
    if argv_list[0] == "run":
        args, forwarded = run.parse_known_args(argv_list[1:])
        if args.n < 2:
            print("n must be >= 2", file=sys.stderr)
            return 2
        artifact = args.artifact_dir or Path(
            f"out/determinism/eval_repeat_{_timestamp()}"
        )
        return cmd_run(
            n=args.n,
            sleep=args.sleep,
            artifact_dir=artifact,
            forwarded=forwarded,
            stage_fingerprint=args.stage_fingerprint,
            localization_config=args.localization_config,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
