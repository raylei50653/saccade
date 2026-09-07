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

Pass/fail is raw MOT identity, including track IDs.  A pass on N runs is not
a determinism proof; a single distinct hash is a failure.

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

from saccade.perception.eval.repeat_identity import (  # noqa: E402
    RepeatReport,
    compare_run_dirs,
    format_report,
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


def merge_eval_flags(forwarded: Sequence[str]) -> list[str]:
    """Fill in #363 block-S defaults unless the caller already set them."""

    merged = list(forwarded)
    for flag, value in DEFAULT_KV.items():
        if not _flag_present(merged, flag):
            merged.extend([flag, value])
    for flag in DEFAULT_SWITCHES:
        if not _flag_present(merged, flag):
            merged.append(flag)
    return merged


def _write_summary(path: Path, report: RepeatReport) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def compare_and_emit(run_dirs: Sequence[Path], *, summary: Path | None) -> int:
    report = compare_run_dirs(run_dirs)
    print(format_report(report))
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        _write_summary(summary, report)
    return 0 if report.ok else 1


def run_one_eval(
    *,
    python: str,
    eval_script: Path,
    out_dir: Path,
    eval_flags: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
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


def cmd_compare(dirs: Sequence[Path], summary: Path | None) -> int:
    return compare_and_emit(dirs, summary=summary)


def cmd_run(
    *,
    n: int,
    sleep: float,
    artifact_dir: Path,
    forwarded: Sequence[str],
) -> int:
    for flag in MANAGED_FLAGS:
        if _flag_present(forwarded, flag):
            print(f"{flag} is managed by this tool", file=sys.stderr)
            return 2
    eval_flags = merge_eval_flags(forwarded)
    root = artifact_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    eval_script = _ROOT / "scripts" / "eval" / "mot17.py"
    run_dirs: list[Path] = []
    print(f"eval-repeat identity: n={n} sleep={sleep}s flags={' '.join(eval_flags)}")
    for index in range(n):
        out_dir = root / f"r{index + 1}"
        print(f"  run {index + 1}/{n} → {out_dir}")
        proc = run_one_eval(
            python=sys.executable,
            eval_script=eval_script,
            out_dir=out_dir,
            eval_flags=eval_flags,
        )
        run_dirs.append(out_dir)
        if proc.returncode != 0:
            print(
                f"  run {index + 1} eval exit {proc.returncode} "
                f"(still compared; fail-closed)",
                file=sys.stderr,
            )
        if index + 1 < n and sleep > 0:
            time.sleep(sleep)
    return compare_and_emit(run_dirs, summary=root / "summary.json")


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="compare existing MOT run directories")
    compare.add_argument("dirs", nargs="+", type=Path)
    compare.add_argument("--summary", type=Path, default=None)

    run = sub.add_parser("run", help="launch N independent evals, then compare")
    run.add_argument("-n", "--n", type=int, default=DEFAULT_N)
    run.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    run.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="default: out/determinism/eval_repeat_<timestamp>/",
    )

    if not argv_list:
        parser.print_help()
        return 2
    if argv_list[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv_list[0] == "compare":
        args = parser.parse_args(argv_list)
        return cmd_compare(args.dirs, args.summary)
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
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
