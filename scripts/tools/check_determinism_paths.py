#!/usr/bin/env python3
"""Detect whether staged/committed changes affect determinism-sensitive paths.

Prints "determinism" to stdout if sensitive paths are changed; prints nothing
otherwise.  Used by ``scripts/pre_push.sh`` to decide whether to run the
routine continuous-chain sentinel.

Exit codes: 0 = sensitive, 1 = not sensitive, 2 = error (fail-closed).
"""
# status: stable

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]

_SENSITIVE_PATTERNS: tuple[str, ...] = (
    r"^src/tracking/tracker_gpu\.cu$",
    r"^src/tracking/pipeline\.cpp$",
    r"^src/tracking/tracker_gpu_python\.cpp$",
    r"^include/tracking/tracker_gpu\.hpp$",
    r"^include/tracking/pipeline\.hpp$",
    r"^include/tracking/box_ops\.hpp$",
    r"^src/saccade/perception/eval/_decimal_hash_tools\.py$",
    r"^src/saccade/perception/eval/decimal_hash\.py$",
    r"^scripts/tools/check_continuous_decimal_hash\.py$",
    r"^scripts/tools/check_decimal_chain_routine\.py$",
    r"^scripts/tools/check_decimal_matrix_2x2\.py$",
    r"^scripts/tools/check_decimal_matrix_all7\.py$",
    r"^tests/unit/eval/test_decimal_hash\.py$",
    r"^tests/unit/eval/test_decimal_chain_routine\.py$",
    r"^tests/unit/eval/test_decimal_matrix_2x2\.py$",
    r"^scripts/tools/check_determinism_paths\.py$",
    r"^scripts/pre_push\.sh$",
)


def _changed_files() -> set[str]:
    files: set[str] = set()
    any_success = False

    try:
        output = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        any_success = True
        if output:
            files.update(line.strip() for line in output.split("\n") if line.strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    base = _merge_base()
    if base:
        try:
            output = subprocess.check_output(
                ["git", "diff", "--name-only", f"{base}...HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            any_success = True
            if output:
                files.update(
                    line.strip() for line in output.split("\n") if line.strip()
                )
        except (OSError, subprocess.CalledProcessError):
            pass

    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        any_success = True
        if output:
            files.update(line.strip() for line in output.split("\n") if line.strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    if not any_success:
        raise RuntimeError(
            "all git diff commands failed — cannot determine changed files"
        )

    return files


def _merge_base() -> str | None:
    for remote in ("origin/main", "origin/master"):
        try:
            subprocess.check_output(
                ["git", "rev-parse", "--verify", remote],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            return remote
        except subprocess.CalledProcessError:
            continue
    return None


def _matches_patterns(path: str, patterns: Sequence[str]) -> bool:
    import re

    return any(re.search(p, path) for p in patterns)


def main() -> int:
    try:
        files = _changed_files()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for f in sorted(files):
        if _matches_patterns(f, _SENSITIVE_PATTERNS):
            print("determinism")
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
