"""Shared helper that runpy-executes a canonical scripts/eval path for compat wrappers."""

# status: stable
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_eval_script(relative_path: str) -> None:
    """Run a canonical scripts/eval entrypoint while preserving CLI semantics."""
    target = Path(__file__).resolve().parent / relative_path
    old_argv0 = sys.argv[0]
    sys.argv[0] = str(target)
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv[0] = old_argv0
