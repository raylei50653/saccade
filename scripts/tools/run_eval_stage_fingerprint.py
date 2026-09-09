#!/usr/bin/env python3
"""Run one ``mot17.py`` eval with opt-in per-stage fingerprints (issue #363).

Default production eval is unchanged.  This wrapper injects hash-first
stage snapshots into a child process and writes compact fingerprints
under ``--fingerprint-dir``.  It does not claim a causal mechanism.
"""
# status: stable

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SCRIPT_DIR))

from eval_stage_fingerprint import (  # noqa: E402
    StageFingerprintCollector,
    install_eval_hooks,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fingerprint-dir",
        type=Path,
        required=True,
        help="Directory for manifest.json + fingerprints.jsonl",
    )
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Persist hashes without compact integer rows",
    )
    args, forwarded = parser.parse_known_args(argv)
    if "--processes" in forwarded or any(
        item.startswith("--processes=") for item in forwarded
    ):
        parser.error("--processes is incompatible with per-stage fingerprints")
    if "--cpp-threads" in forwarded or any(
        item.startswith("--cpp-threads=") for item in forwarded
    ):
        parser.error("--cpp-threads cannot expose Python-evaluator stage callbacks")
    if "--workbench" in forwarded:
        parser.error("--workbench bypasses the Python evaluator stage boundaries")
    return args, forwarded


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded = parse_args(argv)
    fingerprint_dir = args.fingerprint_dir.resolve()
    fingerprint_dir.mkdir(parents=True, exist_ok=True)
    collector = StageFingerprintCollector(
        fingerprint_dir, include_payloads=not args.hash_only
    )
    undo = install_eval_hooks(collector)
    original_argv = sys.argv[:]
    eval_script_dir = str(_ROOT / "scripts" / "eval")
    inserted = eval_script_dir not in sys.path
    if inserted:
        sys.path.insert(0, eval_script_dir)
    exit_code = 0
    try:
        sys.argv = [str(_ROOT / "scripts" / "eval" / "mot17.py"), *forwarded]
        runpy.run_path(
            str(_ROOT / "scripts" / "eval" / "mot17.py"), run_name="__main__"
        )
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            exit_code = 1
    finally:
        undo()
        sys.argv = original_argv
        if inserted:
            sys.path.remove(eval_script_dir)
        collector.finalize()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
