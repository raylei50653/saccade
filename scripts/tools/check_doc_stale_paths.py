#!/usr/bin/env python3
"""Fail if any tracked file references a pre-move (stale) doc path.

Phase 1 of the docs healthcheck relocated a fixed set of files out of the
``docs/`` root into ``docs/research/``. Plain-text references (code comments,
prose, non-link paths) are invisible to ``check_doc_links.py`` because it only
validates Markdown ``[text](target)`` links. This checker closes that gap with a
narrow, fixed denylist: it forbids the *old* locations of the Phase 1 moves and
nothing else. It is intentionally not a general freshness or stale-path scanner.

Scans version-controlled files only (``git ls-files``), so ``.gitignore`` and
generated output are respected automatically. This script excludes itself.

Exit code 1 if any stale path is found, else 0.

Usage: uv run python3 scripts/tools/check_doc_stale_paths.py
"""
# status: stable

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SELF_REL = "scripts/tools/check_doc_stale_paths.py"

# Old location -> new location. Fixed denylist: Phase 1 moves only.
STALE_PATHS: dict[str, str] = {
    "docs/sync_audit.md": "docs/research/pipeline/sync_audit_20260706.md",
    "docs/CPU_BOUND_ANALYSIS.md": "docs/research/pipeline/CPU_BOUND_ANALYSIS.md",
    "docs/CPU_OVERHEAD_ANALYSIS.md": "docs/research/pipeline/cpu_overhead_analysis_20260707.md",
    "docs/optimization_redundant_computations.md": "docs/research/pipeline/optimization_redundant_computations_20260620.md",
    "docs/pp22_full_cadence_interp_training_plan.md": "docs/research/training/pp22_full_cadence_interp_training_plan.md",
    "docs/pp22_stress_test_findings.md": "docs/research/training/pp22_stress_test_findings.md",
}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    hits: list[tuple[str, int, str, str]] = []
    for rel in tracked_files():
        if rel == SELF_REL:
            continue
        path = REPO_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for stale, new in STALE_PATHS.items():
                if stale in line:
                    hits.append((rel, lineno, stale, new))

    if hits:
        print(f"✗ {len(hits)} stale Phase 1 doc path reference(s):")
        for rel, lineno, stale, new in hits:
            print(f"  {rel}:{lineno}  {stale}  →  {new}")
        return 1

    print(f"✓ no stale Phase 1 doc paths ({len(STALE_PATHS)} denylisted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
