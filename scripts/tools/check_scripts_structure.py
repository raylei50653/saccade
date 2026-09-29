#!/usr/bin/env python3
"""Scripts structure report: every script should self-document.

Mirrors ``check_doc_structure.py`` for the ``scripts/`` tree. Conditions, each
cheap to fix by editing the offending script:

  S1  every tracked ``scripts/**.{py,sh}`` has a ``# status: <label>`` header whose
      label is one of the triage labels (stable/diagnostic/experiment/archive-candidate/generated).
  S2  every ``.py`` has a module docstring; every ``.sh`` has a description comment
      near the top (so the index can show a function line).

These are header/format rules, so the default gate runs this report-only
(#493 PR-2): ``pre_push.sh`` and CI do not pass ``--strict``. The generated
index is a derived snapshot, not checked here: CI regenerates and publishes it
(``build_scripts_index.py``); the checked-in copy may lag.

Usage:
    .venv/bin/python scripts/tools/check_scripts_structure.py            # report, exit 0
    .venv/bin/python scripts/tools/check_scripts_structure.py --strict   # exit 1 on violations
"""

# status: stable

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scripts_index as idx  # noqa: E402

VALID = set(idx.LABELS)


def check_self_documentation() -> list[str]:
    problems: list[str] = []
    for path in idx.tracked_scripts():
        status, desc, _usage = idx.extract(path)
        if not status:
            problems.append(f"{path}: missing `# status:` header [S1]")
        elif status not in VALID:
            problems.append(
                f"{path}: invalid status '{status}' (want {sorted(VALID)}) [S1]"
            )
        if not desc:
            kind = "module docstring" if path.endswith(".py") else "description comment"
            problems.append(f"{path}: missing {kind} [S2]")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any scripts-structure violation",
    )
    args = parser.parse_args()

    violations = check_self_documentation()

    if violations:
        print(f"scripts structure: {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v}")
        if args.strict:
            return 1
    else:
        print(
            f"scripts structure: ok ({len(idx.tracked_scripts())} scripts self-documented)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
