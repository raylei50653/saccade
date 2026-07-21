#!/usr/bin/env python3
"""Tests structure contract: every test self-documents, and the index is fresh.

Tests-side mirror of ``check_scripts_structure.py``. Fail-closed (``--strict``)
conditions, each cheap to fix by editing the offending ``test_*.py``:

  S1  every tracked ``tests/**/test_*.py`` carries three valid header tags:
      ``# scope:`` (one or more values, each in SCOPES), ``# function:`` in
      FUNCTIONS, and ``# lifecycle:`` in LIFECYCLES.
  S2  every test file has a module docstring (so the index has a Summary line).
  S3  the generated index (per-dir README blocks + roll-up) is current, i.e.
      ``build_tests_index.py`` would not change anything on disk.
  S4  cross-axis consistency: ``lifecycle: quarantined`` iff the file lives under
      ``tests/research/`` (and vice versa).
  S5  a non-steady lifecycle (legacy/quarantined/deprecated/remove) must carry a
      non-empty ``# lifecycle-note:`` reason or tracking Issue (rule 5).
  S6  a newly-added test file (vs the upstream base) may not be
      ``lifecycle: unclassified`` (rule 4); skipped when no base resolves.

Usage:
    .venv/bin/python scripts/tools/check_tests_structure.py            # report, exit 0
    .venv/bin/python scripts/tools/check_tests_structure.py --strict   # exit 1 on violations
"""

# status: stable

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tests_index as idx  # noqa: E402

VALID_SCOPE = set(idx.SCOPES)
VALID_FUNCTION = set(idx.FUNCTIONS)
VALID_LIFECYCLE = set(idx.LIFECYCLES)
NON_STEADY = set(idx.NON_STEADY)
RESEARCH_PREFIX = "tests/research/"


def check_self_documentation() -> list[str]:
    problems: list[str] = []
    added = idx.added_tests()  # None => rule 4 (S6) not enforceable here
    for path in idx.tracked_tests():
        scopes, function, lifecycle, note, desc = idx.extract(path)

        if not scopes:
            problems.append(f"{path}: missing `# scope:` header [S1]")
        else:
            bad = [s for s in scopes if s not in VALID_SCOPE]
            if bad:
                problems.append(
                    f"{path}: invalid scope {bad} (want {sorted(VALID_SCOPE)}) [S1]"
                )

        if not function:
            problems.append(f"{path}: missing `# function:` header [S1]")
        elif function not in VALID_FUNCTION:
            problems.append(
                f"{path}: invalid function '{function}' (want {sorted(VALID_FUNCTION)}) [S1]"
            )

        if not lifecycle:
            problems.append(f"{path}: missing `# lifecycle:` header [S1]")
        elif lifecycle not in VALID_LIFECYCLE:
            problems.append(
                f"{path}: invalid lifecycle '{lifecycle}' (want {sorted(VALID_LIFECYCLE)}) [S1]"
            )

        if not desc:
            problems.append(f"{path}: missing module docstring [S2]")

        # S4: quarantined <=> lives under tests/research/
        in_research = path.startswith(RESEARCH_PREFIX)
        if lifecycle == "quarantined" and not in_research:
            problems.append(
                f"{path}: lifecycle 'quarantined' but not under {RESEARCH_PREFIX} [S4]"
            )
        elif in_research and lifecycle and lifecycle != "quarantined":
            problems.append(
                f"{path}: under {RESEARCH_PREFIX} but lifecycle '{lifecycle}' != 'quarantined' [S4]"
            )

        # S5: non-steady lifecycle must carry a reason/Issue.
        if lifecycle in NON_STEADY and not note:
            problems.append(
                f"{path}: lifecycle '{lifecycle}' needs a non-empty `# lifecycle-note:` reason/Issue [S5]"
            )

        # S6: a newly-added test may not be unclassified.
        if added is not None and lifecycle == "unclassified" and path in added:
            problems.append(
                f"{path}: newly-added test may not be lifecycle 'unclassified' [S6]"
            )
    return problems


def check_index_fresh() -> list[str]:
    writes, _ = idx.build()
    stale = []
    for p, content in writes.items():
        cur = p.read_text(encoding="utf-8") if p.exists() else None
        if cur != content:
            stale.append(
                f"{p.relative_to(idx.REPO)}: stale, run build_tests_index.py [S3]"
            )
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any tests-structure violation",
    )
    args = parser.parse_args()

    violations = check_self_documentation() + check_index_fresh()

    if violations:
        print(f"tests structure: {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v}")
        if args.strict:
            return 1
    else:
        print(
            f"tests structure: ok ({len(idx.tracked_tests())} tests self-documented, index fresh)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
