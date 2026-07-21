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
  S5  a note-required lifecycle (legacy/quarantined/deprecated/remove) must carry a
      non-empty ``# lifecycle-note:`` reason or tracking Issue (rule 5).
  S6  a newly-added test file (vs the upstream base) may not be
      ``lifecycle: unclassified`` (rule 4); skipped when no base resolves.

Each of scope/function/lifecycle must appear exactly once (scope carries multiple
values via commas, not repeated lines), lifecycle-note at most once — enforced
under S1. Lifecycle is intentionally NOT tied to directory: a test's location
(e.g. tests/research/) is organisation, not governance state; keep quarantined
out of the formal suite via markers/collection, not a path rule.

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
NOTE_REQUIRED = set(idx.NOTE_REQUIRED_LIFECYCLES)


def check_self_documentation() -> list[str]:
    problems: list[str] = []
    added = idx.added_tests()  # None => rule 4 (S6) not enforceable here
    for path in idx.tracked_tests():
        scopes, function, lifecycle, note, desc = idx.extract(path)
        counts = idx.tag_multiplicity(path)

        # Exactly-once: scope/function/lifecycle once; lifecycle-note at most once.
        for key in ("scope", "function", "lifecycle"):
            if counts[key] > 1:
                problems.append(
                    f"{path}: `# {key}:` appears {counts[key]}× (must appear exactly once) [S1]"
                )
        if counts["lifecycle-note"] > 1:
            problems.append(
                f"{path}: `# lifecycle-note:` appears {counts['lifecycle-note']}× (at most once) [S1]"
            )

        if not scopes:
            problems.append(f"{path}: missing `# scope:` header [S1]")
        else:
            bad = [s for s in scopes if s not in VALID_SCOPE]
            if bad:
                problems.append(
                    f"{path}: invalid scope {bad} (want {sorted(VALID_SCOPE)}) [S1]"
                )
            dupes = sorted({s for s in scopes if scopes.count(s) > 1})
            if dupes:
                problems.append(f"{path}: duplicate scope value(s) {dupes} [S1]")

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

        # S5: note-required lifecycle must carry a reason/Issue.
        if lifecycle in NOTE_REQUIRED and not note:
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
