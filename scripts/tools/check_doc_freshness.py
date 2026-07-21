#!/usr/bin/env python3
"""Warn-only documentation freshness / fact-ownership checks.

Companion to ``check_doc_links.py`` (link resolution) and
``check_doc_stale_paths.py`` (moved-file denylist). This checker surfaces
*drift risk* in a fixed allowlist of entry / narrative docs. It is **warn-only**:
it never fails the build unless ``--strict`` is passed (reserved for a future
hard-ban phase). Research, ablation, archive and module docs are intentionally
out of scope so that legitimate historical numbers and dates are not flagged.

Checks:
  C4  hand-written "最後更新 / Last updated" dates in entry docs (dead metadata).
  C2  entry docs that quote the current-baseline signature must carry a
      ``<!-- fact-owner: current-baseline = ... -->`` marker.
  C1  cross-entry duplication: a non-owner entry doc that mirrors the baseline
      signature must point its marker at the fact owner.

Usage: uv run python3 scripts/tools/check_doc_freshness.py [--strict]
"""
# status: stable

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Entry / narrative docs that carry a freshness contract (repo-root relative).
ENTRY_DOCS = [
    "README.md",
    "DEVELOPMENT.md",
    "docs/TODO.md",
    "docs/PIPELINE.md",
    "docs/DATAFLOW.md",
    "docs/PROJECT_SHOWCASE.md",
    "docs/architecture/README.md",
]

# Fact owner for the current-baseline numbers.
BASELINE_OWNER = "docs/TODO.md"

FACT_MARKER_RE = re.compile(
    r"<!--\s*fact-owner:\s*current-baseline\s*=\s*([^\s>]+)\s*-->"
)

# Highly specific headline values; >= 2 present => the doc is quoting the baseline.
BASELINE_SIGNATURE = ["78.2", "78.4", "70.2", "269.47"]
SIGNATURE_MIN_HITS = 2

HANDWRITTEN_DATE_RE = re.compile(
    r"(最後更新|Last updated|last-updated)\s*[:：]?\s*\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)


def signature_hits(text: str) -> int:
    return sum(1 for tok in BASELINE_SIGNATURE if tok in text)


def check_doc(rel: str, text: str) -> list[str]:
    warnings: list[str] = []

    # C4 — hand-written dates are dead metadata.
    for lineno, line in enumerate(text.splitlines(), start=1):
        if HANDWRITTEN_DATE_RE.search(line):
            warnings.append(
                f"[C4] {rel}:{lineno}: hand-written date is dead metadata; "
                "use a fact-owner marker pointing at the living source"
            )

    # C2 / C1 — baseline-bearing docs must be marked and point at the owner.
    if signature_hits(text) >= SIGNATURE_MIN_HITS:
        marker = FACT_MARKER_RE.search(text)
        if not marker:
            warnings.append(
                f"[C2] {rel}: quotes current-baseline numbers but has no "
                f"fact-owner marker → add `<!-- fact-owner: current-baseline = {BASELINE_OWNER} -->`"
            )
        elif rel != BASELINE_OWNER and marker.group(1) != BASELINE_OWNER:
            warnings.append(
                f"[C1] {rel}: baseline marker points to {marker.group(1)}, "
                f"expected fact owner {BASELINE_OWNER}"
            )

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on warnings (future hard-ban phase; default warn-only)",
    )
    args = parser.parse_args()

    warnings: list[str] = []
    for rel in ENTRY_DOCS:
        path = REPO_ROOT / rel
        if not path.exists():
            warnings.append(f"[--] {rel}: listed entry doc is missing")
            continue
        warnings.extend(check_doc(rel, path.read_text(encoding="utf-8")))

    if warnings:
        print(f"! {len(warnings)} doc-freshness warning(s) (warn-only):")
        for warning in warnings:
            print(f"  {warning}")
        return 1 if args.strict else 0

    print("✓ doc freshness: entry docs carry fact-owner markers, no hand-written dates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
