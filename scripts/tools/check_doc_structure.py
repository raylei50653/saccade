#!/usr/bin/env python3
"""Warn-only documentation structure / research index coverage checks.

Companion to ``check_doc_links.py``, ``check_doc_stale_paths.py``, and
``check_doc_freshness.py``. Implements the machine side of the Doc Structure
Contract (``docs/ownership/doc_structure_contract.md`` § C4 / C9):

  S1  Every ``docs/modules/<m>/research/*.md`` must be referenced (by basename)
      in ``docs/modules/<m>/README.md``.
  S2  Every note under ``docs/research/{pipeline,eval,training,reid,threads}/*.md``
      (except README.md) must be referenced by basename in the subdir README
      if it exists, else in ``docs/research/README.md``.
  S3  Every ``docs/modules/<m>/`` directory must contain README.md and TODO.md.

This checker is **warn-only** by default (exit 0 even with findings). Pass
``--strict`` to exit non-zero (reserved for a later hard phase).

Index detection currently uses basename substring match against the owning
README body. That is acceptable for warn-only hygiene; before enabling
``--strict`` in CI, switch to Markdown link parsing to reduce false positives
(basename mentioned only in prose or stale paths).

Usage: uv run python3 scripts/tools/check_doc_structure.py [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODULES_ROOT = REPO_ROOT / "docs" / "modules"
RESEARCH_ROOT = REPO_ROOT / "docs" / "research"
RESEARCH_SUBDIRS = ("pipeline", "eval", "training", "reid", "threads")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _notes_in(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    return sorted(
        p for p in dir_path.glob("*.md") if p.name != "README.md" and p.is_file()
    )


def check_module_packages() -> list[str]:
    warnings: list[str] = []
    if not MODULES_ROOT.is_dir():
        return [f"[S3] missing modules root: {MODULES_ROOT.relative_to(REPO_ROOT)}"]

    for mod_dir in sorted(p for p in MODULES_ROOT.iterdir() if p.is_dir()):
        rel = mod_dir.relative_to(REPO_ROOT).as_posix()
        for required in ("README.md", "TODO.md"):
            if not (mod_dir / required).is_file():
                warnings.append(f"[S3] {rel}: missing {required}")
    return warnings


def check_module_research_indexes() -> list[str]:
    warnings: list[str] = []
    if not MODULES_ROOT.is_dir():
        return warnings

    for mod_dir in sorted(p for p in MODULES_ROOT.iterdir() if p.is_dir()):
        research = mod_dir / "research"
        if not research.is_dir():
            continue
        readme = mod_dir / "README.md"
        body = _read(readme)
        mod_rel = mod_dir.relative_to(REPO_ROOT).as_posix()
        if not body:
            warnings.append(
                f"[S1] {mod_rel}/research: parent README.md missing or empty; "
                "cannot verify index coverage"
            )
            continue
        for note in _notes_in(research):
            if note.name not in body:
                note_rel = note.relative_to(REPO_ROOT).as_posix()
                warnings.append(
                    f"[S1] {note_rel}: not referenced in {mod_rel}/README.md "
                    "(Doc Structure C4 — add index row)"
                )
    return warnings


def check_global_research_indexes() -> list[str]:
    warnings: list[str] = []
    top_readme = RESEARCH_ROOT / "README.md"
    top_body = _read(top_readme)

    for sub in RESEARCH_SUBDIRS:
        sub_dir = RESEARCH_ROOT / sub
        if not sub_dir.is_dir():
            continue
        sub_readme = sub_dir / "README.md"
        if sub_readme.is_file():
            index_body = _read(sub_readme)
            index_label = sub_readme.relative_to(REPO_ROOT).as_posix()
        else:
            index_body = top_body
            index_label = top_readme.relative_to(REPO_ROOT).as_posix()

        if not index_body:
            warnings.append(
                f"[S2] docs/research/{sub}: index file missing/empty ({index_label})"
            )
            continue

        for note in _notes_in(sub_dir):
            if note.name not in index_body:
                note_rel = note.relative_to(REPO_ROOT).as_posix()
                warnings.append(
                    f"[S2] {note_rel}: not referenced in {index_label} "
                    "(Doc Structure C4 — add index row)"
                )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on warnings (future hard phase; default warn-only)",
    )
    args = parser.parse_args()

    warnings: list[str] = []
    warnings.extend(check_module_packages())
    warnings.extend(check_module_research_indexes())
    warnings.extend(check_global_research_indexes())

    if warnings:
        print(f"doc structure: {len(warnings)} warning(s)")
        for w in warnings:
            print(f"  {w}")
    else:
        print("doc structure: ok (no index coverage warnings)")

    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
