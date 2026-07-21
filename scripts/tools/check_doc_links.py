#!/usr/bin/env python3
"""Check that relative markdown links in docs resolve to existing files.

Catches the #1 doc-rot failure mode: links left dangling after a file move.

Scans repo-root ``*.md`` plus everything under ``docs/``. For each markdown
link ``[text](target)`` it verifies the target exists:

- ``http(s)://`` / ``mailto:`` and pure ``#anchor`` links are skipped.
- a ``#fragment`` suffix is stripped before checking (anchors are not verified).
- targets starting with ``/`` resolve from the repo root; otherwise from the
  containing file's directory.

Exit code 1 if any link is broken, else 0.

Usage: uv run python3 scripts/tools/check_doc_links.py
"""
# status: stable

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]

# [text](target) — target captured up to the closing paren.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def iter_markdown_files() -> list[Path]:
    files = sorted(REPO_ROOT.glob("*.md"))
    files += sorted((REPO_ROOT / "docs").rglob("*.md"))
    return files


def extract_links(text: str) -> list[tuple[int, str]]:
    """Return (line_no, raw_target) pairs, skipping fenced code blocks."""
    links: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK_RE.finditer(line):
            # Drop an optional `"title"` after the URL.
            target = match.group(1).split()[0] if match.group(1).split() else ""
            if target:
                links.append((lineno, target))
    return links


def link_exists(md_file: Path, target: str) -> bool:
    path_part = unquote(target.split("#", 1)[0])
    # Strip the repo's `file_path:line[:col]` clickable-reference suffix so the
    # underlying file is what gets checked (the line number is not a real path).
    path_part = re.sub(r":\d+(?::\d+)?$", "", path_part)
    if path_part.startswith("/"):
        # Leading slash is used both for real absolute paths and for
        # repo-root-relative links — accept either.
        return Path(path_part).exists() or (REPO_ROOT / path_part.lstrip("/")).exists()
    return (md_file.parent / path_part).resolve().exists()


def main() -> int:
    broken: list[tuple[Path, int, str]] = []
    checked = 0
    for md_file in iter_markdown_files():
        text = md_file.read_text(encoding="utf-8")
        for lineno, target in extract_links(text):
            low = target.lower()
            if low.startswith(("http://", "https://", "mailto:")) or target.startswith(
                "#"
            ):
                continue
            if target.split("#", 1)[0] == "":  # pure anchor like (#section)
                continue
            checked += 1
            if not link_exists(md_file, target):
                broken.append((md_file, lineno, target))

    if broken:
        print(f"✗ {len(broken)} broken doc link(s) (of {checked} checked):")
        for md_file, lineno, target in broken:
            rel = md_file.relative_to(REPO_ROOT)
            print(f"  {rel}:{lineno}  →  {target}")
        return 1

    print(f"✓ all {checked} relative doc links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
