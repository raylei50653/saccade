#!/usr/bin/env python3
"""Check that relative markdown links in docs resolve to existing files.

Catches the #1 doc-rot failure mode: links left dangling after a file move.

Scans repo-root ``*.md`` plus everything under ``docs/``. For each markdown
link ``[text](target)`` it verifies the target exists:

- ``http(s)://`` / ``mailto:`` and pure ``#anchor`` links are skipped.
- a ``#fragment`` suffix is stripped before checking (anchors are not verified).
- targets starting with ``/`` resolve from the repo root; otherwise from the
  containing file's directory.
- Gitignored targets are reported separately as local artifact warnings,
  whether or not they exist. Tracked targets still require an existing file.
  Only indexed .gitignore rules apply, never machine-local ignore settings.

Exit code 1 if any link is broken, else 0.

Usage: uv run python3 scripts/tools/check_doc_links.py
"""
# status: stable

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
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


def link_paths(md_file: Path, target: str) -> tuple[Path, ...]:
    path_part = unquote(target.split("#", 1)[0])
    # Strip the repo's `file_path:line[:col]` clickable-reference suffix so the
    # underlying file is what gets checked (the line number is not a real path).
    path_part = re.sub(r":\d+(?::\d+)?$", "", path_part)
    if path_part.startswith("/"):
        # Leading slash is used both for real absolute paths and for
        # repo-root-relative links — accept either.
        paths = (Path(path_part), REPO_ROOT / path_part.lstrip("/"))
    else:
        paths = (md_file.parent / path_part,)
    # Normalize '..' without following local artifact symlinks: their presence
    # must not change which Git ignore rule applies.
    return tuple(Path(os.path.abspath(path)) for path in paths)


def link_exists(md_file: Path, target: str) -> bool:
    return any(path.exists() for path in link_paths(md_file, target))


def git_output(
    root: Path,
    *args: str,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    allowed: tuple[int, ...] = (0,),
) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, input=input, capture_output=True, env=env
    )
    if result.returncode not in allowed:
        raise RuntimeError(
            f"git {args[0]} failed: {os.fsdecode(result.stderr).strip()}"
        )
    return result.stdout


def ignored_paths(paths: set[Path]) -> set[Path]:
    """Batch-classify using an isolated copy of the index and its ignore rules."""
    relative = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in paths
        if path.is_relative_to(REPO_ROOT) and path != REPO_ROOT
    )
    if not relative:
        return set()
    index = git_output(REPO_ROOT, "ls-files", "--stage", "-z")
    object_format = git_output(REPO_ROOT, "rev-parse", "--show-object-format").strip()
    # Git must not inherit the caller's repository/index/config overrides.
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    with tempfile.TemporaryDirectory(prefix="doc-link-policy-") as directory:
        policy = Path(directory)
        git_output(
            policy,
            "init",
            "-q",
            "--template=",
            f"--object-format={os.fsdecode(object_format)}",
            env=env,
        )
        # --info-only needs no copied blob objects. The index preserves Git's
        # tracked-file protection, including force-added ignored targets.
        git_output(
            policy,
            "update-index",
            "--info-only",
            "-z",
            "--index-info",
            input=index,
            env=env,
        )
        for entry in index.split(b"\0"):
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            mode, oid, stage = metadata.split()
            if stage != b"0":
                raise RuntimeError("Git index has unresolved merge entries")
            path = Path(os.fsdecode(name))
            if path.name != ".gitignore" or mode not in (b"100644", b"100755"):
                continue
            target = policy / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                git_output(REPO_ROOT, "cat-file", "blob", os.fsdecode(oid))
            )
        # Fresh metadata excludes info/exclude; only indexed .gitignore files
        # are materialized. Explicitly disable even the default global ignore.
        output = git_output(
            policy,
            "-c",
            f"core.excludesFile={os.devnull}",
            "check-ignore",
            "--stdin",
            "-z",
            input=os.fsencode("\0".join(relative) + "\0"),
            env=env,
            allowed=(0, 1),
        )
    return {REPO_ROOT / os.fsdecode(path) for path in output.split(b"\0") if path}


def main() -> int:
    broken: list[tuple[Path, int, str]] = []
    artifacts: list[tuple[Path, int, str]] = []
    links: list[tuple[Path, int, str]] = []
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
            links.append((md_file, lineno, target))

    try:
        ignored = ignored_paths(
            {
                path
                for md_file, _, target in links
                for path in link_paths(md_file, target)
            }
        )
    except (OSError, RuntimeError) as exc:
        print(f"✗ cannot classify doc links: {exc}", file=sys.stderr)
        return 1

    for md_file, lineno, target in links:
        if any(path in ignored for path in link_paths(md_file, target)):
            artifacts.append((md_file, lineno, target))
        elif not link_exists(md_file, target):
            broken.append((md_file, lineno, target))

    if artifacts:
        print(
            f"⚠ {len(artifacts)} local artifact reference(s) (gitignored; warning only):"
        )
        for md_file, lineno, target in artifacts:
            print(f"  {md_file.relative_to(REPO_ROOT)}:{lineno}  →  {target}")

    checked = len(links) - len(artifacts)

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
