#!/usr/bin/env python3
"""H0 declaration frozen-identity checks with SEALED-append tolerance.

GCTM / static-feasibility packages pin the H0 capture declaration via
``path + sha256`` at package freeze time.  A later H0 Seal appends one
owner-event ``SEALED`` row to that same file (Amendment 10 / A7.RC2).  That
append is authority-overlay delta, not a semantic rewrite of the frozen body.

Validators must accept worktree bytes whose SHA-256 equals the frozen digest
**or** that equal the frozen baseline plus one or more pure trailing SEALED
rows (byte-prefix of the frozen content recovered by peeling legal rows from
the end until the remaining prefix hashes to the frozen digest).

All other frozen inputs remain strict path+sha256 equality.
"""
# status: stable

from __future__ import annotations

import datetime
import hashlib
import re
from pathlib import Path

# Repo-relative path of the sole H0 owner-event declaration.
H0_CAPTURE_DECLARATION_RELPATH = (
    "docs/modules/semantic/research/"
    "headline_bridge_full_decision_capture_declaration_20260713.md"
)

# One Seal owner-event row (date + I + F + SEALED), no surrounding whitespace.
_SEALED_LINE = re.compile(
    r"^\| (?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}) \| "
    r"`(?P<instrumentation>[0-9a-f]{40})` \| "
    r"`(?P<freeze>[0-9a-f]{40})` \| "
    r"`SEALED` \|$"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_h0_capture_declaration_path(path: str | Path) -> bool:
    """True when *path* is the H0 capture declaration (posix, repo-relative)."""
    text = path.as_posix() if isinstance(path, Path) else path.replace("\\", "/")
    # Accept absolute paths that end with the repo-relative member.
    return text == H0_CAPTURE_DECLARATION_RELPATH or text.endswith(
        "/" + H0_CAPTURE_DECLARATION_RELPATH
    )


def is_legal_sealed_owner_event_line(line: str) -> bool:
    """Return True iff *line* is one complete SEALED owner-event row."""
    match = _SEALED_LINE.fullmatch(line)
    if match is None:
        return False
    try:
        datetime.date.fromisoformat(match.group("date"))
    except ValueError:
        return False
    return True


def matches_frozen_sha256_or_sealed_append(
    disk_bytes: bytes, expected_sha256: str
) -> bool:
    """True if disk equals frozen digest or frozen baseline + SEALED appends.

    Recovery peels legal SEALED rows from the end of *disk_bytes* until the
    remaining prefix hashes to *expected_sha256*.  Exact match is accepted
    without peeling.  Any non-SEALED trailing line, mid-file mutation, or
    non-UTF-8 suffix fails closed.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        return False
    if sha256_bytes(disk_bytes) == expected_sha256:
        return True
    data = disk_bytes
    peeled = 0
    while data:
        if not data.endswith(b"\n"):
            return False
        # Split off the last line (without its terminating LF).
        body = data[:-1]
        nl = body.rfind(b"\n")
        if nl < 0:
            last_line_bytes = body
            prefix = b""
        else:
            last_line_bytes = body[nl + 1 :]
            prefix = data[: nl + 1]
        try:
            last_line = last_line_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return False
        if not is_legal_sealed_owner_event_line(last_line):
            return False
        data = prefix
        peeled += 1
        if sha256_bytes(data) == expected_sha256:
            return peeled >= 1
    return False


def frozen_path_hash_ok(
    *,
    path: str | Path,
    disk_bytes: bytes,
    expected_sha256: str,
) -> bool:
    """Strict hash for non-declaration paths; SEALED-append-tolerant for H0 decl."""
    if is_h0_capture_declaration_path(path):
        return matches_frozen_sha256_or_sealed_append(disk_bytes, expected_sha256)
    return sha256_bytes(disk_bytes) == expected_sha256
