"""Unit tests for H0 declaration frozen-hash SEALED-append tolerance."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "scripts" / "tools"
sys.path.insert(0, TOOLS.as_posix())

import h0_declaration_frozen_identity as decl_id  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sealed_line(
    *,
    date: str = "2026-07-24",
    instrumentation: str = "a" * 40,
    freeze: str = "b" * 40,
) -> str:
    return f"| {date} | `{instrumentation}` | `{freeze}` | `SEALED` |\n"


def test_exact_hash_match_passes() -> None:
    baseline = b"declaration body\n"
    assert decl_id.matches_frozen_sha256_or_sealed_append(baseline, _sha(baseline))


def test_single_sealed_append_passes() -> None:
    baseline = b"declaration body\n"
    disk = baseline + _sealed_line().encode()
    assert decl_id.matches_frozen_sha256_or_sealed_append(disk, _sha(baseline))


def test_multiple_sealed_appends_pass() -> None:
    baseline = b"declaration body\n"
    disk = (
        baseline
        + _sealed_line(instrumentation="c" * 40, freeze="d" * 40).encode()
        + _sealed_line(instrumentation="e" * 40, freeze="f" * 40).encode()
    )
    assert decl_id.matches_frozen_sha256_or_sealed_append(disk, _sha(baseline))


def test_mid_file_mutation_fails() -> None:
    baseline = b"declaration body\n"
    disk = b"mutated body\n" + _sealed_line().encode()
    assert not decl_id.matches_frozen_sha256_or_sealed_append(disk, _sha(baseline))


def test_illegal_trailing_line_fails() -> None:
    baseline = b"declaration body\n"
    disk = baseline + b"not a sealed row\n"
    assert not decl_id.matches_frozen_sha256_or_sealed_append(disk, _sha(baseline))


def test_invalid_calendar_date_fails() -> None:
    baseline = b"declaration body\n"
    disk = baseline + _sealed_line(date="2026-02-30").encode()
    assert not decl_id.matches_frozen_sha256_or_sealed_append(disk, _sha(baseline))


def test_non_declaration_path_is_strict() -> None:
    baseline = b"other\n"
    disk = baseline + _sealed_line().encode()
    assert decl_id.frozen_path_hash_ok(
        path="docs/other.md",
        disk_bytes=baseline,
        expected_sha256=_sha(baseline),
    )
    assert not decl_id.frozen_path_hash_ok(
        path="docs/other.md",
        disk_bytes=disk,
        expected_sha256=_sha(baseline),
    )


def test_declaration_path_allows_append() -> None:
    baseline = b"declaration body\n"
    disk = baseline + _sealed_line().encode()
    assert decl_id.frozen_path_hash_ok(
        path=decl_id.H0_CAPTURE_DECLARATION_RELPATH,
        disk_bytes=disk,
        expected_sha256=_sha(baseline),
    )


@pytest.mark.parametrize(
    "line",
    [
        "| 2026-07-24 | `aaaa` | `bbbb` | `SEALED` |",  # short hashes
        "| 2026-07-24 | `" + "a" * 40 + "` | `" + "b" * 40 + "` | `OPEN` |",
        "  | 2026-07-24 | `" + "a" * 40 + "` | `" + "b" * 40 + "` | `SEALED` |",
    ],
)
def test_malformed_sealed_lines_rejected(line: str) -> None:
    assert not decl_id.is_legal_sealed_owner_event_line(line)
