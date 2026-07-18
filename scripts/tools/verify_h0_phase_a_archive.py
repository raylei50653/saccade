#!/usr/bin/env python3
"""Versioned archive verifier for immutable H0 Phase-A evidence.

This registry is deliberately outside the execution implementation binding
universe. The controller continues to use ``verify_h0_phase_a.py`` as its
sealed execution verifier; archive-only compatibility repairs add codecs here
without changing a prospective execution authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import verify_h0_phase_a as execution_v1


ARCHIVE_SCHEMA = "h0_phase_a_archive_verifier_v1"
Codec = Callable[[Path], dict[str, Any]]


class ArchiveVerificationError(ValueError):
    """No registered archive codec can validate the evidence."""


def _v1_codec(root: Path) -> dict[str, Any]:
    return execution_v1.verify_evidence_root(root)


CODECS: dict[str, Codec] = {"h0_phase_a_execution_v1": _v1_codec}


def _manifest_schema(root: Path) -> str:
    path = root / "manifest.json"
    try:
        manifest = execution_v1.load_json(path, canonical_file=True)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(
            f"archive manifest is unreadable: {exc}"
        ) from exc
    schema = manifest.get("schema") if isinstance(manifest, Mapping) else None
    if not isinstance(schema, str):
        raise ArchiveVerificationError("archive manifest lacks an execution schema")
    return schema


def verify_archive(root: Path) -> dict[str, Any]:
    """Verify one immutable evidence directory through its schema codec."""
    if root.is_symlink() or not root.is_dir():
        raise ArchiveVerificationError(
            "archive evidence root is not a physical directory"
        )
    schema = _manifest_schema(root)
    codec = CODECS.get(schema)
    if codec is None:
        raise ArchiveVerificationError(
            f"unsupported immutable evidence schema: {schema}"
        )
    report = codec(root)
    if report.get("valid") is not True:
        raise ArchiveVerificationError("archive codec did not return a valid report")
    return {
        "archive_schema": ARCHIVE_SCHEMA,
        "codec_schema": schema,
        "report": report,
        "valid": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_archive(args.evidence)
    except (
        ArchiveVerificationError,
        execution_v1.VerificationError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"H0 archive verification rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
