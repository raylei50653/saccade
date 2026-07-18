#!/usr/bin/env python3
"""Verify every committed H0 Phase-A evidence root through archive codecs."""

from __future__ import annotations

import sys
from pathlib import Path

import verify_h0_phase_a as execution_v1
import verify_h0_phase_a_archive as archive


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/modules/semantic/research/evidence"


def archive_roots(root: Path = EVIDENCE) -> list[Path]:
    values = sorted(
        root.glob("h0_phase_a_*"), key=lambda path: path.name.encode("utf-8")
    )
    result = [
        path for path in values if path.is_dir() and (path / "manifest.json").is_file()
    ]
    if not result:
        raise archive.ArchiveVerificationError("no committed H0 Phase-A evidence roots")
    return result


def main() -> int:
    try:
        roots = archive_roots()
        for root in roots:
            archive.verify_archive(root)
    except (archive.ArchiveVerificationError, execution_v1.VerificationError) as exc:
        print(f"H0 archive corpus rejected: {exc}", file=sys.stderr)
        return 1
    print(f"H0 archive corpus: PASS ({len(roots)} roots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
