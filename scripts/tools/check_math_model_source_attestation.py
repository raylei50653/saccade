#!/usr/bin/env python3
"""Fail closed when the audited math-model document or source bytes move.

The math model is a manually reviewed transcription of production code.  This
checker does not try to re-prove equations with regexes.  It verifies that:

* the attestation has the exact v1 schema and audited source inventory;
* the document and audit record still have their attested SHA-256 identities;
* every current source anchor still has its attested SHA-256 identity; and
* the audited git ref contains those same source bytes.

Any missing, extra, duplicated, malformed, symlinked, unreadable, or changed
input is a hard failure.  A pass establishes byte identity only; it is not a
semantic-equivalence proof, runtime measurement, or execution authorization.

Usage:
  uv run python scripts/tools/check_math_model_source_attestation.py
  uv run python scripts/tools/check_math_model_source_attestation.py --quiet
"""

# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_REL = "docs/reference/math_model_source_attestation_v1.json"
MODEL_REL = "docs/reference/math_model.md"
AUDIT_REL = "docs/research/tracker-decision/audit/math_model_drift_2026-08-30.md"
SCHEMA = "saccade_math_model_source_attestation_v1"
AUDITED_SOURCE_REF = "0e869feae627e3da2c6fe03365c6482671aafe2b"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# This is deliberately code-owned.  The manifest cannot silently shrink the
# audit surface by deleting a row or grow it without a checker review.
AUDITED_SOURCE_PATHS: tuple[str, ...] = (
    "configs/presets/mamba_whole_graph.yaml",
    "configs/presets/mamba_whole_graph_m.yaml",
    "include/tracking/kalman_gpu.cuh",
    "src/saccade/perception/eval/evaluator.py",
    "src/saccade/perception/eval/pipeline.py",
    "src/tracking/gmc_kernel.cu",
    "src/tracking/relink_gate.cu",
    "src/tracking/tracker_gpu.cu",
)

SCOPE = {
    "claim": "source_byte_identity_only",
    "excludes": [
        "execution_authority",
        "runtime_measurement",
        "semantic_equivalence",
    ],
}


class AttestationError(ValueError):
    """The attestation or one of its bound inputs cannot be trusted."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AttestationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_attestation(path: Path) -> dict[str, Any]:
    """Load JSON while rejecting duplicate keys and non-object roots."""
    if path.is_symlink():
        raise AttestationError(f"attestation must not be a symlink: {path}")
    if not path.is_file():
        raise AttestationError(f"attestation is missing: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot load attestation {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AttestationError("attestation root must be a JSON object")
    return payload


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_keys(
    value: Any,
    expected: frozenset[str],
    label: str,
    failures: list[str],
) -> bool:
    if not isinstance(value, dict):
        failures.append(f"{label} must be an object")
        return False
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        failures.append(f"{label} missing fields: {', '.join(missing)}")
    if unknown:
        failures.append(f"{label} has unknown fields: {', '.join(unknown)}")
    return not missing and not unknown


def _valid_sha256(value: Any, label: str, failures: list[str]) -> bool:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        failures.append(f"{label} must be a lowercase full SHA-256")
        return False
    return True


def _check_current_binding(
    *,
    path: str,
    expected_sha256: str,
    label: str,
    read_current: Callable[[str], bytes],
    failures: list[str],
) -> None:
    try:
        current = read_current(path)
    except (AttestationError, OSError) as exc:
        failures.append(f"{label} cannot be read: {exc}")
        return
    actual = _sha256(current)
    if actual != expected_sha256:
        failures.append(
            f"{label} changed: {path} sha256={actual}, "
            f"attested={expected_sha256}; re-audit before updating the attestation"
        )


def validate_attestation(
    payload: dict[str, Any],
    *,
    read_current: Callable[[str], bytes],
    read_at_ref: Callable[[str, str], bytes],
) -> list[str]:
    """Return every hard failure without granting any semantic authority."""
    failures: list[str] = []
    top_ok = _exact_keys(
        payload,
        frozenset({"schema", "scope", "document", "audit", "sources"}),
        "attestation",
        failures,
    )
    if not top_ok:
        return failures

    if payload["schema"] != SCHEMA:
        failures.append(f"schema must be {SCHEMA!r}")
    if payload["scope"] != SCOPE:
        failures.append(
            "scope must declare byte identity only and preserve all three exclusions"
        )

    document = payload["document"]
    if _exact_keys(document, frozenset({"path", "sha256"}), "document", failures):
        if document["path"] != MODEL_REL:
            failures.append(f"document.path must be {MODEL_REL!r}")
        if _valid_sha256(document["sha256"], "document.sha256", failures):
            if document["path"] == MODEL_REL:
                _check_current_binding(
                    path=MODEL_REL,
                    expected_sha256=document["sha256"],
                    label="model document",
                    read_current=read_current,
                    failures=failures,
                )

    audit = payload["audit"]
    source_ref: str | None = None
    if _exact_keys(
        audit,
        frozenset({"path", "sha256", "source_ref", "open_findings"}),
        "audit",
        failures,
    ):
        if audit["path"] != AUDIT_REL:
            failures.append(f"audit.path must be {AUDIT_REL!r}")
        if _valid_sha256(audit["sha256"], "audit.sha256", failures):
            if audit["path"] == AUDIT_REL:
                _check_current_binding(
                    path=AUDIT_REL,
                    expected_sha256=audit["sha256"],
                    label="audit record",
                    read_current=read_current,
                    failures=failures,
                )
        if (
            isinstance(audit["source_ref"], str)
            and COMMIT_RE.fullmatch(audit["source_ref"]) is not None
        ):
            if audit["source_ref"] == AUDITED_SOURCE_REF:
                source_ref = audit["source_ref"]
            else:
                failures.append(
                    f"audit.source_ref must be the reviewed head {AUDITED_SOURCE_REF}"
                )
        else:
            failures.append("audit.source_ref must be a lowercase full git commit id")
        if audit["open_findings"] != []:
            failures.append("audit.open_findings must be an explicit empty list")

    sources = payload["sources"]
    if not isinstance(sources, list):
        failures.append("sources must be an ordered list")
        return failures

    valid_rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(sources):
        label = f"sources[{index}]"
        if not _exact_keys(row, frozenset({"path", "sha256"}), label, failures):
            continue
        path = row["path"]
        digest = row["sha256"]
        if not isinstance(path, str):
            failures.append(f"{label}.path must be a string")
            continue
        if path in seen:
            failures.append(f"duplicate source path: {path}")
            continue
        seen.add(path)
        if not _valid_sha256(digest, f"{label}.sha256", failures):
            continue
        valid_rows.append((path, digest))

    actual_paths = tuple(path for path, _digest in valid_rows)
    if actual_paths != AUDITED_SOURCE_PATHS:
        failures.append(
            "source inventory/order drift: expected "
            + repr(list(AUDITED_SOURCE_PATHS))
            + ", got "
            + repr(list(actual_paths))
        )

    expected_path_set = set(AUDITED_SOURCE_PATHS)
    for path, digest in valid_rows:
        if path not in expected_path_set:
            continue
        _check_current_binding(
            path=path,
            expected_sha256=digest,
            label="source anchor",
            read_current=read_current,
            failures=failures,
        )
        if source_ref is None:
            continue
        try:
            historical = read_at_ref(source_ref, path)
        except (AttestationError, OSError) as exc:
            failures.append(f"audited ref cannot provide {path}: {exc}")
            continue
        historical_sha = _sha256(historical)
        if historical_sha != digest:
            failures.append(
                f"audited ref mismatch: {source_ref}:{path} sha256={historical_sha}, "
                f"attested={digest}"
            )

    return failures


def _repo_reader(root: Path) -> Callable[[str], bytes]:
    root = root.resolve()

    def read(rel: str) -> bytes:
        pure = PurePosixPath(rel)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise AttestationError(f"non-canonical repo path: {rel!r}")
        candidate = root
        for part in pure.parts:
            candidate /= part
            if candidate.is_symlink():
                raise AttestationError(f"bound input must not be a symlink: {rel}")
        if not candidate.is_file():
            raise AttestationError(f"bound input is missing or not a file: {rel}")
        return candidate.read_bytes()

    return read


def _git_reader(root: Path) -> Callable[[str, str], bytes]:
    def read(ref: str, rel: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{ref}:{rel}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise AttestationError(detail or f"git show exited {result.returncode}")
        return result.stdout

    return read


def check_repository(root: Path = REPO_ROOT) -> list[str]:
    manifest = root / MANIFEST_REL
    try:
        payload = load_attestation(manifest)
    except AttestationError as exc:
        return [str(exc)]
    return validate_attestation(
        payload,
        read_current=_repo_reader(root),
        read_at_ref=_git_reader(root),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print failures only")
    args = parser.parse_args(argv)

    failures = check_repository()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(
            "PASS: math_model document, audit, and 8 source anchors match the "
            "closed byte attestation"
        )
        print(
            "NOTE: byte identity only; no semantic-equivalence proof, runtime "
            "measurement, or execution authority"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
