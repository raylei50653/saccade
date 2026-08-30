"""Fail-closed contracts for the math-model source-byte attestation."""

# scope: eval, tracking, cross-module
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import sys
from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "tools" / "check_math_model_source_attestation.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_math_model_source_attestation", _SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load_checker()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _synthetic(
    checker: ModuleType,
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    Callable[[str], bytes],
    Callable[[str, str], bytes],
]:
    current = {
        checker.MODEL_REL: b"model",
        checker.AUDIT_REL: b"audit",
        **{path: f"source:{path}".encode() for path in checker.AUDITED_SOURCE_PATHS},
    }
    ref = checker.AUDITED_SOURCE_REF
    payload = {
        "schema": checker.SCHEMA,
        "scope": deepcopy(checker.SCOPE),
        "document": {
            "path": checker.MODEL_REL,
            "sha256": _digest(current[checker.MODEL_REL]),
        },
        "audit": {
            "path": checker.AUDIT_REL,
            "sha256": _digest(current[checker.AUDIT_REL]),
            "source_ref": ref,
            "open_findings": [],
        },
        "sources": [
            {"path": path, "sha256": _digest(current[path])}
            for path in checker.AUDITED_SOURCE_PATHS
        ],
    }

    def read_current(path: str) -> bytes:
        return current[path]

    def read_at_ref(source_ref: str, path: str) -> bytes:
        assert source_ref == ref
        return current[path]

    return payload, current, read_current, read_at_ref


def test_valid_synthetic_attestation_passes(checker: ModuleType) -> None:
    payload, _current, read_current, read_at_ref = _synthetic(checker)
    assert (
        checker.validate_attestation(
            payload, read_current=read_current, read_at_ref=read_at_ref
        )
        == []
    )


def test_unknown_manifest_field_fails_closed(checker: ModuleType) -> None:
    payload, _current, read_current, read_at_ref = _synthetic(checker)
    payload["semantic_equivalence"] = True
    failures = checker.validate_attestation(
        payload, read_current=read_current, read_at_ref=read_at_ref
    )
    assert any("unknown fields" in failure for failure in failures)


def test_source_inventory_cannot_shrink(checker: ModuleType) -> None:
    payload, _current, read_current, read_at_ref = _synthetic(checker)
    payload["sources"].pop()
    failures = checker.validate_attestation(
        payload, read_current=read_current, read_at_ref=read_at_ref
    )
    assert any("source inventory/order drift" in failure for failure in failures)


def test_duplicate_source_path_fails_closed(checker: ModuleType) -> None:
    payload, _current, read_current, read_at_ref = _synthetic(checker)
    payload["sources"][1] = deepcopy(payload["sources"][0])
    failures = checker.validate_attestation(
        payload, read_current=read_current, read_at_ref=read_at_ref
    )
    assert any("duplicate source path" in failure for failure in failures)


def test_current_source_drift_fails(checker: ModuleType) -> None:
    payload, current, _read_current, read_at_ref = _synthetic(checker)
    changed_path = checker.AUDITED_SOURCE_PATHS[0]
    current[changed_path] += b"-changed"
    failures = checker.validate_attestation(
        payload, read_current=current.__getitem__, read_at_ref=read_at_ref
    )
    assert any(changed_path in failure and "changed" in failure for failure in failures)


def test_model_document_drift_fails(checker: ModuleType) -> None:
    payload, current, _read_current, read_at_ref = _synthetic(checker)
    current[checker.MODEL_REL] += b"-changed"
    failures = checker.validate_attestation(
        payload, read_current=current.__getitem__, read_at_ref=read_at_ref
    )
    assert any("model document changed" in failure for failure in failures)


def test_historical_ref_must_contain_attested_source_bytes(
    checker: ModuleType,
) -> None:
    payload, _current, read_current, _read_at_ref = _synthetic(checker)
    changed_path = checker.AUDITED_SOURCE_PATHS[-1]

    def drifted_ref(_source_ref: str, path: str) -> bytes:
        return b"wrong" if path == changed_path else read_current(path)

    failures = checker.validate_attestation(
        payload, read_current=read_current, read_at_ref=drifted_ref
    )
    assert any("audited ref mismatch" in failure for failure in failures)


def test_source_ref_cannot_be_substituted(checker: ModuleType) -> None:
    payload, _current, read_current, read_at_ref = _synthetic(checker)
    payload["audit"]["source_ref"] = "b" * 40
    failures = checker.validate_attestation(
        payload, read_current=read_current, read_at_ref=read_at_ref
    )
    assert any("reviewed head" in failure for failure in failures)


def test_open_findings_fail_closed(checker: ModuleType) -> None:
    payload, _current, read_current, read_at_ref = _synthetic(checker)
    payload["audit"]["open_findings"] = ["STALE"]
    failures = checker.validate_attestation(
        payload, read_current=read_current, read_at_ref=read_at_ref
    )
    assert any("open_findings" in failure for failure in failures)


def test_duplicate_json_keys_are_rejected(checker: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    path.write_text('{"schema": "one", "schema": "two"}', encoding="utf-8")
    with pytest.raises(checker.AttestationError, match="duplicate JSON key"):
        checker.load_attestation(path)


def test_checked_in_attestation_is_current(checker: ModuleType) -> None:
    assert checker.check_repository(_REPO) == []
