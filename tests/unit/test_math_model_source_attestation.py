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
) -> tuple[dict[str, Any], dict[str, bytes], Callable[..., list[str]]]:
    """Build a passing attestation over synthetic bytes, plus a bound validator.

    The validator pins the code-owned digests to the *initial* document and
    audit bytes, so a test that rewrites a file and re-signs the manifest still
    has to clear the checker-side authority.
    """
    current = {
        checker.MODEL_REL: b"model",
        checker.AUDIT_REL: b"audit",
        **{path: f"source:{path}".encode() for path in checker.AUDITED_SOURCE_PATHS},
    }
    ref = checker.AUDITED_SOURCE_REF
    attested_model_sha256 = _digest(current[checker.MODEL_REL])
    attested_audit_sha256 = _digest(current[checker.AUDIT_REL])
    payload = {
        "schema": checker.SCHEMA,
        "scope": deepcopy(checker.SCOPE),
        "document": {
            "path": checker.MODEL_REL,
            "sha256": attested_model_sha256,
        },
        "audit": {
            "path": checker.AUDIT_REL,
            "sha256": attested_audit_sha256,
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

    def validate(
        *, read_at_ref: Callable[[str, str], bytes] = read_at_ref
    ) -> list[str]:
        return checker.validate_attestation(
            payload,
            read_current=read_current,
            read_at_ref=read_at_ref,
            attested_model_sha256=attested_model_sha256,
            attested_audit_sha256=attested_audit_sha256,
        )

    return payload, current, validate


def test_valid_synthetic_attestation_passes(checker: ModuleType) -> None:
    _payload, _current, validate = _synthetic(checker)
    assert validate() == []


def test_unknown_manifest_field_fails_closed(checker: ModuleType) -> None:
    payload, _current, validate = _synthetic(checker)
    payload["semantic_equivalence"] = True
    assert any("unknown fields" in failure for failure in validate())


def test_source_inventory_cannot_shrink(checker: ModuleType) -> None:
    payload, _current, validate = _synthetic(checker)
    payload["sources"].pop()
    assert any("source inventory/order drift" in failure for failure in validate())


def test_duplicate_source_path_fails_closed(checker: ModuleType) -> None:
    payload, _current, validate = _synthetic(checker)
    payload["sources"][1] = deepcopy(payload["sources"][0])
    assert any("duplicate source path" in failure for failure in validate())


def test_current_source_drift_fails(checker: ModuleType) -> None:
    _payload, current, validate = _synthetic(checker)
    changed_path = checker.AUDITED_SOURCE_PATHS[0]
    current[changed_path] += b"-changed"
    failures = validate()
    assert any(changed_path in failure and "changed" in failure for failure in failures)


def test_model_document_drift_fails(checker: ModuleType) -> None:
    _payload, current, validate = _synthetic(checker)
    current[checker.MODEL_REL] += b"-changed"
    assert any("model document changed" in failure for failure in validate())


def test_manifest_cannot_resign_the_model_document(checker: ModuleType) -> None:
    """Rewriting the document and its own manifest digest must still fail."""
    payload, current, validate = _synthetic(checker)
    current[checker.MODEL_REL] += b"-changed"
    payload["document"]["sha256"] = _digest(current[checker.MODEL_REL])
    failures = validate()
    assert any(
        "document.sha256 is not the code-owned digest" in failure
        for failure in failures
    )
    assert not any("model document changed" in failure for failure in failures)


def test_manifest_cannot_resign_the_audit_record(checker: ModuleType) -> None:
    """Rewriting the audit record and its own manifest digest must still fail."""
    payload, current, validate = _synthetic(checker)
    current[checker.AUDIT_REL] += b"-changed"
    payload["audit"]["sha256"] = _digest(current[checker.AUDIT_REL])
    failures = validate()
    assert any(
        "audit.sha256 is not the code-owned digest" in failure for failure in failures
    )
    assert not any("audit record changed" in failure for failure in failures)


def test_historical_ref_must_contain_attested_source_bytes(
    checker: ModuleType,
) -> None:
    _payload, current, validate = _synthetic(checker)
    changed_path = checker.AUDITED_SOURCE_PATHS[-1]

    def drifted_ref(_source_ref: str, path: str) -> bytes:
        return b"wrong" if path == changed_path else current[path]

    assert any(
        "audited ref mismatch" in failure
        for failure in validate(read_at_ref=drifted_ref)
    )


def test_source_ref_cannot_be_substituted(checker: ModuleType) -> None:
    payload, _current, validate = _synthetic(checker)
    payload["audit"]["source_ref"] = "b" * 40
    assert any("reviewed head" in failure for failure in validate())


def test_open_findings_fail_closed(checker: ModuleType) -> None:
    payload, _current, validate = _synthetic(checker)
    payload["audit"]["open_findings"] = ["STALE"]
    assert any("open_findings" in failure for failure in validate())


def test_duplicate_json_keys_are_rejected(checker: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    path.write_text('{"schema": "one", "schema": "two"}', encoding="utf-8")
    with pytest.raises(checker.AttestationError, match="duplicate JSON key"):
        checker.load_attestation(path)


def test_checked_in_attestation_is_current(checker: ModuleType) -> None:
    assert checker.check_repository(_REPO) == []


def test_code_owned_digests_match_the_checked_in_manifest(checker: ModuleType) -> None:
    """The default authority is the real one, not just whatever tests inject."""
    payload = checker.load_attestation(_REPO / checker.MANIFEST_REL)
    assert payload["document"]["sha256"] == checker.ATTESTED_MODEL_SHA256
    assert payload["audit"]["sha256"] == checker.ATTESTED_AUDIT_SHA256
