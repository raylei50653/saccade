"""Independent v2 authority-envelope verification and corpus admission."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
_CONTRACT_TESTS = Path(__file__).resolve().parent
for path in (_TOOLS, _CONTRACT_TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import check_h2_measure_archives as corpus  # noqa: E402
import h2_execution_driver as driver  # noqa: E402
import h2_import_witness as import_witness  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_successor_authorization as authority  # noqa: E402
import test_h2_execution_producer as producer_fixtures  # noqa: E402
import verify_h2_execution as inner_verifier  # noqa: E402
import verify_h2_measurement_envelope as envelope_verifier  # noqa: E402


@pytest.fixture
def run_spec() -> dict[str, Any]:
    return run_spec_module.build_run_spec()


def _write(root: Path, name: str, document: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _grant(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "authority": request["authority"],
        "authorization_id": "a" * 64,
        "execution_domain": request["execution_domain"],
        "execution_id": request["execution_id"],
        "execution_semantics_projection_digest": request[
            "execution_semantics_projection_digest"
        ],
        "issued_by": request["requested_issuer"],
        "phase": request["phase"],
        "resolved_run_spec_digest": request["resolved_run_spec_digest"],
        "schema": authority.GRANT_SCHEMA,
    }


def _import_witness(run_spec: dict[str, Any]) -> dict[str, Any]:
    projection = run_spec["execution_semantics_projection"]
    closure = projection["execution_code_closure"]
    member = closure["members"][0]
    observations = [
        {
            "authority_domains": [import_witness.DOMAIN_CLOSURE],
            "length": member["length"],
            "loader": "SourceFileLoader",
            "module_names": ["fixture.module"],
            "origin_kind": "source",
            "path": member["path"],
            "sha256": member["sha256"],
        }
    ]
    return {
        "algorithm": import_witness.WITNESS_ALGORITHM,
        "authority": import_witness.WITNESS_AUTHORITY,
        "bootstrap": {
            "entry_module": "run_h2_measurement_child",
            "preloaded_repo_local_paths": sorted(import_witness.BOOTSTRAP_SELF_PATHS),
            "recorder_installed_before_entry_import": True,
            "schema": import_witness.BOOTSTRAP_SCHEMA,
        },
        "declared": {
            "execution_code_closure_digest": closure["digest"],
            "execution_semantics_projection_digest": projection["digest"],
            "roots": closure["roots"],
        },
        "digest": evidence.digest(observations),
        "observations": observations,
        "schema": import_witness.WITNESS_SCHEMA,
    }


def _packet(tmp_path: Path, run_spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "measurement-packet"
    archive = root / authority.ARCHIVE_DIR
    runs_root = root
    ledger = tmp_path / "ledger"
    request = authority.build_request(
        execution_id=producer_fixtures.EXECUTION_ID,
        run_spec=run_spec,
        ledger=ledger,
    )
    grant = _grant(request)
    receipt = authority.receipt_for(request, grant, consumed_utc="2026-08-03T00:00:00Z")
    domain = authority.live_domain(ledger)
    for name, document in (
        (authority.REQUEST_NAME, request),
        (authority.GRANT_NAME, grant),
        (authority.DOMAIN_NAME, domain),
        (authority.RECEIPT_NAME, receipt),
    ):
        _write(root, name, document)

    producer_fixtures._execution(
        run_spec,
        result_schema=authority.RESULT_SCHEMA_V2,
        authorization_binding_digest=authority.canonical_digest(receipt),
    ).produce(archive)

    result_path = archive / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for run in result["ordered_runs"]:
        directory = evidence.run_dir(
            runs_root, result["run_plan"]["sequence"], run["run_id"]
        )
        _write(
            directory,
            evidence.POLICY_INVENTORY_NAME,
            {"schema": evidence.POLICY_INVENTORY_SCHEMA},
        )
        _write(directory, import_witness.WITNESS_NAME, _import_witness(run_spec))
        run["artifact_digest"] = driver.run_artifact_digest(runs_root, run["run_id"])
    _write(archive, "result.json", result)
    _, inner = inner_verifier.commit_verification(archive)
    assert inner["valid"] is True
    return root, domain


def test_closed_measurement_envelope_is_independently_valid(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    before = envelope_verifier.verify_packet(root)
    assert before["valid"] is True, before["reasons"]
    _, committed = envelope_verifier.commit_verification(root)
    assert committed["valid"] is True
    assert envelope_verifier.verify_packet(root) == committed
    assert (
        root
        / authority.RUNS_DIR
        / driver.producer.sequence()
        / "00_capture_off"
        / evidence.POLICY_INVENTORY_NAME
    ).is_file()
    assert not (root / authority.RUNS_DIR / authority.RUNS_DIR).exists()


def test_half_closed_measurement_envelope_is_refused(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    document = envelope_verifier.verify_packet(root)
    evidence.write_document_exclusive(
        root, authority.ENVELOPE_VERIFICATION_NAME, document
    )
    observed = envelope_verifier.verify_packet(root)
    assert observed["valid"] is False
    assert any("half closed" in reason for reason in observed["reasons"])


def test_commit_atomically_publishes_only_the_closed_final_packet(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    incomplete, _ = _packet(tmp_path, run_spec)
    final = tmp_path / "final-measurement-packet"
    path, document = envelope_verifier.commit_verification(incomplete, final)
    assert not incomplete.exists()
    assert path == final / authority.ENVELOPE_VERIFICATION_NAME
    assert document["valid"] is True
    assert envelope_verifier.verify_packet(final) == document


def test_controlled_domain_packet_is_corpus_admitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_spec: dict[str, Any],
) -> None:
    root, domain = _packet(tmp_path, run_spec)
    envelope_verifier.commit_verification(root)
    monkeypatch.setattr(corpus, "controlled_host_execution_domain", lambda: domain)
    attempts = corpus.check_corpus([root])
    assert len(attempts) == 1
    assert attempts[0].verify_class == "successor"


def test_disposable_domain_is_the_only_corpus_refusal(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    envelope_verifier.commit_verification(root)
    reasons = corpus.successor_packet_admission_reasons(root)
    assert len(reasons) == 1
    assert "controlled authorization execution domain" in reasons[0]


def test_receipt_digest_mismatch_invalidates_the_envelope(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    result_path = root / authority.ARCHIVE_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["authorization_binding_digest"] = "f" * 64
    _write(result_path.parent, result_path.name, result)
    inner = inner_verifier.verify_archive(root / authority.ARCHIVE_DIR)
    assert inner["valid"] is False
    envelope = envelope_verifier.verify_packet(root)
    assert envelope["valid"] is False
    assert any("consumption receipt" in reason for reason in envelope["reasons"])


def test_unbound_import_invalidates_run_evidence(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    path = (
        root
        / authority.RUNS_DIR
        / driver.producer.sequence()
        / "00_capture_off"
        / import_witness.WITNESS_NAME
    )
    witness = json.loads(path.read_text(encoding="utf-8"))
    witness["observations"][0]["authority_domains"] = []
    witness["digest"] = evidence.digest(witness["observations"])
    _write(path.parent, path.name, witness)
    observed = envelope_verifier.verify_packet(root)
    assert observed["valid"] is False
    assert any("unbound repository code" in reason for reason in observed["reasons"])


def test_request_without_packet_identity_is_unformable(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    path = root / authority.REQUEST_NAME
    request = json.loads(path.read_text(encoding="utf-8"))
    del request["execution_id"]
    _write(root, authority.REQUEST_NAME, request)
    with pytest.raises(
        envelope_verifier.EnvelopeVerificationError, match="cannot identify"
    ):
        envelope_verifier.verify_packet(root)


def test_packet_discovery_uses_the_nested_family_anchor(
    tmp_path: Path, run_spec: dict[str, Any]
) -> None:
    root, _ = _packet(tmp_path, run_spec)
    assert corpus.archive_roots(tmp_path) == [root]
    assert corpus._is_successor_packet(root)


def test_envelope_verifier_reads_no_live_host_identity() -> None:
    tree = ast.parse(
        (_TOOLS / "verify_h2_measurement_envelope.py").read_text(encoding="utf-8")
    )
    referenced = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"getuid", "gethostname", "environ", "machine_id"} & referenced
