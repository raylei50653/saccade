"""Prospective H0 repair/qualification gate contracts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
WORKFLOW = ROOT / ".github/workflows/h0_qualification.yml"
sys.path.insert(0, TOOLS.as_posix())

import build_h0_preseal_freeze as freezer  # noqa: E402
import check_h0_phase_a_archives as archive_corpus  # noqa: E402
import check_h0_repair_acceptance_matrix as matrix  # noqa: E402
import qualify_h0_phase_a as qualification  # noqa: E402
import verify_h0_phase_a_archive as archive  # noqa: E402
import run_h0_phase_a as controller  # noqa: E402


def test_repair_acceptance_matrix_is_complete() -> None:
    value = matrix.load_matrix()
    matrix.validate_matrix(value)
    assert [gate["id"] for gate in value["gates"]] == list(matrix.GATES)
    assert value["qualification"]["authority"] == "non_authoritative"


def test_repair_acceptance_matrix_rejects_every_requirement_mutation() -> None:
    source = matrix.load_matrix()
    for gate_index, gate in enumerate(source["gates"]):
        requirements = gate["required"]
        assert isinstance(requirements, list)
        for requirement_index in range(len(requirements)):
            for mutation in ("delete", "replace", "reorder", "add"):
                value = copy.deepcopy(source)
                required = value["gates"][gate_index]["required"]
                assert isinstance(required, list)
                if mutation == "delete":
                    required.pop(requirement_index)
                elif mutation == "replace":
                    required[requirement_index] = "optional_note"
                elif mutation == "reorder":
                    other = (requirement_index + 1) % len(required)
                    required[requirement_index], required[other] = (
                        required[other],
                        required[requirement_index],
                    )
                else:
                    required.append("optional_note")
                with pytest.raises(matrix.MatrixError, match="requirements differ"):
                    matrix.validate_matrix(value)


def test_qualification_workspace_refuses_authoritative_or_ambiguous_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "build").mkdir()
    evidence = root / "docs/modules/semantic/research/evidence"
    evidence.mkdir(parents=True)
    with pytest.raises(qualification.QualificationError, match="authoritative"):
        qualification._workspace(root, root / "build/h0_phase_a")
    with pytest.raises(qualification.QualificationError, match="authoritative"):
        qualification._workspace(root, evidence / "qualification")
    with pytest.raises(qualification.QualificationError, match="below"):
        qualification._workspace(root, root / "scratch")
    workspace = qualification._workspace(root, root / "build/h0_qualification/run-1")
    assert workspace.is_dir()


def test_qualification_failure_probe_is_truthful_but_non_authoritative() -> None:
    probe = qualification._failure_probe()
    assert probe["qualification_only"] is True
    assert probe["failure"]["stage"] == "checkpoint_T1"
    assert {
        key: value for key, value in probe["row"].items() if key != "monotonic_ns"
    } == {
        "cause": "inventory_mismatch",
        "digest": None,
        "events_after": [],
        "events_before": [],
        "inventory_comparison_executed": True,
        "inventory_equal": False,
        "name": "T1",
        "observed_digest": "0" * 64,
        "state": "failed",
    }


def test_qualification_runner_uses_an_explicit_synthetic_child() -> None:
    vector = qualification.qualification_runner_argv(
        Path("/tmp/build"),
        Path("/tmp/build/extension.so"),
        Path("/tmp/build/plugin.so"),
    )
    assert vector[:3] == [sys.executable, "-I", "-B"]
    assert vector[3].endswith("scripts/tools/qualify_h0_phase_a_child.py")
    assert "run_h0_phase_a_child.py" not in vector


def test_qualification_t1_verdict_semantics_uses_controller_producer() -> None:
    inventory = {"digest": "a" * 64, "records": []}
    row = controller._checkpoint_inventory_verdict("T1", inventory, dict(inventory))
    assert row["state"] == "completed"
    assert row["inventory_equal"] is True
    assert row["observed_digest"] == inventory["digest"]


def test_qualification_report_binds_resolved_repository_identity(
    tmp_path: Path,
) -> None:
    identity = qualification._repository_identity(
        ROOT, "agent/h0-repair-qualification-gates"
    )
    report = {"result": "passed", **identity}
    path = qualification._write_report(tmp_path, report)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["repository_head_sha"] == identity["repository_head_sha"]
    assert written["repository_tree_sha"] == identity["repository_tree_sha"]
    assert written["requested_ref"] == identity["requested_ref"]

    with pytest.raises(qualification.QualificationError, match="repository identity"):
        qualification._write_report(
            tmp_path,
            {
                "result": "passed",
                "repository_tree_sha": identity["repository_tree_sha"],
                "requested_ref": identity["requested_ref"],
            },
        )


def test_qualification_failure_report_retains_repository_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace = tmp_path / "workspace"
    identity = {
        "repository_head_sha": "a" * 40,
        "repository_tree_sha": "b" * 40,
        "requested_ref": "repair-head",
    }
    monkeypatch.setattr(qualification, "_repository_identity", lambda *_args: identity)
    monkeypatch.setattr(
        qualification,
        "_tool",
        lambda _name: (_ for _ in ()).throw(
            qualification.QualificationError("missing controlled-host tool")
        ),
    )

    with pytest.raises(qualification.QualificationError, match="controlled-host"):
        qualification.run_qualification(
            root,
            workspace,
            requested_ref="repair-head",
            timeout=1.0,
        )

    report = json.loads(
        (workspace / qualification.REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report["result"] == "failed"
    assert {key: report[key] for key in identity} == identity


def test_qualification_workflow_binds_and_publishes_qualified_head() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "id: qualified_commit" in workflow
    assert 'qualified_sha="$(git rev-parse --verify HEAD^{commit})"' in workflow
    assert '--requested-ref "$REQUESTED_REF"' in workflow
    assert (
        "name: h0-qualification-${{ steps.qualified_commit.outputs.sha }}" in workflow
    )
    assert workflow.count("if: ${{ !cancelled() }}") == 2


def test_archive_registry_verifies_immutable_v1_packet_without_execution_binding() -> (
    None
):
    evidence = (
        ROOT
        / "docs/modules/semantic/research/evidence"
        / "h0_phase_a_6ed30243554edfc898de32916298aa863673fced"
    )
    report = archive.verify_archive(evidence)
    assert report["valid"] is True
    assert report["codec_schema"] == "h0_phase_a_execution_v1"
    execution_paths = {path for path, _identity in freezer.IMPLEMENTATION_IDENTITIES}
    assert "scripts/tools/verify_h0_phase_a_archive.py" not in execution_paths
    assert "scripts/tools/qualify_h0_phase_a.py" not in execution_paths
    assert "scripts/tools/qualify_h0_phase_a_child.py" not in execution_paths


def test_archive_corpus_discovers_every_committed_phase_a_root() -> None:
    roots = archive_corpus.archive_roots()
    assert [root.name for root in roots] == [
        "h0_phase_a_1a8c13a890b3490bb7aa50dc2ab491db89b8b474",
        "h0_phase_a_42121c064cd1a3c4202e114cc6f4d8866a9e6af0",
        "h0_phase_a_6ed30243554edfc898de32916298aa863673fced",
    ]
