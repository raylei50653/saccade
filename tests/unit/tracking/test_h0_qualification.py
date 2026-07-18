"""Prospective H0 repair/qualification gate contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
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


def test_qualification_t1_uses_the_controller_checkpoint_producer() -> None:
    inventory = {"digest": "a" * 64, "records": []}
    row = controller._checkpoint_inventory_verdict("T1", inventory, dict(inventory))
    assert row["state"] == "completed"
    assert row["inventory_equal"] is True
    assert row["observed_digest"] == inventory["digest"]


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
