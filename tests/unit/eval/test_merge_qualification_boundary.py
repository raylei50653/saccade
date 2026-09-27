"""Fail-closed boundaries of the offline merge qualification verifier."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

PACKET = Path(
    "docs/modules/semantic/research/evidence/merge_sparse_equivalence_20260927"
)


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location(
        "merge_acceptance", PACKET / "verify_acceptance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_capture_passes_without_claiming_current_qualification(
    verifier, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["verify", "--archive-only"])
    verifier.main()
    result = json.loads(capsys.readouterr().out)
    assert result["all_passed"]
    assert result["reference_sequences"] == 49
    assert result["verification_scope"] == "sealed_capture"
    assert result["current_runtime_qualified"] is False
    assert result["production_eligible"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "gpu",
        "torch",
        "cuda",
        "dtype",
        "block",
        "forced_block",
        "reduction",
        "topk",
        "cublas",
        "autocast",
        "tf32",
        "precision",
        "deterministic",
        "merge",
        "commit",
        "dirty",
        "coverage",
        "row_chunk",
        "embedding",
        "sparse_hash",
        "forced_hash",
    ],
)
def test_original_review_mutations_fail_closed(verifier, monkeypatch, mutation):
    original_load = verifier.load

    def mutated_load(path):
        q = original_load(path)
        if path != verifier.PACKET / "qualification/mot17.json":
            return q
        row = q["sequences"]["MOT17-02-SDP"]
        numeric = q["numeric_contract"]
        if mutation == "source":
            q["source_sha256"]["src/saccade/perception/eval/cheb_gr_merge.py"] = (
                "0" * 64
            )
        elif mutation == "gpu":
            q["device"] = "unqualified GPU"
        elif mutation in ("torch", "cuda", "dtype", "reduction", "topk"):
            numeric[mutation] = "unqualified"
        elif mutation == "block":
            numeric["default_block_elems"] = 16
        elif mutation == "forced_block":
            q["forced_block_elems"] = 16
        elif mutation == "cublas":
            numeric["cublas_workspace_config"] = None
        elif mutation == "autocast":
            numeric["autocast"] = True
        elif mutation == "tf32":
            numeric["allow_tf32"] = True
        elif mutation == "precision":
            numeric["matmul_precision"] = "high"
        elif mutation == "deterministic":
            numeric["deterministic_algorithms"] = True
        elif mutation == "merge":
            q["merge"]["max_cost"] = 0.55
        elif mutation == "commit":
            q["commit"] = "0" * 40
        elif mutation == "dirty":
            q["dirty"] = True
        elif mutation == "coverage":
            q["sequences"].pop("MOT17-13-SDP")
        elif mutation == "row_chunk":
            row["row_chunk"] = 1
        elif mutation == "embedding":
            row["embedding_sha256"] = "0" * 64
        else:
            mode = "sparse" if mutation == "sparse_hash" else "sparse_blocked"
            row[mode]["out_sha256"] = "0" * 64
            assert row[mode]["vs_dense"]["out_identical"] is True
        return q

    monkeypatch.setattr(verifier, "load", mutated_load)
    monkeypatch.setattr(sys, "argv", ["verify", "--archive-only"])
    with pytest.raises(ValueError, match="drift|mismatch"):
        verifier.main()


@pytest.mark.parametrize(
    "mutation", ["bytes", "resign", "delete", "add", "remove_inventory"]
)
def test_archive_integrity_cannot_be_self_resigned(verifier, tmp_path, mutation):
    copy = tmp_path / "packet"
    shutil.copytree(verifier.PACKET, copy, ignore=shutil.ignore_patterns("__pycache__"))
    target = copy / "qualification/mot17.json"
    if mutation == "delete":
        target.unlink()
    elif mutation == "add":
        (copy / "unsealed.json").write_text("{}")
    elif mutation != "remove_inventory":
        target.write_bytes(target.read_bytes() + b"\n")
    if mutation in ("resign", "remove_inventory"):
        inventory = json.loads((copy / "SHA256SUMS.json").read_text())
        if mutation == "resign":
            row = next(
                r for r in inventory["files"] if r["file"] == "qualification/mot17.json"
            )
            row.update(
                bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        else:
            inventory["files"] = [
                r for r in inventory["files"] if r["file"] != "qualification/mot17.json"
            ]
        (copy / "SHA256SUMS.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="drift|unsealed/missing"):
        verifier.verify_inventory(copy, verifier.PACKET_REL, verifier.MAINTAINED_FILES)


def test_current_source_drift_is_not_qualified_by_old_capture(verifier, monkeypatch):
    original_sha = verifier.sha
    source = verifier.REPO / "src/saccade/perception/eval/cheb_gr_merge.py"
    monkeypatch.setattr(
        verifier, "sha", lambda p: "0" * 64 if p == source else original_sha(p)
    )
    monkeypatch.setattr(sys, "argv", ["verify"])
    with pytest.raises(
        ValueError, match="Current source drift requires full reference requalification"
    ):
        verifier.main()


def test_missing_trust_anchor_fails_closed(verifier, monkeypatch):
    monkeypatch.setattr(verifier, "SEALED_REF", "0" * 40)
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        verifier.verify_inventory(
            verifier.PACKET, verifier.PACKET_REL, verifier.MAINTAINED_FILES
        )
