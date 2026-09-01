"""Contract for the ADR 021 run-manifest: schema, and the write-ordering rule.

The rule under test is that the manifest lands **before** the first result
byte. A test that only asserted "the finished directory contains a manifest"
would pass on a producer that writes results first and the manifest last —
which is precisely the producer that leaves anonymous directories behind when
it crashes. So the ordering tests inject a manifest failure and assert that
*nothing else* was created.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.provenance.run_manifest import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    ManifestError,
    build_manifest,
    open_run,
    read_manifest,
    require_manifest,
    validate_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def _load_entry(name: str, relative: str):
    """Import a script entry point by path, the way the eval tree is laid out."""
    for extra in (str(REPO), str(REPO / "src"), str(REPO / "scripts" / "eval")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_open_run_writes_a_manifest_that_reads_back_valid(tmp_path):
    out = tmp_path / "results" / "some_run"
    path = open_run(out, produced_by="eval", preset="p", detector="SDP", dataset="d")

    assert path == out / MANIFEST_FILENAME
    payload = read_manifest(out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_id"] == "some_run"
    assert payload["produced_by"] == "eval"
    assert payload["preset"] == "p"


def test_schema_version_is_present_from_v1(tmp_path):
    """Without a version, adding a field later would break every older reader.

    Unknown fields are fail-closed, so schema evolution is only possible if a
    reader can tell which schema it is looking at.
    """
    open_run(tmp_path / "r", produced_by="eval")
    assert read_manifest(tmp_path / "r")["schema_version"] == 1


def test_unknown_field_is_fail_closed():
    payload = build_manifest("r", produced_by="eval")
    payload["verdict"] = "ACCEPTED"  # a verdict never belongs in a manifest
    with pytest.raises(ManifestError, match="unknown manifest field"):
        validate_manifest(payload)


@pytest.mark.parametrize(
    "field", ["schema_version", "run_id", "commit", "dirty", "cmdline"]
)
def test_missing_required_field_is_fail_closed(field):
    payload = build_manifest("r", produced_by="eval")
    payload.pop(field)
    with pytest.raises(ManifestError, match="missing required manifest field"):
        validate_manifest(payload)


def test_foreign_schema_version_is_rejected():
    payload = build_manifest("r", produced_by="eval")
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ManifestError, match="unsupported manifest schema_version"):
        validate_manifest(payload)


def test_produced_by_is_a_closed_vocabulary():
    payload = build_manifest("r", produced_by="eval")
    payload["produced_by"] = "evaluation"
    with pytest.raises(ManifestError, match="produced_by must be one of"):
        validate_manifest(payload)


def test_claims_hold_ids_not_verdicts():
    payload = build_manifest(
        "r", produced_by="eval", claims=["gate.safe_region.dist_h"]
    )
    validate_manifest(payload)
    broken = deepcopy(payload)
    broken["claims"] = [{"object": "x", "state": "L1"}]
    with pytest.raises(ManifestError, match="claims must be a list"):
        validate_manifest(broken)


def test_missing_git_is_recorded_as_null_not_guessed(monkeypatch):
    """An unavailable git yields an explicit unknown, never an invented commit."""
    monkeypatch.setattr("scripts.provenance.run_manifest._git", lambda *a: None)
    payload = build_manifest("r", produced_by="train")
    validate_manifest(payload)
    assert payload["commit"] is None
    assert payload["dirty"] is None


def test_require_manifest_rejects_an_unclaimed_directory(tmp_path):
    (tmp_path / "anon").mkdir()
    with pytest.raises(ManifestError, match="carries no run_manifest.json"):
        require_manifest(tmp_path / "anon")


def test_corrupt_manifest_is_not_silently_accepted(tmp_path):
    out = tmp_path / "r"
    open_run(out, produced_by="eval")
    (out / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        require_manifest(out)


# ---------------------------------------------------------------------------
# ordering: a manifest failure must leave nothing behind
# ---------------------------------------------------------------------------


def test_failed_manifest_write_leaves_no_partial_manifest(tmp_path, monkeypatch):
    out = tmp_path / "r"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("scripts.provenance.run_manifest.os.replace", boom)
    with pytest.raises(ManifestError, match="cannot write manifest"):
        open_run(out, produced_by="eval")

    assert not (out / MANIFEST_FILENAME).exists()
    assert list(out.iterdir()) == [], "a failed claim must not leave temp files behind"


def test_manifest_failure_stops_the_batch_eval_before_any_result_is_produced(
    tmp_path, monkeypatch
):
    """The ordering contract on the canonical 7-seq entry point.

    ``--dry-run`` still writes ``_per_seq/`` and ``_dispatch_plan.json``, so it
    is enough to prove the claim happens first: if the manifest raises and the
    output directory is still empty, no result writer had started.
    """
    entry = _load_entry("mot17_all_sdp_contract", "scripts/eval/mot17_all_sdp.py")
    out = tmp_path / "results_run"

    def refuse(*args, **kwargs):
        raise ManifestError("injected: manifest unavailable")

    monkeypatch.setattr(entry, "open_run", refuse)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mot17_all_sdp.py",
            "--output",
            str(out),
            "--sequences",
            "MOT17-02-SDP",
            "--dry-run",
        ],
    )

    with pytest.raises(ManifestError, match="injected"):
        entry.main()

    assert not (out / "_per_seq").exists()
    assert not (out / "_dispatch_plan.json").exists()
    assert list(out.glob("*.txt")) == []


def test_batch_eval_claims_the_directory_before_dispatching(tmp_path, monkeypatch):
    """The same entry point, succeeding: the manifest exists by dispatch time."""
    entry = _load_entry("mot17_all_sdp_contract_ok", "scripts/eval/mot17_all_sdp.py")
    out = tmp_path / "results_ok"
    seen: list[bool] = []

    real_run_sequence = entry._run_sequence

    def spy(**kwargs):
        seen.append((out / MANIFEST_FILENAME).exists())
        return real_run_sequence(**kwargs)

    monkeypatch.setattr(entry, "_run_sequence", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mot17_all_sdp.py",
            "--output",
            str(out),
            "--sequences",
            "MOT17-02-SDP",
            "--dry-run",
        ],
    )

    entry.main()

    assert seen == [True], "dispatch started before the directory was claimed"
    assert read_manifest(out)["produced_by"] == "eval"
