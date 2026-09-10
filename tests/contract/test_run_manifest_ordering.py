"""Contract for the ADR 021 run-manifest: schema, ordering, and non-reattribution.

Two rules are under test, and they fail in opposite directions.

**Ordering** — the manifest lands *before* the first result byte. A test that
only asserted "the finished directory contains a manifest" would pass on a
producer that writes results first and the manifest last, which is precisely
the producer that leaves anonymous directories behind when it crashes. So the
ordering tests inject a manifest failure and assert *nothing else* was created.

**Non-reattribution** — a run may only claim an empty or new directory. None of
the wired producers clears its output directory first, so a manifest written
over an existing one would come to stand over whichever old files the new run
never overwrote. That is worse than no provenance: absent provenance announces
itself, while wrong provenance looks exactly like the right answer.
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
    PARENT_RUN_CLAIM_ENV,
    PARENT_RUN_ROOT_ENV,
    SCHEMA_VERSION,
    ManifestError,
    build_manifest,
    claim_or_join_run,
    join_parent_run,
    open_run,
    parent_claim_environ,
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


def test_schema_version_carried_the_v2_evolution(tmp_path):
    """Without a version, adding a field later would break every older reader.

    Unknown fields are fail-closed, so schema evolution is only possible if a
    reader can tell which schema it is looking at. v2 (ADR 021 AP-4) is that
    evolution actually happening: it added ``provenance_mode``, and a reader
    can refuse a v1 file rather than silently assume which mode it meant.
    """
    open_run(tmp_path / "r", produced_by="eval")
    assert read_manifest(tmp_path / "r")["schema_version"] == 2


def test_unknown_field_is_fail_closed():
    payload = build_manifest("r", produced_by="eval")
    payload["verdict"] = "ACCEPTED"  # a verdict never belongs in a manifest
    with pytest.raises(ManifestError, match="unknown manifest field"):
        validate_manifest(payload)


@pytest.mark.parametrize(
    "field",
    ["schema_version", "run_id", "provenance_mode", "commit", "dirty", "cmdline"],
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

    monkeypatch.setattr("scripts.provenance.run_manifest.os.link", boom)
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


# ---------------------------------------------------------------------------
# non-reattribution: a new manifest must never come to stand over old bytes
# ---------------------------------------------------------------------------


def test_an_empty_directory_is_claimable(tmp_path):
    """The boundary: empty is fine, occupied is not."""
    out = tmp_path / "prepared"
    out.mkdir(parents=True)
    open_run(out, produced_by="eval")
    assert read_manifest(out)["run_id"] == "prepared"


def test_a_directory_holding_artifacts_is_refused_and_left_byte_identical(tmp_path):
    out = tmp_path / "run_a"
    open_run(out, produced_by="eval", preset="run_a_preset")
    (out / "MOT17-02-SDP.txt").write_text(
        "1,1,10,10,20,40,1,-1,-1,-1\n", encoding="utf-8"
    )
    (out / "_fps_summary.txt").write_text("OVERALL\tfps=269.5\n", encoding="utf-8")

    before = {
        path.name: path.read_bytes() for path in sorted(out.iterdir()) if path.is_file()
    }

    with pytest.raises(ManifestError, match="is not empty"):
        open_run(out, produced_by="eval", preset="run_b_preset")

    after = {
        path.name: path.read_bytes() for path in sorted(out.iterdir()) if path.is_file()
    }
    assert after == before, "a refused claim must not touch a single existing byte"
    assert read_manifest(out)["preset"] == "run_a_preset", (
        "run A's manifest must still describe run A"
    )


def test_a_directory_holding_only_a_manifest_is_still_refused(tmp_path):
    """Re-claiming is refused even when the previous run produced nothing else.

    Otherwise the cheap case teaches the habit, and the habit is applied to the
    expensive one.
    """
    out = tmp_path / "claimed"
    open_run(out, produced_by="train")
    first = (out / MANIFEST_FILENAME).read_bytes()

    with pytest.raises(ManifestError, match="is not empty"):
        open_run(out, produced_by="train")

    assert (out / MANIFEST_FILENAME).read_bytes() == first


def test_v1_offers_no_overwrite_escape_hatch():
    """An ``overwrite`` flag would answer a design question nobody has settled.

    Run-continuation semantics — what a resumed run inherits, and what it may
    claim about bytes it did not produce — is deferred, so v1 must not ship a
    keyword that quietly decides it.
    """
    import inspect

    parameters = inspect.signature(open_run).parameters
    assert "overwrite" not in parameters
    assert "resume" not in parameters
    assert "force" not in parameters


def test_batch_eval_refuses_an_output_root_that_already_holds_artifacts(
    tmp_path, monkeypatch
):
    """The same rule on the canonical entry point: no dispatch, no mutation."""
    entry = _load_entry("mot17_all_sdp_reclaim", "scripts/eval/mot17_all_sdp.py")
    out = tmp_path / "results_existing"
    out.mkdir(parents=True)
    stale = out / "MOT17-02-SDP.txt"
    stale.write_text("1,7,10,10,20,40,1,-1,-1,-1\n", encoding="utf-8")
    stale_bytes = stale.read_bytes()

    dispatched: list[str] = []

    def spy(**kwargs):
        # A well-formed result, so that a regression fails on the assertion
        # below rather than on a stub that could not stand in for the real one.
        dispatched.append(kwargs["seq"])
        return {"sequence": kwargs["seq"], "returncode": 0, "log_path": ""}

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

    with pytest.raises(ManifestError, match="is not empty"):
        entry.main()

    assert dispatched == [], "dispatch must not start against an occupied directory"
    assert not (out / "_per_seq").exists()
    assert not (out / "_dispatch_plan.json").exists()
    assert not (out / MANIFEST_FILENAME).exists(), (
        "the refused run must not leave its manifest over the old artifacts"
    )
    assert stale.read_bytes() == stale_bytes
    assert sorted(item.name for item in out.iterdir()) == ["MOT17-02-SDP.txt"]


# ---------------------------------------------------------------------------
# parent-owned workers: joining is not "the directory already has a manifest"
# ---------------------------------------------------------------------------


def test_an_existing_manifest_is_not_a_worker_permit(tmp_path, monkeypatch):
    """A leftover identity must not let a new invocation skip open_run."""
    out = tmp_path / "run"
    open_run(out, produced_by="eval", preset="run_a")
    (out / "MOT17-02-SDP.txt").write_text("old\n", encoding="utf-8")
    before = (out / "MOT17-02-SDP.txt").read_bytes()
    monkeypatch.delenv(PARENT_RUN_ROOT_ENV, raising=False)
    monkeypatch.delenv(PARENT_RUN_CLAIM_ENV, raising=False)

    with pytest.raises(ManifestError, match="is not empty"):
        claim_or_join_run(out, produced_by="eval", preset="run_b")
    with pytest.raises(ManifestError, match="not itself a worker permit"):
        join_parent_run(out)

    assert (out / "MOT17-02-SDP.txt").read_bytes() == before
    assert read_manifest(out)["preset"] == "run_a"


def test_a_worker_joins_the_parent_root_without_publishing_a_second_identity(
    tmp_path, monkeypatch
):
    out = tmp_path / "run"
    open_run(out, produced_by="eval", preset="parent")
    for key, value in parent_claim_environ(out).items():
        monkeypatch.setenv(key, value)

    path = claim_or_join_run(out, produced_by="eval", preset="child-must-not-write")
    assert path == out / MANIFEST_FILENAME
    assert read_manifest(out)["preset"] == "parent"
    assert list(out.iterdir()) == [out / MANIFEST_FILENAME]


def test_a_nested_worker_does_not_declare_a_new_run(tmp_path, monkeypatch):
    root = tmp_path / "run"
    open_run(root, produced_by="eval")
    nested = root / "_per_seq" / "MOT17-02-SDP"
    nested.mkdir(parents=True)
    for key, value in parent_claim_environ(root).items():
        monkeypatch.setenv(key, value)

    claim_or_join_run(nested, produced_by="eval")
    assert not (nested / MANIFEST_FILENAME).exists()
    assert (root / MANIFEST_FILENAME).exists()


def test_a_worker_cannot_join_a_different_root(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    other = tmp_path / "other"
    open_run(parent, produced_by="eval")
    open_run(other, produced_by="eval")
    for key, value in parent_claim_environ(parent).items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ManifestError, match="not the claimed run root"):
        join_parent_run(other)


def test_a_wrong_claim_token_is_refused(tmp_path, monkeypatch):
    out = tmp_path / "run"
    open_run(out, produced_by="eval")
    env = parent_claim_environ(out)
    monkeypatch.setenv(PARENT_RUN_ROOT_ENV, env[PARENT_RUN_ROOT_ENV])
    monkeypatch.setenv(PARENT_RUN_CLAIM_ENV, "forged|token")

    with pytest.raises(ManifestError, match="token does not match"):
        join_parent_run(out)


def test_a_partial_parent_claim_env_does_not_fall_through_to_open_run(
    tmp_path, monkeypatch
):
    """A nested empty dir plus a half-set env must not become a second identity."""
    root = tmp_path / "run"
    open_run(root, produced_by="eval")
    nested = root / "_per_seq" / "MOT17-02-SDP"
    nested.mkdir(parents=True)
    monkeypatch.setenv(PARENT_RUN_ROOT_ENV, str(root.resolve()))
    monkeypatch.delenv(PARENT_RUN_CLAIM_ENV, raising=False)

    with pytest.raises(ManifestError, match="incomplete parent-run claim"):
        claim_or_join_run(nested, produced_by="eval")
    assert not (nested / MANIFEST_FILENAME).exists()


def test_batch_eval_workers_receive_a_parent_claim_and_do_not_get_nested_manifests(
    tmp_path, monkeypatch
):
    """mot17_all_sdp.py owns the run; _per_seq/<seq> is a worker slot, not a run."""
    entry = _load_entry("mot17_all_sdp_parent_claim", "scripts/eval/mot17_all_sdp.py")
    out = tmp_path / "results_parent"
    captured_env: list[dict[str, str]] = []

    def fake_run_sequence(*, seq, cmd, output_dir, env, dry_run):
        captured_env.append(dict(env))
        return {
            "sequence": seq,
            "returncode": 0,
            "log_path": str(output_dir / f"{seq}.log"),
            "cmd": cmd,
            "dry_run": dry_run,
            "wall_sec": 0.0,
        }

    monkeypatch.setattr(entry, "_run_sequence", fake_run_sequence)
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

    assert captured_env, "dispatch never ran"
    env = captured_env[0]
    assert env[PARENT_RUN_ROOT_ENV] == str(out.resolve())
    assert env[PARENT_RUN_CLAIM_ENV] == (
        f"{read_manifest(out)['run_id']}|{read_manifest(out)['started_at']}"
    )
    nested = out / "_per_seq" / "MOT17-02-SDP"
    assert nested.is_dir()
    assert not (nested / MANIFEST_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
