"""Contract for the ADR 021 AP-2 producer registry: coverage cannot drift silently.

PR #330 wired ``open_run`` into a hand-picked set of entry points. The set was
correct for the files someone thought of, and said nothing at all about the
files nobody thought of — so an unwired producer and a deliberately excluded one
looked identical from the repository. The registry exists to make that
difference explicit, and these tests exist to make the registry load-bearing.

Four properties are under test, and each is verified by breaking it:

* a **new** entry point in the domain fails closed rather than being ignored;
* a **known** entry point cannot carry an unknown or missing classification;
* **deleting** one of the wired ``open_run`` calls is detected, so the checker
  measures the code rather than the registry's opinion of the code;
* ``run_producer_blocked`` cannot be used to excuse an unprotected file, which
  is the only way this vocabulary could become an escape hatch.

The last one matters most. Every other classification is checkable against the
source; "this produces artifacts but may not be wired" is a claim about
governance, and if the checker took it on trust, any inconvenient producer could
be retired into it.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.provenance.check_producer_coverage import (
    PROTECTED_PARTITION_CLASSES,
    REGISTRY_REL,
    CoverageError,
    calls_open_run,
    check,
    load_registry,
)

REPO = Path(__file__).resolve().parents[2]

# The producer whose coverage ADR 021 §4.3 names as blocked. It is asserted by
# name because "the remainder is empty" and "the remainder is unrecorded" must
# not be able to look the same to this suite.
BLOCKED_PRODUCER = "scripts/eval/mot17.py"

WIRED_IN_THIS_PR = (
    "scripts/eval/concurrent_mot17.py",
    "scripts/eval/baselines/mot17_public.py",
    "scripts/eval/baselines/ultralytics_official_mot17.py",
    "scripts/train/temporal_yolo/train_jde_market.py",
)


@pytest.fixture
def registry() -> dict:
    return load_registry(REPO / REGISTRY_REL)


def _mirror(tmp_path: Path, payload: dict) -> Path:
    """A repo-shaped tree holding only the registry, for negative cases.

    ``check`` reads the registry from the root it is given but resolves domain
    membership through ``git ls-files`` in that root, so mutations are exercised
    against the real repository by writing the registry alone into a worktree
    that shares its git dir.
    """
    root = tmp_path / "repo"
    (root / "scripts" / "provenance").mkdir(parents=True)
    (root / REGISTRY_REL).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# the repository as it stands
# ---------------------------------------------------------------------------


def test_repository_passes_the_coverage_check():
    assert check(REPO) == []


def test_every_domain_file_is_classified(registry):
    """No entry point may be absent, because absence is not a classification."""
    from scripts.provenance.check_producer_coverage import domain_files

    tracked = domain_files(registry, REPO)
    assert tracked, "domain resolved to nothing; the check would be vacuous"
    assert tracked == set(registry["entries"]), (
        "registry and domain disagree: "
        f"unlisted={sorted(tracked - set(registry['entries']))} "
        f"stale={sorted(set(registry['entries']) - tracked)}"
    )


def test_the_four_producers_wired_in_this_pr_are_wired(registry):
    for path in WIRED_IN_THIS_PR:
        assert registry["entries"][path]["classification"] == "run_producer_wired"
        assert calls_open_run((REPO / path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the named remainder
# ---------------------------------------------------------------------------


def test_mot17_is_recorded_as_blocked_not_silently_excluded(registry):
    """§4.3's remainder is a registry row, not a gap in one.

    W-A exit criterion 1 is not satisfied while this row exists. The row is what
    makes that statement checkable instead of a claim in prose.
    """
    entry = registry["entries"][BLOCKED_PRODUCER]
    assert entry["classification"] == "run_producer_blocked"
    assert "021" in entry["blocked_by"]
    assert entry["unblock_requires"]
    assert not calls_open_run((REPO / BLOCKED_PRODUCER).read_text(encoding="utf-8"))


def test_blocked_producer_really_is_in_a_protected_partition():
    """The block is a property of the repository, not of the registry's say-so."""
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "tools"))
    import h2_path_partition as partition

    assert partition.classify(BLOCKED_PRODUCER) in PROTECTED_PARTITION_CLASSES


def test_blocked_is_not_an_escape_hatch_for_unprotected_files(tmp_path, registry):
    """Relabelling a wired producer as blocked must fail, not excuse it."""
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "tools"))
    import h2_path_partition as partition

    victim = "scripts/eval/concurrent_mot17.py"
    assert partition.classify(victim) not in PROTECTED_PARTITION_CLASSES

    payload = deepcopy(registry)
    payload["entries"][victim] = {
        "classification": "run_producer_blocked",
        "reason": "claimed blocked",
        "blocked_by": "docs/decisions/021-asset-provenance-and-progress-reporting.md",
        "unblock_requires": "nothing, which is the point",
    }
    failures = check(REPO, payload)
    assert any("not protected" in f and victim in f for f in failures)


# ---------------------------------------------------------------------------
# mutations: each rule is verified by breaking it
# ---------------------------------------------------------------------------


def test_an_unlisted_entry_point_fails_closed(tmp_path, registry):
    """A newly added eval/train file that nobody classified must fail CI."""
    payload = deepcopy(registry)
    dropped = payload["entries"].pop("scripts/eval/mot17_all_sdp.py")
    assert dropped
    failures = check(REPO, payload)
    assert any("absent from" in f for f in failures)


def test_a_stale_entry_fails_closed(tmp_path, registry):
    payload = deepcopy(registry)
    payload["entries"]["scripts/eval/deleted_yesterday.py"] = {
        "classification": "not_a_run_producer",
        "reason": "gone",
    }
    failures = check(REPO, payload)
    assert any("not a tracked file" in f for f in failures)


@pytest.mark.parametrize("path", WIRED_IN_THIS_PR)
def test_removing_an_open_run_integration_is_detected(path):
    """Delete the call, keep the import: the checker must still notice.

    This is the mutation that a presence-only test would miss, and the reason
    ``calls_open_run`` parses instead of grepping.
    """
    source = (REPO / path).read_text(encoding="utf-8")
    assert calls_open_run(source)
    without_call = source.replace("open_run(", "_disabled_open_run(")
    assert not calls_open_run(without_call)


def test_unknown_classification_fails_closed(registry):
    payload = deepcopy(registry)
    payload["entries"]["scripts/eval/mot17_all_sdp.py"] = {
        "classification": "probably_fine",
        "reason": "nope",
    }
    failures = check(REPO, payload)
    assert any("unknown classification" in f for f in failures)


def test_a_wired_classification_without_the_call_fails_closed(registry):
    """The registry cannot assert coverage the source does not have."""
    payload = deepcopy(registry)
    payload["entries"]["scripts/eval/calculate_mota.py"] = {
        "classification": "run_producer_wired",
        "reason": "claimed wired",
    }
    failures = check(REPO, payload)
    assert any("calls no open_run" in f for f in failures)


def test_registry_rejects_unknown_fields(tmp_path, registry):
    """Unknown-field fail-closed, matching the manifest and terminal-slot schemas."""
    payload = deepcopy(registry)
    payload["notes"] = "a new semantic field nobody agreed to"
    root = _mirror(tmp_path, payload)
    with pytest.raises(CoverageError, match="unknown top-level field"):
        load_registry(root / REGISTRY_REL)


def test_out_of_scope_rows_must_cite_an_authority(registry):
    """Excluded is a decision with a source, not a shrug."""
    excluded = [
        (path, entry)
        for path, entry in registry["entries"].items()
        if entry["classification"] == "run_producer_out_of_scope"
    ]
    assert excluded, "no out-of-scope rows; this test would be vacuous"
    for path, entry in excluded:
        assert "021" in entry["excluded_by"], path
        assert entry["reason"], path
