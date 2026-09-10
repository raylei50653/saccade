"""Contract for ADR 021 AP-5: disposal candidates, not deletion verdicts.

The generator is a projection. Tests lock the predicate, not the wording of
the markdown it happens to emit. Each fail-closed condition is asserted by
its *reason*, so dropping that condition from the implementation reddens
the matching test rather than being absorbed by a neighbouring block.

**Candidate ≠ deletion.** The output is a review list. It does not mean
approved, safe to delete, or disposable, and the tool has no deletion
action.

**Age cannot substitute for provenance.** A cited or manifested unit that
is old is still not a candidate. An orphan that is young is still not a
candidate. Both halves have to hold.

**The generated view is not an authority.** It is not an input to the next
run, it is not a citation source, and it does not store approval.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.provenance.asset_disposal import (
    BLOCK_AGE,
    BLOCK_CITED,
    BLOCK_CONTAINER,
    BLOCK_INVALID,
    BLOCK_MANIFESTED,
    BLOCK_MTIME_FUTURE,
    BLOCK_NOT_ORPHAN,
    BLOCK_SELF_ATTESTING,
    BLOCK_UNKNOWN_STATE,
    BLOCK_UNREADABLE,
    BLOCK_UNSAFE_PATH,
    MIN_AGE_DAYS,
    Candidate,
    DisposalError,
    _build_parser,
    derive_candidates,
    evaluate_unit,
    main,
    parse_as_of,
    render,
)
from scripts.provenance.asset_inventory import Unit, scan
from scripts.provenance.run_manifest import MANIFEST_FILENAME, build_manifest

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("# docs\n", encoding="utf-8")
    _commit(tmp_path)
    return tmp_path


def _commit(repo: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "t", "--allow-empty"], cwd=repo, check=True
    )


def _unit(repo: Path, root: str, name: str) -> Path:
    directory = repo / root / name
    directory.mkdir(parents=True)
    return directory


def _age(directory: Path, days: int) -> None:
    ts = (NOW - timedelta(days=days)).timestamp()
    os.utime(directory, (ts, ts))


def _manifest(directory: Path) -> None:
    payload = build_manifest(directory.name, produced_by="eval")
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _doc(repo: Path, name: str, text: str) -> None:
    (repo / "docs" / name).write_text(text, encoding="utf-8")


def _old_orphan(repo: Path, name: str = "old_run", days: int = MIN_AGE_DAYS) -> Path:
    directory = _unit(repo, "results", name)
    _age(directory, days)
    return directory


def _eval(repo: Path, path: str):
    units = {unit.path: unit for unit in scan(repo)}
    return evaluate_unit(units[path], repo, NOW)


def _candidates(repo: Path, now: datetime = NOW) -> tuple[str, ...]:
    return tuple(c.path for c in derive_candidates(scan(repo), repo, now))


def _table_rows(text: str) -> list[str]:
    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Unit |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue
            rows.append(line.lower())
    return rows


# ---------------------------------------------------------------------------
# the candidate is a review item, not a verdict
# ---------------------------------------------------------------------------


def test_the_age_threshold_is_the_documented_policy():
    assert MIN_AGE_DAYS == 90


def test_a_candidate_carries_no_approval_or_deletion_field():
    assert {field.name for field in fields(Candidate)} == {
        "path",
        "root",
        "age_days",
    }


def test_the_cli_exposes_no_deletion_action():
    dests = {action.dest for action in _build_parser()._actions}
    option_text = " ".join(
        flag for action in _build_parser()._actions for flag in action.option_strings
    ).lower()
    for word in ("delete", "rm", "trash", "dispose", "clean", "remove"):
        assert word not in dests
        assert f"--{word}" not in option_text


def test_emitting_does_not_delete_the_unit(tmp_path):
    repo = _repo(tmp_path)
    directory = _old_orphan(repo)
    assert main(["--repo-root", str(repo), "--emit", "--as-of", NOW.isoformat()]) == 0
    assert directory.is_dir()
    assert list(directory.iterdir()) == []


def test_check_does_not_delete_the_unit(tmp_path):
    repo = _repo(tmp_path)
    directory = _old_orphan(repo)
    assert main(["--repo-root", str(repo), "--check", "--as-of", NOW.isoformat()]) == 0
    assert directory.is_dir()


def test_approval_absence_is_not_approval(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo)
    text = render(
        derive_candidates(scan(repo), repo, NOW),
        repo_root=repo,
        now=NOW,
        unit_count=1,
    ).lower()
    assert "absence of that action is not approval" in text
    assert "not a deletion verdict" in text
    assert "this tool deletes nothing" in text
    for row in _table_rows(text):
        for word in (
            "approved",
            "safe to delete",
            "disposable",
            "auto-delete",
            "deletable",
            "eligible",
        ):
            assert word not in row, row


# ---------------------------------------------------------------------------
# fail-closed: invalid, cited, manifested, unknown, unreadable
# ---------------------------------------------------------------------------


def test_an_invalid_manifest_is_never_a_candidate_even_when_old(tmp_path):
    """Mutation: dropping BLOCK_INVALID from the predicate reddens this."""
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "broken")
    (run / MANIFEST_FILENAME).write_text('{"schema_version": 1}', encoding="utf-8")
    _age(run, MIN_AGE_DAYS + 30)

    evaluation = _eval(repo, "results/broken")
    assert not evaluation.candidate
    assert BLOCK_INVALID in evaluation.blocked_by
    assert BLOCK_NOT_ORPHAN in evaluation.blocked_by
    assert "results/broken" not in _candidates(repo)


def test_unparseable_manifest_bytes_are_invalid_not_candidates(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "broken")
    (run / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    _age(run, MIN_AGE_DAYS + 30)

    evaluation = _eval(repo, "results/broken")
    assert not evaluation.candidate
    assert BLOCK_INVALID in evaluation.blocked_by


def test_a_cited_unit_is_never_a_candidate_because_it_is_old(tmp_path):
    """Mutation: dropping BLOCK_CITED reddens the reason assertion."""
    repo = _repo(tmp_path)
    _old_orphan(repo, "cited_run", days=MIN_AGE_DAYS + 30)
    _doc(repo, "note.md", "measured in `results/cited_run`\n")
    _commit(repo)

    evaluation = _eval(repo, "results/cited_run")
    assert not evaluation.candidate
    assert BLOCK_CITED in evaluation.blocked_by
    assert "results/cited_run" not in _candidates(repo)


def test_a_manifested_unit_is_never_a_candidate_because_it_is_old(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "manifested_run")
    _manifest(run)
    _age(run, MIN_AGE_DAYS + 30)

    evaluation = _eval(repo, "results/manifested_run")
    assert not evaluation.candidate
    assert BLOCK_MANIFESTED in evaluation.blocked_by
    assert BLOCK_NOT_ORPHAN in evaluation.blocked_by


def test_unknown_manifest_state_fails_closed(tmp_path):
    """Mutation: accepting an unknown state as absent reddens this."""
    repo = _repo(tmp_path)
    directory = _old_orphan(repo, "mystery")
    unit = Unit(
        path="results/mystery",
        root="results",
        manifest_state="maybe",
        cited=False,
    )
    evaluation = evaluate_unit(unit, repo, NOW)
    assert directory.is_dir()
    assert not evaluation.candidate
    assert BLOCK_UNKNOWN_STATE in evaluation.blocked_by


def test_a_missing_directory_fails_closed_rather_than_being_guessed(tmp_path):
    repo = _repo(tmp_path)
    unit = Unit(
        path="results/gone",
        root="results",
        manifest_state="absent",
        cited=False,
    )
    evaluation = evaluate_unit(unit, repo, NOW)
    assert not evaluation.candidate
    assert BLOCK_UNREADABLE in evaluation.blocked_by


def test_a_path_that_walks_out_of_the_unit_fails_closed(tmp_path):
    repo = _repo(tmp_path)
    unit = Unit(
        path="results/../secret",
        root="results",
        manifest_state="absent",
        cited=False,
    )
    evaluation = evaluate_unit(unit, repo, NOW)
    assert not evaluation.candidate
    assert BLOCK_UNSAFE_PATH in evaluation.blocked_by


def test_naive_clock_is_refused_rather_than_silently_shifted(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo)
    unit = scan(repo)[0]
    with pytest.raises(DisposalError, match="timezone-aware"):
        evaluate_unit(unit, repo, datetime(2026, 9, 10, 12, 0))


# ---------------------------------------------------------------------------
# age is a filter, not a provenance substitute
# ---------------------------------------------------------------------------


def test_an_orphan_younger_than_the_threshold_is_not_a_candidate(tmp_path):
    """Mutation: dropping BLOCK_AGE reddens this — the unit would qualify."""
    repo = _repo(tmp_path)
    _old_orphan(repo, "young_run", days=MIN_AGE_DAYS - 1)

    evaluation = _eval(repo, "results/young_run")
    assert evaluation.candidate is False
    assert BLOCK_AGE in evaluation.blocked_by
    assert "results/young_run" not in _candidates(repo)


def test_an_orphan_at_the_threshold_is_a_candidate(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo, "aged_run", days=MIN_AGE_DAYS)
    assert _candidates(repo) == ("results/aged_run",)
    evaluation = _eval(repo, "results/aged_run")
    assert evaluation.candidate
    assert evaluation.blocked_by == ()
    assert evaluation.age_days == MIN_AGE_DAYS


def test_future_mtime_fails_closed(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "from_the_future")
    future = (NOW + timedelta(days=3)).timestamp()
    os.utime(run, (future, future))

    evaluation = _eval(repo, "results/from_the_future")
    assert not evaluation.candidate
    assert BLOCK_MTIME_FUTURE in evaluation.blocked_by


def test_as_of_shifts_the_clock_not_the_policy(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo, "young_run", days=10)
    assert _candidates(repo, NOW) == ()
    later = NOW + timedelta(days=MIN_AGE_DAYS)
    assert _candidates(repo, later) == ("results/young_run",)


# ---------------------------------------------------------------------------
# self-attesting / sealed records, and unit granularity
# ---------------------------------------------------------------------------


def test_a_self_attesting_record_is_not_a_candidate(tmp_path):
    """Mutation: ignoring authority-declaring JSON reddens this."""
    repo = _repo(tmp_path)
    run = _old_orphan(repo, "sealed_identity")
    (run / "layer_p.json").write_text(
        json.dumps({"authority": "non_authoritative_pre_seal_engineering"}),
        encoding="utf-8",
    )
    _age(run, MIN_AGE_DAYS)

    evaluation = _eval(repo, "results/sealed_identity")
    assert not evaluation.candidate
    assert BLOCK_SELF_ATTESTING in evaluation.blocked_by


def test_a_covering_seal_is_not_a_candidate(tmp_path):
    repo = _repo(tmp_path)
    run = _old_orphan(repo, "sealed_bytes")
    (run / "output.txt").write_text("x", encoding="utf-8")
    (run / "SHA256SUMS").write_text("abc  output.txt\n", encoding="utf-8")
    _age(run, MIN_AGE_DAYS)

    evaluation = _eval(repo, "results/sealed_bytes")
    assert not evaluation.candidate
    assert BLOCK_SELF_ATTESTING in evaluation.blocked_by


def test_a_nested_seal_that_does_not_cover_the_unit_does_not_block(tmp_path):
    """Same h2_execution shape AP-4 already locked: depth is not identity."""
    repo = _repo(tmp_path)
    run = _old_orphan(repo, "parent_with_archive")
    (run / "archive").mkdir()
    (run / "archive" / "checksums.sha256").write_text(
        "abc  result.json\n", encoding="utf-8"
    )
    _age(run, MIN_AGE_DAYS)
    assert _candidates(repo) == ("results/parent_with_archive",)


def test_an_ordinary_result_json_is_not_self_attesting(tmp_path):
    repo = _repo(tmp_path)
    run = _old_orphan(repo, "latency")
    (run / "_latency_profile.json").write_text(
        json.dumps({"mean_ms": 3.1}), encoding="utf-8"
    )
    _age(run, MIN_AGE_DAYS)
    assert _candidates(repo) == ("results/latency",)


def test_a_multi_run_container_is_not_a_candidate(tmp_path):
    """Mutation: raising the container threshold from 2 to 3 reddens this."""
    repo = _repo(tmp_path)
    parent = _old_orphan(repo, "ablation")
    for arm in ("A_7seq", "B_7seq"):
        child = parent / arm
        child.mkdir()
        (child / "run_meta.txt").write_text("git_sha=abc\n", encoding="utf-8")
    _age(parent, MIN_AGE_DAYS)

    evaluation = _eval(repo, "results/ablation")
    assert not evaluation.candidate
    assert BLOCK_CONTAINER in evaluation.blocked_by


def test_derivation_does_not_relabel_inventory_units(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo)
    cited = _unit(repo, "results", "cited_only")
    _doc(repo, "note.md", "`results/cited_only`\n")
    _commit(repo)
    _age(cited, MIN_AGE_DAYS)

    before = scan(repo)
    derive_candidates(before, repo, NOW)
    after = scan(repo)
    assert before == after
    assert {
        unit.path: (unit.cited, unit.orphan, unit.manifest_state) for unit in after
    } == {
        "results/old_run": (False, True, "absent"),
        "results/cited_only": (True, False, "absent"),
    }


def test_nested_children_are_not_independent_units(tmp_path):
    repo = _repo(tmp_path)
    run = _old_orphan(repo, "parent_run")
    child = run / "_per_seq" / "MOT17-02-SDP"
    child.mkdir(parents=True)
    _age(child, MIN_AGE_DAYS)
    _age(run, MIN_AGE_DAYS)

    paths = _candidates(repo)
    assert paths == ("results/parent_run",)
    assert all("_per_seq" not in path for path in paths)
    assert [unit.path for unit in scan(repo)] == ["results/parent_run"]


def test_loose_files_are_not_units_and_not_candidates(tmp_path):
    repo = _repo(tmp_path)
    (repo / "results").mkdir()
    loose = repo / "results" / "sweep.log"
    loose.write_text("x", encoding="utf-8")
    _age(repo / "results", MIN_AGE_DAYS)
    assert scan(repo) == ()
    assert _candidates(repo) == ()


# ---------------------------------------------------------------------------
# generated view is not an authority and not a citation source
# ---------------------------------------------------------------------------


def test_emitting_into_docs_is_refused(tmp_path, capsys):
    repo = _repo(tmp_path)
    _old_orphan(repo)
    target = repo / "docs" / "asset_disposal_candidates.generated.md"

    code = main(
        [
            "--repo-root",
            str(repo),
            "--emit",
            "docs/asset_disposal_candidates.generated.md",
            "--as-of",
            NOW.isoformat(),
        ]
    )

    assert code == 2
    assert "refusing to emit" in capsys.readouterr().err
    assert not target.exists()
    assert (repo / "results" / "old_run").is_dir()


def test_emitting_outside_docs_is_allowed(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo)
    assert main(["--repo-root", str(repo), "--emit", "--as-of", NOW.isoformat()]) == 0
    rendered = (
        repo / ".provenance" / "asset_disposal_candidates.generated.md"
    ).read_text()
    assert "results/old_run" in rendered
    assert "not a deletion verdict" in rendered.lower()


def test_a_generated_view_naming_every_unit_cites_nothing(tmp_path):
    """Even a committed generated view must not certify its own subjects."""
    repo = _repo(tmp_path)
    _old_orphan(repo, "lonely")
    _doc(repo, "asset_disposal_candidates.generated.md", "| `results/lonely` |\n")
    _commit(repo)

    unit = {u.path: u for u in scan(repo)}["results/lonely"]
    assert not unit.cited
    assert unit.orphan
    assert _candidates(repo) == ("results/lonely",)


def test_a_stale_snapshot_is_not_an_input(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo, "cited_later")
    stale = repo / ".provenance" / "asset_disposal_candidates.generated.md"
    stale.parent.mkdir()
    stale.write_text(
        "| Unit | Root | Age (days) |\n| `results/cited_later` | results | 999 |\n",
        encoding="utf-8",
    )
    _doc(repo, "note.md", "`results/cited_later`\n")
    _commit(repo)

    assert main(["--repo-root", str(repo), "--emit", "--as-of", NOW.isoformat()]) == 0
    text = stale.read_text(encoding="utf-8")
    assert _candidates(repo) == ()
    for row in _table_rows(text):
        assert "cited_later" not in row


def test_workspace_change_regenerates_rather_than_reusing_the_snapshot(tmp_path):
    repo = _repo(tmp_path)
    _old_orphan(repo, "moving_target")
    assert _candidates(repo) == ("results/moving_target",)

    _doc(repo, "note.md", "`results/moving_target`\n")
    _commit(repo)
    assert _candidates(repo) == ()


def test_check_passes_on_a_workspace_with_no_assets(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert main(["--repo-root", str(repo), "--check", "--as-of", NOW.isoformat()]) == 0
    assert "0 candidate(s)" in capsys.readouterr().out


def test_check_fails_closed_on_an_invalid_manifest(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "broken")
    (run / MANIFEST_FILENAME).write_text('{"schema_version": 99}', encoding="utf-8")
    assert main(["--repo-root", str(repo), "--check", "--as-of", NOW.isoformat()]) == 1


def test_parse_as_of_treats_naive_values_as_utc():
    parsed = parse_as_of("2026-09-10T12:00:00")
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert parse_as_of("2026-09-10T12:00:00Z") == parsed
