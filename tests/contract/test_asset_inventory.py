"""Contract for the ADR 021 asset inventory: a projection that must not lie.

Three properties are load-bearing, and each fails silently if untested.

**It is a projection, not a committed authority.** The asset roots are
gitignored, so a clean clone holds none of them. An empty inventory there is a
correct answer, not drift — which is why the CI hook validates rather than
comparing against a committed file.

**An invalid manifest is not an orphan.** Orphan means nothing accounts for
this directory. A manifest that exists and does not validate means something
tried to and cannot be read: a broken producer. Folding the second into the
first would hide it inside a routine backlog number.

**Generated views must not feed the citation corpus.** Render every unit name
into a document once and, without the exclusion, the next scan reports the
whole workspace as cited — the tool would certify its own output.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

from scripts.provenance.asset_inventory import (
    ASSET_ROOTS,
    InventoryError,
    Unit,
    invalid_manifests,
    loose_entries,
    main,
    render,
    scan,
)
from scripts.provenance.run_manifest import MANIFEST_FILENAME, build_manifest


def _repo(tmp_path: Path) -> Path:
    """A minimal tracked repo: the citation corpus is defined as tracked files."""
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


def _manifest(directory: Path, **kwargs) -> None:
    payload = build_manifest(directory.name, produced_by="eval", **kwargs)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _doc(repo: Path, name: str, text: str) -> None:
    (repo / "docs" / name).write_text(text, encoding="utf-8")


def _by_path(units: tuple[Unit, ...]) -> dict[str, Unit]:
    return {unit.path: unit for unit in units}


# ---------------------------------------------------------------------------
# the unit is the immediate child, and nothing deeper
# ---------------------------------------------------------------------------


def test_a_unit_is_an_immediate_child_of_an_asset_root(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "run_a")
    (run / "_per_seq" / "MOT17-02-SDP").mkdir(parents=True)
    (run / "checkpoints").mkdir()

    units = scan(repo)

    assert [unit.path for unit in units] == ["results/run_a"], (
        "a run's own sub-directories are part of that run, not separate orphans"
    )


def test_every_asset_root_is_scanned_and_a_missing_root_is_not_an_error(tmp_path):
    repo = _repo(tmp_path)
    for root in ASSET_ROOTS:
        _unit(repo, root, f"{root}_unit")
    units = scan(repo)
    assert sorted(unit.root for unit in units) == sorted(ASSET_ROOTS)

    (repo / "runs" / "runs_unit").rmdir()
    (repo / "runs").rmdir()
    assert len(scan(repo)) == len(ASSET_ROOTS) - 1


def test_loose_files_are_counted_out_of_scope_rather_than_dropped(tmp_path):
    repo = _repo(tmp_path)
    (repo / "runs").mkdir()
    (repo / "runs" / "sweep.log").write_text("x", encoding="utf-8")

    assert scan(repo) == ()
    assert loose_entries(repo) == ("runs/sweep.log",)


# ---------------------------------------------------------------------------
# the three views
# ---------------------------------------------------------------------------


def test_a_valid_manifest_makes_a_unit_manifested(tmp_path):
    repo = _repo(tmp_path)
    _manifest(_unit(repo, "results", "run_a"))

    unit = _by_path(scan(repo))["results/run_a"]
    assert unit.manifest_state == "valid"
    assert unit.manifested and not unit.orphan


def test_an_unaccounted_directory_is_an_orphan(tmp_path):
    repo = _repo(tmp_path)
    _unit(repo, "results", "run_a")

    unit = _by_path(scan(repo))["results/run_a"]
    assert unit.manifest_state == "absent"
    assert unit.orphan and not unit.cited and not unit.manifested


def test_a_cited_unit_is_not_an_orphan_even_without_a_manifest(tmp_path):
    repo = _repo(tmp_path)
    _unit(repo, "results", "run_a")
    _doc(repo, "note.md", "measured in `results/run_a`\n")
    _commit(repo)

    unit = _by_path(scan(repo))["results/run_a"]
    assert unit.cited and not unit.orphan


def test_orphan_is_exactly_not_cited_and_not_manifested(tmp_path):
    repo = _repo(tmp_path)
    _manifest(_unit(repo, "results", "both"))
    _manifest(_unit(repo, "results", "manifested_only"))
    _unit(repo, "results", "cited_only")
    _unit(repo, "results", "neither")
    _doc(repo, "note.md", "both / cited_only\n")
    _commit(repo)

    units = _by_path(scan(repo))
    assert units["results/both"].cited and units["results/both"].manifested
    assert not units["results/both"].orphan
    assert not units["results/manifested_only"].orphan
    assert not units["results/cited_only"].orphan
    assert units["results/neither"].orphan


def test_the_views_carry_no_age_and_no_deletion_eligibility(tmp_path):
    """Disposal is AP-5, behind owner approval. A view that hinted at it here
    would let a query result be read as authorization."""
    assert {field.name for field in fields(Unit)} == {
        "path",
        "root",
        "manifest_state",
        "cited",
        "detail",
    }

    repo = _repo(tmp_path)
    _unit(repo, "results", "run_a")
    text = render(scan(repo), loose_entries(repo), repo_root=repo).lower()
    for word in ("delete", "deletable", "eligible", "safe to remove", "days old"):
        assert word not in text, f"the projection must not imply disposal: {word!r}"


# ---------------------------------------------------------------------------
# an invalid manifest is a broken producer, never an orphan
# ---------------------------------------------------------------------------


def test_an_invalid_manifest_is_reported_as_invalid_and_not_as_an_orphan(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "run_a")
    (run / MANIFEST_FILENAME).write_text('{"schema_version": 1}', encoding="utf-8")

    unit = _by_path(scan(repo))["results/run_a"]
    assert unit.manifest_state == "invalid"
    assert not unit.manifested
    assert not unit.orphan, (
        "a manifest that exists and does not validate is a broken producer; "
        "counting it as an orphan would hide it in the backlog"
    )
    assert invalid_manifests(scan(repo)) == (unit,)


def test_unparseable_manifest_bytes_are_invalid_not_absent(tmp_path):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "run_a")
    (run / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    assert _by_path(scan(repo))["results/run_a"].manifest_state == "invalid"


def test_check_fails_closed_on_an_invalid_manifest(tmp_path, capsys):
    repo = _repo(tmp_path)
    run = _unit(repo, "results", "run_a")
    (run / MANIFEST_FILENAME).write_text('{"schema_version": 99}', encoding="utf-8")

    assert main(["--repo-root", str(repo), "--check"]) == 1
    assert "invalid manifest" in capsys.readouterr().err


def test_check_passes_on_a_workspace_with_no_assets_at_all(tmp_path, capsys):
    """A clean clone is the normal CI case, and an empty inventory is correct."""
    repo = _repo(tmp_path)
    assert main(["--repo-root", str(repo), "--check"]) == 0
    assert "0 unit(s)" in capsys.readouterr().out


def test_check_passes_when_every_manifest_validates(tmp_path):
    repo = _repo(tmp_path)
    _manifest(_unit(repo, "results", "run_a"))
    _unit(repo, "results", "run_b")  # an orphan is a backlog item, not a failure
    assert main(["--repo-root", str(repo), "--check"]) == 0


# ---------------------------------------------------------------------------
# the citation corpus must not be self-fulfilling
# ---------------------------------------------------------------------------


def test_a_generated_view_naming_every_unit_cites_nothing(tmp_path):
    repo = _repo(tmp_path)
    _unit(repo, "results", "run_a")
    _doc(repo, "asset_inventory.generated.md", "| `results/run_a` | results |\n")
    _commit(repo)

    unit = _by_path(scan(repo))["results/run_a"]
    assert not unit.cited, "a generated view must never certify its own subjects"
    assert unit.orphan


def test_an_untracked_document_cites_nothing(tmp_path):
    """The corpus is tracked-only, which is what keeps this tool's own
    projection from citing anything no matter where it is written."""
    repo = _repo(tmp_path)
    _unit(repo, "results", "run_a")
    _doc(repo, "scratch.md", "results/run_a\n")  # written, never committed

    assert not _by_path(scan(repo))["results/run_a"].cited


def test_emitting_the_projection_into_docs_would_still_cite_nothing(tmp_path):
    """Defence in depth for the output path: even mis-targeted, it is untracked."""
    repo = _repo(tmp_path)
    _unit(repo, "results", "run_a")
    assert (
        main(["--repo-root", str(repo), "--emit", "docs/asset_inventory.generated.md"])
        == 0
    )
    assert not _by_path(scan(repo))["results/run_a"].cited


def test_a_missing_git_index_is_an_error_not_an_empty_corpus(tmp_path):
    """Guessing the corpus would silently turn every unit into an orphan."""
    (tmp_path / "results" / "run_a").mkdir(parents=True)
    with pytest.raises(InventoryError, match="cannot list tracked documents"):
        scan(tmp_path)
