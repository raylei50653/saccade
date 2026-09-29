"""Doc structure gates by rule role: layout warns, state projection fails.

#493 PR-2 split C6.4 by what a miss costs. A closed note left in an active
directory or a checkbox in a TODO register is layout — reported, never blocking.
A closed note listed as Active, prose in the contracts layer, or a threads index
that contradicts its cards lets a reader take closed work as current, so those
stay violations under ``--strict``.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "scripts" / "tools"
sys.path.insert(0, str(TOOLS))

import check_doc_structure as chk  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(chk, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(chk, "MODULES_ROOT", tmp_path / "docs/modules")
    monkeypatch.setattr(chk, "RESEARCH_ROOT", tmp_path / "docs/research")
    _write(tmp_path / "docs/modules/m/README.md", "# m\n\n## Closed\n\n- x\n")
    _write(tmp_path / "docs/modules/m/TODO.md", "# TODO\n")
    _write(tmp_path / "docs/research/README.md", "# research\n")
    return tmp_path


def _closed_note(root: Path) -> None:
    _write(
        root / "docs/modules/m/research/done.md",
        "# done\n\ndoc-status: closed\n",
    )


def test_closed_note_in_active_path_only_warns(tree: Path) -> None:
    _closed_note(tree)
    assert any("[L1]" in w for w in chk.check_layout())
    assert chk.check_lifecycle() == []


def test_todo_checkbox_only_warns(tree: Path) -> None:
    _write(tree / "docs/modules/m/TODO.md", "# TODO\n\n- [ ] a task\n")
    assert any("[L5]" in w for w in chk.check_layout())
    assert chk.check_lifecycle() == []


def test_closed_note_listed_as_active_still_fails(tree: Path) -> None:
    _closed_note(tree)
    _write(tree / "docs/modules/m/README.md", "# m\n\n## Active\n\n- done.md\n")
    violations = chk.check_lifecycle()
    assert any("[L2]" in v for v in violations)
    assert not any("[L1]" in v for v in violations)


def test_closed_note_already_moved_but_listed_as_active_still_fails(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L2 must not depend on L1: moving the file does not fix the index."""
    _write(
        tree / "docs/modules/m/research/closed/done.md",
        "# done\n\ndoc-status: closed\n",
    )
    _write(tree / "docs/modules/m/README.md", "# m\n\n## Active\n\n- done.md\n")
    # A README nearer by depth must not be mistaken for the owner.
    _write(tree / "docs/modules/m/research/README.md", "# research\n")

    assert chk.check_layout() == []
    assert any(
        "[L2]" in v and "docs/modules/m/README.md" in v for v in chk.check_lifecycle()
    )
    monkeypatch.setattr(sys, "argv", ["check_doc_structure.py", "--strict"])
    assert chk.main() == 1


def test_closed_research_note_resolves_to_its_area_index(tree: Path) -> None:
    _write(
        tree / "docs/research/eval/closed/done.md",
        "# done\n\ndoc-status: closed\n",
    )
    _write(tree / "docs/research/eval/README.md", "# eval\n\n## Active\n\n- done.md\n")
    assert any(
        "[L2]" in v and "docs/research/eval/README.md" in v
        for v in chk.check_lifecycle()
    )


def test_closed_note_in_closed_section_passes(tree: Path) -> None:
    _write(
        tree / "docs/modules/m/research/closed/done.md",
        "# done\n\ndoc-status: closed\n",
    )
    _write(
        tree / "docs/modules/m/README.md",
        "# m\n\n## Active\n\n- other.md\n\n## Closed\n\n- done.md\n",
    )
    assert chk.check_lifecycle() == []


def test_owning_readme_ignores_lifecycle_directories(tree: Path) -> None:
    _write(tree / "docs/modules/m/research/README.md", "# research\n")
    note = tree / "docs/modules/m/research/closed/done.md"
    assert chk._owning_readme(note) == tree / "docs/modules/m/README.md"
    _write(tree / "docs/research/eval/README.md", "# eval\n")
    note = tree / "docs/research/eval/closed/sub/done.md"
    assert chk._owning_readme(note) == tree / "docs/research/eval/README.md"
    note = tree / "docs/research/other/closed/done.md"
    assert chk._owning_readme(note) == tree / "docs/research/README.md"


def test_prose_in_contracts_layer_still_fails(tree: Path) -> None:
    _write(tree / "docs/research/contracts/essay.md", "# essay\n")
    assert any("[L3]" in v for v in chk.check_lifecycle())


def test_threads_index_contradicting_a_card_still_fails(tree: Path) -> None:
    threads = tree / "docs/research/threads"
    _write(threads / "t.md", "# t\ndoc-status: parked\nwip-role: parked\n")
    _write(
        threads / "README.md",
        "# threads\n\n## Active\n\n| card | role |\n|---|---|\n| [t.md](t.md) | **parked** |\n",
    )
    assert any("[L4]" in v for v in chk.check_lifecycle())


def test_strict_exits_zero_on_layout_only(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _closed_note(tree)
    _write(tree / "docs/modules/m/TODO.md", "# TODO\n\n- [x] done\n")
    monkeypatch.setattr(sys, "argv", ["check_doc_structure.py", "--strict"])
    assert chk.main() == 0


def test_strict_exits_nonzero_on_state_projection(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tree / "docs/research/contracts/essay.md", "# essay\n")
    monkeypatch.setattr(sys, "argv", ["check_doc_structure.py", "--strict"])
    assert chk.main() == 1
