"""Contract tests for the scripts/ self-documentation + generated index.

These guard the invariants that the committed tree must satisfy its own
`check_scripts_structure` contract, and that the generator/checker actually
detect the failure modes they claim to (missing status, un-indexed script,
orphan block left behind after a directory is emptied).
"""

# scope: system
# function: contract
# lifecycle: active

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "scripts" / "tools"
sys.path.insert(0, str(TOOLS))

import build_scripts_index as idx  # noqa: E402
import check_scripts_structure as chk  # noqa: E402


def _mkfile(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_committed_tree_satisfies_its_own_contract():
    """The regression guard: every tracked script self-documents and the index is fresh."""
    violations = chk.check_self_documentation() + chk.check_index_fresh()
    assert violations == [], (
        "scripts structure violations on committed tree:\n" + "\n".join(violations)
    )


def test_extract_reads_status_docstring_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "REPO", tmp_path)
    _mkfile(
        tmp_path / "scripts/a.py",
        '"""Do a thing."""\n# status: stable\nimport argparse\n',
    )
    assert idx.extract("scripts/a.py") == ("stable", "Do a thing.", "cli")
    _mkfile(tmp_path / "scripts/b.py", "import os\n")
    assert idx.extract("scripts/b.py") == ("", "", "-")


def test_missing_status_and_docstring_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "REPO", tmp_path)
    _mkfile(tmp_path / "scripts/bad.py", "import os\n")
    monkeypatch.setattr(idx, "tracked_scripts", lambda: ["scripts/bad.py"])
    problems = chk.check_self_documentation()
    assert any("missing `# status:`" in p for p in problems)
    assert any("missing module docstring" in p for p in problems)


def test_invalid_label_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "REPO", tmp_path)
    _mkfile(tmp_path / "scripts/x.py", '"""d."""\n# status: bogus\n')
    monkeypatch.setattr(idx, "tracked_scripts", lambda: ["scripts/x.py"])
    assert any("invalid status 'bogus'" in p for p in chk.check_self_documentation())


def test_build_indexes_every_tracked_script(tmp_path, monkeypatch):
    """An un-indexed script must change the roll-up (the 401-vs-403 failure mode)."""
    monkeypatch.setattr(idx, "REPO", tmp_path)
    monkeypatch.setattr(idx, "ROLLUP", tmp_path / "rollup.md")
    _mkfile(tmp_path / "scripts/tools/foo.py", '"""Foo tool."""\n# status: stable\n')
    monkeypatch.setattr(idx, "tracked_scripts", lambda: ["scripts/tools/foo.py"])
    monkeypatch.setattr(idx, "tracked_readmes", lambda: [])
    writes, metas = idx.build()
    assert "scripts/tools/foo.py" in metas
    rollup = writes[tmp_path / "rollup.md"]
    assert "foo.py" in rollup
    assert "Total tracked scripts: **1**" in rollup


def test_marker_mention_in_prose_is_not_treated_as_a_block():
    prose = f"See the `{idx.BEGIN}` block near the end.\n\nmore text\n"
    assert idx.BLOCK_RE.search(prose) is None
    assert idx.STRIP_RE.search(prose) is None
    real = f"{idx.BEGIN}\nrow\n{idx.END}\n"
    assert idx.BLOCK_RE.search(real) is not None
    assert idx.STRIP_RE.search(real) is not None


def test_status_inside_docstring_is_not_a_valid_header(tmp_path, monkeypatch):
    """A `# status:` line living inside the module docstring must not count."""
    monkeypatch.setattr(idx, "REPO", tmp_path)
    body = '"""Do a thing.\n\n# status: stable\n"""\n\nimport os\n'
    _mkfile(tmp_path / "scripts/fake.py", body)
    status, _desc, _usage = idx.extract("scripts/fake.py")
    assert status == "", (
        "docstring-embedded status must not satisfy the header contract"
    )
    # a real header comment after the docstring is picked up
    _mkfile(tmp_path / "scripts/real.py", '"""Do a thing."""\n# status: stable\n')
    assert idx.extract("scripts/real.py")[0] == "stable"


def test_description_is_the_first_paragraph_not_a_wrapped_fragment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(idx, "REPO", tmp_path)
    body = (
        '"""Test the hypothesis that a wrapped summary\n'
        "continues onto a second line.\n\n"
        'Details paragraph that must be excluded."""\n'
        "# status: diagnostic\n"
    )
    _mkfile(tmp_path / "scripts/wrap.py", body)
    _status, desc, _usage = idx.extract("scripts/wrap.py")
    assert (
        desc
        == "Test the hypothesis that a wrapped summary continues onto a second line."
    )
    assert "Details paragraph" not in desc


def test_orphan_block_is_removed_when_directory_has_no_scripts(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "REPO", tmp_path)
    monkeypatch.setattr(idx, "ROLLUP", tmp_path / "rollup.md")
    readme = tmp_path / "scripts/emptied/README.md"
    _mkfile(
        readme,
        f"# emptied\n\nprose kept.\n\n## Script index\n\n{idx.BEGIN}\n{idx.BANNER}\n\n| `gone.py` |\n\n{idx.END}\n",
    )
    monkeypatch.setattr(idx, "tracked_scripts", lambda: [])
    monkeypatch.setattr(idx, "tracked_readmes", lambda: ["scripts/emptied/README.md"])
    writes, _ = idx.build()
    assert readme in writes
    assert idx.BEGIN not in writes[readme]
    assert "prose kept." in writes[readme]
