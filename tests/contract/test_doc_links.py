"""Doc link checks distinguish local artifacts from missing repository files."""

# scope: system
# function: contract
# lifecycle: active

import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "scripts" / "tools"
sys.path.insert(0, str(TOOLS))

import check_doc_links as chk  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("out/\nresults/\n*.json\n!kept.json\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(chk, "REPO_ROOT", tmp_path)
    return tmp_path


def test_artifact_presence_does_not_change_output(repo, capsys):
    (repo / "docs/note.md").write_text(
        "[local](../out/report.json)\n"
        "[root](/results/plot.png)\n"
        "[encoded](../out/a%20b.json:12:3#part)\n"
        "[directory](../out/run/)\n"
    )
    assert chk.main() == 0
    absent = capsys.readouterr().out
    assert "4 local artifact reference(s)" in absent
    for path in ("out/report.json", "results/plot.png", "out/a b.json"):
        target = repo / path
        target.parent.mkdir(exist_ok=True)
        target.touch()
    (repo / "out/run").mkdir()
    assert chk.main() == 0
    assert capsys.readouterr().out == absent


def test_missing_tracked_file_matching_ignore_rule_still_fails(repo, capsys):
    target = repo / "out/tracked.json"
    target.parent.mkdir()
    target.touch()
    subprocess.run(["git", "add", "-f", "out/tracked.json"], cwd=repo, check=True)
    (repo / "README.md").write_text("[tracked](out/tracked.json)\n")
    assert chk.main() == 0
    capsys.readouterr()
    target.unlink()
    assert chk.main() == 1
    output = capsys.readouterr().out
    assert "1 broken doc link(s)" in output
    assert "local artifact" not in output


def test_negated_ignore_and_ordinary_missing_links_fail(repo, capsys):
    (repo / "README.md").write_text(
        "[negated](kept.json)\n[missing](docs/missing.md)\n"
        "[artifact](out/missing.json)\n"
    )
    assert chk.main() == 1
    output = capsys.readouterr().out
    assert "1 local artifact reference(s)" in output
    assert "2 broken doc link(s)" in output


def test_existing_links_fragments_line_suffixes_and_skipped_links(repo, capsys):
    (repo / "docs/target.md").touch()
    (repo / "README.md").write_text(
        "[relative](docs/target.md:12:3#part)\n"
        "[root](/docs/target.md#part)\n"
        f"[absolute]({repo}/docs/target.md)\n"
        "[anchor](#part) [web](https://example.com) [mail](mailto:a@b.com)\n"
        "```\n[example](missing.md)\n```\n"
    )
    assert chk.main() == 0
    assert "all 3 relative doc links resolve" in capsys.readouterr().out


def test_git_failure_is_not_silently_accepted(repo, capsys):
    (repo / "README.md").write_text("[local](out/report.json)\n")
    (repo / ".git").rename(repo / "git-disabled")
    assert chk.main() == 1
    assert "git ls-files failed" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["global", "info", "untracked", "modified"])
def test_local_ignore_rules_do_not_hide_broken_links(repo, capsys, source):
    (repo / "README.md").write_text(
        "[missing](docs/missing.md)\n[artifact](out/report.json)\n"
    )
    assert chk.main() == 1
    baseline = capsys.readouterr().out
    if source == "global":
        exclude = repo / "global-ignore"
        exclude.write_text("*.md\n")
        subprocess.run(
            ["git", "config", "core.excludesFile", str(exclude)], cwd=repo, check=True
        )
    elif source == "info":
        (repo / ".git/info").mkdir(exist_ok=True)
        (repo / ".git/info/exclude").write_text("docs/missing.md\n")
    elif source == "untracked":
        (repo / "docs/.gitignore").write_text("missing.md\n")
    else:
        (repo / ".gitignore").write_text("docs/missing.md\n")
    assert chk.main() == 1
    assert capsys.readouterr().out == baseline


def test_indexed_nested_rules_and_negation_ignore_worktree_edits(repo, capsys):
    rules = repo / "docs/.gitignore"
    rules.write_text("*.png\n!kept.png\n")
    subprocess.run(["git", "add", "docs/.gitignore"], cwd=repo, check=True)
    (repo / "README.md").write_text(
        "[artifact](docs/output.png)\n[missing](docs/kept.png)\n"
    )
    assert chk.main() == 1
    baseline = capsys.readouterr().out
    assert "1 local artifact reference(s)" in baseline
    assert "1 broken doc link(s)" in baseline
    rules.unlink()
    assert chk.main() == 1
    assert capsys.readouterr().out == baseline


def test_global_config_environment_is_not_inherited(repo, capsys, monkeypatch):
    exclude = repo / "global-ignore"
    exclude.write_text("*.md\n")
    (repo / "README.md").write_text("[missing](docs/missing.md)\n")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.excludesFile")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(exclude))
    assert chk.main() == 1
    output = capsys.readouterr().out
    assert "1 broken doc link(s)" in output
    assert "local artifact" not in output
