"""§20.11 research study protocol: tier derived, exploratory uncitable, freeze before data.

Each behavior is exercised in a throwaway git repository with a bare remote as
``origin``, because the freeze is a statement about commits and published tags,
not about files. The three #493 PR-3 negative cases are the classes
``TestFrozenDeclarationEdited``, ``TestRunnerNotBound`` and
``TestDataBeforeFreeze``; none of them may yield a formally valid attempt.

Drift prevented: conclusion drift. Entry points: pytest (pre-push hook and CI).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import research_study as rs  # noqa: E402

SID = "demo_study"
STUDY_DIR = f"{rs.STUDIES_REL}/{SID}"
STUDY_REL = f"{STUDY_DIR}/study.yaml"
DECL_REL = f"{STUDY_DIR}/declaration.md"
RUNNER_REL = "scripts/demo_runner.py"
TAG = "freeze/demo_study/1"

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}

STUDY_YAML = """\
schema: research_study_v1
study_id: demo_study
evidence_tier: exploratory
hypothesis: the drift first appears at the top-k stage
section_20_2:
  output_class: [diagnostic]
  mainline_transition:
    LOCALIZED: none
    UNRESOLVED: none
declaration: declaration.md
results: results.md
runner: scripts/demo_runner.py
inputs:
  packet: data/packet.txt
validity_criteria:
  V1: every input sequence produced a probe row
attempt_policy:
  adoption: {adoption}
  max_valid_attempts: {max_valid}
  max_attempts: 3
"""

RUNNER_SOURCE = """\
from research_study import StudyBinding, open_frozen_study

BINDING = StudyBinding(
    study_id="demo_study",
    runner_file=__file__,
    freeze_tag="freeze/demo_study/1",
    pinned_blobs={},
)


def main():
    study = open_frozen_study(BINDING)
    return study.input("packet").read_text()
"""


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root.as_posix(), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_GIT_ENV,
    ).stdout.strip()


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def commit_all(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


def freeze(root: Path, tag: str = TAG) -> None:
    git(root, "tag", "-a", tag, "-m", "freeze")
    git(root, "push", "-q", "origin", "HEAD:refs/heads/main", f"refs/tags/{tag}")


def pins(root: Path, *paths: str) -> dict[str, str]:
    return {p: git(root, "rev-parse", f"HEAD:{p}") for p in paths}


def binding(
    root: Path, pinned: dict[str, str] | None = None, runner: str = RUNNER_REL
) -> rs.StudyBinding:
    return rs.StudyBinding(
        study_id=SID,
        runner_file=root / runner,
        freeze_tag=TAG,
        pinned_blobs=pins(root, STUDY_REL, DECL_REL) if pinned is None else pinned,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path)


def make_repo(
    tmp_path: Path,
    *,
    adoption: str = "first_valid",
    max_valid: int = 1,
    runner: str = RUNNER_SOURCE,
) -> Path:
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "-q", "--bare", remote.as_posix()], check=True, env=_GIT_ENV
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "main", work.as_posix()], check=True, env=_GIT_ENV
    )
    git(work, "remote", "add", "origin", remote.as_posix())
    write(work, STUDY_REL, STUDY_YAML.format(adoption=adoption, max_valid=max_valid))
    write(
        work, DECL_REL, "<!-- evidence-tier: exploratory -->\n# Declaration\n\nbody\n"
    )
    write(work, RUNNER_REL, runner)
    write(work, "data/packet.txt", "frozen evidence\n")
    commit_all(work, "declare")
    freeze(work)
    return work


def run_attempt(
    root: Path,
    validity: str = "valid",
    terminal: str | None = "LOCALIZED",
    criterion: str | None = None,
) -> Path:
    study = rs.open_frozen_study(binding(root), root=root)
    (study.payload_dir() / "probe.csv").write_text(
        study.input("packet").read_text(), encoding="utf-8"
    )
    path = study.record(validity, terminal=terminal, invalid_criterion=criterion)
    commit_all(root, f"attempt {study.attempt}")
    return path


def study_doc(root: Path) -> dict[str, Any]:
    return rs.parse_study((root / STUDY_REL).read_bytes())


def verify(root: Path) -> list[str]:
    problems, _ = rs.study_problems(root, root / STUDY_DIR)
    return problems


# ------------------------------------------------------------------ happy path


def test_declared_frozen_run_is_a_valid_attempt(repo: Path) -> None:
    path = run_attempt(repo)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["freeze"]["commit"] == git(repo, "rev-parse", f"{TAG}^{{commit}}")
    assert set(record["files"]) == {"probe.csv"}
    assert verify(repo) == []
    assert rs.adopt(study_doc(repo), [record]) == "LOCALIZED"


# ------------------------------------------------ negative 1: declaration edited


class TestFrozenDeclarationEdited:
    def test_uncommitted_edit_blocks_the_freeze(self, repo: Path) -> None:
        write(
            repo,
            DECL_REL,
            "<!-- evidence-tier: exploratory -->\n# Declaration\n\nedited\n",
        )
        with pytest.raises(rs.StudyError, match="not clean"):
            rs.open_frozen_study(binding(repo), root=repo)

    def test_committed_edit_after_the_tag_is_not_the_freeze(self, repo: Path) -> None:
        pinned = pins(repo, STUDY_REL, DECL_REL)
        write(
            repo,
            DECL_REL,
            "<!-- evidence-tier: exploratory -->\n# Declaration\n\nedited\n",
        )
        commit_all(repo, "edit")
        with pytest.raises(rs.StudyError) as exc:
            rs.open_frozen_study(binding(repo, pinned), root=repo)
        assert any("peels to" in p for p in exc.value.problems)
        assert any(DECL_REL in p and "runner pins" in p for p in exc.value.problems)

    def test_tag_moved_onto_an_edited_declaration_fails_the_pin(
        self, repo: Path
    ) -> None:
        pinned = pins(repo, STUDY_REL, DECL_REL)
        write(
            repo,
            DECL_REL,
            "<!-- evidence-tier: exploratory -->\n# Declaration\n\nedited\n",
        )
        commit_all(repo, "edit")
        git(repo, "tag", "-f", "-a", TAG, "-m", "moved")
        git(
            repo,
            "push",
            "-q",
            "-f",
            "origin",
            "HEAD:refs/heads/main",
            f"refs/tags/{TAG}",
        )
        with pytest.raises(rs.StudyError, match="runner pins"):
            rs.open_frozen_study(binding(repo, pinned), root=repo)

    def test_unpublished_retag_is_not_a_freeze(self, repo: Path) -> None:
        commit_all_empty = git(repo, "commit", "-q", "--allow-empty", "-m", "later")
        del commit_all_empty
        git(repo, "tag", "-f", "-a", TAG, "-m", "local only")
        with pytest.raises(rs.StudyError, match="unpublished freeze"):
            rs.open_frozen_study(binding(repo), root=repo)

    def test_editing_the_frozen_body_after_an_attempt_invalidates_it(
        self, repo: Path
    ) -> None:
        run_attempt(repo)
        write(
            repo,
            DECL_REL,
            "<!-- evidence-tier: exploratory -->\n# Declaration\n\nrewritten\n",
        )
        assert any("frozen body changed" in p for p in verify(repo))

    def test_appending_below_the_frozen_body_is_allowed(self, repo: Path) -> None:
        run_attempt(repo)
        with (repo / DECL_REL).open("a", encoding="utf-8") as handle:
            handle.write("\n## Amendment 1\n\nappended\n")
        assert verify(repo) == []


# ---------------------------------------------------- negative 2: runner unbound


class TestRunnerNotBound:
    def test_study_naming_another_runner_refuses_this_one(self, repo: Path) -> None:
        write(repo, "scripts/other_runner.py", RUNNER_SOURCE)
        commit_all(repo, "second runner")
        git(repo, "tag", "-f", "-a", TAG, "-m", "refreeze")
        git(
            repo,
            "push",
            "-q",
            "-f",
            "origin",
            "HEAD:refs/heads/main",
            f"refs/tags/{TAG}",
        )
        with pytest.raises(rs.StudyError, match="binds runner"):
            rs.open_frozen_study(
                binding(repo, runner="scripts/other_runner.py"), root=repo
            )

    def test_binding_that_does_not_pin_the_study_is_refused(self, repo: Path) -> None:
        with pytest.raises(rs.StudyError, match="does not pin"):
            rs.open_frozen_study(binding(repo, pins(repo, DECL_REL)), root=repo)

    def test_lightweight_tag_is_not_a_freeze(self, repo: Path) -> None:
        git(repo, "tag", "-d", TAG)
        git(repo, "tag", TAG)
        git(repo, "push", "-q", "-f", "origin", f"refs/tags/{TAG}")
        with pytest.raises(rs.StudyError, match="not an annotated tag"):
            rs.open_frozen_study(binding(repo), root=repo)

    def test_runner_that_never_opens_the_study_is_unbound(self) -> None:
        problems = rs.runner_source_problems("print('hi')\n", {"inputs": {}})
        assert any("never calls open_frozen_study" in p for p in problems)

    def test_attempt_claiming_a_different_runner_is_rejected(self, repo: Path) -> None:
        path = run_attempt(repo)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["runner"] = "scripts/other_runner.py"
        path.write_text(json.dumps(record), encoding="utf-8")
        assert any("not the study's runner" in p for p in verify(repo))


# ------------------------------------------------- negative 3: data before freeze


class TestDataBeforeFreeze:
    def test_handle_cannot_be_built_without_the_freeze(self, repo: Path) -> None:
        with pytest.raises(rs.StudyError, match="only issued by open_frozen_study"):
            rs.FrozenStudy(
                object(),
                root=repo,
                study=study_doc(repo),
                attempt=1,
                runner=RUNNER_REL,
                tag=TAG,
                commit="0" * 40,
                pinned={},
            )

    def test_failed_freeze_hands_out_no_data(self, repo: Path) -> None:
        write(repo, "stray.txt", "dirty\n")
        with pytest.raises(rs.StudyError):
            rs.open_frozen_study(binding(repo), root=repo)

    def test_undeclared_input_is_not_reachable(self, repo: Path) -> None:
        study = rs.open_frozen_study(binding(repo), root=repo)
        with pytest.raises(rs.StudyError, match="not a declared input"):
            study.input("other")

    def test_runner_naming_a_data_path_is_refused_before_data(
        self, tmp_path: Path
    ) -> None:
        leaky = RUNNER_SOURCE + '\nPREVIEW = open("data/packet.txt").read()\n'
        work = make_repo(tmp_path, runner=leaky)
        with pytest.raises(rs.StudyError, match="names data path"):
            rs.open_frozen_study(binding(work), root=work)

    @pytest.mark.parametrize(
        "literal", ["datasets/MOT17/train", "results/foo/run.json", "data"]
    )
    def test_forbidden_literals(self, literal: str) -> None:
        source = f"from research_study import open_frozen_study\nX = {literal!r}\nopen_frozen_study(None)\n"
        study = {"inputs": {"packet": "data/packet.txt"}}
        assert any(
            "names data path" in p for p in rs.runner_source_problems(source, study)
        )

    def test_hand_written_attempt_without_a_freeze_is_rejected(
        self, repo: Path
    ) -> None:
        head = git(repo, "rev-parse", "HEAD")
        write(repo, DECL_REL.replace("declaration.md", "attempts/001/probe.csv"), "x\n")
        record = {
            "schema": rs.ATTEMPT_SCHEMA,
            "study_id": SID,
            "attempt": 1,
            "runner": RUNNER_REL,
            "freeze": {
                "tag": "freeze/never",
                "commit": head,
                "pinned_blobs": pins(repo, STUDY_REL, DECL_REL),
            },
            "validity": "valid",
            "terminal": "LOCALIZED",
            "invalid_criterion": None,
            "files": {"probe.csv": __import__("hashlib").sha256(b"x\n").hexdigest()},
        }
        write(repo, f"{STUDY_DIR}/attempts/001/attempt.json", json.dumps(record))
        assert any(
            "freeze tag 'freeze/never' does not exist" in p for p in verify(repo)
        )

    def test_results_present_at_the_freeze_invalidate_the_attempt(
        self, tmp_path: Path
    ) -> None:
        work = tmp_path / "w"
        work.mkdir()
        repo = make_repo(work)
        git(repo, "tag", "-d", TAG)
        git(repo, "push", "-q", "origin", f":refs/tags/{TAG}")
        write(
            repo,
            f"{STUDY_DIR}/results.md",
            "<!-- evidence-tier: exploratory -->\nprewritten\n",
        )
        commit_all(repo, "results first")
        freeze(repo)
        run_attempt(repo)
        assert any("results existed at the freeze commit" in p for p in verify(repo))


# ------------------------------------------------------ attempts and adoption


def test_tampered_payload_is_detected(repo: Path) -> None:
    path = run_attempt(repo)
    (path.parent / "probe.csv").write_text("edited\n", encoding="utf-8")
    assert any("payload files differ" in p for p in verify(repo))


def test_invalid_attempt_must_cite_a_predeclared_criterion(repo: Path) -> None:
    study = rs.open_frozen_study(binding(repo), root=repo)
    with pytest.raises(rs.StudyError, match="predeclared validity criterion"):
        study.record("invalid", terminal=None, invalid_criterion="runner looked wrong")


def test_valid_result_is_not_rerun(repo: Path) -> None:
    run_attempt(repo)
    git(repo, "tag", "-f", "-a", TAG, "-m", "refreeze")
    git(repo, "push", "-q", "-f", "origin", "HEAD:refs/heads/main", f"refs/tags/{TAG}")
    with pytest.raises(rs.StudyError, match="not rerun"):
        rs.open_frozen_study(binding(repo), root=repo)


def test_invalid_attempt_allows_a_new_appended_attempt_under_a_new_tag(
    repo: Path,
) -> None:
    run_attempt(repo, "invalid", None, "V1")
    freeze(repo, "freeze/demo_study/2")
    study = rs.open_frozen_study(
        rs.StudyBinding(
            SID,
            repo / RUNNER_REL,
            "freeze/demo_study/2",
            pins(repo, STUDY_REL, DECL_REL),
        ),
        root=repo,
    )
    (study.payload_dir() / "probe.csv").write_text("x\n", encoding="utf-8")
    study.record("valid", terminal="LOCALIZED")
    commit_all(repo, "attempt 2")
    assert study.attempt == 2
    assert verify(repo) == []


def test_reusing_an_attempts_tag_is_refused(repo: Path) -> None:
    run_attempt(repo, "invalid", None, "V1")
    git(repo, "tag", "-f", "-a", TAG, "-m", "refreeze")
    git(repo, "push", "-q", "-f", "origin", "HEAD:refs/heads/main", f"refs/tags/{TAG}")
    with pytest.raises(rs.StudyError, match="already froze an earlier attempt"):
        rs.open_frozen_study(binding(repo), root=repo)


def test_squashed_away_freeze_commit_is_rejected(repo: Path) -> None:
    run_attempt(repo)
    freeze_commit = git(repo, "rev-parse", f"{TAG}^{{commit}}")
    # Rebuild the same tree as a single root commit: the freeze commit is gone
    # from HEAD's history, as after a squash merge.
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    squashed = git(repo, "commit-tree", tree, "-m", "squash")
    git(repo, "reset", "-q", "--hard", squashed)
    problems = verify(repo)
    assert any(freeze_commit[:12] in p and "not an ancestor" in p for p in problems)


def test_adoption_is_by_rule_never_by_recency() -> None:
    study = {"attempt_policy": {"adoption": "first_valid"}}
    records = [
        {"attempt": 1, "validity": "invalid", "terminal": None},
        {"attempt": 2, "validity": "valid", "terminal": "A"},
        {"attempt": 3, "validity": "valid", "terminal": "B"},
    ]
    assert rs.adopt(study, records) == "A"
    assert (
        rs.adopt({"attempt_policy": {"adoption": "unanimous_valid"}}, records)
        == rs.UNRESOLVED
    )
    assert rs.adopt(study, records[:1]) is None


# ------------------------------------------------------------------ tier rules


def _section(transitions: dict[str, str], classes: list[str]) -> dict[str, Any]:
    return {"output_class": classes, "mainline_transition": transitions}


@pytest.mark.parametrize(
    "transitions,classes,tier",
    [
        ({"A": "none"}, ["diagnostic"], "exploratory"),
        ({"A": "closes_core_unknown"}, ["diagnostic"], "exploratory"),
        ({"A": "adds_decision_capability", "B": "none"}, ["diagnostic"], "formal"),
        ({"A": "changes_production_behavior"}, ["diagnostic"], "formal"),
        ({"A": "none"}, ["diagnostic", "design_candidate"], "formal"),
    ],
)
def test_tier_is_derived_from_section_20_2(transitions, classes, tier) -> None:
    assert rs.derive_tier(_section(transitions, classes)) == tier


def test_author_label_disagreeing_with_derivation_fails(repo: Path) -> None:
    text = (
        (repo / STUDY_REL)
        .read_text(encoding="utf-8")
        .replace("LOCALIZED: none", "LOCALIZED: changes_production_behavior")
    )
    write(repo, STUDY_REL, text)
    assert any("derive 'formal'" in p for p in verify(repo))


def test_md_artifacts_carry_the_tier_header(repo: Path) -> None:
    write(repo, DECL_REL, "# Declaration without header\n")
    assert any("evidence-tier: exploratory" in p for p in verify(repo))


def test_formal_study_needs_a_runner() -> None:
    doc = rs.parse_study(
        STUDY_YAML.format(adoption="first_valid", max_valid=1)
        .replace("evidence_tier: exploratory", "evidence_tier: formal")
        .replace("output_class: [diagnostic]", "output_class: [design_candidate]")
    )
    for key in ("runner", "inputs", "validity_criteria", "attempt_policy"):
        doc.pop(key)
    assert any("must name its runner" in p for p in rs.study_schema_problems(doc, SID))


def test_duplicate_yaml_key_is_rejected() -> None:
    with pytest.raises(rs.StudyError, match="duplicate key"):
        rs.parse_study("schema: a\nschema: b\n")


# --------------------------------------------------------------- citation rule


def test_formal_chain_citing_exploratory_fails(repo: Path) -> None:
    write(
        repo,
        "docs/research/contracts/claim_state_registry.md",
        f"see {SID} for evidence\n",
    )
    studies = {SID: study_doc(repo)}
    assert any(
        "cites exploratory study" in p for p in rs.citation_problems(repo, studies)
    )


def test_non_formal_docs_may_cite_exploratory(repo: Path) -> None:
    write(repo, "docs/research/threads/next_steps.md", f"guided by {SID}\n")
    assert rs.citation_problems(repo, {SID: study_doc(repo)}) == []


def test_similar_names_are_not_citations(repo: Path) -> None:
    write(repo, "docs/research/evidence_ledger.md", f"{SID}_v2 is a different study\n")
    assert rs.citation_problems(repo, {SID: study_doc(repo)}) == []


# ------------------------------------------------------------- relabel history


def test_tier_change_after_merge_is_rejected(repo: Path) -> None:
    git(repo, "branch", "base")
    text = (repo / STUDY_REL).read_text(encoding="utf-8")
    text = text.replace("evidence_tier: exploratory", "evidence_tier: formal").replace(
        "output_class: [diagnostic]", "output_class: [design_candidate]"
    )
    write(repo, STUDY_REL, text)
    assert any(
        "tier changed after merge" in p for p in rs.history_problems(repo, "base")
    )


def test_merged_attempt_is_append_only(repo: Path) -> None:
    path = run_attempt(repo)
    git(repo, "branch", "base")
    (path.parent / "probe.csv").write_text("rewritten\n", encoding="utf-8")
    assert any("append-only" in p for p in rs.history_problems(repo, "base"))


# ----------------------------------------------------------------- live tree


def test_live_tree_studies_satisfy_the_protocol() -> None:
    problems = rs.check_all(_REPO, rs.default_base(_REPO))
    assert problems == [], "\n".join(problems)
