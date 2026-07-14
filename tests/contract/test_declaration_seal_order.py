"""A sealed declaration must exist before the results it claims to have predicted.

The § 20.8 seal bar assumes a declaration is fixed *before* the outcome is known.
Nothing enforced it, and the assumption did not hold: P0's declaration, runner,
results and sealed evidence packet all arrived in a single commit (`b136437f`),
while the H0 draft — committed ninety minutes *earlier* — already named
`P0_CAPTURE_SEMANTICS_INVALID` as a settled outcome. The terminal's name existed
before the study that was supposed to produce it.

Git cannot see working-tree order, so this test does not claim a seal was written
after the fact. It enforces the weaker, checkable property whose absence made the
stronger failure invisible: **a declaration must be introduced in a strictly
earlier commit than its results.** A seal nobody can audit cannot stop a terminal
from being named after a result already in hand.

The check keys on the *existence of a results document*, not on `doc-status`: that
field's own vocabulary has already drifted (`sealed-execution`,
`sealed-for-execution`, and plain `active` all name sealed studies), so keying on
it would pin the guard to a string that rots.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RESEARCH = _REPO / "docs" / "modules" / "semantic" / "research"

# Studies that predate this guard and cannot satisfy it. Their seals are **not
# auditable from this repository** — an entry on the books, not an absolution.
# Adding a row here is a deliberate act: it records that a study's pre-registration
# cannot be proved, and it must never be used to wave a new study through.
GRANDFATHERED: dict[str, str] = {
    "d0_runtime_shadow_fidelity": "declaration + results in one commit (79624b05); seal not auditable",
    "frozen_packet_exact_key_recoverability": "declaration + results in one commit (4c5efbbd); seal not auditable",
    "runtime_bridge_decision_path_identifiability": "declaration + runner + results + packet in one commit (b136437f); seal not auditable — see its Correction 1 § C1.9",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _introducing_commit(path: Path) -> str:
    """The commit that first added `path` — its oldest touch."""
    log = _git(
        "log", "--reverse", "--format=%H", "--diff-filter=A", "--", str(path)
    ).splitlines()
    return log[0] if log else ""


def _study_pairs() -> list[tuple[str, Path, Path]]:
    """(study, declaration, results) for every declaration that has results."""
    pairs = []
    for declaration in sorted(_RESEARCH.glob("*_declaration_*.md")):
        stem = declaration.name.replace("_declaration_", "_results_")
        for candidate in (_RESEARCH / stem, _RESEARCH / "closed" / stem):
            if candidate.is_file():
                study = declaration.name.split("_declaration_")[0]
                pairs.append((study, declaration, candidate))
                break
    return pairs


@pytest.fixture(scope="module")
def history_is_complete() -> None:
    """Fail — never skip — when the checkout cannot answer the question.

    A shallow clone makes every `git log` lookup come back empty, which would turn
    this guard into a decoration that always passes. CI therefore checks out with
    `fetch-depth: 0` (see .github/workflows/ci.yml).
    """
    if _git("rev-parse", "--is-shallow-repository") == "true":
        pytest.fail(
            "shallow checkout: commit history is unavailable, so declaration seal "
            "order cannot be verified. Set `fetch-depth: 0` on actions/checkout."
        )


def test_there_are_studies_to_check(history_is_complete) -> None:
    """Guard the guard: a broken pairing rule would silently check nothing."""
    assert _study_pairs(), "no declaration/results pairs found — pairing rule broke"


@pytest.mark.parametrize(
    "study,declaration,results", _study_pairs(), ids=lambda v: getattr(v, "name", v)
)
def test_declaration_is_introduced_before_its_results(
    history_is_complete, study: str, declaration: Path, results: Path
) -> None:
    declared_at = _introducing_commit(declaration)
    resulted_at = _introducing_commit(results)

    assert declared_at, f"cannot find the commit that introduced {declaration.name}"
    assert resulted_at, f"cannot find the commit that introduced {results.name}"

    if study in GRANDFATHERED:
        pytest.xfail(f"{study}: {GRANDFATHERED[study]}")

    assert declared_at != resulted_at, (
        f"{study}: the declaration and its results were introduced in the SAME commit "
        f"({declared_at[:8]}). A seal that lands with its own result proves nothing "
        f"about pre-registration. Commit the sealed declaration first, have it "
        f"reviewed, then run the study."
    )
    # `--is-ancestor` returns 0 when true; the declaration must come first.
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", declared_at, resulted_at],
        cwd=_REPO,
        capture_output=True,
    )
    assert is_ancestor.returncode == 0, (
        f"{study}: the results commit ({resulted_at[:8]}) does not descend from the "
        f"declaration commit ({declared_at[:8]}) — the declaration cannot have been "
        f"sealed before the study ran."
    )


def test_grandfather_list_is_load_bearing(history_is_complete) -> None:
    """Every grandfathered study must actually violate the rule.

    Otherwise the list is decoration, and a stale entry would silently excuse a
    study that no longer needs excusing — or, worse, one that never did.
    """
    pairs = {study: (d, r) for study, d, r in _study_pairs()}

    for study in GRANDFATHERED:
        assert study in pairs, f"grandfathered study {study!r} no longer exists"
        declaration, results = pairs[study]
        assert _introducing_commit(declaration) == _introducing_commit(results), (
            f"{study} no longer violates the seal-order rule — remove it from "
            "GRANDFATHERED rather than leaving a dead exemption on the books."
        )
