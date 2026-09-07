"""ADR 022's four representative cases, pinned to the checker's behavior today.

These fixtures do not assert the taxonomy ADR 022 proposes; they assert what the
current checker does for each case, so that the default flip in a later PR shows
up here as an explicit, reviewable diff rather than as a silent change.

Cases 2 and 4 are invariants: their verdicts must survive the flip. Cases 1 and 3
are the ones the flip is allowed to move, and only case 1's exit code.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import check_runtime_identity_staleness as staleness  # noqa: E402

_D = "d" * 64
_E = "e" * 64
_I = "i" * 64
_S = "s" * 64
_R = "r" * 64
_P = "p" * 64
_MOVED = "9" * 64

_COORDINATE = {
    "decision_surface": _D,
    "environment": _E,
    "implementation": _I,
    "identity_semantics": _S,
    "runtime_inputs": _R,
}


def _publication(*, complete: bool = True, **coordinate: str) -> dict[str, Any]:
    return {
        "coordinate": {**_COORDINATE, **coordinate},
        "equivalence": {
            "proof": None,
            "state": "unproven",
            "note": "probe equality is not equivalence",
        },
        "probe": {
            "digest": _P,
            "kind": "identity_probe",
            "sufficiency": "fixture_change_detector_only",
        },
        "publication_complete": complete,
        "schema": identity.IDENTITY_SCHEMA,
    }


def _bindings(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"schema": staleness.BINDINGS_SCHEMA, "bindings": list(rows)}


def _row(name: str, captured_under: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "object": name,
        "captured_under": captured_under,
        "state_owner": "docs/research/contracts/claim_state_registry.md",
    }


def _captured(**coordinate: str) -> dict[str, Any]:
    return {"coordinate": {**_COORDINATE, **coordinate}, "probe": _P}


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: dict[str, Any],
    bindings: dict[str, Any],
    *,
    recomputes_to: dict[str, str] | None = None,
) -> None:
    """Point the CLI at a synthetic repo and pin what the source axes recompute to."""
    for rel, payload in (
        (staleness.PUBLISHED_REL, publication),
        (staleness.BINDINGS_REL, bindings),
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(staleness, "REPO_ROOT", tmp_path)

    measured = {
        "decision_surface": _D,
        "implementation": _I,
        "identity_semantics": _S,
        "environment": _E,
        **(recomputes_to or {}),
    }
    for axis, digest in measured.items():
        monkeypatch.setattr(
            identity, f"{axis}_axis", lambda digest=digest: {"digest": digest}
        )


# ── Case 1: ordinary source change, no evidence promotion ───────────────────
# A developer edits a decision-relevant source and promotes no evidence. Every
# binding is `unattested`, so nothing consumes the publication as current; the
# only lag is that the publication describes an older HEAD.


def test_case1_ordinary_source_change_is_blocked_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"decision_surface": _MOVED},
    )
    assert staleness.main([]) == 1
    assert "decision_surface moved and was not republished" in capsys.readouterr().err


def test_case1_nothing_consumes_the_publication_as_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precondition that makes ADR 022 decision 1 safe to apply here.

    The block in the sibling test comes from `compare_publication`'s static-axis
    recomputation, not from the binding walk: no row is `current`, so no evidence
    is being read as true of HEAD.
    """
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"decision_surface": _MOVED},
    )
    published = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    bindings = staleness.load_bindings(tmp_path / staleness.BINDINGS_REL)
    target = {
        "coordinate": published["coordinate"],
        "probe": published["probe"]["digest"],
    }
    verdicts = {
        staleness.classify_binding(row.get("captured_under"), target)
        for row in bindings["bindings"]
    }
    assert verdicts == {"unattested"}


# ── Case 2: stale-evidence consumption (invariant) ──────────────────────────
# A binding claims it was captured under a coordinate whose decision surface has
# since moved. This must exit 1 before and after the default flip.


def test_case2_stale_evidence_consumption_is_inadmissible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", _captured(decision_surface=_MOVED))),
    )
    assert staleness.main([]) == 1
    captured = capsys.readouterr()
    assert "stale bindings are inadmissible" in captured.err
    assert "quantity.example" in captured.err


def test_case2_re_attestation_is_a_separate_inadmissible_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Equal probe plus implementation drift is not a behavior-preserving shortcut."""
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", _captured(implementation=_MOVED))),
    )
    assert staleness.main([]) == 1
    assert (
        "re_attestation_required bindings are inadmissible" in capsys.readouterr().err
    )


# ── Case 3: complete new publication ────────────────────────────────────────
# Source axes match recomputation and the publication is marked complete. The
# host-specific axes stay unresolved on a generic runner and are warnings, not
# failures — that separation is what lets CI run this at all.


def test_case3_a_complete_current_publication_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", _captured())),
    )
    assert staleness.main([]) == 0
    captured = capsys.readouterr()
    assert "current                    quantity.example" in captured.out
    assert "warning: host-specific environment was not recomputed" in captured.out
    assert captured.err == ""


def test_case3_strict_still_refuses_unresolved_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--strict` means "unresolved is a failure"; that meaning must not be reused.

    Runtime-input and probe currentness stay unresolved unless the caller supplies
    `--runtime-inputs-from` and `--probe-from`, so a development-mode fixture must
    never be pinned to exit 0 under `--strict`. Note environment is *compared*
    under `--strict`, never unresolved — it is not the reason this exits 1.
    """
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", _captured())),
    )
    assert staleness.main(["--strict"]) == 1
    assert "strict mode: unresolved checks are failures" in capsys.readouterr().err


# ── Case 4: incomplete-publication replacement (invariant, not yet enforced) ─
# ADR 022 §4 requires that an incomplete publication never replaces a complete
# canonical. Today nothing enforces it, in two independent places.


def test_case4_an_incomplete_publication_is_accepted_by_the_checker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gap: `load_published` never reads `publication_complete`.

    ADR 022 §4 makes this fail-closed. Pinned here so that change is visible.
    """
    _install(
        tmp_path,
        monkeypatch,
        _publication(complete=False),
        _bindings(_row("quantity.example", None)),
    )
    published = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    assert published["publication_complete"] is False
    assert staleness.main([]) == 0


def test_case4_require_complete_reports_after_it_has_already_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gap: the builder's completeness guard runs after `--emit` writes the file.

    With no probe and no runtime inputs the publication is incomplete, yet the
    target has already been overwritten by the time the non-zero exit is
    returned. Pointed at the canonical path this destroys a complete publication
    and then reports failure. ADR 022 §4 requires refusing before the write.
    """
    target = tmp_path / "runtime_identity.generated.json"
    target.write_text('{"canonical": "complete"}', encoding="utf-8")

    assert identity.main(["--emit", str(target), "--require-complete"]) == 1
    assert "publication is incomplete" in capsys.readouterr().err

    replaced = json.loads(target.read_text(encoding="utf-8"))
    assert replaced["publication_complete"] is False
    assert "canonical" not in replaced


# ── Live tree ───────────────────────────────────────────────────────────────


def test_the_checked_in_canonical_is_a_complete_publication() -> None:
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    assert published["publication_complete"] is True


def test_the_checked_in_bindings_hold_nothing_current() -> None:
    """Case 1 is the live tree's situation, not a hypothetical."""
    published = staleness.load_published(_REPO / staleness.PUBLISHED_REL)
    bindings = staleness.load_bindings(_REPO / staleness.BINDINGS_REL)
    target = {
        "coordinate": published["coordinate"],
        "probe": published["probe"]["digest"],
    }
    verdicts = {
        staleness.classify_binding(row.get("captured_under"), target)
        for row in bindings["bindings"]
    }
    assert "current" not in verdicts
