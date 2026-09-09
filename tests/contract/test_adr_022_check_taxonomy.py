"""ADR 022's four representative cases, now asserting the taxonomy itself.

These were written before the default flip, pinned to what the checker did then,
so that the flip would show up here as an explicit reviewable diff rather than a
silent change. This is that diff: cases 1 and 4 moved, cases 2 and 3 did not.

* case 1 — ordinary source change, nothing consuming the publication as current:
  was exit 1, is now exit 0 with the lag reported as a warning;
* case 2 — stale-evidence consumption: invariant, still exit 1;
* case 3 — complete current publication: invariant, still exit 0, and `--strict`
  still refuses unresolved checks;
* case 4 — incomplete publication: both gaps are now closed, at read time and
  before the builder writes.
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
import h2_path_partition as partition  # noqa: E402

_CANDIDATE_WORKFLOW_REL = ".github/workflows/runtime_identity_candidate.yml"
_RUNBOOK_REL = "docs/reference/runbooks/runtime_identity_republication.md"
_ARCHIVE_REL = "docs/reference/runtime_identity/archive"

_D = "d" * 64
_E = "e" * 64
_I = "i" * 64
_S = "s" * 64
_R = "r" * 64
_P = "p" * 64
_MOVED = "9" * 64

# The portable half of the environment axis: tracked git blobs, not host state.
_RECIPE = [
    {"blob": "a" * 40, "path": "CMakeLists.txt"},
    {"blob": "b" * 40, "path": "pyproject.toml"},
    {"blob": "c" * 40, "path": "uv.lock"},
]

_COORDINATE = {
    "decision_surface": _D,
    "environment": _E,
    "implementation": _I,
    "identity_semantics": _S,
    "runtime_inputs": _R,
}


def _publication(
    *,
    complete: bool = True,
    recipe: list[dict[str, str]] | None = None,
    **coordinate: str,
) -> dict[str, Any]:
    return {
        "axes": {"environment": {"recipe": list(recipe if recipe else _RECIPE)}},
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
    recipe: list[dict[str, str]] | None = None,
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
    monkeypatch.setattr(
        identity, "environment_recipe", lambda: list(recipe if recipe else _RECIPE)
    )


# ── Case 1: ordinary source change, no evidence promotion ───────────────────
# A developer edits a decision-relevant source and promotes no evidence. Every
# binding is `unattested`, so nothing consumes the publication as current; the
# only lag is that the publication describes an older HEAD.
#
# This is the case the flip moved: ADR 022 §3 decision 1.


def test_case1_ordinary_source_change_no_longer_blocks_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"decision_surface": _MOVED},
    )
    assert staleness.main([]) == 0
    captured = capsys.readouterr()
    assert "publication lag (nothing consumes it as current)" in captured.out
    assert "decision_surface moved and was not republished" in captured.out
    assert captured.err == ""


def test_case1_is_still_a_failure_when_asserting_the_publication_describes_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lag did not stop existing; only who has to care about it changed."""
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"decision_surface": _MOVED},
    )
    assert staleness.main(["--mode", "attested"]) == 1
    assert "decision_surface moved and was not republished" in capsys.readouterr().err


def test_case1_lag_is_a_failure_again_as_soon_as_something_claims_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `current` binding plus source lag is a false current-attestation.

    The demotion in the sibling test is licensed by nothing consuming the
    publication, so it must not survive a row that does. This is ADR 022 §4's
    first remaining exit-1 case and is what keeps decision 1 from being a
    blanket exemption.
    """
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", _captured())),
        recomputes_to={"decision_surface": _MOVED},
    )
    assert staleness.main([]) == 1
    assert "decision_surface moved and was not republished" in capsys.readouterr().err


def test_case1_an_unknown_mode_is_a_failure_not_a_weaker_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
    )
    with pytest.raises(SystemExit) as excinfo:
        staleness.main(["--mode", "historical-only"])
    assert excinfo.value.code != 0


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


def test_case4_an_incomplete_publication_is_refused_at_read_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Closed: `load_published` now reads `publication_complete`.

    This is the second of the two independent places ADR 022 §4 names, and it is
    the one that catches a file which arrived by some route other than the
    builder. Every consumer goes through this reader.
    """
    _install(
        tmp_path,
        monkeypatch,
        _publication(complete=False),
        _bindings(_row("quantity.example", None)),
    )
    with pytest.raises(staleness.StalenessError, match="publication_complete"):
        staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    assert staleness.main([]) == 1
    assert "publication_complete" in capsys.readouterr().err


def test_case4_require_complete_refuses_before_it_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Closed: the builder's completeness guard now runs before `--emit`.

    With no probe and no runtime inputs the publication is incomplete. It used
    to be written first and reported second, so pointing it at the canonical
    path destroyed a complete publication and only then failed. The target must
    now survive untouched.
    """
    target = tmp_path / "runtime_identity.generated.json"
    target.write_text('{"canonical": "complete"}', encoding="utf-8")

    assert identity.main(["--emit", str(target), "--require-complete"]) == 1
    captured = capsys.readouterr()
    assert "publication is incomplete" in captured.err
    assert "nothing was written" in captured.err

    assert json.loads(target.read_text(encoding="utf-8")) == {"canonical": "complete"}


def test_case4_a_complete_publication_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must refuse incompleteness, not refuse to publish."""
    target = tmp_path / "runtime_identity.generated.json"
    monkeypatch.setattr(
        identity,
        "build_publication",
        lambda **_: {**_publication(), "publication_complete": True},
    )
    assert identity.main(["--emit", str(target), "--require-complete"]) == 0
    assert (
        json.loads(target.read_text(encoding="utf-8"))["publication_complete"] is True
    )


# ── Portable vs host-observed environment ───────────────────────────────────
# The environment axis hashes two things with very different checkability: git
# blobs any host can recompute, and the Torch/CUDA/TensorRT closure only the
# controlled host can see. Because the whole axis sat behind `--strict`, a
# `pyproject.toml`/`uv.lock` change was invisible to every ordinary check — which
# is how the canonical publication was stale from `89515241` (Optuna removal)
# without anything noticing.

_MOVED_RECIPE = [
    {"blob": "a" * 40, "path": "CMakeLists.txt"},
    {"blob": "d" * 40, "path": "pyproject.toml"},
    {"blob": "e" * 40, "path": "uv.lock"},
]


def test_recipe_lag_is_caught_without_the_controlled_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The coverage fix: portable, so it does not wait for `--strict`."""
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recipe=_MOVED_RECIPE,
    )
    assert staleness.main(["--mode", "attested"]) == 1
    failures = capsys.readouterr().err
    assert "environment recipe moved and was not republished" in failures
    # Names what moved and what did not, rather than one opaque digest pair.
    assert "pyproject.toml" in failures and "uv.lock" in failures
    assert "CMakeLists.txt" not in failures


def test_recipe_lag_is_publication_lag_not_a_development_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It is lag like any other portable axis, so decision 1 applies to it too."""
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recipe=_MOVED_RECIPE,
    )
    assert staleness.main([]) == 0
    assert "publication lag" in capsys.readouterr().out


def test_the_observed_toolchain_still_waits_for_the_controlled_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split must not drag host state into the portable tier.

    `environment_axis` is what reads the installed Torch/CUDA/TensorRT closure.
    A generic runner must keep reporting it unresolved rather than comparing
    itself to a GPU publication and manufacturing drift.
    """
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"environment": _MOVED},
    )
    published = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    assert staleness.ENVIRONMENT_RECIPE not in staleness.static_axis_lag(published)

    failures, warnings = staleness.compare_publication(published, probe=None)
    assert not failures
    assert any("host-specific environment" in item for item in warnings)

    failures, _ = staleness.compare_publication(
        published, probe=None, verify_environment=True
    )
    assert any("environment moved" in item for item in failures)


def test_a_publication_without_recipe_detail_is_unresolved_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same treatment runtime inputs and the probe already get."""
    publication = _publication()
    del publication["axes"]
    _install(
        tmp_path, monkeypatch, publication, _bindings(_row("quantity.example", None))
    )
    published = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    failures, warnings = staleness.compare_publication(published, probe=None)
    assert not failures
    assert any("records no environment recipe" in item for item in warnings)


# ── The two consumers of `compare_publication` (ADR 022 §7.1) ───────────────
# `research_lock.publication_precondition` and `run_h2_layer_p.preflight` both
# unpack a 2-tuple, and Layer P stores `warnings` verbatim in a certificate
# field. The mode split had to be built out of shared helpers rather than by
# rewriting this function, so the contract is asserted rather than assumed.


def test_compare_publication_keeps_its_shape_and_wording() -> None:
    import inspect

    signature = inspect.signature(staleness.compare_publication)
    assert list(signature.parameters) == [
        "published",
        "probe",
        "runtime_input_manifest",
        "verify_environment",
    ]
    for name in ("probe", "runtime_input_manifest", "verify_environment"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["verify_environment"].default is False


def test_the_axis_failure_wording_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-for-byte, because `main` reclassifies by name but callers read text."""
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"implementation": _MOVED},
    )
    published = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    failures, warnings = staleness.compare_publication(published, probe=None)
    assert isinstance(failures, list) and isinstance(warnings, list)
    assert failures[0] == (
        f"implementation moved and was not republished: published {_I}, "
        f"recomputed {_MOVED}. {staleness.REGENERATE_HINT}"
    )


def test_static_axis_lag_reports_both_sides_of_every_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper both readings share; empty when the publication is current."""
    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
        recomputes_to={"identity_semantics": _MOVED},
    )
    published = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    assert staleness.static_axis_lag(published) == {"identity_semantics": (_S, _MOVED)}

    _install(
        tmp_path,
        monkeypatch,
        _publication(),
        _bindings(_row("quantity.example", None)),
    )
    current = staleness.load_published(tmp_path / staleness.PUBLISHED_REL)
    assert staleness.static_axis_lag(current) == {}


# ── Candidate capture (ADR 022 §5) ──────────────────────────────────────────
# The recovery path out of a stale coordinate. `runtime_identity.yml` refuses to
# produce a probe until the coordinate is already current, so it can verify an
# existing publication but never collect the evidence that would update one.
# These pin the two properties that make the candidate path a recovery path at
# all, plus the one that keeps it from needing a republication of its own.


def _candidate_workflow() -> str:
    return (_REPO / _CANDIDATE_WORKFLOW_REL).read_text(encoding="utf-8")


def test_the_candidate_workflow_does_not_require_a_current_coordinate() -> None:
    """The precondition it exists to avoid, named literally.

    `runtime_identity.yml` gates on this step before capturing anything. If it
    ever reappears here the file stops being a recovery path and silently
    becomes a second verifier.
    """
    workflow = _candidate_workflow()
    assert "Coordinate must already be current" not in workflow
    # The staleness invocation it does carry is a report, not a gate.
    for line in workflow.splitlines():
        if "check_runtime_identity_staleness.py" in line:
            break
    else:  # pragma: no cover - defensive
        pytest.fail("the candidate workflow no longer reports coordinate lag")
    invocation = workflow.split("check_runtime_identity_staleness.py", 1)[1]
    assert invocation.split("- name:", 1)[0].rstrip().endswith("|| true")


def test_the_candidate_workflow_is_manual_and_read_only() -> None:
    """Same shape as `test_controlled_host_diagnostic_is_manual_only`."""
    workflow = _candidate_workflow()
    triggers = workflow.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert "workflow_dispatch:" in triggers
    assert "pull_request:" not in triggers
    assert "push:" not in triggers
    assert "paths:" not in triggers
    assert "permissions:\n  contents: read\n" in workflow
    assert "ref: ${{ inputs.ref }}" in workflow


def test_the_candidate_workflow_captures_a_fresh_probe() -> None:
    """ADR 022 §3 decision 2: a candidate never reuses an earlier probe."""
    workflow = _candidate_workflow()
    assert "h2_behavioral_identity.py \\\n            --identity-mode" in workflow
    assert "--require-complete" in workflow


def test_the_candidate_workflow_never_emits_over_the_canonical_publication() -> None:
    """`--require-complete` still writes before it refuses (case 4 above).

    Until that ordering is fixed, emitting to the canonical path destroys a
    complete publication and only then reports failure, so no automated caller
    may name it as a target.
    """
    workflow = _candidate_workflow()
    assert staleness.PUBLISHED_REL not in workflow
    assert '--emit "${RUNNER_TEMP}/runtime_identity.candidate.json"' in workflow


def test_the_candidate_workflow_does_not_move_a_published_axis() -> None:
    """Why this PR needs no republication of its own.

    Only the exact path `.github/workflows/runtime_identity.yml` is identity
    semantics; the `.github/` prefix is plumbing, and plumbing is not one of the
    published coordinate axes. Adding a sibling workflow therefore moves nothing.
    """
    assert partition.classify(_CANDIDATE_WORKFLOW_REL) == "plumbing_only"
    assert _CANDIDATE_WORKFLOW_REL not in partition.IDENTITY_SEMANTICS_PATHS
    assert "plumbing_only" not in identity.ALL_COORDINATE_AXES


def test_the_runbook_documents_the_archive_and_not_the_destructive_emit() -> None:
    """The supported path is prose, so the two load-bearing parts are pinned.

    The archive location is ADR 022 §3 decision 5; the forbidden combination is
    the one that overwrites a complete canonical before reporting failure.
    """
    runbook = (_REPO / _RUNBOOK_REL).read_text(encoding="utf-8")
    assert _ARCHIVE_REL in runbook
    assert f"--emit {staleness.PUBLISHED_REL}" not in runbook
    assert (_REPO / _ARCHIVE_REL / "README.md").is_file()


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
