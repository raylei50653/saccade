"""Contract regression for the H0 launch-hygiene pre-authorization gate.

The registry's ``future_reentry_precondition`` requires that, before the next
exactly-once authorization is granted, launch hygiene be a machine-checked,
*non-authoritative* gate that reuses the controller's real preflight predicate
and fail-closed rejects a pre-existing ``build/h0_phase_a`` tree — the exact
hazard that terminated both prior owner-authorized re-entries (#209, #224/#227)
at ``provenance_invalid`` before any capture ran.

These tests pin: (1) the gate reuses the controller's own predicate object, not
a copy; (2) the controller preflight and the gate share that single source, so
the historical failure literal cannot drift; (3) the gate is fail-closed on the
hazard and clean otherwise; and (4) it stays non-authoritative — no build tree,
no authoritative writes, no authorization consumed.
"""

# scope: system
# function: regression
# lifecycle: active

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import h0_launch_hygiene_gate as gate  # noqa: E402
import run_h0_phase_a as controller  # noqa: E402

# The historical factual-boundary reason recorded for both re-entries.  Kept as
# an independent literal so a message change in the predicate fails here.
EXPECTED_REASON = "build/h0_phase_a exists at controller launch"


def _make_build_tree(root: Path) -> None:
    (root / controller.AUTHORITATIVE_BUILD_SUBTREE).mkdir(parents=True)


def test_gate_reuses_controller_predicate_object() -> None:
    # The gate binds to the controller's live function, not a re-implementation.
    assert gate.controller is controller
    assert gate.PREDICATE_SOURCE == "run_h0_phase_a.assert_no_preexisting_build_tree"
    assert callable(controller.assert_no_preexisting_build_tree)


def test_controller_preflight_shares_the_single_source() -> None:
    # The inline duplicate is gone: preflight delegates to the named predicate,
    # so the gate's verdict is the controller's own verdict on this check.
    preflight_src = inspect.getsource(controller.preflight_controller_input)
    assert "assert_no_preexisting_build_tree(root)" in preflight_src
    # The old inline raise must not survive anywhere but the single predicate.
    raises = [
        fn
        for name, fn in vars(controller).items()
        if inspect.isfunction(fn)
        and name != "assert_no_preexisting_build_tree"
        and EXPECTED_REASON in (inspect.getsource(fn))
    ]
    assert raises == [], f"duplicate hazard literal in: {[f.__name__ for f in raises]}"


def test_predicate_message_matches_historical_literal(tmp_path: Path) -> None:
    _make_build_tree(tmp_path)
    try:
        controller.assert_no_preexisting_build_tree(tmp_path)
    except controller.ContractError as exc:
        assert str(exc) == EXPECTED_REASON
    else:  # pragma: no cover - defensive
        raise AssertionError("predicate did not fail closed on a pre-existing tree")


def test_gate_clears_on_hygienic_root(tmp_path: Path) -> None:
    report = gate.evaluate(tmp_path)
    assert report["result"] == "clear"
    assert report["reason"] is None
    assert gate.main(["--root", tmp_path.as_posix()]) == 0


def test_gate_rejects_preexisting_build_tree(tmp_path: Path) -> None:
    _make_build_tree(tmp_path)
    report = gate.evaluate(tmp_path)
    assert report["result"] == "rejected"
    assert report["reason"] == EXPECTED_REASON
    # Fail-closed: nonzero exit for the operator / CI to trip on.
    assert gate.main(["--root", tmp_path.as_posix()]) == 1


def test_gate_is_non_authoritative(tmp_path: Path) -> None:
    report = gate.evaluate(tmp_path)
    assert report["authority"] == "non_authoritative"
    assert report["authorization_consumed"] is False
    assert report["capture"] == "forbidden"
    assert report["terminal_claim"] == "forbidden"
    assert report["schema"] == "h0_launch_hygiene_gate_v1"
    assert report["checked_subtree"] == "build/h0_phase_a"
    # Screening a clean root must not materialise the very tree it screens for,
    # nor write anything else into the root.
    assert list(tmp_path.iterdir()) == []
