"""#537 C2: degradation-level hysteresis contract (D-537-2).

Pins the adopted target semantics of `ResourceManager.decide_degradation_level()`
(docs/modules/resource/architecture_degradation.md §5):

* escalation is immediate and may skip levels: > 85 / > 92 / > 96;
* recovery drops at most one level per decision, and only when usage is
  strictly below the current level's threshold − 5: E < 91, F < 87, R < 80;
* equality never triggers a transition.

The fix is the clamp after the escalation step: escalation can no longer lower
a level that recovery should still hold. Mutation negative controls (run
against these tests, output in the PR) cover strict/non-strict comparisons,
multi-level recovery, clamp placement, the deadband width and single-step
escalation. Keying the recovery chain on `new_level` instead of
`current_level` is an equivalent mutant: escalation and recovery are disjoint
whenever thresholds ascend and the deadband is non-negative.

T7 pins behavior the fix must not move. T8 pins O0: usage outside the
finite range 0..100 is refused as an unavailable measurement and leaves the
level untouched (failure_degradation_537.md §22).
"""

# scope: resource
# function: contract
# lifecycle: active

from __future__ import annotations

import itertools
from decimal import Decimal
from fractions import Fraction
from unittest import mock

import numpy as np
import pynvml
import pytest

from saccade.resource.resource_manager import (
    DegradationLevel as L,
    ResourceManager,
    ResourceMonitorUnavailable,
    VRAMStats,
)

N, R, F, E = L.NORMAL, L.REDUCED, L.FAST_PATH, L.EMERGENCY


def _manager(current: L, usages: list[float]) -> ResourceManager:
    with mock.patch.object(pynvml, "nvmlInit", return_value=None):
        manager = ResourceManager()
    feed = iter(usages)
    manager.get_stats = lambda: VRAMStats(0.0, 0.0, 0.0, next(feed))  # type: ignore[method-assign]
    manager.current_level = current
    return manager


def _decide(current: L, usage: float) -> L:
    return _manager(current, [usage]).decide_degradation_level()


def _run(current: L, usages: list[float]) -> list[L]:
    manager = _manager(current, usages)
    return [manager.decide_degradation_level() for _ in usages]


# ── T1: full single-decision table at every threshold edge ──────────────────
# Written by hand from D-537-2, not computed.
USAGES = [0.0, 79.9, 80.0, 84.9, 85.0, 85.1, 86.9, 87.0, 90.9, 91.0, 92.0, 92.1, 96.0, 96.1, 100.0]  # fmt: skip
EXPECTED = {
    #  0  79.9 80 84.9 85 85.1 86.9 87 90.9 91 92 92.1 96 96.1 100
    N: [N, N,  N, N,   N, R,   R,   R, R,   R, R, F,   F, E,   E],
    R: [N, N,  R, R,   R, R,   R,   R, R,   R, R, F,   F, E,   E],
    F: [R, R,  R, R,   R, R,   R,   F, F,   F, F, F,   F, E,   E],
    E: [F, F,  F, F,   F, F,   F,   F, F,   E, E, E,   E, E,   E],
}  # fmt: skip
TABLE = [
    (c, u, x) for c, row in EXPECTED.items() for u, x in zip(USAGES, row, strict=True)
]


@pytest.mark.parametrize(
    ("current", "usage", "expected"),
    TABLE,
    ids=[f"{c.name}@{u}" for c, u, _ in TABLE],
)
def test_t1_single_decision_table(current: L, usage: float, expected: L) -> None:
    assert _decide(current, usage) is expected


# ── T2: issue counterexamples ────────────────────────────────────────────────
def test_t2_fast_path_holds_at_90() -> None:
    assert _decide(F, 90.0) is F


def test_t2_emergency_holds_at_94() -> None:
    assert _decide(E, 94.0) is E


def test_t2_emergency_never_drops_two_levels_at_91_to_92() -> None:
    assert [_decide(E, u) for u in (91.0, 91.5, 92.0)] == [E, E, E]


# ── T3: escalation is immediate and may skip levels ─────────────────────────
@pytest.mark.parametrize(
    ("current", "usage", "expected"),
    [(N, 92.1, F), (N, 96.1, E), (R, 96.1, E), (N, 100.0, E)],
)
def test_t3_cross_level_escalation(current: L, usage: float, expected: L) -> None:
    assert _decide(current, usage) is expected


# ── T4: recovery is one level per decision ──────────────────────────────────
def test_t4_stepwise_recovery_from_emergency_at_zero() -> None:
    assert _run(E, [0.0] * 4) == [F, R, N, N]


def test_t4_recovery_waits_for_each_lower_threshold() -> None:
    # E -> F at 90.9; F holds until < 87; R holds until < 80.
    assert _run(E, [90.9, 90.9, 86.9, 80.0, 79.9]) == [F, F, R, R, N]


# ── T5: oscillation inside a deadband causes no transition ──────────────────
@pytest.mark.parametrize(
    ("entry", "low", "high", "level"),
    [(93.0, 87.0, 93.0, F), (97.0, 91.0, 97.0, E), (86.0, 80.0, 86.0, R)],
    ids=["FAST_PATH", "EMERGENCY", "REDUCED"],
)
def test_t5_oscillation_inside_deadband_is_stable(
    entry: float, low: float, high: float, level: L
) -> None:
    usages = [entry] + [low, high] * 50
    assert _run(N, usages) == [level] * len(usages)


# ── T6: structural properties ───────────────────────────────────────────────
# Supported domain only; out-of-domain usage is refused in T8 (O0).
GRID = [round(x * 0.1, 1) for x in range(1001)]  # 0.0 .. 100.0


@pytest.mark.parametrize("current", [N, R, F, E], ids=lambda c: c.name)
def test_t6_monotone_in_usage(current: L) -> None:
    out = [_decide(current, u) for u in GRID]
    assert all(a <= b for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("current", [N, R, F, E], ids=lambda c: c.name)
def test_t6_drops_at_most_one_level(current: L) -> None:
    assert min(_decide(current, u) for u in GRID) >= max(current - 1, N)


def test_t6_two_decision_paths_match_reference_model() -> None:
    def up(u: float) -> L:
        return E if u > 96 else F if u > 92 else R if u > 85 else N

    def ref(cur: L, u: float) -> L:
        if up(u) > cur:
            return up(u)
        down = {E: 91, F: 87, R: 80}
        return L(cur - 1) if cur in down and u < down[cur] else cur

    coarse = [0.0, 79.9, 80.0, 85.0, 85.1, 86.9, 87.0, 90.9, 91.0, 92.0, 92.1, 96.0, 96.1]  # fmt: skip
    for cur, u1, u2 in itertools.product([N, R, F, E], coarse, coarse):
        first = ref(cur, u1)
        assert _run(cur, [u1, u2]) == [first, ref(first, u2)], (cur, u1, u2)


# ── T7: behavior this fix must not move ─────────────────────────────────────
def test_t7_nvml_init_failure_preserves_emergency() -> None:
    """D-537-4A: unavailable measurements cannot recover hysteresis state."""
    with mock.patch.object(pynvml, "nvmlInit", side_effect=RuntimeError("nvml")):
        manager = ResourceManager()
    manager.current_level = E
    for _ in range(3):
        with pytest.raises(ResourceMonitorUnavailable):
            manager.decide_degradation_level()
        assert manager.current_level is E


def test_t7_initial_level_is_normal() -> None:
    """R0: a new manager starts at NORMAL and inherits no earlier state."""
    with mock.patch.object(pynvml, "nvmlInit", return_value=None):
        assert ResourceManager().current_level is N


# ── T8: out-of-domain usage is refused (O0) ─────────────────────────────────
# Decimal is not numbers.Real; Decimal("NaN") raises InvalidOperation if compared.
INVALID = [float("nan"), float("inf"), float("-inf"), -1.0, -1e-9, 100.0001, 101.0, "50", None, Decimal("NaN"), Decimal("50")]  # fmt: skip


@pytest.mark.parametrize("usage", INVALID, ids=repr)
@pytest.mark.parametrize("current", [N, R, F, E], ids=lambda c: c.name)
def test_t8_out_of_domain_usage_is_refused_and_holds_level(
    current: L, usage: object
) -> None:
    manager = _manager(current, [usage])  # type: ignore[list-item]
    with pytest.raises(ResourceMonitorUnavailable, match="usage_percent") as exc:
        manager.decide_degradation_level()
    assert isinstance(exc.value.__cause__, ValueError)
    assert manager.current_level is current


def test_t8_refusals_do_not_advance_hysteresis() -> None:
    """Refused samples are not decisions: recovery resumes one level at a time."""
    manager = _manager(E, [-1.0, float("nan"), -1.0, 0.0, 0.0])
    for _ in range(3):
        with pytest.raises(ResourceMonitorUnavailable):
            manager.decide_degradation_level()
        assert manager.current_level is E
    assert manager.decide_degradation_level() is F
    assert manager.decide_degradation_level() is R


@pytest.mark.parametrize(
    ("usage", "expected"),
    [(0, F), (100, E), (np.int64(0), F), (np.float64(100.0), E), (Fraction(181, 2), F)],
    ids=repr,
)  # fmt: skip
def test_t8_other_real_types_are_in_domain(usage: object, expected: L) -> None:
    assert _decide(E, usage) is expected  # type: ignore[arg-type]
