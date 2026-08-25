"""Unit tests for the bridge-gate breakpoint locator (scripts/eval/diagnostics)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = PROJECT_ROOT / "scripts/eval/diagnostics/bridge_gate_breakpoints.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("bridge_gate_breakpoints", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bridge_gate_breakpoints"] = module
    spec.loader.exec_module(module)
    return module


bp = _load_tool()


def _measurement(value: float, per_seq: dict[str, dict[str, int]]):
    return bp.Measurement(value, bp._pooled_idf1(per_seq), per_seq, "fake")


class _StepRunner:
    """Synthetic piecewise-constant metric with jumps at known locations."""

    def __init__(self, jumps: list[float]) -> None:
        self.jumps = jumps
        self.runs = 0

    def measure(self, value: float):
        self.runs += 1
        level = sum(1 for j in self.jumps if value > j)
        per_seq = {"S1": {"idtp": 1000 - 10 * level, "idfp": 5, "idfn": 20}}
        return _measurement(value, per_seq)


def test_pooled_idf1_pools_counts_instead_of_averaging() -> None:
    """IDF1 is a ratio, so per-sequence values must never be averaged."""
    per_seq = {
        "big": {"idtp": 1000, "idfp": 0, "idfn": 0},
        "tiny": {"idtp": 0, "idfp": 1, "idfn": 1},
    }
    pooled = bp._pooled_idf1(per_seq)
    naive_mean = sum(bp._idf1(c) for c in per_seq.values()) / len(per_seq)

    assert pooled == pytest.approx(100.0 * 2000 / (2000 + 1 + 1))
    assert pooled != pytest.approx(naive_mean)


def test_fingerprint_separates_runs_that_share_a_pooled_idf1() -> None:
    """The equality predicate is the count vector, not the pooled scalar.

    Two different decision outcomes can pool to exactly the same IDF1; using the
    scalar as the bisection predicate would silently merge them into one
    plateau.
    """
    a = {
        "S1": {"idtp": 100, "idfp": 10, "idfn": 10},
        "S2": {"idtp": 200, "idfp": 20, "idfn": 20},
    }
    b = {
        "S1": {"idtp": 200, "idfp": 20, "idfn": 20},
        "S2": {"idtp": 100, "idfp": 10, "idfn": 10},
    }

    left, right = _measurement(1.0, a), _measurement(2.0, b)
    assert left.idf1 == pytest.approx(right.idf1)
    assert left.fingerprint != right.fingerprint


def test_scan_brackets_only_intervals_that_actually_change() -> None:
    runner = _StepRunner([1.35])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)

    assert len(brackets) == 1
    assert brackets[0].low.value < 1.35 <= brackets[0].high.value


def test_bisect_localises_each_jump_to_the_tolerance() -> None:
    jumps = [1.3178, 1.6821]
    runner = _StepRunner(jumps)
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    assert len(brackets) == len(jumps)

    for bracket in brackets:
        bp.bisect(runner, bracket, 1e-4)
        contained = [j for j in jumps if bracket.low.value < j <= bracket.high.value]
        assert len(contained) == 1, "a refined bracket must isolate one jump"
        assert bracket.width <= 1e-4


def test_bisection_costs_far_less_than_an_equivalent_grid() -> None:
    """The whole point of bisection on a bit-exact metric: log, not linear."""
    runner = _StepRunner([1.3178, 1.6821])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    for bracket in bp.brackets_from_scan(samples):
        bp.bisect(runner, bracket, 1e-4)

    equivalent_grid = (1.8 - 1.2) / 1e-4
    assert runner.runs < equivalent_grid / 100


def test_report_has_one_more_plateau_than_breakpoints() -> None:
    runner = _StepRunner([1.3178, 1.6821])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    for bracket in brackets:
        bp.bisect(runner, bracket, 1e-4)

    args = bp.parse_args(
        ["--axis", "h_hi", "--lo", "1.2", "--hi", "1.8", "--work-dir", "unused"]
    )
    report = bp.build_report(args, samples, brackets, runner.runs)

    assert len(report["breakpoints"]) == 2
    assert len(report["plateaus"]) == 3
    assert [p["from"] for p in report["plateaus"]] == sorted(
        p["from"] for p in report["plateaus"]
    )


def test_report_states_that_a_plateau_is_not_a_proof() -> None:
    """The tool must never let a 'no jump detected' read as 'no jump exists'."""
    runner = _StepRunner([])
    samples = bp.scan(runner, 1.2, 1.8, 3)
    args = bp.parse_args(
        ["--axis", "h_hi", "--lo", "1.2", "--hi", "1.8", "--work-dir", "unused"]
    )
    report = bp.build_report(args, samples, [], runner.runs)

    assert report["breakpoints"] == []
    assert any("do not prove" in limit for limit in report["limits"])
