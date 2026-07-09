"""Tests for constrained FP pruning / safe-reject metrics."""

from __future__ import annotations

import numpy as np

from saccade.perception.eval.signal_tables import (
    classify_safe_level,
    constrained_fp_prune_metrics,
    frontier_fp_removed_at_eps,
)


def test_classify_safe_level() -> None:
    assert classify_safe_level(0.0) == "eps0"
    assert classify_safe_level(0.0005) == "eps0_1pct"
    assert classify_safe_level(0.005) == "eps1pct"
    assert classify_safe_level(0.02) == "unsafe"


def test_safe_reject_zero_gt_hurt() -> None:
    # 2 GT pos, 3 FP; reject only FP
    y = np.array([True, True, False, False, False])
    rej = np.array([False, False, True, True, False])
    m = constrained_fp_prune_metrics(y, rej, rule_name="only_fp")
    assert m["GT_hurt"] == 0
    assert m["FP_removed"] == 2
    assert m["FP_removed_per_GT_hurt"] == "safe"
    assert m["safe_level"] == "eps0"
    assert m["rule_class"] == "safe_reject"


def test_risky_reject_hurts_gt() -> None:
    y = np.array([True, True, False, False, False])
    # reject 1 GT + 2 FP
    rej = np.array([True, False, True, True, False])
    m = constrained_fp_prune_metrics(y, rej, rule_name="mix")
    assert m["GT_hurt"] == 1
    assert m["GT_hurt_rate"] == 0.5
    assert m["FP_removed"] == 2
    assert m["FP_removed_per_GT_hurt"] == 2.0
    assert m["safe_level"] == "unsafe"


def test_frontier_eps0_picks_max_fp_without_gt() -> None:
    # scores higher = more reject-like
    # GT have scores 0.1, 0.2; FP have 0.5, 0.9, 0.95
    y = np.array([True, True, False, False, False])
    s = np.array([0.1, 0.2, 0.5, 0.9, 0.95])
    fr = frontier_fp_removed_at_eps(y, s, higher_means_more_reject=True)
    eps0 = next(x for x in fr if x["epsilon"] == 0.0)
    assert eps0["feasible"] is True
    # thr must be > max GT score so no GT hurt; max FP below that
    assert eps0["GT_hurt"] == 0
    assert eps0["FP_removed"] == 3  # all FP scores > 0.2
