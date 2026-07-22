"""Contracts for the sealed S0 Amendment 1 terminal runner."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/tools/run_s0_safe_domain_runtime_transfer.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("s0_runtime_transfer", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate(
    runner: Any,
    *,
    offline_dist: list[float],
    runtime_dist: list[float] | None = None,
    is_gt: list[bool],
    tracks: list[int],
    unjoined_dist: list[float] | None = None,
) -> dict[str, Any]:
    runtime_dist = runtime_dist if runtime_dist is not None else offline_dist
    unjoined_dist = unjoined_dist if unjoined_dist is not None else []
    n = len(offline_dist)
    matched_ratio = np.linspace(0.001, 0.04, n, dtype=float)
    return runner.evaluate_arrays(
        offline_dist=np.asarray(offline_dist, dtype=float),
        offline_ratio=matched_ratio,
        runtime_dist=np.asarray(runtime_dist, dtype=float),
        runtime_ratio=matched_ratio.copy(),
        is_gt=np.asarray(is_gt, dtype=bool),
        is_fp=~np.asarray(is_gt, dtype=bool),
        track_keys=[("S", track) for track in tracks],
        unjoined_runtime_dist=np.asarray(unjoined_dist, dtype=float),
        unjoined_runtime_ratio=np.zeros(len(unjoined_dist), dtype=float),
    )


def test_zero_hurt_exposure_floor_is_59_tracks() -> None:
    runner = _load_runner()
    assert runner.clopper_pearson_upper(0, 59) <= 0.05
    assert runner.clopper_pearson_upper(0, 58) > 0.05


def test_hurt_is_aggregated_to_lost_track_before_cp() -> None:
    runner = _load_runner()
    # Two rejected GT rows share track 0; they count as one hurt trial.
    gt_tracks = [0, 0] + list(range(1, 59))
    gt_dist = [1.0, 1.0] + [0.0] * 58
    # Add enough non-GT matched pairs to pass the matched-pair floor.
    fp_n = 1000 - len(gt_dist)
    result = _evaluate(
        runner,
        offline_dist=gt_dist + [1.0] * fp_n,
        is_gt=[True] * len(gt_dist) + [False] * fp_n,
        tracks=gt_tracks + list(range(1000, 1000 + fp_n)),
    )
    row = next(
        item
        for item in result["grid"]
        if item["theta_dist_h"] == 0.2 and item["theta_abs_log_h_ratio"] == 0.05
    )
    assert row["n_gt_exposed_tracks"] == 59
    assert row["n_gt_hurt_offline_tracks"] == 1
    assert row["ucb_offline"] == pytest.approx(runner.clopper_pearson_upper(1, 59))


def test_unjoined_events_never_change_cp_and_fail_coverage_gate() -> None:
    runner = _load_runner()
    gt_dist = [0.0] * 59
    fp_dist = [1.0] * 941
    common = {
        "offline_dist": gt_dist + fp_dist,
        "is_gt": [True] * 59 + [False] * 941,
        "tracks": list(range(59)) + list(range(1000, 1941)),
    }
    clean = _evaluate(runner, **common)
    covered = _evaluate(runner, **common, unjoined_dist=[1.0] * 100)
    clean_row = clean["grid"][0]
    covered_row = covered["grid"][0]
    assert covered_row["ucb_offline"] == clean_row["ucb_offline"]
    assert covered_row["ucb_runtime"] == clean_row["ucb_runtime"]
    assert covered_row["unjoined_m"] == 100
    assert covered["terminal"] == "S0_UNDECIDABLE"


def test_identical_axes_with_clear_active_safe_set_hold() -> None:
    runner = _load_runner()
    gt_dist = [0.01 + i * 0.001 for i in range(59)]
    fp_dist = [0.3 + (i % 100) * 0.01 for i in range(941)]
    result = _evaluate(
        runner,
        offline_dist=gt_dist + fp_dist,
        is_gt=[True] * 59 + [False] * 941,
        tracks=list(range(59)) + list(range(1000, 1941)),
        unjoined_dist=[0.0] * 20,
    )
    assert result["terminal"] == "AXES_TRANSFER_HOLDS"
    assert result["validity"] == {
        "V4_exposure_floor": True,
        "V5_adversarial_unjoined_coverage": True,
        "V5_evaluated": True,
        "V7_nonempty_active_safe_set": True,
    }


def test_v5_is_not_evaluated_when_v7_has_no_active_safe_point() -> None:
    runner = _load_runner()
    # Every GT track is rejected at every frozen point, so no offline-safe point exists.
    result = _evaluate(
        runner,
        offline_dist=[3.0] * 59 + [3.0] * 941,
        is_gt=[True] * 59 + [False] * 941,
        tracks=list(range(59)) + list(range(1000, 1941)),
        unjoined_dist=[3.0] * 20,
    )
    assert result["terminal"] == "S0_UNDECIDABLE"
    assert result["validity"]["V7_nonempty_active_safe_set"] is False
    assert result["validity"]["V5_evaluated"] is False
    assert result["validity"]["V5_adversarial_unjoined_coverage"] is None


def test_runtime_gt_hurt_is_broken_when_coverage_passes() -> None:
    runner = _load_runner()
    gt_off = [0.0] * 59
    gt_rt = [1.0] + [0.0] * 58
    fp = [1.0] * 941
    result = _evaluate(
        runner,
        offline_dist=gt_off + fp,
        runtime_dist=gt_rt + fp,
        is_gt=[True] * 59 + [False] * 941,
        tracks=list(range(59)) + list(range(1000, 1941)),
        unjoined_dist=[0.0] * 20,
    )
    assert result["terminal"] == "AXES_TRANSFER_BROKEN"
