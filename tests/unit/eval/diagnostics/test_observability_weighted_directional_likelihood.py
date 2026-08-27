"""Tests for observability-weighted directional likelihood math and preseal lock."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOL_PATH = (
    PROJECT_ROOT / "scripts/tools/observability_weighted_directional_likelihood.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("owdl", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["owdl"] = module
    spec.loader.exec_module(module)
    return module


owdl = _load_tool()


def _linear_window(speed: float) -> np.ndarray:
    frames = np.arange(4, dtype=np.float64)
    return np.column_stack((speed * frames, np.zeros(4, dtype=np.float64)))


def test_ols_uses_actual_frame_spacing() -> None:
    frames = np.array([2.0, 4.0, 7.0, 11.0])
    points = np.column_stack((3.0 + 1.5 * frames, -2.0 + 0.25 * frames))

    fit = owdl.fit_ols_motion(points, frames)

    assert fit.velocity == pytest.approx([1.5, 0.25])
    assert fit.residuals == pytest.approx(np.zeros((4, 2)), abs=1e-12)
    assert sum(fit.slope_weights) == pytest.approx(0.0)


def test_noise_covariance_requires_positive_definite_residual_support() -> None:
    frames = np.arange(4, dtype=np.float64)
    x_only_noise = np.array([[0.1, 0.0], [-0.1, 0.0], [0.1, 0.0], [-0.1, 0.0]])

    with pytest.raises(owdl.ObservabilityError, match="positive definite"):
        owdl.estimate_normalized_noise_covariance(
            [(_linear_window(1.0) + x_only_noise, frames, np.ones(4))]
        )


def test_estimated_noise_covariance_is_symmetric_positive_definite() -> None:
    frames = np.arange(4, dtype=np.float64)
    first_noise = np.array([[0.08, 0.02], [-0.12, -0.04], [0.04, 0.09], [0.0, -0.07]])
    second_noise = np.array(
        [[-0.03, 0.11], [0.09, -0.02], [-0.08, -0.06], [0.02, 0.04]]
    )

    covariance = owdl.estimate_normalized_noise_covariance(
        [
            (_linear_window(0.5) + first_noise, frames, np.ones(4)),
            (_linear_window(1.0) + second_noise, frames, np.ones(4)),
        ]
    )

    assert covariance == pytest.approx(covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_zero_velocity_degenerates_to_uniform_without_a_speed_threshold() -> None:
    observation = owdl.observe_direction(
        lost_points=np.zeros((4, 2)),
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=np.array([1.0, 0.0]),
        candidate_first_height=1.0,
        gap=1,
        normalized_noise_covariance=np.eye(2) * 0.01,
    )

    assert observation.q_v == pytest.approx(0.0)
    assert observation.delta_angle is None
    assert math.isinf(observation.angular_variance)
    assert observation.kappa == 0.0
    assert observation.raw_direction_cost == 1.0
    assert observation.weighted_direction_cost == 0.0


def test_higher_velocity_snr_increases_concentration() -> None:
    common = {
        "lost_frames": np.arange(4),
        "lost_heights": np.ones(4),
        "candidate_first_height": 1.0,
        "gap": 2,
        "normalized_noise_covariance": np.eye(2) * 0.01,
    }
    slow = owdl.observe_direction(
        lost_points=_linear_window(0.1),
        candidate_first_point=np.array([0.5, 0.0]),
        **common,
    )
    fast = owdl.observe_direction(
        lost_points=_linear_window(1.0),
        candidate_first_point=np.array([5.0, 0.0]),
        **common,
    )

    assert fast.q_v > slow.q_v
    assert fast.angular_variance < slow.angular_variance
    assert fast.kappa > slow.kappa


def test_near_zero_velocity_is_continuous_not_thresholded() -> None:
    tiny = owdl.observe_direction(
        lost_points=_linear_window(1e-9),
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=np.array([1.0, 0.0]),
        candidate_first_height=1.0,
        gap=1,
        normalized_noise_covariance=np.eye(2) * 0.01,
    )

    assert tiny.delta_angle == pytest.approx(0.0)
    assert tiny.q_v > 0.0
    assert 0.0 < tiny.kappa < 1e-12
    assert tiny.weighted_direction_cost == pytest.approx(0.0, abs=1e-12)


def test_shared_endpoint_cross_covariance_is_propagated() -> None:
    observation = owdl.observe_direction(
        lost_points=_linear_window(1.0),
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=np.array([5.0, 1.0]),
        candidate_first_height=1.0,
        gap=2,
        normalized_noise_covariance=np.eye(2) * 0.04,
    )
    fit = owdl.fit_ols_motion(_linear_window(1.0), np.arange(4))
    expected = -fit.slope_weights[-1] * np.eye(2) * 0.04 / 2.0

    assert observation.velocity_displacement_cross_covariance == pytest.approx(expected)


def test_von_mises_cost_keeps_normalizer_and_uniform_limit() -> None:
    assert owdl.uniform_relative_von_mises_nll(1.2, 0.0) == 0.0
    aligned_low = owdl.uniform_relative_von_mises_nll(0.0, 0.5)
    aligned_high = owdl.uniform_relative_von_mises_nll(0.0, 5.0)
    opposed_high = owdl.uniform_relative_von_mises_nll(math.pi, 5.0)

    assert aligned_high < aligned_low < 0.0
    assert opposed_high > 0.0


def test_cli_rejects_formal_execution_before_seal() -> None:
    with pytest.raises(SystemExit):
        owdl.parse_args([])


def test_preflight_rejects_an_incomplete_source_identity_set(tmp_path: Path) -> None:
    spec_path = tmp_path / "study.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema": owdl.STUDY_SCHEMA,
                "study_id": owdl.STUDY_ID,
                "status": "preseal_implementation",
                "execution_authorized": False,
                "source_files": [],
                "score_declaration": str(
                    owdl.DEFAULT_SCORE_DECLARATION.relative_to(owdl.ROOT)
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(owdl.ObservabilityError, match="source identity set mismatch"):
        owdl.verify_study_spec(spec_path)
