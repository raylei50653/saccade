from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO / "docs/modules/semantic/research/evidence/"
    "gap_conditioned_motion_e2_family_20260711/run_e2_family_freeze.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("gap_motion_e2", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ou_kernel_is_bounded_by_constant_velocity_kernel():
    runner = _load_runner()
    gaps = np.asarray([1.0, 10.0, 30.0, 90.0, 300.0])
    cv = runner.kernel_scale("M1P-GLOBAL-CV", gaps)

    np.testing.assert_allclose(cv, gaps**2)
    for model_id in runner.HALF_LIFE_BY_MODEL:
        ou = runner.kernel_scale(model_id, gaps)
        assert np.all(ou > 0)
        assert np.all(ou < cv)


def test_fit_and_score_keep_energy_terms_separate():
    runner = _load_runner()
    gaps = np.arange(1.0, 9.0)
    drift = np.asarray([0.25, -0.5])
    residual = np.asarray(
        [
            [0.1, 0.0],
            [-0.1, 0.1],
            [0.0, -0.1],
            [0.2, 0.1],
            [-0.2, -0.1],
            [0.1, -0.2],
            [-0.1, 0.2],
            [0.0, 0.0],
        ]
    )
    displacements = gaps[:, None] * (drift + residual)

    artifact = runner.fit_model("M1P-GLOBAL-CV", displacements, gaps)
    scores = runner.score_model(artifact, displacements, gaps)

    assert artifact["dimension"] == 2
    assert set(scores) == {
        "q_motion",
        "log_det_covariance",
        "gaussian_constant",
        "nll_motion",
    }
    np.testing.assert_allclose(
        scores["nll_motion"],
        0.5
        * (
            scores["q_motion"]
            + scores["log_det_covariance"]
            + scores["gaussian_constant"]
        ),
    )


def test_selector_uses_frozen_order_for_ties():
    runner = _load_runner()
    totals = {model_id: 10.0 for model_id in runner.MODEL_ORDER}
    assert runner.select_family_member(totals) == "M1P-GLOBAL-CV"

    totals["M2P-GLOBAL-OU-H90"] = 9.0
    assert runner.select_family_member(totals) == "M2P-GLOBAL-OU-H90"
