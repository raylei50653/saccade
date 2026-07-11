from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
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


def _write_fold_pairs(path: Path, *, held_out_offset: float = 0.0) -> None:
    fields = [
        "seq",
        "lost_id",
        "cand_id",
        "gt_match",
        "gt_valid",
        "gap",
        "lost_last_frame",
        "cand_first_frame",
        "lost_foot_x",
        "lost_foot_y",
        "cand_foot_x",
        "cand_foot_y",
        "h_ref",
    ]
    rows = []
    for seq_index, seq in enumerate(("SEQ-A", "SEQ-B", "SEQ-C")):
        for row_index in range(4):
            gap = row_index + 1
            offset = held_out_offset if seq == "SEQ-C" else 0.0
            rows.append(
                {
                    "seq": seq,
                    "lost_id": str(10 * seq_index + row_index),
                    "cand_id": str(100 + 10 * seq_index + row_index),
                    "gt_match": "1",
                    "gt_valid": "1",
                    "gap": str(gap),
                    "lost_last_frame": "10",
                    "cand_first_frame": str(10 + gap),
                    "lost_foot_x": "0",
                    "lost_foot_y": "0",
                    "cand_foot_x": str(gap * (1.0 + 0.1 * row_index) + offset),
                    "cand_foot_y": str(gap * (-0.5 + 0.07 * seq_index) - offset),
                    "h_ref": "10",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_held_out_rows_do_not_enter_fit_hash_or_training_nll(tmp_path):
    runner = _load_runner()
    original = tmp_path / "original.csv"
    mutated_held_out = tmp_path / "mutated.csv"
    _write_fold_pairs(original)
    _write_fold_pairs(mutated_held_out, held_out_offset=1000.0)

    params_a, selection_a = runner.build_fold_artifacts(original, "SEQ-C")
    params_b, selection_b = runner.build_fold_artifacts(mutated_held_out, "SEQ-C")

    assert selection_a["fit_row_key_sha256"] == selection_b["fit_row_key_sha256"]
    assert selection_a["training_nll_by_model"] == selection_b["training_nll_by_model"]
    assert all(artifact["held_out_sequence"] == "SEQ-C" for artifact in params_a)
    assert all("SEQ-C" not in artifact["train_sequences"] for artifact in params_a)
    assert [item["training_total_nll"] for item in params_a] == [
        item["training_total_nll"] for item in params_b
    ]


def test_fold_parameter_and_selection_artifacts_are_deterministic(tmp_path):
    runner = _load_runner()
    pairs = tmp_path / "pairs.csv"
    _write_fold_pairs(pairs)

    first = runner.build_fold_artifacts(pairs, "SEQ-B")
    second = runner.build_fold_artifacts(pairs, "SEQ-B")

    assert first == second
    parameters, selection = first
    assert len(parameters) == 4
    assert set(selection["training_nll_by_model"]) == set(runner.MODEL_ORDER)
    assert selection["selected_model_id"] in runner.MODEL_ORDER
