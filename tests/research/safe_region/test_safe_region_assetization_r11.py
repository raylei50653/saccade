"""Synthetic tests for R1.1 transfer-failure attribution."""

# scope: eval
# function: behavior
# lifecycle: quarantined
# lifecycle-note: R1.1 attribution study CLOSED (apparatus module retained);
#   DISPOSITION.md proposes T2 — consolidate or delete with the assetization apparatus.

from __future__ import annotations


from saccade.perception.eval.safe_region_assetization_r11 import (
    _jaccard,
    assign_failure_taxonomy,
    math_isfinite,
)


def test_jaccard_basic():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, {"a"}) == 1.0
    assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9


def test_taxonomy_prefers_single_primary():
    # Synthetic model_results with strong role reversal + margin fail
    fake = [
        {
            "model_id": "L3_sparse_nn_with_and:K5",
            "summary": {
                "mean_pairwise_active_jaccard": 0.9,
                "mean_jaccard_global_vs_train_pure": 0.95,
                "role_reversal_events": 8,
                "total_hold_gt_hurt": 4,
                "n_folds_with_gt_hurt": 3,
                "n_folds": 7,
                "train_prod_basis_holdout_neg_retention": 0.8,
                "pooled_n_neg": 8,
            },
            "folds": [
                {"status": "ok", "train_sequence_dominance": 0.3},
            ]
            * 7,
            "margins": [
                {
                    "train_gt_safety_margin": 0.2,
                    "hold_gt_margin": -0.1,
                }
            ]
            * 3
            + [{"train_gt_safety_margin": 0.2, "hold_gt_margin": 0.1}] * 4,
            "basis_roles": [{"x": 1}] * 10,
        }
    ]
    tax = assign_failure_taxonomy(fake)
    assert tax["primary"] in {"F1", "F2", "F3", "F4", "F5"}
    assert len(tax["secondary"]) <= 2
    assert tax["primary"] not in tax["secondary"]


def test_f5_when_tiny_support():
    fake = [
        {
            "model_id": "L2_sparse_nn_singleton:K2",
            "summary": {
                "mean_pairwise_active_jaccard": 1.0,
                "mean_jaccard_global_vs_train_pure": 1.0,
                "role_reversal_events": 0,
                "total_hold_gt_hurt": 0,
                "n_folds_with_gt_hurt": 0,
                "n_folds": 7,
                "train_prod_basis_holdout_neg_retention": 1.0,
                "pooled_n_neg": 1,
            },
            "folds": [{"status": "ok", "train_sequence_dominance": 1.0}] * 7,
            "margins": [{"train_gt_safety_margin": 0.5, "hold_gt_margin": 0.5}] * 7,
            "basis_roles": [],
        }
    ]
    tax = assign_failure_taxonomy(fake)
    # tiny pool + full dominance → F4 or F5
    assert tax["primary"] in {"F4", "F5"}


def test_math_isfinite():
    assert math_isfinite(1.0)
    assert not math_isfinite(float("nan"))
    assert not math_isfinite(None)
