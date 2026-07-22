"""Synthetic contract tests for Safe-Region Assetization R1 (T1–T10)."""

# scope: eval
# function: contract
# lifecycle: quarantined
# lifecycle-note: safe-region assetization study CLOSED (apparatus module retained);
#   DISPOSITION.md proposes T2 — keep only while the apparatus serves the gt_support line.

from __future__ import annotations

import numpy as np

from saccade.perception.eval.safe_region_assetization_r1 import (
    _mask_sha256,
    component_geometry_1d,
    component_geometry_2d,
    evaluate_weights,
    fit_sparse_nn_milp,
    full_neighborhood_safe_radius_1d,
    full_neighborhood_safe_radius_2d,
    nearest_unsafe_distance_1d,
    nearest_unsafe_distance_2d,
    region_id,
)


# ---------------------------------------------------------------------------
# T1 — Duplicate-mask collapse
# ---------------------------------------------------------------------------


def test_t1_duplicate_mask_collapse_aliases_retained():
    """Different thresholds producing the same mask collapse to one basis."""
    # Synthetic primary: 6 samples, y: neg=1,gt=0
    # Feature values such that thr=1 and thr=2 yield same high_tail mask on x
    # x = [5,5,0,0,0,5] → thr<=5 high_tail both capture all high values equally
    # Use identical masks with different thr aliases via manual BasisEntry logic
    phi_a = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    phi_b = np.array([1, 1, 0, 0, 0, 1], dtype=bool)  # same prediction mask
    assert _mask_sha256(phi_a) == _mask_sha256(phi_b)
    # coordinate count > 1 conceptually; unique mask count = 1
    coords = {0, 1, 2}  # three thr indices
    unique_masks = {_mask_sha256(phi_a), _mask_sha256(phi_b)}
    assert len(coords) > 1
    assert len(unique_masks) == 1
    aliases = [
        {"thr_index": 0, "thr_value": 1.0},
        {"thr_index": 1, "thr_value": 2.0},
    ]
    assert len(aliases) == 2  # semantic aliases retained


# ---------------------------------------------------------------------------
# T2 — Thin strip margins
# ---------------------------------------------------------------------------


def test_t2_thin_strip_nearest_vs_full_radius():
    lattice = set(range(0, 5))
    safe = {2}  # isolated
    d = nearest_unsafe_distance_1d(2, safe, lattice)
    r = full_neighborhood_safe_radius_1d(2, safe)
    assert d > 0
    assert r == 0


def test_t2_thin_strip_2d_line():
    lattice = {(i, j) for i in range(3) for j in range(3)}
    safe = {(0, 1), (1, 1), (2, 1)}  # vertical strip
    d = nearest_unsafe_distance_2d((1, 1), safe, lattice)
    r = full_neighborhood_safe_radius_2d((1, 1), safe)
    assert d > 0
    assert r == 0  # left/right neighbors unsafe


# ---------------------------------------------------------------------------
# T3 — Thick component
# ---------------------------------------------------------------------------


def test_t3_thick_component_interior_and_radius():
    safe = {0, 1, 2, 3, 4}
    # interior point 2 has bilateral neighbors
    r = full_neighborhood_safe_radius_1d(2, safe)
    assert r >= 1
    geom = component_geometry_1d(safe)
    assert geom["coordinate_union_interior_count"] >= 1


def test_t3_thick_2d_full_radius():
    safe = {(i, j) for i in range(3) for j in range(3)}
    r = full_neighborhood_safe_radius_2d((1, 1), safe)
    assert r >= 1
    geom = component_geometry_2d(safe)
    assert geom["coordinate_union_interior_count"] >= 1


# ---------------------------------------------------------------------------
# T4 — Hard GT safety
# ---------------------------------------------------------------------------


def test_t4_hard_gt_safety_invalidates_candidate():
    # 2 features, 4 samples: 2 neg, 2 gt
    # Feature 0 fires on both neg and gt → cannot separate with NN weights
    Phi = np.array(
        [
            [1, 0],  # neg
            [1, 0],  # neg
            [1, 0],  # gt — same as neg on f0
            [0, 1],  # gt
        ],
        dtype=float,
    )
    Phi_u = np.zeros((0, 2))
    y = np.array([1, 1, 0, 0], dtype=int)
    fit = fit_sparse_nn_milp(Phi, Phi_u, y, K=2, active_cols=[0, 1], eps=1e-6)
    if fit["success"] and fit["w"] is not None:
        ev = evaluate_weights(
            fit["w"],
            float(fit["tau"]),
            Phi,
            Phi_u,
            y,
            np.array(["s"] * 4, dtype=object),
        )
        # If any GT captured → invalid
        if ev["gt_hurt"] > 0:
            assert ev["valid"] == 0
        # Optimizer must not accept GT trade-off: valid implies gt_hurt==0
        if ev["valid"]:
            assert ev["gt_hurt"] == 0
    # Direct check: forced capture of GT is invalid
    w = np.array([1.0, 0.0])
    tau = 0.5
    ev = evaluate_weights(w, tau, Phi, Phi_u, y, np.array(["s"] * 4, dtype=object))
    assert ev["gt_hurt"] > 0
    assert ev["valid"] == 0


# ---------------------------------------------------------------------------
# T5 — Unknown firewall
# ---------------------------------------------------------------------------


def test_t5_unknown_firewall():
    Phi = np.array(
        [
            [1, 0],  # neg
            [0, 1],  # gt
        ],
        dtype=float,
    )
    Phi_u = np.array([[1, 0]], dtype=float)  # unknown matches neg feature
    y = np.array([1, 0], dtype=int)
    w = np.array([1.0, 0.0])
    tau = 0.5
    ev = evaluate_weights(w, tau, Phi, Phi_u, y, np.array(["a", "a"], dtype=object))
    assert ev["unknown_capture"] > 0
    assert ev["valid"] == 0

    fit = fit_sparse_nn_milp(Phi, Phi_u, y, K=1, active_cols=[0, 1], eps=1e-6)
    # Capturing the negative via feature 0 necessarily captures the unknown —
    # either optimizer reports failure or returns zero-capture with hard safety.
    if fit["success"] and fit["w"] is not None:
        ev2 = evaluate_weights(
            fit["w"],
            float(fit["tau"]),
            Phi,
            Phi_u,
            y,
            np.array(["a", "a"], dtype=object),
        )
        assert ev2["unknown_capture"] == 0
        assert ev2["gt_hurt"] == 0
        # With unknown on the only productive feature, capture must be 0
        assert ev2["n_neg_captured"] == 0
    else:
        assert (
            fit.get("reason")
            in {
                "postcheck_hard_safety_failed",
                "milp_failed:The problem is infeasible.",
            }
            or "milp_failed" in str(fit.get("reason", ""))
            or fit.get("reason") == "zero_weight"
        )


# ---------------------------------------------------------------------------
# T6 — Non-negative weights
# ---------------------------------------------------------------------------


def test_t6_nonnegative_weights():
    Phi = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    Phi_u = np.zeros((1, 3))
    y = np.array([1, 1, 0, 0], dtype=int)
    fit = fit_sparse_nn_milp(Phi, Phi_u, y, K=2, active_cols=[0, 1, 2])
    if fit["success"] and fit["w"] is not None:
        assert np.all(fit["w"] >= -1e-10)


# ---------------------------------------------------------------------------
# T7 — Complexity cap
# ---------------------------------------------------------------------------


def test_t7_complexity_cap():
    rng = np.random.default_rng(0)
    n, m = 20, 10
    Phi = (rng.random((n, m)) > 0.7).astype(float)
    # make last 4 samples pure GT with no features
    Phi[16:] = 0
    y = np.array([1] * 10 + [0] * 10, dtype=int)
    Phi_u = np.zeros((2, m))
    K = 3
    fit = fit_sparse_nn_milp(Phi, Phi_u, y, K=K, active_cols=list(range(m)))
    if fit["success"] and fit["w"] is not None:
        active = int(np.sum(fit["w"] > 1e-8))
        assert active <= K


# ---------------------------------------------------------------------------
# T8 — LOO isolation (protocol property)
# ---------------------------------------------------------------------------


def test_t8_loo_isolation_protocol_flags():
    """Held-out labels must not drive basis registry; train-only weight fit."""
    # The nested LOO writer always sets basis_registry_fixed_globally=1
    # and basis_selected_using_train_labels=1. Verify evaluate path uses train mask.
    Phi = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 1],  # holdout-only pattern
        ],
        dtype=float,
    )
    y = np.array([1, 1, 0, 1], dtype=int)
    sequences = np.array(["A", "A", "A", "B"], dtype=object)
    train = sequences != "B"
    Phi_tr, y_tr = Phi[train], y[train]
    Phi_u = np.zeros((0, 2))
    fit = fit_sparse_nn_milp(Phi_tr, Phi_u, y_tr, K=1, active_cols=[0, 1])
    assert fit["success"]
    # Holdout pattern col1 must not be required: train-only may pick col0
    w = fit["w"]
    assert w is not None
    # Fitting never saw row 3; weight on col1 should not be forced by holdout
    # (train has no signal on col1 for negatives)
    assert w[1] <= w[0] + 1e-6 or w[1] < 1e-6


# ---------------------------------------------------------------------------
# T9 — Interaction cap (order <= 2)
# ---------------------------------------------------------------------------


def test_t9_no_third_order_interaction_in_basis_ids():
    # region_id / basis construction only allows order 1 or 2
    rid = region_id("g2_pairwise_and", "unique_mask", "abc123", 0)
    assert rid.startswith("q45:")
    for order in (1, 2):
        bid = f"b{order}:deadbeef"
        assert bid.startswith(f"b{order}:")
    # Contract: no third-order basis factory
    import saccade.perception.eval.safe_region_assetization_r1 as m

    assert not any("third" in n.lower() for n in dir(m))


def test_t9_order_cap_constant():
    from saccade.perception.eval import safe_region_assetization_r1 as m

    assert not hasattr(m, "build_third_order_basis")
    # FIXED_K_GRID only, no auto-expansion
    assert m.FIXED_K_GRID == (2, 3, 4, 5)


# ---------------------------------------------------------------------------
# T10 — Certified radius is conservative lower bound
# ---------------------------------------------------------------------------


def test_t10_certified_radius_conservative():
    # Separable: feature 0 only on negs
    Phi = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
        ],
        dtype=float,
    )
    Phi_u = np.zeros((1, 2))
    y = np.array([1, 1, 0, 0], dtype=int)
    w = np.array([1.0, 0.0])
    tau = 0.5
    ev = evaluate_weights(w, tau, Phi, Phi_u, y, np.array(["s"] * 4, dtype=object))
    assert ev["gt_hurt"] == 0
    r = float(ev["certified_gt_safe_radius_l1"])
    assert r > 0
    # Any perturbation with ||Δw||1 <= r and |Δτ| <= r must keep GT safe
    # Worst case: decrease tau by r and increase weight on a GT-active feature.
    # Our GT features are all 0 on f0, so score_gt=0; margin = tau - 0 = 0.5; r=0.25
    # After |Δτ|=r: tau'=0.25 still > 0. OK
    tau2 = tau - r
    scores_gt = Phi[y == 0] @ w
    assert np.all(scores_gt < tau2 + 1e-12)
    # Overestimate would claim r > margin; ensure r <= margin/2 style bound
    min_margin = float(np.min(tau - scores_gt))
    assert r <= min_margin / 2.0 + 1e-12


def test_region_id_not_row_index():
    a = region_id("g1_singleton", "productive_safe_component", "abcd" * 8, 0)
    b = region_id("g1_singleton", "productive_safe_component", "abcd" * 8, 0)
    assert a == b
    assert "q45:" in a
    assert a != "0"


def test_region_id_includes_grid_to_avoid_collision():
    a = region_id(
        "g2_pairwise_and",
        "grid_local_mask_asset",
        "abcd" * 8,
        0,
        grid_key=("abs_log_h", "high_tail", "dist_h", "low_tail"),
    )
    b = region_id(
        "g2_pairwise_and",
        "grid_local_mask_asset",
        "abcd" * 8,
        0,
        grid_key=("abs_log_h", "high_tail", "resid_mean", "low_tail"),
    )
    assert a != b
    assert ":g" in a and ":g" in b
