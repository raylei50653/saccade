"""Tests for constrained FP pruning / safe-reject metrics."""

# scope: eval
# function: behavior
# lifecycle: active

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


def test_prod_proxy_score_and_h_ratio() -> None:
    """Offline production-shaped score uses speed-weighted blend (kernel shape)."""
    import importlib.util
    from pathlib import Path

    path = Path("scripts/tools/audit_relink_safe_reject.py")
    spec = importlib.util.spec_from_file_location("audit_relink_safe_reject", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pool = {
        "gt_match": np.array([True, False]),
        "lost_exit_speed": np.array([0.0, 0.12]),  # w=0 and w=1
        "fwd_resid": np.array([1.0, 2.0]),
        "bwd_resid": np.array([1.0, 2.0]),
        "dist_h": np.array([0.5, 0.5]),
        "h_lost_raw": np.array([100.0, 50.0]),
        "h_cand_raw": np.array([100.0, 100.0]),
    }
    mod.ensure_prod_proxy_scores(pool)
    # w=0 → score = dist_h; w=1 → score = 0.5*(fwd+bwd)
    assert abs(pool["score_m_bridge"][0] - 0.5) < 1e-6
    assert abs(pool["score_m_bridge"][1] - 2.0) < 1e-6
    assert abs(pool["h_ratio_lost_over_cand"][0] - 1.0) < 1e-6
    assert abs(pool["h_ratio_lost_over_cand"][1] - 0.5) < 1e-6

    rules = {name: fn for name, _hint, fn, _n in mod.production_shaped_rules()}
    # second row: h_ratio 0.5 < 0.6 → m h-gate reject
    assert rules["prod_m_h_ratio_out_0.6_1.7"](pool).tolist() == [False, True]


def _load_audit_module():
    import importlib.util
    from pathlib import Path

    path = Path("scripts/tools/audit_relink_safe_reject.py")
    spec = importlib.util.spec_from_file_location("audit_relink_safe_reject", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exposure_pool(
    gt: list[bool], seq: list[str], lost: list[str] | None
) -> dict[str, np.ndarray]:
    pool: dict[str, np.ndarray] = {
        "gt_match": np.array(gt, dtype=bool),
        "seq": np.asarray(seq, dtype=object),
    }
    if lost is not None:
        pool["lost_id"] = np.asarray(lost, dtype=object)
    return pool


def test_exposure_summary_full_metadata_counts_clusters() -> None:
    mod = _load_audit_module()
    # 3 GT rows but only 2 lost tracks in seq A; 1 FP row ignored
    pool = _exposure_pool(
        gt=[True, True, True, False],
        seq=["A", "A", "A", "A"],
        lost=["1", "1", "2", "9"],
    )
    e = mod.exposure_summary(pool)
    assert e["n_gt_exposed"] == 3
    assert e["n_gt_rows_missing_lost_id"] == 0
    assert e["n_gt_exposed_clusters"] == 2
    assert e["per_seq_gt_exposed_clusters"] == {"A": 2}
    assert e["declared_trial_unit"] == "lost_track(seq,lost_id)"
    assert e["remaining_clustering"] == "sequence"


def test_exposure_summary_cross_seq_same_id_distinct_clusters() -> None:
    mod = _load_audit_module()
    # same lost_id "1" in two sequences must be two clusters
    pool = _exposure_pool(
        gt=[True, True],
        seq=["A", "B"],
        lost=["1", "1"],
    )
    e = mod.exposure_summary(pool)
    assert e["n_gt_exposed_clusters"] == 2
    assert e["per_seq_gt_exposed_clusters"] == {"A": 1, "B": 1}


def test_exposure_summary_all_missing_is_unknown() -> None:
    mod = _load_audit_module()
    pool = _exposure_pool(
        gt=[True, True, False],
        seq=["A", "A", "A"],
        lost=["", "  ", "9"],
    )
    e = mod.exposure_summary(pool)
    assert e["n_gt_rows_missing_lost_id"] == 2
    assert e["n_gt_exposed_clusters"] is None
    assert e["clustering"] == "unknown: pairs CSV lacks lost_id"


def test_exposure_summary_partial_missing_is_insufficient_not_collapsed() -> None:
    mod = _load_audit_module()
    # one GT row missing lost_id must NOT silently merge into a "" cluster
    pool = _exposure_pool(
        gt=[True, True, True],
        seq=["A", "A", "A"],
        lost=["1", "", "2"],
    )
    e = mod.exposure_summary(pool)
    assert e["n_gt_rows_missing_lost_id"] == 1
    assert e["n_gt_exposed_clusters"] is None
    assert e["clustering"].startswith("insufficient_metadata: 1/3")
    assert "per_seq_gt_exposed_clusters" not in e


def test_exposure_summary_missing_column_is_unknown() -> None:
    mod = _load_audit_module()
    pool = _exposure_pool(gt=[True, False], seq=["A", "A"], lost=None)
    e = mod.exposure_summary(pool)
    assert e["n_gt_exposed_clusters"] is None
    assert e["clustering"] == "unknown: pairs CSV lacks lost_id"
    assert "n_gt_rows_missing_lost_id" not in e


def test_exposure_summary_zero_gt_rows() -> None:
    mod = _load_audit_module()
    pool = _exposure_pool(gt=[False, False], seq=["A", "A"], lost=["9", ""])
    e = mod.exposure_summary(pool)
    assert e["n_gt_exposed"] == 0
    assert e["n_fp_exposed"] == 2
    # FP rows missing lost_id are irrelevant to the GT trial-unit claim
    assert e["n_gt_rows_missing_lost_id"] == 0
    assert e["n_gt_exposed_clusters"] == 0
    assert e["per_seq_gt_exposed_clusters"] == {}
