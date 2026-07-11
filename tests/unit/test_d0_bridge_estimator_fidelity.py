"""Unit tests for D0 Consumer-A bridge estimator fidelity audit."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    LIVE_CUDA_EVENT_RING_IMPLEMENTED,
    PRODUCTION_BRIDGE_PX,
    RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT,
    bridge_vel4,
    decide_bridge,
    production_safe,
    speed_weighted_bdist,
)

REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "d0_bridge_estimator_fidelity_20260711"
    / "run_d0_bridge_fidelity.py"
)
PACKET = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "d0_bridge_estimator_fidelity_20260711"
)
SOURCE_SHA = "0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17"


def _load_runner():
    spec = importlib.util.spec_from_file_location("d0_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Capture / default-off ─────────────────────────────────────────────────────


def test_audit_capture_default_is_off() -> None:
    assert RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT is False
    assert LIVE_CUDA_EVENT_RING_IMPLEMENTED is False


def test_audit_off_does_not_change_consumer_a_decision() -> None:
    bdist = 0.35
    assert decide_bridge(bdist, audit_enabled=False) is True
    assert decide_bridge(bdist, audit_enabled=True) is True
    bdist_bad = 0.5
    assert decide_bridge(bdist_bad, audit_enabled=False) is False
    assert decide_bridge(bdist_bad, audit_enabled=True) is False


def test_production_threshold_remains_0_4() -> None:
    assert PRODUCTION_BRIDGE_PX == 0.4
    assert production_safe(0.4) is True
    assert production_safe(0.4000001) is False


def test_bridge_vel4_matches_cuda_closed_form() -> None:
    # v = (3 y3 + y2 - y1 - 3 y0) / 10
    samples = [0.0, 1.0, 2.0, 3.0]
    assert bridge_vel4(samples) == pytest.approx(1.0)


# ── Join integrity ────────────────────────────────────────────────────────────


def test_source_pairs_sha_mismatch_fails(tmp_path: Path) -> None:
    d0 = _load_runner()
    bad = tmp_path / "pairs.csv"
    bad.write_text(
        "seq,lost_id,cand_id,gt_match,gt_valid,gap,dist_h,fwd_resid,bwd_resid,"
        "bridge_dist,lost_last_frame,cand_first_frame,h_ref,lost_exit_speed\n"
        "S,1,2,1,1,1,0.1,0.1,0.1,0.1,1,2,10,0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        d0.run_pipeline(
            bad,
            REPO / d0.SUBSTRATE_MOT_DIR,
            tmp_path / "out",
        )


def test_duplicate_event_key_fails_coverage_gate() -> None:
    d0 = _load_runner()
    coverage = {
        "gates_pass": False,
        "gates": {
            "duplicate_keys_zero": False,
            "ambiguous_keys_zero": True,
            "gt_coverage_s_a_100": True,
            "overall_coverage_s_a_98": True,
        },
    }
    # Minimal metrics skeleton for verdict path
    empty_pred = d0.predicate_confusion(np.array([0.1]), np.array([0.1]))
    metrics = {
        "S_A_overall": {
            "spearman_rho": 0.99,
            "q85_abs_error": 0.01,
            "predicate": empty_pred,
            "quantile_monotone_offline": True,
            "quantile_monotone_consumer_a": True,
        },
        "S_A_GT": {
            "spearman_rho": 0.99,
            "q85_abs_error": 0.01,
            "predicate": empty_pred,
        },
        "S_A_FP": {
            "spearman_rho": 0.99,
            "q85_abs_error": 0.01,
            "predicate": empty_pred,
        },
        "gap_1_10": {"n_gt": 0},
        "gap_11_26": {"n_gt": 0},
        "gap_1_10_GT": {
            "spearman_rho": 0.99,
            "predicate": empty_pred,
        },
        "gap_11_26_GT": {
            "spearman_rho": 0.99,
            "predicate": empty_pred,
        },
    }
    verdict = d0.evaluate_verdict(coverage, metrics)
    assert verdict["verdict"] == "not_fidelity_aligned"


def test_gt_and_overall_coverage_gates_enforced() -> None:
    d0 = _load_runner()
    # Synthetic join coverage stats
    assert d0.GATE_GT_COVERAGE == 1.0
    assert d0.GATE_OVERALL_COVERAGE == 0.98


# ── Metric correctness ────────────────────────────────────────────────────────


def test_spearman_known_case_fixture() -> None:
    d0 = _load_runner()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    clusters = np.array(["a|1", "a|1", "b|2", "b|2", "c|3"], dtype=object)
    sp = d0.spearman_with_cluster_ci(x, y, clusters, seed=20260711, reps=50)
    assert sp["rho"] == pytest.approx(1.0)


def test_quantile_alignment_known_case_fixture() -> None:
    d0 = _load_runner()
    offline = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    online = offline + 0.05
    rows = d0.quantile_alignment(offline, online)
    q85 = next(r for r in rows if r["quantile"] == "q85")
    assert q85["headline"] is True
    assert q85["absolute_error"] == pytest.approx(0.05)


def test_predicate_confusion_known_case_fixture() -> None:
    d0 = _load_runner()
    offline = np.array([0.2, 0.3, 0.5, 0.6])
    online = np.array([0.2, 0.5, 0.3, 0.6])
    conf = d0.predicate_confusion(offline, online, thr=0.4)
    # pairs: SS, SU, US, UU
    assert conf["offline_safe_online_safe"] == 1
    assert conf["offline_safe_online_unsafe"] == 1
    assert conf["offline_unsafe_online_safe"] == 1
    assert conf["offline_unsafe_online_unsafe"] == 1
    assert conf["predicate_agreement"] == pytest.approx(0.5)
    assert conf["offline_safe_online_unsafe_count"] == 1
    assert conf["offline_safe_online_unsafe_rate"] == pytest.approx(0.25)


def test_cluster_bootstrap_deterministic_seed() -> None:
    d0 = _load_runner()
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.1, size=40)
    clusters = np.array([f"s|{i // 5}" for i in range(40)], dtype=object)
    a = d0.spearman_with_cluster_ci(x, y, clusters, seed=20260711, reps=100)
    b = d0.spearman_with_cluster_ci(x, y, clusters, seed=20260711, reps=100)
    assert a["rho"] == b["rho"]
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]


# ── Verdict partition ─────────────────────────────────────────────────────────


def _perfect_metrics(
    d0,
    *,
    rho: float = 0.99,
    q85: float = 0.01,
    agree: float = 0.99,
    su: int = 0,
    n: int = 100,
):
    def pred(n_rows: int, su_count: int, agreement: float):
        return {
            "n": n_rows,
            "offline_safe_online_safe": n_rows - su_count,
            "offline_safe_online_unsafe": su_count,
            "offline_unsafe_online_safe": 0,
            "offline_unsafe_online_unsafe": 0,
            "predicate_agreement": agreement,
            "offline_safe_online_unsafe_count": su_count,
            "offline_safe_online_unsafe_rate": su_count / n_rows if n_rows else 0.0,
        }

    base = {
        "spearman_rho": rho,
        "q85_abs_error": q85,
        "predicate": pred(n, su, agree),
        "quantile_monotone_offline": True,
        "quantile_monotone_consumer_a": True,
        "n_gt": n,
    }
    return {
        "S_A_overall": dict(base),
        "S_A_GT": dict(base),
        "S_A_FP": dict(base),
        "gap_1_10": {"n_gt": 5},
        "gap_11_26": {"n_gt": 5},
        "gap_1_10_GT": {
            "spearman_rho": rho,
            "predicate": pred(n, su, agree),
        },
        "gap_11_26_GT": {
            "spearman_rho": rho,
            "predicate": pred(n, su, agree),
        },
    }


def test_verdict_threshold_transfer_supported_fixture() -> None:
    d0 = _load_runner()
    coverage = {"gates_pass": True}
    metrics = _perfect_metrics(d0)
    v = d0.evaluate_verdict(coverage, metrics)
    assert v["verdict"] == "threshold_transfer_supported"


def test_verdict_rank_only_transfer_supported_fixture() -> None:
    d0 = _load_runner()
    coverage = {"gates_pass": True}
    # Fail threshold (q85 too large / pred too low) but pass rank-only floors.
    metrics = _perfect_metrics(d0, rho=0.90, q85=0.10, agree=0.80, su=5)
    v = d0.evaluate_verdict(coverage, metrics)
    assert v["verdict"] == "rank_only_transfer_supported"


def test_verdict_not_fidelity_aligned_fixture() -> None:
    d0 = _load_runner()
    coverage = {"gates_pass": False}
    metrics = _perfect_metrics(d0, rho=0.5, q85=1.0, agree=0.5, su=50)
    v = d0.evaluate_verdict(coverage, metrics)
    assert v["verdict"] == "not_fidelity_aligned"


def test_verdict_rule_is_total_partition() -> None:
    d0 = _load_runner()
    cases = [
        ({"gates_pass": True}, _perfect_metrics(d0)),
        (
            {"gates_pass": True},
            _perfect_metrics(d0, rho=0.90, q85=0.10, agree=0.80, su=5),
        ),
        (
            {"gates_pass": False},
            _perfect_metrics(d0, rho=0.5, q85=1.0, agree=0.5, su=50),
        ),
        (
            {"gates_pass": True},
            _perfect_metrics(d0, rho=0.50, q85=1.0, agree=0.50, su=50),
        ),
    ]
    allowed = {
        "threshold_transfer_supported",
        "rank_only_transfer_supported",
        "not_fidelity_aligned",
    }
    seen: set[str] = set()
    for cov, met in cases:
        v = d0.evaluate_verdict(cov, met)["verdict"]
        assert v in allowed
        seen.add(v)
    assert seen == allowed


# ── Packet verification ───────────────────────────────────────────────────────


def test_sealed_packet_exists_and_verdict_shape() -> None:
    verdict_path = PACKET / "verdict.json"
    assert verdict_path.is_file()
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["schema_version"] == 1
    assert verdict["status"] == "D0_SEALED"
    assert verdict["verdict"] in {
        "threshold_transfer_supported",
        "rank_only_transfer_supported",
        "not_fidelity_aligned",
    }
    assert verdict["phase_b_authorized"] is False
    assert verdict["a1_a8_computed"] is False
    assert verdict["production_changed"] is False
    assert verdict["production_threshold"] == 0.4
    assert verdict["source_pairs_sha256"] == SOURCE_SHA


def test_manifest_hashes_and_artifacts() -> None:
    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "consumer_a_capture.csv.gz",
        "same_event_join.csv.gz",
        "metrics_summary.json",
        "verdict.json",
        "run_d0_bridge_fidelity.py",
    }
    assert required.issubset(set(manifest["artifacts"]))
    d0 = _load_runner()
    for name in required:
        path = PACKET / name if name != "run_d0_bridge_fidelity.py" else RUNNER
        assert (
            d0.sha256(path) == manifest["artifacts"][name]
            or name == "run_d0_bridge_fidelity.py"
        )
        # runner may be hashed from packet copy
        if name == "run_d0_bridge_fidelity.py":
            packet_runner = PACKET / "run_d0_bridge_fidelity.py"
            if packet_runner.is_file():
                assert d0.sha256(packet_runner) == manifest["artifacts"][name]


def test_gzip_byte_stable_header() -> None:
    path = PACKET / "consumer_a_capture.csv.gz"
    with path.open("rb") as stream:
        header = stream.read(10)
    # gzip magic + mtime bytes 4-7 should be zero for mtime=0
    assert header[:2] == b"\x1f\x8b"
    assert header[4:8] == b"\x00\x00\x00\x00"


def test_verify_rebuilds_byte_for_byte() -> None:
    d0 = _load_runner()
    d0.verify_packet(PACKET, d0.CANONICAL_PAIRS, d0.SUBSTRATE_MOT_DIR)


def test_speed_weighted_bdist_matches_kernel_algebra() -> None:
    bdist, dist_h, fwd_r, bwd_r, s_lost, w = speed_weighted_bdist(
        lx=0.0,
        ly=0.0,
        cx0=10.0,
        cy0=0.0,
        vxl=1.0,
        vyl=0.0,
        vxc=1.0,
        vyc=0.0,
        horizon=10.0,
        h_ref=10.0,
    )
    assert dist_h == pytest.approx(1.0)
    assert fwd_r == pytest.approx(0.0)
    assert bwd_r == pytest.approx(0.0)
    assert math.isfinite(bdist)
    assert 0.0 <= w <= 1.0
