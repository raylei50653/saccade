"""The frozen SM-scaling benchmark applies its pre-declared rules and refuses scope drift."""

# scope: eval
# function: contract
# lifecycle: active

import json

import pytest

from scripts.benchmarks.resource_partition import frozen_benchmark as fb


def row(rep, level, mode, *, fps, p99, sm=None, validated=True):
    green = level != "full"
    actual = sm if sm is not None else (level if green else 46)
    return {
        "name": f"r{rep}_sm{level}_{mode}",
        "rep": rep,
        "level": level,
        "mode": mode,
        "fps": fps,
        "frames": 250,
        "latency": {"p50_ms": p99 * 0.6, "p95_ms": p99 * 0.9, "p99_ms": p99},
        "period": {
            "p50_ms": 1000 / fps,
            "p95_ms": 1200 / fps,
            "p99_ms": 1500 / fps,
            "std_ms": 0.3,
        },
        "partition_kind": "green_context" if green else "primary_full_device_baseline",
        "actual_sm_count": actual,
        "device_sm_count": 46,
        "output_sha256": {"MOT17-04-SDP.txt": "abc"},
        "telemetry": {
            "samples": 10,
            "values": {
                "power.draw [W]": {"mean": 100.0},
                "clocks.current.sm [MHz]": {"mean": 2400.0},
                "utilization.gpu [%]": {"mean": 80.0},
            },
        },
        "routing_evidence": {
            "green_context": {"min_partition": 8, "alignment": 8} if green else None
        },
        "true_partition_validated": green and validated,
        "curve_eligible": True,
    }


def synthetic_sweep(tmp_path, gains, *, reps=2, serial=None, head="h1"):
    """gains: {level: db_gain}; serial FPS scales with the SM count unless given."""
    rows, pairs = [], []
    for rep in range(reps):
        for level, gain in gains.items():
            sm = 46 if level == "full" else level
            s = (serial or {}).get(level, 5.0 * sm)
            d = s * gain
            rows.append(row(rep, level, "serial", fps=s, p99=200 / sm))
            rows.append(row(rep, level, "double", fps=d, p99=350 / sm))
            pairs.append(
                {
                    "rep": rep,
                    "level": level,
                    "actual_sm_count": sm,
                    "partition_kind": rows[-1]["partition_kind"],
                    "serial_fps": s,
                    "double_fps": d,
                    "db_gain": gain,
                    "true_partition_validated": level != "full",
                    "curve_eligible": True,
                }
            )
    summary = {
        "schema": "saccade-partition-summary-v1",
        "rows": rows,
        "pairs": pairs,
        "points_not_validated": [],
    }
    sweep = {
        "kind": "green_context_true_partition",
        "arguments": {
            "levels": list(gains),
            "repeats": reps,
            "preset": "mamba_whole_graph_m",
            "sequences": "MOT17-04-SDP",
            "max_frames": 300,
            "seed": 419,
        },
        "head": head,
        "source_sha256": {"sweep.py": head},
        "library_sha256": {},
        "toolchain": {"torch": "2.11.0"},
        "gpu": "Product Name : NVIDIA Test GPU\nKMD Version : 616.92\n",
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "sweep.json").write_text(json.dumps(sweep))
    return tmp_path


SMOOTH = {
    "full": 1.567,
    46: 1.584,
    40: 1.578,
    32: 1.516,
    24: 1.401,
    16: 1.312,
    8: 1.170,
}


def test_knee_rule_classifies_smooth_decline_graceful_and_knee():
    full = 1.5
    smooth = [(8, 1.17), (16, 1.31), (24, 1.40), (32, 1.52), (40, 1.58), (46, 1.58)]
    assert fb.knee_verdict(smooth, 1.567)["verdict"] == "smooth_decline"
    graceful = [(8, 1.42), (16, 1.45), (24, 1.48), (32, 1.5), (40, 1.5), (46, 1.5)]
    assert fb.knee_verdict(graceful, full)["verdict"] == "graceful_scaling"
    # collapse between 24 and 16 SMs: the 16→24 segment loses 0.0925/SM against
    # ≤ 0.0125/SM in every higher segment, and only 10% of the gain is left at 16
    knee = [(8, 1.02), (16, 1.05), (24, 1.42), (32, 1.47), (40, 1.5), (46, 1.5)]
    result = fb.knee_verdict(knee, full)
    assert result["verdict"] == "knee" and result["knee_sm"] == 16
    assert result["no_overlap_budgets"] == [8, 16]
    # a linear ramp that also ends near 1.0 is a decline, not a knee
    ramp = [(8, 1.02), (16, 1.18), (24, 1.35), (32, 1.45), (40, 1.5), (46, 1.5)]
    assert fb.knee_verdict(ramp, full)["verdict"] == "smooth_decline"
    # a steep segment whose low end still keeps half the gain is not a knee
    steep_high = [(8, 1.30), (16, 1.46), (24, 1.48), (32, 1.5), (40, 1.5), (46, 1.5)]
    assert fb.knee_verdict(steep_high, full)["verdict"] == "smooth_decline"
    assert fb.knee_verdict(smooth[:2], full)["verdict"] == "insufficient_budgets"
    assert fb.knee_verdict(smooth, 1.0)["verdict"] == "no_full_device_gain"


def test_build_aggregates_reps_and_reports_frontier(tmp_path):
    record = fb.build(synthetic_sweep(tmp_path / "s", SMOOTH, reps=3))
    assert record["schema"] == fb.SCHEMA
    assert record["knee"]["verdict"] == "smooth_decline"
    assert record["knee"]["verdict_agrees_across_reps"] is True
    budgets = {b["label"]: b for b in record["budgets"]}
    assert budgets["8 (green)"]["reps"] == [0, 1, 2]
    assert budgets["8 (green)"]["serial"]["fps"]["mean"] == pytest.approx(40.0)
    assert budgets["8 (green)"]["db_gain"]["per_rep"] == [1.17] * 3
    assert budgets["46 (primary)"]["true_partition_validated"] is False
    t1 = next(t for t in record["frontier"] if t["name"] == "T1")
    # serial 5 FPS/SM never reaches 300; double reaches 300 first at 40 SMs (315.6)
    assert t1["serial"]["min_verified_sm"] is None
    assert t1["double"]["min_verified_sm"] == 40
    assert t1["double"]["full_device_meets"] is True
    assert record["scope"]["device_name"] == "NVIDIA Test GPU"
    assert record["scope"]["requested_levels"] == sorted(str(k) for k in SMOOTH)
    assert record["all_outputs_byte_identical"] is True


def test_build_fails_closed_on_unvalidated_points_or_too_few_budgets(tmp_path):
    root = synthetic_sweep(tmp_path / "bad", SMOOTH)
    summary = json.loads((root / "summary.json").read_text())
    summary["points_not_validated"] = ["r0_sm8_serial"]
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="unvalidated"):
        fb.build(root)
    few = synthetic_sweep(tmp_path / "few", {"full": 1.5, 16: 1.3, 32: 1.4, 46: 1.5})
    with pytest.raises(ValueError, match="at least four"):
        fb.build(few)


def test_compare_reports_ratios_frontier_movement_and_drift_note(tmp_path):
    reference = fb.build(synthetic_sweep(tmp_path / "ref", SMOOTH, head="before"))
    faster = {k: v for k, v in SMOOTH.items()}
    serial = {level: 6.0 * (46 if level == "full" else level) for level in SMOOTH}
    candidate = fb.build(
        synthetic_sweep(tmp_path / "cand", faster, serial=serial, head="after")
    )
    result = fb.compare(reference, candidate)
    assert result["comparable"] is True
    assert "head" in result["recorded_differences"]
    b8 = next(b for b in result["budgets"] if b["actual_sm_count"] == 8)
    assert b8["serial"]["fps"]["ratio_candidate_over_reference"] == pytest.approx(1.2)
    assert b8["db_gain"]["delta"] == pytest.approx(0.0)
    t1 = next(t for t in result["frontier"] if t["name"] == "T1")
    # double 6 FPS/SM × 1.516 = 291 at 32 SMs, 379 at 40: frontier unchanged at 40;
    # serial now reaches 300 nowhere on the grid (276 at 46) so still none
    assert t1["double"]["delta_sm"] == 0
    assert result["host_drift"] is None and "control" in result["host_drift_note"]
    control = fb.build(synthetic_sweep(tmp_path / "ctl", SMOOTH, head="before"))
    with_control = fb.compare(reference, candidate, control)
    assert with_control["host_drift"]["control_head_equals_reference"] is True
    assert "candidate_over_control" in with_control["host_drift"]
    text = fb.comparison_markdown(with_control)
    assert "Host drift" in text and "Frontier movement" in text


def test_compare_refuses_scope_drift(tmp_path):
    reference = fb.build(synthetic_sweep(tmp_path / "ref", SMOOTH))
    other = fb.build(synthetic_sweep(tmp_path / "other", SMOOTH))
    other["scope"]["max_frames"] = 600
    result = fb.compare(reference, other)
    assert result["comparable"] is False
    assert [m["field"] for m in result["scope_mismatches"]] == ["max_frames"]
    assert "budgets" not in result
    assert "Not comparable" in fb.comparison_markdown(result)
    criteria_drift = fb.build(synthetic_sweep(tmp_path / "crit", SMOOTH))
    criteria_drift["criteria"]["knee"]["slope_factor"] = 3.0
    assert fb.compare(reference, criteria_drift)["comparable"] is False


def test_markdown_tables_are_generated_from_the_record(tmp_path):
    record = fb.build(synthetic_sweep(tmp_path / "s", SMOOTH))
    text = fb.build_markdown(record)
    assert "smooth_decline" in text
    assert "| 8 (green) | 2 |" in text
    assert "Service-level frontier" in text
