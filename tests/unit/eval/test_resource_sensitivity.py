"""The resource report must fail closed on routing and residency failures."""

# scope: eval
# function: contract
# lifecycle: active

import json
from pathlib import Path

import pytest

from scripts.benchmarks.resource_sensitivity.summarize import (
    distribution,
    summarize_point,
    telemetry_summary,
)


def write_point(path: Path, *, blocks=0, double=True):
    data = {
        "error": None,
        "blocks_requested": blocks,
        "completions": {"MOT17-04-SDP": [[51, 1.0], [52, 1.01], [53, 1.03]]},
        "double_buffer_enabled": {"MOT17-04-SDP": double},
        "clock_anchor": {"epoch": 0, "monotonic": 0},
        "blocker_info": {"max_blocks_per_sm": 1},
        "pulses": [
            {
                "host_begin": 1.0,
                "host_end": 1.03,
                "blocks": [[i, 0, 30_000_000] for i in range(blocks)],
            }
        ],
    }
    profile = {"frames": 3, "samples_ms": [50, 60, 70], "throughput_seconds": 0.04}
    (path / "resource_point.json").write_text(json.dumps(data))
    (path / "_latency_profile_MOT17-04-SDP.json").write_text(json.dumps(profile))
    (path / "MOT17-04-SDP.txt").write_text("51,1,0,0,1,1,1,-1,-1,-1\n")
    return data


def test_latency_and_completion_period_remain_distinct(tmp_path):
    write_point(tmp_path)
    row = summarize_point(tmp_path, "double", deadline=60)
    assert row["fps"] == 75  # Completed frames / production timing interval.
    assert row["latency"]["p50_ms"] == 60
    assert row["period"]["p50_ms"] == pytest.approx(15)
    assert row["period"]["n"] == 2
    assert row["deadline_miss_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["double_buffer_enabled"].update({"MOT17-04-SDP": False}), "route"),
        (lambda d: d["completions"]["MOT17-04-SDP"].append([53, 1.04]), "duplicate"),
        (lambda d: d["completions"]["MOT17-04-SDP"].pop(), "count mismatch"),
        (lambda d: d["blocker_info"].update(max_blocks_per_sm=2), "occupancy"),
        (
            lambda d: d["pulses"][0]["blocks"].__setitem__(1, [0, 0, 30_000_000]),
            "distinct SMs",
        ),
        (
            lambda d: d["pulses"][0]["blocks"].__setitem__(
                1, [1, 30_000_000, 60_000_000]
            ),
            "simultaneously",
        ),
        (lambda d: d["pulses"][0].update(host_end=1.01), "less than 90%"),
        (lambda d: d.update(error="kernel failure"), "failed point"),
    ],
)
def test_rejects_invalid_evidence(tmp_path, mutation, message):
    data = write_point(tmp_path, blocks=2)
    mutation(data)
    (tmp_path / "resource_point.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        summarize_point(tmp_path, "double")


def test_rejects_wrong_sweep_level(tmp_path):
    write_point(tmp_path)
    with pytest.raises(ValueError, match="pressure level"):
        summarize_point(tmp_path, "double", expected_blocks=8)


def test_missing_outputs_not_reported_as_quality_equal(tmp_path):
    write_point(tmp_path)
    (tmp_path / "MOT17-04-SDP.txt").unlink()
    with pytest.raises(ValueError, match="MOT outputs"):
        summarize_point(tmp_path, "double")


def test_residency_is_a_proxy_not_available_sm_count(tmp_path):
    write_point(tmp_path, blocks=2)
    row = summarize_point(tmp_path, "double")
    assert row["pressure"]["mean_resident_blocks_within_selected_pulses"] == 2
    assert "available_sm_count" not in row["pressure"]


def test_period_excludes_sequence_setup_gap(tmp_path):
    data = write_point(tmp_path)
    data["completions"]["MOT17-02-SDP"] = [[51, 100], [52, 100.01], [53, 100.03]]
    data["double_buffer_enabled"]["MOT17-02-SDP"] = True
    (tmp_path / "resource_point.json").write_text(json.dumps(data))
    (tmp_path / "_latency_profile_MOT17-02-SDP.json").write_text(
        (tmp_path / "_latency_profile_MOT17-04-SDP.json").read_text()
    )
    (tmp_path / "MOT17-02-SDP.txt").write_text("example\n")
    row = summarize_point(tmp_path, "double")
    assert row["period"]["n"] == 4
    assert row["period"]["mean_ms"] == pytest.approx(15)


@pytest.mark.parametrize("samples", [[], [float("nan")], [-1], [float("inf")]])
def test_bad_timings_rejected(samples):
    with pytest.raises(ValueError):
        distribution(samples)


def test_telemetry_requires_same_run_window(tmp_path):
    write_point(tmp_path)
    point = summarize_point(tmp_path, "double")
    csv = tmp_path / "telemetry.csv"
    csv.write_text("timestamp, power.draw [W]\n2026/09/15 00:00:00.000, 100 W\n")
    with pytest.raises(ValueError, match="no telemetry samples"):
        telemetry_summary(csv, point)


def test_missing_prefix_or_tail_rejected_even_if_counts_match(tmp_path):
    data = write_point(tmp_path)
    data["completions"]["MOT17-04-SDP"] = [[52, 1.0], [53, 1.01], [54, 1.03]]
    (tmp_path / "resource_point.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frame bounds"):
        summarize_point(tmp_path, "double", expected_ranges={"MOT17-04-SDP": (51, 54)})


def test_missing_whole_sequence_rejected(tmp_path):
    write_point(tmp_path)
    with pytest.raises(ValueError, match="requested sequences"):
        summarize_point(
            tmp_path,
            "double",
            expected_ranges={"MOT17-04-SDP": (51, 54), "MOT17-02-SDP": (51, 54)},
        )
