"""Fail closed on corrupt double-buffer mixed-workload evidence."""

# scope: eval
# function: contract
# lifecycle: active
import json

import numpy as np
import pytest

from scripts.benchmarks.resource_mixed.report import STAMP, digest, reference
from scripts.benchmarks.resource_mixed_db.report import derive


def fixture(tmp_path):
    stable = list(range(16))
    elastic = list(range(16, 32))
    probe = dict(direct_sm_ids=stable, graph_sm_ids=stable)
    output = tmp_path / "eval.txt"
    output.write_text("fixed MOT output\n")
    frames = [
        dict(
            frame=frame,
            start=1.0 + index * 0.005,
            cycle_finished=1.006 + index * 0.005,
            output=1.012 + index * 0.005,
            context_owned=True,
        )
        for index, frame in enumerate(range(51, 54))
    ]
    transitions = [
        dict(
            scheduled_frame=frame,
            request=0.99 + index * 0.005,
            drained=0.9901 + index * 0.005,
            admitted=0.9902 + index * 0.005,
        )
        for index, frame in enumerate(range(51, 54))
    ]
    windows = [
        dict(
            release_after_frame=50 + index,
            release_requested=0.991 + index * 0.005,
            released=0.9911 + index * 0.005,
            reclaim_before_scheduled_frame=52 + index,
            request=transitions[index + 1]["request"],
            drained=transitions[index + 1]["drained"],
            admitted=transitions[index + 1]["admitted"],
            within_stable_service=True,
        )
        for index in range(2)
    ]
    windows.append(
        dict(
            release_after_frame=53,
            release_requested=1.021,
            released=1.0211,
            reclaim_before_scheduled_frame=None,
            request=1.022,
            drained=1.0221,
            admitted=None,
            within_stable_service=False,
        )
    )
    point = dict(
        schema="saccade-mixed-db-point-v2",
        arguments=dict(
            policy="dynamic",
            window=1,
            iterations=2048,
            frames=53,
            deadline_ms=20.0,
            bursts=1,
            burst_period_ms=50.0,
            units=3,
            audit=False,
        ),
        frames=frames,
        transitions=transitions,
        windows=windows,
        route=dict(enabled=True, stream_handle=10, owned_class=True, warmup_frames=50),
        origin=1.0,
        native_origin=1_000_000_000,
        native_anchor_after=1.0,
        owner=dict(
            actual_sm_count=16,
            streams=[dict(handle=10, owned=True)],
            main_thread_context_still_owned_at_finalize=True,
            thread_context_switch_errors=[],
            probe=dict(before=probe, after=probe),
        ),
        elastic_pool=dict(
            actual_sms=16,
            probe=dict(direct_sm_ids=elastic, graph_sm_ids=elastic),
            lanes=[
                dict(stream=20, context=2, borrowed=False),
                dict(stream=21, context=2, borrowed=False),
                dict(stream=22, context=1, borrowed=True),
            ],
            borrowed_lane=2,
        ),
        reference=reference(2048),
        error=None,
        output_sha256={"eval.txt": digest(output)},
    )
    np.array(
        [
            [0, 0, 1_000_000_000, 1_005_000_000],
            [0, 1, 1_000_000_001, 1_008_000_000],
            [0, 2, 1_000_000_002, 1_009_000_000],
        ],
        dtype="<u8",
    ).tofile(tmp_path / "elastic.records")
    stamps = np.zeros((3, 256), dtype=STAMP)
    stamps["begin"] = 100
    stamps["end"] = 200
    stamps["sm"][0] = 16
    stamps["sm"][1] = 17
    stamps["sm"][2] = 0
    stamps["value"] = reference(2048)
    stamps.tofile(tmp_path / "elastic.stamps")
    (tmp_path / "point.json").write_text(json.dumps(point))
    return point


def write(tmp_path, point):
    (tmp_path / "point.json").write_text(json.dumps(point))


def test_valid_db_point_replays_latency_period_and_windows(tmp_path):
    fixture(tmp_path)
    result = derive(tmp_path)
    assert result["valid"]
    assert result["frames"] == 3
    assert result["period_ms"]["p99"] == pytest.approx(5.0)
    assert result["borrow_open_ms"]["p50"] > 0
    assert result["deadline_ms"] == 20.0
    assert result["borrowed_units"] == 1
    assert result["elastic_completed_within_service"]
    assert result["bursts_completed_after_cutoff"] == 0


def test_rejects_dynamic_without_two_elastic_lanes_plus_borrow(tmp_path):
    point = fixture(tmp_path)
    point["elastic_pool"]["lanes"] = [
        dict(stream=20, context=2, borrowed=False),
        dict(stream=22, context=1, borrowed=True),
    ]
    point["elastic_pool"]["borrowed_lane"] = 1
    write(tmp_path, point)
    result = derive(tmp_path)
    assert not result["valid"]
    assert "lane_table" in result["failed_checks"]


def test_rejects_borrowed_unit_outside_stable_partition(tmp_path):
    fixture(tmp_path)
    stamps = np.fromfile(tmp_path / "elastic.stamps", dtype=STAMP).reshape(-1, 256)
    stamps["sm"][2] = 16
    stamps.tofile(tmp_path / "elastic.stamps")
    result = derive(tmp_path)
    assert not result["valid"]
    assert "elastic_sm_membership" in result["failed_checks"]


def test_rejects_elastic_load_released_after_stable_service(tmp_path):
    point = fixture(tmp_path)
    point["arguments"]["burst_period_ms"] = 5000.0
    point["arguments"]["bursts"] = 2
    point["arguments"]["units"] = 1
    np.array(
        [[0, 0, 1_000_000_000, 1_005_000_000], [1, 1, 6_000_000_000, 6_001_000_000]],
        dtype="<u8",
    ).tofile(tmp_path / "elastic.records")
    stamps = np.fromfile(tmp_path / "elastic.stamps", dtype=STAMP).reshape(-1, 256)
    stamps[:2].tofile(tmp_path / "elastic.stamps")
    write(tmp_path, point)
    result = derive(tmp_path)
    assert not result["valid"]
    assert "elastic_offered_within_service" in result["failed_checks"]


def test_late_burst_completion_is_counted_not_rejected(tmp_path):
    fixture(tmp_path)
    records = np.fromfile(tmp_path / "elastic.records", dtype="<u8").reshape(-1, 4)
    records[2, 3] = 1_100_000_000
    records.tofile(tmp_path / "elastic.records")
    result = derive(tmp_path)
    assert result["valid"]
    assert not result["elastic_completed_within_service"]
    assert result["bursts_completed_after_cutoff"] == 1
    assert result["elastic_completed"] == 2


def test_rejects_missing_double_buffer_route(tmp_path):
    point = fixture(tmp_path)
    point["route"]["enabled"] = False
    write(tmp_path, point)
    result = derive(tmp_path)
    assert not result["valid"]
    assert "double_buffer" in result["failed_checks"]


def test_rejects_broken_window_linkage(tmp_path):
    point = fixture(tmp_path)
    point["windows"][0]["reclaim_before_scheduled_frame"] = 53
    write(tmp_path, point)
    result = derive(tmp_path)
    assert not result["valid"]
    assert "window_links" in result["failed_checks"]


def test_rejects_frame_completion_before_cycle_boundary(tmp_path):
    point = fixture(tmp_path)
    point["frames"][0]["output"] = 1.005
    write(tmp_path, point)
    result = derive(tmp_path)
    assert not result["valid"]
    assert "frame_timestamps" in result["failed_checks"]


def test_copy_audit_uses_completed_output_boundary():
    from scripts.benchmarks.resource_mixed_db.audit_copies import check_copy

    point = dict(
        arguments=dict(bursts=1, units=1),
        owner=dict(calibration=[(1.0, 1_000_000_000)]),
        frames=[dict(output=2.0)],
    )
    trace = dict(copies=[dict(bytes=256 * 24, start=2_001_000_000)])
    assert check_copy(point, trace) == 1.0
