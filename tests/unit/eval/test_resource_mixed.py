"""Fail closed on corrupt full-pipeline mixed-workload evidence."""

# scope: eval
# function: contract
# lifecycle: active
import json

import numpy as np
import pytest

from scripts.benchmarks.resource_mixed.hardware_report import (
    intersect_intervals,
    interval_ns,
    merge_intervals,
    sm_residency,
)
from scripts.benchmarks.resource_mixed.report import STAMP, derive, digest, reference


def fixture(tmp_path, policy="dynamic"):
    stable = list(range(16))
    elastic = list(range(16, 32))
    probe = dict(direct_sm_ids=stable, graph_sm_ids=stable)
    output = tmp_path / "eval.txt"
    output.write_text("fixed MOT output\n")
    frame = dict(
        frame=51,
        arrival=1.02,
        request=1.0201,
        drained=1.0202,
        admitted=1.0203,
        output=1.025,
        finished=1.026,
        context_owned=True,
    )
    point = dict(
        arguments=dict(
            policy=policy,
            window=1,
            iterations=2048,
            frames=51,
            period_ms=20,
            deadline_ms=16.667,
            units=2,
            audit=False,
        ),
        frames=[frame],
        origin=1.0,
        native_origin=1_000_000_000,
        native_anchor_after=1.0,
        owner=dict(
            actual_sm_count=16,
            streams=[dict(owned=True)],
            main_thread_context_still_owned_at_finalize=True,
            thread_context_switch_errors=[],
            probe=dict(before=probe, after=probe),
        ),
        elastic_pool=dict(
            actual_sms=16, probe=dict(direct_sm_ids=elastic, graph_sm_ids=elastic)
        ),
        bursts=1,
        reference=reference(2048),
        error=None,
        output_sha256={"eval.txt": digest(output)},
    )
    np.array(
        [[0, 0, 1_000_000_000, 1_005_000_000], [0, 1, 1_000_000_001, 1_008_000_000]],
        dtype="<u8",
    ).tofile(tmp_path / "elastic.records")
    stamps = np.zeros((2, 256), dtype=STAMP)
    stamps["begin"] = 100
    stamps["end"] = 200
    stamps["sm"][0] = 16
    stamps["sm"][1] = 0
    stamps["value"] = reference(2048)
    stamps.tofile(tmp_path / "elastic.stamps")
    (tmp_path / "point.json").write_text(json.dumps(point))
    return point


def write(tmp_path, point):
    (tmp_path / "point.json").write_text(json.dumps(point))


def test_valid_pair_and_deadline_includes_admission(tmp_path):
    point = fixture(tmp_path)
    result = derive(tmp_path)
    assert result["valid"]
    assert result["elastic_completed"] == 2
    assert result["service_pass"]
    point["frames"][0].update(output=1.038, finished=1.039)
    write(tmp_path, point)
    result = derive(tmp_path)
    assert result["valid"]
    assert not result["service_pass"]
    assert result["miss_count"] == 1


def test_hardware_timeline_interval_replay():
    stable = merge_intervals([(10, 20), (15, 30), (40, 50)])
    elastic = merge_intervals([(5, 12), (18, 25), (45, 55)])
    assert stable == [[10, 30], [40, 50]]
    assert elastic == [[5, 12], [18, 25], [45, 55]]
    overlap = intersect_intervals(stable, elastic)
    assert overlap == [[10, 12], [18, 25], [45, 50]]
    assert interval_ns(overlap) == 14

    blocks = np.zeros(3, dtype=STAMP)
    blocks["begin"] = [0, 0, 10]
    blocks["end"] = [10, 20, 20]
    residency = sm_residency(blocks)
    assert residency["peak_blocks"] == 2
    assert residency["active_cycles"] == 20
    assert residency["resident_block_cycles"] == 40


@pytest.mark.parametrize(
    "mutation,check",
    [
        (lambda p: p["frames"][0].update(frame=52), "complete_frames"),
        (lambda p: p["frames"][0].update(context_owned=False), "owned_frames"),
        (lambda p: p["frames"][0].update(arrival=1.019), "arrivals"),
        (lambda p: p["frames"][0].update(drained=1.019), "timestamps"),
        (lambda p: p["owner"].update(actual_sm_count=8), "stable_size"),
        (
            lambda p: p["elastic_pool"]["probe"].update(direct_sm_ids=list(range(16))),
            "envelope",
        ),
        (lambda p: p.update(error="CUDA failed"), "no_error"),
        (lambda p: p.update(bursts=2), "elastic_count"),
    ],
)
def test_reject_corrupt_metadata(tmp_path, mutation, check):
    point = fixture(tmp_path)
    mutation(point)
    write(tmp_path, point)
    result = derive(tmp_path)
    assert not result["valid"]
    assert check in result["failed_checks"]


@pytest.mark.parametrize(
    "field,value,check",
    [
        ("sm", 40, "elastic_sm_membership"),
        ("value", 2.0, "elastic_output"),
        ("end", 50, "elastic_times"),
    ],
)
def test_reject_gpu_evidence(tmp_path, field, value, check):
    fixture(tmp_path)
    stamps = np.fromfile(tmp_path / "elastic.stamps", dtype=STAMP)
    stamps[field][0] = value
    stamps.tofile(tmp_path / "elastic.stamps")
    assert check in derive(tmp_path)["failed_checks"]


def test_cutoff_does_not_count_tail_completion(tmp_path):
    fixture(tmp_path)
    records = np.fromfile(tmp_path / "elastic.records", dtype="<u8").reshape(-1, 4)
    records[1, 3] = 1_030_000_000
    records.tofile(tmp_path / "elastic.records")
    result = derive(tmp_path)
    assert result["valid"]
    assert result["elastic_completed"] == 1
    assert result["elastic_drain_after_stable_s"] == pytest.approx(0.004)


def audited_fixture(tmp_path):
    point = fixture(tmp_path)
    point["arguments"]["audit"] = True
    point["owner"]["cupti_execution_context_id"] = 1
    point["elastic_pool"]["cupti_context"] = 2
    write(tmp_path, point)
    trace = (
        "N 0 pipeline_kernel\nN 1 mixed_elastic\n"
        "L 0 mixed_begin\nL 1000 mixed_end\n"
        "K 300 400 1 1 1 0 0 0\n"
        "K 100 200 2 2 0 0 1 0\n"
        "K 100 200 1 3 0 0 1 0\n"
        "E dropped=0\n"
    )
    (tmp_path / "cupti.trace").write_text(trace)
    return trace


def test_audit_proves_real_graphs_and_nonoverlapping_borrow(tmp_path):
    audited_fixture(tmp_path)
    result = derive(tmp_path)
    assert result["valid"]
    assert result["graph_kernels"] == 1
    assert result["borrowed_stable_kernel_overlaps"] == 0


@pytest.mark.parametrize(
    "before,after,check",
    [
        ("K 300 400 1", "K 300 400 9", "audit_stable_context"),
        ("K 100 200 1", "K 100 350 1", "audit_borrow_no_stable_overlap"),
        ("K 100 200 2", "K 100 200 9", "audit_elastic_contexts"),
        ("E dropped=0", "E dropped=1", "audit_no_drops"),
        ("1 1 1 0 0 0", "1 1 0 0 0 0", "audit_graphs"),
    ],
)
def test_audit_rejects_escape_overlap_and_missing_evidence(
    tmp_path, before, after, check
):
    trace = audited_fixture(tmp_path)
    (tmp_path / "cupti.trace").write_text(trace.replace(before, after))
    assert check in derive(tmp_path)["failed_checks"]


def main_audit_fixture(tmp_path):
    from scripts.benchmarks.resource_mixed.audit import audit

    run = tmp_path / "point"
    run.mkdir()
    frames = []
    for i in range(300):
        arrival = 1 + (i + 1) / 60
        frames.append(
            dict(
                frame=51 + i,
                arrival=arrival,
                request=arrival + 0.0001,
                drained=arrival + 0.0002,
                finished=arrival + 0.006,
            )
        )
    point = dict(
        arguments=dict(
            policy="dynamic",
            frames=350,
            units=256,
            period_ms=1000 / 60,
            deadline_ms=1000 / 60,
        ),
        bursts=50,
        frames=frames,
        origin=1.0,
        native_origin=1_000_000_000,
        native_anchor_after=1.0,
    )
    (run / "point.json").write_text(json.dumps(point))
    response = [(f["finished"] - f["arrival"]) * 1000 for f in frames]
    report = dict(
        miss_count=0,
        response_ms=dict(p99=float(np.percentile(response, 99))),
        service_pass=True,
    )
    (run / "report.json").write_text(json.dumps(report))
    raw = np.zeros((12800, 4), dtype="<u8")
    raw[:, 0] = np.repeat(np.arange(50), 256)
    raw[:, 2] = 1_000_000_000 + raw[:, 0] * 100_000_000
    raw[:, 3] = raw[:, 2] + 100_000
    raw[0, 1] = 1
    raw.tofile(run / "elastic.records")
    manifest = dict(
        finished_epoch=2, schedule=[dict(name="point", audit=False, files={})]
    )
    (tmp_path / "sweep.json").write_text(json.dumps(manifest))
    assert audit(tmp_path)["verified"]
    return point, raw


def test_independent_audit_rejects_changed_service_target(tmp_path):
    from scripts.benchmarks.resource_mixed.audit import audit

    point, _ = main_audit_fixture(tmp_path)
    point["arguments"]["deadline_ms"] = 100
    (tmp_path / "point" / "point.json").write_text(json.dumps(point))
    with pytest.raises(ValueError, match="frozen main contract"):
        audit(tmp_path)


def test_independent_audit_rejects_borrow_admission_during_service(tmp_path):
    from scripts.benchmarks.resource_mixed.audit import audit

    _, raw = main_audit_fixture(tmp_path)
    raw[0, 2:] = [1_019_000_000, 1_019_100_000]
    raw.tofile(tmp_path / "point" / "elastic.records")
    with pytest.raises(ValueError, match="borrow admission during stable"):
        audit(tmp_path)


def test_gpu_evidence_copy_must_wait_for_all_stable_frames():
    from scripts.benchmarks.resource_mixed.audit_copies import check_copy

    point = dict(
        bursts=1,
        arguments=dict(units=1),
        owner=dict(calibration=[(1.0, 1_000_000_000)]),
        frames=[dict(finished=2.0)],
    )
    trace = dict(copies=[dict(bytes=256 * 24, start=2_001_000_000)])
    assert check_copy(point, trace) == 1.0
    trace["copies"][0]["start"] = 1_999_000_000
    with pytest.raises(ValueError, match="precedes stable completion"):
        check_copy(point, trace)
