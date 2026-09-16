"""Reject invalid routing ownership, work and timing evidence."""

# scope: eval
# function: contract
# lifecycle: active

import numpy as np
import pytest

from scripts.benchmarks.resource_routing.report import derive, summarize


def fixture():
    case = dict(policy="dynamic", window=1, iterations=2048, lane_pools=[0, 1, 2])
    pools = [dict(actual_sms=1, sm_ids_before=[i]) for i in range(3)]
    pools.append(dict(actual_sms=3, sm_ids_before=[0, 1, 2]))
    rows = [[1, 1, 16, 1000, 1001, 170001], [1, 2, 16, 1002, 1003, 31010]]
    arrivals = []
    for job in range(16):
        target = 1000 + 10000 * (job + 1)
        units = 1 if job % 4 < 2 else 8
        request = target if units == 8 else 0
        drain = target + 10 if request else 0
        first_c = 0
        for unit in range(units):
            lane = 2 if unit == 1 and request else 0
            begin = target + 20 + unit * 10
            if lane == 2:
                first_c = begin
            rows.append([0, lane, job, begin, begin + 1, begin + 5])
        done = rows[-1][5]
        arrivals.append(
            [target, target, request, drain, first_c, done, done if request else 0]
        )
    records = np.array(rows, dtype=np.uint64)
    stamps = np.zeros((len(rows), 256, 3), dtype=np.uint64)
    for i, row in enumerate(rows):
        stamps[i] = [row[3] + 2, row[5] - 1, row[1]]
    arrays = dict(
        host=np.array([1000, 170000, 170002, len(rows)], dtype=np.uint64),
        records=records,
        arrivals=np.array(arrivals, dtype=np.uint64),
        stamps=stamps,
        values=np.ones((len(rows), 256), dtype=np.float32),
    )
    return case, arrays, pools


def test_valid_transfer_and_deadline_misses():
    case, arrays, pools = fixture()
    row = derive(case, arrays, pools, 10000)
    assert len(row["transition_ms"]) == 8
    assert row["gpu_handoff_gap_ms"][0] > 0
    assert 0 <= row["observed_sm_coverage"] <= 1
    row["stable_ms"][0] = 5
    assert summarize([row])["stable_miss_fraction_4ms"] == 1 / 16


@pytest.mark.parametrize(
    "field,index,value,reason",
    [
        ("stamps", (0, 0, 2), 9, "SM escape"),
        ("arrivals", (0, 0), 1, "arrival target"),
        ("records", (2, 0), 1, "elastic job identity"),
        ("values", (2, 1), 2, "block outputs"),
        ("arrivals", (2, 3), 31000, "incomplete drain"),
        ("arrivals", (2, 4), 31001, "first C route"),
        ("records", (1, 5), 31035, "queue bound"),
        ("stamps", (1, 0, 1), 99999, "GPU owner overlap"),
    ],
)
def test_reject_corrupt_evidence(field, index, value, reason):
    case, arrays, pools = fixture()
    arrays[field][index] = value
    with pytest.raises(ValueError, match=reason):
        derive(case, arrays, pools, 10000)


def test_policy_does_not_allow_borrowing_from_floor():
    case, arrays, pools = fixture()
    arrays["records"][1, 1] = 0
    arrays["stamps"][1, :, 2] = 0
    with pytest.raises(ValueError, match="elastic on stable floor"):
        derive(case, arrays, pools, 10000)


def test_independent_audit_detects_gpu_overlap(tmp_path):
    import json

    from scripts.benchmarks.resource_routing.audit import audit

    case, arrays, pools = fixture()
    row = derive(case, arrays, pools, 10000)
    case.update(repeat=0, swap=0, raw=["raw.npz"], metrics=summarize([row]))
    np.savez_compressed(tmp_path / "raw.npz", **arrays)
    (tmp_path / "result.json").write_text(
        json.dumps(dict(status="validated_synthetic_routing", summary=[case]))
    )
    assert audit(tmp_path)["totals"]["elastic_to_stable_transfers"] == 8
    arrays["stamps"][1, 0, 1] = 99999
    np.savez_compressed(tmp_path / "raw.npz", **arrays)
    with pytest.raises(ValueError, match="overlapping GPU owners"):
        audit(tmp_path)
