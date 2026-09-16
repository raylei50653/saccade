"""Verify time-ledger replay and preserve timing misses as measured evidence."""

# scope: eval
# function: contract
# lifecycle: active

import numpy as np
import pytest

from scripts.benchmarks.resource_elastic.time_budget_report import calibrate, derive


def fixture():
    case = dict(window=0, budget_ns=20)
    ch = np.zeros((8, 5), dtype=np.uint64)
    arrivals = np.zeros((4, 9), dtype=np.uint64)
    for k in range(4):
        target = 2002 + 1000 * k
        ch[2 * k] = [target - 20, target - 19, target + 5, 10, 10]
        ch[2 * k + 1] = [target - 10, target - 9, target + 15, 10, 20]
        arrivals[k] = [
            target,
            target,
            target + 1,
            target + 10,
            target + 15,
            2 * k + 2,
            2 * k,
            20,
            target + 30,
        ]
    ch[0, :2] = [1000, 1001]
    stamps = np.zeros((512 + 8 * 1024, 3), dtype=np.uint64)
    stamps[:512] = [200, 205, 0]
    for i in range(8):
        stamps[512 + i * 1024 : 512 + (i + 1) * 1024] = [100 + 20 * i, 110 + 20 * i, 1]
    arrays = dict(
        host=np.array([1000, 6000, 1002], dtype=np.uint64),
        chunk_host=ch,
        arrivals=arrivals,
        stamps=stamps,
        values=np.ones(len(stamps), dtype=np.float32),
    )
    return case, arrays


def replay(case, arrays):
    return derive(case, arrays, [0], [1], [512] * 8, [10] * 8, 1000)


def test_valid_budget_and_fixed_window():
    case, arrays = fixture()
    assert len(replay(case, arrays)) == 4
    assert replay(dict(window=2, budget_ns=0), arrays)[0]["unretired_at_arrival"] == 2
    with pytest.raises(ValueError, match="admission bound"):
        replay(dict(window=1, budget_ns=0), arrays)
    with pytest.raises(ValueError, match="admission bound"):
        replay(dict(window=0, budget_ns=19), arrays)


@pytest.mark.parametrize(
    "field,index,value,match",
    [
        ("host", 2, 1000, "first dispatch before start observation"),
        ("chunk_host", (1, 4), 19, "ledger mismatch"),
        ("chunk_host", (0, 3), 11, "estimate identity"),
        ("arrivals", (0, 7), 19, "arrival ledger"),
        ("arrivals", (1, 0), 3003, "arrival target"),
        ("arrivals", (0, 6), 1, "arrival queue"),
        ("arrivals", (0, 4), 2005, "drain order"),
        ("stamps", (512, 2), 9, "elastic SM escape"),
        ("stamps", (0, 2), 9, "stable SM escape"),
        ("values", 0, float("nan"), "stable output"),
    ],
)
def test_reject_corrupted_trace(field, index, value, match):
    case, arrays = fixture()
    arrays[field][index] = value
    with pytest.raises(ValueError, match=match):
        replay(case, arrays)


def test_observed_budget_miss_is_retained():
    case, arrays = fixture()
    arrays["arrivals"][0, 4] = 2027  # 25 ns observed drain > 20 ns estimate budget.
    row = replay(case, arrays)[0]
    assert row["drain_over_budget"] == 1
    assert row["drain_excess_ms"] == 5 / 1e6


def test_dispatch_during_freeze():
    case, arrays = fixture()
    arrays["chunk_host"][2, :2] = [2020, 2021]
    with pytest.raises(ValueError, match="freeze violation"):
        replay(case, arrays)


def test_calibration_uses_duration_distribution_and_margin():
    s = np.zeros((3, 1024, 3), dtype=np.uint64)
    s[:, :, 0] = 100
    s[:, :, 1] = np.array([110, 120, 130])[:, None]
    s[:, :, 2] = 1
    a = dict(stamps=s, values=np.ones((3, 1024), dtype=np.float32))
    assert calibrate(a, [1]) == 35  # ceil(p95([10,20,30]) * 1.2).
    with pytest.raises(ValueError, match="SM escape"):
        calibrate(a, [0])


def test_rolling_count_requires_retirement_in_current_active_interval():
    case, arrays = fixture()
    arrays["chunk_host"][0, :3] = [1000, 1001, 1970]
    arrays["chunk_host"][1, 4] = 10
    arrays["arrivals"][0, 6:8] = [1, 10]
    rows = replay(case, arrays)
    assert rows[0]["rolling_dispatches"] == 1
    assert all(row["rolling_dispatches"] == 0 for row in rows[1:])
