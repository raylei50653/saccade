"""Reject invalid arrival evidence, queue bounds, timing and block routing."""

# scope: eval
# function: contract
# lifecycle: active

import numpy as np
import pytest

from scripts.benchmarks.resource_elastic.admission_report import derive


def fixture(chunks=1):
    case = dict(chunks=chunks, window=1, offset_us=0)
    host = np.array(
        [
            1000000,
            1010000,
            1010000,
            1020000,
            1025000,
            1030000,
            1040000,
            1200000,
            1900000,
            (chunks + 1) * 1000000,
            1,
            0,
            1,
            200000,
            100000,
        ],
        dtype=np.uint64,
    )
    chunk_host = np.zeros((16, 3), dtype=np.uint64)
    stamps = np.zeros((128 + chunks * 1024, 3), dtype=np.uint64)
    stamps[:128] = [200000, 250000, 0]
    for i in range(chunks):
        chunk_host[i] = [
            1000000 + i * 1000000,
            1005000 + i * 1000000,
            1900000 + i * 1000000,
        ]
        stamps[128 + i * 1024 : 128 + (i + 1) * 1024] = [
            100000 + i * 1000000,
            900000 + i * 1000000,
            1,
        ]
    return case, dict(
        host=host,
        chunk_host=chunk_host,
        stamps=stamps,
        values=np.ones(len(stamps), dtype=np.float32),
    )


@pytest.mark.parametrize("chunks", [1, 16])
def test_valid_bounded_trace(chunks):
    case, arrays = fixture(chunks)
    row = derive(case, arrays, [0], [1])
    assert row["stable_response_ms"] == 0.18
    assert row["drain_observed_ms"] == 0.88
    assert row["unretired_at_arrival"] == 1


@pytest.mark.parametrize(
    "field,index,value,match",
    [
        ("host", 14, 99999, "signal mismatch"),
        ("host", 2, 1011000, "arrival offset"),
        ("host", 12, 2, "admission bound"),
        ("host", 11, 1, "arrival queue"),
        ("host", 8, 1800000, "drain ordering"),
        ("stamps", (0, 2), 9, "stable SM escape"),
        ("stamps", (128, 2), 9, "burst SM escape"),
        ("stamps", (1, 1), 1, "GPU stamp ordering"),
        ("values", 1, float("nan"), "output mismatch"),
    ],
)
def test_invalid_trace(field, index, value, match):
    case, arrays = fixture()
    arrays[field][index] = value
    with pytest.raises(ValueError, match=match):
        derive(case, arrays, [0], [1])


def test_dispatch_during_freeze():
    case, arrays = fixture(16)
    arrays["chunk_host"][1, :2] = [1500000, 1505000]
    with pytest.raises(ValueError, match="during freeze"):
        derive(case, arrays, [0], [1])


def test_signal_does_not_allow_stable_before_burst():
    case, arrays = fixture()
    arrays["stamps"][1, 0] = 90000
    with pytest.raises(ValueError, match="before confirmed burst"):
        derive(case, arrays, [0], [1])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("alter", "checksum mismatch"),
        ("extra", "incomplete checksum manifest"),
        ("duplicate", "checksum path"),
    ],
)
def test_checksum_contract(tmp_path, mutation, match):
    import hashlib

    from scripts.benchmarks.resource_elastic.admission_report import checked_record

    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    entry = f"{hashlib.sha256(payload.read_bytes()).hexdigest()}  payload.json\n"
    (tmp_path / "SHA256SUMS").write_text(entry)
    if mutation == "alter":
        payload.write_text("[]")
    elif mutation == "extra":
        (tmp_path / "extra.json").write_text("{}")
    else:
        (tmp_path / "SHA256SUMS").write_text(entry * 2)
    with pytest.raises(ValueError, match=match):
        checked_record(tmp_path)
