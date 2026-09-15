"""Synthetic elastic summaries reject missing samples, SM escapes and altered metrics."""

# scope: eval
# function: contract
# lifecycle: active

import hashlib
import json

import pytest

from scripts.benchmarks.resource_elastic.report import checked_record


def archive(tmp_path, mutation=None):
    rows, summary = [], []
    for name in (
        "reserved_solo",
        "full_solo",
        "same_pool",
        "same_pool_priority",
        "disjoint",
        "full_shared",
        "full_priority",
    ):
        for kind in [0] if name.endswith("solo") else [1, 2]:
            base = {"repeat": 0, "route": name, "kind": kind}
            for i, v in enumerate((1.0, 3.0)):
                rows.append(
                    {
                        **base,
                        "sample": i,
                        "host_response_ms": v,
                        "stable_sm_ids": [0],
                        "burst_sm_ids": [1 if name == "disjoint" else 0]
                        if kind
                        else [],
                        "envelope_overlap_ms": 0,
                    }
                )
            summary.append(
                {
                    **base,
                    "samples": 2,
                    "overlap_fraction": 0,
                    "host_response_ms": {"p50": 2, "p95": 2.9, "p99": 2.98},
                }
            )
    result = {
        "status": "validated_synthetic_probe",
        "summary": summary,
        "pools": [
            {"actual_sms": 1, "sm_ids_before": [i], "sm_ids_after": [i]}
            for i in range(2)
        ],
    }
    if mutation:
        mutation(rows, result)
    for name, value in (
        ("samples", rows),
        ("result", result),
        ("manifest", {"samples": 2, "repeats": 1}),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted(tmp_path.glob("*.json"))
        )
    )
    return tmp_path


def test_complete_archive(tmp_path):
    assert (
        checked_record(archive(tmp_path))["result"]["status"]
        == "validated_synthetic_probe"
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows, result: rows.pop(), "sample"),
        (lambda rows, result: rows[0].update(stable_sm_ids=[9]), "SM escape"),
        (lambda rows, result: rows[0].update(host_response_ms=100), "summary mismatch"),
        (lambda rows, result: result.update(status="failed"), "unvalidated"),
        (
            lambda rows, result: result["summary"].append(result["summary"][0]),
            "condition",
        ),
    ],
)
def test_rejects_inconsistent_evidence(tmp_path, mutation, match):
    with pytest.raises(ValueError, match=match):
        checked_record(archive(tmp_path, mutation))


def test_rejects_unlisted_artifact(tmp_path):
    root = archive(tmp_path)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete checksum"):
        checked_record(root)


def test_rejects_changed_bytes(tmp_path):
    root = archive(tmp_path)
    (root / "samples.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        checked_record(root)
