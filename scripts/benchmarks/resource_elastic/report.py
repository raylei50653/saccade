"""Replay synthetic probe summaries and render per-repetition ranges."""

# status: diagnostic
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def checked_record(root):
    root = root.resolve()
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("checksum path outside archive")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {name}")
        if name in listed:
            raise ValueError("duplicate checksum entry")
        listed.add(name)
    actual_files = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    }
    if listed != actual_files:
        raise ValueError("incomplete checksum manifest")
    result = json.loads((root / "result.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    samples = json.loads((root / "samples.json").read_text())
    if result["status"] != "validated_synthetic_probe":
        raise ValueError("unvalidated run")
    pools = result["pools"]
    for p in pools:
        if (
            p["sm_ids_before"] != p["sm_ids_after"]
            or len(p["sm_ids_before"]) != p["actual_sms"]
        ):
            raise ValueError("pool probe mismatch")
    if set(pools[0]["sm_ids_before"]) & set(pools[1]["sm_ids_before"]):
        raise ValueError("pools overlap")
    groups = defaultdict(list)
    for row in samples:
        if not row["route"].startswith("full"):
            stable_pool = set(pools[0]["sm_ids_before"])
            burst_pool = set(pools[row["route"] == "disjoint"]["sm_ids_before"])
            if not row["stable_sm_ids"] or not set(row["stable_sm_ids"]) <= stable_pool:
                raise ValueError("stable SM escape")
            if row["kind"] and (
                not row["burst_sm_ids"] or not set(row["burst_sm_ids"]) <= burst_pool
            ):
                raise ValueError("burst SM escape")
        groups[row["repeat"], row["route"], row["kind"]].append(row)
    expected_cases = {
        (name, k)
        for name in (
            "reserved_solo",
            "full_solo",
            "same_pool",
            "same_pool_priority",
            "disjoint",
            "full_shared",
            "full_priority",
        )
        for k in ([0] if name.endswith("solo") else [1, 2])
    }
    expected = {
        (r, name, k) for r in range(manifest["repeats"]) for name, k in expected_cases
    }
    summary_keys = [(s["repeat"], s["route"], s["kind"]) for s in result["summary"]]
    if (
        set(groups) != expected
        or set(summary_keys) != expected
        or len(summary_keys) != len(expected)
    ):
        raise ValueError("missing or duplicate condition")
    for summary in result["summary"]:
        rows = groups[summary["repeat"], summary["route"], summary["kind"]]
        n = manifest["samples"]
        if (
            summary["samples"] != n
            or len(rows) != n
            or {r["sample"] for r in rows} != set(range(n))
        ):
            raise ValueError("missing or duplicate sample")
        for key, value in summary.items():
            if isinstance(value, dict):
                values = np.array([r[key] for r in rows])
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite metric")
                actual = np.percentile(values, [50, 95, 99])
                if not np.allclose(
                    actual,
                    [value[q] for q in ("p50", "p95", "p99")],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError(f"summary mismatch: {key}")
        if (
            summary["overlap_fraction"]
            != sum(r["envelope_overlap_ms"] > 0 for r in rows) / n
        ):
            raise ValueError("overlap summary mismatch")
    return {
        "archive": str(root),
        "manifest": manifest,
        "result": result,
        "samples_sha256": hashlib.sha256(
            (root / "samples.json").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    records = [checked_record(p) for p in args.archives]
    lines = [
        "Ranges below are min–max across repetition quantiles, in ms;",
        "they are descriptive ranges, not confidence intervals.",
        "",
        "| Sweep | Compute chunks | Route | Burst | Stable host p50 | Stable host p99 | Burst GPU p50 |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for sweep, record in enumerate(records, start=1):
        result = record["result"]
        groups = defaultdict(list)
        for s in result["summary"]:
            groups[s["route"], s["kind"]].append(s)
        for (route, kind), rows in sorted(groups.items()):

            def span(metric, quantile):
                values = [s[metric][quantile] for s in rows]
                return f"{min(values):.3f}–{max(values):.3f}"

            lines.append(
                f"| {sweep} | {result.get('compute_chunks', 1)} | {route} | "
                f"{['none', 'compute', 'memory'][kind]} | "
                f"{span('host_response_ms', 'p50')} | "
                f"{span('host_response_ms', 'p99')} | "
                f"{span('burst_gpu_ms', 'p50')} |"
            )
    args.json_output.write_text(json.dumps(records, indent=2) + "\n")
    args.markdown_output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
