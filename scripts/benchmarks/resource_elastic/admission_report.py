"""Replay raw CUDA admission evidence and summarize timing observations."""

# status: diagnostic
import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ("repeat", "route", "chunks", "window", "offset_us", "swap")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(case, arrays, stable_sms, burst_sms):
    h, ch, s, v = (arrays[k] for k in ("host", "chunk_host", "stamps", "values"))
    chunks, window = case["chunks"], case["window"]
    require(
        h.shape == (15,)
        and ch.shape == (16, 3)
        and s.shape == (128 + chunks * 1024, 3)
        and v.shape == (len(s),),
        "raw shape mismatch",
    )
    h = [int(x) for x in h]
    require(h[0] <= h[1] <= h[2] <= h[3] <= h[4] <= h[7] <= h[9], "host ordering")
    require(h[2] - h[1] == case["offset_us"] * 1000, "arrival offset")
    require(
        h[3] <= h[5] <= h[6] <= h[7] and h[4] <= h[8] <= h[9], "observation ordering"
    )
    admitted, completed = h[10:12]
    require(
        0 <= completed <= admitted <= chunks
        and admitted >= window
        and 0 < h[12] <= window,
        "admission bound",
    )
    require(np.all(ch[chunks:] == 0), "unused chunks")
    ch = ch[:chunks]
    require(np.all(ch[:window, 1] <= h[1]), "initial window ordering")
    require(
        np.all(ch[:, 0] <= ch[:, 1]) and np.all(ch[:, 1] <= ch[:, 2]),
        "chunk host ordering",
    )
    require(int(ch[0, 0]) >= h[0] and int(ch[:, 2].max()) <= h[9], "chunk host bounds")
    require(
        int(np.sum(ch[:, 0] < h[3])) == admitted
        and int(np.sum(ch[:, 2] < h[3])) == completed,
        "arrival queue mismatch",
    )
    require(int(ch[:admitted, 2].max()) <= h[8], "drain ordering")
    require(
        all(int(t) >= max(h[7], h[8]) for t in ch[admitted:, 0]),
        "admission continued during freeze",
    )
    outstanding = [
        i + 1 - sum(int(t) <= int(ch[i, 0]) for t in ch[:i, 2]) for i in range(chunks)
    ]
    require(max(outstanding) == h[12], "unretired count mismatch")
    require(np.all(s[:, 0] > 0) and np.all(s[:, 0] <= s[:, 1]), "GPU stamp ordering")
    require(set(map(int, s[:128, 2])) <= set(stable_sms), "stable SM escape")
    require(set(map(int, s[128:, 2])) <= set(burst_sms), "burst SM escape")
    require(
        np.isfinite(v).all() and np.all(v[:128] == v[0]) and np.all(v[128:] == v[128]),
        "output mismatch",
    )
    require(h[13] == int(s[0, 0]) and h[14] == int(s[128, 0]), "start signal mismatch")
    a0, a1 = int(s[:128, 0].min()), int(s[:128, 1].max())
    b = s[128:].reshape(chunks, 1024, 3)
    starts, ends = b[:, :, 0].min(axis=1), b[:, :, 1].max(axis=1)
    require(
        all(int(ends[i]) <= int(starts[i + 1]) for i in range(chunks - 1)),
        "burst stream ordering",
    )
    require(a0 >= h[14], "stable started before confirmed burst")

    def ms(a, b):
        return (a - b) / 1e6

    return {
        "stable_response_ms": ms(h[7], h[3]),
        "stable_signal_observed_ms": ms(h[6], h[3]),
        "signal_poll_interval_ms": ms(h[6], h[5]),
        "stable_launch_ms": ms(h[4], h[3]),
        "arrival_overshoot_ms": ms(h[3], h[2]),
        "burst_start_observed_ms": ms(h[1], h[0]),
        "drain_observed_ms": ms(h[8], h[3]),
        "burst_host_ms": ms(h[9], h[0]),
        "burst_gpu_ms": ms(int(ends[-1]), int(starts[0])),
        "stable_gpu_ms": ms(a1, a0),
        "stable_gpu_start_from_burst_ms": ms(a0, int(starts[0])),
        "admitted_gpu_end_from_stable_start_ms": ms(int(ends[admitted - 1]), a0),
        "chunk_gpu_max_ms": float(np.max(ends - starts)) / 1e6,
        "unretired_at_arrival": admitted - completed,
        "admitted_at_arrival": admitted,
        "already_retired_all_at_arrival": int(completed == chunks),
        "prefix_done_before_stable_start": int(int(ends[admitted - 1]) <= a0),
    }


def summarize(rows):
    return {
        key: dict(
            zip(
                ("p50", "p95", "p99"),
                np.percentile([r[key] for r in rows], [50, 95, 99]).tolist(),
            )
        )
        for key in rows[0]
    }


def checked_record(root):
    root = root.resolve()
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        require(path.is_relative_to(root) and name not in listed, "checksum path")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest, "checksum mismatch"
        )
        listed.add(name)
    require(
        listed
        == {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        },
        "incomplete checksum manifest",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    require(result["status"] == "validated_synthetic_admission", "unvalidated run")
    for name, digest in manifest["source_sha256"].items():
        require(
            hashlib.sha256((root / "execution_sources" / name).read_bytes()).hexdigest()
            == digest,
            "source identity mismatch",
        )
    args, pools = manifest["args"], result["pools"]
    require(len(pools) == 4, "pool count")
    for i, pool in enumerate(pools):
        require(
            pool["sm_ids_before"] == pool["sm_ids_after"]
            and len(set(pool["sm_ids_before"])) == pool["actual_sms"],
            "pool probe mismatch",
        )
        require(
            pool["actual_sms"]
            == (
                args["sms"]
                if i < 2
                else 2 * args["sms"]
                if i == 2
                else result["device_sms"]
            ),
            "pool budget mismatch",
        )
    require(
        not set(pools[0]["sm_ids_before"]) & set(pools[1]["sm_ids_before"]),
        "pools overlap",
    )
    expected = {
        (r, route, chunks, window, off, swap)
        for r, route, (chunks, window), off, swap in itertools.product(
            range(args["repeats"]),
            ("disjoint", "shared_budget_priority", "full_priority"),
            ((1, 1), (16, 1), (16, 4), (16, 16)),
            args["offset_us"],
            (0, 1),
        )
    }
    summaries = result["summary"]
    require(
        len(summaries) == len(expected)
        and {tuple(s[k] for k in KEYS) for s in summaries} == expected,
        "missing or duplicate condition",
    )
    raw_names = [s["raw"] for s in summaries]
    require(len(set(raw_names)) == len(raw_names), "duplicate raw artifact")
    replayed = []
    for summary in summaries:
        case = {k: summary[k] for k in KEYS}
        pa, pb = (
            (pools[case["swap"]], pools[1 - case["swap"]])
            if case["route"] == "disjoint"
            else (pools[2 if case["route"] == "shared_budget_priority" else 3],) * 2
        )
        require(summary["raw"] in listed, "unlisted raw artifact")
        with np.load(root / summary["raw"], allow_pickle=False) as raw:
            require(
                set(raw.files) == {"host", "chunk_host", "stamps", "values"},
                "raw fields",
            )
            require(
                summary["samples"] == args["samples"]
                and all(len(raw[k]) == args["samples"] for k in raw.files),
                "sample count",
            )
            arrays = {k: raw[k] for k in raw.files}
        rows = [
            derive(
                case,
                {k: v[i] for k, v in arrays.items()},
                pa["sm_ids_before"],
                pb["sm_ids_before"],
            )
            for i in range(args["samples"])
        ]
        require(
            np.all(arrays["values"][:, :128] == result["stable_value"]),
            "cross-condition output mismatch",
        )
        recomputed = summarize(rows)
        require(
            set(summary) == set(KEYS) | {"raw", "samples"} | set(recomputed),
            "summary fields",
        )
        for metric, quantiles in recomputed.items():
            require(summary[metric] == quantiles, f"summary mismatch: {metric}")
        replayed.append(
            {
                **case,
                "samples": len(rows),
                **recomputed,
                "counts": {
                    k: sum(row[k] for row in rows)
                    for k in (
                        "already_retired_all_at_arrival",
                        "prefix_done_before_stable_start",
                    )
                },
            }
        )
    return {
        "archive": str(root),
        "manifest": manifest,
        "pools": pools,
        "summary": replayed,
    }


def markdown(record):
    groups = defaultdict(list)
    for row in record["summary"]:
        groups[row["route"], row["chunks"], row["window"], row["offset_us"]].append(row)
    lines = [
        "Ranges are min–max of repetition/role quantiles, in ms; not confidence intervals.",
        "",
        "| Route | Chunks | Window | Offset µs | Stable p50 | Stable p99 | Drain p50 | Burst host p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (route, chunks, window, offset), rows in sorted(groups.items()):

        def span(metric, q="p50"):
            vals = [r[metric][q] for r in rows]
            return f"{min(vals):.3f}–{max(vals):.3f}"

        lines.append(
            f"| {route} | {chunks} | {window} | {offset} | {span('stable_response_ms')} | {span('stable_response_ms', 'p99')} | {span('drain_observed_ms')} | {span('burst_host_ms')} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    record = checked_record(args.archive)
    args.json_output.write_text(json.dumps(record, indent=2) + "\n")
    args.markdown_output.write_text(markdown(record))


if __name__ == "__main__":
    main()
