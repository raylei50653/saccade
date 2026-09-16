"""Replay routing ownership, output, service and observed SM activity evidence."""

# status: diagnostic
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

POLICIES = [
    "fixed_reservation",
    "static_elastic",
    "borrow",
    "headroom",
    "dynamic",
    "shared_24",
    "full_shared",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(case, arrays, pools, period):
    h, r, a, s, v = [
        arrays[k] for k in ("host", "records", "arrivals", "stamps", "values")
    ]
    n = int(h[3])
    require(
        h.shape == (4,)
        and r.shape == (n, 6)
        and a.shape == (16, 7)
        and s.shape == (n, 256, 3)
        and v.shape == (n, 256),
        "raw shape",
    )
    require(h[0] < h[1] <= h[2], "host horizon")
    require(np.all((r[:, 0] <= 1) & (r[:, 1] <= 2)), "kind/lane")
    require(
        np.all(
            (h[0] <= r[:, 3])
            & (r[:, 3] <= r[:, 4])
            & (r[:, 4] <= r[:, 5])
            & (r[:, 5] <= h[2])
        ),
        "launch ordering",
    )
    require(np.all(r[1:, 3] >= r[:-1, 4]), "serial dispatcher")
    require(np.all(s[:, :, 1] > s[:, :, 0]), "GPU duration")
    require(np.all(np.isfinite(v)) and np.all(v == v[:, :1]), "block outputs")
    for kind in (0, 1):
        out = v[r[:, 0] == kind]
        require(len(out) > 0 and np.all(out == out[0, 0]), "work outputs")
    policy = case["policy"]
    for lane, pool in enumerate(case["lane_pools"]):
        require(
            np.all(np.isin(s[r[:, 1] == lane, :, 2], pools[pool]["sm_ids_before"])),
            "SM escape",
        )
    require(np.all(r[r[:, 0] == 1, 2] == 16), "elastic job identity")
    require(np.all(r[r[:, 0] == 0, 1] != 1), "stable on elastic floor")
    require(np.all(r[r[:, 0] == 1, 1] != 0), "elastic on stable floor")
    if policy in ("fixed_reservation", "headroom", "shared_24", "full_shared"):
        require(not np.any((r[:, 0] == 1) & (r[:, 1] == 2)), "forbidden borrowing")
    if policy in ("static_elastic", "shared_24", "full_shared"):
        require(not np.any((r[:, 0] == 0) & (r[:, 1] == 2)), "forbidden stable route")
    # Replay in-flight bounds from enqueue and completion observations.
    for lane in range(3):
        lr = r[r[:, 1] == lane]
        for index, record in enumerate(lr):
            outstanding = lr[:index, 5] > record[3]
            require(
                int(outstanding.sum()) < (case["window"] if record[0] else 1),
                "queue bound",
            )
            if record[0] == 0:
                require(
                    not np.any(outstanding & (lr[:index, 0] == 1)), "ownership overlap"
                )
    latency, reclaim, transition, gpu_transition = [], [], [], []
    for job in range(16):
        target, dispatch, request, drain, first_c, done, release = map(int, a[job])
        require(target == int(h[0]) + period * (job + 1), "arrival target")
        require(target <= dispatch <= done <= int(h[1]), "arrival ordering")
        if job:
            require(
                dispatch >= max(int(a[job - 1, 5]), int(a[job - 1, 6])), "job ordering"
            )
        jr = r[(r[:, 0] == 0) & (r[:, 2] == job)]
        units = 1 if job % 4 < 2 else 8
        require(
            len(jr) == units
            and np.all(jr[:, 3] >= dispatch)
            and done >= int(jr[:, 5].max()),
            "stable work conservation",
        )
        wants_c = policy in ("fixed_reservation", "borrow") or (
            policy in ("headroom", "dynamic") and units == 8
        )
        require(bool(request) == wants_c, "routing policy")
        if wants_c:
            require(
                dispatch <= request <= drain <= release and done <= release,
                "reclaim ordering",
            )
            prior = r[(r[:, 1] == 2) & (r[:, 3] < request)]
            require(
                not len(prior) or int(prior[:, 5].max()) <= drain, "incomplete drain"
            )
            require(
                not np.any(
                    (r[:, 1] == 2)
                    & (r[:, 0] == 1)
                    & (r[:, 3] >= request)
                    & (r[:, 3] < release)
                ),
                "borrow during stable ownership",
            )
            reclaim.append((drain - request) / 1e6)
        else:
            require(not (drain or first_c or release), "spurious transition")
        cr = jr[jr[:, 1] == 2]
        require(bool(first_c) == bool(len(cr)), "missing C launch")
        if len(cr):
            require(first_c == int(cr[0, 3]) and first_c >= drain, "first C route")
            transition.append((first_c - request) / 1e6)
            before = (r[:, 0] == 1) & (r[:, 1] == 2) & (r[:, 3] < request)
            after = (r[:, 0] == 0) & (r[:, 1] == 2) & (r[:, 2] == job)
            if before.any():
                last = int(s[before, :, 1].max())
                start = int(s[after, :, 0].min())
                require(start >= last, "GPU owner overlap")
                gpu_transition.append((start - last) / 1e6)
        latency.append((done - target) / 1e6)
    require(int((r[:, 0] == 0).sum()) == 72, "stable total work")
    completed = (r[:, 0] == 1) & (r[:, 5] <= h[1])
    throughput = (
        int(completed.sum())
        * case["iterations"]
        / 2048
        / ((int(h[1]) - int(h[0])) / 1e9)
    )
    # Union observed block intervals per SM. This is neither occupancy nor a
    # hardware SM-active counter. Denominator includes launch/dispatch gaps.
    flat = s.reshape(-1, 3)
    span = int(flat[:, 1].max()) - int(flat[:, 0].min())
    union = 0
    for sm in np.unique(flat[:, 2]):
        intervals = flat[flat[:, 2] == sm, :2]
        intervals = intervals[np.argsort(intervals[:, 0])].astype(np.int64)
        ends = np.maximum.accumulate(intervals[:, 1])
        gaps = np.maximum(0, intervals[1:, 0] - ends[:-1]).sum()
        union += int(ends[-1] - intervals[0, 0] - gaps)
    return dict(
        stable_ms=latency,
        reclaim_ms=reclaim,
        transition_ms=transition,
        gpu_handoff_gap_ms=gpu_transition,
        elastic_units_s=throughput,
        observed_sm_coverage=union / (span * pools[-1]["actual_sms"]),
        tail_drain_ms=(int(h[2]) - int(h[1])) / 1e6,
        dispatch_lateness_ms=((a[:, 1] - a[:, 0]) / 1e6).tolist(),
        elastic_completed=int(completed.sum()),
        stable_output=float(v[r[:, 0] == 0][0, 0]),
    )


def summarize(rows):
    result = {}
    for key in (
        "stable_ms",
        "reclaim_ms",
        "transition_ms",
        "gpu_handoff_gap_ms",
        "dispatch_lateness_ms",
    ):
        values = [v for row in rows for v in row[key]]
        result[key] = {
            f"p{q}": float(np.percentile(values, q)) if values else None
            for q in (50, 95, 99)
        }
    result["stable_miss_fraction_4ms"] = float(
        np.mean([v > 4 for row in rows for v in row["stable_ms"]])
    )
    for key in ("elastic_units_s", "observed_sm_coverage", "tail_drain_ms"):
        result[key] = float(np.mean([row[key] for row in rows]))
    result["stable_samples"] = 16 * len(rows)
    result["reclaim_samples"] = sum(len(row["reclaim_ms"]) for row in rows)
    result["transition_samples"] = sum(len(row["transition_ms"]) for row in rows)
    return result


def replay(root):
    checksummed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        require(name not in checksummed, "duplicate checksum")
        checksummed.add(name)
        path = root / name
        require(path.is_relative_to(root) and ".." not in path.parts, "checksum path")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == checksum,
            "checksum mismatch",
        )
    require(
        checksummed
        == {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        },
        "checksum coverage",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    require(result["status"] == "validated_synthetic_routing", "unvalidated run")
    for name, checksum in manifest["source_sha256"].items():
        require(
            hashlib.sha256((root / "execution_sources" / name).read_bytes()).hexdigest()
            == checksum,
            "source mismatch",
        )
    args = manifest["args"]
    pools = result["pools"]
    for pool in pools:
        require(
            pool["sm_ids_before"] == pool["sm_ids_after"]
            and len(pool["sm_ids_before"]) == pool["actual_sms"],
            "pool identity",
        )
    for x, y in itertools.combinations(pools[:3], 2):
        require(not set(x["sm_ids_before"]) & set(y["sm_ids_before"]), "pool overlap")
    expected = set(
        itertools.product(
            range(args["repeats"]), POLICIES, (1, 4), args["iterations"], (0, 1)
        )
    )
    seen = set()
    stable_output = None
    raw_seen = set()
    for case in result["summary"]:
        key = tuple(
            case[k] for k in ("repeat", "policy", "window", "iterations", "swap")
        )
        require(key in expected and key not in seen, "condition coverage")
        seen.add(key)
        expected_pools = (
            [case["swap"], 1 - case["swap"], 2]
            if POLICIES.index(case["policy"]) < 5
            else [3 if case["policy"] == "shared_24" else 4] * 3
        )
        require(case["lane_pools"] == expected_pools, "lane mapping")
        require(
            len(case["raw"]) == args["samples"]
            and len(set(case["raw"])) == args["samples"],
            "sample coverage",
        )
        rows = []
        for name in case["raw"]:
            require(name not in raw_seen and name in checksummed, "raw coverage")
            raw_seen.add(name)
            with np.load(root / name, allow_pickle=False) as raw:
                row = derive(case, raw, pools, args["period_us"] * 1000)
            if stable_output is None:
                stable_output = row["stable_output"]
            require(row["stable_output"] == stable_output, "cross-policy stable output")
            rows.append(row)
        require(summarize(rows) == case["metrics"], "saved metrics differ")
    require(seen == expected, "incomplete sweep")
    return dict(
        archive=str(root),
        manifest=manifest,
        result=result,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def table(record):
    cases = record["result"]["summary"]
    lines = [
        "Ranges across repetition/role summaries; descriptive, not confidence intervals.",
        "",
        "| Policy | Iterations | Window | Stable p99 ms | Reclaim p99 ms | Route p99 ms | Elastic units/s | Observed SM coverage % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, iterations, window in itertools.product(
        POLICIES, record["manifest"]["args"]["iterations"], (1, 4)
    ):
        metrics = [
            c["metrics"]
            for c in cases
            if (c["policy"], c["iterations"], c["window"])
            == (policy, iterations, window)
        ]

        def span(key, sub=None, scale=1):
            values = [m[key][sub] if sub else m[key] for m in metrics]
            values = [v * scale for v in values if v is not None]
            return f"{min(values):.3f}–{max(values):.3f}" if values else "—"

        lines.append(
            f"| {policy} | {iterations} | {window} | {span('stable_ms', 'p99')} | {span('reclaim_ms', 'p99')} | {span('transition_ms', 'p99')} | {span('elastic_units_s')} | {span('observed_sm_coverage', scale=100)} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    record = replay(args.archive.resolve())
    args.json_output.write_text(json.dumps(record, indent=2) + "\n")
    args.markdown_output.write_text(table(record))
    print(f"Validated {len(record['result']['summary'])} conditions")


if __name__ == "__main__":
    main()
