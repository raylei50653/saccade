"""Replay time-budget ledgers, calibration, repeated arrivals and measured frontiers."""

# status: diagnostic
import argparse
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .admission_report import require, summarize
else:
    from admission_report import require, summarize

KEYS = ("repeat", "route", "window", "budget_ns", "swap")
ROUTES = ("disjoint", "shared_budget_priority", "full_priority")


def calibrate(arrays, sms):
    s, v = arrays["stamps"], arrays["values"]
    require(s.ndim == 3 and s.shape[1:] == (1024, 3), "calibration shape")
    require(
        np.all(s[:, :, 0] > 0) and np.all(s[:, :, 1] >= s[:, :, 0]),
        "calibration stamps",
    )
    require(set(map(int, s[:, :, 2].flat)) <= set(sms), "calibration SM escape")
    require(
        v.shape == s.shape[:2] and np.isfinite(v).all() and np.all(v == v[0, 0]),
        "calibration outputs",
    )
    durations = s[:, :, 1].max(axis=1) - s[:, :, 0].min(axis=1)
    return int(np.ceil(np.percentile(durations, 95) * 1.2))


def derive(case, arrays, stable_sms, burst_sms, workload, estimates, period):
    h, ch, a, s, v = (
        arrays[k] for k in ("host", "chunk_host", "arrivals", "stamps", "values")
    )
    n, m = len(workload), 4
    require(
        h.shape == (3,)
        and ch.shape == (n, 5)
        and a.shape == (m, 9)
        and s.shape == (m * 128 + n * 1024, 3)
        and v.shape == (len(s),),
        "raw shape",
    )
    h = list(map(int, h))
    require(h[0] <= h[2] <= h[1], "host order")
    require(int(ch[0, 1]) <= h[2], "first dispatch before start observation")
    require(
        np.all(ch[:, 0] >= h[0])
        and np.all(ch[:, 0] <= ch[:, 1])
        and np.all(ch[:, 1] <= ch[:, 2])
        and np.all(ch[:, 2] <= h[1]),
        "chunk order",
    )
    require(
        np.all(ch[1:, 0] >= ch[:-1, 1]) and np.all(ch[1:, 2] >= ch[:-1, 2]),
        "serial host order",
    )
    require(list(map(int, ch[:, 3])) == estimates, "estimate identity")
    for i in range(n):
        live = [j for j in range(i) if int(ch[j, 2]) > int(ch[i, 0])]
        ledger = sum(estimates[j] for j in live) + estimates[i]
        require(int(ch[i, 4]) == ledger, "ledger mismatch")
        require(
            (len(live) + 1 <= case["window"])
            if case["window"]
            else ledger <= case["budget_ns"],
            "admission bound",
        )
    require(np.all(s[:, 0] > 0) and np.all(s[:, 0] <= s[:, 1]), "GPU order")
    require(set(map(int, s[: m * 128, 2])) <= set(stable_sms), "stable SM escape")
    require(set(map(int, s[m * 128 :, 2])) <= set(burst_sms), "elastic SM escape")
    require(np.isfinite(v).all() and np.all(v[: m * 128] == v[0]), "stable output")
    burst = s[m * 128 :].reshape(n, 1024, 3)
    bv = v[m * 128 :].reshape(n, 1024)
    for kind in set(workload):
        vals = bv[np.asarray(workload) == kind]
        require(np.all(vals == vals[0, 0]), "elastic output")
    starts, ends = burst[:, :, 0].min(axis=1), burst[:, :, 1].max(axis=1)
    require(np.all(ends[:-1] <= starts[1:]), "elastic GPU stream order")
    require(int(s[: m * 128, 0].min()) >= int(burst[0, 0, 0]), "confirmed start order")
    rows = []
    previous_admitted, previous_resume = 1, h[2]
    for k, raw in enumerate(a):
        target, begin, launch_end, done, drain, admitted, retired, ledger, resume = map(
            int, raw
        )
        require(target == h[2] + (k + 1) * period, "arrival target")
        require(
            target <= begin <= launch_end <= done <= resume <= h[1]
            and launch_end <= drain <= resume
            and begin >= previous_resume,
            "arrival order",
        )
        require(0 <= retired <= admitted <= n, "arrival counts")
        require(
            sum(int(t) < begin for t in ch[:, 0]) == admitted
            and sum(int(t) < begin for t in ch[:, 2]) == retired,
            "arrival queue",
        )
        require(ledger == sum(estimates[retired:admitted]), "arrival ledger")
        require(all(int(t) >= resume for t in ch[admitted:, 0]), "freeze violation")
        require(admitted > 0 and int(ch[admitted - 1, 2]) <= drain, "drain order")
        replenished = max(0, admitted - max(previous_admitted, retired))
        # Count dispatches during this active interval that follow a retirement
        # from the same interval; excludes merely refilling after a pause.
        retired_since_resume = [
            j for j in range(admitted) if previous_resume <= int(ch[j, 2]) < begin
        ]
        rolling = sum(
            any(j < i and int(ch[j, 2]) <= int(ch[i, 0]) for j in retired_since_resume)
            for i in range(previous_admitted, admitted)
        )
        duration = (ends - starts).astype(float)
        rows.append(
            {
                "stable_response_ms": (done - begin) / 1e6,
                "stable_scheduled_response_ms": (done - target) / 1e6,
                "arrival_overshoot_ms": (begin - target) / 1e6,
                "drain_observed_ms": (drain - begin) / 1e6,
                "estimated_remaining_ms": ledger / 1e6,
                "drain_excess_ms": max(0, drain - begin - ledger) / 1e6,
                "drain_over_budget": int(
                    bool(case["budget_ns"]) and drain - begin > case["budget_ns"]
                ),
                "drain_over_estimate": int(drain - begin > ledger),
                "unretired_at_arrival": admitted - retired,
                "admitted_at_arrival": admitted,
                "retired_at_arrival": retired,
                "rolling_dispatches": rolling,
                "replenished_unretired": replenished,
                "work_exhausted": int(retired == n),
                "pause_ms": (resume - begin) / 1e6,
                "stable_launch_ms": (launch_end - begin) / 1e6,
                "elastic_dispatch_ms": float(np.sum(ch[:, 1] - ch[:, 0])) / 1e6,
                "trial_ms": (h[1] - h[0]) / 1e6,
                "elastic_equivalent_units_s": sum(workload)
                / 2048
                / ((h[1] - h[0]) / 1e9),
                "unit_estimate_exceed_fraction": float(np.mean(duration > estimates)),
                "unit_gpu_ms_p95": float(np.percentile(duration, 95)) / 1e6,
            }
        )
        previous_admitted, previous_resume = admitted, resume
    return rows


def checked_record(root):
    root = root.resolve()
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        p = (root / name).resolve()
        require(p.is_relative_to(root) and name not in listed, "checksum path")
        require(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest, "checksum mismatch"
        )
        listed.add(name)
    require(
        listed
        == {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        },
        "incomplete checksums",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    require(result["status"] == "validated_synthetic_time_budget", "unvalidated run")
    for name, digest in manifest["source_sha256"].items():
        require(
            hashlib.sha256((root / "execution_sources" / name).read_bytes()).hexdigest()
            == digest,
            "source identity",
        )
    args, pools = manifest["args"], result["pools"]
    require(len(pools) == 4, "pool count")
    for i, p in enumerate(pools):
        require(
            p["sm_ids_before"] == p["sm_ids_after"]
            and len(set(p["sm_ids_before"])) == p["actual_sms"]
            and p["actual_sms"]
            == (
                args["sms"]
                if i < 2
                else 2 * args["sms"]
                if i == 2
                else result["device_sms"]
            ),
            "pool identity",
        )
    require(
        not set(pools[0]["sm_ids_before"]) & set(pools[1]["sm_ids_before"]),
        "pool overlap",
    )

    def lanes(route, swap):
        return (
            (pools[swap], pools[1 - swap])
            if route == "disjoint"
            else (pools[2 if route == "shared_budget_priority" else 3],) * 2
        )

    estimates, outputs = {}, {}
    expected_cal = set(itertools.product(ROUTES, (0, 1), (512, 2048, 4096)))
    require(len(result["calibration"]) == len(expected_cal), "calibration count")
    raw_names = set()
    for row in result["calibration"]:
        key = row["route"], row["swap"], row["iterations"]
        require(key in expected_cal and key not in estimates, "calibration identity")
        require(
            row["raw"] in listed and row["raw"] not in raw_names, "calibration artifact"
        )
        raw_names.add(row["raw"])
        with np.load(root / row["raw"], allow_pickle=False) as raw:
            require(
                all(len(raw[k]) == args["calibration_samples"] for k in raw.files),
                "calibration samples",
            )
            estimate = calibrate(raw, lanes(*key[:2])[1]["sm_ids_before"])
            outputs[key] = float(raw["values"][0, 0])
        require(estimate == row["estimate_ns"], "calibration estimate")
        estimates[key] = estimate
    require(max(estimates.values()) <= min(args["budgets_ms"]) * 1e6, "oversize unit")
    policies = [(w, 0) for w in (1, 4, 16)] + [
        (0, b * 1000000) for b in args["budgets_ms"]
    ]
    expected = {
        (r, route, w, b, swap)
        for r, route, (w, b), swap in itertools.product(
            range(args["repeats"]), ROUTES, policies, (0, 1)
        )
    }
    require(
        len(result["summary"]) == len(expected)
        and {tuple(s[k] for k in KEYS) for s in result["summary"]} == expected,
        "condition coverage",
    )
    summaries = []
    for summary in result["summary"]:
        case = {k: summary[k] for k in KEYS}
        require(summary["samples"] == args["samples"], "sample count")
        require(
            summary["raw"] in listed and summary["raw"] not in raw_names, "raw artifact"
        )
        raw_names.add(summary["raw"])
        pa, pb = lanes(case["route"], case["swap"])
        rows = []
        with np.load(root / summary["raw"], allow_pickle=False) as raw:
            require(
                set(raw.files)
                == {"host", "chunk_host", "arrivals", "stamps", "values"},
                "raw fields",
            )
            cached = {k: raw[k] for k in raw.files}
            require(
                all(len(v) == args["samples"] for v in cached.values()),
                "raw sample count",
            )
            for sample in range(args["samples"]):
                workload = [512, 2048, 4096, 2048] * 64
                random.Random(args["seed"] + 10000 * case["repeat"] + sample).shuffle(
                    workload
                )
                est = [estimates[case["route"], case["swap"], i] for i in workload]
                arrays = {k: v[sample] for k, v in cached.items()}
                require(
                    np.all(arrays["values"][:512] == result["stable_value"]),
                    "cross-condition stable output",
                )
                bv = arrays["values"][512:].reshape(256, 1024)
                require(
                    all(
                        np.all(bv[i] == outputs[case["route"], case["swap"], kind])
                        for i, kind in enumerate(workload)
                    ),
                    "calibrated elastic output",
                )
                rows.extend(
                    derive(
                        case,
                        arrays,
                        pa["sm_ids_before"],
                        pb["sm_ids_before"],
                        workload,
                        est,
                        args["period_us"] * 1000,
                    )
                )
        recomputed = summarize(rows)
        require(
            set(summary) == set(KEYS) | {"raw", "samples"} | set(recomputed),
            "summary fields",
        )
        for metric, quantiles in recomputed.items():
            require(summary[metric] == quantiles, f"summary mismatch: {metric}")
        summaries.append(
            {
                **case,
                "samples": args["samples"],
                **recomputed,
                "counts": {
                    k: sum(r[k] for r in rows)
                    for k in (
                        "drain_over_budget",
                        "drain_over_estimate",
                        "work_exhausted",
                    )
                },
                "arrivals_with_rolling_dispatch": sum(
                    r["rolling_dispatches"] > 0 for r in rows
                ),
                "later_arrivals_with_rolling_dispatch": sum(
                    r["rolling_dispatches"] > 0 for i, r in enumerate(rows) if i % 4
                ),
                "arrivals": len(rows),
            }
        )
    return dict(
        archive=str(root),
        replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest=manifest,
        pools=pools,
        calibration=result["calibration"],
        summary=summaries,
    )


def markdown(record):
    groups = defaultdict(list)
    for r in record["summary"]:
        groups[r["route"], r["window"], r["budget_ns"]].append(r)
    lines = [
        "Ranges across repetition/role quantiles; p99 is descriptive over correlated arrivals.",
        "",
        "| Route | Policy | Stable p99 ms | Drain p99 ms | Units/s p50 | Dispatch ms p50 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (route, window, budget), rows in sorted(groups.items()):

        def span(key, q="p99"):
            v = [r[key][q] for r in rows]
            return f"{min(v):.3f}–{max(v):.3f}"

        policy = f"W={window}" if window else f"B={budget / 1e6:g} ms"
        lines.append(
            f"| {route} | {policy} | {span('stable_response_ms')} | {span('drain_observed_ms')} | {span('elastic_equivalent_units_s', 'p50')} | {span('elastic_dispatch_ms', 'p50')} |"
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
