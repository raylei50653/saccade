"""Verify sealed runs and compare paired mixed-workload service frontiers."""

# status: diagnostic
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmarks.resource_mixed.report import derive, digest  # noqa: E402


def summarize(directory):
    manifest = json.loads((directory / "sweep.json").read_text())
    if "finished_epoch" not in manifest:
        raise ValueError("incomplete sweep")
    for name, h in manifest["source_sha256"].items():
        if digest(directory / "sources" / name) != h:
            raise ValueError("source snapshot hash mismatch")
    for name, h in manifest["library_sha256"].items():
        if digest(directory / name) != h:
            raise ValueError("library hash mismatch")
    if (
        json.loads((directory / "input_before.json").read_text())["files"]
        != json.loads((directory / "input_after.json").read_text())["files"]
    ):
        raise ValueError("input identity mismatch")
    rows = []
    twins = {}
    seen = set()
    output_hashes = set()
    for entry in manifest["schedule"]:
        key = (
            entry["rep"],
            entry["policy"],
            entry["iterations"],
            entry["window"],
            entry["audit"],
        )
        if key in seen:
            raise ValueError("duplicate condition")
        seen.add(key)
        if entry.get("returncode") != 0 or not entry.get("valid"):
            raise ValueError("failed execution")
        for name, h in entry["files"].items():
            if digest(directory / name) != h:
                raise ValueError(f"raw hash mismatch: {name}")
        row = derive(directory / entry["name"])
        if row != json.loads((directory / entry["name"] / "report.json").read_text()):
            raise ValueError("saved metric mismatch")
        if not row["valid"]:
            raise ValueError("replay failed")
        point = json.loads((directory / entry["name"] / "point.json").read_text())
        for k in ("policy", "window", "iterations", "audit"):
            if point["arguments"][k] != entry[k]:
                raise ValueError("condition identity mismatch")
        if point["arguments"]["frames"] != manifest["arguments"]["frames"]:
            raise ValueError("frame count differs from frozen sweep")
        row.update(rep=entry["rep"], name=entry["name"])
        identity = (entry["policy"], entry["iterations"], entry["window"])
        owner = point["owner"]
        signature = dict(
            output=row["output_sha256"],
            actual_sms=owner["actual_sm_count"],
            priorities=[s["priority"] for s in owner["streams"]],
            thread_switches=owner["thread_context_switches"],
            set_device_reasserts=owner["set_device_reasserts"],
            stable_probe=owner["probe"],
            elastic_probe=point["elastic_pool"]["probe"],
        )
        if entry["audit"]:
            twins[identity] = signature
        elif identity not in twins or signature != twins[identity]:
            raise ValueError(f"audited condition twin mismatch: {entry['name']}")
        output_hashes.add(json.dumps(row["output_sha256"], sort_keys=True))
        rows.append(row)
    a = manifest["arguments"]
    expected = {
        (r, p, i, w, audit)
        for r in range(a["repeats"])
        for p in a["policies"]
        for i in a["iterations"]
        for w in a["windows"]
        for audit in ([True, False] if r == 0 else [False])
    }
    expected.update(
        (r, "control", 2048, 1, audit)
        for r in range(a["repeats"])
        for audit in ([True, False] if r == 0 else [False])
    )
    if seen != expected or len(output_hashes) != 1:
        raise ValueError("coverage or MOT output mismatch")
    comparisons = []
    measurements = [r for r in rows if not r["audit"]]
    for dynamic in measurements:
        if dynamic["policy"] != "dynamic":
            continue
        controls = {
            r["policy"]: r
            for r in measurements
            if r["rep"] == dynamic["rep"]
            and r["iterations"] == dynamic["iterations"]
            and r["window"] == dynamic["window"]
        }
        for policy in ("fixed", "shared", "headroom"):
            if policy not in controls:
                continue
            other = controls[policy]
            comparisons.append(
                dict(
                    rep=dynamic["rep"],
                    iterations=dynamic["iterations"],
                    window=dynamic["window"],
                    comparison=policy,
                    dynamic_service_pass=dynamic["service_pass"],
                    other_service_pass=other["service_pass"],
                    elastic_throughput_ratio=dynamic["elastic_units_s"]
                    / other["elastic_units_s"],
                    burst_p95_completion_ratio=dynamic["burst_completion_ms"]["p95"]
                    / other["burst_completion_ms"]["p95"],
                    all_done_ratio=dynamic["elastic_all_done_s"]
                    / other["elastic_all_done_s"],
                    stable_p99_delta_ms=dynamic["response_ms"]["p99"]
                    - other["response_ms"]["p99"],
                )
            )
    return dict(
        schema="saccade-mixed-summary-v1",
        archive=str(directory),
        sweep_sha256=digest(directory / "sweep.json"),
        source_head=manifest["head"],
        target=dict(fps=60, p99_ms=1000 / 60, miss_fraction=0.01),
        verified=True,
        rows=rows,
        comparisons=comparisons,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = summarize(args.directory)
    (args.output or args.directory / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        f"Verified {len(result['rows'])} runs; {len(result['comparisons'])} paired comparisons"
    )


if __name__ == "__main__":
    main()
