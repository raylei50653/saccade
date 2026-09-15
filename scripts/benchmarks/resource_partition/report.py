"""Build the machine-readable phase-B study record and its markdown tables."""

# status: diagnostic
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256_manifest(root):
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.file_digest(
                        path.open("rb"), "sha256"
                    ).hexdigest(),
                }
            )
    return entries


def level_key(row):
    return (row["actual_sm_count"], row["partition_kind"] != "green_context")


def label(row):
    if row["partition_kind"] == "green_context":
        return f"{row['actual_sm_count']} (green)"
    return f"{row['actual_sm_count']} (primary)"


def markdown_tables(summary, proxy=None):
    lines = []
    rows = sorted(summary["rows"], key=lambda r: (level_key(r), r["rep"], r["mode"]))
    pairs = sorted(summary["pairs"], key=lambda p: (p["actual_sm_count"], p["rep"]))
    lines.append("### Throughput and paired DB gain per verified budget\n")
    lines.append(
        "| Actual SMs | Rep | Serial FPS | DB FPS | DB gain | Serial rel. to 46 (primary) | DB rel. to 46 (primary) | Validated |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    baseline = {p["rep"]: p for p in pairs if p["partition_kind"] != "green_context"}
    for p in pairs:
        b = baseline[p["rep"]]
        kind = "green" if p["partition_kind"] == "green_context" else "primary"
        flag = (
            "yes"
            if p["true_partition_validated"]
            else ("n/a (baseline)" if kind == "primary" else "**no**")
        )
        lines.append(
            f"| {p['actual_sm_count']} ({kind}) | {p['rep'] + 1} | {p['serial_fps']:.2f} | {p['double_fps']:.2f} | "
            f"{p['db_gain']:.3f} | {p['serial_fps'] / b['serial_fps']:.3f} | {p['double_fps'] / b['double_fps']:.3f} | {flag} |"
        )
    lines.append("\n### Same-run latency, period and jitter (ms; p50 / p95 / p99)\n")
    lines.append("| Actual SMs | Rep | Mode | Frame latency | Period | Period σ |")
    lines.append("|---|---:|---|---|---|---:|")
    for r in rows:
        lat = r["latency"]
        per = r["period"]
        lines.append(
            f"| {label(r)} | {r['rep'] + 1} | {r['mode']} | "
            f"{lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} / {lat['p99_ms']:.2f} | "
            f"{per['p50_ms']:.2f} / {per['p95_ms']:.2f} / {per['p99_ms']:.2f} | {per['std_ms']:.2f} |"
        )
    lines.append("\n### Routing verification per measured point\n")
    lines.append(
        "| Point | Owned streams | Thread switches | set_device re-asserts | Probe SMs (direct/graph, before→after) | Twin kernels in ctx / total | Twin graph kernels | Twin escapes | Output = twin | Validated |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|---|")
    for r in rows:
        ev = r["routing_evidence"]
        tw = r["audit_twin"]
        w = next(iter(tw["routing_evidence"]["audit_windows"].values()))["sequence"]
        probe = ev["probe_sm_ids"]
        probe_text = (
            f"{len(probe['before']['direct_sm_ids'])}/{len(probe['before']['graph_sm_ids'])}"
            f"→{len(probe['after']['direct_sm_ids'])}/{len(probe['after']['graph_sm_ids'])}"
        )
        lines.append(
            f"| `{r['name']}` | {len(ev['owned_streams'])} | {ev['thread_context_switches']} | {ev['set_device_reasserts']} | {probe_text} | "
            f"{w['kernels']['in_execution_context']} / {w['kernels']['total']} | {w['kernels']['graph_launched']} | {w['kernels']['outside']} | "
            f"{'yes' if tw['consistency_with_measurement']['output_sha256_equal'] else 'no'} | "
            f"{'yes' if r['true_partition_validated'] else ('baseline' if r['level'] == 'full' else '**no**')} |"
        )
    lines.append("\n### Audit twin overhead (measurement FPS / audited-twin FPS)\n")
    lines.append("| Actual SMs | Mode | Rep | Ratio |")
    lines.append("|---|---|---:|---:|")
    for r in rows:
        lines.append(
            f"| {label(r)} | {r['mode']} | {r['rep'] + 1} | {r['audit_twin']['fps_measurement_over_audit']:.3f} |"
        )
    lines.append(
        "\n### Sampled telemetry inside measured windows (ranges of per-run means)\n"
    )
    lines.append(
        "| Actual SMs | Mean power (W) | Mean SM clock (MHz) | Mean GPU utilization (%) | Samples per run |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for key in sorted({level_key(r) for r in rows}):
        group = [r for r in rows if level_key(r) == key]

        def span(field):
            vals = [
                r["telemetry"]["values"][field]["mean"]
                for r in group
                if field in r["telemetry"]["values"]
            ]
            return f"{min(vals):.1f}–{max(vals):.1f}" if vals else "n/a"

        samples = sorted(r["telemetry"]["samples"] for r in group)
        lines.append(
            f"| {label(group[0])} | {span('power.draw [W]')} | {span('clocks.current.sm [MHz]')} | {span('utilization.gpu [%]')} | {samples[0]}–{samples[-1]} |"
        )
    if proxy:
        lines.append(
            "\n### Proxy (phase A, K residency blocks) versus true partition (phase B, actual SMs)\n"
        )
        lines.append(
            "| Surface | Setting | Serial FPS (mean of reps) | DB FPS (mean) | DB gain (mean) |"
        )
        lines.append("|---|---|---:|---:|---:|")
        by_k = {}
        for p in proxy["main"]["pairs"]:
            by_k.setdefault(p["blocks"], []).append(p)
        for k, ps in sorted(by_k.items()):
            lines.append(
                f"| proxy 5 ms pulses | K={k} | {statistics.mean(x['serial_fps'] for x in ps):.2f} | "
                f"{statistics.mean(x['double_fps'] for x in ps):.2f} | {statistics.mean(x['db_gain'] for x in ps):.3f} |"
            )
        by_level = {}
        for p in pairs:
            by_level.setdefault((p["actual_sm_count"], p["partition_kind"]), []).append(
                p
            )
        for (sm, kind), ps in sorted(
            by_level.items(), key=lambda kv: (-kv[0][0], kv[0][1])
        ):
            name = "primary" if kind != "green_context" else "green"
            lines.append(
                f"| true partition | {sm} SM ({name}) | {statistics.mean(x['serial_fps'] for x in ps):.2f} | "
                f"{statistics.mean(x['double_fps'] for x in ps):.2f} | {statistics.mean(x['db_gain'] for x in ps):.3f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="completed sweep directory")
    parser.add_argument("--proxy-record", type=Path, help="phase-A study JSON")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    sweep = json.loads((args.root / "sweep.json").read_text())
    proxy = json.loads(args.proxy_record.read_text()) if args.proxy_record else None
    identity_before = json.loads((args.root / "input_identity_before.json").read_text())
    identity_after = json.loads((args.root / "input_identity_after.json").read_text())
    tables = markdown_tables(summary, proxy)
    # Keep the committed record compact: SM-id lists become counts here; the
    # raw archive retains the full lists.
    for row in summary["rows"]:
        for holder in (row["routing_evidence"], row["audit_twin"]["routing_evidence"]):
            probe = holder.pop("probe_sm_ids", None)
            if probe:
                holder["probe_sm_counts"] = {
                    phase: {k: len(v) for k, v in ids.items()}
                    for phase, ids in probe.items()
                }
    record = {
        "schema": "saccade-partition-study-v1",
        "kind": summary["kind"],
        "head": sweep["head"],
        "arguments": sweep["arguments"],
        "source_sha256": sweep["source_sha256"],
        "library_sha256": sweep["library_sha256"],
        "green_context_probe": json.loads((args.root / "green_probe.json").read_text()),
        "summary": summary,
        "input_identity_before_equals_after": identity_before["files"]
        == identity_after["files"],
        "proxy_record": str(args.proxy_record) if args.proxy_record else None,
        "raw_artifact_root": str(args.root),
        "raw_sha256_manifest": sha256_manifest(args.root),
    }
    args.json_output.write_text(json.dumps(record, indent=2) + "\n")
    args.markdown_output.write_text(tables)


if __name__ == "__main__":
    main()
