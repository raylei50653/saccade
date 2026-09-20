"""Freeze the serial/double-buffer SM-scaling benchmark and compare a later sweep against it."""

# status: diagnostic
#
# `build` turns a completed, validated sweep directory (sweep.json + summary.json)
# into a compact frozen benchmark record: per-budget aggregates over repetitions,
# the pre-declared knee verdict and the service-level frontier (minimum verified
# SM budget meeting each declared target). `compare` reads a frozen reference
# record and a candidate sweep (or record), fails closed unless the measured
# scope is identical, and reports per-budget ratios, DB-gain deltas, knee
# verdicts and frontier movement. Neither command invents a mechanism or a
# significance claim: every comparison is a ratio or a delta of observed means,
# with the repetition ranges kept beside it.
import argparse
import copy
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "saccade-partition-frozen-benchmark-v1"
COMPARISON_SCHEMA = "saccade-partition-benchmark-comparison-v1"

# Pre-declared reading rules. They were fixed in the closure evidence document
# before the closure sweep ran (see docs/reference/benchmarks/
# resource_scaling_closure_20260920.md, "Pre-declaration") and must not be
# edited after the fact; a different rule is a new schema version.
KNEE_SLOPE_FACTOR = 2.0
KNEE_MAX_RETAINED = 0.5
GRACEFUL_MIN_RETAINED = 0.8
NO_OVERLAP_MAX_GAIN = 1.05
DEFAULT_TARGETS = [
    {"name": "T1", "fps_min": 300.0, "frame_p99_max_ms": None},
    {"name": "T2", "fps_min": 200.0, "frame_p99_max_ms": 10.0},
    {"name": "T3", "fps_min": 150.0, "frame_p99_max_ms": 15.0},
    {"name": "T4", "fps_min": 100.0, "frame_p99_max_ms": 25.0},
]
CRITERIA = {
    "knee": {
        "curve": "mean DB gain over repetitions per validated Green Context budget",
        "retained_fraction": "(gain - 1) / (mean primary full-device gain - 1)",
        "slope_factor": KNEE_SLOPE_FACTOR,
        "max_retained_at_knee": KNEE_MAX_RETAINED,
        "graceful_min_retained": GRACEFUL_MIN_RETAINED,
        "no_overlap_max_gain": NO_OVERLAP_MAX_GAIN,
    },
    "frontier": {
        "rule": "every repetition of the budget meets fps_min and frame_p99_max_ms",
        "targets": DEFAULT_TARGETS,
    },
}
TELEMETRY_FIELDS = {
    "power_w": "power.draw [W]",
    "sm_clock_mhz": "clocks.current.sm [MHz]",
    "gpu_utilization_pct": "utilization.gpu [%]",
}
SCOPE_MUST_MATCH = (
    "preset",
    "sequences",
    "max_frames",
    "warmup_frames",
    "detect_barrier",
    "requested_levels",
    "device_name",
    "device_sm_count",
    "min_partition",
    "alignment",
)


def sha256_file(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def spread(values):
    values = [float(v) for v in values]
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "per_rep": values,
    }


def budget_key(row):
    return (row["actual_sm_count"], row["partition_kind"] != "green_context")


def budget_label(sm, kind):
    return f"{sm} ({'green' if kind == 'green_context' else 'primary'})"


def scope_from_sweep(sweep, summary):
    args = sweep["arguments"]
    device_name = None
    match = re.search(r"Product Name\s*:\s*(.+)", sweep.get("gpu", ""))
    if match:
        device_name = match.group(1).strip()
    driver = None
    match = re.search(r"KMD Version\s*:\s*(\S+)", sweep.get("gpu", ""))
    if match:
        driver = match.group(1)
    any_row = summary["rows"][0]
    green = any_row["routing_evidence"]["green_context"]
    return {
        "preset": args["preset"],
        "sequences": args["sequences"],
        "max_frames": args["max_frames"],
        "warmup_frames": 50,
        "detect_barrier": "event",
        "requested_levels": sorted(str(level) for level in args["levels"]),
        "repeats": args["repeats"],
        "seed": args["seed"],
        "device_name": device_name,
        "device_sm_count": any_row["device_sm_count"],
        "min_partition": (green or {}).get("min_partition"),
        "alignment": (green or {}).get("alignment"),
        "driver_kmd": driver,
        "toolchain": sweep.get("toolchain"),
        "head": sweep["head"],
        "source_sha256": sweep["source_sha256"],
        "library_sha256": sweep["library_sha256"],
    }


def aggregate_budgets(summary):
    rows = [r for r in summary["rows"] if r["curve_eligible"]]
    pairs = [p for p in summary["pairs"] if p["curve_eligible"]]
    budgets = []
    for key in sorted({budget_key(r) for r in rows}):
        group = [r for r in rows if budget_key(r) == key]
        sm, kind = group[0]["actual_sm_count"], group[0]["partition_kind"]
        reps = sorted({r["rep"] for r in group})
        modes = {}
        for mode in ("serial", "double"):
            selected = sorted(
                (r for r in group if r["mode"] == mode), key=lambda r: r["rep"]
            )
            if [r["rep"] for r in selected] != reps:
                raise ValueError(f"budget {sm} {kind} lacks a {mode} run per rep")
            telemetry = {}
            for name, field in TELEMETRY_FIELDS.items():
                means = [
                    r["telemetry"]["values"][field]["mean"]
                    for r in selected
                    if field in r["telemetry"]["values"]
                ]
                telemetry[name] = spread(means) if means else None
            telemetry["samples_per_run"] = [r["telemetry"]["samples"] for r in selected]
            modes[mode] = {
                "fps": spread(r["fps"] for r in selected),
                "frame_p50_ms": spread(r["latency"]["p50_ms"] for r in selected),
                "frame_p95_ms": spread(r["latency"]["p95_ms"] for r in selected),
                "frame_p99_ms": spread(r["latency"]["p99_ms"] for r in selected),
                "period_p50_ms": spread(r["period"]["p50_ms"] for r in selected),
                "period_p95_ms": spread(r["period"]["p95_ms"] for r in selected),
                "period_p99_ms": spread(r["period"]["p99_ms"] for r in selected),
                "period_std_ms": spread(r["period"]["std_ms"] for r in selected),
                "frames_per_run": [r["frames"] for r in selected],
                "output_sha256": sorted(
                    {v for r in selected for v in r["output_sha256"].values()}
                ),
                "telemetry": telemetry,
            }
        selected_pairs = sorted(
            (
                p
                for p in pairs
                if p["actual_sm_count"] == sm and p["partition_kind"] == kind
            ),
            key=lambda p: p["rep"],
        )
        if [p["rep"] for p in selected_pairs] != reps:
            raise ValueError(f"budget {sm} {kind} lacks a paired DB gain per rep")
        budgets.append(
            {
                "actual_sm_count": sm,
                "partition_kind": kind,
                "label": budget_label(sm, kind),
                "reps": reps,
                "true_partition_validated": all(
                    r["true_partition_validated"] for r in group
                ),
                "serial": modes["serial"],
                "double": modes["double"],
                "db_gain": spread(p["db_gain"] for p in selected_pairs),
            }
        )
    return budgets


def knee_verdict(green_gains, full_gain):
    """Apply the pre-declared knee rule to one curve of (actual SMs, DB gain)
    ordered by ascending SM count against one full-device gain."""
    if len(green_gains) < 3:
        return {"verdict": "insufficient_budgets", "segments": []}
    if full_gain <= 1.0:
        return {"verdict": "no_full_device_gain", "segments": []}
    retained = [(sm, (g - 1.0) / (full_gain - 1.0)) for sm, g in green_gains]
    segments = []
    for (lo_sm, lo_f), (hi_sm, hi_f) in zip(retained, retained[1:], strict=False):
        segments.append(
            {
                "from_sm": lo_sm,
                "to_sm": hi_sm,
                "retained_at_from": lo_f,
                "retained_at_to": hi_f,
                "retained_slope_per_sm": (hi_f - lo_f) / (hi_sm - lo_sm),
            }
        )
    knee = None
    for i, seg in enumerate(segments[:-1]):
        higher = max(s["retained_slope_per_sm"] for s in segments[i + 1 :])
        steep = seg["retained_slope_per_sm"] >= KNEE_SLOPE_FACTOR * max(higher, 0.0)
        if higher <= 0.0:
            steep = seg["retained_slope_per_sm"] > 0.0
        if steep and seg["retained_at_from"] < KNEE_MAX_RETAINED:
            knee = seg["from_sm"]
            break
    no_overlap = [sm for sm, g in green_gains if g <= NO_OVERLAP_MAX_GAIN]
    if knee is not None:
        verdict = "knee"
    elif all(f >= GRACEFUL_MIN_RETAINED for _, f in retained):
        verdict = "graceful_scaling"
    else:
        verdict = "smooth_decline"
    return {
        "verdict": verdict,
        "knee_sm": knee,
        "full_device_gain": full_gain,
        "retained_fraction": {str(sm): f for sm, f in retained},
        "segments": segments,
        "no_overlap_budgets": no_overlap,
    }


def knee_analysis(budgets):
    green = [b for b in budgets if b["partition_kind"] == "green_context"]
    primary = [b for b in budgets if b["partition_kind"] != "green_context"]
    if not primary:
        return {"verdict": "no_primary_baseline", "per_rep": []}
    full = primary[0]
    curve = [(b["actual_sm_count"], b["db_gain"]["mean"]) for b in green]
    result = knee_verdict(curve, full["db_gain"]["mean"])
    per_rep = []
    for i, rep in enumerate(full["reps"]):
        rep_curve = [(b["actual_sm_count"], b["db_gain"]["per_rep"][i]) for b in green]
        per_rep.append(
            {
                "rep": rep,
                **{
                    k: v
                    for k, v in knee_verdict(
                        rep_curve, full["db_gain"]["per_rep"][i]
                    ).items()
                    if k in ("verdict", "knee_sm", "no_overlap_budgets")
                },
            }
        )
    result["per_rep"] = per_rep
    result["verdict_agrees_across_reps"] = all(
        r["verdict"] == result["verdict"] and r.get("knee_sm") == result.get("knee_sm")
        for r in per_rep
    )
    return result


def frontier(budgets, targets):
    out = []
    for target in targets:
        entry = {
            "name": target["name"],
            **{k: target[k] for k in target if k != "name"},
        }
        for mode in ("serial", "double"):
            per_budget = {}
            minimum = None
            full_meets = None
            for b in budgets:
                fps = b[mode]["fps"]["per_rep"]
                p99 = b[mode]["frame_p99_ms"]["per_rep"]
                meets = [
                    f >= target["fps_min"]
                    and (
                        target["frame_p99_max_ms"] is None
                        or p <= target["frame_p99_max_ms"]
                    )
                    for f, p in zip(fps, p99, strict=True)
                ]
                per_budget[b["label"]] = {
                    "reps_meeting": sum(meets),
                    "reps": len(meets),
                    "all_reps_meet": all(meets),
                }
                if b["partition_kind"] != "green_context":
                    full_meets = all(meets)
                elif all(meets) and b["true_partition_validated"]:
                    if minimum is None or b["actual_sm_count"] < minimum:
                        minimum = b["actual_sm_count"]
            entry[mode] = {
                "min_verified_sm": minimum,
                "full_device_meets": full_meets,
                "per_budget": per_budget,
            }
        out.append(entry)
    return out


def build(root, study_record=None, targets=None):
    sweep = json.loads((root / "sweep.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if sweep["kind"] != "green_context_true_partition":
        raise ValueError("not a true-partition sweep")
    if summary["schema"] != "saccade-partition-summary-v1":
        raise ValueError("unexpected summary schema")
    if summary["points_not_validated"]:
        raise ValueError(
            f"sweep has unvalidated points: {summary['points_not_validated']}"
        )
    budgets = aggregate_budgets(summary)
    green = [b for b in budgets if b["partition_kind"] == "green_context"]
    if len(green) < 4 or not any(
        b["partition_kind"] != "green_context" for b in budgets
    ):
        raise ValueError(
            "frozen benchmark needs the primary baseline and at least four "
            "validated Green Context budgets"
        )
    targets = targets or DEFAULT_TARGETS
    record = {
        "schema": SCHEMA,
        "kind": "serial_vs_double_buffer_sm_scaling",
        "scope": scope_from_sweep(sweep, summary),
        "criteria": {
            **copy.deepcopy(CRITERIA),
            "frontier": {**copy.deepcopy(CRITERIA["frontier"]), "targets": targets},
        },
        "budgets": budgets,
        "knee": knee_analysis(budgets),
        "frontier": frontier(budgets, targets),
        "all_outputs_byte_identical": len(
            {
                sha
                for b in budgets
                for mode in ("serial", "double")
                for sha in b[mode]["output_sha256"]
            }
        )
        == 1,
        "provenance": {
            "raw_artifact_root": str(root),
            "summary_sha256": sha256_file(root / "summary.json"),
            "sweep_sha256": sha256_file(root / "sweep.json"),
            "study_record": str(study_record) if study_record else None,
            "study_record_sha256": sha256_file(study_record) if study_record else None,
        },
    }
    return record


def scope_check(reference, candidate):
    mismatches = []
    for key in SCOPE_MUST_MATCH:
        if reference["scope"].get(key) != candidate["scope"].get(key):
            mismatches.append(
                {
                    "field": key,
                    "reference": reference["scope"].get(key),
                    "candidate": candidate["scope"].get(key),
                }
            )
    if reference["criteria"] != candidate["criteria"]:
        mismatches.append(
            {"field": "criteria", "reference": "differs", "candidate": "differs"}
        )
    return mismatches


def ratio(a, b):
    return a / b if b else None


def compare_budgets(reference, candidate):
    ref_by = {b["label"]: b for b in reference["budgets"]}
    rows = []
    for cand in candidate["budgets"]:
        ref = ref_by.get(cand["label"])
        if ref is None:
            continue
        row = {"label": cand["label"], "actual_sm_count": cand["actual_sm_count"]}
        for mode in ("serial", "double"):
            row[mode] = {}
            for metric in (
                "fps",
                "frame_p50_ms",
                "frame_p95_ms",
                "frame_p99_ms",
                "period_p99_ms",
                "period_std_ms",
            ):
                r, c = ref[mode][metric], cand[mode][metric]
                row[mode][metric] = {
                    "reference_mean": r["mean"],
                    "candidate_mean": c["mean"],
                    "ratio_candidate_over_reference": ratio(c["mean"], r["mean"]),
                    "rep_ranges_overlap": not (
                        c["max"] < r["min"] or c["min"] > r["max"]
                    ),
                }
        row["db_gain"] = {
            "reference_mean": ref["db_gain"]["mean"],
            "candidate_mean": cand["db_gain"]["mean"],
            "delta": cand["db_gain"]["mean"] - ref["db_gain"]["mean"],
            "rep_ranges_overlap": not (
                cand["db_gain"]["max"] < ref["db_gain"]["min"]
                or cand["db_gain"]["min"] > ref["db_gain"]["max"]
            ),
        }
        rows.append(row)
    return rows


def compare_frontier(reference, candidate):
    out = []
    for ref_t, cand_t in zip(reference["frontier"], candidate["frontier"], strict=True):
        entry = {"name": ref_t["name"]}
        for mode in ("serial", "double"):
            before = ref_t[mode]["min_verified_sm"]
            after = cand_t[mode]["min_verified_sm"]
            entry[mode] = {
                "reference_min_verified_sm": before,
                "candidate_min_verified_sm": after,
                "delta_sm": (after - before)
                if before is not None and after is not None
                else None,
                "reference_full_device_meets": ref_t[mode]["full_device_meets"],
                "candidate_full_device_meets": cand_t[mode]["full_device_meets"],
            }
        out.append(entry)
    return out


def compare(reference, candidate, control=None):
    mismatches = scope_check(reference, candidate)
    result = {
        "schema": COMPARISON_SCHEMA,
        "comparable": not mismatches,
        "scope_mismatches": mismatches,
        "recorded_differences": {
            key: {
                "reference": reference["scope"].get(key),
                "candidate": candidate["scope"].get(key),
            }
            for key in (
                "head",
                "source_sha256",
                "library_sha256",
                "driver_kmd",
                "toolchain",
                "repeats",
            )
            if reference["scope"].get(key) != candidate["scope"].get(key)
        },
        "reference_provenance": reference["provenance"],
        "candidate_provenance": candidate["provenance"],
    }
    if mismatches:
        return result
    result["budgets"] = compare_budgets(reference, candidate)
    result["knee"] = {
        "reference": {k: reference["knee"].get(k) for k in ("verdict", "knee_sm")},
        "candidate": {k: candidate["knee"].get(k) for k in ("verdict", "knee_sm")},
    }
    result["frontier"] = compare_frontier(reference, candidate)
    result["outputs"] = {
        "reference_byte_identical": reference["all_outputs_byte_identical"],
        "candidate_byte_identical": candidate["all_outputs_byte_identical"],
        "candidate_equals_reference": sorted(
            {
                s
                for b in reference["budgets"]
                for m in ("serial", "double")
                for s in b[m]["output_sha256"]
            }
        )
        == sorted(
            {
                s
                for b in candidate["budgets"]
                for m in ("serial", "double")
                for s in b[m]["output_sha256"]
            }
        ),
    }
    if control is None:
        result["host_drift"] = None
        result["host_drift_note"] = (
            "no same-session control of the reference source was supplied; "
            "cross-session absolute ratios include unmeasured host drift"
        )
    else:
        control_mismatches = scope_check(reference, control)
        result["host_drift"] = {
            "comparable": not control_mismatches,
            "scope_mismatches": control_mismatches,
            "control_head_equals_reference": control["scope"]["head"]
            == reference["scope"]["head"],
            "control_source_equals_reference": control["scope"]["source_sha256"]
            == reference["scope"]["source_sha256"],
        }
        if not control_mismatches:
            result["host_drift"]["control_over_reference"] = compare_budgets(
                reference, control
            )
            result["host_drift"]["candidate_over_control"] = compare_budgets(
                control, candidate
            )
    return result


def build_markdown(record):
    lines = []
    lines.append("### Per-budget aggregates over repetitions (mean [min–max])\n")
    lines.append(
        "| Actual SMs | Reps | Serial FPS | DB FPS | DB gain | Retained gain | Serial frame p50/p95/p99 (ms) | DB frame p50/p95/p99 (ms) | Serial period σ (ms) | DB period σ (ms) | Validated |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|")
    retained = record["knee"].get("retained_fraction", {})

    def rng(s, fmt=".2f"):
        return f"{s['mean']:{fmt}} [{s['min']:{fmt}}–{s['max']:{fmt}}]"

    def pcts(m):
        return " / ".join(
            f"{m[k]['mean']:.2f}"
            for k in ("frame_p50_ms", "frame_p95_ms", "frame_p99_ms")
        )

    for b in record["budgets"]:
        rf = retained.get(str(b["actual_sm_count"]))
        rf_text = (
            f"{rf:.2f}"
            if rf is not None and b["partition_kind"] == "green_context"
            else "1.00 (ref)"
            if b["partition_kind"] != "green_context"
            else "n/a"
        )
        flag = (
            "yes"
            if b["true_partition_validated"]
            else (
                "n/a (baseline)" if b["partition_kind"] != "green_context" else "**no**"
            )
        )
        lines.append(
            f"| {b['label']} | {len(b['reps'])} | {rng(b['serial']['fps'])} | {rng(b['double']['fps'])} | "
            f"{rng(b['db_gain'], '.3f')} | {rf_text} | {pcts(b['serial'])} | {pcts(b['double'])} | "
            f"{rng(b['serial']['period_std_ms'])} | {rng(b['double']['period_std_ms'])} | {flag} |"
        )
    lines.append(
        "\n### Same-run completion period (ms; mean of per-rep p50 / p95 / p99)\n"
    )
    lines.append(
        "| Actual SMs | Serial period | DB period | Serial frames/run | DB frames/run |"
    )
    lines.append("|---|---|---|---|---|")
    for b in record["budgets"]:

        def per(m):
            return " / ".join(
                f"{b[m][k]['mean']:.2f}"
                for k in ("period_p50_ms", "period_p95_ms", "period_p99_ms")
            )

        lines.append(
            f"| {b['label']} | {per('serial')} | {per('double')} | "
            f"{'/'.join(str(f) for f in b['serial']['frames_per_run'])} | {'/'.join(str(f) for f in b['double']['frames_per_run'])} |"
        )
    lines.append(
        "\n### Sampled telemetry inside measured windows (mean of per-run means [min–max])\n"
    )
    lines.append(
        "| Actual SMs | Mode | Power (W) | SM clock (MHz) | GPU utilization (%) | Samples per run |"
    )
    lines.append("|---|---|---|---|---|---|")
    for b in record["budgets"]:
        for mode in ("serial", "double"):
            t = b[mode]["telemetry"]

            def tel(name):
                return rng(t[name], ".1f") if t.get(name) else "n/a"

            lines.append(
                f"| {b['label']} | {mode} | {tel('power_w')} | {tel('sm_clock_mhz')} | {tel('gpu_utilization_pct')} | "
                f"{min(t['samples_per_run'])}–{max(t['samples_per_run'])} |"
            )
    knee = record["knee"]
    lines.append("\n### Knee reading (pre-declared rule)\n")
    lines.append(
        f"Verdict: **{knee['verdict']}**"
        + (f" at {knee['knee_sm']} SMs" if knee.get("knee_sm") is not None else "")
        + f"; per-repetition verdicts agree: {'yes' if knee.get('verdict_agrees_across_reps') else 'no'}"
        + f"; budgets with DB gain ≤ {NO_OVERLAP_MAX_GAIN}: {knee.get('no_overlap_budgets') or 'none'}.\n"
    )
    if knee.get("segments"):
        lines.append(
            "| Segment (SMs) | Retained at low end | Retained at high end | Retained slope per SM |"
        )
        lines.append("|---|---:|---:|---:|")
        for s in knee["segments"]:
            lines.append(
                f"| {s['from_sm']}→{s['to_sm']} | {s['retained_at_from']:.3f} | {s['retained_at_to']:.3f} | {s['retained_slope_per_sm']:.4f} |"
            )
        lines.append("")
        lines.append("| Rep | Verdict | Knee SMs |")
        lines.append("|---:|---|---|")
        for r in knee["per_rep"]:
            lines.append(f"| {r['rep'] + 1} | {r['verdict']} | {r.get('knee_sm')} |")
    lines.append(
        "\n### Service-level frontier (minimum verified SM budget; every repetition must meet the target)\n"
    )
    lines.append(
        "| Target | FPS ≥ | Frame p99 ≤ (ms) | Serial min SMs | Serial full device | DB min SMs | DB full device |"
    )
    lines.append("|---|---:|---:|---|---|---|---|")
    for t in record["frontier"]:
        p99 = t["frame_p99_max_ms"]
        lines.append(
            f"| {t['name']} | {t['fps_min']:.0f} | {p99 if p99 is not None else '—'} | "
            f"{t['serial']['min_verified_sm'] if t['serial']['min_verified_sm'] is not None else 'none'} | "
            f"{'meets' if t['serial']['full_device_meets'] else 'misses'} | "
            f"{t['double']['min_verified_sm'] if t['double']['min_verified_sm'] is not None else 'none'} | "
            f"{'meets' if t['double']['full_device_meets'] else 'misses'} |"
        )
    return "\n".join(lines) + "\n"


def comparison_markdown(result):
    lines = []
    if not result["comparable"]:
        lines.append("**Not comparable — scope mismatch:**\n")
        for m in result["scope_mismatches"]:
            lines.append(
                f"- `{m['field']}`: reference `{m['reference']}` vs candidate `{m['candidate']}`"
            )
        return "\n".join(lines) + "\n"
    lines.append(
        "### Candidate / reference per budget (means; ranges overlap = repetition min–max intervals intersect)\n"
    )
    lines.append(
        "| Actual SMs | Serial FPS ratio | DB FPS ratio | Serial p99 ratio | DB p99 ratio | Serial period σ ratio | DB period σ ratio | DB gain Δ | DB gain ranges overlap |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for b in result["budgets"]:

        def r(mode, metric):
            v = b[mode][metric]["ratio_candidate_over_reference"]
            return f"{v:.3f}" if v is not None else "n/a"

        lines.append(
            f"| {b['label']} | {r('serial', 'fps')} | {r('double', 'fps')} | {r('serial', 'frame_p99_ms')} | {r('double', 'frame_p99_ms')} | "
            f"{r('serial', 'period_std_ms')} | {r('double', 'period_std_ms')} | {b['db_gain']['delta']:+.3f} | "
            f"{'yes' if b['db_gain']['rep_ranges_overlap'] else 'no'} |"
        )
    lines.append(
        "\n### Frontier movement (minimum verified SM budget; negative Δ = fewer SMs needed)\n"
    )
    lines.append("| Target | Serial before → after | Δ | DB before → after | Δ |")
    lines.append("|---|---|---:|---|---:|")
    for t in result["frontier"]:

        def f(mode):
            m = t[mode]
            return (
                f"{m['reference_min_verified_sm'] or 'none'} → {m['candidate_min_verified_sm'] or 'none'}",
                f"{m['delta_sm']:+d}" if m["delta_sm"] is not None else "n/a",
            )

        s, sd = f("serial")
        d, dd = f("double")
        lines.append(f"| {t['name']} | {s} | {sd} | {d} | {dd} |")
    lines.append(
        f"\nKnee: reference **{result['knee']['reference']['verdict']}**, candidate **{result['knee']['candidate']['verdict']}**. "
        f"Outputs byte-identical across the candidate sweep: {result['outputs']['candidate_byte_identical']}; "
        f"equal to reference outputs: {result['outputs']['candidate_equals_reference']}.\n"
    )
    if result["host_drift"] is None:
        lines.append(f"Host drift: {result['host_drift_note']}\n")
    else:
        hd = result["host_drift"]
        if hd["comparable"] and "control_over_reference" in hd:
            lines.append(
                "### Host drift (same-session control of the reference source / frozen reference)\n"
            )
            lines.append(
                "| Actual SMs | Serial FPS control/reference | DB FPS control/reference | Serial FPS candidate/control | DB FPS candidate/control |"
            )
            lines.append("|---|---:|---:|---:|---:|")
            cc = {b["label"]: b for b in hd["candidate_over_control"]}
            for b in hd["control_over_reference"]:
                c = cc.get(b["label"])
                lines.append(
                    f"| {b['label']} | {b['serial']['fps']['ratio_candidate_over_reference']:.3f} | {b['double']['fps']['ratio_candidate_over_reference']:.3f} | "
                    f"{c['serial']['fps']['ratio_candidate_over_reference']:.3f} | {c['double']['fps']['ratio_candidate_over_reference']:.3f} |"
                    if c
                    else f"| {b['label']} | {b['serial']['fps']['ratio_candidate_over_reference']:.3f} | {b['double']['fps']['ratio_candidate_over_reference']:.3f} | n/a | n/a |"
                )
        else:
            lines.append(
                "Host drift control supplied but not comparable to the reference scope.\n"
            )
    return "\n".join(lines) + "\n"


def load_record(path):
    path = Path(path)
    if path.is_dir():
        return build(path)
    record = json.loads(path.read_text())
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} record")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="freeze a validated sweep into a benchmark record")
    b.add_argument("root", type=Path, help="completed sweep directory")
    b.add_argument("--study-record", type=Path, help="report.py JSON for this sweep")
    b.add_argument("--json-output", type=Path, required=True)
    b.add_argument("--markdown-output", type=Path)
    c = sub.add_parser(
        "compare", help="compare a candidate sweep/record to a frozen reference"
    )
    c.add_argument("--reference", type=Path, required=True, help="frozen record JSON")
    c.add_argument(
        "--candidate", type=Path, required=True, help="sweep directory or record"
    )
    c.add_argument(
        "--control",
        type=Path,
        help="same-session sweep/record of the reference source, to expose host drift",
    )
    c.add_argument("--json-output", type=Path, required=True)
    c.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        record = build(args.root, args.study_record)
        args.json_output.write_text(json.dumps(record, indent=2) + "\n")
        if args.markdown_output:
            args.markdown_output.write_text(build_markdown(record))
        print(json.dumps(record["knee"], indent=2))
        return 0
    reference = load_record(args.reference)
    candidate = load_record(args.candidate)
    control = load_record(args.control) if args.control else None
    result = compare(reference, candidate, control)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.write_text(comparison_markdown(result))
    if not result["comparable"]:
        print("not comparable:", result["scope_mismatches"])
        return 2
    print(comparison_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
