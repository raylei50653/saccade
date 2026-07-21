#!/usr/bin/env python
"""Synthesize Cheb-GR handover applicability evidence across summaries.

The output is an applicability map, not a threshold recommendation. It turns
multiple ``parameter_summary.json`` files into stable / condition-sensitive /
unstable labels for candidate rules, feature ranges, and pollution context.

Usage:
  uv run scripts/eval/diagnostics/synthesize_handover_applicability.py \
    results/run_m/parameter_summary.json results/run_s/parameter_summary.json \
    --out-md results/applicability.md --out-json results/applicability.json
"""
# status: diagnostic

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "cheb_gr_offline_handover_summary/v1"
OUT_SCHEMA = "cheb_gr_offline_handover_applicability/v1"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: unsupported schema {data.get('schema')!r}")
    return data


def _run_name(path: Path, summary: dict[str, Any]) -> str:
    pred_dir = str(summary.get("provenance", {}).get("pred_dir") or "")
    return Path(pred_dir).name if pred_dir else path.parent.name


def _rate_text(correct: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{correct}/{total} ({correct / total:.2f})"


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _classify_rule(
    *,
    decision: str,
    rates: list[float],
    selected: list[int],
    seq_fractions: list[float],
    min_selected: int,
    min_seq_fraction: float,
    veto_max_rate: float,
    accept_min_rate: float,
    support_min_rate: float,
    max_rate_spread: float,
) -> str:
    if any(n < min_selected for n in selected):
        return "thin"
    if not rates:
        return "missing"
    lo = min(rates)
    hi = max(rates)
    spread = hi - lo
    seq_ok = min(seq_fractions or [1.0]) >= min_seq_fraction
    if decision == "reject/veto":
        if hi <= veto_max_rate:
            return "stable-veto"
        if lo <= veto_max_rate:
            return "condition-sensitive-veto"
        return "not-veto"
    if decision == "accept-candidate":
        if lo >= accept_min_rate and spread <= max_rate_spread and seq_ok:
            return "stable-accept-candidate"
        if lo >= support_min_rate:
            return "condition-sensitive-accept"
        return "unstable-accept"
    if lo >= support_min_rate and spread <= max_rate_spread and seq_ok:
        return "stable-support"
    if lo >= support_min_rate:
        return "condition-sensitive-support"
    return "support-only-or-unstable"


def _summarize_rules(
    summaries: list[dict[str, Any]],
    names: list[str],
    *,
    min_selected: int,
    min_seq_fraction: float,
    veto_max_rate: float,
    accept_min_rate: float,
    support_min_rate: float,
    max_rate_spread: float,
) -> list[dict[str, Any]]:
    by_run = [
        {rule["name"]: rule for rule in summary.get("candidate_rules", [])}
        for summary in summaries
    ]
    rule_names = sorted(set().union(*(set(rules) for rules in by_run)))
    out: list[dict[str, Any]] = []
    for name in rule_names:
        present = [rules[name] for rules in by_run if name in rules]
        first = present[0] if present else {}
        rates = [float(rule["same_rate"]) for rule in present]
        selected = [int(rule["selected"]) for rule in present]
        seq_fractions = [
            float(rule.get("seqs_with_min_n", rule.get("seq_count", 1)))
            / max(1.0, float(rule.get("seq_count", 1)))
            for rule in present
        ]
        classification = _classify_rule(
            decision=str(first.get("decision", "")),
            rates=rates,
            selected=selected,
            seq_fractions=seq_fractions,
            min_selected=min_selected,
            min_seq_fraction=min_seq_fraction,
            veto_max_rate=veto_max_rate,
            accept_min_rate=accept_min_rate,
            support_min_rate=support_min_rate,
            max_rate_spread=max_rate_spread,
        )
        out.append(
            {
                "name": name,
                "classification": classification,
                "decision": first.get("decision", ""),
                "expression": first.get("expression", ""),
                "use": first.get("use", ""),
                "min_rate": min(rates) if rates else None,
                "max_rate": max(rates) if rates else None,
                "min_selected": min(selected) if selected else 0,
                "min_seq_fraction": min(seq_fractions) if seq_fractions else None,
                "runs": {
                    run_name: {
                        "correct": int(rule["correct"]),
                        "wrong": int(rule["wrong"]),
                        "selected": int(rule["selected"]),
                        "same_rate": float(rule["same_rate"]),
                    }
                    for run_name, rule in zip(
                        names, [rules.get(name) for rules in by_run]
                    )
                    if rule
                },
            }
        )
    return out


def _classify_bucket(
    *,
    rates: list[float],
    totals: list[int],
    zones: list[str],
    min_bucket_n: int,
    danger_max_rate: float,
    support_min_rate: float,
    max_rate_spread: float,
) -> str:
    if any(total < min_bucket_n for total in totals):
        return "thin"
    if not rates:
        return "missing"
    lo = min(rates)
    hi = max(rates)
    spread = hi - lo
    if hi <= danger_max_rate:
        return "stable-danger"
    if lo >= support_min_rate and spread <= max_rate_spread:
        if "accept-candidate" in zones:
            return "stable-accept-range"
        return "stable-support-range"
    if lo >= support_min_rate:
        return "condition-sensitive-support-range"
    if len(set(zones)) > 1:
        return "condition-sensitive"
    return zones[0] if zones else "gray"


def _summarize_feature_ranges(
    summaries: list[dict[str, Any]],
    names: list[str],
    *,
    features: list[str],
    min_bucket_n: int,
    danger_max_rate: float,
    support_min_rate: float,
    max_rate_spread: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in features:
        per_run = []
        for summary in summaries:
            buckets = summary.get("features", {}).get(feature, {}).get("buckets", [])
            per_run.append({bucket["bucket"]: bucket for bucket in buckets})
        bucket_names = sorted(set().union(*(set(buckets) for buckets in per_run)))
        for bucket_name in bucket_names:
            present = [
                buckets[bucket_name] for buckets in per_run if bucket_name in buckets
            ]
            rates = [float(bucket["same_rate"]) for bucket in present]
            totals = [int(bucket["total"]) for bucket in present]
            zones = [str(bucket["zone"]) for bucket in present]
            classification = _classify_bucket(
                rates=rates,
                totals=totals,
                zones=zones,
                min_bucket_n=min_bucket_n,
                danger_max_rate=danger_max_rate,
                support_min_rate=support_min_rate,
                max_rate_spread=max_rate_spread,
            )
            out.append(
                {
                    "feature": feature,
                    "bucket": bucket_name,
                    "classification": classification,
                    "min_rate": min(rates) if rates else None,
                    "max_rate": max(rates) if rates else None,
                    "min_total": min(totals) if totals else 0,
                    "runs": {
                        run_name: {
                            "correct": int(bucket["correct"]),
                            "wrong": int(bucket["wrong"]),
                            "total": int(bucket["total"]),
                            "same_rate": float(bucket["same_rate"]),
                            "zone": str(bucket["zone"]),
                        }
                        for run_name, bucket in zip(
                            names, [buckets.get(bucket_name) for buckets in per_run]
                        )
                        if bucket
                    },
                }
            )
    return out


def _summarize_pollution(
    summaries: list[dict[str, Any]],
    names: list[str],
    *,
    features: list[str],
    min_bucket_n: int,
    pollution_high_rate: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in features:
        per_run = []
        for summary in summaries:
            buckets = (
                summary.get("pollution", {}).get("feature_buckets", {}).get(feature, [])
            )
            per_run.append({bucket["bucket"]: bucket for bucket in buckets})
        bucket_names = sorted(set().union(*(set(buckets) for buckets in per_run)))
        for bucket_name in bucket_names:
            present = [
                buckets[bucket_name] for buckets in per_run if bucket_name in buckets
            ]
            rates = [float(bucket["pollution_rate"]) for bucket in present]
            totals = [int(bucket["total"]) for bucket in present]
            if any(total < min_bucket_n for total in totals):
                classification = "thin"
            elif rates and min(rates) >= pollution_high_rate:
                classification = "stable-high-pollution"
            elif rates and max(rates) >= pollution_high_rate:
                classification = "condition-sensitive-pollution"
            else:
                classification = "context"
            out.append(
                {
                    "feature": feature,
                    "bucket": bucket_name,
                    "classification": classification,
                    "min_pollution_rate": min(rates) if rates else None,
                    "max_pollution_rate": max(rates) if rates else None,
                    "min_total": min(totals) if totals else 0,
                    "runs": {
                        run_name: {
                            "endpoint_polluted": int(bucket["endpoint_polluted"]),
                            "total": int(bucket["total"]),
                            "pollution_rate": float(bucket["pollution_rate"]),
                        }
                        for run_name, bucket in zip(
                            names, [buckets.get(bucket_name) for buckets in per_run]
                        )
                        if bucket
                    },
                }
            )
    return out


def _synthesize(
    paths: list[Path],
    *,
    features: list[str],
    min_selected: int,
    min_bucket_n: int,
    min_seq_fraction: float,
    veto_max_rate: float,
    accept_min_rate: float,
    support_min_rate: float,
    max_rate_spread: float,
    pollution_high_rate: float,
) -> dict[str, Any]:
    summaries = [_load(path) for path in paths]
    names = [_run_name(path, summary) for path, summary in zip(paths, summaries)]
    rules = _summarize_rules(
        summaries,
        names,
        min_selected=min_selected,
        min_seq_fraction=min_seq_fraction,
        veto_max_rate=veto_max_rate,
        accept_min_rate=accept_min_rate,
        support_min_rate=support_min_rate,
        max_rate_spread=max_rate_spread,
    )
    ranges = _summarize_feature_ranges(
        summaries,
        names,
        features=features,
        min_bucket_n=min_bucket_n,
        danger_max_rate=veto_max_rate,
        support_min_rate=support_min_rate,
        max_rate_spread=max_rate_spread,
    )
    pollution = _summarize_pollution(
        summaries,
        names,
        features=["neighbor_iou", "head_tail_neighbor_iou", "match_iou"],
        min_bucket_n=min_bucket_n,
        pollution_high_rate=pollution_high_rate,
    )
    return {
        "schema": OUT_SCHEMA,
        "inputs": [str(path) for path in paths],
        "runs": names,
        "criteria": {
            "min_selected": min_selected,
            "min_bucket_n": min_bucket_n,
            "min_seq_fraction": min_seq_fraction,
            "veto_max_rate": veto_max_rate,
            "accept_min_rate": accept_min_rate,
            "support_min_rate": support_min_rate,
            "max_rate_spread": max_rate_spread,
            "pollution_high_rate": pollution_high_rate,
        },
        "rule_applicability": rules,
        "feature_range_applicability": ranges,
        "pollution_applicability": pollution,
    }


def _md_table_rate_runs(entry: dict[str, Any]) -> str:
    cells = []
    for run_name, data in entry["runs"].items():
        total = int(data.get("selected", data.get("total", 0)))
        correct = int(data.get("correct", 0))
        if "endpoint_polluted" in data:
            cells.append(
                f"{run_name}: {data['endpoint_polluted']}/{data['total']} "
                f"({_fmt(float(data['pollution_rate']))})"
            )
        else:
            cells.append(f"{run_name}: {_rate_text(correct, total)}")
    return "<br>".join(cells)


def _write_md(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Cheb-GR Offline Handover Applicability Map",
        "",
        "This is a cross-run evidence map, not a default threshold config.",
        "",
        "## Criteria",
        "",
    ]
    for key, value in data["criteria"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Rule Applicability", ""])
    lines.extend(
        [
            "| rule | class | expression | min/max same_gt | evidence |",
            "|---|---|---|---:|---|",
        ]
    )
    for rule in data["rule_applicability"]:
        lines.append(
            f"| `{rule['name']}` | {rule['classification']} | "
            f"`{rule['expression']}` | {_fmt(rule['min_rate'])}-{_fmt(rule['max_rate'])} | "
            f"{_md_table_rate_runs(rule)} |"
        )
    lines.extend(["", "## Feature Range Applicability", ""])
    lines.extend(
        [
            "| feature | range | class | min/max same_gt | evidence |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in data["feature_range_applicability"]:
        lines.append(
            f"| `{item['feature']}` | `{item['bucket']}` | {item['classification']} | "
            f"{_fmt(item['min_rate'])}-{_fmt(item['max_rate'])} | "
            f"{_md_table_rate_runs(item)} |"
        )
    lines.extend(["", "## Pollution Applicability", ""])
    lines.extend(
        [
            "| feature | range | class | min/max pollution | evidence |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in data["pollution_applicability"]:
        lines.append(
            f"| `{item['feature']}` | `{item['bucket']}` | {item['classification']} | "
            f"{_fmt(item['min_pollution_rate'])}-{_fmt(item['max_pollution_rate'])} | "
            f"{_md_table_rate_runs(item)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", nargs="+", type=Path)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--feature",
        action="append",
        default=[
            "best_cost",
            "margin",
            "center_dist_norm",
            "match_iou",
            "neighbor_iou",
        ],
        help="Feature ranges to synthesize; can be passed more than once.",
    )
    ap.add_argument("--min-selected", type=int, default=10)
    ap.add_argument("--min-bucket-n", type=int, default=8)
    ap.add_argument(
        "--min-seq-fraction",
        type=float,
        default=0.5,
        help="Minimum sequence coverage before accept/support rules can be stable.",
    )
    ap.add_argument("--veto-max-rate", type=float, default=0.08)
    ap.add_argument("--accept-min-rate", type=float, default=0.70)
    ap.add_argument("--support-min-rate", type=float, default=0.30)
    ap.add_argument("--max-rate-spread", type=float, default=0.18)
    ap.add_argument("--pollution-high-rate", type=float, default=0.30)
    args = ap.parse_args()

    data = _synthesize(
        args.summary,
        features=list(dict.fromkeys(args.feature)),
        min_selected=args.min_selected,
        min_bucket_n=args.min_bucket_n,
        min_seq_fraction=args.min_seq_fraction,
        veto_max_rate=args.veto_max_rate,
        accept_min_rate=args.accept_min_rate,
        support_min_rate=args.support_min_rate,
        max_rate_spread=args.max_rate_spread,
        pollution_high_rate=args.pollution_high_rate,
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.out_md:
        _write_md(args.out_md, data)
    if not args.out_json and not args.out_md:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
