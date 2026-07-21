#!/usr/bin/env python
"""Compare Cheb-GR offline handover parameter summary JSON files.

This is for cross-run applicability analysis, not metric leaderboard tuning.
It highlights whether candidate rules, feature buckets, and pollution evidence
are stable across runs.

Usage:
  uv run scripts/eval/diagnostics/compare_handover_summaries.py \
    results/run_a/parameter_summary.json results/run_b/parameter_summary.json \
    --feature best_cost --feature neighbor_iou
"""
# status: stable

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    schema = data.get("schema")
    if schema != "cheb_gr_offline_handover_summary/v1":
        raise SystemExit(f"{path}: unsupported schema {schema!r}")
    return data


def _run_name(path: Path, data: dict[str, Any]) -> str:
    pred_dir = str(data.get("provenance", {}).get("pred_dir") or "")
    return Path(pred_dir).name if pred_dir else path.parent.name


def _rate(success: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{success}/{total} ({success / total:.2f})"


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _print_counts(names: list[str], summaries: list[dict[str, Any]]) -> None:
    print("## Counts")
    print("")
    print(
        "| run | rows | accepted | known | correct | wrong | same_rate | "
        "accepted_precision |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in zip(names, summaries):
        c = summary["counts"]
        print(
            f"| {name} | {c['rows']} | {c['accepted']} | {c['known']} | "
            f"{c['correct']} | {c['wrong']} | {_fmt_float(c['same_rate'])} | "
            f"{_fmt_float(c['accepted_precision'])} |"
        )
    print("")


def _by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in items}


def _print_candidate_rules(
    names: list[str], summaries: list[dict[str, Any]], *, min_selected: int
) -> None:
    print("## Candidate Rules")
    print("")
    rules_by_run = [_by_name(s["candidate_rules"]) for s in summaries]
    rule_names = sorted(set().union(*(set(r) for r in rules_by_run)))
    header = "| rule | decision | expression | " + " | ".join(names) + " |"
    print(header)
    print("|---|---|---|" + "|".join("---:" for _ in names) + "|")
    for rule_name in rule_names:
        first = next(
            (rules[rule_name] for rules in rules_by_run if rule_name in rules), {}
        )
        cells = []
        for rules in rules_by_run:
            rule = rules.get(rule_name)
            if not rule:
                cells.append("-")
                continue
            selected = int(rule["selected"])
            if selected < min_selected:
                cells.append(f"thin {selected}")
                continue
            cells.append(
                f"{_rate(int(rule['correct']), selected)}; "
                f"LOO {_fmt_float(rule.get('loo_same_rate_min'), 2)}-"
                f"{_fmt_float(rule.get('loo_same_rate_max'), 2)}"
            )
        print(
            f"| `{rule_name}` | {first.get('decision', '-')} | "
            f"`{first.get('expression', '-')}` | " + " | ".join(cells) + " |"
        )
    print("")


def _print_policy(names: list[str], summaries: list[dict[str, Any]]) -> None:
    print("## Policy Simulation")
    print("")
    policies_by_run = [_by_name(s["policy_simulation"]) for s in summaries]
    policy_names = sorted(set().union(*(set(p) for p in policies_by_run)))
    header = "| rule | action | " + " | ".join(names) + " |"
    print(header)
    print("|---|---|" + "|".join("---:" for _ in names) + "|")
    for policy_name in policy_names:
        first = next((p[policy_name] for p in policies_by_run if policy_name in p), {})
        cells = []
        for policies in policies_by_run:
            policy = policies.get(policy_name)
            if not policy:
                cells.append("-")
                continue
            cells.append(
                f"keep {policy['kept_correct']}/{policy['accepted_correct']} ok, "
                f"{policy['kept_wrong']}/{policy['accepted_wrong']} bad; "
                f"cut {policy['wrong_cut']} bad/{policy['correct_cut']} ok"
            )
        print(
            f"| `{policy_name}` | {first.get('action', '-')} | "
            + " | ".join(cells)
            + " |"
        )
    print("")


def _print_discovered_gates(names: list[str], summaries: list[dict[str, Any]]) -> None:
    sections = (
        ("all_known_single_feature", "All Known Single-Feature Gates"),
        ("accepted_known_single_feature", "Accepted Known Single-Feature Gates"),
        ("accepted_known_two_feature", "Accepted Known Two-Feature Gates"),
    )
    discovered_by_run = [s.get("discovered_gates", {}) for s in summaries]
    if not any(discovered_by_run):
        return
    print("## Discovered Gates")
    print("")
    for key, title in sections:
        per_run = [
            {gate["expression"]: gate for gate in discovered.get(key, [])}
            for discovered in discovered_by_run
        ]
        expressions = sorted(set().union(*(set(gates) for gates in per_run)))
        if not expressions:
            continue
        print(f"### {title}")
        print("")
        header = "| gate | " + " | ".join(names) + " |"
        print(header)
        print("|---|" + "|".join("---:" for _ in names) + "|")
        for expr in expressions:
            cells = []
            for gates in per_run:
                gate = gates.get(expr)
                if not gate:
                    cells.append("-")
                    continue
                cells.append(
                    f"{_rate(int(gate['correct']), int(gate['selected']))}; "
                    f"p={_fmt_float(gate['precision'], 2)} "
                    f"r={_fmt_float(gate['correct_recall'], 2)} "
                    f"bad_keep={_fmt_float(gate['wrong_keep'], 2)}"
                )
            print(f"| `{expr}` | " + " | ".join(cells) + " |")
        print("")


def _print_pollution(names: list[str], summaries: list[dict[str, Any]]) -> None:
    print("## Pollution")
    print("")
    print("| run | eligible | endpoint_polluted | pollution_rate |")
    print("|---|---:|---:|---:|")
    for name, summary in zip(names, summaries):
        p = summary["pollution"]
        print(
            f"| {name} | {p['eligible']} | {p['endpoint_polluted']} | "
            f"{_fmt_float(p['pollution_rate'])} |"
        )
    print("")

    bucket_features = sorted(
        set().union(
            *[
                set(summary["pollution"].get("feature_buckets", {}))
                for summary in summaries
            ]
        )
    )
    for feature in bucket_features:
        print(f"### Pollution Buckets: {feature}")
        print("")
        per_run = [
            {
                bucket["bucket"]: bucket
                for bucket in summary["pollution"]["feature_buckets"].get(feature, [])
            }
            for summary in summaries
        ]
        bucket_names = sorted(set().union(*(set(b) for b in per_run)))
        header = "| bucket | " + " | ".join(names) + " |"
        print(header)
        print("|---|" + "|".join("---:" for _ in names) + "|")
        for bucket_name in bucket_names:
            cells = []
            for buckets in per_run:
                bucket = buckets.get(bucket_name)
                if not bucket:
                    cells.append("-")
                    continue
                cells.append(
                    f"{bucket['endpoint_polluted']}/{bucket['total']} "
                    f"({_fmt_float(bucket['pollution_rate'], 2)})"
                )
            print(f"| `{bucket_name}` | " + " | ".join(cells) + " |")
        print("")


def _print_feature_buckets(
    names: list[str],
    summaries: list[dict[str, Any]],
    *,
    features: list[str],
    min_bucket_n: int,
) -> None:
    if not features:
        return
    print("## Feature Buckets")
    print("")
    for feature in features:
        print(f"### {feature}")
        print("")
        per_run = []
        for summary in summaries:
            feature_data = summary["features"].get(feature)
            per_run.append(
                {
                    bucket["bucket"]: bucket
                    for bucket in (feature_data or {}).get("buckets", [])
                }
            )
        bucket_names = sorted(set().union(*(set(b) for b in per_run)))
        header = "| bucket | " + " | ".join(names) + " |"
        print(header)
        print("|---|" + "|".join("---:" for _ in names) + "|")
        for bucket_name in bucket_names:
            cells = []
            for buckets in per_run:
                bucket = buckets.get(bucket_name)
                if not bucket:
                    cells.append("-")
                    continue
                total = int(bucket["total"])
                zone = bucket["zone"] if total >= min_bucket_n else "thin"
                cells.append(f"{_rate(int(bucket['correct']), total)}; {zone}")
            print(f"| `{bucket_name}` | " + " | ".join(cells) + " |")
        print("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", nargs="+", type=Path)
    ap.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Feature bucket table to include; can be passed more than once.",
    )
    ap.add_argument(
        "--min-selected",
        type=int,
        default=5,
        help="Mark candidate rules below this selected count as thin.",
    )
    ap.add_argument(
        "--min-bucket-n",
        type=int,
        default=5,
        help="Mark feature buckets below this count as thin in comparison output.",
    )
    args = ap.parse_args()

    summaries = [_load(path) for path in args.summary]
    names = [_run_name(path, summary) for path, summary in zip(args.summary, summaries)]
    print("# Cheb-GR Offline Handover Summary Comparison")
    print("")
    _print_counts(names, summaries)
    _print_candidate_rules(names, summaries, min_selected=args.min_selected)
    _print_discovered_gates(names, summaries)
    _print_policy(names, summaries)
    _print_pollution(names, summaries)
    _print_feature_buckets(
        names,
        summaries,
        features=args.feature,
        min_bucket_n=args.min_bucket_n,
    )


if __name__ == "__main__":
    main()
