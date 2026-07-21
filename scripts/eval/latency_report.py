#!/usr/bin/env python3
"""Standalone latency analyzer for _stage_profile.json files.

Usage:
  python scripts/eval/latency_report.py <run_dir_or_json>
  python scripts/eval/latency_report.py <run_dir_or_json> --compare <other_dir_or_json>
  python scripts/eval/latency_report.py <run_dir_or_json> --seq MOT17-04-SDP
"""
# status: stable

import argparse
import json
import sys
from pathlib import Path


def _resolve_json(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_dir():
        candidate = p / "_stage_profile.json"
        if not candidate.exists():
            sys.exit(
                f"No _stage_profile.json found in {p}. Run mot17.py --profile-stages first."
            )
        return candidate
    if not p.exists():
        sys.exit(f"File not found: {p}")
    return p


def _load(path_arg: str) -> dict:
    json_path = _resolve_json(path_arg)
    with json_path.open() as f:
        return json.load(f)


def _stage_data(profile: dict, seq: str | None) -> dict[str, dict]:
    if seq:
        seqs = profile.get("sequences", {})
        if seq not in seqs:
            available = ", ".join(seqs.keys()) or "(none)"
            sys.exit(f"Sequence '{seq}' not found. Available: {available}")
        entry = seqs[seq]
        return {k: v for k, v in entry.items() if isinstance(v, dict)}
    return profile.get("overall", {})


def _print_waterfall(stages: dict[str, dict], bar_width: int = 22) -> None:
    ft = stages.get("frame_total", {})
    ft_ms = ft.get("mean_ms", 0.0)
    if ft_ms <= 0:
        print("  (no frame_total data)")
        return

    rows = [
        (name, info["mean_ms"])
        for name, info in stages.items()
        if name != "frame_total" and "mean_ms" in info and info["mean_ms"] > 0
    ]
    rows.sort(key=lambda x: -x[1])

    accounted = sum(ms for _, ms in rows)
    unaccounted = ft_ms - accounted

    label_w = max((len(n) for n, _ in rows), default=14) + 2
    label_w = max(label_w, 14)
    sep = "─" * (label_w + bar_width + 18)
    print(f"\n  Stage Breakdown (frame_total = {ft_ms:.2f} ms)")
    print(f"  {sep}")
    for name, ms in rows:
        pct = ms / ft_ms * 100.0
        filled = round(pct / 100.0 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {name:<{label_w}} {bar}  {ms:6.2f}ms  {pct:5.1f}%")
    if unaccounted > 0.5:
        pct = unaccounted / ft_ms * 100.0
        print(
            f"  {'[unaccounted]':<{label_w}} {'░' * bar_width}  {unaccounted:6.2f}ms  {pct:5.1f}%"
        )
    print(f"  {sep}")

    # p95 annotation for top stages
    top_keys = [n for n, _ in rows[:5]]
    top_keys.append("frame_total")
    jitter_rows = [
        (n, stages[n]) for n in top_keys if n in stages and "p95_ms" in stages[n]
    ]
    if jitter_rows:
        print()
        col_w = label_w
        print(f"  {'Stage':<{col_w}} {'mean':>8}  {'std':>8}  {'p95':>8}  {'p99':>8}")
        print(f"  {'─' * col_w} {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        for name, info in jitter_rows:
            print(
                f"  {name:<{col_w}} {info.get('mean_ms', 0):>7.2f}ms"
                f"  {info.get('std_ms', 0):>7.2f}ms"
                f"  {info.get('p95_ms', 0):>7.2f}ms"
                f"  {info.get('p99_ms', 0):>7.2f}ms"
            )


def _print_comparison(
    stages_a: dict[str, dict],
    stages_b: dict[str, dict],
    label_a: str,
    label_b: str,
) -> None:
    all_stages = sorted(
        set(stages_a) | set(stages_b),
        key=lambda n: -stages_a.get(n, {}).get("mean_ms", 0),
    )
    # Put frame_total at the end
    if "frame_total" in all_stages:
        all_stages.remove("frame_total")
        all_stages.append("frame_total")

    label_w = max((len(s) for s in all_stages), default=14) + 2
    label_w = max(label_w, 14)
    col = 10

    la = label_a[-col:] if len(label_a) > col else label_a
    lb = label_b[-col:] if len(label_b) > col else label_b
    header = f"  {'Stage':<{label_w}} {la:>{col}}  {lb:>{col}}  {'Delta':>{col}}  {'Delta%':>8}"
    sep = "─" * len(header)
    print(f"\n{header}")
    print(f"  {sep}")

    for name in all_stages:
        a_ms = stages_a.get(name, {}).get("mean_ms")
        b_ms = stages_b.get(name, {}).get("mean_ms")
        if a_ms is None and b_ms is None:
            continue
        if (a_ms or 0) == 0 and (b_ms or 0) == 0:
            continue
        a_str = f"{a_ms:.2f}ms" if a_ms is not None else "—"
        b_str = f"{b_ms:.2f}ms" if b_ms is not None else "—"
        if a_ms is not None and b_ms is not None:
            delta = b_ms - a_ms
            if a_ms > 0:
                pct_str = f"{'+' if delta >= 0 else ''}{delta / a_ms * 100.0:.1f}%"
            else:
                pct_str = "n/a"
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}ms"
        else:
            delta_str = "—"
            pct_str = "—"
        print(
            f"  {name:<{label_w}} {a_str:>{col}}  {b_str:>{col}}  {delta_str:>{col}}  {pct_str:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("profile", help="Run directory or path to _stage_profile.json")
    parser.add_argument(
        "--compare", metavar="PATH", help="Second profile to compare against"
    )
    parser.add_argument(
        "--seq", metavar="NAME", help="Show per-sequence breakdown instead of overall"
    )
    args = parser.parse_args()

    data = _load(args.profile)
    meta = data.get("meta", {})

    print(f"\n  Profile: {Path(args.profile).resolve()}")
    if meta:
        print(
            f"  Frames: {meta.get('profiled_frames', '?')}  "
            f"FPS: {meta.get('fps_mean', '?')}  "
            f"mean_ms: {meta.get('fps_mean_ms', '?')}"
        )

    stages = _stage_data(data, args.seq)
    if args.seq:
        print(f"  Sequence: {args.seq}")

    if args.compare:
        data2 = _load(args.compare)
        meta2 = data2.get("meta", {})
        stages2 = _stage_data(data2, args.seq)
        print(f"  Compare:  {Path(args.compare).resolve()}")
        if meta2:
            print(
                f"  Frames: {meta2.get('profiled_frames', '?')}  "
                f"FPS: {meta2.get('fps_mean', '?')}  "
                f"mean_ms: {meta2.get('fps_mean_ms', '?')}"
            )
        label_a = (
            Path(args.profile).parent.name
            if Path(args.profile).is_file()
            else Path(args.profile).name
        )
        label_b = (
            Path(args.compare).parent.name
            if Path(args.compare).is_file()
            else Path(args.compare).name
        )
        _print_comparison(stages, stages2, label_a, label_b)
    else:
        _print_waterfall(stages)

    seqs = data.get("sequences", {})
    if seqs and not args.seq and not args.compare:
        print(f"\n  Sequences in profile: {', '.join(seqs.keys())}")
        print("  (Use --seq NAME to show per-sequence breakdown)")


if __name__ == "__main__":
    main()
