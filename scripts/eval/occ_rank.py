#!/usr/bin/env python3
# mypy: ignore-errors
"""Post-hoc ranker over the tag-indexed occ-tune store — zero eval.

Every objective is a query over results/occ_tune/metrics.jsonl (one row per config tag, with
per-seq + overall metrics).  Cross-compare by any metric, enforce per-seq consistency, slice a
single axis through the baseline (main effect), or print the Pareto front — no re-running.

Usage
-----
  .venv/bin/python scripts/eval/occ_rank.py --by idf1
  .venv/bin/python scripts/eval/occ_rank.py --by idf1 --consistency 0.3
  .venv/bin/python scripts/eval/occ_rank.py --axis occ_foot_gap     # main-effect slice
  .venv/bin/python scripts/eval/occ_rank.py --pareto idf1,assa
"""
# status: stable

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
STORE = PROJECT / "results/occ_tune/metrics.jsonl"
CENTER_TAG = "1,2,1,1"
SEQS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]


def load() -> list[dict]:
    if not STORE.exists():
        raise SystemExit(f"no store at {STORE} — run occ_tune.py first")
    return [json.loads(line) for line in STORE.read_text().splitlines() if line.strip()]


def ov(rec, m):
    return rec["metrics"]["OVERALL"][m]


def per_seq_idf1(rec):
    return {
        s: rec["metrics"][s]["idf1"]
        for s in SEQS
        if s + "-SDP" in rec["metrics"] or s in rec["metrics"]
    }


def worst_seq_delta(rec, base, metric="idf1"):
    """Min over sequences of (rec - baseline) for the given metric (×100)."""
    deltas = []
    for s in rec["metrics"]:
        if s == "OVERALL" or s not in base["metrics"]:
            continue
        deltas.append((rec["metrics"][s][metric] - base["metrics"][s][metric]) * 100)
    return min(deltas) if deltas else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--by", default="idf1", help="metric to rank by (idf1/mota/num_switches/...)"
    )
    ap.add_argument(
        "--consistency",
        type=float,
        default=None,
        help="require worst-seq IDF1 delta vs baseline >= -X (×100, e.g. 0.3)",
    )
    ap.add_argument(
        "--axis",
        default=None,
        help="main-effect slice: hold others at baseline, vary this axis",
    )
    ap.add_argument("--pareto", default=None, help="two metrics, e.g. idf1,assa")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    recs = load()
    by_tag = {r["tag"]: r for r in recs}
    base = by_tag.get(CENTER_TAG)
    lower_better = (
        "num" in args.by
        or "switch" in args.by
        or "miss" in args.by
        or "false" in args.by
    )

    def fmt(rec):
        o = rec["metrics"]["OVERALL"]
        ws = f"{worst_seq_delta(rec, base):+.1f}" if base else "  ?"
        return (
            f"({rec['tag']:<8}) {rec['params']}  "
            f"IDF1 {o['idf1'] * 100:5.1f}  MOTA {o['mota'] * 100:5.1f}  "
            f"IDs {o['num_switches']:4d}  FP {o['num_false_positives']:5d}  worstΔ {ws}"
        )

    if args.axis:
        if args.axis not in __import__("scripts.eval.occ_tune", fromlist=["AXES"]).AXES:
            raise SystemExit(f"unknown axis {args.axis}")
        from scripts.eval.occ_tune import AXES, AXIS_NAMES

        ax = AXIS_NAMES.index(args.axis)
        base_tag = [int(x) for x in CENTER_TAG.split(",")]
        print(f"\n── main-effect slice: {args.axis} (others at baseline) ──")
        for i in range(len(AXES[args.axis])):
            t = list(base_tag)
            t[ax] = i
            r = by_tag.get(",".join(map(str, t)))
            print(
                f"  {args.axis}={AXES[args.axis][i]:<6} "
                + (fmt(r) if r else "·not run·")
            )
        return

    if args.pareto:
        m1, m2 = args.pareto.split(",")
        pts = [(ov(r, m1), ov(r, m2), r) for r in recs]
        front = []
        for x, y, r in sorted(pts, key=lambda p: -p[0]):
            if all(
                not (ox >= x and oy >= y and (ox, oy) != (x, y)) for ox, oy, _ in pts
            ):
                front.append(r)
        print(f"\n── Pareto front ({m1} × {m2}) ──")
        for r in front:
            print(f"  {m1} {ov(r, m1):.3f}  {m2} {ov(r, m2):.3f}  " + fmt(r))
        return

    cand = recs
    if args.consistency is not None and base:
        cand = [r for r in recs if worst_seq_delta(r, base) >= -args.consistency]
        print(
            f"(consistency filter: worst-seq IDF1 Δ ≥ -{args.consistency} → {len(cand)}/{len(recs)} pass)"
        )
    cand = sorted(cand, key=lambda r: ov(r, args.by), reverse=not lower_better)
    print(
        f"\n── ranked by {args.by} ({'lower' if lower_better else 'higher'} better) ──"
    )
    if base:
        print("baseline " + fmt(base))
    for r in cand[: args.top]:
        print("  " + fmt(r))


if __name__ == "__main__":
    main()
