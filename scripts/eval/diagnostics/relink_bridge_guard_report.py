#!/usr/bin/env python3
"""Summarize guarded native bridge relink runs.

The guarded mnv4 route is intentionally narrow: it should reduce wrong bridge
adoptions without loosening relink range or touching primary association. This
report puts the relevant signals in one place:

  * aggregate MOT metrics from metrics.json
  * lost/recover success rate, using the same definition as reconnect_rate.py
  * optional native relink debug counters parsed from evaluator logs

Usage:
  uv run python scripts/eval/diagnostics/relink_bridge_guard_report.py \
      --pred-dir results/guarded \
      --baseline-dir results/baseline \
      --log results/guarded/run.log
"""
# status: diagnostic

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.diagnostics.reconnect_rate import run as reconnect_run  # noqa: E402

DEFAULT_SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]

RELINK_RE = re.compile(
    r"Relink debug (?P<seq>\S+): .*?"
    r"bridge_attempts=(?P<attempts>\d+) "
    r"bridge_accepts=(?P<accepts>\d+)"
    r"(?:.*?bridge_veto=(?P<veto>\d+))?"
)


@dataclass(frozen=True)
class ReconnectSummary:
    opportunities: int
    success: int

    @property
    def rate(self) -> float:
        return self.success / self.opportunities if self.opportunities else 0.0


def _load_metrics(path: Path) -> dict[str, float]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: float(v) for k, v in raw.items() if isinstance(v, int | float)}


def _summarize_reconnect(
    pred_dir: Path, gt_root: Path, seqs: list[str], iou: float, min_gap: int
) -> ReconnectSummary:
    records = reconnect_run(pred_dir, gt_root, seqs, iou, min_gap)
    return ReconnectSummary(
        opportunities=len(records),
        success=sum(int(r["success"]) for r in records),
    )


def _parse_relink_log(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RELINK_RE.search(line)
        if not match:
            continue
        seq = match.group("seq")
        out[seq] = {
            "bridge_attempts": int(match.group("attempts")),
            "bridge_accepts": int(match.group("accepts")),
            "bridge_veto": int(match.group("veto") or 0),
        }
    return out


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def _fmt_delta_pct(value: float) -> str:
    return f"{value * 100:+.2f}"


def _print_metrics(
    label: str, metrics: dict[str, float], base: dict[str, float] | None = None
) -> None:
    if not metrics:
        print(f"{label}: metrics.json not found")
        return
    idf1 = metrics.get("idf1", 0.0)
    mota = metrics.get("mota", 0.0)
    ids = metrics.get("num_switches", 0.0)
    fp = metrics.get("num_false_positives", 0.0)
    fn = metrics.get("num_misses", 0.0)
    if base:
        print(
            f"{label}: IDF1 {_fmt_pct(idf1)} ({_fmt_delta_pct(idf1 - base.get('idf1', 0.0))})  "
            f"MOTA {_fmt_pct(mota)} ({_fmt_delta_pct(mota - base.get('mota', 0.0))})  "
            f"IDs {ids:.0f} ({ids - base.get('num_switches', 0.0):+.0f})  "
            f"FP {fp:.0f} ({fp - base.get('num_false_positives', 0.0):+.0f})  "
            f"FN {fn:.0f} ({fn - base.get('num_misses', 0.0):+.0f})"
        )
    else:
        print(
            f"{label}: IDF1 {_fmt_pct(idf1)}  MOTA {_fmt_pct(mota)}  "
            f"IDs {ids:.0f}  FP {fp:.0f}  FN {fn:.0f}"
        )


def _print_reconnect(
    label: str, summary: ReconnectSummary, base: ReconnectSummary | None = None
) -> None:
    if base:
        print(
            f"{label} reconnect: {summary.success}/{summary.opportunities} "
            f"({_fmt_pct(summary.rate)}%, {summary.rate - base.rate:+.3%})"
        )
    else:
        print(
            f"{label} reconnect: {summary.success}/{summary.opportunities} "
            f"({_fmt_pct(summary.rate)}%)"
        )


def _print_relink_debug(rows: dict[str, dict[str, int]]) -> None:
    if not rows:
        print("Relink debug: no log counters parsed")
        return
    total = {
        "bridge_attempts": sum(r["bridge_attempts"] for r in rows.values()),
        "bridge_accepts": sum(r["bridge_accepts"] for r in rows.values()),
        "bridge_veto": sum(r["bridge_veto"] for r in rows.values()),
    }
    print(
        "Relink debug total: "
        f"bridge_attempts={total['bridge_attempts']} "
        f"bridge_accepts={total['bridge_accepts']} "
        f"bridge_veto={total['bridge_veto']}"
    )
    print("Per-sequence bridge counters:")
    for seq in sorted(rows):
        row = rows[seq]
        print(
            f"  {seq}: attempts={row['bridge_attempts']} "
            f"accepts={row['bridge_accepts']} veto={row['bridge_veto']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="Candidate result directory.")
    ap.add_argument(
        "--baseline-dir", default="", help="Optional baseline result directory."
    )
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument(
        "--sequences",
        default="",
        help="Comma-separated sequence names; default MOT17 SDP train split.",
    )
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument(
        "--log",
        default="",
        help="Optional evaluator stdout/stderr log to parse relink counters.",
    )
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    base_dir = Path(args.baseline_dir) if args.baseline_dir else None
    gt_root = Path(args.gt_root)
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()] or DEFAULT_SEQS

    pred_metrics = _load_metrics(pred_dir)
    base_metrics = _load_metrics(base_dir) if base_dir else None
    if base_metrics:
        _print_metrics("candidate", pred_metrics, base_metrics)
        _print_metrics("baseline", base_metrics)
    else:
        _print_metrics("candidate", pred_metrics)

    pred_reconnect = _summarize_reconnect(
        pred_dir, gt_root, seqs, args.iou, args.min_gap
    )
    base_reconnect = (
        _summarize_reconnect(base_dir, gt_root, seqs, args.iou, args.min_gap)
        if base_dir
        else None
    )
    _print_reconnect("candidate", pred_reconnect, base_reconnect)
    if base_reconnect:
        _print_reconnect("baseline", base_reconnect)

    log_path = Path(args.log) if args.log else None
    _print_relink_debug(_parse_relink_log(log_path))


if __name__ == "__main__":
    main()
