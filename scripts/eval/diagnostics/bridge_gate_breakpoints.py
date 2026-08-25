#!/usr/bin/env python3
"""Locate the exact jump discontinuities of pooled IDF1 along one bridge-gate axis.

Why this exists instead of another grid sweep
---------------------------------------------
The bridge scale gate is ``h_lo < ratio < h_hi`` applied to a *finite* set of
evaluated candidate pairs.  Moving a threshold continuously changes nothing at
all until it crosses one of the ratio values actually observed in the data, at
which point one or more decisions flip and the metric jumps.  The metric surface
is therefore **piecewise constant with jumps**, not a smooth ridge, and a grid
sweep reports an arbitrary sampling of a step function: the reported "optimum"
is a grid artefact and "distance to the nearest bad cell" carries no stability
information (this is what made the old ``nearest_unsafe_distance=1`` result
vacuous).

The right objects are the **plateau** you operate on and the **jump** at its
edge.  This tool finds them by bisection rather than by sampling.

Bisection is available here because the underlying measurement is bit-exact
(``reid_mode: off`` + ``--no-gpu-decode``, so N=1 suffices), which makes
*equality* a usable predicate.  A bracket whose endpoints differ can then be
narrowed to any width in O(log(1/tol)) evaluations instead of O(1/tol) grid
points.

Equality predicate
------------------
Endpoints are compared on the **per-sequence (idtp, idfp, idfn) vector**, not on
pooled IDF1.  Two genuinely different decision sets can collide on a rounded (or
even exact) pooled scalar; the count vector is a far tighter fingerprint of "the
tracker did the same thing".

Limits (these are properties of the method, not of the implementation)
----------------------------------------------------------------------
1. ``f(a) == f(b)`` does **not** prove there is no jump inside ``(a, b)`` — two
   jumps can cancel.  Every "plateau" below means *no jump detected at the scan
   resolution*, never *no jump exists*.
2. If a scan bracket contains several jumps, bisection converges to one of them
   and the others are missed.  Raise ``--scan`` to separate them.
3. Breakpoint locations are **data values**, i.e. a property of this dataset's
   ratio distribution rather than of the gate.  Do not expect them to transfer
   to another dataset; see the 2026-08-08 cross-dataset note.

Usage
-----
  .venv/bin/python scripts/eval/diagnostics/bridge_gate_breakpoints.py \
      --axis h_hi --lo 1.2 --hi 1.8 --scan 7 --tol 1e-3 \
      --px 0.4 --h-lo 0.6 \
      --preset mamba_whole_graph_m --detector SDP \
      --work-dir out/bp_h_hi

  # dataset without a detector suffix (MOT20 / DanceTrack)
  ... --data-root datasets/MOT20/MOT20 --split train   # and omit --detector
"""
# status: diagnostic

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Eval occasionally dies inside CUDA-graph capture (DALI touching the stream
# mid-capture).  It is transient and a plain re-run clears it, so a bisection
# should not lose fifteen completed evaluations to one of them.
TRANSIENT_MARKERS = (
    "cudaErrorStreamCaptureInvalidated",
    "cudaErrorStreamCaptureUnsupported",
    "operation failed due to a previous error during capture",
    "operation not permitted when stream is capturing",
)
AXES = ("px", "h_lo", "h_hi")
FLAG = {
    "px": "--relink-bridge-px",
    "h_lo": "--relink-bridge-h-lo",
    "h_hi": "--relink-bridge-h-hi",
}


@dataclass
class Measurement:
    """One evaluated threshold value."""

    value: float
    idf1: float
    per_seq: dict[str, dict[str, int]]
    out_dir: str

    @property
    def fingerprint(self) -> tuple:
        """Bit-exact identity of the decision outcome (see module docstring)."""
        return tuple(
            (seq, c["idtp"], c["idfp"], c["idfn"])
            for seq, c in sorted(self.per_seq.items())
        )

    def per_seq_idf1(self) -> dict[str, float]:
        return {seq: _idf1(c) for seq, c in sorted(self.per_seq.items())}


def _idf1(counts: dict[str, int]) -> float:
    denom = 2 * counts["idtp"] + counts["idfp"] + counts["idfn"]
    return 0.0 if denom == 0 else 100.0 * 2 * counts["idtp"] / denom


def _pooled_idf1(per_seq: dict[str, dict[str, int]]) -> float:
    """IDF1 is a ratio: pool the counts, never average the per-sequence values."""
    total = {"idtp": 0, "idfp": 0, "idfn": 0}
    for counts in per_seq.values():
        for key in total:
            total[key] += counts[key]
    return _idf1(total)


class Runner:
    """Runs eval at a threshold value and scores it, with an on-disk cache."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work_dir = Path(args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.work_dir / "cache.json"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.runs = 0
        # Every point measured this session, fresh or served from cache.  The
        # report is derived from all of them, never from the scan grid alone.
        self.seen: dict[float, Measurement] = {}

    def _params(self, value: float) -> dict[str, float]:
        params = {
            "px": self.args.px,
            "h_lo": self.args.h_lo,
            "h_hi": self.args.h_hi,
        }
        params[self.args.axis] = value
        return params

    def _key(self, value: float) -> str:
        params = self._params(value)
        return json.dumps(
            {
                "preset": self.args.preset,
                "data_root": self.args.data_root,
                "split": self.args.split,
                "detector": self.args.detector,
                "eval_arg": self.args.eval_arg,
                **{k: f"{v:.10g}" for k, v in params.items()},
            },
            sort_keys=True,
        )

    def adopt_cached_range(self, lo: float, hi: float) -> int:
        """Pull every already-measured point in [lo, hi] into this report.

        The cache outlives a single invocation, so a targeted refinement run
        leaves behind points that a later full-range run would never visit. Not
        adopting them makes the report understate what has actually been
        measured -- it would re-label an already-bisected jump UNREFINED.
        """
        adopted = 0
        for key, hit in self.cache.items():
            entry = json.loads(key)
            if entry.get(self.args.axis) is None:
                continue
            value = float(entry[self.args.axis])
            if not lo <= value <= hi or value in self.seen:
                continue
            same_context = all(
                entry.get(field) == expected
                for field, expected in (
                    ("preset", self.args.preset),
                    ("data_root", self.args.data_root),
                    ("split", self.args.split),
                    ("detector", self.args.detector),
                    ("eval_arg", self.args.eval_arg),
                )
            )
            fixed_match = all(
                entry.get(axis) == f"{getattr(self.args, axis):.10g}"
                for axis in AXES
                if axis != self.args.axis
            )
            if not (same_context and fixed_match):
                continue
            self.seen[value] = Measurement(
                value, hit["idf1"], hit["per_seq"], hit["out_dir"]
            )
            adopted += 1
        return adopted

    def measure(self, value: float) -> Measurement:
        key = self._key(value)
        if key in self.cache:
            hit = self.cache[key]
            cached = Measurement(value, hit["idf1"], hit["per_seq"], hit["out_dir"])
            self.seen[value] = cached
            return cached

        params = self._params(value)
        tag = "_".join(f"{k}{params[k]:.10g}" for k in AXES)
        out_dir = self.work_dir / f"run_{tag}"
        self._run_eval(params, out_dir)
        per_seq = self._score(out_dir)
        if not per_seq:
            raise RuntimeError(f"no scored sequences in {out_dir}")
        measurement = Measurement(value, _pooled_idf1(per_seq), per_seq, str(out_dir))
        self.cache[key] = {
            "idf1": measurement.idf1,
            "per_seq": per_seq,
            "out_dir": str(out_dir),
        }
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.runs += 1
        self.seen[value] = measurement
        return measurement

    def _run_eval(self, params: dict[str, float], out_dir: Path) -> None:
        cmd = [
            self.args.python,
            "scripts/eval/mot17.py",
            "--preset",
            self.args.preset,
            "--data-root",
            self.args.data_root,
            "--split",
            self.args.split,
            "--output",
            str(out_dir),
        ]
        if self.args.detector:
            cmd += ["--detector", self.args.detector]
        cmd += ["--double-buffer", "--no-gpu-decode"]
        for axis in AXES:
            cmd += [FLAG[axis], f"{params[axis]:.10g}"]
        cmd += list(self.args.eval_arg)

        env = dict(os.environ)
        torch_lib = REPO_ROOT / ".venv/lib/python3.12/site-packages/torch/lib"
        if torch_lib.is_dir():
            env["LD_LIBRARY_PATH"] = (
                f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
            )

        print(f"  run {' '.join(cmd[-8:])}", file=sys.stderr, flush=True)
        if self.args.dry_run:
            return

        for attempt in range(1, self.args.retries + 2):
            suffix = "" if attempt == 1 else f".attempt{attempt}"
            log_path = out_dir.parent / f"{out_dir.name}{suffix}.log"
            # A retry must not score leftovers from the failed attempt.
            if out_dir.exists():
                for stale in out_dir.glob("*.txt"):
                    stale.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if proc.returncode == 0:
                return

            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            transient = next((m for m in TRANSIENT_MARKERS if m in tail), None)
            if transient is None or attempt > self.args.retries:
                raise RuntimeError(f"eval failed ({proc.returncode}); see {log_path}")
            print(
                f"    transient failure ({transient}); "
                f"retry {attempt}/{self.args.retries}",
                file=sys.stderr,
                flush=True,
            )

    def _score(self, out_dir: Path) -> dict[str, dict[str, int]]:
        """Recompute IDF1 counts at full precision from the saved MOT output."""
        if self.args.dry_run:
            return {"_dry_run": {"idtp": 1, "idfp": 0, "idfn": 0}}
        src = str(REPO_ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from saccade.perception.eval.metrics import _evaluate_single_sequence

        gt_root = Path(self.args.data_root) / self.args.split
        per_seq: dict[str, dict[str, int]] = {}
        for ts_path in sorted(out_dir.glob("*.txt")):
            seq = ts_path.stem
            if seq.startswith("_"):
                continue
            gt_path = gt_root / seq / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            summary = _evaluate_single_sequence(seq, str(gt_path), str(ts_path))
            per_seq[seq] = {
                "idtp": int(summary["idtp"]),
                "idfp": int(summary["idfp"]),
                "idfn": int(summary["idfn"]),
            }
        return per_seq


@dataclass
class Bracket:
    low: Measurement
    high: Measurement
    refined: bool = False
    history: list[float] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.high.value - self.low.value


def scan(runner: Runner, lo: float, hi: float, points: int) -> list[Measurement]:
    if points < 2:
        raise ValueError("--scan must be at least 2")
    step = (hi - lo) / (points - 1)
    values = [lo + i * step for i in range(points)]
    out = []
    for i, value in enumerate(values, 1):
        print(f"[scan {i}/{points}] {value:.6g}", file=sys.stderr, flush=True)
        out.append(runner.measure(value))
    return out


def brackets_from_scan(samples: list[Measurement]) -> list[Bracket]:
    return [
        Bracket(a, b)
        for a, b in zip(samples, samples[1:])
        if a.fingerprint != b.fingerprint
    ]


def bisect(runner: Runner, bracket: Bracket, tol: float) -> Bracket:
    low, high = bracket.low, bracket.high
    while high.value - low.value > tol:
        mid_value = 0.5 * (low.value + high.value)
        if mid_value <= low.value or mid_value >= high.value:
            break  # float resolution exhausted
        print(
            f"  [bisect] ({low.value:.8g}, {high.value:.8g}) -> {mid_value:.8g}",
            file=sys.stderr,
            flush=True,
        )
        mid = runner.measure(mid_value)
        bracket.history.append(mid_value)
        if mid.fingerprint == low.fingerprint:
            low = mid
        else:
            high = mid
    bracket.low, bracket.high, bracket.refined = low, high, True
    return bracket


def build_report(
    args: argparse.Namespace,
    seen: list[Measurement],
    runs: int,
) -> dict:
    """Derive plateaus and jumps from *every* measured point.

    Deriving them from the scan grid alone lets the report assert a plateau that
    the bisection's own midpoints refute -- the tool would be claiming more than
    it measured.  Grouping all measured points by fingerprint cannot do that: a
    plateau is exactly a maximal run of consecutive points that behaved
    identically, and a jump is exactly the gap between two such runs.
    """
    points = sorted(seen, key=lambda m: m.value)
    groups: list[list[Measurement]] = []
    for point in points:
        if groups and groups[-1][0].fingerprint == point.fingerprint:
            groups[-1].append(point)
        else:
            groups.append([point])

    plateaus = [
        {
            "from": g[0].value,
            "to": g[-1].value,
            "width": g[-1].value - g[0].value,
            "idf1": g[0].idf1,
            "per_seq_idf1": g[0].per_seq_idf1(),
            "measured_points": len(g),
        }
        for g in groups
    ]

    breakpoints = []
    for below, above in zip(groups, groups[1:]):
        low, high = below[-1], above[0]
        breakpoints.append(
            {
                "lower_bound": low.value,
                "upper_bound": high.value,
                "width": high.value - low.value,
                "idf1_below": low.idf1,
                "idf1_above": high.idf1,
                "delta_idf1": high.idf1 - low.idf1,
                "per_seq_delta_idf1": {
                    seq: high.per_seq_idf1()[seq] - value
                    for seq, value in low.per_seq_idf1().items()
                },
                "refined": (high.value - low.value) <= args.tol,
            }
        )

    return {
        "tool": "bridge_gate_breakpoints",
        "schema_version": 1,
        "axis": args.axis,
        "fixed_axes": {a: getattr(args, a) for a in AXES if a != args.axis},
        "dataset": {
            "preset": args.preset,
            "data_root": args.data_root,
            "split": args.split,
            "detector": args.detector,
            "extra_eval_args": list(args.eval_arg),
        },
        "equality_predicate": "per-sequence (idtp, idfp, idfn) vector",
        "scan": {"lo": args.lo, "hi": args.hi, "points": args.scan},
        "measurements": [
            {"value": m.value, "idf1": m.idf1, "out_dir": m.out_dir} for m in points
        ],
        "tolerance": args.tol,
        "eval_runs_executed": runs,
        "breakpoints": breakpoints,
        "plateaus": plateaus,
        "limits": [
            "equal endpoints do not prove the absence of a jump between them; "
            "plateaus mean 'no jump detected at the scan resolution'",
            "an unrefined jump (width > --tol) means the scan bracket held "
            "more than one jump and bisection converged to a different one; "
            "raise --scan to separate them",
            "breakpoint locations are data values, so they are a property of "
            "this dataset's feature distribution, not of the gate",
        ],
    }


def print_table(report: dict) -> None:
    print(f"\naxis={report['axis']}  fixed={report['fixed_axes']}")
    print(f"eval runs executed: {report['eval_runs_executed']}")

    print("\nplateaus (identical outcome across every measured point inside):")
    print(f"  {'from':>12}  {'to':>12}  {'width':>10}  {'IDF1':>9}  {'pts':>4}")
    for p in report["plateaus"]:
        print(
            f"  {p['from']:>12.7g}  {p['to']:>12.7g}  "
            f"{p['width']:>10.4g}  {p['idf1']:>9.3f}  {p['measured_points']:>4d}"
        )

    print("\nbreakpoints (jump localised to):")
    if not report["breakpoints"]:
        print("  none detected at this scan resolution")
    for b in report["breakpoints"]:
        mark = "" if b["refined"] else "   [UNREFINED - may hide further jumps]"
        print(
            f"  ({b['lower_bound']:.8g}, {b['upper_bound']:.8g}]  "
            f"width={b['width']:.3g}  "
            f"IDF1 {b['idf1_below']:.3f} -> {b['idf1_above']:.3f}  "
            f"(delta {b['delta_idf1']:+.3f}){mark}"
        )
        worst = sorted(b["per_seq_delta_idf1"].items(), key=lambda kv: kv[1])[:2]
        for seq, delta in worst:
            if abs(delta) >= 0.05:
                print(f"      {seq}: {delta:+.3f}")

    print("\nlimits:")
    for limit in report["limits"]:
        print(f"  - {limit}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--axis", choices=AXES, required=True, help="Threshold to vary.")
    p.add_argument("--lo", type=float, required=True, help="Scan range lower end.")
    p.add_argument("--hi", type=float, required=True, help="Scan range upper end.")
    p.add_argument(
        "--scan", type=int, default=7, help="Initial scan points (>=2). Default 7."
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Bisect until each jump is bracketed to this width. Default 1e-3.",
    )
    p.add_argument("--px", type=float, default=0.4, help="relink_bridge_px.")
    p.add_argument("--h-lo", dest="h_lo", type=float, default=0.6)
    p.add_argument("--h-hi", dest="h_hi", type=float, default=1.7)
    p.add_argument("--preset", default="mamba_whole_graph_m")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--detector",
        default="SDP",
        help="Detector suffix. Pass empty for MOT20/DanceTrack.",
    )
    p.add_argument(
        "--eval-arg",
        action="append",
        default=[],
        help=(
            "Extra argument forwarded verbatim to mot17.py (repeatable). Use "
            "the equals form for values starting with '-', e.g. "
            "--eval-arg=--seq-limit --eval-arg=2."
        ),
    )
    p.add_argument("--work-dir", required=True, help="Run outputs and result cache.")
    p.add_argument("--json", help="Write the full report here.")
    p.add_argument("--python", default=".venv/bin/python")
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help=(
            "Re-run an eval this many times when it dies with a known "
            "transient CUDA-graph capture error. Default 2."
        ),
    )
    p.add_argument(
        "--scan-only",
        action="store_true",
        help="Find brackets but skip bisection.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned eval commands without running them.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hi <= args.lo:
        print("--hi must exceed --lo", file=sys.stderr)
        return 2

    runner = Runner(args)
    samples = scan(runner, args.lo, args.hi, args.scan)
    brackets = brackets_from_scan(samples)
    print(f"\n{len(brackets)} bracket(s) contain a jump", file=sys.stderr, flush=True)

    failure: str | None = None
    if not args.scan_only:
        for i, bracket in enumerate(brackets, 1):
            print(f"[refine {i}/{len(brackets)}]", file=sys.stderr, flush=True)
            try:
                bisect(runner, bracket, args.tol)
            except RuntimeError as exc:
                # Do not throw away the brackets already refined: report what is
                # known, leave the rest unrefined, and fail loudly at the end.
                failure = str(exc)
                print(f"  ABORTED: {exc}", file=sys.stderr, flush=True)
                break

    adopted = runner.adopt_cached_range(args.lo, args.hi)
    if adopted:
        print(
            f"adopted {adopted} previously measured point(s) from the cache",
            file=sys.stderr,
        )

    report = build_report(args, list(runner.seen.values()), runner.runs)
    if failure is not None:
        report["incomplete"] = failure
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.json}", file=sys.stderr)
    print_table(report)
    if failure is not None:
        print(f"\nINCOMPLETE: {failure}", file=sys.stderr)
        print("cached results are kept; re-run to resume", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
