#!/usr/bin/env python3
# mypy: ignore-errors
"""Tag-indexed, incremental tuner for the same-height occlusion gate.

Each config is a coordinate in the parameter hypercube and is stored under its integer tag,
e.g. iou_thresh=0.45 -> axis index 1.  Run any subset of tags (a one-factor-at-a-time "star"
through the baseline centre, a dense grid, or hand-picked points); results are appended to a
single jsonl keyed by tag, so re-runs are skipped and you can densify later without recompute.

Every run stores **per-sequence + overall** metrics, so any objective (max IDF1 / max AssA /
worst-seq-consistency / Pareto) is a post-hoc query over the table (see occ_rank.py) — one eval
per config, never one eval per objective.

Designs
-------
  --design star    baseline centre + vary one axis at a time (cheap main effects; ~9 runs)
  --design grid    full Cartesian product (interactions; ~72 runs)
  --design list    explicit --tags "1,2,1,1 0,2,1,1 ..."

Usage
-----
  .venv/bin/python scripts/eval/occ_tune.py --design star --detector SDP
  .venv/bin/python scripts/eval/occ_tune.py --design list --tags "1,3,2,1" --detector SDP
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import motmetrics as mm

# motmetrics still calls np.asfarray, removed in NumPy 2.0 — restore a shim.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

PROJECT = Path(__file__).resolve().parent.parent.parent

# Axis order is fixed: (iou_thresh, foot_gap, cost_weight, ttl)
AXES = {
    "occ_iou_thresh": [0.40, 0.45],  # Tier-1 pruned 0.50 (starves the gate)
    "occ_foot_gap": [0.10, 0.125, 0.15, 0.20],
    "occ_cost_weight": [0.3, 0.5, 0.7],
    "occ_ttl": [2, 4, 6],
}
AXIS_NAMES = list(AXES)
CENTER = (1, 2, 1, 1)  # current default: iou0.45 foot0.15 w0.5 ttl4
PRESET = "mamba_whole_graph"
CKPT = PROJECT / "runs/mamba_gt_v14replica_t3_t1/best.ckpt"
STORE = PROJECT / "results/occ_tune/metrics.jsonl"
_METRICS = [
    "idf1",
    "mota",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "recall",
    "precision",
]


def tag_to_params(tag: tuple[int, ...]) -> dict:
    return {name: AXES[name][i] for name, i in zip(AXIS_NAMES, tag)}


def star_tags() -> list[tuple[int, ...]]:
    tags = {CENTER}
    for ax in range(len(AXIS_NAMES)):
        for i in range(len(AXES[AXIS_NAMES[ax]])):
            t = list(CENTER)
            t[ax] = i
            tags.add(tuple(t))
    return sorted(tags)


def grid_tags() -> list[tuple[int, ...]]:
    return sorted(itertools.product(*(range(len(v)) for v in AXES.values())))


def ckpt_md5() -> str:
    h = hashlib.md5()
    with open(CKPT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_done() -> set[str]:
    if not STORE.exists():
        return set()
    return {
        json.loads(line)["tag"]
        for line in STORE.read_text().splitlines()
        if line.strip()
    }


def per_seq_metrics(out_dir: Path, detector: str) -> dict:
    """motmetrics per-seq + OVERALL from the tracker txt files in out_dir."""
    gt_root = PROJECT / "datasets/MOT17/train"
    files = sorted(out_dir.glob(f"*-{detector}.txt")) or sorted(
        f for f in out_dir.glob("*.txt") if detector in f.stem
    )
    accs, names = [], []
    for f in files:
        name = f.stem
        gt_f = gt_root / name / "gt" / "gt.txt"
        if not gt_f.exists():
            continue
        gt = mm.io.loadtxt(str(gt_f), fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(str(f), fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        names.append(name)
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=_METRICS, generate_overall=True
    )
    out = {}
    for row in list(names) + ["OVERALL"]:
        out[row] = {
            m: (
                float(summary.loc[row, m])
                if m in ("idf1", "mota", "recall", "precision")
                else int(summary.loc[row, m])
            )
            for m in _METRICS
        }
    return out


def run_one(tag: tuple[int, ...], detector: str) -> dict:
    params = tag_to_params(tag)
    tag_str = ",".join(map(str, tag))
    out_dir = PROJECT / "results/occ_tune" / f"t{tag_str.replace(',', '')}"
    cmd = [
        sys.executable,
        str(PROJECT / "scripts/eval/mot17.py"),
        "--preset",
        PRESET,
        "--detector",
        detector,
        "--output",
        str(out_dir),
        "--occ-state-enabled",
    ]
    for name, val in params.items():
        cmd += [f"--{name.replace('_', '-')}", str(val)]
    print(f"\n=== tag ({tag_str})  {params} ===")
    subprocess.run(cmd, check=True)
    metrics = per_seq_metrics(out_dir, detector)
    return {
        "tag": tag_str,
        "params": params,
        "ckpt_md5": ckpt_md5(),
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--design", choices=["star", "grid", "list"], default="star")
    ap.add_argument(
        "--tags",
        default="",
        help="space-separated tags for --design list, e.g. '1,2,1,1 0,2,1,1'",
    )
    ap.add_argument("--detector", default="SDP")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.design == "star":
        tags = star_tags()
    elif args.design == "grid":
        tags = grid_tags()
    else:
        tags = [tuple(int(x) for x in t.split(",")) for t in args.tags.split()]

    done = load_done()
    todo = [t for t in tags if ",".join(map(str, t)) not in done]
    print(
        f"design={args.design}  total={len(tags)}  done={len(tags) - len(todo)}  todo={len(todo)}"
    )
    for t in tags:
        m = "·done·" if ",".join(map(str, t)) in done else "  TODO"
        print(f"  {m}  ({','.join(map(str, t))})  {tag_to_params(t)}")
    if args.dry_run or not todo:
        return

    STORE.parent.mkdir(parents=True, exist_ok=True)
    for t in todo:
        rec = run_one(t, args.detector)
        with open(STORE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        ov = rec["metrics"]["OVERALL"]
        print(
            f"  -> IDF1 {ov['idf1'] * 100:.1f}  MOTA {ov['mota'] * 100:.1f}  "
            f"IDs {ov['num_switches']}  (stored)"
        )


if __name__ == "__main__":
    main()
