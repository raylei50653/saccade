#!/usr/bin/env python3
# mypy: ignore-errors
"""
Run cumulative pipeline cutoff experiments for MOT17 and summarize the
"raw uplift" of each downstream module before later modules are enabled.

Typical usage:
    uv run python scripts/eval/pipeline_contribution.py --detector SDP
    uv run python scripts/eval/pipeline_contribution.py --pose-engine models/yolo/yolo26s_pose_960_batch1.engine
    uv run python scripts/eval/pipeline_contribution.py --list-profiles
    uv run python scripts/eval/pipeline_contribution.py --dry-run -- --match-thresh 0.78
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import motmetrics as mm  # noqa: E402


METRICS = [
    "idf1",
    "recall",
    "precision",
    "mota",
    "num_switches",
    "num_false_positives",
    "num_misses",
]
PCT_METRICS = {"idf1", "recall", "precision", "mota"}


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative MOT17 pipeline cutoffs so each row shows the raw gain of "
            "enabling one more downstream module before later modules are turned on."
        )
    )
    parser.add_argument("--engine", default="models/yolo/yolo26s_960_batch1.engine")
    parser.add_argument("--pose-engine", default=None)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--detector", choices=["SDP", "DPM", "FRCNN"], default="SDP")
    parser.add_argument(
        "--sequences",
        default="MOT17-04-SDP,MOT17-10-SDP",
        help="Comma-separated sequences for all profiles.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional per-sequence frame cap. Unset means full sequences.",
    )
    parser.add_argument("--tiling", default="native_960")
    parser.add_argument(
        "--semantic-threshold-full",
        type=float,
        default=0.93,
        help="Semantic threshold used in the final full profile.",
    )
    parser.add_argument(
        "--profiles",
        default="default",
        help="Comma-separated profile names to run, or 'default' for the built-in sequence.",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only summarize existing per-profile output folders under --output-root/runs.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to every mot17.py run. Put them after '--'.",
    )
    return parser.parse_args()


def default_output_root() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/pipeline_contribution/{stamp}"


def build_profiles(
    pose_engine: str | None, semantic_threshold_full: float
) -> list[Profile]:
    profiles = [
        Profile(
            "tracker_core",
            "Detection + postprocess + tracker only. GMC, ReID, relink, bank, and merge are all off.",
            (
                "--reid-mode",
                "off",
                "--no-gmc",
                "--no-appearance-bank",
                "--no-lifecycle-merge",
                "--no-post-lifecycle-merge",
            ),
        ),
        Profile(
            "tracker_core_gmc",
            "Add GMC on top of tracker_core, but keep all identity recovery modules off.",
            (
                "--reid-mode",
                "off",
                "--gmc",
                "--no-appearance-bank",
                "--no-lifecycle-merge",
                "--no-post-lifecycle-merge",
            ),
        ),
        Profile(
            "semantic_core",
            "Add semantic ReID/relink on top of GMC, but keep appearance bank and merge stages off.",
            (
                "--reid-mode",
                "semantic",
                "--gmc",
                "--no-appearance-bank",
                "--no-semantic-bank-inject",
                "--no-lifecycle-merge",
                "--no-post-lifecycle-merge",
            ),
        ),
        Profile(
            "semantic_bank",
            "Add appearance bank + bank inject, still without lifecycle merge or throughput overlap.",
            (
                "--reid-mode",
                "semantic",
                "--gmc",
                "--appearance-bank",
                "--semantic-bank-inject",
                "--no-lifecycle-merge",
                "--no-post-lifecycle-merge",
            ),
        ),
        Profile(
            "full_default",
            "Current full path: semantic bank + async_reid + pipeline_relink + best-known minimal relink threshold.",
            (
                "--reid-mode",
                "semantic",
                "--gmc",
                "--appearance-bank",
                "--semantic-bank-inject",
                "--no-lifecycle-merge",
                "--no-post-lifecycle-merge",
                "--async-reid",
                "--pipeline-relink",
                "--semantic-threshold",
                f"{semantic_threshold_full:.2f}",
            ),
        ),
    ]
    if pose_engine:
        profiles.append(
            Profile(
                "pose_sidecar",
                "Add the pose sidecar on top of the full path to isolate the extra value of pose-only keypoints.",
                (
                    "--reid-mode",
                    "semantic",
                    "--gmc",
                    "--appearance-bank",
                    "--semantic-bank-inject",
                    "--no-lifecycle-merge",
                    "--no-post-lifecycle-merge",
                    "--async-reid",
                    "--pipeline-relink",
                    "--semantic-threshold",
                    f"{semantic_threshold_full:.2f}",
                    "--pose-engine",
                    pose_engine,
                ),
            )
        )
    return profiles


def select_profiles(all_profiles: list[Profile], requested: str) -> list[Profile]:
    if requested.strip() == "default":
        return all_profiles
    names = [name.strip() for name in requested.split(",") if name.strip()]
    lookup = {profile.name: profile for profile in all_profiles}
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(f"Unsupported profiles: {', '.join(unknown)}")
    return [lookup[name] for name in names]


def build_command(
    args: argparse.Namespace, profile: Profile, output_dir: Path
) -> list[str]:
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/eval/mot17.py",
        "--output",
        str(output_dir),
        "--data-root",
        args.data_root,
        "--split",
        args.split,
        "--detector",
        args.detector,
        "--sequences",
        args.sequences,
        "--engine",
        args.engine,
        "--tiling",
        args.tiling,
        *profile.args,
        *extra_args,
    ]
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    return cmd


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(project_root)]
    if src_path.exists():
        pythonpath_entries.append(str(src_path))
    if build_path.exists():
        pythonpath_entries.append(str(build_path))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if build_path.exists():
        ld_library = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{build_path}{os.pathsep}{ld_library}" if ld_library else str(build_path)
        )
    return env


def run_profile(cmd: list[str], dry_run: bool) -> int:
    print(">>> " + " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(project_root), env=make_env())
    return result.returncode


def is_mot_result_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("MOT") and name.endswith(".txt")


def evaluate_dir(
    results_dir: Path, gt_root: str, detector: str | None
) -> dict[str, float | int] | None:
    cache_path = results_dir / "metrics.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    import glob

    files = sorted(
        f for f in glob.glob(str(results_dir / "*.txt")) if is_mot_result_file(f)
    )
    if detector:
        files = [f for f in files if f"-{detector}" in Path(f).stem]
    if not files:
        return None

    gt_files = glob.glob(os.path.join(gt_root, "*/gt/gt.txt"))
    if detector:
        gt_files = [f for f in gt_files if f"-{detector}" in Path(f).parts[-3]]
    gt = {
        Path(f).parts[-3]: mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)
        for f in gt_files
    }

    accs = []
    names = []
    for file_path in files:
        name = Path(file_path).stem
        if name not in gt:
            continue
        ts = mm.io.loadtxt(file_path, fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt[name], ts, "iou", distth=0.5))
        names.append(name)
    if not accs:
        return None

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)
    results: dict[str, float | int] = {}
    for metric in METRICS:
        value = summary.loc["OVERALL", metric]
        results[metric] = float(value) if metric in PCT_METRICS else int(value)

    cache_path.write_text(json.dumps(results, indent=2))
    return results


def load_eval_fps(results_dir: Path) -> float | None:
    summary_path = results_dir / "_fps_summary.txt"
    if not summary_path.exists():
        return None
    for line in summary_path.read_text().splitlines():
        if not line.startswith("OVERALL\t"):
            continue
        for field in line.split("\t")[1:]:
            if field.startswith("fps="):
                try:
                    return float(field.split("=", 1)[1])
                except ValueError:
                    return None
    return None


def pct_delta(current: float, previous: float) -> str:
    delta = (current - previous) * 100.0
    return f"{delta:+.1f}pp"


def int_delta(current: int, previous: int) -> str:
    return f"{current - previous:+d}"


def fps_delta(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "n/a"
    return f"{current - previous:+.2f}"


def write_reports(
    output_root: Path,
    profiles: list[Profile],
    rows: list[dict[str, object]],
    commands: OrderedDict[str, list[str]],
) -> None:
    csv_lines = [
        "profile,idf1,mota,ids,fp,fn,fps,idf1_delta_prev,mota_delta_prev,ids_delta_prev,fps_delta_prev"
    ]
    md_lines = [
        "# Pipeline Contribution Report",
        "",
        "這張表看的是 cumulative cutoff。",
        "每一列只比前一列多開一層下游模組，因此 `Δprev` 就是該模組的裸提升。",
        "",
        "| Profile | Meaning | IDF1 | MOTA | IDs | FP | FN | FPS | IDF1 Δprev | MOTA Δprev | IDs Δprev | FPS Δprev |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["profile"]),
                    str(row.get("idf1", "")),
                    str(row.get("mota", "")),
                    str(row.get("ids", "")),
                    str(row.get("fp", "")),
                    str(row.get("fn", "")),
                    str(row.get("fps", "")),
                    str(row.get("idf1_delta_prev", "")),
                    str(row.get("mota_delta_prev", "")),
                    str(row.get("ids_delta_prev", "")),
                    str(row.get("fps_delta_prev", "")),
                ]
            )
        )
        md_lines.append(
            "| {profile} | {description} | {idf1} | {mota} | {ids} | {fp} | {fn} | {fps} | {idf1_delta_prev} | {mota_delta_prev} | {ids_delta_prev} | {fps_delta_prev} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for profile in profiles:
        cmd = commands.get(profile.name)
        if not cmd:
            continue
        md_lines.append(f"- `{profile.name}`")
        md_lines.append(f"  `{' '.join(cmd)}`")

    (output_root / "contribution_report.csv").write_text("\n".join(csv_lines) + "\n")
    (output_root / "contribution_report.md").write_text("\n".join(md_lines) + "\n")
    (output_root / "commands.txt").write_text(
        "\n".join(" ".join(cmd) for cmd in commands.values()) + "\n"
    )


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root or default_output_root()
    output_root = project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    all_profiles = build_profiles(args.pose_engine, args.semantic_threshold_full)
    if args.list_profiles:
        for profile in all_profiles:
            print(f"{profile.name:18s} {profile.description}")
        return 0

    profiles = select_profiles(all_profiles, args.profiles)
    commands: OrderedDict[str, list[str]] = OrderedDict()
    for profile in profiles:
        commands[profile.name] = build_command(args, profile, runs_root / profile.name)

    if not args.skip_run:
        for profile in profiles:
            rc = run_profile(commands[profile.name], args.dry_run)
            if rc != 0:
                return rc

    rows: list[dict[str, object]] = []
    previous_metrics: dict[str, float | int] | None = None
    previous_fps: float | None = None
    for profile in profiles:
        metrics = evaluate_dir(runs_root / profile.name, args.gt_root, args.detector)
        fps = load_eval_fps(runs_root / profile.name)
        if metrics is None:
            rows.append(
                {
                    "profile": profile.name,
                    "description": profile.description,
                    "idf1": "n/a",
                    "mota": "n/a",
                    "ids": "n/a",
                    "fp": "n/a",
                    "fn": "n/a",
                    "fps": "n/a",
                    "idf1_delta_prev": "n/a",
                    "mota_delta_prev": "n/a",
                    "ids_delta_prev": "n/a",
                    "fps_delta_prev": "n/a",
                }
            )
            continue

        idf1 = float(metrics["idf1"])
        mota = float(metrics["mota"])
        ids = int(metrics["num_switches"])
        fp = int(metrics["num_false_positives"])
        fn = int(metrics["num_misses"])
        row = {
            "profile": profile.name,
            "description": profile.description,
            "idf1": f"{idf1 * 100:.1f}",
            "mota": f"{mota * 100:.1f}",
            "ids": str(ids),
            "fp": str(fp),
            "fn": str(fn),
            "fps": f"{fps:.2f}" if fps is not None else "n/a",
            "idf1_delta_prev": "base",
            "mota_delta_prev": "base",
            "ids_delta_prev": "base",
            "fps_delta_prev": "base",
        }
        if previous_metrics is not None:
            row["idf1_delta_prev"] = pct_delta(idf1, float(previous_metrics["idf1"]))
            row["mota_delta_prev"] = pct_delta(mota, float(previous_metrics["mota"]))
            row["ids_delta_prev"] = int_delta(
                ids, int(previous_metrics["num_switches"])
            )
            row["fps_delta_prev"] = fps_delta(fps, previous_fps)
        rows.append(row)
        previous_metrics = metrics
        previous_fps = fps

    write_reports(output_root, profiles, rows, commands)
    print(f"Saved report to {output_root / 'contribution_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
