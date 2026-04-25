# mypy: ignore-errors
import os
import glob
import numpy as np
import configparser

# 🐒 NumPy 2.0 向下相容補丁：修復 motmetrics 使用已移除的 asfarray 的問題
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import argparse


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            print("Comparing {}...".format(k))
            accs.append(
                mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5)
            )
            names.append(k)
        else:
            print("Warning: No ground truth for {}, skipping.".format(k))
    return accs, names


def load_sequence_fps(gt_folder: str, names: list[str]) -> dict[str, dict[str, int]]:
    seq_meta: dict[str, dict[str, int]] = {}
    for name in names:
        seqinfo_path = Path(gt_folder) / name / "seqinfo.ini"
        if not seqinfo_path.exists():
            continue
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        if "Sequence" not in config:
            continue
        seq_meta[name] = {
            "fps": config.getint("Sequence", "frameRate", fallback=0),
            "frames": config.getint("Sequence", "seqLength", fallback=0),
        }
    return seq_meta


def load_eval_fps_summary(results_folder: str) -> dict[str, dict[str, str]]:
    summary_path = Path(results_folder) / "_fps_summary.txt"
    if not summary_path.exists():
        return {}

    eval_meta: dict[str, dict[str, str]] = {}
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        name = parts[0]
        fields: dict[str, str] = {}
        for field in parts[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            fields[key] = value
        eval_meta[name] = fields
    return eval_meta


def is_mot_result_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("MOT") and name.endswith(".txt")


def run_mota_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/MOT17_eval")
    args = parser.parse_args()

    results_folder = args.results
    gt_folder = "datasets/MOT17/train"

    gtfiles = glob.glob(os.path.join(gt_folder, "*/gt/gt.txt"))
    tsfiles = sorted(
        f for f in glob.glob(os.path.join(results_folder, "*.txt")) if is_mot_result_file(f)
    )

    print(
        "Found {} groundtruths and {} test files in {}.".format(
            len(gtfiles), len(tsfiles), results_folder
        )
    )

    gt = OrderedDict(
        [
            (Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1))
            for f in gtfiles
        ]
    )
    ts = OrderedDict(
        [
            (
                os.path.splitext(Path(f).parts[-1])[0],
                mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1.0),
            )
            for f in tsfiles
        ]
    )

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    print("Running metrics")
    metrics = mm.metrics.motchallenge_metrics + ["num_objects"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(
        mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
    )

    seq_meta = load_sequence_fps(gt_folder, names)
    if seq_meta:
        print("\nSequence FPS")
        total_weighted_fps = 0.0
        total_frames = 0
        for name in names:
            meta = seq_meta.get(name)
            if not meta:
                continue
            fps = meta["fps"]
            frames = meta["frames"]
            total_weighted_fps += fps * frames
            total_frames += frames
            print(f"{name:16s} fps={fps:3d} frames={frames}")
        if total_frames > 0:
            overall_fps = total_weighted_fps / total_frames
            print(f"OVERALL source_fps={overall_fps:.2f} weighted_by_frames ({total_frames} frames)")

    eval_fps_meta = load_eval_fps_summary(results_folder)
    if eval_fps_meta:
        print("\nEval Throughput")
        for name in names:
            meta = eval_fps_meta.get(name)
            if not meta:
                continue
            fps = meta.get("fps", "n/a")
            mean_ms = meta.get("mean_ms", "n/a")
            frames = meta.get("frames", "n/a")
            print(f"{name:16s} fps={fps:>6s} mean_ms={mean_ms:>6s} frames={frames}")
        overall_meta = eval_fps_meta.get("OVERALL")
        if overall_meta:
            print(
                "OVERALL eval_fps={fps} mean_ms={mean_ms} frames={frames}".format(
                    fps=overall_meta.get("fps", "n/a"),
                    mean_ms=overall_meta.get("mean_ms", "n/a"),
                    frames=overall_meta.get("frames", "n/a"),
                )
            )

    print("Completed")


if __name__ == "__main__":
    run_mota_eval()
