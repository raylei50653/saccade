import os
import glob
from pathlib import Path
from collections import OrderedDict
from typing import Any
import numpy as np

def run_motmetrics_evaluation(
    data_root: str,
    split: str,
    output: str,
    sequences: str,
    detector: str | None = None,
) -> dict[str, Any] | None:
    try:
        import motmetrics as mm
    except ImportError:
        print("motmetrics not found, skipping evaluation.")
        return None

    # Force NumPy 2.0 compatibility
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

    gt_folder = os.path.join(data_root, split)
    gtfiles = glob.glob(os.path.join(gt_folder, "*/gt/gt.txt"))
    # Filter by selected sequences and detector
    if sequences:
        seq_list = sequences.split(",")
        gtfiles = [f for f in gtfiles if Path(f).parts[-3] in seq_list]
    elif detector:
        gtfiles = [f for f in gtfiles if f"-{detector}" in Path(f).parts[-3]]

    tsfiles = sorted(glob.glob(os.path.join(output, "*.txt")))
    # Filter tsfiles similarly
    if sequences:
        seq_list = sequences.split(",")
        tsfiles = [f for f in tsfiles if Path(f).stem in seq_list]

    if not gtfiles or not tsfiles:
        return None

    gt = OrderedDict(
        [
            (Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1))
            for f in gtfiles
        ]
    )
    ts = OrderedDict(
        [
            (Path(f).stem, mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1.0))
            for f in tsfiles
        ]
    )

    mh = mm.metrics.create()
    accs = []
    names = []
    for k, tsacc in ts.items():
        matched_gt = None
        for gt_name in gt.keys():
            if k == gt_name or k.startswith(gt_name + "_"):
                matched_gt = gt_name
                break
        if matched_gt:
            accs.append(
                mm.utils.compare_to_groundtruth(gt[matched_gt], tsacc, "iou", distth=0.5)
            )
            names.append(k)

    if not accs:
        return None

    metrics_list = mm.metrics.motchallenge_metrics + ["num_objects"]
    summary = mh.compute_many(
        accs, names=names, metrics=metrics_list, generate_overall=True
    )

    # Return OVERALL metrics
    overall = summary.loc["OVERALL"]
    return {
        "IDF1": f"{overall['idf1']*100:.1f}%",
        "MOTA": f"{overall['mota']*100:.1f}%",
        "IDs": int(overall["num_switches"]),
        "FP": int(overall["num_false_positives"]),
        "FN": int(overall["num_misses"]),
        "Rcll": f"{overall['recall']*100:.1f}%",
        "Prcn": f"{overall['precision']*100:.1f}%",
    }
