#!/usr/bin/env python3
"""Per-sequence HOTA/AssA/IDF1 extractor for a finished output dir.

Usage: _perseq_extract.py <output_dir> [<output_dir> ...]
Reuses the vendored third_party/TrackEval on already-written <seq>.txt files.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TE = ROOT / "third_party" / "TrackEval"
sys.path.insert(0, str(TE))
for name, t in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, name):
        setattr(np, name, t)

import trackeval  # noqa: E402

DATA_ROOT = str(ROOT / "datasets" / "MOT17")
SPLIT = "train"
SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]


def eval_dir(out_dir: str):
    p = Path(out_dir).resolve()
    name = p.name
    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(),
        "DISPLAY_LESS_PROGRESS": True,
        "TIME_PROGRESS": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "USE_PARALLEL": False,
    }
    ds_config = {
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": os.path.join(DATA_ROOT, SPLIT),
        "SKIP_SPLIT_FOL": True,
        "TRACKERS_FOLDER": str(p.parent),
        "TRACKER_SUB_FOLDER": "",
        "TRACKERS_TO_EVAL": [name],
        "SPLIT_TO_EVAL": SPLIT,
        "SEQ_INFO": {s: None for s in SEQS},
    }
    ev = trackeval.Evaluator(eval_config)
    dsl = [trackeval.datasets.MotChallenge2DBox(ds_config)]
    ml = [
        trackeval.metrics.HOTA({"METRICS": ["HOTA"]}),
        trackeval.metrics.Identity(),
        trackeval.metrics.CLEAR(),
    ]
    res, _ = ev.evaluate(dsl, ml)
    tr = next(iter(res.values()))[name]
    rows = {}
    for seq in SEQS + ["COMBINED_SEQ"]:
        c = tr[seq]
        cls = "pedestrian" if "pedestrian" in c else next(iter(c))
        h = c[cls]["HOTA"]
        idf1 = c[cls]["Identity"]["IDF1"]
        cl = c[cls]["CLEAR"]
        rows[seq] = {
            "HOTA": float(np.mean(h["HOTA"])) * 100,
            "AssA": float(np.mean(h["AssA"])) * 100,
            "IDF1": float(idf1) * 100,
            "MOTA": float(cl["MOTA"]) * 100,
            "FP": int(cl["CLR_FP"]),
            "FN": int(cl["CLR_FN"]),
            "IDs": int(cl["IDSW"]),
            "Frag": int(cl["Frag"]),
        }
    return rows


def main():
    dirs = sys.argv[1:]
    allres = {d: eval_dir(d) for d in dirs}
    labels = [Path(d).name for d in dirs]
    if len(dirs) == 2:
        a, b = dirs
        la, lb = labels
        print(f"\n=== {lb} minus {la} (B - baseline) ===")
        hdr = f"{'seq':<16}{'dIDF1':>8}{'dHOTA':>8}{'dAssA':>8}{'dMOTA':>8}{'dFP':>8}{'dFN':>8}{'dIDs':>7}{'dFrag':>7}"
        print(hdr)
        print("-" * len(hdr))
        for seq in SEQS + ["COMBINED_SEQ"]:
            ra, rb = allres[a][seq], allres[b][seq]
            print(
                f"{seq:<16}"
                f"{rb['IDF1'] - ra['IDF1']:>+8.1f}{rb['HOTA'] - ra['HOTA']:>+8.1f}"
                f"{rb['AssA'] - ra['AssA']:>+8.1f}{rb['MOTA'] - ra['MOTA']:>+8.1f}"
                f"{rb['FP'] - ra['FP']:>+8d}{rb['FN'] - ra['FN']:>+8d}"
                f"{rb['IDs'] - ra['IDs']:>+7d}{rb['Frag'] - ra['Frag']:>+7d}"
            )
        return
    print(f"{'seq':<16}" + "".join(f"{label:>26}" for label in labels))
    print(f"{'':16}" + "".join(f"{'HOTA':>8}{'AssA':>9}{'IDF1':>9}" for _ in labels))
    print("-" * (16 + 26 * len(labels)))
    for seq in SEQS + ["COMBINED_SEQ"]:
        line = f"{seq:<16}"
        for d in dirs:
            r = allres[d][seq]
            line += f"{r['HOTA']:>8.1f}{r['AssA']:>9.1f}{r['IDF1']:>9.1f}"
        print(line)


if __name__ == "__main__":
    main()
