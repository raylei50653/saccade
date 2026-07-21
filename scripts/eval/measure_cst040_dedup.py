#!/usr/bin/env python3
"""Measure cst040 + duplicate-suppression end-to-end.

cst040 (confirm_score_thresh 0.40) recovers establishment-latency recall but
adds FP; ~11% of that FP overlaps a previously-confirmed track (duplicate/
fragment, see analyze_confirm_prev_confirmed.py). This post-filters the cst040
output: when two boxes in a frame overlap at IoU>=t, drop the one whose track id
has the SHORTER total lifespan (the likely duplicate), keeping the established
track. Recomputes real motmetrics so we see whether de-dup buys back precision/
AssA without giving up the recall.
"""
# status: experiment

from __future__ import annotations

import argparse
import io
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torchvision  # noqa: E402
from saccade.perception.eval.metrics import run_motmetrics_evaluation as ev  # noqa: E402


def dedup_seq(src: Path, dst: Path, t: float) -> None:
    rows = np.loadtxt(src, delimiter=",", ndmin=2)
    if rows.size == 0:
        np.savetxt(dst, rows, delimiter=",", fmt="%g")
        return
    lifespan: dict[int, int] = defaultdict(int)
    for r in rows:
        lifespan[int(r[1])] += 1
    keep = np.ones(len(rows), dtype=bool)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        by_frame[int(r[0])].append(idx)
    for _, idxs in by_frame.items():
        if len(idxs) < 2:
            continue
        boxes = np.array(
            [
                [
                    rows[i][2],
                    rows[i][3],
                    rows[i][2] + rows[i][4],
                    rows[i][3] + rows[i][5],
                ]
                for i in idxs
            ]
        )
        iou = torchvision.ops.box_iou(
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(boxes, dtype=torch.float32),
        ).numpy()
        np.fill_diagonal(iou, 0.0)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if iou[a, b] < t:
                    continue
                ia, ib = idxs[a], idxs[b]
                la, lb = lifespan[int(rows[ia][1])], lifespan[int(rows[ib][1])]
                drop = ia if la < lb else ib  # drop shorter-lived = likely duplicate
                keep[drop] = False
    np.savetxt(dst, rows[keep], delimiter=",", fmt="%g")


def metrics(out_dir: str, seqs: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return ev(
            data_root="datasets/MOT17", split="train", output=out_dir, sequences=seqs
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="results/conf_baseline")
    p.add_argument("--variant", default="results/conf_cst040")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    p.add_argument("--ts", default="0.5,0.7")
    args = p.parse_args()
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    def line(n, m):
        print(
            f"  {n:22s} IDF1={m['IDF1']} MOTA={m['MOTA']} HOTA={m['HOTA']} "
            f"AssA={m['AssA']} IDs={m['IDs']} Rcll={m['Rcll']} Prcn={m['Prcn']}"
        )

    print("=== cst040 + duplicate-suppression (real motmetrics) ===")
    line("baseline", metrics(args.baseline, ",".join(seqs)))
    line("cst040", metrics(args.variant, ",".join(seqs)))
    for t in (float(x) for x in args.ts.split(",")):
        dst = Path(args.variant).parent / f"{Path(args.variant).name}__dedup{t}"
        dst.mkdir(parents=True, exist_ok=True)
        for s in seqs:
            dedup_seq(Path(args.variant) / f"{s}.txt", dst / f"{s}.txt", t)
        line(f"cst040+dedup(t={t})", metrics(str(dst), ",".join(seqs)))

    # Also de-dup the baseline itself, to separate "dedup helps anyone" from
    # "dedup specifically fixes cst040's extra duplicates".
    dstb = Path(args.baseline).parent / f"{Path(args.baseline).name}__dedup0.5"
    dstb.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        dedup_seq(Path(args.baseline) / f"{s}.txt", dstb / f"{s}.txt", 0.5)
    line("baseline+dedup(t=0.5)", metrics(str(dstb), ",".join(seqs)))


if __name__ == "__main__":
    main()
