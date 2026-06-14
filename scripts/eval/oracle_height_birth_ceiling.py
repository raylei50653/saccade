#!/usr/bin/env python3
"""Oracle ceiling for a height-conditioned birth/output gate (no detector/tracker).

Takes an existing MOT result dir and (1) labels every output box TP/FP by GT
match, tabulating the FP mass by height x score, then (2) post-filters boxes
where height>=H_pivot AND score<S_thresh, recomputing IDF1/MOTA/Prcn via real
motmetrics. This upper-bounds the precision gain of "kill large low-score
boxes" -- a superset of what a height-ramped new_track_thresh could achieve
(the filter removes such boxes at every frame, not only at birth).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torchvision  # noqa: E402
from saccade.perception.eval.metrics import run_motmetrics_evaluation  # noqa: E402

HEIGHT_BINS = ((0, 64), (64, 128), (128, 256), (256, 10**9))
SCORE_BANDS = ((0, 0.15), (0.15, 0.28), (0.28, 0.40), (0.40, 0.50), (0.50, 1.01))


def _load_mot(path: Path) -> dict[int, list[tuple[float, float, float, float, float]]]:
    """frame -> list of (x, y, w, h, score)."""
    by_frame: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        by_frame[int(r[0])].append((r[2], r[3], r[4], r[5], r[6]))
    return by_frame


def _load_gt(path: Path) -> dict[int, list[list[float]]]:
    by_frame: dict[int, list] = defaultdict(list)
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in rows:
        if int(r[6]) != 1 or int(r[7]) != 1 or (len(r) > 8 and r[8] < 0.1):
            continue
        x, y, w, h = r[2], r[3], r[4], r[5]
        by_frame[int(r[0])].append([x, y, x + w, y + h])
    return by_frame


def diagnose(data_root: str, split: str, seqs: list[str], out_dir: Path) -> None:
    # FP/TP output-box mass by height x score, pooled over sequences.
    fp = np.zeros((len(HEIGHT_BINS), len(SCORE_BANDS)))
    tp = np.zeros_like(fp)
    for seq in seqs:
        mot = _load_mot(out_dir / f"{seq}.txt")
        gt = _load_gt(Path(data_root) / split / seq / "gt" / "gt.txt")
        for frame, boxes in mot.items():
            if not boxes:
                continue
            arr = np.array(boxes)  # x,y,w,h,score
            xyxy = torch.tensor(
                np.stack(
                    [
                        arr[:, 0],
                        arr[:, 1],
                        arr[:, 0] + arr[:, 2],
                        arr[:, 1] + arr[:, 3],
                    ],
                    1,
                ),
                dtype=torch.float32,
            )
            scores = arr[:, 4]
            heights = arr[:, 3]
            gtb = torch.tensor(gt.get(frame, []), dtype=torch.float32).reshape(-1, 4)
            is_tp = np.zeros(len(boxes), dtype=bool)
            if gtb.numel():
                ious = torchvision.ops.box_iou(xyxy, gtb)
                used = torch.zeros(gtb.shape[0], dtype=torch.bool)
                for di in np.argsort(-scores).tolist():
                    av = ious[di].masked_fill(used, -1.0)
                    bi, gi = av.max(dim=0)
                    if float(bi) >= 0.5:
                        used[gi] = True
                        is_tp[di] = True
            for di in range(len(boxes)):
                hi = next(
                    i for i, (lo, h) in enumerate(HEIGHT_BINS) if lo <= heights[di] < h
                )
                si = next(
                    i for i, (lo, h) in enumerate(SCORE_BANDS) if lo <= scores[di] < h
                )
                (tp if is_tp[di] else fp)[hi, si] += 1
    print("=== output-box FP count by height x score (FP / total, precision) ===")
    hdr = f"{'height':12s}" + "".join(f"{f's<{b[1]}':>14s}" for b in SCORE_BANDS)
    print(hdr)
    for hi, (lo, h) in enumerate(HEIGHT_BINS):
        row = (
            f"{f'h[{lo},{h if h < 10**8 else 0})':12s}"
            if h < 10**8
            else f"{f'h>={lo}':12s}"
        )
        for si in range(len(SCORE_BANDS)):
            n = tp[hi, si] + fp[hi, si]
            prec = tp[hi, si] / n if n else float("nan")
            row += f"{f'{int(fp[hi, si])}/{int(n)}={prec:.2f}':>14s}"
        print(row)
    print(
        f"\ntotal output boxes: {int((tp + fp).sum())}  FP: {int(fp.sum())}  "
        f"global precision: {tp.sum() / (tp.sum() + fp.sum()):.4f}"
    )
    # The removable cluster: large + low score.
    print("\nremovable 'large x low-score' clusters (FP / total):")
    for hpiv in (128, 256):
        for spiv in (0.28, 0.40, 0.50):
            mask_h = [i for i, (lo, h) in enumerate(HEIGHT_BINS) if lo >= hpiv]
            mask_s = [i for i, (lo, h) in enumerate(SCORE_BANDS) if h <= spiv]
            f = fp[np.ix_(mask_h, mask_s)].sum()
            t = tp[np.ix_(mask_h, mask_s)].sum()
            print(
                f"  h>={hpiv}, s<{spiv}: remove {int(f + t)} boxes "
                f"({int(f)} FP, {int(t)} TP)  -> precision of cut {f / (f + t) if f + t else float('nan'):.2f}"
            )


def post_filter_eval(
    data_root: str, split: str, seqs: list[str], src: Path, hpiv: float, spiv: float
) -> dict:
    dst = src.parent / f"{src.name}__h{int(hpiv)}_s{spiv}"
    dst.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        rows = np.loadtxt(src / f"{seq}.txt", delimiter=",", ndmin=2)
        keep = ~((rows[:, 5] >= hpiv) & (rows[:, 6] < spiv))
        np.savetxt(dst / f"{seq}.txt", rows[keep], delimiter=",", fmt="%g")
    return run_motmetrics_evaluation(
        data_root=data_root, split=split, output=str(dst), sequences=",".join(seqs)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/strat_baseline")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    p.add_argument("--grid", default="128:0.28,128:0.40,256:0.40,256:0.50")
    args = p.parse_args()
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    out_dir = Path(args.output_dir)

    diagnose(args.data_root, args.split, seqs, out_dir)

    print("\n=== oracle post-filter: recomputed metrics ===")
    base = run_motmetrics_evaluation(
        data_root=args.data_root,
        split=args.split,
        output=str(out_dir),
        sequences=",".join(seqs),
    )

    def fmt(m):
        return (
            f"IDF1={m['IDF1']} MOTA={m['MOTA']} HOTA={m['HOTA']} "
            f"IDs={m['IDs']} Rcll={m['Rcll']} Prcn={m['Prcn']}"
        )

    print(f"  baseline           {fmt(base)}")
    for spec in args.grid.split(","):
        hpiv, spiv = spec.split(":")
        m = post_filter_eval(
            args.data_root, args.split, seqs, out_dir, float(hpiv), float(spiv)
        )
        print(f"  h>={hpiv} s<{spiv:5s}  {fmt(m)}")


if __name__ == "__main__":
    main()
