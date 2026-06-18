#!/usr/bin/env python
"""Lost-and-recover (reconnection) success-rate analysis.

The role of ReID here is *reconnecting an identity after it is lost* — when a
predicted track drops a GT person (occlusion / missed detections) and that
person later reappears, does the tracker re-attach the SAME predicted id, or
spawn a new one (fragmentation / ID switch)?

Global AssA/IDF1 wash this out (reconnections are rare vs per-frame IoU
associations). This tool measures it directly, stratified by gap length and
spatial displacement, so we can see (a) how many reconnection opportunities
exist, (b) the baseline success rate, (c) whether a relink/merge mechanism
moves it — and whether the extra reconnections are correct.

Definition (per GT id, over its annotated frames):
  * Match GT<->pred each frame by greedy IoU >= --iou.
  * A reconnection *opportunity* = coverage resumes (a frame matched to some
    pred id) after >= --min-gap consecutive lost frames.
  * *Success* = the resumed pred id equals the pred id held just before the gap.

Usage:
  uv run scripts/eval/diagnostics/reconnect_rate.py --pred-dir results/recon_baseline
  uv run scripts/eval/diagnostics/reconnect_rate.py --pred-dir results/recon_merge \
      --label merge   # compare two runs by eye, or pass --baseline-dir to diff
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

GAP_BUCKETS = [(1, 10), (11, 30), (31, 60), (61, 120), (121, 10**9)]


def _load_mot(path: Path, gt: bool) -> dict[int, dict[int, tuple]]:
    """frame -> {id: (x1, y1, x2, y2)}; GT keeps class==1 & flag==1 only."""
    by_frame: dict[int, dict[int, tuple]] = defaultdict(dict)
    if not path.exists():
        return by_frame
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        frame, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        if gt:
            flag = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else 1
            if flag < 1 or cls != 1:
                continue
        by_frame[frame][tid] = (x, y, x + w, y + h)
    return by_frame


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def _match_frame(
    gt: dict[int, tuple], pred: dict[int, tuple], iou_thr: float
) -> dict[int, int]:
    """Greedy IoU matching -> {gt_id: pred_id}."""
    if not gt or not pred:
        return {}
    gids, gboxes = zip(*gt.items())
    pids, pboxes = zip(*pred.items())
    iou = _iou_matrix(
        np.array(gboxes, dtype=np.float64), np.array(pboxes, dtype=np.float64)
    )
    out: dict[int, int] = {}
    used_p: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_g: set[int] = set()
    for gi, pi in order:
        if iou[gi, pi] < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out[gids[gi]] = pids[pi]
    return out


def _center(box: tuple) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def analyze_sequence(
    gt_by_frame: dict[int, dict[int, tuple]],
    pred_by_frame: dict[int, dict[int, tuple]],
    *,
    iou_thr: float = 0.5,
    min_gap: int = 1,
) -> list[dict]:
    """Return one record per reconnection opportunity."""
    # gt_id -> {frame: pred_id} coverage, and gt_id -> {frame: box}
    cover: dict[int, dict[int, int]] = defaultdict(dict)
    gboxes: dict[int, dict[int, tuple]] = defaultdict(dict)
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt = gt_by_frame.get(frame, {})
        for gid, box in gt.items():
            gboxes[gid][frame] = box
        matched = _match_frame(gt, pred_by_frame.get(frame, {}), iou_thr)
        for gid, pid in matched.items():
            cover[gid][frame] = pid

    records: list[dict] = []
    for gid, frame_boxes in gboxes.items():
        ann = sorted(frame_boxes)
        prev_f: int | None = None
        prev_pid: int | None = None
        for f in ann:
            pid = cover[gid].get(f)
            if pid is None:
                continue
            if prev_f is not None and (f - prev_f) > min_gap:
                gap = f - prev_f - 1
                c0 = _center(frame_boxes[prev_f])
                c1 = _center(frame_boxes[f])
                box = frame_boxes[f]
                scale = max(((box[2] - box[0]) * (box[3] - box[1])) ** 0.5, 1.0)
                disp = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5 / scale
                records.append(
                    {"gap": gap, "disp": disp, "success": int(pid == prev_pid)}
                )
            prev_f = f
            prev_pid = pid
    return records


def _summarize(records: list[dict], label: str) -> None:
    n = len(records)
    if n == 0:
        print(f"[{label}] no reconnection opportunities found")
        return
    succ = sum(r["success"] for r in records)
    print(
        f"\n[{label}] reconnection opportunities={n}  "
        f"success={succ}  rate={succ / n * 100:.1f}%"
    )
    print(f"  {'gap frames':>14} | {'opps':>5} | {'success':>7} | {'rate':>6}")
    for lo, hi in GAP_BUCKETS:
        bucket = [r for r in records if lo <= r["gap"] <= hi]
        if not bucket:
            continue
        bs = sum(r["success"] for r in bucket)
        name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(
            f"  {name:>14} | {len(bucket):>5} | {bs:>7} | {bs / len(bucket) * 100:>5.1f}%"
        )


def run(
    pred_dir: Path, gt_root: Path, seqs: list[str], iou: float, min_gap: int
) -> list[dict]:
    all_records: list[dict] = []
    for seq in seqs:
        gt = _load_mot(gt_root / seq / "gt" / "gt.txt", gt=True)
        pred = _load_mot(pred_dir / f"{seq}.txt", gt=False)
        recs = analyze_sequence(gt, pred, iou_thr=iou, min_gap=min_gap)
        all_records.extend(recs)
    return all_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="Dir with <seq>.txt MOT outputs.")
    ap.add_argument(
        "--baseline-dir", default="", help="Optional second dir to diff against."
    )
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument(
        "--sequences", default="", help="Comma list; default = all SDP seqs in gt-root."
    )
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument(
        "--min-gap", type=int, default=1, help="Min lost frames to count as a gap."
    )
    ap.add_argument("--label", default="pred")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    if args.sequences:
        seqs = [s.strip() for s in args.sequences.split(",")]
    else:
        seqs = sorted(
            d.name for d in gt_root.iterdir() if d.is_dir() and d.name.endswith("-SDP")
        )

    main_recs = run(Path(args.pred_dir), gt_root, seqs, args.iou, args.min_gap)
    _summarize(main_recs, args.label)
    if args.baseline_dir:
        base_recs = run(Path(args.baseline_dir), gt_root, seqs, args.iou, args.min_gap)
        _summarize(base_recs, "baseline")


if __name__ == "__main__":
    main()
