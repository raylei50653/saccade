#!/usr/bin/env python3
"""Characterize accepted Cheb-GR tracklet merges from a repair-replay result.

Reads ``results.json`` written by ``run_output_layer_repair_chaining.py`` (the
merge stage records each accepted pair's ``a_id``/``b_id``/``cost``/``gap``),
the frozen pre-interpolation substrate, and the dataset GT. Each accepted pair
is labelled against GT and described by event features:

  correctness   same_gt / diff_gt / undetermined (a side has no GT majority)
  gap           frames between the earlier tracklet's end and the later start
  len_short     length (rows) of the shorter tracklet of the pair
  motion_resid  |CV extrapolation of the earlier tracklet to the later start
                 − later start centre| / earlier end box height
  crowding      other substrate boxes with IoU > 0.1 against the earlier
                 tracklet's end box, in its end frame
  cost          Cheb-GR distance at acceptance

GT labelling: per frame, Hungarian match (IoU >= 0.5) between substrate boxes
and GT boxes with consider-flag 1; a tracklet's GT id is the majority matched
id if it covers >= 50 % of the tracklet's rows, else undetermined.

Only first-pass merge accepts on the substrate are described (merge_only arm).
Chained transitive pairs are not expanded. Counts describe decisions; they are
not a causal decomposition of the metric delta.

Usage
-----
  uv run python scripts/eval/experiments/characterize_merge_events.py \\
      --results <artifact-dir>/results.json \\
      --substrate results/.../substrate --data-root datasets/MOT17 --split train \\
      --out <artifact-dir>/merge_events.json
"""
# status: experiment

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xywh boxes ``a`` [N,4] and ``b`` [M,4]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]),
        0,
        None,
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]),
        0,
        None,
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_mot(path: Path, *, gt: bool) -> dict[int, list[tuple[int, np.ndarray]]]:
    """frame -> [(id, xywh)]. GT keeps consider-flag == 1 rows only."""
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        if gt and len(f) > 6 and float(f[6]) < 1:
            continue
        by_frame[int(float(f[0]))].append(
            (int(float(f[1])), np.array([float(v) for v in f[2:6]]))
        )
    return by_frame


def gt_majority(
    subs: dict[int, list[tuple[int, np.ndarray]]],
    gts: dict[int, list[tuple[int, np.ndarray]]],
) -> dict[int, int | None]:
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    rows: Counter[int] = Counter()
    for frame, dets in subs.items():
        for tid, _ in dets:
            rows[tid] += 1
        g = gts.get(frame, [])
        if not g:
            continue
        iou = _iou_matrix(np.stack([b for _, b in dets]), np.stack([b for _, b in g]))
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= 0.5:
                votes[dets[i][0]][g[j][0]] += 1
    out: dict[int, int | None] = {}
    for tid, n in rows.items():
        top = votes[tid].most_common(1)
        out[tid] = top[0][0] if top and top[0][1] * 2 >= n else None
    return out


def track_table(
    subs: dict[int, list[tuple[int, np.ndarray]]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    tracks: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for frame in sorted(subs):
        for tid, box in subs[frame]:
            tracks[tid].append((frame, box))
    return tracks


def pair_features(
    a: int,
    b: int,
    tracks: dict[int, list[tuple[int, np.ndarray]]],
    subs: dict[int, list[tuple[int, np.ndarray]]],
) -> dict[str, Any]:
    ta, tb = tracks[a], tracks[b]
    earlier, later = (ta, tb) if ta[-1][0] <= tb[-1][0] else (tb, ta)
    e_id = a if earlier is ta else b
    end_f, end_box = earlier[-1]
    start_f, start_box = later[0]
    tail = earlier[-5:]
    c_end = end_box[:2] + end_box[2:] / 2
    if len(tail) >= 2 and tail[-1][0] > tail[0][0]:
        c0 = tail[0][1][:2] + tail[0][1][2:] / 2
        v = (c_end - c0) / (tail[-1][0] - tail[0][0])
    else:
        v = np.zeros(2)
    pred = c_end + v * (start_f - end_f)
    c_start = start_box[:2] + start_box[2:] / 2
    resid = float(np.linalg.norm(pred - c_start) / max(end_box[3], 1.0))
    others = [box for tid, box in subs.get(end_f, []) if tid != e_id]
    crowd = (
        int((_iou_matrix(end_box[None], np.stack(others))[0] > 0.1).sum())
        if others
        else 0
    )
    return {
        "len_short": int(min(len(ta), len(tb))),
        "len_long": int(max(len(ta), len(tb))),
        "motion_resid": resid,
        "crowding": crowd,
        "end_height": float(end_box[3]),
    }


def _q(vals: list[float]) -> dict[str, float] | None:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
    }


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(list)
    for e in events:
        by[e["correctness"]].append(e)
    feats = ["gap", "len_short", "motion_resid", "crowding", "cost"]
    judged = len(by["same_gt"]) + len(by["diff_gt"])
    return {
        "accepted": len(events),
        "same_gt": len(by["same_gt"]),
        "diff_gt": len(by["diff_gt"]),
        "undetermined": len(by["undetermined"]),
        "judged_precision": (len(by["same_gt"]) / judged) if judged else None,
        "features": {
            label: {f: _q([float(e[f]) for e in group]) for f in feats}
            for label, group in by.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--substrate", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split", default="train")
    p.add_argument("--arm", default="merge_only")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    payload = json.loads(args.results.read_text())
    row = next(r for r in payload["primary"] if r["arm"] == args.arm)
    events: list[dict[str, Any]] = []
    per_seq: dict[str, Any] = {}
    for seq, recs in row["seq_stage_records"].items():
        merge_recs = [r for r in recs if r["stage"] == "merge"]
        if not merge_recs:
            continue
        pairs = merge_recs[0].get("accepted_pairs", [])
        subs = load_mot(args.substrate / f"{seq}.txt", gt=False)
        gts = load_mot(args.data_root / args.split / seq / "gt" / "gt.txt", gt=True)
        maj = gt_majority(subs, gts)
        tracks = track_table(subs)
        seq_events = []
        for pr in pairs:
            ga, gb = maj.get(pr["a_id"]), maj.get(pr["b_id"])
            if ga is None or gb is None:
                label = "undetermined"
            else:
                label = "same_gt" if ga == gb else "diff_gt"
            ev = {
                "seq": seq,
                **pr,
                "correctness": label,
                **pair_features(pr["a_id"], pr["b_id"], tracks, subs),
            }
            seq_events.append(ev)
        events.extend(seq_events)
        per_seq[seq] = {
            "accepted": len(seq_events),
            "same_gt": sum(e["correctness"] == "same_gt" for e in seq_events),
            "diff_gt": sum(e["correctness"] == "diff_gt" for e in seq_events),
            "undetermined": sum(e["correctness"] == "undetermined" for e in seq_events),
        }
    out = {
        "schema": "merge_event_characterization/v1",
        "results": str(args.results),
        "arm": args.arm,
        "summary": summarize(events),
        "per_sequence": per_seq,
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    s = out["summary"]
    print(
        f"accepted={s['accepted']} same_gt={s['same_gt']} diff_gt={s['diff_gt']} "
        f"undetermined={s['undetermined']} judged_precision={s['judged_precision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
