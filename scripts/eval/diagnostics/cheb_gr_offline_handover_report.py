#!/usr/bin/env python
"""Label and analyze Cheb-GR offline handover decisions.

The handover CSV contains the decision-side features (Cheb-GR cost/margin,
geometry, scores). This report joins those rows with MOT17 GT via a baseline
pre-handover output, labels each proposed bridge as correct/wrong/unknown, and
prints feature distributions plus simple one/two-feature gate candidates.

Usage:
  uv run scripts/eval/diagnostics/cheb_gr_offline_handover_report.py \
    --handover-log results/run/_cheb_gr_offline_handover.csv \
    --baseline-dir results/no_handover \
    --pred-dir results/run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.diagnostics.reconnect_rate import _load_mot, _match_frame


FEATURES: tuple[str, ...] = (
    "best_cost",
    "margin",
    "match_iou",
    "direct_iou",
    "candidate_forward_iou",
    "newborn_backward_iou",
    "neighbor_iou",
    "head_tail_neighbor_iou",
    "newborn_neighbor_iou",
    "candidate_neighbor_iou",
    "center_dist_norm",
    "gap",
    "candidate_count",
    "head_n",
    "bank_n",
    "newborn_mean_score",
    "newborn_start_score",
    "candidate_mean_score",
    "candidate_end_score",
)

GATE_FEATURES: tuple[str, ...] = (
    "best_cost",
    "margin",
    "match_iou",
    "direct_iou",
    "candidate_forward_iou",
    "newborn_backward_iou",
    "neighbor_iou",
    "head_tail_neighbor_iou",
    "center_dist_norm",
    "gap",
    "newborn_mean_score",
    "candidate_mean_score",
)

REGISTRY_SPECS: dict[str, dict[str, Any]] = {
    "best_cost": {
        "meaning": "Cheb-GR local gallery distance; lower means stronger appearance/graph support.",
        "edges": (0.25, 0.35, 0.45, 0.50, 0.55, 0.60),
        "use_with": "margin, center_dist_norm, gap",
        "failure": "Can still confirm an already-polluted tracklet; audit local vs full labels.",
    },
    "margin": {
        "meaning": "Top1-vs-top2 Cheb-GR separation; higher means less candidate ambiguity.",
        "edges": (0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30),
        "use_with": "best_cost, candidate_count",
        "failure": "Single clear wrong candidate can have a large margin.",
    },
    "center_dist_norm": {
        "meaning": "Motion-projected center distance normalized by box height; lower is cleaner geometry.",
        "edges": (0.5, 1.0, 2.0, 4.0, 8.0),
        "use_with": "best_cost, gap",
        "failure": "Low distance in crowd scenes can be overlap pollution, not identity confidence.",
    },
    "match_iou": {
        "meaning": "Candidate tail box projected to newborn start vs newborn start box IoU.",
        "edges": (0.1, 0.3, 0.5, 0.7, 0.9),
        "use_with": "neighbor_iou, best_cost",
        "failure": "High match IoU can mean stable geometry or two people mixed in one overlap region.",
    },
    "neighbor_iou": {
        "meaning": "Max same-frame overlap around either endpoint; crowd/crop pollution context.",
        "edges": (0.1, 0.3, 0.5, 0.7),
        "use_with": "match_iou, head_tail_neighbor_iou, crop/ReID quality",
        "failure": "Not a standalone identity signal; it marks when appearance evidence may be unsafe.",
    },
    "gap": {
        "meaning": "Frames between candidate death and newborn start.",
        "edges": (3, 10, 30, 60),
        "use_with": "center_dist_norm, best_cost",
        "failure": "Short gaps are not automatically safe when the endpoint is polluted.",
    },
    "candidate_count": {
        "meaning": "Number of in-window archive candidates in the event-local graph.",
        "edges": (2, 3, 5, 10),
        "use_with": "margin",
        "failure": "More candidates can make top1 meaningful, but also indicate a crowded scene.",
    },
}

STABILITY_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "best_cost_accept_strict",
        "intent": "candidate accept zone",
        "decision": "accept-candidate",
        "use": "Strongest single-feature accept candidate; still audit endpoint pollution.",
        "gates": (("best_cost", 0.25, "low"),),
    },
    {
        "name": "best_cost_accept_broad",
        "intent": "candidate accept/gray boundary",
        "decision": "gray/support",
        "use": "Useful as a broad prefilter; combine with margin or pollution context before accepting.",
        "gates": (("best_cost", 0.35, "low"),),
    },
    {
        "name": "best_cost_danger",
        "intent": "reject/veto context",
        "decision": "reject/veto",
        "use": "Stable wrong-link region; good veto context for offline handover candidates.",
        "gates": (("best_cost", 0.50, "high"),),
    },
    {
        "name": "margin_accept",
        "intent": "candidate accept zone",
        "decision": "support-only",
        "use": "Not stable enough alone; combine with candidate_count or best_cost.",
        "gates": (("margin", 0.12, "high"),),
    },
    {
        "name": "margin_danger",
        "intent": "reject/veto context",
        "decision": "reject/veto",
        "use": "Stable ambiguity region; low margin should block appearance-only handover.",
        "gates": (("margin", 0.05, "low"),),
    },
    {
        "name": "center_dist_support",
        "intent": "geometry support zone",
        "decision": "support-only",
        "use": "Geometry support, not identity proof; combine with cost/margin and overlap pollution.",
        "gates": (("center_dist_norm", 0.50, "low"),),
    },
    {
        "name": "center_dist_danger",
        "intent": "geometry reject/veto context",
        "decision": "reject/veto",
        "use": "Stable geometry danger region; block long spatial jumps unless another mechanism explains them.",
        "gates": (("center_dist_norm", 2.00, "high"),),
    },
    {
        "name": "margin_x_candidate_count_accept",
        "intent": "candidate accept zone",
        "decision": "accept-candidate",
        "use": "Best current ambiguity-aware accept combo; still needs independent detector/preset validation.",
        "gates": (("margin", 0.12, "high"), ("candidate_count", 5.0, "high")),
    },
    {
        "name": "best_cost_x_margin_accept",
        "intent": "conservative accept zone",
        "decision": "accept-candidate",
        "use": "Conservative appearance+ambiguity combo; prefer over margin alone.",
        "gates": (("best_cost", 0.35, "low"), ("margin", 0.12, "high")),
    },
)


@dataclass(frozen=True)
class TrackMatch:
    gt_id: int | None
    matches: int
    frames: int
    purity: float


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _read_handover_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = dict(row)
            for key in (
                "newborn_id",
                "newborn_start",
                "newborn_end",
                "candidate_id",
                "candidate_start",
                "candidate_end",
            ):
                out[key] = int(float(out[key]))
            for key in FEATURES + ("second_cost", "required_margin", "max_cost"):
                if key in out:
                    out[key] = _parse_float(out[key])
            out["accepted"] = _parse_bool(out.get("accepted", False))
            rows.append(out)
    return rows


def _sequence_names(
    gt_root: Path, explicit: str, rows: list[dict[str, Any]]
) -> list[str]:
    if explicit:
        return [s.strip() for s in explicit.split(",") if s.strip()]
    if rows:
        return sorted({str(r["seq"]) for r in rows})
    return sorted(
        d.name for d in gt_root.iterdir() if d.is_dir() and d.name.endswith("-SDP")
    )


def _build_pred_gt_index(
    *,
    seq: str,
    baseline_dir: Path,
    gt_root: Path,
    iou: float,
) -> dict[tuple[int, int], int]:
    gt = _load_mot(gt_root / seq / "gt" / "gt.txt", gt=True)
    pred = _load_mot(baseline_dir / f"{seq}.txt", gt=False)
    index: dict[tuple[int, int], int] = {}
    for frame in sorted(set(gt) | set(pred)):
        matches = _match_frame(gt.get(frame, {}), pred.get(frame, {}), iou)
        for gid, pid in matches.items():
            index[(frame, pid)] = gid
    return index


def _majority(
    index: dict[tuple[int, int], int],
    *,
    track_id: int,
    start: int,
    end: int,
) -> TrackMatch:
    frames = max(0, end - start + 1)
    counts: Counter[int] = Counter()
    for frame in range(start, end + 1):
        gid = index.get((frame, track_id))
        if gid is not None:
            counts[gid] += 1
    if not counts:
        return TrackMatch(None, 0, frames, 0.0)
    gt_id, matches = counts.most_common(1)[0]
    return TrackMatch(gt_id, matches, frames, matches / max(1, sum(counts.values())))


def _annotate_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_dir: Path,
    gt_root: Path,
    iou: float,
    edge_window: int,
    label_mode: str,
) -> list[dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seq[str(row["seq"])].append(row)

    annotated: list[dict[str, Any]] = []
    for seq, seq_rows in sorted(by_seq.items()):
        index = _build_pred_gt_index(
            seq=seq,
            baseline_dir=baseline_dir,
            gt_root=gt_root,
            iou=iou,
        )
        for row in seq_rows:
            cand = _majority(
                index,
                track_id=int(row["candidate_id"]),
                start=int(row["candidate_start"]),
                end=int(row["candidate_end"]),
            )
            cand_tail_start = max(
                int(row["candidate_start"]),
                int(row["candidate_end"]) - edge_window + 1,
            )
            cand_tail = _majority(
                index,
                track_id=int(row["candidate_id"]),
                start=cand_tail_start,
                end=int(row["candidate_end"]),
            )
            newborn = _majority(
                index,
                track_id=int(row["newborn_id"]),
                start=int(row["newborn_start"]),
                end=int(row["newborn_end"]),
            )
            head_end = min(
                int(row["newborn_end"]),
                int(row["newborn_start"]) + edge_window - 1,
            )
            newborn_head = _majority(
                index,
                track_id=int(row["newborn_id"]),
                start=int(row["newborn_start"]),
                end=head_end,
            )
            same_full = cand.gt_id is not None and cand.gt_id == newborn.gt_id
            same_head = cand.gt_id is not None and cand.gt_id == newborn_head.gt_id
            same_local = (
                cand_tail.gt_id is not None and cand_tail.gt_id == newborn_head.gt_id
            )
            known_full = cand.gt_id is not None and newborn.gt_id is not None
            known_head = cand.gt_id is not None and newborn_head.gt_id is not None
            known_local = cand_tail.gt_id is not None and newborn_head.gt_id is not None
            if label_mode == "full":
                label = (
                    "correct" if same_full else ("wrong" if known_full else "unknown")
                )
            else:
                label = (
                    "correct" if same_local else ("wrong" if known_local else "unknown")
                )
            out = dict(row)
            out.update(
                {
                    "candidate_gt": cand.gt_id if cand.gt_id is not None else "",
                    "candidate_gt_matches": cand.matches,
                    "candidate_gt_frames": cand.frames,
                    "candidate_gt_purity": cand.purity,
                    "candidate_tail_gt": (
                        cand_tail.gt_id if cand_tail.gt_id is not None else ""
                    ),
                    "candidate_tail_gt_matches": cand_tail.matches,
                    "candidate_tail_gt_frames": cand_tail.frames,
                    "candidate_tail_gt_purity": cand_tail.purity,
                    "newborn_gt": newborn.gt_id if newborn.gt_id is not None else "",
                    "newborn_gt_matches": newborn.matches,
                    "newborn_gt_frames": newborn.frames,
                    "newborn_gt_purity": newborn.purity,
                    "newborn_head_gt": (
                        newborn_head.gt_id if newborn_head.gt_id is not None else ""
                    ),
                    "newborn_head_gt_matches": newborn_head.matches,
                    "newborn_head_gt_frames": newborn_head.frames,
                    "newborn_head_gt_purity": newborn_head.purity,
                    "known_full": known_full,
                    "same_gt_full": same_full,
                    "known_head": known_head,
                    "same_gt_head": same_head,
                    "known_local": known_local,
                    "same_gt_local": same_local,
                    "candidate_full_tail_same": (
                        cand.gt_id is not None
                        and cand_tail.gt_id is not None
                        and cand.gt_id == cand_tail.gt_id
                    ),
                    "newborn_full_head_same": (
                        newborn.gt_id is not None
                        and newborn_head.gt_id is not None
                        and newborn.gt_id == newborn_head.gt_id
                    ),
                    "label": label,
                }
            )
            annotated.append(out)
    return annotated


def _quantiles(values: list[float]) -> str:
    xs = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    if xs.size == 0:
        return "-"
    qs = np.quantile(xs, [0.1, 0.25, 0.5, 0.75, 0.9])
    return " ".join(f"{q:.3g}" for q in qs)


def _auc(values: list[float], labels: list[int]) -> float:
    pairs = [(v, y) for v, y in zip(values, labels) if math.isfinite(v)]
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs.sort(key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) * 0.5
        rank_sum += avg_rank * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg)


def _known(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["label"] in {"correct", "wrong"}]


def _feature_summary(rows: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    known = _known(rows)
    labels = [1 if r["label"] == "correct" else 0 for r in known]
    orientations: dict[str, tuple[str, float]] = {}
    print("\nFeature distributions by GT label")
    print(
        f"  {'feature':<24} {'correct q10/25/50/75/90':<28} "
        f"{'wrong q10/25/50/75/90':<28} {'best_auc':>8} {'dir':>5}"
    )
    for feat in FEATURES:
        corr = [_parse_float(r.get(feat)) for r in rows if r["label"] == "correct"]
        wrong = [_parse_float(r.get(feat)) for r in rows if r["label"] == "wrong"]
        values = [_parse_float(r.get(feat)) for r in known]
        auc = _auc(values, labels)
        if math.isnan(auc):
            best_auc = float("nan")
            direction = "high"
        elif auc >= 0.5:
            best_auc = auc
            direction = "high"
        else:
            best_auc = 1.0 - auc
            direction = "low"
        orientations[feat] = (direction, best_auc)
        auc_text = "-" if math.isnan(best_auc) else f"{best_auc:.3f}"
        print(
            f"  {feat:<24} {_quantiles(corr):<28} "
            f"{_quantiles(wrong):<28} {auc_text:>8} {direction:>5}"
        )
    return orientations


def _candidate_thresholds(values: list[float]) -> list[float]:
    xs = np.array(sorted({v for v in values if math.isfinite(v)}), dtype=np.float64)
    if xs.size == 0:
        return []
    if xs.size <= 20:
        return [float(v) for v in xs]
    qs = np.quantile(xs, np.linspace(0.05, 0.95, 19))
    return sorted({float(v) for v in qs})


def _passes(row: dict[str, Any], feat: str, thr: float, direction: str) -> bool:
    value = _parse_float(row.get(feat))
    if not math.isfinite(value):
        return False
    if direction == "high":
        return value >= thr
    return value <= thr


def _passes_gates(
    row: dict[str, Any], gates: tuple[tuple[str, float, str], ...]
) -> bool:
    return all(_passes(row, feat, thr, direction) for feat, thr, direction in gates)


def _eval_gate(
    rows: list[dict[str, Any]],
    gates: list[tuple[str, float, str]],
) -> dict[str, float]:
    selected = [
        r
        for r in rows
        if _passes_gates(
            r, tuple((feat, thr, direction) for feat, thr, direction in gates)
        )
    ]
    correct_total = sum(r["label"] == "correct" for r in rows)
    wrong_total = sum(r["label"] == "wrong" for r in rows)
    correct = sum(r["label"] == "correct" for r in selected)
    wrong = sum(r["label"] == "wrong" for r in selected)
    precision = correct / max(1, correct + wrong)
    recall = correct / max(1, correct_total)
    wrong_keep = wrong / max(1, wrong_total)
    return {
        "selected": float(len(selected)),
        "correct": float(correct),
        "wrong": float(wrong),
        "precision": precision,
        "recall": recall,
        "wrong_keep": wrong_keep,
    }


def _gate_expr(gates: list[tuple[str, float, str]]) -> str:
    parts = []
    for feat, thr, direction in gates:
        op = ">=" if direction == "high" else "<="
        parts.append(f"{feat} {op} {thr:.3g}")
    return " && ".join(parts)


def _gate_record(
    *,
    gates: list[tuple[str, float, str]],
    metrics: dict[str, float],
    score: float,
) -> dict[str, Any]:
    return {
        "expression": _gate_expr(gates),
        "gates": [
            {
                "feature": feat,
                "threshold": thr,
                "direction": direction,
                "operator": ">=" if direction == "high" else "<=",
            }
            for feat, thr, direction in gates
        ],
        "score": score,
        "selected": int(metrics["selected"]),
        "correct": int(metrics["correct"]),
        "wrong": int(metrics["wrong"]),
        "precision": metrics["precision"],
        "correct_recall": metrics["recall"],
        "wrong_keep": metrics["wrong_keep"],
    }


def _rank_single_feature_gates(
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    min_selected: int,
    limit: int = 12,
) -> list[dict[str, Any]]:
    known = _known(rows)
    if not known:
        return []
    candidates: list[dict[str, Any]] = []
    for feat in GATE_FEATURES:
        direction = orientations.get(feat, ("high", 0.0))[0]
        values = [_parse_float(r.get(feat)) for r in known]
        for thr in _candidate_thresholds(values):
            gates = [(feat, thr, direction)]
            metrics = _eval_gate(known, gates)
            if metrics["selected"] < min_selected:
                continue
            score = (
                metrics["precision"]
                + 0.35 * metrics["recall"]
                - 0.2 * metrics["wrong_keep"]
            )
            candidates.append(_gate_record(gates=gates, metrics=metrics, score=score))
    candidates.sort(
        key=lambda item: (
            item["precision"],
            -item["wrong"],
            item["correct_recall"],
            item["score"],
        ),
        reverse=True,
    )
    return candidates[:limit]


def _rank_pair_gates(
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    min_correct_recall: float,
    limit: int = 12,
) -> list[dict[str, Any]]:
    known = _known(rows)
    if not known:
        return []
    single: dict[str, list[tuple[float, str]]] = {}
    for feat in GATE_FEATURES:
        direction = orientations.get(feat, ("high", 0.0))[0]
        values = [_parse_float(r.get(feat)) for r in known]
        single[feat] = [(thr, direction) for thr in _candidate_thresholds(values)]

    candidates: list[dict[str, Any]] = []
    for i, feat_a in enumerate(GATE_FEATURES):
        for feat_b in GATE_FEATURES[i + 1 :]:
            for thr_a, dir_a in single[feat_a]:
                for thr_b, dir_b in single[feat_b]:
                    gates = [(feat_a, thr_a, dir_a), (feat_b, thr_b, dir_b)]
                    metrics = _eval_gate(known, gates)
                    if metrics["recall"] < min_correct_recall:
                        continue
                    score = metrics["precision"] - 0.25 * metrics["wrong_keep"]
                    candidates.append(
                        _gate_record(gates=gates, metrics=metrics, score=score)
                    )
    candidates.sort(
        key=lambda item: (
            item["precision"],
            -item["wrong"],
            item["correct_recall"],
            item["score"],
        ),
        reverse=True,
    )
    return candidates[:limit]


def _print_gate_table(
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    title: str,
    min_selected: int,
) -> None:
    known = _known(rows)
    if not known:
        return
    print(f"\n{title}")
    print(
        f"  {'gate':<34} {'sel':>5} {'ok':>4} {'bad':>4} "
        f"{'prec':>6} {'recall':>6} {'bad_keep':>8}"
    )
    for gate in _rank_single_feature_gates(
        known,
        orientations,
        min_selected=min_selected,
    ):
        print(
            f"  {gate['expression']:<34} {gate['selected']:5d} "
            f"{gate['correct']:4d} {gate['wrong']:4d} "
            f"{gate['precision']:6.3f} {gate['correct_recall']:6.3f} "
            f"{gate['wrong_keep']:8.3f}"
        )


def _print_pair_gates(
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    min_correct_recall: float,
) -> None:
    known = _known(rows)
    if not known:
        return
    print(
        f"\nTwo-feature gates on accepted known rows (recall >= {min_correct_recall:.2f})"
    )
    print(
        f"  {'gate':<74} {'sel':>5} {'ok':>4} {'bad':>4} "
        f"{'prec':>6} {'recall':>6} {'bad_keep':>8}"
    )
    for gate in _rank_pair_gates(
        known,
        orientations,
        min_correct_recall=min_correct_recall,
    ):
        print(
            f"  {gate['expression']:<74} {gate['selected']:5d} "
            f"{gate['correct']:4d} {gate['wrong']:4d} "
            f"{gate['precision']:6.3f} {gate['correct_recall']:6.3f} "
            f"{gate['wrong_keep']:8.3f}"
        )


def _bucket(value: float, edges: tuple[float, ...]) -> str:
    if not math.isfinite(value):
        return "nan"
    prev = float("-inf")
    for edge in edges:
        if value < edge:
            lo = "-inf" if prev == float("-inf") else f"{prev:g}"
            return f"[{lo},{edge:g})"
        prev = edge
    return f"[{prev:g},inf)"


def _bucket_names(edges: tuple[float, ...]) -> list[str]:
    names = []
    prev = float("-inf")
    for edge in edges:
        lo = "-inf" if prev == float("-inf") else f"{prev:g}"
        names.append(f"[{lo},{edge:g})")
        prev = edge
    names.append(f"[{prev:g},inf)")
    return names


def _bucket_matrix(
    rows: list[dict[str, Any]],
    *,
    x_feat: str,
    x_edges: tuple[float, ...],
    y_feat: str,
    y_edges: tuple[float, ...],
    title: str,
) -> None:
    known = _known(rows)
    if not known:
        return
    buckets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    x_names = _bucket_names(x_edges)
    y_names = _bucket_names(y_edges)
    for row in known:
        xb = _bucket(_parse_float(row.get(x_feat)), x_edges)
        yb = _bucket(_parse_float(row.get(y_feat)), y_edges)
        buckets[(xb, yb)][row["label"]] += 1
    print(f"\n{title}")
    print(f"  cell = correct/total same_gt_rate; rows={y_feat}, cols={x_feat}")
    header = "  " + f"{y_feat:<18}" + "".join(f"{x:>18}" for x in x_names)
    print(header)
    for yb in y_names:
        line = "  " + f"{yb:<18}"
        for xb in x_names:
            c = buckets[(xb, yb)]
            ok = c["correct"]
            bad = c["wrong"]
            total = ok + bad
            if total == 0:
                cell = "-"
            else:
                cell = f"{ok}/{total} {ok / total:.2f}"
            line += f"{cell:>18}"
        print(line)


def _print_bucket_maps(rows: list[dict[str, Any]]) -> None:
    _bucket_matrix(
        rows,
        x_feat="match_iou",
        x_edges=(0.1, 0.3, 0.5, 0.7, 0.9),
        y_feat="neighbor_iou",
        y_edges=(0.1, 0.3, 0.5, 0.7),
        title="Bucket map: match_iou x neighbor_iou",
    )
    _bucket_matrix(
        rows,
        x_feat="match_iou",
        x_edges=(0.1, 0.3, 0.5, 0.7, 0.9),
        y_feat="gap",
        y_edges=(3, 10, 30, 60),
        title="Bucket map: match_iou x gap",
    )
    _bucket_matrix(
        rows,
        x_feat="margin",
        x_edges=(0.01, 0.03, 0.05, 0.08, 0.12),
        y_feat="candidate_count",
        y_edges=(2, 3, 5, 10),
        title="Bucket map: margin x candidate_count",
    )


def _bucket_stats(
    rows: list[dict[str, Any]],
    feat: str,
    edges: tuple[float, ...],
) -> list[dict[str, Any]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _known(rows):
        bucket = _bucket(_parse_float(row.get(feat)), edges)
        stats[bucket][row["label"]] += 1
    out = []
    for bucket in _bucket_names(edges):
        ok = stats[bucket]["correct"]
        bad = stats[bucket]["wrong"]
        total = ok + bad
        out.append(
            {
                "bucket": bucket,
                "correct": ok,
                "wrong": bad,
                "total": total,
                "same_rate": ok / total if total else 0.0,
            }
        )
    return out


def _zone(row: dict[str, Any], baseline_rate: float, min_bucket_n: int) -> str:
    if row["total"] < min_bucket_n:
        return "thin-sample"
    same_rate = row["same_rate"]
    if same_rate >= max(0.5, baseline_rate * 3.0) and row["correct"] >= 3:
        return "accept-candidate"
    if same_rate >= max(0.3, baseline_rate * 2.0) and row["correct"] >= 3:
        return "support"
    if same_rate <= baseline_rate * 0.6 and row["wrong"] >= min_bucket_n:
        return "danger"
    return "gray"


def _wilson_ci(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = success / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    margin = (
        z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _format_ci(success: int, total: int) -> str:
    lo, hi = _wilson_ci(success, total)
    return f"{lo:.2f}-{hi:.2f}"


def _ci_dict(success: int, total: int) -> dict[str, float]:
    lo, hi = _wilson_ci(success, total)
    return {"low": lo, "high": hi}


def _format_rate(row: dict[str, Any]) -> str:
    return f"{row['correct']}/{row['total']} ({row['same_rate']:.2f})"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _pollution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if _as_bool(r.get("known_full")) and _as_bool(r.get("known_local"))
    ]


def _pollution_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    same_gt = sum(r["label"] == "correct" for r in rows)
    candidate_mismatch = sum(
        not _as_bool(r.get("candidate_full_tail_same")) for r in rows
    )
    newborn_mismatch = sum(not _as_bool(r.get("newborn_full_head_same")) for r in rows)
    polluted = sum(
        (not _as_bool(r.get("candidate_full_tail_same")))
        or (not _as_bool(r.get("newborn_full_head_same")))
        for r in rows
    )
    return {
        "total": float(total),
        "same_gt": float(same_gt),
        "candidate_mismatch": float(candidate_mismatch),
        "newborn_mismatch": float(newborn_mismatch),
        "polluted": float(polluted),
        "same_rate": same_gt / total if total else 0.0,
        "pollution_rate": polluted / total if total else 0.0,
    }


def _pollution_bucket_markdown(
    rows: list[dict[str, Any]],
    *,
    feat: str,
    edges: tuple[float, ...],
    title: str,
) -> list[str]:
    eligible = _pollution_rows(rows)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        buckets[_bucket(_parse_float(row.get(feat)), edges)].append(row)
    lines = [
        f"### {title}",
        "",
        "| range | rows | same_gt | same_gt_ci | endpoint_polluted | pollution_ci | candidate_tail_mismatch | newborn_head_mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in _bucket_names(edges):
        stat = _pollution_stats(buckets[bucket])
        total = int(stat["total"])
        if total == 0:
            lines.append(f"| `{bucket}` | 0 | - | - | - | - | - | - |")
            continue
        same_gt = int(stat["same_gt"])
        polluted = int(stat["polluted"])
        lines.append(
            f"| `{bucket}` | {total} | "
            f"{same_gt}/{total} ({stat['same_rate']:.2f}) | "
            f"{_format_ci(same_gt, total)} | "
            f"{polluted}/{total} ({stat['pollution_rate']:.2f}) | "
            f"{_format_ci(polluted, total)} | "
            f"{int(stat['candidate_mismatch'])}/{total} | "
            f"{int(stat['newborn_mismatch'])}/{total} |"
        )
    lines.append("")
    return lines


def _pollution_bucket_stats(
    rows: list[dict[str, Any]],
    *,
    feat: str,
    edges: tuple[float, ...],
) -> list[dict[str, Any]]:
    eligible = _pollution_rows(rows)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        buckets[_bucket(_parse_float(row.get(feat)), edges)].append(row)
    out: list[dict[str, Any]] = []
    for bucket in _bucket_names(edges):
        stat = _pollution_stats(buckets[bucket])
        total = int(stat["total"])
        same_gt = int(stat["same_gt"])
        polluted = int(stat["polluted"])
        out.append(
            {
                "bucket": bucket,
                "total": total,
                "same_gt": same_gt,
                "same_rate": stat["same_rate"],
                "same_gt_ci": _ci_dict(same_gt, total),
                "endpoint_polluted": polluted,
                "pollution_rate": stat["pollution_rate"],
                "pollution_ci": _ci_dict(polluted, total),
                "candidate_tail_mismatch": int(stat["candidate_mismatch"]),
                "newborn_head_mismatch": int(stat["newborn_mismatch"]),
            }
        )
    return out


def _pollution_matrix_markdown(
    rows: list[dict[str, Any]],
    *,
    x_feat: str,
    x_edges: tuple[float, ...],
    y_feat: str,
    y_edges: tuple[float, ...],
) -> list[str]:
    eligible = _pollution_rows(rows)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    x_names = _bucket_names(x_edges)
    y_names = _bucket_names(y_edges)
    for row in eligible:
        xb = _bucket(_parse_float(row.get(x_feat)), x_edges)
        yb = _bucket(_parse_float(row.get(y_feat)), y_edges)
        buckets[(xb, yb)].append(row)
    lines = [
        f"### Pollution Matrix: {x_feat} x {y_feat}",
        "",
        "Cells are `same_gt / rows ; polluted / rows`. Thin cells are descriptive only.",
        "",
        "| " + y_feat + " | " + " | ".join(x_names) + " |",
        "|---|" + "|".join("---" for _ in x_names) + "|",
    ]
    for yb in y_names:
        cells = []
        for xb in x_names:
            stat = _pollution_stats(buckets[(xb, yb)])
            total = int(stat["total"])
            if total == 0:
                cells.append("-")
            else:
                cells.append(
                    f"{int(stat['same_gt'])}/{total}; {int(stat['polluted'])}/{total}"
                )
        lines.append("| " + yb + " | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _pollution_matrix_stats(
    rows: list[dict[str, Any]],
    *,
    x_feat: str,
    x_edges: tuple[float, ...],
    y_feat: str,
    y_edges: tuple[float, ...],
) -> dict[str, Any]:
    eligible = _pollution_rows(rows)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    x_names = _bucket_names(x_edges)
    y_names = _bucket_names(y_edges)
    for row in eligible:
        xb = _bucket(_parse_float(row.get(x_feat)), x_edges)
        yb = _bucket(_parse_float(row.get(y_feat)), y_edges)
        buckets[(xb, yb)].append(row)
    cells: list[dict[str, Any]] = []
    for yb in y_names:
        for xb in x_names:
            stat = _pollution_stats(buckets[(xb, yb)])
            total = int(stat["total"])
            cells.append(
                {
                    x_feat: xb,
                    y_feat: yb,
                    "total": total,
                    "same_gt": int(stat["same_gt"]),
                    "same_rate": stat["same_rate"],
                    "endpoint_polluted": int(stat["polluted"]),
                    "pollution_rate": stat["pollution_rate"],
                }
            )
    return {
        "x_feature": x_feat,
        "y_feature": y_feat,
        "x_buckets": x_names,
        "y_buckets": y_names,
        "cells": cells,
    }


def _pollution_markdown(rows: list[dict[str, Any]]) -> list[str]:
    eligible = _pollution_rows(rows)
    stat = _pollution_stats(eligible)
    lines = [
        "## Endpoint Pollution Evidence",
        "",
        "Pollution here means the full-track majority GT disagrees with the local endpoint GT on either side of the candidate edge.",
        "This is a diagnostic for cases where geometry/appearance may be reading an already-contaminated tracklet.",
        "",
        f"- eligible rows with full+local labels: `{int(stat['total'])}`",
        f"- endpoint polluted rows: `{int(stat['polluted'])}/{int(stat['total'])} = {stat['pollution_rate']:.3f}`",
        "",
    ]
    lines.extend(
        _pollution_bucket_markdown(
            rows,
            feat="match_iou",
            edges=REGISTRY_SPECS["match_iou"]["edges"],
            title="match_iou pollution buckets",
        )
    )
    lines.extend(
        _pollution_bucket_markdown(
            rows,
            feat="neighbor_iou",
            edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
            title="neighbor_iou pollution buckets",
        )
    )
    lines.extend(
        _pollution_bucket_markdown(
            rows,
            feat="head_tail_neighbor_iou",
            edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
            title="head_tail_neighbor_iou pollution buckets",
        )
    )
    lines.extend(
        _pollution_matrix_markdown(
            rows,
            x_feat="match_iou",
            x_edges=REGISTRY_SPECS["match_iou"]["edges"],
            y_feat="neighbor_iou",
            y_edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
        )
    )
    return lines


def _rate_for_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    correct = sum(r["label"] == "correct" for r in rows)
    wrong = sum(r["label"] == "wrong" for r in rows)
    total = correct + wrong
    return {
        "correct": float(correct),
        "wrong": float(wrong),
        "total": float(total),
        "same_rate": correct / total if total else 0.0,
    }


def _rule_expr(gates: tuple[tuple[str, float, str], ...]) -> str:
    parts = []
    for feat, thr, direction in gates:
        op = ">=" if direction == "high" else "<="
        parts.append(f"{feat} {op} {thr:g}")
    return " && ".join(parts)


def _rule_summary(
    rows: list[dict[str, Any]],
    gates: tuple[tuple[str, float, str], ...],
    *,
    min_seq_n: int,
) -> dict[str, Any]:
    known = _known(rows)
    seqs = sorted({str(r["seq"]) for r in known})
    selected = [r for r in known if _passes_gates(r, gates)]
    overall = _rate_for_rows(selected)
    per_seq = {
        seq: _rate_for_rows([r for r in selected if str(r["seq"]) == seq])
        for seq in seqs
    }
    seqs_with_n = sum(1 for stat in per_seq.values() if stat["total"] >= min_seq_n)
    loo_rates = []
    for seq in seqs:
        rest = [r for r in selected if str(r["seq"]) != seq]
        stat = _rate_for_rows(rest)
        if stat["total"] > 0:
            loo_rates.append(stat["same_rate"])
    return {
        "selected": selected,
        "overall": overall,
        "per_seq": per_seq,
        "seqs": seqs,
        "seqs_with_n": seqs_with_n,
        "loo_min": min(loo_rates) if loo_rates else None,
        "loo_max": max(loo_rates) if loo_rates else None,
    }


def _candidate_rule_map(
    rows: list[dict[str, Any]], min_seq_n: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in STABILITY_RULES:
        summary = _rule_summary(rows, spec["gates"], min_seq_n=min_seq_n)
        overall = summary["overall"]
        total = int(overall["total"])
        correct = int(overall["correct"])
        wrong = int(overall["wrong"])
        out.append(
            {
                "name": spec["name"],
                "intent": spec["intent"],
                "decision": spec["decision"],
                "use": spec["use"],
                "expression": _rule_expr(spec["gates"]),
                "selected": total,
                "correct": correct,
                "wrong": wrong,
                "same_rate": overall["same_rate"],
                "same_gt_ci": _ci_dict(correct, total),
                "seqs_with_min_n": summary["seqs_with_n"],
                "seq_count": len(summary["seqs"]),
                "loo_same_rate_min": summary["loo_min"],
                "loo_same_rate_max": summary["loo_max"],
                "per_seq": {
                    seq: {
                        "correct": int(stat["correct"]),
                        "wrong": int(stat["wrong"]),
                        "total": int(stat["total"]),
                        "same_rate": stat["same_rate"],
                    }
                    for seq, stat in summary["per_seq"].items()
                },
            }
        )
    return out


def _candidate_rule_map_markdown(
    rows: list[dict[str, Any]], min_seq_n: int
) -> list[str]:
    lines = [
        "## Candidate Rule Map",
        "",
        "Candidate rules are evidence-backed hypotheses for the next experiment, not tracker defaults.",
        "",
        "| rule | decision | selected | same_gt | same_gt_ci | seqs_with_n | LOO same_gt | expression | use |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for spec in STABILITY_RULES:
        summary = _rule_summary(rows, spec["gates"], min_seq_n=min_seq_n)
        overall = summary["overall"]
        total = int(overall["total"])
        correct = int(overall["correct"])
        same_text = f"{correct}/{total} ({overall['same_rate']:.2f})"
        loo_text = "-"
        if summary["loo_min"] is not None and summary["loo_max"] is not None:
            loo_text = f"{summary['loo_min']:.2f}-{summary['loo_max']:.2f}"
        lines.append(
            f"| `{spec['name']}` | {spec['decision']} | {total} | "
            f"{same_text} | {_format_ci(correct, total)} | "
            f"{summary['seqs_with_n']}/{len(summary['seqs'])} | {loo_text} | "
            f"`{_rule_expr(spec['gates'])}` | {spec['use']} |"
        )
    lines.append("")
    return lines


def _rule_policy_simulation_markdown(rows: list[dict[str, Any]]) -> list[str]:
    known = _known(rows)
    accepted_known = _known([r for r in rows if r["accepted"]])
    accepted_total_correct = sum(r["label"] == "correct" for r in accepted_known)
    accepted_total_wrong = sum(r["label"] == "wrong" for r in accepted_known)
    all_total_correct = sum(r["label"] == "correct" for r in known)
    lines = [
        "## Candidate Rule Policy Simulation",
        "",
        "Simulation is label-side analysis only. It does not mutate tracker output.",
        "",
        "| rule | action | all_selected | all_precision | all_correct_recall | accepted_action | kept_correct | kept_wrong | wrong_cut | correct_cut |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in STABILITY_RULES:
        gates = spec["gates"]
        is_veto = spec["decision"] == "reject/veto"
        selected_all = [r for r in known if _passes_gates(r, gates)]
        selected_acc = [r for r in accepted_known if _passes_gates(r, gates)]
        all_correct = sum(r["label"] == "correct" for r in selected_all)
        all_wrong = sum(r["label"] == "wrong" for r in selected_all)
        selected_correct = sum(r["label"] == "correct" for r in selected_acc)
        selected_wrong = sum(r["label"] == "wrong" for r in selected_acc)
        all_selected = all_correct + all_wrong
        all_precision = all_correct / max(1, all_selected)
        all_recall = all_correct / max(1, all_total_correct)
        if is_veto:
            action = "cut"
            accepted_action_n = selected_correct + selected_wrong
            correct_cut = selected_correct
            wrong_cut = selected_wrong
            keep_correct = accepted_total_correct - correct_cut
            keep_wrong = accepted_total_wrong - wrong_cut
        else:
            action = "keep"
            accepted_action_n = selected_correct + selected_wrong
            keep_correct = selected_correct
            keep_wrong = selected_wrong
            wrong_cut = accepted_total_wrong - keep_wrong
            correct_cut = accepted_total_correct - keep_correct
        lines.append(
            f"| `{spec['name']}` | {action} | {all_selected} | "
            f"{all_precision:.2f} | {all_recall:.2f} | "
            f"{accepted_action_n}/{len(accepted_known)} | "
            f"{keep_correct}/{accepted_total_correct} | "
            f"{keep_wrong}/{accepted_total_wrong} | "
            f"{wrong_cut}/{accepted_total_wrong} | "
            f"{correct_cut}/{accepted_total_correct} |"
        )
    lines.append("")
    return lines


def _rule_policy_simulation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = _known(rows)
    accepted_known = _known([r for r in rows if r["accepted"]])
    accepted_total_correct = sum(r["label"] == "correct" for r in accepted_known)
    accepted_total_wrong = sum(r["label"] == "wrong" for r in accepted_known)
    all_total_correct = sum(r["label"] == "correct" for r in known)
    out: list[dict[str, Any]] = []
    for spec in STABILITY_RULES:
        gates = spec["gates"]
        is_veto = spec["decision"] == "reject/veto"
        selected_all = [r for r in known if _passes_gates(r, gates)]
        selected_acc = [r for r in accepted_known if _passes_gates(r, gates)]
        all_correct = sum(r["label"] == "correct" for r in selected_all)
        all_wrong = sum(r["label"] == "wrong" for r in selected_all)
        selected_correct = sum(r["label"] == "correct" for r in selected_acc)
        selected_wrong = sum(r["label"] == "wrong" for r in selected_acc)
        all_selected = all_correct + all_wrong
        if is_veto:
            action = "cut"
            correct_cut = selected_correct
            wrong_cut = selected_wrong
            keep_correct = accepted_total_correct - correct_cut
            keep_wrong = accepted_total_wrong - wrong_cut
        else:
            action = "keep"
            keep_correct = selected_correct
            keep_wrong = selected_wrong
            wrong_cut = accepted_total_wrong - keep_wrong
            correct_cut = accepted_total_correct - keep_correct
        out.append(
            {
                "name": spec["name"],
                "action": action,
                "expression": _rule_expr(gates),
                "all_selected": all_selected,
                "all_correct": all_correct,
                "all_wrong": all_wrong,
                "all_precision": all_correct / max(1, all_selected),
                "all_correct_recall": all_correct / max(1, all_total_correct),
                "accepted_action": selected_correct + selected_wrong,
                "accepted_known": len(accepted_known),
                "kept_correct": keep_correct,
                "accepted_correct": accepted_total_correct,
                "kept_wrong": keep_wrong,
                "accepted_wrong": accepted_total_wrong,
                "wrong_cut": wrong_cut,
                "correct_cut": correct_cut,
            }
        )
    return out


def _stability_markdown(rows: list[dict[str, Any]], min_seq_n: int) -> list[str]:
    known = _known(rows)
    seqs = sorted({str(r["seq"]) for r in known})
    lines = [
        "## Stability Checks",
        "",
        "These checks are descriptive guards against sequence-specific artifacts.",
        "LOO means leave-one-sequence-out; thin per-sequence cells are still shown but should not be treated as proof.",
        "",
        "| rule | intent | selected | same_gt | seqs_with_n | LOO same_gt range | expression |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    details: list[str] = []
    for spec in STABILITY_RULES:
        gates = spec["gates"]
        summary = _rule_summary(rows, gates, min_seq_n=min_seq_n)
        overall = summary["overall"]
        per_seq = summary["per_seq"]
        seqs_with_n = summary["seqs_with_n"]
        loo_text = "-"
        if summary["loo_min"] is not None and summary["loo_max"] is not None:
            loo_text = f"{summary['loo_min']:.2f}-{summary['loo_max']:.2f}"
        same_text = (
            f"{int(overall['correct'])}/{int(overall['total'])} "
            f"({overall['same_rate']:.2f})"
        )
        lines.append(
            f"| `{spec['name']}` | {spec['intent']} | {int(overall['total'])} | "
            f"{same_text} | {seqs_with_n}/{len(seqs)} | {loo_text} | "
            f"`{_rule_expr(gates)}` |"
        )
        details.extend(
            [
                f"### {spec['name']}",
                "",
                f"- expression: `{_rule_expr(gates)}`",
                f"- intent: {spec['intent']}",
                "",
                "| sequence | same_gt | wrong |",
                "|---|---:|---:|",
            ]
        )
        for seq in seqs:
            stat = per_seq[seq]
            total = int(stat["total"])
            if total == 0:
                text = "-"
            else:
                text = f"{int(stat['correct'])}/{total} ({stat['same_rate']:.2f})"
            details.append(f"| {seq} | {text} | {int(stat['wrong'])} |")
        details.append("")
    lines.extend(["", "### Per-Sequence Rule Details", ""])
    lines.extend(details)
    return lines


def _matrix_markdown(
    rows: list[dict[str, Any]],
    *,
    x_feat: str,
    x_edges: tuple[float, ...],
    y_feat: str,
    y_edges: tuple[float, ...],
) -> list[str]:
    known = _known(rows)
    buckets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    x_names = _bucket_names(x_edges)
    y_names = _bucket_names(y_edges)
    for row in known:
        xb = _bucket(_parse_float(row.get(x_feat)), x_edges)
        yb = _bucket(_parse_float(row.get(y_feat)), y_edges)
        buckets[(xb, yb)][row["label"]] += 1
    lines = [
        f"### {x_feat} x {y_feat}",
        "",
        "| " + y_feat + " | " + " | ".join(x_names) + " |",
        "|---|" + "|".join("---" for _ in x_names) + "|",
    ]
    for yb in y_names:
        cells = []
        for xb in x_names:
            c = buckets[(xb, yb)]
            total = c["correct"] + c["wrong"]
            if total == 0:
                cells.append("-")
            else:
                cells.append(f"{c['correct']}/{total} ({c['correct'] / total:.2f})")
        lines.append("| " + yb + " | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _matrix_stats(
    rows: list[dict[str, Any]],
    *,
    x_feat: str,
    x_edges: tuple[float, ...],
    y_feat: str,
    y_edges: tuple[float, ...],
) -> dict[str, Any]:
    known = _known(rows)
    buckets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    x_names = _bucket_names(x_edges)
    y_names = _bucket_names(y_edges)
    for row in known:
        xb = _bucket(_parse_float(row.get(x_feat)), x_edges)
        yb = _bucket(_parse_float(row.get(y_feat)), y_edges)
        buckets[(xb, yb)][row["label"]] += 1
    cells: list[dict[str, Any]] = []
    for yb in y_names:
        for xb in x_names:
            c = buckets[(xb, yb)]
            correct = c["correct"]
            wrong = c["wrong"]
            total = correct + wrong
            cells.append(
                {
                    x_feat: xb,
                    y_feat: yb,
                    "correct": correct,
                    "wrong": wrong,
                    "total": total,
                    "same_rate": correct / total if total else 0.0,
                }
            )
    return {
        "x_feature": x_feat,
        "y_feature": y_feat,
        "x_buckets": x_names,
        "y_buckets": y_names,
        "cells": cells,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary_json(
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    handover_log: str,
    baseline_dir: str,
    pred_dir: str,
    label_mode: str,
    edge_window: int,
    min_bucket_n: int,
    pair_min_recall: float,
) -> dict[str, Any]:
    known = _known(rows)
    accepted = [r for r in rows if r["accepted"]]
    accepted_known = _known(accepted)
    correct = sum(r["label"] == "correct" for r in known)
    wrong = sum(r["label"] == "wrong" for r in known)
    accepted_correct = sum(r["label"] == "correct" for r in accepted_known)
    accepted_wrong = sum(r["label"] == "wrong" for r in accepted_known)
    baseline_rate = correct / max(1, len(known))
    features: dict[str, Any] = {}
    for feat, spec in REGISTRY_SPECS.items():
        direction, auc = orientations.get(feat, ("unknown", float("nan")))
        buckets = []
        for row in _bucket_stats(rows, feat, spec["edges"]):
            total = int(row["total"])
            ok = int(row["correct"])
            buckets.append(
                {
                    "bucket": row["bucket"],
                    "correct": ok,
                    "wrong": int(row["wrong"]),
                    "total": total,
                    "same_rate": row["same_rate"],
                    "same_gt_ci": _ci_dict(ok, total),
                    "zone": _zone(row, baseline_rate, min_bucket_n),
                }
            )
        features[feat] = {
            "meaning": spec["meaning"],
            "observed_direction": direction,
            "auc": auc,
            "combine_with": spec["use_with"],
            "failure_mode": spec["failure"],
            "buckets": buckets,
        }
    eligible = _pollution_rows(rows)
    pollution_total = _pollution_stats(eligible)
    summary = {
        "schema": "cheb_gr_offline_handover_summary/v1",
        "provenance": {
            "handover_log": handover_log,
            "baseline_dir": baseline_dir,
            "pred_dir": pred_dir,
            "label_mode": label_mode,
            "edge_window": edge_window,
            "registry_min_bucket_n": min_bucket_n,
        },
        "counts": {
            "rows": len(rows),
            "accepted": len(accepted),
            "known": len(known),
            "unknown": len(rows) - len(known),
            "correct": correct,
            "wrong": wrong,
            "same_rate": baseline_rate,
            "accepted_known": len(accepted_known),
            "accepted_correct": accepted_correct,
            "accepted_wrong": accepted_wrong,
            "accepted_precision": accepted_correct
            / max(1, accepted_correct + accepted_wrong),
        },
        "features": features,
        "candidate_rules": _candidate_rule_map(
            rows, min_seq_n=max(3, min_bucket_n // 2)
        ),
        "discovered_gates": {
            "all_known_single_feature": _rank_single_feature_gates(
                rows,
                orientations,
                min_selected=max(3, len(known) // 20),
            ),
            "accepted_known_single_feature": _rank_single_feature_gates(
                accepted,
                orientations,
                min_selected=max(3, len(accepted_known) // 10),
            ),
            "accepted_known_two_feature": _rank_pair_gates(
                accepted,
                orientations,
                min_correct_recall=pair_min_recall,
            ),
        },
        "policy_simulation": _rule_policy_simulation(rows),
        "combination_matrices": [
            _matrix_stats(
                rows,
                x_feat="match_iou",
                x_edges=REGISTRY_SPECS["match_iou"]["edges"],
                y_feat="neighbor_iou",
                y_edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
            ),
            _matrix_stats(
                rows,
                x_feat="margin",
                x_edges=REGISTRY_SPECS["margin"]["edges"],
                y_feat="candidate_count",
                y_edges=REGISTRY_SPECS["candidate_count"]["edges"],
            ),
            _matrix_stats(
                rows,
                x_feat="center_dist_norm",
                x_edges=REGISTRY_SPECS["center_dist_norm"]["edges"],
                y_feat="gap",
                y_edges=REGISTRY_SPECS["gap"]["edges"],
            ),
        ],
        "pollution": {
            "eligible": int(pollution_total["total"]),
            "endpoint_polluted": int(pollution_total["polluted"]),
            "pollution_rate": pollution_total["pollution_rate"],
            "feature_buckets": {
                "match_iou": _pollution_bucket_stats(
                    rows,
                    feat="match_iou",
                    edges=REGISTRY_SPECS["match_iou"]["edges"],
                ),
                "neighbor_iou": _pollution_bucket_stats(
                    rows,
                    feat="neighbor_iou",
                    edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
                ),
                "head_tail_neighbor_iou": _pollution_bucket_stats(
                    rows,
                    feat="head_tail_neighbor_iou",
                    edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
                ),
            },
            "matrices": [
                _pollution_matrix_stats(
                    rows,
                    x_feat="match_iou",
                    x_edges=REGISTRY_SPECS["match_iou"]["edges"],
                    y_feat="neighbor_iou",
                    y_edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
                )
            ],
        },
    }
    return _json_safe(summary)


def _write_summary_json(
    path: Path,
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    handover_log: str,
    baseline_dir: str,
    pred_dir: str,
    label_mode: str,
    edge_window: int,
    min_bucket_n: int,
    pair_min_recall: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary_json(
        rows,
        orientations,
        handover_log=handover_log,
        baseline_dir=baseline_dir,
        pred_dir=pred_dir,
        label_mode=label_mode,
        edge_window=edge_window,
        min_bucket_n=min_bucket_n,
        pair_min_recall=pair_min_recall,
    )
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _write_registry_md(
    path: Path,
    rows: list[dict[str, Any]],
    orientations: dict[str, tuple[str, float]],
    *,
    handover_log: str,
    baseline_dir: str,
    pred_dir: str,
    label_mode: str,
    edge_window: int,
    min_bucket_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    known = _known(rows)
    correct = sum(r["label"] == "correct" for r in known)
    baseline_rate = correct / max(1, len(known))
    accepted = [r for r in rows if r["accepted"]]
    accepted_known = _known(accepted)
    accepted_correct = sum(r["label"] == "correct" for r in accepted_known)
    lines = [
        "# Cheb-GR Offline Handover Parameter Applicability Registry",
        "",
        "This is an evidence registry, not a final threshold configuration.",
        "",
        "## Provenance",
        "",
        f"- handover_log: `{handover_log}`",
        f"- baseline_dir: `{baseline_dir}`",
        f"- pred_dir: `{pred_dir}`" if pred_dir else "- pred_dir: ``",
        f"- label_mode: `{label_mode}`",
        f"- edge_window: `{edge_window}`",
        f"- known candidate edges: `{len(known)}`",
        f"- global same_gt rate: `{correct}/{len(known)} = {baseline_rate:.3f}`",
        f"- accepted known precision: `{accepted_correct}/{len(accepted_known)} = {accepted_correct / max(1, len(accepted_known)):.3f}`",
        "",
        "Zone labels:",
        "",
        "- `accept-candidate`: high same_gt bucket with enough support; candidate for conservative accept zone.",
        "- `support`: positive signal, but should be combined with another feature.",
        "- `gray`: inconclusive.",
        "- `danger`: high wrong-link density; candidate reject/veto context.",
        "- `thin-sample`: too few rows for a claim.",
        "",
    ]
    lines.extend(
        _candidate_rule_map_markdown(rows, min_seq_n=max(3, min_bucket_n // 2))
    )
    lines.extend(_rule_policy_simulation_markdown(rows))
    lines.extend(
        [
            "## Single-Feature Ranges",
            "",
        ]
    )
    for feat, spec in REGISTRY_SPECS.items():
        direction, auc = orientations.get(feat, ("unknown", float("nan")))
        lines.extend(
            [
                f"### {feat}",
                "",
                f"- meaning: {spec['meaning']}",
                f"- observed direction: `{direction}`; AUC: `{auc:.3f}`",
                f"- combine with: {spec['use_with']}",
                f"- failure mode: {spec['failure']}",
                "",
                "| range | same_gt | same_gt_ci | wrong | zone |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in _bucket_stats(rows, feat, spec["edges"]):
            total = int(row["total"])
            correct = int(row["correct"])
            lines.append(
                f"| `{row['bucket']}` | {_format_rate(row)} | "
                f"{_format_ci(correct, total)} | {row['wrong']} | "
                f"{_zone(row, baseline_rate, min_bucket_n)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Combination Evidence",
            "",
            "Cells are `correct/total (same_gt_rate)`. Use these to separate a signal from its context.",
            "",
        ]
    )
    lines.extend(
        _matrix_markdown(
            rows,
            x_feat="match_iou",
            x_edges=REGISTRY_SPECS["match_iou"]["edges"],
            y_feat="neighbor_iou",
            y_edges=REGISTRY_SPECS["neighbor_iou"]["edges"],
        )
    )
    lines.extend(
        _matrix_markdown(
            rows,
            x_feat="margin",
            x_edges=REGISTRY_SPECS["margin"]["edges"],
            y_feat="candidate_count",
            y_edges=REGISTRY_SPECS["candidate_count"]["edges"],
        )
    )
    lines.extend(
        _matrix_markdown(
            rows,
            x_feat="center_dist_norm",
            x_edges=REGISTRY_SPECS["center_dist_norm"]["edges"],
            y_feat="gap",
            y_edges=REGISTRY_SPECS["gap"]["edges"],
        )
    )
    lines.extend(_pollution_markdown(rows))
    lines.extend(_stability_markdown(rows, min_seq_n=max(3, min_bucket_n // 2)))
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handover-log", required=True, help="Cheb-GR handover CSV.")
    ap.add_argument(
        "--baseline-dir",
        required=True,
        help="Pre-handover MOT output dir, used to label original track ids.",
    )
    ap.add_argument(
        "--pred-dir",
        default="",
        help="Optional post-handover MOT output dir; currently printed for provenance.",
    )
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument("--sequences", default="", help="Comma list; default = SDP seqs.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out-csv", default="", help="Write annotated decision CSV.")
    ap.add_argument(
        "--registry-md",
        default="",
        help="Write parameter applicability registry as Markdown.",
    )
    ap.add_argument(
        "--summary-json",
        default="",
        help="Write machine-readable parameter applicability summary as JSON.",
    )
    ap.add_argument(
        "--registry-min-bucket-n",
        type=int,
        default=8,
        help="Minimum bucket count before assigning support/danger zones.",
    )
    ap.add_argument("--pair-min-recall", type=float, default=0.8)
    ap.add_argument(
        "--edge-window",
        type=int,
        default=5,
        help="Frames used for candidate-tail/newborn-head local edge labels.",
    )
    ap.add_argument(
        "--label-mode",
        choices=("local", "full"),
        default="local",
        help="Label gates by local tail/head edge or full-track majority.",
    )
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    all_rows = _read_handover_log(Path(args.handover_log))
    requested = set(_sequence_names(gt_root, args.sequences, all_rows))
    rows = [r for r in all_rows if r["seq"] in requested]
    annotated = _annotate_rows(
        rows,
        baseline_dir=Path(args.baseline_dir),
        gt_root=gt_root,
        iou=args.iou,
        edge_window=args.edge_window,
        label_mode=args.label_mode,
    )

    total = len(annotated)
    accepted = [r for r in annotated if r["accepted"]]
    known = _known(annotated)
    accepted_known = _known(accepted)
    correct = sum(r["label"] == "correct" for r in known)
    wrong = sum(r["label"] == "wrong" for r in known)
    accepted_correct = sum(r["label"] == "correct" for r in accepted_known)
    accepted_wrong = sum(r["label"] == "wrong" for r in accepted_known)

    print("Cheb-GR offline handover report")
    print(f"  handover_log: {args.handover_log}")
    print(f"  baseline_dir: {args.baseline_dir}")
    if args.pred_dir:
        print(f"  pred_dir:     {args.pred_dir}")
    print(
        f"  rows={total} accepted={len(accepted)} known={len(known)} unknown={total - len(known)}"
    )
    print(f"  label_mode={args.label_mode} edge_window={args.edge_window}")
    print(f"  known labels: correct={correct} wrong={wrong}")
    print(
        "  accepted known: "
        f"correct={accepted_correct} wrong={accepted_wrong} "
        f"precision={accepted_correct / max(1, accepted_correct + accepted_wrong):.3f}"
    )
    accepted_local_correct = sum(
        r.get("same_gt_local") and r.get("known_local") for r in accepted
    )
    accepted_full_correct = sum(
        r.get("same_gt_full") and r.get("known_full") for r in accepted
    )
    accepted_local_known = sum(r.get("known_local") for r in accepted)
    accepted_full_known = sum(r.get("known_full") for r in accepted)
    candidate_polluted = sum(
        r.get("known_full")
        and r.get("known_local")
        and not r.get("candidate_full_tail_same")
        for r in accepted
    )
    newborn_polluted = sum(
        r.get("known_full")
        and r.get("known_local")
        and not r.get("newborn_full_head_same")
        for r in accepted
    )
    print(
        "  accepted label audit: "
        f"local_ok={accepted_local_correct}/{accepted_local_known} "
        f"full_ok={accepted_full_correct}/{accepted_full_known} "
        f"candidate_full_tail_mismatch={candidate_polluted} "
        f"newborn_full_head_mismatch={newborn_polluted}"
    )

    by_seq = Counter((r["seq"], r["label"], bool(r["accepted"])) for r in annotated)
    print("\nBy sequence")
    print(
        f"  {'seq':<12} {'accept_ok':>9} {'accept_bad':>10} {'reject_ok':>9} {'reject_bad':>10} {'unknown':>8}"
    )
    for seq in sorted({str(r["seq"]) for r in annotated}):
        unknown = sum(
            n
            for (s, label, _acc), n in by_seq.items()
            if s == seq and label == "unknown"
        )
        print(
            f"  {seq:<12} "
            f"{by_seq[(seq, 'correct', True)]:9d} "
            f"{by_seq[(seq, 'wrong', True)]:10d} "
            f"{by_seq[(seq, 'correct', False)]:9d} "
            f"{by_seq[(seq, 'wrong', False)]:10d} "
            f"{unknown:8d}"
        )

    orientations = _feature_summary(annotated)
    _print_bucket_maps(annotated)
    _print_gate_table(
        annotated,
        orientations,
        title="Single-feature gates over all known decisions",
        min_selected=max(3, len(known) // 20),
    )
    _print_gate_table(
        accepted,
        orientations,
        title="Single-feature pruning gates over accepted known decisions",
        min_selected=max(3, len(accepted_known) // 10),
    )
    _print_pair_gates(
        accepted,
        orientations,
        min_correct_recall=args.pair_min_recall,
    )

    if args.out_csv:
        _write_csv(Path(args.out_csv), annotated)
        print(f"\nWrote annotated CSV: {args.out_csv}")
    if args.registry_md:
        _write_registry_md(
            Path(args.registry_md),
            annotated,
            orientations,
            handover_log=args.handover_log,
            baseline_dir=args.baseline_dir,
            pred_dir=args.pred_dir,
            label_mode=args.label_mode,
            edge_window=args.edge_window,
            min_bucket_n=args.registry_min_bucket_n,
        )
        print(f"Wrote registry: {args.registry_md}")
    if args.summary_json:
        _write_summary_json(
            Path(args.summary_json),
            annotated,
            orientations,
            handover_log=args.handover_log,
            baseline_dir=args.baseline_dir,
            pred_dir=args.pred_dir,
            label_mode=args.label_mode,
            edge_window=args.edge_window,
            min_bucket_n=args.registry_min_bucket_n,
            pair_min_recall=args.pair_min_recall,
        )
        print(f"Wrote summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
