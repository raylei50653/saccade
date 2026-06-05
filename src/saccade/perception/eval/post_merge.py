import io
import math
import numpy as np
import pandas as pd
from typing import Any
from .types import MotRecord, OutputTracklet
from .utils import (
    mot_box as _mot_box,
    box_iou_tuple as _box_iou_tuple,
    box_center as _box_center,
    shift_box as _shift_box,
)
from .lifecycle import UnionFind


def _tracklet_velocity(
    records: list[MotRecord], from_start: bool, samples: int
) -> tuple[float, float]:
    if len(records) < 2:
        return (0.0, 0.0)
    window = records[:samples] if from_start else records[-samples:]
    if len(window) < 2:
        return (0.0, 0.0)
    first = window[0]
    last = window[-1]
    dt = max(last.frame - first.frame, 1)
    fc = _box_center(_mot_box(first))
    lc = _box_center(_mot_box(last))
    return ((lc[0] - fc[0]) / dt, (lc[1] - fc[1]) / dt)


def _direction_penalty(a: tuple[float, float], b: tuple[float, float]) -> float:
    norm_a = math.hypot(a[0], a[1])
    norm_b = math.hypot(b[0], b[1])
    if norm_a < 1e-3 or norm_b < 1e-3:
        return 0.0
    cos = (a[0] * b[0] + a[1] * b[1]) / max(norm_a * norm_b, 1e-6)
    return max(0.0, -cos)


def _parse_mot_lines(lines: list[str]) -> list[MotRecord]:
    records = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(",")
        records.append(
            MotRecord(
                frame=int(float(parts[0])),
                track_id=int(float(parts[1])),
                x=float(parts[2]),
                y=float(parts[3]),
                w=float(parts[4]),
                h=float(parts[5]),
                score=float(parts[6]),
                tail=parts[7:],
            )
        )
    return records


def _format_mot_records(records: list[MotRecord]) -> list[str]:
    lines = []
    for record in sorted(records, key=lambda item: (item.frame, item.track_id)):
        tail = record.tail if record.tail else ["-1", "-1", "-1"]
        lines.append(
            f"{record.frame},{record.track_id},{record.x:.2f},{record.y:.2f},"
            f"{record.w:.2f},{record.h:.2f},{record.score:.4f},{','.join(tail)}"
        )
    return lines


def _build_output_tracklets(
    records: list[MotRecord], velocity_samples: int
) -> list[OutputTracklet]:
    by_id: dict[int, list[MotRecord]] = {}
    for record in records:
        by_id.setdefault(record.track_id, []).append(record)

    tracklets = []
    for track_id, items in by_id.items():
        items = sorted(items, key=lambda item: item.frame)
        tracklets.append(
            OutputTracklet(
                track_id=track_id,
                records=items,
                start=items[0].frame,
                end=items[-1].frame,
                start_box=_mot_box(items[0]),
                end_box=_mot_box(items[-1]),
                start_velocity=_tracklet_velocity(
                    items, from_start=True, samples=velocity_samples
                ),
                end_velocity=_tracklet_velocity(
                    items, from_start=False, samples=velocity_samples
                ),
                mean_score=sum(item.score for item in items) / len(items),
            )
        )
    return tracklets


def post_merge_output_tracklets(
    lines: list[str],
    *,
    enabled: bool,
    ttl: int,
    min_gap: int,
    velocity_samples: int,
    spatial_weight: float,
    motion_weight: float,
    time_weight: float,
    direction_weight: float,
    max_cost: float,
    appearance_bank: Any = None,
    appearance_gate: bool = False,
    appearance_threshold: float = 0.90,
    appearance_min_samples: int = 1,
    # A5: appearance as soft cost
    appearance_weight: float = 0.0,
    gap_uncertainty_weight: float = 0.0,
    consistency_weight: float = 0.0,
    missing_appearance_cost: float = 0.5,
) -> tuple[list[str], dict[str, int]]:
    stats = {
        "candidates": 0,
        "accepted": 0,
        "ids_before": 0,
        "ids_after": 0,
        "reject_appearance": 0,
        "reject_appearance_missing": 0,
        "reject_appearance_consistency": 0,
        "reject_cost": 0,
    }
    if not enabled or not lines:
        return lines, stats

    records = _parse_mot_lines(lines)
    tracklets = _build_output_tracklets(records, velocity_samples)
    stats["ids_before"] = len(tracklets)
    if len(tracklets) <= 1:
        stats["ids_after"] = len(tracklets)
        return lines, stats

    rows: list[OutputTracklet] = []
    cols: list[OutputTracklet] = []
    costs: list[tuple[int, int, float]] = []
    for row_idx, lost in enumerate(tracklets):
        row_used = False
        for col_idx, new in enumerate(tracklets):
            gap = new.start - lost.end
            if gap < min_gap or gap > ttl:
                continue
            forward_box = _shift_box(lost.end_box, lost.end_velocity, gap)
            backward_box = _shift_box(new.start_box, new.start_velocity, -gap)
            forward_iou = _box_iou_tuple(forward_box, new.start_box)
            backward_iou = _box_iou_tuple(backward_box, lost.end_box)
            motion_iou = max(forward_iou, backward_iou)

            lost_center = _box_center(forward_box)
            new_center = _box_center(new.start_box)
            dist = math.hypot(
                lost_center[0] - new_center[0], lost_center[1] - new_center[1]
            )
            scale = (
                max(
                    (lost.end_box[2] - lost.end_box[0])
                    * (lost.end_box[3] - lost.end_box[1]),
                    (new.start_box[2] - new.start_box[0])
                    * (new.start_box[3] - new.start_box[1]),
                    1.0,
                )
                ** 0.5
            )
            spatial_cost = dist / max(scale, 1.0)
            motion_cost = 1.0 - motion_iou
            time_cost = gap / max(ttl, 1)
            direction_cost = _direction_penalty(lost.end_velocity, new.start_velocity)
            cost = (
                spatial_weight * spatial_cost
                + motion_weight * motion_cost
                + time_weight * time_cost
                + direction_weight * direction_cost
            )

            # A5: appearance as hard gate
            if appearance_gate:
                if appearance_bank is None:
                    stats["reject_appearance_missing"] += 1
                    continue
                if (
                    appearance_bank.count(lost.track_id) < appearance_min_samples
                    or appearance_bank.count(new.track_id) < appearance_min_samples
                ):
                    stats["reject_appearance_missing"] += 1
                    continue
                if (
                    appearance_bank.consistency(lost.track_id)
                    < appearance_bank.min_consistency
                    or appearance_bank.consistency(new.track_id)
                    < appearance_bank.min_consistency
                ):
                    stats["reject_appearance_consistency"] += 1
                    continue
                sim = appearance_bank.similarity(lost.track_id, new.track_id)
                if sim is None:
                    stats["reject_appearance_missing"] += 1
                    continue
                if sim < appearance_threshold:
                    stats["reject_appearance"] += 1
                    continue

            # A5: appearance as soft cost
            if appearance_weight > 0.0:
                normalized_gap = gap / max(ttl, 1)
                eff_app_w = appearance_weight * (
                    1.0 + gap_uncertainty_weight * normalized_gap
                )
                if appearance_bank is not None:
                    sim = appearance_bank.similarity(lost.track_id, new.track_id)
                    if sim is not None:
                        app_cost = 1.0 - sim
                    else:
                        app_cost = missing_appearance_cost
                    consist_a = appearance_bank.consistency(lost.track_id)
                    consist_b = appearance_bank.consistency(new.track_id)
                    consist_cost = 1.0 - min(consist_a, consist_b)
                    cost += eff_app_w * app_cost + consistency_weight * consist_cost
                else:
                    cost += eff_app_w * missing_appearance_cost

            if cost > max_cost:
                stats["reject_cost"] += 1
                continue
            if not row_used:
                rows.append(lost)
                row_used = True
            if new not in cols:
                cols.append(new)
            costs.append((len(rows) - 1, cols.index(new), cost))
            stats["candidates"] += 1

    if not costs:
        stats["ids_after"] = len(tracklets)
        return lines, stats

    large = max_cost + 1000.0
    cost_matrix = np.full((len(rows), len(cols)), large, dtype=np.float32)
    for row_idx, col_idx, cost in costs:
        cost_matrix[row_idx, col_idx] = min(cost_matrix[row_idx, col_idx], cost)

    try:
        from scipy.optimize import linear_sum_assignment

        matched_rows, matched_cols = linear_sum_assignment(cost_matrix)
        matches = [
            (rows[int(row)].track_id, cols[int(col)].track_id)
            for row, col in zip(matched_rows, matched_cols)
            if float(cost_matrix[int(row), int(col)]) <= max_cost
        ]
    except Exception:
        matches = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row_idx, col_idx, cost in sorted(costs, key=lambda item: item[2]):
            if row_idx in used_rows or col_idx in used_cols:
                continue
            used_rows.add(row_idx)
            used_cols.add(col_idx)
            matches.append((rows[row_idx].track_id, cols[col_idx].track_id))

    uf = UnionFind([tracklet.track_id for tracklet in tracklets])
    for keep_id, merge_id in matches:
        uf.union(keep_id, merge_id)
        stats["accepted"] += 1

    for record in records:
        record.track_id = uf.find(record.track_id)
    stats["ids_after"] = len({record.track_id for record in records})
    return _format_mot_records(records), stats


def filter_low_quality_tracklets(
    lines: list[str],
    *,
    min_len: int = 1,
    min_score: float = 0.0,
) -> tuple[list[str], dict[str, int]]:
    stats: dict[str, int] = {"before": 0, "after": 0, "removed": 0}
    if (min_len <= 1 and min_score <= 0.0) or not lines:
        return lines, stats
    records = _parse_mot_lines(lines)
    by_id: dict[int, list[MotRecord]] = {}
    for r in records:
        by_id.setdefault(r.track_id, []).append(r)
    stats["before"] = len(by_id)
    keep_ids: set[int] = set()
    for track_id, recs in by_id.items():
        if len(recs) < min_len:
            continue
        if min_score > 0.0 and sum(r.score for r in recs) / len(recs) < min_score:
            continue
        keep_ids.add(track_id)
    stats["after"] = len(keep_ids)
    stats["removed"] = stats["before"] - stats["after"]
    filtered = [r for r in records if r.track_id in keep_ids]
    return _format_mot_records(filtered), stats


def apply_deferred_alias(
    lines: list[str],
    alias: dict[int, int],
) -> tuple[list[str], dict[str, int]]:
    """Apply finalized delayed-claim aliases to already-emitted MOT lines."""
    stats = {
        "aliases": 0,
        "aliases_skipped_overlap": 0,
        "lines_remapped": 0,
        "ids_before": 0,
        "ids_after": 0,
    }
    if not lines or not alias:
        return lines, stats

    records = _parse_mot_lines(lines)
    stats["ids_before"] = len({record.track_id for record in records})
    remap = {int(k): int(v) for k, v in alias.items() if int(k) != int(v)}
    if not remap:
        stats["ids_after"] = stats["ids_before"]
        return lines, stats

    frames_by_id: dict[int, set[int]] = {}
    for record in records:
        frames_by_id.setdefault(record.track_id, set()).add(record.frame)

    safe_remap: dict[int, int] = {}
    for raw_id, canonical_id in remap.items():
        raw_frames = frames_by_id.get(raw_id, set())
        canonical_frames = frames_by_id.get(canonical_id, set())
        if (
            raw_frames
            and canonical_frames
            and raw_frames.intersection(canonical_frames)
        ):
            stats["aliases_skipped_overlap"] += 1
            continue
        safe_remap[raw_id] = canonical_id

    stats["aliases"] = len(safe_remap)
    for record in records:
        new_id = safe_remap.get(record.track_id)
        if new_id is not None:
            record.track_id = new_id
            stats["lines_remapped"] += 1
    stats["ids_after"] = len({record.track_id for record in records})
    return _format_mot_records(records), stats


def interpolate_tracklets(
    lines: list[str],
    *,
    max_gap: int = 20,
    min_track_len: int = 5,
) -> tuple[list[str], dict[str, int]]:
    """Fill gaps ≤ max_gap in confirmed tracklets with linear interpolation.

    Only operates on tracks with ≥ min_track_len observations (shorter tracklets
    are likely noise and shouldn't be extrapolated).
    Uses pandas + numpy for fast vectorized parse/format.
    """
    stats: dict[str, int] = {
        "tracks_interpolated": 0,
        "gaps_filled": 0,
        "frames_added": 0,
    }
    if not lines or max_gap <= 0:
        return lines, stats

    # Parse only the columns needed for gap detection — avoid full string reformat
    text = "\n".join(lines)
    df = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=["frame", "tid", "x", "y", "w", "h", "score"],
        usecols=[0, 1, 2, 3, 4, 5, 6],
        dtype={
            "frame": int,
            "tid": int,
            "x": float,
            "y": float,
            "w": float,
            "h": float,
            "score": float,
        },
    )

    # Filter to confirmed tracks only
    sizes = df.groupby("tid")["frame"].transform("count")
    df = df[sizes >= min_track_len].sort_values(["tid", "frame"])
    if df.empty:
        return lines, stats

    vals = df.values  # [N, 7]
    same_tid = vals[:-1, 1] == vals[1:, 1]
    gap_frames = np.where(same_tid, vals[1:, 0] - vals[:-1, 0] - 1, 0)
    valid = (gap_frames >= 1) & (gap_frames <= max_gap)
    gap_idx = np.where(valid)[0]

    stats["gaps_filled"] = int(valid.sum())
    if stats["gaps_filled"] == 0:
        return lines, stats

    # Build interpolated rows (only the NEW lines — originals are kept verbatim)
    interp_rows = []
    tracks_done: set[int] = set()
    tail = "-1,-1,-1"

    for i in gap_idx:
        r0 = vals[i]
        r1 = vals[i + 1]
        gap = int(r1[0] - r0[0] - 1)
        stats["frames_added"] += gap
        tracks_done.add(int(r0[1]))
        alphas = np.arange(1, gap + 1, dtype=np.float64) / (gap + 1)
        interp = r0 + alphas[:, None] * (r1 - r0)
        interp[:, 0] = np.arange(int(r0[0]) + 1, int(r0[0]) + gap + 1)
        interp[:, 1] = r0[1]
        for row in interp:
            interp_rows.append(
                f"{int(row[0])},{int(row[1])},{row[2]:.2f},{row[3]:.2f},"
                f"{row[4]:.2f},{row[5]:.2f},{row[6]:.4f},{tail}"
            )

    stats["tracks_interpolated"] = len(tracks_done)

    # Merge original lines + new interpolated lines, sorted by (frame, tid)
    def _sort_key(line: str) -> tuple[int, int]:
        p = line.split(",", 2)
        return int(p[0]), int(p[1])

    all_lines = lines + interp_rows
    all_lines.sort(key=_sort_key)
    return all_lines, stats
