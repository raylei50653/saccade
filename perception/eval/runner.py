# mypy: ignore-errors
import configparser
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from typing import Any

# Perception/eval modules load local extensions before any torchvision fallback.
from perception.cropper import ZeroCopyCropper
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor

from perception.eval.detection import (
    _box_iou_single,
    detect_adaptive_960_tiled,
    detect_960p_3x2_tiled,
    filter_detections_fast,
    merge_cross_tile_duplicates_fast,
    nms_fast,
)
from perception.eval.gmc import SparseOpticalFlowGMC
from perception.eval.pool import AdaptiveFramePool
from perception.eval.preprocess import (
    GeometryScaleState,
    apply_frame_preprocess,
    geometry_mid_thresh_scale,
    parse_preprocess,
)
from perception.eval.relink import PythonSemanticRelinker, SemanticRelinker
from perception.eval.streaming import DALIStreamerStream
from perception.eval.tracking import GlobalTrackIdMapper
from perception.tracking.tracker_gpu import (
    DynamicReIDController,
    ReIDTrackObservation,
    TrackAppearanceBank,
    need_reid_frame,
)


@dataclass
class IdStabilityState:
    frame_id: int
    box: tuple[float, float, float, float]
    stable_hits: int
    total_hits: int
    score_ema: float


class IdStabilityFilter:
    def __init__(
        self,
        min_hits: int,
        min_iou: float,
        max_center_shift: float,
        max_gap: int,
        score_ema: float,
        min_score_ema: float,
    ) -> None:
        self.min_hits = max(1, min_hits)
        self.min_iou = min_iou
        self.max_center_shift = max_center_shift
        self.max_gap = max(0, max_gap)
        self.score_ema = min(max(score_ema, 0.0), 1.0)
        self.min_score_ema = min_score_ema
        self.states: dict[int, IdStabilityState] = {}

    @staticmethod
    def _iou(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _center_shift_ratio(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        acx = (a[0] + a[2]) * 0.5
        acy = (a[1] + a[3]) * 0.5
        bcx = (b[0] + b[2]) * 0.5
        bcy = (b[1] + b[3]) * 0.5
        aw = max(a[2] - a[0], 1e-6)
        ah = max(a[3] - a[1], 1e-6)
        bw = max(b[2] - b[0], 1e-6)
        bh = max(b[3] - b[1], 1e-6)
        scale = max(((aw * ah) ** 0.5 + (bw * bh) ** 0.5) * 0.5, 1e-6)
        return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale

    def accept(
        self,
        obj_id: int,
        box: tuple[float, float, float, float],
        score: float,
        frame_id: int,
    ) -> bool:
        prev = self.states.get(obj_id)
        if prev is None or frame_id - prev.frame_id > self.max_gap + 1:
            self.states[obj_id] = IdStabilityState(frame_id, box, 1, 1, score)
            return self.min_hits <= 1 and score >= self.min_score_ema

        iou = self._iou(prev.box, box)
        shift = self._center_shift_ratio(prev.box, box)
        is_stable = iou >= self.min_iou or shift <= self.max_center_shift
        stable_hits = prev.stable_hits + 1 if is_stable else 1
        score_ema = self.score_ema * prev.score_ema + (1.0 - self.score_ema) * score
        self.states[obj_id] = IdStabilityState(
            frame_id=frame_id,
            box=box,
            stable_hits=stable_hits,
            total_hits=prev.total_hits + 1,
            score_ema=score_ema,
        )
        return stable_hits >= self.min_hits and score_ema >= self.min_score_ema


@dataclass
class TrackletLifecycleState:
    output_id: int
    frame_id: int
    box: tuple[float, float, float, float]
    score: float
    embedding: torch.Tensor | None


class TrackletLifecycleMerger:
    def __init__(
        self,
        enabled: bool,
        ttl: int,
        min_gap: int,
        spatial_gate: float,
        min_iou: float,
        sim_threshold: float,
        require_embedding: bool,
        ema: float,
    ) -> None:
        self.enabled = enabled
        self.ttl = max(1, ttl)
        self.min_gap = max(0, min_gap)
        self.spatial_gate = spatial_gate
        self.min_iou = min_iou
        self.sim_threshold = sim_threshold
        self.require_embedding = require_embedding
        self.ema = min(max(ema, 0.0), 1.0)
        self.alias: dict[int, int] = {}
        self.states: dict[int, TrackletLifecycleState] = {}
        self.stats = {
            "attempts": 0,
            "accepted": 0,
            "new_ids": 0,
            "reject_age": 0,
            "reject_assigned": 0,
            "reject_spatial": 0,
            "reject_similarity": 0,
        }

    @staticmethod
    def _iou(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _center_gate(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> float:
        acx = (a[0] + a[2]) * 0.5
        acy = (a[1] + a[3]) * 0.5
        bcx = (b[0] + b[2]) * 0.5
        bcy = (b[1] + b[3]) * 0.5
        dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
        return dist / max(float(frame_w), float(frame_h), 1.0)

    @staticmethod
    def _normalize(embedding: torch.Tensor | None) -> torch.Tensor | None:
        if embedding is None:
            return None
        return F.normalize(embedding.detach().float(), dim=0)

    def resolve(
        self,
        local_id: int,
        box: tuple[float, float, float, float],
        score: float,
        frame_id: int,
        frame_w: int,
        frame_h: int,
        embedding: torch.Tensor | None,
        assigned_outputs: set[int],
    ) -> int:
        if not self.enabled:
            return local_id

        emb = self._normalize(embedding)
        output_id = self.alias.get(local_id)
        if output_id is None:
            self.stats["attempts"] += 1
            best_id = None
            best_score = -1.0
            for candidate_id, state in self.states.items():
                age = frame_id - state.frame_id
                if candidate_id in assigned_outputs:
                    self.stats["reject_assigned"] += 1
                    continue
                if age < self.min_gap or age > self.ttl:
                    self.stats["reject_age"] += 1
                    continue
                center_norm = self._center_gate(box, state.box, frame_w, frame_h)
                iou = self._iou(box, state.box)
                if center_norm > self.spatial_gate or iou < self.min_iou:
                    self.stats["reject_spatial"] += 1
                    continue
                sim = 0.0
                if emb is not None and state.embedding is not None:
                    sim = float(torch.dot(emb, state.embedding).item())
                    if sim < self.sim_threshold:
                        self.stats["reject_similarity"] += 1
                        continue
                elif self.require_embedding:
                    self.stats["reject_similarity"] += 1
                    continue
                candidate_score = sim + max(0.0, self.spatial_gate - center_norm) + iou
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_id = candidate_id
            if best_id is None:
                self.stats["new_ids"] += 1
                output_id = local_id
            else:
                self.stats["accepted"] += 1
                output_id = best_id
            self.alias[local_id] = output_id

        old = self.states.get(output_id)
        updated_emb = emb
        if old is not None and old.embedding is not None and emb is not None:
            updated_emb = F.normalize(
                self.ema * old.embedding + (1.0 - self.ema) * emb, dim=0
            )
        elif old is not None and emb is None:
            updated_emb = old.embedding
        self.states[output_id] = TrackletLifecycleState(
            output_id, frame_id, box, score, updated_emb
        )
        assigned_outputs.add(output_id)
        return output_id

    def prune(self, frame_id: int) -> None:
        stale = [
            output_id
            for output_id, state in self.states.items()
            if frame_id - state.frame_id > self.ttl
        ]
        for output_id in stale:
            self.states.pop(output_id, None)

    def report(self) -> None:
        if not self.enabled:
            return
        print("🔗 Tracklet Lifecycle Report:")
        print(
            "  attempts={attempts} accepted={accepted} new_ids={new_ids} "
            "reject_age={reject_age} reject_assigned={reject_assigned} "
            "reject_spatial={reject_spatial} reject_similarity={reject_similarity}".format(
                **self.stats
            )
        )


@dataclass
class MotRecord:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    tail: list[str]


@dataclass
class OutputTracklet:
    track_id: int
    records: list[MotRecord]
    start: int
    end: int
    start_box: tuple[float, float, float, float]
    end_box: tuple[float, float, float, float]
    start_velocity: tuple[float, float]
    end_velocity: tuple[float, float]
    mean_score: float


class OutputAppearanceBank:
    def __init__(
        self,
        *,
        max_samples: int,
        min_score: float,
        min_consistency: float,
    ) -> None:
        self.max_samples = max(1, max_samples)
        self.min_score = min_score
        self.min_consistency = min_consistency
        self.samples: dict[int, list[tuple[float, int, torch.Tensor]]] = {}

    def update(
        self,
        track_id: int,
        embedding: torch.Tensor | None,
        *,
        score: float,
        frame_id: int,
    ) -> None:
        if embedding is None or score < self.min_score:
            return
        emb = F.normalize(embedding.detach().float().cpu(), dim=0)
        samples = self.samples.setdefault(track_id, [])
        samples.append((score, frame_id, emb))
        samples.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del samples[self.max_samples :]

    def count(self, track_id: int) -> int:
        return len(self.samples.get(track_id, []))

    def consistency(self, track_id: int) -> float:
        samples = self.samples.get(track_id, [])
        if len(samples) < 2:
            return 1.0
        stacked = torch.stack([sample[2] for sample in samples])
        cosines = stacked @ stacked.T
        n = len(samples)
        return float((cosines.sum() - n) / max(n * (n - 1), 1))

    def representative(self, track_id: int) -> torch.Tensor | None:
        samples = self.samples.get(track_id, [])
        if not samples:
            return None
        return F.normalize(
            torch.stack([sample[2] for sample in samples]).mean(dim=0), dim=0
        )

    def similarity(self, a: int, b: int) -> float | None:
        a_emb = self.representative(a)
        b_emb = self.representative(b)
        if a_emb is None or b_emb is None:
            return None
        return float(torch.dot(a_emb, b_emb).item())


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, keep: int, merge: int) -> None:
        keep_root = self.find(keep)
        merge_root = self.find(merge)
        if keep_root == merge_root:
            return
        self.parent[merge_root] = keep_root


def _mot_box(record: MotRecord) -> tuple[float, float, float, float]:
    return (record.x, record.y, record.x + record.w, record.y + record.h)


def _box_iou_tuple(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _shift_box(
    box: tuple[float, float, float, float],
    velocity: tuple[float, float],
    dt: float,
) -> tuple[float, float, float, float]:
    dx = velocity[0] * dt
    dy = velocity[1] * dt
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


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
    appearance_bank: OutputAppearanceBank | None = None,
    appearance_gate: bool = False,
    appearance_threshold: float = 0.90,
    appearance_min_samples: int = 1,
) -> tuple[list[str], dict[str, int]]:
    stats = {
        "candidates": 0,
        "accepted": 0,
        "ids_before": 0,
        "ids_after": 0,
        "reject_appearance": 0,
        "reject_appearance_missing": 0,
        "reject_appearance_consistency": 0,
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
            if cost > max_cost:
                continue
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


def run_eval(
    engine: str,
    output: str,
    data_root: str,
    split: str,
    sequences: str,
    max_frames: int,
    conf_threshold: float,
    reid_mode: str = "semantic",
    reid_model: str = "siglip2",
    **kwargs: Any,
) -> None:
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    fps_summary_lines = []
    overall_latency_ms = []
    profile_stages = bool(kwargs.get("profile_stages", False))
    stage_summary_lines = []
    global_id_mapper = GlobalTrackIdMapper()

    detector = TRTYoloDetector(engine_path=engine)

    if reid_mode not in {"off", "tracker", "semantic", "hybrid"}:
        raise ValueError(f"Unsupported reid_mode: {reid_mode}")
    reid_enabled = reid_mode != "off"
    profile_lazy_reid_embeddings = bool(
        kwargs.get("profile_lazy_reid_embeddings", False)
    )
    profile_lazy_reid_candidates = (
        bool(kwargs.get("profile_lazy_reid_candidates", False))
        or profile_lazy_reid_embeddings
    )
    reid_work_enabled = reid_enabled or profile_lazy_reid_embeddings
    _reid_engine = kwargs.pop("reid_engine_path", "") or ""
    _crop_hw: tuple[int, int] = (
        (256, 128) if reid_model in {"transreid", "osnet", "fastreid"} else (224, 224)
    )
    extractor = (
        TRTFeatureExtractor(
            engine_path=_reid_engine,
            model_type=reid_model,
            max_batch=64,
        )
        if reid_work_enabled
        else None
    )
    cropper = (
        ZeroCopyCropper(
            output_size=_crop_hw,
            mode=kwargs.get("reid_crop_mode", "tight"),
            padding=float(kwargs.get("reid_crop_padding", 0.0)),
        )
        if reid_work_enabled
        else None
    )

    reid_interval = max(1, int(kwargs.get("reid_interval", 10)))
    reid_crop_layout = kwargs.get("reid_crop_layout", "full")
    if reid_crop_layout not in {"full", "parts"}:
        raise ValueError(f"Unsupported reid_crop_layout: {reid_crop_layout}")

    gmc_enabled = bool(kwargs.get("gmc", False))
    gmc_downscale = max(1, int(kwargs.get("gmc_downscale", 8)))
    semantic_buffer_size = max(1, int(kwargs.get("semantic_buffer_size", 1)))
    semantic_min_consistency = float(kwargs.get("semantic_min_consistency", 0.0))
    semantic_rerank_mode = str(kwargs.get("semantic_rerank_mode", "mean"))
    semantic_reciprocal_margin = float(kwargs.get("semantic_reciprocal_margin", 0.0))
    semantic_bank_inject = bool(kwargs.get("semantic_bank_inject", True))
    use_semantic_mode = reid_mode in {"semantic", "hybrid"}
    use_tracker_reid = reid_mode in {"tracker", "hybrid"}
    person_class = int(kwargs.get("person_class", 0))
    track_person_only = bool(kwargs.get("track_person_only", True))
    track_thresh = float(kwargs.get("track_thresh", 0.05))
    high_thresh = float(kwargs.get("high_thresh", 0.5))
    match_thresh = float(kwargs.get("match_thresh", 0.8))
    mid_thresh = float(kwargs.get("mid_thresh", 0.10))
    new_track_thresh_arg = kwargs.get("new_track_thresh", None)
    new_track_thresh = (
        0.45 if new_track_thresh_arg is None else float(new_track_thresh_arg)
    )
    tiling = kwargs.get("tiling", "960p_2x2")
    _nms_default = 0.35 if tiling == "960p_3x2" else 0.5
    nms_iou_threshold = float(kwargs.get("nms_iou_threshold") or _nms_default)
    detect_fn = (
        detect_960p_3x2_tiled if tiling == "960p_3x2" else detect_adaptive_960_tiled
    )
    cross_tile_merge = bool(kwargs.get("cross_tile_merge", False))
    geometry_mid_scale = bool(kwargs.get("geometry_mid_scale", False))
    geometry_ref_height_ratio = float(kwargs.get("geometry_ref_height_ratio", 0.12))
    geometry_min_scale = float(kwargs.get("geometry_min_scale", 0.875))
    geometry_max_scale = float(kwargs.get("geometry_max_scale", 1.20))
    geometry_ema_beta = float(kwargs.get("geometry_ema_beta", 0.80))
    geometry_loosen_step = float(kwargs.get("geometry_loosen_step", 0.08))
    geometry_tighten_step = float(kwargs.get("geometry_tighten_step", 0.03))
    geometry_min_samples = int(kwargs.get("geometry_min_samples", 5))
    lazy_reid_min_hit_streak = int(kwargs.get("lazy_reid_min_hit_streak", 2))
    lazy_reid_self_threshold = float(kwargs.get("lazy_reid_self_threshold", 0.85))
    preprocess_modes = parse_preprocess(
        kwargs.get("preprocess", "letterbox,gamma,contrast")
    )
    gamma = float(kwargs.get("gamma", 0.8))
    gamma_luma_threshold = float(kwargs.get("gamma_luma_threshold", 0.35))
    contrast = float(kwargs.get("contrast", 1.2))
    id_stability_filter_enabled = bool(kwargs.get("id_stability_filter", True))
    id_stability_min_hits = int(kwargs.get("id_stability_min_hits", 2))
    id_stability_min_iou = float(kwargs.get("id_stability_min_iou", 0.05))
    id_stability_max_center_shift = float(
        kwargs.get("id_stability_max_center_shift", 2.0)
    )
    id_stability_max_gap = int(kwargs.get("id_stability_max_gap", 1))
    id_stability_score_ema = float(kwargs.get("id_stability_score_ema", 0.70))
    id_stability_min_score_ema = float(kwargs.get("id_stability_min_score_ema", 0.15))
    person_geometry_prior = bool(kwargs.get("person_geometry_prior", True))
    person_min_height_ratio = float(kwargs.get("person_min_height_ratio", 0.018))
    person_min_aspect = float(kwargs.get("person_min_aspect", 1.0))
    person_max_aspect = float(kwargs.get("person_max_aspect", 5.5))
    person_min_area_ratio = float(kwargs.get("person_min_area_ratio", 0.00006))
    person_max_area_ratio = float(kwargs.get("person_max_area_ratio", 0.0))
    geometry_suspect_support = bool(kwargs.get("geometry_suspect_support", True))
    suspect_score_arg = kwargs.get("geometry_suspect_score", None)
    if suspect_score_arg is None:
        geometry_suspect_score = track_thresh + max(
            (mid_thresh - track_thresh) * 0.5, 1e-4
        )
    else:
        geometry_suspect_score = float(suspect_score_arg)
    geometry_suspect_support_score = min(
        max(geometry_suspect_score, track_thresh + 1e-4),
        max(mid_thresh - 1e-4, track_thresh + 1e-4),
    )
    lifecycle_merge_enabled = bool(kwargs.get("lifecycle_merge", False))
    lifecycle_ttl = int(kwargs.get("lifecycle_ttl", 45))
    lifecycle_min_gap = int(kwargs.get("lifecycle_min_gap", 2))
    lifecycle_spatial_gate = float(kwargs.get("lifecycle_spatial_gate", 0.08))
    lifecycle_min_iou = float(kwargs.get("lifecycle_min_iou", 0.0))
    lifecycle_sim_threshold = float(kwargs.get("lifecycle_sim_threshold", 0.90))
    lifecycle_require_embedding = bool(kwargs.get("lifecycle_require_embedding", False))
    lifecycle_ema = float(kwargs.get("lifecycle_ema", 0.83))
    post_lifecycle_merge = bool(kwargs.get("post_lifecycle_merge", False))
    post_lifecycle_ttl = int(kwargs.get("post_lifecycle_ttl", 60))
    post_lifecycle_min_gap = int(kwargs.get("post_lifecycle_min_gap", 1))
    post_lifecycle_velocity_samples = int(
        kwargs.get("post_lifecycle_velocity_samples", 5)
    )
    post_lifecycle_spatial_weight = float(
        kwargs.get("post_lifecycle_spatial_weight", 0.35)
    )
    post_lifecycle_motion_weight = float(
        kwargs.get("post_lifecycle_motion_weight", 0.45)
    )
    post_lifecycle_time_weight = float(kwargs.get("post_lifecycle_time_weight", 0.10))
    post_lifecycle_direction_weight = float(
        kwargs.get("post_lifecycle_direction_weight", 0.25)
    )
    post_lifecycle_max_cost = float(kwargs.get("post_lifecycle_max_cost", 1.25))
    post_lifecycle_appearance_gate = bool(
        kwargs.get("post_lifecycle_appearance_gate", False)
    )
    post_lifecycle_appearance_threshold = float(
        kwargs.get("post_lifecycle_appearance_threshold", 0.90)
    )
    post_lifecycle_appearance_min_samples = max(
        1, int(kwargs.get("post_lifecycle_appearance_min_samples", 1))
    )
    post_lifecycle_appearance_max_samples = max(
        1, int(kwargs.get("post_lifecycle_appearance_max_samples", 5))
    )
    post_lifecycle_appearance_min_score = float(
        kwargs.get("post_lifecycle_appearance_min_score", 0.0)
    )
    post_lifecycle_appearance_min_consistency = float(
        kwargs.get("post_lifecycle_appearance_min_consistency", 0.0)
    )
    # Phase 3: pure motion-based merge is confirmed harmful; force appearance gate when PostMerge is enabled.
    if post_lifecycle_merge and not post_lifecycle_appearance_gate:
        print(
            "⚠️  --post-lifecycle-merge requires --post-lifecycle-appearance-gate (Phase 3 design constraint). Enabling appearance gate automatically."
        )
        post_lifecycle_appearance_gate = True
    min_tracklet_len = max(1, int(kwargs.get("min_tracklet_len", 1)))
    min_tracklet_score = float(kwargs.get("min_tracklet_score", 0.0))
    nsa_kalman = bool(kwargs.get("nsa_kalman", False))
    appearance_bank_enabled = bool(kwargs.get("appearance_bank", True))
    appearance_bank_size = max(1, int(kwargs.get("appearance_bank_size", 5)))
    appearance_bank_min_score = float(kwargs.get("appearance_bank_min_score", 0.45))
    appearance_bank_min_iou = float(kwargs.get("appearance_bank_min_iou", 0.35))
    appearance_bank_consistency_threshold = float(
        kwargs.get("appearance_bank_consistency_threshold", 0.75)
    )
    need_reid_enabled = bool(kwargs.get("need_reid", True))
    per_seq_adapt = bool(kwargs.get("per_seq_adapt", True))
    seqs = (
        sequences.split(",")
        if sequences
        else [d.name for d in (Path(data_root) / split).iterdir() if d.is_dir()]
    )

    def time_stage(stage_totals, stage_name, fn, sync_cuda=False):
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if profile_stages:
            stage_totals[stage_name] += elapsed_ms
        return result, elapsed_ms

    overall_stage_totals = OrderedDict(
        (name, 0.0)
        for name in (
            "fetch",
            "ingest_preprocess",
            "detect",
            "postprocess",
            "post_filter",
            "post_nms",
            "post_merge",
            "reid",
            "lazy_reid",
            "track",
            "relink_write",
            "frame_total",
        )
    )
    overall_profiled_frames = 0
    overall_post_counts = OrderedDict(
        (name, 0)
        for name in (
            "raw_boxes",
            "after_filter",
            "after_nms",
            "after_merge",
        )
    )
    overall_lazy_reid_candidates = 0
    overall_lazy_reid_frames = 0
    overall_lazy_reid_crops = 0
    overall_lazy_reid_self_pairs = 0
    overall_lazy_reid_self_pass = 0
    overall_lazy_reid_self_sim_sum = 0.0
    overall_lazy_reid_arbiter_checks = 0
    overall_lazy_reid_arbiter_approve = 0

    for seq in seqs:
        detector.reset_tracker()
        geometry_scale_state = GeometryScaleState()
        gmc_estimator = (
            SparseOpticalFlowGMC(downscale=gmc_downscale) if gmc_enabled else None
        )
        id_stability_filter = (
            IdStabilityFilter(
                min_hits=id_stability_min_hits,
                min_iou=id_stability_min_iou,
                max_center_shift=id_stability_max_center_shift,
                max_gap=id_stability_max_gap,
                score_ema=id_stability_score_ema,
                min_score_ema=id_stability_min_score_ema,
            )
            if id_stability_filter_enabled
            else None
        )
        lifecycle_merger = TrackletLifecycleMerger(
            enabled=lifecycle_merge_enabled,
            ttl=lifecycle_ttl,
            min_gap=lifecycle_min_gap,
            spatial_gate=lifecycle_spatial_gate,
            min_iou=lifecycle_min_iou,
            sim_threshold=lifecycle_sim_threshold,
            require_embedding=lifecycle_require_embedding,
            ema=lifecycle_ema,
        )
        detector.tracker.set_reid_params(
            cos_threshold=float(kwargs.get("reid_cos_threshold", 0.90)),
            iou_low=float(kwargs.get("reid_iou_low", 0.30)),
            iou_high=float(kwargs.get("reid_iou_high", 0.60)),
            weight=float(kwargs.get("reid_weight", 0.80)),
        )

        _use_python_relinker = semantic_rerank_mode != "mean"
        _relinker_cls = (
            PythonSemanticRelinker if _use_python_relinker else SemanticRelinker
        )
        relinker = (
            _relinker_cls(
                sim_threshold=kwargs.get("semantic_threshold", 0.90),
                ttl=kwargs.get("semantic_ttl", 45),
                ema_beta=kwargs.get("semantic_ema", 0.83),
                spatial_gate=kwargs.get("semantic_spatial_gate", 0.20),
                min_lost_frames=kwargs.get("semantic_min_lost_frames", 2),
                min_iou=kwargs.get("semantic_min_iou", 0.20),
                mahalanobis_threshold=kwargs.get("semantic_mahalanobis_threshold", 0.0),
                buffer_size=semantic_buffer_size,
                min_consistency=semantic_min_consistency,
                rerank_mode=semantic_rerank_mode,
                reciprocal_margin=semantic_reciprocal_margin,
                debug=kwargs.get("semantic_debug", False),
            )
            if use_semantic_mode
            else None
        )

        seq_path = Path(data_root) / split / seq
        if not (seq_path / "seqinfo.ini").exists():
            continue
        config = configparser.ConfigParser()
        config.read(seq_path / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))
        seq_fps = config.getint("Sequence", "frameRate", fallback=30)

        # F-1: Per-sequence adaptive params — scale temporal params by fps/30
        seq_reid_interval = reid_interval
        seq_track_buffer = 30
        if per_seq_adapt and seq_fps != 30:
            fps_scale = seq_fps / 30.0
            seq_reid_interval = max(1, round(reid_interval * fps_scale))
            seq_track_buffer = max(10, round(30 * fps_scale))
        detector.tracker.set_params(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            match_thresh=match_thresh,
            track_buffer=seq_track_buffer,
            mid_thresh=mid_thresh,
            confirm_streak=int(kwargs.get("confirm_streak", 1)),
            confirm_score_thresh=float(kwargs.get("confirm_score_thresh", 0.0)),
            adaptive_confirmation=bool(kwargs.get("adaptive_confirmation", False)),
            new_track_thresh=new_track_thresh,
            nsa_kalman=nsa_kalman,
        )

        pool = AdaptiveFramePool(h_orig, w_orig)
        streamer = DALIStreamerStream(seq_path / "img1")
        stream_iter = iter(streamer)
        results_lines, frame_latencies = [], []
        output_appearance_bank = (
            OutputAppearanceBank(
                max_samples=post_lifecycle_appearance_max_samples,
                min_score=post_lifecycle_appearance_min_score,
                min_consistency=post_lifecycle_appearance_min_consistency,
            )
            if post_lifecycle_appearance_gate
            else None
        )
        primary_appearance_bank = (
            TrackAppearanceBank(
                k=appearance_bank_size,
                min_score=appearance_bank_min_score,
                min_iou=appearance_bank_min_iou,
                consistency_threshold=appearance_bank_consistency_threshold,
            )
            if appearance_bank_enabled
            else None
        )
        dynamic_reid = (
            DynamicReIDController(
                history_size=max(2, int(kwargs.get("reid_history_size", 5))),
                mode=str(kwargs.get("reid_trigger_mode", "event_any")),
                long_memory_decay=float(kwargs.get("reid_long_memory_decay", 0.80)),
                long_memory_trigger=float(kwargs.get("reid_long_memory_trigger", 1.25)),
                score_decay=float(kwargs.get("reid_score_decay", 0.80)),
                score_threshold=float(kwargs.get("reid_score_threshold", 2.0)),
                score_threshold_low=float(
                    kwargs.get("reid_score_threshold_low")
                    if kwargs.get("reid_score_threshold_low") is not None
                    else kwargs.get("reid_score_threshold", 2.0)
                ),
                weight_new=float(kwargs.get("reid_weight_new", 1.0)),
                weight_lost=float(kwargs.get("reid_weight_lost", 1.4)),
                weight_geom=float(kwargs.get("reid_weight_geom", 0.5)),
                weight_conf=float(kwargs.get("reid_weight_conf", 0.5)),
                birth_death_boost=float(kwargs.get("reid_birth_death_boost", 1.0)),
                lost_age_cap=int(kwargs.get("reid_lost_age_cap", 30)),
                unstable_shift_weight=float(
                    kwargs.get("reid_unstable_shift_weight", 1.0)
                ),
                unstable_iou_weight=float(kwargs.get("reid_unstable_iou_weight", 1.0)),
                conf_jitter_gate=float(kwargs.get("reid_conf_jitter_gate", 0.10)),
                trigger_persist_frames=int(
                    kwargs.get("reid_trigger_persist_frames", 1)
                ),
                cooldown_frames=int(kwargs.get("reid_cooldown_frames", 0)),
                birth_death_lost_min=float(
                    kwargs.get("reid_birth_death_lost_min", 0.0)
                ),
            )
            if need_reid_enabled
            else None
        )
        prev_track_ids: set[int] = set()
        start_time = time.time()
        warmup_frames = int(kwargs.get("warmup_frames", 50))
        seq_stage_totals = OrderedDict(
            (name, 0.0) for name in overall_stage_totals.keys()
        )
        seq_post_counts = OrderedDict(
            (name, 0)
            for name in ("raw_boxes", "after_filter", "after_nms", "after_merge")
        )
        seq_lazy_reid_candidates = 0
        seq_lazy_reid_frames = 0
        seq_lazy_reid_crops = 0
        seq_lazy_reid_self_pairs = 0
        seq_lazy_reid_self_pass = 0
        seq_lazy_reid_self_sim_sum = 0.0
        seq_lazy_reid_arbiter_checks = 0
        seq_lazy_reid_arbiter_approve = 0
        lazy_reid_prev_embeddings: dict[int, torch.Tensor] = {}
        seq_profiled_frames = 0
        last_reid_frame = -100

        for frame_id in range(1, frame_end + 1):
            t_e2e_start = time.perf_counter()
            try:
                frame_gpu, _fetch_ms = time_stage(
                    seq_stage_totals,
                    "fetch",
                    lambda: next(stream_iter),
                    sync_cuda=False,
                )
            except StopIteration:
                break
            t_frame_start = time.perf_counter()

            _, _ = time_stage(
                seq_stage_totals,
                "ingest_preprocess",
                lambda: (
                    pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0),
                    apply_frame_preprocess(
                        pool.frame_buffer,
                        preprocess_modes,
                        gamma,
                        gamma_luma_threshold,
                        contrast,
                    ),
                ),
                sync_cuda=True,
            )

            (fused_boxes, fused_scores, fused_classes, is_tiled), _ = time_stage(
                seq_stage_totals,
                "detect",
                lambda: detect_fn(detector, pool, h_orig, w_orig, preprocess_modes),
                sync_cuda=True,
            )

            if fused_boxes.numel() == 0:
                if frame_id > warmup_frames:
                    frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (
                        time.perf_counter() - t_e2e_start
                    ) * 1000
                    seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                continue

            # Keep low-score boxes down to track_thresh so ByteTrack's
            # second-stage association can actually use them.
            if profile_stages:
                torch.cuda.synchronize()
                t_post_start = time.perf_counter()
            raw_box_count = int(fused_scores.numel())
            t_sub_start = time.perf_counter()
            keep_indices, geometry_suspect_mask = filter_detections_fast(
                fused_boxes,
                fused_scores,
                fused_classes,
                score_threshold=min(conf_threshold, track_thresh),
                track_person_only=track_person_only,
                person_class=person_class,
                is_tiled=is_tiled,
                frame_w=w_orig,
                frame_h=h_orig,
                person_geometry_prior=person_geometry_prior,
                geometry_suspect_support=geometry_suspect_support,
                person_min_height_ratio=person_min_height_ratio,
                person_min_aspect=person_min_aspect,
                person_max_aspect=person_max_aspect,
                person_min_area_ratio=person_min_area_ratio,
                person_max_area_ratio=person_max_area_ratio,
            )
            fused_boxes = fused_boxes[keep_indices]
            fused_scores = fused_scores[keep_indices]
            fused_classes = fused_classes[keep_indices]
            suspect_boxes = fused_boxes[geometry_suspect_mask]
            if geometry_suspect_support and geometry_suspect_mask.any():
                fused_scores = fused_scores.clone()
                fused_scores[geometry_suspect_mask] = torch.minimum(
                    fused_scores[geometry_suspect_mask],
                    torch.full_like(
                        fused_scores[geometry_suspect_mask],
                        geometry_suspect_support_score,
                    ),
                )
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["post_filter"] += (
                    time.perf_counter() - t_sub_start
                ) * 1000
            after_filter_count = int(fused_scores.numel())

            if fused_boxes.numel() == 0:
                if frame_id > warmup_frames:
                    frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (
                        time.perf_counter() - t_e2e_start
                    ) * 1000
                    seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                continue

            if is_tiled and fused_boxes.numel() > 0:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_sub_start = time.perf_counter()
                keep = nms_fast(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    nms_iou_threshold,
                    class_aware=not track_person_only,
                )
                fused_boxes = fused_boxes[keep]
                fused_scores = fused_scores[keep]
                fused_classes = fused_classes[keep]
                geometry_suspect_mask = geometry_suspect_mask[keep]
                suspect_boxes = fused_boxes[geometry_suspect_mask]
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["post_nms"] += (
                        time.perf_counter() - t_sub_start
                    ) * 1000
            after_nms_count = int(fused_scores.numel())

            if cross_tile_merge and is_tiled and fused_boxes.numel() > 1:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_sub_start = time.perf_counter()
                fused_boxes, fused_scores, fused_classes = (
                    merge_cross_tile_duplicates_fast(
                        fused_boxes, fused_scores, fused_classes
                    )
                )
                geometry_suspect_mask = torch.zeros_like(fused_scores, dtype=torch.bool)
                suspect_boxes = fused_boxes[:0]
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["post_merge"] += (
                        time.perf_counter() - t_sub_start
                    ) * 1000
            after_merge_count = int(fused_scores.numel())
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["postprocess"] += (
                    time.perf_counter() - t_post_start
                ) * 1000
                if frame_id > warmup_frames:
                    seq_post_counts["raw_boxes"] += raw_box_count
                    seq_post_counts["after_filter"] += after_filter_count
                    seq_post_counts["after_nms"] += after_nms_count
                    seq_post_counts["after_merge"] += after_merge_count

            embeddings = None
            _do_reid = (
                reid_work_enabled
                and extractor is not None
                and cropper is not None
                and fused_boxes.numel() > 0
            )
            if _do_reid:
                # 💡 Phase 4: Enforce a minimum interval even with need_reid logic
                # to prevent pathological cases where every frame triggers ReID.
                MIN_REID_GAP = 5
                time_since_last_reid = frame_id - last_reid_frame

                if time_since_last_reid < MIN_REID_GAP:
                    _do_reid = False
                elif need_reid_enabled:
                    if dynamic_reid is not None:
                        _do_reid = dynamic_reid.should_reid(after_merge_count)
                    else:
                        _do_reid = need_reid_frame(prev_track_ids, after_merge_count)
                else:
                    _do_reid = frame_id % seq_reid_interval == 0

                if _do_reid:
                    last_reid_frame = frame_id
                    if primary_appearance_bank is not None:
                        bank_reps = primary_appearance_bank.representatives()
                        if bank_reps:
                            detector.tracker.set_reference_features_from_bank(bank_reps)
                        clean_ids = primary_appearance_bank.clean_ids()
                        if clean_ids:
                            _clean_ids_list = sorted(clean_ids)
                            _ids_t = torch.tensor(_clean_ids_list, dtype=torch.int32)
                            _flags_t = torch.ones(
                                len(_clean_ids_list), dtype=torch.bool
                            )
                            detector.tracker.set_clean_embedding_flags(_ids_t, _flags_t)
                        else:
                            detector.tracker.set_clean_embedding_flags(
                                torch.zeros(0, dtype=torch.int32),
                                torch.zeros(0, dtype=torch.bool),
                            )

            if reid_enabled and _do_reid:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_reid_start = time.perf_counter()
                frame_batch = pool.frame_buffer.unsqueeze(0)
                if reid_crop_layout == "parts":
                    crops = cropper.process_parts(frame_batch, fused_boxes)
                    if crops.numel() > 0:
                        embeddings = extractor.extract_parts_fused(crops)
                else:
                    crops = cropper.process(frame_batch, fused_boxes)
                    if crops.numel() > 0:
                        embeddings = extractor.extract(crops)
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["reid"] += (
                        time.perf_counter() - t_reid_start
                    ) * 1000

            mid_thresh_scale = geometry_mid_thresh_scale(
                fused_boxes,
                fused_classes,
                h_orig,
                enabled=geometry_mid_scale,
                person_class=person_class,
                track_person_only=track_person_only,
                ref_height_ratio=geometry_ref_height_ratio,
                min_scale=geometry_min_scale,
                max_scale=geometry_max_scale,
                ema_beta=geometry_ema_beta,
                loosen_step=geometry_loosen_step,
                tighten_step=geometry_tighten_step,
                min_samples=geometry_min_samples,
                state=geometry_scale_state,
            )
            gmc_warp = None
            if gmc_estimator is not None:
                _raw_warp = gmc_estimator.estimate(pool.frame_buffer)
                if _raw_warp is not None:
                    gmc_warp = _raw_warp.to(fused_boxes.device)
            tracks, _ = time_stage(
                seq_stage_totals,
                "track",
                lambda: detector.tracker.update(
                    fused_boxes,
                    fused_scores,
                    fused_classes.to(torch.int32),
                    embeddings=embeddings if use_tracker_reid else None,
                    gmc=gmc_warp,
                    mid_thresh_scale=mid_thresh_scale,
                ),
                sync_cuda=True,
            )

            if profile_lazy_reid_candidates:
                candidates = detector.tracker.get_tentative_candidates()
                ready_candidates = [
                    c
                    for c in candidates
                    if c.hit_streak >= lazy_reid_min_hit_streak
                    and c.hit_streak < c.required_confirm_streak
                ]
                seq_lazy_reid_candidates += len(ready_candidates)
                seq_lazy_reid_frames += 1
                if (
                    profile_lazy_reid_embeddings
                    and extractor
                    and cropper
                    and candidates
                ):
                    ready_ids = {int(c.obj_id) for c in ready_candidates}

                    def _profile_lazy_reid_embeddings() -> tuple[
                        int, int, int, float, int, int, set[int]
                    ]:
                        embed_candidates = [
                            c
                            for c in candidates
                            if int(c.class_id) == person_class and c.hit_streak >= 1
                        ]
                        if not embed_candidates:
                            return 0, 0, 0, 0.0, 0, 0, set()
                        cand_boxes = torch.tensor(
                            [[c.x1, c.y1, c.x2, c.y2] for c in embed_candidates],
                            device=pool.frame_buffer.device,
                            dtype=torch.float32,
                        )
                        crops = cropper.process(
                            pool.frame_buffer.unsqueeze(0), cand_boxes
                        )
                        if crops.numel() == 0:
                            return 0, 0, 0, 0.0, 0, 0, set()
                        cand_embeddings = extractor.extract(crops)
                        pairs, passed, sim_sum = 0, 0, 0.0
                        arbiter_checks, arbiter_approve = 0, 0
                        seen_ids: set[int] = set()
                        for cand, emb in zip(embed_candidates, cand_embeddings):
                            tid = int(cand.obj_id)
                            seen_ids.add(tid)
                            prev = lazy_reid_prev_embeddings.get(tid)
                            if prev is not None:
                                sim = float(torch.dot(prev, emb).item())
                                pairs += 1
                                sim_sum += sim
                                if sim >= lazy_reid_self_threshold:
                                    passed += 1
                                if tid in ready_ids:
                                    arbiter_checks += 1
                                    if sim >= lazy_reid_self_threshold:
                                        arbiter_approve += 1
                            lazy_reid_prev_embeddings[tid] = emb.detach()
                        return (
                            len(embed_candidates),
                            pairs,
                            passed,
                            sim_sum,
                            arbiter_checks,
                            arbiter_approve,
                            seen_ids,
                        )

                    (
                        (
                            crop_count,
                            pair_count,
                            pass_count,
                            sim_sum,
                            arbiter_checks,
                            arbiter_approve,
                            seen_ids,
                        ),
                        _,
                    ) = time_stage(
                        seq_stage_totals,
                        "lazy_reid",
                        _profile_lazy_reid_embeddings,
                        sync_cuda=True,
                    )
                    seq_lazy_reid_crops += crop_count
                    seq_lazy_reid_self_pairs += pair_count
                    seq_lazy_reid_self_pass += pass_count
                    seq_lazy_reid_self_sim_sum += sim_sum
                    seq_lazy_reid_arbiter_checks += arbiter_checks
                    seq_lazy_reid_arbiter_approve += arbiter_approve
                    if seen_ids:
                        for stale_id in (
                            set(lazy_reid_prev_embeddings.keys()) - seen_ids
                        ):
                            lazy_reid_prev_embeddings.pop(stale_id, None)

            if profile_stages:
                torch.cuda.synchronize()
                t_relink_write_start = time.perf_counter()
            if relinker:
                relinker.update_motion_snapshots(detector.tracker.get_state_snapshots())

            assigned_ids: set = set()
            lifecycle_assigned_ids: set[int] = set()
            for t in tracks:
                if int(t.class_id) != person_class:
                    continue
                raw_box = (float(t.x1), float(t.y1), float(t.x2), float(t.y2))
                if (
                    geometry_suspect_support
                    and suspect_boxes.numel() > 0
                    and float(t.score) <= geometry_suspect_support_score + 1e-4
                ):
                    tb = torch.tensor(
                        [t.x1, t.y1, t.x2, t.y2], device=suspect_boxes.device
                    )
                    if float(_box_iou_single(tb, suspect_boxes).max()) > 0.5:
                        continue
                if id_stability_filter and not id_stability_filter.accept(
                    int(t.obj_id), raw_box, float(t.score), frame_id
                ):
                    continue
                tid = t.obj_id
                emb = None
                det_idx = getattr(t, "det_idx", -1)
                if (
                    embeddings is not None
                    and det_idx >= 0
                    and det_idx < fused_boxes.shape[0]
                ):
                    emb = embeddings[det_idx]
                    if primary_appearance_bank is not None:
                        tb = torch.tensor(
                            [t.x1, t.y1, t.x2, t.y2], device=fused_boxes.device
                        )
                        match_iou = float(
                            _box_iou_single(
                                tb, fused_boxes[det_idx : det_idx + 1]
                            ).item()
                        )
                        if match_iou > 0.35:
                            primary_appearance_bank.update(
                                int(t.obj_id),
                                embeddings[det_idx],
                                det_score=float(fused_scores[det_idx]),
                                iou=match_iou,
                                frame_id=frame_id,
                                geometry_clean=True,
                                suspect_box=bool(geometry_suspect_mask[det_idx])
                                if geometry_suspect_mask.numel() > det_idx
                                else False,
                            )
                if relinker:
                    reid_emb = emb
                    if (
                        primary_appearance_bank is not None
                        and not primary_appearance_bank.is_consistent(int(t.obj_id))
                    ):
                        reid_emb = None  # Phase 2: consistency gate — don't relink with dirty bank
                    tid = relinker.resolve(
                        t.obj_id,
                        reid_emb,
                        (t.x1, t.y1, t.x2, t.y2),
                        t.score,
                        frame_id,
                        w_orig,
                        h_orig,
                        assigned_ids,
                    )
                tid = lifecycle_merger.resolve(
                    int(tid),
                    raw_box,
                    float(t.score),
                    frame_id,
                    w_orig,
                    h_orig,
                    emb if relinker else None,
                    lifecycle_assigned_ids,
                )
                global_tid = global_id_mapper.map(seq, tid)
                x1, y1, x2, y2 = t.x1, t.y1, t.x2, t.y2
                results_lines.append(
                    f"{frame_id},{global_tid},{max(0, x1):.2f},{max(0, y1):.2f},"
                    f"{min(w_orig, x2) - max(0, x1):.2f},{min(h_orig, y2) - max(0, y1):.2f},"
                    f"{t.score:.4f},-1,-1,-1"
                )
                if output_appearance_bank is not None:
                    output_appearance_bank.update(
                        int(global_tid),
                        emb,
                        score=float(t.score),
                        frame_id=frame_id,
                    )
            lifecycle_merger.prune(frame_id)
            curr_track_ids = {int(t.obj_id) for t in tracks}
            # C: at track death, replace relinker's drifted EMA with the bank's quality-filtered representative
            if (
                relinker is not None
                and semantic_bank_inject
                and primary_appearance_bank is not None
            ):
                just_lost = prev_track_ids - curr_track_ids
                for _lost_tid in just_lost:
                    if not primary_appearance_bank.is_consistent(_lost_tid):
                        continue
                    _rep = primary_appearance_bank.representative(_lost_tid)
                    if _rep is None:
                        continue
                    _canonical = relinker.alias.get(_lost_tid, _lost_tid)
                    if _canonical in relinker.features:
                        relinker.inject_reference(_canonical, _rep)
            if dynamic_reid is not None:
                dynamic_reid.observe(
                    {
                        int(t.obj_id): ReIDTrackObservation(
                            box=(float(t.x1), float(t.y1), float(t.x2), float(t.y2)),
                            det_score=float(t.score),
                        )
                        for t in tracks
                        if int(t.class_id) == person_class
                    },
                    gmc=gmc_warp if gmc_enabled else None,
                )
            if primary_appearance_bank is not None:
                primary_appearance_bank.prune(curr_track_ids)
            prev_track_ids = curr_track_ids
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["relink_write"] += (
                    time.perf_counter() - t_relink_write_start
                ) * 1000

            if frame_id > warmup_frames:
                frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
            if profile_stages and frame_id > warmup_frames:
                seq_stage_totals["frame_total"] += (
                    time.perf_counter() - t_e2e_start
                ) * 1000
                seq_profiled_frames += 1
            if frame_id % 100 == 0:
                print(f"🎬 {seq} [{frame_id}/{frame_end}]")

        if frame_latencies:
            lats = np.array(frame_latencies)
            mean_ms = float(np.mean(lats))
            fps = 1000.0 / mean_ms
            print(f"\n📊 Production Latency Report for {seq}:")
            print(f"  - FPS:  {fps:.2f}")
            print(f"  - Mean latency: {mean_ms:.2f} ms")
            fps_summary_lines.append(
                f"{seq}\tfps={fps:.2f}\tmean_ms={mean_ms:.2f}\tframes={len(frame_latencies)}"
            )
            overall_latency_ms.extend(frame_latencies)
        else:
            fps_summary_lines.append(f"{seq}\tfps=n/a\tmean_ms=n/a\tframes=0")

        results_lines, post_merge_stats = post_merge_output_tracklets(
            results_lines,
            enabled=post_lifecycle_merge,
            ttl=post_lifecycle_ttl,
            min_gap=post_lifecycle_min_gap,
            velocity_samples=post_lifecycle_velocity_samples,
            spatial_weight=post_lifecycle_spatial_weight,
            motion_weight=post_lifecycle_motion_weight,
            time_weight=post_lifecycle_time_weight,
            direction_weight=post_lifecycle_direction_weight,
            max_cost=post_lifecycle_max_cost,
            appearance_bank=output_appearance_bank,
            appearance_gate=post_lifecycle_appearance_gate,
            appearance_threshold=post_lifecycle_appearance_threshold,
            appearance_min_samples=post_lifecycle_appearance_min_samples,
        )
        if post_lifecycle_merge:
            print(
                "🔗 Post Lifecycle Merge: "
                f"candidates={post_merge_stats['candidates']} "
                f"accepted={post_merge_stats['accepted']} "
                f"ids={post_merge_stats['ids_before']}->{post_merge_stats['ids_after']} "
                f"reject_app={post_merge_stats['reject_appearance']} "
                f"reject_app_missing={post_merge_stats['reject_appearance_missing']} "
                f"reject_app_consistency={post_merge_stats['reject_appearance_consistency']}"
            )

        results_lines, quality_stats = filter_low_quality_tracklets(
            results_lines,
            min_len=min_tracklet_len,
            min_score=min_tracklet_score,
        )
        if quality_stats["removed"] > 0:
            print(
                f"🧹 Quality Filter: removed={quality_stats['removed']} "
                f"ids={quality_stats['before']}->{quality_stats['after']}"
            )

        Path(output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ Finished {seq} (Total Time: {time.time() - start_time:.2f}s)")
        if relinker:
            relinker.report()
        lifecycle_merger.report()
        if profile_stages and seq_profiled_frames > 0:
            print(f"\n🧪 Stage Profile for {seq}:")
            for stage_name, total_ms in seq_stage_totals.items():
                mean_ms = total_ms / seq_profiled_frames
                share = total_ms / max(seq_stage_totals["frame_total"], 1e-6) * 100.0
                print(f"  - {stage_name}: {mean_ms:.2f} ms/frame ({share:.1f}%)")
                overall_stage_totals[stage_name] += total_ms
            if any(seq_post_counts.values()):
                print("  - post_counts:")
                for count_name, total_count in seq_post_counts.items():
                    mean_count = total_count / seq_profiled_frames
                    print(f"    - {count_name}: {mean_count:.1f} boxes/frame")
                    overall_post_counts[count_name] += total_count
            if profile_lazy_reid_candidates and seq_lazy_reid_frames > 0:
                mean_lazy = seq_lazy_reid_candidates / seq_lazy_reid_frames
                print(
                    f"  - lazy_reid_candidates: {mean_lazy:.2f}/frame ({seq_lazy_reid_candidates} total)"
                )
                overall_lazy_reid_candidates += seq_lazy_reid_candidates
                overall_lazy_reid_frames += seq_lazy_reid_frames
                overall_lazy_reid_crops += seq_lazy_reid_crops
                overall_lazy_reid_self_pairs += seq_lazy_reid_self_pairs
                overall_lazy_reid_self_pass += seq_lazy_reid_self_pass
                overall_lazy_reid_self_sim_sum += seq_lazy_reid_self_sim_sum
                overall_lazy_reid_arbiter_checks += seq_lazy_reid_arbiter_checks
                overall_lazy_reid_arbiter_approve += seq_lazy_reid_arbiter_approve
                if profile_lazy_reid_embeddings:
                    mean_crops = seq_lazy_reid_crops / seq_lazy_reid_frames
                    mean_sim = seq_lazy_reid_self_sim_sum / max(
                        seq_lazy_reid_self_pairs, 1
                    )
                    pass_rate = (
                        seq_lazy_reid_self_pass
                        / max(seq_lazy_reid_self_pairs, 1)
                        * 100.0
                    )
                    print(
                        f"  - lazy_reid_embeddings: {mean_crops:.2f} crops/frame, "
                        f"self_pairs={seq_lazy_reid_self_pairs}, mean_cos={mean_sim:.3f}, "
                        f"pass@{lazy_reid_self_threshold:.2f}={pass_rate:.1f}%"
                    )
                    arbiter_rate = (
                        seq_lazy_reid_arbiter_approve
                        / max(seq_lazy_reid_arbiter_checks, 1)
                        * 100.0
                    )
                    print(
                        f"  - lazy_reid_arbiter_dry_run: checks={seq_lazy_reid_arbiter_checks}, "
                        f"approve={seq_lazy_reid_arbiter_approve} ({arbiter_rate:.1f}%)"
                    )
            overall_profiled_frames += seq_profiled_frames
            stage_summary_lines.append(f"[{seq}] frames={seq_profiled_frames}")
            for stage_name, total_ms in seq_stage_totals.items():
                mean_ms = total_ms / seq_profiled_frames
                share = total_ms / max(seq_stage_totals["frame_total"], 1e-6) * 100.0
                stage_summary_lines.append(
                    f"{stage_name}\tmean_ms={mean_ms:.2f}\ttotal_ms={total_ms:.2f}\tshare={share:.1f}%"
                )
            for count_name, total_count in seq_post_counts.items():
                mean_count = total_count / seq_profiled_frames
                stage_summary_lines.append(
                    f"{count_name}\tmean={mean_count:.1f}\ttotal={total_count}"
                )
            if profile_lazy_reid_candidates and seq_lazy_reid_frames > 0:
                mean_lazy = seq_lazy_reid_candidates / seq_lazy_reid_frames
                stage_summary_lines.append(
                    f"lazy_reid_candidates\tmean={mean_lazy:.2f}\ttotal={seq_lazy_reid_candidates}"
                )
                if profile_lazy_reid_embeddings:
                    mean_crops = seq_lazy_reid_crops / seq_lazy_reid_frames
                    mean_sim = seq_lazy_reid_self_sim_sum / max(
                        seq_lazy_reid_self_pairs, 1
                    )
                    pass_rate = (
                        seq_lazy_reid_self_pass
                        / max(seq_lazy_reid_self_pairs, 1)
                        * 100.0
                    )
                    stage_summary_lines.append(
                        f"lazy_reid_embeddings\tmean_crops={mean_crops:.2f}\t"
                        f"self_pairs={seq_lazy_reid_self_pairs}\tmean_cos={mean_sim:.3f}\t"
                        f"pass_rate={pass_rate:.1f}%"
                    )
                    arbiter_rate = (
                        seq_lazy_reid_arbiter_approve
                        / max(seq_lazy_reid_arbiter_checks, 1)
                        * 100.0
                    )
                    stage_summary_lines.append(
                        f"lazy_reid_arbiter_dry_run\tchecks={seq_lazy_reid_arbiter_checks}\t"
                        f"approve={seq_lazy_reid_arbiter_approve}\tapprove_rate={arbiter_rate:.1f}%"
                    )
            stage_summary_lines.append("")

    if fps_summary_lines:
        if overall_latency_ms:
            overall_mean_ms = float(np.mean(np.array(overall_latency_ms)))
            overall_fps = 1000.0 / overall_mean_ms
            fps_summary_lines.append(
                f"OVERALL\tfps={overall_fps:.2f}\tmean_ms={overall_mean_ms:.2f}\tframes={len(overall_latency_ms)}"
            )
            print(
                f"\n📈 Overall throughput: {overall_fps:.2f} FPS ({overall_mean_ms:.2f} ms)"
            )
        (output_root / "_fps_summary.txt").write_text(
            "\n".join(fps_summary_lines) + "\n"
        )
    mapping_lines = global_id_mapper.dump_lines()
    if mapping_lines:
        (output_root / "_global_id_map.txt").write_text("\n".join(mapping_lines) + "\n")
    if profile_stages and overall_profiled_frames > 0:
        print(f"\n🧪 Overall Stage Profile ({overall_profiled_frames} frames):")
        stage_summary_lines.append(f"[OVERALL] frames={overall_profiled_frames}")
        for stage_name, total_ms in overall_stage_totals.items():
            mean_ms = total_ms / overall_profiled_frames
            share = total_ms / max(overall_stage_totals["frame_total"], 1e-6) * 100.0
            print(f"  - {stage_name}: {mean_ms:.2f} ms/frame ({share:.1f}%)")
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={mean_ms:.2f}\ttotal_ms={total_ms:.2f}\tshare={share:.1f}%"
            )
        if any(overall_post_counts.values()):
            print("  - post_counts:")
            for count_name, total_count in overall_post_counts.items():
                mean_count = total_count / overall_profiled_frames
                print(f"    - {count_name}: {mean_count:.1f} boxes/frame")
                stage_summary_lines.append(
                    f"{count_name}\tmean={mean_count:.1f}\ttotal={total_count}"
                )
        if profile_lazy_reid_candidates and overall_lazy_reid_frames > 0:
            mean_lazy = overall_lazy_reid_candidates / overall_lazy_reid_frames
            print(
                f"  - lazy_reid_candidates: {mean_lazy:.2f}/frame ({overall_lazy_reid_candidates} total)"
            )
            stage_summary_lines.append(
                f"lazy_reid_candidates\tmean={mean_lazy:.2f}\ttotal={overall_lazy_reid_candidates}"
            )
            if profile_lazy_reid_embeddings:
                mean_crops = overall_lazy_reid_crops / overall_lazy_reid_frames
                mean_sim = overall_lazy_reid_self_sim_sum / max(
                    overall_lazy_reid_self_pairs, 1
                )
                pass_rate = (
                    overall_lazy_reid_self_pass
                    / max(overall_lazy_reid_self_pairs, 1)
                    * 100.0
                )
                print(
                    f"  - lazy_reid_embeddings: {mean_crops:.2f} crops/frame, "
                    f"self_pairs={overall_lazy_reid_self_pairs}, mean_cos={mean_sim:.3f}, "
                    f"pass@{lazy_reid_self_threshold:.2f}={pass_rate:.1f}%"
                )
                arbiter_rate = (
                    overall_lazy_reid_arbiter_approve
                    / max(overall_lazy_reid_arbiter_checks, 1)
                    * 100.0
                )
                print(
                    f"  - lazy_reid_arbiter_dry_run: checks={overall_lazy_reid_arbiter_checks}, "
                    f"approve={overall_lazy_reid_arbiter_approve} ({arbiter_rate:.1f}%)"
                )
                stage_summary_lines.append(
                    f"lazy_reid_embeddings\tmean_crops={mean_crops:.2f}\t"
                    f"self_pairs={overall_lazy_reid_self_pairs}\tmean_cos={mean_sim:.3f}\t"
                    f"pass_rate={pass_rate:.1f}%"
                )
                stage_summary_lines.append(
                    f"lazy_reid_arbiter_dry_run\tchecks={overall_lazy_reid_arbiter_checks}\t"
                    f"approve={overall_lazy_reid_arbiter_approve}\tapprove_rate={arbiter_rate:.1f}%"
                )
        (output_root / "_stage_profile.txt").write_text(
            "\n".join(stage_summary_lines) + "\n"
        )
