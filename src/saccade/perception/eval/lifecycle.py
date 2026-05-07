import math
import torch
import torch.nn.functional as F
from .types import (
    IdStabilityState,
    TrackletLifecycleState,
)

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

    def accept_many(
        self,
        candidates: list[tuple[int, tuple[float, float, float, float], float]],
        frame_id: int,
    ) -> list[bool]:
        results: list[bool] = []
        for obj_id, box, score in candidates:
            results.append(self.accept(obj_id, box, score, frame_id))
        return results


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

    def resolve_many(
        self,
        candidates: list[tuple[int, tuple[float, float, float, float], float, torch.Tensor | None]],
        *,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> list[int]:
        assigned_outputs: set[int] = set()
        return [
            self.resolve(
                local_id,
                box,
                score,
                frame_id,
                frame_w,
                frame_h,
                embedding,
                assigned_outputs,
            )
            for local_id, box, score, embedding in candidates
        ]

    def resolve_many_packed(
        self,
        local_ids: list[int],
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
        embeddings: list[torch.Tensor | None],
        *,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> list[int]:
        assigned_outputs: set[int] = set()
        return [
            self.resolve(
                local_id,
                box,
                score,
                frame_id,
                frame_w,
                frame_h,
                embedding,
                assigned_outputs,
            )
            for local_id, box, score, embedding in zip(
                local_ids, boxes, scores, embeddings
            )
        ]

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
