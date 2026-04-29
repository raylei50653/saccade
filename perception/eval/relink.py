import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set


class PythonSemanticRelinker:
    def __init__(
        self,
        sim_threshold: float = 0.985,
        ttl: int = 45,
        ema_beta: float = 0.83,
        spatial_gate: float = 0.11,
        min_lost_frames: int = 2,
        min_iou: float = 0.0,
        mahalanobis_threshold: float = 6.6,
        buffer_size: int = 1,
        min_consistency: float = 0.0,
        rerank_mode: str = "mean",
        reciprocal_margin: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.ttl = ttl
        self.ema_beta = ema_beta
        self.spatial_gate = spatial_gate
        self.min_lost_frames = min_lost_frames
        self.min_iou = min_iou
        self.mahalanobis_threshold = mahalanobis_threshold
        # buffer_size > 1: maintain FIFO buffer of raw embeddings per identity;
        # use their mean for matching (BoT-SORT style tracklet appearance buffer).
        # buffer_size = 1: fall back to single EMA (backward compat).
        self.buffer_size = max(1, buffer_size)
        # min_consistency > 0: reject candidates whose buffer has mean pairwise
        # cosine below this threshold (low consistency → confused identity).
        self.min_consistency = min_consistency
        # rerank_mode controls multi-sample similarity when buffer_size > 1:
        #   "mean"     – dot(query, mean_of_buffer)  [default, BoT-SORT style]
        #   "max"      – max cosine over all samples
        #   "top2_mean"– mean of top-2 per-sample cosines
        #   "weighted" – 0.7*max + 0.3*mean
        if rerank_mode not in {"mean", "max", "top2_mean", "weighted"}:
            raise ValueError(f"Unknown rerank_mode: {rerank_mode!r}")
        self.rerank_mode = rerank_mode
        # D: reciprocal_margin > 0 rejects matches where best_sim - second_best_sim < margin,
        # preventing ambiguous accepts in crowd scenes.
        self.reciprocal_margin = max(0.0, reciprocal_margin)
        self.debug = debug
        self.alias: Dict[int, int] = {}
        self.features: Dict[int, torch.Tensor] = {}
        self.buffers: Dict[int, List[torch.Tensor]] = {}
        self.last_seen: Dict[int, int] = {}
        self.last_boxes: Dict[int, torch.Tensor] = {}
        self.motion: Dict[int, Any] = {}
        self.stats: Dict[str, int] = {
            "attempts": 0,
            "accepted": 0,
            "reject_age": 0,
            "reject_assigned": 0,
            "reject_spatial": 0,
            "reject_mahalanobis": 0,
            "reject_similarity": 0,
            "reject_consistency": 0,
            "reject_margin": 0,
            "new_ids": 0,
        }
        self.accept_sims: List[float] = []
        self.accept_ious: List[float] = []
        self.accept_center_dists: List[float] = []
        self.accept_mahas: List[float] = []

    def _normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(embedding.float(), dim=0)

    def _spatial_metrics(
        self, box: torch.Tensor, old_box: torch.Tensor, w: int, h: int
    ) -> Tuple[float, float]:
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        ocx = (old_box[0] + old_box[2]) * 0.5
        ocy = (old_box[1] + old_box[3]) * 0.5
        dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
        center_norm = float(dist / max(w, h))

        ix1, iy1 = (
            max(float(box[0]), float(old_box[0])),
            max(float(box[1]), float(old_box[1])),
        )
        ix2, iy2 = (
            min(float(box[2]), float(old_box[2])),
            min(float(box[3]), float(old_box[3])),
        )
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        old_area = max(0.0, float(old_box[2] - old_box[0])) * max(
            0.0, float(old_box[3] - old_box[1])
        )
        iou = float(inter / (area + old_area - inter + 1e-6))
        return center_norm, iou

    def update_motion_snapshots(self, snapshots: List[Any]) -> None:
        for snap in snapshots:
            canonical = self.alias.get(snap.obj_id, snap.obj_id)
            self.motion[canonical] = snap

    def _measurement(self, box: torch.Tensor) -> np.ndarray:
        w = max(1e-6, float(box[2] - box[0]))
        h = max(1e-6, float(box[3] - box[1]))
        return np.array(
            [
                (float(box[0]) + float(box[2])) * 0.5,
                (float(box[1]) + float(box[3])) * 0.5,
                w / h,
                h,
            ],
            dtype=np.float32,
        )

    def _mahalanobis(self, box: torch.Tensor, snapshot: Any) -> float:
        state = np.asarray(snapshot.state[:4], dtype=np.float32)
        cov = np.asarray(snapshot.covariance, dtype=np.float32).reshape(8, 8)[:4, :4]
        h = max(float(state[3]), 1e-6)
        pos_std = h / 20.0
        r = np.diag([pos_std**2, pos_std**2, 1e-2, pos_std**2]).astype(np.float32)
        s = cov + r
        residual = self._measurement(box) - state
        try:
            solved = np.linalg.solve(s, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(s) @ residual
        return float(residual @ solved)

    def _buffer_mean(self, cid: int) -> Optional[torch.Tensor]:
        buf = self.buffers.get(cid)
        if not buf:
            return None
        if len(buf) == 1:
            return buf[0]
        return F.normalize(torch.stack(buf).mean(dim=0), dim=0)

    def _buffer_consistency(self, cid: int) -> float:
        buf = self.buffers.get(cid)
        if buf is None or len(buf) < 2:
            return 1.0
        stacked = torch.stack(buf)  # [K, D]
        cosines = stacked @ stacked.T  # [K, K]
        n = len(buf)
        return float((cosines.sum() - n) / (n * (n - 1)))

    def _buffer_sim(self, cid: int, query: torch.Tensor) -> float:
        """Compute similarity between query and stored samples using rerank_mode."""
        buf = self.buffers.get(cid)
        if not buf or self.rerank_mode == "mean":
            ref = self._buffer_mean(cid)
            if ref is None:
                ref = self.features.get(cid)
            if ref is None:
                return -1.0
            return float(torch.dot(query, ref).item())
        stacked = torch.stack(buf)  # [K, D]
        sims = stacked @ query  # [K]
        if self.rerank_mode == "max":
            return float(sims.max().item())
        if self.rerank_mode == "top2_mean":
            k = min(2, sims.shape[0])
            return float(sims.topk(k).values.mean().item())
        # "weighted": 0.7*max + 0.3*mean
        return float(0.7 * sims.max().item() + 0.3 * sims.mean().item())

    def inject_reference(self, canonical_id: int, embedding: torch.Tensor) -> None:
        """C: Replace stored reference with a high-quality external embedding (e.g. from TrackAppearanceBank).
        Called at track-death time so the relinker holds a clean farewell snapshot instead of a drifted EMA."""
        emb = self._normalize(embedding.cpu())
        self.features[canonical_id] = emb.detach()
        if self.buffer_size > 1:
            buf = self.buffers.setdefault(canonical_id, [])
            buf.append(emb.detach())
            del buf[: max(0, len(buf) - self.buffer_size)]

    def resolve(
        self,
        raw_id: int,
        embedding: Optional[torch.Tensor],
        box: torch.Tensor,
        score: float,
        frame_id: int,
        w: int,
        h: int,
        assigned: Set[int],
    ) -> int:
        if embedding is None:
            return self.alias.get(raw_id, raw_id)

        emb = self._normalize(embedding)
        if raw_id not in self.alias:
            self.stats["attempts"] += 1
            best_id, best_sim = None, self.sim_threshold
            second_best_sim = (
                self.sim_threshold - 1.0
            )  # D: track runner-up among gate-passing candidates
            best_iou, best_center, best_maha = 0.0, 0.0, 0.0
            for cid in self.features:
                age = frame_id - self.last_seen.get(cid, -(10**9))
                if cid in assigned:
                    self.stats["reject_assigned"] += 1
                    continue
                if age < self.min_lost_frames or age > self.ttl:
                    self.stats["reject_age"] += 1
                    continue
                center_norm, iou = self._spatial_metrics(
                    box, self.last_boxes[cid], w, h
                )
                if center_norm > self.spatial_gate or iou < self.min_iou:
                    self.stats["reject_spatial"] += 1
                    continue
                maha = 0.0
                if self.mahalanobis_threshold > 0.0:
                    snapshot = self.motion.get(cid)
                    if snapshot is None:
                        self.stats["reject_mahalanobis"] += 1
                        continue
                    maha = self._mahalanobis(box, snapshot)
                    if maha > self.mahalanobis_threshold:
                        self.stats["reject_mahalanobis"] += 1
                        continue
                if self.min_consistency > 0.0 and self.buffer_size > 1:
                    consistency = self._buffer_consistency(cid)
                    if consistency < self.min_consistency:
                        self.stats["reject_consistency"] += 1
                        continue
                if self.buffer_size > 1:
                    sim = self._buffer_sim(cid, emb)
                else:
                    sim = float(torch.dot(emb, self.features[cid]).item())
                if sim > best_sim:
                    if best_id is not None:
                        second_best_sim = best_sim  # D: demote previous winner
                    best_id, best_sim = cid, sim
                    best_iou, best_center, best_maha = iou, center_norm, maha
                else:
                    if sim > second_best_sim:
                        second_best_sim = sim  # D: update runner-up
                    self.stats["reject_similarity"] += 1
            # D: reciprocal margin check — reject ambiguous matches
            if best_id is not None and self.reciprocal_margin > 0.0:
                if best_sim - second_best_sim < self.reciprocal_margin:
                    self.stats["reject_margin"] += 1
                    best_id = None
            if best_id is not None:
                self.stats["accepted"] += 1
                self.accept_sims.append(best_sim)
                self.accept_ious.append(best_iou)
                self.accept_center_dists.append(best_center)
                self.accept_mahas.append(best_maha)
                self.alias[raw_id] = best_id
            else:
                self.stats["new_ids"] += 1
                self.alias[raw_id] = raw_id

        canonical = self.alias[raw_id]
        if self.buffer_size > 1:
            buf = self.buffers.setdefault(canonical, [])
            buf.append(emb.detach())
            if len(buf) > self.buffer_size:
                buf.pop(0)
            self.features[canonical] = F.normalize(
                torch.stack(buf).mean(dim=0), dim=0
            ).detach()
        else:
            old = self.features.get(canonical)
            updated = (
                emb
                if old is None
                else F.normalize(
                    self.ema_beta * old + (1.0 - self.ema_beta) * emb, dim=0
                )
            )
            self.features[canonical] = updated.detach()
        self.last_seen[canonical] = frame_id
        self.last_boxes[canonical] = box
        assigned.add(canonical)
        return canonical

    def report(self) -> None:
        print("🔁 Semantic Relink Report:")
        print(
            "  attempts={attempts} accepted={accepted} new_ids={new_ids} "
            "reject_age={reject_age} reject_assigned={reject_assigned} "
            "reject_spatial={reject_spatial} reject_mahalanobis={reject_mahalanobis} "
            "reject_similarity={reject_similarity} reject_margin={reject_margin}".format(
                **self.stats
            )
        )
        if self.accept_sims:
            print(
                f"  accepted mean_sim={np.mean(self.accept_sims):.3f} "
                f"mean_iou={np.mean(self.accept_ious):.3f} "
                f"mean_center_norm={np.mean(self.accept_center_dists):.3f} "
                f"mean_maha={np.mean(self.accept_mahas):.3f}"
            )
        if self.buffer_size > 1:
            print(
                f"  buffer_size={self.buffer_size} rerank_mode={self.rerank_mode} reject_consistency={self.stats['reject_consistency']}"
            )


try:
    from saccade_tracking_ext import SemanticRelinker as CppSemanticRelinker

    SemanticRelinker = CppSemanticRelinker
except Exception:
    SemanticRelinker = PythonSemanticRelinker
