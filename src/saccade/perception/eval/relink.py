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
        iou_weight: float = 0.0,
        mahalanobis_weight: float = 0.0,
        w_sim_base: float = 1.0,
        w_iou_base: float = 0.0,
        w_maha_base: float = 0.0,
        shift_ambiguity: float = 0.0,
        shift_lost_age: float = 0.0,
        dynamic_margin_crowd: float = 0.0,
        dynamic_margin_age: float = 0.0,
        biometric_threshold: float = 0.0,
        debug: bool = False,
        clean_score_threshold: float = 0.0,
        clean_margin_ratio: float = 0.0,
        clean_min_aspect: float = 0.0,
        clean_max_aspect: float = 99.0,
        strict_sim_threshold: float = 0.0,
        experimental_mode: str = "standard",
        appearance_first_sim_threshold: float = 0.95,
        appearance_first_margin: float = 0.03,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
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
        # Joint scoring: blend IoU and normalised motion evidence into the ranking score.
        # Default 0 → pure cosine (backward-compatible).
        self.iou_weight = max(0.0, float(iou_weight))
        self.mahalanobis_weight = max(0.0, float(mahalanobis_weight))

        # A1 Unified Score base weights and shifts
        self.w_sim_base = max(0.0, float(w_sim_base))
        self.w_iou_base = max(0.0, float(w_iou_base))
        self.w_maha_base = max(0.0, float(w_maha_base))
        self.shift_ambiguity = float(shift_ambiguity)
        self.shift_lost_age = float(shift_lost_age)

        # Dynamic margin: add context-sensitive increments to reciprocal_margin.
        #   crowd  → +margin per extra gate-passing competitor (caps at 8 competitors)
        #   age    → +margin proportional to lost_frames / ttl
        self.dynamic_margin_crowd = max(0.0, float(dynamic_margin_crowd))
        self.dynamic_margin_age = max(0.0, float(dynamic_margin_age))
        self.biometric_threshold = max(0.0, float(biometric_threshold))
        if experimental_mode not in {"standard", "appearance_first"}:
            raise ValueError(f"Unknown experimental_mode: {experimental_mode!r}")
        self.experimental_mode = experimental_mode
        self.appearance_first_sim_threshold = max(
            0.0, float(appearance_first_sim_threshold)
        )
        self.appearance_first_margin = max(0.0, float(appearance_first_margin))
        self.clean_score_threshold = clean_score_threshold
        self.clean_margin_ratio = clean_margin_ratio
        self.clean_min_aspect = clean_min_aspect
        self.clean_max_aspect = clean_max_aspect
        self.strict_sim_threshold = (
            strict_sim_threshold if strict_sim_threshold > 0.0 else sim_threshold
        )
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
            "reject_age_too_fresh": 0,
            "reject_age_too_old": 0,
            "reject_assigned": 0,
            "reject_spatial": 0,
            "reject_mahalanobis": 0,
            "reject_similarity": 0,
            "reject_consistency": 0,
            "reject_margin": 0,
            "reject_quality": 0,
            "reject_quality_score": 0,
            "reject_quality_margin": 0,
            "reject_quality_aspect_low": 0,
            "reject_quality_aspect_high": 0,
            "appearance_first_bypass": 0,
            "new_ids": 0,
            "reject_biometric": 0,
        }
        self._bio_tracker: Any = None
        self.accept_sims: List[float] = []
        self.accept_ious: List[float] = []
        self.accept_center_dists: List[float] = []
        self.accept_mahas: List[float] = []
        self.reject_age_fresh_ages: List[int] = []
        self.reject_age_old_ages: List[int] = []
        self.age_gate_pass_ages: List[int] = []
        self.age_gate_pass_outcomes: List[tuple[int, str]] = []
        self.age_gate_pass_spatial_subreasons: List[tuple[int, str]] = []
        self.reference_alignment_samples: List[
            tuple[int, float, float, float, float]
        ] = []

    def set_biometric_tracker(self, tracker: Any) -> None:
        self._bio_tracker = tracker

    def _bio_distance(self, id_a: int, id_b: int) -> float:
        if self._bio_tracker is None or self.biometric_threshold <= 0.0:
            return 0.0
        pa = self._bio_tracker.get_biometric(int(id_a))
        pb = self._bio_tracker.get_biometric(int(id_b))
        if pa is None or pb is None:
            return 0.0
        dist = 0.0
        for k in ("r_leg", "r_shoulder", "r_head"):
            if k in pa and k in pb:
                dist += abs(pa[k] - pb[k])
        return dist

    def _normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(embedding.float().to(self.device), dim=0)

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

    def motion_candidate_ids(self, frame_id: int = -1) -> List[int]:
        if self.mahalanobis_threshold <= 0.0 and not self.debug:
            return []
        ids: List[int] = []
        for cid in self.features:
            if frame_id >= 0:
                age = frame_id - self.last_seen.get(cid, -(10**9))
                if age < self.min_lost_frames or age > self.ttl:
                    continue
            ids.append(cid)
        return ids

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

    def _motion_box(self, snapshot: Any) -> Optional[torch.Tensor]:
        try:
            cx, cy, aspect, h = [float(v) for v in snapshot.state[:4]]
        except Exception:
            return None
        if h <= 0.0 or aspect <= 0.0:
            return None
        w = aspect * h
        return torch.tensor(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
            dtype=torch.float32,
            device=self.device,
        )

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
        emb = self._normalize(embedding)
        self.features[canonical_id] = emb.detach()
        if self.buffer_size > 1:
            buf = self.buffers.setdefault(canonical_id, [])
            buf.append(emb.detach())
            del buf[: max(0, len(buf) - self.buffer_size)]

    def inject_references_many(
        self, references: List[Tuple[int, torch.Tensor]]
    ) -> None:
        for canonical_id, embedding in references:
            self.inject_reference(canonical_id, embedding)

    def canonical_id(self, raw_id: int) -> int:
        return self.alias.get(raw_id, raw_id)

    def has_feature(self, canonical_id: int) -> bool:
        return canonical_id in self.features

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

        is_clean = True
        quality_score_fail = False
        quality_margin_fail = False
        quality_aspect_low_fail = False
        quality_aspect_high_fail = False
        if self.clean_score_threshold > 0.0 or self.clean_margin_ratio > 0.0:
            bw = float(box[2] - box[0])
            bh = float(box[3] - box[1])
            aspect = bh / bw if bw > 0 else 0.0
            margin_w = w * self.clean_margin_ratio
            margin_h = h * self.clean_margin_ratio
            quality_score_fail = score < self.clean_score_threshold
            quality_margin_fail = (
                float(box[0]) < margin_w
                or float(box[1]) < margin_h
                or float(box[2]) > w - margin_w
                or float(box[3]) > h - margin_h
            )
            quality_aspect_low_fail = aspect < self.clean_min_aspect
            quality_aspect_high_fail = aspect > self.clean_max_aspect
            if (
                quality_score_fail
                or quality_margin_fail
                or quality_aspect_low_fail
                or quality_aspect_high_fail
            ):
                is_clean = False

        current_sim_thresh = (
            self.sim_threshold if is_clean else self.strict_sim_threshold
        )

        if raw_id not in self.alias:
            self.stats["attempts"] += 1
            best_id = None
            best_joint = -1.0  # Sentinel for normalized joint score
            best_sim_raw = 0.0  # raw cosine of the current winner
            second_best_joint = -2.0  # runner-up joint score
            best_iou, best_center, best_maha = 0.0, 0.0, 0.0
            best_bypassed = False

            candidates_to_score = []
            for cid in self.features:
                age = frame_id - self.last_seen.get(cid, -(10**9))
                if cid in assigned:
                    self.stats["reject_assigned"] += 1
                    continue
                if age < self.min_lost_frames or age > self.ttl:
                    self.stats["reject_age"] += 1
                    if age < self.min_lost_frames:
                        self.stats["reject_age_too_fresh"] += 1
                        self.reject_age_fresh_ages.append(age)
                    else:
                        self.stats["reject_age_too_old"] += 1
                        self.reject_age_old_ages.append(age)
                    continue
                self.age_gate_pass_ages.append(age)
                last_box = self.last_boxes[cid]
                center_norm, iou = self._spatial_metrics(box, last_box, w, h)
                motion_box = None
                motion_center_norm = None
                motion_iou = None
                snapshot = self.motion.get(cid)
                if snapshot is not None:
                    motion_box = self._motion_box(snapshot)
                    if motion_box is not None:
                        motion_center_norm, motion_iou = self._spatial_metrics(
                            box, motion_box, w, h
                        )
                        self.reference_alignment_samples.append(
                            (
                                age,
                                float(iou),
                                float(center_norm),
                                float(motion_iou),
                                float(motion_center_norm),
                            )
                        )
                spatial_failed = center_norm > self.spatial_gate or iou < self.min_iou
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
                candidates_to_score.append(
                    (cid, age, iou, center_norm, maha, spatial_failed)
                )

            n_gate_passed = len(candidates_to_score)
            _use_legacy_joint = self.iou_weight > 0.0 or self.mahalanobis_weight > 0.0
            _use_unified_score = (
                self.w_sim_base > 0.0 or self.w_iou_base > 0.0 or self.w_maha_base > 0.0
            )

            if not _use_unified_score and not _use_legacy_joint:
                best_joint = current_sim_thresh
                second_best_joint = current_sim_thresh - 1.0

            # Batch similarity: one matmul + one D2H instead of N dot-products
            if candidates_to_score and self.buffer_size == 1:
                _cand_ids = [c[0] for c in candidates_to_score]
                _bank = torch.stack([self.features[cid] for cid in _cand_ids]).to(
                    self.device
                )
                _batch_sims = (_bank @ emb).tolist()  # single kernel + single D2H
                _sim_iter = iter(_batch_sims)

            for cid, age, iou, center_norm, maha, spatial_failed in candidates_to_score:
                if self.buffer_size > 1:
                    sim = self._buffer_sim(cid, emb)
                else:
                    sim = next(_sim_iter)

                # Hard appearance gate: raw cosine must still pass sim_threshold.
                if sim < current_sim_thresh:
                    self.stats["reject_similarity"] += 1
                    self.age_gate_pass_outcomes.append((age, "similarity"))
                    continue

                appearance_first_bypass = (
                    self.experimental_mode == "appearance_first"
                    and spatial_failed
                    and is_clean
                    and sim >= self.appearance_first_sim_threshold
                )
                if spatial_failed and not appearance_first_bypass:
                    self.stats["reject_spatial"] += 1
                    self.age_gate_pass_outcomes.append((age, "spatial"))
                    if center_norm > self.spatial_gate:
                        self.age_gate_pass_spatial_subreasons.append(
                            (age, "center_norm")
                        )
                    if iou < self.min_iou:
                        self.age_gate_pass_spatial_subreasons.append((age, "iou"))
                    continue

                maha_score = 0.0
                if self.mahalanobis_threshold > 0.0 and maha > 0.0:
                    maha_score = max(0.0, 1.0 - maha / self.mahalanobis_threshold)

                if appearance_first_bypass:
                    joint = sim
                elif _use_unified_score:
                    w_sim = self.w_sim_base
                    w_iou = self.w_iou_base
                    w_maha = self.w_maha_base

                    if n_gate_passed > 1:
                        ambiguity_factor = min(1.0, (n_gate_passed - 1) / 8.0)
                        w_sim += self.shift_ambiguity * ambiguity_factor
                        w_iou -= self.shift_ambiguity * ambiguity_factor

                    lost_factor = min(1.0, age / max(1, self.ttl))
                    w_sim += self.shift_lost_age * lost_factor
                    w_iou -= self.shift_lost_age * lost_factor

                    w_sim = max(0.0, w_sim)
                    w_iou = max(0.0, w_iou)
                    w_maha = max(0.0, w_maha)
                    sum_w = w_sim + w_iou + w_maha
                    if sum_w > 0:
                        w_sim /= sum_w
                        w_iou /= sum_w
                        w_maha /= sum_w

                    joint = w_sim * sim + w_iou * iou + w_maha * maha_score
                elif _use_legacy_joint:
                    joint = (
                        sim
                        + self.iou_weight * iou
                        + self.mahalanobis_weight * maha_score
                    )
                else:
                    joint = sim

                if joint > best_joint:
                    if best_id is not None:
                        second_best_joint = best_joint  # demote previous winner
                    best_id = cid
                    best_joint = joint
                    best_sim_raw = sim
                    best_iou, best_center, best_maha = iou, center_norm, maha
                    best_bypassed = appearance_first_bypass
                else:
                    if joint > second_best_joint:
                        second_best_joint = joint  # update runner-up
                    self.stats["reject_similarity"] += 1
                    self.age_gate_pass_outcomes.append((age, "similarity"))

            # Dynamic margin: base + crowd penalty + age penalty
            effective_margin = self.reciprocal_margin
            if best_id is not None:
                if self.dynamic_margin_crowd > 0.0 and n_gate_passed > 1:
                    crowd_factor = min(1.0, (n_gate_passed - 1) / 8.0)
                    effective_margin += self.dynamic_margin_crowd * crowd_factor
                if self.dynamic_margin_age > 0.0:
                    lost_frames = frame_id - self.last_seen.get(best_id, frame_id)
                    age_factor = min(1.0, lost_frames / max(1, self.ttl))
                    effective_margin += self.dynamic_margin_age * age_factor
                if best_bypassed:
                    effective_margin = max(
                        effective_margin, self.appearance_first_margin
                    )
            if best_id is not None and effective_margin > 0.0:
                if best_joint - second_best_joint < effective_margin:
                    self.stats["reject_margin"] += 1
                    best_age = frame_id - self.last_seen.get(best_id, frame_id)
                    self.age_gate_pass_outcomes.append((best_age, "margin"))
                    best_id = None
            if best_id is not None and self.biometric_threshold > 0.0:
                if self._bio_distance(raw_id, best_id) > self.biometric_threshold:
                    self.stats["reject_biometric"] += 1
                    best_id = None
            if best_id is not None:
                best_age = frame_id - self.last_seen.get(best_id, frame_id)
                self.age_gate_pass_outcomes.append((best_age, "accepted"))
                if best_bypassed:
                    self.stats["appearance_first_bypass"] += 1
                self.stats["accepted"] += 1
                self.accept_sims.append(best_sim_raw)
                self.accept_ious.append(best_iou)
                self.accept_center_dists.append(best_center)
                self.accept_mahas.append(best_maha)
                self.alias[raw_id] = best_id
            else:
                self.stats["new_ids"] += 1
                self.alias[raw_id] = raw_id

        canonical = self.alias[raw_id]
        if not is_clean:
            self.stats["reject_quality"] += 1
            if quality_score_fail:
                self.stats["reject_quality_score"] += 1
            if quality_margin_fail:
                self.stats["reject_quality_margin"] += 1
            if quality_aspect_low_fail:
                self.stats["reject_quality_aspect_low"] += 1
            if quality_aspect_high_fail:
                self.stats["reject_quality_aspect_high"] += 1
        else:
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
                if old is not None:
                    old = old.to(self.device)
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

    def resolve_many(
        self,
        candidates: List[
            Tuple[
                int,
                Optional[torch.Tensor],
                torch.Tensor,
                float,
            ]
        ],
        *,
        frame_id: int,
        w: int,
        h: int,
    ) -> List[int]:
        assigned: Set[int] = set()
        return [
            self.resolve(raw_id, embedding, box, score, frame_id, w, h, assigned)
            for raw_id, embedding, box, score in candidates
        ]

    def resolve_many_packed(
        self,
        raw_ids: List[int],
        embeddings: List[Optional[torch.Tensor]],
        boxes: List[torch.Tensor],
        scores: List[float],
        *,
        frame_id: int,
        w: int,
        h: int,
    ) -> List[int]:
        assigned: Set[int] = set()
        return [
            self.resolve(raw_id, embedding, box, score, frame_id, w, h, assigned)
            for raw_id, embedding, box, score in zip(raw_ids, embeddings, boxes, scores)
        ]

    def report(self) -> None:
        def _histogram(
            values: List[int], bins: List[tuple[str, int, int | None]]
        ) -> str:
            if not values:
                return "none"
            parts = []
            for label, lo, hi in bins:
                if hi is None:
                    count = sum(1 for v in values if v >= lo)
                else:
                    count = sum(1 for v in values if lo <= v <= hi)
                if count > 0:
                    parts.append(f"{label}={count}")
            return " ".join(parts) if parts else "none"

        def _outcome_histogram(
            values: List[tuple[int, str]],
            bins: List[tuple[str, int, int | None]],
            outcomes: List[str],
        ) -> str:
            if not values:
                return "none"
            parts = []
            for label, lo, hi in bins:
                bucket_parts = []
                for outcome in outcomes:
                    if hi is None:
                        count = sum(
                            1 for age, kind in values if age >= lo and kind == outcome
                        )
                    else:
                        count = sum(
                            1
                            for age, kind in values
                            if lo <= age <= hi and kind == outcome
                        )
                    if count > 0:
                        bucket_parts.append(f"{outcome}={count}")
                if bucket_parts:
                    parts.append(f"{label}[{' '.join(bucket_parts)}]")
            return " ".join(parts) if parts else "none"

        def _reference_alignment_report(
            values: List[tuple[int, float, float, float, float]],
            bins: List[tuple[str, int, int | None]],
        ) -> str:
            if not values:
                return "none"
            parts = []
            for label, lo, hi in bins:
                bucket = [
                    item
                    for item in values
                    if (item[0] >= lo if hi is None else lo <= item[0] <= hi)
                ]
                if not bucket:
                    continue
                n = len(bucket)
                mean_last_iou = sum(item[1] for item in bucket) / n
                mean_last_center = sum(item[2] for item in bucket) / n
                mean_motion_iou = sum(item[3] for item in bucket) / n
                mean_motion_center = sum(item[4] for item in bucket) / n
                motion_better_iou = sum(
                    1 for item in bucket if item[3] > item[1] + 1e-6
                )
                motion_better_center = sum(
                    1 for item in bucket if item[4] < item[2] - 1e-6
                )
                parts.append(
                    f"{label}[n={n} "
                    f"last_iou={mean_last_iou:.3f} motion_iou={mean_motion_iou:.3f} "
                    f"last_center={mean_last_center:.3f} motion_center={mean_motion_center:.3f} "
                    f"motion_better_iou={motion_better_iou} motion_better_center={motion_better_center}]"
                )
            return " ".join(parts) if parts else "none"

        print("🔁 Semantic Relink Report:")
        print(
            "  attempts={attempts} accepted={accepted} new_ids={new_ids} "
            "reject_age={reject_age} reject_assigned={reject_assigned} "
            "reject_spatial={reject_spatial} reject_mahalanobis={reject_mahalanobis} "
            "reject_similarity={reject_similarity} reject_margin={reject_margin} "
            "reject_biometric={reject_biometric}".format(**self.stats)
        )
        print(
            "  reject_age_fresh={reject_age_too_fresh} reject_age_old={reject_age_too_old} "
            "reject_quality_score={reject_quality_score} reject_quality_margin={reject_quality_margin} "
            "reject_quality_aspect_low={reject_quality_aspect_low} "
            "reject_quality_aspect_high={reject_quality_aspect_high} "
            "appearance_first_bypass={appearance_first_bypass}".format(**self.stats)
        )
        print(
            "  age_hist gate_pass=("
            + _histogram(
                self.age_gate_pass_ages,
                [
                    ("2-5", 2, 5),
                    ("6-15", 6, 15),
                    ("16-30", 16, 30),
                    ("31-45", 31, 45),
                ],
            )
            + ") reject_old=("
            + _histogram(
                self.reject_age_old_ages,
                [
                    ("46-60", 46, 60),
                    ("61-90", 61, 90),
                    ("91-180", 91, 180),
                    ("181+", 181, None),
                ],
            )
            + ")"
        )
        print(
            "  gate_pass_outcomes "
            + _outcome_histogram(
                self.age_gate_pass_outcomes,
                [
                    ("2-5", 2, 5),
                    ("6-15", 6, 15),
                    ("16-30", 16, 30),
                    ("31-45", 31, 45),
                ],
                ["accepted", "spatial", "similarity", "margin"],
            )
        )
        print(
            "  spatial_subreasons "
            + _outcome_histogram(
                self.age_gate_pass_spatial_subreasons,
                [
                    ("2-5", 2, 5),
                    ("6-15", 6, 15),
                    ("16-30", 16, 30),
                    ("31-45", 31, 45),
                ],
                ["center_norm", "iou"],
            )
        )
        print(
            "  reference_alignment "
            + _reference_alignment_report(
                self.reference_alignment_samples,
                [
                    ("2-5", 2, 5),
                    ("6-15", 6, 15),
                    ("16-30", 16, 30),
                    ("31-45", 31, 45),
                ],
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


class IdentityResolver:
    """Compose semantic relink + tracklet lifecycle merge into a single call.

    Owns no state; alias/features/stats remain with the constituent stages.
    Only instantiated when relinker is not None.
    """

    def __init__(self, relinker: Any, lifecycle_merger: Any) -> None:
        self._relinker = relinker
        self._lifecycle = lifecycle_merger

    def resolve_pass(
        self,
        local_ids: List[int],
        embeddings: List[Optional[torch.Tensor]],
        boxes: List[Any],
        scores: List[float],
        *,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> List[int]:
        if not local_ids:
            return []

        # Stage 1: semantic relink (relinker uses w=/h= kwargs)
        resolve_rk_packed = getattr(self._relinker, "resolve_many_packed", None)
        if callable(resolve_rk_packed):
            relinked_ids = resolve_rk_packed(
                local_ids,
                embeddings,
                boxes,
                scores,
                frame_id=frame_id,
                w=frame_w,
                h=frame_h,
            )
        else:
            resolve_rk_many = getattr(self._relinker, "resolve_many", None)
            if callable(resolve_rk_many):
                relinked_ids = resolve_rk_many(
                    list(zip(local_ids, embeddings, boxes, scores)),
                    frame_id=frame_id,
                    w=frame_w,
                    h=frame_h,
                )
            else:
                assigned: Set[int] = set()
                relinked_ids = [
                    self._relinker.resolve(
                        lid, emb, box, score, frame_id, frame_w, frame_h, assigned
                    )
                    for lid, emb, box, score in zip(
                        local_ids, embeddings, boxes, scores
                    )
                ]

        # Stage 2: lifecycle merge (lifecycle uses frame_w=/frame_h= kwargs)
        resolve_lc_packed = getattr(self._lifecycle, "resolve_many_packed", None)
        if callable(resolve_lc_packed):
            resolved_ids = resolve_lc_packed(
                relinked_ids,
                boxes,
                scores,
                embeddings,
                frame_id=frame_id,
                frame_w=frame_w,
                frame_h=frame_h,
            )
        else:
            resolve_lc_many = getattr(self._lifecycle, "resolve_many", None)
            if callable(resolve_lc_many):
                resolved_ids = resolve_lc_many(
                    list(zip(relinked_ids, boxes, scores, embeddings)),
                    frame_id=frame_id,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
            else:
                assigned_out: Set[int] = set()
                resolved_ids = [
                    self._lifecycle.resolve(
                        rid, box, score, frame_id, frame_w, frame_h, emb, assigned_out
                    )
                    for rid, box, score, emb in zip(
                        relinked_ids, boxes, scores, embeddings
                    )
                ]

        return list(resolved_ids)
