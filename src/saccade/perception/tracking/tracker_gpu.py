from collections import deque
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Any, cast, Optional, TypedDict

try:
    from saccade_tracking_ext import GPUByteTracker as CppGPUByteTracker, TrackResult
except ImportError:
    # Fallback for environments where the extension is not available or has library conflicts
    class TrackResult:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.track_id = -1
            self.tlbr = [0, 0, 0, 0]
            self.score = 0.0
            self.class_id = -1

    class CppGPUByteTracker:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_params(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_reid_params(self, *args: Any, **kwargs: Any) -> None:
            pass

        def update(self, *args: Any, **kwargs: Any) -> List[TrackResult]:
            return []

        def update_into(self, *args: Any, **kwargs: Any) -> None:
            return None

        def get_state_snapshots(self, *args: Any, **kwargs: Any) -> List[Any]:
            return []

        def get_motion_snapshots_for_track_ids(
            self, *args: Any, **kwargs: Any
        ) -> List[Any]:
            return []

        def get_tentative_candidates(self, *args: Any, **kwargs: Any) -> List[Any]:
            return []


@dataclass
class AppearanceSample:
    embedding: torch.Tensor  # L2-normalized float32, shape [D], stored on CPU
    det_score: float
    iou: float
    frame_id: int


@dataclass(frozen=True)
class ReIDFrameStats:
    new_tracks: int
    lost_tracks: int
    unstable_tracks: int


@dataclass(frozen=True)
class ReIDTrackObservation:
    box: tuple[float, float, float, float]
    det_score: float


class GPUTrackResultBuffers(TypedDict):
    boxes: torch.Tensor
    scores: torch.Tensor
    ids: torch.Tensor
    classes: torch.Tensor
    det_idx: torch.Tensor
    count: torch.Tensor


class DynamicReIDController:
    """5-frame bbox-history trigger for event-driven ReID refresh."""

    def __init__(
        self,
        history_size: int = 5,
        mode: str = "event_any",
        unstable_iou: float = 0.50,
        unstable_center_shift: float = 0.30,
        crowd_threshold: int = 8,
        long_memory_decay: float = 0.80,
        long_memory_trigger: float = 1.25,
        score_decay: float = 0.80,
        score_threshold: float = 2.0,
        score_threshold_low: Optional[float] = None,
        weight_new: float = 1.0,
        weight_lost: float = 1.4,
        weight_geom: float = 0.5,
        weight_conf: float = 0.5,
        birth_death_boost: float = 1.0,
        birth_death_lost_min: float = 0.0,
        lost_age_cap: int = 30,
        unstable_shift_weight: float = 1.0,
        unstable_iou_weight: float = 1.0,
        conf_jitter_gate: float = 0.10,
        trigger_persist_frames: int = 1,
        cooldown_frames: int = 0,
    ) -> None:
        self.history_size = max(2, history_size)
        self.mode = mode
        self.unstable_iou = unstable_iou
        self.unstable_center_shift = unstable_center_shift
        self.crowd_threshold = crowd_threshold
        self.long_memory_decay = min(max(long_memory_decay, 0.0), 0.99)
        self.long_memory_trigger = max(long_memory_trigger, 0.0)
        self.score_decay = min(max(score_decay, 0.0), 0.99)
        self.score_threshold = max(score_threshold, 0.0)
        self.score_threshold_low = (
            max(score_threshold_low, 0.0)
            if score_threshold_low is not None
            else self.score_threshold
        )
        self.weight_new = weight_new
        self.weight_lost = weight_lost
        self.weight_geom = weight_geom
        self.weight_conf = weight_conf
        self.birth_death_boost = max(birth_death_boost, 0.0)
        self.birth_death_lost_min = max(birth_death_lost_min, 0.0)
        self.lost_age_cap = max(lost_age_cap, 1)
        self.unstable_shift_weight = max(unstable_shift_weight, 0.0)
        self.unstable_iou_weight = max(unstable_iou_weight, 0.0)
        self.conf_jitter_gate = max(conf_jitter_gate, 0.0)
        self.trigger_persist_frames = max(trigger_persist_frames, 1)
        self.cooldown_frames = max(cooldown_frames, 0)
        self._track_history: deque[dict[int, ReIDTrackObservation]] = deque(
            maxlen=self.history_size
        )
        self._frame_stats: deque[ReIDFrameStats] = deque(maxlen=self.history_size)
        self._event_memory = 0.0
        self._track_ages: dict[int, int] = {}
        self._track_score_ema: dict[int, float] = {}
        self._score_new = 0.0
        self._score_lost = 0.0
        self._score_geom = 0.0
        self._score_conf = 0.0
        self._last_birth_death_boost = 0.0
        self._persist_counter = 0
        self._cooldown_remaining = 0

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
        return float((((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale)

    def observe(
        self,
        tracks: dict[int, ReIDTrackObservation],
        gmc: Optional[torch.Tensor] = None,
    ) -> None:
        prev = self._track_history[-1] if self._track_history else {}
        curr_ids = set(tracks)
        prev_ids = set(prev)
        shared_ids = curr_ids & prev_ids

        # Resolve boxes for geometry comparison.
        curr_geom = {tid: tracks[tid].box for tid in shared_ids}
        prev_geom = {tid: prev[tid].box for tid in shared_ids}

        unstable = 0
        unstable_signal = 0.0
        conf_signal = 0.0

        # Prepare GMC affine parameters if provided
        # gmc is expected to be a 2x3 matrix: [[H00, H01, H02], [H10, H11, H12]]
        h00, h01, h02, h10, h11, h12 = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
        if gmc is not None:
            # Check shape: could be (2, 3) or (6,) or (1, 6)
            gmc_cpu = gmc.detach().cpu().view(-1).tolist()
            if len(gmc_cpu) >= 6:
                h00, h01, h02, h10, h11, h12 = gmc_cpu[:6]

        for tid in shared_ids:
            curr_box = curr_geom.get(tid)
            prev_box = prev_geom.get(tid)
            if curr_box is None or prev_box is None:
                continue

            if gmc is not None:
                # Apply affine transform to prev_box
                # A bounding box is defined by top-left (x1, y1) and bottom-right (x2, y2).
                # We transform all 4 corners to find the new bounding box.
                x1, y1, x2, y2 = prev_box
                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                tx = [h00 * x + h01 * y + h02 for x, y in corners]
                ty = [h10 * x + h11 * y + h12 for x, y in corners]

                # Invert the warp to predict where the old box would be in the current frame.
                # Wait, the GMC matrix in our tracker pipeline typically transforms CURRENT to PREVIOUS.
                # In tracker_gpu.cu / gmc_kernel, it's used to update the predicted state.
                # Actually, in tracker.py, cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                # returns M that maps prev -> curr.
                # So if gmc maps prev -> curr, we just apply it directly to prev_box.
                prev_box = (min(tx), min(ty), max(tx), max(ty))

            iou = self._iou(curr_box, prev_box)
            shift_ratio = self._center_shift_ratio(curr_box, prev_box)
            shift_term = max(0.0, shift_ratio - self.unstable_center_shift)
            iou_term = max(0.0, self.unstable_iou - iou)
            instability = (
                self.unstable_shift_weight * shift_term
                + self.unstable_iou_weight * iou_term
            )
            if instability > 0.0:
                unstable += 1
                unstable_signal += instability
            prev_score_ema = self._track_score_ema.get(tid, prev[tid].det_score)
            conf_signal += max(
                0.0, abs(tracks[tid].det_score - prev_score_ema) - self.conf_jitter_gate
            )

        new_signal = sum(tracks[tid].det_score for tid in curr_ids - prev_ids)
        lost_signal = 0.0
        for tid in prev_ids - curr_ids:
            lost_signal += min(
                1.0, self._track_ages.get(tid, 1) / float(self.lost_age_cap)
            )

        matched_count = max(len(shared_ids), 1)
        unstable_signal /= matched_count
        conf_signal /= matched_count

        self._track_history.append(dict(tracks))
        stats = ReIDFrameStats(
            new_tracks=len(curr_ids - prev_ids),
            lost_tracks=len(prev_ids - curr_ids),
            unstable_tracks=unstable,
        )
        self._frame_stats.append(stats)
        event_strength = float(stats.new_tracks + stats.lost_tracks) + 0.5 * float(
            stats.unstable_tracks
        )
        self._event_memory = (
            self.long_memory_decay * self._event_memory + event_strength
        )
        self._score_new = self.score_decay * self._score_new + new_signal
        self._score_lost = self.score_decay * self._score_lost + lost_signal
        self._score_geom = self.score_decay * self._score_geom + unstable_signal
        self._score_conf = self.score_decay * self._score_conf + conf_signal
        self._last_birth_death_boost = (
            self.birth_death_boost
            if new_signal > 0.0
            and lost_signal > 0.0
            and lost_signal >= self.birth_death_lost_min
            else 0.0
        )
        next_ages: dict[int, int] = {}
        next_score_ema: dict[int, float] = {}
        for tid in curr_ids:
            next_ages[tid] = self._track_ages.get(tid, 0) + 1 if tid in prev_ids else 1
            prev_score_ema = self._track_score_ema.get(tid, tracks[tid].det_score)
            next_score_ema[tid] = (
                self.score_decay * prev_score_ema
                + (1.0 - self.score_decay) * tracks[tid].det_score
                if tid in prev_ids
                else tracks[tid].det_score
            )
        self._track_ages = next_ages
        self._track_score_ema = next_score_ema

    def should_reid(self, det_count: int) -> bool:
        if det_count <= 0 or not self._track_history:
            return False
        active_count = len(self._track_history[-1])
        if active_count <= 0:
            return False
        if len(self._frame_stats) < 2:
            return False

        if det_count >= active_count + 2:
            return True

        recent_new = sum(stats.new_tracks for stats in self._frame_stats)
        recent_lost = sum(stats.lost_tracks for stats in self._frame_stats)
        recent_unstable = sum(stats.unstable_tracks for stats in self._frame_stats)
        latest = self._frame_stats[-1]
        prev = self._frame_stats[-2]
        latest_events = latest.new_tracks + latest.lost_tracks
        persistent_events = (
            latest_events > 0 and (prev.new_tracks + prev.lost_tracks) > 0
        )
        unstable_now = latest.unstable_tracks
        unstable_persistent = unstable_now > 0 and prev.unstable_tracks > 0

        if self.mode == "count_jump":
            return det_count >= active_count + 2

        if self.mode == "event_memory":
            if self._event_memory >= self.long_memory_trigger:
                return True
            if det_count >= active_count + 2:
                return True
            return False

        if self.mode == "score_ema":
            trigger_score = (
                self.weight_new * self._score_new
                + self.weight_lost * self._score_lost
                + self.weight_geom * self._score_geom
                + self.weight_conf * self._score_conf
                + self._last_birth_death_boost
            )
            return self._persist(trigger_score)

        if self.mode == "score_ema_geom":
            trigger_score = (
                self.weight_new * self._score_new
                + self.weight_lost * self._score_lost
                + self.weight_geom * self._score_geom
                + self._last_birth_death_boost
            )
            return self._persist(trigger_score)

        if self.mode == "score_ema_conf":
            trigger_score = (
                self.weight_new * self._score_new
                + self.weight_lost * self._score_lost
                + self.weight_conf * self._score_conf
                + self._last_birth_death_boost
            )
            return self._persist(trigger_score)

        if self.mode == "event_strict":
            if persistent_events:
                return True
            if unstable_now >= max(2, min(active_count, 3)) and unstable_persistent:
                return True
            if self._event_memory >= self.long_memory_trigger and unstable_now > 0:
                return True
            if (
                active_count >= self.crowd_threshold
                and det_count >= active_count + 1
                and unstable_persistent
            ):
                return True
            return False

        if self.mode == "event_persist":
            if persistent_events:
                return True
            if unstable_now >= max(2, min(active_count, 4)) and unstable_persistent:
                return True
            if self._event_memory >= self.long_memory_trigger and latest_events > 0:
                return True
            if (
                active_count >= self.crowd_threshold
                and det_count >= active_count
                and unstable_persistent
            ):
                return True
            return False

        if recent_new > 0 or recent_lost > 0:
            return True

        if recent_unstable >= max(2, min(active_count, 4)):
            return True

        if (
            active_count >= self.crowd_threshold
            and det_count >= active_count
            and recent_unstable > 0
        ):
            return True

        return False

    def _persist(self, trigger_score: float) -> bool:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._persist_counter = 0
            return False

        if self._persist_counter > 0:
            # Already accumulating: use lower threshold to prevent oscillation
            triggered = trigger_score >= self.score_threshold_low
        else:
            # Not accumulating: must cross the high threshold to start
            triggered = trigger_score >= self.score_threshold

        if triggered:
            self._persist_counter += 1
        else:
            self._persist_counter = 0

        if self._persist_counter >= self.trigger_persist_frames:
            self._persist_counter = 0
            self._cooldown_remaining = self.cooldown_frames
            return True
        return False


class TrackAppearanceBank:
    """Per-track Top-K clean appearance sample bank (Phase 1)."""

    MIN_SCORE: float = 0.45
    MIN_IOU: float = 0.35

    def __init__(
        self,
        k: int = 5,
        *,
        min_score: float = 0.45,
        min_iou: float = 0.35,
        consistency_threshold: float = 0.82,
    ) -> None:
        self.k = max(1, k)
        self.min_score = float(min_score)
        self.min_iou = float(min_iou)
        self.consistency_threshold = float(consistency_threshold)
        self._banks: dict[int, list[AppearanceSample]] = {}
        self._representatives: dict[int, torch.Tensor] = {}
        self._consistency: dict[int, float] = {}
        self._clean_ids: set[int] = set()

    def update(
        self,
        track_id: int,
        embedding: torch.Tensor,
        *,
        det_score: float,
        iou: float,
        frame_id: int,
        geometry_clean: bool = True,
        suspect_box: bool = False,
    ) -> None:
        if (
            det_score < self.min_score
            or iou < self.min_iou
            or not geometry_clean
            or suspect_box
        ):
            return
        emb = F.normalize(embedding.detach().to(device="cpu", dtype=torch.float32), dim=0)
        bank = self._banks.setdefault(track_id, [])
        bank.append(AppearanceSample(emb, det_score, iou, frame_id))
        # Rank: 0.5*det_score + 0.3*iou, with frame_id as recency tiebreaker
        bank.sort(
            key=lambda s: (0.5 * s.det_score + 0.3 * s.iou, s.frame_id), reverse=True
        )
        del bank[self.k :]
        self._refresh_track(track_id)

    def update_many(
        self,
        updates: list[
            tuple[int, torch.Tensor, float, float, int, bool, bool]
        ],
    ) -> None:
        touched_track_ids: set[int] = set()
        for (
            track_id,
            embedding,
            det_score,
            iou,
            frame_id,
            geometry_clean,
            suspect_box,
        ) in updates:
            if (
                det_score < self.min_score
                or iou < self.min_iou
                or not geometry_clean
                or suspect_box
            ):
                continue
            emb = F.normalize(embedding.detach().to(device="cpu", dtype=torch.float32), dim=0)
            bank = self._banks.setdefault(track_id, [])
            bank.append(AppearanceSample(emb, det_score, iou, frame_id))
            touched_track_ids.add(track_id)

        for track_id in touched_track_ids:
            bank = self._banks.get(track_id, [])
            bank.sort(
                key=lambda s: (0.5 * s.det_score + 0.3 * s.iou, s.frame_id),
                reverse=True,
            )
            del bank[self.k :]
            self._refresh_track(track_id)

    def has_clean_embedding(self, track_id: int) -> bool:
        return bool(self._banks.get(track_id))

    def consistency(self, track_id: int) -> float:
        return self._consistency.get(track_id, 1.0)

    def is_consistent(self, track_id: int) -> bool:
        return self.consistency(track_id) >= self.consistency_threshold

    def representative(self, track_id: int) -> Optional[torch.Tensor]:
        return self._representatives.get(track_id)

    def representatives(self) -> dict[int, torch.Tensor]:
        """Returns {track_id: representative_embedding} for all tracks with a clean bank."""
        return dict(self._representatives)

    def clean_ids(self) -> set[int]:
        """Returns track_ids whose banks are non-empty and internally consistent."""
        return set(self._clean_ids)

    def prune(self, active_ids: set[int]) -> None:
        for tid in [t for t in list(self._banks) if t not in active_ids]:
            del self._banks[tid]
            self._representatives.pop(tid, None)
            self._consistency.pop(tid, None)
            self._clean_ids.discard(tid)

    def _refresh_track(self, track_id: int) -> None:
        bank = self._banks.get(track_id, [])
        if not bank:
            self._representatives.pop(track_id, None)
            self._consistency.pop(track_id, None)
            self._clean_ids.discard(track_id)
            return

        embs = torch.stack([sample.embedding for sample in bank])
        representative = F.normalize(embs.mean(dim=0), dim=0)
        if len(bank) < 2:
            consistency = 1.0
        else:
            sims = embs @ embs.T
            n = len(bank)
            consistency = float((sims.sum() - n) / max(n * (n - 1), 1))

        self._representatives[track_id] = representative
        self._consistency[track_id] = consistency
        if consistency >= self.consistency_threshold:
            self._clean_ids.add(track_id)
        else:
            self._clean_ids.discard(track_id)


def need_reid_frame(
    prev_track_ids: set[int], det_count_or_scores: int | torch.Tensor
) -> bool:
    """Return True when embedding extraction is warranted this frame.

    This hot-path heuristic must stay CPU-cheap. Avoid GPU reductions here:
    shape/numel metadata is fine, but sum/max/min on CUDA tensors will force sync.
    """
    if isinstance(det_count_or_scores, int):
        n_dets = det_count_or_scores
    else:
        n_dets = int(det_count_or_scores.shape[0])

    if not prev_track_ids or n_dets <= 0:
        return False

    n_prev = len(prev_track_ids)

    # A clear count increase usually means new entrants or a split after occlusion.
    if n_dets >= n_prev + 2:
        return True

    # In sustained crowded scenes, allow periodic ReID refresh when counts stay high.
    if n_prev >= 8 and n_dets >= n_prev:
        return True

    return False


class GPUByteTracker:
    """
    Saccade GPU 追蹤器封裝。
    直接對接 C++ / CUDA 實作，確保 Zero-Copy。
    """

    def __init__(self, max_objects: int = 2048, embedding_dim: int = 768) -> None:
        self.max_objects = max_objects
        self.embedding_dim = embedding_dim
        self.tracker = CppGPUByteTracker(max_objects, embedding_dim)

    def set_params(
        self,
        track_thresh: float = 0.1,
        high_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        mid_thresh: float = 0.40,
        confirm_streak: int = 3,
        confirm_score_thresh: float = 0.50,
        adaptive_confirmation: bool = False,
        new_track_thresh: float = -1.0,
        nsa_kalman: bool = False,
    ) -> None:
        """調整追蹤器門檻與參數。"""
        self.tracker.set_params(
            track_thresh,
            high_thresh,
            match_thresh,
            track_buffer,
            mid_thresh,
            confirm_streak,
            confirm_score_thresh,
            adaptive_confirmation,
            new_track_thresh,
            nsa_kalman,
        )

    def set_reid_params(
        self,
        cos_threshold: float = 0.90,
        iou_low: float = 0.30,
        iou_high: float = 0.60,
        weight: float = 0.40,
    ) -> None:
        """調整 C++ ReID fusion 門檻。"""
        set_reid_params = getattr(self.tracker, "set_reid_params", None)
        if set_reid_params is not None:
            set_reid_params(cos_threshold, iou_low, iou_high, weight)

    def update_reference_features(
        self,
        track_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        """更新追蹤器的參考特徵（用於 ReID）。"""
        num = track_ids.size(0)
        if num == 0:
            return

        ids_contig = track_ids.to(torch.int32).contiguous()
        features_contig = features.to(torch.float32).contiguous()
        stream = torch.cuda.current_stream().cuda_stream

        self.tracker.update_reference_features(
            ids_contig.data_ptr(), features_contig.data_ptr(), num, stream
        )

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        gmc: Optional[torch.Tensor] = None,
        light_factor: float = 0.0,
        mid_thresh_scale: float = 1.0,
    ) -> List[TrackResult]:
        """
        更新追蹤器狀態。
        """
        num_dets = boxes.size(0)

        # 💡 重要：必須保留這些 Tensor 的引用，防止在 data_ptr() 使用期間 be GC
        boxes_contig = boxes.to(torch.float32).contiguous()
        scores_contig = scores.to(torch.float32).contiguous()
        classes_contig = classes.to(torch.int32).contiguous()

        embed_contig = None
        embed_ptr = 0
        if embeddings is not None:
            embed_contig = embeddings.to(torch.float32).contiguous()
            embed_ptr = embed_contig.data_ptr()

        gmc_contig = None
        gmc_ptr = 0
        if gmc is not None:
            gmc_contig = gmc.to(torch.float32).contiguous()
            gmc_ptr = gmc_contig.data_ptr()

        stream = torch.cuda.current_stream().cuda_stream

        return cast(
            List[TrackResult],
            self.tracker.update(
                boxes_contig.data_ptr(),
                scores_contig.data_ptr(),
                classes_contig.data_ptr(),
                num_dets,
                stream,
                embed_ptr,
                gmc_ptr,
                light_factor,
                mid_thresh_scale,
            ),
        )

    def allocate_result_buffers(
        self,
        *,
        device: str | torch.device = "cuda",
    ) -> GPUTrackResultBuffers:
        return {
            "boxes": torch.empty(
                (self.max_objects, 4), device=device, dtype=torch.float32
            ),
            "scores": torch.empty((self.max_objects,), device=device, dtype=torch.float32),
            "ids": torch.empty((self.max_objects,), device=device, dtype=torch.int32),
            "classes": torch.empty((self.max_objects,), device=device, dtype=torch.int32),
            "det_idx": torch.empty((self.max_objects,), device=device, dtype=torch.int32),
            "count": torch.empty((), device=device, dtype=torch.int32),
        }

    def update_into(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        result_buffers: GPUTrackResultBuffers,
        embeddings: Optional[torch.Tensor] = None,
        gmc: Optional[torch.Tensor] = None,
        light_factor: float = 0.0,
        mid_thresh_scale: float = 1.0,
    ) -> GPUTrackResultBuffers:
        num_dets = boxes.size(0)

        boxes_contig = boxes.to(torch.float32).contiguous()
        scores_contig = scores.to(torch.float32).contiguous()
        classes_contig = classes.to(torch.int32).contiguous()

        embed_ptr = 0
        if embeddings is not None:
            embeddings = embeddings.to(torch.float32).contiguous()
            embed_ptr = embeddings.data_ptr()

        gmc_ptr = 0
        if gmc is not None:
            gmc = gmc.to(torch.float32).contiguous()
            gmc_ptr = gmc.data_ptr()

        stream = torch.cuda.current_stream().cuda_stream
        self.tracker.update_into(
            boxes_contig.data_ptr(),
            scores_contig.data_ptr(),
            classes_contig.data_ptr(),
            num_dets,
            stream,
            result_buffers["boxes"].data_ptr(),
            result_buffers["scores"].data_ptr(),
            result_buffers["ids"].data_ptr(),
            result_buffers["classes"].data_ptr(),
            result_buffers["det_idx"].data_ptr(),
            result_buffers["count"].data_ptr(),
            embed_ptr,
            gmc_ptr,
            light_factor,
            mid_thresh_scale,
        )
        return result_buffers

    def get_state_snapshots(self) -> List[Any]:
        """Return active Kalman state/covariance snapshots from the C++ tracker."""
        stream = torch.cuda.current_stream().cuda_stream
        return cast(List[Any], self.tracker.get_state_snapshots(stream))

    def get_motion_snapshots_for_track_ids(self, track_ids: list[int]) -> List[Any]:
        """Return Kalman motion snapshots only for the requested track IDs."""
        if not track_ids:
            return []
        get_motion = getattr(self.tracker, "get_motion_snapshots_for_track_ids", None)
        if get_motion is None:
            return self.get_state_snapshots()
        stream = torch.cuda.current_stream().cuda_stream
        return cast(List[Any], get_motion(track_ids, stream))

    def get_tentative_candidates(self) -> List[Any]:
        """Return active tentative tracks for lazy ReID arbitration/profiling."""
        stream = torch.cuda.current_stream().cuda_stream
        get_candidates = getattr(self.tracker, "get_tentative_candidates", None)
        if get_candidates is None:
            return []
        return cast(List[Any], get_candidates(stream))

    def set_clean_embedding_flags(
        self,
        track_ids: torch.Tensor,
        flags: torch.Tensor,
    ) -> None:
        """Sync per-track clean-embedding flags from Python bank to the CUDA tracker."""
        set_flags = getattr(self.tracker, "set_clean_embedding_flags", None)
        if set_flags is None or track_ids.numel() == 0:
            return
        ids_contig = track_ids.to(torch.int32).cuda().contiguous()
        flags_contig = flags.to(torch.bool).cuda().contiguous()
        stream = torch.cuda.current_stream().cuda_stream
        set_flags(
            ids_contig.data_ptr(),
            flags_contig.data_ptr(),
            int(track_ids.numel()),
            stream,
        )

    def set_reference_features_from_bank(
        self,
        bank_reps: dict[int, torch.Tensor],
    ) -> None:
        """Push bank representative embeddings into C++ d_features_ before association."""
        if not bank_reps:
            return
        all_ids = list(bank_reps.keys())
        stacked = torch.stack([bank_reps[tid] for tid in all_ids])  # [n, D]
        n = len(all_ids)
        device = torch.device("cuda")
        ids_gpu = torch.tensor(all_ids, dtype=torch.int32, device=device).contiguous()
        feats_gpu = stacked.to(device=device, dtype=torch.float32).contiguous()
        stream = torch.cuda.current_stream().cuda_stream
        self.tracker.update_reference_features(
            ids_gpu.data_ptr(), feats_gpu.data_ptr(), n, stream
        )
