import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Any, cast, Optional, TypedDict, Callable

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

        def set_homography(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_unified_score_params(self, *args: Any, **kwargs: Any) -> None:
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

        def push_keypoints(self, *args: Any, **kwargs: Any) -> None:
            return None

        def get_biometric(self, *args: Any, **kwargs: Any) -> Any:
            class _Profile:
                valid = False
                r_leg = float("nan")
                r_shoulder = float("nan")
                r_head = float("nan")

            return _Profile()


@dataclass
class AppearanceSample:
    embedding: torch.Tensor  # L2-normalized float32, shape [D], stored on CPU
    det_score: float
    iou: float
    frame_id: int
    aspect_ratio: float = 0.0  # h/w of detection box; 0.0 = unknown
    quality_score: float = 0.0  # A6 composite score; 0.0 → fall back to legacy formula
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


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


class TrackAppearanceBank:
    """Per-track Top-K clean appearance sample bank with EMA smoothing (Phase 1).

    Stores up to K quality-sorted samples per track for consistency checking.
    The representative used for matching is an EMA-smoothed embedding (like
    Ultralytics BOTSORT, alpha=0.9): each new matched detection is blended
    with the exponential moving average to reduce single-frame noise.
    """

    MIN_SCORE: float = 0.45
    MIN_IOU: float = 0.35

    def __init__(
        self,
        k: int = 5,
        *,
        min_score: float = 0.45,
        min_iou: float = 0.35,
        consistency_threshold: float = 0.82,
        high_quality_min_score: float = 0.70,
        min_aspect: float = 1.2,
        max_aspect: float = 4.5,
        bank_weighted_mean: bool = False,
        ema_alpha: float = 0.8,
        exp_velocity_aligned_bank: bool = False,
    ) -> None:
        self.k = max(1, k)
        self.min_score = float(min_score)
        self.min_iou = float(min_iou)
        self.consistency_threshold = float(consistency_threshold)
        self.high_quality_min_score = float(high_quality_min_score)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self._bank_weighted_mean = bank_weighted_mean
        self.ema_alpha = float(ema_alpha)
        self._use_ema = ema_alpha > 0.0
        self.exp_velocity_aligned_bank = exp_velocity_aligned_bank
        self._banks: dict[int, list[AppearanceSample]] = {}
        self._representatives: dict[int, torch.Tensor] = {}
        self._consistency: dict[int, float] = {}
        self._clean_ids: set[int] = set()
        self._high_quality_reps: dict[int, torch.Tensor] = {}
        self._ema_reps: dict[int, torch.Tensor] = {}

    @staticmethod
    def _rank_key(s: "AppearanceSample") -> tuple[float, int]:
        # A6: if composite quality_score is set, use it; otherwise legacy formula
        q = (
            s.quality_score
            if s.quality_score > 0.0
            else (0.5 * s.det_score + 0.3 * s.iou)
        )
        return (q, s.frame_id)

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
        aspect_ratio: float = 0.0,
        quality_score: float = 0.0,
        box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        if (
            det_score < self.min_score
            or iou < self.min_iou
            or not geometry_clean
            or suspect_box
        ):
            return

        if self.exp_velocity_aligned_bank and track_id in self._banks:
            samples = self._banks[track_id]
            chron_samples = sorted(samples, key=lambda s: s.frame_id)
            if len(chron_samples) >= 2:
                s_prev2 = chron_samples[-2]
                s_prev1 = chron_samples[-1]
                dt1 = s_prev1.frame_id - s_prev2.frame_id
                dt2 = frame_id - s_prev1.frame_id
                if dt1 > 0 and dt2 > 0:
                    if (
                        s_prev2.box != (0.0, 0.0, 0.0, 0.0)
                        and s_prev1.box != (0.0, 0.0, 0.0, 0.0)
                        and box != (0.0, 0.0, 0.0, 0.0)
                    ):
                        c_prev2 = (
                            (s_prev2.box[0] + s_prev2.box[2]) * 0.5,
                            (s_prev2.box[1] + s_prev2.box[3]) * 0.5,
                        )
                        c_prev1 = (
                            (s_prev1.box[0] + s_prev1.box[2]) * 0.5,
                            (s_prev1.box[1] + s_prev1.box[3]) * 0.5,
                        )
                        c_curr = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

                        vx_prev = (c_prev1[0] - c_prev2[0]) / dt1
                        vy_prev = (c_prev1[1] - c_prev2[1]) / dt1
                        vx_curr = (c_curr[0] - c_prev1[0]) / dt2
                        vy_curr = (c_curr[1] - c_prev1[1]) / dt2

                        mag_prev = (vx_prev**2 + vy_prev**2) ** 0.5
                        mag_curr = (vx_curr**2 + vy_curr**2) ** 0.5

                        if mag_prev > 1.0 and mag_curr > 1.0:
                            cos_theta = (vx_curr * vx_prev + vy_curr * vy_prev) / (
                                mag_curr * mag_prev + 1e-6
                            )
                            if cos_theta < 0.5:
                                return  # Reject update

        emb = F.normalize(
            embedding.detach().to(device="cpu", dtype=torch.float32),
            dim=0,  # saccade-allow-cpu
        )
        if self._use_ema:
            if track_id in self._ema_reps:
                ema_unnorm = (
                    self.ema_alpha * self._ema_reps[track_id]
                    + (1.0 - self.ema_alpha) * emb
                )
            else:
                ema_unnorm = emb
            self._ema_reps[track_id] = F.normalize(ema_unnorm, dim=0)
        bank = self._banks.setdefault(track_id, [])
        bank.append(
            AppearanceSample(
                emb, det_score, iou, frame_id, aspect_ratio, quality_score, box
            )
        )
        bank.sort(key=self._rank_key, reverse=True)
        del bank[self.k :]
        self._refresh_track(track_id)

    def update_many(
        self,
        updates: list[Any],
    ) -> None:
        touched_track_ids: set[int] = set()
        for update in updates:
            if len(update) == 9:
                (
                    track_id,
                    embedding,
                    det_score,
                    iou,
                    frame_id,
                    geometry_clean,
                    suspect_box,
                    aspect_ratio,
                    bank_quality_score,
                ) = update
                box = (0.0, 0.0, 0.0, 0.0)
            else:
                (
                    track_id,
                    embedding,
                    det_score,
                    iou,
                    frame_id,
                    geometry_clean,
                    suspect_box,
                    aspect_ratio,
                    bank_quality_score,
                    box,
                ) = update
            if (
                det_score < self.min_score
                or iou < self.min_iou
                or not geometry_clean
                or suspect_box
            ):
                continue

            if self.exp_velocity_aligned_bank and track_id in self._banks:
                samples = self._banks[track_id]
                chron_samples = sorted(samples, key=lambda s: s.frame_id)
                if len(chron_samples) >= 2:
                    s_prev2 = chron_samples[-2]
                    s_prev1 = chron_samples[-1]
                    dt1 = s_prev1.frame_id - s_prev2.frame_id
                    dt2 = frame_id - s_prev1.frame_id
                    if dt1 > 0 and dt2 > 0:
                        if (
                            s_prev2.box != (0.0, 0.0, 0.0, 0.0)
                            and s_prev1.box != (0.0, 0.0, 0.0, 0.0)
                            and box != (0.0, 0.0, 0.0, 0.0)
                        ):
                            c_prev2 = (
                                (s_prev2.box[0] + s_prev2.box[2]) * 0.5,
                                (s_prev2.box[1] + s_prev2.box[3]) * 0.5,
                            )
                            c_prev1 = (
                                (s_prev1.box[0] + s_prev1.box[2]) * 0.5,
                                (s_prev1.box[1] + s_prev1.box[3]) * 0.5,
                            )
                            c_curr = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

                            vx_prev = (c_prev1[0] - c_prev2[0]) / dt1
                            vy_prev = (c_prev1[1] - c_prev2[1]) / dt1
                            vx_curr = (c_curr[0] - c_prev1[0]) / dt2
                            vy_curr = (c_curr[1] - c_prev1[1]) / dt2

                            mag_prev = (vx_prev**2 + vy_prev**2) ** 0.5
                            mag_curr = (vx_curr**2 + vy_curr**2) ** 0.5

                            if mag_prev > 1.0 and mag_curr > 1.0:
                                cos_theta = (vx_curr * vx_prev + vy_curr * vy_prev) / (
                                    mag_curr * mag_prev + 1e-6
                                )
                                if cos_theta < 0.5:
                                    continue  # Reject update

            emb = F.normalize(
                embedding.detach().to(device="cpu", dtype=torch.float32),
                dim=0,  # saccade-allow-cpu
            )
            if self._use_ema:
                if track_id in self._ema_reps:
                    ema_unnorm = (
                        self.ema_alpha * self._ema_reps[track_id]
                        + (1.0 - self.ema_alpha) * emb
                    )
                else:
                    ema_unnorm = emb
                self._ema_reps[track_id] = F.normalize(ema_unnorm, dim=0)
            bank = self._banks.setdefault(track_id, [])
            bank.append(
                AppearanceSample(
                    emb,
                    det_score,
                    iou,
                    frame_id,
                    aspect_ratio,
                    bank_quality_score,
                    box,
                )
            )
            touched_track_ids.add(track_id)

        for track_id in touched_track_ids:
            bank = self._banks.get(track_id, [])
            bank.sort(key=self._rank_key, reverse=True)
            del bank[self.k :]
            self._refresh_track(track_id)

    def has_clean_embedding(self, track_id: int) -> bool:
        return bool(self._banks.get(track_id))

    def is_high_quality(self, track_id: int) -> bool:
        """True if at least one bank sample passes the high-quality score/aspect gate."""
        return track_id in self._high_quality_reps

    def high_quality_representative(self, track_id: int) -> Optional[torch.Tensor]:
        """Mean embedding of all high-quality samples for this track, or None."""
        return self._high_quality_reps.get(track_id)

    def consistency(self, track_id: int) -> float:
        return self._consistency.get(track_id, 1.0)

    def is_consistent(self, track_id: int) -> bool:
        return self.consistency(track_id) >= self.consistency_threshold

    def representative(self, track_id: int) -> Optional[torch.Tensor]:
        if self._use_ema:
            return self._ema_reps.get(track_id)
        return self._representatives.get(track_id)

    def representatives(self) -> dict[int, torch.Tensor]:
        """Returns {track_id: representative_embedding} for all tracks with a clean bank."""
        if self._use_ema:
            return dict(self._ema_reps)
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
            self._high_quality_reps.pop(tid, None)
            self._ema_reps.pop(tid, None)

    def _refresh_track(self, track_id: int) -> None:
        bank = self._banks.get(track_id, [])
        if not bank:
            self._representatives.pop(track_id, None)
            self._consistency.pop(track_id, None)
            self._clean_ids.discard(track_id)
            self._high_quality_reps.pop(track_id, None)
            self._ema_reps.pop(track_id, None)
            return

        embs = torch.stack([sample.embedding for sample in bank])
        if self._bank_weighted_mean:
            w = torch.tensor(
                [
                    s.quality_score if s.quality_score > 0.0 else s.det_score
                    for s in bank
                ],
                dtype=torch.float32,
            )
            w = w / w.sum().clamp(min=1e-6)
            mean_unnorm = (w.unsqueeze(1) * embs).sum(0)
        else:
            mean_unnorm = embs.mean(dim=0)
        representative = F.normalize(mean_unnorm, dim=0)
        n = len(bank)
        if n < 2:
            consistency = 1.0
        else:
            # mean pairwise cosine = (n * ||mean_unnorm||² - 1) / (n - 1)
            # derivation: sum_ij sim_ij = n²*||mean_unnorm||², subtract n diagonal
            q = float(torch.dot(mean_unnorm, mean_unnorm).item())  # saccade-allow-cpu
            consistency = (n * q - 1.0) / (n - 1)

        self._representatives[track_id] = representative
        self._consistency[track_id] = consistency
        if consistency >= self.consistency_threshold:
            self._clean_ids.add(track_id)
        else:
            self._clean_ids.discard(track_id)

        hq_samples = [
            s
            for s in bank
            if s.det_score >= self.high_quality_min_score
            and (
                s.aspect_ratio == 0.0
                or self.min_aspect <= s.aspect_ratio <= self.max_aspect
            )
        ]
        if hq_samples:
            hq_embs = [s.embedding for s in hq_samples]
            if len(hq_embs) >= 2:
                stacked_hq = torch.stack(hq_embs)
                n_hq = len(hq_embs)
                if self._bank_weighted_mean:
                    w_hq = torch.tensor(
                        [
                            s.quality_score if s.quality_score > 0.0 else s.det_score
                            for s in hq_samples
                        ],
                        dtype=torch.float32,
                    )
                    w_hq = w_hq / w_hq.sum().clamp(min=1e-6)
                    mean_hq = (w_hq.unsqueeze(1) * stacked_hq).sum(0)
                else:
                    mean_hq = stacked_hq.mean(dim=0)
                q_hq = float(torch.dot(mean_hq, mean_hq).item())  # saccade-allow-cpu
                hq_consistency = (n_hq * q_hq - 1.0) / (n_hq - 1)
                if hq_consistency >= self.consistency_threshold:
                    self._high_quality_reps[track_id] = F.normalize(mean_hq, dim=0)
                else:
                    self._high_quality_reps.pop(track_id, None)
            else:
                self._high_quality_reps[track_id] = hq_embs[0]
        else:
            self._high_quality_reps.pop(track_id, None)


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
        # Check if we are using the real C++ implementation
        self.is_cuda = not self.tracker.__class__.__name__.startswith(
            "dummy"
        ) and hasattr(self.tracker, "set_params")
        # In fact, the dummy class is defined in the same file if import fails.
        # Let's be more explicit:
        try:
            import saccade_tracking_ext

            self.is_cuda = isinstance(self.tracker, saccade_tracking_ext.GPUByteTracker)
        except ImportError:
            self.is_cuda = False

        # P2: pre-allocate shared GPU feature buffer; bind to C++ d_features_ so
        # Python index-scatter writes are instantly visible to CUDA association kernels.
        if self.is_cuda:
            self._bank_slot_features: torch.Tensor = torch.zeros(
                (max_objects, embedding_dim), device="cuda", dtype=torch.float32
            ).contiguous()
            self.tracker.bind_features_buffer(self._bank_slot_features.data_ptr())
        else:
            self._bank_slot_features = torch.zeros(
                (max_objects, embedding_dim), dtype=torch.float32
            )

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
        r_scale: float = 1.0,
        vel_dir_weight: float = 0.0,
        fuse_score_weight: float = 0.0,
        stage2_match_thresh: float = 0.5,
        birth_low_score_thresh: float = 0.0,
        birth_prox_norm_thresh: float = 0.0,
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
            r_scale,
            vel_dir_weight,
            fuse_score_weight,
            stage2_match_thresh,
            birth_low_score_thresh,
            birth_prox_norm_thresh,
        )

    def set_reid_params(
        self,
        cos_threshold: float = 0.90,
        iou_low: float = 0.30,
        iou_high: float = 0.60,
        weight: float = 0.40,
        cost_cos_w: float = 0.55,
        cost_iou_w: float = 0.30,
        cost_score_w: float = 0.15,
    ) -> None:
        """Set ReID fusion thresholds and appearance cost blend weights."""
        set_reid_params = getattr(self.tracker, "set_reid_params", None)
        if set_reid_params is not None:
            set_reid_params(
                cos_threshold,
                iou_low,
                iou_high,
                weight,
                cost_cos_w,
                cost_iou_w,
                cost_score_w,
            )

    def set_reid_min_candidates(self, min_candidates: int = 1) -> None:
        setter = getattr(self.tracker, "set_reid_min_candidates", None)
        if setter is not None:
            setter(min_candidates)

    def set_relink_params(
        self,
        enabled: bool = False,
        bank_cap: int = 256,
        sim_thresh: float = 0.6,
        cheb_lambda: float = 2.5,
        spatial_gate: float = 4.0,
        max_age: int = 300,
        bidirectional: bool = False,
        bridge_px: float = 0.25,
        bridge_at: int = 4,
        bridge_min_lost: int = 2,
        bridge_ttl: int = 120,
        bridge_max_speed: float = 0.0,
        bridge_person_height: float = 1.65,
        bridge_fps: float = 30.0,
        bridge_margin: float = 0.0,
        bridge_spatial_gate: float = 0.0,
        bridge_anchor: int = 0,
        bridge_anchor_rate: float = 0.0,
        bridge_h_lo: float = 0.0,
        bridge_h_hi: float = 0.0,
    ) -> None:
        setter = getattr(self.tracker, "set_relink_params", None)
        if setter is not None:
            setter(
                enabled,
                bank_cap,
                sim_thresh,
                cheb_lambda,
                spatial_gate,
                max_age,
                bidirectional,
                bridge_px,
                bridge_at,
                bridge_min_lost,
                bridge_ttl,
                bridge_max_speed,
                bridge_person_height,
                bridge_fps,
                bridge_margin,
                bridge_spatial_gate,
                bridge_anchor,
                bridge_anchor_rate,
                bridge_h_lo,
                bridge_h_hi,
            )

    def get_relink_debug(self) -> list[int]:
        getter = getattr(self.tracker, "get_relink_debug", None)
        return list(getter()) if getter is not None else [0, 0, 0, 0, 0]

    def set_oao_params(self, tau: float = 0.0) -> None:
        """OA-SORT Occlusion-Aware Offset (OAO) penalty.

        tau in [0, 1]: cost += tau * inter_track_IoU for each occluded track.
        0 (default) = disabled.  Recommended sweep: 0.1 – 0.4.
        """
        set_oao = getattr(self.tracker, "set_oao_params", None)
        if set_oao is not None:
            set_oao(float(tau))

    def set_quality_params(
        self,
        enabled: bool,
        w_aspect: float = 0.50,
        w_center: float = 0.30,
        w_area: float = 0.20,
    ) -> None:
        """Set Detection Quality Scaling (A6) parameters in CUDA tracker."""
        set_q = getattr(self.tracker, "set_quality_params", None)
        if set_q is not None:
            set_q(enabled, w_aspect, w_center, w_area)

    def set_frame_size(self, w: int, h: int) -> None:
        """Set frame dimensions for quality scaling (A6)."""
        set_fs = getattr(self.tracker, "set_frame_size", None)
        if set_fs is not None:
            set_fs(w, h)

    def set_homography(self, h: Optional[np.ndarray | List[float]]) -> None:
        """Set 3x3 homography matrix for 2D ground plane mapping (ADR 017)."""
        set_h = getattr(self.tracker, "set_homography", None)
        if set_h is not None:
            if h is None:
                set_h(None)
            else:
                import numpy as np

                h_arr = np.array(h, dtype=np.float32).flatten()
                if h_arr.size != 9:
                    raise ValueError("Homography must have 9 elements (3x3)")
                set_h(h_arr)

    def set_unified_score_params(
        self,
        w_sim_base: float = 0.0,
        w_iou_base: float = 0.0,
        w_maha_base: float = 0.0,
        shift_ambiguity: float = 0.0,
        shift_lost_age: float = 0.0,
    ) -> None:
        import saccade_tracking_ext

        set_unified_score_params = getattr(
            self.tracker, "set_unified_score_params", None
        )
        if set_unified_score_params is not None:
            params = saccade_tracking_ext.UnifiedScoreParams()
            params.w_sim_base = w_sim_base
            params.w_iou_base = w_iou_base
            params.w_maha_base = w_maha_base
            params.shift_ambiguity = shift_ambiguity
            params.shift_lost_age = shift_lost_age
            set_unified_score_params(params)

    def update_reference_features(
        self,
        track_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        """更新追蹤器的參考特徵（用於 ReID）。"""
        num = track_ids.size(0)
        if num == 0:
            return

        ids_contig = track_ids.to(torch.int32).cuda().contiguous()
        features_contig = features.to(torch.float32).cuda().contiguous()
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
            "scores": torch.empty(
                (self.max_objects,), device=device, dtype=torch.float32
            ),
            "ids": torch.empty((self.max_objects,), device=device, dtype=torch.int32),
            "classes": torch.empty(
                (self.max_objects,), device=device, dtype=torch.int32
            ),
            "det_idx": torch.empty(
                (self.max_objects,), device=device, dtype=torch.int32
            ),
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

    def push_keypoints(
        self,
        track_ids: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> None:
        """[Reserved] Push per-track keypoints for biometric profile construction.

        Not yet implemented in C++ tracker. Always no-op.
        """
        push = getattr(self.tracker, "push_keypoints", None)
        if push is None or track_ids.numel() == 0 or keypoints.numel() == 0:
            return
        ids_contig = track_ids.to(
            device=keypoints.device, dtype=torch.int32
        ).contiguous()
        kpts_contig = keypoints.to(torch.float32).contiguous()
        stream = torch.cuda.current_stream().cuda_stream
        push(
            ids_contig.data_ptr(),
            kpts_contig.data_ptr(),
            int(ids_contig.numel()),
            stream,
        )

    def get_biometric(self, track_id: int) -> Optional[dict[str, float]]:
        """[Reserved] Return per-track biometric profile (leg/shoulder/head ratios).

        Not yet implemented in C++ tracker. Always returns None.
        """
        get_bio = getattr(self.tracker, "get_biometric", None)
        if get_bio is None:
            return None
        profile = get_bio(int(track_id))
        if not getattr(profile, "valid", False):
            return None
        out = {
            "r_leg": float(profile.r_leg),
            "r_shoulder": float(profile.r_shoulder),
            "r_head": float(profile.r_head),
        }
        return {k: v for k, v in out.items() if not np.isnan(v)}

    def set_clean_embedding_flags(
        self,
        track_ids: torch.Tensor,
        flags: torch.Tensor,
    ) -> None:
        """Sync per-track clean-embedding flags from Python bank to the CUDA tracker."""
        if track_ids.numel() == 0:
            return
        stream = torch.cuda.current_stream().cuda_stream
        set_flags_host = getattr(self.tracker, "set_clean_embedding_flags_host", None)
        if set_flags_host is not None:
            ids_contig = (
                track_ids.to(torch.int32).cpu().contiguous()
            )  # saccade-allow-cpu
            flags_contig = flags.to(torch.bool).cpu().contiguous()  # saccade-allow-cpu
            set_flags_host(
                ids_contig.data_ptr(),
                flags_contig.data_ptr(),
                int(track_ids.numel()),
                stream,
            )
            return
        set_flags = getattr(self.tracker, "set_clean_embedding_flags", None)
        if set_flags is None:
            return
        ids_contig = track_ids.to(torch.int32).cuda().contiguous()
        flags_contig = flags.to(torch.bool).cuda().contiguous()
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
        """Write bank representatives directly into the shared _bank_slot_features buffer.

        P2 zero-copy path: fetches tid→slot from C++ (one shared D2H, cached by
        ensure_slot_map), then does a single batched GPU index-scatter — no N×D2D
        cudaMemcpyAsync calls.
        """
        if not bank_reps or not self.is_cuda:
            return
        tid_to_slot = dict(self.tracker.get_active_tid_slot_pairs())
        valid_tids = [tid for tid in bank_reps if tid in tid_to_slot]
        if not valid_tids:
            return
        slots = torch.tensor(
            [tid_to_slot[tid] for tid in valid_tids], dtype=torch.long, device="cuda"
        )
        # reps are CPU tensors; one batched H2D transfer is cheaper than per-track .cuda() at update time
        stacked = torch.stack(
            [bank_reps[tid] for tid in valid_tids]
        ).cuda()  # [n, D] GPU
        self._bank_slot_features[slots] = stacked


class GraphedTrackerUpdate:
    """CUDA-graph-captured ``update_into`` for the GPU byte tracker.

    Pre-allocates fixed-size input/output buffers (box/scores padded to
    ``max_assoc``) and lazy-captures the C++ :meth:`update_into` call via
    :func:`torch.cuda.make_graphed_callables`.  The tracker's
    ``run_update_device`` has been modified to always launch every kernel
    with fixed grids — zero-padded detection slots are naturally filtered
    by score/IoU gates inside the kernels.

    Usage::

        gtu = GraphedTrackerUpdate(tracker, max_assoc=1024)
        # Per frame:
        gtu.copy_inputs(boxes, scores, classes, gmc)
        result = gtu.replay()
        gtu.read_outputs(boxes_out, scores_out, ...)  # or use result dict
    """

    def __init__(
        self,
        tracker: GPUByteTracker,
        max_assoc: int = 1024,
        max_objs: int = 2048,
    ):
        self._tracker = tracker
        self._max_assoc = max_assoc
        self._max_objs = max_objs

        device = torch.device("cuda")

        self.d_boxes = torch.zeros(max_assoc, 4, dtype=torch.float32, device=device)
        self.d_scores = torch.zeros(max_assoc, dtype=torch.float32, device=device)
        self.d_classes = torch.zeros(max_assoc, dtype=torch.int32, device=device)
        self.d_gmc = (
            torch.eye(2, 3, dtype=torch.float32, device=device)
            .flatten()[:6]
            .contiguous()
        )

        self.out_boxes = torch.zeros(max_objs, 4, dtype=torch.float32, device=device)
        self.out_scores = torch.zeros(max_objs, dtype=torch.float32, device=device)
        self.out_ids = torch.zeros(max_objs, dtype=torch.int32, device=device)
        self.out_classes = torch.zeros(max_objs, dtype=torch.int32, device=device)
        self.out_det_idx = torch.zeros(max_objs, dtype=torch.int32, device=device)
        self.out_count = torch.zeros((), dtype=torch.int32, device=device)

        self._graphed_callable: Callable[..., None] | None = None

    def _warmup(self) -> None:
        with torch.no_grad():
            stream = torch.cuda.current_stream().cuda_stream
            self._tracker.tracker.update_into(
                self.d_boxes.data_ptr(),
                self.d_scores.data_ptr(),
                self.d_classes.data_ptr(),
                self._max_assoc,
                stream,
                self.out_boxes.data_ptr(),
                self.out_scores.data_ptr(),
                self.out_ids.data_ptr(),
                self.out_classes.data_ptr(),
                self.out_det_idx.data_ptr(),
                self.out_count.data_ptr(),
                0,
                self.d_gmc.data_ptr(),
                0.0,
                1.0,
            )
            torch.cuda.synchronize()  # saccade-allow-cpu

    def _capture(self) -> None:
        if self._graphed_callable is not None:
            return
        self._warmup()

        def _graph_fn(
            boxes: torch.Tensor,
            scores: torch.Tensor,
            classes: torch.Tensor,
            d_gmc: torch.Tensor,
            out_boxes: torch.Tensor,
            out_scores: torch.Tensor,
            out_ids: torch.Tensor,
            out_classes: torch.Tensor,
            out_det_idx: torch.Tensor,
            out_count: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            tracker_cpp = self._tracker.tracker
            stream = torch.cuda.current_stream().cuda_stream
            tracker_cpp.update_into(
                boxes.data_ptr(),
                scores.data_ptr(),
                classes.data_ptr(),
                self._max_assoc,
                stream,
                out_boxes.data_ptr(),
                out_scores.data_ptr(),
                out_ids.data_ptr(),
                out_classes.data_ptr(),
                out_det_idx.data_ptr(),
                out_count.data_ptr(),
                0,
                d_gmc.data_ptr(),
                0.0,
                1.0,
            )
            return (out_boxes, out_scores, out_ids, out_classes, out_det_idx, out_count)

        sample = (
            self.d_boxes,
            self.d_scores,
            self.d_classes,
            self.d_gmc,
            self.out_boxes,
            self.out_scores,
            self.out_ids,
            self.out_classes,
            self.out_det_idx,
            self.out_count,
        )
        self._graphed_callable = cast(
            "Callable[..., None]", torch.cuda.make_graphed_callables(_graph_fn, sample)
        )

    def copy_inputs(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        gmc: torch.Tensor | None = None,
    ) -> None:
        n = min(boxes.shape[0], self._max_assoc)
        if n > 0:
            self.d_boxes[:n].copy_(boxes[:n].float())
            self.d_scores[:n].copy_(scores[:n].float())
            self.d_classes[:n].copy_(classes[:n].int())
        # Only zero the tail beyond valid dets — the tracker gates on score,
        # so stale data in slots [n:] must be cleared to avoid ghost associations.
        if n < self._max_assoc:
            self.d_boxes[n:].zero_()
            self.d_scores[n:].zero_()
            self.d_classes[n:].zero_()
        if gmc is not None:
            self.d_gmc.copy_(gmc.flatten()[:6].float().contiguous())

    def replay(self) -> dict[str, torch.Tensor]:
        if self._graphed_callable is None:
            self._capture()
        assert self._graphed_callable is not None
        self._graphed_callable(
            self.d_boxes,
            self.d_scores,
            self.d_classes,
            self.d_gmc,
            self.out_boxes,
            self.out_scores,
            self.out_ids,
            self.out_classes,
            self.out_det_idx,
            self.out_count,
        )
        return {
            "boxes": self.out_boxes,
            "scores": self.out_scores,
            "ids": self.out_ids,
            "classes": self.out_classes,
            "det_idx": self.out_det_idx,
            "count": self.out_count,
        }

    def read_outputs(self, target: dict[str, torch.Tensor]) -> None:
        """Copy internal output buffers into the caller-provided result dict."""
        target["count"].copy_(self.out_count, non_blocking=True)
        count = int(self.out_count.item())  # saccade-allow-cpu
        if count > 0:
            target["boxes"][:count].copy_(self.out_boxes[:count], non_blocking=True)
            target["scores"][:count].copy_(self.out_scores[:count], non_blocking=True)
            target["ids"][:count].copy_(self.out_ids[:count], non_blocking=True)
            target["classes"][:count].copy_(self.out_classes[:count], non_blocking=True)
            target["det_idx"][:count].copy_(self.out_det_idx[:count], non_blocking=True)
        else:
            target["count"].zero_()
