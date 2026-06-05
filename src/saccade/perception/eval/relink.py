import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

from .motion_model import MotionModel, MotionModelRegistry


@dataclass(frozen=False)
class MotionRelinkingConfig:
    """Configuration for motion-based relinking module.

    Motion-based relinking complements appearance-based matching by predicting
    where lost tracks should reappear based on their velocity history. This is
    especially useful during occlusion recovery when ReID features are unavailable
    or unreliable.

    Design principles
    -----------------
    1. Motion evidence is *additional* — it augments (never replaces) appearance
       similarity. Appearance-only matches always take priority.
    2. Motion IoU is added as a soft bonus to the joint score, not a hard gate.
    3. Motion consistency is checked before accepting a motion-only match
       (no embedding) to prevent spurious associations.
    """

    # --- Motion IoU bonus in joint score ---
    w_motion_iou: float = 0.15
    motion_iou_threshold: float = 0.05

    # --- Motion consistency ---
    motion_consistency_check: bool = True
    consistency_tol: float = 2.0

    # --- Motion-only matching (no embedding) ---
    enable_motion_only: bool = False
    motion_only_iou_threshold: float = 0.08
    motion_only_min_lost_frames: int = 3
    motion_only_max_lost_frames: int = 15
    motion_only_lost_frames: int = 5

    # --- Velocity/acceleration model ---
    vel_alpha: float = 0.3
    acc_alpha: float = 0.15
    min_motion_observations: int = 2


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
        # Kalman probabilistic relink gate
        kalman_gate: bool = False,
        kalman_chi2: float = 9.4877,
        kalman_penalty_weight: float = 0.0,
        kalman_dir_min_cos: float = -1.0,
        kalman_dir_min_speed: float = 1.0,
        kalman_person_height_m: float = 0.0,
        kalman_accel_long: float = 2.0,
        kalman_accel_lat: float = 1.0,
        kalman_fps: float = 30.0,
        kalman_max_speed_mps: float = 0.0,
        delayed_claim: bool = False,
        claim_warmup_frames: int = 3,
        bidirectional: bool = False,
        bridge_chi2: float = 1.5,
        # Motion-based relinking params
        motion_vel_alpha: float = 0.3,
        motion_acc_alpha: float = 0.15,
        motion_min_observations: int = 2,
        motion_w_iou: float = 0.3,
        motion_consistency_check: bool = True,
        motion_consistency_tol: float = 2.0,
        motion_enable_motion_only: bool = True,
        motion_motion_only_lost_frames: int = 5,
        motion_motion_only_iou_threshold: float = 0.15,
        motion_motion_only_min_lost_frames: int = 1,
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
        # Kalman probabilistic gate: replace static center/IoU spatial gate with a
        # chi-square test against the lost track's velocity-extrapolated, covariance-
        # inflated Kalman distribution (mirrors the C++ relinker).
        self.kalman_gate = bool(kalman_gate)
        self.kalman_chi2 = float(kalman_chi2)
        self.kalman_penalty_weight = max(0.0, float(kalman_penalty_weight))
        # Velocity-direction gate: reject candidates that lie "behind" the lost
        # track's motion (cos(displacement, velocity) < min_cos) so a person leaving
        # frame isn't relinked to someone entering behind them. <=-1 disables.
        self.kalman_dir_min_cos = float(kalman_dir_min_cos)
        self.kalman_dir_min_speed = max(0.0, float(kalman_dir_min_speed))
        # Physically-grounded, inertia-aware diffusion: assume real person height to
        # get px/m scale, bound human acceleration (longitudinal vs lateral) so the
        # cloud stretches along motion (inertia) and stays tight sideways. >0 enables.
        self.kalman_person_height_m = max(0.0, float(kalman_person_height_m))
        self.kalman_accel_long = max(0.0, float(kalman_accel_long))
        self.kalman_accel_lat = max(0.0, float(kalman_accel_lat))
        self.kalman_fps = kalman_fps if kalman_fps > 0.0 else 30.0
        # Physical reachability cap: reject relink if the implied average speed
        # (dist / time, scaled to m/s via person height) exceeds human limits.
        # Snapshot-independent → always applies, closing the no-snapshot fallback.
        self.kalman_max_speed_mps = max(0.0, float(kalman_max_speed_mps))
        self.delayed_claim = bool(delayed_claim)
        self.claim_warmup_frames = max(1, int(claim_warmup_frames))
        self.bidirectional = bool(bidirectional)
        self.bridge_px = float(bridge_chi2)
        self.clean_score_threshold = clean_score_threshold
        self.clean_margin_ratio = clean_margin_ratio
        self.clean_min_aspect = clean_min_aspect
        self.clean_max_aspect = clean_max_aspect
        self.strict_sim_threshold = (
            strict_sim_threshold if strict_sim_threshold > 0.0 else sim_threshold
        )
        self.debug = debug
        # ------------------------------------------------------------------ #
        # Motion-based relinking (A7) — velocity tracking + motion IoU bonus
        # ------------------------------------------------------------------ #
        self.motion_cfg = MotionRelinkingConfig(
            vel_alpha=motion_vel_alpha,
            acc_alpha=motion_acc_alpha,
            min_motion_observations=motion_min_observations,
            w_motion_iou=motion_w_iou,
            motion_consistency_check=motion_consistency_check,
            consistency_tol=motion_consistency_tol,
            enable_motion_only=motion_enable_motion_only,
            motion_only_lost_frames=motion_motion_only_lost_frames,
            motion_only_iou_threshold=motion_motion_only_iou_threshold,
            motion_only_min_lost_frames=motion_motion_only_min_lost_frames,
        )
        self.motion_registry: MotionModelRegistry = MotionModelRegistry(
            vel_alpha=self.motion_cfg.vel_alpha,
            acc_alpha=self.motion_cfg.acc_alpha,
            min_observations=self.motion_cfg.min_motion_observations,
        )

        self.alias: Dict[int, int] = {}
        # Per-frame uniqueness: a canonical id may be emitted at most once per
        # timestep. Online relink commits ``alias[raw]→canonical`` permanently, so
        # a relinked alias can collide with the *original* canonical when the
        # original re-appears after a long absence. When that happens we split the
        # losing raw track onto a fresh surrogate id (>= ``_split_alias_base``) so
        # the MOT output never contains a duplicate id in one frame.
        self._split_alias_base = 1_000_000
        self._split_counter = 0
        self.features: Dict[int, torch.Tensor] = {}
        self.buffers: Dict[int, List[torch.Tensor]] = {}
        self.last_seen: Dict[int, int] = {}
        self.last_boxes: Dict[int, torch.Tensor] = {}
        self.motion: Dict[int, Any] = {}
        self.motion_frame: Dict[int, int] = {}
        self._ema_h: Dict[int, float] = {}
        self._foot_history: Dict[int, List[Tuple[float, float]]] = {}
        self._pending_claims: Dict[int, Dict[str, int]] = {}
        self._deferred_alias: Dict[int, int] = {}
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
            "reject_kalman": 0,
            "reject_direction": 0,
            "reject_speed": 0,
            "reject_backward": 0,
            "accept_bidir": 0,
            "delayed_claim_pending": 0,
            "delayed_claim_ready": 0,
            "delayed_claim_accepted": 0,
            "reject_quality": 0,
            "reject_quality_score": 0,
            "reject_quality_margin": 0,
            "reject_quality_aspect_low": 0,
            "reject_quality_aspect_high": 0,
            "appearance_first_bypass": 0,
            "new_ids": 0,
            "reject_biometric": 0,
            "relink_split_collision": 0,
            # Motion-based relinking stats
            "motion_bonus": 0,
            "motion_only_match": 0,
            "motion_consistency_fail": 0,
            "motion_iou_gate_pass": 0,
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

    def _split_on_collision(self, raw_id: int) -> int:
        """Resolve a same-frame id collision by splitting ``raw_id`` off.

        The canonical id ``raw_id`` currently resolves to was already emitted this
        frame by a different raw track. Assign ``raw_id`` a fresh surrogate id and
        persist it in ``alias`` so the split is stable across subsequent frames
        (the false merge stays undone from here on). Returns the surrogate id.
        """
        self._split_counter += 1
        surrogate = self._split_alias_base + self._split_counter
        self.alias[raw_id] = surrogate
        self.stats["relink_split_collision"] += 1
        return surrogate

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

    def update_motion_snapshots(self, snapshots: List[Any], frame_id: int = -1) -> None:
        for snap in snapshots:
            canonical = self.alias.get(snap.obj_id, snap.obj_id)
            self.motion[canonical] = snap
            self.motion_frame[canonical] = frame_id

    def _kalman_predict_np(
        self, x: np.ndarray, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One Kalman predict step (constant velocity F + process noise Q).

        Mirrors ``kf_gpu::predict``: advance the mean by velocity then inflate the
        covariance with Q (built from the post-predict height x[3]).
        """
        x = x.copy()
        P = P.copy()
        x[0] += x[4]
        x[1] += x[5]
        x[2] += x[6]
        x[3] += x[7]
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1.0
        P = F @ P @ F.T
        h = max(float(x[3]), 1e-6)
        pos = h / 20.0
        vel = h / 160.0
        Q = np.diag(
            [
                pos * pos,
                pos * pos,
                1e-4,
                pos * pos,
                vel * vel,
                vel * vel,
                1e-10,
                vel * vel,
            ]
        ).astype(np.float64)
        return x, P + Q

    def _predict_phys_np(
        self, x: np.ndarray, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Physically-grounded, inertia-aware predict step (mirrors C++ predict_phys).

        Mean advances at constant velocity (momentum). Acceleration noise is derived
        from assumed person height (px/m scale) and bounded human acceleration,
        decomposed along the velocity direction: longitudinal (speed change) vs
        lateral (turning), so the cloud stretches along motion and stays tight
        sideways.
        """
        x = x.copy()
        P = P.copy()
        x[0] += x[4]
        x[1] += x[5]
        x[2] += x[6]
        x[3] += x[7]
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1.0
        P = F @ P @ F.T

        px_per_m = max(float(x[3]), 1e-3) / max(self.kalman_person_height_m, 1e-3)
        inv_fps2 = 1.0 / max(self.kalman_fps * self.kalman_fps, 1e-6)
        sl = self.kalman_accel_long * px_per_m * inv_fps2
        st = self.kalman_accel_lat * px_per_m * inv_fps2
        sl2, st2 = sl * sl, st * st
        vx, vy = float(x[4]), float(x[5])
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > 1e-6:
            ux, uy = vx / speed, vy / speed
        else:
            ux, uy = 1.0, 0.0
            sl2 = st2  # no heading → isotropic
        # Sigma_a = sl2*(u u^T) + st2*(p p^T), p=(-uy,ux)
        sa00 = sl2 * ux * ux + st2 * uy * uy
        sa01 = (sl2 - st2) * ux * uy
        sa11 = sl2 * uy * uy + st2 * ux * ux
        sa = np.array([[sa00, sa01], [sa01, sa11]], dtype=np.float64)
        pos_idx = (0, 1)
        vel_idx = (4, 5)
        for a in range(2):
            for b in range(2):
                P[pos_idx[a], pos_idx[b]] += 0.25 * sa[a, b]
                P[pos_idx[a], vel_idx[b]] += 0.5 * sa[a, b]
                P[vel_idx[a], pos_idx[b]] += 0.5 * sa[a, b]
                P[vel_idx[a], vel_idx[b]] += sa[a, b]
        # modest noise on aspect & height for conditioning (matches kf_gpu get_Q)
        h = max(float(x[3]), 1e-6)
        P[2, 2] += 1e-4
        P[6, 6] += 1e-10
        P[3, 3] += (h / 20.0) ** 2
        P[7, 7] += (h / 160.0) ** 2
        return x, P

    @staticmethod
    def _regress_velocity_4(
        positions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Closed-form linear regression velocity from 4 equally-spaced frames.

        For frames t ∈ {0,1,2,3} with centres (x_i, y_i)::

            v_x = (3·x₃ + x₂ − x₁ − 3·x₀) / 10
            v_y = (3·y₃ + y₂ − y₁ − 3·y₀) / 10
        """
        if len(positions) < 4:
            return 0.0, 0.0
        x0, y0 = positions[0]
        x1, y1 = positions[1]
        x2, y2 = positions[2]
        x3, y3 = positions[3]
        return (3.0 * x3 + x2 - x1 - 3.0 * x0) / 10.0, (
            3.0 * y3 + y2 - y1 - 3.0 * y0
        ) / 10.0

    def _midpoint_bridge_dist(
        self,
        lost_id: int,
        cand_id: int,
        gap: int,
        cand_cx: float = 0.0,
        cand_cy: float = 0.0,
    ) -> float:
        lost_hist = self._foot_history.get(lost_id, [])
        cand_hist = list(self._foot_history.get(cand_id, []))
        cand_hist.append((cand_cx, cand_cy))
        vx_l, vy_l = self._regress_velocity_4(lost_hist[-4:])
        vx_c, vy_c = self._regress_velocity_4(cand_hist[:4])
        h_lost = self._ema_h.get(lost_id, 1.0)
        h_cand = self._ema_h.get(cand_id, 1.0)
        half = gap * 0.5
        x_l = lost_hist[-1][0] + vx_l * half if lost_hist else 0.0
        y_l = lost_hist[-1][1] + vy_l * half if lost_hist else 0.0
        x_c = cand_hist[0][0] - vx_c * half if cand_hist else 0.0
        y_c = cand_hist[0][1] - vy_c * half if cand_hist else 0.0
        dist_px = float(np.hypot(x_l - x_c, y_l - y_c))
        h_ref = max((h_lost + h_cand) * 0.5, 1.0)
        return dist_px / h_ref

    def _exceeds_max_speed(
        self, box: torch.Tensor, last_box: torch.Tensor, age: int
    ) -> bool:
        """True if relinking would imply an above-human average speed (new person).

        Snapshot-independent (uses last_box + age), so it also guards the path where
        no Kalman snapshot is available. dist/time = average velocity → the physical
        meaning a positional cloud alone lacks. Mirrors C++ ``exceeds_max_speed``.
        """
        if self.kalman_max_speed_mps <= 0.0 or self.kalman_person_height_m <= 0.0:
            return False
        lcx = float((last_box[0] + last_box[2]) * 0.5)
        lcy = float((last_box[1] + last_box[3]) * 0.5)
        lh = float(last_box[3] - last_box[1])
        zcx = float((box[0] + box[2]) * 0.5)
        zcy = float((box[1] + box[3]) * 0.5)
        zh = float(box[3] - box[1])
        dpx = ((zcx - lcx) ** 2 + (zcy - lcy) ** 2) ** 0.5
        dt_s = max(age, 1) / max(self.kalman_fps, 1e-6)
        px_per_m = 0.5 * (max(lh, 1e-3) + max(zh, 1e-3)) / self.kalman_person_height_m
        speed_mps = dpx / dt_s / max(px_per_m, 1e-6)
        return bool(speed_mps > self.kalman_max_speed_mps)

    def _direction_behind(self, box: torch.Tensor, snapshot: Any) -> bool:
        """True if ``box`` lies behind the lost track's velocity direction.

        Mirrors the C++ ``direction_behind``: covariance inflation grows the gate
        symmetrically, so this culls candidates whose displacement from the last
        position opposes the velocity (e.g. someone entering as the track leaves).
        Skipped for near-stationary tracks where direction is just noise.
        """
        if self.kalman_dir_min_cos <= -1.0:
            return False
        state = np.asarray(snapshot.state, dtype=np.float64).reshape(-1)
        vx, vy = float(state[4]), float(state[5])
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < self.kalman_dir_min_speed:
            return False
        zcx = float((box[0] + box[2]) * 0.5)
        zcy = float((box[1] + box[3]) * 0.5)
        dx = zcx - float(state[0])
        dy = zcy - float(state[1])
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-3:
            return False
        cosang = (dx * vx + dy * vy) / (speed * dist)
        return bool(cosang < self.kalman_dir_min_cos)

    def _kalman_gate_dist(
        self, box: torch.Tensor, snapshot: Any, delta: int, dims: int = 4
    ) -> float:
        """Squared Mahalanobis distance of ``box`` to the lost track's Kalman cloud,
        self-extrapolated ``delta`` frames forward (covariance inflation)."""
        x = np.asarray(snapshot.state, dtype=np.float64).reshape(-1)[:8].copy()
        P = np.asarray(snapshot.covariance, dtype=np.float64).reshape(8, 8).copy()
        phys = self.kalman_person_height_m > 0.0
        for _ in range(max(0, delta)):
            x, P = (
                self._predict_phys_np(x, P) if phys else self._kalman_predict_np(x, P)
            )
        if dims <= 2:
            h = max(float(x[3]), 1e-6)
            r = np.diag([(h / 20.0) ** 2, (h / 20.0) ** 2]).astype(np.float64)
            s = P[:2, :2] + r
            meas = np.array(
                [
                    (float(box[0]) + float(box[2])) * 0.5,
                    (float(box[1]) + float(box[3])) * 0.5,
                ],
                dtype=np.float64,
            )
            residual = meas - x[:2]
        else:
            h = max(float(x[3]), 1e-6)
            pos = h / 20.0
            r = np.diag([pos * pos, pos * pos, 1e-2, pos * pos]).astype(np.float64)
            s = P[:4, :4] + r
            residual = self._measurement(box).astype(np.float64) - x[:4]
        try:
            solved = np.linalg.solve(s, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(s) @ residual
        return float(residual @ solved)

    def motion_candidate_ids(self, frame_id: int = -1) -> List[int]:
        if (
            self.mahalanobis_threshold <= 0.0
            and not self.kalman_gate
            and not self.debug
        ):
            return []
        # Kalman gate needs fresh snapshots of still-active lost tracks (age>=0) so it
        # can capture a clean farewell state before the tracker deactivates them.
        min_age = 0 if self.kalman_gate else self.min_lost_frames
        ids: List[int] = []
        for cid in self.features:
            if frame_id >= 0:
                age = frame_id - self.last_seen.get(cid, -(10**9))
                if age < min_age or age > self.ttl:
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

    @property
    def deferred_alias(self) -> Dict[int, int]:
        return dict(self._deferred_alias)

    def _mark_pending_claim(self, raw_id: int, frame_id: int) -> bool:
        if not self.delayed_claim:
            return False
        pending = self._pending_claims.get(raw_id)
        if pending is None:
            pending = {"first": frame_id, "last": frame_id, "hits": 1}
            self._pending_claims[raw_id] = pending
        elif pending["last"] != frame_id:
            pending["last"] = frame_id
            pending["hits"] += 1
        ready = pending["hits"] >= self.claim_warmup_frames
        self.stats["delayed_claim_ready" if ready else "delayed_claim_pending"] += 1
        return not ready

    def _is_pending_ready(self, raw_id: int) -> bool:
        pending = self._pending_claims.get(raw_id)
        return bool(pending and pending["hits"] >= self.claim_warmup_frames)

    def _update_motion_model(
        self, canonical_id: int, box: torch.Tensor, frame_id: int
    ) -> None:
        """Update per-track velocity/acceleration model after a successful match."""
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        bw = max(1e-6, float(box[2] - box[0]))
        bh = max(1e-6, float(box[3] - box[1]))
        self.motion_registry.update(canonical_id, cx, cy, bw, bh, frame_id)

    def _motion_predict(self, track_id: int, offset: int = 1) -> Optional[torch.Tensor]:
        """Predict where a lost track should appear, using EMA velocity/acceleration."""
        return self.motion_registry.predict(track_id, offset)

    def _motion_iou(self, track_id: int, box: torch.Tensor, offset: int = 1) -> float:
        """Compute IoU between a detection and the predicted box for a lost track."""
        return self.motion_registry.motion_iou(track_id, box, offset)

    def _motion_consistent(
        self, track_id: int, box: torch.Tensor, frame_id: int, tol: float = 2.0
    ) -> bool:
        """Check if detection's displacement is consistent with track velocity."""
        return self.motion_registry.velocity_consistent(track_id, box, frame_id, tol)

    def _resolve_motion_only(
        self,
        raw_id: int,
        box: torch.Tensor,
        frame_id: int,
        w: int,
        h: int,
        assigned: Set[int],
    ) -> int:
        """Motion-only matching when no embedding is available.

        When a detection has no ReID embedding, try to recover a lost track
        using motion prediction alone. This is especially useful during
        occlusion recovery when embeddings are unavailable.
        """
        self.stats["attempts"] += 1
        best_id = None
        best_motion_iou = -1.0
        best_age = -1

        for cid in self.features:
            if cid in assigned:
                self.stats["reject_assigned"] += 1
                continue
            age = frame_id - self.last_seen.get(cid, -(10**9))
            if age < self.motion_cfg.motion_only_min_lost_frames or age > self.ttl:
                self.stats["reject_age"] += 1
                continue

            # Predict where this track should be
            pred_box = self._motion_predict(cid, offset=age)
            if pred_box is None:
                continue

            # Motion consistency check
            if self.motion_cfg.motion_consistency_check:
                if not self._motion_consistent(cid, box, frame_id):
                    self.stats["motion_consistency_fail"] += 1
                    continue

            # Compute IoU with predicted box
            miou = MotionModel.compute_iou(box, pred_box)
            if miou < self.motion_cfg.motion_only_iou_threshold:
                continue

            self.stats["motion_iou_gate_pass"] += 1

            # Prefer higher motion IoU (tiebreak: older tracks)
            if miou > best_motion_iou or (miou == best_motion_iou and age > best_age):
                best_motion_iou = miou
                best_id = cid
                best_age = age

        if best_id is not None:
            self.stats["motion_only_match"] += 1
            self.stats["accepted"] += 1
            self.alias[raw_id] = best_id
            self._update_motion_model(best_id, box, frame_id)
            self.last_seen[best_id] = frame_id
            self.last_boxes[best_id] = box
            assigned.add(best_id)
        else:
            self.stats["new_ids"] += 1
            self.alias[raw_id] = raw_id

        canonical = self.alias[raw_id]
        # Per-frame uniqueness guard (see resolve()).
        if canonical in assigned:
            canonical = self._split_on_collision(raw_id)
        self.last_seen.setdefault(canonical, frame_id)
        self.last_boxes.setdefault(canonical, box)
        bh = float(box[3] - box[1])
        old_h = self._ema_h.get(canonical)
        self._ema_h[canonical] = bh if old_h is None else 0.95 * old_h + 0.05 * bh
        if canonical not in self.features:
            self.features[canonical] = torch.zeros(1, device=self.device)
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        hist = self._foot_history.setdefault(canonical, [])
        hist.append((cx, cy))
        if len(hist) > 8:
            hist.pop(0)
        return self.alias[raw_id]

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
            if self.motion_cfg.enable_motion_only and not self.bidirectional:
                return self._resolve_motion_only(raw_id, box, frame_id, w, h, assigned)
            emb = None

        if embedding is not None:
            emb = self._normalize(embedding)
        else:
            emb = None

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

        if (
            self.delayed_claim
            and raw_id in self._pending_claims
            and self.alias.get(raw_id, raw_id) == raw_id
        ):
            self._mark_pending_claim(raw_id, frame_id)
        claim_ready = self._is_pending_ready(raw_id)
        first_claim = raw_id not in self.alias
        delayed_self_claim = (
            self.delayed_claim
            and claim_ready
            and self.alias.get(raw_id, raw_id) == raw_id
        )
        if first_claim and self._mark_pending_claim(raw_id, frame_id):
            self.alias[raw_id] = raw_id
        elif first_claim or delayed_self_claim:
            self.stats["attempts"] += 1
            best_id = None
            best_joint = -1.0  # Sentinel for normalized joint score
            best_sim_raw = 0.0  # raw cosine of the current winner
            second_best_joint = -2.0  # runner-up joint score
            best_iou, best_center, best_maha = 0.0, 0.0, 0.0
            best_bypassed = False

            candidates_to_score = []
            for cid in self.features:
                if self.delayed_claim and cid == raw_id:
                    continue
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
                # Physical reachability gate (always-on, snapshot-independent).
                if self._exceeds_max_speed(box, last_box, age):
                    self.stats["reject_speed"] += 1
                    self.age_gate_pass_outcomes.append((age, "speed"))
                    continue
                center_norm, iou = self._spatial_metrics(box, last_box, w, h)

                # Motion evidence: C++ Kalman motion box
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

                # Motion evidence: EMA velocity/acceleration model
                motion_iou_ema = 0.0
                if self.motion_cfg.w_motion_iou > 0.0:
                    motion_iou_ema = self._motion_iou(cid, box, offset=age)

                # Motion consistency check
                motion_ok = True
                if (
                    self.motion_cfg.motion_consistency_check
                    and self.motion_cfg.w_motion_iou > 0.0
                ):
                    if not self._motion_consistent(cid, box, frame_id):
                        motion_ok = False
                        self.stats["motion_consistency_fail"] += 1

                # Kalman probabilistic gate: when a snapshot exists, gate by chi-square
                # distance to the extrapolated/inflated distribution instead of the
                # static center/IoU gate. Falls back to static gate otherwise.
                kalman_d2 = -1.0
                kalman_gated = False
                if self.kalman_gate and snapshot is not None:
                    if self._direction_behind(box, snapshot):
                        self.stats["reject_direction"] += 1
                        self.age_gate_pass_outcomes.append((age, "direction"))
                        continue
                    delta = frame_id - self.motion_frame.get(cid, frame_id)
                    kalman_d2 = self._kalman_gate_dist(
                        box, snapshot, delta, dims=2 if self.bidirectional else 4
                    )
                    if kalman_d2 > self.kalman_chi2:
                        self.stats["reject_kalman"] += 1
                        self.age_gate_pass_outcomes.append((age, "kalman"))
                        continue
                    kalman_gated = True

                bridge_dist = -1.0
                if self.bidirectional and cid in self._foot_history:
                    cx = (float(box[0]) + float(box[2])) * 0.5
                    cy = (float(box[1]) + float(box[3])) * 0.5
                    gap = frame_id - self.last_seen.get(cid, frame_id)
                    bridge_dist = self._midpoint_bridge_dist(cid, raw_id, gap, cx, cy)
                    if bridge_dist > self.bridge_px:
                        self.stats["reject_backward"] += 1
                        self.age_gate_pass_outcomes.append((age, "backward"))
                        continue
                    self.stats["accept_bidir"] += 1
                spatial_failed = (not kalman_gated) and (
                    center_norm > self.spatial_gate or iou < self.min_iou
                )
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
                    (
                        cid,
                        age,
                        iou,
                        center_norm,
                        maha,
                        spatial_failed,
                        motion_iou_ema,
                        motion_ok,
                        kalman_d2 if kalman_gated else -1.0,
                        bridge_dist,
                    )
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
            _sim_iter = None
            if emb is not None and candidates_to_score and self.buffer_size == 1:
                _cand_ids = [c[0] for c in candidates_to_score]
                _bank = torch.stack([self.features[cid] for cid in _cand_ids]).to(
                    self.device
                )
                _batch_sims = (_bank @ emb).tolist()  # single kernel + single D2H
                _sim_iter = iter(_batch_sims)

            for (
                cid,
                age,
                iou,
                center_norm,
                maha,
                spatial_failed,
                motion_iou_ema,
                motion_ok,
                kalman_d2,
                bridge_dist,
            ) in candidates_to_score:
                if emb is None:
                    sim = 0.0
                elif self.buffer_size > 1:
                    sim = self._buffer_sim(cid, emb)
                else:
                    sim = next(_sim_iter)

                # Hard appearance gate: raw cosine must still pass sim_threshold.
                if emb is not None and sim < current_sim_thresh:
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

                # Motion evidence bonus: add motion IoU as soft evidence
                motion_bonus = 0.0
                if motion_ok:
                    motion_bonus = self.motion_cfg.w_motion_iou * motion_iou_ema

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

                    joint = (
                        w_sim * sim + w_iou * iou + w_maha * maha_score + motion_bonus
                    )
                elif _use_legacy_joint:
                    joint = (
                        sim
                        + self.iou_weight * iou
                        + self.mahalanobis_weight * maha_score
                        + motion_bonus
                    )
                else:
                    joint = sim + motion_bonus

                # Spatial probability penalty: cost = 1 - exp(-D^2/2); closer to the
                # predicted cloud center costs less.
                if self.kalman_penalty_weight > 0.0 and kalman_d2 >= 0.0:
                    joint -= self.kalman_penalty_weight * (
                        1.0 - float(np.exp(-0.5 * kalman_d2))
                    )

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
                # Update motion model with matched position (always, not just when motion_iou_ema > 0)
                self._update_motion_model(best_id, box, frame_id)
                self.alias[raw_id] = best_id
                if self.delayed_claim and raw_id != best_id:
                    self.stats["delayed_claim_accepted"] += 1
                    self._deferred_alias[raw_id] = best_id
                    self._pending_claims.pop(raw_id, None)
            else:
                self.stats["new_ids"] += 1
                self.alias[raw_id] = raw_id
                if self.delayed_claim and claim_ready:
                    self._pending_claims.pop(raw_id, None)

        canonical = self.alias[raw_id]
        # Per-frame uniqueness guard: never emit the same canonical id twice in one
        # timestep. Triggered when a relinked alias collides with the original id
        # re-appearing (fast-path emits skip the matching `assigned` check above).
        if canonical in assigned:
            canonical = self._split_on_collision(raw_id)
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
            if emb is not None:
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
            elif canonical not in self.features:
                self.features[canonical] = torch.zeros(1, device=self.device)
        self.last_seen[canonical] = frame_id
        self.last_boxes[canonical] = box
        bh = float(box[3] - box[1])
        old_h = self._ema_h.get(canonical)
        self._ema_h[canonical] = bh if old_h is None else 0.95 * old_h + 0.05 * bh
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        hist = self._foot_history.setdefault(canonical, [])
        hist.append((cx, cy))
        if len(hist) > 8:
            hist.pop(0)
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
        order = self._frame_order([c[3] for c in candidates])
        out: List[int] = [0] * len(candidates)
        for i in order:
            raw_id, embedding, box, score = candidates[i]
            out[i] = self.resolve(
                raw_id, embedding, box, score, frame_id, w, h, assigned
            )
        return out

    def _frame_order(self, scores: List[float]) -> List[int]:
        """Per-frame resolution order. With bidirectional relink, process higher
        confidence detections first so the long-lived incumbent claims its
        canonical id before a low-confidence re-appearing fragment, which then
        splits via the uniqueness guard. Otherwise keep detector order so tuned
        non-bidirectional baselines stay bit-identical.
        """
        if not self.bidirectional:
            return list(range(len(scores)))
        return sorted(range(len(scores)), key=lambda i: -float(scores[i]))

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
        order = self._frame_order(scores)
        out: List[int] = [0] * len(raw_ids)
        for i in order:
            out[i] = self.resolve(
                raw_ids[i],
                embeddings[i],
                boxes[i],
                scores[i],
                frame_id,
                w,
                h,
                assigned,
            )
        return out

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
            "reject_kalman={reject_kalman} reject_direction={reject_direction} "
            "reject_speed={reject_speed} reject_backward={reject_backward} accept_bidir={accept_bidir} "
            "reject_biometric={reject_biometric} split_collision={relink_split_collision}".format(
                **self.stats
            )
        )
        print(
            "  reject_age_fresh={reject_age_too_fresh} reject_age_old={reject_age_too_old} "
            "reject_quality_score={reject_quality_score} reject_quality_margin={reject_quality_margin} "
            "reject_quality_aspect_low={reject_quality_aspect_low} "
            "reject_quality_aspect_high={reject_quality_aspect_high} "
            "appearance_first_bypass={appearance_first_bypass} "
            "delayed_claim_pending={delayed_claim_pending} "
            "delayed_claim_ready={delayed_claim_ready} "
            "delayed_claim_accepted={delayed_claim_accepted}".format(**self.stats)
        )
        print(
            "  motion={motion_bonus_total} bonus_motion_ok={motion_iou_gate_pass} "
            "motion_cons_fail={motion_consistency_fail} motion_only={motion_only_match} "
            "motion_registry_size={motion_registry_size}".format(
                motion_bonus_total=self.stats.get("motion_bonus_total", 0),
                motion_iou_gate_pass=self.stats.get("motion_iou_gate_pass", 0),
                motion_consistency_fail=self.stats.get("motion_consistency_fail", 0),
                motion_only_match=self.stats.get("motion_only_match", 0),
                motion_registry_size=len(self.motion_registry),
            )
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
