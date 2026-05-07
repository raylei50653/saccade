import torch
from typing import Optional
from collections import deque
from .tracker_gpu import ReIDTrackObservation, ReIDFrameStats

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
        self._last_new_ids: set[int] = set()
        self._last_lost_ids: set[int] = set()
        self._per_track_instability: dict[int, float] = {}
        self._per_track_conf_jitter: dict[int, float] = {}
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
        self._per_track_instability = {}
        self._per_track_conf_jitter = {}

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
                self._per_track_instability[tid] = instability

            prev_score_ema = self._track_score_ema.get(tid, prev[tid].det_score)
            conf_jitter = max(
                0.0, abs(tracks[tid].det_score - prev_score_ema) - self.conf_jitter_gate
            )
            if conf_jitter > 0.0:
                conf_signal += conf_jitter
                self._per_track_conf_jitter[tid] = conf_jitter

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

        self._last_new_ids = curr_ids - prev_ids
        self._last_lost_ids = prev_ids - curr_ids
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

    def get_track_priorities(
        self,
        tracks: dict[int, ReIDTrackObservation],
        weight_recovery: float = 2.0,
    ) -> dict[int, float]:
        """Compute per-track priority scores for budgeted ReID (A3 Phase 1)."""
        priorities: dict[int, float] = {}
        # birth_death_boost applies only to NEW tracks if event is active.
        bd_boost = self._last_birth_death_boost

        for tid, obs in tracks.items():
            priority = 0.0
            if tid in self._last_new_ids:
                # 1. New Track: det_score base + birth_death boost
                priority = self.weight_new * obs.det_score + bd_boost
            else:
                # 2. Matched tracks
                instability = self._per_track_instability.get(tid, 0.0)
                jitter = self._per_track_conf_jitter.get(tid, 0.0)
                priority = self.weight_geom * instability + self.weight_conf * jitter

                # 3. Recovery boost: if this track was lost in the very recent history
                # We can check if its age > 1 (meaning it's not brand new) but it was not
                # matched in the immediately preceding frame.
                age = self._track_ages.get(tid, 1)
                # Note: if it's in shared_ids (it was in prev frame), then it's NOT recovered.
                # If it's in tracks but NOT in _last_new_ids, it MUST be recovered.
                is_recovered = (
                    tid not in self._last_new_ids
                    and len(self._track_history) >= 2
                    and tid not in self._track_history[-2]
                )
                if is_recovered:
                    recovery_factor = min(1.0, age / float(self.lost_age_cap))
                    priority += weight_recovery * recovery_factor

            # Stable tracks get a tiny base priority from their confidence.
            if priority <= 0.0:
                priority = 0.1 * obs.det_score

            priorities[tid] = priority

        return priorities

    def get_priorities(self) -> dict[int, float]:
        """Returns priorities for tracks based on state as of the last observed frame."""
        priorities = {}
        
        # 1. New tracks (born in the frame just observed)
        # We want to ReID them in the NEXT frame to confirm they are stable.
        for tid in self._last_new_ids:
            priorities[tid] = self.weight_new * self._track_score_ema.get(tid, 0.5) + self._last_birth_death_boost
            
        # 2. Matched tracks (active in the frame just observed)
        # We prioritize those that were unstable or had high confidence jitter.
        for tid in self._track_ages:
            if tid in self._last_new_ids: continue
            instability = self._per_track_instability.get(tid, 0.0)
            jitter = self._per_track_conf_jitter.get(tid, 0.0)
            priority = self.weight_geom * instability + self.weight_conf * jitter
            if priority <= 0.0:
                priority = 0.1 * self._track_score_ema.get(tid, 0.5)
            priorities[tid] = priority
            
        # 3. Lost tracks (disappeared in the frame just observed)
        # We want to find them in the current frame.
        for tid in self._last_lost_ids:
            # We don't have a current EMA for them easily accessible if they just disappeared,
            # but we can assume they were important.
            priorities[tid] = self.weight_lost * 1.0 # Mature/Stable lost tracks
            
        return priorities

    def get_last_boxes(self) -> dict[int, tuple[float, float, float, float]]:
        """Returns the last known boxes for all tracks (active and just lost)."""
        boxes = {}
        if not self._track_history:
            return boxes
            
        # Active tracks from the most recent frame
        for tid, obs in self._track_history[-1].items():
            boxes[tid] = obs.box
            
        # Just lost tracks (were in history[-2] but not history[-1])
        if len(self._track_history) >= 2:
            prev = self._track_history[-2]
            curr = self._track_history[-1]
            for tid in set(prev) - set(curr):
                boxes[tid] = prev[tid].box
                
        return boxes

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


