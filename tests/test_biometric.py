"""Unit tests for biometric ratio computation and accumulator."""

import math
import torch

from saccade.perception.biometric import (
    leg_body_ratio,
    shoulder_width_ratio,
    head_body_ratio,
    BiometricAccumulator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_kpts(overrides: dict[int, tuple[float, float, float]]) -> torch.Tensor:
    """Build a single [1, 17, 3] keypoint tensor from (x, y, vis) overrides."""
    kpts = torch.zeros(1, 17, 3)
    for idx, (x, y, v) in overrides.items():
        kpts[0, idx] = torch.tensor([x, y, v])
    return kpts


# ──────────────────────────────────────────────────────────────────────────────
# leg_body_ratio
# ──────────────────────────────────────────────────────────────────────────────


class TestLegBodyRatio:
    def test_simple_vertical_person(self):
        # shoulder y=0, hip y=50, ankle y=150 → leg=100, torso=50 → R=2.0
        kpts = _make_kpts(
            {
                5: (0, 0, 1.0),
                6: (0, 0, 1.0),  # shoulders at y=0
                11: (0, 50, 1.0),
                12: (0, 50, 1.0),  # hips at y=50
                15: (0, 150, 1.0),
                16: (0, 150, 1.0),  # ankles at y=150
            }
        )
        r = leg_body_ratio(kpts)
        assert abs(float(r[0]) - 2.0) < 1e-4

    def test_nan_when_ankle_invisible(self):
        kpts = _make_kpts(
            {
                5: (0, 0, 1.0),
                6: (0, 0, 1.0),
                11: (0, 50, 1.0),
                12: (0, 50, 1.0),
                15: (0, 150, 0.1),
                16: (0, 150, 0.1),  # low visibility
            }
        )
        r = leg_body_ratio(kpts)
        assert math.isnan(float(r[0]))

    def test_scale_invariance(self):
        # Same proportions, 2× scale → same ratio
        def make(scale: float) -> torch.Tensor:
            return _make_kpts(
                {
                    5: (0, 0, 1.0),
                    6: (0, 0, 1.0),
                    11: (0, 50 * scale, 1.0),
                    12: (0, 50 * scale, 1.0),
                    15: (0, 150 * scale, 1.0),
                    16: (0, 150 * scale, 1.0),
                }
            )

        r1 = float(leg_body_ratio(make(1.0))[0])
        r2 = float(leg_body_ratio(make(2.0))[0])
        assert abs(r1 - r2) < 1e-4

    def test_batch(self):
        # Two people with different leg proportions
        kpts = torch.zeros(2, 17, 3)
        for i, (hip_y, ankle_y) in enumerate([(50, 150), (50, 120)]):
            for j in [5, 6]:
                kpts[i, j] = torch.tensor([0.0, 0.0, 1.0])
            for j in [11, 12]:
                kpts[i, j] = torch.tensor([0.0, float(hip_y), 1.0])
            for j in [15, 16]:
                kpts[i, j] = torch.tensor([0.0, float(ankle_y), 1.0])
        r = leg_body_ratio(kpts)
        assert abs(float(r[0]) - 2.0) < 1e-4
        assert abs(float(r[1]) - 1.4) < 1e-4


# ──────────────────────────────────────────────────────────────────────────────
# shoulder_width_ratio
# ──────────────────────────────────────────────────────────────────────────────


class TestShoulderWidthRatio:
    def test_basic(self):
        # shoulders: x=±20 y=0, hips: y=50 → width=40, torso=50 → R=0.8
        kpts = _make_kpts(
            {
                5: (-20, 0, 1.0),
                6: (20, 0, 1.0),
                11: (0, 50, 1.0),
                12: (0, 50, 1.0),
            }
        )
        r = shoulder_width_ratio(kpts)
        assert abs(float(r[0]) - 0.8) < 1e-4

    def test_nan_when_hip_invisible(self):
        kpts = _make_kpts(
            {
                5: (-20, 0, 1.0),
                6: (20, 0, 1.0),
                11: (0, 50, 0.1),
                12: (0, 50, 0.1),
            }
        )
        r = shoulder_width_ratio(kpts)
        assert math.isnan(float(r[0]))


# ──────────────────────────────────────────────────────────────────────────────
# head_body_ratio
# ──────────────────────────────────────────────────────────────────────────────


class TestHeadBodyRatio:
    def test_basic(self):
        # nose at y=-20, shoulder at y=0, hip at y=50 → torso=50, head_seg=20 → R=2.5
        kpts = _make_kpts(
            {
                0: (0, -20, 1.0),
                5: (0, 0, 1.0),
                6: (0, 0, 1.0),
                11: (0, 50, 1.0),
                12: (0, 50, 1.0),
            }
        )
        r = head_body_ratio(kpts)
        assert abs(float(r[0]) - 2.5) < 1e-4


# ──────────────────────────────────────────────────────────────────────────────
# BiometricAccumulator
# ──────────────────────────────────────────────────────────────────────────────


class TestBiometricAccumulator:
    def _visible_kpts(self, leg_ratio: float = 2.0) -> torch.Tensor:
        """[1, 17, 3] with the given leg_body_ratio (torso=50 fixed)."""
        ankle_y = 50 + 50 * leg_ratio
        return _make_kpts(
            {
                5: (0, 0, 1.0),
                6: (0, 0, 1.0),
                11: (0, 50, 1.0),
                12: (0, 50, 1.0),
                15: (0, ankle_y, 1.0),
                16: (0, ankle_y, 1.0),
            }
        )

    def test_none_before_window_filled(self):
        acc = BiometricAccumulator(window=10)
        ids = torch.tensor([1])
        kpts = self._visible_kpts()
        for _ in range(4):  # < window//2
            acc.update(ids, kpts)
        assert acc.get(1) is None

    def test_median_after_enough_frames(self):
        acc = BiometricAccumulator(window=10)
        ids = torch.tensor([1])
        kpts = self._visible_kpts(leg_ratio=2.0)
        for _ in range(8):
            acc.update(ids, kpts)
        result = acc.get(1)
        assert result is not None
        assert abs(result["r_leg"] - 2.0) < 0.01

    def test_median_filters_nan_frames(self):
        acc = BiometricAccumulator(window=20)
        ids = torch.tensor([1])
        good = self._visible_kpts(leg_ratio=2.0)
        bad = _make_kpts(
            {  # ankles invisible
                5: (0, 0, 1.0),
                6: (0, 0, 1.0),
                11: (0, 50, 1.0),
                12: (0, 50, 1.0),
                15: (0, 150, 0.1),
                16: (0, 150, 0.1),
            }
        )
        for _ in range(6):
            acc.update(ids, good)
        for _ in range(4):
            acc.update(ids, bad)
        result = acc.get(1)
        assert result is not None
        assert abs(result["r_leg"] - 2.0) < 0.01

    def test_evict_stale(self):
        acc = BiometricAccumulator(window=10)
        ids = torch.tensor([1, 2])
        kpts = torch.cat([self._visible_kpts(), self._visible_kpts()], dim=0)
        for _ in range(8):
            acc.update(ids, kpts)
        acc.evict_stale(active_ids={1})
        assert acc.get(2) is None
        assert acc.get(1) is not None

    def test_window_sliding(self):
        acc = BiometricAccumulator(window=5)
        ids = torch.tensor([1])
        # Feed 3 frames with ratio 2.0, then 5 frames with ratio 3.0
        for _ in range(3):
            acc.update(ids, self._visible_kpts(2.0))
        for _ in range(5):
            acc.update(ids, self._visible_kpts(3.0))
        result = acc.get(1)
        assert result is not None
        # Window=5 → only last 5 frames kept → median should be 3.0
        assert abs(result["r_leg"] - 3.0) < 0.01
