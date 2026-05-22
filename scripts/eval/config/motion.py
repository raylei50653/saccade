from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _tier


@dataclasses.dataclass
class MotionConfig:
    """Motion-based relinking configuration."""

    # --- EMA smoothing ---
    vel_alpha: float = 0.3  # velocity EMA factor
    acc_alpha: float = 0.15  # acceleration EMA factor
    min_motion_observations: int = 2

    # --- Motion bonus (soft scoring) ---
    w_motion_iou: float = 0.3  # weight for motion IoU in cost function
    motion_consistency_check: bool = (
        True  # require velocity-consistent before motion bonus
    )
    consistency_tol: float = 2.0  # z-score tolerance for velocity check

    # --- Motion-only fallback ---
    enable_motion_only: bool = True
    motion_only_lost_frames: int = 5
    motion_only_iou_threshold: float = 0.15
    motion_only_min_lost_frames: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MotionConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_motion_args(parser: argparse.ArgumentParser) -> None:
    motion = parser.add_argument_group(_tier("Motion-based relinking", "Tier 2"))
    motion.description = (
        "Per-track motion model for improved gap closure and motion-based relinking. "
        "Motion evidence is used as soft scoring in the association cost function "
        "and as a fallback when ReID embeddings are unavailable."
    )

    ema = motion.add_argument_group("EMA smoothing")
    ema.add_argument(
        "--vel-alpha",
        type=float,
        default=0.3,
        help="Velocity EMA smoothing factor (0-1).",
    )
    ema.add_argument(
        "--acc-alpha",
        type=float,
        default=0.15,
        help="Acceleration EMA smoothing factor (0-1).",
    )
    ema.add_argument(
        "--min-motion-observations",
        type=int,
        default=2,
        help="Minimum observations before trusting motion predictions.",
    )

    bonus = motion.add_argument_group("Motion bonus (soft scoring)")
    bonus.add_argument(
        "--w-motion-iou",
        type=float,
        default=0.3,
        help="Weight for motion IoU bonus in the cost function.",
    )
    bonus.add_argument(
        "--motion-consistency-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require velocity-consistent check before applying motion bonus.",
    )
    bonus.add_argument(
        "--consistency-tol",
        type=float,
        default=2.0,
        help="Z-score tolerance for velocity consistency check.",
    )

    fallback = motion.add_argument_group("Motion-only fallback")
    fallback.add_argument(
        "--enable-motion-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable motion-only matching when ReID embeddings are unavailable.",
    )
    fallback.add_argument(
        "--motion-only-lost-frames",
        type=int,
        default=5,
        help="Max lost frames before requiring motion IoU threshold.",
    )
    fallback.add_argument(
        "--motion-only-iou-threshold",
        type=float,
        default=0.15,
        help="Min motion IoU for motion-only matching.",
    )
    fallback.add_argument(
        "--motion-only-min-lost-frames",
        type=int,
        default=1,
        help="Min lost frames before motion-only can trigger.",
    )
