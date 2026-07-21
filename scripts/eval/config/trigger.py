"""Trigger/profile switch config fields for mot17 eval."""

# status: stable
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class TriggerConfig:
    # Enable
    need_reid: bool = False
    # History
    reid_history_size: int = 5
    reid_trigger_mode: str = "score_ema"
    # Long memory
    reid_long_memory_decay: float = 0.80
    reid_long_memory_trigger: float = 1.25
    # Score EMA
    reid_score_decay: float = 0.80
    reid_score_threshold: float = 2.0
    reid_score_threshold_low: float = 2.0
    # Signal weights
    reid_weight_new: float = 1.0
    reid_weight_lost: float = 1.4
    reid_weight_geom: float = 0.5
    reid_weight_conf: float = 0.5
    # Birth/death events
    reid_birth_death_boost: float = 1.0
    reid_birth_death_lost_min: float = 0.1
    # Lost age cap
    reid_lost_age_cap: int = 30
    # Instability scoring
    reid_unstable_shift_weight: float = 1.0
    reid_unstable_iou_weight: float = 1.0
    # Confidence jitter
    reid_conf_jitter_gate: float = 0.10
    # Persistence + cooldown
    reid_trigger_persist_frames: int = 2
    reid_cooldown_frames: int = 4
    # Smoothing
    reid_geom_smooth_window: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TriggerConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_trigger_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group(
        _tier("Dynamic ReID trigger policy", "Experimental")
    )
    grp.description = (
        "Experimental controls for when appearance extraction fires. "
        "Load with: --module-trigger configs/modules/trigger.yaml"
    )
    grp.add_argument(
        "--need-reid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic need_reid_frame() triggering instead of fixed heartbeat.",
    )
    grp.add_argument(
        "--reid-history-size",
        type=int,
        default=5,
        help=_help("BBox history length for trigger state.", range_hint=">=1"),
    )
    grp.add_argument(
        "--reid-trigger-mode",
        choices=(
            "count_jump",
            "event_any",
            "event_persist",
            "event_strict",
            "event_memory",
            "score_ema",
            "score_ema_geom",
            "score_ema_conf",
        ),
        default="score_ema",
        help="Dynamic ReID trigger logic.",
    )
    grp.add_argument(
        "--reid-long-memory-decay",
        type=float,
        default=0.80,
        help=_help("EMA decay for long-memory trigger state.", range_hint="0-1"),
    )
    grp.add_argument(
        "--reid-long-memory-trigger",
        type=float,
        default=1.25,
        help=_help("Trigger threshold for long-memory mode.", range_hint=">0"),
    )
    grp.add_argument(
        "--reid-score-decay",
        type=float,
        default=0.80,
        help=_help("EMA decay for weighted trigger signals.", range_hint="0-1"),
    )
    grp.add_argument(
        "--reid-score-threshold",
        type=float,
        default=2.0,
        help=_help(
            "Fire trigger when weighted score crosses this threshold.", range_hint=">0"
        ),
    )
    grp.add_argument(
        "--reid-score-threshold-low",
        type=float,
        default=2.0,
        help=_help(
            "Lower hysteresis threshold for staying active; unset reuses high threshold.",
            range_hint=">=0 or unset",
        ),
    )
    grp.add_argument(
        "--reid-weight-new",
        type=float,
        default=1.0,
        help=_help("Weight for new-track trigger signal.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-weight-lost",
        type=float,
        default=1.4,
        help=_help("Weight for lost-track trigger signal.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-weight-geom",
        type=float,
        default=0.5,
        help=_help("Weight for geometry-instability trigger signal.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-weight-conf",
        type=float,
        default=0.5,
        help=_help("Weight for confidence-jitter trigger signal.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-birth-death-boost",
        type=float,
        default=1.0,
        help=_help(
            "Extra score when birth and death happen together.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--reid-birth-death-lost-min",
        type=float,
        default=0.1,
        help=_help(
            "Minimum lost-signal before birth/death boost applies.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--reid-lost-age-cap",
        type=int,
        default=30,
        help=_help("Saturating age cap for lost-track weighting.", range_hint=">=1"),
    )
    grp.add_argument(
        "--reid-unstable-shift-weight",
        type=float,
        default=1.0,
        help=_help("Shift contribution weight in instability score.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-unstable-iou-weight",
        type=float,
        default=1.0,
        help=_help(
            "IoU-drop contribution weight in instability score.", range_hint=">=0"
        ),
    )
    grp.add_argument(
        "--reid-conf-jitter-gate",
        type=float,
        default=0.10,
        help=_help("Deadzone for per-track confidence jitter.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-trigger-persist-frames",
        type=int,
        default=2,
        help=_help(
            "Frames score must stay high before trigger fires.", range_hint=">=1"
        ),
    )
    grp.add_argument(
        "--reid-cooldown-frames",
        type=int,
        default=4,
        help=_help("Frames to suppress ReID after each trigger.", range_hint=">=0"),
    )
    grp.add_argument(
        "--reid-geom-smooth-window",
        type=int,
        default=1,
        help=_help(
            "Moving-average window before geometry-instability scoring.",
            range_hint=">=1",
        ),
    )
