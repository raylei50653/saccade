from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class CoreConfig:
    # I/O
    engine: str = "models/yolo/yolo26m_960_batch1.engine"
    pose_engine: str | None = None
    output: str = "results/MOT17_eval"
    data_root: str = "datasets/MOT17"
    split: str = "train"
    sequences: str = ""
    max_frames: int | None = None
    debug_dump_seq: str = ""
    debug_dump_frames: str = ""
    debug_dump_csv: str = ""
    detector: str | None = None
    config: str | None = None
    preset: str | None = None
    # Core thresholds
    conf_threshold: float = 0.05
    track_thresh: float = 0.05
    high_thresh: float = 0.45
    mid_thresh: float = 0.10
    new_track_thresh: float = 0.35
    match_thresh: float = 0.75
    # Confirmation
    confirm_streak: int = 1
    confirm_score_thresh: float = 0.0
    adaptive_confirmation: bool = False
    # GMC (basic controls — advanced GMC in reid module)
    gmc: bool = True
    gmc_mode: str = "gpu"
    gmc_downscale: int = 8
    # Misc
    per_seq_adapt: bool = True
    warmup_frames: int = 50
    profile_stages: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CoreConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_core_args(parser: argparse.ArgumentParser) -> None:
    io = parser.add_argument_group(_tier("I/O and dataset scope", "Tier 1"))
    io.description = (
        "Daily job-scope controls: engine choice, split, and sequence slice. "
        "These define what to run, not how tracking behaves."
    )
    io.add_argument(
        "--engine",
        default="models/yolo/yolo26m_960_batch1.engine",
        help="TensorRT detector engine path.",
    )
    io.add_argument(
        "--pose-engine",
        default=None,
        help="Optional TensorRT pose engine path for two-stage inference.",
    )
    io.add_argument(
        "--output",
        default="results/MOT17_eval",
        help="Directory for outputs, metrics, and dumps.",
    )
    io.add_argument("--data-root", default="datasets/MOT17", help="MOT17 dataset root.")
    io.add_argument(
        "--split",
        default="train",
        help=_help("Dataset split name.", range_hint="train/val/test"),
    )
    io.add_argument(
        "--sequences",
        default="",
        help="Comma-separated sequence names. Empty means all in split.",
    )
    io.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=_help(
            "Per-sequence frame cap for smoke tests.",
            range_hint=">=1 or unset",
            edge="small caps under-sample relink and long-gap behavior",
        ),
    )
    io.add_argument(
        "--debug-dump-seq",
        default="",
        help="Sequence name whose detection stages should be dumped.",
    )
    io.add_argument(
        "--debug-dump-frames",
        default="",
        help="Frame list/ranges for stage dump, e.g. 172-235,300,301.",
    )
    io.add_argument(
        "--debug-dump-csv",
        default="",
        help="CSV path for raw/post_filter/post_nms/post_merge box dumps.",
    )
    io.add_argument(
        "--detector",
        choices=["SDP", "DPM", "FRCNN"],
        default=None,
        help="Filter sequences by detector suffix.",
    )
    io.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML config file. Values override argparse defaults; CLI overrides config.",
    )
    io.add_argument(
        "--preset",
        choices=("baseline", "accuracy", "speed"),
        default=None,
        help="Built-in preset from configs/presets/<name>.yaml.",
    )

    # Module file flags (opt-in per module)
    mod = parser.add_argument_group("Module config files (opt-in)")
    mod.description = (
        "Load per-module YAML configs from configs/modules/. "
        "Only specify the modules you're actively tuning."
    )
    mod.add_argument(
        "--module-detection",
        default=None,
        metavar="PATH",
        help="YAML for detection/preprocessing params (tiling, gamma, FP filter, …).",
    )
    mod.add_argument(
        "--module-geometry",
        default=None,
        metavar="PATH",
        help="YAML for geometry priors, ID stability, Kalman, quality scaling.",
    )
    mod.add_argument(
        "--module-reid",
        default=None,
        metavar="PATH",
        help="YAML for ReID backbone, budget, crop, lazy-ReID, async pipeline.",
    )
    mod.add_argument(
        "--module-semantic",
        default=None,
        metavar="PATH",
        help="YAML for semantic relink, appearance bank, clean filter.",
    )
    mod.add_argument(
        "--module-trigger",
        default=None,
        metavar="PATH",
        help="YAML for dynamic ReID trigger policy (Experimental).",
    )
    mod.add_argument(
        "--module-lifecycle",
        default=None,
        metavar="PATH",
        help="YAML for lifecycle merge, post-merge, tracklet cleanup (Experimental).",
    )

    core = parser.add_argument_group(_tier("Core tracking and thresholds", "Tier 1"))
    core.description = (
        "Primary thresholds and association controls — tune every ablation."
    )
    core.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help=_help(
            "Detector confidence floor.",
            range_hint="0-1, usually 0.01-0.3",
            edge="lower increases recall/noise; higher suppresses weak people",
        ),
    )
    core.add_argument(
        "--track-thresh",
        type=float,
        default=0.05,
        help=_help(
            "Low-confidence association floor.",
            range_hint="0-1, usually <= high-thresh",
        ),
    )
    core.add_argument(
        "--high-thresh",
        type=float,
        default=0.45,
        help=_help(
            "High-confidence matching threshold.",
            range_hint="0-1",
            edge="higher reduces false links but raises fragmentation",
        ),
    )
    core.add_argument(
        "--mid-thresh",
        type=float,
        default=0.10,
        help=_help(
            "Mid-tier confidence bucket.",
            range_hint="0-1",
            edge="keep between track-thresh and high-thresh",
        ),
    )
    core.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.35,
        help=_help(
            "Minimum score for starting a new track.",
            range_hint="0-1",
            edge="higher suppresses weak true positives",
        ),
    )
    core.add_argument(
        "--match-thresh",
        type=float,
        default=0.75,
        help=_help(
            "Association similarity gate.",
            range_hint="0-1",
            edge="near 1 favors purity over continuity",
        ),
    )
    core.add_argument(
        "--confirm-streak",
        type=int,
        default=1,
        help=_help(
            "Hits needed before confirming a tentative track.",
            range_hint=">=1",
            edge="higher removes flicker but delays births",
        ),
    )
    core.add_argument(
        "--confirm-score-thresh",
        type=float,
        default=0.0,
        help=_help("Minimum score for counting toward confirmation.", range_hint="0-1"),
    )
    core.add_argument(
        "--adaptive-confirmation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable score/geometry-aware confirmation logic.",
    )
    core.add_argument(
        "--gmc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable global motion compensation.",
    )
    core.add_argument(
        "--gmc-mode",
        choices=["cpu", "gpu"],
        default="gpu",
        help="GMC algorithm mode. 'cpu' uses OpenCV LK; 'gpu' uses cuFFT phase correlation.",
    )
    core.add_argument(
        "--gmc-downscale",
        type=int,
        default=8,
        help=_help(
            "Downscale factor for GMC estimation.",
            range_hint=">=1",
            edge="too large loses fine camera motion",
        ),
    )
    core.add_argument(
        "--per-seq-adapt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-scale time-like tracker parameters by sequence frameRate.",
    )
    core.add_argument(
        "--warmup-frames",
        type=int,
        default=50,
        help=_help("Frames to ignore for warmup effects.", range_hint=">=0"),
    )
    core.add_argument(
        "--profile-stages", action="store_true", help="Profile stage-level runtime."
    )
