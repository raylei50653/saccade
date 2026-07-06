import argparse
import os
import sys
from pathlib import Path
from typing import MutableMapping

# Allow running from scripts/eval/ directly
_config_dir = Path(__file__).resolve().parent / "config"
if str(_config_dir.parent) not in sys.path:
    sys.path.insert(0, str(_config_dir.parent))

from config import (  # noqa: E402
    add_core_args,
    add_detection_args,
    add_geometry_args,
    add_motion_args,
    add_reid_args,
    add_semantic_args,
    add_trigger_args,
    add_lifecycle_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MOT17 tracking runs. Parameters are grouped by module so "
            "--help shows which stage each knob affects. "
            "Load per-module YAML files with --module-<name> PATH to opt into "
            "advanced parameter sets without exposing them in every run. "
            "Tier legend: Tier 1 = daily knobs; Tier 2 = advanced tuning; "
            "Experimental = ablation-heavy or niche controls."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_core_args(parser)
    add_detection_args(parser)
    add_geometry_args(parser)
    add_motion_args(parser)
    add_reid_args(parser)
    add_semantic_args(parser)
    add_trigger_args(parser)
    add_lifecycle_args(parser)

    # Multi-processing
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Run Python multi-process evaluation with N processes (each running one sequence with GPU decode + CUDA Graph).",
    )

    # Decode backend
    dec = parser.add_argument_group("Decode backend")
    decode_choice = dec.add_mutually_exclusive_group()
    decode_choice.add_argument(
        "--gpu-decode",
        action="store_true",
        help=(
            "Use torchvision/nvJPEG GPU JPEG decode. This is the default decode "
            "domain for GPU-trained/calibrated runs."
        ),
    )
    decode_choice.add_argument(
        "--no-gpu-decode",
        action="store_true",
        help="Use CPU JPEG decode (DALI) for legacy CPU-decode baseline comparisons.",
    )
    parser.set_defaults(gpu_decode=True)

    runtime = parser.add_argument_group("Pipeline scheduling")
    runtime.add_argument(
        "--double-buffer",
        action="store_true",
        help=(
            "Enable detect(N+1) || tracker(N) overlap. Sets "
            "SACCADE_DOUBLE_BUFFER=1 and requires the event barrier."
        ),
    )
    runtime.add_argument(
        "--detect-barrier",
        choices=("full", "no_postproc", "event"),
        default=None,
        help=(
            "Override SACCADE_DETECT_BARRIER for this run. "
            "--double-buffer requires event."
        ),
    )

    # Tracking result visualization
    vis = parser.add_argument_group("Tracking result visualization")
    vis.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualized tracking video using scripts/tools/render_mot_result.py",
    )
    vis.add_argument(
        "--visualize-scale",
        type=float,
        default=0.5,
        help="Resize scale of the visualization video (default: 0.5 for speed)",
    )
    vis.add_argument(
        "--visualize-fps",
        type=int,
        default=None,
        help="FPS of the visualization video. Defaults to seqinfo.ini frameRate if not set.",
    )
    vis.add_argument(
        "--no-visualize-score",
        action="store_true",
        help="Hide confidence score labels in the visualization",
    )
    vis.add_argument(
        "--visualize-trail-len",
        type=int,
        nargs="?",
        const=30,
        default=30,
        help="Length of trajectory trail in frames (default: 30, set to 0 to disable)",
    )

    return parser


def configure_runtime_env(
    args: argparse.Namespace,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Apply CLI runtime choices to process env used by the eval pipeline.

    The lower-level evaluator reads a few scheduling/decode knobs from env for
    historical reasons.  Make the CLI entrypoint authoritative so a previous
    shell export (especially SACCADE_DETECT_BARRIER=event) cannot silently move
    a normal GPU-decode run onto the known drift-prone path.
    """

    env = os.environ if environ is None else environ

    if getattr(args, "double_buffer", False):
        if getattr(args, "detect_barrier", None) not in {None, "event"}:
            raise ValueError("--double-buffer requires --detect-barrier event")
        env["SACCADE_DOUBLE_BUFFER"] = "1"
        env["SACCADE_DETECT_BARRIER"] = "event"
    else:
        env["SACCADE_DOUBLE_BUFFER"] = "0"
        env["SACCADE_DETECT_BARRIER"] = (
            args.detect_barrier if getattr(args, "detect_barrier", None) else "full"
        )

    env["SACCADE_GPU_DECODE"] = "0" if getattr(args, "no_gpu_decode", False) else "1"
