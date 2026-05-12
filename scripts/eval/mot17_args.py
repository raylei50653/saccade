import argparse
import sys
from pathlib import Path

# Allow running from scripts/eval/ directly
_config_dir = Path(__file__).resolve().parent / "config"
if str(_config_dir.parent) not in sys.path:
    sys.path.insert(0, str(_config_dir.parent))

from config import (  # noqa: E402
    add_core_args,
    add_detection_args,
    add_geometry_args,
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
    add_reid_args(parser)
    add_semantic_args(parser)
    add_trigger_args(parser)
    add_lifecycle_args(parser)
    return parser
