#!/usr/bin/env python3
# mypy: ignore-errors
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
from saccade.perception.eval.runner import run_eval  # noqa: E402

import yaml  # noqa: E402
from mot17_args import build_parser  # noqa: E402


def _load_config_defaults(project_root: Path) -> dict:
    """Pre-parse --config / --preset / --module-* and return merged defaults dict.

    Priority (lowest → highest):
      argparse defaults < config file < module YAMLs < preset < CLI flags.

    Only yaml.safe_load is used; no dynamic import or eval.
    """
    _MODULE_FLAGS = [
        "module_detection",
        "module_geometry",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    ]

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--preset", default=None)
    for flag in _MODULE_FLAGS:
        pre.add_argument(f"--{flag.replace('_', '-')}", default=None, dest=flag)
    pre_args, _ = pre.parse_known_args()

    defaults: dict = {}

    # 1. Core config file (or fallback baseline)
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        with config_path.open() as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    elif not pre_args.preset:
        fallback = project_root / "configs" / "mot17_baseline.yaml"
        if fallback.exists():
            with fallback.open() as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update(loaded)

    # 2. Per-module YAML files (opt-in)
    for flag in _MODULE_FLAGS:
        path_str = getattr(pre_args, flag, None)
        if path_str:
            module_path = Path(path_str)
            if not module_path.is_absolute():
                module_path = Path.cwd() / module_path
            with module_path.open() as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update(loaded)

    # 3. Preset (highest priority among file-based configs)
    if pre_args.preset:
        preset_path = project_root / "configs" / "presets" / f"{pre_args.preset}.yaml"
        with preset_path.open() as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)

    return defaults


if __name__ == "__main__":
    parser = build_parser()
    config_defaults = _load_config_defaults(project_root)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.detector and not args.sequences:
        from pathlib import Path as _Path

        data_root = _Path(args.data_root)
        split_dir = data_root / args.split
        if split_dir.exists():
            args.sequences = ",".join(
                sorted(
                    d.name
                    for d in split_dir.iterdir()
                    if d.is_dir() and d.name.endswith(f"-{args.detector}")
                )
            )
    _MODULE_KEYS = {
        "module_detection",
        "module_geometry",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    }
    eval_kwargs = {k: v for k, v in vars(args).items() if k not in _MODULE_KEYS}
    metrics = run_eval(**eval_kwargs)
    if metrics:
        print("\n=== OVERALL METRICS ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
