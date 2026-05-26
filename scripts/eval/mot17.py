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
from saccade.perception.eval.evaluator import run_eval_cpp  # noqa: E402
from mlflow_logger import log_eval_run  # noqa: E402

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
        "module_motion",
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
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    }
    _MAMBA_KEYS = {"mamba_ckpt", "mamba_teacher_ckpt"}
    eval_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in _MODULE_KEYS and k not in _MAMBA_KEYS
    }

    if getattr(args, "mamba_ckpt", None):
        print(f"\n🧠 [Mamba] Loading Mamba head from {args.mamba_ckpt}")
        from saccade.perception.temporal_yolo.mamba_gated_detector import (
            build_mamba_gated_detector,
        )

        mamba_detector = build_mamba_gated_detector(
            yolo_pt_path="models/yolo/yolo26s.pt",
            teacher_ckpt=args.mamba_teacher_ckpt,
            mamba_ckpt=args.mamba_ckpt,
            img_size=960,
            device="cuda",
            conf_thr=0.001,
            max_det=300,
        )
        eval_kwargs["detector"] = mamba_detector
        eval_kwargs["tiling"] = "mamba_960"
        eval_kwargs["engine"] = "mamba"

    if getattr(args, "cpp_threads", 0) > 0:
        n = args.cpp_threads
        print(f"🚀 [C++ EvaluatorPool] Running with {n} threads")
        metrics = run_eval_cpp(
            engine=args.engine,
            output=args.output,
            data_root=args.data_root,
            split=args.split,
            sequences=args.sequences or "",
            n_threads=n,
            **{
                k: v
                for k, v in eval_kwargs.items()
                if k
                not in {
                    "engine",
                    "output",
                    "data_root",
                    "split",
                    "sequences",
                    "workbench",
                    "threads",
                    "cpp_threads",
                }
            },
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    else:
        metrics = run_eval(**eval_kwargs)
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

    if metrics:
        try:
            tags = {}
            if getattr(args, "preset", None):
                tags["preset"] = args.preset
            if getattr(args, "detector", None):
                tags["detector"] = args.detector
            log_eval_run(
                uri=args.mlflow_uri,
                experiment_name=args.mlflow_experiment,
                run_name=args.mlflow_run_name,
                params={k: v for k, v in vars(args).items() if k not in _MODULE_KEYS},
                metrics=metrics,
                tags=tags,
            )
        except Exception as e:
            print(f"[mlflow] ERROR: {e}")
