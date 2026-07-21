#!/usr/bin/env python3
"""Phase 1.5C: Generate golden config snapshots for CI.

Produces tests/fixtures/golden_config_*.json — a complete flat snapshot
of every runtime argparse parameter resolved after YAML preset merging.

These snapshots guard Phase 2 refactoring: any change to default values,
field names, or merge order will cause a diff against the golden file.

Usage:
    uv run python scripts/eval/config/gen_golden_snapshot.py
"""
# status: stable

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ / "src"))
sys.path.insert(0, str(_PROJ / "scripts" / "eval"))

import yaml
from mot17_args import build_parser

_FIXTURE_DIR = _PROJ / "tests" / "fixtures"

_ALLOWED_NON_RUNTIME = {
    "config",
    "cpp_threads",
    "detect_barrier",
    "detector",
    "double_buffer",
    "fpn_backbone_engine",
    "gpu_decode",
    "main_nms_graphed",
    "mamba_ckpt",
    "mamba_head_engine",
    "mamba_small_p3_max_threshold",
    "mamba_teacher_ckpt",
    "mamba_trt",
    "mamba_yolo_weights",
    "max_frames",
    "mlflow_experiment",
    "mlflow_run_name",
    "mlflow_uri",
    "module_detection",
    "module_geometry",
    "module_lifecycle",
    "module_motion",
    "module_reid",
    "module_semantic",
    "module_trigger",
    "no_compile",
    "no_gpu_decode",
    "no_temporal",
    "no_visualize_score",
    "preset",
    "processes",
    "score_on_gt_frames",
    "teacher_head_backbone_engine",
    "teacher_head_ckpt",
    "teacher_head_whole_graph",
    "use_cuda_graph",
    "use_tracker_graph",
    "use_whole_graph",
    "visualize",
    "visualize_fps",
    "visualize_scale",
    "visualize_trail_len",
    "warmup_frames",
}


def _load_preset_overrides(preset_name: str) -> dict:
    path = _PROJ / "configs" / "presets" / f"{preset_name}.yaml"
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_config(preset_name: str | None = None) -> dict[str, object]:
    """Build the argparse parser, apply YAML defaults, and return all runtime dests."""
    parser = build_parser()

    # Apply fallback baseline
    baseline_path = _PROJ / "configs" / "mot17_baseline.yaml"
    if baseline_path.exists():
        with baseline_path.open() as f:
            defaults = yaml.safe_load(f) or {}
        # motion is nested key in baseline yaml, flatten it
        if "motion" in defaults and isinstance(defaults["motion"], dict):
            defaults.update(defaults.pop("motion"))
        parser.set_defaults(**defaults)

    # Apply preset overrides
    if preset_name:
        preset = _load_preset_overrides(preset_name)
        parser.set_defaults(**preset)

    # Parse with no CLI args
    args, _ = parser.parse_known_args([])

    # Filter to runtime-only
    result = {}
    for k, v in vars(args).items():
        if k not in _ALLOWED_NON_RUNTIME:
            if v is None:
                result[k] = None
            elif isinstance(v, (bool, int, float, str)):
                result[k] = v
            elif isinstance(v, list):
                result[k] = v
            elif isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = str(v)
    return result


def _json_serializable(obj: object) -> object:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and obj != obj:  # NaN
            return None
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(x) for x in obj]
    return str(obj)


def main():
    presets = {
        "baseline": "mot17_baseline.yaml (no preset)",
        "mamba_whole_graph": "Production preset",
    }

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for name, desc in presets.items():
        preset_arg = name if name != "baseline" else None
        config = _resolve_config(preset_arg)

        # Build clean snapshot
        snapshot = {
            "_meta": {
                "description": desc,
                "preset": name,
                "field_count": len(config),
                "generated_by": "scripts/eval/config/gen_golden_snapshot.py",
            },
            "config": {k: _json_serializable(v) for k, v in sorted(config.items())},
        }

        out_path = _FIXTURE_DIR / f"golden_config_{name}.json"
        with out_path.open("w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=False)

        print(f"Wrote {out_path} ({len(config)} fields)")

    # Also generate a flat baseline-only snapshot (no preset, no yaml)
    parser = build_parser()
    args, _ = parser.parse_known_args([])
    raw = {}
    for k, v in vars(args).items():
        if k not in _ALLOWED_NON_RUNTIME:
            raw[k] = _json_serializable(v)

    raw_snapshot = {
        "_meta": {
            "description": "Pure argparse defaults (no YAML, no preset)",
            "field_count": len(raw),
            "generated_by": "scripts/eval/config/gen_golden_snapshot.py",
        },
        "config": {k: v for k, v in sorted(raw.items())},
    }
    raw_path = _FIXTURE_DIR / "golden_config_argparse_raw.json"
    with raw_path.open("w") as f:
        json.dump(raw_snapshot, f, indent=2, sort_keys=False)
    print(f"Wrote {raw_path} ({len(raw)} fields)")


if __name__ == "__main__":
    main()
