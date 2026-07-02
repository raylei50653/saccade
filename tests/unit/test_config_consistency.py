"""Guard against silent divergence between the two CLI config sub-layers.

`scripts/eval/config/*.py` declares each knob twice: as a dataclass field
(`LifecycleConfig.relink_bridge_h_lo`) and as an argparse argument
(`--relink-bridge-h-lo`). The runtime reads the *argparse* namespace; the
dataclass fields are a typed mirror that nothing on the hot path consumes, so a
mismatch is invisible until someone trusts the wrong one.

These tests pin the two layers together:
  * every dataclass field that is also an argparse dest must share its default
    (caught a real 0.0-vs-0.75 divergence on relink_bridge_h_lo/h_hi, plus
    relink_bridge_margin, semantic_bridge_px, async_reid, pipeline_relink);
  * every dataclass field name must be a real argparse dest (a rename on one
    side would otherwise silently fall back to the EvalConfig default).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import yaml

_eval_dir = Path(__file__).resolve().parents[2] / "scripts" / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from config import (  # noqa: E402
    CoreConfig,
    DetectionConfig,
    GeometryConfig,
    LifecycleConfig,
    MotionConfig,
    ReIDConfig,
    SemanticConfig,
    TriggerConfig,
)
from mot17_args import build_parser  # noqa: E402

_MODULES = (
    CoreConfig,
    DetectionConfig,
    GeometryConfig,
    MotionConfig,
    ReIDConfig,
    SemanticConfig,
    TriggerConfig,
    LifecycleConfig,
)


def _argparse_defaults() -> dict[str, object]:
    parser = build_parser()
    return {a.dest: a.default for a in parser._actions if a.dest != "help"}


def test_dataclass_defaults_match_argparse() -> None:
    arg = _argparse_defaults()
    mismatches = []
    for mod in _MODULES:
        for f in dataclasses.fields(mod):
            if f.default is dataclasses.MISSING:
                continue
            if f.name in arg and arg[f.name] != f.default:
                mismatches.append(
                    f"{mod.__name__}.{f.name}: dataclass={f.default!r} "
                    f"argparse={arg[f.name]!r}"
                )
    assert not mismatches, "dataclass/argparse default divergence:\n" + "\n".join(
        mismatches
    )


def test_every_field_has_an_argparse_dest() -> None:
    arg = _argparse_defaults()
    orphans = []
    for mod in _MODULES:
        for f in dataclasses.fields(mod):
            if f.name not in arg:
                orphans.append(f"{mod.__name__}.{f.name}")
    assert not orphans, (
        "dataclass fields with no argparse argument (rename would silently "
        "fall back to the EvalConfig default):\n" + "\n".join(orphans)
    )


def test_native_whole_graph_preset_does_not_select_mamba_head() -> None:
    preset = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "presets"
        / "native_whole_graph.yaml"
    )
    with preset.open() as f:
        loaded = yaml.safe_load(f) or {}

    assert loaded.get("mamba_ckpt") == ""
