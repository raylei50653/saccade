"""The headline head exporter binds to the preset and the committed lineage inventory.

``scripts/model/export_headline_mamba_head.py`` (#465 Phase B PR-1) refuses to
export unless the checkpoint the headline preset names is the inventory's
``s.t3t1_phase_b`` node and its bytes hash to that node's sha256. The hashing
needs gitignored ``runs/``, so it runs only on a workspace that has the
checkpoint; the name binding between the two committed files is checked here so
that a preset or inventory edit cannot silently detach the exporter.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "model" / "export_headline_mamba_head.py"


def _tool():
    spec = importlib.util.spec_from_file_location("export_headline_mamba_head", TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preset_and_inventory_name_the_same_checkpoint() -> None:
    tool = _tool()
    preset = yaml.safe_load((REPO / tool.PRESET).read_text())
    nodes = json.loads((REPO / tool.INVENTORY).read_text())["nodes"]
    ckpt = nodes[tool.INVENTORY_CKPT_NODE]
    assert preset["use_whole_graph"] is True
    assert ckpt["kind"] == "mamba_ckpt"
    assert ckpt["path"] == preset["mamba_ckpt"]
    assert re.fullmatch(r"[0-9a-f]{64}", ckpt["sha256"])
    backbone = nodes[tool.INVENTORY_BACKBONE_NODE]
    assert backbone["path"] == preset["fpn_backbone_engine"]


def test_builder_defaults_match_the_eval_harness() -> None:
    import argparse

    from scripts.eval.config.core import add_core_args

    tool = _tool()
    parser = argparse.ArgumentParser()
    add_core_args(parser)
    defaults = parser.parse_args([])
    assert defaults.mamba_yolo_weights == tool.DEFAULT_YOLO_WEIGHTS
    assert defaults.mamba_teacher_ckpt == tool.DEFAULT_TEACHER_CKPT
