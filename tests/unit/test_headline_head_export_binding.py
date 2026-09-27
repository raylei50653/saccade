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


# --- probe_head_load: "loaded cleanly" is whatever the head's own loader says ---


def _small_head():
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead

    return MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=2,
        spatial_reduction=2,
        per_channel_a=True,
    )


A_LOG = "mamba_blocks.0.0.A_log"


def test_probe_accepts_the_heads_own_state_dict() -> None:
    head = _small_head()
    assert _tool().probe_head_load(head, head.state_dict()) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }


def test_probe_accepts_a_legal_shared_a_log_broadcast() -> None:
    head = _small_head()
    sd = head.state_dict()
    sd[A_LOG] = sd[A_LOG][:1].clone()  # shared (1, N) -> per-channel (d_inner, N)
    assert _tool().probe_head_load(head, sd)["missing_keys"] == []


def test_probe_rejects_an_illegal_a_log_shape() -> None:
    import pytest

    head = _small_head()
    sd = head.state_dict()
    sd[A_LOG] = sd[A_LOG][:, :3].clone()  # neither conversion applies -> dropped
    with pytest.raises(SystemExit, match=A_LOG):
        _tool().probe_head_load(head, sd)


def test_probe_rejects_any_other_shape_mismatch() -> None:
    import pytest

    head = _small_head()
    sd = head.state_dict()
    key = next(k for k, v in sd.items() if v.dim() == 4)
    sd[key] = sd[key][:1].clone()
    with pytest.raises(SystemExit, match="missing_keys"):
        _tool().probe_head_load(head, sd)


def test_probe_rejects_unexpected_keys() -> None:
    import pytest
    import torch

    head = _small_head()
    sd = head.state_dict()
    sd["not_a_head_param.weight"] = torch.zeros(1)
    with pytest.raises(SystemExit, match="not_a_head_param"):
        _tool().probe_head_load(head, sd)


def test_probe_does_not_touch_the_exported_head() -> None:
    import torch

    head = _small_head()
    before = {k: v.clone() for k, v in head.state_dict().items()}
    sd = {
        k: torch.randn_like(v) if v.is_floating_point() else v
        for k, v in before.items()
    }
    _tool().probe_head_load(head, sd)
    assert all(torch.equal(before[k], v) for k, v in head.state_dict().items())


# --- PR-1R: TF32 off is the only change against the rejected PR-1 form ---


def test_precisions_keep_the_pr1_form_and_separate_stems() -> None:
    tool = _tool()
    assert tool.PRECISIONS["fp32"] == {"stem": tool.DEFAULT_STEM, "tf32": True}
    assert tool.PRECISIONS["fp32-no-tf32"]["tf32"] is False
    stems = [p["stem"] for p in tool.PRECISIONS.values()]
    assert len(set(stems)) == len(stems)  # the rejected artifact is never overwritten


def test_pr1r_pins_the_onnx_the_pr2_declaration_froze() -> None:
    tool = _tool()
    declaration = REPO / "docs/reference/native_runtime_head_parity_declaration.md"
    assert f"sha256 `{tool.PR1_ONNX_SHA256}`" in declaration.read_text()
