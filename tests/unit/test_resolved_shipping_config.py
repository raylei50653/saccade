"""The headline resolved shipping config equals what the Python oracle uses.

``scripts/model/export_resolved_shipping_config.py`` (#465 Phase B PR-3, U2a)
derives ``configs/shipping/mamba_whole_graph.resolved.json`` by executing the
oracle's own resolution and native-setup statements. These tests pin that the
committed file is a fresh export, cross-check it against sources the exporter
does not use (the preset YAML, the bridge-policy resolver, the boundary doc's
frozen tail), and show that the coverage guards fail on an oracle change the
file does not account for.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "model" / "export_resolved_shipping_config.py"


@pytest.fixture(scope="module")
def tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "export_resolved_shipping_config", TOOL
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through it
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def committed(tool: Any) -> dict[str, Any]:
    return json.loads((REPO / tool.OUTPUT).read_text())


@pytest.fixture(scope="module")
def fresh(tool: Any) -> str:
    return tool.render(tool.build())


@pytest.fixture
def mutate(tool: Any, monkeypatch: pytest.MonkeyPatch):
    """Serve an edited copy of one oracle source to the exporter."""
    real = tool._source.__wrapped__
    cached = (
        tool._source,
        tool._tree,
        tool._oracle_expressions,
        tool.binding_signatures,
    )

    def _clear_caches(_tool: Any) -> None:
        for fn in cached:
            fn.cache_clear()

    def apply(rel: str, old: str, new: str) -> None:
        text = real(rel)
        assert text.count(old) == 1, f"mutation anchor not unique in {rel}: {old!r}"
        edited = text.replace(old, new)

        def fake(r: str) -> str:
            return edited if r == rel else real(r)

        _clear_caches(tool)
        monkeypatch.setattr(tool, "_source", fake)

    yield apply
    _clear_caches(tool)


# ---------------------------------------------------------------------------
# The committed file is the oracle's
# ---------------------------------------------------------------------------


def test_committed_file_is_a_fresh_export(tool: Any, fresh: str) -> None:
    assert (REPO / tool.OUTPUT).read_text() == fresh, (
        "rerun scripts/model/export_resolved_shipping_config.py and review the diff"
    )


def test_schema_and_source_binding(tool: Any, committed: dict[str, Any]) -> None:
    import hashlib

    assert committed["schema"] == tool.SCHEMA
    src = committed["source"]
    assert src["preset"] == tool.PRESET
    assert (
        src["preset_sha256"]
        == hashlib.sha256((REPO / tool.PRESET).read_bytes()).hexdigest()
    )
    assert src["oracle_argv"] == [
        "--preset",
        "mamba_whole_graph",
        "--detector",
        "SDP",
        "--double-buffer",
    ]
    assert set(committed) == {
        "schema",
        "source",
        "native_params",
        "native_env",
        "host_params",
    }


# ---------------------------------------------------------------------------
# Cross-checks against sources the exporter does not read
# ---------------------------------------------------------------------------

# Preset keys whose EvalConfig name differs (config.py ALIAS_MAP) or which the
# eval harness consumes before EvalConfig exists.
_PRESET_ALIASES = {
    "gmc": "gmc_enabled",
    "id_stability_filter": "id_stability_filter_enabled",
}
_PRESET_NOT_IN_CFG = {
    "mamba_ckpt",  # detector build (host_params.detector.build)
    "fpn_backbone_engine",  # idem, also kwargs.fpn_backbone_engine
    "use_whole_graph",  # detector build
    "use_cuda_graph",  # detector build
    "main_nms_graphed",  # runtime env SACCADE_MAIN_NMS_GRAPHED
    "preprocess",  # parsed into preprocess_modes
}


def test_every_preset_value_reaches_the_export(committed: dict[str, Any]) -> None:
    preset = yaml.safe_load(
        (REPO / "configs/presets/mamba_whole_graph.yaml").read_text()
    )
    cfg = committed["host_params"]["cfg"]
    build = committed["host_params"]["detector"]["build"]
    unchecked = []
    for key, value in preset.items():
        name = _PRESET_ALIASES.get(key, key)
        if name in cfg:
            assert cfg[name] == value, (
                f"{key}: preset {value!r} vs export {cfg[name]!r}"
            )
        elif "kwargs." + name in cfg:
            assert cfg["kwargs." + name] == value, key
        elif key not in _PRESET_NOT_IN_CFG:
            unchecked.append(key)
    assert not unchecked, f"preset keys the oracle host never reads: {unchecked}"
    assert build["mamba_ckpt"] == preset["mamba_ckpt"]
    assert build["trt_backbone_engine"] == preset["fpn_backbone_engine"]
    assert build["use_whole_graph"] is preset["use_whole_graph"] is True
    assert build["use_cuda_graph"] is preset["use_cuda_graph"] is True
    assert committed["host_params"]["env"]["SACCADE_MAIN_NMS_GRAPHED"] == "1"
    assert cfg["preprocess_modes"] == [] and preset["preprocess"] == "none"


def test_bridge_policy_resolver_agrees(committed: dict[str, Any]) -> None:
    spec = importlib.util.spec_from_file_location(
        "resolved_bridge_policy_config",
        REPO / "scripts/tools/resolved_bridge_policy_config.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = committed["host_params"]["cfg"]
    shared = {k: v for k, v in mod.resolve("mamba_whole_graph").items() if k in cfg}
    assert len(shared) >= 15
    for key, value in shared.items():
        assert cfg[key] == value, key


def test_native_values_match_the_preset(committed: dict[str, Any]) -> None:
    preset = yaml.safe_load(
        (REPO / "configs/presets/mamba_whole_graph.yaml").read_text()
    )
    tracker = committed["native_params"]["GPUByteTracker"]
    calls = {c["method"]: c["args"] for c in tracker["calls"]}
    assert len(calls) == len(tracker["calls"]), "a tracker setter is called twice"
    p = calls["set_params"]
    assert (
        p["match_thresh"],
        p["new_track_thresh"],
        p["r_scale"],
        p["confirm_streak"],
    ) == (
        preset["match_thresh"],
        preset["new_track_thresh"],
        preset["kalman_r_scale"],
        preset["confirm_streak"],
    )
    assert calls["set_occ_params"] == {
        "enabled": preset["occ_state_enabled"],
        "iou_thresh": preset["occ_iou_thresh"],
        "foot_gap": preset["occ_foot_gap"],
        "ttl": preset["occ_ttl"],
        "cost_weight": preset["occ_cost_weight"],
    }
    assert calls["set_oao_params"]["tau"] == preset["oao_tau"]
    assert calls["set_oao_params"]["ramp_frames"] == preset["oao_ramp_frames"]
    assert calls["set_multiplicative_cost"] == {"enabled": True}
    assert calls["set_sinkhorn_lambda"] == {"lambda": preset["sinkhorn_lambda"]}
    assert calls["set_stability_cost_w"] == {"w": preset["stability_cost_w"]}
    relink = calls["set_relink_params"]
    assert relink["bidirectional"] is preset["relink_bridge_enabled"] is True
    for key in ("px", "margin", "h_lo", "h_hi", "spatial_gate", "dir_bonus"):
        assert relink["bridge_" + key] == preset["relink_bridge_" + key], key
    assert (
        committed["native_params"]["GMC"]["constructor"]["downscale"]
        == preset["gmc_downscale"]
    )
    pc = committed["native_params"]["PerceptionPipelineConfig"]["fields"]
    assert pc["private_continuation_enabled"] is preset["private_continuation_enabled"]
    assert pc["private_candidate_nms_iou"] == preset["private_candidate_nms_iou"]
    assert pc["private_min_score"] == preset["private_min_score"]
    assert pc["private_max_candidates"] == preset["private_max_candidates"]
    assert pc["private_prior_iou_threshold"] == preset["private_prior_iou_threshold"]


def test_boundary_frozen_tail(committed: dict[str, Any]) -> None:
    """docs/reference/native_runtime_shipping_boundary.md §5 B2 freezes the tail."""
    steps = committed["host_params"]["steps"]
    cfg = committed["host_params"]["cfg"]
    assert steps["tail.cheb_gr_or_occ_audit"] is False
    assert cfg["cheb_gr_merge_enabled"] is False
    assert steps["tail.post_lifecycle_merge"] is False
    assert steps["tail.deferred_alias"] is False
    assert cfg["kwargs.semantic_delayed_claim"] is False
    assert steps["tail.tracklet_quality_filter"] is False
    assert (cfg["min_tracklet_len"], cfg["min_tracklet_score"]) == (1, 0.0)
    assert steps["tail.interpolation"] is True
    assert (
        cfg["interpolate_max_gap"],
        cfg["interpolate_min_track_len"],
        cfg["interpolate_min_h"],
    ) == (35, 5, 0)


def test_every_step_is_an_explicit_boolean(
    tool: Any, committed: dict[str, Any]
) -> None:
    steps = committed["host_params"]["steps"]
    assert list(steps) == [s.name for s in tool.STEPS]
    assert all(isinstance(v, bool) for v in steps.values())


def test_native_env_covers_every_native_getenv(committed: dict[str, Any]) -> None:
    import re

    names: set[str] = set()
    for root in ("src", "include"):
        for path in (REPO / root).rglob("*"):
            if (
                path.suffix in {".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}
                and path.is_file()
            ):
                text = path.read_text(errors="replace")
                names |= set(re.findall(r'getenv\("(SACCADE_[A-Z0-9_]+)"', text))
                names |= set(re.findall(r'env_\w+\("(SACCADE_[A-Z0-9_]+)"', text))
    assert names == set(committed["native_env"])
    assert committed["native_env"]["SACCADE_ENABLE_DDA"] is True
    assert committed["native_env"]["SACCADE_STABILITY_W"] == 0.1


# ---------------------------------------------------------------------------
# Coverage guards fail closed on an oracle change
# ---------------------------------------------------------------------------


def test_no_unexplained_native_touch(tool: Any) -> None:
    assert tool.native_touch_gaps() == []


def test_new_native_call_outside_spans_is_a_gap(tool: Any, mutate: Any) -> None:
    mutate(
        tool.PIPELINE,
        "        self.gmc_estimator = gmc_estimator\n",
        "        self.gmc_estimator = gmc_estimator\n        detector.tracker.set_frame_size(1, 1)\n",
    )
    gaps = tool.native_touch_gaps()
    assert len(gaps) == 1 and "detector.tracker.set_frame_size(1, 1)" in gaps[0]


def test_new_native_call_inside_a_span_is_recorded(tool: Any, mutate: Any) -> None:
    mutate(
        tool.PIPELINE,
        "        association_scoring_mode = (\n",
        "        detector.tracker.set_frame_size(7, 9)\n        association_scoring_mode = (\n",
    )
    log = tool.NativeLog()
    with tool.oracle_environment():
        oracle = tool.resolve_oracle(log)
        tool.capture_native(oracle)
    sizes = [c.args for c in log.of("GPUByteTracker") if c.method == "set_frame_size"]
    assert (7, 9) in sizes


def test_new_cfg_read_reaches_host_params(tool: Any, mutate: Any) -> None:
    mutate(
        tool.STAGES,
        "    cfg = state.cfg\n    seq = state.seq\n    w_orig = state.w_orig\n",
        "    cfg = state.cfg\n    _probe = getattr(cfg, 'shipping_probe_knob', 0.5)\n"
        "    seq = state.seq\n    w_orig = state.w_orig\n",
    )
    log = tool.NativeLog()
    with tool.oracle_environment():
        oracle = tool.resolve_oracle(log)
    assert tool.resolved_reads(oracle.cfg)["shipping_probe_knob"] == 0.5


def test_cfg_read_of_a_missing_field_fails(tool: Any, mutate: Any) -> None:
    mutate(
        tool.STAGES,
        "    cfg = state.cfg\n    seq = state.seq\n    w_orig = state.w_orig\n",
        "    cfg = state.cfg\n    _probe = cfg.shipping_probe_knob\n"
        "    seq = state.seq\n    w_orig = state.w_orig\n",
    )
    log = tool.NativeLog()
    with tool.oracle_environment():
        oracle = tool.resolve_oracle(log)
    with pytest.raises(SystemExit, match="shipping_probe_knob"):
        tool.resolved_reads(oracle.cfg)


def test_native_getenv_without_default_fails(tool: Any, mutate: Any) -> None:
    mutate(
        "src/tracking/gmc_kernel.cu",
        '        const char* v = std::getenv("SACCADE_GMC_PCR_THRESH");\n'
        "        return v ? std::strtof(v, nullptr) : 5.0f;\n    }();\n"
        "    find_peak_subpixel_kernel<<<1, 256, 0, stream>>>(\n"
        "        (float*)d_tmp_float, w, h, d_peak_x, d_peak_y, d_peak_val, d_pcr_score,\n"
        "        pcr_thresh);\n",
        '        const char* v = std::getenv("SACCADE_GMC_PCR_THRESH");\n'
        "        return v ? std::strtof(v, nullptr) : 5.0f;\n    }();\n"
        '    if (std::getenv("SACCADE_PROBE_HATCH")) {}\n'
        "    find_peak_subpixel_kernel<<<1, 256, 0, stream>>>(\n"
        "        (float*)d_tmp_float, w, h, d_peak_x, d_peak_y, d_peak_val, d_pcr_score,\n"
        "        pcr_thresh);\n",
    )
    with pytest.raises(SystemExit, match="SACCADE_PROBE_HATCH"):
        tool.native_env()


def test_moved_gate_expression_fails(tool: Any, mutate: Any) -> None:
    mutate(
        tool.EVALUATOR,
        "        if cfg.interpolate_tracklets:\n",
        "        if cfg.interpolate_max_gap > 0:\n",
    )
    with pytest.raises(SystemExit, match="tail.interpolation"):
        tool.verify_step_gates()
