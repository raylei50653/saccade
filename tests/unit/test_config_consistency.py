"""Guard against silent divergence across the three config layers.

``scripts/eval/config/*.py`` declares each knob three times:
  1. argparse argument (``add_*_args``) — the authoritative source at runtime
  2. module dataclass field (``*Config``) — a typed mirror for inspection
  3. EvalConfig field or cfg.kwargs.get() — the actual consumption path

These tests pin all three layers together so a mismatch is caught at CI.

PHASE 1C additions (2026-07-08):
  * every runtime argparse dest must have a dataclass owner
  * PipelineConfig.to_flat_dict() must not silently omit runtime fields
  * EvalConfig field differences must be explicitly categorised in whitelists
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import yaml

_eval_dir = Path(__file__).resolve().parents[2] / "scripts" / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

_proj_root = Path(__file__).resolve().parents[2]
_src = str(_proj_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

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
from saccade.perception.eval.config import EvalConfig  # noqa: E402

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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _argparse_defaults() -> dict[str, object]:
    parser = build_parser()
    return {a.dest: a.default for a in parser._actions if a.dest != "help"}


def _runtime_arg_dests() -> set[str]:
    return set(_argparse_defaults()) - _ALLOWED_NON_RUNTIME


def _dataclass_field_map() -> dict[str, str]:
    """{field_name: owning_class_name} across all 8 module dataclasses."""
    result: dict[str, str] = {}
    for mod in _MODULES:
        for f in dataclasses.fields(mod):
            result[f.name] = mod.__name__
    return result


def _evalconfig_fields() -> set[str]:
    return set(EvalConfig.__dataclass_fields__.keys())


def _kwargs_gets_in_runtime() -> set[str]:
    """Collect all cfg.kwargs.get("X") keys from the runtime Python files."""
    keys: set[str] = set()
    search = [
        _proj_root / "src/saccade/perception/eval/config.py",
        _proj_root / "src/saccade/perception/eval/pipeline.py",
        _proj_root / "src/saccade/perception/eval/stages.py",
        _proj_root / "src/saccade/perception/eval/evaluator.py",
        _proj_root / "src/saccade/perception/eval/cpp_runner.py",
        _proj_root / "src/saccade/perception/eval/relink.py",
    ]
    for fp in search:
        if fp.exists():
            text = fp.read_text()
            keys.update(re.findall(r'cfg\.kwargs\.get\("([^"]+)"', text))
            keys.update(re.findall(r'kwargs\.get\("([^"]+)"', text))
    return keys


# ---------------------------------------------------------------------------
# EvalConfig field difference whitelists (Phase 2C audited 2026-07-08)
#
# Every runtime argparse dest that is NOT an EvalConfig field must be listed
# in exactly ONE of the categories below.  CI fails otherwise.
#
# Audit legend:
#   [ALIAS] Resolved by _resolve_alias() in config.py's ALIAS_MAP
#   [B] Transformer param — consumed by parse_eval_config, then discarded
#   [C] Experimentally gated (reid_mode=off/semantic=off → dead in prod)
#   [DEP] Deprecated — remove argparse + code in Phase 2
#   [K] Keep (function parameter, not in kwargs)
# ---------------------------------------------------------------------------

# (A) Consumed via _resolve_alias() in parse_eval_config.
# Old argparse dest is an alias for an EvalConfig canonical field.
# Mapping is in ALIAS_MAP in src/saccade/perception/eval/config.py.
ALIAS_RESOLVED: dict[str, str] = {
    # Phase 2C: bool _enabled suffix reversals (arg → EvalConfig field)
    "appearance_bank": "[ALIAS] → appearance_bank_enabled (via _resolve_alias)",
    "duplicate_suppression_enabled": "[ALIAS] → duplicate_suppression (via _resolve_alias)",
    "gmc": "[ALIAS] → gmc_enabled (via _resolve_alias)",
    "id_stability_filter": "[ALIAS] → id_stability_filter_enabled (via _resolve_alias)",
    "lifecycle_merge": "[ALIAS] → lifecycle_merge_enabled (via _resolve_alias)",
    "need_reid": "[ALIAS] → need_reid_enabled (via _resolve_alias)",
    # Phase 2C: non-bool renames
    "preprocess": "[ALIAS] → preprocess_modes (via _resolve_alias)",
    "reid_budget": "[ALIAS] → reid_budget_raw (via _resolve_alias)",
    "reid_engine_path": "[ALIAS] → reid_engine (via _resolve_alias)",
}

# (B) Consumed via cfg.kwargs.get("X") directly, NOT through an EvalConfig field.
KWARGS_DIRECT: dict[str, str] = {
    # [A] Promoted to EvalConfig in Phase 2A (0 fields remaining)
    # [B] Transformer: consumed in parse_eval_config, then discarded (1 field)
    "nsa_kalman": "[B] legacy flag → sets kalman_adapt_mode in config.py — "
    "keep as-is, remove argparse in Phase 2",
    # [C] Experimentally gated — dead when reid_mode=off / semantic=off (50 fields)
    "fpn_reid_ckpt": "[C] experimental FPN-ReID — dead when reid_mode=off",
    "reid_birth_death_boost": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_birth_death_lost_min": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_conf_jitter_gate": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_cooldown_frames": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_cos_threshold": "[C] experimental ReID gate — dead when reid_mode=off",
    "reid_cost_cos_w": "[C] experimental ReID cost — dead when reid_mode=off",
    "reid_cost_iou_w": "[C] experimental ReID cost — dead when reid_mode=off",
    "reid_cost_score_w": "[C] experimental ReID cost — dead when reid_mode=off",
    "reid_geom_smooth_window": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_history_size": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_iou_high": "[C] experimental ReID gate — dead when reid_mode=off",
    "reid_iou_low": "[C] experimental ReID gate — dead when reid_mode=off",
    "reid_long_memory_decay": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_long_memory_trigger": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_lost_age_cap": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_score_decay": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_score_threshold": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_score_threshold_low": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_trigger_mode": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_trigger_persist_frames": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_unstable_iou_weight": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_unstable_shift_weight": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_weight": "[C] experimental ReID gate — dead when reid_mode=off",
    "reid_weight_conf": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_weight_geom": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_weight_lost": "[C] experimental ReID trigger — dead when need_reid=off",
    "reid_weight_new": "[C] experimental ReID trigger — dead when need_reid=off",
    "semantic_appearance_first_margin": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_appearance_first_sim_threshold": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_bidirectional": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_bridge_px": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_claim_warmup_frames": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_debug": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_delayed_claim": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_ema": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_experimental_mode": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_accel_lat": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_accel_long": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_chi2": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_dir_min_cos": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_dir_min_speed": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_fps": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_gate": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_max_speed_mps": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_penalty_weight": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_kalman_person_height_m": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_mahalanobis_threshold": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_min_iou": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_min_lost_frames": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_spatial_gate": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_threshold": "[C] experimental semantic — dead when reid_mode=off",
    "semantic_ttl": "[C] experimental semantic — dead when reid_mode=off",
}

# (C) Consumed as direct function parameter of run_eval().
# mot17.py passes these as positional/keyword args, not through eval_kwargs.
FUNC_PARAM: dict[str, str] = {
    "output": "[K] function parameter in run_eval() signature — not in kwargs",
    "sequences": "[K] function parameter in run_eval() — stored as cfg.seqs",
    "engine": "[K] function parameter — detector engine path (evaluator.py:2063)",
    "pose_engine": "[K] function parameter — two-stage pose engine (evaluator.py:2060-2061)",
}

# (D) Deprecated / dead-weight params with --help preserved.
# Code or argparse is retained, but these are not consumed in the production path.
DEAD_BY_DESIGN: dict[str, str] = {
    "tta": "[DEP] test-time augmentation — NO-GO, code retained, default False",
}

# (E) Consumed in mot17.py to build the detector, not passed to run_eval().
PIPELINE_ONLY: dict[str, str] = {
    "max_det": "[K] detector builder param — consumed in mot17.py for Mamba detector, "
    "not consumed by evaluator (evaluator uses per_frame_detection_cap)",
}

# ---------------------------------------------------------------------------
# Phase 2D: Deprecated argument registry
#
# Formal registry of args that are:
#   - legacy compatibility aliases ([B] ns
# a_kalman)
#   - truly dead / no-op ([DEP] tta)
#   - pipeline-scoped not reval ([K] max_det)
#
# Each entry must have: reason, replacement, runtime_effect.
# CI enforces that every key appears in exactly one whitelist category.
# ---------------------------------------------------------------------------

DEPRECATED_ARGS: list[dict] = [
    {
        "key": "nsa_kalman",
        "category": "KWARGS_DIRECT",
        "reason": "Legacy NSA Kalman flag; converts to kalman_adapt_mode=1 in parse_eval_config",
        "replacement": "Use --kalman-adapt-mode 1 instead",
        "runtime_effect": "Sets kalman_adapt_mode=1 when True (backward-compat path in config.py:985-988)",
        "status": "compat",
    },
    {
        "key": "tta",
        "category": "DEAD_BY_DESIGN",
        "reason": "Test-time augmentation (TTA) — NO-GO per PIPELINE_REFERENCE",
        "replacement": None,
        "runtime_effect": "Code retained but path not exercised; default False",
        "status": "dead",
    },
    {
        "key": "max_det",
        "category": "PIPELINE_ONLY",
        "reason": "Mamba head max detections per frame; consumed in mot17.py for detector construction only",
        "replacement": "per_frame_detection_cap (used by evaluator stages.py:2665)",
        "runtime_effect": "Evaluator ignores this param; uses per_frame_detection_cap instead",
        "status": "pipeline_only",
    },
]

# (G) EvalConfig fields that do NOT correspond to any argparse dest.
# These are derived/computed/internal fields built in parse_eval_config().
EC_DERIVED: dict[str, str] = {
    "appearance_bank_enabled": "derived from appearance_bank",
    "crop_hw": "derived from reid_model",
    "duplicate_suppression": "derived from duplicate_suppression_enabled",
    "geometry_suspect_support_score": "derived from geometry_suspect_score + thresholds",
    "gmc_enabled": "derived from gmc",
    "id_stability_filter_enabled": "derived from id_stability_filter",
    "kwargs": "pass-through raw kwargs dict",
    "lifecycle_merge_enabled": "derived from lifecycle_merge",
    "need_reid_enabled": "derived from need_reid",
    "occ_audit_bank_n": "internal derived field",
    "occ_audit_bank_reference": "internal derived field",
    "occ_vel_weight": "internal derived field",
    "output_root": "derived from output (Path conversion)",
    "pose_box_expand": "internal experimental flag",
    "pose_expand_ankle_conf": "internal experimental param",
    "pose_expand_flat_aspect": "internal experimental param",
    "pose_expand_margin": "internal experimental param",
    "preprocess_modes": "derived from preprocess (parse_preprocess)",
    "profile_lazy_reid_candidates": "derived from profile_lazy_reid_embeddings",
    "profile_lazy_reid_embeddings": "derived from kwargs",
    "reid_budget_raw": "derived from reid_budget",
    "reid_enabled": "derived from reid_mode != 'off'",
    "reid_engine": "derived from reid_engine_path",
    "reid_work_enabled": "derived from reid_mode + profiling flags",
    "seqs": "derived from sequences (list conversion)",
    "use_semantic_mode": "derived from reid_mode / semantic_kalman_gate",
    "use_tracker_reid": "derived from reid_mode",
}

# Combined category sets for validation (key names only)
_MISSING_CATEGORIES = (
    set(ALIAS_RESOLVED)
    | set(KWARGS_DIRECT)
    | set(FUNC_PARAM)
    | set(DEAD_BY_DESIGN)
    | set(PIPELINE_ONLY)
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataclass_defaults_match_argparse() -> None:
    """Every dataclass field with an argparse dest must share its default."""
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


def test_every_dataclass_field_has_an_argparse_dest() -> None:
    """Dataclass fields with no argparse dest silently fall back to a dead default."""
    arg = _argparse_defaults()
    _allowed_orphans = set(EC_DERIVED)
    orphans = []
    for mod in _MODULES:
        for f in dataclasses.fields(mod):
            if f.name not in arg and f.name not in _allowed_orphans:
                orphans.append(f"{mod.__name__}.{f.name}")
    assert not orphans, (
        "dataclass fields with no argparse argument (rename would silently "
        "fall back to the EvalConfig default):\n" + "\n".join(orphans)
    )


def test_all_runtime_args_have_dataclass_owner() -> None:
    """After Phase 1B: every runtime argparse dest must have a dataclass home."""
    dc_map = _dataclass_field_map()
    orphans = []
    for dest in sorted(_runtime_arg_dests()):
        if dest not in dc_map:
            orphans.append(dest)
    assert not orphans, (
        "runtime argparse dests with NO module dataclass field. "
        "Add them to the appropriate *Config in scripts/eval/config/:\n"
        + "\n".join(orphans)
    )


def test_pipeline_config_covers_all_runtime_fields() -> None:
    """PipelineConfig.to_flat_dict() must not silently omit runtime fields.

    Every runtime argparse dest must appear either:
      - as a module dataclass field → PipelineConfig covers it, OR
      - in one of the explicit whitelists (ALIAS_RESOLVED, KWARGS_DIRECT, etc.)
    """
    dc_key_set = set(_dataclass_field_map())
    uncovered = []
    for dest in sorted(_runtime_arg_dests()):
        if dest not in dc_key_set and dest not in _MISSING_CATEGORIES:
            uncovered.append(dest)
    assert not uncovered, (
        "runtime argparse dests NOT in any module dataclass AND not in any "
        "whitelist category. Add the field to the correct *Config dataclass.\n"
        + "\n".join(uncovered)
    )


def test_evalconfig_missing_fields_are_categorised() -> None:
    """Every runtime argparse dest NOT in EvalConfig must be in a whitelist.

    Failures mean an arg was added/removed without updating the whitelist —
    either the field should be added to EvalConfig, or the whitelist needs
    updating to explain WHY it's excluded.
    """
    ec = _evalconfig_fields()
    uncategorised = []
    for dest in sorted(_runtime_arg_dests()):
        if dest not in ec and dest not in _MISSING_CATEGORIES:
            uncategorised.append(dest)
    assert not uncategorised, (
        f"runtime argparse dests ({len(uncategorised)}) not in EvalConfig "
        "and not in any whitelist category (NAME_MAP, KWARGS_DIRECT, "
        "KWARGS_PREFIXED, FUNC_PARAM, DEAD_BY_DESIGN, PIPELINE_ONLY).\n"
        "Add the field to EvalConfig OR update the test whitelist:\n"
        + "\n".join(uncategorised)
    )


def test_whitelist_no_duplicates() -> None:
    """Every whitelist key must appear in exactly ONE category."""
    all_sets = [
        ("ALIAS_RESOLVED", ALIAS_RESOLVED),
        ("KWARGS_DIRECT", KWARGS_DIRECT),
        ("FUNC_PARAM", FUNC_PARAM),
        ("DEAD_BY_DESIGN", DEAD_BY_DESIGN),
        ("PIPELINE_ONLY", PIPELINE_ONLY),
    ]
    from collections import Counter

    counter: Counter[str] = Counter()
    for name, s in all_sets:
        for k in s:
            counter[k] += 1
    dups = {k: v for k, v in counter.items() if v > 1}
    assert not dups, f"whitelist keys in multiple categories:\n{dups}"


def test_whitelist_args_are_real_rt_unne_args() -> None:
    """Whitelist keys must correspond to real runtime argparse dests."""
    rt = _runtime_arg_dests()
    phantom = _MISSING_CATEGORIES - rt
    assert not phantom, (
        "whitelist keys that do NOT exist as runtime argparse dests:\n"
        + "\n".join(sorted(phantom))
    )


def test_evalconfig_derived_fields_are_correct() -> None:
    """EvalConfig fields with no argparse dest must be in EC_DERIVED whitelist."""
    arg = set(_argparse_defaults())
    ec = _evalconfig_fields()
    unlisted = ec - arg - set(EC_DERIVED)
    assert not unlisted, (
        "EvalConfig fields not in argparse and not in EC_DERIVED. "
        "Either add the argparse dest or add to EC_DERIVED:\n"
        + "\n".join(sorted(unlisted))
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


# ---------------------------------------------------------------------------
# Phase 1.5C: Golden config snapshot tests
#
# These tests diff the live resolved config against committed JSON snapshots
# in tests/fixtures/.  A mismatch means a Phase 2 refactoring accidentally
# changed a default value, field name, or YAML merge order.
#
# To regenerate snapshots after an intentional change:
#   uv run python scripts/eval/config/gen_golden_snapshot.py
# ---------------------------------------------------------------------------


def _resolve_live_config(preset_name: str | None = None) -> dict[str, object]:
    """Replicate _resolve_config from gen_golden_snapshot.py for testing."""
    parser = build_parser()
    baseline_path = _proj_root / "configs" / "mot17_baseline.yaml"
    if baseline_path.exists():
        with baseline_path.open() as f:
            defaults = yaml.safe_load(f) or {}
        if "motion" in defaults and isinstance(defaults["motion"], dict):
            defaults.update(defaults.pop("motion"))
        parser.set_defaults(**defaults)
    if preset_name:
        preset_path = _proj_root / "configs" / "presets" / f"{preset_name}.yaml"
        if preset_path.exists():
            with preset_path.open() as f:
                preset = yaml.safe_load(f) or {}
            parser.set_defaults(**preset)
    args, _ = parser.parse_known_args([])
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


def _load_golden_snapshot(name: str) -> dict:
    path = _proj_root / "tests" / "fixtures" / f"golden_config_{name}.json"
    with path.open() as f:
        return json.load(f)


def test_golden_snapshot_baseline() -> None:
    """Live resolved baseline config must match committed golden snapshot."""
    live = _resolve_live_config(None)
    golden = _load_golden_snapshot("baseline")

    assert len(live) == golden["_meta"]["field_count"], (
        f"Field count mismatch: live={len(live)}, golden={golden['_meta']['field_count']}"
    )

    mismatches = []
    golden_config = golden["config"]
    for k in sorted(set(list(live) + list(golden_config))):
        lv = live.get(k, "<MISSING>")
        gv = golden_config.get(k, "<MISSING>")
        if lv != gv:
            mismatches.append(f"  {k}: live={lv!r}  golden={gv!r}")

    assert not mismatches, (
        "Golden snapshot mismatch for baseline preset.\n"
        "If this change is intentional, regenerate with:\n"
        "  uv run python scripts/eval/config/gen_golden_snapshot.py\n\n"
        + "\n".join(mismatches)
    )


def test_golden_snapshot_mamba_whole_graph() -> None:
    """Live resolved mamba_whole_graph config must match committed snapshot."""
    live = _resolve_live_config("mamba_whole_graph")
    golden = _load_golden_snapshot("mamba_whole_graph")

    assert len(live) == golden["_meta"]["field_count"], (
        f"Field count mismatch: live={len(live)}, golden={golden['_meta']['field_count']}"
    )

    mismatches = []
    golden_config = golden["config"]
    for k in sorted(set(list(live) + list(golden_config))):
        lv = live.get(k, "<MISSING>")
        gv = golden_config.get(k, "<MISSING>")
        if lv != gv:
            mismatches.append(f"  {k}: live={lv!r}  golden={gv!r}")

    assert not mismatches, (
        "Golden snapshot mismatch for mamba_whole_graph preset.\n"
        "If this change is intentional, regenerate with:\n"
        "  uv run python scripts/eval/config/gen_golden_snapshot.py\n\n"
        + "\n".join(mismatches)
    )


def test_golden_snapshot_argparse_raw() -> None:
    """Pure argparse defaults (no YAML) must match committed snapshot."""
    parser = build_parser()
    args, _ = parser.parse_known_args([])
    live = {}
    for k, v in vars(args).items():
        if k not in _ALLOWED_NON_RUNTIME:
            if v is None:
                live[k] = None
            elif isinstance(v, (bool, int, float, str)):
                live[k] = v
            elif isinstance(v, list):
                live[k] = v
            elif isinstance(v, Path):
                live[k] = str(v)
            else:
                live[k] = str(v)

    golden = _load_golden_snapshot("argparse_raw")

    assert len(live) == golden["_meta"]["field_count"], (
        f"Field count mismatch: live={len(live)}, golden={golden['_meta']['field_count']}"
    )

    mismatches = []
    golden_config = golden["config"]
    for k in sorted(set(list(live) + list(golden_config))):
        lv = live.get(k, "<MISSING>")
        gv = golden_config.get(k, "<MISSING>")
        if lv != gv:
            mismatches.append(f"  {k}: live={lv!r}  golden={gv!r}")

    assert not mismatches, (
        "Golden snapshot mismatch for raw argparse defaults.\n"
        "If this change is intentional, regenerate with:\n"
        "  uv run python scripts/eval/config/gen_golden_snapshot.py\n\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# Phase 2C: Alias conflict detection tests
# ---------------------------------------------------------------------------


def test_alias_resolve_returns_canonical() -> None:
    """_resolve_alias returns canonical value when only canonical key is set."""
    from saccade.perception.eval.config import _resolve_alias

    assert (
        _resolve_alias({"gmc_enabled": True}, "gmc_enabled", False, coerce=bool) is True
    )
    assert (
        _resolve_alias({"gmc_enabled": False}, "gmc_enabled", True, coerce=bool)
        is False
    )


def test_alias_resolve_falls_back_to_alias() -> None:
    """_resolve_alias falls back to alias when canonical key is absent."""
    from saccade.perception.eval.config import _resolve_alias

    assert _resolve_alias({"gmc": True}, "gmc_enabled", False, coerce=bool) is True
    assert _resolve_alias({"gmc": False}, "gmc_enabled", True, coerce=bool) is False


def test_alias_resolve_returns_default_when_none_present() -> None:
    """_resolve_alias returns default when neither canonical nor alias is set."""
    from saccade.perception.eval.config import _resolve_alias

    assert _resolve_alias({}, "gmc_enabled", True, coerce=bool) is True
    assert _resolve_alias({}, "gmc_enabled", False, coerce=bool) is False
    assert _resolve_alias({}, "vel_alpha", 0.3, coerce=float) == 0.3


def test_alias_conflict_raises_value_error() -> None:
    """_resolve_alias raises ValueError when canonical and alias differ."""
    import pytest
    from saccade.perception.eval.config import _resolve_alias

    with pytest.raises(ValueError, match="Conflicting config values"):
        _resolve_alias(
            {"gmc_enabled": True, "gmc": False},
            "gmc_enabled",
            False,
            coerce=bool,
        )


def test_alias_conflict_same_value_does_not_raise() -> None:
    """_resolve_alias does NOT raise when canonical and alias have same value."""
    from saccade.perception.eval.config import _resolve_alias

    assert (
        _resolve_alias(
            {"gmc_enabled": True, "gmc": True}, "gmc_enabled", False, coerce=bool
        )
        is True
    )


def test_alias_prefer_canonical_over_alias() -> None:
    """_resolve_alias uses canonical value, alias is only fallback."""
    import pytest
    from saccade.perception.eval.config import _resolve_alias

    # Motion: canonical `vel_alpha` takes priority over alias `motion_vel_alpha`
    assert _resolve_alias({"vel_alpha": 0.5}, "vel_alpha", 0.3, coerce=float) == 0.5
    assert (
        _resolve_alias({"motion_vel_alpha": 0.7}, "vel_alpha", 0.3, coerce=float) == 0.7
    )

    # Both present with different values → conflict raises
    with pytest.raises(ValueError, match="Conflicting config values"):
        _resolve_alias(
            {"vel_alpha": 0.5, "motion_vel_alpha": 0.7}, "vel_alpha", 0.3, coerce=float
        )


# ---------------------------------------------------------------------------
# Phase 2D: Deprecated argument registry tests
# ---------------------------------------------------------------------------


def test_deprecated_args_have_required_fields() -> None:
    """Every deprecated arg entry must have key/category/reason/replacement/runtime_effect."""
    required = {"key", "category", "reason", "replacement", "runtime_effect", "status"}
    for entry in DEPRECATED_ARGS:
        missing = required - set(entry)
        assert not missing, (
            f"Deprecated arg entry for '{entry.get('key', 'MISSING')}' "
            f"missing fields: {missing}"
        )


def test_deprecated_key_is_real_argparse_dest() -> None:
    """Every deprecated key must be a real argparse dest."""
    arg = _argparse_defaults()
    for entry in DEPRECATED_ARGS:
        key = entry["key"]
        assert key in arg, (
            f"Deprecated key '{key}' is NOT an argparse dest. "
            f"Remove from DEPRECATED_ARGS or fix runne."
        )


def test_deprecated_key_in_correct_whitelist_category() -> None:
    """Each deprecated key's 'category' must match its whitelist placement."""
    for entry in DEPRECATED_ARGS:
        key = entry["key"]
        category = entry["category"]

        in_alias = key in ALIAS_RESOLVED
        in_kwd = key in KWARGS_DIRECT
        in_func = key in FUNC_PARAM
        in_dead = key in DEAD_BY_DESIGN
        in_ppl = key in PIPELINE_ONLY

        if category == "KWARGS_DIRECT":
            assert in_kwd, f"DEPRECATED key '{key}' claims KWARGS_DIRECT but not in it"
        elif category == "DEAD_BY_DESIGN":
            assert in_dead, (
                f"DEPRECATED key '{key}' claims DEAD_BY_DESIGN but not in it"
            )
        elif category == "PIPELINE_ONLY":
            assert in_ppl, f"DEPRECATED key '{key}' claims PIPELINE_ONLY but not in it"
        elif category == "FUNC_PARAM":
            assert in_func, f"DEPRECATED key '{key}' claims FUNC_PARAM but not in it"
        elif category == "ALIAS_RESOLVED":
            assert in_alias, (
                f"DEPRECATED key '{key}' claims ALIAS_RESOLVED but not in it"
            )
        else:
            raise AssertionError(
                f"Unknown category '{category}' for deprecated key '{key}'"
            )


def test_deprecated_b_keys_are_not_in_multiple_categories() -> None:
    """Deprecated keys must not appear in more than one whitelist simultaneously."""
    all_cat_sets = [
        ALIAS_RESOLVED,
        KWARGS_DIRECT,
        FUNC_PARAM,
        DEAD_BY_DESIGN,
        PIPELINE_ONLY,
    ]
    for entry in DEPRECATED_ARGS:
        key = entry["key"]
        count = sum(1 for s in all_cat_sets if key in s)
        assert count >= 1, f"Deprecated key '{key}' not found in ANY whitelist category"
        assert count <= 1, (
            f"Deprecated key '{key}' found in {count} whitelist categories "
            f"(should be exactly 1)"
        )


# ---------------------------------------------------------------------------
# Phase 3D: Dead fallback default enforcement
# ---------------------------------------------------------------------------

# Keys allowed to use constant imports (REF_HEIGHT_RATIO, etc.) as fallbacks.
# These resolve to the correct values at runtime.
_DEAD_FALLBACK_OK: set[str] = {
    "geometry_ref_height_ratio",  # uses REF_HEIGHT_RATIO constant
    "relink_bridge_fps",  # uses SCENE_FPS constant
    "relink_bridge_person_height",  # uses PERSON_HEIGHT_M constant
    "new_track_thresh",  # uses None sentinel for conditional: is None → use 0.35
}

# Keys excluded from the no-bare-default check (legacy/experimental whitelist).
_EXCLUDED_FROM_FALLBACK_CHECK: set[str] = (
    set(KWARGS_DIRECT)
    | set(ALIAS_RESOLVED)
    | set(FUNC_PARAM)
    | set(DEAD_BY_DESIGN)
    | set(PIPELINE_ONLY)
)


def test_parse_ec_no_dead_fallback_defaults() -> None:
    """parse_eval_config kwargs.get("X", DEFAULT) must match dataclass registry.

    Dead fallbacks are defaults in parse_eval_config that differ from both
    argparse and dataclass defaults — they are never exercised in normal
    operation and indicate a historical ghost value.
    """
    import ast
    import dataclasses as _dc
    import re

    # Build registry from module dataclass defaults
    registry: dict[str, object] = {}
    for mod in _MODULES:
        for f in _dc.fields(mod):
            if f.default is not _dc.MISSING and isinstance(
                f.default, (bool, int, float, str)
            ):
                registry[f.name] = f.default

    config_path = _proj_root / "src/saccade/perception/eval/config.py"
    text = config_path.read_text()

    pattern = r'kwargs\.get\("([^"]+)"\s*,\s*([^)]+)\)'
    dead_fallbacks = []
    for m in re.finditer(pattern, text):
        key = m.group(1)
        default_str = m.group(2).strip()

        if key in _EXCLUDED_FROM_FALLBACK_CHECK:
            continue
        if key in _DEAD_FALLBACK_OK:
            continue
        if key not in registry:
            continue

        registry_default = registry.get(key)
        if registry_default is None:
            continue

        # Parse the fallback literal
        try:
            parsed = ast.literal_eval(default_str)
        except (ValueError, SyntaxError):
            continue

        # Compare
        if parsed != registry_default:
            dead_fallbacks.append(
                f"  {key}: parse_ec fallback={parsed!r} "
                f"registry_default={registry_default!r}"
            )

    assert not dead_fallbacks, (
        f"{len(dead_fallbacks)} dead fallback default(s) in parse_eval_config.\n"
        "These fallback values differ from the dataclass/argparse registry.\n"
        "Replace the fallback with the correct registry default:\n\n"
        + "\n".join(dead_fallbacks)
    )


# ---------------------------------------------------------------------------
# Phase 4B: Module view projection tests
# ---------------------------------------------------------------------------


def test_module_views_are_projections_not_second_source() -> None:
    """cfg.motion.vel_alpha must be identical to cfg.vel_alpha (same object)."""
    from saccade.perception.eval.config import (
        EvalConfig,
        _DEFAULTS,
    )

    # Build a minimal EvalConfig from defaults
    fields = dict(_DEFAULTS)
    fields["data_root"] = "/tmp"
    fields["split"] = "train"
    fields["output_root"] = None
    fields["seqs"] = []
    fields["kwargs"] = {}
    fields["use_semantic_mode"] = False
    fields["use_tracker_reid"] = False
    fields["crop_hw"] = (224, 224)
    fields["preprocess_modes"] = []
    fields["geometry_suspect_score"] = 0.05
    fields["geometry_suspect_support_score"] = 0.05
    fields["nms_iou_threshold"] = 0.5
    fields["tiling"] = "native_960"

    cfg = EvalConfig(**fields)

    # Motion: all fields match
    assert cfg.motion.vel_alpha == cfg.vel_alpha
    assert cfg.motion.acc_alpha == cfg.acc_alpha
    assert cfg.motion.enable_motion_only == cfg.enable_motion_only

    # Geometry: sample
    assert cfg.geometry.kalman_r_scale == cfg.kalman_r_scale
    assert cfg.geometry.oao_tau == cfg.oao_tau
    assert cfg.geometry.occ_iou_thresh == cfg.occ_iou_thresh

    # Lifecycle: sample
    assert cfg.lifecycle.track_buffer == cfg.track_buffer
    assert cfg.lifecycle.interpolate_max_gap == cfg.interpolate_max_gap

    # Core: sample
    assert cfg.core.conf_threshold == cfg.conf_threshold
    assert cfg.core.confirm_streak == cfg.confirm_streak

    # Detection: sample
    assert cfg.detection.tiling == cfg.tiling
    assert cfg.detection.nms_iou_threshold == cfg.nms_iou_threshold

    # ReID: sample
    assert cfg.reid.reid_mode == cfg.reid_mode
    assert cfg.reid.async_reid == cfg.async_reid

    # Semantic: sample
    assert cfg.semantic.semantic_buffer_size == cfg.semantic_buffer_size


def test_module_views_are_frozen() -> None:
    """Module views must be frozen — writing should raise."""
    from saccade.perception.eval.config import _DEFAULTS, EvalConfig

    fields = dict(_DEFAULTS)
    fields["data_root"] = ""
    fields["split"] = ""
    fields["output_root"] = None
    fields["seqs"] = []
    fields["kwargs"] = {}
    fields["use_semantic_mode"] = False
    fields["use_tracker_reid"] = False
    fields["crop_hw"] = (0, 0)
    fields["preprocess_modes"] = []
    fields["geometry_suspect_score"] = 0.0
    fields["geometry_suspect_support_score"] = 0.0
    fields["nms_iou_threshold"] = 0.5
    fields["tiling"] = "native_960"
    cfg = EvalConfig(**fields)
    import pytest

    with pytest.raises(Exception):
        cfg.motion.vel_alpha = 999.0


def test_trigger_view_has_no_fields() -> None:
    """TriggerView is a frozen dataclass with no fields (all trigger params are KWARGS_DIRECT)."""
    import dataclasses
    from saccade.perception.eval.config import TriggerView

    assert len(dataclasses.fields(TriggerView)) == 0
