"""Contract for the #421 comparison matrix: a pair is ``controlled`` only when the bytes say so.

The declaration file claims what each comparison is *for*; the inventory says
what the artifacts *are*. These tests pin the predicates that keep the class a
derived fact rather than a label:

* an unmatched non-treatment axis — structural or training — can never yield
  ``controlled``; a structural one yields ``system_comparison``, a training one
  ``historical_not_comparable``;
* an ``unknown`` axis is fail-closed even when it is the declared treatment;
* declaring the schedule as treatment excuses only the declared keys — any
  other differing key (e.g. ``warmup_epochs``) is a confound;
* a design premise the bytes contradict (a "sibling" that is really a
  continuation; a seed replicate whose recipe differs) is ``blocked``;
* the s backbone/teacher split is ``common_mode`` when both endpoints share it
  and ``blocking`` when only one does;
* an unavailable endpoint is ``blocked``; unknown node ids and structural
  treatment axes are rejected at declaration time;
* the committed matrix is exactly what the tool derives from the committed
  inventory and declarations (freshness), and it contains at least one
  controlled same-family training pair and one system pair.
"""

# scope: detection
# function: contract
# lifecycle: active

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.provenance import training_comparison as tc

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- fixtures


def _mamba_node(
    nid: str,
    *,
    init: str | None,
    init_exists: bool = True,
    seed: int = 1,
    epochs: int = 30,
    warmup: int = 3,
    clip_len: int = 4,
    gt_ratio: float = 0.0,
    add_temporal: bool = False,
    cache: str | None = "s.cache",
    cache_exists: bool = False,
    teacher: str = "s.teacher",
    exists: bool = True,
    temporal: bool = False,
    seqs: str = "",
) -> dict:
    fam = nid.split(".")[0]
    args = {
        "epochs": epochs,
        "lr": 1e-4,
        "batch_size": 4,
        "clip_len": clip_len,
        "clip_stride": 8,
        "gt_ratio": gt_ratio,
        "add_temporal": add_temporal,
        "scan_stop_grad": True,
        "warmup_epochs": warmup,
        "seed": seed,
        "seqs": seqs,
        "holdout_seqs": "",
        "teacher_ckpt": f"runs/{teacher}.ckpt",
        "yolo_weights": f"models/{fam}.pt",
    }
    edges = [
        {
            "relation": "teacher",
            "target_path": f"runs/{teacher}.ckpt",
            "target_node": teacher,
            "target_exists": True,
        },
        {
            "relation": "base_yolo",
            "target_path": f"models/{fam}.pt",
            "target_node": f"{fam}.pretrained",
            "target_exists": True,
        },
    ]
    if init is not None:
        args["mamba_ckpt"] = f"runs/{init}.ckpt"
        edges.insert(
            0,
            {
                "relation": "init",
                "target_path": f"runs/{init}.ckpt",
                "target_node": init,
                "target_exists": init_exists,
            },
        )
    if cache is not None:
        args["cache_dir"] = f"runs/{cache}"
        edges.append(
            {
                "relation": "cache",
                "target_path": f"runs/{cache}",
                "target_node": cache,
                "target_exists": cache_exists,
            }
        )
    return {
        "id": nid,
        "family": fam,
        "kind": "mamba_ckpt",
        "path": f"runs/{nid}.ckpt",
        "exists": exists,
        "sha256": "ab" * 32,
        "summary": {
            "epoch": epochs - 1,
            "args": args,
            "mamba_args": {
                "d_model": 128,
                "d_state": 16,
                "num_blocks": 1,
                "spatial_reduction": 4,
                "num_classes": 80,
                "use_pixel_shuffle": True,
                "use_cross_scan": True,
                "use_temporal_mamba": temporal,
            },
            "param_count": 11_000_000 if temporal else 10_000_000,
            "module_groups": ["cls_head"]
            + (["temporal_blocks.proj"] if temporal else []),
        },
        "edges": edges,
        "checks": {},
        "problems": [],
    }


def _teacher_node(nid: str, exists: bool = True) -> dict:
    fam = nid.split(".")[0]
    return {
        "id": nid,
        "family": fam,
        "kind": "gated_teacher_ckpt",
        "path": f"runs/{nid}.ckpt",
        "exists": exists,
        "sha256": "cd" * 32,
        "summary": {
            "epoch": 12,
            "args": {"seed": 7, "epochs": 30, "lr_gate": 1e-3, "seqs": ""},
            "provenance": {"training_sequences": list(tc.DEFAULT_SDP_SEQUENCES)},
            "param_count": 10,
        },
        "edges": [
            {
                "relation": "base_yolo",
                "target_path": f"models/{fam}.pt",
                "target_node": f"{fam}.pretrained",
                "target_exists": True,
            }
        ],
        "checks": {},
        "problems": [],
    }


def _engine_node(nid: str, onnx_match: str, exists: bool = True) -> dict:
    fam = nid.split(".")[0]
    return {
        "id": nid,
        "family": fam,
        "kind": "trt_engine",
        "path": f"models/{nid}",
        "exists": exists,
        "sha256": "ef" * 32,
        "summary": {},
        "edges": [
            {
                "relation": "onnx_matches_checkpoint",
                "target_path": f"runs/{onnx_match}.ckpt",
                "target_node": onnx_match,
                "target_exists": True,
                "detail": {"verdict": "unique_exact"},
            }
        ],
        "checks": {},
        "problems": [],
    }


def _preset_node(
    fam: str, engine_path: str, head_engine_path: str | None = None
) -> dict:
    summary = {
        "fpn_backbone_engine": engine_path,
        "use_whole_graph": True,
        "reid_mode": "off",
    }
    if head_engine_path:
        summary["mamba_head_engine"] = head_engine_path
    return {
        "id": f"{fam}.deployment_preset",
        "family": fam,
        "kind": "preset",
        "path": f"configs/presets/{fam}.yaml",
        "exists": True,
        "sha256": "00" * 32,
        "summary": summary,
        "edges": [],
        "checks": {
            "deployment_forward": {
                "temporal_blocks": "bypassed",
                "final_stage_gt_ratio": 0.0,
                "gate_teacher_at_runtime": None,
                "head_engine": "none (PyTorch head inside whole graph)",
                "embedding": "reid_mode='off'",
                "graphs": {"use_whole_graph": True},
            }
        },
        "problems": [],
    }


def _inventory() -> dict:
    """One s family: teacher, legacy teacher, gt1, two matched siblings, a seed twin, a continuation."""
    nodes = {
        "s.pretrained": {
            "id": "s.pretrained",
            "family": "s",
            "kind": "yolo_pt",
            "path": "models/s.pt",
            "exists": True,
            "sha256": "11" * 32,
            "summary": {},
            "edges": [],
            "checks": {},
            "problems": [],
        },
        "s.teacher": _teacher_node("s.teacher"),
        "s.legacy_teacher": _teacher_node("s.legacy_teacher"),
        "s.cache": {
            "id": "s.cache",
            "family": "s",
            "kind": "teacher_cache",
            "path": "runs/s.cache",
            "exists": False,
            "sha256": None,
            "summary": {},
            "edges": [],
            "checks": {},
            "problems": [],
        },
        "s.distill": _mamba_node(
            "s.distill", init=None, seed=1, clip_len=1, cache="s.cache"
        ),
        "s.gt1": _mamba_node(
            "s.gt1", init="s.distill", seed=1, gt_ratio=0.5, warmup=5, cache=None
        ),
        "s.implicit_42": _mamba_node("s.implicit_42", init="s.gt1", seed=42),
        "s.implicit_43": _mamba_node("s.implicit_43", init="s.gt1", seed=43),
        "s.phase_a_42": _mamba_node(
            "s.phase_a_42",
            init="s.gt1",
            seed=42,
            epochs=15,
            clip_len=3,
            add_temporal=True,
            temporal=True,
        ),
        "s.phase_b_42": _mamba_node(
            "s.phase_b_42",
            init="s.phase_a_42",
            seed=42,
            epochs=15,
            clip_len=1,
            temporal=True,
        ),
        "s.plain_warm5": _mamba_node("s.plain_warm5", init="s.gt1", seed=42, warmup=5),
        "s.continuation": _mamba_node(
            "s.continuation",
            init="s.phase_b_42",
            seed=42,
            epochs=15,
            clip_len=1,
            temporal=True,
            cache="s.cache_gpu",
            cache_exists=True,
        ),
        "s.legacy_head": _mamba_node(
            "s.legacy_head",
            init="s.ghost",
            init_exists=False,
            seed=1,
            teacher="s.legacy_teacher",
        ),
        "s.missing": _mamba_node("s.missing", init="s.gt1", exists=False),
        "s.engine": _engine_node("s.engine", "s.legacy_teacher"),
        "s.engine_own": _engine_node("s.engine_own", "s.teacher"),
        "s.deployment_preset": _preset_node("s", "models/s.engine"),
    }
    nodes["s.cache_gpu"] = {
        "id": "s.cache_gpu",
        "family": "s",
        "kind": "teacher_cache",
        "path": "runs/s.cache_gpu",
        "exists": True,
        "sha256": None,
        "summary": {},
        "edges": [],
        "checks": {},
        "problems": [],
    }
    nodes["s.ghost"] = {
        "id": "s.ghost",
        "family": "s",
        "kind": "mamba_ckpt",
        "path": "runs/s.ghost.ckpt",
        "exists": False,
        "sha256": None,
        "summary": {},
        "edges": [],
        "checks": {},
        "problems": [],
    }
    return {
        "schema": tc.INVENTORY_SCHEMA,
        "issue": "test",
        "captured": {"at_utc": "t", "git_head": "0" * 12},
        "families": {"s": {"backbone": "yolo26s"}},
        "nodes": nodes,
    }


def _decl(*comparisons: dict) -> dict:
    return {"schema": tc.DECLARATIONS_SCHEMA, "comparisons": list(comparisons)}


def _pair(
    cid: str,
    lhs_node: str,
    rhs_node: str,
    design: str = "paired_siblings",
    treatment=("training_schedule",),
    keys=("clip_len", "clip_stride", "add_temporal", "staging"),
    **extra,
) -> dict:
    spec = {
        "comparison_id": cid,
        "design": design,
        "lhs": {"node": lhs_node},
        "rhs": {"node": rhs_node},
        "intended_treatment": "t",
        "treatment_axes": list(treatment),
        "treatment_schedule_keys": list(keys),
    }
    spec.update(extra)
    return spec


def _run(inv: dict, decl: dict, tmp_path: Path) -> dict:
    dpath = tmp_path / "decl.json"
    dpath.write_text(json.dumps(decl), encoding="utf-8")
    declarations = tc.load_declarations(dpath, inv)
    return {
        c["comparison_id"]: c
        for c in tc.build_matrix(inv, declarations, [], tmp_path, probe=None)[
            "comparisons"
        ]
    }


# --------------------------------------------------------------------------- declarations


def test_declarations_reject_unknown_node_and_structural_treatment(
    tmp_path: Path,
) -> None:
    inv = _inventory()
    bad = _decl(_pair("x", "s.implicit_42", "s.nope"))
    p = tmp_path / "d.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(tc.ComparisonError, match="unknown node"):
        tc.load_declarations(p, inv)

    structural = _decl(
        _pair("x", "s.implicit_42", "s.phase_b_42", treatment=("backbone_family",))
    )
    p.write_text(json.dumps(structural), encoding="utf-8")
    with pytest.raises(tc.ComparisonError, match="may not declare"):
        tc.load_declarations(p, inv)

    no_seed_treatment = _decl(
        _pair(
            "x", "s.implicit_42", "s.implicit_43", treatment=("training_seed",), keys=()
        )
    )
    p.write_text(json.dumps(no_seed_treatment), encoding="utf-8")
    with pytest.raises(tc.ComparisonError, match="may not declare"):
        tc.load_declarations(p, inv)

    dup = _decl(
        _pair("x", "s.implicit_42", "s.phase_b_42"),
        _pair("x", "s.implicit_43", "s.phase_b_42"),
    )
    p.write_text(json.dumps(dup), encoding="utf-8")
    with pytest.raises(tc.ComparisonError, match="duplicate"):
        tc.load_declarations(p, inv)


# --------------------------------------------------------------------------- classification rule


def test_matched_siblings_are_controlled_and_share_the_backbone_confound(
    tmp_path: Path,
) -> None:
    rows = _run(
        _inventory(), _decl(_pair("c", "s.implicit_42", "s.phase_b_42")), tmp_path
    )
    c = rows["c"]
    assert c["classification"] == "controlled"
    assert c["treatment_axes_observed"] == ["training_schedule"]
    assert c["axes"]["warm_start"]["status"] == "matched"
    assert c["axes"]["training_budget"]["status"] == "matched"
    assert c["axes"]["head_family"]["status"] == "matched_effective"
    sev = {x["name"]: x["severity"] for x in c["remaining_confounds"]}
    assert sev == {"deployed_backbone_teacher_mismatch": "common_mode"}
    assert c["executability"]["fresh_eval"] is True
    assert c["executability"]["retrain_replay"] is False


def test_undeclared_schedule_key_is_a_confound_even_when_schedule_is_treatment(
    tmp_path: Path,
) -> None:
    rows = _run(
        _inventory(), _decl(_pair("c", "s.plain_warm5", "s.phase_b_42")), tmp_path
    )
    c = rows["c"]
    assert c["classification"] == "historical_not_comparable"
    assert c["axes"]["training_schedule"]["confound_keys"] == ["warmup_epochs"]
    names = [x["name"] for x in c["remaining_confounds"] if x["severity"] == "training"]
    assert names == ["training_schedule"]


def test_seed_difference_outside_treatment_is_historical(tmp_path: Path) -> None:
    rows = _run(
        _inventory(), _decl(_pair("c", "s.implicit_43", "s.phase_b_42")), tmp_path
    )
    c = rows["c"]
    assert c["classification"] == "historical_not_comparable"
    assert [
        x["name"] for x in c["remaining_confounds"] if x["severity"] == "training"
    ] == ["training_seed"]


def test_seed_replicate_is_controlled_only_when_recipe_and_start_match(
    tmp_path: Path,
) -> None:
    inv = _inventory()
    good = _pair(
        "ok",
        "s.implicit_42",
        "s.implicit_43",
        design="seed_replicate",
        treatment=("training_seed",),
        keys=(),
    )
    bad = _pair(
        "bad",
        "s.implicit_42",
        "s.plain_warm5",
        design="seed_replicate",
        treatment=("training_seed",),
        keys=(),
    )
    rows = _run(inv, _decl(good, bad), tmp_path)
    assert rows["ok"]["classification"] == "controlled"
    assert rows["ok"]["treatment_axes_observed"] == ["training_seed"]
    assert rows["bad"]["classification"] == "blocked"
    assert any("seed_replicate premise" in b for b in rows["bad"]["blocking_confounds"])


def test_continuation_declared_as_sibling_is_blocked(tmp_path: Path) -> None:
    spec = _pair(
        "g", "s.phase_b_42", "s.continuation", treatment=("teacher_cache",), keys=()
    )
    c = _run(_inventory(), _decl(spec), tmp_path)["g"]
    assert c["classification"] == "blocked"
    assert any("premise violated" in b for b in c["blocking_confounds"])
    assert "descends from lhs" in c["axes"]["warm_start"]["detail"]


def test_stage_increment_is_controlled_and_lists_the_stage_as_treatment(
    tmp_path: Path,
) -> None:
    spec = _pair(
        "b",
        "s.distill",
        "s.gt1",
        design="stage_increment",
        treatment=(
            "warm_start",
            "training_budget",
            "training_schedule",
            "teacher_cache",
        ),
        keys=("clip_len", "gt_ratio", "warmup_epochs"),
    )
    c = _run(_inventory(), _decl(spec), tmp_path)["b"]
    assert c["classification"] == "controlled"
    assert set(c["treatment_axes_observed"]) == {
        "warm_start",
        "training_budget",
        "training_schedule",
        "teacher_cache",
    }
    assert c["axes"]["training_budget"]["rhs"] == 30
    # dataset: explicit seven vs '' resolve to the same set
    assert c["axes"]["dataset_split"]["status"] == "matched"


def test_stage_increment_with_undeclared_key_is_not_controlled(tmp_path: Path) -> None:
    spec = _pair(
        "b",
        "s.distill",
        "s.gt1",
        design="stage_increment",
        treatment=(
            "warm_start",
            "training_budget",
            "training_schedule",
            "teacher_cache",
        ),
        keys=("clip_len", "gt_ratio"),
    )
    c = _run(_inventory(), _decl(spec), tmp_path)["b"]
    assert c["classification"] == "historical_not_comparable"
    assert c["axes"]["training_schedule"]["confound_keys"] == ["warmup_epochs"]


def test_unavailable_endpoint_is_blocked(tmp_path: Path) -> None:
    c = _run(_inventory(), _decl(_pair("m", "s.implicit_42", "s.missing")), tmp_path)[
        "m"
    ]
    assert c["classification"] == "blocked"
    assert "s.missing unavailable" in c["blocking_confounds"]


def test_unknown_axis_never_yields_controlled(tmp_path: Path) -> None:
    # legacy head: chain reaches an unavailable parent => schedule/budget unknown.
    spec = _pair(
        "f",
        "s.legacy_head",
        "s.legacy_head",
        design="runtime_ab",
        treatment=("deployed_backbone_artifact",),
        keys=(),
        rhs={
            "node": "s.legacy_head",
            "runtime": {
                "binding": "family_preset",
                "backbone_engine_node": "s.engine_own",
            },
        },
    )
    c = _run(_inventory(), _decl(spec), tmp_path)["f"]
    assert c["classification"] != "controlled"
    assert "training_schedule" in c["unknown_axes"]


def test_asymmetric_backbone_consistency_blocks_same_family_pairs(
    tmp_path: Path,
) -> None:
    # legacy head trained against the legacy teacher == engine's sibling ONNX; replica head not.
    spec = _pair(
        "f", "s.legacy_head", "s.implicit_42", design="system", treatment=(), keys=()
    )
    c = _run(_inventory(), _decl(spec), tmp_path)["f"]
    assert c["classification"] == "blocked"
    sev = {x["name"]: x["severity"] for x in c["remaining_confounds"]}
    assert sev["deployed_backbone_teacher_mismatch"] == "blocking"


def test_runtime_ab_attributes_to_the_engine_only(tmp_path: Path) -> None:
    spec = _pair(
        "e",
        "s.phase_b_42",
        "s.phase_b_42",
        design="runtime_ab",
        treatment=("deployed_backbone_artifact",),
        keys=(),
        rhs={
            "node": "s.phase_b_42",
            "runtime": {
                "binding": "family_preset",
                "backbone_engine_node": "s.engine_own",
            },
        },
    )
    c = _run(_inventory(), _decl(spec), tmp_path)["e"]
    assert c["classification"] == "controlled"
    assert c["treatment_axes_observed"] == ["deployed_backbone_artifact"]
    assert (
        c["rhs"]["runtime"]["backbone_teacher_consistency"] == "same_teacher_indicated"
    )
    assert (
        c["lhs"]["runtime"]["backbone_teacher_consistency"]
        == "DIFFERENT_TEACHER_INDICATED"
    )


def test_structural_difference_is_a_system_comparison(tmp_path: Path) -> None:
    spec = {
        "comparison_id": "s",
        "design": "system",
        "lhs": {
            "node": "s.teacher",
            "runtime": {
                "binding": "shared_backbone_node",
                "backbone_node": "s.teacher",
            },
        },
        "rhs": {
            "node": "s.phase_b_42",
            "runtime": {
                "binding": "shared_backbone_node",
                "backbone_node": "s.teacher",
            },
        },
        "intended_treatment": "head",
        "treatment_axes": [],
    }
    c = _run(_inventory(), _decl(spec), tmp_path)["s"]
    assert c["classification"] == "system_comparison"
    assert c["axes"]["head_family"]["status"] == "unmatched"
    assert c["axes"]["deployed_backbone_artifact"]["status"] == "matched"
    assert c["axes"]["tracker_runtime_policy"]["status"] == "unknown"


def _with_fixed_head_engine(inv: dict) -> dict:
    inv = json.loads(json.dumps(inv))
    inv["nodes"]["s.head_engine"] = {
        "id": "s.head_engine",
        "family": "s",
        "kind": "trt_engine",
        "path": "models/s.head_engine",
        "exists": True,
        "sha256": "aa" * 32,
        "summary": {},
        "edges": [
            {
                "relation": "onnx_matches_checkpoint",
                "target_path": "runs/s.implicit_42.ckpt",
                "target_node": "s.implicit_42",
                "target_exists": True,
                "detail": {"verdict": "ambiguous"},
            },
            {
                "relation": "onnx_matches_checkpoint",
                "target_path": "runs/s.implicit_43.ckpt",
                "target_node": "s.implicit_43",
                "target_exists": True,
                "detail": {"verdict": "ambiguous"},
            },
        ],
        "checks": {},
        "problems": [],
    }
    inv["nodes"]["s.deployment_preset"] = _preset_node(
        "s", "models/s.engine", "models/s.head_engine"
    )
    return inv


def test_family_preset_temporal_metadata_follows_endpoint() -> None:
    inv = _with_fixed_head_engine(_inventory())
    binding = {"binding": "family_preset", "head_override": "checkpoint_pytorch"}
    for node, expected in (
        ("s.distill", "absent"),
        ("s.gt1", "absent"),
        ("s.phase_b_42", "present; BYPASSED by whole-graph effective T=1"),
    ):
        runtime = tc.bind_runtime(tc.build_profile(node, inv), binding, inv)
        assert runtime["temporal_blocks"] == expected
        assert runtime["effective_T"] == 1
        assert "final_stage_gt_ratio" not in runtime
        assert runtime["graphs"] == {"use_whole_graph": True}
        assert runtime["embedding"] == "reid_mode='off'"


def test_gt_ratio_is_training_metadata_not_forward_mode(tmp_path: Path) -> None:
    inv = _inventory()
    spec = _pair(
        "gt",
        "s.gt1",
        "s.plain_warm5",
        design="stage_increment",
        treatment=(
            "warm_start",
            "teacher_cache",
            "training_schedule",
            "training_budget",
        ),
        keys=("gt_ratio",),
    )
    row = _run(inv, _decl(spec), tmp_path)["gt"]
    assert "training_schedule" in row["treatment_axes_observed"]
    assert row["axes"]["inference_forward_mode"]["status"] == "matched"
    for side in ("lhs", "rhs"):
        assert "final_stage_gt_ratio" not in row["axes"]["inference_forward_mode"][side]
        assert "final_stage_gt_ratio" not in row[side]["runtime"]
    # Even legacy runtime payloads with different training ratios must not
    # create a structural forward difference.
    lhs = tc.build_profile("s.gt1", inv)
    rhs = tc.build_profile("s.plain_warm5", inv)
    lr = {**row["lhs"]["runtime"], "final_stage_gt_ratio": 0.5}
    rr = {**row["rhs"]["runtime"], "final_stage_gt_ratio": 0.0}
    axes = tc.compare_profiles(lhs, rhs, lr, rr, spec, inv, tmp_path)
    assert axes["inference_forward_mode"]["status"] == "matched"


def test_fixed_head_engine_blocks_training_designs_until_overridden(
    tmp_path: Path,
) -> None:
    inv = _with_fixed_head_engine(_inventory())
    as_is = _pair("c", "s.implicit_42", "s.phase_b_42")
    c = _run(inv, _decl(as_is), tmp_path)["c"]
    assert c["classification"] == "blocked"
    assert any(b.startswith("treatment_not_deployed") for b in c["blocking_confounds"])
    assert c["lhs"]["runtime"]["head_source"]["checkpoint_head_deployed"] is False
    assert c["lhs"]["runtime"]["head_source"]["node"] == "s.head_engine"
    assert c["axes"]["deployed_head_artifact"]["lhs"]["head_engine_source"].startswith(
        "sibling ONNX ambiguous"
    )

    override = {"binding": "family_preset", "head_override": "checkpoint_pytorch"}
    fixed = _pair(
        "c",
        "s.implicit_42",
        "s.phase_b_42",
        lhs={"node": "s.implicit_42", "runtime": override},
        rhs={"node": "s.phase_b_42", "runtime": override},
    )
    c2 = _run(inv, _decl(fixed), tmp_path)["c"]
    assert c2["classification"] == "controlled"
    assert c2["lhs"]["runtime"]["head_source"]["checkpoint_head_deployed"] is True
    assert "--no-mamba-trt" in c2["lhs"]["runtime"]["recipe"]


def test_fixed_head_engine_is_reported_not_blocked_for_system_designs(
    tmp_path: Path,
) -> None:
    inv = _with_fixed_head_engine(_inventory())
    # keep the backbone/teacher consistency symmetric so only the head engine is at stake
    inv["nodes"]["s.engine"] = _engine_node("s.engine", "s.teacher")
    spec = {
        "comparison_id": "e",
        "design": "system",
        "lhs": {
            "node": "s.teacher",
            "runtime": {
                "binding": "shared_backbone_node",
                "backbone_node": "s.teacher",
            },
        },
        "rhs": {"node": "s.phase_b_42"},
        "intended_treatment": "system",
        "treatment_axes": [],
    }
    c = _run(inv, _decl(spec), tmp_path)["e"]
    assert c["classification"] == "system_comparison"
    assert not any(
        b.startswith("treatment_not_deployed") for b in c["blocking_confounds"]
    )
    assert c["rhs"]["runtime"]["head_source"]["checkpoint_head_deployed"] is False
    assert (
        "checkpoint source ambiguous" in c["axes"]["deployed_head_artifact"]["detail"]
    )


def test_head_override_is_validated(tmp_path: Path) -> None:
    inv = _inventory()
    p = tmp_path / "d.json"
    bad = _decl(
        _pair(
            "x",
            "s.implicit_42",
            "s.phase_b_42",
            lhs={
                "node": "s.implicit_42",
                "runtime": {"binding": "family_preset", "head_override": "trt"},
            },
        )
    )
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(tc.ComparisonError, match="head_override"):
        tc.load_declarations(p, inv)
    wrong_binding = _decl(
        _pair(
            "x",
            "s.implicit_42",
            "s.phase_b_42",
            lhs={
                "node": "s.implicit_42",
                "runtime": {
                    "binding": "shared_backbone_node",
                    "backbone_node": "s.teacher",
                    "head_override": "checkpoint_pytorch",
                },
            },
        )
    )
    p.write_text(json.dumps(wrong_binding), encoding="utf-8")
    with pytest.raises(tc.ComparisonError, match="only applies to family_preset"):
        tc.load_declarations(p, inv)


def test_fresh_eval_requires_a_bound_runner(tmp_path: Path) -> None:
    spec = {
        "comparison_id": "a",
        "design": "system",
        "lhs": {
            "node": "s.pretrained",
            "runtime": {
                "binding": "shared_backbone_node",
                "backbone_node": "s.pretrained",
            },
        },
        "rhs": {
            "node": "s.teacher",
            "runtime": {
                "binding": "shared_backbone_node",
                "backbone_node": "s.teacher",
            },
        },
        "intended_treatment": "adaptation",
        "treatment_axes": [],
    }
    c = _run(_inventory(), _decl(spec), tmp_path)["a"]
    ex = c["executability"]
    assert ex["artifacts_available"] is True
    assert ex["runtime_recipe_bound"] is False
    assert ex["fresh_eval"] is False
    assert any("no runner in inventory" in m for m in ex["fresh_eval_missing"])
    assert c["lhs"]["runtime"]["recipe"] is None


# --------------------------------------------------------------------------- committed matrix


def _committed() -> dict:
    return json.loads((REPO / tc.DEFAULT_JSON_OUT).read_text(encoding="utf-8"))


def test_committed_matrix_is_fresh() -> None:
    inventory = tc.load_inventory(REPO / tc.DEFAULT_INVENTORY)
    declarations = tc.load_declarations(tc.DEFAULT_DECLARATIONS, inventory)
    rows = tc.load_results_table(REPO / tc.DEFAULT_RESULTS_TABLE)
    fresh = tc.build_matrix(inventory, declarations, rows, REPO, probe=None)
    committed = _committed()
    assert tc._strip_probe(committed) == tc._strip_probe(fresh)
    md = (REPO / tc.DEFAULT_MD_OUT).read_text(encoding="utf-8")
    assert md == tc.render_markdown(
        {**fresh, "workspace_probe": committed.get("workspace_probe")}
    )


def test_committed_matrix_has_the_required_pairs() -> None:
    committed = _committed()
    rows = {c["comparison_id"]: c for c in committed["comparisons"]}
    controlled_training = [
        c
        for c in rows.values()
        if c["classification"] == "controlled"
        and c["design"] in ("paired_siblings", "seed_replicate")
    ]
    assert controlled_training, "at least one attributable same-family training pair"
    assert all(
        c["lhs"]["node"].split(".")[0] == c["rhs"]["node"].split(".")[0]
        for c in controlled_training
    )
    assert any(c["classification"] == "system_comparison" for c in rows.values())
    # Every controlled row: no unknown axis, every non-treatment axis matched.
    for c in rows.values():
        if c["classification"] != "controlled":
            continue
        assert not c["unknown_axes"], c["comparison_id"]
        for axis, ax in c["axes"].items():
            if axis in c["treatment_axes"]:
                continue
            assert ax["status"] in ("matched", "matched_effective"), (
                c["comparison_id"],
                axis,
            )
        assert not c["axes"]["training_schedule"].get("confound_keys"), c[
            "comparison_id"
        ]
    # Non-controlled rows name what is wrong.
    for c in rows.values():
        if c["classification"] == "blocked":
            assert c["blocking_confounds"], c["comparison_id"]
        elif c["classification"] != "controlled":
            assert c["remaining_confounds"], c["comparison_id"]
        assert c["historical_results"]["reusable_as_paired"] is False


def test_committed_matrix_states_the_known_issues() -> None:
    rows = {c["comparison_id"]: c for c in _committed()["comparisons"]}
    # s production backbone/teacher split: common-mode on the matched s pairs, blocking on legacy-vs-replica.
    c1 = rows["C1.s.explicit_vs_implicit_shared_gt1_s42"]
    assert c1["classification"] == "controlled"
    assert {x["name"]: x["severity"] for x in c1["remaining_confounds"]} == {
        "deployed_backbone_teacher_mismatch": "common_mode"
    }
    assert c1["lhs"]["runtime"]["engine_bytes_attributed"] is False
    f2 = rows["F2.s.legacy_v14_vs_t3t1_production"]
    assert f2["classification"] == "blocked"
    # old seed-chain pairs vs shared-GT1 controls
    assert (
        rows["C7.s.implicit_shared_s43_vs_t3t1_seed_chain_s13"]["classification"]
        == "blocked"
    )
    assert (
        rows["C4.s.plain_gt2_vs_t3t1_seed_chain_s13"]["classification"]
        == "historical_not_comparable"
    )
    assert rows["C4.s.plain_gt2_vs_t3t1_seed_chain_s13"]["axes"]["training_schedule"][
        "confound_keys"
    ] == ["warmup_epochs"]
    # missing caches: no retrain replay on the replica chains
    assert (
        rows["C1.s.explicit_vs_implicit_shared_gt1_s42"]["executability"][
            "retrain_replay"
        ]
        is False
    )
    assert rows["G1.m.t3t1_vs_gpu_decode_cache"]["classification"] == "blocked"
    # m preset's fixed head engine: preset-as-is is blocked, override rows run the checkpoint head
    assert rows["B3x.m.distill_to_gt1_preset_as_is"]["classification"] == "blocked"
    assert any(
        b.startswith("treatment_not_deployed")
        for b in rows["B3x.m.distill_to_gt1_preset_as_is"]["blocking_confounds"]
    )
    for cid in (
        "B3.m.distill_to_gt1",
        "B4.m.gt1_to_gt2_plain",
        "C9.m.plain_gt2_vs_t3t1",
    ):
        for side in ("lhs", "rhs"):
            assert (
                rows[cid][side]["runtime"]["head_source"]["checkpoint_head_deployed"]
                is True
            ), cid
    e1 = rows["E1.s_vs_m.production_system"]
    assert e1["rhs"]["runtime"]["head_source"]["checkpoint_head_deployed"] is False
    assert (
        "ambiguous" in e1["axes"]["deployed_head_artifact"]["rhs"]["head_engine_source"]
    )
    # raw-YOLO rows have no runner in inventory
    assert (
        rows["A1.s.pretrained_vs_adapted_teacher"]["executability"]["fresh_eval"]
        is False
    )
    # every controlled training row actually deploys both checkpoint heads
    for c in rows.values():
        if c["classification"] == "controlled" and c["design"] != "runtime_ab":
            for side in ("lhs", "rhs"):
                assert (
                    c[side]["runtime"]["head_source"]["checkpoint_head_deployed"]
                    is True
                ), c["comparison_id"]
    # m family has no controlled curriculum pair
    assert (
        rows["C9.m.plain_gt2_vs_t3t1"]["classification"] == "historical_not_comparable"
    )
    assert not any(
        c["classification"] == "controlled"
        and c["design"] == "paired_siblings"
        and c["lhs"]["node"].startswith("m.")
        for c in rows.values()
    )
