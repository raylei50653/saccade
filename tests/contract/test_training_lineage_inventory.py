"""Contract for the #421 training-lineage inventory: claims vs bytes stay separate.

The role file is a claim; the artifact is the evidence. These tests pin the
predicates that keep that split honest:

* a role file that cannot be trusted (unknown kind, dangling ``expected_*``,
  two roles on one path) is rejected before any artifact is touched;
* parent edges come from the checkpoint's own ``args``; an ``expected_*`` claim
  the bytes contradict is reported as ``mismatch``, never silently accepted;
* "frozen SSM" is a bit-comparison of the interior tensors, not a flag read;
* ONNX attribution names a source only on a unique, complete match — a partial
  or tied match is a non-answer with its own verdict;
* a missing artifact is ``unavailable``; nothing else is put in its place.
"""

# scope: detection
# function: contract
# lifecycle: active

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.provenance import training_lineage as tl


# --------------------------------------------------------------------------- roles


def _roles(**overrides: object) -> dict:
    base = {
        "schema": tl.ROLES_SCHEMA,
        "families": {"s": {"backbone": "yolo26s"}},
        "roles": {
            "s.teacher": {
                "family": "s",
                "kind": "gated_teacher_ckpt",
                "path": "runs/t/best.ckpt",
            },
            "s.child": {
                "family": "s",
                "kind": "mamba_ckpt",
                "path": "runs/c/best.ckpt",
                "expected_teacher": "s.teacher",
            },
        },
    }
    base.update(overrides)
    return base


def _write_roles(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_roles_schema_is_required(tmp_path: Path) -> None:
    with pytest.raises(tl.LineageError, match="schema"):
        tl.load_roles(_write_roles(tmp_path, _roles(schema="other")))


def test_roles_reject_unknown_kind_and_undeclared_family(tmp_path: Path) -> None:
    bad_kind = _roles()
    bad_kind["roles"]["s.child"]["kind"] = "weights"
    with pytest.raises(tl.LineageError, match="unknown kind"):
        tl.load_roles(_write_roles(tmp_path, bad_kind))

    bad_family = _roles()
    bad_family["roles"]["m.x"] = {
        "family": "m",
        "kind": "yolo_pt",
        "path": "models/x.pt",
    }
    with pytest.raises(tl.LineageError, match="undeclared family"):
        tl.load_roles(_write_roles(tmp_path, bad_family))


def test_roles_reject_dangling_expectation_and_shared_path(tmp_path: Path) -> None:
    dangling = _roles()
    dangling["roles"]["s.child"]["expected_init_parent"] = "s.ghost"
    with pytest.raises(tl.LineageError, match="undeclared role"):
        tl.load_roles(_write_roles(tmp_path, dangling))

    shared = _roles()
    shared["roles"]["s.dup"] = {
        "family": "s",
        "kind": "mamba_ckpt",
        "path": "runs/c/best.ckpt",
    }
    with pytest.raises(tl.LineageError, match="same path"):
        tl.load_roles(_write_roles(tmp_path, shared))


# --------------------------------------------------------------------------- tensors


def test_tensor_group_splits_ssm_interior_from_projections() -> None:
    assert tl.tensor_group("mamba_blocks.0.0.A_log") == "mamba_blocks.ssm_internal"
    assert tl.tensor_group("mamba_blocks.0.0.D") == "mamba_blocks.ssm_internal"
    assert (
        tl.tensor_group("temporal_blocks.2.0.dt_proj.bias")
        == "temporal_blocks.ssm_internal"
    )
    assert (
        tl.tensor_group("temporal_blocks.2.0.x_proj.weight")
        == "temporal_blocks.ssm_internal"
    )
    assert tl.tensor_group("mamba_blocks.0.0.in_proj.weight") == "mamba_blocks.proj"
    assert tl.tensor_group("mamba_blocks.0.0.out_proj.weight") == "mamba_blocks.proj"
    assert tl.tensor_group("cls_head.0.2.weight") == "cls_head"


def _student(
    seed: int, *, temporal: bool = False, perturb_ssm: bool = False
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    base = torch.Generator().manual_seed(0)
    sd = {
        "input_proj.0.weight": torch.randn(4, 4, 1, 1, generator=g),
        "mamba_blocks.0.0.A_log": torch.randn(1, 4, generator=base),
        "mamba_blocks.0.0.D": torch.randn(8, generator=base),
        "mamba_blocks.0.0.in_proj.weight": torch.randn(8, 4, generator=g),
        "cls_head.0.0.weight": torch.randn(4, 8, 3, 3, generator=g),
    }
    if perturb_ssm:
        sd["mamba_blocks.0.0.D"] = sd["mamba_blocks.0.0.D"] + 1e-3
    if temporal:
        sd["temporal_blocks.0.0.A_log"] = torch.randn(1, 4, generator=base)
    return sd


def test_tensor_delta_counts_and_frozen_predicate() -> None:
    parent = _student(1)
    child = _student(2, temporal=True)
    delta = tl.tensor_delta(child, parent)
    assert delta["totals"] == {"identical": 2, "changed": 3, "added": 1, "removed": 0}
    assert delta["groups"]["mamba_blocks.ssm_internal"] == {
        "identical": 2,
        "changed": 0,
        "added": 0,
        "removed": 0,
    }
    assert delta["groups"]["temporal_blocks.ssm_internal"]["added"] == 1

    perturbed = tl.tensor_delta(_student(2, perturb_ssm=True), parent)
    assert perturbed["groups"]["mamba_blocks.ssm_internal"]["changed"] == 1

    removed = tl.tensor_delta(parent, child)
    assert removed["totals"]["removed"] == 1


def test_treatment_delta_separates_changed_from_one_sided_keys() -> None:
    parent = {
        "gt_ratio": 0.5,
        "clip_len": 4,
        "run_dir": "runs/a",
        "old_flag": True,
        "seed": 1,
    }
    child = {
        "gt_ratio": 0.0,
        "clip_len": 4,
        "run_dir": "runs/b",
        "new_flag": False,
        "seed": 1,
    }
    delta = tl.treatment_delta(child, parent)
    assert delta["changed"] == {"gt_ratio": {"parent": 0.5, "child": 0.0}}
    assert delta["only_child"] == ["new_flag"]
    assert delta["only_parent"] == ["old_flag"]
    assert "run_dir" not in delta["changed"]


# --------------------------------------------------------------------------- onnx


def test_fold_bn_convs_matches_manual_fold_and_emits_bias() -> None:
    g = torch.Generator().manual_seed(3)
    sd = {
        "m.conv.weight": torch.randn(2, 3, 3, 3, generator=g),
        "m.bn.weight": torch.rand(2, generator=g) + 0.5,
        "m.bn.bias": torch.randn(2, generator=g),
        "m.bn.running_mean": torch.randn(2, generator=g),
        "m.bn.running_var": torch.rand(2, generator=g) + 0.1,
        "plain.conv.weight": torch.randn(2, 2, 1, 1, generator=g),
    }
    folded = tl.fold_bn_convs(sd)
    scale = sd["m.bn.weight"] / torch.sqrt(sd["m.bn.running_var"] + 1e-3)
    assert torch.equal(
        folded["m.conv.weight"], sd["m.conv.weight"] * scale.reshape(-1, 1, 1, 1)
    )
    assert torch.equal(
        folded["m.fused_bias"], sd["m.bn.bias"] - sd["m.bn.running_mean"] * scale
    )
    assert torch.equal(
        folded["plain.conv.weight"], sd["plain.conv.weight"]
    )  # no BN → unchanged


def test_match_onnx_verdicts() -> None:
    a = torch.arange(64.0).reshape(8, 8)
    b = torch.arange(64.0, 128.0).reshape(8, 8)
    c = torch.arange(200.0, 264.0).reshape(64)
    inits = [
        a.numpy(),
        b.numpy().T.copy(),
        c.numpy(),
    ]  # second one transposed, as Gemm stores it

    full = {"x": {"p": a, "q": b, "r": c}}
    res = tl.match_onnx(inits, full)
    assert res["verdict"] == "unique_exact" and res["best"] == ["x"]

    partial = {"x": {"p": a, "r": c}}
    res = tl.match_onnx(inits, partial)
    assert res["verdict"] == "partial" and res["matches"]["x"] == 2

    tied = {"x": {"p": a}, "y": {"p": a.clone()}}
    res = tl.match_onnx(inits, tied)
    assert res["verdict"] == "ambiguous" and res["best"] == ["x", "y"]

    res = tl.match_onnx(inits, {"x": {"p": a + 1}})
    assert res["verdict"] == "none" and res["best"] == []

    assert tl.match_onnx([], full)["verdict"] == "none"


# --------------------------------------------------------------------------- end to end


def _save_mamba(
    path: Path, *, sd: dict, args: dict, mamba_args: dict, epoch: int = 5
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "student": sd,
            "best_loss": 1.0,
            "args": args,
            "mamba_args": mamba_args,
        },
        path,
    )


def _save_teacher(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 12,
            "model": {"w": torch.zeros(2)},
            "args": {"yolo_weights": "models/yolo/y.pt"},
        },
        path,
    )


@pytest.fixture()
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tl, "REPO_ROOT", tmp_path)
    (tmp_path / "models/yolo").mkdir(parents=True)
    (tmp_path / "models/yolo/y.pt").write_bytes(b"yolo-blob")
    _save_teacher(tmp_path / "runs/teacher/best.ckpt")
    _save_teacher(tmp_path / "runs/other_teacher/best.ckpt")
    teacher_sha = tl.sha256_file(tmp_path / "runs/teacher/best.ckpt")
    yolo_sha = tl.sha256_file(tmp_path / "models/yolo/y.pt")

    common_args = {
        "yolo_weights": "models/yolo/y.pt",
        "teacher_ckpt": "runs/teacher/best.ckpt",
        "seed": 7,
    }
    _save_mamba(
        tmp_path / "runs/distill/best.ckpt",
        sd=_student(1),
        args={**common_args, "cache_dir": "runs/cache_gone", "lr": 1e-3, "clip_len": 1},
        mamba_args={"scan_stop_grad": True, "use_temporal_mamba": False},
    )
    _save_mamba(
        tmp_path / "runs/gt/best.ckpt",
        sd=_student(2, temporal=True),
        args={
            **common_args,
            "mamba_ckpt": "runs/distill/best.ckpt",
            "lr": 1e-4,
            "clip_len": 3,
            "gt_ratio": 0.0,
        },
        mamba_args={
            "scan_stop_grad": True,
            "use_temporal_mamba": True,
            "base_yolo_sha256": yolo_sha,
            "teacher_checkpoint_sha256": teacher_sha,
        },
    )
    # A run whose bytes say "other_teacher" while the role file will claim "s.teacher".
    _save_mamba(
        tmp_path / "runs/liar/best.ckpt",
        sd=_student(2, perturb_ssm=True),
        args={
            **common_args,
            "teacher_ckpt": "runs/other_teacher/best.ckpt",
            "mamba_ckpt": "runs/distill/best.ckpt",
        },
        mamba_args={"scan_stop_grad": True, "teacher_checkpoint_sha256": "00" * 32},
    )
    (tmp_path / "configs/presets").mkdir(parents=True)
    (tmp_path / "configs/presets/p.yaml").write_text(
        yaml.safe_dump(
            {
                "mamba_ckpt": "runs/gt/best.ckpt",
                "use_whole_graph": True,
                "reid_mode": "off",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _fake_roles() -> dict:
    return {
        "schema": tl.ROLES_SCHEMA,
        "families": {"s": {"backbone": "yolo26s"}},
        "roles": {
            "s.yolo": {"family": "s", "kind": "yolo_pt", "path": "models/yolo/y.pt"},
            "s.teacher": {
                "family": "s",
                "kind": "gated_teacher_ckpt",
                "path": "runs/teacher/best.ckpt",
            },
            "s.cache": {
                "family": "s",
                "kind": "teacher_cache",
                "path": "runs/cache_gone",
            },
            "s.distill": {
                "family": "s",
                "kind": "mamba_ckpt",
                "path": "runs/distill/best.ckpt",
                "expected_teacher": "s.teacher",
            },
            "s.gt": {
                "family": "s",
                "kind": "mamba_ckpt",
                "path": "runs/gt/best.ckpt",
                "expected_init_parent": "s.distill",
                "expected_teacher": "s.teacher",
            },
            "s.liar": {
                "family": "s",
                "kind": "mamba_ckpt",
                "path": "runs/liar/best.ckpt",
                "expected_init_parent": "s.gt",
                "expected_teacher": "s.teacher",
            },
            "s.missing": {
                "family": "s",
                "kind": "mamba_ckpt",
                "path": "runs/nope/best.ckpt",
            },
            "s.preset": {
                "family": "s",
                "kind": "preset",
                "path": "configs/presets/p.yaml",
            },
        },
    }


def test_inventory_derives_edges_from_bytes_and_grades_claims(fake_repo: Path) -> None:
    inv = tl.Inventory(_fake_roles(), tensor_diff=True, onnx_match=False)
    inv.build()
    nodes = inv.nodes

    # Unavailable stays unavailable and is never described.
    assert nodes["s.missing"].exists is False and nodes["s.missing"].summary == {}
    assert nodes["s.cache"].exists is False

    # Edges resolve to roles by path; a cache the args name but the disk lacks is flagged.
    distill_edges = {e.relation: e for e in nodes["s.distill"].edges}
    assert distill_edges["teacher"].target_node == "s.teacher"
    assert (
        distill_edges["cache"].target_node == "s.cache"
        and distill_edges["cache"].target_exists is False
    )
    assert nodes["s.distill"].checks["teacher_sha_attested"] == "not_recorded"

    gt = nodes["s.gt"]
    assert gt.checks["expected_init_parent"]["status"] == "ok"
    assert gt.checks["expected_teacher"]["status"] == "ok"
    assert gt.checks["base_yolo_sha_attested"] == "verified"
    assert gt.checks["teacher_sha_attested"] == "verified"
    delta = gt.checks["tensor_delta_vs_init_parent"]
    assert delta["parent"] == "s.distill" and delta["ssm_internal_frozen"] is True
    assert delta["treatment_delta"]["changed"]["clip_len"] == {"parent": 1, "child": 3}

    # The role file's story loses to the checkpoint's own args.
    liar = nodes["s.liar"]
    assert liar.checks["expected_teacher"]["status"] == "mismatch"
    assert liar.checks["expected_teacher"]["derived"] == [
        None
    ]  # other_teacher is not a declared role
    assert liar.checks["expected_init_parent"]["status"] == "mismatch"
    assert liar.checks["teacher_sha_attested"] == "MISMATCH"
    assert liar.checks["tensor_delta_vs_init_parent"]["ssm_internal_frozen"] is False


def test_inventory_deployment_reads_preset_and_dedups_by_sha(fake_repo: Path) -> None:
    inv = tl.Inventory(_fake_roles(), tensor_diff=False, onnx_match=False)
    inv.build()
    dep = inv.nodes["s.preset"].checks["deployment_forward"]
    assert dep["checkpoint"]["node"] == "s.gt"
    assert dep["checkpoint"]["aliases_same_sha256"] == []
    assert dep["temporal_blocks"].startswith("present; BYPASSED")
    assert dep["final_stage_gt_ratio"] == 0.0
    assert dep["backbone_source"] == "PyTorch backbone from mamba_teacher_ckpt"
    assert (
        inv.nodes["s.gt"].checks.get("tensor_delta_vs_init_parent") is None
    )  # tensor diff was off


def test_payload_and_markdown_carry_capture_identity(fake_repo: Path) -> None:
    inv = tl.Inventory(_fake_roles(), tensor_diff=False, onnx_match=False)
    inv.build()
    payload = inv.to_payload()
    assert payload["schema"] == tl.SCHEMA
    assert set(payload["captured"]) >= {
        "at_utc",
        "host",
        "git_head",
        "git_dirty",
        "tensor_diff",
        "onnx_match",
    }
    assert set(payload["nodes"]) == set(_fake_roles()["roles"])
    md = tl.render_markdown(payload, inv)
    assert "<!-- doc-status: active -->" in md
    assert "`s.missing`" in md and "**unavailable**" in md
    assert "mismatch" in md
