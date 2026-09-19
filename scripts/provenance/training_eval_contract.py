"""Shared evaluation contract and identity-bound runner for the #421 lineage (deliverable 3).

Deliverable 2 (``training_comparison.py``) said *which* node pairs a paired
measurement can be attributed to.  This tool freezes *how* such a measurement
is taken, so that two results produced weeks apart, by different people, can
be told apart from two results that merely look alike:

* the **contract** (``training_eval_contract.json``) pins the dataset key
  (sequence set, per-sequence frame count, gt / seqinfo / image-byte digests),
  the evaluator and metric protocol, the timing boundaries, the execution
  profiles (whole-graph vs eager PyTorch), the tracker policy for eager rows,
  the environment hygiene rules, and the measurement procedure (smoke →
  repeat → formal; runtime-repeat variance separate from training-seed
  variance; no determinism claim from one run).  Its ``declared`` block is
  written by hand; everything else (``recipes``, ``pairs``, ``dataset_key``,
  ``frozen``) is derived by ``freeze`` and checked by ``check``.
* a **recipe** is one endpoint under one runtime binding, as the matrix
  bound it, turned into the exact ``scripts/eval/mot17.py`` argv plus the
  sha256 of every artifact the resolved config names and the sha256 of the
  resolved config itself (every argparse dest after preset merge).
* ``run`` claims the output directory with a v3 ``run_manifest.json`` whose
  ``runtime_identity`` block carries all of that **before** ``mot17.py``
  writes a byte (``mot17.py`` joins the claim through the parent-run permit,
  so the entry point itself — a ``decision_relevant`` path — is untouched),
  refuses to start on any drift (artifact / preset / dataset / environment /
  contract sha; a stray ``SACCADE_*`` hatch; a dirty tree for a formal run;
  a formal run without a same-identity repeat report), and writes the raw
  stdout, the parsed metrics and the MOT file hashes next to the manifest.
* ``validate-pair`` decides whether two formal runs are a paired measurement
  of a declared comparison: every identity path must be equal except the
  ones the pair's treatment explains (frozen per pair as
  ``allowed_differences``).  Missing identity, a non-formal stage, a
  contract mismatch or any unexplained difference is ``not_paired``.

Nothing here produces or interprets a baseline number; deliverable 4 runs
the recipes and reads the reports.

Usage:
    .venv/bin/python scripts/provenance/training_eval_contract.py freeze
    .venv/bin/python scripts/provenance/training_eval_contract.py check [--verify-dataset]
    .venv/bin/python scripts/provenance/training_eval_contract.py show <recipe|pair>
    .venv/bin/python scripts/provenance/training_eval_contract.py preflight <recipe> --stage formal
    .venv/bin/python scripts/provenance/training_eval_contract.py run <recipe> --stage smoke|repeat|formal [--runs N]
    .venv/bin/python scripts/provenance/training_eval_contract.py repeat-report <run_dir>...
    .venv/bin/python scripts/provenance/training_eval_contract.py validate-pair <comparison_id> <lhs_run> <rhs_run>
"""

# status: stable

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _extra in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts" / "eval",
    REPO_ROOT / "scripts" / "tools",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import yaml  # noqa: E402

from scripts.provenance import run_manifest as rm  # noqa: E402

DEFAULT_CONTRACT = _HERE.parent / "training_eval_contract.json"
DEFAULT_MATRIX = Path("report_data/training_comparison_matrix.json")
DEFAULT_INVENTORY = Path("report_data/training_lineage_inventory.json")
DEFAULT_MD_OUT = Path("docs/research/training/training_eval_contract.md")
DEFAULT_RUN_ROOT = Path("results/training_eval_contract")

CONTRACT_SCHEMA = "training_eval_contract_v1"
IDENTITY_SCHEMA = "training_eval_runtime_identity_v1"
RUN_RECORD_SCHEMA = "training_eval_run_record_v1"
REPEAT_REPORT_SCHEMA = "training_eval_repeat_report_v1"
PAIR_VERDICT_SCHEMA = "training_eval_pair_verdict_v1"
MATRIX_SCHEMA = "training_comparison_matrix_v1"
INVENTORY_SCHEMA = "training_lineage_inventory_v1"

STAGES = ("smoke", "repeat", "formal")
PAIRABLE_STAGES = ("formal",)
RUN_RECORD_FILENAME = "contract_run.json"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"
REPEAT_REPORT_GLOB = "repeat-*-report.json"

MOT17_ENTRY = Path("scripts/eval/mot17.py")

# argparse dests that name a file the run reads.  Every non-empty value is
# hashed and must exist: ``build_mamba_gated_detector`` silently falls back
# to a gate-free config when ``teacher_ckpt`` is missing, which is exactly the
# drift this contract exists to refuse.
ARTIFACT_KEYS = (
    "mamba_ckpt",
    "mamba_teacher_ckpt",
    "mamba_yolo_weights",
    "fpn_backbone_engine",
    "mamba_head_engine",
    "teacher_head_ckpt",
    "teacher_head_backbone_engine",
    "engine",
)

# Resolved-config keys that belong to the execution substrate or name the
# artifacts under test; they are excluded when the eager tracker policy is
# lifted from the production preset.
EXECUTION_KEYS = frozenset(
    {
        "fpn_backbone_engine",
        "mamba_ckpt",
        "mamba_head_engine",
        "mamba_teacher_ckpt",
        "mamba_yolo_weights",
        "teacher_head_ckpt",
        "teacher_head_backbone_engine",
        "engine",
        "use_whole_graph",
        "use_cuda_graph",
        "use_tracker_graph",
        "main_nms_graphed",
        "mamba_trt",
        "no_compile",
        "no_temporal",
        # which file the defaults came from, not a policy key
        "preset",
        "config",
    }
)

# Flags a recipe may never carry: they would reroute config resolution
# through files the contract does not pin.
FORBIDDEN_RECIPE_FLAGS = (
    "--config",
    "--module-detection",
    "--module-geometry",
    "--module-motion",
    "--module-reid",
    "--module-semantic",
    "--module-trigger",
    "--module-lifecycle",
    "--output",
    "--sequences",
    "--max-frames",
    "--processes",
    "--cpp-threads",
    "--double-buffer",
    "--visualize",
    "--latency-only",
)

REQUIRED_METRIC_KEYS = ("IDF1", "MOTA", "HOTA", "DetA", "AssA", "IDs", "FP", "FN")


class ContractError(RuntimeError):
    """A contract, recipe, identity or procedure rule was violated."""


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- inputs


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return payload


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ContractError(
            f"{path}: schema {payload.get('schema')!r} is not {CONTRACT_SCHEMA}"
        )
    if not isinstance(payload.get("declared"), dict):
        raise ContractError(f"{path}: missing the hand-written 'declared' block")
    return payload


def load_matrix(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != MATRIX_SCHEMA:
        raise ContractError(
            f"{path}: schema {payload.get('schema')!r} is not {MATRIX_SCHEMA}"
        )
    return payload


def load_inventory(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != INVENTORY_SCHEMA:
        raise ContractError(
            f"{path}: schema {payload.get('schema')!r} is not {INVENTORY_SCHEMA}"
        )
    return payload


def contract_sha256(contract: Mapping[str, Any]) -> str:
    """Digest of the contract with its own digest slot blanked."""
    body = json.loads(json.dumps(contract))
    frozen = body.get("frozen")
    if isinstance(frozen, dict):
        frozen.pop("contract_sha256", None)
    return sha256_json(body)


# --------------------------------------------------------------------------- config resolution
#
# This mirrors ``scripts/eval/mot17.py`` for the only path a recipe may use:
# ``--preset NAME`` with no ``--config`` and no ``--module-*`` file, i.e.
# ``parser.set_defaults(**preset_yaml)`` then ``parse_args(argv)``.  The
# entry point is a ``decision_relevant`` path and cannot be imported without
# loading TensorRT, so the merge is re-stated here and pinned by a contract
# test against the golden config fixtures.


def build_mot17_parser() -> argparse.ArgumentParser:
    from mot17_args import build_parser

    parser = build_parser()
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--mamba-trt", action="store_true", default=None)
    parser.add_argument("--no-mamba-trt", action="store_false", dest="mamba_trt")
    parser.add_argument("--mamba-head-engine", default=None)
    return parser


def preset_path(name: str) -> Path:
    return REPO_ROOT / "configs" / "presets" / f"{name}.yaml"


def load_preset(name: str) -> dict[str, Any]:
    path = preset_path(name)
    if not path.is_file():
        raise ContractError(f"preset {name!r} not found at {path}")
    with path.open() as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ContractError(f"preset {path} is not a mapping")
    return loaded


def _preset_of(argv: Sequence[str]) -> str:
    args = list(argv)
    for i, tok in enumerate(args):
        if tok == "--preset" and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith("--preset="):
            return tok.split("=", 1)[1]
    raise ContractError("recipe argv must name --preset")


def check_recipe_argv(argv: Sequence[str]) -> None:
    for tok in argv:
        flag = tok.split("=", 1)[0]
        if flag in FORBIDDEN_RECIPE_FLAGS:
            raise ContractError(
                f"recipe argv may not carry {flag}: the contract fixes it "
                "(or it would route config through unpinned files)"
            )


def resolve_config(argv: Sequence[str]) -> dict[str, Any]:
    """Every argparse dest of ``mot17.py`` after preset merge + argv."""
    check_recipe_argv(argv)
    parser = build_mot17_parser()
    parser.set_defaults(**load_preset(_preset_of(argv)))
    ns = parser.parse_args(list(argv))
    out: dict[str, Any] = {}
    for key, value in sorted(vars(ns).items()):
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = str(value)
    return out


def dest_to_argv(parser: argparse.ArgumentParser, dest: str, value: Any) -> list[str]:
    """The CLI tokens that set ``dest`` to ``value``; fail-closed if none exist."""
    for action in parser._actions:  # noqa: SLF001 — argparse offers no public walk
        if action.dest != dest or not action.option_strings:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ContractError(f"{dest}: BooleanOptionalAction needs a bool")
            return [flag if value else f"--no-{flag[2:]}"]
        if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
            if value is not True:
                raise ContractError(
                    f"{dest}: store_true flag cannot be set to {value!r}"
                )
            return [flag]
        if isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
            if value is not False:
                raise ContractError(
                    f"{dest}: store_false flag cannot be set to {value!r}"
                )
            return [flag]
        return [flag, str(value)]
    raise ContractError(
        f"{dest}: no CLI flag sets this dest; it cannot be pinned by argv"
    )


def eager_tracker_policy_argv(
    declared: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Tracker-policy keys the eager preset must be brought to, as argv.

    Lifted from ``declared.tracker_policy.source_preset`` (the production s
    preset): every resolved key that differs from the eager preset and is not
    an execution-substrate / artifact key.  A differing key with no CLI flag
    fails the freeze rather than being dropped.
    """
    policy = declared["tracker_policy"]
    source = resolve_config(["--preset", policy["source_preset"]])
    target = resolve_config(["--preset", policy["eager_preset"]])
    excluded = EXECUTION_KEYS | set(policy.get("excluded_keys", ()))
    parser = build_mot17_parser()
    overrides: dict[str, Any] = {}
    argv: list[str] = []
    for key in sorted(set(source) | set(target)):
        if key in excluded or source.get(key) == target.get(key):
            continue
        overrides[key] = source.get(key)
        argv.extend(dest_to_argv(parser, key, source.get(key)))
    return argv, overrides


# --------------------------------------------------------------------------- recipes from the matrix


def _row_by_id(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["comparison_id"]: row for row in matrix["comparisons"]}


def _node(inventory: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    node = inventory["nodes"].get(node_id)
    if node is None:
        raise ContractError(f"node {node_id!r} not in inventory")
    return node


def _recipe_id(
    node: str,
    preset: str,
    argv: Sequence[str],
    binding: str,
    runtime: Mapping[str, Any],
) -> str:
    """Stable label; the authoritative identity is the resolved config + artifact sha."""
    if binding == "family_preset":
        parts = [f"wg:{preset}"]
        if "--no-mamba-trt" in argv and "mamba_head_engine" in load_preset(preset):
            parts.append("ckpt-head")
        if "--fpn-backbone-engine" in argv:
            parts.append(f"engine={runtime.get('backbone_engine_node')}")
        return "/".join(parts) + f":{node}"
    if binding == "shared_backbone_node":
        return f"pyt:{runtime['backbone_node']}:{node}"
    raise ContractError(f"unknown binding {binding!r} for {node}")


def _matrix_argv(runtime: Mapping[str, Any]) -> list[str]:
    sketch = runtime.get("recipe")
    if not sketch:
        raise ContractError("runtime binding carries no recipe sketch")
    tokens = shlex.split(sketch)
    if not tokens or tokens[0] != str(MOT17_ENTRY):
        raise ContractError(
            f"recipe sketch does not start with {MOT17_ENTRY}: {sketch}"
        )
    return tokens[1:]


def derive_recipe(
    side: Mapping[str, Any],
    inventory: Mapping[str, Any],
    declared: Mapping[str, Any],
    policy_argv: Sequence[str],
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    node_id = side["node"]
    runtime = side["runtime"]
    if runtime.get("status") != "bound":
        raise ContractError(
            f"{node_id}: runtime binding is {runtime.get('status')!r}, not bound"
        )
    node = _node(inventory, node_id)
    argv = _matrix_argv(runtime)
    binding = runtime["binding"]
    profiles = declared["execution_profiles"]
    if binding == "family_preset":
        profile = "whole_graph_serial"
        argv += list(profiles[profile]["fixed_argv"])
    else:
        profile = "eager_pytorch"
        yolo_weights = ((node.get("summary") or {}).get("args") or {}).get(
            "yolo_weights"
        )
        if not yolo_weights:
            raise ContractError(
                f"{node_id}: checkpoint args record no yolo_weights for the eager backbone"
            )
        argv += ["--mamba-yolo-weights", str(yolo_weights)]
        if node.get("kind") == "gated_teacher_ckpt":
            # The eager preset names a mamba_ckpt by default and mot17.py
            # refuses --teacher-head-ckpt beside it; clear it explicitly.
            # TeacherHeadDetector never reads mamba_teacher_ckpt, but the
            # preset default (runs/gated_det_v1) would otherwise sit in the
            # identity as a difference the pair has to explain.
            backbone = _node(inventory, runtime["backbone_node"])
            argv += ["--mamba-ckpt", "", "--mamba-teacher-ckpt", backbone["path"]]
        argv += list(profiles[profile]["fixed_argv"])
        argv += list(policy_argv)
    config = resolve_config(argv)
    artifacts = expected_artifacts(config, inventory, prior)
    identity_config = pairing_config(config, declared)
    return {
        "recipe_id": _recipe_id(node_id, _preset_of(argv), argv, binding, runtime),
        "node": node_id,
        "family": node.get("family"),
        "node_kind": node.get("kind"),
        "binding": binding,
        "execution_profile": profile,
        "preset": _preset_of(argv),
        "preset_path": str(preset_path(_preset_of(argv)).relative_to(REPO_ROOT)),
        "preset_sha256": sha256_file(preset_path(_preset_of(argv))),
        "argv": argv,
        "command": " ".join(shlex.quote(t) for t in [str(MOT17_ENTRY), *argv]),
        "artifacts": artifacts,
        "resolved_config_sha256": sha256_json(identity_config),
        "resolved_config_keys": len(identity_config),
        "binding_facts": {
            key: runtime.get(key)
            for key in (
                "head_source",
                "head_path",
                "effective_T",
                "temporal_blocks",
                "backbone_engine_node",
                "backbone_node",
                "backbone_teacher_consistency",
                "engine_bytes_attributed",
                "graphs",
                "gate_teacher_at_runtime",
                "tracker_policy_source",
            )
        },
        "tracker_policy_overrides": list(policy_argv)
        if binding != "family_preset"
        else [],
        "comparisons": [],
    }


def expected_artifacts(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """sha256 per artifact key, pinned to the inventory node where one exists.

    ``prior`` maps artifact path → the entry a previous freeze recorded.  A
    file that is present is always re-hashed and must agree with the
    inventory; a file that is absent (CI has no ``runs/`` or ``models/``)
    may carry its prior entry over, so ``check`` can still detect preset /
    argparse / CLI drift there.  Absent with no prior is an error.
    """
    by_path = {
        node["path"]: nid
        for nid, node in inventory["nodes"].items()
        if node.get("path") and node.get("sha256")
    }
    prior = prior or {}
    out: dict[str, Any] = {}
    for key in ARTIFACT_KEYS:
        value = config.get(key)
        if not value:
            out[key] = None
            continue
        rel = str(value)
        path = (REPO_ROOT / rel).resolve()
        nid = by_path.get(rel)
        if not path.is_file():
            carried = prior.get(rel)
            if carried is None:
                raise ContractError(
                    f"artifact {key}={rel} does not exist; the recipe cannot be frozen"
                )
            out[key] = dict(carried)
            continue
        entry: dict[str, Any] = {"path": rel, "size_bytes": path.stat().st_size}
        if nid is not None:
            entry["node"] = nid
            entry["sha256"] = inventory["nodes"][nid]["sha256"]
            entry["sha256_source"] = "inventory"
            on_disk = sha256_file(path)
            if on_disk != entry["sha256"]:
                raise ContractError(
                    f"artifact {key}={rel} differs from inventory node {nid}: "
                    f"{on_disk[:12]} vs {entry['sha256'][:12]}; re-capture the inventory first"
                )
        else:
            entry["node"] = None
            entry["sha256"] = sha256_file(path)
            entry["sha256_source"] = "disk_at_freeze"
        out[key] = entry
    return out


def prior_artifacts(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Artifact entries of a committed contract, by path."""
    out: dict[str, Any] = {}
    for recipe in (contract.get("recipes") or {}).values():
        for entry in (recipe.get("artifacts") or {}).values():
            if entry:
                out[entry["path"]] = entry
    return out


def pairing_config(
    config: Mapping[str, Any], declared: Mapping[str, Any]
) -> dict[str, Any]:
    local = set(declared["identity"]["run_local_keys"])
    return {k: v for k, v in config.items() if k not in local}


def _axis_paths(treatment_axes: Iterable[str], declared: Mapping[str, Any]) -> set[str]:
    mapping = declared["treatment_axis_to_identity_paths"]
    paths: set[str] = set()
    for axis in treatment_axes:
        if axis not in mapping:
            raise ContractError(
                f"treatment axis {axis!r} has no identity-path mapping in the contract"
            )
        paths.update(mapping[axis])
    return paths


def recipe_identity_view(
    recipe: Mapping[str, Any], declared: Mapping[str, Any]
) -> dict[str, Any]:
    config = resolve_config(recipe["argv"])
    return {
        "config": pairing_config(config, declared),
        "artifacts": {
            k: (v or {}).get("sha256") if v else None
            for k, v in recipe["artifacts"].items()
        },
        "execution_profile": recipe["execution_profile"],
        "preset_sha256": recipe["preset_sha256"],
    }


def flat_diff(
    lhs: Mapping[str, Any], rhs: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Leaf-level differences as ``{path: {"lhs": ..., "rhs": ...}}``."""
    out: dict[str, Any] = {}
    for key in sorted(set(lhs) | set(rhs)):
        a, b = lhs.get(key), rhs.get(key)
        path = f"{prefix}{key}"
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out.update(flat_diff(a, b, f"{path}."))
        elif a != b:
            out[path] = {"lhs": a, "rhs": b}
    return out


def derive_recipes_and_pairs(
    matrix: Mapping[str, Any],
    inventory: Mapping[str, Any],
    declared: Mapping[str, Any],
    prior: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = _row_by_id(matrix)
    policy_argv, policy_overrides = eager_tracker_policy_argv(declared)
    recipes: dict[str, Any] = {}
    pairs: dict[str, Any] = {}
    for cid in declared["prepared_comparisons"]:
        row = rows.get(cid)
        if row is None:
            raise ContractError(f"prepared comparison {cid!r} is not in the matrix")
        if row["classification"] not in ("controlled", "system_comparison"):
            raise ContractError(
                f"{cid}: classification {row['classification']!r} cannot be prepared for measurement"
            )
        sides = {}
        for side in ("lhs", "rhs"):
            recipe = derive_recipe(row[side], inventory, declared, policy_argv, prior)
            rid = recipe["recipe_id"]
            if rid in recipes:
                if (
                    recipes[rid]["resolved_config_sha256"]
                    != recipe["resolved_config_sha256"]
                ):
                    raise ContractError(
                        f"recipe id {rid} resolves to two different configs"
                    )
            else:
                recipes[rid] = recipe
            recipes[rid]["comparisons"].append(f"{cid}:{side}")
            sides[side] = rid
        lhs_view = recipe_identity_view(recipes[sides["lhs"]], declared)
        rhs_view = recipe_identity_view(recipes[sides["rhs"]], declared)
        diff = flat_diff(lhs_view, rhs_view)
        if row["classification"] == "controlled":
            allowed = _axis_paths(row["treatment_axes"], declared)
            unexplained = sorted(set(diff) - allowed)
            if unexplained:
                raise ContractError(
                    f"{cid}: controlled pair differs outside its treatment: {unexplained}"
                )
            if sides["lhs"] == sides["rhs"]:
                raise ContractError(f"{cid}: both sides resolve to the same recipe")
            allowed_list = sorted(allowed)
            allowed_source = "treatment_axes"
        else:
            allowed_list = sorted(diff)
            allowed_source = "system_diff_at_freeze"
            if not diff:
                raise ContractError(f"{cid}: system comparison with identical recipes")
        variance_axis = (
            "training_seed" if row["design"] == "seed_replicate" else "treatment"
        )
        pairs[cid] = {
            "design": row["design"],
            "classification": row["classification"],
            "treatment_axes": list(row["treatment_axes"]),
            "lhs_recipe": sides["lhs"],
            "rhs_recipe": sides["rhs"],
            "allowed_differences": allowed_list,
            "allowed_differences_source": allowed_source,
            "frozen_differences": diff,
            "variance_axis": variance_axis,
            "question": row.get("question"),
            "intended_treatment": row.get("intended_treatment"),
            "remaining_confounds": [
                c.get("name") for c in row.get("remaining_confounds", [])
            ],
        }
    for recipe in recipes.values():
        recipe["comparisons"].sort()
    policy = {"argv": policy_argv, "overrides": policy_overrides}
    return recipes, pairs, policy


# --------------------------------------------------------------------------- dataset key


def _seqinfo(path: Path) -> dict[str, str]:
    import configparser

    cp = configparser.ConfigParser()
    cp.read(path)
    return dict(cp["Sequence"]) if cp.has_section("Sequence") else {}


def dataset_key(dataset: Mapping[str, Any], sequences: Sequence[str]) -> dict[str, Any]:
    """Per-sequence digest of everything the eval reads: frames, gt, seqinfo.

    Image bytes are hashed in full (the decode path is part of the measured
    system; a re-encoded frame is a different input).  ``max_frames`` is
    fixed to the full sequence, so ``n_frames`` is the seqinfo length.
    """
    root = REPO_ROOT / dataset["data_root"] / dataset["split"]
    per_seq: dict[str, Any] = {}
    for seq in sequences:
        seq_dir = root / seq
        if not seq_dir.is_dir():
            raise ContractError(f"sequence {seq} not found under {root}")
        info_path = seq_dir / "seqinfo.ini"
        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        for required in (info_path, gt_path):
            if not required.is_file():
                raise ContractError(f"{seq}: missing {required.relative_to(REPO_ROOT)}")
        if not img_dir.is_dir():
            raise ContractError(f"{seq}: missing img1/")
        info = _seqinfo(info_path)
        frames = sorted(p for p in img_dir.iterdir() if p.is_file())
        digest = hashlib.sha256()
        for frame in frames:
            digest.update(frame.name.encode())
            digest.update(b"\0")
            digest.update(sha256_file(frame).encode())
            digest.update(b"\n")
        n_frames = int(info.get("seqlength", len(frames)))
        if n_frames != len(frames):
            raise ContractError(
                f"{seq}: seqinfo seqLength {n_frames} != {len(frames)} files in img1/"
            )
        per_seq[seq] = {
            "n_frames": n_frames,
            "frame_rate": info.get("framerate"),
            "im_size": [int(info["imwidth"]), int(info["imheight"])]
            if "imwidth" in info and "imheight" in info
            else None,
            "seqinfo_sha256": sha256_file(info_path),
            "gt_sha256": sha256_file(gt_path),
            "img1_files": len(frames),
            "img1_digest": digest.hexdigest(),
        }
    return {
        "data_root": dataset["data_root"],
        "split": dataset["split"],
        "detector": dataset["detector"],
        "sequences": list(sequences),
        "max_frames": dataset.get("max_frames"),
        "per_sequence": per_seq,
        "key_sha256": sha256_json(per_seq),
    }


# --------------------------------------------------------------------------- environment


def _nvidia_smi(field: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return "; ".join(lines) if lines else None


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_build_version() -> str | None:
    """``torch.__version__`` (with its CUDA tag) read from version.py, torch not imported."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    text = (Path(spec.origin).parent / "version.py").read_text(errors="replace")
    m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    return m.group(1) if m else None


def torch_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent / "lib"


def build_dir() -> Path:
    from saccade.paths import build_dir as _build_dir

    resolved = _build_dir()
    if resolved is None:
        raise ContractError(
            "no native build directory resolvable (SACCADE_BUILD_PATH unset, no checkout)"
        )
    return resolved


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def trackeval_identity() -> dict[str, Any]:
    if os.environ.get("SACCADE_TRACKEVAL_ROOT", "").strip():
        raise ContractError(
            "SACCADE_TRACKEVAL_ROOT is set; the contract pins the vendored third_party/TrackEval"
        )
    rel = "third_party/TrackEval"
    root = REPO_ROOT / rel
    if not (root / "trackeval").is_dir():
        raise ContractError(
            f"{rel}/trackeval is missing; HOTA would be silently skipped"
        )
    tree = _git("rev-parse", f"HEAD:{rel}")
    status = _git("status", "--porcelain", "--", rel)
    return {"root": rel, "git_tree": tree, "dirty": bool(status)}


def native_extensions(build: Path) -> dict[str, str]:
    if not build.is_dir():
        raise ContractError(f"native build directory {build} does not exist")
    sos = sorted(p for p in build.glob("*.so") if p.is_file())
    if not sos:
        raise ContractError(f"no native extensions (*.so) under {build}")
    return {p.name: sha256_file(p) for p in sos}


def environment_identity(declared: Mapping[str, Any]) -> dict[str, Any]:
    """Everything about the host and toolchain a paired run must share."""
    build = build_dir()
    versions = {
        name: _dist_version(name)
        for name in ("torch", "torchvision", "numpy", "motmetrics")
    }
    # distribution names carry the CUDA flavour (tensorrt_cu12, nvidia-dali-cuda120);
    # record every installed one rather than guess a suffix
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        low = name.lower()
        if low.startswith(
            ("tensorrt", "nvidia-dali", "nvidia-nvjpeg", "nvidia-cuda-runtime")
        ):
            versions[name] = dist.version
    versions["torch_build"] = _torch_build_version()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "gpu": _nvidia_smi("name"),
        "driver": _nvidia_smi("driver_version"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "build_dir": str(build),
        "native_extensions": native_extensions(build),
        "trackeval": trackeval_identity(),
    }


def env_hygiene(declared: Mapping[str, Any]) -> dict[str, str]:
    """Refuse a shell that carries an escape hatch the config cannot see."""
    rules = declared["runtime_env"]
    prefix = rules["forbidden_prefix"]
    allowed = set(rules["allowed"])
    stray = sorted(k for k in os.environ if k.startswith(prefix) and k not in allowed)
    if stray:
        raise ContractError(
            f"environment carries {', '.join(stray)}; unset them (the contract forbids "
            f"{prefix}* hatches outside {sorted(allowed)})"
        )
    return {k: os.environ[k] for k in sorted(allowed) if k in os.environ}


def child_environment(declared: Mapping[str, Any], run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["SACCADE_BUILD_PATH"] = str(build_dir())
    lib = torch_lib_dir()
    if lib is not None:
        prior = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{prior}" if prior else str(lib)
    env.update(rm.parent_claim_environ(run_dir))
    return env


# --------------------------------------------------------------------------- identity


def runtime_identity(
    contract: Mapping[str, Any],
    recipe_id: str,
    *,
    stage: str,
    sequences: Sequence[str],
    verify_dataset: bool = True,
) -> dict[str, Any]:
    """The block ``run`` binds into ``run_manifest.json`` before the first byte.

    Every artifact and the preset are re-hashed on disk and must equal what
    the contract froze; the dataset key is recomputed and must equal the
    frozen one (or its sequence subset for a smoke run).
    """
    declared = contract["declared"]
    recipe = contract["recipes"].get(recipe_id)
    if recipe is None:
        raise ContractError(f"unknown recipe {recipe_id!r}")
    if stage not in STAGES:
        raise ContractError(f"stage must be one of {STAGES}, got {stage!r}")

    # preset drift
    preset_now = sha256_file(REPO_ROOT / recipe["preset_path"])
    if preset_now != recipe["preset_sha256"]:
        raise ContractError(
            f"preset {recipe['preset_path']} drifted: {preset_now[:12]} != frozen {recipe['preset_sha256'][:12]}"
        )
    # config drift (argparse defaults, preset content, CLI resolution)
    config = resolve_config(recipe["argv"])
    identity_config = pairing_config(config, declared)
    config_sha = sha256_json(identity_config)
    if config_sha != recipe["resolved_config_sha256"]:
        raise ContractError(
            f"resolved config drifted for {recipe_id}: {config_sha[:12]} != frozen "
            f"{recipe['resolved_config_sha256'][:12]} (argparse defaults or preset moved)"
        )
    # artifact drift
    artifacts: dict[str, Any] = {}
    for key, frozen in recipe["artifacts"].items():
        if frozen is None:
            artifacts[key] = None
            continue
        path = REPO_ROOT / frozen["path"]
        if not path.is_file():
            raise ContractError(f"artifact {key}={frozen['path']} is missing")
        now = sha256_file(path)
        if now != frozen["sha256"]:
            raise ContractError(
                f"artifact {key}={frozen['path']} drifted: {now[:12]} != frozen {frozen['sha256'][:12]}"
            )
        artifacts[key] = {
            "path": frozen["path"],
            "sha256": now,
            "node": frozen.get("node"),
        }
    # dataset drift
    frozen_key = contract["dataset_key"]
    unknown = sorted(set(sequences) - set(frozen_key["sequences"]))
    if unknown:
        raise ContractError(f"sequences {unknown} are not in the frozen dataset key")
    if verify_dataset:
        key_now = dataset_key(declared["dataset"], sequences)
        for seq in sequences:
            if key_now["per_sequence"][seq] != frozen_key["per_sequence"][seq]:
                raise ContractError(
                    f"dataset drifted for {seq}: per-sequence key differs from the frozen one"
                )
        per_seq = key_now["per_sequence"]
    else:
        per_seq = {seq: frozen_key["per_sequence"][seq] for seq in sequences}
    subset = list(sequences) != list(frozen_key["sequences"])
    env_allowed = env_hygiene(declared)
    environment = environment_identity(declared)
    pairing = {
        "contract_sha256": contract["frozen"]["contract_sha256"],
        "recipe_id": recipe_id,
        "execution_profile": recipe["execution_profile"],
        "preset_sha256": preset_now,
        "resolved_config_sha256": config_sha,
        "artifacts": {k: (v["sha256"] if v else None) for k, v in artifacts.items()},
        "dataset": {
            "sequences": list(sequences),
            "subset": subset,
            "key_sha256": sha256_json({s: per_seq[s] for s in sequences}),
        },
        "environment": environment,
        "commit": rm._git_head(),
    }
    return {
        "schema": IDENTITY_SCHEMA,
        "contract_path": str(DEFAULT_CONTRACT.relative_to(_HERE.parents[2])),
        "stage": stage,
        "identity_sha256": sha256_json(pairing),
        "pairing": pairing,
        "recipe": {
            "node": recipe["node"],
            "binding": recipe["binding"],
            "argv": list(recipe["argv"]),
            "binding_facts": recipe["binding_facts"],
            "comparisons": list(recipe["comparisons"]),
        },
        "resolved_config": identity_config,
        "artifacts": artifacts,
        "dataset_key": {
            "per_sequence": per_seq,
            "max_frames": frozen_key.get("max_frames"),
        },
        "env_allowed": env_allowed,
        "bound_at": _now(),
    }


# --------------------------------------------------------------------------- run


_METRIC_LINE = re.compile(r"^\s{2}([A-Za-z0-9_]+):\s+(.+?)\s*$")


def parse_overall_metrics(stdout: str) -> dict[str, Any]:
    """The ``=== OVERALL METRICS ===`` block ``mot17.py`` prints, as raw strings + floats."""
    lines = stdout.splitlines()
    try:
        start = max(
            i
            for i, line in enumerate(lines)
            if line.strip() == "=== OVERALL METRICS ==="
        )
    except ValueError:
        return {}
    raw: dict[str, str] = {}
    for line in lines[start + 1 :]:
        m = _METRIC_LINE.match(line)
        if not m:
            if raw:
                break
            continue
        raw[m.group(1)] = m.group(2)
    numeric: dict[str, float] = {}
    for key, value in raw.items():
        text = value.rstrip("%")
        try:
            numeric[key] = float(text)
        except ValueError:
            continue
    return {"raw": raw, "numeric": numeric}


def parse_overall_throughput(stdout: str) -> dict[str, float] | None:
    m = re.findall(
        r"Overall throughput: ([0-9.]+) FPS; mean latency: ([0-9.]+) ms", stdout
    )
    if not m:
        return None
    fps, mean_ms = m[-1]
    return {"fps": float(fps), "mean_latency_ms": float(mean_ms)}


def _mot_hashes(run_dir: Path) -> dict[str, str]:
    from eval_repeat_identity import list_mot_files, mot_md5

    return {
        name: mot_md5(path) for name, path in sorted(list_mot_files(run_dir).items())
    }


def _latency_profiles(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(run_dir.glob("_latency_profile*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        payload.pop("samples_ms", None)
        out[path.name] = payload
    return out


def run_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_RECORD_FILENAME
    if not path.is_file():
        raise ContractError(
            f"{run_dir}: no {RUN_RECORD_FILENAME}; the run did not finish under the contract runner"
        )
    payload = _read_json(path)
    if payload.get("schema") != RUN_RECORD_SCHEMA:
        raise ContractError(
            f"{path}: schema {payload.get('schema')!r} is not {RUN_RECORD_SCHEMA}"
        )
    return payload


def run_identity(run_dir: Path) -> dict[str, Any]:
    manifest = rm.require_manifest(run_dir)
    if rm.provenance_mode_of(manifest) != "production":
        raise ContractError(f"{run_dir}: manifest is not a production claim")
    identity = manifest.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise ContractError(f"{run_dir}: manifest carries no runtime_identity")
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise ContractError(
            f"{run_dir}: runtime_identity schema {identity.get('schema')!r} is not {IDENTITY_SCHEMA}"
        )
    return dict(identity)


def find_repeat_reports(recipe_root: Path, identity_sha: str) -> list[Path]:
    hits: list[Path] = []
    for path in sorted(recipe_root.glob(REPEAT_REPORT_GLOB)):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, ContractError):
            continue
        if payload.get("schema") != REPEAT_REPORT_SCHEMA:
            continue
        if payload.get("identity_sha256") == identity_sha and payload.get("complete"):
            hits.append(path)
    return hits


def _slug(recipe_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", recipe_id)


def preflight(
    contract: Mapping[str, Any],
    recipe_id: str,
    *,
    stage: str,
    root: Path,
    lease: str,
    verify_dataset: bool = True,
) -> dict[str, Any]:
    """Everything that must hold before a run directory is claimed."""
    declared = contract["declared"]
    procedure = declared["procedure"]
    if Path.cwd().resolve() != REPO_ROOT:
        raise ContractError(
            f"run from the checkout root {REPO_ROOT} (artifact paths are relative to it)"
        )
    if contract_sha256(contract) != contract["frozen"]["contract_sha256"]:
        raise ContractError(
            "contract file does not match its own frozen sha256; run `freeze` and commit"
        )
    sequences = (
        list(declared["dataset"]["smoke_sequences"])
        if stage == "smoke"
        else list(declared["dataset"]["sequences"])
    )
    identity = runtime_identity(
        contract,
        recipe_id,
        stage=stage,
        sequences=sequences,
        verify_dataset=verify_dataset,
    )
    reasons: list[str] = []
    if stage == "formal":
        if rm._git_dirty() is not False:
            reasons.append(
                "formal run requires a clean working tree (dirty tree or no git)"
            )
        if identity["pairing"]["commit"] is None:
            reasons.append("formal run requires a resolvable HEAD commit")
        if lease != procedure["formal"]["lease"]:
            reasons.append(
                f"formal run requires the {procedure['formal']['lease']!r} lease, got {lease!r}"
            )
        reports = find_repeat_reports(
            root / _slug(recipe_id), identity["identity_sha256"]
        )
        min_runs = int(procedure["repeat"]["min_runs"])
        if not any(int(_read_json(p).get("n_runs", 0)) >= min_runs for p in reports):
            reasons.append(
                f"formal run requires a complete repeat report with >= {min_runs} runs under "
                f"{root / _slug(recipe_id)} for identity {identity['identity_sha256'][:12]} (none found)"
            )
    if reasons:
        raise ContractError("preflight refused: " + "; ".join(reasons))
    return {"identity": identity, "sequences": sequences}


def _stream_child(cmd: Sequence[str], env: Mapping[str, str], run_dir: Path) -> int:
    with (
        (run_dir / STDOUT_FILENAME).open("w") as out,
        (run_dir / STDERR_FILENAME).open("w") as err,
    ):
        proc = subprocess.Popen(
            list(cmd),
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            sys.stdout.write(line)
        return proc.wait()


def execute_run(
    contract: Mapping[str, Any],
    recipe_id: str,
    *,
    stage: str,
    run_dir: Path,
    root: Path,
    lease: str,
    verify_dataset: bool = True,
) -> dict[str, Any]:
    """Preflight, claim, run ``mot17.py`` under the parent claim, record."""
    pre = preflight(
        contract,
        recipe_id,
        stage=stage,
        root=root,
        lease=lease,
        verify_dataset=verify_dataset,
    )
    identity = pre["identity"]
    recipe = contract["recipes"][recipe_id]
    argv = [
        *recipe["argv"],
        "--sequences",
        ",".join(pre["sequences"]),
        "--output",
        str(run_dir),
    ]
    cmd = [sys.executable, str(REPO_ROOT / MOT17_ENTRY), *argv]
    if lease != "none":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "resctl.py"),
            "run",
            "--wait",
            lease,
            "--",
            *cmd,
        ]
    rm.open_run(
        run_dir,
        produced_by="eval",
        preset=recipe["preset"],
        detector=contract["declared"]["dataset"]["detector"],
        dataset=f"{contract['declared']['dataset']['data_root']} {contract['declared']['dataset']['split']}",
        cmdline=cmd,
        runtime_identity=identity,
    )
    env = child_environment(contract["declared"], run_dir)
    started = _now()
    t0 = time.perf_counter()
    code = _stream_child(cmd, env, run_dir)
    wall = time.perf_counter() - t0
    stdout = (run_dir / STDOUT_FILENAME).read_text(errors="replace")
    metrics = parse_overall_metrics(stdout)
    numeric = metrics.get("numeric", {}) if metrics else {}
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in numeric]
    mot = _mot_hashes(run_dir)
    expected_files = set(pre["sequences"])
    complete = code == 0 and not missing and expected_files <= set(mot)
    incomplete_reasons: list[str] = []
    if code != 0:
        incomplete_reasons.append(f"exit code {code}")
    if missing:
        incomplete_reasons.append(
            f"metrics missing {missing} (HOTA absent = TrackEval silently skipped)"
        )
    if not expected_files <= set(mot):
        incomplete_reasons.append(
            f"MOT files missing for {sorted(expected_files - set(mot))}"
        )
    record = {
        "schema": RUN_RECORD_SCHEMA,
        "recipe_id": recipe_id,
        "stage": stage,
        "identity_sha256": identity["identity_sha256"],
        "contract_sha256": identity["pairing"]["contract_sha256"],
        "sequences": pre["sequences"],
        "lease": lease,
        "started_at": started,
        "finished_at": _now(),
        "wall_seconds": round(wall, 3),
        "exit_code": code,
        "complete": complete,
        "incomplete_reasons": incomplete_reasons,
        "metrics": metrics or None,
        "throughput": parse_overall_throughput(stdout),
        "latency_profiles": _latency_profiles(run_dir),
        "mot_md5": mot,
        "stdout": STDOUT_FILENAME,
        "stderr": STDERR_FILENAME,
        "formal_baseline": False if stage != "formal" else None,
        "note": (
            "smoke/repeat runs are procedure evidence, never a deliverable-4 baseline"
            if stage != "formal"
            else "formal run; baseline status is assigned by deliverable 4, not by this record"
        ),
    }
    (run_dir / RUN_RECORD_FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


# --------------------------------------------------------------------------- repeat report


def _metric_ranges(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for rec in records:
        keys.update(((rec.get("metrics") or {}).get("numeric") or {}).keys())
    out: dict[str, Any] = {}
    for key in sorted(keys):
        values = [
            float(((rec.get("metrics") or {}).get("numeric") or {})[key])
            for rec in records
            if key in ((rec.get("metrics") or {}).get("numeric") or {})
        ]
        if not values:
            continue
        out[key] = {
            "values": values,
            "min": min(values),
            "max": max(values),
            "observed_range": max(values) - min(values),
        }
    return out


def repeat_report(
    run_dirs: Sequence[Path], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Byte identity + observed metric range over repeated runs of one identity.

    The report never says "deterministic": it states n runs, k distinct
    outputs per sequence, and the observed metric range, which is a reading
    reference and not a bound (docs/research/eval/nogpudecode_reproducibility_20260907.md).
    """
    from eval_repeat_identity import compare_run_dirs

    min_runs = int(contract["declared"]["procedure"]["repeat"]["min_runs"])
    dirs = [Path(d) for d in run_dirs]
    identities = [run_identity(d) for d in dirs]
    records = [run_record(d) for d in dirs]
    shas = {i["identity_sha256"] for i in identities}
    reasons: list[str] = []
    if len(shas) != 1:
        reasons.append(
            f"runs carry {len(shas)} different identities; a repeat report covers exactly one"
        )
    stages = {i["stage"] for i in identities}
    if stages - {"repeat", "formal"}:
        reasons.append(
            f"stages {sorted(stages)} include a non-repeat stage (smoke is a subset run)"
        )
    if any(i["pairing"]["dataset"]["subset"] for i in identities):
        reasons.append(
            "a run covers a sequence subset; repeatability is read on the full contract"
        )
    incomplete = [str(d) for d, r in zip(dirs, records) if not r.get("complete")]
    if incomplete:
        reasons.append(f"incomplete runs: {incomplete}")
    if len(dirs) < min_runs:
        reasons.append(f"{len(dirs)} runs < min_runs {min_runs}")
    identity_report = compare_run_dirs(dirs).to_dict()
    distinct = {s["sequence"]: s["n_distinct"] for s in identity_report["reports"]}
    return {
        "schema": REPEAT_REPORT_SCHEMA,
        "generated_at": _now(),
        "recipe_id": identities[0]["pairing"]["recipe_id"] if identities else None,
        "identity_sha256": next(iter(shas)) if len(shas) == 1 else None,
        "n_runs": len(dirs),
        "run_dirs": [str(d) for d in dirs],
        "complete": not reasons,
        "reasons": reasons,
        "byte_identity": {
            "all_identical": bool(identity_report["ok"]),
            "distinct_outputs_per_sequence": distinct,
            "detail": identity_report,
        },
        "metric_observed_range": _metric_ranges(records),
        "throughput": [r.get("throughput") for r in records],
        "variance_axis": "runtime_repeat",
        "claim_rule": (
            f"n={len(dirs)}: report 'k distinct outputs in n runs' and the observed range; "
            "never 'deterministic' or 'bit-exact' (n runs bound only rates >~ 3/n)"
        ),
    }


# --------------------------------------------------------------------------- pair validation


def validate_pair(
    contract: Mapping[str, Any],
    comparison_id: str,
    lhs_dir: Path,
    rhs_dir: Path,
) -> dict[str, Any]:
    """Are two runs a paired measurement of ``comparison_id``?  Fail-closed."""
    pair = contract["pairs"].get(comparison_id)
    reasons: list[str] = []
    if pair is None:
        return _verdict(
            comparison_id,
            lhs_dir,
            rhs_dir,
            False,
            [f"{comparison_id} is not a prepared pair"],
            {},
        )
    sides: dict[str, dict[str, Any]] = {}
    for label, run_dir, expected in (
        ("lhs", lhs_dir, pair["lhs_recipe"]),
        ("rhs", rhs_dir, pair["rhs_recipe"]),
    ):
        try:
            identity = run_identity(run_dir)
            record = run_record(run_dir)
        except (ContractError, rm.ManifestError) as exc:
            reasons.append(f"{label}: {exc}")
            continue
        sides[label] = {"identity": identity, "record": record}
        if identity["stage"] not in PAIRABLE_STAGES:
            reasons.append(
                f"{label}: stage {identity['stage']!r} is not pairable (only {PAIRABLE_STAGES})"
            )
        if identity["pairing"]["recipe_id"] != expected:
            reasons.append(
                f"{label}: recipe {identity['pairing']['recipe_id']!r} != {expected!r}"
            )
        if (
            identity["pairing"]["contract_sha256"]
            != contract["frozen"]["contract_sha256"]
        ):
            reasons.append(f"{label}: run was bound to a different contract sha256")
        if identity["pairing"]["dataset"]["subset"]:
            reasons.append(f"{label}: sequence subset run")
        if not record.get("complete"):
            reasons.append(
                f"{label}: run incomplete ({record.get('incomplete_reasons')})"
            )
        if record.get("identity_sha256") != identity["identity_sha256"]:
            reasons.append(f"{label}: run record identity does not match the manifest")
    diff: dict[str, Any] = {}
    if len(sides) == 2:
        lhs_view = _pair_view(sides["lhs"]["identity"])
        rhs_view = _pair_view(sides["rhs"]["identity"])
        diff = flat_diff(lhs_view, rhs_view)
        allowed = set(pair["allowed_differences"])
        unexplained = sorted(set(diff) - allowed)
        if unexplained:
            reasons.append(f"differences outside the treatment: {unexplained}")
        if not diff:
            reasons.append(
                "the two runs are identical in every identity path; nothing is compared"
            )
    paired = not reasons
    return _verdict(
        comparison_id, lhs_dir, rhs_dir, paired, reasons, diff, pair=pair, sides=sides
    )


def _pair_view(identity: Mapping[str, Any]) -> dict[str, Any]:
    pairing = identity["pairing"]
    return {
        "config": identity["resolved_config"],
        "artifacts": pairing["artifacts"],
        "execution_profile": pairing["execution_profile"],
        "preset_sha256": pairing["preset_sha256"],
        "dataset_key_sha256": pairing["dataset"]["key_sha256"],
        "environment": pairing["environment"],
        "commit": pairing["commit"],
        "contract_sha256": pairing["contract_sha256"],
    }


def _verdict(
    comparison_id: str,
    lhs_dir: Path,
    rhs_dir: Path,
    paired: bool,
    reasons: list[str],
    diff: Mapping[str, Any],
    *,
    pair: Mapping[str, Any] | None = None,
    sides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {}
    for label in ("lhs", "rhs"):
        if sides and label in sides:
            metrics[label] = (sides[label]["record"].get("metrics") or {}).get(
                "numeric"
            )
    return {
        "schema": PAIR_VERDICT_SCHEMA,
        "generated_at": _now(),
        "comparison_id": comparison_id,
        "lhs_run": str(lhs_dir),
        "rhs_run": str(rhs_dir),
        "verdict": "paired" if paired else "not_paired",
        "reasons": reasons,
        "observed_differences": diff,
        "allowed_differences": list(pair["allowed_differences"]) if pair else [],
        "variance_axis": pair["variance_axis"] if pair else None,
        "classification": pair["classification"] if pair else None,
        "metrics": metrics,
        "note": (
            "paired = the two runs differ only where the declared treatment says; "
            "it is not a statement about effect size. Read any delta against the "
            "same-identity repeat report's observed range."
        ),
    }


# --------------------------------------------------------------------------- freeze / check


def freeze(
    contract_path: Path,
    matrix_path: Path,
    inventory_path: Path,
    *,
    with_dataset_key: bool = True,
) -> dict[str, Any]:
    """Re-derive every non-declared block from the committed inputs (and disk)."""
    prior = load_contract(contract_path)
    matrix = load_matrix(REPO_ROOT / matrix_path)
    inventory = load_inventory(REPO_ROOT / inventory_path)
    declared = prior["declared"]
    recipes, pairs, policy = derive_recipes_and_pairs(
        matrix, inventory, declared, prior_artifacts(prior)
    )
    if with_dataset_key:
        key = dataset_key(declared["dataset"], declared["dataset"]["sequences"])
        key["captured_at"] = _now()
        key["captured_host"] = socket.gethostname()
    else:
        key = prior.get("dataset_key")
        if not key:
            raise ContractError(
                "no dataset key to carry over; freeze with the dataset present"
            )
    out = {
        "schema": CONTRACT_SCHEMA,
        "issue": 421,
        "deliverable": 3,
        "declared": declared,
        "frozen": {
            "at": _now(),
            "commit": rm._git_head(),
            "matrix_path": str(matrix_path),
            "matrix_sha256": sha256_file(REPO_ROOT / matrix_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": sha256_file(REPO_ROOT / inventory_path),
            "eager_tracker_policy": policy,
            "contract_sha256": None,
        },
        "recipes": recipes,
        "pairs": pairs,
        "dataset_key": key,
    }
    out["frozen"]["contract_sha256"] = contract_sha256(out)
    return out


def _stable_view(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The contract minus its timestamps, for freshness comparison."""
    body = json.loads(json.dumps(contract))
    body["frozen"].pop("at", None)
    body["frozen"].pop("commit", None)
    body["frozen"].pop("contract_sha256", None)
    body["dataset_key"].pop("captured_at", None)
    body["dataset_key"].pop("captured_host", None)
    return body


def check(
    contract_path: Path,
    matrix_path: Path,
    inventory_path: Path,
    md_path: Path,
    *,
    verify_dataset: bool = False,
) -> list[str]:
    """Reasons the committed contract is stale; empty means fresh."""
    reasons: list[str] = []
    committed = load_contract(contract_path)
    if committed["frozen"].get("contract_sha256") != contract_sha256(committed):
        reasons.append("frozen.contract_sha256 does not match the file")
    derived = freeze(
        contract_path, matrix_path, inventory_path, with_dataset_key=verify_dataset
    )
    a, b = _stable_view(committed), _stable_view(derived)
    if a != b:
        diff = flat_diff(a, b)
        reasons.append(
            "derived blocks differ from the committed contract: "
            + ", ".join(sorted(diff)[:12])
        )
    rendered = render_markdown(committed)
    if not (REPO_ROOT / md_path).is_file():
        reasons.append(f"{md_path} missing")
    elif (REPO_ROOT / md_path).read_text() != rendered:
        reasons.append(f"{md_path} is stale; re-run freeze")
    return reasons


# --------------------------------------------------------------------------- markdown


def _md_kv(mapping: Mapping[str, Any], indent: str = "") -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            lines.append(f"{indent}- **{key}**:")
            lines.extend(_md_kv(value, indent + "  "))
        elif (
            isinstance(value, list) and value and all(isinstance(v, str) for v in value)
        ):
            lines.append(f"{indent}- **{key}**: " + ", ".join(f"`{v}`" for v in value))
        else:
            lines.append(f"{indent}- **{key}**: {_fmt(value)}")
    return lines


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f"`{json.dumps(value, ensure_ascii=False)}`"


def render_markdown(contract: Mapping[str, Any]) -> str:
    d = contract["declared"]
    fz = contract["frozen"]
    key = contract["dataset_key"]
    L: list[str] = []
    L += [
        "<!-- doc-status: active -->",
        "<!-- doc-promotion: none -->",
        f"<!-- doc-date: {fz['at'][:10]} -->",
        "<!-- doc-module: detection -->",
        "<!-- Generated by scripts/provenance/training_eval_contract.py from scripts/provenance/training_eval_contract.json; regenerate rather than edit. -->",
        "",
        "# Training eval contract (#421 · deliverable 3)",
        "",
        f"Frozen {fz['at']} at `{(fz.get('commit') or 'unknown')[:12]}`; contract sha256 `{fz['contract_sha256'][:16]}…`. "
        f"Machine-readable source: `scripts/provenance/training_eval_contract.json` (the `declared` block is hand-written; "
        f"`recipes`, `pairs`, `dataset_key` and `frozen` are derived by `freeze` from `{fz['matrix_path']}` "
        f"(`{fz['matrix_sha256'][:12]}`) and `{fz['inventory_path']}` (`{fz['inventory_sha256'][:12]}`); `check` refuses drift).",
        "",
        "This document fixes **how** a paired measurement over the deliverable-2 matrix is taken. It contains no "
        "IDF1/HOTA/FPS; deliverable 4 runs the recipes below. A run that was not produced by "
        "`training_eval_contract.py run` carries no `runtime_identity` and can never be marked paired.",
        "",
        "## 1. Scope and status",
        "",
        f"- {d['scope']}",
        "- Out of scope (each needs its own declaration before any number is cited): "
        + "; ".join(d["out_of_scope"]),
        "",
        "## 2. Dataset and sequence key",
        "",
        f"- `{key['data_root']}` split `{key['split']}`, detector `{key['detector']}`, "
        f"max_frames {'unset (full sequences)' if key['max_frames'] is None else key['max_frames']}.",
        f"- Smoke stage runs only {', '.join(f'`{s}`' for s in d['dataset']['smoke_sequences'])}; smoke runs are sequence-subset runs and are never pairable.",
        f"- Sequence key sha256 `{key['key_sha256'][:16]}…` captured {key.get('captured_at')} on `{key.get('captured_host')}`; "
        "every run recomputes the per-sequence digests (seqinfo, gt.txt, all image bytes) and refuses on mismatch.",
        "",
        "| sequence | frames | size | seqinfo sha | gt sha | img1 digest |",
        "|---|---:|---|---|---|---|",
    ]
    for seq, entry in key["per_sequence"].items():
        size = (
            "×".join(str(x) for x in entry["im_size"]) if entry.get("im_size") else "?"
        )
        L.append(
            f"| `{seq}` | {entry['n_frames']} | {size} | `{entry['seqinfo_sha256'][:12]}` | "
            f"`{entry['gt_sha256'][:12]}` | `{entry['img1_digest'][:12]}` |"
        )
    L += ["", "## 3. Evaluator, ignore rules, metric versions", ""]
    L += _md_kv(d["evaluator"])
    L += ["", "## 4. Timing boundaries", ""]
    L += _md_kv(d["timing"])
    L += ["", "## 5. Execution profiles", ""]
    for name, prof in d["execution_profiles"].items():
        L.append(f"### `{name}`")
        L.append("")
        L += _md_kv(prof)
        L.append("")
    L += ["## 6. Tracker policy for eager rows", ""]
    L += _md_kv(d["tracker_policy"])
    pol = fz["eager_tracker_policy"]
    L.append(
        f"- **derived overrides** (argv appended to every eager recipe): `{' '.join(pol['argv'])}`"
    )
    L.append(f"- **resolved keys**: `{json.dumps(pol['overrides'], sort_keys=True)}`")
    L += ["", "## 7. Runtime environment hygiene", ""]
    L += _md_kv(d["runtime_env"])
    L += ["", "## 8. Runtime identity (what a run binds before its first byte)", ""]
    L += _md_kv(d["identity"])
    L += [
        "",
        "Treatment axis → identity paths a paired run may differ in:",
        "",
    ]
    for axis, paths in d["treatment_axis_to_identity_paths"].items():
        L.append(f"- `{axis}` → " + ", ".join(f"`{p}`" for p in paths))
    L += ["", "## 9. Procedure: smoke → repeat → formal", ""]
    L += _md_kv(d["procedure"])
    L += ["", "## 10. Variance axes", ""]
    L += _md_kv(d["variance_axes"])
    L += ["", "## 11. Recipes", ""]
    L += [
        "Each recipe is one endpoint under one runtime binding. `argv` is exact; the runner appends only "
        "`--sequences <contract>` and `--output <run dir>`. `config sha` is the sha256 of every resolved "
        "argparse dest (preset merge + argv) minus run-local keys; any drift of argparse defaults, the preset "
        "file or the CLI changes it and the runner refuses to start.",
        "",
        "| recipe | node | profile | head source | eff. T | preset sha | config sha | used by |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for rid, r in contract["recipes"].items():
        hs = r["binding_facts"].get("head_source") or {}
        L.append(
            f"| `{rid}` | `{r['node']}` | `{r['execution_profile']}` | {hs.get('kind')} "
            f"({'ckpt head runs' if hs.get('checkpoint_head_deployed') else 'fixed engine head'}) | "
            f"{r['binding_facts'].get('effective_T')} | `{r['preset_sha256'][:12]}` | "
            f"`{r['resolved_config_sha256'][:12]}` | {', '.join(f'`{c}`' for c in r['comparisons'])} |"
        )
    L.append("")
    for rid, r in contract["recipes"].items():
        L.append(f"### `{rid}`")
        L.append("")
        L.append("```")
        L.append(
            f".venv/bin/python {r['command']} --sequences <contract> --output <run dir>"
        )
        L.append("```")
        L.append("")
        L.append(
            f"- binding `{r['binding']}`; head path: {r['binding_facts'].get('head_path')}; temporal blocks: {r['binding_facts'].get('temporal_blocks')}"
        )
        L.append(
            f"- backbone/teacher consistency: `{r['binding_facts'].get('backbone_teacher_consistency')}`; engine bytes attributed: `{r['binding_facts'].get('engine_bytes_attributed')}`"
        )
        L.append("- artifacts (sha256 pinned; runner re-hashes):")
        for key_name, art in r["artifacts"].items():
            if art:
                L.append(
                    f"  - `{key_name}` = `{art['path']}` `{art['sha256'][:12]}` ({art.get('node') or art['sha256_source']})"
                )
        if r.get("tracker_policy_overrides"):
            L.append(
                f"- tracker policy overrides: `{' '.join(r['tracker_policy_overrides'])}`"
            )
        L.append("")
    L += ["## 12. Pairs", ""]
    L += [
        "`allowed_differences` is frozen per pair: for a `controlled` pair it is the identity paths its treatment "
        "axes map to (the freeze fails if the two recipes differ anywhere else); for a `system_comparison` it is "
        "the exact difference set observed at freeze time. `validate-pair` refuses any other difference, any "
        "non-formal stage, any contract/identity mismatch, and any incomplete run.",
        "",
        "| comparison | design | class | lhs recipe | rhs recipe | variance axis | allowed differences |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid, p in contract["pairs"].items():
        L.append(
            f"| `{cid}` | {p['design']} | {p['classification']} | `{p['lhs_recipe']}` | `{p['rhs_recipe']}` | "
            f"{p['variance_axis']} | {', '.join(f'`{a}`' for a in p['allowed_differences'])} |"
        )
    L += ["", "## 13. Commands", ""]
    L += [
        "```",
        ".venv/bin/python scripts/provenance/training_eval_contract.py check                       # contract fresh?",
        ".venv/bin/python scripts/provenance/training_eval_contract.py show <recipe|pair>",
        ".venv/bin/python scripts/provenance/training_eval_contract.py preflight <recipe> --stage formal",
        ".venv/bin/python scripts/provenance/training_eval_contract.py run <recipe> --stage smoke",
        ".venv/bin/python scripts/provenance/training_eval_contract.py run <recipe> --stage repeat --runs 3",
        ".venv/bin/python scripts/provenance/training_eval_contract.py run <recipe> --stage formal --lease machine-bench",
        ".venv/bin/python scripts/provenance/training_eval_contract.py repeat-report <run dir>...",
        ".venv/bin/python scripts/provenance/training_eval_contract.py validate-pair <comparison> <lhs run> <rhs run>",
        "```",
        "",
        f"Runs land under `{DEFAULT_RUN_ROOT}/<recipe slug>/<stage>-<utc stamp>-rNN/` with `run_manifest.json` "
        f"(schema v3, `runtime_identity` bound before the first byte), `{STDOUT_FILENAME}`, `{STDERR_FILENAME}`, "
        f"the MOT files, the evaluator's `_latency_profile*.json` / `_fps_summary.txt`, and `{RUN_RECORD_FILENAME}` "
        "(exit code, parsed metrics, throughput, MOT md5, completeness). Repeat reports are written beside the "
        "runs as `repeat-<stamp>-report.json`.",
        "",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD_OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_freeze = sub.add_parser(
        "freeze", help="re-derive recipes/pairs/dataset key and write contract + doc"
    )
    s_freeze.add_argument(
        "--no-dataset-key",
        action="store_true",
        help="carry the committed dataset key over",
    )

    s_check = sub.add_parser(
        "check", help="exit 1 if the committed contract or doc is stale"
    )
    s_check.add_argument(
        "--verify-dataset", action="store_true", help="also re-hash the dataset on disk"
    )

    s_show = sub.add_parser("show", help="print a recipe or pair")
    s_show.add_argument("name")

    for name in ("preflight", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("recipe")
        sp.add_argument("--stage", choices=STAGES, required=True)
        sp.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
        sp.add_argument(
            "--lease", choices=("none", "gpu0", "machine-bench"), default=None
        )
        sp.add_argument(
            "--no-verify-dataset",
            action="store_true",
            help="trust the frozen dataset key (smoke only)",
        )
        if name == "run":
            sp.add_argument("--runs", type=int, default=1)

    s_rep = sub.add_parser(
        "repeat-report", help="byte identity + observed range over run dirs"
    )
    s_rep.add_argument("run_dirs", nargs="+", type=Path)
    s_rep.add_argument("--emit", type=Path, default=None)

    s_pair = sub.add_parser(
        "validate-pair", help="decide whether two formal runs are paired"
    )
    s_pair.add_argument("comparison_id")
    s_pair.add_argument("lhs_run", type=Path)
    s_pair.add_argument("rhs_run", type=Path)
    s_pair.add_argument("--emit", type=Path, default=None)

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except (ContractError, rm.ManifestError) as exc:
        print(f"training_eval_contract: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "freeze":
        payload = freeze(
            args.contract,
            args.matrix,
            args.inventory,
            with_dataset_key=not args.no_dataset_key,
        )
        _write(args.contract, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        _write(REPO_ROOT / args.md, render_markdown(payload))
        print(
            f"froze {len(payload['recipes'])} recipes / {len(payload['pairs'])} pairs; "
            f"contract sha256 {payload['frozen']['contract_sha256'][:12]} -> {args.contract}, {args.md}"
        )
        return 0
    if args.cmd == "check":
        reasons = check(
            args.contract,
            args.matrix,
            args.inventory,
            args.md,
            verify_dataset=args.verify_dataset,
        )
        if reasons:
            for reason in reasons:
                print(f"STALE: {reason}")
            return 1
        print("training_eval_contract: fresh")
        return 0
    contract = load_contract(args.contract)
    if args.cmd == "show":
        if args.name in contract["recipes"]:
            r = contract["recipes"][args.name]
            print(json.dumps(r, indent=2))
            print(
                f"\n.venv/bin/python {r['command']} --sequences <contract> --output <run dir>"
            )
            return 0
        if args.name in contract["pairs"]:
            print(json.dumps(contract["pairs"][args.name], indent=2))
            return 0
        raise ContractError(
            f"{args.name!r} is neither a recipe nor a pair; recipes: {list(contract['recipes'])}"
        )
    if args.cmd in ("preflight", "run"):
        lease = args.lease or ("machine-bench" if args.stage == "formal" else "gpu0")
        if args.no_verify_dataset and args.stage != "smoke":
            raise ContractError("--no-verify-dataset is only allowed for smoke runs")
        root = Path(args.root)
        if args.cmd == "preflight":
            pre = preflight(
                contract,
                args.recipe,
                stage=args.stage,
                root=root,
                lease=lease,
                verify_dataset=not args.no_verify_dataset,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "identity_sha256": pre["identity"]["identity_sha256"],
                        "sequences": pre["sequences"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.runs < 1:
            raise ContractError("--runs must be >= 1")
        stamp = _stamp()
        recipe_root = root / _slug(args.recipe)
        dirs: list[Path] = []
        for k in range(1, args.runs + 1):
            run_dir = recipe_root / f"{args.stage}-{stamp}-r{k:02d}"
            record = execute_run(
                contract,
                args.recipe,
                stage=args.stage,
                run_dir=run_dir,
                root=root,
                lease=lease,
                verify_dataset=not args.no_verify_dataset,
            )
            dirs.append(run_dir)
            print(
                f"[{args.stage} r{k:02d}] {'complete' if record['complete'] else 'INCOMPLETE'} "
                f"exit={record['exit_code']} wall={record['wall_seconds']}s -> {run_dir}"
            )
            if not record["complete"]:
                print(f"  reasons: {record['incomplete_reasons']}")
        if args.stage in ("repeat", "formal") and args.runs >= 1:
            report = repeat_report(dirs, contract)
            out = recipe_root / f"repeat-{stamp}-report.json"
            _write(out, json.dumps(report, indent=2) + "\n")
            print(
                f"repeat report: n={report['n_runs']} complete={report['complete']} "
                f"all_identical={report['byte_identity']['all_identical']} -> {out}"
            )
            if not report["complete"]:
                print(f"  reasons: {report['reasons']}")
        return 0
    if args.cmd == "repeat-report":
        report = repeat_report(args.run_dirs, contract)
        text = json.dumps(report, indent=2) + "\n"
        if args.emit:
            _write(args.emit, text)
        else:
            print(text)
        return 0 if report["complete"] else 1
    if args.cmd == "validate-pair":
        verdict = validate_pair(
            contract, args.comparison_id, args.lhs_run, args.rhs_run
        )
        text = json.dumps(verdict, indent=2) + "\n"
        if args.emit:
            _write(args.emit, text)
        print(text if not args.emit else f"{verdict['verdict']} -> {args.emit}")
        return 0 if verdict["verdict"] == "paired" else 1
    raise ContractError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
