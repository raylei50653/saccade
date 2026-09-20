# status: stable
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

* ``campaign`` inventories a run root after the procedure has been run for
  every recipe (deliverable-3 closeout): per recipe the formal runs, the
  same-identity repeat report behind them, byte agreement between the two,
  and every invariant a reader would otherwise re-derive (one identity, one
  clean commit, this contract, the bench lease, complete records, no
  unaccounted run dir); per pair, which formal runs ``validate-pair`` can be
  handed.  The committed snapshot (``report_data/training_eval_campaign.json``
  + its generated doc) is the surviving record of the gitignored raw runs.

* ``baselines`` (deliverable 4): campaign inventory = the only input naming
  runs, contract = the only input naming pairs; every lhs × rhs formal
  combination of every prepared pair goes through ``validate-pair`` plus a
  per-side clean-tree / bench-lease / campaign-commit gate; a baseline row
  forms only from ``paired`` verdicts (baseline side = the pair's lhs), each
  delta is read against print precision and both sides' runtime-repeat
  ranges, ``remaining_confounds`` stay on the row, plain-GT2 ↔ T3→T1 is
  historical only.  Observed differences, never effect claims; ``--audit``
  re-checks a committed report from the three JSONs without the raw runs.

Usage:
    .venv/bin/python scripts/provenance/training_eval_contract.py freeze
    .venv/bin/python scripts/provenance/training_eval_contract.py check [--verify-dataset]
    .venv/bin/python scripts/provenance/training_eval_contract.py show <recipe|pair>
    .venv/bin/python scripts/provenance/training_eval_contract.py preflight <recipe> --stage formal
    .venv/bin/python scripts/provenance/training_eval_contract.py run <recipe> --stage smoke|repeat|formal [--runs N]
    .venv/bin/python scripts/provenance/training_eval_contract.py repeat-report <run_dir>...
    .venv/bin/python scripts/provenance/training_eval_contract.py validate-pair <comparison_id> <lhs_run> <rhs_run>
    .venv/bin/python scripts/provenance/training_eval_contract.py campaign [--root DIR] --emit report_data/training_eval_campaign.json --campaign-md docs/research/training/training_eval_campaign.md
    .venv/bin/python scripts/provenance/training_eval_contract.py baselines [--campaign JSON] --emit report_data/training_eval_baselines.json --baselines-md docs/research/training/training_eval_baselines.md [--audit]
"""

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
DEFAULT_CAMPAIGN_OUT = Path("report_data/training_eval_campaign.json")
DEFAULT_CAMPAIGN_MD = Path("docs/research/training/training_eval_campaign.md")
DEFAULT_BASELINES_OUT = Path("report_data/training_eval_baselines.json")
DEFAULT_BASELINES_MD = Path("docs/research/training/training_eval_baselines.md")

CONTRACT_SCHEMA = "training_eval_contract_v1"
IDENTITY_SCHEMA = "training_eval_runtime_identity_v1"
RUN_RECORD_SCHEMA = "training_eval_run_record_v1"
REPEAT_REPORT_SCHEMA = "training_eval_repeat_report_v1"
PAIR_VERDICT_SCHEMA = "training_eval_pair_verdict_v1"
CAMPAIGN_SCHEMA = "training_eval_campaign_v1"
BASELINES_SCHEMA = "training_eval_baselines_v1"
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
        "dirty": rm._git_dirty(),
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
    if stage in ("repeat", "formal"):
        # The identity binds ``commit`` and ``dirty``; a dirty repeat could
        # otherwise be run on edited sources, cleaned up, and still unlock a
        # formal run at the same HEAD.  Both stages therefore require the
        # tree the commit describes.
        if identity["pairing"]["dirty"] is not False:
            reasons.append(
                f"{stage} run requires a clean working tree (dirty tree or no git)"
            )
        if identity["pairing"]["commit"] is None:
            reasons.append(f"{stage} run requires a resolvable HEAD commit")
    if stage == "formal":
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
    if stages - {"repeat"}:
        reasons.append(
            f"stages {sorted(stages)} include a non-repeat stage (a repeat report reads "
            "repeat-stage runs only; smoke is a subset run, formal runs are what it unlocks)"
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
        "dirty": pairing["dirty"],
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


# --------------------------------------------------------------------------- campaign inventory


def _run_dirs_under(recipe_root: Path) -> list[Path]:
    if not recipe_root.is_dir():
        return []
    return sorted(
        d
        for d in recipe_root.iterdir()
        if d.is_dir() and any(d.name.startswith(f"{s}-") for s in STAGES)
    )


def campaign_recipe(
    contract: Mapping[str, Any], recipe_id: str, root: Path
) -> dict[str, Any]:
    """One recipe's formal runs and the same-identity repeat report behind them.

    Every invariant a deliverable-4 reader would otherwise re-derive by hand
    is checked here and written down as a reason when it fails: one identity
    across the formal runs, clean tree, this contract, the bench lease,
    complete records, a complete same-identity repeat report of at least
    ``repeat.min_runs``, and no run directory under the recipe root that the
    inventory does not account for (a stray dirty or superseded run must be
    moved out, not silently ignored).
    """
    procedure = contract["declared"]["procedure"]
    recipe_root = root / _slug(recipe_id)
    reasons: list[str] = []
    formal: list[dict[str, Any]] = []
    identities: set[str] = set()
    commits: set[str] = set()
    environment: dict[str, Any] | None = None
    formal_dirs = [
        d for d in _run_dirs_under(recipe_root) if d.name.startswith("formal-")
    ]
    for run_dir in formal_dirs:
        try:
            identity = run_identity(run_dir)
            record = run_record(run_dir)
        except (ContractError, rm.ManifestError) as exc:
            reasons.append(f"{run_dir.name}: {exc}")
            continue
        pairing = identity["pairing"]
        identities.add(identity["identity_sha256"])
        commits.add(pairing["commit"])
        environment = environment or pairing["environment"]
        if identity["stage"] != "formal" or record.get("stage") != "formal":
            reasons.append(f"{run_dir.name}: not a formal-stage run")
        if pairing["recipe_id"] != recipe_id:
            reasons.append(f"{run_dir.name}: recipe {pairing['recipe_id']!r}")
        if pairing["contract_sha256"] != contract["frozen"]["contract_sha256"]:
            reasons.append(f"{run_dir.name}: bound to a different contract sha256")
        if pairing["dirty"]:
            reasons.append(f"{run_dir.name}: dirty tree")
        if pairing["dataset"]["subset"]:
            reasons.append(f"{run_dir.name}: sequence subset run")
        if record.get("lease") != procedure["formal"]["lease"]:
            reasons.append(
                f"{run_dir.name}: lease {record.get('lease')!r} != {procedure['formal']['lease']!r}"
            )
        if not record.get("complete"):
            reasons.append(
                f"{run_dir.name}: incomplete {record.get('incomplete_reasons')}"
            )
        if record.get("identity_sha256") != identity["identity_sha256"]:
            reasons.append(f"{run_dir.name}: record identity != manifest identity")
        formal.append(
            {
                "run_dir": str(run_dir),
                "identity_sha256": identity["identity_sha256"],
                "commit": pairing["commit"],
                "metrics": (record.get("metrics") or {}).get("numeric"),
                "throughput": record.get("throughput"),
                "mot_md5": record.get("mot_md5") or {},
                "wall_seconds": record.get("wall_seconds"),
            }
        )
    if len(formal) < procedure["formal"]["min_runs"]:
        reasons.append(
            f"{len(formal)} formal runs < formal.min_runs {procedure['formal']['min_runs']}"
        )
    if len(identities) > 1:
        reasons.append(f"formal runs carry {len(identities)} identities")
    identity_sha = next(iter(identities)) if len(identities) == 1 else None

    report_path: Path | None = None
    report: dict[str, Any] | None = None
    if identity_sha:
        hits = find_repeat_reports(recipe_root, identity_sha)
        if hits:
            report_path = hits[-1]
            report = _read_json(report_path)
            if report["n_runs"] < procedure["repeat"]["min_runs"]:
                reasons.append(
                    f"repeat report has {report['n_runs']} runs < repeat.min_runs"
                )
            missing = [d for d in report["run_dirs"] if not Path(d).is_dir()]
            if missing:
                reasons.append(f"repeat report cites missing run dirs {missing}")
        else:
            reasons.append("no complete same-identity repeat report")

    accounted = {Path(d).resolve() for d in (report or {}).get("run_dirs", [])}
    accounted |= {d.resolve() for d in formal_dirs}
    stray = [
        d.name for d in _run_dirs_under(recipe_root) if d.resolve() not in accounted
    ]
    if stray:
        reasons.append(
            f"unaccounted run dirs under {recipe_root} (move superseded runs out): {stray}"
        )

    repeat_hashes: dict[str, set[str]] = {}
    if report:
        for seq_report in report["byte_identity"]["detail"]["reports"]:
            repeat_hashes[seq_report["sequence"]] = set(seq_report["hashes"])
    formal_distinct: dict[str, int] = {}
    for run in formal:
        run["matches_repeat_output"] = bool(repeat_hashes) and all(
            run["mot_md5"].get(seq) in hashes for seq, hashes in repeat_hashes.items()
        )
    for seq in sorted({s for run in formal for s in run["mot_md5"]}):
        formal_distinct[seq] = len({run["mot_md5"].get(seq) for run in formal})

    return {
        "recipe_id": recipe_id,
        "run_root": str(recipe_root),
        "identity_sha256": identity_sha,
        "commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "environment": environment,
        "complete": not reasons,
        "reasons": reasons,
        "repeat": (
            {
                "report": str(report_path),
                "n_runs": report["n_runs"],
                "distinct_outputs_per_sequence": report["byte_identity"][
                    "distinct_outputs_per_sequence"
                ],
                "metric_observed_range": {
                    k: {
                        "min": v["min"],
                        "max": v["max"],
                        "observed_range": v["observed_range"],
                    }
                    for k, v in report["metric_observed_range"].items()
                },
                "throughput": report["throughput"],
            }
            if report
            else None
        ),
        "formal": formal,
        "formal_distinct_outputs_per_sequence": formal_distinct,
    }


def campaign_inventory(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Inventory of every contract recipe's repeat evidence and formal runs.

    Deliverable-3 closeout and deliverable-4 input: which formal runs exist,
    which same-identity repeat report backs each, whether all of them sit on
    one clean commit under this contract, and which prepared pairs have a
    formal run on both sides.  It assigns no baseline status, decides no
    pair (``validate-pair`` does), and never says "deterministic".
    """
    recipes = {
        recipe_id: campaign_recipe(contract, recipe_id, root)
        for recipe_id in contract["recipes"]
    }
    reasons: list[str] = []
    commits = {r["commit"] for r in recipes.values() if isinstance(r["commit"], str)}
    if len(commits) > 1:
        reasons.append(
            f"formal runs span {len(commits)} commits; pairs across commits are not paired"
        )
    incomplete = [k for k, r in recipes.items() if not r["complete"]]
    if incomplete:
        reasons.append(f"recipes without a complete inventory: {incomplete}")
    pairs = {}
    for cid, pair in contract["pairs"].items():
        lhs, rhs = recipes[pair["lhs_recipe"]], recipes[pair["rhs_recipe"]]
        pairs[cid] = {
            "classification": pair["classification"],
            "variance_axis": pair["variance_axis"],
            "lhs_recipe": pair["lhs_recipe"],
            "rhs_recipe": pair["rhs_recipe"],
            "lhs_formal_runs": [f["run_dir"] for f in lhs["formal"]],
            "rhs_formal_runs": [f["run_dir"] for f in rhs["formal"]],
            "both_sides_formal": bool(lhs["formal"] and rhs["formal"])
            and lhs["complete"]
            and rhs["complete"],
        }
    n_repeat = [r["repeat"]["n_runs"] for r in recipes.values() if r["repeat"]]
    return {
        "schema": CAMPAIGN_SCHEMA,
        "generated_at": _now(),
        "contract_sha256": contract["frozen"]["contract_sha256"],
        "run_root": str(root),
        "commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "complete": not reasons,
        "reasons": reasons,
        "n_recipes": len(recipes),
        "n_recipes_complete": len(recipes) - len(incomplete),
        "recipes": recipes,
        "pairs": pairs,
        "variance_axis": "runtime_repeat",
        "claim_rule": (
            "per recipe: 'k distinct outputs in n repeat runs' (n="
            f"{'..'.join(str(x) for x in sorted({min(n_repeat), max(n_repeat)})) if n_repeat else 0}) "
            "and the observed metric range; never 'deterministic' or 'bit-exact'"
        ),
        "note": (
            "campaign inventory, not a baseline table: baseline status and every "
            "pair verdict are assigned by deliverable 4 (validate-pair) against the "
            "same-identity repeat report's observed range and the pair's confounds"
        ),
    }


def render_campaign_table(inventory: Mapping[str, Any]) -> str:
    lines = [
        f"{'recipe':<58} {'rep':>3} {'kmax':>4} {'HOTA':>5} {'IDF1':>5} {'fps':>6} {'formal':>6} ok"
    ]
    for recipe_id, r in inventory["recipes"].items():
        rep = r["repeat"]
        first = (
            r["formal"][0]["metrics"]
            if r["formal"] and r["formal"][0]["metrics"]
            else {}
        )
        fps = [f["throughput"]["fps"] for f in r["formal"] if f.get("throughput")]
        lines.append(
            f"{recipe_id:<58} "
            f"{rep['n_runs'] if rep else 0:>3} "
            f"{max(rep['distinct_outputs_per_sequence'].values()) if rep else '-':>4} "
            f"{first.get('HOTA', '-'):>5} {first.get('IDF1', '-'):>5} "
            f"{(sum(fps) / len(fps)) if fps else 0:>6.1f} "
            f"{len(r['formal']):>6} {'yes' if r['complete'] else 'NO'}"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fmt_count(value: Any) -> str:
    return "—" if value is None else str(int(value))


def render_campaign_markdown(inventory: Mapping[str, Any]) -> str:
    """Human view of ``campaign_inventory``; regenerate, never edit."""
    date = inventory["generated_at"][:10]
    commit = inventory["commit"]
    commit_s = (
        commit[:12] if isinstance(commit, str) else ", ".join(c[:12] for c in commit)
    )
    recipes = inventory["recipes"]
    n_rep = sorted({r["repeat"]["n_runs"] for r in recipes.values() if r["repeat"]})
    kmax = max(
        (
            max(r["repeat"]["distinct_outputs_per_sequence"].values())
            for r in recipes.values()
            if r["repeat"]
        ),
        default=None,
    )
    rmax = max(
        (
            v["observed_range"]
            for r in recipes.values()
            if r["repeat"]
            for v in r["repeat"]["metric_observed_range"].values()
        ),
        default=None,
    )
    envs = {
        (r["environment"] or {}).get("hostname"): r["environment"]
        for r in recipes.values()
        if r["environment"]
    }
    out: list[str] = [
        "<!-- doc-status: active -->",
        "<!-- doc-promotion: report_data -->",
        f"<!-- doc-date: {date} -->",
        "<!-- doc-module: detection -->",
        f"<!-- Generated by scripts/provenance/training_eval_contract.py campaign from {DEFAULT_CAMPAIGN_OUT}; regenerate rather than edit. -->",
        "",
        "# Training eval campaign inventory (#421 · deliverable 3 closeout)",
        "",
        f"Generated {inventory['generated_at']} from `{inventory['run_root']}` (gitignored raw runs); "
        f"contract sha256 `{inventory['contract_sha256'][:16]}…`; every run bound at commit `{commit_s}`. "
        f"Machine-readable: `{DEFAULT_CAMPAIGN_OUT}`.",
        "",
        "This is the **run inventory** the contract's procedure produced, not a baseline table: "
        "baseline status and every pair verdict are assigned by deliverable 4 (`validate-pair`) against the "
        "same-identity repeat report's observed range and the pair's recorded confounds. "
        "A number below is a formal run's printed metric (one decimal); it inherits no claim beyond that.",
        "",
        "## 1. Status",
        "",
        f"- Inventory complete: **{'yes' if inventory['complete'] else 'NO'}** "
        f"({inventory['n_recipes_complete']}/{inventory['n_recipes']} recipes with a same-identity repeat report and complete formal runs on one clean commit under this contract).",
    ]
    for reason in inventory["reasons"]:
        out.append(f"- INCOMPLETE: {reason}")
    out += [
        f"- Repeat evidence: n = {', '.join(map(str, n_rep)) or '—'} runs per recipe; "
        f"max distinct outputs in any sequence = {_fmt(kmax)}; max observed metric range across all keys = {_fmt(rmax)}. "
        f"Claim rule: {inventory['claim_rule']}.",
        "- Formal runs: each recipe's formal outputs are compared byte-wise (MOT md5) against its repeat runs "
        "(`matches_repeat_output`) — a formal run that diverges from its own repeat set is flagged, not averaged in.",
        f"- Pairs with a formal run on both sides: "
        f"{sum(p['both_sides_formal'] for p in inventory['pairs'].values())}/{len(inventory['pairs'])}.",
        f"- Variance axis of everything here: `{inventory['variance_axis']}` (fixed checkpoints re-run); "
        "training-seed variance lives only in the D1–D4 pairs' *treatment*, never in a repeat report.",
        "",
        "## 2. Environment",
        "",
    ]
    for env in envs.values():
        out.append(
            f"- `{env.get('hostname')}` · {env.get('platform', '')} · {env.get('gpu')} (driver {env.get('driver')}) · "
            f"python {env.get('python')} · packages {json.dumps(env.get('packages'), sort_keys=True)} · "
            f"TrackEval tree `{(env.get('trackeval') or {}).get('git_tree', '')[:12]}` · "
            f"{len(env.get('native_extensions') or {})} native extensions digested"
        )
    out += [
        "",
        "## 3. Per-recipe inventory",
        "",
        "`k` = distinct outputs across the n repeat runs (max over sequences); `range` = max observed metric range over all keys; "
        "metrics are formal r01 as printed; `formal=repeat` = every formal run's MOT files are byte-identical to the repeat set; "
        "fps = mean over formal runs (serial profile; eager rows are a lower bound by declaration).",
        "",
        "| recipe | identity | n | k | range | formal | formal=repeat | HOTA | DetA | AssA | IDF1 | MOTA | IDs | FP | FN | fps | ok |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for recipe_id, r in recipes.items():
        rep = r["repeat"]
        first = (r["formal"][0]["metrics"] if r["formal"] else None) or {}
        fps = [f["throughput"]["fps"] for f in r["formal"] if f.get("throughput")]
        out.append(
            "| `{rid}` | `{ident}` | {n} | {k} | {rng} | {nf} | {eq} | {hota} | {deta} | {assa} | {idf1} | {mota} | {ids} | {fp} | {fn} | {fps} | {ok} |".format(
                rid=recipe_id,
                ident=(r["identity_sha256"] or "")[:12],
                n=rep["n_runs"] if rep else 0,
                k=_fmt(max(rep["distinct_outputs_per_sequence"].values()))
                if rep
                else "—",
                rng=_fmt(
                    max(
                        v["observed_range"]
                        for v in rep["metric_observed_range"].values()
                    )
                )
                if rep
                else "—",
                nf=len(r["formal"]),
                eq="yes"
                if r["formal"] and all(f["matches_repeat_output"] for f in r["formal"])
                else "NO",
                hota=_fmt(first.get("HOTA")),
                deta=_fmt(first.get("DetA")),
                assa=_fmt(first.get("AssA")),
                idf1=_fmt(first.get("IDF1")),
                mota=_fmt(first.get("MOTA")),
                ids=_fmt_count(first.get("IDs")),
                fp=_fmt_count(first.get("FP")),
                fn=_fmt_count(first.get("FN")),
                fps=f"{sum(fps) / len(fps):.1f}" if fps else "—",
                ok="yes" if r["complete"] else "NO",
            )
        )
        for reason in r["reasons"]:
            out.append(f"| | | | | | | | | | | | | | | | | ↳ {reason} |")
    out += [
        "",
        "## 4. Prepared pairs — formal runs available to `validate-pair`",
        "",
        "| pair | class | variance axis | lhs formal | rhs formal | both sides |",
        "|---|---|---|---:|---:|---|",
    ]
    for cid, p in inventory["pairs"].items():
        out.append(
            f"| `{cid}` | {p['classification']} | {p['variance_axis']} | {len(p['lhs_formal_runs'])} | "
            f"{len(p['rhs_formal_runs'])} | {'yes' if p['both_sides_formal'] else 'NO'} |"
        )
    out += [
        "",
        "## 5. Handoff to deliverable 4",
        "",
        "1. For each prepared pair, run `training_eval_contract.py validate-pair <pair> <lhs formal run> <rhs formal run>` "
        "on the formal runs listed in section 4 (all three lhs × rhs combinations are the same measurement only if "
        "`formal=repeat` holds on both sides; the inventory says so per recipe). A pair is a baseline row only with a `paired` verdict.",
        "2. Read every delta against the same-identity repeat report's observed range (section 3, `range`) and the print "
        "precision (one decimal): a delta at or below the range is 'not distinguishable from run-to-run variation'; a delta "
        "above it is not by itself an effect — the pair's `remaining_confounds` (contract §pairs) stay attached to the row.",
        "3. Only the 15 prepared pairs are pairs. Any other cross-recipe reading of section 3 (e.g. `gt2_plain` vs `t3t1_phase_b` "
        "in either family) is **not** a prepared comparison: every plain-GT2 ↔ T3→T1 pairing carries the `warmup_epochs` 5→3 "
        "schedule difference (and, for s, a seed difference) outside the curriculum treatment "
        "(`docs/research/training/training_comparison_matrix.md`), so it may be cited only as historical/confounded, never as a training effect.",
        "4. `n` repeat runs with `k = 1` bound only divergence rates above ~1−0.05^(1/n); the words 'deterministic' and 'bit-exact' "
        "are not available to deliverable 4 either.",
        "5. Raw runs live under the gitignored run root (manifests v3, stdout/stderr, MOT files, latency profiles); this inventory "
        "and the repeat reports are the surviving committed record. Superseded pre-merge smoke / dirty-repeat runs were moved to "
        "`results/training_eval_contract_superseded_20260919_pre_merge/` and are never pairable.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- deliverable 4: baselines

# Print precision of the evaluator's OVERALL METRICS line (declared.evaluator.
# precision_of_record): percent metrics at one decimal, counts as integers.
# A delta below this is not resolvable under the contract; the throughput
# line prints two decimals.
COUNT_METRIC_KEYS = frozenset({"IDs", "FP", "FN"})
PERCENT_PRINT_PRECISION = 0.1
COUNT_PRINT_PRECISION = 1.0
THROUGHPUT_PRINT_PRECISION = 0.01
HEADLINE_METRIC_KEYS = ("HOTA", "IDF1", "MOTA")

# Cross-recipe readings that deliverable 4 may cite only as historical: the
# matrix row is the authority for their confounds (``remaining_confounds``
# with ``warmup_epochs`` 5→3 in every plain-GT2 ↔ T3→T1 pairing, plus the
# seed for s); the campaign recipes named here are the formal runs the
# numbers come from.  They are not contract pairs, so ``validate-pair``
# refuses them by construction and no baseline row is formed.
HISTORICAL_COMPARISONS: dict[str, tuple[str, str]] = {
    "C6.s.plain_gt2_vs_t3t1_unpaired_original": (
        "wg:mamba_whole_graph:s.gt2_plain",
        "wg:mamba_whole_graph:s.t3t1_phase_b",
    ),
    "C9.m.plain_gt2_vs_t3t1": (
        "wg:mamba_whole_graph_m/ckpt-head:m.gt2_plain",
        "wg:mamba_whole_graph_m:m.t3t1_phase_b",
    ),
}

# Reading groups the doc summarises: which prepared pairs answer which
# question, and which group is the attributability reference of which.
READING_GROUPS: dict[str, dict[str, Any]] = {
    "explicit_vs_implicit_shared_gt1": {
        "prefixes": ("C1.", "C2.", "C3."),
        "reference_group": "seed_replicates",
        "question": "explicit T3->T1 staging vs implicit all-frames T=4 from the shared GT1, three seed-paired replicates (s)",
    },
    "seed_replicates": {
        "prefixes": ("D1.", "D2.", "D3.", "D4."),
        "reference_group": None,
        "question": "training-seed variability of one recipe (implicit arm 42/43/44, explicit arm 42/43/44); the attributability reference for the C group",
    },
    "stage_increments": {
        "prefixes": ("B1.", "B2.", "B3.", "B4."),
        "reference_group": None,
        "question": "distill -> GT1 -> plain GT2 stage bundles (s and m); one pair per stage, no seed replicate",
    },
    "system_comparisons": {
        "prefixes": ("E1.", "E2.", "E3."),
        "reference_group": None,
        "question": "s vs m production systems; native Detect head vs Mamba head on a shared PyTorch backbone (eager, s tracker policy)",
    },
    "engine_ab": {
        "prefixes": ("E4.",),
        "reference_group": None,
        "question": "the s production backbone engine (sibling ONNX = legacy teacher) vs the head's own teacher engine, same head",
    },
}


def metric_print_precision(key: str) -> float:
    return (
        COUNT_PRINT_PRECISION if key in COUNT_METRIC_KEYS else PERCENT_PRINT_PRECISION
    )


def _rel_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def delta_reading(
    lhs: float,
    rhs: float,
    *,
    precision: float,
    lhs_observed_range: float | None,
    rhs_observed_range: float | None,
) -> dict[str, Any]:
    """One metric's delta read against print precision and both repeat ranges.

    ``reading`` is one of ``no_observed_difference`` (delta is 0 at print
    precision), ``within_runtime_repeat_range`` (|delta| <= the larger of the
    two sides' observed ranges), ``above_runtime_repeat_range`` (|delta| >
    both ranges).  The last one is a statement about the instrument, not
    about cause: an observed range of 0.0 means this batch of repeats saw no
    variation, not that none exists, and the pair's confounds stay attached.
    """
    delta = round(rhs - lhs, 6)
    floor = max(
        lhs_observed_range if lhs_observed_range is not None else 0.0,
        rhs_observed_range if rhs_observed_range is not None else 0.0,
    )
    if abs(delta) < precision:
        reading = "no_observed_difference"
    elif abs(delta) <= floor:
        reading = "within_runtime_repeat_range"
    else:
        reading = "above_runtime_repeat_range"
    return {
        "lhs": lhs,
        "rhs": rhs,
        "delta_rhs_minus_lhs": delta,
        "print_precision": precision,
        "lhs_observed_range": lhs_observed_range,
        "rhs_observed_range": rhs_observed_range,
        "runtime_repeat_floor": floor,
        "reading": reading,
    }


def _throughput_range(values: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    fps = [float(v["fps"]) for v in values if v and v.get("fps") is not None]
    if not fps:
        return {"values": [], "min": None, "max": None, "observed_range": None}
    return {
        "values": fps,
        "min": min(fps),
        "max": max(fps),
        "observed_range": round(max(fps) - min(fps), 6),
    }


def d4_side_gate(
    label: str,
    identity: Mapping[str, Any],
    record: Mapping[str, Any],
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    run_dir: str,
) -> list[str]:
    """Procedure facts ``validate-pair`` does not re-check (it compares the two sides
    to each other; two dirty runs are equal in ``dirty``).  A baseline row needs
    each side on its own to be a clean-tree, bench-lease formal run that the
    campaign inventory lists for its recipe on the campaign's commit."""
    reasons: list[str] = []
    pairing = identity["pairing"]
    procedure = contract["declared"]["procedure"]["formal"]
    if pairing.get("dirty"):
        reasons.append(f"{label}: dirty tree")
    if record.get("lease") != procedure["lease"]:
        reasons.append(
            f"{label}: lease {record.get('lease')!r} != {procedure['lease']!r}"
        )
    if pairing.get("commit") != campaign.get("commit"):
        reasons.append(
            f"{label}: commit {str(pairing.get('commit'))[:12]} is not the campaign commit"
        )
    recipe_entry = campaign["recipes"].get(pairing["recipe_id"])
    listed = {f["run_dir"] for f in (recipe_entry or {}).get("formal", [])}
    if run_dir not in listed:
        reasons.append(
            f"{label}: {run_dir} is not a formal run the campaign inventory lists"
        )
    elif identity["identity_sha256"] != recipe_entry["identity_sha256"]:
        reasons.append(f"{label}: identity differs from the campaign inventory's")
    return reasons


def d4_pair_verdict(
    contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    comparison_id: str,
    lhs_run: str,
    rhs_run: str,
) -> dict[str, Any]:
    """``validate-pair`` plus the per-side procedure gate; fail-closed."""
    verdict = validate_pair(
        contract, comparison_id, _rel_run_dir(lhs_run), _rel_run_dir(rhs_run)
    )
    verdict["lhs_run"], verdict["rhs_run"] = lhs_run, rhs_run
    extra: list[str] = []
    for label, run_dir in (("lhs", lhs_run), ("rhs", rhs_run)):
        try:
            identity = run_identity(_rel_run_dir(run_dir))
            record = run_record(_rel_run_dir(run_dir))
        except (ContractError, rm.ManifestError):
            continue  # already a validate-pair reason
        extra += d4_side_gate(label, identity, record, campaign, contract, run_dir)
    if extra:
        verdict["reasons"] = list(verdict["reasons"]) + extra
        verdict["verdict"] = "not_paired"
    return verdict


def _pair_confounds(
    contract: Mapping[str, Any], matrix_rows: Mapping[str, Mapping[str, Any]], cid: str
) -> list[dict[str, Any]]:
    """The pair's ``remaining_confounds`` (names from the contract, detail from the
    matrix row the contract was frozen from).  Missing or disagreeing lists are a
    contract/matrix inconsistency and refuse the row."""
    pair = contract["pairs"][cid]
    if "remaining_confounds" not in pair or not isinstance(
        pair["remaining_confounds"], list
    ):
        raise ContractError(f"{cid}: contract pair carries no remaining_confounds list")
    names = list(pair["remaining_confounds"])
    row = matrix_rows.get(cid)
    if row is None:
        raise ContractError(f"{cid}: not a matrix row")
    detail = {c.get("name"): c for c in row.get("remaining_confounds", [])}
    if sorted(detail) != sorted(names):
        raise ContractError(
            f"{cid}: contract remaining_confounds {names} != matrix {sorted(detail)}"
        )
    return [
        {
            "name": name,
            "severity": detail[name].get("severity"),
            "detail": detail[name].get("detail"),
        }
        for name in names
    ]


def baseline_pair(
    contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    matrix_rows: Mapping[str, Mapping[str, Any]],
    cid: str,
) -> dict[str, Any]:
    """One prepared pair: every lhs × rhs formal combination through the verdict,
    and a baseline row only when all of them are ``paired`` and each side's formal
    runs agree with each other.  The baseline side is the pair's lhs by contract
    orientation; nothing here ranks or re-selects."""
    if cid not in contract["pairs"]:
        raise ContractError(f"{cid}: not a prepared pair")
    if cid not in campaign.get("pairs", {}):
        raise ContractError(f"{cid}: not in the campaign inventory")
    pair = contract["pairs"][cid]
    inv = campaign["pairs"][cid]
    if (
        inv["lhs_recipe"] != pair["lhs_recipe"]
        or inv["rhs_recipe"] != pair["rhs_recipe"]
        or inv["classification"] != pair["classification"]
        or inv["variance_axis"] != pair["variance_axis"]
    ):
        raise ContractError(
            f"{cid}: campaign inventory disagrees with the contract pair"
        )
    confounds = _pair_confounds(contract, matrix_rows, cid)
    lhs_runs, rhs_runs = list(inv["lhs_formal_runs"]), list(inv["rhs_formal_runs"])
    reasons: list[str] = []
    if not lhs_runs or not rhs_runs:
        reasons.append("a side has no formal run in the campaign inventory")
    combinations = [
        d4_pair_verdict(contract, campaign, cid, lhs, rhs)
        for lhs in lhs_runs
        for rhs in rhs_runs
    ]
    not_paired = [c for c in combinations if c["verdict"] != "paired"]
    if not_paired:
        reasons.append(
            f"{len(not_paired)}/{len(combinations)} formal combinations are not_paired"
        )
    for label, key in (("lhs", "lhs"), ("rhs", "rhs")):
        views = {
            canonical_json(c["metrics"].get(key)) for c in combinations if c["metrics"]
        }
        if len(views) > 1:
            reasons.append(f"{label}: formal runs disagree in their metrics")
    canonical = combinations[0] if combinations else None
    lhs_entry = campaign["recipes"][pair["lhs_recipe"]]
    rhs_entry = campaign["recipes"][pair["rhs_recipe"]]
    row: dict[str, Any] | None = None
    if not reasons and canonical is not None:
        lhs_m, rhs_m = canonical["metrics"]["lhs"], canonical["metrics"]["rhs"]
        lhs_rng = lhs_entry["repeat"]["metric_observed_range"]
        rhs_rng = rhs_entry["repeat"]["metric_observed_range"]
        metrics = {
            key: delta_reading(
                float(lhs_m[key]),
                float(rhs_m[key]),
                precision=metric_print_precision(key),
                lhs_observed_range=lhs_rng.get(key, {}).get("observed_range"),
                rhs_observed_range=rhs_rng.get(key, {}).get("observed_range"),
            )
            for key in sorted(set(lhs_m) & set(rhs_m))
        }
        missing = set(REQUIRED_METRIC_KEYS) - set(metrics)
        if missing:
            raise ContractError(
                f"{cid}: required metric keys missing: {sorted(missing)}"
            )
        lhs_formal = {f["run_dir"]: f for f in lhs_entry["formal"]}
        rhs_formal = {f["run_dir"]: f for f in rhs_entry["formal"]}
        lhs_fps = _throughput_range(lhs_entry["repeat"]["throughput"])
        rhs_fps = _throughput_range(rhs_entry["repeat"]["throughput"])
        lhs_run_fps = float(lhs_formal[canonical["lhs_run"]]["throughput"]["fps"])
        rhs_run_fps = float(rhs_formal[canonical["rhs_run"]]["throughput"]["fps"])
        row = {
            "baseline_side": "lhs",
            "baseline_recipe": pair["lhs_recipe"],
            "treatment_recipe": pair["rhs_recipe"],
            "lhs_run": canonical["lhs_run"],
            "rhs_run": canonical["rhs_run"],
            "lhs_identity_sha256": lhs_entry["identity_sha256"],
            "rhs_identity_sha256": rhs_entry["identity_sha256"],
            "repeat_n": {
                "lhs": lhs_entry["repeat"]["n_runs"],
                "rhs": rhs_entry["repeat"]["n_runs"],
            },
            "repeat_distinct_outputs_max": {
                "lhs": max(
                    lhs_entry["repeat"]["distinct_outputs_per_sequence"].values()
                ),
                "rhs": max(
                    rhs_entry["repeat"]["distinct_outputs_per_sequence"].values()
                ),
            },
            "metrics": metrics,
            "throughput_fps": {
                **delta_reading(
                    lhs_run_fps,
                    rhs_run_fps,
                    precision=THROUGHPUT_PRINT_PRECISION,
                    lhs_observed_range=lhs_fps["observed_range"],
                    rhs_observed_range=rhs_fps["observed_range"],
                ),
                "lhs_repeat": lhs_fps,
                "rhs_repeat": rhs_fps,
                "lhs_formal": _throughput_range(
                    f.get("throughput") for f in lhs_entry["formal"]
                ),
                "rhs_formal": _throughput_range(
                    f.get("throughput") for f in rhs_entry["formal"]
                ),
                "profile": contract["recipes"][pair["lhs_recipe"]]["execution_profile"],
            },
            "observed_differences": canonical["observed_differences"],
        }
    return {
        "comparison_id": cid,
        "design": pair["design"],
        "classification": pair["classification"],
        "variance_axis": pair["variance_axis"],
        "treatment_axes": list(pair["treatment_axes"]),
        "intended_treatment": pair.get("intended_treatment"),
        "lhs_recipe": pair["lhs_recipe"],
        "rhs_recipe": pair["rhs_recipe"],
        "allowed_differences": list(pair["allowed_differences"]),
        "remaining_confounds": confounds,
        "n_combinations": len(combinations),
        "n_paired": len(combinations) - len(not_paired),
        "verdict": "paired" if not reasons else "not_paired",
        "reasons": reasons,
        "combinations": [
            {
                "lhs_run": c["lhs_run"],
                "rhs_run": c["rhs_run"],
                "verdict": c["verdict"],
                "reasons": c["reasons"],
            }
            for c in combinations
        ],
        "baseline_row": row,
    }


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def group_readings(pairs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Cross-row readings per group, computed from the rows only.

    Every statement is about observed differences: sign consistency across
    seed-paired replicates, whether every row is above its runtime-repeat
    floor, and (for the C group) whether the smallest |delta| exceeds the
    largest seed-replicate |delta| — the contract's attributability reference
    (``declared.variance_axes.rule``).  None of them upgrades a row to an
    effect claim.
    """
    out: dict[str, Any] = {}
    keys = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDs")
    for group, spec in READING_GROUPS.items():
        members = [cid for cid in pairs if cid.startswith(spec["prefixes"])]
        rows = {cid: pairs[cid]["baseline_row"] for cid in members}
        entry: dict[str, Any] = {
            "question": spec["question"],
            "members": members,
            "rows_available": [cid for cid, r in rows.items() if r],
            # every declared member present with a row; a partial group reads nothing
            "complete": all(
                any(cid.startswith(prefix) and rows[cid] for cid in members)
                for prefix in spec["prefixes"]
            ),
            "reference_group": spec["reference_group"],
            "per_metric": {},
        }
        if entry["complete"]:
            for key in keys:
                deltas = {
                    cid: r["metrics"][key]["delta_rhs_minus_lhs"]
                    for cid, r in rows.items()
                }
                readings = {
                    cid: r["metrics"][key]["reading"] for cid, r in rows.items()
                }
                signs = {_sign(d) for d in deltas.values()}
                entry["per_metric"][key] = {
                    "deltas": deltas,
                    "readings": readings,
                    "all_above_runtime_repeat_range": all(
                        v == "above_runtime_repeat_range" for v in readings.values()
                    ),
                    "sign_consistent": len(signs) == 1 and 0 not in signs,
                    "min_abs_delta": min(abs(d) for d in deltas.values()),
                    "max_abs_delta": max(abs(d) for d in deltas.values()),
                }
        out[group] = entry
    ref = out.get("seed_replicates")
    target = out.get("explicit_vs_implicit_shared_gt1")
    if target and target["complete"] and ref and ref["complete"]:
        for key, m in target["per_metric"].items():
            seed_max = ref["per_metric"][key]["max_abs_delta"]
            m["seed_replicate_max_abs_delta"] = seed_max
            m["min_abs_delta_exceeds_seed_replicate_max"] = (
                m["min_abs_delta"] > seed_max
            )
    return out


def historical_comparisons(
    contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    matrix_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Non-prepared cross-recipe readings, carried as historical only.

    Each entry records the ``validate-pair`` refusal (not a prepared pair),
    the identity paths the two formal runs actually differ in, the matrix
    row's training confounds, and the printed numbers — never a delta
    reading, never a baseline row.
    """
    out: list[dict[str, Any]] = []
    for cid, (lhs_recipe, rhs_recipe) in HISTORICAL_COMPARISONS.items():
        if cid in contract["pairs"]:
            raise ContractError(f"{cid}: listed as historical but is a prepared pair")
        row = matrix_rows.get(cid)
        if row is None or row.get("classification") != "historical_not_comparable":
            raise ContractError(f"{cid}: not a historical_not_comparable matrix row")
        if (
            lhs_recipe not in campaign["recipes"]
            or rhs_recipe not in campaign["recipes"]
        ):
            raise ContractError(
                f"{cid}: historical recipes are not in the campaign inventory"
            )
        lhs_entry = campaign["recipes"][lhs_recipe]
        rhs_entry = campaign["recipes"][rhs_recipe]
        lhs_run, rhs_run = (
            lhs_entry["formal"][0]["run_dir"],
            rhs_entry["formal"][0]["run_dir"],
        )
        verdict = validate_pair(
            contract, cid, _rel_run_dir(lhs_run), _rel_run_dir(rhs_run)
        )
        if verdict["verdict"] != "not_paired":
            raise ContractError(
                f"{cid}: a historical comparison must not validate as paired"
            )
        identity_diff = flat_diff(
            _pair_view(run_identity(_rel_run_dir(lhs_run))),
            _pair_view(run_identity(_rel_run_dir(rhs_run))),
        )
        schedule = (row.get("axes") or {}).get("training_schedule") or {}
        seed = (row.get("axes") or {}).get("training_seed") or {}
        out.append(
            {
                "comparison_id": cid,
                "classification": row["classification"],
                "intended_treatment": row.get("intended_treatment"),
                "lhs_recipe": lhs_recipe,
                "rhs_recipe": rhs_recipe,
                "lhs_run": lhs_run,
                "rhs_run": rhs_run,
                "validate_pair": {
                    "verdict": verdict["verdict"],
                    "reasons": verdict["reasons"],
                },
                "training_confounds": [
                    {
                        k: c.get(k)
                        for k in ("name", "severity", "status", "lhs", "rhs", "detail")
                    }
                    for c in row.get("remaining_confounds", [])
                ],
                "schedule_confound_keys": schedule.get("confound_keys"),
                "training_seed_axis": seed.get("status"),
                "runtime_identity_differences": identity_diff,
                "printed_metrics": {
                    "lhs": lhs_entry["formal"][0]["metrics"],
                    "rhs": rhs_entry["formal"][0]["metrics"],
                },
                "baseline_row": None,
                "status": (
                    "historical only: not a prepared pair; the numbers may be quoted side by side "
                    "with the confounds attached, never as a training effect"
                ),
            }
        )
    return out


def load_campaign(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != CAMPAIGN_SCHEMA:
        raise ContractError(
            f"{path}: schema {payload.get('schema')!r} is not {CAMPAIGN_SCHEMA}"
        )
    return payload


def _check_campaign_input(
    contract: Mapping[str, Any], campaign: Mapping[str, Any]
) -> None:
    if campaign.get("contract_sha256") != contract["frozen"]["contract_sha256"]:
        raise ContractError(
            "campaign inventory was taken under a different contract sha256"
        )
    if not campaign.get("complete"):
        raise ContractError(
            f"campaign inventory is incomplete: {campaign.get('reasons')}"
        )
    if not isinstance(campaign.get("commit"), str):
        raise ContractError("campaign inventory spans more than one commit")
    prepared = set(contract["declared"]["prepared_comparisons"])
    if set(contract["pairs"]) != prepared:
        raise ContractError("contract pairs differ from declared.prepared_comparisons")
    if set(campaign.get("pairs", {})) != prepared:
        raise ContractError(
            f"campaign pairs {sorted(set(campaign.get('pairs', {})) ^ prepared)} differ from the prepared pairs"
        )
    if set(campaign.get("recipes", {})) != set(contract["recipes"]):
        raise ContractError("campaign recipes differ from the contract recipes")


def baseline_report(
    contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    campaign_path: Path | None = None,
) -> dict[str, Any]:
    """Deliverable 4: the prepared pairs through ``validate-pair`` on the campaign's
    formal runs, baseline rows from ``paired`` verdicts only, every delta read
    against print precision and both sides' runtime-repeat ranges, confounds
    attached, historical readings separated.  The campaign inventory is the only
    input naming runs; the contract is the only input naming pairs."""
    _check_campaign_input(contract, campaign)
    matrix_rows = _row_by_id(matrix)
    pairs = {
        cid: baseline_pair(contract, campaign, matrix_rows, cid)
        for cid in contract["declared"]["prepared_comparisons"]
    }
    rows = [cid for cid, p in pairs.items() if p["baseline_row"]]
    return {
        "schema": BASELINES_SCHEMA,
        "generated_at": _now(),
        "issue": contract.get("issue"),
        "deliverable": 4,
        "contract_sha256": contract["frozen"]["contract_sha256"],
        "campaign_path": str(campaign_path) if campaign_path else None,
        "campaign_sha256": sha256_json(campaign),
        "campaign_generated_at": campaign.get("generated_at"),
        "matrix_sha256": contract["frozen"]["matrix_sha256"],
        "commit": campaign["commit"],
        "run_root": campaign.get("run_root"),
        "n_pairs": len(pairs),
        "n_paired": len(rows),
        "baseline_rows": rows,
        "complete": len(rows) == len(pairs),
        "delta_direction": "rhs_minus_lhs (treatment minus baseline; the baseline side is the contract pair's lhs)",
        "pairs": pairs,
        "readings": group_readings(pairs),
        "historical": historical_comparisons(contract, campaign, matrix_rows),
        "claim_rules": {
            "observed_difference": (
                "a delta between two paired formal runs, read against the print precision and the larger of "
                "the two sides' same-identity runtime-repeat observed ranges; 'above_runtime_repeat_range' means "
                "the instrument resolved it in this campaign"
            ),
            "observed_range_zero": (
                "an observed range of 0.0 means this batch of repeat runs saw no variation (k = 1 distinct output "
                "in n runs bounds only per-run divergence rates above ~1-0.05^(1/n)); it is not a bound and not a "
                "determinism claim"
            ),
            "effect_claim": (
                "not issued by this report: a delta above the runtime-repeat range is not by itself a causal "
                "effect; every row keeps its remaining_confounds; the C group is additionally read against the "
                "training-seed reference (D group) for attributability, and even where it exceeds that reference "
                "the statement is 'consistent across three seed-paired replicates', not an effect size"
            ),
            "baseline_identity": (
                "baseline status is the contract pair's lhs by orientation; no row was chosen by ranking a metric"
            ),
            "historical": (
                "plain-GT2 <-> T3->T1 in either family is not a prepared pair (warmup_epochs 5->3 outside the "
                "curriculum; s also differs in seed); quoted only under section 'historical'"
            ),
        },
        "variance_axes": contract["declared"]["variance_axes"],
    }


def audit_baseline_report(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[str]:
    """Is a committed report still the report of this contract + campaign?

    Pure over the three JSON inputs (no run dirs, so it runs in CI): contract
    and campaign shas, the pair set, every row's confounds, recipes, run dirs
    and metrics against the campaign inventory, every delta recomputed, and a
    row present only under a ``paired`` verdict.  Any drift is a reason.
    """
    reasons: list[str] = []
    if report.get("schema") != BASELINES_SCHEMA:
        reasons.append(f"schema {report.get('schema')!r}")
    if report.get("contract_sha256") != contract["frozen"]["contract_sha256"]:
        reasons.append("report bound to a different contract sha256")
    if report.get("campaign_sha256") != sha256_json(campaign):
        reasons.append("report bound to a different campaign inventory")
    if report.get("commit") != campaign.get("commit"):
        reasons.append("report commit != campaign commit")
    try:
        _check_campaign_input(contract, campaign)
    except ContractError as exc:
        reasons.append(str(exc))
    matrix_rows = _row_by_id(matrix)
    prepared = list(contract["declared"]["prepared_comparisons"])
    if list(report.get("pairs", {})) != prepared:
        reasons.append("report pairs != declared.prepared_comparisons")
    for cid in prepared:
        entry = (report.get("pairs") or {}).get(cid)
        if not entry:
            continue
        pair = contract["pairs"][cid]
        try:
            expected = _pair_confounds(contract, matrix_rows, cid)
        except ContractError as exc:
            reasons.append(str(exc))
            continue
        if [c["name"] for c in entry.get("remaining_confounds", [])] != [
            c["name"] for c in expected
        ]:
            reasons.append(f"{cid}: remaining_confounds differ from the contract")
        for key in (
            "lhs_recipe",
            "rhs_recipe",
            "classification",
            "variance_axis",
            "design",
        ):
            if entry.get(key) != pair[key]:
                reasons.append(f"{cid}: {key} differs from the contract")
        if entry.get("allowed_differences") != list(pair["allowed_differences"]):
            reasons.append(f"{cid}: allowed_differences differ from the contract")
        row = entry.get("baseline_row")
        if entry.get("verdict") != "paired":
            if row is not None:
                reasons.append(f"{cid}: baseline row without a paired verdict")
            continue
        if row is None:
            reasons.append(f"{cid}: paired verdict without a baseline row")
            continue
        if entry.get("n_combinations", 0) < 1 or entry.get("n_paired") != entry.get(
            "n_combinations"
        ):
            reasons.append(f"{cid}: paired verdict but not every combination is paired")
        if any(c["verdict"] != "paired" for c in entry.get("combinations", [])):
            reasons.append(f"{cid}: a combination is not_paired")
        if (
            row.get("baseline_side") != "lhs"
            or row.get("baseline_recipe") != pair["lhs_recipe"]
        ):
            reasons.append(f"{cid}: baseline side is not the contract lhs")
        for side, recipe in (("lhs", pair["lhs_recipe"]), ("rhs", pair["rhs_recipe"])):
            inv_recipe = campaign["recipes"].get(recipe) or {}
            formal = {f["run_dir"]: f for f in inv_recipe.get("formal", [])}
            run = row.get(f"{side}_run")
            if run not in formal:
                reasons.append(
                    f"{cid}: {side} run is not a campaign formal run of {recipe}"
                )
                continue
            if row.get(f"{side}_identity_sha256") != inv_recipe.get("identity_sha256"):
                reasons.append(f"{cid}: {side} identity != campaign inventory")
            rng = (inv_recipe.get("repeat") or {}).get("metric_observed_range") or {}
            for key, m in (row.get("metrics") or {}).items():
                if formal[run]["metrics"].get(key) != m.get(side):
                    reasons.append(f"{cid}: {side} {key} != campaign formal metric")
                if m.get(f"{side}_observed_range") != rng.get(key, {}).get(
                    "observed_range"
                ):
                    reasons.append(
                        f"{cid}: {side} {key} observed range != campaign repeat report"
                    )
        if set(REQUIRED_METRIC_KEYS) - set(row.get("metrics") or {}):
            reasons.append(f"{cid}: required metric keys missing")
        for key, m in (row.get("metrics") or {}).items():
            if not all(
                k in m for k in ("lhs", "rhs", "delta_rhs_minus_lhs", "reading")
            ):
                reasons.append(f"{cid}: {key} row is not a delta reading")
                continue
            expected_m = delta_reading(
                float(m["lhs"]),
                float(m["rhs"]),
                precision=metric_print_precision(key),
                lhs_observed_range=m.get("lhs_observed_range"),
                rhs_observed_range=m.get("rhs_observed_range"),
            )
            if expected_m != m:
                reasons.append(f"{cid}: {key} delta reading does not recompute")
    for hist in report.get("historical", []):
        if hist.get("comparison_id") in contract["pairs"]:
            reasons.append(
                f"{hist.get('comparison_id')}: historical entry is a prepared pair"
            )
        if (
            hist.get("baseline_row") is not None
            or (hist.get("validate_pair") or {}).get("verdict") != "not_paired"
        ):
            reasons.append(
                f"{hist.get('comparison_id')}: historical entry carries a row or a paired verdict"
            )
        if not hist.get("training_confounds"):
            reasons.append(
                f"{hist.get('comparison_id')}: historical entry lost its confounds"
            )
    if set(h.get("comparison_id") for h in report.get("historical", [])) != set(
        HISTORICAL_COMPARISONS
    ):
        reasons.append("historical set differs from HISTORICAL_COMPARISONS")
    if report.get("readings") != group_readings(report.get("pairs") or {}):
        reasons.append("group readings do not recompute from the rows")
    text = json.dumps(report).lower()
    for word in ("deterministic", "bit-exact"):
        if word in text.replace(f"not a {word}", "").replace(
            f"never '{word}'", ""
        ).replace(f"not {word}", ""):
            reasons.append(f"report uses the word {word!r}")
    return reasons


def _fmt_delta(m: Mapping[str, Any], digits: int = 1) -> str:
    d = m["delta_rhs_minus_lhs"]
    return f"{d:+.{digits}f}"


_READING_SHORT = {
    "no_observed_difference": "=",
    "within_runtime_repeat_range": "≤range",
    "above_runtime_repeat_range": ">range",
}


def render_baselines_markdown(report: Mapping[str, Any]) -> str:
    """Human view of ``baseline_report``; regenerate, never edit."""
    date = report["generated_at"][:10]
    pairs = report["pairs"]
    out: list[str] = [
        "<!-- doc-status: active -->",
        "<!-- doc-promotion: report_data -->",
        f"<!-- doc-date: {date} -->",
        "<!-- doc-module: detection -->",
        f"<!-- Generated by scripts/provenance/training_eval_contract.py baselines from {DEFAULT_BASELINES_OUT}; regenerate rather than edit. -->",
        "",
        "# Training eval baselines — pairwise comparison (#421 · deliverable 4)",
        "",
        f"Generated {report['generated_at']} from the committed campaign inventory `{report['campaign_path']}` "
        f"(sha256 `{report['campaign_sha256'][:16]}…`, taken {report['campaign_generated_at']}) under contract sha256 "
        f"`{report['contract_sha256'][:16]}…`; every run at commit `{report['commit'][:12]}`. "
        f"Machine-readable: `{DEFAULT_BASELINES_OUT}`.",
        "",
        "Every number here is an **observed difference** between two formal runs; this document issues no effect claim. "
        "Read the claim rules (§2) before any row.",
        "",
        "## 1. Status",
        "",
        f"- Prepared pairs: {report['n_pairs']}; `validate-pair` verdict `paired` on every lhs × rhs formal combination: "
        f"**{report['n_paired']}/{report['n_pairs']}**; baseline rows formed: {report['n_paired']} "
        f"({'complete' if report['complete'] else 'INCOMPLETE'}).",
    ]
    for cid, p in pairs.items():
        for reason in p["reasons"]:
            out.append(f"- NOT PAIRED `{cid}`: {reason}")
    out += [
        f"- Delta direction: {report['delta_direction']}.",
        "- Baseline identity: the contract pair's lhs by orientation (§4 column *baseline*); no row was chosen or re-chosen by ranking a metric.",
        f"- Historical (non-prepared) readings: {len(report['historical'])} (§6), never rows.",
        "",
        "## 2. Claim rules",
        "",
    ]
    for key, rule in report["claim_rules"].items():
        out.append(f"- **{key}**: {rule}")
    out += [
        f"- **variance axes** (contract): runtime_repeat = {report['variance_axes']['runtime_repeat']}; "
        f"training_seed = {report['variance_axes']['training_seed']}; rule: {report['variance_axes']['rule']}",
        "",
        "## 3. Reading legend",
        "",
        "`=` no observed difference at print precision (0.1 for percent metrics, 1 for counts, 0.01 fps); "
        "`≤range` |Δ| within the larger of the two sides' runtime-repeat observed ranges; "
        "`>range` |Δ| above both ranges — resolved by the instrument in this campaign, not an effect. "
        "`n/k` = repeat runs / max distinct outputs per side.",
        "",
        "## 4. Baseline rows (paired verdicts only)",
        "",
        "| pair | class | axis | baseline (lhs) | treatment (rhs) | n/k lhs · rhs | HOTA lhs→rhs (Δ) | IDF1 lhs→rhs (Δ) | MOTA lhs→rhs (Δ) | IDs (Δ) | fps lhs→rhs (Δ) | confounds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, p in pairs.items():
        row = p["baseline_row"]
        conf = ", ".join(c["name"] for c in p["remaining_confounds"]) or "—"
        if not row:
            out.append(
                f"| `{cid}` | {p['classification']} | {p['variance_axis']} | `{p['lhs_recipe']}` | `{p['rhs_recipe']}` | — | not_paired | | | | | {conf} |"
            )
            continue
        m = row["metrics"]
        fps = row["throughput_fps"]

        def cell(key: str, digits: int = 1) -> str:
            e = m[key]
            return f"{e['lhs']:.{digits}f}→{e['rhs']:.{digits}f} ({_fmt_delta(e, digits)} {_READING_SHORT[e['reading']]})"

        out.append(
            f"| `{cid}` | {p['classification']} | {p['variance_axis']} | `{row['baseline_recipe']}` | `{row['treatment_recipe']}` | "
            f"{row['repeat_n']['lhs']}/{row['repeat_distinct_outputs_max']['lhs']} · {row['repeat_n']['rhs']}/{row['repeat_distinct_outputs_max']['rhs']} | "
            f"{cell('HOTA')} | {cell('IDF1')} | {cell('MOTA')} | {int(m['IDs']['lhs'])}→{int(m['IDs']['rhs'])} ({_fmt_delta(m['IDs'], 0)} {_READING_SHORT[m['IDs']['reading']]}) | "
            f"{fps['lhs']:.1f}→{fps['rhs']:.1f} ({_fmt_delta(fps, 1)} {_READING_SHORT[fps['reading']]}) | {conf} |"
        )
    out += [
        "",
        "## 5. Per-pair detail",
        "",
    ]
    for cid, p in pairs.items():
        out += [
            f"### `{cid}`",
            "",
            f"- design `{p['design']}` · classification `{p['classification']}` · variance axis `{p['variance_axis']}` · "
            f"treatment axes {json.dumps(p['treatment_axes'])}",
            f"- intended treatment: {p['intended_treatment'] or '—'}",
            f"- verdict **{p['verdict']}** ({p['n_paired']}/{p['n_combinations']} formal combinations paired)"
            + (f"; reasons: {p['reasons']}" if p["reasons"] else ""),
            f"- allowed differences: {json.dumps(p['allowed_differences'])}",
        ]
        if p["remaining_confounds"]:
            out.append(
                "- remaining confounds (attached to the row regardless of the numbers):"
            )
            for c in p["remaining_confounds"]:
                out.append(f"  - `{c['name']}` [{c['severity']}]: {c['detail']}")
        else:
            out.append(
                "- remaining confounds: none recorded by the matrix for this pair"
            )
        row = p["baseline_row"]
        if not row:
            out.append("- no baseline row")
            out.append("")
            continue
        out += [
            f"- runs: lhs `{row['lhs_run']}` (identity `{row['lhs_identity_sha256'][:12]}`), "
            f"rhs `{row['rhs_run']}` (identity `{row['rhs_identity_sha256'][:12]}`)",
            f"- observed identity differences: {json.dumps(sorted(row['observed_differences']))}",
            "",
            "| metric | lhs | rhs | Δ (rhs−lhs) | precision | range lhs | range rhs | reading |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for key, e in row["metrics"].items():
            digits = 0 if key in COUNT_METRIC_KEYS else 1
            out.append(
                f"| {key} | {e['lhs']:.{digits}f} | {e['rhs']:.{digits}f} | {_fmt_delta(e, digits)} | {e['print_precision']:g} | "
                f"{_fmt(e['lhs_observed_range'])} | {_fmt(e['rhs_observed_range'])} | {e['reading']} |"
            )
        fps = row["throughput_fps"]
        out.append(
            f"| fps ({fps['profile']}) | {fps['lhs']:.2f} | {fps['rhs']:.2f} | {_fmt_delta(fps, 2)} | {fps['print_precision']:g} | "
            f"{_fmt(fps['lhs_observed_range'])} | {_fmt(fps['rhs_observed_range'])} | {fps['reading']} |"
        )
        out += [
            "",
            f"fps repeat ranges: lhs {fps['lhs_repeat']['min']}–{fps['lhs_repeat']['max']} (n={len(fps['lhs_repeat']['values'])}), "
            f"rhs {fps['rhs_repeat']['min']}–{fps['rhs_repeat']['max']} (n={len(fps['rhs_repeat']['values'])}); "
            f"formal fps lhs {fps['lhs_formal']['values']}, rhs {fps['rhs_formal']['values']}. "
            + (
                "Eager rows are a throughput lower bound by contract declaration."
                if fps["profile"] == "eager_pytorch"
                else "Serial whole-graph profile; not the double-buffer headline throughput."
            ),
            "",
        ]
    out += [
        "## 6. Historical readings (not prepared pairs, no rows)",
        "",
    ]
    for h in report["historical"]:
        lm, rm_ = h["printed_metrics"]["lhs"], h["printed_metrics"]["rhs"]
        out += [
            f"### `{h['comparison_id']}` — {h['classification']}",
            "",
            f"- {h['status']}",
            f"- `validate-pair`: **{h['validate_pair']['verdict']}** — {h['validate_pair']['reasons']}",
            f"- lhs `{h['lhs_recipe']}` (`{h['lhs_run']}`) · rhs `{h['rhs_recipe']}` (`{h['rhs_run']}`)",
            f"- intended treatment (matrix): {h['intended_treatment']}",
            f"- training confounds (matrix; schedule keys {json.dumps(h['schedule_confound_keys'])}; seed axis `{h['training_seed_axis']}`):",
        ]
        for c in h["training_confounds"]:
            extra = (
                f" lhs={c['lhs']} rhs={c['rhs']}"
                if c.get("lhs") is not None or c.get("rhs") is not None
                else ""
            )
            out.append(f"  - `{c['name']}` [{c['severity']}]{extra}: {c['detail']}")
        out += [
            f"- runtime identity differences between the two formal runs: {json.dumps(sorted(h['runtime_identity_differences']))}",
            "- printed side by side (no delta reading is issued): "
            + "; ".join(
                f"{k} {_fmt(lm.get(k))} | {_fmt(rm_.get(k))}"
                for k in HEADLINE_METRIC_KEYS
            )
            + f"; IDs {_fmt_count(lm.get('IDs'))} | {_fmt_count(rm_.get('IDs'))}",
            "",
        ]
    out += [
        "## 7. What the rows support / do not support",
        "",
    ]
    out += _render_readings(report)
    out += [
        "",
        "## 8. Provenance",
        "",
        f"- contract sha256 `{report['contract_sha256']}` · matrix sha256 `{report['matrix_sha256']}` · campaign sha256 `{report['campaign_sha256']}`",
        f"- run root `{report['run_root']}` (gitignored raw runs: manifests v3, stdout, MOT files, latency profiles); the campaign inventory + repeat reports are the surviving record",
        "- regenerate: `.venv/bin/python scripts/provenance/training_eval_contract.py baselines --emit "
        f"{DEFAULT_BASELINES_OUT} --baselines-md {DEFAULT_BASELINES_MD}`; audit without runs: `... baselines --audit`",
        "",
    ]
    return "\n".join(out)


def _render_readings(report: Mapping[str, Any]) -> list[str]:
    readings = report["readings"]
    out: list[str] = []

    def metric_line(group: str, key: str) -> str:
        m = readings[group]["per_metric"][key]
        digits = 0 if key in COUNT_METRIC_KEYS else 1
        deltas = ", ".join(
            f"`{cid.split('.')[0]}` {d:+.{digits}f}" for cid, d in m["deltas"].items()
        )
        return (
            f"  - {key}: {deltas}; sign consistent: {'yes' if m['sign_consistent'] else 'no'}; "
            f"all `>range`: {'yes' if m['all_above_runtime_repeat_range'] else 'no'}"
            + (
                f"; min |Δ| {m['min_abs_delta']:.{digits}f} vs seed-replicate max |Δ| {m['seed_replicate_max_abs_delta']:.{digits}f} → "
                f"{'exceeds' if m['min_abs_delta_exceeds_seed_replicate_max'] else 'does not exceed'}"
                if "seed_replicate_max_abs_delta" in m
                else ""
            )
        )

    g = readings["explicit_vs_implicit_shared_gt1"]
    out += [f"### C1–C3 · {g['question']}", ""]
    if not g["complete"]:
        out += [
            f"- rows available: {g['rows_available']} — group reading unavailable",
            "",
        ]
    else:
        for key in ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDs"):
            out.append(metric_line("explicit_vs_implicit_shared_gt1", key))
        strong = [
            key
            for key in ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDs")
            if g["per_metric"][key]["sign_consistent"]
            and g["per_metric"][key]["all_above_runtime_repeat_range"]
            and g["per_metric"][key].get("min_abs_delta_exceeds_seed_replicate_max")
        ]
        weak = [
            key
            for key in ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDs")
            if key not in strong
        ]
        out += [
            "",
            "- **Supports (observed):** "
            + (
                f"on {', '.join(strong)} the explicit T3→T1 arm differs from the implicit arm with one sign in all three "
                "seed-paired replicates, above the runtime-repeat range of every row, and the smallest |Δ| exceeds the largest "
                "|Δ| observed between seed replicates of either arm (D1–D4). This is the contract's attributability reference "
                "being met: a consistent observed difference across three seeds, not an effect size (3 seeds) and not a mechanism."
                if strong
                else "no metric meets sign consistency + `>range` + exceeds-seed-reference on all three replicates."
            ),
            "- **Does not support:** "
            + (
                f"any claim on {', '.join(weak)} (sign or magnitude does not clear the seed-replicate reference); "
                if weak
                else ""
            )
            + "a causal attribution to staging alone (the treatment is the declared bundle: 15 ep T=3 with temporal blocks then 15 ep T=1 vs 30 ep implicit T=4); "
            "a magnitude estimate; transfer off the preset s backbone engine (`deployed_backbone_teacher_mismatch` is common-mode on every s row, see E4); "
            "the eager or m families.",
            "",
        ]
    g = readings["seed_replicates"]
    out += [f"### D1–D4 · {g['question']}", ""]
    if g["complete"]:
        for key in ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDs"):
            out.append(metric_line("seed_replicates", key))
        out += [
            "",
            "- **Supports (observed):** the spread between seed replicates of one recipe is what these four deltas show; it is the "
            "reference the C group is read against (`declared.variance_axes.rule`). Each is above its runtime-repeat range where marked, "
            "so seed-to-seed variation is resolvable by the instrument.",
            "- **Does not support:** a variance estimate (two deltas per arm), pooling with runtime-repeat variance, or any ordering of seeds.",
            "",
        ]
    else:
        out += [
            f"- rows available: {g['rows_available']} — group reading unavailable",
            "",
        ]
    g = readings["stage_increments"]
    out += [f"### B1–B4 · {g['question']}", ""]
    if g["complete"]:
        for key in ("HOTA", "IDF1", "MOTA", "IDs"):
            out.append(metric_line("stage_increments", key))
        out += [
            "",
            "- **Supports (observed):** each stage bundle changes the printed numbers by more than the runtime-repeat range where marked `>range`; "
            "the direction per stage is as listed.",
            "- **Does not support:** attributing a delta to any single ingredient of the bundle (warm start, budget, schedule and cache move together), "
            "a seed-controlled statement (one checkpoint per stage, no replicate), or a cross-family comparison (s rows carry the common-mode backbone confound; m rows run the checkpoint head via `--no-mamba-trt`).",
            "",
        ]
    else:
        out += [
            f"- rows available: {g['rows_available']} — group reading unavailable",
            "",
        ]
    g = readings["system_comparisons"]
    out += [f"### E1–E3 · {g['question']}", ""]
    if g["complete"]:
        for key in ("HOTA", "IDF1", "MOTA", "IDs"):
            out.append(metric_line("system_comparisons", key))
        out += [
            "",
            "- **Supports (observed):** the two systems / heads differ by the printed amounts under the frozen recipes; E1 is the executable "
            "s-vs-m production comparison, E2/E3 the native-Detect-head vs Mamba-head numbers on the shared PyTorch backbone under the "
            "s tracker policy (a declared limit for E3).",
            "- **Does not support:** attribution to any axis — `remaining_confounds` list every unmatched training and runtime axis "
            "(head family, deployed head artifact, tracker policy, warm start, teacher, cache, seed, schedule, budget); E2/E3 fps are eager lower bounds.",
            "",
        ]
    else:
        out += [
            f"- rows available: {g['rows_available']} — group reading unavailable",
            "",
        ]
    g = readings["engine_ab"]
    out += [f"### E4 · {g['question']}", ""]
    if g["complete"]:
        for key in ("HOTA", "IDF1", "MOTA", "DetA", "AssA", "IDs"):
            out.append(metric_line("engine_ab", key))
        e4_signs = {
            _sign(next(iter(m["deltas"].values())))
            for key, m in g["per_metric"].items()
            if key in ("HOTA", "IDF1", "MOTA")
        }
        out += [
            "",
            "- **Supports (observed):** swapping the deployed s backbone engine under the same head moves the numbers by more than the "
            "runtime-repeat range where marked, "
            + (
                "with mixed sign across HOTA/IDF1/MOTA"
                if len(e4_signs) > 1
                else "with one sign across HOTA/IDF1/MOTA"
            )
            + ": the s production confound is numerically live, so every s-row absolute number is specific to the preset engine.",
            "- **Does not support:** which engine is 'correct' or better (one head, no seed replicate, no per-sequence reading), or that s-internal "
            "paired deltas would change under the other engine (common-mode by design, untested).",
            "",
        ]
    else:
        out += [
            f"- rows available: {g['rows_available']} — group reading unavailable",
            "",
        ]
    out += [
        "### Not available from this deliverable",
        "",
        "- Any plain-GT2 ↔ T3→T1 training effect (s or m): §6 only.",
        "- Any m-family curriculum statement: m has no controlled curriculum pair (matrix).",
        "- Any determinism / bit-exactness statement: repeat evidence is `k distinct outputs in n runs`.",
        "- Any headline (double-buffer) throughput: all fps here are the serial whole-graph or eager profiles.",
    ]
    return out


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

    s_camp = sub.add_parser(
        "campaign",
        help="inventory every recipe's repeat report + formal runs under the run root",
    )
    s_camp.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    s_camp.add_argument("--emit", type=Path, default=None)
    s_camp.add_argument("--campaign-md", type=Path, default=None)

    s_base = sub.add_parser(
        "baselines",
        help="deliverable 4: validate-pair over the campaign inventory's formal runs, baseline rows + delta readings",
    )
    s_base.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN_OUT)
    s_base.add_argument("--emit", type=Path, default=None)
    s_base.add_argument("--baselines-md", type=Path, default=None)
    s_base.add_argument(
        "--audit",
        action="store_true",
        help="exit 1 unless the committed report + doc still follow from contract, campaign and matrix (no runs needed)",
    )

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
        if args.stage == "repeat":
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
    if args.cmd == "campaign":
        inventory = campaign_inventory(contract, Path(args.root))
        print(render_campaign_table(inventory))
        for reason in inventory["reasons"]:
            print(f"INCOMPLETE: {reason}")
        for recipe_id, r in inventory["recipes"].items():
            for reason in r["reasons"]:
                print(f"  {recipe_id}: {reason}")
        if args.emit:
            _write(REPO_ROOT / args.emit, json.dumps(inventory, indent=2) + "\n")
            print(f"campaign inventory -> {args.emit}")
        if args.campaign_md:
            _write(REPO_ROOT / args.campaign_md, render_campaign_markdown(inventory))
            print(f"campaign doc -> {args.campaign_md}")
        return 0 if inventory["complete"] else 1
    if args.cmd == "baselines":
        campaign = load_campaign(REPO_ROOT / args.campaign)
        matrix = load_matrix(REPO_ROOT / args.matrix)
        if args.audit:
            emit = REPO_ROOT / (args.emit or DEFAULT_BASELINES_OUT)
            md = REPO_ROOT / (args.baselines_md or DEFAULT_BASELINES_MD)
            report = _read_json(emit)
            reasons = audit_baseline_report(report, contract, campaign, matrix)
            if not md.is_file() or md.read_text() != render_baselines_markdown(report):
                reasons.append(f"{md} is not the rendering of {emit}")
            for reason in reasons:
                print(f"STALE: {reason}")
            print(
                f"training_eval_baselines: {'fresh' if not reasons else 'stale'} "
                f"({report.get('n_paired')}/{report.get('n_pairs')} paired)"
            )
            return 0 if not reasons else 1
        report = baseline_report(
            contract, campaign, matrix, campaign_path=args.campaign
        )
        for cid, p in report["pairs"].items():
            row = p["baseline_row"]
            head = (
                " ".join(
                    f"{k} {row['metrics'][k]['lhs']:.1f}->{row['metrics'][k]['rhs']:.1f}({row['metrics'][k]['delta_rhs_minus_lhs']:+.1f})"
                    for k in HEADLINE_METRIC_KEYS
                )
                if row
                else "no row"
            )
            print(
                f"{p['verdict']:<10} {cid:<48} {p['n_paired']}/{p['n_combinations']} {head}"
            )
            for reason in p["reasons"]:
                print(f"  {reason}")
        if args.emit:
            _write(
                REPO_ROOT / args.emit,
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            )
            print(f"baselines report -> {args.emit}")
        if args.baselines_md:
            _write(REPO_ROOT / args.baselines_md, render_baselines_markdown(report))
            print(f"baselines doc -> {args.baselines_md}")
        return 0 if report["complete"] else 1
    raise ContractError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
