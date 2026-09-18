"""Comparison-identifiability matrix over the #421 training lineage (deliverable 2).

Deliverable 1 (``training_lineage.py``) froze *what exists*: every checkpoint,
teacher, cache, engine and preset as a node with sha256, derived parent edges and
byte-level checks.  This tool answers the next question on top of that snapshot:
**which pairs of nodes can a measurement be attributed to, and which cannot** —
before any IDF1/HOTA/FPS number is produced.

Inputs, again kept apart on purpose:

* ``report_data/training_lineage_inventory.json`` — the committed inventory.  It
  is the only source of node facts; nothing is re-read from ``runs/``.
* ``training_comparisons.json`` — the **declared candidate comparisons**: which
  two nodes, which question, which axes the declared treatment is *allowed* to
  change, and which design the pair claims to follow.  This is a claim.
* the two family presets (committed) for the tracker-policy diff, and
  ``report_data/tables/mamba_tracking_overall.csv`` for which endpoints already
  have historical results on record.

For each declared comparison the tool derives one **profile** per endpoint
(backbone family, head architecture, warm-start chain, teacher, cache, seed,
per-stage schedule, budget, resolved sequence split, and — under the declared
runtime binding — the deployed backbone / head artifact, forward mode and
tracker policy), compares the two profiles axis by axis, and classifies:

* ``controlled`` — every axis matched except those the declaration names as the
  treatment, no axis unknown, no blocking confound.  A fresh paired eval of the
  two artifacts attributes its delta to the declared treatment.
* ``system_comparison`` — a structural axis (family, head, deployment artifact,
  forward mode, tracker policy, data) differs outside the treatment.  A fresh
  paired eval under one contract is a whole-system quality/cost comparison,
  never a post-training attribution.
* ``historical_not_comparable`` — only *training* axes differ outside the
  treatment (seed, warm start, warm-up, teacher, cache, budget) or one of them
  cannot be established from the bytes.  These differences live inside the
  artifacts, so no fresh eval repairs them; any recorded delta over this pair is
  historical and must not be merged into controlled rows.
* ``blocked`` — an endpoint or its declared runtime artifact is unavailable,
  the declared design's premise is contradicted by the bytes (e.g. a "sibling"
  pair where one side is a continuation of the other), or a blocking confound
  applies.  The entry names the blocker.

The classification is **derived, never declared**: the declaration may say what
the treatment is and what the design claims, but the tool decides the class
from the axis statuses, and the contract tests pin the rule that an unmatched
or unknown non-treatment axis can never be labelled ``controlled``.

Two facts are carried through every row rather than decided per row:

* the s production backbone engine's sibling ONNX matches the *legacy* teacher
  while every replica-lineage s head was trained against the adapted teacher
  (inventory: ``DIFFERENT_TEACHER_INDICATED``, engine bytes unattributed).  It
  is recorded as ``common_mode`` when both endpoints deploy with the same status
  (it cannot explain a paired delta) and ``blocking`` when the two endpoints
  differ in that status;
* no historical result row carries a preset or engine sha (the table has no such
  column and, in this workspace, no referenced ``results/`` directory has a
  ``run_manifest.json``), so historical results are never marked reusable as a
  paired measurement.  Runtime identity for new measurements is deliverable 3.

The tool does not train, evaluate, or infer anything the inventory does not
state; a node the inventory marks unavailable stays unavailable here.

Usage:
    .venv/bin/python scripts/provenance/training_comparison.py             # writes the defaults
    .venv/bin/python scripts/provenance/training_comparison.py --check     # exit 1 if outputs are stale
"""

# status: stable

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
DEFAULT_DECLARATIONS = _HERE.parent / "training_comparisons.json"
DEFAULT_INVENTORY = Path("report_data/training_lineage_inventory.json")
DEFAULT_RESULTS_TABLE = Path("report_data/tables/mamba_tracking_overall.csv")
DEFAULT_JSON_OUT = Path("report_data/training_comparison_matrix.json")
DEFAULT_MD_OUT = Path("docs/research/training/training_comparison_matrix.md")

SCHEMA = "training_comparison_matrix_v1"
DECLARATIONS_SCHEMA = "training_comparisons_v1"
INVENTORY_SCHEMA = "training_lineage_inventory_v1"

CLASSES = ("controlled", "system_comparison", "historical_not_comparable", "blocked")

# Axes.  Structural axes describe *what system* is measured; training axes
# describe *how the head got its weights*.  A structural difference outside the
# treatment makes the pair a system comparison; a training difference outside
# the treatment makes it historical.
STRUCTURAL_AXES: tuple[str, ...] = (
    "backbone_family",
    "head_family",
    "dataset_split",
    "inference_forward_mode",
    "deployed_backbone_artifact",
    "deployed_head_artifact",
    "tracker_runtime_policy",
)
TRAINING_AXES: tuple[str, ...] = (
    "warm_start",
    "teacher",
    "teacher_cache",
    "training_seed",
    "training_schedule",
    "training_budget",
)
AXES: tuple[str, ...] = STRUCTURAL_AXES + TRAINING_AXES

# Which axes a design may declare as its treatment.  Structural axes are never
# a *training* treatment; ``runtime_ab`` is the one design whose treatment is a
# deployment artifact, and it is not a training attribution.
DESIGNS: dict[str, frozenset[str]] = {
    "paired_siblings": frozenset({"training_schedule", "teacher_cache", "teacher"}),
    "stage_increment": frozenset(
        {"warm_start", "training_budget", "training_schedule", "teacher_cache"}
    ),
    "seed_replicate": frozenset({"training_seed"}),
    "runtime_ab": frozenset({"deployed_backbone_artifact"}),
    "system": frozenset(),
}

AXIS_STATUSES = ("matched", "matched_effective", "unmatched", "unknown")

# ``args`` keys that make up a stage's recipe.  ``epochs`` is budget, ``seed`` is
# its own axis, ``seqs``/``holdout_seqs`` are the split; everything path-like is
# bookkeeping (same exclusion set as the inventory's treatment delta).
SCHEDULE_KEYS: tuple[str, ...] = (
    "lr",
    "lr_gate",
    "lr_yolo",
    "batch_size",
    "accum_steps",
    "clip_len",
    "clip_stride",
    "gt_ratio",
    "add_temporal",
    "scan_stop_grad",
    "warmup_epochs",
    "clip_grad",
    "cls_weight",
    "img_size",
    "freeze_temporal",
    "freeze_spatial",
    "use_temporal_attention",
    "consistency_weight",
    "t1_weight",
    "best_by",
)

# Mamba-head architecture flags (``mamba_args``) with the value an older
# checkpoint implies by not recording the key.  ``use_temporal_mamba`` is kept
# out: temporal blocks are reported separately because the deployment bypasses
# them at T=1 (artifact differs, effective forward does not).
HEAD_ARCH_DEFAULTS: dict[str, Any] = {
    "d_model": None,
    "d_state": None,
    "num_blocks": None,
    "spatial_reduction": None,
    "num_classes": None,
    "use_pixel_shuffle": False,
    "use_cross_scan": False,
    "use_hybrid_head": False,
    "use_temporal_attention": False,
    "per_channel_a": False,
    "use_detail_fusion": False,
    "detail_source": "none",
}

# ``--seqs ''`` resolves, in ``train_mamba_gt.py`` / ``train_mamba_head.py``, to
# every ``*-SDP`` directory under ``datasets/MOT17/train`` (dataset.py default
# ``detector="SDP"``).  On the capture host that is exactly these seven.
DEFAULT_SDP_SEQUENCES: tuple[str, ...] = (
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
)

# Preset keys that name model artifacts; everything else is tracker / runtime policy.
PRESET_ARTIFACT_KEYS = frozenset(
    {
        "mamba_ckpt",
        "mamba_teacher_ckpt",
        "mamba_yolo_weights",
        "fpn_backbone_engine",
        "mamba_head_engine",
        "engine",
    }
)


class ComparisonError(RuntimeError):
    """The declarations are malformed or name something the inventory does not have."""


# --------------------------------------------------------------------------- inputs


def load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != INVENTORY_SCHEMA:
        raise ComparisonError(
            f"{path}: schema must be {INVENTORY_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    if not isinstance(payload.get("nodes"), dict):
        raise ComparisonError(f"{path}: 'nodes' must be an object")
    return payload


def load_declarations(path: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != DECLARATIONS_SCHEMA:
        raise ComparisonError(
            f"{path}: schema must be {DECLARATIONS_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ComparisonError(f"{path}: 'comparisons' must be a non-empty list")
    nodes = inventory["nodes"]
    seen: set[str] = set()
    for spec in comparisons:
        cid = spec.get("comparison_id")
        if not isinstance(cid, str) or not cid:
            raise ComparisonError(f"{path}: comparison without 'comparison_id'")
        if cid in seen:
            raise ComparisonError(f"{path}: duplicate comparison_id {cid!r}")
        seen.add(cid)
        design = spec.get("design")
        if design not in DESIGNS:
            raise ComparisonError(f"{path}: {cid}: unknown design {design!r}")
        for side in ("lhs", "rhs"):
            endpoint = spec.get(side)
            if not isinstance(endpoint, dict) or "node" not in endpoint:
                raise ComparisonError(f"{path}: {cid}: {side} must be {{'node': ...}}")
            if endpoint["node"] not in nodes:
                raise ComparisonError(
                    f"{path}: {cid}: {side} names unknown node {endpoint['node']!r}"
                )
            runtime = endpoint.get("runtime", {"binding": "family_preset"})
            binding = runtime.get("binding")
            if binding not in ("family_preset", "shared_backbone_node"):
                raise ComparisonError(
                    f"{path}: {cid}: {side} runtime binding {binding!r} unknown"
                )
            if binding == "shared_backbone_node":
                bnode = runtime.get("backbone_node")
                if bnode not in nodes:
                    raise ComparisonError(
                        f"{path}: {cid}: {side} runtime names unknown node {bnode!r}"
                    )
            if binding == "family_preset" and runtime.get("backbone_engine_node"):
                enode = runtime["backbone_engine_node"]
                if enode not in nodes or nodes[enode]["kind"] != "trt_engine":
                    raise ComparisonError(
                        f"{path}: {cid}: {side} backbone_engine_node {enode!r} is not an engine node"
                    )
        treatment = spec.get("treatment_axes", [])
        if not isinstance(treatment, list):
            raise ComparisonError(f"{path}: {cid}: treatment_axes must be a list")
        allowed = DESIGNS[design]
        for axis in treatment:
            if axis not in AXES:
                raise ComparisonError(f"{path}: {cid}: unknown axis {axis!r}")
            if axis not in allowed:
                raise ComparisonError(
                    f"{path}: {cid}: design {design!r} may not declare {axis!r} as treatment"
                )
        if design != "system" and not treatment:
            raise ComparisonError(
                f"{path}: {cid}: design {design!r} needs treatment_axes"
            )
        if "intended_treatment" not in spec:
            raise ComparisonError(f"{path}: {cid}: lacks 'intended_treatment'")
        keys = spec.get("treatment_schedule_keys", [])
        for key in keys:
            if key not in SCHEDULE_KEYS and key != "staging":
                raise ComparisonError(
                    f"{path}: {cid}: treatment_schedule_keys names non-schedule key {key!r}"
                )
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- profiles


def _edge_target(
    node: dict[str, Any], relation: str
) -> tuple[str | None, str | None, bool]:
    for edge in node.get("edges", []):
        if edge.get("relation") == relation:
            return (
                edge.get("target_node"),
                edge.get("target_path"),
                bool(edge.get("target_exists")),
            )
    return None, None, False


def resolve_sequences(seqs: str | None, holdout: str | None) -> dict[str, Any]:
    """Map the recorded ``--seqs`` / ``--holdout-seqs`` strings to a sequence set."""
    if seqs:
        train = tuple(sorted(s.strip() for s in seqs.split(",") if s.strip()))
        basis = "explicit"
    else:
        train = DEFAULT_SDP_SEQUENCES
        basis = "default:all *-SDP under datasets/MOT17/train"
    held = tuple(sorted(s.strip() for s in (holdout or "").split(",") if s.strip()))
    train = tuple(s for s in train if s not in held)
    return {"train": list(train), "holdout": list(held), "basis": basis}


@dataclass
class Stage:
    node: str
    epochs: int | None
    selected_epoch: int | None
    seed: Any
    schedule: dict[str, Any]
    split: dict[str, Any]
    teacher: str | None
    cache: str | None
    cache_available: bool | None


@dataclass
class Profile:
    node: str
    family: str
    kind: str
    exists: bool
    backbone_family: str | None = None
    head_family: dict[str, Any] | None = None
    temporal_blocks_present: bool | None = None
    param_count: int | None = None
    warm_start: str | None = None
    warm_start_basis: str = ""
    chain: list[Stage] = field(default_factory=list)
    chain_complete: bool = True
    chain_note: str = ""
    teacher: str | None = None
    teacher_basis: str = ""
    cache_nodes: list[str] = field(default_factory=list)
    caches_available: bool | None = None
    seed: Any = "unknown"
    seed_basis: str = ""
    training_stage_split: dict[str, Any] | None = None
    external_pretraining: bool = False
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "chain"}
        out["chain"] = [s.__dict__ for s in self.chain]
        return out


def _stage_of(node_id: str, node: dict[str, Any]) -> Stage:
    args = (node.get("summary") or {}).get("args") or {}
    teacher, _, _ = _edge_target(node, "teacher")
    cache, _, cache_exists = _edge_target(node, "cache")
    return Stage(
        node=node_id,
        epochs=args.get("epochs"),
        selected_epoch=(node.get("summary") or {}).get("epoch"),
        seed=args.get("seed", "unknown"),
        schedule={k: args[k] for k in SCHEDULE_KEYS if k in args},
        split=resolve_sequences(args.get("seqs"), args.get("holdout_seqs")),
        teacher=teacher,
        cache=cache,
        cache_available=(cache_exists if cache else None),
    )


def _head_family(node: dict[str, Any]) -> tuple[dict[str, Any], bool | None]:
    margs = (node.get("summary") or {}).get("mamba_args") or {}
    sig = {k: margs.get(k, default) for k, default in HEAD_ARCH_DEFAULTS.items()}
    sig["family"] = "mamba_head"
    temporal = margs.get("use_temporal_mamba")
    if temporal is None:
        temporal = "temporal_blocks" in (node.get("summary") or {}).get(
            "module_groups", []
        ) or any(
            g.startswith("temporal_blocks")
            for g in (node.get("summary") or {}).get("module_groups", [])
        )
    return sig, bool(temporal)


def build_profile(node_id: str, inventory: dict[str, Any]) -> Profile:
    nodes = inventory["nodes"]
    node = nodes[node_id]
    fam = node["family"]
    prof = Profile(
        node=node_id, family=fam, kind=node["kind"], exists=bool(node.get("exists"))
    )
    prof.backbone_family = inventory["families"][fam]["backbone"]
    if not prof.exists:
        return prof

    if node["kind"] == "yolo_pt":
        prof.head_family = {"family": "yolo_detect", "gate_module": False}
        prof.temporal_blocks_present = False
        prof.external_pretraining = True
        prof.warm_start = None
        prof.warm_start_basis = "external pretraining (COCO); no in-repo parent"
        prof.seed = "external"
        prof.seed_basis = "external pretraining"
        prof.training_stage_split = {
            "train": ["external:COCO"],
            "holdout": [],
            "basis": "external",
        }
        prof.chain_complete = True
        return prof

    if node["kind"] == "gated_teacher_ckpt":
        summary = node.get("summary") or {}
        args = summary.get("args") or {}
        prof.head_family = {"family": "yolo_detect", "gate_module": True}
        prof.temporal_blocks_present = False
        prof.param_count = summary.get("param_count")
        base, _, _ = _edge_target(node, "base_yolo")
        prof.warm_start = base
        prof.warm_start_basis = "args.yolo_weights"
        prov = summary.get("provenance") or {}
        seqs = prov.get("training_sequences")
        split = (
            {
                "train": sorted(seqs),
                "holdout": [],
                "basis": "provenance.training_sequences",
            }
            if seqs
            else resolve_sequences(args.get("seqs"), args.get("holdout_seqs"))
        )
        stage = Stage(
            node=node_id,
            epochs=summary.get("epoch"),
            selected_epoch=summary.get("epoch"),
            seed=args.get("seed", "unknown"),
            schedule={k: args[k] for k in SCHEDULE_KEYS if k in args},
            split=split,
            teacher=None,
            cache=None,
            cache_available=None,
        )
        stage.schedule["protocol_epoch"] = summary.get("epoch")
        prof.chain = [stage]
        prof.seed = stage.seed
        prof.seed_basis = "args.seed" if "seed" in args else "not recorded"
        prof.training_stage_split = split
        prof.teacher = None
        prof.teacher_basis = "teacher itself"
        return prof

    if node["kind"] != "mamba_ckpt":
        raise ComparisonError(
            f"{node_id}: kind {node['kind']!r} has no training profile"
        )

    prof.head_family, prof.temporal_blocks_present = _head_family(node)
    prof.param_count = (node.get("summary") or {}).get("param_count")

    # Walk the warm-start chain root-ward.  Stops at a node the inventory does
    # not have (chain incomplete) or at a self-resume (args are the resume
    # invocation; earlier stages unrecorded).
    chain: list[Stage] = []
    cursor: str | None = node_id
    visited: set[str] = set()
    complete = True
    note = ""
    while cursor is not None:
        if cursor in visited:
            complete = False
            note = f"cycle at {cursor}"
            break
        visited.add(cursor)
        cur = nodes[cursor]
        if not cur.get("exists"):
            complete = False
            note = f"chain reaches unavailable node {cursor}"
            break
        chain.append(_stage_of(cursor, cur))
        init_node, init_path, init_exists = _edge_target(cur, "init")
        self_resume, _, _ = _edge_target(cur, "self_resume")
        if self_resume:
            complete = False
            note = f"{cursor} stores its resume invocation; earlier stages unrecorded"
            break
        if init_path is None:
            break
        if init_node is None:
            complete = False
            note = (
                f"{cursor} warm-starts from {init_path} which is not an inventory node"
            )
            break
        if not init_exists:
            chain.append(
                Stage(
                    init_node,
                    None,
                    None,
                    "unknown",
                    {},
                    resolve_sequences("", ""),
                    None,
                    None,
                    None,
                )
            )
            complete = False
            note = f"chain reaches unavailable node {init_node}"
            break
        cursor = init_node
    chain.reverse()
    prof.chain = chain
    prof.chain_complete = complete
    prof.chain_note = note

    init_node, init_path, _ = _edge_target(node, "init")
    prof.warm_start = init_node if init_node else (init_path or None)
    prof.warm_start_basis = (
        "args.mamba_ckpt" if init_path else "no warm start recorded (chain root)"
    )

    teachers = sorted({s.teacher for s in chain if s.teacher})
    prof.teacher = (
        teachers[0]
        if len(teachers) == 1
        else (",".join(teachers) if teachers else None)
    )
    prof.teacher_basis = "args.teacher_ckpt along the chain"
    caches = [s.cache for s in chain if s.cache]
    prof.cache_nodes = sorted(set(caches))
    avail = [s.cache_available for s in chain if s.cache]
    prof.caches_available = all(avail) if avail else None
    own = chain[-1] if chain else None
    prof.seed = own.seed if own else "unknown"
    prof.seed_basis = "args.seed" if own and own.seed != "unknown" else "not recorded"
    splits = [
        json.dumps(s.split["train"]) + "|" + json.dumps(s.split["holdout"])
        for s in chain
        if s.epochs is not None
    ]
    if own is not None:
        prof.training_stage_split = own.split
        if len(set(splits)) > 1:
            prof.training_stage_split = {
                "train": own.split["train"],
                "holdout": own.split["holdout"],
                "basis": "stages disagree: " + "; ".join(sorted(set(splits))),
            }
    return prof


# --------------------------------------------------------------------------- runtime binding


def _node_by_path(inventory: dict[str, Any], path: str | None) -> str | None:
    if not path:
        return None
    for nid, node in inventory["nodes"].items():
        if node.get("path") == path:
            return nid
    return None


def _engine_teacher_evidence(
    inventory: dict[str, Any], engine_node: str | None
) -> dict[str, Any]:
    if engine_node is None:
        return {"status": "unknown", "onnx_matches": None}
    node = inventory["nodes"][engine_node]
    verdict = None
    matches: list[str] = []
    for edge in node.get("edges", []):
        if edge.get("relation") == "onnx_matches_checkpoint":
            verdict = (edge.get("detail") or {}).get("verdict")
            if edge.get("target_node"):
                matches.append(edge["target_node"])
    if verdict != "unique_exact" or len(matches) != 1:
        return {"status": "unattributed", "onnx_matches": matches, "verdict": verdict}
    return {
        "status": "sibling_onnx_unique",
        "onnx_matches": matches[0],
        "verdict": verdict,
    }


def bind_runtime(
    profile: Profile, runtime: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    """What system the endpoint is measured as, under the declared binding."""
    nodes = inventory["nodes"]
    binding = runtime.get("binding", "family_preset")
    out: dict[str, Any] = {"binding": binding}
    if binding == "family_preset":
        preset_id = f"{profile.family}.deployment_preset"
        preset = nodes.get(preset_id)
        if preset is None or not preset.get("exists"):
            out["status"] = "unknown"
            out["reason"] = f"{preset_id} not in inventory"
            return out
        fwd = (preset.get("checks") or {}).get("deployment_forward") or {}
        summary = preset.get("summary") or {}
        engine_node = runtime.get("backbone_engine_node") or _node_by_path(
            inventory, summary.get("fpn_backbone_engine")
        )
        if profile.kind != "mamba_ckpt":
            out["status"] = "not_applicable"
            out["reason"] = (
                f"{preset_id} drives a Mamba head; a {profile.kind} endpoint needs an explicit runtime binding"
            )
            return out
        evidence = _engine_teacher_evidence(inventory, engine_node)
        engine_exists = bool(engine_node and nodes[engine_node].get("exists"))
        consistency = "unknown"
        if evidence["status"] == "sibling_onnx_unique" and profile.teacher:
            consistency = (
                "same_teacher_indicated"
                if evidence["onnx_matches"] == profile.teacher
                else "DIFFERENT_TEACHER_INDICATED"
            )
        out.update(
            {
                "status": "bound",
                "preset": preset_id,
                "preset_path": preset.get("path"),
                "preset_sha256": preset.get("sha256"),
                "backbone_engine_node": engine_node,
                "backbone_engine_exists": engine_exists,
                "backbone_engine_sibling_onnx": evidence,
                "backbone_teacher_consistency": consistency,
                "engine_bytes_attributed": False,
                "head_path": fwd.get("head_engine"),
                "head_engine_node": _node_by_path(
                    inventory, summary.get("mamba_head_engine")
                ),
                "temporal_blocks": fwd.get("temporal_blocks"),
                "effective_T": 1,
                "gate_teacher_at_runtime": fwd.get("gate_teacher_at_runtime"),
                "final_stage_gt_ratio": fwd.get("final_stage_gt_ratio"),
                "embedding": fwd.get("embedding"),
                "graphs": fwd.get("graphs"),
                "tracker_policy_source": "preset",
            }
        )
        return out

    # shared_backbone_node: both endpoints run on one backbone checkpoint in
    # PyTorch, eager.  The head path follows the endpoint's kind.
    bnode = runtime["backbone_node"]
    b = nodes[bnode]
    head_path = {
        "mamba_ckpt": "mamba head, PyTorch eager (--no-temporal required: streaming temporal is the non-whole-graph default)",
        "gated_teacher_ckpt": "native Detect head via TeacherHeadDetector, PyTorch eager",
        "yolo_pt": "native Detect head, PyTorch eager (runner not in inventory)",
    }[profile.kind]
    out.update(
        {
            "status": "bound" if b.get("exists") else "unknown",
            "reason": None if b.get("exists") else f"{bnode} unavailable",
            "backbone_engine_node": None,
            "backbone_node": bnode,
            "backbone_teacher_consistency": (
                "same_teacher_indicated"
                if profile.teacher == bnode or profile.node == bnode
                else "DIFFERENT"
            ),
            "engine_bytes_attributed": None,
            "head_path": head_path,
            "temporal_blocks": (
                "present; must be bypassed explicitly (--no-temporal)"
                if profile.temporal_blocks_present
                else "absent"
            ),
            "effective_T": 1,
            "gate_teacher_at_runtime": None,
            "graphs": {
                "use_whole_graph": False,
                "use_cuda_graph": False,
                "use_tracker_graph": False,
            },
            "tracker_policy_source": "contract_required",
        }
    )
    return out


# --------------------------------------------------------------------------- axis comparison


def _axis(status: str, lhs: Any, rhs: Any, detail: str = "") -> dict[str, Any]:
    assert status in AXIS_STATUSES, status
    return {"status": status, "lhs": lhs, "rhs": rhs, "detail": detail}


def _eq_or(a: Any, b: Any, unknown_if_none: bool = True) -> str:
    if unknown_if_none and (a is None or b is None or a == "unknown" or b == "unknown"):
        return "unknown"
    return "matched" if a == b else "unmatched"


def _common_ancestor(lp: Profile, rp: Profile) -> str | None:
    l_ids = [s.node for s in lp.chain]
    r_ids = {s.node for s in rp.chain}
    common = [n for n in l_ids if n in r_ids]
    return common[-1] if common else None


def _stages_after(profile: Profile, ancestor: str | None) -> list[Stage]:
    if ancestor is None:
        return list(profile.chain)
    ids = [s.node for s in profile.chain]
    idx = ids.index(ancestor)
    return profile.chain[idx + 1 :]


def compare_schedule(
    l_stages: list[Stage], r_stages: list[Stage], treatment_keys: set[str]
) -> dict[str, Any]:
    """Per-key value sets across the compared stages; keys on one side only are listed, not counted."""
    # A key counts only where both compared stages recorded it; a flag the
    # training script grew later (absent on an older stage) is listed as
    # one-sided, not interpreted — same convention as the inventory.
    differing: dict[str, Any] = {}
    one_sided_set: set[str] = set()
    staging_differs = len(l_stages) != len(r_stages)
    if not l_stages or not r_stages:
        for st in l_stages + r_stages:
            one_sided_set |= set(st.schedule)
    elif staging_differs:
        # No stage alignment possible: compare the value sets each side used,
        # over the keys every compared stage recorded.
        all_stages = l_stages + r_stages
        union = set().union(*(set(st.schedule) for st in all_stages))
        every = set.intersection(*(set(st.schedule) for st in all_stages))
        one_sided_set = union - every
        for key in sorted(every):
            ls = sorted({json.dumps(st.schedule[key]) for st in l_stages})
            rs = sorted({json.dumps(st.schedule[key]) for st in r_stages})
            if ls != rs:
                differing[key] = {
                    "lhs": [json.loads(v) for v in ls],
                    "rhs": [json.loads(v) for v in rs],
                }
    else:
        # Stage-aligned: compare stage i with stage i, key by key.
        per_key: dict[str, tuple[list[Any], list[Any]]] = {}
        for ls_, rs_ in zip(l_stages, r_stages):
            shared = set(ls_.schedule) & set(rs_.schedule)
            one_sided_set |= set(ls_.schedule) ^ set(rs_.schedule)
            for key in shared:
                per_key.setdefault(key, ([], []))
                per_key[key][0].append(ls_.schedule[key])
                per_key[key][1].append(rs_.schedule[key])
        for key, (lv, rv) in sorted(per_key.items()):
            if lv != rv:
                differing[key] = {"lhs": lv, "rhs": rv}
    one_sided = sorted(one_sided_set)
    treatment_diff = {k: v for k, v in differing.items() if k in treatment_keys}
    confound_diff = {k: v for k, v in differing.items() if k not in treatment_keys}
    if staging_differs and "staging" not in treatment_keys:
        confound_diff["staging"] = {"lhs": len(l_stages), "rhs": len(r_stages)}
    elif staging_differs:
        treatment_diff["staging"] = {"lhs": len(l_stages), "rhs": len(r_stages)}
    return {
        "treatment_diff": treatment_diff,
        "confound_diff": confound_diff,
        "one_sided_keys": one_sided,
        "lhs_stages": [s.node for s in l_stages],
        "rhs_stages": [s.node for s in r_stages],
    }


def _budget(stages: list[Stage]) -> int | None:
    total = 0
    for s in stages:
        if s.epochs is None:
            return None
        total += int(s.epochs)
    return total


def _tracker_policy(
    inventory: dict[str, Any], lr: dict[str, Any], rr: dict[str, Any], repo: Path
) -> dict[str, Any]:
    """Diff the two presets' non-artifact keys, fail-closed on preset drift vs the inventory."""
    if (
        lr.get("tracker_policy_source") != "preset"
        or rr.get("tracker_policy_source") != "preset"
    ):
        return _axis(
            "unknown",
            lr.get("tracker_policy_source"),
            rr.get("tracker_policy_source"),
            "tracker policy must be fixed by the shared eval contract (deliverable 3); not derivable from the inventory",
        )
    if lr["preset"] == rr["preset"]:
        return _axis(
            "matched",
            lr["preset"],
            rr["preset"],
            f"same preset {lr['preset_path']} @ {lr['preset_sha256'][:12]}",
        )
    loaded: dict[str, dict[str, Any]] = {}
    for side in (lr, rr):
        path = repo / side["preset_path"]
        if not path.exists():
            return _axis(
                "unknown",
                lr["preset"],
                rr["preset"],
                f"{side['preset_path']} missing in checkout",
            )
        if sha256_file(path) != side["preset_sha256"]:
            return _axis(
                "unknown",
                lr["preset"],
                rr["preset"],
                f"{side['preset_path']} drifted from the inventory snapshot sha; re-capture the inventory first",
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        loaded[side["preset"]] = {
            k: v for k, v in data.items() if k not in PRESET_ARTIFACT_KEYS
        }
    a, b = loaded[lr["preset"]], loaded[rr["preset"]]
    diff = {
        k: {"lhs": a.get(k), "rhs": b.get(k)}
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    }
    if not diff:
        return _axis(
            "matched",
            lr["preset"],
            rr["preset"],
            "presets differ only in artifact keys",
        )
    return _axis(
        "unmatched",
        lr["preset"],
        rr["preset"],
        "tracker/runtime keys differ: " + json.dumps(diff, sort_keys=True),
    )


def compare_profiles(
    lp: Profile,
    rp: Profile,
    lr: dict[str, Any],
    rr: dict[str, Any],
    spec: dict[str, Any],
    inventory: dict[str, Any],
    repo: Path,
) -> dict[str, dict[str, Any]]:
    axes: dict[str, dict[str, Any]] = {}
    treatment_keys = set(spec.get("treatment_schedule_keys", []))

    axes["backbone_family"] = _axis(
        _eq_or(lp.backbone_family, rp.backbone_family),
        lp.backbone_family,
        rp.backbone_family,
    )

    lh, rh = lp.head_family, rp.head_family
    if lh is None or rh is None:
        axes["head_family"] = _axis("unknown", lh, rh, "no architecture record")
    elif lh.get("family") == rh.get("family") == "yolo_detect" and {
        k: v for k, v in lh.items() if k != "gate_module"
    } == {k: v for k, v in rh.items() if k != "gate_module"}:
        axes["head_family"] = _axis(
            "matched_effective",
            lh,
            rh,
            "same native Detect head; the gate module is identity at deployment (gt_ratio 0)",
        )
    elif lh != rh:
        axes["head_family"] = _axis(
            "unmatched", lh, rh, "architecture signature differs"
        )
    elif lp.temporal_blocks_present != rp.temporal_blocks_present:
        axes["head_family"] = _axis(
            "matched_effective",
            {
                **lh,
                "temporal_blocks": lp.temporal_blocks_present,
                "params": lp.param_count,
            },
            {
                **rh,
                "temporal_blocks": rp.temporal_blocks_present,
                "params": rp.param_count,
            },
            "same head architecture; one side carries temporal blocks that the T=1 forward bypasses (artifact differs, effective forward does not)",
        )
    else:
        axes["head_family"] = _axis(
            "matched",
            lh,
            rh,
            ""
            if lp.param_count == rp.param_count
            else f"same signature; param_count {lp.param_count} vs {rp.param_count} (input width follows the backbone)",
        )

    ls, rs = lp.training_stage_split, rp.training_stage_split
    if ls is None or rs is None:
        axes["dataset_split"] = _axis("unknown", ls, rs)
    else:
        same = ls["train"] == rs["train"] and ls["holdout"] == rs["holdout"]
        note = (
            ""
            if ls.get("basis") == rs.get("basis")
            else f"bases differ ({ls.get('basis')} vs {rs.get('basis')}) but resolve to the same set"
            if same
            else ""
        )
        axes["dataset_split"] = _axis("matched" if same else "unmatched", ls, rs, note)

    # Deployment axes under the runtime binding.
    if lr.get("status") != "bound" or rr.get("status") != "bound":
        why = "; ".join(filter(None, [lr.get("reason"), rr.get("reason")]))
        for axis in (
            "inference_forward_mode",
            "deployed_backbone_artifact",
            "deployed_head_artifact",
            "tracker_runtime_policy",
        ):
            axes[axis] = _axis("unknown", lr.get("status"), rr.get("status"), why)
    else:
        fwd_keys = (
            "effective_T",
            "gate_teacher_at_runtime",
            "embedding",
            "graphs",
            "final_stage_gt_ratio",
        )
        lf = {k: lr.get(k) for k in fwd_keys}
        rf = {k: rr.get(k) for k in fwd_keys}
        fwd_diff = {k: {"lhs": lf[k], "rhs": rf[k]} for k in fwd_keys if lf[k] != rf[k]}
        axes["inference_forward_mode"] = _axis(
            "matched" if not fwd_diff else "unmatched",
            lf,
            rf,
            "" if not fwd_diff else "differs: " + json.dumps(fwd_diff, sort_keys=True),
        )

        lb = {
            "engine_node": lr.get("backbone_engine_node") or lr.get("backbone_node"),
            "teacher_consistency": lr.get("backbone_teacher_consistency"),
        }
        rb = {
            "engine_node": rr.get("backbone_engine_node") or rr.get("backbone_node"),
            "teacher_consistency": rr.get("backbone_teacher_consistency"),
        }
        if lb["engine_node"] is None or rb["engine_node"] is None:
            st = "unknown"
        elif lb == rb:
            st = "matched"
        else:
            st = "unmatched"
        exists_note = ""
        if lr.get("backbone_engine_node") and not lr.get("backbone_engine_exists"):
            st, exists_note = "unknown", f"{lr['backbone_engine_node']} unavailable"
        if rr.get("backbone_engine_node") and not rr.get("backbone_engine_exists"):
            st, exists_note = "unknown", f"{rr['backbone_engine_node']} unavailable"
        axes["deployed_backbone_artifact"] = _axis(
            st,
            lb,
            rb,
            exists_note
            or "engine bytes never attributed; identity = node sha, provenance = sibling-ONNX evidence",
        )

        lhp = {
            "head_path": lr.get("head_path"),
            "head_engine_node": lr.get("head_engine_node"),
            "temporal_blocks": lr.get("temporal_blocks"),
        }
        rhp = {
            "head_path": rr.get("head_path"),
            "head_engine_node": rr.get("head_engine_node"),
            "temporal_blocks": rr.get("temporal_blocks"),
        }
        if lp.kind != rp.kind:
            axes["deployed_head_artifact"] = _axis(
                "unmatched", lhp, rhp, "different head kinds"
            )
        elif lhp == rhp:
            axes["deployed_head_artifact"] = _axis("matched", lhp, rhp)
        elif (
            lhp["head_path"] == rhp["head_path"]
            and lhp["head_engine_node"] == rhp["head_engine_node"]
        ):
            axes["deployed_head_artifact"] = _axis(
                "matched_effective",
                lhp,
                rhp,
                "same head path; temporal-block presence differs but is bypassed at T=1",
            )
        else:
            axes["deployed_head_artifact"] = _axis("unmatched", lhp, rhp)

        axes["tracker_runtime_policy"] = _tracker_policy(inventory, lr, rr, repo)

    # Training axes.
    if lp.external_pretraining or rp.external_pretraining:
        ext = "external pretraining on one side"
        axes["warm_start"] = _axis("unmatched", lp.warm_start, rp.warm_start, ext)
        axes["teacher"] = _axis("unmatched", lp.teacher, rp.teacher, ext)
        axes["teacher_cache"] = _axis("unmatched", lp.cache_nodes, rp.cache_nodes, ext)
        axes["training_seed"] = _axis("unmatched", lp.seed, rp.seed, ext)
        axes["training_schedule"] = _axis("unmatched", None, None, ext)
        axes["training_budget"] = _axis("unmatched", None, None, ext)
        return axes

    ancestor = _common_ancestor(lp, rp)
    l_after, r_after = _stages_after(lp, ancestor), _stages_after(rp, ancestor)
    design = spec["design"]
    # A stage increment compares the parent's own recipe with the added
    # stage(s), mirroring the inventory's warm-start edge delta; the budget is
    # the added epochs.
    sched_l = [lp.chain[-1]] if design == "stage_increment" and lp.chain else l_after
    if design == "stage_increment":
        ws_status = "unmatched" if ancestor == lp.node and r_after else "unknown"
        ws_detail = (
            f"rhs descends from lhs through {[s.node for s in r_after]}"
            if ws_status == "unmatched"
            else "rhs is not a descendant of lhs"
        )
    elif lp.node == rp.node:
        ws_status, ws_detail = "matched", "same checkpoint"
    elif ancestor is None:
        ws_status = "unmatched"
        l_root = lp.chain[0].node if lp.chain else lp.warm_start
        r_root = rp.chain[0].node if rp.chain else rp.warm_start
        ws_detail = (
            f"no common ancestor in the inventory (chains root at {l_root} vs {r_root})"
        )
        l_first = lp.chain[1].node if len(lp.chain) > 1 else None
        r_first = rp.chain[1].node if len(rp.chain) > 1 else None
        if l_first and r_first and lp.family == rp.family:
            ws_detail += "; GT1-start confound: the compared heads warm-start from different checkpoints of the same stage kind"
    elif not l_after or not r_after:
        ws_status = "unmatched"
        ws_detail = (
            f"rhs descends from lhs through {[s.node for s in r_after]}"
            if not l_after
            else f"lhs descends from rhs through {[s.node for s in l_after]}"
        )
    else:
        ws_status, ws_detail = (
            "matched",
            f"both sides warm-start from the common ancestor {ancestor}",
        )
    if not lp.chain_complete or not rp.chain_complete:
        ws_detail += "; chain incomplete: " + "; ".join(
            filter(None, [lp.chain_note, rp.chain_note])
        )
    ws_l = (
        ancestor
        if (design != "stage_increment" and ancestor and l_after and r_after)
        else lp.warm_start
    )
    ws_r = (
        ancestor
        if (design != "stage_increment" and ancestor and l_after and r_after)
        else rp.warm_start
    )
    axes["warm_start"] = _axis(ws_status, ws_l, ws_r, ws_detail)

    axes["teacher"] = _axis(_eq_or(lp.teacher, rp.teacher), lp.teacher, rp.teacher)

    lc, rc = lp.cache_nodes, rp.cache_nodes
    l_cache_stages = [s for s in sched_l if s.cache]
    r_cache_stages = [s for s in r_after if s.cache]
    lc_after = sorted({s.cache for s in l_cache_stages})
    rc_after = sorted({s.cache for s in r_cache_stages})
    if lc_after == rc_after:
        cache_status = "matched"
        cache_detail = "same cache node(s) in the compared stages" + (
            ""
            if (lp.caches_available and rp.caches_available)
            else "; cache unavailable ⇒ no retrain replay"
        )
    else:
        cache_status = "unmatched"
        cache_detail = f"compared stages use {lc_after} vs {rc_after}"
    axes["teacher_cache"] = _axis(cache_status, lc, rc, cache_detail)

    axes["training_seed"] = _axis(
        _eq_or(lp.seed, rp.seed),
        lp.seed,
        rp.seed,
        f"bases: {lp.seed_basis} / {rp.seed_basis}",
    )

    if not lp.chain_complete or not rp.chain_complete:
        axes["training_schedule"] = _axis(
            "unknown", None, None, "chain incomplete; recipe not fully recorded"
        )
        axes["training_budget"] = _axis(
            "unknown", None, None, "chain incomplete; budget not fully recorded"
        )
        return axes
    sched = compare_schedule(sched_l, r_after, treatment_keys)
    if sched["confound_diff"]:
        sched_status = "unmatched"
    elif sched["treatment_diff"]:
        sched_status = "unmatched"  # differs, but only in declared treatment keys
    else:
        sched_status = "matched"
    axes["training_schedule"] = _axis(
        sched_status,
        sched["lhs_stages"],
        sched["rhs_stages"],
        json.dumps(sched, sort_keys=True),
    )
    axes["training_schedule"]["treatment_only"] = (
        bool(sched["treatment_diff"]) and not sched["confound_diff"]
    )
    axes["training_schedule"]["confound_keys"] = sorted(sched["confound_diff"].keys())

    lb_, rb_ = _budget(l_after), _budget(r_after)
    axes["training_budget"] = _axis(
        _eq_or(lb_, rb_),
        lb_,
        rb_,
        f"epochs planned after common ancestor {ancestor}"
        if ancestor
        else "epochs planned over the whole recorded chain (no common ancestor)",
    )
    return axes


# --------------------------------------------------------------------------- classification


def classify(
    spec: dict[str, Any],
    lp: Profile,
    rp: Profile,
    lr: dict[str, Any],
    rr: dict[str, Any],
    axes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    treatment = set(spec.get("treatment_axes", []))
    design = spec["design"]
    blockers: list[str] = []
    confounds: list[dict[str, Any]] = []

    for prof in (lp, rp):
        if not prof.exists:
            blockers.append(f"{prof.node} unavailable")
    for side, rt in (("lhs", lr), ("rhs", rr)):
        if (
            rt.get("status") == "unknown"
            and rt.get("reason")
            and "unavailable" in rt["reason"]
        ):
            blockers.append(f"{side} runtime: {rt['reason']}")

    # Design premises the bytes must not contradict.
    if not blockers:
        if design == "paired_siblings":
            if axes["warm_start"]["status"] != "matched":
                blockers.append(
                    "paired_siblings premise violated: warm starts differ or are unknown"
                )
            if axes["training_budget"]["status"] != "matched":
                blockers.append(
                    "paired_siblings premise violated: budgets after the common ancestor differ or are unknown"
                )
        elif design == "stage_increment":
            if (
                axes["warm_start"]["status"] != "unmatched"
                or "descends" not in axes["warm_start"]["detail"]
            ):
                blockers.append(
                    "stage_increment premise violated: rhs is not a warm-start descendant of lhs"
                )
        elif design == "seed_replicate":
            if (
                axes["warm_start"]["status"] != "matched"
                or axes["training_schedule"]["status"] != "matched"
            ):
                blockers.append(
                    "seed_replicate premise violated: warm start or recipe differ"
                )
            if axes["training_seed"]["status"] != "unmatched":
                blockers.append("seed_replicate premise violated: seeds do not differ")
        elif design == "runtime_ab":
            if lp.node != rp.node:
                blockers.append(
                    "runtime_ab premise violated: the two sides must be the same checkpoint"
                )

    # The s backbone/teacher split: common-mode when both endpoints deploy with the
    # same consistency status, blocking when they differ.
    lcons, rcons = (
        lr.get("backbone_teacher_consistency"),
        rr.get("backbone_teacher_consistency"),
    )
    if lr.get("status") == "bound" and rr.get("status") == "bound":
        if lcons == rcons and lcons in ("DIFFERENT_TEACHER_INDICATED", "unknown"):
            confounds.append(
                {
                    "name": "deployed_backbone_teacher_mismatch",
                    "severity": "common_mode",
                    "detail": (
                        f"both endpoints deploy on {lr.get('backbone_engine_node') or lr.get('backbone_node')} whose sibling ONNX indicates "
                        f"{(lr.get('backbone_engine_sibling_onnx') or {}).get('onnx_matches')} while the heads were trained against {lp.teacher}; "
                        "engine bytes unattributed. Cannot explain a paired delta; absolute numbers are on an unintended backbone."
                    ),
                }
            )
        elif lcons != rcons and design != "runtime_ab":
            same_family = lp.family == rp.family
            confounds.append(
                {
                    "name": "deployed_backbone_teacher_mismatch",
                    "severity": "blocking" if same_family else "asymmetric",
                    "detail": (
                        f"endpoints differ in backbone/teacher consistency ({lcons} vs {rcons}); the delta is not common-mode"
                        + (
                            ""
                            if same_family
                            else "; tolerated only because the two sides are different systems by declaration"
                        )
                    ),
                }
            )
            if same_family:
                blockers.append(
                    "asymmetric backbone/teacher consistency between endpoints"
                )

    # An unknown axis is a confound even when declared as treatment: a treatment
    # the bytes cannot show is not one a delta can be attributed to.
    unmatched_struct = [
        a
        for a in STRUCTURAL_AXES
        if axes[a]["status"] == "unknown"
        or (axes[a]["status"] == "unmatched" and a not in treatment)
    ]
    unmatched_train = [
        a
        for a in TRAINING_AXES
        if axes[a]["status"] == "unknown"
        or (axes[a]["status"] == "unmatched" and a not in treatment)
    ]
    # Declaring the schedule as treatment excuses only the declared keys: any
    # other differing key (e.g. warm-up) is a confound whatever the declaration says.
    sched = axes["training_schedule"]
    if (
        sched["status"] == "unmatched"
        and sched.get("confound_keys")
        and "training_schedule" not in unmatched_train
    ):
        unmatched_train.append("training_schedule")
    unknown_axes = [a for a in AXES if axes[a]["status"] == "unknown"]

    for a in unmatched_struct + unmatched_train:
        ax = axes[a]
        confounds.append(
            {
                "name": a,
                "severity": "structural" if a in STRUCTURAL_AXES else "training",
                "status": ax["status"],
                "lhs": ax["lhs"],
                "rhs": ax["rhs"],
                "detail": ax.get("detail", "")
                if a != "training_schedule"
                else f"non-treatment schedule keys differ: {ax.get('confound_keys')}",
            }
        )

    matched_axes = [
        a for a in AXES if axes[a]["status"] in ("matched", "matched_effective")
    ]
    treatment_axes_seen = [
        a for a in AXES if a in treatment and axes[a]["status"] == "unmatched"
    ]

    if blockers:
        cls = "blocked"
    elif unmatched_struct:
        cls = "system_comparison"
    elif unmatched_train:
        cls = "historical_not_comparable"
    elif design == "system":
        cls = "blocked"
        blockers.append(
            "system design with no structural difference: declare a training design instead"
        )
    else:
        cls = "controlled"

    # Fail closed: nothing with an unknown axis is controlled, whatever the arithmetic above said.
    if cls == "controlled" and unknown_axes:
        cls = "historical_not_comparable"
        confounds.append(
            {"name": "unknown_axes", "severity": "training", "detail": unknown_axes}
        )
    if cls == "controlled" and not treatment_axes_seen and design != "runtime_ab":
        cls = "blocked"
        blockers.append("declared treatment does not differ between the endpoints")

    return {
        "classification": cls,
        "matched_axes": matched_axes,
        "treatment_axes_observed": treatment_axes_seen,
        "remaining_confounds": confounds,
        "blocking_confounds": blockers,
        "unknown_axes": unknown_axes,
    }


# --------------------------------------------------------------------------- historical results


def load_results_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def historical_rows(
    rows: list[dict[str, str]], node: dict[str, Any]
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("checkpoint") == node.get("path"):
            out.append(
                {
                    "experiment": row.get("experiment"),
                    "result_dir": row.get("result_dir"),
                    "protocol_status": row.get("protocol_status"),
                    "preset_sha_recorded": "preset_sha256" in row
                    and bool(row["preset_sha256"]),
                    "engine_sha_recorded": "engine_sha256" in row
                    and bool(row["engine_sha256"]),
                }
            )
    return out


def probe_result_manifests(rows: list[dict[str, str]], repo: Path) -> dict[str, Any]:
    """Workspace probe (captured, not derived): do the referenced results dirs carry a manifest?"""
    dirs = sorted({r["result_dir"] for r in rows if r.get("result_dir")})
    found = {}
    for d in dirs:
        p = repo / d
        found[d] = {
            "exists": p.is_dir(),
            "run_manifest": (p / "run_manifest.json").is_file(),
        }
    return {
        "result_dirs": found,
        "with_manifest": sum(1 for v in found.values() if v["run_manifest"]),
        "total": len(found),
    }


# --------------------------------------------------------------------------- matrix


def executability(
    lp: Profile,
    rp: Profile,
    lr: dict[str, Any],
    rr: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    nodes = inventory["nodes"]
    fresh_missing: list[str] = []
    for prof, rt in ((lp, lr), (rp, rr)):
        if not prof.exists:
            fresh_missing.append(prof.node)
        eng = rt.get("backbone_engine_node")
        if eng and not nodes[eng].get("exists"):
            fresh_missing.append(eng)
        heng = rt.get("head_engine_node")
        if heng and not nodes[heng].get("exists"):
            fresh_missing.append(heng)
        bn = rt.get("backbone_node")
        if bn and not nodes[bn].get("exists"):
            fresh_missing.append(bn)
    retrain_missing: list[str] = []
    for prof in (lp, rp):
        if not prof.chain_complete:
            retrain_missing.append(f"{prof.node}: {prof.chain_note}")
        for s in prof.chain:
            if s.cache and s.cache_available is False:
                retrain_missing.append(f"{s.node} needs {s.cache} (unavailable)")
    return {
        "fresh_eval": not fresh_missing,
        "fresh_eval_missing": sorted(set(fresh_missing)),
        "retrain_replay": not retrain_missing,
        "retrain_replay_missing": sorted(set(retrain_missing)),
    }


def build_matrix(
    inventory: dict[str, Any],
    declarations: dict[str, Any],
    results_rows: list[dict[str, str]],
    repo: Path,
    *,
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    comparisons = []
    for spec in declarations["comparisons"]:
        lp = build_profile(spec["lhs"]["node"], inventory)
        rp = build_profile(spec["rhs"]["node"], inventory)
        lr = bind_runtime(
            lp, spec["lhs"].get("runtime", {"binding": "family_preset"}), inventory
        )
        rr = bind_runtime(
            rp, spec["rhs"].get("runtime", {"binding": "family_preset"}), inventory
        )
        lp.runtime, rp.runtime = lr, rr
        axes = compare_profiles(lp, rp, lr, rr, spec, inventory, repo)
        verdict = classify(spec, lp, rp, lr, rr, axes)
        hist_l = historical_rows(results_rows, inventory["nodes"][lp.node])
        hist_r = historical_rows(results_rows, inventory["nodes"][rp.node])
        comparisons.append(
            {
                "comparison_id": spec["comparison_id"],
                "group": spec.get("group"),
                "design": spec["design"],
                "lhs": {"node": lp.node, "runtime": lr},
                "rhs": {"node": rp.node, "runtime": rr},
                "intended_treatment": spec["intended_treatment"],
                "treatment_axes": spec.get("treatment_axes", []),
                "treatment_schedule_keys": spec.get("treatment_schedule_keys", []),
                "question": spec.get("question"),
                "axes": axes,
                **verdict,
                "executability": executability(lp, rp, lr, rr, inventory),
                "historical_results": {
                    "lhs": hist_l,
                    "rhs": hist_r,
                    "reusable_as_paired": False,
                    "reason": "record carries no preset/engine sha for any run; runtime identity of historical runs is not established (deliverable 4)",
                },
                "declared_notes": spec.get("notes", []),
                "profiles": {"lhs": lp.to_json(), "rhs": rp.to_json()},
            }
        )
    counts = {
        c: sum(1 for x in comparisons if x["classification"] == c) for c in CLASSES
    }
    payload = {
        "schema": SCHEMA,
        "issue": inventory.get("issue"),
        "inputs": {
            "inventory": {
                "path": str(DEFAULT_INVENTORY),
                "schema": inventory.get("schema"),
                "captured": inventory.get("captured"),
            },
            "declarations_schema": declarations.get("schema"),
            "results_table": str(DEFAULT_RESULTS_TABLE),
        },
        "axes": {"structural": list(STRUCTURAL_AXES), "training": list(TRAINING_AXES)},
        "designs": {k: sorted(v) for k, v in DESIGNS.items()},
        "classification_rule": [
            "blocked: endpoint/runtime artifact unavailable, design premise contradicted by bytes, or asymmetric backbone/teacher consistency",
            "system_comparison: a structural axis is unmatched/unknown outside the treatment",
            "historical_not_comparable: only training axes are unmatched/unknown outside the treatment",
            "controlled: every non-treatment axis matched (or matched_effective), no unknown axis, treatment observed",
        ],
        "default_sdp_sequences": list(DEFAULT_SDP_SEQUENCES),
        "counts": counts,
        "comparisons": comparisons,
        "not_representable": declarations.get("not_representable", []),
        "workspace_probe": probe,
    }
    return payload


# --------------------------------------------------------------------------- markdown


def _sched_confound_keys(c: dict[str, Any]) -> list[str]:
    return list(c["axes"]["training_schedule"].get("confound_keys") or [])


def findings_section(payload: dict[str, Any]) -> list[str]:
    """Derived readings of the matrix — every statement here is computed from the rows."""
    rows = payload["comparisons"]
    by_class: dict[str, list[dict[str, Any]]] = {k: [] for k in CLASSES}
    for c in rows:
        by_class[c["classification"]].append(c)
    fam = lambda c: c["lhs"]["node"].split(".")[0]  # noqa: E731
    lines = ["## What the matrix establishes", ""]

    ctrl_train = [c for c in by_class["controlled"] if c["design"] != "runtime_ab"]
    lines.append(
        f"**Attributable same-family training comparisons ({len(ctrl_train)}):**"
    )
    for design in ("paired_siblings", "seed_replicate", "stage_increment"):
        ids = [c for c in ctrl_train if c["design"] == design]
        if ids:
            lines.append(
                f"- `{design}`: " + ", ".join(f"`{c['comparison_id']}`" for c in ids)
            )
    ctrl_rt = [c for c in by_class["controlled"] if c["design"] == "runtime_ab"]
    if ctrl_rt:
        lines.append(
            "- runtime attribution (`runtime_ab`, not a training claim): "
            + ", ".join(f"`{c['comparison_id']}`" for c in ctrl_rt)
        )
    lines.append("")

    fams = sorted({fam(c) for c in rows})
    for f in fams:
        cur = [
            c for c in ctrl_train if fam(c) == f and c["design"] == "paired_siblings"
        ]
        if not cur:
            hist = [
                c
                for c in rows
                if fam(c) == f
                and c["design"] == "paired_siblings"
                and c["classification"] != "blocked"
            ]
            why = sorted({k for c in hist for k in _sched_confound_keys(c)})
            lines.append(
                f"- **family `{f}` has no controlled curriculum (paired_siblings) comparison**"
                + (
                    f"; its declared pairs carry non-treatment schedule differences {why}"
                    if why
                    else ""
                )
                + "."
            )
    lines.append("")

    sysc = by_class["system_comparison"]
    lines.append(
        f"**System comparisons ({len(sysc)})** — quality/cost only, never a post-training attribution:"
    )
    for c in sysc:
        need = []
        if c["axes"]["tracker_runtime_policy"]["status"] == "unknown":
            need.append("tracker policy fixed by the eval contract")
        if not c["executability"]["fresh_eval"]:
            need.append(
                "missing: " + ", ".join(c["executability"]["fresh_eval_missing"])
            )
        rt = {c["lhs"]["runtime"].get("binding"), c["rhs"]["runtime"].get("binding")}
        if "shared_backbone_node" in rt:
            need.append("eager PyTorch runner (not the whole-graph deployment)")
        lines.append(
            f"- `{c['comparison_id']}`"
            + (
                f" — requires: {'; '.join(need)}"
                if need
                else " — executable as declared"
            )
        )
    lines.append("")

    hist = by_class["historical_not_comparable"]
    lines.append(
        f"**Historical, not comparable ({len(hist)})** — the difference lives inside the artifacts; no fresh eval repairs it:"
    )
    for c in hist:
        conf = [
            x["name"]
            for x in c["remaining_confounds"]
            if x.get("severity") == "training"
        ]
        keys = _sched_confound_keys(c)
        lines.append(
            f"- `{c['comparison_id']}`: {', '.join(f'`{x}`' for x in conf)}"
            + (f" (schedule keys: {', '.join(keys)})" if keys else "")
        )
    lines.append("")

    blk = by_class["blocked"]
    lines.append(
        f"**Blocked ({len(blk)})** — the declared question has no valid reading over these artifacts:"
    )
    for c in blk:
        lines.append(f"- `{c['comparison_id']}`: {'; '.join(c['blocking_confounds'])}")
    lines.append("")

    lines += ["## Known issues, as they appear in the rows", ""]
    mm = {
        sev: [
            c["comparison_id"]
            for c in rows
            for x in c["remaining_confounds"]
            if x["name"] == "deployed_backbone_teacher_mismatch"
            and x.get("severity") == sev
        ]
        for sev in ("common_mode", "asymmetric", "blocking")
    }
    lines.append(
        "1. **s production backbone ≠ head's teacher (sibling-ONNX evidence; engine provenance unresolved).** "
        f"Common-mode in {len(mm['common_mode'])} rows (cannot explain their paired delta, but their absolute numbers sit on an unintended backbone); "
        f"**blocking** in {', '.join(f'`{i}`' for i in mm['blocking']) or 'none'} (the two heads were trained against different teachers, so no single engine is consistent for both); "
        f"tolerated as asymmetric in {', '.join(f'`{i}`' for i in mm['asymmetric']) or 'none'} (declared system comparisons). "
        "Resolution path: `E4` (same head, two engines) and, for a symmetric legacy-vs-T3→T1 reading, `F4`."
    )
    gt1 = [
        c["comparison_id"]
        for c in rows
        if "GT1-start confound" in c["axes"]["warm_start"].get("detail", "")
    ]
    warm = [
        c["comparison_id"]
        for c in rows
        if c["design"] != "system" and "warmup_epochs" in _sched_confound_keys(c)
    ]
    lines.append(
        "2. **Old seed-chain T3→T1 pairs are not the shared-GT1 matched pairs.** "
        f"Mixing them is blocked outright ({', '.join(f'`{i}`' for i in gt1) or 'none'}: different warm-start checkpoints). "
        f"The old pairs themselves, and every plain-GT2 vs T3→T1 pairing, carry a `warmup_epochs` 5→3 difference outside the curriculum ({', '.join(f'`{i}`' for i in warm)}); "
        "the paper table's `paired=True` rows are therefore historical here."
    )
    retrain = [
        c["comparison_id"] for c in rows if not c["executability"]["retrain_replay"]
    ]
    caches = sorted(
        {
            m.split(" needs ")[1].split(" ")[0]
            for c in rows
            for m in c["executability"]["retrain_replay_missing"]
            if " needs " in m
        }
    )
    lines.append(
        f"3. **Missing teacher caches / legacy distill artifact.** {len(retrain)} of {len(rows)} rows cannot be re-trained from cache ({', '.join(f'`{x}`' for x in caches)} unavailable; legacy chain incomplete). "
        "Fresh *evaluation* of the existing checkpoints is unaffected; any comparison that needs a new training arm (e.g. a matched m implicit control, a proper cache-decode A/B) needs a cache rebuild first, and a rebuilt cache is a new node, not the old one."
    )
    with_hist = [
        c
        for c in rows
        if c["historical_results"]["lhs"] or c["historical_results"]["rhs"]
    ]
    lines.append(
        f"4. **Historical eval runs carry no preset / engine identity.** {len(with_hist)} rows have endpoints with results on record; none is reusable as a paired measurement. "
        "The results table has no preset/engine sha column and the run-manifest schema records a preset *name* only. Runtime identity for new measurements is deliverable 3; whether each historical run used the engine the preset names today is deliverable 4."
    )
    lines.append("")
    return lines


def _short(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (dict, list)):
        s = json.dumps(v, sort_keys=True)
        return s if len(s) <= 90 else s[:87] + "…"
    return str(v)


def render_markdown(payload: dict[str, Any]) -> str:
    cap = payload["inputs"]["inventory"]["captured"] or {}
    lines: list[str] = []
    lines += [
        "<!-- doc-status: active -->",
        "<!-- doc-promotion: none -->",
        "<!-- doc-date: 2026-09-18 -->",
        "<!-- doc-module: detection -->",
        "<!-- Generated by scripts/provenance/training_comparison.py from report_data/training_lineage_inventory.json; regenerate rather than edit. -->",
        "",
        "# Training comparison matrix (#421 · deliverable 2)",
        "",
        f"Derived from the lineage inventory captured {cap.get('at_utc')} at `{(cap.get('git_head') or '')[:12]}` "
        f"(`{payload['inputs']['inventory']['path']}`) and the declared candidates in `scripts/provenance/training_comparisons.json`. "
        "Machine-readable twin: `report_data/training_comparison_matrix.json`.",
        "",
        "This matrix answers **which comparisons hold up methodologically**. It produces no IDF1/HOTA/FPS; the shared eval contract and frozen baseline runs are deliverables 3–4.",
        "",
        "## How to read",
        "",
        "- Every row compares two lineage **nodes** (ids from the inventory) under a declared **runtime binding** (`family_preset` = the artifacts the family's headline preset names; `shared_backbone_node` = both heads on one PyTorch backbone, eager).",
        "- **Structural axes** ("
        + ", ".join(f"`{a}`" for a in payload["axes"]["structural"])
        + ") say *what system* is measured; **training axes** ("
        + ", ".join(f"`{a}`" for a in payload["axes"]["training"])
        + ") say *how the head got its weights*.",
        "- Axis status: `matched` · `matched_effective` (artifact differs, effective forward does not — e.g. bypassed temporal blocks) · `unmatched` · `unknown` (the bytes do not establish it; fail-closed).",
        "- **Classification is derived, not declared.** The declaration names the design and the treatment axes; the rule is:",
    ]
    for r in payload["classification_rule"]:
        lines.append(f"  - {r}")
    lines += [
        "- `remaining_confounds` lists every non-treatment axis that is unmatched or unknown; `common_mode` marks a confound both endpoints share (it cannot explain a paired delta).",
        "- **Historical results are never reusable as a paired measurement here**: the record (`report_data/tables/mamba_tracking_overall.csv`) carries no preset or engine sha, and the referenced `results/` directories carry no `run_manifest.json` (workspace probe below).",
        "",
        "## Summary",
        "",
        "| # | comparison_id | design | lhs | rhs | classification | treatment observed | non-treatment confounds | blocking |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(payload["comparisons"], 1):
        struct = [
            x["name"]
            for x in c["remaining_confounds"]
            if x.get("severity") == "structural"
        ]
        train = [
            x["name"]
            for x in c["remaining_confounds"]
            if x.get("severity") == "training"
        ]
        cm = [
            x["name"]
            for x in c["remaining_confounds"]
            if x.get("severity") in ("common_mode", "asymmetric")
        ]
        parts = []
        if struct:
            parts.append("structural: " + ", ".join(f"`{x}`" for x in struct))
        if train:
            parts.append("training: " + ", ".join(f"`{x}`" for x in train))
        conf_cell = "; ".join(parts) or "none"
        if cm:
            conf_cell += " (+shared: " + ", ".join(f"`{x}`" for x in cm) + ")"
        lines.append(
            f"| {i} | `{c['comparison_id']}` | {c['design']} | `{c['lhs']['node']}` | `{c['rhs']['node']}` | **{c['classification']}** | "
            f"{', '.join(f'`{a}`' for a in c['treatment_axes_observed']) or '—'} | {conf_cell} | {'; '.join(c['blocking_confounds']) or '—'} |"
        )
    counts = payload["counts"]
    lines += [
        "",
        "Counts: " + ", ".join(f"{k} = {v}" for k, v in counts.items()) + ".",
        "",
    ]

    lines += findings_section(payload)

    lines += ["## Comparisons", ""]
    for c in payload["comparisons"]:
        lines += [f"### `{c['comparison_id']}` — **{c['classification']}**", ""]
        lines.append(
            f"- design `{c['design']}`; lhs `{c['lhs']['node']}` ({c['lhs']['runtime'].get('binding')}), rhs `{c['rhs']['node']}` ({c['rhs']['runtime'].get('binding')})"
        )
        lines.append(f"- intended treatment: {c['intended_treatment']}")
        if c.get("question"):
            lines.append(f"- question: {c['question']}")
        lines.append(
            f"- treatment axes declared: {', '.join(f'`{a}`' for a in c['treatment_axes']) or 'none (system)'}"
            + (
                f"; schedule keys: {', '.join(c['treatment_schedule_keys'])}"
                if c["treatment_schedule_keys"]
                else ""
            )
        )
        lines += ["", "| axis | status | lhs | rhs | detail |", "|---|---|---|---|---|"]
        for a in payload["axes"]["structural"] + payload["axes"]["training"]:
            ax = c["axes"][a]
            tag = " (treatment)" if a in c["treatment_axes"] else ""
            detail = ax.get("detail", "")
            if a == "training_schedule" and ax["status"] != "unknown":
                try:
                    sd = json.loads(detail)
                    parts = []
                    if sd["treatment_diff"]:
                        parts.append(
                            "treatment: "
                            + json.dumps(sd["treatment_diff"], sort_keys=True)
                        )
                    if sd["confound_diff"]:
                        parts.append(
                            "**confound: "
                            + json.dumps(sd["confound_diff"], sort_keys=True)
                            + "**"
                        )
                    if sd["one_sided_keys"]:
                        parts.append(
                            "one-sided keys (not counted): "
                            + ", ".join(sd["one_sided_keys"])
                        )
                    detail = (
                        "; ".join(parts) or "identical recipe over the compared stages"
                    )
                except (ValueError, KeyError):
                    pass
            lines.append(
                f"| `{a}`{tag} | {ax['status']} | {_short(ax['lhs'])} | {_short(ax['rhs'])} | {detail or '—'} |"
            )
        lines.append("")
        if c["remaining_confounds"]:
            lines.append("Remaining confounds:")
            for x in c["remaining_confounds"]:
                d_ = (
                    x.get("detail")
                    if isinstance(x.get("detail"), str)
                    else _short(x.get("detail"))
                )
                lines.append(
                    f"- `{x['name']}` [{x.get('severity')}]: {d_ or _short(x.get('lhs')) + ' vs ' + _short(x.get('rhs'))}"
                )
        if c["blocking_confounds"]:
            lines.append("Blocking:")
            for b in c["blocking_confounds"]:
                lines.append(f"- {b}")
        ex = c["executability"]
        lines.append(
            f"- executability: fresh paired eval {'possible' if ex['fresh_eval'] else 'NOT possible (' + ', '.join(ex['fresh_eval_missing']) + ')'}; "
            f"retrain replay {'possible' if ex['retrain_replay'] else 'NOT possible (' + '; '.join(ex['retrain_replay_missing']) + ')'}"
        )
        hl, hr = c["historical_results"]["lhs"], c["historical_results"]["rhs"]
        if hl or hr:
            lines.append(
                f"- historical results on record: lhs {[h['experiment'] for h in hl] or 'none'}, rhs {[h['experiment'] for h in hr] or 'none'} — not reusable as paired ({c['historical_results']['reason']})"
            )
        for n in c["declared_notes"]:
            lines.append(f"- note: {n}")
        lines.append("")

    if payload.get("not_representable"):
        lines += ["## Candidates not representable with the current inventory", ""]
        for n in payload["not_representable"]:
            lines.append(f"- **{n['candidate']}** — {n['reason']}")
        lines.append("")

    probe = payload.get("workspace_probe")
    lines += ["## Workspace probe — historical eval provenance", ""]
    if probe:
        lines.append(
            f"{probe['with_manifest']} of {probe['total']} `results/` directories referenced by the results table carry a `run_manifest.json` (probed on the capture host; captured fact, not a derived one). "
            "The manifest schema itself records a preset *name*, not preset or engine sha, so even a manifest would not bind runtime identity."
        )
    else:
        lines.append("Probe disabled (`--no-probe`).")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def _strip_probe(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["workspace_probe"] = None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--results-table", type=Path, default=DEFAULT_RESULTS_TABLE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument(
        "--no-probe", action="store_true", help="skip the results-dir manifest probe"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed outputs are stale (probe ignored)",
    )
    args = parser.parse_args(argv)

    repo = REPO_ROOT
    inv_path = args.inventory if args.inventory.is_absolute() else repo / args.inventory
    inventory = load_inventory(inv_path)
    declarations = load_declarations(args.declarations, inventory)
    rows = load_results_table(
        args.results_table
        if args.results_table.is_absolute()
        else repo / args.results_table
    )
    probe = None if args.no_probe else probe_result_manifests(rows, repo)
    payload = build_matrix(inventory, declarations, rows, repo, probe=probe)

    json_out = args.json_out if args.json_out.is_absolute() else repo / args.json_out
    md_out = args.md_out if args.md_out.is_absolute() else repo / args.md_out
    if args.check:
        stale = []
        if not json_out.exists():
            stale.append(str(json_out))
        else:
            current = json.loads(json_out.read_text(encoding="utf-8"))
            if _strip_probe(current) != _strip_probe(payload):
                stale.append(str(json_out))
        if not md_out.exists():
            stale.append(str(md_out))
        else:
            committed = md_out.read_text(encoding="utf-8")
            fresh = render_markdown(
                {
                    **payload,
                    "workspace_probe": current.get("workspace_probe")
                    if json_out.exists()
                    else None,
                }
            )
            if committed != fresh:
                stale.append(str(md_out))
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("training comparison matrix up to date")
        return 0

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"wrote {json_out} and {md_out}: "
        + ", ".join(f"{k}={v}" for k, v in payload["counts"].items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
