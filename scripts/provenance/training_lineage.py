"""Model-lineage inventory for the YOLO + Mamba detector chain (issue #421, deliverable 1).

Answers, from bytes on this machine, which training artifacts exist, what each
one says about itself, and how they are connected — so that #421's comparison
matrix and every downstream packet (#422–#425, #419) reference stable node ids
instead of run-directory names remembered from protocol prose.

The inputs are two, and they are kept apart on purpose:

* ``training_lineage_roles.json`` declares **candidate roles → paths**.  That is
  the protocol's story ("this run is GT1", "this is the production checkpoint")
  and it is treated as a claim.
* The **artifacts themselves** are the evidence.  A checkpoint written by
  ``train_mamba_head.py`` / ``train_mamba_gt.py`` stores its full ``args``
  (``yolo_weights``, ``teacher_ckpt``, ``mamba_ckpt`` warm start, ``cache_dir``,
  seed, clip length, ``gt_ratio``, …) and ``mamba_args`` (architecture, and —
  for checkpoints written after 2026-06-13 — ``base_yolo_sha256`` /
  ``teacher_checkpoint_sha256``).  Parent edges are derived from **those**
  fields, never from the role file, and the role file's ``expected_*`` claims
  are then reported as ``ok`` / ``mismatch`` / ``unverifiable`` against them.

Three checks go beyond metadata because metadata cannot answer them:

* **Tensor delta along a warm-start edge.**  ``scan_stop_grad`` in ``mamba_args``
  is a flag; whether the SSM interior actually stayed at its parent's values is
  a raw-bytes comparison (same dtype, shape and bytes — not value equality).
  Every warm-start edge whose parent is in the inventory gets a per-module-group
  count of identical / changed / added / removed tensors.
* **Sibling-ONNX → checkpoint match.**  A TensorRT ``.engine`` is opaque and is
  **never** attributed by this tool.  What can be checked is the ``.onnx`` the
  role file associates with it: its initializers carry weights, backbone exports
  are BN-folded, so the tool folds each candidate teacher's conv+BN pairs and
  looks for exact float32 equality.  The result names the unique matching
  checkpoint for the *ONNX*, or says the match is partial / ambiguous.  The
  engine ↔ ONNX association is a role-file claim with no build manifest or
  cryptographic binding behind it; the tool records whether the two file stems
  even agree and carries the "engine bytes unattributed" caveat into every
  deployment statement derived from it.
* **Deployment forward mode** is read off the preset plus the checkpoint it
  names (temporal blocks present? whole-graph ⇒ bypassed; final-stage
  ``gt_ratio``; ``reid_mode``; backbone source), and the production checkpoint
  is de-duplicated against the lineage node with the same sha256.

The tool does not train, evaluate, or infer anything a file does not state.  A
missing path stays in the table as ``unavailable`` and is never replaced by a
look-alike; "trainable" claims are limited to what the tensor delta shows.

The rendered inventory is a **captured snapshot** of one workspace (host, HEAD,
timestamp recorded).  ``runs/`` and ``models/`` are gitignored, so a clean clone
cannot regenerate it; it is committed as evidence like ``report_data/*.json``,
not as a freshness-checked generated view.

Usage:
    .venv/bin/python scripts/provenance/training_lineage.py            # writes the defaults
    .venv/bin/python scripts/provenance/training_lineage.py --no-tensor-diff --no-onnx-match --no-cache-content-hash
"""

# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
DEFAULT_ROLES = _HERE.parent / "training_lineage_roles.json"
DEFAULT_JSON_OUT = Path("report_data/training_lineage_inventory.json")
DEFAULT_MD_OUT = Path("docs/research/training/training_lineage_inventory.md")

SCHEMA = "training_lineage_inventory_v1"
ROLES_SCHEMA = "training_lineage_roles_v1"

KINDS = frozenset(
    {
        "yolo_pt",
        "gated_teacher_ckpt",
        "mamba_ckpt",
        "teacher_cache",
        "preset",
        "trt_engine",
        "torchscript",
    }
)

# Checkpoint ``args`` fields that name another artifact.  Each becomes an edge.
# ``mamba_ckpt`` is the warm start for GT stages; distill has none.
EDGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("mamba_ckpt", "init"),
    ("resume", "resume"),
    ("teacher_ckpt", "teacher"),
    ("yolo_weights", "base_yolo"),
    ("cache_dir", "cache"),
)

# ``args`` keys that are paths / bookkeeping, excluded from the treatment delta
# so the delta lists what the *training treatment* changed between parent and child.
NON_TREATMENT_ARGS = frozenset(
    {
        "run_dir",
        "resume",
        "mamba_ckpt",
        "teacher_ckpt",
        "yolo_weights",
        "cache_dir",
        "data_root",
        "num_workers",
        "dry_run",
        "save_every",
        "compile",
        "print_only",
    }
)

# Tensor-name → module group.  SSM interior is what the frozen-SSM regime claims
# to hold fixed (protocol: A_log / D / conv1d / x_proj / dt_proj).
_SSM_INTERNAL = re.compile(
    r"\.(A_log|D|conv1d\.(weight|bias)|x_proj\.weight|dt_proj\.(weight|bias))$"
)


class LineageError(RuntimeError):
    """The role file is malformed, or an artifact cannot be read as its declared kind."""


# --------------------------------------------------------------------------- roles


def load_roles(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ROLES_SCHEMA:
        raise LineageError(
            f"{path}: schema must be {ROLES_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    roles = payload.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise LineageError(f"{path}: 'roles' must be a non-empty object")
    families = payload.get("families") or {}
    seen_paths: dict[str, str] = {}
    for rid, spec in roles.items():
        for key in ("family", "kind", "path"):
            if key not in spec:
                raise LineageError(f"{path}: role {rid!r} lacks {key!r}")
        if spec["kind"] not in KINDS:
            raise LineageError(
                f"{path}: role {rid!r} has unknown kind {spec['kind']!r}"
            )
        if spec["family"] not in families:
            raise LineageError(
                f"{path}: role {rid!r} names undeclared family {spec['family']!r}"
            )
        if not rid.startswith(spec["family"] + "."):
            raise LineageError(
                f"{path}: role id {rid!r} must be prefixed by its family"
            )
        if spec["path"] in seen_paths:
            raise LineageError(
                f"{path}: roles {seen_paths[spec['path']]!r} and {rid!r} declare the same path"
            )
        seen_paths[spec["path"]] = rid
        for key in ("expected_init_parent", "expected_teacher"):
            target = spec.get(key)
            if target is not None and target not in roles:
                raise LineageError(
                    f"{path}: role {rid!r} {key} -> undeclared role {target!r}"
                )
    return payload


# --------------------------------------------------------------------------- files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


# --------------------------------------------------------------------------- checkpoints


def _plain(value: Any) -> Any:
    """Make argparse namespaces / dataclasses / tuples JSON-friendly."""
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        value = vars(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch  # local: keep --help and role validation torch-free

    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_of(ckpt: dict[str, Any], kind: str) -> dict[str, Any]:
    key = "student" if kind == "mamba_ckpt" else "model"
    sd = ckpt.get(key)
    if not isinstance(sd, dict):
        raise LineageError(
            f"checkpoint has no {key!r} state dict (keys: {sorted(ckpt)})"
        )
    return sd


def checkpoint_summary(ckpt: dict[str, Any], kind: str) -> dict[str, Any]:
    """Everything a checkpoint states about itself, minus the tensors."""
    sd = state_dict_of(ckpt, kind)
    args = _plain(ckpt.get("args") or {})
    out: dict[str, Any] = {
        "epoch": ckpt.get("epoch"),
        "best_loss": ckpt.get("best_loss"),
        "selection": _plain(ckpt.get("selection")),
        "provenance": _plain(ckpt.get("provenance")),
        "args": args,
        "tensor_count": len(sd),
        "param_count": int(sum(int(t.numel()) for t in sd.values())),
        "top_level_keys": sorted(ckpt.keys()),
    }
    if kind == "mamba_ckpt":
        out["mamba_args"] = _plain(ckpt.get("mamba_args") or {})
        out["module_groups"] = sorted({tensor_group(k) for k in sd})
    else:
        out["cfg"] = _plain(ckpt.get("cfg"))
    # Fields the comparison matrix keys on, pulled up for readability.
    treatment_keys = (
        "seed",
        "epochs",
        "lr",
        "lr_gate",
        "lr_yolo",
        "batch_size",
        "img_size",
        "clip_len",
        "clip_stride",
        "gt_ratio",
        "add_temporal",
        "scan_stop_grad",
        "seqs",
        "holdout_seqs",
        "consistency_weight",
        "protocol_revision",
    )
    out["treatment"] = {k: args[k] for k in treatment_keys if k in args}
    return out


def tensor_group(name: str) -> str:
    head = name.split(".", 1)[0]
    if head in ("mamba_blocks", "temporal_blocks"):
        return f"{head}.ssm_internal" if _SSM_INTERNAL.search(name) else f"{head}.proj"
    return head


def tensors_bit_identical(a: Any, b: Any) -> bool:
    """Same dtype, same shape, same raw bytes.

    Not value equality: ``+0.0`` vs ``-0.0`` differ, a float16 copy of a float32
    tensor differs, and NaN payloads are compared as bytes.  This is the
    predicate "bit-identical" in the rendered tables refers to.
    """
    import torch

    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.numel() == 0:
        return True
    a_bytes = a.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    b_bytes = b.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(a_bytes, b_bytes))


def tensor_delta(child: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    """Per-group identical / changed / added / removed counts between two state dicts.

    ``identical`` means :func:`tensors_bit_identical` (dtype + shape + raw bytes).
    """

    groups: dict[str, dict[str, int]] = {}

    def bump(group: str, key: str) -> None:
        groups.setdefault(
            group, {"identical": 0, "changed": 0, "added": 0, "removed": 0}
        )[key] += 1

    for name, tensor in child.items():
        group = tensor_group(name)
        if name not in parent:
            bump(group, "added")
            continue
        bump(
            group,
            "identical" if tensors_bit_identical(tensor, parent[name]) else "changed",
        )
    for name in parent:
        if name not in child:
            bump(tensor_group(name), "removed")
    totals = {
        k: sum(g[k] for g in groups.values())
        for k in ("identical", "changed", "added", "removed")
    }
    return {"totals": totals, "groups": dict(sorted(groups.items()))}


def treatment_delta(
    child_args: dict[str, Any], parent_args: dict[str, Any]
) -> dict[str, Any]:
    """Which non-path args differ between a child and its warm-start parent.

    ``changed`` holds keys both checkpoints recorded with different values — the
    treatment.  Keys recorded by only one side are listed by name only: the
    training scripts grew new flags between runs, and a key that is absent on
    the parent says nothing about what the parent's script did.
    """
    changed: dict[str, Any] = {}
    only_child: list[str] = []
    only_parent: list[str] = []
    for key in sorted(set(child_args) | set(parent_args)):
        if key in NON_TREATMENT_ARGS:
            continue
        if key not in parent_args:
            only_child.append(key)
        elif key not in child_args:
            only_parent.append(key)
        elif parent_args[key] != child_args[key]:
            changed[key] = {"parent": parent_args[key], "child": child_args[key]}
    return {"changed": changed, "only_child": only_child, "only_parent": only_parent}


# --------------------------------------------------------------------------- onnx ↔ ckpt


def fold_bn_convs(sd: dict[str, Any], eps: float = 1e-3) -> dict[str, Any]:
    """BN-fold every ``<m>.conv.weight`` with its ``<m>.bn.*`` the way the YOLO export does.

    Produces the fused conv weight *and* the fused bias (``beta - mean * scale``),
    since the exported graph carries both.  ``eps`` is Ultralytics' BatchNorm2d
    default (1e-3); exact equality downstream is the proof that the fold matches,
    so a wrong eps produces ``none``, not a near miss.
    """
    import torch

    out: dict[str, Any] = {}
    for name, weight in sd.items():
        if not name.endswith(".conv.weight") or weight.ndim != 4:
            continue
        base = name[: -len(".conv.weight")]
        gamma, beta = sd.get(f"{base}.bn.weight"), sd.get(f"{base}.bn.bias")
        mean, var = sd.get(f"{base}.bn.running_mean"), sd.get(f"{base}.bn.running_var")
        if gamma is None or beta is None or mean is None or var is None:
            out[name] = weight
            continue
        scale = gamma / torch.sqrt(var + eps)
        out[name] = weight * scale.reshape(-1, 1, 1, 1)
        out[f"{base}.fused_bias"] = beta - mean * scale
    return out


def onnx_initializers(path: Path, min_elements: int = 64) -> list[Any]:
    import onnx

    model = onnx.load(str(path))
    arrays = [onnx.numpy_helper.to_array(init) for init in model.graph.initializer]
    return [a for a in arrays if a.size >= min_elements]


def match_onnx(
    initializers: list[Any], candidates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Count ONNX initializers found by exact float32 value equality in each candidate.

    ONNX initializers are float32; candidate tensors are cast to float32 before the
    comparison, so this is exact value equality in float32, not raw-byte identity.

    2-D initializers are also tried transposed (Gemm stores ``W^T``).  Anything the
    exporter reshaped or fused beyond that is simply not matched, which is why a
    verdict can be ``partial`` for a checkpoint that really is the source.
    """
    import numpy as np

    counts: dict[str, int] = {}
    for node_id, tensors in candidates.items():
        by_shape: dict[tuple[int, ...], list[Any]] = {}
        for tensor in tensors.values():
            arr = tensor.detach().to("cpu").float().numpy()
            by_shape.setdefault(tuple(arr.shape), []).append(arr)

        def found(arr: Any) -> bool:
            return any(
                np.array_equal(arr, c) for c in by_shape.get(tuple(arr.shape), [])
            )

        hit = 0
        for arr in initializers:
            arr32 = arr.astype(np.float32)
            if found(arr32) or (arr32.ndim == 2 and found(arr32.T)):
                hit += 1
        counts[node_id] = hit
    total = len(initializers)
    best = max(counts.values(), default=0)
    winners = sorted(n for n, c in counts.items() if c == best and best > 0)
    if total == 0 or best == 0:
        verdict = "none"
    elif len(winners) > 1:
        verdict = "ambiguous"
    elif best == total:
        verdict = "unique_exact"
    else:
        verdict = "partial"
    return {
        "total_initializers": total,
        "matches": counts,
        "best": winners,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- presets / caches


def read_preset(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise LineageError(f"{path}: preset is not a mapping")
    keys = (
        "mamba_ckpt",
        "mamba_teacher_ckpt",
        "mamba_yolo_weights",
        "fpn_backbone_engine",
        "mamba_head_engine",
        "use_whole_graph",
        "use_cuda_graph",
        "use_tracker_graph",
        "reid_mode",
        "tiling",
        "preprocess",
        "gmc",
    )
    return {k: data.get(k) for k in keys if k in data}


def cache_content_digest(path: Path) -> dict[str, Any]:
    """Identity of a cache directory's contents: sha256 over the sorted per-file listing.

    Each line is ``<relative path>\t<size>\t<sha256>``; the digest is the sha256 of
    those lines joined by newlines.  Two caches with the same digest hold the same
    bytes under the same names; mtimes are deliberately not part of it.
    """
    files = sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    total = 0
    for file in files:
        size = file.stat().st_size
        total += size
        digest.update(
            f"{file.relative_to(path).as_posix()}\t{size}\t{sha256_file(file)}\n".encode()
        )
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


def read_cache_dir(path: Path, *, hash_contents: bool) -> dict[str, Any]:
    manifest = path / "manifest.json"
    out: dict[str, Any] = {"has_manifest": manifest.exists()}
    if manifest.exists():
        out["manifest_sha256"] = sha256_file(manifest)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        out["manifest"] = {
            k: data.get(k)
            for k in (
                "schema",
                "status",
                "base_yolo_path",
                "base_yolo_sha256",
                "teacher_checkpoint_path",
                "teacher_checkpoint_sha256",
                "decode_backend",
                "dtype",
                "img_size",
                "resize_mode",
                "sequences",
                "total_frames",
            )
        }
    out["sequence_dirs"] = (
        sorted(p.name for p in path.iterdir() if p.is_dir()) if path.is_dir() else []
    )
    out["content"] = (
        cache_content_digest(path)
        if hash_contents
        else "not_hashed (--no-cache-content-hash)"
    )
    return out


# --------------------------------------------------------------------------- inventory


@dataclass
class Edge:
    relation: str  # init | resume | teacher | base_yolo | cache | names
    target_path: str
    target_node: str | None
    target_exists: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    id: str
    family: str
    kind: str
    path: str
    note: str
    exists: bool
    sha256: str | None = None
    size_bytes: int | None = None
    mtime_utc: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def unavailable(self) -> bool:
        return not self.exists


def _norm(path_str: str) -> str:
    return os.path.normpath(path_str)


class Inventory:
    def __init__(
        self,
        roles: dict[str, Any],
        *,
        tensor_diff: bool,
        onnx_match: bool,
        cache_content_hash: bool = True,
        roles_path: Path | None = None,
    ) -> None:
        self.roles = roles
        self.roles_path = roles_path
        self.tensor_diff = tensor_diff
        self.onnx_match = onnx_match
        self.cache_content_hash = cache_content_hash
        self.nodes: dict[str, Node] = {}
        self.by_path: dict[str, str] = {}
        self._ckpt_cache: dict[str, dict[str, Any]] = {}

    # ---- loading

    def _ckpt(self, node: Node) -> dict[str, Any]:
        if node.id not in self._ckpt_cache:
            self._ckpt_cache[node.id] = load_checkpoint(REPO_ROOT / node.path)
        return self._ckpt_cache[node.id]

    def build(self) -> None:
        for rid, spec in self.roles["roles"].items():
            path = REPO_ROOT / spec["path"]
            node = Node(
                id=rid,
                family=spec["family"],
                kind=spec["kind"],
                path=spec["path"],
                note=spec.get("note", ""),
                exists=path.exists(),
            )
            self.nodes[rid] = node
            self.by_path[_norm(spec["path"])] = rid
            if not node.exists:
                continue
            if path.is_dir():
                node.mtime_utc = _mtime_iso(path)
            else:
                node.sha256 = sha256_file(path)
                node.size_bytes = path.stat().st_size
                node.mtime_utc = _mtime_iso(path)
            try:
                self._describe(node, spec)
            except LineageError as exc:
                node.problems.append(str(exc))
        for node in self.nodes.values():
            if node.exists and node.kind in ("mamba_ckpt", "gated_teacher_ckpt"):
                self._check_expectations(node, self.roles["roles"][node.id])
        for node in self.nodes.values():
            if node.exists and node.kind == "mamba_ckpt" and self.tensor_diff:
                self._diff_against_parent(node)
        for node in self.nodes.values():
            if node.exists and node.kind == "trt_engine" and self.onnx_match:
                self._attribute_engine(node, self.roles["roles"][node.id])
        for node in self.nodes.values():
            if node.exists and node.kind == "preset":
                self._deployment(node)
        self._cross_reference_csv()
        self._ckpt_cache.clear()

    def _cross_reference_csv(self) -> None:
        """Agree or disagree with the paper-facing checkpoint table on shared paths."""
        import csv

        csv_rel = "report_data/tables/mamba_checkpoint_provenance.csv"
        csv_path = REPO_ROOT / csv_rel
        if not csv_path.exists():
            return
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = {
                _norm(r["checkpoint"]): r
                for r in csv.DictReader(handle)
                if r.get("checkpoint")
            }
        for node in self.nodes.values():
            row = rows.get(_norm(node.path))
            if row is None:
                continue
            if not node.exists or not node.sha256:
                status = "listed_but_unavailable_here"
            else:
                status = (
                    "sha256_agrees"
                    if row.get("sha256") == node.sha256
                    else "SHA256_DISAGREES"
                )
            node.checks["paper_table"] = {
                "source": csv_rel,
                "experiment": row.get("experiment"),
                "status": status,
            }

    def _describe(self, node: Node, spec: dict[str, Any]) -> None:
        path = REPO_ROOT / node.path
        if node.kind in ("mamba_ckpt", "gated_teacher_ckpt"):
            ckpt = self._ckpt(node)
            node.summary = checkpoint_summary(ckpt, node.kind)
            args = node.summary["args"]
            for key, relation in EDGE_FIELDS:
                target = args.get(key)
                if not target:
                    continue
                if relation == "init" and _norm(str(target)) == _norm(node.path):
                    # A resumed run stores its own best.ckpt as mamba_ckpt.
                    relation = "self_resume"
                node.edges.append(self._edge(relation, str(target)))
            self._verify_self_attested_shas(node)
        elif node.kind == "teacher_cache":
            node.summary = read_cache_dir(path, hash_contents=self.cache_content_hash)
            manifest = node.summary.get("manifest") or {}
            for key, relation in (
                ("teacher_checkpoint_path", "teacher"),
                ("base_yolo_path", "base_yolo"),
            ):
                if manifest.get(key):
                    node.edges.append(self._edge(relation, manifest[key]))
        elif node.kind == "preset":
            node.summary = read_preset(path)
            for key in (
                "mamba_ckpt",
                "mamba_teacher_ckpt",
                "mamba_yolo_weights",
                "fpn_backbone_engine",
                "mamba_head_engine",
            ):
                target = node.summary.get(key)
                if target:
                    node.edges.append(
                        self._edge("names", str(target), detail={"preset_key": key})
                    )
        elif node.kind == "trt_engine":
            onnx_rel = spec.get("onnx")
            if onnx_rel:
                onnx_path = REPO_ROOT / onnx_rel
                node.summary["onnx"] = {
                    "path": onnx_rel,
                    "exists": onnx_path.exists(),
                    "sha256": sha256_file(onnx_path) if onnx_path.exists() else None,
                    "mtime_utc": _mtime_iso(onnx_path) if onnx_path.exists() else None,
                    "engine_link": {
                        "basis": "role-file claim; no build manifest, no cryptographic binding",
                        "stem_equal": Path(onnx_rel).stem == path.stem,
                        "engine_bytes_attributed": False,
                    },
                }
        # yolo_pt / torchscript: blob identity only, by design (no unpickling).

    def _edge(
        self, relation: str, target: str, detail: dict[str, Any] | None = None
    ) -> Edge:
        norm = _norm(target)
        return Edge(
            relation=relation,
            target_path=target,
            target_node=self.by_path.get(norm),
            target_exists=(REPO_ROOT / target).exists(),
            detail=detail or {},
        )

    def _verify_self_attested_shas(self, node: Node) -> None:
        """Checkpoints after 2026-06-13 record the sha256 of their base YOLO and teacher."""
        mamba_args = node.summary.get("mamba_args") or {}
        args = node.summary.get("args") or {}
        for sha_key, path_key, label in (
            ("base_yolo_sha256", "yolo_weights", "base_yolo"),
            ("teacher_checkpoint_sha256", "teacher_ckpt", "teacher"),
        ):
            expected = mamba_args.get(sha_key)
            if not expected:
                node.checks[f"{label}_sha_attested"] = "not_recorded"
                continue
            target = args.get(path_key)
            target_path = REPO_ROOT / str(target) if target else None
            if target_path is None or not target_path.exists():
                node.checks[f"{label}_sha_attested"] = "recorded_but_target_missing"
                continue
            actual = self._sha_of_path(str(target))
            node.checks[f"{label}_sha_attested"] = (
                "verified" if actual == expected else "MISMATCH"
            )

    def _sha_of_path(self, rel: str) -> str:
        nid = self.by_path.get(_norm(rel))
        if nid and self.nodes[nid].sha256:
            return self.nodes[nid].sha256  # type: ignore[return-value]
        return sha256_file(REPO_ROOT / rel)

    # ---- checks

    def _check_expectations(self, node: Node, spec: dict[str, Any]) -> None:
        for key, relation in (
            ("expected_init_parent", "init"),
            ("expected_teacher", "teacher"),
        ):
            expected = spec.get(key)
            if expected is None:
                continue
            actual = [e.target_node for e in node.edges if e.relation == relation]
            if not actual:
                node.checks[key] = {
                    "expected": expected,
                    "derived": None,
                    "status": "unverifiable",
                }
            elif expected in actual:
                node.checks[key] = {
                    "expected": expected,
                    "derived": actual,
                    "status": "ok",
                }
            else:
                derived_paths = [
                    e.target_path for e in node.edges if e.relation == relation
                ]
                node.checks[key] = {
                    "expected": expected,
                    "derived": actual or derived_paths,
                    "status": "mismatch",
                }

    def _diff_against_parent(self, node: Node) -> None:
        parents = [e for e in node.edges if e.relation == "init" and e.target_node]
        if not parents:
            node.checks["tensor_delta_vs_init_parent"] = "no_init_parent_in_inventory"
            return
        parent = self.nodes[parents[0].target_node]  # type: ignore[index]
        if not parent.exists:
            node.checks["tensor_delta_vs_init_parent"] = (
                f"parent {parent.id} unavailable"
            )
            return
        child_sd = state_dict_of(self._ckpt(node), "mamba_ckpt")
        parent_sd = state_dict_of(self._ckpt(parent), "mamba_ckpt")
        delta = tensor_delta(child_sd, parent_sd)
        delta["parent"] = parent.id
        ssm = {g: c for g, c in delta["groups"].items() if g.endswith(".ssm_internal")}
        delta["ssm_internal_frozen"] = bool(ssm) and all(
            c["changed"] == 0 for c in ssm.values()
        )
        delta["treatment_delta"] = treatment_delta(
            node.summary.get("args") or {}, parent.summary.get("args") or {}
        )
        node.checks["tensor_delta_vs_init_parent"] = delta

    def _attribute_engine(self, node: Node, spec: dict[str, Any]) -> None:
        onnx_info = node.summary.get("onnx") or {}
        if not onnx_info.get("exists"):
            node.checks["onnx_initializer_match"] = "no_onnx_sibling"
            return
        try:
            inits = onnx_initializers(REPO_ROOT / onnx_info["path"])
        except Exception as exc:  # onnx load failure is a finding, not a crash
            node.checks["onnx_initializer_match"] = f"onnx_unreadable: {exc}"
            return
        candidates: dict[str, dict[str, Any]] = {}
        for other in self.nodes.values():
            if not other.exists or other.kind not in (
                "gated_teacher_ckpt",
                "mamba_ckpt",
            ):
                continue
            sd = state_dict_of(self._ckpt(other), other.kind)
            candidates[other.id] = (
                fold_bn_convs(sd) if other.kind == "gated_teacher_ckpt" else sd
            )
        result = match_onnx(inits, candidates)
        result["method"] = (
            "exact float32 value equality of ONNX initializers (>=64 elements) against "
            "candidate tensors; teacher checkpoints BN-folded (eps 1e-3) before comparison"
        )
        result["scope"] = (
            "the ONNX file only; the .engine bytes are not attributed. The engine<->ONNX "
            "association is the role file's claim (see summary.onnx.engine_link)."
        )
        node.checks["onnx_initializer_match"] = result
        for winner in result["best"]:
            node.edges.append(
                Edge(
                    relation="onnx_matches_checkpoint",
                    target_path=self.nodes[winner].path,
                    target_node=winner,
                    target_exists=True,
                    detail={"verdict": result["verdict"], "onnx": onnx_info["path"]},
                )
            )

    def _deployment(self, node: Node) -> None:
        preset = node.summary
        dep: dict[str, Any] = {}
        ckpt_rel = preset.get("mamba_ckpt")
        ckpt_node = self.by_path.get(_norm(str(ckpt_rel))) if ckpt_rel else None
        dep["checkpoint"] = {"path": ckpt_rel, "node": ckpt_node}
        if ckpt_node and self.nodes[ckpt_node].exists:
            cn = self.nodes[ckpt_node]
            aliases = sorted(
                n.id
                for n in self.nodes.values()
                if n.sha256 and n.sha256 == cn.sha256 and n.id != cn.id
            )
            dep["checkpoint"]["sha256"] = cn.sha256
            dep["checkpoint"]["aliases_same_sha256"] = aliases
            margs = cn.summary.get("mamba_args") or {}
            treatment = cn.summary.get("treatment") or {}
            has_temporal = bool(margs.get("use_temporal_mamba"))
            if not has_temporal:
                dep["temporal_blocks"] = "absent in checkpoint"
            elif preset.get("use_whole_graph"):
                dep["temporal_blocks"] = (
                    "present; BYPASSED under whole-graph single-frame forward (effective T=1)"
                )
            else:
                dep["temporal_blocks"] = (
                    "present; ACTIVE unless eval passes --no-temporal (train/eval mismatch risk)"
                )
            dep["final_stage_gt_ratio"] = treatment.get("gt_ratio")
            dep["scan_stop_grad_flag"] = margs.get("scan_stop_grad")
        elif ckpt_rel:
            dep["checkpoint"]["status"] = "not in inventory or unavailable"
        dep["gate_teacher_at_runtime"] = (
            preset.get("mamba_teacher_ckpt")
            or "null (backbone-only; no teacher forward)"
        )
        dep["backbone_source"] = (
            f"TRT engine {preset['fpn_backbone_engine']}"
            if preset.get("fpn_backbone_engine")
            else "PyTorch backbone from mamba_teacher_ckpt"
        )
        engine_node = self.by_path.get(
            _norm(str(preset.get("fpn_backbone_engine") or ""))
        )
        if engine_node:
            match = self.nodes[engine_node].checks.get("onnx_initializer_match")
            if isinstance(match, dict):
                onnx_info = self.nodes[engine_node].summary.get("onnx") or {}
                dep["backbone_sibling_onnx_matches"] = {
                    "verdict": match["verdict"],
                    "nodes": match["best"],
                    "onnx": onnx_info.get("path"),
                    "engine_link": onnx_info.get("engine_link"),
                }
                # The head was trained on features of *its* teacher.  The strongest
                # statement the bytes support about the deployed backbone is: the
                # ONNX the role file associates with the engine matches teacher X.
                # Whether the engine was built from that ONNX is not established.
                if ckpt_node and self.nodes[ckpt_node].exists:
                    training_teachers = [
                        e.target_node
                        for e in self.nodes[ckpt_node].edges
                        if e.relation == "teacher"
                    ]
                    if match["verdict"] == "unique_exact" and training_teachers:
                        same = match["best"][0] in training_teachers
                        dep["sibling_onnx_teacher_evidence"] = {
                            "status": (
                                "same_teacher_indicated"
                                if same
                                else "DIFFERENT_TEACHER_INDICATED"
                            ),
                            "onnx_matches": match["best"][0],
                            "head_trained_against": training_teachers,
                            "engine_bytes_attributed": False,
                            "reading": (
                                "sibling-ONNX evidence only; the deployed engine's own "
                                "provenance is unresolved"
                            ),
                        }
                    else:
                        dep["sibling_onnx_teacher_evidence"] = {"status": "unknown"}
        dep["head_engine"] = (
            preset.get("mamba_head_engine") or "none (PyTorch head inside whole graph)"
        )
        dep["embedding"] = f"reid_mode={preset.get('reid_mode')!r}"
        dep["graphs"] = {
            k: preset.get(k)
            for k in ("use_whole_graph", "use_cuda_graph", "use_tracker_graph")
        }
        node.checks["deployment_forward"] = dep

    # ---- output

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "issue": self.roles.get("issue"),
            "captured": {
                "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": platform.node(),
                "git_head": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "roles_file": _display(self.roles_path) if self.roles_path else None,
                "tensor_diff": self.tensor_diff,
                "onnx_match": self.onnx_match,
                "cache_content_hash": self.cache_content_hash,
            },
            "families": self.roles.get("families", {}),
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
        }


# --------------------------------------------------------------------------- markdown


def _status_cell(node: Node) -> str:
    if not node.exists:
        return "**unavailable**"
    return "problem" if node.problems else "present"


def _short_sha(sha: str | None) -> str:
    return f"`{sha[:12]}`" if sha else "—"


def _edge_cell(node: Node, relation: str) -> str:
    cells = []
    for e in node.edges:
        if e.relation != relation:
            continue
        label = e.target_node or f"unlisted:`{e.target_path}`"
        if not e.target_exists:
            label += " (missing)"
        cells.append(label)
    return ", ".join(cells) if cells else "—"


def _check_cell(node: Node, key: str) -> str:
    val = node.checks.get(key)
    if val is None:
        return "—"
    if isinstance(val, dict):
        return val.get("status", "?")
    return str(val)


def render_markdown(payload: dict[str, Any], inv: Inventory) -> str:
    cap = payload["captured"]
    head = cap["git_head"] or "unknown"
    lines: list[str] = [
        "<!-- doc-status: active -->",
        "<!-- doc-promotion: none -->",
        f"<!-- doc-date: {cap['at_utc'][:10]} -->",
        "<!-- doc-module: detection -->",
        "<!-- Captured by scripts/provenance/training_lineage.py; regenerate rather than edit. -->",
        "",
        "# Training lineage inventory (#421 · deliverable 1)",
        "",
        f"Captured {cap['at_utc']} on `{cap['host']}` at `{head[:12]}`"
        f"{' (dirty tree)' if cap['git_dirty'] else ''}; "
        f"tensor diff {'on' if cap['tensor_diff'] else 'off'}, ONNX match {'on' if cap['onnx_match'] else 'off'}. "
        f"Machine-readable twin: `{DEFAULT_JSON_OUT}`. Role claims: `{cap['roles_file']}`.",
        "",
        "This is a **captured snapshot of one workspace**, not a regenerable view: `runs/` and "
        "`models/` are gitignored. Every column except *role*, *note* and the `expected_*` claims "
        "is read from the artifact bytes. `unavailable` means the path does not exist here; no "
        "substitute was used. Numbers here are identities and counts, never quality metrics.",
        "",
        "## Reading the tables",
        "",
        "- **init** = warm-start parent recorded in the checkpoint's own `args.mamba_ckpt`; "
        "**teacher** / **base_yolo** / **cache** likewise from `args`. `unlisted:` = the artifact "
        "names a path that is not a declared role.",
        "- **expected_*** = the role file's claim checked against the derived edge: `ok` / "
        "`mismatch` / `unverifiable` (checkpoint records no such edge).",
        "- **sha attested** = whether `mamba_args.base_yolo_sha256` / `teacher_checkpoint_sha256` "
        "exist and verify against the file on disk (`not_recorded` for pre-2026-06-13 checkpoints).",
        "- **SSM interior frozen** = along the init edge, every `A_log/D/conv1d/x_proj/dt_proj` tensor "
        "of `mamba_blocks` and `temporal_blocks` is bit-identical to the parent (same dtype, shape "
        "and raw bytes). This is the measured counterpart of the `scan_stop_grad` flag.",
        "- **sibling ONNX ↔ ckpt** = the ONNX the role file associates with an engine, matched by "
        "exact float32 equality of its initializers against every checkpoint (teachers BN-folded). "
        "`unique_exact` names the checkpoint the *ONNX* matches; `partial` / `ambiguous` do not. "
        "**Engine bytes are never attributed**: the engine↔ONNX link is a role-file claim without a "
        "build manifest, so every deployment statement built on it is evidence about the sibling "
        "ONNX, not proof of the deployed engine's source.",
        "",
    ]

    for fam, fspec in payload["families"].items():
        nodes = [n for n in inv.nodes.values() if n.family == fam]
        lines += [f"## Family `{fam}` — backbone {fspec.get('backbone', '?')}", ""]
        lines += [
            "### Artifacts",
            "",
            "| role | kind | path | status | sha256 | size | epoch | seed |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
        for n in nodes:
            s = n.summary
            if n.size_bytes is None:
                size = "dir" if n.exists and not n.sha256 else "—"
            elif n.size_bytes < 1_000_000:
                size = f"{n.size_bytes / 1e3:.1f} KB"
            else:
                size = f"{n.size_bytes / 1e6:.1f} MB"
            epoch = s.get("epoch", "—") if n.exists else "—"
            seed = (s.get("treatment") or {}).get("seed", "—") if n.exists else "—"
            lines.append(
                f"| `{n.id}` | {n.kind} | `{n.path}` | {_status_cell(n)} | {_short_sha(n.sha256)} | {size} | {epoch} | {seed} |"
            )
        lines.append("")

        ckpts = [n for n in nodes if n.kind == "mamba_ckpt" and n.exists]
        if ckpts:
            lines += [
                "### Mamba checkpoints — derived edges and checks",
                "",
                "| role | init (derived) | teacher (derived) | cache (derived) | expected init | expected teacher | base_yolo sha | teacher sha | SSM interior frozen | tensors identical/changed/added | groups left bit-identical |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for n in ckpts:
                delta = n.checks.get("tensor_delta_vs_init_parent")
                if isinstance(delta, dict):
                    t = delta["totals"]
                    frozen = "yes" if delta["ssm_internal_frozen"] else "**no**"
                    counts = f"{t['identical']}/{t['changed']}/{t['added']} vs `{delta['parent']}`"
                    untouched = (
                        ", ".join(
                            g
                            for g, c in delta["groups"].items()
                            if c["identical"] and not c["changed"] and not c["added"]
                        )
                        or "none"
                    )
                else:
                    frozen, counts, untouched = "—", str(delta or "—"), "—"
                init_cell = _edge_cell(n, "init")
                if any(e.relation == "self_resume" for e in n.edges):
                    init_cell = "self (resume invocation)"
                lines.append(
                    f"| `{n.id}` | {init_cell} | {_edge_cell(n, 'teacher')} | {_edge_cell(n, 'cache')} | "
                    f"{_check_cell(n, 'expected_init_parent')} | {_check_cell(n, 'expected_teacher')} | "
                    f"{n.checks.get('base_yolo_sha_attested', '—')} | {n.checks.get('teacher_sha_attested', '—')} | "
                    f"{frozen} | {counts} | {untouched} |"
                )
            lines.append("")
            lines += [
                "### Mamba checkpoints — recorded training treatment",
                "",
                "| role | epochs | lr | clip_len | clip_stride | gt_ratio | add_temporal | scan_stop_grad | seqs | holdout | temporal blocks | params |",
                "|---|---:|---:|---:|---:|---:|---|---|---|---|---|---:|",
            ]
            for n in ckpts:
                t = n.summary.get("treatment", {})
                margs = n.summary.get("mamba_args", {})
                seqs = t.get("seqs")
                seqs_cell = (
                    "7-seq explicit"
                    if seqs and seqs.count(",") == 6
                    else ("all (default)" if not seqs else seqs)
                )
                lines.append(
                    f"| `{n.id}` | {t.get('epochs', '—')} | {t.get('lr', '—')} | {t.get('clip_len', '—')} | "
                    f"{t.get('clip_stride', '—')} | {t.get('gt_ratio', '—')} | {t.get('add_temporal', '—')} | "
                    f"{t.get('scan_stop_grad', margs.get('scan_stop_grad', '—'))} | {seqs_cell} | "
                    f"{t.get('holdout_seqs') or 'none'} | {'yes' if margs.get('use_temporal_mamba') else 'no'} | "
                    f"{n.summary.get('param_count', 0):,} |"
                )
            lines.append("")
            deltas = [
                (n, n.checks["tensor_delta_vs_init_parent"])
                for n in ckpts
                if isinstance(n.checks.get("tensor_delta_vs_init_parent"), dict)
            ]
            if deltas:
                lines += [
                    "### Warm-start edges — what the recorded treatment changed",
                    "",
                    "Keys both checkpoints recorded with different values. Flags that only the child "
                    "records (added to the training script later) are counted, not interpreted.",
                    "",
                ]
                for n, d in deltas:
                    td = d["treatment_delta"]
                    items = (
                        "; ".join(
                            f"`{k}` {v['parent']!r}→{v['child']!r}"
                            for k, v in td["changed"].items()
                        )
                        or "no recorded key differs"
                    )
                    extra = (
                        f" (+{len(td['only_child'])} keys only in child, {len(td['only_parent'])} only in parent)"
                        if td["only_child"] or td["only_parent"]
                        else ""
                    )
                    lines.append(f"- `{d['parent']}` → `{n.id}`: {items}{extra}")
                lines.append("")

        teachers = [n for n in nodes if n.kind == "gated_teacher_ckpt" and n.exists]
        if teachers:
            lines += [
                "### Gated teachers",
                "",
                "| role | base_yolo (derived) | epoch | provenance recorded | git commit | dirty | tensors |",
                "|---|---|---:|---|---|---|---:|",
            ]
            for n in teachers:
                prov = n.summary.get("provenance") or {}
                lines.append(
                    f"| `{n.id}` | {_edge_cell(n, 'base_yolo')} | {n.summary.get('epoch')} | "
                    f"{'yes' if prov else 'no'} | {str(prov.get('git_commit', '—'))[:12]} | {prov.get('git_diff_status', '—')} | "
                    f"{n.summary.get('tensor_count')} |"
                )
            lines.append("")

        caches = [n for n in nodes if n.kind == "teacher_cache"]
        if caches:
            lines += [
                "### Teacher caches",
                "",
                "| role | status | manifest schema | manifest sha256 | teacher (manifest) | decode | frames | content digest (files / bytes) |",
                "|---|---|---|---|---|---|---:|---|",
            ]
            for n in caches:
                m = (n.summary.get("manifest") or {}) if n.exists else {}
                content = n.summary.get("content") if n.exists else None
                if isinstance(content, dict):
                    content_cell = f"`{content['sha256'][:12]}` ({content['file_count']} / {content['total_bytes'] / 1e9:.1f} GB)"
                else:
                    content_cell = str(content) if content else "—"
                lines.append(
                    f"| `{n.id}` | {_status_cell(n)} | {m.get('schema', '—') if n.exists else '—'} | "
                    f"{_short_sha(n.summary.get('manifest_sha256')) if n.exists else '—'} | "
                    f"{_edge_cell(n, 'teacher') if n.exists else '—'} | {m.get('decode_backend', '—')} | {m.get('total_frames', '—')} | {content_cell} |"
                )
            lines.append("")

        engines = [n for n in nodes if n.kind == "trt_engine" and n.exists]
        if engines:
            lines += [
                "### Engines — sibling-ONNX initializer match (engine bytes unattributed)",
                "",
                "| role | sibling onnx (role-file claim) | stem equal | onnx verdict | matches (hits / initializers) |",
                "|---|---|---|---|---|",
            ]
            for n in engines:
                m = n.checks.get("onnx_initializer_match")
                onnx_info = n.summary.get("onnx") or {}
                onnx_path = onnx_info.get("path", "—")
                stem = (onnx_info.get("engine_link") or {}).get("stem_equal")
                stem_cell = "—" if stem is None else ("yes" if stem else "**no**")
                if isinstance(m, dict):
                    best_hits = m["matches"][m["best"][0]] if m["best"] else 0
                    others = sum(
                        1 for k, v in m["matches"].items() if v and k not in m["best"]
                    )
                    hits = f"{best_hits}/{m['total_initializers']} for each best; {others} other checkpoints with fewer hits"
                    lines.append(
                        f"| `{n.id}` | `{onnx_path}` | {stem_cell} | **{m['verdict']}** → {', '.join(f'`{b}`' for b in m['best']) or '—'} | {hits} |"
                    )
                else:
                    lines.append(
                        f"| `{n.id}` | `{onnx_path}` | {stem_cell} | {m} | — |"
                    )
            lines.append("")

        presets = [n for n in nodes if n.kind == "preset" and n.exists]
        for n in presets:
            dep = n.checks.get("deployment_forward", {})
            ck = dep.get("checkpoint", {})
            lines += [f"### Deployment forward — `{n.id}` (`{n.path}`)", ""]
            lines.append(
                f"- checkpoint: `{ck.get('path')}` → node `{ck.get('node')}`"
                + (
                    f"; same-sha aliases: {ck['aliases_same_sha256'] or 'none'}"
                    if "aliases_same_sha256" in ck
                    else f" ({ck.get('status', '')})"
                )
            )
            lines.append(f"- temporal blocks: {dep.get('temporal_blocks', '—')}")
            lines.append(
                f"- final-stage `gt_ratio`: {dep.get('final_stage_gt_ratio', '—')}; runtime gate teacher: `{dep.get('gate_teacher_at_runtime')}`"
            )
            lines.append(f"- backbone: {dep.get('backbone_source')}")
            so = dep.get("backbone_sibling_onnx_matches")
            if so:
                lines.append(
                    f"  - sibling ONNX `{so['onnx']}` (role-file association, stem equal: "
                    f"{(so.get('engine_link') or {}).get('stem_equal')}) matches **{so['verdict']}** → {so['nodes']}; "
                    "engine bytes themselves unattributed"
                )
            ev = dep.get("sibling_onnx_teacher_evidence")
            if ev and ev.get("status") != "unknown":
                lines.append(
                    f"  - sibling-ONNX evidence vs the teacher the head was trained against: **{ev['status']}** "
                    f"(ONNX ↔ `{ev['onnx_matches']}`; head trained against {ev['head_trained_against']}). "
                    "The deployed engine's own provenance is unresolved (no build manifest)."
                )
            lines.append(
                f"- head engine: `{dep.get('head_engine')}`; embedding: {dep.get('embedding')}; graphs: {dep.get('graphs')}"
            )
            lines.append("")

    xref = [
        (n, n.checks["paper_table"])
        for n in inv.nodes.values()
        if "paper_table" in n.checks
    ]
    if xref:
        lines += [
            "## Cross-reference — `report_data/tables/mamba_checkpoint_provenance.csv`",
            "",
            "| role | experiment (csv) | sha256 |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{n.id}` | `{x['experiment']}` | {x['status']} |" for n, x in xref
        ]
        lines.append("")
    problems = [(n.id, p) for n in inv.nodes.values() for p in n.problems]
    lines += ["## Problems", ""]
    if problems:
        lines += [f"- `{nid}`: {p}" for nid, p in problems]
    else:
        lines.append("None: every existing artifact was read as its declared kind.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture the detector training-lineage inventory from this workspace."
    )
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument(
        "--json-out", type=Path, default=None, help=f"default {DEFAULT_JSON_OUT}"
    )
    parser.add_argument(
        "--md-out", type=Path, default=None, help=f"default {DEFAULT_MD_OUT}"
    )
    parser.add_argument(
        "--no-tensor-diff",
        action="store_true",
        help="skip warm-start tensor comparison",
    )
    parser.add_argument(
        "--no-onnx-match",
        action="store_true",
        help="skip sibling-ONNX initializer matching",
    )
    parser.add_argument(
        "--no-cache-content-hash",
        action="store_true",
        help="skip hashing teacher-cache contents (tens of GB); manifest sha256 is still recorded",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the markdown instead of writing files",
    )
    args = parser.parse_args(argv)

    try:
        roles = load_roles(args.roles)
    except (LineageError, OSError, json.JSONDecodeError) as exc:
        print(f"training lineage: {exc}", file=sys.stderr)
        return 2

    inv = Inventory(
        roles,
        tensor_diff=not args.no_tensor_diff,
        onnx_match=not args.no_onnx_match,
        cache_content_hash=not args.no_cache_content_hash,
        roles_path=args.roles.resolve(),
    )
    inv.build()
    payload = inv.to_payload()
    markdown = render_markdown(payload, inv)

    if args.stdout:
        print(markdown)
        return 0
    json_out = REPO_ROOT / (args.json_out or DEFAULT_JSON_OUT)
    md_out = REPO_ROOT / (args.md_out or DEFAULT_MD_OUT)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    md_out.write_text(markdown, encoding="utf-8")
    missing = sorted(n.id for n in inv.nodes.values() if not n.exists)
    problems = sum(len(n.problems) for n in inv.nodes.values())
    print(f"wrote {_display(json_out)} and {_display(md_out)}")
    print(
        f"{len(inv.nodes)} roles; unavailable: {missing or 'none'}; read problems: {problems}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
