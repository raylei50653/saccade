#!/usr/bin/env python3
"""Run the #465 compiled-head reverse-engineering study (R0 provenance, R1 scope/stage localization) and write its packet.

Implements the frozen declaration
``docs/reference/native_runtime_compiled_head_reverse_engineering_declaration.md``
(blob ``FROZEN_DECLARATION_BLOB``) without re-deciding anything in it. The
study identity -- declaration, the PR-2L L1 construction it localizes, the
PR-2L packet it anchors to, arms, thresholds -- is fixed in this file and is
never a command-line option.

* V1 frozen inputs and execution freeze point -- checked in the parent before
  any compile; any mismatch is ``UNRESOLVED``.
* R0 (§4) -- one isolated child per arm on seeded synthetic backbone features
  (no MOT17 frame), with Dynamo/AOT/Inductor logging on: wrap sites, Dynamo
  counters, graph/code logs, an operator inventory and the arm's output hash.
* R1 (§4) -- one child builds all arms from deep copies of the same
  pre-compile head (``C``/``C_H``/``C_B`` through the head's own
  ``set_head_compile``/``set_block_compile``; ``D_s``/``A_s`` with the same
  per-module wrap loop and another backend), with no compile logging. Before
  the first frame it replays the R0 synthetic input: every arm must reproduce
  its isolated R0 output hash (stage identity), and the Dynamo counters must
  not move during the frames (no recompile). It then feeds the PR-2L L1 frame
  path's shared backbone features to every arm on all 5316 frames. The E,C
  pair is also accumulated with PR-2L's own formulas (V-anchor, exact).
* Decision (§5) -- the §5.1 drift class, the §5.2 scope cut and the §5.3
  stage ladder, first matching row; validity failure is ``UNRESOLVED``.

It reads no time quantity and has no smoke mode. The formal run must be the
direct child of a ``machine-bench`` lease on a clean tree, on the execution
freeze commit::

    .venv/bin/python tools/resctl.py run machine-bench -- \\
        .venv/bin/python scripts/eval/diagnostics/compiled_head_reverse_engineering.py

Structural checks without MOT17 data:
``compiled_head_reverse_engineering_structural_check.py``.
"""
# status: experiment

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# --- frozen by the declaration ------------------------------------------------
DECLARATION = (
    "docs/reference/native_runtime_compiled_head_reverse_engineering_declaration.md"
)
FROZEN_DECLARATION_BLOB = "565b4a1d8afc22714d33d9b39d4a5d2b199a6eca"
# The merge commit of #488 (declaration freeze). The head source, the PR-2L
# runner this study reuses and the export tool that resolves the frozen inputs
# must be unchanged from it to the execution freeze commit.
DECLARATION_FREEZE_COMMIT = "23d1736e5871abb6b3c0c71582c2899c449a25dc"
UNCHANGED_SINCE_DECLARATION = (
    "src/saccade",
    "scripts/eval/diagnostics/native_head_parity_libtorch.py",
    "scripts/model/export_headline_mamba_head.py",
)
# Execution freeze point: this runner PR's merge commit, named by an annotated
# tag created after that merge. Compared by peeled commit SHA only.
FREEZE_TAG = "freeze/465-compiled-head-re"
FREEZE_REMOTE = "origin"
FREEZE_BRANCH = "origin/main"

# §0: the construction being localized (PR-2L L1 C, runner blob 437bb97e at
# freeze commit 504ab0e2) and the head source it compiles.
PR2L_RUNNER = "scripts/eval/diagnostics/native_head_parity_libtorch.py"
PR2L_RUNNER_BLOB = "437bb97e116cf9af6597f7431ecd850d3053a89e"
MAMBA_HEAD_SOURCE = "src/saccade/perception/temporal_yolo/mamba_head.py"
MAMBA_HEAD_BLOB = "ddd4524c9b106c79a82f578495db42e9aec0a651"
# §2: the PR-2L packet the V-anchor reproduces (results/ is not tracked).
PR2L_PACKET = "results/native_head_parity_465_libtorch/20260928T152911Z"
PR2L_PACKET_FILES = {
    "packet.json": "e614f517ab089c8e8295bcafd645c30e45ee0e89be838af549a3193bb3ffaddc",
    "l1/l1.json": "467bca1e5aece4a2c7a9af9d5d1a5d6ee7c0d5bfa72b2b7fd16aec93561c4d6a",
    "l1/l1_pairs.csv": (
        "04f9bcb83d466a7a05641ffe5d612b073c074c60f5d74720d08c8495933d6d20"
    ),
}
ANCHOR_PAIR = "E,C"

DATA_ROOT = "datasets/MOT17"
SPLIT = "train"
SEQUENCES = (
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
)
TOTAL_FRAMES = 5316
LEASE = "machine-bench"
PACKET_ROOT = "results/compiled_head_reverse_engineering_465"
IMG_SIZE = 640
SCORE_FLOOR = 0.05
REPEAT_FRAMES = 20  # per sequence, every arm re-run (repeat identity)

# §4 arms. None = eager; "inductor" = torch.compile(m, mode="default") exactly
# as the head's setters do; otherwise torch.compile(m, backend=<name>).
ARM_SPECS: dict[str, dict[str, str | None]] = {
    "E": {"head": None, "block": None},
    "C": {"head": "inductor", "block": "inductor"},
    "C_H": {"head": "inductor", "block": None},
    "C_B": {"head": None, "block": "inductor"},
    "D_H": {"head": "eager", "block": None},
    "D_B": {"head": None, "block": "eager"},
    "A_H": {"head": "aot_eager_decomp_partition", "block": None},
    "A_B": {"head": None, "block": "aot_eager_decomp_partition"},
}
ARM_ORDER = tuple(ARM_SPECS)
# §5.2 / §5.3 reference arm R for every compared arm.
REFERENCE = {
    "C_H": "C",
    "C_B": "C",
    "D_H": "C_H",
    "A_H": "C_H",
    "D_B": "C_B",
    "A_B": "C_B",
}
SCOPE_ARMS = {"H": "C_H", "B": "C_B"}
LADDER = {"H": ("D_H", "A_H"), "B": ("D_B", "A_B")}
# §4: every declared arm is constructible in torch 2.11.0+cu130 (all four
# backends are registered); no arm is omitted before measurement.
OMITTED_ARMS: tuple[str, ...] = ()
HEAD_SITES = ("cls_head", "reg_head")
EXPECTED_HEAD_SITES = 6
EXPECTED_BLOCK_LAYOUT = (1, 1, 1)
EXPECTED_BACKENDS = ("eager", "aot_eager_decomp_partition", "inductor")

# §5.1 drift class (integer arithmetic: overlap/n >= 9/10).
OVERLAP_NUM, OVERLAP_DEN = 9, 10
MAXABS_LO, MAXABS_HI = 0.5, 2.0

SCOPE_RESULTS = (
    "UNRESOLVED",
    "HEAD_SCOPE",
    "BLOCK_SCOPE",
    "BOTH_SCOPES",
    "SCOPE_INTERACTION",
)
STAGE_LABELS = (
    "DYNAMO_OR_CAPTURE_BOUNDARY",
    "AOT_OR_DECOMPOSITION_BOUNDARY",
    "INDUCTOR_LOWERING_BOUNDARY",
    "UNRESOLVED",
)

# R0/R1 synthetic stage-identity input: seeded normal features of the head's
# static input shapes (never MOT17 data).
SYNTHETIC_SEED = 0
SYNTHETIC_CALLS = 3
# Backend policy held fixed at the PR-2L L1 values (§3/§4 R0); the harness
# sets none of them and the PR-2L L1 worker turned the TorchScript graph
# executor's optimize off when loading L (kept, for process fidelity).
POLICY = {
    "graph_executor_optimize": False,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": True,
    "matmul_allow_tf32": False,
}
# Compile cache policy (§7): fresh, empty Inductor and Triton cache
# directories per process under the packet; nothing is reused.
CACHE_ENV = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
FORBIDDEN_ENV_PREFIXES = ("SACCADE_", "TORCH", "PYTORCH_", "TRITON_")
RECOMPILE_LIMIT_MARKERS = ("recompile_limit", "cache_size_limit")
# Operator inventory from the R0 compile log: AOT/Inductor graphs name
# ``torch.ops.<ns>.<op>``; Dynamo's FX graph code (the only graph an ``eager``
# backend has) names the torch-level callables on ``[__graph_code]`` lines.
OP_PATTERN = re.compile(r"torch\.ops\.(aten|prims|saccade)\.([A-Za-z0-9_]+)")
FX_CALL_PATTERN = re.compile(r"= (torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\(")
GRAPH_CODE_MARKER = "[__graph_code]"


def _load_pr2l() -> Any:
    """The frozen PR-2L runner (blob-checked by V1) for its shared helpers."""
    spec = importlib.util.spec_from_file_location(
        "native_head_parity_libtorch", project_root / PR2L_RUNNER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P2L = _load_pr2l()


# --------------------------------------------------------------------------
# pure decision logic (§5; unit-tested without a GPU)
# --------------------------------------------------------------------------
def _overlap_ok(overlap: int, n_ref: int, n_arm: int) -> bool:
    """Recall and precision of the arm's difference set against R's, >= 0.9."""
    if n_ref == 0:
        return n_arm == 0
    if n_arm == 0:
        return False
    return (
        OVERLAP_DEN * overlap >= OVERLAP_NUM * n_ref
        and OVERLAP_DEN * overlap >= OVERLAP_NUM * n_arm
    )


def _maxabs_ok(m_arm: float, m_ref: float) -> bool:
    return MAXABS_LO * m_ref <= m_arm <= MAXABS_HI * m_ref


def class_checks(
    arm: dict[str, Any], ref: dict[str, Any], pair: dict[str, Any]
) -> dict[str, bool]:
    """§5.1 conditions 1–4 for X ∈ class(R); scores and boxes separately.

    ``arm``/``ref``: totals vs E ({"score"|"box": {"n", "maxabs"},
    "cross": {"n"}}); ``pair``: overlaps of the arm's sets with R's
    ({"score"|"box"|"cross": int}).
    """
    out: dict[str, bool] = {}
    for q in ("score", "box"):
        n_r, n_x = ref[q]["n"], arm[q]["n"]
        out[f"{q}_overlap"] = _overlap_ok(pair[q], n_r, n_x)
        out[f"{q}_maxabs"] = (
            n_x == 0 if n_r == 0 else _maxabs_ok(arm[q]["maxabs"], ref[q]["maxabs"])
        )
    out["cross_overlap"] = _overlap_ok(
        pair["cross"], ref["cross"]["n"], arm["cross"]["n"]
    )
    return out


def classify(
    arm: dict[str, Any], ref: dict[str, Any], pair: dict[str, Any], frames: int
) -> dict[str, Any]:
    """§5.1: ``equal_E`` / ``in_class`` / ``partial`` plus the exact qualifier."""
    checks = class_checks(arm, ref, pair)
    equal_e = arm["equal_E_frames"] == frames
    if equal_e:
        cls = "equal_E"
    elif all(checks.values()):
        cls = "in_class"
    else:
        cls = "partial"
    return {
        "class": cls,
        "exact_R": pair["equal_R_frames"] == frames,
        "checks": checks,
    }


def scope_result(
    validity_ok: bool, c_h: dict[str, Any] | None, c_b: dict[str, Any] | None
) -> tuple[str, tuple[str, ...]]:
    """§5.2, first satisfied row; returns (result, scopes whose ladder runs)."""
    if not validity_ok or c_h is None or c_b is None:
        return "UNRESOLVED", ()
    h, b = c_h["class"], c_b["class"]
    if h == "in_class" and b == "equal_E":
        return "HEAD_SCOPE", ("H",)
    if b == "in_class" and h == "equal_E":
        return "BLOCK_SCOPE", ("B",)
    if h != "equal_E" and b != "equal_E":
        return "BOTH_SCOPES", ("H", "B")
    if h == "equal_E" and b == "equal_E":
        return "SCOPE_INTERACTION", ()
    return "UNRESOLVED", ()


def stage_label(d: dict[str, Any] | None, a: dict[str, Any] | None) -> tuple[str, str]:
    """§5.3 for one scope (R = C_s), first satisfied row; (label, reason)."""
    if d is None:
        return "UNRESOLVED", "D_s omitted"
    if d["class"] == "in_class":
        return "DYNAMO_OR_CAPTURE_BOUNDARY", "D_s in class(R)"
    if d["class"] != "equal_E":
        return "UNRESOLVED", "first divergence at capture, not sufficient"
    if a is None:
        return "UNRESOLVED", "A_s omitted"
    if a["class"] == "in_class":
        return "AOT_OR_DECOMPOSITION_BOUNDARY", "D_s = E, A_s in class(R)"
    if a["class"] != "equal_E":
        return "UNRESOLVED", "first divergence at AOT, not sufficient"
    return "INDUCTOR_LOWERING_BOUNDARY", "D_s = E, A_s = E"


def decide(validity_ok: bool, r1: dict[str, Any] | None) -> dict[str, Any]:
    """§5.1–§5.3 over the R1 totals; any validity failure is UNRESOLVED."""
    if not validity_ok or r1 is None:
        return {"scope": "UNRESOLVED", "stages": {}, "classes": {}}
    frames = r1["frames"]
    totals, pairs = r1["arms"], r1["pairs"]
    classes = {
        x: classify(totals[x], totals[r], pairs[f"{x}|{r}"], frames)
        for x, r in REFERENCE.items()
        if x not in OMITTED_ARMS
    }
    scope, ladders = scope_result(
        True, classes.get(SCOPE_ARMS["H"]), classes.get(SCOPE_ARMS["B"])
    )
    stages = {}
    for s in ladders:
        d_arm, a_arm = LADDER[s]
        label, why = stage_label(classes.get(d_arm), classes.get(a_arm))
        stages[s] = {"label": label, "reason": why, "reference": SCOPE_ARMS[s]}
    return {"scope": scope, "stages": stages, "classes": classes}


# --------------------------------------------------------------------------
# pure validity logic (unit-tested without a GPU)
# --------------------------------------------------------------------------
def freeze_problems(
    head: str,
    parents: list[str],
    on_first_parent_chain: bool,
    tag_object_type: str | None,
    tag_local_commit: str | None,
    tag_remote_peeled: str | None,
) -> list[str]:
    """Execution freeze point; every comparison is by commit SHA."""
    problems = []
    if len(parents) != 2:
        problems.append(
            f"HEAD has {len(parents)} parent(s); the freeze commit is a merge"
        )
    if not on_first_parent_chain:
        problems.append(f"HEAD is not on the {FREEZE_BRANCH} first-parent chain")
    if tag_object_type != "tag":
        problems.append(f"{FREEZE_TAG} is {tag_object_type!r}, not an annotated tag")
    if tag_local_commit != head:
        problems.append(f"local {FREEZE_TAG}^{{commit}} {tag_local_commit} != HEAD")
    if tag_remote_peeled != head:
        problems.append(
            f"{FREEZE_REMOTE} {FREEZE_TAG}^{{}} {tag_remote_peeled} != HEAD"
        )
    return problems


def forbidden_env(environ: dict[str, str]) -> dict[str, str]:
    return {
        k: v for k, v in sorted(environ.items()) if k.startswith(FORBIDDEN_ENV_PREFIXES)
    }


def _float_eq(observed: Any, expected: Any) -> bool:
    return (
        isinstance(observed, (int, float))
        and isinstance(expected, (int, float))
        and float(observed) == float(expected)
    )


def anchor_problems(
    observed_rows: dict[str, dict[str, Any]],
    observed_hists: dict[str, dict[str, list[int]]],
    reference_rows: dict[str, dict[str, str]],
    reference_hists: dict[str, dict[str, list[int]]],
) -> list[str]:
    """V-anchor: R1's E,C rows and histograms equal PR-2L's, exactly.

    ``reference_rows`` are the PR-2L ``l1_pairs.csv`` rows of the anchor pair
    (strings, keyed by sequence, including ``ALL``); ``observed_rows`` are
    this run's rows built by the same ``_l1_row``.
    """
    problems = []
    if set(observed_rows) != set(reference_rows):
        problems.append(
            f"anchor sequences {sorted(observed_rows)} != {sorted(reference_rows)}"
        )
    for seq in sorted(set(observed_rows) & set(reference_rows)):
        obs, ref = observed_rows[seq], reference_rows[seq]
        for key, want in ref.items():
            if key in ("pair", "sequence"):
                continue
            got = obs.get(key)
            if want == "":
                ok = got is None
            else:
                ok = _float_eq(got, float(want))
            if not ok:
                problems.append(f"anchor {seq} {key}: {got!r} != {want!r}")
    for seq, ref in sorted(reference_hists.items()):
        obs = observed_hists.get(seq) or {}
        for key in ("score_hist", "box_hist"):
            if obs.get(key) != ref.get(key):
                problems.append(f"anchor {seq} {key} differs")
    if set(observed_hists) != set(reference_hists):
        problems.append("anchor histogram sequences differ")
    return problems


def r0_problems(arm: str, r0: dict[str, Any] | None) -> list[str]:
    """R0 per-arm child: completed, repeatable, sites as expected, no limit hit."""
    if r0 is None:
        return [f"R0 {arm}: no record"]
    problems = []
    if r0.get("arm") != arm:
        problems.append(f"R0 {arm}: record is for {r0.get('arm')!r}")
    if not r0.get("repeat_identical"):
        problems.append(f"R0 {arm}: synthetic output not repeatable")
    if not isinstance(r0.get("output_sha256"), str):
        problems.append(f"R0 {arm}: no output hash")
    problems += [f"R0 {arm}: {p}" for p in site_problems(arm, r0.get("wrap_sites"))]
    if r0.get("recompile_limit_hit"):
        problems.append(f"R0 {arm}: Dynamo recompile limit reached")
    if r0.get("policy") != POLICY:
        problems.append(f"R0 {arm}: policy {r0.get('policy')} != {POLICY}")
    return problems


def expected_sites(arm: str) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    return {
        "head": {
            "backend": spec["head"],
            "count": EXPECTED_HEAD_SITES if spec["head"] else 0,
        },
        "block": {
            "backend": spec["block"],
            "count": sum(EXPECTED_BLOCK_LAYOUT) if spec["block"] else 0,
        },
    }


def site_problems(arm: str, sites: dict[str, Any] | None) -> list[str]:
    if sites != expected_sites(arm):
        return [f"wrap sites {sites} != {expected_sites(arm)}"]
    return []


def r1_problems(
    r1: dict[str, Any], r0: dict[str, dict[str, Any]], expected_frames: int
) -> list[str]:
    """R1 validity (§4/§7) apart from the V-anchor."""
    problems = []
    if r1.get("frames") != expected_frames:
        problems.append(f"R1 frames {r1.get('frames')} != {expected_frames}")
    for arm in ARM_ORDER:
        want = (r0.get(arm) or {}).get("output_sha256")
        got = (r1.get("synthetic_sha256") or {}).get(arm)
        if want is None or got != want:
            problems.append(f"stage identity {arm}: R1 {got} != isolated R0 {want}")
        problems += [
            f"R1 {arm}: {p}"
            for p in site_problems(arm, (r1.get("wrap_sites") or {}).get(arm))
        ]
    before, after = r1.get("dynamo_counters_before"), r1.get("dynamo_counters_after")
    if before is None or before != after:
        problems.append("Dynamo compiled during the frames (counters moved)")
    if r1.get("recompile_limit_hit"):
        problems.append("R1: Dynamo recompile limit reached")
    if r1.get("repeat_failures"):
        problems.append(f"repeat identity failed: {r1['repeat_failures'][:5]}")
    per_seq = r1.get("frames_per_sequence") or {}
    want_repeat = sum(min(REPEAT_FRAMES, n) for n in per_seq.values())
    if not per_seq or r1.get("repeat_frames_checked") != want_repeat:
        problems.append(
            f"repeat frames checked {r1.get('repeat_frames_checked')} != {want_repeat}"
        )
    if r1.get("input_mutations"):
        problems.append(f"input mutated: {r1['input_mutations'][:5]}")
    for arm, t in (r1.get("arms") or {}).items():
        if t.get("nonfinite"):
            problems.append(f"{arm}: {t['nonfinite']} non-finite values")
    if r1.get("jit_fallback_calls") and any(r1["jit_fallback_calls"].values()):
        problems.append(f"JIT scan fallback called: {r1['jit_fallback_calls']}")
    if r1.get("policy") != POLICY:
        problems.append(f"R1 policy {r1.get('policy')} != {POLICY}")
    problems += [
        f"R1 driver/runtime: {p}"
        for p in P2L.driver_runtime_problems(r1.get("driver_runtime") or {})
    ]
    if r1.get("artifact_problems"):
        problems.append(f"R1 L artifact: {r1['artifact_problems']}")
    return problems


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_sha256 = P2L._sha256
_git = P2L._git
_git_or_none = P2L._git_or_none
_blob_or_none = P2L._blob_or_none
_sha_or_none = P2L._sha_or_none
_write_json = P2L._write_json
_utc = P2L._utc


def _seq_frames(seq: str, data_root: str) -> list[Path]:
    return sorted((project_root / data_root / SPLIT / seq / "img1").glob("*.jpg"))


def held_lease() -> dict[str, Any] | None:
    return P2L.held_lease()


def observed_freeze() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    parents = _git("log", "-1", "--format=%P", "HEAD").split()
    chain = _git_or_none("rev-list", "--first-parent", FREEZE_BRANCH) or ""
    ref = f"refs/tags/{FREEZE_TAG}"
    ls = subprocess.run(
        ["git", "ls-remote", FREEZE_REMOTE, f"{ref}^{{}}"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    return {
        "head": head,
        "parents": parents,
        "on_first_parent_chain": head in chain.split(),
        "tag_object_type": _git_or_none("cat-file", "-t", ref),
        "tag_object_sha": _git_or_none("rev-parse", ref),
        "tag_local_commit": _git_or_none("rev-parse", f"{ref}^{{commit}}"),
        "tag_remote_peeled": P2L.parse_ls_remote_peeled(ls.stdout)
        if ls.returncode == 0
        else None,
        "ls_remote_returncode": ls.returncode,
    }


def read_pr2l_anchor() -> tuple[
    dict[str, dict[str, str]], dict[str, dict[str, list[int]]]
]:
    packet = project_root / PR2L_PACKET
    with (packet / "l1" / "l1_pairs.csv").open() as f:
        rows = {r["sequence"]: r for r in csv.DictReader(f) if r["pair"] == ANCHOR_PAIR}
    l1 = json.loads((packet / "l1" / "l1.json").read_text())
    hists = {seq: d[ANCHOR_PAIR] for seq, d in l1["per_sequence_histograms"].items()}
    return rows, hists


# --------------------------------------------------------------------------
# V1 -- frozen inputs and execution freeze point (parent, before any compile)
# --------------------------------------------------------------------------
def check_v1() -> tuple[bool, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(item: str, ok: bool, observed: Any, expected: Any = None) -> None:
        checks.append(
            {"item": item, "ok": bool(ok), "observed": observed, "expected": expected}
        )

    for rel, want in PR2L_PACKET_FILES.items():
        got = _sha_or_none(project_root / PR2L_PACKET / rel)
        check(f"PR-2L packet {rel} sha256", got == want, got, want)
    for rel, want in (
        (PR2L_RUNNER, PR2L_RUNNER_BLOB),
        (MAMBA_HEAD_SOURCE, MAMBA_HEAD_BLOB),
    ):
        head_blob = _blob_or_none(rel)
        wt_blob = _git("hash-object", rel)
        check(
            f"{rel} blob (HEAD and worktree)",
            head_blob == want and wt_blob == want,
            {"head": head_blob, "worktree": wt_blob},
            want,
        )
    for label, rel, want in (
        ("checkpoint", P2L.CKPT, P2L.CKPT_SHA256),
        ("backbone engine", P2L.BACKBONE, P2L.BACKBONE_SHA256),
        (
            "TorchScript artifact (loaded as in PR-2L L1)",
            P2L.ARTIFACT,
            P2L.ARTIFACT_SHA256,
        ),
        ("operator library", P2L.OP_LIBRARY, P2L.OP_LIBRARY_SHA256),
    ):
        got = _sha_or_none(project_root / rel)
        check(f"{label} sha256", got == want, got, want)

    porcelain = _git("status", "--porcelain")
    check("clean tree", porcelain == "", porcelain or "(clean)", "(clean)")
    changed = _git_or_none(
        "diff",
        "--name-only",
        DECLARATION_FREEZE_COMMIT,
        "HEAD",
        "--",
        *UNCHANGED_SINCE_DECLARATION,
    )
    check(
        "head source, PR-2L runner and export tool unchanged since the declaration freeze",
        changed == "",
        changed.splitlines() if changed is not None else None,
        [],
    )
    runner_rel = str(Path(__file__).resolve().relative_to(project_root))
    blobs = {
        "head": _git("rev-parse", "HEAD"),
        "runner_blob": _blob_or_none(runner_rel),
        "declaration_blob": _blob_or_none(DECLARATION),
        "runner_worktree_blob": _git("hash-object", runner_rel),
        "declaration_worktree_blob": _git("hash-object", DECLARATION),
    }
    check(
        "declaration blob (HEAD and worktree) == frozen",
        blobs["declaration_blob"] == FROZEN_DECLARATION_BLOB
        and blobs["declaration_worktree_blob"] == FROZEN_DECLARATION_BLOB,
        blobs["declaration_blob"],
        FROZEN_DECLARATION_BLOB,
    )
    check(
        "runner committed (HEAD blob == worktree blob)",
        blobs["runner_blob"] == blobs["runner_worktree_blob"],
        blobs["runner_blob"],
        blobs["runner_worktree_blob"],
    )
    freeze = observed_freeze()
    problems = freeze_problems(
        freeze["head"],
        freeze["parents"],
        freeze["on_first_parent_chain"],
        freeze["tag_object_type"],
        freeze["tag_local_commit"],
        freeze["tag_remote_peeled"],
    )
    check(
        f"HEAD is the execution freeze commit ({FREEZE_TAG})",
        not problems,
        {**freeze, "problems": problems},
        "2-parent commit on the first-parent chain == peeled tag, local and remote",
    )
    bad_env = forbidden_env(dict(os.environ))
    check(
        "no SACCADE_/TORCH*/PYTORCH_/TRITON_ variable set by the caller",
        not bad_env,
        bad_env,
        {},
    )

    import torch
    import torch._dynamo

    torch.zeros(1, device="cuda")
    cap = torch.cuda.get_device_capability(0)
    env = {
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"{cap[0]}.{cap[1]}",
        "host": platform.node(),
    }
    for k, want in P2L.ENVIRONMENT.items():
        check(f"environment {k}", env[k] == want, env[k], want)
    backends = torch._dynamo.list_backends(exclude_tags=())
    missing = [b for b in EXPECTED_BACKENDS if b not in backends]
    check("declared compile backends registered", not missing, missing, [])
    driver = P2L.probe_driver_runtime()
    driver_bad = P2L.driver_runtime_problems(driver)
    check(
        "NVIDIA driver and CUDA runtime (exact, as PR-2L)",
        not driver_bad,
        {**driver, "problems": driver_bad},
        {
            "nvidia_driver": P2L.NVIDIA_DRIVER,
            "cu_driver_version": P2L.CU_DRIVER_VERSION,
            "cudart_version": P2L.CUDART_VERSION,
        },
    )
    frames = {s: len(_seq_frames(s, DATA_ROOT)) for s in SEQUENCES}
    check(
        "data sequences and frame count",
        sum(frames.values()) == TOTAL_FRAMES and all(frames.values()),
        {"frames": frames, "total": sum(frames.values())},
        {"sequences": list(SEQUENCES), "total": TOTAL_FRAMES},
    )
    lease = held_lease()
    check(
        f"direct child of a {LEASE} lease",
        lease is not None and lease.get("resource") == LEASE,
        lease,
        f"{LEASE} held by parent pid {os.getppid()}",
    )
    ok = all(c["ok"] for c in checks)
    return ok, {
        "ok": ok,
        "checks": checks,
        "git": blobs,
        "freeze": freeze,
        "lease": lease,
    }


# --------------------------------------------------------------------------
# in-process construction (children only)
# --------------------------------------------------------------------------
def _setup_process() -> tuple[Any, dict[str, int], dict[str, Any]]:
    """The PR-2L L1 worker's process setup: extension first, JIT-fallback
    guards, operator library + L loaded (optimize off), nothing else."""
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    import saccade_tracking_ext  # noqa: F401  (before torchvision; see export tool)
    import torch

    guard_counts: dict[str, int] = {}
    P2L.install_jit_fallback_guards(guard_counts)
    _, artifact = P2L.load_l(torch)
    return torch, guard_counts, artifact


def build_base(torch: Any) -> tuple[Any, Any]:
    """The PR-2L L1 detector; returns (detector, pre-compile head copy)."""
    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
    )

    export = P2L._load_trt_export_tool()
    inputs = export.resolve_inputs(
        export.DEFAULT_YOLO_WEIGHTS, export.DEFAULT_TEACHER_CKPT
    )
    det = build_mamba_gated_detector(
        yolo_pt_path=str(inputs["yolo_weights"]),
        teacher_ckpt=str(inputs["teacher_ckpt"]),
        mamba_ckpt=str(inputs["ckpt"]),
        img_size=IMG_SIZE,
        device="cuda",
        conf_thr=0.001,
        trt_backbone_engine=str(inputs["backbone"]),
        use_whole_graph=True,
    )
    det.eval()
    return det, copy.deepcopy(det.mamba_head).eval()


def _compile(torch: Any, module: Any, backend: str) -> Any:
    if backend == "inductor":
        return torch.compile(module, mode="default")
    return torch.compile(module, backend=backend)


def build_arm(torch: Any, base: Any, arm: str) -> tuple[Any, dict[str, Any]]:
    """One arm from its own deep copy of the pre-compile head (§4 table)."""
    import torch.nn as nn

    spec = ARM_SPECS[arm]
    head = copy.deepcopy(base).eval()
    if spec["head"] == "inductor":
        head.set_head_compile(True)  # the setter itself (PR-2L L1 C)
    elif spec["head"] is not None:
        for name in HEAD_SITES:  # the setter's loop, another backend
            setattr(
                head,
                name,
                nn.ModuleList(
                    [_compile(torch, m, spec["head"]) for m in getattr(head, name)]
                ),
            )
    if spec["block"] == "inductor":
        head.set_block_compile(True)
    elif spec["block"] is not None:
        head.mamba_blocks = nn.ModuleList(
            [
                nn.ModuleList([_compile(torch, b, spec["block"]) for b in scale])
                for scale in head.mamba_blocks
            ]
        )
    return head, observe_sites(head)


def observe_sites(head: Any) -> dict[str, Any]:
    """Backend and count of the compile wrappers actually installed."""

    def backend_of(m: Any) -> str | None:
        if not hasattr(m, "_orig_mod"):
            return None
        # CatchErrorsWrapper -> ConvertFrame -> WrapBackendDebug(compiler_name)
        x = getattr(getattr(m, "dynamo_ctx", None), "callback", None)
        for _ in range(8):
            if x is None:
                break
            name = getattr(x, "compiler_name", None)
            if isinstance(name, str):
                return name
            x = getattr(x, "_torchdynamo_orig_backend", None) or getattr(
                x, "_torchdynamo_orig_callable", None
            )
        return "unknown"

    head_mods = [m for n in HEAD_SITES for m in getattr(head, n)]
    block_mods = [b for scale in head.mamba_blocks for b in scale]

    def summarize(mods: list[Any]) -> dict[str, Any]:
        kinds = {backend_of(m) for m in mods}
        if kinds == {None}:
            return {"backend": None, "count": 0}
        if len(kinds) != 1:
            return {"backend": sorted(map(str, kinds)), "count": -1}
        return {"backend": kinds.pop(), "count": len(mods)}

    return {"head": summarize(head_mods), "block": summarize(block_mods)}


def block_layout(head: Any) -> tuple[int, ...]:
    return tuple(len(s) for s in head.mamba_blocks)


def synthetic_features(torch: Any) -> list[Any]:
    g = torch.Generator(device="cuda").manual_seed(SYNTHETIC_SEED)
    return [torch.randn(s, device="cuda", generator=g) for s in P2L.INPUT_SHAPES]


def run_head(head: Any, feats: list[Any]) -> list[Any]:
    """PR-2L L1 ``run_torch``: ``_forward_eager``, outputs cloned."""
    cls, reg = head._forward_eager(list(feats), return_embeddings=False)
    return [t.clone() for t in (*cls, *reg)]


def outputs_sha256(outs: list[Any]) -> str:
    h = hashlib.sha256()
    for t in outs:
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def synthetic_hash(torch: Any, head: Any) -> tuple[str, bool]:
    feats = synthetic_features(torch)
    shas = [
        outputs_sha256(run_head(head, [f.clone() for f in feats]))
        for _ in range(SYNTHETIC_CALLS)
    ]
    return shas[0], len(set(shas)) == 1


def dynamo_counters() -> dict[str, dict[str, int]]:
    from torch._dynamo.utils import counters

    return {k: dict(v) for k, v in sorted(counters.items())}


class _LimitWatcher(logging.Handler):
    """Records any Dynamo recompile-limit warning (observation only)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.hit: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if any(m in msg for m in RECOMPILE_LIMIT_MARKERS):
            self.hit.append(msg[:300])


def _watch_limits() -> _LimitWatcher:
    w = _LimitWatcher()
    logging.getLogger("torch._dynamo").addHandler(w)
    return w


# --------------------------------------------------------------------------
# R0 child: one arm, isolated, synthetic features, compile logging on
# --------------------------------------------------------------------------
def run_r0_child(arm: str, out: Path) -> int:
    torch, guard_counts, artifact = _setup_process()
    watcher = _watch_limits()
    det, base = build_base(torch)
    layout = block_layout(base)
    torch._logging.set_logs(
        graph_code=True,
        aot_graphs=True,
        output_code=True,
        graph_breaks=True,
        recompiles=True,
    )
    head, sites = build_arm(torch, base, arm)
    with torch.inference_mode():
        sha, repeat_ok = synthetic_hash(torch, head)
    _write_json(
        out,
        {
            "arm": arm,
            "spec": ARM_SPECS[arm],
            "wrap_sites": sites,
            "block_layout": list(layout),
            "output_sha256": sha,
            "repeat_identical": repeat_ok,
            "dynamo_counters": dynamo_counters(),
            "recompile_limit_hit": watcher.hit,
            "jit_fallback_calls": dict(guard_counts),
            "policy": P2L.runtime_readback(torch),
            "artifact_problems": P2L.artifact_problems(artifact),
            "torch": torch.__version__,
            "cache_env": {k: os.environ.get(k) for k in CACHE_ENV},
        },
    )
    return 0


def op_inventory(log_text: str) -> dict[str, dict[str, int]]:
    """{"ops": torch.ops.* counts (AOT/Inductor), "fx": Dynamo graph-code calls}."""
    ops: dict[str, int] = {}
    for ns, name in OP_PATTERN.findall(log_text):
        key = f"{ns}.{name}"
        ops[key] = ops.get(key, 0) + 1
    fx: dict[str, int] = {}
    for line in log_text.splitlines():
        if GRAPH_CODE_MARKER in line:
            for name in FX_CALL_PATTERN.findall(line):
                fx[name] = fx.get(name, 0) + 1
    return {"ops": dict(sorted(ops.items())), "fx": dict(sorted(fx.items()))}


# --------------------------------------------------------------------------
# R1 worker: all arms, one process, PR-2L L1 frame path
# --------------------------------------------------------------------------
def run_r1_worker(
    out_dir: Path, sequences: tuple[str, ...], data_root: str = DATA_ROOT
) -> int:
    torch, guard_counts, artifact = _setup_process()
    import torch.nn.functional as F
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        _dfl_decode,
        _dist2bbox_xywh,
    )

    watcher = _watch_limits()
    driver = P2L.probe_driver_runtime()
    det, base = build_base(torch)
    backbone = det._trt_backbone
    anchors_t = det._whole_graph_anchors.T.unsqueeze(0)
    astrides = det._whole_graph_anchor_strides.squeeze(-1).unsqueeze(0)
    heads, sites = {}, {}
    for arm in ARM_ORDER:
        heads[arm], sites[arm] = build_arm(torch, base, arm)
    with torch.inference_mode():
        synthetic = {arm: synthetic_hash(torch, heads[arm])[0] for arm in ARM_ORDER}
    counters_before = dynamo_counters()

    def decode(outs: list[Any], sx: float, sy: float) -> tuple[Any, Any]:
        # PR-2L L1 decode, verbatim
        cls_all = torch.cat([c.flatten(2) for c in outs[:3]], dim=2)[0]
        reg_all = torch.cat([r.flatten(2) for r in outs[3:]], dim=2)
        bboxes = _dist2bbox_xywh(_dfl_decode(reg_all), anchors_t, dim=1) * astrides
        xywh = bboxes[0].T  # (N, 4)
        xyxy = torch.cat(
            [xywh[:, :2] - xywh[:, 2:4] / 2, xywh[:, :2] + xywh[:, 2:4] / 2], 1
        )
        scale = torch.tensor([sx, sy, sx, sy], dtype=xyxy.dtype, device=xyxy.device)
        return cls_all.sigmoid(), xyxy * scale

    edges = torch.tensor(P2L.hist_edges(), dtype=torch.float64, device="cuda")
    nbins = len(edges) + 1

    def hist(values: Any) -> Any:
        v = values.double().flatten()
        idx = torch.where(v == 0, 0, torch.bucketize(v, edges, right=True) + 1)
        idx = idx.clamp(max=nbins)
        return torch.bincount(idx, minlength=nbins + 1)

    def zi() -> Any:
        return torch.zeros((), dtype=torch.int64, device="cuda")

    def zf() -> Any:
        return torch.zeros((), dtype=torch.float64, device="cuda")

    def new_anchor_acc() -> dict[str, Any]:
        return {
            "score_max": zf(),
            "box_max": zf(),
            "score_hist": torch.zeros(nbins + 1, dtype=torch.int64, device="cuda"),
            "box_hist": torch.zeros(nbins + 1, dtype=torch.int64, device="cuda"),
            "masked_anchors": zi(),
            "crossings": zi(),
            "nonfinite": zi(),
            "frames": 0,
        }

    compared = [a for a in ARM_ORDER if a != "E"]
    arm_acc = {
        a: {
            "score_n": zi(),
            "score_max": zf(),
            "box_n": zi(),
            "box_max": zf(),
            "cross_n": zi(),
            "nonfinite": zi(),
            "equal_E_frames": 0,
        }
        for a in ARM_ORDER
    }
    pair_acc = {
        f"{x}|{r}": {"score": zi(), "box": zi(), "cross": zi(), "equal_R_frames": 0}
        for x, r in REFERENCE.items()
    }
    anchor_per_seq: dict[str, dict[str, Any]] = {}
    repeat_failures: list[dict[str, Any]] = []
    input_mutations: list[dict[str, Any]] = []
    repeat_checked = 0
    frames_done = 0
    rgb = ImageReadMode.RGB
    with torch.inference_mode():
        for seq in sequences:
            anchor = new_anchor_acc()
            frames = _seq_frames(seq, data_root)
            for i, path in enumerate(frames):
                img = decode_jpeg(read_file(str(path)), device="cuda", mode=rgb)
                h_orig, w_orig = int(img.shape[1]), int(img.shape[2])
                frame = img.float().unsqueeze(0) / 255
                frame_640 = F.interpolate(
                    frame,
                    size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                feats = [p.clone() for p in backbone.infer(frame_640)]
                snapshot = [p.clone() for p in feats]
                outs = {a: run_head(heads[a], feats) for a in ARM_ORDER}
                if i < REPEAT_FRAMES:
                    repeat_checked += 1
                    for a in ARM_ORDER:
                        again = run_head(heads[a], feats)
                        if not all(torch.equal(x, y) for x, y in zip(outs[a], again)):
                            repeat_failures.append(
                                {"sequence": seq, "frame_index": i, "arm": a}
                            )
                if not all(torch.equal(x, y) for x, y in zip(feats, snapshot)):
                    input_mutations.append({"sequence": seq, "frame_index": i})
                sx, sy = w_orig / IMG_SIZE, h_orig / IMG_SIZE
                dec = {a: decode(v, sx, sy) for a, v in outs.items()}
                s_e, b_e = dec["E"]
                m_e = s_e.max(0).values >= SCORE_FLOOR
                for a in ARM_ORDER:
                    arm_acc[a]["nonfinite"] += sum(
                        (~torch.isfinite(t)).sum() for t in outs[a]
                    )
                sets: dict[str, tuple[Any, Any, Any]] = {}
                for a in compared:
                    s_x, b_x = dec[a]
                    d_s = s_x != s_e
                    d_b = (b_x != b_e)[m_e]
                    k = (s_x.max(0).values >= SCORE_FLOOR) ^ m_e
                    sets[a] = (d_s, d_b, k)
                    acc = arm_acc[a]
                    acc["score_n"] += d_s.sum()
                    acc["score_max"] = torch.maximum(
                        acc["score_max"], (s_x - s_e).abs().max().double()
                    )
                    acc["box_n"] += d_b.sum()
                    db = (b_x - b_e).abs()[m_e]
                    if db.numel():
                        acc["box_max"] = torch.maximum(
                            acc["box_max"], db.max().double()
                        )
                    acc["cross_n"] += k.sum()
                    if all(torch.equal(x, y) for x, y in zip(outs[a], outs["E"])):
                        acc["equal_E_frames"] += 1
                arm_acc["E"]["equal_E_frames"] += 1
                for x, r in REFERENCE.items():
                    pa = pair_acc[f"{x}|{r}"]
                    for j, q in enumerate(("score", "box", "cross")):
                        pa[q] += (sets[x][j] & sets[r][j]).sum()
                    if all(torch.equal(u, v) for u, v in zip(outs[x], outs[r])):
                        pa["equal_R_frames"] += 1
                # V-anchor: the PR-2L L1 pair accumulation for (E, C), verbatim
                (sa, ba), (sb, bb) = dec["E"], dec["C"]
                ds = (sa - sb).abs()
                ma, mb = (
                    sa.max(0).values >= SCORE_FLOOR,
                    sb.max(0).values >= SCORE_FLOOR,
                )
                mask = ma | mb
                db = (ba - bb).abs()[mask]
                anchor["nonfinite"] += (~torch.isfinite(ds)).sum() + (
                    ~torch.isfinite(db)
                ).sum()
                anchor["score_max"] = torch.maximum(
                    anchor["score_max"], ds.max().double()
                )
                if db.numel():
                    anchor["box_max"] = torch.maximum(
                        anchor["box_max"], db.max().double()
                    )
                anchor["score_hist"] += hist(ds)
                anchor["box_hist"] += hist(db)
                anchor["masked_anchors"] += mask.sum()
                anchor["crossings"] += (ma ^ mb).sum()
                anchor["frames"] += 1
                frames_done += 1
            anchor_per_seq[seq] = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in anchor.items()
            }
            print(f"[R1] {seq}: {len(frames)} frames", flush=True)
    counters_after = dynamo_counters()

    def as_py(v: Any) -> Any:
        return v.item() if hasattr(v, "item") else v

    arms_out = {
        a: {
            "score": {"n": as_py(t["score_n"]), "maxabs": as_py(t["score_max"])},
            "box": {"n": as_py(t["box_n"]), "maxabs": as_py(t["box_max"])},
            "cross": {"n": as_py(t["cross_n"])},
            "nonfinite": as_py(t["nonfinite"]),
            "equal_E_frames": t["equal_E_frames"],
        }
        for a, t in arm_acc.items()
    }
    pairs_out = {k: {q: as_py(v) for q, v in p.items()} for k, p in pair_acc.items()}

    anchor_rows = {
        seq: P2L._l1_row(ANCHOR_PAIR, seq, s) for seq, s in anchor_per_seq.items()
    }
    tot = {
        "score_max": max(
            (s["score_max"] for s in anchor_per_seq.values()), default=0.0
        ),
        "box_max": max((s["box_max"] for s in anchor_per_seq.values()), default=0.0),
        "score_hist": [
            sum(x) for x in zip(*(s["score_hist"] for s in anchor_per_seq.values()))
        ],
        "box_hist": [
            sum(x) for x in zip(*(s["box_hist"] for s in anchor_per_seq.values()))
        ],
    }
    for k in ("masked_anchors", "crossings", "nonfinite", "frames"):
        tot[k] = sum(s[k] for s in anchor_per_seq.values())
    anchor_rows["ALL"] = P2L._l1_row(ANCHOR_PAIR, "ALL", tot)

    _write_json(
        out_dir / "r1.json",
        {
            "frames": frames_done,
            "sequences": list(sequences),
            "frames_per_sequence": {
                seq: s["frames"] for seq, s in anchor_per_seq.items()
            },
            "arms": arms_out,
            "pairs": pairs_out,
            "wrap_sites": sites,
            "block_layout": list(block_layout(base)),
            "synthetic_sha256": synthetic,
            "dynamo_counters_before": counters_before,
            "dynamo_counters_after": counters_after,
            "recompile_limit_hit": watcher.hit,
            "repeat_frames_checked": repeat_checked,
            "repeat_failures": repeat_failures,
            "input_mutations": input_mutations,
            "jit_fallback_calls": dict(guard_counts),
            "policy": P2L.runtime_readback(torch),
            "driver_runtime": driver,
            "artifact_problems": P2L.artifact_problems(artifact),
            "anchor": {
                "pair": ANCHOR_PAIR,
                "rows": anchor_rows,
                "per_sequence_histograms": {
                    seq: {"score_hist": s["score_hist"], "box_hist": s["box_hist"]}
                    for seq, s in anchor_per_seq.items()
                },
            },
            "environment": {"torch": torch.__version__},
            "cache_env": {k: os.environ.get(k) for k in CACHE_ENV},
        },
    )
    return 0


# --------------------------------------------------------------------------
# parent orchestration
# --------------------------------------------------------------------------
def child_env(cache_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    cache_dir.mkdir(parents=True, exist_ok=False)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "inductor")
    env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    return env


def _run_child(args: list[str], log: Path, env: dict[str, str]) -> int:
    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    with log.open("w") as f:
        return subprocess.run(
            cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT, env=env
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--_r0", dest="r0", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_r1", dest="r1", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_out", dest="out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.r0:
        if args.r0 not in ARM_SPECS or not args.out:
            parser.error("internal child flags are malformed")
        return run_r0_child(args.r0, project_root / args.out)
    if args.r1:
        return run_r1_worker(project_root / args.r1, SEQUENCES)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    packet = project_root / PACKET_ROOT / stamp
    from scripts.provenance.run_manifest import open_run

    open_run(
        packet,
        produced_by="diagnostic",
        preset=P2L.PRESET_NAME,
        detector="SDP",
        dataset=f"{DATA_ROOT} {SPLIT}",
    )
    record: dict[str, Any] = {
        "schema": "saccade.compiled_head_reverse_engineering_packet/v1",
        "issue": "#465 compiled-head reverse engineering (diagnostic side branch)",
        "declaration": DECLARATION,
        "frozen_declaration_blob": FROZEN_DECLARATION_BLOB,
        "freeze_tag": FREEZE_TAG,
        "evidence": True,
        "arms": ARM_SPECS,
        "omitted_arms": list(OMITTED_ARMS),
        "sequences": list(SEQUENCES),
        "started_utc": _utc(),
    }
    print(f"packet: {packet.relative_to(project_root)}", flush=True)
    reasons: list[str] = []
    r1 = None
    try:
        v1_ok, v1 = check_v1()
        record["v1"] = v1
        _write_json(packet / "v1.json", v1)
        if not v1_ok:
            reasons.append(
                "V1: " + "; ".join(c["item"] for c in v1["checks"] if not c["ok"])
            )
        else:
            r1, m_reasons = _measure(packet, record)
            reasons += m_reasons
            end_lease = held_lease()
            record["lease_at_end"] = end_lease
            if end_lease != v1["lease"]:
                reasons.append("lease changed during the measurement")
    except Exception as exc:  # noqa: BLE001 -- execution-invalid => UNRESOLVED
        reasons.append(f"runner error: {exc!r}")
    result = decide(not reasons, r1)
    record.update(
        {
            "finished_utc": _utc(),
            "result": result,
            "unresolved_reasons": reasons,
            "terminal": result["scope"],
        }
    )
    _write_json(packet / "packet.json", record)
    P2L._write_manifest(packet)
    print(f"scope: {result['scope']}", flush=True)
    for s, st in result["stages"].items():
        print(f"  stage[{s}]: {st['label']} ({st['reason']})", flush=True)
    for r in reasons:
        print(f"  reason: {r}", flush=True)
    return 0 if result["scope"] != "UNRESOLVED" else 2


def _measure(
    packet: Path, record: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    # R0: every arm isolated; the first failure ends the study (fail-fast).
    r0_dir = packet / "r0"
    r0_dir.mkdir()
    r0: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        out = r0_dir / f"{arm}.json"
        log = r0_dir / f"{arm}.log"
        rc = _run_child(
            ["--_r0", arm, "--_out", str(out.relative_to(project_root))],
            log,
            child_env(packet / "compile_cache" / f"r0_{arm}"),
        )
        rec = json.loads(out.read_text()) if out.exists() else None
        if rec is not None:
            rec["log_sha256"] = _sha256(log)
            rec["op_inventory"] = op_inventory(log.read_text(errors="replace"))
            limit_in_log = [
                m
                for m in RECOMPILE_LIMIT_MARKERS
                if m in log.read_text(errors="replace")
            ]
            if limit_in_log:
                rec["recompile_limit_hit"] = (
                    rec.get("recompile_limit_hit") or limit_in_log
                )
            r0[arm] = rec
        problems = ([f"R0 {arm} child exit {rc}"] if rc else []) + r0_problems(arm, rec)
        if problems:
            record["r0"] = r0
            record["aborted_after"] = f"R0 {arm}"
            return None, problems
    record["r0"] = r0

    r1_dir = packet / "r1"
    r1_dir.mkdir()
    rc = _run_child(
        ["--_r1", str(r1_dir.relative_to(project_root))],
        r1_dir / "stdout.log",
        child_env(packet / "compile_cache" / "r1"),
    )
    if rc or not (r1_dir / "r1.json").exists():
        record["aborted_after"] = "R1"
        return None, [f"R1 worker failed (exit {rc})"]
    r1 = json.loads((r1_dir / "r1.json").read_text())
    record["r1"] = {k: v for k, v in r1.items() if k != "anchor"}
    problems = r1_problems(r1, r0, TOTAL_FRAMES)
    ref_rows, ref_hists = read_pr2l_anchor()
    anchor_bad = anchor_problems(
        r1["anchor"]["rows"],
        r1["anchor"]["per_sequence_histograms"],
        ref_rows,
        ref_hists,
    )
    record["v_anchor"] = {"ok": not anchor_bad, "problems": anchor_bad[:50]}
    problems += anchor_bad
    if problems:
        record["aborted_after"] = "R1 validity"
        return None, problems
    return r1, []


if __name__ == "__main__":
    raise SystemExit(main())
