#!/usr/bin/env python3
"""Run the #465 PR-2L head parity gate (LibTorch head vs PyTorch oracle) and write its packet.

Implements the frozen pre-declaration
``docs/reference/native_runtime_head_parity_libtorch_declaration.md`` (blob
``FROZEN_DECLARATION_BLOB``) without re-deciding anything in it. The study
identity -- declaration, artifact, operator library, driver/runtime,
tolerances and arms -- is fixed in this file and is never a command-line
option, so the runner blob recorded in the packet identifies the study by
itself. PR-2's decision constants and functions are carried over verbatim
(``tests/unit/test_native_head_parity_libtorch_runner.py`` pins them).

* V1 frozen inputs and execution freeze point (§2) — checked before any
  measurement; any mismatch is ``UNRESOLVED``.
* L1 (§4) — per-frame head tensors of C (compiled PyTorch head, primary
  oracle), E (eager PyTorch head) and L (PR-1L TorchScript artifact with the
  ``saccade_native`` scan op, graph executor optimize off) on one shared set of
  TRT backbone features, over all 5316 frames; V2 (a) re-runs the first 20
  frames of every sequence, (b) requires the shared features to be unchanged
  after every frame, (c) requires zero calls into the JIT scan fallbacks.
* L2 (§5) — six runs of the unmodified ``scripts/eval/mot17.py`` through one
  child entry (``runpy``) in the order ``A_C#1, A_L#1, A_N#1, A_C#2, A_L#2,
  A_N#2``. Only the ``A_L`` child injects L, into the harness's existing
  ``_trt_head`` slot. Each run is validated when it finishes (V3 byte
  identity, V4 oracle anchor, V5 injection sidecar) and the first failure
  ends the study.
* Terminal (§6) — first matching of UNRESOLVED, HEAD_PARITY_GROSS_ERROR,
  HEAD_PARITY_EXACT, HEAD_PARITY_WITHIN_TOLERANCE, HEAD_PARITY_OUT_OF_TOLERANCE.

It reads no time quantity and has no smoke mode (§8: no MOT17 frame before
the formal run). The formal run must be the direct child of a
``machine-bench`` lease on a clean tree, on the execution freeze commit::

    .venv/bin/python tools/resctl.py run machine-bench -- \\
        .venv/bin/python scripts/eval/diagnostics/native_head_parity_libtorch.py

Structural checks without MOT17 data:
``native_head_parity_libtorch_structural_check.py``.
"""
# status: experiment

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import platform
import runpy
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# --- frozen by the declaration (§2–§6); change only through §11 ------------
DECLARATION = "docs/reference/native_runtime_head_parity_libtorch_declaration.md"
FROZEN_DECLARATION_BLOB = "361639bd0e565e8f40031720c3b8ccf93189913d"  # incl. §11 A1
# The merge commit of #485 (declaration freeze). mot17.py, src/saccade and the
# preset must be unchanged from it to the execution freeze commit (§2 程式碼).
DECLARATION_FREEZE_COMMIT = "883c16a8471de7996d902de17b545128db9bce52"
UNCHANGED_SINCE_DECLARATION = (
    "scripts/eval/mot17.py",
    "src/saccade",
    "configs/presets/mamba_whole_graph.yaml",
)
# §2 execution freeze point: the runner PR's merge commit, named by an
# annotated tag created after that merge. Compared by peeled commit SHA only.
FREEZE_TAG = "freeze/465-pr2l-libtorch-parity"
FREEZE_REMOTE = "origin"
FREEZE_BRANCH = "origin/main"

ARTIFACT_STEM = "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript"
ARTIFACT = ARTIFACT_STEM + ".pt"
ARTIFACT_SHA256 = "1663ec97022879e8078fb5b9dc3b65f4d2b76e4f8df6d5848300c2b3c999d879"
ARTIFACT_CONTENT_SHA256 = (
    "f6a540edfecd6e421c533c5c2280fc26351b887006c1384bb61a567ce2f39487"
)
ARTIFACT_LINEAGE = ARTIFACT_STEM + ".lineage.json"
ARTIFACT_LINEAGE_SHA256 = (
    "677bc82320657a9ae289e78699c7a6f86e890fc629d1bb8189c08d35b57a277a"
)
EXPORT_TOOL = "scripts/model/export_headline_mamba_head_torchscript.py"
OP_LIBRARY = "build/libsaccade_scan_torchop.so"
OP_LIBRARY_SHA256 = "cfea782f320f641a5ca7752dfc54467909afcca96b6d6d9f884a40cbbe81aa43"
OP_SOURCE = "src/tracking/mamba_scan_torchop.cpp"
OP_SOURCE_BLOB = "6b19e9a359fd04f344b92e7d2c596fea4fcbb505"
FORBIDDEN_LINKS = ("libpython", "libtorch_python")
NATIVE_OP = "saccade_native::selective_scan_fwd"
PYTHON_OP = "saccade::selective_scan_fwd"
NATIVE_SCAN_CALLS = 3
# §11 A1: L is loaded without map_location; parameters/buffers must land on
# cuda:0 and every tensor constant must stay on the CPU (a CUDA constant in the
# traced shape arithmetic makes aten::Int a device->host sync, which
# invalidates CUDA graph capture).
PARAM_DEVICE = "cuda:0"
CONSTANT_DEVICE = "cpu"
# Graph-archive entries that change on every save without changing what runs
# (same rule as the exporter's content_sha256; the test pins the two equal).
CONTENT_EXCLUDED = (".data/serialization_id",)
RUNTIME_REQUIREMENTS = {
    "graph_executor_optimize": False,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": True,
    "matmul_allow_tf32": False,
}
CKPT = "runs/mamba_gt_v14replica_t3_t1/best.ckpt"
CKPT_SHA256 = "c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876"
BACKBONE = "models/yolo/yolo26s_backbone_640_best.engine"
BACKBONE_SHA256 = "2ef3d4d40dfb670982cbbb98e6ed7d07e5b1a590cfa126d5ccf7342ee1579ce4"
PRESET_NAME = "mamba_whole_graph"
PRESET = f"configs/presets/{PRESET_NAME}.yaml"
PRESET_SHA256 = "093b66ed124063f035ae9cf2a76e4f5426743cd819fb66e3e54994c97ea42cd1"
PRESET_FORBIDDEN_KEYS = ("mamba_head_engine", "mamba_trt")
ENVIRONMENT = {
    "torch": "2.11.0+cu130",
    "cudnn": 91900,
    "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    "sm": "12.0",
    "host": "DESKTOP-0FLA6SQ",
}
# §2 NVIDIA driver and CUDA runtime (#485 review R1/R2): exact comparison.
NVIDIA_DRIVER = "616.92"
CU_DRIVER_VERSION = 13040
CUDART_VERSION = 13000
CUDART_RELPATH = ".venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
CUDART_SHA256 = "96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"
RUNTIME_IDENTITY = "docs/reference/runtime_identity.generated.json"
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
PACKET_ROOT = "results/native_head_parity_465_libtorch"

IMG_SIZE = 640
SCORE_FLOOR = 0.05  # base_score_floor = min(conf_threshold, track_thresh)
L1_SCORE_MAX = 0.05
L1_BOX_MAX_PX = 4.0
V2_FRAMES = 20
L1_PAIRS = (("L", "C"), ("L", "E"), ("E", "C"))
L1_DECISION_PAIR = ("L", "C")
# Static shapes of the artifact (PR-1L §1) and of the head outputs.
INPUT_SHAPES = ((1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20))
OUTPUT_SHAPES = (
    (1, 80, 80, 80),
    (1, 80, 40, 40),
    (1, 80, 20, 20),
    (1, 4, 80, 80),
    (1, 4, 40, 40),
    (1, 4, 20, 20),
)

ARMS = {
    "A_C": [],
    "A_L": [],  # no harness option: L is injected by the child (§5.1)
    "A_N": ["--no-compile"],
}
INJECTED_ARMS = ("A_L",)
RUN_ORDER = (("A_C", 1), ("A_L", 1), ("A_N", 1), ("A_C", 2), ("A_L", 2), ("A_N", 2))
FORBIDDEN_ARGV = ("--mamba-head-engine", "--mamba-trt")
L2_METRICS = ("IDF1", "HOTA", "MOTA", "IDs")
TOL_FLOOR = {"IDF1": 0.20, "HOTA": 0.20, "MOTA": 0.20, "IDs": 5.0}
TOL_CAP = {"IDF1": 1.00, "HOTA": 1.00, "MOTA": 1.00, "IDs": 30.0}

TERMINALS = (
    "UNRESOLVED",
    "HEAD_PARITY_GROSS_ERROR",
    "HEAD_PARITY_EXACT",
    "HEAD_PARITY_WITHIN_TOLERANCE",
    "HEAD_PARITY_OUT_OF_TOLERANCE",
)

# §5.3 V4: PR-2R packet 20260927T151100Z, A_C_1.
ORACLE_ANCHOR_TXT_SHA256 = {
    "MOT17-02-SDP": "d426ca1b61ae1441b94cc3269a6fee90f01ca7ec3a649755f170c767b7515328",
    "MOT17-04-SDP": "cb55746fdc059fae12a6efcbce5cfa00b40fdb4f23b8766ee021a8b5b1602d7b",
    "MOT17-05-SDP": "ea46b483046879ea28bae14f25f7ec251fea6359e6d850592504ee09f3f4d8dc",
    "MOT17-09-SDP": "da0a74836843293da0e45923905bf25f31d68145b71771a1da043cdf1e576df4",
    "MOT17-10-SDP": "587f2f05bfe9caacf57fba2cd5a6828b1b548ddcaef3d155c96055844bf92705",
    "MOT17-11-SDP": "38dd77309e98a88b3abf827d18b087b6f025bef176b47c280a4961f3013247b0",
    "MOT17-13-SDP": "4c93f75e6e162e7ab6007a746fc7da97250d54b01d8bd63500a7b07a5af73137",
}
# §5.4 report-only byte identity against earlier packets (never gates).
PRIOR_REFERENCE_TXT = {
    "A_N_vs_PR2R_A_N": (
        "A_N",
        "results/native_head_parity_465_tf32_off/20260927T151100Z/l2/A_N_1",
    ),
    "A_L_vs_r2_R_E": (
        "A_L",
        "results/native_head_failure_localization_465_r2/20260928T081021Z/l2/R_E_1",
    ),
}

# The PyTorch head entry points and the two JIT scan fallbacks (§3) that the
# A_L child and the L1 worker replace with raising guards.
HEAD_GUARDS = ("forward", "_forward_eager")
JIT_FALLBACK_GUARDS = ("_selective_scan_jit", "_selective_scan_legacy_n1_jit")
MAMBA_HEAD_MODULE = "saccade.perception.temporal_yolo.mamba_head"
A_L_INSTALLS = (
    "op_library",
    "graph_executor_optimize=false",
    "torchscript_artifact",
    "mamba_gated_detector.build_mamba_gated_detector",
    *(f"{MAMBA_HEAD_MODULE}.{g}" for g in JIT_FALLBACK_GUARDS),
)
BUILD_PRECONDITIONS = {
    "trt_head_engine": "",
    "trt_head_is_none": True,
    "use_whole_graph": True,
    "use_detail_fusion": False,
}

# |Δ| histogram for the report-only 99.9th percentile: a zero bin plus 100
# log-spaced bins per decade over [1e-10, 1e3); the reported percentile is the
# upper edge of the bin that holds it (conservative, never below the truth).
HIST_LO_EXP, HIST_HI_EXP, HIST_PER_DECADE = -10, 3, 100


# --------------------------------------------------------------------------
# pure decision logic (unit-tested without a GPU; verbatim from PR-2)
# --------------------------------------------------------------------------
def tolerance(delta_n: float, metric: str) -> float:
    """b_m = min(max(|Δ_N,m|, floor_m), cap_m) (declaration §5)."""
    return min(max(abs(delta_n), TOL_FLOOR[metric]), TOL_CAP[metric])


def l1_verdict(score_maxabs: float, box_maxabs_px: float) -> str:
    """κ_L1 on the (T,C) pair; NaN or inf fails (a non-finite Δ is not a pass)."""
    ok = (
        math.isfinite(score_maxabs)
        and math.isfinite(box_maxabs_px)
        and score_maxabs <= L1_SCORE_MAX
        and box_maxabs_px <= L1_BOX_MAX_PX
    )
    return "L1_PASS" if ok else "L1_GROSS_ERROR"


def l2_verdict(
    metrics_c: dict[str, float],
    metrics_t: dict[str, float],
    metrics_n: dict[str, float],
    exact: bool,
) -> tuple[str, dict[str, dict[str, float | bool]]]:
    """κ_L2: EXACT if the A_T and A_C txt are identical, else WITHIN/OUT (two-sided)."""
    detail: dict[str, dict[str, float | bool]] = {}
    for m in L2_METRICS:
        d_t = metrics_t[m] - metrics_c[m]
        d_n = metrics_n[m] - metrics_c[m]
        b = tolerance(d_n, m)
        detail[m] = {
            "C": metrics_c[m],
            "T": metrics_t[m],
            "N": metrics_n[m],
            "delta_T": d_t,
            "delta_N": d_n,
            "tolerance": b,
            "within": abs(d_t) <= b,
        }
    if exact:
        return "L2_EXACT", detail
    within = all(bool(v["within"]) for v in detail.values())
    return ("L2_WITHIN" if within else "L2_OUT"), detail


def decide_terminal(validity_ok: bool, l1: str | None, l2: str | None) -> str:
    """Declaration §6, evaluated in order; anything missing is UNRESOLVED."""
    if not validity_ok or l1 is None or l2 is None:
        return "UNRESOLVED"
    if l1 == "L1_GROSS_ERROR":
        return "HEAD_PARITY_GROSS_ERROR"
    if l1 != "L1_PASS":
        return "UNRESOLVED"
    return {
        "L2_EXACT": "HEAD_PARITY_EXACT",
        "L2_WITHIN": "HEAD_PARITY_WITHIN_TOLERANCE",
        "L2_OUT": "HEAD_PARITY_OUT_OF_TOLERANCE",
    }.get(l2, "UNRESOLVED")


def metrics_from_counts(
    counts: dict[str, int], hota: dict[str, float]
) -> dict[str, float]:
    """Unrounded percentages from summed motmetrics counts (the formulas of
    ``metrics._format_overall_metrics_from_counts``) plus TrackEval HOTA."""
    idtp, idfp, idfn = counts["idtp"], counts["idfp"], counts["idfn"]
    idf1_den = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / idf1_den) if idf1_den > 0 else 0.0
    fp, fn, ids = (
        counts["num_false_positives"],
        counts["num_misses"],
        counts["num_switches"],
    )
    mota = 1.0 - (fn + fp + ids) / max(counts["num_objects"], 1)
    return {
        "IDF1": idf1 * 100.0,
        "HOTA": hota["HOTA"] * 100.0,
        "MOTA": mota * 100.0,
        "IDs": float(ids),
        "DetA": hota["DetA"] * 100.0,
        "AssA": hota["AssA"] * 100.0,
        "FP": float(fp),
        "FN": float(fn),
    }


def first_divergent_frame(a: bytes, b: bytes) -> int | None:
    """First MOT frame whose ordered output rows differ; None when identical."""
    if a == b:
        return None

    def by_frame(data: bytes) -> dict[int, list[bytes]]:
        rows: dict[int, list[bytes]] = {}
        for line in data.splitlines():
            if line.strip():
                rows.setdefault(int(float(line.split(b",", 1)[0])), []).append(line)
        return rows

    ra, rb = by_frame(a), by_frame(b)
    for frame in sorted(set(ra) | set(rb)):
        if ra.get(frame) != rb.get(frame):
            return frame
    return -1  # same rows per frame, different bytes (e.g. trailing whitespace)


def hist_edges() -> list[float]:
    n = (HIST_HI_EXP - HIST_LO_EXP) * HIST_PER_DECADE
    return [10.0 ** (HIST_LO_EXP + i / HIST_PER_DECADE) for i in range(n + 1)]


def hist_quantile(counts: list[int], q: float) -> float | None:
    """Upper edge of the bin holding quantile q; bin 0 is exact zero, the last
    bin is overflow (returned as inf)."""
    total = sum(counts)
    if total == 0:
        return None
    edges = hist_edges()
    target = math.ceil(q * total)
    running = 0
    for i, c in enumerate(counts):
        running += c
        if running >= target:
            if i == 0:
                return 0.0
            return edges[i - 1] if i - 1 < len(edges) else math.inf
    return math.inf


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
    """§4 V1 execution freeze point; every comparison is by commit SHA."""
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


def parse_ls_remote_peeled(stdout: str) -> str | None:
    """The peeled SHA from ``git ls-remote <remote> refs/tags/<tag>^{}``."""
    want = f"refs/tags/{FREEZE_TAG}^{{}}"
    hits = [
        line.split("\t", 1)[0]
        for line in stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[1] == want
    ]
    return hits[0] if len(hits) == 1 else None


def maps_paths(maps_text: str, basename_prefix: str) -> list[str]:
    """Pathnames in ``/proc/self/maps`` whose basename starts with the prefix
    (one entry per mapping line; duplicates are expected)."""
    out = []
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or not parts[5].startswith("/"):
            continue
        path = parts[5].removesuffix(" (deleted)")
        if os.path.basename(path).startswith(basename_prefix):
            out.append(path)
    return out


def backing_files(
    paths: Iterable[str], stat: Callable[[str], Any]
) -> dict[tuple[int, int], list[str]]:
    """§2 R2: deduplicate mapped paths by the ``(st_dev, st_ino)`` of the file
    behind them; several segments (or names) of one ELF are one backing file."""
    files: dict[tuple[int, int], list[str]] = {}
    for p in paths:
        st = stat(p)
        key = (int(st.st_dev), int(st.st_ino))
        if p not in files.setdefault(key, []):
            files[key].append(p)
    return files


def driver_runtime_problems(obs: dict[str, Any]) -> list[str]:
    """§2 NVIDIA driver and CUDA runtime: every item compared exactly."""
    problems = []
    if obs.get("nvidia_driver") != NVIDIA_DRIVER:
        problems.append(
            f"NVIDIA driver {obs.get('nvidia_driver')!r} != {NVIDIA_DRIVER!r}"
        )
    if obs.get("cu_driver_version") != CU_DRIVER_VERSION:
        problems.append(
            f"cuDriverGetVersion {obs.get('cu_driver_version')} != {CU_DRIVER_VERSION}"
        )
    if obs.get("cudart_version") != CUDART_VERSION:
        problems.append(
            f"cudaRuntimeGetVersion {obs.get('cudart_version')} != {CUDART_VERSION}"
        )
    files = obs.get("libcudart_backing_files") or []
    if len(files) != 1:
        problems.append(f"{len(files)} libcudart backing files mapped, expected 1")
    else:
        f = files[0]
        if f.get("relpath") != CUDART_RELPATH:
            problems.append(
                f"libcudart realpath {f.get('relpath')!r} != {CUDART_RELPATH!r}"
            )
        if f.get("sha256") != CUDART_SHA256:
            problems.append(f"libcudart sha256 {f.get('sha256')} != {CUDART_SHA256}")
    return problems


def runtime_requirement_problems(readback: dict[str, Any]) -> list[str]:
    """§3: the four runtime requirements read back at every L call."""
    return [
        f"{k} = {readback.get(k)!r}, required {v!r}"
        for k, v in RUNTIME_REQUIREMENTS.items()
        if readback.get(k) is not v
    ]


def graph_problems(graph_text: str) -> list[str]:
    """Fail-closed checks on the traced graph (inlined, as text)."""
    problems = []
    if "prim::PythonOp" in graph_text:
        problems.append("graph contains prim::PythonOp")
    if PYTHON_OP in graph_text:
        problems.append(f"graph calls the Python op {PYTHON_OP}")
    if NATIVE_OP not in graph_text:
        problems.append(f"graph never calls {NATIVE_OP}")
    return problems


def placement_problems(
    param_devices: list[str] | None, constant_devices: list[str] | None
) -> list[str]:
    """§11 A1: parameters/buffers on cuda:0, every tensor constant on the CPU."""
    problems = []
    if not param_devices:
        problems.append("no parameter/buffer devices recorded")
    elif set(param_devices) != {PARAM_DEVICE}:
        problems.append(
            f"parameter/buffer devices {sorted(set(param_devices))} != {PARAM_DEVICE}"
        )
    if constant_devices is None:
        problems.append("tensor constant devices not recorded")
    elif any(d != CONSTANT_DEVICE for d in constant_devices):
        problems.append(
            f"tensor constants on {sorted(set(constant_devices))}, required {CONSTANT_DEVICE}"
        )
    return problems


def artifact_problems(record: dict[str, Any]) -> list[str]:
    """§2 artifact identity and traced-graph conditions, as loaded."""
    problems = []
    if record.get("sha256") != ARTIFACT_SHA256:
        problems.append(f"artifact sha256 {record.get('sha256')} != {ARTIFACT_SHA256}")
    if record.get("content_sha256") != ARTIFACT_CONTENT_SHA256:
        problems.append(
            f"artifact content_sha256 {record.get('content_sha256')} "
            f"!= {ARTIFACT_CONTENT_SHA256}"
        )
    problems += record.get("graph_problems", ["graph not checked"])
    if record.get("native_scan_calls") != NATIVE_SCAN_CALLS:
        problems.append(
            f"{record.get('native_scan_calls')} {NATIVE_OP} calls, "
            f"expected {NATIVE_SCAN_CALLS}"
        )
    problems += placement_problems(
        record.get("param_devices"), record.get("tensor_constant_devices")
    )
    return problems


def sidecar_problems(arm: str, sc: dict[str, Any] | None) -> list[str]:
    """§5.3 V5 for one L2 child from its sidecar."""
    if sc is None:
        return ["sidecar missing"]
    problems = []
    if sc.get("sidecar_error"):
        problems.append(f"sidecar observer failed: {sc['sidecar_error']}")
    if sc.get("arm") != arm:
        problems.append(f"sidecar arm {sc.get('arm')!r} != {arm!r}")
    if not sc.get("runpy_completed"):
        problems.append("mot17.py did not run to completion under the child entry")
    problems += [
        f"driver/runtime: {p}"
        for p in driver_runtime_problems(sc.get("driver_runtime") or {})
    ]
    if arm not in INJECTED_ARMS:
        if sc.get("installed") != []:
            problems.append(f"non-injected arm installed {sc.get('installed')}")
        if sc.get("op_library_mapped") is not False:
            problems.append("operator library mapped in a non-injected arm")
        return problems
    if sc.get("installed") != list(A_L_INSTALLS):
        problems.append(f"installed {sc.get('installed')} != {list(A_L_INSTALLS)}")
    if sc.get("op_library_mapped") is not True:
        problems.append("operator library not mapped in the injected arm")
    if sc.get("op_library_sha256") != OP_LIBRARY_SHA256:
        problems.append(
            f"op library sha256 {sc.get('op_library_sha256')} != {OP_LIBRARY_SHA256}"
        )
    problems += [f"artifact: {p}" for p in artifact_problems(sc.get("artifact") or {})]
    if sc.get("build_wrapper_calls") != 1:
        problems.append(
            f"build wrapper called {sc.get('build_wrapper_calls')} times, expected 1"
        )
    if sc.get("build_preconditions") != BUILD_PRECONDITIONS:
        problems.append(
            f"build preconditions {sc.get('build_preconditions')} != {BUILD_PRECONDITIONS}"
        )
    if sc.get("slot_installed") is not True:
        problems.append("adapter not installed in the _trt_head slot")
    calls = sc.get("adapter_calls") or 0
    if calls < 1:
        problems.append("adapter never called")
    if sc.get("adapter_problems"):
        problems.append(f"adapter problems: {sc['adapter_problems'][:5]}")
    guards = sc.get("guard_calls") or {}
    expected_guards = set(HEAD_GUARDS) | set(JIT_FALLBACK_GUARDS)
    if set(guards) != expected_guards:
        problems.append(f"guards {sorted(guards)} != {sorted(expected_guards)}")
    fired = {k: v for k, v in guards.items() if v}
    if fired:
        problems.append(f"guards called: {fired}")
    return problems


def env_override_problems(
    this: dict[str, Any] | None, first: dict[str, Any] | None
) -> list[str]:
    """§2 env hatch: the child-side ``resolved_env_overrides()`` of every run
    must be recorded and equal to that of the first run (A_C#1)."""
    if not isinstance(this, dict) or not this:
        return ["child resolved_env_overrides not recorded"]
    if not isinstance(first, dict) or not first:
        return ["first run's child resolved_env_overrides not recorded"]
    if this != first:
        diff = sorted(
            k for k in this.keys() | first.keys() if this.get(k) != first.get(k)
        )
        return [f"child resolved_env_overrides differ from the first run on {diff}"]
    return []


def argv_problems(cmdline: list[str] | None, expected: list[str]) -> list[str]:
    """The harness run_manifest cmdline is the arm's frozen argv, with no
    head-engine option."""
    problems = []
    if cmdline != expected:
        problems.append(f"run_manifest cmdline {cmdline} != {expected}")
    bad = [a for a in (cmdline or []) if a.split("=", 1)[0] in FORBIDDEN_ARGV]
    if bad:
        problems.append(f"run_manifest cmdline carries {bad}")
    return problems


def oracle_anchor_problems(txt_sha256: dict[str, str | None]) -> list[str]:
    """§5.3 V4: A_C#1 equals the PR-2R oracle, sequence by sequence."""
    return [
        f"V4: A_C#1 {s} sha256 {txt_sha256.get(s)} != PR-2R A_C_1 {want}"
        for s, want in ORACLE_ANCHOR_TXT_SHA256.items()
        if txt_sha256.get(s) != want
    ]


def l1_problems(l1: dict[str, Any], expected_frames: int) -> list[str]:
    """§4 V2 (a)(b)(c) and the L1 process's own runtime checks."""
    problems = []
    if l1.get("frames") != expected_frames:
        problems.append(
            f"L1 covered {l1.get('frames')} frames, expected {expected_frames}"
        )
    v2 = l1.get("v2") or {}
    if v2.get("rerun_failures") != []:
        problems.append(f"V2(a): re-run mismatches {v2.get('rerun_failures')}")
    if v2.get("input_mutations") != []:
        problems.append(f"V2(b): shared features changed {v2.get('input_mutations')}")
    if v2.get("jit_fallback_calls") != {g: 0 for g in JIT_FALLBACK_GUARDS}:
        problems.append(
            f"V2(c): JIT scan fallback calls {v2.get('jit_fallback_calls')}"
        )
    problems += [
        f"L1 driver/runtime: {p}"
        for p in driver_runtime_problems(l1.get("driver_runtime") or {})
    ]
    problems += [
        f"L1 artifact: {p}" for p in artifact_problems(l1.get("artifact") or {})
    ]
    if l1.get("l_call_problems"):
        problems.append(f"L1 L-call problems: {l1['l_call_problems'][:5]}")
    return problems


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_or_none(*args: str) -> str | None:
    try:
        return _git(*args)
    except subprocess.CalledProcessError:
        return None


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _seq_frames(seq: str, data_root: str = DATA_ROOT) -> list[Path]:
    return sorted((project_root / data_root / SPLIT / seq / "img1").glob("*.jpg"))


def _saccade_env() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("SACCADE_")}


def _dig(obj: Any, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _sha_or_none(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _blob_or_none(rel: str) -> str | None:
    try:
        return _git("rev-parse", f"HEAD:{rel}")
    except subprocess.CalledProcessError:
        return None


def content_sha256(path: Path) -> str:
    """Portable identity of a TorchScript archive: sha256 over its entries
    (name without the archive root folder, then bytes), sorted by name,
    excluding ``.data/serialization_id`` (random per save) and ``*.debug_pkl``
    (the tracing call stack with absolute paths). Neither affects execution."""
    import hashlib
    import zipfile

    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        items = []
        for info in z.infolist():
            name = info.filename.split("/", 1)[1]
            if name in CONTENT_EXCLUDED or name.endswith(".debug_pkl"):
                continue
            items.append((name, z.read(info)))
    for name, data in sorted(items):
        h.update(name.encode() + b"\0" + len(data).to_bytes(8, "little") + data)
    return h.hexdigest()


def needed_libraries(path: Path) -> list[str]:
    out = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [
        line.split("[", 1)[1].rstrip("]").strip()
        for line in out.splitlines()
        if "(NEEDED)" in line
    ]


def held_lease() -> dict[str, Any] | None:
    """The resctl lease whose owner is this process's parent (the ``resctl run``)."""
    out = subprocess.run(
        [sys.executable, "tools/resctl.py", "status", "--json"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    for row in json.loads(out.stdout).get("leases", []):
        owner = row.get("owner") or {}
        if row.get("state") == "BUSY" and owner.get("pid") == os.getppid():
            return {
                "resource": row.get("resource"),
                "pid": owner.get("pid"),
                "start_time": owner.get("start_time"),
                "head": owner.get("head"),
                "command_str": owner.get("command_str"),
            }
    return None


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
        "tag_remote_peeled": parse_ls_remote_peeled(ls.stdout)
        if ls.returncode == 0
        else None,
        "ls_remote_returncode": ls.returncode,
    }


# --------------------------------------------------------------------------
# in-process probes (need torch/CUDA initialised by the caller)
# --------------------------------------------------------------------------
def probe_driver_runtime() -> dict[str, Any]:
    """§2 NVIDIA driver and CUDA runtime, read in this process. The runtime
    version is asked of the mapped libcudart itself (opened by its absolute
    path, which dlopen resolves to the already-loaded object)."""
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    drivers = smi.stdout.strip().splitlines() if smi.returncode == 0 else []
    cu = ctypes.c_int(-1)
    try:
        rc_cu = ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(cu))
    except OSError:
        rc_cu = None
    maps = Path("/proc/self/maps").read_text()
    cudart = backing_files(maps_paths(maps, "libcudart.so"), os.stat)
    libcuda = backing_files(maps_paths(maps, "libcuda.so"), os.stat)

    def describe(paths: list[str]) -> dict[str, Any]:
        real = os.path.realpath(paths[0])
        return {
            "mapped_as": paths,
            "realpath": real,
            "relpath": os.path.relpath(real, project_root),
            "sha256": _sha256(Path(real)),
        }

    rt = ctypes.c_int(-1)
    rc_rt = None
    if len(cudart) == 1:
        only = next(iter(cudart.values()))[0]
        rc_rt = ctypes.CDLL(only).cudaRuntimeGetVersion(ctypes.byref(rt))
    return {
        "nvidia_driver": drivers[0].strip() if len(drivers) == 1 else drivers or None,
        "cu_driver_version": cu.value if rc_cu == 0 else None,
        "cudart_version": rt.value if rc_rt == 0 else None,
        "libcudart_backing_files": [
            {"dev_ino": list(k), **describe(v)} for k, v in cudart.items()
        ],
        "libcuda_backing_files_recorded": [
            {"dev_ino": list(k), **describe(v)} for k, v in libcuda.items()
        ],
    }


def runtime_readback(torch: Any) -> dict[str, Any]:
    return {
        "graph_executor_optimize": torch._C._get_graph_executor_optimize(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def op_library_mapped() -> bool:
    maps = Path("/proc/self/maps").read_text()
    return bool(maps_paths(maps, Path(OP_LIBRARY).name))


def load_l(torch: Any) -> tuple[Any, dict[str, Any]]:
    """§5.1 steps 1–3: operator library, optimize off, artifact (re-verified)."""
    op = project_root / OP_LIBRARY
    op_sha = _sha256(op)
    if op_sha != OP_LIBRARY_SHA256:
        raise RuntimeError(f"{OP_LIBRARY} sha256 {op_sha} != {OP_LIBRARY_SHA256}")
    torch.ops.load_library(str(op))
    torch._C._set_graph_executor_optimize(False)
    path = project_root / ARTIFACT
    record: dict[str, Any] = {
        "path": ARTIFACT,
        "sha256": _sha256(path),
        "content_sha256": content_sha256(path),
    }
    # §11 A1: no map_location (parameters are saved on cuda:0; tensor
    # constants stay on the CPU as saved).
    module = torch.jit.load(str(path)).eval()
    record.update(inspect_loaded(module))
    bad = artifact_problems(record)
    if bad:
        raise RuntimeError(f"artifact rejected: {bad}")
    return module, {**record, "op_library_sha256": op_sha}


def inspect_loaded(module: Any) -> dict[str, Any]:
    """Traced-graph conditions (§2) and placement (§11 A1) of a loaded L."""
    graph = module.inlined_graph
    graph_text = str(graph)
    return {
        "graph_problems": graph_problems(graph_text),
        "native_scan_calls": graph_text.count(NATIVE_OP),
        "param_devices": sorted(
            {str(t.device) for t in (*module.parameters(), *module.buffers())}
        ),
        "tensor_constant_devices": [
            str(n.t("value").device)
            for n in graph.nodes()
            if n.kind() == "prim::Constant" and n.output().type().kind() == "TensorType"
        ],
    }


class Guard:
    """Replaces a callable that must never run; counts and raises."""

    def __init__(self, name: str, counts: dict[str, int]) -> None:
        self.name = name
        self.counts = counts
        counts[name] = 0

    def __call__(self, *_a: Any, **_k: Any) -> Any:
        self.counts[self.name] += 1
        raise RuntimeError(f"guard: {self.name} called while L is under test")


def install_jit_fallback_guards(counts: dict[str, int]) -> None:
    import importlib

    mh = importlib.import_module(MAMBA_HEAD_MODULE)
    for name in JIT_FALLBACK_GUARDS:
        if not hasattr(mh, name):
            raise RuntimeError(f"{MAMBA_HEAD_MODULE}.{name} not found")
        setattr(mh, name, Guard(name, counts))


class LibTorchHeadAdapter:
    """§5.1 step 5: the ``_trt_head`` slot's interface around L. No copy,
    dtype change, ``.contiguous()`` or arithmetic; L runs on the current stream."""

    def __init__(self, module: Any, torch: Any) -> None:
        self.module = module
        self.torch = torch
        self.calls = 0
        self.problems: list[str] = []
        self.shapes_seen: set[tuple[tuple[int, ...], ...]] = set()

    def _fail(self, msg: str) -> None:
        self.problems.append(msg)
        raise RuntimeError(f"LibTorchHeadAdapter: {msg}")

    def infer_graph(self, p3: Any, p4: Any, p5: Any) -> tuple[list[Any], list[Any]]:
        torch = self.torch
        self.calls += 1
        bad = runtime_requirement_problems(runtime_readback(torch))
        if bad:
            self._fail(f"call {self.calls}: {bad}")
        feats = (p3, p4, p5)
        shapes = tuple(tuple(int(d) for d in p.shape) for p in feats)
        self.shapes_seen.add(shapes)
        if shapes != INPUT_SHAPES or not all(
            p.is_cuda and p.dtype == torch.float32 for p in feats
        ):
            self._fail(
                f"call {self.calls}: inputs {shapes} not CUDA float32 {INPUT_SHAPES}"
            )
        outs = self.module(p3, p4, p5)
        if not isinstance(outs, tuple) or len(outs) != 6:
            self._fail(f"call {self.calls}: {type(outs).__name__} is not 6 outputs")
        got = tuple(tuple(int(d) for d in o.shape) for o in outs)
        if got != OUTPUT_SHAPES or not all(
            o.is_cuda and o.dtype == torch.float32 for o in outs
        ):
            self._fail(
                f"call {self.calls}: outputs {got} not CUDA float32 {OUTPUT_SHAPES}"
            )
        return [outs[0], outs[1], outs[2]], [outs[3], outs[4], outs[5]]

    infer = infer_graph


# --------------------------------------------------------------------------
# V1 — frozen inputs and execution freeze point (declaration §2)
# --------------------------------------------------------------------------
def check_v1() -> tuple[bool, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(item: str, ok: bool, observed: Any, expected: Any = None) -> None:
        checks.append(
            {"item": item, "ok": bool(ok), "observed": observed, "expected": expected}
        )

    lineage_path = project_root / ARTIFACT_LINEAGE
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads(lineage_path.read_text())
    except Exception as exc:  # noqa: BLE001 — recorded, fails closed
        check("lineage manifest readable", False, repr(exc), ARTIFACT_LINEAGE)
    get = lambda *keys: _dig(manifest, keys)  # noqa: E731

    art = project_root / ARTIFACT
    art_sha = _sha_or_none(art)
    check("artifact file sha256", art_sha == ARTIFACT_SHA256, art_sha, ARTIFACT_SHA256)
    art_content = content_sha256(art) if art.exists() else None
    check(
        "artifact content_sha256",
        art_content == ARTIFACT_CONTENT_SHA256,
        art_content,
        ARTIFACT_CONTENT_SHA256,
    )
    lineage_sha = _sha_or_none(lineage_path)
    check(
        "lineage manifest file sha256",
        lineage_sha == ARTIFACT_LINEAGE_SHA256,
        lineage_sha,
        ARTIFACT_LINEAGE_SHA256,
    )
    expected_manifest = {
        ("torchscript", "sha256"): ARTIFACT_SHA256,
        ("torchscript", "content_sha256"): ARTIFACT_CONTENT_SHA256,
        ("op_library", "sha256"): OP_LIBRARY_SHA256,
        ("runtime_requirements",): RUNTIME_REQUIREMENTS,
        ("tool", "git_dirty"): False,
        ("companions", "backbone_engine", "sha256"): BACKBONE_SHA256,
        ("preset", "sha256"): PRESET_SHA256,
        ("environment", "torch"): ENVIRONMENT["torch"],
        ("environment", "cudnn"): ENVIRONMENT["cudnn"],
        ("environment", "gpu"): ENVIRONMENT["gpu"],
        ("environment", "sm"): ENVIRONMENT["sm"],
        ("environment", "host"): ENVIRONMENT["host"],
    }
    for keys, want in expected_manifest.items():
        got = get(*keys)
        check(f"manifest {'.'.join(keys)}", got == want, got, want)

    export_check = subprocess.run(
        [sys.executable, EXPORT_TOOL, "--check"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    last = export_check.stdout.strip().splitlines()[-1:] or [""]
    check(
        "export_headline_mamba_head_torchscript.py --check",
        export_check.returncode == 0 and last[0].startswith("OK"),
        {"returncode": export_check.returncode, "last_line": last[0]},
        "exit 0, OK",
    )

    op = project_root / OP_LIBRARY
    op_sha = _sha_or_none(op)
    check(
        "operator library sha256",
        op_sha == OP_LIBRARY_SHA256,
        op_sha,
        OP_LIBRARY_SHA256,
    )
    needed = needed_libraries(op) if op.exists() else None
    check(
        "operator library DT_NEEDED has no Python",
        needed is not None and not any(n.startswith(FORBIDDEN_LINKS) for n in needed),
        needed,
        f"no {FORBIDDEN_LINKS}",
    )
    src_blob = _blob_or_none(OP_SOURCE)
    check(
        f"HEAD {OP_SOURCE} blob", src_blob == OP_SOURCE_BLOB, src_blob, OP_SOURCE_BLOB
    )

    ckpt_sha = _sha_or_none(project_root / CKPT)
    check("checkpoint sha256", ckpt_sha == CKPT_SHA256, ckpt_sha, CKPT_SHA256)
    bb_sha = _sha_or_none(project_root / BACKBONE)
    check("backbone engine sha256", bb_sha == BACKBONE_SHA256, bb_sha, BACKBONE_SHA256)
    preset_path = project_root / PRESET
    preset_sha = _sha_or_none(preset_path)
    check("preset sha256", preset_sha == PRESET_SHA256, preset_sha, PRESET_SHA256)
    import yaml

    preset = yaml.safe_load(preset_path.read_text()) if preset_path.exists() else None
    set_keys = [
        k for k in PRESET_FORBIDDEN_KEYS if isinstance(preset, dict) and preset.get(k)
    ]
    check(
        "preset sets no head engine",
        isinstance(preset, dict) and not set_keys,
        set_keys,
        [],
    )

    porcelain = _git("status", "--porcelain")
    check("clean tree", porcelain == "", porcelain or "(clean)", "(clean)")
    changed = _git(
        "diff",
        "--name-only",
        DECLARATION_FREEZE_COMMIT,
        "HEAD",
        "--",
        *UNCHANGED_SINCE_DECLARATION,
    )
    check(
        "harness, src/saccade and preset unchanged since the declaration freeze",
        changed == "",
        changed.splitlines(),
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

    import torch

    torch.zeros(1, device="cuda")  # maps libcudart/libcuda for the probe
    cap = torch.cuda.get_device_capability(0)
    env = {
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"{cap[0]}.{cap[1]}",
        "host": platform.node(),
    }
    for k, want in ENVIRONMENT.items():
        check(f"environment {k}", env[k] == want, env[k], want)
    defaults = runtime_readback(torch)
    defaults_expected = {**RUNTIME_REQUIREMENTS, "graph_executor_optimize": True}
    check(
        "torch defaults (the harness sets none; the L child turns optimize off)",
        defaults == defaults_expected,
        defaults,
        defaults_expected,
    )
    try:
        _, loaded = load_l(torch)
        loaded_bad = artifact_problems(loaded)
    except Exception as exc:  # noqa: BLE001 — recorded, fails closed
        loaded, loaded_bad = {"error": repr(exc)}, [repr(exc)]
    check(
        "artifact as loaded: traced graph (§2) and placement (§11 A1)",
        not loaded_bad,
        {**loaded, "problems": loaded_bad},
        f"{NATIVE_OP} x{NATIVE_SCAN_CALLS}, no Python op; params {PARAM_DEVICE}, "
        f"tensor constants {CONSTANT_DEVICE}",
    )
    driver = probe_driver_runtime()
    driver_bad = driver_runtime_problems(driver)
    check(
        "NVIDIA driver and CUDA runtime (exact)",
        not driver_bad,
        {**driver, "problems": driver_bad},
        {
            "nvidia_driver": NVIDIA_DRIVER,
            "cu_driver_version": CU_DRIVER_VERSION,
            "cudart_version": CUDART_VERSION,
            "libcudart": {"relpath": CUDART_RELPATH, "sha256": CUDART_SHA256},
        },
    )

    saccade_env = _saccade_env()
    check("no SACCADE_* set by the caller", not saccade_env, saccade_env, {})

    frames = {s: len(_seq_frames(s)) for s in SEQUENCES}
    gts = {
        s: (project_root / DATA_ROOT / SPLIT / s / "gt" / "gt.txt").exists()
        for s in SEQUENCES
    }
    check(
        "data sequences and frame count",
        sum(frames.values()) == TOTAL_FRAMES
        and all(frames.values())
        and all(gts.values()),
        {"frames": frames, "total": sum(frames.values()), "gt": gts},
        {"sequences": list(SEQUENCES), "total": TOTAL_FRAMES},
    )

    lease = held_lease()
    check(
        f"direct child of a {LEASE} lease",
        lease is not None and lease.get("resource") == LEASE,
        lease,
        f"{LEASE} held by parent pid {os.getppid()}",
    )

    # Recorded, not gated (§2 runtime identity): CI on the freeze commit is the gate.
    ri_blob = _blob_or_none(RUNTIME_IDENTITY)
    try:
        ri_coordinate = json.loads((project_root / RUNTIME_IDENTITY).read_text())[
            "coordinate"
        ]
    except Exception as exc:  # noqa: BLE001 — recorded only
        ri_coordinate = {"error": repr(exc)}

    ok = all(c["ok"] for c in checks)
    return ok, {
        "ok": ok,
        "checks": checks,
        "git": blobs,
        "freeze": freeze,
        "lease": lease,
        "runtime_identity": {"blob": ri_blob, "coordinate": ri_coordinate},
    }


# --------------------------------------------------------------------------
# L1 worker (runs in its own process; declaration §4)
# --------------------------------------------------------------------------
def run_l1_worker(
    out_dir: Path, sequences: tuple[str, ...], data_root: str = DATA_ROOT
) -> int:
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    import saccade_tracking_ext  # noqa: F401  (before torchvision; see export tool)

    import torch
    import torch.nn.functional as F
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        _dfl_decode,
        _dist2bbox_xywh,
        build_mamba_gated_detector,
    )

    guard_counts: dict[str, int] = {}
    install_jit_fallback_guards(guard_counts)
    head_l, artifact = load_l(torch)
    driver = probe_driver_runtime()

    export = _load_trt_export_tool()
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
    backbone = det._trt_backbone
    anchors_t = det._whole_graph_anchors.T.unsqueeze(0)
    astrides = det._whole_graph_anchor_strides.squeeze(-1).unsqueeze(0)
    # E: an independent eager instance of the same state dict, taken before C
    # is compiled so that no instance ever toggles compile.
    head_e = copy.deepcopy(det.mamba_head).eval()
    head_c = det.mamba_head
    head_c.set_head_compile(True)
    head_c.set_block_compile(True)
    adapter = LibTorchHeadAdapter(head_l, torch)

    def run_torch(head: Any, feats: list[Any]) -> list[Any]:
        cls, reg = head._forward_eager(list(feats), return_embeddings=False)
        return [t.clone() for t in (*cls, *reg)]

    def run_l(feats: list[Any]) -> list[Any]:
        cls, reg = adapter.infer_graph(*feats)
        return [t.clone() for t in (*cls, *reg)]

    runners = {
        "C": lambda f: run_torch(head_c, f),
        "E": lambda f: run_torch(head_e, f),
        "L": run_l,
    }

    def decode(outs: list[Any], sx: float, sy: float) -> tuple[Any, Any]:
        cls_all = torch.cat([c.flatten(2) for c in outs[:3]], dim=2)[0]
        reg_all = torch.cat([r.flatten(2) for r in outs[3:]], dim=2)
        # the anchor/stride decode of _postprocess_mamba_fixed_eager
        bboxes = _dist2bbox_xywh(_dfl_decode(reg_all), anchors_t, dim=1) * astrides
        xywh = bboxes[0].T  # (N, 4)
        xyxy = torch.cat(
            [xywh[:, :2] - xywh[:, 2:4] / 2, xywh[:, :2] + xywh[:, 2:4] / 2], 1
        )
        scale = torch.tensor([sx, sy, sx, sy], dtype=xyxy.dtype, device=xyxy.device)
        return cls_all.sigmoid(), xyxy * scale

    edges = torch.tensor(hist_edges(), dtype=torch.float64, device="cuda")
    nbins = len(edges) + 1

    def hist(values: Any) -> Any:
        v = values.double().flatten()
        idx = torch.where(v == 0, 0, torch.bucketize(v, edges, right=True) + 1)
        idx = idx.clamp(max=nbins)
        return torch.bincount(idx, minlength=nbins + 1)

    def new_acc() -> dict[str, Any]:
        z = lambda: torch.zeros((), dtype=torch.float64, device="cuda")  # noqa: E731
        return {
            "score_max": z(),
            "box_max": z(),
            "score_hist": torch.zeros(nbins + 1, dtype=torch.int64, device="cuda"),
            "box_hist": torch.zeros(nbins + 1, dtype=torch.int64, device="cuda"),
            "masked_anchors": torch.zeros((), dtype=torch.int64, device="cuda"),
            "crossings": torch.zeros((), dtype=torch.int64, device="cuda"),
            "nonfinite": torch.zeros((), dtype=torch.int64, device="cuda"),
            "frames": 0,
        }

    per_seq: dict[str, dict[str, dict[str, Any]]] = {}
    rerun_failures: list[dict[str, Any]] = []
    input_mutations: list[dict[str, Any]] = []
    v2_checked = 0
    l_e_bitwise_frames = 0
    rgb = ImageReadMode.RGB
    with torch.inference_mode():
        for seq in sequences:
            accs = {f"{a},{b}": new_acc() for a, b in L1_PAIRS}
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
                # One shared set of features for C, E and L (§4), and a
                # snapshot taken before C to prove no head writes into it.
                feats = [p.clone() for p in backbone.infer(frame_640)]
                snapshot = [p.clone() for p in feats]
                outs = {k: fn(feats) for k, fn in runners.items()}
                if i < V2_FRAMES:
                    v2_checked += 1
                    for k, fn in runners.items():
                        again = fn(feats)
                        if not all(torch.equal(a, b) for a, b in zip(outs[k], again)):
                            rerun_failures.append(
                                {"sequence": seq, "frame_index": i, "head": k}
                            )
                if not all(torch.equal(a, b) for a, b in zip(feats, snapshot)):
                    input_mutations.append({"sequence": seq, "frame_index": i})
                if all(torch.equal(a, b) for a, b in zip(outs["L"], outs["E"])):
                    l_e_bitwise_frames += 1
                sx, sy = w_orig / IMG_SIZE, h_orig / IMG_SIZE
                dec = {k: decode(v, sx, sy) for k, v in outs.items()}
                for a, b in L1_PAIRS:
                    acc = accs[f"{a},{b}"]
                    (sa, ba), (sb, bb) = dec[a], dec[b]
                    ds = (sa - sb).abs()
                    ma, mb = (
                        sa.max(0).values >= SCORE_FLOOR,
                        sb.max(0).values >= SCORE_FLOOR,
                    )
                    mask = ma | mb
                    db = (ba - bb).abs()[mask]
                    acc["nonfinite"] += (~torch.isfinite(ds)).sum() + (
                        ~torch.isfinite(db)
                    ).sum()
                    acc["score_max"] = torch.maximum(
                        acc["score_max"], ds.max().double()
                    )
                    if db.numel():
                        acc["box_max"] = torch.maximum(
                            acc["box_max"], db.max().double()
                        )
                    acc["score_hist"] += hist(ds)
                    acc["box_hist"] += hist(db)
                    acc["masked_anchors"] += mask.sum()
                    acc["crossings"] += (ma ^ mb).sum()
                    acc["frames"] += 1
            per_seq[seq] = {
                pair: {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in acc.items()
                }
                for pair, acc in accs.items()
            }
            print(f"[L1] {seq}: {len(frames)} frames", flush=True)

    rows = []
    summary: dict[str, dict[str, Any]] = {}
    for a, b in L1_PAIRS:
        pair = f"{a},{b}"
        tot_score_hist = [0] * (nbins + 1)
        tot_box_hist = [0] * (nbins + 1)
        agg = {
            "score_max": 0.0,
            "box_max": 0.0,
            "masked_anchors": 0,
            "crossings": 0,
            "nonfinite": 0,
            "frames": 0,
        }
        for seq in sequences:
            s = per_seq[seq][pair]
            rows.append(_l1_row(pair, seq, s))
            tot_score_hist = [x + y for x, y in zip(tot_score_hist, s["score_hist"])]
            tot_box_hist = [x + y for x, y in zip(tot_box_hist, s["box_hist"])]
            agg["score_max"] = max(agg["score_max"], s["score_max"])
            agg["box_max"] = max(agg["box_max"], s["box_max"])
            for k in ("masked_anchors", "crossings", "nonfinite", "frames"):
                agg[k] += s[k]
        all_row = _l1_row(
            pair, "ALL", {**agg, "score_hist": tot_score_hist, "box_hist": tot_box_hist}
        )
        rows.append(all_row)
        summary[pair] = all_row
    with (out_dir / "l1_pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lc = summary[",".join(L1_DECISION_PAIR)]
    score_max = math.nan if lc["nonfinite"] else lc["score_maxabs"]
    verdict = l1_verdict(score_max, lc["box_maxabs_px"])
    _write_json(
        out_dir / "l1.json",
        {
            "frames": lc["frames"],
            "decision_pair": ",".join(L1_DECISION_PAIR),
            "thresholds": {
                "score_maxabs": L1_SCORE_MAX,
                "box_maxabs_px": L1_BOX_MAX_PX,
                "score_floor": SCORE_FLOOR,
            },
            "verdict": verdict,
            "summary": summary,
            "v2": {
                "frames_checked": v2_checked,
                "rerun_failures": rerun_failures,
                "input_mutations": input_mutations,
                "jit_fallback_calls": dict(guard_counts),
            },
            "report_only": {
                "L_E_bitwise_equal_frames": l_e_bitwise_frames,
                "frames": lc["frames"],
            },
            "artifact": artifact,
            "driver_runtime": driver,
            "l_calls": adapter.calls,
            "l_call_problems": adapter.problems,
            "histogram": {
                "lo_exp": HIST_LO_EXP,
                "hi_exp": HIST_HI_EXP,
                "per_decade": HIST_PER_DECADE,
            },
            "per_sequence_histograms": {
                seq: {
                    pair: {"score_hist": v["score_hist"], "box_hist": v["box_hist"]}
                    for pair, v in d.items()
                }
                for seq, d in per_seq.items()
            },
            "environment": {**runtime_readback(torch), "torch": torch.__version__},
        },
    )
    print(f"[L1] verdict {verdict}", flush=True)
    return 0


def _l1_row(pair: str, seq: str, s: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": pair,
        "sequence": seq,
        "frames": s["frames"],
        "score_maxabs": s["score_max"],
        "score_p999_upper": hist_quantile(s["score_hist"], 0.999),
        "box_maxabs_px": s["box_max"],
        "box_p999_upper_px": hist_quantile(s["box_hist"], 0.999),
        "masked_anchors": s["masked_anchors"],
        "floor_crossings": s["crossings"],
        "nonfinite": s["nonfinite"],
    }


def _load_trt_export_tool() -> Any:
    import importlib.util

    tool = "scripts/model/export_headline_mamba_head.py"
    sys.path.insert(0, str(project_root / "scripts" / "model"))
    spec = importlib.util.spec_from_file_location(
        "export_headline_mamba_head", project_root / tool
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# L2 child entry (own process; declaration §5.1)
# --------------------------------------------------------------------------
def mot17_argv(
    arm: str, out_rel: str, sequences: tuple[str, ...], data_root: str | None
) -> list[str]:
    argv = [
        "scripts/eval/mot17.py",
        "--preset",
        PRESET_NAME,
        "--detector",
        "SDP",
        "--double-buffer",
        "--sequences",
        ",".join(sequences),
        "--output",
        out_rel,
        *ARMS[arm],
    ]
    if data_root is not None:  # structural checks only; never from the CLI
        argv += ["--data-root", data_root]
    return argv


def run_arm_child(
    arm: str,
    out_dir: Path,
    sidecar_path: Path,
    sequences: tuple[str, ...],
    data_root: str | None = None,
) -> int:
    """Run the unmodified mot17.py via runpy. A_C/A_N replace nothing; A_L
    injects L into the ``_trt_head`` slot (§5.1 steps 1–5). The sidecar is
    written by an ``atexit`` observer only."""
    # Same sys.path and first import as mot17.py's own header (the TRT
    # detector module must load before torchvision; libjpeg conflict).
    build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
    if build_path.exists():
        sys.path.insert(0, str(build_path))
    import saccade.perception.detector_trt  # noqa: F401

    import torch

    sidecar: dict[str, Any] = {
        "arm": arm,
        "installed": [],
        "runpy_completed": False,
    }
    state: dict[str, Any] = {}

    def write_sidecar() -> None:
        try:
            sidecar["op_library_mapped"] = op_library_mapped()
            if arm not in INJECTED_ARMS:
                # A_C/A_N: observed at exit, after the harness used CUDA.
                sidecar["driver_runtime"] = probe_driver_runtime()
            adapter = state.get("adapter")
            if adapter is not None:
                sidecar["adapter_calls"] = adapter.calls
                sidecar["adapter_problems"] = adapter.problems
                sidecar["adapter_input_shapes"] = sorted(
                    [list(map(list, s)) for s in adapter.shapes_seen]
                )
            if "guard_counts" in state:
                sidecar["guard_calls"] = dict(state["guard_counts"])
            sidecar["runtime_readback_at_exit"] = runtime_readback(torch)
            # §2 env hatch: the effective overrides of the process that ran
            # the harness, read after it finished (not the parent's).
            from saccade.perception.eval.assoc_basis import resolved_env_overrides

            sidecar["resolved_env_overrides"] = resolved_env_overrides()
        except Exception as exc:  # noqa: BLE001 — recorded, V5 fails closed
            sidecar["sidecar_error"] = repr(exc)
        _write_json(sidecar_path, sidecar)

    atexit.register(write_sidecar)

    from saccade.perception.temporal_yolo import mamba_gated_detector as mgd

    if arm in INJECTED_ARMS:
        guard_counts: dict[str, int] = {}
        state["guard_counts"] = guard_counts
        module, record = load_l(torch)  # steps 1–3
        sidecar["installed"] += [
            "op_library",
            "graph_executor_optimize=false",
            "torchscript_artifact",
        ]
        sidecar["op_library_sha256"] = record.pop("op_library_sha256")
        sidecar["artifact"] = record
        sidecar["driver_runtime"] = probe_driver_runtime()  # after the op library
        original_build = mgd.build_mamba_gated_detector
        sidecar["build_wrapper_calls"] = 0

        def build(*args: Any, **kwargs: Any) -> Any:  # step 4
            sidecar["build_wrapper_calls"] += 1
            if sidecar["build_wrapper_calls"] != 1:
                raise RuntimeError("build_mamba_gated_detector called more than once")
            det = original_build(*args, **kwargs)
            pre = {
                "trt_head_engine": kwargs.get("trt_head_engine"),
                "trt_head_is_none": det._trt_head is None,
                "use_whole_graph": det.use_whole_graph,
                "use_detail_fusion": det.use_detail_fusion,
            }
            sidecar["build_preconditions"] = pre
            if pre != BUILD_PRECONDITIONS:
                raise RuntimeError(f"injection preconditions failed: {pre}")
            adapter = LibTorchHeadAdapter(module, torch)
            state["adapter"] = adapter
            det._trt_head = adapter
            for name in HEAD_GUARDS:
                setattr(det.mamba_head, name, Guard(name, guard_counts))
            sidecar["slot_installed"] = det._trt_head is adapter
            return det

        mgd.build_mamba_gated_detector = build
        sidecar["installed"].append("mamba_gated_detector.build_mamba_gated_detector")
        install_jit_fallback_guards(guard_counts)
        sidecar["installed"] += [
            f"{MAMBA_HEAD_MODULE}.{g}" for g in JIT_FALLBACK_GUARDS
        ]

    eval_dir = str(project_root / "scripts" / "eval")
    sys.path.insert(0, eval_dir)
    out_arg = (
        str(out_dir.relative_to(project_root))
        if out_dir.is_relative_to(project_root)
        else str(out_dir)  # structural checks write outside the repository
    )
    argv = mot17_argv(arm, out_arg, sequences, data_root)
    sidecar["mot17_argv"] = argv
    sys.argv = list(argv)
    exit_code = 0
    if Path.cwd().resolve() != project_root:
        raise RuntimeError("the arm child must run with the repository as cwd")
    try:
        # runpy sets sys.argv[0] to this path; keep it the relative argv[0] so
        # the harness's run_manifest cmdline is exactly ``argv`` (cwd = root).
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as exc:
        exit_code = (
            exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        )
    sidecar["runpy_completed"] = exit_code == 0
    return exit_code


# --------------------------------------------------------------------------
# L2 orchestration (declaration §5)
# --------------------------------------------------------------------------
def run_arm(l2_dir: Path, arm: str, rep: int) -> dict[str, Any]:
    out = l2_dir / f"{arm}_{rep}"
    out.mkdir(parents=True, exist_ok=False)
    sidecar = l2_dir / f"{arm}_{rep}.sidecar.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(project_root)),
        "--_arm-child",
        arm,
        "--_out",
        str(out.relative_to(project_root)),
    ]
    log = l2_dir / f"{arm}_{rep}.stdout.log"
    started = _utc()
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT)
    txts = {s: out / f"{s}.txt" for s in SEQUENCES}
    return {
        "arm": arm,
        "rep": rep,
        "cmd": cmd,
        "mot17_argv": mot17_argv(
            arm, str(out.relative_to(project_root)), SEQUENCES, None
        ),
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": _utc(),
        "stdout": str(log.relative_to(project_root)),
        "output_dir": str(out.relative_to(project_root)),
        "sidecar": str(sidecar.relative_to(project_root)),
        "txt_sha256": {
            s: (_sha256(p) if p.exists() else None) for s, p in txts.items()
        },
    }


def run_problems(
    run: dict[str, Any], head: str, runs: list[dict[str, Any]]
) -> list[str]:
    """Everything that invalidates the study as soon as ``run`` finishes."""
    tag = f"{run['arm']}#{run['rep']}"
    if run["returncode"] != 0 or any(v is None for v in run["txt_sha256"].values()):
        return [f"{tag} failed (exit {run['returncode']}) or missing output"]
    problems = []
    manifest_path = project_root / run["output_dir"] / "run_manifest.json"
    if not manifest_path.exists():
        problems.append(f"{tag}: run_manifest.json missing")
    else:
        m = json.loads(manifest_path.read_text())
        problems += [
            f"{tag}: {p}" for p in argv_problems(m.get("cmdline"), run["mot17_argv"])
        ]
        if m.get("commit") != head:
            problems.append(f"{tag}: run_manifest commit {m.get('commit')} != {head}")
        if m.get("dirty") is not False:
            problems.append(f"{tag}: run_manifest reports a dirty tree")
    sidecar = _read_sidecar(run)
    problems += [f"V5 {tag}: {p}" for p in sidecar_problems(run["arm"], sidecar)]
    first = sidecar if run is runs[0] else _read_sidecar(runs[0])
    problems += [
        f"{tag}: {p}"
        for p in env_override_problems(
            (sidecar or {}).get("resolved_env_overrides"),
            (first or {}).get("resolved_env_overrides"),
        )
    ]
    if run["arm"] == "A_C" and run["rep"] == 1:
        problems += oracle_anchor_problems(run["txt_sha256"])
    if run["rep"] == 2:
        first = next(r for r in runs if r["arm"] == run["arm"] and r["rep"] == 1)
        if first["txt_sha256"] != run["txt_sha256"]:
            problems.append(f"V3: {run['arm']} runs #1 and #2 are not byte-identical")
    return problems


def _read_sidecar(run: dict[str, Any]) -> dict[str, Any] | None:
    path = project_root / run["sidecar"]
    return json.loads(path.read_text()) if path.exists() else None


def score_arm(out_dir: Path, sequences: tuple[str, ...]) -> dict[str, Any]:
    from saccade.perception.eval import metrics as M

    gt = {
        s: str(project_root / DATA_ROOT / SPLIT / s / "gt" / "gt.txt")
        for s in sequences
    }
    jobs = [(s, gt[s], str(out_dir / f"{s}.txt")) for s in sequences]
    per_counts = {s: M._evaluate_single_sequence(s, g, t) for s, g, t in jobs}
    totals = {
        k: sum(int(c[k]) for c in per_counts.values())
        for k in next(iter(per_counts.values()))
    }
    hota = M._calculate_hota(str(project_root / DATA_ROOT), SPLIT, str(out_dir), jobs)
    if hota is None:
        raise RuntimeError(f"TrackEval HOTA unavailable for {out_dir}")
    per_seq = {}
    for job in jobs:
        h = M._calculate_hota(str(project_root / DATA_ROOT), SPLIT, str(out_dir), [job])
        if h is None:
            raise RuntimeError(f"TrackEval HOTA unavailable for {job[0]}")
        per_seq[job[0]] = {
            "counts": per_counts[job[0]],
            "metrics": metrics_from_counts(per_counts[job[0]], h),
        }
    combined = metrics_from_counts(totals, hota)
    display = M._format_overall_metrics_from_counts(totals)
    display_ok = (
        display["IDF1"] == f"{combined['IDF1']:.1f}%"
        and display["MOTA"] == f"{combined['MOTA']:.1f}%"
        and display["IDs"] == int(combined["IDs"])
    )
    if not display_ok:
        raise RuntimeError(
            f"unrounded metrics disagree with metrics.py display: {display}"
        )
    return {
        "counts": totals,
        "hota_raw": hota,
        "combined": combined,
        "display": {**display, "HOTA": f"{combined['HOTA']:.1f}%"},
        "per_sequence": per_seq,
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--_l1-worker", dest="l1_worker", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_arm-child", dest="arm_child", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--_out", dest="out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.l1_worker:
        return run_l1_worker(Path(args.l1_worker), SEQUENCES)
    if args.arm_child:
        if args.arm_child not in ARMS or not args.out:
            parser.error("internal child flags are malformed")
        out = project_root / args.out
        return run_arm_child(
            args.arm_child, out, out.with_name(out.name + ".sidecar.json"), SEQUENCES
        )

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    packet = project_root / PACKET_ROOT / stamp
    # ADR 021 AP-2: claim the packet directory before the first result byte.
    # Each arm's mot17.py run claims its own sub-directory (no parent-claim
    # env is passed down), so its run_manifest.json records its own argv.
    from scripts.provenance.run_manifest import open_run

    open_run(
        packet,
        produced_by="diagnostic",
        preset=PRESET_NAME,
        detector="SDP",
        dataset=f"{DATA_ROOT} {SPLIT}",
    )
    record: dict[str, Any] = {
        "schema": "saccade.head_parity_packet/libtorch-v1",
        "issue": "#465 Phase B PR-2L (U1 redesign: LibTorch TorchScript)",
        "declaration": DECLARATION,
        "frozen_declaration_blob": FROZEN_DECLARATION_BLOB,
        "freeze_tag": FREEZE_TAG,
        "evidence": True,
        "tested_head": "L (the PR-2 decision functions' 'T' slot holds L)",
        "sequences": list(SEQUENCES),
        "run_order": [f"{a}#{r}" for a, r in RUN_ORDER],
        "started_utc": _utc(),
        "caller_saccade_env": _saccade_env(),
    }
    print(f"packet: {packet.relative_to(project_root)}", flush=True)

    terminal = "UNRESOLVED"
    reasons: list[str] = []
    l1 = l2 = None
    try:
        v1_ok, v1 = check_v1()
        record["v1"] = v1
        _write_json(packet / "v1.json", v1)
        if not v1_ok:
            reasons.append(
                "V1: " + "; ".join(c["item"] for c in v1["checks"] if not c["ok"])
            )
        else:
            l1, l2, m_reasons = _measure(packet, record)
            reasons += m_reasons
            end_lease = held_lease()
            record["lease_at_end"] = end_lease
            if end_lease != v1["lease"]:
                reasons.append("lease changed during the measurement")
            terminal = decide_terminal(not reasons, l1, l2)
    except Exception as exc:  # noqa: BLE001 — execution-invalid ⇒ UNRESOLVED
        reasons.append(f"runner error: {exc!r}")
    if reasons:
        terminal = "UNRESOLVED"

    record.update(
        {
            "finished_utc": _utc(),
            "l1_verdict": l1,
            "l2_verdict": l2,
            "unresolved_reasons": reasons,
            "terminal": terminal,
        }
    )
    _write_json(packet / "packet.json", record)
    _write_manifest(packet)
    print(f"terminal: {terminal}", flush=True)
    for r in reasons:
        print(f"  reason: {r}", flush=True)
    return 0 if terminal != "UNRESOLVED" else 2


def _measure(
    packet: Path, record: dict[str, Any]
) -> tuple[str | None, str | None, list[str]]:
    # L1 first; an invalid L1 ends the study before any MOT run (fail-fast).
    l1_dir = packet / "l1"
    l1_dir.mkdir()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_l1-worker", str(l1_dir)]
    with (l1_dir / "stdout.log").open("w") as f:
        proc = subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not (l1_dir / "l1.json").exists():
        return None, None, [f"L1 worker failed (exit {proc.returncode})"]
    l1 = json.loads((l1_dir / "l1.json").read_text())
    record["l1"] = {
        k: l1[k]
        for k in (
            "frames",
            "verdict",
            "summary",
            "v2",
            "report_only",
            "artifact",
            "driver_runtime",
            "l_calls",
            "environment",
        )
    }
    problems = l1_problems(l1, TOTAL_FRAMES)
    if problems:
        record["aborted_after"] = "L1"
        return l1["verdict"], None, problems
    l1_verdict_value = l1["verdict"]

    # L2: each run is validated before the next one launches; the first
    # failure ends the study with the completed runs kept in the packet.
    l2_dir = packet / "l2"
    l2_dir.mkdir()
    runs: list[dict[str, Any]] = []
    record["l2_runs"] = runs
    head = record["v1"]["git"]["head"]
    for arm, rep in RUN_ORDER:
        print(f"[L2] {arm}#{rep}", flush=True)
        run = run_arm(l2_dir, arm, rep)
        runs.append(run)
        problems = run_problems(run, head, runs)
        if problems:
            record["aborted_after"] = f"{arm}#{rep}"
            return l1_verdict_value, None, problems
    by = {(r["arm"], r["rep"]): r for r in runs}

    scored = {
        arm: score_arm(project_root / by[(arm, 1)]["output_dir"], SEQUENCES)
        for arm in ARMS
    }
    _write_json(packet / "l2_metrics.json", scored)
    exact = by[("A_L", 1)]["txt_sha256"] == by[("A_C", 1)]["txt_sha256"]
    verdict, detail = l2_verdict(
        scored["A_C"]["combined"],
        scored["A_L"]["combined"],
        scored["A_N"]["combined"],
        exact,
    )
    divergence = {}
    for other in ("A_L", "A_N"):
        divergence[other] = {
            s: first_divergent_frame(
                (project_root / by[("A_C", 1)]["output_dir"] / f"{s}.txt").read_bytes(),
                (project_root / by[(other, 1)]["output_dir"] / f"{s}.txt").read_bytes(),
            )
            for s in SEQUENCES
        }
    record["l2"] = {
        "verdict": verdict,
        "exact_A_L_vs_A_C": exact,
        "A_N_identical_to_A_C": by[("A_N", 1)]["txt_sha256"]
        == by[("A_C", 1)]["txt_sha256"],
        "decision": detail,
        "decision_slot_mapping": {"T": "A_L", "C": "A_C", "N": "A_N"},
        "first_divergent_frame_vs_A_C": divergence,
        "report_only": {
            arm: {k: scored[arm]["combined"][k] for k in ("DetA", "AssA", "FP", "FN")}
            for arm in ARMS
        },
        "prior_byte_identity": _prior_identity(by),
    }
    return l1_verdict_value, verdict, []


def _prior_identity(by: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """§5.4 report-only: per-sequence byte identity against earlier packets."""
    out: dict[str, Any] = {}
    for key, (arm, rel) in PRIOR_REFERENCE_TXT.items():
        prior = project_root / rel
        mine = project_root / by[(arm, 1)]["output_dir"]
        out[key] = {
            s: (
                (prior / f"{s}.txt").read_bytes() == (mine / f"{s}.txt").read_bytes()
                if (prior / f"{s}.txt").exists()
                else None
            )
            for s in SEQUENCES
        }
    return out


def _write_manifest(packet: Path) -> None:
    files = sorted(
        p for p in packet.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
    )
    _write_json(
        packet / "MANIFEST.json",
        {
            "generated_utc": _utc(),
            "files": [
                {
                    "path": str(p.relative_to(packet)),
                    "sha256": _sha256(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            ],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
