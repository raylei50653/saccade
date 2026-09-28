#!/usr/bin/env python3
"""Export the headline Mamba head as a LibTorch TorchScript artifact with a lineage manifest.

Issue #465 Phase B PR-1L (U1 redesign, LibTorch form; shared_boundary B1/B5
artifact side). The r2 failure-localization study ended
``EAGER_NUMERICS_WITHIN`` (docs/reference/
native_runtime_head_failure_localization_r2_result.md), so the next candidate
head form is LibTorch. Developer tooling only: it produces the artifact and
records where it came from; it does not change the eval harness, the preset or
any weights, and it measures no parity (that needs its own declaration).

Form: ``torch.jit.trace`` of the **same** head the oracle runs (built through
``export_headline_mamba_head.build_head`` -> ``build_mamba_gated_detector``
with the preset's TRT backbone engine and ``use_whole_graph=True``):
``MambaDetectionHead._forward_eager`` at T=1, ``return_embeddings=False``,
p3/p4/p5 -> cls_p3..p5, reg_p3..p5, FP32, batch 1. S2 is not exported (same
scope as PR-1).

The selective scan: during the trace the head's Python custom op
(``saccade::selective_scan_fwd``) is swapped for the C++ operator
``saccade_native::selective_scan_fwd`` from ``build/libsaccade_scan_torchop.so``
(``src/tracking/mamba_scan_torchop.cpp``), which calls the same launcher with
the same arguments. The artifact therefore needs no Python at run time; the
op library links libtorch and cudart only. The trace fails closed if the graph
contains a Python op, the Python scan op, or no native scan call.

Runtime requirements recorded in the manifest (a consumer must apply them):
the TorchScript graph executor's optimisation off (so the profiling executor
does not fuse, and the kernels are the eager aten kernels), cuDNN benchmark
off, cuDNN TF32 allowed, matmul TF32 off -- the torch defaults the oracle
harness runs with.

Structural check (``--check``, and at export): the saved artifact, loaded in
this process under those requirements, is run on synthetic random features
(no MOT17 frame) and compared bit for bit with the eager head through the
Python scan op. It says the artifact computes the eager head's function; it
is not parity evidence.

Outputs (under gitignored ``models/yolo/`` by default):

* ``<stem>.pt`` -- the TorchScript artifact
* ``<stem>.lineage.json`` -- ``saccade.head_artifact_lineage_torchscript/v1``

Usage:
    .venv/bin/python tools/resctl.py run gpu0 -- \\
        .venv/bin/python scripts/model/export_headline_mamba_head_torchscript.py
    .venv/bin/python tools/resctl.py run gpu0 -- \\
        .venv/bin/python scripts/model/export_headline_mamba_head_torchscript.py --check
"""
# status: diagnostic

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_headline_mamba_head as trt_export  # noqa: E402  (sets sys.path, RTLD_GLOBAL)

project_root = trt_export.project_root

SCHEMA = "saccade.head_artifact_lineage_torchscript/v1"
DEFAULT_STEM = "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript"
OP_LIBRARY = "build/libsaccade_scan_torchop.so"
NATIVE_OP = "saccade_native::selective_scan_fwd"
PYTHON_OP = "saccade::selective_scan_fwd"
OUTPUT_NAMES = trt_export.OUTPUT_NAMES
IMG_SIZE = trt_export.IMG_SIZE
STRIDES = (8, 16, 32)
# The oracle harness sets none of these; they are the torch defaults, which
# are also LibTorch's C++ defaults. A consumer must hold them.
RUNTIME_REQUIREMENTS = {
    "graph_executor_optimize": False,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": True,
    "matmul_allow_tf32": False,
}
FORBIDDEN_LINKS = ("libpython", "libtorch_python")
STRUCTURAL_SEEDS = (0, 1, 2)
WRAPPER_MODULE = "saccade_headline_head"
# Archive entries that change on every save without changing what runs.
CONTENT_EXCLUDED = (".data/serialization_id",)
# TracerWarnings allowed during the trace: (file, stripped source line). Each
# is a Python branch on a **shape** in _selective_scan_cuda, fixed by the
# weights and the artifact's static input shape (batch 1, 640 input), never by
# data: the state-count check on A.shape[-1], the shared/per-channel choice
# from A's shape vs the channel count, and the rank-1 C broadcast from the
# x_proj output width. The artifact is therefore static-shape only.
_MAMBA_HEAD = "src/saccade/perception/temporal_yolo/mamba_head.py"
ALLOWED_TRACER_WARNING_SITES = {
    (_MAMBA_HEAD, "if n <= 0 or n > 32 or (n & (n - 1)) != 0:"),
    (_MAMBA_HEAD, "a_per_channel = 1 if (A.dim() == 2 and A.shape[0] == D_dim) else 0"),
    (_MAMBA_HEAD, "if C.shape[-1] < N:"),
}


def load_op_library() -> dict[str, Any]:
    import torch

    path = project_root / OP_LIBRARY
    if not path.exists():
        raise SystemExit(
            f"{OP_LIBRARY} missing; build it: cmake --build build --target "
            "saccade_scan_torchop"
        )
    torch.ops.load_library(str(path))
    return {**trt_export._file_record(path), "needed": needed_libraries(path)}


def needed_libraries(path: Path) -> list[str]:
    """``DT_NEEDED`` of the op library; fail closed on a Python dependency."""
    import subprocess

    out = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
    ).stdout
    needed = [
        line.split("[", 1)[1].rstrip("]").strip()
        for line in out.splitlines()
        if "(NEEDED)" in line
    ]
    bad = [n for n in needed if n.startswith(FORBIDDEN_LINKS)]
    if bad:
        raise SystemExit(f"{OP_LIBRARY} links Python: {bad}")
    return needed


@contextlib.contextmanager
def native_scan() -> Iterator[None]:
    """Route the head's scan through the C++ operator for the trace."""
    import torch

    from saccade.perception.temporal_yolo import mamba_head as mh

    original = mh._saccade_selective_scan_op
    mh._saccade_selective_scan_op = torch.ops.saccade_native.selective_scan_fwd
    try:
        yield
    finally:
        mh._saccade_selective_scan_op = original


@contextlib.contextmanager
def runtime_requirements() -> Iterator[None]:
    """Hold the manifest's runtime requirements; restore on exit."""
    import torch

    saved = (
        torch._C._get_graph_executor_optimize(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch._C._set_graph_executor_optimize(
        RUNTIME_REQUIREMENTS["graph_executor_optimize"]
    )
    torch.backends.cudnn.benchmark = RUNTIME_REQUIREMENTS["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = RUNTIME_REQUIREMENTS["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = RUNTIME_REQUIREMENTS["matmul_allow_tf32"]
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(saved[0])
        torch.backends.cudnn.benchmark = saved[1]
        torch.backends.cudnn.allow_tf32 = saved[2]
        torch.backends.cuda.matmul.allow_tf32 = saved[3]


def input_shapes(in_channels: list[int]) -> list[tuple[int, int, int, int]]:
    return [(1, c, IMG_SIZE // s, IMG_SIZE // s) for c, s in zip(in_channels, STRIDES)]


def make_wrapper(head: Any) -> Any:
    import torch
    from torch import nn

    class TorchScriptMambaHead(nn.Module):
        """p3/p4/p5 -> (cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5)."""

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.head = inner

        def forward(
            self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            cls, reg = self.head._forward_eager([p3, p4, p5], return_embeddings=False)
            return (cls[0], cls[1], cls[2], reg[0], reg[1], reg[2])

    # The TorchScript qualified name follows __module__; pin it so the archive
    # does not depend on whether this file runs as __main__ or is imported.
    TorchScriptMambaHead.__module__ = WRAPPER_MODULE
    return TorchScriptMambaHead(head).eval()


def tracer_warning_problems(
    records: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every TracerWarning must come from an allowlisted source line: a check
    on a **shape** (fixed by the weights), never on data. Anything else could
    bake a data-dependent branch into the trace, so it fails closed."""
    import linecache

    import torch

    seen, problems = [], []
    for r in records:
        if not issubclass(r.category, torch.jit.TracerWarning):
            continue
        line = linecache.getline(r.filename, r.lineno).strip()
        site = {
            "file": trt_export._rel(Path(r.filename)),
            "line": r.lineno,
            "source": line,
        }
        if site not in seen:
            seen.append(site)
        if (site["file"], line) not in ALLOWED_TRACER_WARNING_SITES:
            problems.append(f"unexpected TracerWarning at {site}")
    return seen, problems


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


def trace_head(head: Any, in_channels: list[int]) -> tuple[Any, dict[str, Any]]:
    import torch

    wrapper = make_wrapper(head)
    dummies = tuple(torch.zeros(s, device="cuda") for s in input_shapes(in_channels))
    with native_scan(), torch.no_grad(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        traced = torch.jit.trace(wrapper, dummies, check_trace=False)
    graph_text = str(traced.inlined_graph)
    problems = graph_problems(graph_text)
    sites, warn_problems = tracer_warning_problems(list(w))
    if problems or warn_problems:
        raise SystemExit(
            f"traced graph is not a Python-free, input-independent artifact: "
            f"{problems + warn_problems}"
        )
    return traced, {
        "native_scan_calls": graph_text.count(NATIVE_OP),
        "tracer_warning_sites": sites,
    }


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


def save(traced: Any, out: Path) -> dict[str, Any]:
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out))
    return {**trt_export._file_record(out), "content_sha256": content_sha256(out)}


def structural_check(
    artifact: Path, head: Any, in_channels: list[int]
) -> dict[str, Any]:
    """Load the saved artifact under the runtime requirements and compare it,
    bit for bit, with the eager head (Python scan op) on synthetic features."""
    import torch

    results = []
    with runtime_requirements(), torch.no_grad():
        module = torch.jit.load(str(artifact), map_location="cuda").eval()
        for seed in STRUCTURAL_SEEDS:
            g = torch.Generator(device="cuda").manual_seed(seed)
            feats = [
                torch.randn(s, device="cuda", generator=g)
                for s in input_shapes(in_channels)
            ]
            got = module(*feats)
            cls, reg = head._forward_eager(list(feats), return_embeddings=False)
            want = (*cls, *reg)
            torch.cuda.synchronize()
            per_output = {
                name: bool(
                    a.shape == b.shape
                    and torch.equal(
                        a.contiguous().view(torch.int32),
                        b.contiguous().view(torch.int32),
                    )
                )
                for name, a, b in zip(OUTPUT_NAMES, got, want)
            }
            max_abs = max(float((a - b).abs().max()) for a, b in zip(got, want))
            results.append(
                {"seed": seed, "bitwise_equal": per_output, "max_abs_diff": max_abs}
            )
    ok = all(all(r["bitwise_equal"].values()) for r in results)
    return {
        "input": "synthetic torch.randn features (no MOT17 frame), seeds "
        + ",".join(map(str, STRUCTURAL_SEEDS)),
        "reference": "MambaDetectionHead._forward_eager, Python scan op, same process",
        "runtime_requirements_held": RUNTIME_REQUIREMENTS,
        "bitwise_equal_all": ok,
        "per_seed": results,
        "reading": "the artifact computes the eager head's function on these "
        "inputs; this is not parity evidence",
    }


def environment() -> dict[str, Any]:
    import platform

    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "sm": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "host": platform.node(),
    }


def run_export(args: argparse.Namespace) -> int:
    stem = project_root / args.stem
    pt_path = stem.with_suffix(".pt")
    lineage_path = stem.with_suffix(".lineage.json")
    for p in (pt_path, lineage_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(
                f"{trt_export._rel(p)} exists; pass --overwrite to replace"
            )

    op_rec = load_op_library()
    inputs = trt_export.resolve_inputs(args.yolo_weights, args.teacher_ckpt)
    head, described = trt_export.build_head(inputs)
    in_channels = described["head_load"]["in_channels"]
    traced, trace_rec = trace_head(head, in_channels)
    pt_rec = save(traced, pt_path)
    check = structural_check(pt_path, head, in_channels)
    record = {
        "schema": SCHEMA,
        "issue": "#465 Phase B PR-1L (U1 redesign: LibTorch TorchScript)",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "path": trt_export._rel(Path(__file__)),
            "git_commit": trt_export._git("rev-parse", "HEAD"),
            "git_dirty": bool(trt_export._git("status", "--porcelain")),
        },
        "preset": inputs["preset"],
        "inventory": inputs["inventory"],
        **described,
        "artifact_scope": {
            "included": (
                "MambaDetectionHead._forward_eager, single frame (T=1), "
                "return_embeddings=False: p3/p4/p5 -> cls_p3..p5, reg_p3..p5"
            ),
            "excluded": (
                "S2 whole-detect tail (640 stretch resize, anchor decode, "
                "sigmoid/class max, top-k, coordinate scaling) -> U3b; "
                "conf_thr/max_det stay in the B2 resolved config"
            ),
        },
        "torchscript": {
            **pt_rec,
            "method": "torch.jit.trace (check_trace=False), no freeze, no optimize",
            "inputs": {
                n: list(s)
                for n, s in zip(["p3", "p4", "p5"], input_shapes(in_channels))
            },
            "outputs": OUTPUT_NAMES,
            "dtype": "float32",
            "batch": "static 1",
            **trace_rec,
        },
        "op_library": {**op_rec, "op": NATIVE_OP},
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "structural_check": check,
        "companions": {
            "backbone_engine": {
                "path": trt_export._rel(inputs["backbone"]),
                "sha256": inputs["backbone_sha256"],
                "reading": (
                    "recorded for B5 hash verification only; its provenance is the "
                    "inventory's unresolved engine link, not attested here"
                ),
            }
        },
        "environment": environment(),
    }
    lineage_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"lineage     : {trt_export._rel(lineage_path)}")
    print(f"torchscript : {pt_rec['sha256']} (content {pt_rec['content_sha256']})")
    print(f"structural  : bitwise_equal_all={check['bitwise_equal_all']}")
    return 0 if check["bitwise_equal_all"] else 1


def run_check(args: argparse.Namespace) -> int:
    """Re-trace from the same inputs, compare to the manifest, re-run the
    structural check on the on-disk artifact."""
    stem = project_root / args.stem
    lineage_path = stem.with_suffix(".lineage.json")
    record = json.loads(lineage_path.read_text())
    if record.get("schema") != SCHEMA:
        raise SystemExit(f"{lineage_path}: unknown schema {record.get('schema')!r}")
    failures = []
    op_rec = load_op_library()
    if op_rec["sha256"] != record["op_library"]["sha256"]:
        failures.append("op library sha256 differs from manifest")
    inputs = trt_export.resolve_inputs(args.yolo_weights, args.teacher_ckpt)
    if inputs["ckpt_sha256"] != record["source"]["mamba_ckpt"]["sha256"]:
        failures.append("mamba_ckpt sha256 differs from manifest")
    head, described = trt_export.build_head(inputs)
    if described["head_load"] != record["head_load"]:
        failures.append("head load description differs from manifest")
    in_channels = described["head_load"]["in_channels"]
    traced, _ = trace_head(head, in_channels)
    pt_path = project_root / record["torchscript"]["path"]
    with tempfile.TemporaryDirectory() as tmp:
        fresh = save(traced, Path(tmp) / pt_path.name)
    # File bytes differ on every save (serialization_id, debug paths); the
    # portable identity is the content hash.
    if fresh["content_sha256"] != record["torchscript"]["content_sha256"]:
        failures.append(
            f"re-traced artifact content_sha256 {fresh['content_sha256']} != "
            f"manifest {record['torchscript']['content_sha256']}"
        )
    if (
        not pt_path.exists()
        or trt_export._sha256(pt_path) != record["torchscript"]["sha256"]
    ):
        failures.append(f"{record['torchscript']['path']} missing or altered")
    else:
        check = structural_check(pt_path, head, in_channels)
        if not check["bitwise_equal_all"]:
            failures.append(
                "structural check: artifact != eager head on synthetic input"
            )
    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print(
            f"OK: re-traced artifact content identical ({fresh['content_sha256']}); "
            "on-disk file unaltered; structural check bit-exact"
        )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--yolo-weights", default=trt_export.DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--teacher-ckpt", default=trt_export.DEFAULT_TEACHER_CKPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-trace and verify the manifest and the structural check",
    )
    args = parser.parse_args()
    return run_check(args) if args.check else run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
