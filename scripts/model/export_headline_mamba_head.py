#!/usr/bin/env python3
"""Export the headline Mamba head as an ONNX + TensorRT artifact with a lineage manifest.

Issue #465 Phase B PR-1 (U1a; shared_boundary B1/B5 artifact side, see
docs/reference/native_runtime_shipping_boundary.md §5–§6). Developer tooling
only: it produces the head artifact a Python-free runtime would load, and
records where it came from. It does not change the eval harness, the preset or
any weights, and it measures no parity (that is PR-2).

The head is built through the **same** constructor the oracle uses
(``build_mamba_gated_detector`` with the preset's TRT backbone engine and
``use_whole_graph=True``), so the exported module is the one ``_whole_graph_fn``
runs: single frame (temporal blocks bypassed, effective T=1),
``return_embeddings=False``, outputs ``cls_p3..p5`` / ``reg_p3..p5``.

Artifact scope: head only. S2 (640 stretch resize, anchor decode, sigmoid/class
max, top-k, coordinate scaling) is **not** exported; it stays with U3b as native
kernels, so ``conf_thr``/``max_det`` keep a single source in the resolved
config (B2) instead of being frozen into an engine.

Fail-closed inputs: the checkpoint must be the one the preset names and its
sha256 must equal the ``s.t3t1_phase_b`` node of the committed lineage
inventory (``report_data/training_lineage_inventory.json``).

Outputs (all under gitignored ``models/yolo/`` by default):

* ``<stem>.onnx``   — portable artifact (``saccade::SelectiveScan`` custom op)
* ``<stem>.engine`` — this machine's build (SM + TensorRT version bound)
* ``<stem>.lineage.json`` — ``saccade.head_artifact_lineage/v1`` manifest

Usage:
    .venv/bin/python tools/resctl.py run gpu0 -- \\
        .venv/bin/python scripts/model/export_headline_mamba_head.py
    # re-export into a temp dir and verify the recorded ONNX sha256
    .venv/bin/python scripts/model/export_headline_mamba_head.py --check
"""
# status: diagnostic

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

SCHEMA = "saccade.head_artifact_lineage/v1"
PRESET = "configs/presets/mamba_whole_graph.yaml"
INVENTORY = "report_data/training_lineage_inventory.json"
INVENTORY_CKPT_NODE = "s.t3t1_phase_b"
INVENTORY_BACKBONE_NODE = "s.backbone_engine"
# mot17.py argparse defaults (scripts/eval/config/core.py); the preset leaves
# both unset. Neither feeds head weights: yolo26s.pt is only hashed by the
# (here inactive) lineage gate, the teacher only supplies GatedDetConfig.
DEFAULT_YOLO_WEIGHTS = "models/yolo/yolo26s.pt"
DEFAULT_TEACHER_CKPT = "runs/gated_det_v1/best.ckpt"
DEFAULT_STEM = "models/yolo/mamba_head_s_v14replica_t3_t1_fp32"
PLUGIN = "build/libsaccade_scan_plugin.so"
IMG_SIZE = 640
OUTPUT_NAMES = ["cls_p3", "cls_p4", "cls_p5", "reg_p3", "reg_p4", "reg_p5"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path.resolve())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return repr(v)


def resolve_inputs(yolo_weights: str, teacher_ckpt: str) -> dict[str, Any]:
    """Read the preset and the lineage inventory; fail closed on any mismatch."""
    import yaml

    preset_path = project_root / PRESET
    preset = yaml.safe_load(preset_path.read_text())
    ckpt = project_root / preset["mamba_ckpt"]
    backbone = project_root / preset["fpn_backbone_engine"]
    if not preset.get("use_whole_graph"):
        raise SystemExit(f"{PRESET}: use_whole_graph is not true")

    inventory = json.loads((project_root / INVENTORY).read_text())
    node = inventory["nodes"][INVENTORY_CKPT_NODE]
    if node["path"] != preset["mamba_ckpt"]:
        raise SystemExit(
            f"inventory {INVENTORY_CKPT_NODE} path {node['path']!r} != preset "
            f"mamba_ckpt {preset['mamba_ckpt']!r}"
        )
    ckpt_sha = _sha256(ckpt)
    if ckpt_sha != node["sha256"]:
        raise SystemExit(
            f"{_rel(ckpt)} sha256 {ckpt_sha} != inventory {INVENTORY_CKPT_NODE} "
            f"{node['sha256']}"
        )
    bb_node = inventory["nodes"][INVENTORY_BACKBONE_NODE]
    bb_sha = _sha256(backbone)
    return {
        "preset": {
            "path": PRESET,
            "sha256": _sha256(preset_path),
            "mamba_ckpt": preset["mamba_ckpt"],
            "fpn_backbone_engine": preset["fpn_backbone_engine"],
        },
        "ckpt": ckpt,
        "ckpt_sha256": ckpt_sha,
        "backbone": backbone,
        "yolo_weights": project_root / yolo_weights,
        "teacher_ckpt": project_root / teacher_ckpt,
        "inventory": {
            "path": INVENTORY,
            "sha256": _sha256(project_root / INVENTORY),
            "captured": _jsonable(inventory.get("captured")),
            "ckpt_node": INVENTORY_CKPT_NODE,
            "ckpt_sha256_match": True,
            "backbone_node": INVENTORY_BACKBONE_NODE,
            "backbone_engine_sha256_match": bb_sha == bb_node["sha256"],
        },
        "backbone_sha256": bb_sha,
    }


def build_head(inputs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Construct the head exactly as the oracle does and describe how it loaded."""
    import saccade_tracking_ext  # noqa: F401  (before torchvision; see export_mamba_head.py)
    import torch

    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
    )

    detector = build_mamba_gated_detector(
        yolo_pt_path=str(inputs["yolo_weights"]),
        teacher_ckpt=str(inputs["teacher_ckpt"]),
        mamba_ckpt=str(inputs["ckpt"]),
        img_size=IMG_SIZE,
        device="cuda",
        conf_thr=0.001,
        trt_backbone_engine=str(inputs["backbone"]),
        use_whole_graph=True,
    )
    detector.eval()
    head = detector.mamba_head

    state = torch.load(inputs["ckpt"], map_location="cpu", weights_only=False)
    mamba_args = state["mamba_args"]
    sd = {k.replace("._orig_mod.", "."): v for k, v in state["student"].items()}
    own = head.state_dict()
    load = {
        "missing_keys": sorted(set(own) - set(sd)),
        "unexpected_keys": sorted(set(sd) - set(own)),
        # MambaDetectionHead.load_state_dict(strict=False) silently drops these.
        "shape_mismatch_dropped": sorted(
            k
            for k in set(own) & set(sd)
            if own[k].shape != sd[k].shape
            and not k.endswith("A_log")  # A_log is broadcast, not dropped
        ),
        "upsample_loaded": bool(getattr(head, "upsample_loaded", False)),
        "in_channels": list(detector.in_channels),
        "temporal_blocks": (
            "present; bypassed (whole-graph forward is single-frame, T=1)"
            if head.temporal_blocks is not None
            else "absent"
        ),
        "use_detail_fusion": bool(detector.use_detail_fusion),
    }
    if load["missing_keys"] or load["shape_mismatch_dropped"]:
        raise SystemExit(f"head did not load cleanly from the checkpoint: {load}")
    source = {
        "mamba_ckpt": {
            "path": _rel(inputs["ckpt"]),
            "sha256": inputs["ckpt_sha256"],
            "bytes": inputs["ckpt"].stat().st_size,
            "epoch": _jsonable(state.get("epoch")),
            "selection": _jsonable(state.get("selection")),
        },
        "mamba_args": _jsonable(mamba_args),
        "lineage_gate": {
            "base_yolo_sha256_recorded": "base_yolo_sha256" in mamba_args,
            "teacher_checkpoint_sha256_recorded": "teacher_checkpoint_sha256"
            in mamba_args,
            "reading": (
                "the oracle's SHA gate (mamba_gated_detector.py, MambaGatedDetector"
                ".__init__) only fires for recorded hashes; lineage is instead "
                "pinned here by the inventory ckpt sha256"
            ),
        },
        "builder_inputs": {
            "yolo_weights": _file_record(inputs["yolo_weights"]),
            "teacher_ckpt": {
                **_file_record(inputs["teacher_ckpt"]),
                "role": "GatedDetConfig only; not unpickled into the head, "
                "no effect on head weights",
            },
        },
    }
    return head, {"source": source, "head_load": load}


def export_onnx(head: Any, out: Path, in_channels: list[int]) -> dict[str, Any]:
    import torch

    from export_mamba_head_onnx import ONNXMambaHead, _patch_selective_scan

    _patch_selective_scan(head)
    wrapper = ONNXMambaHead(head).eval()
    shapes = [
        (1, c, IMG_SIZE // s, IMG_SIZE // s) for c, s in zip(in_channels, (8, 16, 32))
    ]
    dummies = tuple(torch.zeros(s, device="cuda") for s in shapes)
    with torch.no_grad():
        wrapper(*dummies)
    torch.cuda.synchronize()
    out.parent.mkdir(parents=True, exist_ok=True)
    opset = 17
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummies,
            str(out),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["p3", "p4", "p5"],
            output_names=OUTPUT_NAMES,
            dynamo=False,
            verbose=False,
        )
    return {
        **_file_record(out),
        "opset": opset,
        "inputs": {n: list(s) for n, s in zip(["p3", "p4", "p5"], shapes)},
        "outputs": OUTPUT_NAMES,
        "custom_ops": ["saccade::SelectiveScan (libsaccade_scan_plugin.so)"],
        "batch": "static 1",
    }


def build_engine(onnx: Path, engine: Path, in_channels: list[int]) -> dict[str, Any]:
    import tensorrt as trt
    import torch

    from build_mamba_head_trt import build

    build(
        str(onnx),
        str(engine),
        *in_channels,
        min_batch=1,
        opt_batch=1,
        max_batch=1,
        fp16=False,
    )
    major, minor = torch.cuda.get_device_capability()
    return {
        **_file_record(engine),
        "precision": "fp32",
        "builder_flags": "TensorRT defaults (TF32 allowed); FP16 off",
        "profile": "batch min=opt=max=1",
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{major}.{minor}",
        "plugin": _file_record(project_root / PLUGIN),
        "bytes_reproducible": False,
        "reading": (
            "engine bytes depend on tactic timing, SM and TensorRT version; the "
            "portable identity is the ONNX sha256, the engine sha256 identifies "
            "this machine's build only"
        ),
    }


def environment() -> dict[str, Any]:
    import onnx
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "onnx": onnx.__version__,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "host": platform.node(),
    }


def run_export(args: argparse.Namespace) -> int:
    stem = project_root / args.stem
    onnx_path = stem.with_suffix(".onnx")
    engine_path = stem.with_suffix(".engine")
    lineage_path = stem.with_suffix(".lineage.json")
    for p in (onnx_path, engine_path, lineage_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{_rel(p)} exists; pass --overwrite to replace")

    inputs = resolve_inputs(args.yolo_weights, args.teacher_ckpt)
    head, described = build_head(inputs)
    onnx_rec = export_onnx(head, onnx_path, described["head_load"]["in_channels"])
    engine_rec = (
        None
        if args.skip_engine
        else build_engine(onnx_path, engine_path, described["head_load"]["in_channels"])
    )
    record = {
        "schema": SCHEMA,
        "issue": "#465 Phase B PR-1 (U1a)",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "path": _rel(Path(__file__)),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
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
                "sigmoid/class max, top-k, coordinate scaling) -> U3b native "
                "kernels; conf_thr/max_det stay in the B2 resolved config"
            ),
        },
        "onnx": onnx_rec,
        "engine": engine_rec,
        "companions": {
            "backbone_engine": {
                "path": _rel(inputs["backbone"]),
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
    print(f"lineage: {_rel(lineage_path)}")
    print(f"onnx   : {onnx_rec['sha256']}")
    if engine_rec:
        print(f"engine : {engine_rec['sha256']}")
    return 0


def run_check(args: argparse.Namespace) -> int:
    """Re-export the ONNX from the same inputs and compare to the manifest."""
    lineage_path = (project_root / args.stem).with_suffix(".lineage.json")
    record = json.loads(lineage_path.read_text())
    if record.get("schema") != SCHEMA:
        raise SystemExit(
            f"{_rel(lineage_path)}: unknown schema {record.get('schema')!r}"
        )
    inputs = resolve_inputs(args.yolo_weights, args.teacher_ckpt)
    failures = []
    if inputs["ckpt_sha256"] != record["source"]["mamba_ckpt"]["sha256"]:
        failures.append("mamba_ckpt sha256 differs from manifest")
    head, described = build_head(inputs)
    if described["head_load"] != record["head_load"]:
        failures.append("head load description differs from manifest")
    with tempfile.TemporaryDirectory() as tmp:
        fresh = export_onnx(
            head, Path(tmp) / "head.onnx", described["head_load"]["in_channels"]
        )
    if fresh["sha256"] != record["onnx"]["sha256"]:
        failures.append(
            f"re-exported ONNX sha256 {fresh['sha256']} != manifest "
            f"{record['onnx']['sha256']}"
        )
    onnx_path = project_root / record["onnx"]["path"]
    if not onnx_path.exists() or _sha256(onnx_path) != record["onnx"]["sha256"]:
        failures.append(f"{record['onnx']['path']} missing or altered")
    if record.get("engine"):
        engine_path = project_root / record["engine"]["path"]
        if (
            not engine_path.exists()
            or _sha256(engine_path) != record["engine"]["sha256"]
        ):
            failures.append(f"{record['engine']['path']} missing or altered")
    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print(f"OK: ONNX rebuilt bit-identically ({fresh['sha256']})")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stem", default=DEFAULT_STEM, help="output path stem")
    parser.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--teacher-ckpt", default=DEFAULT_TEACHER_CKPT)
    parser.add_argument(
        "--skip-engine", action="store_true", help="ONNX + manifest only"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-export into a temp dir and verify the manifest instead of writing",
    )
    args = parser.parse_args()
    return run_check(args) if args.check else run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
