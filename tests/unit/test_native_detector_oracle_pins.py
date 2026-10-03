"""Oracle pins for the native detector (#465 Phase B PR-8, U3b-2).

The native detector (``shipping/src/detector_plan.cpp``, ``detector_host.cpp``,
``detector_s2.cu``) re-implements the oracle's whole-graph detector with the
PR-1L head (``A_L``) in the headline configuration. Its parity is measured on
real frames by ``scripts/eval/diagnostics/native_detector_parity.py`` and on
committed inputs by ``tests/native/fixtures/shipping_detector_s2.json``; these
checks pin, at source level, the oracle facts that are code literals rather
than config values, and the committed identity records:

* **whole-graph path** -- ``_whole_graph_fn``'s statements (resize, backbone,
  head slot, S2 call, coordinate scaling), the stride and anchor offset
  literals, ``_whole_graph_nms_pad`` (only ever 0), the scales
  ``set_whole_graph_img_dims`` stores, ``detect_single_patch_640``'s
  whole-graph branch returning the S2 rows unchanged;
* **S2** -- ``_postprocess_mamba_fixed_eager``'s source (hash) and its
  ``torch.compile(mode='default')`` default; the torch / triton versions the S2
  twin was read from (a new Inductor is a new lowering to re-read);
* **harness** -- ``oracle_s2`` / ``Oracle.resize`` run exactly that tail;
* **identity** -- the lineage fixture is the frozen PR-1L lineage byte for
  byte; the realization attestation is bound to it, its op-library sources are
  the HEAD blobs, and its A_L reproduction equals the PR-2L reference;
* **fixture** -- the S2 fixture is fresh (CUDA only).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MGD = REPO / "src/saccade/perception/temporal_yolo/mamba_gated_detector.py"
DETECTION = REPO / "src/saccade/perception/eval/detection.py"
HARNESS = REPO / "scripts/eval/diagnostics/native_detector_parity.py"
RENDER = REPO / "scripts/model/render_shipping_detector_s2_fixture.py"
S2_FIXTURE = REPO / "tests/native/fixtures/shipping_detector_s2.json"
LINEAGE_FIXTURE = REPO / "tests/native/fixtures/shipping_head_lineage.json"
LINEAGE = (
    REPO / "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json"
)
ATTESTATION = REPO / "configs/shipping/mamba_head_realization.attestation.json"

# ast.unparse of _postprocess_mamba_fixed_eager when the S2 twin was read
# (detector_s2.hpp). Changing the function is a new S2 to re-read and re-measure.
POSTPROCESS_FIXED_EAGER_SHA256 = (
    "a19fa0910fdae59b3add6bff2d6f1e3b396e4da4acbb1f090bbaa8a0f1c2de42"
)
TORCH_VERSION = "2.11.0+cu130"
TRITON_VERSION = "3.6.0"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _assignments(tree: ast.AST, target: str) -> list[str]:
    out = []
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else []
        if isinstance(n, ast.AnnAssign):
            targets = [n.target]
        out += [ast.unparse(n.value) for t in targets if ast.unparse(t) == target]  # type: ignore[union-attr]
    return out


def _module(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_whole_graph_fn_statements() -> None:
    body = [ast.unparse(s) for s in _function(_tree(MGD), "_whole_graph_fn").body]
    assert body[0] == (
        "frame_640 = F.interpolate(frame, size=(self.img_size, self.img_size), "
        "mode='bilinear', align_corners=False)"
    )
    assert body[1:3] == [
        "backbone = self._trt_backbone",
        "p3, p4, p5 = backbone.infer_graph(frame_640)",
    ]
    head = body[6]
    assert head.startswith(
        "if self._trt_head is not None and (not self.use_detail_fusion):\n"
        "    cls_preds, reg_preds = self._trt_head.infer_graph(p3, p4, p5)"
    )
    assert body[7:] == [
        "detections = _postprocess_mamba_fixed(cls_preds, reg_preds, self.stride, self.conf_thr, "
        "max(self.max_det, self._whole_graph_nms_pad), anchors=self._whole_graph_anchors, "
        "anchor_strides=self._whole_graph_anchor_strides, "
        "small_p3_max_threshold=self.small_p3_max_threshold, box_scale_x=self._whole_graph_sx, "
        "box_scale_y=self._whole_graph_sy)",
        "detections[:, :, self._whole_graph_x_idx] *= self._whole_graph_sx",
        "detections[:, :, self._whole_graph_y_idx] *= self._whole_graph_sy",
        "return detections",
    ]


def test_detector_literals() -> None:
    tree = _tree(MGD)
    assert _assignments(tree, "self.stride") == [
        "torch.tensor([8.0, 16.0, 32.0], device=device)"
    ]
    assert _assignments(tree, "self._whole_graph_x_idx") == [
        "torch.tensor([0, 2], device=device, dtype=torch.long)"
    ]
    assert _assignments(tree, "self._whole_graph_y_idx") == [
        "torch.tensor([1, 3], device=device, dtype=torch.long)"
    ]
    offsets = [
        ast.unparse(n)
        for n in ast.walk(_function(tree, "_precompute_anchor_grid"))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
    ]
    assert offsets == [
        "torch.arange(w, dtype=torch.float32, device=stride_tensor.device) + 0.5",
        "torch.arange(h, dtype=torch.float32, device=stride_tensor.device) + 0.5",
    ]
    dims = [ast.unparse(s) for s in _function(tree, "set_whole_graph_img_dims").body]
    assert "self._whole_graph_sx.fill_(w_orig / self.img_size)" in dims
    assert "self._whole_graph_sy.fill_(h_orig / self.img_size)" in dims


def test_nms_pad_is_only_ever_zero() -> None:
    # max(self.max_det, self._whole_graph_nms_pad) = max_det: no padding rows.
    found = []
    for path in [*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if "_whole_graph_nms_pad" not in text:
            continue
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if (
                        isinstance(t, ast.Attribute)
                        and t.attr == "_whole_graph_nms_pad"
                    ):
                        found.append(
                            (path.relative_to(REPO).as_posix(), ast.unparse(n))
                        )
    assert found == [
        (
            "src/saccade/perception/temporal_yolo/mamba_gated_detector.py",
            "self._whole_graph_nms_pad = 0",
        )
    ]


def test_postprocess_fixed_source_and_compile_default() -> None:
    tree = _tree(MGD)
    eager = ast.unparse(_function(tree, "_postprocess_mamba_fixed_eager"))
    assert hashlib.sha256(eager.encode()).hexdigest() == POSTPROCESS_FIXED_EAGER_SHA256
    # conf_thr is a parameter the fixed S2 never reads.
    names = {
        n.id
        for n in ast.walk(_function(tree, "_postprocess_mamba_fixed_eager"))
        if isinstance(n, ast.Name)
    }
    assert "conf_thr" not in names
    assert _assignments(tree, "_POSTPROCESS_COMPILE_ENABLED")[0] == "True"
    calls = [
        ast.unparse(n)
        for n in ast.walk(_function(tree, "_get_compiled_postprocess"))
        if isinstance(n, ast.Call)
    ]
    assert (
        "torch.compile(_postprocess_mamba_fixed_eager, mode='default', fullgraph=False)"
        in calls
    )


def test_detect_single_patch_640_returns_the_s2_rows() -> None:
    fn = _function(_tree(DETECTION), "detect_single_patch_640")
    whole = fn.body[1]
    assert isinstance(whole, ast.If)
    assert (
        ast.unparse(whole.test) == "_use_whole and 'letterbox' not in preprocess_modes"
    )
    nv12 = whole.body[1]
    assert (
        isinstance(nv12, ast.If)
        and ast.unparse(nv12.test) == "getattr(pool, 'use_nv12', False)"
    )
    assert [ast.unparse(s) for s in nv12.orelse] == [
        "raw_dets = detector.detect_raw(pool.frame_buffer.unsqueeze(0))",
        "return (raw_dets[0, :, :4], raw_dets[0, :, 4], raw_dets[0, :, 5])",
    ]
    native = [
        ast.unparse(s) for s in _function(_tree(DETECTION), "detect_native_640").body
    ]
    assert native[-1] == "return (boxes, scores, classes, False, None)"


def test_s2_twin_versions() -> None:
    # detector_s2.hpp reads Inductor's lowering of this torch / triton; a new
    # one is a new lowering to re-read (and the fixture to re-render).
    fixture = json.loads(S2_FIXTURE.read_text(encoding="utf-8"))
    assert (fixture["torch"], fixture["triton"]) == (TORCH_VERSION, TRITON_VERSION)
    for dist, want in (("torch", TORCH_VERSION), ("triton", TRITON_VERSION)):
        try:
            got = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        assert got.split("+")[0] == want.split("+")[0]


def test_harness_runs_the_pinned_tail() -> None:
    tree = _tree(HARNESS)
    body = [ast.unparse(s) for s in _function(tree, "oracle_s2").body[1:]]
    assert body == [
        "detections = mgd._postprocess_mamba_fixed(cls_preds, reg_preds, stride, conf_thr, "
        "max(max_det, nms_pad), anchors=anchors, anchor_strides=anchor_strides, "
        "small_p3_max_threshold=small_p3, box_scale_x=sx, box_scale_y=sy, _compile=compile_)",
        "raw = detections.clone()",
        "detections[:, :, x_idx] *= sx",
        "detections[:, :, y_idx] *= sy",
        "return (raw, detections)",
    ]
    resize = [
        ast.unparse(n)
        for n in ast.walk(_function(tree, "resize"))
        if isinstance(n, ast.Call)
    ]
    assert (
        "F.interpolate(frame_chw.unsqueeze(0), size=(self.img_size, self.img_size), "
        "mode='bilinear', align_corners=False)"
    ) in resize
    assert _assignments(tree, "self.stride") == [
        "torch.tensor([8.0, 16.0, 32.0], device=dev)"
    ]
    assert _assignments(tree, "self.nms_pad") == ["0"]


def test_lineage_fixture_is_the_frozen_lineage() -> None:
    if not LINEAGE.exists():
        pytest.skip("models/ is not present")
    assert LINEAGE_FIXTURE.read_bytes() == LINEAGE.read_bytes()


def test_attestation_is_bound_and_consistent() -> None:
    att = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    lineage = json.loads(LINEAGE_FIXTURE.read_text(encoding="utf-8"))
    fl = att["frozen_lineage"]
    assert fl["sha256"] == _sha256(LINEAGE_FIXTURE)
    assert fl["tool_commit"] == lineage["tool"]["git_commit"]
    assert fl["torchscript_sha256"] == lineage["torchscript"]["sha256"]
    assert fl["torchscript_content_sha256"] == lineage["torchscript"]["content_sha256"]
    assert fl["op_library_sha256"] == lineage["op_library"]["sha256"]
    op = att["op_library"]
    assert op["path"] == lineage["op_library"]["path"]
    assert op["sha256"] != fl["op_library_sha256"]
    assert not any(n.startswith(("libpython", "libtorch_python")) for n in op["needed"])
    for rel, blob in op["sources_git_blob"].items():
        head = subprocess.run(
            ["git", "rev-parse", f"HEAD:{rel}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert head == blob, f"{rel} changed since the attestation"
    harness = _module(HARNESS, "native_detector_parity_pins")
    repro = att["a_l_reproduction"]
    assert repro["identical"] is True
    assert repro["reference"] == harness.PR2L_A_L_REFERENCE
    assert "--double-buffer" in repro["mot17_argv"]
    for seq, ref in harness.PR2L_A_L_REFERENCE_TXT_SHA256.items():
        assert repro["txt_sha256"][seq] == {"reference": ref, "observed": ref}
        packet = REPO / harness.PR2L_A_L_REFERENCE / f"{seq}.txt"
        if packet.exists():
            assert _sha256(packet) == ref


def test_s2_fixture_is_fresh() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the S2 fixture is the oracle's compiled CUDA output")
    module = _module(RENDER, "render_shipping_detector_s2_fixture")
    assert module.main(["--check"]) == 0, (
        "tests/native/fixtures/shipping_detector_s2.json is stale; re-run "
        "scripts/model/render_shipping_detector_s2_fixture.py"
    )
