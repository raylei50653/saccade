"""Oracle pins for the native ingest (#465 Phase B PR-7, U3b-1).

The native ingest (``shipping/src/ingest_plan.cpp``, ``ingest_host.cpp``,
``ingest_normalize.cu``) re-implements the oracle's frame ingest in the
headline configuration. Its parity is measured on real frames by
``scripts/eval/diagnostics/native_ingest_parity.py`` and on committed inputs
by the golden fixture; these checks pin, at source level, the oracle facts
that are code literals rather than config values:

* **listing** -- ``TorchvisionGpuStreamer`` lists
  ``sorted(str(path.absolute()) for path in img_dir.glob('*.jpg'))`` of
  ``seq_path / 'img1'`` (selected by ``SACCADE_GPU_DECODE == '1'``, the
  exporter's ``ingest.gpu_decode`` gate) and yields them in order;
* **decode** -- ``decode_jpeg(read_file(f), device='cuda',
  mode=ImageReadMode.RGB)``, then a ``permute(1, 2, 0)`` HWC view; the decoder
  is torchvision 0.26.0's, whose source the native decoder mirrors;
* **geometry and bound** -- ``seqinfo.ini`` ``imWidth``/``imHeight`` size a
  float32 ``[3, h, w]`` frame buffer; ``frame_end = min(max_frames or
  int(1e9), seqLength)``; both frame loops call ``next`` once per frame and
  stop at ``StopIteration``;
* **ingest op** -- ``_run_detect``'s non-NV12 branch is
  ``torch.div(frame_gpu.permute(2, 0, 1), 255.0, out=pool.frame_buffer)``
  followed by ``apply_frame_preprocess``, which does nothing for an empty mode
  list; the parity harness runs this same op;
* **fixture** -- ``tests/native/fixtures/shipping_ingest.json`` is fresh
  (the CPU sections always; the CUDA normalize table and decodes only where
  CUDA is available).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "src" / "saccade" / "perception" / "eval"
RENDER = REPO / "scripts" / "model" / "render_shipping_ingest_fixture.py"
HARNESS = REPO / "scripts" / "eval" / "diagnostics" / "native_ingest_parity.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _assigns(node: ast.AST) -> dict[str, str]:
    """``{target: unparsed value}`` for single-target assignments under node."""
    out: dict[str, str] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            out[ast.unparse(n.targets[0])] = ast.unparse(n.value)
    return out


def _calls(node: ast.AST) -> list[str]:
    return [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Call)]


def test_streamer_lists_sorts_and_decodes_rgb() -> None:
    cls = _class(_tree(EVAL / "streaming.py"), "TorchvisionGpuStreamer")
    init = _assigns(_function(cls, "__init__"))
    assert init["self.img_files"] == (
        "sorted((str(path.absolute()) for path in img_dir.glob('*.jpg')))"
    )
    assert init["self._read_file"] == "read_file"
    assert init["self._decode"] == "decode_jpeg"
    assert init["self._rgb"] == "ImageReadMode.RGB"
    imports = [
        (n.module, sorted(a.name for a in n.names))
        for n in ast.walk(_function(cls, "__init__"))
        if isinstance(n, ast.ImportFrom)
    ]
    assert ("torchvision.io", ["ImageReadMode", "decode_jpeg", "read_file"]) in imports

    worker = _function(cls, "_decode_worker")
    loops = [n for n in ast.walk(worker) if isinstance(n, ast.For)]
    assert [ast.unparse(n.iter) for n in loops] == ["self.img_files"]
    body = _assigns(loops[0])
    assert body["data"] == "self._read_file(f)"
    assert body["img_chw"] == "self._decode(data, device='cuda', mode=self._rgb)"
    assert body["img_hwc"] == "img_chw.permute(1, 2, 0)"
    assert "out_queue.put((img_hwc, ready))" in _calls(loops[0])


def test_oracle_decoder_is_the_mirrored_torchvision() -> None:
    # ingest_host.hpp mirrors torchvision 0.26.0's decode_jpegs_cuda.cpp; a new
    # torchvision is a new decoder to re-read (and its bundled nvJPEG must
    # still be the pinned one: shipping/CMakeLists.txt fails configure if not).
    assert importlib.metadata.version("torchvision").split("+")[0] == "0.26.0"


def test_pipeline_selects_the_gpu_streamer_and_sizes_the_pool() -> None:
    tree = _tree(EVAL / "pipeline.py")
    assigns = _assigns(tree)
    assert assigns["seq_path"] == "Path(cfg.core.data_root) / cfg.core.split / seq"
    assert assigns["w_orig"] == "config.getint('Sequence', 'imWidth')"
    assert assigns["h_orig"] == "config.getint('Sequence', 'imHeight')"
    assert assigns["frame_end"] == (
        "min(max_frames or int(1000000000.0), config.getint('Sequence', 'seqLength'))"
    )
    assert assigns["pool"] == "AdaptiveFramePool(h_orig, w_orig)"
    assert (
        assigns["nv12_direct_from_hwc"]
        == "pool.use_nv12 and (not cfg.preprocess_modes)"
    )
    gates = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "os.environ.get('SACCADE_GPU_DECODE') == '1'"
    ]
    assert len(gates) == 1
    assert ast.unparse(gates[0].body[0]) == (
        "streamer: Any = TorchvisionGpuStreamer(seq_path / 'img1')"
    )
    # The exporter's step is that same gate.
    config = json.loads(
        (REPO / "configs/shipping/mamba_whole_graph.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["host_params"]["steps"]["ingest.gpu_decode"] is True
    assert config["host_params"]["cfg"]["preprocess_modes"] == []

    pool = _function(_class(_tree(EVAL / "pool.py"), "AdaptiveFramePool"), "__init__")
    assert _assigns(pool)["self.frame_buffer"] == (
        "torch.zeros((3, h, w), device=device, dtype=torch.float32)"
    )


def test_frame_loops_take_one_frame_each_and_stop_at_stop_iteration() -> None:
    tree = _tree(EVAL / "evaluator.py")
    loops = [
        ast.unparse(n.iter)
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and "frame_end" in ast.unparse(n.iter)
    ]
    assert loops == ["range(1, _seq_state.frame_end + 1)"] * 2
    run_frame = _function(tree, "_run_frame")
    fetch = [
        n
        for n in ast.walk(run_frame)
        if isinstance(n, ast.Try) and "next(stream_iter)" in ast.unparse(n.body)
    ]
    assert len(fetch) == 1
    assert [ast.unparse(h.type) for h in fetch[0].handlers] == ["StopIteration"]
    assert ast.unparse(fetch[0].handlers[0].body) == "return False"
    schedule = _function(tree, "_schedule")
    assert "frame_gpu = next(_seq_state.stream_iter)" in ast.unparse(schedule)


def test_run_detect_ingest_op() -> None:
    fn = _function(_tree(EVAL / "stages.py"), "_run_detect")
    ifexps = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.IfExp) and ast.unparse(n.test) == "nv12_direct_from_hwc"
    ]
    assert len(ifexps) == 1
    rgb_branch = ifexps[0].orelse
    assert isinstance(rgb_branch, ast.Tuple)
    steps = [ast.unparse(e) for e in rgb_branch.elts]
    assert (
        steps[0]
        == "torch.div(frame_gpu.permute(2, 0, 1), 255.0, out=pool.frame_buffer)"
    )
    assert steps[1].startswith(
        "apply_frame_preprocess(pool.frame_buffer, cfg.preprocess_modes,"
    )

    pre = _function(_tree(EVAL / "preprocess.py"), "apply_frame_preprocess")
    first = pre.body[0]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == "not modes"
    assert [ast.unparse(s) for s in first.body] == ["return"]


def test_harness_runs_the_pinned_op_and_reads() -> None:
    tree = _tree(HARNESS)
    op = _function(tree, "oracle_ingest")
    statements = [ast.unparse(n) for n in op.body if isinstance(n, ast.Expr)]
    assert (
        statements[-1]
        == "torch.div(frame_hwc.permute(2, 0, 1), 255.0, out=frame_buffer)"
    )
    bounds = _assigns(_function(tree, "oracle_sequence_bounds"))
    pipeline = _assigns(_tree(EVAL / "pipeline.py"))
    for key in ("w_orig", "h_orig", "frame_end"):
        assert bounds[key] == pipeline[key]
    assert "TorchvisionGpuStreamer(seq_path / 'img1')" in _calls(
        _function(tree, "run_sequence")
    )


def _render_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "render_shipping_ingest_fixture", RENDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ingest_fixture_cpu_sections_are_fresh() -> None:
    module = _render_module()
    committed = json.loads(module.OUTPUT.read_text(encoding="utf-8"))
    fresh = module.render(gpu=False)
    assert module._cpu_view(fresh) == module._cpu_view(committed), (
        "tests/native/fixtures/shipping_ingest.json is stale; re-run "
        "scripts/model/render_shipping_ingest_fixture.py"
    )


def test_ingest_fixture_gpu_sections_are_fresh() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the normalize table and decodes are the oracle's CUDA outputs")
    module = _render_module()
    assert module.OUTPUT.read_text(encoding="utf-8") == module._dump(
        module.render(gpu=True)
    ), (
        "tests/native/fixtures/shipping_ingest.json is stale; re-run "
        "scripts/model/render_shipping_ingest_fixture.py"
    )
