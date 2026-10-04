"""Oracle pins for the native post-detector replay host (#465 Phase B PR-5, U3a).

The native host (``shipping/src/post_detector_host.cpp``) re-implements the
oracle's tracker lifecycle and the host-side detection filters. Its parity is
measured end to end by ``saccade_replay`` against a Python serial dump; these
checks pin, at source level, the oracle facts the host hard-codes, so an oracle
change that would silently break parity fails here first:

* **tracker pre-roll** -- ``GraphedTrackerUpdate`` runs ``update_into`` on its
  zeroed scratch inputs before a sequence's first real update: once in
  ``_warmup`` and ``num_warmup_iters`` (default 3) times in
  ``make_graphed_callables``' warm-up loop. Capture itself records kernels
  without executing them (its host-side code only bumps
  ``processed_frame_count_``, which no output reads). The host's
  ``kGraphedTrackerUpdatePreRoll`` must equal 1 + 3;
* **pre-roll is lazy** -- capture happens on the first ``copy_inputs``, before
  any real input is copied, so the pre-roll sees zero inputs and the identity
  warp (``torch.eye(2, 3)``);
* **tracker update arguments** -- every oracle ``update_into`` call passes
  ``num_dets = max_assoc``, no embeddings, ``light_factor = 0.0``,
  ``mid_thresh_scale = 1.0`` and ``out_capacity = max_objs``; the host's
  ``tracker_update`` passes the same, from both of its call sites;
* **detection filter fixture** -- ``tests/native/fixtures/
  shipping_detection_filters.json`` (replayed through the C++ filter twins by
  ``tests/native/test_shipping_detection_filters.cpp``) is fresh against the
  Python filter functions.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACKER_GPU = REPO / "src" / "saccade" / "perception" / "tracking" / "tracker_gpu.py"
TORCH_GRAPHS = REPO / "src" / "saccade" / "perception" / "eval" / "_torch_graphs.py"
PLAN_HPP = REPO / "shipping" / "include" / "saccade_shipping" / "post_detector_plan.hpp"
HOST_CPP = REPO / "shipping" / "src" / "post_detector_host.cpp"
RENDER = REPO / "scripts" / "model" / "render_shipping_detection_filters_fixture.py"


def _class_methods(path: Path, cls: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(f"{cls} not found in {path}")


def _calls(node: ast.AST, attr: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == attr)
            or (isinstance(n.func, ast.Name) and n.func.id == attr)
        )
    ]


def _gtu() -> dict[str, ast.FunctionDef]:
    return _class_methods(TRACKER_GPU, "GraphedTrackerUpdate")


def _oracle_pre_roll() -> int:
    methods = _gtu()
    warmup_updates = len(_calls(methods["_warmup"], "update_into"))

    capture = methods["_capture"]
    assert len(_calls(capture, "_warmup")) == 1
    graphed = _calls(capture, "graphed_callables")
    assert len(graphed) == 1
    assert "num_warmup_iters" not in {kw.arg for kw in graphed[0].keywords}, (
        "GraphedTrackerUpdate now sets num_warmup_iters; update kGraphedTrackerUpdatePreRoll"
    )

    tree = ast.parse(TORCH_GRAPHS.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "make_graphed_callables"
    )
    params = [a.arg for a in fn.args.args]
    defaults = dict(
        zip(params[len(params) - len(fn.args.defaults) :], fn.args.defaults)
    )
    default = defaults["num_warmup_iters"]
    assert isinstance(default, ast.Constant)
    # The warm-up loop calls each callable once per iteration.
    loops = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Call)
        and ast.unparse(n.iter) == "range(num_warmup_iters)"
    ]
    assert len(loops) == 1
    assert len(_calls(loops[0], "func")) == 1
    return warmup_updates + int(default.value)


def test_pre_roll_matches_graphed_tracker_update() -> None:
    oracle = _oracle_pre_roll()
    assert oracle == 4
    match = re.search(
        r"inline constexpr int kGraphedTrackerUpdatePreRoll = (\d+);",
        PLAN_HPP.read_text(encoding="utf-8"),
    )
    assert match is not None
    assert int(match.group(1)) == oracle


def test_pre_roll_runs_on_zero_inputs_before_the_first_copy() -> None:
    methods = _gtu()
    copy_inputs = methods["copy_inputs"]
    src = ast.unparse(copy_inputs)
    capture_at = src.index("self._capture()")
    assert capture_at < src.index("self.d_boxes[:n].copy_(")
    assert capture_at < src.index("self.d_gmc.copy_(")
    # The scratch inputs start zeroed and the warp at identity.
    init = ast.unparse(methods["__init__"])
    for name in ("d_boxes", "d_scores", "d_classes"):
        assert re.search(rf"self\.{name} = torch\.zeros\(", init), name
    assert "torch.eye(2, 3, dtype=torch.float32" in init
    host = HOST_CPP.read_text(encoding="utf-8")
    assert "const float identity[6] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};" in host


def test_oracle_update_into_arguments() -> None:
    methods = _gtu()
    calls = _calls(methods["_warmup"], "update_into") + _calls(
        methods["_capture"], "update_into"
    )
    assert len(calls) == 2  # _warmup and the captured _graph_fn
    for call in calls:
        args = [ast.unparse(a) for a in call.args]
        assert len(args) == 16
        assert args[3] == "self._max_assoc"  # num_dets: the padded capacity
        assert args[11] == "0"  # embeddings: none
        assert args[13] == "0.0"  # light_factor
        assert args[14] == "1.0"  # mid_thresh_scale
        assert args[15] == "self._max_objs"  # out_capacity


def test_host_update_into_arguments() -> None:
    host = HOST_CPP.read_text(encoding="utf-8")
    body = re.search(
        r"void PostDetectorHost::tracker_update\(int num_dets\) \{(.*?)\n\}", host, re.S
    )
    assert body is not None
    call = " ".join(body.group(1).split())
    assert (
        "update_into(b.trk_boxes, b.trk_scores, b.trk_classes, num_dets, stream_, "
        "b.out_boxes, b.out_scores, b.out_ids, b.out_classes, b.out_det_idx, b.out_count, "
        "/*embeddings_ptr=*/nullptr, b.trk_gmc, /*light_factor=*/0.0f, "
        "/*mid_thresh_scale=*/1.0f, plan_.max_objects)"
    ) in call
    # Pre-roll, the graph capture (GraphMode::Captured, PR-10) and the eager
    # per-frame update all pass the padded capacity.
    sites = re.findall(r"\btracker_update\(([^)]*)\);", host)
    assert sites == ["plan_.max_assoc", "plan_.max_assoc", "plan_.max_assoc"]


def test_detection_filters_fixture_is_fresh() -> None:
    spec = importlib.util.spec_from_file_location(
        "render_shipping_detection_filters_fixture", RENDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.OUTPUT.read_text(encoding="utf-8") == module.render(), (
        "tests/native/fixtures/shipping_detection_filters.json is stale; re-run "
        "scripts/model/render_shipping_detection_filters_fixture.py"
    )
