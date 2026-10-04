"""Oracle pins for the native double-buffer runtime (#465 Phase B PR-10, U5).

``shipping/src/double_buffer_runtime.cpp`` and the graph modes of
``detector_host.cpp`` / ``post_detector_host.cpp`` copy the oracle's schedule
and graph lifecycles. Those are code structure, not config values, so they
are pinned here at source level (the parity harness checks the outcome):

* **frame loop** -- ``_schedule(1)`` primes frame 1; each iteration launches
  ``_schedule(frame_id + 1)`` (when ``frame_id < frame_end``) *before*
  ``_run_frame(frame_id, prepared_detection=pending)``; one detection in flight;
* **parity** -- frame k uses ``double_buffer_pools[(k - 1) % 2]`` and
  ``double_buffer_events[(k - 1) % 2]``; ``EvalPipeline`` builds two pools,
  one side stream and two event pairs only when ``_double_buffer_eligible``,
  which requires the env switch and the ``event`` barrier;
* **launch** -- ``input_ready`` is recorded on the main stream, the side
  stream waits on it, ``_run_detect(..., synchronize=False)`` runs under the
  side stream, the outputs are cloned, then ``ready_event`` is recorded;
  ``_run_frame`` makes the main stream wait on ``ready_event`` and takes the
  prepared pool (GMC reads its frame buffer);
* **whole-detect graph** -- keyed by frame shape + image dims + NMS pad;
  ``set_whole_graph_img_dims`` keeps the graphs for the same dims and
  otherwise clears them and the warm flag; a miss warms up once (when not
  warm) and captures through ``graphed_callables`` on ``frame.clone()``; a
  cache of ten is cleared first; ``make_graphed_callables`` warms up three
  times;
* **GMC graph** -- the capture branch copies the frame into
  ``_gmc_frame_buf``, runs the estimate eagerly, synchronizes, captures and
  does not replay; later frames copy and replay;
* **main NMS graph** -- copy_pad, then (first frame) the capture helper's
  eager warm-up and capture, then the replay on every frame;
* **capture mode** -- ``thread_local``.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "src/saccade/perception/eval"
EVALUATOR = EVAL / "evaluator.py"
PIPELINE = EVAL / "pipeline.py"
STAGES = EVAL / "stages.py"
CAPTURE = EVAL / "cuda_capture.py"
TORCH_GRAPHS = EVAL / "_torch_graphs.py"
DETECTOR = REPO / "src/saccade/perception/temporal_yolo/mamba_gated_detector.py"


def _function(path: Path, name: str, cls: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope: ast.AST = tree
    if cls is not None:
        scope = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
        )
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def _index(text: str, *needles: str) -> list[int]:
    out = []
    for n in needles:
        i = text.find(n)
        assert i >= 0, f"{n!r} not found"
        out.append(i)
    return out


def test_frame_loop_launches_next_detection_before_tracking_this_frame() -> None:
    fn = _src(_function(EVALUATOR, "run_eval"))
    a, b, c = _index(
        fn,
        "pending = _schedule(1)",
        "next_pending = _schedule(frame_id + 1) if frame_id < _seq_state.frame_end else None",
        "prepared_detection=pending",
    )
    assert a < b < c
    assert "pending = next_pending" in fn


def test_parity_selects_pool_and_events() -> None:
    fn = _src(_function(EVALUATOR, "run_eval"))
    assert "_seq_state.double_buffer_pools[(frame_id - 1) % 2]" in fn
    assert "_seq_state.double_buffer_events[(frame_id - 1) % 2]" in fn


def test_two_pools_one_stream_two_event_pairs_when_eligible() -> None:
    init = _src(_function(PIPELINE, "__init__", cls="EvalPipeline"))
    i_if, i_pool, i_stream, i_ev = _index(
        init,
        "if _double_buffer_eligible(cfg, detector, profile_stages):",
        "next_pool = AdaptiveFramePool(h_orig, w_orig)",
        "double_buffer_stream = torch.cuda.Stream()",
        "for _ in range(2):",
    )
    assert i_if < i_pool < i_stream < i_ev
    elig = _src(_function(PIPELINE, "_double_buffer_eligible"))
    assert "os.getenv('SACCADE_DOUBLE_BUFFER', '0')" in elig
    assert "_detect_barrier_mode() == 'event'" in elig
    assert "getattr(detector, 'use_whole_graph', False)" in elig


def test_launch_orders_input_ready_detect_clone_ready() -> None:
    fn = _src(_function(STAGES, "_launch_double_buffer_detect"))
    steps = _index(
        fn,
        "input_ready.record(main_stream)",
        "with torch.cuda.stream(stream):",
        "stream.wait_event(input_ready)",
        "synchronize=False",
        "fused_boxes = fused_boxes.clone()",
        "fused_classes = fused_classes.clone()",
        "ready_event.record()",
    )
    assert steps == sorted(steps)
    frame = _src(_function(EVALUATOR, "_run_frame"))
    w, p = _index(
        frame,
        "torch.cuda.current_stream().wait_event(prepared_detection.ready_event)",
        "state.pool = pool",
    )
    assert w < p
    assert "pool = prepared_detection.pool" in frame


def test_whole_detect_graph_key_cache_and_dims() -> None:
    fwd = _src(_function(DETECTOR, "_forward_whole_graph"))
    assert (
        "key = tuple(frame.shape) + self._whole_graph_img_shape + (self._whole_graph_nms_pad,)"
        in fwd
    )
    w, c = _index(
        fwd, "self._whole_graph_warmup(frame)", "self._whole_graph_capture(frame)"
    )
    assert w < c and "if not self._whole_graph_warm:" in fwd
    cap = _src(_function(DETECTOR, "_whole_graph_capture"))
    clr, smp, gc = _index(
        cap,
        "if len(self._whole_graphed_callables) >= 10:",
        "sample = frame.clone()",
        "graphed_callables(self._whole_graph_fn, (sample,), label='detector.whole')",
    )
    assert clr < smp < gc
    dims = _src(_function(DETECTOR, "set_whole_graph_img_dims"))
    same = next(
        n
        for n in ast.walk(_function(DETECTOR, "set_whole_graph_img_dims"))
        if isinstance(n, ast.If)
        and _src(n.test) == "(h_orig, w_orig) == self._whole_graph_img_shape"
    )
    assert [_src(s) for s in same.body] == ["return"]
    r, clear, warm = _index(
        dims,
        "self._whole_graph_img_shape = (h_orig, w_orig)",
        "self._whole_graphed_callables.clear()",
        "self._whole_graph_warm = False",
    )
    assert r < clear < warm
    warmup = _src(_function(DETECTOR, "_whole_graph_warmup"))
    assert "warm = frame.clone()" in warmup and "torch.cuda.synchronize()" in warmup
    mgc = _function(TORCH_GRAPHS, "make_graphed_callables")
    defaults = dict(
        zip(
            [a.arg for a in mgc.args.args][-len(mgc.args.defaults) :],
            [_src(d) for d in mgc.args.defaults],
        )
    )
    assert defaults["num_warmup_iters"] == "3"


def test_gmc_capture_branch_does_not_replay() -> None:
    fn = _function(STAGES, "_run_gmc_estimate")
    capture = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If) and _src(n.test) == "_gmc_cuda_graph[0] is None"
    )
    body = "\n".join(_src(s) for s in capture.body)
    steps = _index(
        body,
        "_gmc_frame_buf.copy_(_frame_gmc)",
        "gmc_estimator.estimate_into_direct(_gmc_frame_buf.data_ptr()",
        "torch.cuda.synchronize()",
        "graph_capture(_g, label='gmc.direct')",
    )
    assert steps == sorted(steps)
    assert ".replay()" not in body
    steady = "\n".join(_src(s) for s in capture.orelse)
    a, b = _index(
        steady, "_gmc_frame_buf.copy_(_frame_gmc)", "_gmc_cuda_graph[0].replay()"
    )
    assert a < b


def test_main_nms_graph_copy_pad_capture_then_replay() -> None:
    fn = _src(_function(STAGES, "_run_nms"))
    steps = _index(
        fn,
        "copy_pad_detections(",
        "_capture_main_nms_graph_nocopyback(state, is_tiled=is_tiled)",
        "state.main_nms_graph_nocopyback.replay()",
        "perception_pipeline.process_detections_split_pipeline_graphed(",
    )
    assert steps == sorted(steps)
    cap = _src(_function(STAGES, "_capture_main_nms_graph_nocopyback"))
    e, s, g = _index(
        cap,
        "_perception_pipeline.process_detections_main_nms_graph_nocopyback(",
        "torch.cuda.synchronize()",
        "graph_capture(_graph, label='nms.main_nocopyback')",
    )
    assert e < s < g
    # One graph per EvalPipeline, i.e. per sequence.
    init = _src(_function(PIPELINE, "__init__", cls="EvalPipeline"))
    assert "self.main_nms_graph_nocopyback: Any = None" in init


def test_capture_mode_is_thread_local() -> None:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    modes = [
        _src(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and _src(n.targets[0]) == "CAPTURE_ERROR_MODE"
    ]
    assert modes == ["'thread_local'"]
