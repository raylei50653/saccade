"""Oracle pins for the native end-to-end serial runtime (#465 Phase B PR-9).

``shipping/src/serial_runtime.cpp`` owns no stage of its own; it owns which
objects live for the whole run and which are rebuilt per sequence. It copies
the oracle's split (``evaluator.run_eval`` + ``pipeline.EvalPipeline``), which
is code structure rather than a config value, so it is pinned here at source
level:

* **per run** -- the detector, the ``PerceptionPipeline`` and the
  ``GlobalTrackIdMapper`` are bound before ``for seq in cfg.seqs`` and never
  rebound inside it; every sequence's ``EvalPipeline`` receives those same
  objects;
* **per sequence** -- ``EvalPipeline.__init__`` sets the detector's coordinate
  scales from seqinfo.ini (``set_whole_graph_img_dims(h_orig, w_orig)``) and
  builds the frame pool (``AdaptiveFramePool(h_orig, w_orig)``) and the frame
  streamer for that sequence's ``img1``.

The run-global id mapper is why the PR-9 harness compares MOT txt after
relabeling (``native_track_parity.py``): shipping numbers ids per sequence
(boundary §5 B3).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVALUATOR = REPO / "src/saccade/perception/eval/evaluator.py"
PIPELINE = REPO / "src/saccade/perception/eval/pipeline.py"


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


def _sequence_loop(fn: ast.FunctionDef) -> ast.For:
    loops = [
        n
        for n in fn.body
        if isinstance(n, ast.For)
        and ast.unparse(n.target) == "seq"
        and ast.unparse(n.iter) == "cfg.seqs"
    ]
    assert len(loops) == 1, (
        "run_eval must have exactly one top-level `for seq in cfg.seqs`"
    )
    return loops[0]


def _bound_names(nodes: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for stmt in nodes:
        for n in ast.walk(stmt):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    out |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            elif isinstance(n, (ast.For, ast.With)):
                tgt = n.target if isinstance(n, ast.For) else None
                if tgt is not None:
                    out |= {x.id for x in ast.walk(tgt) if isinstance(x, ast.Name)}
    return out


def _calls(node: ast.AST, func: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == func
    ]


def test_per_run_objects_are_bound_before_the_sequence_loop() -> None:
    fn = _function(EVALUATOR, "run_eval")
    loop = _sequence_loop(fn)
    before = fn.body[: fn.body.index(loop)]
    bound_before = _bound_names(before) | {a.arg for a in fn.args.args}
    for name in ("detector", "perception_pipeline", "global_id_mapper"):
        assert name in bound_before, f"{name} is not bound before the sequence loop"
    assert any(_calls(s, "PerceptionPipeline") for s in before)
    assert any(_calls(s, "GlobalTrackIdMapper") for s in before)
    inside = _bound_names(loop.body)
    for name in ("detector", "perception_pipeline", "global_id_mapper"):
        assert name not in inside, f"{name} is rebound per sequence"


def test_each_sequence_gets_the_run_objects() -> None:
    loop = _sequence_loop(_function(EVALUATOR, "run_eval"))
    calls = _calls(loop, "EvalPipeline")
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("detector") == "detector"
    assert kw.get("global_id_mapper") == "global_id_mapper"
    assert kw.get("seq") == "seq"
    passed = set(kw.values())
    assert "perception_pipeline" in passed


def test_per_sequence_detector_scales_pool_and_streamer() -> None:
    init = _function(PIPELINE, "__init__", cls="EvalPipeline")
    scales = _calls(init, "detector.set_whole_graph_img_dims")
    assert [ast.unparse(c) for c in scales] == [
        "detector.set_whole_graph_img_dims(h_orig, w_orig)"
    ]
    pools = {ast.unparse(c) for c in _calls(init, "AdaptiveFramePool")}
    assert pools == {"AdaptiveFramePool(h_orig, w_orig)"}
    streamers = {ast.unparse(c) for c in _calls(init, "TorchvisionGpuStreamer")}
    assert streamers == {"TorchvisionGpuStreamer(seq_path / 'img1')"}
