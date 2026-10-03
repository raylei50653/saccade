"""Oracle pins for the native MOT output path (#465 Phase B PR-6, U4).

The native output path (``shipping/src/mot_output.cpp``,
``shipping/src/sequence_output.cpp``) re-implements the oracle's per-sequence
emit and tail in the headline configuration. Its parity is measured end to end
by ``saccade_replay`` (MOT txt of a replayed dump vs the Python serial run) and
function by function by the golden fixture; these checks pin, at source level,
the oracle facts the native side hard-codes:

* **fast emit is the function the fixture runs** -- both of the oracle's emit
  sites (``stages._run_emit`` and ``evaluator._flush_deferred_emit``) call
  ``helpers.fast_emit_mot_lines``, and it formats each row as
  ``frame,id,x1,y1,x2-x1,y2-y1,score,-1,-1,-1`` with ``.2f``/``.4f``;
* **ids** -- ``GlobalTrackIdMapper`` assigns ids from 1 in first-appearance
  order (per sequence that is the shipping rule; across sequences the counter
  continues, which parity undoes by an offset);
* **tail** -- the sequence tail calls ``post_merge.interpolate_tracklets`` with
  ``cfg.interpolate_max_gap`` / ``_min_track_len`` / ``_min_h`` and writes
  ``"\\n".join(results_lines)`` (no trailing newline);
* **fixture** -- ``tests/native/fixtures/shipping_mot_output.json`` (replayed
  through the C++ twins by ``tests/native/test_shipping_mot_output.cpp``) is
  fresh against the Python functions.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "src" / "saccade" / "perception" / "eval"
RENDER = REPO / "scripts" / "model" / "render_shipping_mot_output_fixture.py"


def _tree(name: str) -> ast.Module:
    return ast.parse((EVAL / name).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == name)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == name)
        )
    ]


def _imports(tree: ast.Module, module: str) -> dict[str, str]:
    """``{local name: imported name}`` for ``from .<module> import ...``."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                out[alias.asname or alias.name] = alias.name
    return out


def test_emit_sites_use_helpers_fast_emit() -> None:
    for name, fn in (
        ("stages.py", "_run_emit"),
        ("evaluator.py", "_flush_deferred_emit"),
    ):
        tree = _tree(name)
        assert _imports(tree, "helpers").get("_fast_emit_mot_lines") == (
            "fast_emit_mot_lines"
        ), name
        assert _calls(_function(tree, fn), "_fast_emit_mot_lines"), f"{name}:{fn}"


def test_fast_emit_line_format() -> None:
    fn = _function(_tree("helpers.py"), "fast_emit_mot_lines")
    formats = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.JoinedStr) and "-1,-1,-1" in ast.unparse(n)
    ]
    assert formats == [
        "f'{frame_id},{gid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1'"
    ]
    src = ast.unparse(fn)
    assert "w = x2 - x1" in src and "h = y2 - y1" in src
    assert "gid = global_id_mapper.map(seq, int(ids_np[i]))" in src
    assert "for i in range(count):" in src


def test_ids_are_first_appearance_from_one() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from saccade.perception.eval.tracking import GlobalTrackIdMapper

    m = GlobalTrackIdMapper()
    assert [m.map("A", i) for i in (7, 3, 7, 9, 3)] == [1, 2, 1, 3, 2]
    # A second sequence continues the run-global counter: parity subtracts it.
    assert [m.map("B", i) for i in (7, 1)] == [4, 5]


def test_tail_interpolation_call_and_write() -> None:
    tree = _tree("evaluator.py")
    assert _imports(tree, "post_merge").get("interpolate_tracklets") == (
        "interpolate_tracklets"
    )
    calls = _calls(tree, "interpolate_tracklets")
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw == {
        "max_gap": "cfg.interpolate_max_gap",
        "min_track_len": "cfg.interpolate_min_track_len",
        "min_h": "cfg.interpolate_min_h",
    }
    assert [ast.unparse(a) for a in calls[0].args] == ["_seq_state.results_lines"]
    writes = [
        ast.unparse(c.args[0])
        for c in _calls(tree, "write_text")
        if c.args and "results_lines" in ast.unparse(c.args[0])
    ]
    assert writes == ["'\\n'.join(_seq_state.results_lines)"]


def test_mot_output_fixture_is_fresh() -> None:
    spec = importlib.util.spec_from_file_location(
        "render_shipping_mot_output_fixture", RENDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.OUTPUT.read_text(encoding="utf-8") == module.render(), (
        "tests/native/fixtures/shipping_mot_output.json is stale; re-run "
        "scripts/model/render_shipping_mot_output_fixture.py"
    )
