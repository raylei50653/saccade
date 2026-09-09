"""Failure-time capture diagnostics on every production capture entrance (#374).

The #340 incidence campaign reproduced a capture invalidation on the **main eval
thread** and kept no stream-level evidence, because the only unconditional dump in
the tree hung off the decode worker's exception handler. These tests pin the
observation surface that replaces it.

What is asserted is deliberately weaker than "a capture failed": the wrapper
reports *any exception escaping a capture entrance*, and the line it prints is a
post-failure snapshot. Nothing here claims the dump identifies which capture was
invalidated, or that the state it reports is the state at the failing instant.

The four properties the closure conditions turn on are: the dump reaches a
main-thread ``make_graphed_callables`` failure; the original error is re-raised
unchanged; a diagnostic that itself fails is recorded rather than swallowed; and
the enumeration of capture entrances stays closed. Everything is device-free --
cudart and the torch stream accessors are stubbed, so the assertions are about
this repo's control flow and line grammar, not about CUDA behaviour.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import builtins
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest
import torch

from saccade.perception.eval import cuda_capture
from saccade.perception.eval.cuda_capture import (
    capture_diagnostics,
    graph_capture,
    graphed_callables,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CURRENT = 0xAAA0
_DEFAULT = 0xBBB0
_LEGACY = 1


class _FakeStream:
    def __init__(self, ptr: int) -> None:
        self.cuda_stream = ptr


def _fake_cudart(flags: dict[int, int], status: dict[int, int]):
    """A cudart whose two probe entry points answer from fixed tables."""

    def get_flags(stream, out):
        out._obj.value = flags.get(stream.value, 1)
        return 0

    def is_capturing(stream, out):
        out._obj.value = status.get(stream.value, 0)
        return 0

    class _Lib:
        cudaStreamGetFlags = staticmethod(get_flags)
        cudaStreamIsCapturing = staticmethod(is_capturing)
        cudaGetLastError = staticmethod(lambda: 0)

    return _Lib()


@pytest.fixture
def probe(monkeypatch):
    """Device-free stand-ins for cudart and torch's stream accessors."""
    lib = _fake_cudart(
        flags={_CURRENT: 1, _DEFAULT: 1, _LEGACY: 0},
        status={_CURRENT: 1, _DEFAULT: 0, _LEGACY: 0},
    )
    monkeypatch.setattr(cuda_capture, "_cudart", lambda: lib)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream(_CURRENT))
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: _FakeStream(_DEFAULT))
    return lib


@pytest.fixture(autouse=True)
def reset_open_captures():
    """The label table is a module global; keep tests from leaking into it."""
    yield
    with cuda_capture._open_capture_lock:
        cuda_capture._open_captures.clear()


class _Sentinel(RuntimeError):
    """Stands in for the AcceleratorError torch raises out of ``capture_end``."""


def _raise_sentinel(exc):
    def _fn(*_args, **_kwargs):
        raise exc

    return _fn


# --------------------------------------------------------------------------
# Closure condition 1 + 2 + 5: the dump reaches a main-thread capture failure
# --------------------------------------------------------------------------


def test_a_main_thread_mgc_failure_emits_the_dump(monkeypatch, capsys, probe) -> None:
    """The campaign's failure shape: make_graphed_callables, main thread."""
    exc = _Sentinel("CUDA error: cudaErrorStreamCaptureInvalidated")
    monkeypatch.setattr(
        cuda_capture._torch_graphs, "make_graphed_callables", _raise_sentinel(exc)
    )

    with pytest.raises(_Sentinel):
        graphed_callables(object(), (), label="tracker.update")

    out = capsys.readouterr().out
    assert "[capture-state]" in out
    assert "event=capture_failure" in out
    assert "at=tracker.update" in out
    assert f"thread={threading.current_thread().name}#" in out
    assert "error=_Sentinel" in out
    # The site's own label is still open when the dump is taken.
    assert "open_capture=tracker.update@" in out
    # Site, thread, current stream, stream flags, capture state -- all five.
    assert f"current=0x{_CURRENT:x}(non-blocking,capture=active)" in out
    assert f"legacy=0x{_LEGACY:x}(BLOCKING,capture=none)" in out


def test_the_dump_names_the_thread_that_failed(monkeypatch, capsys, probe) -> None:
    """A worker-thread capture reports that worker, not the main thread."""
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(_Sentinel("boom")),
    )

    def _run():
        with pytest.raises(_Sentinel):
            graphed_callables(object(), (), label="detector.whole")

    worker = threading.Thread(target=_run, name="StreamWorker-7")
    worker.start()
    worker.join()

    out = capsys.readouterr().out
    assert "thread=StreamWorker-7#" in out
    assert "open_capture=detector.whole@StreamWorker-7#" in out


# --------------------------------------------------------------------------
# Closure condition 4: the diagnostic never masks the original error
# --------------------------------------------------------------------------


def test_the_original_error_is_re_raised_unchanged(monkeypatch, probe) -> None:
    """The same exception object, with its own frames, reaches the caller.

    Asserted as *retention* -- object identity, no substituted cause or context,
    and the raising frame still present -- not as an exact traceback shape, which
    would break on any unrelated refactor of the wrapper.
    """
    sentinel = _Sentinel("cudaErrorStreamCaptureInvalidated")

    def _boom(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(cuda_capture._torch_graphs, "make_graphed_callables", _boom)

    with pytest.raises(_Sentinel) as excinfo:
        graphed_callables(object(), (), label="tracker.update")

    assert excinfo.value is sentinel
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None

    frames = []
    tb = excinfo.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "_boom" in frames


def test_a_broken_stdout_cannot_replace_the_error(monkeypatch, capsys, probe) -> None:
    """print() failing falls back to stderr and still re-raises the original."""
    sentinel = _Sentinel("boom")
    monkeypatch.setattr(
        cuda_capture._torch_graphs, "make_graphed_callables", _raise_sentinel(sentinel)
    )

    def _no_print(*_args, **_kwargs):
        raise OSError("stdout is closed")

    monkeypatch.setattr(builtins, "print", _no_print)
    try:
        with pytest.raises(_Sentinel) as excinfo:
            graphed_callables(object(), (), label="tracker.update")
    finally:
        monkeypatch.undo()

    assert excinfo.value is sentinel
    err = capsys.readouterr().err
    assert "[capture-state]" in err
    assert "at=tracker.update" in err


def test_an_interrupt_is_not_dumped_but_still_releases_the_label(
    monkeypatch, capsys, probe
) -> None:
    """``except Exception`` is deliberate: a Ctrl-C mid-capture is not evidence."""
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        graphed_callables(object(), (), label="tracker.update")

    assert "[capture-state]" not in capsys.readouterr().out
    assert cuda_capture._render_open_captures() == "none"


# --------------------------------------------------------------------------
# Closure condition 3: a failing diagnostic is secondary evidence, not silence
# --------------------------------------------------------------------------


def test_a_failing_diagnostic_is_recorded_not_swallowed(
    monkeypatch, capsys, probe
) -> None:
    sentinel = _Sentinel("boom")
    monkeypatch.setattr(
        cuda_capture._torch_graphs, "make_graphed_callables", _raise_sentinel(sentinel)
    )

    def _broken(*_args, **_kwargs):
        raise ValueError("the diagnostic is broken")

    monkeypatch.setattr(cuda_capture, "describe_capture_state", _broken)

    with pytest.raises(_Sentinel) as excinfo:
        graphed_callables(object(), (), label="tracker.update")

    assert excinfo.value is sentinel
    out = capsys.readouterr().out
    assert "diagnostic_failed=ValueError: the diagnostic is broken" in out
    assert "at=tracker.update" in out


def test_the_dump_still_lands_when_the_stream_probe_fails(
    monkeypatch, capsys, probe
) -> None:
    """A probe failure degrades one field; the identity half survives."""
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(_Sentinel("boom")),
    )

    def _no_stream():
        raise RuntimeError("no CUDA context")

    monkeypatch.setattr(torch.cuda, "current_stream", _no_stream)

    with pytest.raises(_Sentinel):
        graphed_callables(object(), (), label="gmc.pygraphed")

    out = capsys.readouterr().out
    assert "at=gmc.pygraphed" in out
    assert "probe_failed=RuntimeError" in out
    assert out.count("\n") == 1


# --------------------------------------------------------------------------
# Line grammar: one physical line, and no interference with the campaign harness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["tracker.update", "weird\nlabel\twith  space"])
def test_the_dump_is_always_one_physical_line(
    monkeypatch, capsys, probe, label
) -> None:
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(_Sentinel("a\nb")),
    )

    with pytest.raises(_Sentinel):
        graphed_callables(object(), (), label=label)

    assert capsys.readouterr().out.count("\n") == 1


def test_a_failing_diagnostic_line_is_also_one_line(monkeypatch, capsys, probe) -> None:
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(_Sentinel("boom")),
    )

    def _broken(*_args, **_kwargs):
        raise ValueError("multi\nline\nmessage")

    monkeypatch.setattr(cuda_capture, "describe_capture_state", _broken)

    with pytest.raises(_Sentinel):
        graphed_callables(object(), (), label="tracker.update")

    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert "diagnostic_failed=ValueError: multi line message" in out


def test_error_carries_the_type_only_so_the_harness_count_stays_clean(
    monkeypatch, capsys, probe
) -> None:
    """The incidence harness counts capture-error hits by substring-matching
    every log line.  Echoing the CUDA message here would let this diagnostic
    inflate that count with hits it manufactured itself.
    """
    message = "CUDA error: cudaErrorStreamCaptureInvalidated"
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(_Sentinel(message)),
    )

    with pytest.raises(_Sentinel):
        graphed_callables(object(), (), label="tracker.update")

    dump = capsys.readouterr().out
    assert "error=_Sentinel" in dump
    assert "cudaErrorStreamCaptureInvalidated" not in dump


# --------------------------------------------------------------------------
# Label lifecycle, including the concurrency the multi-stream runner creates
# --------------------------------------------------------------------------


def test_the_label_is_open_during_capture_and_released_after(
    monkeypatch, probe
) -> None:
    seen: list[str] = []

    def _capture(*_args, **_kwargs):
        seen.append(cuda_capture._render_open_captures())
        return "graphed"

    monkeypatch.setattr(cuda_capture._torch_graphs, "make_graphed_callables", _capture)

    assert graphed_callables(object(), (), label="mamba_head.per_shape") == "graphed"
    assert seen[0].startswith("mamba_head.per_shape@")
    assert cuda_capture._render_open_captures() == "none"


def test_the_label_is_released_on_the_failure_path_too(monkeypatch, probe) -> None:
    monkeypatch.setattr(
        cuda_capture._torch_graphs,
        "make_graphed_callables",
        _raise_sentinel(_Sentinel("boom")),
    )

    with pytest.raises(_Sentinel):
        graphed_callables(object(), (), label="tracker.update")

    assert cuda_capture._render_open_captures() == "none"


def test_a_nested_entrance_restores_the_outer_label(probe) -> None:
    with capture_diagnostics("outer"):
        with capture_diagnostics("inner"):
            rendered = cuda_capture._render_open_captures()
            assert "outer@" in rendered and "inner@" in rendered
        assert "inner@" not in cuda_capture._render_open_captures()
        assert "outer@" in cuda_capture._render_open_captures()
    assert cuda_capture._render_open_captures() == "none"


def test_another_threads_label_is_not_erased(probe) -> None:
    """multi_stream captures concurrently and never acquires its capture lock,
    so a thread must release only its own label.
    """
    entered = threading.Event()
    release = threading.Event()

    def _hold():
        with capture_diagnostics("worker.site"):
            entered.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=_hold, name="StreamWorker-3")
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with capture_diagnostics("main.site"):
            pass
        # The main thread's push/pop must not have cleared the worker's entry.
        assert "worker.site@StreamWorker-3#" in cuda_capture._render_open_captures()
    finally:
        release.set()
        worker.join(timeout=5)

    assert cuda_capture._render_open_captures() == "none"


# --------------------------------------------------------------------------
# The graph_capture refactor keeps its policy and gains the failure path
# --------------------------------------------------------------------------


class _GraphStub:
    """Records the capture_error_mode ``graph_capture`` forwards."""

    seen: list[str | None] = []

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode=None):
        _GraphStub.seen.append(capture_error_mode)
        self.capture_stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_graph_capture_still_forwards_the_capture_error_mode(
    monkeypatch, probe
) -> None:
    _GraphStub.seen.clear()
    monkeypatch.setattr(torch.cuda, "graph", _GraphStub)

    with graph_capture(object(), label="nms.main"):
        pass

    assert _GraphStub.seen == [cuda_capture.CAPTURE_ERROR_MODE]


def test_graph_capture_gained_the_failure_dump(monkeypatch, capsys, probe) -> None:
    """The six direct sites had no failure path at all before #374."""
    monkeypatch.setattr(torch.cuda, "graph", _GraphStub)
    sentinel = _Sentinel("boom")

    with pytest.raises(_Sentinel) as excinfo:
        with graph_capture(object(), label="nms.main"):
            raise sentinel

    assert excinfo.value is sentinel
    out = capsys.readouterr().out
    assert "event=capture_failure" in out
    assert "at=nms.main" in out


def test_the_open_time_print_stays_gated_on_the_debug_flag(
    monkeypatch, capsys, probe
) -> None:
    """The failure line is unconditional; the capture-open line must not be.

    Promoting the open-time print would put an observer on the capture path
    itself, which the campaign preregistration's observer ban forbids.
    """
    monkeypatch.setattr(torch.cuda, "graph", _GraphStub)

    monkeypatch.delenv("SACCADE_CAPTURE_DEBUG", raising=False)
    with graph_capture(object(), label="nms.main"):
        pass
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("SACCADE_CAPTURE_DEBUG", "1")
    with graph_capture(object(), label="nms.main"):
        pass
    out = capsys.readouterr().out
    assert "[capture-site] nms.main" in out
    assert "event=capture_open" in out


def test_the_mgc_open_time_print_is_gated_too(monkeypatch, capsys, probe) -> None:
    """Both entrances behave the same way: only the failure line is free of the flag."""
    monkeypatch.setattr(
        cuda_capture._torch_graphs, "make_graphed_callables", lambda *a, **k: "graphed"
    )

    monkeypatch.delenv("SACCADE_CAPTURE_DEBUG", raising=False)
    graphed_callables(object(), (), label="tracker.update")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("SACCADE_CAPTURE_DEBUG", "1")
    graphed_callables(object(), (), label="tracker.update")
    out = capsys.readouterr().out
    assert "[capture-site] tracker.update mode=thread_local" in out
    assert "event=capture_open" in out


# --------------------------------------------------------------------------
# The enumeration of capture entrances stays closed
# --------------------------------------------------------------------------

# Dotted paths that begin a capture without the diagnostic.
_RAW_CAPTURE_ENTRANCES = {
    ("torch", "cuda", "make_graphed_callables"),
    ("torch", "cuda", "graph"),
}
# A capture can also be begun straight off a graph object, bypassing both of the
# above.  There is no type inference here, so this matches on method name alone:
# a false positive is a loud, fixable failure, whereas the silent gap is the
# thing that cost #340 a reproduction's worth of evidence.
_RAW_CAPTURE_METHODS = {"capture_begin"}

_ENTRANCE_OWNER = Path("src/saccade/perception/eval/cuda_capture.py")

# The owner file is not exempt, it is *pinned*: exactly these raw calls, at these
# multiplicities, each in exactly this function.  A third one -- including a
# second copy inside a wrapper that already has one -- or one of these moved
# elsewhere in the file, fails.
#
# A multiset, not a set: comparing sets would silently collapse a duplicate
# entrance inside an existing wrapper, which is precisely the shape this pin
# exists to catch.
_OWNER_RAW_CALLS = Counter(
    {
        ("graph_capture", ("torch", "cuda", "graph")): 1,
        ("capture_context", ("torch", "cuda", "graph")): 1,
    }
)

# Native capture entrances.  Nothing under ``src/`` or ``include/`` begins a
# capture today, and the Python-side diagnostic cannot see one that does, so a
# new native capture has to be a deliberate decision rather than a silent gap.
_NATIVE_CAPTURE_ENTRANCES = ("cudaStreamBeginCapture", "cuStreamBeginCapture")
_NATIVE_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
)


def _alias_map(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Names in this module that resolve to ``torch`` or below.

    ``import torch as t``, ``from torch import cuda``, ``from torch.cuda import
    graph`` and friends all reach the same entry points as ``torch.cuda.graph``,
    so the scan has to follow the binding rather than the spelling.
    """
    aliases: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                if parts[0] != "torch":
                    continue
                if alias.asname:
                    aliases[alias.asname] = parts
                else:
                    # ``import torch.cuda`` also binds the root name ``torch``.
                    aliases[parts[0]] = (parts[0],)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module = tuple(node.module.split("."))
            if module[0] != "torch":
                continue
            for alias in node.names:
                aliases[alias.asname or alias.name] = module + (alias.name,)
    return aliases


def _dotted(
    node: ast.AST, aliases: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Resolve an attribute chain to its fully-qualified dotted path."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    head = aliases.get(node.id)
    if head is None:
        return None
    return head + tuple(reversed(parts))


def _function_scopes(tree: ast.Module):
    """Yield ``(enclosing function name or None, node)`` for every node."""

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield scope, child
                yield from walk(child, child.name)
            else:
                yield scope, child
                yield from walk(child, scope)

    yield from walk(tree, None)


def _raw_calls(path: Path):
    """Every raw capture entrance in one file, as ``(scope, dotted, lineno)``."""
    tree = ast.parse(path.read_text())
    aliases = _alias_map(tree)
    for scope, node in _function_scopes(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func, aliases)
        if dotted in _RAW_CAPTURE_ENTRANCES:
            yield scope, dotted, node.lineno
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _RAW_CAPTURE_METHODS
        ):
            yield scope, (node.func.attr,), node.lineno


def test_no_production_site_calls_the_raw_torch_capture_api() -> None:
    """Every production capture entrance goes through ``cuda_capture``.

    AST rather than text matching, because ``cuda_capture`` itself reads
    ``torch.cuda.graph``'s ``default_capture_stream`` attribute -- a bare
    attribute access that a textual scan would report as a capture site -- and
    because ``import torch as t`` reaches the same entry point under a different
    spelling.

    Scoped to ``src/saccade/``.  ``scripts/tools/capture_attribution/control.py``
    deliberately exercises the raw API as an experimental control, and
    ``scripts/tools/test_cufft_graph.py``, ``scripts/tools/test_py_gmc.py`` and
    ``scripts/benchmarks/debug_head_graph.py`` are diagnostics and benchmarks --
    none of them is production, and none is what a campaign runs.
    """
    offenders: list[str] = []
    for path in sorted((_REPO_ROOT / "src" / "saccade").rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT)
        if rel == _ENTRANCE_OWNER:
            continue
        for _scope, dotted, lineno in _raw_calls(path):
            offenders.append(f"{rel}:{lineno} calls {'.'.join(dotted)}")

    assert offenders == [], (
        "capture entrances must go through saccade.perception.eval.cuda_capture "
        "so a failure there is observable: " + "; ".join(offenders)
    )


def test_the_entrance_owner_holds_exactly_the_expected_raw_calls() -> None:
    """``cuda_capture.py`` is pinned, not exempt.

    It is the one file allowed to touch the raw API, so exempting it would let a
    third unwrapped entrance be added in the very module whose job is to make
    entrances observable.  Each permitted call is pinned to the wrapper that owns
    it *and to its multiplicity*, so moving one out of its wrapper fails, and so
    does adding a second copy inside a wrapper that already has one.
    """
    calls = list(_raw_calls(_REPO_ROOT / _ENTRANCE_OWNER))
    found = Counter((scope, dotted) for scope, dotted, _ in calls)
    sites = ", ".join(
        f"{scope}:{lineno} {'.'.join(dotted)}" for scope, dotted, lineno in calls
    )
    assert found == _OWNER_RAW_CALLS, (
        "the raw capture calls in cuda_capture.py drifted; each must stay inside "
        f"its own wrapper, exactly once. expected {sorted(_OWNER_RAW_CALLS.items())}, "
        f"found {sorted(found.items())} at [{sites}]"
    )


def test_the_scan_follows_aliased_imports_and_direct_capture_begin() -> None:
    """The guard has to see the spellings a future site would plausibly use.

    Without this, ``import torch as t; t.cuda.graph(...)`` and a bare
    ``g.capture_begin()`` would both pass a scan that only knows the literal
    ``torch.cuda.graph`` chain -- which is how a coverage guard quietly stops
    guarding anything.
    """
    source = (
        "import torch as t\n"
        "from torch import cuda\n"
        "from torch.cuda import make_graphed_callables as mgc\n"
        "def a():\n"
        "    return t.cuda.graph(None)\n"
        "def b():\n"
        "    return cuda.graph(None)\n"
        "def c():\n"
        "    return mgc(None, ())\n"
        "def d():\n"
        "    g.capture_begin()\n"
        "def e():\n"
        "    return t.cuda.graph(None), t.cuda.graph(None)\n"
    )
    probe = _REPO_ROOT / "src" / "saccade" / "_alias_probe_for_tests.py"
    probe.write_text(source)
    try:
        found = Counter((scope, dotted) for scope, dotted, _ in _raw_calls(probe))
    finally:
        probe.unlink()

    # ``e`` is counted twice: the scan is a multiset, so a duplicated entrance
    # inside one function cannot hide behind an identical sibling.
    assert found == Counter(
        {
            ("a", ("torch", "cuda", "graph")): 1,
            ("b", ("torch", "cuda", "graph")): 1,
            ("c", ("torch", "cuda", "make_graphed_callables")): 1,
            ("d", ("capture_begin",)): 1,
            ("e", ("torch", "cuda", "graph")): 2,
        }
    )


def test_no_native_source_begins_a_capture() -> None:
    """A native capture would be invisible to the Python-side diagnostic.

    Everything the failure-time dump knows comes from Python wrappers, so a
    ``cudaStreamBeginCapture`` in ``src/`` or ``include/`` would reopen exactly
    the coverage gap #374 closed -- silently.  There are none today; if one is
    added deliberately, this test is the place to record how it is made
    observable.
    """
    offenders: list[str] = []
    for root in ("src", "include"):
        for path in sorted((_REPO_ROOT / root).rglob("*")):
            if not path.is_file() or path.suffix not in _NATIVE_SUFFIXES:
                continue
            text = path.read_text(errors="replace")
            for entrance in _NATIVE_CAPTURE_ENTRANCES:
                if entrance in text:
                    rel = path.relative_to(_REPO_ROOT)
                    offenders.append(f"{rel} references {entrance}")

    assert offenders == [], (
        "a native capture entrance is not covered by the Python failure-time "
        "diagnostic: " + "; ".join(offenders)
    )


def test_the_diagnostic_helpers_are_importable_without_cuda() -> None:
    """The wiring must not force a CUDA import at module scope on any site."""
    assert callable(cuda_capture.emit_capture_failure_state)
    assert "saccade.perception.eval.cuda_capture" in sys.modules
