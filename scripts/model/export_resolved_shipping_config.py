#!/usr/bin/env python3
"""Export the headline preset's resolved shipping config (one flat JSON).

Issue #465 Phase B PR-3 (U2a; shared_boundary B2, see
docs/reference/native_runtime_shipping_boundary.md §5 B2 and §6). Developer
tooling only: it writes the file a Python-free runtime will read instead of the
four-layer preset resolution. It does not change the eval harness, the preset,
any native code or any threshold, and it runs no frames.

Values come from the oracle, not from a second copy of the preset:

* resolution runs ``mot17.py``'s own statements (``_load_config_defaults``,
  the ``__main__`` parser/argv block, ``configure_runtime_env``, the
  ``eval_kwargs`` filter and the Mamba builder call) and then
  ``parse_eval_config`` exactly as ``run_eval`` calls it;
* ``native_params`` executes the oracle statements that construct and
  configure native objects (``run_eval``'s ``PerceptionPipelineConfig`` block,
  ``EvalPipeline.__init__``'s tracker/GMC setup) against recording stand-ins
  for the native classes. Calls go through the real Python ``GPUByteTracker``
  wrapper, so the recorded values are what the extension receives; argument
  names and unpassed defaults are read from the pybind binding source;
* ``native_env`` is every ``SACCADE_*`` ``getenv`` in the native sources with
  the value it resolves to under the oracle environment (the binding default,
  since the oracle never exports them);
* ``host_params`` holds the per-frame stage order, the gate of every
  conditional step (the oracle's own gate expression, evaluated), the detector
  build, host constants the executed statements produce, every ``SACCADE_*``
  the host modules name, and every ``cfg``/``kwargs`` read in
  ``evaluator.py``/``pipeline.py``/``stages.py`` with the value that read
  returns.

Coverage is fail-closed: an oracle statement that touches a native object but
is neither executed nor excused with a reason, a ``cfg`` read of a field
``EvalConfig`` lacks, a gate expression no longer in the oracle, a span anchor
that moved, or a native ``getenv`` with no recoverable default stops the
export. A new ``cfg`` read or setter call changes the export, so
``--check`` (and the contract test) fail until the file is regenerated and the
diff reviewed.

Usage:
    .venv/bin/python scripts/model/export_resolved_shipping_config.py          # write
    .venv/bin/python scripts/model/export_resolved_shipping_config.py --check  # compare
"""
# status: diagnostic

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import io
import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src", REPO / "scripts" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "saccade.resolved_shipping_config/v1"
PRESET = "configs/presets/mamba_whole_graph.yaml"
OUTPUT = "configs/shipping/mamba_whole_graph.resolved.json"
# Boundary §2: the oracle is `mot17.py --preset mamba_whole_graph --detector
# SDP --double-buffer` with every other flag at its default.
ORACLE_ARGV = ("--preset", "mamba_whole_graph", "--detector", "SDP", "--double-buffer")
# Sequence selection is input, not config; a concrete name keeps
# parse_eval_config from listing a dataset directory.
PLACEHOLDER_SEQUENCE = "MOT17-02-SDP"

MOT17 = "scripts/eval/mot17.py"
MOT17_ARGS = "scripts/eval/mot17_args.py"
EVALUATOR = "src/saccade/perception/eval/evaluator.py"
PIPELINE = "src/saccade/perception/eval/pipeline.py"
STAGES = "src/saccade/perception/eval/stages.py"
TRACKER_WRAPPER = "src/saccade/perception/tracking/tracker_gpu.py"
BINDING_SOURCES = ("src/tracking/tracker_gpu_python.cpp",)
NATIVE_SOURCE_ROOTS = ("src", "include")
NATIVE_SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp")


def native_sources() -> list[str]:
    return sorted(
        p.relative_to(REPO).as_posix()
        for root in NATIVE_SOURCE_ROOTS
        for p in (REPO / root).rglob("*")
        if p.suffix in NATIVE_SOURCE_SUFFIXES and p.is_file()
    )


# ---------------------------------------------------------------------------
# Per-sequence placeholders
# ---------------------------------------------------------------------------


class PerSequence(str):
    """A value the oracle reads from the sequence, not from config."""

    def json(self) -> dict[str, str]:
        return {"per_sequence": str(self)}


SEQ_WIDTH = PerSequence("seqinfo.ini:Sequence.imWidth")
SEQ_HEIGHT = PerSequence("seqinfo.ini:Sequence.imHeight")


# ---------------------------------------------------------------------------
# Recording stand-ins for native classes
# ---------------------------------------------------------------------------


@dataclass
class NativeCall:
    cls: str
    instance: int
    method: str  # "__init__", a method name, or ".attr" for an attribute set
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class NativeLog:
    calls: list[NativeCall] = field(default_factory=list)
    _next: int = 0

    def new_id(self) -> int:
        self._next += 1
        return self._next

    def native_class(self, name: str, properties: dict[str, Any] | None = None) -> type:
        """A stand-in that records construction, method calls and attribute sets.

        ``properties`` maps a read-only native property to a function of the
        constructor arguments (the stand-in cannot run the native getter).
        """
        log = self
        props = dict(properties or {})

        class _Native:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                object.__setattr__(self, "_rec_id", log.new_id())
                object.__setattr__(self, "_rec_ctor", (args, kwargs))
                log.calls.append(
                    NativeCall(name, self._rec_id, "__init__", args, kwargs)
                )

            def __getattr__(self, attr: str) -> Any:
                if attr.startswith("__") or attr.startswith("_rec_"):
                    raise AttributeError(attr)
                if attr in props:
                    args, kwargs = self._rec_ctor
                    return props[attr](*args, **kwargs)

                def _call(*args: Any, **kwargs: Any) -> None:
                    log.calls.append(NativeCall(name, self._rec_id, attr, args, kwargs))

                return _call

            def __setattr__(self, attr: str, value: Any) -> None:
                log.calls.append(
                    NativeCall(name, self._rec_id, "." + attr, (value,), {})
                )
                object.__setattr__(self, attr, value)

        _Native.__name__ = _Native.__qualname__ = name
        return _Native

    def of(self, cls: str) -> list[NativeCall]:
        return [c for c in self.calls if c.cls == cls]


class _Inert:
    """Attribute sink for host objects whose methods carry no config."""

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return lambda *a, **k: None


# ---------------------------------------------------------------------------
# Oracle source access
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _source(rel: str) -> str:
    return (REPO / rel).read_text()


@functools.lru_cache(maxsize=None)
def _tree(rel: str) -> ast.Module:
    return ast.parse(_source(rel), filename=rel)


def function_node(rel: str, qualname: str) -> Any:
    """``qualname`` of a function/method, or ``"__main__"`` for the script block."""
    node: Any = _tree(rel)
    if qualname == "__main__":
        mains = [
            n
            for n in node.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "__name__"
        ]
        if len(mains) != 1:
            raise SystemExit(f"oracle drift: {rel} has no single __main__ block")
        return mains[0]
    for part in qualname.split("."):
        matches = [
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part
        ]
        if len(matches) != 1:
            raise SystemExit(f"oracle drift: {rel}:{qualname} not found exactly once")
        node = matches[0]
    return node


def _head(rel: str, stmt: ast.stmt) -> str:
    seg = ast.get_source_segment(_source(rel), stmt) or ""
    return seg.splitlines()[0].strip() if seg else ""


# An anchor is the start of a statement's first source line; ``(text, n)``
# picks the n-th (0-based) statement with that start when it is not unique.
Anchor = str | tuple[str, int]


def _index(rel: str, body: list[ast.stmt], anchor: Anchor) -> int:
    text, nth = (anchor, None) if isinstance(anchor, str) else anchor
    hits = [i for i, s in enumerate(body) if _head(rel, s).startswith(text)]
    if nth is None:
        if len(hits) != 1:
            raise SystemExit(
                f"oracle drift: anchor {text!r} matches {len(hits)} statements in {rel}"
            )
        return hits[0]
    if nth >= len(hits):
        raise SystemExit(f"oracle drift: anchor {anchor!r} not found in {rel}")
    return hits[nth]


@dataclass(frozen=True)
class Span:
    """Contiguous oracle statements, executed verbatim minus named exclusions."""

    rel: str
    qualname: str
    first: Anchor
    last: Anchor
    exclude: tuple[tuple[Anchor, str], ...] = ()  # (anchor, reason)

    def statements(self, body: list[ast.stmt] | None = None) -> list[ast.stmt]:
        if body is None:
            body = function_node(self.rel, self.qualname).body
        lo, hi = _index(self.rel, body, self.first), _index(self.rel, body, self.last)
        if hi < lo:
            raise SystemExit(
                f"oracle drift: span {self.first!r}..{self.last!r} reversed"
            )
        skip = {_index(self.rel, body, a) for a, _ in self.exclude}
        if not skip <= set(range(lo, hi + 1)):
            raise SystemExit(f"oracle drift: exclusion outside span {self.first!r}")
        return [body[i] for i in range(lo, hi + 1) if i not in skip]

    def line_ranges(self) -> list[tuple[int, int]]:
        return [(s.lineno, s.end_lineno or s.lineno) for s in self.statements()]


def run_statements(rel: str, stmts: list[ast.stmt], ns: dict[str, Any]) -> None:
    module = ast.Module(body=list(stmts), type_ignores=[])
    exec(compile(module, str(REPO / rel), "exec"), ns)  # noqa: S102


# ---------------------------------------------------------------------------
# Spans: which oracle statements are executed
# ---------------------------------------------------------------------------

# mot17.py `__main__`: parser + argv parse, runtime env, the eval_kwargs filter
# and the detector builder block, up to (not including) the run_eval dispatch.
MOT17_MAIN = Span(
    MOT17,
    "__main__",
    "parser = build_parser()",
    'eval_kwargs["research_bridge_fidelity_detector_provenance"] = {',
    exclude=(
        (
            'if os.environ.get("SACCADE_NV12_BUFFER") == "1":',
            "NV12 relaunch hatch; the variable is absent from the oracle environment",
        ),
        ("if args.detector and not args.sequences:", "sequence selection is input"),
        ("claim_or_join_run(", "run-manifest side effect (eval_research)"),
        (
            'if getattr(args, "processes", 0) > 0 and args.sequences:',
            "multi-process launcher; processes=0 in the oracle",
        ),
    ),
)

# run_eval: config parse, then everything from ReID/FPN mode selection to the
# ONMS knobs (detect_fn choice, DetectionContract, PerceptionPipelineConfig).
RUN_EVAL_CFG = Span(
    EVALUATOR,
    "run_eval",
    "from .config import parse_eval_config",
    "cfg = parse_eval_config(",
)
RUN_EVAL_PROFILE = Span(
    EVALUATOR,
    "run_eval",
    "profile_stages = cfg.core.profile_stages",
    "profile_stages = cfg.core.profile_stages",
)
RUN_EVAL_NATIVE = Span(
    EVALUATOR,
    "run_eval",
    "_fpn_reid_mode = reid_model in",
    "onms_min_track_score = cfg.core.high_thresh",
)

# EvalPipeline.__init__: tracker creation, GMC, homography, lifecycle/relinker
# construction and every tracker setter; then the per-sequence setter block.
PIPELINE_SETUP = Span(
    PIPELINE,
    "EvalPipeline.__init__",
    "wb = None",
    ("if (", 2),  # live evfifo handover gate, the last statement before seqinfo
)
PIPELINE_SEQ = Span(
    PIPELINE,
    "EvalPipeline.__init__",
    "seq_reid_interval = cfg.reid_interval",
    "active_tracker_thresholds = (",
    exclude=(
        ("_gmc_frame_buf = None", "GMC frame buffer allocation; carries no parameter"),
        ("if _use_direct_gmc:", "GMC frame buffer allocation; carries no parameter"),
    ),
)
PIPELINE_CAPS = (
    Span(
        PIPELINE,
        "EvalPipeline.__init__",
        "_TRACK_RESULT_CAP = int(",
        "_TRACK_RESULT_CAP = int(",
    ),
    Span(
        PIPELINE, "EvalPipeline.__init__", "_NMS_FIXED_N = int(", "_NMS_FIXED_N = int("
    ),
)


# Coverage of the native-configuring functions: every top-level statement that
# mentions a native object is executed by a span above or listed here.
NATIVE_TOUCH_FUNCS = ((EVALUATOR, "run_eval"), (PIPELINE, "EvalPipeline.__init__"))
NATIVE_TOUCH_MARKERS = (
    "detector.tracker",
    "reset_tracker(",
    "GPUByteTracker(",
    "native_cfg",
    "PerceptionPipeline",
    "perception_pipeline.",
    "gmc_estimator",
    "_LIFECYCLE_CLS",
    "Relinker",
)
NOT_EXECUTED: dict[tuple[str, str, str], str] = {
    (EVALUATOR, "run_eval", "for seq in cfg.seqs:"): (
        "per-sequence loop: constructs EvalPipeline (whose native setup the "
        "PIPELINE_* spans execute) and reads relink-debug/D0 diagnostics"
    ),
    (
        PIPELINE,
        "EvalPipeline.__init__",
        "self.native_cfg = native_cfg",
    ): "stores a reference",
    (
        PIPELINE,
        "EvalPipeline.__init__",
        "self.gmc_estimator = gmc_estimator",
    ): "stores a reference",
    (
        PIPELINE,
        "EvalPipeline.__init__",
        "tracker_result_buffers = detector.tracker.allocate_result_buffers(",
    ): ("allocates tracker output buffers; their capacity is _TRACK_RESULT_CAP"),
    (
        PIPELINE,
        "EvalPipeline.__init__",
        'if cfg.kwargs.get("use_tracker_graph", False) and not cfg.relink_enabled:',
    ): (
        "captures the tracker update into a CUDA graph; gate is steps['track.graphed_update']"
    ),
    (
        PIPELINE,
        "EvalPipeline.__init__",
        "def _bg_relink_write(",
    ): "background relink-write closure (ReID relink path)",
}


def native_touch_gaps(spans: tuple[Span, ...] | None = None) -> list[str]:
    """Native-touching oracle statements neither executed nor explained."""
    spans = ALL_SPANS if spans is None else spans
    gaps: list[str] = []
    for rel, qualname in NATIVE_TOUCH_FUNCS:
        body = function_node(rel, qualname).body
        executed = {
            id(st)
            for sp in spans
            if (sp.rel, sp.qualname) == (rel, qualname)
            for st in sp.statements(body)
        }
        not_executed = {
            _index(rel, body, anchor)
            for (r, q, anchor) in NOT_EXECUTED
            if (r, q) == (rel, qualname)
        }
        span_excluded = {
            _index(rel, body, a)
            for sp in spans
            if (sp.rel, sp.qualname) == (rel, qualname)
            for a, _ in sp.exclude
        }
        for i, st in enumerate(body):
            seg = ast.get_source_segment(_source(rel), st) or ""
            if id(st) in executed and i in not_executed:
                gaps.append(f"{rel}:{st.lineno} is executed but listed in NOT_EXECUTED")
            elif (
                id(st) not in executed
                and i not in not_executed | span_excluded
                and any(m in seg for m in NATIVE_TOUCH_MARKERS)
            ):
                gaps.append(f"{rel}:{st.lineno} {_head(rel, st)}")
    return gaps


# ---------------------------------------------------------------------------
# Oracle environment
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def oracle_environment() -> Iterator[dict[str, str]]:
    """Process env as the oracle sees it: no inherited ``SACCADE_*`` hatch.

    ``configure_runtime_env`` later writes the variables the CLI owns into it.
    """
    scrubbed = {k: v for k, v in os.environ.items() if not k.startswith("SACCADE_")}
    with mock.patch.dict(os.environ, scrubbed, clear=True):
        yield os.environ  # type: ignore[misc]


@contextlib.contextmanager
def patched_modules(**modules: types.ModuleType) -> Iterator[None]:
    with mock.patch.dict(
        sys.modules, {k.replace("__", "."): v for k, v in modules.items()}
    ):
        yield


def _fake_module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Resolution: argv -> args -> runtime env -> eval_kwargs -> EvalConfig
# ---------------------------------------------------------------------------


@dataclass
class Oracle:
    args: argparse.Namespace
    eval_kwargs: dict[str, Any]
    runtime_env: dict[str, str]
    detector_build: dict[str, Any]
    detector_calls: list[NativeCall]
    cfg: Any
    run_eval_ns: dict[str, Any]
    log: NativeLog
    pipeline_ns: dict[str, Any] = field(default_factory=dict)


def _mot17_namespace() -> dict[str, Any]:
    """Names the mot17 ``__main__`` block uses, bound to the real objects.

    ``_load_config_defaults`` is mot17's own function, compiled from its source
    (importing mot17.py would load TensorRT and the CUDA runner).
    """
    import mot17_args

    ns: dict[str, Any] = {
        "__name__": "mot17_oracle",
        "argparse": argparse,
        "os": os,
        "sys": sys,
        "Path": Path,
        "yaml": __import__("yaml"),
        "project_root": REPO,
        "build_parser": mot17_args.build_parser,
        "configure_runtime_env": mot17_args.configure_runtime_env,
    }
    loader = function_node(MOT17, "_load_config_defaults")
    run_statements(MOT17, [loader], ns)
    return ns


def resolve_oracle(log: NativeLog) -> Oracle:
    ns = _mot17_namespace()
    builder_log: list[dict[str, Any]] = []
    head = log.native_class("MambaHead")

    class _BuiltDetector:
        def __init__(self) -> None:
            self.mamba_head = head()

    def build_mamba_gated_detector(**kwargs: Any) -> Any:
        builder_log.append(kwargs)
        return _BuiltDetector()

    compile_calls: list[bool] = []
    detector_module = _fake_module(
        "saccade.perception.temporal_yolo.mamba_gated_detector",
        build_mamba_gated_detector=build_mamba_gated_detector,
        set_postprocess_compile=compile_calls.append,
    )
    argv = [MOT17, *ORACLE_ARGV]
    with (
        mock.patch.object(sys, "argv", argv),
        patched_modules(
            saccade__perception__temporal_yolo__mamba_gated_detector=detector_module
        ),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        main = function_node(MOT17, "__main__")
        run_statements(MOT17, MOT17_MAIN.statements(main.body), ns)
    if len(builder_log) != 1:
        raise SystemExit("oracle drift: mot17 did not build exactly one Mamba detector")
    eval_kwargs = dict(ns["eval_kwargs"])
    eval_kwargs["sequences"] = PLACEHOLDER_SEQUENCE
    runtime_env = {k: v for k, v in os.environ.items() if k.startswith("SACCADE_")}
    build = dict(builder_log[0])
    build["postprocess_compile"] = compile_calls[-1] if compile_calls else None
    cfg, run_ns = _parse_eval_config(eval_kwargs)
    return Oracle(
        args=ns["args"],
        eval_kwargs=eval_kwargs,
        runtime_env=runtime_env,
        detector_build=build,
        detector_calls=log.of("MambaHead"),
        cfg=cfg,
        run_eval_ns=run_ns,
        log=log,
    )


def _run_eval_namespace(eval_kwargs: dict[str, Any]) -> dict[str, Any]:
    """run_eval's module globals plus its bound parameters."""
    import inspect

    from saccade.perception.eval import evaluator

    sig = inspect.signature(evaluator.run_eval)
    bound = sig.bind(
        **{k: v for k, v in eval_kwargs.items() if k != "detector"},
        detector=eval_kwargs.get("detector"),
    )
    bound.apply_defaults()
    ns = dict(vars(evaluator))
    ns.update(bound.arguments)
    ns["kwargs"] = dict(bound.arguments.get("kwargs", {}))
    return ns


def _parse_eval_config(eval_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    ns = _run_eval_namespace(eval_kwargs)
    ns["__package__"] = "saccade.perception.eval"
    run_statements(EVALUATOR, RUN_EVAL_CFG.statements(), ns)
    return ns["cfg"], ns


# ---------------------------------------------------------------------------
# Native capture: run the oracle's construction/setter statements
# ---------------------------------------------------------------------------

# Native classes whose configuration is part of the shipping runtime
# (boundary §5 B2). Anything else the spans construct must be on the
# non-shipping list below, and its gate must be off.
SHIPPING_NATIVE = (
    "GPUByteTracker",
    "GMC",
    "PerceptionPipelineConfig",
    "PerceptionPipeline",
)
NON_SHIPPING_NATIVE: dict[str, str] = {
    "TrackletLifecycleMerger": "constructed unconditionally; lifecycle merge gate is the `enabled` argument",
    "DynamicReIDController": "constructed by the tracker wrapper; ReID is off (reid_mode)",
}

# Read-only native properties the tracker wrapper consults. Mirrors
# tracker_gpu.cu (`max_assoc_(std::max(1, max_assoc))`, `max_objects_`).
_TRACKER_PROPERTIES = {
    "max_objects": lambda max_objects=2048, *_a, **_k: int(max_objects),
    "max_assoc": lambda _mo=2048, _ed=768, max_assoc=1024, *_a, **_k: max(
        1, int(max_assoc)
    ),
}


class _NotCudaTracker:
    """``isinstance`` target for the wrapper's real-extension probe: never matches."""


def _fake_extension(log: NativeLog) -> types.ModuleType:
    mod = types.ModuleType("saccade_tracking_ext")
    classes: dict[str, type] = {"GPUByteTracker": _NotCudaTracker}

    def __getattr__(name: str) -> Any:  # PEP 562
        if name.startswith("__"):
            raise AttributeError(name)
        if name not in classes:
            classes[name] = log.native_class(name)
        return classes[name]

    mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
    return mod


class _OracleDetector:
    """The built Mamba detector, seen by EvalPipeline.

    Methods the real ``MambaGatedDetector`` defines are recorded; anything it
    lacks raises AttributeError, so ``hasattr`` gates resolve as in the oracle.
    ``reset_tracker`` runs the real method body.
    """

    def __init__(self, log: NativeLog, build: dict[str, Any]) -> None:
        from saccade.perception.temporal_yolo.mamba_gated_detector import (
            MambaGatedDetector,
        )

        object.__setattr__(self, "_cls", MambaGatedDetector)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_stream_state", _Inert())
        object.__setattr__(self, "use_whole_graph", bool(build["use_whole_graph"]))

    def reset_tracker(self) -> None:
        self._cls.reset_tracker(self)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") or not hasattr(self._cls, attr):
            raise AttributeError(attr)

        def _call(*args: Any, **kwargs: Any) -> None:
            self._log.calls.append(NativeCall("Detector", 0, attr, args, kwargs))

        return _call


def capture_native(oracle: Oracle, *, seq_fps: int = 30) -> None:
    """Execute the native-touching oracle spans; calls land in ``oracle.log``."""
    from saccade.perception.eval import pipeline
    from saccade.perception.tracking import tracker_gpu

    log = oracle.log
    tracker_cls = log.native_class("GPUByteTracker", _TRACKER_PROPERTIES)
    ext = _fake_extension(log)

    run_ns = oracle.run_eval_ns
    run_ns.update(
        PerceptionPipelineConfig=log.native_class("PerceptionPipelineConfig"),
        PerceptionPipeline=log.native_class("PerceptionPipeline"),
        ZeroCopyCropper=log.native_class("ZeroCopyCropper"),
        TRTFeatureExtractor=log.native_class("TRTFeatureExtractor"),
    )
    with (
        mock.patch.object(tracker_gpu, "CppGPUByteTracker", tracker_cls),
        mock.patch.object(
            tracker_gpu,
            "CppDynamicReIDController",
            log.native_class("DynamicReIDController"),
        ),
        patched_modules(saccade_tracking_ext=ext),
    ):
        run_statements(EVALUATOR, RUN_EVAL_PROFILE.statements(), run_ns)
        run_statements(EVALUATOR, RUN_EVAL_NATIVE.statements(), run_ns)

        ns = dict(vars(pipeline))
        ns.update(
            self=types.SimpleNamespace(),
            cfg=oracle.cfg,
            seq=PLACEHOLDER_SEQUENCE,
            profile_stages=run_ns["profile_stages"],
            contract=run_ns["contract"],
            detector=_OracleDetector(log, oracle.detector_build),
            cropper=run_ns["cropper"],
            extractor=run_ns["extractor"],
            native_cfg=run_ns["native_cfg"],
            perception_pipeline=run_ns["perception_pipeline"],
            max_frames=None,
            _LIFECYCLE_CLS=log.native_class("TrackletLifecycleMerger"),
            SemanticRelinker=log.native_class("SemanticRelinker"),
            PythonSemanticRelinker=log.native_class("PythonSemanticRelinker"),
            w_orig=SEQ_WIDTH,
            h_orig=SEQ_HEIGHT,
            seq_fps=seq_fps,
        )
        for span in (PIPELINE_SETUP, PIPELINE_SEQ, *PIPELINE_CAPS):
            run_statements(PIPELINE, span.statements(), ns)
    oracle.pipeline_ns = ns


# ---------------------------------------------------------------------------
# pybind binding source: argument names and defaults
# ---------------------------------------------------------------------------


def _strip_cpp_comments(src: str) -> str:
    out, i, n = [], 0, len(src)
    while i < n:
        ch = src[i]
        if ch == '"':
            j = i + 1
            while src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i == -1 else i
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 2
        elif ch == "'":  # char literal
            j = src.index("'", i + 2 if src[i + 1] == "\\" else i + 1)
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _balanced(text: str, start: int) -> int:
    """Index just past the parenthesis group opening at ``text[start]``."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced binding source")


def _split_top(text: str) -> list[str]:
    parts, depth, cur, i = [], 0, [], 0
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            cur.append(text[i : j + 1])
            i = j + 1
            continue
        if ch in "([{<" and not (ch == "<" and depth == 0 and text[i - 1 : i] == " "):
            depth += 1
        elif ch in ")]}>" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _cpp_literal(text: str) -> Any:
    t = text.strip()
    if t in {"true", "false"}:
        return t == "true"
    m = re.fullmatch(r"(-?\d+\.\d*(?:e-?\d+)?)f?", t) or re.fullmatch(r"(-?\d+)", t)
    if m:
        return float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    raise SystemExit(f"binding default not a literal: {text!r}")


@dataclass(frozen=True)
class Param:
    name: str
    default: str | None = None  # C++ source text; parsed only when used
    has_default: bool = False


@functools.lru_cache(maxsize=None)
def binding_signatures(cls_name: str) -> dict[str, tuple[tuple[Param, ...], ...]]:
    """``{method: (overload params, ...)}`` for one ``py::class_``; ``__init__``
    for constructors and ``.field`` entries for ``def_readwrite`` fields."""
    for rel in BINDING_SOURCES:
        src = _strip_cpp_comments(_source(rel))
        m = re.search(r'py::class_<[^>]*>\(m, "%s"\)' % re.escape(cls_name), src)
        if m:
            break
    else:
        raise SystemExit(f"no pybind class {cls_name!r} in {BINDING_SOURCES}")
    nxt = src.find("py::class_<", m.end())
    section = src[m.end() : nxt if nxt != -1 else len(src)]
    out: dict[str, list[tuple[Param, ...]]] = {}
    for d in re.finditer(r"\.def(_readwrite)?\(", section):
        body = section[d.end() - 1 : _balanced(section, d.end() - 1)][1:-1]
        parts = _split_top(body)
        if d.group(1):
            out.setdefault("." + parts[0].strip('"'), []).append(())
            continue
        head = parts[0]
        name = "__init__" if head.startswith("py::init") else head.strip('"')
        args = [p for p in parts[1:] if p.startswith("py::arg(")]
        params: list[Param] = []
        if args:
            for a in args:
                am = re.fullmatch(r'py::arg\("(\w+)"\)(?:\s*=\s*(.+))?', a, re.S)
                if not am:
                    raise SystemExit(f"unparsed binding arg {a!r} in {cls_name}.{name}")
                if am.group(2) is None:
                    params.append(Param(am.group(1)))
                else:
                    params.append(Param(am.group(1), am.group(2).strip(), True))
        else:
            lam = re.search(r"\[\]\s*\(([^)]*)\)", body)
            if lam:
                names = [
                    re.split(r"[\s&*]+", p.strip())[-1] for p in lam.group(1).split(",")
                ]
                params = [Param(n) for n in names[1:] if n]
        out.setdefault(name, []).append(tuple(params))
    return {k: tuple(v) for k, v in out.items()}


def name_call(cls_name: str, call: NativeCall) -> dict[str, Any]:
    """Bind a recorded call to the binding's names, filling unpassed defaults."""
    if call.method.startswith("."):
        fields = binding_signatures(cls_name)
        if call.method not in fields:
            raise SystemExit(f"{cls_name} has no bound field {call.method[1:]!r}")
        return {call.method[1:]: call.args[0]}
    overloads = binding_signatures(cls_name).get(call.method, ())
    fits = [
        ps
        for ps in overloads
        if len(call.args) <= len(ps)
        and set(call.kwargs) <= {p.name for p in ps}
        and all(
            p.has_default or i < len(call.args) or p.name in call.kwargs
            for i, p in enumerate(ps)
        )
    ]
    if len(fits) != 1:
        raise SystemExit(
            f"{cls_name}.{call.method}: {len(fits)} binding overloads fit "
            f"{len(call.args)} positional + {sorted(call.kwargs)}"
        )
    named: dict[str, Any] = {}
    for i, p in enumerate(fits[0]):
        if i < len(call.args):
            named[p.name] = call.args[i]
        elif p.name in call.kwargs:
            named[p.name] = call.kwargs[p.name]
        else:
            named[p.name] = _cpp_literal(p.default or "")
    return named


# ---------------------------------------------------------------------------
# Environment hatches
# ---------------------------------------------------------------------------

_NATIVE_ENV_HELPER = re.compile(
    r'(env_flag_enabled|env_float_value|env_diagnostic_on)\("(SACCADE_[A-Z0-9_]+)"(?:,\s*([^)]+))?\)'
)
_NATIVE_GETENV = re.compile(r'getenv\("(SACCADE_[A-Z0-9_]+)"\)')
_NATIVE_GETENV_DEFAULT = re.compile(r"\?\s*std::str\w+\([^;]*\)\s*:\s*([-0-9.]+f?)\s*;")
# Raw getenv sites without a ternary default: the effect when the variable is
# unset, which is the oracle's case.
NATIVE_ENV_UNSET_EFFECT: dict[str, str] = {
    "SACCADE_KALMAN_ADAPT_MODE": "no override; set_params kalman_adapt_mode stands",
    "SACCADE_ASSOC_DUMP": "no association dump file",
    "SACCADE_HO_DEBUG_LEVEL": "handover debug logging off",
    "SACCADE_ASSOC_STATS": "association statistics diagnostic off",
}


def native_env() -> dict[str, Any]:
    """Every native ``SACCADE_*`` read with the value it resolves to when unset."""
    out: dict[str, Any] = {}

    def put(name: str, value: Any) -> None:
        if name in out and out[name] != value:
            raise SystemExit(f"native env {name} has conflicting defaults")
        out[name] = value

    for rel in native_sources():
        src = _source(rel)
        if "SACCADE_" not in src:
            continue
        for m in _NATIVE_ENV_HELPER.finditer(src):
            kind, name, default = m.groups()
            if kind == "env_diagnostic_on":
                put(name, {"unset_effect": NATIVE_ENV_UNSET_EFFECT[name]})
            else:
                put(name, _cpp_literal(default))
        for m in _NATIVE_GETENV.finditer(src):
            name = m.group(1)
            window = src[m.end() : m.end() + 200]
            d = _NATIVE_GETENV_DEFAULT.search(window.split("}", 1)[0])
            if d:
                put(name, _cpp_literal(d.group(1)))
            elif name in NATIVE_ENV_UNSET_EFFECT:
                put(name, {"unset_effect": NATIVE_ENV_UNSET_EFFECT[name]})
            else:
                raise SystemExit(
                    f"native getenv {name} in {rel}: no default and no unset effect"
                )
    return dict(sorted(out.items()))


# Python host modules on the oracle path; every SACCADE_* literal they contain
# is recorded with its value in the oracle environment (None = unset).
HOST_ENV_FILES = (
    MOT17,
    MOT17_ARGS,
    EVALUATOR,
    PIPELINE,
    STAGES,
    "src/saccade/perception/eval/helpers.py",
    "src/saccade/perception/eval/cuda_capture.py",
    "src/saccade/perception/eval/portable_or_tail.py",
    "src/saccade/perception/eval/detection_filters.py",
    "src/saccade/perception/eval/gmc.py",
    "src/saccade/perception/eval/post_merge.py",
    "src/saccade/perception/eval/tracking.py",
    TRACKER_WRAPPER,
    "src/saccade/perception/temporal_yolo/mamba_gated_detector.py",
    "src/saccade/perception/temporal_yolo/mamba_head.py",
)


def host_env(runtime_env: dict[str, str]) -> dict[str, str | None]:
    names: set[str] = set()
    for rel in HOST_ENV_FILES:
        names |= set(re.findall(r'"(SACCADE_[A-Z0-9_]+)"', _source(rel)))
    return {n: runtime_env.get(n) for n in sorted(names)}


# ---------------------------------------------------------------------------
# cfg / kwargs reads in the oracle host modules
# ---------------------------------------------------------------------------

CFG_READ_FILES = (EVALUATOR, PIPELINE, STAGES)
_VIEWS = {
    "core",
    "detection",
    "geometry",
    "motion",
    "reid",
    "semantic",
    "trigger",
    "lifecycle",
}
_KWARGS_NAMES = {"kwargs", "_kw"}


@dataclass(frozen=True)
class Read:
    key: str  # field name, or "kwargs.<key>"
    default: Any  # literal default at the site (getattr/.get), else _NO_DEFAULT
    site: str


_NO_DEFAULT = "<no default>"


def _is_cfg(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "cfg") or (
        isinstance(node, ast.Attribute) and node.attr == "cfg"
    )


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return _NO_DEFAULT
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ("expr", ast.unparse(node))


def cfg_reads() -> list[Read]:
    reads: list[Read] = []
    for rel in CFG_READ_FILES:
        for n in ast.walk(_tree(rel)):
            site = f"{rel}:{getattr(n, 'lineno', 0)}"
            if isinstance(n, ast.Attribute) and _is_cfg(n.value):
                if n.attr not in _VIEWS and n.attr != "kwargs":
                    reads.append(Read(n.attr, _NO_DEFAULT, site))
            elif (
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Attribute)
                and n.value.attr in _VIEWS
                and _is_cfg(n.value.value)
            ):
                reads.append(Read(n.attr, _NO_DEFAULT, site))
            elif isinstance(n, ast.Call):
                f = n.func
                if (
                    isinstance(f, ast.Name)
                    and f.id == "getattr"
                    and len(n.args) >= 2
                    and _is_cfg(n.args[0])
                    and isinstance(n.args[1], ast.Constant)
                    and n.args[1].value != "kwargs"
                ):
                    reads.append(
                        Read(
                            n.args[1].value,
                            _literal(n.args[2] if len(n.args) > 2 else None),
                            site,
                        )
                    )
                elif (
                    isinstance(f, ast.Attribute)
                    and f.attr == "get"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)
                    and (
                        (
                            isinstance(f.value, ast.Attribute)
                            and f.value.attr == "kwargs"
                        )
                        or (
                            isinstance(f.value, ast.Name)
                            and f.value.id in _KWARGS_NAMES
                        )
                        or (
                            isinstance(f.value, ast.Call)
                            and isinstance(f.value.func, ast.Name)
                            and f.value.func.id == "getattr"
                            and len(f.value.args) >= 2
                            and isinstance(f.value.args[1], ast.Constant)
                            and f.value.args[1].value == "kwargs"
                        )
                    )
                ):
                    reads.append(
                        Read(
                            "kwargs." + n.args[0].value,
                            _literal(n.args[1] if len(n.args) > 1 else None),
                            site,
                        )
                    )
    return reads


def read_value(cfg: Any, read: Read) -> Any:
    if read.key.startswith("kwargs."):
        key = read.key[len("kwargs.") :]
        if key in cfg.kwargs:
            return cfg.kwargs[key]
        return None if read.default == _NO_DEFAULT else read.default
    if hasattr(cfg, read.key):
        return getattr(cfg, read.key)
    for view in _VIEWS:
        v = getattr(cfg, view, None)
        if v is not None and hasattr(v, read.key):
            return getattr(v, read.key)
    if read.default == _NO_DEFAULT:
        raise SystemExit(
            f"oracle reads cfg.{read.key} ({read.site}) but EvalConfig lacks it"
        )
    return read.default


def resolved_reads(cfg: Any) -> dict[str, Any]:
    """``{key: value}`` over every read site; one key, one value."""
    out: dict[str, Any] = {}
    for r in cfg_reads():
        v = read_value(cfg, r)
        if isinstance(v, tuple) and v and v[0] == "expr":
            raise SystemExit(
                f"cfg read {r.key} ({r.site}) falls back to a non-literal default {v[1]!r}"
            )
        if (
            r.key in out
            and out[r.key] != v
            and not (
                isinstance(out[r.key], float)
                and isinstance(v, float)
                and out[r.key] == v
            )
        ):
            raise SystemExit(
                f"cfg read {r.key} resolves to {out[r.key]!r} and {v!r} ({r.site})"
            )
        out[r.key] = v
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Host steps: per-frame stage order and the gate of every conditional step
# ---------------------------------------------------------------------------

RUN_EVAL_RULE_CONFIG = Span(
    EVALUATOR,
    "run_eval",
    "external_fp_rule_config = RuleBaselineConfig()",
    "external_fp_rule_config = RuleBaselineConfig()",
)
RUN_EVAL_TAIL_GATES = Span(
    EVALUATOR, "run_eval", "cheb_gr_extractor = None", "_live_bank_enabled = bool("
)


ALL_SPANS: tuple[Span, ...] = (
    MOT17_MAIN,
    RUN_EVAL_CFG,
    RUN_EVAL_PROFILE,
    RUN_EVAL_NATIVE,
    RUN_EVAL_RULE_CONFIG,
    RUN_EVAL_TAIL_GATES,
    PIPELINE_SETUP,
    PIPELINE_SEQ,
    *PIPELINE_CAPS,
)


@dataclass(frozen=True)
class Step:
    name: str
    rel: str  # oracle module whose source contains ``expr`` (AST-normalised)
    expr: str  # the oracle's own gate expression, evaluated against the oracle


# Ordered as the oracle runs them. Disabled steps stay listed so the shipping
# loader can refuse a config that turns one on (boundary §5 B2/B4).
STEPS: tuple[Step, ...] = (
    Step("ingest.gpu_decode", PIPELINE, 'os.environ.get("SACCADE_GPU_DECODE") == "1"'),
    Step(
        "ingest.nv12_buffer", PIPELINE, 'os.environ.get("SACCADE_NV12_BUFFER") == "1"'
    ),
    Step(
        "schedule.double_buffer",
        PIPELINE,
        "_double_buffer_eligible(cfg, detector, profile_stages)",
    ),
    Step("post.native_postprocess", EVALUATOR, "native_postprocess_available"),
    Step("post.private_continuation", EVALUATOR, "native_private_available"),
    Step("post.onms_priors", EVALUATOR, "enable_onms"),
    Step("post.detection_quality", STAGES, "cfg.geometry.detection_quality_scaling"),
    Step("filter.crowd_low_score", EVALUATOR, "cfg.detection.crowd_low_score_mode"),
    Step(
        "filter.external_fp", STAGES, "cfg.detection.external_fp_filter_mode != 'off'"
    ),
    Step("filter.fp_hard", STAGES, "cfg.detection.fp_hard_filter_enabled"),
    Step("filter.duplicate_suppression", STAGES, "cfg.duplicate_suppression"),
    Step("filter.detection_cap", STAGES, "cfg.detection.per_frame_detection_cap > 0"),
    Step("birth.consecutive_gate", STAGES, "cfg.birth_consecutive_gate"),
    Step("birth.quality_gate", STAGES, "cfg.birth_quality_gate"),
    Step("birth.multi_birth", PIPELINE, "cfg.multi_birth_enabled"),
    Step(
        "reid.work", EVALUATOR, "cfg.reid_work_enabled"
    ),  # builds the extractor/cropper
    Step("gmc", PIPELINE, "cfg.gmc_enabled"),
    Step("gmc.fg_mask", STAGES, "cfg.gmc_fg_mask"),
    Step("relink.semantic_relinker", PIPELINE, "cfg.use_semantic_mode"),
    Step(
        "relink.native_bridge",
        PIPELINE,
        "(cfg.relink_enabled or _bridge_enabled) and hasattr(detector.tracker, 'set_relink_params')",
    ),
    Step(
        "track.graphed_update",
        PIPELINE,
        "cfg.kwargs.get('use_tracker_graph', False) and (not cfg.relink_enabled)",
    ),
    Step("emit.pipeline_relink", STAGES, "cfg.pipeline_relink"),
    Step(
        "tail.cheb_gr_or_occ_audit",
        EVALUATOR,
        "cfg.cheb_gr_merge_enabled or cheb_gr_online or occ_audit_enabled or _live_bank_enabled",
    ),
    Step("tail.post_lifecycle_merge", EVALUATOR, "cfg.post_lifecycle_merge"),
    Step(
        "tail.deferred_alias",
        EVALUATOR,
        "_seq_state.relinker is not None and cfg.kwargs.get('semantic_delayed_claim', False)",
    ),
    Step(
        "tail.tracklet_quality_filter",
        EVALUATOR,
        "cfg.min_tracklet_len > 1 or cfg.min_tracklet_score > 0.0",
    ),
    Step("tail.interpolation", EVALUATOR, "cfg.interpolate_tracklets"),
    Step("tail.write_output", EVALUATOR, "not cfg.core.latency_only"),
)


@functools.lru_cache(maxsize=None)
def _oracle_expressions(rel: str) -> frozenset[str]:
    """Decision expressions in ``rel``: whole branch tests and their ``and``
    conjuncts (a false conjunct means the branch is not taken), plus assigned
    and keyword-argument values (gates that are computed, then passed on)."""
    exprs: set[str] = set()

    def add(node: ast.expr, conjuncts: bool) -> None:
        exprs.add(ast.unparse(node))
        if conjuncts and isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for operand in node.values:
                add(operand, True)

    for n in ast.walk(_tree(rel)):
        if isinstance(n, (ast.If, ast.IfExp, ast.While)):
            add(n.test, True)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            add(n.value, False)
        elif isinstance(n, ast.keyword):
            add(n.value, False)
    return frozenset(exprs)


# Stage helpers ``_run_frame`` can call that are diagnostics, with the env
# variable that selects them (it must be unset in the oracle environment).
DIAGNOSTIC_STAGES: dict[str, tuple[str, ...]] = {
    "_run_nms_shadow_compare": (
        "SACCADE_MAIN_NMS_SHADOW",
        "SACCADE_MAIN_NMS_GRAPH_SHADOW",
        "SACCADE_MAIN_NMS_GRAPHED_SHADOW",
    ),
}


def stage_order() -> list[str]:
    """``_run_*`` stage helpers in the order ``_run_frame`` calls them."""
    order: list[tuple[int, str]] = []
    for n in ast.walk(function_node(EVALUATOR, "_run_frame")):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id.startswith("_run_")
        ):
            order.append((n.lineno, n.func.id))
    seen: list[str] = []
    for _, name in sorted(order):
        if name not in seen and name not in DIAGNOSTIC_STAGES:
            seen.append(name)
    return seen


def verify_step_gates() -> None:
    for step in STEPS:
        normalised = ast.unparse(ast.parse(step.expr, mode="eval").body)
        if normalised not in _oracle_expressions(step.rel):
            raise SystemExit(
                f"oracle drift: gate of {step.name} not in {step.rel}: {step.expr}"
            )


def evaluate_steps(oracle: Oracle) -> dict[str, bool]:
    import torch

    verify_step_gates()
    ns = dict(oracle.run_eval_ns)
    ns.update(oracle.pipeline_ns)
    ns["os"] = os
    ns["_seq_state"] = types.SimpleNamespace(relinker=oracle.pipeline_ns["relinker"])
    out: dict[str, bool] = {}
    with mock.patch.object(torch.cuda, "is_available", lambda: True):
        for step in STEPS:
            out[step.name] = bool(eval(step.expr, ns))  # noqa: S307
    return out


# ---------------------------------------------------------------------------
# cfg read classification
# ---------------------------------------------------------------------------

# Reads that name a run's inputs/outputs rather than behaviour: machine paths,
# sequence selection, Python objects/callbacks handed to run_eval. Every other
# read lands in host_params.cfg with its value.
HARNESS_READS: dict[str, str] = {
    "data_root": "dataset location (input)",
    "split": "dataset split (input)",
    "seqs": "sequence selection (input)",
    "output_root": "output directory (eval harness)",
    "kwargs.detector": "the built detector object (host_params.detector records its build)",
    "kwargs.sequence_result_callback": "eval harness callback",
    "kwargs.stage_probe_callback": "eval harness callback",
}


def _jsonable(value: Any, log: NativeLog | None = None) -> Any:
    if isinstance(value, PerSequence):
        return value.json()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, log) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, log) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    rec = getattr(value, "_rec_id", None)
    if rec is not None and log is not None:
        cls = type(value).__name__
        if cls in SHIPPING_NATIVE:
            return {"ref": cls}
        return {
            name: _jsonable(v, log)
            for c in log.calls
            if c.instance == rec and c.method.startswith(".")
            for name, v in name_call(cls, c).items()
        }
    raise SystemExit(
        f"value of type {type(value).__name__} is not exportable: {value!r}"
    )


def native_params(log: NativeLog) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in log.calls:
        if c.cls in NON_SHIPPING_NATIVE or c.cls in {
            "UnifiedScoreParams",
            "MambaHead",
            "Detector",
        }:
            continue
        if c.cls not in SHIPPING_NATIVE:
            raise SystemExit(
                f"oracle configured unlisted native class {c.cls}.{c.method}"
            )
        entry = out.setdefault(c.cls, {})
        named = {k: _jsonable(v, log) for k, v in name_call(c.cls, c).items()}
        if c.method == "__init__":
            if "constructor" in entry:
                raise SystemExit(f"{c.cls} constructed twice")
            entry["constructor"] = named
        elif c.method.startswith("."):
            entry.setdefault("fields", {}).update(named)
        else:
            entry.setdefault("calls", []).append({"method": c.method, "args": named})
    cfg_fields = {
        f[1:]
        for f in binding_signatures("PerceptionPipelineConfig")
        if f.startswith(".")
    }
    unset = cfg_fields - set(out.get("PerceptionPipelineConfig", {}).get("fields", {}))
    if unset:
        raise SystemExit(
            f"PerceptionPipelineConfig fields left at C++ defaults: {sorted(unset)}"
        )
    return out


def _check_non_shipping(log: NativeLog) -> None:
    for c in log.calls:
        if c.cls == "TrackletLifecycleMerger" and c.method == "__init__":
            if c.kwargs.get("enabled") is not False:
                raise SystemExit(
                    "lifecycle merger is enabled; shipping does not implement it"
                )
        if c.cls in {
            "SemanticRelinker",
            "PythonSemanticRelinker",
            "ZeroCopyCropper",
            "TRTFeatureExtractor",
        }:
            raise SystemExit(
                f"oracle constructed {c.cls}; headline ReID/relinker must be off"
            )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _fp_hard_reject_score() -> float:
    from saccade.perception.eval import detection_filters, stages

    if stages._FP_HARD_REJECT_SCORE != detection_filters._FP_HARD_REJECT_SCORE:
        raise SystemExit(
            "stages and detection_filters disagree on _FP_HARD_REJECT_SCORE"
        )
    return float(stages._FP_HARD_REJECT_SCORE)


def _sha256(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def _capture(seq_fps: int = 30) -> tuple[Oracle, dict[str, Any]]:
    gaps = native_touch_gaps()
    if gaps:
        raise SystemExit(
            "native-touching oracle statements not covered:\n  " + "\n  ".join(gaps)
        )
    log = NativeLog()
    with oracle_environment():
        oracle = resolve_oracle(log)
        capture_native(oracle, seq_fps=seq_fps)
        run_statements(EVALUATOR, RUN_EVAL_RULE_CONFIG.statements(), oracle.run_eval_ns)
        run_statements(EVALUATOR, RUN_EVAL_TAIL_GATES.statements(), oracle.run_eval_ns)
        steps = evaluate_steps(oracle)
        env = host_env(oracle.runtime_env)
    for stage, names in DIAGNOSTIC_STAGES.items():
        if any(env.get(n) for n in names):
            raise SystemExit(f"diagnostic stage {stage} is selected in the oracle env")
    _check_non_shipping(log)
    return oracle, {"steps": steps, "env": env, "native": native_params(log)}


def build() -> dict[str, Any]:
    import dataclasses

    oracle, cap = _capture()
    # set_params' track_buffer and every other native value must not depend
    # on the sequence frame rate (per_seq_adapt is off in the headline).
    for fps in (14, 25):
        if _capture(seq_fps=fps)[1]["native"] != cap["native"]:
            raise SystemExit(f"native params depend on seqinfo frameRate ({fps} fps)")

    reads = resolved_reads(oracle.cfg)
    unknown_harness = set(HARNESS_READS) - set(reads)
    if unknown_harness:
        raise SystemExit(
            f"HARNESS_READS lists keys the oracle no longer reads: {sorted(unknown_harness)}"
        )
    cfg_values = {k: _jsonable(v) for k, v in reads.items() if k not in HARNESS_READS}

    run_ns, pipe_ns = oracle.run_eval_ns, oracle.pipeline_ns
    build_kwargs = dict(oracle.detector_build)
    head_calls = {
        c.method: list(c.args) for c in oracle.detector_calls if c.method != "__init__"
    }
    return {
        "schema": SCHEMA,
        "source": {
            "preset": PRESET,
            "preset_sha256": _sha256(PRESET),
            "oracle_argv": list(ORACLE_ARGV),
            "exporter": str(Path(__file__).resolve().relative_to(REPO)),
            "per_sequence_values": [SEQ_WIDTH.json(), SEQ_HEIGHT.json()],
            "native_objects_not_shipped": dict(sorted(NON_SHIPPING_NATIVE.items())),
            "harness_reads_not_exported": dict(sorted(HARNESS_READS.items())),
        },
        "native_params": cap["native"],
        "native_env": native_env(),
        "host_params": {
            "stage_order": stage_order(),
            "steps": cap["steps"],
            "detector": {
                "build": _jsonable(build_kwargs),
                "head_calls": _jsonable(head_calls),
                "detect_fn": run_ns["detect_fn"].__name__,
                "contract": _jsonable(dataclasses.asdict(run_ns["contract"])),
                "calls": [
                    {"method": c.method, "args": _jsonable(list(c.args))}
                    for c in oracle.log.of("Detector")
                ],
            },
            "capacities": {
                "track_result_cap": pipe_ns["_TRACK_RESULT_CAP"],
                "nms_fixed_n": pipe_ns["_NMS_FIXED_N"],
            },
            "external_fp_rule_config": _jsonable(
                dataclasses.asdict(run_ns["external_fp_rule_config"])
            ),
            "onms": {
                "enabled": run_ns["enable_onms"],
                "prior_iou_threshold": run_ns["onms_prior_iou_threshold"],
                "min_track_age": run_ns["onms_min_track_age"],
                "min_track_score": run_ns["onms_min_track_score"],
            },
            "initial_tracker_thresholds": _jsonable(
                pipe_ns["active_tracker_thresholds"]
            ),
            # Score written over FP-hard rejects (mask-in-place; the tracker's
            # score gate then drops them).
            "fp_hard_reject_score": _fp_hard_reject_score(),
            "env": cap["env"],
            "cfg": cfg_values,
        },
    }


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=False, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed file"
    )
    ap.add_argument("--output", default=OUTPUT)
    a = ap.parse_args(argv)
    text = render(build())
    out = REPO / a.output
    if a.check:
        if not out.exists() or out.read_text() != text:
            print(
                f"STALE: {a.output} differs from a fresh export; rerun without --check"
            )
            return 1
        print(f"OK: {a.output} matches the oracle")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
