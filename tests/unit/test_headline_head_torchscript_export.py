"""The LibTorch head exporter produces a Python-free, input-independent, identifiable artifact.

``scripts/model/export_headline_mamba_head_torchscript.py`` (#465 Phase B
PR-1L) traces the headline head with the scan routed through the C++ operator
``saccade_native::selective_scan_fwd`` (``src/tracking/mamba_scan_torchop.cpp``).
These tests pin the fail-closed rules that do not need the checkpoint: the
graph check (no Python op, no Python scan, at least one native scan), the
TracerWarning allowlist (shape-only sites that still exist verbatim in
``mamba_head.py``), the portable content hash, the runtime-requirement and
native-scan context managers, and the op library's no-Python link rule. When
CUDA and ``build/libsaccade_scan_torchop.so`` are present, the native operator
must equal the Python custom op bit for bit, accept every legal input form, and
reject every contract violation (device, dtype, shape, ``a_per_channel``) with
its own error before any kernel launch -- it is the shipping runtime's operator
boundary.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import sys
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "model" / "export_headline_mamba_head_torchscript.py"
OP_LIBRARY = REPO / "build" / "libsaccade_scan_torchop.so"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "export_headline_mamba_head_torchscript", TOOL
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _tool()


# --- graph and tracer checks -------------------------------------------------
def test_graph_problems_require_native_scan_and_no_python():
    ok = "%y = saccade_native::selective_scan_fwd(%u, ...)"
    assert T.graph_problems(ok) == []
    assert T.graph_problems("aten::conv2d") == [
        "graph never calls saccade_native::selective_scan_fwd"
    ]
    assert "graph contains prim::PythonOp" in T.graph_problems(ok + "\nprim::PythonOp")
    assert any(
        "Python op saccade::selective_scan_fwd" in p
        for p in T.graph_problems(ok + "\nsaccade::selective_scan_fwd(")
    )


def test_allowed_tracer_warning_sites_exist_verbatim():
    """If mamba_head.py changes these lines, the allowlist must be re-reviewed."""
    for rel, line in T.ALLOWED_TRACER_WARNING_SITES:
        lines = {x.strip() for x in (REPO / rel).read_text().splitlines()}
        assert line in lines, (rel, line)


def _warning(path: Path, line_text: str, category=torch.jit.TracerWarning):
    lines = path.read_text().splitlines()
    lineno = next(i for i, x in enumerate(lines, 1) if x.strip() == line_text)
    return SimpleNamespace(category=category, filename=str(path), lineno=lineno)


def test_tracer_warnings_outside_the_allowlist_fail_closed(tmp_path):
    rel, line = sorted(T.ALLOWED_TRACER_WARNING_SITES)[0]
    allowed = _warning(REPO / rel, line)
    sites, problems = T.tracer_warning_problems([allowed, allowed])
    assert problems == [] and len(sites) == 1 and sites[0]["source"] == line
    data_branch = tmp_path / "head.py"
    data_branch.write_text("if x.sum() > 0:\n    pass\n")
    _, problems = T.tracer_warning_problems([_warning(data_branch, "if x.sum() > 0:")])
    assert len(problems) == 1 and "unexpected TracerWarning" in problems[0]
    # non-tracer warnings (e.g. the jit.trace deprecation) are not sites
    other = _warning(data_branch, "if x.sum() > 0:", category=DeprecationWarning)
    assert T.tracer_warning_problems([other]) == ([], [])


# --- identity ------------------------------------------------------------------
def _archive(path: Path, entries: dict[str, bytes], root: str = "a") -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(f"{root}/{name}", data)


def test_content_sha256_ignores_only_save_noise(tmp_path):
    base = {
        "data.pkl": b"weights",
        "code/__torch__/m.py": b"def forward(self): ...",
        "code/__torch__/m.py.debug_pkl": b"/home/x/trace.py(1)",
        ".data/serialization_id": b"123",
    }
    a, b, c, d = (tmp_path / f"{k}.pt" for k in "abcd")
    _archive(a, base)
    _archive(
        b,
        {
            **base,
            ".data/serialization_id": b"456",
            "code/__torch__/m.py.debug_pkl": b"/tmp/y(2)",
        },
        root="other_root",
    )
    _archive(c, {**base, "data.pkl": b"weightz"})
    _archive(d, {**base, "code/__torch__/extra.py": b""})
    assert T.content_sha256(a) == T.content_sha256(b)
    assert T.content_sha256(a) != T.content_sha256(c)
    assert T.content_sha256(a) != T.content_sha256(d)


# --- runtime requirements and scan routing ------------------------------------
def test_runtime_requirements_are_the_oracle_defaults_and_restored():
    assert T.RUNTIME_REQUIREMENTS == {
        "graph_executor_optimize": False,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": True,
        "matmul_allow_tf32": False,
    }
    before = (
        torch._C._get_graph_executor_optimize(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    with T.runtime_requirements():
        assert torch._C._get_graph_executor_optimize() is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
    after = (
        torch._C._get_graph_executor_optimize(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    assert after == before


def test_native_scan_swaps_only_the_scan_op_and_restores_it(monkeypatch):
    sys.path.insert(0, str(REPO / "src"))
    from saccade.perception.temporal_yolo import mamba_head as mh

    sentinel = object()
    fake_ops = SimpleNamespace(
        saccade_native=SimpleNamespace(selective_scan_fwd=sentinel)
    )
    monkeypatch.setattr(torch, "ops", fake_ops)
    original = mh._saccade_selective_scan_op
    with T.native_scan():
        assert mh._saccade_selective_scan_op is sentinel
    assert mh._saccade_selective_scan_op is original


def test_op_library_must_not_link_python(tmp_path, monkeypatch):
    def fake_readelf(needed):
        out = "\n".join(
            f" 0x01 (NEEDED)             Shared library: [{n}]" for n in needed
        )
        return lambda *a, **k: SimpleNamespace(stdout=out)

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_readelf(["libtorch.so", "libc.so.6"]))
    assert T.needed_libraries(tmp_path / "x.so") == ["libtorch.so", "libc.so.6"]
    for bad in ("libpython3.12.so.1.0", "libtorch_python.so"):
        monkeypatch.setattr(subprocess, "run", fake_readelf(["libtorch.so", bad]))
        with pytest.raises(SystemExit):
            T.needed_libraries(tmp_path / "x.so")


def test_wrapper_module_is_pinned():
    head = torch.nn.Identity()
    assert type(T.make_wrapper(head)).__module__ == T.WRAPPER_MODULE


# --- native operator == Python custom op (GPU + built library only) -----------
@pytest.mark.skipif(
    not torch.cuda.is_available() or not OP_LIBRARY.exists(),
    reason="needs CUDA and build/libsaccade_scan_torchop.so",
)
@pytest.mark.parametrize(
    ("b", "length", "d", "n", "per_channel", "has_d"),
    [(1, 400, 256, 16, 1, True), (2, 100, 128, 8, 0, True), (1, 64, 128, 16, 1, False)],
)
def test_native_op_equals_python_op_bitwise(b, length, d, n, per_channel, has_d):
    sys.path.insert(0, str(REPO / "build"))
    sys.path.insert(0, str(REPO / "src"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytest.importorskip("saccade_tracking_ext")
    from saccade.perception.temporal_yolo import mamba_head as mh

    torch.ops.load_library(str(OP_LIBRARY))
    g = torch.Generator(device="cuda").manual_seed(0)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g)

    u, delta, B, C = r(b, length, d), r(b, length, d), r(b, length, n), r(b, length, n)
    A = -torch.rand(d if per_channel else 1, n, device="cuda", generator=g)
    D = r(d) if has_d else torch.empty(0, device="cuda")
    y_py = mh._saccade_selective_scan_op(u, delta, A, B, C, D, per_channel, False)
    y_nat = torch.ops.saccade_native.selective_scan_fwd(
        u, delta, A, B, C, D, per_channel, False
    )
    torch.cuda.synchronize()
    assert torch.equal(y_py.view(torch.int32), y_nat.view(torch.int32))


# --- operator boundary contract (GPU + built library only) --------------------
needs_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not OP_LIBRARY.exists(),
    reason="needs CUDA and build/libsaccade_scan_torchop.so",
)


def _op_inputs(
    b=1, length=16, d=8, n=4, per_channel=1, has_d=True, dtype=torch.float32
):
    torch.ops.load_library(str(OP_LIBRARY))
    g = torch.Generator(device="cuda").manual_seed(0)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g).to(dtype)

    return {
        "u": r(b, length, d),
        "delta": r(b, length, d),
        "A": -torch.rand(d if per_channel else 1, n, device="cuda", generator=g).to(
            dtype
        ),
        "B": r(b, length, n),
        "C": r(b, length, n),
        "D": r(d) if has_d else torch.empty(0, device="cuda", dtype=dtype),
        "a_per_channel": per_channel,
        "is_half": dtype == torch.float16,
    }


def _call(kw):
    return torch.ops.saccade_native.selective_scan_fwd(
        kw["u"],
        kw["delta"],
        kw["A"],
        kw["B"],
        kw["C"],
        kw["D"],
        kw["a_per_channel"],
        kw["is_half"],
    )


@needs_op
@pytest.mark.parametrize(
    "form",
    ["per_channel", "shared_2d", "shared_1d", "no_D", "half", "non_contiguous"],
)
def test_operator_accepts_every_legal_form(form):
    if form == "per_channel":
        kw = _op_inputs()
    elif form == "shared_2d":
        kw = _op_inputs(per_channel=0)
    elif form == "shared_1d":
        kw = _op_inputs(per_channel=0)
        kw["A"] = kw["A"].reshape(-1)
    elif form == "no_D":
        kw = _op_inputs(has_d=False)
    elif form == "half":
        kw = _op_inputs(dtype=torch.float16)
    else:
        kw = _op_inputs()
        kw["u"] = kw["u"].transpose(1, 2).contiguous().transpose(1, 2)
    y = _call(kw)
    torch.cuda.synchronize()
    assert y.shape == kw["u"].shape and y.dtype == kw["u"].dtype


def _mutations():
    def cpu(name):
        return lambda kw: kw.__setitem__(name, kw[name].cpu())

    def dtype(name):
        return lambda kw: kw.__setitem__(name, kw[name].double())

    def reshape(name, *shape):
        return lambda kw: kw.__setitem__(
            name, torch.zeros(*shape, device="cuda", dtype=kw["u"].dtype)
        )

    return {
        "u on CPU": cpu("u"),
        "u rank 2": reshape("u", 16, 8),
        "u empty": lambda kw: kw.update(
            u=kw["u"][:, :0],
            delta=kw["delta"][:, :0],
            B=kw["B"][:, :0],
            C=kw["C"][:, :0],
        ),
        "u float64": lambda kw: [
            kw.__setitem__(k, kw[k].double()) for k in "u delta A B C D".split()
        ],
        "is_half mismatch": lambda kw: kw.__setitem__("is_half", True),
        "a_per_channel 2": lambda kw: kw.__setitem__("a_per_channel", 2),
        "delta on CPU": cpu("delta"),
        "delta dtype": dtype("delta"),
        "delta shape": reshape("delta", 1, 16, 9),
        "A on CPU": cpu("A"),
        "A dtype": dtype("A"),
        "A N not power of two": reshape("A", 8, 3),
        "A N too large": reshape("A", 8, 64),
        "A rank 3": reshape("A", 1, 8, 4),
        "per-channel A wrong rows": reshape("A", 7, 4),
        "shared A wrong numel": lambda kw: kw.update(a_per_channel=0, A=kw["A"][:2]),
        "B on CPU": cpu("B"),
        "B dtype": dtype("B"),
        "B wrong N": reshape("B", 1, 16, 8),
        "B wrong L": reshape("B", 1, 15, 4),
        "C on CPU": cpu("C"),
        "C dtype": dtype("C"),
        "C rank-1 (not broadcast)": reshape("C", 1, 16, 1),
        "D on CPU": cpu("D"),
        "D dtype": dtype("D"),
        "D wrong length": reshape("D", 7),
        "D rank 2": reshape("D", 1, 8),
    }


@needs_op
@pytest.mark.parametrize("case", sorted(_mutations()))
def test_operator_rejects_contract_violations_before_launch(case):
    kw = _op_inputs()
    _mutations()[case](kw)
    with pytest.raises(RuntimeError, match="saccade_native::selective_scan_fwd: "):
        _call(kw)
    # the error is raised before any launch: the CUDA context is still healthy
    torch.cuda.synchronize()
    assert _call(_op_inputs()).isfinite().all()
