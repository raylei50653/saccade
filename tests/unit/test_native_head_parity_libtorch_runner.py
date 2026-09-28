"""The #465 PR-2L runner implements its frozen declaration and PR-2's policy verbatim.

``scripts/eval/diagnostics/native_head_parity_libtorch.py`` implements
``docs/reference/native_runtime_head_parity_libtorch_declaration.md``. These
tests pin (a) the frozen declaration blob and every frozen identity the
declaration cites, (b) PR-2's decision constants and functions (source-equal
to the PR-2 runner), the r2 freeze-tag logic and the exporter's artifact
identity rules, (c) the fail-closed validity functions V1 (driver/runtime,
R2 libcudart dedup by ``(st_dev, st_ino)``), V2, V4, V5, and (d) a command
line with no option that could change the study identity. No GPU, no MOT17.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "scripts" / "eval" / "diagnostics"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIAG / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("native_head_parity_libtorch")
PR2 = _load("native_head_parity")


def _top_level(path: Path) -> dict[str, str]:
    """Source of every top-level function/class and constant assignment."""
    src = path.read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                out[t.id] = ast.get_source_segment(src, node.value)
    return out


RUNNER_SRC = _top_level(DIAG / "native_head_parity_libtorch.py")
PR2_SRC = _top_level(DIAG / "native_head_parity.py")
R2_SRC = _top_level(DIAG / "native_head_failure_localization_r2.py")
EXPORT_SRC = _top_level(
    REPO / "scripts/model/export_headline_mamba_head_torchscript.py"
)
DECL_TEXT = (REPO / R.DECLARATION).read_text()


# --- (a) frozen declaration and identities --------------------------------


def test_declaration_blob_is_the_frozen_one():
    blob = subprocess.run(
        ["git", "hash-object", R.DECLARATION],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert blob == R.FROZEN_DECLARATION_BLOB  # an amendment must move this constant


def test_declaration_freeze_commit_is_the_485_merge():
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s%n%P", R.DECLARATION_FREEZE_COMMIT],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if subject.returncode != 0:
        pytest.skip("shallow clone without the declaration freeze commit")
    first, parents = subject.stdout.strip().splitlines()
    assert first.startswith("Merge pull request #485 ")
    assert len(parents.split()) == 2
    frozen = subprocess.run(
        ["git", "show", f"{R.DECLARATION_FREEZE_COMMIT}:{R.DECLARATION}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # Since the freeze the declaration has only been appended to (§11 amendments).
    marker = "**凍結後的 amendments**（append-only）："
    cut = frozen.index(marker) + len(marker)
    assert DECL_TEXT[:cut] == frozen[:cut]
    assert frozen[cut:].strip() == "（無）"
    assert "- **A1（2026-09-28" in DECL_TEXT[cut:]


@pytest.mark.parametrize(
    "value",
    [
        R.ARTIFACT,
        R.ARTIFACT_SHA256,
        R.ARTIFACT_CONTENT_SHA256,
        R.ARTIFACT_LINEAGE,
        R.ARTIFACT_LINEAGE_SHA256,
        R.OP_LIBRARY,
        R.OP_LIBRARY_SHA256,
        R.OP_SOURCE,
        R.OP_SOURCE_BLOB,
        R.CKPT,
        R.CKPT_SHA256,
        R.BACKBONE,
        R.BACKBONE_SHA256,
        R.PRESET,
        R.PRESET_SHA256,
        R.ENVIRONMENT["torch"],
        str(R.ENVIRONMENT["cudnn"]),
        R.ENVIRONMENT["gpu"],
        f"SM {R.ENVIRONMENT['sm']}",
        R.ENVIRONMENT["host"],
        f"`{R.NVIDIA_DRIVER}`",
        f"`cuDriverGetVersion` == `{R.CU_DRIVER_VERSION}`",
        f"`cudaRuntimeGetVersion` == `{R.CUDART_VERSION}`",
        R.CUDART_RELPATH,
        R.CUDART_SHA256,
        "(st_dev, st_ino)",
        R.FREEZE_TAG,
        R.PACKET_ROOT,
        R.RUNTIME_IDENTITY,
        ",".join(R.SEQUENCES),
        f"共 {R.TOTAL_FRAMES} frames",
        R.NATIVE_OP,
        R.PYTHON_OP,
        "`--no-compile`",
        "`A_C#1, A_L#1, A_N#1, A_C#2, A_L#2, A_N#2`",
        *R.ORACLE_ANCHOR_TXT_SHA256.values(),
        *(g for g in R.JIT_FALLBACK_GUARDS),
    ],
)
def test_frozen_value_appears_in_the_declaration(value):
    assert value in DECL_TEXT, value


def test_runtime_requirements_match_the_declaration_and_exporter():
    assert R.RUNTIME_REQUIREMENTS == ast.literal_eval(
        EXPORT_SRC["RUNTIME_REQUIREMENTS"]
    )
    assert (
        "`{graph_executor_optimize: false, cudnn_benchmark: false, "
        "cudnn_allow_tf32: true, matmul_allow_tf32: false}`" in DECL_TEXT
    )


def test_native_scan_call_count_is_declared():
    assert R.NATIVE_SCAN_CALLS == 3
    assert f"`{R.NATIVE_OP}` 恰 3 次" in DECL_TEXT


def test_driver_runtime_values_match_the_declaration_row():
    row = next(
        line for line in DECL_TEXT.splitlines() if line.startswith("| NVIDIA driver")
    )
    for v in (
        R.NVIDIA_DRIVER,
        str(R.CU_DRIVER_VERSION),
        str(R.CUDART_VERSION),
        R.CUDART_RELPATH,
        R.CUDART_SHA256,
    ):
        assert v in row, v


def test_prior_reference_packets_are_declared_report_only_sources():
    assert R.PRIOR_REFERENCE_TXT["A_N_vs_PR2R_A_N"][1].startswith(
        "results/native_head_parity_465_tf32_off/20260927T151100Z/"
    )
    assert "20260927T151100Z" in DECL_TEXT
    assert "`R_E`" in DECL_TEXT


# --- (b) PR-2 policy verbatim, r2 freeze logic, exporter identity rules ---


@pytest.mark.parametrize(
    "name",
    [
        "PRESET_NAME",
        "DATA_ROOT",
        "SPLIT",
        "SEQUENCES",
        "TOTAL_FRAMES",
        "LEASE",
        "IMG_SIZE",
        "SCORE_FLOOR",
        "L1_SCORE_MAX",
        "L1_BOX_MAX_PX",
        "V2_FRAMES",
        "L2_METRICS",
        "TOL_FLOOR",
        "TOL_CAP",
        "TERMINALS",
        "CKPT",
        "CKPT_SHA256",
        "BACKBONE",
        "HIST_LO_EXP",
    ],
)
def test_decision_constants_equal_pr2(name):
    assert getattr(R, name) == getattr(PR2, name)


def test_only_t_becomes_l():
    swap = {"T": "L", "A_T": "A_L"}
    assert R.L1_PAIRS == tuple(tuple(swap.get(x, x) for x in p) for p in PR2.L1_PAIRS)
    assert R.L1_DECISION_PAIR == tuple(swap.get(x, x) for x in PR2.L1_DECISION_PAIR)
    assert R.RUN_ORDER == tuple((swap.get(a, a), r) for a, r in PR2.RUN_ORDER)
    assert list(R.ARMS) == [swap.get(a, a) for a in PR2.ARMS]
    assert R.ARMS["A_C"] == PR2.ARMS["A_C"] == []
    assert R.ARMS["A_N"] == PR2.ARMS["A_N"] == ["--no-compile"]
    assert R.ARMS["A_L"] == []  # injected by the child, never a harness option
    assert (R.HIST_LO_EXP, R.HIST_HI_EXP, R.HIST_PER_DECADE) == (
        PR2.HIST_LO_EXP,
        PR2.HIST_HI_EXP,
        PR2.HIST_PER_DECADE,
    )


@pytest.mark.parametrize(
    "name",
    [
        "tolerance",
        "l1_verdict",
        "l2_verdict",
        "decide_terminal",
        "metrics_from_counts",
        "first_divergent_frame",
        "hist_edges",
        "hist_quantile",
        "score_arm",
        "_l1_row",
        "held_lease",
        "_write_manifest",
    ],
)
def test_decision_functions_are_pr2_verbatim(name):
    assert RUNNER_SRC[name] == PR2_SRC[name]


@pytest.mark.parametrize(
    "name", ["freeze_problems", "parse_ls_remote_peeled", "observed_freeze"]
)
def test_freeze_logic_is_r2_verbatim(name):
    assert RUNNER_SRC[name] == R2_SRC[name]


@pytest.mark.parametrize(
    "name",
    ["content_sha256", "graph_problems", "NATIVE_OP", "PYTHON_OP", "CONTENT_EXCLUDED"],
)
def test_artifact_identity_rules_equal_the_exporter(name):
    assert RUNNER_SRC[name] == EXPORT_SRC[name]


def test_l1_worker_measures_what_pr2_measures():
    """The per-frame L1 loop keeps PR-2's decode, mask and accumulation lines."""
    mine, pr2 = RUNNER_SRC["run_l1_worker"], PR2_SRC["run_l1_worker"]
    start = pr2.index("                sx, sy = w_orig / IMG_SIZE")
    end = pr2.index("            per_seq[seq] = {")
    assert pr2[start:end] in mine
    for block in ("    def decode(", "    def hist(", "    def new_acc("):
        a = pr2[pr2.index(block) :]
        assert a[: a.index("\n\n")] in mine


@pytest.mark.parametrize(
    ("validity", "l1", "l2", "terminal"),
    [
        (False, "L1_PASS", "L2_EXACT", "UNRESOLVED"),
        (True, None, "L2_EXACT", "UNRESOLVED"),
        (True, "L1_PASS", None, "UNRESOLVED"),
        (True, "L1_GROSS_ERROR", "L2_OUT", "HEAD_PARITY_GROSS_ERROR"),
        (True, "L1_GROSS_ERROR", "L2_EXACT", "HEAD_PARITY_GROSS_ERROR"),
        (True, "L1_PASS", "L2_EXACT", "HEAD_PARITY_EXACT"),
        (True, "L1_PASS", "L2_WITHIN", "HEAD_PARITY_WITHIN_TOLERANCE"),
        (True, "L1_PASS", "L2_OUT", "HEAD_PARITY_OUT_OF_TOLERANCE"),
        (True, "L1_PASS", "bogus", "UNRESOLVED"),
    ],
)
def test_terminal_order(validity, l1, l2, terminal):
    assert R.decide_terminal(validity, l1, l2) == terminal
    assert terminal in R.TERMINALS


def test_tolerance_floor_and_cap():
    assert R.tolerance(0.0, "IDF1") == 0.20
    assert R.tolerance(-0.5, "IDF1") == 0.5
    assert R.tolerance(3.0, "HOTA") == 1.00
    assert R.tolerance(0.0, "IDs") == 5.0
    assert R.tolerance(100.0, "IDs") == 30.0


# --- (c) validity functions ---------------------------------------------


def _st(dev: int, ino: int) -> SimpleNamespace:
    return SimpleNamespace(st_dev=dev, st_ino=ino)


MAPS = "\n".join(
    [
        "7f00-7f01 r--p 00000000 08:20 100 /v/nvidia/cu13/lib/libcudart.so.13",
        "7f01-7f02 r-xp 00001000 08:20 100 /v/nvidia/cu13/lib/libcudart.so.13",
        "7f02-7f03 r--p 00002000 08:20 100 /v/nvidia/cu13/lib/libcudart.so.13",
        "7f03-7f04 rw-p 00003000 08:20 100 /v/nvidia/cu13/lib/libcudart.so.13",
        "7f05-7f06 r--p 00000000 08:20 200 /usr/lib/wsl/lib/libcuda.so.1",
        "7f06-7f07 rw-p 00000000 00:00 0 ",
        "7f07-7f08 rw-p 00000000 00:00 0 [heap]",
        "7f08-7f09 r--p 00000000 08:20 300 /opt/lib/libcudart_static_helper.so",
    ]
)


def test_libcudart_segments_of_one_elf_are_one_backing_file():
    paths = R.maps_paths(MAPS, "libcudart.so")
    assert len(paths) == 4  # four mapping lines ...
    files = R.backing_files(paths, lambda p: _st(1, 100))
    assert len(files) == 1  # ... one backing file (R2: not the line count)
    assert files[(1, 100)] == ["/v/nvidia/cu13/lib/libcudart.so.13"]


def test_two_names_for_one_inode_are_one_backing_file():
    paths = ["/repo/build/cuda_devlink/libcudart.so.13", "/v/cu13/lib/libcudart.so.13"]
    files = R.backing_files(paths, lambda p: _st(1, 100))
    assert list(files) == [(1, 100)]
    assert len(files[(1, 100)]) == 2


def test_two_inodes_are_two_backing_files():
    paths = ["/a/libcudart.so.13", "/b/libcudart.so.13"]
    inodes = {"/a/libcudart.so.13": _st(1, 100), "/b/libcudart.so.13": _st(1, 101)}
    assert len(R.backing_files(paths, inodes.__getitem__)) == 2


def test_maps_paths_filters_by_basename_and_strips_deleted():
    text = MAPS + "\n7f09-7f0a r--p 0 08:20 400 /x/libcudart.so.13 (deleted)"
    paths = R.maps_paths(text, "libcudart.so")
    assert "/x/libcudart.so.13" in paths
    assert not any("static_helper" in p or "libcuda.so" in p for p in paths)
    assert R.maps_paths(MAPS, "libcuda.so") == ["/usr/lib/wsl/lib/libcuda.so.1"]


GOOD_DRIVER = {
    "nvidia_driver": R.NVIDIA_DRIVER,
    "cu_driver_version": R.CU_DRIVER_VERSION,
    "cudart_version": R.CUDART_VERSION,
    "libcudart_backing_files": [
        {"dev_ino": [1, 100], "relpath": R.CUDART_RELPATH, "sha256": R.CUDART_SHA256}
    ],
}


def test_driver_runtime_exact_match_passes():
    assert R.driver_runtime_problems(GOOD_DRIVER) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("nvidia_driver", "616.93"),
        ("nvidia_driver", None),
        ("nvidia_driver", ["616.92", "616.92"]),  # two GPUs: not a single string
        ("cu_driver_version", 13050),
        ("cu_driver_version", None),
        ("cudart_version", 13010),
        ("cudart_version", None),
        ("libcudart_backing_files", []),
        (
            "libcudart_backing_files",
            GOOD_DRIVER["libcudart_backing_files"] * 2,
        ),
        (
            "libcudart_backing_files",
            [
                {
                    "relpath": "build/cuda_devlink/libcudart.so.13",
                    "sha256": R.CUDART_SHA256,
                }
            ],
        ),
        (
            "libcudart_backing_files",
            [{"relpath": R.CUDART_RELPATH, "sha256": "0" * 64}],
        ),
    ],
)
def test_driver_runtime_any_difference_fails(key, value):
    obs = {**GOOD_DRIVER, key: value}
    assert R.driver_runtime_problems(obs)


def test_runtime_requirements_are_exact():
    assert R.runtime_requirement_problems(dict(R.RUNTIME_REQUIREMENTS)) == []
    for k in R.RUNTIME_REQUIREMENTS:
        flipped = {**R.RUNTIME_REQUIREMENTS, k: not R.RUNTIME_REQUIREMENTS[k]}
        assert R.runtime_requirement_problems(flipped)
    truthy = {**R.RUNTIME_REQUIREMENTS, "cudnn_allow_tf32": 1}
    assert R.runtime_requirement_problems(truthy)  # identity, not truthiness
    assert R.runtime_requirement_problems({})


GOOD_ARTIFACT = {
    "sha256": R.ARTIFACT_SHA256,
    "content_sha256": R.ARTIFACT_CONTENT_SHA256,
    "graph_problems": [],
    "native_scan_calls": 3,
    "param_devices": ["cuda:0"],
    "tensor_constant_devices": ["cpu"],
}


def test_artifact_problems():
    assert R.artifact_problems(GOOD_ARTIFACT) == []
    for key, value in [
        ("sha256", "0" * 64),
        ("content_sha256", "0" * 64),
        ("graph_problems", ["graph contains prim::PythonOp"]),
        ("native_scan_calls", 2),
        ("native_scan_calls", 4),
        ("param_devices", ["cuda:0", "cpu"]),
        ("param_devices", []),
        ("param_devices", None),
        ("tensor_constant_devices", ["cuda:0"]),
        ("tensor_constant_devices", ["cpu", "cuda:0"]),
        ("tensor_constant_devices", None),
    ]:
        assert R.artifact_problems({**GOOD_ARTIFACT, key: value}), key
    no_graph = {k: v for k, v in GOOD_ARTIFACT.items() if k != "graph_problems"}
    assert R.artifact_problems(no_graph)


def test_graph_problems():
    ok = f"%1 = {R.NATIVE_OP}(%a)\n%2 = {R.NATIVE_OP}(%b)"
    assert R.graph_problems(ok) == []
    assert R.graph_problems("aten::add") != []
    assert R.graph_problems(ok + "\nprim::PythonOp") != []
    assert R.graph_problems(ok + f"\n{R.PYTHON_OP}(%c)") != []


GOOD_A_L = {
    "arm": "A_L",
    "installed": list(R.A_L_INSTALLS),
    "runpy_completed": True,
    "op_library_mapped": True,
    "op_library_sha256": R.OP_LIBRARY_SHA256,
    "artifact": GOOD_ARTIFACT,
    "driver_runtime": GOOD_DRIVER,
    "build_wrapper_calls": 1,
    "build_preconditions": dict(R.BUILD_PRECONDITIONS),
    "slot_installed": True,
    "adapter_calls": 4,
    "adapter_problems": [],
    "guard_calls": {g: 0 for g in (*R.HEAD_GUARDS, *R.JIT_FALLBACK_GUARDS)},
}
GOOD_A_C = {
    "arm": "A_C",
    "installed": [],
    "runpy_completed": True,
    "op_library_mapped": False,
    "driver_runtime": GOOD_DRIVER,
}


def test_good_sidecars_pass():
    assert R.sidecar_problems("A_L", GOOD_A_L) == []
    assert R.sidecar_problems("A_C", GOOD_A_C) == []
    assert R.sidecar_problems("A_N", {**GOOD_A_C, "arm": "A_N"}) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("arm", "A_C"),
        ("installed", list(R.A_L_INSTALLS)[:-1]),
        ("installed", list(reversed(R.A_L_INSTALLS))),
        ("runpy_completed", False),
        ("op_library_mapped", False),
        ("op_library_sha256", "0" * 64),
        ("artifact", {**GOOD_ARTIFACT, "native_scan_calls": 0}),
        ("driver_runtime", {**GOOD_DRIVER, "nvidia_driver": "600.00"}),
        ("build_wrapper_calls", 0),
        ("build_wrapper_calls", 2),
        ("build_preconditions", {**R.BUILD_PRECONDITIONS, "trt_head_is_none": False}),
        (
            "build_preconditions",
            {**R.BUILD_PRECONDITIONS, "trt_head_engine": "x.engine"},
        ),
        ("build_preconditions", {**R.BUILD_PRECONDITIONS, "use_detail_fusion": True}),
        ("build_preconditions", None),
        ("slot_installed", False),
        ("adapter_calls", 0),
        ("adapter_calls", None),
        ("adapter_problems", ["call 1: inputs wrong"]),
        ("guard_calls", {**GOOD_A_L["guard_calls"], "_forward_eager": 1}),
        ("guard_calls", {**GOOD_A_L["guard_calls"], "forward": 3}),
        ("guard_calls", {**GOOD_A_L["guard_calls"], "_selective_scan_jit": 1}),
        ("guard_calls", {"forward": 0}),  # a guard that was never installed
    ],
)
def test_injected_sidecar_fails_closed(key, value):
    assert R.sidecar_problems("A_L", {**GOOD_A_L, key: value}), key


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("installed", ["mamba_gated_detector.build_mamba_gated_detector"]),
        ("installed", None),
        ("op_library_mapped", True),
        ("op_library_mapped", None),
        ("runpy_completed", False),
        ("driver_runtime", {}),
    ],
)
def test_non_injected_sidecar_fails_closed(key, value):
    assert R.sidecar_problems("A_C", {**GOOD_A_C, key: value}), key


def test_missing_sidecar_fails():
    assert R.sidecar_problems("A_C", None)
    assert R.sidecar_problems("A_L", None)


def test_argv_problems():
    argv = R.mot17_argv("A_L", "results/x/A_L_1", R.SEQUENCES, None)
    assert R.argv_problems(argv, argv) == []
    assert "--mamba-head-engine" not in argv and "--no-compile" not in argv
    assert R.argv_problems(argv + ["--mamba-head-engine", "h.engine"], argv)
    assert R.argv_problems(argv + ["--mamba-trt"], argv + ["--mamba-trt"])
    assert R.argv_problems(None, argv)
    n = R.mot17_argv("A_N", "o", R.SEQUENCES, None)
    assert n[-1] == "--no-compile"


def test_mot17_argv_matches_pr2_command():
    """Same mot17.py argv PR-2 launched directly (minus the interpreter)."""
    argv = R.mot17_argv("A_C", "OUT", R.SEQUENCES, None)
    assert argv == [
        "scripts/eval/mot17.py",
        "--preset",
        "mamba_whole_graph",
        "--detector",
        "SDP",
        "--double-buffer",
        "--sequences",
        ",".join(R.SEQUENCES),
        "--output",
        "OUT",
    ]


def test_oracle_anchor():
    assert R.oracle_anchor_problems(dict(R.ORACLE_ANCHOR_TXT_SHA256)) == []
    moved = {**R.ORACLE_ANCHOR_TXT_SHA256, "MOT17-10-SDP": "0" * 64}
    assert len(R.oracle_anchor_problems(moved)) == 1
    missing = dict(R.ORACLE_ANCHOR_TXT_SHA256)
    missing.pop("MOT17-04-SDP")
    assert len(R.oracle_anchor_problems(missing)) == 1
    assert list(R.ORACLE_ANCHOR_TXT_SHA256) == list(R.SEQUENCES)


GOOD_L1 = {
    "frames": 10,
    "v2": {
        "rerun_failures": [],
        "input_mutations": [],
        "jit_fallback_calls": {g: 0 for g in R.JIT_FALLBACK_GUARDS},
    },
    "driver_runtime": GOOD_DRIVER,
    "artifact": GOOD_ARTIFACT,
    "l_call_problems": [],
}


def test_l1_problems():
    assert R.l1_problems(GOOD_L1, 10) == []
    assert R.l1_problems(GOOD_L1, 11)
    for key, value in [
        ("rerun_failures", [{"sequence": "s", "frame_index": 0, "head": "L"}]),
        ("input_mutations", [{"sequence": "s", "frame_index": 3}]),
        (
            "jit_fallback_calls",
            {"_selective_scan_jit": 1, "_selective_scan_legacy_n1_jit": 0},
        ),
        ("jit_fallback_calls", {}),
        ("rerun_failures", None),
    ]:
        bad = copy.deepcopy(GOOD_L1)
        bad["v2"][key] = value
        assert R.l1_problems(bad, 10), key
    for key, value in [
        ("driver_runtime", {}),
        ("artifact", {}),
        ("l_call_problems", ["call 1: x"]),
    ]:
        assert R.l1_problems({**GOOD_L1, key: value}, 10), key


def test_guard_counts_and_raises():
    counts: dict[str, int] = {}
    g = R.Guard("forward", counts)
    assert counts == {"forward": 0}
    with pytest.raises(RuntimeError):
        g(1, x=2)
    assert counts == {"forward": 1}


class _T:
    """Tensor stand-in for the adapter's checks."""

    def __init__(self, shape, is_cuda=True, dtype="f32"):
        self.shape = shape
        self.is_cuda = is_cuda
        self.dtype = dtype


def _fake_torch(**readback):
    rb = {**R.RUNTIME_REQUIREMENTS, **readback}
    return SimpleNamespace(
        float32="f32",
        _C=SimpleNamespace(
            _get_graph_executor_optimize=lambda: rb["graph_executor_optimize"]
        ),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                benchmark=rb["cudnn_benchmark"], allow_tf32=rb["cudnn_allow_tf32"]
            ),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=rb["matmul_allow_tf32"])
            ),
        ),
    )


def _module(outs=None):
    return lambda *a: tuple(_T(s) for s in R.OUTPUT_SHAPES) if outs is None else outs


def test_adapter_passes_through_the_slot_interface():
    ad = R.LibTorchHeadAdapter(_module(), _fake_torch())
    cls, reg = ad.infer_graph(*(_T(s) for s in R.INPUT_SHAPES))
    assert [c.shape for c in cls] == list(R.OUTPUT_SHAPES[:3])
    assert [r.shape for r in reg] == list(R.OUTPUT_SHAPES[3:])
    assert ad.calls == 1 and ad.problems == []
    assert R.LibTorchHeadAdapter.infer is R.LibTorchHeadAdapter.infer_graph


@pytest.mark.parametrize(
    ("torch_kw", "inputs", "outs"),
    [
        ({"graph_executor_optimize": True}, None, None),
        ({"cudnn_benchmark": True}, None, None),
        ({"matmul_allow_tf32": True}, None, None),
        (
            {},
            [
                _T((1, 128, 80, 80), is_cuda=False),
                _T((1, 256, 40, 40)),
                _T((1, 512, 20, 20)),
            ],
            None,
        ),
        ({}, [_T((2, 128, 80, 80)), _T((2, 256, 40, 40)), _T((2, 512, 20, 20))], None),
        (
            {},
            [
                _T((1, 128, 80, 80), dtype="f16"),
                _T((1, 256, 40, 40)),
                _T((1, 512, 20, 20)),
            ],
            None,
        ),
        ({}, None, tuple(_T(s) for s in R.OUTPUT_SHAPES[:5])),
        ({}, None, tuple(_T(s, dtype="f16") for s in R.OUTPUT_SHAPES)),
        ({}, None, [_T(s) for s in R.OUTPUT_SHAPES]),  # a list, not the traced tuple
    ],
)
def test_adapter_fails_closed(torch_kw, inputs, outs):
    ad = R.LibTorchHeadAdapter(_module(outs), _fake_torch(**torch_kw))
    feats = inputs or [_T(s) for s in R.INPUT_SHAPES]
    with pytest.raises(RuntimeError):
        ad.infer_graph(*feats)
    assert ad.problems


def test_a_l_install_list_follows_the_declared_steps():
    assert R.A_L_INSTALLS[:4] == (
        "op_library",
        "graph_executor_optimize=false",
        "torchscript_artifact",
        "mamba_gated_detector.build_mamba_gated_detector",
    )
    assert R.HEAD_GUARDS == ("forward", "_forward_eager")
    assert R.BUILD_PRECONDITIONS == {
        "trt_head_engine": "",
        "trt_head_is_none": True,
        "use_whole_graph": True,
        "use_detail_fusion": False,
    }
    assert R.INJECTED_ARMS == ("A_L",)


def test_injection_uses_only_the_trt_head_slot():
    """The child assigns L to ``_trt_head`` and touches no other detector path."""
    child = RUNNER_SRC["run_arm_child"]
    assert "det._trt_head = adapter" in child
    for forbidden in ("detect_raw", "run_eval", "_whole_graph_fn", "_postprocess"):
        assert forbidden not in child, forbidden


def test_guarded_names_exist_in_the_harness():
    mh = (REPO / "src/saccade/perception/temporal_yolo/mamba_head.py").read_text()
    for name in R.JIT_FALLBACK_GUARDS:
        assert f"def {name}(" in mh
    assert "def _forward_eager(" in mh
    mgd = (
        REPO / "src/saccade/perception/temporal_yolo/mamba_gated_detector.py"
    ).read_text()
    assert "self._trt_head.infer_graph(p3, p4, p5)" in mgd
    assert "def build_mamba_gated_detector(" in mgd


# --- run_problems: fail-fast validation of one L2 run --------------------


def _write_run(tmp: Path, arm: str, rep: int, txt: dict, sidecar: dict, head: str):
    out = tmp / f"{arm}_{rep}"
    out.mkdir()
    argv = R.mot17_argv(arm, f"{arm}_{rep}", R.SEQUENCES, None)
    (out / "run_manifest.json").write_text(
        json.dumps({"cmdline": argv, "commit": head, "dirty": False})
    )
    (tmp / f"{arm}_{rep}.sidecar.json").write_text(json.dumps(sidecar))
    return {
        "arm": arm,
        "rep": rep,
        "returncode": 0,
        "mot17_argv": argv,
        "output_dir": f"{arm}_{rep}",
        "sidecar": f"{arm}_{rep}.sidecar.json",
        "resolved_env_overrides": {},
        "txt_sha256": txt,
    }


def test_run_problems_checks_v4_v5_v3(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "project_root", tmp_path)
    anchor = dict(R.ORACLE_ANCHOR_TXT_SHA256)
    c1 = _write_run(tmp_path, "A_C", 1, anchor, GOOD_A_C, "H")
    assert R.run_problems(c1, "H", [c1]) == []
    other = {s: "1" * 64 for s in R.SEQUENCES}
    l1 = _write_run(tmp_path, "A_L", 1, other, GOOD_A_L, "H")
    assert R.run_problems(l1, "H", [c1, l1]) == []  # V4 applies to A_C#1 only
    l2 = _write_run(
        tmp_path, "A_L", 2, {**other, "MOT17-02-SDP": "2" * 64}, GOOD_A_L, "H"
    )
    assert any("V3" in p for p in R.run_problems(l2, "H", [c1, l1, l2]))
    c1b = _write_run(tmp_path, "A_C", 2, anchor, GOOD_A_C, "H")
    assert R.run_problems(c1b, "H", [c1, l1, c1b]) == []


def test_run_problems_v4_fails_on_a_moved_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "project_root", tmp_path)
    moved = {**R.ORACLE_ANCHOR_TXT_SHA256, "MOT17-13-SDP": "0" * 64}
    c1 = _write_run(tmp_path, "A_C", 1, moved, GOOD_A_C, "H")
    assert any(p.startswith("V4") for p in R.run_problems(c1, "H", [c1]))


def test_run_problems_v5_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "project_root", tmp_path)
    anchor = dict(R.ORACLE_ANCHOR_TXT_SHA256)
    bad = {**GOOD_A_L, "guard_calls": {**GOOD_A_L["guard_calls"], "forward": 1}}
    run = _write_run(tmp_path, "A_L", 1, anchor, bad, "H")
    assert any(p.startswith("V5") for p in R.run_problems(run, "H", [run]))
    good = _write_run(tmp_path, "A_N", 1, anchor, {**GOOD_A_C, "arm": "A_N"}, "H")
    assert any("commit" in p for p in R.run_problems(good, "OTHER", [good]))
    failed = {**good, "returncode": 1}
    assert len(R.run_problems(failed, "H", [failed])) == 1
    (tmp_path / good["sidecar"]).unlink()
    assert any("sidecar missing" in p for p in R.run_problems(good, "H", [good]))


# --- (d) command line ------------------------------------------------------


def test_cli_exposes_no_study_identity_option():
    seen: list[str] = []
    real = argparse.ArgumentParser.add_argument

    def spy(self, *names, **kwargs):
        seen.extend(n for n in names if n.startswith("--"))
        return real(self, *names, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "add_argument", spy):
        with mock.patch("sys.argv", ["x", "--help"]), pytest.raises(SystemExit):
            R.main()
    assert set(seen) - {"--help"} == {"--_l1-worker", "--_arm-child", "--_out"}


def test_no_smoke_mode():
    assert "--smoke-frames" not in (DIAG / "native_head_parity_libtorch.py").read_text()


# --- §11 A1: load without map_location ------------------------------------


def test_a1_is_declared_and_implemented():
    assert "`torch.jit.load(<§2 artifact>)`（不帶 `map_location`）" in DECL_TEXT
    src = (DIAG / "native_head_parity_libtorch.py").read_text()
    assert "map_location=" not in RUNNER_SRC["load_l"]
    assert src.count("torch.jit.load(") == 1
    assert "torch.jit.load(str(path)).eval()" in RUNNER_SRC["load_l"]
    assert "inspect_loaded(module)" in RUNNER_SRC["load_l"]


def test_placement_problems():
    assert R.placement_problems(["cuda:0"], ["cpu"]) == []
    assert R.placement_problems(["cuda:0"], []) == []  # no tensor constant at all
    assert R.placement_problems(["cuda:0"], ["cuda:0"])  # the A1 failure mode
    assert R.placement_problems(["cuda:1"], ["cpu"])
    assert R.placement_problems([], ["cpu"])
    assert R.placement_problems(None, None)


def test_v1_loads_the_artifact_for_the_graph_row():
    assert "load_l(torch)" in RUNNER_SRC["check_v1"]
