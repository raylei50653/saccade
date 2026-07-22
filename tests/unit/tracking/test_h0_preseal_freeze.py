"""Amendment 6 classifier/admission semantics for the H0 pre-seal freeze.

Only the pure decision logic is pinned here — the git-walking assembly is
exercised for real when the freeze artifact is produced. What must never
drift silently is the classifier's fail-closed default and the admitted
runtime surface, because a wrong `non_runtime` classification would hide a
runtime-policy change from the projection admission.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "tools"))

from build_h0_preseal_freeze import (  # noqa: E402
    ADMITTED_RUNTIME_BLOBS,
    ADMITTED_RUNTIME_PATHS,
    admission_verdict,
    classify_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "src/tracking/tracker_gpu.cu",
        "include/tracking/tracker_gpu.hpp",
        "configs/presets/mamba_whole_graph_m.yaml",
        "cmake/toolchain.cmake",
        "pyproject.toml",
        "uv.lock",
    ],
)
def test_runtime_surface_is_runtime(path: str) -> None:
    assert classify_path(path) == "runtime_build_consumable"


@pytest.mark.parametrize(
    "path",
    [
        "docs/research/contracts/claim_state_registry.md",
        ".github/workflows/ci.yml",
        "tests/unit/tracking/test_gpu_bidir_bridge.py",
        "scripts/tools/check_h0_bridge_decision_trace_contract.py",
        "scripts/docs/build_master_map.py",
    ],
)
def test_non_runtime_paths_are_recorded_not_admission_relevant(path: str) -> None:
    assert classify_path(path) == "non_runtime_recorded"


@pytest.mark.parametrize(
    "path",
    [
        "weird/binary.bin",
        "third_party/vendored.c",
        "README.md",  # root file, not a declared build manifest — stays fail-closed
        "srcs/decoy.py",  # prefix near-miss must not match src/
        "documentation/notes.md",  # prefix near-miss must not match docs/
    ],
)
def test_unmatched_paths_fail_closed_to_runtime(path: str) -> None:
    assert classify_path(path) == "runtime_build_consumable"


def test_admitted_runtime_surface_admits_exactly_the_h0_sources() -> None:
    classified = {path: "runtime_build_consumable" for path in ADMITTED_RUNTIME_PATHS}
    classified["docs/anything.md"] = "non_runtime_recorded"
    admitted, violations = admission_verdict(classified, {})
    assert admitted and violations == []


def test_any_other_runtime_path_blocks_admission() -> None:
    classified = {
        "src/tracking/tracker_gpu.cu": "runtime_build_consumable",
        "src/saccade/perception/eval/consumer_a_bridge_fidelity.py": "runtime_build_consumable",
    }
    admitted, violations = admission_verdict(classified, {})
    assert not admitted
    assert violations == ["src/saccade/perception/eval/consumer_a_bridge_fidelity.py"]


def test_admitted_set_is_exactly_the_declared_five() -> None:
    assert ADMITTED_RUNTIME_PATHS == {
        "include/tracking/tracker_gpu.hpp",
        "src/tracking/tracker_gpu.cu",
        "src/tracking/tracker_gpu_python.cpp",
        "src/saccade/perception/tracking/tracker_gpu.py",
        "src/saccade/perception/eval/stages.py",
    }


def test_development_md_is_the_sole_non_runtime_root_exception() -> None:
    # Amendment 6 Correction 1(b): documentation-only root file; every other
    # root file stays fail-closed (see test_unmatched_paths_fail_closed_to_runtime).
    assert classify_path("DEVELOPMENT.md") == "non_runtime_recorded"


def test_content_pinned_admission_is_exactly_the_declared_four() -> None:
    # Amendment 6 Correction 1(a): the frozen-CUDA-substrate contents
    # (#214/#216), admitted only at these exact after-blob SHA-256 values.
    assert ADMITTED_RUNTIME_BLOBS == {
        "CMakeLists.txt": (
            "3d5a576d632109255c2f0537fbd9b302b66d69a61ee90759301ae4da3960890e"
        ),
        "pyproject.toml": (
            "85ec43e498adfdf0d433b6a8a621055a5b5232999ec8366e2616125d2cc8627b"
        ),
        "uv.lock": ("c45f37916351fd246eb54bb8bac7fd28df7816180b551c50574c189d85687923"),
        "src/perception/preprocessor.cpp": (
            "11aa959b94efc49e4bb4544beb1462757c229c2f5175980540636f61c1a98313"
        ),
    }


def test_content_pinned_path_admits_only_the_exact_blob() -> None:
    classified = {"CMakeLists.txt": "runtime_build_consumable"}
    pinned = ADMITTED_RUNTIME_BLOBS["CMakeLists.txt"]
    admitted, violations = admission_verdict(classified, {"CMakeLists.txt": pinned})
    assert admitted and violations == []
    admitted, violations = admission_verdict(classified, {"CMakeLists.txt": "0" * 64})
    assert not admitted and violations == ["CMakeLists.txt"]
    admitted, violations = admission_verdict(classified, {})
    assert not admitted and violations == ["CMakeLists.txt"]
    # An absent after side (deleted file) never matches a pin.
    admitted, violations = admission_verdict(classified, {"CMakeLists.txt": None})
    assert not admitted and violations == ["CMakeLists.txt"]
