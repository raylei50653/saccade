"""The native extension is delivered the way ADR 025 says, and the metadata agrees.

ADR 025 settles how ``saccade_tracking_ext`` reaches a third party: the
consumer builds it from a source checkout, against the venv it will be loaded
in, and registers the build directory with that venv. ``pip install saccade``
alone yields the Python package and its load-time dependencies, not a usable
native tracker. This suite holds the pieces that have to agree for that model
to be the one a consumer actually gets:

  * **the dependency metadata matches the model** -- ``tensorrt-cu12`` is a
    default dependency because the built extension needs ``libnvinfer.so.10``
    at load; the compiler line (``nvidia-cuda-nvcc``/``nvvm``/``crt``/``cccl``)
    and ``pybind11`` are build-only and live in the ``native-build`` extra,
    nowhere else;
  * **the classification is evidence-bound** -- when a built extension is at
    hand, its ``NEEDED`` entries name the TensorRT runtime library and name
    nothing the build-only extra provides;
  * **the build targets the venv it is told to** -- ``CMakeLists.txt`` honours
    ``-DPYTHON_EXECUTABLE`` and derives its FindPython root from that
    interpreter; the only ``.venv`` it knows about is the checkout fallback
    for the repository workflow;
  * **the install path is written down and runnable** -- the runbook, the ADR
    and the consumer verification script exist and name the same extra, the
    same registration mechanisms and the same build target;
  * **discovery has no checkout in it** -- ``saccade.paths.build_dir`` is the
    only fallback the tracker uses, and it already answers ``None`` outside a
    checkout (``test_package_runtime_paths.py``); a ``.pth`` or
    ``SACCADE_BUILD_PATH`` is what a consumer supplies.

The end-to-end consumer test itself (fresh venv, non-editable install, cmake
build, smoke from outside the checkout) is
``scripts/native/verify_consumer_install.sh``; it takes minutes and a GPU, so
it is run and recorded per ADR 025 rather than collected here.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CMAKELISTS = REPO_ROOT / "CMakeLists.txt"
ADR = REPO_ROOT / "docs" / "decisions" / "025-native-extension-delivery.md"
RUNBOOK = REPO_ROOT / "docs" / "reference" / "runbooks" / "native_extension_install.md"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "native" / "verify_consumer_install.sh"

BUILD_EXTRA = "native-build"
BUILD_ONLY = {
    "nvidia-cuda-nvcc",
    "nvidia-nvvm",
    "nvidia-cuda-crt",
    "nvidia-cuda-cccl",
    "pybind11",
}
RUNTIME_LOADER = {"tensorrt-cu12"}
# SONAMEs the build-only distributions ship; none may be NEEDED by the extension.
BUILD_ONLY_SONAMES = re.compile(r"^lib(nvvm|nvptxcompiler|nvJitLink|cudadevrt)\b")

# The registration mechanisms and build target the consumer path is made of.
EXTENSION = "saccade_tracking_ext"
PTH_NAME = "saccade_build.pth"
ENV_NAME = "SACCADE_BUILD_PATH"
PYTHON_FLAG = "-DPYTHON_EXECUTABLE"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(specs: list[str]) -> dict[str, str]:
    """name -> version spec (empty when unpinned)."""
    out = {}
    for spec in specs:
        m = re.match(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*(.*)", spec.strip())
        assert m, spec
        out[_normalize(m.group(1))] = m.group(2).strip()
    return out


def _pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


# ── The dependency metadata matches the delivery model ───────────────────────


def test_tensorrt_is_a_default_dependency_and_the_toolchain_is_not() -> None:
    project = _pyproject()["project"]
    default = _requirements(project["dependencies"])
    assert RUNTIME_LOADER <= set(default), sorted(RUNTIME_LOADER - set(default))
    assert not BUILD_ONLY & set(default), sorted(BUILD_ONLY & set(default))


def test_native_build_extra_is_exactly_the_build_toolchain() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert BUILD_EXTRA in extras, sorted(extras)
    declared = _requirements(extras[BUILD_EXTRA])
    assert set(declared) == BUILD_ONLY, sorted(set(declared) ^ BUILD_ONLY)
    # nvcc / nvvm / crt are one matched line (issue #214); a version that
    # drifts on one of them is a different toolchain, not a bump.
    line = {declared[n] for n in ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt")}
    assert len(line) == 1 and next(iter(line)).startswith("=="), sorted(line)
    for other, reqs in extras.items():
        if other == BUILD_EXTRA:
            continue
        assert not BUILD_ONLY & set(_requirements(reqs)), (
            f"build toolchain declared again under extra {other!r}"
        )


def test_dev_group_installs_the_build_toolchain() -> None:
    """`uv sync` must still be able to build the extension in the checkout."""
    dev = _pyproject()["dependency-groups"]["dev"]
    selfref = [s for s in dev if isinstance(s, str) and s.startswith("saccade[")]
    assert len(selfref) == 1, dev
    extras = {e.strip() for e in selfref[0][len("saccade[") : -1].split(",")}
    assert BUILD_EXTRA in extras, sorted(extras)


# ── The classification is evidence-bound ─────────────────────────────────────


def _built_extension() -> Path | None:
    candidates = []
    explicit = os.environ.get(ENV_NAME, "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(REPO_ROOT / "build")
    for build_dir in candidates:
        hits = sorted(build_dir.glob(f"{EXTENSION}*.so"))
        if hits:
            return hits[0]
    return None


def _needed(so: Path) -> list[str]:
    out = subprocess.run(
        ["readelf", "-d", str(so)], capture_output=True, text=True, check=True
    ).stdout
    return [
        line.split("[", 1)[1].rstrip("]").strip()
        for line in out.splitlines()
        if "(NEEDED)" in line
    ]


def test_built_extension_needs_tensorrt_and_nothing_build_only() -> None:
    so = _built_extension()
    if so is None:
        pytest.skip("no built saccade_tracking_ext in build/ or $SACCADE_BUILD_PATH")
    if shutil.which("readelf") is None:
        pytest.skip("readelf (binutils) not available")
    needed = _needed(so)
    assert "libnvinfer.so.10" in needed, needed
    assert not [n for n in needed if BUILD_ONLY_SONAMES.match(n)], needed
    # torch's libraries are what the default `torch` pin provides.
    assert {"libtorch.so", "libc10.so", "libcudart.so.13"} <= set(needed), needed


# ── The build targets the venv it is told to ─────────────────────────────────


def test_cmake_takes_the_target_interpreter_as_input() -> None:
    text = CMAKELISTS.read_text()
    assert "if(DEFINED PYTHON_EXECUTABLE" in text
    # FindPython's root hint follows the interpreter, not the checkout.
    assert 'set(Python_ROOT_DIR "${_saccade_python_prefix}")' in text
    assert "PROJECT_SOURCE_DIR}/.venv" not in text
    venv_refs = [
        line.strip()
        for line in text.splitlines()
        if ".venv" in line and not line.strip().startswith("#")
    ]
    assert venv_refs == [
        'set(_repo_venv_python "${CMAKE_SOURCE_DIR}/.venv/bin/python3")'
    ], venv_refs
    # The toolchain comes from the target venv (the native-build extra), never
    # from the checkout or the system.
    assert 'set(SACCADE_CUDA_VENV_ROOT "${_saccade_purelib}/nvidia/cu13")' in text
    assert BUILD_EXTRA in text


# ── The install path is written down and runnable ────────────────────────────


def test_runbook_names_the_real_mechanisms() -> None:
    text = RUNBOOK.read_text()
    for token in (
        f"saccade[{BUILD_EXTRA}]",
        PTH_NAME,
        ENV_NAME,
        PYTHON_FLAG,
        f"--target {EXTENSION}",
        VERIFY_SCRIPT.relative_to(REPO_ROOT).as_posix(),
        "ADR 025",
    ):
        assert token in text, token


def test_adr_is_accepted_and_names_the_model() -> None:
    text = ADR.read_text()
    assert "<!-- doc-status: accepted -->" in text.splitlines()[0]
    for token in (
        BUILD_EXTRA,
        "tensorrt-cu12",
        RUNBOOK.name,
        PTH_NAME,
        ENV_NAME,
    ):
        assert token in text, token


def test_consumer_verification_script_is_executable_and_follows_the_runbook() -> None:
    assert VERIFY_SCRIPT.is_file()
    assert os.access(VERIFY_SCRIPT, os.X_OK), (
        "verify_consumer_install.sh must be executable"
    )
    text = VERIFY_SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash\n# status: ")
    for token in (
        f"saccade[{BUILD_EXTRA}]",
        PTH_NAME,
        ENV_NAME,
        PYTHON_FLAG,
        f"--target {EXTENSION}",
        "source_checkout_root() is None",
    ):
        assert token in text, token
    subprocess.run(["bash", "-n", str(VERIFY_SCRIPT)], check=True)


def test_readme_points_at_the_runbook() -> None:
    text = (REPO_ROOT / "README.md").read_text()
    assert "docs/reference/runbooks/native_extension_install.md" in text
