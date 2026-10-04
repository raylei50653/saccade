"""The shipping link surface: tracking without perception, no OpenCV (#465 PR-11).

Boundary §6 PR-11: ``saccade_tracking`` no longer links ``saccade_perception``
and OpenCV becomes optional, so that ``saccade_track`` needs neither. The build
enforces this twice -- a configure-time check on ``saccade_tracking``'s link
libraries (root ``CMakeLists.txt``) and a POST_BUILD ``DT_NEEDED`` check on
``saccade_track`` (``shipping/cmake/check_link_surface.cmake``) -- but both need
a CUDA build. These run in the ordinary pytest job:

* the sources of ``saccade_tracking`` and ``saccade_perception`` (read from the
  root ``CMakeLists.txt``), the shipping sources and every repository header
  they reach include no OpenCV header, and the tracking ones no perception
  header;
* the POST_BUILD check script rejects a binary whose NEEDED holds OpenCV,
  libpython or libtorch_python, and one with no NEEDED at all (driven by a fake
  ``readelf``; skips without ``cmake``).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CMAKE_LISTS = REPO / "CMakeLists.txt"
CHECK_SCRIPT = REPO / "shipping" / "cmake" / "check_link_surface.cmake"
INCLUDE_DIRS = (REPO / "include", REPO / "shipping" / "include")

_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
_ANGLE_INCLUDE = re.compile(r"^\s*#\s*include\s+<([^>]+)>", re.M)


def _library_sources(target: str) -> list[Path]:
    text = CMAKE_LISTS.read_text(encoding="utf-8")
    m = re.search(rf"add_library\({target}\s+STATIC\s+(.*?)\)", text, re.S)
    assert m, f"add_library({target} STATIC ...) not found in CMakeLists.txt"
    sources = [REPO / s for s in m.group(1).split()]
    assert sources and all(s.is_file() for s in sources), sources
    return sources


def _shipping_sources() -> list[Path]:
    root = REPO / "shipping"
    return sorted(
        p
        for sub in ("src", "tools")
        for p in (root / sub).iterdir()
        if p.suffix in {".cpp", ".cu"}
    )


def _resolve(name: str, including: Path) -> Path | None:
    for base in (including.parent, *INCLUDE_DIRS):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None  # system / third-party header


def _reachable(sources: list[Path]) -> dict[Path, str]:
    """Every source and repository header reachable from *sources* -> its text."""
    seen: dict[Path, str] = {}
    stack = list(sources)
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        seen[path] = text
        for name in _QUOTED_INCLUDE.findall(text):
            header = _resolve(name, path)
            if header is not None:
                stack.append(header)
    return seen


def _offenders(files: dict[Path, str], forbidden: re.Pattern[str]) -> list[str]:
    out = []
    for path, text in files.items():
        for name in _QUOTED_INCLUDE.findall(text) + _ANGLE_INCLUDE.findall(text):
            if forbidden.search(name):
                out.append(f"{path.relative_to(REPO)} includes {name}")
    return sorted(out)


def test_tracking_reaches_neither_perception_nor_opencv() -> None:
    files = _reachable(_library_sources("saccade_tracking"))
    assert REPO / "include" / "tracking" / "pipeline.hpp" in files  # the scan is live
    assert _offenders(files, re.compile(r"^(perception/|opencv2/)")) == []


def test_perception_reaches_no_opencv() -> None:
    files = _reachable(_library_sources("saccade_perception"))
    assert REPO / "include" / "perception" / "preprocessor.hpp" in files
    assert _offenders(files, re.compile(r"^opencv2/")) == []


def test_shipping_reaches_no_opencv() -> None:
    files = _reachable(_shipping_sources())
    assert REPO / "include" / "tracking" / "gmc.hpp" in files
    assert _offenders(files, re.compile(r"^opencv2/")) == []


def test_the_opencv_paths_moved_out_of_the_core_libraries() -> None:
    tracking = {p.name for p in _library_sources("saccade_tracking")}
    perception = {p.name for p in _library_sources("saccade_perception")}
    assert {"gmc_cpu.cpp", "seq_runner.cpp", "eval_pool.cpp"}.isdisjoint(tracking)
    assert "preprocessor_cpu.cpp" not in perception
    assert {p.name for p in _library_sources("saccade_opencv_host")} == {
        "preprocessor_cpu.cpp",
        "gmc_cpu.cpp",
    }


_NEEDED_CLEAN = [
    "libtorch.so",
    "libnvinfer.so.10",
    "libcudart.so.13",
    "libstdc++.so.6",
    "libc.so.6",
]


def _fake_readelf(tmp_path: Path, needed: list[str]) -> Path:
    lines = [
        f" 0x0000000000000001 (NEEDED)             Shared library: [{lib}]"
        for lib in needed
    ]
    lines.append(
        " 0x000000000000001d (RUNPATH)            Library runpath: [/opt/opencv/lib]"
    )
    script = tmp_path / "readelf"
    script.write_text("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(lines) + "\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _run_check(tmp_path: Path, needed: list[str]) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")
    return subprocess.run(
        [
            cmake,
            f"-DREADELF={_fake_readelf(tmp_path, needed)}",
            f"-DBINARY={tmp_path / 'saccade_track'}",
            "-P",
            str(CHECK_SCRIPT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_link_check_accepts_a_clean_entrypoint(tmp_path: Path) -> None:
    # The RUNPATH line mentions opencv; only NEEDED entries count.
    result = _run_check(tmp_path, _NEEDED_CLEAN)
    assert result.returncode == 0, result.stderr
    assert "link surface OK" in result.stdout


@pytest.mark.parametrize(
    "lib",
    ["libopencv_core.so.500", "libpython3.12.so.1.0", "libtorch_python.so"],
)
def test_link_check_rejects_opencv_and_python(tmp_path: Path, lib: str) -> None:
    result = _run_check(tmp_path, _NEEDED_CLEAN + [lib])
    assert result.returncode != 0
    assert lib in result.stderr


def test_link_check_rejects_a_binary_without_needed(tmp_path: Path) -> None:
    result = _run_check(tmp_path, [])
    assert result.returncode != 0
    assert "no DT_NEEDED" in result.stderr
