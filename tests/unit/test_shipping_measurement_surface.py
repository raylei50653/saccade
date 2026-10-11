"""The shipping entrypoint carries no measurement surface (#465 Phase C PR-C2).

docs/reference/native_runtime_resolved_config.md §18. The measurement hooks
(the negative-control mutations and the ``*_for_measurement`` setters) exist
only in the ``_measurement`` variant of the runtime libraries, compiled with
``SACCADE_SHIPPING_MEASUREMENT_HOOKS``, and the developer options only in
``saccade_track_measurement``. The built binary is checked against
``shipping/measurement_surface.json`` (POST_BUILD and ``check_shipping_bundle.py
static``); these tests keep that list and the source boundary honest without a
build:

* every mutation name the sources define, and every option
  ``saccade_track_measurement`` adds, is covered by a forbidden byte string;
* every source line that names a mutation, a measurement setter or a hook
  member sits in the ``#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS`` branch;
* ``saccade_track.cpp`` refuses to compile with the hooks, the measurement
  main refuses to compile without them, and the shipping options are the same
  seven in the entrypoint, the shared driver and the harness;
* CMake: ``saccade_track`` links the shipping runtime, no ``_measurement``
  target is installed, and the POST_BUILD step runs the surface check.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SHIPPING = REPO / "shipping"
FORBIDDEN = json.loads((SHIPPING / "measurement_surface.json").read_text())["forbidden"]
HOOKS = "SACCADE_SHIPPING_MEASUREMENT_HOOKS"
# Names that exist only to measure: the mutation enums / values / names, the
# setters, and the members they set.
_HOOK_NAME = re.compile(
    r"(?<!Per)(?<!per)[Mm]utation|_for_measurement|force_decoupled_|stale_gmc_input_|"
    r"stale_graph_input|shared_post_|shared_geometry_|next_ulp_inplace|\bprevious\b"
)
SHIPPING_OPTIONS = {
    "--config",
    "--lineage",
    "--attestation",
    "--model-root",
    "--model-bundle",  # #549 S2-1 manifest mode
    "--require-identity",
    "--out",
    "--report",
    "--trace",
}


def _covered(text: str) -> bool:
    return any(t in text for t in FORBIDDEN)


def test_every_mutation_name_is_forbidden() -> None:
    names = set()
    for src in ("detector_host.cpp", "serial_runtime.cpp", "double_buffer_runtime.cpp"):
        names |= set(
            re.findall(
                r'case \w*Mutation::\w+: return "(\w+)";',
                (SHIPPING / "src" / src).read_text(),
            )
        )
    names.discard("none")
    assert len(names) == 12  # detector 6, serial 3, double buffer 3
    assert {n for n in names if not _covered(n)} == set()


def _options(path: Path) -> set[str]:
    return set(re.findall(r'a == "(--[\w-]+)"', path.read_text()))


def test_developer_options_are_forbidden_and_shipping_options_agree() -> None:
    driver = _options(SHIPPING / "tools" / "track_driver.hpp")
    assert driver == SHIPPING_OPTIONS
    assert _options(SHIPPING / "tools" / "saccade_track.cpp") == set()
    extra = _options(SHIPPING / "tools" / "saccade_track_measurement.cpp")
    assert extra == {"--measurement-mutation", "--schedule", "--max-frames"}
    assert all(_covered(o) for o in extra)
    assert not any(_covered(o) for o in SHIPPING_OPTIONS)
    usage = re.search(
        r'kUsage =\s*((?:"[^"]*"\s*)+);',
        (SHIPPING / "tools/saccade_track.cpp").read_text(),
    )
    assert usage and not _covered(usage.group(1))
    assert set(re.findall(r"--[\w-]+", usage.group(1))) == SHIPPING_OPTIONS


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0]


def hook_lines_outside(path: Path) -> list[str]:
    """Lines naming a hook that are not in the #ifdef HOOKS branch."""
    stack: list[bool] = []  # per open conditional: is this branch the hooks branch
    bad = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if line.startswith("#if"):
            stack.append(line in (f"#ifdef {HOOKS}", f"#if defined({HOOKS})"))
            continue
        if line.startswith("#else") or line.startswith("#elif"):
            stack[-1] = False
            continue
        if line.startswith("#endif"):
            stack.pop()
            continue
        if line.startswith("#error"):
            continue
        if _HOOK_NAME.search(_strip_comment(raw)) and not any(stack):
            bad.append(f"{path.name}:{n}: {raw.strip()}")
    assert not stack, f"{path}: unbalanced #if"
    return bad


@pytest.mark.parametrize(
    "path",
    sorted((SHIPPING / "include" / "saccade_shipping").glob("*.hpp"))
    + sorted((SHIPPING / "src").glob("*.cpp"))
    + sorted((SHIPPING / "src").glob("*.cu"))
    + [
        SHIPPING / "tools" / "track_driver.hpp",
        SHIPPING / "tools" / "saccade_track.cpp",
    ],
    ids=lambda p: p.name,
)
def test_hooks_only_in_the_measurement_branch(path: Path) -> None:
    assert hook_lines_outside(path) == []


def test_the_branch_tracker_sees_a_hook_outside(tmp_path: Path) -> None:
    f = tmp_path / "x.cpp"
    f.write_text(
        f"#ifdef {HOOKS}\nvoid set_x_for_measurement();\n#else\nint mutation_;\n#endif\n"
        "// a comment about a mutation\nbool stale_gmc_input_ = false;\n"
    )
    bad = hook_lines_outside(f)
    assert [b.split(": ", 1)[1] for b in bad] == [
        "int mutation_;",
        "bool stale_gmc_input_ = false;",
    ]


def test_entrypoints_guard_their_variant() -> None:
    ship = (SHIPPING / "tools" / "saccade_track.cpp").read_text()
    meas = (SHIPPING / "tools" / "saccade_track_measurement.cpp").read_text()
    assert re.search(rf"#ifdef {HOOKS}\n#error", ship)
    assert re.search(rf"#ifndef {HOOKS}\n#error", meas)
    # The shared driver compiles the same in both binaries.
    assert not re.search(
        rf"^#.*{HOOKS}", (SHIPPING / "tools" / "track_driver.hpp").read_text(), re.M
    )


def test_cmake_boundary() -> None:
    cm = (SHIPPING / "CMakeLists.txt").read_text()
    assert "target_link_libraries(saccade_track PRIVATE saccade_shipping_runtime)" in cm
    assert (
        "target_link_libraries(saccade_track_measurement PRIVATE "
        "saccade_shipping_runtime_measurement)" in cm
    )
    installs = re.findall(r"install\(TARGETS ([^\n]+)", cm)
    assert installs and not any("_measurement" in i for i in installs)
    assert "saccade_track_measurement" not in "".join(
        re.findall(r"install\([^)]*\)", cm)
    )
    assert f"target_compile_definitions(${{name}}_measurement PUBLIC {HOOKS}=1)" in cm
    assert "cmake/check_no_measurement_surface.cmake" in cm
    # Every runtime library with a hook exists in both variants.
    for lib in ("native", "ingest", "detector", "runtime"):
        assert f"saccade_shipping_variants(saccade_shipping_{lib} " in cm
