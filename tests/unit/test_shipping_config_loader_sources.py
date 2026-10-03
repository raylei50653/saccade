"""Source-level guards for the native shipping config (#465 PR-4a, PR-4b).

The loader's behaviour is pinned by ``tests/native/test_resolved_config.cpp``
and the JSON -> native mapping/readback by
``tests/native/test_shipping_native_config.cpp`` (CI job
``shipping-config-loader``) and ``test_shipping_native_build.cpp`` (GPU). These
checks run in the ordinary pytest job and pin properties that hold by
construction:

* the compiled ``host_params.cfg`` field list is fresh against the committed
  resolved JSON (so a re-export cannot silently diverge from the native schema);
* nothing under ``shipping/`` reads the process environment, so env cannot
  change a load result;
* ``shipping/`` reaches native code only through the CUDA-free parameter
  headers, except the GPU builder and the post-detector host / replay tool
  (PR-5), and never through the legacy env resolver;
* the native tracker, GMC and pipeline read no ``SACCADE_*`` variable: only the
  legacy front-ends (``legacy_env.cpp``, the pybind binding) do.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RENDER = REPO / "scripts" / "model" / "render_shipping_host_cfg_schema.py"
SHIPPING = REPO / "shipping"


def _render_module():
    spec = importlib.util.spec_from_file_location(
        "render_shipping_host_cfg_schema", RENDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_host_cfg_field_list_is_fresh() -> None:
    render = _render_module()
    assert render.OUTPUT.read_text(encoding="utf-8") == render.render(), (
        "shipping/include/saccade_shipping/host_cfg_fields.inc is stale; re-run "
        "scripts/model/render_shipping_host_cfg_schema.py"
    )


def test_field_list_carries_types_not_values() -> None:
    text = (
        SHIPPING / "include" / "saccade_shipping" / "host_cfg_fields.inc"
    ).read_text()
    rows = [line for line in text.splitlines() if not line.startswith("//")]
    assert len(rows) == 416
    pattern = re.compile(
        r'^SACCADE_HOST_CFG_FIELD\([A-Za-z_][A-Za-z0-9_]*, "[^"]+", '
        r"(bool|std::int64_t|double|std::string|NullValue|IntList|StringList)\)$"
    )
    assert [row for row in rows if not pattern.match(row)] == []


def test_render_rejects_an_untyped_empty_list(tmp_path: Path) -> None:
    render = _render_module()
    config = tmp_path / "resolved.json"
    config.write_text('{"host_params": {"cfg": {"new_modes": []}}}')
    try:
        render.render(config)
    except render.SchemaError as exc:
        assert "EMPTY_LIST_ELEMENT" in str(exc)
    else:
        raise AssertionError("an empty list without a declared element type must fail")


def test_shipping_sources_never_read_the_environment() -> None:
    sources = sorted(
        p for p in SHIPPING.rglob("*") if p.suffix in {".cpp", ".hpp", ".inc", ".h"}
    )
    assert sources, "shipping/ sources not found"
    forbidden = re.compile(
        r"\b(getenv|secure_getenv|setenv|putenv|environ|_wgetenv|GetEnvironmentVariable)\b"
    )
    offenders = [
        f"{p.relative_to(REPO)}:{n}"
        for p in sources
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if forbidden.search(line.split("//", 1)[0])
    ]
    assert offenders == []


# Native headers shipping/ may include. The CUDA-free mapping sees only the
# parameter stores; the GPU builder, the post-detector host (PR-5) and its replay
# tool additionally see the objects they build or drive, and the host the
# copy-pad launcher it feeds main NMS with.
_CUDA_FREE_NATIVE = {"tracking/tracker_params.hpp", "tracking/perception_params.hpp"}
_GPU_BUILDER = {"native_build.cpp", "post_detector_host.cpp", "saccade_replay.cpp"}
_GPU_OBJECTS = {
    "tracking/tracker_gpu.hpp",
    "tracking/gmc.hpp",
    "tracking/pipeline.hpp",
    "tracking/copy_pad.cuh",
}


def test_shipping_sources_include_only_admitted_headers() -> None:
    include = re.compile(r'^\s*#\s*include\s+([<"])([^>"]+)[>"]')
    offenders = []
    for p in sorted(SHIPPING.rglob("*")):
        if p.suffix not in {".cpp", ".hpp", ".inc", ".h"}:
            continue
        admitted = _CUDA_FREE_NATIVE | (
            _GPU_OBJECTS if p.name in _GPU_BUILDER else set()
        )
        for line in p.read_text().splitlines():
            m = include.match(line)
            if (
                m
                and m.group(1) == '"'
                and not m.group(2).startswith("saccade_shipping/")
                and m.group(2) not in admitted
            ):
                offenders.append(f"{p.relative_to(REPO)}: {m.group(2)}")
    assert offenders == []


def test_native_env_reads_are_confined_to_legacy_front_ends() -> None:
    """PR-4b: env -> explicit parameter happens only in the legacy front-ends.

    ``legacy_env.cpp`` resolves every hatch the tracker/GMC/pipeline used to
    read; the pybind binding keeps its own handover-debug read (a non-shipping
    object). Any other native ``getenv`` or env helper is a regression.
    """
    native = sorted(
        p
        for root in ("src", "include")
        for p in (REPO / root).rglob("*")
        if p.suffix in {".cu", ".cuh", ".cpp", ".cc", ".hpp", ".h"}
    )
    reader = re.compile(
        r"\b(getenv|secure_getenv|env_flag_enabled|env_float_value|env_diagnostic_on)\s*\("
    )
    allowed = {
        "src/tracking/legacy_env.cpp",
        "src/tracking/tracker_gpu_python.cpp",
        "include/saccade/env_flag.hpp",  # the helper legacy_env.cpp uses
    }
    readers = {
        p.relative_to(REPO).as_posix()
        for p in native
        if any(
            reader.search(line.split("//", 1)[0]) for line in p.read_text().splitlines()
        )
    }
    assert readers - allowed == set()
    binding = (REPO / "src/tracking/tracker_gpu_python.cpp").read_text()
    assert re.findall(r'getenv\("(SACCADE_[A-Z0-9_]+)"\)', binding) == [
        "SACCADE_HO_DEBUG_LEVEL"
    ]
    users = {
        p.relative_to(REPO).as_posix()
        for p in native
        if '#include "tracking/legacy_env.hpp"' in p.read_text()
    }
    assert users == {
        "src/tracking/legacy_env.cpp",
        "src/tracking/seq_runner.cpp",
        "src/tracking/tracker_gpu_python.cpp",
    }
