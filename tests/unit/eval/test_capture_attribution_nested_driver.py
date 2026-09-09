"""Nested-driver parentage for runtime creates whose backtrace lacks the API symbol (#340).

CUPTI ``functionName`` is the create-API identity. A missing ``cudaStreamCreate*``
frame in the glibc backtrace is not unknown API. Unique nested DRIVER
``cuStreamCreate*`` parentage may supply the owner; it is never recorded as a
runtime-backtrace direct caller. Ambiguous, cross-thread, or missing nested
creates stay named evidence gaps.
"""

# scope: eval
# function: contract
# lifecycle: active

from tests.unit.eval.test_capture_attribution_harness import (
    fixture_rows,
    write_fixture,
)
from scripts.tools.capture_attribution.analyze import analyze


def _caller(path, symbol, address=1):
    return {
        "address": address,
        "status": "resolved",
        "symbol": symbol,
        "location": "fixture.cpp:1",
        "module": {"path": path, "sha256": "b" * 64},
    }


def _resolver(runtime_mode, driver_caller):
    """runtime_mode is ``absent`` (API symbol missing) or ``present`` (old path)."""

    def resolve(stack, api, truncated):
        frames = [{"address": item, "status": "located"} for item in stack]
        if not stack:
            return {
                "status": "gap",
                "gap": "missing_native_stack",
                "frames": [],
                "api_identity": "cupti_callback",
            }
        if truncated is not False:
            return {
                "status": "gap",
                "gap": "native_stack_truncated",
                "frames": frames,
                "api_identity": "cupti_callback",
            }
        if api.startswith("cudaStreamCreate"):
            if runtime_mode == "present":
                caller = _caller(
                    "/fixture/runtime_owner.so", "runtime_creator", stack[-1]
                )
                return {
                    "status": "resolved",
                    "api_identity": "cupti_callback",
                    "api_frame_index": 0,
                    "caller_frame_index": 1,
                    "caller": caller,
                    "frames": frames + [caller],
                }
            return {
                "status": "gap",
                "gap": "creation_api_frame_unresolved",
                "api_identity": "cupti_callback",
                "frames": frames,
            }
        if api.startswith("cuStreamCreate"):
            return {
                "status": "resolved",
                "api_identity": "cupti_callback",
                "api_frame_index": 0,
                "caller_frame_index": 1,
                "caller": driver_caller,
                "frames": frames + [driver_caller],
            }
        return {
            "status": "gap",
            "gap": "creation_api_frame_unresolved",
            "api_identity": "cupti_callback",
            "frames": frames,
        }

    return resolve


def _analyze(root, rows, resolver):
    write_fixture(root, rows)
    return analyze(root, stack_resolver=resolver)


def _nvinfer_caller():
    return _caller(
        "/opt/tensorrt/lib/libnvinfer.so.10",
        "getBuilderSafePluginRegistry",
        0x10,
    )


def _nvjpeg_caller():
    return _caller(
        "/venv/torchvision.libs/libnvjpeg.36e11081.so.13",
        "nvjpegDecodeJpegDevice",
        0x20,
    )


def test_absent_runtime_api_frame_with_unique_nested_driver_is_attributed(tmp_path):
    result = _analyze(tmp_path, fixture_rows(), _resolver("absent", _nvinfer_caller()))
    assert result["trace_structure_ok"]
    assert result["ownership_evidence_ok"]
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["status"] == "resolved"
    assert owner["runtime_api_identity"] == "cupti_callback"
    assert owner["owner_resolution"] == "nested_driver_parentage"
    assert owner["nested_driver_api"] == "cuStreamCreate"
    assert owner["frame"]["module"]["path"].endswith("libnvinfer.so.10")
    assert owner["frame"]["symbol"] == "getBuilderSafePluginRegistry"
    assert "runtime backtrace direct caller" in owner["evidence_rule"]
    runtime = next(
        item
        for item in result["stream_lifetimes"][0]["observed_creation_apis"]
        if item["api"].startswith("cudaStreamCreate")
    )
    assert runtime["stack_resolution"]["status"] == "gap"
    assert runtime["stack_resolution"]["gap"] == "creation_api_frame_unresolved"
    assert runtime["stack_resolution"]["frames"]
    assert runtime["native_stack"]
    assert result["capture_errors"][0]["error"]["rc"] == 906


def test_present_runtime_api_frame_keeps_the_direct_caller_path(tmp_path):
    result = _analyze(tmp_path, fixture_rows(), _resolver("present", _nvinfer_caller()))
    assert result["trace_structure_ok"]
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["owner_resolution"] == "runtime_backtrace_direct_caller"
    assert owner["nested_driver_api"] is None
    assert owner["frame"]["symbol"] == "runtime_creator"
    assert owner["frame"]["module"]["path"].endswith("runtime_owner.so")


def test_unique_tensorrt_like_nested_driver_caller(tmp_path):
    rows = fixture_rows()
    for row in rows:
        if row["api"] == "cudaStreamCreateWithFlags":
            row["api"] = "cudaStreamCreate"
    result = _analyze(tmp_path, rows, _resolver("absent", _nvinfer_caller()))
    assert result["trace_structure_ok"]
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["owner_resolution"] == "nested_driver_parentage"
    assert owner["correlation"]["runtime_api"] == "cudaStreamCreate"
    assert "libnvinfer.so.10" in owner["frame"]["module"]["path"]


def test_unique_nvjpeg_like_create_with_flags_nested_driver_caller(tmp_path):
    result = _analyze(tmp_path, fixture_rows(), _resolver("absent", _nvjpeg_caller()))
    assert result["trace_structure_ok"]
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["owner_resolution"] == "nested_driver_parentage"
    assert owner["correlation"]["runtime_api"] == "cudaStreamCreateWithFlags"
    assert owner["frame"]["symbol"] == "nvjpegDecodeJpegDevice"
    assert "libnvjpeg" in owner["frame"]["module"]["path"]


def test_nested_driver_absent_is_a_named_gap(tmp_path):
    rows = [
        row for row in fixture_rows() if not row["api"].startswith("cuStreamCreate")
    ]
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    result = _analyze(tmp_path, rows, _resolver("absent", _nvinfer_caller()))
    assert not result["trace_structure_ok"]
    assert not result["ownership_evidence_ok"]
    assert any(
        value.endswith("nested_driver_create_absent")
        for value in result["evidence_gaps"]
    )
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["status"] == "gap"
    assert owner["gap"] == "nested_driver_create_absent"
    assert owner["owner_resolution"] is None


def test_two_ambiguous_nested_driver_creates_are_a_named_gap(tmp_path):
    rows = fixture_rows()
    extra = []
    for original in rows:
        if not original["api"].startswith("cuStreamCreate"):
            continue
        row = dict(original)
        row["ns"] = 15 if original["phase"] == "enter" else 25
        row["correlation"] += 50
        row["cbid"] += 50
        row["stream"] = 77 if original["phase"] == "exit" else 0
        extra.append(row)
    rows.extend(extra)
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    result = _analyze(tmp_path, rows, _resolver("absent", _nvinfer_caller()))
    assert not result["trace_structure_ok"]
    assert any(
        value.endswith("nested_driver_create_ambiguous")
        for value in result["evidence_gaps"]
    )
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["status"] == "gap"
    assert owner["gap"] == "nested_driver_create_ambiguous"


def test_mismatched_thread_is_not_attributed(tmp_path):
    rows = fixture_rows()
    for row in rows:
        if row["api"].startswith("cuStreamCreate"):
            row["tid"] = 99
    result = _analyze(tmp_path, rows, _resolver("absent", _nvinfer_caller()))
    assert not result["trace_structure_ok"]
    assert any(
        value.endswith("nested_driver_thread_mismatch")
        for value in result["evidence_gaps"]
    )
    owner = result["stream_lifetimes"][0]["owner"]
    assert owner["status"] == "gap"
    assert owner["gap"] == "nested_driver_thread_mismatch"


def test_parentage_does_not_mask_the_primary_capture_error(tmp_path):
    result = _analyze(tmp_path, fixture_rows(), _resolver("absent", _nvinfer_caller()))
    errors = result["capture_errors"]
    assert errors
    assert errors[0]["error"]["rc"] == 906
    assert "IsCapturing" in errors[0]["error"]["api"]
    assert errors[0]["observed_open_captures"]
    assert result["root_cause_closed"] is False
