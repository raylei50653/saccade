"""``native_track_parity.py`` comparators on synthetic inputs (#465 PR-9, PR-10).

The PR-9 verdict rests on two comparisons: the native detector rows at the
end-to-end wiring against the oracle run's ``detector.bin``, and the native
MOT txt against the oracle's after relabeling its run-global ids. These tests
pin that both comparators see a one-bit / one-character difference, that the
relabeling refuses an id map it cannot invert, and that the validity check
fails closed on every field of the ``saccade_track`` report it reads.

PR-10 adds the double-buffer schedule: the graph-capture section (the oracle
log's per-sequence captures vs the native counts, and the native replay
counts vs the frames) must see a one-count difference, the oracle-log parser
must attribute captures to the sequence block they fall in, and the
double-buffer oracle (an ``anchor`` run) must fail closed on each validity
field.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "eval" / "diagnostics" / "native_track_parity.py"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("native_track_parity", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T = _tool()


def _record(frame: int, n: int, x: float = 1.0) -> bytes:
    return (
        struct.pack("<3i", frame, n, 0)
        + struct.pack(f"<{4 * n}f", *([x] * 4 * n))
        + struct.pack(f"<{n}f", *([0.5] * n))
        + struct.pack(f"<{n}i", *([0] * n))
    )


def test_detector_rows_equal_and_one_bit(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(_record(1, 3) + _record(2, 3))
    b.write_bytes(_record(1, 3) + _record(2, 3))
    assert T.compare_detector(a, b)["verdict"] == "EXACT"
    one_ulp = struct.unpack(
        "<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", 1.0))[0] + 1)
    )[0]
    b.write_bytes(_record(1, 3) + _record(2, 3, one_ulp))
    r = T.compare_detector(a, b)
    assert r["verdict"] == "DIFFERS" and r["equal_frames"] == 1
    assert r["first_differing"] == [{"frame": 2, "rows": [3, 3]}]
    b.write_bytes(_record(1, 3) + _record(2, 2))
    assert T.compare_detector(a, b)["count_mismatch_frames"] == 1


def test_detector_frame_sets_and_truncation(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(_record(1, 2) + _record(2, 2))
    b.write_bytes(_record(1, 2))
    r = T.compare_detector(a, b)
    assert r["verdict"] == "DIFFERS" and not r["frame_sets_equal"]
    b.write_bytes(_record(1, 2)[:-1])
    with pytest.raises(RuntimeError, match="truncated"):
        T.compare_detector(a, b)


def test_id_blocks_and_relabel(tmp_path: Path) -> None:
    m = tmp_path / "_global_id_map.txt"
    m.write_text(
        "S1\tlocal_id=1\tglobal_id=1\nS1\tlocal_id=2\tglobal_id=2\n"
        "S2\tlocal_id=1\tglobal_id=4\nS2\tlocal_id=2\tglobal_id=3\n"
    )
    assert T.id_blocks(m) == {"S1": (0, 2), "S2": (2, 2)}
    assert T.relabel("1,3,1.00,2\n2,4,1.00,2", 2) == "1,1,1.00,2\n2,2,1.00,2"
    assert T.relabel("", 5) == ""
    m.write_text("S1\tlocal_id=1\tglobal_id=1\nS1\tlocal_id=2\tglobal_id=3\n")
    with pytest.raises(RuntimeError, match="contiguous"):
        T.id_blocks(m)
    with pytest.raises(RuntimeError, match="id field"):
        T.relabel("1,2", 0)


def test_mot_compare_one_character() -> None:
    ref = "1,1,10.00,20.00,5.00,6.00,0.9000,-1,-1,-1\n2,1,11.00,20.00,5.00,6.00,0.9000,-1,-1,-1"
    assert T.compare_mot(ref, ref)["byte_identical_after_relabel"]
    r = T.compare_mot(ref.replace("11.00", "11.01"), ref)
    assert not r["byte_identical_after_relabel"] and r["first_difference"]["line"] == 1
    r = T.compare_mot(ref + "\n", ref)  # a trailing newline is a difference
    assert not r["byte_identical_after_relabel"]


def _good_report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    att = {"op_library": {"sha256": "op"}}
    lineage = {
        "torchscript": {"sha256": "ts", "native_scan_calls": 3},
        "companions": {"backbone_engine": {"sha256": "eng"}},
        "runtime_requirements": {"graph_executor_optimize": False},
    }
    rep = {
        "format": T.TRACK_REPORT_FORMAT,
        "schedule": "serial",
        "mutation": "none",
        "max_frames": 0,
        "python_libraries_mapped": [],
        "sequence_order": ["S1", "S2"],
        "detector": {
            "plan": {"op_library": {"from_attestation": True}},
            "load": {
                "op_library_sha256": "op",
                "head_artifact_sha256": "ts",
                "backbone_engine_sha256": "eng",
                "runtime_readback": {"graph_executor_optimize": False},
                "native_scan_calls": 3,
                "param_devices": ["cuda:0"],
                "constant_devices": ["cpu", "cpu"],
            },
        },
    }
    return rep, att, lineage


@pytest.mark.parametrize(
    "path,value",
    [
        (("format",), "x"),
        (("schedule",), "double_buffer"),
        (("mutation",), "gmc_previous_frame"),
        (("max_frames",), 30),
        (("python_libraries_mapped",), ["libpython3.12.so"]),
        (("sequence_order",), ["S2", "S1"]),
        (("detector", "plan", "op_library", "from_attestation"), False),
        (("detector", "load", "op_library_sha256"), "other"),
        (("detector", "load", "head_artifact_sha256"), "other"),
        (("detector", "load", "backbone_engine_sha256"), "other"),
        (("detector", "load", "runtime_readback"), {"graph_executor_optimize": True}),
        (("detector", "load", "native_scan_calls"), 2),
        (("detector", "load", "param_devices"), ["cpu"]),
        (("detector", "load", "constant_devices"), ["cuda:0"]),
    ],
)
def test_report_problems_fail_closed(path: tuple[str, ...], value: Any) -> None:
    rep, att, lineage = _good_report()
    assert T.report_problems(rep, att, lineage, ["S1", "S2"], "none", None) == []
    bad = copy.deepcopy(rep)
    node = bad
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert T.report_problems(bad, att, lineage, ["S1", "S2"], "none", None) != []


def test_report_problems_checks_the_schedule() -> None:
    rep, att, lineage = _good_report()
    rep["schedule"] = "double_buffer"
    args = (att, lineage, ["S1", "S2"], "none", None)
    assert T.report_problems(rep, *args, "double_buffer") == []
    assert T.report_problems(rep, *args, "serial") != []


def test_negative_controls_name_a_section_and_a_schedule() -> None:
    assert set(T.NEGCTL_EXPECT.values()) <= {"detector", "mot_txt"}
    assert {
        "shared_post_host",
        "stale_image_dims",
        "gmc_previous_frame",
        "stale_detector_input",
        "stale_gmc_input",
        "swapped_detection_parity",
        "ref_edit",
    } == set(T.NEGCTL_EXPECT)
    assert set(T.NEGCTL_SCHEDULE) == set(T.NEGCTL_EXPECT) - {"ref_edit"}
    assert set(T.NEGCTL_SCHEDULE.values()) == set(T.SCHEDULES)


_ORACLE_LOG = """\
🕯️ [TrackerGraph] Captured tracker update for seq S1
  [double-buffer] detect(N+1) overlaps tracker(N) on a side stream
🕯️ [WholeDetectGraph] Capturing graphed callable for shape (1, 3, 1080, 1920) img=(1080, 1920)
🕯️ [MainNMSGraphNoCopyback] Captured main NMS nocopyback graph (graph=<x>)
🕯️ [GMCGraph] Captured C++ cuFFT GMC graph for seq S1 (img=1080×1920 ds=4)
🕯️ [TrackerGraph] Captured tracker update for seq S2
🕯️ [MainNMSGraphNoCopyback] Captured main NMS nocopyback graph (graph=<y>)
🕯️ [GMCGraph] Captured C++ cuFFT GMC graph for seq S2 (img=1080×1920 ds=4)
"""


def test_oracle_graph_captures_per_sequence_block() -> None:
    assert T.oracle_graph_captures(_ORACLE_LOG) == {
        "S1": {"detector": 1, "nms": 1, "gmc": 1},
        "S2": {"detector": 0, "nms": 1, "gmc": 1},
    }
    with pytest.raises(RuntimeError, match="before any sequence"):
        T.oracle_graph_captures(_ORACLE_LOG.split("\n", 2)[2])
    swapped = _ORACLE_LOG.replace("graph for seq S2", "graph for seq S1")
    with pytest.raises(RuntimeError, match="inside S2"):
        T.oracle_graph_captures(swapped)


def _native_graphs() -> dict[str, Any]:
    return {
        "frames": 10,
        "tracker_updates": 10,
        "graphs": {
            "detector_captures": 1,
            "detector_warmup_runs": 4,
            "detector_replays": 10,
            "nms_captures": 1,
            "nms_replays": 10,
            "gmc_captures": 1,
            "gmc_replays": 9,
            "tracker_captures": 1,
            "tracker_replays": 10,
        },
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("detector_captures", 0),
        ("nms_captures", 2),
        ("gmc_captures", 0),
        ("tracker_captures", 2),
        ("detector_replays", 9),
        ("nms_replays", 9),
        ("gmc_replays", 10),
        ("tracker_replays", 11),
    ],
)
def test_compare_graphs_one_count(key: str, value: int) -> None:
    oracle = {"detector": 1, "nms": 1, "gmc": 1}
    assert T.compare_graphs(_native_graphs(), oracle)["verdict"] == "EXACT"
    bad = _native_graphs()
    bad["graphs"][key] = value
    assert T.compare_graphs(bad, oracle)["verdict"] == "DIFFERS"


def test_compare_graphs_needs_an_oracle_block() -> None:
    r = T.compare_graphs(_native_graphs(), None)
    assert r["verdict"] == "DIFFERS" and r["problems"]


def _good_anchor() -> dict[str, Any]:
    return {
        "schema": T.det.ANCHOR_SCHEMA,
        "problems": [],
        "identical": True,
        "git": {"head": "abc", "dirty": False},
        "mot17_argv": [
            "scripts/eval/mot17.py",
            "--double-buffer",
            "--sequences",
            "S1,S2",
        ],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "other"),
        ("problems", ["V5: x"]),
        ("identical", False),
        ("git", {"head": "abc", "dirty": True}),
        ("git", {"head": "def", "dirty": False}),
        ("mot17_argv", ["scripts/eval/mot17.py", "--sequences", "S1,S2"]),
        (
            "mot17_argv",
            ["scripts/eval/mot17.py", "--double-buffer", "--sequences", "S2,S1"],
        ),
    ],
)
def test_oracle_txt_problems_fail_closed(tmp_path: Path, key: str, value: Any) -> None:
    good = _good_anchor()
    assert T.oracle_txt_problems(tmp_path, good, ["S1", "S2"], "abc") == []
    bad = copy.deepcopy(good)
    bad[key] = value
    assert T.oracle_txt_problems(tmp_path, bad, ["S1", "S2"], "abc") != []
