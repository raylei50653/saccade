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

Phase C PR-C2 splits the entrypoint: the shipping ``saccade_track`` gets its
interface only and its report must carry no ``measurement`` record; the
negative controls, the serial override and ``--max-frames`` run the developer
build ``saccade_track_measurement``, whose report must echo exactly what was
asked. The command line the harness builds for the shipping binary holds no
developer option, and ``--against`` also compares the native graph counts.

#536 CC-536-01-01 (run completion): the report is v3 and must carry a run id
and an identity that claims no level; the journal reader fails closed on a
run id that is not the invocation's, a state other than ``complete``, a
``pending`` sequence whose txt is present, and a txt / trace / report whose
bytes are not the ones recorded. The same reader is run over the files the
real writer leaves in each case of ``tests/native/test_shipping_run_completion.cpp``
(``--keep``; skipped when that test is not built).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "eval" / "diagnostics" / "native_track_parity.py"
COMPLETION_TEST = REPO / "build" / "shipping" / "saccade_shipping_run_completion_test"
RUN_ID = "0123456789abcdef0123456789abcdef"


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
        "run_id": RUN_ID,
        "identity": {"level": None},
        "entrypoint": "saccade_track",
        "schedule": "serial",
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
        (("format",), "saccade.native_track_report/v1"),
        (("format",), "saccade.native_track_report/v2"),
        (("run_id",), None),
        (("run_id",), "run-1"),
        (("identity",), None),
        (("identity",), {"level": "checksum_matched"}),
        (("identity",), {"level": "expected_source_verified"}),
        (("entrypoint",), "saccade_track_measurement"),
        (("schedule",), "double_buffer"),
        (
            ("measurement",),
            {"mutation": "none", "schedule_override": None, "max_frames": 0},
        ),
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
    args = (att, lineage, ["S1", "S2"], "shipping", "none", None)
    assert T.report_problems(rep, *args) == []
    bad = copy.deepcopy(rep)
    node = bad
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert T.report_problems(bad, *args) != []


def test_report_problems_checks_the_schedule() -> None:
    rep, att, lineage = _good_report()
    rep["schedule"] = "double_buffer"
    args = (att, lineage, ["S1", "S2"], "shipping", "none", None)
    assert T.report_problems(rep, *args, "double_buffer") == []
    assert T.report_problems(rep, *args, "serial") != []


def _measurement_report(
    mutation: str, max_frames: int | None, schedule: str
) -> dict[str, Any]:
    rep, _, _ = _good_report()
    rep["entrypoint"] = "saccade_track_measurement"
    rep["schedule"] = schedule
    rep["measurement"] = T.expected_measurement(mutation, max_frames, schedule)
    return rep


@pytest.mark.parametrize(
    "key,value",
    [
        ("mutation", "none"),
        ("mutation", "stale_image_dims"),
        ("schedule_override", None),
        ("max_frames", 0),
        ("max_frames", 31),
    ],
)
def test_report_problems_measurement_record(key: str, value: Any) -> None:
    _, att, lineage = _good_report()
    rep = _measurement_report("gmc_previous_frame", 30, "serial")
    args = (
        att,
        lineage,
        ["S1", "S2"],
        "measurement",
        "gmc_previous_frame",
        30,
        "serial",
    )
    assert T.report_problems(rep, *args) == []
    bad = copy.deepcopy(rep)
    bad["measurement"][key] = value
    assert T.report_problems(bad, *args) != []
    # A measurement report never passes as a shipping one, and the reverse.
    assert (
        T.report_problems(
            rep, att, lineage, ["S1", "S2"], "shipping", "none", None, "serial"
        )
        != []
    )
    ship, _, _ = _good_report()
    assert T.report_problems(ship, *args) != []


def _parity_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "parity",
        "--out",
        str(tmp_path / "out"),
        "--oracle-rows",
        str(tmp_path),
        "--oracle-txt",
        str(tmp_path),
        *extra,
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ("--mutation", "stale_gmc_input"),
        ("--schedule", "serial"),
        ("--max-frames", "5"),
    ],
)
def test_developer_options_need_the_measurement_entrypoint(
    tmp_path: Path, extra: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as e:
        T.main(_parity_args(tmp_path, *extra))
    assert e.value.code == 2
    assert "--entrypoint measurement" in capsys.readouterr().err


def _run_track_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any
) -> list[str]:
    seen: list[list[str]] = []

    class _Done:
        returncode = 0

    def fake_run(cmd: list[str], **_: Any) -> Any:
        seen.append(cmd)
        return _Done()

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    args = T.argparse.Namespace(
        model_root=Path("."),
        track_binary=Path("bin/x"),
        no_trace=False,
        track_library_path=None,
        sequences=["S1"],
        **kw,
    )
    T.run_track(args, tmp_path)
    return seen[0]


def test_shipping_command_line_has_no_developer_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _run_track_cmd(
        tmp_path,
        monkeypatch,
        entrypoint="shipping",
        mutation="none",
        schedule="double_buffer",
        max_frames=None,
    )
    options = {a for a in cmd if a.startswith("--")}
    assert options == {
        "--config",
        "--lineage",
        "--attestation",
        "--model-root",
        "--out",
        "--report",
        "--trace",
    }
    m = _run_track_cmd(
        tmp_path,
        monkeypatch,
        entrypoint="measurement",
        mutation="gmc_previous_frame",
        schedule="serial",
        max_frames=30,
    )
    i = m.index("--measurement-mutation")
    assert m[i + 1] == "gmc_previous_frame"
    assert {"--schedule", "--max-frames"} <= set(m)


def _per_seq(txt: str, graphs: dict[str, int] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"native_txt_sha256": txt, "native_trace_sha256": "t"}
    if graphs is not None:
        out["graph_captures"] = {"native": graphs}
    return out


def test_compare_against_graph_counts(tmp_path: Path) -> None:
    g = {"detector_captures": 1, "nms_replays": 10}
    prior = tmp_path / "prior.json"
    prior.write_text(T.json.dumps({"per_sequence": {"S1": _per_seq("a", g)}}))
    same = {"per_sequence": {"S1": _per_seq("a", dict(g))}}
    assert T.compare_against(same, prior)["identical"]
    one = {"per_sequence": {"S1": _per_seq("a", {**g, "nms_replays": 9})}}
    r = T.compare_against(one, prior)
    assert not r["identical"]
    assert r["differing"] == [{"sequence": "S1", "key": "graph_captures.native"}]
    # A prior without graph counts cannot vouch for them.
    prior.write_text(T.json.dumps({"per_sequence": {"S1": _per_seq("a", None)}}))
    assert not T.compare_against(same, prior)["identical"]


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


# ── completion (#536 CC-536-01-01) ─────────────────────────────────────────────

SEQS = ["S1", "S2"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _completed_run(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    """A complete run's files: out/, trace/, the report, and its journal."""
    out, trace, report = tmp_path / "out", tmp_path / "trace", tmp_path / "r.json"
    out.mkdir()
    entries = []
    for s in SEQS:
        txt, tb = f"1,1,{s}".encode(), f"trace {s}".encode()
        (out / f"{s}.txt").write_bytes(txt)
        (trace / s).mkdir(parents=True)
        (trace / s / "detector.bin").write_bytes(tb)
        entries.append(
            {
                "name": s,
                "state": "written",
                "txt": str(out / f"{s}.txt"),
                "txt_sha256": _sha(txt),
                "trace": str(trace / s / "detector.bin"),
                "trace_sha256": _sha(tb),
            }
        )
    report.write_bytes(b'{"run_id": "x"}\n')
    journal = {
        "format": T.JOURNAL_FORMAT,
        "run_id": RUN_ID,
        "entrypoint": "saccade_track",
        "state": "complete",
        "identity": {"level": None},
        "sequences": entries,
        "report": {"path": str(report), "sha256": _sha(report.read_bytes())},
        "failure": None,
    }
    return out, trace, report, journal


def _problems(
    out: Path, trace: Path, report: Path, journal: dict[str, Any]
) -> list[str]:
    (out / T.JOURNAL_NAME).write_text(json.dumps(journal))
    return T.journal_problems(out, RUN_ID, SEQS, report, trace)


def test_journal_complete_run(tmp_path: Path) -> None:
    out, trace, report, j = _completed_run(tmp_path)
    assert _problems(out, trace, report, j) == []
    assert T.committed_sequences(j, out, RUN_ID) == SEQS


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j.update(run_id="f" * 32),  # an earlier run's journal
        lambda j: j.update(state="running"),
        lambda j: j.update(state="failed"),
        lambda j: j.update(format="saccade.native_track_journal/v0"),
        lambda j: j.update(identity={"level": "checksum_matched"}),
        lambda j: j["sequences"].reverse(),
        lambda j: j["sequences"].pop(),
        lambda j: j["sequences"][1].update(state="pending", txt_sha256=None),
        lambda j: j["sequences"][1].update(txt_sha256="0" * 64),
        lambda j: j["sequences"][0].update(trace_sha256="0" * 64),
        lambda j: j["report"].update(sha256=None),
        lambda j: j.update(report=None),
    ],
)
def test_journal_problems_fail_closed(tmp_path: Path, edit: Any) -> None:
    out, trace, report, j = _completed_run(tmp_path)
    edit(j)
    assert _problems(out, trace, report, j) != []


def test_pending_with_its_file_present_is_not_committed(tmp_path: Path) -> None:
    # A kill between the rename and the journal update: the txt is there and
    # may be this run's whole output; the journal never confirmed it.
    out, trace, report, j = _completed_run(tmp_path)
    j["state"] = "running"
    j["sequences"][1].update(state="pending", txt_sha256=None, trace_sha256=None)
    assert (out / "S2.txt").is_file()
    assert T.committed_sequences(j, out, RUN_ID) == ["S1"]
    assert _problems(out, trace, report, j) != []


def test_stale_or_replaced_files_are_not_committed(tmp_path: Path) -> None:
    out, trace, report, j = _completed_run(tmp_path)
    (out / "S1.txt").write_bytes(b"an older run's txt")
    assert T.committed_sequences(j, out, RUN_ID) == ["S2"]
    assert T.committed_sequences(j, out, "f" * 32) == []
    (out / "S1.txt").unlink()
    assert T.committed_sequences(j, out, RUN_ID) == ["S2"]
    # A report outside <out> replaced by another run: detected by its hash.
    (tmp_path / "b").mkdir()
    out2, trace2, report2, j2 = _completed_run(tmp_path / "b")
    report2.write_bytes(b'{"run_id": "another run"}\n')
    assert any("report" in p for p in _problems(out2, trace2, report2, j2))
    (out2 / T.JOURNAL_NAME).unlink()
    assert T.journal_problems(out2, RUN_ID, SEQS, report2, trace2) != []
    assert T.journal_problems(out, None, SEQS, report, trace) != []


def test_invocation_run_id() -> None:
    a, b = "a" * 32, "b" * 32
    log = f"saccade_track: run_id {a}\n[saccade_track] S1: 5 frames\n"
    assert T.invocation_run_id(log, "saccade_track") == a
    assert T.invocation_run_id(log, "saccade_track_measurement") is None
    assert (
        T.invocation_run_id(log + f"saccade_track: run_id {b}\n", "saccade_track")
        is None
    )
    assert T.invocation_run_id("saccade_track: run_id xyz\n", "saccade_track") is None
    assert T.invocation_run_id("", "saccade_track") is None


@pytest.mark.skipif(not COMPLETION_TEST.exists(), reason="completion test not built")
def test_reader_on_the_writer_s_files(tmp_path: Path) -> None:
    """The real writer's files in each completion case, judged by this reader."""
    keep = tmp_path / "keep"
    r = subprocess.run(
        [str(COMPLETION_TEST), "--keep", str(keep)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    seqs = ["SEQ-A", "SEQ-B", "SEQ-C"]

    def judge(
        case: str, out: str = "out", run_id_file: str = "invocation.run_id"
    ) -> tuple[list[str], list[str]]:
        d = keep / case
        run_id = (d / run_id_file).read_text()
        j = json.loads((d / out / T.JOURNAL_NAME).read_text())
        committed = T.committed_sequences(j, d / out, run_id)
        problems = T.journal_problems(
            d / out, run_id, seqs, d / "track_report.json", d / "trace"
        )
        return committed, problems

    assert judge("fresh", "missing/parent/out") == (seqs, [])
    assert judge("killed_in_sequence") == (seqs, [])  # the run after the kill
    committed, problems = judge("failed_rerun")
    assert committed == ["SEQ-A"] and problems
    committed, problems = judge("killed_after_rename")
    assert committed == ["SEQ-A"] and problems
    assert (keep / "killed_after_rename" / "out" / "SEQ-B.txt").is_file()
    committed, problems = judge("killed_after_report")
    assert committed == seqs and any("state 'running'" in p for p in problems)
    assert (keep / "killed_after_report" / "track_report.json").is_file()
    committed, problems = judge("lock_busy")  # the refused run names nothing
    assert committed == [] and problems
    committed, problems = judge("lock_busy", run_id_file="holder.run_id")
    assert committed == ["SEQ-A"] and problems
    assert judge("collisions")[1]
