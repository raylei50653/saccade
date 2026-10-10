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

#536 CC-536-01-01 (run completion) and #549 S2-1a: current report v4 / journal
v2 records Gate A checksum observations; historical v3 / v1 identity stays
exactly null. Cross-version pairs or differing identities are invalid. The
journal reader fails closed on a
run id that is not the invocation's, a state other than ``complete``, a
``pending`` sequence whose txt is present, a txt / trace / report whose
bytes are not the ones recorded, and a journal that is not JSON, not an
object, or whose ``sequences`` is not a list of objects (problems, not an
exception); the run id is read only from the log's first line. The same reader is run over the files the
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


def _good_identity(attested: bool = True) -> dict[str, Any]:
    paths = {name: f"/frozen/{name}" for name in T.IDENTITY_BINDINGS}
    hashes = {
        "config": "e" * 64,
        "lineage": "d" * 64,
        "attestation": "f" * 64,
        "op_library": "a" * 64,
        "head": "b" * 64,
        "engine": "c" * 64,
    }
    bindings = {}
    for name in T.IDENTITY_BINDINGS:
        source = None
        expected = None
        status = "unchecked"
        if name == "lineage" and attested:
            source = {
                "path": paths["attestation"],
                "json_pointer": "/frozen_lineage/sha256",
            }
        elif name == "op_library":
            source = {
                "path": paths["attestation" if attested else "lineage"],
                "json_pointer": "/op_library/sha256",
            }
        elif name in ("head", "engine"):
            pointer = (
                "/torchscript/sha256"
                if name == "head"
                else "/companions/backbone_engine/sha256"
            )
            source = {"path": paths["lineage"], "json_pointer": pointer}
        if source is not None:
            expected, status = hashes[name], "matched"
        bindings[name] = {
            "path": paths[name],
            "expected_sha256": expected,
            "observed_sha256": hashes[name],
            "status": status,
            "expected_source": source,
        }
    if not attested:
        bindings["attestation"] = {
            "path": None,
            "expected_sha256": None,
            "observed_sha256": None,
            "status": "unchecked",
            "expected_source": None,
        }
    return {
        "level": "checksum_matched",
        "expected_source": None,
        "publisher_authentication": "not_checked_by_runtime",
        "bindings": bindings,
    }


def _good_report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = _good_identity()
    bindings = identity["bindings"]
    att = {
        "op_library": {"sha256": bindings["op_library"]["expected_sha256"]},
        "frozen_lineage": {"sha256": bindings["lineage"]["expected_sha256"]},
    }
    lineage = {
        "torchscript": {
            "sha256": bindings["head"]["expected_sha256"],
            "native_scan_calls": 3,
        },
        "companions": {
            "backbone_engine": {"sha256": bindings["engine"]["expected_sha256"]}
        },
        "runtime_requirements": {"graph_executor_optimize": False},
    }
    rep = {
        "format": T.TRACK_REPORT_FORMAT,
        "run_id": RUN_ID,
        "identity": identity,
        "entrypoint": "saccade_track",
        "model_root": "/frozen",
        "config": bindings["config"]["path"],
        "lineage": bindings["lineage"]["path"],
        "attestation": bindings["attestation"]["path"],
        "schedule": "serial",
        "python_libraries_mapped": [],
        "sequence_order": ["S1", "S2"],
        "detector": {
            "plan": {
                "op_library": {
                    "from_attestation": True,
                    "path": bindings["op_library"]["path"],
                    "sha256": bindings["op_library"]["expected_sha256"],
                },
                "head_artifact": {
                    "path": bindings["head"]["path"],
                    "sha256": bindings["head"]["expected_sha256"],
                },
                "backbone_engine": {
                    "path": bindings["engine"]["path"],
                    "sha256": bindings["engine"]["expected_sha256"],
                },
            },
            "load": {
                "op_library_sha256": bindings["op_library"]["expected_sha256"],
                "head_artifact_sha256": bindings["head"]["expected_sha256"],
                "backbone_engine_sha256": bindings["engine"]["expected_sha256"],
                "runtime_readback": {"graph_executor_optimize": False},
                "native_scan_calls": 3,
                "param_devices": ["cuda:0"],
                "constant_devices": ["cpu", "cpu"],
            },
        },
    }
    return rep, att, lineage


@pytest.mark.parametrize("attested", [False, True])
def test_gate_a_identity_with_and_without_optional_attestation(attested: bool) -> None:
    identity = _good_identity(attested)
    assert T.identity_problems(identity) == []
    # These synthetic paths do not exist: reading evidence never reopens the
    # original metadata or derives a trusted source from the observations.
    assert identity["bindings"]["config"]["expected_sha256"] is None
    assert identity["expected_source"] is None


@pytest.mark.parametrize(
    "path,value,diagnostic",
    [
        (("level",), None, "identity level"),
        (("level",), "expected_source_verified", "identity level"),
        (("expected_source",), "runtime_allowlist", "expected_source"),
        (("publisher_authentication",), "authenticated", "publisher_authentication"),
        (("bindings",), [], "six legacy slots"),
        (("bindings", "config"), None, "invalid binding fields"),
        (
            ("bindings", "head", "expected_sha256"),
            None,
            "requires path and both hashes",
        ),
        (
            ("bindings", "head", "observed_sha256"),
            "0" * 64,
            "contradicts hash comparison",
        ),
        (("bindings", "head", "observed_sha256"), "bad", "invalid observed_sha256"),
        (("bindings", "head", "status"), "mismatch", "contradicts hash comparison"),
        (("bindings", "head", "path"), None, "requires path and both hashes"),
        (("bindings", "head", "path"), [], "invalid path"),
        (("bindings", "head", "expected_source"), None, "present together"),
        (("bindings", "head", "expected_source"), [], "invalid expected_source"),
        (
            ("bindings", "head", "expected_source", "path"),
            "/different/lineage",
            "supplied metadata field",
        ),
        (
            ("bindings", "head", "expected_source", "json_pointer"),
            "/op_library/sha256",
            "supplied metadata field",
        ),
        (
            ("bindings", "config", "expected_sha256"),
            "e" * 64,
            "supplies no expected hash",
        ),
        (
            ("bindings", "attestation", "status"),
            "matched",
            "requires path and both hashes",
        ),
        (("bindings", "lineage", "status"), "unchecked", "incomplete comparison"),
        (
            ("bindings", "attestation", "observed_sha256"),
            None,
            "requires observed bytes",
        ),
        (
            ("bindings", "op_library", "expected_source", "path"),
            "/frozen/lineage",
            "differs from accepted metadata",
        ),
    ],
)
def test_identity_invalid_claims_are_diagnostic(
    path: tuple[str, ...], value: Any, diagnostic: str
) -> None:
    identity = _good_identity()
    node = identity
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert any(diagnostic in p for p in T.identity_problems(identity))


def test_identity_slots_and_fields_are_exact() -> None:
    identity = _good_identity()
    identity["bindings"]["extra"] = {}
    assert "six legacy slots" in " ".join(T.identity_problems(identity))
    identity = _good_identity()
    identity["bindings"]["head"]["loaded_buffer"] = True
    assert "invalid binding fields" in " ".join(T.identity_problems(identity))


@pytest.mark.parametrize("status", ["mismatch", "missing", "unchecked"])
def test_partial_gate_a_diagnostics_cannot_complete(status: str) -> None:
    identity = _good_identity()
    identity["level"] = None
    head = identity["bindings"]["head"]
    head["status"] = status
    head["observed_sha256"] = "0" * 64 if status == "mismatch" else None
    assert T.identity_problems(identity, complete=False) == []
    assert any("identity level" in p for p in T.identity_problems(identity))


def test_requested_missing_attestation_cannot_become_optional_omission() -> None:
    identity = _good_identity()
    identity["level"] = None
    identity["bindings"]["attestation"].update(status="missing", observed_sha256=None)
    assert T.identity_problems(identity, complete=False) == []
    identity["level"] = "checksum_matched"
    assert any("attestation" in p for p in T.identity_problems(identity))
    omitted = _good_identity(False)
    omitted["bindings"]["attestation"]["status"] = "missing"
    assert any("attestation" in p for p in T.identity_problems(omitted))


def test_historical_report_keeps_null_identity() -> None:
    rep, att, lineage = _good_report()
    rep.update(format=T.LEGACY_TRACK_REPORT_FORMAT, identity={"level": None})
    args = (att, lineage, ["S1", "S2"], "shipping", "none", None)
    assert T.report_problems(rep, *args) == []
    rep["identity"] = _good_identity()
    assert any("historical identity" in p for p in T.report_problems(rep, *args))


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


@pytest.mark.parametrize(
    "path,value,diagnostic",
    [
        (("detector",), None, "report detector is not an object"),
        (("detector", "plan"), [], "report detector plan is not an object"),
        (("detector", "load"), "loaded", "report detector load is not an object"),
        (
            ("detector", "plan", "op_library"),
            None,
            "report operator plan is not an object",
        ),
        (
            ("detector", "plan", "head_artifact"),
            [],
            "report plan head_artifact is not an object",
        ),
        (("detector", "load", "constant_devices"), None, "head placement"),
        (("detector", "load", "constant_devices"), 7, "head placement"),
        (
            ("detector", "plan", "head_artifact", "sha256"),
            "0" * 64,
            "identity binding head differs",
        ),
        (
            ("detector", "plan", "head_artifact", "path"),
            "/other/head",
            "identity binding head differs",
        ),
    ],
)
def test_report_nested_corruption_is_diagnostic(
    path: tuple[str, ...], value: Any, diagnostic: str
) -> None:
    rep, att, lineage = _good_report()
    node = rep
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    problems = T.report_problems(
        rep, att, lineage, ["S1", "S2"], "shipping", "none", None
    )
    assert any(diagnostic in p for p in problems)


def test_identity_cannot_describe_other_bytes_than_the_load() -> None:
    rep, att, lineage = _good_report()
    rep["detector"]["plan"]["head_artifact"]["sha256"] = "0" * 64
    rep["identity"]["bindings"]["head"].update(
        expected_sha256="0" * 64, observed_sha256="0" * 64
    )
    problems = T.report_problems(
        rep, att, lineage, ["S1", "S2"], "shipping", "none", None
    )
    assert any("identity binding head differs" in p for p in problems)


def test_identity_lineage_expected_hash_is_the_supplied_attestation_value() -> None:
    rep, att, lineage = _good_report()
    rep["identity"]["bindings"]["lineage"].update(
        expected_sha256="0" * 64, observed_sha256="0" * 64
    )
    problems = T.report_problems(
        rep, att, lineage, ["S1", "S2"], "shipping", "none", None
    )
    assert any("lineage differs from supplied attestation" in p for p in problems)


@pytest.mark.parametrize("root", ["", ".", "relative/root", "/absolute/root"])
@pytest.mark.parametrize("absolute_artifact", [False, True])
def test_report_binding_paths_use_recorded_model_root(
    root: str, absolute_artifact: bool
) -> None:
    rep, att, lineage = _good_report()
    rep["model_root"] = root
    bindings = rep["identity"]["bindings"]
    for name, argument in (
        ("config", "./cfg/config.json"),
        ("lineage", "./meta/lineage.json"),
        ("attestation", "./meta/attestation.json"),
    ):
        rep[name] = argument
        bindings[name]["path"] = argument
    for name in ("lineage", "op_library", "head", "engine"):
        source = bindings[name]["expected_source"]
        source["path"] = bindings[
            "attestation" if name in ("lineage", "op_library") else "lineage"
        ]["path"]
    for name, plan_name in (
        ("op_library", "op_library"),
        ("head", "head_artifact"),
        ("engine", "backbone_engine"),
    ):
        raw = f"/elsewhere/{name}" if absolute_artifact else f"models/{name}"
        rep["detector"]["plan"][plan_name]["path"] = raw
        bindings[name]["path"] = raw if absolute_artifact else str(Path(root) / raw)
    args = (att, lineage, ["S1", "S2"], "shipping", "none", None)
    assert T.report_problems(rep, *args) == []
    bindings["head"]["path"] = "/other/head"
    assert any(
        "identity binding head differs" in p for p in T.report_problems(rep, *args)
    )


@pytest.mark.parametrize("root", [None, 7, []])
def test_current_report_requires_model_root(root: Any) -> None:
    rep, att, lineage = _good_report()
    rep["model_root"] = root
    problems = T.report_problems(
        rep, att, lineage, ["S1", "S2"], "shipping", "none", None
    )
    assert any("model_root" in p for p in problems)


@pytest.mark.parametrize("name", ["config", "lineage", "attestation"])
def test_current_report_metadata_argument_matches_observation(name: str) -> None:
    rep, att, lineage = _good_report()
    rep[name] = f"/different/{name}"
    problems = T.report_problems(
        rep, att, lineage, ["S1", "S2"], "shipping", "none", None
    )
    assert any(
        f"identity binding {name} differs from metadata argument" in p for p in problems
    )


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
    identity = _good_identity()
    report.write_text(
        json.dumps(
            {"format": T.TRACK_REPORT_FORMAT, "run_id": RUN_ID, "identity": identity}
        )
    )
    journal = {
        "format": T.JOURNAL_FORMAT,
        "run_id": RUN_ID,
        "entrypoint": "saccade_track",
        "state": "complete",
        "identity": identity,
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


def _replace_report(report: Path, journal: dict[str, Any], **changes: Any) -> None:
    record = json.loads(report.read_text())
    record.update(changes)
    report.write_text(json.dumps(record))
    journal["report"]["sha256"] = _sha(report.read_bytes())


def test_historical_journal_report_pair_retains_null_identity(tmp_path: Path) -> None:
    out, trace, report, journal = _completed_run(tmp_path)
    journal.update(format=T.LEGACY_JOURNAL_FORMAT, identity={"level": None})
    _replace_report(
        report, journal, format=T.LEGACY_TRACK_REPORT_FORMAT, identity={"level": None}
    )
    assert _problems(out, trace, report, journal) == []
    assert T.committed_sequences(journal, out, RUN_ID) == SEQS
    journal["identity"] = _good_identity()
    _replace_report(report, journal, identity=journal["identity"])
    assert any(
        "historical identity" in p for p in _problems(out, trace, report, journal)
    )


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("field", ["format", "run_id", "identity"])
def test_report_journal_divergence_with_correct_report_hash(
    tmp_path: Path, historical: bool, field: str
) -> None:
    out, trace, report, journal = _completed_run(tmp_path)
    if historical:
        journal.update(format=T.LEGACY_JOURNAL_FORMAT, identity={"level": None})
        _replace_report(
            report,
            journal,
            format=T.LEGACY_TRACK_REPORT_FORMAT,
            identity={"level": None},
        )
    changes = {
        "format": T.TRACK_REPORT_FORMAT if historical else T.LEGACY_TRACK_REPORT_FORMAT,
        "run_id": "f" * 32,
        "identity": _good_identity() if historical else {"level": None},
    }
    _replace_report(report, journal, **{field: changes[field]})
    assert journal["report"]["sha256"] == _sha(report.read_bytes())
    assert any(
        f"report/journal {field}" in p for p in _problems(out, trace, report, journal)
    )


@pytest.mark.parametrize("state", ["running", "failed"])
def test_checksum_identity_does_not_make_a_partial_run_complete(
    tmp_path: Path, state: str
) -> None:
    out, trace, report, journal = _completed_run(tmp_path)
    journal["state"] = state
    assert T.identity_problems(journal["identity"], complete=False) == []
    assert T.committed_sequences(journal, out, RUN_ID) == SEQS
    problems = _problems(out, trace, report, journal)
    assert any(f"state '{state}'" in p for p in problems)
    assert not any("identity" in p for p in problems)


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j.update(run_id="f" * 32),  # an earlier run's journal
        lambda j: j.update(state="running"),
        lambda j: j.update(state="failed"),
        lambda j: j.update(format="saccade.native_track_journal/v0"),
        lambda j: j.update(format=[]),
        lambda j: j.update(identity={"level": None}),
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


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        '{"format": "saccade.native_track_journal/v1", ',
        "\x00\xff",
        "[]",
        "null",
        '"complete"',
        "42",
        '{"state":"running","state":"complete"}',
        '{"failure":NaN}',
    ],
)
def test_corrupt_journal_is_a_problem(tmp_path: Path, text: str) -> None:
    out, trace, report, _ = _completed_run(tmp_path)
    (out / T.JOURNAL_NAME).write_bytes(text.encode("latin-1"))
    problems = T.journal_problems(out, RUN_ID, SEQS, report, trace)
    assert len(problems) == 1 and "journal" in problems[0]


@pytest.mark.parametrize(
    "sequences", [None, "S1", {"name": "S1"}, ["S1", "S2"], [None, None], 7]
)
def test_journal_sequences_not_a_list_of_objects(
    tmp_path: Path, sequences: Any
) -> None:
    out, trace, report, j = _completed_run(tmp_path)
    j["sequences"] = sequences
    problems = _problems(out, trace, report, j)
    assert len(problems) == 1 and "sequences" in problems[0]
    assert T.committed_sequences(j, out, RUN_ID) == []


@pytest.mark.parametrize("rec", ["r.json", ["sha256"], 7])
def test_journal_report_not_an_object(tmp_path: Path, rec: Any) -> None:
    out, trace, report, j = _completed_run(tmp_path)
    j["report"] = rec
    assert any("report" in p for p in _problems(out, trace, report, j))


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
    assert T.invocation_run_id(f"saccade_track: run_id {a}", "saccade_track") == a
    # Only the first line: anything printed before it is not this contract.
    assert T.invocation_run_id(f"warning\n{log}", "saccade_track") is None
    assert T.invocation_run_id(f"\n{log}", "saccade_track") is None
    assert T.invocation_run_id(f" saccade_track: run_id {a}\n", "saccade_track") is None
    # A second run id line anywhere, of either entrypoint, is two runs' log.
    two = log + f"saccade_track_measurement: run_id {b}\n"
    assert T.invocation_run_id(two, "saccade_track") is None


def _invalid_report_bytes(path: tuple[str, ...], value: Any) -> bytes:
    rep, _, _ = _good_report()
    rep.update(
        entrypoint="saccade_track_measurement",
        measurement=T.expected_measurement("none", None, "serial"),
    )
    node = rep
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return json.dumps(rep).encode()


@pytest.mark.parametrize(
    "payload,diagnostic",
    [
        (b"", "unreadable track report (JSONDecodeError"),
        (b"{", "unreadable track report (JSONDecodeError"),
        (b'{"format": "saccade.native_track_report/v3", ', "JSONDecodeError"),
        (b"\xff", "unreadable track report (UnicodeDecodeError"),
        (b"[]", "track report is a list, not an object"),
        (b"null", "track report is a NoneType, not an object"),
        (b'"complete"', "track report is a str, not an object"),
        (b"42", "track report is a int, not an object"),
        (b"true", "track report is a bool, not an object"),
        (b'{"run_id":"first","run_id":"second"}', "duplicate JSON key"),
        (b'{"loop_seconds":NaN}', "nonfinite JSON value"),
        (
            _invalid_report_bytes(("detector",), None),
            "report detector is not an object",
        ),
        (
            _invalid_report_bytes(("detector", "load", "constant_devices"), None),
            "head placement",
        ),
        (
            _invalid_report_bytes(("sequences",), []),
            "report sequences is not an object",
        ),
        (
            _invalid_report_bytes(("sequences",), {"S1": None}),
            "sequence stats is not an object",
        ),
        (
            _invalid_report_bytes(("sequences",), {"S1": {"frames": "10"}}),
            "sequence stats frames is not an integer",
        ),
    ],
)
def test_corrupt_report_run_is_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    diagnostic: str,
) -> None:
    """The report's bytes can match its journal and still be invalid evidence."""
    native, rows, output = (
        tmp_path / "native_run",
        tmp_path / "oracle_rows",
        tmp_path / "parity",
    )
    out = native / "native"
    out.mkdir(parents=True)
    (rows / "mot").mkdir(parents=True)
    output.mkdir()
    (rows / "oracle_rows.json").write_text(
        json.dumps({"ok": True, "mot17_argv": ["--sequences", "S1"]})
    )
    (rows / "mot" / "_global_id_map.txt").write_text("")
    report = native / "track_report.json"
    report.write_bytes(payload)
    txt = out / "S1.txt"
    txt.write_bytes(b"1,1,10,20,30,40,0.9,-1,-1,-1\n")
    journal = {
        "format": T.JOURNAL_FORMAT,
        "run_id": RUN_ID,
        "state": "complete",
        "identity": _good_identity(),
        "sequences": [
            {"name": "S1", "state": "written", "txt_sha256": _sha(txt.read_bytes())}
        ],
        "report": {"path": str(report), "sha256": _sha(payload)},
        "failure": None,
    }
    (out / T.JOURNAL_NAME).write_text(json.dumps(journal))
    (native / "saccade_track.log").write_text(
        f"saccade_track_measurement: run_id {RUN_ID}\nexit=0\n"
    )
    # Supply only the CPU evidence this malformed-report path consumes.
    _, att, lineage = _good_report()
    (tmp_path / "lineage.json").write_text(json.dumps(lineage))
    (tmp_path / "attestation.json").write_text(json.dumps(att))
    (tmp_path / "track_binary").write_bytes(b"synthetic binary")
    monkeypatch.setattr(T, "project_root", tmp_path)
    monkeypatch.setattr(T.det, "LINEAGE", "lineage.json")
    monkeypatch.setattr(T.det, "ATTESTATION", "attestation.json")
    monkeypatch.setattr(T.det, "read_attestation", lambda: att)
    monkeypatch.setattr(T.det, "_git_state", lambda: {"head": "test", "dirty": False})
    monkeypatch.setattr(T, "_load_module", lambda *_: T.argparse.Namespace())
    args = T.argparse.Namespace(
        out=output,
        native_from=native,
        oracle_rows=rows,
        oracle_txt=None,
        sequences=["S1"],
        schedule="serial",
        entrypoint="measurement",
        mutation="none",
        ref_edit=False,
        max_frames=None,
        no_trace=True,
        against=None,
        track_binary=Path("track_binary"),
        model_root=tmp_path,
        track_library_path=None,
    )
    assert T.run_parity(args) == 2
    result = json.loads((output / "report.json").read_text())
    assert result["verdict"] == "UNRESOLVED"
    assert any(str(report) in p and diagnostic in p for p in result["problems"])


def test_sequence_stats_nested_shapes_are_checked_before_comparison() -> None:
    stats = {
        key: 1
        for key in (
            "im_width",
            "im_height",
            "frames",
            "track_ids",
            "tracker_updates",
            "skipped_empty_frames",
            "pre_roll_updates",
            "hardware_decodes",
            "decoupled_decodes",
        )
    }
    stats.update(interpolation={}, loop_seconds=0.5, graphs=_native_graphs()["graphs"])
    assert T.sequence_stats_problems(stats, "serial") == []
    assert T.sequence_stats_problems(stats, "double_buffer") == []
    stats["graphs"] = None
    assert "graphs is not an object" in " ".join(
        T.sequence_stats_problems(stats, "double_buffer")
    )
    stats["loop_seconds"] = "0.5"
    assert "loop_seconds is not a number" in " ".join(
        T.sequence_stats_problems(stats, "serial")
    )


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
