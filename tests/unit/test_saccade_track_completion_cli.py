"""``saccade_track`` run completion at the real entrypoint, without a GPU (#536).

docs/architecture/ship_export_contracts_536.md CC-536-01-01. The protocol
itself (failed reruns, kills inside a sequence and between a rename and the
journal update, a second process) is pinned CUDA-free by
``tests/native/test_shipping_run_completion.cpp``; these pin that the built
entrypoints use it, before any model or sequence is read (the lineage given
here does not exist, so every run stops in the CUDA-free detector plan):

* stderr's first line is ``<entrypoint>: run_id <32 hex>``, a new id per
  invocation, and the journal and the refusal carry it;
* a held ``<out>`` (another process's flock on ``saccade_track.lock``): exit 2,
  and no file under ``<out>`` or at ``--report`` changes;
* ``--report`` on the journal is refused before ``<out>`` is created;
* a failed rerun removes the earlier report (outside ``<out>``), the earlier
  ``<seq>.txt`` of its sequences and the earlier journal, and leaves other
  files in ``<out>`` alone.

By default, skips when ``build/shipping/saccade_track`` has not been built.
Set ``SACCADE_SHIPPING_TEST_PREFIX`` to an installed tree to run its
``bin/saccade_track`` launcher with its installed config. An explicit prefix
must be usable; it never skips or falls back to a build binary. Installed mode
collects only shipping cases, since the measurement tool is not installed.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build" / "shipping"
_PREFIX = os.environ.get("SACCADE_SHIPPING_TEST_PREFIX")
PREFIX = Path(_PREFIX).expanduser().resolve() if _PREFIX else None
INSTALLED = _PREFIX is not None
MODEL_ROOT = PREFIX / "share" / "saccade" if PREFIX else REPO
CONFIG = MODEL_ROOT / "configs" / "shipping" / "mamba_whole_graph.resolved.json"
BINARIES = (
    ["saccade_track"] if INSTALLED else ["saccade_track", "saccade_track_measurement"]
)
JOURNAL = "saccade_track.journal.json"
LOCK = "saccade_track.lock"

pytestmark = pytest.mark.skipif(
    not INSTALLED and not (BUILD / "saccade_track").exists(),
    reason="build/shipping/saccade_track not built",
)


@pytest.fixture(scope="module", autouse=True)
def _require_installed_tree() -> None:
    if INSTALLED:
        assert PREFIX is not None, (
            "SACCADE_SHIPPING_TEST_PREFIX must name an installed tree"
        )
        launcher = PREFIX / "bin" / "saccade_track"
        assert launcher.is_file() and os.access(launcher, os.X_OK), launcher
        assert (PREFIX / "libexec" / "saccade_track").is_file(), PREFIX
        assert CONFIG.is_file(), CONFIG


def _run(
    tmp_path: Path, binary: str = "saccade_track", report: Path | None = None
) -> subprocess.CompletedProcess[str]:
    exe = PREFIX / "bin" / binary if PREFIX else BUILD / binary
    if not exe.exists():
        if INSTALLED:
            pytest.fail(f"installed entrypoint {exe} is missing")
        pytest.skip(f"{exe.relative_to(REPO)} not built")
    cmd = [
        str(exe),
        "--config",
        str(CONFIG),
        "--lineage",
        str(tmp_path / "no_such.lineage.json"),
        "--out",
        str(tmp_path / "out"),
    ]
    if report is not None:
        cmd += ["--report", str(report)]
    cmd += [str(tmp_path / "MOT17-02-SDP"), str(tmp_path / "MOT17-04-SDP")]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)


def _run_id(r: subprocess.CompletedProcess[str], name: str = "saccade_track") -> str:
    m = re.fullmatch(rf"{name}: run_id ([0-9a-f]{{32}})", r.stderr.splitlines()[0])
    assert m, r.stderr
    return m.group(1)


def _snapshot(*roots: Path) -> dict[str, tuple[int, bytes]]:
    out = {}
    for root in roots:
        for p in [root, *root.rglob("*")] if root.is_dir() else [root]:
            if p.is_file():
                out[str(p)] = (p.stat().st_mtime_ns, p.read_bytes())
    return out


@pytest.mark.parametrize("binary", BINARIES)
def test_run_id_is_the_first_line_and_names_the_journal(
    tmp_path: Path, binary: str
) -> None:
    a, b = _run(tmp_path, binary), _run(tmp_path, binary)
    assert a.returncode == 2 and b.returncode == 2
    ida, idb = _run_id(a, binary), _run_id(b, binary)
    assert ida != idb
    j = json.loads((tmp_path / "out" / JOURNAL).read_text())
    assert j["format"] == (
        "saccade.native_track_journal/v1"
        if INSTALLED
        else "saccade.native_track_journal/v2"
    )
    assert (j["run_id"], j["entrypoint"], j["state"]) == (idb, binary, "failed")
    assert j["identity"]["level"] is None
    if INSTALLED:
        assert j["identity"] == {"level": None}
    else:
        assert j["identity"]["bindings"]["lineage"]["status"] == "missing"
        assert j["identity"]["bindings"]["config"]["observed_sha256"] is not None
    assert [(s["name"], s["state"]) for s in j["sequences"]] == [
        ("MOT17-02-SDP", "pending"),
        ("MOT17-04-SDP", "pending"),
    ]
    assert j["failure"]["sequence"] is None
    assert f"{binary}: {j['failure']['message']}" in b.stderr.splitlines()


def test_held_out_changes_nothing(tmp_path: Path) -> None:
    out, report = tmp_path / "out", tmp_path / "track_report.json"
    first = _run(tmp_path, report=report)
    assert first.returncode == 2
    report.write_text("an earlier report")
    (out / "MOT17-02-SDP.txt").write_text("an earlier txt")
    before = _snapshot(out, report)
    with (out / LOCK).open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = _run(tmp_path, report=report)
    assert r.returncode == 2
    rid = _run_id(r)
    assert "is in use by another run" in r.stderr
    assert _snapshot(out, report) == before
    assert json.loads((out / JOURNAL).read_text())["run_id"] == _run_id(first) != rid


def test_report_on_the_journal_is_refused_before_out(tmp_path: Path) -> None:
    r = _run(tmp_path, report=tmp_path / "out" / JOURNAL)
    assert r.returncode == 2
    _run_id(r)
    assert "is the journal, the lock or another output" in r.stderr
    assert not (tmp_path / "out").exists()


def test_failed_rerun_leaves_no_earlier_output(tmp_path: Path) -> None:
    out, report = tmp_path / "out", tmp_path / "track_report.json"
    out.mkdir()
    report.write_text('{"format": "saccade.native_track_report/v3"}')
    (out / "MOT17-02-SDP.txt").write_text("an earlier run's txt")
    (out / "MOT17-04-SDP.txt").write_text("an earlier run's txt")
    (out / "MOT17-09-SDP.txt").write_text("not this run's sequence")
    (out / JOURNAL).write_text('{"run_id": "earlier", "state": "complete"}')
    r = _run(tmp_path, report=report)
    assert r.returncode == 2
    j = json.loads((out / JOURNAL).read_text())
    assert j["run_id"] == _run_id(r) and j["state"] == "failed"
    assert not report.exists()
    assert not (out / "MOT17-02-SDP.txt").exists()
    assert not (out / "MOT17-04-SDP.txt").exists()
    assert (out / "MOT17-09-SDP.txt").read_text() == "not this run's sequence"
