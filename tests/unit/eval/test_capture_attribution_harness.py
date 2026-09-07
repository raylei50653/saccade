"""Diagnostic validity checks; no production inference or incidence sampling."""

# scope: eval
# function: contract
# lifecycle: active

import hashlib
import inspect
import json
import threading
import time
from types import SimpleNamespace

import pytest

from scripts.tools.capture_attribution.analyze import analyze
import scripts.tools.capture_attribution.run as run_module
from scripts.tools.capture_attribution.run import (
    install,
    label_for,
    quiesce,
    teardown_log,
)


def test_wrapper_preserves_arguments_return_and_exception():
    calls, emitted, native_ids = [], [], []
    failure = RuntimeError("original end error")

    class Graph:
        def capture_begin(self, pool=None, capture_error_mode="global"):
            calls.append((pool, capture_error_mode))
            return 42

        def capture_end(self):
            raise failure

    original_begin, original_end = Graph.capture_begin, Graph.capture_end
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            CUDAGraph=Graph,
            current_stream=lambda: SimpleNamespace(cuda_stream=123),
            current_device=lambda: 0,
        )
    )
    restore = install(
        fake,
        SimpleNamespace(attribution_site=native_ids.append),
        lambda event, **fields: emitted.append((event, fields)),
    )
    try:
        graph = Graph()
        pool = (7, 8)
        assert graph.capture_begin(pool, "relaxed") == 42
        assert calls == [(pool, "relaxed")]
        with pytest.raises(RuntimeError) as caught:
            graph.capture_end()
        assert caught.value is failure
        assert emitted[0][1]["mode"] == "relaxed"
        assert emitted[-1][0] == "python_end_error"
        assert native_ids[-1] == 0
    finally:
        restore()
    assert Graph.capture_begin is original_begin
    assert Graph.capture_end is original_end


def test_tracker_stack_is_classified_without_repo_wrapper():
    assert (
        label_for(
            [
                {
                    "file": "/repo/src/saccade/perception/tracking/tracker_gpu.py",
                    "function": "__init__",
                }
            ],
            [],
        )
        == "tracker.update"
    )
    assert label_for([], []) == "unclassified.python"


def fixture_rows():
    common = {
        "tid": 1,
        "domain": 1,
        "context": 88,
        "stream": 99,
        "flags": 0,
        "flags_source": "query",
        "mode": 1,
        "has_stream": True,
        "site_id": 0,
        "native_stack": [],
    }
    return [
        {
            **common,
            "seq": n,
            "ns": n * 10,
            "cbid": cb,
            "correlation": corr,
            "api": api,
            "phase": phase,
            "rc": rc,
        }
        for n, cb, corr, api, phase, rc in (
            (1, 10, 1, "cuStreamBeginCapture_v2", "enter", -1),
            (2, 10, 1, "cuStreamBeginCapture_v2", "exit", 0),
            (3, 11, 2, "cuStreamIsCapturing", "enter", -1),
            (4, 11, 2, "cuStreamIsCapturing", "exit", 906),
            (5, 12, 3, "cuStreamEndCapture", "enter", -1),
            (6, 12, 3, "cuStreamEndCapture", "exit", 0),
        )
    ]


def write_fixture(root, rows):
    (root / "cuda.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "python.jsonl").write_text(
        json.dumps({"event": "harness_stopped", "cupti_rc": 0, "live_threads": []})
        + "\n"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source_drift": [],
                "artifacts_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.iterdir()
                    if p.name != "manifest.json"
                },
            }
        )
    )


def test_overlap_is_not_root_cause_or_external_library_proof(tmp_path):
    rows = fixture_rows()
    for row in rows[2:4]:
        row.update(tid=2, stream=1)
    write_fixture(tmp_path, rows)
    result = analyze(tmp_path)
    assert result["trace_structure_ok"]
    assert len(result["capture_errors"][0]["observed_open_captures"]) == 1
    assert result["unclassified_captures"] == 1
    assert not result["root_cause_closed"]


@pytest.mark.parametrize(
    "damage", ["no_capture", "missing_end", "unknown_flags", "hash"]
)
def test_missing_or_tampered_evidence_cannot_pass(tmp_path, damage):
    rows = fixture_rows()
    if damage == "no_capture":
        rows = []
    elif damage == "missing_end":
        rows = rows[:-1]
    elif damage == "unknown_flags":
        rows[1]["flags"] = -1
    write_fixture(tmp_path, rows)
    if damage == "hash":
        with (tmp_path / "cuda.jsonl").open("a") as output:
            output.write("\n")
        # Valid JSONL modification, rather than a parser error.
        text = (tmp_path / "cuda.jsonl").read_text().replace('"flags": 0', '"flags": 1')
        (tmp_path / "cuda.jsonl").write_text(text.rstrip() + "\n")
    assert not analyze(tmp_path)["trace_structure_ok"]


def test_different_context_is_not_attributed(tmp_path):
    rows = fixture_rows()
    for row in rows[2:4]:
        row["context"] = 100
    write_fixture(tmp_path, rows)
    assert analyze(tmp_path)["capture_errors"][0]["observed_open_captures"] == []


def test_missing_query_enter_cannot_pass(tmp_path):
    rows = fixture_rows()
    rows.pop(2)
    for n, row in enumerate(rows, 1):
        row.update(seq=n, selected=True)
    write_fixture(tmp_path, rows)
    assert not analyze(tmp_path)["trace_structure_ok"]


def stopped_row(**overrides):
    return {"event": "harness_stopped", "cupti_rc": 0, "live_threads": [], **overrides}


def write_teardown_fixture(root, stopped, final_manifest=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "cuda.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fixture_rows())
    )
    (root / "python.jsonl").write_text(json.dumps(stopped) + "\n")
    (root / "tail.log").write_text("1 workload_completed:target_returned\n")
    manifest = {"source_drift": []}
    if final_manifest:
        excluded = ("manifest.json", "tail.log")
        manifest["artifacts_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.iterdir()
            if p.name not in excluded
        }
        manifest["artifacts_sha256_excluded"] = list(excluded)
    (root / "manifest.json").write_text(json.dumps(manifest))


class StubWorker:
    """Stands in for a library-owned auxiliary worker that outlives the target."""

    def __init__(self, stoppable=True):
        self.released = threading.Event()
        self.stoppable = stoppable
        self.thread = threading.Thread(target=self.released.wait, daemon=True)

    def shutdown(self):
        if self.stoppable:
            self.released.set()
            self.thread.join(5)


@pytest.fixture
def owned_workers(monkeypatch):
    """Stand in for the two shutdowns quiesce drives, in the order it drives them."""
    started = []

    def install(compile_workers, monitor=lambda: None):
        monkeypatch.setattr(run_module, "_shutdown_compile_workers", compile_workers)
        monkeypatch.setattr(run_module, "_shutdown_tqdm_monitor", monitor)
        return started

    def worker(stoppable=True):
        made = StubWorker(stoppable)
        made.thread.start()
        started.append(made)
        return made

    yield install, worker
    for made in started:
        made.released.set()
    for thread in threading.enumerate():
        if thread.name.startswith("quiesce:"):
            thread.join(5)


def test_quiescence_lets_a_worker_stop_before_the_observer_does(owned_workers):
    install, spawn = owned_workers
    worker = spawn()
    install(worker.shutdown)
    quiesced = quiesce(30.0)
    assert not worker.thread.is_alive()
    assert quiesced["errors"] == {}
    assert not quiesced["timed_out"]
    # Bounded means it returns when the worker stops, not when the deadline does.
    assert quiesced["seconds"] < 30.0


def test_a_worker_that_will_not_stop_leaves_the_trace_structure_invalid(
    tmp_path, owned_workers
):
    install, spawn = owned_workers
    worker = spawn(stoppable=False)
    install(worker.shutdown)
    quiesced = quiesce(0.5)
    assert quiesced["timed_out"]
    assert worker.thread.is_alive()
    # The bound is a bound, not an excuse: the surviving worker still reaches the
    # shutdown check, which fails closed on it exactly as before.
    live = [{"name": worker.thread.name, "native_id": worker.thread.native_id}]
    write_teardown_fixture(tmp_path, stopped_row(quiesce=quiesced, live_threads=live))
    result = analyze(tmp_path)
    assert not result["trace_structure_ok"]
    assert "observer_shutdown_incomplete_or_workers_alive" in result["problems"]


def test_a_cleanup_error_is_recorded_rather_than_raised_or_dropped(owned_workers):
    def refuse():
        raise RuntimeError("compile pool refused")

    install, spawn = owned_workers
    worker = spawn()
    # The failing shutdown is the one quiesce drives first, so a skipped
    # remainder would show up as the worker still running.
    install(refuse, worker.shutdown)
    quiesced = quiesce(30.0)
    assert "compile pool refused" in quiesced["errors"]["inductor_compile_workers"]
    # Recorded, not raised and not dropped: a teardown convenience is not
    # allowed to destroy the trace it is serving, or to hide that it failed.
    assert not worker.thread.is_alive()
    assert not quiesced["timed_out"]


def test_a_shutdown_that_never_returns_is_bounded_and_still_fails_closed(
    tmp_path, owned_workers
):
    install, spawn = owned_workers
    blocked = threading.Event()
    install(blocked.wait)
    try:
        started = time.monotonic()
        quiesced = quiesce(0.5)
        # The bound has to enclose the shutdown call itself, not just the wait
        # that follows it, or a shutdown that never returns hangs teardown with
        # the timeout never reaching it.
        assert time.monotonic() - started < 5.0
        assert quiesced["timed_out"]
        # The thread still executing that shutdown is not exempt from the
        # accounting: it reaches live_threads and fails the trace closed like
        # any other survivor, so a stuck shutdown cannot pass structure.
        live = [
            {"name": t.name, "native_id": t.native_id}
            for t in threading.enumerate()
            if t is not threading.current_thread()
        ]
        assert "quiesce:inductor_compile_workers" in [e["name"] for e in live]
        write_teardown_fixture(
            tmp_path, stopped_row(quiesce=quiesced, live_threads=live)
        )
        result = analyze(tmp_path)
        assert not result["trace_structure_ok"]
        assert "observer_shutdown_incomplete_or_workers_alive" in result["problems"]
    finally:
        blocked.set()


def test_teardown_records_progress_that_outlives_a_missing_manifest(tmp_path):
    tail, note = teardown_log(tmp_path)
    note("workload_completed:target_returned")
    note("quiesce_begin:timeout=60.0")
    # Readable before close, so a teardown that dies here still names its last step.
    lines = (tmp_path / "tail.log").read_text().splitlines()
    assert [line.split(maxsplit=1)[1] for line in lines] == [
        "workload_completed:target_returned",
        "quiesce_begin:timeout=60.0",
    ]
    assert [int(line.split(maxsplit=1)[0]) for line in lines] == sorted(
        int(line.split(maxsplit=1)[0]) for line in lines
    )
    tail.close()
    with pytest.raises(FileExistsError):
        teardown_log(tmp_path)


def test_run_notes_every_teardown_stage_the_tail_has_to_locate():
    teardown = inspect.getsource(run_module.run).split("finally:")[1]
    for stage in (
        "workload_completed",
        "quiesce_begin",
        "quiesce_end",
        "attribution_stopped",
        "hashing_mapped_files",
        "mapped_files_hashed",
        "manifest_written",
    ):
        assert f'note(f"{stage}' in teardown or f'note("{stage}' in teardown


def test_final_manifest_is_what_marks_legal_finalization(tmp_path):
    write_teardown_fixture(tmp_path / "died", stopped_row(), final_manifest=False)
    result = analyze(tmp_path / "died")
    assert not result["trace_structure_ok"]
    assert "missing_final_manifest" in result["problems"]
    # The two still-open files are excluded by name, not silently: a complete
    # teardown passes with tail.log present and deliberately unhashed.
    write_teardown_fixture(tmp_path / "complete", stopped_row())
    assert analyze(tmp_path / "complete")["trace_structure_ok"]
