"""Diagnostic validity checks; no production inference or incidence sampling."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.tools.capture_attribution.analyze import analyze
from scripts.tools.capture_attribution.run import install, label_for


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
