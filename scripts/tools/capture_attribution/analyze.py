"""Validate trace structure and attribute observed errors without exclusion claims."""

# status: diagnostic

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(root: Path) -> dict:
    rows = read_rows(root / "cuda.jsonl")
    python = read_rows(root / "python.jsonl")
    manifest = json.loads((root / "manifest.json").read_text())
    problems = []
    for name, expected in manifest.get("artifacts_sha256", {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            problems.append(f"artifact_hash_mismatch:{name}")
    if not manifest.get("artifacts_sha256"):
        problems.append("missing_final_manifest")
    if manifest.get("source_drift"):
        problems.append("source_drift")
    if manifest.get("asset_drift") or manifest.get("asset_inventory_added"):
        problems.append("asset_drift")
    if [r["seq"] for r in rows] != list(range(1, len(rows) + 1)):
        problems.append("noncontiguous_event_sequence")
    stopped = [r for r in python if r["event"] == "harness_stopped"]
    if len(stopped) != 1 or stopped[0]["cupti_rc"] != 0 or stopped[0]["live_threads"]:
        problems.append("observer_shutdown_incomplete_or_workers_alive")
    sites = {r["site_id"]: r for r in python if r["event"] == "python_begin_enter"}
    pending = {}
    intervals = []
    open_captures = {}
    errors = []
    event_records = {}
    event_edges = []
    status_observations = []
    for row in rows:
        api = row["api"]
        capture = "StreamBeginCapture" in api or "StreamBeginRecapture" in api
        end = "StreamEndCapture" in api
        call = (row["tid"], row["domain"], row["cbid"], row["correlation"])
        if row["phase"] == "enter":
            if call in pending:
                problems.append(f"duplicate_api_enter:{call}")
            pending[call] = row
            if end:
                key = (row["domain"], row["context"], row["stream"])
                if key in open_captures:
                    open_captures[key]["end_enter_ns"] = row["ns"]
            continue
        entered = pending.pop(call, None)
        if entered is None and row.get("selected", False):
            problems.append(f"selected_api_exit_without_enter:{row['seq']}")
        if row["rc"] != 0:
            errors.append(row)
        if row["rc"] == 0 and row.get("status", -1) >= 0:
            status_observations.append(row)
        event_key = (row["domain"], row["context"], row.get("event"))
        if row["rc"] == 0 and row.get("event"):
            if api.startswith(("cudaEventDestroy", "cuEventDestroy")):
                event_records.pop(event_key, None)
            elif api.startswith(("cudaEventRecord", "cuEventRecord")):
                event_records[event_key] = row
            elif "StreamWaitEvent" in api:
                event_edges.append(
                    {
                        "wait": row,
                        "last_observed_record": event_records.get(event_key),
                        "interpretation": "observed event edge; check external flags and capture status",
                    }
                )
        if capture:
            if not entered:
                problems.append(f"capture_without_enter:{row['seq']}")
            if not row["has_stream"] or row["flags"] < 0 or row["mode"] < 0:
                problems.append(f"unknown_capture_metadata:{row['seq']}")
            if row["rc"] != 0:
                continue
            key = (row["domain"], row["context"], row["stream"])
            if key in open_captures:
                problems.append(f"overlapping_capture_same_stream:{row['seq']}")
            record = {
                "domain": row["domain"],
                "context": row["context"],
                "stream": row["stream"],
                "flags": row["flags"],
                "flags_source": row["flags_source"],
                "mode": row["mode"],
                "tid": row["tid"],
                "site_id": row["site_id"],
                "label": sites.get(row["site_id"], {}).get(
                    "label", "unclassified.native"
                ),
                "begin_enter_ns": entered["ns"] if entered else None,
                "begin_exit_ns": row["ns"],
                "end_enter_ns": None,
                "end_exit_ns": None,
                "end_rc": None,
                "native_stack": entered["native_stack"] if entered else [],
            }
            intervals.append(record)
            open_captures[key] = record
        if end:
            key = (row["domain"], row["context"], row["stream"])
            record = open_captures.pop(key, None)
            if record is None:
                # A failed begin legitimately may have an end attempt; retain it
                # as a gap requiring inspection rather than inventing a capture.
                problems.append(f"unmatched_capture_end:{row['seq']}")
            else:
                record["end_exit_ns"], record["end_rc"] = row["ns"], row["rc"]
    if pending:
        problems.append(f"unpaired_api_calls:{len(pending)}")
    if open_captures:
        problems.append(f"unclosed_captures:{len(open_captures)}")
    if not intervals:
        problems.append("no_successful_capture_observed")
    correlations = []
    for error in errors:
        if error["rc"] not in (900, 901, 906):
            continue
        matching = [
            i
            for i in intervals
            if i["context"] == error["context"]
            and i["begin_exit_ns"] <= error["ns"]
            and (i["end_enter_ns"] is None or error["ns"] <= i["end_enter_ns"])
        ]
        correlations.append(
            {
                "error": error,
                "observed_open_captures": matching,
                "interpretation": "temporal overlap, not proof of causation",
            }
        )
    return {
        "trace_structure_ok": not problems,
        "problems": problems,
        "scope": "observed callbacks only; runtime/driver rows may describe the same capture",
        "api_counts": dict(Counter(r["api"] for r in rows if r["phase"] == "enter")),
        "captures": intervals,
        "event_edges": event_edges,
        "stream_status_observations": status_observations,
        "capture_errors": correlations,
        "unclassified_captures": sum(
            i["label"].startswith("unclassified") for i in intervals
        ),
        "native_stack_note": "addresses resolve against maps.txt; unloaded modules may be unresolved",
        "root_cause_closed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    result = analyze(args.trace)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["trace_structure_ok"] else 1)
