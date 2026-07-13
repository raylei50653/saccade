#!/usr/bin/env python3
"""Canonicalize and persist H0's unlabeled four-stream bridge trace.

This tool deliberately consumes only trace records and a caller-supplied
sequence identity.  It neither opens GT/FP labels nor evaluates thresholds.
Raw CUDA append order and capture-run UUID are provenance; the semantic digest
is computed from stable-key sorted records only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "h0_bridge_decision_trace_v1"
STREAMS: dict[str, tuple[str, ...]] = {
    "pair_records": (
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "lost_slot",
        "lost_instance_uid",
    ),
    "candidate_records": ("seq", "frame", "cand_slot", "cand_instance_uid"),
    "claim_records": (
        "seq",
        "frame",
        "proposing_cand_slot",
        "proposing_cand_instance_uid",
        "proposed_lost_slot",
        "proposed_lost_instance_uid",
    ),
    "commit_records": (
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "lost_slot",
        "lost_instance_uid",
    ),
}
TOTAL_KEYS = {
    "pair_records": "total_pair_records",
    "candidate_records": "total_candidate_records",
    "claim_records": "total_claim_records",
    "commit_records": "total_commit_records",
}
OVERFLOW_KEYS = {
    "pair_records": "overflow_pair_records",
    "candidate_records": "overflow_candidate_records",
    "claim_records": "overflow_claim_records",
    "commit_records": "overflow_commit_records",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_key(stream: str, row: Mapping[str, Any]) -> tuple[object, ...]:
    try:
        return tuple(row[name] for name in STREAMS[stream])
    except KeyError as exc:
        raise ValueError(
            f"{stream} record missing stable-key field {exc.args[0]!r}"
        ) from exc


def canonical_semantic_packet(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate counters and return the UUID-free stable-key ordered packet."""
    if capture.get("capture_schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected H0 capture schema version")
    if int(capture.get("identity_uid_wrap_events", 0)) != 0:
        raise ValueError("track_instance_uid_v1 generation wrap detected")

    streams: dict[str, list[dict[str, Any]]] = {}
    for stream in STREAMS:
        raw_rows = capture.get(stream)
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError(f"{stream} must be a record list")
        rows = [dict(row) for row in raw_rows]
        overflow = int(capture.get(OVERFLOW_KEYS[stream], 0))
        total = int(capture.get(TOTAL_KEYS[stream], len(rows)))
        if overflow != 0:
            raise ValueError(f"{stream} overflow={overflow}")
        if total != len(rows):
            raise ValueError(
                f"{stream} total={total} does not equal drained records={len(rows)}"
            )
        rows.sort(key=lambda row: stable_key(stream, row))
        keys = [stable_key(stream, row) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{stream} has duplicate stable keys")
        if any(int(row["frame"]) <= 0 for row in rows):
            raise ValueError(
                f"{stream} contains a non-positive evaluator frame identity"
            )
        streams[stream] = rows

    return {
        "capture_schema_version": SCHEMA_VERSION,
        "streams": streams,
        "stream_totals": {stream: len(rows) for stream, rows in streams.items()},
    }


def semantic_digest(capture: Mapping[str, Any]) -> str:
    """SHA-256 of semantic fields only; raw order/UUID cannot affect it."""
    return hashlib.sha256(
        canonical_json(canonical_semantic_packet(capture))
    ).hexdigest()


def write_packet(capture: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write canonical stream JSON plus a provenance/semantic digest manifest."""
    packet = canonical_semantic_packet(capture)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for stream, rows in packet["streams"].items():
        path = output_dir / f"{stream}.json"
        path.write_bytes(canonical_json(rows) + b"\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "capture_schema_version": SCHEMA_VERSION,
        "capture_run_uuid": capture.get("capture_run_uuid"),
        "raw_stream_order_authoritative": False,
        "semantic_digest_sha256": semantic_digest(capture),
        "stream_totals": packet["stream_totals"],
        "files": files,
    }
    (output_dir / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.input_json.read_text(encoding="utf-8"))
    print(json.dumps(write_packet(capture, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
