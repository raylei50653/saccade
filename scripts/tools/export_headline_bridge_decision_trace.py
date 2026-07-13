#!/usr/bin/env python3
"""Canonicalize H0 records plus its independent native-universe sidecar.

The tool consumes H0 outputs and a caller-supplied sequence identity only. It
neither opens GT/FP labels nor evaluates thresholds. Raw CUDA append order and
capture-run UUID are provenance; the semantic digest uses canonical key order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_PATH = Path(__file__).with_name("h0_bridge_decision_trace_schema_v2.json")
CAPTURE_SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
SCHEMA_VERSION = str(CAPTURE_SCHEMA["capture_schema_version"])
RECORD_FIELDS: dict[str, tuple[str, ...]] = {
    stream: tuple(fields)
    for stream, fields in dict(CAPTURE_SCHEMA["record_fields"]).items()
}
STREAMS: dict[str, tuple[str, ...]] = {
    stream: tuple(fields)
    for stream, fields in dict(CAPTURE_SCHEMA["stable_keys"]).items()
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
UNIVERSE_STREAMS: dict[str, tuple[str, ...]] = {
    stream: tuple(fields)
    for stream, fields in dict(CAPTURE_SCHEMA["native_universe_keys"]).items()
}
UNIVERSE_TOTAL_KEYS = {
    "native_candidate_keys": "total_native_candidate_keys",
    "native_pair_keys": "total_native_pair_keys",
    "native_proposal_keys": "total_native_proposal_keys",
    "native_claim_winner_keys": "total_native_claim_winner_keys",
    "native_commit_keys": "total_native_commit_keys",
}
UNIVERSE_OVERFLOW_KEYS = {
    "native_candidate_keys": "overflow_native_candidate_keys",
    "native_pair_keys": "overflow_native_pair_keys",
    "native_proposal_keys": "overflow_native_proposal_keys",
    "native_claim_winner_keys": "overflow_native_claim_winner_keys",
    "native_commit_keys": "overflow_native_commit_keys",
}
OBSERVED_UNIVERSE: dict[str, str] = dict(
    CAPTURE_SCHEMA["native_universe_observed_stream"]
)
ENVELOPE_FIELDS: tuple[str, ...] = tuple(CAPTURE_SCHEMA["envelope_fields"])
SEMANTIC_ENVELOPE_FIELDS = (
    "trace_armed",
    "bridge_attempt_count",
    "bridge_commit_count",
    "capture_phase",
    "require_candidate_exposure",
    "require_commit_exposure",
    "identity_uid_wrap_events",
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_key(
    stream: str, fields: tuple[str, ...], row: Mapping[str, Any]
) -> tuple[object, ...]:
    try:
        return tuple(row[name] for name in fields)
    except KeyError as exc:
        raise ValueError(
            f"{stream} record missing stable-key field {exc.args[0]!r}"
        ) from exc


def stable_key(stream: str, row: Mapping[str, Any]) -> tuple[object, ...]:
    return _stable_key(stream, STREAMS[stream], row)


def universe_key(stream: str, row: Mapping[str, Any]) -> tuple[object, ...]:
    return _stable_key(stream, UNIVERSE_STREAMS[stream], row)


def _validate_record_schema(stream: str, row: Mapping[str, Any]) -> None:
    expected = set(RECORD_FIELDS[stream])
    actual = set(row)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{stream} field schema mismatch: missing={missing}, unexpected={unexpected}"
        )


def _validate_envelope(capture: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in ENVELOPE_FIELDS if field not in capture]
    if missing:
        raise ValueError(f"H0 capture envelope missing required fields: {missing}")
    if capture["capture_schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected H0 capture schema version")
    if (
        not isinstance(capture["capture_run_uuid"], str)
        or not capture["capture_run_uuid"].strip()
    ):
        raise ValueError("H0 capture envelope has no capture_run_uuid")
    if capture["trace_armed"] is not True:
        raise ValueError("H0 capture envelope is not trace_armed")
    if capture["capture_phase"] not in {"phase_a", "phase_b"}:
        raise ValueError("H0 capture envelope has an invalid capture_phase")
    for field in ("require_candidate_exposure", "require_commit_exposure"):
        if not isinstance(capture[field], bool):
            raise ValueError(f"H0 capture envelope {field} must be boolean")
    if (
        capture["capture_phase"] == "phase_a"
        and not capture["require_candidate_exposure"]
    ):
        raise ValueError("Phase A capture must require candidate exposure")
    if capture["capture_phase"] == "phase_b" and (
        not capture["require_candidate_exposure"]
        or not capture["require_commit_exposure"]
    ):
        raise ValueError("Phase B capture must require candidate and commit exposure")
    counter_fields = {
        "processed_frame_count",
        "bridge_attempt_count",
        "bridge_commit_count",
        "identity_uid_wrap_events",
    }
    invalid_counter_types = [
        field
        for field in counter_fields
        if not isinstance(capture[field], int) or isinstance(capture[field], bool)
    ]
    if invalid_counter_types:
        raise ValueError(
            f"H0 capture envelope counters must be integers: {invalid_counter_types}"
        )
    try:
        envelope = {
            "trace_armed": capture["trace_armed"],
            "processed_frame_count": int(capture["processed_frame_count"]),
            "bridge_attempt_count": int(capture["bridge_attempt_count"]),
            "bridge_commit_count": int(capture["bridge_commit_count"]),
            "capture_phase": capture["capture_phase"],
            "require_candidate_exposure": capture["require_candidate_exposure"],
            "require_commit_exposure": capture["require_commit_exposure"],
            "identity_uid_wrap_events": int(capture["identity_uid_wrap_events"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "H0 capture envelope has a non-integer native counter"
        ) from exc
    if envelope["processed_frame_count"] <= 0:
        raise ValueError("H0 capture envelope has no processed-frame provenance")
    if envelope["bridge_attempt_count"] < 0 or envelope["bridge_commit_count"] < 0:
        raise ValueError("H0 capture envelope has a negative production exposure count")
    if envelope["identity_uid_wrap_events"] != 0:
        raise ValueError("track_instance_uid_v1 generation wrap detected")
    return envelope


def canonical_semantic_packet(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate counters and return the UUID-free stable-key ordered packet."""
    envelope = _validate_envelope(capture)

    streams: dict[str, list[dict[str, Any]]] = {}
    for stream in STREAMS:
        raw_rows = capture[stream]
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError(f"{stream} must be a record list")
        if not all(isinstance(row, Mapping) for row in raw_rows):
            raise ValueError(f"{stream} must contain mapping records")
        rows = [dict(row) for row in raw_rows]
        for row in rows:
            _validate_record_schema(stream, row)
        overflow = int(capture[OVERFLOW_KEYS[stream]])
        total = int(capture[TOTAL_KEYS[stream]])
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

    native_universe: dict[str, list[dict[str, Any]]] = {}
    for stream in UNIVERSE_STREAMS:
        raw_rows = capture[stream]
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError(f"{stream} must be a key list")
        rows = [dict(row) for row in raw_rows]
        overflow = int(capture[UNIVERSE_OVERFLOW_KEYS[stream]])
        total = int(capture[UNIVERSE_TOTAL_KEYS[stream]])
        if overflow != 0:
            raise ValueError(f"{stream} overflow={overflow}")
        if total != len(rows):
            raise ValueError(
                f"{stream} total={total} does not equal drained keys={len(rows)}"
            )
        rows.sort(key=lambda row: universe_key(stream, row))
        keys = [universe_key(stream, row) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{stream} has duplicate native keys")
        if any(int(row["frame"]) <= 0 for row in rows):
            raise ValueError(
                f"{stream} contains a non-positive evaluator frame identity"
            )
        native_universe[stream] = rows

    for universe_stream, record_stream in OBSERVED_UNIVERSE.items():
        expected = [
            universe_key(universe_stream, row)
            for row in native_universe[universe_stream]
        ]
        observed = [stable_key(record_stream, row) for row in streams[record_stream]]
        if expected != observed:
            raise ValueError(
                f"{universe_stream} does not equal observed {record_stream} keys"
            )

    observed_winners = sorted(
        stable_key("claim_records", row)
        for row in streams["claim_records"]
        if int(row["claim_won"]) == 1
    )
    expected_winners = [
        universe_key("native_claim_winner_keys", row)
        for row in native_universe["native_claim_winner_keys"]
    ]
    if expected_winners != observed_winners:
        raise ValueError(
            "native_claim_winner_keys does not equal observed winning claim keys"
        )
    if envelope["bridge_attempt_count"] != len(
        native_universe["native_candidate_keys"]
    ):
        raise ValueError(
            "bridge_attempt_count does not equal native candidate exposure keys"
        )
    if envelope["bridge_commit_count"] != len(native_universe["native_commit_keys"]):
        raise ValueError(
            "bridge_commit_count does not equal native commit exposure keys"
        )
    if envelope["require_candidate_exposure"] and envelope["bridge_attempt_count"] == 0:
        raise ValueError("H0 capture required candidate exposure but observed none")
    if envelope["require_commit_exposure"] and envelope["bridge_commit_count"] == 0:
        raise ValueError("H0 capture required commit exposure but observed none")

    return {
        "capture_schema_version": SCHEMA_VERSION,
        "envelope": envelope,
        "streams": streams,
        "stream_totals": {stream: len(rows) for stream, rows in streams.items()},
        "native_universe": native_universe,
        "native_universe_totals": {
            stream: len(rows) for stream, rows in native_universe.items()
        },
    }


def semantic_digest(capture: Mapping[str, Any]) -> str:
    """SHA-256 of semantic fields only; raw order/provenance cannot affect it."""
    packet = canonical_semantic_packet(capture)
    semantic_packet = dict(packet)
    semantic_packet["envelope"] = {
        field: packet["envelope"][field] for field in SEMANTIC_ENVELOPE_FIELDS
    }
    return hashlib.sha256(canonical_json(semantic_packet)).hexdigest()


def write_packet(capture: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write canonical stream JSON plus a provenance/semantic digest manifest."""
    packet = canonical_semantic_packet(capture)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for stream, rows in packet["streams"].items():
        path = output_dir / f"{stream}.json"
        path.write_bytes(canonical_json(rows) + b"\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for stream, rows in packet["native_universe"].items():
        path = output_dir / f"{stream}.json"
        path.write_bytes(canonical_json(rows) + b"\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "capture_schema_version": SCHEMA_VERSION,
        "capture_run_uuid": capture["capture_run_uuid"],
        "raw_stream_order_authoritative": False,
        "semantic_digest_sha256": semantic_digest(capture),
        "stream_totals": packet["stream_totals"],
        "native_universe_totals": packet["native_universe_totals"],
        "envelope": packet["envelope"],
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
