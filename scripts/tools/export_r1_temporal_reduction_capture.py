#!/usr/bin/env python3
"""Seal native shadow observations into the R1 temporal-reduction payload.

This is deliberately separate from ``export_d0_runtime_capture.py``. D0's
CSV schema and provenance are sealed evidence; R1 needs the exact two
``bridge_anchor4`` input windows (or the one-point short-lost fallback) as
nested, chronological state and therefore uses a versioned JSONL payload.

The input directory is written by an explicitly enabled evaluator run:

  SACCADE_RESEARCH_R1_TEMPORAL_REDUCTION_CAPTURE_DIR=out/r1-native \
  SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_SHADOW=1 \
    uv run scripts/eval/mot17.py --relink-bridge-enabled ...

An evaluator ``_global_id_map.txt`` may be supplied as optional provenance, but
R1 does not require a MOT-output join: a native bridge proposal can be valid
temporal state even when one endpoint was not emitted to MOT.

  uv run python scripts/tools/export_r1_temporal_reduction_capture.py \
    --capture-dir out/r1-native --id-map results/<run>/_global_id_map.txt \
    --output out/r1-temporal/events.jsonl

The tool reads no outcome labels and performs no score fitting. It rejects
missing state, non-shadow input, overflow, unresolved IDs, duplicate events,
or any stale/non-R1 capture provenance. It never drops a native event merely
because an output-layer global id is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from saccade.perception.eval.d_online_stage2 import parse_global_id_map


REPO = Path(__file__).resolve().parents[2]
CAPTURE_CONTRACT = "r1_temporal_reduction_capture_v1"
PAYLOAD_SCHEMA_VERSION = "r1_temporal_reduction_payload_v1"
PAYLOAD_KIND = "TemporalReductionContract"

_SCALAR_FIELDS: tuple[str, ...] = (
    "gap",
    "bridge_at",
    "la",
    "anchor_mode",
    "anchor_rate",
    "bdist",
    "dist_h",
    "fwd_r",
    "bwd_r",
    "v_lost_x",
    "v_lost_y",
    "v_cand_x",
    "v_cand_y",
    "ax",
    "ay",
    "cx0",
    "cy0",
    "ema_lost",
    "ema_cand",
    "h_ref",
    "s_lost",
    "w",
    "production_threshold",
    "bridge_dir_bonus",
    "lost_window_size",
    "cand_window_size",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: object, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}")
    return result


def _same_f32(left: object, right: object, *, field: str) -> bool:
    """Compare the exact float32 values consumed by the CUDA kernel."""
    lhs = _finite(left, field=field)
    rhs = _finite(right, field=field)
    return struct.pack("<f", lhs) == struct.pack("<f", rhs)


def _window(value: object, *, field: str, consumed: int) -> list[list[float]]:
    if consumed not in {1, 4}:
        raise ValueError(f"{field}_size must be 1 or 4, got {consumed}")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be a 4-sample chronological window")
    result: list[list[float]] = []
    for index, sample in enumerate(value):
        if not isinstance(sample, list) or len(sample) != 3:
            raise ValueError(f"{field}[{index}] must be [cx, cy, h]")
        result.append([_finite(v, field=f"{field}[{index}]") for v in sample])
    # CUDA zeroes the unused tail. Rejecting nonzero values prevents a consumer
    # from accidentally treating unconsumed ring history as causal R1 input.
    if consumed == 1 and any(any(v != 0.0 for v in sample) for sample in result[1:]):
        raise ValueError(f"{field} has nonzero samples outside the short-lost fallback")
    return result


def _load_capture_dir(
    capture_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    files = sorted(
        path for path in capture_dir.glob("*.json") if path.name != "manifest.json"
    )
    if not files:
        raise ValueError(f"no per-sequence capture JSON files in {capture_dir}")

    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    provenance_ref: dict[str, Any] | None = None
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("events")
        total = int(payload.get("total_events", -1))
        if (
            payload.get("complete") is not True
            or int(payload.get("overflow_events", -1)) != 0
        ):
            raise ValueError(f"incomplete native capture: {path}")
        if not isinstance(rows, list) or total != len(rows):
            raise ValueError(f"event count mismatch in native capture: {path}")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"native capture lacks provenance: {path}")
        if provenance.get("capture_contract") != CAPTURE_CONTRACT:
            raise ValueError(f"not an R1 capture contract: {path}")
        if provenance.get("shadow") is not True:
            raise ValueError(f"R1 capture must be shadow mode: {path}")
        if provenance_ref is None:
            provenance_ref = provenance
        elif provenance != provenance_ref:
            raise ValueError(f"mixed R1 capture provenance: {path}")
        for ordinal, row in enumerate(rows):
            if row.get("capture_mode") != "runtime_cuda_event_ring":
                raise ValueError(f"non-native capture row in {path}")
            if row.get("evidence_role") != "runtime_cuda_observation":
                raise ValueError(f"invalid native evidence role in {path}")
            event = dict(row)
            # CUDA's append cursor gives every copied record a unique buffer
            # position. Keep that native observation ordinal: a graph-captured
            # scalar frame argument is not a trustworthy event identity.
            event["native_capture_ordinal"] = ordinal
            events.append(event)
        sources.append({"path": str(path), "sha256": _sha256(path), "events": total})
    assert provenance_ref is not None
    return events, provenance_ref, sources


def _records(
    events: Iterable[dict[str, Any]],
    id_map: dict[tuple[str, int], int],
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    bridge = provenance.get("bridge")
    if not isinstance(bridge, dict):
        raise ValueError("R1 provenance lacks bridge configuration")
    for row in events:
        seq = str(row.get("seq", "")).strip()
        if not seq:
            raise ValueError("R1 row lacks sequence identity")
        try:
            lost_local = int(row["lost_id"])
            cand_local = int(row["cand_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"R1 row lacks integer local ids: {row}") from exc
        try:
            native_capture_ordinal = int(row["native_capture_ordinal"])
            lost_slot = int(row["lost_slot"])
            cand_slot = int(row["cand_slot"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"R1 row lacks native event identity: {row}") from exc
        if native_capture_ordinal < 0 or lost_slot < 0 or cand_slot < 0:
            raise ValueError(
                f"invalid R1 native event identity: {seq} ordinal={native_capture_ordinal}"
            )
        # The global map is an evaluator-output convenience, not native bridge
        # identity. In particular, a shadow proposal can reference a valid
        # track that was never emitted to MOT. Preserve such state rather than
        # laundering it into an incomplete sample.
        lost_global = id_map.get((seq, lost_local))
        cand_global = id_map.get((seq, cand_local))

        scalar = {field: row[field] for field in _SCALAR_FIELDS if field in row}
        if set(scalar) != set(_SCALAR_FIELDS):
            missing = sorted(set(_SCALAR_FIELDS) - set(scalar))
            raise ValueError(f"R1 row is missing fields: {missing}")
        for field in _SCALAR_FIELDS:
            if field in {
                "gap",
                "bridge_at",
                "la",
                "anchor_mode",
                "lost_window_size",
                "cand_window_size",
            }:
                scalar[field] = int(scalar[field])
            else:
                scalar[field] = _finite(scalar[field], field=field)

        if scalar["cand_window_size"] != 4:
            raise ValueError(
                "R1 candidate bridge_anchor4 window must have exactly 4 samples"
            )
        if scalar["la"] != scalar["gap"] + scalar["bridge_at"] - 1:
            raise ValueError("R1 la must equal gap + bridge_at - 1")
        anchor_modes = {"center": 0, "foot": 1, "adaptive": 2}
        provenance_anchor = bridge.get("anchor")
        if provenance_anchor not in anchor_modes:
            raise ValueError("R1 provenance bridge.anchor is invalid")
        if scalar["bridge_at"] != int(bridge.get("at", -1)):
            raise ValueError("R1 row/config bridge_at mismatch")
        if scalar["anchor_mode"] != anchor_modes[provenance_anchor]:
            raise ValueError("R1 row/config anchor_mode mismatch")
        for field, provenance_field in (
            ("anchor_rate", "anchor_rate"),
            ("production_threshold", "px"),
            ("bridge_dir_bonus", "dir_bonus"),
        ):
            if not _same_f32(
                scalar[field],
                bridge.get(provenance_field),
                field=f"provenance.bridge.{provenance_field}",
            ):
                raise ValueError(f"R1 row/config {field} mismatch")

        lost_window = _window(
            row.get("lost_anchor_window"),
            field="lost_anchor_window",
            consumed=scalar["lost_window_size"],
        )
        cand_window = _window(
            row.get("cand_anchor_window"),
            field="cand_anchor_window",
            consumed=scalar["cand_window_size"],
        )
        lost_branch = (
            "bridge_anchor4_last4"
            if scalar["lost_window_size"] == 4
            else "short_lost_last_point_zero_velocity"
        )
        record = {
            "payload_kind": PAYLOAD_KIND,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "capture_contract": CAPTURE_CONTRACT,
            "event_key": (
                f"{seq}|capture_ordinal={native_capture_ordinal}|lost_slot={lost_slot}|"
                f"cand_slot={cand_slot}"
            ),
            "event_key_fields": [
                "seq",
                "native_capture_ordinal",
                "lost_slot",
                "cand_slot",
            ],
            "seq": seq,
            "native_capture_ordinal": native_capture_ordinal,
            "lost_slot": lost_slot,
            "cand_slot": cand_slot,
            "lost_global_id": lost_global,
            "cand_global_id": cand_global,
            "lost_local_id": lost_local,
            "cand_local_id": cand_local,
            "lost_reduction": {
                "branch": lost_branch,
                "consumed_samples": scalar.pop("lost_window_size"),
                "chronological_cx_cy_h": lost_window,
            },
            "candidate_reduction": {
                "branch": "bridge_anchor4_head4",
                "consumed_samples": scalar.pop("cand_window_size"),
                "chronological_cx_cy_h": cand_window,
            },
            "kernel_terms": scalar,
        }
        records.append(record)

    records.sort(key=lambda record: str(record["event_key"]))
    seen: set[str] = set()
    duplicates = [
        str(record["event_key"])
        for record in records
        if str(record["event_key"]) in seen or seen.add(str(record["event_key"]))
    ]
    if duplicates:
        raise ValueError(f"duplicate R1 event keys: {duplicates[:3]}")
    return records


def export_r1_capture(
    capture_dir: Path, output: Path, *, id_map_path: Path | None = None
) -> dict[str, Any]:
    """Write deterministic R1 JSONL and its provenance manifest."""
    events, provenance, sources = _load_capture_dir(capture_dir)
    id_map = parse_global_id_map(id_map_path) if id_map_path is not None else {}
    if id_map_path is not None and not id_map:
        raise ValueError(f"empty global id map: {id_map_path}")
    records = _records(events, id_map, provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(_canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "payload_kind": PAYLOAD_KIND,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "capture_contract": CAPTURE_CONTRACT,
        "payload": str(output),
        "payload_sha256": _sha256(output),
        "events": len(records),
        "overflow_events": 0,
        "exporter": {
            "path": "scripts/tools/export_r1_temporal_reduction_capture.py",
            "sha256": _sha256(Path(__file__)),
        },
        "id_map": (
            {"path": str(id_map_path), "sha256": _sha256(id_map_path)}
            if id_map_path is not None
            else None
        ),
        "events_without_global_id": sum(
            1
            for record in records
            if record["lost_global_id"] is None or record["cand_global_id"] is None
        ),
        "provenance": provenance,
        "sources": sources,
    }
    (output.parent / f"{output.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--id-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else REPO / path

    manifest = export_r1_capture(
        absolute(args.capture_dir),
        absolute(args.output),
        id_map_path=absolute(args.id_map) if args.id_map is not None else None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
