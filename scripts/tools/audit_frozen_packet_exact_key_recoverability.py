#!/usr/bin/env python3
"""EK0 frozen-packet exact-key recoverability audit (pure consistency audit).

Scope: within the frozen D0 capture packet only, does any unjoined runtime
event contradict its own partition label — i.e. is it recoverable through the
exact v2 event key (or its redundant canonical-field triple) into the frozen
offline pair universe, or is its identity provenance ambiguous?  For a
well-formed v2 packet both are unreachable by the exporter's partition
definitions, so the audit is a consistency check that pins the counts.

The audit never reads GT/outcome columns: classification uses only identity,
event provenance, frozen offline pair membership, and frozen coordinate
availability.  It is single-phase; the packet manifest seals the declaration,
this runner, the inventory, and the metrics by SHA256, and a completed packet
is immutable — reruns against it fail closed without touching it.

The audit claims nothing about wider runtime joins that would expand the
offline universe, add identity observability, or re-capture, and it carries
no statistical exposure machinery: presence/absence of recoverable rows is
the entire outcome.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
STUDY = "ek0_frozen_packet_exact_key_recoverability_20260713"
DECLARATION = (
    "docs/modules/semantic/research/"
    "frozen_packet_exact_key_recoverability_declaration_20260713.md"
)
RUNNER_RELPATH = "scripts/tools/audit_frozen_packet_exact_key_recoverability.py"
STUDY_DIR = "out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z"
SUBSTRATE_DIR = "results/MOT17_eval_d0_shadow_substrate_20260712T085642Z"

PARTITIONS = ("matched", "cohort_gap", "unemitted")
TARGET_PARTITIONS = ("cohort_gap", "unemitted")
EXPECTED_PARTITION = {"matched": 1684, "cohort_gap": 539, "unemitted": 354}
EVENT_KEY_VERSION = "d0_event_key_v2_global"
EVENT_KEY_FIELDS = ("seq", "lost_global_id", "cand_global_id")
RECONSTRUCTABLE = {"exact-key reconstructable"}
AMBIGUOUS = "provenance ambiguous"
LABEL_INCONSISTENT = "partition label inconsistent"

EXPECTED_INPUT_HASHES = {
    "pairs.csv": "ee2898a25ef7f01ed46331c49c12d667846975f25769bc4c3e6b8bad493f8e87",
    "capture.csv.gz": "96093b9b723ed4500b389f8ad74600d75bb49a75064630dd2205cea0b0887047",
    "capture.csv.gz.manifest.json": (
        "4547ed29df726497e1050bda7326044a573579d7268571bd29a4c107bd0d8d99"
    ),
    "_global_id_map.txt": "ae3b6441d1712bcce0826d611cee2cfdf7a01b4d37ec331336f91a0b9148f366",
    "substrate_mot_concat": "4c5e322a3b8c026de584baa883e26353720837ffa2bf146dfcef2679426a670e",
}


class AuditInvalid(RuntimeError):
    """A frozen provenance, partition, or immutability condition failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive_finite(value: str) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _require_columns(
    header: list[str] | None, required: Iterable[str], source: str
) -> dict[str, int]:
    if header is None:
        raise AuditInvalid(f"{source} is missing a CSV header")
    index = {name: i for i, name in enumerate(header)}
    missing = sorted(set(required) - set(index))
    if missing:
        raise AuditInvalid(f"{source} missing columns: {missing}")
    return index


def _parse_int(value: str, *, field: str, ordinal: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AuditInvalid(
            f"capture row {ordinal} has invalid {field}: {value!r}"
        ) from exc


def _read_capture_identity(capture_path: Path) -> list[dict[str, Any]]:
    """Read only runtime identity/provenance/coordinate columns; capture has no GT."""
    needed = {
        "event_key",
        "event_key_version",
        "partition",
        "seq",
        "lost_global_id",
        "cand_global_id",
        "lost_local_id",
        "cand_local_id",
        "dist_h",
        "ema_lost",
        "ema_cand",
    }
    records: list[dict[str, Any]] = []
    with gzip.open(capture_path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        index = _require_columns(next(reader, None), needed, "capture.csv.gz")
        for ordinal, values in enumerate(reader, start=1):
            if len(values) != len(index) and len(values) < max(index.values()) + 1:
                raise AuditInvalid(f"capture row {ordinal} is truncated")
            partition = values[index["partition"]]
            if partition not in PARTITIONS:
                raise AuditInvalid(
                    f"capture row {ordinal} unknown partition: {partition!r}"
                )
            event_key = values[index["event_key"]]
            records.append(
                {
                    "event_ordinal": ordinal,
                    "partition": partition,
                    "seq": values[index["seq"]],
                    "event_key": event_key,
                    "event_key_version": values[index["event_key_version"]],
                    "lost_global_id": _parse_int(
                        values[index["lost_global_id"]],
                        field="lost_global_id",
                        ordinal=ordinal,
                    ),
                    "cand_global_id": _parse_int(
                        values[index["cand_global_id"]],
                        field="cand_global_id",
                        ordinal=ordinal,
                    ),
                    "lost_local_id_present": bool(values[index["lost_local_id"]]),
                    "cand_local_id_present": bool(values[index["cand_local_id"]]),
                    "runtime_coordinates_available": (
                        _finite(values[index["dist_h"]])
                        and _positive_finite(values[index["ema_lost"]])
                        and _positive_finite(values[index["ema_cand"]])
                    ),
                    "runtime_dist_h": values[index["dist_h"]],
                    "runtime_ema_lost": values[index["ema_lost"]],
                    "runtime_ema_cand": values[index["ema_cand"]],
                }
            )
    return records


def _read_offline_universe_blind(
    pairs_path: Path,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], set[tuple[str, int, int]], int]:
    """Project pairs.csv without reading any GT/outcome column values."""
    needed = {"seq", "lost_id", "cand_id", "dist_h", "h_lost_raw", "h_cand_raw"}
    universe: dict[tuple[str, int, int], dict[str, Any]] = {}
    nonunique: set[tuple[str, int, int]] = set()
    rows = 0
    with pairs_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        index = _require_columns(next(reader, None), needed, "pairs.csv")
        width = max(index.values()) + 1
        for ordinal, values in enumerate(reader, start=1):
            rows += 1
            if len(values) < width:
                raise AuditInvalid(f"pairs row {ordinal} is truncated")
            try:
                key = (
                    values[index["seq"]],
                    int(values[index["lost_id"]]),
                    int(values[index["cand_id"]]),
                )
            except ValueError as exc:
                raise AuditInvalid(f"pairs row {ordinal} has invalid identity") from exc
            row = {
                "offline_coordinates_available": (
                    _finite(values[index["dist_h"]])
                    and _positive_finite(values[index["h_lost_raw"]])
                    and _positive_finite(values[index["h_cand_raw"]])
                )
            }
            if key in universe:
                nonunique.add(key)
            else:
                universe[key] = row
    return universe, nonunique, rows


def _mot_concat_sha256(substrate_dir: Path) -> str:
    mot_files = sorted(substrate_dir.glob("MOT17-*.txt"))
    if not mot_files:
        raise AuditInvalid(f"no MOT17 substrate files in {substrate_dir}")
    digest = hashlib.sha256()
    for path in mot_files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_j1(*, study_dir: Path, substrate_dir: Path) -> dict[str, Any]:
    """Reproduce the frozen input hashes, partition, and event-key version."""
    paths = {
        "pairs.csv": study_dir / "pairs.csv",
        "capture.csv.gz": study_dir / "capture.csv.gz",
        "_global_id_map.txt": substrate_dir / "_global_id_map.txt",
    }
    observed: dict[str, str] = {}
    failures: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing {name}: {path}")
            continue
        observed[name] = _sha256(path)
        if observed[name] != EXPECTED_INPUT_HASHES[name]:
            failures.append(f"{name} hash mismatch")
    observed["substrate_mot_concat"] = _mot_concat_sha256(substrate_dir)
    if (
        observed["substrate_mot_concat"]
        != EXPECTED_INPUT_HASHES["substrate_mot_concat"]
    ):
        failures.append("substrate_mot_concat hash mismatch")

    capture_manifest_path = study_dir / "capture.csv.gz.manifest.json"
    if not capture_manifest_path.is_file():
        failures.append(f"missing capture manifest: {capture_manifest_path}")
    else:
        observed["capture.csv.gz.manifest.json"] = _sha256(capture_manifest_path)
        if (
            observed["capture.csv.gz.manifest.json"]
            != EXPECTED_INPUT_HASHES["capture.csv.gz.manifest.json"]
        ):
            # The sidecar carries shadow/overflow/partition assertions; its
            # fields must not be trusted unless its frozen hash reproduces.
            failures.append("capture.csv.gz.manifest.json hash mismatch")
        else:
            capture_manifest = json.loads(
                capture_manifest_path.read_text(encoding="utf-8")
            )
            if capture_manifest.get("event_key_version") != EVENT_KEY_VERSION:
                failures.append("event-key version mismatch")
            if tuple(capture_manifest.get("event_key_fields", [])) != EVENT_KEY_FIELDS:
                failures.append("event-key field tuple mismatch")
            if capture_manifest.get("partition") != EXPECTED_PARTITION:
                failures.append("capture-manifest partition mismatch")
            if capture_manifest.get("provenance", {}).get("shadow") is not True:
                failures.append("capture provenance is not shadow")
            if int(capture_manifest.get("overflow_events", -1)) != 0:
                failures.append("capture overflow is non-zero")

    if any(name not in observed for name in paths) or failures:
        raise AuditInvalid("J1 provenance failed: " + "; ".join(failures))

    capture = _read_capture_identity(study_dir / "capture.csv.gz")
    partition = {
        name: sum(row["partition"] == name for row in capture) for name in PARTITIONS
    }
    if partition != EXPECTED_PARTITION or sum(partition.values()) != len(capture):
        failures.append(f"capture partition conservation mismatch: {partition}")
    if any(row["event_key_version"] != EVENT_KEY_VERSION for row in capture):
        failures.append("capture row event-key version mismatch")
    if failures:
        raise AuditInvalid("J1 provenance failed: " + "; ".join(failures))

    return {
        "passed": True,
        "input_hashes": observed,
        "partition": partition,
        "partition_total": len(capture),
        "event_key": {"version": EVENT_KEY_VERSION, "fields": list(EVENT_KEY_FIELDS)},
        "capture_shadow": True,
        "capture_overflow_events": 0,
    }


def classify_event(
    event: dict[str, Any],
    *,
    offline_universe: dict[tuple[str, int, int], dict[str, Any]],
    offline_nonunique: set[tuple[str, int, int]],
    duplicated_event_keys: set[str],
) -> tuple[str, str, str, bool]:
    """Return frozen J2 class, reason, offline status, coordinate availability.

    Partition-aware: each event must first satisfy the exporter shape its own
    partition label asserts (``cohort_gap`` = resolved IDs + canonical key +
    pair not enumerated offline; ``unemitted`` = unresolved identity).  A
    cross-label shape is a packet defect, distinct from recoverability.
    """
    if event["event_key_version"] != EVENT_KEY_VERSION:
        return (
            "provenance ambiguous",
            "event_key_version_mismatch",
            "not_checked",
            False,
        )
    key = (event["seq"], event["lost_global_id"], event["cand_global_id"])
    has_global_identity = event["lost_global_id"] >= 0 and event["cand_global_id"] >= 0
    expected_event_key = "|".join(str(part) for part in key)
    event_key = event["event_key"]
    if event_key and event_key in duplicated_event_keys:
        return (
            "provenance ambiguous",
            "duplicate_capture_event_key",
            "not_checked",
            False,
        )
    if has_global_identity and event_key and event_key != expected_event_key:
        return (
            "provenance ambiguous",
            "event_key_global_field_inconsistency",
            "not_checked",
            False,
        )
    if not has_global_identity and event_key:
        return (
            "provenance ambiguous",
            "keyed_row_has_unresolved_global_identity",
            "not_checked",
            False,
        )
    if event["partition"] == "unemitted":
        if has_global_identity:
            return (
                LABEL_INCONSISTENT,
                "unemitted_row_has_resolved_global_identity",
                "not_checked",
                False,
            )
        return (
            "structurally unjoinable",
            "unresolved_global_identity_no_local_id_fallback",
            "absent",
            False,
        )
    if not has_global_identity:
        return (
            LABEL_INCONSISTENT,
            "cohort_gap_row_has_unresolved_global_identity",
            "not_checked",
            False,
        )
    if not event_key:
        return (
            LABEL_INCONSISTENT,
            "cohort_gap_row_missing_canonical_event_key",
            "not_checked",
            False,
        )
    if key in offline_nonunique:
        return (
            "provenance ambiguous",
            "nonunique_offline_pair_identity",
            "not_checked",
            False,
        )
    offline = offline_universe.get(key)
    if offline is None:
        return (
            "structurally unjoinable",
            "same_global_pair_absent_from_offline_universe",
            "absent",
            False,
        )
    # The pair being enumerated offline already refutes the cohort_gap label,
    # whether or not the frozen coordinates would be usable downstream.
    coordinates = bool(
        event["runtime_coordinates_available"]
        and offline["offline_coordinates_available"]
    )
    if not coordinates:
        return (
            LABEL_INCONSISTENT,
            "cohort_gap_pair_enumerated_without_frozen_coordinates",
            "unique",
            False,
        )
    return "exact-key reconstructable", "exact_v2_global_pair_key", "unique", True


def _sequence_distribution(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for sequence in sorted({row["seq"] for row in records}):
        rows = [row for row in records if row["seq"] == sequence]
        identified = {
            (row["seq"], row["lost_global_id"])
            for row in rows
            if row["lost_global_id"] >= 0
        }
        reconstructable = {
            (row["seq"], row["lost_global_id"])
            for row in rows
            if row["classification"] in RECONSTRUCTABLE
        }
        out[sequence] = {
            "events": len(rows),
            "identified_unique_lost_tracks": len(identified),
            "reconstructable_unique_lost_tracks": len(reconstructable),
        }
    return out


def reduce_j3(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the event inventory to descriptive (seq, lost_global_id) counts.

    Purely descriptive: EK0 carries no exposure or feasibility arithmetic, so
    nothing here feeds a statistical envelope.
    """
    partitions: dict[str, Any] = {}
    for partition in TARGET_PARTITIONS:
        rows = [row for row in records if row["partition"] == partition]
        identified = {
            (row["seq"], row["lost_global_id"])
            for row in rows
            if row["lost_global_id"] >= 0
        }
        reconstructable = {
            (row["seq"], row["lost_global_id"])
            for row in rows
            if row["classification"] in RECONSTRUCTABLE
        }
        events_with_identity = sum(row["lost_global_id"] >= 0 for row in rows)
        partitions[partition] = {
            "events": len(rows),
            "events_with_identified_lost_track": events_with_identity,
            "identified_unique_lost_tracks": len(identified),
            "reconstructable_unique_lost_tracks": len(reconstructable),
            "repeat_events_after_lost_track_reduction": events_with_identity
            - len(identified),
            "repeat_rate_among_identified_events": (
                (events_with_identity - len(identified)) / events_with_identity
                if events_with_identity
                else None
            ),
            "unidentified_events": len(rows) - events_with_identity,
            "classification": dict(
                sorted(Counter(row["classification"] for row in rows).items())
            ),
            "reasons": dict(sorted(Counter(row["reason"] for row in rows).items())),
            "by_sequence": _sequence_distribution(rows),
        }
    return {
        "trial_unit": "(seq, lost_global_id)",
        "partitions": partitions,
        "reconstructable_events": sum(
            row["classification"] in RECONSTRUCTABLE for row in records
        ),
        "provenance_ambiguous_events": sum(
            row["classification"] == AMBIGUOUS for row in records
        ),
        "partition_label_inconsistent_events": sum(
            row["classification"] == LABEL_INCONSISTENT for row in records
        ),
    }


def determine_terminal(j3: dict[str, Any]) -> str:
    """Apply EK0's ordered, exhaustive terminal mapping.

    ``EK0_INVALID`` is raised out-of-band via ``AuditInvalid``; every valid run
    lands in exactly one of the two remaining terminals.
    """
    if (
        j3["reconstructable_events"]
        or j3["provenance_ambiguous_events"]
        or j3["partition_label_inconsistent_events"]
    ):
        return "EK0_PACKET_INCONSISTENT"
    return "EK0_NO_RECOVERABLE_SUPPORT"


def _write_inventory(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "event_ordinal",
        "partition",
        "seq",
        "event_key",
        "lost_global_id",
        "cand_global_id",
        "lost_local_id_present",
        "cand_local_id_present",
        "runtime_coordinates_available",
        "runtime_dist_h",
        "runtime_ema_lost",
        "runtime_ema_cand",
        "offline_pair_status",
        "frozen_coordinate_pair_available",
        "classification",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: record[name] for name in fields} for record in records)


def _require_mutable_output(output_dir: Path) -> None:
    """A completed packet is immutable; refuse to run against it."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if existing.get("phase") == "complete":
        raise AuditInvalid(
            f"output dir holds a completed immutable packet: {manifest_path}"
        )


def run_audit(
    *, study_dir: Path, substrate_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Single-phase J1–J3 consistency audit.  No GT column is ever read."""
    _require_mutable_output(output_dir)
    j1 = verify_j1(study_dir=study_dir, substrate_dir=substrate_dir)
    capture = _read_capture_identity(study_dir / "capture.csv.gz")
    universe, nonunique, offline_rows = _read_offline_universe_blind(
        study_dir / "pairs.csv"
    )
    target = [row for row in capture if row["partition"] in TARGET_PARTITIONS]
    event_keys = [row["event_key"] for row in target if row["event_key"]]
    duplicated = {key for key, count in Counter(event_keys).items() if count > 1}
    inventory: list[dict[str, Any]] = []
    for row in target:
        classification, reason, pair_status, coordinates = classify_event(
            row,
            offline_universe=universe,
            offline_nonunique=nonunique,
            duplicated_event_keys=duplicated,
        )
        inventory.append(
            {
                **row,
                "classification": classification,
                "reason": reason,
                "offline_pair_status": pair_status,
                "frozen_coordinate_pair_available": coordinates,
            }
        )
    j3 = reduce_j3(inventory)
    terminal = determine_terminal(j3)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "inventory.csv"
    _write_inventory(inventory_path, inventory)
    inventory_hash = _sha256(inventory_path)
    metrics = {
        "study": STUDY,
        "phase": "complete",
        "terminal": terminal,
        "gt_label_accessed": False,
        "j1_provenance": j1,
        "j2_classification": {
            "target_partitions": {
                name: sum(row["partition"] == name for row in inventory)
                for name in TARGET_PARTITIONS
            },
            "offline_pair_rows_scanned_without_gt_projection": offline_rows,
            "offline_nonunique_pair_identities": len(nonunique),
            "duplicate_target_event_keys": len(duplicated),
            "classification_rules": "sealed declaration §2",
        },
        "j3_reduction": j3,
        "inventory_sha256": inventory_hash,
    }
    metrics_path = output_dir / "metrics.json"
    _json_write(metrics_path, metrics)
    manifest = {
        "study": STUDY,
        "phase": "complete",
        "terminal": terminal,
        "declaration": DECLARATION,
        "declaration_sha256": _sha256(_absolute(DECLARATION)),
        "runner": RUNNER_RELPATH,
        "runner_sha256": _sha256(Path(__file__)),
        "input_hashes": j1["input_hashes"],
        "event_key": j1["event_key"],
        "seal": {
            "inventory_sha256": inventory_hash,
            "gt_label_accessed": False,
            "classification_rules": "sealed declaration §2",
        },
        "files": {
            "inventory.csv": inventory_hash,
            "metrics.json": _sha256(metrics_path),
        },
    }
    _json_write(output_dir / "manifest.json", manifest)
    return metrics


def _invalid_packet(output_dir: Path, failure: str) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("phase") == "complete":
            # Never clobber a completed packet; the failure is reported only.
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        output_dir / "metrics.json",
        {
            "study": STUDY,
            "phase": "invalid",
            "terminal": "EK0_INVALID",
            "failure": failure,
        },
    )
    _json_write(
        output_dir / "manifest.json",
        {
            "study": STUDY,
            "phase": "invalid",
            "terminal": "EK0_INVALID",
            "declaration": DECLARATION,
            "files": {"metrics.json": _sha256(output_dir / "metrics.json")},
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path(STUDY_DIR))
    parser.add_argument("--substrate-dir", type=Path, default=Path(SUBSTRATE_DIR))
    args = parser.parse_args(argv)
    output_dir = _absolute(args.output_dir)
    try:
        result = run_audit(
            study_dir=_absolute(args.study_dir),
            substrate_dir=_absolute(args.substrate_dir),
            output_dir=output_dir,
        )
    except AuditInvalid as exc:
        _invalid_packet(output_dir, str(exc))
        print(json.dumps({"terminal": "EK0_INVALID", "failure": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "study": STUDY,
                "phase": result["phase"],
                "terminal": result["terminal"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
