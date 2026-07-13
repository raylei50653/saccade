#!/usr/bin/env python3
"""RJ0 runtime-join support sufficiency audit.

The audit is intentionally two-phase.  ``blind`` verifies frozen provenance and
seals an outcome-blind inventory.  ``reveal`` refuses to run unless that exact
inventory and the sealed RJ0 declaration are intact; it may read GT labels only
for the inventory's already-reconstructable rows.
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

from scipy import stats


REPO = Path(__file__).resolve().parents[2]
STUDY = "rj0_runtime_join_support_sufficiency_20260713"
DECLARATION = (
    "docs/modules/semantic/research/"
    "runtime_join_support_sufficiency_declaration_20260713.md"
)
S0_PACKET = (
    "docs/modules/semantic/research/evidence/s0_safe_domain_runtime_transfer_20260713"
)
STUDY_DIR = "out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z"
SUBSTRATE_DIR = "results/MOT17_eval_d0_shadow_substrate_20260712T085642Z"

BASE_N = 116
BASE_K = 3
EPSILON = 0.05
CONFIDENCE = 0.95
CP_ZERO_HURT_FLOOR = 153

PARTITIONS = ("matched", "cohort_gap", "unemitted")
TARGET_PARTITIONS = ("cohort_gap", "unemitted")
EXPECTED_PARTITION = {"matched": 1684, "cohort_gap": 539, "unemitted": 354}
EVENT_KEY_VERSION = "d0_event_key_v2_global"
EVENT_KEY_FIELDS = ("seq", "lost_global_id", "cand_global_id")
RECONSTRUCTABLE = {
    "exact-key reconstructable",
    "deterministic auxiliary-key reconstructable",
}

EXPECTED_INPUT_HASHES = {
    "pairs.csv": "ee2898a25ef7f01ed46331c49c12d667846975f25769bc4c3e6b8bad493f8e87",
    "capture.csv.gz": "96093b9b723ed4500b389f8ad74600d75bb49a75064630dd2205cea0b0887047",
    "_global_id_map.txt": "ae3b6441d1712bcce0826d611cee2cfdf7a01b4d37ec331336f91a0b9148f366",
    "substrate_mot_concat": "4c5e322a3b8c026de584baa883e26353720837ffa2bf146dfcef2679426a670e",
}
EXPECTED_S0_FILES = {
    "grid.csv": "300bdd6ea670f4c0ea236d69d3c625cfe9a4da2c71ccbbbc24b30bacec471e04",
    "metrics.json": "296fb996f65b9ec3c2a845962e8dfde6cedafe9d32bdf746933840c5e2531009",
}
EXPECTED_S0_RUNNER_SHA256 = (
    "07d4eb40433e915a4335f44756ab2e0a3e4b8d5b5f86bfe190193070867da4e2"
)


class AuditInvalid(RuntimeError):
    """A frozen provenance or pre-GT seal condition failed."""


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


def _as_bool01(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise AuditInvalid(f"invalid frozen GT boolean encoding: {value!r}")


def clopper_pearson_upper(hurt: int, exposed: int) -> float:
    """One-sided 95% Clopper–Pearson upper bound at lost-track unit."""
    if exposed <= 0 or hurt < 0 or hurt > exposed:
        raise ValueError(f"invalid binomial counts: hurt={hurt}, exposed={exposed}")
    if hurt == exposed:
        return 1.0
    return float(stats.beta.ppf(CONFIDENCE, hurt + 1, exposed - hurt))


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


def _read_offline_gt(
    pairs_path: Path, keys: set[tuple[str, int, int]]
) -> dict[tuple[str, int, int], dict[str, str]]:
    """J4-only projection.  This is the sole GT-label reader in the program."""
    needed = {
        "seq",
        "lost_id",
        "cand_id",
        "gt_valid",
        "gt_match",
        "dist_h",
        "h_lost_raw",
        "h_cand_raw",
    }
    found: dict[tuple[str, int, int], dict[str, str]] = {}
    with pairs_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader.fieldnames, needed, "pairs.csv")
        for row in reader:
            key = (row["seq"], int(row["lost_id"]), int(row["cand_id"]))
            if key in keys:
                if key in found:
                    raise AuditInvalid(f"J4 found non-unique frozen pair key: {key}")
                found[key] = row
    missing = keys - set(found)
    if missing:
        raise AuditInvalid(
            f"J4 inventory key absent from frozen pair universe: {sorted(missing)[:3]}"
        )
    return found


def _mot_concat_sha256(substrate_dir: Path) -> str:
    mot_files = sorted(substrate_dir.glob("MOT17-*.txt"))
    if not mot_files:
        raise AuditInvalid(f"no MOT17 substrate files in {substrate_dir}")
    digest = hashlib.sha256()
    for path in mot_files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_j1(
    *, study_dir: Path, substrate_dir: Path, s0_packet: Path
) -> dict[str, Any]:
    """Reproduce S0/D0 frozen inputs, artifact hashes, partition, and key version."""
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

    s0_manifest_path = s0_packet / "manifest.json"
    if not s0_manifest_path.is_file():
        failures.append(f"missing S0 manifest: {s0_manifest_path}")
        s0_manifest: dict[str, Any] = {}
    else:
        s0_manifest = json.loads(s0_manifest_path.read_text(encoding="utf-8"))
        if s0_manifest.get("input_hashes") != EXPECTED_INPUT_HASHES:
            failures.append("S0 manifest input_hashes do not match frozen declaration")
        if s0_manifest.get("files") != EXPECTED_S0_FILES:
            failures.append(
                "S0 manifest packet-file hashes do not match frozen declaration"
            )
        if s0_manifest.get("runner_sha256") != EXPECTED_S0_RUNNER_SHA256:
            failures.append("S0 manifest runner hash does not match frozen declaration")
    for name, expected in EXPECTED_S0_FILES.items():
        path = s0_packet / name
        if not path.is_file() or _sha256(path) != expected:
            failures.append(f"S0 {name} hash mismatch")
    runner_path = REPO / "scripts/tools/run_s0_safe_domain_runtime_transfer.py"
    if not runner_path.is_file() or _sha256(runner_path) != EXPECTED_S0_RUNNER_SHA256:
        failures.append("S0 runner hash mismatch")

    capture_manifest_path = study_dir / "capture.csv.gz.manifest.json"
    if not capture_manifest_path.is_file():
        failures.append(f"missing capture manifest: {capture_manifest_path}")
        capture_manifest: dict[str, Any] = {}
    else:
        capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
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
        "s0_packet_hashes": {
            "grid.csv": _sha256(s0_packet / "grid.csv"),
            "metrics.json": _sha256(s0_packet / "metrics.json"),
            "manifest.json": _sha256(s0_manifest_path),
        },
        "s0_runner_sha256": _sha256(runner_path),
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
    """Return frozen J2 class, reason, offline status, coordinate availability."""
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
    if not has_global_identity:
        if event_key:
            return (
                "provenance ambiguous",
                "keyed_row_has_unresolved_global_identity",
                "not_checked",
                False,
            )
        return (
            "structurally unjoinable",
            "unresolved_global_identity_no_local_id_fallback",
            "absent",
            False,
        )
    if key in offline_nonunique:
        return (
            "provenance ambiguous",
            "nonunique_offline_pair_identity",
            "nonunique",
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
    coordinates = bool(
        event["runtime_coordinates_available"]
        and offline["offline_coordinates_available"]
    )
    if not coordinates:
        return (
            "structurally unjoinable",
            "same_pair_frozen_coordinates_unavailable",
            "unique",
            False,
        )
    if event_key:
        return "exact-key reconstructable", "exact_v2_global_pair_key", "unique", True
    return (
        "deterministic auxiliary-key reconstructable",
        "canonical_v2_global_fields_without_event_key",
        "unique",
        True,
    )


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
    """Reduce event inventory to the declared (seq, lost_global_id) trial unit."""
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
            "reconstructable_unique_lost_track_upper_bound": len(reconstructable),
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
    all_reconstructable = {
        (row["seq"], row["lost_global_id"])
        for row in records
        if row["classification"] in RECONSTRUCTABLE
    }
    return {
        "trial_unit": "(seq, lost_global_id)",
        "partitions": partitions,
        "reconstructable_unique_lost_track_upper_bound": len(all_reconstructable),
        "n_max_zero_new_hurt": BASE_N + len(all_reconstructable),
        "best_case_floor_n": CP_ZERO_HURT_FLOOR,
    }


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


def run_blind(
    *, study_dir: Path, substrate_dir: Path, s0_packet: Path, output_dir: Path
) -> dict[str, Any]:
    """J1–J3.  No GT label reader is called on this path."""
    j1 = verify_j1(
        study_dir=study_dir, substrate_dir=substrate_dir, s0_packet=s0_packet
    )
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
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "inventory.csv"
    _write_inventory(inventory_path, inventory)
    inventory_hash = _sha256(inventory_path)
    metrics = {
        "study": STUDY,
        "phase": "outcome_blind_sealed",
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
            "classification_rules": "sealed declaration §3",
        },
        "j3_independence_reduction": j3,
        "frozen_reference": {
            "base_n": BASE_N,
            "base_k": BASE_K,
            "epsilon": EPSILON,
            "confidence": CONFIDENCE,
            "zero_new_hurt_pass_floor_n": CP_ZERO_HURT_FLOOR,
        },
        "pre_gt_inventory_sha256": inventory_hash,
    }
    metrics_path = output_dir / "metrics.json"
    _json_write(metrics_path, metrics)
    manifest = {
        "study": STUDY,
        "phase": "outcome_blind_sealed",
        "terminal": None,
        "declaration": DECLARATION,
        "declaration_sha256": _sha256(_absolute(DECLARATION)),
        "runner": "scripts/tools/audit_runtime_join_support_sufficiency.py",
        "runner_sha256": _sha256(Path(__file__)),
        "input_hashes": j1["input_hashes"],
        "event_key": j1["event_key"],
        "pre_gt_seal": {
            "inventory_sha256": inventory_hash,
            "gt_label_accessed": False,
            "classification_rules": "sealed declaration §3",
            "trial_unit": "(seq, lost_global_id)",
        },
        "files": {
            "inventory.csv": inventory_hash,
            "metrics.json": _sha256(metrics_path),
        },
    }
    _json_write(output_dir / "manifest.json", manifest)
    return metrics


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_reference_cells(grid_path: Path) -> list[dict[str, float]]:
    """Use every frozen S0 cell with the supplied base (N, k); choose none."""
    required = {
        "theta_dist_h",
        "theta_abs_log_h_ratio",
        "n_gt_exposed_tracks",
        "n_gt_hurt_runtime_tracks",
    }
    cells: list[dict[str, float]] = []
    with grid_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader.fieldnames, required, "S0 grid.csv")
        for row in reader:
            if (
                int(row["n_gt_exposed_tracks"]) == BASE_N
                and int(row["n_gt_hurt_runtime_tracks"]) == BASE_K
            ):
                cells.append(
                    {
                        "theta_dist_h": float(row["theta_dist_h"]),
                        "theta_abs_log_h_ratio": float(row["theta_abs_log_h_ratio"]),
                    }
                )
    if not cells:
        raise AuditInvalid("S0 grid does not contain the supplied frozen base (N, k)")
    return cells


def _runtime_reject(inventory: dict[str, str], cell: dict[str, float]) -> bool:
    dist = float(inventory["runtime_dist_h"])
    ratio = abs(
        math.log(
            float(inventory["runtime_ema_lost"]) / float(inventory["runtime_ema_cand"])
        )
    )
    return dist >= cell["theta_dist_h"] or ratio >= cell["theta_abs_log_h_ratio"]


def evaluate_reveal(
    *, inventory: list[dict[str, str]], pairs_path: Path, grid_path: Path
) -> dict[str, Any]:
    """J4 on exactly the sealed reconstructable inventory; no strata mutation."""
    eligible = [row for row in inventory if row["classification"] in RECONSTRUCTABLE]
    keys = {
        (row["seq"], int(row["lost_global_id"]), int(row["cand_global_id"]))
        for row in eligible
    }
    if not eligible:
        base_ucb = clopper_pearson_upper(BASE_K, BASE_N)
        return {
            "gt_label_accessed": False,
            "reason": "frozen reconstructable inventory is empty; no GT-label projection exists",
            "additional_gt_valid_match_unique_lost_tracks": 0,
            "additional_hurt_observable_range": [0, 0],
            "reference_cells": [],
            "possible_merged_counts": [{"n": BASE_N, "k": BASE_K, "ucb": base_ucb}],
        }
    pairs = _read_offline_gt(pairs_path, keys)
    cells = _read_reference_cells(grid_path)
    gt_rows = []
    for row in eligible:
        key = (row["seq"], int(row["lost_global_id"]), int(row["cand_global_id"]))
        pair = pairs[key]
        if _as_bool01(pair["gt_valid"]) and _as_bool01(pair["gt_match"]):
            gt_rows.append(row)
    gt_tracks = {(row["seq"], int(row["lost_global_id"])) for row in gt_rows}
    combinations: list[dict[str, Any]] = []
    for cell in cells:
        hurt_tracks = {
            (row["seq"], int(row["lost_global_id"]))
            for row in gt_rows
            if _runtime_reject(row, cell)
        }
        n = BASE_N + len(gt_tracks)
        k = BASE_K + len(hurt_tracks)
        combinations.append(
            {
                **cell,
                "additional_gt_valid_match_unique_lost_tracks": len(gt_tracks),
                "additional_hurt_tracks": len(hurt_tracks),
                "n": n,
                "k": k,
                "ucb": clopper_pearson_upper(k, n),
            }
        )
    return {
        "gt_label_accessed": True,
        "reason": "GT projected only for pre-GT sealed reconstructable records",
        "additional_gt_valid_match_unique_lost_tracks": len(gt_tracks),
        "additional_hurt_observable_range": [
            min(row["additional_hurt_tracks"] for row in combinations),
            max(row["additional_hurt_tracks"] for row in combinations),
        ],
        "reference_cells": cells,
        "possible_merged_counts": combinations,
    }


def determine_terminal(
    *, j3: dict[str, Any], reveal: dict[str, Any], semantic_admissible: bool = True
) -> str:
    """Apply RJ0's ordered terminal mapping after frozen J1–J4 evidence."""
    if j3["n_max_zero_new_hurt"] < CP_ZERO_HURT_FLOOR:
        return "RJ0_EXPANSION_FUTILE"
    if not semantic_admissible:
        return "RJ0_EXPANSION_INADMISSIBLE"
    combinations = reveal["possible_merged_counts"]
    if any(row["ucb"] <= EPSILON for row in combinations):
        return "RJ0_EXPANSION_ADMISSIBLE"
    # All legal deterministic support is retained; none crosses the frozen bar.
    return "RJ0_EXPANSION_FUTILE"


def run_reveal(*, study_dir: Path, s0_packet: Path, output_dir: Path) -> dict[str, Any]:
    """J4–J5, allowed only after an intact outcome-blind packet seal."""
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    inventory_path = output_dir / "inventory.csv"
    if not all(
        path.is_file() for path in (manifest_path, metrics_path, inventory_path)
    ):
        raise AuditInvalid("J4 requires an existing J1–J3 blind packet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blind_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_declaration_sha = _sha256(_absolute(DECLARATION))
    if (
        manifest.get("phase") != "outcome_blind_sealed"
        or manifest.get("terminal") is not None
    ):
        raise AuditInvalid("J4 requires an unmodified outcome_blind_sealed manifest")
    if manifest.get("declaration_sha256") != expected_declaration_sha:
        raise AuditInvalid("RJ0 declaration changed after the blind seal")
    if manifest.get("pre_gt_seal", {}).get("gt_label_accessed") is not False:
        raise AuditInvalid("blind seal does not prove GT was unread")
    if manifest.get("pre_gt_seal", {}).get("inventory_sha256") != _sha256(
        inventory_path
    ):
        raise AuditInvalid("inventory changed after the blind seal")
    if (
        blind_metrics.get("phase") != "outcome_blind_sealed"
        or blind_metrics.get("gt_label_accessed") is not False
    ):
        raise AuditInvalid("metrics do not prove an outcome-blind J1–J3 phase")

    inventory = _read_inventory(inventory_path)
    reveal = evaluate_reveal(
        inventory=inventory,
        pairs_path=study_dir / "pairs.csv",
        grid_path=s0_packet / "grid.csv",
    )
    j5 = {
        "same_research_population": True,
        "same_offline_runtime_coordinate_definitions": True,
        "same_lost_track_independence_unit": True,
        "same_frozen_axes_and_grid": True,
        "local_id_fallback_used": False,
        "outcome_conditioned_join_selection": False,
        "new_proxy_or_refit": False,
        "production_state_mutation": False,
        "admissibility_note": "No reconstructable support may be replaced by an alternate join key.",
    }
    semantic_admissible = (
        j5["same_research_population"]
        and j5["same_offline_runtime_coordinate_definitions"]
        and j5["same_lost_track_independence_unit"]
        and j5["same_frozen_axes_and_grid"]
        and not j5["local_id_fallback_used"]
        and not j5["outcome_conditioned_join_selection"]
        and not j5["new_proxy_or_refit"]
        and not j5["production_state_mutation"]
    )
    terminal = determine_terminal(
        j3=blind_metrics["j3_independence_reduction"],
        reveal=reveal,
        semantic_admissible=semantic_admissible,
    )
    final_metrics = {
        **blind_metrics,
        "phase": "complete",
        "terminal": terminal,
        "j4_gt_reveal": reveal,
        "j5_join_semantic_admissibility": j5,
    }
    _json_write(metrics_path, final_metrics)
    final_manifest = {
        **manifest,
        "phase": "complete",
        "terminal": terminal,
        "files": {
            "inventory.csv": _sha256(inventory_path),
            "metrics.json": _sha256(metrics_path),
        },
    }
    _json_write(manifest_path, final_manifest)
    return final_metrics


def _invalid_packet(output_dir: Path, failure: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        output_dir / "metrics.json",
        {
            "study": STUDY,
            "phase": "invalid",
            "terminal": "RJ0_INVALID",
            "failure": failure,
        },
    )
    _json_write(
        output_dir / "manifest.json",
        {
            "study": STUDY,
            "phase": "invalid",
            "terminal": "RJ0_INVALID",
            "declaration": DECLARATION,
            "files": {"metrics.json": _sha256(output_dir / "metrics.json")},
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("blind", "reveal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path(STUDY_DIR))
    parser.add_argument("--substrate-dir", type=Path, default=Path(SUBSTRATE_DIR))
    parser.add_argument("--s0-packet", type=Path, default=Path(S0_PACKET))
    args = parser.parse_args(argv)
    output_dir = _absolute(args.output_dir)
    try:
        if args.phase == "blind":
            result = run_blind(
                study_dir=_absolute(args.study_dir),
                substrate_dir=_absolute(args.substrate_dir),
                s0_packet=_absolute(args.s0_packet),
                output_dir=output_dir,
            )
        else:
            result = run_reveal(
                study_dir=_absolute(args.study_dir),
                s0_packet=_absolute(args.s0_packet),
                output_dir=output_dir,
            )
    except AuditInvalid as exc:
        _invalid_packet(output_dir, str(exc))
        print(json.dumps({"terminal": "RJ0_INVALID", "failure": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "study": STUDY,
                "phase": result["phase"],
                "terminal": result.get("terminal"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
