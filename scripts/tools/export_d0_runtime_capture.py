#!/usr/bin/env python3
"""Merge per-sequence Issue #112 native captures into D0's CSV contract.

Two event-key contracts exist. They are versioned, never reinterpreted.

``v1`` (legacy, frozen)
    The 5-field key of the sealed reconstruction packet. Retained only so
    historical sealed packets keep validating under their original semantics.
    Its ids are tracker-**local** and two of its fields are impossible frame
    indices; it must not be used for new runtime evidence. See
    ``consumer_a_bridge_fidelity.EVENT_KEY_VERSION_V1_LEGACY``.

``v2`` (runtime shadow fidelity)
    Key is ``(seq, lost_global_id, cand_global_id)``. Requires the evaluator's
    ``_global_id_map.txt`` to lift tracker-local ids into the global id space
    that the MOT output -- and therefore the offline pair cohort -- is written
    in. Never falls back to raw/local ids: an unresolvable id is partitioned as
    ``unemitted``, not silently joined.

The v2 export also partitions every captured proposal into exactly one of
``matched`` / ``cohort_gap`` / ``unemitted`` and asserts the counts sum to the
number of captured events. Fidelity may be computed only on ``matched``; the
other two bound how far that conclusion extrapolates and must never enter an
agreement denominator.

Usage (v2):
  SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_DIR=out/d0-native \
  SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_SHADOW=1 \
    .venv/bin/python scripts/eval/mot17.py --relink-bridge-enabled ...
  .venv/bin/python scripts/tools/export_d0_runtime_capture.py \
    --event-key-version v2 \
    --capture-dir out/d0-native \
    --id-map results/<substrate>/_global_id_map.txt \
    --pairs out/signal_study/<cohort>/pairs.csv \
    --output out/d0-native/capture.csv.gz

This tool only creates an input artifact; the D0 verifier remains responsible
for the fidelity verdict and its coverage gates.
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    EVENT_KEY_FIELDS_V2,
    EVENT_KEY_V1_UNSOUND_FIELDS,
    EVENT_KEY_VERSION_V2,
    PARTITION_COHORT_GAP,
    PARTITION_MATCHED,
    PARTITION_UNEMITTED,
    PARTITIONS,
    event_key_v2,
)
from saccade.perception.eval.d_online_stage2 import parse_global_id_map


REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "d0_bridge_estimator_fidelity_20260711"
    / "run_d0_bridge_fidelity.py"
)

# v2 physics payload. Deliberately excludes EVENT_KEY_V1_UNSOUND_FIELDS.
CAPTURE_FIELDS_V2: tuple[str, ...] = (
    "event_key",
    "event_key_version",
    "partition",
    "seq",
    "lost_global_id",
    "cand_global_id",
    "lost_local_id",
    "cand_local_id",
    "gap",
    "bridge_at",
    "la",
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
    "capture_mode",
    "evidence_role",
    "anchor_mode",
    "anchor_rate",
)


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("d0_runtime_export", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load D0 runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_captures(
    capture_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Load per-sequence capture JSON, fail-closed on incompleteness/overflow."""
    files = sorted(
        path for path in capture_dir.glob("*.json") if path.name != "manifest.json"
    )
    if not files:
        raise ValueError(f"no per-sequence capture JSON files in {capture_dir}")

    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total = int(payload.get("total_events", -1))
        overflow = int(payload.get("overflow_events", -1))
        rows = payload.get("events")
        if payload.get("complete") is not True or overflow != 0:
            raise ValueError(f"incomplete native capture: {path}")
        if not isinstance(rows, list) or total != len(rows):
            raise ValueError(f"event count mismatch in native capture: {path}")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"native capture lacks provenance: {path}")
        if provenance.get("capture_contract") != "d0_runtime_cuda_v1":
            raise ValueError(f"unsupported native capture contract: {path}")
        if shared is None:
            shared = provenance
        elif provenance != shared:
            raise ValueError(f"mixed native-capture provenance: {path}")
        for row in rows:
            if row.get("capture_mode") != "runtime_cuda_event_ring":
                raise ValueError(f"non-native capture row in {path}")
            if row.get("evidence_role") != "runtime_cuda_observation":
                raise ValueError(f"invalid native evidence role in {path}")
            events.append(dict(row))
        sources.append({"path": str(path), "sha256": _sha256(path), "events": total})
    assert shared is not None
    return events, shared, sources


def export_capture(capture_dir: Path, output: Path) -> dict[str, Any]:
    """Legacy v1 export. Frozen: for historical sealed packets only."""
    runner = _load_runner()
    rows, shared_provenance, sources = _read_captures(capture_dir)
    for row in rows:
        key = runner.event_key_from_row(row)
        if str(row.get("event_key")) != key:
            raise ValueError(f"event-key mismatch: {key}")

    rows.sort(key=lambda row: str(row["event_key"]))
    seen: set[str] = set()
    duplicates = [
        str(row["event_key"])
        for row in rows
        if str(row["event_key"]) in seen or seen.add(str(row["event_key"]))
    ]
    if duplicates:
        raise ValueError(f"duplicate native event keys: {duplicates[:3]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    capture_sha = runner.write_gzip_csv(output, runner.CAPTURE_FIELDS, rows)
    manifest = {
        "capture_mode": "runtime_cuda_event_ring",
        "evidence_role": "runtime_cuda_observation",
        "event_key_version": "d0_event_key_v1_local_legacy",
        "capture_csv": str(output),
        "capture_sha256": capture_sha,
        "events": len(rows),
        "overflow_events": 0,
        "provenance": shared_provenance,
        "sources": sources,
    }
    (output.parent / f"{output.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_cohort(pairs_csv: Path) -> set[tuple[str, int, int]]:
    with pairs_csv.open(encoding="utf-8") as stream:
        return {
            (r["seq"], int(r["lost_id"]), int(r["cand_id"]))
            for r in csv.DictReader(stream)
        }


def export_capture_v2(
    capture_dir: Path, output: Path, *, id_map_path: Path, pairs_csv: Path
) -> dict[str, Any]:
    """Global-id export with an exhaustive, conserved partition."""
    runner = _load_runner()
    events, shared_provenance, sources = _read_captures(capture_dir)

    # A committing bridge rewrites track identity, so its ids do not address a
    # bridge-off cohort. Refuse rather than emit an unjoinable packet.
    if shared_provenance.get("shadow") is not True:
        raise ValueError(
            "v2 export requires a shadow capture (propose without commit); "
            "provenance says shadow != true, so track identity was mutated "
            "by the bridge and cannot be joined to a bridge-off cohort"
        )

    id_map = parse_global_id_map(id_map_path)
    if not id_map:
        raise ValueError(f"empty global id map: {id_map_path}")
    # The lift local -> global must be injective per sequence, else a global key
    # could address two distinct tracks.
    per_seq: dict[str, set[int]] = {}
    for (seq, _local), glob in id_map.items():
        bucket = per_seq.setdefault(seq, set())
        if glob in bucket:
            raise ValueError(f"global id map is not injective for {seq}: {glob}")
        bucket.add(glob)

    cohort = _load_cohort(pairs_csv)

    rows: list[dict[str, Any]] = []
    counts = dict.fromkeys(PARTITIONS, 0)
    for ev in events:
        seq = str(ev["seq"])
        lost_local = int(ev["lost_id"])
        cand_local = int(ev["cand_id"])
        gl = id_map.get((seq, lost_local))
        gc = id_map.get((seq, cand_local))

        row = {k: v for k, v in ev.items() if k not in EVENT_KEY_V1_UNSOUND_FIELDS}
        row["seq"] = seq
        row["lost_local_id"] = lost_local
        row["cand_local_id"] = cand_local
        row["event_key_version"] = EVENT_KEY_VERSION_V2

        if gl is None or gc is None:
            # Never fall back to raw/local ids: that is exactly the bug that
            # produced false matches where the remap was the identity.
            partition = PARTITION_UNEMITTED
            row["lost_global_id"] = -1
            row["cand_global_id"] = -1
            row["event_key"] = ""
        else:
            row["lost_global_id"] = gl
            row["cand_global_id"] = gc
            row["event_key"] = event_key_v2(seq, gl, gc)
            partition = (
                PARTITION_MATCHED if (seq, gl, gc) in cohort else PARTITION_COHORT_GAP
            )
        row["partition"] = partition
        counts[partition] += 1
        rows.append(row)

    # Conservation: the partition is exhaustive and mutually exclusive.
    if sum(counts.values()) != len(events):
        raise ValueError(
            f"partition does not conserve events: {counts} vs {len(events)}"
        )

    # Uniqueness holds over every keyed row (propose fires once per track life).
    keyed = [r for r in rows if r["event_key"]]
    seen: set[str] = set()
    dupes = [
        str(r["event_key"])
        for r in keyed
        if str(r["event_key"]) in seen or seen.add(str(r["event_key"]))
    ]
    if dupes:
        raise ValueError(f"duplicate v2 event keys: {dupes[:3]}")

    for field in EVENT_KEY_FIELDS_V2:
        if any(field not in r for r in keyed):
            raise ValueError(f"v2 key field missing from keyed rows: {field}")

    rows.sort(key=lambda r: (str(r["partition"]), str(r["event_key"]), str(r["seq"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_sha = runner.write_gzip_csv(output, CAPTURE_FIELDS_V2, rows)

    manifest = {
        "capture_mode": "runtime_cuda_event_ring",
        "evidence_role": "runtime_cuda_observation",
        "event_key_version": EVENT_KEY_VERSION_V2,
        "event_key_fields": list(EVENT_KEY_FIELDS_V2),
        "capture_csv": str(output),
        "capture_sha256": capture_sha,
        "events": len(rows),
        "overflow_events": 0,
        "partition": counts,
        "partition_note": (
            "fidelity may be computed only on 'matched'; 'cohort_gap' and "
            "'unemitted' bound extrapolation and must not enter an agreement "
            "denominator"
        ),
        "id_map": {"path": str(id_map_path), "sha256": _sha256(id_map_path)},
        "cohort_pairs": {"path": str(pairs_csv), "sha256": _sha256(pairs_csv)},
        "provenance": shared_provenance,
        "sources": sources,
    }
    (output.parent / f"{output.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-key-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 = frozen legacy key (sealed packets only); v2 = global-id key.",
    )
    parser.add_argument(
        "--id-map", type=Path, help="evaluator _global_id_map.txt (required for v2)"
    )
    parser.add_argument(
        "--pairs", type=Path, help="offline cohort pairs.csv (required for v2)"
    )
    args = parser.parse_args(argv)

    def _abs(path: Path) -> Path:
        return path if path.is_absolute() else REPO / path

    capture_dir = _abs(args.capture_dir)
    output = _abs(args.output)

    if args.event_key_version == "v2":
        if args.id_map is None or args.pairs is None:
            parser.error("--event-key-version v2 requires --id-map and --pairs")
        manifest = export_capture_v2(
            capture_dir,
            output,
            id_map_path=_abs(args.id_map),
            pairs_csv=_abs(args.pairs),
        )
    else:
        manifest = export_capture(capture_dir, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
