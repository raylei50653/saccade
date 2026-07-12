#!/usr/bin/env python3
"""Merge per-sequence Issue #112 native captures into D0's CSV contract.

Usage:
  SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_DIR=out/d0-native \
    uv run scripts/eval/eval_mamba_head.py ...
  uv run python scripts/tools/export_d0_runtime_capture.py \
    --capture-dir out/d0-native --output out/d0-native/capture.csv.gz

The evaluator writes one JSON file per sequence. This tool refuses overflow,
missing/duplicate event keys, or rows that were not produced by the native
CUDA event buffer. It only creates an input artifact; the D0 verifier remains
responsible for the fidelity verdict and its coverage gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "d0_bridge_estimator_fidelity_20260711"
    / "run_d0_bridge_fidelity.py"
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


def export_capture(capture_dir: Path, output: Path) -> dict[str, Any]:
    runner = _load_runner()
    files = sorted(
        path for path in capture_dir.glob("*.json") if path.name != "manifest.json"
    )
    if not files:
        raise ValueError(f"no per-sequence capture JSON files in {capture_dir}")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    shared_provenance: dict[str, Any] | None = None
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total = int(payload.get("total_events", -1))
        overflow = int(payload.get("overflow_events", -1))
        events = payload.get("events")
        if payload.get("complete") is not True or overflow != 0:
            raise ValueError(f"incomplete native capture: {path}")
        if not isinstance(events, list) or total != len(events):
            raise ValueError(f"event count mismatch in native capture: {path}")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"native capture lacks provenance: {path}")
        if provenance.get("capture_contract") != "d0_runtime_cuda_v1":
            raise ValueError(f"unsupported native capture contract: {path}")
        if shared_provenance is None:
            shared_provenance = provenance
        elif provenance != shared_provenance:
            raise ValueError(f"mixed native-capture provenance: {path}")
        for row in events:
            if row.get("capture_mode") != "runtime_cuda_event_ring":
                raise ValueError(f"non-native capture row in {path}")
            if row.get("evidence_role") != "runtime_cuda_observation":
                raise ValueError(f"invalid native evidence role in {path}")
            key = runner.event_key_from_row(row)
            if str(row.get("event_key")) != key:
                raise ValueError(f"event-key mismatch in {path}: {key}")
            rows.append(dict(row))
        sources.append({"path": str(path), "sha256": _sha256(path), "events": total})

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    capture_dir = (
        args.capture_dir if args.capture_dir.is_absolute() else REPO / args.capture_dir
    )
    output = args.output if args.output.is_absolute() else REPO / args.output
    print(json.dumps(export_capture(capture_dir, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
