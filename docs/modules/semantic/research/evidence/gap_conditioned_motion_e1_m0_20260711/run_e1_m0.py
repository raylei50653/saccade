#!/usr/bin/env python3
"""Rebuild the deterministic M0 role-reversal baseline by frozen gap bin."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, cast


REPO = Path(__file__).resolve().parents[6]
PACKET_DIR = Path(__file__).resolve().parent
CANONICAL_PAIRS = Path(
    "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
)
GAP_BINS = (
    ("1-10", 1, 10),
    ("11-30", 11, 30),
    ("31-60", 31, 60),
    ("61-150", 61, 150),
    ("151-300", 151, 300),
)
SIGNALS: dict[str, Callable[[dict[str, str]], float]] = {
    "bridge_dist": lambda row: float(row["bridge_dist"]),
    "speed_mismatch": lambda row: abs(
        float(row["lost_exit_speed"]) - float(row["cand_entry_speed"])
    ),
    "dir_cos": lambda row: 1.0 - float(row["dir_cos"]),
    "resid_mean": lambda row: 0.5 * (float(row["fwd_resid"]) + float(row["bwd_resid"])),
}
REQUIRED = {
    "seq",
    "gt_match",
    "gt_valid",
    "gap",
    "bridge_dist",
    "lost_exit_speed",
    "cand_entry_speed",
    "dir_cos",
    "fwd_resid",
    "bwd_resid",
}
TAIL_QUANTILE = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _auc(labels: list[bool], scores: list[float]) -> float:
    """Tie-aware rank AUC; scores are higher-is-more-GT."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        rank_sum_pos += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)


def analyze(pairs: Path) -> dict[str, Any]:
    bins: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in GAP_BINS}
    with pairs.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = set(reader.fieldnames or [])
        missing = sorted(REQUIRED - header)
        if missing:
            raise ValueError(f"missing required M0 fields: {missing}")
        for row in reader:
            if not _as_bool(row["gt_valid"]):
                continue
            gap = int(row["gap"])
            bucket = next(
                (name for name, low, high in GAP_BINS if low <= gap <= high), None
            )
            if bucket is None:
                raise ValueError(f"gap outside frozen contract: {gap}")
            bins[bucket].append(
                {
                    "seq": row["seq"],
                    "gt": _as_bool(row["gt_match"]),
                    "signals": {name: fn(row) for name, fn in SIGNALS.items()},
                }
            )

    cells: list[dict[str, Any]] = []
    for gap_name, _, _ in GAP_BINS:
        rows = bins[gap_name]
        if not rows:
            continue
        labels = [bool(row["gt"]) for row in rows]
        base_rate = sum(labels) / len(labels)
        for signal in SIGNALS:
            mismatches = [float(row["signals"][signal]) for row in rows]
            threshold = _quantile(mismatches, TAIL_QUANTILE)
            tail_indices = [
                index for index, value in enumerate(mismatches) if value >= threshold
            ]
            tail_gt = sum(labels[index] for index in tail_indices)
            tail_n = len(tail_indices)
            tail_rate = tail_gt / tail_n if tail_n else float("nan")
            enrichment = tail_rate / base_rate if base_rate else float("nan")
            auc_gt = _auc(labels, [-value for value in mismatches])
            reversal = bool(auc_gt < 0.5 and enrichment > 1.0)
            tail_seq_gt: dict[str, int] = {}
            for index in tail_indices:
                if labels[index]:
                    seq = str(rows[index]["seq"])
                    tail_seq_gt[seq] = tail_seq_gt.get(seq, 0) + 1
            cells.append(
                {
                    "gap_bin": gap_name,
                    "signal": signal,
                    "n": len(rows),
                    "gt": sum(labels),
                    "fp": len(rows) - sum(labels),
                    "auc_gt_low_mismatch": auc_gt,
                    "high_mismatch_tail_quantile": TAIL_QUANTILE,
                    "high_mismatch_tail_threshold": threshold,
                    "tail_n": tail_n,
                    "tail_gt": tail_gt,
                    "tail_fp": tail_n - tail_gt,
                    "tail_gt_rate": tail_rate,
                    "bin_gt_rate": base_rate,
                    "tail_gt_enrichment": enrichment,
                    "role_reversal_descriptive": reversal,
                    "tail_gt_by_sequence": tail_seq_gt,
                }
            )

    reversal_cells = [
        {"gap_bin": cell["gap_bin"], "signal": cell["signal"]}
        for cell in cells
        if cell["role_reversal_descriptive"]
    ]
    return {
        "schema_version": 1,
        "source": {"pairs_csv": str(pairs.resolve()), "sha256": sha256(pairs)},
        "protocol": {
            "universe": "gt_valid U_relink_pair",
            "gap_bins": [name for name, _, _ in GAP_BINS],
            "signals": {
                "bridge_dist": "bridge_dist (higher = more mismatch)",
                "speed_mismatch": "abs(lost_exit_speed - cand_entry_speed)",
                "dir_cos": "1 - dir_cos (stored metric keeps atom name)",
                "resid_mean": "0.5 * (fwd_resid + bwd_resid)",
            },
            "auc": "GT AUC using -mismatch; <0.5 reverses expected direction",
            "tail": "pooled within-bin q90 mismatch, inclusive on ties",
            "role_reversal_descriptive": "auc < 0.5 and tail GT enrichment > 1",
            "claim_ceiling": "descriptive E1 baseline; no model-family verdict",
        },
        "cells": cells,
        "summary": {
            "cell_count": len(cells),
            "role_reversal_cell_count": len(reversal_cells),
            "role_reversal_cells": reversal_cells,
            "phase_b_authorized": False,
        },
    }


def _stable(result: dict[str, Any]) -> dict[str, Any]:
    stable = cast(dict[str, Any], json.loads(json.dumps(result)))
    stable["source"]["pairs_csv"] = str(CANONICAL_PAIRS)
    return stable


CSV_FIELDS = (
    "gap_bin",
    "signal",
    "n",
    "gt",
    "fp",
    "auc_gt_low_mismatch",
    "high_mismatch_tail_threshold",
    "tail_n",
    "tail_gt",
    "tail_fp",
    "tail_gt_rate",
    "bin_gt_rate",
    "tail_gt_enrichment",
    "role_reversal_descriptive",
    "tail_gt_by_sequence",
)


def _write_packet(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stable = _stable(result)
    summary_path = output_dir / "m0_summary.json"
    table_path = output_dir / "m0_by_gap.csv"
    summary_path.write_text(
        json.dumps(stable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for cell in stable["cells"]:
            row = {name: cell[name] for name in CSV_FIELDS}
            row["tail_gt_by_sequence"] = json.dumps(
                row["tail_gt_by_sequence"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(row)
    manifest = {
        "schema_version": 1,
        "source_pairs_csv": str(CANONICAL_PAIRS),
        "source_pairs_csv_sha256": stable["source"]["sha256"],
        "runner_sha256": sha256(Path(__file__)),
        "claim_ceiling": stable["protocol"]["claim_ceiling"],
        "artifacts": {
            "m0_summary.json": sha256(summary_path),
            "m0_by_gap.csv": sha256(table_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def verify(pairs: Path) -> None:
    manifest = json.loads((PACKET_DIR / "manifest.json").read_text(encoding="utf-8"))
    if sha256(pairs) != manifest["source_pairs_csv_sha256"]:
        raise SystemExit("source pairs SHA mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_packet(analyze(pairs), tmp_path)
        for name in ("m0_summary.json", "m0_by_gap.csv", "manifest.json"):
            if (tmp_path / name).read_bytes() != (PACKET_DIR / name).read_bytes():
                raise SystemExit(f"verification failed: {name} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=REPO / CANONICAL_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=PACKET_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    pairs = args.pairs.resolve()
    if args.verify:
        verify(pairs)
        print("E1 M0 packet verification: PASS")
        return
    result = analyze(pairs)
    _write_packet(result, args.output_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
