#!/usr/bin/env python3
"""Audit whether the frozen relink-pair table supports the motion probe.

The audit is deliberately schema-only and label-agnostic: it does not fit a
motion model or select a signal.  It establishes which predeclared outputs can
be reconstructed from the already-frozen pair universe and fails closed on
missing vector-velocity/context fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, cast


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

BASE_FIELDS = {
    "seq",
    "lost_id",
    "cand_id",
    "gt_match",
    "gt_valid",
    "gap",
    "lost_last_frame",
    "cand_first_frame",
}
POSITION_FIELDS = {
    "lost_foot_x",
    "lost_foot_y",
    "cand_foot_x",
    "cand_foot_y",
    "h_ref",
}
M0_FIELDS = {
    "bridge_dist",
    "fwd_resid",
    "bwd_resid",
    "dir_cos",
    "lost_exit_speed",
    "cand_entry_speed",
}
VECTOR_VELOCITY_FIELDS = {
    "lost_exit_vx",
    "lost_exit_vy",
    "cand_entry_vx",
    "cand_entry_vy",
}
TRANSFERABLE_CONTEXT_FIELDS = {
    "exit_zone",
    "image_width",
    "image_height",
    "gmc_direction_cluster",
    "route_group",
}


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


def _gap_bin(gap: int) -> str:
    for name, low, high in GAP_BINS:
        if low <= gap <= high:
            return name
    return "out_of_contract"


def _missing(header: Iterable[str], required: set[str]) -> list[str]:
    return sorted(required - set(header))


def analyze(pairs: Path) -> dict[str, Any]:
    pairs = pairs.resolve()
    with pairs.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or [])
        missing_base = _missing(header, BASE_FIELDS)
        if missing_base:
            raise ValueError(f"missing required base fields: {missing_base}")

        rows_total = 0
        rows_gt_valid = 0
        gt_total = 0
        fp_total = 0
        invalid_gap_identity = 0
        invalid_position_rows = 0
        duplicate_pair_keys = 0
        seen_keys: set[tuple[str, str, str]] = set()
        seq_counts: Counter[str] = Counter()
        seq_gt_counts: Counter[str] = Counter()
        gap_counts: dict[str, Counter[str]] = {
            name: Counter() for name, _, _ in GAP_BINS
        }
        gap_counts["out_of_contract"] = Counter()

        position_available = not _missing(header, POSITION_FIELDS)
        for row in reader:
            rows_total += 1
            gap = int(row["gap"])
            if gap != int(row["cand_first_frame"]) - int(row["lost_last_frame"]):
                invalid_gap_identity += 1

            key = (row["seq"], row["lost_id"], row["cand_id"])
            if key in seen_keys:
                duplicate_pair_keys += 1
            seen_keys.add(key)

            if position_available:
                values = [float(row[name]) for name in sorted(POSITION_FIELDS)]
                if (
                    not all(math.isfinite(value) for value in values)
                    or float(row["h_ref"]) <= 0
                ):
                    invalid_position_rows += 1

            if not _as_bool(row["gt_valid"]):
                continue
            rows_gt_valid += 1
            is_gt = _as_bool(row["gt_match"])
            gt_total += int(is_gt)
            fp_total += int(not is_gt)
            seq = row["seq"]
            seq_counts[seq] += 1
            seq_gt_counts[seq] += int(is_gt)
            bucket = gap_counts[_gap_bin(gap)]
            bucket["pairs"] += 1
            bucket["gt"] += int(is_gt)
            bucket["fp"] += int(not is_gt)

    missing_position = _missing(header, POSITION_FIELDS)
    missing_m0 = _missing(header, M0_FIELDS)
    missing_vector_velocity = _missing(header, VECTOR_VELOCITY_FIELDS)
    missing_transferable_context = _missing(header, TRANSFERABLE_CONTEXT_FIELDS)

    position_ok = (
        not missing_position
        and invalid_position_rows == 0
        and invalid_gap_identity == 0
    )
    joint_ok = position_ok and not missing_vector_velocity
    m0_ok = not missing_m0 and invalid_gap_identity == 0
    verdict = (
        "IDENTIFIABLE"
        if position_ok and joint_ok and m0_ok
        else "PARTIALLY_IDENTIFIABLE"
        if position_ok and m0_ok
        else "NOT_IDENTIFIABLE"
    )

    return {
        "schema_version": 1,
        "source": {
            "pairs_csv": str(pairs),
            "sha256": sha256(pairs),
            "columns": header,
        },
        "integrity": {
            "rows_total": rows_total,
            "rows_gt_valid": rows_gt_valid,
            "gt_total": gt_total,
            "fp_total": fp_total,
            "sequence_count": len(seq_counts),
            "invalid_gap_identity": invalid_gap_identity,
            "invalid_position_rows": invalid_position_rows,
            "duplicate_pair_keys": duplicate_pair_keys,
        },
        "gap_bins": {name: dict(gap_counts[name]) for name, _, _ in GAP_BINS},
        "sequences": {
            seq: {"pairs": seq_counts[seq], "gt": seq_gt_counts[seq]}
            for seq in sorted(seq_counts)
        },
        "identifiability": {
            "verdict": verdict,
            "m0_deterministic": {
                "identifiable": m0_ok,
                "missing_fields": missing_m0,
            },
            "position_only_transition": {
                "identifiable": position_ok,
                "missing_fields": missing_position,
                "allowed_observation": "delta_foot_xy / h_ref conditioned on gap",
            },
            "velocity_only_transition": {
                "identifiable": False if missing_vector_velocity else position_ok,
                "missing_fields": missing_vector_velocity,
            },
            "joint_position_velocity_transition": {
                "identifiable": joint_ok,
                "missing_fields": missing_vector_velocity,
            },
            "contexts": {
                "loo_headline_eligible": ["global"],
                "diagnostic_only": ["sequence"],
                "missing_transferable_fields": missing_transferable_context,
            },
        },
        "gate": {
            "phase_b_authorized": False,
            "next_allowed": [
                "rebuild M0 role-reversal baseline on canonical gap bins",
                "specify a position-only M1-P/M2-P family without vector velocity claims",
            ],
            "blocked": [
                "velocity-only and joint NLL/q_motion",
                "sequence-conditioned LOO headline",
                "exit-zone/GMC/route context claims",
            ],
        },
    }


def _stable_summary(result: dict[str, Any]) -> dict[str, Any]:
    stable = cast(dict[str, Any], json.loads(json.dumps(result)))
    stable["source"]["pairs_csv"] = str(CANONICAL_PAIRS)
    return stable


def _render(result: dict[str, Any]) -> str:
    ident = result["identifiability"]
    integrity = result["integrity"]
    lines = [
        f"verdict={ident['verdict']}",
        f"source_sha256={result['source']['sha256']}",
        (
            "pool="
            f"{integrity['rows_gt_valid']} "
            f"(GT={integrity['gt_total']}, FP={integrity['fp_total']})"
        ),
        f"sequences={integrity['sequence_count']}",
        f"m0_identifiable={str(ident['m0_deterministic']['identifiable']).lower()}",
        (
            "position_only_identifiable="
            f"{str(ident['position_only_transition']['identifiable']).lower()}"
        ),
        (
            "joint_identifiable="
            f"{str(ident['joint_position_velocity_transition']['identifiable']).lower()}"
        ),
        "loo_headline_contexts=global",
        "phase_b_authorized=false",
    ]
    return "\n".join(lines) + "\n"


def _write_packet(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _stable_summary(result)
    summary_path = output_dir / "substrate_audit.json"
    output_path = output_dir / "recorded_output.txt"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_path.write_text(_render(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "verdict": summary["identifiability"]["verdict"],
        "source_pairs_csv": str(CANONICAL_PAIRS),
        "source_pairs_csv_sha256": summary["source"]["sha256"],
        "runner_sha256": sha256(Path(__file__)),
        "artifacts": {
            "substrate_audit.json": sha256(summary_path),
            "recorded_output.txt": sha256(output_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def verify(pairs: Path) -> None:
    manifest = json.loads((PACKET_DIR / "manifest.json").read_text(encoding="utf-8"))
    actual_sha = sha256(pairs)
    expected_sha = manifest["source_pairs_csv_sha256"]
    if actual_sha != expected_sha:
        raise SystemExit(
            f"source SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_packet(analyze(pairs), tmp_path)
        for name in ("substrate_audit.json", "recorded_output.txt", "manifest.json"):
            expected = (PACKET_DIR / name).read_bytes()
            actual = (tmp_path / name).read_bytes()
            if actual != expected:
                raise SystemExit(f"verification failed: {name} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=REPO / CANONICAL_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=PACKET_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.pairs.resolve())
        print("E0 packet verification: PASS")
        return
    result = analyze(args.pairs)
    _write_packet(result, args.output_dir)
    print(_render(_stable_summary(result)), end="")


if __name__ == "__main__":
    main()
