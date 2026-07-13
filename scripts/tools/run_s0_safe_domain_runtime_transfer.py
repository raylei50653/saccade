#!/usr/bin/env python3
"""Execute sealed S0 Amendment 1: offline-to-runtime safe-axis transfer.

The binding declaration is
``docs/modules/semantic/research/safe_domain_runtime_transfer_declaration_20260712.md``.
This runner has no search or selection surface: it evaluates every frozen grid
point, computes §§3–7, applies the ordered terminal mapping in §8, and emits an
evidence packet.  In particular, unjoined runtime events are a fail-closed
coverage diagnostic; they never enter a track-level Clopper-Pearson bound.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]

STUDY = "s0_safe_domain_runtime_transfer_20260713"
DECLARATION = (
    "docs/modules/semantic/research/"
    "safe_domain_runtime_transfer_declaration_20260712.md"
)
REVIEWED_HEAD = "70a40cf9d61eb6512b9b5096049ca59efd58aa95"

EPSILON = 0.05
CONFIDENCE = 0.95
RHO_MIN = 0.98
JACCARD_MIN = 0.90
MIN_GT_TRACKS = 59
MIN_MATCHED_PAIRS = 1000

DIST_THRESHOLDS = tuple(round(float(x), 10) for x in np.arange(0.2, 2.01, 0.1))
RATIO_THRESHOLDS = tuple(round(float(x), 10) for x in np.arange(0.05, 0.601, 0.05))

PARTITIONS = ("matched", "cohort_gap", "unemitted")
EXPECTED_PARTITION = {"matched": 1684, "cohort_gap": 539, "unemitted": 354}

TERMINAL_INVALID = "S0_INVALID"
TERMINAL_UNDECIDABLE = "S0_UNDECIDABLE"
TERMINAL_BROKEN = "AXES_TRANSFER_BROKEN"
TERMINAL_DEGRADED = "AXES_TRANSFER_DEGRADED"
TERMINAL_HOLDS = "AXES_TRANSFER_HOLDS"


class ValidityFailure(RuntimeError):
    """A structural V-gate failed; no scientific terminal may be inferred."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool01(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    return bool(int(float(normalized)))


def clopper_pearson_upper(hurt: int, exposed: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound at the declared trial unit."""
    if exposed <= 0 or hurt < 0 or hurt > exposed:
        raise ValueError(f"invalid binomial counts: hurt={hurt}, exposed={exposed}")
    if hurt == exposed:
        return 1.0
    return float(stats.beta.ppf(CONFIDENCE, hurt + 1, exposed - hurt))


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.sum(left | right))
    if union == 0:
        return 1.0
    return float(np.sum(left & right) / union)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(stats.spearmanr(left, right).statistic)


def _read_inputs(study_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with (study_dir / "pairs.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    with gzip.open(
        study_dir / "capture.csv.gz", "rt", newline="", encoding="utf-8"
    ) as stream:
        capture = list(csv.DictReader(stream))
    return pairs, capture


def verify_frozen_inputs(
    study_dir: Path, substrate_dir: Path, expected: dict[str, str]
) -> dict[str, str]:
    resolved = {
        "pairs.csv": study_dir / "pairs.csv",
        "capture.csv.gz": study_dir / "capture.csv.gz",
        "_global_id_map.txt": substrate_dir / "_global_id_map.txt",
    }
    observed: dict[str, str] = {}
    errors: list[str] = []
    for name, path in resolved.items():
        if not path.is_file():
            errors.append(f"missing {name}: {path}")
            continue
        observed[name] = _sha256(path)
        if observed[name] != expected.get(name):
            errors.append(
                f"{name}: expected {expected.get(name, '<missing>')}, "
                f"got {observed[name]}"
            )

    mot_digest = hashlib.sha256()
    mot_files = sorted(substrate_dir.glob("MOT17-*.txt"))
    if not mot_files:
        errors.append(f"no substrate MOT files in {substrate_dir}")
    for path in mot_files:
        mot_digest.update(path.read_bytes())
    observed["substrate_mot_concat"] = mot_digest.hexdigest()
    if observed["substrate_mot_concat"] != expected.get("substrate_mot_concat"):
        errors.append(
            "substrate_mot_concat: expected "
            f"{expected.get('substrate_mot_concat', '<missing>')}, "
            f"got {observed['substrate_mot_concat']}"
        )
    if errors:
        raise ValidityFailure("V1 frozen provenance failed: " + "; ".join(errors))
    return observed


def evaluate_arrays(
    *,
    offline_dist: np.ndarray,
    offline_ratio: np.ndarray,
    runtime_dist: np.ndarray,
    runtime_ratio: np.ndarray,
    is_gt: np.ndarray,
    is_fp: np.ndarray,
    track_keys: Sequence[tuple[str, int]],
    unjoined_runtime_dist: np.ndarray,
    unjoined_runtime_ratio: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the frozen grid from already joined, validated arrays."""
    n = len(offline_dist)
    arrays = (offline_ratio, runtime_dist, runtime_ratio, is_gt, is_fp)
    if any(len(values) != n for values in arrays) or len(track_keys) != n:
        raise ValueError("matched arrays and track_keys must have equal length")

    gt_tracks = {track_keys[i] for i in np.flatnonzero(is_gt)}
    n_gt_tracks = len(gt_tracks)
    n_fp_pairs = int(np.sum(is_fp))
    grid: list[dict[str, Any]] = []

    for theta_d in DIST_THRESHOLDS:
        for theta_r in RATIO_THRESHOLDS:
            reject_off = (offline_dist >= theta_d) | (offline_ratio >= theta_r)
            reject_rt = (runtime_dist >= theta_d) | (runtime_ratio >= theta_r)

            hurt_off = {track_keys[i] for i in np.flatnonzero(is_gt & reject_off)}
            hurt_rt = {track_keys[i] for i in np.flatnonzero(is_gt & reject_rt)}
            n_hurt_off = len(hurt_off)
            n_hurt_rt = len(hurt_rt)
            ucb_off = clopper_pearson_upper(n_hurt_off, n_gt_tracks)
            ucb_rt = clopper_pearson_upper(n_hurt_rt, n_gt_tracks)

            fp_removed_off = int(np.sum(is_fp & reject_off))
            fp_removed_rt = int(np.sum(is_fp & reject_rt))
            safe_off = ucb_off <= EPSILON
            safe_rt = ucb_rt <= EPSILON
            active_offline_safe = safe_off and fp_removed_off > 0

            dist_flips = int(
                np.sum((offline_dist >= theta_d) != (runtime_dist >= theta_d))
            )
            ratio_flips = int(
                np.sum((offline_ratio >= theta_r) != (runtime_ratio >= theta_r))
            )
            unjoined_reject = (unjoined_runtime_dist >= theta_d) | (
                unjoined_runtime_ratio >= theta_r
            )
            unjoined_m = int(np.sum(unjoined_reject))

            grid.append(
                {
                    "theta_dist_h": theta_d,
                    "theta_abs_log_h_ratio": theta_r,
                    "n_gt_exposed_tracks": n_gt_tracks,
                    "n_gt_hurt_offline_tracks": n_hurt_off,
                    "n_gt_hurt_runtime_tracks": n_hurt_rt,
                    "l_gt_offline": n_hurt_off / n_gt_tracks,
                    "l_gt_runtime": n_hurt_rt / n_gt_tracks,
                    "ucb_offline": ucb_off,
                    "ucb_runtime": ucb_rt,
                    "offline_safe": safe_off,
                    "runtime_safe": safe_rt,
                    "n_fp_exposed_pairs": n_fp_pairs,
                    "n_fp_removed_offline_pairs": fp_removed_off,
                    "n_fp_removed_runtime_pairs": fp_removed_rt,
                    "g_fp_offline": (
                        fp_removed_off / n_fp_pairs if n_fp_pairs else float("nan")
                    ),
                    "g_fp_runtime": (
                        fp_removed_rt / n_fp_pairs if n_fp_pairs else float("nan")
                    ),
                    "active_offline_safe": active_offline_safe,
                    "dist_direction_flips": dist_flips,
                    "ratio_direction_flips": ratio_flips,
                    "region_jaccard": _jaccard(reject_off, reject_rt),
                    "unjoined_m": unjoined_m,
                    "v5_coverage_pass": unjoined_m == 0,
                }
            )

    rho_dist = _spearman(offline_dist, runtime_dist)
    rho_ratio = _spearman(offline_ratio, runtime_ratio)
    active_safe = [row for row in grid if row["active_offline_safe"]]
    offline_safe = [row for row in grid if row["offline_safe"]]

    v4_pass = n >= MIN_MATCHED_PAIRS and n_gt_tracks >= MIN_GT_TRACKS
    v7_pass = bool(active_safe)
    v5_evaluated = v7_pass
    v5_failed = v5_evaluated and any(not row["v5_coverage_pass"] for row in active_safe)
    v5_pass: bool | None = None if not v5_evaluated else not v5_failed

    if not v4_pass or not v7_pass or v5_failed:
        terminal = TERMINAL_UNDECIDABLE
    elif any(not row["runtime_safe"] for row in offline_safe):
        terminal = TERMINAL_BROKEN
    else:
        degraded = (
            not math.isfinite(rho_dist)
            or not math.isfinite(rho_ratio)
            or rho_dist < RHO_MIN
            or rho_ratio < RHO_MIN
            or any(
                row["dist_direction_flips"] > 0
                or row["ratio_direction_flips"] > 0
                or row["region_jaccard"] < JACCARD_MIN
                for row in offline_safe
            )
        )
        terminal = TERMINAL_DEGRADED if degraded else TERMINAL_HOLDS

    return {
        "terminal": terminal,
        "validity": {
            "V4_exposure_floor": v4_pass,
            "V5_adversarial_unjoined_coverage": v5_pass,
            "V5_evaluated": v5_evaluated,
            "V7_nonempty_active_safe_set": v7_pass,
        },
        "exposures": {
            "matched_pairs": n,
            "gt_tracks": n_gt_tracks,
            "fp_pairs": n_fp_pairs,
            "unjoined_events": len(unjoined_runtime_dist),
            "trial_unit": "lost_track(seq,lost_global_id)",
            "fp_reporting_unit": "matched_pair",
        },
        "axis_agreement": {
            "dist_h_spearman_rho": rho_dist,
            "abs_log_h_ratio_spearman_rho": rho_ratio,
            "minimum": RHO_MIN,
        },
        "grid_summary": {
            "points": len(grid),
            "offline_safe_points": len(offline_safe),
            "active_offline_safe_points": len(active_safe),
            "runtime_safe_among_offline_safe": sum(
                bool(row["runtime_safe"]) for row in offline_safe
            ),
            "active_safe_points_with_unjoined_m_gt_zero": sum(
                row["unjoined_m"] > 0 for row in active_safe
            ),
        },
        "grid": grid,
    }


def run(
    study_dir: Path, substrate_dir: Path, expected_hashes: dict[str, str]
) -> dict[str, Any]:
    hashes = verify_frozen_inputs(study_dir, substrate_dir, expected_hashes)
    pairs, capture = _read_inputs(study_dir)

    manifest_path = study_dir / "capture.csv.gz.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("provenance", {}).get("shadow") is not True:
        raise ValidityFailure("V1 capture is not declared shadow provenance")
    if int(manifest.get("overflow_events", -1)) != 0:
        raise ValidityFailure("V1 capture overflow is non-zero")

    unknown = {row.get("partition", "") for row in capture} - set(PARTITIONS)
    if unknown:
        raise ValidityFailure(f"V2 unknown partitions: {sorted(unknown)}")
    parts = {
        name: [row for row in capture if row["partition"] == name]
        for name in PARTITIONS
    }
    partition = {name: len(parts[name]) for name in PARTITIONS}
    if partition != EXPECTED_PARTITION or sum(partition.values()) != len(capture):
        raise ValidityFailure(
            f"V2 partition mismatch: expected {EXPECTED_PARTITION}, got {partition}"
        )

    pair_by_key: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in pairs:
        key = (row["seq"], int(row["lost_id"]), int(row["cand_id"]))
        if key in pair_by_key:
            raise ValidityFailure(f"V3 duplicate offline pair key: {key}")
        pair_by_key[key] = row

    keyed_capture: set[tuple[str, int, int]] = set()
    joined_pairs: list[dict[str, str]] = []
    matched = parts["matched"]
    for row in capture:
        if not row.get("event_key"):
            continue
        key = (
            row["seq"],
            int(row["lost_global_id"]),
            int(row["cand_global_id"]),
        )
        if key in keyed_capture:
            raise ValidityFailure(f"V3 duplicate capture key: {key}")
        keyed_capture.add(key)
    for row in matched:
        key = (
            row["seq"],
            int(row["lost_global_id"]),
            int(row["cand_global_id"]),
        )
        pair = pair_by_key.get(key)
        if pair is None:
            raise ValidityFailure(f"V3 matched row lacks offline pair: {key}")
        if "gt_match" not in pair or "gt_valid" not in pair:
            raise ValidityFailure(f"V3 matched row lacks GT flags: {key}")
        joined_pairs.append(pair)

    def values(rows: Sequence[dict[str, str]], field: str) -> np.ndarray:
        try:
            out = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidityFailure(f"V3 invalid numeric field {field}: {exc}") from exc
        if not np.all(np.isfinite(out)):
            raise ValidityFailure(f"V3 non-finite field {field}")
        return out

    off_dist = values(joined_pairs, "dist_h")
    off_h_lost = values(joined_pairs, "h_lost_raw")
    off_h_cand = values(joined_pairs, "h_cand_raw")
    rt_dist = values(matched, "dist_h")
    rt_h_lost = values(matched, "ema_lost")
    rt_h_cand = values(matched, "ema_cand")
    if np.any(off_h_lost <= 0) or np.any(off_h_cand <= 0):
        raise ValidityFailure("V3 offline raw heights must be positive")
    if np.any(rt_h_lost <= 0) or np.any(rt_h_cand <= 0):
        raise ValidityFailure("V3 runtime EMA heights must be positive")

    off_ratio = np.abs(np.log(off_h_lost / off_h_cand))
    rt_ratio = np.abs(np.log(rt_h_lost / rt_h_cand))
    gt_valid = np.asarray([_as_bool01(row["gt_valid"]) for row in joined_pairs])
    gt_match = np.asarray([_as_bool01(row["gt_match"]) for row in joined_pairs])
    is_gt = gt_valid & gt_match
    is_fp = gt_valid & ~gt_match
    track_keys = [(row["seq"], int(row["lost_global_id"])) for row in matched]

    unjoined = parts["cohort_gap"] + parts["unemitted"]
    unjoined_dist = values(unjoined, "dist_h")
    unjoined_h_lost = values(unjoined, "ema_lost")
    unjoined_h_cand = values(unjoined, "ema_cand")
    if np.any(unjoined_h_lost <= 0) or np.any(unjoined_h_cand <= 0):
        raise ValidityFailure("V3 unjoined runtime EMA heights must be positive")
    unjoined_ratio = np.abs(np.log(unjoined_h_lost / unjoined_h_cand))

    result = evaluate_arrays(
        offline_dist=off_dist,
        offline_ratio=off_ratio,
        runtime_dist=rt_dist,
        runtime_ratio=rt_ratio,
        is_gt=is_gt,
        is_fp=is_fp,
        track_keys=track_keys,
        unjoined_runtime_dist=unjoined_dist,
        unjoined_runtime_ratio=unjoined_ratio,
    )
    result.update(
        {
            "study": STUDY,
            "declaration": DECLARATION,
            "reviewed_head": REVIEWED_HEAD,
            "partition": partition,
            "input_hashes": hashes,
            "validity": {
                "V1_provenance": True,
                "V2_partition_conservation": True,
                "V3_join_integrity": True,
                **result["validity"],
                "V6_no_gt_leakage": True,
            },
            "conventions": {
                "epsilon": EPSILON,
                "confidence": CONFIDENCE,
                "rho_min": RHO_MIN,
                "jaccard_min": JACCARD_MIN,
                "dist_thresholds": DIST_THRESHOLDS,
                "abs_log_h_ratio_thresholds": RATIO_THRESHOLDS,
                "unjoined_events_are_statistical_trials": False,
                "selection": "none",
            },
        }
    )
    return result


def write_packet(output_dir: Path, metrics: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = metrics.pop("grid")
    metrics_path = output_dir / "metrics.json"
    grid_path = output_dir / "grid.csv"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with grid_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(grid)

    files = {"metrics.json": _sha256(metrics_path), "grid.csv": _sha256(grid_path)}
    manifest = {
        "study": STUDY,
        "terminal": metrics["terminal"],
        "declaration": DECLARATION,
        "reviewed_head": REVIEWED_HEAD,
        "runner": "scripts/tools/run_s0_safe_domain_runtime_transfer.py",
        "runner_sha256": _sha256(Path(__file__)),
        "files": files,
        "input_hashes": metrics["input_hashes"],
        "selection": "none; every frozen grid point evaluated",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files["manifest.json"] = _sha256(manifest_path)
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--substrate-dir", type=Path, required=True)
    parser.add_argument("--expected-hashes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else REPO / path

    expected = json.loads(absolute(args.expected_hashes).read_text(encoding="utf-8"))
    try:
        metrics = run(absolute(args.study_dir), absolute(args.substrate_dir), expected)
    except ValidityFailure as exc:
        print(json.dumps({"terminal": TERMINAL_INVALID, "failure": str(exc)}, indent=2))
        return 1

    summary = {
        "study": metrics["study"],
        "terminal": metrics["terminal"],
        "validity": metrics["validity"],
        "exposures": metrics["exposures"],
        "axis_agreement": metrics["axis_agreement"],
        "grid_summary": metrics["grid_summary"],
    }
    files = write_packet(absolute(args.output_dir), metrics)
    summary["files"] = files
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
