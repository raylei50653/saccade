#!/usr/bin/env python3
"""Freeze and verify the position-only M1-P/M2-P motion family.

This runner intentionally emits no fitted headline result.  It owns the E2
model equations, fit/scoring primitives, substrate gates, and byte-stable
family packet that E3 must consume without redefining the family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[6]
PACKET_DIR = Path(__file__).resolve().parent
CANONICAL_PAIRS = Path(
    "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
)
SOURCE_SHA256 = "0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17"
REQUIRED_FIELDS = {
    "seq",
    "lost_id",
    "cand_id",
    "gt_match",
    "gt_valid",
    "gap",
    "lost_last_frame",
    "cand_first_frame",
    "lost_foot_x",
    "lost_foot_y",
    "cand_foot_x",
    "cand_foot_y",
    "h_ref",
}
MODEL_ORDER = (
    "M1P-GLOBAL-CV",
    "M2P-GLOBAL-OU-H270",
    "M2P-GLOBAL-OU-H90",
    "M2P-GLOBAL-OU-H30",
)
HALF_LIFE_BY_MODEL = {
    "M2P-GLOBAL-OU-H30": 30.0,
    "M2P-GLOBAL-OU-H90": 90.0,
    "M2P-GLOBAL-OU-H270": 270.0,
}
DIMENSION = 2
EIGEN_ABS_FLOOR = 1e-8
EIGEN_REL_FLOOR = 1e-6
SELECTION_REL_TOLERANCE = 1e-12
MINIMUM_FIT_ROWS = 3


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


def kernel_scale(model_id: str, gaps: np.ndarray) -> np.ndarray:
    """Return scalar covariance growth k(dt) for a frozen family member."""
    t = np.asarray(gaps, dtype=np.float64)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("gaps must be a finite positive 1D array")
    if model_id == "M1P-GLOBAL-CV":
        return t * t
    if model_id not in HALF_LIFE_BY_MODEL:
        raise ValueError(f"unknown frozen model_id: {model_id}")
    gamma = math.log(2.0) / HALF_LIFE_BY_MODEL[model_id]
    z = gamma * t
    # 2 * (z - 1 + exp(-z)) / gamma^2, written with expm1 and a
    # short series to avoid cancellation near the M1 (gamma -> 0) limit.
    core = z + np.expm1(-z)
    small = np.abs(z) < 1e-4
    if np.any(small):
        zs = z[small]
        core[small] = 0.5 * zs**2 - zs**3 / 6.0 + zs**4 / 24.0
    return 2.0 * core / (gamma * gamma)


def _regularize_covariance(covariance: np.ndarray) -> tuple[np.ndarray, bool, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.trace(symmetric)) / DIMENSION, 0.0)
    floor = max(EIGEN_ABS_FLOOR, EIGEN_REL_FLOOR * scale)
    clipped = np.maximum(values, floor)
    regularized = bool(np.any(clipped != values))
    return (vectors * clipped) @ vectors.T, regularized, floor


def fit_model(
    model_id: str, displacements: np.ndarray, gaps: np.ndarray
) -> dict[str, Any]:
    """Fit one family member on training-fold GT transitions only."""
    d = np.asarray(displacements, dtype=np.float64)
    t = np.asarray(gaps, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != DIMENSION or d.shape[0] != t.shape[0]:
        raise ValueError("displacements must have shape (n, 2) aligned with gaps")
    if d.shape[0] < MINIMUM_FIT_ROWS or not np.all(np.isfinite(d)):
        raise ValueError(
            f"at least {MINIMUM_FIT_ROWS} finite training displacements are required"
        )
    k = kernel_scale(model_id, t)
    denominator = float(np.sum(t * t / k))
    drift = np.sum((t / k)[:, None] * d, axis=0) / denominator
    standardized = (d - t[:, None] * drift) / np.sqrt(k)[:, None]
    covariance = standardized.T @ standardized / d.shape[0]
    covariance, regularized, floor = _regularize_covariance(covariance)
    return {
        "schema_version": 1,
        "model_id": model_id,
        "dimension": DIMENSION,
        "drift_per_frame": drift.tolist(),
        "base_covariance": covariance.tolist(),
        "n_fit_gt": int(d.shape[0]),
        "regularization_applied": regularized,
        "eigenvalue_floor": floor,
    }


def score_model(
    artifact: dict[str, Any], displacements: np.ndarray, gaps: np.ndarray
) -> dict[str, np.ndarray]:
    """Score transitions while retaining q, log-det, and constant separately."""
    model_id = str(artifact["model_id"])
    d = np.asarray(displacements, dtype=np.float64)
    t = np.asarray(gaps, dtype=np.float64)
    if d.ndim != 2 or d.shape != (t.shape[0], DIMENSION):
        raise ValueError("displacements must have shape (n, 2) aligned with gaps")
    drift = np.asarray(artifact["drift_per_frame"], dtype=np.float64)
    base_covariance = np.asarray(artifact["base_covariance"], dtype=np.float64)
    inverse = np.linalg.inv(base_covariance)
    sign, base_logdet = np.linalg.slogdet(base_covariance)
    if sign <= 0:
        raise ValueError("base covariance must be positive definite")
    k = kernel_scale(model_id, t)
    innovation = d - t[:, None] * drift
    q = np.einsum("ni,ij,nj->n", innovation, inverse, innovation) / k
    log_det = DIMENSION * np.log(k) + base_logdet
    constant = np.full_like(q, DIMENSION * math.log(2.0 * math.pi))
    nll = 0.5 * (q + log_det + constant)
    return {
        "q_motion": q,
        "log_det_covariance": log_det,
        "gaussian_constant": constant,
        "nll_motion": nll,
    }


def select_family_member(training_total_nll: dict[str, float]) -> str:
    """Apply the frozen train-only selector; order resolves numerical ties."""
    if set(training_total_nll) != set(MODEL_ORDER):
        raise ValueError("selector requires exactly the four frozen family members")
    if not all(math.isfinite(value) for value in training_total_nll.values()):
        raise ValueError("training NLL values must be finite")
    best = min(training_total_nll.values())
    tolerance = SELECTION_REL_TOLERANCE * max(1.0, abs(best))
    return next(
        model_id
        for model_id in MODEL_ORDER
        if training_total_nll[model_id] <= best + tolerance
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fit_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["seq"]), str(row["lost_id"]), str(row["cand_id"])


def fit_row_key_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the sorted `(seq, lost_id, cand_id)` fit-row lineage."""
    keys = [list(_fit_row_key(row)) for row in sorted(rows, key=_fit_row_key)]
    return _canonical_json_sha256(keys)


def load_gt_fit_rows(pairs: Path) -> list[dict[str, Any]]:
    """Load the only rows eligible for fitting, with deterministic ordering."""
    rows: list[dict[str, Any]] = []
    with pairs.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing E2 fields: {missing}")
        for source_row in reader:
            if not (
                _as_bool(source_row["gt_valid"]) and _as_bool(source_row["gt_match"])
            ):
                continue
            gap = int(source_row["gap"])
            h_ref = float(source_row["h_ref"])
            endpoints = np.asarray(
                [
                    float(source_row["lost_foot_x"]),
                    float(source_row["lost_foot_y"]),
                    float(source_row["cand_foot_x"]),
                    float(source_row["cand_foot_y"]),
                ],
                dtype=np.float64,
            )
            if (
                h_ref <= 0
                or not math.isfinite(h_ref)
                or not np.all(np.isfinite(endpoints))
            ):
                raise ValueError("invalid GT fit row passed the E2 support contract")
            displacement = [
                float((endpoints[2] - endpoints[0]) / h_ref),
                float((endpoints[3] - endpoints[1]) / h_ref),
            ]
            if gap < 1 or gap > 300 or not np.all(np.isfinite(displacement)):
                raise ValueError("invalid GT fit row passed the E2 support contract")
            rows.append(
                {
                    "seq": source_row["seq"],
                    "lost_id": source_row["lost_id"],
                    "cand_id": source_row["cand_id"],
                    "gap": gap,
                    "displacement": displacement,
                }
            )
    return sorted(rows, key=_fit_row_key)


def build_fold_artifacts(
    pairs: Path, held_out_sequence: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build auditable train-only parameter and selection artifacts for one LOO fold."""
    source_sha = sha256(pairs)
    all_rows = load_gt_fit_rows(pairs)
    all_sequences = sorted({str(row["seq"]) for row in all_rows})
    if held_out_sequence not in all_sequences:
        raise ValueError(f"unknown held-out sequence: {held_out_sequence}")
    train_rows = [row for row in all_rows if row["seq"] != held_out_sequence]
    train_sequences = sorted({str(row["seq"]) for row in train_rows})
    if len(train_rows) < MINIMUM_FIT_ROWS:
        raise ValueError("LOO fold has insufficient training support")

    displacements = np.asarray(
        [row["displacement"] for row in train_rows], dtype=np.float64
    )
    gaps = np.asarray([row["gap"] for row in train_rows], dtype=np.float64)
    fold_id = f"LOO::{held_out_sequence}"
    lineage_hash = fit_row_key_sha256(train_rows)
    parameter_artifacts: list[dict[str, Any]] = []
    training_nll_by_model: dict[str, float] = {}

    for model_id in MODEL_ORDER:
        fitted = fit_model(model_id, displacements, gaps)
        scores = score_model(fitted, displacements, gaps)
        training_total_nll = float(np.sum(scores["nll_motion"]))
        payload = {
            "schema_version": 1,
            "freeze_id": "GCM-E2-POSITION-ONLY-v1",
            "model_id": model_id,
            "fold_id": fold_id,
            "held_out_sequence": held_out_sequence,
            "train_sequences": train_sequences,
            "fit_row_count": len(train_rows),
            "fit_row_key_sha256": lineage_hash,
            "source_pairs_sha256": source_sha,
            "dimension": fitted["dimension"],
            "drift_per_frame": fitted["drift_per_frame"],
            "base_covariance": fitted["base_covariance"],
            "regularization_applied": fitted["regularization_applied"],
            "eigenvalue_floor": fitted["eigenvalue_floor"],
            "training_total_nll": training_total_nll,
        }
        artifact = {
            **payload,
            "parameter_artifact_id": f"sha256:{_canonical_json_sha256(payload)}",
        }
        parameter_artifacts.append(artifact)
        training_nll_by_model[model_id] = training_total_nll

    selected_model_id = select_family_member(training_nll_by_model)
    selection_payload = {
        "schema_version": 1,
        "freeze_id": "GCM-E2-POSITION-ONLY-v1",
        "fold_id": fold_id,
        "held_out_sequence": held_out_sequence,
        "train_sequences": train_sequences,
        "fit_row_count": len(train_rows),
        "fit_row_key_sha256": lineage_hash,
        "source_pairs_sha256": source_sha,
        "training_nll_by_model": training_nll_by_model,
        "selected_model_id": selected_model_id,
        "selection_tolerance": SELECTION_REL_TOLERANCE,
        "model_order": list(MODEL_ORDER),
    }
    selection_artifact = {
        **selection_payload,
        "selection_artifact_id": f"sha256:{_canonical_json_sha256(selection_payload)}",
    }
    return parameter_artifacts, selection_artifact


def validate_loo_artifact_contract(pairs: Path, audit: dict[str, Any]) -> None:
    """Exercise every canonical fold and match it to the sealed lineage map."""
    for held_out, expected_count in audit[
        "loo_training_fit_rows_by_held_out_sequence"
    ].items():
        parameters, selection = build_fold_artifacts(pairs, held_out)
        expected_hash = audit["loo_fit_row_key_sha256_by_held_out_sequence"][held_out]
        if len(parameters) != len(MODEL_ORDER):
            raise ValueError(f"LOO artifact model count mismatch for {held_out}")
        if set(selection["training_nll_by_model"]) != set(MODEL_ORDER):
            raise ValueError(f"LOO selection family mismatch for {held_out}")
        if (
            selection["fit_row_count"] != expected_count
            or selection["fit_row_key_sha256"] != expected_hash
        ):
            raise ValueError(f"LOO selection lineage mismatch for {held_out}")
        for parameter in parameters:
            if (
                parameter["held_out_sequence"] != held_out
                or held_out in parameter["train_sequences"]
                or parameter["fit_row_count"] != expected_count
                or parameter["fit_row_key_sha256"] != expected_hash
            ):
                raise ValueError(f"LOO parameter lineage mismatch for {held_out}")


def audit_substrate(pairs: Path) -> dict[str, Any]:
    counts = {
        "rows_total": 0,
        "rows_gt_valid": 0,
        "rows_gt_fit_eligible": 0,
        "invalid_nonfinite": 0,
        "invalid_h_ref": 0,
        "invalid_gap": 0,
        "invalid_frame_window": 0,
    }
    sequences: set[str] = set()
    gt_fit_rows_by_sequence: dict[str, int] = {}
    with pairs.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing E2 fields: {missing}")
        for row in reader:
            counts["rows_total"] += 1
            if not _as_bool(row["gt_valid"]):
                continue
            counts["rows_gt_valid"] += 1
            sequences.add(row["seq"])
            gap = int(row["gap"])
            if gap < 1 or gap > 300:
                counts["invalid_gap"] += 1
            if int(row["cand_first_frame"]) - int(row["lost_last_frame"]) != gap:
                counts["invalid_frame_window"] += 1
            values = np.asarray(
                [
                    float(row["lost_foot_x"]),
                    float(row["lost_foot_y"]),
                    float(row["cand_foot_x"]),
                    float(row["cand_foot_y"]),
                    float(row["h_ref"]),
                ]
            )
            if not np.all(np.isfinite(values)):
                counts["invalid_nonfinite"] += 1
            if float(row["h_ref"]) <= 0:
                counts["invalid_h_ref"] += 1
            if _as_bool(row["gt_match"]):
                counts["rows_gt_fit_eligible"] += 1
                gt_fit_rows_by_sequence[row["seq"]] = (
                    gt_fit_rows_by_sequence.get(row["seq"], 0) + 1
                )
    valid = not any(
        counts[name]
        for name in (
            "invalid_nonfinite",
            "invalid_h_ref",
            "invalid_gap",
            "invalid_frame_window",
        )
    )
    fit_rows = load_gt_fit_rows(pairs)
    loo_training_counts: dict[str, int] = {}
    loo_training_hashes: dict[str, str] = {}
    for held_out in sorted(sequences):
        train_rows = [row for row in fit_rows if row["seq"] != held_out]
        loo_training_counts[held_out] = len(train_rows)
        loo_training_hashes[held_out] = fit_row_key_sha256(train_rows)
    loo_support_ok = bool(loo_training_counts) and all(
        count >= MINIMUM_FIT_ROWS for count in loo_training_counts.values()
    )
    return {
        **counts,
        "sequence_count": len(sequences),
        "finite_support_gate": "PASS" if valid else "FAIL",
        "minimum_fit_rows_required": MINIMUM_FIT_ROWS,
        "gt_fit_rows_by_sequence": dict(sorted(gt_fit_rows_by_sequence.items())),
        "loo_training_fit_rows_by_held_out_sequence": loo_training_counts,
        "loo_fit_row_key_sha256_by_held_out_sequence": loo_training_hashes,
        "loo_minimum_support_gate": "PASS" if loo_support_ok else "FAIL",
        "coordinate_semantics": "foot-point displacement divided by pair h_ref",
        "time_semantics": "integer frame gap = cand_first_frame - lost_last_frame",
    }


def family_spec(source_sha: str, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "freeze_id": "GCM-E2-POSITION-ONLY-v1",
        "status": "FROZEN_ACCEPTED_WITH_LIMITS",
        "research_acceptance": "ACCEPTED_WITH_LIMITS",
        "e3_signal_generation": "AUTHORIZED",
        "phase_b_authorized": False,
        "claim_ceiling": (
            "position-only global transition marginal; E3 signal generation "
            "authorized under sealed LOO/output contracts; not joint x/v, not "
            "sequence-conditioned LOO, not Phase B, not a V1-V5 verdict"
        ),
        "source": {
            "pairs_csv": str(CANONICAL_PAIRS),
            "sha256": source_sha,
            "audit": audit,
        },
        "observation": {
            "symbol": "d",
            "definition": "(cand_foot_xy - lost_foot_xy) / h_ref",
            "dimension": DIMENSION,
            "coordinate_system": "image x-right/y-down; dimensionless by pair h_ref",
            "delta_t": "gap in frames; allowed integer range 1..300",
        },
        "mean": "E[d|dt] = beta * dt; beta is global train-GT weighted MLE",
        "models": [
            {
                "model_id": "M1P-GLOBAL-CV",
                "equation": "d|dt ~ N(beta*dt, dt^2 * Sigma_v)",
                "interpretation": "global random constant residual velocity marginal",
            },
            *[
                {
                    "model_id": f"M2P-GLOBAL-OU-H{half_life}",
                    "half_life_frames": half_life,
                    "gamma": f"log(2)/{half_life}",
                    "equation": (
                        "d|dt ~ N(beta*dt, k_gamma(dt)*Sigma_u); "
                        "k_gamma=2*(gamma*dt-1+exp(-gamma*dt))/gamma^2"
                    ),
                    "interpretation": "integrated stationary residual OU marginal",
                }
                for half_life in (30, 90, 270)
            ],
        ],
        "fit_protocol": {
            "fit_rows": "gt_valid AND gt_match from training sequences only",
            "headline_context": "global only",
            "loo_firewall": (
                "held-out sequence contributes no fit, covariance, calibration, "
                "fallback, or family selection statistic"
            ),
            "selection": (
                "minimum summed training-GT NLL across the four frozen members; "
                "ties within 1e-12 relative use declared model order"
            ),
            "model_order": list(MODEL_ORDER),
            "selection_tolerance": SELECTION_REL_TOLERANCE,
            "fit_row_key": "sorted canonical JSON of [seq,lost_id,cand_id] rows",
            "large_search": False,
        },
        "regularization": {
            "covariance": "full 2x2 MLE after kernel standardization",
            "eigen_floor": "max(1e-8, 1e-6 * max(trace(Sigma)/2, 0))",
            "same_for_gt_and_fp": True,
            "must_record_flag_and_floor": True,
        },
        "required_parameter_artifact_fields": [
            "freeze_id",
            "model_id",
            "parameter_artifact_id",
            "fold_id",
            "held_out_sequence",
            "train_sequences",
            "fit_row_count",
            "fit_row_key_sha256",
            "source_pairs_sha256",
            "dimension",
            "drift_per_frame",
            "base_covariance",
            "regularization_applied",
            "eigenvalue_floor",
            "training_total_nll",
        ],
        "required_fold_selection_artifact_fields": [
            "selection_artifact_id",
            "fold_id",
            "held_out_sequence",
            "train_sequences",
            "fit_row_count",
            "fit_row_key_sha256",
            "source_pairs_sha256",
            "training_nll_by_model",
            "selected_model_id",
            "selection_tolerance",
            "model_order",
        ],
        "required_signal_fields": [
            "freeze_id",
            "model_id",
            "parameter_artifact_id",
            "fold_id",
            "q_motion",
            "log_det_covariance",
            "gaussian_constant",
            "nll_motion",
        ],
        "e3_score_retention": (
            "emit pair scores for all four frozen members; selected_model_id is "
            "an additional fold marker and must not filter non-winners"
        ),
        "blocked": [
            "velocity-only or joint position-velocity likelihood",
            "sequence-conditioned LOO headline",
            "held-out retuning or calibration",
            "Phase B analysis or V1-V5 verdict",
            "tracker, hook, preset, baseline, or production change",
        ],
    }


def _stable_spec(pairs: Path) -> dict[str, Any]:
    source_sha = sha256(pairs)
    if source_sha != SOURCE_SHA256:
        raise ValueError(
            f"source SHA mismatch: expected {SOURCE_SHA256}, got {source_sha}"
        )
    audit = audit_substrate(pairs)
    if (
        audit["finite_support_gate"] != "PASS"
        or audit["loo_minimum_support_gate"] != "PASS"
    ):
        raise ValueError("E2 finite/support/window gate failed")
    validate_loo_artifact_contract(pairs, audit)
    audit["loo_artifact_contract_gate"] = "PASS"
    return family_spec(source_sha, audit)


def _render(spec: dict[str, Any]) -> str:
    audit = spec["source"]["audit"]
    return (
        "\n".join(
            [
                f"freeze_id={spec['freeze_id']}",
                f"status={spec['status']}",
                f"source_sha256={spec['source']['sha256']}",
                f"support_gate={audit['finite_support_gate']}",
                f"fit_eligible_gt={audit['rows_gt_fit_eligible']}",
                f"loo_support_gate={audit['loo_minimum_support_gate']}",
                f"loo_artifact_gate={audit['loo_artifact_contract_gate']}",
                f"models={','.join(model['model_id'] for model in spec['models'])}",
                "headline_context=global",
                f"e3_authorized={str(spec.get('e3_signal_generation') == 'AUTHORIZED').lower()}",
                f"phase_b_authorized={str(bool(spec.get('phase_b_authorized'))).lower()}",
            ]
        )
        + "\n"
    )


def _write_packet(spec: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "model_family.json"
    output_path = output_dir / "recorded_output.txt"
    spec_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_path.write_text(_render(spec), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "freeze_id": spec["freeze_id"],
        "status": spec["status"],
        "source_pairs_csv": str(CANONICAL_PAIRS),
        "source_pairs_csv_sha256": spec["source"]["sha256"],
        "runner_sha256": sha256(Path(__file__)),
        "artifacts": {
            "model_family.json": sha256(spec_path),
            "recorded_output.txt": sha256(output_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def verify(pairs: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_packet(_stable_spec(pairs), tmp_path)
        for name in ("model_family.json", "recorded_output.txt", "manifest.json"):
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
        print("E2 family packet verification: PASS")
        return
    spec = _stable_spec(pairs)
    _write_packet(spec, args.output_dir)
    print(_render(spec), end="")


if __name__ == "__main__":
    main()
