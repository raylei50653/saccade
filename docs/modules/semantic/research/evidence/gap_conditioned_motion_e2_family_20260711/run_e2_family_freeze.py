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
    if d.shape[0] < 3 or not np.all(np.isfinite(d)):
        raise ValueError("at least three finite training displacements are required")
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
    tolerance = 1e-12 * max(1.0, abs(best))
    return next(
        model_id
        for model_id in MODEL_ORDER
        if training_total_nll[model_id] <= best + tolerance
    )


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
    valid = not any(
        counts[name]
        for name in (
            "invalid_nonfinite",
            "invalid_h_ref",
            "invalid_gap",
            "invalid_frame_window",
        )
    )
    return {
        **counts,
        "sequence_count": len(sequences),
        "finite_support_gate": "PASS" if valid else "FAIL",
        "coordinate_semantics": "foot-point displacement divided by pair h_ref",
        "time_semantics": "integer frame gap = cand_first_frame - lost_last_frame",
    }


def family_spec(source_sha: str, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "freeze_id": "GCM-E2-POSITION-ONLY-v1",
        "status": "FROZEN_PENDING_RESEARCH_ACCEPTANCE",
        "claim_ceiling": (
            "position-only global transition marginal; not joint x/v, not "
            "sequence-conditioned LOO, not a V1-V5 verdict"
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
            "large_search": False,
        },
        "regularization": {
            "covariance": "full 2x2 MLE after kernel standardization",
            "eigen_floor": "max(1e-8, 1e-6 * max(trace(Sigma)/2, 0))",
            "same_for_gt_and_fp": True,
            "must_record_flag_and_floor": True,
        },
        "required_signal_fields": [
            "model_id",
            "parameter_artifact_id",
            "fold_id",
            "fit_row_count",
            "dimension",
            "q_motion",
            "log_det_covariance",
            "gaussian_constant",
            "nll_motion",
            "regularization_applied",
        ],
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
    if audit["finite_support_gate"] != "PASS":
        raise ValueError("E2 finite/support/window gate failed")
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
                f"models={','.join(model['model_id'] for model in spec['models'])}",
                "headline_context=global",
                "phase_b_authorized=false",
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
