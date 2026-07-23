"""Synthetic fixture pack for GCTM D1 (immutable non-runtime substrate)."""

# status: experiment

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import CandidateObservation


FIXTURE_PACK_ID = "gctm_d1_synthetic_fixture_pack_v1"
FIXTURE_SCHEMA = "gctm_d1_fixture_pack_v1"
SUBSTRATE_CLASS = "synthetic"


def _eye_scaled(d: int, scale: float) -> list[list[float]]:
    return (scale * np.eye(d)).tolist()


def _aniso(d: int, major: float, minor: float) -> list[list[float]]:
    assert d == 2, "synthetic pack freezes d=2 for constructive counterexamples"
    return [[major, 0.0], [0.0, minor]]


def build_fixture_pack() -> dict[str, Any]:
    """Construct an immutable synthetic fixture pack with constructive cases.

    Events:
      E_shared_iso: shared isotropic S — calibration-only vs Euclidean
      E_shared_aniso: shared anisotropic S — ranking-active vs Euclidean
      E_cand_spec: candidate-specific S — ranking-active; q vs NLL can diverge
      E_short_gap: protected short-gap stratum
      E_long_gap: long-gap contrast
      E_m2_drift: M2 context-drift reorders relative to M1 under fixed interface
      E_tie: exact score tie resolved by cand_id
    """
    d = 2
    events: list[dict[str, Any]] = []

    # --- E_shared_iso: same ordering under M0 Euclidean and M1 isotropic ---
    events.append(
        {
            "event_id": "E_shared_iso",
            "stratum": "short_gap",
            "delta": 2.0,
            "cov_mode": "isotropic_shared",
            "shared_S": _eye_scaled(d, 4.0),
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_true",
                    "residual": [1.0, 0.0],
                    "is_true_match": True,
                    "context_drift": [0.0, 0.0],
                },
                {
                    "cand_id": "c_false",
                    "residual": [2.0, 0.0],
                    "is_true_match": False,
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "shared isotropic S preserves Euclidean order (calibration-only)",
        }
    )

    # --- E_shared_aniso: anisotropic shared S flips Euclidean order ---
    # r_true = [3, 0], r_false = [0, 1.1]
    # ||r_true||^2=9 > ||r_false||^2=1.21  (Euclidean prefers false)
    # S = diag(9, 1) => q_true=1, q_false=1.21  (Mahalanobis prefers true)
    events.append(
        {
            "event_id": "E_shared_aniso",
            "stratum": "short_gap",
            "delta": 3.0,
            "cov_mode": "anisotropic_shared",
            "shared_S": _aniso(d, 9.0, 1.0),
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_true",
                    "residual": [3.0, 0.0],
                    "is_true_match": True,
                    "context_drift": [0.0, 0.0],
                },
                {
                    "cand_id": "c_false",
                    "residual": [0.0, 1.1],
                    "is_true_match": False,
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "shared anisotropic S is ranking-active vs Euclidean M0",
        }
    )

    # --- E_cand_spec: candidate-specific S flips q vs NLL (D2 L5.2 style) ---
    # k=d=2 but same spirit: unequal S can flip q vs NLL order
    events.append(
        {
            "event_id": "E_cand_spec",
            "stratum": "long_gap",
            "delta": 12.0,
            "cov_mode": "candidate_specific",
            "shared_S": _eye_scaled(d, 1.0),  # unused when candidate S present
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_a",
                    "residual": [1.0, 0.0],
                    "is_true_match": True,
                    "candidate_S": _eye_scaled(d, 1.0),
                    "context_drift": [0.0, 0.0],
                },
                {
                    "cand_id": "c_b",
                    "residual": [1.2, 0.0],
                    "is_true_match": False,
                    "candidate_S": _eye_scaled(d, 4.0),
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "candidate-specific S: q prefers c_a; NLL may prefer c_b",
        }
    )

    # --- protected short-gap stratum event ---
    events.append(
        {
            "event_id": "E_short_gap",
            "stratum": "short_gap",
            "delta": 1.0,
            "cov_mode": "anisotropic_shared",
            "shared_S": _aniso(d, 4.0, 1.0),
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_true",
                    "residual": [0.5, 0.0],
                    "is_true_match": True,
                    "context_drift": [0.0, 0.0],
                },
                {
                    "cand_id": "c_false",
                    "residual": [0.0, 0.8],
                    "is_true_match": False,
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "protected short-gap stratum for aggregate-hiding guard",
        }
    )

    # --- long-gap contrast (aggregate can improve while short-gap harmed) ---
    events.append(
        {
            "event_id": "E_long_gap",
            "stratum": "long_gap",
            "delta": 20.0,
            "cov_mode": "anisotropic_shared",
            "shared_S": _aniso(d, 4.0, 1.0),
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_true",
                    "residual": [0.2, 0.0],
                    "is_true_match": True,
                    "context_drift": [0.0, 0.0],
                },
                {
                    "cand_id": "c_false",
                    "residual": [1.5, 0.0],
                    "is_true_match": False,
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "long-gap event; not a substitute for short-gap ranking",
        }
    )

    # --- M2 context drift reorders under fixed observation interface ---
    # Under M1, r_true=[2,0], r_false=[1,0] => false ranks better (bad).
    # M2 drift for true is [1.5,0] => residual becomes [0.5,0]; false drift 0.
    events.append(
        {
            "event_id": "E_m2_drift",
            "stratum": "short_gap",
            "delta": 4.0,
            "cov_mode": "anisotropic_shared",
            "shared_S": _eye_scaled(d, 1.0),
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_true",
                    "residual": [2.0, 0.0],
                    "is_true_match": True,
                    "context_drift": [1.5, 0.0],
                },
                {
                    "cand_id": "c_false",
                    "residual": [1.0, 0.0],
                    "is_true_match": False,
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "M2 leakage-free context drift changes within-event order",
        }
    )

    # --- exact tie event ---
    events.append(
        {
            "event_id": "E_tie",
            "stratum": "short_gap",
            "delta": 2.0,
            "cov_mode": "isotropic_shared",
            "shared_S": _eye_scaled(d, 1.0),
            "scale_alpha": 1.0,
            "candidates": [
                {
                    "cand_id": "c_b",
                    "residual": [1.0, 0.0],
                    "is_true_match": False,
                    "context_drift": [0.0, 0.0],
                },
                {
                    "cand_id": "c_a",
                    "residual": [1.0, 0.0],
                    "is_true_match": True,
                    "context_drift": [0.0, 0.0],
                },
            ],
            "notes": "exact score tie; stable cand_id_asc => c_a before c_b",
        }
    )

    pack = {
        "schema": FIXTURE_SCHEMA,
        "fixture_pack_id": FIXTURE_PACK_ID,
        "substrate_class": SUBSTRATE_CLASS,
        "coordinate_dim_d": d,
        "observation_mode": "H_x",
        "purpose": "gctm_d1_ranking_diagnostic_only",
        "non_runtime": True,
        "h0_forbidden": True,
        "events": events,
        "protected_strata": ["short_gap"],
        "counterexample_search_space": {
            "d": 2,
            "cov_modes": [
                "isotropic_shared",
                "anisotropic_shared",
                "candidate_specific",
            ],
            "models": ["M0", "M1", "M2"],
            "mechanisms": [
                "shared_isotropic_scalar_covariance",
                "anisotropic_shared_innovation_covariance",
                "candidate_specific_observation_covariance",
                "leakage_free_context_drift",
            ],
            "residual_domain": "R^2 synthetic finite vectors",
            "forbidden": [
                "pooled_row_independence_assumptions",
                "h0_runtime_rows",
                "post_reveal_refit",
            ],
        },
    }
    return pack


def pack_to_candidates(pack: dict[str, Any]) -> list[CandidateObservation]:
    out: list[CandidateObservation] = []
    d = int(pack["coordinate_dim_d"])
    for event in pack["events"]:
        shared = np.asarray(event["shared_S"], dtype=float)
        for cand in event["candidates"]:
            cand_s = cand.get("candidate_S")
            out.append(
                CandidateObservation(
                    event_id=event["event_id"],
                    cand_id=cand["cand_id"],
                    residual=np.asarray(cand["residual"], dtype=float),
                    delta=float(event["delta"]),
                    is_true_match=bool(cand["is_true_match"]),
                    stratum=str(event["stratum"]),
                    context_drift=np.asarray(
                        cand.get("context_drift", [0.0] * d), dtype=float
                    ),
                    cov_shared=shared,
                    cov_candidate=(
                        np.asarray(cand_s, dtype=float) if cand_s is not None else None
                    ),
                    scale_alpha=float(event.get("scale_alpha", 1.0)),
                )
            )
    return out


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def write_fixture_pack(path: Path) -> dict[str, Any]:
    pack = build_fixture_pack()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(pack) + b"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.with_suffix(path.suffix + ".sha256")).write_text(
        digest + "\n", encoding="utf-8"
    )
    return {"pack": pack, "path": str(path), "sha256": digest}
