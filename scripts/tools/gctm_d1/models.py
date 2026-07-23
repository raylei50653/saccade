"""M0 / M1 / M2 score models under a frozen observation interface.

M0: deterministic Euclidean residual energy (baseline).
M1: gap-conditioned innovation Mahalanobis / NLL under the same mean residual.
M2: leakage-free context drift mean correction; every other interface field fixed.

Ranking orientation is lower_better for all native scores.
"""

# status: experiment

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ModelId = Literal["M0", "M1", "M2"]
CovMode = Literal["isotropic_shared", "anisotropic_shared", "candidate_specific"]


class FailClosedError(ValueError):
    """Undefined inverse / det / covariance / missing / tie behavior."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CandidateObservation:
    event_id: str
    cand_id: str
    residual: np.ndarray  # shape (d,) position innovation under M0/M1 mean
    delta: float  # gap in frames
    is_true_match: bool
    stratum: str  # e.g. short_gap | long_gap
    context_drift: np.ndarray | None = None  # M2 mean correction only; exit-causal
    cov_shared: np.ndarray | None = None  # (d,d) event-shared S
    cov_candidate: np.ndarray | None = None  # (d,d) candidate-specific S
    scale_alpha: float = 1.0  # shared isotropic/anisotropic rescaling factor


@dataclass(frozen=True)
class ScoredCandidate:
    event_id: str
    cand_id: str
    model_id: ModelId
    residual_used: np.ndarray
    q: float
    nll: float
    score_for_rank: float
    is_true_match: bool
    stratum: str
    delta: float


def _as_vector(x: Any, d: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (d,):
        raise FailClosedError(
            "shape_mismatch",
            f"{name} must have shape ({d},), got {arr.shape}",
        )
    if not np.all(np.isfinite(arr)):
        raise FailClosedError("non_finite", f"{name} contains non-finite values")
    return arr


def _as_cov(s: Any, d: int, name: str) -> np.ndarray:
    if s is None:
        raise FailClosedError("missing_covariance", f"{name} is required")
    arr = np.asarray(s, dtype=float)
    if arr.shape != (d, d):
        raise FailClosedError(
            "shape_mismatch",
            f"{name} must have shape ({d}, {d}), got {arr.shape}",
        )
    if not np.all(np.isfinite(arr)):
        raise FailClosedError("non_finite", f"{name} contains non-finite values")
    if not np.allclose(arr, arr.T, atol=1e-12):
        raise FailClosedError("asymmetric_covariance", f"{name} must be symmetric")
    # PSD / PD check via eigenvalues
    eig = np.linalg.eigvalsh(arr)
    if np.any(eig < -1e-10):
        raise FailClosedError("non_psd_covariance", f"{name} is not PSD")
    if np.any(eig <= 1e-12):
        raise FailClosedError(
            "singular_covariance",
            f"{name} is singular or numerically non-invertible",
        )
    return 0.5 * (arr + arr.T)


def mahalanobis_q(residual: np.ndarray, cov: np.ndarray) -> float:
    d = residual.shape[0]
    s = _as_cov(cov, d, "S")
    r = _as_vector(residual, d, "residual")
    try:
        inv = np.linalg.inv(s)
    except np.linalg.LinAlgError as exc:
        raise FailClosedError("undefined_inverse", "S^{-1} undefined") from exc
    q = float(r @ inv @ r)
    if not np.isfinite(q) or q < -1e-12:
        raise FailClosedError("undefined_q", f"q not well-defined: {q}")
    return max(q, 0.0)


def log_det(cov: np.ndarray) -> float:
    d = cov.shape[0]
    s = _as_cov(cov, d, "S")
    sign, ld = np.linalg.slogdet(s)
    if sign <= 0 or not np.isfinite(ld):
        raise FailClosedError("undefined_logdet", "log det S undefined")
    return float(ld)


def gaussian_nll(residual: np.ndarray, cov: np.ndarray) -> float:
    d = residual.shape[0]
    q = mahalanobis_q(residual, cov)
    ld = log_det(cov)
    return 0.5 * q + 0.5 * ld + 0.5 * d * np.log(2.0 * np.pi)


def resolve_covariance(cand: CandidateObservation, mode: CovMode) -> np.ndarray:
    d = int(np.asarray(cand.residual).reshape(-1).shape[0])
    alpha = float(cand.scale_alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise FailClosedError("invalid_scale", "scale_alpha must be positive finite")

    if mode == "candidate_specific":
        base = cand.cov_candidate if cand.cov_candidate is not None else cand.cov_shared
        s = _as_cov(base, d, "candidate_specific_S")
    else:
        s = _as_cov(cand.cov_shared, d, "shared_S")
        if mode == "isotropic_shared":
            # Enforce isotropic structure: α * σ^2 * I from mean diagonal.
            # If caller already supplied isotropic matrix, keep eigenvalues equal.
            diag_mean = float(np.mean(np.diag(s)))
            if diag_mean <= 0:
                raise FailClosedError("invalid_scale", "isotropic scale non-positive")
            s = diag_mean * np.eye(d)
    return alpha * s


def residual_for_model(cand: CandidateObservation, model_id: ModelId) -> np.ndarray:
    r0 = _as_vector(
        cand.residual, int(np.asarray(cand.residual).reshape(-1).shape[0]), "residual"
    )
    if model_id in ("M0", "M1"):
        return r0
    # M2: leakage-free context drift correction r' = r - H d_Δ(c).
    # Drift must be exit-causal; synthetic fixtures supply it explicitly.
    if cand.context_drift is None:
        raise FailClosedError(
            "missing_context_drift",
            "M2 requires candidate.context_drift (exit-causal)",
        )
    drift = _as_vector(cand.context_drift, r0.shape[0], "context_drift")
    return r0 - drift


def score_candidate(
    cand: CandidateObservation,
    model_id: ModelId,
    *,
    cov_mode: CovMode = "anisotropic_shared",
    rank_score: Literal["q", "nll", "euclid"] = "q",
) -> ScoredCandidate:
    residual = residual_for_model(cand, model_id)

    if model_id == "M0":
        # Deterministic baseline: Euclidean energy; covariance unused for ranking.
        q = float(residual @ residual)
        nll = q  # report same native value; not a calibrated NLL
        if rank_score == "euclid":
            score = q
        elif rank_score == "q":
            score = q
        else:
            score = nll
        return ScoredCandidate(
            event_id=cand.event_id,
            cand_id=cand.cand_id,
            model_id=model_id,
            residual_used=residual,
            q=q,
            nll=nll,
            score_for_rank=score,
            is_true_match=cand.is_true_match,
            stratum=cand.stratum,
            delta=cand.delta,
        )

    cov = resolve_covariance(cand, cov_mode)
    q = mahalanobis_q(residual, cov)
    nll = gaussian_nll(residual, cov)
    if rank_score == "q":
        score = q
    elif rank_score == "nll":
        score = nll
    else:
        score = float(residual @ residual)
    return ScoredCandidate(
        event_id=cand.event_id,
        cand_id=cand.cand_id,
        model_id=model_id,
        residual_used=residual,
        q=q,
        nll=nll,
        score_for_rank=score,
        is_true_match=cand.is_true_match,
        stratum=cand.stratum,
        delta=cand.delta,
    )


def rank_event(
    scored: list[ScoredCandidate],
    *,
    orientation: str = "lower_better",
    tie_rule: str = "stable_cand_id_asc",
) -> list[tuple[str, int, float]]:
    """Return (cand_id, rank, score) with rank 1 = best under orientation."""
    if not scored:
        raise FailClosedError("empty_event", "cannot rank empty event")
    if orientation != "lower_better":
        raise FailClosedError(
            "unsupported_orientation",
            "D1 freezes lower_better only",
        )
    if tie_rule != "stable_cand_id_asc":
        raise FailClosedError("unsupported_tie_rule", f"unknown tie_rule {tie_rule}")

    # Sort by score ascending, then cand_id ascending for deterministic ties.
    ordered = sorted(scored, key=lambda s: (s.score_for_rank, s.cand_id))
    out: list[tuple[str, int, float]] = []
    for rank, item in enumerate(ordered, start=1):
        out.append((item.cand_id, rank, item.score_for_rank))
    return out


def ordering_tuple(
    scored: list[ScoredCandidate],
    *,
    orientation: str = "lower_better",
    tie_rule: str = "stable_cand_id_asc",
) -> tuple[str, ...]:
    return tuple(
        cid
        for cid, _, _ in rank_event(scored, orientation=orientation, tie_rule=tie_rule)
    )


def is_strictly_monotone_increasing(xs: list[float], ys: list[float]) -> bool:
    """Return True if x_i < x_j implies y_i < y_j for all pairs (strict)."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return False
    for i in range(n):
        for j in range(n):
            if xs[i] < xs[j] and not (ys[i] < ys[j]):
                return False
            if xs[i] == xs[j] and ys[i] != ys[j]:
                # Strict monotone map cannot split ties; fail closed for ranking transforms.
                return False
    return True
