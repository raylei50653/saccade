#!/usr/bin/env python3
"""Preflight and math core for observability-weighted directional likelihood.

The formal B1 study remains locked until its research declaration is sealed.
This entry point therefore exposes only identity validation via ``--check-only``;
the pure functions below are implemented and tested on synthetic inputs without
reading outcome rows from the frozen study table.
"""

# status: experiment

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import i0e

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.validate_score_ranking_declaration import (  # noqa: E402
    validate_declaration_file,
)


DEFAULT_STUDY_SPEC = (
    ROOT / "docs/modules/semantic/research/"
    "observability_weighted_directional_likelihood_study_v1.json"
)
DEFAULT_SCORE_DECLARATION = (
    ROOT / "docs/modules/semantic/research/"
    "observability_weighted_directional_likelihood_declaration_20260827.score.json"
)
STUDY_SCHEMA = "observability_weighted_directional_likelihood_study_v1"
STUDY_SCHEMA_PATH = (
    ROOT / "scripts/tools/"
    "observability_weighted_directional_likelihood_study_schema_v1.json"
)
STUDY_ID = "owdl_m_b1_v1"
EXPECTED_SOURCE_ROLES = {
    "out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv": "pair_table",
    "out/signal_study/m_b1_smoke_20260709T092543Z/context.json": "source_context",
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-02-SDP.txt": (
        "trajectory_mot17_02_sdp"
    ),
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-04-SDP.txt": (
        "trajectory_mot17_04_sdp"
    ),
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-05-SDP.txt": (
        "trajectory_mot17_05_sdp"
    ),
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-09-SDP.txt": (
        "trajectory_mot17_09_sdp"
    ),
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-10-SDP.txt": (
        "trajectory_mot17_10_sdp"
    ),
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-11-SDP.txt": (
        "trajectory_mot17_11_sdp"
    ),
    "results/MOT17_eval_m_b1_substrate_20260709T092543Z/MOT17-13-SDP.txt": (
        "trajectory_mot17_13_sdp"
    ),
}


class ObservabilityError(ValueError):
    """One declared estimator or input-identity invariant was violated."""


@dataclass(frozen=True)
class OlsMotion:
    """Two-dimensional OLS motion fit and its linear slope weights."""

    velocity: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    slope_weights: np.ndarray


@dataclass(frozen=True)
class DirectionObservation:
    """Declared pair-local direction quantities; lower costs are better."""

    velocity: np.ndarray
    displacement_rate: np.ndarray
    velocity_covariance: np.ndarray
    displacement_covariance: np.ndarray
    velocity_displacement_cross_covariance: np.ndarray
    q_v: float
    delta_angle: float | None
    angular_variance: float
    kappa: float
    raw_direction_cost: float
    weighted_direction_cost: float


def _finite_array(
    value: object, *, shape_tail: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < len(shape_tail) or array.shape[-len(shape_tail) :] != shape_tail:
        raise ObservabilityError(
            f"{name} must end in shape {shape_tail}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ObservabilityError(f"{name} contains non-finite values")
    return array


def _spd_covariance(value: object, *, name: str) -> np.ndarray:
    covariance = _finite_array(value, shape_tail=(2, 2), name=name)
    if covariance.shape != (2, 2):
        raise ObservabilityError(f"{name} must have shape (2, 2)")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ObservabilityError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ObservabilityError(f"{name} must be positive definite") from exc
    return covariance


def fit_ols_motion(points: object, frames: object) -> OlsMotion:
    """Fit position = intercept + velocity * frame using actual frame values."""

    xy = _finite_array(points, shape_tail=(2,), name="points")
    time = np.asarray(frames, dtype=np.float64)
    if xy.ndim != 2 or len(xy) < 3:
        raise ObservabilityError("points must have shape (n, 2) with n >= 3")
    if time.shape != (len(xy),) or not np.all(np.isfinite(time)):
        raise ObservabilityError("frames must be one finite value per point")
    if np.any(np.diff(time) <= 0):
        raise ObservabilityError("frames must be strictly increasing")

    centered_time = time - float(np.mean(time))
    design = np.column_stack((np.ones(len(time), dtype=np.float64), centered_time))
    gram = design.T @ design
    if np.linalg.matrix_rank(gram) != 2:
        raise ObservabilityError("OLS design is rank deficient")
    inverse_gram = np.linalg.inv(gram)
    coefficients = inverse_gram @ design.T @ xy
    fitted = design @ coefficients
    slope_weights = (inverse_gram @ design.T)[1]
    return OlsMotion(
        velocity=coefficients[1].copy(),
        fitted=fitted,
        residuals=xy - fitted,
        slope_weights=slope_weights,
    )


def estimate_normalized_noise_covariance(
    windows: Iterable[tuple[object, object, object]],
) -> np.ndarray:
    """Pool height-normalized 4-point OLS residuals with residual-DoF scaling."""

    scatter = np.zeros((2, 2), dtype=np.float64)
    residual_dof = 0
    window_count = 0
    for points, frames, heights in windows:
        xy = _finite_array(points, shape_tail=(2,), name="window points")
        height = np.asarray(heights, dtype=np.float64)
        if xy.shape != (4, 2) or height.shape != (4,):
            raise ObservabilityError(
                "every calibration window must have four points/heights"
            )
        if not np.all(np.isfinite(height)) or np.any(height <= 0):
            raise ObservabilityError("window heights must be finite and positive")
        fit = fit_ols_motion(xy, frames)
        normalized_residual = fit.residuals / height[:, None]
        scatter += normalized_residual.T @ normalized_residual
        residual_dof += len(xy) - 2
        window_count += 1
    if window_count == 0 or residual_dof <= 0:
        raise ObservabilityError("at least one valid calibration window is required")
    return _spd_covariance(scatter / residual_dof, name="normalized noise covariance")


def _angle_gradient(vector: np.ndarray) -> np.ndarray | None:
    norm_squared = float(vector @ vector)
    if norm_squared == 0.0:
        return None
    return np.array([-vector[1], vector[0]], dtype=np.float64) / norm_squared


def _mahalanobis(vector: np.ndarray, covariance: np.ndarray) -> float:
    solved = np.linalg.solve(covariance, vector)
    value = float(vector @ solved)
    if value < -1e-12:
        raise ObservabilityError("Mahalanobis evidence became negative")
    return max(0.0, value)


def uniform_relative_von_mises_nll(delta_angle: float, kappa: float) -> float:
    """Return log(I0(kappa)) - kappa*cos(delta), zero under uniform direction."""

    if not math.isfinite(delta_angle):
        raise ObservabilityError("delta_angle must be finite")
    if not math.isfinite(kappa) or kappa < 0:
        raise ObservabilityError("kappa must be finite and non-negative")
    if kappa == 0.0:
        return 0.0
    scaled_i0 = float(i0e(kappa))
    if not math.isfinite(scaled_i0) or scaled_i0 <= 0:
        raise ObservabilityError("stable I0 evaluation failed")
    log_i0 = math.log(scaled_i0) + kappa
    return log_i0 - kappa * math.cos(delta_angle)


def observe_direction(
    *,
    lost_points: object,
    lost_frames: object,
    lost_heights: object,
    candidate_first_point: object,
    candidate_first_height: float,
    gap: int,
    normalized_noise_covariance: object,
) -> DirectionObservation:
    """Build the declared covariance-aware angular observation for one pair."""

    points = _finite_array(lost_points, shape_tail=(2,), name="lost_points")
    heights = np.asarray(lost_heights, dtype=np.float64)
    candidate_point = _finite_array(
        candidate_first_point, shape_tail=(2,), name="candidate_first_point"
    )
    if points.shape != (4, 2) or heights.shape != (4,):
        raise ObservabilityError(
            "the tested estimator requires exactly four lost points"
        )
    if candidate_point.shape != (2,):
        raise ObservabilityError("candidate_first_point must have shape (2,)")
    if not np.all(np.isfinite(heights)) or np.any(heights <= 0):
        raise ObservabilityError("lost_heights must be finite and positive")
    if not math.isfinite(candidate_first_height) or candidate_first_height <= 0:
        raise ObservabilityError("candidate_first_height must be finite and positive")
    if isinstance(gap, bool) or not isinstance(gap, Integral) or int(gap) <= 0:
        raise ObservabilityError("gap must be a positive integer")

    normalized_covariance = _spd_covariance(
        normalized_noise_covariance, name="normalized_noise_covariance"
    )
    fit = fit_ols_motion(points, lost_frames)
    point_covariances = np.asarray(
        [normalized_covariance * float(height * height) for height in heights]
    )
    candidate_covariance = normalized_covariance * candidate_first_height**2

    velocity_covariance = np.sum(
        fit.slope_weights[:, None, None] ** 2 * point_covariances,
        axis=0,
    )
    velocity_covariance = _spd_covariance(
        velocity_covariance, name="velocity_covariance"
    )
    displacement_rate = (candidate_point - points[-1]) / float(gap)
    displacement_covariance = (point_covariances[-1] + candidate_covariance) / float(
        gap * gap
    )
    displacement_covariance = _spd_covariance(
        displacement_covariance, name="displacement_covariance"
    )
    cross_covariance = -fit.slope_weights[-1] * point_covariances[-1] / float(gap)
    q_v = _mahalanobis(fit.velocity, velocity_covariance)

    velocity_gradient = _angle_gradient(fit.velocity)
    displacement_gradient = _angle_gradient(displacement_rate)
    if velocity_gradient is None or displacement_gradient is None:
        return DirectionObservation(
            velocity=fit.velocity,
            displacement_rate=displacement_rate,
            velocity_covariance=velocity_covariance,
            displacement_covariance=displacement_covariance,
            velocity_displacement_cross_covariance=cross_covariance,
            q_v=q_v,
            delta_angle=None,
            angular_variance=math.inf,
            kappa=0.0,
            raw_direction_cost=1.0,
            weighted_direction_cost=0.0,
        )

    angular_variance = float(
        velocity_gradient @ velocity_covariance @ velocity_gradient
        + displacement_gradient @ displacement_covariance @ displacement_gradient
        - 2.0 * velocity_gradient @ cross_covariance @ displacement_gradient
    )
    if not math.isfinite(angular_variance) or angular_variance <= 0:
        raise ObservabilityError(
            "propagated angular variance must be finite and positive"
        )
    kappa = 1.0 / angular_variance
    delta_angle = math.atan2(
        math.sin(
            math.atan2(displacement_rate[1], displacement_rate[0])
            - math.atan2(fit.velocity[1], fit.velocity[0])
        ),
        math.cos(
            math.atan2(displacement_rate[1], displacement_rate[0])
            - math.atan2(fit.velocity[1], fit.velocity[0])
        ),
    )
    raw_cost = 1.0 - math.cos(delta_angle)
    weighted_cost = uniform_relative_von_mises_nll(delta_angle, kappa)
    return DirectionObservation(
        velocity=fit.velocity,
        displacement_rate=displacement_rate,
        velocity_covariance=velocity_covariance,
        displacement_covariance=displacement_covariance,
        velocity_displacement_cross_covariance=cross_covariance,
        q_v=q_v,
        delta_angle=delta_angle,
        angular_variance=angular_variance,
        kappa=kappa,
        raw_direction_cost=raw_cost,
        weighted_direction_cost=weighted_cost,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ObservabilityError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ObservabilityError(f"non-finite JSON token {token!r} in {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_non_finite,
        )
    except ObservabilityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservabilityError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ObservabilityError(f"JSON root must be an object: {path}")

    def require_finite_json(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ObservabilityError(f"non-finite JSON number in {path}")
        if isinstance(item, Mapping):
            for child in item.values():
                require_finite_json(child)
        elif isinstance(item, list):
            for child in item:
                require_finite_json(child)

    require_finite_json(value)
    return value


def _repo_path(relative_path: str, *, name: str) -> Path:
    path = (ROOT / relative_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ObservabilityError(f"{name} escapes the repository root") from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ObservabilityError(f"cannot read frozen source {path}: {exc}") from exc
    return digest.hexdigest()


def _schema_validate_study_spec(spec: Mapping[str, Any]) -> None:
    """Reject any study spec that drifts from the frozen degrees of freedom.

    Identity is pinned twice — once here and once in ``EXPECTED_SOURCE_ROLES`` —
    but the boxes, bins, estimator rules and terminal order live only in the
    record. Without this the record is prose in JSON clothing: deleting
    ``ranking_box`` outright still reported ``valid``.
    """

    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ObservabilityError("jsonschema dependency unavailable") from exc

    schema = _load_json(STUDY_SCHEMA_PATH)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(spec),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
    except jsonschema.SchemaError as exc:  # pragma: no cover - static schema
        raise ObservabilityError(f"invalid study schema: {exc.message}") from exc
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ObservabilityError(f"study spec rejected at {location}: {error.message}")


def verify_study_spec(path: Path) -> dict[str, object]:
    """Check only frozen identities and declaration shape; never load outcome rows."""

    spec = _load_json(path)
    _schema_validate_study_spec(spec)
    if spec.get("schema") != STUDY_SCHEMA or spec.get("study_id") != STUDY_ID:
        raise ObservabilityError("unsupported study schema or study_id")
    if spec.get("status") != "preseal_implementation":
        raise ObservabilityError(
            "check-only accepts only preseal_implementation status"
        )
    if spec.get("execution_authorized") is not False:
        raise ObservabilityError(
            "formal execution must remain unauthorized before seal"
        )
    source_files = spec.get("source_files")
    if not isinstance(source_files, Sequence) or isinstance(source_files, (str, bytes)):
        raise ObservabilityError("source_files must be an array")

    verified: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for item in source_files:
        if not isinstance(item, Mapping):
            raise ObservabilityError("every source_files item must be an object")
        relative_path = item.get("path")
        role = item.get("role")
        expected_hash = item.get("sha256")
        expected_bytes = item.get("bytes")
        if not isinstance(relative_path, str) or relative_path in seen_paths:
            raise ObservabilityError("source path must be a unique string")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ObservabilityError(
                f"invalid source SHA256 declaration for {relative_path}"
            )
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ObservabilityError(f"invalid source byte count for {relative_path}")
        if EXPECTED_SOURCE_ROLES.get(relative_path) != role:
            raise ObservabilityError(f"unexpected source path or role: {relative_path}")
        source_path = _repo_path(relative_path, name="source path")
        actual_hash = _sha256(source_path)
        actual_bytes = source_path.stat().st_size
        if actual_hash != expected_hash or actual_bytes != expected_bytes:
            raise ObservabilityError(
                f"frozen source identity mismatch: {relative_path}"
            )
        seen_paths.add(relative_path)
        verified.append(
            {"path": relative_path, "sha256": actual_hash, "bytes": actual_bytes}
        )
    if seen_paths != set(EXPECTED_SOURCE_ROLES):
        missing = sorted(set(EXPECTED_SOURCE_ROLES) - seen_paths)
        unexpected = sorted(seen_paths - set(EXPECTED_SOURCE_ROLES))
        raise ObservabilityError(
            f"source identity set mismatch (missing={missing}, unexpected={unexpected})"
        )

    score_path_value = spec.get("score_declaration")
    if not isinstance(score_path_value, str):
        raise ObservabilityError("score_declaration must be a repo-relative path")
    score_path = _repo_path(score_path_value, name="score declaration path")
    if score_path != DEFAULT_SCORE_DECLARATION.resolve():
        raise ObservabilityError("unexpected score declaration identity")
    score_report = validate_declaration_file(score_path)
    return {
        "schema": STUDY_SCHEMA,
        "study_id": STUDY_ID,
        "valid": True,
        "source_files_verified": verified,
        "score_declaration": score_report,
        "formal_rows_read": 0,
        "execution_authorized": False,
        "next_action": "owner_seal_review",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="verify identities only"
    )
    parser.add_argument("--study-spec", type=Path, default=DEFAULT_STUDY_SPEC)
    args = parser.parse_args(argv)
    if not args.check_only:
        parser.error(
            "formal execution is locked; only --check-only is available pre-seal"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_study_spec(args.study_spec)
    except ObservabilityError as exc:
        raise SystemExit(f"preflight rejected: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
