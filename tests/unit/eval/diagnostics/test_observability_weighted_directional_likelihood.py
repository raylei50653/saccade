"""Tests for observability-weighted directional likelihood math and preseal lock."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOL_PATH = (
    PROJECT_ROOT / "scripts/tools/observability_weighted_directional_likelihood.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("owdl", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["owdl"] = module
    spec.loader.exec_module(module)
    return module


owdl = _load_tool()


def _linear_window(speed: float) -> np.ndarray:
    frames = np.arange(4, dtype=np.float64)
    return np.column_stack((speed * frames, np.zeros(4, dtype=np.float64)))


def test_ols_uses_actual_frame_spacing() -> None:
    frames = np.array([2.0, 4.0, 7.0, 11.0])
    points = np.column_stack((3.0 + 1.5 * frames, -2.0 + 0.25 * frames))

    fit = owdl.fit_ols_motion(points, frames)

    assert fit.velocity == pytest.approx([1.5, 0.25])
    assert fit.residuals == pytest.approx(np.zeros((4, 2)), abs=1e-12)
    assert sum(fit.slope_weights) == pytest.approx(0.0)


def test_noise_covariance_requires_positive_definite_residual_support() -> None:
    frames = np.arange(4, dtype=np.float64)
    x_only_noise = np.array([[0.1, 0.0], [-0.1, 0.0], [0.1, 0.0], [-0.1, 0.0]])

    with pytest.raises(owdl.ObservabilityError, match="positive definite"):
        owdl.estimate_normalized_effective_covariance(
            [(_linear_window(1.0) + x_only_noise, frames, np.ones(4))]
        )


def test_estimated_noise_covariance_is_symmetric_positive_definite() -> None:
    frames = np.arange(4, dtype=np.float64)
    first_noise = np.array([[0.08, 0.02], [-0.12, -0.04], [0.04, 0.09], [0.0, -0.07]])
    second_noise = np.array(
        [[-0.03, 0.11], [0.09, -0.02], [-0.08, -0.06], [0.02, 0.04]]
    )

    covariance = owdl.estimate_normalized_effective_covariance(
        [
            (_linear_window(0.5) + first_noise, frames, np.ones(4)),
            (_linear_window(1.0) + second_noise, frames, np.ones(4)),
        ]
    )

    assert covariance == pytest.approx(covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_zero_velocity_degenerates_to_uniform_without_a_speed_threshold() -> None:
    observation = owdl.observe_direction(
        lost_points=np.zeros((4, 2)),
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=np.array([1.0, 0.0]),
        candidate_first_height=1.0,
        gap=1,
        normalized_effective_covariance=np.eye(2) * 0.01,
    )

    assert observation.q_v == pytest.approx(0.0)
    assert observation.delta_angle is None
    assert observation.undefined_angle_reason == "exact_zero_velocity"
    assert math.isinf(observation.angular_variance)
    assert observation.kappa == 0.0
    assert observation.raw_direction_cost == 1.0
    assert observation.weighted_direction_cost == 0.0


@pytest.mark.parametrize(
    ("lost_points", "candidate_first_point", "reason"),
    [
        (_linear_window(1.0), np.array([3.0, 0.0]), "exact_zero_displacement"),
        (
            np.zeros((4, 2)),
            np.array([0.0, 0.0]),
            "exact_zero_velocity_and_displacement",
        ),
    ],
)
def test_each_undefined_angle_has_one_frozen_exact_zero_reason(
    lost_points: np.ndarray,
    candidate_first_point: np.ndarray,
    reason: str,
) -> None:
    observation = owdl.observe_direction(
        lost_points=lost_points,
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=candidate_first_point,
        candidate_first_height=1.0,
        gap=1,
        normalized_effective_covariance=np.eye(2) * 0.01,
    )

    assert observation.delta_angle is None
    assert observation.undefined_angle_reason == reason
    assert observation.raw_direction_cost == 1.0
    assert observation.weighted_direction_cost == 0.0


def test_the_machine_record_pins_the_math_cores_undefined_angle_reasons() -> None:
    phenomenon = _frozen_study_spec()["phenomenon_box"]

    assert phenomenon["undefined_angle_reasons"] == list(owdl.UNDEFINED_ANGLE_REASONS)


def test_higher_observability_index_increases_concentration() -> None:
    common = {
        "lost_frames": np.arange(4),
        "lost_heights": np.ones(4),
        "candidate_first_height": 1.0,
        "gap": 2,
        "normalized_effective_covariance": np.eye(2) * 0.01,
    }
    slow = owdl.observe_direction(
        lost_points=_linear_window(0.1),
        candidate_first_point=np.array([0.5, 0.0]),
        **common,
    )
    fast = owdl.observe_direction(
        lost_points=_linear_window(1.0),
        candidate_first_point=np.array([5.0, 0.0]),
        **common,
    )

    assert fast.q_v > slow.q_v
    assert fast.angular_variance < slow.angular_variance
    assert fast.kappa > slow.kappa
    assert fast.undefined_angle_reason is None
    assert slow.undefined_angle_reason is None


def test_near_zero_velocity_is_continuous_not_thresholded() -> None:
    tiny = owdl.observe_direction(
        lost_points=_linear_window(1e-9),
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=np.array([1.0, 0.0]),
        candidate_first_height=1.0,
        gap=1,
        normalized_effective_covariance=np.eye(2) * 0.01,
    )

    assert tiny.delta_angle == pytest.approx(0.0)
    assert tiny.q_v > 0.0
    assert tiny.kappa < 1e-12
    assert tiny.weighted_direction_cost == pytest.approx(0.0, abs=1e-12)


def test_concentration_decays_without_a_step_as_velocity_shrinks() -> None:
    """No threshold: kappa and the cost fall monotonically toward the uniform limit."""

    common = {
        "lost_frames": np.arange(4),
        "lost_heights": np.ones(4),
        "candidate_first_point": np.array([3.0, 3.0]),
        "candidate_first_height": 1.0,
        "gap": 2,
        "normalized_effective_covariance": np.eye(2) * 0.01,
    }
    observations = [
        owdl.observe_direction(lost_points=_linear_window(speed), **common)
        for speed in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
    ]
    kappas = [observation.kappa for observation in observations]
    costs = [abs(observation.weighted_direction_cost) for observation in observations]

    assert kappas == sorted(kappas, reverse=True)
    assert costs == sorted(costs, reverse=True)
    zero = owdl.observe_direction(lost_points=np.zeros((4, 2)), **common)
    assert zero.kappa == 0.0
    assert kappas[-1] >= zero.kappa


def test_kappa_matches_the_wrapped_normal_resultant() -> None:
    """The frozen mapping is resultant matching, not the small-angle shortcut."""

    for variance in (0.01, 0.3, 1.0, 3.0, 12.0):
        kappa = owdl.resultant_matched_concentration(variance)
        assert owdl.von_mises_mean_resultant(kappa) == pytest.approx(
            math.exp(-variance / 2.0), rel=1e-12
        )


def test_kappa_recovers_one_over_variance_in_the_small_angle_limit() -> None:
    for variance in (1e-6, 1e-5, 1e-4):
        kappa = owdl.resultant_matched_concentration(variance)
        assert kappa * variance == pytest.approx(1.0, rel=1e-3)


def test_kappa_reaches_uniform_for_a_large_or_infinite_angular_variance() -> None:
    assert owdl.resultant_matched_concentration(math.inf) == 0.0
    assert owdl.resultant_matched_concentration(5.0e3) == 0.0
    assert owdl.resultant_matched_concentration(30.0) < 1e-6


def test_resultant_matched_concentration_is_strictly_decreasing() -> None:
    variances = [0.05, 0.2, 0.8, 2.0, 6.0, 15.0]
    kappas = [owdl.resultant_matched_concentration(v) for v in variances]

    assert kappas == sorted(kappas, reverse=True)
    assert len(set(kappas)) == len(kappas)


def test_a_non_positive_angular_variance_fails_closed() -> None:
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(owdl.ObservabilityError, match="angular variance"):
            owdl.resultant_matched_concentration(bad)


def test_shared_endpoint_cross_covariance_is_propagated() -> None:
    observation = owdl.observe_direction(
        lost_points=_linear_window(1.0),
        lost_frames=np.arange(4),
        lost_heights=np.ones(4),
        candidate_first_point=np.array([5.0, 1.0]),
        candidate_first_height=1.0,
        gap=2,
        normalized_effective_covariance=np.eye(2) * 0.04,
    )
    fit = owdl.fit_ols_motion(_linear_window(1.0), np.arange(4))
    expected = -fit.slope_weights[-1] * np.eye(2) * 0.04 / 2.0

    assert observation.velocity_displacement_cross_covariance == pytest.approx(expected)


def test_von_mises_cost_keeps_normalizer_and_uniform_limit() -> None:
    assert owdl.uniform_relative_von_mises_nll(1.2, 0.0) == 0.0
    aligned_low = owdl.uniform_relative_von_mises_nll(0.0, 0.5)
    aligned_high = owdl.uniform_relative_von_mises_nll(0.0, 5.0)
    opposed_high = owdl.uniform_relative_von_mises_nll(math.pi, 5.0)

    assert aligned_high < aligned_low < 0.0
    assert opposed_high > 0.0


def test_cli_rejects_formal_execution_before_seal() -> None:
    with pytest.raises(SystemExit):
        owdl.parse_args([])


def test_a_numpy_integer_gap_is_accepted_and_a_bool_is_not() -> None:
    """The formal runner reads gaps from a table, so `np.int64` must not fail closed."""

    common = {
        "lost_points": _linear_window(1.0),
        "lost_frames": np.arange(4),
        "lost_heights": np.ones(4),
        "candidate_first_point": np.array([5.0, 1.0]),
        "candidate_first_height": 1.0,
        "normalized_effective_covariance": np.eye(2) * 0.04,
    }

    assert owdl.observe_direction(gap=np.int64(2), **common).kappa == pytest.approx(
        owdl.observe_direction(gap=2, **common).kappa
    )
    with pytest.raises(owdl.ObservabilityError, match="positive integer"):
        owdl.observe_direction(gap=True, **common)
    with pytest.raises(owdl.ObservabilityError, match="positive integer"):
        owdl.observe_direction(gap=np.int64(0), **common)


def test_the_von_mises_cost_is_stable_at_extreme_concentration() -> None:
    """Undoing the i0e scaling would difference two large numbers and lose the tail.

    At `delta == 0` the exact cost is `log(I0(kappa)) - kappa`, which is exactly
    `log(i0e(kappa))` — no high-precision library needed to know the answer. The
    naive form restores `+ kappa` and subtracts `kappa * cos(0)`, and what comes
    back is the ulp of kappa, not the value.
    """

    for kappa, tolerated in ((1.0e12, 1.0e-8), (1.0e15, 1.0e-8), (9.0e15, 1.0e-8)):
        exact = math.log(float(owdl.i0e(kappa)))
        naive = (math.log(float(owdl.i0e(kappa))) + kappa) - kappa * math.cos(0.0)

        assert owdl.uniform_relative_von_mises_nll(0.0, kappa) == exact
        assert abs(naive - exact) > 1.0e-5
        assert abs(owdl.uniform_relative_von_mises_nll(0.0, kappa) - exact) < tolerated


def test_the_von_mises_cost_matches_arbitrary_precision_where_available() -> None:
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    for kappa in (1.0e12, 1.0e15, 9.0e15):
        for delta in (0.0, 1.0e-8):
            exact = float(
                mp.log(mp.besseli(0, mp.mpf(kappa)))
                - mp.mpf(kappa) * mp.cos(mp.mpf(delta))
            )
            assert owdl.uniform_relative_von_mises_nll(delta, kappa) == pytest.approx(
                exact, abs=1e-9
            )


def test_the_raw_baseline_does_not_collapse_well_aligned_candidates_into_a_tie() -> (
    None
):
    """`1 - cos(1e-8)` rounds to exactly 0.0; a tie is worth half a pairwise win."""

    assert 1.0 - math.cos(1e-8) == 0.0
    assert owdl.half_angle_cosine_deficit(1e-8) == pytest.approx(5.0e-17, rel=1e-9)
    deficits = [owdl.half_angle_cosine_deficit(d) for d in (1e-7, 1e-8, 1e-9)]
    assert deficits == sorted(deficits, reverse=True)
    assert len(set(deficits)) == 3


def test_the_stable_cost_equals_the_declared_algebraic_form() -> None:
    for kappa in (0.5, 2.0, 40.0):
        for delta in (0.0, 0.3, math.pi / 2, math.pi):
            declared = (
                math.log(float(owdl.i0e(kappa))) + kappa - kappa * math.cos(delta)
            )
            assert owdl.uniform_relative_von_mises_nll(delta, kappa) == pytest.approx(
                declared, rel=1e-12, abs=1e-12
            )


def test_the_declared_numerical_domain_floor_is_where_binary64_actually_fails() -> None:
    """The floor is a representability failure, not a model threshold."""

    floor = 2.0**-53
    assert math.exp(-floor / 2.0) == 1.0
    just_above = math.nextafter(floor, math.inf)
    assert math.exp(-just_above / 2.0) < 1.0

    with pytest.raises(owdl.ObservabilityError, match="unit resultant"):
        owdl.resultant_matched_concentration(floor)
    kappa = owdl.resultant_matched_concentration(just_above)
    assert 1.0e15 < kappa < 1.0 / just_above


def test_the_declared_bracket_limit_cannot_bind_above_that_floor() -> None:
    """Declared for completeness: the search terminates by 2**52, well under 2**60."""

    largest_target = math.nextafter(1.0, 0.0)
    high, doublings = 1.0, 0
    while owdl.von_mises_mean_resultant(high) < largest_target:
        high *= 2.0
        doublings += 1

    assert high == 2.0**52
    assert doublings == 52
    assert high < owdl._CONCENTRATION_BRACKET_LIMIT


def test_the_preseal_authority_binding_names_three_identities_and_fills_none() -> None:
    """The sealed head carries no runner, so seal head and runner head differ."""

    authority = _frozen_study_spec()["authority_binding"]

    assert authority["declaration_seal_head"] is None
    assert authority["runner_review_head"] is None
    assert authority["runner_sha256"] is None
    assert authority["sealed_spec_slots_remain_null"] is True
    assert authority["values_carried_in"] == (
        "external_authority_receipt_and_evidence_packet"
    )
    assert authority["requires_sealed_declaration_bytes_unchanged"] is True
    assert authority["formal_execution_binds"] == [
        "declaration_seal_head",
        "runner_review_head",
        "runner_sha256",
        "nine_frozen_source_sha256",
    ]


@pytest.mark.parametrize(
    "identity", ["declaration_seal_head", "runner_review_head", "runner_sha256"]
)
def test_a_spec_that_writes_an_authority_value_back_into_itself_is_rejected(
    tmp_path: Path, identity: str
) -> None:
    """Back-filling would change the bytes that the value was supposed to name."""
    spec = _frozen_study_spec()
    spec["authority_binding"][identity] = "0" * 40
    spec_path = tmp_path / "study.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(owdl.ObservabilityError, match="study spec rejected"):
        owdl.verify_study_spec(spec_path)


def test_both_calibration_exposures_are_bound_to_the_score_record() -> None:
    study, score = _frozen_study_spec(), _frozen_score_record()
    study["validity"]["minimum_gt_pairs_high_q"] = 5

    with pytest.raises(owdl.ObservabilityError, match="high-q calibration exposure"):
        owdl._validate_study_score_binding(study, score)


def _frozen_score_record() -> dict:
    return json.loads(owdl.DEFAULT_SCORE_DECLARATION.read_text(encoding="utf-8"))


def test_the_two_frozen_records_bind_to_each_other() -> None:
    owdl._validate_study_score_binding(_frozen_study_spec(), _frozen_score_record())


@pytest.mark.parametrize(
    ("record", "pointer", "replacement"),
    [
        ("score", ("claim", "minimum_effect", "value"), 0.01),
        ("score", ("claim", "minimum_exposure"), 50),
        ("score", ("claim", "uncertainty_method_id"), "bootstrap_only_v1"),
        ("score", ("claim", "short_gap_retention_rule_id"), "short_gap_any_row_v1"),
        ("score", ("claim", "folds_id"), "leave_one_out_v1"),
        (
            "score",
            ("spaces", "calibration_space_id"),
            "m_b1_gt_positive_angular_residual_space_v1",
        ),
        (
            "score",
            ("calibration_claim", "estimator_id"),
            "fold_heldout_resultant_length_by_observability_index_bin_v1",
        ),
        (
            "score",
            ("policy", "candidate_universe", "candidate_key_fields"),
            ["seq", "cand_id"],
        ),
        ("study", ("ranking_box", "cluster_bootstrap", "gates_terminal"), True),
        ("study", ("ranking_box", "delete_one_sequence", "required_passes"), 5),
        ("study", ("ranking_box", "minimum_delta"), 0.01),
    ],
)
def test_a_record_that_drifts_from_its_partner_is_rejected(
    record: str, pointer: tuple[str, ...], replacement: object
) -> None:
    """Each record can be valid on its own while promising the other something else."""

    study, score = _frozen_study_spec(), _frozen_score_record()
    target = study if record == "study" else score
    for part in pointer[:-1]:
        target = target[part]
    target[pointer[-1]] = replacement

    with pytest.raises(owdl.ObservabilityError, match="binding disagrees"):
        owdl._validate_study_score_binding(study, score)


def test_a_terminal_that_loses_its_outcome_semantics_is_rejected() -> None:
    study, score = _frozen_study_spec(), _frozen_score_record()
    score["terminals"][3]["state_transition"]["kind"] = "none"

    with pytest.raises(owdl.ObservabilityError, match="transitions"):
        owdl._validate_study_score_binding(study, score)


def _frozen_study_spec() -> dict:
    return json.loads(owdl.DEFAULT_STUDY_SPEC.read_text(encoding="utf-8"))


def test_the_frozen_study_spec_matches_its_schema() -> None:
    """The record that carries the study's frozen degrees of freedom is checked.

    This runs without the nine frozen source files, so unlike `--check-only` it
    is reachable on a machine that holds only the repository.
    """

    owdl._schema_validate_study_spec(_frozen_study_spec())


@pytest.mark.parametrize(
    "frozen_key",
    [
        "candidate_universe",
        "estimator",
        "phenomenon_box",
        "ranking_box",
        "sequences",
        "terminal_order",
        "validity",
    ],
)
def test_dropping_a_frozen_box_from_the_study_spec_is_rejected(
    tmp_path: Path, frozen_key: str
) -> None:
    """Deleting any frozen box must fail the preflight, not pass it silently."""

    spec = _frozen_study_spec()
    del spec[frozen_key]
    spec_path = tmp_path / "study.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(owdl.ObservabilityError, match="study spec rejected"):
        owdl.verify_study_spec(spec_path)


@pytest.mark.parametrize(
    ("pointer", "replacement"),
    [
        (("phenomenon_box", "minimum_high_minus_low"), 0.05),
        (
            ("phenomenon_box", "undefined_angle_rule"),
            "count_as_zero_resultant",
        ),
        (
            ("phenomenon_box", "exposure_denominator"),
            "all_gt_pairs",
        ),
        (("ranking_box", "minimum_delta"), 0.0),
        (("ranking_box", "minimum_positive_folds"), 1),
        (("validity", "minimum_rankable_events"), 1),
        (("estimator", "posthoc_covariance_regularization"), "allowed"),
        (("estimator", "history_window"), 8),
        (("estimator", "exact_zero_vector_rule"), "drop_candidate"),
        (("ranking_box", "short_gap", "positive_row_rule"), "gt_match == 1"),
        (
            ("ranking_box", "short_gap", "negative_row_rule"),
            "gt_match == 0 and gap <= 10",
        ),
        (("ranking_box", "short_gap", "minimum_events"), 1),
        (("ranking_box", "cluster_bootstrap", "gates_terminal"), True),
        (("ranking_box", "delete_one_sequence", "required_passes"), 5),
        (("source_relation_contract", "executed_pre_seal"), True),
        (("ranking_box", "delete_one_sequence", "scope"), "full_corpus_deletion"),
        (("ranking_box", "delete_one_sequence", "refit_effective_covariance"), True),
        (("ranking_box", "delete_one_sequence", "reuse_primary_fold_scores"), False),
        (("validity", "require_defined_per_fold_delta"), False),
        (("validity", "minimum_gt_pairs_high_q"), 5),
        (("estimator", "numeric_domain", "arithmetic"), "binary32"),
        (("estimator", "numeric_domain", "concentration_bisections"), 20),
        (("estimator", "numeric_domain", "unit_resultant_rounding_rule"), "clamp"),
        (("authority_binding", "requires_sealed_declaration_bytes_unchanged"), False),
        (("authority_binding", "sealed_spec_slots_remain_null"), False),
        (("authority_binding", "values_carried_in"), "sealed_study_spec"),
        (("candidate_universe", "tie_contribution"), 1.0),
        (("execution_authorized",), True),
    ],
)
def test_loosening_a_frozen_value_in_the_study_spec_is_rejected(
    tmp_path: Path, pointer: tuple[str, ...], replacement: object
) -> None:
    """A box may move only by moving the schema with it, in the same reviewed diff."""

    spec = _frozen_study_spec()
    target = spec
    for part in pointer[:-1]:
        target = target[part]
    target[pointer[-1]] = replacement
    spec_path = tmp_path / "study.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(owdl.ObservabilityError, match="study spec rejected"):
        owdl.verify_study_spec(spec_path)


def test_an_unexpected_source_role_is_rejected_before_any_file_is_read(
    tmp_path: Path,
) -> None:
    spec = _frozen_study_spec()
    spec["source_files"][0]["role"] = "trajectory_mot17_02_sdp"
    spec_path = tmp_path / "study.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(owdl.ObservabilityError, match="unexpected source path or role"):
        owdl.verify_study_spec(spec_path)


def test_an_empty_source_identity_set_is_rejected(tmp_path: Path) -> None:
    spec = _frozen_study_spec()
    spec["source_files"] = []
    spec_path = tmp_path / "study.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(owdl.ObservabilityError, match="study spec rejected"):
        owdl.verify_study_spec(spec_path)


def test_the_frozen_score_declaration_validates_against_the_sealed_contract() -> None:
    """The real SR2 record, not the fixture, is the one the study is bound to."""

    report = owdl.validate_declaration_file(owdl.DEFAULT_SCORE_DECLARATION)

    assert report["valid"] is True
    assert report["target_rung"] == "SR2"
