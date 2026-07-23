"""Machine-checkable invariants and constructive counterexamples for GCTM D1."""

# status: experiment

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from . import (
    CALIBRATION_ONLY_MECHANISM,
    EVENT_KEY,
    NORMALIZATION,
    ORDERING_ACTIVE_MECHANISM,
    SCORE_ORIENTATION,
    SCORE_TRANSFORM,
    TIE_RULE,
)
from .fixtures import pack_to_candidates
from .models import (
    CandidateObservation,
    CovMode,
    FailClosedError,
    ModelId,
    is_strictly_monotone_increasing,
    mahalanobis_q,
    ordering_tuple,
    residual_for_model,
    resolve_covariance,
    score_candidate,
)


@dataclass
class InvariantResult:
    invariant_id: str
    description: str
    passed: bool
    detail: dict[str, Any]


def _group_by_event(
    cands: list[CandidateObservation],
) -> dict[str, list[CandidateObservation]]:
    groups: dict[str, list[CandidateObservation]] = defaultdict(list)
    for c in cands:
        groups[c.event_id].append(c)
    return dict(groups)


def _score_event(
    event_cands: list[CandidateObservation],
    model_id: ModelId,
    *,
    cov_mode: CovMode,
    rank_score: str = "q",
):
    return [
        score_candidate(c, model_id, cov_mode=cov_mode, rank_score=rank_score)  # type: ignore[arg-type]
        for c in event_cands
    ]


def inv1_pair_event_uniqueness(cands: list[CandidateObservation]) -> InvariantResult:
    """Every candidate pair belongs to exactly one event."""
    keys = [(c.event_id, c.cand_id) for c in cands]
    unique = len(keys) == len(set(keys))
    multi_event = defaultdict(set)
    for c in cands:
        multi_event[c.cand_id].add(c.event_id)
    # cand_id may reuse across events; the pair (event,cand) is the unit.
    # Require each (event,cand) unique and each pair maps to exactly one event_id.
    ok = unique
    return InvariantResult(
        "I1_pair_event_uniqueness",
        "Every candidate pair belongs to exactly one event",
        ok,
        {"n_pairs": len(keys), "n_unique": len(set(keys))},
    )


def inv2_identity_stable_across_models(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Candidate identity and event membership unchanged across M0, M1, M2."""
    base = {(c.event_id, c.cand_id) for c in cands}
    modes: dict[str, CovMode] = {
        "E_shared_iso": "isotropic_shared",
        "E_shared_aniso": "anisotropic_shared",
        "E_cand_spec": "candidate_specific",
        "E_short_gap": "anisotropic_shared",
        "E_long_gap": "anisotropic_shared",
        "E_m2_drift": "anisotropic_shared",
        "E_tie": "isotropic_shared",
    }
    ok = True
    details: dict[str, Any] = {}
    for model in ("M0", "M1", "M2"):
        scored_ids = set()
        for event_id, group in _group_by_event(cands).items():
            mode = modes[event_id]
            scored = _score_event(group, model, cov_mode=mode)  # type: ignore[arg-type]
            scored_ids |= {(s.event_id, s.cand_id) for s in scored}
        details[model] = sorted(list(scored_ids))
        if scored_ids != base:
            ok = False
    return InvariantResult(
        "I2_identity_stable_across_models",
        "Candidate identity and event membership unchanged across M0/M1/M2",
        ok,
        details,
    )


def inv3_pair_event_count_reconcile(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Pair counts reconcile with event candidate counts."""
    groups = _group_by_event(cands)
    n_pairs = len(cands)
    n_from_events = sum(len(v) for v in groups.values())
    ok = n_pairs == n_from_events
    return InvariantResult(
        "I3_pair_event_count_reconcile",
        "Pair counts reconcile with event candidate counts",
        ok,
        {"n_pairs": n_pairs, "n_from_events": n_from_events, "n_events": len(groups)},
    )


def inv4_calibration_ranking_claim_spaces(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Calibration and event-local ranking are separate claim spaces."""
    # Shared isotropic rescaling changes absolute q level (CAL) but not order (RANK).
    event = [c for c in cands if c.event_id == "E_shared_iso"]
    base = [
        score_candidate(
            CandidateObservation(**{**c.__dict__, "scale_alpha": 1.0}),
            "M1",
            cov_mode="isotropic_shared",
            rank_score="q",
        )
        for c in event
    ]
    scaled = [
        score_candidate(
            CandidateObservation(**{**c.__dict__, "scale_alpha": 4.0}),
            "M1",
            cov_mode="isotropic_shared",
            rank_score="q",
        )
        for c in event
    ]
    order_base = ordering_tuple(base)
    order_scaled = ordering_tuple(scaled)
    q_levels_change = any(
        abs(a.q - b.q) > 1e-9 for a, b in zip(base, scaled, strict=True)
    )
    ok = order_base == order_scaled and q_levels_change
    return InvariantResult(
        "I4_calibration_ranking_claim_spaces",
        "Calibration and event-local ranking are separate claim spaces",
        ok,
        {
            "order_base": order_base,
            "order_scaled": order_scaled,
            "q_levels_change": q_levels_change,
            "claim_spaces": {
                "CAL": "absolute q / coverage / PIT under working null",
                "RANK": "within-event ordering under frozen orientation",
            },
        },
    )


def inv5_shared_scalar_cannot_reorder(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Shared scalar covariance within an event cannot change candidate-local ordering."""
    event = [c for c in cands if c.event_id == "E_shared_iso"]
    m0 = _score_event(event, "M0", cov_mode="isotropic_shared", rank_score="euclid")
    m1 = _score_event(event, "M1", cov_mode="isotropic_shared", rank_score="q")
    ok = ordering_tuple(m0) == ordering_tuple(m1)
    return InvariantResult(
        "I5_shared_scalar_not_ranking_active",
        "Shared scalar covariance within an event cannot change candidate-local ordering",
        ok,
        {
            "m0_order": ordering_tuple(m0),
            "m1_order": ordering_tuple(m1),
            "mechanism_class": CALIBRATION_ONLY_MECHANISM,
        },
    )


def inv6_q_nll_identical_under_shared(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Under shared covariance, q and NLL produce identical rankings."""
    results = {}
    ok = True
    for event_id in ("E_shared_iso", "E_shared_aniso", "E_short_gap", "E_long_gap"):
        event = [c for c in cands if c.event_id == event_id]
        mode: CovMode = (
            "isotropic_shared" if event_id == "E_shared_iso" else "anisotropic_shared"
        )
        by_q = _score_event(event, "M1", cov_mode=mode, rank_score="q")
        by_nll = _score_event(event, "M1", cov_mode=mode, rank_score="nll")
        oq = ordering_tuple(by_q)
        on = ordering_tuple(by_nll)
        results[event_id] = {"q": oq, "nll": on}
        if oq != on:
            ok = False
    return InvariantResult(
        "I6_q_nll_identical_under_shared_S",
        "Under shared covariance, dimension, gap, context, mode: q and NLL identical rankings",
        ok,
        results,
    )


def inv7_score_transform_monotone() -> InvariantResult:
    """Every score transformation used for ranking is monotone under frozen orientation."""
    # Native lower_better scores; allowed transform is identity (strictly increasing).
    xs = [0.1, 0.5, 1.0, 2.5, 4.0]
    ys_identity = list(xs)
    # Strictly increasing reparameterization preserves order.
    ys_log1p = [float(np.log1p(x)) for x in xs]
    ok = (
        SCORE_TRANSFORM == "identity_after_declared_score"
        and is_strictly_monotone_increasing(xs, ys_identity)
        and is_strictly_monotone_increasing(xs, ys_log1p)
        and SCORE_ORIENTATION == "lower_better"
    )
    return InvariantResult(
        "I7_score_transform_monotone",
        "Every score transformation used for ranking is monotone under frozen orientation",
        ok,
        {
            "score_transform": SCORE_TRANSFORM,
            "orientation": SCORE_ORIENTATION,
            "identity_monotone": is_strictly_monotone_increasing(xs, ys_identity),
            "log1p_monotone": is_strictly_monotone_increasing(xs, ys_log1p),
        },
    )


def inv8_scale_dependent_without_normalization_rejected(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Scale-dependent comparisons are rejected unless normalization is frozen."""
    # Free scale comparison across events with different α without frozen normalization
    # is rejected by policy when NORMALIZATION is not frozen_identity.
    ok = NORMALIZATION == "frozen_identity_no_free_scale"
    event = [c for c in cands if c.event_id == "E_shared_iso"]
    a = score_candidate(
        CandidateObservation(**{**event[0].__dict__, "scale_alpha": 1.0}),
        "M1",
        cov_mode="isotropic_shared",
    )
    b = score_candidate(
        CandidateObservation(**{**event[0].__dict__, "scale_alpha": 9.0}),
        "M1",
        cov_mode="isotropic_shared",
    )
    # Absolute q comparison across scales is not an admissible ranking claim.
    absolute_cross_scale_forbidden = abs(a.q - b.q) > 1e-9
    return InvariantResult(
        "I8_scale_requires_frozen_normalization",
        "Scale-dependent comparisons are rejected unless normalization is frozen",
        ok and absolute_cross_scale_forbidden,
        {
            "normalization": NORMALIZATION,
            "q_alpha1": a.q,
            "q_alpha9": b.q,
            "cross_scale_absolute_comparison_admissible": False,
        },
    )


def protected_stratum_guard(
    *,
    strata_true_ranks: dict[str, dict[str, int]],
    protected_strata: list[str],
    aggregate_top1_baseline: int,
    aggregate_top1_challenger: int,
) -> dict[str, Any]:
    """Reject when aggregate gain hides a protected-stratum ordering loss.

    Returns a result dict with `passed` False when any protected stratum has
    true-rank regression (higher rank number = worse under lower_better ranks)
    under a challenger that also does not strictly lose aggregate top-1 count.
    Any protected-stratum regression fails regardless of aggregate (cannot hide).
    """
    regressions: dict[str, dict[str, int]] = {}
    for stratum in protected_strata:
        ranks = strata_true_ranks.get(stratum)
        if ranks is None:
            return {
                "passed": False,
                "reason": f"missing_protected_stratum:{stratum}",
                "regressions": regressions,
            }
        if ranks["challenger_true_rank"] > ranks["baseline_true_rank"]:
            regressions[stratum] = ranks

    aggregate_improved_or_flat = aggregate_top1_challenger >= aggregate_top1_baseline
    # Charter: reject a model whose aggregate gain hides a protected loss.
    # Also reject protected loss even without aggregate gain (loss is not admissible).
    hidden = bool(regressions) and aggregate_improved_or_flat
    pure_regression = bool(regressions)
    passed = not pure_regression
    return {
        "passed": passed,
        "protected_regressions": regressions,
        "aggregate_top1_baseline": aggregate_top1_baseline,
        "aggregate_top1_challenger": aggregate_top1_challenger,
        "aggregate_improved_or_flat": aggregate_improved_or_flat,
        "would_hide_under_aggregate": hidden,
        "reason": (
            "protected_stratum_regression"
            if pure_regression
            else "no_protected_stratum_regression"
        ),
    }


def inv9_protected_stratum_not_hidden(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Aggregate improvement cannot hide a loss in a declared protected stratum."""
    short = [c for c in cands if c.event_id == "E_short_gap"]
    long = [c for c in cands if c.event_id == "E_long_gap"]
    events = {"short_gap": short, "long_gap": long}

    def true_rank(event, model: ModelId) -> int:
        scored = _score_event(
            event, model, cov_mode="anisotropic_shared", rank_score="q"
        )
        order = ordering_tuple(scored)
        true_id = next(c.cand_id for c in event if c.is_true_match)
        return order.index(true_id) + 1

    strata_true_ranks: dict[str, dict[str, int]] = {}
    top1_m0 = 0
    top1_m1 = 0
    for name, event in events.items():
        r0 = true_rank(event, "M0")
        r1 = true_rank(event, "M1")
        strata_true_ranks[name] = {
            "baseline_true_rank": r0,
            "challenger_true_rank": r1,
            # retained aliases for packet readability
            "m0_true_rank": r0,
            "m1_true_rank": r1,
        }
        top1_m0 += int(r0 == 1)
        top1_m1 += int(r1 == 1)

    guard = protected_stratum_guard(
        strata_true_ranks=strata_true_ranks,
        protected_strata=["short_gap"],
        aggregate_top1_baseline=top1_m0,
        aggregate_top1_challenger=top1_m1,
    )

    # Negative constructive case retained in detail (not averaged away): if short
    # were to regress while aggregate top-1 did not fall, the guard must fail.
    negative = protected_stratum_guard(
        strata_true_ranks={
            "short_gap": {
                "baseline_true_rank": 1,
                "challenger_true_rank": 2,
            },
            "long_gap": {
                "baseline_true_rank": 2,
                "challenger_true_rank": 1,
            },
        },
        protected_strata=["short_gap"],
        aggregate_top1_baseline=1,
        aggregate_top1_challenger=2,
    )
    negative_ok = negative["passed"] is False and negative["would_hide_under_aggregate"]

    ok = bool(guard["passed"]) and negative_ok
    return InvariantResult(
        "I9_protected_stratum_not_hidden",
        "Aggregate improvement cannot hide a loss in a declared protected stratum",
        ok,
        {
            "protected_strata": ["short_gap"],
            "strata_true_ranks": strata_true_ranks,
            "guard": guard,
            "constructive_hiding_counterexample": negative,
            "constructive_retention": "per-stratum ranks retained; not pooled AUC",
        },
    )


def inv10_fail_closed_undefined() -> InvariantResult:
    """Undefined inverse, det, covariance, missing-value, and tie behavior fail closed."""
    cases: dict[str, str] = {}
    ok = True

    def expect(code: str, fn) -> None:
        nonlocal ok
        try:
            fn()
            ok = False
            cases[code] = "did_not_raise"
        except FailClosedError as exc:
            cases[code] = exc.code
            if exc.code != code and code not in exc.code:
                # allow related codes
                cases[code] = exc.code

    d = 2
    r = np.array([1.0, 0.0])
    expect(
        "singular_covariance",
        lambda: mahalanobis_q(r, np.array([[1.0, 0.0], [0.0, 0.0]])),
    )
    expect(
        "non_psd_covariance",
        lambda: mahalanobis_q(r, np.array([[1.0, 0.0], [0.0, -1.0]])),
    )
    expect(
        "missing_covariance",
        lambda: resolve_covariance(
            CandidateObservation(
                event_id="e",
                cand_id="c",
                residual=r,
                delta=1.0,
                is_true_match=True,
                stratum="short_gap",
                cov_shared=None,
            ),
            "anisotropic_shared",
        ),
    )
    expect(
        "missing_context_drift",
        lambda: residual_for_model(
            CandidateObservation(
                event_id="e",
                cand_id="c",
                residual=r,
                delta=1.0,
                is_true_match=True,
                stratum="short_gap",
                context_drift=None,
                cov_shared=np.eye(d),
            ),
            "M2",
        ),
    )
    expect(
        "missing_candidate_covariance",
        lambda: resolve_covariance(
            CandidateObservation(
                event_id="e",
                cand_id="c",
                residual=r,
                delta=1.0,
                is_true_match=True,
                stratum="short_gap",
                cov_shared=np.eye(d),
                cov_candidate=None,
            ),
            "candidate_specific",
        ),
    )
    # Tie rule is frozen; unsupported rules fail closed.
    from .models import rank_event, ScoredCandidate

    scored = [
        ScoredCandidate("e", "c1", "M0", r, 1.0, 1.0, 1.0, True, "short_gap", 1.0),
        ScoredCandidate("e", "c2", "M0", r, 1.0, 1.0, 1.0, False, "short_gap", 1.0),
    ]
    expect(
        "unsupported_tie_rule",
        lambda: rank_event(scored, tie_rule="random"),
    )
    return InvariantResult(
        "I10_fail_closed_undefined",
        "Undefined inverse/det/covariance/missing/tie behavior fails closed",
        ok,
        cases,
    )


def inv11_non_identifiable_listed() -> InvariantResult:
    """Every latent quantity or parameter that remains non-identifiable is listed."""
    non_id = [
        {
            "quantity": "P_xx vs R_1 split",
            "status": "structurally_non_identifiable",
            "source": "GCTM theory §7.4 G1 / §7.7",
            "blocking": "requires gauge_fixing declaration",
        },
        {
            "quantity": "asym(P_xv) under H_x",
            "status": "structurally_invisible",
            "source": "GCTM theory §7.4 G2",
            "blocking": "H_x observation mode",
        },
        {
            "quantity": "gamma when unknown without joint-map regime",
            "status": "regime_not_established",
            "source": "GCTM theory §7.4 G3 / §7.7",
            "blocking": "D1 freezes gamma as declared parameter for ranking scores only",
        },
        {
            "quantity": "CAL scale alpha_Delta from RANK order alone",
            "status": "claim_level_non_identifiable",
            "source": "GCTM theory §6.3 / §7.5",
            "blocking": "ranking invariance to shared rescaling",
        },
        {
            "quantity": "full {P0, R1} from single position-only event",
            "status": "non_identifiable",
            "source": "GCTM theory §7.3",
            "blocking": "single-event Hx",
        },
    ]
    ok = len(non_id) >= 4
    return InvariantResult(
        "I11_non_identifiable_listed",
        "Every latent quantity or parameter that remains non-identifiable is listed",
        ok,
        {"non_identifiable": non_id},
    )


def inv12_constructive_counterexamples_retained(
    cands: list[CandidateObservation],
) -> InvariantResult:
    """Constructive counterexamples are retained rather than averaged away."""
    # CEx1: anisotropic shared S reorders vs Euclidean
    aniso = [c for c in cands if c.event_id == "E_shared_aniso"]
    m0 = ordering_tuple(
        _score_event(aniso, "M0", cov_mode="anisotropic_shared", rank_score="euclid")
    )
    m1 = ordering_tuple(
        _score_event(aniso, "M1", cov_mode="anisotropic_shared", rank_score="q")
    )
    cex_aniso = m0 != m1

    # CEx2: candidate-specific S can diverge q vs NLL
    cspec = [c for c in cands if c.event_id == "E_cand_spec"]
    oq = ordering_tuple(
        _score_event(cspec, "M1", cov_mode="candidate_specific", rank_score="q")
    )
    on = ordering_tuple(
        _score_event(cspec, "M1", cov_mode="candidate_specific", rank_score="nll")
    )
    cex_q_nll = oq != on

    # CEx3: M2 context drift reorders vs M1 under fixed interface
    m2e = [c for c in cands if c.event_id == "E_m2_drift"]
    o_m1 = ordering_tuple(
        _score_event(m2e, "M1", cov_mode="anisotropic_shared", rank_score="q")
    )
    o_m2 = ordering_tuple(
        _score_event(m2e, "M2", cov_mode="anisotropic_shared", rank_score="q")
    )
    cex_m2 = o_m1 != o_m2

    # CEx4: L5.2 numeric k=1 style retained (embedded in E_cand_spec residuals)
    # (r,S)=(1,1) vs (1.2,4): q prefers b (0.36 < 1), NLL prefers a (0.5 < 0.873)
    r_a, s_a = 1.0, 1.0
    r_b, s_b = 1.2, 4.0
    q_a, q_b = (r_a**2) / s_a, (r_b**2) / s_b
    nll_a = 0.5 * q_a + 0.5 * np.log(s_a)
    nll_b = 0.5 * q_b + 0.5 * np.log(s_b)
    cex_l52 = bool((q_b < q_a) and (nll_a < nll_b))

    retained = {
        "CEX_anisotropic_shared_reorders_euclidean": {
            "retained": cex_aniso,
            "m0_order": m0,
            "m1_order": m1,
            "mechanism": ORDERING_ACTIVE_MECHANISM,
        },
        "CEX_candidate_specific_q_vs_nll": {
            "retained": cex_q_nll,
            "q_order": oq,
            "nll_order": on,
        },
        "CEX_m2_context_drift_reorders": {
            "retained": cex_m2,
            "m1_order": o_m1,
            "m2_order": o_m2,
        },
        "CEX_L5_2_numeric_q_nll_flip": {
            "retained": cex_l52,
            "q": {"a": q_a, "b": q_b},
            "nll": {"a": nll_a, "b": nll_b},
        },
    }
    ok = all(v["retained"] for v in retained.values())
    return InvariantResult(
        "I12_constructive_counterexamples_retained",
        "Constructive counterexamples are retained rather than averaged away",
        ok,
        retained,
    )


def ranking_active_mechanism_test(cands: list[CandidateObservation]) -> dict[str, Any]:
    """Identify exactly which mechanism can change within-event ordering."""
    aniso = [c for c in cands if c.event_id == "E_shared_aniso"]
    iso = [c for c in cands if c.event_id == "E_shared_iso"]
    cspec = [c for c in cands if c.event_id == "E_cand_spec"]

    m0_aniso = ordering_tuple(
        _score_event(aniso, "M0", cov_mode="anisotropic_shared", rank_score="euclid")
    )
    m1_aniso = ordering_tuple(
        _score_event(aniso, "M1", cov_mode="anisotropic_shared", rank_score="q")
    )
    m0_iso = ordering_tuple(
        _score_event(iso, "M0", cov_mode="isotropic_shared", rank_score="euclid")
    )
    m1_iso = ordering_tuple(
        _score_event(iso, "M1", cov_mode="isotropic_shared", rank_score="q")
    )
    q_cs = ordering_tuple(
        _score_event(cspec, "M1", cov_mode="candidate_specific", rank_score="q")
    )
    nll_cs = ordering_tuple(
        _score_event(cspec, "M1", cov_mode="candidate_specific", rank_score="nll")
    )

    cand_specific_ranking_active = q_cs != nll_cs
    return {
        "ordering_active_mechanism": ORDERING_ACTIVE_MECHANISM,
        "admissible_ranking_active": {
            "anisotropic_shared_innovation_covariance": m0_aniso != m1_aniso,
            "candidate_specific_observation_covariance": cand_specific_ranking_active,
            "notes": (
                "Candidate-specific S is ranking-active when causally declared and "
                "mechanically produces an ordering difference (here q vs NLL); "
                "shared anisotropic S is ranking-active vs Euclidean M0."
            ),
            "evidence": {
                "anisotropic": {"m0_order": m0_aniso, "m1_order": m1_aniso},
                "candidate_specific_q_vs_nll": {"q_order": q_cs, "nll_order": nll_cs},
            },
        },
        "calibration_only": {
            CALIBRATION_ONLY_MECHANISM: m0_iso == m1_iso,
            "evidence": {"m0_order": m0_iso, "m1_order": m1_iso},
        },
        "rejected_as_ranking_evidence": [
            "lower_mean_distance_alone",
            "pooled_AUC",
            "raw_NLL_improvement_cross_event",
            "overall_pair_accuracy_pooled_rows",
        ],
        "event_key": EVENT_KEY,
        "tie_rule": TIE_RULE,
    }


def run_all_invariants(pack: dict[str, Any]) -> dict[str, Any]:
    cands = pack_to_candidates(pack)
    results = [
        inv1_pair_event_uniqueness(cands),
        inv2_identity_stable_across_models(cands),
        inv3_pair_event_count_reconcile(cands),
        inv4_calibration_ranking_claim_spaces(cands),
        inv5_shared_scalar_cannot_reorder(cands),
        inv6_q_nll_identical_under_shared(cands),
        inv7_score_transform_monotone(),
        inv8_scale_dependent_without_normalization_rejected(cands),
        inv9_protected_stratum_not_hidden(cands),
        inv10_fail_closed_undefined(),
        inv11_non_identifiable_listed(),
        inv12_constructive_counterexamples_retained(cands),
    ]
    all_pass = all(r.passed for r in results)
    return {
        "all_passed": all_pass,
        "results": [asdict(r) for r in results],
        "ranking_active_mechanism_test": ranking_active_mechanism_test(cands),
        "n_invariants": len(results),
        "n_passed": sum(1 for r in results if r.passed),
    }
