"""M-B1.5 Stage 2 Q4: signal separability audit on D_online.

Research-only. Does **not** search thresholds, Boolean rules, or change presets.

Primary cohort (decision-relevant, label-resolved only):
  negative class: resolved ∧ baseline_selected ∧ pair_label==negative
  positive protection: resolved ∧ baseline_selected ∧ pair_label==gt_consistent

Contract:
  docs/modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from saccade.perception.eval.d_online_stage2 import (
    CANDIDATE_UNIVERSE_ID,
    SUBSTRATE_ID,
    write_csv,
    write_json,
    write_parquet,
    _sha256_file,
)
from saccade.perception.eval.portable_or_tail import FROZEN_THR_VECTOR, ORDERED_SIGNALS

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

TAXONOMY_VERSION = "stage2_q4_separability_v1"
Q1Q3_STUDY_ID = "m_b1_5_stage2_q1q3_20260710"
EXPECTED_D_ONLINE_N = 244

# Production online bridge gate (predefined; not a searched thr).
# score_m_bridge is the online bridge distance feature (same family as bdist).
ONLINE_BRIDGE_GATE_PX = 0.4

# Terminal gates (descriptive; not thr candidates)
AUC_STRONG = 0.70
AUC_MODERATE = 0.65
CLIFF_MEDIUM = 0.33  # |δ| medium effect (Romano et al. scale)
CLIFF_SMALL = 0.147
MIN_N_PER_CLASS = 5
MIN_SEQS_WITH_BOTH = 3
MIN_LOO_ORIENTED_AUC = 0.65
PURE_TAIL_MIN_N = 3  # pure-neg prefix must have ≥3 for non-empty claim
PURE_TAIL_MIN_SEQS = 2

# Predefined context slices only (no invented cuts for better AUC)
COMPETITOR_SLICES = (
    ("solo_competitor_0", lambda r: int(r.get("competitor_count", 0)) == 0),
    ("has_competitors", lambda r: int(r.get("competitor_count", 0)) >= 1),
)
# score_m_bridge vs online gate midpoint (predefined production context)
BDIST_SLICES = (
    (
        "bdist_le_half_gate",
        lambda r: float(r["score_m_bridge"]) <= ONLINE_BRIDGE_GATE_PX * 0.5,
    ),
    (
        "bdist_gt_half_gate",
        lambda r: float(r["score_m_bridge"]) > ONLINE_BRIDGE_GATE_PX * 0.5,
    ),
)

TERMINAL_VALUES = (
    "single_signal_separability_supported",  # A
    "conditional_separability_supported",  # B
    "separability_weak_or_unstable",  # C
    "insufficient_labeled_decision_mass",  # D
)


class Stage2Q4Error(ValueError):
    """Fail-closed Q4 audit error."""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_d_online_events(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        pq = path / "d_online_events.parquet"
        csv_p = path / "d_online_events.csv"
        if pq.is_file():
            path = pq
        elif csv_p.is_file():
            path = csv_p
        else:
            raise Stage2Q4Error(f"no d_online_events in {path}")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq_mod

        table = pq_mod.read_table(path)
        cols = table.column_names
        col_data = [table.column(i).to_pylist() for i in range(len(cols))]
        n = len(col_data[0]) if col_data else 0
        return [{cols[j]: col_data[j][i] for j in range(len(cols))} for i in range(n)]
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _i(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Cohort lock
# ---------------------------------------------------------------------------


def lock_q4_cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Partition D_online into primary cohort + secondary + excluded."""
    primary: list[dict[str, Any]] = []
    secondary_non_selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for raw in rows:
        r = dict(raw)
        selected = _i(
            r.get("baseline_selected", r.get("baseline_accepted_candidate", 0))
        )
        status = str(r.get("label_status", ""))
        pl = str(r.get("pair_label", ""))
        r["baseline_selected"] = selected
        r["_selected"] = selected
        r["_resolved"] = status == "resolved"
        r["_in_primary"] = 0
        r["_q4_class"] = "excluded"

        if status != "resolved":
            r["_q4_class"] = f"excluded_{status or 'unknown'}"
            excluded.append(r)
            continue
        if selected != 1:
            if pl in ("negative", "gt_consistent"):
                r["_q4_class"] = f"secondary_non_selected_{pl}"
                secondary_non_selected.append(r)
            else:
                r["_q4_class"] = "excluded_resolved_other"
                excluded.append(r)
            continue
        if pl == "negative":
            r["_q4_class"] = "primary_negative"
            r["_in_primary"] = 1
            r["q4_y"] = 1  # negative class (reject target)
            primary.append(r)
        elif pl == "gt_consistent":
            r["_q4_class"] = "primary_positive_protect"
            r["_in_primary"] = 1
            r["q4_y"] = 0  # protect GT selected bridges
            primary.append(r)
        else:
            r["_q4_class"] = "excluded_resolved_selected_other"
            excluded.append(r)

    n_neg = sum(1 for r in primary if r.get("q4_y") == 1)
    n_pos = sum(1 for r in primary if r.get("q4_y") == 0)
    by_seq: dict[str, dict[str, int]] = defaultdict(lambda: {"n_neg": 0, "n_pos": 0})
    for r in primary:
        seq = str(r["sequence"])
        if r.get("q4_y") == 1:
            by_seq[seq]["n_neg"] += 1
        else:
            by_seq[seq]["n_pos"] += 1

    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "cohort_definition": {
            "negative_class": "resolved AND baseline_selected AND pair_label==negative",
            "positive_protection_class": (
                "resolved AND baseline_selected AND pair_label==gt_consistent"
            ),
            "excluded_from_main": "unresolved, ambiguous, non-selected, other labels",
            "secondary_analysis": "resolved non-selected negative / gt_consistent",
        },
        "n_d_online": len(rows),
        "n_primary": len(primary),
        "n_primary_negative": n_neg,
        "n_primary_positive_protect": n_pos,
        "n_secondary_non_selected": len(secondary_non_selected),
        "n_excluded": len(excluded),
        "per_sequence_primary": {s: by_seq[s] for s in sorted(by_seq)},
        "n_sequences_with_both_classes": sum(
            1 for s in by_seq.values() if s["n_neg"] > 0 and s["n_pos"] > 0
        ),
        "n_sequences_with_neg": sum(1 for s in by_seq.values() if s["n_neg"] > 0),
        "n_sequences_with_pos": sum(1 for s in by_seq.values() if s["n_pos"] > 0),
        "sufficient_mass": n_neg >= MIN_N_PER_CLASS and n_pos >= MIN_N_PER_CLASS,
    }
    return {
        "primary": primary,
        "secondary": secondary_non_selected,
        "excluded": excluded,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Sibling transforms (semantic-preserving, no new sources)
# ---------------------------------------------------------------------------


def sibling_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Build named feature columns for primary/secondary rows.

    Only transforms of the five frozen signals + margin to production bridge gate.
    """
    feats: dict[str, np.ndarray] = {}
    for sig in ORDERED_SIGNALS:
        x = np.asarray([_f(r.get(sig)) for r in rows], dtype=float)
        feats[sig] = x
        # log1p (signals are non-negative distances / abs ratios)
        with np.errstate(invalid="ignore"):
            feats[f"log1p__{sig}"] = np.log1p(np.clip(x, 0.0, None))
            feats[f"sq__{sig}"] = x * x
            feats[f"neg__{sig}"] = -x  # direction sibling
    # margin to production online bridge gate (predefined)
    sm = feats["score_m_bridge"]
    feats["margin_to_online_bridge_gate"] = ONLINE_BRIDGE_GATE_PX - sm
    # normalized residual sibling: resid / (dist_h + eps)
    dist = feats["dist_h"]
    feats["resid_over_dist_h"] = feats["resid_mean"] / (dist + 1e-6)
    return feats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _pair_auc_and_cliff(
    y: np.ndarray, x: np.ndarray
) -> tuple[float, float, float, str]:
    """AUC (P(x_neg > x_pos)), oriented max(AUC,1-AUC), Cliff's δ, preferred dir.

    y=1 negative class, y=0 positive protect.
    Cliff's δ = (n_gt - n_lt) / (n_neg * n_pos) for x_neg vs x_pos.
    """
    xn = x[y == 1]
    xp = x[y == 0]
    xn = xn[np.isfinite(xn)]
    xp = xp[np.isfinite(xp)]
    if len(xn) == 0 or len(xp) == 0:
        return float("nan"), float("nan"), float("nan"), "undefined"
    # Mann–Whitney: AUC = P(x_neg > x_pos) + 0.5 P(eq)
    xp_s = np.sort(xp)
    n_neg_gt = 0  # a > xp
    n_neg_lt = 0  # a < xp
    n_neg_eq = 0
    for a in xn:
        left = int(np.searchsorted(xp_s, a, side="left"))
        right = int(np.searchsorted(xp_s, a, side="right"))
        n_neg_gt += left  # count of xp strictly < a
        n_neg_eq += right - left
        n_neg_lt += len(xp_s) - right
    n_pairs = float(len(xn) * len(xp))
    auc = (n_neg_gt + 0.5 * n_neg_eq) / n_pairs
    cliff = (n_neg_gt - n_neg_lt) / n_pairs
    oriented = max(auc, 1.0 - auc)
    if abs(auc - 0.5) < 1e-12:
        direction = "none"
    elif auc >= 0.5:
        direction = "higher_in_negative"
    else:
        direction = "lower_in_negative"
    return float(auc), float(oriented), float(cliff), direction


def _class_stats(vals: np.ndarray) -> dict[str, Any]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return {
            "n": 0,
            "missing": int(np.sum(~np.isfinite(vals))),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
            "unique_count": 0,
            "constant": True,
        }
    qs = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(v)),
        "missing": int(np.sum(~np.isfinite(vals))),
        "min": float(np.min(v)),
        "q05": float(qs[0]),
        "q25": float(qs[1]),
        "median": float(qs[2]),
        "q75": float(qs[3]),
        "q95": float(qs[4]),
        "max": float(np.max(v)),
        "unique_count": int(len(np.unique(v))),
        "constant": bool(len(np.unique(v)) <= 1),
    }


def _support_overlap(xn: np.ndarray, xp: np.ndarray) -> dict[str, Any]:
    xn = xn[np.isfinite(xn)]
    xp = xp[np.isfinite(xp)]
    if len(xn) == 0 or len(xp) == 0:
        return {
            "overlap_interval": None,
            "overlap_width": float("nan"),
            "neg_in_pos_range_frac": float("nan"),
            "pos_in_neg_range_frac": float("nan"),
            "range_intersection_over_union": float("nan"),
        }
    nmin, nmax = float(np.min(xn)), float(np.max(xn))
    pmin, pmax = float(np.min(xp)), float(np.max(xp))
    lo, hi = max(nmin, pmin), min(nmax, pmax)
    width = max(0.0, hi - lo)
    union = max(nmax, pmax) - min(nmin, pmin)
    neg_in_pos = (
        float(np.mean((xn >= pmin) & (xn <= pmax))) if len(xn) else float("nan")
    )
    pos_in_neg = (
        float(np.mean((xp >= nmin) & (xp <= nmax))) if len(xp) else float("nan")
    )
    return {
        "overlap_interval": [lo, hi] if width > 0 or lo <= hi else None,
        "overlap_width": width if hi >= lo else 0.0,
        "neg_in_pos_range_frac": neg_in_pos,
        "pos_in_neg_range_frac": pos_in_neg,
        "range_intersection_over_union": (width / union) if union > 0 else float("nan"),
        "neg_range": [nmin, nmax],
        "pos_range": [pmin, pmax],
    }


def _ecdf_points(vals: np.ndarray, max_points: int = 64) -> list[dict[str, float]]:
    v = np.sort(vals[np.isfinite(vals)])
    if len(v) == 0:
        return []
    if len(v) <= max_points:
        idxs = np.arange(len(v))
    else:
        idxs = np.unique(np.linspace(0, len(v) - 1, max_points).astype(int))
    out = []
    n = len(v)
    for i in idxs:
        out.append({"x": float(v[i]), "ecdf": float((i + 1) / n)})
    return out


def pure_negative_tail_audit(
    y: np.ndarray, x: np.ndarray, sequences: np.ndarray
) -> dict[str, Any]:
    """Descriptive pure-neg prefix from each extreme. Not a thr candidate."""
    mask = np.isfinite(x)
    y, x, sequences = y[mask], x[mask], sequences[mask]
    out: dict[str, Any] = {}
    for name, ascending in (("from_low", True), ("from_high", False)):
        order = np.argsort(x, kind="mergesort")
        if not ascending:
            order = order[::-1]
        pure_n = 0
        pure_seqs: set[str] = set()
        for i in order:
            if int(y[i]) == 1:
                pure_n += 1
                pure_seqs.add(str(sequences[i]))
            else:
                break
        # GT contamination in extreme decile
        n = len(order)
        k = max(1, n // 10)
        head = order[:k]
        n_neg_head = int(np.sum(y[head] == 1))
        n_pos_head = int(np.sum(y[head] == 0))
        out[name] = {
            "pure_negative_prefix_n": pure_n,
            "pure_negative_prefix_n_sequences": len(pure_seqs),
            "pure_negative_prefix_sequences": sorted(pure_seqs),
            "extreme_decile_n": int(k),
            "extreme_decile_n_negative": n_neg_head,
            "extreme_decile_n_positive_gt": n_pos_head,
            "extreme_decile_gt_contamination_rate": float(n_pos_head / k)
            if k
            else float("nan"),
            "claim_status": (
                "descriptive_pure_neg_tail"
                if pure_n >= PURE_TAIL_MIN_N and len(pure_seqs) >= PURE_TAIL_MIN_SEQS
                else "no_multi_seq_pure_neg_tail"
            ),
            # Explicit: not a rule/candidate
            "not_a_rule_or_candidate": True,
        }
    return out


def evaluate_feature(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, Any]:
    xn = x[y == 1]
    xp = x[y == 0]
    auc, oriented, cliff, direction = _pair_auc_and_cliff(y, x)
    n_missing = int(np.sum(~np.isfinite(x)))
    n_ties = 0
    if np.isfinite(x).any():
        # fraction of pairs that are ties (approx via unique)
        n_ties = int(len(x) - len(np.unique(x[np.isfinite(x)])))
    row = {
        "feature": name,
        "n_total": int(len(y)),
        "n_negative": int(np.sum(y == 1)),
        "n_positive_protect": int(np.sum(y == 0)),
        "n_missing": n_missing,
        "missing_rate": float(n_missing / len(y)) if len(y) else float("nan"),
        "tie_proxy_n_nonunique": n_ties,
        "constant_rate": float(1.0 if len(np.unique(x[np.isfinite(x)])) <= 1 else 0.0),
        "auc_raw_higher_neg": auc,
        "auc_oriented": oriented,
        "cliffs_delta": cliff,
        "rank_biserial_approx": cliff,  # equal to Cliff's δ for binary
        "direction": direction,
        "neg_stats": _class_stats(xn),
        "pos_stats": _class_stats(xp),
        "support_overlap": _support_overlap(xn, xp),
        "ecdf_neg": _ecdf_points(xn),
        "ecdf_pos": _ecdf_points(xp),
        "pure_tail": pure_negative_tail_audit(y, x, sequences),
        "effect_band": _effect_band(oriented, cliff),
    }
    return row


def _effect_band(oriented_auc: float, cliff: float) -> str:
    if not math.isfinite(oriented_auc):
        return "undefined"
    ad = abs(cliff) if math.isfinite(cliff) else 0.0
    if oriented_auc >= AUC_STRONG and ad >= CLIFF_MEDIUM:
        return "strong"
    if oriented_auc >= AUC_MODERATE and ad >= CLIFF_SMALL:
        return "moderate"
    if oriented_auc >= 0.55 or ad >= CLIFF_SMALL:
        return "weak"
    return "negligible"


# ---------------------------------------------------------------------------
# Per-sequence + LOO
# ---------------------------------------------------------------------------


def per_sequence_feature_rows(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
) -> list[dict[str, Any]]:
    out = []
    for seq in sorted(set(str(s) for s in sequences)):
        m = sequences == seq
        ys, xs = y[m], x[m]
        n_neg = int(np.sum(ys == 1))
        n_pos = int(np.sum(ys == 0))
        if n_neg == 0 or n_pos == 0:
            out.append(
                {
                    "feature": name,
                    "sequence": seq,
                    "n_neg": n_neg,
                    "n_pos": n_pos,
                    "auc_raw_higher_neg": float("nan"),
                    "auc_oriented": float("nan"),
                    "cliffs_delta": float("nan"),
                    "direction": "undefined_small_n",
                    "flag": "small_n_missing_class",
                }
            )
            continue
        auc, oriented, cliff, direction = _pair_auc_and_cliff(ys, xs)
        flag = "ok"
        if n_neg < 2 or n_pos < 2:
            flag = "small_n_unstable"
        out.append(
            {
                "feature": name,
                "sequence": seq,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "auc_raw_higher_neg": auc,
                "auc_oriented": oriented,
                "cliffs_delta": cliff,
                "direction": direction,
                "flag": flag,
            }
        )
    return out


def loo_sequence_feature_rows(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
) -> list[dict[str, Any]]:
    seqs = sorted(set(str(s) for s in sequences))
    out = []
    for hold in seqs:
        m = sequences != hold
        ys, xs = y[m], x[m]
        n_neg = int(np.sum(ys == 1))
        n_pos = int(np.sum(ys == 0))
        if n_neg == 0 or n_pos == 0:
            out.append(
                {
                    "feature": name,
                    "hold_out_sequence": hold,
                    "n_neg": n_neg,
                    "n_pos": n_pos,
                    "auc_raw_higher_neg": float("nan"),
                    "auc_oriented": float("nan"),
                    "cliffs_delta": float("nan"),
                    "direction": "undefined",
                    "flag": "missing_class_after_holdout",
                }
            )
            continue
        auc, oriented, cliff, direction = _pair_auc_and_cliff(ys, xs)
        out.append(
            {
                "feature": name,
                "hold_out_sequence": hold,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "auc_raw_higher_neg": auc,
                "auc_oriented": oriented,
                "cliffs_delta": cliff,
                "direction": direction,
                "flag": "ok",
            }
        )
    return out


def stability_flags(
    pooled: Mapping[str, Any],
    per_seq: Sequence[Mapping[str, Any]],
    loo: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Sign flip / single-seq dependence / small-n."""
    base_dir = pooled.get("direction")
    usable_seq = [
        r
        for r in per_seq
        if r.get("direction") not in ("undefined_small_n", "undefined", "none")
        and r.get("flag") != "small_n_missing_class"
    ]
    dirs = [r["direction"] for r in usable_seq if r.get("direction") != "none"]
    sign_flips = 0
    if base_dir in ("higher_in_negative", "lower_in_negative"):
        sign_flips = sum(1 for d in dirs if d != base_dir and d != "none")
    loo_dirs = [
        r["direction"]
        for r in loo
        if r.get("direction") in ("higher_in_negative", "lower_in_negative")
    ]
    loo_flip = False
    if base_dir in ("higher_in_negative", "lower_in_negative") and loo_dirs:
        loo_flip = any(d != base_dir for d in loo_dirs)
    loo_aucs = [
        float(r["auc_oriented"])
        for r in loo
        if math.isfinite(float(r.get("auc_oriented", float("nan"))))
    ]
    min_loo = min(loo_aucs) if loo_aucs else float("nan")
    # single-seq dependence: one seq accounts for >50% of |cliff| contribution heuristic
    # simpler: drop max-neg seq and check oriented drop > 0.1
    return {
        "pooled_direction": base_dir,
        "n_sequences_evaluated": len(usable_seq),
        "per_seq_sign_flips_vs_pooled": sign_flips,
        "loo_direction_flip": loo_flip,
        "loo_min_oriented_auc": min_loo,
        "loo_all_oriented_ge_min": (
            bool(loo_aucs) and all(a >= MIN_LOO_ORIENTED_AUC for a in loo_aucs)
        ),
        "small_n_seq_count": sum(
            1
            for r in per_seq
            if r.get("flag") in ("small_n_unstable", "small_n_missing_class")
        ),
        "stable_candidate": bool(
            pooled.get("effect_band") in ("strong", "moderate")
            and not loo_flip
            and sign_flips == 0
            and loo_aucs
            and all(a >= MIN_LOO_ORIENTED_AUC for a in loo_aucs)
            and len(usable_seq) >= MIN_SEQS_WITH_BOTH
        ),
    }


# ---------------------------------------------------------------------------
# Conditional slices (predefined only)
# ---------------------------------------------------------------------------


def apply_predefined_slices(
    primary: Sequence[Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Evaluate frozen signals on predefined context slices only."""
    y = np.asarray([int(r["q4_y"]) for r in primary], dtype=int)
    sequences = np.asarray([str(r["sequence"]) for r in primary], dtype=object)
    slice_defs: list[tuple[str, Any]] = [
        ("all_primary", lambda r: True),
    ]
    for name, pred in COMPETITOR_SLICES:
        slice_defs.append((name, pred))
    for name, pred in BDIST_SLICES:
        slice_defs.append((name, pred))
    # per-sequence as context (predefined)
    for seq in sorted({str(r["sequence"]) for r in primary}):
        slice_defs.append(
            (f"sequence::{seq}", lambda r, s=seq: str(r["sequence"]) == s)
        )

    rows_out: list[dict[str, Any]] = []
    for slice_name, pred in slice_defs:
        mask = np.asarray([bool(pred(r)) for r in primary], dtype=bool)
        n = int(mask.sum())
        n_neg = int(np.sum(y[mask] == 1))
        n_pos = int(np.sum(y[mask] == 0))
        n_seqs = len(
            {str(primary[i]["sequence"]) for i in range(len(primary)) if mask[i]}
        )
        n_seqs_both = 0
        by_seq: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for i, r in enumerate(primary):
            if not mask[i]:
                continue
            s = str(r["sequence"])
            if y[i] == 1:
                by_seq[s][0] += 1
            else:
                by_seq[s][1] += 1
        n_seqs_both = sum(1 for a, b in by_seq.values() if a > 0 and b > 0)

        base = {
            "slice": slice_name,
            "coverage": float(n / len(primary)) if primary else 0.0,
            "n": n,
            "n_neg": n_neg,
            "n_pos": n_pos,
            "n_sequences": n_seqs,
            "n_sequences_with_both_classes": n_seqs_both,
        }
        if n_neg < 2 or n_pos < 2:
            for sig in ORDERED_SIGNALS:
                rows_out.append(
                    {
                        **base,
                        "feature": sig,
                        "auc_oriented": float("nan"),
                        "cliffs_delta": float("nan"),
                        "direction": "undefined",
                        "effect_band": "insufficient_slice_mass",
                        "stable_candidate": False,
                    }
                )
            continue
        for sig in ORDERED_SIGNALS:
            x = feats[sig][mask]
            ys = y[mask]
            seqs = sequences[mask]
            pooled = evaluate_feature(sig, x, ys, seqs)
            per = per_sequence_feature_rows(sig, x, ys, seqs)
            loo = loo_sequence_feature_rows(sig, x, ys, seqs)
            stab = stability_flags(pooled, per, loo)
            rows_out.append(
                {
                    **base,
                    "feature": sig,
                    "auc_raw_higher_neg": pooled["auc_raw_higher_neg"],
                    "auc_oriented": pooled["auc_oriented"],
                    "cliffs_delta": pooled["cliffs_delta"],
                    "direction": pooled["direction"],
                    "effect_band": pooled["effect_band"],
                    "loo_direction_flip": stab["loo_direction_flip"],
                    "loo_min_oriented_auc": stab["loo_min_oriented_auc"],
                    "stable_candidate": stab["stable_candidate"],
                    "pure_tail_high": pooled["pure_tail"]["from_high"][
                        "pure_negative_prefix_n"
                    ],
                    "pure_tail_low": pooled["pure_tail"]["from_low"][
                        "pure_negative_prefix_n"
                    ],
                }
            )
    return rows_out


# ---------------------------------------------------------------------------
# Terminal classification
# ---------------------------------------------------------------------------


def classify_q4_terminal(
    *,
    cohort_summary: Mapping[str, Any],
    pooled_features: Sequence[Mapping[str, Any]],
    stability_by_feature: Mapping[str, Mapping[str, Any]],
    slice_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n_neg = int(cohort_summary["n_primary_negative"])
    n_pos = int(cohort_summary["n_primary_positive_protect"])
    blocked = [
        "threshold_search_not_authorized",
        "boolean_rule_search_not_authorized",
        "formal_safe_region_not_authorized",
        "hook_policy_not_authorized",
        "production_promotion_blocked",
        "frozen_policy_effect_claim_inadmissible",
    ]
    allowed: list[str] = []

    if n_neg < MIN_N_PER_CLASS or n_pos < MIN_N_PER_CLASS:
        return {
            "stage2_q4_separability": "insufficient_labeled_decision_mass",
            "terminal_letter": "D",
            "reason": f"n_neg={n_neg} n_pos={n_pos} below MIN_N_PER_CLASS={MIN_N_PER_CLASS}",
            "claims_blocked": blocked,
            "claims_allowed": [],
            "next_authorized_step": (
                "increase labeled selected decision mass or fix join coverage "
                "before separability claims"
            ),
            "supporting_features": [],
            "production_preset": "unchanged",
        }

    # A: single-signal stable
    a_feats = []
    for p in pooled_features:
        name = p["feature"]
        # only raw frozen signals for A (siblings reported but A needs primary)
        if name not in ORDERED_SIGNALS:
            continue
        stab = stability_by_feature.get(name, {})
        pure = p.get("pure_tail", {})
        pure_ok = False
        for side in ("from_high", "from_low"):
            t = pure.get(side, {})
            if (
                int(t.get("pure_negative_prefix_n", 0)) >= PURE_TAIL_MIN_N
                and int(t.get("pure_negative_prefix_n_sequences", 0))
                >= PURE_TAIL_MIN_SEQS
            ):
                pure_ok = True
        if (
            stab.get("stable_candidate")
            and pure_ok
            and p.get("effect_band") in ("strong", "moderate")
            and float(p.get("auc_oriented", 0)) >= AUC_MODERATE
        ):
            a_feats.append(name)

    if a_feats:
        allowed.append("restricted_safe_region_modeling_authorized")
        allowed.append(f"single_signal_stable:{','.join(a_feats)}")
        return {
            "stage2_q4_separability": "single_signal_separability_supported",
            "terminal_letter": "A",
            "reason": f"stable single-signal(s) with multi-seq pure-neg tail: {a_feats}",
            "claims_blocked": blocked,
            "claims_allowed": allowed,
            "next_authorized_step": (
                "restricted safe-region modeling on supported signal(s) only; "
                "still no production preset change"
            ),
            "supporting_features": a_feats,
            "production_preset": "unchanged",
        }

    # B: conditional on predefined non-sequence-all slices
    b_hits = []
    for r in slice_rows:
        if str(r.get("slice", "")).startswith("sequence::"):
            continue
        if r.get("slice") == "all_primary":
            continue
        if (
            r.get("stable_candidate")
            and r.get("effect_band") in ("strong", "moderate")
            and int(r.get("n_sequences_with_both_classes", 0)) >= MIN_SEQS_WITH_BOTH
            and float(r.get("coverage", 0)) >= 0.2
            and float(r.get("auc_oriented", 0) or 0) >= AUC_MODERATE
        ):
            # pure tail in slice
            if (
                int(r.get("pure_tail_high", 0)) >= PURE_TAIL_MIN_N
                or int(r.get("pure_tail_low", 0)) >= PURE_TAIL_MIN_N
            ):
                b_hits.append(f"{r['slice']}::{r['feature']}")

    if b_hits:
        allowed.append("restricted_conditional_safe_region_modeling_authorized")
        return {
            "stage2_q4_separability": "conditional_separability_supported",
            "terminal_letter": "B",
            "reason": f"stable predefined-context slices: {b_hits[:5]}",
            "claims_blocked": blocked,
            "claims_allowed": allowed,
            "next_authorized_step": (
                "restricted safe-region modeling inside declared context slices only"
            ),
            "supporting_features": b_hits,
            "production_preset": "unchanged",
        }

    # C: weak/unstable
    best = max(
        (
            float(p.get("auc_oriented") or 0)
            for p in pooled_features
            if p["feature"] in ORDERED_SIGNALS
        ),
        default=0.0,
    )
    best_feat = None
    for p in pooled_features:
        if (
            p["feature"] in ORDERED_SIGNALS
            and float(p.get("auc_oriented") or 0) == best
        ):
            best_feat = p["feature"]
            break
    blocked.append("safe_region_modeling_not_authorized")
    blocked.append("change_signal_family_recommended")
    return {
        "stage2_q4_separability": "separability_weak_or_unstable",
        "terminal_letter": "C",
        "reason": (
            f"best oriented AUC among frozen signals={best:.3f} ({best_feat}); "
            "thick overlap / no multi-seq pure-neg tail / LOO or effect gates fail"
        ),
        "claims_blocked": blocked,
        "claims_allowed": [
            "report_descriptive_separability_facts",
            "authorize_signal_family_change_or_placement_revisit",
        ],
        "next_authorized_step": (
            "change signal family (or earlier hook placement) — "
            "do NOT threshold-chase on inseparable support"
        ),
        "supporting_features": [],
        "best_oriented_auc": best,
        "best_feature": best_feat,
        "production_preset": "unchanged",
    }


# ---------------------------------------------------------------------------
# Secondary analysis (non-selected; not main conclusion)
# ---------------------------------------------------------------------------


def secondary_non_selected_summary(
    secondary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n_neg = sum(1 for r in secondary if r.get("pair_label") == "negative")
    n_gt = sum(1 for r in secondary if r.get("pair_label") == "gt_consistent")
    if n_neg < 2 or n_gt < 2:
        return {
            "role": "secondary_only_not_for_main_conclusion",
            "n": len(secondary),
            "n_negative": n_neg,
            "n_gt_consistent": n_gt,
            "note": "insufficient or reported only as secondary; not mixed into primary cohort",
            "features": [],
        }
    # build y: neg=1, gt=0
    rows = [
        r for r in secondary if r.get("pair_label") in ("negative", "gt_consistent")
    ]
    y = np.asarray([1 if r["pair_label"] == "negative" else 0 for r in rows], dtype=int)
    sequences = np.asarray([str(r["sequence"]) for r in rows], dtype=object)
    feats = sibling_feature_matrix(rows)
    features = []
    for sig in ORDERED_SIGNALS:
        p = evaluate_feature(sig, feats[sig], y, sequences)
        features.append(
            {
                "feature": sig,
                "auc_oriented": p["auc_oriented"],
                "cliffs_delta": p["cliffs_delta"],
                "direction": p["direction"],
                "effect_band": p["effect_band"],
            }
        )
    return {
        "role": "secondary_only_not_for_main_conclusion",
        "n": len(rows),
        "n_negative": n_neg,
        "n_gt_consistent": n_gt,
        "note": "resolved non-selected; decision-irrelevant for reject-selected policy",
        "features": features,
    }


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile_q4(
    *,
    n_d_online: int,
    cohort: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    s = cohort
    n_pri = len(primary)
    n_neg = sum(1 for r in primary if r.get("q4_y") == 1)
    n_pos = sum(1 for r in primary if r.get("q4_y") == 0)
    checks = {
        "n_primary_eq_neg_plus_pos": n_pri == n_neg + n_pos,
        "n_primary_matches_summary": n_pri == int(s["n_primary"]),
        "n_neg_matches_summary": n_neg == int(s["n_primary_negative"]),
        "n_pos_matches_summary": n_pos == int(s["n_primary_positive_protect"]),
        "partition_covers_d_online": (
            int(s["n_primary"])
            + int(s["n_secondary_non_selected"])
            + int(s["n_excluded"])
            == n_d_online
        ),
        "no_unresolved_in_primary": all(
            str(r.get("label_status")) == "resolved" for r in primary
        ),
        "all_primary_selected": all(
            _i(r.get("baseline_selected", 0)) == 1 for r in primary
        ),
        "primary_labels_only_neg_or_gt": all(
            r.get("pair_label") in ("negative", "gt_consistent") for r in primary
        ),
    }
    # per-seq sum
    by_seq = s.get("per_sequence_primary", {})
    sum_neg = sum(int(v["n_neg"]) for v in by_seq.values())
    sum_pos = sum(int(v["n_pos"]) for v in by_seq.values())
    checks["per_seq_neg_sum"] = sum_neg == n_neg
    checks["per_seq_pos_sum"] = sum_pos == n_pos
    ok = all(checks.values())
    return {
        "ok": ok,
        "acceptance": "PASS" if ok else "FAIL_CLOSED",
        "checks": checks,
        "errors": [k for k, v in checks.items() if not v],
        "counts": {
            "n_d_online": n_d_online,
            "n_primary": n_pri,
            "n_neg": n_neg,
            "n_pos": n_pos,
        },
    }


# ---------------------------------------------------------------------------
# Full study runner
# ---------------------------------------------------------------------------


def run_stage2_q4_audit(
    *,
    q1q3_study_dir: Path,
    out_dir: Path,
    git_commit: str = "",
    study_id: str | None = None,
) -> dict[str, Any]:
    q1q3_study_dir = Path(q1q3_study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_d_online_events(q1q3_study_dir)
    source = q1q3_study_dir / "d_online_events.parquet"
    if not source.is_file():
        source = q1q3_study_dir / "d_online_events.csv"
    source_hash = _sha256_file(source) if source.is_file() else ""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = study_id or f"m_b1_5_stage2_q4_{stamp}"

    locked = lock_q4_cohort(events)
    primary = locked["primary"]
    secondary = locked["secondary"]
    cohort_sum = locked["summary"]

    recon = reconcile_q4(n_d_online=len(events), cohort=cohort_sum, primary=primary)
    if not recon["ok"]:
        raise Stage2Q4Error(f"Q4 reconciliation FAIL_CLOSED: {recon['errors']}")

    y = np.asarray([int(r["q4_y"]) for r in primary], dtype=int)
    sequences = np.asarray([str(r["sequence"]) for r in primary], dtype=object)
    feats = sibling_feature_matrix(primary)

    # Feature order: raw signals first, then siblings
    feature_names = list(ORDERED_SIGNALS) + [
        k for k in feats.keys() if k not in ORDERED_SIGNALS
    ]

    pooled_rows: list[dict[str, Any]] = []
    per_seq_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    stability_by_feature: dict[str, dict[str, Any]] = {}
    flat_signal_table: list[dict[str, Any]] = []

    for name in feature_names:
        x = feats[name]
        pooled = evaluate_feature(name, x, y, sequences)
        per = per_sequence_feature_rows(name, x, y, sequences)
        loo = loo_sequence_feature_rows(name, x, y, sequences)
        stab = stability_flags(pooled, per, loo)
        pooled["stability"] = stab
        pooled_rows.append(pooled)
        per_seq_rows.extend(per)
        loo_rows.extend(loo)
        stability_by_feature[name] = stab
        flat_signal_table.append(
            {
                "feature": name,
                "is_frozen_raw": int(name in ORDERED_SIGNALS),
                "n_neg": pooled["n_negative"],
                "n_pos": pooled["n_positive_protect"],
                "auc_raw_higher_neg": pooled["auc_raw_higher_neg"],
                "auc_oriented": pooled["auc_oriented"],
                "cliffs_delta": pooled["cliffs_delta"],
                "direction": pooled["direction"],
                "effect_band": pooled["effect_band"],
                "neg_median": pooled["neg_stats"]["median"],
                "pos_median": pooled["pos_stats"]["median"],
                "neg_q05": pooled["neg_stats"]["q05"],
                "neg_q95": pooled["neg_stats"]["q95"],
                "pos_q05": pooled["pos_stats"]["q05"],
                "pos_q95": pooled["pos_stats"]["q95"],
                "overlap_iou_range": pooled["support_overlap"][
                    "range_intersection_over_union"
                ],
                "neg_in_pos_range_frac": pooled["support_overlap"][
                    "neg_in_pos_range_frac"
                ],
                "pure_neg_prefix_high": pooled["pure_tail"]["from_high"][
                    "pure_negative_prefix_n"
                ],
                "pure_neg_prefix_low": pooled["pure_tail"]["from_low"][
                    "pure_negative_prefix_n"
                ],
                "pure_neg_prefix_high_seqs": pooled["pure_tail"]["from_high"][
                    "pure_negative_prefix_n_sequences"
                ],
                "pure_neg_prefix_low_seqs": pooled["pure_tail"]["from_low"][
                    "pure_negative_prefix_n_sequences"
                ],
                "loo_direction_flip": stab["loo_direction_flip"],
                "loo_min_oriented_auc": stab["loo_min_oriented_auc"],
                "stable_candidate": stab["stable_candidate"],
                "missing_rate": pooled["missing_rate"],
                "constant_rate": pooled["constant_rate"],
            }
        )

    slice_rows = apply_predefined_slices(primary, feats)
    secondary_sum = secondary_non_selected_summary(secondary)
    terminal = classify_q4_terminal(
        cohort_summary=cohort_sum,
        pooled_features=pooled_rows,
        stability_by_feature=stability_by_feature,
        slice_rows=slice_rows,
    )

    # Cohort export rows (machine-readable)
    cohort_export = []
    for r in primary:
        row = {
            "study_id": sid,
            "sequence": r["sequence"],
            "frame": r.get("frame"),
            "event_id": r.get("event_id"),
            "join_key": r.get("join_key"),
            "runtime_candidate_id": r.get("runtime_candidate_id"),
            "pair_label": r.get("pair_label"),
            "label_status": r.get("label_status"),
            "baseline_selected": r.get("baseline_selected"),
            "q4_class": r.get("_q4_class"),
            "q4_y": r.get("q4_y"),
            "competitor_count": r.get("competitor_count"),
            "score_m_bridge": r.get("score_m_bridge"),
            "abs_log_h": r.get("abs_log_h"),
            "dist_h": r.get("dist_h"),
            "abs_ratio_m1": r.get("abs_ratio_m1"),
            "resid_mean": r.get("resid_mean"),
            "margin_to_online_bridge_gate": ONLINE_BRIDGE_GATE_PX
            - _f(r.get("score_m_bridge")),
        }
        cohort_export.append(row)

    # Tail audit table (descriptive)
    tail_rows = []
    for p in pooled_rows:
        if p["feature"] not in ORDERED_SIGNALS:
            continue
        for side, t in p["pure_tail"].items():
            tail_rows.append(
                {
                    "feature": p["feature"],
                    "side": side,
                    **{
                        k: v
                        for k, v in t.items()
                        if k != "pure_negative_prefix_sequences"
                    },
                    "sequences": "|".join(
                        t.get("pure_negative_prefix_sequences") or []
                    ),
                }
            )

    hashes: dict[str, str] = {}
    hashes["q4_cohort"] = write_csv(out_dir / "q4_cohort.csv", cohort_export)
    write_parquet(out_dir / "q4_cohort.parquet", cohort_export)
    if (out_dir / "q4_cohort.parquet").is_file():
        hashes["q4_cohort_parquet"] = _sha256_file(out_dir / "q4_cohort.parquet")

    hashes["q4_cohort_summary"] = write_json(
        out_dir / "q4_cohort_summary.json", cohort_sum
    )
    hashes["q4_signal_separability"] = write_csv(
        out_dir / "q4_signal_separability.csv", flat_signal_table
    )
    # full pooled json (ECDFs etc.)
    pooled_detail_features = []
    for p in pooled_rows:
        detail = {k: v for k, v in p.items() if k not in ("ecdf_neg", "ecdf_pos")}
        detail["ecdf_neg_n"] = len(p.get("ecdf_neg") or [])
        detail["ecdf_pos_n"] = len(p.get("ecdf_pos") or [])
        pooled_detail_features.append(detail)
    hashes["q4_pooled_detail"] = write_json(
        out_dir / "q4_pooled_detail.json",
        {"features": pooled_detail_features},
    )
    # ECDF data separate
    ecdf_rows = []
    for p in pooled_rows:
        for cls, key in (("neg", "ecdf_neg"), ("pos", "ecdf_pos")):
            for pt in p.get(key) or []:
                ecdf_rows.append(
                    {
                        "feature": p["feature"],
                        "class": cls,
                        "x": pt["x"],
                        "ecdf": pt["ecdf"],
                    }
                )
    hashes["q4_ecdf"] = write_csv(out_dir / "q4_ecdf.csv", ecdf_rows)
    hashes["q4_per_sequence"] = write_csv(out_dir / "q4_per_sequence.csv", per_seq_rows)
    hashes["q4_loo"] = write_csv(out_dir / "q4_loo.csv", loo_rows)
    hashes["q4_tail_audit"] = write_csv(out_dir / "q4_tail_audit.csv", tail_rows)
    hashes["q4_slice_audit"] = write_csv(out_dir / "q4_slice_audit.csv", slice_rows)
    hashes["q4_secondary_non_selected"] = write_json(
        out_dir / "q4_secondary_non_selected.json", secondary_sum
    )
    hashes["reconciliation"] = write_json(out_dir / "reconciliation.json", recon)

    # Flatten stability
    stab_rows = [{"feature": k, **v} for k, v in stability_by_feature.items()]
    hashes["q4_stability"] = write_csv(out_dir / "q4_stability.csv", stab_rows)

    summary = {
        "study_id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_q1q3_study": str(q1q3_study_dir),
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        "D_online_total": len(events),
        "n_primary": cohort_sum["n_primary"],
        "n_primary_negative": cohort_sum["n_primary_negative"],
        "n_primary_positive_protect": cohort_sum["n_primary_positive_protect"],
        "n_secondary_non_selected": cohort_sum["n_secondary_non_selected"],
        "n_sequences_with_both_classes": cohort_sum["n_sequences_with_both_classes"],
        **terminal,
        "reconciliation_acceptance": recon["acceptance"],
        "online_bridge_gate_px": ONLINE_BRIDGE_GATE_PX,
        "frozen_thr_vector": list(FROZEN_THR_VECTOR),
        "forbidden_this_round": [
            "threshold_grid",
            "boolean_rule_search",
            "formal_safe_region_candidate",
            "hook_policy",
            "production_preset_change",
            "mixing_non_selected_into_primary",
        ],
    }
    hashes["summary"] = write_json(out_dir / "summary.json", summary)
    md = _render_q4_md(summary, flat_signal_table, cohort_sum)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    hashes["summary_md"] = hashlib.sha256(md.encode("utf-8")).hexdigest()

    manifest = {
        "schema": "m_b1_5_stage2_q4_manifest_v1",
        "study_id": sid,
        "git_commit": git_commit,
        "source_q1q3_study": str(q1q3_study_dir),
        "source_event_table": str(source),
        "source_event_table_hash": source_hash,
        "candidate_universe_id": CANDIDATE_UNIVERSE_ID,
        "substrate_id": SUBSTRATE_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "cohort_definition": cohort_sum["cohort_definition"],
        "sequence_set": sorted({str(r["sequence"]) for r in primary}),
        "artifact_hashes": hashes,
        "created_utc": summary["created_utc"],
    }
    write_json(out_dir / "manifest.json", manifest)
    summary["manifest"] = manifest
    return summary


def _render_q4_md(
    summary: Mapping[str, Any],
    flat: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
) -> str:
    lines = [
        f"# Stage 2 Q4 separability — `{summary.get('study_id')}`",
        "",
        "<!-- doc-status: research-artifact -->",
        "",
        "## Terminal",
        "",
        "```text",
        f"stage2_q4_separability: {summary.get('stage2_q4_separability')}",
        f"terminal_letter: {summary.get('terminal_letter')}",
        f"n_primary_negative: {summary.get('n_primary_negative')}",
        f"n_primary_positive_protect: {summary.get('n_primary_positive_protect')}",
        f"next_authorized_step: {summary.get('next_authorized_step')}",
        f"production_preset: {summary.get('production_preset')}",
        "```",
        "",
        f"**Reason:** {summary.get('reason')}",
        "",
        "## Primary cohort",
        "",
        f"- negative (selected∧resolved∧FP): **{cohort['n_primary_negative']}**",
        f"- positive protect (selected∧resolved∧GT): **{cohort['n_primary_positive_protect']}**",
        f"- sequences with both classes: {cohort['n_sequences_with_both_classes']}",
        "",
        "## Frozen signals (pooled)",
        "",
        "| feature | AUC_oriented | Cliff δ | direction | effect | pure_hi | pure_lo | LOO flip |",
        "|:--|--:|--:|:--|:--|--:|--:|:--|",
    ]
    for r in flat:
        if not int(r.get("is_frozen_raw", 0)):
            continue
        lines.append(
            f"| {r['feature']} | {float(r['auc_oriented']):.3f} | "
            f"{float(r['cliffs_delta']):.3f} | {r['direction']} | "
            f"{r['effect_band']} | {r['pure_neg_prefix_high']} | "
            f"{r['pure_neg_prefix_low']} | {r['loo_direction_flip']} |"
        )
    lines += [
        "",
        "## Claim firewall",
        "",
        "Blocked:",
        "",
    ]
    for b in summary.get("claims_blocked", []):
        lines.append(f"- `{b}`")
    lines += ["", "Allowed:", ""]
    for a in summary.get("claims_allowed", []) or ["(none)"]:
        lines.append(f"- `{a}`")
    lines += [
        "",
        "No formal threshold, Boolean sufficient condition, or hook policy in this pack.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_q4_primary_row(
    *,
    event_id: str,
    sequence: str,
    pair_label: str,
    score_m_bridge: float,
    abs_log_h: float = 0.1,
    dist_h: float = 0.2,
    abs_ratio_m1: float = 0.1,
    resid_mean: float = 0.2,
    competitor_count: int = 0,
    frame: int = 1,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "join_key": event_id,
        "sequence": sequence,
        "frame": frame,
        "runtime_candidate_id": event_id,
        "label_status": "resolved",
        "pair_label": pair_label,
        "baseline_selected": 1,
        "baseline_accepted_candidate": 1,
        "competitor_count": competitor_count,
        "score_m_bridge": score_m_bridge,
        "abs_log_h": abs_log_h,
        "dist_h": dist_h,
        "abs_ratio_m1": abs_ratio_m1,
        "resid_mean": resid_mean,
    }
