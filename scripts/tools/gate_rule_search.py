#!/usr/bin/env python3
"""Constrained multi-gate rule search (not combinatorial thrashing).

Architecture
------------
Layer 0  transform roles   (documented; raw→log for fusion, not thr magic)
Layer 1  atom generation   continuous → interpretable boolean reject masks
Layer 2  conjunction mining  FP-heavy / GT-rare AND clauses (itemset + prune)
Layer 3  submodular greedy   select complementary OR of clauses under ε

Math tools used
---------------
* Pareto / dominance pruning on (FP_removed, −GT_hurt, −seq_std, −complexity)
* ε-constrained opt: max FP_removed s.t. GT_hurt≤ε (and optional τ, β)
* Monotone AND: A∩B∩C ⊆ A∩B ⊆ A  → support / upper-bound prune
* Submodular-ish greedy coverage of FP with diminishing returns

Usage
-----
  uv run python scripts/tools/gate_rule_search.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_gate_rule_search_<stamp> \\
    --eps 0.0 --max-and-size 3 --max-or-rules 5

Does NOT flip production presets. RESEARCH / default-off.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_AUDIT = Path(__file__).resolve().parent / "audit_relink_safe_reject.py"
_spec = importlib.util.spec_from_file_location("audit_relink_safe_reject", _AUDIT)
assert _spec and _spec.loader
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


# ── roles (Layer 0 semantics; not free-form fusion) ─────────────────────────
# condition: defines when to apply / zone
# support: candidate safe-reject evidence
# fusion: for weighted scores (log/z) — NOT used as hard thr here
# diagnostic: stability metrics only

ROLE_MAP: dict[str, str] = {
    "score_m_bridge": "condition",  # operation-zone energy
    "bridge_dist": "condition",
    "dist_h": "condition",
    "resid_mean": "condition",
    "abs_log_h": "support",
    "abs_ratio_m1": "support",
    "neg_dir_cos": "support",  # weak; still atom for mining
    "speed_mismatch": "diagnostic",  # weak hard-pool — limited atom use
    "gap": "condition",
}


@dataclass
class Atom:
    atom_id: str
    signal: str
    role: str
    kind: str  # tail_q / hard_zone / body_band / gap_bin
    description: str
    reject: np.ndarray  # bool, True = reject
    complexity: int = 1  # atom cost
    # portable definition for LOO / freeze (fit thr on train, apply elsewhere)
    thr: float | None = None
    op: str = ">"  # score op thr  (or "in_range" for gap bins)
    thr_hi: float | None = None  # for gap bins: lo=thr, hi=thr_hi
    quantile: float | None = None


@dataclass
class Clause:
    clause_id: str
    atom_ids: tuple[str, ...]
    reject: np.ndarray
    complexity: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleMetrics:
    FP_removed: int
    GT_hurt: int
    FP_removed_rate: float
    GT_hurt_rate: float
    n_pos: int
    n_neg: int
    seq_hurt_std: float
    boundary_mass: (
        float  # proxy: fraction of rejected GT among all GT (same as hurt rate)
    )
    complexity: int
    safe_level: str


def _metrics(
    y: np.ndarray,
    rej: np.ndarray,
    seq: np.ndarray,
    complexity: int,
) -> RuleMetrics:
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    hurt = int((y & rej).sum())
    fprm = int((~y & rej).sum())
    rates = []
    for s in np.unique(seq):
        m = seq == s
        ys = y[m]
        if ys.sum() == 0:
            continue
        rates.append(float((ys & rej[m]).sum() / ys.sum()))
    seq_std = float(np.std(rates)) if len(rates) >= 2 else 0.0
    gtr = hurt / n_pos if n_pos else 0.0
    if gtr <= 0.0:
        lvl = "eps0"
    elif gtr <= 0.001:
        lvl = "eps0_1pct"
    elif gtr <= 0.01:
        lvl = "eps1pct"
    else:
        lvl = "unsafe"
    return RuleMetrics(
        FP_removed=fprm,
        GT_hurt=hurt,
        FP_removed_rate=fprm / n_neg if n_neg else 0.0,
        GT_hurt_rate=gtr,
        n_pos=n_pos,
        n_neg=n_neg,
        seq_hurt_std=seq_std,
        boundary_mass=gtr,  # for atoms: same; clauses can refine later
        complexity=complexity,
        safe_level=lvl,
    )


def extract_signals(pool: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _audit.ensure_prod_proxy_scores(pool)
    return {
        "score_m_bridge": pool["score_m_bridge"],
        "bridge_dist": pool["bridge_dist"],
        "dist_h": pool["dist_h"],
        "resid_mean": 0.5 * (pool["fwd_resid"] + pool["bwd_resid"]),
        "abs_log_h": pool["log_h_ratio"],
        "abs_ratio_m1": np.abs(pool["h_ratio_lost_over_cand"] - 1.0),
        "neg_dir_cos": -pool["dir_cos"],
        "speed_mismatch": pool["speed_mismatch"],
        "gap": pool["gap"],
    }


def atom_spec(a: Atom) -> dict[str, Any]:
    """Serializable portable atom definition (no reject mask)."""
    return {
        "atom_id": a.atom_id,
        "signal": a.signal,
        "role": a.role,
        "kind": a.kind,
        "description": a.description,
        "thr": a.thr,
        "thr_hi": a.thr_hi,
        "op": a.op,
        "quantile": a.quantile,
        "complexity": a.complexity,
    }


def apply_atom_spec(spec: dict[str, Any], signals: dict[str, np.ndarray]) -> np.ndarray:
    """Apply frozen thr definition to any row set."""
    x = np.asarray(signals[spec["signal"]], dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    op = spec.get("op", ">")
    if op == "in_range":
        lo = float(spec["thr"])
        hi = float(spec["thr_hi"])
        return (x >= lo) & (x <= hi)
    thr = float(spec["thr"])
    if op == ">":
        return x > thr
    if op == ">=":
        return x >= thr
    if op == "<":
        return x < thr
    raise ValueError(f"unknown op {op}")


def apply_clause_specs(
    clause_atom_ids: list[str],
    atom_specs: dict[str, dict[str, Any]],
    signals: dict[str, np.ndarray],
) -> np.ndarray:
    """AND of atoms in a clause."""
    masks = [apply_atom_spec(atom_specs[aid], signals) for aid in clause_atom_ids]
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out


def apply_policy_or(
    clause_list: list[list[str]],
    atom_specs: dict[str, dict[str, Any]],
    signals: dict[str, np.ndarray],
) -> np.ndarray:
    """OR of AND-clauses."""
    n = next(iter(signals.values())).shape[0]
    union = np.zeros(n, dtype=bool)
    for atom_ids in clause_list:
        union |= apply_clause_specs(atom_ids, atom_specs, signals)
    return union


def slice_pool(pool: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean-index all equal-length array fields in pool."""
    out: dict[str, np.ndarray] = {}
    n = pool["gt_match"].shape[0]
    for k, v in pool.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


# ── Layer 1: atom generation ────────────────────────────────────────────────


@dataclass(frozen=True)
class AtomRepairConfig:
    """Constraints that drop / tighten risky atom families (LOO-driven).

    Defaults preserve legacy behavior (gap bins + zone_q50/70).
    """

    ban_gap_bins: bool = False
    min_zone_q: float = 0.0  # e.g. 0.70 drops zone_q50
    ban_zone: bool = False
    require_support_in_and: bool = False  # AND size≥2 must include support role
    ban_signals: tuple[str, ...] = ()


DEFAULT_REPAIR = AtomRepairConfig()


def generate_atoms(
    signals: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    tail_qs: tuple[float, ...] = (0.85, 0.90, 0.95, 0.99),
    hard_qs: tuple[float, ...] = (0.50, 0.70),
    repair: AtomRepairConfig = DEFAULT_REPAIR,
) -> list[Atom]:
    """Interpretable reject atoms only (no free thr grid explosion)."""
    atoms: list[Atom] = []
    ban_sig = set(repair.ban_signals)
    # continuous energy atoms (higher = more reject-like)
    energy_sigs = [
        "score_m_bridge",
        "bridge_dist",
        "dist_h",
        "resid_mean",
        "abs_log_h",
        "abs_ratio_m1",
        "neg_dir_cos",
        "speed_mismatch",
    ]
    for name in energy_sigs:
        if name not in signals or name in ban_sig:
            continue
        role = ROLE_MAP.get(name, "support")
        if role == "diagnostic" and name == "speed_mismatch":
            # still allow one weak tail atom
            pass
        x = np.asarray(signals[name], dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        for q in tail_qs:
            thr = float(np.quantile(x, q))
            rej = x > thr
            atoms.append(
                Atom(
                    atom_id=f"{name}:tail_q{int(q * 100)}",
                    signal=name,
                    role=role,
                    kind="tail_q",
                    description=f"{name} > q{q:.2f} ({thr:.4g})",
                    reject=rej,
                    complexity=1,
                    thr=thr,
                    op=">",
                    quantile=q,
                )
            )
        # hard-zone (condition): lower quantile of energy still "in zone"
        # reject candidates that are ABOVE mid energy — operation-zone body+tail
        if role == "condition" and not repair.ban_zone:
            for q in hard_qs:
                if q + 1e-12 < repair.min_zone_q:
                    continue
                thr = float(np.quantile(x, q))
                rej = x > thr
                atoms.append(
                    Atom(
                        atom_id=f"{name}:zone_q{int(q * 100)}",
                        signal=name,
                        role="condition",
                        kind="hard_zone",
                        description=f"{name} in hard-zone > q{q:.2f} ({thr:.4g})",
                        reject=rej,
                        complexity=1,
                        thr=thr,
                        op=">",
                        quantile=q,
                    )
                )

    # gap condition bins (reject = True for candidates in that gap bin only
    # used as AND condition, not standalone safe reject)
    if not repair.ban_gap_bins and "gap" not in ban_sig:
        gap = signals["gap"]
        for lo, hi, tag in (
            (1, 10, "gap_1_10"),
            (11, 30, "gap_11_30"),
            (31, 60, "gap_31_60"),
            (61, 150, "gap_61_150"),
            (151, 300, "gap_151_300"),
        ):
            # "in bin" as condition: for AND we want reject only if ALSO other evidence
            # Represent as: atom is true when gap in bin (pattern presence)
            # For reject mask: we use "gap in bin" as a filter that *enables* reject
            # when AND with another atom — so the atom reject = (gap in bin) is
            # "suspect zone by gap" alone would hurt many GT — role=condition
            rej = (gap >= lo) & (gap <= hi)
            atoms.append(
                Atom(
                    atom_id=f"gap:bin_{tag}",
                    signal="gap",
                    role="condition",
                    kind="gap_bin",
                    description=f"gap in [{lo},{hi}]",
                    reject=rej,
                    complexity=1,
                    thr=float(lo),
                    thr_hi=float(hi),
                    op="in_range",
                )
            )

    # drop empty / near-empty / all-reject atoms
    pruned: list[Atom] = []
    n = y.size
    for a in atoms:
        k = int(a.reject.sum())
        if k < 20 or k > 0.98 * n:
            continue
        pruned.append(a)
    return pruned


# ── Pareto dominance ────────────────────────────────────────────────────────


def dominates(a: RuleMetrics, b: RuleMetrics, *, eps: float) -> bool:
    """A dominates B if A is better/equal on all objectives and strict on one.

    Objectives (higher better): FP_removed, -GT_hurt, -seq_hurt_std, -complexity
    Only compare among rules that both satisfy GT_hurt<=eps OR both fail.
    """
    # Prefer feasible rules; a feasible never dominated by infeasible for keep set
    a_ok = a.GT_hurt_rate <= eps + 1e-15
    b_ok = b.GT_hurt_rate <= eps + 1e-15
    if a_ok and not b_ok:
        return True
    if b_ok and not a_ok:
        return False
    better_or_eq = (
        a.FP_removed >= b.FP_removed
        and a.GT_hurt <= b.GT_hurt
        and a.seq_hurt_std <= b.seq_hurt_std + 1e-12
        and a.complexity <= b.complexity
    )
    strict = (
        a.FP_removed > b.FP_removed
        or a.GT_hurt < b.GT_hurt
        or a.seq_hurt_std < b.seq_hurt_std - 1e-12
        or a.complexity < b.complexity
    )
    return better_or_eq and strict


def pareto_front(
    items: list[tuple[str, RuleMetrics]],
    *,
    eps: float,
) -> list[str]:
    """Return ids of non-dominated items."""
    keep = []
    for i, (id_i, m_i) in enumerate(items):
        dominated = False
        for j, (id_j, m_j) in enumerate(items):
            if i == j:
                continue
            if dominates(m_j, m_i, eps=eps):
                dominated = True
                break
        if not dominated:
            keep.append(id_i)
    return keep


# ── Layer 2: conjunction mining ─────────────────────────────────────────────


def mine_conjunctions(
    atoms: list[Atom],
    y: np.ndarray,
    seq: np.ndarray,
    *,
    eps: float,
    max_and_size: int = 3,
    min_fp_support: int = 50,
    max_clauses: int = 200,
    repair: AtomRepairConfig = DEFAULT_REPAIR,
) -> list[Clause]:
    """Grow AND clauses with monotone pruning.

    Prune if:
      * FP support too small (children only smaller)
      * complexity budget
    Prefer mixing condition + support roles.
    """
    atom_by_id = {a.atom_id: a for a in atoms}

    def _has_support(ids: tuple[str, ...]) -> bool:
        return any(atom_by_id[i].role == "support" for i in ids if i in atom_by_id)

    # evaluate singles first
    singles: list[Clause] = []
    for a in atoms:
        m = _metrics(y, a.reject, seq, a.complexity)
        # skip pure gap_bin as standalone clause (condition only)
        if a.kind == "gap_bin":
            continue
        if m.FP_removed < min_fp_support and m.GT_hurt_rate > eps:
            continue
        c = Clause(
            clause_id=a.atom_id,
            atom_ids=(a.atom_id,),
            reject=a.reject.copy(),
            complexity=a.complexity,
            metrics=asdict(m),
        )
        singles.append(c)

    clauses: list[Clause] = list(singles)
    # level-wise growth
    frontier = [c for c in singles if c.metrics["FP_removed"] >= min_fp_support]
    for size in range(2, max_and_size + 1):
        next_frontier: list[Clause] = []
        # only extend with atoms that add condition or support diversity
        for base in frontier:
            base_sigs = {atom_by_id[i].signal for i in base.atom_ids}
            for a in atoms:
                if a.atom_id in base.atom_ids:
                    continue
                if a.signal in base_sigs:
                    continue  # one atom per signal
                # encourage condition ∩ support
                if size == 2:
                    # prefer different roles when possible
                    pass
                new_ids = tuple(sorted(base.atom_ids + (a.atom_id,)))
                if repair.require_support_in_and and not _has_support(new_ids):
                    continue  # ban pure condition∧condition (e.g. zone∧zone, gap∧zone)
                cid = " AND ".join(new_ids)
                if any(c.clause_id == cid for c in clauses):
                    continue
                rej = base.reject & a.reject
                fp_sup = int((~y & rej).sum())
                if fp_sup < min_fp_support:
                    continue  # children can't recover FP
                m = _metrics(y, rej, seq, base.complexity + a.complexity)
                # upper bound: if FP small relative to size, skip
                if m.FP_removed < min_fp_support:
                    continue
                cl = Clause(
                    clause_id=cid,
                    atom_ids=new_ids,
                    reject=rej,
                    complexity=base.complexity + a.complexity,
                    metrics=asdict(m),
                )
                clauses.append(cl)
                # only grow further if still enough FP
                if m.FP_removed >= min_fp_support:
                    next_frontier.append(cl)
        # prune frontier to Pareto among new + keep diversity
        if len(next_frontier) > 80:
            items = []
            for c in next_frontier:
                mm = c.metrics
                items.append(
                    (
                        c.clause_id,
                        RuleMetrics(
                            FP_removed=mm["FP_removed"],
                            GT_hurt=mm["GT_hurt"],
                            FP_removed_rate=mm["FP_removed_rate"],
                            GT_hurt_rate=mm["GT_hurt_rate"],
                            n_pos=mm["n_pos"],
                            n_neg=mm["n_neg"],
                            seq_hurt_std=mm["seq_hurt_std"],
                            boundary_mass=mm["boundary_mass"],
                            complexity=c.complexity,
                            safe_level=mm["safe_level"],
                        ),
                    )
                )
            keep_ids = set(pareto_front(items, eps=eps))
            # also keep top-K by FP among feasible
            feas = [
                c for c in next_frontier if c.metrics["GT_hurt_rate"] <= eps + 1e-15
            ]
            feas.sort(key=lambda c: -c.metrics["FP_removed"])
            for c in feas[:30]:
                keep_ids.add(c.clause_id)
            next_frontier = [c for c in next_frontier if c.clause_id in keep_ids]
        frontier = next_frontier
        if not frontier:
            break

    # global Pareto on all clauses (optional soft keep of high FP unsafe for OR later? no)
    items = []
    for c in clauses:
        mm = c.metrics
        items.append(
            (
                c.clause_id,
                RuleMetrics(
                    FP_removed=mm["FP_removed"],
                    GT_hurt=mm["GT_hurt"],
                    FP_removed_rate=mm["FP_removed_rate"],
                    GT_hurt_rate=mm["GT_hurt_rate"],
                    n_pos=mm["n_pos"],
                    n_neg=mm["n_neg"],
                    seq_hurt_std=mm["seq_hurt_std"],
                    boundary_mass=mm["boundary_mass"],
                    complexity=c.complexity,
                    safe_level=mm["safe_level"],
                ),
            )
        )
    front_ids = set(pareto_front(items, eps=eps))
    # also keep all ε-feasible sorted by FP
    feas_ids = {
        c.clause_id for c in clauses if c.metrics["GT_hurt_rate"] <= eps + 1e-15
    }
    selected = [
        c for c in clauses if c.clause_id in front_ids or c.clause_id in feas_ids
    ]
    # cap
    selected.sort(
        key=lambda c: (
            0 if c.metrics["GT_hurt_rate"] <= eps + 1e-15 else 1,
            -c.metrics["FP_removed"],
            c.complexity,
        )
    )
    return selected[:max_clauses]


# ── Layer 3: submodular greedy OR ───────────────────────────────────────────


def greedy_or_select(
    clauses: list[Clause],
    y: np.ndarray,
    seq: np.ndarray,
    *,
    eps: float,
    max_or_rules: int = 5,
    tau_seq_std: float = 0.05,
    lambda_hurt: float = 1000.0,
    mu_complexity: float = 1.0,
) -> dict[str, Any]:
    """Select OR of clauses maximizing FP coverage under constraints.

    Greedy step maximizes:
      score = ΔFP - λ * ΔGT_hurt_if_breaks - μ * complexity
    Only accept steps that keep total GT_hurt_rate <= eps.
    """
    # candidate pool: prefer ε-feasible singles/clauses; also allow near-feasible
    pool = [
        c
        for c in clauses
        if c.metrics["GT_hurt_rate"] <= max(eps, 0.01) + 1e-15
        or c.metrics["FP_removed"] >= 500
    ]
    # de-dup identical masks
    seen = set()
    uniq: list[Clause] = []
    for c in pool:
        key = c.reject.tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    pool = uniq

    selected: list[Clause] = []
    union = np.zeros(y.shape, dtype=bool)
    history = []

    for step in range(max_or_rules):
        best_c = None
        best_score = -1e18
        best_union = None
        best_m = None
        for c in pool:
            if any(c.clause_id == s.clause_id for s in selected):
                continue
            new_u = union | c.reject
            m = _metrics(
                y, new_u, seq, sum(s.complexity for s in selected) + c.complexity
            )
            if m.GT_hurt_rate > eps + 1e-15:
                continue
            if m.seq_hurt_std > tau_seq_std + 1e-12 and eps <= 0.0:
                # soft: allow if still better FP a lot
                pass
            prev_fp = int((~y & union).sum())
            delta_fp = m.FP_removed - prev_fp
            if delta_fp <= 0:
                continue
            prev_hurt = int((y & union).sum())
            delta_hurt = m.GT_hurt - prev_hurt
            score = (
                delta_fp
                - lambda_hurt * max(delta_hurt, 0)
                - mu_complexity * c.complexity
            )
            # prefer lower seq std
            score -= 50.0 * m.seq_hurt_std * m.n_pos
            # require positive score: real FP gain after penalties
            if score > best_score and score > 0:
                best_score = score
                best_c = c
                best_union = new_u
                best_m = m
        if best_c is None:
            break
        selected.append(best_c)
        union = best_union
        history.append(
            {
                "step": step + 1,
                "added": best_c.clause_id,
                "delta_score": best_score,
                "metrics": asdict(best_m),
            }
        )

    final = _metrics(
        y,
        union,
        seq,
        sum(c.complexity for c in selected) if selected else 0,
    )
    return {
        "selected_clauses": [c.clause_id for c in selected],
        "selected_atoms": sorted({a for c in selected for a in c.atom_ids}),
        "n_rules": len(selected),
        "policy_or": " OR ".join(f"({c.clause_id})" for c in selected)
        if selected
        else "(empty)",
        "final_metrics": asdict(final),
        "history": history,
        "n_candidates_considered": len(pool),
    }


def run_search(
    pool: dict[str, np.ndarray],
    *,
    eps: float = 0.0,
    max_and_size: int = 3,
    max_or_rules: int = 5,
    min_fp_support: int = 80,
    tau_seq_std: float = 0.05,
    repair: AtomRepairConfig = DEFAULT_REPAIR,
) -> dict[str, Any]:
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    signals = extract_signals(pool)

    atoms = generate_atoms(signals, y, repair=repair)
    # metrics for all atoms
    atom_rows = []
    atom_items = []
    for a in atoms:
        m = _metrics(y, a.reject, seq, a.complexity)
        atom_rows.append(
            {
                "atom_id": a.atom_id,
                "signal": a.signal,
                "role": a.role,
                "kind": a.kind,
                "description": a.description,
                **asdict(m),
            }
        )
        atom_items.append((a.atom_id, m))

    atom_front = set(pareto_front(atom_items, eps=eps))
    # keep atoms on front OR ε-feasible OR condition atoms for mining
    atoms_for_mine = [
        a
        for a in atoms
        if a.atom_id in atom_front
        or a.role == "condition"
        or next(r for r in atom_rows if r["atom_id"] == a.atom_id)["GT_hurt_rate"]
        <= eps + 1e-15
    ]

    clauses = mine_conjunctions(
        atoms_for_mine,
        y,
        seq,
        eps=eps,
        max_and_size=max_and_size,
        min_fp_support=min_fp_support,
        repair=repair,
    )
    clause_rows = []
    for c in clauses:
        clause_rows.append(
            {
                "clause_id": c.clause_id,
                "n_atoms": len(c.atom_ids),
                "atom_ids": " | ".join(c.atom_ids),
                **c.metrics,
            }
        )

    # Pareto on clauses
    c_items = []
    for c in clauses:
        mm = c.metrics
        c_items.append(
            (
                c.clause_id,
                RuleMetrics(
                    FP_removed=mm["FP_removed"],
                    GT_hurt=mm["GT_hurt"],
                    FP_removed_rate=mm["FP_removed_rate"],
                    GT_hurt_rate=mm["GT_hurt_rate"],
                    n_pos=mm["n_pos"],
                    n_neg=mm["n_neg"],
                    seq_hurt_std=mm["seq_hurt_std"],
                    boundary_mass=mm["boundary_mass"],
                    complexity=c.complexity,
                    safe_level=mm["safe_level"],
                ),
            )
        )
    clause_front = set(pareto_front(c_items, eps=eps)) if c_items else set()

    policy = greedy_or_select(
        clauses,
        y,
        seq,
        eps=eps,
        max_or_rules=max_or_rules,
        tau_seq_std=tau_seq_std,
    )

    # portable freeze: thr fitted on this search split
    atom_by_id = {a.atom_id: a for a in atoms}
    selected_clause_atom_lists: list[list[str]] = []
    for cid in policy["selected_clauses"]:
        cl = next((c for c in clauses if c.clause_id == cid), None)
        if cl is None:
            selected_clause_atom_lists.append([p.strip() for p in cid.split(" AND ")])
        else:
            selected_clause_atom_lists.append(list(cl.atom_ids))
    needed_atoms = sorted({aid for cl in selected_clause_atom_lists for aid in cl})
    portable_atoms = {
        aid: atom_spec(atom_by_id[aid]) for aid in needed_atoms if aid in atom_by_id
    }
    policy["portable"] = {
        "clauses": selected_clause_atom_lists,
        "atom_specs": portable_atoms,
        "eps": eps,
    }

    # baselines: best single atom under eps
    feas_atoms = [r for r in atom_rows if r["GT_hurt_rate"] <= eps + 1e-15]
    best_atom = max(feas_atoms, key=lambda r: r["FP_removed"]) if feas_atoms else None
    feas_clauses = [r for r in clause_rows if r["GT_hurt_rate"] <= eps + 1e-15]
    best_clause = (
        max(feas_clauses, key=lambda r: r["FP_removed"]) if feas_clauses else None
    )

    # per-seq breakdown (in-sample) for freeze card
    union = np.zeros(y.shape, dtype=bool)
    for c in clauses:
        if c.clause_id in set(policy["selected_clauses"]):
            union |= c.reject
    per_seq = []
    for s in sorted(set(seq.tolist())):
        m = seq == s
        ys = y[m]
        if ys.size == 0:
            continue
        rs = union[m]
        n_pos = int(ys.sum())
        n_neg = int((~ys).sum())
        per_seq.append(
            {
                "seq": str(s),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "FP_removed": int((~ys & rs).sum()),
                "GT_hurt": int((ys & rs).sum()),
                "GT_hurt_rate": float((ys & rs).sum() / n_pos) if n_pos else 0.0,
                "FP_removed_rate": float((~ys & rs).sum() / n_neg) if n_neg else 0.0,
            }
        )

    return {
        "role_map": ROLE_MAP,
        "repair": asdict(repair),
        "n_atoms": len(atoms),
        "n_atoms_for_mine": len(atoms_for_mine),
        "n_atom_pareto": len(atom_front),
        "n_clauses": len(clauses),
        "n_clause_pareto": len(clause_front),
        "atom_pareto_ids": sorted(atom_front),
        "clause_pareto_ids": sorted(clause_front),
        "best_single_atom": best_atom,
        "best_single_clause": best_clause,
        "policy": policy,
        "per_seq": per_seq,
        "gain_vs_best_atom_FP": (
            policy["final_metrics"]["FP_removed"] - best_atom["FP_removed"]
            if best_atom
            else None
        ),
        "gain_vs_best_clause_FP": (
            policy["final_metrics"]["FP_removed"] - best_clause["FP_removed"]
            if best_clause
            else None
        ),
        "_atom_rows": atom_rows,
        "_clause_rows": clause_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--eps-grid", default="0.0,0.001,0.01", help="comma list")
    ap.add_argument("--max-and-size", type=int, default=3)
    ap.add_argument("--max-or-rules", type=int, default=5)
    ap.add_argument("--min-fp-support", type=int, default=80)
    ap.add_argument("--tau-seq-std", type=float, default=0.05)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_gate_rule_search_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)

    eps_list = [float(x) for x in args.eps_grid.split(",") if x.strip()]
    batch: dict[str, Any] = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "architecture": [
            "L0 roles + transform (external)",
            "L1 atoms",
            "L2 conjunction mining + Pareto",
            "L3 submodular greedy OR",
        ],
        "math_tools": [
            "Pareto dominance pruning",
            "ε-constrained opt",
            "monotone AND support prune",
            "submodular greedy coverage",
        ],
        "by_eps": {},
    }

    print(f"STUDY={study}")
    for eps in eps_list:
        print(f"\n=== ε={eps} ===")
        res = run_search(
            pool,
            eps=eps,
            max_and_size=args.max_and_size,
            max_or_rules=args.max_or_rules,
            min_fp_support=args.min_fp_support,
            tau_seq_std=args.tau_seq_std,
        )
        sub = study / f"eps_{str(eps).replace('.', 'p')}"
        sub.mkdir(exist_ok=True)
        write_csv(sub / "atoms.csv", res.pop("_atom_rows"))
        write_csv(sub / "clauses.csv", res.pop("_clause_rows"))
        (sub / "summary.json").write_text(
            json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8"
        )
        batch["by_eps"][str(eps)] = {
            k: res[k]
            for k in res
            if k
            in (
                "n_atoms",
                "n_atoms_for_mine",
                "n_atom_pareto",
                "n_clauses",
                "n_clause_pareto",
                "best_single_atom",
                "best_single_clause",
                "policy",
                "gain_vs_best_atom_FP",
                "gain_vs_best_clause_FP",
            )
        }
        pol = res["policy"]
        fm = pol["final_metrics"]
        print(
            f"  atoms={res['n_atoms']} mine={res['n_atoms_for_mine']} pareto_a={res['n_atom_pareto']}"
        )
        print(f"  clauses={res['n_clauses']} pareto_c={res['n_clause_pareto']}")
        if res["best_single_atom"]:
            ba = res["best_single_atom"]
            print(
                f"  best atom: {ba['atom_id']}  FPrm={ba['FP_removed']}  "
                f"hurt={ba['GT_hurt']}  seq_std={ba['seq_hurt_std']:.4f}"
            )
        if res["best_single_clause"]:
            bc = res["best_single_clause"]
            print(
                f"  best clause: {bc['clause_id'][:80]}  FPrm={bc['FP_removed']}  "
                f"hurt={bc['GT_hurt']}"
            )
        print(
            f"  policy OR n={pol['n_rules']}  FPrm={fm['FP_removed']}  "
            f"hurt={fm['GT_hurt']} ({100 * fm['GT_hurt_rate']:.2f}%)  "
            f"seq_std={fm['seq_hurt_std']:.4f}"
        )
        print(
            f"  gain vs atom: {res['gain_vs_best_atom_FP']}  "
            f"vs clause: {res['gain_vs_best_clause_FP']}"
        )
        for h in pol["history"]:
            print(
                f"    + {h['added'][:70]}  Δscore={h['delta_score']:.1f}  "
                f"FP={h['metrics']['FP_removed']}"
            )
        print(f"  POLICY: {pol['policy_or'][:200]}")

    (study / "summary.json").write_text(
        json.dumps(batch, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {study / 'summary.json'}")


if __name__ == "__main__":
    main()
