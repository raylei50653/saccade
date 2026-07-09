#!/usr/bin/env python3
"""LOO GT-hurt attribution → atom classification → repair LOO compare.

Answers: which heldout / clause / atom / thr interval causes LOO leak?

Protocol
--------
For each held-out sequence S (same as gate_rule_search_loo):
  1. Search policy on train-6 with optional AtomRepairConfig
  2. Apply portable thr to held-out
  3. If test GT_hurt > 0, decompose:
       - per-clause hurt / FP
       - per-atom fire on hurt GT rows
       - leave-one-atom-out of failing clause
       - leave-one-clause-out of policy
       - atom value quantile of hurt GT vs train thr
       - distance_to_safe_boundary in score units
  4. Classify atoms across folds
  5. Optionally re-LOO under repair configs and report retained FP

  uv run python scripts/tools/loo_hurt_attribution.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_loo_attr_<stamp> \\
    --eps 0.0 --jobs 7 --run-repairs
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_tools = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


grs = _load("gate_rule_search", _tools / "gate_rule_search.py")
_audit = _load("audit_relink_safe_reject", _tools / "audit_relink_safe_reject.py")

# ── worker state ─────────────────────────────────────────────────────────────
_POOL: dict[str, np.ndarray] | None = None


def _mp_init(pool: dict[str, np.ndarray]) -> None:
    global _POOL
    _POOL = pool


def _repair_from_dict(d: dict[str, Any] | None) -> grs.AtomRepairConfig:
    if not d:
        return grs.AtomRepairConfig()
    return grs.AtomRepairConfig(
        ban_gap_bins=bool(d.get("ban_gap_bins", False)),
        min_zone_q=float(d.get("min_zone_q", 0.0)),
        ban_zone=bool(d.get("ban_zone", False)),
        require_support_in_and=bool(d.get("require_support_in_and", False)),
        ban_signals=tuple(d.get("ban_signals") or ()),
    )


def _atom_value_and_boundary(
    spec: dict[str, Any],
    x: float,
    train_pos: np.ndarray | None,
) -> dict[str, Any]:
    """Quantile of x under train positives + distance past thr."""
    op = spec.get("op", ">")
    thr = spec.get("thr")
    thr_hi = spec.get("thr_hi")
    out: dict[str, Any] = {
        "atom_value": float(x),
        "thr": thr,
        "thr_hi": thr_hi,
        "op": op,
    }
    if train_pos is not None and train_pos.size > 0:
        # empirical CDF rank of x among train GT
        sp = np.sort(train_pos)
        rank = float(np.searchsorted(sp, x, side="right") / len(sp))
        out["atom_value_quantile_train_GT"] = rank
        med = float(np.median(sp))
        std = float(np.std(sp)) or 1.0
        out["train_GT_median"] = med
        out["train_GT_std"] = std
    else:
        out["atom_value_quantile_train_GT"] = None
        std = 1.0

    if op == "in_range" and thr is not None and thr_hi is not None:
        lo, hi = float(thr), float(thr_hi)
        # inside bin distance to nearest edge (0 = on edge)
        if lo <= x <= hi:
            dist = min(x - lo, hi - x)
            out["distance_to_safe_boundary"] = float(dist)
            out["boundary_side"] = "inside_bin"
        else:
            out["distance_to_safe_boundary"] = float(min(abs(x - lo), abs(x - hi)))
            out["boundary_side"] = "outside_bin"
    elif thr is not None and op in (">", ">="):
        # how far above thr (positive = past thr into reject)
        margin = float(x) - float(thr)
        out["distance_to_safe_boundary"] = margin / std
        out["boundary_side"] = "above_thr" if margin > 0 else "below_thr"
        # tight if just barely over thr
        out["boundary_touching"] = bool(0 < margin / std < 0.25)
    else:
        out["distance_to_safe_boundary"] = None
        out["boundary_side"] = None
    return out


def attribute_fold(
    pool: dict[str, np.ndarray],
    held: str,
    *,
    eps: float,
    max_and_size: int,
    max_or_rules: int,
    min_fp_support: int,
    repair: grs.AtomRepairConfig,
) -> dict[str, Any]:
    seq = pool["seq"]
    train_m = seq != held
    test_m = seq == held
    train_pool = grs.slice_pool(pool, train_m)
    test_pool = grs.slice_pool(pool, test_m)

    train_res = grs.run_search(
        train_pool,
        eps=eps,
        max_and_size=max_and_size,
        max_or_rules=max_or_rules,
        min_fp_support=min_fp_support,
        repair=repair,
    )
    pol = train_res["policy"]
    portable = pol.get("portable") or {}
    clauses: list[list[str]] = portable.get("clauses") or []
    atom_specs: dict[str, dict[str, Any]] = portable.get("atom_specs") or {}

    y_tr = train_pool["gt_match"].astype(bool)
    y_te = test_pool["gt_match"].astype(bool)
    sig_tr = grs.extract_signals(train_pool)
    sig_te = grs.extract_signals(test_pool)

    if clauses and atom_specs:
        rej = grs.apply_policy_or(clauses, atom_specs, sig_te)
    else:
        rej = np.zeros(y_te.shape, dtype=bool)

    n_pos = int(y_te.sum())
    n_neg = int((~y_te).sum())
    test_hurt = int((y_te & rej).sum())
    test_fp = int((~y_te & rej).sum())
    train_fp = int(pol["final_metrics"]["FP_removed"])
    train_hurt = int(pol["final_metrics"]["GT_hurt"])

    # train signal positives for quantile
    train_pos_by_sig: dict[str, np.ndarray] = {}
    for sname, arr in sig_tr.items():
        train_pos_by_sig[sname] = np.asarray(arr, dtype=float)[y_tr]

    # clause-level
    clause_rows: list[dict[str, Any]] = []
    failed_clauses: list[str] = []
    for atom_ids in clauses:
        cm = grs.apply_clause_specs(atom_ids, atom_specs, sig_te)
        h = int((y_te & cm).sum())
        f = int((~y_te & cm).sum())
        cid = " AND ".join(atom_ids)
        row = {
            "heldout_seq": held,
            "epsilon": eps,
            "clause_id": cid,
            "atom_ids": list(atom_ids),
            "GT_hurt": h,
            "FP_removed": f,
            "n_atoms": len(atom_ids),
            "roles": [atom_specs[a].get("role") for a in atom_ids if a in atom_specs],
            "kinds": [atom_specs[a].get("kind") for a in atom_ids if a in atom_specs],
        }
        # leave-one-atom-out of this clause
        loo_atoms = []
        if len(atom_ids) >= 2 and h > 0:
            for drop in atom_ids:
                keep = [a for a in atom_ids if a != drop]
                if not keep:
                    continue
                cm2 = grs.apply_clause_specs(keep, atom_specs, sig_te)
                h2 = int((y_te & cm2).sum())
                f2 = int((~y_te & cm2).sum())
                loo_atoms.append(
                    {
                        "dropped_atom": drop,
                        "GT_hurt_after_drop": h2,
                        "FP_after_drop": f2,
                        "hurt_eliminated": h2 == 0 and h > 0,
                    }
                )
        row["leave_one_atom_out"] = loo_atoms
        clause_rows.append(row)
        if h > 0:
            failed_clauses.append(cid)

    # leave-one-clause-out of full policy
    policy_loo: list[dict[str, Any]] = []
    for i, atom_ids in enumerate(clauses):
        rest = [c for j, c in enumerate(clauses) if j != i]
        if rest:
            rej2 = grs.apply_policy_or(rest, atom_specs, sig_te)
        else:
            rej2 = np.zeros(y_te.shape, dtype=bool)
        policy_loo.append(
            {
                "dropped_clause": " AND ".join(atom_ids),
                "GT_hurt_after_drop": int((y_te & rej2).sum()),
                "FP_after_drop": int((~y_te & rej2).sum()),
                "hurt_eliminated": int((y_te & rej2).sum()) == 0 and test_hurt > 0,
            }
        )

    # atom-level rows (all atoms in policy + fire on hurt GT)
    atom_rows: list[dict[str, Any]] = []
    hurt_idx = np.where(y_te & rej)[0]
    for aid, spec in atom_specs.items():
        am = grs.apply_atom_spec(spec, sig_te)
        a_hurt = int((y_te & am).sum())
        a_fp = int((~y_te & am).sum())
        # how many policy-hurt GT also fire this atom
        hurt_and_atom = 0
        hurt_cands: list[dict[str, Any]] = []
        sig_name = spec["signal"]
        x_all = np.asarray(sig_te[sig_name], dtype=float)
        for j in hurt_idx:
            fires = bool(am[j])
            if fires:
                hurt_and_atom += 1
            detail = _atom_value_and_boundary(
                spec, float(x_all[j]), train_pos_by_sig.get(sig_name)
            )
            detail["fires"] = fires
            detail["row_i"] = int(j)
            hurt_cands.append(detail)

        # clauses containing this atom
        in_clauses = [" AND ".join(c) for c in clauses if aid in c]
        in_failed = [c for c in in_clauses if c in failed_clauses]

        # aggregate boundary for hurt candidates that fire
        fire_hurt = [h for h in hurt_cands if h.get("fires")]
        med_q = (
            float(
                np.median(
                    [
                        h["atom_value_quantile_train_GT"]
                        for h in fire_hurt
                        if h.get("atom_value_quantile_train_GT") is not None
                    ]
                )
            )
            if fire_hurt
            else None
        )
        med_bdist = (
            float(
                np.median(
                    [
                        h["distance_to_safe_boundary"]
                        for h in fire_hurt
                        if h.get("distance_to_safe_boundary") is not None
                    ]
                )
            )
            if fire_hurt
            else None
        )
        any_touch = any(h.get("boundary_touching") for h in fire_hurt)

        atom_rows.append(
            {
                "heldout_seq": held,
                "epsilon": eps,
                "atom_id": aid,
                "signal": sig_name,
                "role": spec.get("role"),
                "kind": spec.get("kind"),
                "thr": spec.get("thr"),
                "quantile": spec.get("quantile"),
                "op": spec.get("op"),
                "GT_hurt_alone": a_hurt,
                "FP_removed_alone": a_fp,
                "in_policy_clauses": in_clauses,
                "in_failed_clauses": in_failed,
                "n_policy_hurt_GT": int(len(hurt_idx)),
                "n_policy_hurt_and_atom_fires": hurt_and_atom,
                "atom_value_quantile_median_hurt": med_q,
                "distance_to_safe_boundary_median": med_bdist,
                "boundary_touching_any": any_touch,
                "hurt_candidates": hurt_cands,
            }
        )

    return {
        "heldout_seq": held,
        "epsilon": eps,
        "repair": asdict(repair),
        "policy_size": len(clauses),
        "train_FP_removed": train_fp,
        "train_GT_hurt": train_hurt,
        "test_n_pos": n_pos,
        "test_n_neg": n_neg,
        "test_FP_removed": test_fp,
        "test_GT_hurt": test_hurt,
        "test_GT_hurt_rate": test_hurt / n_pos if n_pos else 0.0,
        "test_FP_removed_rate": test_fp / n_neg if n_neg else 0.0,
        "failed_clauses": failed_clauses,
        "selected_clauses": [" AND ".join(c) for c in clauses],
        "portable": portable,
        "clause_attr": clause_rows,
        "policy_leave_one_clause": policy_loo,
        "atom_attr": atom_rows,
        "classification": (
            "pass_eps0_transfer"
            if test_hurt == 0 and test_fp > 0
            else (
                "gt_leak_with_capacity"
                if test_hurt > 0 and test_fp >= 50
                else ("gt_leak_low_capacity" if test_hurt > 0 else "no_capacity")
            )
        ),
    }


def _mp_fold(payload: tuple) -> dict[str, Any]:
    held, eps, max_and, max_or, min_fp, repair_d = payload
    assert _POOL is not None
    return attribute_fold(
        _POOL,
        held,
        eps=eps,
        max_and_size=max_and,
        max_or_rules=max_or,
        min_fp_support=min_fp,
        repair=_repair_from_dict(repair_d),
    )


def classify_atoms_global(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-fold atom labels."""
    # atom_id patterns (kind-level and exact)
    stats: dict[str, dict[str, Any]] = {}

    def _bump(key: str, **kw: Any) -> None:
        if key not in stats:
            stats[key] = {
                "atom_key": key,
                "n_selected_folds": 0,
                "n_failed_folds": 0,
                "sum_test_hurt_when_in_failed": 0,
                "sum_fp_alone": 0,
                "n_boundary_touch": 0,
                "kinds": set(),
                "signals": set(),
                "heldouts_failed": [],
            }
        s = stats[key]
        for k, v in kw.items():
            if k == "kinds":
                s["kinds"].add(v)
            elif k == "signals":
                s["signals"].add(v)
            elif k == "heldouts_failed":
                s["heldouts_failed"].append(v)
            elif k.startswith("add_"):
                s[k[4:]] = s.get(k[4:], 0) + v
            else:
                s[k] = s.get(k, 0) + v

    for f in folds:
        selected_atoms = set()
        for c in f.get("selected_clauses") or []:
            for a in c.split(" AND "):
                selected_atoms.add(a.strip())
        failed_atoms = set()
        for c in f.get("failed_clauses") or []:
            for a in c.split(" AND "):
                failed_atoms.add(a.strip())

        # map atom details
        by_id = {r["atom_id"]: r for r in f.get("atom_attr") or []}

        for aid in selected_atoms:
            r = by_id.get(aid, {})
            kind = r.get("kind") or aid.split(":")[-1].split("_")[0]
            sig = r.get("signal") or aid.split(":")[0]
            # exact id
            _bump(
                aid,
                n_selected_folds=1,
                add_sum_fp_alone=int(r.get("FP_removed_alone") or 0),
                kinds=kind,
                signals=sig,
            )
            if aid in failed_atoms:
                _bump(
                    aid,
                    n_failed_folds=1,
                    add_sum_test_hurt_when_in_failed=int(f.get("test_GT_hurt") or 0),
                    heldouts_failed=f["heldout_seq"],
                )
                if r.get("boundary_touching_any"):
                    _bump(aid, n_boundary_touch=1)
            # kind family
            fam = f"kind:{kind}"
            _bump(fam, n_selected_folds=1, kinds=kind, signals=sig)
            if aid in failed_atoms:
                _bump(
                    fam,
                    n_failed_folds=1,
                    add_sum_test_hurt_when_in_failed=int(f.get("test_GT_hurt") or 0),
                    heldouts_failed=f["heldout_seq"],
                )

    rows = []
    n_folds = max(len(folds), 1)
    for key, s in stats.items():
        n_sel = s["n_selected_folds"]
        n_fail = s["n_failed_folds"]
        mean_fp = s["sum_fp_alone"] / max(n_sel, 1)
        # classification
        if n_sel == 0:
            label = "dead_atom"
        elif n_fail == 0 and mean_fp >= 50:
            label = "stable_clean_atom"
        elif n_fail == 0 and mean_fp < 50:
            label = "dead_atom"
        elif n_fail >= 1 and n_fail == n_sel and n_sel == 1:
            label = "seq_specific_atom"
        elif n_fail >= 1 and s["n_boundary_touch"] >= 1:
            label = "boundary_touching_atom"
        elif n_fail >= 1 and mean_fp >= 200:
            label = "productive_but_risky_atom"
        elif n_fail >= 1:
            label = "productive_but_risky_atom"
        else:
            label = "stable_clean_atom"

        rows.append(
            {
                "atom_key": key,
                "label": label,
                "n_selected_folds": n_sel,
                "n_failed_folds": n_fail,
                "fail_rate_when_selected": n_fail / max(n_sel, 1),
                "sum_test_hurt_when_in_failed": s["sum_test_hurt_when_in_failed"],
                "mean_FP_alone": mean_fp,
                "n_boundary_touch": s["n_boundary_touch"],
                "kinds": ",".join(sorted(s["kinds"])),
                "signals": ",".join(sorted(s["signals"])),
                "heldouts_failed": ",".join(s["heldouts_failed"]),
                "n_folds": n_folds,
            }
        )
    rows.sort(
        key=lambda r: (
            -r["n_failed_folds"],
            -r["sum_test_hurt_when_in_failed"],
            -r["mean_FP_alone"],
        )
    )
    return rows


REPAIR_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "ban_gap": {"ban_gap_bins": True},
    "zone_q70_only": {"min_zone_q": 0.70},
    "ban_gap_zone70": {"ban_gap_bins": True, "min_zone_q": 0.70},
    "ban_gap_zone70_require_support": {
        "ban_gap_bins": True,
        "min_zone_q": 0.70,
        "require_support_in_and": True,
    },
    "ban_gap_ban_zone": {"ban_gap_bins": True, "ban_zone": True},
    "strict_tail_only": {
        "ban_gap_bins": True,
        "ban_zone": True,
        "require_support_in_and": True,
    },
}


def run_config_loo(
    pool: dict[str, np.ndarray],
    *,
    name: str,
    repair_d: dict[str, Any],
    eps: float,
    max_and_size: int,
    max_or_rules: int,
    min_fp_support: int,
    jobs: int,
) -> dict[str, Any]:
    seqs = sorted({str(s) for s in pool["seq"].tolist()})
    payloads = [
        (held, eps, max_and_size, max_or_rules, min_fp_support, repair_d)
        for held in seqs
    ]
    folds: list[dict[str, Any]] = []
    if jobs <= 1:
        for p in payloads:
            folds.append(
                _mp_fold(p)
                if _POOL is not None
                else attribute_fold(
                    pool,
                    p[0],
                    eps=p[1],
                    max_and_size=p[2],
                    max_or_rules=p[3],
                    min_fp_support=p[4],
                    repair=_repair_from_dict(p[5]),
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=jobs, initializer=_mp_init, initargs=(pool,)
        ) as ex:
            futs = {ex.submit(_mp_fold, p): p[0] for p in payloads}
            by_held = {}
            for fut in as_completed(futs):
                by_held[futs[fut]] = fut.result()
            folds = [by_held[s] for s in seqs]

    n = len(folds)
    n_gt0 = sum(1 for f in folds if f["test_GT_hurt"] == 0)
    sum_hurt = int(sum(f["test_GT_hurt"] for f in folds))
    mean_fp = float(np.mean([f["test_FP_removed"] for f in folds]))
    mean_tr_fp = float(np.mean([f["train_FP_removed"] for f in folds]))
    atom_labels = classify_atoms_global(folds)

    # productivity retained vs baseline filled later
    return {
        "config_name": name,
        "repair": repair_d,
        "n_folds": n,
        "n_test_GT_hurt_zero": n_gt0,
        "sum_test_GT_hurt": sum_hurt,
        "mean_test_FP_removed": mean_fp,
        "mean_train_FP_removed": mean_tr_fp,
        "mean_test_GT_hurt_rate": float(
            np.mean([f["test_GT_hurt_rate"] for f in folds])
        ),
        "verdict": (
            "loo_pass_eps0"
            if n_gt0 == n
            else ("loo_partial" if n_gt0 >= n // 2 else "loo_fail_gt_leak")
        ),
        "folds": folds,
        "atom_labels": atom_labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--max-and-size", type=int, default=3)
    ap.add_argument("--max-or-rules", type=int, default=5)
    ap.add_argument("--min-fp-support", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=0, help="0 = n_seqs")
    ap.add_argument(
        "--run-repairs",
        action="store_true",
        help="also LOO under repair presets and compare FP retained",
    )
    ap.add_argument(
        "--repair-only",
        default=None,
        help="comma list of preset names; default all when --run-repairs",
    )
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_loo_attr_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)
    n_seq = len({str(s) for s in pool["seq"].tolist()})
    jobs = args.jobs if args.jobs > 0 else min(n_seq, os.cpu_count() or 4)

    print(f"STUDY={study}")
    print(f"jobs={jobs} eps={args.eps}")

    # always run baseline attribution
    names = ["baseline"]
    if args.run_repairs:
        if args.repair_only:
            names = [x.strip() for x in args.repair_only.split(",") if x.strip()]
        else:
            names = list(REPAIR_PRESETS.keys())

    results: dict[str, Any] = {}
    for name in names:
        repair_d = REPAIR_PRESETS.get(name, {})
        print(f"\n=== config={name} repair={repair_d} ===", flush=True)
        # set worker pool for sequential path
        global _POOL
        _POOL = pool
        res = run_config_loo(
            pool,
            name=name,
            repair_d=repair_d,
            eps=args.eps,
            max_and_size=args.max_and_size,
            max_or_rules=args.max_or_rules,
            min_fp_support=args.min_fp_support,
            jobs=jobs,
        )
        results[name] = res
        print(
            f"  verdict={res['verdict']}  GT0={res['n_test_GT_hurt_zero']}/{res['n_folds']}  "
            f"sum_hurt={res['sum_test_GT_hurt']}  mean_te_FP={res['mean_test_FP_removed']:.1f}",
            flush=True,
        )
        for f in res["folds"]:
            print(
                f"    {f['heldout_seq']:<18} te_FP={f['test_FP_removed']:5d} "
                f"te_H={f['test_GT_hurt']}  {f['classification']:<22} "
                f"{';'.join(f['failed_clauses'])[:50]}",
                flush=True,
            )

    base = results["baseline"]
    base_fp = base["mean_test_FP_removed"] or 1.0

    # comparison table
    cmp_rows = []
    for name, res in results.items():
        cmp_rows.append(
            {
                "config": name,
                "verdict": res["verdict"],
                "n_GT0": res["n_test_GT_hurt_zero"],
                "n_folds": res["n_folds"],
                "sum_test_GT_hurt": res["sum_test_GT_hurt"],
                "mean_test_FP": res["mean_test_FP_removed"],
                "mean_train_FP": res["mean_train_FP_removed"],
                "FP_retained_vs_baseline": res["mean_test_FP_removed"] / base_fp,
                "hurt_reduction_vs_baseline": (
                    base["sum_test_GT_hurt"] - res["sum_test_GT_hurt"]
                ),
                "repair": json.dumps(res["repair"]),
            }
        )

    # flat attribution CSV from baseline
    attr_rows = []
    for f in base["folds"]:
        for c in f["clause_attr"]:
            attr_rows.append(
                {
                    "heldout_seq": f["heldout_seq"],
                    "method": "gate_rule_search_OR",
                    "epsilon": args.eps,
                    "clause_id": c["clause_id"],
                    "atom_id": "",
                    "GT_hurt": c["GT_hurt"],
                    "FP_removed": c["FP_removed"],
                    "level": "clause",
                    "roles": "|".join(c.get("roles") or []),
                    "kinds": "|".join(c.get("kinds") or []),
                    "leave_one_atom_out": json.dumps(c.get("leave_one_atom_out") or []),
                }
            )
        for a in f["atom_attr"]:
            attr_rows.append(
                {
                    "heldout_seq": f["heldout_seq"],
                    "method": "gate_rule_search_OR",
                    "epsilon": args.eps,
                    "clause_id": "|".join(a.get("in_failed_clauses") or []),
                    "atom_id": a["atom_id"],
                    "GT_hurt": a["GT_hurt_alone"],
                    "FP_removed": a["FP_removed_alone"],
                    "level": "atom",
                    "roles": a.get("role"),
                    "kinds": a.get("kind"),
                    "atom_value_quantile": a.get("atom_value_quantile_median_hurt"),
                    "distance_to_safe_boundary": a.get(
                        "distance_to_safe_boundary_median"
                    ),
                    "boundary_touching": a.get("boundary_touching_any"),
                    "n_policy_hurt_and_fires": a.get("n_policy_hurt_and_atom_fires"),
                    "leave_one_atom_out": "",
                }
            )

    def _write(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        # union keys
        cols: list[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    _write(study / "attribution_table.csv", attr_rows)
    _write(study / "atom_labels.csv", base["atom_labels"])
    _write(study / "repair_compare.csv", cmp_rows)

    # hurt-only deep dump
    hurt_folds = [f for f in base["folds"] if f["test_GT_hurt"] > 0]
    (study / "hurt_folds_detail.json").write_text(
        json.dumps(hurt_folds, indent=2, default=float) + "\n", encoding="utf-8"
    )

    # slim folds for full json (drop huge hurt_candidates lists in non-hurt)
    slim_results = {}
    for name, res in results.items():
        slim_folds = []
        for f in res["folds"]:
            sf = {k: v for k, v in f.items() if k not in ("atom_attr", "portable")}
            # keep atom_attr without hurt_candidates for size
            slim_atoms = []
            for a in f.get("atom_attr") or []:
                aa = {k: v for k, v in a.items() if k != "hurt_candidates"}
                slim_atoms.append(aa)
            sf["atom_attr"] = slim_atoms
            sf["selected_clauses"] = f.get("selected_clauses")
            slim_folds.append(sf)
        slim_results[name] = {
            **{k: v for k, v in res.items() if k != "folds"},
            "folds": slim_folds,
        }

    summary = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "protocol": (
            "strict LOO attribution + optional atom-repair re-LOO; not production"
        ),
        "eps": args.eps,
        "headline": (
            "Weight-method audit: no thick ε=0 production-safe plateau. "
            "Next is atom-level LOO hurt repair — not more weighting."
        ),
        "baseline_aggregate": {
            "verdict": base["verdict"],
            "n_GT0": base["n_test_GT_hurt_zero"],
            "sum_hurt": base["sum_test_GT_hurt"],
            "mean_test_FP": base["mean_test_FP_removed"],
        },
        "repair_compare": cmp_rows,
        "atom_labels_top": base["atom_labels"][:25],
        "by_config": slim_results,
        "not_production": True,
    }
    (study / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )

    print("\n=== REPAIR COMPARE (FP retained vs baseline) ===")
    for r in cmp_rows:
        print(
            f"  {r['config']:<36} {r['verdict']:<16} "
            f"GT0={r['n_GT0']}/{r['n_folds']} hurt={r['sum_test_GT_hurt']} "
            f"teFP={r['mean_test_FP']:.0f} retained={100 * r['FP_retained_vs_baseline']:.1f}%"
        )
    print("\n=== ATOM LABELS (baseline, failed first) ===")
    for r in base["atom_labels"][:15]:
        if r["atom_key"].startswith("kind:"):
            continue
        print(
            f"  {r['label']:<28} {r['atom_key']:<40} "
            f"fail={r['n_failed_folds']}/{r['n_selected_folds']} "
            f"FP~{r['mean_FP_alone']:.0f} held={r['heldouts_failed']}"
        )
    print(f"\nWrote {study}")


if __name__ == "__main__":
    main()
