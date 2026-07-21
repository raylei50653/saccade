#!/usr/bin/env python3
"""Leave-one-sequence-out validation for gate_rule_search policies.

Strict protocol
---------------
For each held-out sequence S:
  1. Fit atoms + search policy on the other 6 sequences only
     (quantile thr computed on train).
  2. Apply portable thr definitions to held-out S.
  3. Report train vs test FP_removed / GT_hurt.

This answers: is ε=0 in-sample OR policy a lucky safe region, or does it
transfer?

  uv run python scripts/tools/gate_rule_search_loo.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_gate_rule_loo_<stamp> \\
    --eps 0.0
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required for @dataclass
    spec.loader.exec_module(mod)
    return mod


_tools = Path(__file__).resolve().parent
grs = _load_module("gate_rule_search", _tools / "gate_rule_search.py")
_audit = _load_module(
    "audit_relink_safe_reject", _tools / "audit_relink_safe_reject.py"
)


def classify_fold(
    test_gt_hurt: int,
    test_gt_hurt_rate: float,
    test_fp: int,
    train_fp: int,
    eps: float,
) -> str:
    if test_gt_hurt == 0 and test_fp > 0:
        return "pass_eps0_transfer"
    if test_gt_hurt_rate <= eps + 1e-15 and test_fp > 0:
        return "pass_eps_transfer"
    if test_gt_hurt > 0 and test_fp > 0.5 * max(train_fp, 1) * 0.3:
        # still productive but leaks GT
        return "gt_leak_with_capacity"
    if test_gt_hurt > 0 and test_fp < 50:
        return "gt_leak_low_capacity"
    if test_fp == 0:
        return "no_capacity"
    return "fail_other"


def run_loo(
    pool: dict[str, np.ndarray],
    *,
    eps: float = 0.0,
    max_and_size: int = 3,
    max_or_rules: int = 5,
    min_fp_support: int = 100,
    tau_seq_std: float = 0.05,
) -> dict[str, Any]:
    seq = pool["seq"]
    seqs = sorted({str(s) for s in seq.tolist()})
    folds = []

    for held in seqs:
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
            tau_seq_std=tau_seq_std,
        )
        pol = train_res["policy"]
        portable = pol.get("portable") or {}
        clauses = portable.get("clauses") or []
        atom_specs = portable.get("atom_specs") or {}

        train_metrics = pol["final_metrics"]
        y_te = test_pool["gt_match"].astype(bool)
        sig_te = grs.extract_signals(test_pool)

        if not clauses or not atom_specs:
            rej = np.zeros(y_te.shape, dtype=bool)
        else:
            rej = grs.apply_policy_or(clauses, atom_specs, sig_te)

        n_pos = int(y_te.sum())
        n_neg = int((~y_te).sum())
        test_hurt = int((y_te & rej).sum())
        test_fp = int((~y_te & rej).sum())

        # per-clause failure on test
        clause_diag = []
        failed = []
        for atom_ids in clauses:
            cm = grs.apply_clause_specs(atom_ids, atom_specs, sig_te)
            h = int((y_te & cm).sum())
            f = int((~y_te & cm).sum())
            cid = " AND ".join(atom_ids)
            row = {
                "clause": cid,
                "test_GT_hurt": h,
                "test_FP_removed": f,
            }
            clause_diag.append(row)
            if h > 0:
                failed.append(cid)

        fold = {
            "heldout_seq": held,
            "epsilon": eps,
            "policy_size": len(clauses),
            "train_FP_removed": train_metrics["FP_removed"],
            "train_GT_hurt": train_metrics["GT_hurt"],
            "train_GT_hurt_rate": train_metrics["GT_hurt_rate"],
            "train_n_pos": train_metrics["n_pos"],
            "train_n_neg": train_metrics["n_neg"],
            "test_n_pos": n_pos,
            "test_n_neg": n_neg,
            "test_FP_removed": test_fp,
            "test_GT_hurt": test_hurt,
            "test_GT_hurt_rate": test_hurt / n_pos if n_pos else 0.0,
            "test_FP_removed_rate": test_fp / n_neg if n_neg else 0.0,
            "failed_clauses": failed,
            "failed_clause": failed[0] if failed else "",
            "failure_atom": (failed[0].split(" AND ")[0] if failed else ""),
            "classification": classify_fold(
                test_hurt,
                test_hurt / n_pos if n_pos else 0.0,
                test_fp,
                train_metrics["FP_removed"],
                eps,
            ),
            "policy_or": pol.get("policy_or"),
            "selected_clauses": pol.get("selected_clauses"),
            "portable": portable,
            "clause_diag": clause_diag,
            "train_per_seq": train_res.get("per_seq"),
        }
        folds.append(fold)

    # aggregate
    n = len(folds)
    n_pass = sum(
        1
        for f in folds
        if f["classification"] in ("pass_eps0_transfer", "pass_eps_transfer")
    )
    n_gt0 = sum(1 for f in folds if f["test_GT_hurt"] == 0)
    agg = {
        "n_folds": n,
        "n_test_GT_hurt_zero": n_gt0,
        "n_pass_transfer": n_pass,
        "mean_test_FP_removed": float(np.mean([f["test_FP_removed"] for f in folds])),
        "mean_test_GT_hurt": float(np.mean([f["test_GT_hurt"] for f in folds])),
        "mean_test_GT_hurt_rate": float(
            np.mean([f["test_GT_hurt_rate"] for f in folds])
        ),
        "sum_test_GT_hurt": int(sum(f["test_GT_hurt"] for f in folds)),
        "mean_train_FP_removed": float(np.mean([f["train_FP_removed"] for f in folds])),
        "verdict": (
            "loo_pass_eps0"
            if n_gt0 == n and n_pass == n
            else ("loo_partial" if n_gt0 >= n // 2 else "loo_fail_gt_leak")
        ),
    }
    return {"folds": folds, "aggregate": agg, "eps": eps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--max-and-size", type=int, default=3)
    ap.add_argument("--max-or-rules", type=int, default=5)
    ap.add_argument("--min-fp-support", type=int, default=100)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_gate_rule_loo_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)

    print(f"STUDY={study}")
    print(f"LOO eps={args.eps} …")
    res = run_loo(
        pool,
        eps=args.eps,
        max_and_size=args.max_and_size,
        max_or_rules=args.max_or_rules,
        min_fp_support=args.min_fp_support,
    )

    # CSV summary
    rows = []
    for f in res["folds"]:
        rows.append(
            {
                "heldout_seq": f["heldout_seq"],
                "epsilon": f["epsilon"],
                "policy_size": f["policy_size"],
                "train_FP_removed": f["train_FP_removed"],
                "train_GT_hurt": f["train_GT_hurt"],
                "test_FP_removed": f["test_FP_removed"],
                "test_GT_hurt": f["test_GT_hurt"],
                "test_GT_hurt_rate": f["test_GT_hurt_rate"],
                "failed_clause": f["failed_clause"],
                "failure_atom": f["failure_atom"],
                "classification": f["classification"],
            }
        )
    cols = list(rows[0].keys()) if rows else []
    with (study / "loo_folds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    (study / "loo_full.json").write_text(
        json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8"
    )
    summary = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "protocol": "strict LOO: search on 6-seq train, apply train thr to held-out",
        "eps": args.eps,
        "aggregate": res["aggregate"],
        "folds_table": rows,
        "status": "loo_validation",
        "not_production": True,
    }
    (study / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )

    print(
        f"{'heldout':<18} {'tr_FP':>6} {'tr_H':>4} {'te_FP':>6} {'te_H':>4} "
        f"{'te_H%':>7} {'class':<22} failed"
    )
    for f in res["folds"]:
        print(
            f"{f['heldout_seq']:<18} {f['train_FP_removed']:6d} {f['train_GT_hurt']:4d} "
            f"{f['test_FP_removed']:6d} {f['test_GT_hurt']:4d} "
            f"{100 * f['test_GT_hurt_rate']:6.2f}% {f['classification']:<22} "
            f"{f['failed_clause'][:40]}"
        )
    agg = res["aggregate"]
    print(
        f"\nAGGREGATE: verdict={agg['verdict']}  "
        f"folds_GT0={agg['n_test_GT_hurt_zero']}/{agg['n_folds']}  "
        f"mean_te_FP={agg['mean_test_FP_removed']:.1f}  "
        f"sum_te_hurt={agg['sum_test_GT_hurt']}"
    )
    print(f"Wrote {study / 'summary.json'}")


if __name__ == "__main__":
    main()
