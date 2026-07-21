#!/usr/bin/env python3
"""Narrow B2/e2e smoke contract for frozen repaired candidate only.

Target
------
  candidate_id = m_b1_repaired_eps0_loo_pass_20260709  ONLY

Allowed
-------
  offline replay of portable_policy on pairs
  same candidate_id study packaging
  default-off / research path documentation
  read-only attach of prior B2 e2e reconnect baseline (production-like substrate)

Not allowed
-----------
  preset change
  silent default-on
  new atom search
  extra repair during smoke
  mixing ε=0.01 relaxed frontier
  claiming online injection if tracker has no hook

Questions answered
------------------
  1. Can policy apply on B2/e2e substrate?
  2. GT_hurt still 0 / no contracted regression (offline claim re-check)?
  3. FP pruning / reconnect side effects expected?
  4. IDF1 / AssA / reconnect rates (substrate baseline only until online hook)?
  5. Runtime coupling fail offline claim? (blocked until online path)

  uv run python scripts/tools/smoke_repaired_candidate_b2e2e.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --candidate-dir out/signal_study/m_b1_repaired_eps0_loo_pass_20260709 \\
    --b2-study out/signal_study/m_b2_bridge_ab_20260709T094646Z \\
    --study-dir out/signal_study/m_b2e2e_smoke_<stamp>
"""
# status: experiment

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_tools = Path(__file__).resolve().parent

CANDIDATE_ID = "m_b1_repaired_eps0_loo_pass_20260709"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


grs = _load("gate_rule_search", _tools / "gate_rule_search.py")
_audit = _load("audit_relink_safe_reject", _tools / "audit_relink_safe_reject.py")


def offline_replay(pairs: Path, portable: dict[str, Any]) -> dict[str, Any]:
    pool = _audit.load_gt_valid_pool(pairs)
    _audit.ensure_prod_proxy_scores(pool)
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    sig = grs.extract_signals(pool)
    clauses = portable.get("clauses") or []
    atom_specs = portable.get("atom_specs") or {}
    if not clauses or not atom_specs:
        raise SystemExit("portable_policy missing clauses/atom_specs")

    rej = grs.apply_policy_or(clauses, atom_specs, sig)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    hurt = int((y & rej).sum())
    fprm = int((~y & rej).sum())

    per_seq = []
    for s in sorted({str(x) for x in seq.tolist()}):
        m = seq == s
        ys = y[m]
        rs = rej[m]
        npos = int(ys.sum())
        nneg = int((~ys).sum())
        per_seq.append(
            {
                "seq": s,
                "n_pos": npos,
                "n_neg": nneg,
                "GT_hurt": int((ys & rs).sum()),
                "FP_removed": int((~ys & rs).sum()),
                "GT_hurt_rate": float((ys & rs).sum() / npos) if npos else 0.0,
                "FP_removed_rate": float((~ys & rs).sum() / nneg) if nneg else 0.0,
            }
        )

    # per-clause
    clause_rows = []
    for atom_ids in clauses:
        cm = grs.apply_clause_specs(atom_ids, atom_specs, sig)
        clause_rows.append(
            {
                "clause": " AND ".join(atom_ids),
                "GT_hurt": int((y & cm).sum()),
                "FP_removed": int((~y & cm).sum()),
            }
        )

    return {
        "pairs_csv": str(pairs.resolve()),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "GT_hurt": hurt,
        "GT_hurt_rate": hurt / n_pos if n_pos else 0.0,
        "FP_removed": fprm,
        "FP_removed_rate": fprm / n_neg if n_neg else 0.0,
        "per_seq": per_seq,
        "clause_diag": clause_rows,
        "policy_or": " OR ".join(f"({' AND '.join(c)})" for c in clauses),
        "pass_eps0": hurt == 0,
    }


def load_b2_reference(b2_study: Path | None) -> dict[str, Any]:
    if b2_study is None or not b2_study.exists():
        return {
            "available": False,
            "note": "no --b2-study provided or path missing",
        }
    out: dict[str, Any] = {
        "available": True,
        "b2_study": str(b2_study.resolve()),
        "role": (
            "production-like B2 reconnect A/B reference (bridge on/off). "
            "NOT an apply of this candidate policy online."
        ),
        "candidate_applied_online": False,
    }
    mr = b2_study / "metrics_reconnect.json"
    ctx = b2_study / "context.json"
    if mr.exists():
        out["metrics_reconnect"] = json.loads(mr.read_text(encoding="utf-8"))
    if ctx.exists():
        out["context"] = json.loads(ctx.read_text(encoding="utf-8"))
    # try extract OVERALL-ish numbers if present in metrics
    return out


def decide(
    offline: dict[str, Any],
    b2: dict[str, Any],
    *,
    freeze_fp: int | None,
    freeze_hurt: int | None,
) -> dict[str, Any]:
    """Answer the five smoke questions + result block."""
    q1 = {
        "question": "candidate policy apply on B2/e2e substrate?",
        "offline_U_relink_pair": True,
        "online_tracker_hook": False,
        "answer": (
            "offline_yes_online_not_wired — portable_policy applies on pairs; "
            "no production/runtime path injects this OR-5 tail policy yet"
        ),
    }
    q2 = {
        "question": "GT_hurt still 0 / no contracted regression offline?",
        "GT_hurt": offline["GT_hurt"],
        "pass_eps0": offline["pass_eps0"],
        "freeze_GT_hurt": freeze_hurt,
        "answer": "pass" if offline["pass_eps0"] else "FAIL_GT_LEAK",
    }
    fp_ok = True
    if freeze_fp is not None and offline["FP_removed"] > 0:
        # allow tiny numeric drift only
        ratio = offline["FP_removed"] / max(freeze_fp, 1)
        fp_ok = 0.99 <= ratio <= 1.01 or offline["FP_removed"] == freeze_fp
    q3 = {
        "question": "FP pruning / reconnect side effects expected?",
        "offline_FP_removed": offline["FP_removed"],
        "freeze_FP_removed": freeze_fp,
        "fp_matches_freeze": fp_ok,
        "reconnect_side_effects": (
            "not_measured_for_candidate — no online apply; "
            "B2 reference is production bridge ablate only"
        ),
        "answer": (
            "offline_FP_as_expected"
            if fp_ok and offline["pass_eps0"]
            else "offline_FP_or_hurt_mismatch"
        ),
    }
    q4 = {
        "question": "IDF1 / AssA / reconnect rates no clear regression?",
        "candidate_e2e_delta": None,
        "note": (
            "cannot attribute e2e to this candidate without online injection; "
            "B2 study attached as substrate baseline only"
        ),
        "b2_available": b2.get("available", False),
        "answer": "not_applicable_until_online_hook",
    }
    q5 = {
        "question": "runtime ordering / candidate-generation coupling breaks offline claim?",
        "answer": "untested — requires default-off online path applying portable_policy",
    }

    offline_ok = offline["pass_eps0"] and fp_ok
    blockers = []
    if not offline_ok:
        blockers.append("offline_replay_failed_eps0_or_fp_drift")
    blockers.append(
        "no_online_tracker_hook_for_portable_or5_tails "
        "(cannot answer e2e/reconnect under candidate apply)"
    )
    if not b2.get("available"):
        blockers.append("b2_reference_study_missing")

    # validation_status: region candidate remains; e2e not yet elevating
    if offline_ok:
        val_status = "LOO_pass_region_candidate"
        e2e_safe = "no"  # cannot be yes without online
        smoke_verdict = "offline_smoke_pass__online_blocked"
    else:
        val_status = "offline_smoke_fail"
        e2e_safe = "no"
        smoke_verdict = "offline_smoke_fail"

    return {
        "questions": {
            "q1_apply": q1,
            "q2_gt_hurt": q2,
            "q3_fp_reconnect": q3,
            "q4_e2e_metrics": q4,
            "q5_runtime_coupling": q5,
        },
        "result_block": {
            "candidate_id": CANDIDATE_ID,
            "validation_status": val_status,
            "validation_status_change": "unchanged (offline re-confirmed; e2e not elevating)",
            "e2e_safe_for_default_off": e2e_safe,
            "production_preset": "unchanged",
            "smoke_verdict": smoke_verdict,
            "blockers_before_default_off": blockers,
            "lifecycle_status": "candidate_only / pre-production research",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path(f"out/signal_study/{CANDIDATE_ID}"),
    )
    ap.add_argument(
        "--b2-study",
        type=Path,
        default=Path("out/signal_study/m_b2_bridge_ab_20260709T094646Z"),
    )
    ap.add_argument("--study-dir", type=Path, default=None)
    args = ap.parse_args()

    cand_dir = args.candidate_dir
    portable_path = cand_dir / "portable_policy.json"
    cand_json_path = cand_dir / "candidate.json"
    if not portable_path.exists():
        raise SystemExit(f"missing {portable_path}")

    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    freeze_fp = freeze_hurt = None
    if cand_json_path.exists():
        cj = json.loads(cand_json_path.read_text(encoding="utf-8"))
        fm = (cj.get("in_sample_7seq") or {}).get("final_metrics") or {}
        freeze_fp = fm.get("FP_removed")
        freeze_hurt = fm.get("GT_hurt")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(
        f"out/signal_study/m_b2e2e_smoke_{CANDIDATE_ID}_{stamp}"
    )
    study.mkdir(parents=True, exist_ok=True)

    print(f"STUDY={study}")
    print(f"candidate_id={CANDIDATE_ID}")
    print("contract: offline replay only; preset unchanged; no atom search/repair")

    offline = offline_replay(args.pairs, portable)
    print(
        f"offline: GT_hurt={offline['GT_hurt']} FP={offline['FP_removed']} "
        f"pass_eps0={offline['pass_eps0']}"
    )
    for r in offline["per_seq"]:
        print(f"  {r['seq']:<18} hurt={r['GT_hurt']} FP={r['FP_removed']}")

    b2 = load_b2_reference(args.b2_study if args.b2_study.exists() else None)
    print(f"b2_reference: available={b2.get('available')} path={b2.get('b2_study')}")

    decision = decide(offline, b2, freeze_fp=freeze_fp, freeze_hurt=freeze_hurt)
    rb = decision["result_block"]

    summary = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID,
        "contract": {
            "target": CANDIDATE_ID + " only",
            "allowed": [
                "offline replay",
                "same candidate_id",
                "default-off research path",
                "attach prior B2 baseline as substrate reference",
            ],
            "not_allowed": [
                "preset change",
                "silent default-on",
                "new atom search",
                "extra repair during smoke",
                "mixing ε=0.01 relaxed frontier",
                "claim online apply without tracker hook",
            ],
        },
        "offline_replay": offline,
        "b2_substrate_reference": {
            k: v
            for k, v in b2.items()
            if k not in ("metrics_reconnect",)  # keep full in separate file
        },
        "decision": decision,
        "not_production": True,
    }
    if b2.get("metrics_reconnect") is not None:
        (study / "b2_metrics_reconnect_ref.json").write_text(
            json.dumps(b2["metrics_reconnect"], indent=2, default=float) + "\n",
            encoding="utf-8",
        )

    (study / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    (study / "result_block.json").write_text(
        json.dumps(rb, indent=2, default=float) + "\n", encoding="utf-8"
    )

    print("\n=== B2/e2e smoke result ===")
    print(f"B2/e2e smoke result for {CANDIDATE_ID}:")
    print(f"  validation_status remains / changes to: {rb['validation_status']}")
    print(f"  ({rb['validation_status_change']})")
    print(f"  e2e_safe_for_default_off: {rb['e2e_safe_for_default_off']}")
    print(f"  production_preset: {rb['production_preset']}")
    print(f"  smoke_verdict: {rb['smoke_verdict']}")
    print("  blockers_before_default_off:")
    for b in rb["blockers_before_default_off"]:
        print(f"    - {b}")
    print(f"\nWrote {study / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()
