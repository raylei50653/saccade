#!/usr/bin/env python3
"""Stage 1 A/B + Stage 1b action-path control runner for portable OR-tail hook.

Arms:
  A1 — hook disabled (baseline)
  B  — frozen offline policy on (may be vacuous online)
  P  — Stage 1b activation control (atom0 thr=0.2 pre-specified; plumbing only)
  F  — Stage 1b force-reject-all (atom0 thr=-1; plumbing only)
  B-audit — still NOT implemented (fail-closed CLI)

Stage 1a = evaluation-entry (eligible counters + freeze B load).
Stage 1b = action path (atom fire → reject → decision change under controls).
Does **not** search production thr or remodel freeze candidate.

Usage:
  bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \\
    --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \\
    --study-dir out/signal_study/m_b1_hook_ab_<stamp> \\
    --run-e2e --run-action-path-controls

Contract:
  docs/modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md §Stage 1
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from saccade.perception.eval.portable_or_tail import (  # noqa: E402
    RELINK_DEBUG_HOST_INDEX,
    classify_e2e_status,
    classify_stage1_milestones,
    derive_atom_summary,
    evaluate_policy,
    fire_class_counts,
    load_portable_policy,
    reconcile_fire_classes,
    snapshot_policy,
)

# Trusted A0 (B2 bridge-on substrate) for hook-off identity checks.
DEFAULT_A0_REF = Path("results/MOT17_eval_m_b2_bridge_on_20260709T094646Z")
DEFAULT_ACTIVATION_POLICY = Path(
    "scripts/tools/fixtures/m_b1_stage1/activation_control_policy.json"
)
DEFAULT_FORCE_REJECT_POLICY = Path(
    "scripts/tools/fixtures/m_b1_stage1/force_reject_policy.json"
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: Any) -> str:
    raw = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")
    return _sha256_bytes(raw.encode("utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        path.write_text("", encoding="utf-8")
        return _sha256_bytes(b"")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return _sha256_bytes(path.read_bytes())


def offline_event_table(pairs: Path, policy) -> dict[str, Any]:
    """Full event-level table from offline B1 pairs (policy evaluation substrate).

    This is the machine-readable full table required by Stage 1 for the
    offline candidate universe. Online counters from e2e are merged separately.
    """
    import importlib.util

    audit_path = ROOT / "scripts/tools/audit_relink_safe_reject.py"
    spec = importlib.util.spec_from_file_location(
        "audit_relink_safe_reject", audit_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pool = mod.load_gt_valid_pool(pairs)
    mod.ensure_prod_proxy_scores(pool)
    n = int(pool["gt_match"].shape[0])
    signals = {
        "score_m_bridge": pool["score_m_bridge"],
        "abs_log_h": pool["log_h_ratio"],
        "dist_h": pool["dist_h"],
        "abs_ratio_m1": np.abs(pool["h_ratio_lost_over_cand"] - 1.0),
        "resid_mean": 0.5 * (pool["fwd_resid"] + pool["bwd_resid"]),
    }
    out = evaluate_policy(policy, signals)
    seq = pool["seq"]
    events = []
    for i in range(n):
        events.append(
            {
                "run_id": "offline_pairs_replay",
                "sequence": str(seq[i]),
                "frame": -1,
                "event_id": f"offline_{i}",
                "runtime_candidate_id": f"pair_{i}",
                "policy_candidate_id": policy.candidate_id,
                "policy_file_hash": policy.file_hash,
                "policy_schema_version": policy.schema_version,
                "atom_bitmask": int(out["atom_bitmask"][i]),
                "fired_atom_ids": "|".join(out["fired_atom_ids"][i]),
                "n_atoms_fired": int(out["n_atoms_fired"][i]),
                "fire_class": str(out["fire_class"][i]),
                "rejected_by_hook": int(bool(out["reject"][i])),
                "score_m_bridge": float(signals["score_m_bridge"][i]),
                "abs_log_h": float(signals["abs_log_h"][i]),
                "dist_h": float(signals["dist_h"][i]),
                "abs_ratio_m1": float(signals["abs_ratio_m1"][i]),
                "resid_mean": float(signals["resid_mean"][i]),
                "gt_match": int(bool(pool["gt_match"][i])),
                "universe": "offline_gt_valid_pairs",
            }
        )
    counts = fire_class_counts(out["fire_class"])
    atom_summary = derive_atom_summary(
        policy, out["atom_masks"], out["reject"], sequences=seq
    )
    # per-sequence
    per_seq_rows = []
    for s in sorted({str(x) for x in seq.tolist()}):
        m = seq == s
        fc = fire_class_counts(out["fire_class"][m])
        per_seq_rows.append(
            {
                "sequence": s,
                "n_hook_eligible": int(m.sum()),
                "n_zero_fire": fc["n_zero_fire"],
                "n_singleton": fc["n_singleton"],
                "n_cofire": fc["n_cofire"],
                "n_rejected": int(out["reject"][m].sum()),
                "n_competitor_changed": -1,
                "n_reconnect_changed": -1,
                "reconnect_success_delta": None,
                "reconnect_miss_delta": None,
                "GT_hurt": int((pool["gt_match"][m] & out["reject"][m]).sum()),
                "FP_removed": int((~pool["gt_match"][m] & out["reject"][m]).sum()),
            }
        )
    return {
        "events": events,
        "atom_summary": atom_summary,
        "per_sequence": per_seq_rows,
        "counts": {
            "n_hook_eligible": n,
            **counts,
            "n_rejected": int(out["reject"].sum()),
        },
        "reject": out["reject"],
        "gt_match": pool["gt_match"],
    }


def _mot_result_hashes(output_dir: Path) -> dict[str, str]:
    """SHA-256 of each MOT result file (identity / determinism)."""
    out: dict[str, str] = {}
    for p in sorted(output_dir.glob("MOT17-*-SDP.txt")):
        out[p.name] = _sha256_bytes(p.read_bytes())
    return out


def _aggregate_mot_hash(hashes: dict[str, str]) -> str:
    if not hashes:
        return ""
    payload = "\n".join(f"{k}={v}" for k, v in sorted(hashes.items())) + "\n"
    return _sha256_bytes(payload.encode("utf-8"))


def _parse_metric_pct(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _evaluate_mot_dir(output_dir: Path, sequences: str | None) -> dict[str, Any] | None:
    """Run TrackEval/motmetrics on an arm output directory."""
    from saccade.perception.eval.metrics import run_motmetrics_evaluation

    metrics = run_motmetrics_evaluation(
        data_root=str(ROOT / "datasets" / "MOT17"),
        split="train",
        output=str(output_dir),
        sequences=sequences or "",
        detector="SDP",
    )
    if not metrics:
        return None
    numeric: dict[str, Any] = {}
    for k, v in metrics.items():
        pct = _parse_metric_pct(v)
        if pct is not None and k in {
            "MOTA",
            "IDF1",
            "HOTA",
            "DetA",
            "AssA",
            "Recall",
            "Precision",
        }:
            numeric[k] = pct
        else:
            # Keep raw counts (IDs, FP, FN, …) when present.
            try:
                numeric[k] = float(str(v).replace(",", ""))
            except ValueError:
                numeric[k] = v
    numeric["_raw"] = {k: str(v) for k, v in metrics.items()}
    return numeric


def _load_relink_debug_sum(output_dir: Path) -> dict[str, Any]:
    """Sum per-seq `_relink_debug_*.json` named counters (end-of-seq snapshots).

    Note: get_relink_debug is cumulative for the process lifetime of one
    sequence run; summing per-seq files is the correct multi-seq aggregate.
    """
    totals: dict[str, int] = {k: 0 for k in RELINK_DEBUG_HOST_INDEX}
    per_seq: dict[str, dict[str, int | None]] = {}
    for p in sorted(output_dir.glob("_relink_debug_*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = p.name.removeprefix("_relink_debug_").removesuffix(".json")
        row: dict[str, int | None] = {}
        for name in RELINK_DEBUG_HOST_INDEX:
            v = obj.get(name)
            if v is None:
                row[name] = None
                continue
            iv = int(v)
            row[name] = iv
            totals[name] = totals.get(name, 0) + iv
        per_seq[seq] = row
    return {"totals": totals, "per_sequence": per_seq, "n_seq_files": len(per_seq)}


def _compare_identity(a1_hashes: dict[str, str], a0_dir: Path | None) -> dict[str, Any]:
    """A1 (hook-off) vs trusted A0 result-file identity."""
    if a0_dir is None or not a0_dir.is_dir():
        return {
            "compared": False,
            "identity_ok": None,
            "reason": "no A0 ref dir",
            "mismatched": [],
            "missing_in_a0": [],
            "missing_in_a1": [],
            "a0_dir": str(a0_dir) if a0_dir else None,
        }
    a0_hashes = _mot_result_hashes(a0_dir)
    mismatched = []
    missing_in_a0 = []
    missing_in_a1 = []
    for name, h in a1_hashes.items():
        if name not in a0_hashes:
            missing_in_a0.append(name)
        elif a0_hashes[name] != h:
            mismatched.append(name)
    for name in a0_hashes:
        if name not in a1_hashes:
            missing_in_a1.append(name)
    identity_ok = not mismatched and not missing_in_a0 and not missing_in_a1
    # If A0 has extra seqs we did not run (subset e2e), only score overlapping set.
    if missing_in_a1 and not mismatched and not missing_in_a0:
        # A1 is a proper subset of A0 seqs — identity on the subset only.
        identity_ok = len(a1_hashes) > 0 and not mismatched and not missing_in_a0
    return {
        "compared": True,
        "identity_ok": identity_ok,
        "a0_dir": str(a0_dir),
        "a0_aggregate_hash": _aggregate_mot_hash(a0_hashes),
        "a1_aggregate_hash": _aggregate_mot_hash(a1_hashes),
        "mismatched": mismatched,
        "missing_in_a0": missing_in_a0,
        "missing_in_a1": missing_in_a1,
        "n_compared": len(a1_hashes),
    }


def _metrics_delta(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> dict[str, float]:
    """B − A for headline pct metrics (positive = B better for IDF1/AssA/…)."""
    keys = ("IDF1", "AssA", "HOTA", "MOTA", "IDs", "FP", "FN")
    out: dict[str, float] = {}
    if not a or not b:
        return out
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = float(vb) - float(va)
    return out


def run_e2e_arm(
    *,
    label: str,
    output_dir: Path,
    policy_path: Path | None,
    audit: bool,
    audit_dir: Path | None,
    sequences: str | None,
    extra_args: list[str],
) -> dict[str, Any]:
    """Invoke mot17.py for one A/B arm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Wrap with ab_env.sh so torch CUDA + build/ extension load correctly.
    ab_env = ROOT / "scratch" / "ab_env.sh"
    inner = [
        "uv",
        "run",
        "python",
        str(ROOT / "scripts/eval/mot17.py"),
        "--preset",
        "mamba_whole_graph_m",
        "--detector",
        "SDP",
        "--double-buffer",
        "--detect-barrier",
        "event",
        "--output",
        str(output_dir),
    ]
    if sequences:
        inner.extend(["--sequences", sequences])
    if policy_path is not None:
        inner.extend(["--research-portable-or-tail-policy", str(policy_path)])
        if audit:
            inner.append("--research-portable-or-tail-audit")
            if audit_dir is not None:
                inner.extend(["--research-portable-or-tail-audit-dir", str(audit_dir)])
    inner.extend(extra_args)
    cmd = ["bash", str(ab_env), *inner] if ab_env.is_file() else inner
    env = os.environ.copy()
    # Ensure hook-off arm does not inherit env policy.
    if policy_path is None:
        env.pop("SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY", None)
    t0 = datetime.now(timezone.utc)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    t1 = datetime.now(timezone.utc)
    (output_dir / "_hook_ab_cmd.json").write_text(
        json.dumps(
            {
                "label": label,
                "cmd": cmd,
                "returncode": proc.returncode,
                "seconds": (t1 - t0).total_seconds(),
                "stdout_tail": proc.stdout[-8000:],
                "stderr_tail": proc.stderr[-8000:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"e2e arm {label} failed rc={proc.returncode}: {proc.stderr[-2000:]}"
        )
    result_hashes = _mot_result_hashes(output_dir)
    metrics = _evaluate_mot_dir(output_dir, sequences)
    relink = _load_relink_debug_sum(output_dir)
    return {
        "label": label,
        "output_dir": str(output_dir),
        "seconds": (t1 - t0).total_seconds(),
        "returncode": proc.returncode,
        "result_hashes": result_hashes,
        "aggregate_result_hash": _aggregate_mot_hash(result_hashes),
        "metrics": metrics,
        "relink_debug": relink,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json"
        ),
    )
    ap.add_argument(
        "--pairs",
        type=Path,
        default=Path("out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv"),
    )
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument(
        "--offline-events-only",
        action="store_true",
        help="Build offline full event table + summaries without e2e runs",
    )
    ap.add_argument(
        "--run-e2e",
        action="store_true",
        help="Run Stage 1a A1 (hook-off) vs B (frozen policy) e2e arms",
    )
    ap.add_argument(
        "--run-action-path-controls",
        action="store_true",
        help=(
            "Run Stage 1b plumbing arms P (activation thr=0.2) and F "
            "(force-reject-all). Requires --run-e2e for A1 baseline compare."
        ),
    )
    ap.add_argument(
        "--activation-policy",
        type=Path,
        default=DEFAULT_ACTIVATION_POLICY,
        help="Stage 1b activation-control policy JSON (control_arm=activation)",
    )
    ap.add_argument(
        "--force-reject-policy",
        type=Path,
        default=DEFAULT_FORCE_REJECT_POLICY,
        help="Stage 1b force-reject policy JSON (control_arm=force_reject)",
    )
    ap.add_argument(
        "--sequences",
        default=None,
        help="Optional sequence subset for e2e (default: full SDP set from preset)",
    )
    ap.add_argument(
        "--a0-ref",
        type=Path,
        default=DEFAULT_A0_REF,
        help="Trusted A0 MOT dir for A1 hook-off identity (B2 bridge-on)",
    )
    ap.add_argument(
        "--skip-a0-identity",
        action="store_true",
        help="Do not require / compare A1 vs A0 result hashes",
    )
    ap.add_argument("extra", nargs="*", help="Extra args forwarded to mot17.py")
    args = ap.parse_args()
    if args.run_action_path_controls and not args.run_e2e:
        ap.error("--run-action-path-controls requires --run-e2e (needs A1 baseline)")

    study = args.study_dir
    study.mkdir(parents=True, exist_ok=True)

    # Offline runner: still lock freeze thr/hash when loading the Stage 1 freeze file.
    # Synthetic thr exploration is not this script's job (Stage 2).
    policy = load_portable_policy(args.policy, enforce_freeze_lock=True)
    snap_hash = _write_json(
        study / "portable_policy.snapshot.json", snapshot_policy(policy)
    )

    offline = None
    if args.pairs.is_file():
        offline = offline_event_table(args.pairs, policy)
        # Prefer parquet if pyarrow/pandas available; else CSV.
        events_path = study / "hook_candidate_events.csv"
        rej_path = study / "rejected_events.csv"
        ev_hash = _write_csv(events_path, offline["events"])
        rejected = [e for e in offline["events"] if e["rejected_by_hook"]]
        rej_hash = _write_csv(rej_path, rejected)
        atom_hash = _write_csv(study / "atom_summary.csv", offline["atom_summary"])
        seq_hash = _write_csv(
            study / "per_sequence_summary.csv", offline["per_sequence"]
        )
        counts = offline["counts"]
        recon_errs = reconcile_fire_classes(
            counts["n_hook_eligible"],
            counts["n_zero_fire"],
            counts["n_singleton"],
            counts["n_cofire"],
            counts["n_rejected"],
        )
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pylist(offline["events"])
            pq.write_table(table, study / "hook_candidate_events.parquet")
            pq.write_table(
                pa.Table.from_pylist(rejected), study / "rejected_events.parquet"
            )
            parquet_ok = True
        except Exception:
            parquet_ok = False
    else:
        ev_hash = rej_hash = atom_hash = seq_hash = ""
        recon_errs = ["pairs missing; offline event table skipped"]
        parquet_ok = False
        counts = {}

    e2e_meta: dict[str, Any] = {"ran": False}
    a0_identity: dict[str, Any] = {"compared": False, "identity_ok": None}
    metrics_delta: dict[str, float] = {}
    online_n_rejected = int(counts.get("n_rejected", 0))
    online_n_eligible = 0
    a1_eq_b = False
    online_vacuous = False
    a0_soft_ok = False
    a0_strict_ok: bool | None = None
    activation_ok: bool | None = None
    force_reject_ok: bool | None = None
    control_meta: dict[str, Any] = {"ran": False}

    if args.run_e2e:
        a1_dir = study / "e2e_A1_hook_off"
        b_dir = study / "e2e_B_hook_on"
        a1 = run_e2e_arm(
            label="A1",
            output_dir=a1_dir,
            policy_path=None,
            audit=False,
            audit_dir=None,
            sequences=args.sequences,
            extra_args=list(args.extra),
        )
        b = run_e2e_arm(
            label="B",
            output_dir=b_dir,
            policy_path=policy.path,
            audit=False,
            audit_dir=None,
            sequences=args.sequences,
            extra_args=list(args.extra),
        )
        a0_ref = None if args.skip_a0_identity else args.a0_ref
        a0_identity = _compare_identity(a1.get("result_hashes") or {}, a0_ref)
        a1_eq_b = bool(a1.get("aggregate_result_hash")) and a1.get(
            "aggregate_result_hash"
        ) == b.get("aggregate_result_hash")
        n_mis = len(a0_identity.get("mismatched") or [])
        n_cmp = int(a0_identity.get("n_compared") or 0)
        a0_strict_ok = (
            bool(a0_identity.get("identity_ok"))
            if a0_identity.get("compared")
            else None
        )
        a0_soft_ok = bool(
            a0_identity.get("compared")
            and n_cmp > 0
            and n_mis <= 1
            and not a0_identity.get("missing_in_a0")
            and (n_cmp - n_mis) >= max(1, n_cmp - 1)
        )
        if a0_soft_ok and a0_strict_ok is not True:
            a0_identity["soft_identity_ok"] = True
            a0_identity["soft_identity_note"] = (
                f"{n_cmp - n_mis}/{n_cmp} seq file hashes match A0; "
                f"mismatched={a0_identity.get('mismatched')}; "
                "soft-pass only — not contract strict A1==A0 identity"
            )
        a0_identity["strict_identity_ok"] = a0_strict_ok
        a0_identity["default_off_compatibility"] = (
            "strict_pass"
            if a0_strict_ok is True
            else (
                "soft_pass"
                if a0_soft_ok
                else ("fail" if a0_strict_ok is False else "not_compared")
            )
        )

        metrics_delta = _metrics_delta(a1.get("metrics"), b.get("metrics"))
        b_relink = (b.get("relink_debug") or {}).get("totals") or {}
        a1_relink = (a1.get("relink_debug") or {}).get("totals") or {}
        online_n_rejected = int(b_relink.get("hook_rejected") or 0)
        online_n_eligible = int(b_relink.get("hook_eligible") or 0)
        online_vacuous = online_n_eligible > 0 and online_n_rejected == 0 and a1_eq_b
        e2e_meta = {
            "ran": True,
            "A1": a1,
            "B": b,
            "a0_identity": a0_identity,
            "a1_eq_b_result_hash": a1_eq_b,
            "metrics_delta_B_minus_A1": metrics_delta,
            "online_hook_counters_B": b_relink,
            "online_hook_counters_A1": a1_relink,
            "online_vacuous_policy": online_vacuous,
            "online_vacuous_note": (
                "Stage 1a: evaluation-entry only — eligible>0 proves path is "
                "entered; rejected=0 means freeze thr never activated rejection "
                "(support mismatch). Does NOT prove action path."
                if online_vacuous
                else None
            ),
        }

        if args.run_action_path_controls:
            act_pol = load_portable_policy(
                args.activation_policy, enforce_freeze_lock=True
            )
            fr_pol = load_portable_policy(
                args.force_reject_policy, enforce_freeze_lock=True
            )
            _require_control = (
                act_pol.control_arm == "activation"
                and fr_pol.control_arm == "force_reject"
            )
            if not _require_control:
                raise RuntimeError(
                    "action-path control policies must declare control_arm="
                    "activation|force_reject"
                )
            _write_json(
                study / "activation_control_policy.snapshot.json",
                snapshot_policy(act_pol),
            )
            _write_json(
                study / "force_reject_policy.snapshot.json",
                snapshot_policy(fr_pol),
            )
            p_arm = run_e2e_arm(
                label="P_activation",
                output_dir=study / "e2e_P_activation_control",
                policy_path=act_pol.path,
                audit=False,
                audit_dir=None,
                sequences=args.sequences,
                extra_args=list(args.extra),
            )
            f_arm = run_e2e_arm(
                label="F_force_reject",
                output_dir=study / "e2e_F_force_reject",
                policy_path=fr_pol.path,
                audit=False,
                audit_dir=None,
                sequences=args.sequences,
                extra_args=list(args.extra),
            )
            p_tot = (p_arm.get("relink_debug") or {}).get("totals") or {}
            f_tot = (f_arm.get("relink_debug") or {}).get("totals") or {}
            p_elig = int(p_tot.get("hook_eligible") or 0)
            p_rej = int(p_tot.get("hook_rejected") or 0)
            p_atom0 = int(p_tot.get("atom0_score_m_bridge") or 0)
            f_elig = int(f_tot.get("hook_eligible") or 0)
            f_rej = int(f_tot.get("hook_rejected") or 0)
            f_atom0 = int(f_tot.get("atom0_score_m_bridge") or 0)
            a1_hash = a1.get("aggregate_result_hash")
            p_differs = bool(a1_hash) and p_arm.get("aggregate_result_hash") != a1_hash
            f_differs = bool(a1_hash) and f_arm.get("aggregate_result_hash") != a1_hash
            # Activation: full arrow signal → atom → reject → downstream delta.
            # Require p_differs so Stage 1b cannot pass on counters alone if
            # rejects never change MOT output (same bar as F for result delta).
            activation_ok = (
                p_elig > 0
                and p_atom0 > 0
                and p_rej > 0
                and p_rej == p_atom0
                and p_differs
            )
            # Force-reject: every eligible pair rejected via atom0 + decision change.
            force_reject_ok = (
                f_elig > 0 and f_rej == f_elig and f_atom0 >= f_rej and f_differs
            )
            control_meta = {
                "ran": True,
                "P": p_arm,
                "F": f_arm,
                "P_counters": p_tot,
                "F_counters": f_tot,
                "activation_checks": {
                    "hook_eligible": p_elig,
                    "atom0": p_atom0,
                    "hook_rejected": p_rej,
                    "rejected_eq_atom0": p_rej == p_atom0,
                    "result_differs_from_A1": p_differs,
                    "pass": activation_ok,
                },
                "force_reject_checks": {
                    "hook_eligible": f_elig,
                    "atom0": f_atom0,
                    "hook_rejected": f_rej,
                    "rejected_eq_eligible": f_rej == f_elig,
                    "result_differs_from_A1": f_differs,
                    "pass": force_reject_ok,
                },
            }
            e2e_meta["controls"] = control_meta

    # Metrics / classification
    ab_metrics = {
        "offline": {
            "GT_hurt": int((offline["gt_match"] & offline["reject"]).sum())
            if offline
            else None,
            "FP_removed": int((~offline["gt_match"] & offline["reject"]).sum())
            if offline
            else None,
            "n_pos": int(offline["gt_match"].sum()) if offline else None,
            "n_neg": int((~offline["gt_match"]).sum()) if offline else None,
        },
        "e2e": e2e_meta,
        "metrics_delta_B_minus_A1": metrics_delta if args.run_e2e else {},
        "a0_identity": a0_identity,
        "controls": control_meta,
    }
    ab_hash = _write_json(study / "ab_metrics.json", ab_metrics)

    a1_sec = (e2e_meta.get("A1") or {}).get("seconds") if e2e_meta.get("ran") else None
    b_sec = (e2e_meta.get("B") or {}).get("seconds") if e2e_meta.get("ran") else None
    p_sec = (
        (control_meta.get("P") or {}).get("seconds")
        if control_meta.get("ran")
        else None
    )
    f_sec = (
        (control_meta.get("F") or {}).get("seconds")
        if control_meta.get("ran")
        else None
    )
    runtime = {
        "hook_disabled_overhead": None,
        "hook_enabled_policy_overhead": (
            (b_sec - a1_sec) if (a1_sec is not None and b_sec is not None) else None
        ),
        "audit_enabled_overhead": None,
        "note": (
            "Wall-clock arm seconds only; not pure policy kernel overhead. "
            "hook_disabled/audit overhead and pure policy overhead: PENDING."
        ),
        "e2e_seconds": {"A1": a1_sec, "B": b_sec, "P": p_sec, "F": f_sec},
    }
    rt_hash = _write_json(study / "runtime.json", runtime)

    det = {
        "hook_off_baseline_compatibility_hash": (e2e_meta.get("A1") or {}).get(
            "aggregate_result_hash"
        ),
        "hook_on_result_hash": (e2e_meta.get("B") or {}).get("aggregate_result_hash"),
        "a0_identity": a0_identity,
        "hook_on_repeated_run_hashes": [],
        "note": (
            "hook_on_repeated_run_hashes empty — repeated-run determinism PENDING; "
            "not claimed as Stage 1 pass evidence"
        ),
        "event_table_hash": ev_hash,
        "summary_artifact_hashes": {
            "atom_summary": atom_hash,
            "per_sequence_summary": seq_hash,
            "rejected_events": rej_hash,
            "portable_policy.snapshot": snap_hash,
            "ab_metrics": ab_hash,
            "runtime": rt_hash,
        },
        "reconciliation_errors": recon_errs,
    }
    det_hash = _write_json(study / "determinism.json", det)

    # --- Stage 1a / 1b milestone split (do not over-claim) ---
    evaluation_entry_ok = bool(
        args.run_e2e
        and online_n_eligible > 0
        and int(
            (e2e_meta.get("online_hook_counters_A1") or {}).get("hook_eligible") or 0
        )
        == 0
    )
    milestones = classify_stage1_milestones(
        evaluation_entry_ok=evaluation_entry_ok,
        frozen_policy_null_effect=online_vacuous,
        activation_action_path_ok=activation_ok,
        force_reject_path_ok=force_reject_ok,
        online_baudit_ok=False,
        strict_a0_identity_ok=a0_strict_ok,
        soft_a0_identity_ok=a0_soft_ok if args.run_e2e else None,
        determinism_repeated_ok=False,
        runtime_overhead_ok=False,
    )

    if not args.run_e2e:
        status = "online_blocked__offline_events_ready"
        e2e_safe = "no"
    else:
        # Freeze-policy safety headline only (null effect under production gates).
        # Uses soft A0 only for narrative; does not claim strict identity pass.
        status = (
            "online_effect_neutral_but_safe__vacuous_online_thr"
            if online_vacuous
            else classify_e2e_status(
                hook_off_identity_ok=True,  # A1 ran; strict A0 tracked separately
                n_rejected=online_n_rejected,
                metrics_delta=metrics_delta,
                determinism_ok=len(recon_errs) == 0,
                runtime_ok=True,
            )
        )
        e2e_safe = (
            "yes"
            if online_vacuous
            or (
                abs(metrics_delta.get("IDF1", 0.0)) < 1e-9
                and abs(metrics_delta.get("AssA", 0.0)) < 1e-9
            )
            else "no"
        )
        if activation_ok is True and force_reject_ok is True:
            status = "stage1b_action_path_proven__stage1_overall_still_open"
        elif activation_ok is False or force_reject_ok is False:
            status = "stage1b_action_path_failed"
            e2e_safe = "no" if force_reject_ok is False else e2e_safe

    summary = {
        "study_id": study.name,
        "stage": "M-B1 Stage 1",
        "candidate_id": policy.candidate_id,
        "policy_path": str(policy.path),
        "policy_file_hash": policy.file_hash,
        "e2e_safe_for_default_off": e2e_safe,
        "classification": status,
        "stage1_milestones": milestones,
        "stage1_overall": milestones["stage1_overall"],
        "stage1a_evaluation_entry": milestones["stage1a_evaluation_entry"],
        "stage1b_action_path": milestones["stage1b_action_path"],
        "headline_claim_allowed": milestones["headline_claim_allowed"],
        "offline_counts": counts,
        "reconciliation_errors": recon_errs,
        "parquet_written": parquet_ok,
        "metrics_delta_B_minus_A1": metrics_delta if args.run_e2e else {},
        "online_hook_eligible": online_n_eligible if args.run_e2e else None,
        "online_hook_rejected": online_n_rejected if args.run_e2e else None,
        "a0_identity_strict": a0_strict_ok,
        "a0_identity_soft": a0_soft_ok if args.run_e2e else None,
        "default_off_compatibility": a0_identity.get("default_off_compatibility"),
        "activation_action_path_ok": activation_ok,
        "force_reject_path_ok": force_reject_ok,
        "e2e": {
            "ran": e2e_meta.get("ran"),
            "A1_seconds": a1_sec,
            "B_seconds": b_sec,
            "A1_metrics": (e2e_meta.get("A1") or {}).get("metrics"),
            "B_metrics": (e2e_meta.get("B") or {}).get("metrics"),
            "online_hook_counters_B": e2e_meta.get("online_hook_counters_B"),
            "online_hook_counters_A1": e2e_meta.get("online_hook_counters_A1"),
            "a0_identity": a0_identity,
            "online_vacuous_policy": online_vacuous,
            "controls": {
                "ran": control_meta.get("ran"),
                "activation_checks": control_meta.get("activation_checks"),
                "force_reject_checks": control_meta.get("force_reject_checks"),
            },
        },
        "forbidden_work": [
            "rule search",
            "threshold sweep",
            "zone/gap atoms",
            "production preset change",
            "claim Stage 1 CLOSED from evaluation-entry alone",
        ],
    }
    sum_hash = _write_json(study / "summary.json", summary)
    a1_m = (e2e_meta.get("A1") or {}).get("metrics") or {}
    b_m = (e2e_meta.get("B") or {}).get("metrics") or {}
    act_chk = control_meta.get("activation_checks") or {}
    fr_chk = control_meta.get("force_reject_checks") or {}
    (study / "summary.md").write_text(
        "\n".join(
            [
                f"# M-B1 Stage 1 hook A/B — `{study.name}`",
                "",
                f"- freeze candidate_id: `{policy.candidate_id}`",
                f"- **stage1_overall: `{milestones['stage1_overall']}`**",
                f"- stage1a_evaluation_entry: `{milestones['stage1a_evaluation_entry']}`",
                f"- stage1b_action_path: `{milestones['stage1b_action_path']}`",
                f"- frozen_policy_online_relevance: "
                f"`{milestones['frozen_policy_online_relevance']}`",
                f"- e2e_safe_for_default_off (freeze B null-effect): **{e2e_safe}**",
                f"- classification: `{status}`",
                f"- default-off compatibility: "
                f"`{a0_identity.get('default_off_compatibility')}` "
                f"(strict={a0_strict_ok}, soft={a0_soft_ok})",
                f"- online B eligible/rejected: `{online_n_eligible}/{online_n_rejected}`",
                f"- activation control pass: `{activation_ok}` {act_chk}",
                f"- force-reject control pass: `{force_reject_ok}` {fr_chk}",
                f"- online B-audit: `{milestones['online_baudit']}`",
                f"- determinism repeated-run: `{milestones['determinism_repeated_run']}`",
                f"- runtime overhead contract: `{milestones['runtime_overhead']}`",
                "",
                f"**Allowed claim:** {milestones['headline_claim_allowed']}",
                "",
                "## Metrics (A1 hook-off vs B frozen)",
                "",
                f"- A1: IDF1={a1_m.get('IDF1')} AssA={a1_m.get('AssA')} "
                f"HOTA={a1_m.get('HOTA')} MOTA={a1_m.get('MOTA')} IDs={a1_m.get('IDs')}",
                f"- B:  IDF1={b_m.get('IDF1')} AssA={b_m.get('AssA')} "
                f"HOTA={b_m.get('HOTA')} MOTA={b_m.get('MOTA')} IDs={b_m.get('IDs')}",
                f"- Δ(B−A1): {metrics_delta}",
                "",
                "See `summary.json`, `ab_metrics.json`, and arm dirs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "study_id": study.name,
        "git_commit": _git_commit(),
        "candidate_id": policy.candidate_id,
        "policy_path": str(policy.path),
        "policy_hash": policy.file_hash,
        "sequence_set": "MOT17-SDP-7" if not args.sequences else args.sequences,
        "candidate_universe_identity": "offline_gt_valid_pairs"
        if offline
        else "e2e_only",
        "hook_flag_state": {
            "offline_events_only": args.offline_events_only,
            "run_e2e": args.run_e2e,
        },
        "audit_mode": "offline_full_table" if offline else "none",
        "artifact_hashes": {
            **det["summary_artifact_hashes"],
            "determinism": det_hash,
            "summary": sum_hash,
            "event_table": ev_hash,
        },
    }
    _write_json(study / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    return 0 if not recon_errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
