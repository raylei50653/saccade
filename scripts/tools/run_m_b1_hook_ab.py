#!/usr/bin/env python3
"""Stage 1 A/B runner scaffold: frozen portable OR-tail online hook.

Arms (plan §9):
  A1 — new code, hook disabled (must match production baseline identity)
  B  — frozen hook enabled
  B-audit — hook + full event audit (runtime not mixed into policy overhead)

Does **not** search thresholds or remodel policy.

Usage:
  # Dry: loader + offline event table + summary skeleton (no e2e)
  uv run python scripts/tools/run_m_b1_hook_ab.py \\
    --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \\
    --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \\
    --study-dir out/signal_study/m_b1_hook_ab_<stamp> \\
    --offline-events-only

  # Full online A/B (requires GPU + MOT17 data + rebuilt tracking ext):
  uv run python scripts/tools/run_m_b1_hook_ab.py \\
    --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \\
    --study-dir out/signal_study/m_b1_hook_ab_<stamp> \\
    --run-e2e

Contract:
  docs/modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md §Stage 1
"""

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
    classify_e2e_status,
    derive_atom_summary,
    evaluate_policy,
    fire_class_counts,
    load_portable_policy,
    reconcile_fire_classes,
    snapshot_policy,
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
    cmd = [
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
        cmd.extend(["--sequences", sequences])
    if policy_path is not None:
        cmd.extend(["--research-portable-or-tail-policy", str(policy_path)])
        if audit:
            cmd.append("--research-portable-or-tail-audit")
            if audit_dir is not None:
                cmd.extend(["--research-portable-or-tail-audit-dir", str(audit_dir)])
    cmd.extend(extra_args)
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
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"e2e arm {label} failed rc={proc.returncode}: {proc.stderr[-2000:]}"
        )
    return {
        "label": label,
        "output_dir": str(output_dir),
        "seconds": (t1 - t0).total_seconds(),
        "returncode": proc.returncode,
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
    ap.add_argument("--run-e2e", action="store_true", help="Run A1 vs B e2e arms")
    ap.add_argument(
        "--sequences",
        default=None,
        help="Optional sequence subset for e2e (default: full SDP set from preset)",
    )
    ap.add_argument("extra", nargs="*", help="Extra args forwarded to mot17.py")
    args = ap.parse_args()

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
    if args.run_e2e:
        a1_dir = study / "e2e_A1_hook_off"
        b_dir = study / "e2e_B_hook_on"
        e2e_meta = {
            "ran": True,
            "A1": run_e2e_arm(
                label="A1",
                output_dir=a1_dir,
                policy_path=None,
                audit=False,
                audit_dir=None,
                sequences=args.sequences,
                extra_args=list(args.extra),
            ),
            "B": run_e2e_arm(
                label="B",
                output_dir=b_dir,
                policy_path=policy.path,
                audit=False,
                audit_dir=None,
                sequences=args.sequences,
                extra_args=list(args.extra),
            ),
        }

    # Metrics / classification placeholders until e2e metrics are parsed.
    ab_metrics = {
        "note": "Fill from TrackEval after --run-e2e; offline replay metrics below",
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
    }
    ab_hash = _write_json(study / "ab_metrics.json", ab_metrics)

    runtime = {
        "hook_disabled_overhead": None,
        "hook_enabled_policy_overhead": None,
        "audit_enabled_overhead": None,
        "note": "Populate from e2e wall times / nvtx when available",
        "e2e_seconds": {
            k: e2e_meta.get(k, {}).get("seconds")
            for k in ("A1", "B")
            if e2e_meta.get("ran")
        },
    }
    rt_hash = _write_json(study / "runtime.json", runtime)

    det = {
        "hook_off_baseline_compatibility_hash": None,
        "hook_on_repeated_run_hashes": [],
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

    status = classify_e2e_status(
        hook_off_identity_ok=not args.run_e2e,  # unknown until A1==A0 checked
        n_rejected=int(counts.get("n_rejected", 0)),
        metrics_delta={"IDF1": 0.0} if not args.run_e2e else {},
        determinism_ok=len(recon_errs) == 0,
    )
    if not args.run_e2e:
        status = "online_blocked__offline_events_ready"
        e2e_safe = "no"
    else:
        e2e_safe = "pending_metrics_parse"

    summary = {
        "study_id": study.name,
        "stage": "M-B1 Stage 1",
        "candidate_id": policy.candidate_id,
        "policy_path": str(policy.path),
        "policy_file_hash": policy.file_hash,
        "e2e_safe_for_default_off": e2e_safe,
        "classification": status,
        "offline_counts": counts,
        "reconciliation_errors": recon_errs,
        "parquet_written": parquet_ok,
        "e2e": e2e_meta,
        "forbidden_work": [
            "rule search",
            "threshold sweep",
            "zone/gap atoms",
            "production preset change",
        ],
    }
    sum_hash = _write_json(study / "summary.json", summary)
    (study / "summary.md").write_text(
        "\n".join(
            [
                f"# M-B1 Stage 1 hook A/B — `{study.name}`",
                "",
                f"- candidate_id: `{policy.candidate_id}`",
                f"- policy_hash: `{policy.file_hash[:16]}…`",
                f"- e2e_safe_for_default_off: **{e2e_safe}**",
                f"- classification: `{status}`",
                f"- offline n_rejected: `{counts.get('n_rejected')}`",
                f"- recon_errors: `{recon_errs}`",
                f"- e2e ran: `{e2e_meta.get('ran')}`",
                "",
                "See `summary.json` and full tables in this directory.",
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
