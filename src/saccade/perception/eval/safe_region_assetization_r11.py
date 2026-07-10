"""R1.1 — Transfer Failure Attribution Pack.

Explains why L2/L3 pooled hard-safe fits hurt held-out GT under sequence LOO.
No new model family, grammar, signals, or optimizer retuning.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from saccade.perception.eval.d_online_stage2 import write_csv, write_json
from saccade.perception.eval.safe_region_assetization_r1 import (
    BEAM_TOP_M,
    SIGNAL_FAMILY,
    TERMINAL_B,
    BasisEntry,
    build_basis_registry,
    evaluate_weights,
    fit_sparse_nn_combinatorial,
    load_cohort_bundle,
    select_candidate_columns,
    _sha256_file,
)

TASK_NAME = "safe_region_assetization_r11_transfer_failure"
# Representative models for deep attribution (fixed; not tuned for LOO)
FOCUS_SPECS: tuple[tuple[str, int, int], ...] = (
    # family, order_max, K
    ("L2_sparse_nn_singleton", 1, 2),
    ("L2_sparse_nn_singleton", 1, 5),
    ("L3_sparse_nn_with_and", 2, 2),
    ("L3_sparse_nn_with_and", 2, 5),
)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return float(len(a & b) / len(u))


def _active_ids(w: np.ndarray, entries: Sequence[BasisEntry]) -> list[str]:
    return [entries[i].basis_id for i, wi in enumerate(w) if wi > 1e-8]


def _active_set(w: np.ndarray) -> set[int]:
    return {i for i, wi in enumerate(w) if wi > 1e-8}


def _pure_safe_indices(
    entries: Sequence[BasisEntry],
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    *,
    order_max: int,
    row_mask: np.ndarray | None = None,
) -> list[int]:
    """Bases with zero GT/unknown and ≥1 neg under the given label slice."""
    if row_mask is None:
        row_mask = np.ones(len(y), dtype=bool)
    out: list[int] = []
    for i, e in enumerate(entries):
        if e.order > order_max:
            continue
        phi = Phi[row_mask, i].astype(bool)
        n_neg = int(np.sum(phi & (y[row_mask] == 1)))
        n_gt = int(np.sum(phi & (y[row_mask] == 0)))
        # unknown always global firewall (selected unresolved)
        n_unk = int(e.phi_unknown.sum()) if len(Phi_u) else 0
        if n_gt == 0 and n_unk == 0 and n_neg > 0:
            out.append(i)
    return out


def _fit_model(
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    entries: Sequence[BasisEntry],
    *,
    order_max: int,
    K: int,
    row_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if row_mask is None:
        Phi_fit, y_fit = Phi, y
    else:
        Phi_fit, y_fit = Phi[row_mask], y[row_mask]
    cand = select_candidate_columns(
        entries, y_fit, order_max=order_max, top_m=BEAM_TOP_M
    )
    # Restrict candidate purity signal to train slice via Phi/y already sliced
    # but BasisEntry.n_* are global — combinatorial uses Phi columns only.
    fit = fit_sparse_nn_combinatorial(Phi_fit, Phi_u, y_fit, K=K, active_cols=cand)
    if not fit.get("success") or fit.get("w") is None:
        return {
            "success": False,
            "reason": fit.get("reason", "blocked"),
            "w": None,
            "tau": None,
        }
    w = np.asarray(fit["w"], dtype=float)
    w = np.where(w > 1e-8, w, 0.0)
    if w.sum() > 0:
        w = w / w.sum()
    tau = float(fit["tau"])
    return {
        "success": True,
        "reason": fit.get("reason", "ok"),
        "w": w,
        "tau": tau,
        "active_ids": _active_ids(w, entries),
        "active_idx": sorted(_active_set(w)),
        "cand": cand,
    }


def _event_attribution(
    *,
    model_id: str,
    hold: str,
    w: np.ndarray,
    tau: float,
    Phi_row: np.ndarray,
    y_i: int,
    event_id: str,
    row_index: int,
    entries: Sequence[BasisEntry],
    role: str,
) -> dict[str, Any]:
    score = float(Phi_row @ w)
    pred = int(score >= tau - 1e-15)
    firing = []
    contrib = []
    for j, wi in enumerate(w):
        if wi <= 1e-8:
            continue
        if Phi_row[j] > 0.5:
            firing.append(entries[j].basis_id)
            contrib.append(
                {"basis_id": entries[j].basis_id, "weight": float(wi), "phi": 1.0}
            )
    return {
        "model_id": model_id,
        "hold_out_sequence": hold,
        "row_index": row_index,
        "event_id": event_id,
        "y": int(y_i),
        "role": role,  # hold_gt_hurt | hold_neg_captured | hold_gt_safe | hold_neg_missed
        "score": score,
        "tau": tau,
        "pred": pred,
        "margin_to_tau": float(score - tau),
        "n_active_firing": len(firing),
        "firing_basis_ids": json.dumps(firing),
        "contributions_json": json.dumps(contrib),
    }


def run_attribution_for_spec(
    *,
    family: str,
    order_max: int,
    K: int,
    entries: Sequence[BasisEntry],
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
    event_ids: Sequence[str],
) -> dict[str, Any]:
    model_id = f"{family}:K{K}"
    seqs = sorted(set(str(s) for s in sequences))

    # Pooled fit (reference)
    pooled = _fit_model(Phi, Phi_u, y, entries, order_max=order_max, K=K)
    pooled_active = set(pooled.get("active_ids") or [])

    fold_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    stability_parts: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    basis_role_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    fold_actives: list[set[str]] = []
    fold_weights: list[dict[str, float]] = []
    fold_taus: list[float] = []

    global_pure = set(
        entries[i].basis_id
        for i in _pure_safe_indices(entries, Phi, Phi_u, y, order_max=order_max)
    )

    total_hold_gt_hurt = 0
    total_hold_neg = 0
    folds_with_gt_hurt = 0
    role_reversal_count = 0
    train_prod_hold_retention_num = 0
    train_prod_hold_retention_den = 0

    for hold in seqs:
        train_m = sequences != hold
        hold_m = sequences == hold
        fit = _fit_model(
            Phi, Phi_u, y, entries, order_max=order_max, K=K, row_mask=train_m
        )
        train_pure = set(
            entries[i].basis_id
            for i in _pure_safe_indices(
                entries, Phi, Phi_u, y, order_max=order_max, row_mask=train_m
            )
        )
        registry_rows.append(
            {
                "model_id": model_id,
                "hold_out_sequence": hold,
                "n_global_pure_safe_basis": len(global_pure),
                "n_train_pure_safe_basis": len(train_pure),
                "jaccard_global_vs_train_pure": _jaccard(global_pure, train_pure),
                "n_only_global": len(global_pure - train_pure),
                "n_only_train": len(train_pure - global_pure),
                "only_global_json": json.dumps(sorted(global_pure - train_pure)[:20]),
                "only_train_json": json.dumps(sorted(train_pure - global_pure)[:20]),
            }
        )

        if not fit.get("success") or fit["w"] is None:
            fold_rows.append(
                {
                    "model_id": model_id,
                    "hold_out_sequence": hold,
                    "status": fit.get("reason", "blocked"),
                    "hold_gt_hurt": "",
                    "hold_n_neg_captured": "",
                }
            )
            fold_actives.append(set())
            continue

        w = fit["w"]
        tau = float(fit["tau"])
        active_ids = set(fit["active_ids"])
        fold_actives.append(active_ids)
        fold_weights.append(
            {
                bid: float(w[i])
                for i, bid in enumerate([e.basis_id for e in entries])
                if w[i] > 1e-8
            }
        )
        fold_taus.append(tau)

        # train / hold eval
        ev_tr = evaluate_weights(
            w, tau, Phi[train_m], Phi_u, y[train_m], sequences[train_m]
        )
        scores_h = Phi[hold_m] @ w
        pred_h = scores_h >= tau - 1e-15
        y_h = y[hold_m]
        idx_h = np.where(hold_m)[0]
        hold_gt_hurt = int(np.sum(pred_h & (y_h == 0)))
        hold_neg = int(np.sum(pred_h & (y_h == 1)))
        total_hold_gt_hurt += hold_gt_hurt
        total_hold_neg += hold_neg
        if hold_gt_hurt > 0:
            folds_with_gt_hurt += 1

        # margins
        if np.any(y[train_m] == 0):
            train_gt_margin = float(np.min(tau - (Phi[train_m] @ w)[y[train_m] == 0]))
        else:
            train_gt_margin = float("nan")
        if np.any(y_h == 0):
            hold_gt_margin = float(np.min(tau - scores_h[y_h == 0]))
        else:
            hold_gt_margin = float("nan")
        if np.any(pred_h & (y_h == 1)):
            hold_prod_margin = float(np.min(scores_h[pred_h & (y_h == 1)] - tau))
        else:
            hold_prod_margin = float("nan")
        if ev_tr["n_neg_captured"] > 0 and math_isfinite(
            ev_tr.get("captured_negative_margin_min")
        ):
            train_prod_margin = float(ev_tr["captured_negative_margin_min"])
        else:
            train_prod_margin = float("nan")

        margin_rows.append(
            {
                "model_id": model_id,
                "hold_out_sequence": hold,
                "train_gt_safety_margin": train_gt_margin,
                "hold_gt_margin": hold_gt_margin,
                "train_productive_margin_min": train_prod_margin,
                "hold_productive_margin_min": hold_prod_margin,
                "gt_margin_contraction": (
                    float(train_gt_margin - hold_gt_margin)
                    if math_isfinite(train_gt_margin) and math_isfinite(hold_gt_margin)
                    else float("nan")
                ),
                "tau": tau,
                "train_n_neg_captured": ev_tr["n_neg_captured"],
                "hold_gt_hurt": hold_gt_hurt,
                "hold_n_neg_captured": hold_neg,
            }
        )

        # sequence dominance on train productive captures
        train_pred = (Phi[train_m] @ w) >= tau - 1e-15
        y_tr = y[train_m]
        seq_tr = sequences[train_m]
        neg_by_seq: dict[str, int] = defaultdict(int)
        for s, cap in zip(seq_tr.astype(str), train_pred & (y_tr == 1)):
            if cap:
                neg_by_seq[s] += 1
        n_prod_train = sum(neg_by_seq.values())
        dom = max(neg_by_seq.values()) / n_prod_train if n_prod_train else float("nan")

        fold_rows.append(
            {
                "model_id": model_id,
                "hold_out_sequence": hold,
                "status": "ok",
                "active_basis_ids": json.dumps(sorted(active_ids)),
                "n_active": len(active_ids),
                "jaccard_vs_pooled": _jaccard(active_ids, pooled_active),
                "tau": tau,
                "train_n_neg_captured": ev_tr["n_neg_captured"],
                "train_gt_hurt": ev_tr["gt_hurt"],
                "hold_gt_hurt": hold_gt_hurt,
                "hold_n_neg_captured": hold_neg,
                "hold_n_gt": int(np.sum(y_h == 0)),
                "hold_n_neg": int(np.sum(y_h == 1)),
                "train_sequence_dominance": dom,
                "train_productive_sequences": json.dumps(
                    sorted(s for s, v in neg_by_seq.items() if v > 0)
                ),
            }
        )

        # per-basis roles on train vs hold
        for j in fit["active_idx"]:
            e = entries[j]
            phi_tr = Phi[train_m, j].astype(bool)
            phi_h = Phi[hold_m, j].astype(bool)
            tr_neg = int(np.sum(phi_tr & (y_tr == 1)))
            tr_gt = int(np.sum(phi_tr & (y_tr == 0)))
            h_neg = int(np.sum(phi_h & (y_h == 1)))
            h_gt = int(np.sum(phi_h & (y_h == 0)))
            reversal = int(tr_neg > 0 and h_gt > 0)
            role_reversal_count += reversal
            # retention: same basis fires on holdout GT? (bad) or holdout neg?
            basis_role_rows.append(
                {
                    "model_id": model_id,
                    "hold_out_sequence": hold,
                    "basis_id": e.basis_id,
                    "order": e.order,
                    "weight": float(w[j]),
                    "train_n_neg_support": tr_neg,
                    "train_n_gt_support": tr_gt,
                    "hold_n_neg_support": h_neg,
                    "hold_n_gt_support": h_gt,
                    "role_reversal_train_neg_hold_gt": reversal,
                    "fires_on_hold_gt_hurt": int(
                        h_gt > 0 and np.any(pred_h & (y_h == 0) & phi_h)
                    ),
                }
            )

        # train productive support retention on holdout:
        # among active bases that capture ≥1 train negative, fraction that also
        # fire on ≥1 holdout negative.
        for j in fit["active_idx"]:
            tr_cap = train_pred & (y_tr == 1) & Phi[train_m, j].astype(bool)
            if int(tr_cap.sum()) <= 0:
                continue
            train_prod_hold_retention_den += 1
            hold_phi_j = Phi[hold_m, j].astype(bool)
            if int(np.sum(hold_phi_j & (y_h == 1))) > 0:
                train_prod_hold_retention_num += 1

        # event-level holdout attribution
        for local_i, global_i in enumerate(idx_h):
            yi = int(y[global_i])
            pr = bool(pred_h[local_i])
            if yi == 0 and pr:
                role = "hold_gt_hurt"
            elif yi == 1 and pr:
                role = "hold_neg_captured"
            elif yi == 0 and not pr:
                role = "hold_gt_safe"
            else:
                role = "hold_neg_missed"
            if role in ("hold_gt_hurt", "hold_neg_captured"):
                event_rows.append(
                    _event_attribution(
                        model_id=model_id,
                        hold=hold,
                        w=w,
                        tau=tau,
                        Phi_row=Phi[global_i],
                        y_i=yi,
                        event_id=str(event_ids[global_i]),
                        row_index=int(global_i),
                        entries=entries,
                        role=role,
                    )
                )

    # pairwise fold Jaccard
    for i, hold_i in enumerate(seqs):
        for j, hold_j in enumerate(seqs):
            if j <= i:
                continue
            if i >= len(fold_actives) or j >= len(fold_actives):
                continue
            overlap_rows.append(
                {
                    "model_id": model_id,
                    "fold_a": hold_i,
                    "fold_b": hold_j,
                    "jaccard": _jaccard(fold_actives[i], fold_actives[j]),
                    "n_shared": len(fold_actives[i] & fold_actives[j]),
                    "n_union": len(fold_actives[i] | fold_actives[j]),
                    "set_a": json.dumps(sorted(fold_actives[i])),
                    "set_b": json.dumps(sorted(fold_actives[j])),
                }
            )

    # stability: mean pairwise Jaccard, tau std, weight presence frequency
    if len(fold_actives) >= 2:
        js = [r["jaccard"] for r in overlap_rows if r["model_id"] == model_id]
        mean_j = float(np.mean(js)) if js else float("nan")
    else:
        mean_j = float("nan")
    tau_std = float(np.std(fold_taus)) if fold_taus else float("nan")
    tau_mean = float(np.mean(fold_taus)) if fold_taus else float("nan")
    # basis selection frequency
    freq: dict[str, int] = defaultdict(int)
    for s in fold_actives:
        for bid in s:
            freq[bid] += 1
    n_folds_ok = sum(1 for s in fold_actives if s)
    for bid, c in sorted(freq.items()):
        stability_parts.append(
            {
                "model_id": model_id,
                "basis_id": bid,
                "n_folds_selected": c,
                "selection_rate": c / n_folds_ok if n_folds_ok else float("nan"),
                "in_pooled": int(bid in pooled_active),
            }
        )

    retention = (
        train_prod_hold_retention_num / train_prod_hold_retention_den
        if train_prod_hold_retention_den
        else float("nan")
    )

    summary = {
        "model_id": model_id,
        "family": family,
        "K": K,
        "order_max": order_max,
        "pooled_success": int(pooled.get("success", False)),
        "pooled_active": sorted(pooled_active),
        "pooled_n_neg": None,
        "n_folds": len(seqs),
        "n_folds_with_gt_hurt": folds_with_gt_hurt,
        "total_hold_gt_hurt": total_hold_gt_hurt,
        "total_hold_neg_captured": total_hold_neg,
        "mean_pairwise_active_jaccard": mean_j,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "role_reversal_events": role_reversal_count,
        "train_prod_basis_holdout_neg_retention": retention,
        "mean_jaccard_global_vs_train_pure": float(
            np.mean([r["jaccard_global_vs_train_pure"] for r in registry_rows])
        )
        if registry_rows
        else float("nan"),
    }
    if pooled.get("success") and pooled.get("w") is not None:
        ev_p = evaluate_weights(
            pooled["w"], float(pooled["tau"]), Phi, Phi_u, y, sequences
        )
        summary["pooled_n_neg"] = ev_p["n_neg_captured"]
        summary["pooled_n_productive_sequences"] = ev_p["n_productive_sequences"]

    return {
        "model_id": model_id,
        "summary": summary,
        "folds": fold_rows,
        "overlap": overlap_rows,
        "stability": stability_parts,
        "events": event_rows,
        "basis_roles": basis_role_rows,
        "margins": margin_rows,
        "registry": registry_rows,
        "fold_actives": fold_actives,
        "pooled_active": pooled_active,
    }


def math_isfinite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def assign_failure_taxonomy(
    model_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Primary + up to two secondary failure codes from attribution evidence."""
    # Aggregate across focus models (emphasize L3 where LOO hurt is material)
    l3 = [m for m in model_results if "L3" in m["model_id"]]
    use = l3 if l3 else list(model_results)
    if not use:
        return {
            "primary": "F5",
            "secondary": [],
            "scores": {},
            "rationale": "no successful model attribution",
            "decision_mapping": "F5 dominant → inconclusive; no stronger model conclusion",
        }

    mean_j = float(
        np.nanmean([m["summary"]["mean_pairwise_active_jaccard"] for m in use])
    )
    mean_reg_j = float(
        np.nanmean([m["summary"]["mean_jaccard_global_vs_train_pure"] for m in use])
    )
    total_rev = sum(int(m["summary"]["role_reversal_events"]) for m in use)
    total_gt_hurt = sum(int(m["summary"]["total_hold_gt_hurt"]) for m in use)
    folds_hurt = sum(int(m["summary"]["n_folds_with_gt_hurt"]) for m in use)
    n_folds = sum(int(m["summary"]["n_folds"]) for m in use)
    ret = float(
        np.nanmean(
            [m["summary"]["train_prod_basis_holdout_neg_retention"] for m in use]
        )
    )
    # sequence dominance: mean over folds
    doms = []
    for m in use:
        for f in m["folds"]:
            if f.get("status") == "ok" and math_isfinite(
                f.get("train_sequence_dominance")
            ):
                doms.append(float(f["train_sequence_dominance"]))
    mean_dom = float(np.mean(doms)) if doms else float("nan")

    # margin: fraction of folds with hold_gt_margin < 0 while train_gt_margin > 0
    margin_fail_folds = 0
    margin_folds = 0
    for m in use:
        for r in m["margins"]:
            margin_folds += 1
            if math_isfinite(r.get("train_gt_safety_margin")) and math_isfinite(
                r.get("hold_gt_margin")
            ):
                if (
                    float(r["train_gt_safety_margin"]) > 0
                    and float(r["hold_gt_margin"]) < 0
                ):
                    margin_fail_folds += 1

    scores = {
        "F1": 0.0,  # coordinate transport: train margin ok, hold margin collapses
        "F2": 0.0,  # basis identity instability
        "F3": 0.0,  # semantic/sign conflict / role reversal
        "F4": 0.0,  # single-sequence islands
        "F5": 0.0,  # insufficient support
    }

    # F1: margin contraction with hold GT hurt
    if margin_folds:
        scores["F1"] = 100.0 * margin_fail_folds / margin_folds
    if total_gt_hurt > 0:
        scores["F1"] = max(scores["F1"], 40.0)

    # F2: low active-set Jaccard across folds
    if math_isfinite(mean_j):
        scores["F2"] = 100.0 * (1.0 - mean_j)
    # registry train vs global divergence contributes lightly to F2
    if math_isfinite(mean_reg_j):
        scores["F2"] = 0.7 * scores["F2"] + 0.3 * (100.0 * (1.0 - mean_reg_j))

    # F3: role reversal train-neg / hold-gt
    # normalize by number of basis-role rows approx
    n_role_rows = sum(len(m["basis_roles"]) for m in use) or 1
    scores["F3"] = min(100.0, 100.0 * total_rev / max(1, n_role_rows) * 5.0)
    if total_rev >= 3 and total_gt_hurt > 0:
        scores["F3"] = max(scores["F3"], 55.0)

    # F4: high sequence dominance on train productive captures
    if math_isfinite(mean_dom):
        scores["F4"] = 100.0 * mean_dom
    # low holdout productive retention strengthens islands
    if math_isfinite(ret) and ret < 0.5:
        scores["F4"] = max(scores["F4"], 50.0)

    # F5: tiny productive mass
    pooled_negs = [m["summary"].get("pooled_n_neg") or 0 for m in use]
    max_pool = max(int(x) for x in pooled_negs) if pooled_negs else 0
    if max_pool < 3:
        scores["F5"] = 80.0
    elif max_pool < 5:
        scores["F5"] = 40.0
    else:
        scores["F5"] = 10.0

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    primary = ranked[0][0]
    secondary = [c for c, s in ranked[1:] if s >= 0.55 * ranked[0][1] and s >= 35.0][:2]

    mapping = {
        "F1": (
            "F1 dominant → consider relative/normalized coordinate transport spec "
            "(narrow follow-up only)"
        ),
        "F2": "F2 dominant → close fixed-basis grammar path",
        "F3": (
            "F3 dominant → frozen signals lack invariant semantics; "
            "conditional applicability or new signal family"
        ),
        "F4": (
            "F4 dominant → treat regions as sequence-conditioned islands; no global gate"
        ),
        "F5": (
            "F5 dominant → inconclusive due support; do not infer stronger model conclusion"
        ),
    }

    rationale_bits = [
        f"mean pairwise active Jaccard={mean_j:.3f}",
        f"mean global↔train pure Jaccard={mean_reg_j:.3f}",
        f"role_reversal_count={total_rev}",
        f"hold_gt_hurt_total={total_gt_hurt}",
        f"folds_with_gt_hurt={folds_hurt}/{n_folds}",
        f"mean_train_seq_dominance={mean_dom:.3f}",
        f"train_prod_hold_retention={ret:.3f}",
        f"margin_fail_folds={margin_fail_folds}/{margin_folds}",
        f"max_pooled_n_neg={max_pool}",
    ]

    return {
        "primary": primary,
        "secondary": secondary,
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "ranked": ranked,
        "rationale": "; ".join(rationale_bits),
        "decision_mapping": mapping[primary],
        "secondary_mappings": [mapping[s] for s in secondary],
        "evidence_emphasis": "L3 focus models" if l3 else "all focus models",
    }


def run_r11_study(
    *,
    q45_dir: Path,
    events_path: Path,
    out_dir: Path,
    study_id: str | None = None,
) -> dict[str, Any]:
    q45_dir = Path(q45_dir)
    events_path = Path(events_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    study_id = study_id or f"safe_region_assetization_r11_{ts}"

    thr_reg = json.loads(
        (q45_dir / "threshold_registry.json").read_text(encoding="utf-8")
    )
    atom_path = q45_dir / "atom_atlas.parquet"
    and_path = q45_dir / "pairwise_and_atlas.parquet"
    atom_df = (
        pd.read_parquet(atom_path)
        if atom_path.exists()
        else pd.read_csv(q45_dir / "atom_atlas.csv")
    )
    and_df = (
        pd.read_parquet(and_path)
        if and_path.exists()
        else pd.read_csv(q45_dir / "pairwise_and_atlas.csv")
    )
    # atom_df unused for fit but kept for provenance parity
    _ = atom_df

    cohort = load_cohort_bundle(events_path, thr_reg)
    entries, basis_rows, _aliases = build_basis_registry(cohort, thr_reg, and_df)
    Phi = np.column_stack([e.phi_primary for e in entries])
    Phi_u = np.column_stack([e.phi_unknown for e in entries])
    y = cohort.y
    sequences = cohort.sequences
    event_ids = [str(r.get("event_id", i)) for i, r in enumerate(cohort.primary)]

    all_folds: list[dict[str, Any]] = []
    all_overlap: list[dict[str, Any]] = []
    all_stability: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_roles: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    all_registry: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []

    for family, order_max, K in FOCUS_SPECS:
        res = run_attribution_for_spec(
            family=family,
            order_max=order_max,
            K=K,
            entries=entries,
            Phi=Phi,
            Phi_u=Phi_u,
            y=y,
            sequences=sequences,
            event_ids=event_ids,
        )
        model_results.append(res)
        model_summaries.append(res["summary"])
        all_folds.extend(res["folds"])
        all_overlap.extend(res["overlap"])
        all_stability.extend(res["stability"])
        all_events.extend(res["events"])
        all_roles.extend(res["basis_roles"])
        all_margins.extend(res["margins"])
        all_registry.extend(res["registry"])

    taxonomy = assign_failure_taxonomy(model_results)

    def _w(name: str, rows: list[dict[str, Any]]) -> None:
        write_csv(out_dir / name, rows)

    _w("fold_summary.csv", all_folds)
    _w("basis_overlap_jaccard.csv", all_overlap)
    _w("basis_selection_stability.csv", all_stability)
    _w("holdout_event_attribution.csv", all_events)
    _w("basis_role_reversal.csv", all_roles)
    _w("margin_contraction.csv", all_margins)
    _w("registry_global_vs_train.csv", all_registry)
    _w("model_attribution_summary.csv", model_summaries)

    write_json(out_dir / "failure_taxonomy.json", taxonomy)
    write_json(
        out_dir / "manifest.json",
        {
            "study_id": study_id,
            "task": TASK_NAME,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parent_r1_study": "safe_region_assetization_r1_20260710",
            "q45_dir": str(q45_dir),
            "events_path": str(events_path),
            "events_sha256": _sha256_file(events_path),
            "focus_specs": [
                {"family": f, "order_max": o, "K": k} for f, o, k in FOCUS_SPECS
            ],
            "loo_protocol": "transductive_globally_registered_basis_LOO",
            "non_goals": [
                "new_model_family",
                "new_grammar",
                "new_signals",
                "optimizer_tuning_for_loo",
                "hook_preset_production",
                "grammar_distillation",
                "evidence_ledger_promotion",
                "reopen_terminal_A",
            ],
            "terminal_b_retained": True,
            "signal_family": list(SIGNAL_FAMILY),
            "n_basis": len(basis_rows),
            "outputs": sorted(p.name for p in out_dir.iterdir()),
        },
    )
    summary = {
        "study_id": study_id,
        "task": TASK_NAME,
        "verdict_parent": "V-C",
        "terminal_b": TERMINAL_B,
        "failure_taxonomy": taxonomy,
        "model_summaries": model_summaries,
        "decision": {
            "primary": taxonomy["primary"],
            "secondary": taxonomy["secondary"],
            "mapping": taxonomy["decision_mapping"],
            "r2_still_unauthorized": True,
            "grammar_search_still_closed": True,
        },
        "research_acceptance_boundary": {
            "purpose": "convert V-C from outcome description into operable failure mechanism asset",
            "not": "rescue model / improve LOO fit",
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary
