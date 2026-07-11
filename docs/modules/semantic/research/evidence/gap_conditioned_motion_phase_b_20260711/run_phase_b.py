#!/usr/bin/env python3
"""Run the sealed A1--A8 Phase-B analysis for gap-conditioned motion.

This is deliberately an *analysis consumer*: it never fits a motion model,
rewrites E3, or selects on held-out rows.  The only threshold selection is A6
and it is made from the ``evaluation_role=train`` surface sealed in E3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2


REPO = Path(__file__).resolve().parents[6]
PACKET = Path(__file__).resolve().parent
E3 = (
    REPO
    / "docs/modules/semantic/research/evidence/gap_conditioned_motion_e3_signals_20260711"
)
E3_SCORES = E3 / "pair_fold_model_scores.csv.gz"
E3_MANIFEST = E3 / "manifest.json"
PAIRS = REPO / "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
FORENSIC = (
    REPO / "docs/modules/semantic/research/evidence/escape_tail_forensic_20260711"
)
DESIGN = (
    REPO
    / "docs/modules/semantic/research/gap_conditioned_motion_phase_b_design_20260711.md"
)
MODELS = (
    "M1P-GLOBAL-CV",
    "M2P-GLOBAL-OU-H270",
    "M2P-GLOBAL-OU-H90",
    "M2P-GLOBAL-OU-H30",
)
BINS = (
    ("1-10", 1, 10),
    ("11-30", 11, 30),
    ("31-60", 31, 60),
    ("61-150", 61, 150),
    ("151-300", 151, 300),
)
PRIMARY = (("1-10", 1, 10), ("11-26", 11, 26))
SUPPORT_LAYERS = (
    ("S_A", 1, 26),
    ("S_C2", 1, 60),
    ("S_B", 2, 45),
    ("all_gap", 1, 300),
)
QUANTILES = (50, 60, 70, 75, 80, 85, 90, 95)
EPSILON = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable(value: Any) -> Any:
    """Make JSON output deterministic and avoid non-standard NaN tokens."""
    if isinstance(value, dict):
        return {str(k): stable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def auc_gt_low_score(frame: pd.DataFrame, column: str) -> float:
    positives = int(frame.gt_match.sum())
    negatives = len(frame) - positives
    if not positives or not negatives:
        return float("nan")
    # pandas average ranking is tie-aware.  A low mismatch/energy is positive.
    ranks = (-frame[column]).rank(method="average")
    return float(
        (ranks[frame.gt_match].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def gap_name(
    gap: pd.Series, bins: tuple[tuple[str, int, int], ...] = BINS
) -> pd.Series:
    result = pd.Series(index=gap.index, dtype="object")
    for name, low, high in bins:
        result.loc[(gap >= low) & (gap <= high)] = name
    if result.isna().any():
        raise ValueError("score rows outside frozen gap support")
    return result


def calibration(gt: pd.DataFrame) -> dict[str, Any]:
    q = gt.q_motion
    c90 = float((q <= chi2.ppf(0.90, 2)).mean()) if len(q) else float("nan")
    if c90 < 0.85:
        status = "under-dispersed"
    elif c90 > 0.95:
        status = "over-dispersed"
    else:
        status = "approximately-calibrated"
    return {
        "n_gt": len(gt),
        "dispersion_ratio": float(q.mean() / 2) if len(q) else float("nan"),
        "c50": float((q <= chi2.ppf(0.50, 2)).mean()) if len(q) else float("nan"),
        "c90": c90,
        "c95": float((q <= chi2.ppf(0.95, 2)).mean()) if len(q) else float("nan"),
        "calibration_class": status,
    }


def tail_cell(frame: pd.DataFrame, score: str) -> dict[str, Any]:
    threshold = float(frame[score].quantile(0.90))
    tail = frame[frame[score] >= threshold]
    base_rate = float(frame.gt_match.mean())
    tail_rate = float(tail.gt_match.mean())
    enrichment = tail_rate / base_rate if base_rate else float("nan")
    auc = auc_gt_low_score(frame, score)
    return {
        "n": len(frame),
        "gt": int(frame.gt_match.sum()),
        "score": score,
        "auc_gt_low_score": auc,
        "q90_threshold": threshold,
        "tail_n": len(tail),
        "tail_gt": int(tail.gt_match.sum()),
        "tail_gt_enrichment": enrichment,
        "role_reversal": bool(auc < 0.5 and enrichment > 1.0),
        "tail_gt_by_sequence": tail[tail.gt_match].groupby("seq").size().to_dict(),
    }


def m0_signal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen E1 M0 atoms on exactly ``frame``'s rows."""
    return pd.DataFrame(
        {
            "bridge_dist": frame["bridge_dist"],
            "speed_mismatch": (
                frame["lost_exit_speed"] - frame["cand_entry_speed"]
            ).abs(),
            "dir_cos": 1.0 - frame["dir_cos"],
            "resid_mean": 0.5 * (frame["fwd_resid"] + frame["bwd_resid"]),
        },
        index=frame.index,
    )


def separability_row(
    frame: pd.DataFrame, *, model_id: str, support_layer: str, gap_cell: str
) -> dict[str, Any]:
    """A5: all six low-score GT AUCs on one frozen support/cell row set."""
    m0 = m0_signal_columns(frame)
    return {
        "model_id": model_id,
        "support_layer": support_layer,
        "gap_cell": gap_cell,
        "n": len(frame),
        "gt": int(frame.gt_match.sum()),
        "fp": int((~frame.gt_match).sum()),
        "e_motion_auc": auc_gt_low_score(frame, "nll_motion"),
        "q_motion_auc": auc_gt_low_score(frame, "q_motion"),
        "bridge_dist_auc": auc_gt_low_score(
            m0.assign(gt_match=frame.gt_match), "bridge_dist"
        ),
        "speed_mismatch_auc": auc_gt_low_score(
            m0.assign(gt_match=frame.gt_match), "speed_mismatch"
        ),
        "dir_cos_auc": auc_gt_low_score(m0.assign(gt_match=frame.gt_match), "dir_cos"),
        "resid_mean_auc": auc_gt_low_score(
            m0.assign(gt_match=frame.gt_match), "resid_mean"
        ),
    }


def retention_pass(
    *, e_motion_auc: float, bridge_dist_auc: float, n_gt: int
) -> bool | None:
    return None if n_gt < 15 else bool(e_motion_auc >= bridge_dist_auc - 0.05)


def m2_dominates(
    *, retention: bool, held_out_nll_better: bool, calibration_mismatch: bool
) -> bool:
    return bool(retention and held_out_nll_better and not calibration_mismatch)


def verdict_partition(
    member_boxes: dict[str, dict[str, Any]], *, low_primary: bool
) -> tuple[str, str | None]:
    """Apply the sealed V4 → V2 → V1 → V5 priority partition."""
    if low_primary:
        return "V4", None
    for model in ("M2P-GLOBAL-OU-H270", "M2P-GLOBAL-OU-H90", "M2P-GLOBAL-OU-H30"):
        if member_boxes[model]["boxes_pass"] and member_boxes[model].get(
            "dominates_m1"
        ):
            return "V2", None
    if member_boxes["M1P-GLOBAL-CV"]["boxes_pass"]:
        return "V1", None
    passed_m2 = [model for model in MODELS[1:] if member_boxes[model]["boxes_pass"]]
    return (
        "V5",
        "a member passed all success boxes without a claimable verdict slot"
        if passed_m2
        else None,
    )


def cluster_stats(frame: pd.DataFrame, score: str, threshold: float) -> dict[str, Any]:
    gt = frame[frame.gt_match]
    clusters = gt.groupby(["seq", "lost_id"], sort=True)
    n = len(clusters)
    contained = sum(bool((group[score] > threshold).any()) for _, group in clusters)
    ucb = (
        1.0
        if n == 0
        else (
            1.0
            if contained == n
            else float(beta.ppf(0.95, contained + 1, n - contained))
        )
    )
    fp = frame[~frame.gt_match]
    return {
        "n_gt_clusters": n,
        "contained_gt_clusters": contained,
        "gt_leakage": contained / n if n else float("nan"),
        "gt_ucb95": ucb,
        "n_fp": len(fp),
        "fp_removed_count": int((fp[score] > threshold).sum()),
        "fp_removed": float((fp[score] > threshold).mean())
        if len(fp)
        else float("nan"),
    }


def select_a6(train: pd.DataFrame, score: str) -> dict[str, Any]:
    fp = train[~train.gt_match]
    choices = []
    for percentile in QUANTILES:
        threshold = (
            float(fp[score].quantile(percentile / 100)) if len(fp) else float("nan")
        )
        stats = cluster_stats(train, score, threshold)
        choices.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                **stats,
                "feasible": bool(stats["gt_ucb95"] <= EPSILON),
            }
        )
    feasible = [item for item in choices if item["feasible"]]
    if not feasible:
        return {"terminal": "NO_FEASIBLE_THRESHOLD", "choices": choices}
    chosen = max(feasible, key=lambda item: (item["fp_removed"], -item["percentile"]))
    return {"terminal": "SELECTED", "choices": choices, "selected": chosen}


def pooled_fp_removed(cells: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Pool captured FP counts, never independently weighted cell rates."""
    stats = [cell.get(key) for cell in cells]
    if not stats or any(item is None for item in stats):
        return None
    typed = [item for item in stats if item is not None]
    n_fp = sum(int(item["n_fp"]) for item in typed)
    removed = sum(int(item["fp_removed_count"]) for item in typed)
    return {
        "n_cells": len(typed),
        "n_fp": n_fp,
        "fp_removed_count": removed,
        "fp_removed": removed / n_fp if n_fp else float("nan"),
    }


def a6_no_thinner(
    *,
    safety: bool,
    fold_pools: list[dict[str, Any]],
    global_old: dict[str, Any] | None,
    global_new: dict[str, Any] | None,
) -> bool:
    if not safety or not fold_pools or global_old is None or global_new is None:
        return False
    if (
        not global_new["fp_removed"] > 0
        or global_new["fp_removed"] < global_old["fp_removed"]
    ):
        return False
    return all(
        pool["new"]["fp_removed"] >= 0.8 * pool["old"]["fp_removed"]
        for pool in fold_pools
    )


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(E3_MANIFEST.read_text())
    if manifest["status"] != "E3_SIGNALS_SEALED" or manifest["a1_a8_computed"]:
        raise ValueError("E3 input is not a sealed signal-only packet")
    if manifest["phase_b_design"]["content_sha256"] != sha256(DESIGN):
        raise ValueError("Phase-B design content differs from the sealed E3 provenance")
    scores = pd.read_csv(E3_SCORES)
    # E0/E1's frozen universe is U_relink_pair (`gt_valid=1`).  E3 preserved
    # the broader scoring surface for lineage, so make the analysis cut here
    # explicitly rather than silently relying on the baseline join to do it.
    scores = scores[scores.gt_valid.astype(bool)].copy()
    pairs = pd.read_csv(
        PAIRS,
        usecols=[
            "seq",
            "lost_id",
            "cand_id",
            "gt_valid",
            "bridge_dist",
            "lost_exit_speed",
            "cand_entry_speed",
            "dir_cos",
            "fwd_resid",
            "bwd_resid",
        ],
    )
    pairs = pairs[pairs.gt_valid.astype(bool)].drop(columns="gt_valid")
    keys = ["seq", "lost_id", "cand_id"]
    scores = scores.merge(pairs, on=keys, how="left", validate="many_to_one")
    if scores.bridge_dist.isna().any():
        raise ValueError("sealed E3 row did not join a frozen bridge_dist baseline")
    scores["gt_match"] = scores.gt_match.astype(bool)
    scores["gap_bin"] = gap_name(scores.gap)
    scores["primary_cell"] = pd.NA
    primary_mask = scores.gap.le(26)
    scores.loc[primary_mask, "primary_cell"] = gap_name(
        scores.loc[primary_mask, "gap"], PRIMARY
    )
    return scores, manifest


def load_selection_artifacts() -> dict[str, dict[str, Any]]:
    artifacts = {}
    for path in sorted((E3 / "selections").glob("*.json")):
        payload = json.loads(path.read_text())
        artifacts[payload["fold_id"]] = payload
    if set(artifacts) != {
        f"LOO::{sequence}"
        for sequence in (
            "MOT17-02-SDP",
            "MOT17-04-SDP",
            "MOT17-05-SDP",
            "MOT17-09-SDP",
            "MOT17-10-SDP",
            "MOT17-11-SDP",
            "MOT17-13-SDP",
        )
    }:
        raise ValueError("sealed E3 selection artifact map is incomplete")
    return artifacts


def compute(scores: pd.DataFrame, e3_manifest: dict[str, Any]) -> dict[str, Any]:
    held = scores[scores.evaluation_role.eq("held_out")].copy()
    train = scores[scores.evaluation_role.eq("train")].copy()
    if (
        not (held.seq == held.held_out_sequence).all()
        or (train.seq == train.held_out_sequence).any()
    ):
        raise ValueError("E3 role firewall violated")
    selections = load_selection_artifacts()
    a1, a2, a3, a4_rows, a5, a6, a7, a8 = [], [], [], [], [], [], [], []
    # A1/A2/A5 use the held-out surface only.
    for model in MODELS:
        model_rows = held[held.model_id.eq(model)]
        for cell, low, high in BINS:
            cell_rows = model_rows[(model_rows.gap >= low) & (model_rows.gap <= high)]
            a1.append(
                {
                    "model_id": model,
                    "layer": "exploratory",
                    "gap_bin": cell,
                    **calibration(cell_rows[cell_rows.gt_match]),
                }
            )
            for score in ("nll_motion", "q_motion"):
                a2.append(
                    {"model_id": model, "gap_bin": cell, **tail_cell(cell_rows, score)}
                )
        for cell, low, high in PRIMARY:
            cell_rows = model_rows[(model_rows.gap >= low) & (model_rows.gap <= high)]
            a1.append(
                {
                    "model_id": model,
                    "layer": "primary",
                    "gap_bin": cell,
                    **calibration(cell_rows[cell_rows.gt_match]),
                }
            )
            baseline_auc = auc_gt_low_score(cell_rows, "bridge_dist")
            energy_auc = auc_gt_low_score(cell_rows, "nll_motion")
            a3.append(
                {
                    "model_id": model,
                    "gap_bin": cell,
                    "n": len(cell_rows),
                    "gt": int(cell_rows.gt_match.sum()),
                    "bridge_dist_auc": baseline_auc,
                    "e_motion_auc": energy_auc,
                    "q_motion_auc": auc_gt_low_score(cell_rows, "q_motion"),
                    "retained": retention_pass(
                        e_motion_auc=energy_auc,
                        bridge_dist_auc=baseline_auc,
                        n_gt=int(cell_rows.gt_match.sum()),
                    ),
                    "low_support": bool(len(cell_rows[cell_rows.gt_match]) < 15),
                }
            )
        # A5 is deliberately not a tail/reversal table.  Each layer intersects
        # the frozen canonical cells and recomputes both motion scores and all
        # four M0 atoms on the identical held-out rows.
        for layer, layer_low, layer_high in SUPPORT_LAYERS:
            for bin_name, bin_low, bin_high in BINS:
                low, high = max(layer_low, bin_low), min(layer_high, bin_high)
                if low > high:
                    continue
                rows = model_rows[(model_rows.gap >= low) & (model_rows.gap <= high)]
                cell_name = (
                    bin_name
                    if (low, high) == (bin_low, bin_high)
                    else f"{bin_name}∩{low}-{high}"
                )
                a5.append(
                    separability_row(
                        rows,
                        model_id=model,
                        support_layer=layer,
                        gap_cell=cell_name,
                    )
                )
    # A4: exact four forensic keys, always held-out under fold 10.
    cards = json.loads((FORENSIC / "track_cards.json").read_text())
    cohort = []
    for track_key in json.loads((FORENSIC / "manifest.json").read_text())[
        "frozen_cohort"
    ]:
        card = cards[track_key]
        cohort.append(
            {
                "seq": card["sequence"],
                "lost_id": int(card["lost_id"]),
                "cand_id": int(card["min_d_h_row"]["cand_id"]),
                "gap": int(card["min_d_h_row"]["gap"]),
                "d_h": int(card["min_d_h_row"]["atom_analysis"]["d_h"]),
            }
        )
    cohort_frame = pd.DataFrame(cohort)
    for model in MODELS:
        rows = held[held.model_id.eq(model)]
        selected = cohort_frame.merge(
            rows,
            on=["seq", "lost_id", "cand_id", "gap"],
            how="left",
            validate="one_to_one",
        )
        if (
            len(selected) != 4
            or selected.nll_motion.isna().any()
            or not selected.held_out_sequence.eq("MOT17-10-SDP").all()
        ):
            raise ValueError(
                "A4 cohort did not resolve to four held-out fold-10 score rows"
            )
        flags = []
        for _, row in selected.iterrows():
            native_bin = gap_name(pd.Series([row.gap])).iat[0]
            native = rows[rows.gap_bin.eq(native_bin)]
            q90 = float(native.nll_motion.quantile(0.90))
            native_m0 = m0_signal_columns(native)
            bridge_q90 = float(native_m0["bridge_dist"].quantile(0.90))
            resid_q90 = float(native_m0["resid_mean"].quantile(0.90))
            resid_mean = float(0.5 * (row.fwd_resid + row.bwd_resid))
            flags.append(
                {
                    "seq": row.seq,
                    "lost_id": int(row.lost_id),
                    "cand_id": int(row.cand_id),
                    "gap": int(row.gap),
                    "d_h": int(row.d_h),
                    "nll_motion": float(row.nll_motion),
                    "q90": q90,
                    "gap_bin": native_bin,
                    "not_high_energy": bool(row.nll_motion < q90),
                    "bridge_dist": float(row.bridge_dist),
                    "bridge_dist_q90": bridge_q90,
                    "bridge_dist_in_pooled_q90": bool(row.bridge_dist >= bridge_q90),
                    "resid_mean": resid_mean,
                    "resid_mean_q90": resid_q90,
                    "resid_mean_in_pooled_q90": bool(resid_mean >= resid_q90),
                }
            )
        cohort_bins = {flag["gap_bin"] for flag in flags}
        classes = {
            row["gap_bin"]: row["calibration_class"]
            for row in a1
            if row["model_id"] == model
            and row["layer"] == "exploratory"
            and row["gap_bin"] in cohort_bins
        }
        a4_rows.append(
            {
                "model_id": model,
                "forensic_manifest_sha256": sha256(FORENSIC / "manifest.json"),
                "pairs": flags,
                "not_high_energy_count": sum(x["not_high_energy"] for x in flags),
                "passes_escape_box": sum(x["not_high_energy"] for x in flags) >= 3,
                "any_native_cohort_over_dispersed": "over-dispersed"
                in classes.values(),
            }
        )
    # A6: select from train-only surface, evaluate held-out after freezing tau.
    for model in MODELS:
        for cell, low, high in PRIMARY:
            fold_ids = sorted(held.fold_id.unique())
            for fold_id in fold_ids:
                h = held[
                    (held.model_id.eq(model))
                    & held.fold_id.eq(fold_id)
                    & (held.gap >= low)
                    & (held.gap <= high)
                ]
                t = train[
                    (train.model_id.eq(model))
                    & train.fold_id.eq(fold_id)
                    & (train.gap >= low)
                    & (train.gap <= high)
                ]
                old, new = select_a6(t, "bridge_dist"), select_a6(t, "nll_motion")
                terminal = (
                    "BOTH_EMPTY"
                    if old["terminal"] != "SELECTED" and new["terminal"] != "SELECTED"
                    else "EVALUATED"
                )
                payload: dict[str, Any] = {
                    "model_id": model,
                    "fold_id": fold_id,
                    "gap_bin": cell,
                    "held_out_sequence": h.held_out_sequence.iloc[0],
                    "qualifying_fold": bool(h.gt_match.sum() >= 20),
                    "terminal": terminal,
                    "old": old,
                    "new": new,
                }
                if terminal == "EVALUATED":
                    for name, item, score in (
                        ("old_held_out", old, "bridge_dist"),
                        ("new_held_out", new, "nll_motion"),
                    ):
                        payload[name] = (
                            None
                            if item["terminal"] != "SELECTED"
                            else cluster_stats(h, score, item["selected"]["threshold"])
                        )
                a6.append(payload)
    # Verdict-facing A6 aggregation is over captured FP counts on S_A.  A
    # primary cell contributes its row count, not a unit-weighted cell rate;
    # the 0.8 guard is then evaluated after pooling both cells within each fold.
    a6_pooled = []
    for model in MODELS:
        qualifying = [
            row for row in a6 if row["model_id"] == model and row["qualifying_fold"]
        ]
        nonempty = [row for row in qualifying if row["terminal"] != "BOTH_EMPTY"]
        safety = bool(nonempty) and all(
            row.get("new_held_out") and row["new_held_out"]["gt_leakage"] <= EPSILON
            for row in nonempty
        )
        complete = [
            row
            for row in nonempty
            if row.get("old_held_out") is not None
            and row.get("new_held_out") is not None
        ]
        fold_pools = []
        for fold_id in sorted({row["fold_id"] for row in complete}):
            cells = [row for row in complete if row["fold_id"] == fold_id]
            old_pool = pooled_fp_removed(cells, "old_held_out")
            new_pool = pooled_fp_removed(cells, "new_held_out")
            if old_pool is not None and new_pool is not None:
                fold_pools.append(
                    {
                        "fold_id": fold_id,
                        "held_out_sequence": cells[0]["held_out_sequence"],
                        "cells": [cell["gap_bin"] for cell in cells],
                        "old": old_pool,
                        "new": new_pool,
                        "passes_fold_0_8_guard": bool(
                            new_pool["fp_removed"] >= 0.8 * old_pool["fp_removed"]
                        ),
                    }
                )
        global_old = pooled_fp_removed(complete, "old_held_out")
        global_new = pooled_fp_removed(complete, "new_held_out")
        a6_pooled.append(
            {
                "model_id": model,
                "n_qualifying_cells": len(qualifying),
                "n_non_both_empty_cells": len(nonempty),
                "n_complete_cells": len(complete),
                "held_out_safety": safety,
                "fold_pools": fold_pools,
                "pooled_s_a_old": global_old,
                "pooled_s_a_new": global_new,
                "no_thinner": a6_no_thinner(
                    safety=safety,
                    fold_pools=fold_pools,
                    global_old=global_old,
                    global_new=global_new,
                ),
            }
        )
    a1_frame = pd.DataFrame(a1)
    # A7 fold transfer keeps the A6 threshold/evaluation lineage visible.
    for model in MODELS:
        for fold_id, fold in held[held.model_id.eq(model)].groupby(
            "fold_id", sort=True
        ):
            primary = fold[fold.gap.le(26)]
            a7.append(
                {
                    "model_id": model,
                    "fold_id": fold_id,
                    "held_out_sequence": fold.held_out_sequence.iloc[0],
                    "n_gt_primary": int(primary.gt_match.sum()),
                    "qualifying_fold": bool(primary.gt_match.sum() >= 20),
                    "pooled_primary_auc": auc_gt_low_score(primary, "nll_motion"),
                    "a6_threshold_link": [
                        {
                            "gap_bin": row["gap_bin"],
                            "terminal": row["terminal"],
                            "old_threshold": row["old"]
                            .get("selected", {})
                            .get("threshold"),
                            "new_threshold": row["new"]
                            .get("selected", {})
                            .get("threshold"),
                            "held_out_new_leakage": (
                                row.get("new_held_out", {}) or {}
                            ).get("gt_leakage"),
                        }
                        for row in a6
                        if row["model_id"] == model and row["fold_id"] == fold_id
                    ],
                    **calibration(primary[primary.gt_match]),
                }
            )
        # A8 retains train-side selection NLL, held-out total NLL, log-det
        # growth by canonical gap, and the relevant per-cell calibration map.
        primary = held[(held.model_id.eq(model)) & held.gap.le(26) & held.gt_match]
        for fold_id, fold in primary.groupby("fold_id", sort=True):
            all_fold = held[(held.model_id.eq(model)) & held.fold_id.eq(fold_id)]
            selection = selections[fold_id]
            growth = []
            for cell, low, high in BINS:
                cell_rows = all_fold[(all_fold.gap >= low) & (all_fold.gap <= high)]
                growth.append(
                    {
                        "gap_bin": cell,
                        "n": len(cell_rows),
                        "mean_log_det_covariance": float(
                            cell_rows.log_det_covariance.mean()
                        ),
                        "median_log_det_covariance": float(
                            cell_rows.log_det_covariance.median()
                        ),
                    }
                )
            classes = {
                row["gap_bin"]: row["calibration_class"]
                for _, row in a1_frame[
                    (a1_frame.model_id.eq(model)) & a1_frame.layer.eq("primary")
                ].iterrows()
            }
            a8.append(
                {
                    "model_id": model,
                    "fold_id": fold_id,
                    "held_out_sequence": fold.held_out_sequence.iloc[0],
                    "held_out_gt_total_nll": float(fold.nll_motion.sum()),
                    "held_out_gt_n": len(fold),
                    "mean_log_det_covariance": float(fold.log_det_covariance.mean()),
                    "training_nll": selection["training_nll_by_model"][model],
                    "selected_model_id": selection["selected_model_id"],
                    "is_training_selection_winner": model
                    == selection["selected_model_id"],
                    "log_det_growth_by_gap": growth,
                    "primary_calibration_classes": classes,
                }
            )
    a1_frame, a2_frame, a3_frame, a7_frame, a8_frame = map(
        pd.DataFrame, (a1, a2, a3, a7, a8)
    )
    member = {}
    for model in MODELS:
        ret = a3_frame[a3_frame.model_id.eq(model)].retained.dropna()
        retention = bool(len(ret) == 2 and ret.all())
        reversal = not a2_frame[
            (a2_frame.model_id.eq(model)) & a2_frame.score.eq("nll_motion")
        ].role_reversal.any()
        transfer_rows = a7_frame[a7_frame.model_id.eq(model) & a7_frame.qualifying_fold]
        transfer = bool(
            len(transfer_rows) and (transfer_rows.pooled_primary_auc > 0.5).all()
        )
        a6_summary = next(row for row in a6_pooled if row["model_id"] == model)
        no_thinner = bool(a6_summary["no_thinner"])
        escape = next(x for x in a4_rows if x["model_id"] == model)
        primary_classes = a1_frame[
            (a1_frame.model_id.eq(model)) & a1_frame.layer.eq("primary")
        ].calibration_class.tolist()
        diffusion = (
            "over-dispersed" not in primary_classes
            and not escape["any_native_cohort_over_dispersed"]
        )
        member[model] = {
            "retention": retention,
            "no_new_reversal": bool(reversal),
            "transfer": transfer,
            "no_thinner": no_thinner,
            "escape": bool(escape["passes_escape_box"]),
            "not_over_diffused": diffusion,
            "boxes_pass": bool(
                retention
                and reversal
                and transfer
                and no_thinner
                and escape["passes_escape_box"]
                and diffusion
            ),
        }
    m1_nll = {
        x["fold_id"]: x["held_out_gt_total_nll"]
        for x in a8
        if x["model_id"] == "M1P-GLOBAL-CV"
    }
    for model in MODELS[1:]:
        rows = [x for x in a8 if x["model_id"] == model]
        better_nll = all(
            x["held_out_gt_total_nll"] < m1_nll[x["fold_id"]] for x in rows
        )
        # Match the frozen wording: no primary cell M2 over-dispersed where M1 is approximately calibrated.
        mismatch = False
        for cell in ("1-10", "11-26"):
            m2 = a1_frame[
                (a1_frame.model_id.eq(model))
                & a1_frame.layer.eq("primary")
                & a1_frame.gap_bin.eq(cell)
            ].calibration_class.iloc[0]
            m1 = a1_frame[
                (a1_frame.model_id.eq("M1P-GLOBAL-CV"))
                & a1_frame.layer.eq("primary")
                & a1_frame.gap_bin.eq(cell)
            ].calibration_class.iloc[0]
            mismatch |= m2 == "over-dispersed" and m1 == "approximately-calibrated"
        member[model]["dominates_m1"] = m2_dominates(
            retention=member[model]["retention"],
            held_out_nll_better=better_nll,
            calibration_mismatch=mismatch,
        )
    low_primary = a3_frame.low_support.sum() > len(a3_frame) / 2
    verdict, anomaly = verdict_partition(member, low_primary=bool(low_primary))
    return {
        "schema_version": 1,
        "status": "PHASE_B_EXECUTED",
        "phase_b_authorized": True,
        "a1_a8_computed": True,
        "inputs": {
            "e3_manifest_sha256": sha256(E3_MANIFEST),
            "e3_scores_sha256": sha256(E3_SCORES),
            "pairs_sha256": sha256(PAIRS),
            "design_sha256": sha256(DESIGN),
        },
        "a1_calibration": a1,
        "a2_role_reversal": a2,
        "a3_short_gap_retention": a3,
        "a4_escape_tail": a4_rows,
        "a5_separability": a5,
        "a6_conditional_safe_region": a6,
        "a6_pooled_summary": a6_pooled,
        "a7_loo_transfer": a7,
        "a8_attribution": a8,
        "member_boxes": member,
        "verdict": {
            "code": verdict,
            "anomaly_note": anomaly,
            "claim_ceiling": "representation-level only; D0 not_fidelity_aligned prevents bridge threshold transfer and E_motion has no consumer-A counterpart",
        },
    }


def write_packet(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    clean = stable(result)
    (output / "phase_b_result.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True) + "\n"
    )
    for key, filename in (
        ("a1_calibration", "a1_calibration.csv"),
        ("a2_role_reversal", "a2_role_reversal.csv"),
        ("a3_short_gap_retention", "a3_short_gap_retention.csv"),
        ("a4_escape_tail", "a4_escape_tail.csv"),
        ("a5_separability", "a5_separability.csv"),
        ("a6_pooled_summary", "a6_pooled_summary.csv"),
        ("a7_loo_transfer", "a7_loo_transfer.csv"),
        ("a8_attribution", "a8_attribution.csv"),
    ):
        pd.DataFrame(clean[key]).to_csv(output / filename, index=False)
    (output / "recorded_output.txt").write_text(
        "\n".join(
            (
                f"status={clean['status']}",
                f"verdict={clean['verdict']['code']}",
                f"models={','.join(MODELS)}",
                "claim_ceiling=representation-level only",
                "",
            )
        )
    )
    artifacts = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "status": clean["status"],
        "phase_b_authorized": True,
        "a1_a8_computed": True,
        "verdict": clean["verdict"]["code"],
        "artifacts": artifacts,
        "runner_sha256": sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PACKET)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rebuild to a temporary directory and compare deterministic files",
    )
    args = parser.parse_args()
    scores, e3_manifest = load_inputs()
    result = compute(scores, e3_manifest)
    if args.verify:
        # The full reproduction contract starts at the frozen pair table: E3
        # first reconstructs the fold artifacts and its sealed score cube in a
        # temporary directory, then this runner deterministically consumes the
        # checked-in sealed cube for A1--A8.
        subprocess.run(
            ["uv", "run", "python", str(E3 / "run_e3_signals.py"), "--verify"],
            cwd=REPO,
            check=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            # The runner itself is a sealed packet artifact, so place the
            # identical source in the temporary packet before hashing.
            shutil.copy2(Path(__file__), target / Path(__file__).name)
            write_packet(result, target)
            expected = {p.name: sha256(p) for p in PACKET.iterdir() if p.is_file()}
            actual = {p.name: sha256(p) for p in target.iterdir() if p.is_file()}
            if expected != actual:
                raise AssertionError("Phase-B packet is not byte-reproducible")
        print("Phase-B packet verification: PASS")
        return
    write_packet(result, args.output_dir)
    print(f"Phase-B packet: {result['verdict']['code']}")


if __name__ == "__main__":
    main()
