"""M-B1.5 Stage 2 Q4.5 — structured threshold-combination atlas.

Descriptive atlas of single-atom and pairwise AND/OR regions on the locked
selected∧resolved D_online decision cohort.

Does **not**:
  - pick a single best threshold as a rule
  - unrestricted Boolean mining / 3+ atom combos
  - learned classifiers
  - hook policy / e2e effect / production preset change
  - promote observed GT_hurt==0 points to safe rules
  - reopen Q4 thr-chase on inseparable absolute tails as policy

Q4 weak AUC only blocks promoting *singleton frozen tails* to production thr;
it does **not** forbid threshold/Boolean structure as a data-analysis method.
"""

from __future__ import annotations

import hashlib
import itertools
import json
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
from saccade.perception.eval.d_online_stage2_q4 import (
    _f,
    _i,
    load_d_online_events,
    lock_q4_cohort,
    reconcile_q4,
)
from saccade.perception.eval.portable_or_tail import ORDERED_SIGNALS

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

TAXONOMY_VERSION = "stage2_q45_atlas_v2"
Q1Q3_STUDY_ID = "m_b1_5_stage2_q1q3_20260710"
Q4_STUDY_ID = "m_b1_5_stage2_q4_20260710"
EXPECTED_PRIMARY_N = 87
EXPECTED_PRIMARY_NEG = 23
EXPECTED_PRIMARY_POS = 64

# Declared rank/quantile lattice for pairwise complete enumeration.
# Fractions of the empirical CDF on the primary pool (inclusive endpoints).
PAIRWISE_QUANTILE_LATTICE: tuple[float, ...] = tuple(
    i / 20.0 for i in range(0, 21)
)  # 0.00, 0.05, ..., 1.00

# Single-atom lattice: all observed unique values on primary (complete family).
SINGLE_LATTICE_KIND = "primary_unique_boundaries"
PAIRWISE_LATTICE_KIND = "primary_quantile_lattice_q05"

DIRECTIONS = ("high_tail", "low_tail")  # x >= t , x <= t
COMBINATORS = ("AND", "OR")

TERMINAL_A = "structured_region_supported"
TERMINAL_B = "isolated_safe_points_only"
TERMINAL_C = "conditional_enrichment_without_safe_region"
TERMINAL_D = "no_useful_conditional_structure"

# Enrichment: precision / base_rate; require min support for claim
MIN_SUPPORT_FOR_ENRICHMENT = 2
MIN_ENRICHMENT_USEFUL = 1.25  # 25% above base negative rate
MIN_SEQS_FOR_REGION = 2
MAX_SEQ_SHARE_NON_DOMINANT = 0.5

# True held-out LOO: freeze absolute thr on train, evaluate on holdout.
# Deletion-consistency (old LOO) is NOT portability and cannot promote.
HOLD_OUT_PORTABILITY_GATE = True

ASSIGNMENT_GROUP_KEY_STATUS = (
    "invalid_frame_provenance"  # (seq, frame, cand_slot) uses host counter frame==4
)


class Stage2Q45Error(ValueError):
    """Fail-closed Q4.5 atlas error."""


# ---------------------------------------------------------------------------
# Frame provenance check
# ---------------------------------------------------------------------------


def check_frame_column_provenance(
    events: Sequence[Mapping[str, Any]],
    *,
    stage1_study_dir: Path | None = None,
) -> dict[str, Any]:
    """Document exact semantics of the ``frame`` column on D_online events.

    Does **not** authorize temporal-family conclusions.
    """
    frames = [int(r.get("frame", -999)) for r in events]
    uniq = sorted(set(frames))
    # Consistency with event_id / join_key
    event_id_frames = []
    join_key_frames = []
    mismatches = []
    for r in events:
        f = int(r.get("frame", -999))
        eid = str(r.get("event_id", ""))
        jk = str(r.get("join_key", ""))
        # event_id pattern: SEQ:f{frame}:c...
        ef = None
        if ":f" in eid:
            try:
                ef = int(eid.split(":f", 1)[1].split(":", 1)[0])
            except ValueError:
                ef = None
        event_id_frames.append(ef)
        jf = None
        parts = jk.split("|")
        if len(parts) >= 2:
            try:
                jf = int(parts[1])
            except ValueError:
                jf = None
        join_key_frames.append(jf)
        if ef is not None and ef != f:
            mismatches.append(
                {"kind": "event_id", "event_id": eid, "frame": f, "parsed": ef}
            )
        if jf is not None and jf != f:
            mismatches.append(
                {"kind": "join_key", "join_key": jk, "frame": f, "parsed": jf}
            )

    # Cross-check: MOT birth frames for sample track ids (if A1 MOT available)
    mot_cross: list[dict[str, Any]] = []
    if stage1_study_dir is not None:
        a1 = Path(stage1_study_dir) / "e2e_A1_hook_off"
        if a1.is_dir():
            by_seq: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for r in events[:40]:
                by_seq[str(r["sequence"])].append(r)
            for seq, rows in list(by_seq.items())[:3]:
                mot_path = a1 / f"{seq}.txt"
                if not mot_path.is_file():
                    continue
                # load MOT: frame,id,...
                tid_frames: dict[int, list[int]] = defaultdict(list)
                with mot_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split(",")
                        if len(parts) < 2:
                            continue
                        try:
                            fr, tid = int(float(parts[0])), int(float(parts[1]))
                        except ValueError:
                            continue
                        tid_frames[tid].append(fr)
                # Prefer global_id_map namespace (same as Q1 join); fall back to local
                gmap_path = a1 / "_global_id_map.txt"
                local_to_global: dict[int, int] = {}
                if gmap_path.is_file():
                    with gmap_path.open(encoding="utf-8") as gf:
                        for gl in gf:
                            gp = gl.strip().split()
                            if len(gp) >= 2:
                                try:
                                    # formats vary: local global OR seq local global
                                    if len(gp) == 2:
                                        local_to_global[int(gp[0])] = int(gp[1])
                                    else:
                                        local_to_global[int(gp[-2])] = int(gp[-1])
                                except ValueError:
                                    continue
                for r in rows[:5]:
                    for role, key, gkey in (
                        ("cand", "cand_track_id", "cand_global_id"),
                        ("lost", "lost_track_id", "lost_global_id"),
                    ):
                        local_tid = _i(r.get(key, -1))
                        # prefer explicit global columns from Q1q3 join
                        g_tid = (
                            _i(r.get(gkey, -1)) if r.get(gkey) not in (None, "") else -1
                        )
                        if g_tid < 0 and local_tid in local_to_global:
                            g_tid = local_to_global[local_tid]
                        # MOT file uses global trajectory ids when map present
                        query_tid = g_tid if g_tid >= 0 else local_tid
                        frs = tid_frames.get(query_tid, [])
                        mot_cross.append(
                            {
                                "sequence": seq,
                                "event_id": r.get("event_id"),
                                "audit_frame": int(r.get("frame", -1)),
                                "role": role,
                                "local_track_id": local_tid,
                                "global_track_id": g_tid if g_tid >= 0 else None,
                                "mot_query_id": query_tid,
                                "id_namespace": (
                                    "global" if g_tid >= 0 else "local_fallback"
                                ),
                                "mot_frame_min": min(frs) if frs else None,
                                "mot_frame_max": max(frs) if frs else None,
                                "mot_n_boxes": len(frs),
                                "audit_frame_equals_mot_min": (
                                    bool(frs) and int(r.get("frame", -1)) == min(frs)
                                ),
                            }
                        )

    n_mot_disagree = sum(
        1
        for m in mot_cross
        if m.get("mot_frame_min") is not None and m["audit_frame"] != m["mot_frame_min"]
    )

    # Source-code provenance (static)
    code_provenance = {
        "writer": (
            "src/tracking/tracker_gpu.cu relink_bidir_propose_kernel: "
            "ev.frame = frame_idx"
        ),
        "frame_idx_source": (
            "Host counter portable_audit_frame_++ passed into propose kernel "
            "(comment: 'Host frame counter for B-audit')"
        ),
        "reset_sites": [
            "ensure_portable_audit_ring_",
            "free_portable_audit_ring_",
            "clear_portable_or_tail_audit",
        ],
        "python_enrichment": (
            "portable_or_tail.enrich_online_audit_events copies e['frame'] into "
            "event_id (f{frame}) and join_key second field"
        ),
        "not_absolute_mot_frame": True,
        "not_gap_length": True,
        "not_fixed_probe_index_by_design": True,
        "observed_limitation": (
            "All Stage1 B-audit rows currently carry frame==4; counter is "
            "0-based host propose-invocation index, but observed constancy "
            "indicates the field is not a reliable multi-frame MOT timeline "
            "in this export (possible capture/staleness or single-write pattern). "
            "Cross-check vs A1 MOT track spans disagrees with absolute MOT frame."
        ),
    }

    is_absolute_mot = False  # established by code + MOT cross-check
    return {
        "check_id": "frame_column_provenance_q45",
        "column": "frame",
        "n_events": len(events),
        "unique_values": uniq,
        "n_unique": len(uniq),
        "all_equal_to_4": uniq == [4],
        "event_id_consistent": all(
            ef is None or ef == f for ef, f in zip(event_id_frames, frames)
        ),
        "join_key_consistent": all(
            jf is None or jf == f for jf, f in zip(join_key_frames, frames)
        ),
        "n_parse_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:10],
        "semantic": {
            "kind": "host_audit_propose_invocation_counter",
            "is_absolute_mot_frame": is_absolute_mot,
            "is_relative_event_frame": False,
            "is_gap_length": False,
            "is_fixed_probe_index": False,
            "is_export_limitation": True,
            "intended_meaning": (
                "0-based counter of bidirectional propose-kernel invocations "
                "since audit ring clear (portable_audit_frame_++)"
            ),
            "observed_meaning_limitation": (
                "Constant value 4 across all 244 D_online rows; "
                "must not be treated as absolute MOT frame index"
            ),
        },
        "code_provenance": code_provenance,
        "mot_crosscheck_sample": mot_cross[:15],
        "mot_crosscheck_n_disagree_with_mot_min": n_mot_disagree,
        "mot_crosscheck_n_compared": len(mot_cross),
        "conclusions": {
            "may_claim_absolute_mot_frame": False,
            "may_claim_temporal_information_unavailable_from_frame_alone": False,
            "observation_limitation_only": True,
            "affects_q45_threshold_atlas_mainline": False,
            "affects_competition_grouping": True,
            "assignment_group_key_status": ASSIGNMENT_GROUP_KEY_STATUS,
            "note": (
                "Do not write 'temporal information unavailable' solely because "
                "frame==4. The field is not absolute MOT frame. This is an "
                "instrumentation/export limitation for temporal studies; "
                "Q4.5 threshold-combination atlas does not depend on it."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Cohort lock
# ---------------------------------------------------------------------------


def lock_q45_cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Same primary lock as Q4."""
    locked = lock_q4_cohort(rows)
    s = locked["summary"]
    if int(s["n_primary"]) != EXPECTED_PRIMARY_N:
        raise Stage2Q45Error(
            f"primary n={s['n_primary']} != locked {EXPECTED_PRIMARY_N}"
        )
    if int(s["n_primary_negative"]) != EXPECTED_PRIMARY_NEG:
        raise Stage2Q45Error(f"neg={s['n_primary_negative']} != {EXPECTED_PRIMARY_NEG}")
    if int(s["n_primary_positive_protect"]) != EXPECTED_PRIMARY_POS:
        raise Stage2Q45Error(
            f"pos={s['n_primary_positive_protect']} != {EXPECTED_PRIMARY_POS}"
        )
    return locked


# ---------------------------------------------------------------------------
# Threshold registry
# ---------------------------------------------------------------------------


def _sorted_unique(x: np.ndarray) -> np.ndarray:
    v = x[np.isfinite(x)]
    if len(v) == 0:
        return np.asarray([], dtype=float)
    return np.unique(v)


def _quantile_thresholds(x: np.ndarray, qs: Sequence[float]) -> list[float]:
    v = x[np.isfinite(x)]
    if len(v) == 0:
        return []
    out = []
    for q in qs:
        t = float(np.quantile(v, q, method="linear"))
        out.append(t)
    # dedupe preserving order
    seen: set[float] = set()
    deduped = []
    for t in out:
        # use round-trip key for float stability
        key = round(t, 12)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def build_threshold_registry(
    primary: Sequence[Mapping[str, Any]],
    *,
    signals: Sequence[str] = ORDERED_SIGNALS,
    secondary_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Register complete lattices for single-atom and pairwise atlases."""
    y = np.asarray([int(r["q4_y"]) for r in primary], dtype=int)
    seqs = np.asarray([str(r["sequence"]) for r in primary], dtype=object)
    n = len(primary)
    base_neg_rate = float(np.mean(y == 1)) if n else float("nan")

    all_feats = list(signals) + list(secondary_features or [])
    matrices: dict[str, np.ndarray] = {}
    for name in all_feats:
        matrices[name] = np.asarray([_f(r.get(name)) for r in primary], dtype=float)

    single_entries: list[dict[str, Any]] = []
    pairwise_entries: list[dict[str, Any]] = []

    for name in all_feats:
        x = matrices[name]
        uniq = _sorted_unique(x)
        q_thrs = _quantile_thresholds(x, PAIRWISE_QUANTILE_LATTICE)
        is_secondary = name not in signals
        for thr_idx, t in enumerate(uniq.tolist()):
            for direction in DIRECTIONS:
                single_entries.append(
                    {
                        "atom_id": f"S::{name}::{direction}::u{thr_idx}",
                        "scope": "single_atom",
                        "lattice_kind": SINGLE_LATTICE_KIND,
                        "feature": name,
                        "is_secondary_feature": int(is_secondary),
                        "direction": direction,
                        "thr_index": thr_idx,
                        "thr_value": float(t),
                        "n_unique_on_primary": int(len(uniq)),
                    }
                )
        for thr_idx, t in enumerate(q_thrs):
            for direction in DIRECTIONS:
                pairwise_entries.append(
                    {
                        "atom_id": f"P::{name}::{direction}::q{thr_idx}",
                        "scope": "pairwise_atom",
                        "lattice_kind": PAIRWISE_LATTICE_KIND,
                        "feature": name,
                        "is_secondary_feature": int(is_secondary),
                        "direction": direction,
                        "thr_index": thr_idx,
                        "thr_value": float(t),
                        "quantile_lattice_point": PAIRWISE_QUANTILE_LATTICE[
                            min(thr_idx, len(PAIRWISE_QUANTILE_LATTICE) - 1)
                        ]
                        if thr_idx < len(PAIRWISE_QUANTILE_LATTICE)
                        else None,
                        "n_quantile_levels_registered": len(q_thrs),
                    }
                )

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "cohort_n": n,
        "n_negative": int(np.sum(y == 1)),
        "n_positive_protect": int(np.sum(y == 0)),
        "base_negative_rate": base_neg_rate,
        "signals_primary": list(signals),
        "secondary_features": list(secondary_features or []),
        "single_lattice_kind": SINGLE_LATTICE_KIND,
        "pairwise_lattice_kind": PAIRWISE_LATTICE_KIND,
        "pairwise_quantile_lattice": list(PAIRWISE_QUANTILE_LATTICE),
        "directions": list(DIRECTIONS),
        "combinators": list(COMBINATORS),
        "n_single_atoms": len(single_entries),
        "n_pairwise_atoms": len(pairwise_entries),
        "single_atoms": single_entries,
        "pairwise_atoms": pairwise_entries,
        "feature_matrices_meta": {
            name: {
                "n_finite": int(np.sum(np.isfinite(matrices[name]))),
                "n_unique": int(len(_sorted_unique(matrices[name]))),
                "min": float(np.nanmin(matrices[name]))
                if np.isfinite(matrices[name]).any()
                else float("nan"),
                "max": float(np.nanmax(matrices[name]))
                if np.isfinite(matrices[name]).any()
                else float("nan"),
            }
            for name in all_feats
        },
        # keep arrays out of JSON registry; returned separately for eval
        "_matrices": matrices,
        "_y": y,
        "_sequences": seqs,
    }


def atom_mask(x: np.ndarray, direction: str, thr: float) -> np.ndarray:
    if direction == "high_tail":
        return np.isfinite(x) & (x >= thr)
    if direction == "low_tail":
        return np.isfinite(x) & (x <= thr)
    raise Stage2Q45Error(f"unknown direction {direction}")


# ---------------------------------------------------------------------------
# Region metrics
# ---------------------------------------------------------------------------


def region_metrics(
    mask: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
    *,
    base_neg_rate: float,
    unknown_mask: np.ndarray | None = None,
    ambiguous_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Descriptive region metrics on resolved primary labels.

    LOO-by-deletion fields are renamed to leave_one_sequence_deleted_*;
    they are **not** portability evidence (deletion cannot increase GT hurt
    if full-cohort GT_hurt==0).
    """
    mask = np.asarray(mask, dtype=bool)
    n = len(y)
    support = int(mask.sum())
    n_neg = int(np.sum(mask & (y == 1)))
    n_gt = int(np.sum(mask & (y == 0)))
    precision = float(n_neg / support) if support > 0 else float("nan")
    enrichment = (
        float(precision / base_neg_rate)
        if support > 0 and base_neg_rate > 0
        else float("nan")
    )
    # per-sequence
    seq_neg: dict[str, int] = defaultdict(int)
    seq_gt: dict[str, int] = defaultdict(int)
    seq_sup: dict[str, int] = defaultdict(int)
    for i in np.where(mask)[0]:
        s = str(sequences[i])
        seq_sup[s] += 1
        if y[i] == 1:
            seq_neg[s] += 1
        else:
            seq_gt[s] += 1
    n_seq_with_support = len(seq_sup)
    n_seq_with_neg = sum(1 for v in seq_neg.values() if v > 0)
    max_seq_share = max(seq_sup.values()) / support if support > 0 else float("nan")
    max_neg_seq_share = max(seq_neg.values()) / n_neg if n_neg > 0 else float("nan")

    # --- deletion consistency (NOT portability) ---
    del_rows = []
    all_seqs = sorted(set(str(s) for s in sequences))
    for hold in all_seqs:
        m = sequences != hold
        mm = mask[m]
        yy = y[m]
        del_rows.append(
            {
                "hold_out_sequence": hold,
                "support": int(mm.sum()),
                "n_neg": int(np.sum(mm & (yy == 1))),
                "n_gt": int(np.sum(mm & (yy == 0))),
                "gt_hurt": int(np.sum(mm & (yy == 0))),
                "productive_safe": int(
                    int(np.sum(mm & (yy == 0))) == 0 and int(np.sum(mm & (yy == 1))) > 0
                ),
            }
        )
    del_max_gt = max((r["gt_hurt"] for r in del_rows), default=0)
    del_all_safe = all(r["gt_hurt"] == 0 for r in del_rows) if del_rows else False

    # --- unknown / unresolved contamination (selected only) ---
    n_unresolved = int(unknown_mask.sum()) if unknown_mask is not None else 0
    n_ambiguous = int(ambiguous_mask.sum()) if ambiguous_mask is not None else 0
    optimistic_gt_hurt = n_gt  # unresolved treated as non-GT
    pessimistic_gt_hurt = n_gt + n_unresolved + n_ambiguous  # all unknown as GT
    unknown_capture = n_unresolved + n_ambiguous
    unknown_capture_rate = (
        float(unknown_capture / (support + unknown_capture))
        if (support + unknown_capture) > 0
        else float("nan")
    )
    if unknown_capture > 0:
        safety_status = "unresolved_contaminated"
    elif n_gt == 0 and n_neg > 0:
        safety_status = "resolved_sample_zero_gt"
    elif n_gt == 0 and support > 0:
        safety_status = "resolved_empty_or_no_neg"
    else:
        safety_status = "resolved_gt_contaminated"

    observed_safe_point = support > 0 and n_gt == 0 and unknown_capture == 0
    # productive_safe on resolved only; unknown blocks promotion via safety_status
    productive_safe_resolved = n_gt == 0 and n_neg > 0
    productive_safe_point = productive_safe_resolved and unknown_capture == 0

    mask_sig = hashlib.sha256(np.packbits(mask.astype(np.uint8)).tobytes()).hexdigest()

    return {
        "support": support,
        "coverage": float(support / n) if n else float("nan"),
        "n_neg_captured": n_neg,
        "n_gt_captured": n_gt,
        "n_resolved_negative": n_neg,
        "n_resolved_gt": n_gt,
        "n_unresolved_selected": n_unresolved,
        "n_ambiguous_selected": n_ambiguous,
        "optimistic_gt_hurt": int(optimistic_gt_hurt),
        "pessimistic_gt_hurt": int(pessimistic_gt_hurt),
        "unknown_capture_rate": unknown_capture_rate,
        "safety_status": safety_status,
        "gt_hurt": n_gt,
        "gt_hurt_rate": float(n_gt / int(np.sum(y == 0)))
        if np.sum(y == 0)
        else float("nan"),
        "neg_capture_rate": float(n_neg / int(np.sum(y == 1)))
        if np.sum(y == 1)
        else float("nan"),
        "precision": precision,
        "enrichment": enrichment,
        "n_sequences_with_support": n_seq_with_support,
        "n_sequences_with_neg": n_seq_with_neg,
        "max_sequence_share": max_seq_share,
        "max_neg_sequence_share": max_neg_seq_share,
        "single_seq_neg_dominance": bool(
            n_neg > 0
            and math.isfinite(max_neg_seq_share)
            and max_neg_seq_share > MAX_SEQ_SHARE_NON_DOMINANT
        ),
        "observed_safe_point": bool(observed_safe_point),
        "productive_safe_resolved_only": bool(productive_safe_resolved),
        "productive_safe_point": bool(productive_safe_point),
        "not_a_safe_rule": True,
        # renamed: deletion consistency ≠ portability
        "leave_one_sequence_deleted_max_gt_hurt": int(del_max_gt),
        "leave_one_sequence_deleted_all_gt_hurt_zero": bool(del_all_safe),
        "leave_one_sequence_deleted_rows": del_rows,
        # legacy aliases kept for readers; explicit non-portability
        "loo_max_gt_hurt": int(del_max_gt),
        "loo_all_gt_hurt_zero": bool(del_all_safe),
        "loo_is_deletion_consistency_only": True,
        "loo_not_portability_evidence": True,
        "per_sequence_support": dict(seq_sup),
        "per_sequence_neg": dict(seq_neg),
        "per_sequence_gt": dict(seq_gt),
        "mask_sha256": mask_sig,
        "loo": del_rows,  # backward compat export
    }


def true_holdout_sequence_validation(
    *,
    x_parts: Mapping[str, np.ndarray],
    y: np.ndarray,
    sequences: np.ndarray,
    feature: str,
    direction: str,
    thr_value: float,
    feature_b: str | None = None,
    direction_b: str | None = None,
    thr_value_b: float | None = None,
    combinator: str | None = None,
) -> dict[str, Any]:
    """Freeze absolute thr on train folds; apply to held-out sequence.

    For each holdout H:
      1. build mask with frozen thr on train (all ≠ H)
      2. require train has both classes available (else flag)
      3. apply same thr to H only; report holdout GT hurt / neg capture
    """
    seqs = sorted(set(str(s) for s in sequences))
    rows = []
    for hold in seqs:
        train = sequences != hold
        hold_m = sequences == hold
        xa = x_parts[feature]
        m_train = atom_mask(xa[train], direction, thr_value)
        m_hold = atom_mask(xa[hold_m], direction, thr_value)
        if (
            feature_b is not None
            and thr_value_b is not None
            and direction_b is not None
        ):
            xb = x_parts[feature_b]
            mb_tr = atom_mask(xb[train], direction_b, thr_value_b)
            mb_ho = atom_mask(xb[hold_m], direction_b, thr_value_b)
            if combinator == "OR":
                m_train = m_train | mb_tr
                m_hold = m_hold | mb_ho
            else:
                m_train = m_train & mb_tr
                m_hold = m_hold & mb_ho
        y_tr, y_ho = y[train], y[hold_m]
        n_tr_neg = int(np.sum(m_train & (y_tr == 1)))
        n_tr_gt = int(np.sum(m_train & (y_tr == 0)))
        n_ho_neg = int(np.sum(m_hold & (y_ho == 1)))
        n_ho_gt = int(np.sum(m_hold & (y_ho == 0)))
        rows.append(
            {
                "hold_out_sequence": hold,
                "train_support": int(m_train.sum()),
                "train_n_neg": n_tr_neg,
                "train_gt_hurt": n_tr_gt,
                "holdout_support": int(m_hold.sum()),
                "holdout_n_neg": n_ho_neg,
                "holdout_gt_hurt": n_ho_gt,
                "holdout_productive_safe": int(n_ho_gt == 0 and n_ho_neg > 0),
                "train_has_both_classes_in_region": int(n_tr_neg > 0 and n_tr_gt >= 0),
            }
        )
    worst = max((r["holdout_gt_hurt"] for r in rows), default=0)
    all_zero = all(r["holdout_gt_hurt"] == 0 for r in rows) if rows else False
    any_prod = any(r["holdout_productive_safe"] == 1 for r in rows)
    return {
        "true_holdout_rows": rows,
        "true_holdout_worst_gt_hurt": int(worst),
        "true_holdout_all_gt_hurt_zero": bool(all_zero),
        "true_holdout_any_productive_safe": bool(any_prod),
        "true_holdout_portability_ok": bool(all_zero and any_prod),
    }


# ---------------------------------------------------------------------------
# Single-atom atlas
# ---------------------------------------------------------------------------


def build_single_atom_atlas(
    registry: Mapping[str, Any],
    *,
    primary_signals_only: bool = False,
    unknown_matrices: Mapping[str, np.ndarray] | None = None,
    ambiguous_matrices: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    matrices: dict[str, np.ndarray] = registry["_matrices"]  # type: ignore
    y: np.ndarray = registry["_y"]  # type: ignore
    sequences: np.ndarray = registry["_sequences"]  # type: ignore
    base = float(registry["base_negative_rate"])
    rows: list[dict[str, Any]] = []
    unknown_matrices = unknown_matrices or {}
    ambiguous_matrices = ambiguous_matrices or {}

    # Group atoms by (feature, direction) for neighbor continuity
    by_fd: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for atom in registry["single_atoms"]:
        if primary_signals_only and int(atom.get("is_secondary_feature", 0)):
            continue
        by_fd[(atom["feature"], atom["direction"])].append(atom)

    for (feature, direction), atoms in by_fd.items():
        atoms_sorted = sorted(atoms, key=lambda a: int(a["thr_index"]))
        x = matrices[feature]
        masks = []
        metrics_list = []
        for atom in atoms_sorted:
            thr = float(atom["thr_value"])
            m = atom_mask(x, direction, thr)
            um = None
            am = None
            if feature in unknown_matrices:
                um = atom_mask(unknown_matrices[feature], direction, thr)
            if feature in ambiguous_matrices:
                am = atom_mask(ambiguous_matrices[feature], direction, thr)
            met = region_metrics(
                m, y, sequences, base_neg_rate=base, unknown_mask=um, ambiguous_mask=am
            )
            masks.append(m)
            metrics_list.append(met)

        for i, atom in enumerate(atoms_sorted):
            met = metrics_list[i]
            # neighbor continuity: adjacent thr_index
            neighbor_gt = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(atoms_sorted):
                    neighbor_gt.append(metrics_list[j]["gt_hurt"])
                    # subset relation
            # subset vs adjacent: for high_tail, higher thr ⇒ subset of lower thr
            subset_of_prev = None
            prev_subset_of_this = None
            if i > 0:
                subset_of_prev = bool(np.all(~masks[i] | masks[i - 1]))
                prev_subset_of_this = bool(np.all(~masks[i - 1] | masks[i]))
            subset_of_next = None
            next_subset_of_this = None
            if i + 1 < len(masks):
                subset_of_next = bool(np.all(~masks[i] | masks[i + 1]))
                next_subset_of_this = bool(np.all(~masks[i + 1] | masks[i]))

            # neighborhood GT continuity: neighbors also productive-safe?
            adj_productive_safe = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(metrics_list):
                    adj_productive_safe.append(
                        bool(metrics_list[j]["productive_safe_point"])
                    )

            rows.append(
                {
                    "atom_id": atom["atom_id"],
                    "feature": feature,
                    "is_secondary_feature": int(atom.get("is_secondary_feature", 0)),
                    "direction": direction,
                    "thr_index": int(atom["thr_index"]),
                    "thr_value": float(atom["thr_value"]),
                    "lattice_kind": atom["lattice_kind"],
                    "support": met["support"],
                    "coverage": met["coverage"],
                    "n_neg_captured": met["n_neg_captured"],
                    "n_gt_captured": met["n_gt_captured"],
                    "gt_hurt": met["gt_hurt"],
                    "gt_hurt_rate": met["gt_hurt_rate"],
                    "neg_capture_rate": met["neg_capture_rate"],
                    "precision": met["precision"],
                    "enrichment": met["enrichment"],
                    "n_sequences_with_support": met["n_sequences_with_support"],
                    "n_sequences_with_neg": met["n_sequences_with_neg"],
                    "max_sequence_share": met["max_sequence_share"],
                    "max_neg_sequence_share": met["max_neg_sequence_share"],
                    "single_seq_neg_dominance": int(met["single_seq_neg_dominance"]),
                    "observed_safe_point": int(met["observed_safe_point"]),
                    "productive_safe_point": int(met["productive_safe_point"]),
                    "productive_safe_resolved_only": int(
                        met.get("productive_safe_resolved_only", False)
                    ),
                    "n_resolved_negative": met["n_resolved_negative"],
                    "n_resolved_gt": met["n_resolved_gt"],
                    "n_unresolved_selected": met["n_unresolved_selected"],
                    "n_ambiguous_selected": met["n_ambiguous_selected"],
                    "optimistic_gt_hurt": met["optimistic_gt_hurt"],
                    "pessimistic_gt_hurt": met["pessimistic_gt_hurt"],
                    "unknown_capture_rate": met["unknown_capture_rate"],
                    "safety_status": met["safety_status"],
                    "mask_sha256": met["mask_sha256"],
                    "not_a_safe_rule": 1,
                    "loo_max_gt_hurt": met["loo_max_gt_hurt"],
                    "loo_all_gt_hurt_zero": int(met["loo_all_gt_hurt_zero"]),
                    "loo_is_deletion_consistency_only": 1,
                    "loo_not_portability_evidence": 1,
                    "leave_one_sequence_deleted_all_gt_hurt_zero": int(
                        met["leave_one_sequence_deleted_all_gt_hurt_zero"]
                    ),
                    "n_adjacent_neighbors": len(adj_productive_safe),
                    "n_adjacent_also_productive_safe": sum(adj_productive_safe),
                    "subset_of_prev_thr": (
                        int(subset_of_prev) if subset_of_prev is not None else None
                    ),
                    "prev_subset_of_this": (
                        int(prev_subset_of_this)
                        if prev_subset_of_this is not None
                        else None
                    ),
                    "subset_of_next_thr": (
                        int(subset_of_next) if subset_of_next is not None else None
                    ),
                    "next_subset_of_this": (
                        int(next_subset_of_this)
                        if next_subset_of_this is not None
                        else None
                    ),
                    "neighbor_gt_hurts": "|".join(str(g) for g in neighbor_gt),
                    "necessity_neg_coverage": met["neg_capture_rate"],
                    "observed_sufficiency_gt_contamination": met["gt_hurt"],
                    "per_sequence_neg_json": json.dumps(
                        met["per_sequence_neg"], sort_keys=True
                    ),
                    "per_sequence_gt_json": json.dumps(
                        met["per_sequence_gt"], sort_keys=True
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Pairwise atlas
# ---------------------------------------------------------------------------


def build_pairwise_atlas(
    registry: Mapping[str, Any],
    *,
    combinator: str,
    primary_signals_only: bool = True,
    unknown_matrices: Mapping[str, np.ndarray] | None = None,
    ambiguous_matrices: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Complete enumeration of registered pairwise lattice (AND or OR)."""
    if combinator not in COMBINATORS:
        raise Stage2Q45Error(f"bad combinator {combinator}")
    matrices: dict[str, np.ndarray] = registry["_matrices"]  # type: ignore
    y: np.ndarray = registry["_y"]  # type: ignore
    sequences: np.ndarray = registry["_sequences"]  # type: ignore
    base = float(registry["base_negative_rate"])
    unknown_matrices = unknown_matrices or {}
    ambiguous_matrices = ambiguous_matrices or {}

    atoms = [
        a
        for a in registry["pairwise_atoms"]
        if (not primary_signals_only) or not int(a.get("is_secondary_feature", 0))
    ]
    # group by feature for pairing across different features only
    by_feat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        by_feat[a["feature"]].append(a)

    features = sorted(by_feat.keys())
    # Precompute masks for each pairwise atom
    atom_masks: dict[str, np.ndarray] = {}
    for a in atoms:
        atom_masks[a["atom_id"]] = atom_mask(
            matrices[a["feature"]], a["direction"], float(a["thr_value"])
        )

    rows: list[dict[str, Any]] = []
    seen_sig: set[tuple[Any, ...]] = set()

    for f1, f2 in itertools.combinations(features, 2):
        for a in by_feat[f1]:
            for b in by_feat[f2]:
                m1 = atom_masks[a["atom_id"]]
                m2 = atom_masks[b["atom_id"]]
                if combinator == "AND":
                    m = m1 & m2
                else:
                    m = m1 | m2
                # unknown masks on selected unresolved/ambiguous
                um = am = None
                if (
                    a["feature"] in unknown_matrices
                    and b["feature"] in unknown_matrices
                ):
                    u1 = atom_mask(
                        unknown_matrices[a["feature"]],
                        a["direction"],
                        float(a["thr_value"]),
                    )
                    u2 = atom_mask(
                        unknown_matrices[b["feature"]],
                        b["direction"],
                        float(b["thr_value"]),
                    )
                    um = (u1 & u2) if combinator == "AND" else (u1 | u2)
                if (
                    a["feature"] in ambiguous_matrices
                    and b["feature"] in ambiguous_matrices
                ):
                    a1m = atom_mask(
                        ambiguous_matrices[a["feature"]],
                        a["direction"],
                        float(a["thr_value"]),
                    )
                    a2m = atom_mask(
                        ambiguous_matrices[b["feature"]],
                        b["direction"],
                        float(b["thr_value"]),
                    )
                    am = (a1m & a2m) if combinator == "AND" else (a1m | a2m)
                sig = (combinator, m.tobytes())
                is_dup = sig in seen_sig
                seen_sig.add(sig)

                met = region_metrics(
                    m,
                    y,
                    sequences,
                    base_neg_rate=base,
                    unknown_mask=um,
                    ambiguous_mask=am,
                )
                rows.append(
                    {
                        "combo_id": (f"{combinator}::{a['atom_id']}::{b['atom_id']}"),
                        "combinator": combinator,
                        "atom_a_id": a["atom_id"],
                        "atom_b_id": b["atom_id"],
                        "feature_a": a["feature"],
                        "feature_b": b["feature"],
                        "direction_a": a["direction"],
                        "direction_b": b["direction"],
                        "thr_index_a": int(a["thr_index"]),
                        "thr_index_b": int(b["thr_index"]),
                        "thr_value_a": float(a["thr_value"]),
                        "thr_value_b": float(b["thr_value"]),
                        "lattice_kind": PAIRWISE_LATTICE_KIND,
                        "semantic_duplicate_mask": int(is_dup),
                        "empty_region": int(met["support"] == 0),
                        "support": met["support"],
                        "coverage": met["coverage"],
                        "n_neg_captured": met["n_neg_captured"],
                        "n_gt_captured": met["n_gt_captured"],
                        "gt_hurt": met["gt_hurt"],
                        "gt_hurt_rate": met["gt_hurt_rate"],
                        "neg_capture_rate": met["neg_capture_rate"],
                        "precision": met["precision"],
                        "enrichment": met["enrichment"],
                        "n_sequences_with_support": met["n_sequences_with_support"],
                        "n_sequences_with_neg": met["n_sequences_with_neg"],
                        "max_sequence_share": met["max_sequence_share"],
                        "max_neg_sequence_share": met["max_neg_sequence_share"],
                        "single_seq_neg_dominance": int(
                            met["single_seq_neg_dominance"]
                        ),
                        "observed_safe_point": int(met["observed_safe_point"]),
                        "productive_safe_point": int(met["productive_safe_point"]),
                        "productive_safe_resolved_only": int(
                            met.get("productive_safe_resolved_only", False)
                        ),
                        "n_resolved_negative": met["n_resolved_negative"],
                        "n_resolved_gt": met["n_resolved_gt"],
                        "n_unresolved_selected": met["n_unresolved_selected"],
                        "n_ambiguous_selected": met["n_ambiguous_selected"],
                        "optimistic_gt_hurt": met["optimistic_gt_hurt"],
                        "pessimistic_gt_hurt": met["pessimistic_gt_hurt"],
                        "unknown_capture_rate": met["unknown_capture_rate"],
                        "safety_status": met["safety_status"],
                        "mask_sha256": met["mask_sha256"],
                        "not_a_safe_rule": 1,
                        "loo_max_gt_hurt": met["loo_max_gt_hurt"],
                        "loo_all_gt_hurt_zero": int(met["loo_all_gt_hurt_zero"]),
                        "loo_is_deletion_consistency_only": 1,
                        "loo_not_portability_evidence": 1,
                        "leave_one_sequence_deleted_all_gt_hurt_zero": int(
                            met["leave_one_sequence_deleted_all_gt_hurt_zero"]
                        ),
                        "necessity_neg_coverage": met["neg_capture_rate"],
                        "observed_sufficiency_gt_contamination": met["gt_hurt"],
                        "per_sequence_neg_json": json.dumps(
                            met["per_sequence_neg"], sort_keys=True
                        ),
                        "per_sequence_gt_json": json.dumps(
                            met["per_sequence_gt"], sort_keys=True
                        ),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Stability classification for productive-safe points
# ---------------------------------------------------------------------------


def classify_region_stability(
    atom_rows: Sequence[Mapping[str, Any]],
    pairwise_and: Sequence[Mapping[str, Any]],
    pairwise_or: Sequence[Mapping[str, Any]],
    *,
    holdout_by_region: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Classify productive-safe points with duplicate-free topology.

    Rules (review #89):
      - semantic duplicate masks do not form neighbors (one node per mask_sha256)
      - interior requires BOTH sides (1D) or full 4-neighborhood (pairwise)
      - boundary points are edge_candidate only — never thick region
      - unresolved_contaminated cannot be region_candidate
      - true_holdout portability required for loo_stable_region; deletion LOO banned
    """
    holdout_by_region = holdout_by_region or {}
    out: list[dict[str, Any]] = []

    def _eligible(r: Mapping[str, Any]) -> bool:
        if not int(r.get("productive_safe_point", 0)):
            return False
        if int(r.get("is_secondary_feature", 0)):
            return False
        if int(r.get("semantic_duplicate_mask", 0)):
            return False
        if str(r.get("safety_status", "")) == "unresolved_contaminated":
            return False
        return True

    # --- single atoms: unique by mask, topology on thr_index ---
    by_fd: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in atom_rows:
        if not _eligible(r):
            continue
        by_fd[(str(r["feature"]), str(r["direction"]))].append(r)

    for (feat, direction), cells in by_fd.items():
        # unique mask representatives ordered by thr_index
        seen_mask: set[str] = set()
        uniq = []
        for r in sorted(cells, key=lambda x: int(x["thr_index"])):
            sig = str(r.get("mask_sha256", ""))
            if sig and sig in seen_mask:
                continue
            if sig:
                seen_mask.add(sig)
            uniq.append(r)
        idx_map = {int(r["thr_index"]): r for r in uniq}
        for r in uniq:
            ti = int(r["thr_index"])
            left = idx_map.get(ti - 1)
            right = idx_map.get(ti + 1)
            has_left = left is not None
            has_right = right is not None
            left_ok = bool(left and int(left.get("productive_safe_point", 0)))
            right_ok = bool(right and int(right.get("productive_safe_point", 0)))
            is_boundary = not (has_left and has_right)
            n_adj = int(has_left) + int(has_right)
            n_adj_also = int(left_ok) + int(right_ok)
            ho = holdout_by_region.get(str(r["atom_id"]), {})
            true_port = bool(ho.get("true_holdout_portability_ok", False))
            true_worst = int(ho.get("true_holdout_worst_gt_hurt", 99))

            if is_boundary:
                label = "edge_candidate" if n_adj_also > 0 else "isolated_safe_point"
            elif n_adj_also < 2:
                label = "thin_safe_edge" if n_adj_also == 1 else "isolated_safe_point"
            elif int(r.get("n_sequences_with_neg", 0)) < MIN_SEQS_FOR_REGION or int(
                r.get("single_seq_neg_dominance", 0)
            ):
                label = "locally_stable_region_but_seq_thin"
            elif true_port and true_worst == 0:
                label = "loo_stable_region"
            else:
                label = "locally_stable_region"

            is_cand = int(
                label in ("loo_stable_region", "locally_stable_region")
                and not is_boundary
                and str(r.get("safety_status")) != "unresolved_contaminated"
            )
            out.append(
                {
                    "region_id": r["atom_id"],
                    "kind": "single_atom",
                    "feature_a": feat,
                    "feature_b": "",
                    "direction_a": direction,
                    "direction_b": "",
                    "thr_index_a": ti,
                    "thr_index_b": "",
                    "thr_value_a": r["thr_value"],
                    "thr_value_b": "",
                    "support": r["support"],
                    "n_neg_captured": r["n_neg_captured"],
                    "gt_hurt": r["gt_hurt"],
                    "n_unresolved_selected": r.get("n_unresolved_selected", 0),
                    "safety_status": r.get("safety_status"),
                    "mask_sha256": r.get("mask_sha256"),
                    "n_sequences_with_neg": r["n_sequences_with_neg"],
                    "max_neg_sequence_share": r["max_neg_sequence_share"],
                    "is_boundary": int(is_boundary),
                    "n_adjacent_neighbors": n_adj,
                    "n_adjacent_also_productive_safe": n_adj_also,
                    "requires_bilateral_neighbors": 1,
                    "true_holdout_worst_gt_hurt": true_worst,
                    "true_holdout_portability_ok": int(true_port),
                    "loo_deletion_consistency_not_portability": 1,
                    "stability_class": label,
                    "is_region_candidate": is_cand,
                    "not_a_safe_rule": 1,
                }
            )

    # --- pairwise: unique mask only; full 4-neighborhood for interior ---
    for comb_name, rows in (("AND", pairwise_and), ("OR", pairwise_or)):
        # first representative per mask
        by_mask: dict[str, Mapping[str, Any]] = {}
        for r in rows:
            if not _eligible(r):
                continue
            sig = str(r.get("mask_sha256", ""))
            if not sig or sig in by_mask:
                continue
            by_mask[sig] = r
        # index by thr coords among unique
        idx = {}
        for r in by_mask.values():
            key = (
                r["feature_a"],
                r["direction_a"],
                int(r["thr_index_a"]),
                r["feature_b"],
                r["direction_b"],
                int(r["thr_index_b"]),
            )
            idx[key] = r
        for r in by_mask.values():
            key = (
                r["feature_a"],
                r["direction_a"],
                int(r["thr_index_a"]),
                r["feature_b"],
                r["direction_b"],
                int(r["thr_index_b"]),
            )
            fa, da, ia, fb, db, ib = key
            neighbors = []
            for dia, dib in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = idx.get((fa, da, ia + dia, fb, db, ib + dib))
                if nb is not None:
                    neighbors.append(nb)
            n_adj = len(neighbors)
            n_adj_also = sum(
                1 for a in neighbors if int(a.get("productive_safe_point", 0))
            )
            is_boundary = n_adj < 4
            ho = holdout_by_region.get(str(r["combo_id"]), {})
            true_port = bool(ho.get("true_holdout_portability_ok", False))
            true_worst = int(ho.get("true_holdout_worst_gt_hurt", 99))
            if is_boundary:
                label = "edge_candidate" if n_adj_also > 0 else "isolated_safe_point"
            elif n_adj_also < 4:
                label = "thin_safe_edge"
            elif int(r.get("n_sequences_with_neg", 0)) < MIN_SEQS_FOR_REGION or int(
                r.get("single_seq_neg_dominance", 0)
            ):
                label = "locally_stable_region_but_seq_thin"
            elif true_port and true_worst == 0:
                label = "loo_stable_region"
            else:
                label = "locally_stable_region"
            is_cand = int(
                label in ("loo_stable_region", "locally_stable_region")
                and not is_boundary
            )
            out.append(
                {
                    "region_id": r["combo_id"],
                    "kind": f"pairwise_{comb_name}",
                    "feature_a": r["feature_a"],
                    "feature_b": r["feature_b"],
                    "direction_a": r["direction_a"],
                    "direction_b": r["direction_b"],
                    "thr_index_a": r["thr_index_a"],
                    "thr_index_b": r["thr_index_b"],
                    "thr_value_a": r["thr_value_a"],
                    "thr_value_b": r["thr_value_b"],
                    "support": r["support"],
                    "n_neg_captured": r["n_neg_captured"],
                    "gt_hurt": r["gt_hurt"],
                    "n_unresolved_selected": r.get("n_unresolved_selected", 0),
                    "safety_status": r.get("safety_status"),
                    "mask_sha256": r.get("mask_sha256"),
                    "n_sequences_with_neg": r["n_sequences_with_neg"],
                    "max_neg_sequence_share": r["max_neg_sequence_share"],
                    "is_boundary": int(is_boundary),
                    "n_adjacent_neighbors": n_adj,
                    "n_adjacent_also_productive_safe": n_adj_also,
                    "requires_full_4_neighborhood": 1,
                    "true_holdout_worst_gt_hurt": true_worst,
                    "true_holdout_portability_ok": int(true_port),
                    "loo_deletion_consistency_not_portability": 1,
                    "stability_class": label,
                    "is_region_candidate": is_cand,
                    "not_a_safe_rule": 1,
                }
            )
    return out


def build_pareto_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    """Pareto over multi-objective region quality (full atlas, not top-k only).

    Objectives (direction):
      gt_hurt ↓, n_neg_captured ↑, n_sequences_with_neg ↑,
      max_sequence_share ↓, loo_all_gt_hurt_zero ↑, loo_max_gt_hurt ↓

    Implementation: O(n log n) sort + linear scan after collapsing exact
    objective-ties, with a final O(f^2) pass on the candidate front (f ≪ n).
    """
    candidates: list[Mapping[str, Any]] = []
    for r in rows:
        if int(r.get("support", 0) or 0) <= 0:
            continue
        if int(r.get("semantic_duplicate_mask", 0)):
            continue
        candidates.append(r)

    if not candidates:
        return []

    def obj_key(r: Mapping[str, Any]) -> tuple:
        return (
            int(r["gt_hurt"]),
            -int(r["n_neg_captured"]),
            -int(r["n_sequences_with_neg"]),
            float(r.get("max_sequence_share") or 1.0),
            -int(r.get("loo_all_gt_hurt_zero", 0)),
            int(r.get("loo_max_gt_hurt", 99)),
        )

    def dominates_tuple(a: tuple, b: tuple) -> bool:
        """a dominates b on the obj_key encoding (lower is better for all)."""
        better_or_eq = all(x <= y for x, y in zip(a, b))
        strictly = any(x < y for x, y in zip(a, b))
        return better_or_eq and strictly

    # Keep one representative per identical objective vector
    best_by_obj: dict[tuple, Mapping[str, Any]] = {}
    for r in candidates:
        k = obj_key(r)
        if k not in best_by_obj:
            best_by_obj[k] = r
    uniq = list(best_by_obj.items())  # (obj_tuple, row)

    # Sort by first objective then others; greedy 1D front candidates
    uniq.sort(key=lambda t: t[0])
    prelim: list[tuple[tuple, Mapping[str, Any]]] = []
    for k, r in uniq:
        if any(dominates_tuple(pk, k) for pk, _ in prelim):
            continue
        # remove previously kept that this dominates
        prelim = [(pk, pr) for pk, pr in prelim if not dominates_tuple(k, pk)]
        prelim.append((k, r))

    # Exact front among prelim (small)
    frontier_rows: list[Mapping[str, Any]] = []
    for i, (ki, ri) in enumerate(prelim):
        if any(dominates_tuple(kj, ki) for j, (kj, _) in enumerate(prelim) if i != j):
            continue
        frontier_rows.append(ri)

    frontier_sorted = sorted(frontier_rows, key=obj_key)
    out = []
    for r in frontier_sorted:
        rid = r.get("atom_id") or r.get("combo_id") or ""
        out.append(
            {
                "kind": kind,
                "region_id": rid,
                "gt_hurt": r["gt_hurt"],
                "n_neg_captured": r["n_neg_captured"],
                "coverage": r["coverage"],
                "support": r["support"],
                "n_sequences_with_neg": r["n_sequences_with_neg"],
                "max_sequence_share": r["max_sequence_share"],
                "loo_all_gt_hurt_zero": r.get("loo_all_gt_hurt_zero"),
                "loo_max_gt_hurt": r.get("loo_max_gt_hurt"),
                "enrichment": r.get("enrichment"),
                "precision": r.get("precision"),
                "productive_safe_point": r.get("productive_safe_point"),
                "feature_a": r.get("feature") or r.get("feature_a"),
                "feature_b": r.get("feature_b", ""),
                "direction_a": r.get("direction") or r.get("direction_a"),
                "direction_b": r.get("direction_b", ""),
                "combinator": r.get("combinator", "single"),
                "not_a_safe_rule": 1,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Terminal classification
# ---------------------------------------------------------------------------


def classify_q45_terminal(
    *,
    stability_rows: Sequence[Mapping[str, Any]],
    atom_rows: Sequence[Mapping[str, Any]],
    pairwise_and: Sequence[Mapping[str, Any]],
    pairwise_or: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    blocked = [
        "threshold_as_production_rule_not_authorized",
        "boolean_as_hook_policy_not_authorized",
        "e2e_effect_claim_not_authorized",
        "production_promotion_blocked",
        "observed_safe_point_is_not_safe_rule",
        "unrestricted_boolean_mining_forbidden",
        "three_plus_atom_combos_forbidden",
    ]

    primary_atoms = [r for r in atom_rows if not int(r.get("is_secondary_feature", 0))]
    all_primary_regions = primary_atoms + list(pairwise_and) + list(pairwise_or)

    region_candidates = [
        r
        for r in stability_rows
        if int(r.get("is_region_candidate", 0))
        and r.get("stability_class")
        in (
            "loo_stable_region",
            "locally_stable_region",
        )
    ]
    loo_stable = [
        r
        for r in region_candidates
        if r.get("stability_class") == "loo_stable_region"
        and int(r.get("true_holdout_portability_ok", 0)) == 1
    ]

    # A: multi-seq, interior thickness, *true* holdout portability, zero unknown
    if loo_stable:
        return {
            "stage2_q45_terminal": TERMINAL_A,
            "terminal_letter": "A",
            "reason": (
                f"{len(loo_stable)} true-holdout-portable interior region(s) with "
                "local thickness; authorize formal restricted safe-region "
                "candidate validation (still not production rule)"
            ),
            "claims_blocked": blocked,
            "claims_allowed": [
                "formal_restricted_safe_region_candidate_validation_authorized",
            ],
            "next_authorized_step": (
                "formal restricted safe-region candidate validation on "
                "declared LOO-stable regions only"
            ),
            "n_region_candidates": len(region_candidates),
            "n_loo_stable_regions": len(loo_stable),
            "supporting_region_ids": [r["region_id"] for r in loo_stable[:20]],
            "production_preset": "unchanged",
        }

    isolated = [
        r
        for r in stability_rows
        if r.get("stability_class") in ("isolated_safe_point", "thin_safe_edge")
    ]
    any_productive_safe = any(
        int(r.get("productive_safe_point", 0)) for r in all_primary_regions
    )

    if isolated or (any_productive_safe and not region_candidates):
        n_prod = sum(
            1 for r in all_primary_regions if int(r.get("productive_safe_point", 0))
        )
        return {
            "stage2_q45_terminal": TERMINAL_B,
            "terminal_letter": "B",
            "reason": (
                f"On resolved∧selected cohort: sample-zero-GT cells exist "
                f"(n_atlas_cells={n_prod}) but no unique interior multi-seq thick "
                "region under duplicate-free topology + true holdout; "
                "unknown/unresolved selected coverage further limits claims → "
                "isolated_safe_points_only"
            ),
            "claims_blocked": blocked
            + [
                "formal_safe_region_not_authorized",
                "threshold_formulation_global_closure_not_authorized",
                "portable_safe_region_claim_inadmissible",
                "deletion_loo_as_portability_forbidden",
            ],
            "claims_allowed": [
                "retain_atlas_for_followup",
                "report_isolated_observed_safe_points_descriptively",
                "ranking_research_reasonable_next_line_not_threshold_closure",
            ],
            "next_authorized_step": (
                "threshold path: no promotable region yet; ranking/assignment "
                "is a reasonable next research line AFTER valid assignment-group "
                "key + unknown coverage + true holdout LOO are closed"
            ),
            "bounded_finding": (
                "resolved∧selected restricted atlas reports sample-zero-GT cells "
                "but no unique interior multi-sequence thick region"
            ),
            "n_productive_safe_cells": n_prod,
            "n_isolated_or_thin": len(isolated),
            "production_preset": "unchanged",
        }

    # C: enrichment without safe region
    enriching = [
        r
        for r in all_primary_regions
        if int(r.get("support", 0)) >= MIN_SUPPORT_FOR_ENRICHMENT
        and math.isfinite(float(r.get("enrichment") or 0))
        and float(r.get("enrichment") or 0) >= MIN_ENRICHMENT_USEFUL
        and int(r.get("n_neg_captured", 0)) > 0
        and int(r.get("gt_hurt", 0)) > 0
    ]
    if enriching:
        best = max(enriching, key=lambda r: float(r.get("enrichment") or 0))
        return {
            "stage2_q45_terminal": TERMINAL_C,
            "terminal_letter": "C",
            "reason": (
                f"conditional enrichment without zero-GT region "
                f"(n_enriching_cells={len(enriching)}; "
                f"best_enrichment={float(best.get('enrichment') or 0):.3f}); "
                "may study ranking/calibration, not reject rule"
            ),
            "claims_blocked": blocked
            + [
                "reject_rule_not_authorized",
                "formal_safe_region_not_authorized",
            ],
            "claims_allowed": [
                "report_conditional_enrichment_structure",
                "ranking_or_calibration_research_authorized",
            ],
            "next_authorized_step": (
                "ranking / calibration research on enriched cells; "
                "not reject-threshold promotion"
            ),
            "n_enriching_cells": len(enriching),
            "best_enrichment": float(best.get("enrichment") or 0),
            "production_preset": "unchanged",
        }

    # D: no useful structure
    return {
        "stage2_q45_terminal": TERMINAL_D,
        "terminal_letter": "D",
        "reason": (
            "complete restricted combination atlas shows no useful enrichment "
            "or productive-safe structure on locked cohort"
        ),
        "claims_blocked": blocked
        + [
            "reject_rule_not_authorized",
            "formal_safe_region_not_authorized",
        ],
        "claims_allowed": [
            "report_null_atlas_result",
            "new_signal_family_or_earlier_placement_authorized",
        ],
        "next_authorized_step": (
            "turn to new signal family or earlier placement "
            "(after full atlas null result)"
        ),
        "production_preset": "unchanged",
    }


# ---------------------------------------------------------------------------
# Secondary competition features (optional columns; not mainline)
# ---------------------------------------------------------------------------


def attach_secondary_competition_features(
    events: list[dict[str, Any]],
) -> list[str]:
    """DEPRECATED reconstruction: assignment key uses invalid frame provenance.

    Host audit `frame` is constant 4 — (seq, frame, cand_slot) is **not** a
    valid multi-moment assignment group. Values may still be written for
    audit/debug but are flagged untrusted and excluded from mainline conclusions.
    """
    from collections import defaultdict as dd

    groups: dict[tuple[Any, ...], list[int]] = dd(list)
    for i, r in enumerate(events):
        key = (str(r["sequence"]), int(r["frame"]), int(r.get("cand_slot", -1)))
        groups[key].append(i)

    secondary = [
        "sec_winner_runnerup_score_margin",
        "sec_delta_vs_ru_abs_log_h",
        "sec_competitor_count",
    ]
    for i, r in enumerate(events):
        for name in secondary:
            r[name] = float("nan")
        r["sec_assignment_group_key_status"] = ASSIGNMENT_GROUP_KEY_STATUS
        r["sec_competition_features_trusted"] = 0
        r["sec_competitor_count"] = float("nan")  # do not trust Q1 competitor_count
        # Intentionally leave margins NaN: do not emit ranking-path evidence
        # from invalid grouping. Existing competitor_count column remains
        # descriptive with provenance warning in summary.
    return secondary


# ---------------------------------------------------------------------------
# Full runner
# ---------------------------------------------------------------------------


def run_stage2_q45_atlas(
    *,
    q1q3_study_dir: Path,
    out_dir: Path,
    stage1_study_dir: Path | None = None,
    git_commit: str = "",
    study_id: str | None = None,
    include_secondary_competition: bool = True,
) -> dict[str, Any]:
    q1q3_study_dir = Path(q1q3_study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_raw = load_d_online_events(q1q3_study_dir)
    source = q1q3_study_dir / "d_online_events.parquet"
    if not source.is_file():
        source = q1q3_study_dir / "d_online_events.csv"
    source_hash = _sha256_file(source) if source.is_file() else ""

    # Resolve stage1 dir for frame provenance MOT cross-check
    if stage1_study_dir is None:
        # try from events or default
        stage1_study_dir = Path(
            "out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close"
        )
    stage1_study_dir = Path(stage1_study_dir)

    frame_prov = check_frame_column_provenance(
        events_raw, stage1_study_dir=stage1_study_dir
    )

    events = [dict(r) for r in events_raw]
    secondary_names: list[str] = []
    if include_secondary_competition:
        secondary_names = attach_secondary_competition_features(events)

    locked = lock_q45_cohort(events)
    primary = locked["primary"]
    cohort_sum = locked["summary"]
    # selected unresolved / ambiguous for unknown contamination
    selected_unresolved = [
        r
        for r in events
        if _i(r.get("baseline_selected", r.get("baseline_accepted_candidate", 0))) == 1
        and str(r.get("label_status", "")) == "unresolved"
    ]
    selected_ambiguous = [
        r
        for r in events
        if _i(r.get("baseline_selected", r.get("baseline_accepted_candidate", 0))) == 1
        and str(r.get("label_status", "")) == "ambiguous"
    ]
    cohort_sum["n_selected_total"] = sum(
        1
        for r in events
        if _i(r.get("baseline_selected", r.get("baseline_accepted_candidate", 0))) == 1
    )
    cohort_sum["n_selected_unresolved"] = len(selected_unresolved)
    cohort_sum["n_selected_ambiguous"] = len(selected_ambiguous)
    cohort_sum["unknown_coverage_note"] = (
        "Atlas primary remains resolved∧selected; cells report "
        "n_unresolved_selected / pessimistic_gt_hurt; unresolved_contaminated "
        "cannot enter region_candidate"
    )
    recon = reconcile_q4(n_d_online=len(events_raw), cohort=cohort_sum, primary=primary)
    if not recon["ok"]:
        raise Stage2Q45Error(f"recon FAIL: {recon.get('errors')}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = study_id or f"m_b1_5_stage2_q45_{stamp}"

    # secondary features untrusted — still not used for mainline pairwise
    registry = build_threshold_registry(
        primary,
        signals=ORDERED_SIGNALS,
        secondary_features=None,  # demote competition columns from atlas lattice
    )
    registry_json = {k: v for k, v in registry.items() if not k.startswith("_")}
    registry_json["assignment_group_key_status"] = ASSIGNMENT_GROUP_KEY_STATUS
    registry_json["secondary_competition_trusted"] = False

    unknown_matrices = {
        s: np.asarray([_f(r.get(s)) for r in selected_unresolved], dtype=float)
        for s in ORDERED_SIGNALS
    }
    ambiguous_matrices = {
        s: np.asarray([_f(r.get(s)) for r in selected_ambiguous], dtype=float)
        for s in ORDERED_SIGNALS
    }

    atom_rows = build_single_atom_atlas(
        registry,
        primary_signals_only=True,
        unknown_matrices=unknown_matrices,
        ambiguous_matrices=ambiguous_matrices,
    )
    pairwise_and = build_pairwise_atlas(
        registry,
        combinator="AND",
        primary_signals_only=True,
        unknown_matrices=unknown_matrices,
        ambiguous_matrices=ambiguous_matrices,
    )
    pairwise_or = build_pairwise_atlas(
        registry,
        combinator="OR",
        primary_signals_only=True,
        unknown_matrices=unknown_matrices,
        ambiguous_matrices=ambiguous_matrices,
    )

    # True holdout validation for productive-safe unique cells
    y = registry["_y"]
    sequences = registry["_sequences"]
    x_parts = {s: registry["_matrices"][s] for s in ORDERED_SIGNALS}
    holdout_by_region: dict[str, dict[str, Any]] = {}
    for r in atom_rows:
        if not int(r.get("productive_safe_point", 0)):
            continue
        if int(r.get("is_secondary_feature", 0)):
            continue
        if str(r.get("safety_status")) == "unresolved_contaminated":
            continue
        ho = true_holdout_sequence_validation(
            x_parts=x_parts,
            y=y,
            sequences=sequences,
            feature=str(r["feature"]),
            direction=str(r["direction"]),
            thr_value=float(r["thr_value"]),
        )
        holdout_by_region[str(r["atom_id"])] = ho
        r["true_holdout_worst_gt_hurt"] = ho["true_holdout_worst_gt_hurt"]
        r["true_holdout_portability_ok"] = int(ho["true_holdout_portability_ok"])
    for r in list(pairwise_and) + list(pairwise_or):
        if not int(r.get("productive_safe_point", 0)):
            continue
        if int(r.get("semantic_duplicate_mask", 0)):
            continue
        if str(r.get("safety_status")) == "unresolved_contaminated":
            continue
        ho = true_holdout_sequence_validation(
            x_parts=x_parts,
            y=y,
            sequences=sequences,
            feature=str(r["feature_a"]),
            direction=str(r["direction_a"]),
            thr_value=float(r["thr_value_a"]),
            feature_b=str(r["feature_b"]),
            direction_b=str(r["direction_b"]),
            thr_value_b=float(r["thr_value_b"]),
            combinator=str(r["combinator"]),
        )
        holdout_by_region[str(r["combo_id"])] = ho
        r["true_holdout_worst_gt_hurt"] = ho["true_holdout_worst_gt_hurt"]
        r["true_holdout_portability_ok"] = int(ho["true_holdout_portability_ok"])

    stability = classify_region_stability(
        atom_rows, pairwise_and, pairwise_or, holdout_by_region=holdout_by_region
    )

    pareto_single = build_pareto_frontier(
        [r for r in atom_rows if not int(r.get("is_secondary_feature", 0))],
        kind="single_atom",
    )
    pareto_and = build_pareto_frontier(pairwise_and, kind="pairwise_AND")
    pareto_or = build_pareto_frontier(pairwise_or, kind="pairwise_OR")
    pareto_all = pareto_single + pareto_and + pareto_or

    terminal = classify_q45_terminal(
        stability_rows=stability,
        atom_rows=atom_rows,
        pairwise_and=pairwise_and,
        pairwise_or=pairwise_or,
    )

    # Per-sequence / LOO export (flattened from productive-safe + frontier)
    loo_rows = []
    per_seq_rows = []
    for r in atom_rows + pairwise_and + pairwise_or:
        rid = r.get("atom_id") or r.get("combo_id")
        if not int(r.get("productive_safe_point", 0)) and int(r.get("gt_hurt", 1)) > 0:
            # still keep high-enrichment cells' per-seq for atlas completeness?
            # Export only productive-safe + enrichment>=threshold to limit size
            enr = float(r.get("enrichment") or 0)
            if not (
                math.isfinite(enr)
                and enr >= MIN_ENRICHMENT_USEFUL
                and int(r.get("n_neg_captured", 0)) > 0
            ):
                continue
        # parse per-seq json
        try:
            pn = json.loads(r.get("per_sequence_neg_json") or "{}")
            pg = json.loads(r.get("per_sequence_gt_json") or "{}")
        except json.JSONDecodeError:
            pn, pg = {}, {}
        for s in sorted(set(list(pn) + list(pg))):
            per_seq_rows.append(
                {
                    "region_id": rid,
                    "sequence": s,
                    "n_neg": int(pn.get(s, 0)),
                    "n_gt": int(pg.get(s, 0)),
                    "support": int(pn.get(s, 0)) + int(pg.get(s, 0)),
                }
            )

    # LOO table from recompute on productive-safe primary regions only
    matrices = registry["_matrices"]
    y = registry["_y"]
    sequences = registry["_sequences"]
    base = float(registry["base_negative_rate"])
    for r in atom_rows:
        if not int(r.get("productive_safe_point", 0)):
            continue
        if int(r.get("is_secondary_feature", 0)):
            continue
        m = atom_mask(matrices[r["feature"]], r["direction"], float(r["thr_value"]))
        met = region_metrics(m, y, sequences, base_neg_rate=base)
        for lo in met["loo"]:
            loo_rows.append(
                {
                    "region_id": r["atom_id"],
                    "kind": "single_atom",
                    **lo,
                }
            )
    for r in pairwise_and + pairwise_or:
        if not int(r.get("productive_safe_point", 0)):
            continue
        m1 = atom_mask(
            matrices[r["feature_a"]],
            r["direction_a"],
            float(r["thr_value_a"]),
        )
        m2 = atom_mask(
            matrices[r["feature_b"]],
            r["direction_b"],
            float(r["thr_value_b"]),
        )
        m = (m1 & m2) if r["combinator"] == "AND" else (m1 | m2)
        met = region_metrics(m, y, sequences, base_neg_rate=base)
        for lo in met["loo"]:
            loo_rows.append(
                {
                    "region_id": r["combo_id"],
                    "kind": f"pairwise_{r['combinator']}",
                    **lo,
                }
            )

    # Counts for summary
    n_prod_single = sum(
        1
        for r in atom_rows
        if int(r.get("productive_safe_point", 0))
        and not int(r.get("is_secondary_feature", 0))
    )
    n_prod_and = sum(1 for r in pairwise_and if int(r.get("productive_safe_point", 0)))
    n_prod_or = sum(1 for r in pairwise_or if int(r.get("productive_safe_point", 0)))

    hashes: dict[str, str] = {}
    hashes["threshold_registry"] = write_json(
        out_dir / "threshold_registry.json", registry_json
    )
    hashes["frame_column_provenance"] = write_json(
        out_dir / "frame_column_provenance.json", frame_prov
    )
    hashes["atom_atlas"] = write_csv(out_dir / "atom_atlas.csv", atom_rows)
    write_parquet(out_dir / "atom_atlas.parquet", atom_rows)
    if (out_dir / "atom_atlas.parquet").is_file():
        hashes["atom_atlas_parquet"] = _sha256_file(out_dir / "atom_atlas.parquet")

    hashes["pairwise_and_atlas"] = write_csv(
        out_dir / "pairwise_and_atlas.csv", pairwise_and
    )
    write_parquet(out_dir / "pairwise_and_atlas.parquet", pairwise_and)
    if (out_dir / "pairwise_and_atlas.parquet").is_file():
        hashes["pairwise_and_atlas_parquet"] = _sha256_file(
            out_dir / "pairwise_and_atlas.parquet"
        )

    hashes["pairwise_or_atlas"] = write_csv(
        out_dir / "pairwise_or_atlas.csv", pairwise_or
    )
    write_parquet(out_dir / "pairwise_or_atlas.parquet", pairwise_or)
    if (out_dir / "pairwise_or_atlas.parquet").is_file():
        hashes["pairwise_or_atlas_parquet"] = _sha256_file(
            out_dir / "pairwise_or_atlas.parquet"
        )

    hashes["pareto_frontier"] = write_csv(out_dir / "pareto_frontier.csv", pareto_all)
    hashes["region_stability"] = write_csv(out_dir / "region_stability.csv", stability)
    hashes["per_sequence"] = write_csv(out_dir / "per_sequence.csv", per_seq_rows)
    hashes["loo"] = write_csv(out_dir / "loo.csv", loo_rows)
    hashes["cohort_summary"] = write_json(out_dir / "cohort_summary.json", cohort_sum)
    hashes["reconciliation"] = write_json(out_dir / "reconciliation.json", recon)

    summary = {
        "study_id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_q1q3_study": str(q1q3_study_dir),
        "upstream_q4_study": Q4_STUDY_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        "D_online_total": len(events_raw),
        "n_primary": cohort_sum["n_primary"],
        "n_primary_negative": cohort_sum["n_primary_negative"],
        "n_primary_positive_protect": cohort_sum["n_primary_positive_protect"],
        "n_sequences_with_both_classes": cohort_sum["n_sequences_with_both_classes"],
        "cohort_definition": cohort_sum["cohort_definition"],
        "single_lattice_kind": SINGLE_LATTICE_KIND,
        "pairwise_lattice_kind": PAIRWISE_LATTICE_KIND,
        "n_single_atoms_registered": registry_json["n_single_atoms"],
        "n_pairwise_atoms_registered": registry_json["n_pairwise_atoms"],
        "n_atom_atlas_rows": len(atom_rows),
        "n_pairwise_and_rows": len(pairwise_and),
        "n_pairwise_or_rows": len(pairwise_or),
        "n_productive_safe_single": n_prod_single,
        "n_productive_safe_and": n_prod_and,
        "n_productive_safe_or": n_prod_or,
        "n_pareto_frontier": len(pareto_all),
        "n_stability_rows": len(stability),
        "secondary_competition_features": secondary_names,
        "secondary_competition_trusted": False,
        "assignment_group_key_status": ASSIGNMENT_GROUP_KEY_STATUS,
        "n_selected_unresolved": len(selected_unresolved),
        "n_selected_ambiguous": len(selected_ambiguous),
        "evaluator_review_gates": {
            "deletion_loo_is_portability": False,
            "true_holdout_required_for_region_A": True,
            "unresolved_contaminated_blocks_candidate": True,
            "duplicate_masks_not_neighbors": True,
            "interior_requires_full_neighborhood": True,
        },
        "frame_provenance_kind": frame_prov["semantic"]["kind"],
        "frame_is_absolute_mot": frame_prov["semantic"]["is_absolute_mot_frame"],
        "frame_observation_limitation_only": frame_prov["conclusions"][
            "observation_limitation_only"
        ],
        **terminal,
        "reconciliation_acceptance": recon["acceptance"],
        "forbidden_this_round": [
            "single_best_threshold_as_rule",
            "unrestricted_boolean_mining",
            "three_plus_atom_combos",
            "learned_classifier",
            "post_hoc_grammar_expansion",
            "hook_policy",
            "e2e_effect_claim",
            "production_preset_change",
            "sample_zero_gt_named_safe_rule",
            "q4_weak_auc_means_boolean_impossible",
            "new_signal_terminal_classification",
        ],
        "q4_interpretation": (
            "Q4 weak marginal AUC closes singleton frozen-tail thr promotion; "
            "does not forbid threshold/Boolean combination as analysis method."
        ),
    }
    hashes["summary"] = write_json(out_dir / "summary.json", summary)

    # Canonical evidence pack (small, PR-auditable)
    evidence_dir = out_dir / "evidence_pack"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pack_files = {
        "manifest.json": out_dir / "manifest.json",
        "summary.json": out_dir / "summary.json",
        "reconciliation.json": out_dir / "reconciliation.json",
        "cohort_summary.json": out_dir / "cohort_summary.json",
        "threshold_registry.json": out_dir / "threshold_registry.json",
        "region_stability.csv": out_dir / "region_stability.csv",
        "pareto_frontier.csv": out_dir / "pareto_frontier.csv",
        "frame_column_provenance.json": out_dir / "frame_column_provenance.json",
    }
    # true holdout table
    ho_flat = []
    for rid, ho in holdout_by_region.items():
        for row in ho.get("true_holdout_rows", []):
            ho_flat.append({"region_id": rid, **row})
    hashes["true_holdout_loo"] = write_csv(out_dir / "true_holdout_loo.csv", ho_flat)
    pack_files["true_holdout_loo.csv"] = out_dir / "true_holdout_loo.csv"
    sha_rows = []
    for name, src in pack_files.items():
        if src.is_file():
            (evidence_dir / name).write_bytes(src.read_bytes())
            sha_rows.append(
                {
                    "file": name,
                    "sha256": _sha256_file(src),
                    "bytes": src.stat().st_size,
                }
            )
    write_json(evidence_dir / "SHA256SUMS.json", {"files": sha_rows})
    write_json(
        evidence_dir / "README.json",
        {
            "study_id": sid,
            "taxonomy_version": TAXONOMY_VERSION,
            "note": (
                "Canonical evidence pack for PR audit. Large full atlases remain "
                "in parent study dir; rebuild via run_m_b1_5_stage2_q45_atlas.py"
            ),
            "reproduce": (
                "uv run python scripts/tools/run_m_b1_5_stage2_q45_atlas.py "
                f"--q1q3-study out/signal_study/{Q1Q3_STUDY_ID} "
                f"--out out/signal_study/{sid}"
            ),
            "terminal": terminal.get("stage2_q45_terminal"),
            "assignment_group_key_status": ASSIGNMENT_GROUP_KEY_STATUS,
        },
    )

    md = _render_md(summary, terminal, frame_prov, stability, pareto_all)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    hashes["summary_md"] = hashlib.sha256(md.encode("utf-8")).hexdigest()

    manifest = {
        "schema": "m_b1_5_stage2_q45_atlas_manifest_v1",
        "study_id": sid,
        "git_commit": git_commit,
        "source_q1q3_study": str(q1q3_study_dir),
        "source_event_table": str(source),
        "source_event_table_hash": source_hash,
        "upstream_q4_study": Q4_STUDY_ID,
        "candidate_universe_id": CANDIDATE_UNIVERSE_ID,
        "substrate_id": SUBSTRATE_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "cohort_definition": cohort_sum["cohort_definition"],
        "sequence_set": sorted({str(r["sequence"]) for r in primary}),
        "artifact_hashes": hashes,
        "created_utc": summary["created_utc"],
        "terminal_letter": terminal.get("terminal_letter"),
        "stage2_q45_terminal": terminal.get("stage2_q45_terminal"),
        "production_preset": "unchanged",
    }
    write_json(out_dir / "manifest.json", manifest)
    summary["manifest"] = manifest
    return summary


def _render_md(
    summary: Mapping[str, Any],
    terminal: Mapping[str, Any],
    frame_prov: Mapping[str, Any],
    stability: Sequence[Mapping[str, Any]],
    pareto: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# Stage 2 Q4.5 threshold-combination atlas — `{summary.get('study_id')}`",
        "",
        "<!-- doc-status: research-artifact -->",
        "",
        "## Terminal",
        "",
        "```text",
        f"stage2_q45_terminal: {terminal.get('stage2_q45_terminal')}",
        f"terminal_letter: {terminal.get('terminal_letter')}",
        f"n_primary: {summary.get('n_primary')} "
        f"(neg={summary.get('n_primary_negative')}, "
        f"pos={summary.get('n_primary_positive_protect')})",
        f"productive_safe cells: single={summary.get('n_productive_safe_single')} "
        f"AND={summary.get('n_productive_safe_and')} "
        f"OR={summary.get('n_productive_safe_or')}",
        f"next: {terminal.get('next_authorized_step')}",
        f"production_preset: {summary.get('production_preset')}",
        "```",
        "",
        f"**Reason:** {terminal.get('reason')}",
        "",
        "## Frame column provenance",
        "",
        f"- semantic: `{frame_prov['semantic']['kind']}`",
        f"- absolute MOT frame: **{frame_prov['semantic']['is_absolute_mot_frame']}**",
        f"- unique values on D_online: {frame_prov.get('unique_values')}",
        f"- observation limitation only: "
        f"{frame_prov['conclusions']['observation_limitation_only']}",
        f"- affects Q4.5 mainline: "
        f"{frame_prov['conclusions']['affects_q45_threshold_atlas_mainline']}",
        "",
        frame_prov["conclusions"]["note"],
        "",
        "## Atlas sizes",
        "",
        f"- single-atom rows: {summary.get('n_atom_atlas_rows')}",
        f"- pairwise AND rows: {summary.get('n_pairwise_and_rows')}",
        f"- pairwise OR rows: {summary.get('n_pairwise_or_rows')}",
        f"- pareto frontier rows: {summary.get('n_pareto_frontier')}",
        f"- stability rows (productive-safe): {summary.get('n_stability_rows')}",
        "",
        "## Stability class counts",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for r in stability:
        counts[str(r.get("stability_class"))] += 1
    for k in sorted(counts):
        lines.append(f"- `{k}`: {counts[k]}")
    lines += [
        "",
        "## Pareto head (first 12)",
        "",
        "| kind | gt_hurt | n_neg | cov | n_seq_neg | max_share | loo_safe |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for r in pareto[:12]:
        lines.append(
            f"| {r.get('kind')} | {r.get('gt_hurt')} | {r.get('n_neg_captured')} | "
            f"{float(r.get('coverage') or 0):.3f} | {r.get('n_sequences_with_neg')} | "
            f"{float(r.get('max_sequence_share') or 0):.2f} | "
            f"{r.get('loo_all_gt_hurt_zero')} |"
        )
    lines += [
        "",
        "## Claim firewall",
        "",
        "Blocked:",
        "",
    ]
    for b in terminal.get("claims_blocked", []):
        lines.append(f"- `{b}`")
    lines += ["", "Allowed:", ""]
    for a in terminal.get("claims_allowed", []) or ["(none)"]:
        lines.append(f"- `{a}`")
    lines += [
        "",
        "Observed productive-safe points are **not** safe rules.",
        "Q4 weak AUC ≠ Boolean combination impossible.",
        "",
    ]
    return "\n".join(lines) + "\n"
