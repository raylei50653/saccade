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
import shutil
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

TAXONOMY_VERSION = "stage2_q45_atlas_v4"
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

# Nested LOSO: rebuild lattice + select on train, freeze, evaluate holdout.
# Deletion-consistency and fixed-full-sample thr re-check are NOT portability.
NESTED_LOSO_PORTABILITY_GATE = True

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


def fixed_full_sample_region_partition_check(
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
    """Partition check only: same thr on train/holdout subsets of full sample.

    **Not portability.** If full-cohort GT_hurt==0, every holdout subset also has
    GT_hurt==0. Kept for diagnostics; never used for A promotion.
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
        rows.append(
            {
                "hold_out_sequence": hold,
                "train_gt_hurt": int(np.sum(m_train & (y_tr == 0))),
                "holdout_gt_hurt": int(np.sum(m_hold & (y_ho == 0))),
                "holdout_n_neg": int(np.sum(m_hold & (y_ho == 1))),
            }
        )
    return {
        "kind": "fixed_full_sample_region_partition_check",
        "not_portability_evidence": True,
        "rows": rows,
    }


def _clause_id_single(feature: str, direction: str, thr: float) -> str:
    return f"S::{feature}::{direction}::t{round(thr, 12)}"


def _clause_id_pair(
    comb: str,
    fa: str,
    da: str,
    ta: float,
    fb: str,
    db: str,
    tb: float,
) -> str:
    return f"{comb}::{fa}::{da}::t{round(ta, 12)}::{fb}::{db}::t{round(tb, 12)}"


def nested_loso_portability_audit(
    primary: Sequence[Mapping[str, Any]],
    *,
    selected_unresolved: Sequence[Mapping[str, Any]],
    selected_ambiguous: Sequence[Mapping[str, Any]],
    signals: Sequence[str] = ORDERED_SIGNALS,
) -> dict[str, Any]:
    """Nested leave-one-sequence-out: train lattice → select → freeze → holdout.

    For each held-out sequence H:
      1. Build single unique thr lattice + pairwise quantile lattice on train only
      2. Select train cells with train GT_hurt==0, train n_neg>0, train unknown==0
      3. Freeze absolute thr / Boolean structure
      4. Apply to H; record holdout GT hurt / neg capture
    Aggregates selection frequency and worst holdout GT across folds.
    """
    primary = list(primary)
    sequences = np.asarray([str(r["sequence"]) for r in primary], dtype=object)
    y = np.asarray([int(r["q4_y"]) for r in primary], dtype=int)
    mats = {
        s: np.asarray([_f(r.get(s)) for r in primary], dtype=float) for s in signals
    }
    unres = list(selected_unresolved)
    amb = list(selected_ambiguous)
    unres_seq = (
        np.asarray([str(r["sequence"]) for r in unres], dtype=object)
        if unres
        else np.array([], dtype=object)
    )
    amb_seq = (
        np.asarray([str(r["sequence"]) for r in amb], dtype=object)
        if amb
        else np.array([], dtype=object)
    )
    unres_mats = {
        s: np.asarray([_f(r.get(s)) for r in unres], dtype=float) for s in signals
    }
    amb_mats = {
        s: np.asarray([_f(r.get(s)) for r in amb], dtype=float) for s in signals
    }

    all_seqs = sorted(set(str(s) for s in sequences))
    # clause_id -> stats
    clause_stats: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []

    def _bump(cid: str, **kw: Any) -> None:
        st = clause_stats.setdefault(
            cid,
            {
                "clause_id": cid,
                "n_folds_selected": 0,
                "n_folds_holdout_gt_zero": 0,
                "n_folds_holdout_productive": 0,
                "worst_holdout_gt_hurt": 0,
                "sum_holdout_n_neg": 0,
                "definitions": [],
            },
        )
        for k, v in kw.items():
            if k == "definition":
                st["definitions"].append(v)
            elif k == "selected":
                st["n_folds_selected"] += 1
            elif k == "ho_gt0":
                st["n_folds_holdout_gt_zero"] += 1
            elif k == "ho_prod":
                st["n_folds_holdout_productive"] += 1
            elif k == "ho_gt":
                st["worst_holdout_gt_hurt"] = max(st["worst_holdout_gt_hurt"], int(v))
            elif k == "ho_neg":
                st["sum_holdout_n_neg"] += int(v)

    for hold in all_seqs:
        train_m = sequences != hold
        hold_m = sequences == hold
        if int(train_m.sum()) < 5 or int(hold_m.sum()) == 0:
            continue
        y_tr, y_ho = y[train_m], y[hold_m]
        # train unknown rows
        if len(unres_seq):
            u_tr = unres_seq != hold
        else:
            u_tr = np.array([], dtype=bool)
        if len(amb_seq):
            a_tr = amb_seq != hold
        else:
            a_tr = np.array([], dtype=bool)

        # --- single atoms: unique thr on train ---
        for sig in signals:
            x_tr = mats[sig][train_m]
            x_ho = mats[sig][hold_m]
            uniq = _sorted_unique(x_tr)
            for thr in uniq.tolist():
                for direction in DIRECTIONS:
                    m_tr = atom_mask(x_tr, direction, float(thr))
                    n_gt = int(np.sum(m_tr & (y_tr == 0)))
                    n_neg = int(np.sum(m_tr & (y_tr == 1)))
                    if n_gt != 0 or n_neg <= 0:
                        continue
                    # train unknown capture
                    n_u = 0
                    if len(unres) and u_tr.any():
                        n_u += int(
                            atom_mask(
                                unres_mats[sig][u_tr], direction, float(thr)
                            ).sum()
                        )
                    if len(amb) and a_tr.any():
                        n_u += int(
                            atom_mask(amb_mats[sig][a_tr], direction, float(thr)).sum()
                        )
                    if n_u > 0:
                        continue
                    # freeze and apply to holdout
                    m_ho = atom_mask(x_ho, direction, float(thr))
                    ho_gt = int(np.sum(m_ho & (y_ho == 0)))
                    ho_neg = int(np.sum(m_ho & (y_ho == 1)))
                    cid = _clause_id_single(sig, direction, float(thr))
                    _bump(
                        cid,
                        selected=1,
                        definition={
                            "kind": "single",
                            "feature": sig,
                            "direction": direction,
                            "thr_value": float(thr),
                            "hold_out": hold,
                        },
                        ho_gt=ho_gt,
                        ho_neg=ho_neg,
                    )
                    if ho_gt == 0:
                        _bump(cid, ho_gt0=1)
                    if ho_gt == 0 and ho_neg > 0:
                        _bump(cid, ho_prod=1)
                    fold_rows.append(
                        {
                            "hold_out_sequence": hold,
                            "clause_id": cid,
                            "kind": "single",
                            "feature_a": sig,
                            "direction_a": direction,
                            "thr_value_a": float(thr),
                            "train_n_neg": n_neg,
                            "train_gt_hurt": 0,
                            "holdout_gt_hurt": ho_gt,
                            "holdout_n_neg": ho_neg,
                        }
                    )

        # --- pairwise: quantile lattice on train ---
        q_thrs: dict[str, list[float]] = {
            s: _quantile_thresholds(mats[s][train_m], PAIRWISE_QUANTILE_LATTICE)
            for s in signals
        }
        for f1, f2 in itertools.combinations(list(signals), 2):
            for d1 in DIRECTIONS:
                for d2 in DIRECTIONS:
                    for t1 in q_thrs[f1]:
                        for t2 in q_thrs[f2]:
                            for comb in COMBINATORS:
                                m1 = atom_mask(mats[f1][train_m], d1, float(t1))
                                m2 = atom_mask(mats[f2][train_m], d2, float(t2))
                                m_tr = (m1 & m2) if comb == "AND" else (m1 | m2)
                                n_gt = int(np.sum(m_tr & (y_tr == 0)))
                                n_neg = int(np.sum(m_tr & (y_tr == 1)))
                                if n_gt != 0 or n_neg <= 0:
                                    continue
                                n_u = 0
                                if len(unres) and u_tr.any():
                                    u1 = atom_mask(unres_mats[f1][u_tr], d1, float(t1))
                                    u2 = atom_mask(unres_mats[f2][u_tr], d2, float(t2))
                                    n_u += int(
                                        (
                                            (u1 & u2) if comb == "AND" else (u1 | u2)
                                        ).sum()
                                    )
                                if len(amb) and a_tr.any():
                                    a1 = atom_mask(amb_mats[f1][a_tr], d1, float(t1))
                                    a2 = atom_mask(amb_mats[f2][a_tr], d2, float(t2))
                                    n_u += int(
                                        (
                                            (a1 & a2) if comb == "AND" else (a1 | a2)
                                        ).sum()
                                    )
                                if n_u > 0:
                                    continue
                                h1 = atom_mask(mats[f1][hold_m], d1, float(t1))
                                h2 = atom_mask(mats[f2][hold_m], d2, float(t2))
                                m_ho = (h1 & h2) if comb == "AND" else (h1 | h2)
                                ho_gt = int(np.sum(m_ho & (y_ho == 0)))
                                ho_neg = int(np.sum(m_ho & (y_ho == 1)))
                                cid = _clause_id_pair(
                                    comb, f1, d1, float(t1), f2, d2, float(t2)
                                )
                                _bump(
                                    cid,
                                    selected=1,
                                    definition={
                                        "kind": "pairwise",
                                        "combinator": comb,
                                        "feature_a": f1,
                                        "direction_a": d1,
                                        "thr_value_a": float(t1),
                                        "feature_b": f2,
                                        "direction_b": d2,
                                        "thr_value_b": float(t2),
                                        "hold_out": hold,
                                    },
                                    ho_gt=ho_gt,
                                    ho_neg=ho_neg,
                                )
                                if ho_gt == 0:
                                    _bump(cid, ho_gt0=1)
                                if ho_gt == 0 and ho_neg > 0:
                                    _bump(cid, ho_prod=1)
                                fold_rows.append(
                                    {
                                        "hold_out_sequence": hold,
                                        "clause_id": cid,
                                        "kind": f"pairwise_{comb}",
                                        "feature_a": f1,
                                        "direction_a": d1,
                                        "thr_value_a": float(t1),
                                        "feature_b": f2,
                                        "direction_b": d2,
                                        "thr_value_b": float(t2),
                                        "train_n_neg": n_neg,
                                        "train_gt_hurt": 0,
                                        "holdout_gt_hurt": ho_gt,
                                        "holdout_n_neg": ho_neg,
                                    }
                                )

    n_folds = len(all_seqs)
    summary_rows = []
    n_portable = 0
    for cid, st in sorted(
        clause_stats.items(), key=lambda kv: -kv[1]["n_folds_selected"]
    ):
        freq = st["n_folds_selected"] / n_folds if n_folds else 0.0
        # Exact absolute-clause repeatability (feature+dir+thr float@12dp),
        # not quantile/rank-coordinate region portability.
        portable = (
            st["n_folds_selected"] >= max(2, n_folds // 2)
            and st["worst_holdout_gt_hurt"] == 0
            and st["n_folds_holdout_productive"] >= 1
        )
        if portable:
            n_portable += 1
        summary_rows.append(
            {
                "clause_id": cid,
                "n_folds_selected": st["n_folds_selected"],
                "selection_frequency": freq,
                "worst_holdout_gt_hurt": st["worst_holdout_gt_hurt"],
                "n_folds_holdout_gt_zero": st["n_folds_holdout_gt_zero"],
                "n_folds_holdout_productive": st["n_folds_holdout_productive"],
                "sum_holdout_n_neg": st["sum_holdout_n_neg"],
                "exact_absolute_nested_loso_portability_ok": int(portable),
                # Alias kept for stability_rows / terminal wiring.
                "nested_loso_portability_ok": int(portable),
            }
        )

    return {
        "kind": "nested_loso_train_select_holdout_eval",
        "clause_identity": "exact_absolute_threshold_float_round12",
        "portability_definition": (
            "exact_absolute_clause_repeatability: same clause_id selected in "
            ">= max(2, n_folds//2) folds, worst holdout GT hurt==0, and at "
            "least one holdout productive (n_neg>0). Not quantile/rank "
            "coordinate region portability."
        ),
        "n_folds": n_folds,
        "n_clauses_ever_selected": len(clause_stats),
        "n_exact_absolute_clauses_nested_loso_portable": n_portable,
        # Alias: prefer n_exact_absolute_clauses_nested_loso_portable in claims.
        "n_clauses_nested_loso_portable": n_portable,
        "fold_detail_rows": fold_rows,
        "clause_summary_rows": summary_rows,
        "not_fixed_full_sample_partition": True,
    }


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
    # Per grammar/grid only — never global across feature pairs.
    # Grid = (feature_a, direction_a, feature_b, direction_b, combinator).
    # semantic_duplicate_mask means "same mask already seen in this grid";
    # coordinates still feed quotient topology (see classify_region_stability).
    seen_sig_by_grid: dict[tuple[Any, ...], set[bytes]] = defaultdict(set)

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
                grid_key = (
                    a["feature"],
                    a["direction"],
                    b["feature"],
                    b["direction"],
                    combinator,
                )
                mask_bytes = m.tobytes()
                is_dup = mask_bytes in seen_sig_by_grid[grid_key]
                seen_sig_by_grid[grid_key].add(mask_bytes)

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


def _connected_components_1d(coords: set[int]) -> dict[int, int]:
    """Map thr_index → component id (adjacent indices share a component)."""
    if not coords:
        return {}
    ordered = sorted(coords)
    parent: dict[int, int] = {c: c for c in ordered}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ordered) - 1):
        if ordered[i + 1] - ordered[i] == 1:
            union(ordered[i], ordered[i + 1])
    # compact ids
    roots = {find(c) for c in ordered}
    root_to_id = {r: i for i, r in enumerate(sorted(roots))}
    return {c: root_to_id[find(c)] for c in ordered}


def _connected_components_2d(
    coords: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Map (i,j) → component id (4-neighborhood)."""
    if not coords:
        return {}
    parent: dict[tuple[int, int], tuple[int, int]] = {c: c for c in coords}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in coords:
        for di, dj in ((1, 0), (0, 1)):
            n = (i + di, j + dj)
            if n in coords:
                union((i, j), n)
    roots = {find(c) for c in coords}
    root_to_id = {r: i for i, r in enumerate(sorted(roots))}
    return {c: root_to_id[find(c)] for c in coords}


def classify_region_stability(
    atom_rows: Sequence[Mapping[str, Any]],
    pairwise_and: Sequence[Mapping[str, Any]],
    pairwise_or: Sequence[Mapping[str, Any]],
    *,
    nested_loso: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Quotient topology with safe-region thickness on coordinate union.

    Two layers per grammar/grid:

    1. **unique-mask node** — retains all thr coordinates sharing that mask
       (prediction-invariant plateau). Semantic-duplicate cells still
       contribute coordinates.
    2. **productive-safe coordinate union** — interior / thickness is computed
       on the full set of productive-safe lattice coordinates, *regardless of
       mask identity*. Adjacent thr cells that are all productive-safe form a
       thick safe region even when masks differ.

    Same-mask plateau width remains reported as prediction-invariant thickness
    only; it is not the sole gate for interior.

    Nested LOSO exact-absolute portability is reported separately and required
    for terminal A.
    """
    nested_loso = nested_loso or {}
    # clause_id -> nested exact-absolute portability
    portable_clauses: set[str] = set()
    for row in nested_loso.get("clause_summary_rows") or []:
        ok = int(
            row.get(
                "exact_absolute_nested_loso_portability_ok",
                row.get("nested_loso_portability_ok", 0),
            )
        )
        if ok:
            portable_clauses.add(str(row["clause_id"]))

    out: list[dict[str, Any]] = []

    def _prod(r: Mapping[str, Any]) -> bool:
        return (
            bool(int(r.get("productive_safe_point", 0)))
            and str(r.get("safety_status", "")) != "unresolved_contaminated"
        )

    def _label_and_cand(
        *,
        has_interior: bool,
        n_coords: int,
        n_adj_masks: int,
        n_seq_neg: int,
        single_dom: bool,
        portable: bool,
    ) -> tuple[str, int]:
        if not has_interior:
            if n_coords <= 1 and n_adj_masks == 0:
                label = "isolated_safe_point"
            else:
                label = "edge_candidate"
        elif n_seq_neg < MIN_SEQS_FOR_REGION or single_dom:
            label = "locally_stable_region_but_seq_thin"
        elif portable:
            label = "loo_stable_region"
        else:
            label = "locally_stable_region"
        is_cand = int(
            label in ("loo_stable_region", "locally_stable_region") and has_interior
        )
        return label, is_cand

    # ---- single-atom grids ----
    by_grid: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in atom_rows:
        if int(r.get("is_secondary_feature", 0)):
            continue
        if not _prod(r):
            continue
        by_grid[(str(r["feature"]), str(r["direction"]))].append(r)

    for (feat, direction), cells in by_grid.items():
        mask_coords: dict[str, list[int]] = defaultdict(list)
        mask_rep: dict[str, Mapping[str, Any]] = {}
        mask_all: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in cells:
            sig = str(r.get("mask_sha256") or "")
            if not sig:
                continue
            mask_coords[sig].append(int(r["thr_index"]))
            mask_all[sig].append(r)
            if sig not in mask_rep:
                mask_rep[sig] = r

        # Layer 2: full productive-safe coordinate union (any mask).
        safe_coords: set[int] = set()
        for coords in mask_coords.values():
            safe_coords.update(coords)
        safe_interior = {
            i for i in safe_coords if (i - 1 in safe_coords and i + 1 in safe_coords)
        }
        comp_of = _connected_components_1d(safe_coords)
        # component id → (n_coords, n_unique_masks)
        comp_coords: dict[int, set[int]] = defaultdict(set)
        comp_masks: dict[int, set[str]] = defaultdict(set)
        for sig, coords in mask_coords.items():
            for i in set(coords):
                cid_c = comp_of[i]
                comp_coords[cid_c].add(i)
                comp_masks[cid_c].add(sig)

        for sig, coords in mask_coords.items():
            cset = set(coords)
            width = max(coords) - min(coords) + 1 if coords else 0
            n_coords = len(cset)
            # Safe-region interior: neighborhood in *union*, not same-mask only.
            mask_interior = cset & safe_interior
            has_interior = len(mask_interior) > 0
            n_interior = len(mask_interior)
            # Prediction-invariant same-mask plateau (diagnostic only).
            same_mask_plateau_has_interior = any(
                (i - 1 in cset and i + 1 in cset) for i in cset
            )
            n_adj_masks = 0
            for other, ocoords in mask_coords.items():
                if other == sig:
                    continue
                oset = set(ocoords)
                if any(abs(i - j) == 1 for i in cset for j in oset):
                    n_adj_masks += 1
            # Component of first coordinate (mask may span one component typically)
            first_c = next(iter(cset))
            safe_comp = int(comp_of[first_c])
            rep = mask_rep[sig]
            n_seq_neg = int(rep.get("n_sequences_with_neg", 0))
            single_dom = bool(int(rep.get("single_seq_neg_dominance", 0)))
            cid = _clause_id_single(feat, direction, float(rep["thr_value"]))
            portable = False
            for r in mask_all[sig]:
                cid_i = _clause_id_single(feat, direction, float(r["thr_value"]))
                if cid_i in portable_clauses:
                    portable = True
                    cid = cid_i
                    break

            label, is_cand = _label_and_cand(
                has_interior=has_interior,
                n_coords=n_coords,
                n_adj_masks=n_adj_masks,
                n_seq_neg=n_seq_neg,
                single_dom=single_dom,
                portable=portable,
            )
            out.append(
                {
                    "region_id": f"mask::{feat}::{direction}::{sig[:12]}",
                    "kind": "single_atom_quotient",
                    "feature_a": feat,
                    "direction_a": direction,
                    "mask_sha256": sig,
                    "n_coordinates": n_coords,
                    "plateau_width_thr_index": width,
                    "same_mask_plateau_has_interior": int(
                        same_mask_plateau_has_interior
                    ),
                    "has_interior_coordinate": int(has_interior),
                    "has_any_interior_coordinate": int(has_interior),
                    "n_interior_coordinates": n_interior,
                    "safe_component_id": safe_comp,
                    "component_size_coordinates": len(comp_coords[safe_comp]),
                    "component_size_unique_masks": len(comp_masks[safe_comp]),
                    "n_adjacent_other_masks": n_adj_masks,
                    "support": rep["support"],
                    "n_neg_captured": rep["n_neg_captured"],
                    "gt_hurt": rep["gt_hurt"],
                    "n_unresolved_selected": rep.get("n_unresolved_selected", 0),
                    "safety_status": rep.get("safety_status"),
                    "n_sequences_with_neg": n_seq_neg,
                    "max_neg_sequence_share": rep.get("max_neg_sequence_share"),
                    "nested_loso_portability_ok": int(portable),
                    "exact_absolute_nested_loso_portability_ok": int(portable),
                    "nested_loso_clause_id": cid,
                    "stability_class": label,
                    "is_region_candidate": is_cand,
                    "not_a_safe_rule": 1,
                    "topology_note": (
                        "safe_region_interior_on_productive_safe_coordinate_union;"
                        "same_mask_plateau_is_prediction_invariant_only"
                    ),
                }
            )

    # ---- pairwise grids (per feature-pair × dirs × comb) ----
    def _pair_grid_key(r: Mapping[str, Any]) -> tuple:
        return (
            r["feature_a"],
            r["direction_a"],
            r["feature_b"],
            r["direction_b"],
            r["combinator"],
        )

    for _comb_name, rows in (("AND", pairwise_and), ("OR", pairwise_or)):
        by_pg: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
        for r in rows:
            if not _prod(r):
                continue
            by_pg[_pair_grid_key(r)].append(r)
        for gkey, cells in by_pg.items():
            fa, da, fb, db, comb = gkey
            mask_coords: dict[str, list[tuple[int, int]]] = defaultdict(list)
            mask_rep: dict[str, Mapping[str, Any]] = {}
            mask_all: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for r in cells:
                sig = str(r.get("mask_sha256") or "")
                if not sig:
                    continue
                coord = (int(r["thr_index_a"]), int(r["thr_index_b"]))
                mask_coords[sig].append(coord)
                mask_all[sig].append(r)
                if sig not in mask_rep:
                    mask_rep[sig] = r

            safe_coords_2d: set[tuple[int, int]] = set()
            for coords in mask_coords.values():
                safe_coords_2d.update(coords)
            safe_interior_2d = {
                (i, j)
                for i, j in safe_coords_2d
                if (
                    (i - 1, j) in safe_coords_2d
                    and (i + 1, j) in safe_coords_2d
                    and (i, j - 1) in safe_coords_2d
                    and (i, j + 1) in safe_coords_2d
                )
            }
            comp_of_2d = _connected_components_2d(safe_coords_2d)
            comp_coords_2d: dict[int, set[tuple[int, int]]] = defaultdict(set)
            comp_masks_2d: dict[int, set[str]] = defaultdict(set)
            for sig, coords in mask_coords.items():
                for c in set(coords):
                    cid_c = comp_of_2d[c]
                    comp_coords_2d[cid_c].add(c)
                    comp_masks_2d[cid_c].add(sig)

            for sig, coords in mask_coords.items():
                cset = set(coords)
                if not coords:
                    continue
                as_ = [c[0] for c in coords]
                bs_ = [c[1] for c in coords]
                width_a = max(as_) - min(as_) + 1
                width_b = max(bs_) - min(bs_) + 1
                mask_interior = cset & safe_interior_2d
                has_interior = len(mask_interior) > 0
                n_interior = len(mask_interior)
                same_mask_plateau_has_interior = any(
                    (
                        (i - 1, j) in cset
                        and (i + 1, j) in cset
                        and (i, j - 1) in cset
                        and (i, j + 1) in cset
                    )
                    for i, j in cset
                )
                n_adj_masks = 0
                for other, ocoords in mask_coords.items():
                    if other == sig:
                        continue
                    oset = set(ocoords)
                    if any(
                        abs(i - oi) + abs(j - oj) == 1
                        for i, j in cset
                        for oi, oj in oset
                    ):
                        n_adj_masks += 1
                first_c = next(iter(cset))
                safe_comp = int(comp_of_2d[first_c])
                rep = mask_rep[sig]
                n_seq_neg = int(rep.get("n_sequences_with_neg", 0))
                single_dom = bool(int(rep.get("single_seq_neg_dominance", 0)))
                portable = False
                cid = ""
                for r in mask_all[sig]:
                    cid_i = _clause_id_pair(
                        comb,
                        str(r["feature_a"]),
                        str(r["direction_a"]),
                        float(r["thr_value_a"]),
                        str(r["feature_b"]),
                        str(r["direction_b"]),
                        float(r["thr_value_b"]),
                    )
                    if cid_i in portable_clauses:
                        portable = True
                        cid = cid_i
                        break
                label, is_cand = _label_and_cand(
                    has_interior=has_interior,
                    n_coords=len(cset),
                    n_adj_masks=n_adj_masks,
                    n_seq_neg=n_seq_neg,
                    single_dom=single_dom,
                    portable=portable,
                )
                out.append(
                    {
                        "region_id": (
                            f"mask::{comb}::{fa}::{da}::{fb}::{db}::{sig[:12]}"
                        ),
                        "kind": f"pairwise_{comb}_quotient",
                        "feature_a": fa,
                        "feature_b": fb,
                        "direction_a": da,
                        "direction_b": db,
                        "mask_sha256": sig,
                        "n_coordinates": len(cset),
                        "plateau_width_a": width_a,
                        "plateau_width_b": width_b,
                        "same_mask_plateau_has_interior": int(
                            same_mask_plateau_has_interior
                        ),
                        "has_interior_coordinate": int(has_interior),
                        "has_any_interior_coordinate": int(has_interior),
                        "n_interior_coordinates": n_interior,
                        "safe_component_id": safe_comp,
                        "component_size_coordinates": len(comp_coords_2d[safe_comp]),
                        "component_size_unique_masks": len(comp_masks_2d[safe_comp]),
                        "n_adjacent_other_masks": n_adj_masks,
                        "support": rep["support"],
                        "n_neg_captured": rep["n_neg_captured"],
                        "gt_hurt": rep["gt_hurt"],
                        "n_unresolved_selected": rep.get("n_unresolved_selected", 0),
                        "safety_status": rep.get("safety_status"),
                        "n_sequences_with_neg": n_seq_neg,
                        "max_neg_sequence_share": rep.get("max_neg_sequence_share"),
                        "nested_loso_portability_ok": int(portable),
                        "exact_absolute_nested_loso_portability_ok": int(portable),
                        "nested_loso_clause_id": cid,
                        "stability_class": label,
                        "is_region_candidate": is_cand,
                        "not_a_safe_rule": 1,
                        "topology_note": (
                            "safe_region_interior_on_productive_safe_coordinate_union;"
                            "same_mask_plateau_is_prediction_invariant_only"
                        ),
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
        and int(r.get("nested_loso_portability_ok", 0)) == 1
    ]

    # A: multi-seq, safe-coord-union interior, exact-absolute nested LOSO, zero unknown
    if loo_stable:
        return {
            "stage2_q45_terminal": TERMINAL_A,
            "terminal_letter": "A",
            "reason": (
                f"{len(loo_stable)} exact-absolute nested-LOSO-portable interior "
                "region(s) with thickness under productive-safe coordinate-union "
                "topology; authorize formal restricted safe-region candidate "
                "validation (still not production rule)"
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
        if r.get("stability_class")
        in ("isolated_safe_point", "thin_safe_edge", "edge_candidate")
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
                f"(n_atlas_cells={n_prod}) but no multi-seq thick region with "
                "productive-safe coordinate-union interior + exact-absolute "
                "nested LOSO portability; unknown/unresolved selected coverage "
                "further limits claims → isolated_safe_points_only"
            ),
            "claims_blocked": blocked
            + [
                "formal_safe_region_not_authorized",
                "threshold_formulation_global_closure_not_authorized",
                "portable_safe_region_claim_inadmissible",
                "deletion_loo_as_portability_forbidden",
                "fixed_full_sample_partition_as_portability_forbidden",
            ],
            "claims_allowed": [
                "retain_atlas_for_followup",
                "report_isolated_observed_safe_points_descriptively",
                "ranking_research_reasonable_next_line_not_threshold_closure",
            ],
            "next_authorized_step": (
                "threshold path: no promotable region yet; ranking/assignment "
                "is a reasonable next research line AFTER valid assignment-group "
                "key + unknown coverage + nested train-select holdout are closed"
            ),
            "bounded_finding": (
                "resolved∧selected restricted atlas reports sample-zero-GT cells; "
                "interior is measured on productive-safe coordinate union "
                "(not same-mask plateau alone); exact-absolute nested LOSO "
                "portability reported separately"
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


def _file_sha256_if_exists(path: Path) -> str:
    p = Path(path)
    return _sha256_file(p) if p.is_file() else ""


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
    # Wipe stale artifacts so evidence pack cannot copy a previous manifest.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_raw = load_d_online_events(q1q3_study_dir)
    source = q1q3_study_dir / "d_online_events.parquet"
    if not source.is_file():
        source = q1q3_study_dir / "d_online_events.csv"
    source_hash = _sha256_file(source) if source.is_file() else ""
    evaluator_src = Path(__file__).resolve()
    runner_src = Path("scripts/tools/run_m_b1_5_stage2_q45_atlas.py").resolve()
    evaluator_sha = _file_sha256_if_exists(evaluator_src)
    runner_sha = _file_sha256_if_exists(runner_src)

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

    # Nested LOSO portability (train lattice + select → freeze → holdout)
    nested_loso = nested_loso_portability_audit(
        primary,
        selected_unresolved=selected_unresolved,
        selected_ambiguous=selected_ambiguous,
        signals=ORDERED_SIGNALS,
    )

    stability = classify_region_stability(
        atom_rows, pairwise_and, pairwise_or, nested_loso=nested_loso
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

    # Nested LOSO artifacts before summary/manifest/pack (no stale copies).
    hashes["nested_loso_fold_detail"] = write_csv(
        out_dir / "nested_loso_fold_detail.csv",
        nested_loso.get("fold_detail_rows") or [],
    )
    hashes["nested_loso_clause_summary"] = write_csv(
        out_dir / "nested_loso_clause_summary.csv",
        nested_loso.get("clause_summary_rows") or [],
    )
    nested_loso_meta = {
        k: v
        for k, v in nested_loso.items()
        if k not in ("fold_detail_rows", "clause_summary_rows")
    }
    hashes["nested_loso_summary"] = write_json(
        out_dir / "nested_loso_summary.json", nested_loso_meta
    )

    summary = {
        "study_id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_q1q3_study": str(q1q3_study_dir),
        "upstream_q4_study": Q4_STUDY_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        "repository_head_sha": git_commit,
        "evaluator_source": str(evaluator_src),
        "evaluator_source_sha256": evaluator_sha,
        "runner_source": str(runner_src),
        "runner_source_sha256": runner_sha,
        "source_event_table": str(source),
        "source_event_table_sha256": source_hash,
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
        "n_region_candidates": int(terminal.get("n_region_candidates") or 0),
        "n_nested_loso_folds": nested_loso_meta.get("n_folds"),
        "n_clauses_ever_selected_nested_loso": nested_loso_meta.get(
            "n_clauses_ever_selected"
        ),
        "n_exact_absolute_clauses_nested_loso_portable": nested_loso_meta.get(
            "n_exact_absolute_clauses_nested_loso_portable",
            nested_loso_meta.get("n_clauses_nested_loso_portable"),
        ),
        # Alias — prefer n_exact_absolute_clauses_nested_loso_portable in claims.
        "n_clauses_nested_loso_portable": nested_loso_meta.get(
            "n_exact_absolute_clauses_nested_loso_portable",
            nested_loso_meta.get("n_clauses_nested_loso_portable"),
        ),
        "nested_loso_clause_identity": nested_loso_meta.get(
            "clause_identity", "exact_absolute_threshold_float_round12"
        ),
        "secondary_competition_features": secondary_names,
        "secondary_competition_trusted": False,
        "assignment_group_key_status": ASSIGNMENT_GROUP_KEY_STATUS,
        "n_selected_unresolved": len(selected_unresolved),
        "n_selected_ambiguous": len(selected_ambiguous),
        "evaluator_review_gates": {
            "deletion_loo_is_portability": False,
            "nested_loso_required_for_region_A": True,
            "nested_loso_is_exact_absolute_clause_repeatability": True,
            "fixed_full_sample_partition_not_portability": True,
            "unresolved_contaminated_blocks_candidate": True,
            "semantic_duplicate_is_per_grid_not_global": True,
            "quotient_topology_retains_all_coordinates": True,
            "interior_on_productive_safe_coordinate_union": True,
            "same_mask_plateau_is_prediction_invariant_only": True,
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
            "fixed_full_sample_partition_as_portability",
        ],
        "q4_interpretation": (
            "Q4 weak marginal AUC closes singleton frozen-tail thr promotion; "
            "does not forbid threshold/Boolean combination as analysis method."
        ),
    }
    hashes["summary"] = write_json(out_dir / "summary.json", summary)

    md = _render_md(summary, terminal, frame_prov, stability, pareto_all)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    hashes["summary_md"] = hashlib.sha256(md.encode("utf-8")).hexdigest()

    # Manifest last among primary artifacts (describes this run only).
    manifest = {
        "schema": "m_b1_5_stage2_q45_atlas_manifest_v4",
        "study_id": sid,
        "git_commit": git_commit,
        "repository_head_sha": git_commit,
        "taxonomy_version": TAXONOMY_VERSION,
        "evaluator_source": str(evaluator_src),
        "evaluator_source_sha256": evaluator_sha,
        "runner_source": str(runner_src),
        "runner_source_sha256": runner_sha,
        "source_q1q3_study": str(q1q3_study_dir),
        "source_event_table": str(source),
        "source_event_table_sha256": source_hash,
        "upstream_q4_study": Q4_STUDY_ID,
        "candidate_universe_id": CANDIDATE_UNIVERSE_ID,
        "substrate_id": SUBSTRATE_ID,
        "cohort_definition": cohort_sum["cohort_definition"],
        "sequence_set": sorted({str(r["sequence"]) for r in primary}),
        "artifact_hashes": hashes,
        "created_utc": summary["created_utc"],
        "terminal_letter": terminal.get("terminal_letter"),
        "stage2_q45_terminal": terminal.get("stage2_q45_terminal"),
        "n_exact_absolute_clauses_nested_loso_portable": nested_loso_meta.get(
            "n_exact_absolute_clauses_nested_loso_portable",
            nested_loso_meta.get("n_clauses_nested_loso_portable"),
        ),
        "n_clauses_nested_loso_portable": nested_loso_meta.get(
            "n_exact_absolute_clauses_nested_loso_portable",
            nested_loso_meta.get("n_clauses_nested_loso_portable"),
        ),
        "production_preset": "unchanged",
        "write_order": (
            "artifacts → summary → manifest → evidence_pack "
            "(pack copies current manifest only)"
        ),
    }
    hashes["manifest"] = write_json(out_dir / "manifest.json", manifest)

    # Evidence pack AFTER current manifest exists (never copy stale prior run).
    evidence_dir = out_dir / "evidence_pack"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
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
        "nested_loso_clause_summary.csv": out_dir / "nested_loso_clause_summary.csv",
        "nested_loso_summary.json": out_dir / "nested_loso_summary.json",
    }
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
    write_json(
        evidence_dir / "SHA256SUMS.json",
        {
            "taxonomy_version": TAXONOMY_VERSION,
            "git_commit": git_commit,
            "study_id": sid,
            "files": sha_rows,
        },
    )
    write_json(
        evidence_dir / "README.json",
        {
            "study_id": sid,
            "taxonomy_version": TAXONOMY_VERSION,
            "git_commit": git_commit,
            "evaluator_source_sha256": evaluator_sha,
            "runner_source_sha256": runner_sha,
            "source_event_table_sha256": source_hash,
            "note": (
                "Canonical evidence pack for PR audit. Large full atlases remain "
                "in parent study dir; rebuild via run_m_b1_5_stage2_q45_atlas.py. "
                "manifest.json is written for this run before pack copy."
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
