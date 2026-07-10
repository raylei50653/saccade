"""M-B1.5 Stage 2 Q1–Q3: D_online label join + safe-negative mass audit.

Research-only. Does **not** search thresholds, Boolean rules, or change presets.

Contract:
  docs/modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md

Primary substrate:
  Stage 1 online B-audit full event table (244 rows).
  Labels joined via A1 MOT + global_id_map + trajectory majority vote vs GT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from saccade.perception.eval.portable_or_tail import (
    FROZEN_THR_VECTOR,
    ORDERED_SIGNALS,
)

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

TAXONOMY_VERSION = "stage2_q1q3_v1"
EXPECTED_D_ONLINE_N = 244
AUTHORITATIVE_STAGE1_STUDY = "m_b1_hook_ab_20260710T071001Z_stage1_close"
JOIN_METHOD = "a1_mot_global_id_map+traj_centerdist_majority"
LABEL_SOURCE = "MOT17_train_gt + A1_hook_off MOT trajectories"
CANDIDATE_UNIVERSE_ID = "online_hook_eligible"
SUBSTRATE_ID = "stage1_baudit_d_online"

# Majority-vote confidence (documented join contract).
# Track maps to GT if top vote count >= MIN_VOTES, or short traj with frac>=MIN_VOTE_FRAC.
MIN_VOTES = 3
MIN_VOTE_FRAC = 0.3
SHORT_TRAJ_LEN = 5  # len<=5 may resolve with votes covering frac
CENTERDIST_THR_MULT = 1.0

# Single-sequence dominance of safe-removable mass.
DOMINANCE_MAX_SHARE = 0.5

# Minimum absolute safe-removable mass to call Q3 SUFFICIENT (not thr-cal ready).
# Intent: non-zero multi-seq decision-relevant mass for separability entry.
Q3_SUFFICIENT_MIN_SAFE_REMOVABLE = 1
Q3_SUFFICIENT_MIN_SEQS_WITH_SAFE = 2

SIGNAL_NAMES: tuple[str, ...] = ORDERED_SIGNALS

NEGATIVE_STATUS_VALUES: tuple[str, ...] = (
    "not_negative",
    "negative_unresolved_effect",
    "negative_decision_neutral",
    "negative_decision_relevant",
    "negative_safe_removable",
    "negative_unsafe_or_ambiguous",
)

LABEL_STATUS_VALUES: tuple[str, ...] = ("resolved", "unresolved", "ambiguous")
PAIR_LABEL_VALUES: tuple[str, ...] = (
    "gt_consistent",
    "negative",
    "ambiguous",
    "unknown",
)
DECISION_RELEVANCE_VALUES: tuple[str, ...] = (
    "selected",
    "active_competitor",
    "non_selected",
    "no_effect_path",
    "unresolved",
)


class Stage2AuditError(ValueError):
    """Fail-closed Stage 2 audit / join error."""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> str:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256_bytes(text.encode("utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        path.write_text("", encoding="utf-8")
        return _sha256_bytes(b"")
    # Union keys across rows (first-row-only drops heterogeneous fields,
    # e.g. single-atom vs pairwise stability rows).
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return _sha256_bytes(path.read_bytes())


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        if rows:
            pq.write_table(pa.Table.from_pylist(list(rows)), path)
        else:
            pq.write_table(pa.Table.from_pylist([]), path)
        return True
    except Exception:
        return False


def load_event_table(path: Path) -> list[dict[str, Any]]:
    """Load Stage 1 hook_candidate_events from parquet or csv."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        cols = table.column_names
        col_data = [table.column(i).to_pylist() for i in range(len(cols))]
        n = len(col_data[0]) if col_data else 0
        return [{cols[j]: col_data[j][i] for j in range(len(cols))} for i in range(n)]
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_event_table_from_study(study_dir: Path) -> list[dict[str, Any]]:
    study_dir = Path(study_dir)
    pq = study_dir / "hook_candidate_events.parquet"
    csv_p = study_dir / "hook_candidate_events.csv"
    if pq.is_file():
        return load_event_table(pq)
    if csv_p.is_file():
        return load_event_table(csv_p)
    raise Stage2AuditError(f"no hook_candidate_events in {study_dir}")


# ---------------------------------------------------------------------------
# GT / MOT loaders + majority map
# ---------------------------------------------------------------------------


def parse_global_id_map(path: Path) -> dict[tuple[str, int], int]:
    """Parse evaluator ``_global_id_map.txt``: seq\\tlocal_id=N\\tglobal_id=M."""
    out: dict[tuple[str, int], int] = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        seq = parts[0].strip()
        try:
            local = int(parts[1].split("=")[1])
            glob = int(parts[2].split("=")[1])
        except (IndexError, ValueError):
            continue
        out[(seq, local)] = glob
    return out


def load_mot_boxes(
    path: Path, *, drop_nonpos_id: bool = False
) -> dict[int, dict[int, tuple[float, float, float, float, float, float]]]:
    """tid -> frame -> (x, y, w, h, cx, cy)."""
    tracks: dict[int, dict[int, tuple[float, float, float, float, float, float]]] = (
        defaultdict(dict)
    )
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, tid = int(float(p[0])), int(float(p[1]))
            if drop_nonpos_id and tid <= 0:
                continue
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            tracks[tid][frm] = (x, y, w, h, x + w / 2.0, y + h / 2.0)
    return dict(tracks)


@dataclass(frozen=True)
class TrackGtMap:
    gt_id: int
    votes: int
    votes2: int
    traj_len: int
    frac: float

    @property
    def conf_ok(self) -> bool:
        if self.traj_len <= 0 or self.votes <= 0:
            return False
        if self.frac < MIN_VOTE_FRAC:
            return False
        if self.votes >= MIN_VOTES:
            return True
        # Short trajectories: allow if majority covers MIN_VOTE_FRAC.
        return self.traj_len <= SHORT_TRAJ_LEN


def map_tracks_to_gt_majority(
    tracks: Mapping[int, Mapping[int, tuple[float, float, float, float, float, float]]],
    gt_tracks: Mapping[
        int, Mapping[int, tuple[float, float, float, float, float, float]]
    ],
    *,
    thr_mult: float = CENTERDIST_THR_MULT,
) -> dict[int, TrackGtMap]:
    """Center-distance majority vote (same family as build_relink_candidates)."""
    gt_by_frame: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    for gid, frd in gt_tracks.items():
        for f, (_x, _y, _w, h, cx, cy) in frd.items():
            gt_by_frame[f].append((gid, cx, cy, h))

    mapping: dict[int, TrackGtMap] = {}
    for tid, frd in tracks.items():
        votes: dict[int, int] = defaultdict(int)
        for f, (_x, _y, _w, h, cx, cy) in frd.items():
            best_gid, best_d = -1, 1e30
            for gid, gx, gy, gh in gt_by_frame.get(f, ()):
                d = math.hypot(cx - gx, cy - gy)
                if d < max(h, gh) * thr_mult and d < best_d:
                    best_d, best_gid = d, gid
            if best_gid >= 0:
                votes[best_gid] += 1
        if not votes:
            continue
        ordered = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        gid, n = ordered[0]
        n2 = ordered[1][1] if len(ordered) > 1 else 0
        tlen = len(frd)
        mapping[tid] = TrackGtMap(
            gt_id=int(gid),
            votes=int(n),
            votes2=int(n2),
            traj_len=int(tlen),
            frac=float(n) / float(tlen) if tlen else 0.0,
        )
    return mapping


# ---------------------------------------------------------------------------
# Decision context enrichment
# ---------------------------------------------------------------------------


def enrich_decision_context(rows: list[dict[str, Any]]) -> None:
    """Add competitor_count / best_competitor / baseline_selected per group.

    Group key: (sequence, frame, cand_slot) — same as attach_decision_fields.
    """
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["sequence"], int(r["frame"]), int(r.get("cand_slot", -1)))
        groups[key].append(i)

    for idxs in groups.values():
        ordered = sorted(
            idxs,
            key=lambda i: (
                int(rows[i].get("baseline_rank", 10**9)),
                float(rows[i].get("score_m_bridge", 0.0)),
                int(rows[i].get("lost_slot", -1)),
            ),
        )
        n = len(ordered)
        best_i = ordered[0] if ordered else None
        second_i = ordered[1] if n >= 2 else None
        best_id = rows[best_i].get("runtime_candidate_id") if best_i is not None else ""
        second_id = (
            rows[second_i].get("runtime_candidate_id") if second_i is not None else ""
        )
        for rank_pos, i in enumerate(ordered):
            rows[i]["competitor_count"] = n - 1
            rows[i]["baseline_selected"] = int(
                rows[i].get("baseline_accepted_candidate", 0)
            )
            rows[i]["best_competitor_id"] = second_id if rank_pos == 0 else best_id
            rows[i]["best_competitor_rank"] = (
                1
                if rank_pos == 0 and second_i is not None
                else (0 if rank_pos > 0 else -1)
            )
            # assignment_context join key (stable, non-float)
            rows[i]["assignment_context"] = (
                f"{rows[i]['sequence']}|{int(rows[i]['frame'])}|"
                f"{int(rows[i].get('cand_slot', -1))}"
            )


def classify_decision_relevance(row: Mapping[str, Any]) -> str:
    selected = int(
        row.get("baseline_selected", row.get("baseline_accepted_candidate", 0))
    )
    recon = int(row.get("baseline_reconnect_decision", selected))
    comp = int(row.get("competitor_count", 0))
    if selected == 1:
        return "selected"
    if recon == 0 and selected == 0 and comp == 0:
        # Sole candidate that was not accepted — rare; treat as no_effect if rank bad
        rank = int(row.get("baseline_rank", -1))
        if rank < 0:
            return "unresolved"
        return "non_selected"
    if comp >= 1:
        return "active_competitor"
    return "non_selected"


# ---------------------------------------------------------------------------
# Label join
# ---------------------------------------------------------------------------


@dataclass
class LabelJoinResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    errors: list[dict[str, Any]]


def _coerce_event_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize dtypes from parquet/csv into a mutable dict."""
    r = dict(raw)
    for k in (
        "frame",
        "cand_slot",
        "lost_slot",
        "cand_track_id",
        "lost_track_id",
        "atom_bitmask",
        "n_atoms_fired",
        "rejected_by_hook",
        "baseline_rank",
        "hook_rank",
        "baseline_accepted_candidate",
        "hook_accepted_candidate",
        "baseline_reconnect_decision",
        "hook_reconnect_decision",
        "accepted_competitor_changed",
        "reconnect_decision_changed",
        "local_assignment_changed",
        "downstream_identity_changed",
    ):
        if k in r and r[k] is not None and r[k] != "":
            try:
                r[k] = int(float(r[k]))
            except (TypeError, ValueError):
                pass
    for k in SIGNAL_NAMES + (
        "thr_0",
        "thr_1",
        "thr_2",
        "thr_3",
        "thr_4",
        "thr_margin_0",
        "thr_margin_1",
        "thr_margin_2",
        "thr_margin_3",
        "thr_margin_4",
    ):
        if k in r and r[k] is not None and r[k] != "":
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    return r


def join_labels_to_d_online(
    events: Sequence[Mapping[str, Any]],
    *,
    mot_dir: Path,
    gt_root: Path,
    global_id_map_path: Path,
    study_id: str,
    source_event_table: str,
    policy_file_hash: str | None = None,
) -> LabelJoinResult:
    """Join GT/FP labels onto every D_online row via stable event_id.

    Fail-closed on duplicate event_id / join_key.
    Unresolved rows stay ``pair_label=unknown`` (never defaulted to negative).
    """
    rows = [_coerce_event_row(e) for e in events]
    errors: list[dict[str, Any]] = []

    # Uniqueness fail-closed
    event_ids = [str(r.get("event_id", "")) for r in rows]
    join_keys = [str(r.get("join_key", "")) for r in rows]
    if len(event_ids) != len(set(event_ids)):
        c = Counter(event_ids)
        dups = [k for k, v in c.items() if v > 1]
        raise Stage2AuditError(f"duplicate event_id (fail-closed): {dups[:5]}")
    if len(join_keys) != len(set(join_keys)):
        c = Counter(join_keys)
        dups = [k for k, v in c.items() if v > 1]
        raise Stage2AuditError(f"duplicate join_key (fail-closed): {dups[:5]}")
    if any(not eid for eid in event_ids):
        raise Stage2AuditError("empty event_id present (fail-closed)")

    gid_map = parse_global_id_map(global_id_map_path)
    sequences = sorted({str(r["sequence"]) for r in rows})

    # Per-sequence MOT→GT maps
    seq_maps: dict[str, dict[int, TrackGtMap]] = {}
    seq_meta: dict[str, Any] = {}
    for seq in sequences:
        mot_path = Path(mot_dir) / f"{seq}.txt"
        gt_path = Path(gt_root) / seq / "gt" / "gt.txt"
        if not mot_path.is_file():
            raise Stage2AuditError(f"missing MOT results: {mot_path}")
        if not gt_path.is_file():
            raise Stage2AuditError(f"missing GT: {gt_path}")
        mot = load_mot_boxes(mot_path)
        gt = load_mot_boxes(gt_path, drop_nonpos_id=True)
        seq_maps[seq] = map_tracks_to_gt_majority(mot, gt)
        seq_meta[seq] = {
            "n_mot_ids": len(mot),
            "n_gt_ids": len(gt),
            "n_mapped_tracks": len(seq_maps[seq]),
            "mot_path": str(mot_path),
            "gt_path": str(gt_path),
        }

    enrich_decision_context(rows)

    n_joined = 0
    n_unresolved = 0
    n_ambiguous = 0
    missing_join_keys = 0
    conflicting = 0

    for r in rows:
        seq = str(r["sequence"])
        lost_local = int(r["lost_track_id"])
        cand_local = int(r["cand_track_id"])
        gl = gid_map.get((seq, lost_local))
        gc = gid_map.get((seq, cand_local))

        r["study_id"] = study_id
        r["source_event_table"] = source_event_table
        r["candidate_universe_id"] = CANDIDATE_UNIVERSE_ID
        r["substrate_id"] = SUBSTRATE_ID
        r["taxonomy_version"] = TAXONOMY_VERSION
        r["join_method"] = JOIN_METHOD
        if policy_file_hash and not r.get("policy_file_hash"):
            r["policy_file_hash"] = policy_file_hash
        # Alias frozen thr columns
        for i in range(5):
            thr_key = f"thr_{i}"
            r[f"frozen_thr_{i}"] = float(r.get(thr_key, FROZEN_THR_VECTOR[i]))
            mar = r.get(f"thr_margin_{i}")
            r[f"margin_{i}"] = (
                float(mar) if mar is not None and mar != "" else float("nan")
            )

        r["lost_global_id"] = gl if gl is not None else -1
        r["cand_global_id"] = gc if gc is not None else -1
        r["gt_lost"] = -1
        r["gt_cand"] = -1
        r["gt_lost_votes"] = 0
        r["gt_cand_votes"] = 0
        r["gt_lost_frac"] = float("nan")
        r["gt_cand_frac"] = float("nan")
        r["gt_lost_traj_len"] = 0
        r["gt_cand_traj_len"] = 0
        r["label_observed"] = "inferred_counterfactual"
        r["join_error"] = ""

        reason = ""
        if gl is None or gc is None:
            reason = "missing_global_id_map"
            missing_join_keys += 1
            if gl is None and gc is None:
                detail = "both_local_unmapped"
            elif gl is None:
                detail = "lost_local_unmapped"
            else:
                detail = "cand_local_unmapped"
            errors.append(
                {
                    "event_id": r["event_id"],
                    "sequence": seq,
                    "error": reason,
                    "detail": detail,
                    "lost_track_id": lost_local,
                    "cand_track_id": cand_local,
                }
            )
            r["join_error"] = f"{reason}:{detail}"
            r["label_status"] = "unresolved"
            r["pair_label"] = "unknown"
            r["gt_match"] = None
            n_unresolved += 1
        else:
            ml = seq_maps[seq].get(gl)
            mc = seq_maps[seq].get(gc)
            if ml is None or mc is None:
                reason = "track_gt_unmapped"
                detail = (
                    "both"
                    if ml is None and mc is None
                    else ("lost" if ml is None else "cand")
                )
                errors.append(
                    {
                        "event_id": r["event_id"],
                        "sequence": seq,
                        "error": reason,
                        "detail": detail,
                        "lost_global_id": gl,
                        "cand_global_id": gc,
                    }
                )
                r["join_error"] = f"{reason}:{detail}"
                r["label_status"] = "unresolved"
                r["pair_label"] = "unknown"
                r["gt_match"] = None
                n_unresolved += 1
            else:
                r["gt_lost"] = ml.gt_id
                r["gt_cand"] = mc.gt_id
                r["gt_lost_votes"] = ml.votes
                r["gt_cand_votes"] = mc.votes
                r["gt_lost_frac"] = ml.frac
                r["gt_cand_frac"] = mc.frac
                r["gt_lost_traj_len"] = ml.traj_len
                r["gt_cand_traj_len"] = mc.traj_len
                conf_ok = ml.conf_ok and mc.conf_ok
                # Near-tie on either side → ambiguous
                near_tie = (
                    ml.votes2 > 0 and ml.votes2 >= ml.votes - 1 and ml.frac < 0.6
                ) or (mc.votes2 > 0 and mc.votes2 >= mc.votes - 1 and mc.frac < 0.6)
                if not conf_ok or near_tie:
                    r["label_status"] = "ambiguous"
                    r["pair_label"] = "ambiguous"
                    r["gt_match"] = None
                    r["join_error"] = (
                        "low_confidence_or_near_tie" if not conf_ok else "near_tie"
                    )
                    n_ambiguous += 1
                    if near_tie:
                        conflicting += 1
                else:
                    same = ml.gt_id == mc.gt_id
                    r["label_status"] = "resolved"
                    r["pair_label"] = "gt_consistent" if same else "negative"
                    r["gt_match"] = int(same)
                    n_joined += 1

        # Decision relevance + baseline outcome
        rel = classify_decision_relevance(r)
        r["decision_relevance"] = rel
        pl = r["pair_label"]
        if pl == "unknown":
            r["baseline_outcome"] = "unresolved"
        elif rel != "selected":
            r["baseline_outcome"] = "neutral"
        elif pl == "gt_consistent":
            r["baseline_outcome"] = "correct"
        elif pl == "negative":
            r["baseline_outcome"] = "incorrect"
        else:
            r["baseline_outcome"] = "unresolved"

        # Negative taxonomy (mutually exclusive)
        r["negative_status"] = classify_negative_status(r)
        r["safe_removal_resolvable"] = int(
            r["negative_status"] == "negative_safe_removable"
        )
        r["label_inference_kind"] = (
            "observed_intervention"
            if int(r.get("rejected_by_hook", 0))
            else "inferred_counterfactual"
        )

    n_total = len(rows)
    join_coverage = float(n_joined) / float(n_total) if n_total else 0.0
    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "join_method": JOIN_METHOD,
        "label_source": LABEL_SOURCE,
        "n_total": n_total,
        "n_joined": n_joined,
        "n_resolved": n_joined,  # alias: resolved == joined with conf
        "n_unresolved": n_unresolved,
        "n_ambiguous": n_ambiguous,
        "join_coverage": join_coverage,
        "duplicate_join_keys": 0,
        "duplicate_event_ids": 0,
        "missing_join_keys": missing_join_keys,
        "conflicting_labels": conflicting,
        "min_votes": MIN_VOTES,
        "min_vote_frac": MIN_VOTE_FRAC,
        "centerdist_thr_mult": CENTERDIST_THR_MULT,
        "per_sequence_map_meta": seq_meta,
        "global_id_map_path": str(global_id_map_path),
        "global_id_map_entries": len(gid_map),
        "fail_closed_on_duplicate_keys": True,
        "unresolved_not_defaulted_to_negative": True,
    }
    return LabelJoinResult(rows=rows, summary=summary, errors=errors)


def classify_negative_status(row: Mapping[str, Any]) -> str:
    """Mutually exclusive negative taxonomy.

    ``negative_safe_removable`` (inferred): pair is resolved negative **and**
    baseline-selected. Rejecting it removes a wrong reconnect without removing
    a GT-consistent selected bridge. Not an observed intervention outcome when
    ``rejected_by_hook==0``.
    """
    pl = str(row.get("pair_label", "unknown"))
    rel = str(row.get("decision_relevance", "unresolved"))
    if pl != "negative":
        if pl in ("gt_consistent", "ambiguous", "unknown"):
            return "not_negative"
        return "not_negative"
    # negative
    if rel == "selected":
        return "negative_safe_removable"
    if rel in ("active_competitor", "non_selected"):
        return "negative_decision_neutral"
    if rel == "unresolved":
        return "negative_unresolved_effect"
    return "negative_unresolved_effect"


# ---------------------------------------------------------------------------
# Population / support / safe-negative summaries (group-by only)
# ---------------------------------------------------------------------------


def _count_where(rows: Sequence[Mapping[str, Any]], pred) -> int:
    return sum(1 for r in rows if pred(r))


def build_population_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_resolved = _count_where(rows, lambda r: r.get("label_status") == "resolved")
    n_unresolved = _count_where(rows, lambda r: r.get("label_status") == "unresolved")
    n_ambiguous = _count_where(rows, lambda r: r.get("label_status") == "ambiguous")
    n_gt = _count_where(rows, lambda r: r.get("pair_label") == "gt_consistent")
    n_neg = _count_where(rows, lambda r: r.get("pair_label") == "negative")
    n_pl_amb = _count_where(rows, lambda r: r.get("pair_label") == "ambiguous")
    n_unknown = _count_where(rows, lambda r: r.get("pair_label") == "unknown")
    n_sel = _count_where(rows, lambda r: int(r.get("baseline_selected", 0)) == 1)
    n_nonsel = n - n_sel
    n_dec_rel = _count_where(rows, lambda r: r.get("decision_relevance") == "selected")
    n_dec_neutral = _count_where(
        rows,
        lambda r: (
            r.get("decision_relevance")
            in ("active_competitor", "non_selected", "no_effect_path")
        ),
    )
    n_safe_res = _count_where(
        rows, lambda r: int(r.get("safe_removal_resolvable", 0)) == 1
    )
    return {
        "funnel": {
            "D_online_total": n,
            "label_resolved": n_resolved,
            "label_unresolved": n_unresolved,
            "label_ambiguous": n_ambiguous,
            "gt_consistent": n_gt,
            "negative": n_neg,
            "pair_ambiguous": n_pl_amb,
            "pair_unknown": n_unknown,
            "baseline_selected": n_sel,
            "baseline_non_selected": n_nonsel,
            "decision_relevant_selected": n_dec_rel,
            "decision_neutral_or_other": n_dec_neutral,
            "safe_removal_resolvable": n_safe_res,
            "not_safe_removal_resolvable": n - n_safe_res,
        },
        "identity_checks": {
            "n_resolved_eq_gt_plus_neg": n_resolved == n_gt + n_neg,
            "n_label_status_partition": n == n_resolved + n_unresolved + n_ambiguous,
        },
    }


def build_per_sequence_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_seq: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seq[str(r["sequence"])].append(r)
    out = []
    for seq in sorted(by_seq):
        g = by_seq[seq]
        n_neg = _count_where(g, lambda r: r.get("pair_label") == "negative")
        n_dr_neg = _count_where(
            g,
            lambda r: (
                r.get("pair_label") == "negative"
                and r.get("decision_relevance") == "selected"
            ),
        )
        n_safe = _count_where(
            g, lambda r: r.get("negative_status") == "negative_safe_removable"
        )
        out.append(
            {
                "sequence": seq,
                "n_total": len(g),
                "n_resolved": _count_where(
                    g, lambda r: r.get("label_status") == "resolved"
                ),
                "n_unresolved": _count_where(
                    g, lambda r: r.get("label_status") == "unresolved"
                ),
                "n_ambiguous": _count_where(
                    g, lambda r: r.get("label_status") == "ambiguous"
                ),
                "n_gt_consistent": _count_where(
                    g, lambda r: r.get("pair_label") == "gt_consistent"
                ),
                "n_negative": n_neg,
                "n_baseline_selected": _count_where(
                    g, lambda r: int(r.get("baseline_selected", 0)) == 1
                ),
                "n_decision_relevant_negative": n_dr_neg,
                "n_safe_removable_negative": n_safe,
            }
        )
    return out


def _quantiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {
            "min": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(vals, dtype=float)
    qs = np.quantile(arr, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(np.min(arr)),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "median": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "max": float(np.max(arr)),
    }


def build_signal_support_rows(
    rows: Sequence[Mapping[str, Any]],
    thr_vector: Sequence[float] = FROZEN_THR_VECTOR,
) -> list[dict[str, Any]]:
    """Support stats overall + per-sequence + label slice. No thr grid."""
    slices: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = [
        ("overall", "all", rows),
    ]
    for seq in sorted({str(r["sequence"]) for r in rows}):
        slices.append(("sequence", seq, [r for r in rows if str(r["sequence"]) == seq]))
    for lab in ("gt_consistent", "negative", "ambiguous", "unknown"):
        slices.append(
            (
                "pair_label",
                lab,
                [r for r in rows if r.get("pair_label") == lab],
            )
        )

    out: list[dict[str, Any]] = []
    for slice_kind, slice_value, subset in slices:
        for i, sig in enumerate(SIGNAL_NAMES):
            vals: list[float] = []
            missing = 0
            for r in subset:
                v = r.get(sig)
                if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
                    missing += 1
                else:
                    vals.append(float(v))
            thr = float(thr_vector[i])
            q = _quantiles(vals)
            n_above = sum(1 for v in vals if v > thr)
            out.append(
                {
                    "slice_kind": slice_kind,
                    "slice_value": slice_value,
                    "signal": sig,
                    "atom_index": i,
                    "count": len(vals),
                    "missing": missing,
                    **q,
                    "unique_count": len(set(vals)),
                    "frozen_threshold": thr,
                    "threshold_minus_observed_max": thr - q["max"]
                    if vals
                    else float("nan"),
                    "n_above_frozen_thr": n_above,
                    "support_overlap_with_frozen_policy": int(n_above > 0),
                }
            )
    return out


def build_safe_negative_mass_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n = len(rows)
    n_neg = _count_where(rows, lambda r: r.get("pair_label") == "negative")
    n_neg_sel = _count_where(
        rows,
        lambda r: (
            r.get("pair_label") == "negative"
            and r.get("decision_relevance") == "selected"
        ),
    )
    n_neg_ac = _count_where(
        rows,
        lambda r: (
            r.get("pair_label") == "negative"
            and r.get("decision_relevance") == "active_competitor"
        ),
    )
    n_neg_dr = n_neg_sel  # decision-relevant for reject-this-row = selected
    n_neg_dn = _count_where(
        rows,
        lambda r: r.get("negative_status") == "negative_decision_neutral",
    )
    n_safe = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_safe_removable"
    )
    n_neg_unres_eff = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_unresolved_effect"
    )
    n_neg_unsafe = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_unsafe_or_ambiguous"
    )
    # Note: under current taxonomy selected-negatives are safe_removable;
    # gt_consistent selected would be not_negative.

    per_seq = build_per_sequence_rows(rows)
    safe_by_seq = {r["sequence"]: int(r["n_safe_removable_negative"]) for r in per_seq}
    total_safe = sum(safe_by_seq.values())
    max_seq = max(safe_by_seq, key=safe_by_seq.get) if safe_by_seq else ""
    max_share = (
        float(safe_by_seq[max_seq]) / float(total_safe) if total_safe > 0 else 0.0
    )
    n_seqs_with_safe = sum(1 for v in safe_by_seq.values() if v > 0)
    single_seq_dominance = bool(total_safe > 0 and max_share > DOMINANCE_MAX_SHARE)

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "definition": {
            "negative_safe_removable": (
                "pair_label==negative AND decision_relevance==selected "
                "(inferred counterfactual: rejecting removes a wrong baseline "
                "reconnect). NOT observed intervention when rejected_by_hook==0."
            ),
            "decision_relevant_negative": (
                "pair_label==negative AND decision_relevance==selected"
            ),
            "decision_neutral_negative": (
                "pair_label==negative AND decision_relevance in "
                "{active_competitor, non_selected}"
            ),
            "observed_vs_inferred": "inferred_counterfactual unless rejected_by_hook==1",
        },
        "counts": {
            "N_total": n,
            "N_negative": n_neg,
            "N_negative_selected": n_neg_sel,
            "N_negative_active_competitor": n_neg_ac,
            "N_negative_decision_relevant": n_neg_dr,
            "N_negative_decision_neutral": n_neg_dn,
            "N_negative_safe_removable": n_safe,
            "N_negative_unresolved_effect": n_neg_unres_eff,
            "N_negative_unsafe_or_ambiguous": n_neg_unsafe,
        },
        "rates": {
            "negative_rate_over_D_online": float(n_neg) / n if n else 0.0,
            "decision_relevant_negative_rate": float(n_neg_dr) / n if n else 0.0,
            "safe_removable_negative_rate": float(n_safe) / n if n else 0.0,
            "safe_removable_rate_given_negative": float(n_safe) / n_neg
            if n_neg
            else 0.0,
        },
        "per_sequence_safe_removable": safe_by_seq,
        "n_sequences_with_safe_removable": n_seqs_with_safe,
        "max_seq_safe_removable": max_seq,
        "max_seq_share": max_share,
        "single_sequence_dominance": single_seq_dominance,
        "dominance_threshold": DOMINANCE_MAX_SHARE,
    }


# ---------------------------------------------------------------------------
# Reconciliation + claim firewall
# ---------------------------------------------------------------------------


def reconcile_stage2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Programmatic partition identities. Any failure → FAIL_CLOSED."""
    n_total = len(rows)
    n_resolved = _count_where(rows, lambda r: r.get("label_status") == "resolved")
    n_unresolved = _count_where(rows, lambda r: r.get("label_status") == "unresolved")
    n_ambiguous = _count_where(rows, lambda r: r.get("label_status") == "ambiguous")
    n_gt = _count_where(rows, lambda r: r.get("pair_label") == "gt_consistent")
    n_neg = _count_where(rows, lambda r: r.get("pair_label") == "negative")
    n_pl_amb = _count_where(rows, lambda r: r.get("pair_label") == "ambiguous")
    n_unknown = _count_where(rows, lambda r: r.get("pair_label") == "unknown")

    n_dn = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_decision_neutral"
    )
    # Current taxonomy: safe_removable IS the decision-relevant negative class;
    # negative_decision_relevant bucket unused but counted for partition.
    n_safe = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_safe_removable"
    )
    n_neg_unres = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_unresolved_effect"
    )
    n_neg_unsafe = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_unsafe_or_ambiguous"
    )

    # Negative partition among pair_label==negative
    n_neg_from_status = n_dn + n_safe + n_neg_unres + n_neg_unsafe
    # Also count negative_decision_relevant if ever used
    n_neg_dr_status = _count_where(
        rows, lambda r: r.get("negative_status") == "negative_decision_relevant"
    )
    n_neg_from_status += n_neg_dr_status

    per_seq = build_per_sequence_rows(rows)
    sum_total = sum(r["n_total"] for r in per_seq)
    sum_resolved = sum(r["n_resolved"] for r in per_seq)
    sum_neg = sum(r["n_negative"] for r in per_seq)
    sum_safe = sum(r["n_safe_removable_negative"] for r in per_seq)

    # Taxonomy MECE: every row has exactly one negative_status
    status_counts = Counter(str(r.get("negative_status")) for r in rows)
    unknown_status = [s for s in status_counts if s not in NEGATIVE_STATUS_VALUES]
    status_sum = sum(status_counts.values())

    checks = {
        "n_total_expected_244": n_total == EXPECTED_D_ONLINE_N
        or n_total != EXPECTED_D_ONLINE_N,
        # Keep explicit:
        "n_total_eq_244": n_total == EXPECTED_D_ONLINE_N,
        "label_status_partition": n_total == n_resolved + n_unresolved + n_ambiguous,
        "resolved_eq_gt_plus_neg": n_resolved == n_gt + n_neg,
        "pair_label_partition": n_total == n_gt + n_neg + n_pl_amb + n_unknown,
        "negative_status_mece": status_sum == n_total and not unknown_status,
        "negative_partition": n_neg == n_neg_from_status,
        "safe_le_negative": n_safe <= n_neg,
        "negative_le_total": n_neg <= n_total,
        "per_seq_sum_total": sum_total == n_total,
        "per_seq_sum_resolved": sum_resolved == n_resolved,
        "per_seq_sum_negative": sum_neg == n_neg,
        "per_seq_sum_safe": sum_safe == n_safe,
        "taxonomy_unknown_status_empty": len(unknown_status) == 0,
    }
    n_sel_neg = _count_where(
        rows,
        lambda r: (
            r.get("pair_label") == "negative"
            and r.get("decision_relevance") == "selected"
        ),
    )
    checks["safe_le_decision_relevant"] = n_safe <= n_sel_neg
    checks["safe_eq_selected_negative"] = n_safe == n_sel_neg

    errors = [k for k, ok in checks.items() if k != "n_total_expected_244" and not ok]
    # n_total_eq_244 is soft for fixtures; hard for authoritative run (caller)
    ok = all(
        v
        for k, v in checks.items()
        if k not in ("n_total_eq_244", "n_total_expected_244")
    )
    return {
        "ok": ok,
        "acceptance": "PASS" if ok else "FAIL_CLOSED",
        "checks": checks,
        "errors": errors,
        "counts": {
            "n_total": n_total,
            "n_resolved": n_resolved,
            "n_unresolved": n_unresolved,
            "n_ambiguous": n_ambiguous,
            "n_gt_consistent": n_gt,
            "n_negative": n_neg,
            "n_safe_removable": n_safe,
            "n_decision_neutral_negative": n_dn,
            "negative_status_counts": dict(status_counts),
        },
        "per_sequence_rows": per_seq,
    }


def n_neg_sel_safe(rows: Sequence[Mapping[str, Any]]) -> int:
    return _count_where(
        rows,
        lambda r: (
            r.get("pair_label") == "negative"
            and r.get("decision_relevance") == "selected"
        ),
    )


def apply_claim_firewall(
    *,
    join_summary: Mapping[str, Any],
    mass: Mapping[str, Any],
    recon: Mapping[str, Any],
    frozen_triggered: int = 0,
    min_join_coverage: float = 0.5,
) -> dict[str, Any]:
    """Programmatic claim limits. Does not invent effect-supported claims."""
    blocked: list[str] = []
    allowed: list[str] = []

    join_cov = float(join_summary.get("join_coverage", 0.0))
    n_joined = int(join_summary.get("n_joined", 0))
    n_total = int(join_summary.get("n_total", 0))
    counts = mass.get("counts", {})
    n_neg = int(counts.get("N_negative", 0))
    n_dr = int(counts.get("N_negative_decision_relevant", 0))
    n_safe = int(counts.get("N_negative_safe_removable", 0))
    dominance = bool(mass.get("single_sequence_dominance", False))
    recon_ok = bool(recon.get("ok", False))

    q1 = "PASSED"
    if not recon_ok:
        q1 = "FAILED"
        blocked.append("reconciliation_fail_closed")
    if n_total == 0:
        q1 = "FAILED"
        blocked.append("empty_d_online")
    if join_cov < min_join_coverage or n_joined == 0:
        q1 = "FAILED"
        blocked.append("label_join_coverage_insufficient")
        blocked.append("FP_mass_claim_inadmissible")

    q2 = "PASSED" if q1 == "PASSED" else "FAILED"

    # Q3 classification
    if q1 != "PASSED":
        q3 = "INADMISSIBLE"
        blocked.append("q3_inadmissible_due_to_q1")
    elif n_neg == 0:
        q3 = "CURRENT_PLACEMENT_TOO_LATE_CANDIDATE"
        blocked.append("resolved_negative_count_zero")
        blocked.append("safe_negative_research_unsupported")
    elif n_dr == 0 or n_safe == 0:
        if n_neg > 0 and n_dr == 0:
            q3 = "INSUFFICIENT_DECISION_RELEVANT_MASS"
            blocked.append("decision_relevant_negative_count_zero")
            blocked.append("reject_policy_study_at_current_placement_unsupported")
        else:
            q3 = "CURRENT_PLACEMENT_TOO_LATE_CANDIDATE"
            blocked.append("safe_removable_negative_count_zero")
            blocked.append("threshold_boolean_study_blocked")
    elif (
        n_safe >= Q3_SUFFICIENT_MIN_SAFE_REMOVABLE
        and int(mass.get("n_sequences_with_safe_removable", 0))
        >= Q3_SUFFICIENT_MIN_SEQS_WITH_SAFE
    ):
        q3 = "SUFFICIENT"
        allowed.append("enter_signal_separability_audit_on_D_online")
        allowed.append("fp_mass_measured_inferred_counterfactual")
    else:
        q3 = "INSUFFICIENT_DECISION_RELEVANT_MASS"
        blocked.append("safe_removable_mass_below_entry_threshold")

    if dominance:
        blocked.append("single_sequence_dominance_portability_claim_blocked")

    if int(frozen_triggered) == 0:
        blocked.append("frozen_policy_effect_claim_inadmissible")
        blocked.append("triggered_eq_0")

    # Never allow thr / effect claims this round
    blocked.append("threshold_or_boolean_claim_not_authorized_in_q1q3")
    blocked.append("production_promotion_blocked")

    next_step = {
        "SUFFICIENT": "signal_separability_audit_on_D_online (Q4); no thr yet",
        "INSUFFICIENT_DECISION_RELEVANT_MASS": (
            "reconsider decision form, audit depth, or hook placement; "
            "no threshold sweep"
        ),
        "CURRENT_PLACEMENT_TOO_LATE_CANDIDATE": (
            "earlier candidate domain or ranking/margin formulation; "
            "do not conclude signals invalid"
        ),
        "INADMISSIBLE": "fix join/recon blockers before any mass claim",
    }.get(q3, "unknown")

    return {
        "stage2_q1_label_join": q1,
        "stage2_q2_population_support": q2,
        "stage2_q3_safe_negative_mass": q3,
        "claims_blocked": blocked,
        "claims_allowed": allowed,
        "next_authorized_step": next_step,
        "production_preset": "unchanged",
        "frozen_triggered": int(frozen_triggered),
        "policy_effect_supported": False if frozen_triggered == 0 else None,
        "min_join_coverage_required": min_join_coverage,
        "join_coverage_observed": join_cov,
    }


# ---------------------------------------------------------------------------
# Full study runner
# ---------------------------------------------------------------------------


def run_stage2_q1q3_audit(
    *,
    stage1_study_dir: Path,
    out_dir: Path,
    gt_root: Path = Path("datasets/MOT17/train"),
    git_commit: str = "",
    study_id: str | None = None,
    expected_n: int | None = EXPECTED_D_ONLINE_N,
    min_join_coverage: float = 0.5,
    enforce_n_total: bool = True,
) -> dict[str, Any]:
    """Build full Q1–Q3 artifact pack from Stage 1 study directory."""
    stage1_study_dir = Path(stage1_study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_event_table_from_study(stage1_study_dir)
    source_table = stage1_study_dir / "hook_candidate_events.parquet"
    if not source_table.is_file():
        source_table = stage1_study_dir / "hook_candidate_events.csv"
    source_hash = _sha256_file(source_table) if source_table.is_file() else ""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = study_id or f"m_b1_5_stage2_q1q3_{stamp}"

    mot_dir = stage1_study_dir / "e2e_A1_hook_off"
    gid_path = mot_dir / "_global_id_map.txt"
    if not gid_path.is_file():
        raise Stage2AuditError(f"missing global_id_map: {gid_path}")

    # Frozen triggered from baudit / summary if present
    frozen_triggered = 0
    baudit_path = stage1_study_dir / "baudit_summary.json"
    if baudit_path.is_file():
        try:
            b = json.loads(baudit_path.read_text(encoding="utf-8"))
            frozen_triggered = int(b.get("n_rejected", b.get("n_triggered", 0)) or 0)
        except Exception:
            frozen_triggered = 0
    if frozen_triggered == 0:
        # All zero-fire expected
        frozen_triggered = sum(int(e.get("rejected_by_hook") or 0) for e in events)

    policy_hash = None
    if events:
        policy_hash = events[0].get("policy_file_hash")

    join = join_labels_to_d_online(
        events,
        mot_dir=mot_dir,
        gt_root=gt_root,
        global_id_map_path=gid_path,
        study_id=sid,
        source_event_table=str(source_table),
        policy_file_hash=str(policy_hash) if policy_hash else None,
    )
    rows = join.rows

    if enforce_n_total and expected_n is not None and len(rows) != expected_n:
        raise Stage2AuditError(f"D_online n_total={len(rows)} != expected {expected_n}")

    pop = build_population_summary(rows)
    per_seq = build_per_sequence_rows(rows)
    support = build_signal_support_rows(rows)
    mass = build_safe_negative_mass_summary(rows)
    recon = reconcile_stage2(rows)
    if enforce_n_total and expected_n is not None:
        if not recon["checks"].get("n_total_eq_244", True) and expected_n == 244:
            recon = dict(recon)
            recon["ok"] = False
            recon["acceptance"] = "FAIL_CLOSED"
            recon["errors"] = list(recon.get("errors", [])) + ["n_total_eq_244"]

    firewall = apply_claim_firewall(
        join_summary=join.summary,
        mass=mass,
        recon=recon,
        frozen_triggered=frozen_triggered,
        min_join_coverage=min_join_coverage,
    )

    # Write artifacts
    hashes: dict[str, str] = {}
    hashes["d_online_events_csv"] = write_csv(out_dir / "d_online_events.csv", rows)
    write_parquet(out_dir / "d_online_events.parquet", rows)
    if (out_dir / "d_online_events.parquet").is_file():
        hashes["d_online_events_parquet"] = _sha256_file(
            out_dir / "d_online_events.parquet"
        )

    hashes["label_join_summary"] = write_json(
        out_dir / "label_join_summary.json", join.summary
    )
    hashes["label_join_errors"] = write_csv(
        out_dir / "label_join_errors.csv", join.errors
    )
    hashes["d_online_population_summary"] = write_json(
        out_dir / "d_online_population_summary.json", pop
    )
    hashes["d_online_signal_support"] = write_csv(
        out_dir / "d_online_signal_support.csv", support
    )
    hashes["d_online_per_sequence"] = write_csv(
        out_dir / "d_online_per_sequence.csv", per_seq
    )
    hashes["safe_negative_mass_summary"] = write_json(
        out_dir / "safe_negative_mass_summary.json", mass
    )
    hashes["safe_negative_mass_per_sequence"] = write_csv(
        out_dir / "safe_negative_mass_per_sequence.csv",
        [
            {
                "sequence": r["sequence"],
                "n_total": r["n_total"],
                "n_negative": r["n_negative"],
                "n_decision_relevant_negative": r["n_decision_relevant_negative"],
                "n_safe_removable_negative": r["n_safe_removable_negative"],
                "share_of_safe_removable": (
                    float(r["n_safe_removable_negative"])
                    / float(mass["counts"]["N_negative_safe_removable"])
                    if mass["counts"]["N_negative_safe_removable"]
                    else 0.0
                ),
            }
            for r in per_seq
        ],
    )
    hashes["reconciliation"] = write_json(out_dir / "reconciliation.json", recon)

    funnel = pop["funnel"]
    summary = {
        "study_id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage1_study": str(stage1_study_dir),
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        **firewall,
        "D_online_total": funnel["D_online_total"],
        "label_resolved": funnel["label_resolved"],
        "label_unresolved": funnel["label_unresolved"],
        "label_ambiguous": funnel["label_ambiguous"],
        "gt_consistent": funnel["gt_consistent"],
        "negative": funnel["negative"],
        "decision_relevant_negative": mass["counts"]["N_negative_decision_relevant"],
        "safe_removable_negative": mass["counts"]["N_negative_safe_removable"],
        "single_sequence_dominance": mass["single_sequence_dominance"],
        "max_seq_share_safe_removable": mass["max_seq_share"],
        "max_seq_safe_removable": mass["max_seq_safe_removable"],
        "join_coverage": join.summary["join_coverage"],
        "reconciliation_acceptance": recon["acceptance"],
        "frozen_policy_online_relevance": "NULL_support_mismatch",
        "production_preset": "unchanged",
        "forbidden_this_round": [
            "threshold_grid",
            "boolean_rule_search",
            "new_atoms",
            "production_preset_change",
            "default_on",
            "offline_pairs_in_D_online",
            "effect_supported_claim",
        ],
    }
    hashes["summary"] = write_json(out_dir / "summary.json", summary)

    md = _render_summary_md(summary, mass, join.summary, recon)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    hashes["summary_md"] = _sha256_bytes(md.encode("utf-8"))

    manifest = {
        "schema": "m_b1_5_stage2_q1q3_manifest_v1",
        "study_id": sid,
        "git_commit": git_commit,
        "source_stage1_study": str(stage1_study_dir),
        "source_event_table": str(source_table),
        "source_event_table_hash": source_hash,
        "candidate_universe_id": CANDIDATE_UNIVERSE_ID,
        "substrate_id": SUBSTRATE_ID,
        "sequence_set": sorted({str(r["sequence"]) for r in rows}),
        "label_source": LABEL_SOURCE,
        "join_method": JOIN_METHOD,
        "taxonomy_version": TAXONOMY_VERSION,
        "mot_dir": str(mot_dir),
        "gt_root": str(gt_root),
        "global_id_map": str(gid_path),
        "artifact_hashes": hashes,
        "created_utc": summary["created_utc"],
    }
    write_json(out_dir / "manifest.json", manifest)
    summary["manifest"] = manifest
    return summary


def _render_summary_md(
    summary: Mapping[str, Any],
    mass: Mapping[str, Any],
    join: Mapping[str, Any],
    recon: Mapping[str, Any],
) -> str:
    lines = [
        f"# Stage 2 Q1–Q3 audit — `{summary.get('study_id')}`",
        "",
        "<!-- doc-status: research-artifact -->",
        "<!-- doc-promotion: none -->",
        "",
        "## Final classification",
        "",
        "```text",
        f"stage2_q1_label_join: {summary.get('stage2_q1_label_join')}",
        f"stage2_q2_population_support: {summary.get('stage2_q2_population_support')}",
        f"stage2_q3_safe_negative_mass: {summary.get('stage2_q3_safe_negative_mass')}",
        "",
        f"D_online_total: {summary.get('D_online_total')}",
        f"label_resolved: {summary.get('label_resolved')}",
        f"negative: {summary.get('negative')}",
        f"decision_relevant_negative: {summary.get('decision_relevant_negative')}",
        f"safe_removable_negative: {summary.get('safe_removable_negative')}",
        f"single_sequence_dominance: {'yes' if summary.get('single_sequence_dominance') else 'no'}",
        f"next_authorized_step: {summary.get('next_authorized_step')}",
        f"production_preset: {summary.get('production_preset')}",
        "```",
        "",
        "## Join",
        "",
        f"- method: `{join.get('join_method')}`",
        f"- coverage: **{join.get('join_coverage'):.3f}** "
        f"({join.get('n_joined')}/{join.get('n_total')} resolved)",
        f"- unresolved: {join.get('n_unresolved')} · ambiguous: {join.get('n_ambiguous')}",
        f"- reconciliation: **{recon.get('acceptance')}**",
        "",
        "## Safe-negative mass (inferred counterfactual)",
        "",
        f"- N_negative: {mass['counts']['N_negative']}",
        f"- N_negative_decision_relevant (selected): "
        f"{mass['counts']['N_negative_decision_relevant']}",
        f"- N_negative_safe_removable: {mass['counts']['N_negative_safe_removable']}",
        f"- N_negative_decision_neutral: {mass['counts']['N_negative_decision_neutral']}",
        f"- max seq share: {mass.get('max_seq_share', 0):.3f} "
        f"({mass.get('max_seq_safe_removable')})",
        "",
        "### Definition (not observed intervention)",
        "",
        "```text",
        mass["definition"]["negative_safe_removable"],
        "```",
        "",
        "## Claim firewall",
        "",
        "Blocked:",
        "",
    ]
    for b in summary.get("claims_blocked", []):
        lines.append(f"- `{b}`")
    lines += ["", "Allowed:", ""]
    for a in summary.get("claims_allowed", []) or ["(none beyond substrate facts)"]:
        lines.append(f"- `{a}`")
    lines += [
        "",
        "## Forbidden this round",
        "",
        "- threshold / Boolean search",
        "- frozen policy effect-supported claim",
        "- production preset change",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


def make_synthetic_d_online_row(
    *,
    event_id: str,
    sequence: str = "MOT17-02-SDP",
    frame: int = 1,
    cand_slot: int = 0,
    lost_slot: int = 1,
    cand_track_id: int = 10,
    lost_track_id: int = 20,
    score_m_bridge: float = 0.1,
    baseline_rank: int = 0,
    baseline_accepted: int = 1,
    pair_label: str = "negative",
    label_status: str = "resolved",
    decision_relevance: str = "selected",
    **extra: Any,
) -> dict[str, Any]:
    """Minimal synthetic row for unit tests (post-join shape)."""
    row = {
        "run_id": "test",
        "sequence": sequence,
        "frame": frame,
        "event_id": event_id,
        "runtime_candidate_id": f"{sequence}:c{cand_slot}:l{lost_slot}:f{frame}",
        "policy_candidate_id": "m_b1_repaired_eps0_loo_pass_20260709",
        "policy_file_hash": "test",
        "cand_slot": cand_slot,
        "lost_slot": lost_slot,
        "cand_track_id": cand_track_id,
        "lost_track_id": lost_track_id,
        "join_key": f"{sequence}|{frame}|{cand_track_id}|{lost_track_id}|{cand_slot}|{lost_slot}",
        "atom_bitmask": 0,
        "fired_atom_ids": "",
        "n_atoms_fired": 0,
        "fire_class": "zero",
        "rejected_by_hook": 0,
        "score_m_bridge": score_m_bridge,
        "abs_log_h": 0.01,
        "dist_h": 0.1,
        "abs_ratio_m1": 0.01,
        "resid_mean": 0.1,
        "thr_0": FROZEN_THR_VECTOR[0],
        "thr_1": FROZEN_THR_VECTOR[1],
        "thr_2": FROZEN_THR_VECTOR[2],
        "thr_3": FROZEN_THR_VECTOR[3],
        "thr_4": FROZEN_THR_VECTOR[4],
        "thr_margin_0": score_m_bridge - FROZEN_THR_VECTOR[0],
        "thr_margin_1": 0.01 - FROZEN_THR_VECTOR[1],
        "thr_margin_2": 0.1 - FROZEN_THR_VECTOR[2],
        "thr_margin_3": 0.01 - FROZEN_THR_VECTOR[3],
        "thr_margin_4": 0.1 - FROZEN_THR_VECTOR[4],
        "baseline_rank": baseline_rank,
        "baseline_accepted_candidate": baseline_accepted,
        "baseline_selected": baseline_accepted,
        "baseline_reconnect_decision": baseline_accepted,
        "competitor_count": 0,
        "pair_label": pair_label,
        "label_status": label_status,
        "decision_relevance": decision_relevance,
        "baseline_outcome": (
            "incorrect"
            if pair_label == "negative" and decision_relevance == "selected"
            else (
                "correct"
                if pair_label == "gt_consistent" and decision_relevance == "selected"
                else "neutral"
            )
        ),
        "gt_match": (
            1
            if pair_label == "gt_consistent"
            else (0 if pair_label == "negative" else None)
        ),
    }
    row.update(extra)
    row["negative_status"] = classify_negative_status(row)
    row["safe_removal_resolvable"] = int(
        row["negative_status"] == "negative_safe_removable"
    )
    return row
