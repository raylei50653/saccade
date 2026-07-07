"""Causal online Cheb-GR ID handover (real-time analogue of cheb_gr_merge).

:mod:`cheb_gr_merge` stitches tracklets *offline*: it sees the whole sequence,
re-ranks a global sample graph, and retroactively relabels merged tracklets.
This module is the causal counterpart, structured exactly like the deployment
mechanism so its numbers transfer:

  * A track's identity decision happens once, ``decide_n`` frames after birth
    (the online tracker's confirmation window, during which nothing has been
    emitted yet) — no retroactive rewriting of already-emitted frames.
  * The decision only uses information available at that moment: the newborn's
    clean detections from its first ``decide_n`` frames (the "head") versus an
    archive of recently-dead tracks and their full-life appearance banks. Dead
    banks are causal by construction (a candidate died before the newborn was
    born).
  * The k-reciprocal graph is the *event-local causal gallery* (head samples +
    gated archive banks), not the offline whole-sequence graph.

Evidence probe (2026-07-03): with the offline graph, ``decide_n = 5`` already
matches the full offline merge (IDF1 80.3 = ref); this module additionally
answers the causal-gallery question end-to-end.

Like cheb_gr_merge, the numeric core (:func:`causal_handover_lines`) takes
ready embeddings so it stays unit-testable; the evaluator adapter owns crop
extraction (:func:`extract_handover_embeddings`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..reid.cheb_gr import cheb_gr_kreciprocal
from .cheb_gr_merge import _extract_native_crops_trt, temporal_sample_indices
from .clean_fifo_bank import CleanFifoPlan, plan_clean_fifo_crops
from .helpers import front_occlusion_mask_xyxy
from .post_merge import _build_output_tracklets, _format_mot_records, _parse_mot_lines
from .utils import box_center, box_iou_tuple, shift_box

__all__ = [
    "causal_handover_lines",
    "extract_handover_embeddings",
]


@dataclass
class _HandoverStats:
    events: int = 0
    events_with_candidates: int = 0
    handovers: int = 0
    reject_no_head: int = 0
    reject_min_head: int = 0
    reject_cost: int = 0
    reject_margin: int = 0
    reject_key_sim: int = 0
    reject_key_margin: int = 0
    reject_center_dist: int = 0
    reject_pollution: int = 0
    requeries: int = 0
    decisions_logged: int = 0
    ids_before: int = 0
    ids_after: int = 0


def _key_bank_block_metrics(
    head: Tensor,
    bank_embs: dict[int, Tensor],
    *,
    candidate_ids: list[int],
    best_track_id: int,
    topk: int = 3,
) -> dict[str, int | float]:
    """Graph-external key-bank cosine diagnostics for one handover event.

    The Cheb-GR decision still uses k-reciprocal graph distance. These metrics
    expose the simpler sparse-memory question in the same log row: how strongly
    the newborn head matches the selected candidate bank, and by what margin
    against nearby candidate banks.
    """
    target = bank_embs.get(best_track_id)
    if target is None or head.numel() == 0 or target.numel() == 0:
        return {
            "key_best_sim": -1.0,
            "key_mean_topk_sim": -1.0,
            "key_best_other_id": -1,
            "key_best_other_sim": -1.0,
            "key_margin": -999.0,
            "key_support": 0,
            "key_other_support": 0,
        }

    block = head @ target.t()
    flat = block.reshape(-1)
    k = min(max(1, int(topk)), int(flat.numel()))
    key_best_sim = float(flat.max().item())
    key_mean_topk_sim = float(torch.topk(flat, k=k, largest=True).values.mean().item())

    best_other_id = -1
    best_other_sim = float("-inf")
    other_support = 0
    for tid in candidate_ids:
        if tid == best_track_id:
            continue
        other = bank_embs.get(tid)
        if other is None or other.numel() == 0:
            continue
        other_support += int(head.shape[0] * other.shape[0])
        other_sim = float((head @ other.t()).max().item())
        if other_sim > best_other_sim:
            best_other_sim = other_sim
            best_other_id = int(tid)

    if best_other_id >= 0:
        key_best_other_sim = best_other_sim
        key_margin = key_best_sim - best_other_sim
    else:
        key_best_other_sim = -1.0
        key_margin = 999.0

    return {
        "key_best_sim": key_best_sim,
        "key_mean_topk_sim": key_mean_topk_sim,
        "key_best_other_id": best_other_id,
        "key_best_other_sim": key_best_other_sim,
        "key_margin": key_margin,
        "key_support": int(head.shape[0] * target.shape[0]),
        "key_other_support": other_support,
    }


def causal_handover_lines(
    results_lines: list[str],
    head_embs: dict[int, Tensor],
    bank_embs: dict[int, Tensor],
    *,
    enabled: bool = False,
    max_cost: float = 0.45,
    max_gap: int = 60,
    decide_n: int = 5,
    min_head_samples: int = 1,
    margin: float = 0.0,
    key_sim_min: float = 0.0,
    key_sim_cost_floor: float = 0.0,
    key_margin_min: float = 0.0,
    center_dist_veto: float = 0.0,
    pollution_veto: float = 0.0,
    requery_bank_embs: dict[int, Tensor] | None = None,
    requery_band: float = 0.0,
    requery_top: int = 0,
    pool_frac: float = 0.3,
    cheb_lambda: float = 2.0,
    k2: int = 6,
    max_fwd: int = 50,
    fuse_lambda: float = 0.3,
    decision_log: list[dict[str, int | float | str | bool]] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Causally relabel newborn tracklets that hand over a dead identity.

    Args:
        results_lines: MOT ``frame,id,x,y,w,h,score,...`` lines.
        head_embs: ``{track_id: [H_i, D]}`` L2-normalized clean samples from the
            track's first ``decide_n`` frames (the evidence available at the
            confirmation-time decision). Tracks without an entry never hand over.
        bank_embs: ``{track_id: [S_i, D]}`` full-life bank used when the track
            later acts as a dead candidate. Causal: only read after death.
        max_cost: max Cheb-GR distance accepted for a handover (same scale and
            default operating point as the offline merge).
        max_gap: max frames between a candidate's death and the newborn's birth.
        decide_n: frames after birth at which the one-shot decision happens.
        min_head_samples: min clean newborn head samples required before a
            handover can be accepted.
        margin: minimum separation between the best and second-best candidate
            costs. A single valid candidate has infinite margin.
        key_sim_min: if > 0, reject when direct newborn-head vs selected-bank
            cosine support is below this threshold.
        key_sim_cost_floor: only apply ``key_sim_min`` when the Cheb-GR cost is
            at least this value. This lets very strong graph matches survive raw
            key-sim ambiguity.
        key_margin_min: if > 0, reject when direct key-bank hard-negative
            margin is below this threshold. A no-hard-negative event has
            sentinel margin 999 and passes this gate.
        center_dist_veto: reject a handover whose forward-projected candidate
            center is at least this many average box heights from the newborn
            start center (0 = off). Cross-run applicability evidence:
            ``center_dist_norm >= 2`` is a stable veto.
        pollution_veto: reject a handover whose newborn head / candidate tail
            crops overlap another track above this IoU (0 = off). Cross-run
            applicability evidence: ``head_tail_neighbor_iou >= 0.5`` is a
            stable high-pollution zone.
        requery_bank_embs: optional denser fallback banks. When a decision
            lands in the borderline band (``|best_cost - max_cost| <=
            requery_band`` or the observed margin is within ``requery_band``
            of the required margin), the event graph is rebuilt with these
            banks and rescored before gating — removing sparse-bank phase
            jitter exactly where flips live. Online analogue: keep a raw-crop
            ring buffer per track tail and re-extract on demand for the
            handful of borderline events.
        requery_band: borderline half-width that triggers the re-query
            (0 = off).
        requery_top: if > 0, only the top-``requery_top`` sparse-ranked
            candidates get their dense fallback bank in the re-query (the
            rest keep sparse banks as graph context) — the cheap online form:
            ~2 candidate tails re-extracted per borderline event instead of
            the whole archive. 0 = densify every in-graph bank.
        decision_log: optional mutable list that receives one row per scored
            handover decision.

    Returns:
        (rewritten lines, stats dict).
    """
    stats = _HandoverStats()
    if not enabled or not results_lines:
        return results_lines, vars(stats)

    records = _parse_mot_lines(results_lines)
    tracklets = _build_output_tracklets(records, velocity_samples=5)
    stats.ids_before = len(tracklets)
    if len(tracklets) <= 1:
        stats.ids_after = len(tracklets)
        return results_lines, vars(stats)
    boxes_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = (
        defaultdict(list)
    )
    for r in records:
        boxes_by_frame[r.frame].append((r.track_id, (r.x, r.y, r.x + r.w, r.y + r.h)))

    def max_neighbor_iou(
        track_id: int,
        frame: int,
        box: tuple[float, float, float, float],
    ) -> float:
        best = 0.0
        for other_id, other_box in boxes_by_frame.get(frame, []):
            if other_id == track_id:
                continue
            best = max(best, float(box_iou_tuple(box, other_box)))
        return best

    def max_record_neighbor_iou(rs: list[Any]) -> float:
        best = 0.0
        for r in rs:
            box = (r.x, r.y, r.x + r.w, r.y + r.h)
            best = max(best, max_neighbor_iou(r.track_id, r.frame, box))
        return best

    # label: output identity per original track id (follows handover chains).
    label = {t.track_id: t.track_id for t in tracklets}

    # Timeline: deaths feed the archive, decisions consume it. A decision at
    # t_d = start + decide_n may only see candidates that died at or before the
    # newborn's birth (gap >= 1 keeps chain frame-sets strictly disjoint, which
    # is also what the offline merge's disjointness check reduces to).
    deaths = sorted(tracklets, key=lambda t: (t.end, t.start, t.track_id))
    decisions = sorted(
        tracklets, key=lambda t: (t.start + decide_n, t.start, t.track_id)
    )

    archive: dict[int, Any] = {}  # track_id -> OutputTracklet (dead, unconsumed)
    di = 0
    for tb in decisions:
        t_decide = tb.start + decide_n
        while di < len(deaths) and deaths[di].end < t_decide:
            archive[deaths[di].track_id] = deaths[di]
            di += 1

        # Real-time archive semantics: identities dead for more than max_gap
        # frames can never be claimed again — drop them (bounded memory).
        expired = [tid for tid, ta in archive.items() if ta.end < tb.start - max_gap]
        for tid in expired:
            del archive[tid]

        stats.events += 1
        head = head_embs.get(tb.track_id)
        if head is None or head.shape[0] == 0:
            stats.reject_no_head += 1
            continue
        if head.shape[0] < min_head_samples:
            stats.reject_min_head += 1
            continue

        in_graph = [
            ta
            for ta in archive.values()
            if bank_embs.get(ta.track_id) is not None
            and bank_embs[ta.track_id].shape[0] > 0
        ]
        cands = [ta for ta in in_graph if 1 <= tb.start - ta.end <= max_gap]
        if not cands:
            continue
        stats.events_with_candidates += 1

        # Event-local causal graph: newborn head + every in-window archive bank.
        # Non-candidate banks (e.g. died inside the head window) join as context
        # only — a larger graph keeps the re-ranked distance scale closer to the
        # offline whole-sequence graph the max_cost operating point came from.
        def _score_event(bmap: dict[int, Tensor]) -> list[tuple[float, Any]]:
            feats_list = [head]
            span_by_tid: dict[int, tuple[int, int]] = {}
            pos = head.shape[0]
            for ta in in_graph:
                bank = bmap.get(ta.track_id)
                if bank is None or bank.shape[0] == 0:
                    bank = bank_embs[ta.track_id]
                feats_list.append(bank)
                span_by_tid[ta.track_id] = (pos, pos + bank.shape[0])
                pos += bank.shape[0]
            feats = torch.cat(feats_list, dim=0)
            n = feats.shape[0]
            sdist = cheb_gr_kreciprocal(
                feats,
                feats,
                cheb_lambda=cheb_lambda,
                k2=min(k2, n),
                max_fwd=max_fwd,
                fuse_lambda=fuse_lambda,
            )
            out: list[tuple[float, Any]] = []
            head_rows = sdist[: head.shape[0]]
            for ta in cands:
                lo, hi = span_by_tid[ta.track_id]
                block = head_rows[:, lo:hi].reshape(-1)
                k = max(1, int(round(pool_frac * block.numel())))
                cost = float(torch.topk(block, k, largest=False).values.mean())
                out.append((cost, ta))
            out.sort(key=lambda item: (item[0], item[1].track_id))
            return out

        scored = _score_event(bank_embs)
        if not scored:
            continue
        best_cost, best = scored[0]
        second_cost = scored[1][0] if len(scored) > 1 else float("inf")
        observed_margin = second_cost - best_cost

        # A decision is flippable only if a band-sized perturbation could change
        # a gate outcome: the cost gate near max_cost, or the margin gate — the
        # latter only matters when the cost gate itself is (nearly) passing.
        cost_borderline = abs(best_cost - max_cost) <= requery_band
        margin_borderline = (
            margin > 0.0
            and best_cost <= max_cost + requery_band
            and abs(observed_margin - margin) <= requery_band
        )
        requeried = False
        if (
            requery_bank_embs is not None
            and requery_band > 0.0
            and (cost_borderline or margin_borderline)
        ):
            requery_map = requery_bank_embs
            if requery_top > 0:
                top_ids = {ta.track_id for _, ta in scored[:requery_top]}
                requery_map = {
                    tid: bank
                    for tid, bank in requery_bank_embs.items()
                    if tid in top_ids
                }
            scored = _score_event(requery_map)
            best_cost, best = scored[0]
            second_cost = scored[1][0] if len(scored) > 1 else float("inf")
            observed_margin = second_cost - best_cost
            requeried = True
            stats.requeries += 1

        gap = int(tb.start - best.end)
        direct_iou = box_iou_tuple(best.end_box, tb.start_box)
        candidate_forward_box = shift_box(best.end_box, best.end_velocity, gap)
        newborn_backward_box = shift_box(tb.start_box, tb.start_velocity, -gap)
        candidate_forward_iou = box_iou_tuple(candidate_forward_box, tb.start_box)
        newborn_backward_iou = box_iou_tuple(newborn_backward_box, best.end_box)
        c0 = box_center(candidate_forward_box)
        c1 = box_center(tb.start_box)
        avg_h = max(
            ((best.end_box[3] - best.end_box[1]) + (tb.start_box[3] - tb.start_box[1]))
            * 0.5,
            1.0,
        )
        center_dist_norm = float(
            ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5 / avg_h
        )
        newborn_head_records = [
            r for r in tb.records if r.frame < tb.start + max(1, decide_n)
        ]
        candidate_tail_records = [
            r for r in best.records if r.frame > best.end - max(1, decide_n)
        ]
        newborn_head_neighbor_iou = max_record_neighbor_iou(newborn_head_records)
        candidate_tail_neighbor_iou = max_record_neighbor_iou(candidate_tail_records)
        head_tail_neighbor_iou = max(
            newborn_head_neighbor_iou, candidate_tail_neighbor_iou
        )
        key_metrics = _key_bank_block_metrics(
            head,
            bank_embs,
            candidate_ids=[ta.track_id for _, ta in scored],
            best_track_id=best.track_id,
        )

        reason = "accepted"
        accepted = True
        if best_cost > max_cost:
            reason = "cost"
            accepted = False
            stats.reject_cost += 1
        elif margin > 0.0 and observed_margin < margin:
            reason = "margin"
            accepted = False
            stats.reject_margin += 1
        elif (
            key_sim_min > 0.0
            and best_cost >= key_sim_cost_floor
            and float(key_metrics["key_best_sim"]) < key_sim_min
        ):
            reason = "key_sim"
            accepted = False
            stats.reject_key_sim += 1
        elif key_margin_min > 0.0 and float(key_metrics["key_margin"]) < key_margin_min:
            reason = "key_margin"
            accepted = False
            stats.reject_key_margin += 1
        elif center_dist_veto > 0.0 and center_dist_norm >= center_dist_veto:
            reason = "center_dist"
            accepted = False
            stats.reject_center_dist += 1
        elif pollution_veto > 0.0 and head_tail_neighbor_iou >= pollution_veto:
            reason = "pollution"
            accepted = False
            stats.reject_pollution += 1

        if decision_log is not None:
            row_second_cost = second_cost if np.isfinite(second_cost) else -1.0
            row_margin = observed_margin if np.isfinite(observed_margin) else 999.0
            newborn_neighbor_iou = max_neighbor_iou(
                tb.track_id,
                tb.records[0].frame,
                tb.start_box,
            )
            candidate_neighbor_iou = max_neighbor_iou(
                best.track_id,
                best.records[-1].frame,
                best.end_box,
            )
            decision_log.append(
                {
                    "newborn_id": int(tb.track_id),
                    "newborn_start": int(tb.start),
                    "newborn_end": int(tb.end),
                    "newborn_mean_score": float(tb.mean_score),
                    "newborn_start_score": float(tb.records[0].score),
                    "candidate_id": int(best.track_id),
                    "candidate_label": int(label[best.track_id]),
                    "candidate_start": int(best.start),
                    "candidate_end": int(best.end),
                    "candidate_mean_score": float(best.mean_score),
                    "candidate_end_score": float(best.records[-1].score),
                    "gap": gap,
                    "direct_iou": float(direct_iou),
                    "candidate_forward_iou": float(candidate_forward_iou),
                    "newborn_backward_iou": float(newborn_backward_iou),
                    "match_iou": float(candidate_forward_iou),
                    "newborn_neighbor_iou": float(newborn_neighbor_iou),
                    "candidate_neighbor_iou": float(candidate_neighbor_iou),
                    "neighbor_iou": float(
                        max(newborn_neighbor_iou, candidate_neighbor_iou)
                    ),
                    "newborn_head_neighbor_iou": float(newborn_head_neighbor_iou),
                    "candidate_tail_neighbor_iou": float(candidate_tail_neighbor_iou),
                    "head_tail_neighbor_iou": float(head_tail_neighbor_iou),
                    "center_dist_norm": center_dist_norm,
                    "head_n": int(head.shape[0]),
                    "bank_n": int(bank_embs[best.track_id].shape[0]),
                    "requeried": bool(requeried),
                    "candidate_count": int(len(scored)),
                    "best_cost": float(best_cost),
                    "second_cost": float(row_second_cost),
                    "margin": float(row_margin),
                    "required_margin": float(margin),
                    "max_cost": float(max_cost),
                    "key_sim_min": float(key_sim_min),
                    "key_sim_cost_floor": float(key_sim_cost_floor),
                    "key_margin_min": float(key_margin_min),
                    **key_metrics,
                    "accepted": bool(accepted),
                    "reason": reason,
                }
            )
            stats.decisions_logged += 1

        if not accepted:
            continue

        label[tb.track_id] = label[best.track_id]
        del archive[best.track_id]  # an identity is revived at most once
        stats.handovers += 1

    if stats.handovers == 0:
        stats.ids_after = stats.ids_before
        return results_lines, vars(stats)

    for r in records:
        r.track_id = label[r.track_id]
    stats.ids_after = len(set(label.values()))
    return _format_mot_records(records), vars(stats)


_POOL = tuple[int, int, tuple[float, float, float, float]]


def _build_pool_from_plan(
    plan: CleanFifoPlan,
) -> tuple[list[_POOL], dict[int, list[int]], dict[int, list[int]]]:
    """Build shared dedup crop pool from a ``CleanFifoPlan``."""
    pool_idx: dict[tuple[int, int, float, float, float, float], int] = {}
    pool: list[_POOL] = []
    head_rows: dict[int, list[int]] = {}
    bank_rows: dict[int, list[int]] = {}

    for tid in plan.head_crops:
        hr: list[int] = []
        for frame, (x1, y1, x2, y2) in plan.head_crops[tid]:
            w, h = x2 - x1, y2 - y1
            key = (tid, frame, x1, y1, w, h)
            if key not in pool_idx:
                pool_idx[key] = len(pool)
                pool.append((tid, frame, (x1, y1, x2, y2)))
            hr.append(pool_idx[key])
        head_rows[tid] = hr

        br: list[int] = []
        for frame, (x1, y1, x2, y2) in plan.bank_crops[tid]:
            w, h = x2 - x1, y2 - y1
            key = (tid, frame, x1, y1, w, h)
            if key not in pool_idx:
                pool_idx[key] = len(pool)
                pool.append((tid, frame, (x1, y1, x2, y2)))
            br.append(pool_idx[key])
        bank_rows[tid] = br

    return pool, head_rows, bank_rows


def _build_pool_spread(
    records: list[Any],
    *,
    decide_n: int,
    bank_budget: int,
    appearance_occlusion_cov: float,
    neighbor_iou_max: float,
) -> tuple[list[_POOL], dict[int, list[int]], dict[int, list[int]]]:
    """Legacy spread-mode pool (temporal distribution over whole track life)."""
    tracklets = _build_output_tracklets(records, velocity_samples=5)
    start_by_id = {t.track_id: t.start for t in tracklets}

    by_frame_idx: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_frame_idx[r.frame].append(ri)
    dirty: set[int] = set()
    for idxs in by_frame_idx.values():
        boxes = torch.tensor(
            [
                (
                    records[i].x,
                    records[i].y,
                    records[i].x + records[i].w,
                    records[i].y + records[i].h,
                )
                for i in idxs
            ],
            dtype=torch.float32,
        )
        mask = front_occlusion_mask_xyxy(boxes, appearance_occlusion_cov)
        dirty.update(i for i, d in zip(idxs, mask.tolist()) if d)

    clean_by_id: dict[int, list[Any]] = defaultdict(list)
    for ri, r in enumerate(records):
        if ri not in dirty:
            clean_by_id[r.track_id].append(r)

    def _rkey(r: Any) -> tuple[int, int, float, float, float, float]:
        return (r.track_id, r.frame, r.x, r.y, r.w, r.h)

    polluted: set[tuple[int, int, float, float, float, float]] = set()
    if neighbor_iou_max > 0.0:
        for idxs in by_frame_idx.values():
            box_tuples = [
                (
                    records[i].x,
                    records[i].y,
                    records[i].x + records[i].w,
                    records[i].y + records[i].h,
                )
                for i in idxs
            ]
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if records[idxs[a]].track_id == records[idxs[b]].track_id:
                        continue
                    if box_iou_tuple(box_tuples[a], box_tuples[b]) >= neighbor_iou_max:
                        polluted.add(_rkey(records[idxs[a]]))
                        polluted.add(_rkey(records[idxs[b]]))

    pool_idx: dict[tuple[int, int, float, float, float, float], int] = {}
    pool: list[_POOL] = []

    def add(rs: list[Any]) -> list[int]:
        out = []
        for r in rs:
            key = (r.track_id, r.frame, r.x, r.y, r.w, r.h)
            if key not in pool_idx:
                pool_idx[key] = len(pool)
                pool.append((r.track_id, r.frame, (r.x, r.y, r.x + r.w, r.y + r.h)))
            out.append(pool_idx[key])
        return out

    head_rows: dict[int, list[int]] = {}
    bank_rows: dict[int, list[int]] = {}
    for tid, items in clean_by_id.items():
        items.sort(key=lambda r: r.frame)
        birth = start_by_id[tid]
        unpolluted = [r for r in items if _rkey(r) not in polluted]
        head_rows[tid] = add([r for r in unpolluted if r.frame < birth + decide_n])
        bank_pool = unpolluted if unpolluted else items
        scores = np.asarray([r.score for r in bank_pool], dtype=np.float32)
        sel = temporal_sample_indices(len(bank_pool), bank_budget, scores=scores)
        bank_rows[tid] = add([bank_pool[j] for j in sel])

    return pool, head_rows, bank_rows


def _cache_key(
    frame: int, box: tuple[float, float, float, float]
) -> tuple[int, int, int, int, int]:
    """(frame, x, y, w, h) with coordinates quantized through the exact ``.2f``
    formatting MOT lines are serialized with, so a raw live-side float and its
    parsed round-trip value land on the same key by construction. (Any other
    quantization disagrees at rounding boundaries: a 0.1px grid flips ~2.5% of
    coordinates, and ``round(x*100)`` still flips at .005 ties where printf's
    decimal rounding and the fp multiply resolve differently.)"""

    def _q(v: float) -> int:
        return int(round(float(f"{v:.2f}") * 100))

    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (int(frame), _q(x1), _q(y1), _q(x2 - x1), _q(y2 - y1))


def extract_handover_embeddings(
    results_lines: list[str],
    seq_dir: str,
    extractor: Any,
    *,
    decide_n: int = 5,
    n_samples: int = 50,
    crop_hw: tuple[int, int] | None = None,
    im_ext: str = ".jpg",
    batch: int = 256,
    appearance_occlusion_cov: float = 0.4,
    neighbor_iou_max: float = 0.0,
    bank_mode: str = "spread",
    bank_n: int = 0,
    embedding_cache: dict[tuple[int, int, int, int, int], Tensor] | None = None,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """Head + bank embeddings for the causal handover, one extraction pass.

    Mirrors the offline path's visclean contract (full-frame-context front
    occlusion gate on every sample). ``head`` = all clean detections in the
    track's first ``decide_n`` frames; ``bank`` = per ``bank_mode`` either
    temporally-distributed samples over the whole life (``"spread"``) or the
    most recent clean samples before death (``"recent"``, a visclean-gated
    FIFO — probe 2026-07-04: recent-20 beats spread on both m/s substrates
    with the highest accept-set fidelity). ``bank_n`` caps the bank budget
    (0 = ``n_samples``). Both come from the same deduplicated crop pool,
    extracted through the native C++/CUDA + TRT path (PIL fallback).

    ``embedding_cache``, when provided, is a live (frame,box) → embedding dict
    populated during tracking; no disk I/O happens. Planned crops that miss the
    cache (coasting/interpolated boxes where no detection fired → no reid ran)
    are dropped from the head/bank rows — never emitted as zero vectors, which
    are mutual nearest neighbours in the k-reciprocal graph. The planner
    (spread/recent) is identical to the offline variant, so the output is
    bit-identical when the cache has full coverage, and simply sparser
    otherwise (the min_head / margin gates absorb the sparseness).
    """
    from PIL import Image

    if crop_hw is None:
        crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))

    records = _parse_mot_lines(results_lines)

    if bank_mode not in ("spread", "recent"):
        raise ValueError(f"unknown bank_mode {bank_mode!r}")
    bank_budget = bank_n if bank_n > 0 else n_samples

    if bank_mode == "recent":
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=appearance_occlusion_cov,
            fifo_n=bank_budget,
            stride=1,
            decide_n=decide_n,
            neighbor_iou_max=neighbor_iou_max,
        )
        pool, head_rows, bank_rows = _build_pool_from_plan(plan)
    else:
        pool, head_rows, bank_rows = _build_pool_spread(
            records,
            decide_n=decide_n,
            bank_budget=bank_budget,
            appearance_occlusion_cov=appearance_occlusion_cov,
            neighbor_iou_max=neighbor_iou_max,
        )

    if not pool:
        return {}, {}
    n_pool = len(pool)
    feats: Tensor | None

    # When a live embedding cache is provided, use it directly (no disk I/O).
    # Cached crops skip extraction; uncached pool entries stay zero and are
    # dropped per-track in gather() below.
    if embedding_cache is not None:
        feats = torch.zeros((n_pool, extractor.feature_dim), device="cpu")
        for si, (_, fr, box_xyxy) in enumerate(pool):
            emb = embedding_cache.get(_cache_key(fr, box_xyxy))
            if emb is not None:
                feats[si] = emb.detach().float()
        feats = feats.to(getattr(extractor, "device", "cuda"), copy=True)
        nonzero = feats.norm(dim=1) > 1e-8
        if nonzero.any():
            feats[nonzero] = torch.nn.functional.normalize(feats[nonzero], dim=1)
    else:
        # No live cache — full disk extraction (original offline path).
        by_frame_s: dict[int, list[int]] = defaultdict(list)
        for si, (_, fr, _) in enumerate(pool):
            by_frame_s[fr].append(si)

        feats = _extract_native_crops_trt(
            pool,
            by_frame_s,
            seq_dir,
            extractor,
            crop_hw=crop_hw,
            im_ext=im_ext,
            batch=batch,
        )
        if feats is None:
            out_h, out_w = crop_hw
            arrs: list[np.ndarray | None] = [None] * n_pool
            for fr, si_list in by_frame_s.items():
                img = Image.open(f"{seq_dir}/{fr:06d}{im_ext}").convert("RGB")
                fw, fh = img.size
                for si in si_list:
                    x1, y1, x2, y2 = pool[si][2]
                    box = (
                        max(0, int(round(x1))),
                        max(0, int(round(y1))),
                        min(fw, int(round(x2))),
                        min(fh, int(round(y2))),
                    )
                    if box[2] <= box[0] or box[3] <= box[1]:
                        box = (0, 0, fw, fh)
                    crop = img.crop(box).resize(
                        (out_w, out_h),
                        Image.BILINEAR,  # type: ignore[attr-defined]
                    )
                    arrs[si] = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)
            device = getattr(extractor, "device", "cuda")
            feats = torch.empty((n_pool, extractor.feature_dim), device=device)
            for s in range(0, n_pool, max(1, batch)):
                chunk = [a for a in arrs[s : s + batch] if a is not None]
                t = torch.from_numpy(np.stack(chunk)).to(device).float().div_(255.0)
                feats[s : s + t.shape[0]] = extractor.extract(t)
            feats = torch.nn.functional.normalize(feats, dim=1)

    assert feats is not None

    def gather(rows: dict[int, list[int]]) -> dict[int, Tensor]:
        # Zero rows (cache misses) must not reach the decision core: zero
        # vectors are mutual nearest neighbours in the k-reciprocal graph.
        out: dict[int, Tensor] = {}
        for tid, idxs in rows.items():
            if idxs:
                t = feats[torch.tensor(idxs, device=feats.device)]
                keep = t.norm(dim=1) > 1e-8
                if keep.any():
                    out[tid] = t[keep]
        return out

    return gather(head_rows), gather(bank_rows)
