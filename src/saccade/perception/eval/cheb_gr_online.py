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
from .helpers import front_occlusion_mask_xyxy
from .post_merge import _build_output_tracklets, _format_mot_records, _parse_mot_lines

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
    decisions_logged: int = 0
    ids_before: int = 0
    ids_after: int = 0


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
        feats_list = [head]
        span_by_tid: dict[int, tuple[int, int]] = {}
        pos = head.shape[0]
        for ta in in_graph:
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

        scored: list[tuple[float, Any]] = []
        head_rows = sdist[: head.shape[0]]
        for ta in cands:
            lo, hi = span_by_tid[ta.track_id]
            block = head_rows[:, lo:hi].reshape(-1)
            k = max(1, int(round(pool_frac * block.numel())))
            cost = float(torch.topk(block, k, largest=False).values.mean())
            scored.append((cost, ta))
        if not scored:
            continue
        scored.sort(key=lambda item: (item[0], item[1].track_id))
        best_cost, best = scored[0]
        second_cost = scored[1][0] if len(scored) > 1 else float("inf")
        observed_margin = second_cost - best_cost
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

        if decision_log is not None:
            row_second_cost = second_cost if np.isfinite(second_cost) else -1.0
            row_margin = observed_margin if np.isfinite(observed_margin) else 999.0
            decision_log.append(
                {
                    "newborn_id": int(tb.track_id),
                    "newborn_start": int(tb.start),
                    "newborn_end": int(tb.end),
                    "candidate_id": int(best.track_id),
                    "candidate_label": int(label[best.track_id]),
                    "candidate_start": int(best.start),
                    "candidate_end": int(best.end),
                    "gap": int(tb.start - best.end),
                    "head_n": int(head.shape[0]),
                    "bank_n": int(bank_embs[best.track_id].shape[0]),
                    "candidate_count": int(len(scored)),
                    "best_cost": float(best_cost),
                    "second_cost": float(row_second_cost),
                    "margin": float(row_margin),
                    "required_margin": float(margin),
                    "max_cost": float(max_cost),
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
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """Head + bank embeddings for the causal handover, one extraction pass.

    Mirrors the offline path's visclean contract (full-frame-context front
    occlusion gate on every sample). ``head`` = all clean detections in the
    track's first ``decide_n`` frames; ``bank`` = temporally-distributed
    ``n_samples`` over the whole life. Both come from the same deduplicated
    crop pool, extracted through the native C++/CUDA + TRT path (PIL fallback).
    """
    from PIL import Image

    if crop_hw is None:
        crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))  # type: ignore[arg-type]

    records = _parse_mot_lines(results_lines)
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

    pool_idx: dict[tuple, int] = {}
    pool: list[tuple[int, int, tuple[float, float, float, float]]] = []

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
        head_rows[tid] = add([r for r in items if r.frame < birth + decide_n])
        scores = np.asarray([r.score for r in items], dtype=np.float32)
        sel = temporal_sample_indices(len(items), n_samples, scores=scores)
        bank_rows[tid] = add([items[j] for j in sel])

    if not pool:
        return {}, {}

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
        # PIL fallback (same crop contract as the offline merge's fallback).
        out_h, out_w = crop_hw
        arrs: list[np.ndarray | None] = [None] * len(pool)
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
                crop = img.crop(box).resize((out_w, out_h), Image.BILINEAR)  # type: ignore[attr-defined]
                arrs[si] = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)
        device = getattr(extractor, "device", "cuda")
        feats = torch.empty((len(pool), extractor.feature_dim), device=device)
        for s in range(0, len(pool), max(1, batch)):
            chunk = [a for a in arrs[s : s + batch] if a is not None]
            t = torch.from_numpy(np.stack(chunk)).to(device).float().div_(255.0)
            feats[s : s + t.shape[0]] = extractor.extract(t)
        feats = torch.nn.functional.normalize(feats, dim=1)

    def gather(rows: dict[int, list[int]]) -> dict[int, Tensor]:
        out: dict[int, Tensor] = {}
        for tid, idxs in rows.items():
            if idxs:
                out[tid] = feats[torch.tensor(idxs, device=feats.device)]
        return out

    return gather(head_rows), gather(bank_rows)
