"""Causal occ-exit identity audit (ABSORB-side twin of the online handover).

:mod:`cheb_gr_online` claims a dead identity for a *newborn* track at its
confirmation frame (REBORN side). This module audits the complementary error:
an *existing* track that absorbs another person's box during an occlusion and
keeps tracking the wrong identity afterwards (ABSORB side).

Probe evidence (2026-07-03, ``probe_assoc_appearance_veto.py``): at the swap
frame the claimed crop is usually occluded, so an instant appearance veto
cannot fire — but within a few frames of occlusion exit the crop is clean
again, and a min-of-3 cosine against the track's pre-occlusion reference flags
14–23 of 26 auditable identity transfers at a 0.1–0.5% clean-stream false rate
(median delay 5–6 frames).

Causal contract (mirrors the online handover):

  * Occlusion episodes are detected with the same geometric front-occlusion
    coverage rule used by the visclean training filter and the handover's
    clean-sample gate (:func:`front_occlusion_mask_xyxy`) — no GT visibility.
  * The reference is the track's last clean detections *before* the episode
    (what an occ-freeze appearance bank would hold). Dirty frames never enter
    the reference; post-exit frames only enter after the audit passes.
  * The decision fires at the first post-exit clean frame whose cosine drops
    below ``tau``. Only frames from the decision frame onward are relabelled
    to a fresh id — nothing already emitted is rewritten.

The numeric core (:func:`occ_exit_audit_lines`) takes ready embeddings so it
stays unit-testable; the evaluator adapter owns crop extraction
(:func:`extract_audit_embeddings`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .cheb_gr_merge import _extract_native_crops_trt
from .helpers import front_occlusion_mask_xyxy
from .post_merge import _format_mot_records, _parse_mot_lines

__all__ = [
    "extract_audit_embeddings",
    "occ_exit_audit_lines",
    "plan_occ_audit_episodes",
]


@dataclass
class _Episode:
    track_id: int
    occ_start: int  # first dirty frame of the episode
    occ_end: int  # last dirty frame of the episode
    ref_frames: list[int] = field(default_factory=list)  # clean, before episode
    audit_frames: list[int] = field(default_factory=list)  # clean, after episode
    occluder_id: int = -1  # most frequent fronting coverer during the episode
    occluder_ref_frames: list[int] = field(default_factory=list)


@dataclass
class _AuditStats:
    episodes: int = 0
    audited: int = 0
    flags: int = 0
    abstain_no_ref: int = 0
    abstain_no_crops: int = 0
    abstain_no_occref: int = 0
    decisions_logged: int = 0
    ids_before: int = 0
    ids_after: int = 0


def _clean_flags_by_track(
    records: list[Any], appearance_occlusion_cov: float
) -> tuple[dict[int, list[tuple[int, bool]]], dict[tuple[int, int], int]]:
    """Per track: sorted ``(frame, clean)`` using the visclean coverage rule,
    plus ``{(track_id, frame): coverer_track_id}`` for every dirty record."""
    by_frame_idx: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_frame_idx[r.frame].append(ri)
    dirty: set[int] = set()
    coverer: dict[tuple[int, int], int] = {}
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
        dirty_local = [i for i, d in zip(range(len(idxs)), mask.tolist()) if d]
        if dirty_local:
            # coverage of each box by each fronting box (mirrors the mask rule)
            x1, y1 = boxes[:, 0], boxes[:, 1]
            x2, y2 = boxes[:, 2], boxes[:, 3]
            area = ((x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)).clamp(min=1e-6)
            iw = (
                torch.minimum(x2[:, None], x2[None, :])
                - torch.maximum(x1[:, None], x1[None, :])
            ).clamp(min=0.0)
            ih = (
                torch.minimum(y2[:, None], y2[None, :])
                - torch.maximum(y1[:, None], y1[None, :])
            ).clamp(min=0.0)
            cov = iw * ih / area[:, None]
            front = y2[None, :] > y2[:, None]
            cov = torch.where(front, cov, torch.zeros_like(cov))
            for li in dirty_local:
                lj = int(cov[li].argmax())
                ri, rj = idxs[li], idxs[lj]
                coverer[(records[ri].track_id, records[ri].frame)] = records[
                    rj
                ].track_id
        dirty.update(idxs[li] for li in dirty_local)

    by_track: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_track[r.track_id].append((r.frame, ri not in dirty))
    for tid in by_track:
        by_track[tid].sort()
    return dict(by_track), coverer


def plan_occ_audit_episodes(
    records: list[Any],
    *,
    appearance_occlusion_cov: float = 0.4,
    ref_n: int = 5,
    audit_crops: int = 3,
    audit_window: int = 30,
    min_occ_frames: int = 2,
) -> list[_Episode]:
    """Geometry-only pass: occlusion episodes + the frames the audit will need.

    ``ref_frames`` are the last ``ref_n`` clean frames strictly before the
    episode; ``audit_frames`` the first ``audit_crops`` clean frames within
    ``audit_window`` frames after it. Identity boundaries from earlier flags
    are applied later, at decision time (they only shrink the reference).
    """
    episodes: list[_Episode] = []
    by_track, coverer = _clean_flags_by_track(records, appearance_occlusion_cov)
    clean_by_track = {tid: [f for f, c in seq if c] for tid, seq in by_track.items()}
    for tid, seq in by_track.items():
        clean_frames = clean_by_track[tid]
        i = 0
        n = len(seq)
        while i < n:
            if seq[i][1]:
                i += 1
                continue
            j = i
            while j < n and not seq[j][1]:
                j += 1
            # dirty run spans records i..j-1
            if (j - i) >= min_occ_frames and j < n:  # needs a post-exit frame
                occ_start, occ_end = seq[i][0], seq[j - 1][0]
                refs = [f for f in clean_frames if f < occ_start][-ref_n:]
                audits = [
                    f for f in clean_frames if occ_end < f <= occ_end + audit_window
                ][:audit_crops]
                covs = [coverer[(tid, f)] for f, _ in seq[i:j] if (tid, f) in coverer]
                occ_id = max(set(covs), key=covs.count) if covs else -1
                occ_refs = (
                    [f for f in clean_by_track.get(occ_id, []) if f < occ_start][
                        -ref_n:
                    ]
                    if occ_id >= 0
                    else []
                )
                episodes.append(
                    _Episode(tid, occ_start, occ_end, refs, audits, occ_id, occ_refs)
                )
            i = j
    episodes.sort(key=lambda e: (e.track_id, e.occ_start))
    return episodes


def occ_exit_audit_lines(
    results_lines: list[str],
    embs: dict[tuple[int, int], Tensor],
    *,
    enabled: bool = False,
    tau: float = 0.45,
    min_ref: int = 2,
    ref_n: int = 5,
    audit_crops: int = 3,
    audit_window: int = 30,
    min_occ_frames: int = 2,
    appearance_occlusion_cov: float = 0.4,
    flag_consensus: int = 1,
    self_consistency_min: float = 0.0,
    occluder_margin: float = -1.0,
    decision_log: list[dict[str, int | float | bool]] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Causally split tracks whose post-occlusion appearance contradicts their
    pre-occlusion reference.

    Args:
        results_lines: MOT ``frame,id,x,y,w,h,score,...`` lines.
        embs: ``{(track_id, frame): [D]}`` L2-normalized embeddings covering at
            least the frames returned by :func:`plan_occ_audit_episodes`.
        tau: flag when an audit-crop cosine falls below this (clean-stream
            quantile; probe: 0.386 = 0.1% false rate, 0.498 = 0.5%).
        min_ref: minimum clean reference samples, else the episode abstains.
        flag_consensus: how many audit crops must fall below ``tau`` before the
            episode flags (1 = min-of-N; 2+ trades recall for precision — the
            geometric clean gate passes far dirtier crops than the GT-vis gate
            the probe thresholds came from).
        self_consistency_min: if > 0, the audit crops must agree with each
            other (mean pairwise cosine >= this) for a flag to count — a true
            identity transfer is a *consistent different* person, while junk
            crops disagree with everything including each other.
        occluder_margin: if >= 0, a below-``tau`` crop only counts when it is
            also *more similar to the episode's occluder* than to the track's
            own reference by at least this margin (ranking test: a true ABSORB
            follows the occluder's person; appearance drift does not). Episodes
            whose occluder has no clean reference abstain. Negative disables.

    Returns:
        (rewritten lines, stats dict).
    """
    stats = _AuditStats()
    if not enabled or not results_lines:
        return results_lines, vars(stats)

    records = _parse_mot_lines(results_lines)
    track_ids = {r.track_id for r in records}
    stats.ids_before = len(track_ids)
    episodes = plan_occ_audit_episodes(
        records,
        appearance_occlusion_cov=appearance_occlusion_cov,
        ref_n=ref_n,
        audit_crops=audit_crops,
        audit_window=audit_window,
        min_occ_frames=min_occ_frames,
    )
    stats.episodes = len(episodes)

    # (track_id, cut_frame): output id becomes fresh from cut_frame onward.
    cuts: dict[int, list[int]] = defaultdict(list)
    last_cut: dict[int, int] = {}

    for ep in episodes:
        boundary = last_cut.get(ep.track_id, -1)
        ref_vecs = [
            embs[(ep.track_id, f)]
            for f in ep.ref_frames
            if f >= boundary and (ep.track_id, f) in embs
        ]
        if len(ref_vecs) < min_ref:
            stats.abstain_no_ref += 1
            continue
        audit = [
            (f, embs[(ep.track_id, f)])
            for f in ep.audit_frames
            if (ep.track_id, f) in embs
        ]
        if not audit:
            stats.abstain_no_crops += 1
            continue
        stats.audited += 1

        ref = torch.nn.functional.normalize(
            torch.stack(ref_vecs).mean(dim=0, keepdim=True), dim=1
        )[0]
        cosines = [(f, float(ref @ v)) for f, v in audit]

        if len(audit) > 1:
            stack = torch.stack([v for _, v in audit])
            sim = stack @ stack.T
            n_a = sim.shape[0]
            self_consistency = float((sim.sum() - n_a) / max(1, n_a * (n_a - 1)))
        else:
            self_consistency = 1.0

        occ_ref = None
        max_contrast = float("nan")
        if occluder_margin >= 0.0:
            occ_vecs = [
                embs[(ep.occluder_id, f)]
                for f in ep.occluder_ref_frames
                if (ep.occluder_id, f) in embs
            ]
            if len(occ_vecs) < min_ref:
                stats.abstain_no_occref += 1
                continue
            occ_ref = torch.nn.functional.normalize(
                torch.stack(occ_vecs).mean(dim=0, keepdim=True), dim=1
            )[0]
            max_contrast = max(
                float(occ_ref @ v) - c for (_, c), (_, v) in zip(cosines, audit)
            )

        flag_frame = None
        below = 0
        for (f, c), (_, v) in zip(cosines, audit):
            if c >= tau:
                continue
            if occ_ref is not None and float(occ_ref @ v) - c < occluder_margin:
                continue  # not closer to the occluder: drift, not ABSORB
            below += 1
            if below >= flag_consensus:
                flag_frame = f  # causal: decision at the k-th low crop
                break
        if flag_frame is not None and self_consistency < self_consistency_min:
            flag_frame = None

        if decision_log is not None:
            decision_log.append(
                {
                    "track_id": int(ep.track_id),
                    "occ_start": int(ep.occ_start),
                    "occ_end": int(ep.occ_end),
                    "ref_n_used": int(len(ref_vecs)),
                    "ref_gap": int(
                        ep.occ_start - max(f for f in ep.ref_frames if f >= boundary)
                    ),
                    "audit_n": int(len(cosines)),
                    "min_cos": float(min(c for _, c in cosines)),
                    "median_cos": float(np.median([c for _, c in cosines])),
                    "self_consistency": float(self_consistency),
                    "cos_list": " ".join(f"{c:.3f}" for _, c in cosines),
                    "occluder_id": int(ep.occluder_id),
                    "max_contrast": float(max_contrast),
                    "tau": float(tau),
                    "flagged": bool(flag_frame is not None),
                    "flag_frame": int(flag_frame if flag_frame is not None else -1),
                }
            )
            stats.decisions_logged += 1

        if flag_frame is None:
            continue
        stats.flags += 1
        cuts[ep.track_id].append(flag_frame)
        last_cut[ep.track_id] = flag_frame

    if not cuts:
        stats.ids_after = stats.ids_before
        return results_lines, vars(stats)

    next_id = max(track_ids) + 1
    seg_id: dict[tuple[int, int], int] = {}  # (track_id, cut_frame) -> fresh id
    for tid, frames in cuts.items():
        for f in sorted(frames):
            seg_id[(tid, f)] = next_id
            next_id += 1
    for r in records:
        past = [f for f in cuts.get(r.track_id, []) if f <= r.frame]
        if past:
            r.track_id = seg_id[(r.track_id, max(past))]
    stats.ids_after = len({r.track_id for r in records})
    return _format_mot_records(records), vars(stats)


def extract_audit_embeddings(
    results_lines: list[str],
    seq_dir: str,
    extractor: Any,
    *,
    ref_n: int = 5,
    audit_crops: int = 3,
    audit_window: int = 30,
    min_occ_frames: int = 2,
    crop_hw: tuple[int, int] | None = None,
    im_ext: str = ".jpg",
    batch: int = 256,
    appearance_occlusion_cov: float = 0.4,
) -> dict[tuple[int, int], Tensor]:
    """Embeddings for exactly the frames the audit plan needs, one TRT pass."""
    from PIL import Image

    if crop_hw is None:
        crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))  # type: ignore[arg-type]

    records = _parse_mot_lines(results_lines)
    episodes = plan_occ_audit_episodes(
        records,
        appearance_occlusion_cov=appearance_occlusion_cov,
        ref_n=ref_n,
        audit_crops=audit_crops,
        audit_window=audit_window,
        min_occ_frames=min_occ_frames,
    )
    needed: set[tuple[int, int]] = set()
    for ep in episodes:
        needed.update((ep.track_id, f) for f in ep.ref_frames)
        needed.update((ep.track_id, f) for f in ep.audit_frames)
        if ep.occluder_id >= 0:
            needed.update((ep.occluder_id, f) for f in ep.occluder_ref_frames)
    if not needed:
        return {}

    box_by_key = {
        (r.track_id, r.frame): (r.x, r.y, r.x + r.w, r.y + r.h)
        for r in records
        if (r.track_id, r.frame) in needed
    }
    keys = sorted(box_by_key)
    pool = [(tid, fr, box_by_key[(tid, fr)]) for tid, fr in keys]
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
        # PIL fallback (same crop contract as the handover's fallback).
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

    feats = torch.nn.functional.normalize(feats.float(), dim=1)
    return {key: feats[i] for i, key in enumerate(keys)}
