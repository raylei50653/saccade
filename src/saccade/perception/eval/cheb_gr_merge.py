"""Cheb-GR offline tracklet merge (Cheb-GR path 2).

Appearance-based analogue of :mod:`post_merge`: instead of velocity/spatial
priors, tracklets are stitched by Chebyshev-guided graph re-ranking
(:func:`saccade.perception.reid.cheb_gr.cheb_gr_kreciprocal`) over per-tracklet
appearance samples. Targets the AssA bottleneck on MOT17.

Design (all flags default off; see docs/modules/semantic/TODO.md):

  1. Build tracklets from MOT output lines (reuses post_merge tracklet model).
  2. Per tracklet, *temporally-distributed* sampling of <= N detections:
     split the lifespan into N equal-width temporal bins and keep the
     highest-score detection in each bin -> covers the whole trajectory while
     preferring clean samples within each segment. (N = 20..100; 5 was too few
     to represent a tracklet robustly.)
  3. The caller crops + embeds the sampled detections (the C++ eval path emits
     no per-det embedding, so extraction is self-supplied) and hands back one
     [S_i, D] L2-normalized tensor per tracklet.
  4. All samples are pooled into a single graph; cheb_gr_kreciprocal refines the
     sample-level distances; tracklet-to-tracklet distance is a robust min over
     the cross-sample block (this is where the 20-100 samples pay off).
  5. Temporally-disjoint tracklet pairs are matched with linear_sum_assignment
     and merged via UnionFind, then the lines are relabeled.

This module keeps frame I/O and the embedding extractor out: callers pass
ready embeddings keyed by track_id. That makes the numeric core unit-testable
and lets the evaluator adapter own the (heavier, integration-y) extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..reid.cheb_gr import cheb_gr_kreciprocal
from .lifecycle import UnionFind
from .post_merge import _build_output_tracklets, _format_mot_records, _parse_mot_lines
from .types import OutputTracklet

__all__ = [
    "temporal_sample_indices",
    "tracklet_distance_matrix",
    "cheb_gr_merge_output_tracklets",
    "extract_tracklet_embeddings",
]


def temporal_sample_indices(
    n: int,
    n_samples: int,
    scores: np.ndarray | None = None,
) -> list[int]:
    """Indices of a temporally-distributed sample of a tracklet's records.

    Records are assumed time-ordered. The lifespan is split into ``n_samples``
    equal-width bins; from each non-empty bin the highest-``scores`` index is
    kept (or the bin midpoint when ``scores`` is None). Returns up to
    ``n_samples`` strictly increasing indices.

    Returns all indices unchanged when ``n <= n_samples``.
    """
    if n <= 0:
        return []
    if n_samples <= 0:
        return []
    if n <= n_samples:
        return list(range(n))

    edges = np.linspace(0, n, n_samples + 1).astype(int)
    out: list[int] = []
    for b in range(n_samples):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        if scores is not None:
            j = lo + int(np.argmax(scores[lo:hi]))
        else:
            j = (lo + hi - 1) // 2
        out.append(j)
    return out


def tracklet_distance_matrix(
    sample_feats: Tensor,
    sample_owner: Tensor,
    n_tracklets: int,
    *,
    pool_frac: float = 0.3,
    cheb_lambda: float = 2.0,
    k2: int = 6,
    max_fwd: int = 50,
    fuse_lambda: float = 0.3,
) -> Tensor:
    """Tracklet-to-tracklet distance [T, T] via Cheb-GR over pooled samples.

    Args:
        sample_feats: [S, D] L2-normalized appearance samples from all tracklets.
        sample_owner: [S] int tensor mapping each sample to its tracklet index
            in ``[0, n_tracklets)``.
        n_tracklets: number of tracklets T.
        pool_frac: tracklet distance = mean of the smallest ``pool_frac`` of the
            cross-sample distances (robust min — one clean matching view is
            enough to link, but a single outlier should not).

    Returns:
        [T, T] distance matrix; the diagonal is set to +inf.
    """
    inf = float("inf")
    dmat = torch.full((n_tracklets, n_tracklets), inf, device=sample_feats.device)
    if sample_feats.shape[0] == 0:
        return dmat

    # Sample-level Cheb-GR re-ranking over the whole set (query == gallery).
    sample_dist = cheb_gr_kreciprocal(
        sample_feats,
        sample_feats,
        cheb_lambda=cheb_lambda,
        k2=k2,
        max_fwd=max_fwd,
        fuse_lambda=fuse_lambda,
    )  # [S, S]

    owner = sample_owner.to(sample_feats.device)
    member_idx = [
        torch.nonzero(owner == t, as_tuple=True)[0] for t in range(n_tracklets)
    ]

    for a in range(n_tracklets):
        ia = member_idx[a]
        if ia.numel() == 0:
            continue
        for b in range(a + 1, n_tracklets):
            ib = member_idx[b]
            if ib.numel() == 0:
                continue
            block = sample_dist[ia][:, ib].reshape(-1)
            k = max(1, int(round(pool_frac * block.numel())))
            pooled = torch.topk(block, k, largest=False).values.mean()
            dmat[a, b] = pooled
            dmat[b, a] = pooled
    return dmat


@dataclass
class _MergeStats:
    ids_before: int = 0
    ids_after: int = 0
    merges: int = 0


def cheb_gr_merge_output_tracklets(
    results_lines: list[str],
    embeddings: dict[int, Tensor],
    *,
    enabled: bool = False,
    max_cost: float = 0.55,
    max_gap: int = 60,
    min_overlap_frames: int = 1,
    pool_frac: float = 0.3,
    cheb_lambda: float = 2.0,
    k2: int = 6,
    max_fwd: int = 50,
    fuse_lambda: float = 0.3,
) -> tuple[list[str], dict[str, int]]:
    """Merge temporally-disjoint tracklets by Cheb-GR appearance similarity.

    Args:
        results_lines: MOT17 ``frame,id,x,y,w,h,score,...`` lines.
        embeddings: ``{track_id: [S_i, D]}`` L2-normalized per-tracklet samples
            (already temporally sampled + extracted by the caller). Tracklets
            without an entry (or empty) are left untouched.
        max_cost: maximum Cheb-GR distance accepted for a merge.
        max_gap: max frame gap between an earlier tracklet's end and a later
            tracklet's start to be merge-eligible.
        min_overlap_frames: tracklets overlapping by more than this many frames
            are never merged (they coexist -> different identities).

    Returns:
        (rewritten lines, stats dict).
    """
    stats = _MergeStats()
    if not enabled or not results_lines:
        return results_lines, vars(stats)

    records = _parse_mot_lines(results_lines)
    tracklets: list[OutputTracklet] = _build_output_tracklets(
        records, velocity_samples=5
    )
    stats.ids_before = len(tracklets)
    if len(tracklets) <= 1:
        stats.ids_after = len(tracklets)
        return results_lines, vars(stats)

    # Keep only tracklets we have embeddings for; index them densely.
    indexed: list[tuple[int, OutputTracklet]] = [
        (i, t)
        for i, t in enumerate(tracklets)
        if t.track_id in embeddings and embeddings[t.track_id].shape[0] > 0
    ]
    if len(indexed) <= 1:
        stats.ids_after = len(tracklets)
        return results_lines, vars(stats)

    feats_list: list[Tensor] = []
    owner_list: list[int] = []
    for dense_i, (_, t) in enumerate(indexed):
        emb = embeddings[t.track_id]
        feats_list.append(emb)
        owner_list.extend([dense_i] * emb.shape[0])
    sample_feats = torch.cat(feats_list, dim=0)
    sample_owner = torch.tensor(owner_list, dtype=torch.long)

    dmat = tracklet_distance_matrix(
        sample_feats,
        sample_owner,
        len(indexed),
        pool_frac=pool_frac,
        cheb_lambda=cheb_lambda,
        k2=k2,
        max_fwd=max_fwd,
        fuse_lambda=fuse_lambda,
    )

    # Build merge candidates (a < b) that pass the pairwise temporal gate and
    # the cost ceiling. Pairwise disjointness alone is NOT enough: UnionFind is
    # transitive, so A~B and B~C can chain A and C into one id even when A and C
    # co-exist -> duplicate id in a frame. We therefore greedily accept the
    # cheapest candidates and only union when the two components' frame sets are
    # actually disjoint (verified against the live merged frame set).
    candidates: list[tuple[float, int, int]] = []
    for ai in range(len(indexed)):
        ta = indexed[ai][1]
        for bi in range(ai + 1, len(indexed)):
            tb = indexed[bi][1]
            c = float(dmat[ai, bi])
            if c > max_cost:
                continue
            earlier, later = (ta, tb) if ta.end <= tb.end else (tb, ta)
            overlap = earlier.end - later.start
            gap = later.start - earlier.end
            if overlap > min_overlap_frames or gap < 0 or gap > max_gap:
                continue
            candidates.append((c, ai, bi))
    candidates.sort(key=lambda x: x[0])

    # Per-tracklet frame sets; components accumulate the union of their frames.
    frames_by_id: dict[int, set[int]] = {}
    for r in records:
        frames_by_id.setdefault(r.track_id, set()).add(r.frame)

    uf = UnionFind([t.track_id for t in tracklets])
    comp_frames: dict[int, set[int]] = {
        t.track_id: set(frames_by_id.get(t.track_id, set())) for t in tracklets
    }
    for _, ai, bi in candidates:
        ida = indexed[ai][1].track_id
        idb = indexed[bi][1].track_id
        ra, rb = uf.find(ida), uf.find(idb)
        if ra == rb:
            continue
        fa, fb = comp_frames[ra], comp_frames[rb]
        # Component frame sets must be STRICTLY disjoint: any shared frame would
        # put the merged id twice in that timestep (invalid MOT output). The
        # pairwise min_overlap_frames leniency above does not survive chaining.
        if fa & fb:
            continue
        # Canonical id = the earlier-ending component's root (stable identity).
        keep, drop = (ra, rb) if max(fa) <= max(fb) else (rb, ra)
        uf.union(keep, drop)
        comp_frames[keep] = fa | fb
        stats.merges += 1

    if stats.merges == 0:
        stats.ids_after = len(tracklets)
        return results_lines, vars(stats)

    for record in records:
        record.track_id = uf.find(record.track_id)

    remaining = {uf.find(t.track_id) for t in tracklets}
    stats.ids_after = len(remaining)
    return _format_mot_records(records), vars(stats)


def extract_tracklet_embeddings(
    results_lines: list[str],
    seq_dir: str,
    extractor: Any,
    *,
    n_samples: int = 50,
    crop_hw: tuple[int, int] = (224, 224),
    im_ext: str = ".jpg",
    batch: int = 256,
) -> dict[int, Tensor]:
    """Per-tracklet L2-normalized appearance samples for offline merge.

    Mirrors the *proven* siglip2_reid preprocessing of the standalone Market gate
    (``scripts/eval/cheb_gr_osnet_gate.py``, mAP 81.31%): PIL ``convert("RGB")``
    + bilinear resize + ``/255`` -> ``extractor.extract``. The C++ eval path
    emits no per-det embedding, so detections are re-cropped from ``img1`` here.

    Frames are read once each (tracklets share frames). Returns
    ``{track_id: [S_i, D]}`` on the extractor's device.
    """
    from collections import defaultdict

    from PIL import Image

    records = _parse_mot_lines(results_lines)
    by_id: dict[int, list[Any]] = defaultdict(list)
    for r in records:
        by_id[r.track_id].append(r)

    # Temporally-distributed sampling -> flat list of (track_id, frame, xyxy box).
    samples: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for tid, items in by_id.items():
        items.sort(key=lambda r: r.frame)
        scores = np.asarray([r.score for r in items], dtype=np.float32)
        for j in temporal_sample_indices(len(items), n_samples, scores=scores):
            r = items[j]
            samples.append((tid, r.frame, (r.x, r.y, r.x + r.w, r.y + r.h)))
    if not samples:
        return {}

    # Group by frame so each JPEG is decoded exactly once.
    by_frame: dict[int, list[int]] = defaultdict(list)
    for si, (_, frame, _) in enumerate(samples):
        by_frame[frame].append(si)

    out_h, out_w = crop_hw
    crop_arrs: list[np.ndarray | None] = [None] * len(samples)
    seq_path = seq_dir
    for frame, si_list in by_frame.items():
        img = Image.open(f"{seq_path}/{frame:06d}{im_ext}").convert("RGB")
        fw, fh = img.size
        for si in si_list:
            x1, y1, x2, y2 = samples[si][2]
            box = (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(fw, int(round(x2))),
                min(fh, int(round(y2))),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                box = (0, 0, fw, fh)
            crop = img.crop(box).resize((out_w, out_h), Image.BILINEAR)  # type: ignore[attr-defined]
            crop_arrs[si] = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)

    device = getattr(extractor, "device", "cuda")
    feats = torch.empty((len(samples), extractor.feature_dim), device=device)
    for start in range(0, len(samples), batch):
        chunk = crop_arrs[start : start + batch]
        arr = np.stack([c for c in chunk if c is not None])
        tensor = torch.from_numpy(arr).to(device).float().div_(255.0)
        feats[start : start + tensor.shape[0]] = extractor.extract(tensor)
    feats = torch.nn.functional.normalize(feats, dim=1)

    result: dict[int, Tensor] = {}
    owner = torch.tensor([s[0] for s in samples])
    for tid in by_id:
        rows = torch.nonzero(owner == tid, as_tuple=True)[0]
        if rows.numel() > 0:
            result[tid] = feats[rows.to(device)]
    return result
