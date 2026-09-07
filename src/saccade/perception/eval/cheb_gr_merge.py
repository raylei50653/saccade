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
from .helpers import front_occlusion_mask_xyxy
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
    decision_log: list[dict[str, Any]] | None = None,
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
    if decision_log is not None:
        indexed_ids = {t.track_id for _, t in indexed}
        for t in tracklets:
            emb = embeddings.get(t.track_id)
            decision_log.append(
                {
                    "kind": "tracklet",
                    "a_id": t.track_id,
                    "b_id": -1,
                    "cost": "",
                    "gap": "",
                    "overlap": "",
                    "n_samples": 0 if emb is None else int(emb.shape[0]),
                    "start": t.start,
                    "end": t.end,
                    "verdict": "has_embedding"
                    if t.track_id in indexed_ids
                    else "no_embedding",
                }
            )
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
            earlier, later = (ta, tb) if ta.end <= tb.end else (tb, ta)
            overlap = earlier.end - later.start
            gap = later.start - earlier.end
            if c > max_cost:
                verdict = "reject_cost"
            elif overlap > min_overlap_frames or gap < 0 or gap > max_gap:
                verdict = "reject_temporal"
            else:
                verdict = "pending"
                candidates.append((c, ai, bi))
            if decision_log is not None:
                decision_log.append(
                    {
                        "kind": "pair",
                        "a_id": ta.track_id,
                        "b_id": tb.track_id,
                        "cost": c,
                        "gap": gap,
                        "overlap": overlap,
                        "n_samples": "",
                        "start": "",
                        "end": "",
                        "verdict": verdict,
                    }
                )
    candidates.sort(key=lambda x: x[0])

    # Per-tracklet frame sets; components accumulate the union of their frames.
    frames_by_id: dict[int, set[int]] = {}
    for r in records:
        frames_by_id.setdefault(r.track_id, set()).add(r.frame)

    uf = UnionFind([t.track_id for t in tracklets])
    comp_frames: dict[int, set[int]] = {
        t.track_id: set(frames_by_id.get(t.track_id, set())) for t in tracklets
    }
    pending_rows: dict[tuple[int, int], dict[str, Any]] = {}
    if decision_log is not None:
        pending_rows = {
            (row["a_id"], row["b_id"]): row
            for row in decision_log
            if row["kind"] == "pair" and row["verdict"] == "pending"
        }

    def _resolve(a: int, b: int, verdict: str) -> None:
        row = pending_rows.get((a, b))
        if row is not None:
            row["verdict"] = verdict

    for _, ai, bi in candidates:
        ida = indexed[ai][1].track_id
        idb = indexed[bi][1].track_id
        ra, rb = uf.find(ida), uf.find(idb)
        if ra == rb:
            _resolve(ida, idb, "reject_same_component")
            continue
        fa, fb = comp_frames[ra], comp_frames[rb]
        # Component frame sets must be STRICTLY disjoint: any shared frame would
        # put the merged id twice in that timestep (invalid MOT output). The
        # pairwise min_overlap_frames leniency above does not survive chaining.
        if fa & fb:
            _resolve(ida, idb, "reject_component_overlap")
            continue
        # Canonical id = the earlier-ending component's root (stable identity).
        keep, drop = (ra, rb) if max(fa) <= max(fb) else (rb, ra)
        uf.union(keep, drop)
        comp_frames[keep] = fa | fb
        _resolve(ida, idb, "accepted")
        stats.merges += 1

    if stats.merges == 0:
        stats.ids_after = len(tracklets)
        return results_lines, vars(stats)

    for record in records:
        record.track_id = uf.find(record.track_id)

    remaining = {uf.find(t.track_id) for t in tracklets}
    stats.ids_after = len(remaining)
    return _format_mot_records(records), vars(stats)


def _native_fallback(reason: str) -> None:
    """Announce PIL fallback loudly: a silent fallback makes native-vs-PIL
    A/B reports lie about which path actually produced the embeddings."""
    print(f"⚠️  cheb-gr merge: native crop path unavailable ({reason}), using PIL")


def _native_cropper(output_hw: tuple[int, int]) -> Any | None:
    try:
        from saccade_perception_ext import Cropper as CropperCpp
    except ImportError:
        _native_fallback("saccade_perception_ext import failed")
        return None

    out_h, out_w = output_hw
    try:
        return CropperCpp(out_w, out_h)
    except Exception as exc:
        _native_fallback(f"CropperCpp init failed: {exc}")
        return None


def _load_frame_chw_gpu(
    path: str, device: str | torch.device
) -> tuple[Tensor, int, int]:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    frame = (
        torch.from_numpy(arr)
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .div_(255.0)
        .permute(2, 0, 1)
        .contiguous()
    )
    return frame, h, w


def _extract_native_crops_trt(
    samples: list[tuple[int, int, tuple[float, float, float, float]]],
    by_frame: dict[int, list[int]],
    seq_dir: str,
    extractor: Any,
    *,
    crop_hw: tuple[int, int],
    im_ext: str,
    batch: int,
) -> Tensor | None:
    """CUDA crop + TensorRT extraction for offline tracklet samples.

    Frames are still JPEG-decoded on CPU, but every expensive per-crop operation
    moves to the same native path used online: C++ CUDA cropper + C++ TRT
    FeatureExtractor. Returns ``None`` when native cropper/extractor support is
    unavailable so the caller can use the PIL fallback.
    """
    if not torch.cuda.is_available():
        _native_fallback("CUDA not available")
        return None
    try:
        if int(extractor.cpp_ptr) == 0:
            _native_fallback("extractor has no C++ TRT backend (cpp_ptr == 0)")
            return None
    except Exception as exc:
        _native_fallback(f"extractor cpp_ptr check failed: {exc}")
        return None

    cropper = _native_cropper(crop_hw)
    if cropper is None:
        return None

    device = torch.device(getattr(extractor, "device", "cuda"))
    if device.type != "cuda":
        _native_fallback(f"extractor device is {device}, need cuda")
        return None
    feat_dim = int(extractor.feature_dim)
    feats = torch.empty((len(samples), feat_dim), device=device, dtype=torch.float32)
    pending_crops: list[Tensor] = []
    pending_rows: list[int] = []
    pending_n = 0
    stream = torch.cuda.current_stream(device)

    def flush() -> None:
        nonlocal pending_crops, pending_rows, pending_n
        if pending_n == 0:
            return
        crops = (
            pending_crops[0]
            if len(pending_crops) == 1
            else torch.cat(pending_crops, dim=0)
        )
        embeds = extractor.extract(crops, stream=stream)
        rows = torch.tensor(pending_rows, device=device, dtype=torch.long)
        feats[rows] = embeds
        pending_crops = []
        pending_rows = []
        pending_n = 0

    for frame, sample_indices in by_frame.items():
        frame_tensor, frame_h, frame_w = _load_frame_chw_gpu(
            f"{seq_dir}/{frame:06d}{im_ext}", device
        )
        boxes = torch.tensor(
            [samples[si][2] for si in sample_indices],
            device=device,
            dtype=torch.float32,
        ).contiguous()
        crops = torch.empty(
            (len(sample_indices), 3, crop_hw[0], crop_hw[1]),
            device=device,
            dtype=torch.float32,
        )
        cropper.process_gpu(
            frame_tensor.data_ptr(),
            frame_w,
            frame_h,
            boxes.data_ptr(),
            len(sample_indices),
            crops.data_ptr(),
            stream.cuda_stream,
        )
        pending_crops.append(crops)
        pending_rows.extend(sample_indices)
        pending_n += len(sample_indices)
        if pending_n >= batch:
            flush()

    flush()
    return feats


def extract_tracklet_embeddings(
    results_lines: list[str],
    seq_dir: str,
    extractor: Any,
    *,
    n_samples: int = 50,
    crop_hw: tuple[int, int] | None = None,
    im_ext: str = ".jpg",
    batch: int = 256,
    appearance_occlusion_gate: bool | None = None,
    appearance_occlusion_cov: float = 0.4,
    resample: str | None = None,
    prefer_native: bool = True,
) -> dict[int, Tensor]:
    """Per-tracklet L2-normalized appearance samples for offline merge.

    The C++ eval path emits no per-det embedding, so detections are re-cropped
    from ``img1`` here. For ``mobilenetv4_reid`` this mirrors the visclean
    train/online contract: filter lower-foot front-occluded crops before temporal
    sampling, stretch to the extractor input size, then ``/255`` ->
    ``extractor.extract``. When native support is available, the resize/crop and
    TensorRT inference use the C++/CUDA path; PIL remains the fallback.

    Frames are read once each (tracklets share frames). Returns
    ``{track_id: [S_i, D]}`` on the extractor's device.
    """
    from collections import defaultdict

    from PIL import Image

    model_type = str(getattr(extractor, "model_type", ""))
    if crop_hw is None:
        crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))
    batch = max(1, int(batch))
    if appearance_occlusion_gate is None:
        appearance_occlusion_gate = model_type == "mobilenetv4_reid"
    requested_resample = resample
    if resample is None:
        resample = "bicubic" if model_type == "mobilenetv4_reid" else "bilinear"
    resample_mode = (
        Image.Resampling.BICUBIC
        if resample.lower() == "bicubic"
        else Image.Resampling.BILINEAR
    )

    records = _parse_mot_lines(results_lines)
    dirty_record_idx: set[int] = set()
    if appearance_occlusion_gate:
        by_frame_idx: dict[int, list[int]] = defaultdict(list)
        for ri, r in enumerate(records):
            by_frame_idx[r.frame].append(ri)
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
            dirty_record_idx.update(i for i, dirty in zip(idxs, mask.tolist()) if dirty)

    by_id: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for ri, r in enumerate(records):
        if ri in dirty_record_idx:
            continue
        by_id[r.track_id].append((ri, r))

    # Temporally-distributed sampling -> flat list of (track_id, frame, xyxy box).
    samples: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for tid, items in by_id.items():
        items.sort(key=lambda item: item[1].frame)
        scores = np.asarray([r.score for _, r in items], dtype=np.float32)
        for j in temporal_sample_indices(len(items), n_samples, scores=scores):
            _, r = items[j]
            samples.append((tid, r.frame, (r.x, r.y, r.x + r.w, r.y + r.h)))
    if not samples:
        return {}

    # Group by frame so each JPEG is decoded exactly once.
    by_frame: dict[int, list[int]] = defaultdict(list)
    for si, (_, frame, _) in enumerate(samples):
        by_frame[frame].append(si)

    native_allowed = prefer_native and (
        requested_resample is None
        or requested_resample.lower() in {"native", "bilinear"}
    )
    if native_allowed:
        native_feats = _extract_native_crops_trt(
            samples,
            by_frame,
            seq_dir,
            extractor,
            crop_hw=crop_hw,
            im_ext=im_ext,
            batch=batch,
        )
        if native_feats is not None:
            result: dict[int, Tensor] = {}
            owner = torch.tensor([s[0] for s in samples])
            device = native_feats.device
            for tid in by_id:
                rows = torch.nonzero(owner == tid, as_tuple=True)[0]
                if rows.numel() > 0:
                    result[tid] = native_feats[rows.to(device)]
            return result

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
            crop = img.crop(box).resize((out_w, out_h), resample_mode)
            crop_arrs[si] = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)

    device = getattr(extractor, "device", torch.device("cuda"))
    feats = torch.empty((len(samples), extractor.feature_dim), device=device)
    for start in range(0, len(samples), batch):
        chunk = crop_arrs[start : start + batch]
        arr = np.stack([c for c in chunk if c is not None])
        tensor = torch.from_numpy(arr).to(device).float().div_(255.0)
        feats[start : start + tensor.shape[0]] = extractor.extract(tensor)
    feats = torch.nn.functional.normalize(feats, dim=1)

    result = {}
    owner = torch.tensor([s[0] for s in samples])
    for tid in by_id:
        rows = torch.nonzero(owner == tid, as_tuple=True)[0]
        if rows.numel() > 0:
            result[tid] = feats[rows.to(device)]
    return result
