"""Clean-FIFO embedding bank — reusable substrate for identity relink.

Encapsulates the probe-validated (2026-07-04, registry #58) sparse key-embedding
strategy: per-track FIFO of the most recent ``fifo_n`` visclean-gated clean
samples, with optional stride-scheduled extraction and event-aware birth window.

Hard constraints (probe findings, must not be violated by consumers):

1. **No mean / prototype for Cheb-GR** — store raw embeddings only.
   ``mean1`` single-prototype collapsed intra-track variance and made
   Cheb-GR k-reciprocal distances meaningless (IDF1 −2.61 / −1.14).
   ``representative()`` returns a mean but is labelled GRAPH-EXTERNAL only
   (direct cosine for assoc cost), never for Cheb-GR k-reciprocal.

2. **No duplicate embeddings in Cheb-GR graph** — ``dupfill`` copies are
   self-nearest-neighbours that squeeze out k2 neighbourhood, making the
   mechanism conservative (accepts 51→31, IDs up). Copies may be used for
   graph-external purposes (per-frame assoc cost) but never fed into the
   k-reciprocal gallery. The bank stores unique extractions only.

3. **No quality-signal selection** — after visclean front-occlusion gating,
   det-score / box-height / neighbour-IoU have no residual correlation with
   embedding self-consistency (Spearman ≈ 0 / +0.15 / +0.1, m/s reproduced).
   Quality-gated top-K selection loses to same-budget temporal sampling.

Consumers:
  - ``cheb_gr_online.extract_handover_embeddings`` (current, via adapter)
  - ``occ_audit`` pre-episode reference (#55, Phase 2a)
  - ``TrackAppearanceBank`` / ``OutputAppearanceBank`` FIFO replacement
    (Phase 2b, probe-only)
  - Online assoc cost query via forwarded embeddings (Phase 2c, probe-only)
  - C++ async sidecar ring buffer (#57, spec only)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .helpers import front_occlusion_mask_xyxy
from .post_merge import _build_output_tracklets, _parse_mot_lines
from .utils import box_iou_tuple

__all__ = [
    "CleanFifoBank",
    "CleanFifoPlan",
    "build_filled_bank",
    "plan_clean_fifo_crops",
]

_BOX = tuple[float, float, float, float]
_CROP = tuple[int, _BOX]  # (frame_id, (x1, y1, x2, y2))


@dataclass
class _FifoEntry:
    frame_id: int
    embedding: Tensor


@dataclass
class CleanFifoPlan:
    """Post-hoc crop extraction plan built from substrate records.

    ``head_crops``: per-track clean unpolluted crops in the birth window
    (first ``decide_n`` frames) — the handover "head" evidence.
    ``bank_crops``: per-track crops selected by the FIFO + stride scheduler
    — to be extracted and stored into ``bank``.
    ``bank``: a ``CleanFifoBank`` with clean-frame counters already filled
    by the planner's dry run; call ``bank.store()`` for each extracted
    embedding to complete the FIFO.
    """

    head_crops: dict[int, list[_CROP]]
    bank_crops: dict[int, list[_CROP]]
    bank: CleanFifoBank


class CleanFifoBank:
    """Per-track clean-FIFO raw-embedding bank.

    Holds up to ``fifo_n`` most recent visclean-clean samples per track.
    Stride scheduling (``stride > 1``) extracts every ``stride``-th clean
    frame after the birth window, reducing extraction cost by ~1/stride.
    The birth window (first ``decide_n`` frames) is always dense-extracted
    so ``min_head`` evidence is not starved (#56 lesson).
    """

    def __init__(
        self,
        *,
        fifo_n: int = 20,
        stride: int = 1,
        decide_n: int = 5,
    ) -> None:
        self.fifo_n = max(1, fifo_n)
        self.stride = max(1, stride)
        self.decide_n = max(1, decide_n)
        self._samples: dict[int, list[_FifoEntry]] = {}
        self._clean_seen: dict[int, int] = {}

    def should_extract(
        self,
        track_id: int,
        *,
        is_clean: bool,
        frame_offset_from_birth: int = 0,
    ) -> bool:
        """Decide whether to extract this frame's embedding.

        Increments the per-track clean-frame counter as a side effect
        (needed for stride scheduling). Call once per frame in frame order.
        """
        if not is_clean:
            return False
        seen = self._clean_seen.get(track_id, 0)
        self._clean_seen[track_id] = seen + 1
        in_birth = frame_offset_from_birth < self.decide_n
        if in_birth:
            return True
        if self.stride <= 1:
            return True
        return (seen % self.stride) == 0

    def store(
        self,
        track_id: int,
        embedding: Tensor,
        frame_id: int,
    ) -> None:
        """Store a raw embedding (already extracted). FIFO eviction by age."""
        emb = F.normalize(embedding.detach().float(), dim=0)
        samples = self._samples.setdefault(track_id, [])
        samples.append(_FifoEntry(frame_id, emb))
        excess = len(samples) - self.fifo_n
        if excess > 0:
            del samples[:excess]

    def update(
        self,
        track_id: int,
        embedding: Tensor,
        frame_id: int,
        *,
        is_clean: bool,
        frame_offset_from_birth: int = 0,
    ) -> bool:
        """Online entry point: schedule + store. Returns True if stored."""
        if self.should_extract(
            track_id,
            is_clean=is_clean,
            frame_offset_from_birth=frame_offset_from_birth,
        ):
            self.store(track_id, embedding, frame_id)
            return True
        return False

    def samples(self, track_id: int) -> Tensor | None:
        """Raw FIFO embeddings for Cheb-GR k-reciprocal graph (NOT mean)."""
        s = self._samples.get(track_id, [])
        if not s:
            return None
        return torch.stack([e.embedding for e in s])

    def samples_before(self, track_id: int, frame_id: int) -> Tensor | None:
        """FIFO entries with frame < ``frame_id`` (occ-audit pre-episode ref)."""
        s = self._samples.get(track_id, [])
        filtered = [e.embedding for e in s if e.frame_id < frame_id]
        if not filtered:
            return None
        return torch.stack(filtered)

    def frames(self, track_id: int) -> list[int]:
        """Frame IDs of stored samples (for audit/debug)."""
        return [e.frame_id for e in self._samples.get(track_id, [])]

    def representative(self, track_id: int) -> Tensor | None:
        """Mean of FIFO samples — **GRAPH-EXTERNAL cosine only**.

        Safe for direct cosine assoc cost. Never feed into Cheb-GR
        k-reciprocal (probe: mean1 = −2.61 IDF1, distance collapse).
        """
        s = self.samples(track_id)
        if s is None:
            return None
        return F.normalize(s.mean(dim=0), dim=0)

    def clean_ids(self) -> set[int]:
        return {tid for tid, s in self._samples.items() if s}

    def count(self, track_id: int) -> int:
        return len(self._samples.get(track_id, []))

    def clean_seen(self, track_id: int) -> int:
        return self._clean_seen.get(track_id, 0)

    def prune(self, alive_ids: set[int] | list[int]) -> None:
        alive = set(alive_ids)
        self._samples = {t: s for t, s in self._samples.items() if t in alive}
        self._clean_seen = {t: c for t, c in self._clean_seen.items() if t in alive}

    def fill_from(
        self,
        crops: dict[int, list[_CROP]],
        embeddings: dict[int, Tensor],
    ) -> None:
        """Store extracted embeddings indexed by crop order.

        ``crops[tid]`` is the ordered list planned by ``plan_clean_fifo_crops``;
        ``embeddings[tid]`` is the tensor row for each crop in order.
        """
        for tid, crop_list in crops.items():
            embs = embeddings.get(tid)
            if embs is None:
                continue
            for i, (frame_id, _box) in enumerate(crop_list):
                if i < embs.shape[0]:
                    self.store(tid, embs[i], frame_id)


def _compute_dirty_set(
    records: list[Any],
    appearance_occlusion_cov: float,
) -> set[int]:
    """Records covered by a lower-foot foreground box (visclean gate)."""
    by_frame: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_frame[r.frame].append(ri)
    dirty: set[int] = set()
    for idxs in by_frame.values():
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
    return dirty


def _compute_polluted_set(
    records: list[Any],
    neighbor_iou_max: float,
) -> set[tuple[int, int, float, float, float, float]]:
    """Records whose box overlaps another track's box at >= ``neighbor_iou_max``."""
    if neighbor_iou_max <= 0.0:
        return set()
    by_frame: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_frame[r.frame].append(ri)
    polluted: set[tuple[int, int, float, float, float, float]] = set()
    for idxs in by_frame.values():
        boxes = [
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
                if box_iou_tuple(boxes[a], boxes[b]) >= neighbor_iou_max:
                    polluted.add(
                        (
                            records[idxs[a]].track_id,
                            records[idxs[a]].frame,
                            records[idxs[a]].x,
                            records[idxs[a]].y,
                            records[idxs[a]].w,
                            records[idxs[a]].h,
                        )
                    )
                    polluted.add(
                        (
                            records[idxs[b]].track_id,
                            records[idxs[b]].frame,
                            records[idxs[b]].x,
                            records[idxs[b]].y,
                            records[idxs[b]].w,
                            records[idxs[b]].h,
                        )
                    )
    return polluted


def plan_clean_fifo_crops(
    records: list[Any],
    *,
    appearance_occlusion_cov: float = 0.4,
    fifo_n: int = 20,
    stride: int = 1,
    decide_n: int = 5,
    neighbor_iou_max: float = 0.0,
) -> CleanFifoPlan:
    """Plan head + bank crops from substrate records (post-hoc, pre-extraction).

    Determines which (track_id, frame, box) crops to extract:
    - **head**: clean + unpolluted records in the birth window
      (frame < ``decide_n`` after track start). Falls to empty if the
      strict filter empties it (newborn without enough clean head crops
      hits ``min_head`` gate downstream).
    - **bank**: clean records selected by the FIFO + stride scheduler.
      Falls back to the unfiltered clean pool when the pollution filter
      would empty it, so archive candidates stay scoreable.

    The returned ``CleanFifoBank`` has its clean-frame counters filled by
    the planner's dry run — call ``bank.store()`` for each extracted
    embedding (in the same frame order the planner saw) to complete the FIFO.
    """
    dirty = _compute_dirty_set(records, appearance_occlusion_cov)
    clean_by_id: dict[int, list[Any]] = defaultdict(list)
    for ri, r in enumerate(records):
        if ri not in dirty:
            clean_by_id[r.track_id].append(r)

    polluted = _compute_polluted_set(records, neighbor_iou_max)

    def _rkey(r: Any) -> tuple[int, int, float, float, float, float]:
        return (r.track_id, r.frame, r.x, r.y, r.w, r.h)

    tracklets = _build_output_tracklets(records, velocity_samples=5)
    start_by_id = {t.track_id: t.start for t in tracklets}

    bank = CleanFifoBank(fifo_n=fifo_n, stride=stride, decide_n=decide_n)
    head_crops: dict[int, list[_CROP]] = {}
    bank_crops: dict[int, list[_CROP]] = {}

    for tid, items in clean_by_id.items():
        items.sort(key=lambda r: r.frame)
        birth = start_by_id.get(tid, items[0].frame)
        unpolluted = [r for r in items if _rkey(r) not in polluted]

        head_crops[tid] = [
            (r.frame, (r.x, r.y, r.x + r.w, r.y + r.h))
            for r in unpolluted
            if r.frame < birth + decide_n
        ]

        bank_pool = unpolluted if unpolluted else items
        accepted: list[_CROP] = []
        for r in bank_pool:
            offset = r.frame - birth
            if bank.should_extract(tid, is_clean=True, frame_offset_from_birth=offset):
                accepted.append((r.frame, (r.x, r.y, r.x + r.w, r.y + r.h)))
        bank_crops[tid] = accepted[-bank.fifo_n :]

    return CleanFifoPlan(head_crops=head_crops, bank_crops=bank_crops, bank=bank)


def build_filled_bank(
    results_lines: list[str],
    seq_dir: str,
    extractor: Any,
    *,
    appearance_occlusion_cov: float = 0.4,
    fifo_n: int = 20,
    stride: int = 1,
    decide_n: int = 5,
    neighbor_iou_max: float = 0.0,
    crop_hw: tuple[int, int] | None = None,
    im_ext: str = ".jpg",
    batch: int = 256,
) -> CleanFifoBank:
    """Build a filled ``CleanFifoBank`` from substrate MOT lines, one TRT pass.

    Plans crops → extracts via native C++/CUDA + TRT (PIL fallback) → fills
    the bank's FIFO. The returned bank is ready for ``samples()``,
    ``samples_before()``, ``representative()`` queries by any consumer
    (occ-audit reference, handover, assoc cost).
    """
    from PIL import Image

    from .cheb_gr_merge import _extract_native_crops_trt

    if crop_hw is None:
        crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))

    records = _parse_mot_lines(results_lines)
    plan = plan_clean_fifo_crops(
        records,
        appearance_occlusion_cov=appearance_occlusion_cov,
        fifo_n=fifo_n,
        stride=stride,
        decide_n=decide_n,
        neighbor_iou_max=neighbor_iou_max,
    )

    pool: list[tuple[int, int, _BOX]] = []
    pool_idx: dict[tuple[int, int, float, float, float, float], int] = {}
    for tid, crops in plan.bank_crops.items():
        for frame, box in crops:
            key = (tid, frame, box[0], box[1], box[2] - box[0], box[3] - box[1])
            if key not in pool_idx:
                pool_idx[key] = len(pool)
                pool.append((tid, frame, box))
    if not pool:
        return plan.bank

    by_frame: dict[int, list[int]] = defaultdict(list)
    for si, (_, fr, _) in enumerate(pool):
        by_frame[fr].append(si)

    feats = _extract_native_crops_trt(
        pool,
        by_frame,
        seq_dir,
        extractor,
        crop_hw=crop_hw,
        im_ext=im_ext,
        batch=batch,
    )
    if feats is None:
        import numpy as _np

        out_h, out_w = crop_hw
        arrs: list[_np.ndarray | None] = [None] * len(pool)
        for fr, si_list in by_frame.items():
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
                arrs[si] = _np.asarray(crop, dtype=_np.uint8).transpose(2, 0, 1)
        device = getattr(extractor, "device", "cuda")
        feats = torch.empty((len(pool), extractor.feature_dim), device=device)
        for s in range(0, len(pool), max(1, batch)):
            chunk = [a for a in arrs[s : s + batch] if a is not None]
            t = torch.from_numpy(_np.stack(chunk)).to(device).float().div_(255.0)
            feats[s : s + t.shape[0]] = extractor.extract(t)
        feats = torch.nn.functional.normalize(feats, dim=1)

    for tid, crops in plan.bank_crops.items():
        indices = [
            pool_idx[(tid, f, b[0], b[1], b[2] - b[0], b[3] - b[1])] for f, b in crops
        ]
        if indices:
            embs = feats[torch.tensor(indices, device=feats.device)]
            for i, (frame_id, _) in enumerate(crops):
                plan.bank.store(tid, embs[i], frame_id)

    return plan.bank
