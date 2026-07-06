"""Streaming (causal, real-time) online Cheb-GR handover substrate.

:mod:`cheb_gr_online` provides the *numeric decision core*
(:func:`causal_handover_lines`) plus an offline-style adapter that re-reads
crops from disk at end-of-sequence. This module provides the pieces needed to
run that same decision **live** during tracking, without a disk re-read and
without stalling the frame loop:

  * :class:`TrackCropRing` — a GPU pre-allocated raw-crop ring keyed by the
    never-reused ``track_uid``. Crops are harvested from the already-decoded
    frame buffer (``pool.as_rgb_chw()``) through the crop kernel on a side
    stream, so harvesting never blocks the tracker. Bounded memory: one
    contiguous ``[capacity, 3, H, W]`` buffer with global LRU eviction. This
    backs borderline re-query (dense recent-tail re-extraction on demand) —
    the Python analogue of the ``ReidCropStore`` C++ interface.

  * :class:`EmbeddingRing` — per-uid recent-tail embedding FIFO. New clean
    crops are embedded once and **appended** (opt #5: incremental, never
    re-extract the whole track, never collapse to a single mean — the mean-1
    prototype is a known regression). The sparse bank is a strided view of the
    tail; the dense re-query bank is the tail itself.

  * :class:`StreamingHandoverDriver` — buffers emitted MOT lines and the live
    embedding caches, gates *when* to run the decision (opt #1 lazy: only when
    an unresolved newborn matured since the last pass; opt #6 adaptive: trigger
    on newborn density rather than a fixed frame cadence), optionally runs the
    pass on a background thread (opt #3), and exposes a monotonic ``alias`` map
    that downstream emission applies. The decision itself is delegated verbatim
    to :func:`causal_handover_lines` so the validated operating point transfers
    exactly.

The decision-preserving optimizations (#1, #3, #5, #6) are implemented here.
The approximate ones — incremental graph reuse (#2) and LSH candidate
prefilter (#4) — change or reorder scored candidates and so must be validated
against the frozen operating point separately; they are exposed as explicit,
default-off knobs and documented at their call sites rather than folded into
the exact path.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .cheb_gr_online import _cache_key, causal_handover_lines

__all__ = [
    "TrackCropRing",
    "EmbeddingRing",
    "StreamingHandoverDriver",
]


class TrackCropRing:
    """GPU pre-allocated raw-crop ring keyed by never-reused ``track_uid``.

    Each uid retains its most recent ``depth`` crops (a per-track tail). The
    backing store is a single contiguous ``[capacity, 3, H, W]`` buffer; when
    it is full, the globally least-recently-stashed slot is recycled, so total
    GPU memory is bounded independent of how many tracks the sequence spawns.

    All bookkeeping is plain Python (host-side); only the crop payload lives on
    ``device``, which keeps the class unit-testable on CPU tensors.
    """

    def __init__(
        self,
        capacity: int,
        *,
        crop_hw: tuple[int, int] = (224, 224),
        depth: int = 20,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if capacity < depth:
            raise ValueError("capacity must be >= depth")
        self.capacity = int(capacity)
        self.depth = int(depth)
        self.crop_hw = (int(crop_hw[0]), int(crop_hw[1]))
        self.buf = torch.zeros(
            (self.capacity, 3, *self.crop_hw), device=device, dtype=dtype
        )
        self._uid_slots: dict[int, deque[int]] = {}
        self._slot_uid: list[int] = [-1] * self.capacity
        self._slot_frame: list[int] = [-1] * self.capacity
        self._slot_clean: list[bool] = [False] * self.capacity
        self._free: list[int] = list(range(self.capacity))
        # slot -> None, insertion order = recency (front = least recent).
        self._lru: OrderedDict[int, None] = OrderedDict()

    # -- internal slot management ------------------------------------------
    def _acquire_slot(self) -> int:
        if self._free:
            return self._free.pop()
        slot, _ = self._lru.popitem(last=False)  # evict global LRU
        uid = self._slot_uid[slot]
        dq = self._uid_slots.get(uid)
        if dq is not None:
            try:
                dq.remove(slot)
            except ValueError:
                pass
            if not dq:
                del self._uid_slots[uid]
        return slot

    def _reserve(self, uid: int, frame: int, clean: bool) -> int:
        """Reserve (and book-keep) a slot for ``uid`` without writing pixels."""
        dq = self._uid_slots.get(uid)
        if dq is not None and len(dq) >= self.depth:
            old = dq.popleft()  # drop this uid's oldest crop
            self._lru.pop(old, None)
            self._slot_uid[old] = -1
            self._free.append(old)
        # _acquire_slot may evict this same uid's last slot and delete its map
        # entry, so the deque is (re-)fetched only after acquisition.
        slot = self._acquire_slot()
        self._slot_uid[slot] = int(uid)
        self._slot_frame[slot] = int(frame)
        self._slot_clean[slot] = bool(clean)
        self._uid_slots.setdefault(uid, deque()).append(slot)
        self._lru[slot] = None
        self._lru.move_to_end(slot)
        return slot

    # -- public API --------------------------------------------------------
    def stash(self, uid: int, frame: int, crop: Tensor, *, clean: bool = True) -> int:
        """Stash one ``[3, H, W]`` crop for ``uid``; returns the slot index."""
        slot = self._reserve(uid, frame, clean)
        self.buf[slot].copy_(crop)
        return slot

    def stash_batch(
        self,
        uids: list[int],
        frames: list[int],
        crops: Tensor,
        *,
        clean: list[bool] | None = None,
    ) -> list[int]:
        """Stash ``[N, 3, H, W]`` crops in one ``index_copy_`` (side-stream safe)."""
        n = int(crops.shape[0])
        if n == 0:
            return []
        clean_flags = [True] * n if clean is None else clean
        slots = [self._reserve(uids[i], frames[i], clean_flags[i]) for i in range(n)]
        idx = torch.as_tensor(slots, device=self.buf.device, dtype=torch.long)
        self.buf.index_copy_(0, idx, crops.to(self.buf.dtype))
        return slots

    def gather(
        self,
        uid: int,
        *,
        clean_only: bool = False,
        most_recent: int | None = None,
    ) -> tuple[Tensor, list[int], list[bool]] | None:
        """Recent-first-order crops for ``uid`` (``None`` if it has none)."""
        dq = self._uid_slots.get(uid)
        if not dq:
            return None
        slots = list(dq)  # oldest .. newest
        if clean_only:
            slots = [s for s in slots if self._slot_clean[s]]
        if most_recent is not None:
            slots = slots[-most_recent:]
        if not slots:
            return None
        idx = torch.as_tensor(slots, device=self.buf.device, dtype=torch.long)
        crops = self.buf.index_select(0, idx)
        frames = [self._slot_frame[s] for s in slots]
        clean_mask = [self._slot_clean[s] for s in slots]
        return crops, frames, clean_mask

    def evict(self, uid: int) -> None:
        """Drop all of ``uid``'s stashed crops (e.g. when its archive expires)."""
        dq = self._uid_slots.pop(uid, None)
        if not dq:
            return
        for s in dq:
            self._lru.pop(s, None)
            self._slot_uid[s] = -1
            self._free.append(s)

    def has(self, uid: int) -> bool:
        return bool(self._uid_slots.get(uid))

    def __len__(self) -> int:
        return self.capacity - len(self._free)


class EmbeddingRing:
    """Per-uid recent-tail embedding FIFO with head-window accumulation.

    Opt #5 (done safely): a track's clean crops are embedded **once** as they
    arrive and appended to a bounded recent-tail deque — new frames never
    trigger a whole-track re-extraction, and the tail is never collapsed to a
    single mean vector (the validated recent-N sample bank, not a prototype).

    * ``bank(uid)`` — the sparse operating-point bank: a strided view of the
      recent tail (``bank_n`` samples at ``bank_stride``).
    * ``dense(uid)`` — the dense re-query bank: the untouched recent tail.
    * ``head(uid)`` — the newborn's first-``decide_n``-frame evidence, frozen
      at the decision so later frames cannot contaminate it.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        tail_n: int = 40,
        head_n: int = 5,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.tail_n = int(tail_n)
        self.head_n = int(head_n)
        self._tail: dict[int, deque[Tensor]] = {}
        self._head: dict[int, list[Tensor]] = {}
        self._head_frozen: set[int] = set()

    def append_tail(self, uid: int, emb: Tensor) -> None:
        """Append one ``[D]`` (or ``[1, D]``) embedding to ``uid``'s tail."""
        dq = self._tail.get(uid)
        if dq is None:
            dq = deque(maxlen=self.tail_n)
            self._tail[uid] = dq
        dq.append(emb.reshape(-1))

    def append_head(self, uid: int, emb: Tensor) -> None:
        """Append a head-window sample unless the head is already frozen/full."""
        if uid in self._head_frozen:
            return
        buf = self._head.setdefault(uid, [])
        if len(buf) < self.head_n:
            buf.append(emb.reshape(-1))

    def freeze_head(self, uid: int) -> None:
        self._head_frozen.add(uid)

    def head(self, uid: int) -> Tensor | None:
        buf = self._head.get(uid)
        if not buf:
            return None
        return torch.stack(buf, dim=0)

    def bank(self, uid: int, *, bank_n: int, bank_stride: int = 1) -> Tensor | None:
        dq = self._tail.get(uid)
        if not dq:
            return None
        tail = list(dq)
        if bank_stride > 1:
            tail = tail[::bank_stride]
        tail = tail[-bank_n:] if bank_n > 0 else tail
        if not tail:
            return None
        return torch.stack(tail, dim=0)

    def dense(self, uid: int, *, n: int) -> Tensor | None:
        dq = self._tail.get(uid)
        if not dq:
            return None
        tail = list(dq)[-n:] if n > 0 else list(dq)
        if not tail:
            return None
        return torch.stack(tail, dim=0)

    def tail_uids(self) -> list[int]:
        """Uids that currently hold a recent-tail bank (candidate banks)."""
        return list(self._tail.keys())

    def evict(self, uid: int) -> None:
        self._tail.pop(uid, None)
        self._head.pop(uid, None)
        self._head_frozen.discard(uid)


class LiveEvfifoHandover:
    """Live evfifo-5-20-w3 bank feeding the validated causal handover.

    The offline handover re-extracts a clean recent-N bank from disk at
    end-of-sequence, keyed off the final MOT lines. This class reproduces that
    exactly with bounded VRAM and no disk re-read: during tracking it caches
    each detected box's embedding keyed by the id-independent ``(frame, box)``
    invariant, and at sequence end it runs the **same** offline planner
    (:func:`plan_clean_fifo_crops`, evfifo-5-20-w3) on the *final* lines and
    looks each planned crop's embedding up from the cache.

    Keying by ``(frame, box)`` — not the live track id — is essential: the
    tracker's bridge relink remaps ids mid-sequence, so a bank keyed by the
    during-tracking id would not match the final post-remap tracklets the
    decision scores. The box is unchanged by a relabel, so the lookup is
    remap-proof and the result is bit-identical to the offline decision.
    """

    def __init__(
        self,
        *,
        fifo_n: int = 20,
        stride: int = 5,
        decide_n: int = 5,
        preocc_window: int = 3,
        appearance_occlusion_cov: float = 0.4,
        neighbor_iou_max: float = 0.0,
        postocc_snapshot: bool = True,
    ) -> None:
        self.fifo_n = int(fifo_n)
        self.stride = int(stride)
        self.decide_n = int(decide_n)
        self.preocc_window = max(1, int(preocc_window))
        self.appearance_occlusion_cov = float(appearance_occlusion_cov)
        self.neighbor_iou_max = float(neighbor_iou_max)
        self.postocc_snapshot = bool(postocc_snapshot)
        # (frame, x, y, w, h) → L2-normalizable embedding, one per box.
        self._cache: dict[tuple[int, int, int, int, int], Tensor] = {}
        # (frame, x, y, w, h) → during-tracking uid (for ring lookup).
        self._box_to_uid: dict[tuple[int, int, int, int, int], int] = {}

    @staticmethod
    def _key(frame: int, box: Any) -> tuple[int, int, int, int, int]:
        # Must quantize identically to the finalize-side lookup, which keys on
        # values parsed back from .2f-serialized MOT lines.
        return _cache_key(frame, box)

    def observe_frame(
        self, frame_id: int, ids: list[int], boxes: Any, embs: Tensor
    ) -> None:
        """Cache this frame's per-track embeddings keyed by (frame, box).

        Also records the (frame, box) → uid mapping for ring-based extraction at
        finalize (uid = during-tracking id, invariant under post_merge remap).
        """
        if len(ids) == 0:
            return
        embs_f = embs.detach().float()
        # One reduction for all rows; a per-row emb.dot(emb) is ~40 torch
        # dispatches per frame.
        norms = torch.einsum("ij,ij->i", embs_f, embs_f).tolist()
        for i, tid in enumerate(ids):
            if norms[i] < 1e-8:
                continue
            k = self._key(frame_id, boxes[i])
            self._cache[k] = embs_f[i]
            self._box_to_uid[k] = tid

    def uid_for_box(self, frame: int, box: Any) -> int | None:
        return self._box_to_uid.get(self._key(frame, box))

    def build_embeddings(
        self,
        results_lines: list[str],
        *,
        extractor: Any = None,
        bank_mode: str = "spread",
        bank_n: int = 0,
        decide_n: int = 5,
        crop_ring: Any = None,
        alias: dict[int, int] | None = None,
        stream_ptr: int = 0,
        **kwargs: Any,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Build head/bank dicts via the same planner as offline.

        When ``crop_ring`` (a :class:`PerceptionPipeline`) is provided, raw
        crops are gathered by (uid, frame) from its GPU ring and batch-extracted
        via TRT — this covers coasting tracks the embedding cache misses.
        Otherwise falls back to the live embedding cache.
        """
        from .cheb_gr_online import (
            _build_pool_from_plan,
            _build_pool_spread,
            _cache_key,
            extract_handover_embeddings,
        )
        from .clean_fifo_bank import plan_clean_fifo_crops
        from .post_merge import _parse_mot_lines

        def _cache_fallback() -> tuple[dict[int, Tensor], dict[int, Tensor]]:
            return extract_handover_embeddings(
                results_lines,
                seq_dir="",
                extractor=extractor,
                decide_n=decide_n,
                bank_mode=bank_mode,
                bank_n=bank_n,
                embedding_cache=self._cache,
                **kwargs,
            )

        if crop_ring is None or not hasattr(crop_ring, "gather_crops_framed"):
            return _cache_fallback()

        # ── ring-based extraction ────────────────────────────────────────
        n_samples = kwargs.get("n_samples", 50)
        appearance_occlusion_cov = kwargs.get("appearance_occlusion_cov", 0.4)
        neighbor_iou_max = kwargs.get("neighbor_iou_max", 0.0)
        batch = kwargs.get("batch", 256)
        crop_hw = kwargs.get("crop_hw") or tuple(
            getattr(extractor, "input_hw", (224, 224))
        )

        records = _parse_mot_lines(results_lines)
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

        # Build (uid, frame) pairs. Pre-filter: only gather crops that are
        # actually in the ring (coasting frames have no detection → no crop).
        n_pool = len(pool)
        pool_uids = np.zeros(n_pool, dtype=np.uint64)
        pool_frames = np.zeros(n_pool, dtype=np.int32)
        found_mask = np.zeros(n_pool, dtype=bool)
        for si, (tid, fr, box_xyxy) in enumerate(pool):
            # Resolve the ring uid from the (frame, box) index — id-remap-proof.
            uid = self._box_to_uid.get(
                _cache_key(fr, box_xyxy),
                alias.get(tid, tid) if alias else tid,
            )
            pool_uids[si] = uid
            pool_frames[si] = fr
            found_mask[si] = crop_ring.has_crop(int(uid), int(fr), clean_only=True)

        found_idx = np.where(found_mask)[0]
        n_found = len(found_idx)

        dim = crop_ring.embed_dim  # pybind property → get_embed_dim()
        feats = torch.zeros((n_pool, dim), device="cuda", dtype=torch.float32)

        if n_found > 0:
            ring_h, ring_w = crop_hw
            batch_buf = torch.empty(
                (n_found, 3, ring_h, ring_w),
                device="cuda",
                dtype=torch.float32,
            )
            # Dense (uid, frame) arrays for found entries only.
            found_uids = pool_uids[found_idx]
            found_frames = pool_frames[found_idx]
            n_gathered = crop_ring.gather_crops_framed(
                found_uids.ctypes.data,
                found_frames.ctypes.data,
                n_found,
                batch_buf.data_ptr(),
                stream_ptr,
                True,  # clean_only — mirror the has_crop pre-filter above
            )
            if n_gathered != n_found:
                # gather_crops_framed compacts skipped pairs, so any miss
                # shifts every later row onto the wrong pool entry. Bail to
                # the cache path rather than score misaligned banks.
                return _cache_fallback()

            # TRT extract in batches.
            for s in range(0, n_gathered, max(1, batch)):
                e = min(s + batch, n_gathered)
                feats_chunk = extractor.extract(batch_buf[s:e])
                # Map batch rows → pool indices.
                for bi in range(e - s):
                    pi = found_idx[s + bi]
                    feats[pi] = feats_chunk[bi]

        # Normalize non-zero rows.
        nonzero = feats.norm(dim=1) > 1e-8
        if nonzero.any():
            feats[nonzero] = torch.nn.functional.normalize(feats[nonzero], dim=1)

        def _gather(rows: dict[int, list[int]]) -> dict[int, Tensor]:
            # Drop zero rows (ring misses): zero vectors are mutual nearest
            # neighbours in the k-reciprocal graph and must not be scored.
            out: dict[int, Tensor] = {}
            for tid, idxs in rows.items():
                if idxs:
                    t = feats[torch.tensor(idxs, device=feats.device)]
                    keep = t.norm(dim=1) > 1e-8
                    if keep.any():
                        out[tid] = t[keep]
            return out

        return _gather(head_rows), _gather(bank_rows)


@dataclass
class _StreamState:
    lines: list[str] = field(default_factory=list)
    alias: dict[int, int] = field(default_factory=dict)
    # newborns that reached decide_n and have been through a pass already.
    resolved: set[int] = field(default_factory=set)
    # newborns that matured (>= decide_n frames seen) but not yet passed.
    pending_newborns: set[int] = field(default_factory=set)
    last_pass_frame: int = -1


class StreamingHandoverDriver:
    """Drive causal handover live, gating *when* the decision runs.

    The driver owns no CUDA/TRT: crop harvesting and embedding extraction are
    the caller's job (side-stream), fed in through :meth:`observe_frame` /
    :meth:`note_head` / :meth:`note_tail`. Its only decision surface is
    :func:`causal_handover_lines`, run over the buffered trailing window when
    the lazy/adaptive gate fires. The resulting relabels are folded into a
    monotonic :attr:`alias` map that emission applies going forward — already
    emitted frames are never rewritten (the confirmation-window decision is
    causal by construction).

    Args:
        embeddings: the :class:`EmbeddingRing` the caller appends to.
        decide_n: frames after birth at which a newborn's identity is decided.
        handover_kwargs: forwarded verbatim to :func:`causal_handover_lines`
            (``max_cost``, ``margin``, ``max_gap``, ``pool_frac`` ...), so the
            frozen operating point is reused unchanged.
        bank_n / bank_stride: sparse operating-point bank shape from the tail.
        requery_ring_n: dense recent-tail depth for borderline re-query
            (0 disables the dense fallback).
        trigger_newborns: opt #6 — run a pass once this many unresolved
            newborns have matured (1 = eager, larger = coarser/cheaper).
        max_pass_gap: safety cadence — force a pass at least this often even
            below the newborn trigger (0 = never force on cadence).
        async_pass: opt #3 — run the (CPU/GPU) decision on a background thread
            so the frame loop is not blocked; results merge atomically.
    """

    def __init__(
        self,
        *,
        embeddings: EmbeddingRing,
        decide_n: int = 5,
        handover_kwargs: dict[str, Any] | None = None,
        bank_n: int = 20,
        bank_stride: int = 1,
        requery_ring_n: int = 0,
        trigger_newborns: int = 1,
        max_pass_gap: int = 0,
        async_pass: bool = False,
    ) -> None:
        self.emb = embeddings
        self.decide_n = int(decide_n)
        self.handover_kwargs: dict[str, Any] = dict(handover_kwargs or {})
        self.bank_n = int(bank_n)
        self.bank_stride = int(bank_stride)
        self.requery_ring_n = int(requery_ring_n)
        self.trigger_newborns = max(1, int(trigger_newborns))
        self.max_pass_gap = int(max_pass_gap)
        self.async_pass = bool(async_pass)

        self._state = _StreamState()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._birth_frame: dict[int, int] = {}
        self._alive: set[int] = set()
        self.stats: dict[str, int] = {"passes": 0, "passes_skipped": 0, "handovers": 0}

    # -- ingestion ---------------------------------------------------------
    def observe_frame(
        self, frame_id: int, lines: list[str], active_ids: list[int]
    ) -> None:
        """Record this frame's emitted MOT lines and which track ids are alive.

        A track id seen for the first time is a birth; ids that disappear are
        deaths (their tail is now a causal candidate bank). Maturing a newborn
        past ``decide_n`` frames makes it eligible for the next pass — that is
        the signal the lazy gate (#1) keys on.
        """
        with self._lock:
            self._state.lines.extend(lines)
            seen = set(active_ids)
            for tid in seen:
                if tid not in self._birth_frame:
                    self._birth_frame[tid] = frame_id
            # Maturity: a newborn that has lived >= decide_n frames and has not
            # yet been decided becomes a pending trigger.
            for tid in seen:
                if (
                    tid not in self._state.resolved
                    and frame_id - self._birth_frame[tid] >= self.decide_n
                ):
                    self._state.pending_newborns.add(tid)
            self._alive = seen

    def note_head(self, uid: int, emb: Tensor) -> None:
        self.emb.append_head(uid, emb)

    def note_tail(self, uid: int, emb: Tensor) -> None:
        self.emb.append_tail(uid, emb)

    # -- gating ------------------------------------------------------------
    def should_run(self, frame_id: int, *, force: bool = False) -> bool:
        """Opt #1 + #6: only run when work exists (or the safety cadence fires)."""
        with self._lock:
            n_pending = len(self._state.pending_newborns)
            if n_pending == 0:
                return force
            if force or n_pending >= self.trigger_newborns:
                return True
            if (
                self.max_pass_gap > 0
                and frame_id - self._state.last_pass_frame >= self.max_pass_gap
            ):
                return True
            return False

    # -- the pass ----------------------------------------------------------
    def _build_embeddings(
        self,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
        head_embs: dict[int, Tensor] = {}
        bank_embs: dict[int, Tensor] = {}
        requery_embs: dict[int, Tensor] = {}
        with self._lock:
            pending = set(self._state.pending_newborns)
        for uid in pending:
            h = self.emb.head(uid)
            if h is not None:
                head_embs[uid] = h
        # Every id that has a tail is a potential dead candidate.
        for uid in self.emb.tail_uids():
            b = self.emb.bank(uid, bank_n=self.bank_n, bank_stride=self.bank_stride)
            if b is not None:
                bank_embs[uid] = b
            if self.requery_ring_n > 0:
                d = self.emb.dense(uid, n=self.requery_ring_n)
                if d is not None:
                    requery_embs[uid] = d
        return head_embs, bank_embs, requery_embs

    def _run_locked_inputs(
        self,
    ) -> tuple[
        list[str],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        list[int],
    ]:
        with self._lock:
            lines = list(self._state.lines)
            pending = list(self._state.pending_newborns)
        head, bank, requery = self._build_embeddings()
        return lines, head, bank, requery, pending

    def _do_pass(self, frame_id: int) -> None:
        lines, head, bank, requery, pending = self._run_locked_inputs()
        if not lines or not head:
            return
        kwargs = dict(self.handover_kwargs)
        kwargs.setdefault("decide_n", self.decide_n)
        if self.requery_ring_n > 0 and requery:
            kwargs.setdefault("requery_bank_embs", requery)

        # One decision-core call yields both the accepted handovers (via the
        # decision log) and the aggregate stats — the alias is therefore
        # bit-identical to what the offline end-of-sequence pass produces on the
        # same inputs.
        log: list[dict[str, Any]] = []
        _, stats = causal_handover_lines(
            lines, head, bank, enabled=True, decision_log=log, **kwargs
        )
        alias = {
            int(r["newborn_id"]): int(r["candidate_label"])
            for r in log
            if r.get("accepted")
        }
        with self._lock:
            for src, dst in alias.items():
                # follow existing chains so aliases stay monotonic
                self._state.alias[src] = self._state.alias.get(dst, dst)
            self._state.resolved.update(pending)
            self._state.pending_newborns.difference_update(pending)
            self._state.last_pass_frame = frame_id
            self.stats["passes"] += 1
            self.stats["handovers"] += int(stats.get("handovers", 0))

    def run(self, frame_id: int, *, force: bool = False) -> bool:
        """Run a pass if the gate permits. Returns whether a pass was launched."""
        if not self.should_run(frame_id, force=force):
            self.stats["passes_skipped"] += 1
            return False
        if self.async_pass:
            self._join_worker()
            self._worker = threading.Thread(
                target=self._do_pass, args=(frame_id,), daemon=True
            )
            self._worker.start()
        else:
            self._do_pass(frame_id)
        return True

    def _join_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.join()
        self._worker = None

    def finalize(self, frame_id: int) -> dict[int, int]:
        """Force a final pass, join any async worker, return the alias map."""
        self._do_pass(frame_id)
        self._join_worker()
        with self._lock:
            return dict(self._state.alias)

    def resolve(self, track_id: int) -> int:
        with self._lock:
            return self._state.alias.get(track_id, track_id)
