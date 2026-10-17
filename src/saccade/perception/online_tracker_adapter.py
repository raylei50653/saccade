"""Geometry-only tracker adapter for the online dispatcher (#537 R-1).

``AsyncDispatcher`` needs ``update(boxes, scores, classes) -> (ids, boxes, classes)``.
This adapter provides that on top of the Python ``GPUByteTracker`` facade, which
owns the raw-pointer call into the native binding. Scope is geometry only: no
embeddings, no ReID, and no tracker-side degradation parameters.

A missing native extension or CUDA device raises ``OnlineTrackerUnavailable``
instead of falling back to the facade's stub, whose empty results would look
like legitimate empty frames.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from saccade.perception.tracking import tracker_gpu as _facade


class OnlineTrackerUnavailable(RuntimeError):
    """The native tracker cannot run here; the online tracking path refuses."""


def native_unavailable_reason() -> Optional[str]:
    """Why the native tracker cannot run here, or ``None`` when it can."""
    try:
        import saccade_tracking_ext
    except ImportError as e:
        return f"saccade_tracking_ext is not importable ({e})"
    if (
        getattr(_facade, "CppGPUByteTracker", None)
        is not saccade_tracking_ext.GPUByteTracker
    ):
        return "the tracker facade is bound to its stub, not saccade_tracking_ext"
    if not torch.cuda.is_available():
        return "no CUDA device is available"
    return None


class OnlineTrackerAdapter:
    """``update(boxes, scores, classes)`` → ``(ids, boxes, classes)`` tensors.

    ``_tracker`` replaces the facade in host-only tests. It also skips the
    native and CUDA checks, which hold only for the facade built here.
    """

    def __init__(self, max_objects: int = 100, *, _tracker: Any = None) -> None:
        self._require_cuda = _tracker is None
        if _tracker is None:
            reason = native_unavailable_reason()
            if reason is not None:
                raise OnlineTrackerUnavailable(reason)
            _tracker = _facade.GPUByteTracker(max_objects=max_objects, embedding_dim=0)
            if not _tracker.is_cuda:
                raise OnlineTrackerUnavailable("the tracker facade is not on CUDA")
        self._tracker = _tracker

    def update(
        self, boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = boxes.device
        if self._require_cuda and not all(t.is_cuda for t in (boxes, scores, classes)):
            # The facade hands data_ptr() to CUDA kernels; a host pointer there
            # is undefined behavior, not an error.
            raise ValueError("OnlineTrackerAdapter.update() needs CUDA tensors")
        results = self._tracker.update(boxes, scores, classes)
        ids = torch.tensor([r.obj_id for r in results], dtype=torch.int32)
        out_boxes = torch.tensor(
            [[r.x1, r.y1, r.x2, r.y2] for r in results], dtype=torch.float32
        ).reshape(-1, 4)
        out_classes = torch.tensor([r.class_id for r in results], dtype=torch.int32)
        return ids.to(device), out_boxes.to(device), out_classes.to(device)
