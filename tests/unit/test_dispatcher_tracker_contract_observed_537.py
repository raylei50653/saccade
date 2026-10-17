"""#537 C0-11: pin the tracker surfaces that AsyncDispatcher used to call.

Characterization only. Pinned on ``main@82126e7c``, then trimmed by the R-1 and
#538 D2-AD fixes.

Before R-1, ``_process_batch()`` made two calls. At FAST_PATH and above it called
``tracker.set_degradation_params(level)``. Below FAST_PATH it called
``tracker.update(..., frame_tensor=..., stream_id=...)``. It unpacked a 3-tuple
either way. Neither the native binding nor the Python facade supports those
calls. The surface checks below still hold, because R-1 added
``OnlineTrackerAdapter`` and left both surfaces unchanged.

Moved out of this file: the worker-death cases went to
``test_dispatcher_tracker_contract_537.py`` (R-1), and the dead-worker
observation went to ``test_dispatcher_failure_contract_538.py`` (D2-AD).
Evidence from before each fix is in failure_degradation_537.md §9/§10/§11 and in
PRs #600/#601.
"""

# scope: perception, resource
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import inspect

import pytest
import torch

from saccade.perception.tracking import tracker_gpu as facade
from saccade.resource.resource_manager import DegradationLevel

_HW = (8, 8)


# ── Tracker surfaces (no dispatcher) ────────────────────────────────────


def test_facade_does_not_support_dispatcher_calls() -> None:
    sig = inspect.signature(facade.GPUByteTracker.update)
    b, s, c = torch.zeros(1, 4), torch.zeros(1), torch.zeros(1, dtype=torch.int32)
    sig.bind(None, b, s, c)  # the tensor form itself is accepted
    with pytest.raises(TypeError, match="frame_tensor"):
        sig.bind(None, b, s, c, frame_tensor=torch.zeros(3, *_HW), stream_id=1)
    assert not hasattr(facade.GPUByteTracker, "set_degradation_params")
    assert "TrackResult" in str(sig.return_annotation)


def test_native_binding_has_no_degradation_api() -> None:
    ext = pytest.importorskip("saccade_tracking_ext")
    assert not hasattr(ext.GPUByteTracker, "set_degradation_params")
    first = (ext.GPUByteTracker.update.__doc__ or "").splitlines()[0]
    assert "boxes_ptr" in first and "frame_tensor" not in first
    assert first.endswith("-> list[saccade_tracking_ext.TrackResult]")


@pytest.mark.gpu
def test_native_binding_rejects_dispatcher_calls() -> None:
    ext = pytest.importorskip("saccade_tracking_ext")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to construct the native tracker")
    t = ext.GPUByteTracker(max_objects=16, embedding_dim=0)
    b = torch.tensor([[10.0, 10.0, 50.0, 80.0]], device="cuda")
    s = torch.tensor([0.9], device="cuda")
    c = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(AttributeError, match="set_degradation_params"):
        t.set_degradation_params(DegradationLevel.FAST_PATH)
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.update(b, s, c)
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.update(b, s, c, frame_tensor=torch.zeros(3, 8, 8, device="cuda"), stream_id=1)
