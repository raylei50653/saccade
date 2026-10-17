"""#537 R-1: the AsyncDispatcher↔tracker contract after the adapter fix.

Owner decisions (2026-10-17):

- The dispatcher reaches the native tracker only through ``OnlineTrackerAdapter``,
  which wraps the Python facade. The adapter takes ``update(boxes, scores, classes)``
  and returns ``(ids, boxes, classes)`` tensors.
- Tracking is geometry only. A missing native extension or CUDA device is refused
  with ``OnlineTrackerUnavailable``. An extractor (ReID) is refused.
- FAST_PATH and EMERGENCY only cap the batch at ``min(max_batch, 2)``. The tracker
  is not degraded, and ``get_stats()`` says so.

The host-only tests use a strict fake: ``create_autospec`` of the real facade
class. Unknown methods or keywords raise on it as they do on the facade, so the
calls that the pre-fix dispatcher made fail on it. Worker failure semantics are
#538 D2-AD's (test_dispatcher_failure_contract_538.py).
"""

# scope: perception, resource
# function: contract
# lifecycle: active

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch

from saccade.perception import dispatcher as dmod
from saccade.perception.online_tracker_adapter import (
    OnlineTrackerAdapter,
    OnlineTrackerUnavailable,
)
from saccade.perception.tracking import tracker_gpu as facade
from saccade.resource.resource_manager import DegradationLevel, ResourceManager

REPO = Path(__file__).resolve().parents[2]
_HW = (8, 8)
_LEVELS = list(DegradationLevel)


def _track(obj_id: int, box: list[float], class_id: int = 0) -> SimpleNamespace:
    # The fields that saccade_tracking_ext.TrackResult exposes.
    x1, y1, x2, y2 = box
    return SimpleNamespace(
        obj_id=obj_id,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=0.9,
        class_id=class_id,
        det_idx=0,
    )


def _strict_facade(results: list[Any] | None = None) -> MagicMock:
    fake = create_autospec(facade.GPUByteTracker, instance=True)
    fake.update.return_value = results if results is not None else []
    return fake


def _detector(n_boxes: int = 1) -> MagicMock:
    det = MagicMock()
    det.detect_batch.side_effect = lambda x, conf_threshold: [
        (
            torch.tensor([[10.0, 10.0, 50.0, 50.0]] * n_boxes).reshape(n_boxes, 4),
            torch.full((n_boxes,), 0.9),
            torch.zeros(n_boxes, dtype=torch.int32),
            None,
        )
        for _ in range(x.shape[0])
    ]
    return det


def _dispatcher(
    level: DegradationLevel,
    *,
    n_boxes: int = 1,
    max_batch: int = 4,
    facade_factory: Any = None,
    on_track_result: Any = None,
) -> dmod.AsyncDispatcher:
    rm = MagicMock(spec=ResourceManager)
    rm.decide_degradation_level.return_value = level
    disp = dmod.AsyncDispatcher(
        _detector(n_boxes),
        rm,
        max_batch=max_batch,
        input_hw=_HW,
        on_track_result=on_track_result,
    )
    make = facade_factory or (lambda: _strict_facade([_track(7, [10, 10, 50, 50])]))
    disp._make_tracker = lambda: OnlineTrackerAdapter(_tracker=make())  # type: ignore[method-assign]
    return disp


# ── Adapter ─────────────────────────────────────────────────────────────


def test_adapter_converts_track_results_to_tensors() -> None:
    fake = _strict_facade([_track(3, [1, 2, 3, 4], 0), _track(9, [5, 6, 7, 8], 2)])
    adapter = OnlineTrackerAdapter(_tracker=fake)
    b, s, c = torch.zeros(2, 4), torch.ones(2), torch.zeros(2, dtype=torch.int32)
    ids, boxes, classes = adapter.update(b, s, c)
    fake.update.assert_called_once_with(b, s, c)
    assert ids.dtype == torch.int32 and ids.tolist() == [3, 9]
    assert boxes.dtype == torch.float32 and boxes.tolist() == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    assert classes.dtype == torch.int32 and classes.tolist() == [0, 2]


def test_adapter_empty_result_shapes() -> None:
    adapter = OnlineTrackerAdapter(_tracker=_strict_facade([]))
    ids, boxes, classes = adapter.update(
        torch.zeros(1, 4), torch.ones(1), torch.zeros(1, dtype=torch.int32)
    )
    assert (ids.shape, boxes.shape, classes.shape) == ((0,), (0, 4), (0,))


def test_adapter_has_no_degradation_api() -> None:
    assert not hasattr(OnlineTrackerAdapter, "set_degradation_params")


def test_strict_facade_rejects_the_pre_fix_calls() -> None:
    """Negative control: the fake fails the calls the pre-fix dispatcher made."""
    fake = _strict_facade()
    b, s, c = torch.zeros(1, 4), torch.ones(1), torch.zeros(1, dtype=torch.int32)
    with pytest.raises(AttributeError, match="set_degradation_params"):
        fake.set_degradation_params(DegradationLevel.FAST_PATH)
    with pytest.raises(TypeError):
        fake.update(b, s, c, frame_tensor=torch.zeros(3, *_HW), stream_id=1)


# ── Dispatcher with the adapter ──────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("level", _LEVELS, ids=[lv.name for lv in _LEVELS])
async def test_detections_are_tracked_at_every_level(level: DegradationLevel) -> None:
    seen: list[tuple[Any, ...]] = []

    async def on_result(sid, ts, ids, boxes, classes, kpts) -> None:  # type: ignore[no-untyped-def]
        seen.append((sid, ts, ids, boxes, classes, kpts))

    with (
        patch.object(dmod, "native_unavailable_reason", return_value=None),
        patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()),
    ):
        disp = _dispatcher(level, max_batch=1, on_track_result=on_result)
        await disp.start()
        for i in range(3):
            await disp.put_frame("cam0", torch.rand(3, *_HW), float(i))
        await asyncio.wait_for(disp.queue.join(), timeout=2.0)
        assert disp._worker_task is not None and not disp._worker_task.done()
        stats = disp.get_stats()
        await disp.stop()

    assert [s[1] for s in seen] == [0.0, 1.0, 2.0]
    for _, _, ids, boxes, classes, kpts in seen:
        assert ids.tolist() == [7] and boxes.tolist() == [[10, 10, 50, 50]]
        assert classes.tolist() == [0] and kpts is None
    assert stats["current_level"] == level.name
    assert stats["tracker_degradation"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "max_batch", "cap"),
    [
        (DegradationLevel.NORMAL, 4, 4),
        (DegradationLevel.REDUCED, 4, 4),
        (DegradationLevel.FAST_PATH, 4, 2),
        (DegradationLevel.EMERGENCY, 4, 2),
        (DegradationLevel.FAST_PATH, 1, 1),
        (DegradationLevel.EMERGENCY, 1, 1),
    ],
)
async def test_degradation_only_caps_the_batch(
    level: DegradationLevel, max_batch: int, cap: int
) -> None:
    with patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()):
        disp = _dispatcher(level, max_batch=max_batch)
        disp.batch_timeout_ms = 200.0
        for i in range(8):
            disp.queue.put_nowait(
                dmod._QueuedFrame("cam0", torch.rand(3, *_HW), float(i), 0.0)
            )
        disp._running = True
        task = asyncio.create_task(disp._worker_loop())
        await asyncio.wait_for(disp.queue.join(), timeout=5.0)
        disp._running = False
        await asyncio.wait_for(task, timeout=2.0)
    sizes = [c.args[0].shape[0] for c in disp.detector.detect_batch.call_args_list]
    assert max(sizes) == cap and sum(sizes) == 8


@pytest.mark.asyncio
async def test_empty_frames_do_not_create_trackers() -> None:
    made: list[MagicMock] = []

    def factory() -> MagicMock:
        made.append(_strict_facade())
        return made[-1]

    with (
        patch.object(dmod, "native_unavailable_reason", return_value=None),
        patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()),
    ):
        disp = _dispatcher(
            DegradationLevel.FAST_PATH, n_boxes=0, facade_factory=factory
        )
        await disp.start()
        for i in range(3):
            await disp.put_frame("cam0", torch.rand(3, *_HW), float(i))
        await asyncio.wait_for(disp.queue.join(), timeout=2.0)
        await disp.stop()
    assert made == [] and disp.trackers == {}


def test_extractor_is_refused() -> None:
    with pytest.raises(ValueError, match="geometry-only"):
        dmod.AsyncDispatcher(MagicMock(), MagicMock(), extractor=MagicMock())


@pytest.mark.asyncio
async def test_tracker_errors_are_not_swallowed() -> None:
    """The adapter passes tracker errors through; #538 D2-AD defines the FAILED state."""
    fake = _strict_facade()
    fake.update.side_effect = RuntimeError("native tracker failed")
    with (
        patch.object(dmod, "native_unavailable_reason", return_value=None),
        patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()),
    ):
        disp = _dispatcher(DegradationLevel.NORMAL, facade_factory=lambda: fake)
        await disp.start()
        task = disp._worker_task
        assert task is not None
        await disp.put_frame("cam0", torch.rand(3, *_HW), 0.0)
        await asyncio.wait_for(asyncio.shield(asyncio.wait({task})), timeout=2.0)
        assert disp.failure is fake.update.side_effect
        with pytest.raises(dmod.DispatcherFailed):
            await disp.stop()


# ── Native unavailable (fresh interpreter, extension blocked) ────────────

_BLOCKED = """
import sys, asyncio
sys.modules['saccade_tracking_ext'] = None
from unittest import mock
import saccade.perception.dispatcher as d
from saccade.perception.online_tracker_adapter import OnlineTrackerAdapter
disp = d.AsyncDispatcher(detector=mock.MagicMock(), resource_manager=mock.MagicMock())
for name, fn in (
    ('start', lambda: asyncio.run(disp.start())),
    ('get_tracker', lambda: disp.get_tracker('cam0')),
    ('adapter', lambda: OnlineTrackerAdapter()),
):
    try:
        fn()
        print(name, 'ok')
    except Exception as e:
        print(name, type(e).__name__)
print('running', disp._running, disp._worker_task is None)
"""


@pytest.mark.parametrize("hide_cuda", [True, False], ids=["no_cuda", "with_cuda_env"])
def test_native_unavailable_is_refused(hide_cuda: bool) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [
        ln
        for ln in proc.stdout.splitlines()
        if ln.split(" ", 1)[0] in {"start", "get_tracker", "adapter", "running"}
    ]
    assert lines == [
        "start OnlineTrackerUnavailable",
        "get_tracker OnlineTrackerUnavailable",
        "adapter OnlineTrackerUnavailable",
        "running False True",
    ]


# ── Real native tracker (GPU) ────────────────────────────────────────────


def _need_native() -> None:
    pytest.importorskip("saccade_tracking_ext")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


@pytest.mark.gpu
def test_native_adapter_smoke() -> None:
    _need_native()
    adapter = OnlineTrackerAdapter(max_objects=100)
    dev = "cuda"
    boxes = torch.tensor([[200.0, 200.0, 260.0, 330.0]], device=dev)
    scores = torch.tensor([0.9], device=dev)
    classes = torch.tensor([0], dtype=torch.int32, device=dev)
    outs = [adapter.update(boxes + f, scores, classes) for f in range(5)]
    for ids, out_boxes, out_classes in outs:
        assert (
            ids.device.type
            == out_boxes.device.type
            == out_classes.device.type
            == "cuda"
        )
        assert (ids.dtype, out_boxes.dtype, out_classes.dtype) == (
            torch.int32,
            torch.float32,
            torch.int32,
        )
        assert out_boxes.shape == (ids.shape[0], 4) and out_classes.shape == ids.shape
    # A steady detection becomes a confirmed track (no claim on how many frames it takes).
    assert outs[-1][0].numel() == 1
    empty = adapter.update(
        torch.empty((0, 4), device=dev),
        torch.empty((0,), device=dev),
        torch.empty((0,), dtype=torch.int32, device=dev),
    )
    assert tuple(t.shape for t in empty) == ((0,), (0, 4), (0,))
    with pytest.raises(ValueError, match="CUDA tensors"):
        adapter.update(boxes.cpu(), scores.cpu(), classes.cpu())


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level",
    [DegradationLevel.NORMAL, DegradationLevel.FAST_PATH],
    ids=["NORMAL", "FAST_PATH"],
)
async def test_native_dispatcher_smoke(level: DegradationLevel) -> None:
    _need_native()
    seen: list[torch.Tensor] = []

    async def on_result(sid, ts, ids, boxes, classes, kpts) -> None:  # type: ignore[no-untyped-def]
        seen.append(ids)

    det = MagicMock()
    det.detect_batch.side_effect = lambda x, conf_threshold: [
        (
            torch.tensor([[200.0, 200.0, 260.0, 330.0]], device="cuda"),
            torch.tensor([0.9], device="cuda"),
            torch.tensor([0], dtype=torch.int32, device="cuda"),
            None,
        )
        for _ in range(x.shape[0])
    ]
    rm = MagicMock(spec=ResourceManager)
    rm.decide_degradation_level.return_value = level
    with patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()):
        disp = dmod.AsyncDispatcher(
            det, rm, max_batch=1, input_hw=_HW, on_track_result=on_result
        )
        await disp.start()
        for i in range(5):
            await disp.put_frame("cam0", torch.rand(3, *_HW, device="cuda"), float(i))
        await asyncio.wait_for(disp.queue.join(), timeout=10.0)
        assert disp._worker_task is not None and not disp._worker_task.done()
        await disp.stop()
    assert len(seen) == 5 and all(t.device.type == "cuda" for t in seen)
    assert seen[-1].numel() == 1


@pytest.mark.asyncio
async def test_start_refuses_before_any_side_effect() -> None:
    writer_cls = MagicMock()
    with (
        patch.object(dmod, "native_unavailable_reason", return_value="no native"),
        patch.object(dmod, "VRAMLevelWriter", writer_cls),
    ):
        disp = _dispatcher(DegradationLevel.NORMAL)
        with pytest.raises(OnlineTrackerUnavailable, match="no native"):
            await disp.start()
    assert disp._running is False and disp._worker_task is None
    writer_cls.assert_not_called()
