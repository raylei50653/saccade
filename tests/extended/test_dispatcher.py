"""Tests for saccade.perception.dispatcher (AsyncDispatcher).

Tests the dispatcher's stream management, batch processing logic,
and resource-aware degradation without requiring actual TensorRT models.
"""

# scope: perception
# function: behavior
# lifecycle: active

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch

from saccade.perception.dispatcher import (
    AsyncDispatcher,
    WorkbenchPool,
    _match_keypoints_to_tracks,
)
from saccade.resource.resource_manager import ResourceManager, DegradationLevel


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_mock_detector() -> MagicMock:
    """Create a mock detector that returns empty detections."""
    detector = MagicMock()
    detector.detect_batch.return_value = [
        (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int32),
            None,
        )
    ]
    detector.detect.return_value = detector.detect_batch.return_value[0]
    return detector


def _make_mock_detector_batch(batch_size: int) -> MagicMock:
    detector = MagicMock()
    detector.detect_batch.return_value = [
        (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int32),
            None,
        )
        for _ in range(batch_size)
    ]
    detector.detect.return_value = detector.detect_batch.return_value[0]
    return detector


def _make_detection_result() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32)
    scores = torch.tensor([0.9], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int32)
    keypoints = torch.randn(1, 17, 3)
    return boxes, scores, classes, keypoints


def _queued_item(
    dispatcher: AsyncDispatcher, stream_id: str, ts: float
) -> tuple[str, torch.Tensor, float]:
    return (stream_id, torch.rand(3, *dispatcher.input_hw), ts)


def _queue_payload_item(
    stream_id: str, frame: torch.Tensor, timestamp: float
) -> tuple[str, torch.Tensor, float]:
    return (stream_id, frame, timestamp)


def _queue_item_shape(item) -> tuple[str, torch.Tensor, float]:
    return item.stream_id, item.frame, item.timestamp


def _empty_track_result():
    return (
        torch.empty((0,), dtype=torch.int32),
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0,), dtype=torch.int32),
    )


def _make_mock_tracker() -> MagicMock:
    """Create a mock tracker with expected update signature."""
    tracker = MagicMock()
    tracker.update.return_value = (
        torch.empty((0,), dtype=torch.int32),
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0,), dtype=torch.int32),
    )
    tracker.set_degradation_params = MagicMock()
    return tracker


def _make_mock_resource_manager() -> MagicMock:
    rm = MagicMock(spec=ResourceManager)
    rm.decide_degradation_level.return_value = DegradationLevel.NORMAL
    return rm


# ── _match_keypoints_to_tracks ──────────────────────────────────────────


def test_match_keypoints_empty_tracked() -> None:
    tracked = torch.empty((0, 4), dtype=torch.float32)
    det = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    kps = torch.randn(1, 17, 3)
    result = _match_keypoints_to_tracks(tracked, det, kps)
    assert result.shape == (0, 17, 3)


def test_match_keypoints_single() -> None:
    """Each tracked box matches nearest detection by center distance."""
    tracked = torch.tensor(
        [[15.0, 15.0, 25.0, 25.0]], dtype=torch.float32
    )  # center (20, 20)
    det = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]], dtype=torch.float32
    )
    # det[0] center = (5, 5), det[1] center = (30, 30)
    # tracked center = (20, 20) → closer to det[1] (dist=14.14) vs det[0] (dist=21.21)
    kps = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=torch.float32
    )  # 2 kps
    result = _match_keypoints_to_tracks(tracked, det, kps)
    assert result.shape == (1, 1, 3)
    assert torch.allclose(result[0], kps[1])  # Should match det[1]


def test_match_keypoints_multiple_tracked() -> None:
    tracked = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # center (5, 5)
            [20.0, 20.0, 30.0, 30.0],  # center (25, 25)
        ],
        dtype=torch.float32,
    )
    det = torch.tensor([[5.0, 5.0, 15.0, 15.0]], dtype=torch.float32)  # center (10, 10)
    kps = torch.tensor([[[100.0, 200.0, 300.0]]], dtype=torch.float32)
    result = _match_keypoints_to_tracks(tracked, det, kps)
    assert result.shape == (2, 1, 3)
    # Both tracked boxes are closer to det[0] → both get same kps
    assert torch.allclose(result[0], kps[0])
    assert torch.allclose(result[1], kps[0])


# ── AsyncDispatcher.__init__ ───────────────────────────────────────────


def test_init_defaults() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        max_batch=4,
        conf_threshold=0.25,
    )
    assert dispatcher.detector is detector
    assert dispatcher.resource_manager is rm
    assert dispatcher.max_batch == 4
    assert dispatcher.conf_threshold == 0.25
    assert dispatcher.on_track_result is None
    assert dispatcher.extractor is None
    assert dispatcher.heartbeat_interval == 10
    assert dispatcher.max_streams == 8
    assert not dispatcher._running
    assert dispatcher.queue.maxsize == 100


def test_init_with_extractor() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    extractor = MagicMock()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        extractor=extractor,
    )
    assert dispatcher.extractor is extractor


@pytest.mark.asyncio
async def test_workbench_pool_nv12_requires_kernel() -> None:
    with patch("saccade.perception.dispatcher._nv12_letterbox_kernel", None):
        pool = WorkbenchPool(
            engine_path="dummy.engine",
            n_streams=1,
            use_nv12=True,
        )
        with pytest.raises(RuntimeError, match="nv12_to_chw_letterbox"):
            await pool.start()


# ── Tracker management ─────────────────────────────────────────────────


def test_get_tracker_creates_new() -> None:
    with patch("saccade.perception.dispatcher.GPUByteTracker") as MockTracker:
        # Side effect ensures each call returns a new instance with any args
        MockTracker.side_effect = lambda *args, **kwargs: MagicMock()
        detector = _make_mock_detector()
        rm = _make_mock_resource_manager()
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
            max_streams=2,
        )
        tracker1 = dispatcher.get_tracker("stream_a")
        tracker2 = dispatcher.get_tracker("stream_b")
        assert tracker1 is not tracker2
        assert "stream_a" in dispatcher.trackers
        assert "stream_b" in dispatcher.trackers
        assert len(dispatcher.trackers) == 2
        # Verify GPUByteTracker was called twice
        assert MockTracker.call_count == 2


def test_get_tracker_reuses_existing() -> None:
    with patch("saccade.perception.dispatcher.GPUByteTracker") as MockTracker:
        MockTracker.return_value = MagicMock()
        detector = _make_mock_detector()
        rm = _make_mock_resource_manager()
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
        )
        tracker1 = dispatcher.get_tracker("stream_a")
        tracker2 = dispatcher.get_tracker("stream_a")
        assert tracker1 is tracker2


def test_get_tracker_lru_order() -> None:
    with patch("saccade.perception.dispatcher.GPUByteTracker") as MockTracker:
        MockTracker.return_value = MagicMock()
        detector = _make_mock_detector()
        rm = _make_mock_resource_manager()
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
            max_streams=3,
        )
        _ = dispatcher.get_tracker("a")
        _ = dispatcher.get_tracker("b")
        _ = dispatcher.get_tracker("c")
        # Access "a" to move it to end
        _ = dispatcher.get_tracker("a")
        keys = list(dispatcher.trackers.keys())
        assert keys[-1] == "a"  # "a" should be last (most recent)


def test_get_tracker_evicts_lru() -> None:
    with patch("saccade.perception.dispatcher.GPUByteTracker") as MockTracker:
        MockTracker.return_value = MagicMock()
        detector = _make_mock_detector()
        rm = _make_mock_resource_manager()
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
            max_streams=2,
        )
        _ = dispatcher.get_tracker("a")
        _ = dispatcher.get_tracker("b")
        # Adding "c" should evict "a" (LRU)
        _ = dispatcher.get_tracker("c")
        assert "a" not in dispatcher.trackers
        assert "b" in dispatcher.trackers
        assert "c" in dispatcher.trackers
        assert len(dispatcher.trackers) == 2


def test_deregister_stream_existing() -> None:
    with patch("saccade.perception.dispatcher.GPUByteTracker") as MockTracker:
        MockTracker.return_value = MagicMock()
        detector = _make_mock_detector()
        rm = _make_mock_resource_manager()
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
        )
        _ = dispatcher.get_tracker("stream_a")
        assert len(dispatcher.trackers) == 1
        dispatcher.deregister_stream("stream_a")
        assert "stream_a" not in dispatcher.trackers
        assert len(dispatcher.trackers) == 0


def test_deregister_stream_nonexistent() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
    )
    # Should not raise
    dispatcher.deregister_stream("nonexistent")
    assert len(dispatcher.trackers) == 0


# ── Queue operations ────────────────────────────────────────────────────


def test_put_frame() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
    )
    frame = torch.rand(3, 480, 640)
    # Synchronous put (non-async context)
    dispatcher.queue.put_nowait(
        dispatcher._normalize_batch_item(_queue_payload_item("stream_a", frame, 1.0))
    )
    assert not dispatcher.queue.empty()
    item = dispatcher.queue.get_nowait()
    stream_id, queued_frame, timestamp = _queue_item_shape(item)
    assert stream_id == "stream_a"
    assert torch.equal(queued_frame, frame)
    assert timestamp == 1.0


def test_put_frame_queue_full() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
    )
    # Fill queue to capacity
    for i in range(100):
        dispatcher.queue.put_nowait(
            dispatcher._normalize_batch_item(
                _queue_payload_item("s", torch.rand(3, 10, 10), float(i))
            )
        )
    # Next put should raise QueueFull (caught silently in async path)
    with pytest.raises(asyncio.QueueFull):
        dispatcher.queue.put_nowait(
            dispatcher._normalize_batch_item(
                _queue_payload_item("s", torch.rand(3, 10, 10), 999.0)
            )
        )


# ── Async lifecycle ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_and_stop() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        heartbeat_interval=5,
    )
    await dispatcher.start()
    assert dispatcher._running
    await dispatcher.stop()
    assert not dispatcher._running
    assert len(dispatcher.trackers) == 0


@pytest.mark.asyncio
async def test_put_frame_async() -> None:
    with patch("saccade.perception.dispatcher.GPUByteTracker") as MockTracker:
        MockTracker.return_value = MagicMock()
        detector = _make_mock_detector()
        rm = _make_mock_resource_manager()
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
        )
        await dispatcher.start()
        frame = torch.rand(3, 480, 640)
        await dispatcher.put_frame("stream_a", frame, 1.5)
        # Worker will consume the item, so we need to wait and put a new one
        await asyncio.sleep(0.3)
        # Put another item to verify queue still works
        await dispatcher.put_frame("stream_b", frame, 2.0)
        await asyncio.sleep(0.2)
        await dispatcher.stop()


@pytest.mark.asyncio
async def test_worker_loop_no_items() -> None:
    """Worker should timeout gracefully when no items in queue."""
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        heartbeat_interval=5,
    )
    await dispatcher.start()
    await asyncio.sleep(0.3)  # Worker should timeout, not hang
    await dispatcher.stop()
    assert dispatcher._running is False


# ── _process_batch logic ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_batch_empty_detections() -> None:
    """When detector returns empty boxes, on_track_result called with empty tensors."""
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    results = []

    async def on_track_result(
        stream_id: str,
        timestamp: float,
        tracked_ids: torch.Tensor,
        tracked_boxes: torch.Tensor,
        tracked_classes: torch.Tensor,
        tracked_kpts,
    ) -> None:
        results.append(
            {
                "stream_id": stream_id,
                "timestamp": timestamp,
                "ids_len": len(tracked_ids),
            }
        )

    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        on_track_result=on_track_result,
    )
    batch = [_queued_item(dispatcher, "stream_a", 1.0)]
    await dispatcher._process_batch(batch, DegradationLevel.NORMAL)
    assert len(results) == 1
    assert results[0]["stream_id"] == "stream_a"
    assert results[0]["ids_len"] == 0
    detector.detect_batch.assert_called_once()


@pytest.mark.asyncio
async def test_process_batch_with_detections() -> None:
    """When detector returns detections, tracker.update is called."""
    # Create detector that returns actual detections
    detector = MagicMock()
    boxes, scores, classes, keypoints = _make_detection_result()
    detector.detect_batch.return_value = [(boxes, scores, classes, keypoints)]
    detector.detect.return_value = detector.detect_batch.return_value[0]

    rm = _make_mock_resource_manager()
    tracker = _make_mock_tracker()

    with patch.object(AsyncDispatcher, "get_tracker", return_value=tracker):
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
        )
        batch = [_queued_item(dispatcher, "stream_a", 1.0)]
        await dispatcher._process_batch(batch, DegradationLevel.NORMAL)
        tracker.update.assert_called_once()
        call_args = tracker.update.call_args
        assert call_args[0][0] is boxes  # First positional arg is boxes
        detector.detect_batch.assert_called_once()


@pytest.mark.asyncio
async def test_process_batch_degradation_mode() -> None:
    """In FAST_PATH degradation, tracker.set_degradation_params is called."""
    # Create detector that returns actual detections (not empty)
    detector = MagicMock()
    boxes, scores, classes, keypoints = _make_detection_result()
    detector.detect_batch.return_value = [(boxes, scores, classes, keypoints)]
    detector.detect.return_value = detector.detect_batch.return_value[0]

    rm = _make_mock_resource_manager()
    tracker = _make_mock_tracker()

    with patch(
        "saccade.perception.dispatcher.AsyncDispatcher.get_tracker",
        return_value=tracker,
    ):
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
        )
        batch = [_queued_item(dispatcher, "stream_a", 1.0)]
        await dispatcher._process_batch(batch, DegradationLevel.FAST_PATH)
        tracker.set_degradation_params.assert_called_once_with(
            DegradationLevel.FAST_PATH
        )


@pytest.mark.asyncio
async def test_process_batch_batched_items() -> None:
    """Process multiple items in a single batch."""
    detector = _make_mock_detector_batch(3)
    rm = _make_mock_resource_manager()
    results = []

    async def on_track_result(*args, **kwargs) -> None:
        results.append(args[0])  # stream_id

    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        on_track_result=on_track_result,
    )
    batch = [
        _queued_item(dispatcher, "stream_a", 1.0),
        _queued_item(dispatcher, "stream_b", 2.0),
        _queued_item(dispatcher, "stream_c", 3.0),
    ]
    await dispatcher._process_batch(batch, DegradationLevel.NORMAL)
    assert len(results) == 3
    assert set(results) == {"stream_a", "stream_b", "stream_c"}
    detector.detect_batch.assert_called_once()
    input_tensor = detector.detect_batch.call_args.args[0]
    assert tuple(input_tensor.shape) == (3, 3, *dispatcher.input_hw)


@pytest.mark.asyncio
async def test_process_batch_collects_stats() -> None:
    detector = _make_mock_detector_batch(2)
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
        stats_window=32,
    )
    batch = [
        _queued_item(dispatcher, "stream_a", 1.0),
        _queued_item(dispatcher, "stream_b", 2.0),
    ]
    await dispatcher._process_batch(batch, DegradationLevel.NORMAL)
    stats = dispatcher.get_stats()
    assert stats["mean_batch_size"] == 2.0
    assert stats["batch_size_p95"] == 2.0
    assert stats["infer_ms_mean"] >= 0.0
    assert stats["end_to_end_ms_p95"] >= 0.0


# ── VRAM writer ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_with_vram_writer() -> None:
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()
    dispatcher = AsyncDispatcher(
        detector=detector,
        resource_manager=rm,
    )
    await dispatcher.start()
    assert dispatcher._vram_writer is not None
    await dispatcher.stop()


@pytest.mark.asyncio
async def test_start_vram_writer_failure() -> None:
    """VRAMLevelWriter failure should not prevent start."""
    detector = _make_mock_detector()
    rm = _make_mock_resource_manager()

    with patch(
        "saccade.perception.dispatcher.VRAMLevelWriter",
        side_effect=RuntimeError("simulated failure"),
    ):
        dispatcher = AsyncDispatcher(
            detector=detector,
            resource_manager=rm,
        )
        await dispatcher.start()
        assert dispatcher._running
        assert dispatcher._vram_writer is None
        await dispatcher.stop()
