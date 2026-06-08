import asyncio
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from saccade.perception.detector_trt import TRTYoloDetector, BatchingTRTDetector

try:
    from saccade_tracking_ext import GPUByteTracker
except ImportError:
    GPUByteTracker = Any

try:
    from saccade_tracking_ext import nv12_to_chw_letterbox as _nv12_letterbox_kernel
except ImportError:
    _nv12_letterbox_kernel = None

from saccade.perception.zero_copy import Nv12Frame
from saccade.resource.resource_manager import (
    DegradationLevel,
    ResourceManager,
    VRAMLevelWriter,
)

try:
    from saccade_tracking_ext import PerceptionPipelineConfig
except ImportError:
    PerceptionPipelineConfig = None

TrackResultCallback = Callable[
    [str, float, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Any
]


@dataclass(frozen=True)
class _QueuedFrame:
    stream_id: str
    frame: torch.Tensor
    timestamp: float
    enqueued_at: float


@dataclass
class _StreamFrame:
    """Frame for per-stream queues (latest-wins overflow)."""

    frame: torch.Tensor | Any  # RGB [H,W,3] uint8 or Nv12Frame
    timestamp: float
    enqueued_at: float


def _match_keypoints_to_tracks(
    tracked_boxes: torch.Tensor,
    det_boxes: torch.Tensor,
    keypoints: torch.Tensor,
) -> torch.Tensor:
    """For each tracked box, return keypoints of the nearest detection by center distance."""
    if tracked_boxes.numel() == 0:
        return torch.empty((0, *keypoints.shape[1:]), device=tracked_boxes.device)
    det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) * 0.5
    det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) * 0.5
    trk_cx = (tracked_boxes[:, 0] + tracked_boxes[:, 2]) * 0.5
    trk_cy = (tracked_boxes[:, 1] + tracked_boxes[:, 3]) * 0.5
    dx = trk_cx.unsqueeze(1) - det_cx.unsqueeze(0)
    dy = trk_cy.unsqueeze(1) - det_cy.unsqueeze(0)
    nearest = (dx * dx + dy * dy).argmin(dim=1)
    return keypoints[nearest]


def _percentile(values: list[float] | list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


class _WorkbenchWorker(threading.Thread):
    """Per-thread worker that pulls from a per-stream queue and processes with Workbench."""

    def __init__(
        self,
        stream_id: str,
        workbench: Any,
        stream_queue: queue.Queue[Optional[_StreamFrame]],  # None = stop signal
        on_result: Callable[
            [
                str,
                float,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Optional[torch.Tensor],
            ],
            None,
        ],
        frame_w: int,
        frame_h: int,
        input_size: tuple[int, int],
        stats_window: deque[float],
        queue_wait_window: deque[float],
        use_nv12: bool = False,
    ):
        super().__init__(daemon=True, name=f"wb-{stream_id}")
        self.stream_id = stream_id
        self.workbench = workbench
        self.stream_queue = stream_queue
        self.on_result = on_result
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.input_size = input_size
        self.stats_window = stats_window
        self.queue_wait_window = queue_wait_window
        self.use_nv12 = use_nv12

    def run(self) -> None:
        while True:
            item = self.stream_queue.get()
            if item is None:
                break  # stop signal

            enqueued_at = item.enqueued_at
            queue_wait_ms = (time.perf_counter() - enqueued_at) * 1000.0
            self.queue_wait_window.append(queue_wait_ms)

            t_start = time.perf_counter()

            if self.use_nv12:
                if _nv12_letterbox_kernel is None:
                    raise RuntimeError(
                        "WorkbenchPool NV12 mode requires nv12_to_chw_letterbox "
                        "from saccade_tracking_ext to be available."
                    )
                if not isinstance(item.frame, Nv12Frame):
                    raise TypeError("WorkbenchPool NV12 mode expects Nv12Frame inputs.")
                nv12_frame: Nv12Frame = item.frame
                src_w = nv12_frame.width
                src_h = nv12_frame.height
                y_pitch = nv12_frame.y_pitch
                uv_pitch = nv12_frame.uv_pitch
                nv12_buf = nv12_frame.buf

                dst_w, dst_h = self.input_size
                scale = min(dst_w / src_w, dst_h / src_h)
                new_w = int(round(src_w * scale))
                new_h = int(round(src_h * scale))
                x_off = (dst_w - new_w) // 2
                y_off = (dst_h - new_h) // 2

                padded = torch.zeros(
                    3,
                    dst_h,
                    dst_w,
                    dtype=torch.float32,
                    device=nv12_buf.device,
                )
                stream = torch.cuda.current_stream().cuda_stream
                _nv12_letterbox_kernel(
                    nv12_buf.data_ptr(),
                    y_pitch,
                    nv12_buf.data_ptr() + nv12_buf.element_size() * src_h * y_pitch,
                    uv_pitch,
                    src_w,
                    src_h,
                    padded.data_ptr(),
                    dst_w,
                    x_off,
                    y_off,
                    new_w,
                    new_h,
                    114.0 / 255.0,
                    stream,
                )
            else:
                # Preprocess: normalize to [0,1] and letterbox to input_size
                frame_hwc = item.frame
                frame_chw = frame_hwc.permute(2, 0, 1).float() / 255.0
                # Letterbox to input_size
                w, h = frame_chw.shape[2], frame_chw.shape[1]
                scale = min(self.input_size[0] / w, self.input_size[1] / h)
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                letterbox = torch.nn.functional.interpolate(
                    frame_chw.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                padded = torch.zeros(
                    3,
                    self.input_size[1],
                    self.input_size[0],
                    dtype=torch.float32,
                    device=letterbox.device,
                )
                padded[:, :new_h, :new_w] = letterbox

            # Process frame through Workbench
            result = self.workbench.process_frame(
                padded,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.stats_window.append(elapsed_ms)

            # Emit result via callback (runs in caller's thread)
            self.on_result(
                self.stream_id,
                item.timestamp,
                result.ids,
                result.boxes,
                result.classes,
                None,  # keypoints not available in workbench path
            )


class WorkbenchPool:
    """Production-ready multi-stream pool using per-thread Workbenches + shared BatchingTRTDetector.

    Architecture:
      - One BatchingTRTDetector (shared YOLO engine, server thread handles batching)
      - Per-stream queue.Queue (latest-wins on overflow, maxsize=4)
      - N daemon threads, each owns one Workbench (pipeline + tracker + CUDA stream)
      - Results emitted via on_result callback (caller handles threading)

    This replaces AsyncDispatcher for the multi-stream case. Key advantages:
      - True cross-thread parallelism (no GIL serialization in hot path)
      - Per-stream isolation (tracker state, GPU scratch, CUDA stream)
      - BatchingTRTDetector handles YOLO batching with CUDA event handshake
    """

    def __init__(
        self,
        engine_path: str,
        n_streams: int,
        on_result: Optional[TrackResultCallback] = None,
        input_hw: tuple[int, int] = (960, 960),
        frame_size: tuple[int, int] = (1920, 1080),
        max_dets: int = 2048,
        max_tracks: int = 256,
        stats_window: int = 512,
        use_nv12: bool = False,
    ):
        self.engine_path = engine_path
        self.n_streams = n_streams
        self.on_result = on_result
        self.input_hw = input_hw
        self.frame_size = frame_size
        self.max_dets = max_dets
        self.max_tracks = max_tracks
        self.use_nv12 = use_nv12

        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._workers: list[_WorkbenchWorker] = []
        self._queues: list[queue.Queue[Optional[_StreamFrame]]] = []
        self._batcher: Optional[BatchingTRTDetector] = None

        self._stats_window: deque[float] = deque(maxlen=stats_window)
        self._queue_wait_window: deque[float] = deque(maxlen=stats_window)
        self._drop_count = 0
        self._lock = threading.Lock()

    async def start(self) -> None:
        """Start the pool: create batcher, workbenches, and worker threads."""
        self._running = True
        self._loop = asyncio.get_event_loop()

        if self.use_nv12 and _nv12_letterbox_kernel is None:
            raise RuntimeError(
                "WorkbenchPool NV12 mode requires nv12_to_chw_letterbox from "
                "saccade_tracking_ext to be available."
            )

        # Create shared batching detector
        self._batcher = BatchingTRTDetector(self.engine_path, batch_size=self.n_streams)

        if PerceptionPipelineConfig is None:
            raise RuntimeError(
                "saccade_tracking_ext C++ extension is not available. "
                "Please build the extension: `pip install -e .` or check "
                "CONCURRENT_EVAL.md for build instructions."
            )
        pipeline_cfg = PerceptionPipelineConfig()

        # Create per-workbench worker threads
        for i in range(self.n_streams):
            sid = f"stream_{i}"
            q: queue.Queue[Optional[_StreamFrame]] = queue.Queue(maxsize=4)
            self._queues.append(q)

            assert self._batcher is not None
            batcher = self._batcher

            def make_wb() -> Any:
                proxy = batcher.make_proxy()
                from saccade.perception.workbench import Workbench

                return Workbench(
                    proxy,
                    pipeline_cfg,
                    device=str(batcher._device),
                    max_dets=self.max_dets,
                    max_tracks=self.max_tracks,
                )

            wb = make_wb()
            q_frame_w, q_frame_h = self.frame_size
            stats_w: deque[float] = deque(maxlen=self._stats_window.maxlen)
            wait_w: deque[float] = deque(maxlen=self._queue_wait_window.maxlen)

            loop = self._loop
            assert loop is not None

            # Callback: forward actual workbench results to asyncio event loop
            def make_callback(sid: str = sid) -> Any:
                def _cb(
                    sid: str,
                    ts: float,
                    ids: torch.Tensor,
                    boxes: torch.Tensor,
                    classes: torch.Tensor,
                    kps: Optional[torch.Tensor],
                ) -> None:
                    if self.on_result:
                        loop.call_soon_threadsafe(
                            self.on_result, sid, ts, ids, boxes, classes, kps
                        )

                return _cb

            worker = _WorkbenchWorker(
                stream_id=sid,
                workbench=wb,
                stream_queue=q,
                on_result=make_callback(sid),
                frame_w=q_frame_w,
                frame_h=q_frame_h,
                input_size=self.input_hw,
                stats_window=stats_w,
                queue_wait_window=wait_w,
                use_nv12=self.use_nv12,
            )
            worker.start()
            self._workers.append(worker)

        print(
            f"🚀 [WorkbenchPool] Started {self.n_streams} workbench threads "
            f"sharing {self.engine_path}"
        )

    def put_frame(
        self, stream_id: str, frame: torch.Tensor | Any, timestamp: float
    ) -> None:
        """Enqueue a frame for the given stream. Latest-wins on overflow."""
        idx = int(stream_id.split("_")[-1]) if "_" in stream_id else 0
        if idx >= len(self._queues):
            return
        q = self._queues[idx]
        item = _StreamFrame(
            frame=frame,
            timestamp=timestamp,
            enqueued_at=time.perf_counter(),
        )
        try:
            q.put_nowait(item)
        except queue.Full:
            # Latest-wins: drain oldest and enqueue new one
            with self._lock:
                self._drop_count += 1
            try:
                while not q.empty():
                    q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def stop(self) -> None:
        """Stop all workers and shut down the batcher."""
        self._running = False
        for q in self._queues:
            q.put(None)  # stop signal
        for w in self._workers:
            w.join(timeout=5.0)
        if self._batcher is not None:
            self._batcher.shutdown()
            self._batcher = None
        print("🛑 [WorkbenchPool] Stopped.")

    def get_stats(self) -> dict[str, Any]:
        """Return current stats."""
        drops = self._drop_count
        self._drop_count = 0

        stats = list(self._stats_window)
        wait = list(self._queue_wait_window)

        return {
            "current_level": "NORMAL",
            "active_streams": self.n_streams,
            "queue_depth": sum(q.qsize() for q in self._queues),
            "drops": drops,
            "e2e_ms_mean": round(float(np.mean(stats)), 2) if stats else 0.0,
            "e2e_ms_p95": round(_percentile(stats, 95), 2) if stats else 0.0,
            "e2e_ms_p99": round(_percentile(stats, 99), 2) if stats else 0.0,
            "queue_wait_ms_mean": round(float(np.mean(wait)), 2) if wait else 0.0,
            "queue_wait_ms_p95": round(_percentile(wait, 95), 2) if wait else 0.0,
            "queue_wait_ms_p99": round(_percentile(wait, 99), 2) if wait else 0.0,
        }


class AsyncDispatcher:
    def __init__(
        self,
        detector: TRTYoloDetector,
        resource_manager: ResourceManager,
        max_batch: int = 4,
        conf_threshold: float = 0.25,
        on_track_result: Optional[TrackResultCallback] = None,
        extractor: Optional[Any] = None,
        heartbeat_interval: int = 10,
        max_streams: int = 8,
        input_hw: tuple[int, int] = (960, 960),
        batch_timeout_ms: float = 3.0,
        stats_window: int = 512,
    ):
        self.detector = detector
        self.resource_manager = resource_manager
        self.max_batch = max_batch
        self.conf_threshold = conf_threshold
        self.on_track_result = on_track_result
        self.extractor = extractor
        self.heartbeat_interval = heartbeat_interval
        self.max_streams = max_streams
        self.input_hw = input_hw
        self.batch_timeout_ms = batch_timeout_ms
        self.stats_window = stats_window

        self.queue: asyncio.Queue[_QueuedFrame] = asyncio.Queue(maxsize=100)
        self.trackers: OrderedDict[str, Any] = OrderedDict()
        self._running = False
        self._vram_writer: Optional[VRAMLevelWriter] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._current_level = DegradationLevel.NORMAL

        self._batch_sizes: deque[int] = deque(maxlen=stats_window)
        self._queue_wait_ms: deque[float] = deque(maxlen=stats_window)
        self._infer_ms: deque[float] = deque(maxlen=stats_window)
        self._track_ms: deque[float] = deque(maxlen=stats_window)
        self._end_to_end_ms: deque[float] = deque(maxlen=stats_window)

    def _make_tracker(self) -> Any:
        return GPUByteTracker(max_objs=100, embedding_dim=768 if self.extractor else 0)

    def get_tracker(self, stream_id: str) -> Any:
        if stream_id in self.trackers:
            self.trackers.move_to_end(stream_id)
            return self.trackers[stream_id]

        if len(self.trackers) >= self.max_streams:
            evicted_id, evicted = self.trackers.popitem(last=False)
            del evicted
            print(
                f"[Dispatcher] Evicted tracker '{evicted_id}' (LRU cap={self.max_streams})"
            )

        tracker = self._make_tracker()
        self.trackers[stream_id] = tracker
        return tracker

    def deregister_stream(self, stream_id: str) -> None:
        tracker = self.trackers.pop(stream_id, None)
        if tracker is not None:
            del tracker

    async def start(self) -> None:
        self._running = True
        try:
            self._vram_writer = VRAMLevelWriter()
        except Exception as e:
            print(f"[Dispatcher] VRAMLevelWriter init failed: {e}")
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        while self.trackers:
            _, tracker = self.trackers.popitem()
            del tracker
        if self._vram_writer is not None:
            self._vram_writer.close()
            self._vram_writer = None

    async def put_frame(
        self, stream_id: str, frame: torch.Tensor, timestamp: float
    ) -> None:
        try:
            self.queue.put_nowait(
                _QueuedFrame(
                    stream_id=stream_id,
                    frame=frame,
                    timestamp=timestamp,
                    enqueued_at=time.perf_counter(),
                )
            )
        except asyncio.QueueFull:
            pass

    def get_stats(self) -> dict[str, float | int | str]:
        batch_sizes = list(self._batch_sizes)
        queue_wait = list(self._queue_wait_ms)
        infer = list(self._infer_ms)
        track = list(self._track_ms)
        end_to_end = list(self._end_to_end_ms)

        return {
            "queue_depth": self.queue.qsize(),
            "active_streams": len(self.trackers),
            "current_level": self._current_level.name,
            "max_batch": self.max_batch,
            "batch_timeout_ms": self.batch_timeout_ms,
            "mean_batch_size": round(float(np.mean(batch_sizes)), 3)
            if batch_sizes
            else 0.0,
            "batch_size_p95": round(_percentile(batch_sizes, 95), 3)
            if batch_sizes
            else 0.0,
            "queue_wait_ms_mean": round(float(np.mean(queue_wait)), 3)
            if queue_wait
            else 0.0,
            "queue_wait_ms_p95": round(_percentile(queue_wait, 95), 3),
            "queue_wait_ms_p99": round(_percentile(queue_wait, 99), 3),
            "infer_ms_mean": round(float(np.mean(infer)), 3) if infer else 0.0,
            "infer_ms_p95": round(_percentile(infer, 95), 3),
            "infer_ms_p99": round(_percentile(infer, 99), 3),
            "track_ms_mean": round(float(np.mean(track)), 3) if track else 0.0,
            "track_ms_p95": round(_percentile(track, 95), 3),
            "track_ms_p99": round(_percentile(track, 99), 3),
            "end_to_end_ms_mean": round(float(np.mean(end_to_end)), 3)
            if end_to_end
            else 0.0,
            "end_to_end_ms_p95": round(_percentile(end_to_end, 95), 3),
            "end_to_end_ms_p99": round(_percentile(end_to_end, 99), 3),
        }

    def _normalize_batch_item(
        self, item: _QueuedFrame | tuple[str, torch.Tensor, float]
    ) -> _QueuedFrame:
        if isinstance(item, _QueuedFrame):
            return item
        stream_id, frame, timestamp = item
        return _QueuedFrame(
            stream_id=stream_id,
            frame=frame,
            timestamp=timestamp,
            enqueued_at=time.perf_counter(),
        )

    async def _worker_loop(self) -> None:
        reid_info = (
            f"heartbeat={self.heartbeat_interval}f"
            if self.extractor
            else "ReID disabled"
        )
        print(f"🚀 [Dispatcher] Worker started. {reid_info}")

        while self._running:
            try:
                first_item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            batch_items: list[_QueuedFrame | tuple[str, torch.Tensor, float]] = [
                first_item
            ]
            level = self.resource_manager.decide_degradation_level()
            self._current_level = level
            if self._vram_writer is not None:
                self._vram_writer.write(level)
            current_max = self.max_batch if level < DegradationLevel.FAST_PATH else 2
            deadline = time.perf_counter() + max(self.batch_timeout_ms, 0.0) / 1000.0

            while len(batch_items) < current_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch_items.append(
                        await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    )
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch_items, level)

            for _ in batch_items:
                self.queue.task_done()

    async def _process_batch(
        self,
        batch_items: list[_QueuedFrame | tuple[str, torch.Tensor, float]],
        level: int,
    ) -> None:
        normalized = [self._normalize_batch_item(item) for item in batch_items]
        batch_size = len(normalized)
        if batch_size == 0:
            return

        batch_start = time.perf_counter()
        for item in normalized:
            self._queue_wait_ms.append((batch_start - item.enqueued_at) * 1000.0)

        input_tensor = torch.stack(
            [item.frame.contiguous() for item in normalized], dim=0
        ).contiguous()

        infer_start = time.perf_counter()
        with torch.no_grad():
            batch_results = self.detector.detect_batch(
                input_tensor, conf_threshold=self.conf_threshold
            )
        infer_elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
        self._infer_ms.append(infer_elapsed_ms)
        self._batch_sizes.append(batch_size)

        track_start = time.perf_counter()
        for item, result in zip(normalized, batch_results):
            boxes, scores, classes, keypoints = result

            if boxes.numel() == 0:
                if self.on_track_result:
                    dev = item.frame.device
                    await self.on_track_result(
                        item.stream_id,
                        item.timestamp,
                        torch.empty((0,), dtype=torch.int32, device=dev),
                        torch.empty((0, 4), dtype=torch.float32, device=dev),
                        torch.empty((0,), dtype=torch.int32, device=dev),
                        None,
                    )
                self._end_to_end_ms.append(
                    (time.perf_counter() - item.enqueued_at) * 1000.0
                )
                continue

            tracker = self.get_tracker(item.stream_id)
            if level >= DegradationLevel.FAST_PATH:
                tracker.set_degradation_params(level)
                tracked_ids, tracked_boxes, tracked_classes = tracker.update(
                    boxes, scores, classes
                )
            else:
                tracked_ids, tracked_boxes, tracked_classes = tracker.update(
                    boxes,
                    scores,
                    classes,
                    frame_tensor=item.frame,
                    stream_id=hash(item.stream_id) & 0x7FFFFFFF,
                )

            tracked_kpts = (
                _match_keypoints_to_tracks(tracked_boxes, boxes, keypoints)
                if keypoints is not None
                else None
            )

            if self.on_track_result:
                await self.on_track_result(
                    item.stream_id,
                    item.timestamp,
                    tracked_ids,
                    tracked_boxes,
                    tracked_classes,
                    tracked_kpts,
                )
            self._end_to_end_ms.append(
                (time.perf_counter() - item.enqueued_at) * 1000.0
            )

        track_elapsed_ms = (time.perf_counter() - track_start) * 1000.0
        self._track_ms.append(track_elapsed_ms / max(batch_size, 1))
        total_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
        print(
            f"⚡ [Batch] size={batch_size} infer={infer_elapsed_ms:.2f}ms "
            f"track={track_elapsed_ms:.2f}ms total={total_elapsed_ms:.2f}ms"
        )
