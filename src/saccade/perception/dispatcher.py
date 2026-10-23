import asyncio
import inspect
import queue
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch

from saccade.perception.detector_trt import TRTYoloDetector, BatchingTRTDetector
from saccade.perception.online_telemetry import OnlineTelemetry
from saccade.perception.online_tracker_adapter import (
    OnlineTrackerAdapter,
    OnlineTrackerUnavailable,
    native_unavailable_reason,
)

try:
    from saccade_tracking_ext import nv12_to_chw_letterbox as _nv12_letterbox_kernel
except ImportError:
    _nv12_letterbox_kernel = None

from saccade.perception.zero_copy import Nv12Frame
from saccade.resource.resource_manager import (
    DegradationLevel,
    ResourceManager,
    ResourceMonitorUnavailable,
    VRAMLevelWriter,
    vram_level_clock_ns,
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


def _own_result(result: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copies of a Workbench result that the consumer owns (#538 D2-GPU G2).

    Workbench returns views of buffers that its next frame overwrites. The
    copies are queued on the calling thread's current stream, which
    ``Workbench.process_frame`` made wait for its work; a consumer on that
    stream (the legacy default stream, for the event loop) reads them in order.
    """
    return result.ids.clone(), result.boxes.clone(), result.classes.clone()


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
        telemetry: OnlineTelemetry | None = None,
        use_nv12: bool = False,
        stop_event: Optional[threading.Event] = None,
        on_failure: Optional[Callable[[str, Exception], None]] = None,
    ):
        super().__init__(daemon=True, name=f"wb-{stream_id}")
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.on_failure = on_failure
        self.stream_id = stream_id
        self.workbench = workbench
        self.stream_queue = stream_queue
        self.on_result = on_result
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.input_size = input_size
        self.stats_window = stats_window
        self.queue_wait_window = queue_wait_window
        self.telemetry = telemetry
        self.use_nv12 = use_nv12

    def run(self) -> None:
        # #538 D2-WB-1: exit on stop_event (no blocking sentinel put), and hand
        # any exception to the pool instead of dying silently.
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.stream_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break  # legacy stop signal
                self._process(item)
        except Exception as e:
            if self.on_failure is None:
                raise
            self.on_failure(self.stream_id, e)

    def _process(self, item: _StreamFrame) -> None:
        enqueued_at = item.enqueued_at
        queue_wait_ms = (time.perf_counter() - enqueued_at) * 1000.0
        self.queue_wait_window.append(queue_wait_ms)

        t_start = time.perf_counter()
        preproc_start = t_start

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

        preproc_ms = (time.perf_counter() - preproc_start) * 1000.0
        if self.telemetry is not None:
            self.telemetry.observe("workbench_preprocess", preproc_ms)

        # #538 D2-GPU G1: preprocessing (the RGB ops or the NV12 kernel) ran on
        # this thread's current stream, but Workbench reads `padded` on its own
        # non-blocking stream. Order that stream after ours. The same wait orders
        # this frame's writes to Workbench.out_* after the previous frame's
        # result copies (G2), which were queued on our stream.
        if padded.is_cuda:
            self.workbench.stream.wait_stream(torch.cuda.current_stream(padded.device))

        # Process frame through Workbench
        process_start = time.perf_counter()
        result = self.workbench.process_frame(
            padded,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )
        if self.telemetry is not None:
            self.telemetry.observe(
                "workbench_process",
                (time.perf_counter() - process_start) * 1000.0,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.stats_window.append(elapsed_ms)
        if self.telemetry is not None:
            self.telemetry.observe("workbench_worker", elapsed_ms)

        # Emit result via callback (runs in caller's thread)
        ids, boxes, classes = _own_result(result)
        self.on_result(
            self.stream_id,
            item.timestamp,
            ids,
            boxes,
            classes,
            None,  # keypoints not available in workbench path
        )


class PoolState(Enum):
    CREATED = "created"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"  # stop() hit its deadline; call stop() again
    STOPPED = "stopped"


class WorkbenchPoolFailed(RuntimeError):
    """A worker failed; ``__cause__`` is the original error (#538 D2-WB-1)."""


class WorkbenchPoolStopTimeout(RuntimeError):
    """stop() reached its deadline with workers alive; the batcher was kept."""


class WorkbenchPoolNotRunning(RuntimeError):
    """put_frame() outside RUNNING: there is no consumer for new work (#538)."""


def _is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
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

    No resource monitoring is wired: stats report ``current_level="UNKNOWN"``
    and ``resource_monitor_status="not_wired"`` (#537 C0-09).
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
        telemetry: OnlineTelemetry | None = None,
    ):
        self.engine_path = engine_path
        self.n_streams = n_streams
        self.on_result = on_result
        self.input_hw = input_hw
        self.frame_size = frame_size
        self.max_dets = max_dets
        self.max_tracks = max_tracks
        self.use_nv12 = use_nv12
        self.telemetry = telemetry

        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._workers: list[_WorkbenchWorker] = []
        self._queues: list[queue.Queue[Optional[_StreamFrame]]] = []
        self._batcher: Optional[BatchingTRTDetector] = None

        self._stats_window: deque[float] = deque(maxlen=stats_window)
        self._queue_wait_window: deque[float] = deque(maxlen=stats_window)
        self._drop_count = 0
        self._lock = threading.Lock()

        # #538 D2-WB-1: lifecycle state, the first worker failure, and counters.
        self._state = PoolState.CREATED
        self._failure: Optional[Exception] = None
        self._failed_stream: Optional[str] = None
        self._stop_event = threading.Event()
        # Bumped by start(); a callback carries the generation it was produced
        # in and is delivered only if that generation is still current and the
        # pool is RUNNING/FAILED.
        self._generation = 0
        self._late_callbacks = 0
        self._discarded_on_stop = 0
        self._stop_timeouts = 0

    @property
    def state(self) -> PoolState:
        return self._state

    @property
    def failure(self) -> Optional[Exception]:
        """The first worker's original error, kept until the next ``start()``."""
        return self._failure

    def _failed(self) -> WorkbenchPoolFailed:
        err = WorkbenchPoolFailed(
            f"WorkbenchPool worker {self._failed_stream} failed: {self._failure!r}"
        )
        err.__cause__ = self._failure
        return err

    def _on_worker_failure(self, stream_id: str, error: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
                self._failed_stream = stream_id
            if self._state is PoolState.RUNNING:
                self._state = PoolState.FAILED
                self._running = False
        print(f"❌ [WorkbenchPool] Worker {stream_id} failed: {error!r}")

    async def start(self) -> None:
        """Start the pool: create batcher, workbenches, and worker threads.

        A failure part-way through stops what was started before re-raising.
        """
        if self._state is PoolState.RUNNING:
            raise RuntimeError("WorkbenchPool is already running")
        if self._state is PoolState.FAILED:
            raise self._failed()  # stop() first
        if self._state is PoolState.STOPPING:
            # STOPPED is set only after every worker ended and the batcher was
            # released, so refusing STOPPING refuses a restart over live workers.
            raise WorkbenchPoolStopTimeout(
                "WorkbenchPool is still stopping; call stop() again first"
            )

        # Checks that need no resources come before the batcher is created.
        if self.on_result is not None and _is_async_callable(self.on_result):
            # Results are handed over with call_soon_threadsafe; a coroutine
            # would be created and never awaited. Async callbacks need their
            # own task ownership/failure/cancellation contract (#538).
            raise TypeError("WorkbenchPool does not support a coroutine on_result")
        if self.use_nv12 and _nv12_letterbox_kernel is None:
            raise RuntimeError(
                "WorkbenchPool NV12 mode requires nv12_to_chw_letterbox from "
                "saccade_tracking_ext to be available."
            )
        if PerceptionPipelineConfig is None:
            raise RuntimeError(
                "saccade_tracking_ext C++ extension is not available. "
                "Please build the extension: `pip install -e .` or check "
                "CONCURRENT_EVAL.md for build instructions."
            )

        # A restart gets fresh queues, workers and stop event.
        self._queues = []
        self._workers = []
        self._stop_event = threading.Event()
        self._failure = None
        self._failed_stream = None
        self._loop = asyncio.get_event_loop()
        with self._lock:
            self._generation += 1
            self._state = PoolState.RUNNING
            self._running = True
        try:
            self._start_workers()
        except BaseException as e:
            try:
                self.stop()
            except Exception as cleanup_error:
                e.add_note(f"rollback after failed start: {cleanup_error!r}")
            raise

    def _start_workers(self) -> None:
        # Create shared batching detector
        self._batcher = BatchingTRTDetector(self.engine_path, batch_size=self.n_streams)
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
            loop = self._loop
            assert loop is not None

            # Callback: forward actual workbench results to asyncio event loop.
            # Results that arrive after stop() began, or after the loop closed,
            # are dropped and counted instead of killing the worker (D2-WB-1).
            # _deliver() checks the generation again when the loop runs it.
            def make_callback(sid: str = sid, gen: int = self._generation) -> Any:
                def _cb(
                    sid: str,
                    ts: float,
                    ids: torch.Tensor,
                    boxes: torch.Tensor,
                    classes: torch.Tensor,
                    kps: Optional[torch.Tensor],
                ) -> None:
                    if not self.on_result:
                        return
                    if (
                        self._state not in (PoolState.RUNNING, PoolState.FAILED)
                        or loop.is_closed()
                    ):
                        self._count_late_callback()
                        return
                    try:
                        loop.call_soon_threadsafe(
                            self._deliver, gen, sid, ts, ids, boxes, classes, kps
                        )
                    except RuntimeError:  # the loop closed after the check
                        self._count_late_callback()

                return _cb

            worker = _WorkbenchWorker(
                stream_id=sid,
                workbench=wb,
                stream_queue=q,
                on_result=make_callback(sid),
                frame_w=q_frame_w,
                frame_h=q_frame_h,
                input_size=self.input_hw,
                stats_window=self._stats_window,
                queue_wait_window=self._queue_wait_window,
                telemetry=self.telemetry,
                use_nv12=self.use_nv12,
                stop_event=self._stop_event,
                on_failure=self._on_worker_failure,
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
        """Enqueue a frame for ``stream_0``..``stream_{n_streams-1}``.

        Latest-wins on overflow. Refused outside RUNNING (#538): FAILED raises
        ``WorkbenchPoolFailed``, any other state ``WorkbenchPoolNotRunning``.
        Any other stream id raises ``ValueError``.
        """
        if self._state is PoolState.FAILED:
            raise self._failed()
        if self._state is not PoolState.RUNNING:
            raise WorkbenchPoolNotRunning(
                f"WorkbenchPool is {self._state.value}; put_frame() needs running"
            )
        q = self._queues[self._stream_index(stream_id)]
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
            if self.telemetry is not None:
                self.telemetry.increment("workbench_drops")
            try:
                while not q.empty():
                    q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def _stream_index(self, stream_id: str) -> int:
        m = re.fullmatch(r"stream_(0|[1-9][0-9]*)", stream_id)
        if m is None or int(m.group(1)) >= self.n_streams:
            raise ValueError(
                f"WorkbenchPool stream id must be stream_0..stream_{self.n_streams - 1}, "
                f"got {stream_id!r}"
            )
        return int(m.group(1))

    def _count_late_callback(self) -> None:
        with self._lock:
            self._late_callbacks += 1

    def _deliver(
        self,
        gen: int,
        sid: str,
        ts: float,
        ids: torch.Tensor,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        kps: Optional[torch.Tensor],
    ) -> None:
        """Runs on the event loop. Drops a result queued before a stop/restart."""
        if gen != self._generation or self._state not in (
            PoolState.RUNNING,
            PoolState.FAILED,
        ):
            self._count_late_callback()
            return
        if self.on_result:
            self.on_result(sid, ts, ids, boxes, classes, kps)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop all workers within ``timeout`` seconds, then shut down the batcher.

        The worker joins share one deadline. If a worker is still alive at the
        deadline, the batcher is kept, the state is STOPPING, and
        ``WorkbenchPoolStopTimeout`` is raised; call ``stop()`` again to retry.
        ``batcher.shutdown()`` runs after the deadline with its own bound
        (``BatchingTRTDetector`` joins its server for up to 2 s). If it raises
        or reports that its server is still alive, the batcher reference is
        kept and the state stays STOPPING. STOPPED is reported only after the
        workers ended and the batcher was released. Then a failed run raises
        ``WorkbenchPoolFailed``.
        """
        if self._state in (PoolState.CREATED, PoolState.STOPPED):
            return
        with self._lock:
            self._state = PoolState.STOPPING
            self._running = False
        self._stop_event.set()
        deadline = time.monotonic() + max(timeout, 0.0)
        for w in self._workers:
            w.join(timeout=max(deadline - time.monotonic(), 0.0))
        alive = [w.name for w in self._workers if w.is_alive()]
        if alive:
            with self._lock:
                self._stop_timeouts += 1
            raise WorkbenchPoolStopTimeout(
                f"WorkbenchPool workers still running after {timeout}s: {alive}; "
                "the batcher was not shut down"
            )

        discarded = 0
        for q in self._queues:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                discarded += 1
        with self._lock:
            self._discarded_on_stop += discarded
        if self._batcher is not None:
            # A raise propagates with the batcher kept and the state STOPPING.
            if self._batcher.shutdown() is False:
                with self._lock:
                    self._stop_timeouts += 1
                raise WorkbenchPoolStopTimeout(
                    "WorkbenchPool batcher server did not end; the batcher was kept"
                )
            self._batcher = None
        self._state = PoolState.STOPPED
        print("🛑 [WorkbenchPool] Stopped.")
        if self._failure is not None:
            raise self._failed()

    def get_stats(self) -> dict[str, Any]:
        """Return current stats. Does not reset counters."""
        with self._lock:
            drops = self._drop_count

        stats = list(self._stats_window)
        wait = list(self._queue_wait_window)
        telemetry = self.telemetry.summary() if self.telemetry is not None else {}

        return {
            "state": self._state.value,
            "failure": repr(self._failure) if self._failure is not None else "",
            "failed_stream": self._failed_stream or "",
            "late_callbacks": self._late_callbacks,
            "discarded_on_stop": self._discarded_on_stop,
            "stop_timeouts": self._stop_timeouts,
            # C0-09: no resource manager or VRAM writer is wired to this path,
            # so the level is unknown rather than a healthy-looking NORMAL.
            "current_level": "UNKNOWN",
            "resource_monitor_status": "not_wired",
            "resource_monitor_error": "",
            "active_streams": self.n_streams,
            "queue_depth": sum(q.qsize() for q in self._queues),
            "drops": drops,
            "e2e_ms_mean": round(float(np.mean(stats)), 2) if stats else 0.0,
            "e2e_ms_p95": round(_percentile(stats, 95), 2) if stats else 0.0,
            "e2e_ms_p99": round(_percentile(stats, 99), 2) if stats else 0.0,
            "queue_wait_ms_mean": round(float(np.mean(wait)), 2) if wait else 0.0,
            "queue_wait_ms_p95": round(_percentile(wait, 95), 2) if wait else 0.0,
            "queue_wait_ms_p99": round(_percentile(wait, 99), 2) if wait else 0.0,
            **telemetry,
        }

    def reset_stats(self) -> None:
        """Reset the drop counter."""
        with self._lock:
            self._drop_count = 0


class DispatcherState(Enum):
    CREATED = "created"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class DispatcherFailed(RuntimeError):
    """The worker task failed; ``__cause__`` is the original error (#538 D2-AD)."""


class DispatcherNotRunning(RuntimeError):
    """put_frame() outside RUNNING: there is no consumer for new work (#538)."""


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
        telemetry: OnlineTelemetry | None = None,
    ):
        if extractor is not None:
            # #537 R-1: only geometry-only tracking is supported. The tracker
            # never received embeddings here, so ReID is refused, not implied.
            raise ValueError(
                "AsyncDispatcher supports geometry-only tracking; "
                "extractor/ReID is not supported (#537 R-1)"
            )
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
        self.telemetry = telemetry

        self.queue: asyncio.Queue[_QueuedFrame] = asyncio.Queue(maxsize=100)
        self.trackers: OrderedDict[str, Any] = OrderedDict()
        self._running = False
        self._vram_writer: Optional[VRAMLevelWriter] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        # #537 D-537-4A: a retained hysteresis value is not a valid observation
        # for a new run. ResourceManager owns that state; this is only the last
        # successful decision in the current run, hidden after monitor failure.
        self._current_level: Optional[DegradationLevel] = None
        self._resource_monitor_status = "unprobed"
        self._resource_monitor_error = ""
        # #538 D2-AD: lifecycle state, the worker's original error, the batch
        # taken off the queue but not yet task_done(), and frame accounting.
        self._state = DispatcherState.CREATED
        self._failure: Optional[Exception] = None
        self._inflight: list[_QueuedFrame | tuple[str, torch.Tensor, float]] = []
        self._dropped_queue_full = 0
        self._failed_batch_frames = 0
        self._discarded_on_failure = 0
        self._discarded_on_stop = 0

        self._batch_sizes: deque[int] = deque(maxlen=stats_window)
        self._queue_wait_ms: deque[float] = deque(maxlen=stats_window)
        self._infer_ms: deque[float] = deque(maxlen=stats_window)
        self._track_ms: deque[float] = deque(maxlen=stats_window)
        self._end_to_end_ms: deque[float] = deque(maxlen=stats_window)
        self._batch_build_ms: deque[float] = deque(maxlen=stats_window)

    def _make_tracker(self) -> Any:
        return OnlineTrackerAdapter(max_objects=100)

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

    @property
    def state(self) -> DispatcherState:
        return self._state

    @property
    def failure(self) -> Optional[Exception]:
        """The worker's original error, kept until the next ``start()``."""
        return self._failure

    def _failed(self) -> DispatcherFailed:
        err = DispatcherFailed(f"AsyncDispatcher worker failed: {self._failure!r}")
        err.__cause__ = self._failure
        return err

    def _monitor_unavailable(self, error: ResourceMonitorUnavailable) -> None:
        self._resource_monitor_status = "unavailable"
        self._resource_monitor_error = repr(error)
        self._current_level = None

    async def start(self) -> None:
        if self._state is DispatcherState.RUNNING:
            raise RuntimeError("AsyncDispatcher is already running")
        if self._state is DispatcherState.FAILED:
            # A failed run must be stopped (and cleaned up) before a restart.
            raise self._failed()
        self._current_level = None
        reason = native_unavailable_reason()
        if reason is not None:
            raise OnlineTrackerUnavailable(reason)
        try:
            # Probe before acquiring a writer or starting a worker. This does
            # not decide a level or advance the manager's hysteresis state.
            self.resource_manager.get_stats()
        except ResourceMonitorUnavailable as error:
            self._monitor_unavailable(error)
            raise
        self._resource_monitor_status = "available"
        self._resource_monitor_error = ""
        self._failure = None
        self._state = DispatcherState.RUNNING
        self._running = True
        try:
            self._vram_writer = VRAMLevelWriter()
        except Exception as e:
            print(f"[Dispatcher] VRAMLevelWriter init failed: {e}")
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the worker and always clean up; raise ``DispatcherFailed`` afterwards
        when this run's worker had failed."""
        self._running = False
        task, self._worker_task = self._worker_task, None
        try:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            # Every step runs even if an earlier one raises; the first cleanup
            # error is raised only after all of them (#538 D2-AD).
            cleanup_errors: list[Exception] = []
            for step in (self._stop_settle, self._stop_trackers, self._stop_writer):
                try:
                    step()
                except Exception as e:
                    cleanup_errors.append(e)
            failed = self._state is DispatcherState.FAILED
            if self._state is not DispatcherState.CREATED:
                self._state = DispatcherState.STOPPED
        if failed:
            err = self._failed()
            for cleanup_error in cleanup_errors:
                err.add_note(f"cleanup also failed: {cleanup_error!r}")
            raise err
        if cleanup_errors:
            raise cleanup_errors[0]

    def _stop_settle(self) -> None:
        self._discarded_on_stop += self._settle_queue()

    def _stop_trackers(self) -> None:
        while self.trackers:
            _, tracker = self.trackers.popitem()
            del tracker

    def _stop_writer(self) -> None:
        writer, self._vram_writer = self._vram_writer, None
        if writer is not None:
            writer.close()

    def _settle_queue(self) -> int:
        """task_done() the in-flight batch and every queued frame; return how many."""
        settled = len(self._inflight)
        for _ in self._inflight:
            self.queue.task_done()
        self._inflight = []
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.queue.task_done()
            settled += 1
        return settled

    def _fail(self, error: Exception) -> None:
        self._failure = error
        self._state = DispatcherState.FAILED
        self._running = False
        in_batch = len(self._inflight)
        self._failed_batch_frames += in_batch
        self._discarded_on_failure += self._settle_queue() - in_batch
        print(f"❌ [Dispatcher] Worker failed: {error!r}")

    async def put_frame(
        self, stream_id: str, frame: torch.Tensor, timestamp: float
    ) -> None:
        if self._state is DispatcherState.FAILED:
            raise self._failed()
        if self._state is not DispatcherState.RUNNING or not self._running:
            # Also covers stop() in progress (_running is already False).
            raise DispatcherNotRunning(
                f"AsyncDispatcher is {self._state.value}"
                f"{'' if self._running else ' (stopping)'}; put_frame() needs running"
            )
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
            self._dropped_queue_full += 1

    def get_stats(self) -> dict[str, float | int | str]:
        batch_sizes = list(self._batch_sizes)
        queue_wait = list(self._queue_wait_ms)
        infer = list(self._infer_ms)
        track = list(self._track_ms)
        end_to_end = list(self._end_to_end_ms)
        batch_build = list(self._batch_build_ms)
        telemetry = self.telemetry.summary() if self.telemetry is not None else {}

        return {
            "queue_depth": self.queue.qsize(),
            "active_streams": len(self.trackers),
            "state": self._state.value,
            "failure": repr(self._failure) if self._failure is not None else "",
            "dropped_queue_full": self._dropped_queue_full,
            "failed_batch_frames": self._failed_batch_frames,
            "discarded_on_failure": self._discarded_on_failure,
            "discarded_on_stop": self._discarded_on_stop,
            "current_level": self._current_level.name
            if self._current_level is not None
            else "UNKNOWN",
            "resource_monitor_status": self._resource_monitor_status,
            "resource_monitor_error": self._resource_monitor_error,
            # The level only caps the batch size; the tracker itself is never
            # degraded (#537 R-1).
            "tracker_degradation": "none",
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
            "batch_build_ms_mean": round(float(np.mean(batch_build)), 3)
            if batch_build
            else 0.0,
            "batch_build_ms_p95": round(_percentile(batch_build, 95), 3),
            "track_ms_mean": round(float(np.mean(track)), 3) if track else 0.0,
            "track_ms_p95": round(_percentile(track, 95), 3),
            "track_ms_p99": round(_percentile(track, 99), 3),
            "end_to_end_ms_mean": round(float(np.mean(end_to_end)), 3)
            if end_to_end
            else 0.0,
            "end_to_end_ms_p95": round(_percentile(end_to_end, 95), 3),
            "end_to_end_ms_p99": round(_percentile(end_to_end, 99), 3),
            **telemetry,
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
            self._inflight = batch_items
            try:
                decided_at_ns = vram_level_clock_ns()
                try:
                    level = self.resource_manager.decide_degradation_level()
                except ResourceMonitorUnavailable as error:
                    self._monitor_unavailable(error)
                    raise
                self._resource_monitor_status = "available"
                self._resource_monitor_error = ""
                self._current_level = level
                if self._vram_writer is not None:
                    self._vram_writer.write(level, decided_at_ns)
                current_max = (
                    self.max_batch
                    if level < DegradationLevel.FAST_PATH
                    else min(self.max_batch, 2)
                )
                deadline = (
                    time.perf_counter() + max(self.batch_timeout_ms, 0.0) / 1000.0
                )

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
            except Exception as e:
                self._fail(e)
                return

            for _ in batch_items:
                self.queue.task_done()
            self._inflight = []

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

        build_start = time.perf_counter()
        input_tensor = torch.stack(
            [item.frame.contiguous() for item in normalized], dim=0
        ).contiguous()
        batch_build_ms = (time.perf_counter() - build_start) * 1000.0
        self._batch_build_ms.append(batch_build_ms)
        if self.telemetry is not None:
            self.telemetry.observe("dispatcher_batch_build", batch_build_ms)

        infer_start = time.perf_counter()
        with torch.no_grad():
            batch_results = self.detector.detect_batch(
                input_tensor, conf_threshold=self.conf_threshold
            )
        infer_elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
        self._infer_ms.append(infer_elapsed_ms)
        if self.telemetry is not None:
            self.telemetry.observe("dispatcher_infer", infer_elapsed_ms)
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
            tracked_ids, tracked_boxes, tracked_classes = tracker.update(
                boxes, scores, classes
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
        if self.telemetry is not None:
            self.telemetry.observe("dispatcher_track", track_elapsed_ms)
        total_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
        print(
            f"⚡ [Batch] size={batch_size} infer={infer_elapsed_ms:.2f}ms "
            f"track={track_elapsed_ms:.2f}ms total={total_elapsed_ms:.2f}ms"
        )
