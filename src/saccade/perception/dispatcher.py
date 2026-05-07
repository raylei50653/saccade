import asyncio
import torch
import time
from collections import OrderedDict
from typing import List, Tuple, Any, Optional, Callable
from saccade.perception.detector_trt import TRTYoloDetector

try:
    from saccade_tracking_ext import GPUByteTracker
except ImportError:
    # Fallback or placeholder if ext not built
    GPUByteTracker = Any
from saccade.resource.resource_manager import (
    ResourceManager,
    DegradationLevel,
    VRAMLevelWriter,
)

# 定義 CallBack 類型以滿足 Mypy
TrackResultCallback = Callable[
    [str, float, torch.Tensor, torch.Tensor, torch.Tensor], Any
]


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
    ):
        self.detector = detector
        self.resource_manager = resource_manager
        self.max_batch = max_batch
        self.conf_threshold = conf_threshold
        self.on_track_result = on_track_result
        self.extractor = extractor
        self.heartbeat_interval = heartbeat_interval
        self.max_streams = max_streams

        self.queue: asyncio.Queue[Tuple[str, torch.Tensor, float]] = asyncio.Queue(
            maxsize=100
        )
        # OrderedDict preserves insertion/access order for O(1) LRU eviction
        self.trackers: OrderedDict[str, Any] = OrderedDict()
        self._running = False
        self._vram_writer: Optional[VRAMLevelWriter] = None

    def _make_tracker(self) -> Any:
        return GPUByteTracker(
            max_objs=100, embedding_dim=768 if self.extractor else 0
        )

    def get_tracker(self, stream_id: str) -> Any:
        if stream_id in self.trackers:
            self.trackers.move_to_end(stream_id)
            return self.trackers[stream_id]

        if len(self.trackers) >= self.max_streams:
            evicted_id, evicted = self.trackers.popitem(last=False)
            del evicted  # triggers ~GPUByteTracker → cudaFree for all GPU buffers
            print(f"[Dispatcher] Evicted tracker '{evicted_id}' (LRU cap={self.max_streams})")

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
        asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._running = False
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
            self.queue.put_nowait((stream_id, frame, timestamp))
        except asyncio.QueueFull:
            pass

    async def _worker_loop(self) -> None:
        reid_info = (
            f"heartbeat={self.heartbeat_interval}f"
            if self.extractor
            else "ReID disabled"
        )
        print(f"🚀 [Dispatcher] Worker started. {reid_info}")

        while self._running:
            try:
                # 1. 獲取第一個任務 (非阻塞等待)
                first_item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            batch_items = [first_item]

            # 依資源等級動態調整 batch 上限，並廣播給 orchestrator
            level = self.resource_manager.decide_degradation_level()
            if self._vram_writer is not None:
                self._vram_writer.write(level)
            current_max = self.max_batch if level < DegradationLevel.FAST_PATH else 2

            while len(batch_items) < current_max:
                try:
                    batch_items.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # 2. YOLO 推理與追蹤
            await self._process_batch(batch_items, level)

            for _ in batch_items:
                self.queue.task_done()

    async def _process_batch(
        self,
        batch_items: List[Tuple[str, torch.Tensor, float]],
        level: int,
    ) -> None:
        start_time = time.perf_counter()

        for stream_id, yolo_input, timestamp in batch_items:
            # --- YOLO 偵測 ---
            with torch.no_grad():
                boxes, scores, classes, _ = self.detector.detect(
                    yolo_input.unsqueeze(0), conf_threshold=self.conf_threshold
                )

            if boxes.numel() == 0:
                if self.on_track_result:
                    dev = yolo_input.device
                    await self.on_track_result(
                        stream_id,
                        timestamp,
                        torch.empty((0,), dtype=torch.int32, device=dev),
                        torch.empty((0, 4), dtype=torch.float32, device=dev),
                        torch.empty((0,), dtype=torch.int32, device=dev),
                    )
                continue

            # --- 追蹤 ---
            tracker = self.get_tracker(stream_id)
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
                    frame_tensor=yolo_input,
                    stream_id=hash(stream_id) & 0x7FFFFFFF,
                )

            if self.on_track_result:
                await self.on_track_result(
                    stream_id, timestamp, tracked_ids, tracked_boxes, tracked_classes
                )

        elapsed = (time.perf_counter() - start_time) * 1000
        if len(batch_items) > 0:
            print(f"⚡ [Batch] Processed {len(batch_items)} frames in {elapsed:.2f}ms")
