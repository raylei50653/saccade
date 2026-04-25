import asyncio
import threading
import queue
import torch
import time
from typing import List, Tuple, Dict, Optional, Any, Callable, Awaitable
from perception.detector_trt import TRTYoloDetector
from perception.tracking import SmartTracker
from cognition.resource_manager import ResourceManager, DegradationLevel


# 下游結果回呼型別：(stream_id, timestamp, track_ids, boxes, classes) → None
TrackCallback = Callable[
    [str, float, torch.Tensor, torch.Tensor, torch.Tensor],
    Awaitable[None],
]


class AsyncDispatcher:
    """
    Saccade 多路感知分發器 (ReID-Enabled Edition)

    職責：
    1. 收集各路串流影格（put_frame）。
    2. 貪婪動態打包（greedy batching）送入 YOLO TRT 推理。
    3. 對每路串流呼叫其專屬 SmartTracker（GPUByteTracker + Saccade Heartbeat ReID）。
    4. 將追蹤結果透過 on_track_result 回呼送至下游（Redis / Orchestrator）。

    ReID 架構：
    - heartbeat_interval 幀對偵測框提取 SigLIP 2 embedding → 送入 C++ Sinkhorn 融合匹配
    - 其餘幀：純 IoU 匹配，零額外 GPU 開銷
    - 每路串流維護獨立 SmartTracker，ID 空間互不干擾
    """

    def __init__(
        self,
        detector: TRTYoloDetector,
        extractor: Optional[Any] = None,        # TRTFeatureExtractor
        cropper: Optional[Any] = None,          # ZeroCopyCropper
        heartbeat_interval: int = 10,
        conf_threshold: float = 0.25,
        max_batch: int = 8,
        on_track_result: Optional[TrackCallback] = None,
    ) -> None:
        self.detector = detector
        self.extractor = extractor
        self.cropper = cropper
        self.heartbeat_interval = heartbeat_interval
        self.conf_threshold = conf_threshold
        self.max_batch = max_batch
        self.on_track_result = on_track_result

        self.queue: asyncio.Queue[Tuple[str, torch.Tensor, float]] = asyncio.Queue(maxsize=64)
        self.resource_manager = ResourceManager()
        self._running = False

        # per-stream SmartTracker（懶建立）
        self._trackers: Dict[str, SmartTracker] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tracker(self, stream_id: str) -> SmartTracker:
        """取得（或建立）指定串流的 SmartTracker。"""
        if stream_id not in self._trackers:
            self._trackers[stream_id] = SmartTracker(
                extractor=self.extractor,
                cropper=self.cropper,
                heartbeat_interval=self.heartbeat_interval,
            )
            reid_status = "✅ ReID" if self.extractor else "⚠️  IoU-only"
            print(f"🔍 [Dispatcher] SmartTracker created for stream '{stream_id}' ({reid_status})")
        return self._trackers[stream_id]

    async def put_frame(
        self, stream_id: str, frame_tensor: torch.Tensor, timestamp: float
    ) -> None:
        """生產者：將影格推入佇列。佇列滿時 Drop Frame 以維持實時性。"""
        try:
            self.queue.put_nowait((stream_id, frame_tensor, timestamp))
        except asyncio.QueueFull:
            pass

    def start(self) -> None:
        self._running = True
        asyncio.create_task(self._worker_loop())

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """消費者主迴圈：貪婪抓取 → YOLO 推理 → per-stream 追蹤 → 回呼。"""
        reid_info = f"heartbeat={self.heartbeat_interval}f" if self.extractor else "ReID disabled"
        print(f"🚀 [Dispatcher] Worker started. {reid_info}")

        while self._running:
            try:
                # 1. 獲取第一個任務 (阻塞等待)
                items = [self.queue.get(timeout=0.1)]
            except queue.Empty:
                continue

            batch_items = [first_item]

            # 依資源等級動態調整 batch 上限
            level = self.resource_manager.decide_degradation_level()
            current_max = self.max_batch if level < DegradationLevel.FAST_PATH else 2

            while len(batch_items) < current_max:
                try:
                    batch_items.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # 2. YOLO 推理（每幀獨立，確保動態 batch 不超出 engine profile）
            await self._process_batch(batch_items, level)

            for _ in batch_items:
                self.queue.task_done()

    async def _process_batch(
        self,
        batch_items: List[Tuple[str, torch.Tensor, float]],
        level: int,
    ) -> None:
        """對一批影格執行推理 + 追蹤。"""
        loop = asyncio.get_running_loop()

        for stream_id, yolo_input, timestamp in batch_items:
            input_4d = yolo_input.unsqueeze(0)  # [3,H,W] → [1,3,H,W]

            # --- YOLO 偵測 ---
            with torch.no_grad():
                boxes, scores, classes, _ = self.detector.detect(
                    input_4d, conf_threshold=self.conf_threshold
                )

            if boxes.numel() == 0:
                if self.on_track_result:
                    dev = yolo_input.device
                    await self.on_track_result(
                        stream_id, timestamp,
                        torch.empty((0,), dtype=torch.int32, device=dev),
                        torch.empty((0, 4), dtype=torch.float32, device=dev),
                        torch.empty((0,), dtype=torch.int32, device=dev),
                    )
                continue

            # --- FAST_PATH：跳過 L2/ReID，降低 VRAM 壓力 ---
            if level >= DegradationLevel.FAST_PATH:
                tracker = self.get_tracker(stream_id)
                tracker.set_degradation_params(level)
                tracked_ids, tracked_boxes, tracked_classes = tracker.update(
                    boxes, scores, classes
                )
            else:
                # --- 正常路徑：Saccade Heartbeat ReID ---
                tracker = self.get_tracker(stream_id)
                tracked_ids, tracked_boxes, tracked_classes = tracker.update(
                    boxes, scores, classes,
                    frame_tensor=yolo_input,   # [3,640,640]，0~1
                    stream_id=hash(stream_id) & 0x7FFFFFFF,
                )

            # --- 下游回呼 ---
            if self.on_track_result:
                await self.on_track_result(
                    stream_id, timestamp,
                    tracked_ids, tracked_boxes, tracked_classes,
                )
