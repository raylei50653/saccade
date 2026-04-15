import asyncio
import threading
import queue
import torch
import time
from typing import Tuple, Any, Optional, Callable
from perception.detector_trt import TRTYoloDetector
from cognition.resource_manager import ResourceManager, DegradationLevel


class AsyncDispatcher:
    """
    Saccade 高效能異步分發器 (Thread-Safe & Batching Optimized)

    1. 使用 queue.Queue 橋接 asyncio 與 Inference Thread。
    2. 專屬 Inference Thread 執行同步 GPU 運算，不阻塞事件循環。
    3. 觸發回調函式進行後續 L2 處理。
    """

    def __init__(self, detector: TRTYoloDetector, max_batch: int = 8) -> None:
        self.detector = detector
        self.max_batch = max_batch
        # 使用線程安全隊列
        self.queue: queue.Queue[Tuple[str, torch.Tensor, float]] = queue.Queue(
            maxsize=128
        )
        self.resource_manager = ResourceManager()
        self._running = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self.on_finished: Optional[Callable[[str, Any, Any, float], Any]] = (
            None  # 回調函式
        )

        # 🚀 可配置的批次等待時間 (預設 1ms)
        self.wait_time = 0.001

    async def put_frame(
        self, stream_id: str, frame_tensor: torch.Tensor, timestamp: float
    ) -> None:
        """生產者 (Asyncio Context)：將影格推入隊列"""
        try:
            # queue.Queue.put_nowait 是線程安全的
            self.queue.put_nowait((stream_id, frame_tensor, timestamp))
        except queue.Full:
            pass

    def _inference_worker(self) -> None:
        """背景推論線程 (Synchronous Context)"""
        print("🔥 [Dispatcher] Native Inference Worker active.")
        torch.cuda.set_device(self.detector.device)

        # 🚀 建立 L1 專用 CUDA Stream，與預設 Stream 脫鉤
        l1_stream = torch.cuda.Stream(device=self.detector.device)  # type: ignore

        while self._running:
            try:
                # 1. 獲取第一個任務 (阻塞等待)
                items = [self.queue.get(timeout=0.1)]
            except queue.Empty:
                continue

            # 🚀 積極批次化：使用可配置的等待窗口
            if self.wait_time > 0:
                time.sleep(self.wait_time)

            # 2. 獲取更多任務以填滿 Batch
            level = self.resource_manager.decide_degradation_level()
            current_max = self.max_batch if level < DegradationLevel.FAST_PATH else 2

            while len(items) < current_max:
                try:
                    items.append(self.queue.get_nowait())
                except queue.Empty:
                    break

            # 3. 批次推論 (在專屬 Stream 內執行)
            stream_ids, tensors, timestamps = zip(*items)

            with torch.cuda.stream(l1_stream):
                batch_tensor = (
                    torch.stack(tensors).to(self.detector.device).contiguous()
                )

                with torch.no_grad():
                    batch_results = self.detector.detect_batch(
                        batch_tensor, stream=l1_stream
                    )

                # 確保 GPU 運算完成才將結果往下傳遞
                l1_stream.synchronize()

            # 4. 結果分發
            for i, result in enumerate(batch_results):
                if self.on_finished and self._loop:
                    # 🚀 觸發 L1 -> L2 橋接回調
                    def _create_task(
                        s: str = stream_ids[i],
                        r: Tuple[
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                            Optional[torch.Tensor],
                        ] = result,
                        t: torch.Tensor = tensors[i],
                        ts: float = timestamps[i],
                    ) -> None:
                        if self.on_finished:
                            asyncio.create_task(self.on_finished(s, r, t, ts))

                    self._loop.call_soon_threadsafe(_create_task)

            # 標記 Queue 任務完成
            for _ in range(len(items)):
                self.queue.task_done()

    def start(
        self,
        callback: Optional[
            Callable[
                [
                    str,
                    Tuple[
                        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
                    ],
                    torch.Tensor,
                    float,
                ],
                Any,
            ]
        ] = None,
    ) -> None:
        self.on_finished = callback
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._worker_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
