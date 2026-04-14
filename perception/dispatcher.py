import asyncio
import torch
import time
from typing import List, Tuple, Dict, Any, Optional
from perception.detector_trt import TRTYoloDetector
from cognition.resource_manager import ResourceManager, DegradationLevel

class AsyncDispatcher:
    """
    Saccade 異步分發者 (Async-Batching Dispatcher)
    
    1. 收集來自不同 L1 Producer 的影格數據。
    2. 打包為動態 Batch 並提交給單一 TensorRT Engine 推理。
    3. 將結果異步分發回各串流處理任務。
    """
    
    def __init__(self, detector: TRTYoloDetector, max_batch: int = 8):
        self.detector = detector
        self.max_batch = max_batch
        self.queue: asyncio.Queue[Tuple[str, torch.Tensor, float]] = asyncio.Queue(maxsize=32)
        self.resource_manager = ResourceManager()
        self._running = False
        
    async def put_frame(self, stream_id: str, frame_tensor: torch.Tensor, timestamp: float) -> None:
        """生產者：將影格放入隊列"""
        try:
            # 這裡我們使用 nowait 因為 L1 Callback 不能被阻塞
            self.queue.put_nowait((stream_id, frame_tensor, timestamp))
        except asyncio.QueueFull:
            # 如果隊列滿了，執行 Drop Frame 以維持實時性
            pass

    async def _worker_loop(self) -> None:
        """消費者：貪婪抓取並執行 Batch 推理"""
        print(f"🚀 [Dispatcher] Inference Worker started (N_opt={self.max_batch}).")
        
        while self._running:
            # 1. 等待至少一個影格
            first_item = await self.queue.get()
            batch_items = [first_item]
            
            # 2. 貪婪抓取 (Greedy Pull) 剩餘可用的影格，直到達到 max_batch
            # 根據 L6 資源級別動態調整 max_batch
            level = self.resource_manager.decide_degradation_level()
            current_max = self.max_batch if level < DegradationLevel.FAST_PATH else 4
            
            while len(batch_items) < current_max:
                try:
                    item = self.queue.get_nowait()
                    batch_items.append(item)
                except asyncio.QueueEmpty:
                    break
            
            # 3. 打包 Batch Tensor
            stream_ids = [item[0] for item in batch_items]
            tensors = [item[1] for item in batch_items]
            timestamps = [item[2] for item in batch_items]
            
            # [N, 3, 640, 640]
            batch_tensor = torch.stack(tensors)
            
            # 4. 執行 Batch 推理 (Zero-Copy)
            with torch.no_grad():
                results = self.detector.detect_batch(batch_tensor)
            
            # 5. 分發結果 (這裡可以進一步觸發異步的回調函數)
            for i, result in enumerate(results):
                # result = (boxes, scores, cls_ids, ids)
                # 此處我們暫時以 Log 展示，實戰中應呼叫 stream_ids[i] 對應的 L2 處理器
                pass
                
            # 標記 Queue Item 已處理
            for _ in range(len(batch_items)):
                self.queue.task_done()

    def start(self) -> None:
        self._running = True
        asyncio.create_task(self._worker_loop())

    def stop(self) -> None:
        self._running = False
