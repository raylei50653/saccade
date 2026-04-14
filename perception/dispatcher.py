import asyncio
import torch
import torch.multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any, Optional
from perception.detector_trt import TRTYoloDetector
from cognition.resource_manager import ResourceManager, DegradationLevel

# L2 處理函式 (預期在子進程執行，避免 GIL)
def process_drift_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 這裡未來會實作 CPU-side Cosine Similarity
    # 利用 Ryzen 9 的多核進行 768-d 向量運算
    return {"status": "processed", "stream_id": payload["stream_id"]}

class AsyncDispatcher:
    """
    Saccade 異步分發者 (Industrial Multi-Process Edition)
    
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
        
        # 🛠️ 關鍵：使用 'spawn' 上下文以相容 CUDA
        ctx = mp.get_context('spawn')
        self.executor = ProcessPoolExecutor(max_workers=4, mp_context=ctx) 
        
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
        print(f"🚀 [Dispatcher] Inference Worker active on uvloop. Offloading L2 to ProcessPool.")
        loop = asyncio.get_running_loop()

        while self._running:
            # 1. 等待至少一個影格
            try:
                first_item = await self.queue.get()
            except asyncio.CancelledError:
                break

            batch_items = [first_item]
            
            # 根據資源級別動態調整 max_batch
            level = self.resource_manager.decide_degradation_level()
            current_max = self.max_batch if level < DegradationLevel.FAST_PATH else 4
            
            while len(batch_items) < current_max:
                try:
                    item = self.queue.get_nowait()
                    batch_items.append(item)
                except asyncio.QueueEmpty:
                    break
            
            # 2. 執行 Batch 推理 (GPU)
            # 由於目前的 TensorRT Engine 可能編譯為 Static Batch=1
            # 我們先暫時以 Batch=1 循環處理，或確保 Batch 符合 Engine 限制
            for i, item in enumerate(batch_items):
                stream_id, yolo_input, timestamp = item
                # [3, 640, 640] -> [1, 3, 640, 640]
                input_tensor = yolo_input.unsqueeze(0)
                
                with torch.no_grad():
                    # 偵測單一影格
                    result = self.detector.detect(input_tensor)
                
                # 3. 異步扇出 (Scatter to Sub-processes)
                # 將結果移回 CPU，避免子進程存取 GPU 導致的 IPC 錯誤
                cpu_result = []
                for res in result:
                    if isinstance(res, torch.Tensor):
                        cpu_result.append(res.detach().cpu().numpy())
                    else:
                        cpu_result.append(res)

                payload = {
                    "stream_id": stream_id,
                    "timestamp": timestamp,
                    "results": tuple(cpu_result),
                    "level": level
                }
                
                # 將 L2 任務丟給 Cognition Pool
                loop.run_in_executor(self.executor, process_drift_check, payload)
                
            # 標記 Queue Item 已處理
            for _ in range(len(batch_items)):
                self.queue.task_done()

    def start(self) -> None:
        self._running = True
        asyncio.create_task(self._worker_loop())

    def stop(self) -> None:
        self._running = False
        self.executor.shutdown(wait=False)
