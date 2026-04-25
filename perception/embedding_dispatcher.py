import asyncio
import torch
from typing import List, Tuple, Optional
from perception.feature_extractor import TRTFeatureExtractor

class AsyncEmbeddingDispatcher:
    """
    Saccade SigLIP 2 併行調度器
    
    核心目標：利用獨立的 CUDA Stream 執行特徵提取，確保 L2 任務不阻塞 L1 感知主循環。
    """
    
    def __init__(self, extractor: TRTFeatureExtractor, max_batch: int = 16):
        self.extractor = extractor
        self.max_batch = max_batch
        self.queue: asyncio.Queue[Tuple[torch.Tensor, List[int], asyncio.Future]] = asyncio.Queue(maxsize=64)
        
        # 🛠️ 建立獨立的 CUDA Stream (若裝置支援)
        self.device = getattr(extractor, "device", "cuda:0")
        self.stream = torch.cuda.Stream(device=self.device) if "cuda" in self.device and torch.cuda.is_available() else None
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def submit(self, crops: torch.Tensor, track_ids: List[int]) -> torch.Tensor:
        """
        提交一批裁切好的物件圖進行特徵提取。
        """
        if crops.size(0) == 0:
            return torch.empty((0, 768), device=crops.device)
            
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((crops, track_ids, future))
        return await future

    async def _worker_loop(self) -> None:
        """
        背景 Worker：負責從隊列中抓取任務並執行推理。
        """
        print(f"🚀 [EmbeddingDispatcher] Parallel Worker started on Device: {self.device}, Stream: {self.stream}")
        
        while self._running:
            try:
                # 抓取第一個任務
                crops, ids, future = await self.queue.get()
                
                batch_crops = [crops]
                batch_futures = [future]
                
                while len(batch_crops[0]) < self.max_batch:
                    try:
                        next_crops, next_ids, next_future = self.queue.get_nowait()
                        batch_crops.append(next_crops)
                        batch_futures.append(next_future)
                    except (asyncio.QueueEmpty, IndexError):
                        break
                
                input_tensor = torch.cat(batch_crops, dim=0) if len(batch_crops) > 1 else batch_crops[0]
                
                # 執行推理 (根據是否有 Stream 決定是否異步)
                if self.stream:
                    with torch.cuda.stream(self.stream):
                        features = self.extractor.extract(input_tensor)
                        event = torch.cuda.Event()
                        event.record(self.stream)
                        while not event.query():
                            await asyncio.sleep(0.001)
                else:
                    features = self.extractor.extract(input_tensor)
                
                # 分發結果
                start_idx = 0
                for i, future_obj in enumerate(batch_futures):
                    num_obj = batch_crops[i].size(0)
                    future_obj.set_result(features[start_idx : start_idx + num_obj].detach().clone())
                    start_idx += num_obj
                
                for _ in range(len(batch_futures)):
                    self.queue.task_done()
                    
            except Exception as e:
                print(f"❌ [EmbeddingDispatcher] Error: {e}")
                if 'future' in locals() and not future.done():
                    future.set_exception(e)

    def start(self) -> None:
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
