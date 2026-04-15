import asyncio
import threading
import queue
import torch
from typing import List, Tuple, Dict, Any, Optional, Callable
from perception.feature_extractor import TRTFeatureExtractor


class EmbeddingDispatcher:
    """
    Saccade 高效能語義分發器 (Vector Path Optimizer)

    1. 接收來自各路串流的物件裁切圖 (Crops)。
    2. 專屬 Embedding Worker Thread 執行 SigLIP2 推論。
    3. 實現 Aggressive Batching：將多個串流的物件打包成單一大 Batch (例如 32)。
    4. 將產出的 768-d 向量異步回傳至 Orchestrator。
    """

    def __init__(self, extractor: TRTFeatureExtractor, max_batch: int = 32) -> None:
        self.extractor = extractor
        self.max_batch = max_batch
        # 使用線程安全隊列存放待嵌入的物件資訊
        self.queue: queue.Queue[Tuple[str, torch.Tensor, List[Dict[str, Any]]]] = (
            queue.Queue(maxsize=256)
        )
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None

        # 回調函式 (用於將向量傳回 Orchestrator/Redis)
        self.on_embeddings_ready: Optional[
            Callable[[str, Any, List[Dict[str, Any]]], Any]
        ] = None

    async def put_crops(
        self, stream_id: str, crops: torch.Tensor, metadata: List[Dict[str, Any]]
    ) -> None:
        """生產者 (Asyncio Context)：將裁切圖與元數據推入隊列"""
        if crops.size(0) == 0:
            return

        try:
            # metadata 應包含物件 ID, 類別, 時間戳等
            self.queue.put_nowait((stream_id, crops, metadata))
        except queue.Full:
            # 負載過重時捨棄 Vector Path (優先保證 Fast Path)
            pass

    def _embedding_worker(self) -> None:
        """背景嵌入線程 (Synchronous Context)"""
        print("🧠 [EmbeddingDispatcher] SigLIP2 Worker active.")
        torch.cuda.set_device(self.extractor.device)

        # 🚀 建立 L2 專用 CUDA Stream
        l2_stream = torch.cuda.Stream(device=self.extractor.device)  # type: ignore

        while self._running:
            try:
                # 1. 獲取第一組任務 (阻塞等待)
                items = [self.queue.get(timeout=0.2)]
            except queue.Empty:
                continue

            # 2. 積極批次化 (Aggressive Batching)
            # 收集儘量多的物件裁切圖，直到達到 max_batch 或隊列清空
            current_batch_crops = [items[0][1]]
            current_batch_meta = [items[0][2]]
            current_batch_streams = [items[0][0]]

            total_objects = items[0][1].size(0)

            while total_objects < self.max_batch:
                try:
                    s_id, crops, meta = self.queue.get_nowait()
                    current_batch_crops.append(crops)
                    current_batch_meta.append(meta)
                    current_batch_streams.append(s_id)
                    total_objects += crops.size(0)
                except queue.Empty:
                    break

            # 3. 執行 SigLIP2 批次推論 (在專屬 Stream 內執行)
            with torch.cuda.stream(l2_stream):
                # 將所有裁切圖合併為一個大 Tensor [B, 3, 224, 224]
                batch_tensor = torch.cat(current_batch_crops, dim=0).contiguous()

                # 如果總數超過 max_batch，進行切片 (這通常不會發生，因為我們在循環中控制了)
                if batch_tensor.size(0) > self.max_batch:
                    batch_tensor = batch_tensor[: self.max_batch]
                    total_objects = self.max_batch

                with torch.no_grad():
                    # GPU -> GPU 零拷貝推論
                    embeddings = self.extractor.extract(batch_tensor, stream=l2_stream)
                    # 確保 GPU 計算完成
                    l2_stream.synchronize()
                    # 搬移至 CPU Buffer
                    cpu_embeddings = embeddings.cpu().numpy()

            # 4. 解包並回發結果
            cursor = 0
            for i, stream_id in enumerate(current_batch_streams):
                num_objs = current_batch_meta[i].__len__()
                if cursor + num_objs > total_objects:
                    num_objs = total_objects - cursor

                stream_embeddings = cpu_embeddings[cursor : cursor + num_objs]
                stream_meta = current_batch_meta[i][:num_objs]

                if self.on_embeddings_ready and self._loop:
                    # 🚀 修正：使用內部函式包裝以確保 asyncio.create_task 被正確呼叫
                    def _create_task(
                        s: str = stream_id,
                        e: Any = stream_embeddings,
                        m: List[Dict[str, Any]] = stream_meta,
                    ) -> None:
                        if self.on_embeddings_ready:
                            asyncio.create_task(self.on_embeddings_ready(s, e, m))

                    self._loop.call_soon_threadsafe(_create_task)

                cursor += num_objs

            # 標記任務完成
            for _ in range(len(current_batch_streams)):
                self.queue.task_done()

    def start(
        self, callback: Optional[Callable[[str, Any, List[Dict[str, Any]]], Any]] = None
    ) -> None:
        self.on_embeddings_ready = callback
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._worker_thread = threading.Thread(
            target=self._embedding_worker, daemon=True
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)

            self._worker_thread.join(timeout=1.0)
