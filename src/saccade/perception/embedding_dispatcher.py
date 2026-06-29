import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch

# 這裡不匯入 TRTFeatureExtractor 以避免循環匯入
# 但在類型註解中使用 Any 或字串參考


class AsyncEmbeddingDispatcher:
    """
    Saccade SigLIP 2 併行調度器

    核心目標：利用獨立的 CUDA Stream 執行特徵提取，確保 L2 任務不阻塞 L1 感知主循環。
    支援：
    1. 直接提交裁切好的 Tensor (submit/put_crops)
    2. 提交原圖與 Bbox，由調度器異步裁切 (需提供 cropper)
    """

    def __init__(
        self,
        extractor: Any,
        cropper: Optional[Any] = None,
        max_batch: int = 16,
        on_embeddings_ready: Optional[
            Callable[[str, Any, List[Dict[str, Any]]], Any]
        ] = None,
    ) -> None:
        self.extractor = extractor
        self.cropper = cropper
        self.max_batch = max_batch
        self.on_embeddings_ready = on_embeddings_ready
        # 隊列中存儲：(image_or_crops, boxes_or_metadata, future, stream_id, extra_metadata)
        self.queue: asyncio.Queue[
            Tuple[torch.Tensor, Any, Optional[asyncio.Future[Any]], Optional[str], Any]
        ] = asyncio.Queue(maxsize=64)

        # 🛠️ 建立獨立的 CUDA Stream (若裝置支援)
        self.device = getattr(extractor, "device", "cuda:0")
        self.stream = (
            cast(Any, torch.cuda.Stream)(device=self.device)
            if "cuda" in self.device and torch.cuda.is_available()
            else None
        )
        self._running = False
        self._worker_task: Optional[asyncio.Task[None]] = None

    async def submit(self, crops: torch.Tensor, metadata: Any = None) -> torch.Tensor:
        """
        提交一批裁切好的物件圖進行特徵提取，並等待結果。
        """
        if crops.size(0) == 0:
            feature_dim = int(getattr(self.extractor, "feature_dim", 768))
            return torch.empty((0, feature_dim), device=crops.device)

        future: asyncio.Future[torch.Tensor] = (
            asyncio.get_running_loop().create_future()
        )
        await self.queue.put((crops, metadata, future, None, None))
        return await future

    async def put_crops(
        self, stream_id: str, crops: torch.Tensor, metadata: List[Dict[str, Any]]
    ) -> None:
        """
        非阻塞提交（Fire-and-forget 模式，透過 callback 返回結果）。
        """
        if crops.size(0) == 0:
            return
        await self.queue.put((crops, metadata, None, stream_id, None))

    async def _worker_loop(self) -> None:
        """
        背景 Worker：負責從隊列中抓取任務並執行推理。
        """
        print(
            f"🚀 [EmbeddingDispatcher] Parallel Worker started on Device: {self.device}, Stream: {self.stream}"
        )

        while self._running:
            try:
                # 抓取第一個任務
                (
                    img_or_crops,
                    boxes_or_meta,
                    future,
                    stream_id,
                    extra,
                ) = await self.queue.get()

                # 如果有 cropper 且傳入的是原圖 (dim=3 or dim=4 but large)，則進行裁切
                # 這裡簡化處理：如果 img_or_crops 是 [3, H, W] 且 boxes_or_meta 是 [N, 4]
                if (
                    self.cropper is not None
                    and img_or_crops.dim() == 3
                    and boxes_or_meta is not None
                    and isinstance(boxes_or_meta, torch.Tensor)
                ):
                    # 執行裁切
                    crops = self.cropper.process(
                        img_or_crops.unsqueeze(0), boxes_or_meta
                    )
                    meta = extra  # 在這種模式下，metadata 可能存在 extra 中
                else:
                    crops = img_or_crops
                    meta = boxes_or_meta

                batch_crops = [crops]
                batch_metas = [meta]
                batch_futures = [future]
                batch_stream_ids = [stream_id]

                total_len = crops.size(0)
                while total_len < self.max_batch:
                    try:
                        # 嘗試抓取更多任務來湊 Batch
                        n_img, n_boxes, n_future, n_sid, n_extra = (
                            self.queue.get_nowait()
                        )
                        if (
                            self.cropper is not None
                            and n_img.dim() == 3
                            and isinstance(n_boxes, torch.Tensor)
                        ):
                            n_crops = self.cropper.process(n_img.unsqueeze(0), n_boxes)
                            n_meta = n_extra
                        else:
                            n_crops = n_img
                            n_meta = n_boxes

                        batch_crops.append(n_crops)
                        batch_metas.append(n_meta)
                        batch_futures.append(n_future)
                        batch_stream_ids.append(n_sid)
                        total_len += n_crops.size(0)
                    except (asyncio.QueueEmpty, IndexError):
                        break

                if total_len == 0:
                    continue

                input_tensor = (
                    torch.cat(batch_crops, dim=0)
                    if len(batch_crops) > 1
                    else batch_crops[0]
                )

                # 執行推理 (根據是否有 Stream 決定是否異步)
                if self.stream:
                    with torch.cuda.stream(self.stream):
                        features = self.extractor.extract(input_tensor)
                        event = cast(Any, torch.cuda.Event)()
                        event.record(self.stream)
                        while not event.query():
                            await asyncio.sleep(0.001)
                else:
                    features = self.extractor.extract(input_tensor)

                # 分發結果
                start_idx = 0
                for i, (future_obj, s_id, m) in enumerate(
                    zip(batch_futures, batch_stream_ids, batch_metas)
                ):
                    num_obj = batch_crops[i].size(0)
                    res = features[start_idx : start_idx + num_obj].detach().clone()

                    if future_obj is not None:
                        # 支援傳回 (embeddings, metadata) 元組以相容舊代碼
                        if s_id is not None:  # 如果有 s_id 代表是某種特殊模式
                            future_obj.set_result((res, batch_crops[i], m))
                        else:
                            future_obj.set_result(res)

                    if s_id is not None and self.on_embeddings_ready is not None:
                        res_np = res.cpu().numpy()
                        if asyncio.iscoroutinefunction(self.on_embeddings_ready):
                            asyncio.create_task(
                                self.on_embeddings_ready(s_id, res_np, m)
                            )
                        else:
                            self.on_embeddings_ready(s_id, res_np, m)

                    start_idx += num_obj

                for _ in range(len(batch_futures)):
                    self.queue.task_done()

            except Exception as e:
                print(f"❌ [EmbeddingDispatcher] Error: {e}")
                if "future" in locals() and future is not None and not future.done():
                    future.set_exception(e)

    def start(self) -> None:
        if self._running and self._worker_task is not None:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()


EmbeddingDispatcher = AsyncEmbeddingDispatcher
