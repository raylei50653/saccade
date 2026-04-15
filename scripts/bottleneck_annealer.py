import asyncio
import time
import torch
import numpy as np
import pynvml
from typing import Tuple, Dict, Any, List
from perception.detector_trt import TRTYoloDetector
from perception.dispatcher import AsyncDispatcher
from perception.feature_extractor import TRTFeatureExtractor
from perception.embedding_dispatcher import EmbeddingDispatcher
from storage.redis_cache import RedisCache


class BottleneckAnnealer:
    """
    Saccade 效能瓶頸退火探測器

    透過動態擾動系統負載與參數，尋找系統崩潰的臨界點 (Phase Transition)。
    """

    def __init__(self) -> None:
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # 初始化全鏈路
        self.detector = TRTYoloDetector()
        self.extractor = TRTFeatureExtractor(max_batch=64)
        self.redis_cache = RedisCache()

        self.results: List[Dict[str, Any]] = []

    def get_gpu_util(self) -> Tuple[int, float]:
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        return util.gpu, (mem.used / mem.total) * 100

    async def run_step(
        self, num_streams: int, objects_per_frame: int, yolo_batch: int, embed_batch: int
    ) -> Dict[str, Any]:
        """
        執行單一負載步進，測量系統穩定性
        """
        dispatcher = AsyncDispatcher(self.detector, max_batch=yolo_batch)
        embed_dispatcher = EmbeddingDispatcher(self.extractor, max_batch=embed_batch)

        dispatcher.start()
        embed_dispatcher.start()
        await self.redis_cache.connect()

        print(
            f"🌡️ Testing: Streams={num_streams}, Objs={objects_per_frame}, Y-Batch={yolo_batch}, E-Batch={embed_batch}"
        )

        latencies = []

        # 模擬高壓推送
        for f in range(50):  # 測試 50 幀
            t_f = time.perf_counter()
            for s in range(num_streams):
                # 模擬 L1 影格
                frame = torch.randn(3, 640, 640, device="cuda")
                await dispatcher.put_frame(f"s_{s}", frame, t_f)

                # 模擬 L2 裁切圖 (直接推入以測試後端壓力)
                crops = torch.randn(objects_per_frame, 3, 224, 224, device="cuda")
                meta = [{"id": i} for i in range(objects_per_frame)]
                await embed_dispatcher.put_crops(f"s_{s}", crops, meta)

            await asyncio.sleep(0.01)  # 100 FPS
            latencies.append((time.perf_counter() - t_f) * 1000)

        gpu_util, mem_usage = self.get_gpu_util()

        # 取得 Redis 堆積情況
        queue_len = 0
        if self.redis_cache.client is not None:
            try:
                info = await self.redis_cache.client.xinfo_stream(
                    self.redis_cache.stream_name
                )
                queue_len = info.get("length", 0)
            except Exception:
                queue_len = 0

        avg_lat = np.mean(latencies)

        print(
            f"📊 Result: Latency={avg_lat:.2f}ms, GPU={gpu_util}%, Mem={mem_usage:.1f}%, Queue={queue_len}"
        )

        dispatcher.stop()
        embed_dispatcher.stop()

        return {
            "latency": avg_lat,
            "gpu": gpu_util,
            "queue": queue_len,
            "config": (num_streams, objects_per_frame, yolo_batch, embed_batch),
        }

    async def explore(self) -> None:
        """
        退火探測循環：從低負載到崩潰
        """
        print("🚀 Starting Bottleneck Annealing Exploration...")

        # 我們保持 Batch Size 最佳化，逐步增加串流數量
        for n_streams in [1, 2, 4, 8, 16, 24, 32]:
            res = await self.run_step(n_streams, 4, 8, 32)
            self.results.append(res)

            # 瓶頸判定邏輯
            if res["latency"] > 100:  # 延遲超過 100ms
                print(f"💥 BOTTLE NECK DETECTED at {n_streams} streams!")
                if res["gpu"] > 95:
                    print(
                        "🔍 Primary Cause: GPU COMPUTE BOUND (L1/L2 Models saturated)"
                    )
                elif res["queue"] > 1000:
                    print(
                        "🔍 Primary Cause: I/O STORAGE BOUND (Redis/ChromaDB backlog)"
                    )
                else:
                    print(
                        "🔍 Primary Cause: CPU/GIL BOUND (Python event loop overhead)"
                    )
                break

            await asyncio.sleep(1)


if __name__ == "__main__":
    annealer = BottleneckAnnealer()
    asyncio.run(annealer.explore())
