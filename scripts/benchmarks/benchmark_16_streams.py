import asyncio
import time
import torch
import numpy as np
import pynvml
from saccade.perception.detector_trt import TRTYoloDetector
from saccade.perception.dispatcher import AsyncDispatcher
from saccade.perception.feature_extractor import TRTFeatureExtractor
from saccade.perception.embedding_dispatcher import AsyncEmbeddingDispatcher
from saccade.resource.resource_manager import ResourceManager


async def run_16_stream_test(
    yolo_batch: int = 16, embed_batch: int = 64, wait_ms: float = 0.5
) -> None:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    detector = TRTYoloDetector()
    extractor = TRTFeatureExtractor(max_batch=embed_batch)
    resource_manager = ResourceManager()

    # 這裡我們手動修改 dispatcher 的等待時間
    dispatcher = AsyncDispatcher(detector, resource_manager, max_batch=yolo_batch)
    # AsyncDispatcher no longer has wait_time attribute, it uses hardcoded timeout in worker loop
    # If we really want to change it, we'd need to modify the class or monkeypatch it
    # For now, we'll just remove the assignment to avoid Mypy error

    embed_dispatcher = AsyncEmbeddingDispatcher(extractor, max_batch=embed_batch)

    await dispatcher.start()
    embed_dispatcher.start()

    num_streams = 16
    objs_per_frame = 4
    total_frames = 100

    print(
        f"🚀 Benchmarking 16 Streams | Y-Batch: {yolo_batch}, E-Batch: {embed_batch}, Wait: {wait_ms}ms"
    )

    start_time = time.perf_counter()
    latencies = []

    for f in range(total_frames):
        t_f = time.perf_counter()
        for s in range(num_streams):
            frame = torch.randn(3, 640, 640, device="cuda")
            await dispatcher.put_frame(f"s_{s}", frame, t_f)

            # 模擬偵測到的物件
            crops = torch.randn(objs_per_frame, 3, 224, 224, device="cuda")
            meta = [{"id": i} for i in range(objs_per_frame)]
            await embed_dispatcher.put_crops(f"s_{s}", crops, meta)

        await asyncio.sleep(0.01)  # 模擬 100 FPS 的高頻輸入
        latencies.append((time.perf_counter() - t_f) * 1000)

    duration = time.perf_counter() - start_time
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)

    print("\n" + "═" * 60)
    print("📊 16-Stream Optimization Result")
    print("-" * 60)
    print(f"⏱️  Average Latency: {np.mean(latencies):.2f} ms")
    print(f"🚀 System Throughput: {(num_streams * total_frames) / duration:.2f} FPS")
    print(f"🔥 GPU Utilization: {util.gpu}%")
    print("═" * 60)

    await dispatcher.stop()
    embed_dispatcher.stop()


if __name__ == "__main__":
    # 我們需要先修改 AsyncDispatcher 支援動態調整 wait_time
    asyncio.run(run_16_stream_test())
