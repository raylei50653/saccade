import asyncio
import time
import torch
from perception.feature_extractor import TRTFeatureExtractor
from perception.embedding_dispatcher import EmbeddingDispatcher


async def stream_producer(
    stream_id, num_objects_per_frame, total_frames, dispatcher: EmbeddingDispatcher
):
    """
    模擬串流生產者：每一影格產生固定數量的物件裁切圖 (Crops)
    """
    for f in range(total_frames):
        # 模擬 224x224 的物件裁切圖
        crops = torch.randn(
            num_objects_per_frame, 3, 224, 224, device="cuda", dtype=torch.float32
        )
        # 模擬元數據
        metadata = [
            {"id": i, "cls": "person", "ts": time.time()}
            for i in range(num_objects_per_frame)
        ]

        await dispatcher.put_crops(f"stream_{stream_id}", crops, metadata)

        # 模擬 30 FPS 的處理速率
        # await asyncio.sleep(0.033)


async def run_siglip_benchmark(num_streams=8, objs_per_stream=4, total_frames=100):
    """
    SigLIP2 並行推論基準測試
    """
    print("🔥 Starting SigLIP2 Parallel Benchmark...")
    print(
        f"📡 Config: {num_streams} streams, {objs_per_stream} objects/stream, {total_frames} frames/stream"
    )

    extractor = TRTFeatureExtractor(max_batch=64)
    dispatcher = EmbeddingDispatcher(extractor, max_batch=64)

    processed_count = 0
    start_time = 0

    # 回調函式：計算處理完畢的向量數量
    def on_embeddings_ready(stream_id, embeddings, metadata):
        nonlocal processed_count
        processed_count += len(embeddings)

    dispatcher.start(callback=on_embeddings_ready)

    # 預熱
    dummy_input = torch.randn(num_streams * objs_per_stream, 3, 224, 224, device="cuda")
    _ = extractor.extract(dummy_input)
    torch.cuda.synchronize()

    print(
        f"🚀 Dispatcher Active. Pushing {num_streams * objs_per_stream * total_frames} total objects..."
    )

    start_time = time.perf_counter()

    tasks = []
    for i in range(num_streams):
        tasks.append(
            asyncio.create_task(
                stream_producer(i, objs_per_stream, total_frames, dispatcher)
            )
        )

    await asyncio.gather(*tasks)

    # 等待隊列清空
    while not dispatcher.queue.empty() or processed_count < (
        num_streams * objs_per_stream * total_frames
    ):
        await asyncio.sleep(0.1)
        # 安全機制：如果超時太久則跳出
        if time.perf_counter() - start_time > 20:
            print("⚠️ Timeout waiting for embeddings.")
            break

    end_time = time.perf_counter()
    duration = end_time - start_time

    total_objects = num_streams * objs_per_stream * total_frames
    throughput = total_objects / duration

    print("\n" + "═" * 80)
    print("📊 SigLIP2 Vector Path Results")
    print("-" * 80)
    print(f"🚀 Overall Throughput: {throughput:,.2f} Embeddings/sec")
    print(f"📦 Total Vectors Indexed: {processed_count}")
    print(f"⏱️  Total Elapsed Time: {duration:.2f} s")
    print(f"📉 Average Latency per Object: {1000 / throughput:.4f} ms")
    print("-" * 80)
    print(
        "💡 Summary: EmbeddingDispatcher aggregates crops from multiple streams for high-batch ViT inference."
    )
    print("═" * 80)

    dispatcher.stop()


if __name__ == "__main__":
    asyncio.run(
        run_siglip_benchmark(num_streams=8, objs_per_stream=4, total_frames=100)
    )
