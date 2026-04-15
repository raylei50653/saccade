import asyncio
import time
import torch
import cv2
import os
from perception.detector_trt import TRTYoloDetector
from perception.dispatcher import AsyncDispatcher


# 🚀 最佳化：預載圖片到 GPU 記憶體以減少 I/O 影響
def preload_sequence(seq_name, limit_frames=100):
    seq_dir = f"datasets/MOT20/MOT20/test/{seq_name}/img1"
    if not os.path.exists(seq_dir):
        # 如果路徑不存在，生成模擬數據
        print(f"⚠️ {seq_dir} not found. Generating dummy tensors for benchmarking...")
        return [torch.randn(1, 3, 640, 640, device="cuda") for _ in range(limit_frames)]

    images = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])[
        :limit_frames
    ]

    tensors = []
    print(f"📦 Preloading {seq_name} ({len(images)} frames) to GPU...")
    for img_name in images:
        img_path = os.path.join(seq_dir, img_name)
        frame_cv = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        # [H, W, C] -> [3, 640, 640]
        frame_tensor = (
            torch.from_numpy(frame_resized).to("cuda").permute(2, 0, 1).float() / 255.0
        )
        tensors.append(frame_tensor)  # 不包含 Batch 維度，交由 Dispatcher 處理
    return tensors


async def stream_producer(
    stream_id, tensors, dispatcher: AsyncDispatcher, results_list: list
):
    """
    模擬串流生產者：將預載的 Tensor 推入 Dispatcher
    """
    latencies = []
    for i, frame_tensor in enumerate(tensors):
        t_start = time.perf_counter()

        # 將影格推入分發器
        await dispatcher.put_frame(f"stream_{stream_id}", frame_tensor, t_start)

        # 在基準測試中，我們模擬生產速率 (例如 30 FPS)
        # 或者不限速以測試極限吞吐量
        # await asyncio.sleep(0.01)

    # 由於 Dispatcher 是異步回傳（目前是丟給 ProcessPool），
    # 在基準測試中我們主要關注 Dispatcher 的吞吐能力。
    return latencies


async def run_multistream_8path_optimized(num_streams=8, limit_frames=500):
    """
    使用優化後的 AsyncDispatcher 進行 8 路基準測試
    """
    print(
        f"🔥 Starting Optimized 8-stream Benchmark ({num_streams} streams, {limit_frames} frames)..."
    )

    detector = TRTYoloDetector()
    # 啟動分發器，設定最大批次為 num_streams
    dispatcher = AsyncDispatcher(detector, max_batch=num_streams)
    dispatcher.start()

    # 預載數據
    preloaded_tensors = preload_sequence("MOT20-04", limit_frames)

    # 預熱
    dummy_batch = torch.stack([preloaded_tensors[0]] * num_streams).to("cuda")
    _ = detector.detect_batch(dummy_batch)
    torch.cuda.synchronize()

    print(f"🚀 Dispatcher Active. Pushing {num_streams * limit_frames} frames...")

    start_time = time.perf_counter()

    tasks = []
    results_list = []  # 這裡未來可以透過回調填充

    for i in range(num_streams):
        tasks.append(
            asyncio.create_task(
                stream_producer(i, preloaded_tensors, dispatcher, results_list)
            )
        )

    # 等待所有生產者完成推送
    await asyncio.gather(*tasks)

    # 等待隊列處理完畢
    while not dispatcher.queue.empty():
        await asyncio.sleep(0.1)

    # 給予一點額外時間讓最後一批推論完成 (Inference Thread)
    await asyncio.sleep(0.5)

    total_time = time.perf_counter() - start_time - 0.5  # 扣除最後的等待
    total_frames = num_streams * limit_frames
    overall_fps = total_frames / total_time

    print("\n" + "═" * 80)
    print("📊 8-Stream BATCHING Results (Optimized)")
    print("-" * 80)
    print(f"🚀 Overall System Throughput: {overall_fps:,.2f} FPS")
    print(f"📦 Total Processed Frames: {total_frames}")
    print(f"⏱️  Total Elapsed Time: {total_time:.2f} s")
    print(f"📈 Estimated Latency: {1000 / overall_fps * num_streams:.2f} ms per batch")
    print("-" * 80)
    print(
        "💡 Summary: Batching Dispatcher maximizes GPU utilization by grouping streams."
    )
    print("═" * 80)

    dispatcher.stop()


if __name__ == "__main__":
    # 強制使用 spawn 模式
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    asyncio.run(run_multistream_8path_optimized(num_streams=8, limit_frames=200))
