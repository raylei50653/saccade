import asyncio
import time
import os
from perception.detector_trt import TRTYoloDetector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

async def benchmark_yolo():
    print("🚀 [Benchmark] Starting Zero-Copy C++ Pool Latency Test...")
    
    # 1. 初始化 (優先使用高效能偵測器)
    detector = TRTYoloDetector(device="cuda:0")
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=dummy_video)
    
    if not media.connect():
        print("❌ Failed to connect to media source.")
        return

    print("⏳ Waiting for GStreamer C++ Pool to stabilize...")
    # 等待第一幀
    for _ in range(100):
        ret, _ = media.grab_tensor()
        if ret: break
        await asyncio.sleep(0.1)
    
    # 2. 測試循環
    num_frames = 200
    inference_latencies = []
    acquisition_latencies = []
    
    print(f"📊 Benchmarking {num_frames} frames via C++ Buffer Pool...")
    
    start_total = time.perf_counter()
    processed_count = 0
    
    while processed_count < num_frames:
        # Step 1: 測量獲取延遲 (Acquisition Latency)
        acq_start = time.perf_counter()
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
        acq_latency = (time.perf_counter() - acq_start) * 1000
        
        # Step 2: 測量偵測延遲 (Inference Latency)
        inf_start = time.perf_counter()
        _ = detector.detect(tensor)
        inf_latency = (time.perf_counter() - inf_start) * 1000
        
        acquisition_latencies.append(acq_latency)
        inference_latencies.append(inf_latency)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"  - Processed {processed_count}/{num_frames} frames...")

    total_duration = time.perf_counter() - start_total
    
    avg_acq = sum(acquisition_latencies) / len(acquisition_latencies)
    avg_inf = sum(inference_latencies) / len(inference_latencies)
    fps = num_frames / total_duration
    
    print("\n✅ Benchmark Complete (C++ Pool + Zero-Copy)!")
    print(f"  - Average Acquisition Latency: {avg_acq:.4f} ms  <-- 零拷貝優勢指標")
    print(f"  - Average Inference Latency:   {avg_inf:.2f} ms")
    print(f"  - System Throughput:           {fps:.2f} FPS")
    print(f"  - Tensor Device:               {tensor.device}")
    
    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_yolo())
