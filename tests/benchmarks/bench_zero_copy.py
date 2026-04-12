import asyncio
import time
import os
from perception.detector import Detector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

async def benchmark_yolo():
    print("🚀 [Benchmark] Starting YOLO Perception Performance Test...")
    
    # 1. 初始化
    detector = Detector(device="cuda:0")
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=dummy_video)
    
    if not media.connect():
        print("❌ Failed to connect to media source.")
        return

    print("⏳ Warming up (5s)...")
    time.sleep(5) # 等待串流穩定
    
    # 2. 測試循環
    num_frames = 100
    latencies = []
    
    print(f"📊 Benchmarking {num_frames} frames...")
    
    start_bench = time.perf_counter()
    
    processed_count = 0
    while processed_count < num_frames:
        ret, frame = media.grab_frame()
        if not ret or frame is None:
            await asyncio.sleep(0.01)
            continue
            
        start_frame = time.perf_counter()
        
        # 執行偵測
        _ = detector.detect(frame)
        
        latency = (time.perf_counter() - start_frame) * 1000
        latencies.append(latency)
        processed_count += 1
        
        if processed_count % 20 == 0:
            print(f"  - Processed {processed_count}/{num_frames} frames...")

    total_time = time.perf_counter() - start_bench
    avg_latency = sum(latencies) / len(latencies)
    fps = num_frames / total_time
    
    print("\n✅ Benchmark Complete!")
    print(f"  - Average Inference Latency: {avg_latency:.2f} ms")
    print(f"  - End-to-End Throughput: {fps:.2f} FPS")
    print(f"  - Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_yolo())
