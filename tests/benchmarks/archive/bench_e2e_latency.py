import asyncio
import time
import os
import torch
import numpy as np
from perception.detector_trt import TRTYoloDetector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

async def benchmark_e2e():
    print("🚀 [Benchmark] Starting End-to-End (Full-Link) Latency Analysis...")
    
    # 1. 初始化
    detector = TRTYoloDetector(device="cuda:0")
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=dummy_video)
    
    if not media.connect():
        print("❌ Failed to connect to media source.")
        return

    # 等待第一幀
    for _ in range(100):
        ret, _ = media.grab_tensor()
        if ret: break
        await asyncio.sleep(0.1)
    
    # 2. 測試循環
    num_frames = 200
    e2e_latencies = []
    
    print(f"📊 Analyzing {num_frames} frames for End-to-End latency...")
    
    processed_count = 0
    while processed_count < num_frames:
        # 我們測量從「嘗試抓取影格」到「完成推理」的全過程
        start_time = time.perf_counter()
        
        # Step 1: 獲取 Tensor
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
            
        # Step 2: 執行偵測
        _ = detector.detect(tensor)
        
        # 計算總耗時
        full_latency = (time.perf_counter() - start_time) * 1000
        e2e_latencies.append(full_latency)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"  - Analysing {processed_count}/{num_frames}...")

    avg_e2e = np.mean(e2e_latencies)
    p99_e2e = np.percentile(e2e_latencies, 99)
    std_e2e = np.std(e2e_latencies)
    
    print("\n✅ End-to-End Latency Report (Full-Link):")
    print(f"  - Average E2E Latency: {avg_e2e:.2f} ms")
    print(f"  - P99 Tail Latency:    {p99_e2e:.2f} ms")
    print(f"  - Jitter (Std Dev):    {std_e2e:.2f} ms")
    print(f"  - Resolution:          {tensor.shape[1]}x{tensor.shape[0]}")
    
    # 計算理論最大 FPS
    theoretical_fps = 1000.0 / avg_e2e
    print(f"  - Theoretical Peak:    {theoretical_fps:.2f} FPS")
    
    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_e2e())
