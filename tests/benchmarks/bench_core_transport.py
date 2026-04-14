import asyncio
import time
import os
import torch
import numpy as np
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

async def benchmark_transport(num_frames=500):
    print(f"🚀 Starting Core Transport Benchmark ({num_frames} frames)...")
    print("Testing C++ 5-Buffer Pool -> Python Zero-Copy latency.")
    
    media = MediaMTXClient(dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))
    if not media.connect(): return

    # 穩健等待第一幀
    for _ in range(100):
        ret, _ = media.grab_tensor()
        if ret: break
        await asyncio.sleep(0.1)
    
    latencies = []
    processed = 0
    
    while processed < num_frames:
        start = time.perf_counter()
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
        
        # 僅測量從 C++ 交付到 Python 拿到 Tensor 的純開銷
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        processed += 1

    print("\n" + "═"*60)
    print(f"📡 Transport Efficiency Report")
    print("═"*60)
    print(f"Average Acquisition Latency: {np.mean(latencies):.6f} ms")
    print(f"P99 Tail Latency:            {np.percentile(latencies, 99):.6f} ms")
    print(f"Standard Deviation:          {np.std(latencies):.6f} ms")
    print(f"Data Path:                   C++ Shared GPU Memory -> PyTorch")
    print("═"*60)
    
    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_transport())
