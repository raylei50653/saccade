import asyncio
import time
import subprocess
import pynvml
import json
import os
from datetime import datetime

# 配置
TEST_DURATION = 60 # 測試 60 秒
SAMPLING_INTERVAL = 1.0
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

async def trigger_high_load():
    """模擬短時間內產生大量高價值事件，迫使 VLM 並發工作"""
    import redis.asyncio as redis
    r = redis.from_url(REDIS_URL)
    print("🔥 [Load Generator] Starting to flood event queue...")
    
    # 模擬 20 個高頻發生的並發事件
    for i in range(20):
        event = {
            "event_id": f"stress-test-{i}",
            "timestamp": time.time(),
            "type": "entropy_trigger",
            "metadata": {
                "entropy_value": 1.0,
                "source_path": "stress_test",
                "frame_id": 9999,
                "objects": ["person", "car", "stress_test_object"]
            }
        }
        await r.rpush("saccade:events", json.dumps(event))
        await asyncio.sleep(0.5) # 每 0.5 秒發一個
    
    await r.aclose()
    print("✅ [Load Generator] Flood complete.")

async def monitor_vram_stability():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    print(f"🚀 [VRAM Stability] Starting {TEST_DURATION}s stability test...")
    print(f"⚙️  Current System State: VLM NP=4, Context=32K, Zero-Copy Enabled.")
    
    start_time = time.time()
    vram_history = []
    
    # 同步啟動負載生成
    load_task = asyncio.create_task(trigger_high_load())
    
    while time.time() - start_time < TEST_DURATION:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024**3
        vram_history.append(used_gb)
        
        # 每秒印出一次狀態
        elapsed = int(time.time() - start_time)
        bar = "█" * int(used_gb) + "░" * (12 - int(used_gb))
        print(f"[{elapsed:02d}s] VRAM: {used_gb:.2f} GB |{bar}|")
        
        await asyncio.sleep(SAMPLING_INTERVAL)
    
    pynvml.nvmlShutdown()
    
    # 分析結果
    min_vram = min(vram_history)
    max_vram = max(vram_history)
    avg_vram = sum(vram_history) / len(vram_history)
    
    print(f"\n📊 --- [ Stability Test Results ] ---")
    print(f"  - Minimum VRAM: {min_vram:.2f} GB")
    print(f"  - Maximum VRAM: {max_vram:.2f} GB")
    print(f"  - Average VRAM: {avg_vram:.2f} GB")
    print(f"  - Peak Fluctuation: {max_vram - min_vram:.2f} GB")
    
    if max_vram > 11.5:
        print("🚨 CRITICAL: VRAM almost exhausted! OOM Risk detected.")
    elif max_vram > 10.0:
        print("⚠️ WARNING: High VRAM usage observed.")
    else:
        print("✅ STABLE: VRAM usage within safe limits.")

if __name__ == "__main__":
    asyncio.run(monitor_vram_stability())
