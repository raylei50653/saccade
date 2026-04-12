import asyncio
import time
import subprocess
import pynvml

async def monitor_vram(duration: int):
    """即時監測 VRAM 變化"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    start_time = time.time()
    max_used = 0.0
    
    print(f"📈 [VRAM Monitor] Monitoring started for {duration}s...")
    while time.time() - start_time < duration:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024**3
        max_used = max(max_used, used_gb)
        await asyncio.sleep(1)
    
    pynvml.nvmlShutdown()
    return max_used

async def run_stress_test():
    print("🚀 [Stress Test] Starting 20s Demo Stress Test (-c 32768, -np 4)...")
    
    # 1. 確保相機開啟
    subprocess.run(["./scripts/saccade", "camera-on"], check=True)
    
    # 2. 同步運行監測與等待
    monitor_task = asyncio.create_task(monitor_vram(20))
    
    print("⏳ Running demo...")
    await asyncio.sleep(20)
    
    # 3. 關閉相機
    subprocess.run(["./scripts/saccade", "camera-off"], check=True)
    
    max_vram = await monitor_task
    print("\n📊 --- [ Stress Test Results ] ---")
    print(f"  - Peak VRAM Usage: {max_vram:.2f} GB")
    print("  - System Context: 32768")
    print("  - Slots (NP): 4")
    
    # 4. 擷取日誌觀察瓶頸
    print("\n🔍 [Log Analysis] Recent Orchestrator Logs:")
    log_output = subprocess.check_output(
        ["journalctl", "--user", "-u", "yolo-orchestrator", "-n", "10", "--no-pager"]
    ).decode("utf-8")
    print(log_output)

if __name__ == "__main__":
    asyncio.run(run_stress_test())
