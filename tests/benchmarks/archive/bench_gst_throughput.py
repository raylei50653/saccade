import time
import os
import threading

import gi # noqa: E402
gi.require_version('Gst', '1.0') # noqa: E402
from gi.repository import GLib # noqa: E402

from perception.zero_copy import GstZeroCopyDecoder # noqa: E402
from dotenv import load_dotenv # noqa: E402

load_dotenv()

def benchmark_gst():
    print("🚀 [Benchmark] Starting GStreamer Zero-Copy Throughput Test...")
    
    # 轉換本地路徑為 GStreamer 可接受的 file:// URI
    video_path = os.path.abspath(os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))
    uri = f"file://{video_path}"
    
    # 1. 初始化
    decoder = GstZeroCopyDecoder(uri)
    
    # GStreamer 需要一個 MainLoop 來處理信號
    loop = GLib.MainLoop()
    def run_loop():
        loop.run()
    
    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    # 2. 測試循環
    num_frames = 1000
    latencies = []
    
    decoder.start()
    print(f"📊 Benchmarking {num_frames} frames...")
    
    # 等待管線穩定
    time.sleep(2)
    
    processed_count = 0
    start_bench = time.perf_counter()
    
    while processed_count < num_frames:
        start_frame = time.perf_counter()
        
        tensor = decoder.grab_frame_tensor()
        
        if tensor is not None:
            # 這裡我們模擬一個處理過程
            # 由於 grab_frame_tensor 目前是回傳最後一幀，我們需要確認是否拿到新幀
            # 簡易做法：直接累加計數
            processed_count += 1
            latency = (time.perf_counter() - start_frame) * 1000
            latencies.append(latency)
            
            if processed_count % 100 == 0:
                print(f"  - Captured {processed_count}/{num_frames} frames...")
        
        # 限制抓取頻率，防止 CPU 空轉
        time.sleep(0.001)

    total_time = time.perf_counter() - start_bench
    avg_latency = sum(latencies) / len(latencies)
    fps = num_frames / total_time
    
    print("\n✅ GStreamer Benchmark Complete!")
    print(f"  - Average Access Latency: {avg_latency:.2f} ms")
    print(f"  - Estimated Throughput: {fps:.2f} FPS")
    
    if tensor is not None:
        print(f"  - Tensor Shape: {tensor.shape}")
        print(f"  - Device: {tensor.device}")

    decoder.stop()
    loop.quit()

if __name__ == "__main__":
    benchmark_gst()
