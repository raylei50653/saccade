import time
import torch
import os
from perception.detector import Detector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

def benchmark_trt_extreme():
    print("🚀 [Benchmark] Starting TensorRT + Zero-Copy Extreme Performance Test...")
    
    # 1. 初始化
    detector = Detector(model_path="./models/yolo/yolo11n.engine")
    video_path = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=video_path)
    
    if not media.connect():
        print("❌ Failed to connect.")
        return

    print("⏳ Warming up (5s)...")
    time.sleep(5)
    
    # 2. 測試循環
    num_frames = 200
    latencies = []
    
    print(f"📊 Benchmarking {num_frames} frames (End-to-End GPU Path)...")
    
    # 預熱
    for _ in range(10):
        _ = media.grab_tensor()

    start_bench = time.perf_counter()
    
    for i in range(num_frames):
        start_frame = time.perf_counter()
        
        # Step 1: Zero-Copy Grab
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            continue
            
        # Step 2: TRT Inference
        with torch.no_grad():
            # [1080, 1920, 3] -> [1, 3, 640, 640] 直接在 GPU 縮放
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            # 執行 TRT 預測
            _ = detector.model.predict(input_tensor, verbose=False, device=0)
        
        latency = (time.perf_counter() - start_frame) * 1000
        latencies.append(latency)
        
        if (i + 1) % 50 == 0:
            print(f"  - Processed {i+1}/{num_frames} frames...")

    total_time = time.perf_counter() - start_bench
    avg_latency = sum(latencies) / len(latencies)
    fps = num_frames / total_time
    
    print("\n✅ Extreme Benchmark Complete!")
    print(f"  - Average End-to-End Latency: {avg_latency:.2f} ms")
    print(f"  - Peak Throughput: {fps:.2f} FPS")
    
    media.release()

if __name__ == "__main__":
    benchmark_trt_extreme()
