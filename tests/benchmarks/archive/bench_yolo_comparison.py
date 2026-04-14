import time
import torch
import os
import numpy as np
from perception.detector_trt import TRTYoloDetector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

def run_benchmark(model_name, model_path, num_frames=300):
    print(f"\n🚀 [Benchmark] Testing {model_name} ({model_path})...")
    
    # 1. 初始化
    try:
        detector = TRTYoloDetector(engine_path=model_path)
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

    video_path = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=video_path)
    
    if not media.connect():
        print("❌ Media connection failed.")
        return None

    # 預熱
    print("⏳ Waiting for stream to start (Warming up)...")
    for attempt in range(500): # 增加等待次數，並加入時間延遲
        ret, tensor = media.grab_tensor()
        if ret and tensor is not None:
            break
        time.sleep(0.01)
    else:
        print("❌ Timeout waiting for the first frame.")
        media.release()
        return None
    
    # 2. 測試循環
    latencies = []
    
    print(f"📊 Running {num_frames} frames...")
    
    start_bench = time.perf_counter()
    processed_frames = 0
    
    while processed_frames < num_frames:
        start_frame = time.perf_counter()
        
        # Step 1: Zero-Copy Grab
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            time.sleep(0.005) # 稍微等待下一個影格，避免空轉死循環
            continue
            
        # Step 2: Inference
        with torch.no_grad():
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            # 使用統一的封裝方法，支援 Native TRT 與 Ultralytics
            _ = detector.detect(input_tensor)
        
        latency = (time.perf_counter() - start_frame) * 1000
        latencies.append(latency)
        processed_frames += 1
        
        if processed_frames % 50 == 0:
            print(f"  - Processed {processed_frames}/{num_frames} frames...")

    if not latencies:

        print("❌ No frames were processed successfully.")
        media.release()
        return None

    total_time = time.perf_counter() - start_bench
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = num_frames / total_time
    
    print(f"✅ {model_name} Results:")
    print(f"  - Average Latency: {avg_latency:.2f} ms")
    print(f"  - P95 Latency:     {p95_latency:.2f} ms")
    print(f"  - Throughput:      {fps:.2f} FPS")
    
    media.release()
    return {"avg": avg_latency, "p95": p95_latency, "fps": fps}

if __name__ == "__main__":
    # 比較 YOLO11 與 YOLO26
    yolo11_results = run_benchmark("YOLO11n (TRT)", "./models/yolo/yolo11n.engine")
    yolo26_results = run_benchmark("YOLO26n (TRT)", "./models/yolo/yolo26n.engine")
    
    if yolo11_results and yolo26_results:
        improvement = (yolo11_results['avg'] - yolo26_results['avg']) / yolo11_results['avg'] * 100
        print("\n📈 Performance Summary:")
        print(f"  - Latency Reduction: {improvement:.2f}%")