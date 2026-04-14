import time
import torch
import os
from perception.detector_trt import TRTYoloDetector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

def benchmark_trt_extreme():
    print("🚀 [Benchmark] Starting TensorRT + Zero-Copy Extreme Performance Test...")
    
    # 1. 初始化 (優先使用高效能的 yolo26n)
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
    video_path = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=video_path)
    
    if not media.connect():
        print("❌ Failed to connect.")
        return

    print("⏳ Waiting for GStreamer C++ Pool to start...")
    # 確保第一幀已經解碼出來，這對極限測試的準確性至關重要
    for _ in range(500):
        ret, _ = media.grab_tensor()
        if ret: break
        time.sleep(0.01)
    else:
        print("❌ Timeout waiting for first frame.")
        return
    
    # 2. 測試循環
    num_frames = 300 # 增加測試規模以獲得更穩定的數據
    latencies = []
    
    print(f"📊 Benchmarking {num_frames} frames (Native C++ Path + GPU Pool)...")
    
    start_bench = time.perf_counter()
    processed_count = 0
    
    while processed_count < num_frames:
        start_frame = time.perf_counter()
        
        # Step 1: Zero-Copy Grab
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            time.sleep(0.001)
            continue
            
        # Step 2: Full GPU Path Inference
        with torch.no_grad():
            # HWC -> CHW, Normalize & Resize
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            
            # 使用統一封裝，支援 Native C++ TRT
            _ = detector.detect(input_tensor)
        
        latency = (time.perf_counter() - start_frame) * 1000
        latencies.append(latency)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"  - Processed {processed_count}/{num_frames} frames...")

    total_time = time.perf_counter() - start_bench
    avg_latency = sum(latencies) / len(latencies)
    fps = num_frames / total_time
    
    print("\n✅ Extreme Benchmark Complete!")
    print(f"  - Average E2E Latency: {avg_latency:.2f} ms")
    print(f"  - Max Possible Throughput: {fps:.2f} FPS")
    print(f"  - GPU Buffer Pool: 5 Frames (Circular)")
    
    media.release()

if __name__ == "__main__":
    benchmark_trt_extreme()
