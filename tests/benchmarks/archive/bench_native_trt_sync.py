import time
import torch
import numpy as np
from perception.detector_trt import TRTYoloDetector

def benchmark_native_trt_sync(num_frames=100):
    print(f"🚀 [Benchmark] Native TRT Detector (with Sync and Preprocessing)...")
    detector = TRTYoloDetector()
    
    # 1. 準備輸入 [1, 3, 640, 640]
    dummy_input = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.float32)
    
    # 2. 預熱
    for _ in range(20):
        detector.detect(dummy_input)
    torch.cuda.synchronize()
    
    # 3. 測試
    latencies = []
    for i in range(num_frames):
        start = time.perf_counter()
        
        # 執行偵測
        boxes, scores, classes = detector.detect(dummy_input)
        
        # 為了測量真實耗時，必須同步
        torch.cuda.synchronize()
        
        latencies.append((time.perf_counter() - start) * 1000)
        
    print(f"✅ Benchmark Complete!")
    print(f"  - Average Latency: {np.mean(latencies):.2f} ms")
    print(f"  - StdDev (Jitter): {np.std(latencies):.2f} ms")
    print(f"  - Min/Max:         {np.min(latencies):.2f} / {np.max(latencies):.2f} ms")

if __name__ == "__main__":
    benchmark_native_trt_sync()
