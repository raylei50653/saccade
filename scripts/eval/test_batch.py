import torch
import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from perception.detector_trt import TRTYoloDetector

def test_batch_sizes():
    engine_path = "models/yolo/yolo26m_batch16.engine"
    print(f"Loading engine: {engine_path}")
    detector = TRTYoloDetector(engine_path=engine_path)
    
    # Warmup
    dummy = torch.randn(1, 3, 640, 640, device="cuda")
    detector.detect_batch(dummy)
    torch.cuda.synchronize()
    
    for bs in [1, 2, 4, 8, 15, 16]:
        dummy_batch = torch.randn(bs, 3, 640, 640, device="cuda")
        
        # Warmup for this specific shape
        detector.detect_batch(dummy_batch)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            detector.detect_batch(dummy_batch)
        torch.cuda.synchronize()
        
        avg_time = (time.perf_counter() - start) / iters * 1000
        fps = 1000 / avg_time
        print(f"Batch Size {bs:2d} -> Avg Latency: {avg_time:.2f} ms | Equivalent FPS: {fps:.2f}")

if __name__ == "__main__":
    test_batch_sizes()
