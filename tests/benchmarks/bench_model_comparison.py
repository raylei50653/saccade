import time
import torch
import os
import numpy as np
from perception.detector_trt import TRTYoloDetector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

def run_model_test(model_name, model_path, num_frames=200):
    print(f"\n🔥 Evaluating {model_name}...")
    detector = TRTYoloDetector(engine_path=model_path)
    media = MediaMTXClient(dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))
    
    if not media.connect(): return None

    # 等待串流穩定
    for _ in range(50):
        ret, _ = media.grab_tensor()
        if ret: break
        time.sleep(0.1)
    
    latencies = []
    processed = 0
    while processed < num_frames:
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            time.sleep(0.001)
            continue
            
        start = time.perf_counter()
        with torch.no_grad():
            # 標準化預處理路徑
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            _ = detector.detect(input_tensor)
        
        latencies.append((time.perf_counter() - start) * 1000)
        processed += 1

    media.release()
    return {"avg": np.mean(latencies), "p99": np.percentile(latencies, 99), "fps": num_frames / (sum(latencies)/1000)}

if __name__ == "__main__":
    print("🚀 [Benchmark] YOLO Generation Comparison (TRT Only)")
    results_11 = run_model_test("YOLO11n (Ultralytics)", "./models/yolo/yolo11n.engine")
    results_26 = run_model_test("YOLO26n (Saccade Native)", "./models/yolo/yolo26n.engine")
    
    print("\n" + "═"*60)
    print(f"{'Model':<25} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'Max FPS'}")
    print("-" * 60)
    if results_11:
        print(f"{'YOLO11n (TRT)':<25} | {results_11['avg']:10.2f} | {results_11['p99']:10.2f} | {1000/results_11['avg']:8.2f}")
    if results_26:
        print(f"{'YOLO26n (Native TRT)':<25} | {results_26['avg']:10.2f} | {results_26['p99']:10.2f} | {1000/results_26['avg']:8.2f}")
    print("═"*60)
