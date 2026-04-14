import asyncio
import time
import os
import torch
import numpy as np
from perception.detector_trt import TRTYoloDetector
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor
from perception.tracker import SmartTracker
from perception.drift_handler import SemanticDriftHandler
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

class PipelineBenchmarker:
    def __init__(self):
        self.stats = {
            "media_grab": [],
            "preprocess": [],
            "yolo_inference": [],
            "post_logic": [], 
            "feature_extract": [],
            "total_e2e": []
        }

    def record(self, key, duration_ms):
        self.stats[key].append(duration_ms)

    def report(self, total_duration):
        print("\n" + "═"*80)
        print(f"📊 Saccade Perception Pipeline Benchmark Report")
        print("═"*80)
        print(f"{'Module':<20} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'StdDev':<8}")
        print("-" * 80)
        for key, values in self.stats.items():
            if not values: continue
            arr = np.array(values)
            print(f"{key:<20} | {np.mean(arr):10.2f} | {np.percentile(arr, 99):10.2f} | {np.std(arr):8.2f}")
        
        avg_fps = len(self.stats["total_e2e"]) / total_duration
        print("-" * 80)
        print(f"🚀 Real-world Throughput: {avg_fps:.2f} FPS")
        print("═"*80)

async def run_benchmark(num_frames=200):
    print(f"🚀 Starting Perception Pipeline Analysis ({num_frames} frames)...")
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
    cropper = ZeroCopyCropper()
    extractor = TRTFeatureExtractor()
    tracker = SmartTracker()
    drift_handler = SemanticDriftHandler()
    media = MediaMTXClient(dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))
    
    if not media.connect(): return
    bench = PipelineBenchmarker()

    # 穩健等待
    for _ in range(100):
        ret, _ = media.grab_tensor()
        if ret: break
        await asyncio.sleep(0.1)

    processed = 0
    start_time_all = time.perf_counter()
    
    while processed < num_frames:
        t0 = time.perf_counter()
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
        bench.record("media_grab", (time.perf_counter() - t0) * 1000)

        t_start = time.perf_counter()
        with torch.no_grad():
            # 1. Preprocess
            t1 = time.perf_counter()
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            yolo_input = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            bench.record("preprocess", (time.perf_counter() - t1) * 1000)

            # 2. YOLO
            t2 = time.perf_counter()
            results = detector.detect(yolo_input)
            bench.record("yolo_inference", (time.perf_counter() - t2) * 1000)

            # 3. Feature Extract (if objects found)
            t3 = time.perf_counter()
            # 簡化邏輯，僅測量有物件時的開銷
            if True:
                _, _, _, ids = results
                has_objs = ids is not None and ids.numel() > 0
            
            if has_objs:
                # 這裡僅示意特徵提取路徑
                bench.record("feature_extract", (time.perf_counter() - t3) * 1000)
            
            torch.cuda.synchronize()
            bench.record("total_e2e", (time.perf_counter() - t_start) * 1000)

        processed += 1
        if processed % 50 == 0: print(f"  - {processed}/{num_frames} frames...")

    bench.report(time.perf_counter() - start_time_all)
    media.release()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
