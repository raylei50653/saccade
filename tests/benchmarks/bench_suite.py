import asyncio
import time
import os
import torch
import numpy as np
import argparse
from typing import List, Dict, Any
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor
from perception.cropper import ZeroCopyCropper
from perception.tracking import SmartTracker
from perception.drift_handler import SemanticDriftHandler
from media.mediamtx_client import MediaMTXClient
from cognition.resource_manager import DegradationLevel
from dotenv import load_dotenv

load_dotenv()

# --- Utility: Performance Stats ---

class PerformanceStats:
    def __init__(self, name: str):
        self.name = name
        self.records: Dict[str, List[float]] = {}

    def record(self, key: str, duration_ms: float):
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(duration_ms)

    def report(self):
        print(f"\n📊 Benchmark Report: {self.name}")
        print("=" * 80)
        print(f"{'Module':<20} | {'Mean (ms)':<12} | {'P99 (ms)':<12} | {'StdDev':<10}")
        print("-" * 80)
        for key, values in self.records.items():
            arr = np.array(values)
            print(f"{key:<20} | {np.mean(arr):12.4f} | {np.percentile(arr, 99):12.4f} | {np.std(arr):10.4f}")
        print("=" * 80)

# --- Benchmark: Component Level ---

def bench_components():
    print("🔥 Starting Component-level Benchmarks...")
    stats = PerformanceStats("Components")
    
    # 1. Drift Handler Stress
    drift_handler = SemanticDriftHandler()
    obj_ids = list(range(100, 164))
    boxes = torch.randn(64, 4, device="cuda")
    
    for _ in range(1000):
        t0 = time.perf_counter()
        drift_handler.filter_for_batch(obj_ids, boxes, DegradationLevel.NORMAL)
        stats.record("drift_filter_64", (time.perf_counter() - t0) * 1000)
    
    # 2. Smart Tracker Buffer Reordering
    tracker = SmartTracker()
    ts_sequence = [1000, 1066, 1132, 1033, 1099]
    for _ in range(200):
        for ts in ts_sequence:
            tracker.process_frame(ts, torch.tensor([1], device="cuda"), torch.randn(1, 4, device="cuda"))
        t0 = time.perf_counter()
        tracker.update_and_filter()
        stats.record("tracker_reorder_5f", (time.perf_counter() - t0) * 1000)
    
    stats.report()

# --- Benchmark: Pipeline Level ---

async def bench_pipeline(num_frames: int = 1000):
    print(f"🚀 Starting Pipeline E2E Benchmark ({num_frames} frames)...")
    stats = PerformanceStats("Pipeline E2E")
    
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
    extractor = TRTFeatureExtractor(engine_path="models/embedding/google_siglip2-base-patch16-224.engine")
    cropper = ZeroCopyCropper(output_size=(224, 224))
    media = MediaMTXClient(dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))

    if not media.connect(): return
    
    processed = 0
    start_time = time.perf_counter()
    
    while processed < num_frames:
        t_grab = time.perf_counter()
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
        stats.record("media_grab", (time.perf_counter() - t_grab) * 1000)

        t_e2e = time.perf_counter()
        with torch.no_grad():
            # Preprocess
            t1 = time.perf_counter()
            frame_chw = tensor.float().permute(2, 0, 1).unsqueeze(0) / 255.0
            yolo_input = torch.nn.functional.interpolate(frame_chw, size=(640, 640))
            stats.record("preprocess", (time.perf_counter() - t1) * 1000)

            # YOLO Inference
            t2 = time.perf_counter()
            bboxes, scores, classes, ids = detector.detect(yolo_input)
            stats.record("yolo_inference", (time.perf_counter() - t2) * 1000)

            # Crop & Extract
            if bboxes is not None and bboxes.numel() > 0:
                t3 = time.perf_counter()
                crops = cropper.process(frame_chw, bboxes[:8])
                stats.record("zero_copy_crop", (time.perf_counter() - t3) * 1000)

                t4 = time.perf_counter()
                extractor.extract(crops)
                stats.record("feature_extract", (time.perf_counter() - t4) * 1000)
        
        torch.cuda.synchronize()
        stats.record("total_e2e", (time.perf_counter() - t_e2e) * 1000)
        processed += 1

    duration = time.perf_counter() - start_time
    stats.report()
    print(f"Overall Throughput: {num_frames / duration:.2f} FPS")
    media.release()

# --- Main Entry ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saccade Benchmark Suite")
    parser.add_argument("--mode", choices=["component", "pipeline", "all"], default="all")
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args()

    if args.mode in ["component", "all"]:
        bench_components()
    
    if args.mode in ["pipeline", "all"]:
        if torch.cuda.is_available():
            asyncio.run(bench_pipeline(num_frames=args.frames))
        else:
            print("❌ CUDA not available, skipping pipeline benchmark.")
