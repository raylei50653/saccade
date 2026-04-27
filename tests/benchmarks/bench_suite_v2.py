import asyncio
import time
import os
import torch
import numpy as np
import argparse
import cv2
from typing import List, Dict, Any, Optional
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor
from perception.cropper import ZeroCopyCropper
from perception.tracking import SmartTracker
from perception.feature_bank import FeatureBank
from media.mediamtx_client import MediaMTXClient
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
        print("=" * 90)
        print(f"{'Module':<25} | {'Mean (ms)':<12} | {'P99 (ms)':<12} | {'StdDev':<10}")
        print("-" * 90)
        for key, values in self.records.items():
            arr = np.array(values)
            if len(arr) == 0: continue
            print(f"{key:<25} | {np.mean(arr):12.4f} | {np.percentile(arr, 99):12.4f} | {np.std(arr):10.4f}")
        print("=" * 90)

# --- Benchmark: Component Level ---

def bench_components():
    print("🔥 Starting Component-level Benchmarks (v2)...")
    stats = PerformanceStats("Components")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. SmartTracker Update Stress (Pure IoU + GMC)
    # Note: Without extractor/cropper, it falls back to IoU + GMC
    tracker = SmartTracker(iou_threshold=0.5)
    
    # Simulate 50 objects
    num_objs = 50
    boxes = torch.zeros((num_objs, 4), device=device)
    for i in range(num_objs):
        boxes[i] = torch.tensor([100+i, 100+i, 200+i, 200+i], device=device)
    scores = torch.ones((num_objs,), device=device) * 0.9
    classes = torch.zeros((num_objs,), dtype=torch.int32, device=device)
    frame_tensor = torch.rand((3, 1080, 1920), device=device) # 1080p frame

    print(f"  - Stress testing SmartTracker.update with {num_objs} objects...")
    for i in range(500):
        # Slightly move boxes to simulate movement
        boxes += 1.0
        t0 = time.perf_counter()
        tracker.update(boxes, scores, classes, frame_tensor=frame_tensor)
        stats.record("tracker_update_50obj_gmc", (time.perf_counter() - t0) * 1000)
    
    stats.report()

# --- Benchmark: Pipeline Level ---

async def bench_pipeline(num_frames: int = 500):
    print(f"🚀 Starting Pipeline E2E Benchmark v2 ({num_frames} frames)...")
    stats = PerformanceStats("Pipeline E2E")
    device = "cuda"
    
    # Engines found in previous search
    yolo_engine = "models/yolo/yolo26s_batch4.engine"
    reid_engine = "models/embedding/google_siglip2-base-patch16-224.engine"
    
    if not os.path.exists(yolo_engine):
        print(f"❌ YOLO Engine not found: {yolo_engine}")
        return
    
    detector = TRTYoloDetector(engine_path=yolo_engine)
    extractor = TRTFeatureExtractor(engine_path=reid_engine)
    cropper = ZeroCopyCropper(output_size=(224, 224))
    
    # Inject L2 into Tracker for Heartbeat ReID
    tracker = SmartTracker(
        extractor=extractor,
        cropper=cropper,
        heartbeat_interval=10
    )
    
    media = MediaMTXClient(dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))
    if not media.connect():
        print("❌ Media connection failed.")
        return
    
    processed = 0
    start_time = time.perf_counter()
    
    while processed < num_frames:
        t_grab = time.perf_counter()
        ret, tensor = media.grab_tensor() # Should return [H, W, 3] uint8 on CPU or GPU
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
        stats.record("01_media_grab", (time.perf_counter() - t_grab) * 1000)

        t_e2e = time.perf_counter()
        with torch.no_grad():
            # 1. Preprocess for YOLO (640x640)
            t1 = time.perf_counter()
            # Convert HWC to CHW and normalize
            if tensor.device.type == 'cpu':
                tensor = tensor.to(device)
            
            frame_chw = tensor.float().permute(2, 0, 1) / 255.0 # [3, H, W]
            input_4d = frame_chw.unsqueeze(0)
            yolo_input = torch.nn.functional.interpolate(input_4d, size=(640, 640))
            stats.record("02_preprocess", (time.perf_counter() - t1) * 1000)

            # 2. YOLO Inference
            t2 = time.perf_counter()
            bboxes, scores, classes, _ = detector.detect(yolo_input)
            # YOLO results are in 640x640 space usually, but TRTYoloDetector might scale them back.
            # Assuming detector.detect returns boxes in original scale or normalized.
            # If they are in 640x640, we need to scale to 1080p for cropper.
            # Looking at detector_trt.py, it doesn't seem to rescale.
            h, w = frame_chw.shape[1], frame_chw.shape[2]
            bboxes[:, [0, 2]] *= (w / 640.0)
            bboxes[:, [1, 3]] *= (h / 640.0)
            stats.record("03_yolo_inference", (time.perf_counter() - t2) * 1000)

            # 3. SmartTracker Update (Deep Dive)
            t3_start = time.perf_counter()
            
            # Sub-component 1: GMC + Pre-processing inside tracker
            t_gmc = time.perf_counter()
            gmc_matrix = tracker._calculate_gmc(frame_chw)
            light_factor = tracker._calculate_light_factor(frame_chw)
            stats.record("04a_tracker_gmc_logic", (time.perf_counter() - t_gmc) * 1000)

            # Sub-component 2: ReID Submission / Polling
            t_reid_sub = time.perf_counter()
            tracker._poll_reid()
            is_heartbeat = (tracker.extractor is not None and tracker.frame_count % tracker.heartbeat_interval == 0)
            if is_heartbeat:
                tracker._submit_reid_async(frame_chw, bboxes)
            stats.record("04b_tracker_reid_sub", (time.perf_counter() - t_reid_sub) * 1000)

            # Sub-component 3: C++ Core Update
            t_core = time.perf_counter()
            embeddings = None
            if tracker._ready_reid is not None:
                embeddings, _ = tracker._ready_reid
            
            results = tracker.gpu_tracker.update(
                bboxes, scores, classes,
                embeddings=embeddings,
                gmc=gmc_matrix,
                light_factor=light_factor,
                mid_thresh_scale=tracker._geometry_mid_thresh_scale(bboxes, frame_chw),
            )
            stats.record("04c_tracker_cpp_core", (time.perf_counter() - t_core) * 1000)

            # Sub-component 4: Post-processing (FeatureBank, Farewell)
            t_post = time.perf_counter()
            # Minimal emulation of the rest of update() to keep timing accurate
            dev = bboxes.device
            if results:
                tracked_ids = torch.tensor([r.obj_id for r in results], dtype=torch.int32, device=dev)
                tracked_boxes = torch.tensor([[r.x1, r.y1, r.x2, r.y2] for r in results], dtype=torch.float32, device=dev)
            
            # Update frame count to simulate heartbeat
            tracker.frame_count += 1
            stats.record("04d_tracker_post_logic", (time.perf_counter() - t_post) * 1000)
            
            stats.record("04_tracker_total", (time.perf_counter() - t3_start) * 1000)
        
        torch.cuda.synchronize()
        stats.record("00_total_e2e", (time.perf_counter() - t_e2e) * 1000)
        processed += 1
        
        if processed % 100 == 0:
            print(f"  Processed {processed}/{num_frames} frames...")

    duration = time.perf_counter() - start_time
    stats.report()
    print(f"Overall Throughput: {num_frames / duration:.2f} FPS")
    media.release()

# --- Main Entry ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saccade Benchmark Suite v2")
    parser.add_argument("--mode", choices=["component", "pipeline", "all"], default="all")
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()

    if args.mode in ["component", "all"]:
        bench_components()
    
    if args.mode in ["pipeline", "all"]:
        if torch.cuda.is_available():
            asyncio.run(bench_pipeline(num_frames=args.frames))
        else:
            print("❌ CUDA not available, skipping pipeline benchmark.")
