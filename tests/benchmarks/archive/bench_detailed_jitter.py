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

class JitterBenchmarker:
    def __init__(self):
        self.stats = {
            "media_grab": [],
            "preprocess": [],
            "yolo_track": [],
            "post_yolo_logic": [], 
            "roi_align": [],
            "siglip_extract": [],
            "drift_handling": [],
            "total_e2e": []
        }

    def record(self, key, duration_ms):
        self.stats[key].append(duration_ms)

    def report(self, total_duration):
        print("\n" + "═"*80)
        print(f"📊 Saccade Detailed Jitter & Latency Report (Total: {total_duration:.2f}s)")
        print("═"*80)
        print(f"{'Module':<20} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'StdDev':<8} | {'Max':<8}")
        print("-" * 80)
        for key, values in self.stats.items():
            if not values: continue
            arr = np.array(values)
            p99 = np.percentile(arr, 99)
            print(f"{key:<20} | {np.mean(arr):10.2f} | {p99:10.2f} | {np.std(arr):8.2f} | {np.max(arr):8.2f}")
        
        avg_fps = len(self.stats["total_e2e"]) / total_duration
        print("-" * 80)
        print(f"🚀 Overall Throughput: {avg_fps:.2f} FPS")
        print("═"*80)

async def benchmark_jitter(num_frames=300):
    print(f"🚀 [Jitter Benchmark] Analyzing {num_frames} frames for YOLO26 + SigLIP 2 + C++ Pool...")
    
    # 1. 初始化所有組件
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
    cropper = ZeroCopyCropper(output_size=(224, 224))
    extractor = TRTFeatureExtractor()
    tracker = SmartTracker(iou_threshold=0.7)
    drift_handler = SemanticDriftHandler(similarity_threshold=0.95)
    
    media = MediaMTXClient(dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4"))
    if not media.connect(): return

    bench = JitterBenchmarker()
    
    print("⏳ Waiting for C++ Buffer Pool to stabilize...")
    # 穩健等待第一幀
    for _ in range(500):
        ret, _ = media.grab_tensor()
        if ret: break
        await asyncio.sleep(0.01)
    else:
        print("❌ Timeout waiting for first frame.")
        return

    print("🔥 Warming up (20 frames)...")
    for _ in range(20):
        ret, tensor = media.grab_tensor()
        if not ret: continue
        with torch.no_grad():
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            yolo_input = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            _ = detector.detect(yolo_input)
        torch.cuda.synchronize()
    
    # 2. 核心測試循環
    processed = 0
    start_time_all = time.perf_counter()
    
    while processed < num_frames:
        t0 = time.perf_counter()
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.001)
            continue
        
        t_grab = time.perf_counter()
        bench.record("media_grab", (t_grab - t0) * 1000)

        t_frame_start = time.perf_counter()
        
        with torch.no_grad():
            # [A] Preprocess
            t1 = time.perf_counter()
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
            yolo_input = torch.nn.functional.interpolate(input_tensor, size=(640, 640)) / 255.0
            t_preprocess = time.perf_counter()
            bench.record("preprocess", (t_preprocess - t1) * 1000)

            # [B] YOLO26 Detection
            t2 = time.perf_counter()
            results = detector.detect(yolo_input, conf_threshold=0.25)
            t_yolo = time.perf_counter()
            bench.record("yolo_track", (t_yolo - t2) * 1000)

            # 解析結果
            if True:
                boxes, scores, cls_ids, ids = results
                has_objects = ids is not None and ids.numel() > 0

            if has_objects:
                # [C] Post-YOLO Logic
                t3 = time.perf_counter()
                scale_x, scale_y = tensor.shape[1] / 640.0, tensor.shape[0] / 640.0
                boxes_1080p = boxes.clone()
                boxes_1080p[:, [0, 2]] *= scale_x
                boxes_1080p[:, [1, 3]] *= scale_y
                ext_ids, ext_boxes = tracker.update_and_filter(ids, boxes_1080p)
                t_post = time.perf_counter()
                bench.record("post_yolo_logic", (t_post - t3) * 1000)

                # [D] Zero-Copy RoI Align
                if ext_boxes.numel() > 0:
                    t4 = time.perf_counter()
                    crops = cropper.process(input_tensor / 255.0, ext_boxes)
                    t_roi = time.perf_counter()
                    bench.record("roi_align", (t_roi - t4) * 1000)

                    # [E] SigLIP 2 Vector Extraction
                    t5 = time.perf_counter()
                    features = extractor.extract(crops)
                    t_siglip = time.perf_counter()
                    bench.record("siglip_extract", (t_siglip - t5) * 1000)

                    # [F] Semantic Drift Handling
                    t6 = time.perf_counter()
                    _, _ = drift_handler.filter_novel_features(ext_ids, features)
                    t_drift = time.perf_counter()
                    bench.record("drift_handling", (t_drift - t6) * 1000)
            
            # 同步 CUDA，確保測量的是真實 GPU 耗時
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            bench.record("total_e2e", (t_end - t_frame_start) * 1000)

        processed += 1
        if processed % 50 == 0: print(f"  - {processed}/{num_frames} frames benchmarked")

    total_duration = time.perf_counter() - start_time_all
    bench.report(total_duration)
    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_jitter())
