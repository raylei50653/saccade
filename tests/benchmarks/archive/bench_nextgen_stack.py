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

async def benchmark_nextgen_stack(num_frames=200):
    print(f"🚀 [Benchmark] Starting Next-Gen Stack (YOLO26 + SigLIP 2) Performance Test...")
    
    # 1. 初始化組件
    detector = TRTYoloDetector() # 預設使用 yolo26n.engine
    cropper = ZeroCopyCropper(output_size=(224, 224))
    extractor = TRTFeatureExtractor() # 預設使用 siglip2 engine
    tracker = SmartTracker(iou_threshold=0.7)
    drift_handler = SemanticDriftHandler(similarity_threshold=0.95)
    
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=dummy_video)
    
    if not media.connect():
        print("❌ Failed to connect to media source.")
        return

    print("⏳ Warming up (5s)...")
    await asyncio.sleep(5)
    
    # 2. 測試循環
    latencies_yolo = []
    latencies_siglip = []
    latencies_total = []
    
    print(f"📊 Benchmarking {num_frames} frames (Zero-Copy Pipeline)...")
    
    processed_count = 0
    start_bench = time.perf_counter()
    
    while processed_count < num_frames:
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            await asyncio.sleep(0.005)
            continue
            
        start_frame = time.perf_counter()
        
        with torch.no_grad():
            # [1080, 1920, 3] -> [1, 3, 1080, 1920]
            input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            yolo_input = torch.nn.functional.interpolate(input_tensor, size=(640, 640))
            
            # Step 1: YOLO26 Tracking (NMS-Free)
            start_yolo = time.perf_counter()
            results = detector.model.track(yolo_input, verbose=False, persist=True)
            latencies_yolo.append((time.perf_counter() - start_yolo) * 1000)
            
            # Step 2: Extract Features if objects found
            if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                start_siglip = time.perf_counter()
                boxes = results[0].boxes.xyxy
                ids = results[0].boxes.id
                
                # 映射座標到 1080p
                scale_x, scale_y = tensor.shape[1] / 640.0, tensor.shape[0] / 640.0
                boxes_1080p = boxes.clone()
                boxes_1080p[:, [0, 2]] *= scale_x
                boxes_1080p[:, [1, 3]] *= scale_y
                
                # 智能過濾與特徵提取
                ext_ids, ext_boxes = tracker.update_and_filter(ids, boxes_1080p)
                features = tracker.async_extract_features(input_tensor, ext_boxes, cropper, extractor)
                
                if features is not None:
                    torch.cuda.current_stream().wait_stream(tracker.extraction_stream)
                    _, _ = drift_handler.filter_novel_features(ext_ids, features)
                
                latencies_siglip.append((time.perf_counter() - start_siglip) * 1000)
        
        latencies_total.append((time.perf_counter() - start_frame) * 1000)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"  - Processed {processed_count}/{num_frames} frames...")

    total_time = time.perf_counter() - start_bench
    
    print("\n✅ Next-Gen Stack Benchmark Complete!")
    print(f"  - YOLO26 Latency:       {np.mean(latencies_yolo):.2f} ms")
    print(f"  - SigLIP 2 Latency:     {np.mean(latencies_siglip):.2f} ms (avg per frame with objects)")
    print(f"  - Total E2E Latency:    {np.mean(latencies_total):.2f} ms")
    print(f"  - Peak Throughput:      {num_frames / total_time:.2f} FPS")
    
    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_nextgen_stack())
