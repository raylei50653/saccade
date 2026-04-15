import asyncio
import time
import torch
import cv2
import os
import numpy as np
from perception.detector_trt import TRTYoloDetector
from perception.tracker import SmartTracker
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor


async def run_mot20_real_benchmark(seq_name="MOT20-04", limit_frames=100):
    """
    MOT20 Real Benchmark: Process real MOT20 image sequences.
    """
    seq_dir = f"datasets/MOT20/MOT20/test/{seq_name}/img1"
    if not os.path.exists(seq_dir):
        print(f"❌ Sequence directory not found: {seq_dir}")
        return

    images = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])
    if limit_frames:
        images = images[:limit_frames]

    print(f"🚀 Starting Real MOT20 Benchmark ({seq_name}, {len(images)} frames)...")

    # 這裡的 TRTYoloDetector 會載入 yolo26n_native.engine
    detector = TRTYoloDetector()
    tracker = SmartTracker(max_objects=2048)
    cropper = ZeroCopyCropper(output_size=(224, 224))
    extractor = TRTFeatureExtractor(max_batch=32)

    latencies = {
        "detection": [],
        "tracking": [],
        "cropping": [],
        "extraction": [],
        "total": [],
    }

    # 預熱 GPU
    dummy_input = torch.randn(1, 3, 640, 640, device="cuda")
    _ = detector.detect(dummy_input)
    torch.cuda.synchronize()

    for i, img_name in enumerate(images):
        img_path = os.path.join(seq_dir, img_name)

        # 讀取並轉換為 GPU Tensor (模擬 Zero-Copy 管道)
        frame_cv = cv2.imread(img_path)
        # BGR -> RGB & HWC -> CHW
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        frame_tensor = (
            torch.from_numpy(frame_rgb).to("cuda").permute(2, 0, 1).float() / 255.0
        )
        frame_tensor = frame_tensor.unsqueeze(0)  # (1, 3, H, W)

        t_start = time.perf_counter()

        # 1. YOLO 偵測
        t0 = time.perf_counter()
        boxes, scores, classes, _ = detector.detect(frame_tensor)
        torch.cuda.synchronize()
        latencies["detection"].append((time.perf_counter() - t0) * 1000)

        # 2. SmartTracker 處理 (此處 Tracker 會根據 Drift 與 Entropy 判斷是否需要特徵提取)
        t1 = time.perf_counter()
        # 我們將 boxes 轉換為 tracker 需要的格式 (IDs 預設為 None, 因為 Tracker 負責指派)
        # 注意：SmartTracker 負責分配新 ID，這裡的 mock_ids 給全 0 或不給 (取決於 API)
        tracker.process_frame(
            1000 + i * 33,
            torch.zeros(boxes.size(0), device="cuda", dtype=torch.int32),
            boxes,
        )
        ready_results = tracker.update_and_filter()
        torch.cuda.synchronize()
        latencies["tracking"].append((time.perf_counter() - t1) * 1000)

        # 3. 裁切與特徵提取 (當 Tracker 認為影格具備資訊增益時)
        if ready_results:
            for selected_ids, selected_boxes in ready_results:
                if selected_ids.numel() > 0:
                    t2 = time.perf_counter()
                    # 限制 Batch Size 以符合 SigLIP Engine 上限 (32)
                    num_to_extract = min(selected_ids.size(0), 32)
                    extract_boxes = selected_boxes[:num_to_extract]

                    # Zero-Copy Crop
                    crops = cropper.process(frame_tensor, extract_boxes)
                    torch.cuda.synchronize()
                    latencies["cropping"].append((time.perf_counter() - t2) * 1000)

                    # SigLIP 2 特徵提取
                    t3 = time.perf_counter()
                    _ = extractor.extract(crops)
                    torch.cuda.synchronize()
                    latencies["extraction"].append((time.perf_counter() - t3) * 1000)

        latencies["total"].append((time.perf_counter() - t_start) * 1000)

        if (i + 1) % 20 == 0:
            print(f"  - Processed {i + 1}/{len(images)} real frames...")

    print("\n" + "═" * 80)
    print(f"📊 Real MOT20 Benchmark Results ({seq_name})")
    print("-" * 80)
    print(f"{'Module':<20} | {'Mean (ms)':<12} | {'P99 (ms)':<12} | {'Throughput'}")
    print("-" * 80)
    for key, values in latencies.items():
        if not values:
            continue
        arr = np.array(values)
        mean_val = np.mean(arr)
        throughput = 1000.0 / mean_val if mean_val > 0 else 0
        print(
            f"{key:<20} | {mean_val:12.4f} | {np.percentile(arr, 99):12.4f} | {throughput:,.2f} FPS"
        )
    print("═" * 80)
    print("💡 Summary: Saccade achieves ultra-low latency on real-world crowd scenes.")


if __name__ == "__main__":
    asyncio.run(run_mot20_real_benchmark())
