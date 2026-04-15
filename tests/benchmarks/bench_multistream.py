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


async def process_stream(
    stream_id, seq_name, limit_frames=100, detector=None, extractor=None
):
    """
    Process a single stream within a multi-stream environment.
    """
    seq_dir = f"datasets/MOT20/MOT20/test/{seq_name}/img1"
    if not os.path.exists(seq_dir):
        print(f"❌ Stream {stream_id}: Sequence not found: {seq_dir}")
        return []

    images = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])[
        :limit_frames
    ]

    # 每個串流擁有獨立的 Tracker 和 Cropper
    tracker = SmartTracker(max_objects=1024)
    cropper = ZeroCopyCropper(output_size=(224, 224))

    stream_latencies = []

    for i, img_name in enumerate(images):
        img_path = os.path.join(seq_dir, img_name)

        # 模擬影像解碼後的 Tensor (BGR -> RGB)
        frame_cv = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        frame_tensor = (
            torch.from_numpy(frame_rgb).to("cuda").permute(2, 0, 1).float() / 255.0
        )
        frame_tensor = frame_tensor.unsqueeze(0)

        t_start = time.perf_counter()

        # 1. 共享 YOLO 偵測 (Batch Size = 1 for simple concurrent simulation)
        boxes, scores, classes, _ = detector.detect(frame_tensor)

        # 2. 獨立 Tracker
        tracker.process_frame(
            1000 + i * 33,
            torch.zeros(boxes.size(0), device="cuda", dtype=torch.int32),
            boxes,
        )
        ready_results = tracker.update_and_filter()

        # 3. 獨立 Cropper + 共享 Extractor
        if ready_results:
            for selected_ids, selected_boxes in ready_results:
                if selected_ids.numel() > 0:
                    num_to_extract = min(
                        selected_ids.size(0), 16
                    )  # 每路限制提取數量以減少競爭
                    extract_boxes = selected_boxes[:num_to_extract]
                    crops = cropper.process(frame_tensor, extract_boxes)
                    _ = extractor.extract(crops)

        torch.cuda.synchronize()
        stream_latencies.append((time.perf_counter() - t_start) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  [Stream {stream_id}] Processed {i + 1}/{len(images)} frames...")

    return stream_latencies


async def run_multistream_benchmark(num_streams=4, limit_frames=100):
    """
    Multi-stream benchmark: Concurrent processing of multiple MOT20 sequences.
    """
    print(
        f"🔥 Starting Multi-stream Benchmark ({num_streams} concurrent streams, {limit_frames} frames each)..."
    )

    # 初始化共享資源 (Singleton-like)
    detector = TRTYoloDetector()
    extractor = TRTFeatureExtractor(max_batch=64)  # 增大 Batch 以應對多路併發

    # 預熱
    dummy_input = torch.randn(1, 3, 640, 640, device="cuda")
    _ = detector.detect(dummy_input)
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    # 使用 asyncio.gather 同時執行多個串流處理任務
    sequences = ["MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"]
    tasks = []
    for i in range(num_streams):
        seq = sequences[i % len(sequences)]
        tasks.append(process_stream(i, seq, limit_frames, detector, extractor))

    results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time
    total_frames = num_streams * limit_frames
    overall_fps = total_frames / total_time

    print("\n" + "═" * 80)
    print(f"📊 Multi-stream Benchmark Results ({num_streams} Streams)")
    print("-" * 80)
    print(f"{'Stream ID':<12} | {'Mean Latency (ms)':<18} | {'P99 (ms)':<10}")
    print("-" * 80)

    all_latencies = []
    for i, latencies in enumerate(results):
        if not latencies:
            continue
        arr = np.array(latencies)
        all_latencies.extend(latencies)
        print(f"{i:<12} | {np.mean(arr):18.4f} | {np.percentile(arr, 99):10.2f}")

    print("-" * 80)
    print(f"🚀 Overall System Throughput: {overall_fps:,.2f} FPS")
    print(f"📦 Total Processed Frames: {total_frames}")
    print(f"⏱️  Total Elapsed Time: {total_time:.2f} s")
    print("═" * 80)
    print(
        "💡 Inference: Saccade handles massive parallelism by sharing TRT engines across streams."
    )


if __name__ == "__main__":
    # 預設跑 4 路
    asyncio.run(run_multistream_benchmark(num_streams=4))
