import asyncio
import time
import os
import torch
import numpy as np
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor
from perception.cropper import ZeroCopyCropper
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()


class PipelineBenchmarker:
    def __init__(self):
        self.stats = {
            "media_grab": [],
            "preprocess": [],
            "yolo_inference": [],
            "zero_copy_crop": [],
            "feature_extract": [],
            "total_e2e": [],
        }

    def record(self, key, duration_ms):
        self.stats[key].append(duration_ms)

    def report(self, total_duration):
        print("\n" + "═" * 100)
        print("📊 Saccade Perception Pipeline Benchmark Report (L1-L2 SigLIP 2)")
        print("═" * 100)
        print(
            f"{'Module':<20} | {'Mean (ms)':<12} | {'P99 (ms)':<12} | {'StdDev':<10} | {'% Total'}"
        )
        print("-" * 100)
        total_mean = sum(np.mean(v) for v in self.stats.values() if v and "total" not in v)
        for key, values in self.stats.items():
            if not values:
                continue
            arr = np.array(values)
            mean_val = np.mean(arr)
            pct = (mean_val / total_mean * 100) if "total" not in key else 100
            print(
                f"{key:<20} | {mean_val:12.4f} | {np.percentile(arr, 99):12.4f} | {np.std(arr):10.4f} | {pct:6.1f}%"
            )

        avg_fps = len(self.stats["total_e2e"]) / total_duration
        print("-" * 100)
        print(f"🚀 Real-world Throughput: {avg_fps:.2f} FPS")
        print("═" * 100)


async def run_benchmark(num_frames=5000):
    print(f"🚀 Starting L1-L2 Pipeline Long-run Analysis ({num_frames} frames)...")
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
    extractor = TRTFeatureExtractor(
        engine_path="models/embedding/google_siglip2-base-patch16-224.engine"
    )
    cropper = ZeroCopyCropper(output_size=(224, 224))

    media = MediaMTXClient(
        dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    )

    if not media.connect():
        return
    bench = PipelineBenchmarker()

    # 穩健等待
    print("⏳ Waiting for stream stability...")
    for _ in range(100):
        ret, _ = media.grab_tensor()
        if ret:
            break
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
            # 確保在 GPU 上進行預處理
            frame_gpu = tensor.float() / 255.0  # [H, W, C]
            frame_chw = frame_gpu.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            yolo_input = torch.nn.functional.interpolate(frame_chw, size=(640, 640))
            bench.record("preprocess", (time.perf_counter() - t1) * 1000)

            # 2. YOLO Inference (L1)
            t2 = time.perf_counter()
            # detector.detect returns (bboxes, scores, classes, ids)
            bboxes, scores, classes, ids = detector.detect(yolo_input)
            bench.record("yolo_inference", (time.perf_counter() - t2) * 1000)

            # 3. Zero-Copy Crop
            t3 = time.perf_counter()
            # 假設有抓到目標（若無則略過 L2）
            if bboxes is not None and bboxes.numel() > 0:
                # 這裡需要將 bboxes 縮放回原始 frame 尺寸，
                # 但為了 benchmark 延遲，我們直接模擬裁切
                crops = cropper.process(frame_chw, bboxes[:8])  # 限制最多 8 個物件
                bench.record("zero_copy_crop", (time.perf_counter() - t3) * 1000)

                # 4. SigLIP 2 Feature Extract (L2)
                t4 = time.perf_counter()
                _ = extractor.extract(crops)
                bench.record("feature_extract", (time.perf_counter() - t4) * 1000)
            else:
                bench.record("zero_copy_crop", 0)
                bench.record("feature_extract", 0)

            torch.cuda.synchronize()
            bench.record("total_e2e", (time.perf_counter() - t_start) * 1000)

        processed += 1
        if processed % 1000 == 0:
            print(f"  - {processed}/{num_frames} frames...")

    bench.report(time.perf_counter() - start_time_all)
    media.release()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
