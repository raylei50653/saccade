import sys
import os
import asyncio
import time

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import argparse  # noqa: E402
from typing import List, Dict  # noqa: E402
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402  # noqa: E402
from saccade.perception.cropper import ZeroCopyCropper  # noqa: E402  # noqa: E402
from saccade.perception.tracking import SmartTracker  # noqa: E402  # noqa: E402
from saccade.perception.feature_bank import FeatureBank  # noqa: E402  # noqa: E402
from saccade.media.mediamtx_client import MediaMTXClient  # noqa: E402  # noqa: E402
from saccade.media.rtsp import build_reader_url, DEFAULT_RTSP_SINGLE_STREAM_PATH  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

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
            if len(arr) == 0:
                continue
            print(
                f"{key:<25} | {np.mean(arr):12.4f} | {np.percentile(arr, 99):12.4f} | {np.std(arr):10.4f}"
            )
        print("=" * 90)


# --- Benchmark: Component Level ---


def bench_components():
    print("🔥 Starting Component-level Benchmarks (v2)...")
    stats = PerformanceStats("Components")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. SmartTracker Update Stress (Pure IoU + GMC)
    # Note: Without extractor/cropper, it falls back to IoU + GMC
    tracker = SmartTracker(max_objects=2048)

    # Mock detections (N=100)
    dets = torch.randn(100, 4, device=device) * 500
    scores = torch.rand(100, device=device)
    classes = torch.zeros(100, dtype=torch.int32, device=device)

    for i in range(100):
        start = time.perf_counter()
        tracker.update(dets, scores, classes)
        stats.record("SmartTracker_IoU_Update", (time.perf_counter() - start) * 1000)

    stats.report()


# --- Benchmark: E2E Pipeline ---


async def bench_pipeline_e2e(frames_to_bench: int = 500):
    print(f"🚀 Starting Pipeline E2E Benchmark v2 ({frames_to_bench} frames)...")
    stats = PerformanceStats("Pipeline E2E")

    # Initialize components
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26s_batch4.engine")
    extractor = TRTFeatureExtractor(model_type="siglip2")
    cropper = ZeroCopyCropper()

    feature_bank = FeatureBank()
    tracker = SmartTracker(
        extractor=extractor,
        cropper=cropper,
        feature_bank=feature_bank,
        heartbeat_interval=10,  # ReID every 10 frames
    )

    client = MediaMTXClient(rtsp_url=build_reader_url(DEFAULT_RTSP_SINGLE_STREAM_PATH))
    # Start the client
    client.connect()

    # Warmup
    dummy_frame = torch.zeros((1, 3, 640, 640), device="cuda")
    for _ in range(5):
        detector.detect(dummy_frame)

    # E2E Loop
    total_start = time.perf_counter()
    for i in range(frames_to_bench):
        loop_start = time.perf_counter()

        # 1. Media Grab
        t0 = time.perf_counter()
        ret, frame_gpu = client.grab_tensor()
        if not ret or frame_gpu is None:
            # Fallback to dummy if stream not ready
            frame_gpu = torch.zeros((1080, 1920, 3), dtype=torch.uint8, device="cuda")
        stats.record("01_media_grab", (time.perf_counter() - t0) * 1000)

        # 2. Preprocess
        t1 = time.perf_counter()
        # Convert HWC to CHW, normalize, and resize to 640x640
        frame_chw = frame_gpu.float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = F.interpolate(
            frame_chw, size=(640, 640), mode="bilinear", align_corners=False
        )
        stats.record("02_preprocess", (time.perf_counter() - t1) * 1000)

        # 3. YOLO Inference
        t2 = time.perf_counter()
        dets, scores, classes, extra = detector.detect(input_tensor)
        stats.record("03_yolo_inference", (time.perf_counter() - t2) * 1000)

        # 4. Smart Tracking (includes ReID heartbeat inside)
        t3 = time.perf_counter()
        # SmartTracker.update expects frame_tensor as [3,H,W] or [1,3,H,W]
        tracker.update(dets, scores, classes, frame_chw)
        stats.record("04_tracker_total", (time.perf_counter() - t3) * 1000)

        stats.record("00_total_e2e", (time.perf_counter() - loop_start) * 1000)

        if i % 100 == 0:
            print(f"  Processed {i}/{frames_to_bench} frames...")

    total_duration = time.perf_counter() - total_start
    fps = frames_to_bench / total_duration

    stats.report()
    print(f"Overall Throughput: {fps:.2f} FPS")
    client.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["components", "pipeline"], default="pipeline"
    )
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()

    if args.mode == "components":
        bench_components()
    else:
        asyncio.run(bench_pipeline_e2e(args.frames))
