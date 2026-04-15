import asyncio
import time
import torch
import numpy as np
import pynvml
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor


async def run_latency_breakdown() -> None:
    pynvml.nvmlInit()

    print("⏳ Loading Engines...")
    detector = TRTYoloDetector()
    extractor = TRTFeatureExtractor(max_batch=32)

    batch_sizes = [1, 4, 8, 16, 32]

    print("\n" + "═" * 80)
    print(
        f"{'Batch Size':<12} | {'L1 YOLO (ms)':<15} | {'L2 SigLIP2 (ms)':<18} | {'Total GPU (ms)':<15}"
    )
    print("-" * 80)

    for batch_size in batch_sizes:
        # Prepare dummy data
        frames = torch.randn(batch_size, 3, 640, 640, device="cuda")
        crops = torch.randn(batch_size, 3, 224, 224, device="cuda")

        # Warm up
        for _ in range(5):
            detector.detect_batch(frames)
            extractor.extract(crops)

        torch.cuda.synchronize()

        # Measure L1 YOLO
        l1_times = []
        for _ in range(20):
            start = time.perf_counter()
            detector.detect_batch(frames)
            torch.cuda.synchronize()
            l1_times.append((time.perf_counter() - start) * 1000)

        # Measure L2 SigLIP2
        l2_times = []
        for _ in range(20):
            start = time.perf_counter()
            extractor.extract(crops)
            torch.cuda.synchronize()
            l2_times.append((time.perf_counter() - start) * 1000)

        avg_l1 = np.mean(l1_times)
        avg_l2 = np.mean(l2_times)
        total = avg_l1 + avg_l2

        print(f"{batch_size:<12} | {avg_l1:<15.2f} | {avg_l2:<18.2f} | {total:<15.2f}")

    print("═" * 80)
    print(
        "💡 Notes: L1 handles 640x640, L2 handles 224x224. Values are per batch inference."
    )


if __name__ == "__main__":
    asyncio.run(run_latency_breakdown())
