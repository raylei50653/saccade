import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from perception.detector_trt import TRTYoloDetector  # noqa: E402


def run_benchmark(engine: str, batches: list[int], warmup: int, iters: int) -> None:
    detector = TRTYoloDetector(engine_path=engine)
    print(f"Benchmarking {engine}")
    print(f"{'batch':>5} {'mean_ms':>10} {'p99_ms':>10} {'fps':>10}")

    for batch_size in batches:
        input_tensor = torch.rand(
            (batch_size, 3, 640, 640), device=detector.device, dtype=torch.float32
        )

        for _ in range(warmup):
            detector.detect_batch(input_tensor, conf_threshold=0.25)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            detector.detect_batch(input_tensor, conf_threshold=0.25)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        arr = np.asarray(times)
        fps = batch_size * 1000.0 / arr.mean()
        print(
            f"{batch_size:5d} {arr.mean():10.3f} {np.percentile(arr, 99):10.3f} {fps:10.1f}"
        )


def parse_batches(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO TensorRT batch latency.")
    parser.add_argument("--engine", default="models/yolo/yolo26m_batch4.engine")
    parser.add_argument("--batches", default="1,2,3,4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.engine, parse_batches(args.batches), args.warmup, args.iters)


if __name__ == "__main__":
    main()
