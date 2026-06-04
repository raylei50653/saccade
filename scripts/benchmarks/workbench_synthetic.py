import torch
import time
import argparse
import concurrent.futures
from saccade.perception.detector_trt import BatchingTRTDetector
from saccade.perception.workbench import Workbench
from saccade_tracking_ext import PerceptionPipelineConfig


def worker_loop(
    workbench: Workbench,
    num_iters: int,
    frame: torch.Tensor,
    frame_w: int,
    frame_h: int,
):
    # Warmup
    for _ in range(10):
        workbench.process_frame(frame, frame_w=frame_w, frame_h=frame_h)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iters):
        workbench.process_frame(frame, frame_w=frame_w, frame_h=frame_h)
    torch.cuda.synchronize()
    end_time = time.time()

    return end_time - start_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", type=str, default="models/yolo/yolo26s_960_batch4.engine"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading engine {args.engine} with batch_size={args.threads}...")
    batcher = BatchingTRTDetector(args.engine, batch_size=args.threads)

    cfg = PerceptionPipelineConfig()

    workbenches = [
        Workbench(batcher.make_proxy(), cfg, device="cuda:0")
        for _ in range(args.threads)
    ]

    frame_w, frame_h = 1920, 1080
    frame = torch.randn((3, 960, 960), device="cuda", dtype=torch.float32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(worker_loop, wb, args.iters, frame, frame_w, frame_h)
            for wb in workbenches
        ]

        times = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_frames = args.threads * args.iters
    fps = total_frames / max(times)

    print(f"Aggregate FPS: {fps:.1f}")

    for i, wb in enumerate(workbenches):
        if hasattr(wb, "stats"):
            s = wb.stats
            c = s["count"]
            print(
                f"WB {i} average per frame: submit={s['submit'] / c * 1000:.2f}ms, prep={s['prep'] / c * 1000:.2f}ms, cpp={s['cpp'] / c * 1000:.2f}ms"
            )


if __name__ == "__main__":
    main()
