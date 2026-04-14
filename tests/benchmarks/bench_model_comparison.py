import time
import torch
import os
import numpy as np
import multiprocessing
from perception.detector_trt import TRTYoloDetector
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()


def run_model_test(model_name, model_path, num_frames=300):
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name}: {model_path} not found.")
        return None

    print(f"\n🔥 Evaluating {model_name}...")
    detector = TRTYoloDetector(engine_path=model_path)
    media = MediaMTXClient(
        dummy_video=os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    )

    if not media.connect():
        return None

    # 等待串流完全穩定
    time.sleep(1.0)

    # 等待串流穩定並預熱 GPU
    print("⏳ Warming up GPU (10 iterations)...")
    warmup_count = 0
    while warmup_count < 10:
        ret, tensor = media.grab_tensor()
        if ret and tensor is not None:
            # 預熱推理
            with torch.no_grad():
                frame_chw = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                yolo_input = torch.nn.functional.interpolate(frame_chw, size=(640, 640))
                _ = detector.detect(yolo_input)
            warmup_count += 1
        time.sleep(0.01)

    latencies = []
    processed = 0
    while processed < num_frames:
        ret, tensor = media.grab_tensor()
        if not ret or tensor is None:
            time.sleep(0.001)
            continue

        start = time.perf_counter()
        with torch.no_grad():
            # 直接在 GPU 上進行張量轉換與縮放
            frame_chw = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            yolo_input = torch.nn.functional.interpolate(frame_chw, size=(640, 640))
            _ = detector.detect(yolo_input)

        # 確保 CUDA 操作完成再計時
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        processed += 1

    media.release()
    
    # 清理 GPU 記憶體，確保後續進程安全
    del detector
    torch.cuda.empty_cache()
    
    return {
        "avg": np.mean(latencies),
        "p99": np.percentile(latencies, 99),
        "fps": num_frames / (sum(latencies) / 1000),
    }

def _worker_process(model_name, model_path, queue):
    res = run_model_test(model_name, model_path)
    queue.put(res)

def run_isolated(model_name, model_path):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_process, args=(model_name, model_path, queue))
    p.start()
    p.join()
    if p.exitcode == 0:
        return queue.get()
    return None

if __name__ == "__main__":
    # 使用 spawn 模式確保 CUDA Context 完全隔離
    multiprocessing.set_start_method("spawn", force=True)
    
    print("🚀 [Benchmark] YOLO Generation Comparison (TRT Only)")
    print("═" * 70)

    results_11 = run_isolated("YOLO11n (Ultralytics)", "models/yolo/yolo11n.engine")
    results_26 = run_isolated("YOLO26n (Saccade Native)", "models/yolo/yolo26n.engine")

    print("\n" + "═" * 70)
    print(
        f"{'Model Variant':<25} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'Est. FPS'}"
    )
    print("-" * 70)
    if results_11:
        print(
            f"{'YOLO11n (FP16 TRT)':<25} | {results_11['avg']:10.2f} | {results_11['p99']:10.2f} | {1000 / results_11['avg']:8.2f}"
        )
    if results_26:
        print(
            f"{'YOLO26n (FP16 TRT)':<25} | {results_26['avg']:10.2f} | {results_26['p99']:10.2f} | {1000 / results_26['avg']:8.2f}"
        )
    print("═" * 70)
    print("💡 YOLO26n features NMS-free architecture for lower latency.")
