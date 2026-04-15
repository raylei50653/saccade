import time
import torch
import numpy as np
from perception.drift_handler import SemanticDriftHandler
from cognition.resource_manager import DegradationLevel


def benchmark_drift_handler(num_iterations=1000, batch_size=8):
    print(
        f"🚀 Starting Semantic Drift Handler Benchmark ({num_iterations} iterations, batch size {batch_size})..."
    )
    feature_dim = 768
    # 修正：移除不支援的 feature_dim 參數
    handler = SemanticDriftHandler(similarity_threshold=0.95)

    # 模擬 8 個物件的 ID 與 特徵
    track_ids = list(range(batch_size))
    features = torch.randn(batch_size, feature_dim, device="cuda")
    # 模擬 BBox [x1, y1, x2, y2]
    boxes = torch.tensor(
        [[0, 0, 100, 100]] * batch_size, device="cuda", dtype=torch.float32
    )

    # 預熱
    _ = handler.filter_for_batch(track_ids, boxes, DegradationLevel.NORMAL)
    handler.update_history(track_ids, features)
    torch.cuda.synchronize()

    latencies_filter = []
    latencies_update = []

    # 測試情境 1：過濾邏輯 (Filtering logic)
    print("🔹 Scenario 1: Filtering Logic")
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = handler.filter_for_batch(track_ids, boxes, DegradationLevel.NORMAL)
        torch.cuda.synchronize()
        latencies_filter.append((time.perf_counter() - start) * 1000)

    avg_latency_filter = np.mean(latencies_filter)
    print(f"  - Avg Filter Latency: {avg_latency_filter:.6f} ms")

    # 測試情境 2：語義更新 (Semantic updates)
    print("🔹 Scenario 2: Semantic Update (History update)")
    for i in range(num_iterations):
        # 每次產生隨機特徵模擬漂移
        new_features = torch.randn(batch_size, feature_dim, device="cuda")
        start = time.perf_counter()
        handler.update_history(track_ids, new_features)
        torch.cuda.synchronize()
        latencies_update.append((time.perf_counter() - start) * 1000)

    avg_latency_update = np.mean(latencies_update)
    print(f"  - Avg Update Latency: {avg_latency_update:.6f} ms")

    print("\n" + "═" * 60)
    print("🧠 Semantic Drift Handler Efficiency")
    print("═" * 60)
    print(
        f"Filter Throughput: {(1000 / avg_latency_filter) * batch_size:,.0f} objects/sec"
    )
    print(
        f"Update Throughput: {(1000 / avg_latency_update) * batch_size:,.0f} objects/sec"
    )
    print(f"P99 Latency (Update): {np.percentile(latencies_update, 99):.6f} ms")
    print("Data Path:         Zero-Sync GPU Centroid Update")
    print("═" * 60)


if __name__ == "__main__":
    benchmark_drift_handler()
