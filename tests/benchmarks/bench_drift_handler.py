import time
import torch
import numpy as np
from perception.drift_handler import SemanticDriftHandler


def benchmark_drift_handler(num_iterations=1000, batch_size=8):
    print(f"🚀 Starting Semantic Drift Handler Benchmark ({num_iterations} iterations, batch size {batch_size})...")
    feature_dim = 768
    handler = SemanticDriftHandler(similarity_threshold=0.95, feature_dim=feature_dim)

    # 模擬 8 個物件的 ID 與 特徵
    obj_ids = torch.arange(batch_size, device="cuda")
    features = torch.randn(batch_size, feature_dim, device="cuda")

    # 預熱
    _ = handler.filter_novel_features(obj_ids, features)
    torch.cuda.synchronize()

    latencies = []
    
    # 測試情境 1：全重複 (Worst case for filtering logic)
    print("🔹 Scenario 1: Redundant Features (100% hits)")
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = handler.filter_novel_features(obj_ids, features)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency_redundant = np.mean(latencies)
    print(f"  - Avg Latency: {avg_latency_redundant:.6f} ms")

    # 測試情境 2：全漂移 (Worst case for cache updates)
    print("🔹 Scenario 2: Semantic Drift (100% updates)")
    latencies = []
    for i in range(num_iterations):
        # 每次產生隨機特徵模擬漂移
        new_features = torch.randn(batch_size, feature_dim, device="cuda")
        start = time.perf_counter()
        _ = handler.filter_novel_features(obj_ids, new_features)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency_drift = np.mean(latencies)
    print(f"  - Avg Latency: {avg_latency_drift:.6f} ms")

    print("\n" + "═" * 60)
    print("🧠 Semantic Drift Handler Efficiency")
    print("═" * 60)
    print(f"Filter Throughput: { (1000 / avg_latency_redundant) * batch_size:,.0f} objects/sec")
    print(f"Update Throughput: { (1000 / avg_latency_drift) * batch_size:,.0f} objects/sec")
    print(f"P99 Latency (Drift): {np.percentile(latencies, 99):.6f} ms")
    print("Data Path:         Zero-Sync GPU Cosine Similarity")
    print("═" * 60)


if __name__ == "__main__":
    benchmark_drift_handler()
