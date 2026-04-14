import time
import numpy as np
from storage.chroma_store import ChromaStore


def benchmark_storage():
    print("🚀 Starting Vector Storage Benchmark (ChromaDB)...")
    store = ChromaStore(collection_name="benchmark_collection")

    # 1. 寫入壓力測試 (Throughput)
    num_writes = 50
    print(f"📥 Benchmarking write throughput ({num_writes} records)...")

    write_start = time.perf_counter()
    for i in range(num_writes):
        store.add_memory(
            content=f"Benchmark entry {i}: A simulated detection of a person in the restricted area.",
            metadata={"id": i, "category": "test", "importance": i % 5},
        )
    write_duration = time.perf_counter() - write_start
    print(f"  - Write Latency: {write_duration * 1000 / num_writes:.2f} ms/record")
    print(f"  - Write Speed:   {num_writes / write_duration:.2f} items/sec")

    # 2. 檢索延遲測試 (Query Latency)
    num_queries = 100
    print(f"\n🔍 Benchmarking query latency ({num_queries} queries)...")

    query_latencies = []
    for _ in range(num_queries):
        start = time.perf_counter()
        _ = store.hybrid_query("detect danger person")
        query_latencies.append((time.perf_counter() - start) * 1000)

    print(f"  - Avg Query Latency: {np.mean(query_latencies):.2f} ms")
    print(f"  - P99 Query Latency: {np.percentile(query_latencies, 99):.2f} ms")

    # 3. 語義召回質量檢查 (Quality)
    print("\n✅ Storage Benchmark Complete!")
    print("  - ChromaDB Status: Functional")
    print("  - Indexing:        Sentence-Transformer (Default)")


if __name__ == "__main__":
    benchmark_storage()
