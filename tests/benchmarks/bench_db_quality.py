import time
from storage.chroma_store import ChromaStore
from datetime import datetime

def benchmark_db_quality():
    print("🚀 [Benchmark] Starting ChromaDB Storage & Retrieval Quality Test...")
    store = ChromaStore(collection_name="quality_test")
    
    # 1. 寫入多樣化場景數據
    test_data = [
        {"content": "A white dog is running on the green grass.", "meta": {"category": "animal", "risk": "low"}},
        {"content": "A person wearing a black jacket is holding a knife near the ATM.", "meta": {"category": "security", "risk": "high"}},
        {"content": "Traffic is heavy at the main intersection, many cars are stationary.", "meta": {"category": "traffic", "risk": "medium"}},
        {"content": "A fire was detected in the kitchen area with smoke rising.", "meta": {"category": "emergency", "risk": "critical"}},
        {"content": "The parking lot is empty, only one blue motorcycle is parked.", "meta": {"category": "parking", "risk": "low"}}
    ]
    
    print(f"📥 Writing {len(test_data)} sample memories...")
    for item in test_data:
        store.add_memory(content=item["content"], metadata=item["meta"])
    
    time.sleep(1) # 等待索引更新
    
    # 2. 測試語義檢索準確度 (Semantic Recall)
    queries = [
        {"q": "Is there any weapon or immediate danger?", "expected": "security"},
        {"q": "How is the road condition?", "expected": "traffic"},
        {"q": "Any animals in the field?", "expected": "animal"},
        {"q": "Is there a fire?", "expected": "emergency"}
    ]
    
    print("\n🔍 Testing Semantic Retrieval Quality:")
    score = 0
    for query in queries:
        results = store.query_memories(query["q"], n_results=1)
        # 取得最匹配的一筆
        top_doc = results["documents"][0][0]
        top_meta = results["metadatas"][0][0]
        
        # 檢查類別是否符合預期
        is_correct = top_meta["category"] == query["expected"]
        status = "✅ PASS" if is_correct else "❌ FAIL"
        if is_correct: score += 1
        
        print(f"  - Query: '{query['q']}'")
        print(f"    Match: '{top_doc[:50]}...' (Category: {top_meta['category']}) -> {status}")

    # 3. 測試持久化穩定性
    print(f"\n📊 --- [ Quality Summary ] ---")
    print(f"  - Semantic Accuracy: {score}/{len(queries)} ({score/len(queries)*100:.1f}%)")
    print(f"  - Vector DB Latency: Testing query speed...")
    
    start_bench = time.perf_counter()
    for _ in range(10):
        _ = store.query_memories("random search", n_results=1)
    avg_latency = (time.perf_counter() - start_bench) / 10 * 1000
    print(f"  - Avg Query Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    benchmark_db_quality()
