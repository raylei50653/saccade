#!/bin/bash
# Saccade Core Benchmark Suite 🚀

export PYTHONPATH=$PYTHONPATH:$(pwd)/build
echo "=================================================="
echo "📊 Saccade High-Performance AI Suite Benchmark"
echo "=================================================="

# 1. 核心傳輸 (C++ Pool -> Python Zero-Copy)
echo -e "\n[1/4] Testing Core Transport Latency..."
uv run python tests/benchmarks/bench_core_transport.py

# 2. 感知全鏈路 (E2E Pipeline Stress)
echo -e "\n[2/4] Testing Full Perception Pipeline..."
uv run python tests/benchmarks/bench_perception_pipeline.py

# 3. 模型對比 (YOLO11 vs YOLO26)
echo -e "\n[3/4] Testing Model Generation Comparison..."
uv run python tests/benchmarks/bench_model_comparison.py

# 4. 存儲效能 (ChromaDB Vector Store)
echo -e "\n[4/4] Testing Vector Storage Performance..."
uv run python tests/benchmarks/bench_storage_vector.py

echo -e "\n=================================================="
echo "✅ All Benchmarks Completed!"
echo "=================================================="
