# Saccade Core Benchmarks 📊

本目錄包含 Saccade 專案的核心效能基準測試，用於量化系統在邊緣 AI 環境下的極限表現。

## 🚀 快速執行
在 Docker 容器內執行總控腳本，即可獲得完整報告：
```bash
./tests/benchmarks/run_all.sh
```

## 核心測試清單

### 1. 核心傳輸 (`bench_core_transport.py`)
*   **目標**：驗證 C++ GPU 緩衝池與 Python PyTorch 的零拷貝對接效率。
*   **關鍵指標**：Acquisition Latency (預期 < 0.01ms)。

### 2. 感知全鏈路 (`bench_perception_pipeline.py`)
*   **目標**：拆解全生命週期延遲 (Grab -> Preprocess -> YOLO -> Post-processing)。
*   **關鍵指標**：P99 延遲、平均 E2E 耗時。

### 3. 模型對比 (`bench_model_comparison.py`)
*   **目標**：追蹤不同世代 YOLO 模型在當前環境下的推理速度與吞吐量差異。
*   **關鍵指標**：Average Latency, Peak FPS。

### 4. 向量存儲 (`bench_storage_vector.py`)
*   **目標**：測試 ChromaDB 向量資料庫在高頻寫入與檢索下的表現。
*   **關鍵指標**：Query Latency, Write Throughput。

---
## 歸檔說明
過時、重複或研發中的原型腳本已移至 `archive/` 目錄，以保持主目錄整潔。
