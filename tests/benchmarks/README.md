# Saccade 效能評測手冊 (Benchmarks Guide)

本目錄包含 Saccade 核心管線的效能評測腳本，旨在量化系統在不同負載與硬體環境下的表現。測試範圍涵蓋從底層 C++ 傳輸到高層語義去重。

---

## 1. 測試環境 (Baseline)
- **OS**: Ubuntu 22.04 (WSL2 / Arch Linux)
- **GPU**: NVIDIA RTX 5070 Ti (Laptop)
- **CUDA/TensorRT**: CUDA 12.x / TensorRT 10.x
- **核心架構**: 工業級零拷貝 V2 (5-Buffer State-Machine)

---

## 2. 核心評測腳本

### A. 全管線端到端評測 (`bench_perception_pipeline.py`)
量化 L1 感知層與 L2 向量層的整合效能。
- **測試規模**: 預設執行 **5000 幀**，以獲取統計穩定的 Mean 與 P99 指標。
- **輸出規格**: 具備 **4 位小數精度 (High Precision)**，每 1000 幀輸出一次進度。
- **執行指令**:
  ```bash
  docker-compose exec saccade uv run python tests/benchmarks/bench_perception_pipeline.py
  ```
- **關鍵指標**:
    - **Throughput (FPS)**: 系統實測吞吐量 (目標 > 120 FPS)。
    - **Total E2E Latency**: 單幀總耗時 (目標 < 10 ms)。
    - **StdDev**: 延遲標準差，用於評估 WSL2 環境下的運算確定性。

### B. 極端壓力與邊緣案例 (`bench_stress_extreme.py`)
驗證系統在極限負載下的魯棒性與自適應策略。
- **測試內容**: 物件飽和、影格亂序、VRAM 臨界降級。
- **執行指令**:
  ```bash
  docker-compose exec saccade uv run python tests/benchmarks/bench_stress_extreme.py
  ```
- **極端情境定義**:
    1. **Object Saturation**: 單幀出現 >32 個物件，驗證 **面積優先截斷 (Salience-based)**。
    2. **Network Jitter**: 模擬亂序推入影格，驗證 **Reordering Buffer (150ms)**。
    3. **Emergency Level 3**: 模擬 VRAM >96%，驗證 **Target Culling (TTL 30->5)**。

---

## 3. 數據解讀指引

| 指標名稱 | 理想範圍 | 說明 |
| :--- | :--- | :--- |
| **media_grab** | < 0.1 ms | 驗證 5-Buffer Pool 的非同步抓取是否達成 Zero-Wait。 |
| **preprocess** | < 1.0 ms | 驗證 NPP GPU 預處理（NV12 -> RGB -> Resize）的效率。 |
| **yolo_inference**| < 5.0 ms | YOLO11n 在 TensorRT 下的核心推理耗時。 |
| **feature_extract**| < 2.0 ms | SigLIP 2 在 Batch Mode ($N=8$) 下的提取效率。 |
| **Jitter Buffer** | < 150 ms | 系統容忍影格亂序的最大時間窗口。 |

---

## 4. 異常狀態代碼

- **🚨 [LATENCY_SPIKE]**: 影格處理時間超過 200ms，系統將觸發降級警告。
- **🧹 [Target Culling]**: 系統處於 Level 3，正主動釋放非核心追蹤目標以防止 OOM。
- **📉 [Batch Truncated]**: 單幀物件過多，系統依據顯著性排序截斷了部分推理請求。

---

## 5. 自動化執行
建議在每次重大架構變更後，執行全量評測：
```bash
# 執行全量評測 (需先完成 C++ 編譯)
./tests/benchmarks/run_all.sh
```

最後更新：2026-04-14 (Saccade Benchmark Team)
