# L5: 認知層 (Agentic RAG)

## 1. 定義與目標
L5 是 Saccade 的「語義大腦」，負責將 L1-L4 累積的結構化視覺記憶轉化為可推理的知識。目標是在邊緣設備上提供事件觸發式的語義推理，回答「過去一小時是否有可疑人物？」等複雜查詢，同時不干擾 L1 感知主循環。

## 2. 核心組件
- **Orchestrator** (`pipeline/orchestrator.py`): 事件監聽主迴圈，負責接收 Redis 事件並決定是否觸發 RAG。
- **LlamaIndex RAG Engine**: 連接 ChromaDB 的向量索引，提供 ReAct Agent 工具集。
- **本地 LLM (Ollama)**: 預設 `llama3:8b`，不呼叫外部 API，符合邊緣運算限制。
- **Local Embedding**: `BAAI/bge-small-en-v1.5`，與 L4 向量空間對齊。

## 3. 觸發機制
- **High-Entropy Trigger**: 影格 entropy > 0.9，觸發場景脈絡分析。
- **Anomaly Trigger**: 偵測到 `risk_objects`（knife, fire, smoke 等）。
- 以上條件未滿足時，Orchestrator 僅執行索引寫入，不觸發 LLM，避免資源浪費。

## 4. ReAct Agent 工具集
| Tool | 功能 |
|---|---|
| `semantic_search` | 搜尋 ChromaDB 歷史相似場景（文字描述 → 向量查詢） |
| `get_track_history` | 取得特定 track_id 在過去 N 分鐘的軌跡記錄 |
| `visual_requery` | 從 FeatureBank 拉取 SigLIP 2 embedding → ChromaDB Image-to-Image 比對 |

## 5. 資料流向
- **Input**: Redis `saccade:events` 佇列、FeatureBank embedding、ChromaDB 歷史向量。
- **Output**: LLM Insight 文字輸出（目前 print，後續可接 API）。

## 6. 效能保護
- RAG 查詢使用 `asyncio.run_in_executor` 包裝，防止同步阻塞主迴圈。
- L6 進入 FAST_PATH 時，Orchestrator 跳過 RAG 觸發，僅執行儲存。

---

# L6: 資源層 (Resource Management)

## 1. 定義與目標
L6 是 Saccade 的「決策大腦」，負責高層級的 VRAM 資源監控與系統平衡。目標是確保系統在資源極限環境下透過自適應策略優雅降級，優先保證核心感知（L1）。

## 2. 核心組件
- **ResourceManager** (`cognition/resource_manager.py`): 透過 NVML 實時監測 VRAM 負載，輸出降級指令。
- **FrameSelector** (`cognition/frame_selector.py`): 基於 L2 漂移分數動態調整 L1 偵測頻率。

## 3. 資料流向
- **Input**: VRAM Stats (NVML)、Latency Spike (L1)、Drift Score (L2)。
- **Output**: DegradationLevel（NORMAL / REDUCED / FAST_PATH / EMERGENCY）。

## 4. 階梯式降級策略

| Level | VRAM 閾值 | 動作 |
|---|---|---|
| NORMAL | < 85% | 正常運行 |
| REDUCED | > 85% | 縮減 5-Buffer Pool 大小 |
| FAST_PATH | > 92% | 暫停 L2（SigLIP 2）與 L5（RAG） |
| EMERGENCY | > 96% | 解析度 640→320、Target Culling（Confidence < 0.4）、track_buffer 30→10 |

## 5. Hysteresis（遲滯保護）
- 升級門檻：85% / 92% / 96%
- 降級恢復：需降至觸發點 **-5%** 才恢復上一級，防止臨界點頻繁切換（Thrashing）。
