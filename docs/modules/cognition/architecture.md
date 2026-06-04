# L5: 認知層 (Cognition - Agentic RAG)

## 1. 定義與目標
L5 是 Saccade 的「語義大腦」，負責將 L1-L4 累積的結構化視覺記憶轉化為可推理的知識。目標是在邊緣設備上提供事件觸發式的語義推理，回答「過去一小時是否有可疑人物？」等複雜查詢，同時不干擾 L1 感知主循環。

## 2. 核心組件
- **Orchestrator** (`src/saccade/cognition/orchestrator.py`): 事件監聽主迴圈，負責接收 Redis 事件並決定是否觸發 RAG。
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
