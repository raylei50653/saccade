# Cognition Module (認知推理)

## 📐 模組職責
負責本地 Llama 3 邊緣 Agentic RAG 本地推理、高熵事件分析與視覺二次查詢 (Visual Requery)。

## 🟢 目前現況
* **RAG 編排器 (Orchestrator)**：
  * 整合 LlamaIndex，預設加載本地 `BAAI/bge-small-en-v1.5` 為 Embedding，加載本地 Ollama `llama3:8b` 為推理大模型。
  * 將 ChromaDB 包裝成 `ChromaVectorStore` 並加載 `ReActAgent`。
  * 註冊 **Visual Re-query 工具**：包裝成 `FunctionTool` 嵌入 `ReActAgent`，當 LLM 需要確認特徵時，可透過傳入的 `track_id` 主動檢索 ChromaDB 中的歷史視覺向量特徵。
* **VRAM 壓力感知整合**：
  * 每個事件都透過 `VRAMLevelReader.read_snapshot()` 讀取 dispatcher 發佈的降級快照。FRESH `FAST_PATH` 時不啟動新的 RAG，只寫入記憶；FRESH `EMERGENCY` 時刻意丟棄非 anomaly 事件（計入 `shed_events`）。level 不是 FRESH（STALE／MISSING／INVALID）時，所有事件都嘗試寫入、不啟動新的 RAG，並計入 `unknown_events`；計數器由 `vram_policy_stats()` 提供。這不保證不遺失資料，也不保證不使用 GPU，詳見 [resource 契約](../resource/architecture_degradation.md#6-vram-level-跨程序快照契約537-b0-f07)。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | 外圍慢路徑（見 [pipeline_flow.md](../../reference/pipeline_flow.md) §4.2） |
| **輸入** | Redis stream batch（scene events，來自 [storage](../storage/README.md)） |
| **輸出** | scene description + metadata → ChromaDB；高熵事件觸發 RAG query / visual requery |
| **上游 → 下游** | `Redis stream → orchestrator → scene desc → Chroma write →(高熵)→ ReActAgent RAG / visual requery`；受 VRAM level 降級（FAST_PATH 跳 RAG） |

## ⚖️ GO / NO-GO 決策

🟢 穩定落地，無 active ablation。

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
