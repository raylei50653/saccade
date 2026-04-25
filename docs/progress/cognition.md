# Cognition & Agentic RAG (L5) 進度核對

## 系統架構層次
- [x] **L5: Agentic RAG (認知與推理層)** — 已完成

## 模組狀態：Agentic RAG 管線完整落地

> **架構演進：**
> 為了解決邊緣設備 VRAM 限制，Saccade 已將舊版 VLM 推理 (`llama-server`) 替換為 **Agentic RAG**。
> 系統重點在於將 L1-L4 收集到的語義特徵，透過 Orchestrator 轉化為可被 LLM 檢索與推理的知識。

## L5: Agentic RAG (實作詳情)
- [x] **語義合成 (Semantic Synthesis)**: Orchestrator 整合 YOLO 標籤與時空特徵生成結構化場景描述。
- [x] **LlamaIndex 整合 (ADR 014)**: ChromaDB → LlamaIndex VectorStoreIndex，local embedding（BAAI/bge-small-en-v1.5）+ Ollama llama3，不呼叫外部 API。
- [x] **事件觸發式查詢**: `entropy > 0.9` 或 `is_anomaly=True` 時才觸發 RAG，`run_in_executor` 防止阻塞主迴圈。
- [x] **Visual Re-query**: Orchestrator 註冊 `visual_requery` Tool 給 ReAct Agent，LLM 可發起 ChromaDB 純向量搜尋（Image-to-Image 語義比對）。
- [x] **跨鏡頭 Re-ID**: `FeatureBank` 重構支援 `stream_map`，`find_cross_camera_matches` 矩陣運算讓多路共享特徵索引。

## L6: Resource Management (資源調度層)
- [x] **動態資源調配 (ResourceManager)**:
    - 滯後控制 (Hysteresis)，防止在臨界點頻繁切換（Thrashing）。
    - 門檻：90% 觸發降級，降至 85% 才恢復正常模式。
    - 階梯式降級：NORMAL → REDUCED → FAST_PATH → EMERGENCY。

## 歷史里程碑 (已棄用)
- [x] **`llm_engine.py`**: 已從核心管線移除，改為 Agentic RAG。
- [x] **`yolo-vlm-backend.service`**: 停用以釋放 6.5GB VRAM。

## 待處理

（已清空）

最後更新：2026-04-25
