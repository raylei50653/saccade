# Cognition & Agentic RAG (L5) 進度核對

## 系統架構層次
- [ ] **L5: Agentic RAG (認知與推理層)** - 規劃中

## 模組狀態：從傳統 VLM 轉向 Agentic RAG 架構

> 💡 **架構演進：** 
> 為了解決邊緣設備 VRAM 限制，Saccade 已將舊版 VLM 推理 (`llama-server`) 替換為更靈活的 **Agentic RAG**。
> 現在系統重點在於如何將 L1-L4 收集到的語義特徵，透過 Orchestrator 轉化為可被 LLM 檢索與推理的知識。

## L5: Agentic RAG (實作詳情)
- [x] **語義合成 (Semantic Synthesis)**: Orchestrator 已初步具備整合 YOLO 標籤與時空特徵的語義合成能力。
- [ ] **LlamaIndex 整合**: 規劃將時空數據對接到 LlamaIndex，建立 Agentic RAG 管線 (待實作)。
- [ ] **多模態檢索**: 結合文本與 Jina-CLIP 向量的跨模態檢索優化。

## 歷史里程碑 (已棄用項目)
- [x] **`llm_engine.py`**: 已從核心管線移除，改為隨插即用的 API 呼叫。
- [x] **`resource_manager.py`**: 由於不再頻繁加載多個大模型，權限已交還給全域系統調配。
- [x] **`yolo-vlm-backend.service`**: 相關服務已停用以釋放 6.5GB VRAM。

## 待處理
- [ ] 完成與本地 LlamaIndex 環境的對接實驗。
- [ ] 實作針對特定事件的「視覺重查 (Visual Re-query)」邏輯。

最後更新：2024-05-23
