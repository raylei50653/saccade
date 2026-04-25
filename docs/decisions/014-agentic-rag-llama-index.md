# ADR 014: Agentic RAG Integration with LlamaIndex

## 背景 (Background)
Saccade 的 `Orchestrator` 目前僅是一個基於規則的引擎，缺乏對歷史事件的語義理解與長短期記憶推理能力。為了達成 L5 (Agentic RAG) 的目標，我們需要引入一個成熟的檢索增強生成架構。

## 決策 (Decision)
我們決定採用 **LlamaIndex** 作為 Saccade L5 層的核心 RAG 框架，並對接到現有的 ChromaDB 向量存儲。

### 1. 架構組件
- **Vector Store**: 使用已有的 `ChromaStore` (對應 `saccade_memory` collection)。
- **LlamaIndex Integration**: 透過 `ChromaVectorStore` 與 `StorageContext` 封裝 ChromaDB。
- **Agentic Loop**: 
    - 採用 `ReAct` 或 `FunctionCalling` Agent 模式。
    - 工具集：`semantic_search` (搜尋過去事件), `get_track_history` (獲取特定物件軌跡), `visual_requery` (提取特徵並執行跨幀比對)。

### 2. 觸發機制 (Triggering)
- **High-Entropy Trigger**: 當影格 Entropy > 0.8 時觸發 RAG 進行場景脈絡分析。
- **Anomaly Trigger**: 當偵測到 `risk_objects` 時觸發。
- **Periodic Summary**: 每隔一定時間（如 5 分鐘）進行一次語義總結。

### 3. 本地化優先 (Edge-First)
- **LLM**: 預設對接本地 **Ollama** (如 `llama3:8b` 或 `phi3`) 或使用小型 ONNX/TensorRT 加速的模型，以符合邊緣運算限制。
- **Embedding**: 使用 `BAAI/bge-small-en-v1.5` 本地 Embedding，不呼叫外部 API。

## 影響 (Consequences)
- **優點**: 提供強大的語義推理能力，能夠回答如「過去一小時內是否有可疑人物出沒？」等複雜查詢。
- **缺點**: 增加 CPU/GPU 負載。需嚴格控制 Agent 的呼叫頻率。
- **依賴**: 需新增 `llama-index`, `llama-index-vector-stores-chroma` 等依賴。

## 狀態 (Status)
**已落地 (Accepted)**
