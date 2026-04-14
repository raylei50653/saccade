# Saccade 系統架構說明書 (Vision-Vector Pipeline)

Saccade 採用「模組化雙軌感知架構」，將邊緣端的即時偵測與長效語義記憶解耦。系統邏輯劃分為六個核心層級 (L1-L6)，確保高吞吐量與低延遲。

---

## 1. 邏輯分層架構 (L1-L6 Framework)

| 層級 | 名稱 | 核心技術 | 職責定義 | 對應目錄 |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | **感知層 (Perception)** | YOLO26 (C++ TensorRT) | NMS-Free 極速物件偵測、GPU 追蹤與 ROI 提取。 | `perception/`, `src/` |
| **L2** | **去重層 (Deduplication)** | SigLIP 2 / Cosine Sim | 語義漂移檢測 (Semantic Drift)，過濾冗餘影格。 | `perception/` |
| **L3** | **緩衝層 (Streaming)** | Redis Streams / asyncio | 事件非同步推送與微批次聚合 (Micro-batching)。 | `storage/`, `pipeline/` |
| **L4** | **儲存層 (Vector DB)** | ChromaDB / HNSW | 多維度向量索引 (語義 + 時間 + 標籤)。 | `storage/` |
| **L5** | **應用層 (Retrieval API)**| FastAPI / Semantic Search | 將偵測事件轉化為可檢索的結構化數據與語義搜尋。 | `api/`, `pipeline/` |
| **L6** | **認知與資源層 (Cognition)**| 自適應幀率、VRAM 監控 | 處理高層次的資源監控與決策（如 Frame Selection, Resource Management）。 | `cognition/` |

---

## 2. 核心設計原則

### 🚀 Zero-Copy 原則 (零拷貝)
所有的影像處理流程（從 RTSP 解碼到 YOLO26 偵測、ROI 裁切、最後到 SigLIP 2 特徵提取）完全維持在 **GPU VRAM** 中。禁止在主循環內呼叫 `.cpu()` 或轉為 `numpy`，以消除 PCIe 頻寬瓶頸。

### 🧠 VLM-Free 語義合成
為了在極低 VRAM 的邊緣設備達成極致效能，系統捨棄了緩慢的端到端 VLM 與複雜的 Agentic LLM 推理。改用「YOLO26 偵測 + SigLIP 2 嵌入」的組合方式，在 `Orchestrator` 中動態聚合場景特徵，實現「具備視覺理解力但無推理延遲」的高效管線。

---

## 3. 數據流向 (Data Flow)

1.  **Ingestion**: `media/` 透過 C++ GStreamer (nvh264dec) 將串流載入為 CUDA Tensor。
2.  **Fast Path (L1)**: YOLO26 在主 CUDA Stream 執行偵測與 GPU ByteTrack 追蹤。
3.  **Vector Path (L2)**: 針對新物體或產生位移的物體，在背景 Stream 進行 SigLIP 2 特徵提取。
4.  **Buffering (L3)**: 事件進入 Redis 隊列，由 `Orchestrator` 進行微批次聚合。
5.  **Persistence (L4)**: 批次寫入 `ChromaStore`。
6.  **Retrieval (L5)**: 使用者透過 FastAPI 進行語義搜尋與時空檢索。

---

## 4. 資料夾功能定義

- **`src/` 與 `include/`**: C++ 核心感知與媒體層擴充套件，負責最底層的極速運算。
- **`perception/`**: 視覺算法層。包含 YOLO26 偵測、裁切、特徵提取與語義去重。
- **`pipeline/`**: 系統的中樞神經。負責跨層級的非同步調度與系統健康監控。
- **`storage/`**: 數據的終點。處理實時快取 (Redis) 與長效記憶 (ChromaDB)。
- **`media/`**: 負責影音串流接入與硬體解碼。
- **`cognition/`**: 處理高層次的資源監控與決策。

---

## 5. 開發約定與擴展性

- **模組化**: 每個層級均可獨立升級。核心模組由 C++ 實作並透過 `pybind11` 提供 Python 介面。
- **非同步設計**: 所有的 I/O 操作（Redis, DB, API）必須使用 `asyncio`，嚴禁阻塞感知主循環。
- **資源安全**: 透過 `ResourceManager` 嚴格限制顯存使用，預估核心管線僅需 ~1.5GB VRAM。
