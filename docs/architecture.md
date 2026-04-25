# Saccade 系統架構說明書 (Vision-Vector Pipeline)

Saccade 採用「模組化雙軌感知架構」，將邊緣端的即時偵測與長效語義記憶解耦。系統邏輯劃分為六個核心層級 (L1-L6)，確保高吞吐量與低延遲。

---

## 1. 邏輯分層架構 (L1-L6 Framework)

| 層級 | 名稱 | 核心技術 | 職責定義 | 對應目錄 |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | **感知層 (Perception)** | YOLO26 (TRT) + GPUByteTracker (C++/CUDA) | NMS-Free 極速物件偵測、GPU 追蹤、GMC、光線補償。 | `perception/`, `src/` |
| **L2** | **去重層 (Deduplication)** | SigLIP 2 / Saccade Heartbeat / FeatureBank | 語義漂移檢測，過濾冗餘影格；跨鏡頭 Re-ID。 | `perception/` |
| **L3** | **緩衝層 (Streaming)** | Redis / MicroBatcher / asyncio | 事件非同步推送與微批次聚合。 | `storage/`, `pipeline/` |
| **L4** | **儲存層 (Vector DB)** | ChromaDB / HNSW | 多維度向量索引（語義 + 時間 + 標籤）；定期 snapshot 備份。 | `storage/` |
| **L5** | **認知層 (Agentic RAG)** | LlamaIndex + Ollama + ChromaDB | 事件觸發式語義推理；Visual Re-query；歷史事件脈絡分析。 | `pipeline/` |
| **L6** | **資源層 (Resource Management)** | NVML / ResourceManager | 實時 VRAM 監控與階梯式降級決策。 | `cognition/` |

---

## 2. 核心設計原則

### Zero-Copy 原則
所有影像處理流程（RTSP 解碼 → YOLO26 偵測 → ROI 裁切 → SigLIP 2 特徵提取）完全維持在 GPU VRAM 中。主循環內禁止呼叫 `.cpu()` 或轉為 `numpy`，以消除 PCIe 頻寬瓶頸。

### Agentic RAG（L5 認知推理）
系統採用 **LlamaIndex + Ollama** 在邊緣設備本地執行語義推理，不依賴外部 API。僅在高熵場景（entropy > 0.9）或偵測異常時才觸發查詢，避免對感知主循環造成負載。

### 階梯式降級（L6 自適應）
透過 `ResourceManager` 在 VRAM 壓力不同級別下自動降級，保證核心感知（L1）優先於語義提取（L2）與推理（L5）。

---

## 3. 數據流向 (Data Flow)

```
RTSP Stream
    │
    ▼
[L1] GstClient (C++ / nvh264dec) → 5-Buffer GPU Pool
    │  Zero-Copy CUDA Tensor
    ▼
[L1] YOLO26 TensorRT (NMS-Free) + GPUByteTracker
    │  bbox, track_id, score, class
    ▼
[L2] Saccade Heartbeat (每 10 幀) → SigLIP 2 TRT → FeatureBank
    │  768-dim embedding, drift_score
    ▼
[L3] RedisCache.publish_event() → MicroBatcher (100ms)
    │  JSON event batch
    ▼
[L4] ChromaStore.add_memory() → ChromaDB HNSW
    │  indexed vectors + metadata
    ▼
[L5] Orchestrator (entropy > 0.9 / anomaly)
    │  → LlamaIndex ReAct Agent
    │  → visual_requery / semantic_search / get_track_history
    ▼
[L6] ResourceManager (NVML) → 降級指令 → L1/L2/L5
```

---

## 4. 追蹤器架構（Tracker Stack）

```
SmartTracker (Python 協調層)
├── GPUByteTracker (C++/CUDA)        ← 核心匹配引擎
│   ├── Dual-stage Sinkhorn           高/低分偵測框二次匹配
│   ├── ReID Fusion Cost Matrix       (1-w)*IoU + w*CosSim
│   ├── Strong ReID Gate              CosSim > 0.75 強制配對
│   ├── GPU Kalman Filter             predict + update kernel
│   └── GMC kernel                   仿射矩陣修正 Kalman 狀態
├── GMC 計算 (OpenCV optical flow)   逐幀仿射矩陣 → C++
├── Light Compensation               frame 亮度 → light_factor → R 矩陣
├── Saccade Heartbeat (% 10)         每 10 幀觸發 SigLIP 2 特徵更新
└── ReorderingBuffer (150ms)         並行亂序修正
```

---

## 5. 資料夾功能定義

- **`src/` + `include/`**: C++/CUDA 核心，包含 GPUByteTracker、Kalman Filter、Sinkhorn/Hungarian/Auction 匹配算法、GstClient。
- **`perception/`**: 視覺算法層。YOLO26 偵測、SigLIP 2 特徵提取、SmartTracker、FeatureBank、跨鏡頭 Re-ID。
- **`pipeline/`**: 系統中樞。Orchestrator（Agentic RAG 調度）、HealthChecker。
- **`storage/`**: 數據終點。RedisCache（MicroBatcher）、ChromaStore（向量索引 + 備份）。
- **`media/`**: 影音串流接入、硬體解碼、RTSP Watchdog。
- **`cognition/`**: ResourceManager（VRAM 監控與降級）、FrameSelector。

---

## 6. 開發約定

- **模組化**: 每個層級可獨立升級。C++ 核心透過 `pybind11` 提供 Python 介面。
- **非同步設計**: 所有 I/O（Redis、ChromaDB、RAG 查詢）必須使用 `asyncio`，嚴禁阻塞感知主循環。RAG 查詢使用 `run_in_executor` 包裝。
- **資源安全**: `ResourceManager` 嚴格限制 VRAM 使用。核心管線（YOLO26 + SigLIP 2）約需 ~450MB，總估算含追蹤器 < 1.5GB。
- **Zero-Copy First**: 任何新增路徑在引入 CPU 拷貝前需在 ADR 中說明理由。
