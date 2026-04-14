# Saccade：針對邊緣 AI 的高效率雙軌視覺感知系統

**邊緣端連續視覺推理與語義索引系統。**

---

> **🌍 [English Version](README.md)**

---

## 1. 專題動機與問題定義

### 目前面臨的問題
傳統視訊監控系統主要為「即時監控」設計，缺乏 **長效語義記憶**，這導致了幾個關鍵瓶頸：
*   **檢索效率低下**：要搜尋特定的歷史事件（例如：「尋找昨天的紅色貨車」）需要數小時的人工回放。
*   **運算資源浪費**：在邊緣設備上對每一影格執行沉重的視覺語言模型 (VLM) 是不切實際的。
*   **儲存膨脹**：儲存每一影格會產生大量冗餘，且無法提供具備意義的資訊摘要。

### 專題目標
Saccade 旨在將原始視訊流轉化為 **可搜尋的視覺記憶系統**。透過模仿人類視覺系統的「掃視」機制，我們將高速感知與深層語義理解解耦，在資源受限的邊緣硬體上實現自然語言檢索。

## 2. 核心解決方案：雙軌感知架構

Saccade 的邏輯架構橫向擴展為六個核心層級 (L1-L6)，系統性地整合了多項開源技術，確保高吞吐量與低資源消耗：

```` Mermaid
graph TD
    %% 自定義樣式 (配合原圖的淺綠色調)
    classDef default fill:#d1fae5,stroke:#10b981,stroke-width:1px,color:#065f46;
    classDef transparent fill:none,stroke:none;
    classDef db fill:#d1fae5,stroke:#10b981,stroke-width:1px,color:#065f46;

    Stream["📡 RTSP / WebRTC 串流<br>MediaMTX 接收"]

    subgraph L6 ["L6 認知層（全域監控）"]
        direction TB
        Monitor["psutil / pynvml<br>VRAM ≤ 1.5 GB 監控"]
        Action["動態調節<br>降幀率 / 卸載模型"]
        Monitor -- "節流 / 卸載" --> Action
    end

    subgraph Ingestion ["媒體接入層"]
        direction TB
        GStreamer["GStreamer<br>nvh264dec 硬體解碼"]
        GPUPool["5-Buffer GPU Pool<br>NV12 格式駐留 VRAM"]
        GStreamer --> GPUPool
    end

    Stream --> GStreamer

    subgraph L1 ["L1 感知層 · Fast Path（主 CUDA Stream）"]
        direction TB
        YOLO["YOLO26<br>Native TensorRT 推理<br>~3.53 ms"]
        ByteTrack["GPU ByteTrack<br>軌跡追蹤"]
        YOLO --> ByteTrack
    end

    GPUPool -- "Zero-Copy 指標<br>0.0004 ms" --> YOLO

    %% 動態調節的虛線路徑
    Action -. "調節幀率" .-> YOLO

    subgraph L2 ["L2 去重層 · Vector Path（背景 CUDA Stream）"]
        direction TB
        ROIAlign["torchvision roi_align<br>GPU 裁切 ~0.21 ms"]
        NextStep(("↓"))
        ROIAlign --> NextStep
    end

    Action -. "卸載至 CPU" .-> L2

    %% 分支路徑
    ByteTrack -- "位移物件<br>ROI 座標" --> ROIAlign
    ByteTrack -- "無變化物件<br>忽略" --> Skip[(略過)]
    
    class Skip db;
````

*   **媒體接入 (Ingestion) - GStreamer & MediaMTX**：負責接收 RTSP/WebRTC 影像串流。底層使用 C++ 與 `nvh264dec` 進行硬體解碼，將影像幀直接放入 GPU 記憶體池。
*   **L1 感知層 (Perception) - YOLO26 & TensorRT**：主 CUDA Stream。直接讀取 GPU 內的影像指標，使用 Native TensorRT 執行極速、無 NMS 的物件偵測，並利用 GPU ByteTrack 進行軌跡追蹤。
*   **L2 去重層 (Deduplication) - SigLIP 2 & PyTorch**：背景 CUDA Stream。當物體產生位移時，觸發 `torchvision` 進行微秒級的 ROI 裁切，並透過 TensorRT 版本的 SigLIP 2 提取特徵，計算餘弦相似度以過濾語義漂移 (Semantic Drift)。
*   **L3 緩衝層 (Streaming) - Redis Streams & Asyncio**：為防止後端資料庫寫入拖慢感知主循環，所有通過 L2 過濾的特徵向量與事件，都會被非同步地推送到 Redis 記憶體佇列中進行微批次 (Micro-batching) 聚合。
*   **L4 儲存層 (Vector DB) - ChromaDB**：作為系統的長效記憶。從 Redis 接收批次資料後，將語義向量、時間戳記與空間座標寫入 ChromaDB，利用 HNSW 演算法建立多維度索引。
*   **L5 應用層 (Retrieval API) - FastAPI**：提供輕量級的 RESTful API，允許使用者利用自然語言結合時空條件（例如：「尋找昨天下午出現在門口的紅色車子」）進行混合搜尋。
*   **L6 認知與資源層 (Cognition) - Python `psutil`/`pynvml`**：高階決策層。動態監控 VRAM (如確保 4GB 環境下的 1.5GB 上限) 與 CPU 使用率，在資源緊繃時自動調節處理幀率或卸載模型。

### 核心資料流向 (Data Flow Interaction)
1. **硬體解碼**：MediaMTX 接收串流，GStreamer 解碼出 NV12 格式並駐留於 GPU 5-Buffer Pool。
2. **零拷貝感知**：指標透過 PyBind11 傳遞，PyTorch 封裝為 Tensor，YOLO26 在 GPU 上輸出 Bounding Boxes。
3. **分流處理**：追蹤結果不變的物件被忽略；出現位移的物件被裁切並送入 SigLIP 2 提取高維特徵。
4. **過濾與緩衝**：與 GPU Hot Cache 比對發生「漂移」的特徵，被推入 Redis 佇列。
5. **持久化與檢索**：背景 Orchestrator 讀取 Redis，將向量持久化至 ChromaDB，供前端 FastAPI 查詢。

## 3. 技術設計理由 (Design Rationale)

### 為什麼要「解耦」？
解耦讓系統能維持極高的處理幀率 (**120+ FPS**)，而不受語義提取複雜度的影響。沉重的嵌入運算被移至非同步的 CUDA Stream 中，僅由特定的視覺事件觸發。

### 為什麼要「零拷貝 (Zero-Copy)」？
CPU 與 GPU 之間的資料搬運是邊緣 AI 的頭號瓶頸。Saccade 實作了嚴格的零拷貝管線：從硬體解碼到裁切與推理，影像資料全程駐留於 GPU VRAM 中，為系統省下了 85% 以上的 PCIe 頻寬與 CPU 負載。

### 為什麼要「語義漂移 (Semantic Drift)」？
儲存每一次偵測結果是冗餘的。我們使用語義漂移管理器（Cosine Similarity < 0.95）來過濾掉視覺上高度相似的影格，確保向量資料庫僅儲存具備唯一性與高價值的視覺記憶。

## 4. 效能評估 (Performance Evaluation)

*測試環境：NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB), 1080p @ 30fps RTSP 輸入。*

| 項目 | 測試結果 | 工程意義與影響 |
| :--- | :--- | :--- |
| **端到端延遲** | **8.31 ms** | 保證極速的即時反應能力 |
| **管線吞吐量** | **120.2 FPS** | 可處理高解析度、高幀率的監控串流 |
| **顯存 (VRAM) 佔用** | **1.42 GB** | 可流暢運行於 4GB 等級的邊緣設備 |
| **儲存空間節省** | **> 90%** | 大幅降低向量資料庫的寫入負擔 |

## 5. 限制與未來展望 (Limitations & Future Work)

*   **目前限制**：
    *   效能高度依賴針對特定硬體優化的 TensorRT 引擎。
    *   目前僅支援單攝影機輸入，尚未實作多攝影機協同調度。
*   **未來方向**：
    *   實作 **時序性推理 (Temporal Reasoning)**，從偵測靜態物件進化到識別「跌倒」、「奔跑」等動作行為。
    *   分散式向量查詢，支援跨攝影機的目標重識別 (Re-ID)。

## 6. 入門指南 (Getting Started)

```bash
# 1. 啟動環境
docker-compose up -d --build
docker-compose exec saccade bash

# 2. 編譯優化引擎 (TensorRT)
uv run python scripts/build_yolo_engine.py --onnx models/yolo/yolo26n.onnx --engine models/yolo/yolo26n_native.engine
uv run python scripts/build_siglip_engine.py

# 3. 啟動 Saccade
./scripts/saccade up
```

---
*本專題展示了一種仿生設計方法，旨在彌合邊緣感知與語義推理之間的鴻溝。*
