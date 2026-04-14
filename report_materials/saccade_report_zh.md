# Saccade：針對邊緣 AI 的高效率雙軌視覺感知系統

## 核心論點 (Core Arguments)

本專題報告旨在解決邊緣運算設備（如 4GB VRAM 限制）在進行連續且深度的視覺感知時所面臨的資源瓶頸。我們提出以下四個核心論點：

1. **邊緣運算的資源困境**：傳統端到端的大型視覺語言模型 (VLM) 佔用大量 VRAM 且推理延遲高，無法在邊緣硬體上實現高幀率的即時連續感知；而輕量級的物件偵測模型又缺乏深度的語義理解能力。
2. **仿生雙軌感知架構 (Saccadic Motion)**：借鑒人類視覺的「掃視」與「注視」機制，系統將**快速的空間位置感知**（YOLO26 負責的 Fast Path）與**深度的語義特徵提取**（SigLIP 2 負責的 Vector Path）徹底解耦，透過非同步並行處理打破效能瓶頸。
3. **極致的 Zero-Copy (零拷貝) 硬體管線**：影像從 RTSP 硬體解碼、ROI 裁切到向量特徵提取，全程將資料保留在 GPU VRAM 內，嚴格禁止 CPU 與 GPU 之間的資料拷貝，從而消除 PCIe 頻寬瓶頸，實現極低延遲。
4. **語義漂移 (Semantic Drift) 記憶過濾機制**：透過計算特徵向量的餘弦相似度 (Cosine Similarity)，系統僅在物體發生明顯視覺與語義變化時，才將資料寫入 ChromaDB 向量資料庫，大幅減少冗餘運算並防止資料庫膨脹。

---

## 1. 系統架構與技術堆疊 (System Architecture & Tech Stack)

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

## 2. 核心技術實作與細節 (Technical Implementations)

### 2.1 零拷貝記憶體與 Native TensorRT (L1 極限最佳化)
根據架構決策紀錄 (ADR-006)，為了消除 Python 封裝庫帶來的物件建立開銷與 GIL 阻塞，我們實作了 `TRTYoloDetector`。傳輸過程採用了 **5-Buffer GPU Pool**，影像幀解碼後直接作為 CUDA Tensor。

**實作細節：** 
系統避免了傳統的 `numpy()` 轉換，而是透過記憶體指標直接映射給 PyTorch 模型使用，概念如下：
```python
# Zero-Copy 綁定範例：直接映射 GPU 指標，無 CPU 介入
tensor = torch.as_tensor(cuda_data_ptr, device='cuda')
```
這項技術將影像傳輸延遲壓縮至不可思議的 **0.0004 ms** (幾乎為零)，徹底消除 PCIe 頻寬瓶頸。

### 2.2 語義漂移過濾機制 (Semantic Drift)
系統捨棄了緩慢的端到端 VLM。當 L1 偵測到物件後，系統使用 `torchvision.ops.roi_align` 在 GPU 內進行微秒級的裁切 (約耗時 **0.21 ms**)。隨後，由 SigLIP 2 模型提取 768 維的語義特徵。

**過濾演算法設計：**
為了防止影片連續畫面造成 ChromaDB 暴增，我們在 GPU 上建立了一個熱快取 (Hot Cache) 進行 Cosine Similarity (餘弦相似度) 比對。
```math
\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
```
只有當新特徵與舊特徵的相似度低於特定閥值（例如 `< 0.9`），系統才會判定發生「語義漂移」，並觸發後續寫入 ChromaDB 的動作。這使得過濾吞吐量高達每秒 **8,600+ 個物件**，且單次比對延遲小於 **1 ms**。

### 2.3 徹底的 C++ 遷移與動態連結 (ADR-007)
為確保模組化與極致效能，核心的感知管線、追蹤與媒體模組均遷移至 C++17。各元件編譯為獨立的動態連結庫 (`.so`)，並透過 `pybind11` 提供 Python 介面。此架構嚴格執行 RAII 資源管理，避免了 Python 垃圾回收帶來的延遲毛刺。

## 3. 實驗與效能數據 (Performance & Evaluation)

透過最新的 Benchmark 測試，在邊緣運算設備上，Saccade 展現了突破性的效能：

*   **端到端延遲 (End-to-End Latency)**： 
    L1 (YOLO26) + L2 (SigLIP 2 特徵提取) 完整感知管線的平均延遲為 **7.80 ms**，P99 尾部延遲為 **11.21 ms**。
    *   YOLO 推理：~3.53 ms
    *   SigLIP 2 特徵提取：~1.34 ms
*   **超高吞吐量 (Throughput)**：
    連續感知主路徑 (Fast Path) 可穩定維持在 **128 FPS** 以上。
*   **儲存層效率**：
    寫入 ChromaDB 的延遲為 ~194 ms/record，查詢延遲約為 181 ms，因已透過 Redis Streams 進行非同步解耦，故不影響感知層的 128 FPS 吞吐量。

---

## 4. 當前實作亮點 (Implementation Highlights)

在專案目前的發想與實作階段，我們已完成多項挑戰硬體極限的工程實作，展現了對底層記憶體與系統效能的強大控制力：

1. **C++ 影音管線與跨語言記憶體共享 (PyBind11)**：
   我們實作了基於 C++ 與 GStreamer (`nvh264dec`) 的媒體擷取模組 (`GstClient`)。有別於傳統將影像轉換為 Python 物件的做法，C++ 端分配了固定大小的 **5-Buffer GPU Pool**，並透過 `PyBind11` 將顯存指標 (`cuda_ptr`) 以整數形式暴露給 Python。這不僅避免了頻繁的記憶體分配 (Allocation)，也解決了跨語言傳輸的開銷。

2. **NV12 原生格式的硬體 Zero-Copy 支援**：
   在 Python 的擷取層 (`MediaMTXClient`) 中，我們成功實作了對 NV12 (1.5 bytes per pixel) 等硬體原生解碼格式的直接支援。利用 `__cuda_array_interface__` 建立自定義的 `CudaPointerHolder`，使得 PyTorch 能夠將記憶體指標直接視為張量 (Tensor)，省去了昂貴的色彩空間轉換 (Color Space Conversion) 步驟。

3. **動態 TensorRT 引擎自適應編譯**：
   為了應對模型架構的快速迭代，我們設計了自動化的 TensorRT 引擎建置腳本，其能夠動態解析 ONNX 檔案的輸入節點名稱與維度（如 YOLO11 或 YOLO26 的 `images`），並自動設置動態張量形狀 (Dynamic Shapes) 來優化編譯，大幅簡化了未來替換新模型的負擔。

---

## 5. 總結與未來展望

Saccade 系統成功證明了在資源受限的邊緣設備上，透過仿生雙軌架構與極致的硬體管線最佳化（Zero-Copy、Native TensorRT 與 C++ 記憶體池），可以實現具備深層語義記憶的連續視覺感知。未來的發展將聚焦於 L6 認知層的擴展，包括多攝影機的協同調度與分散式的向量查詢。