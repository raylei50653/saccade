# Saccade 系統架構說明書

Saccade 是一個模擬人類視覺系統的「雙軌視訊感知系統」。本文件詳述系統架構設計與資料夾結構的功能定義。

---

## 1. 核心設計理念：純視覺向量管線 (Vision-Vector Pipeline)

系統為了解決邊緣設備 (12GB VRAM) 資源受限的問題，捨棄了龐大緩慢的 VLM，轉而採用全 GPU、微秒級延遲的純視覺管線：

- **感知層 (Perception):** 使用 YOLO11 (TensorRT) 進行 140+ FPS 的即時偵測與追蹤。
- **特徵提取層 (Extraction):** 透過 `torchvision.ops.roi_align` 在 VRAM 內直接裁切，並餵入 **Jina-CLIP-v2** (TensorRT) 提取高維語義特徵。
- **時空記憶 (Memory):** 將語義特徵與結構化 Metadata (物件類型、時間戳) 存入 ChromaDB，實現混合檢索。

這條管線透過 **Semantic Drift (語義漂移)** 機制進行優化，確保只有「新物件」或「產生形變/位移」的物件兩者才會被提取與寫入，大幅節省算力與儲存空間。

---

## 2. 資料夾功能指南

### 🏗️ 核心組件 (Core Components)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`perception/`** | **視覺核心**。包含 YOLO 偵測器 (`detector.py`)、零拷貝裁切 (`cropper.py`)、**Jina-CLIP-v2** 特徵提取 (`feature_extractor.py`)、追蹤過濾 (`tracker.py`) 與語義去重 (`drift_handler.py`)。 |
| **`pipeline/`** | **系統調度層**。包含協調管線運作的調度器 (`orchestrator.py`) 與全系統健康檢查機制 (`health.py`)。 |

### 🛠️ 基礎設施與媒體 (Infrastructure & Media)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`media/`** | **串流處理**。封裝 MediaMTX 用戶端 (`mediamtx_client.py`)，處理 GStreamer Zero-Copy 硬體解碼。 |
| **`infra/`** | **維運配置**。存放 Systemd 服務單元檔與 MediaMTX 的核心設定檔 (`mediamtx.yml`)。 |
| **`storage/`** | **狀態與記憶**。實作 ChromaDB 向量記憶庫 (`chroma_store.py`) 與 Redis 狀態管理 (`redis_cache.py`)。 |

### ⚙️ 設定與模型 (Configs & Models)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`models/`** | **模型權重**。包含 YOLO (.pt, .engine) 與 **Jina-CLIP-v2** 的 TRT Engine (放置於 `embedding/`)。注意：二進制檔受 `.gitignore` 保護。 |

### 🧪 開發輔助 (Dev Tools)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`scripts/`** | **維運腳本**。包含一鍵啟動 CLI (`saccade`)、ONNX 導出 (`export_jina_clip.py`) 與 TRT 編譯 (`build_engine.py`) 腳本。 |
| **`tests/`** | **驗證套件**。包含單元測試與針對 Zero-Copy、VRAM 穩定性與 DB 品質的效能基準測試（Benchmarks）。 |

---

## 3. 數據流向 (Data Flow)

1.  **影像源**: 攝影機透過 RTSP 傳輸至 **MediaMTX**。
2.  **解碼**: `media/` 透過 GStreamer (nvh264dec) 將資料留在 GPU 轉為 CUDA Tensor。
3.  **感知**: YOLO 讀取 Tensor 產出 BBox。
4.  **裁切與提取**: `perception/` 在背景 CUDA Stream 裁切新物件並提取 **Jina-CLIP-v2** 特徵。
5.  **去重**: 透過 Cosine Similarity 過濾冗餘特徵。
6.  **Memory**: 寫入 `storage/` (ChromaDB) 供未來關聯檢索。

---

## 4. 開發約定

- **Zero-Copy 原則**: 所有的影像處理流程 (解碼 -> 偵測 -> 裁切 -> 特徵提取) 必須維持在 GPU VRAM 中，禁止呼叫會引發 `.cpu()` 或 `numpy()` 轉換的操作。
- **異步與非阻塞**: 特徵提取 (`TRTFeatureExtractor`) 必須在獨立的 `torch.cuda.Stream` 中執行，嚴禁阻塞 YOLO 的主追蹤迴圈。
- **資源節約**: 透過 `SemanticDriftHandler` 過濾重複狀態，確保資料庫的高品質。
