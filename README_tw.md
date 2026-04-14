# Saccade (眼動系統)

持續的視覺感知與向量資料庫記憶 — 每一次偵測與語義特徵都帶有標籤、時間戳記且可供毫秒級檢索。

---

> **🌍 [English Version](README.md)**

---

## 概述

Saccade 是一個設計用於受限 GPU 硬體（12GB VRAM）的高效能視覺感知系統。它結合了實時感知與事件驅動的語義提取，將偵測結果與高維特徵向量儲存為帶有時間標籤的記憶，實現「可語義搜尋」的視訊監控。

就像人類眼球的跳動（saccadic motion） — 先快速掃描（Perception），再集中理解（Extraction） — Saccade 將感知與語義提取拆分為兩個非同步軌道，透過 CUDA Streams 達成並行運作且互不阻塞。

## 系統架構

**感知快路徑 (Fast track)** 持續以 140+ FPS 運行，使用 YOLO26 與 TensorRT 引擎評估每一影格。透過 GStreamer (NVDEC) 與 C++ 核心實現零拷貝硬體解碼，數據全程保留在 GPU VRAM 中。

**語義提取慢路徑 (Slow track)** 運作於獨立的 CUDA Stream。當偵測到新物件或物件行為發生重大改變（語義漂移）時，使用 `roi_align` 在顯存中直接裁切，並透過 **SigLIP 2** (TensorRT) 提取特徵向量存入 ChromaDB。

## 核心設計決策

**純視覺向量管線 (Pure Vision-Vector)** — 為了極大化吞吐量，我們採用「YOLO26 + SigLIP 2」架構。VRAM 佔用降至 **~1.5GB**，效能提升數倍。

**語義漂移處理 (Semantic Drift Handling)** — 為防止資料庫膨脹，提取的特徵會與 GPU 內的熱快取進行餘弦相似度比對。僅當語義發生顯著變化時才會寫入資料庫。

**零拷貝 GPU 路徑 (Zero-copy GPU path)** — 影像路徑為 `NVDEC → NVMM → CUDA Tensor → TensorRT`，完全不經過 CPU 記憶體搬運，最大限度減少 PCIe 頻寬損耗。

**時空向量記憶 (Vector-indexed memory)** — 結合 Redis 與 ChromaDB。Redis 負責實時物件軌跡追蹤；ChromaDB 負責長期的語義搜尋。支援混合檢索（語義 + 元數據 + 時間過濾）。

## 技術棧

| 層級 | 技術 |
| :--- | :--- |
| 偵測 (Detection) | YOLO26 (TensorRT Engine, C++ Fast Path), GPU 實時物件追蹤 |
| 提取 (Extraction) | SigLIP2 (TensorRT Engine) |
| 媒體 (Media) | MediaMTX, GStreamer (nvh264dec), OpenCV (CUDA) |
| 記憶 (Memory) | ChromaDB (向量), Redis (時空快取/事件) |
| 介面 (API) | FastAPI (時空檢索與語義搜尋) |
| 環境 (Env) | Docker, uv |

## 入門指南

**需求：** NVIDIA GPU (12GB+ VRAM), 安裝 NVIDIA Container Toolkit 的 Docker, CUDA 12.x

```bash
# 啟動並進入開發環境 (鎖定 CUDA, GStreamer, 系統依賴)
docker-compose up -d --build
docker-compose exec perception bash

# 安裝 Python 依賴
uv sync

# 編譯 TensorRT 模型 (首次啟動需執行)
uv run python scripts/build_engine.py

# 啟動全系統服務 (感知 + 調度 + API)
./scripts/saccade up
```

## 檢索 API (Retrieval API)

系統提供 FastAPI 介面供外部查詢：
- **活躍物件**: `GET /objects` (獲取目前畫面上所有目標 ID)
- **歷史軌跡**: `GET /objects/{id}` (獲取特定物件的出現時間與移動路徑)
- **語義搜尋**: `POST /search` (輸入文字搜尋相關的歷史影像記憶)

## 專案結構

詳細的資料夾功能對照請參見 [**docs/architecture.md**](docs/architecture.md)。

```
saccade/
├── perception/    # 視覺核心：YOLO TRT, 零拷貝裁切, Jina-CLIP TRT, 漂移偵測
├── pipeline/      # 調度層：事件路由、結構化索引與系統健康監控
├── media/         # 串流層：MediaMTX 客戶端 (GStreamer Zero-Copy)
├── storage/       # 存儲層：ChromaDB 向量庫與 Redis 時空快取
├── api/           # 介面層：FastAPI 時空檢索伺服器
├── infra/         # 維運層：Systemd --user 服務單元配置
├── scripts/       # 工具集：服務管理 CLI、模型編譯、VRAM 監控
└── tests/         # 品質保證：單元測試與效能基準 (Benchmarks)
```

## 文件連結

- [`docs/architecture.md`](docs/architecture.md) — 系統架構深度設計與數據流
- [`docs/progress/`](docs/progress/) — 各模組開發狀態追蹤
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — 完整開發與編譯指南
