# Saccade (眼動系統)

持續的視覺感知與向量資料庫記憶 — 每一次偵測都帶有標籤、時間戳記且可供查詢。

---

## 概述

Saccade 是一個設計用於受限 GPU 硬體（12GB VRAM）的雙軌視覺推理系統。它結合了持續的實時感知與事件驅動的認知分析，將每一次偵測儲存為帶有向量索引、時間標籤的記憶，並可透過語義描述進行查詢。

就像人類眼球的跳動（saccadic motion） — 先快速掃描，再集中理解 — Saccade 將感知與認知拆分為兩個非同步軌道，並行運作且互不阻塞。

## 系統架構

![架構圖](./assets/images/architecture_tw.svg)

**快路徑 (Fast track)** 持續運行，評估每一影格的資訊熵。只有高價值的事件會觸發慢路徑 — 讓 LLM 完全遠離熱路徑 (hot path)。

**慢路徑 (Slow track)** 根據需求從 MediaMTX 抓取關鍵影格，執行深度的視覺語言分析，並將結構化的偵測結果寫入 ChromaDB，帶有語義標籤與時間戳記。

## 核心設計決策

**分叉管線 (Bifurcated pipeline)** — 感知與認知作為獨立服務運行。Redis 佇列是它們之間唯一的耦合點。任何一個服務都可以重啟而不中斷影片串流。

**動態算力調度 (Dynamic compute provisioning)** — llama.cpp 的 `-c` (上下文) 與 `-ngl` (GPU offload 層數) 會在執行時根據可用 VRAM 進行調優，並在需要時平滑回退至 64GB 系統記憶體。

**零拷貝 GPU 路徑 (Zero-copy GPU path)** — 影片影格路徑為 `NVDEC → NVMM → CUDA Tensor`，完全不觸碰 CPU 記憶體，最大限度地減少 PCIe 頻寬使用。

**無狀態推理 (Stateless inference)** — 推理單元不持有持久狀態。所有記憶都存在於 ChromaDB（向量儲存）和 Redis（事件佇列）中，因此服務可以透過 Systemd 安全地進行熱切換而不會遺失數據。

**向量索引記憶 (Vector-indexed memory)** — 每一次偵測都會被嵌入 (embedded)、標記並儲存 Unix 時間戳記。查詢範例：*"顯示 18:00 後在入口附近偵測到的所有人"*, *"找出車輛停放超過 30 秒的影格"*。

## 技術棧

| 層級 | 技術 |
| :--- | :--- |
| 偵測 (Detection) | YOLO26, CV 追蹤 |
| 認知 (Cognition) | Qwen-3.5 (GGUF), CLIP, SigLIP, llama.cpp |
| 媒體 (Media) | MediaMTX, FFmpeg (NVDEC/NVENC), GStreamer |
| 記憶 (Memory) | ChromaDB (向量), Redis (快取/佇列) |
| 計算 (Compute) | TensorRT, PagedAttention, NVML |
| 環境 (Env) | Nix Flakes, uv |

## 入門指南

**需求：** NVIDIA GPU (12GB+ VRAM), 已啟用 flakes 的 Nix, CUDA 12.x

```bash
# 進入開發環境 (鎖定 CUDA, GStreamer, 系統依賴)
nix develop

# 安裝 Python 依賴
uv sync

# 複製並配置環境變數
cp .env.example .env

# 啟動管線
python main.py
```

**串流壓力測試：**
```bash
ffmpeg -re -i source.mp4 -c copy -f rtsp rtsp://localhost:8554/stream
```

## 專案結構

```
saccade/
├── perception/       # 快路徑 — YOLO26, CV 追蹤, 資訊熵評估
├── cognition/        # 慢路徑 — LLM/VLM 推理, 關鍵影格分析
├── pipeline/         # 調度器 — 軌道間的事件路由
├── media/            # MediaMTX 用戶端, FFmpeg 工具集
├── storage/          # ChromaDB 向量儲存, Redis 快取
├── infra/            # Systemd 單元, MediaMTX 配置
├── configs/          # 模型配置檔, YOLO 閾值
├── models/           # 模型權重 (Git 不追蹤 — 參見 models/README.md)
├── scripts/          # 開發工具 — 串流測試, VRAM 監控
├── tests/            # 單元與整合測試, 效能基準
└── docs/             # 架構決策, 模組進度, 維運手冊
```

## 開發規範

- **效能優先** — 所有新的 Python 運算子若存在 CUDA/TensorRT 實作，應優先選用。
- **無狀態推理** — 推理單元必須隨時可重啟；狀態屬於 ChromaDB 或 Redis。
- **型別安全** — 透過 CI 中的 mypy 強制執行嚴格的型別標註。

## 文件連結

- [`docs/architecture.md`](docs/architecture.md) — 系統設計與 ADRs
- [`docs/progress/`](docs/progress/) — 各模組開發狀態
- [`docs/runbooks/`](docs/runbooks/) — 維運程序 (熱切換, 串流復原, OOM 處理)
- [`docs/benchmarks/`](docs/benchmarks/) — 延遲、VRAM 與吞吐量量測
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — 完整開發指南
