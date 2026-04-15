# YOLO-LLM 開發指南

本專案旨在建構一個高效、低延遲的視覺推理系統，結合即時目標偵測與深度認知推理，並極大化 NVIDIA GPU 的運算效率。

## 1. 系統架構核心 (The 5 Pillars)

本系統採用純視覺與向量檢索管線，以確保在極低 VRAM (1.5GB) 佔用下，達到毫秒級即時回應與精準檢索。

1. **純視覺向量管線 (Vision-Vector Pipeline)**
   - **感知層 (Perception)：** 使用 YOLO11 與 TensorRT 引擎，負責即時物件追蹤與偵測。
   - **特徵提取 (Extraction)：** 運用 Zero-Copy Cropper 與 **Jina-CLIP-v2** TRT 引擎，將物件裁切為 512x512 格式並提取 1024 維高品質特徵向量。

2. **語義漂移去重 (Semantic Drift Handling)**
   - 透過 GPU 內的 `Cosine Similarity`，將新特徵與快取進行比對。避免連續幀重複存儲，僅當物體姿態或特徵產生「漂移 (Drift)」時寫入。

3. **防禦性熱切換 (Systemd + NVML Hot Swapping)**
   - 透過 Systemd `--user` 管理進程，配合 MediaMTX 處理串流緩衝，確保在模組切換或重啟時，視訊串流不中斷。

4. **底層算力優化 (Pure NVIDIA Native Zero-Copy)**
   - 實現 `MediaMTX -> NVDEC -> CUDA Tensor -> TensorRT (YOLO & SigLIP)` 的 100% 零拷貝數據路徑，全程無 CPU 記憶體搬運。

5. **環境與狀態管理 (Unified Environment)**
   - **Docker:** 使用 NVIDIA 官方 TensorRT 基礎鏡像，鎖定 CUDA、GStreamer 等系統級依賴。
   - **uv：** 在容器內進行 Python 環境隔離，實現秒級依賴安裝與重現性。

## 2. 開發環境與建構 (DevOps)

本專案依賴 Docker 與 uv 進行宣告式環境管理。

- **初始化環境：**
  ```bash
  # 啟動開發容器
  docker-compose up -d
  # 進入容器
  docker-compose exec saccade zsh
  # 安裝 Python 依賴
  uv sync
  ```
  
- **編譯 TensorRT 模型：**
  ```bash
  # 首次啟動前需將 ONNX 轉為 TRT Engine
  uv run python scripts/build_engine.py
  ```

## 3. 媒體與串流管理 (Media Gateway)

使用 **MediaMTX** 作為核心媒介閘道，並以 GStreamer `nvh264dec` 作為解碼前端。

- **啟動所有服務：**
  ```bash
  ./scripts/saccade up
  ```

## 4. 關鍵技術堆疊

| 層級 | 技術 |
| :--- | :--- |
| **算法** | YOLO26 (TRT), SigLIP2 (TRT), `torchvision.ops.roi_align` |
| **媒體** | MediaMTX, FFmpeg (NVDEC), GStreamer (`appsink`) |
| **計算與資源** | TensorRT, CUDA Streams, Pynvml |
| **環境維運** | Nix Flakes, uv (Rust-based) |

## 5. 開發工作流 (Development Workflow)

本專案嚴格執行 **「文檔先行 (Documentation-First)」** 的開發策略。在進行任何代碼變更或核心邏輯重構前，開發者必須遵守以下流程：

1. **實作提案 (Planning)**: 
   - 必須先在 `docs/decisions/` (若涉及重大架構決策) 或 `docs/progress/` (若為功能擴展) 中提交目標實作規劃。
   - 規劃中應包含：目標 (Objective)、技術路徑 (Technical Path)、對 L1-L5 架構的影響、以及 VRAM 資源預估。
2. **審核與確認 (Review)**: 
   - 確保提案符合 **Zero-Copy** 原則與 **Pillar 5** 核心理念。
3. **實作與驗證 (Execution)**: 
   - 僅在規劃文檔提交並確認後，才開始編寫代碼。
   - 實作完成後，應同步更新相關進度文檔。

## 6. 文檔結構指南 (Documentation Structure)

為了維護 **L1-L5 架構** 的一致性，開發者應根據變更性質，將文檔提交至對應目錄：

| 目錄 | 功能說明 | 適用情境 |
| :--- | :--- | :--- |
| **`docs/decisions/`** | **架構決策紀錄 (ADR)** | 當涉及技術選型變更 (如更換資料庫、推論後端) 或重大架構調整時使用。 |
| **`docs/progress/`** | **模組實作規劃與進度** | 當進行新功能開發或模組功能擴展時使用。需包含目標實作路徑與 VRAM 預估。 |
| **`docs/api_spec.md`** | **API 與事件規範** | 當修改 Redis 事件結構、ChromaDB Schema 或對外接口時更新。 |
| **`docs/runbooks/`** | **運作與維護手冊** | 當新增模組後的故障排除流程 (Troubleshooting) 或日常維運指令。 |
| **`docs/benchmarks/`** | **效能基準測試紀錄** | 提交效能優化後的數據對比 (Latency, Throughput, VRAM)。 |
| **`docs/architecture.md`** | **系統架構說明書** | 僅在全系統分層 (L1-L5) 定義產生變化時更新。 |

## 7. 開發約定 (Coding Standards) 與 Git 規範

### 7.1 程式碼風格 (Coding Style)
- **Python**: 遵循 PEP 8，使用 `ruff` 進行 Linter 與 Formatting，並強制 `mypy` 嚴格型別檢查 (`strict = true`)。
- **C++/CUDA**: 遵循 C++17 標準，使用 `PascalCase` 命名類別，`camelCase` 命名函式，`snake_case` 命名變數與私有成員 (`_` 結尾)。

### 7.2 Git 提交與分支規範 (Git Conventions)
本專案嚴格採用 **[Conventional Commits](https://www.conventionalcommits.org/zh-hant/v1.0.0/)** 規範，以利自動化生成 Changelog 並追蹤版本歷史。

#### 提交格式 (Commit Format)
```text
<type>(<scope>): <subject>

<body>
```

#### 常見的 Type 定義
- `feat`: 新增功能 (Feature)。
- `fix`: 錯誤修復 (Bug fix)。
- `refactor`: 重構 (既不是新增功能也不是修復 bug 的程式碼變動，例如：架構遷移、重寫邏輯)。
- `docs`: 文檔變更 (Documentation only changes，包含 ADR 或進度更新)。
- `perf`: 改善效能的程式碼變更 (Performance improvements，如 Zero-Copy 最佳化)。
- `test`: 增加或修正測試 (Adding missing tests or correcting existing tests)。
- `chore`: 構建過程或輔助工具的變更 (例如更新依賴、修改 Dockerfile 或 CMakeLists)。

#### 提交原則 (Commit Rules)
1. **原子性提交 (Atomic Commits)**: 每個 Commit 應該只處理一個邏輯單元的變更，嚴禁將不相關的修改混在一起。
2. **清晰的 Scope**: 盡量在括號中註明影響的模組範圍（例如 `perception`, `tracking`, `docs`, `infra`）。
3. **詳細的 Body**: 對於重大變更或重構，必須在 Body 中說明「為什麼 (Why) 這樣做」以及「達成了什麼 (What)」。
4. **提交前驗證 (Pre-commit Validation)**: 在提交任何 Python 程式碼前，**必須**執行型別檢查 (`uv run mypy .`) 並確保完全通過。嚴禁提交帶有未解決型別錯誤的程式碼。

#### 範例 (Example)
```text
refactor(tracking): migrate SmartTracker to C++/CUDA

- Implement SmartTracker in C++/CUDA to avoid Python CPU/GPU sync bottleneck
- Expose SmartTracker via pybind11
- Update tests and benchmark scripts to use the new native module
```
