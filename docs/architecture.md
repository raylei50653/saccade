# Saccade 系統架構說明書

Saccade 是一個模擬人類視覺系統的「雙軌視訊感知系統」。本文件詳述系統架構設計與資料夾結構的功能定義。

---

## 1. 核心設計理念：雙軌異步 (Bifurcated Pipeline)

系統將運算資源拆分為兩條平行路徑，以解決即時性與深度理解之間的矛盾：

- **感知快路徑 (Fast Path - Perception):** 負責「看見」。使用 YOLO 進行每秒 30 幀以上的即時偵測，並計算資訊熵（Entropy）。
- **認知慢路徑 (Slow Path - Cognition):** 負責「理解」。僅在快路徑發現「高價值事件」時，由 VLM (如 Qwen2-VL) 進行深度分析。

這兩條路徑透過 **Redis 事件佇列** 進行解耦，互不阻塞。

---

## 2. 資料夾功能指南

### 🏗️ 核心組件 (Core Components)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`perception/`** | **快路徑核心**。包含 YOLO 偵測器 (`detector.py`)、追蹤演算法 (`tracker.py`) 以及評估影格價值的資訊熵觸發器 (`entropy.py`)。 |
| **`cognition/`** | **慢路徑核心**。負責 VLM 推理引擎 (`llm_engine.py`)、VRAM 資源動態調配 (`resource_manager.py`) 以及關鍵幀選取策略。 |
| **`pipeline/`** | **系統調度層**。包含協調雙軌運作的調度器 (`orchestrator.py`) 與全系統健康檢查機制 (`health.py`)。 |

### 🛠️ 基礎設施與媒體 (Infrastructure & Media)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`media/`** | **串流處理**。封裝 MediaMTX 用戶端 (`mediamtx_client.py`)，處理 RTSP 抓幀、重連機制與 FFmpeg 推流工具。 |
| **`infra/`** | **維運配置**。存放 Systemd 服務單元檔與 MediaMTX 的核心設定檔 (`mediamtx.yml`)。 |
| **`storage/`** | **狀態與記憶**。實作 ChromaDB 向量記憶庫與 Redis 快取狀態管理，確保系統「無狀態化」。 |

### ⚙️ 設定與模型 (Configs & Models)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`configs/`** | **靜態配置**。包含 YOLO 門檻值、VRAM 分配 Profile (`llm_profiles.yaml`) 與串流路由設定。 |
| **`models/`** | **模型權重**。依類別存放 `.pt`, `.gguf`, `.engine` 檔案。注意：此處檔案受 `.gitignore` 保護，不應上傳至 Git。 |

### 🧪 開發輔助 (Dev Tools)

| 資料夾 | 功能說明 |
| :--- | :--- |
| **`scripts/`** | **維運腳本**。包含一鍵啟動 CLI (`saccade`)、VRAM 監控與環境同步腳本。 |
| **`tests/`** | **驗證套件**。包含單元測試、整合測試以及針對 VLM 延遲與 Zero-Copy 效能的基準測試（Benchmarks）。 |
| **`docs/`** | **文件庫**。包含架構決策紀錄 (ADRs)、各模組開發進度與運作手冊 (Runbooks)。 |

---

## 3. 數據流向 (Data Flow)

1.  **影像源**: 攝影機透過 RTSP 傳輸至 **MediaMTX**。
2.  **Perception**: `perception/` 連續抓幀進行 YOLO 偵測，若發現異常（資訊熵高），則向 **Redis** 發送事件。
3.  **Orchestrator**: `pipeline/` 監聽到 Redis 事件，通知 `cognition/`。
4.  **Cognition**: 根據當前 VRAM 狀態載入模型，從 `media/` 抓取關鍵幀進行分析。
5.  **Memory**: 分析結果標註時間戳後存入 `storage/` (ChromaDB)。

---

## 4. 開發約定

- **無狀態原則**: 所有的暫存狀態與長效記憶必須存在 Redis/ChromaDB。任何進程（Process）重啟後應能立即恢復工作。
- **異步優先**: 所有的 I/O (網路呼叫、模型推理、Redis 通訊) 必須使用 `async/await`。
- **資源隔離**: 確保 Perception 模組擁有保留的 VRAM (透過 `resource_manager.py`)，防止 LLM 推理造成感知中斷。
