# Saccade 核心設計與維護文檔庫

> **文檔維護原則：一個文檔只回答一個問題。透過相對索引串聯，不靠無限合併。**

---

## 📂 核心目錄架構

Saccade 的文檔庫採用「模組化物理結構」配合「全局共享目錄」的組織形式：

* 🧭 **[PIPELINE.md](PIPELINE.md)**：**演算法主線精煉（單檔入口）**。按處理流程 `前處理 → ssm-head → 後處理 → GMC → ID分配 → relink` 走一遍，每階段只留「現行最優 / GO / NO-GO」結論，細節下沉到下列 modules/ 與 registry。**想快速掌握全鏈先讀這份。**
* 📦 **[modules/](modules)**：**核心模組化文檔庫**。物理上按系統功能與 `mot17.py` 評測模組劃分，包含各自的設計、決策 (ADR) 與實驗分析。
* 📊 **[reference/](reference)**：全局評測基準 (benchmarks)、多進程並行評測手冊與跨模組共享的流程規範。亦含 **[NO-GO 全局登記表](reference/no_go_registry.md)**（已結案/已踩雷方向總覽，探索新方向前先查）。
* 📦 **[archive/](archive)**：過時或已完成的歷史參考資料（如 Option D 探索結論）。
* 🔬 **[research/](research)**：全局性的評測與訓練流程（如 evaluator 分析或訓練共享設施）。

---

## 🗺️ 模組導覽地圖 (Module Map)

為了方便進行組件級的「性能與精度分析」，各模組的架構、決策與實驗文檔已被物理歸檔至 `modules/` 下對應的子目錄中：

### 1. 感知與追蹤核心鏈路 (Perception & Tracking - mot17 對齊)

| 模組名稱 | 核心職責 | 主要文檔連結 |
|---|---|---|
| 🔍 **[detection/](modules/detection)** | YOLO偵測、Mamba Head 網絡、NMS 抑制、影像預處理與 Tiling | * [Mamba Head 設計與 PixelShuffle](modules/detection/option-f-mamba-head.md)<br>* [v14-R 訓練規範](modules/detection/mamba-v14r-training-protocol.md)<br>* [CUDA Graph Capture 重大 Bug 分析](modules/detection/research/mamba-cuda-graph-bug.md) |
| 📐 **[geometry/](modules/geometry)** | GMC 全局運動補償、卡爾曼濾波協方差、寬高比限制與幾何優先級 | * [GPU Tracker 深度解析](modules/geometry/tracker_deep_dive.md)<br>* [GMC 與卡爾曼濾波消融實驗](modules/geometry/research/fp_fn_recovery_and_gmc.md) |
| 🌀 **[motion/](modules/motion)** | 軌跡速度/加速度 EMA 平滑、運動一致性檢查 (z-score) 與運動 Fallback | * [Motion 參數配置說明](modules/motion/README.md) |
| 🧬 **[reid/](modules/reid)** | SigLIP 2 特徵提取、Feature Bank 外觀庫更新與去重、裁剪影格邊緣 | * [ReID 與 Feature Bank 架構](modules/reid/architecture.md)<br>* [SigLIP 2 升級決策](decisions/005-yolo26-siglip2-upgrade.md) |
| 🤝 **[semantic/](modules/semantic)** | 外觀相似度門檻（ReID Threshold）、匈牙利演算法匹配權重、外觀重排 | * [Sinkhorn 混合關聯決策](decisions/015-sinkhorn-auction-hybrid-association.md)<br>* [外觀特徵重排與品質過濾](decisions/016-rerank-phase3-reference-quality.md) |
| ⚡ **[trigger/](modules/trigger)** | 非同步 ReID 觸發機制（Async ReID Trigger）、外觀抽取預算觸發 | * [動態外觀觸發機制實驗](modules/trigger/research/dynamic_trigger.md)<br>* [Saccade 心跳閘控觸發](decisions/013-gpubytetracker-saccade-heartbeat.md) |
| 🔄 **[lifecycle/](modules/lifecycle)** | 軌跡生命週期狀態機（Tentative/Confirmed/Lost）判定與 Tracker LRU 釋放 | * [生命週期狀態轉移實驗](modules/lifecycle/research/tentative_confirmed_state.md) |

### 2. 系統外圍與基礎設施 (System Infrastructure)

* 🖥️ **[streaming/ (L3 媒體接入)](modules/streaming)**：RTSP 解碼、DALI 預處理與零拷貝解碼。
  * *參考*：[RTSP 傳輸規範](modules/streaming/runbooks/rtsp_contract.md) \| [DALI GPU 預處理決策](decisions/010-dali-gpu-preprocessing.md)
* 💾 **[storage/ (L4 長期記憶)](modules/storage)**：Redis 微批次寫入、ChromaDB 向量索引與混合檢索。
  * *參考*：[快取儲存架構](modules/storage/architecture.md) \| [API / Schema 規格說明](modules/storage/api_spec.md)
* 🧠 **[cognition/ (L5 認知推理)](modules/cognition)**：本地 Llama 3 邊緣 Agentic RAG、事件分析與視覺二次查詢。
  * *參考*：[認知層架構](modules/cognition/architecture.md) \| [Agentic RAG 決策](decisions/014-agentic-rag-llama-index.md)
* ⚙️ **[resource/ (L6 資源與健康度)](modules/resource)**：VRAM 實時監測、動態三階 Hysteresis 降級機制。
  * *參考*：[階梯降級架構](modules/resource/architecture_degradation.md) \| [VRAM OOM 維運手冊](modules/resource/runbooks/vram_oom.md)

---

## ✍️ 開發者寫作導覽 (我去哪裡寫？)

當您在開發過程中需要編寫或更新文檔時，請遵循此決策路徑：

```
我做了什麼？                     →  去哪個目錄？                →  寫什麼格式/檔案？
─────────────────────────────────────────────────────────────────────────────────────────────
1. 修改/優化某模組 (如 ReID)      →  modules/reid/             →  更新 architecture.md 或於 research/ 紀錄實驗
2. 做了一個重大技術選型           →  modules/<name>/decisions/ →  撰寫下一個編號的 ADR 文件 (.md)
3. 完成全局/多模組系統實驗        →  research/                 →  在對應領域或 training/ 目錄下紀錄配置與結論
4. 跑了全局效能評測或需維運指南    →  reference/                →  寫入 benchmarks/ 或 runbooks/ 下的 markdown
5. 新增/完成任務項目              →  TODO.md                   →  更新最上方模組進度矩陣並勾選待辦 checkbox [x]
─────────────────────────────────────────────────────────────────────────────────────────────
※ 日常 Bug fix、純重構 (API 外部行為未變) → 無需新增/更新文檔。
```

---

## 🔗 文檔寫作規範

請務必嚴格遵守 **[DOC_MAINTENANCE.md](DOC_MAINTENANCE.md)** 所規定的寫作標準：
* **路徑連結規範**：所有文檔內的相對連結必須使用**相對於目前文件所在位置的相對路徑**，絕對不允許使用 `file:///` 等本機絕對路徑。
* **PR 前檢查**：請確保新實驗、決策已同步勾選並與代碼進度一致。
