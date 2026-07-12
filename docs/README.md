# Saccade 核心設計與維護文檔庫

> **文檔維護原則：一個文檔只回答一個問題。透過相對索引串聯，不靠無限合併。**

---

## 📂 核心目錄架構

Saccade 的文檔庫採用「模組化物理結構」配合「全局共享目錄」的組織形式：

* 🎓 **[PROJECT_SHOWCASE.md](PROJECT_SHOWCASE.md)**：**專題展示與答辯主敘事**。將研究問題、SSM-FPN、GPU tracker、可驗證結果、展示流程與數字邊界收斂為單一文件。
* 🧭 **[PIPELINE.md](PIPELINE.md)**：**演算法主線精煉（單檔入口）**。按處理流程 `前處理 → ssm-head → 後處理 → GMC → ID分配 → relink` 走一遍，每階段只留「現行最優 / GO / NO-GO」結論，細節下沉到下列 modules/ 與 registry。**想快速掌握全鏈先讀這份。**
* 📦 **[modules/](modules)**：**核心模組化文檔庫**。物理上按系統功能與 `mot17.py` 評測模組劃分，包含各自的設計、決策 (ADR) 與實驗分析。
* 📊 **[reference/](reference)**：全局評測基準 (benchmarks)、多進程並行評測手冊與跨模組共享的流程規範。亦含 **[NO-GO 全局登記表](reference/no_go_registry.md)**（已結案/已踩雷方向總覽，探索新方向前先查）。
* 📦 **[archive/](archive)**：過時或已完成的歷史參考資料（如 Option D 探索結論）。
* 🔬 **[research/](research)**：跨模組實驗、決策語義、evidence ledger；結構契約見 [ownership/doc_structure_contract.md](ownership/doc_structure_contract.md)。
* 📑 **[../report_data/](../report_data)**：Paper 可重建表/圖與 Mamba method 素材（與 `research/paper_outline` 互指、不互相覆寫）。

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
| 🤝 **[semantic/](modules/semantic)** | Offline identity / Cheb-GR / bridge relink 研究主家（headline ReID off；critical-path ReID NO-GO #57） | * [模組 README · 研究全表](modules/semantic/README.md)<br>* [occ-exit WP3 promotion](modules/semantic/research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md)（🔄 active）<br>* [offline relink hub](modules/semantic/research/offline_relink_candidate_analysis.md)<br>* ADR [015](decisions/015-sinkhorn-auction-hybrid-association.md) / [016](decisions/016-rerank-phase3-reference-quality.md) |
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
我做了什麼？                          →  去哪個目錄？                    →  寫什麼 / 必做
──────────────────────────────────────────────────────────────────────────────────────────────
1. 單模組實驗 / ablation              →  modules/<m>/research/          →  全文 + 父 README 索引一行
2. Cheb-GR / bank / occ-exit / offline identity
                                      →  modules/semantic/              →  非 reid（code 可在 reid 路徑）
3. 特徵抽取 / Feature Bank 實作       →  modules/reid/                  →  architecture 或 research/
4. 全局 / 跨模組實驗                  →  research/<area>/               →  子目錄或 research/README 索引
5. 可引用 baseline / 決策數字         →  research/evidence_ledger.md    →  加列 + 連 source
6. 論文 claim / 可重建表圖            →  report_data/                   →  source_map 或 README 回連
7. 結案 one-shot / 廢棄設計           →  archive/                       →  活躍索引移除或標 historical
8. 重大技術選型                       →  decisions/                     →  下一號 ADR + 模組 README/TODO 連結
9. 全局效能評測 / 維運                →  reference/                     →  benchmarks/ 或 runbooks/
10. 任務勾選                          →  TODO.md / modules/<m>/TODO.md  →  WIP=1；長文不塞 TODO
──────────────────────────────────────────────────────────────────────────────────────────────
※ 日常 Bug fix、純重構 (API 外部行為未變) → 無需新增/更新文檔。
※ 先依 [DEVELOPMENT action cards](../DEVELOPMENT.md#agent-action-cards) 選動作；完整規範以 `ownership/doc_structure_contract.md` 為準。
```

---

## 🔗 文檔寫作規範

路徑與 PR 維護規範見 **[DOC_MAINTENANCE.md](DOC_MAINTENANCE.md)**；文件家、promotion 與 lifecycle 以 **[Doc Structure Contract](ownership/doc_structure_contract.md)** 為準：
* **路徑連結規範**：所有文檔內的相對連結必須使用**相對於目前文件所在位置的相對路徑**，絕對不允許使用 `file:///`、`/docs/...`、`/src/...` 等本機或 repo-root 絕對路徑。
* **PR 前檢查**：請確保新實驗、決策已同步勾選並與代碼進度一致。
