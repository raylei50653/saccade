# Saccade 開發指南

**角色：** 開發者**薄入口**——先對齊**需求層級**，再取**文檔組合**；細節留在各自的家，本檔不百科化。

```text
進入 → 選層級 → 打開文檔組合 → 改 code / 寫 note → 對層級做驗證
```

長文契約與寫作路由：

- 文件家 / research 索引 / 數字升格 → [docs/ownership/doc_structure_contract.md](docs/ownership/doc_structure_contract.md)（**O1.5**）
- 跨子類連續研究任務 → [docs/research/threads/](docs/research/threads/) 建 navigation-only thread；不放長表、不取代 evidence_ledger / module research。
- 接續任務 → [docs/research/threads/README.md](docs/research/threads/README.md)（先看 Active threads，再進單卡）
- 若 active thread 有同名 `*.dispatch.yaml`，該 sidecar 是 branch／ancestor／scope／concurrency／return-packet 的 execution authority；Chat 複製文字只作導航，不能擴 scope，衝突時 fail closed。
- **TODO = WIP 鎖**（sole active 一句 + link）；**不是**任務敘事 / 上下文恢復 → [DOC_MAINTENANCE § WIP](docs/DOC_MAINTENANCE.md) · [契約 C7](docs/ownership/doc_structure_contract.md)
- 格式、WIP=1、fact-owner → [docs/DOC_MAINTENANCE.md](docs/DOC_MAINTENANCE.md)
- 「我去哪寫」決策樹 → [docs/README.md](docs/README.md)
- 模組目標隔離 → [docs/ownership/README.md](docs/ownership/README.md)

---

## 1. 需求層級（先選這個）

層級愈高，**必讀 + 必寫 + 必驗**愈重。可只升不降：不確定時往上一級。

| 層級 | 何時用 | 意圖 |
|:--|:--|:--|
| **D0** | Bug fix / 純重構 / API 與預設行為不變 | 低摩擦合入 |
| **D1** | 單模組實驗、ablation、default-off probe、文檔-only | RESEARCH 線；不翻 production default |
| **D2** | 跨模組實驗、可被 PR/README **引用**的數字、NO-GO 結案 | 要有 evidence 家 |
| **D3** | 動到 eval 預設路徑、headline preset、inject/contract、native hot path | 行為 / 合約敏感 |
| **D4** | 架構選型、ADR、paper claim、對外敘事 | 決策或出版物級 |

**與 O-series 對照（不互相取代）：**

| 治理 | 管什麼 |
|:--|:--|
| **O0 WIP=1** | 每模組同時最多一個 sole active |
| **O1 objective** | 這次 PR 的 primary 是 RUNTIME / RESEARCH / … |
| **O1.5 結構** | 檔案家、README 索引、promotion |
| **本表 D0–D4** | 這次工作要帶哪一包文檔與驗證 |

---

## 2. 各層文檔組合（讀 / 寫 / 驗）

「組合」= 該層**最低限度**應打開的檔。細節仍以各檔為準。

### D0 — 修復 / 重構

| | 文檔組合 |
|:--|:--|
| **讀** | 相關模組 `docs/modules/<m>/README.md`（I/O 一眼）；熱路徑見下方 §5 |
| **寫** | 通常**不寫**新 docs（見 [DOC_MAINTENANCE](docs/DOC_MAINTENANCE.md)「不需要寫文檔」） |
| **驗** | 單元 / 既有測試；`bash scripts/pre_push.sh` |

### D1 — 單模組研究 / default-off

| | 文檔組合 |
|:--|:--|
| **讀** | 模組 README + TODO；[doc_structure_contract](docs/ownership/doc_structure_contract.md) C1/C4；相關 [no_go_registry](docs/reference/no_go_registry.md) |
| **寫** | `docs/modules/<m>/research/<note>.md`（或跨模組則 `docs/research/<area>/`）+ **owning README 索引一行** + 文首 `doc-status` / `doc-promotion`；TODO 只更新 sole active **one-liner + link**；跨多步 → [threads/](docs/research/threads/) |
| **驗** | 實驗協議自洽即可；**不**要求改 headline；`check_doc_structure`（pre_push warn） |
| **禁** | 同 PR 翻 production default（RESEARCH + default → 拆 PR，見 [change_routing_matrix](docs/ownership/change_routing_matrix.md)） |

Cheb-GR / bank / offline identity / occ-exit → 文檔家 **semantic**（非 reid）。

### D2 — 可引用結果 / 跨模組結論

| | 文檔組合 |
|:--|:--|
| **讀** | D1 組合 + [evidence_ledger](docs/research/evidence_ledger.md) 協議列；必要時 [report_data/README](report_data/README.md) |
| **寫** | D1 正文與索引；**若數字要被引用** → ledger 一列 和/或 no_go 一條 和/或 report_data 表（[契約 C5](docs/ownership/doc_structure_contract.md)）；模組 README GO/NO-GO **一行** |
| **驗** | 標註 commit/preset/host；noise 意識（決策旋鈕 ΔIDF1 ≲ 0.2 見 ledger） |

### D3 — 生產路徑 / 合約

| | 文檔組合 |
|:--|:--|
| **讀** | [PIPELINE.md](docs/PIPELINE.md) 或 [pipeline_flow](docs/reference/pipeline_flow.md)；[mot17_default_config](docs/reference/mot17_default_config.md)；動 association 時 [tracker-decision status](docs/research/tracker-decision/status_2026-07-09.md)（closed 只讀）+ 相關 knobs；CUDA graph → [CUDA_GRAPH_CAPTURE_STREAM_RULE](docs/CUDA_GRAPH_CAPTURE_STREAM_RULE.md)；[change_routing_matrix](docs/ownership/change_routing_matrix.md) |
| **寫** | 行為變更說明；必要時 ADR；合約 / preset 與 [module TODO](docs/modules/) / [docs/TODO.md](docs/TODO.md) 對齊；**新**決策線不得 silent reopen closed P0–P8 |
| **驗** | smoke 至少 MOT17-04-SDP；identity/default → 7-seq；headline YAML → `check_headline_decision_contract.py`；native 改動 → build + smoke |

### D4 — 架構 / 論文 / 對外敘事

| | 文檔組合 |
|:--|:--|
| **讀** | [docs/architecture/README](docs/architecture/README.md)；決策敘事 [paper_outline](docs/research/paper_outline.md) + ledger；Mamba method [report_data](report_data/README.md)（**兩線互指、不互相覆寫**） |
| **寫** | ADR（`docs/decisions/`）；paper 素材進 report_data 或 outline；claim 必須能指回 ledger / tables |
| **驗** | 無「只存在 chat 的數字」；必要時重建 `report_data/build_paper_assets.py` |

---

## 3. 場景速查（靈活微調組合）

| 我要… | 建議層級 | 文檔組合（在層級底稿上加減） |
|:--|:--|:--|
| 修 crash / flake，行為不變 | **D0** | 測試 + pre_push |
| 單模組 ablation，default-off | **D1** | module research + README 索引 + TODO 連結 |
| 數據驅動 gate / relink 訊號（不改 preset） | **D1** | 契約 [signal_table_schema](docs/research/eval/signal_table_schema.md) · **深度分析總帳** [signal_analysis_ledger](docs/research/eval/signal_analysis_ledger.md)（一訊號一列；數字在 `out/signal_study/`） |
| occ-exit / Cheb-GR / sparse bank | **D1→D2** | **semantic** research 全家；引用數字再 D2 promotion |
| 外觀 ceiling / 特徵抽取實作 | **D1** | **reid** README/research；關聯政策仍看 semantic |
| GMC / Kalman 實驗 | **D1→D3** | geometry research + eval 筆記；動 default → 加上 tracker-decision + contract |
| 改 `mamba_whole_graph` 預設旋鈕 | **D3** | mot17_default_config + routing matrix + ledger（若報數字） |
| 新 native kernel / pybind | **D3** | ownership 熱檔卡 + bridge 檢查；CUDA stream 規則 |
| 訓練 Mamba / VGT | **D1→D2** | detection 協議 + research；可引用結果 → ledger/report_data |
| 寫 arXiv / tech report 段落 | **D4** | paper_outline **或** report_data（先選線）+ source_map |
| 工業 RTSP / storage / RAG | **D1–D3** | 對應 `modules/streaming|storage|cognition|resource` + runbooks |
| 只更新文檔結構 / 索引 | **D1** | O1.5 契約 + DOC_MAINTENANCE checklist |

寫作目錄細節（檔放哪）：[docs/README 決策樹](docs/README.md)。

---

## 4. 現況快照

### Baseline

**生產 baseline preset = `mamba_whole_graph`**（ReID off；whole-graph detect）。

<!-- fact-owner: current-baseline = docs/TODO.md -->

**數字唯一來源：** [docs/TODO.md](docs/TODO.md)「當前 Baseline」——本檔不嵌指標表。  
CLI 默認細節：[docs/reference/mot17_default_config.md](docs/reference/mot17_default_config.md)。

```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer
```

### 模組現狀總覽

> **Dashboard fact-owner：本節。** 只鏡射各 `docs/modules/<m>/TODO.md` 的 **sole active one-liner**（WIP 鎖），**不**鏡射細節、**不**另立第二待辦清單。  
> **任務敘事 / 接續** → [docs/research/threads/](docs/research/threads/README.md)；**事實** → module research / ledger。  
> **O0 / WIP=1：** 每模組 🔄 最多一個 active。規則：[DOC_MAINTENANCE § WIP](docs/DOC_MAINTENANCE.md)。  
> tracker-decision P0–P8 **closed**：[status](docs/research/tracker-decision/status_2026-07-09.md) — 非 O-series 延續。

| 模組 | 狀態 | sole active（WIP 鎖） | TODO |
|------|------|---------------------|------|
| 🔍 detection | 🔄 active | VGT-Mamba（訓練中） | [↗](docs/modules/detection/TODO.md) |
| 📐 geometry | 🔄 active | GMC Warp 精度驗證（依賴 VGT） | [↗](docs/modules/geometry/TODO.md) |
| 🧬 reid | ⏸️ 暫緩 | — | [↗](docs/modules/reid/TODO.md) |
| 🔄 lifecycle | 📋 待辦 | evaluator lifecycle 測試切片 | [↗](docs/modules/lifecycle/TODO.md) |
| 🌀 motion | 🟢 收斂 | — | [↗](docs/modules/motion/TODO.md) |
| 🤝 semantic | 🔄 active | Safe-Region Assetization — R1 deterministic G1–G3 asset conversion | [↗](docs/modules/semantic/TODO.md) |
| ⚡ trigger | 🟢 收斂 | — | [↗](docs/modules/trigger/TODO.md) |
| 🖥️ streaming | 🟢 收斂 | — | [↗](docs/modules/streaming/TODO.md) |
| 💾 storage | 🟢 收斂 | — | [↗](docs/modules/storage/TODO.md) |
| 🧠 cognition | 🟢 收斂 | — | [↗](docs/modules/cognition/TODO.md) |
| ⚙️ resource | 🟢 收斂 | — | [↗](docs/modules/resource/TODO.md) |

全局矩陣 / 跨模組待辦：[docs/TODO.md](docs/TODO.md)。  
Detection 設計索引（非本檔展開）：[docs/modules/detection/README.md](docs/modules/detection/README.md)。

---

## 5. 熱路徑與高頻命令

### MOT 主線常改檔

| 意圖 | 起點 |
|:--|:--|
| Tracker / association | `src/tracking/tracker_gpu.cu` → `tracker_gpu.hpp` → `tracker_gpu.py` |
| Relink / identity | `src/saccade/perception/eval/relink.py`、native bridge |
| Detect / post | `eval/detection.py`、Mamba / TRT 路徑 |
| Eval 編排 | `scripts/eval/mot17.py`、`eval/evaluator.py`、`eval/pipeline.py` |
| Preset / knobs | `scripts/eval/config/`、`configs/presets/` |

系統分層與工業路徑總圖：[docs/architecture/README.md](docs/architecture/README.md)。

### 推送前

```bash
git config core.hooksPath .githooks   # 一次
bash scripts/pre_push.sh              # lockfile + ruff + mypy + pytest + doc checks + C++ 偵測
bash scripts/pre_push.sh --fix        # 先 ruff fix/format
```

Doc 相關（pre_push 已掛）：`check_doc_links` · `check_doc_stale_paths` · freshness warn · **structure warn（O1.5）**。

### 依層級加驗證

```bash
# D0/D1
uv run pytest tests/ --ignore=tests/benchmarks

# D3 常見
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer --sequences MOT17-04-SDP
uv run python scripts/tools/check_headline_decision_contract.py

# native 有改且 build/ 存在時，pre_push 會觸發 tracking ext build
```

實驗分層（避免只看最終 IDF1）：module-local signal → 再 e2e；`local↑ + downstream↓` = regression。

### 分支（摘要）

- `main` only 合入目標；工作分支 `feat/*` `fix/*` `perf/*` `docs/*` `research/*`
- 不直接 push `main`；PR + CI
- 開分支前：工作項對齊 **module TODO sole active**（WIP 鎖）或全局 [docs/TODO.md](docs/TODO.md)；連續任務先讀 [threads/](docs/research/threads/README.md)

### 實驗追蹤（可選）

MLflow / Optuna 未啟動不阻 eval。啟動與查詢：

- `scripts/ops/mlflow_server.sh`
- `scripts/eval/mlflow_logger.py`
- `scripts/tools/compare_trials.py`

---

## 6. 衝突時誰說了算

對有 chat-owned dispatch 的 governed research task：dispatch 只負責 branch、ancestor、current gate、write scope、concurrency、authorization 與 return packet；它不改寫主程式或研究事實。任何複製進 agent 的 Chat 尾部都只是導航，與 repo authority 衝突時必須 fail closed。

1. **主路徑程式碼**（`src/saccade/perception/`、`src/tracking/`、`scripts/eval/mot17.py`）
2. **合約 / 預設**（headline YAML、`check_headline_decision_contract`、accepted ADR）
3. **事實家**（baseline → `docs/TODO.md`；決策數字 → `evidence_ledger`；模組 sole active → 本檔 dashboard 鏡射 module TODO）
4. **入口敘事**（本檔、PIPELINE、showcase）— 只鏡射，不另造數字

本檔**不是** association 語意百科，也**不是** paper 第二真相。

---

## 7. 入口地圖（其餘家）

| 需求 | 去 |
|:--|:--|
| 寫 docs / research 路由 | [docs/README.md](docs/README.md) · [O1.5 契約](docs/ownership/doc_structure_contract.md) |
| 格式 · WIP · fact-owner | [DOC_MAINTENANCE.md](docs/DOC_MAINTENANCE.md) |
| 目標隔離 · PR 檢查矩陣 | [docs/ownership/](docs/ownership/README.md) |
| 演算法主線精煉 | [docs/PIPELINE.md](docs/PIPELINE.md) |
| Stage dataflow | [docs/reference/pipeline_flow.md](docs/reference/pipeline_flow.md) |
| NO-GO 總表 | [docs/reference/no_go_registry.md](docs/reference/no_go_registry.md) |
| 決策層（closed） | [tracker-decision/](docs/research/tracker-decision/README.md) |
| 全局 research 入口 | [docs/research/README.md](docs/research/README.md) |
| 連續任務母線 | [docs/research/threads/](docs/research/threads/README.md) |
| Paper assets | [report_data/README.md](report_data/README.md) |
| 倉庫目錄約定 | [REPO_LAYOUT.md](REPO_LAYOUT.md) |

---

## 原則（三條）

- 架構與合約決定什麼值得做；TODO sole active = WIP 鎖；threads = 怎麼接續。
- 單一原始碼檔原則上不超過 **1000** 行；主熱路徑 GPU / native first。
- **每個事實一個家**；入口只組合連結，不複製長表。
