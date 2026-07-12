# Saccade 開發指南

**角色：** 開發者**薄入口**——先對齊**需求層級**，再取**文檔組合**；細節留在各自的家，本檔不百科化。

```text
選層級 → 用下方 action card 找 owner / 必讀 / 必改 → 驗證
```

長文契約與寫作路由：

- 文件家 / research 索引 / 數字升格 → [docs/ownership/doc_structure_contract.md](docs/ownership/doc_structure_contract.md)（**O1.5**）
- 跨子類連續研究任務 → [docs/research/threads/](docs/research/threads/) 建 navigation-only thread；不放長表、不取代 evidence_ledger / module research。
- 接續任務 → [docs/research/threads/README.md](docs/research/threads/README.md)（先看 Active threads，再進單卡）
- **Active research task routing（contract-driven；四層分離）：**
  ```text
  registry    決定「現在是什麼」  → docs/research/contracts/claim_state_registry.md（state fact-owner）
  contracts   決定「可以去哪裡」  → docs/research/contracts/（合法轉移 / 證據型別 / claim ladder）
  O0          決定「現在選哪一個」→ module TODO sole active（從候選集中選,WIP=1）
  DEVELOPMENT 只公告選擇結果      → 本檔（projection;不得成為第二個狀態源）
  ```
  registry **只產生合法候選集**，**不選任務**：`next admissible unit ≠ next task`。
  推導規則：**證據型別錯層 → 改層**（不是修統計）；**substrate 未證（L4 缺）→ 先做 transfer 審計**
  （「已被授權」≠「admissible」）；**blocker 分型**（inadmissibility ＝排除／dependency ＝展開依賴）；
  **decision relevance 過不了反事實測試（正反結果都不改變任何已知決策）→ 不得取得 WIP 鎖**。
  某 layer 若無契約，其 object ＝ `transition_semantics: unavailable`（**合法狀態**，把缺口顯式化）。
  **本檔不重述任何 object 的 rung／limits／substrate**——那些只從 registry 投影。
  GitHub PR metadata is the authority for head/base branch, merge base, head SHA, changed files, CI/test status, review findings, and update history. Research documents remain the authority for research gate, accepted priors, normative inputs, authorized/unauthorized scope, claim boundaries, research acceptance, and next-stage authorization.
- **Lock:** PR merge ≠ research acceptance; engineering-ready ≠ evidence promotion; research acceptance and next-stage authorization remain chat-side / research-owner gates.
- **TODO = WIP 鎖**（sole active 一句 + link）；**不是**任務敘事 / 上下文恢復 → [DOC_MAINTENANCE § WIP](docs/DOC_MAINTENANCE.md) · [契約 C7](docs/ownership/doc_structure_contract.md)
- 格式、WIP=1、fact-owner → [docs/DOC_MAINTENANCE.md](docs/DOC_MAINTENANCE.md)
- 「我去哪寫」決策樹 → [docs/README.md](docs/README.md)
- 模組目標隔離 → [docs/ownership/README.md](docs/ownership/README.md)
- **Retired:** direct Chat↔agent `*.dispatch.yaml` sidecars (branch/ancestor/start-protocol/return-packet authority) are **not** active execution authority. Do not recreate them.

---

## Agent action cards

先選 D-level；下表是執行時的最小路由，**不取代**契約的規範 authority。

| 我要做什麼 | 唯一 owner / 必改 | 必讀 | 驗證 |
|:--|:--|:--|:--|
| 新增單模組 research note（D1） | note 本體；owning module README 索引；若選為 sole active，module TODO 只留 pointer | 契約 C1、C3、C4、C7；module README/TODO | `check_doc_structure.py`（索引為 warn）|
| 將結果作決策、baseline、NO-GO 或 paper 引用（D2） | 上列 + C5 選定的 `evidence_ledger`、`no_go_registry` 和/或 `report_data` owner | D1 包 + C5 + 對應 evidence 文件 | source 可追溯；commit/preset/host 齊全 |
| 收尾 research（不論 D1/D2） | canonical 高密度結論；檔案/索引；module TODO；**只有**已登記 object 的 accepted state / substrate / limits / transition metadata 改變才更新 registry；有 thread 才更新 thread | C4、C6、C7；有 promotion 再讀 C5；有 thread 再讀 [thread close checklist](docs/research/threads/README.md#how-to-close-a-thread) | `check_doc_structure.py --strict` + link/stale-path checks |

### 研究收尾卡

同一個 PR 依此順序完成；不要另開「整理文件」任務。

1. 在 canonical research note 寫一份結論：裁決、適用範圍、限制、證據位置。
2. 將狀態改為 `closed`，移出 active 路徑到 owner 的 `closed/`（或僅在 one-shot 時 archive）；更新 owning README 的索引。
3. 只有 terminal acceptance 改變**已登記** object 的 accepted state、substrate、limits 或 transition metadata 時，才更新 [claim state registry](docs/research/contracts/claim_state_registry.md)；否則不碰 registry。TODO 只改 sole-active pointer 或標成無 active，不能貼結案正文。
4. 有 thread 才依 thread close checklist 更新 frontmatter、`threads/closed/` 與 Closed 表；沒有 thread 不需建立一張。
5. 若結果在 note 外被引用，依 C5 promotion；否則 `doc-promotion: none`。
6. 跑 strict lifecycle、link 與 stale-path checks。預設 `check_doc_structure.py` 只警告索引；`--strict` 使 lifecycle L1–L4 失敗，且 pre-push 使用它。

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
| **讀** | 模組 README + TODO；[doc_structure_contract](docs/ownership/doc_structure_contract.md) C1/C3/C4/C7；相關 [no_go_registry](docs/reference/no_go_registry.md) |
| **寫** | `docs/modules/<m>/research/<note>.md`（或跨模組則 `docs/research/<area>/`）+ **owning README 索引一行**（跨模組：local `<area>/README.md` 優先；不存在才 `docs/research/README.md`）+ 文首 `doc-status` / `doc-promotion`；TODO 只更新 sole active **one-liner + link**；跨多步 → [threads/](docs/research/threads/) |
| **驗** | 實驗協議自洽即可；**不**要求改 headline；`check_doc_structure` 索引覆蓋為 warn（pre-push 對 lifecycle 另跑 strict） |
| **禁** | 同 PR 翻 production default（RESEARCH + default → 拆 PR，見 [change_routing_matrix](docs/ownership/change_routing_matrix.md)） |

Cheb-GR / bank / offline identity / occ-exit → 文檔家 **semantic**（非 reid）。

### D2 — 可引用結果 / 跨模組結論

| | 文檔組合 |
|:--|:--|
| **讀** | D1 組合 + [契約 C5](docs/ownership/doc_structure_contract.md#c5--evidence--promotion)；[evidence_ledger](docs/research/evidence_ledger.md) 協議列；必要時 [report_data/README](report_data/README.md) |
| **寫** | D1 正文與索引；**若數字要被引用** → ledger 一列 和/或 no_go 一條 和/或 report_data 表（[契約 C5](docs/ownership/doc_structure_contract.md)）；module README 只保留中性索引與連結，不複製 GO/NO-GO |
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
| 數據驅動 gate / relink 訊號（不改 preset） | **D1** | 契約 [signal_table_schema](docs/research/contracts/signal_table_schema.md) · **深度分析總帳** [signal_analysis_ledger](docs/research/eval/signal_analysis_ledger.md)（一訊號一列；數字在 `out/signal_study/`） |
| **安全域 / safe region / reject 規則**（`max G_FP` s.t. `L_GT ≤ ε`） | **D2** | **數學契約（先讀，勿自造統計）** [feasible-set framework](docs/research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（feasible/productive-safe set · region geometry · **independence unit 強制宣告** · claim ladder L0–L6 · §13 forbidden shortcuts）· 打包 [RegionAsset 契約](docs/research/contracts/safe_region_asset_contract.md) · 分層 [gate vs score](docs/research/contracts/signal_table_schema.md) |
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

本節只提供**穩定入口**，不再手寫 live status。模組 sole active 的唯一來源是各 module
`TODO.md`；研究物件狀態的唯一來源是
[claim state registry](docs/research/contracts/claim_state_registry.md)。

| 模組 | 入口（live WIP 在 TODO） |
|------|------|
| 🔍 detection | [TODO](docs/modules/detection/TODO.md) · [README](docs/modules/detection/README.md) |
| 📐 geometry | [TODO](docs/modules/geometry/TODO.md) |
| 🧬 reid | [TODO](docs/modules/reid/TODO.md) |
| 🔄 lifecycle | [TODO](docs/modules/lifecycle/TODO.md) |
| 🌀 motion | [TODO](docs/modules/motion/TODO.md) |
| 🤝 semantic | [TODO](docs/modules/semantic/TODO.md) · [registry](docs/research/contracts/claim_state_registry.md) |
| ⚡ trigger | [TODO](docs/modules/trigger/TODO.md) |
| 🖥️ streaming | [TODO](docs/modules/streaming/TODO.md) |
| 💾 storage | [TODO](docs/modules/storage/TODO.md) |
| 🧠 cognition | [TODO](docs/modules/cognition/TODO.md) |
| ⚙️ resource | [TODO](docs/modules/resource/TODO.md) |

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

### 推送前（約束）

```text
修改 → 必要時 pre_push --fix → commit → pre_push（必須綠、working tree clean）→ push 工作分支 → PR + CI
```

- `pre_push` 失敗**不得** push。`--fix` 會改 working tree；修完（含 auto-fix / review 補丁）須**先 re-commit 或 amend**，確認 clean 後再重跑至綠——綠燈只對**已提交**內容有效。
- 不直推 `main`。檢查清單以 [`scripts/pre_push.sh`](scripts/pre_push.sh) 為準，本檔不展開。
- PR merge ≠ research acceptance（§6）。

```bash
git config core.hooksPath .githooks   # 一次
bash scripts/pre_push.sh --fix        # 可選；改檔後必須再 commit
bash scripts/pre_push.sh              # 每次 push 前；須綠且 tree clean
```

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
- 不直接 push `main`；**PR + CI** is the delivery path for engineering work (including governed research implementation)
- 開分支前：工作項對齊 **module TODO sole active**（WIP 鎖）或全局 [docs/TODO.md](docs/TODO.md)；連續任務先讀 [threads/](docs/research/threads/README.md) → normative contract → open/update PR against the agreed base
- Do **not** put mutable branch tips or commit hashes in this file; live tip/SHA/CI live on the PR

### 實驗追蹤（可選）

MLflow / Optuna 未啟動不阻 eval。啟動與查詢：

- `scripts/ops/mlflow_server.sh`
- `scripts/eval/mlflow_logger.py`
- `scripts/tools/compare_trials.py`

---

## 6. 衝突時誰說了算

For governed research delivery, **PR metadata** owns engineering location and concurrency (head/base, merge base, head SHA, files, CI, reviews). **Research documents** (thread + accepted contracts + module research notes) own research gate, scope, claim boundaries, research acceptance, and next-stage authorization.

**Chat text is navigation until written back:** Chat may propose fixes, scope adjustments, or verdicts. It does **not** become cross-conversation authority until those decisions are recorded in the thread, contract, and/or PR. Chat cannot expand scope or override repo authorities merely by being pasted into an agent session.

```text
PR merge ≠ research acceptance
engineering-ready ≠ evidence promotion
research acceptance / next-stage auth = chat-side / research-owner gates
```

1. **主路徑程式碼**（`src/saccade/perception/`、`src/tracking/`、`scripts/eval/mot17.py`）
2. **合約 / 預設**（headline YAML、`check_headline_decision_contract`、accepted ADR）
3. **事實家**（baseline → `docs/TODO.md`；決策數字 → `evidence_ledger`；模組 sole active → module TODO）
4. **Research gates**（active thread + accepted contracts）— maturity, claim levels, stage authorization
5. **入口敘事**（本檔、PIPELINE、showcase）— 只鏡射，不另造數字；不含 live branch tips / commit hashes

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
| **研究規範層**（方法／統計／claim ladder；**先讀，勿自造**） | **[docs/research/contracts/](docs/research/contracts/README.md)** |
| 全局 research 入口 | [docs/research/README.md](docs/research/README.md) |
| 連續任務母線 | [docs/research/threads/](docs/research/threads/README.md) |
| Paper assets | [report_data/README.md](report_data/README.md) |
| 倉庫目錄約定 | [REPO_LAYOUT.md](REPO_LAYOUT.md) |

---

## 原則（三條）

- 架構與合約決定什麼值得做；TODO sole active = WIP 鎖；threads = 怎麼接續。
- 單一原始碼檔原則上不超過 **1000** 行；主熱路徑 GPU / native first。
- **每個事實一個家**；入口只組合連結，不複製長表。
