# 文檔撰寫規範 (Doc Writing Conventions)

> **去哪寫？** → 見 [docs/README.md](README.md) 的決策樹。
> 本文件定義的是**格式細節**（ADR 編號、progress 歸檔、連結寫法）。

---

## 路徑連結規範

所有文檔內連結使用**相對於目前文件所在位置的相對路徑**。不要使用
`/docs/...`、`/src/...` 這類 repo-root 絕對 Markdown links。

```text
架構入口: architecture/README.md
ADR 004: decisions/004-yolo26-perception.md
tracker_gpu.py from docs/modules/geometry/README.md: ../../../src/saccade/perception/tracking/tracker_gpu.py
```

不對：
```text
../architecture/README.md       ← docs/README.md 內不對：多餘 `../`
/docs/decisions/004.md          ← 不對：repo-root 絕對路徑
```

---

## ADR 編寫規範

### 何時新增
- 技術選型變更（換模型、換框架、換資料庫）
- 核心算法調整
- 設計原則變更

### 狀態流轉
```
Proposed → Accepted → (必要時) Superseded by ADR XXX
```

### 編號規則
- 序號連續。若範圍擴大，新增下一號，不修改舊的。
- 不回頭改已 Accepted 的內容（另開新 ADR 標 Superseded）。

---

## TODO / research / archive 維護規範

- 完成項目時立即勾選 `[x]`
- 實驗、ablation、訓練記錄放 `research/` 或 `modules/<name>/research/`
- 穩定架構說明放 `architecture/` 或 `modules/<name>/architecture.md`
- 操作步驟放 `reference/runbooks/` 或 `modules/<name>/runbooks/`
- 目前沒有 `progress/`、`layers/` 或 `modules/<name>/decisions/` 目錄；不要在文件中指向這些路徑

**生命週期（何時搬家 / 何時收單 / 誰觸發）＝ [契約 C6](ownership/doc_structure_contract.md#c6--lifecycle適用所有-doc-class不只-threads)，本檔不複述。**
重點：**owner 接受 terminal 的同一個 PR 必須完成收單**（registry 更新 → terminal record → note 搬進 `closed/` → 移出 active 索引），
`check_doc_structure.py --strict` 對違規紅燈。「之後再整理」是 doc 只增不減的主因，契約不允許。

---

## Doc Structure Contract（研究 / 模組家與索引）

**This is O1.5 of O-series**（docs-only）：檔案家、入口可發現性、evidence promotion。  
**不是** tracker-decision P9。全文：[ownership/doc_structure_contract.md](ownership/doc_structure_contract.md)。

### 規則的唯一 owner ＝ 契約本身（本檔只指路，不複述）

規則重複陳述會漂移，漂移時沒人知道哪份是真的——那是「以後還是打補釘」的溫床。所以：

| 我要查 | 唯一 owner |
|:--|:--|
| 檔案家 / 分層 · **決策層不出文檔** | [C0 · C0.1](ownership/doc_structure_contract.md#c0--homes-layering) |
| 寫去哪（決策樹） | [C1](ownership/doc_structure_contract.md#c1--writing-decision-tree) |
| 文首狀態標記（`doc-status` / `doc-promotion` / `doc-date`） | [C3](ownership/doc_structure_contract.md#c3--research-note-status-markers-new-notes-required) |
| 索引新鮮度（入口＝目錄） | [C4](ownership/doc_structure_contract.md#c4--index-freshness-entry--catalog) |
| 數字升格 / 不得有第二真相 | [C5](ownership/doc_structure_contract.md#c5--evidence--promotion) |
| **生命週期 · 檔名 vs 目錄 · 收單觸發器 · enforcement** | [C6 · C6.1–C6.4](ownership/doc_structure_contract.md#c6--lifecycle適用所有-doc-class不只-threads) |
| TODO vs research vs threads 角色分工 | [C7](ownership/doc_structure_contract.md#c7--todo-vs-research--threads-role-split) |
| 研究方法 / 證據語義 / claim ladder | [research/contracts/](research/contracts/README.md) |
| 研究對象的**當前狀態** | [claim_state_registry](research/contracts/claim_state_registry.md) |

僅此一條本檔自有：**Cheb-GR / bank / occ-exit 的文檔家 = `modules/semantic/`**（即使 code 在 reid 路徑）。

### 相關 checker

- `scripts/tools/check_doc_structure.py`：索引覆蓋率 warn；**`--strict`（pre_push 使用）對 C6.4 的 L1–L3 違規紅燈**

---

## Workstream WIP（一模主一目標）

**This is O0 of O-series: Ownership / Objective Isolation**（模組目標隔離階段）。

WIP=1 是 **ownership governance 的 process seal**（docs-only）：每個模組負責人至多一個 concurrent active 目標。  
**不是** tracker-decision 的 P9，也**不是** dual-stability / decision-layer 研究線的延伸。

與 [事實所有權](#事實所有權與新鮮度-fact-ownership--freshness) 平行（fact-owner = 事實家；WIP=1 = 進行中目標家）。

已結案研究線（只讀、勿並開）：[research/tracker-decision/status_2026-07-09.md](research/tracker-decision/status_2026-07-09.md)（P0–P8 closed）。  
**O1+** module objective map / routing / extraction：[ownership/README.md](ownership/README.md)（annotate-only；不在本節展開完整表格）。

### 規則

```text
WIP = 1 per module owner  (O0 seal)
- DEVELOPMENT.md「模組現狀總覽」每個 🔄 列只寫「一個」active 目標 one-liner
- 各 docs/modules/<m>/TODO.md = WIP register：
    sole active 一句 + link(s) 到 thread / research
    無 active → 明確 ⏸️ / 無 active
    不寫長文、結果表、推理流水帳
- 跨多步 / 跨家任務 → docs/research/threads/ 導航卡（不消耗 WIP 名額以外的敘事空間）
- 要開第二目標：同一變更內先收合或 park 第一個
- 已結案研究線（如 tracker-decision P0–P8）僅在新證據下以「具名新線」重開，
  不做 drive-by 平行重構
- 寫 paper（paper_outline + evidence_ledger）與改 tracker 行為是不同線；
  同一負責人不同時並進
```

### Dashboard

模組現狀一覽：[DEVELOPMENT.md 模組現狀總覽](../DEVELOPMENT.md)（**只**鏡射 module TODO 的 sole active one-liner；細節與接續在 threads / research）。

### 不在此規則內

- 跨模組 **依賴**（例如 geometry GMC 支援 detection VGT）不算第二目標，但 dashboard 應標註依賴關係。
- Parked one-liners 可列多項；只有 **sole active** 受 WIP=1 約束。
- Research threads 可多張並存（navigation）；**不得**被當成第二個 sole active 來源。

---

## 事實所有權與新鮮度 (Fact Ownership & Freshness)

核心原則見 [DEVELOPMENT.md §6 衝突時誰說了算](../DEVELOPMENT.md) 與本節：**每個事實只有一個家，其餘只鏡射並回連，不複製成獨立事實。** 入口文件最容易 drift 的是 baseline 數字。

### fact-owner marker

在「擁有」或「鏡射」某類事實的段落上方，加一行機器可讀標記：

```html
<!-- fact-owner: <fact-id> = <repo-root-relative path> -->
```

- 路徑一律用 **repo-root 相對路徑**（如 `docs/TODO.md`），與檔案位置無關，方便 checker 解析。
- 標記路徑 **等於所在檔案本身** → 該檔是這項事實的 **owner**。
- 標記路徑 **指向別的檔案** → 該段只是 **mirror**（鏡射），必須同時附一句人類可讀的「唯一來源在 X」並回連。

目前定義的 fact-id：

| fact-id | owner |
|---------|-------|
| `current-baseline` | `docs/TODO.md`「當前 Baseline」節 |

### 入口文件新鮮度合約 (entry freshness contract)

- 入口/敘事文件（`README.md`、`docs/PIPELINE.md`、`docs/DATAFLOW.md`、`docs/PROJECT_SHOWCASE.md` 等）可保留數字作為 headline / 展示 / ablation，但**必須**帶 mirror marker 並回連 owner。
- 不要在入口文件用手寫「最後更新：YYYY-MM-DD」當新鮮度證明——它是死資料。改用 fact-owner marker 指向真正會更新的來源。

### 相關 checker

- `scripts/tools/check_doc_stale_paths.py`：**hard fail**，禁止引用已搬移文件的舊路徑（固定 denylist）。
- `scripts/tools/check_doc_freshness.py`：**warn-only**，提醒手寫日期、缺 marker 的鏡射數字、跨入口重複的 baseline 數字；只警告不擋 CI。
- `scripts/tools/check_doc_structure.py`：**warn-only**，research 索引覆蓋（見 [Doc Structure Contract](#doc-structure-contract研究--模組家與索引)）。
- `scripts/tools/check_doc_links.py`：**hard fail**，相對 Markdown 連結必須可解析。

---

## 不需要寫文檔的情況

- Bug fix（外部行為不變）
- 重構（API 不變）
- 調參/LR sweep（除非寫入 `training/` 實驗記錄）

---

## PR 前檢查

```
□ 新模組有 architecture/ 版本快照？
□ 新實驗有 research/ 記錄？
□ 新 research 檔已更新 owning README 索引行？（Doc Structure C4）
□ 若數字要被 PR/README/paper 引用：已 promotion 到 ledger / report_data / no_go？（C5）
□ 無幽靈路徑、無只寫在 chat 的數字？
□ 系統模組實作進度表與代碼一致？
□ ADR 狀態正確？
□ TODO 仍是 WIP 鎖（sole active 一句 + link），未塞長文 / 結果表？（C7）
□ WIP=1：DEVELOPMENT dashboard 🔄 one-liner 與 module TODO sole active 一致；未雙開？
□ 跨多步任務有 threads 卡或已確認不需？（threads README）
□ 必要時通過 scripts/tools/check_gpu_contract.py 靜態效能合約檢查？
□ 無失效連結或舊模型名稱？（check_doc_links / check_doc_stale_paths）
```
