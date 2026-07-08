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
- 已完成、收斂或放棄的長篇脈絡移至 [TODO_history.md](TODO_history.md) 或 [archive/](archive/)
- 實驗、ablation、訓練記錄放 `research/` 或 `modules/<name>/research/`
- 穩定架構說明放 `architecture/` 或 `modules/<name>/architecture.md`
- 操作步驟放 `reference/runbooks/` 或 `modules/<name>/runbooks/`
- 目前沒有 `progress/`、`layers/` 或 `modules/<name>/decisions/` 目錄；不要在文件中指向這些路徑

---

## 事實所有權與新鮮度 (Fact Ownership & Freshness)

核心原則見 [DEVELOPMENT.md §11](../DEVELOPMENT.md)：**每個事實只有一個家，其餘只鏡射並回連，不複製成獨立事實。** 入口文件最容易 drift 的是 baseline 數字。

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
□ 系統模組實作進度表與代碼一致？
□ ADR 狀態正確？
□ TODO.md 已完成項目已勾選？
□ 必要時通過 scripts/tools/check_gpu_contract.py 靜態效能合約檢查？
□ 無失效連結或舊模型名稱？
```
