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
