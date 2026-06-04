# 文檔撰寫規範 (Doc Writing Conventions)

> **去哪寫？** → 見 [docs/README.md](README.md) 的決策樹。
> 本文件定義的是**格式細節**（ADR 編號、progress 歸檔、連結寫法）。

---

## 路徑連結規範

所有文檔內連結使用**相對於 `docs/` 的路徑**：

```
[架構 v2](architecture/v2-gmc.md)      ← 正確
[ADR 004](decisions/004-yolo26.md)     ← 正確
[README](../README.md)                 ← 正確（從 docs/ 子目錄往上）
```

不對：
```
[架構](../architecture/v2-gmc.md)      ← 不對：多餘 `../`
[ADR](/docs/decisions/004.md)         ← 不對：絕對路徑
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

## progress/ 維護規範

- 完成項目時立即勾選 `[x]`
- 全部完成後移至 `decisions/archive/`
- `progress/` 不寫架構說明（放 `layers/` 或 `architecture/`）
- `progress/` 不寫操作步驟（放 `runbooks/`）

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
□ 通過 check_gpu_contract.py 靜態效能合約檢查？
□ 無失效連結或舊模型名稱？
```
