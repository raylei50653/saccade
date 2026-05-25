# Docs

> **文檔系統原則：一個文檔回答一個問題。透過索引串聯，不靠合併。**

---

## 我去哪寫？

```
我做了什麼？              →  去哪個目錄？         →  寫什麼格式？
─────────────────────────────────────────────────────────────────
新增/重構核心模組          →  architecture/       →  版本快照：vN 改了什麼、為何
做完一個訓練實驗            →  training/           →  實驗記錄：配置、loss、結論
做了一個架構決策            →  decisions/          →  ADR：選項、理由、後果
新增/完成 TODO 項目        →  TODO.md             →  勾選 [x]
寫了一個功能模組的設計      →  temporal_yolo/ 等    →  設計文件
需要操作步驟               →  runbooks/           →  操作手冊
需要規格/數字參考           →  reference/          →  規格表
─────────────────────────────────────────────────────────────────
日常 Bug fix / 調參       →  不用寫文檔
純重構（外部行為不變）      →  不用寫文檔
```

---

## 每個目錄做什麼？

| 目錄 | 回答的問題 | 文檔範例 |
|------|-----------|---------|
| `architecture/` | 系統長什麼樣？版本間改什麼？ | `v1-core.md`, `v2-gmc.md` |
| `training/` | 這個實驗試了什麼？結果如何？ | `jde-market-1501.md` |
| `decisions/` | 我們為什麼這樣選？ | `004-yolo26-perception.md` |
| `temporal_yolo/` | Option E/F 設計原理 | `option-e-v2-design.md` |
| `reference/` | 精確的數字、規格、CLI flags | `PIPELINE_REFERENCE.md` |
| `runbooks/` | 我怎麼操作？ | `runbooks/latency_profile.md` |
| `progress/` | 各模組做到哪了？ | `perception.md` |
| `layers/` | 各層架構詳解 (What & Why) | `L1_perception.md` |
| `archive/` | 過時的實驗、已完成的專案 | 歷史文件 |
| `TODO.md` | 接下來要做什麼？ | checkbox list |

---

## 文檔格式

### 架構版本快照 (`architecture/vN-name.md`)

```markdown
# 架構 vN：標題

日期：YYYY-MM-DD

## 變更
- 改什麼

## 理由
- 為什麼改

## 涉及模組
| 模組 | 變更 |
|------|------|
| x   | y    |
```

### 訓練實驗記錄 (`training/experiment-name.md`)

```markdown
# 實驗：標題

日期：YYYY-MM-DD  
命令：...
狀態：進行中 / GO / NO-GO

## 配置
## 架構（圖/表）
## 結果
## 結論
```

### 架構決策 (`decisions/NNN-title.md`)

依循現有 ADR 格式。
