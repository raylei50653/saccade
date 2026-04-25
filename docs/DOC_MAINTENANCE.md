# 文件維護規範 (Doc Maintenance Guide)

本文件定義 `docs/` 目錄的結構規範、各類文件的職責邊界，以及**何時必須更新哪份文件**。目標是讓文件與代碼保持同步，避免新人看到矛盾或過時的資訊。

---

## 1. 目錄結構與職責

```
docs/
├── architecture.md          ← 系統全貌（L1-L6 分層、資料流、資料夾定義）
├── pipeline_flow.md         ← 完整資料流程圖與每階段細節
├── api_spec.md              ← Redis 事件結構、ChromaDB Schema、對外 API
├── TODO.md                  ← 具體待辦清單（對應代碼位置）
│
├── layers/                  ← 各層架構說明（What & Why，不追蹤進度）
│   ├── L1_perception.md
│   ├── L2_vector_path.md
│   ├── L3_L4_storage.md
│   └── L5_L6_cognition.md
│
├── decisions/               ← 架構決策紀錄 ADR（不可回頭改決策內容）
│   ├── README.md            ← ADR 索引
│   ├── 004-yolo26-perception.md
│   ├── ...
│   └── archive/             ← 已完成的整合路線圖（唯讀）
│
├── progress/                ← 各模組實作進度（純狀態快照）
│   ├── perception.md
│   ├── storage.md
│   ├── cognition.md
│   ├── media.md
│   └── infra.md
│
├── experiments/             ← 實驗紀錄（日期命名，唯讀）
├── benchmarks/              ← 效能測試數據
└── runbooks/                ← 故障排除與日常維運指令
```

---

## 2. 各類文件的職責邊界

| 文件類型 | 寫什麼 | 不寫什麼 |
|---|---|---|
| `architecture.md` | 系統整體分層、資料流、設計原則 | 實作細節、進度狀態 |
| `pipeline_flow.md` | 完整流程圖與各階段說明 | 模組內部實作 |
| `layers/` | 各層的定義、組件、資料流、效能指標 | 進度 checkbox、待辦事項 |
| `decisions/` | 決策背景、選擇理由、影響 | 實作步驟、進度追蹤 |
| `progress/` | checkbox 狀態、已完成里程碑、待辦事項 | 架構說明（放 layers/） |
| `TODO.md` | 具體待辦、對應文件位置、落差說明 | 已完成的項目（移至 progress 里程碑） |
| `runbooks/` | 操作指令、故障排除步驟 | 架構決策理由 |

**核心原則：同一件事只在一個地方寫。**
- 組件說明 → `layers/`
- 進度狀態 → `progress/`
- 決策理由 → `decisions/`

---

## 3. 觸發更新的時機

### 改了代碼，必須同步更新的文件

| 代碼變動類型 | 必須更新 | 選擇性更新 |
|---|---|---|
| 新增 / 替換核心模型（如 YOLO、SigLIP） | `architecture.md`、`layers/` 對應層、`progress/` | `pipeline_flow.md` |
| 新增追蹤算法或參數調整 | `layers/L1_perception.md`、`progress/perception.md` | `architecture.md` Tracker Stack |
| 修改 Redis 事件結構或 ChromaDB Schema | `api_spec.md` | `layers/L3_L4_storage.md` |
| 新增 / 完成 TODO 項目 | `TODO.md`（勾選）、`progress/` 對應模組 | — |
| 重大架構決策（換技術棧、改設計原則） | 新增 ADR、`architecture.md`、`layers/` | `pipeline_flow.md` |
| 刪除腳本 / 移除功能 | 所有引用該腳本的文件 | — |

### 不需要更新文件的情況
- Bug fix（不改行為）
- 重構（外部介面不變）
- 效能調參（除非更新 benchmarks/）

---

## 4. ADR 編寫規範

### 何時新增 ADR
- 技術選型變更（換模型、換框架、換資料庫）
- 核心算法調整（改匹配策略、改 Kalman 參數定義）
- 設計原則變更（如從 VLM-Free 改為 Agentic RAG）

### ADR 狀態流轉
```
Proposed → Accepted → (必要時) Superseded by ADR XXX
```
- **Proposed**：提案中，尚未實作。
- **Accepted**：已落地，代碼已更新。
- **Superseded**：被更新的 ADR 取代，原文件保留作歷史參考，不刪除。

### ADR 編號規則
- 序號連續，不重複。若決策範圍擴大，新增下一號 ADR 而非修改舊的。
- 已落地的整合路線圖（非決策）移至 `decisions/archive/`。

### ADR 不應做的事
- 不回頭修改已 Accepted 的決策內容（加 Superseded 另開新 ADR）。
- 不記錄實作細節（放 `progress/` 或 `layers/`）。

---

## 5. progress/ 維護規範

### 更新時機
- 完成一個 checkbox 項目時，立即標記 `[x]`。
- 發現文件描述與代碼不符時，優先更新文件（不是等到「空了再一起改」）。

### 歸檔時機
當一份 progress 文件**所有項目全部完成**，且對應的架構說明已回寫到 `layers/` 或 ADR 時，將其移至 `decisions/archive/`。

### 禁止事項
- `progress/` 不寫架構說明（架構說明放 `layers/`）。
- `progress/` 不寫操作指令（操作指令放 `runbooks/`）。

---

## 6. 常見錯誤與預防

| 常見錯誤 | 預防方式 |
|---|---|
| 文件描述舊模型名稱（如 YOLO11、Jina-CLIP） | 替換模型後，全域搜尋舊名稱 `grep -r "YOLO11" docs/` |
| progress/ 待辦與 TODO.md 不同步 | 完成項目時兩個都更新 |
| ADR 編號重複或跳號 | 新增 ADR 前先看 `decisions/README.md` 索引 |
| 腳本路徑失效（引用已刪除的 scripts/） | 刪除腳本後搜尋 `grep -r "scripts/" docs/` |
| architecture.md 與 layers/ 描述矛盾 | `architecture.md` 只寫摘要，細節下放 `layers/`，兩者不重複 |
| ADR 狀態停在「執行中」但已落地 | 代碼合併後立即更新 ADR 狀態為 Accepted |

---

## 7. 快速檢查清單（PR 前）

```
□ 新模型/算法有更新 layers/ 對應層？
□ progress/ checkbox 與實際代碼狀態一致？
□ 若有重大架構決策，是否新增 ADR？
□ ADR 狀態是 Accepted（若已落地）？
□ TODO.md 已完成項目是否勾選？
□ 無引用已刪除腳本或舊模型名稱？
□ architecture.md 與 pipeline_flow.md 是否反映最新流程？
```

---

最後更新：2026-04-25
