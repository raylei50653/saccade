# Project Direction

> **這份文件是什麼**：專案級方向語義的唯一敘事面。它定義已確立的範式邊界、execution-chartered 的近程目標、以及長期戰略路徑的進入/退出語義。
> **這份文件不是什麼**：不是決策層、不是狀態層、不是數字家、不是進度頁。

**Strategic thesis：** Saccade is a real-time-first, geometry-first, ReID-free MOT research system; new capabilities must earn their way into production through attributable evidence rather than architectural expansion by default.

本檔對 state / 數字 / 進度為零權威。執行鎖見 [docs/TODO.md](TODO.md)；研究對象狀態見 [claim_state_registry](research/contracts/claim_state_registry.md)；可引用數字見 [evidence_ledger](research/evidence_ledger.md)；負結果與 revival 見 [no_go_registry](reference/no_go_registry.md)。

---

## Purpose / Scope

**管什麼**

- 專案級範式與能力邊界的單句總結（只鏡射 owner，不重定義）
- execution-chartered 的 project-level 里程碑：意圖、完成語義、執行狀態指針（execution-chartered ≠ ADR Accepted）
- 長期戰略路徑：為何存在、語義上的進入/退出條件、證據指針

**不管什麼**

- live progress、完成百分比、WIP、戰略路徑的 live 狀態欄
- baseline、gate 門檻、或任何定量閾值（含複製 `AUC` / latency / identity-score 不等式）
- 模組任務、per-module 能力表、或任何新的分層 taxonomy
- research object state、NO-GO 過程細節、資產清單、generated status

**定量條件的唯一合法形式**：`satisfied according to <authoritative contract or decision pointer>`。本檔不定義、不複製門檻。

**更新觸發**（其餘一律不更新；ordinary implementation PR 不得同步本檔）：

1. 新 ADR Accepted，改變範式或能力邊界（同 PR 附 pointer）
2. project-level objective 的 **execution charter 被掛上 / 卸下 / 替換 / 語義重定義**（例如正式轉向下一條全域 workstream）。這不是 ADR Accepted。**milestone 完成本身不觸發**——完成狀態由 `docs/TODO.md` 的 sole active（以及未來的 generated status）表示
3. 長期戰略路徑的語義，或其進入/退出條件定義，改變

**事件紀錄**

- 隨 C0「專案戰略」home 落地而建立；其後只在上列觸發時追加。

---

## Established Capability Boundaries

每條只保留一句話 + owner pointer。邊界事實由 ADR / no_go / preset 合約定義；本檔不得發明新邊界。

| 邊界 | Owner |
|:--|:--|
| Production tracker 採 **geometry-first、ReID-free** 架構；production path 不依賴 appearance / ReID 基礎設施。 | [ADR 019](decisions/019-demote-reid-geometry-first-production-tracker.md) Decision；production rule 同檔 Consequences；preset 合約見 `check_headline_decision_contract` |
| ReID 降為 experimental / archive extension，不由 production preset 初始化或執行；特定 regime 仍保留研究價值。 | [ADR 019](decisions/019-demote-reid-geometry-first-production-tracker.md)（retain-as-research）；模組鎖見 [reid TODO](modules/reid/TODO.md) |
| 主線是 real-time-first、無 ReID 依賴的 MOT；新研究只有通過 ablation 與一致性檢查才合併回主線。B 線停損已結案，不以原 appearance 路線重開。 | [ADR 018](decisions/018-project-main-line-direction.md) §0 / §3 |
| 評測範圍是 MOT17 train / SDP 內部 7-seq；不宣稱 MOTChallenge server 成績。對外數字主張的誠實邊界（in-sample vs tracker-delta）由展示敘事持有。 | [ADR 018](decisions/018-project-main-line-direction.md) §1；[PROJECT_SHOWCASE](PROJECT_SHOWCASE.md) Limitations |
| 現行穩定 GO 組合與 appearance 牆的教訓，以負結果登記表為準，不在本檔複述。 | [no_go_registry](reference/no_go_registry.md) Reusable Lessons / Current Stable GO Counterparts；[PIPELINE](PIPELINE.md) 結構性鐵律（結論指針，非數字） |
| 目前穩定的系統形狀是 MOT17-centered evaluation path，不是完整產品型多服務拓樸。 | [architecture/README](architecture/README.md) §3 |

---

## Near-term Strategic Objectives

此處列出 **execution-chartered** 或僅定義的 project-level 里程碑。本檔不複製它們是否正在執行。execution-chartered 只表示 `docs/TODO.md` 掛了執行授權，**不是**對應 ADR 已 Accepted。

### 1. 資產身分層（ADR 021 W-A）— execution-chartered

- **Intent**：讓實驗產物可回溯到 commit / preset / host / producer，使「這個數字哪來的」有機械答案。
- **Exit semantics**：satisfied according to [ADR 021](decisions/021-asset-provenance-and-progress-reporting.md) W-A Exit criteria，含同檔 §4.3 named limit。本檔不重述條文。
- **Execution-status pointer**：[docs/TODO.md § Sole active](TODO.md) 是 charter。ADR 021 仍為 Proposed。本檔不複製該行，也不因該行完成而更新。

### 2. 生成式進度報告（ADR 021 W-C）— defined, not execution-chartered

- **Intent**：進度呈現面必須是既有 fact-owner 的生成投影，而不是手寫 living status。
- **Exit semantics**：satisfied according to [ADR 021](decisions/021-asset-provenance-and-progress-reporting.md) W-C Exit criteria 與硬約束（只讀、不放數字、link-don't-relabel）。
- **Execution-status pointer**：是否掛上全域 execution charter，見 [docs/TODO.md § Sole active](TODO.md)。**W-A 的 execution charter 被替換或卸下才更新本檔**；W-C 尚未 execution-chartered。

不列為本節目標的例子：單模組 bugfix、perf 修復、尚未 execution-chartered 的 owner 決策候選（見下節 Runtime-grounded 路徑）。

---

## Long-term Strategic Paths

本節是可並存的長期軸向（研究、產品化、runtime substrate、publication/release），不是互斥分支。只定義路徑語義與指針，**不持有** live 狀態欄；狀態跟隨各 decision pointer。

### Appearance / ReID revival

- **Rationale**：geometry-first 是現行 production 範式；ReID 在特定 regime（長遮擋、低幀率、跨鏡、鏡頭切斷、弱幾何）仍可能有研究價值。
- **Semantic entry**：出現**新的** appearance 模態或訊號源，且通過既有 revival / unlock 規則——不是重放已結案的 MOT17 appearance 路線。
- **Semantic exit**：satisfied according to 相關 [no_go_registry](reference/no_go_registry.md) revival rule；模組程序閘見 [reid TODO](modules/reid/TODO.md) parked unlock。
- **Evidence**：[ADR 019](decisions/019-demote-reid-geometry-first-production-tracker.md) retain-as-research；[ADR 018](decisions/018-project-main-line-direction.md) §3 B 線停損；no_go appearance lesson。

### Industrial / multi-stream path

- **Rationale**：streaming / storage / cognition / resource 的模組家已在，但現行系統形狀明確不是完整產品拓樸。
- **Semantic entry**：owner 決定回到工業/部署線，並把它掛上 project-level execution charter，而不是單一模組 runbook 工作。
- **Semantic exit**：該 execution charter 被卸下或替換；部署細節仍歸各模組 runbook。
- **Evidence**：[architecture/README](architecture/README.md) §3；[module_objective_map](ownership/module_objective_map.md) 外圍模組 Primary；[ADR 018](decisions/018-project-main-line-direction.md) §7 release 兩層。

### Runtime-grounded capability

- **Rationale**：部分 research 路線（B1 / O1 等）需要 runtime substrate / guarantee 才可解鎖；缺 substrate 時不得假裝可編排。
- **Semantic entry**：owner 將「H0 re-entry + actual baseline capture」或等價的 substrate 設計掛上 project-level execution charter。**目前不是本檔的近程目標。**
- **Semantic exit**：satisfied according to 該層 research contracts 與 registry 的 admissible / gate 規則；本檔不複製閘值。
- **Evidence**：[claim_state_registry](research/contracts/claim_state_registry.md) §7 架構缺口、§8 候選集；[threads/README](research/threads/README.md) Current transition panel（導航投影，owner wins）。

### Dual paper / release track

- **Rationale**：決策/production 敘事與 method-paper 素材是兩條互指、不互相覆寫的線；研究倉與 release/demo 倉是兩層發布策略。
- **Semantic entry**：owner 將論文凍結或 release 切分掛上 project-level execution charter。
- **Semantic exit**：該 execution charter 被卸下或替換；表圖與數字仍歸 `report_data` / ledger。
- **Evidence**：[ADR 018](decisions/018-project-main-line-direction.md) §6–§7；[Doc Structure C5 dual paper lines](ownership/doc_structure_contract.md#c5--evidence--promotion)。
