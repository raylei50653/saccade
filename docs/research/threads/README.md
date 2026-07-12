# Research threads（navigation-only）

**定位：** 跨子類連續任務的**導航卡 / 母線**，不是新的事實家。

```text
DEVELOPMENT.md       = stable global router (D0–D4 · dashboard)
TODO.md              = WIP lock / active pointer（sole active 一句）
threads/             = 進行中母線（proposed · active · parked）
threads/closed/       = 已結案 thread 檔案家（歷史導航；非 evidence）
implementation PR    = engineering delivery authority (head/base/SHA/CI/files)
module research      = 事實與分析
ledger/report_data/no_go = 升格後的正式事實
```

```text
DEVELOPMENT.md → module TODO sole-active → active thread / contract → PR
```

**不做：**

- 不放長表、完整分析、可引用數字正文
- 不取代 `evidence_ledger` / `signal_analysis_ledger` / module research / `report_data`
- **不**取代 module TODO 的 WIP=1 鎖（thread 可多張；sole active 仍以 TODO 為準）
- **不**使用 direct-agent `*.dispatch.yaml` sidecars（retired; do not recreate）

**結構契約：** [../../ownership/doc_structure_contract.md](../../ownership/doc_structure_contract.md)（O1.5）  
**薄入口：** [../../../DEVELOPMENT.md](../../../DEVELOPMENT.md)

---

## Lifecycle（thread 狀態機）

狀態以 frontmatter `doc-status` + 本 README 分表為準。  
**進行中**卡在 `threads/`；**結案**卡移入 [`closed/`](closed/)。關閉 ≠ 刪檔。  
只有不再需要 threads 導航的 one-shot 才進 `docs/archive/`。

```text
proposed  →  任務已成文、未開跑 / 未授權 sole-active     →  threads/
active    →  進行中（可多張；sole active 以 module TODO） →  threads/
parked    →  有意暫停；不佔 WIP；可再激活                 →  threads/
closed    →  結案；terminal + handoff 導航               →  threads/closed/
archived  →  不再當現況導航                               →  docs/archive/
```

| status | 本 README 表 | 檔案位置 | frontmatter |
|:--|:--|:--|:--|
| `proposed` | **Proposed** | `threads/` | `doc-status: proposed` |
| `active` | **Active** | `threads/` | `doc-status: active` |
| `parked` | **Parked** | `threads/` | `doc-status: parked` |
| `closed` | **Closed** | **`threads/closed/`** | `doc-status: closed` + `closed: YYYY-MM-DD` |
| `archived` | Archive 附註或移除 | `docs/archive/` | `doc-status: archived` |

**歷史狀態放哪：**

| 層 | 放什麼 |
|:--|:--|
| 單張 thread 的 `## History` | 逐步時間線（開線、PR、gate、handoff） |
| **`threads/closed/<card>.md`** | 結案卡本體（Final status · terminal · History） |
| **本 README Closed 表** | 全域索引：terminal one-liner + close date |
| module research / evidence / ledger | 可引用事實（thread **不**當第二真相） |

---

## When to create a thread

```text
跨 2 個以上文檔家 / 子類
或連續 3 步以上
或會產生可引用數字 / policy / hook / audit
→ 建 thread
```

**不建：** 單次 bug fix、單次 ablation、一步能在 module research 結案的工作。

**Same-PR：** 新增 / 改狀態 thread → 更新本 README 對應表（Proposed / Active / Parked / Closed）。

---

## How to close a thread

關閉時 **同一變更** 做齊：

1. **frontmatter**
   - `doc-status: closed`
   - `closed: YYYY-MM-DD`
   - 可選：`closed-verdict: <token>`（如 `A1_ACCEPTED_WITH_LIMITS` · `V5` · `SUPERSEDED`）
2. **thread body**
   - 頂部 one-liner 標 **CLOSED** / **SUPERSEDED** / **PARKED→CLOSED**
   - 補或改寫 `## Final status`（或等價 terminal 表）：verdict · handoff · preset 未改 · 不再授權 next work
   - `## History` 末行記關閉事件 + 指向事實 note / PR / packet
   - `## Current step` 改為 **none — closed**（勿留假 next step）
3. **移動檔案**
   - `git mv docs/research/threads/<card>.md docs/research/threads/closed/<card>.md`
   - 修正相對連結深度（`../eval` → `../../eval`；`../../modules` → `../../../modules`；進行中 sibling → `../<active>.md`）
   - 全庫把指向舊路徑的 link 改成 `threads/closed/<card>.md`
4. **本 README**
   - 從 Proposed / Active / Parked **移出**
   - 加入 **Closed** 表一行：連結 `closed/<card>.md` · terminal one-liner · close date
5. **下游**
   - module `TODO.md` sole-active 若仍指此 thread → 改指 handoff 目標或清掉
   - 若有 superseding thread → 雙方交叉連結
6. **不要**
   - 刪 thread 檔（要用 `archived` + `docs/archive/`）
   - 把結案長表貼進 thread（寫進 module research / ledger）
   - 結案卡繼續留在 `threads/` 根目錄

**Park（非關閉）：** 檔案**仍在** `threads/`；`doc-status: parked` + 本 README Parked 表；History 記 pause reason + 再開條件。

**Archive（很少用）：** 不再需要 threads 導航時 → `docs/archive/` + 本 README Closed 改為 archive 指標或移除。

---

## Proposed threads

目前無 proposed thread。

## Active threads

| Thread | Status (one-line) | Owner |
|:--|:--|:--|
| [gap_conditioned_probabilistic_motion_probe_20260711.md](gap_conditioned_probabilistic_motion_probe_20260711.md) | Phase B **`V5 ACCEPTED_WITH_LIMITS`** recorded · D0 fail-closed pending #112 · E0–E2 `ACCEPTED_WITH_LIMITS` · E3 `E3_SIGNALS_SEALED` · representation / level 1 only | semantic |
| [association_recovery_registry_20260709.md](association_recovery_registry_20260709.md) | Scripts index + tools YAML + contracts 就位；registry 維護母線 | semantic |
| [doc_structure_o15_followup_20260709.md](doc_structure_o15_followup_20260709.md) | O1.5 + TODO-as-WIP-lock；follow-up = index debt / optional strict | ownership |

## Parked threads

| Thread | Status (one-line) | Owner |
|:--|:--|:--|
| [gt_support_morphology_20260711.md](gt_support_morphology_20260711.md) | **PARKED after PR-D** · #107 `ACCEPTED_WITH_LIMITS` boundary preserved · restricted-closure prototype not started · resume after semantic owner gate | semantic |
| [occ_exit_audit_20260709.md](occ_exit_audit_20260709.md) | **PARKED** · WP1–WP3 complete · future RegionAsset producer/intervention consumer after assetization gate | semantic |

## Closed threads（檔案在 [`closed/`](closed/)）

| Thread | Closed | Terminal (one-line) | Owner |
|:--|:--|:--|:--|
| [ambiguous_band_ranking_power_probe_20260712.md](closed/ambiguous_band_ranking_power_probe_20260712.md) | 2026-07-12 | **T2 `NO_USABLE_RANKING_POWER_IN_CLASS` ACCEPTED**（12-member class-scoped；step ⑤ 生效；step ④ 未開）· [PR #135](https://github.com/raylei50653/saccade/pull/135) seal／[PR #136](https://github.com/raylei50653/saccade/pull/136) acceptance | semantic |
| [safe_region_assetization_20260710.md](closed/safe_region_assetization_20260710.md) | 2026-07-11 | **A1 CLOSED**（`A1_ACCEPTED_WITH_LIMITS`）· R2–R4 fail-closed · handoff → [gt_support_morphology](gt_support_morphology_20260711.md) · [PR #95](https://github.com/raylei50653/saccade/pull/95)/[#97](https://github.com/raylei50653/saccade/pull/97) | semantic |
| [composition_grammar_coverage_program_20260710.md](closed/composition_grammar_coverage_program_20260710.md) | 2026-07-10 | **SUPERSEDED** · coverage map absorbed into assetization R2–R4 · no C1–C6 execution | semantic |
| [composition_grammar_safe_region.md](closed/composition_grammar_safe_region.md) | 2026-07-10 | **CLOSED A0 baseline** · T0-A/B/R1 · radius≥1=0 · terminal B · handoff → assetization | semantic |
| [m_b1_online_hook_20260709.md](closed/m_b1_online_hook_20260709.md) | 2026-07-10 | **CLOSED** · S1+S2 Q4.5 B complete · handoff → composition T0/A0 · ranking deferred | semantic |

> 目錄索引見 [closed/README.md](closed/README.md)。數字與 claim 在 module research / ledger。

---

## File shape

### Active / proposed / parked

```md
---
doc-status: proposed | active | parked
doc-promotion: navigation-only; not evidence
owner-module: <module|cross|ownership>
created: YYYY-MM-DD
---

# <short title>

## Status
## Current boundary
## Read first
## Artifacts
## Current step
## Acceptance
## Must not
## History
```

### Closed（結案形）

```md
---
doc-status: closed
doc-promotion: navigation-only; not evidence
owner-module: <module|cross|ownership>
created: YYYY-MM-DD
closed: YYYY-MM-DD
closed-verdict: <optional token>
---

# <short title>

> **One-line (CLOSED):** …

## Final status          # terminal table · handoff · preset unchanged
## Read first            # 結案後仍需要的入口
## Final evidence state  # optional；只放 pointer / 一句，不放長表
## Must not              # 關閉後仍禁止的 claim
## History               # 含關閉事件
```

閉合後可刪或折疊 `Current step` / `Acceptance` 中未完成 checklist；**不可**刪掉使 handoff 斷鏈的 Read first / History。

**命名：** `<topic>_<YYYYMMDD>.md`（母線主題，不是每步一個檔）。

---

## Role split

| 層 | 負責 |
|:--|:--|
| [DEVELOPMENT.md](../../../DEVELOPMENT.md) | 分層路由 D0–D4 · dashboard 只鏡射 sole active one-liner · PR-driven research routing |
| module `TODO.md` | **WIP 鎖**（sole active + links）；非任務敘事 |
| module / research README | 局部入口與檔案索引 |
| **threads/** | 進行中母線（proposed · active · parked） |
| **threads/closed/** | 已結案 thread 檔案家；本 README Closed 表 = 索引 |
| **implementation PR** | engineering delivery：branch tips、SHA、CI、changed files、review findings |
| research contracts / notes | normative inputs · claim boundaries · research acceptance · stage authorization |
| ledger / report_data / out/ | 事實與數字升格 |

```text
PR merge ≠ research acceptance
engineering-ready ≠ evidence promotion
thread closed ≠ evidence promoted
```
