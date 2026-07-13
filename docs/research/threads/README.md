# Research threads（navigation-only）

**定位：** 跨子類連續任務的**導航卡 / 母線**，不是新的事實家。

```text
DEVELOPMENT.md       = stable global action router (D0–D4)
TODO.md              = mainline-charter WIP lock / pointer（sole active 一句）
threads/             = 進行中 charter（proposed · active · parked）
Expected state       = charter 內可替換的 planning lease（非 accepted state）
Current step / PR    = fast probes（可停止、替換、丟棄）
threads/closed/       = 已結案 thread 檔案家（歷史導航；非 evidence）
implementation PR    = engineering delivery authority (head/base/SHA/CI/files)
module research      = 事實與分析
ledger/report_data/no_go = 升格後的正式事實
```

```text
DEVELOPMENT.md → module TODO sole-active → active thread / contract → PR
```

## Mainline charter / expected state / probe

| 層 | 作用 | 更新代價 |
|:--|:--|:--|
| Mainline charter | 定義 decision question、boundary 與下一個 commit point；只有它可取得 `sole-active` | 中；改主線才 park / close / replace charter |
| Expected state lease | 寫「預計抵達什麼」與 disconfirm / discard condition | 低；可在同一 charter 內替換或刪除，不改 registry |
| Probe | `Current step` 或 implementation PR 中的執行嘗試 | 最低；可停止、替換、丟棄；無可重用 evidence 就不建 formal note |

WIP=1 **只綁 mainline charter**。工程 follow-up、資料補件、文件收尾與 probe 可同時以
`non-wip` 執行；它們只有在 owner 指定為新的 decision-changing charter 時才取得 WIP 鎖。

**不做：**

- 不放長表、完整分析、可引用數字正文
- 不取代 `evidence_ledger` / `signal_analysis_ledger` / module research / `report_data`
- **不**取代 module TODO 的 WIP=1 鎖（thread 可多張；sole active 仍以 TODO 為準）
- **不**使用 direct-agent `*.dispatch.yaml` sidecars（retired; do not recreate）

**結構契約：** [../../ownership/doc_structure_contract.md](../../ownership/doc_structure_contract.md)（O1.5）  
**行動入口：** [../../../DEVELOPMENT.md](../../../DEVELOPMENT.md#agent-action-cards)

---

## 統一工作／交接匯報

任何會改變 thread 狀態、WIP 指向或 handoff 的更新，先用下列四段
匯報；**先說現有工作怎麼收尾，再談是否另開新題**。

```text
全域 WIP／既有工作帳
- sole active：<以 module TODO 為準；可為 none>
- active but non-WIP：<工程 follow-up / 維護 / 治理卡>
- parked：<暫停中的既有母線>
- closed：<本輪剛結案的母線>

交接帳（只記 direct handoff disposition）
來源／terminal → direct receiver（或 no receiver） → 被交接的工作／class → 現在狀態

跨 thread consequence（非 handoff）
<terminal 對其他既有卡造成的 constraint / closure；既有卡仍留在全域工作帳>

本輪唯一具體工作
- thread：
- 工作：
- 完成條件：

不在本輪
- 未授權的新研究、擴 class、或候選方向；不得寫成既有 handoff 或 current step。
```

**判讀規則：**

- `closed` 的交接帳只記 terminal **直接**交出的工作或 class：交給 direct
  receiver，或 **no receiver / no continuation**；不能只寫抽象的「下一步」。
- 已存在於其他卡的工程債、maintenance、parked branch 一律留在「全域 WIP／
  既有工作帳」。terminal 對它們的影響只能寫為 **cross-thread consequence**，
  不得冒充為 handoff。
- `active` 不等於 sole-active。active 卡必須標出工作類別與 WIP 角色，
  讓工程收尾、維護、治理不會被誤讀為科學主線。
- 同一 charter 內替換 expected-state lease 或 probe 不是另開主線；不得因此更新 registry 或製造 handoff。
- `parked` 的既有分支只記 pause／resume 條件；它不是本輪可執行工作。
- 新研究候選只有在 owner 明示要排下一個 charter 時才另列；在此之前不進
  交接帳、Active 表或 `Current step`。

**卡片標記：** proposed／active／parked 卡都必須以 frontmatter 補充
`work-class: mainline-study | engineering-follow-up | maintenance | governance`
與 `wip-role: sole-active | non-wip | parked`。**proposed 一律必填
`work-class` 並使用 `wip-role: non-wip`，直到 owner activation；**不得在
proposed 階段使用 `sole-active` 或 `parked`。`TODO.md` 仍是 sole-active 的
唯一權威；這些標記只消除導航歧義。

---

## Lifecycle（thread 狀態機）

狀態以 frontmatter `doc-status` + 本 README 分表為準。  
**進行中**卡在 `threads/`；**結案**卡移入 [`closed/`](closed/)。關閉 ≠ 刪檔。  
只有不再需要 threads 導航的 one-shot 才進 `docs/archive/`。

```text
proposed  →  任務已成文、未開跑 / 未授權 sole-active；`wip-role: non-wip` →  threads/
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
沒有可重用 evidence 的 disposable probe 也不建 thread / research note；直接從 `Current step` 或 PR 移除即可。

**Same-PR：** 新增 / 改狀態 thread → 更新本 README 對應表（Proposed / Active / Parked / Closed）。

---

## How to close a thread

先完成 [通用研究收尾卡](../../../DEVELOPMENT.md#研究收尾卡)；本節只補 thread 特有的
frontmatter、搬移與索引要求。

只有整個 charter terminal 才關 thread。丟棄 probe 或替換 expected-state lease 不走此流程。

關閉時 **同一變更** 做齊：

1. **frontmatter**
   - `doc-status: closed`
   - `closed: YYYY-MM-DD`
   - 可選：`closed-verdict: <token>`（如 `A1_ACCEPTED_WITH_LIMITS` · `V5` · `SUPERSEDED`）
2. **thread body**
   - 頂部 one-liner 標 **CLOSED** / **SUPERSEDED** / **PARKED→CLOSED**
   - 補或改寫 `## Final status`（或等價 terminal 表）：verdict · **direct handoff（direct receiver 或 no receiver）** · cross-thread consequence（若有，須明標非 handoff）· preset 未改 · 不再授權 next work
   - `## History` 末行記關閉事件 + 指向事實 note / PR / packet
   - `## Current step` 改為 **none — closed**（勿留假 next step）
3. **移動檔案**
   - `git mv docs/research/threads/<card>.md docs/research/threads/closed/<card>.md`
   - 修正相對連結深度（`../eval` → `../../eval`；`../../modules` → `../../../modules`；進行中 sibling → `../<active>.md`）
   - 全庫把指向舊路徑的 link 改成 `threads/closed/<card>.md`
4. **本 README**
   - 從 Proposed / Active / Parked **移出**
   - 加入 **Closed** 表一行：連結 `closed/<card>.md` · terminal one-liner · **direct handoff disposition** · close date；cross-thread consequence 另欄記錄
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

| Thread | Work class / WIP role | Current concrete work / Intent | Owner |
|:--|:--|:--|:--|
| *(none)* | — | — | — |

## Active threads

| Thread | Work class / WIP role | Current concrete work | Owner |
|:--|:--|:--|:--|
| [gap_conditioned_probabilistic_motion_probe_20260711.md](gap_conditioned_probabilistic_motion_probe_20260711.md) | engineering follow-up · **non-WIP** | **Only:** D0 / #112 runtime CUDA capture；Phase B `V5` is already recorded and authorizes no new semantic mainline work | semantic |
| [association_recovery_registry_20260709.md](association_recovery_registry_20260709.md) | maintenance · **non-WIP** | Keep R/H ownership and path-health registry current | semantic |
| [doc_structure_o15_followup_20260709.md](doc_structure_o15_followup_20260709.md) | governance · **non-WIP** | Pay down research-index debt; optional structure checks remain non-blocking | ownership |

## Parked threads

| Thread | Work class / WIP role | Pause / resume boundary | Owner |
|:--|:--|:--|:--|
| [score_temporal_to_stable_domain_20260712.md](score_temporal_to_stable_domain_20260712.md) | mainline-study · **parked** | `R1_FAITHFUL` closed (it is what made runtime coordinates auditable); discrete-\(M\) follow-on **reclassified as a score-ranking feature, not a gate** → parked unsealed. | semantic |
| [gt_support_morphology_20260711.md](gt_support_morphology_20260711.md) | mainline-study · **parked** | Score continuation is closed for Door 0's tested class; gate direction remains parked pending explicit re-charter and WIP authorization | semantic |
| [occ_exit_audit_20260709.md](occ_exit_audit_20260709.md) | mainline-study · **parked** | WP1–WP3 complete; waits for a RegionAsset producer/intervention consumer after the assetization gate | semantic |

## Closed threads（檔案在 [`closed/`](closed/)）

| Thread | Closed | Terminal (one-line) | Direct handoff disposition | Cross-thread consequence (not handoff) | Owner |
|:--|:--|:--|:--|:--|:--|
| [runtime_faithful_safe_domain_20260712.md](closed/runtime_faithful_safe_domain_20260712.md) | 2026-07-13 | **S0 `S0_UNDECIDABLE` ACCEPTED** · V7 has no offline-safe grid point, so runtime transfer is not assessed · [PR #152](https://github.com/raylei50653/saccade/pull/152) | **no receiver / no continuation** — wider runtime join requires a new decision-relevance and O0 decision | offline partial-order state unchanged; runtime transfer unaccepted and closure remains inadmissible | semantic |
| [ambiguous_band_ranking_power_probe_20260712.md](closed/ambiguous_band_ranking_power_probe_20260712.md) | 2026-07-12 | **T2 `NO_USABLE_RANKING_POWER_IN_CLASS` ACCEPTED**（12-member class-scoped；step ⑤ 生效；step ④ 未開）· [PR #135](https://github.com/raylei50653/saccade/pull/135) seal／[PR #136](https://github.com/raylei50653/saccade/pull/136) acceptance | tested 12-member score class → **no receiver / no continuation** | morphology score branch is closed for this class; its gate branch remains parked in its own card | semantic |
| [safe_region_assetization_20260710.md](closed/safe_region_assetization_20260710.md) | 2026-07-11 | **A1 CLOSED**（`A1_ACCEPTED_WITH_LIMITS`）· R2–R4 fail-closed | → [gt-support morphology](gt_support_morphology_20260711.md) (parked) | — | semantic |
| [composition_grammar_coverage_program_20260710.md](closed/composition_grammar_coverage_program_20260710.md) | 2026-07-10 | **SUPERSEDED** · coverage map absorbed into assetization R2–R4 · no C1–C6 execution | → assetization closed line; **no further continuation** | — | semantic |
| [composition_grammar_safe_region.md](closed/composition_grammar_safe_region.md) | 2026-07-10 | **CLOSED A0 baseline** · T0-A/B/R1 · radius≥1=0 · terminal B | → assetization closed line | — | semantic |
| [m_b1_online_hook_20260709.md](closed/m_b1_online_hook_20260709.md) | 2026-07-10 | **CLOSED** · S1+S2 Q4.5 B complete | → composition T0/A0 closed line; ranking deferred, **no active receiver** | — | semantic |

> 目錄索引見 [closed/README.md](closed/README.md)。數字與 claim 在 module research / ledger。

---

## File shape

### Active / proposed / parked

```md
---
doc-status: proposed | active | parked
doc-promotion: navigation-only; not evidence
owner-module: <module|cross|ownership>
work-class: mainline-study | engineering-follow-up | maintenance | governance
wip-role: sole-active | non-wip | parked
created: YYYY-MM-DD
---

# <short title>

## Status                 # work class + WIP role; proposed=non-wip; sole-active links to module TODO
## Current boundary
## Expected state (lease) # target only;not accepted state;replaceable
## Commit point           # when owner reviews whether state changed
## Discard when           # disconfirm / expiry / replacement condition
## Read first
## Artifacts
## Current step           # disposable probe(s);may be replaced without close / registry update
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

## Final status          # terminal table · direct handoff (receiver or no receiver) · cross-thread consequence (if any) · preset unchanged
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
| [DEVELOPMENT.md](../../../DEVELOPMENT.md) | D0–D4 action cards；模組入口不承載 live status |
| module `TODO.md` | **mainline-charter WIP 鎖**（sole active + links）；不放 expected state / probe / 任務敘事 |
| module / research README | 局部入口與檔案索引 |
| **threads/** | 進行中母線（proposed · active · parked） |
| Expected state / Current step | charter 內 planning lease / probe；可替換，不是 accepted state |
| **threads/closed/** | 已結案 thread 檔案家；本 README Closed 表 = 索引 |
| **implementation PR** | engineering delivery：branch tips、SHA、CI、changed files、review findings |
| research contracts / notes | normative inputs · claim boundaries · research acceptance · stage authorization |
| ledger / report_data / out/ | 事實與數字升格 |

```text
PR merge ≠ research acceptance
engineering-ready ≠ evidence promotion
thread closed ≠ evidence promoted
```
