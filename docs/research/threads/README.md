# Research threads（navigation-only）

**定位：** 跨子類連續任務的**導航卡 / 母線**，不是新的事實家。

```text
DEVELOPMENT.md       = stable global router (D0–D4 · dashboard)
TODO.md              = WIP lock / active pointer（sole active 一句）
threads/             = 連續任務母線（狀態 · 邊界 · 下一步 · normative links）
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

## When to create a thread

```text
跨 2 個以上文檔家 / 子類
或連續 3 步以上
或會產生可引用數字 / policy / hook / audit
→ 建 thread
```

**不建：** 單次 bug fix、單次 ablation、一步能在 module research 結案的工作。

---

## Active threads

| Thread | Status (one-line) | Owner |
|:--|:--|:--|
| [gt_support_morphology_20260711.md](gt_support_morphology_20260711.md) | **semantic sole active** · PR-A/B sealed · PR-C forensic `ROLE_REVERSAL_SUPPORTED`（3 TRUE + 1 UNRESOLVED；MOT17-10 bound）· next PR-D partial-order audit | semantic |
| [safe_region_assetization_20260710.md](safe_region_assetization_20260710.md) | **A1 CLOSED**（`A1_ACCEPTED_WITH_LIMITS`, maturity A1）· R2–R4 fail-closed · mainline handed off → gt_support_morphology · [PR #95](https://github.com/raylei50653/saccade/pull/95)/[#97](https://github.com/raylei50653/saccade/pull/97) history | semantic |
| [association_recovery_registry_20260709.md](association_recovery_registry_20260709.md) | Scripts index + tools YAML + contracts 就位；registry 維護母線 | semantic |
| [doc_structure_o15_followup_20260709.md](doc_structure_o15_followup_20260709.md) | O1.5 + TODO-as-WIP-lock；follow-up = index debt / optional strict | ownership |

## Parked threads

| Thread | Status (one-line) | Owner |
|:--|:--|:--|
| [occ_exit_audit_20260709.md](occ_exit_audit_20260709.md) | **PARKED** · WP1–WP3 complete · future RegionAsset producer/intervention consumer after assetization gate | semantic |

## Closed threads (nav only)

| Thread | Status (one-line) | Owner |
|:--|:--|:--|
| [composition_grammar_coverage_program_20260710.md](composition_grammar_coverage_program_20260710.md) | **SUPERSEDED** · coverage map absorbed into assetization R2–R4 · no C1–C6 execution | semantic |
| [composition_grammar_safe_region.md](composition_grammar_safe_region.md) | **CLOSED A0 baseline** · T0-A/B/R1 · radius≥1=0 · terminal B · handoff → assetization | semantic |
| [m_b1_online_hook_20260709.md](m_b1_online_hook_20260709.md) | **CLOSED** · S1+S2 Q4.5 B complete · handoff → composition T0/A0 · ranking deferred | semantic |

---

## File shape（每個 thread 只放這些）

```md
---
doc-status: active
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

**命名：** `<topic>_<YYYYMMDD>.md`（母線主題，不是每步一個檔）。  
**Same-PR：** 新增 thread → 本 README Active 表加一行。

---

## Role split

| 層 | 負責 |
|:--|:--|
| [DEVELOPMENT.md](../../../DEVELOPMENT.md) | 分層路由 D0–D4 · dashboard 只鏡射 sole active one-liner · PR-driven research routing |
| module `TODO.md` | **WIP 鎖**（sole active + links）；非任務敘事 |
| module / research README | 局部入口與檔案索引 |
| **threads/** | 連續任務母線（狀態 · 邊界 · 研究 gate · next step） |
| **implementation PR** | engineering delivery：branch tips、SHA、CI、changed files、review findings |
| research contracts / notes | normative inputs · claim boundaries · research acceptance · stage authorization |
| ledger / report_data / out/ | 事實與數字升格 |

```text
PR merge ≠ research acceptance
engineering-ready ≠ evidence promotion
```
