<!-- doc-status: active -->
<!-- doc-promotion: navigation-only; not evidence -->

# Experiment workspaces（實驗工作區 · prototype）

> **非規範(non-normative)。** 這是一個**形狀原型**,不是 repo-wide canonical layout。目前只有一個樣例,用來觀察「一個實驗=一個資料夾」的物理結構是否比平面 research 目錄好用。**canonical home routing 未改**:實驗的權威歸屬仍照 [`doc_structure_contract.md`](../ownership/doc_structure_contract.md)(single-module 實驗仍屬 `docs/modules/<m>/research/`;Cheb-GR / offline identity / occ-exit 的 home 仍是 `docs/modules/semantic/`)。是否升格為正式 layout,要另開 architecture PR 決定。

## 這個原型在試什麼

- 一個實驗聚成一個 `<實驗名>/` 資料夾 + 一份 `README.md` 導航入口,避免平面 research 目錄繼續膨脹。
- entry `README.md` **只導航**:概況 + 成員連結 + 指回 owner(thread / registry)。**不複寫**任何 state(verdict / 數字 / disposition / object-rung 各有 owner,見樣例)。

## 樣例

[`occ_exit_audit_p55/`](occ_exit_audit_p55/README.md) — 首個(也是目前唯一)工作區。docs-only、fully reversible;檔案物理搬到這裡是原型觀察,不代表 routing 已改。

## 尚未決定(要 architecture PR)

workspace ↔ decision-object 的基數(1:1?1:N?有些只有 evidence/no-go 而無 registry object)、entry 是否可投影任何 state、以及 fail-closed 一致性檢查——**都還沒定案**,所以本原型一律**不宣告** `object` / `evidence` 等會與 owner 打架的欄位。
