# Phase 2 Activation Boundary — Module Doc Question Map

**中文：** `docs/modules/` 問題導向重整（#368）Phase 2 啟動邊界——這是閘門，不是授權  
**Status:** gate only — states the boundary a Phase 2 activation must satisfy and contain; it neither performs an activation nor records one. Live WIP state is owned by [`docs/TODO.md`](../TODO.md).  
**Issue:** [#368](https://github.com/raylei50653/saccade/issues/368) — Phase 2 direction 由 issue body 擁有，本檔**不**重述  
**Provenance:** materialized 2026-09-13 from the repo owner's Phase 2 activation-boundary statement, delivered in the working session that followed the Phase 1 close-out (`e5688742`). The owner's statement is the authority; this file is its materialization, not a re-derivation.  
**Home:** `docs/ownership/` — like the [Phase 1 protocol](module_doc_question_protocol.md), this governs *how the module doc surface may change*, not what any module currently answers. Filing it under `docs/modules/` would create exactly the extra first-class module document that #368 acceptance criterion 5 exists to prevent (C0.1).  
**Related:** [Phase 1 protocol](module_doc_question_protocol.md) · [question map](module_doc_question_map.yaml) · [doc_structure_contract.md](doc_structure_contract.md) (§ C5.1, § C6) · [DOC_MAINTENANCE.md](../DOC_MAINTENANCE.md) · [ownership README](README.md)

**Purpose:** draw the line between *describing what each document answers* (Phase 1) and *deciding which of those answers is the canonical answer a reader sees now* (Phase 2), so the second happens only under explicit repo-owner authorization and only inside a frozen scope.

**Not this file:** it is neither the authorization, nor the Phase 2 scope, nor a completion record. The authorization is a separate owner act; the live WIP state is owned by the Sole active register in [`docs/TODO.md`](../TODO.md) and is never restated here (C5.1).

---

## Why a gate exists

Phase 2 does not open because Phase 1 completed, because [#403](https://github.com/raylei50653/saccade/pull/403) merged, because the question map is available, or because [#404](https://github.com/raylei50653/saccade/issues/404) exists. **None of those is an authorization, and none may be read as one.**

Activation is a category change, not a continuation. The work crosses from "describe what each document answers" to "decide which answers become the current canonical reading surface". That crossing is granted by the repo owner explicitly and is recorded in the Sole active register — it is never inferred from upstream completion.

---

## Preconditions — all seven, before any synthesis

| # | Precondition |
|:--|:--|
| 1 | Phase 1 is **accepted**, and [`module_doc_question_map.yaml`](module_doc_question_map.yaml) is **accepted as the frozen input** for this synthesis round. |
| 2 | The Phase 2 scope is **explicitly frozen**: it names the module(s), durable questions, or synthesis unit to be handled. Refactoring all 123 documents in one pass is **not** a default and may not be assumed. |
| 3 | The **current-answer / authority decision rules** are explicitly designated. Phase 1 outputs — lifecycle marker, question ledger, taxonomy classification — are **inputs only**; none of them may be taken directly as a supersession finding or as an accepted-state conclusion. |
| 4 | Questions of **state / supersession / authority that are registry-owned remain with their existing owner**. Phase 2 may raise candidates that need adjudication; it may not rewrite fact ownership itself. |
| 5 | **#404's evidence / artifact governance stays an external boundary.** Phase 2 may create `canonical answer → evidence` pointers, but it must not begin moving, deleting, reordering, or re-governing `semantic/research/evidence/`. |
| 6 | This round's **completion conditions and output shape** are defined — for example `durable question → current answer → limits / unresolved → evidence pointers` — and the **READMEs / canonical docs that may be edited are named**. |
| 7 | [`docs/TODO.md`](../TODO.md) **explicitly sets that Phase 2 unit as Sole active**. Without this step, synthesis must not begin. |

Preconditions 1–6 are properties the authorization must carry; precondition 7 is the mechanical switch. An authorization that leaves any of the seven unset is not an activation.

---

## Once activated, this is authorized

- Merge knowledge from multiple historical research documents into one durable answer.
- Decide which content belongs on the canonical reading surface.
- Downgrade historical tasks / reports to provenance or evidence pointers.
- Rewrite a module README so it navigates by reader question.
- Raise supersession / state adjudication requests and hand them to the correct owner.

## Activation still does not authorize

- Re-adjudicating research verdicts.
- Modifying accepted state owned by other registries, threads, or markers.
- Evidence / artifact retention, layout, or disposal.
- Refactoring other modules unrelated to this round's Phase 2 scope.
- Automatically starting the next synthesis unit when this one completes.

Scope is a lease, not a slope: a named unit is what was authorized, and finishing it arms nothing.

---

## How activation is recorded

The Sole active register in [`docs/TODO.md`](../TODO.md) is the only surface that says a Phase 2 unit is active. This file is read as the gate that entry must satisfy; it is never itself evidence that Phase 2 started. Completing one unit returns the register to idle, leaving Phase 2 in backlog until the owner authorizes the next unit.

---

## One-liner

> **Phase 1 告訴我們「每份文件回答什麼」；Phase 2 activation boundary 則是明確授權開始決定「這些答案中，哪一些應成為讀者現在看到的 canonical answer」。**

Phase 1 tells us *what each document answers*; the Phase 2 activation boundary is the explicit authorization to start deciding *which of those answers should be the canonical answer a reader sees now*.
