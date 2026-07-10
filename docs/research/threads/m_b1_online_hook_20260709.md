---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# m_b1 online hook thread

> **One-line:** **Stage 1 CLOSED**. S2 through **Q4.5 = isolated_safe_points_only**  
> (`q45_atlas_terminal: B` · **154** productive-safe · 0 region candidates · evaluator **v4**).  
> Threshold/hook-policy promotion blocked. Next preferred: ranking / assignment-relative. Preset unchanged.

## Status (honest split)

| Milestone | Status |
|:--|:--|
| Offline M-B1 freeze | **LOO_pass_region_candidate** (offline only) |
| **Stage 1 overall** | **CLOSED** |
| Frozen-policy online relevance | **NULL** (support mismatch; rejected=0) |
| Stage 1b action-path P/F | **PASSED** |
| Online B-audit | **PASSED** — 244 rows; recon ok |
| Stage 2 Q1–Q3 | **PASSED / PASSED / SUFFICIENT** (23 safe-removable) |
| Stage 2 Q4 | **`q4_separability_grade: C`** (weak/unstable) · maps to **`stage2_entry_terminal_after_q4: B`** |
| Stage 2 Q4.5 atlas | **`q45_atlas_terminal: B`** isolated_safe_points_only — **154** productive-safe · 0 region candidates (v4) |
| Production preset | **unchanged** |

```text
Allowed claim:
  Stage 1 eng closure complete — observation + intervention substrate.
  Stage 2: multi-seq safe-negative mass exists; singleton thr inseparable;
  restricted thr×AND/OR atlas (v4) has only isolated productive-safe points
  (0 coordinate-union interior; 0 exact-absolute nested-LOSO portable).

Forbidden claim:
  frozen offline thr online-effective
  observed GT_hurt==0 atlas point = safe rule
  production preset / default-on
```

## Terminal letter namespaces (do not collapse)

| Field | Value | Rubric |
|:--|:--|:--|
| `q4_separability_grade` | **C** | Q4 effect-size / LOO / pure-neg tail grade (weak/unstable) |
| `stage2_entry_terminal_after_q4` | **B** | Entry-contract legal set: mass>0 but no stable separation → change family, not thr chase |
| `q45_atlas_terminal` | **B** | Q4.5 atlas taxonomy: `isolated_safe_points_only` |

Entry-contract legal Stage 2 terminals **A/B/C** are **not** the same alphabet as Q4 grades A–D.

## Read first (canonical)

1. [**Stage 2 D_online final**](../../modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md) — current terminal
2. [Stage 2 entry contract](../../modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md) — G0–G4
3. [Stage 1 online hook final](../../modules/semantic/research/m_b1_stage1_online_hook_final_20260710.md)
4. [Hook ABI contract](../../modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md)
5. [Offline research history](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md)
6. [Freeze candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
7. [Two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md)
8. [Doc consolidation report](../../modules/semantic/research/m_b1_doc_consolidation_report_20260710.md)
9. [ledger](../eval/signal_analysis_ledger.md)
10. Q4.5 evidence pack: [`docs/.../evidence/m_b1_5_stage2_q45_20260710/`](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/)

## Artifacts

```text
out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/          # offline freeze
out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/   # Stage 1 CLOSED
  hook_candidate_events.{csv,parquet}   # D_online 244
out/signal_study/m_b1_hook_ab_20260710T064657Z_stage1b/       # P/F controls
out/signal_study/m_b1_5_stage2_q1q3_20260710/
out/signal_study/m_b1_5_stage2_q4_20260710/
out/signal_study/m_b1_5_stage2_q45_20260710/                  # v4 evaluator HEAD
docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/
```

## Current step

```text
DONE:
  offline freeze + Stage 1 CLOSED + Stage 2 Q1–Q4.5 (evaluator v4)
NEXT (authorized):
  ranking / assignment-relative decision modeling
  (requires valid assignment-group key; frame provenance currently invalid)
  optional atlas thickness diagnostics
FORBIDDEN:
  thr-as-rule / hook policy / production preset from isolated safe points
```

## History

- 2026-07-09: offline candidate + thread
- 2026-07-10: Stage 1 wire/e2e; review split 1a/1b; Stage 1 CLOSED (B-audit)
- 2026-07-10: Stage 2 entry + Q1–Q3 SUFFICIENT + Q4 grade C + Q4.5 atlas B
- 2026-07-10: Q4.5 evaluator v2–v4 (nested LOSO, coord-union interior, exact-absolute portable)
- 2026-07-10: **doc consolidation** → 3 canonicals + retained contracts (rebase onto v4)
