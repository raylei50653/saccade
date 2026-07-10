---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# m_b1 online hook thread

> **One-line:** Offline CLOSED. Stage 1 hook **wired + e2e safe (vacuous online thr)**. Preset unchanged. Online B-audit still pending.

## Status

- **M-B1 offline gate / safe-region research = CLOSED**
- Offline: `LOO_pass_region_candidate` · offline smoke pass (GT0, FP=8721)
- Online: **hook wired (default-off)** · e2e A1/B **done**
- `e2e_safe_for_default_off`: **yes**
- Classification: `online_effect_neutral_but_safe__vacuous_online_thr`
- Production: **not** preset · **not** default-on
- Stage 1 plan: [two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md)
- Wire note: [m_b1_hook_stage1_wire_20260710.md](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md)
- E2e note: [m_b1_hook_stage1_e2e_20260710.md](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md)

```text
candidate_id: m_b1_repaired_eps0_loo_pass_20260709
  = LOO_pass_region_candidate
  = offline_smoke_pass (GT0, FP=8721)
  = online_hook_wired__e2e_safe_vacuous
  = e2e_safe_for_default_off: yes
  ≠ production preset
  ≠ online reject power under prod bridge_px/height gates

online_hook:              wired_default_off
e2e_validation:           done (A1≡B, Δ=0)
online_hook_eligible:     244
online_hook_rejected:     0
e2e_safe_for_default_off: yes
production_preset:        unchanged
```

## Current boundary

Research-only default-off online hook is **code-wired and e2e-safe**.  
Frozen offline thr is **vacuous** on the online baseline-ok universe (`bridge_px=0.4`, height gate) — safe null effect, not a production rejecter.

```text
e2e A1≡B; online thr vacuous vs prod gates
  (not: online/e2e not wired)

portable policy:
  out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json

policy shape:
  OR of 5 singleton tail_q85 atoms
  no zone / no gap
  ban_gap + ban_zone
```

## Read first

1. [e2e result](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md) — **start here for Stage 1 close**
2. [wire status](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md)
3. [portable OR-tail hook contract](../../modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md)
4. [two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md)
5. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
6. [signal_analysis_ledger](../eval/signal_analysis_ledger.md) §5

Also: [DEVELOPMENT.md](../../../DEVELOPMENT.md) D1/D3 · [semantic TODO](../../modules/semantic/TODO.md)

## Artifacts

**Canonical e2e study**

```text
out/signal_study/m_b1_hook_ab_20260710T062345Z/
```

**Contract path (required):**

- `out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json`

**Ledger ids:** `m.gate.repaired_b2e2e_smoke` · `m.gate.portable_or_tail_hook`

## Current step

```text
DONE:
  offline freeze + pairs event table (n_rejected=8721)
  Stage 1 code wire (loader/CLI/CUDA thr inject)
  A1/B e2e 7-seq + metrics parse + native counters
  e2e_safe_for_default_off: yes
  classification: online_effect_neutral_but_safe__vacuous_online_thr

PENDING (optional Stage 1 depth / not blocking safe headline):
  online full B-audit event table (CLI still fail-closed)
  decision-change joins (N/A while rejected=0)

STOP Stage 1 eng for preset claims.
DO NOT start Stage 2 thr re-fit in same PR.
If thr re-fit on online universe is desired later → separate Stage 2 PR only.
```

Runner:

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp> \
  --run-e2e
```

## Acceptance

**Headline (met):**

```text
e2e_safe_for_default_off: yes
```

**Supporting (met for minimum eng milestone):**

```text
A1 ≡ B metrics (Δ=0)
A1 ≡ B MOT result hashes
online counters: eligible=244 rejected=0 (B); eligible=0 (A1)
A0 soft identity: 6/7 seq + B2 metrics match
```

**Still open for full artifact freeze:** online B-audit full table (not required for safe/null classification).

## Must not

- search / repair / learned weights
- zone/gap atoms / runtime refit
- production preset change / silent default-on
- re-open rule search
- claim online reject power from offline FP=8721
- freeze thr re-fit as Stage 1 “fix”
- Stage 2 in the same PR
- weaken tests

## Status churn only

When results land, update **only**:

1. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
2. [signal_analysis_ledger](../eval/signal_analysis_ledger.md)
3. [e2e note](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md) / wire note

Plus this thread’s Status / Current step / History.

## History

- 2026-07-09: offline research candidate established (`m_b1_repaired_eps0_loo_pass_20260709`)
- 2026-07-09: offline smoke pass · online_blocked（correct boundary *at that time*）
- 2026-07-09: offline phase CLOSED; thread opened with session hook consolidated
- 2026-07-10: two-stage plan landed; Stage 1 eng starts
- 2026-07-10: Stage 1 wire — loader/CLI/CUDA thr inject/`run_m_b1_hook_ab.py`; offline pairs replay n_rejected=8721
- 2026-07-10: status reconciliation — `online_hook_wired__e2e_pending`
- 2026-07-10: **e2e A1/B 7-seq** study `m_b1_hook_ab_20260710T062345Z` — A1≡B Δ=0; online eligible=244 rejected=0; thr vacuous vs `bridge_px=0.4`/height gates; **e2e_safe_for_default_off=yes**
