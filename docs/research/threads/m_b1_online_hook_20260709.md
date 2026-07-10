---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# m_b1 online hook thread

> **One-line:** Offline research CLOSED. Online hook **wired default-off** (PR #87). e2e validation still pending. Preset unchanged.

## Status

- **M-B1 offline gate / safe-region research = CLOSED**
- Offline: `LOO_pass_region_candidate` · offline smoke pass (GT0, FP=8721)
- Online: **hook wired (default-off)** · e2e A/B **not finished**
- `e2e_safe_for_default_off`: **no** (until A/B metrics land)
- Production: **not** preset · **not** default-on
- Stage 1 plan: [two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md)
- Wire note: [m_b1_hook_stage1_wire_20260710.md](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md)

```text
candidate_id: m_b1_repaired_eps0_loo_pass_20260709
  = LOO_pass_region_candidate
  = offline_smoke_pass (GT0, FP=8721)
  = online_hook_wired__e2e_pending
  ≠ e2e_safe_for_default_off
  ≠ production preset

online_hook:              wired_default_off
e2e_validation:           pending
e2e_safe_for_default_off: no
production_preset:        unchanged
```

## Current boundary

Research-only default-off online hook is **code-wired**. Offline candidate remains frozen. **e2e is not validated.**

```text
online hook wired; e2e not validated
  (not: online/e2e not wired)

portable policy:
  out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json

policy shape:
  OR of 5 singleton tail_q85 atoms
  no zone
  no gap
  ban_gap + ban_zone
```

## Read first

1. [m_b1 offline phase hub](../../modules/semantic/research/m_b1_offline_safe_region_phase_20260709.md) — closed nav
2. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md) — freeze identity
3. [portable OR-tail hook contract](../../modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md) — Stage 1 eng task
4. [two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md) — Stage 1 full contract + Stage 2 boundary
5. [wire status](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md) — what code landed vs still open
6. [signal_analysis_ledger](../eval/signal_analysis_ledger.md) §5

Also: [DEVELOPMENT.md](../../../DEVELOPMENT.md) D1/D3 · [semantic TODO](../../modules/semantic/TODO.md)

Optional: [PR #83](https://github.com/raylei50653/saccade/pull/83) · [PR #87](https://github.com/raylei50653/saccade/pull/87)

## Artifacts

**Contract path (required):**

- `out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json`

**Local studies (if present; missing does not block — regenerate from docs/tools):**

```text
out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/
out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv
out/signal_study/m_loo_attr_*/
out/signal_study/m_repaired_tail_region_*/
out/signal_study/m_b2e2e_smoke_*/
out/signal_study/m_b2_bridge_ab_*/
out/signal_study/m_b1_hook_ab_*/   # offline pairs replay tables only (so far)
```

**Ledger ids:** `m.gate.repaired_b2e2e_smoke` · `m.gate.portable_or_tail_hook`

## Current step

```text
DONE (code):
  policy loader + CLI default-off + CUDA thr inject
  offline full event table
    = offline pairs replay only (n_rejected=8721 = freeze)
    ≠ online B-audit / runtime event table / decision-change join

NEXT (PR #87 scope if kept narrow):
  A1 hook-off identity vs trusted B2 baseline
  B  hook-on e2e metrics + native counters (hook_eligible / rejected / atom fires)
  publish e2e_safe_for_default_off: yes/no  (classification may be neutral_but_safe)

PENDING (not implied done by offline table; may be same PR or follow-up):
  online full event audit (B-audit)
  online fired atom ids / singleton vs co-fire / per-seq rejected
    on the online candidate universe
  decision-change / reconnect-change joins
  Stage 1 artifact freeze only after e2e + agreed audit depth

STOP after Stage 1; do not start Stage 2 in same PR
```

Runner:

```bash
# offline full tables — offline pairs replay only (not online B-audit)
uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp> \
  --offline-events-only

# e2e A1/B (hook-off identity + hook-on); counters from get_relink_debug
# full B-audit still pending unless runner grows online event export
uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp> \
  --run-e2e
```

## Acceptance

**Allowed:**

- frozen portable OR-tail policy only
- default-off flag / env / CLI
- baseline B2 vs B2+hook
- rejected-candidate audit (**online** when claiming Stage 1 close)

**Required headline:**

```text
e2e_safe_for_default_off: yes/no
```

**Supporting audit — online candidate universe (pending unless B-audit lands):**

```text
policy_path
candidate_id
n_rejected
fired atom ids
singleton-only vs co-fire
per-seq rejected counts
reconnect / IDF1 / AssA / HOTA / MOTA deltas
IDs / FP / FN deltas
runtime overhead
determinism / hash
```

**Already available offline (pairs replay):** full event table + atom/per-seq summaries with `n_rejected=8721`. Does **not** substitute online Acceptance rows above.

**Minimum for a partial Stage 1 eng milestone (not full artifact freeze):** A1 identity + B e2e metrics + native counters; leave online full-table Acceptance as explicit pending.

## Must not

- search / repair / learned weights
- zone/gap atoms / runtime refit
- production preset change / silent default-on
- re-open rule search
- edit Tier-B as-of notes for status
- create another method note for the same freeze
- claim online/e2e safety before A/B
- treat offline pairs replay as online B-audit complete
- freeze Stage 1 artifacts without e2e (+ agreed audit depth)
- weaken tests

## Status churn only

When results land, update **only**:

1. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
2. [signal_analysis_ledger](../eval/signal_analysis_ledger.md)
3. [hook contract](../../modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md) **or** new hook result note

Plus this thread’s Status / Current step / History.

## History

- 2026-07-09: offline research candidate established (`m_b1_repaired_eps0_loo_pass_20260709`)
- 2026-07-09: offline smoke pass · online_blocked（correct boundary *at that time*）
- 2026-07-09: offline phase CLOSED; thread opened with session hook consolidated
- 2026-07-10: two-stage plan landed (`m_b1_to_m_b1_5_two_stage_plan_20260710`); Stage 1 eng starts
- 2026-07-10: Stage 1 wire — loader/CLI/CUDA thr inject/`run_m_b1_hook_ab.py`; offline pairs replay n_rejected=8721; e2e A/B pending
- 2026-07-10: status reconciliation — `online_hook_wired__e2e_pending`; boundary = hook wired / e2e not validated; offline table ≠ online B-audit; NEXT split A1/B+counters vs full B-audit pending
