---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# m_b1 online hook thread

> **One-line:** Research is closed. Start from hub + card + hook contract. Do only default-off portable OR-tail hook → B2 vs B2+hook e2e A/B. Preset unchanged.

## Status

- **M-B1 offline gate / safe-region research = CLOSED**
- Offline: `LOO_pass_region_candidate` · offline smoke pass (GT0, FP=8721)
- Online: **blocked**
- `e2e_safe_for_default_off`: **no** (until A/B)
- Production: **not** preset · **not** default-on

```text
candidate_id: m_b1_repaired_eps0_loo_pass_20260709
  = LOO_pass_region_candidate
  = offline_smoke_pass (GT0, FP=8721)
  = online_blocked
  ≠ e2e_safe_for_default_off
  ≠ production preset
```

## Current boundary

Research-only default-off online hook. Offline candidate frozen; online/e2e not wired.

```text
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
3. [portable OR-tail hook contract](../../modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md) — **only next eng doc**
4. [signal_analysis_ledger](../eval/signal_analysis_ledger.md) §5

Also: [DEVELOPMENT.md](../../../DEVELOPMENT.md) D1/D3 · [semantic TODO](../../modules/semantic/TODO.md)

Optional: [PR #83](https://github.com/raylei50653/saccade/pull/83)

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
```

**Ledger ids:** `m.gate.repaired_b2e2e_smoke` · `m.gate.portable_or_tail_hook`

## Current step

```text
Implement research-only default-off online hook that applies:
  out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json

Follow:
  docs/modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md
```

Then: **baseline B2 vs B2+hook** e2e A/B.

## Acceptance

**Allowed:**

- frozen portable OR-tail policy only
- default-off flag / env / CLI
- baseline B2 vs B2+hook
- rejected-candidate audit

**Required headline:**

```text
e2e_safe_for_default_off: yes/no
```

**Supporting audit:**

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

## Must not

- search / repair / learned weights
- zone/gap atoms / runtime refit
- production preset change / silent default-on
- re-open rule search
- edit Tier-B as-of notes for status
- create another method note for the same freeze
- claim online/e2e safety before A/B
- weaken tests

## Status churn only

When results land, update **only**:

1. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
2. [signal_analysis_ledger](../eval/signal_analysis_ledger.md)
3. [hook contract](../../modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md) **or** new hook result note

Plus this thread’s Status / Current step / History.

## History

- 2026-07-09: offline research candidate established (`m_b1_repaired_eps0_loo_pass_20260709`)
- 2026-07-09: offline smoke pass · online_blocked（correct boundary）
- 2026-07-09: offline phase CLOSED; thread opened with session hook consolidated
