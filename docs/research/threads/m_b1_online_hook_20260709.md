---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# m_b1 online hook thread

> **One-line:** **Stage 1 CLOSED** (eng milestone passed). Offline portable rule **failed online relevance, not safety**. Next = online B-audit event table (domain redefinition), **not** thr re-fit.

## Status

- **M-B1 offline gate / safe-region research = CLOSED**
- **Stage 1 frozen-hook eng milestone = CLOSED**
- Offline: `LOO_pass_region_candidate` · offline smoke pass (GT0, FP=8721)
- Online: hook **wired default-off** · e2e A1/B **done**
- `e2e_safe_for_default_off`: **yes**
- Classification: `online_effect_neutral_but_safe__vacuous_online_thr`
- Production: **not** preset · **not** default-on
- Canonical e2e: [m_b1_hook_stage1_e2e_20260710.md](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md)
- Wire: [m_b1_hook_stage1_wire_20260710.md](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md)
- Plan: [two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md)

```text
Stage 1 engineering milestone: PASSED
portable offline rule online relevance: FAILED  (support mismatch)
portable offline rule online safety:     PASSED (A1≡B, Δ=0)

e2e_safe_for_default_off: yes
classification: online_effect_neutral_but_safe__vacuous_online_thr
production_preset: unchanged

online_hook_eligible (B): 244
online_hook_rejected (B): 0
offline n_rejected:       8721   ← not online pruning power
```

### What Stage 1 proved (locked)

1. **Engineering safety** — hook-off is invisible; hook-on does not break baseline (A1≡B metrics + MOT hashes).
2. **Online wiring** — `eligible=244` means the policy evaluator runs (not dead code / wrong counters).
3. **Offline hypothesis is not online-portable** — offline 8721 rejects mostly live outside production gates; offline pruning power ≠ online pruning power.

### Domain diagnosis (locked)

Not “threshold too large only” — **conditional-domain / support mismatch**:

```text
D_offline  = all recorded offline pairs

D_online   = D_offline
             ∩ { bdist ≤ bridge_px (=0.4) }
             ∩ { 0.6 ≤ hr ≤ 1.7 }
             ∩ { other baseline gates pass }

q85 thr estimated on D_offline, applied on highly truncated D_online
  → thr outside online support is the natural outcome
```

## Current boundary

```text
Stage 1 CLOSED for eng + safety headline.
PR boundary: wire + A/B runner + counters + Stage 1 evidence only.
NO thr calibration in Stage 1 / this PR.

portable policy freeze still:
  out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json
  OR of 5 singleton tail_q85 · no zone/gap · ban_gap+ban_zone
```

## Read first

1. [e2e close note](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md) — Stage 1 formal close
2. [wire status](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md)
3. [two-stage plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md) § Stage 2 entry
4. [signal_analysis_ledger](../eval/signal_analysis_ledger.md)
5. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)

## Artifacts

```text
out/signal_study/m_b1_hook_ab_20260710T062345Z/     # canonical Stage 1 e2e
out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json
```

**Ledger:** `m.gate.portable_or_tail_hook`

## Current step

```text
Stage 1: CLOSED

NEXT (separate work; not thr re-fit first):
  1. Online full B-audit event table for the 244 baseline-ok pairs
     signals · GT/FP outcome · atom margins · final association
  2. Only then decide Stage 2 study domain

Stage 2 research object (redefined — not offline q85 again):
  In the production-accepted conditional domain D_online,
  does a stable, generalizable safe-negative region exist?

  max_C  FP_removed(C | D_online)
  s.t.   GT_hurt(C | D_online) ≤ ε

If audit shows insufficient FP mass in the 244:
  conclusion may be placement-too-late
  (safe negatives already consumed by upstream production gates)

Then compare three directions (only after audit):
  A. keep placement, fine calibration only
  B. move research hook earlier (pre bridge gate) for larger domain
  C. stop reject; use signals as ranking / assignment margin
```

## Acceptance (Stage 1 — met)

```text
e2e_safe_for_default_off: yes
A1 ≡ B metrics + result hashes
eligible(B)=244, rejected(B)=0; eligible(A1)=0
default-off does not enter policy eval
production preset unchanged
offline 8721 not claimed as online effect
```

## Must not

- thr re-fit / rule search as “Stage 1 fix”
- claim offline FP=8721 as online pruning power
- silent default-on / preset flip
- Stage 2 remodeling in the Stage 1 PR
- skip B-audit and jump to re-fit

## Status churn only

1. [e2e note](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md)
2. [candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
3. [signal_analysis_ledger](../eval/signal_analysis_ledger.md)
4. This thread Status / Current step / History

## History

- 2026-07-09: offline research candidate established
- 2026-07-09: offline smoke pass · online_blocked (boundary at that time)
- 2026-07-09: offline phase CLOSED; thread opened
- 2026-07-10: two-stage plan; Stage 1 wire
- 2026-07-10: e2e A1/B — A1≡B; elig=244 rej=0; e2e_safe=yes
- 2026-07-10: **Stage 1 formally CLOSED** — eng safety + wiring OK; portable offline rule failed **online relevance** (support mismatch), not safety; next = online B-audit domain redefinition, not thr re-fit
