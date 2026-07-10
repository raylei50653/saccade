---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# m_b1 online hook thread

> **One-line:** **Stage 1a evaluation-entry PASSED**; freeze policy **null online relevance**. **Stage 1b action-path** via plumbing controls (see study). **Stage 1 overall OPEN** until B-audit + strict A0 + determinism. Not “Stage 1 CLOSED” from vacuous freeze alone.

## Status (honest split)

| Milestone | Status |
|:--|:--|
| **Stage 1a evaluation-entry** | **PASSED** — policy load + eval entry; A1 eligible=0; B eligible>0 |
| **Frozen-policy online relevance** | **NULL** — support mismatch; B rejected=0; A1≡B |
| **Stage 1b action-path** | plumbing controls P/F (activation thr=0.2 + force-reject) — see latest study |
| **Online B-audit full event table** | **PENDING** (contract still requires it for full Stage 1) |
| **Strict A0 identity** | **NOT MET** (6/7 soft identity only) |
| **Determinism repeated-run / runtime overhead** | **PENDING** |
| **Stage 1 overall** | **OPEN** |
| Production preset | **unchanged** |

```text
Allowed claim:
  policy loading and evaluation-entry wiring are valid;
  online rejection/action chain requires Stage 1b controls (not freeze B alone).

Forbidden claim:
  “the full hook engineering chain is valid” from freeze A1/B only
  “Stage 1 CLOSED” while B-audit / strict A0 / action-path incomplete
```

## What freeze A1/B actually proved

1. **Evaluation-entry** — hook-off does not enter policy eval; hook-on increments eligible.
2. **Freeze null effect** — offline thr outside \(D_{\text{online}}\) support → rejected=0, A1≡B.
3. **Not proven by freeze B alone** — atom fire → reject → candidate suppression → decision change.

## Domain diagnosis (still valid)

```text
D_offline = all recorded pairs
D_online  = D_offline ∩ {bdist≤0.4} ∩ {0.6≤hr≤1.7} ∩ {other baseline gates}
```

q85 on \(D_{\text{offline}}\) applied on \(D_{\text{online}}\) → natural support mismatch.

## Stage 1b plumbing controls (not production thr search)

Pre-specified, non-metric-picked:

| Arm | Policy | Intent |
|:--|:--|:--|
| **P** | `control_arm=activation`, atom0 thr=**0.2** (midpoint of bridge_px=0.4), others disabled | signal → comparison → atom0 → reject |
| **F** | `control_arm=force_reject`, atom0 thr=**−1** | `hook_rejected == hook_eligible` + decision change vs A1 |

Fixtures: `scripts/tools/fixtures/m_b1_stage1/`

Accept P: `eligible>0`, `atom0>0`, `rejected==atom0` (and preferably result ≠ A1).  
Accept F: `rejected==eligible`, result ≠ A1.

## Read first

1. [e2e / close note](../../modules/semantic/research/m_b1_hook_stage1_e2e_20260710.md) — **Stage 1a only; 1b controls**
2. [wire](../../modules/semantic/research/m_b1_hook_stage1_wire_20260710.md)
3. [plan](../../modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md)
4. [ledger](../eval/signal_analysis_ledger.md)

## Artifacts

```text
out/signal_study/m_b1_hook_ab_20260710T062345Z/           # Stage 1a freeze A1/B
out/signal_study/m_b1_hook_ab_*_stage1b/                  # + P/F controls when present
scripts/tools/fixtures/m_b1_stage1/*.json
out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json
```

## Current step

```text
DONE:
  Stage 1a evaluation-entry + freeze vacuous diagnosis
  control_arm loader + fixtures + runner --run-action-path-controls
  milestone classifier (1a / 1b / overall OPEN)

RUN / RECORD:
  --run-e2e --run-action-path-controls  (P + F arms)
  publish Stage 1b pass/fail without claiming Stage 1 CLOSED

STILL PENDING for Stage 1 overall CLOSED:
  online full B-audit event table (zero/singleton/cofire/rejected/decision-changed)
  strict A0 identity (or rebased A0 stamp)
  hook-on repeated-run hashes
  proper runtime overhead accounting
```

Runner:

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp>_stage1b \
  --run-e2e --run-action-path-controls
```

## Must not

- claim Stage 1 CLOSED from freeze vacuous A1/B alone
- thr search / production re-fit as Stage 1 “fix”
- silent soft-A0 → strict identity upgrade
- treat offline 8721 as online pruning power
- preset / default-on

## History

- 2026-07-09: offline candidate + thread
- 2026-07-10: wire + e2e freeze A1/B; over-claimed Stage 1 CLOSED
- 2026-07-10: **review correction** — Stage 1a only; action path unactivated; split 1a/1b; plumbing controls added; overall remains OPEN
