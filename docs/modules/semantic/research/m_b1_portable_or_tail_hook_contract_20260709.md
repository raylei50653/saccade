# Research contract — default-off portable OR-tail hook

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Task name:** `research default-off portable OR-tail hook`  
**Contract role:** normative ABI + acceptance criteria (historical/normative).  
**Execution evidence:** [m_b1_stage1_online_hook_final_20260710.md](m_b1_stage1_online_hook_final_20260710.md)  
**Candidate freeze:** [`m_b1_repaired_eps0_loo_pass_20260709`](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)  
**Offline smoke (historical):** [history §14](m_b1_research_history_20260709_20260710.md)  
  → as-of 2026-07-09: `offline_smoke_pass__online_blocked` (pre–Stage 1 wire)

> **Contract completed (Stage 1 CLOSED).** This file remains the code/docstring ABI.  
> Do not read residual “implement hook” language as current work — see §0 and Stage 1 final.  
> Offline nav: [history](m_b1_research_history_20260709_20260710.md).

> **§20 re-classification (2026-07-12, bookkeeping).** Under the sealed
> role-aligned experiment contract
> ([framework §20](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md),
> PR #133), the frozen portable OR-tail is a **performance upper-bound
> candidate** (§20.4) — it documents what the frozen atom family can remove
> offline under the permitted complexity class — **not a gate design
> candidate**. The hook's level-3 acceptance path is unaffected: it is
> engineering **plumbing**, and its value is (a) intervention-chain
> validation on the live surface, (b) online-retention measurement of the
> offline upper bound, and (c) a control arm for any future monotone-CORE
> comparison. Any promotion of the OR-tail toward a design recommendation
> requires a new design-evaluation study passing §20.5; re-labeling is not
> promotion. No result, limit, or ABI in this file changes.

### Main line (locked — current)

```text
Stage 1 overall: CLOSED  (contract executed)
  Stage 1a evaluation-entry: PASSED
  freeze online relevance: NULL_support_mismatch
  Stage 1b action-path (plumbing controls): PASSED
  online B-audit full table: PASSED (244 rows)
  strict A0 (rebased stamp): PASSED
  determinism repeated-run: PASSED
  runtime contract (named wall-clock): PASSED
  e2e_safe_for_default_off (freeze null-effect): yes
    = null-effect mount safe ≠ online policy effective
production_preset: unchanged
policy_effect_supported: no  (triggered=0 on D_online)
Stage 2: Q1–Q4.5 completed → terminal isolated_safe_points_only
next preferred: ranking / assignment-relative audit (not thr-as-rule)
```

### Development goal (normative contract sentence — still binding for ABI)

```text
Implement a research-only, default-off online hook that applies the frozen
portable OR-tail policy from portable_policy.json, without search, repair,
learned weights, zone/gap atoms, or preset changes.
```

**Purpose of the hook is not to ship preset.**  
It exists so the offline region candidate can be **correctly validated by online e2e**.  
**Execution status:** implemented and closed under Stage 1 final evidence.

```text
production_preset: unchanged  (always, for this task)
```

---

## 0. Current status (locked — post Stage 1 close)

```text
Stage 1 overall: CLOSED
contract_execution: completed

m_b1_repaired_eps0_loo_pass_20260709
  = LOO_pass_region_candidate  (offline freeze identity)
  = offline_smoke_pass
  = Stage 1a evaluation-entry PASSED
  = freeze online relevance NULL_support_mismatch
  = Stage 1b action-path PASSED under control arms only
  = online B-audit full table PASSED (244 zero-fire)
  = strict A0 rebased stamp PASSED
  = determinism + runtime contract PASSED
  = e2e_safe_for_default_off: yes  (freeze B null-effect only)
  ≠ production preset
  ≠ online-effective freeze thr
  ≠ Stage 2 safe-region GO  (Q4.5 = isolated_safe_points_only)
```

| layer | result |
|:--|:--|
| offline replay | GT_hurt=0 · FP=8721 · freeze-aligned |
| Stage 1a | A1 eligible=0 · B eligible=244 · freeze rej=0 · A1≡B |
| Stage 1b controls | P atom0=168 rej=168 · F rej=elig=305 · decision Δ vs A1 |
| B-audit | 244 rows · recon ok · native counters align |
| A0 identity | **strict_pass** (rebased from A1; legacy Jul-09 stale) |
| Determinism / runtime | **PASSED** (named wall-clock; pure kernel NOT_MEASURED) |
| Stage 2 | Q1–Q4.5 complete · terminal B isolated_safe_points_only |
| next | ranking / assignment-relative (see Stage 2 final) |

---

## 1. Hook task contract (narrow)

### Allowed

```text
- 只吃 portable_policy.json (from freeze study)
- 只支援 frozen OR singleton tail atoms
    score_m_bridge | abs_log_h | dist_h | abs_ratio_m1 | resid_mean
    thr = portable atom thr; op = >
- default-off env / CLI flag only
- 不做新 search
- 不改 preset / 不 merge production config
- 不支援 ε=0.01 / zone / gap atoms
- 只輸出 A/B metrics + rejected-candidate audit
```

### Forbidden

```text
- silent default-on
- runtime repair / re-fit quantiles from online stream
- learned weights / soft-AND fusion / CVaR
- zone/gap atom re-open
- production config merge
- expanding policy beyond freeze portable_policy.json
```

### Flag shape (Stage 1 — implement against two-stage plan)

```text
--research-portable-or-tail-policy PATH/portable_policy.json
  default: unset / off
  when set: load thr with freeze lock (hash + thr vector + op='>');
            reject relink candidate if ANY tail fires
  when unset: zero behavior change vs production preset

--research-portable-or-tail-audit
  online full candidate-event export (B-audit event ring)
  default off; export-only (no decision change)
  offline pairs replay: run_m_b1_hook_ab.py --offline-events-only
```

Must log: `policy_path`, `candidate_id`, `n_rejected`, never enable without explicit flag.
Loader refuse: missing candidate_id soft-fallback, thr drift, hash drift, op≠`>`.

---

## 2. True e2e A/B (historical/normative acceptance criteria)

**execution_status:** completed under Stage 1 close  
**evidence:** [Stage 1 final](m_b1_stage1_online_hook_final_20260710.md) · study `m_b1_hook_ab_20260710T071001Z_stage1_close`

Normative arms (still the contract shape):

```text
A: baseline (hook-off / A1)
B: baseline + portable OR-tail hook (freeze policy)
```

Same substrate family as [B2 bridge note](m_b2_reconnect_bridge_ab_20260709.md) when B enables the research flag + freeze `portable_policy.json`.

### Required report (supporting table) — normative checklist

```text
IDF1 / AssA / HOTA / MOTA
IDs / FP / FN
reconnect counts + success rate
rejected candidate count (hook audit)
GT_hurt proxy / matched-GT audit if available
per-seq deltas
runtime overhead
determinism / hash
```

### Watch specially (normative)

```text
- 剪候選後 reconnect 是否變差
- FP 降了但 FN / IDs 上升
- 某條 seq 單獨退化
- runtime ordering 是否讓 offline GT0 不成立
```

### Sole headline conclusion from A/B

```text
e2e_safe_for_default_off: yes / no
```

Supporting metrics justify that bit; they do not replace it.

**Observed outcome (Stage 1 close — freeze B null-effect):**

```text
e2e_safe_for_default_off: yes
  = null-effect mount is safe (A1≡B; rejected=0)
  ≠ online policy effective (NULL_support_mismatch)
```

### Historical checkpoint as of 2026-07-09 (pre-wire)

```text
prior checkpoint: Until A/B exists, e2e_safe_for_default_off remains no.
prior smoke: offline_smoke_pass__online_blocked
superseded by: Stage 1 CLOSED evidence pack
```

---

## 3. Implementation notes (ABI attachment — as implemented)

Attach points (landed; see Stage 1 final wire inventory):

| area | path hint |
|:--|:--|
| live bridge score | `src/tracking/tracker_gpu.cu` / `tracker_gpu.py` bridge_px path |
| Python relink gates | `src/saccade/perception/eval/relink.py` reject counters |
| CLI / config | research flag on runner — **never** preset yaml |
| policy loader | `src/saccade/perception/eval/portable_or_tail.py` |
| A/B + B-audit runner | `scripts/tools/run_m_b1_hook_ab.py` |

Policy semantics to preserve (normative):

```text
reject if
  score_m_bridge > thr_s
  OR abs_log_h > thr_h
  OR dist_h > thr_d
  OR abs_ratio_m1 > thr_r
  OR resid_mean > thr_res
```

with thr from freeze `portable_policy.json` only.

---

## 4. Headline (current)

> Stage 1 **CLOSED**: research default-off portable OR-tail hook is wired and validated (evaluation-entry, P/F action-path, B-audit 244, rebased A0, determinism, named runtime). Freeze remains **LOO_pass_region_candidate** offline, with **NULL online relevance** (support mismatch; rejected=0). `e2e_safe_for_default_off=yes` means null-effect mount safety only — **not** online thr power. Production preset **unchanged**. Stage 2 Q4.5: `isolated_safe_points_only` — no thr/hook-policy promotion.

中文：hook 工程已完成；freeze 線上仍無 thr 效力；preset 不動；下一研究方向是 ranking/assignment，不是再 chase offline thr。

### Historical headline as of 2026-07-09 (superseded)

> Prior checkpoint (pre-wire): B2/e2e smoke confirmed offline GT0/FP8721 but online not yet wired (`offline_smoke_pass__online_blocked`). Next step then was implement default-off hook + e2e A/B. **Superseded by Stage 1 close.**

---

## 5. Related

| doc | role |
|:--|:--|
| [Stage 1 final](m_b1_stage1_online_hook_final_20260710.md) | **execution evidence** |
| [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md) | freeze + lifecycle |
| [smoke (historical)](m_b1_research_history_20260709_20260710.md) | offline pass / online_blocked checkpoint |
| [region audit](m_b1_research_history_20260709_20260710.md) | q85 productive region |
| [B2 bridge A/B](m_b2_reconnect_bridge_ab_20260709.md) | baseline B2 recipe |
| [two-stage plan](m_b1_to_m_b1_5_two_stage_plan_20260710.md) | Stage 1+2 plan body |
| [Stage 2 final](m_b1_5_stage2_d_online_final_20260710.md) | Q1–Q4.5 terminals |
