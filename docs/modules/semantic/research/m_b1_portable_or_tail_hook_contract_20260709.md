# Research task — default-off portable OR-tail hook

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Task name:** `research default-off portable OR-tail hook`  
**Candidate only:** [`m_b1_repaired_eps0_loo_pass_20260709`](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)  
**Blocked by smoke:** [B2/e2e smoke](m_b1_repaired_candidate_b2e2e_smoke_contract_20260709.md)  
  → `offline_smoke_pass__online_blocked`

> **Only living eng doc for next phase.** Intermediate M-B1 notes are closed; nav: [phase hub](m_b1_offline_safe_region_phase_20260709.md).

### Main line (locked)

```text
Stage 1 overall OPEN
  Stage 1a evaluation-entry: PASSED
  freeze online relevance: NULL (support mismatch)
  Stage 1b action-path (plumbing controls): PASSED
  e2e_safe_for_default_off (freeze null-effect): yes
  online B-audit / strict A0 / determinism: PENDING
preset 不動
next for Stage 1 close = B-audit + strict A0 + determinism rows
Stage 2 thr domain work = separate PR; not offline q85 first
```

### Development goal (one contract sentence)

```text
Implement a research-only, default-off online hook that applies the frozen
portable OR-tail policy from portable_policy.json, without search, repair,
learned weights, zone/gap atoms, or preset changes.
```

**Purpose of the hook is not to ship preset.**  
It exists so the offline region candidate can be **correctly validated by online e2e**.

```text
production_preset: unchanged  (always, for this task)
```

---

## 0. Current status (locked)

```text
Stage 1 overall: OPEN

m_b1_repaired_eps0_loo_pass_20260709
  = LOO_pass_region_candidate
  = offline_smoke_pass
  = Stage 1a evaluation-entry PASSED
  = freeze online relevance NULL (support mismatch)
  = Stage 1b action-path PASSED under control arms only
  = e2e_safe_for_default_off: yes  (freeze B null-effect)
  ≠ production preset
  ≠ full Stage 1 CLOSED
```

| layer | result |
|:--|:--|
| offline replay | GT_hurt=0 · FP=8721 · freeze-aligned |
| Stage 1a | A1 eligible=0 · B eligible=244 · freeze rej=0 · A1≡B |
| Stage 1b controls | P atom0=168 rej=168 · F rej=elig=305 · decision Δ vs A1 |
| A0 identity | soft 6/7 · **strict not met** |
| B-audit / determinism / runtime contract | **PENDING** |
| next for overall close | online full event table + strict A0 + repeated-run hashes |

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
  RESERVED / NOT IMPLEMENTED (fail-closed if set)
  online full candidate-event export = PENDING (B-audit)
  offline pairs replay: run_m_b1_hook_ab.py --offline-events-only
```

Must log: `policy_path`, `candidate_id`, `n_rejected`, never enable without explicit flag.
Loader refuse: missing candidate_id soft-fallback, thr drift, hash drift, op≠`>`.

---

## 2. True e2e A/B (only after hook is implemented)

```text
A: baseline B2
B: baseline B2 + portable OR-tail hook
```

Same substrate recipe as [B2 bridge note](m_b2_reconnect_bridge_ab_20260709.md) except B enables the research flag + freeze `portable_policy.json`.

### Required report (supporting table)

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

### Watch specially

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
Until A/B exists, the answer remains **`no`**.

---

## 3. Implementation notes (non-binding)

Likely attach points (to inspect when coding — not a commit plan):

| area | path hint |
|:--|:--|
| live bridge score | `src/tracking/tracker_gpu.cu` / `tracker_gpu.py` bridge_px path |
| Python relink gates | `src/saccade/perception/eval/relink.py` reject counters |
| CLI / config | research flag on `mot17.py` / runner — **never** preset yaml |

Policy semantics to preserve:

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

## 4. Headline

> B2/e2e smoke confirms the frozen repaired candidate can be replayed offline with freeze-aligned GT0 / FP=8721, but it is not yet wired into the online tracker. The candidate remains LOO_pass_region_candidate with offline_smoke_pass__online_blocked verdict. No preset change is supported. Next step is a research-only default-off online hook for portable OR-tail policy injection, followed by true e2e A/B.

中文：這輪不是卡住，是正確停在「缺 online hook」工程邊界；下一步只補 **default-off hook**，不該動 preset。

---

## 5. Related

| doc | role |
|:--|:--|
| [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md) | freeze + region status |
| [smoke contract](m_b1_repaired_candidate_b2e2e_smoke_contract_20260709.md) | offline pass / online blocked |
| [region audit](m_b1_repaired_tail_or_safe_region_20260709.md) | q85 productive region |
| [B2 bridge A/B](m_b2_reconnect_bridge_ab_20260709.md) | baseline B2 recipe |
| [two-stage plan](m_b1_to_m_b1_5_two_stage_plan_20260710.md) | Stage 1 full eng contract + Stage 2 boundary |
