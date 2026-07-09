# B2/e2e smoke contract — repaired region candidate only

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Candidate:** [`m_b1_repaired_eps0_loo_pass_20260709`](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)  
**Tool:** `scripts/tools/smoke_repaired_candidate_b2e2e.py`  
**Smoke study:** [`out/signal_study/m_b2e2e_smoke_m_b1_repaired_eps0_loo_pass_20260709T151000Z/`](../../../../out/signal_study/m_b2e2e_smoke_m_b1_repaired_eps0_loo_pass_20260709T151000Z/)  
**Ledger:** `m.gate.repaired_b2e2e_smoke`

```text
m_b1_repaired_eps0_loo_pass_20260709
  = LOO_pass_region_candidate
  = offline_smoke_pass
  = online_blocked
  ≠ e2e_safe_for_default_off
  ≠ production preset

smoke_verdict: offline_smoke_pass__online_blocked
next: research default-off portable OR-tail hook
      (see m_b1_portable_or_tail_hook_contract_20260709.md)
```

---

## 0. Boundary (locked)

### Target

```text
m_b1_repaired_eps0_loo_pass_20260709 only
```

### Allowed

```text
offline replay / same candidate_id / default-off research path
attach prior B2 reconnect A/B as substrate reference (read-only)
```

### Not allowed

```text
preset change
silent default-on
new atom search
extra repair during smoke
mixing ε=0.01 relaxed frontier
claiming online apply without tracker hook
```

**Docs support B2/e2e smoke for this frozen region candidate only — not preset change.**

---

## 1. Questions this smoke must answer

| # | Question | How answered this stamp |
|--:|:--|:--|
| 1 | Policy apply on B2/e2e substrate? | **offline yes** / **online not wired** |
| 2 | GT_hurt still 0 / no contracted regression? | offline replay **GT_hurt=0** all 7 seq |
| 3 | FP pruning / reconnect side effects expected? | FP **8721** matches freeze; reconnect **N/A** (no online apply) |
| 4 | IDF1 / AssA / reconnect no clear regression? | **N/A for candidate** — B2 ref is production bridge ablate, not this policy |
| 5 | Runtime ordering / cand-gen coupling breaks offline? | **untested** until default-off online path exists |

---

## 2. Smoke result (as-of study)

```text
B2/e2e smoke result for m_b1_repaired_eps0_loo_pass_20260709:
  validation_status remains / changes to: LOO_pass_region_candidate
  (unchanged — offline re-confirmed; e2e not elevating)
  e2e_safe_for_default_off: no
  production_preset: unchanged
  smoke_verdict: offline_smoke_pass__online_blocked
  blockers_before_default_off:
    - no_online_tracker_hook_for_portable_or5_tails
      (cannot answer e2e/reconnect under candidate apply)
```

### Offline replay (portable_policy on B1 pairs)

| metric | value |
|:--|:--|
| GT_hurt | **0** |
| FP_removed | **8721** (= freeze) |
| per-seq GT_hurt | all **0** |
| policy | OR of 5 `*:tail_q85` (no zone/gap) |

### B2 substrate reference (not candidate apply)

| | |
|:--|:--|
| study | `out/signal_study/m_b2_bridge_ab_20260709T094646Z/` |
| role | production-like bridge ON/OFF reconnect + e2e baseline |
| candidate_applied_online | **false** |

---

## 3. Interpretation

```text
✅ Offline claim still holds under re-apply (GT0, FP stable).
✅ Smoke protocol is candidate_id-scoped; preset untouched.
❌ Cannot promote e2e_safe_for_default_off=yes yet:
     tracker has no hook to inject portable OR-5 tail rejects at runtime.
```

Honest next engineering step (still default-off research):

```text
1. research default-off portable OR-tail hook
   → contract: m_b1_portable_or_tail_hook_contract_20260709.md
2. Re-run B2 reconnect + e2e A/B: baseline vs baseline+hook
3. Only then re-evaluate e2e_safe_for_default_off
4. production_preset still NO until that + no regression
```

**Not stuck — correctly stopped at missing online hook.**

---

## 4. Reproduce

```bash
uv run python scripts/tools/smoke_repaired_candidate_b2e2e.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --candidate-dir out/signal_study/m_b1_repaired_eps0_loo_pass_20260709 \
  --b2-study out/signal_study/m_b2_bridge_ab_20260709T094646Z \
  --study-dir out/signal_study/m_b2e2e_smoke_<stamp>
```

---

## 5. Related

| doc | role |
|:--|:--|
| [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md) | freeze + region upgrade (canonical) |
| [region audit](m_b1_repaired_tail_or_safe_region_20260709.md) | q85 productive region |
| [B2 bridge A/B](m_b2_reconnect_bridge_ab_20260709.md) | production substrate reference |
| Schema §0.1 | B1 ≠ B2 |
