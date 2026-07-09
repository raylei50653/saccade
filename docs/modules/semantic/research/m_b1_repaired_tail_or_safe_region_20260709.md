# Repaired all-tail OR — ε=0 safe region near q85

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Candidate:** [`m_b1_repaired_eps0_loo_pass_20260709`](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)  
**Tool:** `scripts/tools/repaired_tail_or_safe_region.py`  
**Study:** [`out/signal_study/m_repaired_tail_region_20260709T150000Z/`](../../../../out/signal_study/m_repaired_tail_region_20260709T150000Z/)  
**Ledger:** `m.gate.repaired_tail_region`

```text
Question:  在 all-tail repaired OR 下，q85 附近是否有厚 ε=0 productive plateau？
Answer:    YES → UPGRADE = LOO_pass_region_candidate
status:    still offline / ≠ production preset
```

---

## 0. What was measured (not new search)

Frozen policy family:

```text
OR of singleton tails (no zone, no gap):
  score_m_bridge | abs_log_h | dist_h | abs_ratio_m1 | resid_mean
```

Coordinates:

| mode | definition |
|:--|:--|
| **shared_pool_q** | thr_i = quantile(pool signal_i, q); **same q** for all atoms |
| **2D pair** | free (q_a, q_b); other atoms fixed at freeze thr@q85 |
| **LOO shared_q** | thr fit on train-6 at q; apply held-out |

Domain: q ∈ **[0.70, 0.99]** (grid 60×1D / 25×25 2D).  
Freeze point: **q=0.85**.

---

## 1. Shared-q (primary)

| ε | safe% | **p80%** | p80 width (Δq) | robust% | best FP | best_q | freeze safe? | class |
|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| **0** | **55.9%** | **13.6%** | **0.034** | 55.9% | 9546 | **0.833** | **True** | **usable_safe_region** |
| 0.001 | 55.9% | 13.6% | 0.034 | 55.9% | 9546 | 0.833 | True | usable |
| 0.01 | 79.7% | 22.0% | 0.059 | 67.8% | 12504 | 0.764 | True | usable |

```text
ε=0:
  • safe band covers majority of q-domain (not a single grid cell)
  • productive@80 ≈ 14% of domain; Δq ≈ 0.034 around best
  • best_q ≈ 0.833  ≈ freeze 0.85 (not a distant sweet spot)
  • freeze q=0.85 is inside safe region
```

Curve snippets near freeze (`shared_q_curve.csv`):

| q | GT_hurt | FP_removed |
|--:|--:|--:|
| 0.80 | 0 | (high) |
| 0.83 | 0 | ~best |
| **0.85** | **0** | **~8721-class** |
| … | 0 while safe | … |

---

## 2. LOO shared-q (transfer of region)

| metric @ ε=0 | value |
|:--|:--|
| LOO safe% | **56.4%** |
| LOO productive@80 | **15.4%** |
| LOO productive width Δq | **0.037** |
| LOO best_q | **0.834** |
| freeze q=0.85 folds GT0 | **7/7** |
| freeze mean te FP | **1281** |
| class | **usable_safe_region** |

```text
Region transfers: LOO safe/productive fractions ≈ in-sample shared-q.
Freeze remains LOO-clean inside that region.
```

---

## 3. 2D pairs (others @ q85) @ ε=0

| pair | safe% | p80% | bdist | freeze safe | class |
|:--|--:|--:|--:|:--|:--|
| score ∩ abs_log_h | 69.4% | 41.7% | 0.012 | True | **broad_safe_productive** |
| score ∩ dist_h | 52.8% | 28.5% | 0.012 | True | broad |
| abs_log_h ∩ abs_ratio | **91.2%** | **46.4%** | 0.012 | True | broad |
| dist_h ∩ abs_log_h | 82.5% | 22.4% | 0.012 | True | broad |
| score ∩ resid | 38.9% | 38.9% | 0.012 | True | broad |

2D slices with free thr on two tails (rest frozen) show **broad** productive regions; freeze (0.85,0.85) stays safe.

---

## 4. Upgrade decision

| level | criteria | result |
|:--|:--|:--|
| LOO-pass **point** | freeze LOO GT0 | already true from candidate card |
| LOO-pass **region** | thick ε=0 productive band + LOO band + freeze inside | **YES** |

```text
UPGRADE = LOO_pass_region_candidate
```

Still **not** production:

```text
offline region ≠ e2e safety
preset unchanged
B2/e2e still required
```

---

## 5. Contrast vs pre-repair narrative

| | unrepaired OR-5 (zones/gap) | repaired all-tail |
|:--|:--|:--|
| LOO | 5/7 partial | 7/7 pass |
| ε=0 geometry | isolated / brittle thr | **shared-q safe ~56%**, productive band |
| freeze q85 | LOO-risky soft atoms | **inside** LOO-safe region |

Ban zone/gap did not only clean LOO points — it left a **usable q-band** around the freeze.

---

## 6. Reproduce

```bash
uv run python scripts/tools/repaired_tail_or_safe_region.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --portable out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --study-dir out/signal_study/m_repaired_tail_region_<stamp>
```

---

## 7. Next

```text
1. ✅ Freeze repaired candidate
2. ✅ repaired 2D / shared-q region → LOO_pass_region_candidate
3. → B2/e2e smoke on candidate_id only
4. → default-off prototype discussion if e2e clean
5. → preset still NO
```
