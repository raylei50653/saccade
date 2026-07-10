# Candidate card — repaired ε=0 LOO-pass

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->
<!-- fact-owner: freeze-identity = this card; phase nav = m_b1_research_history_20260709_20260710.md -->

### Offline freeze identity (unchanged historical value)

```text
candidate_id:       m_b1_repaired_eps0_loo_pass_20260709
validation_status:  LOO_pass_region_candidate
offline_smoke:      pass  (GT0 · FP=8721 · freeze-aligned)
lifecycle_status:   candidate_only / pre-production research
≠ production gate
```

### Current lifecycle (as of 2026-07-10 Stage 1 + Stage 2 close)

```text
Stage 1 overall:              CLOSED
hook mechanism:               validated
freeze online relevance:      NULL_support_mismatch
  (A1≡B · elig=244 · rej=0 · triggered=0)
e2e_safe_for_default_off:     yes
  = null-effect mount is safe
  ≠ online policy effective
  ≠ freeze thr fires on D_online
Stage 2 Q4.5 terminal:        isolated_safe_points_only
  (stable region candidates = 0)
production promotion:         blocked
production_preset:            unchanged
```

Evidence: [Stage 1 final](m_b1_stage1_online_hook_final_20260710.md) ·  
[Stage 2 final](m_b1_5_stage2_d_online_final_20260710.md) ·  
[`m_b1_hook_ab_20260710T071001Z_stage1_close`](../../../../out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/).

> **Phase hub / maintenance:** [m_b1_research_history_20260709_20260710.md](m_b1_research_history_20260709_20260710.md) — offline method notes closed as-of.  
> Do not re-open freeze thr identity when reading Stage 1/2 lifecycle.

`lifecycle_status` 與 `validation_status` **不是互斥**：前者是「能不能當 production 物件」，後者是 offline 驗證到哪一層。

---

## 0. One-line freeze

```text
AtomRepairConfig(ban_gap_bins=True, ban_zone=True)
  = repaired ε=0 all-tail OR candidate
  = LOO_pass_region_candidate (offline)
  ≠ production preset
```

**Headline**

> LOO hurt attribution localized ε=0 transfer failures to `score_m_bridge:zone_q70` plus sequence-specific gap/dist_h zone atoms. Banning gap-bin and zone atoms yields a repaired candidate with **7/7 LOO GT_hurt=0** while retaining **97.3%** test FP removal. Offline-only; half-repair showed replacement risk (`speed_mismatch:tail_q95`). **Repaired shared-q / 2D safe-region audit upgrades this card to `LOO_pass_region_candidate`.**

**Current (not a re-open of freeze):** Stage 1 hook eng is **CLOSED** with freeze online relevance **NULL**; Stage 2 Q4.5 finds **no stable reject region** on \(D_{\text{online}}\). Preset promotion remains **blocked**.

中文：offline freeze 身份不變（可歸因、可修復、region candidate）；線上 lifecycle 已推進到 Stage 1 CLOSED + Q4.5 isolated_safe_points_only，**不得**把 offline LOO-pass 當 production GO。

---

## 1. Identity

| Field | Value |
|:--|:--|
| **candidate_id** | `m_b1_repaired_eps0_loo_pass_20260709` |
| **freeze study** | [`out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/`](../../../../out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/) |
| **source LOO / attr** | [`out/signal_study/m_loo_attr_20260709T143000Z/`](../../../../out/signal_study/m_loo_attr_20260709T143000Z/) |
| **region study** | [`out/signal_study/m_repaired_tail_region_20260709T150000Z/`](../../../../out/signal_study/m_repaired_tail_region_20260709T150000Z/) |
| **repair_config** | `ban_gap_bins=True`, `ban_zone=True` |
| **epsilon** | **0.0** only（ε=0.01 不混入本 card） |
| **lifecycle_status** | `candidate_only` / pre-production research |
| **validation_status** | **`LOO_pass_region_candidate`** |
| **production_preset** | **unchanged** |
| **pairs** | `out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv` |
| **n_pos / n_neg** | 340 / 21449 |
| **tool** | `gate_rule_search.py` + `AtomRepairConfig` · `loo_hurt_attribution.py` · `repaired_tail_or_safe_region.py` |
| **substrate** | `mamba_whole_graph_m` B1 offline 7-seq SDP |

Artifacts:

| file | content |
|:--|:--|
| `candidate.json` | full freeze (metrics, LOO pointer, risks) |
| `portable_policy.json` | thr-portable OR policy for re-apply |

---

## 2. Repair config (normative)

```python
AtomRepairConfig(
    ban_gap_bins=True,   # drop gap:bin_*
    ban_zone=True,       # drop *:zone_q* hard_zone family
)
```

Search params (same as prior OR-5 line except repair):

```text
eps            = 0.0
max_and_size   = 3
max_or_rules   = 5
min_fp_support = 80
tail quantiles = 0.85, 0.90, 0.95, 0.99
zone quantiles = (banned)
gap bins       = (banned)
```

---

## 3. Frozen policy (7-seq in-sample under repair)

```text
reject if
  score_m_bridge:tail_q85
  OR abs_log_h:tail_q85
  OR dist_h:tail_q85
  OR abs_ratio_m1:tail_q85
  OR resid_mean:tail_q85
```

**All tails — no zone, no gap.** OR of 5 singleton clauses.

### Atom thr (fit on full 7-seq; portable JSON)

| atom_id | role | kind | thr (7-seq fit) |
|:--|:--|:--|--:|
| `score_m_bridge:tail_q85` | condition | tail_q | **11.91** |
| `abs_log_h:tail_q85` | support | tail_q | **1.349** |
| `dist_h:tail_q85` | condition | tail_q | **6.732** |
| `abs_ratio_m1:tail_q85` | support | tail_q | **2.086** |
| `resid_mean:tail_q85` | condition | tail_q | **14.04** |

> Thr numbers are **audit freeze as-of 7-seq fit**, not a production contract. LOO re-fits thr on train-6; re-apply live via `portable_policy.json` only for offline replay of this freeze.

### In-sample metrics (7-seq, this freeze)

| Metric | Value |
|:--|:--|
| FP_removed | **8721** (40.7% of FP) |
| GT_hurt | **0** |
| seq_hurt_std | 0 |
| vs prior unrepaired in-sample OR-5 (9130 FP) | −409 FP in-sample (−4.5%) — expected after ban soft zones |

Per-seq in-sample (frozen apply):

| seq | FP_removed | GT_hurt |
|:--|--:|--:|
| MOT17-02-SDP | 1014 | 0 |
| MOT17-04-SDP | 285 | 0 |
| MOT17-05-SDP | 1740 | 0 |
| MOT17-09-SDP | 46 | 0 |
| MOT17-10-SDP | 2133 | 0 |
| MOT17-11-SDP | 620 | 0 |
| MOT17-13-SDP | 2883 | 0 |

---

## 4. LOO (from attribution study — locks transfer claim)

| Field | Value |
|:--|:--|
| protocol | strict 6-train search under **same repair**, apply train thr to held-out |
| config name | `ban_gap_ban_zone` |
| **GT0 folds** | **7 / 7** |
| **sum test GT_hurt** | **0** |
| mean test FP | **1244** |
| baseline unrepaired mean te FP | 1278 |
| **FP retained** | **97.3%** |
| verdict | **`loo_pass_eps0`** |

```text
Repair is NOT “safety by killing capacity”.
hurt 3 → 0 while holding ~97% held-out FP.
```

---

## 5. Attribution recorded on this card

### known_risky_atoms（banned by this candidate）

| atom | role in failure |
|:--|:--|
| `score_m_bridge:zone_q70` | productive_but_risky — co-conspirator on 02 & 10 |
| `gap:bin_gap_61_150` | seq_specific — MOT17-10 |
| `dist_h:zone_q50` | seq_specific / boundary — MOT17-02 |

### replacement_risk（must not re-open casually）

| risk | evidence |
|:--|:--|
| `speed_mismatch:tail_q95` | under **half-repair** (`ban_gap` + `min_zone_q=0.70` only) search backfills and **hurts MOT17-05** (2 GT) |
| partial ban | search will substitute another soft atom — card requires **full** `ban_zone`, not “tighten zone to q70” alone |

### stable_clean_atoms（keep）

```text
abs_log_h:tail_q85+
abs_ratio_m1:tail_q85+
dist_h:tail_q85+
score_m_bridge:tail_q85+   (tail, not zone)
resid_mean:tail_q85+
```

---

## 6. How to read / how NOT to

```text
✅ What this card is:
   - Auditable freeze of repaired ε=0 all-tail OR candidate
   - Proof that LOO failure was atom-local, not architecture-dead
   - Offline LOO_pass_region_candidate (shared-q + 2D region audit done)
   - Pointer for B2 / e2e smoke on this candidate_id only

❌ What this card is NOT:
   - production preset change
   - permission to flip default-on gates
   - claim that offline LOO-pass / region-pass ⇒ e2e safety
   - claim that production coupling has been validated
```

**Mandatory caveat before B2/e2e:**

```text
offline ε=0 LOO-pass / region-pass does not imply production e2e safety.
```

e2e brings candidate generation differences, runtime ordering, upstream/downstream coupling, per-frame association side effects.

---

## 7. Region audit (done) + completion ledger

**Region study:** [`m_repaired_tail_region_20260709T150000Z`](../../../../out/signal_study/m_repaired_tail_region_20260709T150000Z/)  
**Note:** [m_b1_research_history_20260709_20260710.md](m_b1_research_history_20260709_20260710.md)

| check | ε=0 result |
|:--|:--|
| shared_pool_q safe% | **55.9%** of q∈[0.70,0.99] |
| productive_safe_area@80 | **13.6%** (Δq≈0.034) |
| best_q | **0.833** (near freeze 0.85) |
| freeze q=0.85 safe | **True** |
| LOO shared-q safe% / p80 | **56.4% / 15.4%** |
| LOO freeze GT0 | **7/7** |
| **upgrade** | **`LOO_pass_region_candidate`** |

### Completion ledger (required_before_preset — current)

```text
✅ repaired 2D / shared-q productive_safe_area
✅ B2/e2e smoke (candidate_id-scoped)
   historical as-of 2026-07-09: offline_smoke_pass__online_blocked
   (online not yet wired at that checkpoint — correct boundary)
✅ research default-off portable OR-tail hook (Stage 1 CLOSED)
✅ evaluation-entry A/B + B-audit + strict A0 + determinism + runtime
✅ e2e_safe_for_default_off = yes  (null-effect mount only)
   ≠ online policy effective
✅ freeze online power: ABSENT (NULL_support_mismatch; rejected=0)
✅ Stage 2 Q1–Q4.5 on D_online
   Q4.5 terminal: isolated_safe_points_only
   stable region candidates: 0
🚫 production preset: unchanged / promotion blocked
```

### Historical checkpoint as of 2026-07-09 (do not treat as current)

```text
prior checkpoint (pre–Stage 1 close):
  offline_smoke_pass__online_blocked
  e2e_safe_for_default_off: no   (A/B not yet run)
  next step then: implement default-off hook + true e2e A/B
superseded by: Stage 1 close study m_b1_hook_ab_20260710T071001Z_stage1_close
               + Stage 2 Q4.5 atlas terminal B
```

---

## 8. Related

| doc | role |
|:--|:--|
| [LOO atom repair](m_b1_research_history_20260709_20260710.md) | attribution + repair table (source of LOO numbers) |
| [unrepaired OR-5 card](m_b1_research_history_20260709_20260710.md) | prior in-sample freeze (**superseded for LOO claims**) |
| [LOO baseline](m_b1_research_history_20260709_20260710.md) | unrepaired 5/7 partial |
| [weight methods](m_b1_research_history_20260709_20260710.md) | no thick ε=0 weight plateau; motivated atom line |
| [repaired tail OR region](m_b1_research_history_20260709_20260710.md) | shared-q / 2D ε=0 region → validation upgrade |
| [GT safe region area](m_b1_research_history_20260709_20260710.md) | general 2D area protocol (historical) |
| Ledger | `m.gate.repaired_eps0_loo_pass` |

---

## 9. Reproduce freeze

```bash
# LOO + repairs (source numbers)
uv run python scripts/tools/loo_hurt_attribution.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_loo_attr_<stamp> \
  --eps 0.0 --jobs 7 --run-repairs

# In-sample freeze under repair (this candidate study)
# see out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/candidate.json
```
