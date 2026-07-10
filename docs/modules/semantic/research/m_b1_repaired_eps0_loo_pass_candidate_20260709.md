# Candidate card — repaired ε=0 LOO-pass

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->
<!-- fact-owner: freeze-identity = this card; phase nav = m_b1_offline_safe_region_phase_20260709.md -->

```text
candidate_id:       m_b1_repaired_eps0_loo_pass_20260709
lifecycle_status:   candidate_only / pre-production research
validation_status:  LOO_pass_region_candidate
offline_smoke:      pass  (GT0 · FP=8721 · freeze-aligned)
online_hook:        wired_default_off
e2e:                A1≡B Δ0 · eligible=244 · rejected=0
e2e_safe_for_default_off: yes
classification:     online_effect_neutral_but_safe__vacuous_online_thr
production_preset:  unchanged
≠ production gate  (thr vacuous under prod bridge_px/height)
```

E2e study: [`m_b1_hook_ab_20260710T062345Z`](../../../../out/signal_study/m_b1_hook_ab_20260710T062345Z/) · [e2e note](m_b1_hook_stage1_e2e_20260710.md).

> **Phase hub / maintenance:** [m_b1_offline_safe_region_phase_20260709.md](m_b1_offline_safe_region_phase_20260709.md) — intermediate method notes are closed as-of; do not re-edit their verdicts when working the hook.

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

> LOO hurt attribution localized ε=0 transfer failures to `score_m_bridge:zone_q70` plus sequence-specific gap/dist_h zone atoms. Banning gap-bin and zone atoms yields a repaired candidate with **7/7 LOO GT_hurt=0** while retaining **97.3%** test FP removal. Offline-only; half-repair showed replacement risk (`speed_mismatch:tail_q95`). **Repaired shared-q / 2D safe-region audit upgrades this card to `LOO_pass_region_candidate`. B2/e2e is still required before any preset discussion.**

中文：這輪不是證明可上線，而是證明 ε=0 failure **可歸因、可修復**，且幾乎不掉 FP；region audit 後升級為 **region candidate**；下一關只剩 **B2/e2e**（preset 仍 NO）。

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

## 7. Region audit (done) + required_before_preset

**Region study:** [`m_repaired_tail_region_20260709T150000Z`](../../../../out/signal_study/m_repaired_tail_region_20260709T150000Z/)  
**Note:** [m_b1_repaired_tail_or_safe_region_20260709.md](m_b1_repaired_tail_or_safe_region_20260709.md)

| check | ε=0 result |
|:--|:--|
| shared_pool_q safe% | **55.9%** of q∈[0.70,0.99] |
| productive_safe_area@80 | **13.6%** (Δq≈0.034) |
| best_q | **0.833** (near freeze 0.85) |
| freeze q=0.85 safe | **True** |
| LOO shared-q safe% / p80 | **56.4% / 15.4%** |
| LOO freeze GT0 | **7/7** |
| **upgrade** | **`LOO_pass_region_candidate`** |

Still required before preset:

```text
1. ✅ repaired 2D / shared-q productive_safe_area
2. ✅ B2/e2e smoke (candidate_id-scoped)
     → offline_smoke_pass__online_blocked
     → e2e_safe_for_default_off: no
3. research default-off portable OR-tail hook
     → m_b1_portable_or_tail_hook_contract_20260709.md
4. e2e A/B: baseline B2 vs baseline B2 + hook
5. no metric regression (AssA / IDF1 / reconnect / per-seq)
6. explicit default-off prototype discussion — not silent preset merge
```

Order:

```text
✅ offline research candidate 已成立
✅ online/e2e 邊界尚未打通（smoke 正確暴露）
→ 下一步只補 default-off hook
     contract: Implement a research-only, default-off online hook that
     applies frozen portable OR-tail policy from portable_policy.json,
     without search/repair/learned weights/zone-gap/preset changes.
→ then A/B: baseline B2 vs baseline B2 + hook
→ sole headline: e2e_safe_for_default_off: yes/no
→ preset 不動（全程）
```

---

## 8. Related

| doc | role |
|:--|:--|
| [LOO atom repair](m_b1_loo_hurt_atom_repair_20260709.md) | attribution + repair table (source of LOO numbers) |
| [unrepaired OR-5 card](m_b1_policy_card_eps0_or5_20260709.md) | prior in-sample freeze (**superseded for LOO claims**) |
| [LOO baseline](m_b1_gate_rule_search_loo_20260709.md) | unrepaired 5/7 partial |
| [weight methods](m_b1_weight_method_safe_region_20260709.md) | no thick ε=0 weight plateau; motivated atom line |
| [repaired tail OR region](m_b1_repaired_tail_or_safe_region_20260709.md) | shared-q / 2D ε=0 region → validation upgrade |
| [GT safe region area](m_b1_gt_safe_region_area_20260709.md) | general 2D area protocol (historical) |
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
