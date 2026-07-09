# LOO hurt attribution → atom repair → re-LOO

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Tool:** `scripts/tools/loo_hurt_attribution.py`  

> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

（repair 旗標：`gate_rule_search.AtomRepairConfig`）  
**Study:** [`out/signal_study/m_loo_attr_20260709T143000Z/`](../../../../out/signal_study/m_loo_attr_20260709T143000Z/)  
**Ledger:** `m.gate.loo_atom_repair`  
**Prior LOO:** [m_b1_gate_rule_search_loo_20260709.md](m_b1_gate_rule_search_loo_20260709.md)（partial 5/7）  
**Weight context:** [m_b1_weight_method_safe_region_20260709.md](m_b1_weight_method_safe_region_20260709.md)

```text
status: RESEARCH — ban_gap+ban_zone achieves loo_pass_eps0 (7/7)
         still NOT production preset; e2e/B2 not run
```

---

## 0. 問題改寫（承接 weight 結論）

```text
Weight methods: no thick ε=0 production-safe plateau.
ε=0.01 clipped_logz = relaxed frontier only.

Research question is now:
  which atoms prevent ε=0 safe region from transferring under LOO?
```

Priority path:

```text
1. LOO hurt attribution
2. atom repair / blacklist / conditionize
3. re-LOO (+ FP retained %)
4. only then 2D productive_safe_area maximize / B2 / e2e
```

---

## 1. Baseline LOO attribution（ε=0，同 strict 6+1）

| heldout | te FP | te hurt | failed clause | atom kinds |
|:--|--:|--:|:--|:--|
| MOT17-02 | 981 | **1** | `dist_h:zone_q50 ∧ score_m_bridge:zone_q70` | hard_zone ∧ hard_zone |
| MOT17-10 | 2149 | **2** | `gap:bin_61_150 ∧ score_m_bridge:zone_q70` | gap_bin ∧ hard_zone |
| 04/05/09/11/13 | — | **0** | — | — |

Aggregate: **5/7 GT0**, sum hurt **3**, mean te FP **1278** — matches prior LOO.

### Atom labels（baseline）

| label | atom | fail folds | mean FP alone | heldouts |
|:--|:--|--:|--:|:--|
| **productive_but_risky** | `score_m_bridge:zone_q70` | **2/5** | ~537 | 02, 10 |
| **seq_specific** | `gap:bin_gap_61_150` | 1/1 | ~1968 | 10 |
| **seq_specific** | `dist_h:zone_q50` | 1/1 | ~1896 | 02 |
| **stable_clean** | `abs_log_h:tail_q85` | 0/7 | ~476 | — |
| **stable_clean** | `abs_ratio_m1:tail_q85` | 0/7 | ~467 | — |
| **stable_clean** | `dist_h:tail_q85` | 0/6 | ~558 | — |
| **stable_clean** | `score_m_bridge:tail_q85/90` | 0 | 高 | — |
| **stable_clean** | `dist_h:zone_q70` | 0/5 | ~729 | —（本批未打 GT） |

解讀：

```text
1. Leaks are condition∧condition soft zones / gap bins — not support tails.
2. zone_q50 = boundary-touching soft body cut (train-quantile lucky).
3. gap_61_150 = seq-specific long-gap condition (10 only).
4. zone_q70 on score is the shared co-conspirator on both leak folds.
5. Support tail_q85 family is clean across all folds.
```

Full tables: `attribution_table.csv` · `atom_labels.csv` · `hurt_folds_detail.json`.

---

## 2. Repair presets → re-LOO

| config | repair | GT0 | sum hurt | mean te FP | **FP retained** | verdict |
|:--|:--|--:|--:|--:|--:|:--|
| baseline | {} | 5/7 | 3 | 1278 | 100% | loo_partial |
| ban_gap | ban gap bins | **6/7** | 1 | 1253 | **98.1%** | partial（02 still） |
| zone_q70_only | min_zone_q=0.70 | **6/7** | 2 | 1278 | 100% | partial（10 still） |
| ban_gap_zone70 | gap+min_q70 | 6/7 | 2 | 1249 | 97.7% | partial（05: **speed_mismatch:tail_q95**） |
| ban_gap_zone70_require_support | + require support in AND | 6/7 | 2 | 1248 | 97.6% | same 05 leak |
| **ban_gap_ban_zone** | ban gap + **ban all zone** | **7/7** | **0** | **1244** | **97.3%** | **loo_pass_eps0** |
| **strict_tail_only** | ban gap+zone + require support | **7/7** | **0** | **1244** | **97.3%** | **loo_pass_eps0** |

### 產能保留（productivity preservation）

```text
ban_gap_ban_zone:
  GT_hurt: 3 → 0  (LOO sum)
  mean held-out FP: 1278 → 1244  (−2.7%)
  ⇒ 不是「修成安全但沒產能」
```

單修 gap 或單抬 zone 都**不夠**；兩者一起 ban zone 才關乾。  
`ban_gap+zone70` 後搜尋會改選 **speed_mismatch:tail_q95** 打到 05（diagnostic 竄入）——提醒 repair 要連 **diagnostic 弱 atom** 一起看，不能只砍最初兩個 failed clause。

---

## 3. Recommended repair policy（candidate，非 preset）

```text
AtomRepairConfig(
  ban_gap_bins=True,
  ban_zone=True,          # drop hard_zone family entirely
  # optional: require_support_in_and=True  (same LOO result here)
)
```

Allowed mainline atoms after repair:

```text
support tails:   abs_log_h:tail_q85+, abs_ratio_m1:tail_q85+
condition tails: score_m_bridge / dist_h / resid / bridge : tail_q85+
banned:          gap:bin_*, *:zone_q*, preferably speed_mismatch as hard gate
```

**Still not production:**

```text
- LOO pass ≠ e2e AssA/IDF1 improvement
- MOT17-09 te FP still ~46 (thin seq capacity)
- no B2 / live reconnect test
- preset frozen unchanged
```

---

## 4. Ledger headlines

**EN**

> Weight-method safe-region audit found no thick ε=0 production-safe plateau. At ε=0.01, clipped log-z produced the thickest relaxed productive region and highest FP removal, while GT-CDF / soft-AND remained cleaner but thin. All results remain research-only.  
> **Atom-level LOO attribution:** leaks = `zone_q50/70` and `gap_61_150` condition clauses. **`ban_gap_bins + ban_zone` re-LOO → 7/7 GT0 with ~97% held-out FP retained.** Next: freeze repaired candidate card → optional e2e smoke; production preset still NO.

**中文**

> 加權法沒有厚 ε=0 安全域；ε=0.01 clipped log-z 最能形成 relaxed productive plateau，但仍 research-only。  
> **Atom LOO 歸因：** 漏點在 zone / gap condition。**ban_gap + ban_zone 後 re-LOO 7/7 GT0，held-out FP 保留約 97%。** 下一步：freeze 修復後 candidate → 可選 e2e；preset 不動。

---

## 5. Reproduce

```bash
uv run python scripts/tools/loo_hurt_attribution.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_loo_attr_<stamp> \
  --eps 0.0 --jobs 7 --run-repairs
```

~60s with 7-way pack-parallel LOO folds × repair presets.

---

## 6. Next ordered

```text
1. ✅ LOO hurt attribution
2. ✅ repair configs + re-LOO (ban_gap_ban_zone = pass)
3. ✅ freeze repaired candidate card
     → [m_b1_repaired_eps0_loo_pass_candidate_20260709.md](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
     → out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/
4. → repaired 2D productive_safe_area
5. → e2e / B2 smoke on that candidate_id only
6. → production preset: still NO
```
