# M-B1.5 Stage 2 Q4 — signal separability on D_online

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 2 Q4 separability audit only. No thr / Boolean / production.
**Upstream:** [Q1–Q3](m_b1_5_stage2_q1q3_d_online_audit_20260710.md)
**Entry contract:** [m_b1_5_stage2_entry_contract_20260710.md](m_b1_5_stage2_entry_contract_20260710.md)
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)

## Terminal classification

```text
stage2_q4_separability: separability_weak_or_unstable
terminal_letter: C

n_primary_negative: 23
n_primary_positive_protect: 64
sequences_with_both_classes: 7
best_oriented_AUC (frozen): 0.588  (abs_ratio_m1)
best_Cliff_δ: 0.177

next_authorized_step:
  change signal family (or earlier hook placement)
  — do NOT threshold-chase on inseparable support

production_preset: unchanged
restricted_safe_region_modeling: NOT authorized
```

**Entry-contract mapping:** terminal **B** in the Stage 2 legal set  
(*FP mass > 0 AND no stable separation → change signal family, not thr chase*).

---

## Primary cohort (locked)

```text
negative class:
  resolved ∧ baseline_selected ∧ pair_label == negative
  → n = 23

positive protection class:
  resolved ∧ baseline_selected ∧ pair_label == gt_consistent
  → n = 64

excluded from main conclusion:
  unresolved (39), ambiguous (4), all non-selected resolved rows
```

Non-selected resolved rows were audited only as **secondary**  
(`q4_secondary_non_selected.json`) — not mixed into main AUC.

| seq | n_neg | n_pos |
|:--|--:|--:|
| MOT17-02-SDP | 7 | 11 |
| MOT17-04-SDP | 1 | 6 |
| MOT17-05-SDP | 2 | 13 |
| MOT17-09-SDP | 3 | 2 |
| MOT17-10-SDP | 5 | 17 |
| MOT17-11-SDP | 2 | 5 |
| MOT17-13-SDP | 3 | 10 |

Mass is **sufficient** for Q4 judgment (not terminal D).

---

## Single-signal pooled (frozen raw)

| feature | AUC_oriented | Cliff δ | direction | effect | pure-neg prefix (hi/lo) | LOO dir flip |
|:--|--:|--:|:--|:--|:--|:--|
| score_m_bridge | 0.513 | 0.026 | higher_neg | negligible | 0 / 0 | **yes** |
| abs_log_h | 0.584 | 0.167 | higher_neg | weak | 0 / 0 | no |
| dist_h | 0.535 | 0.071 | higher_neg | negligible | 0 / 0 | no |
| abs_ratio_m1 | **0.588** | **0.177** | higher_neg | weak | 0 / 0 | no |
| resid_mean | 0.550 | 0.101 | higher_neg | weak | 1 / 0 | no |

### Overlap

All five show **thick range overlap**: `neg_in_pos_range_frac ≈ 0.96–1.0`  
(negatives almost fully inside GT-selected range).

### Sibling transforms

`log1p`, `sq`, `neg` (direction flip), `margin_to_online_bridge_gate` (0.4 − score),  
`resid_over_dist_h` — **none** improve beyond the raw best (~0.59).  
No Boolean combinations or learned weights.

---

## Cross-sequence / LOO

- 7/7 sequences have both classes (some small-n: 04 has 1 neg).
- Per-seq AUCs hover near 0.5 with frequent **sign noise** (small n).
- LOO recomputation: best features stay weak (min LOO oriented AUC ≈ 0.51–0.55).
- `score_m_bridge` shows LOO **direction flip**.
- **No** feature meets: moderate+ effect ∧ LOO-stable ∧ multi-seq pure-neg tail.

---

## Pure-negative tail (descriptive only)

```text
Not a rule / not a candidate / not a threshold.
```

- Extreme high/low prefixes of frozen signals: **pure-neg length 0** for four signals.
- `resid_mean` high: pure prefix **1** (single point, single seq) — fails multi-seq gate.
- Extreme deciles always GT-contaminated.

→ **No observed multi-seq zero-GT-hurt pure-neg tail.**

---

## Predefined context slices

Only predefined online context used:

- sequence
- competitor_count ∈ {0, ≥1}
- score_m_bridge vs half production bridge gate (0.2 / 0.4 family)

No invented adaptive cuts. Slice audit: no slice achieves stable moderate+ separation with multi-seq pure-neg tail and coverage gates → **terminal B (conditional) not met**.

`gap` / explicit height-ratio columns are **not present** on the Stage 1 B-audit export; not fabricated.

---

## Why not thr search

Entry contract Q4 fail path:

```text
no stable separation
  → change signal family / features
  → DO NOT hard-tune thresholds on inseparable support
```

Pooled max oriented AUC 0.59 with thick overlap means any thr that removes meaningful FP mass will also hit GT-selected bridges. Searching thr would only **overfit noise**.

---

## Claim firewall

```text
threshold_search_not_authorized
boolean_rule_search_not_authorized
formal_safe_region_not_authorized
hook_policy_not_authorized
production_promotion_blocked
frozen_policy_effect_claim_inadmissible   (triggered still 0)
safe_region_modeling_not_authorized
```

Allowed:

```text
report_descriptive_separability_facts
authorize_signal_family_change_or_placement_revisit
```

---

## Study artifacts

```text
out/signal_study/m_b1_5_stage2_q4_20260710/
  q4_cohort.{csv,parquet}
  q4_cohort_summary.json
  q4_signal_separability.csv
  q4_pooled_detail.json
  q4_ecdf.csv
  q4_per_sequence.csv
  q4_loo.csv
  q4_stability.csv
  q4_tail_audit.csv
  q4_slice_audit.csv
  q4_secondary_non_selected.json
  reconciliation.json   # PASS
  summary.{json,md}
  manifest.json
```

```bash
uv run python scripts/tools/run_m_b1_5_stage2_q4.py \
  --q1q3-study out/signal_study/m_b1_5_stage2_q1q3_20260710 \
  --out out/signal_study/m_b1_5_stage2_q4_20260710
```

Code: `src/saccade/perception/eval/d_online_stage2_q4.py`  
Tests: `tests/unit/test_d_online_stage2_q4.py`

---

## Authorized next steps (only)

```text
1. Q4 closes singleton frozen-tail thr *promotion* — not threshold/Boolean
   as a descriptive analysis method on the locked cohort
2. Next executed: Stage 2 Q4.5 structured threshold-combination atlas
   → m_b1_5_stage2_q45_threshold_atlas_20260710.md
3. Do NOT promote freeze thr or production preset from Q4 alone
4. Do NOT read Q4-C as “Boolean combination impossible”
```

---

## One-line mainline (Q4 close — historical as-of pre-Q4.5)

```text
M-B1 Stage 1: CLOSED
Stage 2 Q1–Q3: SUFFICIENT mass (23 safe-removable)
Stage 2 Q4: C — frozen signals inseparable on decision-relevant D_online
next (as-of Q4, superseded by Q4.5 atlas + ranking path):
  change signal family and/or earlier placement — not thr chase
production: blocked
```

---


