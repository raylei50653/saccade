# M-B1.5 Stage 2 Q1–Q3 — D_online label join & safe-negative mass

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 2 Q1–Q3 evidence (not thr / Boolean / production).
**Entry contract:** [m_b1_5_stage2_entry_contract_20260710.md](m_b1_5_stage2_entry_contract_20260710.md)
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)

## Final classification

```text
stage2_q1_label_join: PASSED
stage2_q2_population_support: PASSED
stage2_q3_safe_negative_mass: SUFFICIENT

D_online_total: 244
label_resolved: 201
label_unresolved: 39
label_ambiguous: 4
gt_consistent: 114
negative: 87
decision_relevant_negative: 23
safe_removable_negative: 23
single_sequence_dominance: no   (max share 30.4% on MOT17-02-SDP; 7/7 seqs >0)

next_authorized_step: signal_separability_audit_on_D_online (Q4); no thr yet
production_preset: unchanged
```

Terminal **A** (entry contract mass path): safe-negative mass exists with multi-seq support → **enter Q4 separability**.  
**Q4 result (2026-07-10):** [m_b1_5_stage2_q4_separability_20260710.md](m_b1_5_stage2_q4_separability_20260710.md) → **C weak/unstable** (not thr).

---

## Study artifact

```text
out/signal_study/m_b1_5_stage2_q1q3_20260710/
  d_online_events.{csv,parquet}     # full 244 + labels + taxonomy
  label_join_summary.json
  label_join_errors.csv
  d_online_population_summary.json
  d_online_signal_support.csv
  d_online_per_sequence.csv
  safe_negative_mass_summary.json
  safe_negative_mass_per_sequence.csv
  reconciliation.json               # PASS
  summary.{json,md}
  manifest.json
```

Reproduce:

```bash
uv run python scripts/tools/run_m_b1_5_stage2_q1q3.py \
  --stage1-study out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close \
  --out out/signal_study/m_b1_5_stage2_q1q3_20260710
```

Code: `src/saccade/perception/eval/d_online_stage2.py` · tests `tests/unit/test_d_online_stage2_q1q3.py`

---

## Q1 — Label join

| | |
|:--|:--|
| **Join method** | `a1_mot_global_id_map + traj_centerdist_majority` |
| **MOT source** | Stage 1 `e2e_A1_hook_off` + `_global_id_map.txt` |
| **GT source** | `datasets/MOT17/train/<seq>/gt/gt.txt` |
| **Key** | stable `event_id` / `join_key` (duplicate → fail-closed) |
| **Coverage** | 201/244 resolved (**82.4%**); 39 unresolved; 4 ambiguous |

### Unresolved / ambiguous policy (declared)

- **Unresolved** (39): almost all = `cand_local` missing from `global_id_map` (cand ID never exported — typically short-lived pre-merge IDs). **Not** defaulted to FP or GT-safe.
- **Ambiguous** (4): low majority confidence or near-tie votes. Kept separate.
- Conf confidence: vote frac ≥ 0.3 and (votes ≥ 3 or short traj ≤ 5).

Join is **sufficient for population + mass measurement** under this declared incomplete-label policy. Rows without conf labels are excluded from FP-mass numerators.

---

## Q2 — Population & signal support

### Funnel

```text
D_online 244
  → resolved 201 / unresolved 39 / ambiguous 4
  → gt_consistent 114 / negative 87
  → baseline_selected 108 / non-selected 136
  → safe_removal_resolvable 23
```

### Frozen signal support (overall)

All five frozen thr remain **above observed max** on \(D_{\text{online}}\):

| signal | max on D_online | frozen thr | n_above_thr |
|:--|--:|--:|--:|
| score_m_bridge | ~0.40 | 11.91 | **0** |
| abs_log_h | ~0.53 | 1.35 | **0** |
| dist_h | ~1.86 | 6.73 | **0** |
| abs_ratio_m1 | ~0.69 | 2.09 | **0** |
| resid_mean | ~0.76 | 14.04 | **0** |

→ Confirms Stage 1 freeze null: **support_overlap_with_frozen_policy = 0** for every atom.  
Q2 only **describes** support; no thr grid.

---

## Q3 — Safe-negative mass taxonomy

### Definitions (inferred counterfactual, not observed intervention)

```text
pair_is_negative
  ≠ rejecting_pair_is_safe
  ≠ candidate_is_decision_relevant
  ≠ rejection_changes_decision

negative_safe_removable ≔
  pair_label == negative
  AND decision_relevance == selected
  (baseline accepted this wrong reconnect;
   rejecting it would remove a non-GT bridge)

When rejected_by_hook == 0:
  label_inference_kind = inferred_counterfactual
  — not an observed intervention outcome
```

### Counts

| quantity | n |
|:--|--:|
| N_negative | 87 |
| N_negative_selected / decision_relevant | 23 |
| N_negative_active_competitor (decision-neutral) | (rest of 87−23) |
| N_negative_safe_removable | **23** |
| safe_removable rate over D_online | 23/244 ≈ 9.4% |
| safe_removable \| negative | 23/87 ≈ 26.4% |

### Per-sequence safe-removable (all 7 seqs)

| seq | n_total | n_neg | safe_removable | share |
|:--|--:|--:|--:|--:|
| MOT17-02-SDP | 26 | 10 | 7 | 30.4% |
| MOT17-10-SDP | 73 | 27 | 5 | 21.7% |
| MOT17-09-SDP | 13 | 6 | 3 | 13.0% |
| MOT17-13-SDP | 49 | 12 | 3 | 13.0% |
| MOT17-05-SDP | 63 | 28 | 2 | 8.7% |
| MOT17-11-SDP | 10 | 3 | 2 | 8.7% |
| MOT17-04-SDP | 10 | 1 | 1 | 4.3% |

**single_sequence_dominance: no** (threshold 50%; max 30.4%).

---

## Claim firewall (this round)

```text
triggered == 0
  → frozen policy effect claim remains inadmissible

decision_relevant / safe_removable mass > 0 multi-seq
  → SUFFICIENT for Q4 separability entry only

threshold / Boolean claims
  → NOT authorized in Q1–Q3

production promotion
  → blocked

unresolved 39 / selected unresolved 21
  → mass is a lower bound under join policy;
    do not treat unresolved as FP
```

---

## What this does **not** mean

- Frozen offline thr is online-effective — **still no**
- 23 safe-removable is enough for thick safe-region / LOO — **not claimed**
- Best thr / Boolean clause exists — **not searched**
- Production preset should change — **no**

---

## Next authorized step

```text
Q4: signal separability audit on D_online
  domain: labeled resolved rows (esp. safe_removable vs gt_consistent selected)
  question: do the five frozen signals (or siblings) separate
            safe-removable negatives from GT-selected bridges
            inside D_online support?
  if yes → conditional safe-region (restricted grammar)
  if no  → change signal family (terminal B), not thr chase
```

---

## Reconciliation

`reconciliation.json`: **PASS**  
All derived summaries are group-by of `d_online_events`; partitions MECE.

---


