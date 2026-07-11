# M-B1.5 Stage 2 D_online — final (through Q4.5)

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: stage2-d-online-final = this file; entry claim firewall = m_b1_5_stage2_entry_contract_20260710.md -->

**Role:** Canonical Stage 2 evidence on \(D_{\text{online}}\) through Q4.5 atlas terminal.  
**Entry contract (retain separately):** [m_b1_5_stage2_entry_contract_20260710.md](m_b1_5_stage2_entry_contract_20260710.md)  
**Stage 1 final:** [m_b1_stage1_online_hook_final_20260710.md](m_b1_stage1_online_hook_final_20260710.md)  
**Closed thread (history):** [m_b1_online_hook_20260709.md](../../../research/threads/closed/m_b1_online_hook_20260709.md)  
**Active next (T0):** [composition_grammar_safe_region.md](../../../research/threads/closed/composition_grammar_safe_region.md)  
**Coverage audit:** [composition_grammar_safe_region_coverage_audit_20260710.md](composition_grammar_safe_region_coverage_audit_20260710.md)  
**Consolidation:** [m_b1_doc_consolidation_report_20260710.md](m_b1_doc_consolidation_report_20260710.md)

```text
Stage 1: CLOSED · freeze online relevance NULL · production blocked
Stage 2 Q1–Q3: label join PASSED · population PASSED · safe-negative mass SUFFICIENT
Stage 2 Q4:
  q4_separability_grade: C  (separability_weak_or_unstable · best oriented AUC 0.588)
  stage2_entry_terminal_after_q4: B  (entry legal set: mass>0, no stable separation)
Stage 2 Q4.5 (evaluator v4):
  q45_atlas_terminal: B  (isolated_safe_points_only)
  productive_safe: 154 (single 1 / AND 153 / OR 0)
  region_candidates: 0 · coordinate-union interior: 0
  exact_absolute nested LOSO portable: 0
  selected_unresolved: 21 (blocks candidate; untrusted competition columns)
threshold / hook-policy promotion: blocked
next authorized: T0 Existing Atlas Region Interpretation Pack only
  (composition_grammar_safe_region thread)
ranking / assignment-relative: DEFERRED / BLOCKED
  until valid assignment-group key (currently invalid_frame_provenance)
production_preset: unchanged
```

Primary substrate:

```text
D_online = Stage 1 B-audit full table · 244 rows
study: out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/
locked decision cohort (Q4/Q4.5):
  negative: resolved ∧ selected ∧ negative = 23
  protect:  resolved ∧ selected ∧ gt_consistent = 64
  total 87 · both classes on 7/7 sequences
```

---

# Part A — Q1–Q3 label join & safe-negative mass

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
**Q4 result (2026-07-10):** `q4_separability_grade: C` (weak/unstable) → maps to `stage2_entry_terminal_after_q4: B` (not thr).

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

# Part B — Q4 signal separability

## Terminal classification

```text
stage2_q4_separability: separability_weak_or_unstable
q4_separability_grade: C
stage2_entry_terminal_after_q4: B

n_primary_negative: 23
n_primary_positive_protect: 64
sequences_with_both_classes: 7
best_oriented_AUC (frozen): 0.588  (abs_ratio_m1)
best_Cliff_δ: 0.177

next_authorized_step:
  change signal family (or earlier hook placement)
  — do NOT threshold-chase on inseparable support
  (Q4.5 later used thr/Boolean only as descriptive atlas — not policy)

production_preset: unchanged
restricted_safe_region_modeling: NOT authorized
```

**Namespace note:** `q4_separability_grade: C` is the Q4 effect/LOO grade.  
It maps to entry-contract legal terminal **`stage2_entry_terminal_after_q4: B`**  
(*FP mass > 0 AND no stable separation → change signal family, not thr chase*).  
Do **not** collapse Q4 grades with entry A/B/C or with `q45_atlas_terminal`.

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
   → m_b1_5_stage2_d_online_final_20260710.md
3. Do NOT promote freeze thr or production preset from Q4 alone
4. Do NOT read Q4-C as “Boolean combination impossible”
```

---

## One-line mainline (Q4 close — historical as-of pre-Q4.5)

```text
M-B1 Stage 1: CLOSED
Stage 2 Q1–Q3: SUFFICIENT mass (23 safe-removable)
Stage 2 Q4: q4_separability_grade C — frozen signals inseparable on decision-relevant D_online
next (as-of Q4, superseded by Q4.5 atlas v4 + ranking path):
  change signal family and/or earlier placement — not thr chase
production: blocked
```

---

# Part C — Q4.5 threshold-combination atlas (evaluator v4)

**Taxonomy:** `stage2_q45_atlas_v4`  
**Evidence pack:** [evidence/m_b1_5_stage2_q45_20260710/](evidence/m_b1_5_stage2_q45_20260710/)  
**Primary machine source:** `out/signal_study/m_b1_5_stage2_q45_20260710/summary.json`  
**PR truth base:** #89 final HEAD `6df1739b` (v4 evidence pack refresh)

## Terminal classification (v4)

```text
stage2_q45_terminal: isolated_safe_points_only
q45_atlas_terminal: B
taxonomy_version: stage2_q45_atlas_v4

n_primary_negative: 23
n_primary_positive_protect: 64
n_primary_resolved_selected: 87
n_selected_unresolved: 21   # tracked per-cell; not primary labels
n_selected_total: 108

atlas (frozen signals only; competition columns demoted / untrusted):
  single_atom rows:     870
  pairwise AND rows:  17640
  pairwise OR rows:   17640

productive_safe (resolved GT_hurt==0 ∧ n_neg>0 ∧ no unresolved capture):
  single: 1
  AND:  153
  OR:     0
  total: 154

stability (productive-safe coordinate-union interior; unique-mask nodes retained):
  isolated_safe_point:  7
  edge_candidate:      27
  has_interior (coord-union): 0
  same_mask_plateau_has_interior: 0
  region_candidates:    0
  max_component_size_coordinates: 19   # thin strips (e.g. 1×19 / 18×1)
  max_component_unique_masks: 4

nested LOSO (train lattice → select → freeze → holdout):
  n_folds: 7
  n_clauses_ever_selected: 1352
  n_exact_absolute_clauses_nested_loso_portable: 0
  # = exact absolute thr float@12dp repeatability; not quantile/rank region portability

evaluator gates:
  deletion_loo_is_portability: false
  fixed_full_sample_partition_not_portability: true
  nested_loso_required_for_region_A: true
  nested_loso_is_exact_absolute_clause_repeatability: true
  unresolved_contaminated_blocks_candidate: true
  semantic_duplicate_is_per_grid_not_global: true
  quotient_topology_retains_all_coordinates: true
  interior_on_productive_safe_coordinate_union: true
  same_mask_plateau_is_prediction_invariant_only: true
  assignment_group_key_status: invalid_frame_provenance
  secondary_competition_trusted: false

production_preset: unchanged
```

**Bounded finding (not global thr closure):**

> On the **resolved∧selected** cohort there are **154** sample-zero-GT atlas cells.  
> Interior is measured on the **productive-safe coordinate union** (adjacent thr  
> cells need not share the same mask). After that correction: **0** interior  
> coordinates, **0** region candidates (largest connected safe components are  
> thin lattice strips, e.g. 1×19 / 18×1, which cannot form a full bilateral /  
> 4-neighborhood interior). Exact-absolute nested LOSO finds **0** productive  
> portable clauses. Full selected population (incl. **21 unresolved**) still limits  
> safety claims; unresolved-contaminated cells cannot enter `region_candidate`.  
> Competition-relative columns are **untrusted** (`invalid_frame_provenance`).  
> **Still inadmissible:** portable safe-region · thr global closure ·  
> hook-policy promotion · e2e effect · production preset change.

**Interpretation of `q45_atlas_terminal: B`:** observed sample-zero-GT cells exist,  
but **none** meet coordinate-union interior + multi-seq + exact-absolute nested-LOSO  
region-candidate gates. They remain **atlas points**, not safe rules.

---

## Q4 boundary correction (locked language)

```text
Q4 weak marginal AUC (best oriented ≈ 0.588)
  → closes: singleton frozen-tail threshold *promotion* as policy
  → does NOT alone close: thr/Boolean as analysis
  → does NOT alone authorize: “threshold path fully falsified → ranking only”

Q4.5 maps structure. Ranking is a reasonable next research line after
valid assignment-group key + unknown coverage + nested LOSO portability.
```

Competition-relative features: **untrusted** (`invalid_frame_provenance`).  
Not ranking-path evidence until export provides a stable assignment key.

---

## Review #89 evaluator lineage (v2 → v4)

### v2

| Issue | Fix |
|:--|:--|
| LOO deletion = portability | Renamed `leave_one_sequence_deleted_*`; **cannot** promote |
| Unknown selected | per-cell `n_unresolved_selected`, pessimistic hurt, `unresolved_contaminated` |
| Boundary interior | bilateral (1D) / full-4 (pairwise); else `edge_candidate` |
| Assignment group | competition features demoted; frame not used as truth |

### v3

| Issue | Fix |
|:--|:--|
| Tautological “true holdout” | **nested LOSO**: rebuild lattice on train fold, select train-safe clauses, freeze thr/Boolean, evaluate holdout. Old fixed-thr partition kept only as `fixed_full_sample_region_partition_check` (not portability). |
| Global `seen_sig` isolation | **Per-grid** semantic-dup flag; **quotient topology** keeps all coordinates of each unique mask within a feature/direction (or pairwise) grid. |
| Stale evidence manifest | Clear `out_dir` → write all artifacts → write **current** manifest (HEAD + SHAs) → **then** copy evidence pack. |

### v4 (truth for this canonical)

| Issue | Fix |
|:--|:--|
| Interior = same-mask plateau only | **Safe-region interior** on productive-safe **coordinate union** (1D bilateral / 2D 4-neigh). Same-mask plateau = prediction-invariant only. |
| Nested LOSO over-claim | Report `n_exact_absolute_clauses_nested_loso_portable` (clause_id = feature+dir+thr@12dp). Not quantile/rank region portability. |
| Heterogeneous CSV fields | `write_csv` unions keys so pairwise stability columns are not dropped. |

Re-run under v4: still `has_interior=0`, `region_candidates=0`, exact-absolute portable=0 → **`q45_atlas_terminal: B` fully supported** under coordinate-union topology (no Q1–Q4 redo).

---

## Locked primary cohort

```text
negative:
  resolved ∧ baseline_selected ∧ pair_label == negative   → 23

positive protect:
  resolved ∧ baseline_selected ∧ pair_label == gt_consistent → 64

total = 87; both classes on 7/7 sequences
selected unresolved (not in primary labels): 21
```

Non-selected rows are excluded from main atlas conclusions.  
Unresolved selected rows are **tracked per cell** and **block** region candidacy when captured.

---

## Frame-column provenance (substrate check)

| | |
|:--|:--|
| **Column** | `frame` on Stage 1 B-audit / \(D_{\text{online}}\) |
| **Writer** | `tracker_gpu.cu` propose kernel: `ev.frame = frame_idx` |
| **Source of frame_idx** | host counter `portable_audit_frame_++` (“Host frame counter for B-audit”) |
| **Propagated into** | `event_id` (`…:f{frame}:…`) and `join_key` second field — **consistent** |
| **Absolute MOT frame?** | **No** |
| **Gap length?** | No |
| **Observed value** | all 244 events = `4` |
| **MOT cross-check** | track spans for event track_ids disagree with audit `frame==4` |
| **assignment_group_key_status** | **invalid_frame_provenance** |

```text
May claim absolute MOT frame: NO
May claim “temporal information unavailable” from frame==4 alone: NO
Observation limitation only: YES
Affects Q4.5 thr×AND/OR frozen-signal mainline: NO (mainline uses frozen atoms)
Affects competition-relative / ranking path: YES — columns untrusted until key fixed
```

Artifact: `frame_column_provenance.json` in the study dir / evidence pack.

---

## Method (registered lattices — complete families)

### Single-atom

- Lattice: **all observed unique values** on primary (`primary_unique_boundaries`)
- Directions: `high_tail` (\(x \ge t\)), `low_tail` (\(x \le t\))
- Signals: five frozen (`score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean`)
- Secondary (optional, **untrusted**): `sec_winner_runnerup_score_margin`,  
  `sec_delta_vs_ru_abs_log_h`, `sec_competitor_count`  
  (NaN when no competitor — **not** zero-filled)

### Pairwise (mainline frozen signals only)

- Lattice: **declared quantile lattice** \(q \in \{0, 0.05, \ldots, 1.0\}\)  
  (`primary_quantile_lattice_q05`)
- Combinators: **AND**, **OR** only (no 3+ atoms)
- Complete enumeration of registered cells; semantic-duplicate masks marked  
  **per-grid** (not global); coordinates retained under **quotient topology**
- No label-adaptive grammar expansion

### Per cell metrics

support, coverage, neg captured, GT hurt, precision, enrichment,  
per-sequence counts, n_seq with neg, max sequence share,  
`n_unresolved_selected` / `unresolved_contaminated`,  
deletion-LOO descriptors (not portability), nested-LOSO clause ids,  
neighbor / subset relations, necessity / observed-sufficiency descriptors.

### Stability classes (productive-safe only; v4)

| class | meaning |
|:--|:--|
| `isolated_safe_point` | productive-safe; no productive-safe lattice neighbor (coord-union) |
| `edge_candidate` | on a safe component but no interior coordinate under coord-union |
| `locally_stable_region` | has coord-union interior; multi-seq; not exact-absolute nested-LOSO portable |
| `loo_stable_region` | interior + exact-absolute nested LOSO portable + multi-seq non-dominant |
| `*_but_seq_thin` | geometric ok but single-seq or <2 seq neg mass |

**Interior definition (v4):** a thr coordinate is interior iff all required lattice  
neighbors are also productive-safe **in the coordinate union**, independent of  
mask identity. Unique-mask nodes still report prediction-invariant plateau width.

Only `loo_stable_region` / `locally_stable_region` set `is_region_candidate=1`.  
`unresolved_contaminated` **blocks** candidacy.

### Pareto

Full multi-objective frontier retained  
(`gt_hurt`, `n_neg_captured`, coverage, sequence support, max share, LOO).  
Not “best threshold only.” (`n_pareto_frontier = 303`)

---

## Headline facts (descriptive — v4)

1. **Single-atom productive-safe:** 1 cell  
   (`resid_mean` extreme high-tail, **n_neg=1**, single sequence).  
   Matches Q4 pure-neg prefix anecdote — **not** multi-seq.

2. **Pairwise AND productive-safe:** **153** cells (unknown-blocked; many semantic  
   duplicate masks collapsed at quotient nodes). Largest connected safe  
   components under coordinate-union topology are **thin lattice strips**  
   (e.g. 1×19, 18×1) → **0** full-neighborhood interiors → no region candidates.

3. **Pairwise OR productive-safe:** **0** cells  
   (OR inflates support into GT mass).

4. **Pareto frontier:** 303 objective-unique points spanning the GT_hurt vs  
   neg-capture tradeoff (including non-safe enriching cells). Full table in  
   `pareto_frontier.csv`.

5. **No A-eligible region candidate** under coord-union interior + multi-seq +  
   exact-absolute nested LOSO gates.

6. **Nested LOSO:** 1352 clauses ever selected across 7 folds;  
   **0** exact-absolute nested-LOSO portable.

---

## Claim firewall

```text
observed_safe_point  ≠  safe rule
productive_safe_point ≠  portable reject policy
enrichment without zero GT ≠  reject authorization
Q4 weak AUC ≠  Boolean impossible
deletion LOO ≠ portability
fixed full-sample partition ≠ portability
same-mask plateau interior ≠ coord-union safe-region interior

Blocked:
  production thr / hook policy / e2e effect / preset change
  unrestricted Boolean mining / 3+ atoms
  naming sample-zero-GT as safe rule
  competition-relative claims (untrusted frame provenance)

Allowed:
  retain and query full atlas
  describe isolated observed safe points
  ranking research as reasonable next line AFTER valid assignment-group key
```

---

## Study artifacts

```text
out/signal_study/m_b1_5_stage2_q45_20260710/          # runtime study (gitignored)
docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/  # committed pack
  threshold_registry.json
  frame_column_provenance.json
  cohort_summary.json
  region_stability.csv
  nested_loso_summary.json
  nested_loso_clause_summary.csv
  pareto_frontier.csv
  reconciliation.json
  summary.{json,md}
  manifest.json
  SHA256SUMS.json
  README.json
```

```bash
uv run python scripts/tools/run_m_b1_5_stage2_q45_atlas.py \
  --q1q3-study out/signal_study/m_b1_5_stage2_q1q3_20260710 \
  --stage1-study out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close \
  --out out/signal_study/m_b1_5_stage2_q45_20260710
```

Code: `src/saccade/perception/eval/d_online_stage2_q45_atlas.py`  
Tests: `tests/unit/test_d_online_stage2_q45_atlas.py`

---

## Authorized next steps (only)

```text
1. T0 Existing Atlas Region Interpretation Pack only
   (composition_grammar_safe_region thread / coverage audit)
2. Retain atlas; do not promote isolated productive-safe points to rules
3. Ranking / assignment-relative research is DEFERRED / BLOCKED until:
     - valid assignment-group key (fix frame provenance)
     - unknown/unresolved selected coverage policy closed
     - nested train-select holdout semantics understood
     and only after T0 closes or explicitly re-authorizes
4. Do NOT start production thr / hook policy from these points
5. Do NOT claim new-signal terminal A/B/C/D from parked exploratory path
6. Frame field fix / absolute MOT frame export is instrumentation follow-up
   (not a substitute for T0; not current sole next)
```

---

## One-line mainline

```text
Stage 2 Q4: frozen singleton tails inseparable (q4_separability_grade C)
Stage 2 Q4.5 v4: thr×AND/OR atlas → q45_atlas_terminal B isolated_safe_points_only
  (154 productive-safe; 0 coord-union interior; 0 region candidates;
   0 exact-absolute nested-LOSO portable; 21 selected unresolved)
next authorized: T0 region interpretation pack only
ranking / assignment: deferred until valid assignment-group key
production: blocked / preset unchanged
```

---

# Part D — Bounded conclusion of threshold formulation

```text
1. Singleton frozen absolute tails: no stable thr promotion
   (q4_separability_grade C → stage2_entry_terminal_after_q4 B).
2. Restricted thr × pairwise AND/OR atlas on locked cohort (evaluator v4):
   productive-safe cells exist (154) but all isolated/edge;
   0 coordinate-union interiors; 0 region candidates;
   0 exact-absolute nested-LOSO portable (q45_atlas_terminal B).
3. observed GT_hurt==0 ≠ safe rule ≠ portable reject policy.
4. Q4 weak AUC does NOT forbid thr/Boolean as analysis — atlas used it.
5. Frame column on D_online is host audit counter, not absolute MOT frame;
   assignment_group_key_status = invalid_frame_provenance
   → competition-relative columns untrusted.
6. Selected unresolved (21) further limits safety claims
   (unresolved_contaminated blocks region_candidate).
7. Authorized next: T0 Existing Atlas Region Interpretation Pack only.
   Ranking / assignment-relative is deferred / blocked until a valid
   assignment-group key — not thr-as-rule, and not concurrent with T0.
```

## Absorbed source files (git blob provenance)

| Original basename | blob SHA (at pre-delete / v4 tip) | Destination |
|:--|:--|:--|
| `m_b1_5_stage2_q1q3_d_online_audit_20260710.md` | `7b745472203610556774cf642bd8b86a453f21ea` | Part A |
| `m_b1_5_stage2_q4_separability_20260710.md` | `68c9126051dfcea4f767bb45fc027d96808b772a` | Part B |
| `m_b1_5_stage2_q45_threshold_atlas_20260710.md` | `e25574d9bc28955e2b94bfb0c1e053e4382b8935` | Part C–D (claims/evidence; v4 numbers supersede pre-v4 body tail) |

Recover: `git show <blob_sha>` or `git show 6df1739b:docs/modules/semantic/research/<basename>`.

**Not absorbed (standalone contract):**

- `m_b1_5_stage2_entry_contract_20260710.md` (G0–G4 claim firewall; code refs)

```text
research_claims: Q4.5 terminal unchanged = isolated_safe_points_only
  (v4 counts/gates supersede pre-v4 211/210 language)
q45_atlas_terminal: B
production_preset: unchanged
```
