# M-B1.5 Stage 2 Q4.5 — structured threshold-combination atlas

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 2 Q4.5 full restricted thr×AND/OR atlas. Descriptive only.
**Upstream:** [Q4](m_b1_5_stage2_q4_separability_20260710.md) · [Q1–Q3](m_b1_5_stage2_q1q3_d_online_audit_20260710.md)
**Entry contract:** [m_b1_5_stage2_entry_contract_20260710.md](m_b1_5_stage2_entry_contract_20260710.md)
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)

## Terminal classification (evaluator v3 — review #89)

```text
stage2_q45_terminal: isolated_safe_points_only
terminal_letter: B
taxonomy_version: stage2_q45_atlas_v3

n_primary_negative: 23
n_primary_positive_protect: 64
n_primary_resolved_selected: 87
n_selected_unresolved: 21   # tracked per-cell; not primary labels
n_selected_total: 108

atlas (frozen signals only; competition columns demoted):
  single_atom rows:     870
  pairwise AND rows:  17640
  pairwise OR rows:   17640

productive_safe (resolved GT_hurt==0 ∧ n_neg>0 ∧ no unresolved capture):
  single: 1
  AND:  153
  OR:     0

stability (per-grid quotient topology; all mask coordinates retained):
  isolated_safe_point:  7
  edge_candidate:      27
  has_interior:         0
  region_candidates:    0
  max_plateau_n_coords: 19

nested LOSO (train lattice → select → freeze → holdout):
  n_folds: 7
  n_clauses_ever_selected: 1352
  n_clauses_nested_loso_portable: 0   # non-tautological

evaluator gates:
  deletion_loo_is_portability: false
  fixed_full_sample_partition_not_portability: true
  nested_loso_required_for_region_A: true
  unresolved_contaminated_blocks_candidate: true
  semantic_duplicate_is_per_grid_not_global: true
  quotient_topology_retains_all_coordinates: true
  interior_requires_full_neighborhood_in_mask_coordinate_set: true
  assignment_group_key_status: invalid_frame_provenance

production_preset: unchanged
```

**Bounded finding (not global thr closure):**

> On the **resolved∧selected** cohort, the restricted atlas reports sample-zero-GT  
> cells but **no interior multi-sequence thick region** under per-grid quotient  
> topology (plateau coordinates retained; max plateau width 19 thr-cells still  
> has no full bilateral/4-neighborhood interior). Nested train-select → holdout  
> finds **0** portable clauses. Full selected population (incl. 21 unresolved)  
> still limits safety claims.  
> **Still inadmissible:** portable safe-region · thr global closure ·  
> hook-policy promotion · e2e effect · production preset change.

Evidence pack: [evidence/m_b1_5_stage2_q45_20260710/](evidence/m_b1_5_stage2_q45_20260710/)

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

## Review #89 evaluator fixes

### v2 (prior)

| Issue | Fix |
|:--|:--|
| LOO deletion = portability | Renamed `leave_one_sequence_deleted_*`; **cannot** promote |
| Unknown selected | per-cell `n_unresolved_selected`, pessimistic hurt, `unresolved_contaminated` |
| Boundary interior | bilateral (1D) / full-4 (pairwise); else `edge_candidate` |
| Assignment group | competition features demoted; frame not used as truth |
| MOT cross-check | prefers `cand_global_id` / `lost_global_id` + `_global_id_map.txt` |

### v3 (this round — review follow-up)

| Issue | Fix |
|:--|:--|
| Tautological “true holdout” | Replaced by **nested LOSO**: rebuild lattice on train fold, select train-safe clauses, freeze thr/Boolean, evaluate holdout. Old fixed-thr partition kept only as `fixed_full_sample_region_partition_check` (not portability). |
| Global `seen_sig` isolation | **Per-grid** semantic-dup flag; **quotient topology** keeps all coordinates of each unique mask within a feature/direction (or pairwise) grid. Plateau width from full coordinate set. |
| Stale evidence manifest | Clear `out_dir` → write all artifacts → write **current** manifest (HEAD + evaluator/runner/source SHAs) → **then** copy evidence pack. |

---

## Locked primary cohort

```text
negative:
  resolved ∧ baseline_selected ∧ pair_label == negative   → 23

positive protect:
  resolved ∧ baseline_selected ∧ pair_label == gt_consistent → 64

total = 87; both classes on 7/7 sequences
```

Non-selected rows are excluded from main atlas conclusions.

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

```text
May claim absolute MOT frame: NO
May claim “temporal information unavailable” from frame==4 alone: NO
Observation limitation only: YES
Affects Q4.5 threshold atlas mainline: NO
```

Artifact: `frame_column_provenance.json` in the study dir.

---

## Method (registered lattices — complete families)

### Single-atom

- Lattice: **all observed unique values** on primary (`primary_unique_boundaries`)
- Directions: `high_tail` (\(x \ge t\)), `low_tail` (\(x \le t\))
- Signals: five frozen (`score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean`)
- Secondary (optional): `sec_winner_runnerup_score_margin`,  
  `sec_delta_vs_ru_abs_log_h`, `sec_competitor_count`  
  (NaN when no competitor — **not** zero-filled)

### Pairwise (mainline frozen signals only)

- Lattice: **declared quantile lattice** \(q \in \{0, 0.05, \ldots, 1.0\}\)  
  (`primary_quantile_lattice_q05`)
- Combinators: **AND**, **OR** only (no 3+ atoms)
- Complete enumeration of registered cells; semantic-duplicate masks marked,  
  **not** dropped from the atlas file
- No label-adaptive grammar expansion

### Per cell metrics

support, coverage, neg captured, GT hurt, precision, enrichment,  
per-sequence counts, n_seq with neg, max sequence share, LOO holdouts,  
neighbor / subset relations (single-atom), necessity / observed-sufficiency  
descriptors.

### Stability classes (productive-safe only)

| class | meaning |
|:--|:--|
| `isolated_safe_point` | productive-safe; no productive-safe lattice neighbor |
| `thin_safe_edge` | partial neighborhood only |
| `locally_stable_region` | full local neighborhood; LOO not clean |
| `loo_stable_region` | neighborhood + LOO GT_hurt=0 + multi-seq non-dominant |
| `*_but_seq_thin` | geometric/LOO ok but single-seq or <2 seq neg mass |

Only `loo_stable_region` / `locally_stable_region` set `is_region_candidate=1`.

### Pareto

Full multi-objective frontier retained  
(`gt_hurt`, `n_neg_captured`, coverage, sequence support, max share, LOO).  
Not “best threshold only.”

---

## Headline facts (descriptive)

1. **Single-atom productive-safe:** 1 cell  
   (`resid_mean` extreme high-tail, **n_neg=1**, single sequence).  
   Matches Q4 pure-neg prefix anecdote — **not** multi-seq.

2. **Pairwise AND productive-safe:** 210 cells (many semantic-duplicate masks).  
   Max `n_neg_captured` among them = **4**.  
   Multi-seq neg support on 15 cells; all LOO-reported zero GT on remaining  
   sequences, but **neighborhood thickness** fails → thin/isolated, not region  
   candidates.

3. **Pairwise OR productive-safe:** **0** cells  
   (OR inflates support into GT mass).

4. **Pareto frontier:** 303 objective-unique points spanning the GT_hurt vs  
   neg-capture tradeoff (including non-safe enriching cells). Full table in  
   `pareto_frontier.csv`.

5. **No A-eligible region candidate** under declared multi-seq + neighborhood  
   + LOO gates.

---

## Claim firewall

```text
observed_safe_point  ≠  safe rule
productive_safe_point ≠  portable reject policy
enrichment without zero GT ≠  reject authorization
Q4 weak AUC ≠  Boolean impossible

Blocked:
  production thr / hook policy / e2e effect / preset change
  unrestricted Boolean mining / 3+ atoms
  naming sample-zero-GT as safe rule

Allowed:
  retain and query full atlas
  describe isolated observed safe points
  optional ranking / thickness follow-up
```

---

## Study artifacts

```text
out/signal_study/m_b1_5_stage2_q45_20260710/
  threshold_registry.json
  frame_column_provenance.json
  atom_atlas.{csv,parquet}
  pairwise_and_atlas.{csv,parquet}
  pairwise_or_atlas.{csv,parquet}
  pareto_frontier.csv
  region_stability.csv
  per_sequence.csv
  loo.csv
  cohort_summary.json
  reconciliation.json   # PASS
  summary.{json,md}
  manifest.json
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
1. Retain atlas; do not promote isolated productive-safe points to rules
2. Optional: deeper thickness / multi-seq gates on thin_safe_edge clusters
3. Optional: ranking / assignment policy research (if enrichment path pursued)
4. Do NOT start production thr / hook policy from these points
5. Do NOT claim new-signal terminal A/B/C/D from parked exploratory path
6. Frame field fix / absolute MOT frame export is instrumentation follow-up
   (orthogonal to atlas mainline)
```

---

## One-line mainline

```text
Stage 2 Q4: frozen singleton tails inseparable (weak AUC)
Stage 2 Q4.5: full restricted thr×AND/OR atlas → B isolated_safe_points_only
  (211 productive-safe cells; 0 region candidates)
next: retain atlas; no rule promotion; optional ranking / thickness work
production: blocked / preset unchanged
```

---

## Bounded conclusion of threshold formulation


```text
1. Singleton frozen absolute tails: no stable thr promotion (Q4-C).
2. Restricted thr × pairwise AND/OR atlas on locked cohort:
   productive-safe cells exist (211) but all isolated/thin;
   0 multi-seq thick LOO region candidates (Q4.5-B).
3. observed GT_hurt==0 ≠ safe rule ≠ portable reject policy.
4. Q4 weak AUC does NOT forbid thr/Boolean as analysis — atlas used it.
5. Frame column on D_online is host audit counter, not absolute MOT frame
   (observation limitation only; does not alter atlas terminal).
6. Authorized next: ranking / assignment-relative modeling
   (and optional atlas thickness diagnostics) — not thr-as-rule.
```


