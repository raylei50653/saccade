# LOO validation — ε=0 gate rule search

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Protocol:** strict leave-one-sequence-out  
**Tool:** `scripts/tools/gate_rule_search_loo.py`  
**Study:** [`out/signal_study/m_gate_rule_loo_20260709T125245Z/`](../../../../out/signal_study/m_gate_rule_loo_20260709T125245Z/)  
**Policy card (in-sample freeze):** [m_b1_policy_card_eps0_or5_20260709.md](m_b1_policy_card_eps0_or5_20260709.md)  
**Ledger:** `m.gate.rule_search.loo`


> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

```text
status: LOO_PARTIAL — not production
```

---

## 1. Protocol (what was tested)

For each held-out seq \(S\):

1. **Train** on the other 6 sequences only  
2. **Re-fit** atom quantiles + full atom→AND→greedy OR search on train  
3. **Apply** portable train thr to held-out \(S\)  
4. Measure test FP_removed / GT_hurt  

This is **not** “freeze 7-seq policy and score each seq” (that is still in-sample).  
This asks whether **re-searched** policies transfer.

```text
eps=0.0  max_and_size=3  max_or_rules=5  min_fp_support=80
```

---

## 2. Fold table (ε=0)

| heldout | train FP | train hurt | **test FP** | **test hurt** | test hurt% | classification | failed clause (if any) |
|:--|--:|--:|--:|--:|--:|:--|:--|
| MOT17-02-SDP | 7985 | 0 | 981 | **1** | 1.39% | fail / leak | `dist_h:zone_q50 ∧ score_m_bridge:zone_…` |
| MOT17-04-SDP | 8832 | 0 | 293 | **0** | 0 | **pass_eps0_transfer** | — |
| MOT17-05-SDP | 8289 | 0 | 1811 | **0** | 0 | **pass_eps0_transfer** | — |
| MOT17-09-SDP | 8958 | 0 | 46 | **0** | 0 | **pass_eps0_transfer** | — |
| MOT17-10-SDP | 6278 | 0 | 2149 | **2** | 1.27% | **gt_leak_with_capacity** | `gap:bin_61_150 ∧ score_m_bridge:zone_…` |
| MOT17-11-SDP | 8517 | 0 | 654 | **0** | 0 | **pass_eps0_transfer** | — |
| MOT17-13-SDP | 6742 | 0 | 3013 | **0** | 0 | **pass_eps0_transfer** | — |

---

## 3. Aggregate

| Metric | Value |
|:--|:--|
| folds with test GT_hurt=0 | **5 / 7** |
| sum test GT_hurt | **3** (1+2) |
| mean test FP_removed | **1278** |
| mean train FP_removed | ~7957 |
| **verdict** | **`loo_partial`** |

```text
In-sample:     GT_hurt = 0, FP = 9130  (looks production-safe)
LOO transfer:  5/7 clean; 02 and 10 leak 1–2 GT each
               still real FP capacity on held-out
```

---

## 4. Interpretation

### What holds

1. **Architecture is right** — train-side search still finds high-FP ε=0 policies (~6k–9k train FP).  
2. **Most sequences transfer** — 04/05/09/11/13 clean at test.  
3. **Capacity is real** — held-out mean FP~1.3k is not zero; not a vacuous empty rejector.

### What fails (why not production)

1. **ε=0 is not LOO-global** — two folds leak GT (total 3).  
2. **Leaks involve zone / gap condition clauses** re-fit on train (not the exact 7-seq frozen OR-5).  
   - 02: zone-q50 AND zone style  
   - 10: **gap_61_150 ∧ score zone** — long-gap condition overfits  
3. Confirms earlier distribution lesson: **fixed/in-sample safe thr can be sequence-lucky**.

### ε=0 vs ε=0.01

| | use |
|:--|:--|
| ε=0 in-sample OR-5 | freeze as **candidate card** only |
| LOO ε=0 | **gate** for promotion: currently partial |
| ε=0.01 | research frontier (do not mix into production talk) |

---

## 5. Next (ordered)

```text
1. ✅ Freeze policy card (done)
2. ✅ LOO strict (done — partial)
3. ✅ Atom repair re-LOO → ban_gap+ban_zone **loo_pass_eps0**  
     see [m_b1_loo_hurt_atom_repair_20260709.md](m_b1_loo_hurt_atom_repair_20260709.md)
3.  → Clause ablation on LOO: drop gap-bin atoms; zone-only vs support-only
4.  → Optional: freeze 7-seq thr, evaluate per-seq (weaker than LOO; diagnostic)
5.  → Only if LOO clean (or after repair): B2 / e2e smoke
6.  → Production preset: still NO
```

Repair hypothesis (not yet run):

```text
forbid gap:* atoms in mining
raise zone quantile (only q70+, no q50)
or require support atom in every AND clause
```

---

## 6. Reproduce

```bash
uv run python scripts/tools/gate_rule_search_loo.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_gate_rule_loo_<stamp> \
  --eps 0.0 --max-and-size 3 --max-or-rules 5 --min-fp-support 80
```

---

## 7. One-line status

> **In-sample ε=0 OR-5 is a strong candidate; LOO is partial (5/7 clean, 3 GT leaks on 02/10). Not production. Next: repair atoms/clauses, not preset.**
