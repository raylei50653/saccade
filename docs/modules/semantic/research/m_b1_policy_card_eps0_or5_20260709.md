# Candidate policy card — ε=0 OR-5 (in-sample freeze)

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->


> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

```text
status: IN_SAMPLE_CANDIDATE_ONLY  — SUPERSEDED for LOO claims
production: UNCHANGED / default-off
superseded_by: m_b1_repaired_eps0_loo_pass_20260709
  (ban_gap+ban_zone; LOO 7/7 pass — see repaired candidate card)
blocked_before_production: repaired 2D · B2/e2e · preset review
```

---

## Identity

| Field | Value |
|:--|:--|
| **policy_id** | `m_b1_eps0_or5_20260709T124534Z` |
| **study_path** | `out/signal_study/m_gate_rule_search_20260709T124534Z/` |
| **eps0 detail** | `eps_0p0/summary.json` (portable thr) |
| **epsilon** | **0.0** (conservative; separate from ε=0.01) |
| **search tool** | `scripts/tools/gate_rule_search.py` |
| **input pairs** | `out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv` |
| **input rows** | 24284 data + header (24285 lines) |
| **input sha256** | `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| **n_pos / n_neg** | 340 / 21449 |
| **preset / substrate** | `mamba_whole_graph_m` B1 (bridge/interp off, 7-seq SDP) |

---

## Search parameters (reproducible)

```text
eps                 = 0.0
max_and_size        = 3
max_or_rules        = 5
min_fp_support      = 100
tau_seq_std         = 0.05
tail quantiles      = 0.85, 0.90, 0.95, 0.99
zone quantiles      = 0.50, 0.70
roles               = condition | support | diagnostic (see ROLE_MAP)
```

---

## Policy (OR of 5 clauses)

```text
reject if
  (dist_h:zone_q70 AND score_m_bridge:zone_q70)
  OR abs_log_h:tail_q85
  OR resid_mean:tail_q85
  OR abs_ratio_m1:tail_q85
  OR dist_h:tail_q85
```

### Atom definitions (fitted thr on full 7-seq in-sample)

| atom_id | role | definition (as-of fit) |
|:--|:--|:--|
| `dist_h:zone_q70` | condition | dist_h > **4.510** (q0.70) |
| `score_m_bridge:zone_q70` | condition | score_m_bridge > **7.401** (q0.70) |
| `abs_log_h:tail_q85` | support | abs_log_h > **1.349** (q0.85) |
| `resid_mean:tail_q85` | condition | resid_mean > **14.04** (q0.85) |
| `abs_ratio_m1:tail_q85` | support | \|h_ratio−1\| > **2.086** (q0.85) |
| `dist_h:tail_q85` | condition | dist_h > **6.732** (q0.85) |

Portable JSON: `eps_0p0/summary.json` → `policy.portable`.

---

## In-sample headline metrics

| Metric | Value |
|:--|:--|
| FP_removed | **9130** (42.6% of FP) |
| GT_hurt | **0** |
| GT_hurt_rate | 0 |
| seq_hurt_std | 0 |
| vs best atom | +5861 FP |
| vs best AND clause | +4956 FP |

### Per-seq (in-sample apply of frozen policy)

| seq | n_pos | FP_removed | GT_hurt |
|:--|--:|--:|--:|
| MOT17-02-SDP | 72 | 1043 | 0 |
| MOT17-04-SDP | 12 | 285 | 0 |
| MOT17-05-SDP | 42 | 1738 | 0 |
| MOT17-09-SDP | 14 | 46 | 0 |
| MOT17-10-SDP | 157 | 2396 | 0 |
| MOT17-11-SDP | 20 | 638 | 0 |
| MOT17-13-SDP | 23 | 2984 | 0 |

---

## How to read this card

```text
✅  What it is:
    in-sample ε=0 rule-search candidate; architecture validated
    (atoms → AND → greedy OR under hard GT constraint)

❌  What it is NOT:
    production policy
    LOO-proven gate (see LOO card — partial transfer)
    permission to change preset / px thr
```

ε=0.01 results live under `eps_0p01/` — **research frontier only**, not this card.

---

## Related

- Architecture: [m_b1_gate_rule_search_architecture_20260709.md](m_b1_gate_rule_search_architecture_20260709.md)  
- LOO: [m_b1_gate_rule_search_loo_20260709.md](m_b1_gate_rule_search_loo_20260709.md)  
- Ledger: [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
